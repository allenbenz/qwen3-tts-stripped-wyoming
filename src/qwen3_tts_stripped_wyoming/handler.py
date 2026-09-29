"""Wyoming event handler implementing the TTS service."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info, SelectProgram
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
    SynthesizeVoice,
)

from .asr import LanguageResolutionError, TranscriptionService
from .audio import (
    CHANNELS,
    SAMPLE_WIDTH,
    asr_language_to_bcp47,
    chunk_bytes_for_ms,
    float_to_int16_bytes,
    pcm_bytes_to_float,
    split_bytes,
)
from .config import Settings
from .runtime import SynthesisService, VoiceResolutionError

_LOGGER = logging.getLogger(__name__)


@dataclass
class _StreamState:
    """Buffered text for an in-progress streaming synthesize request."""

    voice: SynthesizeVoice | None = None
    text_format: Any = None
    parts: list[str] = field(default_factory=list)
    saw_synthesize = False

    @property
    def text(self) -> str:
        return "".join(self.parts)


@dataclass
class _AsrStreamState:
    """Buffered PCM for an in-progress transcribe request.

    The Wyoming STT flow is: transcribe -> audio-start -> audio-chunk+ ->
    audio-stop, then the server replies with a single transcript event.
    """

    language: str | None = None
    context: str | None = None
    rate: int = 16_000
    width: int = SAMPLE_WIDTH
    channels: int = CHANNELS
    parts: list[bytes] = field(default_factory=list)

    def extend(self, audio: bytes) -> None:
        if audio:
            self.parts.append(audio)

    @property
    def audio_bytes(self) -> bytes:
        return b"".join(self.parts)


class Qwen3TtsEventHandler(AsyncEventHandler):
    """One instance per client connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        service: SynthesisService,
        info: Info,
        settings: Settings,
        synth_lock: asyncio.Lock,
        program_name: str,
        asr_service: TranscriptionService | None = None,
        asr_lock: asyncio.Lock | None = None,
    ) -> None:
        super().__init__(reader, writer)
        self._service = service
        self._asr = asr_service
        self._asr_lock = asr_lock
        self._info = info
        self._settings = settings
        self._synth_lock = synth_lock
        self._program_name = program_name
        self._stream: _StreamState | None = None
        self._asr_stream: _AsrStreamState | None = None
        peer = writer.get_extra_info("peername")
        _LOGGER.debug("client connected: %s", peer)

    async def handle_event(self, event: Event) -> bool:
        try:
            return await self._dispatch(event)
        except (VoiceResolutionError, LanguageResolutionError) as exc:
            await self._write_error(str(exc), code="invalid-request")
        except Exception:  # keep malformed events from killing the connection
            _LOGGER.exception("unexpected error handling %s event", event.type)
            await self._write_error("internal server error; see server logs", code="internal-error")
        return True

    async def _dispatch(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
        elif Ping.is_type(event.type):
            ping = Ping.from_event(event)
            await self.write_event(Pong(text=ping.text).event())
        elif SelectProgram.is_type(event.type):
            selected = SelectProgram.from_event(event)
            if selected.name != self._program_name:
                _LOGGER.debug("ignoring select-program for %r", selected.name)
        elif Synthesize.is_type(event.type):
            request = Synthesize.from_event(event)
            if self._stream is not None:
                # Backwards-compat echo inside a streaming request: merge anything
                # the earlier events did not carry, but do not synthesize twice.
                self._stream.saw_synthesize = True
                if self._stream.voice is None and request.voice is not None:
                    self._stream.voice = request.voice
                if not self._stream.text and request.text:
                    self._stream.parts = [request.text]
            else:
                await self._handle_synthesis(request.text, request.voice, request.text_format)
        elif SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._stream = _StreamState(voice=start.voice, text_format=start.text_format)
        elif SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            if self._stream is None:
                _LOGGER.debug("synthesize-chunk without synthesize-start; ignoring")
            else:
                self._stream.parts.append(chunk.text)
        elif SynthesizeStop.is_type(event.type):
            state, self._stream = self._stream, None
            if state is None:
                _LOGGER.debug("synthesize-stop without synthesize-start; ignoring")
                return True
            await self._handle_synthesis(state.text, state.voice, state.text_format)
            await self.write_event(SynthesizeStopped().event())
        elif Transcribe.is_type(event.type):
            asr_request = Transcribe.from_event(event)
            if self._asr is None:
                await self._write_error(
                    "speech-to-text is disabled on this server", code="asr-disabled"
                )
            else:
                # bias the transcription with names/terms from the request;
                # wyoming's `context` dict (prior interaction state) is ignored
                terms: list[str] = list(asr_request.transcript_names or [])
                terms += list(asr_request.transcript_terms or [])
                context = ", ".join(terms) or None
                self._asr_stream = _AsrStreamState(language=asr_request.language, context=context)
        elif AudioStart.is_type(event.type):
            if self._asr_stream is not None:
                audio_start = AudioStart.from_event(event)
                self._asr_stream.rate = int(audio_start.rate or 16_000)
                self._asr_stream.width = int(audio_start.width or SAMPLE_WIDTH)
                self._asr_stream.channels = int(audio_start.channels or CHANNELS)
        elif AudioChunk.is_type(event.type):
            if self._asr_stream is not None:
                self._asr_stream.extend(AudioChunk.from_event(event).audio)
        elif AudioStop.is_type(event.type):
            asr_state, self._asr_stream = self._asr_stream, None
            if asr_state is None:
                _LOGGER.debug("audio-stop without transcribe; ignoring")
                return True
            await self._handle_transcription(asr_state)
        else:
            _LOGGER.debug("ignoring event type %r", event.type)
        return True

    # ------------------------------------------------------------------
    async def _handle_synthesis(
        self,
        text: str | None,
        voice: SynthesizeVoice | None,
        text_format: Any,
    ) -> None:
        text = (text or "").strip()
        if text_format not in (None, "", SynthesizeTextFormat.TEXT, "text"):
            await self._write_error(
                f"unsupported text_format {text_format!r}; only plain text is supported",
                code="text-format-not-supported",
            )
            return
        if not text:
            await self._write_error("no text to synthesize", code="empty-text")
            return

        try:
            speaker = self._service.resolve_speaker_for_request(
                voice.name if voice is not None else None,
                voice.language if voice is not None else None,
            )
        except VoiceResolutionError as exc:
            await self._write_error(str(exc), code="invalid-voice")
            return
        try:
            language = self._service.resolve_language(voice.language if voice is not None else None)
        except VoiceResolutionError as exc:
            await self._write_error(str(exc), code="invalid-language")
            return

        started = time.monotonic()
        async with self._synth_lock:
            await self.write_event(
                AudioStart(
                    rate=self._service.sample_rate,
                    width=SAMPLE_WIDTH,
                    channels=CHANNELS,
                    timestamp=0,
                ).event()
            )
            chunk_bytes = chunk_bytes_for_ms(
                self._service.sample_rate, self._settings.output_chunk_ms
            )
            gain = self._settings.energy_gain
            try:
                # fast backend: audio pieces stream out during generation;
                # stock backend: one piece after full synthesis
                async for piece in self._service.stream(
                    text=text, speaker_id=speaker.id, language=language
                ):
                    data = float_to_int16_bytes(piece, gain=gain)
                    for part in split_bytes(data, chunk_bytes):
                        await self.write_event(
                            AudioChunk(
                                rate=self._service.sample_rate,
                                width=SAMPLE_WIDTH,
                                channels=CHANNELS,
                                audio=part,
                                timestamp=_elapsed_ms(started),
                            ).event()
                        )
                await self.write_event(AudioStop(timestamp=_elapsed_ms(started)).event())
            except VoiceResolutionError as exc:
                await self._write_error(str(exc), code="invalid-request")
            except Exception:
                _LOGGER.exception("synthesis failed")
                await self._write_error(
                    "synthesis failed; see server logs", code="synthesis-failed"
                )

    # ------------------------------------------------------------------
    async def _handle_transcription(self, state: _AsrStreamState) -> None:
        assert self._asr is not None
        raw = state.audio_bytes
        if not raw:
            await self._write_error("no audio to transcribe", code="empty-audio")
            return
        if state.width != SAMPLE_WIDTH:
            await self._write_error(
                f"unsupported audio width {state.width} (16-bit PCM required)",
                code="audio-format-not-supported",
            )
            return
        try:
            language = self._asr.resolve_language(state.language)
        except LanguageResolutionError as exc:
            await self._write_error(str(exc), code="invalid-language")
            return

        audio = pcm_bytes_to_float(raw, channels=state.channels)
        lock = self._asr_lock or asyncio.Lock()
        async with lock:
            try:
                text, detected = await self._asr.transcribe(
                    audio,
                    sample_rate=state.rate,
                    language=language,
                    context=state.context,
                )
            except LanguageResolutionError as exc:
                await self._write_error(str(exc), code="invalid-language")
                return
            except Exception:
                _LOGGER.exception("transcription failed")
                await self._write_error(
                    "transcription failed; see server logs", code="transcription-failed"
                )
                return
        detected_bcp47 = asr_language_to_bcp47(detected) if detected else None
        await self.write_event(Transcript(text=text, language=detected_bcp47).event())

    async def _write_error(self, text: str, *, code: str) -> None:
        _LOGGER.warning("sending error to client: [%s] %s", code, text)
        await self.write_event(Error(text=text, code=code).event())


def _elapsed_ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)
