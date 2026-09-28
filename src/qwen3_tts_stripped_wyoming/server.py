"""Server bootstrap: build the wyoming Info message and run the TCP server."""

from __future__ import annotations

import asyncio
import logging
from importlib.metadata import PackageNotFoundError, version

from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .asr import TranscriptionService
from .config import Settings
from .handler import Qwen3TtsEventHandler
from .runtime import SynthesisService

_LOGGER = logging.getLogger(__name__)

PROGRAM_NAME = "qwen3-tts-stripped-wyoming"
PROGRAM_URL = "https://github.com/allenbenz/qwen3-tts-stripped-wyoming"


def program_version() -> str:
    try:
        return version("qwen3-tts-stripped-wyoming")
    except PackageNotFoundError:
        return __version__


def build_info(service: SynthesisService, asr_service: TranscriptionService | None) -> Info:
    """Describe the service for Home Assistant's Wyoming integration."""
    device = "CUDA" if service.using_cuda else "CPU"
    attribution = Attribution(name="Qwen3-TTS", url="https://huggingface.co/Qwen")
    voices = [
        TtsVoice(
            name=speaker.id,
            attribution=attribution,
            installed=True,
            description=str(speaker.id).replace("_", " ").title(),
            version=None,
            languages=service.speaker_languages_bcp47(speaker),
        )
        for speaker in service.speakers()
    ]
    program = TtsProgram(
        name=PROGRAM_NAME,
        attribution=Attribution(name=PROGRAM_NAME, url=PROGRAM_URL),
        installed=True,
        description=(f"Qwen3-TTS {service.variant} text to speech (torch, {device})"),
        version=program_version(),
        voices=voices,
        supports_synthesize_streaming=True,
    )
    info = Info(tts=[program])
    if asr_service is not None:
        asr_program = AsrProgram(
            name=PROGRAM_NAME,
            attribution=Attribution(name=PROGRAM_NAME, url=PROGRAM_URL),
            installed=True,
            description=(
                f"Qwen3-ASR speech to text (torch, {device}); "
                f"{len(asr_service.languages)} languages"
            ),
            version=program_version(),
            models=[
                AsrModel(
                    name=PROGRAM_NAME,
                    attribution=Attribution(name="Qwen3-ASR", url="https://huggingface.co/Qwen"),
                    installed=True,
                    description="Qwen3-ASR speech recognition",
                    version=None,
                    languages=asr_service.languages_bcp47(),
                )
            ],
        )
        info = Info(tts=[program], asr=[asr_program])
    return info


def make_handler_factory(
    service: SynthesisService,
    info: Info,
    settings: Settings,
    model_lock: asyncio.Lock,
    asr_service: TranscriptionService | None = None,
):
    """HandlerFactory closure wiring shared state into every connection.

    One lock serializes all model work (synthesis and transcription): the
    models share the GPU, and a voice pipeline is half-duplex anyway.
    """

    def factory(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> Qwen3TtsEventHandler:
        return Qwen3TtsEventHandler(
            reader,
            writer,
            service=service,
            info=info,
            settings=settings,
            synth_lock=model_lock,
            program_name=PROGRAM_NAME,
            asr_service=asr_service,
            asr_lock=model_lock,
        )

    return factory


async def run_server(settings: Settings) -> None:
    """Load the models, then serve Wyoming events until shutdown."""
    service = await SynthesisService.create(settings)
    _LOGGER.info(
        "voices: %s (variant %s)",
        ", ".join(s.id for s in service.speakers()),
        service.variant,
    )

    asr_service: TranscriptionService | None = None
    if settings.asr_model:
        try:
            asr_service = await TranscriptionService.create(settings)
            _LOGGER.info(
                "asr: %s (languages: %s)",
                settings.asr_model,
                ", ".join(asr_service.languages_bcp47()[:8])
                + ("..." if len(asr_service.languages_bcp47()) > 8 else ""),
            )
        except Exception:
            # STT is additive: refuse to take the TTS side down with it
            _LOGGER.exception("failed to load the ASR model; continuing without STT")
            asr_service = None
    else:
        _LOGGER.info("asr: disabled (asr-model is empty)")

    info = build_info(service, asr_service)
    server = AsyncServer.from_uri(settings.uri)
    _LOGGER.info("listening on %s", settings.uri)
    await server.run(make_handler_factory(service, info, settings, asyncio.Lock(), asr_service))
