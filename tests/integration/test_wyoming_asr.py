"""Integration tests for the Wyoming STT (transcribe) protocol path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Describe, Info

from qwen3_tts_stripped_wyoming.config import Settings

from ..fakes import FakeAsrModel, FakeQwenModel
from ..utils import connect, make_fake_asr_service, make_fake_service, start_test_server


def _pcm_bytes(seconds: float = 1.0, rate: int = 16000, channels: int = 1) -> bytes:
    samples = int(seconds * rate)
    t = np.arange(samples, dtype=np.float32) / np.float32(rate)
    wave = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    if channels > 1:
        wave = np.stack([wave] * channels, axis=1).reshape(-1)
    return (np.clip(wave, -1.0, 1.0) * 32767).astype("<i2").tobytes()


async def _start(with_asr: bool = True):
    model = FakeQwenModel()
    asr_model = FakeAsrModel(text="the living room lamp is on")
    settings = Settings(model_dir=Path("."), output_chunk_ms=50)
    service = make_fake_service(model, settings)
    asr_service = make_fake_asr_service(asr_model, settings) if with_asr else None
    server, port = await start_test_server(service, settings, asr_service)
    client = await connect(port)
    return client, asr_model, server


async def _transcribe(
    client,
    *,
    language: str | None = None,
    audio: bytes | None = None,
    rate: int = 16000,
    channels: int = 1,
    names=None,
    terms=None,
):
    await client.write_event(
        Transcribe(language=language, transcript_names=names, transcript_terms=terms).event()
    )
    await client.write_event(AudioStart(rate=rate, width=2, channels=channels).event())
    for offset in range(0, len(audio or b""), 4096):
        await client.write_event(
            AudioChunk(
                rate=rate, width=2, channels=channels, audio=(audio or b"")[offset : offset + 4096]
            ).event()
        )
    await client.write_event(AudioStop().event())


async def test_transcribe_roundtrip() -> None:
    client, asr_model, server = await _start()
    try:
        await _transcribe(client, audio=_pcm_bytes())
        event = await client.read_event()
        assert event is not None and Transcript.is_type(event.type)
        assert Transcript.from_event(event).text == "the living room lamp is on"
        request = asr_model.requests[-1]
        assert request["samples"] == 16000
        assert request["sr"] == 16000
    finally:
        await client.disconnect()
        await server.stop()


async def test_transcribe_language_forced() -> None:
    client, asr_model, server = await _start()
    try:
        await _transcribe(client, language="en-US", audio=_pcm_bytes())
        event = await client.read_event()
        assert event is not None and Transcript.is_type(event.type)
        assert asr_model.requests[-1]["language"] == "English"
    finally:
        await client.disconnect()
        await server.stop()


async def test_transcribe_unsupported_language_error() -> None:
    client, _, server = await _start()
    try:
        await _transcribe(client, language="xx-YY", audio=_pcm_bytes())
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        assert Error.from_event(event).code == "invalid-language"
    finally:
        await client.disconnect()
        await server.stop()


async def test_transcribe_empty_audio_error() -> None:
    client, _, server = await _start()
    try:
        await _transcribe(client, audio=b"")
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        assert Error.from_event(event).code == "empty-audio"
    finally:
        await client.disconnect()
        await server.stop()


async def test_transcribe_forwards_bias_terms_as_context() -> None:
    client, asr_model, server = await _start()
    try:
        await _transcribe(
            client,
            audio=_pcm_bytes(),
            names=["Aiden"],
            terms=["living room lamp", "kitchen light"],
        )
        event = await client.read_event()
        assert event is not None and Transcript.is_type(event.type)
        assert asr_model.requests[-1]["context"] == "Aiden, living room lamp, kitchen light"
    finally:
        await client.disconnect()
        await server.stop()


async def test_transcribe_disabled_server() -> None:
    client, _, server = await _start(with_asr=False)
    try:
        await _transcribe(client, audio=_pcm_bytes())
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        assert Error.from_event(event).code == "asr-disabled"
    finally:
        await client.disconnect()
        await server.stop()


async def test_describe_lists_asr_program() -> None:
    client, _, server = await _start()
    try:
        await client.write_event(Describe().event())
        event = await client.read_event()
        assert event is not None and Info.is_type(event.type)
        info = Info.from_event(event)
        assert len(info.asr) == 1
        program = info.asr[0]
        assert program.models
        assert "en" in program.models[0].languages
        assert "zh" in program.models[0].languages
    finally:
        await client.disconnect()
        await server.stop()


async def test_describe_without_asr() -> None:
    client, _, server = await _start(with_asr=False)
    try:
        await client.write_event(Describe().event())
        event = await client.read_event()
        assert event is not None and Info.is_type(event.type)
        info = Info.from_event(event)
        assert not info.asr
        assert info.tts  # tts still advertised
    finally:
        await client.disconnect()
        await server.stop()


async def test_transcribe_failure_sends_error() -> None:
    client, asr_model, server = await _start()
    asr_model.fail = RuntimeError("boom")
    try:
        await _transcribe(client, audio=_pcm_bytes())
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        assert Error.from_event(event).code == "transcription-failed"
    finally:
        await client.disconnect()
        await server.stop()


async def test_audio_events_without_transcribe_ignored() -> None:
    """Stray audio events (no transcribe before them) must not crash."""
    from wyoming.ping import Ping, Pong

    client, _, server = await _start()
    try:
        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"x").event())
        await client.write_event(AudioStop().event())
        await client.write_event(Ping(text="alive").event())
        event = await client.read_event()
        assert event is not None and Pong.is_type(event.type)
    finally:
        await client.disconnect()
        await server.stop()
