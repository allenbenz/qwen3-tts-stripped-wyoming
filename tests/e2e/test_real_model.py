"""E2E tests against a real model directory (marker: e2e).

Enable with QWEN3TTS_E2E=1. Point QWEN3TTS_MODEL at a model directory in any
variant (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice, downloaded into
QWEN3TTS_MODEL_DIR). When QWEN3TTS_VARIANT selects a converted variant, the
server converts the source once on first run (several minutes, CPU) and
reuses the cache afterwards.
"""

from __future__ import annotations

import asyncio
import os
import socket

import numpy as np
import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncTcpServer
from wyoming.tts import Synthesize, SynthesizeVoice

from qwen3_tts_stripped_wyoming.asr import TranscriptionService
from qwen3_tts_stripped_wyoming.config import settings_from_env
from qwen3_tts_stripped_wyoming.runtime import SynthesisService
from qwen3_tts_stripped_wyoming.server import build_info, make_handler_factory

pytestmark = pytest.mark.e2e


def _settings():
    env = {k: v for k, v in os.environ.items() if k.startswith("QWEN3TTS_")}
    env.setdefault("QWEN3TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    env.setdefault("QWEN3TTS_DEVICE", "cpu")
    env.setdefault("QWEN3TTS_WARMUP", "false")  # keep the test fast
    env.setdefault("QWEN3TTS_VARIANT", "q8")
    return settings_from_env(env)


@pytest.fixture(scope="module")
def settings() -> object:
    return _settings()


@pytest.fixture(scope="module")
async def service(settings):
    svc = await SynthesisService.create(settings)
    return svc


@pytest.fixture(scope="module")
async def asr_service(settings):
    if not settings.asr_model:
        pytest.skip("asr disabled")
    return await TranscriptionService.create(settings)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
async def client(service, settings):
    info = build_info(service, None)
    port = _free_port()
    server = AsyncTcpServer("127.0.0.1", port)
    await server.start(make_handler_factory(service, info, settings, asyncio.Lock()))
    client = AsyncClient.from_uri(
        f"tcp://127.0.0.1:{port}", connect_timeout=10.0, read_timeout=300.0
    )
    await client.connect()
    try:
        yield client, service
    finally:
        await client.disconnect()
        await server.stop()


async def _read_until(client: AsyncClient, *stop_types: str) -> list[Event]:
    events: list[Event] = []
    while True:
        event = await client.read_event()
        assert event is not None
        events.append(event)
        if event.type in stop_types:
            return events


async def test_ping(client) -> None:
    c, _ = client
    await c.write_event(Ping(text="e2e").event())
    event = await c.read_event()
    assert event is not None and Pong.is_type(event.type)


async def test_describe(client) -> None:
    c, svc = client
    await c.write_event(Describe().event())
    event = await c.read_event()
    assert event is not None and Info.is_type(event.type)
    program = Info.from_event(event).tts[0]
    assert program.voices
    assert svc.variant in program.description


async def test_synthesize_speech(client) -> None:
    c, svc = client
    await c.write_event(
        Synthesize(
            text="Home Assistant is talking to Qwen three T T S.",
            voice=SynthesizeVoice(name=svc.speakers()[0].id),
        ).event()
    )
    events = await _read_until(c, "audio-stop")
    start = AudioStart.from_event(next(e for e in events if AudioStart.is_type(e.type)))
    assert (start.rate, start.width, start.channels) == (24000, 2, 1)
    assert any(AudioStop.is_type(e.type) for e in events)
    audio = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    samples = np.frombuffer(audio, dtype="<i2")
    assert len(samples) >= 24000  # at least a second of audio
    assert np.abs(samples).max() > 1000  # not silence


@pytest.fixture
async def combined_client(service, asr_service, settings):
    """Server with both TTS and STT wired, like production startup."""
    info = build_info(service, asr_service)
    port = _free_port()
    server = AsyncTcpServer("127.0.0.1", port)
    lock = asyncio.Lock()
    await server.start(make_handler_factory(service, info, settings, lock, asr_service))
    client = AsyncClient.from_uri(
        f"tcp://127.0.0.1:{port}", connect_timeout=10.0, read_timeout=600.0
    )
    await client.connect()
    try:
        yield client, service, asr_service
    finally:
        await client.disconnect()
        await server.stop()


async def test_stt_advertised(combined_client) -> None:
    c, _, _ = combined_client
    await c.write_event(Describe().event())
    event = await c.read_event()
    assert event is not None and Info.is_type(event.type)
    info = Info.from_event(event)
    assert info.asr and info.tts
    assert "en" in info.asr[0].models[0].languages


async def test_transcribe_tts_output_roundtrip(combined_client) -> None:
    """Closed loop: synthesize speech, then transcribe it back."""
    c, svc, _ = combined_client
    text = "The living room lamp is on."
    await c.write_event(
        Synthesize(text=text, voice=SynthesizeVoice(name=svc.speakers()[0].id)).event()
    )
    events = await _read_until(c, "audio-stop")
    audio = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    samples = np.frombuffer(audio, dtype="<i2")
    assert len(samples) >= 24000

    # feed the synthesized 24 kHz audio to the ASR side (it resamples)
    await c.write_event(Transcribe(language="en").event())
    await c.write_event(AudioStart(rate=24000, width=2, channels=1).event())
    for offset in range(0, len(audio), 16384):
        await c.write_event(
            AudioChunk(
                rate=24000, width=2, channels=1, audio=audio[offset : offset + 16384]
            ).event()
        )
    await c.write_event(AudioStop().event())
    events = await _read_until(c, "transcript", "error")
    assert not any(e.type == "error" for e in events), events
    transcript = Transcript.from_event(events[-1])
    lowered = transcript.text.lower()
    # "the" pins the first-word regression: the old parser split the model
    # output on the first space and swallowed the first transcription word
    for word in ("the", "living", "room", "lamp"):
        assert word in lowered, transcript.text


async def test_transcribe_non_multiple_mel_length(combined_client) -> None:
    """Regression: 3.08 s of 24 kHz audio resamples to 49280 samples = 308
    mel frames, which the native encoder rejects unless the feature extractor
    pads to a multiple of 2*n_window (deployed failure: "padded_feature_length
    ... multiple of `n_window * 2` (100), but got 308")."""
    c, svc, _ = combined_client
    await c.write_event(
        Synthesize(
            text="The living room lamp is on.",
            voice=SynthesizeVoice(name=svc.speakers()[0].id),
        ).event()
    )
    events = await _read_until(c, "audio-stop")
    audio = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    # exactly 3.08 s at 24 kHz -> 308 mel frames at 16 kHz; pad with silence
    # if the synthesis came out shorter (the frame count is what matters)
    target = 73920 * 2
    trimmed = audio[:target]
    if len(trimmed) < target:
        trimmed += b"\x00" * (target - len(trimmed))
    assert len(trimmed) == target

    await c.write_event(Transcribe(language="en").event())
    await c.write_event(AudioStart(rate=24000, width=2, channels=1).event())
    for offset in range(0, len(trimmed), 16384):
        chunk = trimmed[offset : offset + 16384]
        await c.write_event(AudioChunk(rate=24000, width=2, channels=1, audio=chunk).event())
    await c.write_event(AudioStop().event())
    events = await _read_until(c, "transcript", "error")
    assert not any(e.type == "error" for e in events), events
    assert Transcript.from_event(events[-1]).text is not None
