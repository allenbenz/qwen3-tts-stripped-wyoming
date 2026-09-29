"""Integration tests: a real Wyoming TCP server backed by the fake model.

These exercise the full protocol path -- describe/ping, synthesize (buffered
and streaming), error handling, and audio byte-exactness -- without the model.
"""

from __future__ import annotations

import asyncio

import numpy as np
from wyoming.audio import AudioChunk, AudioStart
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info, SelectProgram
from wyoming.ping import Ping, Pong
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from qwen3_tts_stripped_wyoming.config import Settings

from ..fakes import FakeQwenModel, fake_audio_piece
from ..utils import connect, make_fake_service, read_until, start_test_server


async def _start(fake_model: FakeQwenModel, settings: Settings | None = None):
    service = make_fake_service(fake_model, settings)
    server, port = await start_test_server(service, settings or Settings())
    client = await connect(port)
    return client, fake_model, server


def expected_pcm(gain: float = 1.0, pieces: int = 1) -> bytes:
    wav = np.concatenate([fake_audio_piece()] * pieces)
    scaled = wav * np.float32(gain) if gain != 1.0 else wav
    clipped = np.clip(scaled, np.float32(-1.0), np.float32(1.0))
    return (clipped * np.float32(32767.0)).astype("<i2").tobytes()


def collect_audio(events: list) -> bytes:
    audio = bytearray()
    for event in events:
        if AudioChunk.is_type(event.type):
            audio.extend(AudioChunk.from_event(event).audio)
    return bytes(audio)


async def test_ping_pong() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        await client.write_event(Ping(text="hello").event())
        event = await client.read_event()
        assert event is not None and Pong.is_type(event.type)
        assert Pong.from_event(event).text == "hello"
    finally:
        await client.disconnect()
        await server.stop()


async def test_describe_lists_voices() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        await client.write_event(Describe().event())
        event = await client.read_event()
        assert event is not None and Info.is_type(event.type)
        info = Info.from_event(event)
        assert len(info.tts) == 1
        program = info.tts[0]
        assert program.supports_synthesize_streaming is True
        by_name = {voice.name: voice for voice in program.voices}
        assert set(by_name) == {"narrator", "aiden"}
        assert "en" in by_name["narrator"].languages
        assert "zh" in by_name["narrator"].languages
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesize_returns_exact_pcm() -> None:
    model = FakeQwenModel()
    client, _, server = await _start(model, Settings(output_chunk_ms=50))
    try:
        await client.write_event(
            Synthesize(text="Hello", voice=SynthesizeVoice(name="aiden")).event()
        )
        events = await read_until(client, "audio-stop")
        start_event = next(e for e in events if AudioStart.is_type(e.type))
        start = AudioStart.from_event(start_event)
        assert (start.rate, start.width, start.channels) == (24000, 2, 1)
        assert collect_audio(events) == expected_pcm()
        assert model.requests[-1]["speaker"] == "aiden"
        assert model.requests[-1]["language"] is None  # no language given -> Auto
    finally:
        await client.disconnect()
        await server.stop()


async def test_long_text_chunks_output() -> None:
    client, _, server = await _start(FakeQwenModel(), Settings(output_chunk_ms=50))
    try:
        text = "This sentence is deliberately longer than forty characters."
        await client.write_event(Synthesize(text=text).event())
        events = await read_until(client, "audio-stop")
        chunks = [e for e in events if AudioChunk.is_type(e.type)]
        # fake model returns two 100 ms pieces; output_chunk_ms=50 -> 4 chunks
        assert len(chunks) == 4
        assert collect_audio(events) == expected_pcm(pieces=2)
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesize_default_voice_and_auto_language() -> None:
    model = FakeQwenModel()
    client, _, server = await _start(model)
    try:
        await client.write_event(Synthesize(text="Hi").event())
        await read_until(client, "audio-stop")
        assert model.requests[-1]["speaker"] == "narrator"
        assert model.requests[-1]["language"] is None  # auto-detect
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesize_language_picks_speaker() -> None:
    model = FakeQwenModel()
    service = make_fake_service(
        model, Settings(), speakers=("aiden", "ryan"), languages=("english", "chinese")
    )
    server, port = await start_test_server(service, Settings())
    client = await connect(port)
    try:
        await client.write_event(
            Synthesize(text="你好", voice=SynthesizeVoice(language="zh")).event()
        )
        await read_until(client, "audio-stop")
        assert model.requests[-1]["speaker"] in {"aiden", "ryan"}
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesize_unknown_voice_error() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        await client.write_event(
            Synthesize(text="Hello", voice=SynthesizeVoice(name="nope")).event()
        )
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        error = Error.from_event(event)
        assert error.code == "invalid-voice"
        assert "aiden" in error.text
        # connection stays usable
        await client.write_event(Ping(text="still-alive").event())
        event = await client.read_event()
        assert event is not None and Pong.is_type(event.type)
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesize_unsupported_language_error() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        await client.write_event(
            Synthesize(text="Ola", voice=SynthesizeVoice(language="pt-BR")).event()
        )
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        assert Error.from_event(event).code == "invalid-language"
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesize_empty_text_error() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        await client.write_event(Synthesize(text="   ").event())
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        assert Error.from_event(event).code == "empty-text"
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesize_ssml_rejected() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        await client.write_event(Synthesize(text="<speak>hi</speak>", text_format="ssml").event())
        event = await client.read_event()
        assert event is not None and Error.is_type(event.type)
        assert Error.from_event(event).code == "text-format-not-supported"
    finally:
        await client.disconnect()
        await server.stop()


async def test_streaming_synthesize_flow() -> None:
    model = FakeQwenModel()
    client, _, server = await _start(model)
    try:
        await client.write_event(SynthesizeStart(voice=SynthesizeVoice(name="aiden")).event())
        await client.write_event(SynthesizeChunk(text="It is a real ").event())
        await client.write_event(SynthesizeChunk(text="pleasure to meet you.").event())
        # backwards-compat echo of the full text inside the streaming flow
        await client.write_event(Synthesize(text="It is a real pleasure to meet you.").event())
        await client.write_event(SynthesizeStop().event())

        events = await read_until(client, "synthesize-stopped")
        assert SynthesizeStopped.is_type(events[-1].type)
        assert collect_audio(events) == expected_pcm()  # 34 chars -> single piece
        assert model.requests[-1]["speaker"] == "aiden"
        assert model.requests[-1]["text"] == "It is a real pleasure to meet you."
        assert len(model.requests) == 1
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesis_failure_sends_error() -> None:
    model = FakeQwenModel()
    model.fail = RuntimeError("boom")
    client, _, server = await _start(model)
    try:
        await client.write_event(Synthesize(text="Hello").event())
        # audio-start is written before synthesis; the error follows it
        events = await read_until(client, "error")
        error_event = next(e for e in events if Error.is_type(e.type))
        assert Error.from_event(error_event).code == "synthesis-failed"
    finally:
        await client.disconnect()
        await server.stop()


async def test_select_program_ignored() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        await client.write_event(SelectProgram(name="something-else").event())
        await client.write_event(Ping(text="ok").event())
        event = await client.read_event()
        assert event is not None and Pong.is_type(event.type)
    finally:
        await client.disconnect()
        await server.stop()


async def test_unknown_event_ignored() -> None:
    client, _, server = await _start(FakeQwenModel())
    try:
        # transcribe is handled now, so use a genuinely unknown event type
        await client.write_event(Event(type="transcribe-nope", data={}))
        await client.write_event(Ping(text="after").event())
        event = await client.read_event()
        assert event is not None and Pong.is_type(event.type)
    finally:
        await client.disconnect()
        await server.stop()


async def test_synthesis_serialized_across_connections() -> None:
    model = FakeQwenModel()
    service = make_fake_service(model, Settings(output_chunk_ms=100))
    server, port = await start_test_server(service, Settings())
    clients = [await connect(port) for _ in range(2)]
    try:

        async def synthesize(client) -> bytes:
            await client.write_event(Synthesize(text="Concurrent").event())
            events = await read_until(client, "audio-stop")
            return collect_audio(events)

        results = await asyncio.gather(*(synthesize(c) for c in clients))
        assert all(r == expected_pcm() for r in results)
    finally:
        for c in clients:
            await c.disconnect()
        await server.stop()


async def test_fast_backend_streams_multiple_audio_chunks(tmp_path) -> None:
    """The fast backend must emit audio while generating, not one blob at the
    end: 3 streamed pieces x (100ms piece / 50ms chunk) = 6 chunks."""
    from wyoming.tts import Synthesize, SynthesizeVoice

    from qwen3_tts_stripped_wyoming.audio import float_to_int16_bytes
    from qwen3_tts_stripped_wyoming.config import Settings

    from ..fakes import FakeFastModel, fake_audio_piece
    from ..utils import connect, make_fake_service, read_until, start_test_server

    model = FakeFastModel(pieces=3)
    settings = Settings(model_dir=tmp_path, output_chunk_ms=50, stream_chunk_steps=8)
    service = make_fake_service(model, settings, backend="fast")
    server, port = await start_test_server(service, settings)
    client = await connect(port)
    try:
        await client.write_event(
            Synthesize(text="Stream me.", voice=SynthesizeVoice(name="aiden")).event()
        )
        events = await read_until(client, "audio-stop")
        chunks = [e for e in events if AudioChunk.is_type(e.type)]
        assert len(chunks) == 6
        assert model.requests[-1]["chunk_size"] == 8
        audio = b"".join(AudioChunk.from_event(e).audio for e in chunks)
        expected = np.concatenate([fake_audio_piece()] * 3)
        assert audio == float_to_int16_bytes(expected)
    finally:
        await client.disconnect()
        await server.stop()
