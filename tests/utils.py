"""Helpers for starting a real Wyoming TCP server backed by a fake service."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.server import AsyncTcpServer

from qwen3_tts_stripped_wyoming.asr import TranscriptionService
from qwen3_tts_stripped_wyoming.config import Settings
from qwen3_tts_stripped_wyoming.runtime import Speaker, SynthesisService
from qwen3_tts_stripped_wyoming.server import make_handler_factory


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_fake_service(
    model,
    settings: Settings | None = None,
    *,
    speakers: tuple[str, ...] = ("narrator", "aiden"),
    languages: tuple[str, ...] = ("english", "chinese", "german"),
    variant: str = "q8",
) -> SynthesisService:
    """A SynthesisService around a fake model (no network, no GPU)."""
    speaker_specs = tuple(Speaker(id=name, languages=languages) for name in speakers)
    return SynthesisService(
        model,
        settings or Settings(),
        variant=variant,
        model_dir=Path("."),
        speakers=speaker_specs,
        languages=languages,
        using_cuda=False,
    )


def make_fake_asr_service(
    model,
    settings: Settings | None = None,
    *,
    languages: tuple[str, ...] = ("English", "Chinese", "German"),
) -> TranscriptionService:
    return TranscriptionService(model, settings or Settings(), languages=languages)


async def start_test_server(
    service: SynthesisService,
    settings: Settings,
    asr_service: TranscriptionService | None = None,
) -> tuple[AsyncTcpServer, int]:
    """Start a Wyoming TCP server on an ephemeral port. Caller must stop()."""
    from qwen3_tts_stripped_wyoming.server import build_info

    info = build_info(service, asr_service)
    port = free_port()
    server = AsyncTcpServer("127.0.0.1", port)
    lock = asyncio.Lock()
    factory = make_handler_factory(service, info, settings, lock, asr_service)
    await server.start(factory)
    return server, port


async def connect(port: int, read_timeout: float = 10.0) -> AsyncClient:
    client = AsyncClient.from_uri(
        f"tcp://127.0.0.1:{port}", connect_timeout=5.0, read_timeout=read_timeout
    )
    await client.connect()
    return client


async def read_until(client: AsyncClient, *stop_types: str) -> list[Event]:
    """Read events until one of ``stop_types`` arrives; returns everything read."""
    events: list[Event] = []
    while True:
        event = await client.read_event()
        assert event is not None, "connection closed before expected event"
        events.append(event)
        if event.type in stop_types:
            return events
