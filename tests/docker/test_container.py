"""Docker image tests (marker: docker). Enable with QWEN3TTS_DOCKER_E2E=1.

Builds the image and checks that the server starts, answers ping/describe,
and synthesizes speech end-to-end in the container. Skipped everywhere else.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time

import pytest

pytestmark = pytest.mark.docker

IMAGE = os.environ.get("QWEN3TTS_IMAGE", "qwen3-tts-stripped-wyoming:dev")
MODEL_DIR = os.environ.get("QWEN3TTS_DOCKER_MODEL_DIR", "")


@pytest.fixture(scope="module")
def container():
    if not os.environ.get("QWEN3TTS_DOCKER_E2E"):
        pytest.skip("set QWEN3TTS_DOCKER_E2E=1 to run docker tests")
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("docker not available")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    cmd = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        "qwen3tts-wyoming-test",
        "-p",
        f"127.0.0.1:{port}:10200",
    ]
    if MODEL_DIR:
        cmd += ["-v", f"{MODEL_DIR}:/data"]
    cmd += [IMAGE]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    container_id = result.stdout.strip()
    try:
        # wait for the server to answer pings (model download + convert + load)
        deadline = time.monotonic() + 45 * 60
        ready = False
        while time.monotonic() < deadline:
            try:
                from wyoming.client import AsyncClient
                from wyoming.ping import Ping

                async def probe() -> bool:
                    client = AsyncClient.from_uri(
                        f"tcp://127.0.0.1:{port}", connect_timeout=3.0, read_timeout=3.0
                    )
                    try:
                        await client.connect()
                        await client.write_event(Ping().event())
                        return await client.read_event() is not None
                    finally:
                        await client.disconnect()

                if asyncio.run(probe()):
                    ready = True
                    break
            except Exception:
                time.sleep(5)
        assert ready, "server did not become ready in time"
        yield port
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


async def test_container_synthesizes(container) -> None:
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.client import AsyncClient
    from wyoming.tts import Synthesize

    port = container
    client = AsyncClient.from_uri(
        f"tcp://127.0.0.1:{port}", connect_timeout=10.0, read_timeout=300.0
    )
    await client.connect()
    try:
        await client.write_event(Synthesize(text="Docker test.").event())
        saw_start = saw_stop = False
        total = bytearray()
        while True:
            event = await client.read_event()
            assert event is not None
            if AudioStart.is_type(event.type):
                saw_start = True
            elif AudioChunk.is_type(event.type):
                total.extend(AudioChunk.from_event(event).audio)
            elif AudioStop.is_type(event.type):
                saw_stop = True
                break
        assert saw_start and saw_stop
        assert len(total) > 24000  # at least ~0.5 s of 16 kHz+ audio
    finally:
        await client.disconnect()
