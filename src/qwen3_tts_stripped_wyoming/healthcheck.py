"""Container healthcheck: ping the Wyoming server and exit 0/1.

Used by the Dockerfile HEALTHCHECK. Usage:
``python -m qwen3_tts_stripped_wyoming.healthcheck [uri]``
"""

from __future__ import annotations

import asyncio
import sys

from wyoming.client import AsyncClient
from wyoming.ping import Ping

DEFAULT_URI = "tcp://127.0.0.1:10200"


async def check(uri: str = DEFAULT_URI) -> bool:
    client = AsyncClient.from_uri(uri, connect_timeout=3.0, read_timeout=3.0)
    try:
        await client.connect()
        await client.write_event(Ping(text="health").event())
        event = await client.read_event()
        return event is not None
    finally:
        await client.disconnect()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    uri = argv[0] if argv else DEFAULT_URI
    try:
        healthy = asyncio.run(check(uri))
    except Exception:
        healthy = False  # any failure (refused, timeout, protocol) is unhealthy
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
