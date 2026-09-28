"""CLI entry point: ``qwen3-tts-stripped-wyoming`` or ``python -m qwen3_tts_stripped_wyoming``."""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import build_arg_parser, settings_from_args
from .server import run_server


def run(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = settings_from_args(args)
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # qwen-tts / transformers are chatty at INFO; keep our own signal readable
    for noisy in ("transformers", "qwen_tts", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        asyncio.run(run_server(settings))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
