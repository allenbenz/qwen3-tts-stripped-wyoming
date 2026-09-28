"""CLI entry point: ``qwen3-tts-stripped-wyoming`` or ``python -m qwen3_tts_stripped_wyoming``."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# PyPI linux torch bundles Triton, and torch >= 2.14's native-op registry
# silently routes tiny outer-product bmms (e.g. the TTS RoPE matmul) to a
# Triton kernel. Triton JIT-compiles its bootstrap module with the system C
# compiler at runtime, which slim containers don't have. We run eager
# inference only (no torch.compile), so disable the registry and keep aten /
# cuBLAS. setdefault: an explicit env value still wins.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

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
