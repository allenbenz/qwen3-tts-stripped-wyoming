"""Runtime patches for lite / q8 checkpoints (applied on subpackage import).

Importing :mod:`qwen3_tts_stripped_wyoming.patches.lite` installs vocabulary
pruning + decoder-only speech-tokenizer support; importing
:mod:`...patches.q8` additionally swaps talker linears for int8 weight-only
modules (it implies the lite patches). Both are no-ops for plain bf16
checkpoints except the q8 linear swap, which must only be imported when
actually serving a q8 model (one variant per process).
"""

from __future__ import annotations

import importlib

__all__ = ["apply_for_variant"]


def apply_for_variant(variant: str) -> None:
    """Import the patch module matching the served checkpoint variant.

    bf16 needs no patches; lite needs the marker-driven lite patches; q8 needs
    the lite patches plus the linear swap (importing q8 applies both).
    """
    if variant not in ("bf16", "lite", "q8"):
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "q8":
        importlib.import_module(f"{__name__}.q8")
    elif variant == "lite":
        importlib.import_module(f"{__name__}.lite")
