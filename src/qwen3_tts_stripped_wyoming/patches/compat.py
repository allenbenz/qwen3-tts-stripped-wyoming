"""Compatibility shims for qwen-tts-hf (Transformers 5) at runtime.

``qwen_tts._transformers_compat._default_rope_parameters`` reads
``config.rope_theta`` unconditionally. The speech tokenizer's *encoder* is a
transformers ``MimiModel`` whose native ``MimiConfig`` has no ``rope_theta``
attribute, so loading an unconverted (bf16) checkpoint crashes in weight init
with ``AttributeError: 'MimiConfig' object has no attribute 'rope_theta'``.
The shim falls back to the default theta (10000.0) that the ST config itself
documents for its decoder.

Import this before loading any model (the server does, in runtime/asr).
"""

from __future__ import annotations

import qwen_tts._transformers_compat as _tf_compat

__all__ = ["apply"]


def apply() -> None:
    """Idempotently install the shim."""
    if getattr(_tf_compat._default_rope_parameters, "_qwen3tts_compat", False):
        return

    original = _tf_compat._default_rope_parameters

    def _default_rope_parameters_compat(config, *args, **kwargs):
        if not hasattr(config, "rope_theta"):
            # MimiConfig (ST encoder) has no rope_theta; the ST decoder's
            # documented default is 10000.0
            config = _ShimConfigView(config, rope_theta=10000.0)
        return original(config, *args, **kwargs)

    _default_rope_parameters_compat._qwen3tts_compat = True  # type: ignore[attr-defined]
    _tf_compat._default_rope_parameters = _default_rope_parameters_compat
    # qwen_tts registers the original into transformers' RoPE init registry at
    # import time; the registry entry holds the original reference, so update
    # it too (if qwen_tts was already imported).
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if ROPE_INIT_FUNCTIONS.get("default") is original:
            ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters_compat
    except ImportError:
        pass


class _ShimConfigView:
    """Read-through wrapper adding attributes to a config without mutating it."""

    def __init__(self, inner, **extra):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_extra", extra)

    def __getattr__(self, name):
        if name in object.__getattribute__(self, "_extra"):
            return object.__getattribute__(self, "_extra")[name]
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)


apply()
