"""Compatibility shim: qwen-asr 0.0.6 on Transformers 5 (import-time only).

The faster-qwen3-tts integration moved the environment to Transformers 5.
The qwen-asr package (pinned to 4.57) can still be *imported* for its
utilities after two fixes:

1. qwen-asr decorates forwards with ``@check_model_inputs()`` (parens form);
   Transformers 5 removed the no-argument form. The shim wraps it so both
   ``@check_model_inputs()`` and ``@check_model_inputs`` work.
2. Transformers 5 ships native ``qwen3_asr`` support, so qwen-asr's
   ``AutoConfig/AutoModel/AutoProcessor.register("qwen3_asr", ...)`` raises;
   duplicate registrations become no-ops.

Inference itself no longer runs through qwen-asr: the server drives the
native Transformers-5 Qwen3ASR classes with checkpoints migrated by
:func:`qwen3_tts_stripped_wyoming.convert.migrate_asr_for_transformers5`.
"""

from __future__ import annotations

import functools

import transformers.utils.generic as _tf_generic
from transformers.models.auto import configuration_auto, modeling_auto, processing_auto

_orig_check_model_inputs = _tf_generic.check_model_inputs


@functools.wraps(_orig_check_model_inputs)
def _check_model_inputs_compat(func=None):
    if func is not None:  # bare decorator form: transformers 5 native
        return _orig_check_model_inputs(func)

    def decorator(fn):  # parens form used by qwen-asr
        return _orig_check_model_inputs(fn)

    return decorator


def _tolerant_register(original):
    """Make Auto*.register ignore name collisions (keep the native mapping)."""

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ValueError:
            return None  # already registered natively; keep the native class

    return wrapper


def _make_tolerant(auto_cls, attr: str) -> None:
    current = getattr(auto_cls, attr)
    setattr(auto_cls, attr, _tolerant_register(current))


def apply() -> None:
    """Idempotently install the shims."""
    if getattr(_tf_generic.check_model_inputs, "_qwen3tts_compat", False):
        return
    _tf_generic.check_model_inputs = _check_model_inputs_compat
    _tf_generic.check_model_inputs._qwen3tts_compat = True  # type: ignore[attr-defined]
    _make_tolerant(configuration_auto.AutoConfig, "register")
    _make_tolerant(modeling_auto.AutoModel, "register")
    _make_tolerant(processing_auto.AutoProcessor, "register")


apply()
