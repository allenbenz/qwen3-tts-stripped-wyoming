"""Runtime support for int8 weight-only ("q8") Qwen3-TTS checkpoints.

Stacks on the lite patches (importing this module applies both). Every
nn.Linear under the talker is replaced by a Q8Linear that keeps the weight as
group-wise symmetric int8 (``weight_q8`` int8 [out, in] plus ``weight_scale``
fp32 [out, in//group]) -- exactly the tensors written by the converter -- and
dequantizes in fp32 at forward time. Embeddings/norms/biases are untouched.

Mismatch behaviour: loading a non-q8 checkpoint with this module imported (or
a q8 checkpoint without it) makes transformers log missing/unexpected weight
keys and produces garbage audio -- match the patch to the checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

from . import lite

__all__ = ["Q8Linear", "apply"]


class Q8Linear(nn.Module):
    """nn.Linear with the weight stored as group-wise symmetric int8.

    Buffers match the converter's checkpoint keys:
        <prefix>.weight_q8     int8  [out_features, in_features]
        <prefix>.weight_scale  fp32  [out_features, n_groups]
    (n_groups == 1 means per-channel; the group size is implied by
        in_features // n_groups, so checkpoints of either flavor load.)
    """

    def __init__(self, linear: nn.Linear, group: int = 0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group = group or self.in_features  # 0 -> per-channel equivalent
        self.weight_q8: torch.Tensor
        self.weight_scale: torch.Tensor
        self._checked = False
        if linear.bias is not None:
            self.bias = nn.Parameter(torch.empty_like(linear.bias))
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "weight_q8",
            torch.empty(self.out_features, self.in_features, dtype=torch.int8),
            persistent=True,
        )
        # placeholder shaped exactly like the checkpoint tensor: group>0 ->
        # [out, in//effective_group], else per-channel [out]. load_state_dict
        # checks shapes even under assign=True, so this must match; the
        # converter applies the same effective-group rule.
        if group:
            from ..convert import effective_group

            eff = effective_group(linear.in_features, group)
            shape: tuple[int, ...] = (self.out_features, linear.in_features // eff)
        else:
            shape = (self.out_features,)
        self.register_buffer(
            "weight_scale", torch.empty(shape, dtype=torch.float32), persistent=True
        )
        self._checked = False

    def _sanity(self) -> None:
        if self.weight_q8.numel() == 0 or self.weight_scale.numel() == 0:
            raise RuntimeError(
                "Q8Linear has empty buffers -- the checkpoint was not loaded. "
                "If you meant to run a bf16/lite (non-q8) checkpoint, do not "
                "import the q8 patch."
            )
        self._checked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._checked:
            self._sanity()
        # dequant in fp32, cast once: int8 values are exact in fp32 and the
        # fp32 scale buffer is not pre-cast, so the only rounding vs the
        # original bf16 weight is the int8 step (+ the final cast to x.dtype).
        q = self.weight_q8.to(torch.float32)  # [out, in]
        s = self.weight_scale  # [out] or [out, ng]
        if s.dim() == 1:  # per-channel
            w = q * s.unsqueeze(1)
        else:  # group-wise
            ng = s.shape[1]
            w = (q.view(self.out_features, ng, -1) * s.unsqueeze(2)).view_as(q)
        return F.linear(x, w.to(dtype=x.dtype), self.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight (introspection / debugging only)."""
        q = self.weight_q8.float()
        s = self.weight_scale
        if s.dim() == 1:
            return q * s.unsqueeze(1)
        return (q.view(self.out_features, s.shape[1], -1) * s.unsqueeze(2)).view_as(q)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant=int8-per-channel"
        )


def _swap_linears(module: nn.Module, group: int = 0) -> None:
    """Recursively replace nn.Linear children with Q8Linear."""
    for name, child in list(module.named_children()):
        if isinstance(child, Q8Linear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, Q8Linear(child, group=group))
        else:
            _swap_linears(child, group)


_orig_init = Qwen3TTSTalkerForConditionalGeneration.__init__


def _talker_cfg_init(self, config, *args, **kwargs):
    _orig_init(self, config, *args, **kwargs)
    # All quantized weights live under the talker (talker.model layers,
    # code predictor, text_projection, codec_head). The speech tokenizer is
    # attached to the top-level model later and is never touched here.
    # The talker-level "q8" block written by the converter sizes the scale
    # buffers to match the checkpoint exactly.
    q8cfg = getattr(config, "q8", None)
    group = int(q8cfg.get("group_size", 0)) if isinstance(q8cfg, dict) else 0
    _swap_linears(self, group)


def apply() -> None:
    """Idempotently install the q8 patches (implies the lite patches)."""
    lite.apply()
    if getattr(Qwen3TTSTalkerForConditionalGeneration, "_q8_patched", False):
        return
    Qwen3TTSTalkerForConditionalGeneration.__init__ = _talker_cfg_init
    Qwen3TTSTalkerForConditionalGeneration._q8_patched = True


apply()
