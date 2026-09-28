"""Unit tests for the vendored runtime patches (Q8Linear dequant)."""

from __future__ import annotations


def _fake_checkpoint_q8linear(shape: tuple[int, int], group: int, seed: int = 7):
    """A Q8Linear whose buffers were filled as convert_to_q8 would fill them."""
    import torch

    from qwen3_tts_stripped_wyoming.patches.q8 import Q8Linear

    torch.manual_seed(seed)
    linear = torch.nn.Linear(shape[1], shape[0], bias=True, dtype=torch.bfloat16)
    q8 = Q8Linear(linear, group=group)
    with torch.no_grad():
        w = linear.weight.float()
        out_f, in_f = w.shape
        if group:
            w3 = w.view(out_f, in_f // group, group)
            scale = w3.abs().amax(dim=2).div(127.0).clamp_(min=1e-12)
            q = torch.round(w3 / scale.unsqueeze(2)).clamp_(-127, 127)
            q8.weight_q8.copy_(q.view(out_f, in_f).to(torch.int8))
            q8.weight_scale.copy_(scale)
        else:  # per-channel layout
            scale = w.abs().amax(dim=1).div(127.0).clamp_(min=1e-12)
            q = torch.round(w / scale.unsqueeze(1)).clamp_(-127, 127)
            q8.weight_q8.copy_(q.to(torch.int8))
            q8.weight_scale.copy_(scale)
        q8.bias.copy_(linear.bias)
    q8._checked = True
    return q8


class TestQ8LinearForward:
    def test_groupwise_dequant_linear(self) -> None:
        import torch

        q8 = _fake_checkpoint_q8linear((8, 16), group=8)
        x = torch.randn(3, 16, dtype=torch.bfloat16)  # x dtype matches the bias
        out = q8(x)
        assert out.shape == (3, 8)
        # dequantized property exposes the reconstruction
        w = q8.weight
        assert w.shape == (8, 16)
        torch.testing.assert_close(
            out.float(), (x.float() @ w.T + q8.bias.float()), rtol=1e-2, atol=1e-2
        )

    def test_per_channel_dequant_linear(self) -> None:
        import torch

        q8 = _fake_checkpoint_q8linear((8, 16), group=0)
        assert q8.weight_scale.shape == (8,)
        assert q8(torch.randn(2, 16, dtype=torch.bfloat16)).shape == (2, 8)

    def test_effective_group_rule_matches_converter(self) -> None:
        from qwen3_tts_stripped_wyoming.convert import effective_group

        assert effective_group(2048, 64) == 64
        assert effective_group(1024, 64) == 64
        assert effective_group(8, 64) == 8  # shrinks for odd shapes
        assert effective_group(48, 64) == 16
        assert effective_group(7, 64) == 1
