"""Runtime support for pruned-vocab ("lite") Qwen3-TTS checkpoints.

The lite format stores a compact text embedding
(talker.model.text_embedding, [N+1, text_hidden]) plus an indirection map
(talker.model.text_token_map, int32 [151936]): embed(t) = E'[map[t]].
Row 0 is an all-zero trash row; every id outside the keep-set maps to it.
The speech tokenizer is decoder-only (encoder stripped) and fp16.

The stock ``qwen-tts`` package knows nothing about either change, so this
module monkeypatches it on import. Everything is backward compatible: bf16
checkpoints with a full speech tokenizer load exactly as before.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2Model,
)

__all__ = ["TokenMappedEmbedding", "apply"]

_warned_unmapped = False


class TokenMappedEmbedding(nn.Module):
    """embed(ids) = E'[ map[ids] ]  -- token-map indirection wrapper."""

    def __init__(self, embedding: nn.Embedding, token_map: torch.Tensor):
        super().__init__()
        self.embedding = embedding
        self.token_map = token_map

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        global _warned_unmapped
        m = self.token_map
        if m.device != input_ids.device:
            m = m.to(input_ids.device)
        mapped = m[input_ids]
        if not bool((mapped != 0).all()):
            # input contains tokens outside the keep-set; they embed as zeros.
            # Usually means unsupported-language text on a pruned checkpoint.
            if not _warned_unmapped and not torch.is_grad_enabled():
                warnings.warn(
                    "Qwen3-TTS lite: input token ids fell outside the pruned "
                    "vocabulary keep-set (embedded as zero vectors). If this is "
                    "unexpected, convert with keep_set=ml.",
                    stacklevel=2,
                )
                _warned_unmapped = True
        return self.embedding(mapped)


def _pruning_info(config):
    vp = getattr(config, "vocab_pruning", None)
    if isinstance(vp, dict) and vp.get("has_token_map"):
        return vp
    return None


# ---------------------------------------------------------------- talker ----
_orig_talker_init = Qwen3TTSTalkerModel.__init__
_orig_talker_get_text_embeddings = Qwen3TTSTalkerModel.get_text_embeddings


def _talker_init(self, config):
    _orig_talker_init(self, config)
    vp = _pruning_info(config)
    if vp is None:
        return
    # shrink the embedding to the compact vocab and register the indirection
    # map as a persistent buffer named exactly like the checkpoint tensor
    # (talker.model.text_token_map). Unused ids point at row 0 (all zeros).
    self.text_embedding = nn.Embedding(int(vp["compact_vocab_size"]), config.text_hidden_size)
    self.register_buffer(
        "text_token_map",
        torch.zeros(int(vp["original_text_vocab_size"]), dtype=torch.long),
        persistent=True,
    )


def _talker_get_text_embeddings(self):
    if getattr(self, "text_token_map", None) is not None:
        return TokenMappedEmbedding(self.text_embedding, self.text_token_map)
    return _orig_talker_get_text_embeddings(self)


# ------------------------------------------------------- speech tokenizer ---
_orig_v2_init = Qwen3TTSTokenizerV2Model.__init__
_orig_v2_encode = Qwen3TTSTokenizerV2Model.encode


def _v2_init(self, config):
    if not getattr(config, "decoder_only_lite", False):
        _orig_v2_init(self, config)
        return
    # decoder-only "lite" speech tokenizer (encoder stripped; CustomVoice
    # generate/decode never touches it). NB: checking encoder_config is not
    # enough -- the config class defaults it.
    super(Qwen3TTSTokenizerV2Model, self).__init__(config)
    self.config = config
    self.encoder_valid_num_quantizers = getattr(config, "encoder_valid_num_quantizers", None)
    self.input_sample_rate = config.input_sample_rate
    self.output_sample_rate = config.output_sample_rate
    self.decode_upsample_rate = config.decode_upsample_rate
    self.encode_downsample_rate = getattr(config, "encode_downsample_rate", None)
    self.encoder = None
    self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(self.config.decoder_config)
    self.post_init()


def _v2_encode(self, *args, **kwargs):
    if getattr(self, "encoder", None) is None:
        raise RuntimeError(
            "this checkpoint ships a decoder-only speech tokenizer "
            "(encoder stripped to save space); audio->codes (voice cloning) is "
            "not available. Use the full speech tokenizer for encode."
        )
    return _orig_v2_encode(self, *args, **kwargs)


def apply() -> None:
    """Idempotently install the patches."""
    if getattr(Qwen3TTSTalkerModel, "_lite_patched", False):
        return
    Qwen3TTSTalkerModel.__init__ = _talker_init
    Qwen3TTSTalkerModel.get_text_embeddings = _talker_get_text_embeddings
    Qwen3TTSTalkerModel._lite_patched = True
    Qwen3TTSTokenizerV2Model.__init__ = _v2_init
    Qwen3TTSTokenizerV2Model.encode = _v2_encode
    Qwen3TTSTokenizerV2Model._lite_patched = True


apply()
