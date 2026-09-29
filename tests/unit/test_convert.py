"""Unit tests for the lite/q8 converters on a tiny synthetic model.

The fixture builds a miniature Qwen3-TTS-shaped directory: a 256-token
byte-alphabet vocab with one added special token, a config.json with the
custom_voice markers, a small text embedding + two linear weights, and a
speech tokenizer with one encoder and one decoder tensor. The conversion
functions are structural, so tiny tensors exercise the whole pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from qwen3_tts_stripped_wyoming.convert import (
    ConversionError,
    asr_needs_migration,
    build_closure_keep_set,
    bytes_to_unicode,
    convert_to_lite,
    convert_to_q8,
    directory_size,
    ensure_variant,
    migrate_asr_for_transformers5,
    read_model_info,
)


def _tiny_vocab() -> dict[str, int]:
    """Full 256-token byte alphabet plus two merged ASCII tokens."""
    vocab = {char: i for i, char in enumerate(bytes_to_unicode().values())}
    vocab["ab"] = len(vocab)
    vocab["Ġc"] = len(vocab)
    return vocab


def _write_source(root: Path) -> Path:
    model = root / "src-model"
    (model / "speech_tokenizer").mkdir(parents=True)

    vocab = _tiny_vocab()
    full_vocab = 260  # 258 + added tokens
    (model / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (model / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {"258": {"content": "<|im_end|>"}}}),
        encoding="utf-8",
    )
    (model / "merges.txt").write_text("# version\n", encoding="utf-8")

    tensors = {
        "talker.model.text_embedding.weight": torch.randn(full_vocab, 8, dtype=torch.bfloat16),
        "talker.model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "talker.text_projection.linear_fc1.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "talker.text_projection.linear_fc1.bias": torch.randn(8, dtype=torch.bfloat16),
        "talker.model.norm.weight": torch.randn(8, dtype=torch.bfloat16),
    }
    save_file(tensors, str(model / "model.safetensors"))

    st = {
        "encoder.encoder.conv.weight": torch.randn(4, 4, 3),
        "decoder.decoder.conv.weight": torch.randn(4, 4, 3),
    }
    save_file(st, str(model / "speech_tokenizer" / "model.safetensors"))
    (model / "speech_tokenizer" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_tts_tokenizer_12hz",
                "encoder_config": {"hidden_size": 4},
                "decoder_config": {"hidden_size": 4},
            }
        ),
        encoding="utf-8",
    )
    (model / "speech_tokenizer" / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    (model / "config.json").write_text(
        json.dumps(
            {
                "tts_model_type": "custom_voice",
                "talker_config": {
                    "text_vocab_size": full_vocab,
                    "text_hidden_size": 8,
                    "spk_id": {"narrator": 3000},
                    "codec_language_id": {"english": 2050, "chinese": 2055},
                },
            }
        ),
        encoding="utf-8",
    )
    return model


def _write_asr_source(root: Path) -> Path:
    """Miniature 4.57-era qwen-asr checkpoint (thinker_config nesting)."""
    model = root / "asr-src"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_asr",
                "thinker_config": {
                    "audio_config": {"num_mel_bins": 128, "n_window": 50},
                    "text_config": {"vocab_size": 100},
                    "audio_token_id": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        "thinker.lm_head.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "thinker.model.layers.0.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "thinker.audio_tower.proj1.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "thinker.audio_tower.proj2.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "thinker.audio_tower.conv.weight": torch.randn(4, 4, 3, dtype=torch.bfloat16),
    }
    save_file(tensors, str(model / "model.safetensors"))
    (model / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor_type": "WhisperFeatureExtractor",
                "feature_size": 128,
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return model


@pytest.fixture
def source_model(tmp_path: Path) -> Path:
    return _write_source(tmp_path)


class TestReadModelInfo:
    def test_detects_bf16(self, source_model: Path) -> None:
        info = read_model_info(source_model)
        assert info.variant == "bf16"
        assert info.tts_model_type == "custom_voice"
        assert info.convertible
        assert info.speakers == ("narrator",)
        assert set(info.languages) == {"english", "chinese"}

    def test_detects_lite(self, source_model: Path, tmp_path: Path) -> None:
        lite = convert_to_lite(source_model, tmp_path / "lite")
        info = read_model_info(lite)
        assert info.variant == "lite"

    def test_detects_q8(self, source_model: Path, tmp_path: Path) -> None:
        lite = convert_to_lite(source_model, tmp_path / "lite")
        q8 = convert_to_q8(lite, tmp_path / "q8")
        assert read_model_info(q8).variant == "q8"

    def test_missing_config(self, tmp_path: Path) -> None:
        with pytest.raises(ConversionError):
            read_model_info(tmp_path)


class TestClosureKeepSet:
    def test_ascii_and_added_tokens_kept(self) -> None:
        vocab = _tiny_vocab()
        keep = set(build_closure_keep_set(vocab, {258}, [(0x0000, 0x007F)]))
        assert set(vocab.values()) <= keep  # everything in the tiny vocab is ASCII
        assert 258 in keep  # added tokens always kept

    def test_cjk_excluded_from_latin(self) -> None:
        vocab = _tiny_vocab()
        b2u = bytes_to_unicode()
        # a token whose decoded bytes are CJK (U+4F60 '你' = E4 BD A0), encoded
        # with the same byte-alphabet the real vocab.json uses
        cjk_token = "".join(b2u[b] for b in "你".encode())
        vocab[cjk_token] = len(vocab)
        latin = set(build_closure_keep_set(vocab, set(), [(0x0000, 0x007F)]))
        ml = set(build_closure_keep_set(vocab, set(), [(0x0000, 0x007F), (0x4E00, 0x9FFF)]))
        assert vocab[cjk_token] in ml
        assert vocab[cjk_token] not in latin


class TestConvertToLite:
    def test_prunes_and_maps(self, source_model: Path, tmp_path: Path) -> None:
        dst = convert_to_lite(source_model, tmp_path / "out", keep_set="latin")
        src_tensors = load_file(str(source_model / "model.safetensors"))
        out_tensors = load_file(str(dst / "model.safetensors"))

        embed = src_tensors["talker.model.text_embedding.weight"]
        token_map = out_tensors["talker.model.text_token_map"]
        keep = (token_map != 0).nonzero(as_tuple=True)[0]
        assert token_map.shape == (260,)
        assert out_tensors["talker.model.text_embedding.weight"].shape == (
            len(keep) + 1,
            8,
        )
        # kept rows are bit-identical copies; row 0 is the zero trash row
        pruned = out_tensors["talker.model.text_embedding.weight"]
        assert torch.equal(pruned[1:], embed[keep])
        assert torch.equal(pruned[0], torch.zeros_like(pruned[0]))
        assert torch.equal(embed[keep], pruned[token_map[keep].long()])
        # other tensors untouched
        assert torch.equal(
            out_tensors["talker.model.layers.0.self_attn.q_proj.weight"],
            src_tensors["talker.model.layers.0.self_attn.q_proj.weight"],
        )

        cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
        assert cfg["talker_config"]["vocab_pruning"]["keep_set"] == "latin"

    def test_speech_tokenizer_decoder_only(self, source_model: Path, tmp_path: Path) -> None:
        dst = convert_to_lite(source_model, tmp_path / "out", st_dtype="float16")
        st = load_file(str(dst / "speech_tokenizer" / "model.safetensors"))
        assert set(st) == {"decoder.decoder.conv.weight"}
        assert st["decoder.decoder.conv.weight"].dtype == torch.float16
        cfg = json.loads((dst / "speech_tokenizer" / "config.json").read_text(encoding="utf-8"))
        assert "encoder_config" not in cfg
        assert cfg["decoder_only_lite"] is True

    def test_refuses_base_model(self, source_model: Path, tmp_path: Path) -> None:
        cfg = json.loads((source_model / "config.json").read_text(encoding="utf-8"))
        cfg["tts_model_type"] = "base"
        (source_model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ConversionError, match="custom_voice"):
            convert_to_lite(source_model, tmp_path / "out")


class TestConvertToQ8:
    def test_quantizes_linears(self, tmp_path: Path, source_model: Path) -> None:
        lite = convert_to_lite(source_model, tmp_path / "lite")
        dst = convert_to_q8(lite, tmp_path / "q8", group=8)
        src_tensors = load_file(str(lite / "model.safetensors"))
        out_tensors = load_file(str(dst / "model.safetensors"))

        assert "talker.model.layers.0.self_attn.q_proj.weight_q8" in out_tensors
        assert "talker.model.layers.0.self_attn.q_proj.weight" not in out_tensors
        q = out_tensors["talker.model.layers.0.self_attn.q_proj.weight_q8"]
        s = out_tensors["talker.model.layers.0.self_attn.q_proj.weight_scale"]
        assert q.dtype == torch.int8 and q.shape == (8, 8)
        assert s.dtype == torch.float32 and s.shape == (8, 1)  # 8 inputs / group 8

        w = src_tensors["talker.model.layers.0.self_attn.q_proj.weight"].float()
        deq = (q.float().view(8, 1, 8) * s.unsqueeze(2)).view_as(q)
        rel = ((w - deq).norm() / w.norm()).item()
        assert rel < 0.05  # int8 on a random 8x8: coarse but bounded

        # embeddings and norms are not quantized
        assert "talker.model.text_embedding.weight" in out_tensors
        assert out_tensors["talker.model.text_embedding.weight"].dtype == torch.bfloat16
        assert "talker.model.norm.weight" in out_tensors
        # vocab pruning markers survive
        cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
        assert "vocab_pruning" in cfg["talker_config"]
        assert cfg["talker_config"]["q8"]["group_size"] == 8
        assert cfg["q8_quantization"]["group_size"] == 8
        # speech tokenizer copied verbatim (already lite)
        assert (dst / "speech_tokenizer" / "config.json").is_file()

    def test_bf16_direct_to_q8(self, source_model: Path, tmp_path: Path) -> None:
        dst = convert_to_q8(source_model, tmp_path / "q8", group=8)
        out_tensors = load_file(str(dst / "model.safetensors"))
        assert "talker.model.layers.0.self_attn.q_proj.weight_q8" in out_tensors
        assert "talker.model.text_token_map" not in out_tensors  # no pruning without lite


class TestEnsureVariant:
    def test_auto_returns_source(self, source_model: Path, tmp_path: Path) -> None:
        from qwen3_tts_stripped_wyoming.config import Settings

        settings = Settings(model_dir=tmp_path / "models")
        assert ensure_variant(source_model, "auto", settings, tmp_path) == source_model

    def test_q8_pipeline_with_cache(self, source_model: Path, tmp_path: Path) -> None:
        from qwen3_tts_stripped_wyoming.config import Settings

        settings = Settings(model_dir=tmp_path / "models")
        cache = tmp_path / "converted"
        dst = ensure_variant(source_model, "q8", settings, cache)
        assert read_model_info(dst).variant == "q8"
        # intermediate lite is cached too
        lite_dirs = [p for p in cache.iterdir() if "-lite-" in p.name]
        assert len(lite_dirs) == 1
        # second call reuses without converting
        dst2 = ensure_variant(source_model, "q8", settings, cache)
        assert dst2 == dst

    def test_refuses_when_disabled(self, source_model: Path, tmp_path: Path) -> None:
        from qwen3_tts_stripped_wyoming.config import Settings

        settings = Settings(model_dir=tmp_path, convert=False)
        with pytest.raises(ConversionError, match="convert=no"):
            ensure_variant(source_model, "q8", settings, tmp_path / "c")

    def test_directory_size(self, source_model: Path) -> None:
        assert directory_size(source_model) > 0
        assert directory_size(source_model.parent / "missing") == 0.0


class TestAsrMigration:
    def test_needs_migration(self, tmp_path: Path) -> None:
        assert asr_needs_migration(_write_asr_source(tmp_path)) is True
        assert asr_needs_migration(_write_source(tmp_path)) is False

    def test_migrates_layout(self, tmp_path: Path) -> None:
        dst = migrate_asr_for_transformers5(_write_asr_source(tmp_path), tmp_path / "out")
        cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
        assert "thinker_config" not in cfg
        assert cfg["audio_config"]["model_type"] == "qwen3_asr_encoder"
        assert cfg["text_config"] == {"vocab_size": 100}
        assert cfg["audio_token_id"] == 1
        tensors = load_file(str(dst / "model.safetensors"))
        assert set(tensors) == {
            "lm_head.weight",
            "model.language_model.layers.0.weight",
            "model.multi_modal_projector.linear_1.weight",
            "model.multi_modal_projector.linear_2.weight",
            "model.audio_tower.conv.weight",
        }
        # ancillary files are copied verbatim
        assert (dst / "tokenizer_config.json").is_file()

    def test_rewrites_preprocessor_config(self, tmp_path: Path) -> None:
        dst = migrate_asr_for_transformers5(_write_asr_source(tmp_path), tmp_path / "out")
        pre = json.loads((dst / "preprocessor_config.json").read_text(encoding="utf-8"))
        assert pre["feature_extractor_type"] == "Qwen3ASRFeatureExtractor"
        assert pre["processor_class"] == "Qwen3ASRProcessor"
        assert pre["n_window"] == 50
        assert pre["min_length"] == 8000
        assert pre["return_attention_mask"] is True
        assert pre["feature_size"] == 128
        # stale whisper-only keys are dropped
        assert "n_samples" not in pre
        assert "nb_max_frames" not in pre
