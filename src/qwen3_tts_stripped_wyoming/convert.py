"""Offline conversion of Qwen3-TTS checkpoints into the lite / q8 variants.

Two transformations, both pure-local (safetensors + JSON on CPU, no network,
no training), generalized from the narrator-tts work:

* **lite** -- vocabulary pruning via token-map indirection (script-closure
  keep-set, so any input written in the covered scripts is guaranteed to stay
  inside the kept vocabulary) plus a decoder-only fp16 speech tokenizer.
* **q8** -- every non-embedding ``nn.Linear`` weight becomes group-wise
  symmetric int8 (fp32 scales).

Only ``custom_voice`` models are convertible: the Base models need the speech
tokenizer encoder for voice cloning, which lite strips.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_LOGGER = logging.getLogger(__name__)

TEXT_EMBED = "talker.model.text_embedding.weight"
TOKEN_MAP = "talker.model.text_token_map"
FULL_VOCAB = 151_936

VARIANTS = ("bf16", "lite", "q8")


class ConversionError(RuntimeError):
    """The source model cannot be converted (message is user-facing)."""


@dataclass(frozen=True)
class ModelInfo:
    """Facts read from a model directory's config.json."""

    tts_model_type: str
    variant: str  # bf16 | lite | q8 (detected from config markers)
    speakers: tuple[str, ...]
    languages: tuple[str, ...]

    @property
    def convertible(self) -> bool:
        return self.tts_model_type == "custom_voice"


def read_model_info(model_dir: str | Path) -> ModelInfo:
    """Read model type, variant markers, speakers and languages."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ConversionError(f"no config.json under {model_dir}: not a model directory?")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    variant = "bf16"
    if "q8_quantization" in cfg:
        variant = "q8"
    else:
        talker = cfg.get("talker_config", {})
        if isinstance(talker, dict) and "vocab_pruning" in talker:
            variant = "lite"

    talker = cfg.get("talker_config", {}) if isinstance(cfg.get("talker_config"), dict) else {}
    speakers = tuple(str(k) for k in talker.get("spk_id", {}))
    languages = tuple(str(k) for k in talker.get("codec_language_id", {}))
    return ModelInfo(
        tts_model_type=str(cfg.get("tts_model_type", "")),
        variant=variant,
        speakers=speakers,
        languages=languages,
    )


# ---------------------------------------------------------------------------
# byte-level BPE alphabet (GPT-2 style, used by the Qwen text tokenizer)
# ---------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


_U2B = {v: k for k, v in bytes_to_unicode().items()}

# Scripts of the supported languages + punctuation/symbols/emoji.
LATIN_BLOCKS = [
    (0x0000, 0x007F),
    (0x00A0, 0x024F),
    (0x0300, 0x036F),
    (0x2000, 0x206F),
    (0x2070, 0x209F),
    (0x20A0, 0x20CF),
    (0x2100, 0x214F),
    (0x2150, 0x218F),
    (0x2190, 0x22FF),
    (0x2300, 0x23FF),
    (0x2460, 0x24FF),
    (0x2500, 0x25FF),
    (0x2600, 0x27BF),
    (0xFE00, 0xFE0F),
    (0x200B, 0x200F),
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
]
ML_EXTRA_BLOCKS = [
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x052F),  # Cyrillic
    (0x2E80, 0x2EFF),
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x3105, 0x312F),
    (0x3130, 0x318F),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xAC00, 0xD7AF),
    (0x1100, 0x11FF),
    (0xFF00, 0xFFEF),
]


def build_closure_keep_set(vocab: dict[str, int], added_ids: set[int], blocks) -> list[int]:
    """Every token whose bytes consist solely of allowed codepoints (plus all
    byte-alphabet tokens and all added/special tokens). BPE cannot emit tokens
    containing characters absent from the input, so allowed-script text always
    stays inside this set."""

    def cp_ok(cp: int) -> bool:
        return any(lo <= cp <= hi for lo, hi in blocks)

    allowed_bytes: set[int] = set()
    for cp in range(0x110000):
        if cp_ok(cp):
            allowed_bytes.update(chr(cp).encode("utf-8"))

    keep = set(added_ids)  # special/added tokens: always keep
    for s, i in vocab.items():
        if i in keep or len(s) == 1:  # byte-alphabet base tokens: always keep
            keep.add(i)
            continue
        try:
            bs = bytes(_U2B[c] for c in s)
        except KeyError:
            keep.add(i)
            continue
        try:
            if all(cp_ok(ord(c)) for c in bs.decode("utf-8")):
                keep.add(i)
        except UnicodeDecodeError:  # partial utf-8 fragment token
            if all(b in allowed_bytes for b in bs):
                keep.add(i)
    return sorted(keep)


def _keep_set_for(model_dir: Path, keep_set: str) -> torch.Tensor:
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    tkcfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    added_ids = {int(i) for i in tkcfg.get("added_tokens_decoder", {})}
    blocks = LATIN_BLOCKS if keep_set == "latin" else LATIN_BLOCKS + ML_EXTRA_BLOCKS
    return torch.tensor(build_closure_keep_set(vocab, added_ids, blocks), dtype=torch.long)


# ---------------------------------------------------------------------------
# lite conversion
# ---------------------------------------------------------------------------


def convert_to_lite(
    src: str | Path, dst: str | Path, *, keep_set: str = "latin", st_dtype: str = "float16"
) -> Path:
    """Prune the text vocabulary, strip the ST encoder, halve the ST decoder."""
    src, dst = Path(src), Path(dst)
    info = read_model_info(src)
    if not info.convertible:
        raise ConversionError(
            f"only custom_voice models can be converted (source is "
            f"tts_model_type={info.tts_model_type!r}); Base models need the "
            f"speech tokenizer encoder for voice cloning"
        )
    if info.variant == "lite":
        raise ConversionError("source is already lite; nothing to do")
    if keep_set not in ("latin", "ml"):
        raise ValueError(f"keep_set must be latin or ml, got {keep_set!r}")

    keep = _keep_set_for(src, keep_set)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    talker_cfg = cfg.get("talker_config", {})
    vocab_size = int(talker_cfg.get("text_vocab_size", FULL_VOCAB))
    m = torch.zeros(vocab_size, dtype=torch.int32)
    for tid in keep:
        if tid >= vocab_size:
            raise ConversionError(
                f"keep-set contains id {tid} beyond the model's text vocab size "
                f"{vocab_size}; is this tokenizer incompatible with the reference?"
            )
    m[keep] = torch.arange(1, len(keep) + 1, dtype=torch.int32)
    _LOGGER.info(
        "lite: keep-set %s keeps %d/%d text ids (%.1f%%)",
        keep_set,
        len(keep),
        vocab_size,
        100.0 * len(keep) / vocab_size,
    )

    # --- main model -----------------------------------------------------
    tensors = load_file(str(src / "model.safetensors"), device="cpu")
    embed = tensors.pop(TEXT_EMBED)
    if embed.ndim != 2 or embed.shape[0] != vocab_size:
        raise ConversionError(f"unexpected text embedding shape {tuple(embed.shape)}")
    pruned = torch.zeros(len(keep) + 1, embed.shape[1], dtype=embed.dtype)
    pruned[1:] = embed[keep]
    if not torch.equal(embed[keep], pruned[m[keep].long()]):
        raise ConversionError("internal error: token map round-trip failed")
    tensors[TEXT_EMBED] = pruned.contiguous()
    tensors[TOKEN_MAP] = m.contiguous()

    dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})
    del tensors, embed, pruned

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["talker_config"]["vocab_pruning"] = {
        "original_text_vocab_size": vocab_size,
        "compact_vocab_size": len(keep) + 1,
        "keep_ids_count": len(keep),
        "has_token_map": True,
        "keep_set": keep_set,
    }
    (dst / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- speech tokenizer -------------------------------------------------
    _convert_speech_tokenizer(src, dst, st_dtype)

    for f in [
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]:
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)
    _LOGGER.info("lite: wrote %s", dst)
    return dst


def _convert_speech_tokenizer(src: Path, dst: Path, st_dtype: str) -> None:
    st = src / "speech_tokenizer"
    tensors = load_file(str(st / "model.safetensors"), device="cpu")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        st_dtype
    ]

    kept, dropped = {}, 0
    for k, v in tensors.items():
        if k.startswith("encoder."):
            dropped += 1
            continue
        if v.dtype == torch.float32 and torch_dtype != torch.float32:
            v = v.to(torch_dtype)
        kept[k] = v.contiguous()
    if not any(k.startswith("decoder.") for k in kept):
        raise ConversionError("no decoder tensors found in the speech tokenizer")

    out_dir = dst / "speech_tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((st / "config.json").read_text(encoding="utf-8"))
    cfg.pop("encoder_config", None)
    cfg["decoder_only_lite"] = True
    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # the qwen-tts loader reads AutoFeatureExtractor from this subdir
    for f in ("preprocessor_config.json", "configuration.json"):
        if (st / f).is_file():
            shutil.copy2(st / f, out_dir / f)
    _LOGGER.info(
        "lite: speech tokenizer decoder-only %s (%d tensors, dropped %d)",
        st_dtype,
        len(kept),
        dropped,
    )


# ---------------------------------------------------------------------------
# q8 conversion
# ---------------------------------------------------------------------------


def _is_linear_weight(key: str, t: torch.Tensor) -> bool:
    return (
        key.endswith(".weight")
        and t.ndim == 2
        and t.dtype.is_floating_point
        and "embedding" not in key
    )


def effective_group(in_features: int, group: int) -> int:
    """Largest power-of-two divisor of ``in_features`` that is <= ``group``.

    Both the converter and the runtime patch use this rule, so scale-buffer
    shapes always match even for oddly-shaped layers.
    """
    if group <= 0:
        return 1
    g = group
    while in_features % g != 0:
        g //= 2
    return max(g, 1)


def quantize_groupwise(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [out, in] -> (int8 [out, in], fp32 scale [out, in//group]).

    When ``in`` is not divisible by ``group`` the group shrinks per
    :func:`effective_group` so oddly-shaped layers still quantize.
    """
    out_f, in_f = w.shape
    group = effective_group(in_f, group)
    w32 = w.float().view(out_f, in_f // group, group)
    scale = w32.abs().amax(dim=2).div(127.0).clamp_(min=1e-12)
    q = torch.round(w32 / scale.unsqueeze(2)).clamp_(-127, 127).to(torch.int8)
    return q.view(out_f, in_f).contiguous(), scale.contiguous()


def convert_to_q8(src: str | Path, dst: str | Path, *, group: int = 64) -> Path:
    """Quantize every non-embedding linear weight to group-wise symmetric int8.

    ``src`` may be a bf16 or lite checkpoint (vocabulary pruning is preserved
    as-is when present).
    """
    src, dst = Path(src), Path(dst)
    info = read_model_info(src)
    if not info.convertible:
        raise ConversionError(
            f"only custom_voice models can be converted (source is "
            f"tts_model_type={info.tts_model_type!r})"
        )
    if info.variant == "q8":
        raise ConversionError("source is already q8; nothing to do")

    tensors = load_file(str(src / "model.safetensors"), device="cpu")
    out: dict[str, torch.Tensor] = {}
    n_quant, worst_rel = 0, 0.0
    for key in sorted(tensors):
        t = tensors[key]
        if _is_linear_weight(key, t):
            parent = key[: -len(".weight")]
            eff = effective_group(t.shape[1], group)
            q, scale = quantize_groupwise(t, group)
            out[f"{parent}.weight_q8"] = q
            out[f"{parent}.weight_scale"] = scale
            n_quant += 1
            w32 = t.float()
            deq = q.float().view(scale.shape[0], scale.shape[1], eff) * scale.unsqueeze(2)
            err = w32 - deq.view_as(w32)
            worst_rel = max(worst_rel, (err.norm() / w32.norm().clamp(min=1e-12)).item())
        else:
            out[key] = t
    if n_quant == 0:
        raise ConversionError("no linear layers found to quantize")

    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    del tensors, out

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    vp = cfg.get("talker_config", {}).get("vocab_pruning", {})
    cfg["q8_quantization"] = {
        "format": "int8-weight-only",
        "scheme": f"group-wise symmetric (group={group}, scale = groupmax/127, fp32)",
        "group_size": group,
        "layers": n_quant,
        "embeddings": "unchanged",
        "source": src.name,
        "keep_set": vp.get("keep_set"),
    }
    talker = cfg.setdefault("talker_config", {})
    talker["q8"] = {"group_size": group}
    (dst / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # speech tokenizer and tokenizer files pass through verbatim
    if (dst / "speech_tokenizer").exists():
        shutil.rmtree(dst / "speech_tokenizer")
    shutil.copytree(src / "speech_tokenizer", dst / "speech_tokenizer")
    for f in [
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]:
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)
    _LOGGER.info(
        "q8: quantized %d linear layers (group=%d, worst rel-RMS %.3f%%); wrote %s",
        n_quant,
        group,
        100.0 * worst_rel,
        dst,
    )
    return dst


# ---------------------------------------------------------------------------
# variant pipeline
# ---------------------------------------------------------------------------


def converted_dir_name(source_dir: Path, target: str, settings) -> str:
    """Cache directory name encoding the conversion parameters."""
    slug = source_dir.name
    if target == "lite":
        return f"{slug}-lite-{settings.keep_set}-{settings.st_dtype}"
    return f"{slug}-q8-{settings.keep_set}-{settings.st_dtype}-g{settings.q8_group}"


def ensure_variant(source_dir: str | Path, target: str, settings, cache_dir: str | Path) -> Path:
    """Return a directory holding ``source_dir`` in variant ``target``.

    ``target`` of "auto" returns the source unchanged. Otherwise the source is
    converted once into ``cache_dir`` (reused on later runs); sources already
    in the target variant are returned unchanged.
    """
    source_dir = Path(source_dir)
    if target == "auto":
        return source_dir
    info = read_model_info(source_dir)
    if info.variant == target:
        return source_dir
    if not settings.convert:
        raise ConversionError(
            f"variant={target} requested but the source is {info.variant} and "
            "conversion is disabled (convert=no)"
        )
    if not info.convertible:
        raise ConversionError(
            f"cannot convert tts_model_type={info.tts_model_type!r} to {target}: "
            "only custom_voice models are convertible"
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / converted_dir_name(source_dir, target, settings)
    if (dst / "config.json").is_file():
        _LOGGER.info("reusing converted %s at %s", target, dst)
        return dst

    _LOGGER.info("converting %s (%s) to %s -> %s", source_dir.name, info.variant, target, dst)
    if info.variant == "bf16":
        if target == "lite":
            convert_to_lite(source_dir, dst, keep_set=settings.keep_set, st_dtype=settings.st_dtype)
        else:  # q8 from bf16: lite first, then quantize
            lite_dst = cache_dir / converted_dir_name(source_dir, "lite", settings)
            if not (lite_dst / "config.json").is_file():
                convert_to_lite(
                    source_dir, lite_dst, keep_set=settings.keep_set, st_dtype=settings.st_dtype
                )
            else:
                _LOGGER.info("reusing converted lite at %s", lite_dst)
            convert_to_q8(lite_dst, dst, group=settings.q8_group)
    else:  # lite source
        if target == "lite":
            convert_to_lite(source_dir, dst, keep_set=settings.keep_set, st_dtype=settings.st_dtype)
        else:
            convert_to_q8(source_dir, dst, group=settings.q8_group)
    return dst


def directory_size(path: Path) -> float:
    """Total size of regular files under ``path``, in GB (0 when missing)."""
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.glob("**/*") if p.is_file()) / 1e9


# ---------------------------------------------------------------------------
# ASR migration: 4.57-era qwen-asr checkpoints -> Transformers 5 native layout
# ---------------------------------------------------------------------------


def asr_needs_migration(model_dir: Path) -> bool:
    """True when the ASR config uses the pre-Transformers-5 nesting."""
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return "thinker_config" in cfg


def migrate_asr_for_transformers5(src: str | Path, dst: str | Path) -> Path:
    """Materialize a Transformers-5-native copy of a qwen-asr checkpoint.

    The 4.57-era layout nests everything under ``thinker_config`` and prefixes
    weights with ``thinker.``; the Transformers 5 native model expects
    ``audio_config``/``text_config`` at the top level and ``model.*`` weight
    names. This rewrites config + weight keys in place into ``dst``:

      - thinker_config.audio_config -> audio_config   (model_type fixed to
        ``qwen3_asr_encoder``)
      - thinker_config.text_config  -> text_config
      - thinker.model.*      -> model.language_model.*
      - thinker.audio_tower.proj{1,2}.* -> model.multi_modal_projector.linear_{1,2}.*
      - thinker.audio_tower.* -> model.audio_tower.*
      - thinker.lm_head.*     -> lm_head.*
    """
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    thinker = cfg.pop("thinker_config")
    audio = dict(thinker["audio_config"])
    audio["model_type"] = "qwen3_asr_encoder"
    cfg["audio_config"] = audio
    cfg["text_config"] = thinker["text_config"]
    for key in (
        "audio_token_id",
        "audio_start_token_id",
        "audio_end_token_id",
        "initializer_range",
        "tie_word_embeddings",
    ):
        if key in thinker:
            cfg[key] = thinker[key]

    # 4.57-era checkpoints declare WhisperFeatureExtractor, which lacks the
    # mel-axis padding to 2*n_window that the native encoder requires; write
    # a Qwen3ASRFeatureExtractor config instead of copying the stale one
    pre: dict = {}
    pre_src = src / "preprocessor_config.json"
    if pre_src.is_file():
        pre = json.loads(pre_src.read_text(encoding="utf-8"))
    preprocessor_config = {
        "feature_extractor_type": "Qwen3ASRFeatureExtractor",
        "processor_class": "Qwen3ASRProcessor",
        "feature_size": int(pre.get("feature_size") or audio.get("num_mel_bins") or 128),
        "sampling_rate": int(pre.get("sampling_rate") or 16000),
        "hop_length": int(pre.get("hop_length") or 160),
        "n_fft": int(pre.get("n_fft") or 400),
        "chunk_length": int(pre.get("chunk_length") or 30),
        "padding_value": pre.get("padding_value", 0.0),
        "dither": float(pre.get("dither") or 0.0),
        "return_attention_mask": True,
        "n_window": int(audio.get("n_window") or 50),
        "min_length": 8000,
    }

    tensors = load_file(str(src / "model.safetensors"), device="cpu")
    migrated: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key.startswith("thinker.lm_head."):
            migrated["lm_head" + key[len("thinker.lm_head") :]] = value
        elif key.startswith("thinker.model."):
            migrated["model.language_model." + key[len("thinker.model.") :]] = value
        elif key.startswith("thinker.audio_tower.proj1."):
            migrated[
                "model.multi_modal_projector.linear_1." + key.split("thinker.audio_tower.proj1.")[1]
            ] = value
        elif key.startswith("thinker.audio_tower.proj2."):
            migrated[
                "model.multi_modal_projector.linear_2." + key.split("thinker.audio_tower.proj2.")[1]
            ] = value
        elif key.startswith("thinker.audio_tower."):
            migrated["model.audio_tower." + key[len("thinker.audio_tower.") :]] = value
        else:
            migrated[key] = value

    dst.mkdir(parents=True, exist_ok=True)
    save_file(migrated, str(dst / "model.safetensors"), metadata={"format": "pt"})
    (dst / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dst / "preprocessor_config.json").write_text(
        json.dumps(preprocessor_config, indent=2), encoding="utf-8"
    )
    for f in src.glob("*"):
        if f.is_file() and f.name not in (
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ):
            shutil.copy2(f, dst / f.name)
    _LOGGER.info("asr: migrated %s to Transformers-5 layout at %s", src.name, dst)
    return dst
