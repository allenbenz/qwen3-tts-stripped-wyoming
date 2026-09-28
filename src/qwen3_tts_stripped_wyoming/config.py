"""Server configuration: environment variables, CLI flags, and validation.

Every knob can be set through a ``QWEN3TTS_*`` environment variable (the
Docker interface) or a CLI flag (flags win over the environment). Values the
Wyoming protocol controls per request -- text, voice name, voice language --
are deliberately *not* configurable here beyond server-side defaults.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "QWEN3TTS_"

VALID_VARIANTS = ("auto", "bf16", "lite", "q8")
VALID_KEEP_SETS = ("latin", "ml")
VALID_ST_DTYPES = ("float16", "bfloat16", "float32")
VALID_DOWNLOAD_MODES = ("auto", "always", "never")
VALID_DEVICES = ("auto", "cuda", "cpu")
VALID_DTYPES = ("auto", "bfloat16", "float16", "float32")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VALID_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"

MIN_OUTPUT_CHUNK_MS = 20
MAX_Q8_GROUP = 256


@dataclass(frozen=True)
class Settings:
    """Validated server settings."""

    uri: str = "tcp://0.0.0.0:10200"
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    model_dir: Path = Path("data/models")
    variant: str = "auto"
    convert: bool = True
    keep_set: str = "latin"
    st_dtype: str = "float16"
    q8_group: int = 64
    download: str = "auto"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    default_voice: str | None = None
    default_language: str | None = None
    instruct: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    seed: int | None = None
    output_chunk_ms: int = 200
    energy_gain: float = 1.0
    warmup: bool = True
    asr_model: str | None = DEFAULT_ASR_MODEL
    asr_language: str | None = None
    asr_max_new_tokens: int | None = None
    asr_context: bool = True
    compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("uri must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {self.variant!r}")
        if self.keep_set not in VALID_KEEP_SETS:
            raise ValueError(f"keep_set must be one of {VALID_KEEP_SETS}, got {self.keep_set!r}")
        if self.st_dtype not in VALID_ST_DTYPES:
            raise ValueError(f"st_dtype must be one of {VALID_ST_DTYPES}, got {self.st_dtype!r}")
        if not 1 <= self.q8_group <= MAX_Q8_GROUP:
            raise ValueError(f"q8_group must be between 1 and {MAX_Q8_GROUP}, got {self.q8_group}")
        if self.download not in VALID_DOWNLOAD_MODES:
            raise ValueError(
                f"download must be one of {VALID_DOWNLOAD_MODES}, got {self.download!r}"
            )
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}, got {self.device!r}")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"dtype must be one of {VALID_DTYPES}, got {self.dtype!r}")
        for name in ("temperature", "top_p", "repetition_penalty"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.energy_gain < 0:
            raise ValueError(f"energy_gain must be >= 0, got {self.energy_gain}")
        if self.asr_max_new_tokens is not None and self.asr_max_new_tokens <= 0:
            raise ValueError(f"asr_max_new_tokens must be > 0, got {self.asr_max_new_tokens}")
        if self.compile_mode not in VALID_COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {VALID_COMPILE_MODES}, got {self.compile_mode!r}"
            )
        if self.output_chunk_ms < MIN_OUTPUT_CHUNK_MS:
            raise ValueError(
                f"output_chunk_ms must be >= {MIN_OUTPUT_CHUNK_MS}, got {self.output_chunk_ms}"
            )
        if self.log_level not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}, got {self.log_level!r}")


def _raw(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    value = _raw(env, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {value!r}") from None


def _env_float(env: Mapping[str, str], name: str) -> float | None:
    value = _raw(env, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {value!r}") from None


def _env_bool(env: Mapping[str, str], name: str) -> bool | None:
    value = _raw(env, name)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{ENV_PREFIX}{name} must be a boolean, got {value!r}")


def settings_from_env(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from environment variables (defaults for the rest)."""
    source = os.environ if env is None else env
    kwargs: dict[str, Any] = {}

    if text := _raw(source, "URI"):
        kwargs["uri"] = text
    if text := _raw(source, "MODEL"):
        kwargs["model"] = text
    if text := _raw(source, "MODEL_DIR"):
        kwargs["model_dir"] = Path(text)
    if text := _raw(source, "VARIANT"):
        kwargs["variant"] = text.lower()
    if (flag := _env_bool(source, "CONVERT")) is not None:
        kwargs["convert"] = flag
    if text := _raw(source, "KEEP_SET"):
        kwargs["keep_set"] = text.lower()
    if text := _raw(source, "ST_DTYPE"):
        kwargs["st_dtype"] = text.lower()
    if (count := _env_int(source, "Q8_GROUP")) is not None:
        kwargs["q8_group"] = count
    if text := _raw(source, "DOWNLOAD"):
        kwargs["download"] = text.lower()
    kwargs["revision"] = _raw(source, "REVISION")
    if text := _raw(source, "DEVICE"):
        kwargs["device"] = text.lower()
    if text := _raw(source, "DTYPE"):
        kwargs["dtype"] = text.lower()
    kwargs["default_voice"] = _raw(source, "VOICE")
    kwargs["default_language"] = _raw(source, "LANGUAGE")
    kwargs["instruct"] = _raw(source, "INSTRUCT")
    for name in ("TEMPERATURE", "TOP_P", "REPETITION_PENALTY"):
        if (value := _env_float(source, name)) is not None:
            kwargs[name.lower()] = value
    if (count := _env_int(source, "TOP_K")) is not None:
        kwargs["top_k"] = count
    if (count := _env_int(source, "MAX_NEW_TOKENS")) is not None:
        kwargs["max_new_tokens"] = count
    kwargs["seed"] = _env_int(source, "SEED")
    if (count := _env_int(source, "OUTPUT_CHUNK_MS")) is not None:
        kwargs["output_chunk_ms"] = count
    if (factor := _env_float(source, "ENERGY_GAIN")) is not None:
        kwargs["energy_gain"] = factor
    if (flag := _env_bool(source, "WARMUP")) is not None:
        kwargs["warmup"] = flag
    # empty string disables the ASR side; absent keeps the default model
    # NB: an explicitly-empty (or whitespace) value disables STT, unlike the
    # other string settings where empty means "unset"
    if ENV_PREFIX + "ASR_MODEL" in source:
        kwargs["asr_model"] = _raw(source, "ASR_MODEL") or None
    kwargs["asr_language"] = _raw(source, "ASR_LANGUAGE")
    if (count := _env_int(source, "ASR_MAX_NEW_TOKENS")) is not None:
        kwargs["asr_max_new_tokens"] = count
    if (flag := _env_bool(source, "ASR_CONTEXT")) is not None:
        kwargs["asr_context"] = flag
    if (flag := _env_bool(source, "COMPILE")) is not None:
        kwargs["compile"] = flag
    if text := _raw(source, "COMPILE_MODE"):
        kwargs["compile_mode"] = text.lower()
    if (flag := _env_bool(source, "COMPILE_DYNAMIC")) is not None:
        kwargs["compile_dynamic"] = flag
    if text := _raw(source, "LOG_LEVEL"):
        kwargs["log_level"] = text.upper()

    return Settings(**kwargs)


def build_arg_parser(env: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    """Argument parser whose defaults come from the environment."""
    base = settings_from_env(env)
    parser = argparse.ArgumentParser(
        prog="qwen3-tts-stripped-wyoming",
        description="Wyoming-protocol TTS server for Qwen3-TTS CustomVoice models",
    )
    parser.add_argument(
        "--uri",
        default=base.uri,
        metavar="URI",
        help="Wyoming listen URI: tcp://HOST:PORT, unix://PATH or stdio:// (env QWEN3TTS_URI)",
    )
    parser.add_argument(
        "--model",
        default=base.model,
        metavar="SRC",
        help=(
            "model source: a Hugging Face repo id (downloaded to --model-dir) or a "
            "local model directory in any variant (env QWEN3TTS_MODEL)"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=base.model_dir,
        metavar="DIR",
        help="cache for downloads and converted variants (env QWEN3TTS_MODEL_DIR)",
    )
    parser.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=base.variant,
        help=(
            "serve this variant: auto detects from the model dir, bf16/lite/q8 "
            "converts the source once when it is not already that variant "
            "(env QWEN3TTS_VARIANT)"
        ),
    )
    parser.add_argument(
        "--convert",
        action=argparse.BooleanOptionalAction,
        default=base.convert,
        help=(
            "allow one-time conversion of non-local sources into the requested "
            "variant (env QWEN3TTS_CONVERT)"
        ),
    )
    parser.add_argument(
        "--keep-set",
        choices=VALID_KEEP_SETS,
        default=base.keep_set,
        help=(
            "vocabulary keep-set for lite/q8 conversion: latin guarantees coverage "
            "for Latin-script languages, ml for all supported languages "
            "(env QWEN3TTS_KEEP_SET)"
        ),
    )
    parser.add_argument(
        "--st-dtype",
        choices=VALID_ST_DTYPES,
        default=base.st_dtype,
        help="storage dtype for the speech-tokenizer decoder when converting "
        "(env QWEN3TTS_ST_DTYPE)",
    )
    parser.add_argument(
        "--q8-group",
        type=int,
        default=base.q8_group,
        metavar="N",
        help=f"inputs per int8 scale group for q8 conversion, 1-{MAX_Q8_GROUP} "
        "(env QWEN3TTS_Q8_GROUP)",
    )
    parser.add_argument(
        "--download",
        choices=VALID_DOWNLOAD_MODES,
        default=base.download,
        help="download behaviour for HF sources: auto when missing, always at "
        "startup, never (env QWEN3TTS_DOWNLOAD)",
    )
    parser.add_argument(
        "--revision",
        default=base.revision,
        metavar="REV",
        help="Hugging Face revision to download (env QWEN3TTS_REVISION)",
    )
    parser.add_argument(
        "--device",
        choices=VALID_DEVICES,
        default=base.device,
        help="execution device: auto prefers CUDA and falls back to CPU (env QWEN3TTS_DEVICE)",
    )
    parser.add_argument(
        "--dtype",
        choices=VALID_DTYPES,
        default=base.dtype,
        help="model dtype: auto picks bfloat16 on CUDA and float32 on CPU (env QWEN3TTS_DTYPE)",
    )
    parser.add_argument(
        "--voice",
        dest="default_voice",
        default=base.default_voice,
        metavar="ID",
        help="default speaker id for requests without one (env QWEN3TTS_VOICE)",
    )
    parser.add_argument(
        "--language",
        dest="default_language",
        default=base.default_language,
        metavar="LANG",
        help="default language for requests without one (env QWEN3TTS_LANGUAGE)",
    )
    parser.add_argument(
        "--instruct",
        default=base.instruct,
        metavar="TEXT",
        help="default style instruction applied to every request, e.g. "
        '"warm and gentle" (env QWEN3TTS_INSTRUCT)',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=base.temperature,
        metavar="X",
        help="sampling temperature override (default: model generation config; "
        "env QWEN3TTS_TEMPERATURE)",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=base.top_k,
        metavar="N",
        help="top-k sampling override (env QWEN3TTS_TOP_K)",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=base.top_p,
        metavar="X",
        help="top-p sampling override (env QWEN3TTS_TOP_P)",
    )
    parser.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=base.repetition_penalty,
        metavar="X",
        help="repetition penalty override (env QWEN3TTS_REPETITION_PENALTY)",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=base.max_new_tokens,
        metavar="N",
        help="generation cap override (env QWEN3TTS_MAX_NEW_TOKENS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=base.seed,
        metavar="N",
        help="fixed sampling seed for reproducible output (env QWEN3TTS_SEED)",
    )
    parser.add_argument(
        "--output-chunk-ms",
        type=int,
        default=base.output_chunk_ms,
        metavar="MS",
        help="max milliseconds of audio per audio-chunk event (env QWEN3TTS_OUTPUT_CHUNK_MS)",
    )
    parser.add_argument(
        "--energy-gain",
        type=float,
        default=base.energy_gain,
        metavar="X",
        help="post-synthesis waveform gain, linear (env QWEN3TTS_ENERGY_GAIN)",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=base.warmup,
        help="run a warmup synthesis at startup so the first request is fast (env QWEN3TTS_WARMUP)",
    )
    parser.add_argument(
        "--asr-model",
        dest="asr_model",
        default=base.asr_model,
        metavar="SRC",
        help=(
            "STT model source (HF repo id or local dir); set an empty value to "
            f"disable speech-to-text (default: {DEFAULT_ASR_MODEL}; env QWEN3TTS_ASR_MODEL)"
        ),
    )
    parser.add_argument(
        "--asr-language",
        dest="asr_language",
        default=base.asr_language,
        metavar="LANG",
        help="default transcription language (BCP-47); unset = auto-detect "
        "(env QWEN3TTS_ASR_LANGUAGE)",
    )
    parser.add_argument(
        "--asr-max-new-tokens",
        dest="asr_max_new_tokens",
        type=int,
        default=base.asr_max_new_tokens,
        metavar="N",
        help="generation cap for transcription; raise for long audio "
        "(env QWEN3TTS_ASR_MAX_NEW_TOKENS)",
    )
    parser.add_argument(
        "--asr-context",
        action=argparse.BooleanOptionalAction,
        default=base.asr_context,
        help="forward the Wyoming transcribe context (names/terms) to the model "
        "(env QWEN3TTS_ASR_CONTEXT)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=base.compile,
        help=(
            "torch.compile the per-step decoder stacks (talker + code predictor) "
            "with Inductor/Triton. First inference compiles -- minutes, cached "
            "under TORCHINDUCTOR_CACHE_DIR afterwards. Needs a C compiler at "
            "runtime (the Docker image ships gcc; venv users need one in PATH) "
            "(env QWEN3TTS_COMPILE)"
        ),
    )
    parser.add_argument(
        "--compile-mode",
        dest="compile_mode",
        choices=VALID_COMPILE_MODES,
        default=base.compile_mode,
        help=(
            "torch.compile mode: default (fusion only, safest), "
            "reduce-overhead (adds CUDA graphs; may not engage with the growing "
            "KV cache), max-autotune (longest compile) "
            "(env QWEN3TTS_COMPILE_MODE)"
        ),
    )
    parser.add_argument(
        "--compile-dynamic",
        dest="compile_dynamic",
        action=argparse.BooleanOptionalAction,
        default=base.compile_dynamic,
        help=(
            "compile with dynamic shapes so the growing decode sequence does "
            "not trigger recompilation per length (env QWEN3TTS_COMPILE_DYNAMIC)"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=VALID_LOG_LEVELS,
        default=base.log_level,
        help="logging verbosity (env QWEN3TTS_LOG_LEVEL)",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build settings from a parsed namespace (whose defaults came from env)."""
    names = {field.name for field in fields(Settings)}
    kwargs = {name: getattr(args, name) for name in names if hasattr(args, name)}
    return Settings(**kwargs)
