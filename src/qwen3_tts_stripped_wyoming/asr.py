"""Qwen3-ASR model resolution, loading, and transcription.

Since the faster-qwen3-tts integration moved the environment to Transformers
5, the qwen-asr package (pinned to 4.57) is driven through compatibility
shims for import only; inference runs on the *native* Transformers-5 Qwen3ASR
classes, fed by :func:`convert.migrate_asr_for_transformers5` for checkpoints
in the old layout. The native stack is wrapped to keep the service surface
(languages, bias context, output parsing) identical to the qwen-asr package's.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .audio import asr_language_to_bcp47, bcp47_to_asr_language
from .config import Settings
from .download import download_model, looks_like_repo_id

_LOGGER = logging.getLogger(__name__)

# minimum input length the model handles well (seconds at 16 kHz)
MIN_INPUT_SECONDS = 0.5
TARGET_SAMPLE_RATE = 16_000

# language names shared with the qwen-asr package (keep naming parity)
_SUPPORTED_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)


class TranscriptionServiceError(RuntimeError):
    """Fatal startup or configuration error (message goes to the logs)."""


class LanguageResolutionError(ValueError):
    """Request referenced an unsupported language (message is client-safe)."""


class TranscriptionService:
    """Wraps an ASR model with source resolution and async transcription."""

    def __init__(self, model: Any, settings: Settings, *, languages: tuple[str, ...]) -> None:
        self._model = model
        self._settings = settings
        self._languages = languages

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def languages(self) -> tuple[str, ...]:
        return self._languages

    def languages_bcp47(self) -> list[str]:
        return sorted({asr_language_to_bcp47(lang) for lang in self._languages})

    def resolve_language(self, requested: str | None) -> str | None:
        """Map a BCP-47 request to a Qwen3-ASR language name (None = auto)."""
        candidates: list[str] = []
        if requested is not None and str(requested).strip():
            candidates.append(str(requested).strip())
        if self._settings.asr_language:
            candidates.append(self._settings.asr_language.strip())
        for candidate in candidates:
            mapped = bcp47_to_asr_language(candidate, self._languages)
            if mapped is not None:
                return mapped
            primary = candidate.split("-", 1)[0].lower()
            known_primaries = {
                asr_language_to_bcp47(lang).split("-", 1)[0] for lang in self._languages
            }
            if primary not in known_primaries:
                options = ", ".join(self.languages_bcp47())
                raise LanguageResolutionError(
                    f"Unsupported language {candidate!r}. Supported languages: {options}"
                )
        return None

    async def transcribe(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str | None,
        context: str | None = None,
    ) -> tuple[str, str | None]:
        """Transcribe float32 mono audio; returns (text, detected_language)."""
        min_samples = int(MIN_INPUT_SECONDS * sample_rate)
        if audio.size < min_samples:
            audio = np.concatenate([audio, np.zeros(min_samples - audio.size, dtype=np.float32)])
        ctx = ""
        if self._settings.asr_context and context:
            ctx = str(context)
        results = await asyncio.to_thread(
            self._model.transcribe,
            audio=(audio, sample_rate),
            language=language,
            context=ctx,
        )
        result = results[0]
        return str(result.text), getattr(result, "language", None)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    @classmethod
    async def create(cls, settings: Settings) -> TranscriptionService:
        """Resolve the ASR source, load it, and (optionally) warm it up."""
        source = (settings.asr_model or "").strip()
        if not source:
            raise TranscriptionServiceError("asr model is not configured")
        if looks_like_repo_id(source):
            if settings.download == "never":
                raise TranscriptionServiceError(
                    f"{source} looks like a Hugging Face repo id but downloading "
                    "is disabled (download=never)"
                )
            source_dir = await asyncio.to_thread(
                download_model,
                source,
                settings.model_dir,
                revision=settings.revision,
                force=settings.download == "always",
            )
        else:
            source_dir = Path(source)
            if not (source_dir / "config.json").is_file():
                raise TranscriptionServiceError(
                    f"{source_dir} is neither an existing model directory nor a "
                    "Hugging Face repo id"
                )

        model, languages = await asyncio.to_thread(_load_asr_sync, source_dir, settings)
        service = cls(model, settings, languages=languages)

        if settings.warmup:
            _LOGGER.info("warming up ASR (short silence transcription)")
            silence = np.zeros(int(TARGET_SAMPLE_RATE * MIN_INPUT_SECONDS), dtype=np.float32)
            try:
                await service.transcribe(silence, sample_rate=TARGET_SAMPLE_RATE, language=None)
            except Exception as exc:  # warmup is best-effort
                _LOGGER.warning("ASR warmup failed (continuing anyway): %s", exc)
        return service


def _load_asr_sync(model_dir: Path, settings: Settings) -> tuple[Any, tuple[str, ...]]:
    import torch

    from . import convert
    from .patches import asr as asr_compat  # noqa: F401  (transformers-5 import shim)

    # 4.57-era qwen-asr checkpoints must be migrated to the Transformers-5
    # layout before the native model can load them (runs in this thread; the
    # caller already dispatched us off the event loop)
    if convert.asr_needs_migration(model_dir):
        migrated = model_dir.parent / f"{model_dir.name}-tf5"
        if not (migrated / "config.json").is_file():
            model_dir = convert.migrate_asr_for_transformers5(model_dir, migrated)
        else:
            model_dir = migrated

    from transformers.models.qwen3_asr.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor

    using_cuda = torch.cuda.is_available() and settings.device != "cpu"
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise TranscriptionServiceError(
            "device=cuda but torch.cuda.is_available() is false "
            "(CPU-only torch build or no visible GPU)"
        )
    device = "cuda:0" if using_cuda else "cpu"
    dtype = settings.dtype
    if dtype == "auto":
        dtype = "bfloat16" if using_cuda else "float32"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    _LOGGER.info(
        "loading ASR model from %s (device=%s, dtype=%s, native transformers-5 stack)",
        model_dir,
        device,
        dtype,
    )
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch_dtype, device_map=device
    )
    processor = Qwen3ASRProcessor.from_pretrained(str(model_dir))
    return _NativeAsrWrapper(model, processor, settings.asr_max_new_tokens), _SUPPORTED_LANGUAGES


class _NativeAsrWrapper:
    """Drives the native Transformers-5 Qwen3ASR classes behind the surface
    our TranscriptionService expects (``transcribe(audio=(wav, sr), language,
    context)`` -> results with ``.text``/``.language``), keeping language
    naming and output parsing compatible with the qwen-asr package."""

    def __init__(self, model: Any, processor: Any, max_new_tokens: int | None) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens or 512

    def get_supported_languages(self) -> list[str]:
        return list(_SUPPORTED_LANGUAGES)

    def transcribe(
        self,
        *,
        audio,
        language: str | None = None,
        context: str = "",
        **kwargs: Any,
    ) -> list[Any]:
        import torch

        waveform, sr = audio
        if sr != TARGET_SAMPLE_RATE:
            # native processor expects 16 kHz float arrays
            import numpy as np

            ratio = TARGET_SAMPLE_RATE / sr
            n_out = max(1, int(len(waveform) * ratio))
            x_old = np.linspace(0.0, 1.0, len(waveform), endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            waveform = np.interp(x_new, x_old, waveform).astype(np.float32)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": waveform},
                    {"type": "text", "text": self._build_prompt(language, context)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], audio=[waveform], return_tensors="pt", padding=True
        ).to(self.model.device)
        # cast only floating inputs; integer ids must stay integral
        for key in list(inputs.keys()):
            value = inputs[key]
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        # transformers 5 generate returns a bare tensor; 4.x returned an
        # object with .sequences
        sequences = getattr(out, "sequences", out)
        decoded = self.processor.batch_decode(
            sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results = []
        for raw in decoded:
            lang, txt = self._parse_output(raw, language)
            results.append(SimpleNamespace(text=txt, language=lang))
        return results

    def _build_prompt(self, language: str | None, context: str) -> str:
        parts = []
        if context:
            parts.append(context)
        if language:
            parts.append(f"language {language}")
        return " ".join(parts) if parts else "transcribe the audio"

    def _parse_output(self, raw: str, requested: str | None) -> tuple[str | None, str]:
        text = raw
        lang = requested
        # qwen-asr output convention: "language <Name> <asr_text> ..."
        if raw.startswith("language "):
            rest = raw[len("language ") :].strip()
            head, _, tail = rest.partition(" ")
            candidate = head.strip().lower().capitalize()
            if candidate in _SUPPORTED_LANGUAGES:
                lang = candidate
            # strip the asr_text wrapper the model emits
            tail = tail.strip()
            if tail.startswith("<asr_text>"):
                tail = tail[len("<asr_text>") :]
            if tail.endswith("</asr_text>"):
                tail = tail[: -len("</asr_text>")]
            text = tail.strip()
        return lang, text
