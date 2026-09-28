"""Qwen3-ASR model resolution, loading, and transcription."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .audio import asr_language_to_bcp47, bcp47_to_asr_language
from .config import Settings
from .download import download_model, looks_like_repo_id

_LOGGER = logging.getLogger(__name__)

# minimum input length the qwen-asr package accepts (seconds at 16 kHz)
MIN_INPUT_SECONDS = 0.5
TARGET_SAMPLE_RATE = 16_000


class TranscriptionServiceError(RuntimeError):
    """Fatal startup or configuration error (message goes to the logs)."""


class LanguageResolutionError(ValueError):
    """Request referenced an unsupported language (message is client-safe)."""


class TranscriptionService:
    """Wraps a qwen-asr model with source resolution and async transcription."""

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
        # qwen-asr rejects clips shorter than MIN_INPUT_SECONDS -- pad silence
        min_samples = int(MIN_INPUT_SECONDS * sample_rate)
        if audio.size < min_samples:
            audio = np.concatenate([audio, np.zeros(min_samples - audio.size, dtype=np.float32)])
        # the package resamples (np.ndarray, sr) inputs to 16 kHz itself
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
    from qwen_asr import Qwen3ASRModel

    using_cuda = torch.cuda.is_available() and settings.device != "cpu"
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise TranscriptionServiceError(
            "device=cuda but torch.cuda.is_available() is false "
            "(CPU-only torch build or no visible GPU)"
        )
    device_map = "cuda:0" if using_cuda else "cpu"
    dtype = settings.dtype
    if dtype == "auto":
        dtype = "bfloat16" if using_cuda else "float32"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]
    kwargs: dict[str, Any] = {"dtype": torch_dtype, "device_map": device_map}
    if settings.asr_max_new_tokens is not None:
        kwargs["max_new_tokens"] = settings.asr_max_new_tokens
    _LOGGER.info("loading ASR model from %s (device=%s, dtype=%s)", model_dir, device_map, dtype)
    model = Qwen3ASRModel.from_pretrained(str(model_dir), **kwargs)
    try:
        languages = tuple(model.get_supported_languages())
    except Exception:
        languages = ()
    return model, languages
