"""Model resolution, loading, and synthesis around the qwen-tts runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import convert
from .audio import match_model_language, model_language_to_bcp47
from .config import Settings
from .download import download_model, looks_like_repo_id

_LOGGER = logging.getLogger(__name__)

_SAMPLE_RATE = 24_000
_SENTINEL = object()


class SynthesisServiceError(RuntimeError):
    """Fatal startup or configuration error (message goes to the logs)."""


class VoiceResolutionError(ValueError):
    """Request referenced an unknown voice or language (message is client-safe)."""


@dataclass(frozen=True)
class Speaker:
    """One advertised voice."""

    id: str
    languages: tuple[str, ...]  # model language names


class SynthesisService:
    """Wraps a TTS model (fast or stock backend) with variant detection,
    conversion, and async synthesis/streaming."""

    def __init__(
        self,
        model: Any,
        settings: Settings,
        *,
        variant: str,
        model_dir: Path,
        speakers: tuple[Speaker, ...],
        languages: tuple[str, ...],
        using_cuda: bool,
        backend: str = "stock",
    ) -> None:
        self._model = model
        self._settings = settings
        self._variant = variant
        self._model_dir = Path(model_dir)
        self._speakers = speakers
        self._languages = languages
        self._using_cuda = using_cuda
        self._backend = backend

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Introspection (used for the wyoming info message and logging)
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def using_cuda(self) -> bool:
        return self._using_cuda

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    def speakers(self) -> tuple[Speaker, ...]:
        return self._speakers

    def speaker_languages_bcp47(self, speaker: Speaker) -> list[str]:
        return sorted({model_language_to_bcp47(lang) for lang in speaker.languages})

    # ------------------------------------------------------------------
    # Voice/language resolution
    # ------------------------------------------------------------------
    def default_speaker(self) -> Speaker:
        if not self._speakers:
            raise SynthesisServiceError("model config exposes no speakers (spk_id)")
        if self._settings.default_voice:
            wanted = self._settings.default_voice.strip().lower().replace("-", "_")
            for speaker in self._speakers:
                if speaker.id.lower().replace("-", "_") == wanted:
                    return speaker
            options = ", ".join(s.id for s in self._speakers)
            raise SynthesisServiceError(f"invalid default voice {wanted!r}; options: {options}")
        return self._speakers[0]

    def resolve_speaker(self, name: str | None) -> Speaker:
        if name is None or not str(name).strip():
            return self.default_speaker()
        wanted = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        for speaker in self._speakers:
            if speaker.id.lower().replace("-", "_") == wanted:
                return speaker
        options = ", ".join(s.id for s in self._speakers)
        raise VoiceResolutionError(f"Unknown voice {name!r}. Available voices: {options}")

    def resolve_language(self, language: str | None) -> str | None:
        """Resolve a requested BCP-47 language to a model language name.

        Returns None (qwen-tts "Auto") when the request and settings carry
        nothing usable -- the model then auto-detects per text.
        """
        available = list(self._languages) or [
            lang for speaker in self._speakers for lang in speaker.languages
        ]
        if not available:
            return None
        candidates: list[str] = []
        if language is not None and str(language).strip():
            candidates.append(str(language).strip())
        if self._settings.default_language:
            candidates.append(self._settings.default_language.strip())
        for candidate in candidates:
            mapped = match_model_language(available, candidate)
            if mapped in available:
                return mapped
        primary = match_model_language(available, candidates[0]) if candidates else ""
        if candidates and primary not in available:
            options = ", ".join(sorted(set(available)))
            raise VoiceResolutionError(
                f"Unsupported language {candidates[0]!r}. Available languages: {options}"
            )
        return None

    def resolve_speaker_for_request(self, name: str | None, language: str | None) -> Speaker:
        """Explicit name wins; otherwise a speaker covering the requested language."""
        if name is not None and str(name).strip():
            return self.resolve_speaker(name)
        if language is not None and str(language).strip():
            target = match_model_language(
                (lang for s in self._speakers for lang in s.languages), str(language).strip()
            )
            for speaker in self._speakers:
                if target in speaker.languages:
                    return speaker
        return self.default_speaker()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def _sampling_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        s = self._settings
        if s.temperature is not None:
            kwargs["temperature"] = s.temperature
        if s.top_k is not None:
            kwargs["top_k"] = s.top_k
        if s.top_p is not None:
            kwargs["top_p"] = s.top_p
        if s.repetition_penalty is not None:
            kwargs["repetition_penalty"] = s.repetition_penalty
        if s.max_new_tokens is not None:
            kwargs["max_new_tokens"] = s.max_new_tokens
        return kwargs

    async def synthesize(self, *, text: str, speaker_id: str, language: str | None) -> np.ndarray:
        """Synthesize ``text`` into float32 mono audio at 24 kHz (full clip)."""
        kwargs = self._sampling_kwargs()
        if language is not None:
            kwargs["language"] = language
        instruct = self._settings.instruct or None
        if self._settings.seed is not None:
            import torch

            torch.manual_seed(self._settings.seed)
        if self._backend == "fast":
            chunks: list[np.ndarray] = []
            gen = self._model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker_id,
                language=language or "Auto",
                instruct=instruct,
                chunk_size=self._settings.stream_chunk_steps,
                **kwargs,
            )
            for chunk, sr, _timing in await _drain_generator(gen):
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                if sr and int(sr) != _SAMPLE_RATE:
                    raise SynthesisServiceError(f"unexpected sample rate {sr}")
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        wavs, sr = await asyncio.to_thread(
            self._model.generate_custom_voice,
            text=text,
            speaker=speaker_id,
            instruct=instruct,
            **kwargs,
        )
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if sr and int(sr) != _SAMPLE_RATE:
            raise SynthesisServiceError(f"unexpected sample rate {sr}")
        return audio

    async def stream(
        self, *, text: str, speaker_id: str, language: str | None
    ) -> AsyncIterator[np.ndarray]:
        """Yield synthesized audio pieces as they become available.

        The fast backend streams real chunks during generation; the stock
        backend yields the complete clip once (matching the old behaviour).
        """
        if self._backend != "fast":
            yield await self.synthesize(text=text, speaker_id=speaker_id, language=language)
            return
        kwargs = self._sampling_kwargs()
        instruct = self._settings.instruct or None
        if self._settings.seed is not None:
            import torch

            torch.manual_seed(self._settings.seed)
        gen = self._model.generate_custom_voice_streaming(
            text=text,
            speaker=speaker_id,
            language=language or "Auto",
            instruct=instruct,
            chunk_size=self._settings.stream_chunk_steps,
            **kwargs,
        )
        loop = asyncio.get_running_loop()
        iterator = iter(gen)
        while True:
            item = await loop.run_in_executor(None, partial(next, iterator, _SENTINEL))
            if item is _SENTINEL:
                break
            chunk, sr, _timing = item
            if sr and int(sr) != _SAMPLE_RATE:
                raise SynthesisServiceError(f"unexpected sample rate {sr}")
            array = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if array.size:
                yield array

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    @classmethod
    async def create(cls, settings: Settings) -> SynthesisService:
        """Resolve the model source, convert if needed, load, and warm up."""
        source = settings.model.strip()
        model_dir = Path(settings.model_dir)

        # 1. resolve the source to a local directory
        if looks_like_repo_id(source):
            if settings.download == "never":
                raise SynthesisServiceError(
                    f"{source} looks like a Hugging Face repo id but downloading is "
                    "disabled (download=never)"
                )
            source_dir = await asyncio.to_thread(
                download_model,
                source,
                model_dir,
                revision=settings.revision,
                force=settings.download == "always",
            )
        else:
            source_dir = Path(source)
            if not (source_dir / "config.json").is_file():
                raise SynthesisServiceError(
                    f"{source_dir} is neither an existing model directory nor a "
                    "Hugging Face repo id"
                )

        # 2. pick / materialize the serving variant
        info = convert.read_model_info(source_dir)
        _LOGGER.info(
            "model %s: tts_model_type=%s, detected variant=%s, speakers=%s",
            source_dir,
            info.tts_model_type,
            info.variant,
            ",".join(info.speakers) or "-",
        )
        if info.tts_model_type != "custom_voice":
            raise SynthesisServiceError(
                f"this server serves custom_voice models; {source_dir} is "
                f"tts_model_type={info.tts_model_type!r}"
            )
        target = settings.variant
        if target != "auto" and info.variant != target:
            source_dir = await asyncio.to_thread(
                convert.ensure_variant,
                source_dir,
                target,
                settings,
                model_dir / "converted",
            )
            info = convert.read_model_info(source_dir)

        variant = info.variant
        size_gb = convert.directory_size(source_dir)
        _LOGGER.info("serving %s (variant %s, %.2f GB)", source_dir, variant, size_gb)

        # 3. load (patches first: marker-driven, one variant per process)
        backend = settings.tts_backend
        if backend == "fast":
            import torch

            if not torch.cuda.is_available() and settings.device == "auto":
                _LOGGER.warning(
                    "fast TTS backend requires CUDA and none is visible; "
                    "falling back to the stock backend"
                )
                backend = "stock"
            elif settings.device == "cpu":
                raise SynthesisServiceError(
                    "the fast TTS backend requires CUDA (static-cache CUDA graphs); "
                    "use --tts-backend stock for CPU inference"
                )
        if backend == "fast":
            model, using_cuda = await asyncio.to_thread(
                _load_fast_sync, source_dir, settings, variant
            )
        else:
            model, using_cuda = await asyncio.to_thread(
                _load_model_sync, source_dir, settings, variant
            )

        speakers = tuple(
            Speaker(
                id=name,
                languages=info.languages or _fallback_languages(),
            )
            for name in info.speakers
        )
        service = cls(
            model,
            settings,
            variant=variant,
            model_dir=source_dir,
            speakers=speakers,
            languages=info.languages or (),
            using_cuda=using_cuda,
            backend=backend,
        )

        if settings.warmup:
            if service.backend == "fast":
                # graph capture + lazy prep, no synthesis needed
                _LOGGER.info("warming up fast backend (CUDA graph capture)")
                try:
                    await asyncio.to_thread(service._model.warmup, 64)
                except Exception:
                    _LOGGER.exception(
                        "fast backend warmup failed; continuing (first request will lazy-prepare)"
                    )
            else:
                _LOGGER.info("warming up (one short synthesis)")
                try:
                    await service.synthesize(
                        text="Ready.",
                        speaker_id=service.default_speaker().id,
                        language=service.resolve_language(None),
                    )
                except Exception:
                    if not settings.compile:
                        raise
                    # a compile-mode/warmup failure must not take the server down:
                    # restore the eager modules and try once more
                    _LOGGER.exception(
                        "compiled warmup failed; reverting torch.compile and "
                        "retrying eager (set QWEN3TTS_COMPILE=0 to skip this, or "
                        "try QWEN3TTS_COMPILE_MODE=default)"
                    )
                    restored = await asyncio.to_thread(revert_torch_compile, service._model.model)
                    _LOGGER.warning("torch.compile reverted on %d modules; running eager", restored)
                    await service.synthesize(
                        text="Ready.",
                        speaker_id=service.default_speaker().id,
                        language=service.resolve_language(None),
                    )
        return service


async def _drain_generator(gen: Any) -> list[Any]:
    """Pull a sync generator to completion in the default executor."""
    loop = asyncio.get_running_loop()
    iterator = iter(gen)
    items: list[Any] = []
    while True:
        item = await loop.run_in_executor(None, partial(next, iterator, _SENTINEL))
        if item is _SENTINEL:
            return items
        items.append(item)


def _load_fast_sync(model_dir: Path, settings: Settings, variant: str) -> tuple[Any, bool]:
    """Load via faster-qwen3-tts (static KV cache + CUDA graphs + streaming).

    The lite/q8 patches must be applied BEFORE FasterQwen3TTS constructs the
    model: they are class-level __init__ monkeypatches on the qwen_tts model
    classes, which FasterQwen3TTS builds on (validated for all variants --
    the Q8Linear dequant and token-map lookup capture fine inside CUDA graphs).
    """

    from .patches import apply_for_variant, compat  # noqa: F401  (rope_theta shim)

    apply_for_variant(variant)

    from faster_qwen3_tts import FasterQwen3TTS

    dtype = settings.dtype
    if dtype == "auto":
        dtype = "bfloat16"
    model = FasterQwen3TTS.from_pretrained(str(model_dir), device="cuda", dtype=dtype)
    return model, True


def _fallback_languages() -> tuple[str, ...]:
    """Languages to advertise when the config lists none (all CustomVoice)."""
    return (
        "chinese",
        "english",
        "german",
        "italian",
        "portuguese",
        "spanish",
        "japanese",
        "korean",
        "french",
        "russian",
    )


def apply_torch_compile(tts_model: Any, *, mode: str, dynamic: bool) -> list[str]:
    """torch.compile the per-step decoder stacks of a loaded TTS model.

    Compiling the top-level model would be a no-op: ``generate()`` resolves
    bound methods on the original module, bypassing an OptimizedModule
    wrapper. Instead the inner submodules that the parent forward invokes via
    ``self.model(...)`` are replaced, so those calls dispatch into compiled
    regions. Compiling also fuses Q8Linear's int8 dequant into the GEMMs.

    Returns the compiled target paths.
    """
    import torch

    talker = tts_model.talker
    candidates: list[tuple[str, Any, str]] = [("talker.model", talker, "model")]
    predictor = getattr(talker, "code_predictor", None)
    if predictor is not None:
        candidates.append(("talker.code_predictor.model", predictor, "model"))
    compiled = []
    for path, parent, name in candidates:
        module = getattr(parent, name, None)
        if module is None:
            continue
        setattr(parent, name, torch.compile(module, mode=mode, dynamic=dynamic))
        compiled.append(path)
    return compiled


def revert_torch_compile(tts_model: Any) -> int:
    """Undo :func:`apply_torch_compile` by restoring the original modules.

    Used when a compiled warmup fails: the server falls back to eager instead
    of dying, so a bad compile configuration degrades instead of taking the
    service down. Returns the number of restored modules.
    """
    import torch._dynamo

    talker = tts_model.talker
    parents = [talker]
    predictor = getattr(talker, "code_predictor", None)
    if predictor is not None:
        parents.append(predictor)
    restored = 0
    for parent in parents:
        current = getattr(parent, "model", None)
        original = getattr(current, "_orig_mod", None)  # OptimizedModule
        if original is not None:
            parent.model = original
            restored += 1
    torch._dynamo.reset()
    return restored


def _load_model_sync(model_dir: Path, settings: Settings, variant: str) -> tuple[Any, bool]:
    import torch

    from .patches import apply_for_variant, compat  # noqa: F401  (rope_theta shim)

    apply_for_variant(variant)

    from qwen_tts import Qwen3TTSModel

    using_cuda = torch.cuda.is_available()
    if settings.device == "cuda" and not using_cuda:
        raise SynthesisServiceError(
            "device=cuda but torch.cuda.is_available() is false "
            "(CPU-only torch build or no visible GPU)"
        )
    if settings.device == "cpu":
        using_cuda = False
    device_map = "cuda:0" if using_cuda else "cpu"
    dtype = settings.dtype
    if dtype == "auto":
        dtype = "bfloat16" if using_cuda else "float32"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]
    _LOGGER.info(
        "loading model from %s (device=%s, dtype=%s)",
        model_dir,
        device_map,
        dtype,
    )
    model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map=device_map, dtype=torch_dtype)
    if settings.compile:
        missing = _missing_compile_toolchain(using_cuda)
        if missing:
            _LOGGER.error(
                "QWEN3TTS_COMPILE=1 but the Inductor toolchain is incomplete "
                "(missing %s); falling back to eager. The Docker image ships "
                "gcc+g++; for venv installs add a C/C++ compiler to PATH.",
                ", ".join(missing),
            )
        else:
            import torch._dynamo

            # a failing region logs and falls back to eager instead of killing
            # the server; the whole feature is experimental
            torch._dynamo.config.suppress_errors = True
            compiled = apply_torch_compile(
                model.model, mode=settings.compile_mode, dynamic=settings.compile_dynamic
            )
            _LOGGER.info(
                "torch.compile enabled (mode=%s, dynamic=%s) on %s; the first "
                "inference triggers compilation -- expect minutes, cached under "
                "TORCHINDUCTOR_CACHE_DIR for subsequent starts",
                settings.compile_mode,
                settings.compile_dynamic,
                ", ".join(compiled),
            )
    return model, using_cuda


def _missing_compile_toolchain(using_cuda: bool) -> list[str]:
    """Compilers Inductor/Triton need at runtime, by device.

    CUDA compilation goes through Triton (C compiler); CPU compilation builds
    C++ wrappers (C++ compiler). Missing tools disable compile up front --
    otherwise every synthesis request would fail at first inference.
    """
    import shutil

    missing = []
    if using_cuda and not (shutil.which("gcc") or shutil.which("cc")):
        missing.append("gcc (Triton kernel compilation)")
    if not using_cuda and not (shutil.which("g++") or shutil.which("c++")):
        missing.append("g++ (Inductor CPU wrappers)")
    return missing
