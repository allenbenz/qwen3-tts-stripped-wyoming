"""Audio helpers: PCM conversion, chunking, and language-code mapping."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

SAMPLE_WIDTH = 2  # bytes per sample (16-bit PCM)
CHANNELS = 1  # mono

# Qwen3-TTS language ids (codec_language_id keys) -> BCP-47 for Home Assistant.
# Dialects map to their base language so HA's language matching finds the voice;
# requests for the base language select the voice, which then uses its dialect.
LANGUAGE_TO_BCP47 = {
    "chinese": "zh",
    "english": "en",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "spanish": "es",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "russian": "ru",
    "beijing_dialect": "zh",
    "sichuan_dialect": "zh",
}

# Qwen3-ASR language names (SUPPORTED_LANGUAGES in the qwen-asr package) ->
# BCP-47 for Home Assistant.
ASR_LANGUAGE_TO_BCP47 = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "fil",
    "Persian": "fa",
    "Greek": "el",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Macedonian": "mk",
}


def float_to_int16_bytes(audio: np.ndarray, gain: float = 1.0) -> bytes:
    """Convert float32 [-1, 1] samples to little-endian 16-bit PCM bytes.

    An optional linear ``gain`` is applied before clipping.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if gain != 1.0:
        samples = samples * np.float32(gain)
    clipped = np.clip(samples, np.float32(-1.0), np.float32(1.0))
    return (clipped * np.float32(32767.0)).astype("<i2").tobytes()


def split_bytes(data: bytes, max_bytes: int) -> Iterator[bytes]:
    """Yield ``data`` in pieces of at most ``max_bytes`` bytes (never empty)."""
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
    for offset in range(0, len(data), max_bytes):
        yield data[offset : offset + max_bytes]


def chunk_bytes_for_ms(
    sample_rate: int,
    chunk_ms: int,
    *,
    width: int = SAMPLE_WIDTH,
    channels: int = CHANNELS,
) -> int:
    """Byte size of one output chunk holding at most ``chunk_ms`` of audio."""
    samples = max(1, round(sample_rate * chunk_ms / 1000))
    return samples * width * channels


def model_language_to_bcp47(code: str) -> str:
    """Map a model language name (``english``) to BCP-47 (``en``).

    Unknown names fall back to the raw code lowercased with ``_`` -> ``-``.
    """
    code = code.strip().lower()
    if code in LANGUAGE_TO_BCP47:
        return LANGUAGE_TO_BCP47[code]
    return code.replace("_", "-")


def bcp47_to_model_language(code: str) -> str:
    """Map a BCP-47 code (``en``/``en-US``/``zh-CN``) to the model's language name."""
    normalized = code.strip().lower().replace("-", "_")
    for name, bcp in LANGUAGE_TO_BCP47.items():
        if bcp.replace("-", "_") == normalized or name == normalized:
            return name
    primary = normalized.split("_", 1)[0]
    for name, bcp in LANGUAGE_TO_BCP47.items():
        if bcp.replace("-", "_").split("_", 1)[0] == primary:
            return name
    return normalized


def match_model_language(available, requested: str) -> str:
    """Best-match a requested BCP-47 code against available model language names.

    Tries the full mapping (``en-US`` -> ``english``), then the primary subtag
    (``de-DE`` -> ``german``). Returns the mapped name unchanged when nothing
    matches so callers can report it.
    """
    available = tuple(available)
    mapped = bcp47_to_model_language(requested)
    if mapped in available:
        return mapped
    primary = mapped.split("_", 1)[0]
    for name in available:
        if name.split("_", 1)[0] == primary:
            return name
    return mapped


def asr_language_to_bcp47(name: str) -> str:
    """Map a Qwen3-ASR language name (``English``) to BCP-47 (``en``)."""
    return ASR_LANGUAGE_TO_BCP47.get(name.strip(), name.strip().lower())


def bcp47_to_asr_language(code: str, available) -> str | None:
    """Best-match a BCP-47 code (``en-US``) against available ASR language names.

    Returns the Qwen3-ASR language name, or None when nothing matches (the
    model then auto-detects).
    """
    available = tuple(available)
    lowered = code.strip().lower().replace("-", "_")
    for name in available:
        if name.lower() == lowered:
            return name
    primary = lowered.split("_", 1)[0]
    for name, bcp in ASR_LANGUAGE_TO_BCP47.items():
        if bcp.replace("-", "_").split("_", 1)[0] == primary and name in available:
            return name
    return None


def pcm_bytes_to_float(audio: bytes, *, channels: int = 1) -> np.ndarray:
    """Little-endian 16-bit PCM bytes -> mono float32 in [-1, 1]."""
    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples
