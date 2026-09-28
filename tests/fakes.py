"""Test doubles for the qwen-tts model surface used by SynthesisService."""

from __future__ import annotations

from typing import Any

import numpy as np

FAKE_SAMPLE_RATE = 24000
FAKE_TONE_HZ = 440.0
FAKE_AMPLITUDE = 0.5
FAKE_PIECE_MS = 100


def fake_audio_piece(
    sample_rate: int = FAKE_SAMPLE_RATE, piece_ms: int = FAKE_PIECE_MS
) -> np.ndarray:
    """Deterministic sine wave, mirroring float_to_int16_bytes's math order."""
    samples = sample_rate * piece_ms // 1000
    t = np.arange(samples, dtype=np.float32) / np.float32(sample_rate)
    return (np.float32(FAKE_AMPLITUDE) * np.sin(2 * np.pi * np.float32(FAKE_TONE_HZ) * t)).astype(
        np.float32
    )


class FakeQwenModel:
    """Minimal stand-in for Qwen3TTSModel covering the surface we call.

    ``generate_custom_voice`` records requests and returns one piece of
    deterministic audio (two pieces worth of samples when the text is longer
    than 40 chars), mimicking the (wavs, sr) return shape.
    """

    def __init__(self, sample_rate: int = FAKE_SAMPLE_RATE) -> None:
        self.requests: list[dict[str, Any]] = []
        self._sample_rate = sample_rate
        self.fail: Exception | None = None

    def generate_custom_voice(
        self,
        *,
        text: str | None = None,
        speaker: str = "aiden",
        language: str | None = None,
        instruct: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        if self.fail is not None:
            raise self.fail
        self.requests.append(
            {
                "text": text,
                "speaker": speaker,
                "language": language,
                "instruct": instruct,
                **kwargs,
            }
        )
        piece = fake_audio_piece(self._sample_rate)
        count = 2 if text and len(text) > 40 else 1
        wav = np.concatenate([piece] * count)
        return [wav], self._sample_rate


class FakeAsrModel:
    """Minimal stand-in for Qwen3ASRModel covering the surface we call.

    ``transcribe`` records requests and echoes a canned transcript for audio
    that resembles the deterministic sine piece.
    """

    def __init__(self, text: str = "hello world", language: str = "English") -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail: Exception | None = None
        self._text = text
        self._language = language

    def transcribe(self, *, audio, language=None, context="", **kwargs: Any) -> list[Any]:
        if self.fail is not None:
            raise self.fail
        waveform, sr = audio
        self.requests.append(
            {"samples": len(waveform), "sr": sr, "language": language, "context": context}
        )
        from types import SimpleNamespace

        return [SimpleNamespace(text=self._text, language=self._language)]
