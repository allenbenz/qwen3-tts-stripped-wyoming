"""Unit tests for audio helpers and language mapping."""

from __future__ import annotations

import numpy as np
import pytest

from qwen3_tts_stripped_wyoming.audio import (
    bcp47_to_model_language,
    chunk_bytes_for_ms,
    float_to_int16_bytes,
    match_model_language,
    model_language_to_bcp47,
    split_bytes,
)


class TestPcm:
    def test_sine_to_pcm(self) -> None:
        samples = np.zeros(240, dtype=np.float32)
        samples[:120] = 0.5
        samples[120:] = -0.5
        data = float_to_int16_bytes(samples)
        assert len(data) == 240 * 2
        values = np.frombuffer(data, dtype="<i2")
        assert values[0] == pytest.approx(16383, abs=2)
        assert values[-1] == pytest.approx(-16383, abs=2)

    def test_clipping(self) -> None:
        data = float_to_int16_bytes(np.array([2.0, -2.0], dtype=np.float32))
        values = np.frombuffer(data, dtype="<i2")
        assert values[0] == 32767
        assert values[1] == -32767

    def test_gain(self) -> None:
        data = float_to_int16_bytes(np.array([0.5], dtype=np.float32), gain=0.5)
        assert np.frombuffer(data, dtype="<i2")[0] == pytest.approx(8191, abs=2)

    def test_split_bytes(self) -> None:
        parts = list(split_bytes(b"abcdef", 4))
        assert parts == [b"abcd", b"ef"]
        with pytest.raises(ValueError):
            list(split_bytes(b"x", 0))

    def test_chunk_bytes_for_ms(self) -> None:
        # 24000 Hz * 0.2 s = 4800 samples * 2 bytes * 1 channel
        assert chunk_bytes_for_ms(24000, 200) == 9600


class TestLanguages:
    def test_model_to_bcp47(self) -> None:
        assert model_language_to_bcp47("english") == "en"
        assert model_language_to_bcp47("chinese") == "zh"
        assert model_language_to_bcp47("beijing_dialect") == "zh"
        assert model_language_to_bcp47("klingon") == "klingon"

    def test_bcp47_to_model(self) -> None:
        assert bcp47_to_model_language("en") == "english"
        assert bcp47_to_model_language("en-US") == "english"
        assert bcp47_to_model_language("zh-CN") == "chinese"
        assert bcp47_to_model_language("de") == "german"

    def test_match_model_language(self) -> None:
        available = ("english", "chinese", "german")
        assert match_model_language(available, "en-US") == "english"
        assert match_model_language(available, "zh") == "chinese"
        assert match_model_language(available, "de-DE") == "german"
        # a language the model family knows but this model does not advertise
        # comes back mapped-but-unmatched so callers can report it
        assert match_model_language(available, "pt-BR") == "portuguese"
        assert "portuguese" not in available
