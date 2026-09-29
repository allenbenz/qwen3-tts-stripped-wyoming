"""Unit tests for the ASR service: language mapping and transcription."""

from __future__ import annotations

import numpy as np
import pytest

from qwen3_tts_stripped_wyoming.asr import LanguageResolutionError, TranscriptionService
from qwen3_tts_stripped_wyoming.audio import (
    asr_language_to_bcp47,
    bcp47_to_asr_language,
    pcm_bytes_to_float,
)

from ..fakes import FakeAsrModel
from ..utils import make_fake_asr_service

LANGUAGES = ("Chinese", "English", "Cantonese", "German", "Japanese")


def test_asr_language_to_bcp47() -> None:
    assert asr_language_to_bcp47("English") == "en"
    assert asr_language_to_bcp47("Cantonese") == "yue"
    assert asr_language_to_bcp47("Filipino") == "fil"


def test_bcp47_to_asr_language() -> None:
    assert bcp47_to_asr_language("en", LANGUAGES) == "English"
    assert bcp47_to_asr_language("en-US", LANGUAGES) == "English"
    assert bcp47_to_asr_language("zh-CN", LANGUAGES) == "Chinese"
    assert bcp47_to_asr_language("yue", LANGUAGES) == "Cantonese"
    assert bcp47_to_asr_language("xx", LANGUAGES) is None


def test_pcm_bytes_to_float() -> None:
    data = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    out = pcm_bytes_to_float(data)
    assert out.dtype == np.float32
    assert out[1] == pytest.approx(0.5, abs=1e-4)
    assert out[2] == pytest.approx(-0.5, abs=1e-4)
    # stereo -> mono averages channels
    stereo = np.array([[16384, 0], [0, -16384]], dtype="<i2").tobytes()
    mono = pcm_bytes_to_float(stereo, channels=2)
    assert mono.shape == (2,)
    assert mono[0] == pytest.approx(0.25, abs=1e-4)


def test_resolve_language_auto(fake_asr_service) -> None:
    assert fake_asr_service.resolve_language(None) is None


def test_resolve_language_bcp47(fake_asr_service) -> None:
    assert fake_asr_service.resolve_language("en-US") == "English"
    assert fake_asr_service.resolve_language("zh") == "Chinese"


def test_resolve_language_default_setting(fake_asr_model, tmp_path) -> None:
    from qwen3_tts_stripped_wyoming.config import Settings

    service = make_fake_asr_service(fake_asr_model, Settings(model_dir=tmp_path, asr_language="de"))
    assert service.resolve_language(None) == "German"


def test_resolve_language_unsupported(fake_asr_service) -> None:
    with pytest.raises(LanguageResolutionError, match="Supported languages"):
        fake_asr_service.resolve_language("xx-YY")


def test_languages_bcp47(fake_asr_service) -> None:
    langs = fake_asr_service.languages_bcp47()
    assert "en" in langs and "zh" in langs and "de" in langs


async def test_transcribe_passes_audio_and_language(fake_asr_model, tmp_path) -> None:
    from qwen3_tts_stripped_wyoming.config import Settings

    service = make_fake_asr_service(fake_asr_model, Settings(model_dir=tmp_path))
    audio = np.zeros(16000, dtype=np.float32)
    text, language = await service.transcribe(
        audio, sample_rate=16000, language="English", context=None
    )
    assert text == "hello world"
    assert language == "English"
    request = fake_asr_model.requests[-1]
    assert request["samples"] == 16000
    assert request["sr"] == 16000
    assert request["language"] == "English"
    assert request["context"] == ""


async def test_transcribe_pads_short_audio(fake_asr_model, tmp_path) -> None:
    from qwen3_tts_stripped_wyoming.config import Settings

    service = make_fake_asr_service(fake_asr_model, Settings(model_dir=tmp_path))
    audio = np.zeros(100, dtype=np.float32)
    await service.transcribe(audio, sample_rate=16000, language=None)
    # padded to the 0.5 s minimum
    assert fake_asr_model.requests[-1]["samples"] == 8000


async def test_transcribe_forwards_context(fake_asr_model, tmp_path) -> None:
    from qwen3_tts_stripped_wyoming.config import Settings

    service = make_fake_asr_service(fake_asr_model, Settings(model_dir=tmp_path))
    await service.transcribe(
        np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        language=None,
        context="living room lamp, kitchen light",
    )
    assert fake_asr_model.requests[-1]["context"] == "living room lamp, kitchen light"


async def test_transcribe_failure_propagates(fake_asr_model, tmp_path) -> None:
    from qwen3_tts_stripped_wyoming.config import Settings

    service = make_fake_asr_service(fake_asr_model, Settings(model_dir=tmp_path))
    fake_asr_model.fail = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await service.transcribe(
            np.zeros(16000, dtype=np.float32), sample_rate=16000, language=None
        )


def test_transcription_service_requires_model(fake_settings) -> None:
    # constructor only wires things; create() validates the source
    service = TranscriptionService(FakeAsrModel(), fake_settings, languages=("English",))
    assert service.languages == ("English",)


class TestEnsureNativeFeatureExtractor:
    """The load-time heal for stale (Whisper) feature extractor configs."""

    def test_swaps_foreign_extractor(self) -> None:
        from types import SimpleNamespace

        from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import (
            Qwen3ASRFeatureExtractor,
        )

        from qwen3_tts_stripped_wyoming.asr import _ensure_native_feature_extractor

        class FakeWhisper:
            feature_size = 128
            sampling_rate = 16000
            hop_length = 160
            n_fft = 400
            dither = 0.0

        processor = SimpleNamespace(feature_extractor=FakeWhisper())
        audio_config = SimpleNamespace(num_mel_bins=128, n_window=50)
        model = SimpleNamespace(config=SimpleNamespace(audio_config=audio_config))
        _ensure_native_feature_extractor(processor, model)
        fe = processor.feature_extractor
        assert isinstance(fe, Qwen3ASRFeatureExtractor)
        assert fe.n_window == 50
        assert fe.min_length == 8000
        assert fe.return_attention_mask is True
        assert fe.feature_size == 128

    def test_native_extractor_untouched(self) -> None:
        from types import SimpleNamespace

        from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import (
            Qwen3ASRFeatureExtractor,
        )

        from qwen3_tts_stripped_wyoming.asr import _ensure_native_feature_extractor

        fe = Qwen3ASRFeatureExtractor()
        processor = SimpleNamespace(feature_extractor=fe)
        model = SimpleNamespace(config=SimpleNamespace(audio_config=None))
        _ensure_native_feature_extractor(processor, model)
        assert processor.feature_extractor is fe
