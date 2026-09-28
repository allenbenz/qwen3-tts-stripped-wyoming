"""Unit tests for SynthesisService voice/language resolution and synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from qwen3_tts_stripped_wyoming.config import Settings
from qwen3_tts_stripped_wyoming.runtime import SynthesisServiceError, VoiceResolutionError

from ..fakes import FAKE_SAMPLE_RATE, fake_audio_piece
from ..utils import make_fake_service


def test_default_speaker(fake_service) -> None:
    assert fake_service.default_speaker().id == "narrator"


def test_default_speaker_from_settings(fake_model, tmp_path) -> None:
    service = make_fake_service(fake_model, Settings(model_dir=tmp_path, default_voice="Aiden"))
    assert service.default_speaker().id == "aiden"


def test_invalid_default_speaker(fake_model, tmp_path) -> None:
    service = make_fake_service(fake_model, Settings(model_dir=tmp_path, default_voice="nope"))
    with pytest.raises(SynthesisServiceError, match="aiden"):
        service.default_speaker()


def test_resolve_speaker_case_and_spaces(fake_service) -> None:
    assert fake_service.resolve_speaker("AIDEN").id == "aiden"
    assert fake_service.resolve_speaker(None).id == "narrator"
    with pytest.raises(VoiceResolutionError, match="narrator"):
        fake_service.resolve_speaker("unknown")


def test_resolve_language(fake_service) -> None:
    assert fake_service.resolve_language("en-US") == "english"
    assert fake_service.resolve_language("zh") == "chinese"
    assert fake_service.resolve_language(None) is None  # auto-detect
    with pytest.raises(VoiceResolutionError, match="Unsupported language"):
        fake_service.resolve_language("pt-BR")


def test_resolve_language_default_setting(fake_model, tmp_path) -> None:
    service = make_fake_service(fake_model, Settings(model_dir=tmp_path, default_language="de"))
    assert service.resolve_language(None) == "german"


def test_resolve_speaker_for_request(fake_service) -> None:
    assert fake_service.resolve_speaker_for_request(None, None).id == "narrator"
    assert fake_service.resolve_speaker_for_request("aiden", None).id == "aiden"
    assert fake_service.resolve_speaker_for_request(None, "zh").id in {"narrator", "aiden"}


async def test_synthesize_passes_params(fake_model, tmp_path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        temperature=0.5,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=512,
        instruct="warm and gentle",
        seed=7,
    )
    service = make_fake_service(fake_model, settings)
    audio = await service.synthesize(text="Hello", speaker_id="aiden", language="english")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    request = fake_model.requests[-1]
    assert request["speaker"] == "aiden"
    assert request["language"] == "english"
    assert request["instruct"] == "warm and gentle"
    assert request["temperature"] == 0.5
    assert request["top_k"] == 10
    assert request["top_p"] == 0.9
    assert request["repetition_penalty"] == 1.1
    assert request["max_new_tokens"] == 512


async def test_synthesize_omits_language_for_auto(fake_model, tmp_path) -> None:
    service = make_fake_service(fake_model, Settings(model_dir=tmp_path))
    await service.synthesize(text="Hello", speaker_id="aiden", language=None)
    # the fake records its own default; the real model treats None as Auto
    assert fake_model.requests[-1]["language"] is None


async def test_synthesize_failure_propagates(fake_model, tmp_path) -> None:
    service = make_fake_service(fake_model, Settings(model_dir=tmp_path))
    fake_model.fail = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await service.synthesize(text="Hello", speaker_id="aiden", language=None)


def test_speaker_languages_bcp47(fake_service) -> None:
    langs = fake_service.speaker_languages_bcp47(fake_service.default_speaker())
    assert set(langs) == {"en", "zh", "de"}


def test_sample_rate(fake_service) -> None:
    assert fake_service.sample_rate == FAKE_SAMPLE_RATE


def test_fake_audio_deterministic() -> None:
    assert np.array_equal(fake_audio_piece(), fake_audio_piece())
