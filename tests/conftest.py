"""Shared fixtures. FakeQwenModel lives in tests/fakes.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen3_tts_stripped_wyoming.config import Settings
from qwen3_tts_stripped_wyoming.runtime import SynthesisService

from .fakes import FakeAsrModel, FakeQwenModel
from .utils import make_fake_asr_service, make_fake_service


@pytest.fixture
def fake_model() -> FakeQwenModel:
    return FakeQwenModel()


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    return Settings(model_dir=tmp_path, output_chunk_ms=50)


@pytest.fixture
def fake_service(fake_model: FakeQwenModel, fake_settings: Settings) -> SynthesisService:
    return make_fake_service(fake_model, fake_settings)


@pytest.fixture
def multi_voice_service(fake_model: FakeQwenModel, fake_settings: Settings) -> SynthesisService:
    """Service with two speakers so voice/language routing is observable."""
    return make_fake_service(
        fake_model,
        fake_settings,
        speakers=("aiden", "ryan"),
        languages=("english", "chinese"),
    )


@pytest.fixture
def fake_asr_model() -> FakeAsrModel:
    return FakeAsrModel()


@pytest.fixture
def fake_asr_service(fake_asr_model: FakeAsrModel, fake_settings: Settings):
    return make_fake_asr_service(fake_asr_model, fake_settings)
