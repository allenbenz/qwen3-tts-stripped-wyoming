"""Unit tests for the fast (faster-qwen3-tts) backend wiring."""

from __future__ import annotations

import numpy as np
import pytest

from qwen3_tts_stripped_wyoming.config import Settings, settings_from_env

from ..fakes import FakeFastModel, fake_audio_piece
from ..utils import make_fake_service


class TestBackendSettings:
    def test_defaults(self) -> None:
        settings = settings_from_env({})
        assert settings.tts_backend == "fast"
        assert settings.stream_chunk_steps == 8

    def test_env_overrides(self) -> None:
        settings = settings_from_env(
            {"QWEN3TTS_TTS_BACKEND": "stock", "QWEN3TTS_STREAM_CHUNK_STEPS": "4"}
        )
        assert settings.tts_backend == "stock"
        assert settings.stream_chunk_steps == 4

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="tts_backend"):
            settings_from_env({"QWEN3TTS_TTS_BACKEND": "turbo"})
        with pytest.raises(ValueError, match="stream_chunk_steps"):
            settings_from_env({"QWEN3TTS_STREAM_CHUNK_STEPS": "0"})
        with pytest.raises(ValueError, match="stream_chunk_steps"):
            settings_from_env({"QWEN3TTS_STREAM_CHUNK_STEPS": "65"})


class TestFastStreaming:
    async def test_stream_yields_pieces_in_order(self, tmp_path) -> None:
        model = FakeFastModel(pieces=3)
        service = make_fake_service(model, Settings(model_dir=tmp_path), backend="fast")
        collected = [
            piece
            async for piece in service.stream(text="Hello", speaker_id="aiden", language="English")
        ]
        assert len(collected) == 3
        expected = fake_audio_piece()
        for piece in collected:
            np.testing.assert_array_equal(piece, expected)
        request = model.requests[-1]
        assert request["speaker"] == "aiden"
        assert request["language"] == "English"
        assert request["chunk_size"] == 8  # default stream_chunk_steps

    async def test_stream_chunk_steps_forwarded(self, tmp_path) -> None:
        model = FakeFastModel(pieces=2)
        service = make_fake_service(
            model, Settings(model_dir=tmp_path, stream_chunk_steps=2), backend="fast"
        )
        _ = [piece async for piece in service.stream(text="Hi", speaker_id="aiden", language=None)]
        assert model.requests[-1]["chunk_size"] == 2

    async def test_synthesize_concatenates_chunks(self, tmp_path) -> None:
        model = FakeFastModel(pieces=3)
        service = make_fake_service(model, Settings(model_dir=tmp_path), backend="fast")
        audio = await service.synthesize(text="Hello", speaker_id="aiden", language=None)
        expected = np.concatenate([fake_audio_piece()] * 3)
        np.testing.assert_array_equal(audio, expected)
        # language=None maps to Auto on the fast wrapper
        assert model.requests[-1]["language"] == "Auto"

    async def test_stock_backend_stream_yields_single_piece(self, fake_model, tmp_path) -> None:
        from ..fakes import FakeQwenModel

        assert isinstance(fake_model, FakeQwenModel)
        service = make_fake_service(fake_model, Settings(model_dir=tmp_path), backend="stock")
        collected = [
            piece
            async for piece in service.stream(text="Hello", speaker_id="aiden", language="english")
        ]
        assert len(collected) == 1  # one-shot: the whole clip at once
        assert fake_model.requests[-1]["language"] == "english"

    async def test_stream_failure_propagates(self, tmp_path) -> None:
        model = FakeFastModel(pieces=2)
        model.fail = RuntimeError("boom")
        service = make_fake_service(model, Settings(model_dir=tmp_path), backend="fast")
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in service.stream(text="Hi", speaker_id="aiden", language=None):
                pass
