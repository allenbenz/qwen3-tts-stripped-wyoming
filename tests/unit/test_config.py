"""Unit tests for settings parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen3_tts_stripped_wyoming.config import (
    build_arg_parser,
    settings_from_args,
    settings_from_env,
)


class TestEnv:
    def test_defaults(self) -> None:
        settings = settings_from_env({})
        assert settings.uri == "tcp://0.0.0.0:10200"
        assert settings.variant == "auto"
        assert settings.keep_set == "latin"
        assert settings.st_dtype == "float16"
        assert settings.q8_group == 64
        assert settings.convert is True
        assert settings.temperature is None

    def test_env_overrides(self) -> None:
        settings = settings_from_env(
            {
                "QWEN3TTS_URI": "tcp://0.0.0.0:10300",
                "QWEN3TTS_MODEL": "../narrator-tts",
                "QWEN3TTS_VARIANT": "q8",
                "QWEN3TTS_KEEP_SET": "ml",
                "QWEN3TTS_Q8_GROUP": "32",
                "QWEN3TTS_TEMPERATURE": "0.5",
                "QWEN3TTS_INSTRUCT": "warm and gentle",
                "QWEN3TTS_WARMUP": "false",
                "QWEN3TTS_SEED": "9",
            }
        )
        assert settings.uri == "tcp://0.0.0.0:10300"
        assert settings.model == "../narrator-tts"
        assert settings.variant == "q8"
        assert settings.keep_set == "ml"
        assert settings.q8_group == 32
        assert settings.temperature == 0.5
        assert settings.instruct == "warm and gentle"
        assert settings.warmup is False
        assert settings.seed == 9

    def test_invalid_values(self) -> None:
        with pytest.raises(ValueError, match="variant"):
            settings_from_env({"QWEN3TTS_VARIANT": "q4"})
        with pytest.raises(ValueError, match="keep_set"):
            settings_from_env({"QWEN3TTS_KEEP_SET": "tiny"})
        with pytest.raises(ValueError, match="q8_group"):
            settings_from_env({"QWEN3TTS_Q8_GROUP": "0"})
        with pytest.raises(ValueError, match="boolean"):
            settings_from_env({"QWEN3TTS_WARMUP": "maybe"})

    def test_flag_overrides_env(self) -> None:
        parser = build_arg_parser({"QWEN3TTS_VARIANT": "lite", "QWEN3TTS_Q8_GROUP": "32"})
        args = parser.parse_args(["--variant", "q8"])
        settings = settings_from_args(args)
        assert settings.variant == "q8"
        assert settings.q8_group == 32  # untouched env default came through

    def test_model_dir_path(self) -> None:
        settings = settings_from_env({"QWEN3TTS_MODEL_DIR": "/data/models"})
        assert settings.model_dir == Path("/data/models")
