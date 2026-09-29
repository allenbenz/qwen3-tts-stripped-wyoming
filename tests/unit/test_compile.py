"""Unit tests for the torch.compile wiring (Settings + apply_torch_compile)."""

from __future__ import annotations

import types

import pytest


class TestCompileSettings:
    def test_defaults(self) -> None:
        from qwen3_tts_stripped_wyoming.config import settings_from_env

        settings = settings_from_env({})
        assert settings.compile is False
        assert settings.compile_mode == "default"
        assert settings.compile_dynamic is True

    def test_env_overrides(self) -> None:
        from qwen3_tts_stripped_wyoming.config import settings_from_env

        settings = settings_from_env(
            {
                "QWEN3TTS_COMPILE": "1",
                "QWEN3TTS_COMPILE_MODE": "max-autotune",
                "QWEN3TTS_COMPILE_DYNAMIC": "false",
            }
        )
        assert settings.compile is True
        assert settings.compile_mode == "max-autotune"
        assert settings.compile_dynamic is False

    def test_reduce_overhead_rejected(self) -> None:
        """reduce-overhead (CUDA graphs) is incompatible with qwen-tts's
        DynamicCache (torch.cat KV growth from eager code) and is rejected at
        config validation, before any model loads."""
        from qwen3_tts_stripped_wyoming.config import settings_from_env

        with pytest.raises(ValueError, match="compile_mode"):
            settings_from_env({"QWEN3TTS_COMPILE_MODE": "reduce-overhead"})

    def test_invalid_mode(self) -> None:
        from qwen3_tts_stripped_wyoming.config import settings_from_env

        with pytest.raises(ValueError, match="compile_mode"):
            settings_from_env({"QWEN3TTS_COMPILE_MODE": "turbo"})

    def test_asr_model_empty_disables(self) -> None:
        from qwen3_tts_stripped_wyoming.config import settings_from_env

        assert settings_from_env({}).asr_model == "Qwen/Qwen3-ASR-0.6B"
        assert settings_from_env({"QWEN3TTS_ASR_MODEL": ""}).asr_model is None
        assert settings_from_env({"QWEN3TTS_ASR_MODEL": " "}).asr_model is None
        assert (
            settings_from_env({"QWEN3TTS_ASR_MODEL": "Qwen/Qwen3-ASR-1.7B"}).asr_model
            == "Qwen/Qwen3-ASR-1.7B"
        )


class TestApplyTorchCompile:
    def test_compiles_inner_submodules_not_the_top(self, monkeypatch) -> None:
        """The wrapper must replace talker.model / code_predictor.model in
        place (so parent self.model(...) calls hit the compiled region), not
        wrap the top-level model (generate() would bypass it)."""
        import torch

        from qwen3_tts_stripped_wyoming import runtime

        compiled: list[object] = []

        class FakeOptimized:
            def __init__(self, module: object) -> None:
                self.module = module
                compiled.append(self)

        seen_kwargs: list[dict] = []

        def fake_compile(module, mode=None, dynamic=None, **kwargs):
            seen_kwargs.append({"mode": mode, "dynamic": dynamic, **kwargs})
            return FakeOptimized(module)

        monkeypatch.setattr(torch, "compile", fake_compile)

        talker_inner = types.SimpleNamespace()
        predictor_inner = types.SimpleNamespace()
        predictor = types.SimpleNamespace(model=predictor_inner)
        talker = types.SimpleNamespace(model=talker_inner, code_predictor=predictor)
        tts_model = types.SimpleNamespace(talker=talker)

        targets = runtime.apply_torch_compile(tts_model, mode="default", dynamic=True)

        assert targets == ["talker.model", "talker.code_predictor.model"]
        assert len(compiled) == 2
        # the wrappers replaced the inner modules in place on the parents
        assert talker.model is compiled[0]
        assert talker.model.module is talker_inner
        assert predictor.model is compiled[1]
        assert predictor.model.module is predictor_inner
        # the top-level model object was never wrapped
        assert not isinstance(tts_model, FakeOptimized)
        assert all(k == {"mode": "default", "dynamic": True} for k in seen_kwargs)

    def test_missing_predictor_is_skipped(self, monkeypatch) -> None:
        import torch

        from qwen3_tts_stripped_wyoming import runtime

        monkeypatch.setattr(torch, "compile", lambda m, **k: types.SimpleNamespace(orig=m))
        talker = types.SimpleNamespace(model=types.SimpleNamespace())  # no code_predictor
        tts_model = types.SimpleNamespace(talker=talker)
        assert runtime.apply_torch_compile(tts_model, mode="default", dynamic=True) == [
            "talker.model"
        ]


class TestRevertTorchCompile:
    def test_revert_restores_original_modules(self, monkeypatch) -> None:
        """apply -> revert must leave exactly the original module objects in
        place (the warmup-failure fallback path depends on this)."""
        import torch

        from qwen3_tts_stripped_wyoming import runtime

        class FakeOptimized:
            def __init__(self, module: object) -> None:
                self._orig_mod = module

        monkeypatch.setattr(torch, "compile", lambda m, **k: FakeOptimized(m))

        talker_inner = types.SimpleNamespace()
        predictor_inner = types.SimpleNamespace()
        predictor = types.SimpleNamespace(model=predictor_inner)
        talker = types.SimpleNamespace(model=talker_inner, code_predictor=predictor)
        tts_model = types.SimpleNamespace(talker=talker)

        runtime.apply_torch_compile(tts_model, mode="default", dynamic=True)
        assert talker.model is not talker_inner

        restored = runtime.revert_torch_compile(tts_model)
        assert restored == 2
        assert talker.model is talker_inner
        assert predictor.model is predictor_inner

    def test_revert_without_compile_is_noop(self) -> None:
        from qwen3_tts_stripped_wyoming import runtime

        talker = types.SimpleNamespace(model=types.SimpleNamespace())
        tts_model = types.SimpleNamespace(talker=talker)
        assert runtime.revert_torch_compile(tts_model) == 0


class TestMissingCompileToolchain:
    """The upfront check that keeps a compiler-less image (the Docker image
    ships no gcc/g++ and no triton) from crashing at first inference."""

    def test_cuda_reports_missing_gcc_and_triton(self, monkeypatch) -> None:
        import importlib
        import shutil

        from qwen3_tts_stripped_wyoming import runtime

        monkeypatch.setattr(shutil, "which", lambda name: None)

        def fake_import(name: str, *args: object) -> object:
            if name == "triton":
                raise ImportError(name)
            return importlib.import_module(name, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(importlib, "import_module", fake_import)

        missing = runtime._missing_compile_toolchain(using_cuda=True)
        assert any("gcc" in item for item in missing)
        assert any("triton" in item for item in missing)

    def test_cpu_reports_missing_gpp(self, monkeypatch) -> None:
        import shutil

        from qwen3_tts_stripped_wyoming import runtime

        monkeypatch.setattr(shutil, "which", lambda name: None)
        missing = runtime._missing_compile_toolchain(using_cuda=False)
        assert any("g++" in item for item in missing)

    def test_complete_toolchain_is_silent(self, monkeypatch) -> None:
        import importlib
        import types as types_mod

        from qwen3_tts_stripped_wyoming import runtime

        monkeypatch.setattr(importlib, "import_module", lambda name: types_mod.SimpleNamespace())

        def which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("gcc", "cc", "g++", "c++") else None

        import shutil

        monkeypatch.setattr(shutil, "which", which)
        assert runtime._missing_compile_toolchain(using_cuda=True) == []
        assert runtime._missing_compile_toolchain(using_cuda=False) == []
