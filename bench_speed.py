"""Diagnose inference speed: resolved attention impl + decode throughput.

.venv/Scripts/python.exe bench_speed.py narrator-tts-lite-q8
.venv/Scripts/python.exe bench_speed.py ../qwen3-tts-scratch/narrator-tts-lite
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch._dynamo  # ensures torch._dynamo.config exists for the bench compile path


def bench(model_dir: str, compile_mode: str | None = None) -> None:
    variant = Path(model_dir).name
    if "q8" in variant:
        from qwen3_tts_stripped_wyoming.patches import q8  # noqa: F401
    else:
        from qwen3_tts_stripped_wyoming.patches import lite  # noqa: F401
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.bfloat16)
    if compile_mode:
        from qwen3_tts_stripped_wyoming.runtime import apply_torch_compile

        torch._dynamo.config.suppress_errors = True
        targets = apply_torch_compile(model.model, mode=compile_mode, dynamic=True)
        print(f"[{variant}] torch.compile ({compile_mode}): {targets}")

    talker_cfg = model.model.talker.config
    st_cfg = getattr(model.model.speech_tokenizer, "config", None)
    print(f"[{variant}] talker attn: {talker_cfg._attn_implementation}")
    if st_cfg is not None:
        print(f"[{variant}] ST attn: {getattr(st_cfg, '_attn_implementation', '?')}")

    text = (
        "Once upon a time, in a forest older than memory, there lived a bear "
        "who collected stories the way other bears collected honey."
    )
    # warmup
    torch.manual_seed(1)
    wavs, sr = model.generate_custom_voice(text=text, language="English", speaker="narrator")
    torch.cuda.synchronize()

    for i in range(3):
        torch.manual_seed(1)
        t0 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=text, language="English", speaker="narrator")
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        dur = len(wavs[0]) / sr
        print(
            f"[{variant}] run {i}: {wall:.2f}s wall for {dur:.2f}s audio "
            f"-> RTF {wall / dur:.2f}x ({dur / wall:.2f}x realtime)"
        )
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = sys.argv[1:]
    compile_mode = "default" if "--compile" in args else None
    if "--compile-mode" in args:
        compile_mode = args[args.index("--compile-mode") + 1]
    bench(args[0], compile_mode=compile_mode)
