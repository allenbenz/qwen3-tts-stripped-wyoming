# qwen3-tts-stripped-wyoming

A [Wyoming](https://github.com/rhasspy/wyoming)-protocol server for **Qwen3-TTS CustomVoice models** and **Qwen3-ASR speech recognition** — text-to-speech and speech-to-text on one port for Home Assistant. TTS serves the stock checkpoints, community finetunes (e.g. [scrappylabs/narrator-tts](https://huggingface.co/scrappylabs/narrator-tts)), or size-optimized "stripped" conversions of either; audio is emitted as 16-bit PCM at 24 kHz mono. STT transcribes `transcribe` requests with [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (30 languages, auto language detection). Inference runs on CPU or NVIDIA GPU via **PyTorch** (`qwen-tts` / `qwen-asr`).

The server serves any of three checkpoint variants and can convert between them on first start (see [Variants](#variants)):

| Variant | 1.7B payload | What it is |
| --- | --- | --- |
| `bf16` | 4.52 GB | the source checkpoint as shipped |
| `lite` | 3.86 GB | vocabulary pruning (token-map indirection) + decoder-only fp16 speech tokenizer — talker is **bit-exact** vs `bf16` |
| `q8` (default) | 2.42 GB | `lite` + every non-embedding `nn.Linear` quantized to group-wise int8 — near-lossless (argmax agreement 100% in measured A/B) |

## How it works

- Answers Wyoming `describe` and `ping` requests so Home Assistant can discover the server and its voices (one voice per model speaker: `aiden`, `serena`, … or `narrator` for the finetune).
- Handles both one-shot `synthesize` and streaming `synthesize-start` / `synthesize-chunk` / `synthesize-stop` requests (text is buffered; synthesis runs once on stop).
- Emits `audio-start`, `audio-chunk`, and `audio-stop` events carrying 16-bit PCM at 24 kHz mono.
- **STT**: handles `transcribe` → `audio-start` / `audio-chunk`+ / `audio-stop` and replies with a `transcript` event. Language is taken from the request (BCP-47) or auto-detected; `transcript_names` / `transcript_terms` are forwarded as recognition bias context. Any input sample rate works (audio is resampled to 16 kHz).
- Serializes all model work (synthesis and transcription) across connections with one lock: the models share the GPU, and a voice pipeline is half-duplex anyway.
- TTS language is mapped per request from Home Assistant (`en`, `en-US`, `zh`, `de`, …) to the model's language ids; when no language is given the model auto-detects (`Auto`).

## Requirements

- Python 3.12+
- ~5 GB disk for a downloaded 1.7B TTS source (or point at a local copy); converted variants add ~2.4–3.9 GB each; the default ASR model adds ~1.5 GB
- Optional: NVIDIA GPU (CUDA build of torch; CPU works but is slow for 1.7B TTS)

## Quickstart (local)

Requires [mise](https://mise.jdx.dev/) (which provides Python 3.12 and uv):

```sh
mise run install                      # or: mise exec -- uv sync
mise exec -- uv run qwen3-tts-stripped-wyoming \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --variant q8
```

The first run downloads the model from Hugging Face into the model directory and (for `--variant q8`) converts it once — a few minutes on CPU. Later starts reuse the cached conversion. The server then listens on `tcp://0.0.0.0:10200`.

Serve an existing local model (any variant, no conversion — the variant is detected from `config.json` markers):

```sh
mise exec -- uv run qwen3-tts-stripped-wyoming --model ../narrator-tts --variant auto
```

## Variants

`--variant` selects what to serve:

- `auto` (default for local dirs is effectively this): serve the source as-is; the variant is detected from config markers (`q8_quantization` → q8, `talker_config.vocab_pruning` → lite, else bf16).
- `bf16` / `lite` / `q8`: serve that variant, converting the source once if needed (cached under `<model-dir>/converted/…`; disable with `--no-convert`).

Conversion knobs (also used for cache-directory naming):

- `--keep-set latin|ml` — vocabulary keep-set for lite/q8. `latin` (default) guarantees that **any Latin-script text** (English, French, German, Spanish, Italian, Portuguese, plus symbols/emoji) tokenizes entirely inside the kept vocabulary. `ml` adds Chinese/Japanese/Korean/Russian/Greek coverage (much bigger, little savings).
- `--st-dtype float16|bfloat16|float32` — storage dtype of the speech-tokenizer decoder (`float16` matches the upstream "lite" recipe).
- `--q8-group N` — inputs per int8 scale group (default 64; comparable to or finer than llama.cpp Q8_0).

Only `custom_voice` models are convertible: Base (voice-clone) models need the speech-tokenizer encoder that lite strips, and Wyoming has no reference-audio concept anyway. The server refuses anything that is not `tts_model_type: custom_voice`.

### Fidelity summary (measured on the narrator-tts 1.7B finetune, CUDA, bf16)

- `lite` vs `bf16`: identical codec-token streams with identical seeds (bit-exact talker); the fp16 speech-tokenizer decoder measures ~37 dB SNR / mel-corr 1.0000 vs fp32.
- `q8` vs `lite`: same-context next-token argmax agreement 100% across sampled steps, top-50 overlap 96–99%, KL ≤ 1e-2; sampled audio differs token-wise (any near-lossless quantization diverges under sampling — same class as re-seeding the model) but produces valid narrations.

## Performance

Measured on an RTX 3090 (bf16, ~7.5 s of audio per request, eager inference):

| Configuration | Real-time factor | Notes |
| --- | --- | --- |
| `lite` (bf16) | 1.7x | baseline; fastest |
| `q8` | 2.3x | int8 → fp32 → bf16 on every decode step; ~1.5 GB less VRAM, 2.4 GB on disk |

Notes:

- **torch.compile / Triton (experimental, opt-in via `QWEN3TTS_COMPILE=1`; enabled by default in the Docker image).** Because decode is launch-bound (see below), Inductor fusion is the one remaining speed lever. The server compiles the talker and code-predictor stacks at first inference (`mode=default`, `dynamic=True`; both configurable) and caches the artifacts under `/data/compile-cache` in Docker, so the multi-minute compile happens once per model, not per container. Measured on CPU (podman, 8 cores): the full path compiles and synthesizes correctly; expect ~1.5–2.5x on GPU, but treat it as an experiment — if anything misbehaves set `QWEN3TTS_COMPILE=0`. Inductor needs a C/C++ compiler at runtime (the image ships `gcc`+`g++`; venv users need them in PATH — the server checks and falls back to eager with a clear log line when missing).
- **The `flash-attn is not installed` warning is noise.** It is printed unconditionally at import time by the (unused) 25 Hz tokenizer's whisper encoder. The talker, 12 Hz speech tokenizer, and ASR all resolve to **SDPA** attention (`sdpa` in the startup logs), which at these sequence lengths (~100 tokens) is within noise of flash-attn. Installing flash-attn is not worth pinning a torch-2.14-matching wheel for.
- The remaining gap to real time is the qwen-tts eager stack itself: per audio frame it runs one 28-layer talker step plus 15 small code-predictor steps, so decode is kernel-launch-bound rather than attention- or bandwidth-bound. `torch.compile` would help but pulls Triton JIT (compiler in the image) — deliberately out of scope.
- Rule of thumb: `lite` when speed matters (bit-exact and fastest); `q8` when VRAM or disk is the constraint (~35% slower for ~1.5 GB less VRAM and a 2.4 GB payload). Materializing q8 weights to bf16 at load time would just recreate `lite`'s runtime, so no such option exists.

## Docker

```sh
docker run -p 10200:10200 -v qwen3tts-models:/data ghcr.io/allenbenz/qwen3-tts-stripped-wyoming:latest
```

GPU variant (requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```sh
docker run --gpus all -p 10200:10200 \
  -e QWEN3TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  -v qwen3tts-models:/data ghcr.io/allenbenz/qwen3-tts-stripped-wyoming:latest
```

The image is **linux/amd64 only** — the PyPI torch wheels bundle CUDA for x86_64.

## Speech-to-text (Qwen3-ASR)

The server also runs a [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) model (default `Qwen/Qwen3-ASR-0.6B`; `Qwen/Qwen3-ASR-1.7B` for higher accuracy) alongside TTS:

- 30 languages with automatic language detection (or force one per request from Home Assistant's language setting).
- Recognizes speech, singing, and songs with background music; inputs of any sample rate are resampled (Home Assistant sends 16 kHz 16-bit mono).
- `transcript_names` / `transcript_terms` from the Wyoming request are forwarded as bias context (e.g. entity names — disable with `--no-asr-context`).
- ASR sources resolve exactly like TTS sources: HF repo ids are downloaded into `--model-dir`, local directories are used in place. ASR models need no conversion.
- If the ASR model fails to load, the server logs the error and keeps serving TTS (STT requests then get an `asr-disabled` error).

Disable STT entirely with `--asr-model ""` (or `QWEN3TTS_ASR_MODEL=`).

## Home Assistant

1. Go to **Settings → Devices & Services → Add Integration → Wyoming Protocol**.
2. Enter the server's host and port `10200`.
3. Voices appear under the TTS entity dropdown (one per model speaker); the STT engine appears under the speech-to-text provider list.

Voice and language are chosen **per request** from Home Assistant — they are not environment variables. The `QWEN3TTS_VOICE` / `QWEN3TTS_LANGUAGE` settings below are only server-side defaults used when a request omits them. A global default style instruction (e.g. `"warm and gentle"`) can be set with `QWEN3TTS_INSTRUCT` — the Wyoming protocol has no per-request instruct field.

## Configuration

Every setting is available as an environment variable and as a CLI flag of the same meaning (`--uri`, `--model`, `--model-dir`, `--variant`, `--convert`, `--keep-set`, `--st-dtype`, `--q8-group`, `--download`, `--revision`, `--device`, `--dtype`, `--voice`, `--language`, `--instruct`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--max-new-tokens`, `--seed`, `--output-chunk-ms`, `--energy-gain`, `--warmup`, `--asr-model`, `--asr-language`, `--asr-max-new-tokens`, `--asr-context`, `--log-level`). CLI flags override environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `QWEN3TTS_URI` | `tcp://0.0.0.0:10200` | Wyoming listen URI (`tcp://`, `unix://`, or `stdio://`) |
| `QWEN3TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Model source: HF repo id (downloaded into the model dir) or a local model directory in any variant |
| `QWEN3TTS_MODEL_DIR` | `data/models` (`/data/models` in docker) | Cache for downloads (`sources/`) and conversions (`converted/`) |
| `QWEN3TTS_VARIANT` | `auto` | Serve this variant: `auto` detect, or `bf16`/`lite`/`q8` (converts once when needed) |
| `QWEN3TTS_CONVERT` | `true` | Allow one-time conversion into the requested variant |
| `QWEN3TTS_KEEP_SET` | `latin` | Vocabulary keep-set when converting (`latin` or `ml`) |
| `QWEN3TTS_ST_DTYPE` | `float16` | Speech-tokenizer decoder storage dtype when converting |
| `QWEN3TTS_Q8_GROUP` | `64` | Inputs per int8 scale group when converting to q8 |
| `QWEN3TTS_COMPILE` | `false` (`true` in the Docker image) | torch.compile the per-step decoder stacks (Inductor/Triton); first inference compiles — minutes, then cached. Needs `gcc` (CUDA/Triton) or `g++` (CPU) at runtime |
| `QWEN3TTS_COMPILE_MODE` | `default` | `default` (fusion only, safest), `reduce-overhead` (adds CUDA graphs; may not engage with the growing KV cache), `max-autotune` (longest compile) |
| `QWEN3TTS_COMPILE_DYNAMIC` | `true` | Compile with dynamic shapes so growing decode lengths don't recompile per step |
| `QWEN3TTS_DOWNLOAD` | `auto` | `auto` downloads when missing, `always` re-downloads at startup, `never` fails if missing |
| `QWEN3TTS_REVISION` | *(empty)* | Hugging Face revision to pin |
| `HF_TOKEN` | *(empty)* | Passed through to `huggingface_hub` for gated or rate-limited downloads |
| `QWEN3TTS_DEVICE` | `auto` | `auto` prefers CUDA and falls back to CPU; `cuda` fails fast; `cpu` forces CPU |
| `QWEN3TTS_DTYPE` | `auto` | `auto` = bfloat16 on CUDA / float32 on CPU; or force `bfloat16`/`float16`/`float32` |
| `QWEN3TTS_VOICE` | *(empty = first speaker)* | Default speaker id for requests without one (e.g. `aiden`, `narrator`) |
| `QWEN3TTS_LANGUAGE` | *(empty = auto-detect)* | Default language for requests without one |
| `QWEN3TTS_INSTRUCT` | *(empty)* | Default style instruction for every request (e.g. `warm and gentle`) |
| `QWEN3TTS_TEMPERATURE` / `_TOP_K` / `_TOP_P` / `_REPETITION_PENALTY` | *(model defaults)* | Sampling overrides forwarded to `generate` |
| `QWEN3TTS_MAX_NEW_TOKENS` | *(model default, 8192)* | Generation cap |
| `QWEN3TTS_SEED` | *(empty = random per request)* | Deterministic seeding |
| `QWEN3TTS_OUTPUT_CHUNK_MS` | `200` | Max milliseconds of audio per `audio-chunk` event |
| `QWEN3TTS_ENERGY_GAIN` | `1.0` | Post-synthesis waveform gain (linear) |
| `QWEN3TTS_WARMUP` | `true` | Run a warmup synthesis + transcription at startup so the first request is fast |
| `QWEN3TTS_ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | STT model source (HF repo id or local dir); **empty disables STT**. `Qwen/Qwen3-ASR-1.7B` for higher accuracy |
| `QWEN3TTS_ASR_LANGUAGE` | *(empty = auto-detect)* | Default transcription language (BCP-47) |
| `QWEN3TTS_ASR_MAX_NEW_TOKENS` | *(package default)* | Transcription generation cap; raise for very long audio |
| `QWEN3TTS_ASR_CONTEXT` | `true` | Forward Wyoming `transcript_names`/`transcript_terms` as recognition bias |
| `QWEN3TTS_LOG_LEVEL` | `INFO` | Log level |

The input **text**, **voice name**, and **language** always arrive per request via the Wyoming protocol from Home Assistant — they are intentionally *not* environment variables.

## Development

Useful mise tasks:

| Task | What it does |
| --- | --- |
| `mise run install` | `uv sync` — install dependencies |
| `mise run lock` | Regenerate `uv.lock` (commit the result) |
| `mise run test` | `uv run pytest -m "not e2e and not docker"` — unit + integration |
| `mise run test-e2e` | E2E tests (see below) |
| `mise run lint` / `format` / `typecheck` | ruff / mypy |

Test markers:

- Default suite skips heavy tests: `pytest -m "not e2e and not docker"`.
- `QWEN3TTS_E2E=1 mise run test-e2e` runs e2e against a **real model** — set `QWEN3TTS_MODEL` to a local directory (e.g. `../qwen3-tts-scratch/narrator-tts-lite-q8` for an instant start, or any source which will be downloaded/converted once).
- `QWEN3TTS_DOCKER_E2E=1 mise run test-docker` builds and runs the Docker image end-to-end.

### CUDA on Windows

The PyPI `torch` wheel for Windows is CPU-only. For GPU inference install the
CUDA build into the venv manually after `uv sync`:

```sh
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 --python .venv/Scripts/python.exe
```

(`qwen-tts` requires torchaudio, so reinstall both from the CUDA index; use
`--reinstall-package torch --reinstall-package torchaudio` if plain install is
a no-op.) **`uv run` re-syncs the venv to the lockfile and would revert this**
— run with `uv run --no-sync` afterwards, or invoke `.venv/Scripts/python`
directly. The Docker image gets CUDA torch from the linux wheels
automatically.

### The transformers pin (qwen-tts vs qwen-asr)

`qwen-tts` pins `transformers==4.57.3` while `qwen-asr` pins `==4.57.6` — two
patch releases of the same minor that cannot be satisfied together. This
project forces `4.57.3` via `[tool.uv] override-dependencies` (qwen-tts's pin
wins; both qwen packages vendor their own modeling code, and the e2e suites
exercise both models in one process). If an ASR e2e test breaks after a
qwen-asr upgrade, revisit the override.

## Troubleshooting

- **First start is slow** — the TTS source is downloaded (~4.5 GB for 1.7B) and converted when `--variant` selects a converted variant (minutes, CPU), plus the ASR model (~1.5 GB). All cached under `--model-dir`. With `QWEN3TTS_COMPILE=1`, the first synthesis additionally compiles (minutes) and caches under `/data/compile-cache` (Docker) — make sure `/data` is a volume so the cache survives restarts.
- **`torch.compile` requests fail with `InvalidCxxCompiler` / `Failed to find C compiler`** — the runtime toolchain is missing (`gcc` for CUDA/Triton, `g++` for CPU Inductor). The Docker image ships both; the server detects a missing toolchain at load and falls back to eager with a log line. In a venv, install a C/C++ compiler or set `QWEN3TTS_COMPILE=0`.
- **GPU not used** — check startup logs; `device=cuda` fails fast with the reason, `auto` falls back to CPU (CPU-only torch build on Windows is the usual cause).
- **Voice list missing speakers** — voices come from `config.json → talker_config.spk_id`; finetunes that add speakers appear automatically.
- **Non-English text sounds wrong on a latin keep-set** — convert with `--keep-set ml` (Chinese/Japanese/Korean/Russian need the bigger keep-set); the server also logs a warning when input tokens fall outside the keep-set.
- **`RuntimeError: Failed to find C compiler` (Docker)** — PyPI linux torch bundles Triton, and torch ≥ 2.14's native-op registry routes tiny outer-product matmuls (the TTS RoPE step) to Triton kernels that JIT-compile with the system C compiler at runtime. The image (and `__main__.py`) sets `TORCH_DISABLE_NATIVE_JIT=1` so eager/cuBLAS is used instead — nothing in this server needs Triton (no `torch.compile`). Set the variable to `0` only if you enable `torch.compile`, and then also add a C toolchain to the image.
- **`sox: not found` warning** — the `sox` *python* package probing for its (unused) binary; harmless. The Docker image installs the `sox` binary purely to quiet the log line.
- **STT missing from Home Assistant** — the ASR model failed to load at startup (see the server logs); the server keeps running TTS-only and answers STT requests with `asr-disabled`.

## License

- This project: [Apache-2.0](LICENSE).
- Model weights: Apache-2.0 (Qwen3-TTS) — check each finetune's license.
- The [Wyoming](https://github.com/rhasspy/wyoming) library is MIT/Apache dual-licensed.
