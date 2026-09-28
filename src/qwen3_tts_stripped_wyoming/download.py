"""Hugging Face snapshot downloads for model sources."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# repo ids look like "org/name" (with optional subdirs); local paths contain a
# drive letter, a scheme, or an existing/absolute path separator prefix.
_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+(/[\w.-]+)*$")


def looks_like_repo_id(source: str) -> bool:
    """True when ``source`` should be treated as a Hugging Face repo id."""
    if Path(source).exists():
        return False
    return bool(_REPO_ID_RE.match(source.strip()))


def slugify_repo_id(repo_id: str) -> str:
    """Filesystem-safe directory name for a repo id."""
    return repo_id.strip().replace("/", "--")


def download_model(
    repo_id: str, model_dir: str | Path, *, revision: str | None = None, force: bool = False
) -> Path:
    """Download (or reuse) a model snapshot under ``model_dir/sources/<slug>``."""
    from huggingface_hub import snapshot_download

    target = Path(model_dir) / "sources" / slugify_repo_id(repo_id)
    if not force and (target / "config.json").is_file():
        _LOGGER.info("reusing downloaded snapshot at %s", target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("downloading %s from Hugging Face to %s", repo_id, target)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        # the qwen-tts loader also probes a speech_tokenizer/ subdir; keep the
        # layout verbatim
        ignore_patterns=["*.msgpack", "*.h5", "*.ckpt", "*.gguf", "*.onnx"],
    )
    if not (target / "config.json").is_file():
        raise RuntimeError(f"download finished but {target}/config.json is missing")
    return target
