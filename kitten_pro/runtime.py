"""Lazy, bounded inference runtime — no installs or background downloads on import.

The loader's lightweight descriptor does not own a CUDA model. The shared cache does,
which makes explicit unload work even if graph results retain descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import os
import platform
from pathlib import Path
from threading import RLock
from typing import Any

MODEL_REPO = "KittenML/kitten-tts-2"
_LOCK = RLock()
_MODELS: dict[tuple, Any] = {}


@dataclass(frozen=True)
class ModelSpec:
    repo: str = MODEL_REPO
    device: str = "auto"
    weights: str = "packed"
    decoder: str = "default"
    backend: str = "pytorch"
    cache_dir: str = ""
    vllm_gpu_utilization: float = 0.5

    def __post_init__(self):
        if not self.repo.strip():
            raise ValueError("Model ID or local model directory is required")
        if self.device not in ("auto", "cuda", "cpu", "mps"):
            raise ValueError(f"Unknown device: {self.device}")
        if self.weights not in ("packed", "emb4", "full"):
            raise ValueError(f"Unknown weight format: {self.weights}")
        if self.decoder not in ("default", "student_w4"):
            raise ValueError(f"Unknown decoder: {self.decoder}")
        if self.backend not in ("pytorch", "vllm"):
            raise ValueError(f"Unknown backend: {self.backend}")
        if self.backend == "vllm" and (os.name == "nt" or platform.system() != "Linux"):
            raise ValueError("vLLM is supported under Linux/WSL2, not native Windows/macOS")
        if not 0.1 <= float(self.vllm_gpu_utilization) <= 0.9:
            raise ValueError("vLLM GPU utilization must be between 0.1 and 0.9")

    @property
    def key(self) -> tuple:
        return (self.repo, self.device, self.weights, self.decoder,
                self.backend, self.cache_dir, self.vllm_gpu_utilization)


def get_model(spec: ModelSpec):
    """Construct once per configuration; model initialization may fetch weights."""
    with _LOCK:
        if spec.key in _MODELS:
            return _MODELS[spec.key]
        try:
            from kittenml import KittenTTS
        except ImportError as exc:
            raise RuntimeError(
                "KittenTTS 2 is missing. Stop ComfyUI and run the supplied installer "
                "using ComfyUI's own Python interpreter, then restart."
            ) from exc
        kwargs = {"weights": spec.weights}
        if spec.device != "auto":
            kwargs["device"] = spec.device
        if spec.cache_dir:
            kwargs["cache_dir"] = str(Path(spec.cache_dir).expanduser())
        if spec.decoder != "default":
            kwargs["decoder"] = spec.decoder
        if spec.backend == "vllm":
            kwargs["backend"] = "vllm"
            kwargs["vllm_options"] = {"gpu_memory_utilization": spec.vllm_gpu_utilization}
        try:
            model = KittenTTS(spec.repo, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to load KittenTTS 2 ({spec.repo}): {exc}") from exc
        _MODELS[spec.key] = model
        return model


def unload_models() -> int:
    with _LOCK:
        count = len(_MODELS)
        _MODELS.clear()
        gc.collect()
        try:
            import torch
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                cuda.empty_cache()
                cuda.ipc_collect()
        except (ImportError, RuntimeError, AttributeError):
            pass
        return count


def cache_count() -> int:
    with _LOCK:
        return len(_MODELS)


def runtime_lock():
    """Serialize calls into the same model, including Whisper, for GPU safety."""
    return _LOCK


def check_installed() -> str:
    from importlib.util import find_spec
    import sys
    lines = [f"Python: {sys.version.split()[0]}",
             f"Platform: {platform.platform()}",
             f"kittenml: {'found' if find_spec('kittenml') else 'NOT INSTALLED'}",
             f"Cached models: {cache_count()}"]
    try:
        import importlib.metadata as md
        lines.append(f"kittenml version: {md.version('kittenml')}")
    except Exception:
        pass
    try:
        import torch
        lines += [f"Torch: {torch.__version__}",
                  f"CUDA available: {torch.cuda.is_available()}"]
        if torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            lines.append(f"Allocated VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
            lines.append(f"Reserved VRAM: {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
    except Exception as exc:
        lines.append(f"Torch inspection unavailable: {exc}")
    return "\n".join(lines)
