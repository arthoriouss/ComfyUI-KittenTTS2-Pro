#!/usr/bin/env python3
"""Careful installer. Run while ComfyUI is stopped, using its Python interpreter.

Never upgrades Torch/Torchaudio to satisfy a dependency; fails on incompatible environments.
"""
from __future__ import annotations

import importlib.metadata as md
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# Upstream KittenML 0.9.7 direct non-torch dependencies, per pyproject.toml.
# Includes heavy optional upstream transcriber and audio decoder dependencies.
UPSTREAM_DEPENDENCIES = [
    "transformers>=4.55", "safetensors", "numpy", "scipy", "soundfile", "librosa",
    "huggingface_hub", "brotli", "einops", "diffusers", "omegaconf", "hydra-core",
    "s3tokenizer", "resemble-perth", "pyloudnorm", "kitten-text-processing>=0.1.0",
    "espeakng_loader", "phonemizer", "onnxruntime",
]


def version(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="KittenTTS 2 safe installer")
    ap.add_argument("--dry-run", action="store_true", help="Print planned install without changing Python")
    args = ap.parse_args()
    torch_version = version("torch")
    audio_version = version("torchaudio")
    if not torch_version:
        sys.exit("ERROR: ComfyUI's CUDA PyTorch is not installed. Install/repair ComfyUI first.")
    if not audio_version:
        sys.exit("ERROR: torchaudio is missing. Install a Torch-compatible torchaudio build first.")
    try:
        from packaging.version import Version
        if Version(torch_version.split('+')[0]) < Version("2.6"):
            sys.exit(f"ERROR: KittenML 2 requires torch>=2.6; current torch={torch_version}. Fix the ComfyUI Python environment first.")
    except ImportError:
        pass
    print(f"Protected existing builds: torch={torch_version}, torchaudio={audio_version}")
    with tempfile.TemporaryDirectory(prefix="kitten_constraints_") as tmp:
        constraints = Path(tmp) / "constraints.txt"
        # Constraints prevent indirect dependency resolution replacing either preinstalled package.
        constraints.write_text(f"torch=={torch_version}\ntorchaudio=={audio_version}\n", encoding="utf-8")
        pip = [sys.executable, "-m", "pip"]
        first = pip + ["install", "--upgrade-strategy", "only-if-needed", "-c", str(constraints), *UPSTREAM_DEPENDENCIES]
        second = pip + ["install", "--no-deps", "kittenml==0.9.7"]
        if args.dry_run:
            print("Would run:", " ".join(first))
            print("Would run:", " ".join(second))
            return
        run(first)
        run(second)
    if version('torch') != torch_version or version('torchaudio') != audio_version:
        sys.exit("ERROR: Torch version changed unexpectedly. Repair your ComfyUI environment.")
    print("KittenTTS2 backend installed. Restart ComfyUI completely.")
    print("NOTE: transformers/diffusers requirements may conflict with existing custom nodes; use a separate ComfyUI instance if needed.")


if __name__ == "__main__":
    main()
