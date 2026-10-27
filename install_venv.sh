#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/install_backend.py
printf '\nRestart ComfyUI completely.\n'
