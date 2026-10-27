# Contributing

Please include a reproducible issue or test for bug fixes. Do not add downloads to import-time code.
Install development dependencies with `python -m pip install -r requirements-dev.txt`; run `pytest -q`.
PRs should preserve native AUDIO [B,C,T] conventions, never replace ComfyUI's CUDA Torch during installation,
and follow official KittenML API rather than guesses about internal model classes.
