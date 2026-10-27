# Troubleshooting

**`No module named kittenml`**: Run the supplied installer while ComfyUI is stopped, using the SAME Python interpreter ComfyUI runs. Restart fully.

**CUDA unavailable / CPU selected by mistake**: `KittenTTS 2 Pro · Diagnostics` should show `CUDA available: True`. Check ComfyUI's Torch CUDA wheel and driver, not just a standalone Python installation. Do not install PyTorch from generic PyPI over ComfyUI's CUDA wheel.

**Version resolution / Transformers conflict**: KittenTTS2 requires recent `transformers` and `torch>=2.6`; other extensions may pin incompatible versions. Clone your ComfyUI environment rather than forcing an upgrade. The installer prevents `torch` / `torchaudio` version changes but cannot guarantee every custom node stays compatible.

**`No module named s3tokenizer` / `diffusers` / `omegaconf`**: Re-run `scripts/install_backend.py`. It installs the non-Torch upstream dependencies from KittenML 0.9.7.

**Out of memory**: First close other GPU workloads and run `Free Memory`. Consider `weights=emb4` and `decoder=student_w4` for reduced on-disk model or decoder footprint, but remember `emb4` mainly reduces download size, not necessarily runtime VRAM and `student_w4` is not much faster. For a dedicated Linux/WSL2 backend, see upstream vLLM docs.

**French sounds English**: Set `voice_name=French` and `normalize_english_text=false`. Voice conditioning strongly affects language and accent.

**Voice does not match**: Use 5–30 seconds, one speaker, clean dry speech; supply the verbatim reference transcript to avoid extra automatic Whisper work.

**No audio from Clone**: Reference must be connected from `Prepare Reference`, not raw ComfyUI `AUDIO` directly; connect `prepared_audio` only to preview nodes.

**Short text produces only one chunk**: KittenML upstream states that `generate_stream()` is *sentence chunk streaming*. This is expected.

**Attempting vLLM on native Windows**: Unsupported by upstream. Switch to `pytorch` or run ComfyUI with Linux/WSL2 in a separate, compatible environment.

**ComfyUI V3 API / node registration**: The custom-node interface is backward compatible in typical installations, but this preview is not tested on every nightly. Provide Python version, ComfyUI commit, Torch version, full stack trace and node version when filing an issue (without any private voice samples).
