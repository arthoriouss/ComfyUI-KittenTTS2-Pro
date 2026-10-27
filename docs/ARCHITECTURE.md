# Architecture — KittenTTS 2 Pro

## Separation of concerns

- **`__init__.py`**: ComfyUI registration only. No model download or backend initialization during import.
- **`kitten_pro/nodes.py`**: ComfyUI V1 input and output contracts, typed ports, UI defaults and glue code.
- **`kitten_pro/runtime.py`**: immutable `ModelSpec` descriptors, thread-safe shared runtime cache, model lifetime and CUDA release.
- **`kitten_pro/synthesis.py`**: official KittenML API adapter, temporary reference WAV and generation metrics.
- **`kitten_pro/audio.py`**: native ComfyUI AUDIO validation and pure NumPy/SciPy audio utilities.
- **`scripts/install_backend.py`**: dependency installation and Torch/Torchaudio freeze; not executed by nodes.
- **`scripts/smoke_test.py`**: external real-model test, measure runtime on actual hardware.

## Loading and lifetime

```text
ComfyUI model-loader
   | creates immutable ModelSpec (no CUDA model embedded in graph)
   v
runtime.get_model(spec) — shared RLock — per-spec cache
   | lazy import kittenml (at first preload/generation)
   v
KittenTTS(repo, device=..., weights=..., decoder=...)
   | generate() or generate_stream()
   v
NumPy samples → [1, 1, samples] float32 Torch tensor → native AUDIO
```

For clone: `AUDIO → Prepare Reference (mono 24 kHz) → ephemeral WAV → model.generate(reference=...) → WAV removed`. A reference transcript can come from the preparation node's text field, the node's override widget or the dedicated typed Whisper-transcription port.

Shared RLock currently serializes Kitten inference and built-in Whisper work; this optimizes stability in single-GPU setups, not throughput.

## Notes on real-time claims

This is a **graph nodepack**, not an asynchronous streaming server. `model.generate_stream()` yields complete audio chunks as documented upstream; ComfyUI consumers execute only after the upstream node returns. The list output is intended for per-chunk downstream processing, not live microphone output. Real-time conversation requires a separate generation/playback pipeline and latency measurements.

## Compatibility philosophy

- Never install Torch automatically from an import hook or a node execution.
- Do not patch global `transformers` / CUDA kernels at runtime.
- Use public KittenML methods only; no fragile model internals or monkeypatches.
- Do not force native Windows users onto unsupported vLLM.
- Do not store voice samples in the repository, persistent app settings or caches.
- Do not suggest the model's own weight license is Apache-2.0.
