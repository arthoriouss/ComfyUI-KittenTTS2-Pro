# Changelog

## 0.1.1 — ComfyUI workflow widget serialization fix

- Correct `widgets_values` order in Basic TTS and Longform example workflows by serializing ComfyUI's seed `control after generate` widget (`fixed`). This prevents `temperature`, `top_p`, and other controls from shifting into invalid values after import.
- Added regression tests to protect the serialized settings contract.
- Model inference remains unverified on an RTX 3090; all offline tests use lightweight mocks.

## 0.1.0 — developer preview

- 14 modular ComfyUI nodes: model, voices, reference, transcription, expression, settings, TTS, clone, longform, audio DSP and diagnostics.
- Runtime cache with explicit memory release; no model weights bundled.
- Native ComfyUI AUDIO and offline unit tests using a mocked official KittenML API.
- Windows Portable installer which protects the existing CUDA PyTorch build.
