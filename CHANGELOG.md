# Changelog

## 0.1.4 — Direct seed on Voice Clone

- `Voice Clone` now has an optional `seed` (0–2147483647) and `seed_mode`: `settings` (backward-compatible default), `fixed`, or `randomize`.
- `randomize` uses a fresh seed for each queued graph execution and marks the node changed to bypass ComfyUI result caching; no Generation Settings node needed.
- `generation_report` includes the actual seed used, making variants reproducible.
- When `fixed` or `randomize`, only the seed overrides an optional connected Generation Settings node; preset, temperature and other options remain untouched.
- Updated both Voice Clone example workflows; regression tests cover serialization, seed precedence, randomization and caching.
- Developer preview, tested with mock KittenML backend only (real model inference still needs RTX 3090 testing).

## 0.1.3 — Tempo and reproducible seed variations

- Add `KittenTTS 2 Pro · Audio Speed / Tempo` node, 0.50–2.00x with pitch-preserving `librosa` time-stretch (or pitch-changing resampling).
- Add `examples/faster_french_seed_variations.json` with `1.20x` and ComfyUI `control after generate = randomize`.
- Make the Generation Settings seed's tooltip explicit: it already works with advanced sampling disabled.
- Native KittenML 2 has no global speech-rate argument; `chunk_gap_s` only changes between-chunk pauses.
- Tests include pitch/duration checks, node registration, workflow connectivity; still no actual GPU voice-quality validation.

## 0.1.2 — French cloning and transcription fidelity

- Force French transcription by default in Transcribe Reference instead of KittenML 0.9.7\'s hard-coded English; allow Auto and English.
- Give manual `reference_text_override` priority over connected Whisper transcript.
- Expose reference prompt policy as an optional controlled experiment (`default`/`on`/`off`) for speaker similarity.
- Update cloning examples, docs, and tests.
- Still a developer preview, not hardware-validated.

## 0.1.1 — ComfyUI workflow widget serialization fix

- Correct `widgets_values` order in Basic TTS and Longform example workflows by serializing ComfyUI's seed `control after generate` widget (`fixed`). This prevents `temperature`, `top_p`, and other controls from shifting into invalid values after import.
- Added regression tests to protect the serialized settings contract.
- Model inference remains unverified on an RTX 3090; all offline tests use lightweight mocks.

## 0.1.0 — developer preview

- 14 modular ComfyUI nodes: model, voices, reference, transcription, expression, settings, TTS, clone, longform, audio DSP and diagnostics.
- Runtime cache with explicit memory release; no model weights bundled.
- Native ComfyUI AUDIO and offline unit tests using a mocked official KittenML API.
- Windows Portable installer which protects the existing CUDA PyTorch build.
