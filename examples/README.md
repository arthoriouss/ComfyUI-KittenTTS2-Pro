# Example workflows

- `basic_tts.json`: RTX 3090 / French built-in voice / settings / Preview Audio.
- `voice_clone.json`: select an audio file in `Load Audio`, prepare it, clone the reference.
- `voice_clone_with_whisper.json`: same as clone but **transcribes the reference** and forwards a typed transcript (slower first run).
- `longform.json`: run official sentence-chunk `generate_stream()`; joined output to Preview Audio.

Drop JSON onto ComfyUI canvas. Re-select WAV in the clone example before queuing.
Widget values and node signatures can change with future ComfyUI versions; if inputs display as disconnected, rebuild the links using the same diagram in README.md.

- `faster_french_seed_variations.json`: French TTS → pitch-preserving 1.2× tempo → Preview Audio; seed auto-randomized after each queued generation.

**Voice Clone seed (v0.1.4):** `voice_clone.json` and `voice_clone_with_whisper.json` now start with `seed_mode=randomize` directly on `Voice Clone`, without needing a Generation Settings node. ComfyUI's separate `control after generate` widget should remain `fixed` for the built-in `seed` field; `seed_mode` is what does the randomization. Use `fixed` to repeat a variant or `settings` to inherit a connected settings seed. The `generation_report` output displays the seed actually used.
