# Example workflows

- `basic_tts.json`: RTX 3090 / French built-in voice / settings / Preview Audio.
- `voice_clone.json`: select an audio file in `Load Audio`, prepare it, clone the reference.
- `voice_clone_with_whisper.json`: same as clone but **transcribes the reference** and forwards a typed transcript (slower first run).
- `longform.json`: run official sentence-chunk `generate_stream()`; joined output to Preview Audio.

Drop JSON onto ComfyUI canvas. Re-select WAV in the clone example before queuing.
Widget values and node signatures can change with future ComfyUI versions; if inputs display as disconnected, rebuild the links using the same diagram in README.md.
