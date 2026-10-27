# ComfyUI KittenTTS 2 Pro 🐱

[![CI](https://img.shields.io/badge/tests-pytest-blue)](.github/workflows/tests.yml)
[![License](https://img.shields.io/badge/wrapper-Apache--2.0-blue)](LICENSE)
[![KittenTTS](https://img.shields.io/badge/KittenML-V2-orange)](https://github.com/KittenML/KittenTTS)

Unofficial, local-first **ComfyUI** node suite for the official **KittenTTS 2 (1.7B)** voice model.
Built for modular TTS, multilingual voices, **true in-context voice cloning**, beta expressions and long-form chunk generation.

> **Developer preview 0.1.3.** The nodepack has unit tests with a mocked upstream model, but has **not yet been validated against real model weights in this environment**. A compatible `kittenml` release is required. No model weights are bundled.

[Documentation française](README_FR.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Responsible use](RESPONSIBLE_USE.md) · [Upstream KittenTTS 2](https://github.com/KittenML/KittenTTS)

## Highlights

- Direct official `from kittenml import KittenTTS` API; not the old KittenTTS 0.8 ONNX engine.
- Native ComfyUI `AUDIO` **[batch, channels, samples]** and 24 kHz model output.
- Built-in voice selection including **French**; retrieve all official voices from the `List Voices` node.
- Voice cloning from **Load Audio → Prepare Reference**, automatically resampled, with optional exact transcript or built-in Whisper transcription.
- Reusable settings: stable/expressive, reproducible seed, token budget, repetition penalty, optional advanced sampling.
- Official `[emotion]`, `<vocal event>` and `(((emphasis)))` syntax builder.
- Long-form `generate_stream`: joined AUDIO + raw chunk list for downstream pipelines (see caveat below).
- Minimal audio DSP for clean references, normalization, joining voice clips.
- Cache on ComfyUI model drive, lazy imports, explicit VRAM cleanup and diagnostics.
- Safe Windows Portable installer that protects ComfyUI's **existing torch/torchaudio builds**.
- Pytest/CI with mock model tests and example ComfyUI workflows.

## 15 nodes

| Node | Function |
|---|---|
| **Load Model** | CUDA/CPU selection, `packed`/`emb4`/`full` weights, `default`/`student_w4` decoder; PyTorch or Linux vLLM. |
| **Choose Voice** | Preset voice + optional exact name. |
| **List Voices** | Query all voices available in the loaded model. |
| **Prepare Reference** | Input AUDIO → 24 kHz mono / trim / peak normalize / max 5–30s / text. |
| **Transcribe Reference** | Whisper transcription with French/Auto/English language selection; a typed `KITTEN2_TRANSCRIPT` output plugs into Voice Clone. |
| **Expression Tags** | Emotion, vocal event and emphasized phrase. |
| **Generation Settings** | Reusable presets and official sampling controls. |
| **Generate TTS** | Named-voice speech synthesis. |
| **Voice Clone** | In-context cloning via a recording, with optional typed Whisper transcript or manual override. |
| **Longform Chunks** | `generate_stream` → joined output and chunk list. |
| **Polish Audio** | Trim, target peak, fades. |
| **Audio Speed / Tempo** | Adjust speech duration after synthesis; optionally preserve voice pitch. |
| **Join Audio** | Join two clips with a gap or crossfade (resample as needed). |
| **Diagnostics** | Environment / CUDA VRAM report without loading a model. |
| **Free Memory** | Discard cache and empty Kitten-model allocations. |

## Quick start: Windows Portable (RTX 3090 recommended settings)

1. Stop ComfyUI. Ensure its bundled Python has a **CUDA PyTorch >= 2.6** and matching torchaudio. This project does not install or replace Torch.
2. From `ComfyUI_windows_portable/ComfyUI/custom_nodes`, put the folder `ComfyUI-KittenTTS2-Pro` (or clone your new repository here).
3. Run `install_portable_windows.bat` **from the project folder**. A ComfyUI Manager install alone provides only lightweight node dependencies; the backend requires this explicit installer. Installer pins the *currently installed* Torch and torchaudio versions during dependency resolution and then installs `kittenml` without its dependency resolver. If an incompatibility is detected, it fails rather than replacing Torch.
4. Restart ComfyUI and search for `KittenTTS 2 Pro` in the node menu.
5. Drop the [basic TTS workflow](examples/basic_tts.json) into ComfyUI.

### Recommended RTX 3090 profile

```text
KittenTTS2 Load Model:
  device: cuda
  backend: pytorch
  weights: packed       # lossless; emb4 saves disk space with small quality cost
  decoder: default      # best quality; student_w4 is smaller, not significantly faster
  preload: true
KittenTTS2 Generate:
  voice_name: French    # important: source voice language should match target text
KittenTTS2 Generation Settings:
  preset: stable (try expressive afterwards)
  normalize_english_text: false  # for French
  seed: 17
```

> The upstream Python backend is designed for quality and integration, not guaranteed live inference speed. For accelerated GPU inference on Linux/WSL2, use KittenML's optional **vLLM** backend in a dedicated environment. Native Windows vLLM is not supported. The C++ engine is CPU-oriented and its CLI needs **offline reference preparation** for cloning; not a drop-in GPU streaming backend.

### Python venv (Linux / macOS / ComfyUI Desktop)

With the exact Python interpreter used by your running ComfyUI and CUDA Torch already installed:

```bash
cd ComfyUI/custom_nodes/ComfyUI-KittenTTS2-Pro
bash install_venv.sh
```

For vLLM on Linux, follow [KittenML's vLLM documentation](https://github.com/KittenML/KittenTTS/blob/main/docs/vllm.md) **before** selecting the `vllm` backend in our model loader. vLLM may need a dedicated process/environment because of `torch` and driver compatibility.

## Workflows

**French TTS**

```text
KittenTTS2 Load Model ──► KittenTTS2 Generate TTS ──► Preview Audio
                                      ▲
                        KittenTTS2 Generation Settings (optional)
```

**Voice cloning**

```text
Load Audio ──► KittenTTS2 Prepare Reference ──────────► KittenTTS2 Voice Clone ──► Preview Audio
                            │                                     ▲
                            └──► KittenTTS2 Transcribe Reference ─┘ (optional override)
KittenTTS2 Load Model ────────────────────────────────► Voice Clone
```

**Long-form**

```text
KittenTTS2 Load Model ──► KittenTTS2 Longform Chunks ──► Preview Audio (joined)
                                                └─────► raw chunk AUDIO list for mapped nodes
```

Load example JSON files from [`examples/`](examples/). The `voice_clone.json` and `voice_clone_with_whisper.json` workflows require selecting your own short recording in **Load Audio**. The latter transcribes first and passes a typed transcript to Voice Clone.


## French voice cloning and Whisper language (0.1.2)

- KittenML 0.9.7 forces `language="en"` inside its default Whisper transcriber. **Transcribe Reference** now offers **French** (default), **Auto**, or **English**, using its lazily loaded ASR pipeline. The private `_load()` compatibility boundary is isolated and checked; a future upstream version may require an update.
- **Voice Clone** now prioritizes manually entered `reference_text_override` over a connected transcript. Always transcribe the *reference recording only*, not the text you intend to synthesize; match any trimming/clipping in Prepare Reference.
- Compare `reference_prompt_policy=default` and `on` for speaker similarity; upstream reports that full in-context audio can help long text but destabilize short text. Do not promise better identity for every speaker.
- Use a clean 10–20 second recording of a single speaker; choose `packed` weights, `default` decoder and `normalize_english_text=false` for French output.

## IMPORTANT: what streaming means here

The official `generate_stream()` yields **completed sentence-sized audio chunks**, not individual codec tokens; short text can produce one chunk. In ComfyUI the graph must finish its node before downstream audio preview, so this is **not real-time microphone-to-speaker conversational streaming**. The raw chunks output is a ComfyUI mapped-list convenience, not live playback. The joined result uses a 120 ms pause and does **not** promise bit-identical joins versus upstream `generate()`.

## Model files, security and licensing

- Default Hugging Face cache: `ComfyUI/models/kittentts2/`.
- First model load downloads its necessary assets. Allow disk space for weights + decoder and dependencies.
- Do **not** put model weights, private voice samples or output audio into your Git repository.
- The **wrapper** is Apache-2.0; **KittenTTS 2 weights** have their own **Stellon Labs Community License**, **not** Apache-2.0. Read it before commercial use or redistribution.
- Voice reference is written as a temporary local WAV only for the KittenML inference call, then deleted.
- This project does not launch an HTTP server.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q .
```

Unit tests do not install model weights, ComfyUI or Torch/CUDA; a mocked KittenML API checks the bridge.

## Roadmap

- Real hardware verification on Windows ComfyUI Portable + RTX 3090, with GPU memory and real-time-factor measurements.
- Optional (explicit, opt-in) Whisper acceleration/alternative transcription.
- Optional vLLM bridge in a **separate service**, if native dependency conflicts arise.
- Visual UX extensions only after core nodes are stable.
- Versioned Registry release, install compatibility matrix and CI hardware checks.

## Credits

KittenTTS 2 and `kittenml` were developed by [KittenML](https://github.com/KittenML/KittenTTS). Wrapper is unofficial and not affiliated with KittenML/ComfyUI. See [NOTICE.md](NOTICE.md).

## Optional real GPU smoke test (after installation)

Stop ComfyUI and run with **its** Python, from the repository root:

```powershell
# Windows Portable: adjust path to the Python used by ComfyUI
..\..\..\python_embeded\python.exe scripts\smoke_test.py --device cuda --weights packed --voice French
```

The script writes `kitten-smoke.wav` to the working directory and reports wall time, audio duration and realtime factor. This is the recommended first **real model** test before announcing a stable release.

## GitHub publication

This is a standalone repository candidate named `ComfyUI-KittenTTS2-Pro` (no writes have been made to GitHub).
Before publishing to the ComfyUI Registry, set a valid publisher ID in `pyproject.toml`, complete the real GPU test and follow [`docs/COMFY_REGISTRY_TEMPLATE.md`](docs/COMFY_REGISTRY_TEMPLATE.md).
The dependency installer pins the KittenML API version `kittenml==0.9.7` as its validated **interface target**; change it only after review and testing. This does not imply the actual model generation has been hardware-validated.

## Example workflow seed widget note (0.1.2)

Fixed missing `control after generate` serialized field in the Basic TTS and Longform examples. If you imported the old 0.1.0 examples, import the new JSON or manually correct `max_new_tokens`, `temperature`, and `top_p` (ComfyUI validates hidden/unused advanced sampling controls).

## Speech tempo and seed variations — 0.1.3

KittenTTS 2 exposes no native speech rate control. `advanced.chunk_gap_s` only changes the pauses between chunks. Use **KittenTTS 2 Pro · Audio Speed / Tempo** between generation and Preview Audio, starting at `speed=1.20`, `preserve_pitch=true` (librosa phase vocoder). Preserving pitch can introduce mild audio artifacts; disabling it also raises/lowers pitch. The audio sample rate is not changed.

**Seed is already exposed** in Generation Settings and is honored with `advanced_sampling=false`. Choose ComfyUI's `control after generate = randomize` to automatically try new seeds; choose `fixed` for repeatable tests. Try `examples/faster_french_seed_variations.json` for a working graph.
