"""ComfyUI V1-compatible custom nodes: pure registration, lazy KittenTTS2 imports."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .audio import (mono_numpy, resample_mono, to_comfy, trim_edges, peak_normalize,
                    fade_edges, join_audio, prepare_reference, change_tempo)
from .runtime import ModelSpec, MODEL_REPO, get_model, unload_models, runtime_lock, check_installed
from .synthesis import synthesize, transcribe

CATEGORY = "audio/KittenTTS 2 Pro"
VOICES = ["French", "Bruno", "Bella", "Jasper", "Luna", "Rosie", "Hugo", "Kiki", "Leo",
          "Matthew", "Dante", "Claire", "Maeve", "Marcus", "Victor", "Willow",
          "German", "Spanish", "Italian", "Portuguese", "Arabic", "Chinese",
          "Custom (type below)"]
EMOTIONS = ["none", "angry", "contemplative", "excited", "joyful", "mundane",
            "nervous", "sad", "stern", "surprised", "tender"]
EVENTS = ["none", "gasp", "giggle", "growl", "gulp", "laugh", "pause", "scoff", "sigh", "sob", "um"]


def model_directory() -> str:
    try:
        import folder_paths
        path = Path(folder_paths.models_dir) / "kittentts2"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except (ImportError, AttributeError):
        return ""


class KittenModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "device": (["auto", "cuda", "cpu"], {"default": "cuda"}),
            "weights": (["packed", "emb4", "full"], {"default": "packed"}),
            "decoder": (["default", "student_w4"], {"default": "default"}),
            "backend": (["pytorch", "vllm"], {"default": "pytorch", "tooltip": "vLLM requires Linux/WSL2; use PyTorch on native Windows."}),
            "preload": ("BOOLEAN", {"default": True, "tooltip": "Load and optionally download weights now, otherwise load on first generation."}),
        }, "optional": {
            "repository": ("STRING", {"default": MODEL_REPO}),
            "cache_directory": ("STRING", {"default": "", "tooltip": "Optional Hugging Face cache root; otherwise ComfyUI/models/kittentts2."}),
            "vllm_gpu_utilization": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
        }}

    RETURN_TYPES = ("KITTEN2_MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load and cache KittenTTS 2 (1.7B). RTX 3090: CUDA + packed + default decoder. No model files shipped."

    def load(self, device="cuda", weights="packed", decoder="default", backend="pytorch", preload=True,
             repository=MODEL_REPO, cache_directory="", vllm_gpu_utilization=.5):
        repo = repository.strip() or MODEL_REPO
        path = cache_directory.strip() or model_directory()
        spec = ModelSpec(repo=repo, device=device, weights=weights, decoder=decoder,
                         backend=backend, cache_dir=path, vllm_gpu_utilization=vllm_gpu_utilization)
        msg = f"KittenTTS2 {repo} | {backend}/{device} | {weights}/{decoder} | "
        if preload:
            model = get_model(spec)
            msg += f"Loaded | {len(getattr(model, 'available_voices', []))} built-in voices"
        else:
            msg += "Lazy load (no weights loaded yet)"
        return (spec, msg)


class KittenVoicePicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preset": (VOICES, {"default": "French"})},
                "optional": {"custom_voice": ("STRING", {"default": "", "tooltip": "Name from List Voices; overrides the preset if provided."})}}

    RETURN_TYPES = ("KITTEN2_VOICE", "STRING")
    RETURN_NAMES = ("voice", "name")
    FUNCTION = "choose"
    CATEGORY = CATEGORY
    DESCRIPTION = "Choose a built-in voice; French and other language-named voices should be used for non-English text."

    def choose(self, preset="French", custom_voice=""):
        value = custom_voice.strip() or preset
        if value == "Custom (type below)":
            raise ValueError("Enter a voice name in custom_voice")
        return value, value


class KittenListVoices:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("KITTEN2_MODEL",)}}

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("voices", "count")
    FUNCTION = "list_voices"
    CATEGORY = CATEGORY
    DESCRIPTION = "List every official built-in voice exposed by the loaded checkpoint."

    def list_voices(self, model):
        with runtime_lock():
            voices = list(get_model(model).available_voices)
        return "\n".join(voices), len(voices)


class KittenPrepareReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "max_seconds": ("FLOAT", {"default": 20.0, "min": 5.0, "max": 30.0, "step": 1}),
            "trim_silence": ("BOOLEAN", {"default": True}),
            "normalize_peak": ("BOOLEAN", {"default": True}),
        }, "optional": {"reference_text": ("STRING", {"default": "", "multiline": True,
                                                      "tooltip": "Exact spoken transcript, if known. Speeds up cloning and often improves accuracy."})}}

    RETURN_TYPES = ("KITTEN2_REFERENCE", "AUDIO", "STRING")
    RETURN_NAMES = ("reference", "prepared_audio", "status")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = "Convert Load Audio to mono 24 kHz, trim edges and clip to <= 30s. Does not denoise or alter voice timbre."

    def prepare(self, audio, max_seconds=20, trim_silence=True, normalize_peak=True, reference_text=""):
        ref = prepare_reference(audio, max_seconds, trim_silence, normalize_peak)
        ref['reference_text'] = reference_text.strip()
        status = f"Reference: {ref['duration']:.2f}s at 24kHz mono. "
        if ref['duration'] < 5:
            status += "Warning: <5s is below the recommended reference length."
        else:
            status += "Ready for cloning."
        return ref, to_comfy(ref['samples'], ref['sample_rate']), status


class KittenTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("KITTEN2_MODEL",),
            "reference": ("KITTEN2_REFERENCE",),
            "language": (["French", "Auto", "English"], {"default": "French",
                         "tooltip": "French forces a French transcript. Auto lets Whisper detect the language. "
                                    "KittenML 0.9.7's built-in transcription otherwise forces English."}),
        }}

    RETURN_TYPES = ("STRING", "KITTEN2_TRANSCRIPT")
    RETURN_NAMES = ("transcript_text", "transcript_for_clone")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Transcribe a reference in French, English, or automatic language detection (on demand)."

    def run(self, model, reference, language="French"):
        text = transcribe(model, reference, language=language)
        return text, text


class KittenExpression:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "Bonjour, comment allez-vous aujourd'hui ?"}),
            "emotion": (EMOTIONS, {"default": "none"}),
            "event": (EVENTS, {"default": "none"}),
        }, "optional": {"emphasize_phrase": ("STRING", {"default": "", "tooltip": "First literal match is wrapped in (((triple parentheses)))."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_text",)
    FUNCTION = "format_text"
    CATEGORY = CATEGORY
    DESCRIPTION = "Build official emotion / vocal event / emphasis tags; beta, effects vary by voice."

    def format_text(self, text, emotion="none", event="none", emphasize_phrase=""):
        result = str(text)
        phrase = str(emphasize_phrase).strip()
        if phrase:
            result = result.replace(phrase, f"((({phrase})))", 1)
        if event != "none":
            result += f" <{event}>"
        if emotion != "none":
            result = f"[{emotion}] {result}"
        return (result,)


class KittenSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (["stable", "expressive"], {"default": "stable"}),
            "normalize_english_text": ("BOOLEAN", {"default": False,
                                                      "tooltip": "False for French and other non-English languages; upstream normalizer is English-only."}),
            "seed": ("INT", {"default": 17, "min": 0, "max": 2147483647,
                             "tooltip": "Use control after generate → randomize in ComfyUI to explore new voice variants; this seed is passed even when advanced_sampling=false."}),
            "max_new_tokens": ("INT", {"default": 1000, "min": 250, "max": 3000, "step": 50}),
            "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            "advanced_sampling": ("BOOLEAN", {"default": False}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 2.0, "step": 0.05}),
            "top_k": ("INT", {"default": 0, "min": 0, "max": 200}),
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
            "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
        }}

    RETURN_TYPES = ("KITTEN2_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Reusable official KittenML generation settings. Changes sampling only when advanced_sampling is checked."

    def build(self, preset, normalize_english_text, seed, max_new_tokens, repetition_penalty,
              advanced_sampling, temperature, top_k, top_p, min_p):
        params = {"preset": preset, "normalize": bool(normalize_english_text),
                  "max_new_tokens": int(max_new_tokens),
                  "advanced": {"seed": int(seed), "repetition_penalty": float(repetition_penalty)}}
        if advanced_sampling:
            params.update(temperature=float(temperature), top_k=int(top_k),
                          top_p=float(top_p), min_p=float(min_p))
        return (params,)


def _effective_voice(voice_name, voice):
    return (voice or voice_name).strip()


class KittenGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("KITTEN2_MODEL",),
            "text": ("STRING", {"default": "Bonjour ! KittenTTS 2 fonctionne avec ComfyUI.", "multiline": True}),
            "voice_name": ("STRING", {"default": "French"}),
        }, "optional": {"voice": ("KITTEN2_VOICE",), "settings": ("KITTEN2_SETTINGS",)}}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Native TTS with named voice; exports standard ComfyUI AUDIO. Defaults to French and English normalizer OFF."

    def run(self, model, text, voice_name="French", voice=None, settings=None):
        audio, _, sr, stats = synthesize(model, text, voice=_effective_voice(voice_name, voice),
                                        settings=settings or {"normalize": False})
        return to_comfy(audio, sr), stats


class KittenClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("KITTEN2_MODEL",),
            "reference": ("KITTEN2_REFERENCE",),
            "text": ("STRING", {"default": "Bonjour, ceci est un test de clonage vocal.", "multiline": True}),
        }, "optional": {
            "reference_text_override": ("STRING", {"default": "", "multiline": True,
                "tooltip": "Exact words in reference audio. Takes priority over the connected Whisper transcript."}),
            "reference_prompt_policy": (["default", "on", "off"], {"default": "default",
                "tooltip": "Advanced: in-context reference audio may improve identity on longer text; "
                           "can cause repeats/drift on short text. 'default' is safest."}),
            "transcript": ("KITTEN2_TRANSCRIPT",),
            "settings": ("KITTEN2_SETTINGS",),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "True in-context voice cloning via reference WAV; requires permission of the voice owner."

    def run(self, model, reference, text, reference_text_override="", settings=None, transcript=None,
            reference_prompt_policy="default"):
        # Manual override must be authoritative even when a Whisper output is connected.
        actual_transcript = (reference_text_override or "").strip() or (transcript or "").strip()
        options = dict(settings or {"normalize": False})
        if reference_prompt_policy not in ("default", "on", "off"):
            raise ValueError("reference_prompt_policy must be default, on or off")
        if reference_prompt_policy != "default":
            options["use_reference_prompt"] = reference_prompt_policy == "on"
        audio, _, sr, stats = synthesize(model, text, reference=reference,
                                        transcript=actual_transcript,
                                        settings=options)
        return to_comfy(audio, sr), stats


class KittenLongform:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("KITTEN2_MODEL",),
            "text": ("STRING", {"default": "Une première phrase. Une deuxième phrase.", "multiline": True}),
            "voice_name": ("STRING", {"default": "French"}),
        }, "optional": {
            "voice": ("KITTEN2_VOICE",),
            "reference": ("KITTEN2_REFERENCE",),
            "settings": ("KITTEN2_SETTINGS",),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("joined_audio", "raw_chunks_list", "generation_report")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Long-form generate_stream; raw chunk list plus joined WAV. ComfyUI completes graph before playback: not token-level live audio."

    def run(self, model, text, voice_name="French", voice=None, reference=None, settings=None):
        audio, chunks, sr, stats = synthesize(
            model, text, voice="" if reference is not None else _effective_voice(voice_name, voice),
            reference=reference, settings=settings or {"normalize": False}, stream=True)
        return to_comfy(audio, sr), [to_comfy(c, sr) for c in chunks], stats


class KittenAudioPolish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "trim_silence": ("BOOLEAN", {"default": False}),
            "target_peak_dbfs": ("FLOAT", {"default": -3, "min": -24, "max": -0.1, "step": 0.5}),
            "fade_ms": ("FLOAT", {"default": 8, "min": 0, "max": 200, "step": 1}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "polish"
    CATEGORY = CATEGORY
    DESCRIPTION = "Conservative peak normalization, optional edge trimming and fades. Mono output."

    def polish(self, audio, trim_silence=False, target_peak_dbfs=-3.0, fade_ms=8):
        samples, sr = mono_numpy(audio)
        if trim_silence:
            samples = trim_edges(samples, sr)
        samples = fade_edges(peak_normalize(samples, target_peak_dbfs), sr, fade_ms)
        return (to_comfy(samples, sr),)


class KittenAudioTempo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "speed": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.05,
                                "tooltip": "Speech duration control: 1.20x is 20% faster. Does not affect KittenML sampling."}),
            "preserve_pitch": ("BOOLEAN", {"default": True,
                                          "tooltip": "Keep the same voice pitch with a phase vocoder (librosa). Uncheck for faster chipmunk-like pitch changes."}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "tempo_report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Change audio playback tempo after TTS without changing 24 kHz sample-rate metadata. 1.15–1.25x recommended."

    def apply(self, audio, speed=1.2, preserve_pitch=True):
        samples, sample_rate = mono_numpy(audio)
        output = change_tempo(samples, speed=speed, preserve_pitch=preserve_pitch)
        original_seconds = len(samples) / sample_rate
        final_seconds = len(output) / sample_rate
        report = (f"KittenTTS 2 tempo {speed:.2f}x | pitch preserved: {bool(preserve_pitch)} | "
                  f"{original_seconds:.2f}s → {final_seconds:.2f}s | {sample_rate} Hz")
        return to_comfy(output, sample_rate), report


class KittenAudioJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "first": ("AUDIO",),
            "second": ("AUDIO",),
            "gap_ms": ("INT", {"default": 180, "min": 0, "max": 10000, "step": 10}),
            "crossfade_ms": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 5,
                                     "tooltip": "When >0, crossfade replaces silence gap."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "join"
    CATEGORY = CATEGORY
    DESCRIPTION = "Join two mono clips, resampling the second to the first sample rate, with optional crossfade."

    def join(self, first, second, gap_ms=180, crossfade_ms=0):
        a, sr = mono_numpy(first)
        b, other_sr = mono_numpy(second)
        b = resample_mono(b, other_sr, sr)
        return (to_comfy(join_audio(a, b, sr, gap_ms, crossfade_ms), sr),)


class KittenDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refresh": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Check Python, KittenML, Torch/CUDA and live VRAM without loading model weights."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, refresh=True):
        report = check_installed()
        print("[KittenTTS2-Pro] Diagnostics:\n" + report)
        return (report,)


class KittenFreeMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Clear KittenTTS2 shared model cache and release unreferenced CUDA allocations."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, unload=True):
        number = unload_models() if unload else 0
        return (f"Unloaded {number} KittenTTS2 model(s). Other ComfyUI models untouched.",)


NODES = [
    ("KittenTTS2ModelLoader", "KittenTTS 2 Pro · Load Model", KittenModelLoader),
    ("KittenTTS2VoicePicker", "KittenTTS 2 Pro · Choose Voice", KittenVoicePicker),
    ("KittenTTS2ListVoices", "KittenTTS 2 Pro · List Voices", KittenListVoices),
    ("KittenTTS2PrepareReference", "KittenTTS 2 Pro · Prepare Reference", KittenPrepareReference),
    ("KittenTTS2Transcribe", "KittenTTS 2 Pro · Transcribe Reference", KittenTranscribe),
    ("KittenTTS2Expression", "KittenTTS 2 Pro · Expression Tags", KittenExpression),
    ("KittenTTS2Settings", "KittenTTS 2 Pro · Generation Settings", KittenSettings),
    ("KittenTTS2Generate", "KittenTTS 2 Pro · Generate TTS", KittenGenerate),
    ("KittenTTS2Clone", "KittenTTS 2 Pro · Voice Clone", KittenClone),
    ("KittenTTS2Longform", "KittenTTS 2 Pro · Longform Chunks", KittenLongform),
    ("KittenTTS2AudioPolish", "KittenTTS 2 Pro · Polish Audio", KittenAudioPolish),
    ("KittenTTS2AudioTempo", "KittenTTS 2 Pro · Audio Speed / Tempo", KittenAudioTempo),
    ("KittenTTS2AudioJoin", "KittenTTS 2 Pro · Join Audio", KittenAudioJoin),
    ("KittenTTS2Diagnostics", "KittenTTS 2 Pro · Diagnostics", KittenDiagnostics),
    ("KittenTTS2FreeMemory", "KittenTTS 2 Pro · Free Memory", KittenFreeMemory),
]
