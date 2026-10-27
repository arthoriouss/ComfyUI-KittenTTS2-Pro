"""Official KittenML API adapter, with ephemeral WAV references and clean metrics."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import time
import numpy as np

from .runtime import ModelSpec, get_model, runtime_lock
from .audio import join_audio


@contextmanager
def reference_file(reference: dict):
    """Only temporary copies are written; no microphone clips are permanently stored."""
    import soundfile as sf
    if not isinstance(reference, dict) or 'samples' not in reference:
        raise ValueError("Connect KittenTTS2 Prepare Reference before cloning")
    x = np.asarray(reference['samples'], dtype=np.float32).reshape(-1)
    if len(x) < 24000:
        raise ValueError("Reference is too short")
    with tempfile.TemporaryDirectory(prefix="kitten2_ref_") as tmp:
        path = Path(tmp) / "reference.wav"
        sf.write(str(path), x, int(reference.get('sample_rate', 24000)), subtype='PCM_16')
        yield str(path)


def make_kwargs(voice: str | None, reference_path: str | None,
                ref_text: str, settings: dict | None) -> dict:
    options = dict(settings or {})
    valid = {'preset', 'normalize', 'max_new_tokens', 'advanced',
             'top_k', 'top_p', 'min_p', 'temperature', 'use_reference_prompt'}
    extra = set(options) - valid
    if extra:
        raise ValueError(f"Unsupported KittenTTS2 setting(s): {', '.join(sorted(extra))}")
    if bool(voice) == bool(reference_path):
        raise ValueError("Select exactly one of built-in voice or reference")
    if reference_path:
        options['reference'] = reference_path
        if ref_text.strip():
            options['reference_text'] = ref_text.strip()
    else:
        options['voice'] = voice
    return options


def synthesize(spec: ModelSpec, text: str, voice: str = '', reference: dict | None = None,
               settings: dict | None = None, transcript: str = '', stream: bool = False):
    if not str(text).strip():
        raise ValueError("Text must not be empty")
    with runtime_lock():
        model = get_model(spec)
        voices = getattr(model, 'available_voices', None)
        if voice and voices is not None and voice not in voices:
            raise ValueError(f"Voice '{voice}' not available. Use 'KittenTTS2 List Voices' to inspect this model.")
        start = time.perf_counter()
        if reference is not None:
            with reference_file(reference) as path:
                kwargs = make_kwargs(None, path, transcript or reference.get('reference_text',''), settings)
                samples = list(model.generate_stream(text, **kwargs)) if stream else [model.generate(text, **kwargs)]
        else:
            kwargs = make_kwargs(voice or 'Bruno', None, '', settings)
            samples = list(model.generate_stream(text, **kwargs)) if stream else [model.generate(text, **kwargs)]
        chunks = [np.asarray(c, dtype=np.float32).reshape(-1) for c in samples]
        if not chunks or any(c.size == 0 for c in chunks):
            raise RuntimeError("KittenTTS2 returned an empty audio chunk")
        rate = int(model.sample_rate)
        if stream:
            # Stream chunks are intentionally exposed raw; ComfyUI can process each chunk.
            # A short, deterministic join is an *approximation* of generate() post-processing.
            merged = chunks[0]
            for segment in chunks[1:]:
                merged = join_audio(merged, segment, rate, gap_ms=120)
        else:
            merged = chunks[0]
        took = time.perf_counter()-start
        duration = len(merged)/rate
        stats = (f"KittenTTS 2 | {len(chunks)} chunk(s) | {duration:.2f}s audio / {took:.2f}s wall "
                 f"| RTF {took / max(duration, .001):.2f} | {rate} Hz "
                 f"| {spec.backend}/{spec.device}/{spec.weights}")
        return merged, chunks, rate, stats


def transcribe(spec: ModelSpec, reference: dict, language: str = "French") -> str:
    """Transcribe the prepared reference with an explicit Whisper language policy.

    kittenml 0.9.7's ReferenceTranscriber.__call__ unconditionally forces English.
    Its lazy ASR pipeline is accessed via _load() to support French and Auto.
    This private upstream API is isolated here and guarded for future upgrades.
    """
    supported = {"French": "french", "English": "english", "Auto": None}
    if language not in supported:
        raise ValueError(f"Unknown transcription language {language!r}; choose {list(supported)}")

    from .audio import resample_mono
    with runtime_lock():
        model = get_model(spec)
        tool = getattr(model, "transcriber", None)
        if not callable(tool):
            raise RuntimeError("Current KittenML backend exposes no 'transcriber'. Supply a manual transcript.")
        factory = getattr(tool, "_load", None)
        if not callable(factory):
            if language == "English":
                with reference_file(reference) as path:
                    return str(tool(path)).strip()
            raise RuntimeError("Installed kittenml version does not expose its Whisper pipeline. "
                               "Enter an exact French transcript in Voice Clone > reference_text_override, "
                               "or use kittenml==0.9.7.")
        raw = np.asarray(reference["samples"], dtype=np.float32).reshape(-1)
        sr = int(reference.get("sample_rate", 24000))
        if raw.size < sr:
            raise ValueError("Reference must contain at least 1 second of audio")
        samples = resample_mono(raw, sr, 16000)
        generate_kwargs = {"task": "transcribe"}
        if supported[language] is not None:
            generate_kwargs["language"] = supported[language]
        result = factory()(samples, generate_kwargs=generate_kwargs)
        transcript = str(result.get("text", "") if isinstance(result, dict) else result).strip()
        if not transcript:
            raise ValueError("Whisper returned an empty transcript; check the reference recording")
        return transcript
