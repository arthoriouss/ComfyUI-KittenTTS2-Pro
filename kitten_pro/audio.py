"""Independent, tested audio conversions for native ComfyUI AUDIO dictionaries."""
from __future__ import annotations

from math import gcd
import numpy as np


def mono_numpy(audio: dict) -> tuple[np.ndarray, int]:
    if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
        raise ValueError("Expected ComfyUI AUDIO: {'waveform': [B,C,T], 'sample_rate': int}")
    x = audio['waveform']
    if hasattr(x, 'detach'):
        x = x.detach().to('cpu').float().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[-1] < 1:
        raise ValueError(f"AUDIO must have shape [1, channels, samples], got {x.shape}")
    sr = int(audio['sample_rate'])
    if sr <= 0:
        raise ValueError("Invalid sample rate")
    x = np.nan_to_num(x[0].mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    return x, sr


def resample_mono(x: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return np.asarray(x, dtype=np.float32).copy()
    from scipy.signal import resample_poly
    d = gcd(int(source_sr), int(target_sr))
    return np.asarray(resample_poly(x, target_sr // d, source_sr // d), dtype=np.float32)


def to_comfy(x: np.ndarray, sample_rate: int) -> dict:
    import torch
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("No generated audio samples")
    y = np.clip(np.nan_to_num(y, nan=0.0, posinf=0, neginf=0), -1.0, 1.0)
    return {'waveform': torch.from_numpy(y.copy()).view(1, 1, -1), 'sample_rate': int(sample_rate)}


def trim_edges(x: np.ndarray, sr: int, threshold_db: float = -46.0, padding_ms: float = 90) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    indices = np.flatnonzero(np.abs(x) > 10 ** (float(threshold_db) / 20.0))
    if not indices.size:
        return x.copy()
    pad = int(sr * padding_ms / 1000)
    return x[max(0, indices[0] - pad):min(len(x), indices[-1] + 1 + pad)].copy()


def peak_normalize(x: np.ndarray, dbfs: float = -3.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-6:
        return x.copy()
    target = 10 ** (float(dbfs) / 20.0)
    return np.asarray(np.clip(x * target / peak, -1.0, 1.0), dtype=np.float32)


def fade_edges(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    n = min(len(x)//2, round(sr * max(0, fade_ms) / 1000))
    if n:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def join_audio(left: np.ndarray, right: np.ndarray, sr: int, gap_ms: int = 150,
               crossfade_ms: int = 0) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if not a.size:
        return b.copy()
    if not b.size:
        return a.copy()
    if crossfade_ms > 0:
        n = min(len(a), len(b), int(sr * crossfade_ms / 1000))
        if n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            mid = a[-n:] * (1-ramp) + b[:n] * ramp
            return np.concatenate((a[:-n], mid, b[n:])).astype(np.float32)
    gap = np.zeros(round(sr * max(gap_ms, 0) / 1000), dtype=np.float32)
    return np.concatenate((a, gap, b)).astype(np.float32)


def prepare_reference(audio: dict, max_seconds: float = 20, trim: bool = True,
                      normalize: bool = True) -> dict:
    x, sr = mono_numpy(audio)
    x = resample_mono(x, sr, 24000)
    x = x[:round(float(max_seconds) * 24000)]
    if trim:
        x = trim_edges(x, 24000)
    if normalize:
        x = peak_normalize(x, -3.0)
    if len(x) < 24000:
        raise ValueError("Reference too short after trimming (need at least 1 s; ideally 5–30 s).")
    return {'samples': x, 'sample_rate': 24000, 'duration': len(x)/24000, 'reference_text': ''}


def change_tempo(x: np.ndarray, speed: float = 1.2, preserve_pitch: bool = True) -> np.ndarray:
    """Change speaking duration without changing sample rate.

    With preserve_pitch=True, use librosa's phase vocoder (speed >1 => shorter).
    Without pitch preservation, use bandlimited polyphase resampling: the
    original samples are played at a changed pace, so pitch changes too.
    Never confuse a changed sample_rate metadata value with a tempo change.
    """
    from fractions import Fraction

    rate = float(speed)
    if not np.isfinite(rate) or not 0.5 <= rate <= 2.0:
        raise ValueError("speed must be between 0.5x and 2.0x")
    audio = np.asarray(x, dtype=np.float32).reshape(-1)
    if not audio.size:
        raise ValueError("Cannot change tempo of empty audio")
    if rate == 1.0:
        return audio.copy()

    if preserve_pitch:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("Pitch-preserving tempo requires librosa (installed with kittenml)") from exc
        output = librosa.effects.time_stretch(audio, rate=rate)
    else:
        from scipy.signal import resample_poly
        # 1/rate changes duration and pitch but keeps AUDIO sample-rate metadata intact.
        factor = Fraction(1.0 / rate).limit_denominator(1000)
        output = resample_poly(audio, factor.numerator, factor.denominator)

    result = np.asarray(output, dtype=np.float32).reshape(-1)
    if not result.size:
        raise RuntimeError("Tempo processing returned empty audio")
    return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
