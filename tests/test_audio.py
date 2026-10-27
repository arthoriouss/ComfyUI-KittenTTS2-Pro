import numpy as np
import pytest

from kitten_pro.audio import (mono_numpy, resample_mono, prepare_reference,
                              trim_edges, peak_normalize, fade_edges, join_audio, change_tempo)


def make_audio(seconds=6, sr=48000, channels=2):
    t = np.arange(int(seconds * sr)) / sr
    channel = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return {'waveform': np.stack([channel] * channels)[None, ...], 'sample_rate': sr}


def test_mono_and_reference_shape():
    x, sr = mono_numpy(make_audio())
    assert x.ndim == 1 and sr == 48000
    ref = prepare_reference(make_audio(), 6, trim=False, normalize=True)
    assert ref['sample_rate'] == 24000
    assert ref['samples'].shape == (144000,)
    assert max(abs(ref['samples'])) <= 1


def test_clip_20_seconds():
    ref = prepare_reference(make_audio(25), 20, trim=False, normalize=False)
    assert abs(ref['duration'] - 20) < 0.0001


def test_trim_and_fades():
    sr = 1000
    x = np.concatenate([np.zeros(200), np.ones(100), np.zeros(200)]).astype('float32')
    y = trim_edges(x, sr, padding_ms=0)
    assert len(y) == 100
    z = fade_edges(y, sr, 10)
    assert z[0] == 0 and z[-1] == 0


def test_silence_is_not_amplified():
    assert not peak_normalize(np.zeros(100)).any()


def test_join_gap_and_crossfade():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)
    assert len(join_audio(a, b, 1000, gap_ms=50)) == 250
    assert len(join_audio(a, b, 1000, crossfade_ms=20)) == 180


def test_resample():
    x = np.ones(320, dtype=np.float32)
    assert len(resample_mono(x, 16000, 24000)) == 480


def test_invalid_batch_rejected():
    with pytest.raises(ValueError):
        mono_numpy({'waveform': np.ones((2, 1, 100)), 'sample_rate': 24000})


def test_tempo_pitch_preserving_duration_and_frequency():
    pytest.importorskip("librosa")  # The real kittenml backend provides this optional audio dependency.
    sr = 24000
    samples = (0.15 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)
    output = change_tempo(samples, speed=1.25, preserve_pitch=True)
    assert abs(len(output) - int(len(samples) / 1.25)) <= 2
    assert output.dtype == np.float32 and np.max(np.abs(output)) <= 1
    middle = output[len(output) // 4: len(output) * 3 // 4]
    peak = np.argmax(np.abs(np.fft.rfft(middle * np.hanning(len(middle)))))
    peak_hz = peak * sr / len(middle)
    assert abs(peak_hz - 440) < 5


def test_tempo_without_pitch_preservation_changes_frequency():
    sr = 24000
    samples = (0.15 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)
    faster = change_tempo(samples, speed=1.5, preserve_pitch=False)
    middle = faster[len(faster) // 4: len(faster) * 3 // 4]
    peak = np.argmax(np.abs(np.fft.rfft(middle * np.hanning(len(middle)))))
    peak_hz = peak * sr / len(middle)
    assert abs(peak_hz - 660) < 10


def test_tempo_input_validation():
    with pytest.raises(ValueError):
        change_tempo(np.ones(100, dtype=np.float32), speed=0)
    with pytest.raises(ValueError):
        change_tempo(np.ones(100, dtype=np.float32), speed=float('nan'))
    with pytest.raises(ValueError):
        change_tempo(np.ones(0, dtype=np.float32), speed=1.5)
    assert np.array_equal(change_tempo(np.ones(100, dtype=np.float32), 1), np.ones(100))
