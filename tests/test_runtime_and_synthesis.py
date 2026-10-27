from pathlib import Path
import sys
import types
import numpy as np
import pytest

from kitten_pro.runtime import ModelSpec, get_model, unload_models, cache_count
from kitten_pro.synthesis import synthesize, transcribe, make_kwargs, reference_file


class DummyKitten:
    started = 0

    def __init__(self, repo, **kwargs):
        DummyKitten.started += 1
        self.sample_rate = 24000
        self.available_voices = ['French', 'Bruno']
        self.repo = repo
        self.kwargs = kwargs
        self.received = []

    def generate(self, text, **kwargs):
        self.received.append((text, kwargs))
        assert bool(kwargs.get('voice')) ^ bool(kwargs.get('reference'))
        return np.ones(24000, dtype=np.float32) * .05

    def generate_stream(self, text, **kwargs):
        self.received.append((text, kwargs))
        assert bool(kwargs.get('voice')) ^ bool(kwargs.get('reference'))
        yield np.ones(12000, dtype=np.float32) * .05
        yield np.ones(12000, dtype=np.float32) * .05

    def transcriber(self, path):
        assert Path(path).is_file()
        return 'Bonjour test'


@pytest.fixture

def mocked_kitten(monkeypatch):
    import kitten_pro.runtime as runtime
    unload_models()
    DummyKitten.started = 0
    monkeypatch.setitem(sys.modules, 'kittenml', types.SimpleNamespace(KittenTTS=DummyKitten))
    yield
    unload_models()


def test_cache_and_cleanup(mocked_kitten):
    spec = ModelSpec(device='cpu')
    obj1, obj2 = get_model(spec), get_model(spec)
    assert obj1 is obj2
    assert DummyKitten.started == 1
    assert cache_count() == 1
    assert unload_models() == 1
    assert cache_count() == 0


def test_builtin_generation(mocked_kitten):
    audio, chunks, sr, report = synthesize(ModelSpec(), 'Bonjour', voice='French', settings={'normalize': False})
    assert sr == 24000 and len(audio) == 24000 and len(chunks) == 1
    assert 'RTF' in report
    assert get_model(ModelSpec()).received[-1][1]['voice'] == 'French'


def test_clone_and_temp_cleanup(mocked_kitten):
    ref = {'samples': np.ones(24000*6, dtype=np.float32)*.01, 'sample_rate':24000,
           'reference_text':'Bonjour original'}
    with reference_file(ref) as path:
        assert Path(path).exists()
    assert not Path(path).exists()
    audio, _, _, _ = synthesize(ModelSpec(), 'Je suis une voix', reference=ref)
    assert len(audio) == 24000
    kwargs = get_model(ModelSpec()).received[-1][1]
    assert kwargs['reference_text'] == 'Bonjour original'
    assert not Path(kwargs['reference']).exists()
    assert transcribe(ModelSpec(), ref) == 'Bonjour test'


def test_longform_chunks(mocked_kitten):
    audio, chunks, _, report = synthesize(ModelSpec(), 'Phrase A. Phrase B.', voice='Bruno', stream=True)
    assert len(chunks) == 2 and len(audio) == 12000 + 2880 + 12000
    assert '2 chunk(s)' in report


def test_invalid_voice_fails(mocked_kitten):
    with pytest.raises(ValueError, match='not available'):
        synthesize(ModelSpec(), 'Hello', voice='DoesNotExist')


def test_no_dual_speaker():
    with pytest.raises(ValueError):
        make_kwargs('Bruno', 'fake.wav', '', {})


def test_unknown_setting_rejected():
    with pytest.raises(ValueError):
        make_kwargs('Bruno', None, '', {'speed': 1.5})


def test_no_voices_downloaded_on_import(mocked_kitten):
    assert DummyKitten.started == 0
