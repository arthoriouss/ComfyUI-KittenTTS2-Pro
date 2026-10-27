import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

from kitten_pro.nodes import (NODES, KittenSettings, KittenExpression, KittenVoicePicker,
                              KittenPrepareReference, KittenAudioJoin, KittenLongform, KittenModelLoader)
from kitten_pro.runtime import ModelSpec


def test_registration_without_comfy():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('kitten2_mockpack', root / '__init__.py',
                                                   submodule_search_locations=[str(root)])
    pack = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pack
    spec.loader.exec_module(pack)
    assert len(pack.NODE_CLASS_MAPPINGS) == 15
    assert not pack.REGISTRATION_ERRORS
    assert all(cls.FUNCTION in cls.__dict__ for _, _, cls in NODES)


def test_settings_presets_and_seed():
    obj = KittenSettings()
    result = obj.build('expressive', False, 20, 1000, 1.2, False, 1.5, 0, .9, .01)[0]
    assert result['advanced'] == {'seed': 20, 'repetition_penalty': 1.2}
    assert 'temperature' not in result
    advanced = obj.build('expressive', False, 20, 1000, 1.2, True, 1.5, 0, .9, .01)[0]
    assert advanced['temperature'] == 1.5


def test_voice_picker():
    obj = KittenVoicePicker()
    assert obj.choose('French') == ('French', 'French')
    assert obj.choose('French', 'Bruno') == ('Bruno', 'Bruno')
    with pytest.raises(ValueError):
        obj.choose('Custom (type below)')


def test_expression_builder():
    result = KittenExpression().format_text('Elle gagne vraiment.', 'joyful', 'laugh', 'vraiment')[0]
    assert result == '[joyful] Elle gagne (((vraiment))). <laugh>'


def test_model_specs_refuse_windows_vllm(monkeypatch):
    import kitten_pro.runtime as runtime
    monkeypatch.setattr(runtime.os, 'name', 'nt')
    with pytest.raises(ValueError):
        ModelSpec(backend='vllm')


def test_model_load_is_lazy(monkeypatch):
    monkeypatch.setattr('kitten_pro.nodes.model_directory', lambda: '/tmp/kitten-test')
    spec, report = KittenModelLoader().load(preload=False, device='cuda')
    assert spec.repo == 'KittenML/kitten-tts-2'
    assert spec.cache_dir == '/tmp/kitten-test'
    assert 'Lazy' in report


def test_node_output_contracts():
    for _, _, cls in NODES:
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
        assert cls.CATEGORY == 'audio/KittenTTS 2 Pro'
    assert KittenLongform.OUTPUT_IS_LIST == (False, True, False)


def test_prepare_reference(monkeypatch):
    class FakeTensor:
        def __init__(self, data):
            self.array = data
        def view(self, *dims):
            self.array = self.array.reshape(*dims)
            return self
    class FakeTorch:
        @staticmethod
        def from_numpy(x):
            return FakeTensor(x)
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch())
    wave = np.ones((1, 1, 144000), dtype=np.float32) * .1
    ref, cleaned, status = KittenPrepareReference().prepare({'waveform': wave, 'sample_rate': 24000}, trim_silence=False)
    assert len(ref['samples']) == 144000
    assert cleaned['waveform'].array.shape == (1, 1, 144000)
    assert 'Ready' in status
    # Optional waveform tools also operate on native [B,C,T].
    joined = KittenAudioJoin().join({'waveform': wave, 'sample_rate': 24000}, {'waveform': wave, 'sample_rate': 24000})[0]
    assert joined['waveform'].array.shape[-1] > 144000


def test_clone_manual_transcript_overrides_whisper_and_prompt_policy(monkeypatch):
    import kitten_pro.nodes as nodes
    from kitten_pro.nodes import KittenClone

    captured = []

    def fake_synthesize(*args, **kwargs):
        captured.append(kwargs)
        return np.ones(24000, dtype=np.float32), [], 24000, 'ok'

    monkeypatch.setattr(nodes, 'synthesize', fake_synthesize)
    monkeypatch.setattr(nodes, 'to_comfy', lambda x, sr: {'samples': len(x), 'sample_rate': sr})
    ref = {'reference_text': 'texte du prepare reference'}
    node = KittenClone()
    node.run(None, ref, 'nouveau texte', reference_text_override=' bonjour exact ',
             transcript='Wrong English words', reference_prompt_policy='on')
    assert captured[-1]['transcript'] == 'bonjour exact'
    assert captured[-1]['settings']['use_reference_prompt'] is True
    node.run(None, ref, 'nouveau texte', transcript='Whisper texte', reference_prompt_policy='off')
    assert captured[-1]['transcript'] == 'Whisper texte'
    assert captured[-1]['settings']['use_reference_prompt'] is False
    node.run(None, ref, 'nouveau texte')
    assert captured[-1]['transcript'] == ''
    assert 'use_reference_prompt' not in captured[-1]['settings']
    with pytest.raises(ValueError):
        node.run(None, ref, 'texte', reference_prompt_policy='invalid')


def test_transcribe_node_default_french(monkeypatch):
    from kitten_pro.nodes import KittenTranscribe
    calls = []
    monkeypatch.setattr('kitten_pro.nodes.transcribe',
                        lambda model, reference, language: calls.append(language) or 'Bonjour')
    assert KittenTranscribe().run(None, {}) == ('Bonjour', 'Bonjour')
    assert calls == ['French']
    assert KittenTranscribe.INPUT_TYPES()['required']['language'][0] == ['French', 'Auto', 'English']


def test_tempo_node_preserves_rate_and_seed_control(monkeypatch):
    import kitten_pro.nodes as nodes
    from kitten_pro.nodes import KittenAudioTempo
    monkeypatch.setattr(nodes, 'to_comfy', lambda samples, rate: {'waveform': samples.reshape(1, 1, -1), 'sample_rate': rate})

    sr = 24000
    x = 0.1 * np.sin(2 * np.pi * 230 * np.arange(sr, dtype=np.float32) / sr)
    audio = {'waveform': x.reshape(1, 1, -1), 'sample_rate': sr}
    output, report = KittenAudioTempo().apply(audio, speed=1.25, preserve_pitch=False)
    assert output['sample_rate'] == sr
    assert abs(output['waveform'].shape[-1] - int(sr / 1.25)) <= 2
    assert '1.25x' in report
    spec = KittenSettings.INPUT_TYPES()['required']['seed'][1]
    assert spec['default'] == 17
    assert 'randomize' in spec['tooltip']
    assert 'seed' in KittenSettings().build('stable', False, 12345, 1000, 1.1, False, 1., 0, .95, 0.)[0]['advanced']


def test_clone_seed_modes_and_settings_precedence(monkeypatch):
    import kitten_pro.nodes as nodes
    from kitten_pro.nodes import KittenClone

    captured = []

    def fake_synthesize(*args, **kwargs):
        captured.append(kwargs)
        return np.ones(24000, dtype=np.float32), [], 24000, "test audio"

    monkeypatch.setattr(nodes, "synthesize", fake_synthesize)
    monkeypatch.setattr(nodes, "to_comfy", lambda audio, rate: {"length": len(audio), "rate": rate})
    clone = KittenClone()
    shared = {"normalize": False, "preset": "expressive", "advanced": {"seed": 100, "repetition_penalty": 1.25}}
    ref = {"reference_text": "Bonjour."}

    # Original workflows keep their Settings seed, without touching their config.
    _, report = clone.run(None, ref, "Test", settings=shared)
    assert captured[-1]["settings"]["advanced"]["seed"] == 100
    assert "seed=100 (settings)" in report

    # An explicit fixed seed overrides only the seed, retaining other settings.
    _, report = clone.run(None, ref, "Test", settings=shared, seed_mode="fixed", seed=42)
    opts = captured[-1]["settings"]
    assert opts["preset"] == "expressive"
    assert opts["advanced"] == {"seed": 42, "repetition_penalty": 1.25}
    assert shared["advanced"]["seed"] == 100
    assert "seed=42 (fixed)" in report

    # Seed control works without a connected Generation Settings node.
    _, report = clone.run(None, ref, "Test", seed_mode="fixed", seed=123)
    assert captured[-1]["settings"]["advanced"]["seed"] == 123
    assert "seed=123 (fixed)" in report

    random_seeds = iter([101, 202])
    monkeypatch.setattr(nodes.secrets, "randbelow", lambda n: next(random_seeds))
    first = clone.run(None, ref, "Test", settings=shared, seed_mode="randomize")[1]
    second = clone.run(None, ref, "Test", settings=shared, seed_mode="randomize")[1]
    assert "seed=101 (randomize)" in first
    assert "seed=202 (randomize)" in second
    assert captured[-1]["settings"]["advanced"]["seed"] == 202
    assert shared["advanced"]["seed"] == 100


def test_clone_random_mode_invalidates_comfy_cache():
    from kitten_pro.nodes import KittenClone
    import math
    assert math.isnan(KittenClone.IS_CHANGED(seed_mode="randomize"))
    assert KittenClone.IS_CHANGED(seed_mode="fixed") == "fixed"
    assert KittenClone.IS_CHANGED(seed_mode="settings") == "settings"


def test_clone_seed_inputs_and_invalid_modes(monkeypatch):
    from kitten_pro.nodes import KittenClone
    inputs = KittenClone.INPUT_TYPES()["optional"]
    assert inputs["seed_mode"][0] == ["settings", "fixed", "randomize"]
    assert inputs["seed_mode"][1]["default"] == "settings"
    assert inputs["seed"][0] == "INT"
    assert inputs["seed"][1]["max"] == 2147483647
    node = KittenClone()
    with pytest.raises(ValueError, match="seed_mode"):
        node.run(None, {}, "Test", seed_mode="unknown")
    with pytest.raises(ValueError, match="seed"):
        node.run(None, {}, "Test", seed_mode="fixed", seed=-1)
