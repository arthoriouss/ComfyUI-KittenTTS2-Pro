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
    assert len(pack.NODE_CLASS_MAPPINGS) == 14
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
