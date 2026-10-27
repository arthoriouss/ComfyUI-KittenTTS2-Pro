"""Validate native node call shapes against a fake KittenML + fake torch."""
import sys
import types
import numpy as np

from kitten_pro.nodes import KittenGenerate, KittenClone, KittenLongform
from kitten_pro.runtime import ModelSpec, unload_models


class Tensor:
    def __init__(self, x): self.data = x
    def view(self, *dims): self.data = self.data.reshape(*dims); return self


class Torch:
    def from_numpy(self, x): return Tensor(x)


class Kitten:
    def __init__(self, repo, **kwargs):
        self.available_voices = ['French']
        self.sample_rate = 24000
    def generate(self, text, **kwargs):
        assert text
        return np.ones(24000, dtype=np.float32)*.05
    def generate_stream(self, text, **kwargs):
        yield np.ones(6000, dtype=np.float32)*.05
        yield np.ones(6000, dtype=np.float32)*.05


def test_node_paths(monkeypatch):
    unload_models()
    monkeypatch.setitem(sys.modules,'torch',Torch())
    monkeypatch.setitem(sys.modules,'kittenml',types.SimpleNamespace(KittenTTS=Kitten))
    spec=ModelSpec()
    voice, info=KittenGenerate().run(spec,'Bonjour!')
    assert voice['waveform'].data.shape == (1,1,24000)
    ref={'samples':np.ones(24000*6,dtype=np.float32)*.05,'sample_rate':24000}
    voice,info=KittenClone().run(spec,ref,'Je clone',transcript='Bonjour test')
    assert voice['waveform'].data.shape == (1,1,24000)
    joined,chunks,info=KittenLongform().run(spec,'A. B.')
    assert len(chunks)==2 and joined['waveform'].data.shape[-1] == 14880
    unload_models()
