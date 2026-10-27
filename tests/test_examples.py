import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_example_files_links():
    for path in (ROOT / 'examples').glob('*.json'):
        doc = json.loads(path.read_text('utf-8'))
        nodes = {n['id']: n for n in doc['nodes']}
        assert nodes
        for link_id, source, output, target, input_index, data_type in doc['links']:
            assert source in nodes and target in nodes
            assert 0 <= output < len(nodes[source]['outputs'])
            assert 0 <= input_index < len(nodes[target]['inputs'])
            assert nodes[source]['outputs'][output]['type'] == data_type
            assert nodes[target]['inputs'][input_index]['type'] == data_type


def test_settings_workflow_widgets_serialization():
    # ComfyUI adds a frontend-only `control after generate` widget immediately
    # after the seed. Missing it shifts every later widget (e.g. temperature).
    for name in ("basic_tts.json", "longform.json"):
        doc = json.loads((ROOT / "examples" / name).read_text("utf-8"))
        settings = [n for n in doc["nodes"] if n["type"] == "KittenTTS2Settings"]
        assert len(settings) == 1
        values = settings[0]["widgets_values"]
        assert len(values) == 11
        assert values[3] == "fixed"
        assert values[4] >= 250  # max_new_tokens
        assert 0.3 <= values[7] <= 2.0  # temperature
        assert 0.5 <= values[9] <= 1.0  # top_p
