"""Golden-output test of the full post-processing (stages 3b-10) on the synthetic floor of tests/fixtures: to_json()
must equal tests/golden/synthetic_floor.json exactly, and the DXF must hold the same entity counts per layer.
`pytest --update-golden` rewrites the file after an intended change; the diff of the golden file in the commit then
documents what moved."""
import json
from collections import Counter
from pathlib import Path

import ezdxf
import pytest

from fixtures.synthetic_floor import synthetic_floor
from fpx import pipeline

GOLDEN = Path(__file__).with_name("golden") / "synthetic_floor.json"


def dxf_counts(path):
    msp = ezdxf.readfile(path).modelspace()
    return {f"{layer} {kind}": n for (kind, layer), n in sorted(Counter((e.dxftype(), e.dxf.layer) for e in msp).items())}


def first_diff(a, b, path="$"):
    """Where two JSON values first differ, for a readable failure."""
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: {'missing' if k not in a else 'unexpected'}"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} != {len(b)} items"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
    elif a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def portable(value, out_dir):
    """The JSON with the output folder (a temp path) replaced by <out>, so that the golden file is machine-independent."""
    if isinstance(value, dict):
        return {k: portable(v, out_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, out_dir) for v in value]
    if isinstance(value, str) and str(out_dir) in value:
        return value.replace(str(out_dir), "<out>").replace("\\", "/")
    return value


def test_synthetic_floor_golden(tmp_path, update_golden):
    sheet = synthetic_floor()
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION + ("export",), out_dir=tmp_path)
    got = {"json": portable(json.loads((tmp_path / f"{sheet.id}.json").read_text(encoding="utf-8")), tmp_path),
           "dxf": dxf_counts(tmp_path / f"{sheet.id}.dxf")}
    if update_golden:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        pytest.skip(f"golden file rewritten: {GOLDEN}")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got["dxf"] == expected["dxf"], first_diff(got["dxf"], expected["dxf"], "dxf")
    assert got["json"] == expected["json"], first_diff(got["json"], expected["json"], "json")


def test_synthetic_floor_features():
    """The fixture exercises what its docstring says (guards the golden file against a silently weaker fixture)."""
    sheet = synthetic_floor()
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION)
    names = {r["name"] for r in sheet.rooms}
    assert {"Büro", "Sitzungszimmer", "Halle", "Korridor", "Archiv", "Lager", "Treppe"} <= names
    assert sum(r["stair"] for r in sheet.rooms) == 1 and sum(r.get("small_region", False) for r in sheet.rooms) == 1
    assert [v["kind"] for v in sheet.voids] == ["stair eye"]
    kinds = Counter(o["kind"] for o in sheet.openings)
    assert kinds["passage"] == 1 and kinds["interior opening"] == 1 and kinds["exterior door"] == 1
    assert any(len(r["poly"].interiors) == 1 and r["name"] == "Archiv" for r in sheet.rooms)       # the column island
    assert next(r for r in sheet.rooms if r["name"] == "Sitzungszimmer")["aoid_export"] == {"written": True, "aoid": "2051.ZZ.01.012"}
    assert sheet.wall_bridges == []                                                                 # the missed door is a leak, not a gap
