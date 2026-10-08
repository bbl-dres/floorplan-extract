"""End-to-end tests of stages 3b-10 with a known label map instead of the segmenter."""
import json

import ezdxf
import numpy as np
import pytest

from common import DATA
from conftest import needs_floors, two_rooms
from fpx import DEFAULT, pipeline


def test_two_rooms(tmp_path):
    sheet = two_rooms()
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION + ("export",), out_dir=tmp_path)
    assert len(sheet.rooms) == 2
    area = 4.55 * 5.4                                    # clear width x depth of each room
    for r in sheet.rooms:
        assert DEFAULT.m2(r["poly"].area) == pytest.approx(area, rel=0.01)
    west = min(sheet.rooms, key=lambda r: r["poly"].centroid.x)
    assert west["name"] == "Büro" and west["area_stamp"] == 24.57 and west["confidence"] == "high"
    assert sheet.gf_area == pytest.approx(60.0, rel=0.01)
    kinds = sorted(o["kind"] for o in sheet.openings)
    assert kinds == ["door", "window"]
    window = next(o for o in sheet.openings if o["kind"] == "window")
    assert window["exterior"] and window["width"] == pytest.approx(1.2, abs=0.05)
    assert list(sheet.connectivity.edges) == [(sheet.rooms[0]["id"], sheet.rooms[1]["id"])]

    data = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert [r["area"] for r in data["rooms"]] == pytest.approx([area, area], rel=0.01)
    x0, y0, x1, y1 = data["sheet"]["extent"]
    assert (x0, y0, x1, y1) == (0.0, 0.0, 12.0, 8.0)
    msp = ezdxf.readfile(tmp_path / "t.dxf").modelspace()
    assert len(msp.query('LWPOLYLINE[layer=="R_RAUMPOLYGON"]')) == 2
    assert len(msp.query('LWPOLYLINE[layer=="R_GESCHOSSPOLYGON"]')) == 1


def test_run_checks_arguments():
    sheet = two_rooms()
    with pytest.raises(ValueError):
        pipeline.run(sheet, stages=("segment",))         # no model
    with pytest.raises(ValueError):
        pipeline.run(sheet, stages=("export",))          # no out_dir
    with pytest.raises(ValueError):
        pipeline.run(sheet, stages=("no such stage",))


@needs_floors
def test_oracle_swiss_dwellings():
    """Stages 3b-9 on perfect labels of 10 held-out Swiss Dwellings floors. A regression guard, not a target: rooms
    split only by open-plan boundaries (kitchen/living) cannot be found from walls and doors alone."""
    from fpeval.oracle import floors, match, oracle_sheet
    tot, errs = np.zeros(3), []
    for rec in floors(DATA / "floors-test.pkl", 10):
        sheet, ref = oracle_sheet(rec)
        pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION)
        r = match([r["poly"] for r in sheet.rooms + sheet.fragments], [p for _, p in ref])
        tot += (r["matched"], r["ref"], r["pred"])
        errs += r["area_err_pct"]
    recall, precision = tot[0] / tot[1], tot[0] / tot[2]
    print(f"oracle: recall {recall:.3f} precision {precision:.3f} median area error {np.median(errs):+.2f}%")
    assert recall >= 0.80 and precision >= 0.80
    assert abs(np.median(errs)) <= 3.0
