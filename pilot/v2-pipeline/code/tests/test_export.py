"""Export conformance: the synthetic two-room plan (test_pipeline.two_rooms) through stages 3b-10 and the local
plan-check rules (fpx.conformance), plus the pieces that the two-room plan does not exercise on its own."""
import json

import ezdxf
import pytest
import shapely
from shapely.geometry import Polygon, box

from conftest import paint, two_rooms
from fpx import DEFAULT, pipeline
from fpx.attributes import aoid_export
from fpx.conformance import RULES, check, has_self_intersection, region, shoelace
from fpx.export import export, keyhole
from fpx.model import WALL

M = DEFAULT.px_per_m
PLAN_CHECK = {code for code, _, _ in RULES}


def run(sheet, tmp_path):
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION + ("export",), out_dir=tmp_path)
    data = json.loads((tmp_path / f"{sheet.id}.json").read_text(encoding="utf-8"))
    result = check(tmp_path / f"{sheet.id}.dxf")
    return data, {r["code"]: r for r in result["rules"]}, result


def failing(rules):
    return {c for c, r in rules.items() if r["status"] == "fail"}


def test_two_rooms_conform(tmp_path):
    """Every plan-check rule and every extra check passes, except AOID_001: the plan carries no AOID, and none is made up."""
    data, rules, result = run(two_rooms(), tmp_path)
    assert failing(rules) == {"AOID_001"} and rules["AOID_001"]["count"] == 2
    assert result["units"] == 4 and result["score"] == round(100 * 39 / 40)
    assert result["summary"]["floor_area_m2"] == pytest.approx(60.0, rel=0.01)
    assert all(not r["aoid_export"]["written"] for r in data["rooms"])
    assert any(q["check"] == "AOID" and q["element"] == "floor" for q in data["qa"])
    doc = ezdxf.readfile(tmp_path / "t.dxf")
    msp = doc.modelspace()
    assert not msp.query('TEXT[layer=="R_AOID"]')                              # no invented AOIDs
    assert [t.dxf.text for t in msp.query('TEXT[layer=="V_TEXT"]')] == ["Büro"]
    assert {t.dxf.style for t in msp.query("TEXT")} == {"ARIAL"} and doc.styles.get("ARIAL").dxf.font == "arial.ttf"
    hatches = msp.query('HATCH[layer=="A_SCHRAFFUR"]')
    assert hatches and all(h.dxf.solid_fill == 1 for h in hatches)            # 0.3 m walls: massive
    assert {w["construction"] for w in data["walls"]} == {"massive"}
    assert data["wall_construction"]["method"] == "heuristic"
    assert len(msp.query('DIMENSION[layer=="V_BEMASSUNG"]')) == 2


def test_aoid_written_only_when_read_and_unique(tmp_path):
    sheet = two_rooms()
    stamp = dict(sheet.text[0], text="0000.ZZ.01.012", box=(150, 225, 210, 237))     # third line of the west stamp
    sheet.text.append(stamp)
    data, rules, _ = run(sheet, tmp_path)
    west = min(data["rooms"], key=lambda r: shapely.geometry.shape(r["geometry"]).centroid.x)
    assert west["aoid"] == "0000.ZZ.01.012" and west["aoid_export"] == {"written": True, "aoid": "0000.ZZ.01.012"}
    assert failing(rules) == {"AOID_001"} and rules["AOID_001"]["count"] == 1        # the east room only
    aoids = ezdxf.readfile(tmp_path / "t.dxf").modelspace().query('TEXT[layer=="R_AOID"]')
    assert [t.dxf.text for t in aoids] == ["0000.ZZ.01.012"]
    assert any(q["check"] == "AOID" and q["element"] != "floor" for q in data["qa"])
    # read twice, or two in one room: written nowhere
    rooms = [{"id": "a", "aoids": ["0000.ZZ.01.001"]}, {"id": "b", "aoids": ["0000.ZZ.01.001"]},
             {"id": "c", "aoids": ["0000.ZZ.01.002", "0000.ZZ.01.003"]}, {"id": "d", "aoids": []}]
    aoid_export(rooms)
    assert not any(r["aoid_export"]["written"] for r in rooms)
    assert "not unique" in rooms[0]["aoid_export"]["reason"] and "2 AOIDs" in rooms[2]["aoid_export"]["reason"]


def test_small_regions_kept_and_flagged(tmp_path):
    """A walled 0.8 x 0.8 m shaft becomes a room flagged for review; a 0.4 x 0.4 m cavity (0.16 m²) is dropped. The east
    room around them is written as one polyline with the islands cut out."""
    sheet = two_rooms()
    paint(sheet, 8.0, 2.0, 9.3, 3.3, WALL)
    paint(sheet, 8.25, 2.25, 9.05, 3.05, 0)
    paint(sheet, 8.0, 4.5, 8.9, 5.4, WALL)
    paint(sheet, 8.25, 4.75, 8.65, 5.15, 0)
    data, rules, result = run(sheet, tmp_path)
    assert len(sheet.rooms) == 3 and sheet.fragments == []
    shaft = next(r for r in data["rooms"] if r["small_region"])
    assert shaft["area"] == pytest.approx(0.64, rel=0.05) and shaft["confidence"] == "low"
    assert shaft["review"]["flag"] == "small unlabelled region" and shaft["review"]["guess"] == "shaft"
    assert "0.25 m²" in shaft["review"]["reason"] and len(data["fragments"]) == 1
    assert failing(rules) == {"AOID_001"} and result["summary"]["rooms"] == 3
    east = max((r for r in data["rooms"] if shapely.geometry.shape(r["geometry"]).centroid.x > 6.15), key=lambda r: r["area"])
    assert len(shapely.geometry.shape(east["geometry"]).interiors) == 2
    assert result["summary"]["room_area_m2"] == pytest.approx(sum(r["area_polygon"] for r in data["rooms"]), abs=0.05)
    assert all(r["area_gross"] > r["area_net"] for r in data["rooms"])     # the wall share to the centre lines


def test_floor_polygon_with_void_cut_out(tmp_path):
    """A void over 5 m² is cut out of the GF as one continuous polyline (keyhole) and kept on R_RAUMPOLYGON-ABZUG."""
    sheet = two_rooms()
    pipeline.run(sheet, stages=pipeline.AFTER_SEGMENTATION)
    void = box(2 * M, 2 * M, 4.5 * M, 4.5 * M)                                       # 6.25 m², in the west room
    sheet.voids.append({"px": 0, "poly": void, "label": "Luftraum", "gf_deducted": True})
    sheet.gf = sheet.gf.difference(void)
    export(sheet, tmp_path)
    rules = {r["code"]: r for r in check(tmp_path / "t.dxf")["rules"]}
    assert failing(rules) == {"AOID_001"}
    msp = ezdxf.readfile(tmp_path / "t.dxf").modelspace()
    (gf,) = msp.query('LWPOLYLINE[layer=="R_GESCHOSSPOLYGON"]')
    ring = [p[:2] for p in gf.get_points("xy")]
    assert gf.closed and shoelace(ring) / 1e6 == pytest.approx(60.0 - 6.25, rel=0.01)
    (deduction,) = msp.query('LWPOLYLINE[layer=="R_RAUMPOLYGON-ABZUG"]')
    assert shoelace([p[:2] for p in deduction.get_points("xy")]) / 1e6 == pytest.approx(6.25, rel=0.001)


def test_keyhole():
    p = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)], [[(2, 2), (4, 2), (4, 4), (2, 4)], [(6, 5), (8, 5), (8, 6), (6, 6)]])
    ring = keyhole(p)
    assert shoelace(ring) == pytest.approx(p.area)
    assert not has_self_intersection(ring)
    assert region(ring).area == pytest.approx(p.area) and region(ring).geom_type == "Polygon"
    assert keyhole(box(0, 0, 1, 1)) == [tuple(c) for c in shapely.geometry.polygon.orient(box(0, 0, 1, 1)).exterior.coords[:-1]]


def test_wall_construction(tmp_path):
    """A 0.12 m wall stub drawn open (white core) is lightweight; the 0.3 m walls drawn solid are massive."""
    sheet = two_rooms()
    paint(sheet, 1.3, 2.4, 3.0, 2.52, WALL)                      # 2.85 m short of the next wall: no passage
    paint(sheet, 1.3, 2.44, 3.0, 2.48, WALL, ink=255)
    data, rules, _ = run(sheet, tmp_path)
    light = [w for w in data["walls"] if w["construction"] == "lightweight"]
    assert light and all(w["thickness"] < DEFAULT.massive_wall_min_thickness for w in light)
    assert all(w["construction"] == "massive" for w in data["walls"] if w["thickness"] >= 0.25)
    massive = shapely.unary_union([shapely.geometry.shape(g) for g in data["massive_wall_polygons"]])
    walls = shapely.unary_union([shapely.geometry.shape(g) for g in data["wall_polygons"]])
    assert 0 < massive.area < walls.area - 0.1                                        # the stub (0.2 m²) is not hatched
    assert failing(rules) == {"AOID_001"}


def test_checker_catches_violations(tmp_path):
    """The local checker reports what plan-check would: units, missing layers, open and crossing polygons, AOID
    problems, text layers and fonts, colours, hatches, frame and dimensions."""
    doc = ezdxf.new("R2018", units=6)                                                   # metres
    for name in ("R_RAUMPOLYGON", "R_AOID", "R_GESCHOSSPOLYGON", "A_SCHRAFFUR", "MY_LAYER"):
        doc.layers.add(name)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (4, 0), (4, 4), (0, 4)], close=True, dxfattribs={"layer": "R_RAUMPOLYGON"})
    msp.add_lwpolyline([(5, 0), (9, 0), (9, 4), (5, 4)], close=True, dxfattribs={"layer": "R_RAUMPOLYGON"})
    msp.add_lwpolyline([(10, 0), (14, 4), (14, 0), (10, 4)], close=True, dxfattribs={"layer": "R_RAUMPOLYGON"})   # bow tie
    msp.add_lwpolyline([(0, 6), (4, 6), (4, 9)], dxfattribs={"layer": "R_RAUMPOLYGON", "color": 1})              # open
    for x in (1, 6):
        msp.add_text("0000.ZZ.01.001", dxfattribs={"layer": "R_AOID"}).set_placement((x, 1))
    msp.add_text("Büro", dxfattribs={"layer": "R_AOID"}).set_placement((20, 20))
    msp.add_text("Note", dxfattribs={"layer": "MY_LAYER"}).set_placement((1, 2))
    h = msp.add_hatch(dxfattribs={"layer": "A_SCHRAFFUR"})
    h.set_pattern_fill("ANSI31")
    h.paths.add_polyline_path([(0, 0), (1, 0), (1, 1)], is_closed=True)
    path = tmp_path / "bad.dxf"
    doc.saveas(path)
    result = check(path)
    rules = {r["code"]: r for r in result["rules"]}
    expected = {"LAYER_004", "LAYER_005", "LAYER_006", "LAYER_008", "POLY_001", "POLY_007", "GPOLY_004", "AOID_002",
                "AOID_005", "GEOM_001", "TEXT_001", "TEXT_002", "STYLE_002", "LAYOUT_002", "DIM_001", "HATCH_001", "FPX_VALID"}
    assert expected <= failing(rules)
    assert {"POLY_002", "POLY_006", "GEOM_003", "LAYOUT_001"}.isdisjoint(failing(rules))
    assert result["score"] < 70 and set(rules) >= PLAN_CHECK
