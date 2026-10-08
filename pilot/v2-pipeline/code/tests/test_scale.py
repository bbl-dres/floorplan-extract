"""Scale cues and consensus (fpx.scale): parsing, pairing numbers with dimension lines, outlier rejection, consensus."""
import math

import cv2
import numpy as np
import pytest

from fpx import scale
from fpx.config import DEFAULT

INCH = 0.0254


@pytest.mark.parametrize("text, expected", [
    ("1:50", [(50, False)]),
    ("M 1:100", [(100, True)]),
    ("Mst. 1:200", [(200, True)]),
    ("Massstab 1:50", [(50, True)]),
    ("Maßstab 1 : 650", [(650, True)]),
    ("échelle 1/100", [(100, True)]),
    ("échelle 1/100e soit 1cm pour 1m", [(100, True)]),
    ("scala 1:50", [(50, True)]),
    ("Masstab 1 : 3oo.", [(300, True)]),               # OCR reads zeros as o
    ("M 1:20 / 1:50", [(20, True), (50, False)]),
    ("Grundriss EG 1:100", [(100, False)]),             # a common scale inside a caption
    ("1 cm = 2 m", [(200, True)]),
    ("11:50", []),                                       # a time
    ("1/2", []),                                         # a fraction
    ("1.50", []),                                        # a length
    ("m 1.50", []),
    ("Zimmer 1:7", []),
    ("Tür 1/100", []),                                   # slash without keyword
])
def test_parse_scale_note(text, expected):
    assert scale.parse_scale_note(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("4.50", [(4.5, "m")]),
    ("4,50", [(4.5, "m")]),
    ("2.38⁵", [(2.385, "m")]),                           # Swiss half centimetre in superscript
    ("2.38 5", [(2.385, "m")]),                          # the same as OCR reads it
    ("12⁵", [(0.125, "cm")]),
    ("450", [(4.5, "cm"), (0.45, "mm")]),                # integers: the sheet decides between cm and mm
    ("1 425", [(14.25, "cm"), (1.425, "mm")]),
    ("1'425", [(14.25, "cm"), (1.425, "mm")]),
    ("3.665", [(3.665, "m"), (36.65, "cm")]),
    ("4.50 m", [(4.5, "m")]),
    ("450 cm", [(4.5, "cm")]),
    ("24.50 m²", []),                                    # area
    ("24.50 m2", []),
    ("+2.85", []),                                       # level
    ("±0.00", []),
    ("Büro", []),
    ("18.5/23.5", []),
])
def test_parse_dimension(text, expected):
    got = scale.parse_dimension(text)
    assert [u for _, u in got] == [u for _, u in expected]
    assert [v for v, _ in got] == pytest.approx([v for v, _ in expected])


def draw_chain(img, y, ticks, labels, h=24, tick="slash", vertical=False):
    """Dimension chain on img (white background): a line through tick positions, a number centred above each
    segment. Returns OCR-like text items (boxes padded as PP-OCR does)."""
    items = []
    ink = img if not vertical else np.ascontiguousarray(img.T)
    cv2.line(ink, (ticks[0] - 30, y), (ticks[-1] + 30, y), 0, 2)
    for x in ticks:
        if tick == "slash":
            cv2.line(ink, (x - 8, y + 8), (x + 8, y - 8), 0, 3)
        else:
            cv2.line(ink, (x, y - 12), (x, y + 12), 0, 2)
    for (a, b), s in zip(zip(ticks, ticks[1:]), labels):
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, h / 30, 2)
        x0, y1 = (a + b) // 2 - tw // 2, y - 8
        cv2.putText(ink, s, (x0, y1), cv2.FONT_HERSHEY_SIMPLEX, h / 30, 0, 2)
        p = 0.3 * th
        box = (x0 - p, y1 - th - p, x0 + tw + p, y1 + p)
        if vertical:
            box = (box[1], box[0], box[3], box[2])
        items.append({"text": s, "box": box, "conf": 0.99, "source": "ocr", "angle": 90 if vertical else 0, "height": th})
    if vertical:
        img[:] = ink.T
    return items


def test_measure_span_slash_ticks():
    img = np.full((400, 1400), 255, np.uint8)
    items = draw_chain(img, 300, [100, 400, 1100], ["3.00", "7.00"])
    ink = scale.binarise(img)
    for t, span in zip(items, (300, 700)):
        m = scale.measure_span(ink, t["box"])
        assert m is not None and m["tight"]
        assert m["span"] == pytest.approx(span, abs=2)


def test_measure_span_skips_crossing_wall():
    """A wall crossing the dimension line between the ticks is not taken for a tick: the text is centred between
    the true ticks only."""
    img = np.full((500, 1400), 255, np.uint8)
    (t,) = draw_chain(img, 300, [100, 1100], ["10.00"])
    cv2.line(img, (330, 150), (330, 450), 0, 3)             # a wall line crossing on both sides
    m = scale.measure_span(scale.binarise(img), t["box"])
    assert m is not None and m["span"] == pytest.approx(1000, abs=2)


def test_measure_span_ignores_text_on_one_side():
    img = np.full((400, 1000), 255, np.uint8)
    (t,) = draw_chain(img, 200, [100, 700], ["6.00"])
    cv2.putText(img, "80", (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)    # a number below the line
    m = scale.measure_span(scale.binarise(img), t["box"])
    assert m is not None and m["span"] == pytest.approx(600, abs=2)


def test_dimension_cue_robust_with_vertical_text_and_outlier():
    img = np.full((1300, 1600), 255, np.uint8)
    items = draw_chain(img, 1200, [100, 350, 850, 1450], ["250", "500", "600"])                       # cm, 100 px/m
    items += draw_chain(img, 1500, [100, 500, 1100], ["4.00", "6.00"], vertical=True, tick="bar")    # metres, vertical
    bad = draw_chain(img, 200, [300, 700], ["9.00"])                                                  # misdrawn: outlier
    res = scale.dimension_cue(items + bad, scale.binarise(img))
    best = res["candidates"][0]
    assert res["pairs"] == 6
    assert best["px_per_m"] == pytest.approx(100, rel=0.01)
    assert best["support"] == 5 and best["unit"] == "cm"
    assert not any(e["inlier"] for e in res["examples"] if e["text"] == "9.00")


def test_dimension_cue_small_integers_count_little_and_single_digits_not_at_all():
    img = np.full((400, 1400), 255, np.uint8)
    items = draw_chain(img, 300, [100, 400, 1000], ["30", "60"])                 # two-digit cm values only
    items += draw_chain(img, 150, [100, 300], ["7"])                             # a single digit: a label
    res = scale.dimension_cue(items, scale.binarise(img))
    assert res["numbers"] == 2
    best = res["candidates"][0]
    assert best["px_per_m"] == pytest.approx(1000, rel=0.02) and best["support"] == pytest.approx(0.5)
    assert not best["precise"]
    why = []
    assert scale.measure_span(scale.binarise(img), (1200, 20, 1260, 44), why=why) is None and why


def test_cluster_rejects_outliers_and_keeps_unit_alternative():
    hyps = [[100.0, 1000.0], [101.0, 1010.0], [99.5, 995.0], [140.0, 1400.0], [100.5]]
    cl = scale.cluster(hyps, 0.03)
    assert cl[0]["support"] == 4 and cl[0]["value"] == pytest.approx(100.25, rel=0.01)
    assert 3 not in cl[0]["members"]
    assert cl[1]["value"] == pytest.approx(1002.5, rel=0.01)       # the same numbers read in mm


def test_infer_dpi_is_ambiguous_for_paper_sizes():
    dpis = {(d, s) for d, s, _ in scale.infer_dpi((3508, 2480), meta_dpi=72)}
    assert {(300.0, "paper"), (150.0, "paper"), (72.0, "metadata")} <= dpis
    assert scale.infer_dpi((1234, 987)) == []


def cand(cue, v, w, tol, **kw):
    return scale.candidate(cue, v, w, tol, "test", **kw)


def test_consensus_agreement_and_snapping():
    p100 = 300 / INCH / 100                               # 1:100 at 300 dpi = 118.11 px/m
    cands = [cand("scale_note", p100, 0.8, DEFAULT.scale_tol_note, scale=100, dpi=300.0, dpi_source="scan"),
             cand("dimension_strings", p100 * 1.012, 1.0, DEFAULT.scale_tol_dims),
             cand("door_widths", p100 * 0.9, 0.4, DEFAULT.scale_tol_doors)]
    c = scale.consensus(cands, [(300.0, "scan", "scan header")], [100])
    assert c["px_per_m"] == pytest.approx(p100, rel=1e-5) and c["snapped"] and c["scale"] == "1:100"
    assert c["confidence"] == "high"
    assert {a["cue"] for a in c["agreeing"]} == {"scale_note", "dimension_strings", "door_widths"}
    assert not c["disagreeing"] and not c["flags"]


def test_consensus_reports_disagreement():
    """A plan printed reduced: the note says 1:100, the drawing measures 1:250 (an A1 sheet printed on A3). The
    dimension strings and the doors win; the note is reported as disagreeing with a high-severity flag."""
    note = 300 / INCH / 100                                # 1:100 at 300 dpi
    true = note / 2.5
    cands = [cand("scale_note", note, 0.8, DEFAULT.scale_tol_note, scale=100, dpi=300.0, dpi_source="render"),
             cand("dimension_strings", true, 1.0, DEFAULT.scale_tol_dims),
             cand("door_widths", true * 1.1, 0.4, DEFAULT.scale_tol_doors)]
    c = scale.consensus(cands, [(300.0, "render", "render")], [100])
    assert c["px_per_m"] == pytest.approx(true, rel=0.01) and not c["snapped"]
    assert [d["cue"] for d in c["disagreeing"]] == ["scale_note"]
    assert c["confidence"] == "low"
    assert any(f["severity"] == "high" and "scale_note" in f["message"] and "printed reduced" in f["message"]
               for f in c["flags"])


def test_consensus_single_weak_cue_and_none():
    c = scale.consensus([cand("door_widths", 55.0, 0.4, DEFAULT.scale_tol_doors)])
    assert c["px_per_m"] == pytest.approx(55.0) and c["confidence"] == "low"
    assert any("one cue" in f["message"] for f in c["flags"])
    c = scale.consensus([])
    assert c["px_per_m"] is None and c["confidence"] == "none" and c["flags"][0]["severity"] == "high"


def test_consensus_door_cue_resolves_unit_ambiguity():
    cands = [cand("dimension_strings", 100.0, 0.6, DEFAULT.scale_tol_dims, unit="cm"),
             cand("dimension_strings", 1000.0, 0.6 * 0.6, DEFAULT.scale_tol_dims, unit="mm"),
             cand("door_widths", 950.0, 0.4, DEFAULT.scale_tol_doors)]
    c = scale.consensus(cands)
    assert c["px_per_m"] == pytest.approx(1000.0, rel=0.01)
    assert {a["cue"] for a in c["agreeing"]} == {"dimension_strings", "door_widths"}


def test_note_cue_with_paper_alternatives():
    texts = [{"text": "échelle 1/100", "box": (0, 0, 10, 10)}]
    res = scale.note_cue(texts, scale.infer_dpi((3508, 2480)))
    by_dpi = {c["dpi"]: c for c in res["candidates"]}
    assert by_dpi[300.0]["px_per_m"] == pytest.approx(300 / INCH / 100, rel=1e-4)
    assert by_dpi[300.0]["weight"] > by_dpi[150.0]["weight"]       # 300 dpi is the more common scan resolution


def test_door_crossing_ignores_spurious_rows_at_extreme_scales():
    """A door-search table (CubiCasa plan at 100 px/m): the closest-to-target row is a spurious one at 290.8 px/m
    (5 blobs measuring 0.96 m); the crossing of 0.9 m between 81.3 and 111.8 px/m is the answer."""
    rows = [(42.9, 124, 0.73), (59.1, 33, 1.03), (81.3, 21, 1.0), (111.8, 17, 0.74), (153.7, 11, 0.56),
            (211.4, 10, 0.48), (290.8, 5, 0.96)]
    rows = [{"px_per_m": p, "doors": n, "door_m": d} for p, n, d in rows]
    c = scale.door_crossing(rows, 0.9)
    assert 81.3 < c["px_per_m"] < 111.8 and c["px_per_m"] == pytest.approx(90.8, abs=0.5)
    assert scale.door_crossing([{"px_per_m": 50.0, "doors": 4, "door_m": 0.5}], 0.9) is None
    # far below the training scale the segmenter finds dozens of door-like blobs: the most confident crossing wins
    rows = [(22.7, 98, 0.62, 0.888), (31.2, 54, 0.96, 0.899), (42.9, 30, 0.8, 0.938), (59.1, 11, 1.52, 0.945),
            (81.3, 9, 1.12, 0.955), (111.8, 6, 0.9, 0.952), (153.7, 6, 0.62, 0.947)]
    rows = [{"px_per_m": p, "doors": n, "door_m": d, "confidence": c} for p, n, d, c in rows]
    assert scale.door_crossing(rows, 0.9)["px_per_m"] == pytest.approx(111.8, abs=0.1)


def test_stamp_area_cue_and_known_scale_check():
    res = scale.stamp_area_cue({"length_factor": 1.1, "rooms": 6, "agrees": False}, 100.0)
    (c,) = res["candidates"]
    assert c["px_per_m"] == pytest.approx(100 / 1.1)       # polygons 21 % too small: the true scale is coarser
    assert scale.stamp_area_cue(None, 100.0)["candidates"] == []
    con = scale.consensus([cand("dimension_strings", 118.0, 1.0, DEFAULT.scale_tol_dims),
                           cand("scale_note", 118.1, 0.8, DEFAULT.scale_tol_note, scale=100, dpi=300.0, dpi_source="scan")])
    assert scale.compare_known(con, 118.11, "title block")
    assert not scale.compare_known(con, 59.0, "wrong title block")
    assert any("scale used" in f["message"] for f in con["flags"])
    empty = scale.consensus([])
    assert scale.compare_known(empty, 100.0, "dataset") is None
    assert [f["severity"] for f in empty["flags"]] == ["low"]      # nothing to check against: not an error


def test_prepass_with_native_text_only():
    """The pre-pass on a synthetic sheet with a native text layer (no OCR, no model)."""
    img = np.full((700, 1600), 255, np.uint8)
    texts = draw_chain(img, 600, [100, 600, 1000, 1500], ["5.00", "4.00", "5.00"])
    for t in texts:
        t["source"] = "pdf"
    texts.append({"text": "M 1:100", "box": (100, 50, 300, 80), "conf": 1.0, "source": "pdf", "angle": 0, "height": 30})
    dpi = 100 * INCH * 100                                # 1:100 at 254 dpi = 100 px/m
    r = scale.prepass(img, dpi=dpi, dpi_source="render", native_text=texts)
    c = r["consensus"]
    assert c["px_per_m"] == pytest.approx(100, rel=1e-3) and c["confidence"] == "high"
    assert {a["cue"] for a in c["agreeing"]} == {"scale_note", "dimension_strings"}
    rec = scale.record(r)
    assert set(rec) >= {"cues", "consensus"} and "texts" not in rec
