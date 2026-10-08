"""fpx.pipeline.run_document on a synthetic scanned sheet with three drawings: the sheet is OCR'd once and every
drawing gets its own items in its working frame; the OCR cache; the overrides hook (confirmed region, scale, storey,
selection); the no-cue route (sheet-size prior) with the second pass after the stamp-area cue; and the scale cues
the review added (reduced-print confidence, feet and inches, scale bars with unit words, room sizes)."""
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image

from fpx import DEFAULT, layout, pipeline, scale
from fpx.model import CLASSES, WALL

DPI = 400
TRUE = DPI / 0.0254 / 100                                # 1:100 at 400 dpi: 157.48 px/m
H_TXT = 20                                               # every word is a solid rectangle 20 px high; its aspect ratio names it
WORDS = {"Grundriss 1. OG 1:100": 9.0, "Schnitt A-A": 6.2, "Buero": 3.0, "Bad": 2.2, "Korridor": 5.4}


def area_word(w_px, h_px):
    """A stamp area for a clear room of w x h sheet pixels at the true scale, with a ratio of its own."""
    a = (w_px / TRUE) * (h_px / TRUE)
    text = f"{a:.2f} m2"
    WORDS.setdefault(text, 3.8 + 0.4 * len([k for k in WORDS if k.endswith("m2")]))
    return text


def word(img, text, x, y):
    w = int(round(WORDS[text] * H_TXT))
    img[y:y + H_TXT, x:x + w] = 0
    return (x, y, x + w, y + H_TXT)


def ring(img, x0, y0, x1, y1, t=12):
    img[y0:y1, x0:x1] = 0
    img[y0 + t:y1 - t, x0 + t:x1 - t] = 255


def sheet_tiff(path):
    """A 2400 x 1600 px 1-bit scan at 400 dpi: drawing A (two rooms with name and area stamps, no caption), drawing B
    (one room, caption "Grundriss 1. OG 1:100"), drawing C (a box captioned "Schnitt A-A"). Returns the stamps."""
    img = np.full((1600, 2400), 255, np.uint8)
    stamps = {}
    # A: 800 x 600 outer, an interior wall at x = 680, rooms 468 x 576 and 296 x 576 clear
    ring(img, 200, 200, 1000, 800)
    img[212:788, 680:692] = 0
    a1, a2 = area_word(468, 576), area_word(296, 576)
    word(img, "Buero", 330, 440)
    word(img, a1, 330, 466)
    word(img, "Bad", 760, 440)
    word(img, a2, 760, 466)
    stamps["A"] = {"Buero": float(a1.split()[0]), "Bad": float(a2.split()[0])}
    # B: 800 x 600, one room 776 x 576 clear, caption 60 px below
    ring(img, 1400, 200, 2200, 800)
    b1 = area_word(776, 576)
    word(img, "Korridor", 1700, 440)
    word(img, b1, 1700, 466)
    word(img, "Grundriss 1. OG 1:100", 1500, 860)
    stamps["B"] = {"Korridor": float(b1.split()[0])}
    # C: a section-like box with its caption
    ring(img, 300, 1000, 900, 1300, t=6)
    word(img, "Schnitt A-A", 350, 1360)
    Image.fromarray(img).convert("1").save(path, dpi=(DPI, DPI), compression="group4")
    return stamps


class FakeResult:
    def __init__(self, boxes, txts, scores):
        self.boxes, self.txts, self.scores = (boxes or None), txts, scores


class FakeEngine:
    """An OCR engine with RapidOCR's one-call interface: reads the solid word rectangles by their aspect ratio (so
    it is independent of the resolution), counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, bgr, text_score=None):
        self.calls += 1
        dark = (bgr[..., 0] < 128).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        boxes, txts, scores = [], [], []
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if h < 8:
                continue
            for text, ratio in WORDS.items():
                if abs(w / h / ratio - 1) < 0.06:
                    boxes.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float))
                    txts.append(text)
                    scores.append(0.95)
        return FakeResult(boxes, tuple(txts), tuple(scores))


def fake_segment(sheet, model, cfg):
    """Walls where the working image is dark, except inside the text boxes; one-hot probabilities."""
    g = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2GRAY)
    lab = np.where(g < 128, WALL, 0).astype(np.uint8)
    for t in sheet.text:
        x0, y0, x1, y1 = (int(round(v)) for v in t["box"])
        lab[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = 0
    sheet.label, sheet.seg_label = lab, lab.copy()
    sheet.prob = (np.arange(len(CLASSES))[:, None, None] == lab[None]).astype(np.float32)


@pytest.fixture(scope="module")
def tiff(tmp_path_factory):
    path = tmp_path_factory.mktemp("doc") / "scan.tif"
    stamps = sheet_tiff(path)
    return path, stamps


def by_x(entry):
    """The drawing records of a package sorted left to right, then top to bottom: A, B, C."""
    return sorted(entry["drawings"], key=lambda r: (r["drawing"]["bbox_px"][1] > 900, r["drawing"]["bbox_px"][0]))


def texts_of(sheet):
    """The OCR'd words of a drawing's text layer, its caption (role "caption", inside the mask margin) left out."""
    return sorted(t["text"] for t in sheet.text if t["source"] == "ocr" and t["role"] != "caption")


def test_sheet_read_once_and_items_land_in_each_drawing(tiff, tmp_path):
    path, stamps = tiff
    eng = FakeEngine()
    (entry,) = pipeline.run_document(path, engine=eng, stages=("triage", "text"), out_dir=tmp_path)
    A, B, C = by_x(entry)
    assert C["skipped"] and C["drawing"]["kind"] == "section" and C["sheet"] is None
    assert A["skipped"] is None and B["skipped"] is None
    # B has its caption note at the trusted scan dpi; A has no cue and runs at the sheet-size prior, flagged
    assert B["scale"]["px_per_m"] == pytest.approx(TRUE, rel=1e-3) and B["scale"]["note"]["source"] == "caption"
    assert "sheet_size_prior" in A["scale"]["cues"] and A["scale"]["confidence"] == "low"
    assert any(f["severity"] == "high" and "no scale cue" in f["message"] for f in A["scale"]["flags"])
    assert A["scale"]["px_per_m"] == pytest.approx(scale.note_px_per_m(200, DPI), rel=0.05)
    # each drawing's text layer holds its own words only, in its working frame
    a1, a2 = (f"{v:.2f} m2" for v in stamps["A"].values())
    assert texts_of(A["sheet"]) == sorted(["Buero", a1, "Bad", a2])
    assert texts_of(B["sheet"]) == sorted(["Korridor", f"{stamps['B']['Korridor']:.2f} m2"])
    assert [t["role"] for t in B["sheet"].text if t["text"].startswith("Grundriss")] == ["caption"]
    buero = next(t for t in A["sheet"].text if t["text"] == "Buero")
    cx = (buero["box"][0] + buero["box"][2]) / 2
    assert A["sheet"].img.shape[1] * 0.1 < cx < A["sheet"].img.shape[1] * 0.5     # in the left room of A
    assert all(t["role"] in ("room stamp", "number") for t in A["sheet"].text)
    # B's resolution is close to the OCR resolution: the sheet pre-pass is reused, no second OCR; A at the coarser
    # prior gets one pass inside its mask (and none at the second rotation: the fake reads nothing rotated)
    assert B["text_ocr"]["source"] == "sheet pre-pass"
    assert A["text_ocr"]["source"] == "drawing pass" and A["text_ocr"]["factor"] == pytest.approx(2.0)
    calls = eng.calls
    assert calls > 0
    # the cache: a second run on the same sheet costs no OCR call
    (entry2,) = pipeline.run_document(path, engine=eng, stages=("triage", "text"), out_dir=tmp_path)
    assert eng.calls == calls
    assert entry2["package"].provenance["sheet_ocr"]["cached"] is True
    A2 = by_x(entry2)[0]
    assert A2["text_ocr"]["cached"] is True and texts_of(A2["sheet"]) == texts_of(A["sheet"])
    doc = json.loads((tmp_path / "scan_sheet.json").read_text(encoding="utf-8"))
    assert [d["kind"] for d in doc["drawings"]].count("floor plan") == 2
    assert all(d["text_ocr"] is None or "source" in d["text_ocr"] for d in doc["drawings"])


def test_overrides_region_scale_storey_and_selection(tiff, tmp_path):
    path, stamps = tiff
    eng = FakeEngine()
    (entry,) = pipeline.run_document(path, engine=eng, stages=("triage", "text"), out_dir=tmp_path)
    A, B, C = by_x(entry)
    pid = entry["package"].id
    poly = [[1380, 180], [2220, 180], [2220, 820], [1380, 820]]
    overrides = {f"{pid}/{B['drawing']['id']}": {"polygon_px": poly, "px_per_m": 120.0, "storey": "2. OG",
                                                 "note": "measured by hand: 1200 px = 10 m"},
                 f"{pid}/{A['drawing']['id']}": {"extract": False, "note": "not needed"},
                 f"{pid}/{C['drawing']['id']}": {"extract": True}}
    (entry,) = pipeline.run_document(path, engine=eng, stages=("triage", "text"), out_dir=tmp_path, overrides=overrides)
    A, B, C = by_x(entry)
    assert A["skipped"] == "deselected by the user: not needed" and A["sheet"] is None
    assert C["skipped"] is None and C["sheet"] is not None             # selected although it is a section
    d, s, sh = B["drawing"], B["scale"], B["sheet"]
    assert d["source"] == "human" and d["polygon_px"] == poly and d["storey"] == "2. OG"
    assert d["mask"]["reason"].startswith("region confirmed by the user")
    assert s["confirmed"]["px_per_m"] == 120.0 and s["confidence"] == "confirmed" and s["px_per_m"] == 120.0
    assert s["confirmed"]["method"] == "confirmed by the user" and "measured by hand" in s["confirmed"]["note"]
    assert "scale_note" in s["confirmed"]["disagrees_with"]            # the caption's 1:100 at 400 dpi is 157.5 px/m
    assert any("scale_note gives" in f["message"] and "confirmed 120.0 px/m" in f["message"] for f in s["flags"])
    assert sh.scale["method"].startswith("confirmed by the user (measured by hand")
    assert sh.scale["confirmed"]["px_per_m"] == 120.0 and sh.meta["storey"] == "2. OG"
    assert sh.drawings[0]["scale"]["confirmed"] is True and sh.drawings[0]["scale"]["confirmation"]["note"].startswith("measured")
    m = d["mask"]["margin_px"]                                          # the crop is the polygon plus the layout margin
    assert sh.px_per_m == DEFAULT.px_per_m and sh.img.shape[1] == pytest.approx((2220 - 1380 + 2 * m) * 50 / 120 + 100, abs=3)
    assert texts_of(sh) == sorted(["Korridor", f"{stamps['B']['Korridor']:.2f} m2"])
    doc = json.loads((tmp_path / "scan_sheet.json").read_text(encoding="utf-8"))
    rb = next(x for x in doc["drawings"] if x["id"] == d["id"])
    assert rb["scale"]["confirmed"] is True and rb["override"]["px_per_m"] == 120.0
    with pytest.raises(ValueError):
        pipeline.with_region(B["drawing"], [[0, 0], [1, 1]], entry["layout"])


def test_no_cue_route_and_second_pass_after_stamp_areas(tiff, tmp_path, monkeypatch):
    """Drawing A has no scale cue: it runs at the sheet-size prior (1:200), its stamp areas move the scale by 2x, the
    drawing is run again at the new scale and the rooms then match their stamps. B (noted 1:100) runs once."""
    import fpx.segment
    monkeypatch.setattr(fpx.segment, "segment", fake_segment)
    # the door-width search finds nothing on the fake labels (no doors): the sheet-size prior is the last resort
    monkeypatch.setattr(scale, "door_search", lambda img, model, cfg: {"px_per_m": None, "method": "no doors", "rows": [], "candidates": []})
    path, stamps = tiff
    eng = FakeEngine()
    cfg = replace(DEFAULT, scale_cue_min_rooms=1)
    stages = [s for s in pipeline.STAGES if s != "export"]
    (entry,) = pipeline.run_document(path, model=object(), engine=eng, cfg=cfg, stages=stages, out_dir=tmp_path)
    A, B, C = by_x(entry)
    assert A["error"] is None and B["error"] is None, (A["error"], B["error"], A.get("traceback"), B.get("traceback"))
    assert B["second_pass"] is None and B["scale"]["px_per_m"] == pytest.approx(TRUE, rel=1e-3)
    sp = A["second_pass"]
    assert sp is not None and sp["from"] == pytest.approx(scale.note_px_per_m(200, DPI), rel=0.05)
    assert sp["to"] == pytest.approx(TRUE, rel=0.06) and "times_first_pass" in sp
    assert A["scale"]["px_per_m"] == sp["to"]
    assert {a["cue"] for a in A["scale"]["consensus"]["agreeing"]} >= {"stamp_areas"}
    sh = A["sheet"]
    assert sh.meta["second_pass"] == sp and sh.scale["native_px_per_m"] == pytest.approx(sp["to"], rel=1e-3)
    stamped = {r["area_stamp"]: r for r in sh.rooms if r.get("area_stamp")}
    assert set(stamped) == set(stamps["A"].values())                   # "Buero" is snapped to the vocabulary's "Büro"
    for name, stamp in stamps["A"].items():
        r = stamped[stamp]
        assert r["area_net"] == pytest.approx(stamp, rel=0.1) and any(x.startswith(name) for x in r["names_raw"])
    assert any("second pass" in f["message"] for f in sh.qa)
    assert any("no scale cue" in f["message"] and f["severity"] == "high" for f in sh.qa)
    (bk,) = [r for r in B["sheet"].rooms if r.get("area_stamp")]
    assert bk["area_net"] == pytest.approx(stamps["B"]["Korridor"], rel=0.1) and any(x.startswith("Korridor") for x in bk["names_raw"])


# ---------- scale cues of the review ----------

def cand(cue, v, w, tol, **kw):
    return scale.candidate(cue, v, w, tol, "test", **kw)


def test_reduced_print_is_high_confidence():
    """A 1:100 plan printed three A formats smaller (r = 2^-1.5) at a trusted 600 dpi: the dimension strings measure
    1:282.8; the note re-enters at that step, the two agree, confidence is high, the disagreement is one medium
    "reduced print" flag."""
    v = scale.note_px_per_m(100, 600, r=2 ** -1.5)
    dims = cand("dimension_strings", v * 1.004, 1.0, DEFAULT.scale_tol_dims, precise=True)
    s = scale.drawing_scale(None, (), 600, "render", caption_notes=[100], extra=[dims])
    assert s["confidence"] == "high" and s["px_per_m"] == pytest.approx(v, rel=1e-3)
    assert "scale_note (r=0.3536)" in s["cues"] and s["cues"]["scale_note (r=0.3536)"]["print_factor"] == pytest.approx(2 ** -1.5, rel=1e-3)
    assert {a["cue"] for a in s["consensus"]["agreeing"]} == {"dimension_strings", "scale_note (r=0.3536)"}
    assert not s["consensus"]["disagreeing"] and s["consensus"]["snapped"] and s["consensus"]["scale"] == "1:100"
    assert s["consensus"]["print_factor"] == pytest.approx(2 ** -1.5, rel=1e-3)
    assert s["print_factor"]["step"] == 3 and s["print_factor"]["fits"]
    reduced = [f for f in s["flags"] if "reduced or enlarged print" in f["message"]]
    assert len(reduced) == 1 and reduced[0]["severity"] == "medium"
    assert not any("scale cues disagree" in f["message"] for f in s["flags"])
    # a factor that is no sqrt(2) step stays low with the high flag
    odd = cand("dimension_strings", scale.note_px_per_m(100, 600, r=0.6), 1.0, DEFAULT.scale_tol_dims, precise=True)
    s = scale.drawing_scale(None, (), 600, "render", caption_notes=[100], extra=[odd])
    assert s["confidence"] == "low" and any(f["severity"] == "high" and "no sqrt(2) step" in f["message"] for f in s["flags"])


def test_add_drawing_cue_and_confirm():
    s = scale.drawing_scale(None, (), 300, "scan", caption_notes=[100])
    p100 = scale.note_px_per_m(100, 300)
    assert s["px_per_m"] == pytest.approx(p100, rel=1e-4)
    scale.add_drawing_cue(s, "stamp_areas", scale.stamp_area_cue({"length_factor": 1.0, "rooms": 6, "agrees": True}, p100))
    assert {a["cue"] for a in s["consensus"]["agreeing"]} == {"scale_note", "stamp_areas"} and s["confidence"] == "high"
    scale.confirm(s, p100 * 1.5, "measured")
    assert s["confidence"] == "confirmed" and s["confirmed"]["disagrees_with"] == ["scale_note", "stamp_areas"]
    assert s["consensus"]["confirmed_by"]["ratio_to_proposal"] == pytest.approx(1.5)
    scale.add_drawing_cue(s, "stamp_areas", scale.stamp_area_cue({"length_factor": 1.0, "rooms": 6, "agrees": True}, p100 * 1.5))
    assert s["px_per_m"] == pytest.approx(p100 * 1.5, rel=1e-4) and s["confirmed"]["agrees_with"] == ["stamp_areas"]


def test_sheet_size_prior():
    r = scale.sheet_size_prior((2000, 1500), dpi=300)
    assert r["scale"] == 200 and r["extent_mm"] == pytest.approx(169.3, abs=0.1)
    (c,) = r["candidates"]
    assert c["px_per_m"] == pytest.approx(scale.note_px_per_m(200, 300), rel=1e-3) and not scale.is_precise(c)
    assert scale.sheet_size_prior((6000, 4000), dpi=300)["scale"] == 50
    assert scale.sheet_size_prior((2000, 1500))["dpi_assumed"]


@pytest.mark.parametrize("text, expected", [
    ('3/16" = 1\'-0"', [(64, True)]),
    ("1/8 in. 1 Foot.", [(96, True)]),
    ("Scale /8in.lFoot.", [(96, True)]),                 # the 1 of 1/8 and of 1 Foot as OCR reads hand lettering
    ('1" = 20\'', [(240, True)]),
    ("1 in = 8 ft", [(96, True)]),
    ("Masstab : 150.", [(50, True)]),                    # the colon of 1:50 lost
    ("Massstab 1100", [(100, True)]),
])
def test_imperial_and_lossy_scale_notes(text, expected):
    assert scale.parse_scale_note(text) == expected


@pytest.mark.parametrize("text, metres", [
    ("20'", 6.096), ("6'-6\"", 1.9812), ("6'6\"", 1.9812), ("14'.0\"", 4.2672), ("30'-6 1/2\"", 9.3091), ("6 ft 6 in", 1.9812),
])
def test_feet_and_inches_dimensions(text, metres):
    assert scale.parse_dimension(text) == [(pytest.approx(metres, abs=1e-3), "ft")]
    assert scale.parse_dimension("1'425") == [(14.25, "cm"), (1.425, "mm")]     # the Swiss thousands apostrophe


def test_room_sizes_parse_and_cue():
    assert scale.parse_room_size("30'-6\" x 48'-6\"") == [((pytest.approx(9.2964), pytest.approx(14.7828)), "ft")]
    assert scale.parse_room_size('30.6" × 48!6') == [((pytest.approx(9.2964), pytest.approx(14.7828)), "ft")]
    assert scale.parse_room_size("4.50 x 3.20") == [((4.5, 3.2), "m")]
    assert scale.parse_room_size("Büro x 2") == []
    # three rooms at 100 px/m with their sizes in feet on the stamps; one stamp with a wrong size
    img = np.full((1400, 2400), 255, np.uint8)
    texts = []
    for k, (ft_w, ft_d) in enumerate(((20, 30), (12, 16), (24, 18))):
        w, d = int(round(ft_w * 0.3048 * 100)), int(round(ft_d * 0.3048 * 100))
        x0, y0 = 100 + k * 700, 100
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + d), 0, 6)
        texts.append({"text": f"{ft_w}'-0\" x {ft_d}'-0\"", "box": (x0 + w // 2 - 60, y0 + d // 2 - 12, x0 + w // 2 + 60, y0 + d // 2 + 12),
                      "conf": 0.9, "source": "ocr", "angle": 0, "height": 24.0})
    texts.append(dict(texts[0], text="10'-0\" x 15'-0\"", box=(170, 650, 290, 674)))      # a wrong size in room 1
    res = scale.room_size_cue(texts, scale.binarise(img))
    best = res["candidates"][0]
    assert res["pairs"] == 4 and best["pairs"] == 3 and best["px_per_m"] == pytest.approx(100, rel=0.02)
    assert best["precise"] and "sizes in feet" in best["evidence"]


def test_scale_bars_with_unit_words_and_without_zero():
    def bar(labels, y, step_px, x0=200, unit_word=None):
        img = np.full((300, 2000), 255, np.uint8)
        cv2.line(img, (x0, y), (x0 + int(step_px * max(v for v, _ in labels)), y), 0, 3)
        items = []
        for v, s in labels:
            x = x0 + int(step_px * v)
            items.append({"text": s, "box": (x - 15, y - 40, x + 15 + 10 * (len(s) - 1), y - 16), "conf": 0.95,
                          "source": "ocr", "angle": 0, "height": 24.0})
        return img, items
    img, items = bar([(0, "0"), (5, "5"), (10, "10"), (20, "20 m")], 150, 40)
    (c,) = scale.scale_bar_cue(items, scale.binarise(img))["candidates"]
    assert c["px_per_m"] == pytest.approx(40, rel=0.02)
    img, items = bar([(0, "0"), (20, "20 Meter.")], 150, 40)                   # two labels, one with the unit word
    (c,) = scale.scale_bar_cue(items, scale.binarise(img))["candidates"]
    assert c["px_per_m"] == pytest.approx(40, rel=0.02)
    img, items = bar([(1, "1"), (2, "2"), (3, "3"), (10, "10"), (20, "20 Meter")], 150, 40)     # the 0 unread
    (c,) = scale.scale_bar_cue(items, scale.binarise(img))["candidates"]
    assert c["px_per_m"] == pytest.approx(40, rel=0.02)
    img, items = bar([(0, "0"), (25, "25"), (50, "50 ft")], 150, 12)              # labels in feet: 12 px per foot
    (c,) = scale.scale_bar_cue(items, scale.binarise(img))["candidates"]
    assert c["px_per_m"] == pytest.approx(12 / 0.3048, rel=0.02) and "feet" in c["evidence"]
    (b,) = scale.find_scale_bars(items, scale.binarise(img))
    assert b["px_per_m"] == pytest.approx(12 / 0.3048, rel=0.02)
    img, items = bar([(0, "0"), (2, "2 3 4 5"), (10, "10")], 150, 40)             # several labels read as one item
    items[1]["box"] = (items[1]["box"][0], items[1]["box"][1], items[1]["box"][0] + 30 + 3 * 40, items[1]["box"][3])
    (c,) = scale.scale_bar_cue(items, scale.binarise(img))["candidates"]
    assert c["px_per_m"] == pytest.approx(40, rel=0.03)


# ---------- drawing kinds and storeys (T9) ----------

@pytest.mark.parametrize("text, kind", [
    ("Grundriss EG (Schnitt A-A)", "floor plan"), ("Schnitt A-A - Grundriss", "section"), ("Plan de situation", "site plan"),
    ("Plan masse", "site plan"), ("Deckenspiegel EG", "ceiling plan"), ("Dachaufsicht", "roof plan"),
    ("Grundriss Dachgeschoss", "floor plan"), ("Fundamentplan", "foundation plan"), ("Fluchtwegplan 1. OG", "escape route plan"),
    ("Life Safety Plan", "escape route plan"), ("Reflected ceiling plan level 2", "ceiling plan"), ("Niveau 2", "floor plan"),
    ("Musterhaus", None),
])
def test_drawing_kind_scoring(text, kind):
    assert layout.drawing_kind(text) == kind
    if kind in layout.NOT_FLOOR_PLANS:
        assert kind in layout.KINDS and kind != "floor plan"


@pytest.mark.parametrize("text, storeys", [
    ("Grundriss EG und 1. OG", ["EG", "1. OG"]), ("Grundrisse EG, 1. OG und DG", ["EG", "1. OG", "DG"]),
    ("Niveau 2", ["2. OG"]), ("Niveau -1", ["1. UG"]), ("Niveau 0", ["EG"]), ("Level 1", ["1. OG"]), ("Level 0", ["EG"]),
    ("Piano primo", ["1. OG"]), ("Piano 2", ["2. OG"]), ("Pianta piano terra", ["EG"]), ("Zwischengeschoss", ["ZG"]),
    ("Grundriss 2. UG", ["2. UG"]), ("Etage 1:100", []), ("II. Stock", ["2. OG"]), ("3rd Floor Plan", ["3. OG"]),
    ("GRUNDRISS 1.OBERGESCHOSS 1:100", ["1. OG"]),
])
def test_parse_storeys(text, storeys):
    assert layout.parse_storeys(text) == storeys
    assert layout.parse_storey(text) == (storeys[0] if storeys else None)


def test_layout_flags_several_storeys(tmp_path):
    from test_layout import sheet_pdf
    import pymupdf
    path = tmp_path / "two.pdf"
    sheet_pdf(path)
    doc = pymupdf.open(path)
    page = doc[0]
    MM = 72 / 25.4
    page.insert_text((190 * MM, 140 * MM), "Grundriss EG und 1. OG 1:50", fontsize=5 * MM)   # a second caption line under plan 2
    doc.save(tmp_path / "two2.pdf")
    from fpx import inputs
    pkg = inputs.load(tmp_path / "two2.pdf", dpi=200)[0]
    res = layout.analyse(pkg)
    multi = [d for d in res["drawings"] if len(d.get("storeys", [])) > 1]
    assert multi and multi[0]["storey"] == multi[0]["storeys"][0]
    assert any("several storeys" in f["message"] for f in res["flags"])
