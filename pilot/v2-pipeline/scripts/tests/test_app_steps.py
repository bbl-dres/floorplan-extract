"""The app's three steps on the synthetic scanned sheet of test_document.py: the layout without text and without a
model (run_document(stages=(), scale=False)), then the layout with text anchored to it (ids follow the overlap, a
drawing the text layout no longer finds is revived with its area) with the sheet OCR bounded to boxes and cached per
box, as the app's scale step and extraction call it."""
import pytest

from fpx import pipeline
from test_document import FakeEngine, sheet_tiff


@pytest.fixture(scope="module")
def tiff(tmp_path_factory):
    path = tmp_path_factory.mktemp("steps") / "scan.tif"
    sheet_tiff(path)
    return path


def test_layout_only_then_anchored_layout_with_bounded_ocr(tiff, tmp_path):
    cache = tmp_path / "ocr_cache"
    # step 1: the layout alone, no OCR, no scale search
    (e1,) = pipeline.run_document(tiff, engine=None, stages=(), scale=False, cache_dir=cache)
    assert len(e1["drawings"]) == 3
    assert all(r["scale"] is None and r["error"] is None for r in e1["drawings"])
    assert e1["package"].provenance.get("sheet_ocr") is None
    recs = {r["drawing"]["id"]: {k: v for k, v in r["drawing"].items() if not k.startswith("_")} for r in e1["drawings"]}
    # anchors: two of the three step-1 records under other ids (the third is "new" to the anchored layout), plus one
    # over a free part of the sheet that no drawing overlaps
    unanchored = max(recs)                                 # the last drawing (C) is left out of the anchors
    anchors = {f"d{int(i[1:]) + 10}": rec for i, rec in recs.items() if i != unanchored}
    ghost = dict(next(iter(recs.values())))
    ghost.update(polygon_px=[[1400, 1000], [2200, 1000], [2200, 1500], [1400, 1500]], bbox_px=[1400, 1000, 2200, 1500])
    anchors["d99"] = ghost
    pid = e1["package"].id
    boxes = [tuple(r["bbox_px"]) for r in anchors.values()]
    # step 2: the layout with text, the OCR bounded to the boxes of the anchored drawings
    eng = FakeEngine()
    (e2,) = pipeline.run_document(tiff, engine=eng, stages=(), cache_dir=cache, anchors={pid: anchors}, ocr_boxes={pid: boxes})
    by_id = {r["drawing"]["id"]: r for r in e2["drawings"]}
    new = [r for r in e2["drawings"] if r["drawing"]["anchored"] == "new"]
    assert len(new) == 1 and new[0]["drawing"]["id"] not in anchors and "not among the confirmed" in new[0]["skipped"]
    assert set(by_id) == set(anchors) | {new[0]["drawing"]["id"]}
    for aid, rec in anchors.items():
        if aid != "d99":
            assert pipeline._bbox_iou(by_id[aid]["drawing"]["bbox_px"], rec["bbox_px"]) > 0.5
            assert by_id[aid]["drawing"]["anchored"] == "matched"
            assert by_id[aid]["scale"] is not None and by_id[aid]["error"] is None
    g = by_id["d99"]
    assert g["drawing"]["anchored"] == "revived" and g["drawing"]["bbox_px"] == [1400, 1000, 2200, 1500]
    assert g["drawing"]["source"] == "human" and "earlier layout" in g["drawing"]["mask"]["reason"]
    assert g["error"] is None and g["scale"] is not None                      # the scale step ran on it as well
    assert any("d99" in f["message"] for f in e2["layout"]["flags"])
    info = e2["package"].provenance["sheet_ocr"]
    assert info["source"] == "sheet pre-pass (bounded)" and info["boxes"] == len(boxes) and info["cached"] is False
    assert info["items"] > 0 and eng.calls > 0
    # every item lies in one of the boxes, and none is read twice
    items = e2["texts"]
    assert all(any(x0 <= (t["box"][0] + t["box"][2]) / 2 < x1 and y0 <= (t["box"][1] + t["box"][3]) / 2 < y1
                   for x0, y0, x1, y1 in boxes) for t in items if t.get("source") != "pdf")
    assert len({(t["text"], tuple(round(v) for v in t["box"])) for t in items}) == len(items)
    # step 3 (here: the same call again) finds every box in the cache; the ids stay
    calls = eng.calls
    (e3,) = pipeline.run_document(tiff, engine=eng, stages=(), cache_dir=cache, anchors={pid: anchors}, ocr_boxes={pid: boxes})
    assert eng.calls == calls and e3["package"].provenance["sheet_ocr"]["cached"] is True
    assert [r["drawing"]["id"] for r in e3["drawings"]] == [r["drawing"]["id"] for r in e2["drawings"]]


def test_after_hook_sees_each_stage_with_the_sheet(tiff):
    calls = []
    (entry,) = pipeline.run_document(tiff, engine=FakeEngine(), stages=("triage", "text"),
                                     after=lambda key, stage, sheet: calls.append((key, stage, sheet.id)))
    keys = [f"{entry['package'].id}/{r['drawing']['id']}" for r in entry["drawings"] if r["skipped"] is None]
    assert len(keys) == 2                                                       # the two floor plans; the section is skipped
    for key in keys:
        assert [c[1] for c in calls if c[0] == key] == ["triage", "text"]
    assert all(c[2] for c in calls)


def test_anchor_drawings_relabels_and_revives():
    lay = {"drawings": [{"id": "d1", "bbox_px": [0, 0, 100, 100]}, {"id": "d2", "bbox_px": [200, 0, 300, 100]}],
           "flags": [], "sheet": {"raster_px": [400, 400], "dpi": None}}
    anchors = {"d1": {"bbox_px": [205, 0, 300, 100], "polygon_px": [[205, 0], [300, 0], [300, 100], [205, 100]], "mask": {"margin_px": 0}},
               "d3": {"bbox_px": [0, 200, 100, 300], "polygon_px": [[0, 200], [100, 200], [100, 300], [0, 300]], "mask": {"margin_px": 5}}}
    pipeline._anchor_drawings(lay, anchors)
    ids = [d["id"] for d in lay["drawings"]]
    assert ids == ["d2", "d1", "d3"]                                            # d1 moved to the overlapping drawing, the first got a fresh id
    assert [d["anchored"] for d in lay["drawings"]] == ["new", "matched", "revived"]
    revived = lay["drawings"][2]
    assert revived["anchored"] == "revived" and revived["bbox_px"] == [0, 200, 100, 300]
    assert revived["mask"]["margin_px"] == 5 and len(lay["flags"]) == 1
