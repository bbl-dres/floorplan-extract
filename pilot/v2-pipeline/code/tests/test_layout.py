"""Stage 1b, sheet layout and masking (fpx.layout), and the per-drawing scale of stage 1c (fpx.scale.drawing_scale), on
a synthetic A3 sheet: a frame, a ruled title block with text, a legend and two floor plans at 1:100 and 1:50 with
captions and dimension chains, drawn as a vector PDF (native text)."""
import numpy as np
import pymupdf
import pytest

from fpx import inputs, layout, scale

MM = 72 / 25.4                                           # PDF points per millimetre


def plan(page, x0, y0, w_m, d_m, n, rooms, caption):
    """A floor plan at 1:n with its top-left corner at (x0, y0) mm: outer and inner wall faces (0.3 m), one interior
    wall, room names, a dimension chain below with its number, and the caption below that. Returns its paper box."""
    s = 1000.0 / n                                       # paper mm per plan metre
    t = 0.3 * s
    W, D = w_m * s, d_m * s
    r = lambda a, b, c, d: pymupdf.Rect(a * MM, b * MM, c * MM, d * MM)
    page.draw_rect(r(x0, y0, x0 + W, y0 + D), width=0.8)
    page.draw_rect(r(x0 + t, y0 + t, x0 + W - t, y0 + D - t), width=0.5)
    xm = x0 + 0.55 * W
    page.draw_rect(r(xm, y0 + t, xm + 0.15 * s, y0 + D - t), width=0.5)
    for k, name in enumerate(rooms):
        page.insert_text(((x0 + (0.2 + 0.55 * k) * W) * MM, (y0 + D / 2) * MM), name, fontsize=2.5 * MM)
    yd = y0 + D + 8                                      # dimension chain 8 mm below the facade
    page.draw_line((x0 * MM, yd * MM), ((x0 + W) * MM, yd * MM), width=0.35)
    for x in (x0, x0 + W):
        page.draw_line(((x - 1.2) * MM, (yd + 1.2) * MM), ((x + 1.2) * MM, (yd - 1.2) * MM), width=0.5)
        page.draw_line((x * MM, (y0 + D + 1.5) * MM), (x * MM, (yd + 2) * MM), width=0.25)
    label = f"{w_m:.2f}"
    page.insert_text(((x0 + W / 2 - 4) * MM, (yd - 1) * MM), label, fontsize=2.5 * MM)
    page.insert_text((x0 * MM, (yd + 14) * MM), caption, fontsize=5 * MM)
    return (x0, y0, x0 + W, y0 + D)


def sheet_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page(width=420 * MM, height=297 * MM)
    r = lambda a, b, c, d: pymupdf.Rect(a * MM, b * MM, c * MM, d * MM)
    page.draw_rect(r(10, 10, 410, 287), width=1.0)                        # frame
    boxes = [plan(page, 30, 40, 12.0, 8.0, 100, ["Wohnen", "Kueche"], "Grundriss EG 1:100"),
             plan(page, 190, 30, 5.0, 4.0, 50, ["Zimmer", "Bad"], "Grundriss 1. OG 1:50")]
    # title block bottom right: 180 x 60 mm, five rows, one column divider
    x0, y0, x1, y1 = 230, 227, 410, 287
    page.draw_rect(r(x0, y0, x1, y1), width=0.7)
    for y in range(y0 + 12, y1, 12):
        page.draw_line((x0 * MM, y * MM), (x1 * MM, y * MM), width=0.35)
    page.draw_line((290 * MM, y0 * MM), (290 * MM, y1 * MM), width=0.35)
    for k, (a, b) in enumerate([("Projekt", "Musterhaus Bern"), ("Plan", "Grundrisse EG und 1. OG"), ("Plan-Nr.", "A-101"),
                                ("Massstab", "1:100 / 1:50"), ("Datum", "07.10.2026")]):
        page.insert_text(((x0 + 3) * MM, (y0 + 8 + 12 * k) * MM), a, fontsize=3 * MM)
        page.insert_text((293 * MM, (y0 + 8 + 12 * k) * MM), b, fontsize=3 * MM)
    # legend: three swatches with text, under a title
    page.insert_text((30 * MM, 200 * MM), "Legende", fontsize=3.5 * MM)
    for k, (name, grey) in enumerate([("Mauerwerk", 0.0), ("Beton", 0.45), ("Leichtbau", 0.75)]):
        y = 206 + 9 * k
        page.draw_rect(r(30, y, 42, y + 6), color=(0, 0, 0), fill=(grey, grey, grey), width=0.3)
        page.insert_text((46 * MM, (y + 5) * MM), name, fontsize=3 * MM)
    doc.save(path)
    return boxes


@pytest.fixture(scope="module")
def sheet(tmp_path_factory):
    path = tmp_path_factory.mktemp("layout") / "sheet.pdf"
    boxes = sheet_pdf(path)
    pkg = inputs.load(path, dpi=200)[0]
    return pkg, boxes, layout.analyse(pkg)


def test_two_drawings_with_captions_and_scale_notes(sheet):
    pkg, boxes, res = sheet
    ds = sorted(res["drawings"], key=lambda d: d["bbox_px"][0])
    assert len(ds) == 2
    pxmm = pkg.px_per_mm
    for d, b in zip(ds, boxes):
        x0, y0, x1, y1 = (v / pxmm for v in d["bbox_px"])
        assert x0 <= b[0] + 1 and y0 <= b[1] + 1 and x1 >= b[2] - 1 and y1 >= b[3] - 1   # the whole plan inside
        assert x1 - x0 < (b[2] - b[0]) + 30 and y1 - y0 < (b[3] - b[1]) + 30             # but not the sheet
        assert d["kind"] == "floor plan" and d["polygon_mm"] is not None
    assert ds[0]["title"] == "Grundriss EG 1:100" and ds[0]["storey"] == "EG" and ds[0]["scale_note"]["scale"] == 100
    assert ds[1]["title"] == "Grundriss 1. OG 1:50" and ds[1]["storey"] == "1. OG" and ds[1]["scale_note"]["scale"] == 50


def test_one_title_block_bottom_right_and_legend(sheet):
    pkg, _, res = sheet
    tbs = [r for r in res["regions"] if r["class"] == "title block"]
    assert len(tbs) == 1
    x0, y0, x1, y1 = (v / pkg.px_per_mm for v in tbs[0]["bbox_px"])
    assert x0 == pytest.approx(230, abs=4) and y0 == pytest.approx(227, abs=4) and x1 > 400 and y1 > 280
    assert res["sheet"]["sheet_scale_notes"] == [50, 100]
    legends = [r for r in res["regions"] if r["class"] == "legend"]
    assert len(legends) == 1
    lx0, ly0, lx1, ly1 = (v / pkg.px_per_mm for v in legends[0]["bbox_px"])
    assert lx0 < 31 and ly1 > 230 and lx1 < 120
    assert any(r["class"] == "frame" for r in res["regions"])
    # no region of another class covers a drawing
    for d in res["drawings"]:
        assert d["mask"]["polygon_px"]


def test_masks_keep_dimension_chains_and_exclude_other_regions(sheet):
    pkg, boxes, res = sheet
    d = min(res["drawings"], key=lambda d: d["bbox_px"][0])
    m = layout.drawing_mask(d, pkg.img.shape)
    pxmm = pkg.px_per_mm
    b = boxes[0]
    assert m[int((b[3] + 8) * pxmm), int((b[0] + b[2]) / 2 * pxmm)]       # the dimension chain 8 mm below
    assert not m[int(250 * pxmm), int(300 * pxmm)]                         # the title block
    assert not m[int(210 * pxmm), int(36 * pxmm)]                          # the legend


def test_drawing_scale_per_drawing(sheet):
    """Stage 1c on each drawing's crop: its own caption note (not the title block's two notes), confirmed by its
    dimension string; the factor model gives exactly 1:100 and 1:50 at the render resolution."""
    pkg, _, res = sheet
    for d, n in zip(sorted(res["drawings"], key=lambda d: d["bbox_px"][0]), (100, 50)):
        x0, y0, x1, y1 = (int(v) for v in d["bbox_px"])
        crop = pkg.img[y0:y1, x0:x1]
        texts = [{**t, "box": (t["box"][0] - x0, t["box"][1] - y0, t["box"][2] - x0, t["box"][3] - y0)}
                 for t in pkg.text if x0 <= t["box"][0] and t["box"][2] <= x1 and y0 <= t["box"][1] and t["box"][3] <= y1]
        s = scale.drawing_scale(crop, texts, pkg.dpi, pkg.dpi_source, caption_notes=[d["scale_note"]["scale"]],
                                sheet_notes=res["sheet"]["sheet_scale_notes"])
        assert s["px_per_m"] == pytest.approx(scale.note_px_per_m(n, pkg.dpi), rel=0.005)
        assert s["note"]["source"] == "caption" and s["note"]["scale"] == n
        assert s["confidence"] in ("high", "medium")


def test_parse_caption_words():
    assert layout.parse_storey("Grundriss 1. Obergeschoss") == "1. OG"
    assert layout.parse_storey("Plan du 2e étage") == "2. OG"
    assert layout.parse_storey("Plan du Rez de Chaussée") == "EG"
    assert layout.parse_storey("ERSTER STOCK") == "1. OG"
    assert layout.parse_storey("I. Etage") == "1. OG"
    assert layout.parse_storey("Untergeschoss") == "UG"
    assert layout.parse_storey("Pianta piano terra") == "EG"
    assert layout.drawing_kind("Schnitt A-A 1:50") == "section"
    assert layout.drawing_kind("Ansicht Süd") == "elevation"
    assert layout.drawing_kind("Situation 1:500") == "site plan"
    assert layout.drawing_kind("Plan de situation") == "site plan"
    assert layout.drawing_kind("Grundriss EG") == "floor plan"
    assert layout.drawing_kind("Musterhaus") is None


def test_print_factor_model():
    """m = N 25.4 / (1000 d r): a 1:100 plan printed from A1 on A4 (r = 1/(2 sqrt 2)) measures 1:282.8."""
    assert scale.metres_per_px(100, 254) == pytest.approx(0.01)
    v = scale.note_px_per_m(100, 300, r=2 ** -1.5)
    h, flag = scale.print_factor_check(v, 100, 300, "dimension_strings")
    assert h["step"] == 3 and h["fits"] and flag["severity"] == "medium" and "reduced" in flag["message"]
    h, flag = scale.print_factor_check(scale.note_px_per_m(100, 300), 100, 300, "dimension_strings")
    assert flag is None and h["hypothesis"].startswith("original")
    h, flag = scale.print_factor_check(scale.note_px_per_m(100, 300, r=0.6), 100, 300, "dimension_strings")
    assert not h["fits"] and flag["severity"] == "high"


def test_viewports_become_drawings_with_exact_scale(tmp_path):
    from test_inputs import dxf_with_viewports
    path = tmp_path / "vp.dxf"
    dxf_with_viewports(path)
    pkg = inputs.load(path, dpi=100)[0]
    res = layout.analyse(pkg)
    vds = sorted([d for d in res["drawings"] if d["source"] == "dwg-viewport"], key=lambda d: d["bbox_px"][0])
    assert [d["scale_note"]["scale"] for d in vds] == [pytest.approx(100), pytest.approx(50)]
    assert vds[0]["title"] and vds[0]["title"].startswith("Grundriss EG")
    s = scale.drawing_scale(None, (), pkg.dpi, pkg.dpi_source, viewport_scale=vds[1]["scale_note"]["scale"])
    assert s["px_per_m"] == pytest.approx(pkg.px_per_mm * 1000 / 50, rel=1e-4) and s["confidence"] in ("medium", "high")


def test_run_document_layout_and_scale_per_drawing(tmp_path):
    """pipeline.run_document up to the text layer (no segmenter): one Sheet per floor plan at the working resolution,
    its scale from its own caption and dimension string, the sheet JSON with regions and drawings."""
    import json
    from fpx import DEFAULT, pipeline
    path = tmp_path / "sheet.pdf"
    sheet_pdf(path)
    (entry,) = pipeline.run_document(path, stages=("triage", "text"), out_dir=tmp_path, load_opts={"dpi": 200})
    recs = sorted(entry["drawings"], key=lambda r: r["drawing"]["bbox_px"][0])
    assert [r["skipped"] for r in recs] == [None, None]
    for r, n in zip(recs, (100, 50)):
        assert r["scale"]["px_per_m"] == pytest.approx(scale.note_px_per_m(n, 200), rel=0.005)
        sh = r["sheet"]
        assert sh.px_per_m == DEFAULT.px_per_m and sh.drawings[0]["storey"] in ("EG", "1. OG")
        assert any(t["role"] == "dimension" or t["text"].startswith(("12.00", "5.00")) for t in sh.text)
        assert not any("Massstab" in t["text"] for t in sh.text)                  # the title block is masked out
    doc = json.loads((tmp_path / "sheet-p1_sheet.json").read_text(encoding="utf-8"))
    assert {r["class"] for r in doc["regions"]} >= {"frame", "title block", "legend", "drawing", "caption"}
    assert [d["scale"]["note"]["scale"] for d in sorted(doc["drawings"], key=lambda d: d["bbox_px"][0])] == [100, 50]
