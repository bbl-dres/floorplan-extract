"""Stage 0a, normalisation (fpx.inputs), on generated files: a multi-page 1-bit TIFF, PNGs with and without a dpi, a PDF
with vector lines, real text, invisible text, outlined text, an annotation and a measurement viewport, a DXF with
model-space geometry and two paper-space viewports at 1:100 and 1:50, and a DWG without a converter."""
import cv2
import ezdxf
import numpy as np
import pymupdf
import pytest
from PIL import Image

from fpx import inputs

MM = 72 / 25.4


def test_multipage_tiff_1bit(tmp_path):
    a = np.full((400, 600), 255, np.uint8)
    cv2.rectangle(a, (50, 50), (550, 350), 0, 3)
    pages = [Image.fromarray(a).convert("1"), Image.fromarray(255 - a).convert("1")]
    path = tmp_path / "scan.tif"
    pages[0].save(path, save_all=True, append_images=pages[1:], dpi=(400, 400), compression="group4")
    pkgs = inputs.load(path)
    assert [p.id for p in pkgs] == ["scan-p1", "scan-p2"]
    p = pkgs[0]
    assert p.input_class == "raster scan (1-bit)" and p.img.shape == (400, 600, 3) and p.img.dtype == np.uint8
    assert set(np.unique(p.img)) == {0, 255}                                    # 1-bit expanded
    assert p.dpi == 400 and p.dpi_source == "scan" and p.dpi_trusted
    assert p.paper_mm == pytest.approx((600 / 400 * 25.4, 400 / 400 * 25.4))
    assert p.to_paper_mm([[400, 0]])[0, 0] == pytest.approx(25.4)
    assert p.img[52, 300].max() == 0 and pkgs[1].img[52, 300].min() == 255


def test_png_dpi_trust(tmp_path):
    a = np.full((300, 400, 3), 255, np.uint8)
    Image.fromarray(a).save(tmp_path / "screen.png", dpi=(96, 96))
    p = inputs.load(tmp_path / "screen.png")[0]
    assert p.dpi is None and p.dpi_source == "default"                          # never trust a screen default
    assert any(s == "metadata" and round(d) == 96 for d, s, _ in p.dpi_alternatives)
    Image.fromarray(np.full((2480, 3508, 3), 255, np.uint8)).save(tmp_path / "a4.png")
    p = inputs.load(tmp_path / "a4.png")[0]                                     # A4 landscape at 300 dpi or A2 at 150
    assert p.dpi is None and p.dpi_source == "unknown" and {300.0, 150.0} <= {d for d, _, _ in p.dpi_alternatives}
    Image.fromarray(a).save(tmp_path / "meta.png", dpi=(200, 200))
    p = inputs.load(tmp_path / "meta.png")[0]
    assert p.dpi == pytest.approx(200, abs=0.1) and p.dpi_source == "metadata" and not p.dpi_trusted


def outlined_text(page, text, x, y, height_mm):
    """Text as geometry: glyph contours of a raster rendering drawn as filled closed paths (no text span)."""
    img = np.full((80, 40 * len(text)), 255, np.uint8)
    cv2.putText(img, text, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
    ys, xs = np.nonzero(img < 128)
    f = height_mm * MM / (ys.max() - ys.min())
    cnts, _ = cv2.findContours((img < 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if len(c) >= 3:
            page.draw_polyline([(x + px * f, y + py * f) for px, py in c[:, 0, :]], color=None, fill=(0, 0, 0), closePath=True)


def make_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page(width=420 * MM, height=297 * MM)                       # A3 landscape
    page.draw_rect(pymupdf.Rect(10 * MM, 10 * MM, 410 * MM, 287 * MM), width=1)
    for k in range(12):                                                          # vector lines (walls)
        page.draw_line((40 * MM, (40 + 10 * k) * MM), (200 * MM, (40 + 10 * k) * MM), width=0.7)
    page.insert_text((220 * MM, 50 * MM), "Grundriss 1. OG 1:100", fontsize=5 * MM)      # real text, 5 mm
    page.insert_text((220 * MM, 70 * MM), "4.50", fontsize=2 * MM)                       # real text, 2 mm
    page.insert_text((220 * MM, 90 * MM), "hidden OCR layer", fontsize=3 * MM, render_mode=3)
    outlined_text(page, "BUERO", 220 * MM, 110 * MM, 1.5)                                  # outlined text, 1.5 mm
    page.add_rect_annot(pymupdf.Rect(300 * MM, 150 * MM, 380 * MM, 200 * MM))
    # a measurement viewport (ISO 32000-1 12.9): 1 pt on paper = 100 x 0.3528 mm in the plan, i.e. 1:100
    c = 25.4 / 72 * 100 / 1000
    doc.xref_set_key(page.xref, "VP", f"[<</Type/Viewport/BBox[0 0 {420 * MM:.2f} {297 * MM:.2f}]/Name(Plan)"
                                      f"/Measure<</Type/Measure/Subtype/RL/R(1:100)/X[<</Type/NumberFormat/U(m)/C {c:.8f}/D 100>>]"
                                      f"/D[<</Type/NumberFormat/U(m)/C 1/D 100>>]/A[<</Type/NumberFormat/U(sq m)/C 1/D 100>>]>>>>]")
    doc.save(path)


def test_pdf_vector_text_outlines_annotations_measure(tmp_path):
    path = tmp_path / "plan.pdf"
    make_pdf(path)
    (p,) = inputs.load(path)
    assert p.input_class == "vector PDF" and p.dpi_source == "render" and p.dpi_trusted
    # the smallest text (1.5 mm outlined, 2 mm native; quantile 0.1 of 1.5, 2, 5 mm = 1.6 mm) sets the resolution:
    # 24 px / 1.6 mm -> 381 dpi, rounded up to 400
    assert p.dpi == 400 and "smallest text" in p.provenance["dpi_reason"]
    assert p.paper_mm == pytest.approx((420, 297), abs=0.1) and p.paper_source == "PDF page box"
    assert p.img.shape[:2] == pytest.approx((297 / 25.4 * 400, 420 / 25.4 * 400), abs=1)
    runs = {t["text"]: t for t in p.text if t["text"]}
    cap = runs["Grundriss 1. OG 1:100"]
    assert cap["source"] == "pdf" and not cap["invisible"] and cap["angle"] == 0
    assert cap["box"][0] == pytest.approx(220 * p.px_per_mm, abs=3)
    assert cap["height"] == pytest.approx(5 * p.px_per_mm, rel=0.01)
    assert runs["hidden OCR layer"]["invisible"]
    out = [t for t in p.text if t["outlined"]]
    assert len(out) == 1 and out[0]["text"] == "" and out[0]["box"][0] == pytest.approx(220 * p.px_per_mm, abs=6)
    assert any(w["check"] == "outlined text" for w in p.warnings)
    assert len(p.paths) >= 13 and all(q["segments"].shape[1] == 4 for q in p.paths)
    assert p.annotations["count"] == 1 and p.annotations["mask"].any()
    ys, xs = np.nonzero(p.annotations["mask"])
    assert xs.min() / p.px_per_mm == pytest.approx(300, abs=2)
    assert p.measure and p.measure[0]["scale"] == pytest.approx(100, rel=1e-4) and p.measure[0]["unit"] == "m"
    # a raster page: an embedded scan covering the page
    doc = pymupdf.open()
    page = doc.new_page(width=210 * MM, height=297 * MM)
    img = np.full((1169, 827), 255, np.uint8)                                    # A4 at 100 dpi
    cv2.rectangle(img, (100, 100), (700, 1000), 0, 4)
    ok, png = cv2.imencode(".png", img)
    page.insert_image(page.rect, stream=png.tobytes())
    doc.save(tmp_path / "scan.pdf")
    (q,) = inputs.load(tmp_path / "scan.pdf")
    assert q.input_class == "raster PDF" and q.images and q.images[0]["dpi"] == pytest.approx(100, rel=0.02)
    assert q.dpi == 150                                                          # never below input_dpi_min


def test_pdf_object_reader():
    o = inputs.parse_pdf_object("<</Type/Viewport/BBox[0 0 10.5 20]/Name(Plan \\(A\\))/Measure 12 0 R/X true>>")
    assert o == {"Type": "Viewport", "BBox": [0, 0, 10.5, 20], "Name": "Plan (A)", "Measure": ("ref", 12), "X": True}
    r = inputs.resolve(o, lambda n: "<</Subtype/RL/X[<</U(mm)/C 0.5>>]>>")
    assert inputs.measure_scale(r["Measure"])["scale"] == pytest.approx(0.0005 / (0.0254 / 72), rel=1e-3)


def dxf_with_viewports(path):
    """Model space in millimetres: a 10 x 8 m building with an interior wall and a room name. Paper space (A3, mm):
    a viewport at 1:100 showing all of it, a viewport at 1:50 showing part of it, a caption below the first."""
    doc = ezdxf.new("R2018", setup=True, units=4)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (10000, 0), (10000, 8000), (0, 8000)], close=True)
    msp.add_lwpolyline([(300, 300), (9700, 300), (9700, 7700), (300, 7700)], close=True)
    msp.add_line((5000, 300), (5000, 7700))
    msp.add_text("Wohnen", height=250).set_placement((2000, 4000))
    lay = doc.layouts.new("Plan")
    lay.page_setup(size=(420, 297), margins=(10, 10, 10, 20), units="mm")
    lay.add_viewport(center=(80, 160), size=(120, 90), view_center_point=(5000, 4000), view_height=9000)    # 1:100
    lay.add_viewport(center=(290, 160), size=(160, 120), view_center_point=(2500, 3000), view_height=6000)  # 1:50
    lay.add_text("Grundriss EG 1:100", height=3.5).set_placement((30, 100))
    lay.add_lwpolyline([(-15, -5), (395, -5), (395, 282), (-15, 282)], close=True)          # frame
    doc.saveas(path)
    return doc


def test_dxf_layout_viewports(tmp_path):
    path = tmp_path / "plan.dxf"
    dxf_with_viewports(path)
    (p,) = inputs.load(path, dpi=100)
    assert p.input_class == "DXF layout" and p.page == "Plan" and p.paper_mm == pytest.approx((420, 297))
    assert p.dpi == pytest.approx(100, rel=0.005) and p.dpi_source == "render"
    vps = sorted(p.viewports, key=lambda v: v["scale"])
    assert [v["scale"] for v in vps] == [pytest.approx(50), pytest.approx(100)]
    # model (0, 0) and (10000, 8000) of the 1:100 viewport land where paper space puts them: (30, 120) and (130, 200) mm
    m = np.array(vps[1]["model_to_px"])
    lo = (-20, -10)                                                              # paper limits: margins left 20, bottom 10
    for model, paper in (((0, 0), (30, 120)), ((10000, 8000), (130, 200))):
        x, y, _ = m @ [*model, 1]
        assert x / p.px_per_mm == pytest.approx(paper[0] - lo[0], abs=0.6)
        assert y / p.px_per_mm == pytest.approx(287 - paper[1], abs=0.6)
    ink = p.img.min(2) < 128                                                    # the building is drawn there
    x, y, _ = m @ [0, 0, 1]
    assert ink[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].any()
    assert len(vps[1]["clip_px"]) == 4
    texts = {t["text"]: t for t in p.text}
    assert "Grundriss EG 1:100" in texts and texts["Grundriss EG 1:100"]["source"] == "dxf"
    assert sum(t["text"] == "Wohnen" for t in p.text) == 2                       # model text through both viewports
    (q,) = inputs.load(path, model_space=True)
    assert q.input_class == "DXF model space" and q.model_scale["px_per_m"] == pytest.approx(100, rel=0.01)


def test_dwg_without_converter(tmp_path, monkeypatch):
    monkeypatch.delenv("FPX_DWG_CONVERTER", raising=False)
    for v in ("FPX_DWG2DXF", "FPX_QCAD_DWG2DWG", "FPX_ODAFC_EXE"):
        monkeypatch.delenv(v, raising=False)
    path = tmp_path / "plan.dwg"
    path.write_bytes(b"AC1032")
    if any(c().available() for c in inputs.DWG_CONVERTERS.values()):
        pytest.skip("a DWG converter is installed here")
    with pytest.raises(inputs.DwgConverterMissing) as e:
        inputs.load(path)
    msg = str(e.value)
    assert "libredwg" in msg and "qcad" in msg and "oda" in msg and "pipeline.md section 9" in msg
    with pytest.raises(inputs.DwgConverterMissing):
        inputs.load(path, converter="nonesuch")
