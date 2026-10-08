"""Stage 0a, normalisation: every upload becomes one sheet package per page or DWG/DXF layout (docs/pipeline.md 0a).

    from fpx import inputs
    for pkg in inputs.load("plan.pdf"):          # also .png .jpg .tif (multi-page, 1-bit) .dxf .dwg
        pkg.img, pkg.dpi, pkg.paper_mm, pkg.text, pkg.viewports, pkg.warnings

A SheetPackage holds the raster that every later stage reads (RGB uint8 at a recorded paper resolution), the paper size
and where it comes from, the input class, the vector side channels where the file has them (text runs, paths, embedded
image placements, DWG/DXF viewports, PDF measurement viewports), the transforms source <-> raster pixels <-> paper
millimetres, the provenance and warnings. Paper millimetres run from the top-left corner of the page, y down, like the
raster. Later stages never open the original file again.

Per input type:
- JPG, PNG, TIFF (multi-page and 1-bit): pages split, 1-bit expanded. The dpi is trusted by its source as in fpx.scale:
  a TIFF resolution tag of a scan ("scan") > a standard paper size at a standard dpi ("paper") > other file metadata
  ("metadata", unverified); the screen defaults 72 and 96 dpi are never trusted ("default").
- PDF, behind a small backend interface (PdfDocument; PyMuPDF now, pypdfium2 + pdfplumber could replace it): each
  page classified as vector, raster or mixed by its content; rendered at a resolution chosen from the smallest native
  text height (outlined text included), capped; rendered without annotations (the canonical raster) and, where the page
  has annotations, once more with them, kept as an overlay mask; native text runs (flagged invisible for a scanner's
  OCR layer); outlined text found heuristically (rows of small glyph-like paths without a text span), flagged for OCR;
  measurement viewports (/VP with /Measure, ISO 32000-1 12.9) read as a scale cue, never as truth.
- DXF via ezdxf: every paper-space layout rendered as plotted (ezdxf drawing add-on, PyMuPDF backend, monochrome plot
  style), each layout viewport becoming a viewport entry with its scale and clip boundary; model space only if no
  layout has content; native TEXT/MTEXT/ATTRIB into the text channel (model-space text through each viewport);
  external references flagged, missing ones as such.
- DWG: a swappable DWG -> DXF converter (DwgConverter: LibreDWG dwg2dxf, QCAD Professional dwg2dwg, ODA File Converter
  through ezdxf.addons.odafc), chosen by load(converter=...) or the FPX_DWG_CONVERTER environment variable. None is
  installed or decided (docs/pipeline.md 9), so load() on a .dwg raises DwgConverterMissing naming the options.

Limits: PDF optional-content groups, clipping paths and dash patterns are recorded as PyMuPDF reports them, not
interpreted; the paths channel approximates curves by their chords; DXF text boxes come from ezdxf's text extents;
plot style tables (CTB/STB) are not read (monochrome or ACI colours); DXF plot rotation is not applied (the layout is
rendered as laid out); xrefs are flagged, never merged. Rasters are held whole in memory (cfg.input_max_mpx caps
renders; scans are loaded at native size).
"""
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, TypedDict

import cv2
import numpy as np

from .config import DEFAULT
from .scale import DPI_TRUST, PAPER_MM, SCREEN_DPI, infer_dpi

MM_PER_INCH = 25.4
PT_MM = MM_PER_INCH / 72                                 # one PDF point in millimetres
RENDER_DPI = (150, 200, 300, 400, 600, 800, 1200)        # render resolutions are rounded up to one of these
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


# ---------------------------------------------------------------------------------------------------------------------
# Package

class TextRun(TypedDict, total=False):
    text: str                           # the string ("" for outlined text: geometry only, needs OCR)
    box: tuple                          # x0, y0, x1, y1 in raster pixels
    height: float                       # font size (em) in raster pixels
    rotation: float                     # degrees, counter-clockwise from the page x axis
    angle: int                          # 0 or 90 (fpx.text convention: horizontal or vertical)
    font: Optional[str]
    colour: Optional[tuple]
    layer: Optional[str]
    conf: float
    source: str                         # "pdf", "dxf" or "pdf-outlines"
    outlined: bool                      # text drawn as geometry (no string)
    invisible: bool                     # not rendered (a scanner's OCR layer, render mode 3)


@dataclass
class SheetPackage:
    id: str                                             # file stem plus page or layout, e.g. "plan-p1", "plan-A1"
    source: str                                         # path of the uploaded file
    page: object                                        # 0-based page index (PDF, TIFF) or layout name (DXF)
    input_class: str                                    # see INPUT_CLASSES
    img: np.ndarray                                     # RGB uint8 (H, W, 3): the canonical raster
    dpi: Optional[float] = None                         # raster pixels per paper inch; None: unknown
    dpi_source: str = "unknown"                         # render | scan | paper | metadata | default | unknown
    dpi_alternatives: list = field(default_factory=list)    # [(dpi, source, note)] for fpx.scale when dpi is uncertain
    paper_mm: Optional[tuple] = None                    # (width, height) of the page in millimetres
    paper_source: str = "unknown"                       # where the paper size comes from
    text: list = field(default_factory=list)            # [TextRun]
    paths: list = field(default_factory=list)           # vector paths: {bbox, segments (k, 4) px, width_mm, stroke, fill, layer, closed, kind}
    images: list = field(default_factory=list)          # embedded image placements: {bbox px, native_size, dpi}
    viewports: list = field(default_factory=list)       # DWG/DXF layout viewports: {id, clip_px, scale (N of 1:N), ...}
    measure: list = field(default_factory=list)         # PDF measurement viewports: {bbox_px, scale, units, ...}
    annotations: dict = field(default_factory=dict)     # PDF: {count, types, mask (bool H x W: pixels the annotations change)}
    model_scale: Optional[dict] = None                  # DXF model space: {mm_per_unit, px_per_m} exact for the whole raster
    transforms: dict = field(default_factory=dict)      # source_to_px, px_to_paper_mm (3 x 3 affine, nested lists), units
    provenance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)        # [{check, severity, message}]

    @property
    def px_per_mm(self):
        """Raster pixels per paper millimetre, or None if the resolution is unknown."""
        return None if not self.dpi else self.dpi / MM_PER_INCH

    @property
    def dpi_trusted(self):
        """The dpi is exact (a render or a scan header), not a guess from the paper size or file metadata."""
        return bool(self.dpi) and self.dpi_source in ("render", "scan")

    def grey(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    def to_paper_mm(self, xy):
        """Raster pixels -> paper millimetres (top-left origin, y down); None without a known resolution."""
        if not self.px_per_mm:
            return None
        return np.asarray(xy, float) / self.px_per_mm

    def warn(self, check, severity, message):
        self.warnings.append({"check": check, "severity": severity, "message": message})

    def summary(self):
        """JSON-friendly description without the arrays (raster, annotation mask, path geometry)."""
        h, w = self.img.shape[:2]
        return {"id": self.id, "source": str(self.source), "page": self.page, "input_class": self.input_class,
                "raster_px": [w, h], "dpi": None if self.dpi is None else round(float(self.dpi), 3),
                "dpi_source": self.dpi_source, "dpi_alternatives": [list(d) for d in self.dpi_alternatives],
                "px_per_mm": None if self.px_per_mm is None else round(self.px_per_mm, 4),
                "paper_mm": None if self.paper_mm is None else [round(v, 2) for v in self.paper_mm],
                "paper_source": self.paper_source,
                "text_runs": {"native": sum(1 for t in self.text if not t.get("outlined") and not t.get("invisible")),
                              "invisible": sum(1 for t in self.text if t.get("invisible")),
                              "outlined": sum(1 for t in self.text if t.get("outlined"))},
                "paths": len(self.paths), "images": [{k: v for k, v in im.items()} for im in self.images],
                "viewports": [{k: v for k, v in vp.items() if k != "model_to_px"} for vp in self.viewports],
                "measure": self.measure,
                "annotations": {k: v for k, v in self.annotations.items() if k != "mask"},
                "model_scale": self.model_scale, "transforms": self.transforms, "provenance": self.provenance,
                "warnings": self.warnings}


INPUT_CLASSES = ("raster image", "raster scan (1-bit)", "raster PDF", "mixed PDF", "vector PDF", "DXF layout",
                 "DXF model space", "DWG layout", "DWG model space")


def _affine(sx=1.0, sy=1.0, tx=0.0, ty=0.0):
    return [[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]]


def _apply(m, xy):
    """3 x 3 affine (nested lists or array) applied to points (..., 2)."""
    m = np.asarray(m, float)
    xy = np.asarray(xy, float)
    return xy @ m[:2, :2].T + m[:2, 2]


def _paper_transform(dpi):
    return None if not dpi else _affine(MM_PER_INCH / dpi, MM_PER_INCH / dpi)


def _render_dpi(text_mm, cfg, area_in2, floor=None):
    """Render resolution from the smallest text height on paper: the text quantile cfg.input_text_quantile gets
    cfg.input_text_px pixels of font size; rounded up to a standard dpi, kept within [input_dpi_min, input_dpi_max]
    and under cfg.input_max_mpx megapixels. Returns (dpi, reason)."""
    if text_mm:
        h = float(np.quantile(np.asarray(text_mm, float), cfg.input_text_quantile))
        want = cfg.input_text_px / max(h, 1e-3) * MM_PER_INCH
        why = f"smallest text {h:.2f} mm (quantile {cfg.input_text_quantile:g} of {len(text_mm)} runs) -> {want:.0f} dpi"
    else:
        want, why = float(cfg.input_dpi_default), "no text on the page: default resolution"
    if floor:
        want = max(want, floor)
        why += f"; at least the embedded images' {floor:.0f} dpi"
    dpi = next((d for d in RENDER_DPI if d >= want - 0.5), RENDER_DPI[-1])
    dpi = float(min(max(dpi, cfg.input_dpi_min), cfg.input_dpi_max))
    if want > cfg.input_dpi_max:
        why += f"; capped at {cfg.input_dpi_max} dpi"
    cap = math.sqrt(cfg.input_max_mpx * 1e6 / max(area_in2, 1e-6))
    if dpi > cap:
        dpi = float(math.floor(cap))
        why += f"; reduced to {dpi:.0f} dpi to stay under {cfg.input_max_mpx:g} megapixels"
    return dpi, why


# ---------------------------------------------------------------------------------------------------------------------
# Raster images (JPG, PNG, TIFF incl. multi-page and 1-bit)

def _tiff_resolution(im):
    """(dpi, unit tag) from the TIFF resolution tags, or (None, None)."""
    tags = getattr(im, "tag_v2", None)
    if not tags or 282 not in tags:
        return None, None
    unit = tags.get(296, 2)
    x = float(tags[282])
    if unit == 3:                                        # pixels per centimetre
        x *= 2.54
    elif unit != 2:                                      # no absolute unit: an aspect ratio only
        return None, unit
    return x, unit


def image_dpi(im, shape, fmt):
    """(dpi or None, dpi_source, alternatives, note) of a raster file by the trust order of docs/pipeline.md 0a:
    TIFF resolution tags of a scan > paper-size match > other metadata; screen defaults (72, 96) never."""
    paper = infer_dpi(shape)                             # [(dpi, "paper", note)], ambiguous by nature
    meta, src = None, None
    if fmt == "TIFF":
        meta, unit = _tiff_resolution(im)
        src = "scan" if meta else None
    if meta is None:
        d = im.info.get("dpi")
        meta = float(d[0]) if d and d[0] and d[0] > 1 else None
        src = "metadata" if meta else None
    if meta is not None and int(round(meta)) in SCREEN_DPI:
        alts = paper + [(meta, "metadata", f"image metadata {meta:g} dpi (screen default, not trusted)")]
        return None, "default", alts, f"{meta:g} dpi in the file is a screen default: not trusted"
    if meta is not None:
        agree = any(abs(p - meta) < 0.5 for p, _, _ in paper)
        if src == "scan":
            return meta, "scan", [(meta, "scan", f"TIFF resolution tags {meta:g} dpi")], "TIFF resolution tags"
        if agree:
            return meta, "paper", [(meta, "paper", f"metadata {meta:g} dpi matches a standard paper size")], \
                "file metadata agrees with a standard paper size"
        return meta, "metadata", [(meta, "metadata", f"image metadata {meta:g} dpi (unverified)")] + paper, \
            "file metadata, unverified"
    if len(paper) == 1:
        d, _, note = paper[0]
        return d, "paper", paper, f"no dpi in the file; the pixel size matches {note}"
    return None, "unknown", paper, ("no dpi in the file; the pixel size fits " + ", ".join(n for _, _, n in paper)
                                    if paper else "no dpi in the file and no standard paper size fits")


def _exif_camera(im):
    try:
        ex = im.getexif()
        return bool(ex.get(271) or ex.get(272))           # Make, Model
    except Exception:
        return False


def load_image(path, cfg=DEFAULT, pages=None, **_):
    from PIL import Image, ImageSequence
    Image.MAX_IMAGE_PIXELS = None
    path = Path(path)
    out = []
    with Image.open(path) as im:
        fmt = im.format or path.suffix.upper().lstrip(".")
        n = getattr(im, "n_frames", 1)
        for k, frame in enumerate(ImageSequence.Iterator(im)):
            if pages is not None and k not in pages:
                continue
            mode = frame.mode
            if mode in ("RGBA", "LA", "P", "PA"):
                rgba = frame.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"))
            elif mode == "1":
                rgb = np.asarray(frame.convert("L").convert("RGB"))       # 1-bit expanded to 0/255
            elif mode in ("I;16", "I;16B", "I", "F"):
                a = np.asarray(frame, np.float32)
                lo, hi = np.percentile(a, (0.5, 99.5))
                g = np.clip((a - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
                rgb = np.repeat(g[..., None], 3, axis=2)
            else:
                rgb = np.asarray(frame.convert("RGB"))
            rgb = np.ascontiguousarray(rgb)
            dpi, src, alts, note = image_dpi(frame, rgb.shape, fmt)
            cls = "raster scan (1-bit)" if mode == "1" else "raster image"
            pid = path.stem if n == 1 else f"{path.stem}-p{k + 1}"
            pkg = SheetPackage(pid, str(path), k, cls, rgb, dpi=dpi, dpi_source=src, dpi_alternatives=alts)
            if dpi:
                pkg.paper_mm = (rgb.shape[1] / dpi * MM_PER_INCH, rgb.shape[0] / dpi * MM_PER_INCH)
                pkg.paper_source = f"raster size at {dpi:g} dpi ({src}: {note})"
            pkg.transforms = {"source_units": "image pixels", "source_to_px": _affine(),
                              "px_to_paper_mm": _paper_transform(dpi)}
            pkg.provenance = {"loader": "fpx.inputs.load_image (Pillow)", "format": fmt, "mode": mode, "frame": k,
                              "frames": n, "dpi_note": note}
            if src in ("default", "unknown"):
                pkg.warn("dpi", "medium", f"resolution unknown ({note}): the scale needs a direct cue")
            elif src in ("metadata", "paper"):
                pkg.warn("dpi", "low", f"resolution {dpi:g} dpi from {note}: a guess, to be confirmed by a direct cue")
            if fmt in ("JPEG", "MPO") and _exif_camera(frame):
                pkg.warn("photo", "medium", "camera EXIF data: possibly a photograph (perspective, uneven light); "
                                            "check flatness at triage")
            if max(rgb.shape[:2]) > 20000:
                pkg.warn("size", "low", f"very large raster {rgb.shape[1]} x {rgb.shape[0]} px: later stages tile it")
            out.append(pkg)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# PDF: backend interface and the PyMuPDF backend

class PdfDocument(Protocol):
    """What the PDF loader needs from a PDF library. Coordinates are page points, y down from the top-left corner of the
    displayed (rotated, cropped) page; the backend applies /Rotate and the crop box. A pypdfium2 (render) + pdfplumber
    (text, paths) backend can implement the same methods."""
    name: str

    def page_count(self) -> int: ...
    def page_size(self, i) -> tuple: ...                         # (width_pt, height_pt, user_unit, rotation)
    def text_spans(self, i) -> list: ...                         # {text, bbox, size, font, colour, dir, invisible, chars}
    def paths(self, i) -> list: ...                              # {bbox, segments (k, 4), width, stroke, fill, layer, closed, kind, curves}
    def images(self, i) -> list: ...                             # {bbox, native_size}
    def annotations(self, i) -> list: ...                        # {type, bbox}
    def measure_viewports(self, i) -> list: ...                  # {bbox, name, measure (parsed dict)}
    def render(self, i, dpi, annotations=False, clip=None) -> np.ndarray: ...   # RGB uint8 (clip in page points)
    def close(self) -> None: ...


class PyMuPdfDocument:
    """PdfDocument on PyMuPDF (AGPL-3.0 or commercial; see docs/pipeline.md 6 for the permissive alternative)."""
    name = "pymupdf"

    def __init__(self, path):
        import pymupdf
        self.mu = pymupdf
        self.doc = pymupdf.open(str(path))
        self.version = pymupdf.VersionBind

    def page_count(self):
        return len(self.doc)

    def _page(self, i):
        return self.doc[i]

    def _m(self, page):
        """Unrotated page space -> displayed page space (PyMuPDF reports text and paths unrotated)."""
        m = page.rotation_matrix
        return np.array([[m.a, m.c, m.e], [m.b, m.d, m.f], [0, 0, 1]], float)

    def user_unit(self, page):
        t, v = self.doc.xref_get_key(page.xref, "UserUnit")
        try:
            return float(v) if t != "null" else 1.0
        except ValueError:
            return 1.0

    def page_size(self, i):
        p = self._page(i)
        return float(p.rect.width), float(p.rect.height), self.user_unit(p), int(p.rotation)

    def text_spans(self, i):
        p = self._page(i)
        m = self._m(p)
        out = []
        for b in p.get_text("rawdict")["blocks"]:
            for ln in b.get("lines", []):
                d = np.array(ln["dir"], float) @ m[:2, :2].T
                for sp in ln["spans"]:
                    flags = sp.get("char_flags", 16)
                    if flags & 64:                       # a clipping copy of a span (render modes 4-7): skip
                        continue
                    chars = [(c["c"], _box(m, c["bbox"])) for c in sp.get("chars", [])]
                    text = "".join(c for c, _ in chars)
                    if not text.strip():
                        continue
                    out.append({"text": text, "bbox": _box(m, sp["bbox"]), "size": float(sp["size"]),
                                "font": sp.get("font"), "colour": _rgb_int(sp.get("color", 0)), "dir": tuple(d),
                                "invisible": sp.get("alpha", 255) == 0 or (flags & 0x30) == 0, "chars": chars})
        return out

    def paths(self, i):
        p = self._page(i)
        m = self._m(p)
        out = []
        for d in p.get_drawings():
            segs, curves = [], 0
            for it in d["items"]:
                op = it[0]
                if op == "l":
                    segs.append((*it[1], *it[2]))
                elif op == "c":
                    segs.append((*it[1], *it[4]))
                    curves += 1
                elif op == "re":
                    r = it[1]
                    segs += [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
                elif op == "qu":
                    q = it[1]
                    pts = [q.ul, q.ur, q.lr, q.ll]
                    segs += [(*pts[k], *pts[(k + 1) % 4]) for k in range(4)]
            if not segs:
                continue
            s = np.asarray(segs, float).reshape(-1, 2, 2)
            s = (s @ m[:2, :2].T + m[:2, 2]).reshape(-1, 4)
            first, last = s[0, :2], s[-1, 2:]
            out.append({"bbox": (float(s[:, [0, 2]].min()), float(s[:, [1, 3]].min()), float(s[:, [0, 2]].max()),
                                 float(s[:, [1, 3]].max())),
                        "segments": s.astype(np.float32), "width": d.get("width"), "kind": d.get("type"),
                        "stroke": _rgb_tuple(d.get("color")), "fill": _rgb_tuple(d.get("fill")),
                        "layer": d.get("layer") or None, "dashes": d.get("dashes"),
                        "closed": bool(d.get("closePath")) or d.get("type") in ("f", "fs") or bool(np.allclose(first, last)),
                        "curves": curves})
        return out

    def images(self, i):
        p = self._page(i)
        m = self._m(p)
        out = []
        for info in p.get_image_info():
            bx = _box(m, info["bbox"])
            out.append({"bbox": bx, "native_size": (int(info.get("width", 0)), int(info.get("height", 0)))})
        return out

    def annotations(self, i):
        p = self._page(i)
        return [{"type": a.type[1], "bbox": tuple(a.rect)} for a in p.annots()]

    def measure_viewports(self, i):
        p = self._page(i)
        t, v = self.doc.xref_get_key(p.xref, "VP")
        if t == "null":
            return []
        obj = resolve(parse_pdf_object(v), self._xref_text)
        vps = obj if isinstance(obj, list) else [obj]
        tm = p.transformation_matrix                    # PDF user space (y up) -> unrotated page space (y down)
        to_page = np.array([[tm.a, tm.c, tm.e], [tm.b, tm.d, tm.f], [0, 0, 1]], float)
        m = self._m(p) @ to_page
        out = []
        for vp in vps:
            if not isinstance(vp, dict):
                continue
            bb = vp.get("BBox")
            box = _box(m, bb) if isinstance(bb, list) and len(bb) == 4 else None
            out.append({"bbox": box, "name": vp.get("Name"), "measure": vp.get("Measure")})
        return out

    def _xref_text(self, n):
        return self.doc.xref_object(n, compressed=True)

    def render(self, i, dpi, annotations=False, clip=None):
        """The page (or the clip (x0, y0, x1, y1) in displayed page points) as RGB at dpi."""
        p = self._page(i)
        z = dpi * self.user_unit(p) / 72.0
        r = None
        if clip is not None:                             # displayed page space -> unrotated page space for PyMuPDF
            r = self.mu.Rect(clip) * p.derotation_matrix
            r.normalize()
        pix = p.get_pixmap(matrix=self.mu.Matrix(z, z), colorspace=self.mu.csRGB, alpha=False, annots=annotations, clip=r)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()

    def close(self):
        self.doc.close()


PDF_BACKENDS = {"pymupdf": PyMuPdfDocument}


def rerender(pkg, box_px, dpi):
    """Part of a PDF sheet package rendered again from the vector source at a higher dpi (box in the package's raster
    pixels, without annotations), or None if the package is not a PDF page. Used where a drawing's scale asks for more
    pixels per metre than the sheet render has (OCR of small lettering on reduced prints)."""
    if not str(pkg.source).lower().endswith(".pdf") or not isinstance(pkg.page, int) or "PDF" not in pkg.input_class:
        return None
    z = np.asarray(pkg.transforms["source_to_px"], float)[0, 0]          # raster px per page point
    clip = tuple(v / z for v in box_px)
    doc = PDF_BACKENDS[pkg.provenance.get("backend", "pymupdf").split()[0]](pkg.source)
    try:
        return doc.render(pkg.page, dpi, annotations=False, clip=clip)
    finally:
        doc.close()


def _box(m, b):
    pts = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], float) @ m[:2, :2].T + m[:2, 2]
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def _rgb_int(c):
    try:
        c = int(c)
    except (TypeError, ValueError):
        return None
    return ((c >> 16) & 255, (c >> 8) & 255, c & 255)


def _rgb_tuple(c):
    if c is None:
        return None
    if len(c) == 1:
        c = (c[0],) * 3
    elif len(c) == 4:                                    # CMYK
        k = c[3]
        c = ((1 - c[0]) * (1 - k), (1 - c[1]) * (1 - k), (1 - c[2]) * (1 - k))
    return tuple(int(round(255 * v)) for v in c[:3])


# A small reader of PDF object syntax for /VP (PyMuPDF returns objects as text; references are resolved on demand)

_TOK = re.compile(r"\s*(?:(<<|>>|\[|\]|\{|\})|(/[^\s/<>\[\]()%{}]*)|(\((?:\\.|[^\\()]|\((?:\\.|[^\\()])*\))*\))"
                  r"|(<[0-9A-Fa-f\s]*>)|([+-]?(?:\d+\.?\d*|\.\d+))|(true|false|null)|(R)\b)", re.S)


def parse_pdf_object(text):
    """PDF object syntax -> Python: dict, list, str (names without the slash, strings unescaped), float/int, bool,
    None, ("ref", n) for indirect references."""
    toks = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        if text[pos] == "%":                             # comment to the end of the line
            nl = text.find("\n", pos)
            pos = len(text) if nl < 0 else nl + 1
            continue
        m = _TOK.match(text, pos)
        if not m or m.end() == pos:
            pos += 1                                     # skip what this reader does not know
            continue
        toks.append(m)
        pos = m.end()

    def value(k):
        m = toks[k]
        delim, name, string, hexs, num, kw, ref = m.groups()
        if delim == "<<":
            out, k = {}, k + 1
            while k < len(toks) and toks[k].group(1) != ">>":
                key = toks[k].group(2)
                v, k = value(k + 1)
                if key:
                    out[key[1:]] = v
            return out, k + 1
        if delim == "[":
            out, k = [], k + 1
            while k < len(toks) and toks[k].group(1) != "]":
                v, k = value(k)
                out.append(v)
            return out, k + 1
        if name is not None:
            return name[1:], k + 1
        if string is not None:
            s = string[1:-1]
            s = re.sub(r"\\([nrtbf()\\])", lambda q: {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}.get(q.group(1), q.group(1)), s)
            return s, k + 1
        if hexs is not None:
            h = re.sub(r"\s", "", hexs[1:-1])
            try:
                return bytes.fromhex(h + "0" * (len(h) % 2)).decode("latin-1"), k + 1
            except ValueError:
                return h, k + 1
        if num is not None:
            # "n g R": an indirect reference
            if k + 2 < len(toks) and toks[k + 1].group(5) is not None and toks[k + 2].group(7):
                return ("ref", int(float(num))), k + 3
            v = float(num)
            return (int(v) if v.is_integer() and "." not in num else v), k + 1
        if kw is not None:
            return {"true": True, "false": False, "null": None}[kw], k + 1
        return None, k + 1

    if not toks:
        return None
    v, _ = value(0)
    return v


def resolve(obj, xref_text, depth=0):
    """Replace ("ref", n) by the parsed object n (xref_text(n) -> its text), to a limited depth."""
    if depth > 6:
        return obj
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "ref":
        try:
            return resolve(parse_pdf_object(xref_text(obj[1])), xref_text, depth + 1)
        except Exception:
            return None
    if isinstance(obj, dict):
        return {k: resolve(v, xref_text, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v, xref_text, depth + 1) for v in obj]
    return obj


UNIT_M = {"m": 1.0, "mm": 0.001, "cm": 0.01, "km": 1000.0, "in": 0.0254, "inch": 0.0254, "ft": 0.3048,
          "yd": 0.9144, "mi": 1609.344, "dm": 0.1}


def measure_scale(measure, user_unit=1.0):
    """Scale of a PDF measurement dictionary (/Measure, subtype RL): /X[0] /C converts page units to the unit /U.
    Returns {"scale": N of 1:N (paper to plan), "metres_per_pt", "unit", "ratio" (the /R label), "anisotropy"} or None."""
    if not isinstance(measure, dict) or measure.get("Subtype", "RL") != "RL":
        return None
    out = {"ratio": measure.get("R")}
    vals = {}
    for axis in ("X", "Y"):
        fmt = measure.get(axis)
        if isinstance(fmt, list) and fmt and isinstance(fmt[0], dict):
            c, u = fmt[0].get("C"), str(fmt[0].get("U", "")).strip().lower()
            if isinstance(c, (int, float)) and c > 0 and u in UNIT_M:
                vals[axis] = c * UNIT_M[u]
                out["unit"] = u
    if "X" not in vals:
        return None
    paper_m = 0.0254 / 72 * user_unit
    out["metres_per_pt"] = vals["X"]
    out["scale"] = round(vals["X"] / paper_m, 3)
    if "Y" in vals:
        out["anisotropy"] = round(vals["Y"] / vals["X"], 5)
    return out


# PDF loader

def _span_runs(sp, zoom_m, user_unit, gap=0.8):
    """Text runs of one span, split where characters are further apart than gap x font size (CAD text entities that
    the text extractor joined into one line), boxes and heights in raster pixels."""
    size = sp["size"]
    chars = sp["chars"] or [(sp["text"], sp["bbox"])]
    d = np.asarray(sp["dir"], float)
    horiz = abs(d[0]) >= abs(d[1])
    runs, cur = [], []
    for c, b in chars:
        if cur:
            prev = cur[-1][1]
            g = (b[0] - prev[2]) if horiz and d[0] >= 0 else (prev[0] - b[2]) if horiz else \
                (b[1] - prev[3]) if d[1] >= 0 else (prev[1] - b[3])
            if g > gap * size:
                runs.append(cur)
                cur = []
        cur.append((c, b))
    if cur:
        runs.append(cur)
    out = []
    rot = math.degrees(math.atan2(-d[1], d[0]))
    for r in runs:
        text = "".join(c for c, _ in r).strip()
        if not text:
            continue
        b = np.array([q[1] for q in r], float)
        box = _box(zoom_m, (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()))
        out.append({"text": text, "box": box, "height": float(size * zoom_m[0, 0]),
                    "rotation": round(rot, 2), "angle": 0 if horiz else 90, "font": sp.get("font"),
                    "colour": sp.get("colour"), "layer": None, "conf": 1.0, "source": "pdf", "outlined": False,
                    "invisible": bool(sp["invisible"]), "size_mm": size * PT_MM * user_unit})
    return out


def find_outlined_text(paths, text_boxes, mm_per_unit, cfg=DEFAULT):
    """Rows of small glyph-like paths without a text span: text exported as geometry (SHX fonts always are; TrueType
    text when 'convert text to outlines' was set). A heuristic: glyph candidates are paths no larger than
    cfg.outlined_glyph_max_mm on paper and at least cfg.outlined_glyph_min_mm high or wide; they are joined into rows
    (gaps under 0.6 x the glyph height, along x or along y for vertical text); a row of at least cfg.outlined_min_glyphs
    glyphs, one glyph high, in which most glyphs have curves or strokes in two directions (hatch and stair treads are
    parallel strokes), and that native text covers less than 30 %, counts as outlined text.
    Coordinates in source units; returns [{"box", "height", "glyphs", "rotation"}]."""
    lo, hi = cfg.outlined_glyph_min_mm / mm_per_unit, cfg.outlined_glyph_max_mm / mm_per_unit
    glyphs = []
    for p in paths:
        x0, y0, x1, y1 = p["bbox"]
        w, h = x1 - x0, y1 - y0
        if max(w, h) > hi or max(w, h) < lo:
            continue
        s = p["segments"]
        ang = np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) % 180
        ln = np.hypot(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])
        dirs = len({int(a // 30) for a, l_ in zip(ang, ln) if l_ > 0.15 * max(w, h)})
        shaped = p.get("curves", 0) > 0 or dirs >= 2 or (p["closed"] and len(s) >= 4)
        glyphs.append((x0, y0, x1, y1, shaped))
    if len(glyphs) < cfg.outlined_min_glyphs:
        return []
    g = np.array(glyphs, float)
    found = []
    taken = np.zeros(len(g), bool)                       # glyphs of accepted rows
    for axis in (0, 1):                                  # rows along x (horizontal text), then along y (vertical)
        size = np.maximum(g[:, 3] - g[:, 1], 1e-9) if axis == 0 else np.maximum(g[:, 2] - g[:, 0], 1e-9)
        order = np.lexsort((g[:, axis], np.round((g[:, 1 - axis] + g[:, 3 - axis]) / 2 / np.median(size))))
        used = taken.copy()
        for i in order:
            if used[i]:
                continue
            row = [i]
            used[i] = True
            # grow the row along the axis: neighbours on about the same centre line, within 0.6 x height
            while True:
                rb = g[row]
                ca, cb = rb[:, 1 - axis].min(), rb[:, 3 - axis].max()
                hh = float(np.median(size[row]))
                ea, eb = rb[:, axis].min() - 0.6 * hh, rb[:, 2 + axis].max() + 0.6 * hh
                cand = np.flatnonzero(~used & (g[:, 2 + axis] >= ea) & (g[:, axis] <= eb)
                                      & (np.minimum(g[:, 3 - axis], cb) - np.maximum(g[:, 1 - axis], ca) > 0.5 * np.minimum(size, hh))
                                      & (size < 2.2 * hh) & (size > 0.3 * hh))
                if not len(cand):
                    break
                row += list(cand)
                used[cand] = True                        # a rejected row keeps its glyphs: no glyph seeds twice
            if len(row) < cfg.outlined_min_glyphs:
                continue
            rb = g[row]
            box = (rb[:, 0].min(), rb[:, 1].min(), rb[:, 2].max(), rb[:, 3].max())
            length, across = (box[2] - box[0], box[3] - box[1]) if axis == 0 else (box[3] - box[1], box[2] - box[0])
            hh = float(np.median(size[row]))
            if across > 1.8 * hh or length < 1.5 * across or rb[:, 4].mean() < 0.5:
                continue                                 # a block of symbols or hatch, not one line of text
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-12)
            cov = sum(_overlap(box, t) for t in text_boxes)
            if cov > 0.3 * area:
                continue
            taken[row] = True
            found.append({"box": tuple(float(v) for v in box), "height": hh, "glyphs": len(row),
                          "rotation": 0.0 if axis == 0 else 90.0})
    return found


def _overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _coverage(boxes, w, h):
    """Share of the page (w x h) covered by the union of boxes (on a coarse grid)."""
    if not boxes:
        return 0.0
    g = np.zeros((200, 200), bool)
    for x0, y0, x1, y1 in boxes:
        g[max(0, int(y0 / h * 200)):max(0, int(math.ceil(y1 / h * 200))), max(0, int(x0 / w * 200)):max(0, int(math.ceil(x1 / w * 200)))] = True
    return float(g.mean())


def classify_pdf_page(n_paths, image_share, cfg=DEFAULT):
    """Vector, raster or mixed by content: a page mostly covered by images with hardly any paths is a scan; images and
    paths together are mixed (a scan with vector redlines, or a vector plan with raster parts)."""
    if image_share >= cfg.input_raster_page_share:
        return "raster PDF" if n_paths < cfg.input_vector_min_paths else "mixed PDF"
    if image_share >= 0.05 and n_paths >= cfg.input_vector_min_paths:
        return "mixed PDF"
    if image_share >= 0.05:
        return "raster PDF"
    return "vector PDF"


def load_pdf(path, cfg=DEFAULT, pages=None, dpi=None, backend="pymupdf", with_paths=True, **_):
    path = Path(path)
    doc = PDF_BACKENDS[backend](path)
    out = []
    try:
        for i in range(doc.page_count()):
            if pages is not None and i not in pages:
                continue
            out.append(_pdf_page(doc, path, i, cfg, dpi, with_paths))
    finally:
        doc.close()
    return out


def _pdf_page(doc, path, i, cfg, force_dpi, with_paths):
    w_pt, h_pt, uu, rot = doc.page_size(i)
    mm_per_pt = PT_MM * uu
    spans = doc.text_spans(i)
    paths = doc.paths(i)
    images = doc.images(i)
    annots = doc.annotations(i)
    vps = doc.measure_viewports(i)
    img_share = _coverage([im["bbox"] for im in images], w_pt, h_pt)
    cls = classify_pdf_page(len(paths), img_share, cfg)
    visible = [s for s in spans if not s["invisible"]]
    outlined = find_outlined_text(paths, [s["bbox"] for s in visible], mm_per_pt, cfg) if cls != "raster PDF" else []
    # resolution: from the smallest text (native and outlined), and never below the embedded scans' own resolution
    text_mm = [s["size"] * mm_per_pt for s in visible] + [o["height"] * mm_per_pt for o in outlined]
    img_dpi = [im["native_size"][0] / max((im["bbox"][2] - im["bbox"][0]) * mm_per_pt / MM_PER_INCH, 1e-6)
               for im in images if im["native_size"][0]]
    big = [d for d, im in zip(img_dpi, images) if (im["bbox"][2] - im["bbox"][0]) * (im["bbox"][3] - im["bbox"][1]) > 0.3 * w_pt * h_pt]
    area_in2 = (w_pt * mm_per_pt / MM_PER_INCH) * (h_pt * mm_per_pt / MM_PER_INCH)
    if force_dpi:
        dpi, why = float(force_dpi), "set by the caller"
    elif cls == "raster PDF" and big:                   # a scan: its own resolution, rounded up, nothing invented
        want = max(big)
        dpi = float(min(max(next((d for d in RENDER_DPI if d >= want - 0.5), RENDER_DPI[-1]), cfg.input_dpi_min),
                        cfg.input_dpi_max))
        dpi = min(dpi, float(math.floor(math.sqrt(cfg.input_max_mpx * 1e6 / max(area_in2, 1e-6)))))
        why = f"scanned page: the page image's {want:.0f} dpi, rendered at {dpi:g} dpi"
    else:
        dpi, why = _render_dpi(text_mm, cfg, area_in2, floor=max(big) if big else None)
    z = dpi / MM_PER_INCH * mm_per_pt                    # raster pixels per page point
    zm = np.array([[z, 0, 0], [0, z, 0], [0, 0, 1]], float)
    img = doc.render(i, dpi, annotations=False)
    pid = f"{path.stem}-p{i + 1}"
    pkg = SheetPackage(pid, str(path), i, cls, img, dpi=dpi, dpi_source="render",
                       dpi_alternatives=[(dpi, "render", f"PDF rendered at {dpi:g} dpi")],
                       paper_mm=(w_pt * mm_per_pt, h_pt * mm_per_pt),
                       paper_source="PDF page box" + (f" x UserUnit {uu:g}" if uu != 1 else ""))
    for sp in spans:
        pkg.text += _span_runs(sp, zm, uu)
    for o in outlined:
        pkg.text.append({"text": "", "box": _box(zm, o["box"]), "height": o["height"] * z, "rotation": o["rotation"],
                         "angle": 0 if o["rotation"] == 0 else 90, "font": None, "colour": None, "layer": None,
                         "conf": 0.0, "source": "pdf-outlines", "outlined": True, "invisible": False,
                         "glyphs": o["glyphs"], "size_mm": o["height"] * mm_per_pt})
    if with_paths:
        for p in paths:
            pkg.paths.append({"bbox": _box(zm, p["bbox"]), "segments": (p["segments"] * z).astype(np.float32),
                              "width_mm": None if p["width"] is None else round(p["width"] * mm_per_pt, 4),
                              "stroke": p["stroke"], "fill": p["fill"], "layer": p["layer"], "dashes": p["dashes"],
                              "closed": p["closed"], "kind": p["kind"]})
    for im, d in zip(images, img_dpi + [None] * (len(images) - len(img_dpi))):
        pkg.images.append({"bbox": [round(v, 1) for v in _box(zm, im["bbox"])], "native_size": list(im["native_size"]),
                           "dpi": None if d is None else round(d, 1)})
    for vp in vps:
        sc = measure_scale(vp.get("measure"), uu)
        pkg.measure.append({"bbox_px": None if vp["bbox"] is None else [round(v, 1) for v in _box(zm, vp["bbox"])],
                            "name": vp.get("name"), **(sc or {"scale": None, "note": "measure dictionary not understood"})})
    if annots:
        with_a = doc.render(i, dpi, annotations=True)
        diff = np.abs(with_a.astype(np.int16) - img.astype(np.int16)).max(2) > 24
        pkg.annotations = {"count": len(annots), "types": sorted({a["type"] for a in annots}),
                           "changed_share": round(float(diff.mean()), 5), "mask": diff}
        del with_a
        pkg.warn("annotations", "medium", f"{len(annots)} annotations ({', '.join(pkg.annotations['types'])}): the raster "
                                          "is rendered without them; their pixels are in annotations['mask']")
    inv = sum(1 for s in spans if s["invisible"])
    if inv:
        pkg.warn("text layer", "low", f"{inv} invisible text spans (a scanner's OCR layer): flagged, to be checked against OCR")
    if outlined:
        pkg.warn("outlined text", "medium", f"{len(outlined)} rows of outlined text ({sum(o['glyphs'] for o in outlined)} "
                                             "glyph paths without a text span): to be read by OCR")
    if cls != "vector PDF":
        pkg.warn("input class", "low", f"{cls}: images cover {img_share:.0%} of the page; the image is treated as a scan")
    if pkg.measure:
        pkg.warn("measure", "low", f"{len(pkg.measure)} PDF measurement viewports: a scale cue, never taken as truth")
    pkg.transforms = {"source_units": "PDF points of the displayed page (rotation and crop box applied), y down",
                      "source_to_px": zm.tolist(), "px_to_paper_mm": _paper_transform(dpi), "page_rotation": rot,
                      "user_unit": uu}
    pkg.provenance = {"loader": "fpx.inputs.load_pdf", "backend": f"{doc.name} {getattr(doc, 'version', '')}".strip(),
                      "page": i, "pages": doc.page_count(), "dpi_reason": why, "paths": len(paths),
                      "text_spans": len(spans), "image_share": round(img_share, 3),
                      "render": "without annotations" + ("; annotation mask from a second render" if annots else "")}
    return pkg


# ---------------------------------------------------------------------------------------------------------------------
# DXF (ezdxf) and DWG (through a converter)

def _insunits_mm(doc):
    from .conformance import INSUNITS_TO_MM
    u = int(doc.header.get("$INSUNITS", 0) or 0)
    return INSUNITS_TO_MM.get(u), u


def _dxf_texts(entities, transform=None):
    """TEXT, MTEXT and ATTRIB of an entity space (INSERTs expanded one level deep for attributes and block text) ->
    [(text, (x0, y0, x1, y1) in the space's units, height, rotation degrees, layer)]; transform maps points into
    another space (model -> paper through a viewport)."""
    import ezdxf.bbox
    out = []

    def add(e):
        t = e.dxftype()
        if t not in ("TEXT", "MTEXT", "ATTRIB"):
            return
        s = e.plain_text() if t == "MTEXT" else e.dxf.text
        s = (s or "").replace("\n", " ").strip()
        if not s:
            return
        try:
            bb = ezdxf.bbox.extents([e], fast=True)
        except Exception:
            return
        if not bb.has_data:
            return
        pts = np.array([[bb.extmin.x, bb.extmin.y], [bb.extmax.x, bb.extmin.y], [bb.extmax.x, bb.extmax.y],
                        [bb.extmin.x, bb.extmax.y]], float)
        h = float(e.dxf.char_height if t == "MTEXT" else e.dxf.height)
        rot = float(e.dxf.get("rotation", 0.0) or 0.0)
        if transform is not None:
            pts = transform(pts)
            h *= transform.scale
            rot += transform.twist
        out.append((s, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()), h, rot, e.dxf.layer))

    for e in entities:
        if e.dxftype() == "INSERT":
            for a in e.attribs:
                add(a)
            try:
                for v in e.virtual_entities():
                    add(v)
            except Exception:
                pass
        else:
            add(e)
    return out


class _VpTransform:
    """Model space -> paper space of one viewport (top views only), with its scale and twist."""

    def __init__(self, vp):
        self.m = vp.get_transformation_matrix()
        self.scale = vp.get_scale()
        self.twist = float(vp.dxf.get("view_twist_angle", 0.0) or 0.0)

    def __call__(self, pts):
        return np.array([[v.x, v.y] for v in self.m.transform_vertices([(float(x), float(y), 0.0) for x, y in pts])], float)


def _xref_warnings(doc, path):
    out = []
    for block in doc.blocks:
        b = block.block                                  # the BLOCK entity carries the xref flags and path
        if b is None or not (b.is_xref or b.is_xref_overlay):
            continue
        ref = b.dxf.get("xref_path", "") or ""
        cand = [Path(ref), Path(path).parent / ref, Path(path).parent / Path(ref.replace("\\", "/")).name]
        found = any(c.exists() for c in cand if str(c))
        out.append({"check": "xref", "severity": "medium" if found else "high",
                    "message": f"external reference {block.name!r} ({ref or 'no path'}) "
                               + ("present beside the file but not merged: content missing from the render"
                                  if found else "cannot be resolved: content missing; ask for the complete package (eTransmit)")})
    return out


def _clip_polygon(doc, vp):
    """Viewport clip boundary in paper-space units: the non-rectangular clipping entity if set, else its rectangle."""
    if vp.has_extended_clipping_path:
        try:
            from ezdxf import path as ezpath
            e = doc.entitydb.get(vp.dxf.clipping_boundary_handle)
            if e is not None:
                pts = [(v.x, v.y) for v in ezpath.make_path(e).flattening(0.1)]
                if len(pts) >= 3:
                    return pts, "clipping entity"
        except Exception:
            pass
    return [(v.x, v.y) for v in vp.clipping_rect_corners()], "rectangle"


def _render_dxf(doc, layout_obj, lo, hi, page_w_mm, page_h_mm, dpi, monochrome):
    from ezdxf.addons.drawing import Frontend, RenderContext, config as dcfg, layout as dlayout
    from ezdxf.addons.drawing import pymupdf as dpymupdf
    from ezdxf.math import BoundingBox2d
    ctx = RenderContext(doc)
    ctx.set_current_layout(layout_obj)
    backend = dpymupdf.PyMuPdfBackend()
    conf = dcfg.Configuration(background_policy=dcfg.BackgroundPolicy.WHITE,
                              color_policy=dcfg.ColorPolicy.BLACK if monochrome else dcfg.ColorPolicy.COLOR_NEGATIVE)
    Frontend(ctx, backend, config=conf).draw_layout(layout_obj, finalize=True)
    page = dlayout.Page(page_w_mm, page_h_mm, dlayout.Units.mm, margins=dlayout.Margins.all(0))
    png = backend.get_pixmap_bytes(page, fmt="png", settings=dlayout.Settings(fit_page=True), dpi=int(round(dpi)),
                                   render_box=BoundingBox2d([lo, hi]))
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_dxf(path, cfg=DEFAULT, layouts=None, dpi=None, monochrome=True, model_space=None, origin_class="DXF", **_):
    import ezdxf
    from ezdxf import recover
    path = Path(path)
    warnings = []
    try:
        doc = ezdxf.readfile(str(path))
    except ezdxf.DXFStructureError:
        doc, auditor = recover.readfile(str(path))
        warnings.append({"check": "dxf", "severity": "medium",
                         "message": f"damaged DXF recovered: {len(auditor.errors)} errors, {len(auditor.fixes)} fixes"})
    mm_unit, insunits = _insunits_mm(doc)
    if mm_unit is None:
        warnings.append({"check": "units", "severity": "high",
                         "message": f"$INSUNITS {insunits} (unitless or unknown): model units assumed to be millimetres"})
        mm_unit = 1.0
    warnings += _xref_warnings(doc, path)
    paper = []
    for name in doc.layouts.names_in_taborder():
        if name.lower() == "model" or (layouts is not None and name not in layouts):
            continue
        lay = doc.layouts.get(name)
        content = [e for e in lay if not (e.dxftype() == "VIEWPORT" and e.dxf.get("id", 0) == 1)]
        if content:
            paper.append(lay)
    common = {"cfg": cfg, "dpi": dpi, "monochrome": monochrome, "mm_unit": mm_unit, "insunits": insunits,
              "warnings": warnings, "origin_class": origin_class}
    if paper and not model_space:
        return [_dxf_layout(doc, path, lay, **common) for lay in paper]
    return [_dxf_model(doc, path, **common)]


def _dxf_layout(doc, path, lay, cfg, dpi, monochrome, mm_unit, insunits, warnings, origin_class):
    import ezdxf.bbox
    unit_mm = 25.4 if lay.dxf.get("plot_paper_units", 1) == 0 else 1.0      # paper-space unit in millimetres
    lo, hi = lay.get_paper_limits()
    pw, ph = float(lay.dxf.get("paper_width", 0) or 0), float(lay.dxf.get("paper_height", 0) or 0)
    w_mm, h_mm = (hi.x - lo.x) * unit_mm, (hi.y - lo.y) * unit_mm
    paper_source = f"DXF layout paper size ({lay.dxf.get('paper_size', '') or 'unnamed'}) and limits"
    if w_mm <= 1 or h_mm <= 1 or (pw > 1 and ph > 1 and abs(w_mm * h_mm / (pw * ph) - 1) > 0.05):
        bb = ezdxf.bbox.extents(lay, fast=True)
        if bb.has_data:
            m = 0.02 * max(bb.size.x, bb.size.y)
            from ezdxf.math import Vec2
            lo, hi = Vec2(bb.extmin.x - m, bb.extmin.y - m), Vec2(bb.extmax.x + m, bb.extmax.y + m)
            w_mm, h_mm = (hi.x - lo.x) * unit_mm, (hi.y - lo.y) * unit_mm
            paper_source = "content extents of the layout plus 2 % (no usable paper size)"
    vps = [v for v in lay.viewports() if v.dxf.get("id", 0) != 1 and v.dxf.get("status", 1) != 0]
    texts = [(t, "paper") for t in _dxf_texts(lay)]
    msp_texts = _dxf_texts(doc.modelspace()) if vps else []
    for vp in vps:
        tf = _VpTransform(vp)
        clip, _ = _clip_polygon(doc, vp)
        from shapely.geometry import Point, Polygon
        cp = Polygon(clip)
        for s, box, h, rot, layer in msp_texts:
            pts = tf(np.array([[box[0], box[1]], [box[2], box[3]]]))
            c = pts.mean(0)
            if cp.contains(Point(c)) and layer not in set(vp.frozen_layers or ()):
                texts.append(((s, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()), h * tf.scale,
                               rot + tf.twist, layer), f"model space through viewport {vp.dxf.handle}"))
    text_mm = [t[2] * unit_mm for t, _ in texts if t[2] > 0]
    area_in2 = (w_mm / MM_PER_INCH) * (h_mm / MM_PER_INCH)
    if dpi:
        dpi_, why = float(dpi), "set by the caller"
    else:
        dpi_, why = _render_dpi(text_mm, cfg, area_in2)
    img = _render_dxf(doc, lay, lo, hi, w_mm, h_mm, dpi_, monochrome)
    pxmm = img.shape[1] / w_mm                           # actual pixels per paper millimetre of the render
    s = pxmm * unit_mm
    to_px = np.array([[s, 0, -lo.x * s], [0, -s, hi.y * s], [0, 0, 1]], float)       # paper-space units -> raster px
    cls = f"{origin_class} layout"
    pkg = SheetPackage(f"{path.stem}-{_slug(lay.name)}", str(path), lay.name, cls, img, dpi=pxmm * MM_PER_INCH,
                       dpi_source="render", dpi_alternatives=[(pxmm * MM_PER_INCH, "render", f"{origin_class} layout rendered")],
                       paper_mm=(w_mm, h_mm), paper_source=paper_source)
    pkg.warnings = [dict(w) for w in warnings]
    for (st, box, h, rot, layer), where in texts:
        b = _apply(to_px, np.array([[box[0], box[1]], [box[2], box[3]]]))
        r = rot % 180
        pkg.text.append({"text": st, "box": (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max())),
                         "height": float(h * s), "rotation": round(rot, 2), "angle": 90 if 45 <= r < 135 else 0,
                         "font": None, "colour": None, "layer": layer, "conf": 1.0, "source": "dxf", "outlined": False,
                         "invisible": False, "space": where, "size_mm": h * unit_mm})
    for vp in vps:
        sc = vp.get_scale()
        clip, kind = _clip_polygon(doc, vp)
        tf = _VpTransform(vp)
        n = None if sc <= 0 else mm_unit / (sc * unit_mm)
        m3 = np.eye(3)
        mm = tf.m
        m3[:2, :2] = [[mm[0, 0], mm[1, 0]], [mm[0, 1], mm[1, 1]]]     # ezdxf Matrix44 is row-major (row vectors)
        m3[:2, 2] = [mm[3, 0], mm[3, 1]]
        model_to_px = to_px @ m3
        pkg.viewports.append({"id": str(vp.dxf.handle), "clip_px": [[round(float(x), 2), round(float(y), 2)] for x, y in _apply(to_px, clip)],
                              "clip_kind": kind, "scale": None if n is None else round(n, 4),
                              "paper_per_model": sc, "model_units": insunits, "model_mm_per_unit": mm_unit,
                              "twist_deg": tf.twist, "frozen_layers": list(vp.frozen_layers or ()),
                              "px_per_m": None if sc <= 0 else round(pxmm * 1000.0 / n, 4),
                              "model_to_px": model_to_px.tolist()})
        if n is not None and abs(n - round(n)) > 0.01 * n:
            pkg.warn("viewport scale", "low", f"viewport {vp.dxf.handle}: custom scale 1:{n:.2f}")
    pkg.transforms = {"source_units": f"{origin_class} paper-space units ({'inch' if unit_mm != 1 else 'mm'}), y up",
                      "source_to_px": to_px.tolist(), "px_to_paper_mm": _paper_transform(pxmm * MM_PER_INCH)}
    pkg.provenance = {"loader": "fpx.inputs.load_dxf", "renderer": "ezdxf drawing add-on, PyMuPDF backend",
                      "plot_style": "monochrome (black on white)" if monochrome else "ACI colours on white",
                      "layout": lay.name, "dpi_reason": why, "insunits": insunits,
                      "plot_rotation": int(lay.dxf.get("plot_rotation", 0) or 0)}
    if pkg.provenance["plot_rotation"]:
        pkg.warn("plot rotation", "low", "the layout's plot rotation is not applied: rendered as laid out")
    if not vps:
        pkg.warn("viewports", "low", "layout without viewports: the drawing is in paper space")
    return pkg


def _dxf_model(doc, path, cfg, dpi, monochrome, mm_unit, insunits, warnings, origin_class):
    import ezdxf.bbox
    from ezdxf.math import Vec2
    msp = doc.modelspace()
    bb = ezdxf.bbox.extents(msp, fast=True)
    if not bb.has_data:
        raise ValueError(f"{path}: empty model space and no paper-space layout with content")
    m = 0.02 * max(bb.size.x, bb.size.y)
    lo, hi = Vec2(bb.extmin.x - m, bb.extmin.y - m), Vec2(bb.extmax.x + m, bb.extmax.y + m)
    w_m, h_m = (hi.x - lo.x) * mm_unit / 1000, (hi.y - lo.y) * mm_unit / 1000
    ppm = float(dpi) / MM_PER_INCH * 1000 / 100 if dpi else cfg.dxf_model_px_per_m      # dpi given: as if 1:100
    cap = math.sqrt(cfg.input_max_mpx * 1e6 / max(w_m * h_m, 1e-9))
    why = f"{ppm:g} px per model metre"
    if ppm > cap:
        ppm, why = cap, f"{cap:.1f} px per model metre (capped at {cfg.input_max_mpx:g} megapixels)"
    # the renderer works in paper millimetres: a virtual page at 10 px/mm carries the model extents
    vdpi = 254.0
    page_w, page_h = w_m * ppm / 10, h_m * ppm / 10
    img = _render_dxf(doc, msp, lo, hi, page_w, page_h, vdpi, monochrome)
    s = img.shape[1] / (hi.x - lo.x)                     # raster px per model unit
    to_px = np.array([[s, 0, -lo.x * s], [0, -s, hi.y * s], [0, 0, 1]], float)
    cls = f"{origin_class} model space"
    pkg = SheetPackage(f"{path.stem}-model", str(path), "Model", cls, img, dpi=None, dpi_source="unknown",
                       paper_mm=None, paper_source="none: model space has no paper")
    pkg.warnings = [dict(w) for w in warnings]
    pkg.model_scale = {"mm_per_unit": mm_unit, "insunits": insunits, "px_per_m": round(s * 1000.0 / mm_unit, 6),
                       "note": "exact: the raster is drawn from model coordinates"}
    for st, box, h, rot, layer in _dxf_texts(msp):
        b = _apply(to_px, np.array([[box[0], box[1]], [box[2], box[3]]]))
        r = rot % 180
        pkg.text.append({"text": st, "box": (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max())),
                         "height": float(h * s), "rotation": round(rot, 2), "angle": 90 if 45 <= r < 135 else 0,
                         "font": None, "colour": None, "layer": layer, "conf": 1.0, "source": "dxf", "outlined": False,
                         "invisible": False, "space": "model"})
    pkg.transforms = {"source_units": f"model units (INSUNITS {insunits}, {mm_unit:g} mm), y up",
                      "source_to_px": to_px.tolist(), "px_to_paper_mm": None}
    pkg.provenance = {"loader": "fpx.inputs.load_dxf", "renderer": "ezdxf drawing add-on, PyMuPDF backend",
                      "plot_style": "monochrome (black on white)" if monochrome else "ACI colours on white",
                      "layout": "Model", "resolution": why, "insunits": insunits}
    pkg.warn("model space", "low", "no paper-space layout with content: model space rendered; expect several plans side by side")
    return pkg


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "layout"


class DwgConverterMissing(RuntimeError):
    pass


class DwgConverter:
    """DWG -> DXF, swappable (docs/pipeline.md 0a and 6). Subclasses set name, licence and implement available() and
    convert(src, out_dir) -> DXF path. Converters run on untrusted uploads: run them sandboxed, with a timeout."""
    name = "abstract"
    licence = ""
    timeout = 300

    def available(self):
        return False

    def convert(self, src, out_dir):
        raise NotImplementedError

    def _run(self, cmd, out):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if not Path(out).exists():
            raise RuntimeError(f"{self.name} failed ({r.returncode}): {(r.stderr or r.stdout)[-500:]}")
        return Path(out)


class LibreDwgConverter(DwgConverter):
    """GNU LibreDWG dwg2dxf (GPL-3.0; reads all versions with gaps; CVE history: run sandboxed). Executable: the
    FPX_DWG2DXF environment variable or dwg2dxf on PATH."""
    name, licence = "libredwg", "GPL-3.0"

    def exe(self):
        return os.environ.get("FPX_DWG2DXF") or shutil.which("dwg2dxf")

    def available(self):
        exe = self.exe()
        return bool(exe) and (Path(exe).exists() or shutil.which(exe) is not None)

    def convert(self, src, out_dir):
        out = Path(out_dir) / (Path(src).stem + ".dxf")
        return self._run([self.exe(), "-y", "-o", str(out), str(src)], out)


class QcadConverter(DwgConverter):
    """QCAD Professional command-line tool dwg2dwg (proprietary; the server licence allows use as a web service).
    Executable: FPX_QCAD_DWG2DWG or dwg2dwg on PATH."""
    name, licence = "qcad", "proprietary (QCAD Professional server licence)"

    def exe(self):
        return os.environ.get("FPX_QCAD_DWG2DWG") or shutil.which("dwg2dwg") or shutil.which("dwg2dwg.bat")

    def available(self):
        return bool(self.exe())

    def convert(self, src, out_dir):
        out = Path(out_dir) / (Path(src).stem + ".dxf")
        return self._run([self.exe(), "-f", "-o", str(out), str(src)], out)


class OdaConverter(DwgConverter):
    """ODA File Converter through ezdxf.addons.odafc (proprietary freeware, 'for non-commercial applications only' for
    non-members: needs ODA's written confirmation for internal federal use). Executable: FPX_ODAFC_EXE or ezdxf's
    odafc-addon configuration."""
    name, licence = "oda", "proprietary freeware (ODA; non-commercial use only for non-members)"

    def _setup(self):
        exe = os.environ.get("FPX_ODAFC_EXE")
        if exe:
            import ezdxf
            ezdxf.options.set("odafc-addon", "win_exec_path" if os.name == "nt" else "unix_exec_path", exe)

    def available(self):
        try:
            from ezdxf.addons import odafc
            self._setup()
            return bool(odafc.is_installed())
        except Exception:
            return False

    def convert(self, src, out_dir):
        from ezdxf.addons import odafc
        self._setup()
        out = Path(out_dir) / (Path(src).stem + ".dxf")
        odafc.convert(str(src), str(out), version="R2018", audit=True, replace=True)
        return out


DWG_CONVERTERS = {"libredwg": LibreDwgConverter, "qcad": QcadConverter, "oda": OdaConverter}


def dwg_converter(name=None):
    """The converter named by `name` or FPX_DWG_CONVERTER (libredwg, qcad, oda), else the first one installed.
    Raises DwgConverterMissing naming the options when none is available."""
    name = name or os.environ.get("FPX_DWG_CONVERTER")
    order = [name] if name else list(DWG_CONVERTERS)
    for n in order:
        if n not in DWG_CONVERTERS:
            raise DwgConverterMissing(f"unknown DWG converter {n!r}: one of {sorted(DWG_CONVERTERS)}")
        c = DWG_CONVERTERS[n]()
        if c.available():
            return c
    opts = "; ".join(f"{c.name} ({c.licence})" for c in (k() for k in DWG_CONVERTERS.values()))
    raise DwgConverterMissing(
        ("DWG converter " + repr(name) + " is not installed. " if name else "No DWG converter is installed. ")
        + f"DWG files are read through a swappable DWG -> DXF converter: {opts}. Install one and select it with "
        "load(path, converter=...) or FPX_DWG_CONVERTER (executables: FPX_DWG2DXF, FPX_QCAD_DWG2DWG, FPX_ODAFC_EXE); "
        "the choice and its licence are open (docs/pipeline.md section 9). Meanwhile export the drawing to DXF or PDF.")


def load_dwg(path, cfg=DEFAULT, converter=None, **opts):
    conv = converter if isinstance(converter, DwgConverter) else dwg_converter(converter)
    with tempfile.TemporaryDirectory(prefix="fpx-dwg-") as tmp:
        dxf = conv.convert(path, tmp)
        pkgs = load_dxf(dxf, cfg, origin_class="DWG", **opts)
    for p in pkgs:
        p.source = str(path)
        p.provenance["converter"] = f"{conv.name} ({conv.licence})"
    return pkgs


# ---------------------------------------------------------------------------------------------------------------------

def load(path, cfg=DEFAULT, **opts):
    """Any upload -> [SheetPackage], one per page or layout.

    Options: pages (0-based indices; PDF and TIFF), dpi (force the render resolution; PDF and DXF), layouts (DXF
    layout names), model_space (DXF: render model space even if layouts exist), monochrome (DXF plot style, default
    True), with_paths (PDF path channel, default True), backend (PDF backend name), converter (DWG)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, cfg, **opts)
    if ext == ".dxf":
        return load_dxf(path, cfg, **opts)
    if ext == ".dwg":
        return load_dwg(path, cfg, **opts)
    if ext in IMAGE_EXT:
        return load_image(path, cfg, **opts)
    raise ValueError(f"{path}: unsupported input type {ext!r} (raster image, PDF, DXF or DWG)")
