"""Scale cues and their consensus: a pre-pass on the native sheet before resampling (stage 1) and stage 9.

The segmenter works at one resolution (Config.px_per_m), so every raster sheet is resampled from its scale first.
Each cue returns candidates in native image pixels per plan metre, each with a weight (how much the cue is trusted),
a tolerance (how precise it is, relative, in length) and its evidence:

    scale_note          "1:50", "M 1:100", "Mst. 1:200", "Massstab 1:50", "échelle 1/100", "scala 1:50", with the dpi
    dimension_strings   numbers on dimension lines ("4.50", "450", "1 425", "2.38⁵") over the distance between ticks
    scale_bar           labels 0, 5, 10 ... along a bar: pixels per labelled metre
    door_widths         the segmenter run at candidate scales; detected doors measure Config.scale_door_width
    stamp_areas         after a first pass: stamp areas against room polygon areas (computed in fpx.qa)

consensus() lets every cue vote once, picks the group of agreeing candidates with the largest weight, averages it
(precision-weighted), snaps it to a standard or noted scale when the dpi is known, and reports the agreeing and the
disagreeing cues, a confidence and QA flags. Disagreement is always reported, never resolved silently.

prepass() runs the text cues (and the door search if a model is given) on the native image. Its OCR works on the
native image resampled by the typical character height: characters under 12 px are enlarged (small hand lettering is
misread), characters over 21 px are reduced to 16 px (digits read as well, 2-5x faster), the longer side is capped
at 7000 px. On CPU this costs 5-25 s for an A4 sheet at 300 dpi and 1-1.5 min for an A1 scan at 400 dpi; the text
cues themselves take under 0.1 s (2 s on a 14k px sheet) and the door search about 15 s.
"""
import math
import re
import time
from dataclasses import replace

import cv2
import numpy as np

from .config import DEFAULT

INCH = 0.0254
STANDARD_SCALES = (20, 50, 100, 200, 500)            # Swiss plan scales: snapping targets when the dpi is known
COMMON_SCALES = (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000)   # a bare "1:N" inside a text counts
SCAN_DPI = (72, 96, 100, 150, 200, 240, 300, 400, 600, 1200)
PAPER_MM = {"A0": (841, 1189), "A1": (594, 841), "A2": (420, 594), "A3": (297, 420), "A4": (210, 297)}
# How far a dpi is trusted: a PDF render or a scan header is exact; a standard paper size at a standard dpi is a
# good guess; image-file metadata of a downloaded rendition is often a default (72, 96, 300) unrelated to the scan.
DPI_TRUST = {"render": 1.0, "scan": 1.0, "paper": 0.6, "metadata": 0.5}
SCREEN_DPI = (72, 96)                                # metadata defaults of screen renditions: trusted even less
# Paper-size matches are ambiguous (A4 at 300 dpi = A2 at 150 dpi); archive and office scans are mostly 300 dpi.
DPI_PRIOR = {300: 1.0, 200: 0.8, 400: 0.8, 150: 0.7, 600: 0.7}
PRECISE = 0.04                                       # cues with a tolerance up to this are "precise"


# ---------- candidates ----------

def candidate(cue, px_per_m, weight, tol, evidence, **extra):
    """One scale hypothesis of one cue: native pixels per plan metre, weight in [0, 1], relative tolerance."""
    return {"cue": cue, "px_per_m": round(float(px_per_m), 3), "weight": round(float(weight), 3),
            "tolerance": float(tol), "evidence": evidence, **extra}


def is_precise(c):
    """A candidate that can calibrate on its own: a tight tolerance, unless the cue says otherwise."""
    return bool(c.get("precise", c["tolerance"] <= PRECISE))


def ratio_text(a, b):
    return f"{a / b:.3f}x"


# ---------- image helpers ----------

def grey(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def binarise(gray):
    """Ink mask: Otsu threshold, clamped so that faint scans and dark paper both keep their lines."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray < min(max(t, 100), 200)


def char_height(ink):
    """Typical character height in pixels: the mode of the heights of compact ink components (letters and digits are
    the most frequent compact components on a plan). None if there are too few."""
    n, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    w, h, a = st[1:, 2].astype(float), st[1:, 3].astype(float), st[1:, 4].astype(float)
    ok = (h >= 6) & (h <= 400) & (w >= 0.15 * h) & (w <= 1.1 * h) & (a >= 0.12 * w * h) & (a <= 0.8 * w * h)
    if ok.sum() < 30:
        return None
    edges = np.geomspace(6, 400, 61)
    hist = np.convolve(np.histogram(h[ok], edges)[0], [1, 2, 1], mode="same")
    k = int(np.argmax(hist))
    return float(np.sqrt(edges[k] * edges[k + 1]))


def infer_dpi(shape, meta_dpi=None):
    """Possible dpi of a sheet image without a known resolution: every standard paper size at a standard dpi that
    fits the pixel size within 1 % (ambiguous by nature: A4 at 300 dpi has the pixels of A2 at 150 dpi), and the file
    metadata. Returns [(dpi, source, note)]."""
    hp, wp = sorted(shape[:2])
    out = []
    for name, (a, b) in PAPER_MM.items():
        for dpi in SCAN_DPI:
            if abs(hp / (a / 25.4 * dpi) - 1) < 0.01 and abs(wp / (b / 25.4 * dpi) - 1) < 0.01:
                out.append((float(dpi), "paper", f"{name} at {dpi} dpi"))
    if meta_dpi and not any(abs(d - meta_dpi) < 0.5 for d, _, _ in out):
        out.append((float(meta_dpi), "metadata", f"image metadata {meta_dpi:g} dpi (unverified)"))
    return out


# ---------- text: pre-pass OCR ----------

def read_text(img, engine, cfg=DEFAULT, factor=None, dpi=None):
    """OCR for the scale cues on the native image, resampled so that characters are about cfg.scale_ocr_char_px high
    (never above cfg.scale_ocr_max_side). With a known dpi the character height is at least 1.2 mm on paper: on
    speckled scans the most frequent compact components are specks, not letters. Boxes and heights are returned in
    native pixels."""
    from .text import ocr
    gray = grey(img)
    t0 = time.time()
    lo, paper = np.percentile(gray[::7, ::7], (0.5, 50))   # faint scans on tinted paper: stretch so ink is dark
    if paper - lo > 20 and (lo > 40 or paper < 235):
        gray = np.clip((gray.astype(np.float32) - lo) * (255.0 / (paper - lo)), 0, 255).astype(np.uint8)
    ch = char_height(binarise(gray))
    if dpi:
        ch = max(ch or 0.0, 1.2 / 25.4 * dpi)
    if factor is None:
        factor = 1.0                                     # in between: resampling would only blur the strokes
        if ch is not None and ch < cfg.scale_ocr_small_px:
            factor = min(2.0, cfg.scale_ocr_enlarge_px / ch)
        elif ch is not None and ch > 1.33 * cfg.scale_ocr_char_px:
            factor = max(0.3, cfg.scale_ocr_char_px / ch)
        factor = min(factor, cfg.scale_ocr_max_side / max(gray.shape))
    im = gray if factor == 1.0 else cv2.resize(gray, None, fx=factor, fy=factor,
                                               interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
    items = []
    for t in ocr(im, engine, cfg.scale_ocr_tile, cfg.scale_ocr_overlap):
        if t["conf"] < cfg.ocr_min_conf or not t["text"].strip():
            continue
        t["box"] = tuple(float(v) / factor for v in t["box"])
        t["height"] = float(t["height"]) / factor
        items.append(t)
    info = {"char_px": None if ch is None else round(ch, 1), "factor": round(factor, 3),
            "ocr_shape": list(im.shape[:2]), "items": len(items), "seconds": round(time.time() - t0, 1)}
    return items, info


# ---------- scale notes ----------

NOTE_KW = r"massstab|maßstab|masstab|mass-stab|mstb|mst|[ée]?chelle|éch|ech|escala|esc|scala|scale|sc|m"
NOTE_RE = re.compile(
    rf"(?:(?<![^\W\d_])(?P<kw>{NOTE_KW})\b\.?\s*[:=]?\s*)?(?<![\d.,])(?P<one>[1lI|])\s*(?P<sep>[:;/.])\s*"
    r"(?P<n>\d{1,3}(?:['’ ]\d{3})+|\d{1,5})(?![\d]|[.,]\d)", re.I)
# "1 cm pour 1 m", "1cm = 1m", "1 cm für 2 m": paper centimetres per plan metres
CM_PER_M_RE = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*cm\s*(?:pour|=|für|fur|per|pro|:|par)\s*(\d+(?:[.,]\d+)?)\s*m\b", re.I)
# imperial notes: paper inches per plan feet, '3/16" = 1\'-0"', '1/8 in. 1 Foot', '1" = 20\'', '1 in = 8 ft'
FT, IN = 0.3048, 0.0254
INCH_MARK = r"(?:\"|''|”|″|in\.?|inch(?:es)?)"
FOOT_MARK = r"(?:'|’|′|ft\.?|feet|foot)"
IN_PER_FT_RE = re.compile(
    rf"(?<![\d/])(?P<inch>\d{{1,2}}(?:\s+\d{{1,2}}/\d{{1,2}}|/\d{{1,2}})?(?:\.\d+)?|/\d{{1,2}})\s*{INCH_MARK}\s*(?:=|:|-|to|per|pour)?\s*"
    rf"(?P<feet>\d{{1,3}}(?:\.\d+)?|[lI])\s*{FOOT_MARK}(?:\s*-?\s*0+\s*{INCH_MARK})?", re.I)
# a keyword followed by "150", "1100": the separator of "1:50", "1:100" lost by OCR (hand-lettered colons and dots)
NOTE_NOSEP_RE = re.compile(rf"(?<![^\W\d_])(?:{NOTE_KW})\b\.?\s*[:=]?\s*1(?P<n>20|25|50|100|200|250|500|1000)(?!\d|[.,:]\d)", re.I)


def _mixed_number(s):
    """'3/16', '1 1/2', '/8' (a leading 1 lost by OCR), '0.25' -> float."""
    s = s.strip()
    m = re.fullmatch(r"(?:(\d+)\s+)?(\d*)/(\d+)", s)
    if m:
        return (int(m.group(1) or 0)) + int(m.group(2) or 1) / int(m.group(3))
    return float(s)


def parse_scale_note(text):
    """Scale ratios written on the sheet -> [(N, keyword)]. "M 1:50" -> [(50, True)]; without a keyword a "1:N" counts
    if it is the whole text item, follows a keyword in the same item ("M 1:20 / 1:50") or N is a common scale
    ("Grundriss EG 1:100"). Slashes and dots, and an OCR'd "l" or "I" for the 1, need a keyword ("échelle 1/100").
    "1 cm pour 1 m" -> [(100, True)]; imperial notes give their ratio: '1/8" = 1\'-0"' -> [(96, True)]."""
    out = []
    s = text.strip()
    while (z := re.sub(r"(?<=[\d:/])\s?[oO]|[oO](?=\d)", "0", s)) != s:     # OCR reads zeros as o: "1 : 3oo"
        s = z
    for m in CM_PER_M_RE.finditer(s):
        cm, metres = (float(g.replace(",", ".")) for g in m.groups())
        if cm > 0 and 10 <= metres * 100 / cm <= 10000:
            out.append((int(round(metres * 100 / cm)), True))
    imperial = []
    for m in IN_PER_FT_RE.finditer(s):
        try:
            inch = _mixed_number(m.group("inch"))
            feet = 1.0 if m.group("feet") in ("l", "I") else float(m.group("feet"))
        except (ValueError, ZeroDivisionError):
            continue
        if inch > 0 and feet > 0 and 1 <= feet * 12 / inch <= 10000:
            out.append((int(round(feet * 12 / inch)), True))
            imperial.append(m.span())
    any_kw = False
    matched = False
    for m in NOTE_RE.finditer(s):
        if any(m.start() < b and a < m.end() for a, b in imperial):
            continue                                     # the "1/8" of an imperial note is not 1:8
        matched = True
        kw = m.group("kw") is not None and not (m.group("kw").lower() == "m" and m.group("sep") not in ":;")
        any_kw |= kw
        n = int(re.sub(r"\D", "", m.group("n")))
        if not 10 <= n <= 10000:
            continue
        whole = re.sub(r"\s", "", s) == re.sub(r"\s", "", m.group(0))
        if not kw and not (m.group("sep") in ":;" and m.group("one") == "1"
                           and (whole or any_kw or n in COMMON_SCALES)):
            continue                                     # "Grundriss EG 1:100" counts; "Zimmer 1:7" does not
        out.append((n, kw))
    if not matched and not imperial:
        for m in NOTE_NOSEP_RE.finditer(s):              # "Masstab 150": the colon lost, a common scale after the 1
            out.append((int(m.group("n")), True))
    return sorted(set(out), key=out.index)


def note_cue(texts, dpis=(), cfg=DEFAULT):
    """Scale notes; with a dpi each gives px per m = dpi / 0.0254 / N. dpis: [(dpi, source, note)]; several
    alternatives (paper sizes) share the weight. Without any dpi the notes only feed snapping and the report."""
    found = []
    for t in texts:
        for n, kw in parse_scale_note(t["text"]):
            found.append({"text": t["text"], "scale": n, "keyword": kw, "source": t.get("source", "ocr")})
    cands = []
    prior = {d: DPI_PRIOR.get(int(d), 0.5) for d, s, _ in dpis if s == "paper"}
    for n in sorted({f["scale"] for f in found}):
        fs = [f for f in found if f["scale"] == n]
        kw = any(f["keyword"] for f in fs)
        for dpi, src, note in dpis:
            share = prior[dpi] / sum(prior.values()) if src == "paper" else 1.0
            trust = DPI_TRUST.get(src, 0.3) * (0.3 if src == "metadata" and round(dpi) in SCREEN_DPI else 1.0)
            w = 0.8 * trust * (1.0 if kw else 0.7) * share
            ev = f"1:{n} read {len(fs)}x ({fs[0]['text']!r}) at {dpi:g} dpi ({note})"
            cands.append(candidate("scale_note", dpi / INCH / n, w, cfg.scale_tol_note, ev, scale=n, dpi=dpi,
                                   dpi_source=src, precise=src != "metadata"))   # file metadata may be a default
    return {"candidates": cands, "found": found[:20]}


# ---------- dimension strings ----------

SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
# feet and inches: 20', 6'-6", 6'6", 14'.0", 30'-6 1/2", 8", 20 ft, 6 ft 6 in; OCR'd hand lettering also as 30.6",
# 48!6, 14:0 (the foot mark read as a dot, a colon or an exclamation mark)
FT_IN_RE = re.compile(
    rf"(?:(?P<ft>\d{{1,3}})\s*{FOOT_MARK}\s*[-.\s]?\s*)?(?:(?P<inch>\d{{1,2}})(?:\s+(?P<num>\d)/(?P<den>\d{{1,2}}))?\s*{INCH_MARK})?", re.I)
FT_IN_OCR_RE = re.compile(r"(?P<ft>\d{1,3})\s*[.:!'′]+\s*(?P<inch>\d{1,2})\s*[\"”″]?")


def parse_feet_inches(text, lenient=False):
    """A length in feet and inches -> metres, or None. Needs a foot or inch mark; "6'-6\"" -> 1.981. lenient: the
    foot mark may be a dot, colon or exclamation mark as OCR reads hand lettering ("30.6\"", "48!6", "14:0")."""
    s = text.strip().replace("’", "'").replace("′", "'").replace("”", '"').replace("″", '"').replace("''", '"')
    m = FT_IN_RE.fullmatch(s)
    if m and (m.group("ft") or m.group("inch")) and re.search(rf"{FOOT_MARK}|{INCH_MARK}", s, re.I):
        if m.group("inch") and not m.group("ft") and not re.search(INCH_MARK, s, re.I):
            return None
        inches = int(m.group("inch") or 0)
        if m.group("num"):
            inches += int(m.group("num")) / int(m.group("den"))
        if inches >= 12 and m.group("ft"):
            return None
        return int(m.group("ft") or 0) * FT + inches * IN
    if lenient and (m := FT_IN_OCR_RE.fullmatch(s)) and int(m.group("inch")) < 12:
        return int(m.group("ft")) * FT + int(m.group("inch")) * IN
    return None


def parse_dimension(text):
    """A number written on a dimension line -> [(metres, unit)] hypotheses; [] if it is not a plain length.

    '4.50', '4,5' -> metres; '2.38⁵' or '2.38 5' -> 2.385 m (Swiss half centimetres in superscript); '450', '1 425',
    "1'425" -> centimetres or millimetres (the sheet decides); '12⁵' -> 12.5 cm; 'd.d' and 'd.ddd' are ambiguous as
    well; explicit 'm', 'cm', 'mm' are taken as written; feet and inches ("20'", "6'-6\"", "6 ft 6 in") are taken as
    written (unit "ft"). Areas (m²), levels (+2.85, ±0.00) and words are rejected."""
    s = text.strip().replace("’", "'").replace("‘", "'").replace("´", "'")
    ft = parse_feet_inches(s)
    if ft is not None:
        return [(round(ft, 4), "ft")] if 0.05 <= ft <= 300 else []
    if re.search(r"m\s*[2²]|[+±-]|[A-Za-z]{2,}(?<!mm)(?<!cm)", s):
        return []
    sup = None
    m = re.fullmatch(r"(.*\d)\s*([⁰¹²³⁴⁵⁶⁷⁸⁹])", s)
    if m:
        s, sup = m.group(1), int(m.group(2).translate(SUPER))
    unit = None
    m = re.fullmatch(r"(.*?\d)\s*(mm|cm|m)\.?", s, re.I)
    if m:
        s, unit = m.group(1).strip(), m.group(2).lower()
    hyp = []
    if (m := re.fullmatch(r"(\d{1,3})[.,](\d{2})(?:\s([05]))?", s)):          # 4.50, 2.38 5
        half = sup if sup is not None else int(m.group(3)) if m.group(3) else 0
        hyp = [(int(m.group(1)) + int(m.group(2)) / 100 + half / 1000, "m")]
    elif (m := re.fullmatch(r"(\d{1,3})[.,](\d{3})", s)) and sup is None:      # 2.385 m, or 1.425 = 1425
        v = int(m.group(1) + m.group(2))
        hyp = [(v / 1000, "m"), (v / 100, "cm")]
    elif (m := re.fullmatch(r"(\d{1,3})[.,](\d)", s)) and sup is None:         # 4.5 m, or 12.5 cm
        v = float(f"{m.group(1)}.{m.group(2)}")
        hyp = [(v, "m"), (v / 100, "cm")]
    elif (m := re.fullmatch(r"\d{1,3}(?:['  ]\d{3})+|\d{1,5}", s)):            # 450, 1 425, 1'425
        v = int(re.sub(r"\D", "", s))
        hyp = [((v + sup / 10) / 100, "cm")] if sup is not None else [(v / 100, "cm"), (v / 1000, "mm")]
    if unit and hyp:
        v = {"m": 1.0, "cm": 0.01, "mm": 0.001}[unit] * float(re.sub(r"[^\d.]", "", s.replace(",", ".")) or 0)
        hyp = [(v, unit)]
    return [(round(v, 4), u) for v, u in hyp if 0.05 <= v <= 300]


def _runs(mask):
    """Start and end (exclusive) of the True runs of a 1-D boolean array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def measure_span(ink, box, ink_ticks=None, why=None):
    """Pixel length of the dimension that a horizontal number labels (transpose the ink and the box for vertical text).

    The dimension line is a long ink row just above or below the text (it may overlap the generous OCR box), or
    through it, the text sitting in a gap of the line. Ticks are strokes that cross the line on the side away from
    the text: extension lines, slashes, dots or arrowheads (ink_ticks: the ink without other text, so that a number
    on the other side of the line is not taken for ticks). Among the nearest ticks left and right of the text, the
    pair that centres the text best wins (CAD centres dimension text exactly; this skips walls and axes that cross
    the line); a loosely centred pair is accepted as "loose". Returns a dict, or None if no line, no tick on either
    side, a span not wider than the text, or text far off the span centre (why: a list that receives the reason)."""
    no = lambda reason: (why.append(reason) if why is not None else None)
    H, W = ink.shape
    ink_ticks = ink if ink_ticks is None else ink_ticks
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    h, w = max(y1 - y0, 4), x1 - x0
    if w < 3 or x0 < 0 or x1 > W or y0 < 0 or y1 > H:
        return no("box outside the sheet")
    ya, yb = max(0, y0 - int(1.5 * h)), min(H, y1 + int(1.5 * h) + 1)
    band = ink[ya:yb]
    bd = band.copy()
    bd[1:] |= band[:-1]
    bd[:-1] |= band[1:]                                  # tolerate one pixel of tilt or anti-aliasing
    side = max(4, h)
    mid = bd[:, x0:x1].mean(1)
    left = bd[:, max(0, x0 - side):max(0, x0 - 1)]
    right = bd[:, min(W, x1 + 1):min(W, x1 + side)]
    if left.shape[1] < 2 or right.shape[1] < 2:
        return no("text at the sheet edge")
    left, right = left.mean(1), right.mean(1)
    rows = np.arange(ya, yb)
    inside = (rows > y0 + 0.2 * h) & (rows < y1 - 0.2 * h)
    outside = (rows <= y0 + 0.35 * h) | (rows >= y1 - 0.35 * h)         # OCR boxes are padded: the line may be inside
    along = (mid >= 0.9) & (left >= 0.8) & (right >= 0.8) & outside       # line above or below the text
    through = (mid < 0.5) & (left >= 0.9) & (right >= 0.9) & inside        # text in a gap of the line
    cand = np.flatnonzero(along | through)
    if not len(cand):
        return no("no line along the text")
    dist = np.where(rows[cand] < y0, y0 - rows[cand], np.where(rows[cand] > y1, rows[cand] - y1, 0))
    r = int(cand[np.argmin(dist)])
    mode = "through" if through[r] else "below" if rows[r] > (y0 + y1) / 2 else "above"
    ok = along | through
    ra = rb = r
    while ra > 0 and ok[ra - 1]:
        ra -= 1
    while rb < len(ok) - 1 and ok[rb + 1]:
        rb += 1
    ra, rb = ra + ya, rb + ya                             # line rows (dilated by one) in sheet pixels
    if rb - ra > 0.6 * h:
        return no("a solid band, not a line")
    line = _follow(ink, (ra + rb) / 2, (x0, x1), max(2.0, (rb - ra) / 2 + 1), step=max(4, h // 2),
                   gap=max(2 * max(4, h // 2) + 2, int(0.6 * h)))  # from the text edges outward: glyphs pull a follower
    if line is None:
        return no("line not followed")
    xl, xr, fit, half = line
    if xr - xl < 1.2 * w:
        return no(f"line {xr - xl} px, not longer than the text")
    # ticks: strokes crossing the line on both sides (slashes, dots, arrowheads, extension lines); text next to the
    # line touches one side only. Fallback: one side, away from the text, deeper.
    k = max(1, int(round(0.2 * h)))                      # horizontal slack for slashes and slightly tilted strokes
    m = max(2, int(round(0.06 * h)))                     # a tick reaches at least this far beyond the line on a side
    lo, hi = max(0, int(xl) - 3 * k - 2), min(W, int(xr) + 3 * k + 3)
    xs = np.arange(lo, hi)
    centre = fit[0] + fit[1] * xs
    core = (xs > x0 + 0.15 * w) & (xs < x1 - 0.15 * w)  # the text itself never carries a tick

    def reach(sign, depth):
        """Per column: how many rows of ink follow the line outward on one side (after horizontal dilation by k)."""
        j = np.arange(depth)[:, None]
        rows = np.round(centre[None, :] + sign * (half + j)).astype(int)
        valid = (rows >= 0) & (rows < H)
        b = np.where(valid, ink_ticks[np.clip(rows, 0, H - 1), xs[None, :]], False).astype(np.uint8)
        b = cv2.dilate(b, np.ones((1, 2 * k + 1), np.uint8)) > 0
        e = np.argmin(np.vstack([b, np.zeros((1, b.shape[1]), bool)]), axis=0)
        e[core] = 0
        return e

    up, dn = reach(-1, 2 * m), reach(1, 2 * m)
    ticks = _tick_positions(np.minimum(up, dn) >= m, lo)
    pairs = _tick_pairs(ticks, x0, x1, h, xl, xr, k)
    sided = "both"
    if pairs is None:
        depth = max(m + 1, int(round(0.6 * h)))
        far = reach(1, depth) if mode in ("below", "through") else reach(-1, depth)
        pairs = _tick_pairs(_tick_positions(far >= max(m, int(0.4 * h)), lo), x0, x1, h, xl, xr, k)
        sided = "far"
    if pairs is None:
        xc = (x0 + x1) / 2
        return no(f"no centred tick pair (line {xl - xc:.0f}..{xr - xc:.0f} px from the text centre, text {w} px; "
                  f"ticks at {[round(t - xc) for t in ticks][:12]}; crossing reach up to {int(np.minimum(up, dn).max())} px, "
                  f"needed {m}; line half width {half:.1f})")
    tight, a, b = pairs
    return {"span": float(b - a), "left": float(a), "right": float(b), "row": float(fit[0] + fit[1] * (a + b) / 2),
            "mode": mode, "h": float(h), "tight": bool(tight), "ticks": sided, "tilt": float(fit[1])}


def _tick_positions(is_tick, lo):
    starts, ends = _runs(is_tick)
    return [lo + (a + b - 1) / 2 for a, b in zip(starts, ends)]


def _tick_pairs(ticks, x0, x1, h, xl, xr, k):
    """Among the four nearest ticks on each side of the text, the pair that centres the text best: tightly centred
    (as CAD does) first, then the fewest ticks skipped. Returns (tight, left, right) or None."""
    w, xc = x1 - x0, (x0 + x1) / 2
    tl = sorted((x for x in ticks if xl - 3 * k <= x < xc - 0.3 * w), reverse=True)[:4]    # nearest first
    tr = sorted(x for x in ticks if xc + 0.3 * w < x <= xr + 3 * k)[:4]
    best = None
    for i, a in enumerate(tl):
        for j, b in enumerate(tr):
            span, err = b - a, abs(xc - (a + b) / 2)
            if span < 1.3 * w or err > 0.2 * span + 0.5 * h:
                continue
            key = (err <= 0.04 * span + 0.3 * h, -(i + j), -err)
            if best is None or key > best[0]:
                best = (key, a, b)
    return None if best is None else (best[0][0], best[1], best[2])


def _follow(ink, r, start, half, step, gap):
    """Follow a roughly horizontal line from row r on both sides of start = (x_left, x_right) in steps, re-centring on
    its ink, bridging gaps up to gap pixels. Returns (x_left_end, x_right_end, (a, b) with row = a + b x, half
    thickness) or None."""
    H, W = ink.shape
    pts = []
    ends = []
    for direction, x in ((-1, start[0]), (1, start[1])):
        rr, missed, last = r, 0, x
        while 0 <= x < W:
            xa, xb = (x - step, x) if direction < 0 else (x, x + step)
            xa, xb = max(0, xa), min(W, xb)
            if xb <= xa:
                break
            ya, yb = max(0, int(rr - half - 2)), min(H, int(rr + half + 3))
            prof = ink[ya:yb, xa:xb].mean(1)
            runs = list(zip(*_runs(prof >= 0.35)))
            centres = [ya + float(np.average(np.arange(s, e), weights=prof[s:e])) for s, e in runs]
            near = [k for k, c in enumerate(centres) if abs(c - rr) <= 1.5]    # the line itself, not text beside it
            if near:
                k = min(near, key=lambda k: abs(centres[k] - rr))
                rr = centres[k]
                pts.append(((xa + xb) / 2, rr, int(runs[k][1] - runs[k][0])))
                last, missed = (xa if direction < 0 else xb), 0
            else:
                missed += step
                if missed > gap:
                    break
            x += direction * step
        ends.append(last)
    if len(pts) < 2:
        return None
    p = np.array(pts)
    b, a = np.polyfit(p[:, 0], p[:, 1], 1) if np.ptp(p[:, 0]) > 0 else (0.0, float(p[0, 1]))
    res = np.abs(p[:, 1] - (a + b * p[:, 0]))
    keep = res <= max(1.5, 2 * np.median(res))           # one refit without stray samples (crossing strokes)
    if keep.sum() >= 2 and np.ptp(p[keep, 0]) > 0:
        b, a = np.polyfit(p[keep, 0], p[keep, 1], 1)
    thick = float(np.median(p[keep, 2])) if keep.any() else 1.0
    return ends[0], ends[1], (float(a), float(b)), max(1.0, thick / 2 + 0.5)


def cluster(hyps, tol, max_clusters=3, min_support=1, weights=None):
    """Robust estimate from several pairs, each with one or more hypotheses (px per m). Returns clusters, best first:
    {"value": median of members, "members": {pair index: value}, "support": summed pair weights (default 1 each),
    "count": number of pairs, "spread": MAD (rel.)}. A cluster is the set of pairs with the largest support that have
    a hypothesis within tol (in log) of a centre; its hypotheses are then removed and the next cluster is searched
    (e.g. the same numbers read as millimetres instead of centimetres)."""
    wt = (lambda j: 1.0) if weights is None else (lambda j: weights[j])
    pts = [(math.log(v), i) for i, hs in enumerate(hyps) for v in hs if v > 0]
    out = []
    while pts and len(out) < max_clusters:
        best = None
        for c, _ in pts:
            mem = {}
            for lv, j in pts:
                if abs(lv - c) <= tol and (j not in mem or abs(lv - c) < abs(mem[j] - c)):
                    mem[j] = lv
            key = (sum(wt(j) for j in mem), -float(np.median(np.abs(np.array(list(mem.values())) - c))))
            if best is None or key > best[0]:
                best = (key, c, mem)
        _, c, mem = best
        for _ in range(2):                                # re-centre on the median and collect again
            c = float(np.median(list(mem.values())))
            mem = {}
            for lv, j in pts:
                if abs(lv - c) <= tol and (j not in mem or abs(lv - c) < abs(mem[j] - c)):
                    mem[j] = lv
        support = sum(wt(j) for j in mem)
        if support < min_support:
            break
        vals = np.array(list(mem.values()))
        out.append({"value": math.exp(float(np.median(vals))), "members": {j: math.exp(v) for j, v in mem.items()},
                    "support": support, "count": len(mem), "spread": float(np.median(np.abs(vals - np.median(vals))))})
        pts = [(lv, j) for lv, j in pts if abs(lv - c) > tol]
    return out


def dimension_cue(texts, ink, cfg=DEFAULT):
    """Numbers on dimension lines paired with the measured distance between their ticks; robust estimate across the
    pairs (largest cluster, median, outliers rejected). Integer-only sheets are ambiguous between cm and mm: the
    cluster whose text height is plausible in plan metres wins; if both are plausible both are returned."""
    numbers = [(t, parse_dimension(t["text"])) for t in texts if not re.fullmatch(r"\s*\d\s*", t["text"])]
    numbers = [(t, h) for t, h in numbers if h]                     # single digits are labels, not dimensions
    res = {"candidates": [], "pairs": 0, "numbers": len(numbers)}
    if not numbers:
        return res
    text_mask = np.zeros(ink.shape, bool)                # all text boxes, shrunk by a quarter of their height
    for t in texts:
        x0, y0, x1, y1 = t["box"]
        s = 0.25 * min(x1 - x0, y1 - y0)
        text_mask[max(0, int(y0 + s)):max(0, int(y1 - s) + 1), max(0, int(x0 + s)):max(0, int(x1 - s) + 1)] = True
    ink_ticks = ink & ~text_mask
    pairs, inkT, ticksT = [], None, None
    for t, hyp in numbers:
        x0, y0, x1, y1 = t["box"]
        if t.get("angle", 0) == 90:
            if inkT is None:
                inkT, ticksT = np.ascontiguousarray(ink.T), np.ascontiguousarray(ink_ticks.T)
            m = measure_span(inkT, (y0, x0, y1, x1), ticksT)
        else:
            m = measure_span(ink, (x0, y0, x1, y1), ink_ticks)
        if m is None:
            continue
        pairs.append({"text": t["text"], "hyp": hyp, "span_px": round(m["span"], 1), "text_px": m["h"],
                      "angle": t.get("angle", 0), "mode": m["mode"], "tight": m["tight"], "source": t.get("source", "ocr")})
    tight = [p for p in pairs if p["tight"]]
    res["pairs"], res["tight_pairs"] = len(pairs), len(tight)
    if len(tight) >= 3:
        pairs = tight                                    # well-centred pairs only, when there are enough of them
    if not pairs:
        return res
    hyps = [[p["span_px"] / v for v, _ in p["hyp"]] for p in pairs]
    # two-digit integers are mostly room numbers or labels next to some line (right in 0-22 % of the benchmark
    # pairs, against 57-89 % for other numbers): they count a quarter
    quality = [0.25 if re.fullmatch(r"\s*\d{2}\s*", p["text"]) else 1.0 for p in pairs]
    cl = cluster(hyps, cfg.scale_tol_dims, min_support=0.25, weights=quality)
    if not cl:
        return res
    top = cl[0]["support"]
    rows = []
    for c in cl:
        if c["support"] < max(0.25, 0.6 * top):
            continue
        th = float(np.median([pairs[j]["text_px"] for j in c["members"]])) / c["value"]
        units = [next(u for v, u in pairs[j]["hyp"] if abs(pairs[j]["span_px"] / v / c["members"][j] - 1) < 1e-6)
                 for j in c["members"]]
        unit = max(set(units), key=units.count)
        plausible = cfg.scale_text_min <= th <= cfg.scale_text_max
        rows.append((c, th, unit, plausible))
    rows.sort(key=lambda r: (not r[3], -r[0]["support"], r[2] != "cm"))  # plausible first, then support, then cm
    n_pairs, total = len(pairs), sum(quality)
    for k, (c, th, unit, plausible) in enumerate(rows):
        n, frac = c["support"], c["support"] / total
        w = min(1.0, 0.25 * n) * math.sqrt(frac) * (1.0 if plausible else 0.3) * (1.0 if k == 0 else 0.6)
        ex = sorted(c["members"], key=lambda j: -pairs[j]["span_px"])[:3]
        ev = (f"{c['count']} of {n_pairs} dimension strings agree (unit {unit}, text {th:.2f} m high"
              f"{'' if plausible else ', implausible'}); e.g. " +
              ", ".join(f"{pairs[j]['text']!r} over {pairs[j]['span_px']:.0f} px" for j in ex))
        # precise as a calibration with at least 3 agreeing pairs that are at least 60 % of all pairs (the
        # calibration-seed rule of Talebi-Kalaleh et al. 2026, for vector PDFs), or with 6 and more: on rasters many
        # OCR'd numbers pair with some line, but mis-pairs do not cluster within a few per cent. Fewer pairs count
        # as a weak cue.
        res["candidates"].append(candidate("dimension_strings", c["value"], w, cfg.scale_tol_dims, ev,
                                           support=round(n, 2), pairs=c["count"], fraction=round(frac, 3), unit=unit,
                                           text_height_m=round(th, 3), spread=round(c["spread"], 4),
                                           precise=bool(plausible and n >= 3 and (frac >= 0.6 or n >= 6))))
    best = rows[0][0] if rows else None
    res["examples"] = [{"text": p["text"], "span_px": p["span_px"], "angle": p["angle"], "mode": p["mode"],
                        "tight": p["tight"], "inlier": bool(best and i in best["members"]),
                        "px_per_m": [round(p["span_px"] / v, 2) for v, _ in p["hyp"]]} for i, p in enumerate(pairs[:40])]
    if len(rows) > 1 and abs(math.log(rows[0][0]["value"] / rows[1][0]["value"])) > 1.5:
        res["ambiguous"] = f"units ambiguous: {rows[0][2]} ({rows[0][0]['value']:.1f} px/m) or {rows[1][2]} ({rows[1][0]['value']:.1f} px/m)"
    return res


# ---------- scale bar ----------

BAR_UNIT_RE = re.compile(r"^(?P<n>[0Oo]|\d{1,3})\s*(?P<unit>m|meter|metres?|meters?|metri|m[èe]tres?|ft|feet|foot|fuss|yds?|yards?)?\.?$", re.I)
BAR_UNIT_M = {"m": 1.0, "meter": 1.0, "metre": 1.0, "metres": 1.0, "meters": 1.0, "metri": 1.0, "mètre": 1.0, "mètres": 1.0,
              "metre": 1.0, "ft": FT, "feet": FT, "foot": FT, "fuss": FT, "yd": 3 * FT, "yds": 3 * FT, "yard": 3 * FT, "yards": 3 * FT}


def _bar_labels(texts):
    """Horizontal number labels of a scale bar: (value, centre x, centre y, height, item, unit metres or None).
    A unit word on a label ("20 m", "20 Meter", "50 ft") names the bar's unit. Several numbers read as one item
    ("2 3 4 5") are split, each at its share of the box."""
    nums = []
    for t in texts:
        if t.get("angle", 0) != 0 or not t.get("text"):
            continue
        s = t["text"].strip()
        x0, y0, x1, y1 = t["box"]
        parts = re.fullmatch(r"(?:[0Oo]|\d{1,3})(?:\s+(?:[0Oo]|\d{1,3}))+", s)
        if parts:
            vals = s.split()
            w = (x1 - x0) / len(vals)
            for k, v in enumerate(vals):
                v = "0" if re.fullmatch(r"[0Oo]", v) else v
                nums.append((int(v), x0 + (k + 0.5) * w, (y0 + y1) / 2, y1 - y0, t, None))
            continue
        m = BAR_UNIT_RE.match(s)
        if not m:
            continue
        v = "0" if re.fullmatch(r"[0Oo]", m.group("n")) else m.group("n")
        unit = BAR_UNIT_M.get(m.group("unit").lower()) if m.group("unit") else None
        cx = (x0 + x1) / 2
        if m.group("unit"):                              # "20 Meter.": the number's centre, not the item's
            cx = x0 + (x1 - x0) * len(m.group("n")) / len(s) / 2
        nums.append((int(v), cx, (y0 + y1) / 2, y1 - y0, t, unit))
    return nums


def _bar_rows(nums):
    """Candidate rows of bar labels: (row sorted by x, anchor item, px per labelled unit, residuals, unit metres).
    Labels on one baseline increasing left to right; with a 0 label the ones left of it count negative ("10 0 10").
    Three distinct labels, or two when one of them carries a unit word ("0 ... 20 m"); a straight-line fit of
    position against value, with an intercept when there is no 0 label (its label unread or absent)."""
    seen = set()
    for v0, zx, zy, zh, zt, zu in sorted(nums, key=lambda n: (n[0] != 0, n[1])):      # 0 labels first as anchors
        # one baseline; OCR boxes of single digits and of labels with a unit word differ in height, so the height
        # ratio is lenient and the reach is measured in the row's typical height
        row = [n for n in nums if abs(n[2] - zy) < 0.5 * max(zh, n[3]) and 0.4 < n[3] / zh < 2.5]
        hm = float(np.median([n[3] for n in row]))
        row = sorted((n for n in row if abs(n[1] - zx) < 80 * hm), key=lambda n: n[1])
        key = tuple(id(n[4]) for n in row) + tuple(round(n[1]) for n in row)
        if key in seen or len(row) < 2:
            continue
        seen.add(key)
        units = {n[5] for n in row if n[5]}
        if len(units) > 1:
            continue
        zero = next((n for n in row if n[0] == 0), None)
        if zero is not None:
            zx0 = zero[1]
            signed = np.array([n[0] if n[1] - zx0 >= -0.5 * zh else -n[0] for n in row], float)
        else:
            signed = np.array([n[0] for n in row], float)
        # the longest run of increasing values (a label of another row, or a misread, is left out)
        best = []
        for i in range(len(row)):
            run = [i]
            for j in range(i + 1, len(row)):
                if signed[j] > signed[run[-1]]:
                    run.append(j)
            if len(run) > len(best):
                best = run
        row = [row[i] for i in best]
        signed = signed[best]
        distinct = len(set(signed.tolist()))
        if distinct < 3 and not (distinct == 2 and units):
            continue
        xs = np.array([n[1] for n in row])
        if zero is not None:
            xs = xs - zero[1]
            b = float((xs * signed).sum() / (signed * signed).sum())
            res = xs - b * signed
        else:
            if distinct < 3:
                continue                                 # two labels without a zero: no fit
            b, x0 = np.polyfit(signed, xs, 1)
            b = float(b)
            res = xs - (x0 + b * signed)
        step = np.min(np.diff(signed)) * abs(b)
        # label centres stand for the divisions to within 15 % of a step, or half a label height (labels read as one
        # item are placed by their share of the box)
        if b <= 0 or step < hm or np.max(np.abs(res)) > max(0.15 * step, 0.6 * hm):
            continue
        yield row, (v0, zx, zy, hm, zt), b, res, (units.pop() if units else 1.0)


BAR_INK = 0.3                                        # a bar's line covers at least this share of its length (engraved and scanned bars break up)


def _bar_ink(band):
    """Whether a line runs along a band of ink rows: some row, thickened to three, has ink on BAR_INK of its length."""
    if band.shape[0] < 3 or band.shape[1] < 2:
        return False
    rows = band[:-2] | band[1:-1] | band[2:]
    return bool((rows.mean(1) > BAR_INK).any())


def scale_bar_cue(texts, ink, cfg=DEFAULT):
    """A row of labels 0, 5, 10 ... (or 10, 0, 10) along a horizontal bar: the label centres stand for the divisions,
    so a straight-line fit of position against value gives pixels per metre. Labels are in metres unless a unit word
    says otherwise ("20 m", "50 ft"); two labels suffice when one carries the unit ("0 ... 20 m")."""
    cands, found = [], []
    H, W = ink.shape
    for row, (v0, zx, zy, zh, zt), b, res, unit_m in _bar_rows(_bar_labels(texts)):
        # a bar or line must run along the labels (within three label heights)
        xa, xb = int(max(0, min(n[1] for n in row))), int(min(W, max(n[1] for n in row)))
        ya, yb = int(max(0, zy - 3 * zh)), int(min(H, zy + 3 * zh))
        if xb - xa < 10 or not _bar_ink(ink[ya:yb, xa:xb]):
            continue
        ev = (f"labels {', '.join(_label_text(n) for n in row)} over {xb - xa} px (fit residual {np.max(np.abs(res)):.1f} px"
              + (", labels in feet" if unit_m == FT else "") + ")")
        c = candidate("scale_bar", b / unit_m, 0.7 if len(row) >= 4 else 0.5, cfg.scale_tol_bar, ev, labels=len(row))
        same = next((k for k, o in enumerate(cands) if abs(math.log(o["px_per_m"] / c["px_per_m"])) < 0.02), None)
        if same is None:
            cands.append(c)
            found.append(ev)
        elif c["labels"] > cands[same]["labels"]:        # the same bar read from another anchor: the fuller row
            cands[same], found[same] = c, ev
    return {"candidates": cands[:3], "found": found[:5]}


def _label_text(n):
    """A bar label as read, or its number when several were read as one item ("2 3 4 5")."""
    return n[4]["text"] if n[4]["text"].strip() == str(n[0]) or len(n[4]["text"].split()) == 1 else str(n[0])


# ---------- door widths (moved from bench.py) ----------

def door_search(img, model, cfg=DEFAULT):
    """Segment the sheet at candidate scales and choose the one where detected doors are closest to
    cfg.scale_door_width wide (review step 6a). Returns the grid choice (as bench.propose_scale did), the table of
    rows and a candidate refined between grid steps: p * door width measured / door width expected."""
    from .model import DOOR, WALL, Sheet
    from .segment import segment
    m = cfg.px_per_m
    no_tta = replace(cfg, seg_tta=False)                 # the search segments each sheet up to 12 times
    rows = []
    for p in np.geomspace(12, 400, 12):
        f = m / p
        side = max(img.shape[:2]) * f
        if side < 300 or side > 3200:
            continue
        work = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        if work.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)
        s = Sheet("scale", "", "", "", work, (0.0, 0.0), {})
        segment(s, model, no_tta)
        n, cc, stats, _ = cv2.connectedComponentsWithStats((s.label == DOOR).astype(np.uint8), connectivity=8)
        widths = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.05 * m ** 2:
                continue
            ys, xs = np.nonzero(cc == i)
            (_, _), (w, h), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
            widths.append(max(w, h) / m)
        walls = s.label == WALL
        conf = float(s.prob.max(0)[s.label > 0].mean()) if (s.label > 0).any() else 0.0
        rows.append({"px_per_m": round(float(p), 1), "doors": len(widths), "door_m": round(float(np.median(widths)), 2) if widths else None,
                     "wall_share": round(float(walls.mean()), 3), "confidence": round(conf, 3)})
    if not rows:
        return {"px_per_m": None, "method": "no candidate scale fits the image size", "rows": rows, "candidates": []}
    grid, method, cand = door_choice(rows, cfg)
    return {"px_per_m": grid, "method": method, "rows": rows, "candidates": [cand]}


def door_choice(rows, cfg=DEFAULT):
    """From a door-search table: the grid scale and method as the original search chose them (closest to the target
    width; kept for bench.propose_scale), and the cue's candidate (the crossing, else the closest row refined)."""
    target = cfg.scale_door_width
    with_doors = [r for r in rows if r["doors"] >= 3]
    if with_doors:
        best = min(with_doors, key=lambda r: abs(np.log(r["door_m"] / target)))
        method = f"proposed: {best['doors']} detected doors measure {best['door_m']} m at this scale (target {target} m); to be confirmed"
        cand = door_crossing(rows, target, cfg)
        if cand is None:                                 # no crossing: the grid row closest to the target, refined
            refined = best["px_per_m"] * best["door_m"] / target
            cand = candidate("door_widths", refined, 0.25, cfg.scale_tol_doors,
                             f"{best['doors']} doors measure {best['door_m']} m at {best['px_per_m']} px/m (target "
                             f"{target} m; no crossing of the target)", doors=best["doors"])
    else:
        best = max(rows, key=lambda r: r["confidence"])
        method = "proposed: fewer than 3 doors found, highest segmentation confidence; weak, to be confirmed"
        cand = candidate("door_widths", best["px_per_m"], 0.05, 0.35,
                         "fewer than 3 doors at every candidate scale: highest segmentation confidence", doors=best["doors"])
    return best["px_per_m"], method, cand


def door_crossing(rows, target, cfg=DEFAULT):
    """Door-width candidate from the search table: detected doors look wide when the assumed scale is too small and
    narrow when it is too large, so the measured width crosses the target from above between two neighbouring
    candidate scales; interpolate there (in log). Far from its training scale the segmenter takes blobs for doors
    (dozens of them on a large working image), so among several crossings the one where the segmentation is most
    confident wins: it peaks near the training scale. None if there is no crossing."""
    rs = sorted((r for r in rows if r["doors"] >= 3), key=lambda r: r["px_per_m"])
    best = None
    for a, b in zip(rs, rs[1:]):
        if a["door_m"] >= target >= b["door_m"] and a["door_m"] > b["door_m"]:
            t = math.log(a["door_m"] / target) / math.log(a["door_m"] / b["door_m"])
            p = math.exp(math.log(a["px_per_m"]) + t * math.log(b["px_per_m"] / a["px_per_m"]))
            key = (a.get("confidence", 0) + b.get("confidence", 0), a["doors"] + b["doors"])
            if best is None or key > best[0]:
                best = (key, p, a, b)
    if best is None:
        return None
    n, p, a, b = best
    return candidate("door_widths", p, 0.4 if min(a["doors"], b["doors"]) >= 5 else 0.3, cfg.scale_tol_doors,
                     f"detected doors cross {target} m between {a['px_per_m']} px/m ({a['doors']} doors, {a['door_m']} m) "
                     f"and {b['px_per_m']} px/m ({b['doors']} doors, {b['door_m']} m)", doors=int(min(a["doors"], b["doors"])))


# ---------- stamp areas (after a first pass; the cue itself is computed in fpx.qa) ----------

def stamp_area_cue(stamp_cue, px_per_m_used, cfg=DEFAULT):
    """fpx.qa's stamp-area cue (length factor sqrt(stamp area / polygon area)) after a pass at px_per_m_used native
    pixels per metre -> candidate px_per_m_used / length factor."""
    if not stamp_cue or not stamp_cue.get("length_factor"):
        return {"candidates": []}
    f, n = stamp_cue["length_factor"], stamp_cue["rooms"]
    ev = f"{n} stamped rooms: polygon areas x {1 / f ** 2:.3f} at {px_per_m_used:.1f} px/m (length factor {f:.3f})"
    return {"candidates": [candidate("stamp_areas", px_per_m_used / f, 0.6 * min(1.0, n / 5), cfg.scale_tol_stamps, ev,
                                     rooms=n)], "used_px_per_m": round(float(px_per_m_used), 3)}


# ---------- room sizes on stamps ----------

SIZE_RE = re.compile(r"^(?P<a>[^x×X*]{2,24}?)\s*[x×X*]\s*(?P<b>[^x×X*]{2,24})$")


def parse_room_size(text):
    """A room size written on a stamp ("30'-6\" x 48'-6\"", "4.50 x 3.20", "450 x 320") -> [((w, d) in metres, unit)]
    hypotheses, or []. A quote-like mark on either side (also OCR's "30.6\"" and "48!6") makes both sides feet and
    inches; otherwise both sides are metric lengths with their unit hypotheses."""
    m = SIZE_RE.match(text.strip())
    if not m:
        return []
    sa, sb = m.group("a").strip(), m.group("b").strip()
    if re.search(r"[\"”″'’′!]", sa + sb):
        fa, fb = parse_feet_inches(sa, lenient=True), parse_feet_inches(sb, lenient=True)
        if fa is None or fb is None:
            return []
        return [((round(fa, 4), round(fb, 4)), "ft")] if 0.5 <= fa <= 60 and 0.5 <= fb <= 60 else []
    a, b = parse_dimension(sa), parse_dimension(sb)
    out = []
    for va, ua in a:
        for vb, ub in b:
            if ua == ub and 0.5 <= va <= 60 and 0.5 <= vb <= 60:
                out.append(((va, vb), ua))
    return out


def _ray(mask, y, x, step, min_run=1):
    """Distance from (x, y) to the first ink run of at least min_run pixels along a row (step +-1 in x), or None:
    with min_run > 1 thin lines (floor borders, furniture) are passed, a wall stops the ray."""
    line = mask[y, x:] if step > 0 else mask[y, :x + 1][::-1]
    if min_run <= 1:
        hit = np.flatnonzero(line)
        return int(hit[0]) if len(hit) else None
    starts, ends = _runs(line)
    for a, b in zip(starts, ends):
        if b - a >= min_run:
            return int(a)
    return None


def _extent(ink, box, w_tol=0.1, min_run=1):
    """Clear width and height of the white region around a text box: rays from the box's edges to the first ink (run
    of at least min_run pixels), three rows and three columns each; None unless the three agree within w_tol (a door
    gap or a fixture breaks the agreement) and the region is at least three boxes wide and high."""
    H, W = ink.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    if x0 < 1 or y0 < 1 or x1 >= W - 1 or y1 >= H - 1:
        return None
    h, w = max(y1 - y0, 2), max(x1 - x0, 2)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    widths, heights = [], []
    for y in (cy - h, cy, cy + h):
        if 0 <= y < H:
            L, R = _ray(ink, y, x0, -1, min_run), _ray(ink, y, x1, 1, min_run)
            if L is not None and R is not None:
                widths.append(w + L + R)
    inkT = np.ascontiguousarray(ink.T)
    for x in (cx - w // 3, cx, cx + w // 3):
        if 0 <= x < W:
            U, D = _ray(inkT, x, y0, -1, min_run), _ray(inkT, x, y1, 1, min_run)
            if U is not None and D is not None:
                heights.append(h + U + D)
    if len(widths) < 3 or len(heights) < 3:
        return None
    wm, hm = float(np.median(widths)), float(np.median(heights))
    if max(widths) - min(widths) > w_tol * wm or max(heights) - min(heights) > w_tol * hm:
        return None
    if wm < 1.5 * w or hm < 2 * h:                       # a stamp fills at most two thirds of its room
        return None
    return wm, hm


def wall_ink(ink, k):
    """Ink of the walls only: closed over speckle, then opened by k pixels so that lines thinner than k (floor borders,
    furniture, outlined walls) vanish and poché walls stay."""
    m = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0


def room_size_cue(texts, ink, cfg=DEFAULT):
    """Room sizes written on stamps ("30'-6\" x 48'-6\"", "4.50 x 3.20") against the clear extent of the white region
    around the stamp (rays to the walls): pixels per metre from the width and the depth, the two agreeing within the
    tolerance, either way round; robust estimate across the stamps as for dimension strings."""
    sizes = [(t, parse_room_size(t["text"])) for t in texts if t.get("angle", 0) == 0]
    sizes = [(t, h) for t, h in sizes if h]
    res = {"candidates": [], "pairs": 0, "numbers": len(sizes)}
    if not sizes:
        return res
    text_mask = np.zeros(ink.shape, bool)                # all text boxes, so that other stamps do not stop the rays
    for t in texts:
        x0, y0, x1, y1 = t["box"]
        text_mask[max(0, int(y0)):max(0, int(y1) + 1), max(0, int(x0)):max(0, int(x1) + 1)] = True
    clean = ink & ~text_mask
    # poché walls stop the rays where floor borders and furniture lines (thinner than a sixth of the text height) are
    # passed; on sheets with outlined walls that mask is empty and the rays go to the first ink instead
    th = float(np.median([t["box"][3] - t["box"][1] for t, _ in sizes]))
    walls = wall_ink(clean, max(3, int(0.15 * th)))
    tol = cfg.scale_tol_sizes
    rough_tol = 4 * tol                                  # an extent to the first line may stop at a floor border or furniture
    pairs = []
    for t, hyps in sizes:
        ext, rough = _extent(walls, t["box"]), False
        if ext is None:
            ext, rough = _extent(clean, t["box"]), True
        if ext is None:
            continue
        wpx, hpx = ext
        for (a, b), unit in hyps:
            best = None
            for wa, hb in ((a, b), (b, a)):              # "W x D" may be written either way round
                pw, ph = wpx / wa, hpx / hb
                if abs(math.log(pw / ph)) <= 2 * tol and (best is None or abs(math.log(pw / ph)) < best[0]):
                    best = (abs(math.log(pw / ph)), math.sqrt(pw * ph))
            if best is not None:
                pairs.append({"text": t["text"], "px_per_m": best[1], "unit": unit, "extent_px": (round(wpx), round(hpx)),
                              "rough": rough})
    res["pairs"] = len(pairs)
    if not pairs:
        return res
    cl = cluster([[p["px_per_m"]] for p in pairs], rough_tol if all(p["rough"] for p in pairs) else tol)
    if not cl:
        return res
    n_all = len(pairs)
    for k, c in enumerate(cl[:2]):
        n, frac = c["count"], c["count"] / n_all
        members = [pairs[j] for j in c["members"]]
        rough = sum(p["rough"] for p in members) > 0.4 * n       # mostly measured to the first line, not to a wall
        w = min(1.0, 0.3 * n) * math.sqrt(frac) * (1.0 if k == 0 else 0.6) * (0.5 if rough else 1.0)
        ex = members[:3]
        ev = (f"{n} of {n_all} room sizes agree with the clear extent around the stamp"
              f"{' (sizes in feet)' if ex and ex[0]['unit'] == 'ft' else ''}"
              f"{' (measured to the first line, not to a wall: rough)' if rough else ''}; e.g. "
              + ", ".join(f"{p['text']!r} in {p['extent_px'][0]} x {p['extent_px'][1]} px" for p in ex))
        res["candidates"].append(candidate("room_sizes", c["value"], w, rough_tol if rough else tol, ev, pairs=n,
                                           fraction=round(frac, 3), rough=rough, precise=bool(n >= 3 and frac >= 0.6 and not rough)))
    res["examples"] = [{"text": p["text"], "px_per_m": round(p["px_per_m"], 2), "extent_px": p["extent_px"], "rough": p["rough"],
                        "inlier": i in cl[0]["members"]} for i, p in enumerate(pairs[:40])]
    return res


# ---------- prior from the drawing's size on paper ----------

PRIOR_SCALES = (50, 100, 200)


def sheet_size_prior(extent_px, dpi=None, cfg=DEFAULT):
    """Without any scale cue: the drawing's extent on paper and a typical building size pick one of the standard
    scales 1:50, 1:100, 1:200 (the one whose building is closest to cfg.scale_prior_building_m on its longer side).
    Without a dpi 300 dpi is assumed. Returns {"candidates": [one weak candidate], "scale", "extent_mm"}."""
    d = float(dpi) if dpi else 300.0
    long_px = max(extent_px)
    long_mm = long_px / d * 25.4
    n = min(PRIOR_SCALES, key=lambda n: abs(math.log(long_mm * n / 1000 / cfg.scale_prior_building_m)))
    size_m = long_mm * n / 1000
    ev = (f"no scale cue: the drawing is {long_mm:.0f} mm long on paper{'' if dpi else ' (300 dpi assumed)'}, "
          f"1:{n} makes it {size_m:.0f} m (prior, to be confirmed)")
    return {"candidates": [candidate("sheet_size_prior", long_px / size_m, 0.1, 0.5, ev, scale=n, precise=False)],
            "scale": n, "extent_mm": round(long_mm, 1), "dpi_assumed": not dpi}


# ---------- consensus ----------

def _members(cands, v, tol_centre):
    """One candidate per cue that agrees with v: the closest within max(its tolerance, the centre's)."""
    mem = {}
    for c in cands:
        d = abs(math.log(c["px_per_m"] / v))
        if d <= max(c["tolerance"], tol_centre) and (c["cue"] not in mem or d < mem[c["cue"]][1]):
            mem[c["cue"]] = (c, d)
    return {k: c for k, (c, _) in mem.items()}


def consensus(cands, dpis=(), notes=(), cfg=DEFAULT):
    """Combine candidates of all cues (one vote per cue) into one scale with a confidence and QA flags.

    The winning group is the one with the largest summed weight around any candidate; ties go to the more precise
    centre. Its value is the precision-weighted mean (weight / tolerance²) in log space. It is snapped to an exact
    scale within cfg.scale_snap: to the scale note that agrees with it (at that note's dpi), or, with a known dpi
    (render, scan), to a standard scale or a note read on the sheet. dpis: [(dpi, source, note)]."""
    cands = [c for c in cands if c.get("px_per_m") and c["px_per_m"] > 0 and c["weight"] > 0]
    cues = sorted({c["cue"] for c in cands})
    out = {"px_per_m": None, "scale": None, "snapped": False, "confidence": "none", "agreeing": [], "disagreeing": [],
           "cues": cues, "flags": []}
    if not cands:
        out["flags"].append({"check": "scale", "severity": "high", "element": "sheet",
                             "message": "no scale cue found: the scale must be set by hand"})
        return out
    best = None
    for c in cands:
        mem = _members(cands, c["px_per_m"], c["tolerance"])
        key = (round(sum(m["weight"] for m in mem.values()), 6), -c["tolerance"], c["weight"])
        if best is None or key > best[0]:
            best = (key, mem)
    mem = best[1]
    wsum = sum(m["weight"] / m["tolerance"] ** 2 for m in mem.values())
    v = math.exp(sum(m["weight"] / m["tolerance"] ** 2 * math.log(m["px_per_m"]) for m in mem.values()) / wsum)
    # snapping: to an agreeing scale note at its dpi, else (known dpi only) to a standard or noted scale
    trusted = [d for d, s, _ in dpis if s in ("render", "scan")]
    targets = []                                         # (N, dpi, source, print factor r)
    note = next((m for k, m in mem.items() if k.startswith(("scale_note", "sheet_scale_note")) and m.get("dpi")), None)
    if note is not None:
        targets = [(note["scale"], note["dpi"], note["dpi_source"], note.get("r", 1.0))]
    elif trusted:
        targets = [(n, trusted[0], "known dpi", 1.0) for n in sorted(set(STANDARD_SCALES) | {int(n) for n in notes})]
    if targets:
        n, d, src, r = min(targets, key=lambda t: abs(math.log(v / (t[1] / INCH / t[0] * t[3]))))
        pn = d / INCH / n * r
        if abs(math.log(v / pn)) <= cfg.scale_snap:
            out.update(snapped=True, scale=f"1:{n}", dpi=d, unsnapped_px_per_m=round(v, 3),
                       snap_reason=f"1:{n} at {d:g} dpi ({src})" + (f", printed at r = {r:g}" if r != 1.0 else ""))
            if r != 1.0:
                out["print_factor"] = r
                out["scale_equivalent"] = f"1:{n / r:.1f} at {d:g} dpi"
            v = pn
    if out["scale"] is None and trusted:
        out["scale_equivalent"] = f"1:{trusted[0] / INCH / v:.1f} at {trusted[0]:g} dpi"
    if notes and not trusted:
        out["implied_dpi"] = {f"1:{n}": round(v * INCH * n, 1) for n in sorted(set(notes))}
    # which cues agree with the final value
    agree, disagree = {}, {}
    for cue in cues:
        cs = [c for c in cands if c["cue"] == cue]
        ok = [c for c in cs if abs(math.log(c["px_per_m"] / v)) <= max(c["tolerance"], cfg.scale_tol_dims)]
        if ok:
            agree[cue] = max(ok, key=lambda c: c["weight"])
        else:
            disagree[cue] = max(cs, key=lambda c: c["weight"])
    w_agree = sum(c["weight"] for c in agree.values())
    out["px_per_m"] = round(v, 3)
    out["agreeing"] = [{"cue": k, "px_per_m": c["px_per_m"], "ratio": round(c["px_per_m"] / v, 3), "weight": c["weight"],
                        "tolerance": c["tolerance"], "evidence": c["evidence"]} for k, c in agree.items()]
    out["disagreeing"] = [{"cue": k, "px_per_m": c["px_per_m"], "ratio": round(c["px_per_m"] / v, 3), "weight": c["weight"],
                           "evidence": c["evidence"]} for k, c in disagree.items()]
    precise = any(is_precise(c) for c in agree.values())
    heavy = [k for k, c in disagree.items() if c["weight"] >= 0.25]
    if len(agree) >= 2 and precise and not heavy:
        conf = "high"
    elif (precise and w_agree >= 0.5 and not heavy) or (len(agree) >= 2 and not heavy):
        conf = "medium"
    else:
        conf = "low"
    out["confidence"] = conf
    out["weight"] = round(w_agree, 3)
    for k, c in disagree.items():                        # every disagreement is flagged; its weight sets the severity
        big = is_precise(c) and abs(math.log(c["px_per_m"] / v)) > math.log(1.15)   # a precise cue far off
        sev = ("high" if c["weight"] >= 0.25 and (c["weight"] >= 0.5 * w_agree or big)
               else "medium" if c["weight"] >= 0.1 else "low")
        why = ""
        if k == "scale_note" and c.get("dpi") and any(a in agree for a in ("dimension_strings", "scale_bar")):
            why = (f"; the drawing measures 1:{c['dpi'] / INCH / v:.0f} at {c['dpi']:g} dpi against the noted "
                   f"1:{c['scale']}: printed reduced or enlarged, or a wrong dpi")
        out["flags"].append({"check": "scale", "severity": sev, "element": "sheet",
                             "message": f"scale cues disagree: {k} gives {c['px_per_m']:.1f} px/m ({c['evidence']}), "
                                        f"{ratio_text(c['px_per_m'], v)} the chosen {v:.1f} px/m "
                                        f"from {', '.join(agree)}{why}"})
    if len(agree) == 1:
        (k, c), = agree.items()
        out["flags"].append({"check": "scale", "severity": "medium" if is_precise(c) else "high",
                             "element": "sheet", "message": f"scale rests on one cue ({k}): to be confirmed"})
    return out


def compare_known(result, known_px_per_m, known_method, cfg=DEFAULT):
    """A loader or dataset scale is used as given; the cue consensus is checked against it and a flag raised if the
    two disagree beyond the precision of the consensus."""
    v = result.get("px_per_m")
    result["known"] = {"px_per_m": round(float(known_px_per_m), 3), "method": known_method}
    if not v:                                            # nothing on the sheet to check the scale used against
        result["flags"] = [{"check": "scale", "severity": "low", "element": "sheet",
                            "message": f"no scale cue on the sheet confirms the scale used ({known_method})"}]
        return None
    tol = max(cfg.scale_tol_dims, min(a["tolerance"] for a in result["agreeing"]))
    r = v / known_px_per_m
    agrees = abs(math.log(r)) <= tol
    result["known"] = {"px_per_m": round(float(known_px_per_m), 3), "method": known_method, "ratio": round(r, 3),
                       "agrees": agrees}
    if agrees:                                           # the scale used and the cues confirm each other
        result["flags"] = [f for f in result["flags"] if "rests on one cue" not in f["message"]
                           and not f["message"].startswith("units ambiguous")]
    else:
        result["flags"].append({"check": "scale", "severity": "high" if result["confidence"] != "low" else "medium",
                                "element": "sheet",
                                "message": f"scale cues ({result['confidence']} confidence) give {v:.1f} px/m, "
                                           f"{ratio_text(v, known_px_per_m)} the scale used ({known_px_per_m:.1f} px/m, {known_method})"})
    return agrees


# ---------- pre-pass ----------

def prepass(img, engine=None, model=None, dpi=None, dpi_source=None, native_text=(), meta_dpi=None, cfg=DEFAULT,
            texts=None):
    """Scale cues on the native sheet (before resampling) and their consensus.

    img: native image (RGB or grey); engine: OCR engine (None: native text only); model: segmenter for the door
    search (None: skipped); dpi/dpi_source: known resolution ("render", "scan"); otherwise inferred from the paper
    size or meta_dpi. native_text: text items in native pixels (vector PDF). texts: OCR items from an earlier call.
    Returns {"cues": {...}, "consensus": {...}, "seconds": {...}, "ocr": {...}}."""
    t0 = time.time()
    gray = grey(img)
    dpis = [(float(dpi), dpi_source, f"{dpi:g} dpi ({dpi_source})")] if dpi else infer_dpi(gray.shape, meta_dpi)
    secs, ocr_info = {}, None
    items = list(native_text)
    if texts is None and engine is not None:
        texts, ocr_info = read_text(gray, engine, cfg, dpi=dpi if dpi_source in ("render", "scan") else None)
        secs["ocr"] = ocr_info["seconds"]
    for t in texts or ():                                # OCR of what the native text layer already holds is dropped
        bx = t["box"]
        area = max((bx[2] - bx[0]) * (bx[3] - bx[1]), 1e-6)
        if not any(max(0, min(bx[2], n["box"][2]) - max(bx[0], n["box"][0])) *
                   max(0, min(bx[3], n["box"][3]) - max(bx[1], n["box"][1])) > 0.3 * area for n in native_text):
            items.append(t)
    t = time.time()
    ink = binarise(gray)
    cues = {"scale_note": note_cue(items, dpis, cfg)}
    t1 = time.time()
    cues["dimension_strings"] = dimension_cue(items, ink, cfg)
    secs["dimension_strings"] = round(time.time() - t1, 2)
    cues["scale_bar"] = scale_bar_cue(items, ink, cfg)
    secs["text_cues"] = round(time.time() - t, 2)
    if model is not None:
        t = time.time()
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cues["door_widths"] = door_search(rgb, model, cfg)
        secs["door_widths"] = round(time.time() - t, 1)
    result = {"cues": cues, "dpis": [list(d) for d in dpis], "ocr": ocr_info, "seconds": secs, "texts": items}
    combine(result, cfg)
    secs["total"] = round(time.time() - t0, 1)
    return result


def combine(result, cfg=DEFAULT):
    """(Re)compute the consensus of a prepass result, e.g. after add_cue."""
    cues = result["cues"]
    notes = [f["scale"] for f in cues["scale_note"]["found"]]
    cands = [c for cue in cues.values() for c in cue["candidates"]]
    con = consensus(cands, [tuple(d) for d in result["dpis"]], notes, cfg)
    amb = cues.get("dimension_strings", {}).get("ambiguous")
    if amb and [a["cue"] for a in con["agreeing"]] == ["dimension_strings"]:
        con["flags"].append({"check": "scale", "severity": "medium", "element": "sheet", "message": amb})
    result["consensus"] = con
    return con


def add_cue(result, name, cue, cfg=DEFAULT):
    """Add a cue after a first pass (stamp areas) and recompute the consensus."""
    result["cues"][name] = cue
    return combine(result, cfg)


def record(result):
    """JSON-friendly scale record for Sheet.scale: cues with their candidates and evidence, and the consensus."""
    return {"cues": result["cues"], "consensus": result["consensus"], "dpis": result["dpis"], "ocr": result["ocr"],
            "seconds": result["seconds"]}


# ---------- stage 1c: scale proposal per drawing (docs/pipeline.md 1c) ----------
#
# The factor model keeps the three factors apart: metres per pixel m = N * 25.4 / (1000 * d * r), with N the drawing
# scale (1:N), d the resolution in dpi and r the reproduction factor of the print (1 for the original, 0.5 for an A1
# sheet copied to A3). Dimension strings, scale bars, viewports and stamp areas measure m directly; a scale note gives
# N only and needs d and r. SIA 400 expects reductions in sqrt(2) steps (one A format per step).

SQRT2_FORMATS = {1: "one A format smaller (e.g. A1 -> A2)", 2: "two A formats smaller (e.g. A1 -> A3)",
                 3: "three A formats smaller (e.g. A1 -> A4)", 4: "four A formats smaller (e.g. A0 -> A4)",
                 -1: "one A format larger", -2: "two A formats larger"}
NTS_RE = re.compile(r"nicht\s*ma[sß]+st|not\s*to\s*scale|\bn\.?\s*t\.?\s*s\.?\b|sans\s*[ée]chelle|non\s*in\s*scala", re.I)


def metres_per_px(n, dpi, r=1.0):
    """Factor model: plan metres per raster pixel of a drawing at 1:n, scanned or rendered at dpi, printed at factor r."""
    return n * 25.4 / (1000.0 * dpi * r)


def note_px_per_m(n, dpi, r=1.0):
    """Raster pixels per plan metre for a scale note 1:n at dpi and print factor r (1 / metres_per_px)."""
    return 1.0 / metres_per_px(n, dpi, r)


def print_factor(px_per_m, n, dpi):
    """Reproduction factor r implied by a measured scale (px per plan metre), a noted 1:n and the dpi."""
    return px_per_m * metres_per_px(n, dpi, 1.0)


def sqrt2_step(r, tol=0.04):
    """Nearest sqrt(2) step k (r = 2^(-k/2), k > 0 reduced) of a print factor and whether r lies within tol of it."""
    k = int(round(-2 * math.log2(r))) if r > 0 else 0
    rk = 2.0 ** (-k / 2)
    return {"r": round(r, 4), "step": k, "r_step": round(rk, 4), "fits": abs(math.log(r / rk)) <= tol,
            "off_pct": round((r / rk - 1) * 100, 1), "format": SQRT2_FORMATS.get(k, "original size" if k == 0 else f"{k} steps")}


def print_factor_check(px_per_m, n, dpi, measured_by, cfg=DEFAULT, tol=0.04):
    """Compare a measured scale with a noted 1:n at a trusted dpi through the factor model. Returns the print factor
    hypothesis and a flag when the drawing was printed reduced or enlarged (the note and the measurement differ by a
    power of sqrt(2)) or by a factor that is no sqrt(2) step (fit-to-page, a wrong dpi or a wrong note)."""
    r = print_factor(px_per_m, n, dpi)
    h = sqrt2_step(r, tol)
    h.update({"note": f"1:{n}", "dpi": dpi, "measured_by": measured_by, "equivalent_scale": f"1:{n / r:.1f}"})
    flag = None
    if abs(math.log(r)) <= max(cfg.scale_tol_dims, cfg.scale_snap):
        h["hypothesis"] = "original print (r = 1)"
    elif h["fits"] and h["step"] != 0:
        h["hypothesis"] = f"{'reduced' if h['step'] > 0 else 'enlarged'} print, r = {h['r_step']:g} ({h['format']})"
        flag = {"check": "scale", "severity": "medium", "element": "drawing",
                "message": f"reduced or enlarged print: the drawing measures {h['equivalent_scale']} ({measured_by}) "
                           f"against the noted 1:{n} at {dpi:g} dpi, print factor {r:.3f} ~ {h['r_step']:g} "
                           f"({h['format']}, {h['off_pct']:+.1f} %); measured scale kept"}
    else:
        h["hypothesis"] = f"print factor {r:.3f}, no sqrt(2) step (fit to page, a wrong dpi or a note of another drawing)"
        flag = {"check": "scale", "severity": "high", "element": "drawing",
                "message": f"noted 1:{n} and the measured scale ({measured_by}, {h['equivalent_scale']} at {dpi:g} dpi) "
                           f"differ by {r:.3f}, which is no sqrt(2) step: check the note, the dpi and the print"}
    return h, flag


def find_scale_bars(texts, ink, cfg=DEFAULT):
    """Labelled scale bars with their position: the search of scale_bar_cue (labels 0, 5, 10 ... along a bar, label
    centres standing for the divisions) returning, per bar, the box of labels and bar (raster px), px per metre, the
    number of labels and the evidence. scale_bar_cue is unchanged for existing callers."""
    out = []
    H, W = ink.shape
    for row, (v0, zx, zy, zh, zt), b, res, unit_m in _bar_rows(_bar_labels(texts)):
        xa, xb = int(max(0, min(n[1] for n in row))), int(min(W, max(n[1] for n in row)))
        ya, yb = int(max(0, zy - 3 * zh)), int(min(H, zy + 3 * zh))
        if xb - xa < 10 or not _bar_ink(ink[ya:yb, xa:xb]):
            continue
        prof = ink[ya:yb, xa:xb].mean(1)
        bar_rows = np.flatnonzero(prof >= max(BAR_INK / 2, prof.max() * 0.5))
        boxes = [n[4]["box"] for n in row]
        bx = (min(min(bb[0] for bb in boxes), xa), min(min(bb[1] for bb in boxes), ya + bar_rows.min()),
              max(max(bb[2] for bb in boxes), xb), max(max(bb[3] for bb in boxes), ya + bar_rows.max() + 1))
        ev = (f"labels {', '.join(_label_text(n) for n in row)} over {xb - xa} px (fit residual {np.max(np.abs(res)):.1f} px"
              + (", labels in feet" if unit_m == FT else "") + ")")
        if any(abs(o["box"][0] - bx[0]) < zh and abs(o["box"][1] - bx[1]) < zh for o in out):
            continue
        same = next((k for k, o in enumerate(out) if abs(math.log(o["px_per_m"] / (b / unit_m))) < 0.02
                     and min(o["box"][2], bx[2]) - max(o["box"][0], bx[0]) > 0), None)
        if same is not None:                             # the same bar read from another anchor: keep the fuller row
            if len(row) > out[same]["labels"]:
                out[same] = {"box": tuple(float(v) for v in bx), "px_per_m": round(b / unit_m, 3), "labels": len(row), "evidence": ev}
            continue
        out.append({"box": tuple(float(v) for v in bx), "px_per_m": round(b / unit_m, 3), "labels": len(row), "evidence": ev})
    return out[:5]


def drawing_scale(img, texts=(), dpi=None, dpi_source=None, dpis=None, caption_notes=(), sheet_notes=(),
                  viewport_scale=None, model_px_per_m=None, extra=(), nts=False, model=None, extent_px=None, cfg=DEFAULT):
    """Scale proposal of one drawing region before segmentation (stage 1c): its own cues only.

    img: the drawing's crop (native raster pixels, masked); texts: text items inside the drawing (crop pixels);
    dpi, dpi_source: the sheet's resolution (render and scan are trusted); dpis: [(dpi, source, note)] alternatives when
    it is not; caption_notes: N of the drawing's own caption notes; sheet_notes: N of the title-block notes, used as
    the sheet default only when the drawing has no note of its own; viewport_scale: N of a DWG/DXF viewport (exact);
    model_px_per_m: exact pixels per metre of a model-space render; extra: further candidates (a scale bar linked to
    the drawing but outside its polygon); model: the segmenter, for the door-width cue (door_search on the crop), run
    only when no precise cue (viewport, model units, dimension strings, scale bar, room sizes, note at a trusted dpi)
    exists; extent_px: (width, height) of the drawing in raster pixels for the sheet-size prior, the last resort when
    there is no cue at all (sheet_size_prior: 1:50, 1:100 or 1:200 by the drawing's size on paper, flagged high).
    The factor model is applied explicitly: each note gives px_per_m = 1 / metres_per_px(N, d, r = 1), and the print
    factor implied by direct cues is tested against sqrt(2) steps; when a step fits, the note is re-entered at that
    step ("scale_note (r=0.5)") and the disagreement becomes one medium "reduced print" flag.
    Returns {"px_per_m", "metres_per_px", "confidence", "consensus", "cues", "note", "print_factor", "flags",
    "dpis", "notes", "confirmed"}; add_drawing_cue() and confirm() update it later."""
    gray = grey(img) if img is not None and img.size else None
    if dpis is None:
        dpis = [(float(dpi), dpi_source, f"{dpi:g} dpi ({dpi_source})")] if dpi else []
    trusted = [d for d, s, _ in dpis if s in ("render", "scan")]
    cues = {}
    own = sorted({int(n) for n in caption_notes})
    sheet = sorted({int(n) for n in sheet_notes})
    note_n, note_src = (own, "caption") if own else (sheet, "title block (sheet default)") if sheet else ([], None)
    cands = []
    if viewport_scale:
        d0 = trusted[0] if trusted else (dpi or 0)
        if d0:
            v = note_px_per_m(viewport_scale, d0)
            cues["viewport"] = {"candidates": [candidate("viewport", v, 1.0, 0.002, f"layout viewport 1:{viewport_scale:g} at {d0:g} dpi (exact)",
                                                         scale=viewport_scale, dpi=d0, precise=True)]}
            cands += cues["viewport"]["candidates"]
    if model_px_per_m:
        cues["model_units"] = {"candidates": [candidate("model_units", model_px_per_m, 1.0, 0.001, "model-space render: exact model units",
                                                        precise=True)]}
        cands += cues["model_units"]["candidates"]
    note_texts = [{"text": f"1:{n}", "source": note_src} for n in note_n]
    nc = note_cue(note_texts, dpis, cfg) if note_texts else {"candidates": [], "found": []}
    if note_src and note_src != "caption":                    # the sheet default counts half
        for c in nc["candidates"]:
            c["cue"], c["weight"] = "sheet_scale_note", round(c["weight"] * 0.5, 3)
            c["evidence"] = "title block (sheet default): " + c["evidence"]
    note_key = "scale_note" if note_src == "caption" else "sheet_scale_note"
    cues[note_key] = nc
    cands += nc["candidates"]
    if gray is not None and len(texts):
        ink = binarise(gray)
        cues["dimension_strings"] = dimension_cue(list(texts), ink, cfg)
        cues["scale_bar"] = scale_bar_cue(list(texts), ink, cfg)
        cues["room_sizes"] = room_size_cue(list(texts), ink, cfg)
        cands += cues["dimension_strings"]["candidates"] + cues["scale_bar"]["candidates"] + cues["room_sizes"]["candidates"]
    if extra:
        cues["linked"] = {"candidates": list(extra)}
        cands += list(extra)
    if model is not None and gray is not None and not any(is_precise(c) for c in cands):
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cues["door_widths"] = door_search(rgb, model, cfg)
        cands += cues["door_widths"]["candidates"]
    if not cands and (extent_px or gray is not None):
        ext = extent_px or (gray.shape[1], gray.shape[0])
        cues["sheet_size_prior"] = sheet_size_prior(ext, trusted[0] if trusted else (dpis[0][0] if dpis else None), cfg)
        cands += cues["sheet_size_prior"]["candidates"]
    ctx = {"dpis": [list(d) for d in dpis], "notes": note_n, "note_key": note_key, "own": own, "sheet": sheet,
           "viewport": bool(viewport_scale), "model_units": bool(model_px_per_m),
           "nts": bool(nts or any(NTS_RE.search(t.get("text", "")) for t in texts))}
    s = {"cues": cues, "note": None if not note_n else {"scale": note_n[0], "all": note_n, "source": note_src},
         "dpis": ctx["dpis"], "notes": note_n, "context": ctx, "confirmed": None}
    _resolve(s, cfg)
    return s


def _note_candidates(cues, note_key):
    return [c for c in cues.get(note_key, {}).get("candidates", [])]


def _resolve(s, cfg=DEFAULT):
    """Consensus, print factor and flags of a drawing's cues (drawing_scale's result s, updated in place). A print
    factor that fits a sqrt(2) step re-enters the note at that step and recomputes the consensus."""
    cues, ctx = s["cues"], s["context"]
    dpis = [tuple(d) for d in s["dpis"]]
    trusted = [d for d, src, _ in dpis if src in ("render", "scan")]
    note_n, note_key = s["notes"], ctx["note_key"]
    stepped = [k for k in cues if k.startswith(note_key + " (r=")]
    for k in stepped:                                    # start again from the plain cues
        del cues[k]
    cands = [c for k, cue in cues.items() for c in cue["candidates"]]
    con = consensus(cands, dpis, note_n, cfg)
    pf, pf_flag = _print_factor(con, note_n, trusted, cfg)
    if pf and pf.get("fits") and pf.get("step"):
        # T7: the note at the fitted print step is a candidate of its own; the plain note is left out of the vote,
        # so that the measurement and the stepped note agree instead of the note disagreeing
        r = pf["r_step"]
        n = int(pf["note"].split(":")[1])
        key = f"{note_key} (r={r:g})"
        stepped_cands = []
        for c in _note_candidates(cues, note_key):
            if c.get("scale") == n and c.get("dpi"):
                stepped_cands.append(candidate(key, note_px_per_m(n, c["dpi"], r), c["weight"], c["tolerance"],
                                               f"{c['evidence']}, printed at r = {r:g} ({pf['format']})", scale=n,
                                               dpi=c["dpi"], dpi_source=c.get("dpi_source"), precise=c.get("precise", True), r=r))
        if stepped_cands:
            cues[key] = {"candidates": stepped_cands, "note": f"1:{n}", "print_factor": r}
            cands = [c for c in cands if not (c["cue"] == note_key and c.get("scale") == n)] + stepped_cands
            con = consensus(cands, dpis, note_n, cfg)
            pf["hypothesis"] += "; the stepped note joins the consensus"
    amb = cues.get("dimension_strings", {}).get("ambiguous")
    if amb and [a["cue"] for a in con["agreeing"]] == ["dimension_strings"]:
        con["flags"].append({"check": "scale", "severity": "medium", "element": "drawing", "message": amb})
    for f in con["flags"]:
        f["element"] = "drawing"
    flags = list(con["flags"])
    if "sheet_size_prior" in cues and con["px_per_m"]:
        prior = cues["sheet_size_prior"]
        flags = [f for f in flags if "rests on one cue" not in f["message"]]
        flags.append({"check": "scale", "severity": "high", "element": "drawing",
                      "message": f"no scale cue on the drawing: run at the prior 1:{prior['scale']} from its size on "
                                 f"paper ({prior['extent_mm']} mm{', 300 dpi assumed' if prior.get('dpi_assumed') else ''}); "
                                 "to be confirmed (two-point calibration or a known room size)"})
    if pf_flag:
        flags.append(pf_flag)
    v = con["px_per_m"]
    if pf is None and v and note_n and trusted:
        pf = {"r": 1.0, "hypothesis": "print factor assumed 1 (original print): no direct cue to test it; a print "
                                      "reduced by one or two A formats would be off by 1.41 or 2", "note": f"1:{note_n[0]}"}
        flags.append({"check": "scale", "severity": "medium", "element": "drawing",
                      "message": f"scale from the note 1:{note_n[0]} and {trusted[0]:g} dpi only: the print factor "
                                 "(reduced or enlarged copy) is not confirmed by a direct cue"})
    own, sheet = ctx["own"], ctx["sheet"]
    if not own and not ctx["viewport"] and not ctx["model_units"]:
        flags.append({"check": "scale", "severity": "low", "element": "drawing",
                      "message": "drawing without its own scale note" + (f": the sheet default 1:{sheet[0]} is used"
                                                                         if sheet else "")})
    if len(own) > 1:
        flags.append({"check": "scale", "severity": "medium", "element": "drawing",
                      "message": f"several scale notes in the caption: {', '.join(f'1:{n}' for n in own)}"})
    if ctx["nts"]:
        flags.append({"check": "scale", "severity": "high", "element": "drawing",
                      "message": "'not to scale' (nicht massstäblich / NTS) on the drawing: measurements are unreliable"})
    if not trusted:
        why = "missing" if not dpis else "a screen default" if any(src == "metadata" and round(d) in SCREEN_DPI for d, src, _ in dpis) \
            else "not verified"
        direct = [a for a in con["agreeing"] if a["cue"] in DIRECT_CUES]
        flags.append({"check": "scale", "severity": "low" if v and direct else "medium", "element": "drawing",
                      "message": f"resolution {why}: scale notes cannot be converted exactly"})
    s.update(px_per_m=v, metres_per_px=None if not v else round(1.0 / v, 8), confidence=con["confidence"],
             consensus=con, print_factor=pf, flags=flags)
    return s


DIRECT_CUES = ("dimension_strings", "scale_bar", "viewport", "model_units", "linked", "room_sizes", "stamp_areas")


def _print_factor(con, note_n, trusted, cfg=DEFAULT):
    """Print factor of the consensus against the noted scales (the closest), or (None, None) without a note, a
    trusted dpi or a direct measurement among the agreeing cues."""
    v = con["px_per_m"]
    direct = [a for a in con["agreeing"] if a["cue"] in DIRECT_CUES]
    if not (v and note_n and trusted and direct):
        return None, None
    pf = pf_flag = None
    for n in note_n:
        h, flag = print_factor_check(v, n, trusted[0], ", ".join(a["cue"] for a in direct), cfg, tol=cfg.scale_print_step_tol)
        if pf is None or abs(math.log(h["r"])) < abs(math.log(pf["r"])):
            pf, pf_flag = h, flag
    return pf, pf_flag


def add_drawing_cue(s, name, cue, cfg=DEFAULT):
    """Add a cue after a first pass (stamp areas after stage 9) to a drawing_scale result and recompute its
    consensus, print factor and flags. A confirmed scale is not moved: the cue is recorded as agreeing or not."""
    s["cues"][name] = cue
    if s.get("confirmed"):
        return confirm(s, s["confirmed"]["px_per_m"], s["confirmed"]["note"], cfg)
    return _resolve(s, cfg)


def confirm(s, px_per_m, note=None, cfg=DEFAULT):
    """A scale confirmed by the user replaces the consensus of a drawing_scale result: the value becomes the one
    confirmed, every measured cue is listed as agreeing or disagreeing against it (a disagreement is a flag), the
    proposal's own verdicts go. Recorded under s["confirmed"] and the consensus; method "confirmed by the user"."""
    v = float(px_per_m)
    if not v > 0:
        raise ValueError("the confirmed scale must be a positive number of pixels per metre")
    note = note or "confirmed by the user"
    cands = [c for cue in s["cues"].values() for c in cue["candidates"] if c.get("px_per_m") and c["weight"] > 0]
    con = s.get("consensus") or {}
    old = con.get("px_per_m")
    agree, disagree = {}, {}
    for c in cands:
        d = abs(math.log(c["px_per_m"] / v))
        ok = d <= max(c["tolerance"], cfg.scale_tol_dims)
        pool = agree if ok else disagree
        if c["cue"] not in pool or c["weight"] > pool[c["cue"]]["weight"]:
            pool[c["cue"]] = c
    for k in list(disagree):                             # a cue agrees if any of its candidates does
        if k in agree:
            del disagree[k]
    row = lambda c: {"cue": c["cue"], "px_per_m": c["px_per_m"], "ratio": round(c["px_per_m"] / v, 3), "weight": c["weight"],
                     "tolerance": c["tolerance"], "evidence": c["evidence"]}
    rec = {"px_per_m": round(v, 3), "note": note, "method": "confirmed by the user", "proposal_px_per_m": old,
           "ratio_to_proposal": None if not old else round(v / old, 4),
           "agrees_with": sorted(agree), "disagrees_with": sorted(disagree)}
    flags = [{"check": "scale", "severity": "medium" if c["weight"] >= 0.25 else "low", "element": "drawing",
              "message": f"{k} gives {c['px_per_m']:.1f} px/m ({c['evidence']}), {ratio_text(c['px_per_m'], v)} the "
                         f"confirmed {v:.1f} px/m ({note})"} for k, c in disagree.items()]
    if old and abs(math.log(v / old)) > cfg.scale_snap:
        flags.append({"check": "scale", "severity": "low", "element": "drawing",
                      "message": f"scale confirmed at {v:.1f} px/m, {v / old:.3f}x the pipeline's proposal {old:.1f} px/m ({note})"})
    if s["context"]["nts"]:
        flags.append({"check": "scale", "severity": "high", "element": "drawing",
                      "message": "'not to scale' (nicht massstäblich / NTS) on the drawing: measurements are unreliable"})
    new_con = {"px_per_m": round(v, 3), "scale": None, "snapped": False, "confidence": "confirmed",
               "agreeing": [row(c) for c in agree.values()], "disagreeing": [row(c) for c in disagree.values()],
               "cues": sorted({c["cue"] for c in cands}), "flags": flags, "confirmed_by": rec,
               "weight": round(sum(c["weight"] for c in agree.values()), 3), "proposal": con or None}
    s.update(px_per_m=round(v, 3), metres_per_px=round(1.0 / v, 8), confidence="confirmed", consensus=new_con,
             flags=flags, confirmed=rec)
    return s
