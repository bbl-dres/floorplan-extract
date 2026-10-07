"""Local conformance check of an exported DXF against BBL's CAD-Richtlinie, as plan-check implements it.

    python -m fpx.conformance data/out/s1.dxf [data/out/s2.dxf ...] [--json]

plan-check (github.com/bbl-dres/plan-check, docs/pruefregeln-de.md and js/validation.js, commit 420fd9b of
17 September 2026) reads DWG only, and the DWG conversion needs a converter that is not approved yet. This module
re-implements plan-check's 40 rules on the DXF with ezdxf, following its code rather than only its documentation:
block and dimension content is expanded (entities on layer 0 take the layer of the reference), colours resolve as in
plan-check (BYBLOCK counts as not BYLAYER), text extents use plan-check's rough glyph width, and its tolerances are
kept (0.5 mm for AOID base points on a room edge, 100 mm around the plan frame, 1 mm for implicitly closed polylines).
What a DWG conversion could still change is listed in DWG_NOTES.

Extra checks (FPX_*) cover what the guideline asks for but plan-check does not test: valid room polygons, rooms inside
the floor polygon, overlapping rooms, one floor polygon per floor, and deductions over 5 m² cut out of the floor polygon.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ezdxf
import shapely
from shapely.geometry import LineString, Point, Polygon

# (code, severity, description) as in plan-check's js/rules.js and locales/de.json (translated)
RULES = [
    ("LAYER_001", "error", "required layer R_RAUMPOLYGON missing"),
    ("LAYER_002", "error", "required layer R_AOID missing"),
    ("LAYER_003", "error", "required layer R_GESCHOSSPOLYGON missing"),
    ("LAYER_004", "warning", "required layer A_ARCHITEKTUR missing"),
    ("LAYER_005", "warning", "required layer V_PLANLAYOUT missing"),
    ("LAYER_006", "warning", "required layer V_BEMASSUNG missing"),
    ("LAYER_007", "warning", "required layer A_SCHRAFFUR missing"),
    ("LAYER_008", "warning", "unknown layer"),
    ("POLY_001", "error", "room polygon not closed"),
    ("POLY_002", "error", "room polygon has arc segments"),
    ("POLY_003", "error", "polygon with fewer than 3 vertices"),
    ("POLY_004", "warning", "room area very small (< 0.25 m²)"),
    ("POLY_005", "warning", "possible duplicate room polygon"),
    ("POLY_006", "error", "element on R_RAUMPOLYGON is not an LWPOLYLINE"),
    ("POLY_007", "warning", "room polygon intersects itself"),
    ("POLY_008", "warning", "no room polygon"),
    ("GPOLY_001", "error", "floor polygon not closed"),
    ("GPOLY_002", "error", "floor polygon has arc segments"),
    ("GPOLY_003", "error", "element on R_GESCHOSSPOLYGON is not an LWPOLYLINE"),
    ("GPOLY_004", "warning", "no floor polygon"),
    ("GPOLY_005", "warning", "possible duplicate floor polygon"),
    ("AOID_001", "error", "room polygon without AOID"),
    ("AOID_002", "error", "AOID not unique"),
    ("AOID_003", "warning", "AOID format invalid (WWWW.GG.EE.RRR)"),
    ("AOID_004", "warning", "several texts on R_AOID in one room polygon"),
    ("AOID_005", "warning", "AOID text outside all room polygons"),
    ("AOID_006", "warning", "AOID base point outside its room polygon"),
    ("GEOM_001", "error", "drawing unit is not millimetres"),
    ("GEOM_002", "warning", "element with Z coordinate != 0"),
    ("GEOM_003", "error", "forbidden entity type (MLINE, ELLIPSE, SPLINE, OLE)"),
    ("GEOM_004", "warning", "external reference (XREF)"),
    ("GEOM_005", "warning", "element outside the plan frame"),
    ("TEXT_001", "warning", "text on a layer not allowed for text"),
    ("TEXT_002", "warning", "font is not ARIAL"),
    ("STYLE_001", "warning", "polyline width is not 0"),
    ("STYLE_002", "warning", "colour is not BYLAYER"),
    ("LAYOUT_001", "warning", "layout tab (paper space) in use"),
    ("LAYOUT_002", "warning", "no plan frame on V_PLANLAYOUT"),
    ("DIM_001", "warning", "no dimensions on V_BEMASSUNG"),
    ("HATCH_001", "warning", "hatch on A_SCHRAFFUR is not SOLID"),
]
EXTRA_RULES = [
    ("FPX_VALID", "error", "room polygon is not a valid area (overlapping or crossing edges; zero-width keyhole bridges allowed)"),
    ("FPX_IN_GF", "error", "room polygon (minus its deductions) not inside the floor polygon"),
    ("FPX_OVERLAP", "error", "room polygons overlap"),
    ("FPX_GF_ONE", "warning", "not exactly one floor polygon"),
    ("FPX_GF_CUT", "error", "deduction over 5 m² inside the floor not cut out of the floor polygon"),
]
# What the DWG step (and plan-check's DWG reader, LibreDWG) could change compared with this DXF check
DWG_NOTES = {
    "GEOM_001": "$INSUNITS must survive the conversion (ODA keeps header variables)",
    "TEXT_002": "plan-check reads the font file of the text style (STYLE table); the converter must keep it",
    "LAYOUT_001": "the converter may add a viewport to the empty Layout1; plan-check ignores VIEWPORT entities",
    "POLY_001": "plan-check reads LibreDWG's raw LWPOLYLINE closed flag (bit 512); the converter must set it",
    "GPOLY_001": "as POLY_001",
}

ROOM, AOID, GF, DEDUCTION = "R_RAUMPOLYGON", "R_AOID", "R_GESCHOSSPOLYGON", "R_RAUMPOLYGON-ABZUG"
FRAME, HATCH, DIMENSION = "V_PLANLAYOUT", "A_SCHRAFFUR", "V_BEMASSUNG"
# CAD-Richtlinie BBL V1.0, Kap. 5.2, Tabelle 4: layer -> ACI colour; the first seven are required
CAFM_LAYERS = {
    ROOM: 210, AOID: 7, GF: 214, "A_ARCHITEKTUR": 253, HATCH: 8, DIMENSION: 251, FRAME: 252,
    DEDUCTION: 230, "A_ELEKTRO": 150, "A_HEIZUNG-KUEHLUNG": 1, "A_LUEFTUNG": 4, "A_SANITAER": 92, "V_ACHSEN": 251,
    "V_REFERENZPUNKT": 30, "V_TEXT": 253,
}
REQUIRED = {"LAYER_001": ROOM, "LAYER_002": AOID, "LAYER_003": GF, "LAYER_004": "A_ARCHITEKTUR", "LAYER_005": FRAME,
            "LAYER_006": DIMENSION, "LAYER_007": HATCH}
SYSTEM_LAYERS = {"0", "Defpoints", "R_RAUMSTEMPEL"}
TEXT_LAYERS = {FRAME, "V_ACHSEN", "V_TEXT", AOID}
AOID_PATTERN = re.compile(r"^\d{4}\.[A-Za-z0-9]{1,4}\.\d{2}\.\d{3}$")
PARKING_AOID_PATTERN = re.compile(r"^\d{4}\.\d+\.\d{3}$")
INSUNITS_MM = 4
INSUNITS_TO_MM = {1: 25.4, 2: 304.8, 3: 1609344, 4: 1, 5: 10, 6: 1000, 7: 1e6, 10: 914.4, 14: 100, 15: 10000, 16: 100000}
FORBIDDEN = {"MLINE", "ELLIPSE", "SPLINE", "OLE2FRAME", "OLEFRAME"}
MIN_ROOM_AREA_M2 = 0.25
BULGE_EPSILON = 1e-6
FRAME_TOLERANCE_MM = 100
BOUNDARY_TOLERANCE_MM = 0.5
CLOSE_TOLERANCE_MM = 1
TEXT_WIDTH_PER_CHAR = 0.6
GF_DEDUCTION_M2 = 5.0              # CAD-Richtlinie Kap. 5.9: stair eyes and air spaces over 5 m² are cut out of the GF
AREA_TOLERANCE_M2 = 0.01           # FPX checks: overlaps and protrusions smaller than this are rounding


# ---------------------------------------------------------------------------------------------------------------------
# Flattening: entities -> items as plan-check builds them (dwg-processing.js, prepareDrawingData)

def _xy(v):
    return float(v[0]), float(v[1])


def _bounds(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys)) if pts else None


def _by_layer(e):
    """plan-check: a true colour or any ACI other than 256 (BYLAYER) is not BYLAYER; BYBLOCK (0) neither."""
    if e.dxf.hasattr("true_color"):
        return False
    return e.dxf.get("color", 256) == 256


def _text_item(e, base):
    """TEXT/MTEXT -> anchor, insertion point (base point), alignment and extent as in plan-check's addText/pushText."""
    if e.dxftype() == "MTEXT":
        x, y = _xy(e.dxf.insert)
        text = e.plain_text()
        h, wf, spacing = e.dxf.get("char_height", 2.5), 1.0, e.dxf.get("line_spacing_factor", 1.0)
        ap = min(9, max(1, e.dxf.get("attachment_point", 1))) - 1
        halign = ("left", "center", "right")[ap % 3]
        anchor, basept = (x, y), (x, y)
        rot = math.atan2(*reversed(_xy(e.dxf.text_direction))) if e.dxf.hasattr("text_direction") else math.radians(e.dxf.get("rotation", 0))
    else:
        ins = _xy(e.dxf.insert)
        text = e.dxf.text
        h, wf, spacing = e.dxf.get("height", 2.5), e.dxf.get("width", 1.0), 1.0
        ha, va = e.dxf.get("halign", 0), e.dxf.get("valign", 0)
        if ha == 0 and va == 0:
            anchor, halign = ins, "left"
        else:
            al = _xy(e.dxf.align_point) if e.dxf.hasattr("align_point") else ins
            anchor = ins if ha in (3, 5) else al
            halign = {0: "left", 1: "center", 2: "right", 3: "left", 4: "center", 5: "left"}.get(ha, "left")
        basept = ins
        rot = math.radians(e.dxf.get("rotation", 0))
    lines = text.split("\n")
    width = max(len(s) for s in lines) * h * TEXT_WIDTH_PER_CHAR * wf
    reach = max(width, h * (1 + (len(lines) - 1) * 1.67 * spacing))
    style = e.dxf.get("style", "Standard")
    return dict(base, kind="text", text=text, anchor=anchor, basept=basept, height=h, width_factor=wf, halign=halign,
                rotation=rot, style=style,
                bounds=(anchor[0] - reach, anchor[1] - reach, anchor[0] + reach, anchor[1] + reach))


def _flatten(doc):
    """Model-space entities -> items with kind, resolved layer, colour status and extent; also returns facts for the
    rules (dimensions, Z values, entity types)."""
    items, dims, nonzero_z = [], [], []

    def z_of(e):
        for attr in ("start", "end", "insert", "center", "location", "defpoint", "elevation"):
            if e.dxf.hasattr(attr):
                v = e.dxf.get(attr)
                z = v if isinstance(v, (int, float)) else (v[2] if hasattr(v, "__len__") and len(v) > 2 else 0)
                if abs(z) > 1e-6:
                    return z
        return 0

    def add(e, parent_layer, depth):
        t = e.dxftype()
        layer = parent_layer if (e.dxf.layer == "0" and parent_layer) else e.dxf.layer
        base = {"type": t, "layer": layer, "handle": e.dxf.get("handle", ""), "by_layer": _by_layer(e), "in_block": depth > 0}
        if z_of(e):
            nonzero_z.append((t, layer))
        if t == "LWPOLYLINE":
            pts = [(x, y, b) for x, y, _, _, b in e.get_points("xyseb")]
            width = max([e.dxf.get("const_width", 0)] + [max(s, w) for _, _, s, w, _ in e.get_points("xyseb")])
            items.append(dict(base, kind="poly", verts=[(x, y) for x, y, _ in pts], bulges=[b for *_, b in pts],
                              flag_closed=e.closed, width=width, bounds=_bounds(pts)))
        elif t == "POLYLINE":
            pts = [tuple(_xy(v.dxf.location)) for v in e.vertices]
            items.append(dict(base, type="POLYLINE3D" if e.is_3d_polyline else "POLYLINE2D", kind="poly", verts=pts,
                              bulges=[v.dxf.get("bulge", 0) for v in e.vertices], flag_closed=e.is_closed,
                              width=max([e.dxf.get("default_start_width", 0), e.dxf.get("default_end_width", 0)]),
                              bounds=_bounds(pts)))
        elif t in ("TEXT", "MTEXT", "ATTRIB"):
            items.append(_text_item(e, base))
        elif t == "HATCH":
            paths = []
            for p in e.paths:
                if hasattr(p, "vertices"):
                    paths.append([tuple(_xy(v)) for v in p.vertices])
                else:
                    pts = []
                    for edge in p.edges:
                        for attr in ("start", "end"):
                            if hasattr(edge, attr):
                                pts.append(tuple(_xy(getattr(edge, attr))))
                    paths.append(pts)
            allp = [q for p in paths for q in p]
            items.append(dict(base, kind="hatch", paths=paths, solid=e.dxf.solid_fill == 1 or e.dxf.pattern_name == "SOLID",
                              pattern=e.dxf.pattern_name, bounds=_bounds(allp)))
        elif t == "LINE":
            items.append(dict(base, kind="line", bounds=_bounds([_xy(e.dxf.start), _xy(e.dxf.end)])))
        elif t == "POINT":
            items.append(dict(base, kind="point", bounds=_bounds([_xy(e.dxf.location)])))
        elif t in ("CIRCLE", "ARC"):
            (x, y), r = _xy(e.dxf.center), e.dxf.radius
            items.append(dict(base, kind="circle", bounds=(x - r, y - r, x + r, y + r)))
        elif t in ("SOLID", "TRACE", "3DFACE"):
            pts = [_xy(e.dxf.get(f"vtx{i}")) for i in range(4) if e.dxf.hasattr(f"vtx{i}")]
            items.append(dict(base, kind="solid", bounds=_bounds(pts)))
        elif t in ("INSERT", "DIMENSION"):
            if t == "DIMENSION":
                dims.append(layer)
            try:
                subs = list(e.virtual_entities())
            except Exception:                           # unrenderable block: nothing to expand
                subs = []
            if depth < 16:
                for s in subs:
                    add(s, layer, depth + 1)
        else:
            items.append(dict(base, kind="other", bounds=None))

    for e in doc.modelspace():
        add(e, None, 0)
    return items, dims, nonzero_z


# ---------------------------------------------------------------------------------------------------------------------
# Geometry helpers as in plan-check's js/geometry.js

def shoelace(verts):
    n = len(verts)
    return abs(sum(verts[i][0] * verts[(i + 1) % n][1] - verts[(i + 1) % n][0] * verts[i][1] for i in range(n)) / 2)


def point_in_polygon(px, py, verts):
    inside, j = False, len(verts) - 1
    for i in range(len(verts)):
        (xi, yi), (xj, yj) = verts[i], verts[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _dist_ring(px, py, verts):
    return LineString(list(verts) + [verts[0]]).distance(Point(px, py))


def point_in_polygon_tolerant(px, py, verts, bounds, eps):
    if px < bounds[0] - eps or px > bounds[2] + eps or py < bounds[1] - eps or py > bounds[3] + eps:
        return False
    return point_in_polygon(px, py, verts) or (eps > 0 and _dist_ring(px, py, verts) <= eps)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def has_self_intersection(verts):
    """Proper crossings only, as plan-check: touching or collinear edges (e.g. a zero-width keyhole bridge) do not count."""
    n = len(verts)
    if n < 4:
        return False
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = verts[j], verts[(j + 1) % n]
            d1, d2, d3, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
            if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
                return True
    return False


def canonical_hash(verts):
    pts = [(round(x * 10), round(y * 10)) for x, y in verts]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if not pts:
        return ""
    if sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts))) < 0:
        pts.reverse()
    s = min(range(len(pts)), key=lambda i: pts[i])
    return "|".join(f"{x},{y}" for x, y in pts[s:] + pts[:s])


def region(verts):
    """Closed ring (possibly a keyhole with zero-width bridges) -> the area it encloses (polygonal part only: a
    collapsed bridge becomes a line, which is dropped)."""
    if len(verts) < 3:
        return Polygon()
    parts = [p for p in shapely.get_parts(shapely.make_valid(Polygon(verts))) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.unary_union(parts) if parts else Polygon()


# ---------------------------------------------------------------------------------------------------------------------
# Rules

def check(path):
    """Run all rules on one DXF. Returns {"file", "units", "rules": [{code, severity, status, count, messages}],
    "score": plan-check's score (rules without findings / 40), "summary": {...}}."""
    doc = ezdxf.readfile(path)
    findings = defaultdict(list)
    note = lambda code, msg: findings[code].append(msg)
    insunits = doc.header.get("$INSUNITS")
    unit = INSUNITS_TO_MM.get(insunits, 1)            # mm per drawing unit
    m2 = lambda a: a * unit * unit / 1e6
    items, dims, nonzero_z = _flatten(doc)
    on = defaultdict(list)
    for it in items:
        on[it["layer"]].append(it)
    for it in items:                                   # plan-check: ends within 1 mm count as closed, but flagged
        if it["kind"] == "poly":
            v = it["verts"]
            it["implicit"] = not it["flag_closed"] and len(v) > 2 and math.dist(v[0], v[-1]) <= CLOSE_TOLERANCE_MM / unit
            it["closed"] = it["flag_closed"] or it["implicit"]

    # LAYER
    layers = [layer.dxf.name for layer in doc.layers]
    names, lower = set(layers), {n.lower(): n for n in layers}
    for code, name in REQUIRED.items():
        if name not in names:
            note(code, f"{name} missing" + (f" (found '{lower[name.lower()]}')" if name.lower() in lower else ""))
    cafm_lower = {n.lower(): n for n in CAFM_LAYERS}
    for n in layers:
        if n in CAFM_LAYERS or n in SYSTEM_LAYERS:
            continue
        if cafm_lower.get(n.lower()) in REQUIRED.values():
            continue
        note("LAYER_008", f"unknown layer {n}")

    # POLY and GPOLY
    def polygon_layer(layer, none, wrong, not_closed, arcs, dup, few=None, small=None, crossing=None):
        lay = on[layer]
        polys = [i for i in lay if i["kind"] == "poly"]
        if not polys:
            note(none, f"no polygon on {layer}")
            return
        for i in lay:
            if i["type"] != "LWPOLYLINE":
                note(wrong, f"{i['type']} on {layer} ({i['handle']})")
        seen = set()
        for p in polys:
            if not p["closed"]:
                note(not_closed, f"{p['handle']} not closed")
            elif p["implicit"]:
                note(not_closed, f"{p['handle']}: ends coincide but the closed flag is not set")
            if any(abs(b) > BULGE_EPSILON for b in p["bulges"]):
                note(arcs, f"{p['handle']} has arcs")
            if few and len(p["verts"]) < 3:
                note(few, f"{p['handle']} has {len(p['verts'])} vertices")
            if small and p["closed"] and len(p["verts"]) >= 3 and m2(shoelace(p["verts"])) < MIN_ROOM_AREA_M2:
                note(small, f"{p['handle']}: {m2(shoelace(p['verts'])):.2f} m²")
            if crossing and p["closed"] and len(p["verts"]) >= 4 and has_self_intersection(p["verts"]):
                note(crossing, f"{p['handle']} intersects itself")
            h = canonical_hash(p["verts"])
            if h in seen:
                note(dup, f"{p['handle']} duplicates another polygon")
            seen.add(h)

    polygon_layer(ROOM, "POLY_008", "POLY_006", "POLY_001", "POLY_002", "POLY_005", "POLY_003", "POLY_004", "POLY_007")
    polygon_layer(GF, "GPOLY_004", "GPOLY_003", "GPOLY_001", "GPOLY_002", "GPOLY_005")

    # AOID: texts assigned to the smallest room polygon containing the base point, else the text centre
    closed_on = lambda layer: [i for i in on[layer] if i["kind"] == "poly" and i["type"] in ("LWPOLYLINE", "POLYLINE2D") and i["closed"]]
    rooms = closed_on(ROOM)
    eps = BOUNDARY_TOLERANCE_MM / unit
    for r in rooms:
        r["area_raw"], r["matches"] = shoelace(r["verts"]), []
    smallest = lambda x, y: min((r for r in rooms if point_in_polygon_tolerant(x, y, r["verts"], r["bounds"], eps)),
                                key=lambda r: r["area_raw"], default=None)
    outside = []
    for t in (i for i in on[AOID] if i["kind"] == "text"):
        r = smallest(*t["basept"])
        if r:
            r["matches"].append((t["text"].strip(), True))
            continue
        w = len(t["text"]) * t["height"] * TEXT_WIDTH_PER_CHAR * t["width_factor"]
        along = 0 if t["halign"] == "center" else -w / 2 if t["halign"] == "right" else w / 2
        r = smallest(t["anchor"][0] + math.cos(t["rotation"]) * along, t["anchor"][1] + math.sin(t["rotation"]) * along)
        if r:
            r["matches"].append((t["text"].strip(), False))
        else:
            outside.append(t)
    hashes = Counter()
    for r in rooms:
        valid = [m for m, _ in r["matches"] if AOID_PATTERN.match(m) or PARKING_AOID_PATTERN.match(m)]
        pool = valid or [m for m, _ in r["matches"]]
        r["aoid"] = min(pool, key=len) if pool else ""
        r["dup"] = hashes[canonical_hash(r["verts"])] > 0
        hashes[canonical_hash(r["verts"])] += 1
        if not r["matches"]:
            note("AOID_001", f"room polygon {r['handle']} ({m2(r['area_raw']):.2f} m²) has no AOID")
        if len(r["matches"]) > 1:
            note("AOID_004", f"room polygon {r['handle']}: {len(r['matches'])} texts on R_AOID")
        if r["aoid"] and not (AOID_PATTERN.match(r["aoid"]) or PARKING_AOID_PATTERN.match(r["aoid"])):
            note("AOID_003", f"invalid AOID '{r['aoid']}'")
        for m, inside in r["matches"]:
            if not inside:
                note("AOID_006", f"base point of '{m}' outside room polygon {r['handle']}")
    by_aoid = Counter(r["aoid"] for r in rooms if r["aoid"] and not r["dup"])
    for a, n in by_aoid.items():
        if n > 1:
            note("AOID_002", f"AOID {a} in {n} room polygons")
    for t in outside:
        note("AOID_005", f"'{t['text']}' outside all room polygons")

    # GEOM
    if insunits is not None and insunits != INSUNITS_MM:
        note("GEOM_001", f"$INSUNITS = {insunits}")
    if nonzero_z:
        note("GEOM_002", f"{len(nonzero_z)} elements with Z != 0, e.g. {nonzero_z[:3]}")
    types = Counter(e.dxftype() for e in doc.modelspace())
    for t, n in types.items():
        if t in FORBIDDEN:
            note("GEOM_003", f"{n} x {t}")
    for b in doc.blocks:
        if getattr(b.block, "is_xref", False):
            note("GEOM_004", f"XREF {b.name}")
    frame = plan_frame(on[FRAME], m2)
    if frame:
        tol = FRAME_TOLERANCE_MM / unit
        out = [i for layer, its in on.items() if layer in CAFM_LAYERS and layer != FRAME for i in its
               if i["bounds"] and not (i["bounds"][0] >= frame[0] - tol and i["bounds"][2] <= frame[2] + tol
                                       and i["bounds"][1] >= frame[1] - tol and i["bounds"][3] <= frame[3] + tol)]
        if out:
            note("GEOM_005", f"{len(out)} elements outside the plan frame, e.g. {[(i['type'], i['layer']) for i in out[:3]]}")

    # TEXT
    fonts = {s.dxf.name.lower(): s.dxf.get("font", "") or s.dxf.get("bigfont", "") for s in doc.styles}
    texts = [i for i in items if i["kind"] == "text" and i["type"] != "DIMENSION" and not i["in_block"]]
    for layer, n in Counter(i["layer"] for i in texts if i["layer"] not in TEXT_LAYERS).items():
        note("TEXT_001", f"{n} texts on {layer}")
    for font, n in Counter(fonts.get(i["style"].lower(), "") for i in texts if i["layer"] != FRAME).items():
        if font and not re.search("arial", font, re.I):
            note("TEXT_002", f"{n} texts in font '{font}'")

    # STYLE
    wide = [i for i in items if i["kind"] == "poly" and i["width"] > 0]
    if wide:
        note("STYLE_001", f"{len(wide)} polylines with width, layers {sorted({i['layer'] for i in wide})[:3]}")
    colour = [i for i in items if i["layer"] in CAFM_LAYERS and not i["by_layer"]]
    if colour:
        note("STYLE_002", f"{len(colour)} elements not BYLAYER, e.g. {[(i['type'], i['layer']) for i in colour[:3]]}")

    # LAYOUT, DIM, HATCH
    paper = sum(1 for layout in doc.layouts if layout.name != "Model" for e in layout if e.dxftype() != "VIEWPORT")
    if paper:
        note("LAYOUT_001", f"{paper} entities in paper space")
    if not frame:
        note("LAYOUT_002", "no plan frame on V_PLANLAYOUT")
    if DIMENSION not in dims:
        note("DIM_001", "no DIMENSION on V_BEMASSUNG")
    open_hatches = [i for i in on[HATCH] if i["kind"] == "hatch" and not i["solid"]]
    if open_hatches:
        note("HATCH_001", f"{len(open_hatches)} hatches not SOLID ({sorted({i['pattern'] for i in open_hatches})})")

    # FPX: area validity, nesting in the GF, overlaps, deductions
    regions = [(r["handle"], region(r["verts"]), r) for r in rooms]
    for h, g, r in regions:
        if g.is_empty or g.geom_type != "Polygon" or abs(g.area - r["area_raw"]) > 1e-4 * r["area_raw"]:
            note("FPX_VALID", f"room polygon {h}: enclosed area {m2(g.area):.3f} m² vs shoelace {m2(r['area_raw']):.3f} m²")
    gfs = closed_on(GF)
    if len(gfs) != 1:
        note("FPX_GF_ONE", f"{len(gfs)} closed polygons on {GF}")
    deductions = [region(d["verts"]) for d in closed_on(DEDUCTION)]
    if gfs:
        gf = shapely.unary_union([region(g["verts"]) for g in gfs])
        gf_outline = shapely.unary_union([shapely.Polygon(p.exterior) for p in getattr(gf, "geoms", [gf])])
        tol = AREA_TOLERANCE_M2 * 1e6 / unit / unit
        ded = shapely.unary_union(deductions) if deductions else Polygon()
        for h, g, r in regions:
            stray = g.difference(ded).difference(gf).area
            if stray > max(tol, 0.005 * g.area):
                note("FPX_IN_GF", f"room polygon {h}: {m2(stray):.2f} m² outside the floor polygon")
        for d in deductions:
            if m2(d.area) > GF_DEDUCTION_M2 and d.intersection(gf_outline).area > 0.5 * d.area and d.intersection(gf).area > max(tol, 0.01 * d.area):
                note("FPX_GF_CUT", f"deduction of {m2(d.area):.1f} m² not cut out of the floor polygon")
    tree = shapely.STRtree([g for _, g, _ in regions])
    tol = AREA_TOLERANCE_M2 * 1e6 / unit / unit
    for i, (h, g, _) in enumerate(regions):
        for j in tree.query(g):
            if j > i and g.intersection(regions[j][1]).area > tol:
                note("FPX_OVERLAP", f"room polygons {h} and {regions[j][0]} overlap by {m2(g.intersection(regions[j][1]).area):.2f} m²")

    rules = [{"code": c, "severity": s, "description": d, "status": "fail" if findings[c] else "pass", "count": len(findings[c]),
              "messages": findings[c][:5], "dwg_note": DWG_NOTES.get(c)} for c, s, d in RULES + EXTRA_RULES]
    passed = sum(r["status"] == "pass" for r in rules[:len(RULES)])
    return {"file": str(path), "units": insunits, "rules": rules, "score": round(100 * passed / len(RULES)),
            "summary": {"rooms": len(rooms), "room_area_m2": round(float(m2(sum(r["area_raw"] for r in rooms))), 2),
                        "floor_polygons": len(gfs), "floor_area_m2": round(float(m2(sum(shoelace(g["verts"]) for g in gfs))), 2),
                        "aoid_texts": sum(1 for i in on[AOID] if i["kind"] == "text"), "hatches": len(on[HATCH]),
                        "passed": passed, "rules": len(RULES)}}


def plan_frame(frame_items, m2):
    """plan-check's plan frame: the largest closed polyline (4+ vertices) on V_PLANLAYOUT, else the extent of the layer."""
    polys = [i for i in frame_items if i["kind"] == "poly" and i["closed"] and len(i["verts"]) >= 4]
    if polys:
        return max(polys, key=lambda i: shoelace(i["verts"]))["bounds"]
    b = [i["bounds"] for i in frame_items if i["bounds"]]
    return (min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b)) if b else None


def report(result):
    """Plain-text table of one check result."""
    lines = [f"{result['file']}: plan-check score {result['score']}% ({result['summary']['passed']}/{result['summary']['rules']} rules "
             f"without findings) | {result['summary']}"]
    for r in result["rules"]:
        msg = "; ".join(r["messages"][:2])
        lines.append(f"  {r['code']:10s} {r['severity']:7s} {r['status'].upper():4s} {r['count']:3d}  {msg}"[:220])
    return "\n".join(lines)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--json"]
    results = [check(Path(a)) for a in args]
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1, ensure_ascii=False))
    else:
        print("\n\n".join(report(r) for r in results))
