"""Stage 10, export: JSON with every element in plan metres, and a DXF on the layers of BBL's CAD-Richtlinie.

The DXF follows the CAD-Richtlinie BBL V1.0 as plan-check (github.com/bbl-dres/plan-check) checks it; fpx.conformance
runs the same rules locally, since plan-check reads DWG only:
- drawing unit millimetres ($INSUNITS 4), model space only, Z = 0, every colour BYLAYER, polyline width 0, every text
  in the ARIAL text style (font arial.ttf), layers with the guideline's colours;
- R_RAUMPOLYGON: one closed LWPOLYLINE per room of 0.25 m² or more. Islands (columns) are cut out through zero-width
  bridges, so that one polyline carries the room's area; rooms split along a diagonal stair outline overlap by up to a
  pixel, so the polygons are made disjoint (disjoint());
- R_AOID: only AOIDs read on the sheet, one per room and unique, base point inside the room (attributes.aoid_export).
  Never generated: rooms without one are flagged in the QA list, to be matched from SAP;
- V_TEXT: room names;
- R_GESCHOSSPOLYGON: one closed LWPOLYLINE per floor with the voids over 5 m² cut out through a zero-width bridge (one
  continuous polyline, Kap. 5.9); R_RAUMPOLYGON-ABZUG: those voids, which plan-check deducts from the room containing them;
- A_ARCHITEKTUR: wall outlines (outer and inner rings), openings, stairs. A_SCHRAFFUR: SOLID hatches of the walls
  classed massive, a heuristic (wall_construction);
- V_BEMASSUNG: overall dimensions of the floor polygon (measured on the extracted outline, not read from the sheet);
- V_PLANLAYOUT: a plan frame around everything drawn, and a title block.
"""
import json

import cv2
import ezdxf
import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, polylabel

from .config import DEFAULT
from .geometry import mask_polys, to_plan

MM = 1000.0                         # plan metres -> drawing units (millimetres, CAD-Richtlinie Kap. 4.2)
STYLE, FONT = "ARIAL", "arial.ttf"  # text style for every text (Kap. 5.5); plan-check reads the font file of the style
TEXT_HEIGHT = 0.25                  # m: room names and AOIDs (2.5 mm at 1:100)
TITLE_HEIGHT = 0.3                  # m: title block lines
DIM_OFFSET = 1.0                    # m: overall dimension lines this far outside the floor polygon
FRAME_MARGIN = 0.5                  # m: plan frame this far around everything drawn
GLYPH = 0.6                         # glyph width / text height, the estimate plan-check uses for text extents
# CAD-Richtlinie BBL V1.0, Kap. 5.2, Tabelle 4: ACI colour of each layer the export uses (the seven required ones first)
LAYERS = {"R_RAUMPOLYGON": 210, "R_AOID": 7, "R_GESCHOSSPOLYGON": 214, "A_ARCHITEKTUR": 253, "A_SCHRAFFUR": 8,
          "V_BEMASSUNG": 251, "V_PLANLAYOUT": 252, "R_RAUMPOLYGON-ABZUG": 230, "V_TEXT": 253}
CONVENTIONS = {
    "units": "mm ($INSUNITS 4); JSON coordinates stay in plan metres",
    "R_RAUMPOLYGON": "one closed LWPOLYLINE per room of 0.25 m² or more, islands cut out through zero-width bridges; "
                     "made disjoint where rooms split along a stair outline overlap by a pixel; XDATA 'FPX' = room id",
    "R_AOID": "only AOIDs read on the sheet, unique and one per room, base point inside the room; never generated",
    "V_TEXT": "room names",
    "R_GESCHOSSPOLYGON": "one closed LWPOLYLINE per floor, voids over 5 m² cut out through a zero-width bridge",
    "R_RAUMPOLYGON-ABZUG": "voids over 5 m² (deducted from the GF and from the room containing them)",
    "A_ARCHITEKTUR": "wall outlines (all rings), openings, stairs",
    "A_SCHRAFFUR": "SOLID hatches of walls classed massive by a heuristic (see wall_construction)",
    "V_BEMASSUNG": "overall dimensions of the floor polygon, measured on the extracted outline",
    "V_PLANLAYOUT": "plan frame around everything drawn, title block",
    "text style": f"{STYLE} ({FONT}) for every text",
}


# ---------------------------------------------------------------------------------------------------------------------
# Geometry for the DXF

def _polygon(g):
    """Largest polygon of a geometry (differences and precision snapping can leave slivers or collections)."""
    parts = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon" and not p.is_empty]
    return max(parts, key=lambda p: p.area) if parts else Polygon()


def _on_ring(ring, p):
    """Index of point p in a ring (list of xy without the closing repeat), inserted on its nearest edge if needed."""
    for k, c in enumerate(ring):
        if abs(c[0] - p[0]) < 1e-9 and abs(c[1] - p[1]) < 1e-9:
            return ring, k
    n = len(ring)
    k = min(range(n), key=lambda k: LineString([ring[k], ring[(k + 1) % n]]).distance(Point(p)))
    return ring[:k + 1] + [tuple(p)] + ring[k + 1:], k + 1


def keyhole(poly):
    """Polygon with holes -> one closed ring that cuts every hole out through a zero-width bridge (a keyhole), the
    "durchgängiger Polygonzug" of the CAD-Richtlinie. The outline runs counter-clockwise and the holes clockwise, so the
    ring's shoelace area is the polygon's area. Holes are joined nearest first, each along the shortest segment to the
    outline built so far, so no bridge crosses another hole."""
    poly = shapely.geometry.polygon.orient(poly, 1.0)
    ring = [tuple(c) for c in poly.exterior.coords[:-1]]
    holes = [[tuple(c) for c in h.coords[:-1]] for h in poly.interiors]
    while holes:
        outline = LineString(ring + [ring[0]])
        k = min(range(len(holes)), key=lambda i: LineString(holes[i] + [holes[i][0]]).distance(outline))
        hole = holes.pop(k)
        b, a = nearest_points(LineString(hole + [hole[0]]), outline)
        ring, i = _on_ring(ring, a.coords[0])
        hole, j = _on_ring(hole, b.coords[0])
        hole = hole[j:] + hole[:j]
        ring = ring[:i + 1] + hole + [hole[0]] + ring[i:]
    out = [c for k, c in enumerate(ring) if k == 0 or c != ring[k - 1]]
    return out[:-1] if len(out) > 1 and out[-1] == out[0] else out


def disjoint(polys):
    """Polygons made disjoint for the DXF: where two overlap, the overlap stays with the earlier one. Rooms split along
    a diagonal stair outline overlap by up to a pixel along it; the JSON keeps the traced polygons."""
    out = []
    for p in polys:
        q = p
        for o in out:
            if q.intersects(o) and q.intersection(o).area > 0:
                q = _polygon(q.difference(o))
        out.append(q if q is p else _polygon(shapely.set_precision(q, 0.001)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Massive or lightweight walls (a heuristic)

def _construction(t, fill, cfg):
    if t >= cfg.massive_wall_min_thickness:
        return "massive", f"heuristic: {t:.2f} m thick (massive from {cfg.massive_wall_min_thickness} m)"
    if fill is not None and fill >= cfg.solid_fill_share and t >= cfg.massive_solid_min_thickness:
        return "massive", f"heuristic: drawn solid ({fill:.0%} ink in the wall core) and {t:.2f} m thick"
    seen = "core too thin to see a fill" if fill is None else f"{fill:.0%} ink in the wall core, not drawn solid"
    return "lightweight", f"heuristic: {t:.2f} m thick (massive from {cfg.massive_wall_min_thickness} m), {seen}"


def wall_construction(sheet, cfg=DEFAULT):
    """Massive or lightweight per wall segment. A heuristic: the guideline hatches massive walls only (A_SCHRAFFUR),
    and a raster plan rarely says what a wall is made of.

    Every wall pixel belongs to the nearest centre line in its wall piece, so each segment owns part of the wall mask.
    A segment is massive if (1) it is at least massive_wall_min_thickness thick (thickness from the wall graph), or (2)
    its core, the pixels at least two pixels inside the wall faces, is drawn solid (solid_fill_share or more dark ink)
    and it is at least massive_solid_min_thickness thick: on plans that fill walls, poché marks masonry and concrete.
    Otherwise lightweight. Wall pieces without a centre line use their largest inscribed width.

    Sets construction, construction_basis and fill_share on each wall segment; returns the massive wall polygons
    (working pixels) and a summary."""
    wall = sheet.wall_mask
    segs = {s["id"]: s for s in sheet.wall_segments}
    if wall is None or not wall.any():
        return [], {"method": "heuristic", "massive_segments": 0, "lightweight_segments": 0}
    dark = cv2.cvtColor(sheet.img, cv2.COLOR_RGB2GRAY) < 128
    depth = ndimage.distance_transform_edt(wall)
    core = depth >= 2.5
    skel, order = np.zeros(wall.shape, np.int32), []
    for _, _, d in (sheet.wall_graph.edges(data=True) if sheet.wall_graph is not None else []):
        if d.get("id") in segs:
            order.append(segs[d["id"]])
            p = np.clip(np.round(d["path"]).astype(int), 0, [wall.shape[1] - 1, wall.shape[0] - 1])
            skel[p[:, 1], p[:, 0]] = len(order)
    skel[~wall] = 0
    _, comp = cv2.connectedComponents(wall.astype(np.uint8), connectivity=8)
    owner, massive, loose = np.zeros(wall.shape, np.int32), np.zeros(wall.shape, bool), 0
    for c, sl in enumerate(ndimage.find_objects(comp), 1):
        if sl is None:
            continue
        mc = comp[sl] == c
        sk = np.where(mc, skel[sl], 0)
        if sk.any():
            _, (iy, ix) = ndimage.distance_transform_edt(sk == 0, return_indices=True)
            owner[sl][mc] = sk[iy, ix][mc]
        else:                                          # a wall piece without centre line: classified on its own
            cm = core[sl] & mc
            fill = float(dark[sl][cm].mean()) if cm.sum() >= 10 else None
            if _construction(2 * float(depth[sl][mc].max()) / cfg.m, fill, cfg)[0] == "massive":
                massive[sl] |= mc
                loose += 1
    n = len(order) + 1
    core_n = np.bincount(owner[core], minlength=n)
    core_dark = np.bincount(owner[core], weights=dark[core], minlength=n)
    heavy = np.zeros(n, bool)
    for k, s in enumerate(order, 1):
        fill = float(core_dark[k] / core_n[k]) if core_n[k] >= 10 else None
        s["construction"], s["construction_basis"] = _construction(s["thickness"], fill, cfg)
        s["fill_share"] = None if fill is None else round(fill, 2)
        heavy[k] = s["construction"] == "massive"
    massive |= heavy[owner] & wall
    polys = [p.simplify(cfg.px(cfg.poly_simplify)) for p in mask_polys(massive, cfg.px2(cfg.wall_min_area))]
    summary = {"method": "heuristic", "rule": f"massive if at least {cfg.massive_wall_min_thickness} m thick, or drawn solid "
                                             f"(>= {cfg.solid_fill_share:.0%} ink in the wall core) and at least "
                                             f"{cfg.massive_solid_min_thickness} m thick; otherwise lightweight",
               "massive_segments": int(heavy.sum()), "lightweight_segments": len(order) - int(heavy.sum()),
               "massive_pieces_without_centre_line": loose,
               "massive_area": round(cfg.m2(float(massive.sum())), 2), "wall_area": round(cfg.m2(float(wall.sum())), 2),
               "note": "thickness and fill are measured on the sheet; the wall material is not known. Check before use."}
    return polys, summary


# ---------------------------------------------------------------------------------------------------------------------
# JSON

def to_json(sheet, cfg=DEFAULT, walls=None):
    walls = walls or wall_construction(sheet, cfg)
    P = lambda g: to_plan(sheet, g)
    rnd = lambda g: shapely.set_precision(P(g), 0.001)
    geo = lambda g: shapely.geometry.mapping(rnd(g))
    return {
        "sheet": {"id": sheet.id, "title": sheet.title, "source": sheet.source, "input_class": sheet.input_class,
                  "scale": sheet.scale, "triage": sheet.meta.get("triage"), "deskew": sheet.meta.get("deskew"),
                  "extent": [round(v, 3) for v in (*sheet.to_plan([0, sheet.img.shape[0]])[0], *sheet.to_plan([sheet.img.shape[1], 0])[0])]},
        "text": [{"text": t["text"], "role": t["role"], "source": t["source"], "conf": round(t["conf"], 3),
                  "box": geo(shapely.box(*t["box"]))} for t in sheet.text],
        "walls": [{"id": s["id"], "centre_line": geo(s["line"]), "thickness": s["thickness"],
                   "construction": s.get("construction"), "construction_basis": s.get("construction_basis"),
                   "fill_share": s.get("fill_share")} for s in sheet.wall_segments],
        "wall_polygons": [geo(p) for p in sheet.wall_polys],
        "wall_construction": walls[1],
        "massive_wall_polygons": [geo(p) for p in walls[0]],
        "openings": [{"id": o["id"], "kind": o["kind"], "host": o["host"], "exterior": o["exterior"],
                      "width": round(o["width"], 2), "connects": o.get("connects"), "source": o["source"],
                      "geometry": geo(o["poly"])} for o in sheet.openings],
        "stairs": [geo(p) for p in sheet.stairs],
        "voids": [{"label": v["label"], "area": round(cfg.m2(v["poly"].area), 2), "gf_deducted": v.get("gf_deducted", False),
                   "geometry": geo(v["poly"])} for v in sheet.voids],
        # small unlabelled regions: kept as rooms since the CAD-Richtlinie needs every room of 0.25 m² or more; listed
        # here as well, as before, so viewers can highlight them
        "fragments": [geo(r["poly"]) for r in sheet.rooms if r.get("small_region")],
        "rooms": [{"id": r["id"], "name": r["name"], "aoid": r["aoid"], "usage": r["usage"], "area_stamp": r["area_stamp"],
                   "area": round(P(r["poly"]).area - sum(P(v["poly"]).area for v in r.get("voids", [])), 2),
                   "area_gross": round(P(r["poly"]).area, 2), "name_raw": " / ".join(r.get("names_raw", [])) or None,
                   "confidence": r["confidence"], "reasons": r["reasons"],
                   "small_region": bool(r.get("small_region")), "review": r.get("review"),
                   "aoid_export": r.get("aoid_export"),
                   "neighbours": sorted(sheet.connectivity.neighbors(r["id"])) if r["id"] in sheet.connectivity else [],
                   "geometry": geo(r["poly"])} for r in sheet.rooms],
        "floor": {"gf": geo(sheet.gf), "gf_area": round(sheet.gf_area, 2),
                  "ebf_proposal": sheet.ebf_proposal, "zones": "not applicable (no zone information on the sheet)"},
        "connectivity": [{"a": a, "b": b, "opening": d["opening"]} for a, b, d in sheet.connectivity.edges(data=True)],
        "qa": sheet.qa,
        "export": {"dxf": CONVENTIONS, "guideline": "CAD-Richtlinie BBL V1.0 as checked by plan-check; local check: fpx.conformance"},
        # stage 1b (fpx.layout): the sheet's regions and this drawing, in sheet raster pixels and paper millimetres;
        # empty when the sheet was loaded whole (pipeline.run)
        "regions": getattr(sheet, "regions", None) or [],
        "drawings": getattr(sheet, "drawings", None) or [],
    }


# ---------------------------------------------------------------------------------------------------------------------
# DXF

def _new_doc():
    doc = ezdxf.new("R2018", setup=False, units=4)             # $INSUNITS 4: millimetres
    for name, colour in LAYERS.items():
        doc.layers.add(name, color=colour)
    doc.styles.add(STYLE, font=FONT)
    doc.header["$TEXTSTYLE"] = STYLE
    ds = doc.dimstyles.new("FPX")
    for key, value in {"dimtxsty": STYLE, "dimtxt": TEXT_HEIGHT * MM, "dimasz": 0.15 * MM, "dimtsz": 0.1 * MM,
                       "dimexo": 0.1 * MM, "dimexe": 0.1 * MM, "dimgap": 0.05 * MM, "dimtad": 1, "dimlfac": 1 / MM,
                       "dimdec": 2, "dimclrd": 256, "dimclre": 256, "dimclrt": 256}.items():
        ds.dxf.set(key, value)                                 # dimension shown in metres, colours BYLAYER
    doc.header["$DIMSTYLE"] = "FPX"
    doc.appids.add("FPX")
    return doc


def _mm(ring):
    return [(x * MM, y * MM) for x, y in ring]


def _text(msp, text, xy, layer, height=TEXT_HEIGHT):
    msp.add_text(text, height=height * MM, dxfattribs={"layer": layer, "style": STYLE}).set_placement((xy[0] * MM, xy[1] * MM))
    reach = max(len(text) * height * GLYPH, height)            # plan-check's text extent around the base point
    return (xy[0] - reach, xy[1] - reach, xy[0] + reach, xy[1] + reach)


def _room_texts(msp, room, p):
    """AOID (R_AOID) and name (V_TEXT) centred on the room's pole of inaccessibility, the AOID above the name; a base
    point that would fall outside the room moves to the pole itself, which is inside."""
    c = polylabel(p, tolerance=0.01)
    aoid = room["aoid_export"]["aoid"] if (room.get("aoid_export") or {}).get("written") else None
    lines = [(t, layer) for t, layer in ((aoid, "R_AOID"), (room.get("name"), "V_TEXT")) if t]
    boxes = []
    for k, (t, layer) in enumerate(lines):
        dy = (len(lines) / 2 - k - 1) * 1.4 * TEXT_HEIGHT + 0.2 * TEXT_HEIGHT
        q = (c.x - len(t) * TEXT_HEIGHT * GLYPH / 2, c.y + dy)
        boxes.append(_text(msp, t, q if p.contains(Point(q)) else (c.x, c.y), layer))
    return boxes


def _dimensions(doc, msp, gf):
    """Overall dimensions (SIA 400 Hauptmasse) of the floor polygon: its width below it, its depth to its left."""
    pts = np.asarray(gf.exterior.coords)
    left, right = pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()]
    low, high = pts[pts[:, 1].argmin()], pts[pts[:, 1].argmax()]
    x0, y0, _, _ = gf.bounds
    for p1, p2, base, angle in ((left, right, (x0, y0 - DIM_OFFSET), 0), (low, high, (x0 - DIM_OFFSET, y0), 90)):
        msp.add_linear_dim(base=(base[0] * MM, base[1] * MM), p1=(p1[0] * MM, p1[1] * MM), p2=(p2[0] * MM, p2[1] * MM),
                           angle=angle, dimstyle="FPX", dxfattribs={"layer": "V_BEMASSUNG"}).render()
    for block in doc.blocks:                                   # arrow and tick blocks: BYBLOCK -> BYLAYER (STYLE_002)
        if block.name.startswith("_") or block.name.startswith("*D"):
            for e in block:
                e.dxf.color = 256
    reach = DIM_OFFSET + 1.0
    b = gf.bounds
    return (b[0] - reach, b[1] - reach, b[2], b[3])


def _rings(p):
    return [] if p.is_empty else [list(p.exterior.coords)[:-1]] + [list(h.coords)[:-1] for h in p.interiors]


def to_dxf(sheet, path, cfg=DEFAULT, walls=None):
    walls = walls or wall_construction(sheet, cfg)
    P = lambda g: _polygon(shapely.set_precision(to_plan(sheet, g), 0.001))
    doc = _new_doc()
    msp = doc.modelspace()
    boxes = []                                                 # extents of everything drawn, for the plan frame

    def add_ring(ring, layer):
        if len(ring) < 3:
            return None
        boxes.append(shapely.MultiPoint(ring).bounds)
        return msp.add_lwpolyline(_mm(ring), close=True, dxfattribs={"layer": layer})

    rooms = disjoint([P(r["poly"]) for r in sheet.rooms])
    for r, p in zip(sheet.rooms, rooms):
        if p.is_empty:
            continue
        add_ring(keyhole(p), "R_RAUMPOLYGON").set_xdata("FPX", [(1000, r["id"])])
        boxes += _room_texts(msp, r, p)
    gf = P(sheet.gf) if not sheet.gf.is_empty else Polygon()
    if not gf.is_empty:
        add_ring(keyhole(gf), "R_GESCHOSSPOLYGON")
        boxes.append(_dimensions(doc, msp, gf))
    for v in sheet.voids:
        if v.get("gf_deducted") and not P(v["poly"]).is_empty:
            add_ring(keyhole(P(v["poly"])), "R_RAUMPOLYGON-ABZUG")
    for w in sheet.wall_polys:
        for ring in _rings(P(w)):
            add_ring(ring, "A_ARCHITEKTUR")
    for w in walls[0]:
        p = P(w)
        if p.is_empty:
            continue
        h = msp.add_hatch(color=256, dxfattribs={"layer": "A_SCHRAFFUR"})          # SOLID fill, colour BYLAYER
        h.paths.add_polyline_path(_mm(p.exterior.coords[:-1]), is_closed=True, flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL)
        for hole in p.interiors:
            h.paths.add_polyline_path(_mm(hole.coords[:-1]), is_closed=True, flags=ezdxf.const.BOUNDARY_PATH_DEFAULT)
    for g in [o["poly"] for o in sheet.openings] + list(sheet.stairs):
        for ring in _rings(P(g)):
            add_ring(ring, "A_ARCHITEKTUR")
    # plan frame around the sheet and everything drawn, with the title block below the drawing
    x0, y0, x1, y1 = (*sheet.to_plan([0, sheet.img.shape[0]])[0], *sheet.to_plan([sheet.img.shape[1], 0])[0])
    for b in boxes:
        x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
    title = [sheet.title, f"Sheet {sheet.id} | source {sheet.source} | scale {sheet.scale.get('value')}",
             "Extracted automatically (fpx, pilot v2): unverified, check against the sheet",
             "Units mm | R_AOID only where read on the sheet | A_SCHRAFFUR: massive walls by heuristic"]
    x0, y0, x1, y1 = x0 - FRAME_MARGIN, y0 - FRAME_MARGIN - 1.6 * TITLE_HEIGHT * len(title), x1 + FRAME_MARGIN, y1 + FRAME_MARGIN
    x1 = max(x1, x0 + 2 * FRAME_MARGIN + max(len(line) for line in title) * TITLE_HEIGHT * GLYPH)
    msp.add_lwpolyline(_mm([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), close=True, dxfattribs={"layer": "V_PLANLAYOUT"})
    for k, line in enumerate(title):
        _text(msp, line, (x0 + FRAME_MARGIN, y0 + FRAME_MARGIN / 2 + (len(title) - 1 - k) * 1.6 * TITLE_HEIGHT), "V_PLANLAYOUT", TITLE_HEIGHT)
    doc.saveas(path)
    return doc


def export(sheet, out_dir, cfg=DEFAULT):
    walls = wall_construction(sheet, cfg)
    data = to_json(sheet, cfg, walls)
    (out_dir / f"{sheet.id}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    to_dxf(sheet, out_dir / f"{sheet.id}.dxf", cfg, walls)
    return data
