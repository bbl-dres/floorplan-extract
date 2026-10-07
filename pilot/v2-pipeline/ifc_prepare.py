"""IFC models -> data/floors-ifc.pkl: one record per storey, in the same format as sd_prepare.py, for the renderer.

    python ifc_prepare.py            # all permissively licensed IFC-Bench models in data/public/ifc-bench

Plan cut at 1.1 m above the storey's floor level (the lowest wall base): walls, columns, railings and curtain walls
from the section; doors and windows from the openings they fill (their footprint in the wall); stairs from the
projected flights; rooms from projected IfcSpace geometry with their real names (rendered as room stamps);
furniture as clutter. Only models whose licence allows training (CC BY 3.0/4.0, MIT) are listed in MODELS.
"""
import json
import pickle
import re
import zlib

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as ue
import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import Polygon

from common import DATA, REPO
from sd_prepare import dominant_angle

ROOT = REPO / "data/public/ifc-bench/projects"
MODELS = {   # project: (licence, architecture file, structure file or None)
    "ac20": ("CC BY 4.0", "arc.ifc", None), "fzk_house": ("CC BY 4.0", "arc.ifc", None), "smiley_west": ("CC BY 4.0", "arc.ifc", None),
    "dental_clinic": ("CC BY 4.0", "arc.ifc", "str.ifc"), "duplex": ("CC BY 4.0", "arc.ifc", None),
    "wbdg_office": ("CC BY 4.0", "arc.ifc", "str.ifc"), "molio": ("CC BY 4.0", "arc.ifc", None),
    "sixty5": ("CC BY 4.0", "arc.ifc", "str.ifc"), "city_house_munich": ("CC BY 4.0", "arc.ifc", None),
    "digital_hub": ("MIT", "arc.ifc", None), "fantasy_hotel_1": ("MIT", "arc.ifc", None), "fantasy_hotel_2": ("MIT", "arc.ifc", None),
    "fantasy_office_building_1": ("MIT", "arc.ifc", None), "fantasy_office_building_2": ("MIT", "arc.ifc", None),
    "fantasy_office_building_3": ("MIT", "arc.ifc", None), "fantasy_residential_building_1": ("MIT", "arc.ifc", None),
    "west_riverside_hospital": ("CC BY 3.0", "arc_ifc4.ifc", "str_ifc4.ifc"),
}
CUT = 1.1
SETTINGS = ifcopenshell.geom.settings()
SETTINGS.set("use-world-coords", True)


def meshes(model, types):
    """Element id -> trimesh for all elements of the given types (world coordinates, metres)."""
    elems = []
    for t in types:
        try:
            elems += model.by_type(t)
        except RuntimeError:                               # type not in this schema (e.g. IfcFurniture in IFC2X3)
            pass
    out = {}
    if not elems:
        return out
    it = ifcopenshell.geom.iterator(SETTINGS, model, 8, include=elems)
    if it.initialize():
        while True:
            sh = it.get()
            v = np.asarray(sh.geometry.verts, float).reshape(-1, 3)
            f = np.asarray(sh.geometry.faces, int).reshape(-1, 3)
            if len(f):
                out[sh.id] = trimesh.Trimesh(v, f, process=False)
            if not it.next():
                break
    return out


def section(m, z):
    lines = trimesh.intersections.mesh_plane(m, [0, 0, 1], [0, 0, z])
    if not len(lines):
        return None
    polys = list(shapely.polygonize([shapely.LineString(l[:, :2]) for l in lines]).geoms)
    g = shapely.union_all([p for p in polys if p.area > 1e-4]) if polys else None
    return g if g is not None and not g.is_empty else None


def footprint(m):
    tri = m.vertices[m.faces][:, :, :2]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    keep = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-8
    polys = [Polygon(t) for t in tri[keep]]
    return shapely.union_all(polys).buffer(0.005).buffer(-0.005) if polys else None


def storey_of(e):
    s = ue.get_container(e) or ue.get_aggregate(e)
    while s is not None and not s.is_a("IfcBuildingStorey"):
        s = ue.get_aggregate(s) or ue.get_container(s)
    return s


def convert(project, licence, arc, strf):
    model = ifcopenshell.open(ROOT / project / arc)
    smodel = ifcopenshell.open(ROOT / project / strf) if strf else None
    elem = {}
    for key, types in {"walls": ["IfcWall"], "curtain": ["IfcCurtainWall"], "columns": ["IfcColumn"], "railings": ["IfcRailing"],
                       "doors": ["IfcDoor"], "windows": ["IfcWindow"], "stairs": ["IfcStair", "IfcStairFlight"],
                       "spaces": ["IfcSpace"], "furniture": ["IfcFurnishingElement", "IfcFurniture"]}.items():
        ms = meshes(model, types)
        elem[key] = [(model.by_id(i), m) for i, m in ms.items()]
    if smodel is not None and not elem["columns"]:
        elem["columns"] = [(smodel.by_id(i), m) for i, m in meshes(smodel, ["IfcColumn"]).items()]
    openings = {}
    for kind in ("doors", "windows"):
        for e, m in elem[kind]:
            fills = getattr(e, "FillsVoids", None) or []
            op = fills[0].RelatingOpeningElement if fills else None
            openings[e.id()] = op
    op_meshes = meshes(model, ["IfcOpeningElement"])
    records = []
    storeys = {}
    for key, items in elem.items():
        for e, m in items:
            s = storey_of(e)
            storeys.setdefault(s.id() if s is not None else None, {}).setdefault(key, []).append((e, m))
    for sid, groups in storeys.items():
        walls = groups.get("walls", [])
        if sid is None or len(walls) < 4:
            continue
        z0 = float(np.median([m.bounds[0][2] for _, m in walls])) + CUT
        cut = lambda items: [g for g in (section(m, z0) for _, m in items) if g is not None]
        in_band = lambda m: m.bounds[0][2] < z0 < m.bounds[1][2]
        wall_g = shapely.union_all(cut(walls))
        if wall_g.is_empty or wall_g.area < 3:
            continue
        def opening_poly(e, m):
            op = openings.get(e.id())
            om = op_meshes.get(op.id()) if op is not None else None
            g = footprint(om) if om is not None else footprint(m)
            return g.intersection(wall_g.buffer(0.05)) if g is not None else None
        doors = [opening_poly(e, m) for e, m in groups.get("doors", []) if in_band(m)]
        windows = [opening_poly(e, m) for e, m in groups.get("windows", []) if in_band(m)]
        windows += cut(groups.get("curtain", []))
        cols = cut([(e, m) for e, m in groups.get("columns", []) if in_band(m)])
        if smodel is not None and not groups.get("columns"):
            cols += [g for g in (section(m, z0) for e, m in elem["columns"] if in_band(m)) if g is not None]
        rails = cut(groups.get("railings", []))
        stairs = [g for g in (footprint(m) for _, m in groups.get("stairs", [])) if g is not None and g.area > 0.5]
        spaces = []
        for e, m in groups.get("spaces", []):
            g = footprint(m)
            name = (e.LongName or e.Name or "").strip()
            if g is not None and g.area > 0.5:
                spaces.append(("NAME:" + name if name and not re.fullmatch(r"[\d\s.]+", name) else "ROOM", g))
        furniture = [("FURNITURE", g) for g in (footprint(m) for e, m in groups.get("furniture", []) if in_band(m) or m.bounds[0][2] < z0)
                     if g is not None and 0.1 < g.area < 20]
        clean = lambda gs: shapely.union_all([g for g in gs if g is not None and not g.is_empty]) if gs else shapely.Polygon()
        rec = {"walls": wall_g, "doors": clean(doors), "windows": clean(windows), "columns": clean(cols), "railings": clean(rails),
               "stairs": stairs, "features": furniture, "areas": spaces}
        ang = dominant_angle(rec["walls"])
        x0, y0, *_ = shapely.total_bounds(affinity.rotate(rec["walls"], -ang, origin=(0, 0)))
        tf = lambda p: affinity.translate(affinity.rotate(p, -ang, origin=(0, 0)), -x0, -y0)
        out = {k: tf(rec[k]) for k in ("walls", "doors", "windows", "columns", "railings")}
        out["stairs"] = [tf(p) for p in rec["stairs"]]
        out["features"] = [(k, tf(p)) for k, p in rec["features"]]
        out["areas"] = [(k, tf(p)) for k, p in rec["areas"]]
        storey = model.by_id(sid)
        out.update(rotation=ang, floor_id=f"{project}/{storey.Name or sid}", site=project, public=1.0, licence=licence)
        records.append(out)
        print(f"  {project:30s} {str(storey.Name)[:20]:20s} walls {wall_g.area:7.1f} m² doors {len(doors):3d} windows {len(windows):3d} "
              f"columns {len(cols):3d} stairs {len(stairs):2d} rooms {len(spaces):3d} ({sum(k.startswith('NAME:') for k, _ in spaces)} named) furniture {len(furniture)}", flush=True)
    return records


if __name__ == "__main__":
    import sys
    only = set(sys.argv[1:])
    allrecs, summary = [], {}
    for project, (licence, arc, strf) in MODELS.items():
        if only and project not in only:
            continue
        try:
            recs = convert(project, licence, arc, strf)
        except Exception as ex:
            print(f"FAIL {project}: {type(ex).__name__}: {ex}", flush=True)
            continue
        summary[project] = {"licence": licence, "storeys": len(recs)}
        allrecs += [zlib.compress(pickle.dumps(r, protocol=5)) for r in recs]
    (DATA / "floors-ifc.pkl").write_bytes(pickle.dumps(allrecs, protocol=5))
    (DATA / "floors-ifc.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(len(allrecs), "storeys from", len(summary), "models")
