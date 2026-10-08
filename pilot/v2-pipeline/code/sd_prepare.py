"""Swiss Dwellings -> data/floors-{train,val,test}.pkl: one floor per plan, grouped by element class.

    python sd_prepare.py

Each floor is a FloorRecord (below): rotated so that its dominant wall direction is axis-aligned (archive plans are
drawn that way) and shifted to the origin by normalise_record(), which ifc_prepare.py shares. The split is by site,
so no building appears in two splits.
"""
import pickle
import zlib
from typing import NotRequired, TypedDict

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from common import DATA, SD_CSV

FEATURES = {"KITCHEN", "SINK", "TOILET", "BATHTUB", "SHOWER", "ELEVATOR", "BUILT_IN_FURNITURE", "WASHING_MACHINE", "RAMP"}
GEOMETRY_KEYS = ("walls", "doors", "windows", "columns", "railings")


class FloorRecord(TypedDict, total=False):
    """One floor as the renderer (synth.py), the oracle and the harness read it: geometry in plan metres (y up), the
    floor rotated by `rotation` degrees so that its dominant wall direction is axis-aligned and shifted so that the
    bounding box of the walls starts at the origin. Written by sd_prepare.py (Swiss Dwellings) and ifc_prepare.py
    (IFC storeys), one record per floor, pickled and zlib-compressed into a list."""
    walls: BaseGeometry                 # union of all wall polygons (Polygon or MultiPolygon)
    walls_parts: list                   # [Polygon]: the walls one by one as drawn, for per-wall axes (centre-line labels later)
    doors: BaseGeometry                 # union of the door and entrance-door polygons: the opening in the wall
    windows: BaseGeometry
    columns: BaseGeometry
    railings: BaseGeometry
    stairs: list                        # [Polygon], one per flight
    features: list                      # [(kind, Polygon)]: fixtures (FEATURES), or ("FURNITURE", Polygon) from IFC
    areas: list                         # [(kind, Polygon)]: rooms by Swiss Dwellings area type; IFC: "NAME:<room name>" or "ROOM"
    rotation: float                     # degrees the floor was rotated by (counter-clockwise), for the record
    floor_id: str
    site: str                           # Swiss Dwellings site (the split is by site), or the IFC project
    public: float                       # share of the floor's units with PUBLIC usage (0-1); 1.0 for IFC storeys
    licence: NotRequired[str]           # IFC storeys only (the renderer takes this key as the mark of an IFC storey)


def dominant_angle(walls):
    """Length-weighted mode of wall edge directions modulo 90 degrees."""
    coords = [np.asarray(r.coords) for p in getattr(walls, "geoms", [walls]) for r in [p.exterior, *p.interiors]]
    seg = np.concatenate([c[1:] - c[:-1] for c in coords])
    length = np.hypot(seg[:, 0], seg[:, 1])
    ang = np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) % 90
    hist, edges = np.histogram(ang, bins=180, range=(0, 90), weights=length)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)                 # smooth, wrapping at 0/90
    return float(edges[np.argmax(hist)] + 0.25)


def normalise_record(rec):
    """A FloorRecord rotated so that its dominant wall direction is axis-aligned and shifted to the origin, every
    geometry alike (walls_parts, stairs, features and areas included); `rotation` records the angle. The walls must
    not be empty. Keys that are not geometry (floor_id, site, public, licence) are carried over."""
    ang = dominant_angle(rec["walls"])
    x0, y0, *_ = shapely.total_bounds(shapely.affinity.rotate(rec["walls"], -ang, origin=(0, 0)))
    tf = lambda p: shapely.affinity.translate(shapely.affinity.rotate(p, -ang, origin=(0, 0)), -x0, -y0)
    out = {k: tf(rec[k]) for k in GEOMETRY_KEYS}
    out["walls_parts"] = [tf(p) for p in rec.get("walls_parts", [])]
    out["stairs"] = [tf(p) for p in rec["stairs"]]
    out["features"] = [(k, tf(p)) for k, p in rec["features"]]
    out["areas"] = [(k, tf(p)) for k, p in rec["areas"]]
    out["rotation"] = ang
    out.update({k: v for k, v in rec.items() if k not in out})
    return out


def floor_record(rows):
    """FloorRecord of one Swiss Dwellings floor (the rows of its floor_id), or None without walls."""
    g = shapely.from_wkt(rows.geometry.values)
    sub, typ = rows.entity_subtype.values, rows.entity_type.values
    union = lambda m: shapely.union_all(g[m]) if m.any() else shapely.Polygon()
    rec = {
        "walls": union(sub == "WALL"),
        "walls_parts": [p for p in g[sub == "WALL"]],
        "doors": union((sub == "DOOR") | (sub == "ENTRANCE_DOOR")),
        "windows": union(sub == "WINDOW"),
        "columns": union(sub == "COLUMN"),
        "railings": union(sub == "RAILING"),
        "stairs": [p for p in g[sub == "STAIRS"]],
        "features": [(s, p) for s, p in zip(sub, g) if s in FEATURES],
        "areas": [(s, p) for s, p, t in zip(sub, g, typ) if t == "area"],
    }
    if rec["walls"].is_empty:
        return None
    return normalise_record(rec)


if __name__ == "__main__":
    import pandas as pd
    DATA.mkdir(exist_ok=True)
    cols = ["site_id", "building_id", "plan_id", "floor_id", "unit_usage", "entity_type", "entity_subtype", "geometry"]
    d = pd.read_csv(SD_CSV, usecols=cols, dtype=str)
    # one floor per plan: floors of the same plan repeat the same geometry
    floors = d.groupby("floor_id").agg(plan=("plan_id", "first"), site=("site_id", "first"), n=("geometry", "size"))
    keep = floors.sort_values("n", ascending=False).drop_duplicates("plan").index
    sites = np.array(sorted(floors.site.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(sites)
    n = len(sites)
    split_of = {s: ("test" if i < n * 0.04 else "val" if i < n * 0.08 else "train") for i, s in enumerate(sites)}
    out = {"train": [], "val": [], "test": []}
    d = d[d.floor_id.isin(set(keep))]
    for i, (fid, rows) in enumerate(d.groupby("floor_id", sort=False)):
        rec = floor_record(rows)
        if rec is None:
            continue
        rec.update(floor_id=fid, site=rows.site_id.iat[0], public=float((rows.unit_usage == "PUBLIC").mean()))
        out[split_of[rec["site"]]].append(zlib.compress(pickle.dumps(rec, protocol=5)))
        if i % 1000 == 0:
            print(i, "floors", flush=True)
    for k, v in out.items():
        (DATA / f"floors-{k}.pkl").write_bytes(pickle.dumps(v, protocol=5))
        print(k, len(v), "floors", f"{sum(map(len, v)) / 1e6:.0f} MB")
