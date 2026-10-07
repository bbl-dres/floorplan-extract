"""Swiss Dwellings -> data/floors-{train,val,test}.pkl: one floor per plan, grouped by element class.

    python sd_prepare.py

Each floor is rotated so that its dominant wall direction is axis-aligned (archive plans are drawn that way)
and shifted to the origin. The split is by site, so no building appears in two splits.
"""
import pickle
import zlib

import numpy as np
import pandas as pd
import shapely

from common import DATA, SD_CSV

FEATURES = {"KITCHEN", "SINK", "TOILET", "BATHTUB", "SHOWER", "ELEVATOR", "BUILT_IN_FURNITURE", "WASHING_MACHINE", "RAMP"}


def dominant_angle(walls):
    """Length-weighted mode of wall edge directions modulo 90 degrees."""
    coords = [np.asarray(r.coords) for p in getattr(walls, "geoms", [walls]) for r in [p.exterior, *p.interiors]]
    seg = np.concatenate([c[1:] - c[:-1] for c in coords])
    length = np.hypot(seg[:, 0], seg[:, 1])
    ang = np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) % 90
    hist, edges = np.histogram(ang, bins=180, range=(0, 90), weights=length)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)                 # smooth, wrapping at 0/90
    return float(edges[np.argmax(hist)] + 0.25)


def floor_record(rows):
    g = shapely.from_wkt(rows.geometry.values)
    sub, typ = rows.entity_subtype.values, rows.entity_type.values
    union = lambda m: shapely.union_all(g[m]) if m.any() else shapely.Polygon()
    rec = {
        "walls": union(sub == "WALL"),
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
    ang = dominant_angle(rec["walls"])
    x0, y0, *_ = shapely.total_bounds(shapely.affinity.rotate(rec["walls"], -ang, origin=(0, 0)))
    tf = lambda p: shapely.affinity.translate(shapely.affinity.rotate(p, -ang, origin=(0, 0)), -x0, -y0)
    out = {k: tf(rec[k]) for k in ("walls", "doors", "windows", "columns", "railings")}
    out["stairs"] = [tf(p) for p in rec["stairs"]]
    out["features"] = [(s, tf(p)) for s, p in rec["features"]]
    out["areas"] = [(s, tf(p)) for s, p in rec["areas"]]
    out["rotation"] = ang
    return out


if __name__ == "__main__":
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
