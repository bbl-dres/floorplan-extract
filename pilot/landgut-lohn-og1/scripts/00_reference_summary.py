"""Summarise the reference: rooms, stamp areas, and openings split into exterior and interior."""
from common import BUILDING, OPENINGS, ROOMS

for name, label_area, poly in ROOMS:
    print(f"{name:<22} stamp {label_area:6.2f} m2   reference polygon {poly.area:6.2f} m2")

outer = BUILDING.exterior
exterior = sorted(round(g, 2) for g, p in OPENINGS if p.distance(outer) < 0.15)
interior = sorted(round(g, 2) for g, p in OPENINGS if p.distance(outer) >= 0.15)
print(f"\nbuilding footprint {BUILDING.area:.2f} m2, bounds {[round(v, 2) for v in BUILDING.bounds]}")
print(f"openings in exterior walls: {len(exterior)} {exterior}")
print(f"interior openings: {len(interior)} {interior}")
print(f"sum of stamp areas {sum(a for _, a, _ in ROOMS):.2f} m2")
