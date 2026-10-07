"""Stage 9, scale confirmation and QA: stamp-area scale cue, per-room confidence, review flags for small regions,
AOID flags, floor and opening checks."""
import networkx as nx
import numpy as np
import shapely

from .config import DEFAULT


def small_region_review(sheet, r, cfg=DEFAULT):
    """Review flag of a small unlabelled region, with a guess for the reviewer: a gap (narrow), a niche (a door, window
    or passage on its outline) or a shaft (none). A guess from geometry only, never a usage."""
    a = cfg.m2(r["poly"].area)
    width = 2 * shapely.maximum_inscribed_circle(r["poly"], tolerance=1).length / cfg.m
    beside = [o["id"] for o in sheet.openings if o["poly"].distance(r["poly"]) < 2]          # within 2 px
    if width < cfg.gap_max_width:
        guess, why = "gap", f"{width:.2f} m across: a gap between wall lines or behind a fixture, or a wall split in two"
    elif beside:
        guess, why = "niche", f"opening {', '.join(beside)} on its outline: a niche, window recess, closet or vestibule"
    else:
        guess, why = "shaft", "no door, window or passage on its outline: a shaft, chimney, closed cavity, or a niche " \
                              "closed off where walls nearly meet"
    return {"flag": "small unlabelled region", "guess": guess, "width": round(width, 2), "openings": beside,
            "reason": f"{a:.2f} m² without a stamp, {why}. Kept as a room because every room of 0.25 m² or more needs "
                      "a polygon (CAD-Richtlinie 5.8); not part of the circulation graph; check it on the sheet"}


def join_small_regions(sheet):
    """Small unlabelled regions (sheet.fragments, set aside by stage 7 while stage 8 built the circulation graph) join
    the rooms, in id order; each gets its review flag in qa()."""
    sheet.rooms = sorted(sheet.rooms + sheet.fragments, key=lambda r: r["id"])
    sheet.fragments = []


def qa(sheet, cfg=DEFAULT):
    join_small_regions(sheet)
    issues, area = [], lambda g: cfg.m2(g.area)
    net = lambda r: area(r["poly"]) - sum(area(v["poly"]) for v in r.get("voids", []))     # voids are not floor area
    stamped = [(r, net(r)) for r in sheet.rooms if r["area_stamp"]]
    cue = None
    if len(stamped) >= cfg.scale_cue_min_rooms:        # scale cue: stamp areas vs polygon areas
        ratios = [r["area_stamp"] / a for r, a in stamped]
        cue = float(np.sqrt(np.median(ratios)))
    sheet.scale["stamp_area_cue"] = None if cue is None else {"length_factor": round(cue, 3), "rooms": len(stamped),
                                                              "agrees": abs(cue - 1) < cfg.scale_cue_agreement}
    tol, low, small = cfg.stamp_tolerance_pct, cfg.stamp_low_pct, cfg.small_room_area
    for r in sheet.rooms:
        a = net(r)
        reasons = [f"void {area(v['poly']):.1f} m² ({v['label']}) excluded" for v in r.get("voids", [])]
        if r["area_stamp"]:
            dev = (a - r["area_stamp"]) / r["area_stamp"] * 100
            r["area_deviation_pct"] = round(dev, 1)
            if abs(dev) > tol:
                reasons.append(f"area {a:.1f} m² vs stamp {r['area_stamp']:.2f} ({dev:+.0f}%)")
        if len(r["names"]) > 1:
            reasons.append(f"{len(r['names'])} stamps: a wall may be missing")
        if not r["names"] and not r["stair"]:
            reasons.append("no stamp")
        if a < small:
            reasons.append("very small")
        if r["stair"] and not r["voids"]:
            reasons.append("stair: no void identified")
        r["review"] = small_region_review(sheet, r, cfg) if r.get("small_region") else None
        if r["review"]:
            reasons.append(f"small unlabelled region kept for review (a {r['review']['guess']}?)")
        conf = "high" if r["area_stamp"] and abs(r["area_deviation_pct"]) <= tol and len(r["names"]) == 1 else \
            "low" if (r["area_stamp"] and abs(r["area_deviation_pct"]) > low) or len(r["names"]) > 1 or a < small else "medium"
        r["confidence"], r["reasons"] = conf, reasons or ["stamp area matches" if conf == "high" else "no contradiction found"]
        for reason in reasons:
            issues.append({"check": "room", "severity": conf, "element": r["id"], "message": reason})
    rooms_sum = sum(area(r["poly"]) for r in sheet.rooms)
    if sheet.gf_area < rooms_sum:
        issues.append({"check": "GF > sum of rooms", "severity": "high", "element": "floor",
                       "message": f"GF {sheet.gf_area:.1f} < rooms {rooms_sum:.1f}"})
    for o in sheet.openings:
        if o["host"] is None:
            issues.append({"check": "opening hosted by a wall", "severity": "medium", "element": o["id"], "message": "no host wall"})
        if o.get("flag"):
            issues.append({"check": "opening type", "severity": "medium", "element": o["id"], "message": o["flag"]})
    g = sheet.connectivity
    if len(g):
        main = max(nx.connected_components(g), key=len)
        for r in sheet.rooms:
            if r["id"] not in main and not r.get("small_region"):     # small regions carry their own review flag
                issues.append({"check": "reachable", "severity": "medium", "element": r["id"],
                               "message": "not connected to the main circulation"})
    # R_AOID stays empty where no unique AOID was read (plan-check AOID_001): one flag per room, or one per floor when
    # the sheet carries no AOID at all (historical plans)
    unwritten = [r for r in sheet.rooms if not (r.get("aoid_export") or {}).get("written")]
    if unwritten and not any(r.get("aoids") for r in sheet.rooms):
        issues.append({"check": "AOID", "severity": "medium", "element": "floor",
                       "message": f"no AOID on the sheet: R_AOID left empty for all {len(unwritten)} rooms; "
                                  "AOIDs are to be matched from SAP, never generated"})
    else:
        for r in unwritten:
            issues.append({"check": "AOID", "severity": "high" if r.get("aoids") else "medium", "element": r["id"],
                           "message": (r.get("aoid_export") or {}).get("reason", "no AOID")})
    sheet.qa = issues
    return issues
