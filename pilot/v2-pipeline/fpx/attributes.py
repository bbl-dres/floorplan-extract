"""Stage 7, room attributes: room stamps from text pieces, names, stamp areas, AOIDs (and which are exported), usage,
and small unlabelled regions (kept as rooms, flagged for review)."""
import difflib
import re
from collections import Counter

import networkx as nx
import numpy as np
from shapely.geometry import Point

from .config import DEFAULT
from .text import AOID_RE

USAGE = [(r"treppe|escalier|scala", "stair"), (r"bad|wc|toilet|dusche|lavabo|sanit", "sanitary"),
         (r"flur|gang|korridor|halle|vorplatz|vestibul|entr|foyer|corridor|couloir", "circulation"),
         (r"balkon|loggia|terrasse|balcon", "outdoor"), (r"küche|kuche|cuisine|cucina|office", "kitchen"),
         (r"lager|abstell|reduit|archiv|estrich|keller", "storage"),
         (r"zimmer|büro|buro|bureau|ufficio|chambre|salon|boudoir|antichambre|saal|stube|sitzung", "room")]
# Room-name vocabulary for correcting OCR (DE/FR/IT, office and historical residential terms). Written after seeing
# the historical test sheet, so corrected-name scores are optimistic; raw OCR is reported as well.
VOCAB = ["Zimmer", "Büro", "Bureau", "Ufficio", "Sitzungszimmer", "Besprechung", "Archiv", "Lager", "Abstellraum",
         "Reduit", "Korridor", "Gang", "Flur", "Halle", "Obere Halle", "Vorplatz", "Vestibül", "Vestibule", "Entrée",
         "Eingang", "Foyer", "Treppe", "Treppenhaus", "Escalier", "Lift", "Bad", "Bad WC", "WC", "Toilette", "Dusche",
         "Küche", "Office", "Speisezimmer", "Esszimmer", "Wohnzimmer", "Salon", "Saal", "Stube", "Schlafzimmer",
         "Kinderzimmer", "Gastzimmer", "Herrenzimmer", "Herren-Zimmer", "Damenzimmer", "Dienstenzimmer", "Diensten-Zimmer",
         "Mädchenzimmer", "Bibliothek", "Boudoir", "Antichambre", "Kabinett", "Garderobe", "Balkon", "Loggia", "Terrasse",
         "Veranda", "Estrich", "Keller", "Waschküche", "Kammer", "Archives", "Chambre", "Cuisine", "Salle à manger",
         "Couloir", "Dégagement", "Camera", "Cucina", "Bagno", "Corridoio", "Scala", "Sala"]
STAMP_AREA_RE = re.compile(r"(\d{1,3})\s*[.,]\s*(\d{1,2})\s*(m2|m²)?", re.I)
# an area written with a short prefix, e.g. "NF 24.50 m²", "F=18.2m2": read as a room stamp word by the text layer
PREFIXED_AREA_RE = re.compile(r"[A-Za-z]{1,3}\s*[:=]?\s*(\d{1,3})\s*[.,]\s*(\d{1,2})\s*(m2|m²)", re.I)


def stamp_area(parts):
    """Area of a stamp from its parts: a number with an explicit unit wins over bare numbers (room numbers such as
    "1.12" look like areas); without a unit, the last number in reading order. Returns (area, part used)."""
    found = []
    for p in parts:
        s = p["text"].strip()
        m = STAMP_AREA_RE.fullmatch(s) if p["role"] == "number" else PREFIXED_AREA_RE.fullmatch(s)
        if m:
            found.append((m.group(3) is not None, float(f"{m.group(1)}.{m.group(2)}"), p))
    with_unit = [f for f in found if f[0]]
    best = with_unit[0] if with_unit else found[-1] if found else None
    return (best[1], best[2]) if best else (None, None)


def usage(name):
    low = name.lower()
    for pat, u in USAGE:
        if re.search(pat, low):
            return u
    return "unknown"


def cluster_stamps(items):
    """Group text pieces into room stamps: words on one line, and lines stacked below each other."""
    cand = [t for t in items if t["role"] in ("room stamp", "number")]
    frame = lambda t: t["box"] if t["angle"] == 0 else (-t["box"][3], t["box"][0], -t["box"][1], t["box"][2])
    g = nx.Graph()
    g.add_nodes_from(range(len(cand)))
    for i, a in enumerate(cand):
        ax0, ay0, ax1, ay1 = frame(a)
        for j in range(i + 1, len(cand)):
            b = cand[j]
            if a["angle"] != b["angle"]:
                continue
            bx0, by0, bx1, by1 = frame(b)
            h = max(min(ay1 - ay0, by1 - by0), 3)
            same_line = min(ay1, by1) - max(ay0, by0) > 0.5 * h and max(bx0 - ax1, ax0 - bx1) < 1.2 * h
            centred = abs((ax0 + ax1) - (bx0 + bx1)) / 2 < 0.6 * max(ax1 - ax0, bx1 - bx0)
            stacked = (min(ax1, bx1) > max(ax0, bx0) or centred) and max(by0 - ay1, ay0 - by1) < 0.9 * h
            if same_line or stacked:
                g.add_edge(i, j)
    stamps = []
    for comp in nx.connected_components(g):
        parts = [cand[i] for i in comp]
        if not any(p["role"] == "room stamp" for p in parts):
            continue                                   # numbers only: dimensions
        parts.sort(key=lambda p: (frame(p)[1] + frame(p)[3]) / 2)       # rows top to bottom, then left to right
        rows = []
        for p in parts:
            yc, h = (frame(p)[1] + frame(p)[3]) / 2, frame(p)[3] - frame(p)[1]
            if rows and abs(yc - rows[-1][0]) < 0.5 * h:
                rows[-1][1].append(p)
            else:
                rows.append([yc, [p]])
        parts = [p for _, row in rows for p in sorted(row, key=lambda p: frame(p)[0])]
        area, area_part = stamp_area(parts)
        words = [p["text"].strip() for p in parts
                 if p["role"] == "room stamp" and p is not area_part and not AOID_RE.search(p["text"])]
        name = ""
        for w in words:                                # rejoin hyphenated line breaks: "Sitzungs-/zimmer", "Herren-/Zimmer"
            if name.endswith("-"):
                head = name[:-1].split(" ")[-1]
                name = name[:-1] + w if len(head) < 4 or w[:1].islower() else name + w
            else:
                name = f"{name} {w}".strip()
        aoid = next((AOID_RE.search(p["text"]).group(0) for p in parts if AOID_RE.search(p["text"])), None)
        box = (min(p["box"][0] for p in parts), min(p["box"][1] for p in parts),
               max(p["box"][2] for p in parts), max(p["box"][3] for p in parts))
        for p in parts:
            p["role"] = "room stamp"
        stamps.append({"name": name or None, "area": area, "aoid": aoid, "box": box,
                       "conf": float(np.mean([p["conf"] for p in parts])), "parts": parts})
    return stamps


def correct_name(raw, cfg=DEFAULT):
    """Snap OCR'd room names to the vocabulary: the whole name if very close, otherwise each word separately, so
    proper-name prefixes such as 'Kreidolf-' are kept."""
    key = lambda s: re.sub(r"[^a-zäöüéèà]", "", s.lower())
    ratio = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
    k = key(raw)
    if len(k) < 2:
        return raw
    best = max(VOCAB, key=lambda v: ratio(k, key(v)))
    if ratio(k, key(best)) >= cfg.name_match_whole:
        return best
    words = sorted({w for v in VOCAB for w in re.split(r"[-\s]", v) if len(w) >= 2} | {"Herren", "Damen", "Diensten"})
    out = []
    for part in re.split(r"([-\s]+)", raw.strip(" .,:;")):
        pk = key(part)
        if len(pk) >= 3:
            w = max(words, key=lambda v: ratio(pk, key(v)))
            part = (w.upper() if part.isupper() else w) if ratio(pk, key(w)) >= cfg.name_match_word else part
        out.append(part)
    return "".join(out)


def aoid_export(rooms):
    """R_AOID policy (CAD-Richtlinie Kap. 5.10, plan-check AOID_001-006): a room's AOID is written only when exactly
    one AOID was read in the room and in no other room. AOIDs are never generated: rooms without one are left empty
    and flagged, to be matched from SAP (step 8)."""
    owners = Counter(a for r in rooms for a in set(r["aoids"]))
    for r in rooms:
        read = sorted(set(r["aoids"]))
        if not read:
            r["aoid_export"] = {"written": False, "reason": "no AOID read on the sheet: left empty, to be matched from SAP, "
                                                            "never generated"}
        elif len(read) > 1:
            r["aoid_export"] = {"written": False, "reason": f"{len(read)} AOIDs read in one room ({', '.join(read)}): "
                                                            "a wall may be missing"}
        elif owners[read[0]] > 1:
            others = [o["id"] for o in rooms if o is not r and read[0] in o["aoids"]]
            r["aoid_export"] = {"written": False, "reason": f"AOID {read[0]} also read in {', '.join(others)}: not unique"}
        else:
            r["aoid_export"] = {"written": True, "aoid": read[0]}


def attributes(sheet, cfg=DEFAULT):
    for r in sheet.rooms:
        r.update(names=[], area_stamp=None, aoid=None, aoids=[], voids=[])

    def room_of(pt):
        hit = [r for r in sheet.rooms if r["poly"].contains(pt)]
        if hit:
            return hit[0]
        near = min(sheet.rooms, key=lambda r: r["poly"].distance(pt), default=None)
        return near if near is not None and near["poly"].distance(pt) < cfg.px(cfg.stamp_near_room) else None

    sheet.stamps = cluster_stamps(sheet.text)
    for s in sheet.stamps:
        r = room_of(Point((s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2))
        s["room"] = r["id"] if r else None
        if r is None:
            continue
        if s["name"]:
            s["name_raw"], s["name"] = s["name"], correct_name(s["name"], cfg) if sheet.ocr_img is not None else s["name"]
            r["names"].append(s["name"])
            r.setdefault("names_raw", []).append(s["name_raw"])
        if s["area"] and r["area_stamp"] is None:
            r["area_stamp"] = s["area"]
        if s["aoid"]:
            r["aoids"].append(s["aoid"])
        r["aoid"] = r["aoid"] or s["aoid"]
    for v in sheet.voids:
        r = room_of(v["poly"].representative_point())
        if r:
            r["voids"].append(v)
    for t in sheet.text:                               # numbers outside stamps are dimensions
        if t["role"] == "number":
            t["role"] = "dimension"
    for r in sheet.rooms:
        r["small_region"] = not r["names"] and r["poly"].area < cfg.px2(cfg.fragment_max_area)
        r["name"] = " / ".join(r["names"]) if r["names"] else None
        r["usage"] = "stair" if r["stair"] else usage(r["name"] or "")
    aoid_export(sheet.rooms)
    # every room of 0.25 m² or more needs a polygon (CAD-Richtlinie Kap. 5.8), so small unlabelled regions (shafts,
    # niches, window recesses, gaps behind fixtures) are no longer dropped. They wait in sheet.fragments while stage 8
    # builds the circulation graph, so that a niche beside a door does not take the door's connection, and join the
    # rooms in stage 9 with a review flag (qa.join_small_regions).
    sheet.fragments = [r for r in sheet.rooms if r["small_region"]]
    sheet.rooms = [r for r in sheet.rooms if not r["small_region"]]
    return sheet.rooms
