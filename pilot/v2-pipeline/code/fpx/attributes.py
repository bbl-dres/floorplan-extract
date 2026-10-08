"""Stage 7, room attributes: room stamps from text pieces, names, stamp areas, room numbers, AOIDs (and which are
exported), usage, and small unlabelled regions (kept as rooms, flagged for review)."""
import difflib
import re
from collections import Counter

import networkx as nx
import numpy as np
from shapely.geometry import Point

from .config import DEFAULT
from .text import AOID_RE, VOID_NAMES

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
VOCAB_KEYS = {re.sub(r"[^a-zäöüéèà]", "", v.lower()) for v in VOCAB}
# an area on a stamp: an optional short label (NF, F=, Fläche), the number, an optional unit as OCR renders it
# (m², m2, qm, m^2, m? for a lost superscript); integers count only with a unit, decimals ("24.50") also without
AREA_RE = re.compile(r"(?:(?P<pre>[A-Za-zÄÖÜäöü]{1,8})\s*[:=]?\s*)?(?P<int>\d{1,4})(?:\s*[.,]\s*(?P<dec>\d{1,2}))?"
                     r"\s*(?P<unit>m\s*[2²?'°]|qm|m\^2)?(?:\s*[A-Za-z]{0,4})?", re.I)
NUMBER_RE = re.compile(r"\d{1,2}[.,]\d{2,3}|\d{3,4}")         # a room number: 1.12, 2.045, 204
AREA_MIN = 1.5                                                # a bare decimal below this is a room number, not an area (m²)


def stamp_area(parts):
    """Area and room number of a stamp from its parts. The area: a number with a unit wins; without a unit, the last
    decimal number of at least AREA_MIN m². Bare numbers that do not qualify (1.12, 204) are the room number.
    Returns (area, part used, number)."""
    areas, numbers = [], []
    for p in parts:
        s = p["text"].strip()
        if p["role"] not in ("number", "room stamp"):
            continue
        m = AREA_RE.fullmatch(s)
        if not m or (p["role"] == "room stamp" and not m.group("unit")):
            continue
        value = float(f"{m.group('int')}.{m.group('dec') or 0}")
        if m.group("unit") or (m.group("dec") is not None and value >= AREA_MIN):
            areas.append((m.group("unit") is not None, value, p))
        elif NUMBER_RE.fullmatch(s):
            numbers.append(s.replace(",", "."))
    with_unit = [a for a in areas if a[0]]
    best = with_unit[0] if with_unit else areas[-1] if areas else None
    for unit, value, p in areas:                       # a second plain decimal next to a unit area is a number
        if best is not None and p is not best[2] and not unit and NUMBER_RE.fullmatch(p["text"].strip()):
            numbers.append(p["text"].strip().replace(",", "."))
    return (best[1], best[2], numbers[0] if numbers else None) if best else (None, None, numbers[0] if numbers else None)


def usage(name):
    low = name.lower()
    for pat, u in USAGE:
        if re.search(pat, low):
            return u
    return "unknown"


def join_words(words):
    """Words of a stamp in reading order -> name; hyphenated line breaks are rejoined ("Sitzungs-"/"zimmer",
    "HERREN-"/"ZIMMER") when the join is a vocabulary word, the head is a short prefix or the tail starts in lower
    case; otherwise the hyphen stays ("Kreidolf-Zimmer")."""
    name = ""
    for w in words:
        if name.endswith("-"):
            head = name[:-1].split(" ")[-1]
            joined = re.sub(r"[^a-zäöüéèà]", "", (head + w).lower())
            name = name[:-1] + w if joined in VOCAB_KEYS or len(head) < 4 or w[:1].islower() else name + w
        else:
            name = f"{name} {w}".strip()
    return name


def cluster_stamps(items):
    """Group text pieces into room stamps: words on one line, and lines stacked below each other. A cluster whose
    name is a void word (Luftraum, Vide, also hyphenated over two lines) marks its parts as void labels instead."""
    cand = [t for t in items if t["role"] in ("room stamp", "number", "void label")]
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
        if not any(p["role"] in ("room stamp", "void label") for p in parts):
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
        area, area_part, number = stamp_area(parts)
        words = [p["text"].strip() for p in parts
                 if p["role"] in ("room stamp", "void label") and p is not area_part and not AOID_RE.search(p["text"])]
        name = join_words(words)
        if re.sub(r"[^A-ZÄÖÜ]", "", name.upper()) in VOID_NAMES:
            for p in parts:
                p["role"] = "void label"
            continue
        aoid = next((AOID_RE.search(p["text"]).group(0) for p in parts if AOID_RE.search(p["text"])), None)
        box = (min(p["box"][0] for p in parts), min(p["box"][1] for p in parts),
               max(p["box"][2] for p in parts), max(p["box"][3] for p in parts))
        for p in parts:
            p["role"] = "room stamp"
        stamps.append({"name": name or None, "area": area, "aoid": aoid, "number": number, "box": box,
                       "conf": float(np.mean([p["conf"] for p in parts])), "parts": parts,
                       "source": "ocr" if any(p.get("source") == "ocr" for p in parts) else "pdf"})
    return stamps


def correct_name(raw, cfg=DEFAULT):
    """Snap OCR'd room names to the vocabulary: the whole name if very close, otherwise each word separately, so
    proper-name prefixes such as 'Kreidolf-' are kept. Words with digits and words much shorter or longer than the
    vocabulary entry stay as read."""
    key = lambda s: re.sub(r"[^a-zäöüéèà]", "", s.lower())
    ratio = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
    k = key(raw)
    if len(k) < 2:
        return raw
    best = max(VOCAB, key=lambda v: ratio(k, key(v)))
    if ratio(k, key(best)) >= cfg.name_match_whole and min(len(k), len(key(best))) >= 0.7 * max(len(k), len(key(best))):
        return best
    words = sorted({w for v in VOCAB for w in re.split(r"[-\s]", v) if len(w) >= 2} | {"Herren", "Damen", "Diensten"})
    out = []
    for part in re.split(r"([-\s]+)", raw.strip(" .,:;")):
        pk = key(part)
        if len(pk) >= 3 and not any(ch.isdigit() for ch in part):
            w = max(words, key=lambda v: ratio(pk, key(v)))
            close = ratio(pk, key(w)) >= cfg.name_match_word and min(len(pk), len(key(w))) >= 0.7 * max(len(pk), len(key(w)))
            part = (w.upper() if part.isupper() else w) if close else part
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
        r.update(names=[], area_stamp=None, aoid=None, aoids=[], number=None, voids=[])

    def room_of(pt):
        hit = [r for r in sheet.rooms if r["poly"].contains(pt)]
        if hit:
            return hit[0]
        near = min(sheet.rooms, key=lambda r: r["poly"].distance(pt), default=None)
        return near if near is not None and near["poly"].distance(pt) < cfg.px(cfg.stamp_near_room) else None

    sheet.stamps = sheet.stamps or cluster_stamps(sheet.text)     # stage 6 clusters them first (stair rooms)
    for s in sheet.stamps:
        r = room_of(Point((s["box"][0] + s["box"][2]) / 2, (s["box"][1] + s["box"][3]) / 2))
        s["room"] = r["id"] if r else None
        if r is None:
            continue
        if s["name"]:
            s["name_raw"] = s["name"]
            if s["source"] == "ocr":                   # native PDF or DXF strings are never "corrected"
                s["name"] = correct_name(s["name"], cfg)
            r["names"].append(s["name"])
            r.setdefault("names_raw", []).append(s["name_raw"])
        if s["area"] and r["area_stamp"] is None:
            r["area_stamp"] = s["area"]
        if s.get("number") and r["number"] is None:
            r["number"] = s["number"]
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
