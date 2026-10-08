"""Curated plan collection (data/curated): the manifest, its validation, the loaders and the synthetic renders.

The manifest data/curated/plans.json (tracked) documents its fields and controlled vocabularies in its meta branch, as
research/papers.json does. BBL plans go only into data/curated/plans.local.json: same schema, gitignored, never published
or sent anywhere. Images, label maps and room files are in data/curated/images (gitignored).

load(entry) returns Loaded(image, px_per_m or None, scale_method, reference). The reference comes from the source
dataset through the dataset modules (CubiCasa5K and CVC-FP polygons, WAFFLE masks; read-only), from the label map
and room polygons stored with a render, or from the six-class label map copied with a FloorPlanCAD block.
native(entry) returns the Native the other dataset modules return, so that bench.py can take its samples from the
manifest (source "curated", sid = manifest id).

Errors (exit code 1): malformed meta, duplicate or malformed ids, missing or unknown fields, categorical values outside
their vocabulary, scale or reference records that do not fit the schema, licences that do not fit `use`, BBL plans in
the tracked manifest, image files missing or not named after the id, render specs on non-render sources.
Warnings: vocabulary values never used, origins missing on disk (a dataset not downloaded on this machine).
"""
import json
import re
from collections import Counter, namedtuple
from pathlib import Path

from common import REPO
from fpeval.datasets import Native, reference as reference_of
from fpeval.raster import rgb

CURATED = REPO / "data/curated"
MANIFEST = CURATED / "plans.json"
LOCAL = CURATED / "plans.local.json"
IMAGES = CURATED / "images"

SINGLE = ("source", "category", "drawing_type", "input", "era", "building_type", "use")
ARRAYS = ("wall_style", "content", "language", "challenges")
NON_EMPTY = ("wall_style", "language", "challenges")            # content may be empty (walls only)
OPTIONAL = {"render"}
RENDER_SOURCES = {"IFC-Bench render", "Swiss Dwellings render"}
ID_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
# exact licence -> permitted use (meta.vocabularies.use)
USE_OF = [(re.compile(r"CC BY-NC"), "benchmark only"), (re.compile(r"^(Public domain|CC0)"), "public domain or CC0"),
          (re.compile(r"^(CC BY-SA|GPL)"), "share-alike"), (re.compile(r"^(CC BY \d|MIT)"), "attribution"),
          (re.compile(r"^BBL"), "BBL internal")]

Loaded = namedtuple("Loaded", "image px_per_m scale_method reference")
# USACE PDF pages rendered as data/benchmark/construction-plans/images/cpNN.png (see its attribution.csv)
CP_PAGES = {("usace-hnc-acsc-standard-drawings-2011.pdf", p): f"cp{k:02d}" for k, p in ((1, 11), (2, 13), (3, 14), (4, 17))}
CP_PAGES.update({("usace-sas-bde-bn-hq-standard-design-rev7-2026.pdf", 56): "cp05",
                 ("usace-sas-bde-bn-hq-standard-design-rev7-2026.pdf", 65): "cp06"})


# ---------- manifest ----------

def read(path=MANIFEST):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plans(local=True, path=MANIFEST):
    """All manifest entries, plans.local.json appended when present (each entry marked with "_local")."""
    entries = list(read(path)["plans"])
    if local and LOCAL.exists():
        entries += [{**e, "_local": True} for e in read(LOCAL).get("plans", [])]
    return entries


def by_id(local=True):
    return {e["id"]: e for e in plans(local)}


def entry_of(x):
    return by_id()[x] if isinstance(x, str) else x


def dataset_key(e):
    """(bench source, sample id) of an entry whose origin lies in a benchmark dataset, so that outputs of bench.py's
    own samples can be matched to the manifest; None otherwise."""
    o = e.get("origin", "")
    m = re.match(r"data/benchmark/cubicasa5k/(.+)/F1_scaled\.png$", o)
    if m:
        return "cubicasa", m.group(1)
    m = re.match(r"data/benchmark/cvc-fp/ImagesGT/([^/]+)\.(png|jpg)$", o)
    if m:
        return "cvcfp", m.group(1)
    m = re.match(r"data/benchmark/waffle/data/benchmark/pngs/(.+)\.png$", o)
    if m:
        return "waffle", m.group(1)
    m = re.match(r"data/benchmark/commons-plans/images/(c\d+)\.\w+$", o)
    if m:
        return "commons", m.group(1)
    m = re.match(r"data/benchmark/construction-plans/images/(cp\d+)\.\w+$", o)
    if m:
        return "construction", m.group(1)
    m = re.match(r"data/benchmark/construction-plans/pdf/(.+\.pdf)#page=(\d+)$", o)
    if m and (m.group(1), int(m.group(2))) in CP_PAGES:
        return "construction", CP_PAGES[(m.group(1), int(m.group(2)))]
    m = re.match(r"data/benchmark/floorplancad/sample/(\d+-\d+)\.png$", o)
    if m:
        return "floorplancad", m.group(1)
    return None


# ---------- validation ----------

def check(data, local=None, errors=None, warnings=None):
    """Validate a manifest (and the local entries) against data["meta"]. Returns (errors, warnings, used values)."""
    errors = [] if errors is None else errors
    warnings = [] if warnings is None else warnings
    err, warn = errors.append, warnings.append
    meta = data.get("meta") or {}
    for k in ("schema_version", "updated", "description", "fields", "vocabularies"):
        if k not in meta:
            err(f"meta: missing {k}")
    fields, vocab = meta.get("fields", {}), meta.get("vocabularies", {})
    for name, values in vocab.items():
        if not isinstance(values, dict) or not values:
            err(f"vocabularies.{name}: must be a non-empty object value -> description")
            continue
        for v, d in values.items():
            if not isinstance(d, str) or not d:
                err(f"vocabularies.{name}[{v!r}]: needs a description")
    for name in SINGLE + ARRAYS + ("reference",):
        if name not in vocab:
            err(f"vocabularies: missing {name}")
    for f in fields:
        if not isinstance(fields[f], str) or not fields[f]:
            err(f"fields.{f}: needs a description")
    entries = [(e, False) for e in data.get("plans", [])] + [(e, True) for e in (local or [])]
    used = {name: Counter() for name in SINGLE + ARRAYS + ("reference",)}
    ids = Counter(e.get("id") for e, _ in entries)
    for i, n in ids.items():
        if n > 1:
            err(f"id {i!r} used {n} times")
    for e, is_local in entries:
        pid = e.get("id", "<no id>")
        where = f"{pid}{' (local)' if is_local else ''}:"
        missing = [f for f in fields if f not in OPTIONAL and f not in e]
        unknown = [f for f in e if f not in fields and not f.startswith("_")]
        if missing:
            err(f"{where} missing fields {missing}")
        if unknown:
            err(f"{where} unknown fields {unknown} (document them in meta.fields)")
        if not ID_RE.fullmatch(str(pid)) or len(str(pid)) > 40:
            err(f"{where} id must be a short slug (lowercase letters, digits, hyphens)")
        for f in ("title", "origin", "licence", "why"):
            if not isinstance(e.get(f), str) or not e.get(f, "").strip():
                err(f"{where} {f} must be a non-empty string")
        why = e.get("why") or ""
        if why and (not why.rstrip().endswith(".") or len(why) > 300 or why.count(". ") > 1):
            err(f"{where} why must be one sentence ending in a full stop (at most 300 characters)")
        for f in SINGLE:
            v = e.get(f)
            if f in vocab and v not in vocab[f]:
                err(f"{where} {f} {v!r} not in vocabularies.{f}")
            else:
                used[f][v] += 1
        for f in ARRAYS:
            v = e.get(f)
            if not isinstance(v, list):
                err(f"{where} {f} must be an array")
                continue
            if f in NON_EMPTY and not v:
                err(f"{where} {f} is empty")
            dup = [x for x, n in Counter(v).items() if n > 1]
            if dup:
                err(f"{where} {f} has duplicates {dup}")
            for x in v:
                if f in vocab and x not in vocab[f]:
                    err(f"{where} {f} value {x!r} not in vocabularies.{f}")
                used[f][x] += 1
        if "none" in (e.get("language") or []) and len(e["language"]) > 1:
            err(f"{where} language 'none' cannot be combined with languages")
        # file: images/<id>.<ext>, present on disk (when the images folder exists at all)
        file = e.get("file") or ""
        if not re.fullmatch(r"images/[^/]+\.(png|jpg|jpeg|tif|tiff|pdf)", file, re.I) or Path(file).stem != pid:
            err(f"{where} file must be images/<id>.<ext>, got {file!r}")
        elif IMAGES.exists() and not (CURATED / file).exists():
            err(f"{where} {file} missing")
        # origin: URL or a path in the repository
        origin = e.get("origin") or ""
        if origin and not origin.startswith(("http://", "https://")):
            if origin.startswith(("/", "\\")) or re.match(r"[A-Za-z]:", origin):
                err(f"{where} origin must be a URL or a path relative to the repository")
            elif not (REPO / origin.split("#")[0]).exists():
                warn(f"{where} origin {origin} not on disk")
        # scale
        s = e.get("scale")
        if not isinstance(s, dict) or set(s) != {"px_per_m", "method"}:
            err(f"{where} scale must be {{px_per_m, method}}")
        else:
            if s["px_per_m"] is not None and not (isinstance(s["px_per_m"], (int, float)) and s["px_per_m"] > 0):
                err(f"{where} scale.px_per_m must be a positive number or null")
            if not isinstance(s["method"], str) or not s["method"]:
                err(f"{where} scale.method must be a non-empty string")
            elif (s["px_per_m"] is None) != s["method"].startswith("unknown"):
                err(f"{where} scale.method must start with 'unknown' exactly when px_per_m is null")
        # reference
        r = e.get("reference")
        if not isinstance(r, dict) or set(r) != {"content", "source"}:
            err(f"{where} reference must be {{content, source}}")
        else:
            if r["content"] not in vocab.get("reference", {}):
                err(f"{where} reference.content {r['content']!r} not in vocabularies.reference")
            else:
                used["reference"][r["content"]] += 1
            if not isinstance(r["source"], str) or (r["content"] != "none" and not r["source"]):
                err(f"{where} reference.source must say where the reference comes from")
            if r["content"] != "none" and e.get("source") not in REFERENCE_SOURCES:
                err(f"{where} no reference loader for source {e.get('source')!r}")
        # attribution
        a = e.get("attribution")
        if not isinstance(a, dict) or set(a) != {"author", "date", "page"}:
            err(f"{where} attribution must be {{author, date, page}}")
        elif e.get("use") in ("attribution", "share-alike") and not (a["author"] and a["page"]):
            err(f"{where} licence {e.get('licence')!r} requires attribution: author and page")
        # licence and use
        lic = e.get("licence") or ""
        want = next((u for rx, u in USE_OF if rx.search(lic)), None)
        if want is None:
            err(f"{where} licence {lic!r} unknown: add it to USE_OF in fpeval/datasets/curated.py")
        elif want != e.get("use"):
            err(f"{where} use {e.get('use')!r} does not fit licence {lic!r} (expected {want!r})")
        # BBL plans only in plans.local.json
        bbl = e.get("source") == "BBL" or e.get("use") == "BBL internal" or "data/bbl" in origin.replace("\\", "/")
        if bbl and not is_local:
            err(f"{where} BBL plans belong in plans.local.json only (never in the tracked manifest)")
        # render spec
        if "render" in e:
            if e.get("source") not in RENDER_SOURCES:
                err(f"{where} render spec only for {sorted(RENDER_SOURCES)}")
            elif not isinstance(e["render"], dict) or not {"floors", "floor_id", "seed", "style"} <= set(e["render"]):
                err(f"{where} render needs floors, floor_id, seed, style")
        elif e.get("source") in RENDER_SOURCES:
            err(f"{where} renders need a render spec")
    for f, counter in used.items():
        for value in vocab.get(f, {}):
            if not counter[value]:
                warn(f"vocabularies.{f}: {value!r} never used")
    return errors, warnings, used


def validate(path=MANIFEST, local=True):
    data = read(path)
    loc = read(LOCAL).get("plans", []) if local and LOCAL.exists() else []
    return check(data, loc)


# ---------- loader ----------

def _cubicasa(e):
    from fpeval.datasets import cubicasa
    return {"kind": "polygons", **cubicasa.native(dataset_key(e)[1]).polys}


def _cvcfp(e):
    from fpeval.datasets import cvcfp
    return {"kind": "polygons", **cvcfp.native(dataset_key(e)[1]).polys}


def _waffle(e):
    from fpeval.datasets import waffle
    return {"kind": "masks", **waffle.native(dataset_key(e)[1]).masks}


def render_files(e):
    stem = CURATED / Path(e["file"]).with_suffix("")
    return stem.with_name(stem.name + "_label.png"), stem.with_name(stem.name + "_rooms.json")


def _label_masks(lab):
    from fpx.model import COLUMN, DOOR, STAIRS, WALL, WINDOW
    return {"label": lab, "walls": lab == WALL, "doors": lab == DOOR, "windows": lab == WINDOW,
            "columns": lab == COLUMN, "stairs": lab == STAIRS}


def _render(e):
    """Label map of the renderer (fpx.model classes) as masks, plus the floor's rooms as typed polygons (image pixels)."""
    import cv2
    from shapely.geometry import shape
    label_path, rooms_path = render_files(e)
    lab = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(f"{label_path} missing: run python curated.py render {e['id']}")
    rooms = json.loads(rooms_path.read_text(encoding="utf-8"))
    return {"kind": "masks", **_label_masks(lab), "rooms": [(r["type"], shape(r["geometry"])) for r in rooms]}


def _label_map(e):
    """A six-class label map (fpx.model classes) without rooms, copied next to the image as images/<id>_label.png, or
    read next to the origin (FloorPlanCAD sample: <block>_label.png)."""
    import cv2
    path = render_files(e)[0]
    if not path.exists():
        o = REPO / e["origin"]
        path = o.with_name(o.stem + "_label.png")
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(f"label map of {e['id']} missing ({path})")
    return {"kind": "masks", **_label_masks(lab)}


REFERENCE_SOURCES = {"CubiCasa5K": _cubicasa, "CVC-FP": _cvcfp, "WAFFLE benchmark": _waffle,
                     "IFC-Bench render": _render, "Swiss Dwellings render": _render, "FloorPlanCAD": _label_map}
MASK_KEYS = ("walls", "doors", "windows", "interior", "columns", "stairs")


def native_reference(e):
    """The plan's reference in native image pixels, or None. Polygons {"kind": "polygons", walls, doors, windows,
    rooms: [(type, polygon)], ...} or masks {"kind": "masks", walls, doors, windows, (interior | columns, stairs,
    label, rooms)}."""
    e = entry_of(e)
    if e["reference"]["content"] == "none":
        return None
    return REFERENCE_SOURCES[e["source"]](e)


def load(e, with_reference=True):
    """Loaded(image (H, W, 3) uint8 RGB, px_per_m or None, scale method, reference or None) of a manifest entry or id."""
    e = entry_of(e)
    img = rgb(CURATED / e["file"])
    return Loaded(img, e["scale"]["px_per_m"], e["scale"]["method"], native_reference(e) if with_reference else None)


def native(e):
    """The Native of a manifest entry: polygons as polygons, masks as masks, a render's rooms as polygons next to
    its label-map masks."""
    e = entry_of(e)
    img, px, method, ref = load(e)
    if ref is None:
        return Native(img, px, method, None, None)
    if ref["kind"] == "polygons":
        return Native(img, px, method, {k: v for k, v in ref.items() if k != "kind"}, None)
    masks = {k: ref[k] for k in MASK_KEYS if k in ref}
    polys = {"rooms": ref["rooms"]} if "rooms" in ref else None
    return Native(img, px, method, polys, masks)


def reference(nat, work, cfg, e=None):
    """Reference in the working frame: the source dataset's own rule where it has one (CVC-FP door centres), else
    the generic one from polygons and masks."""
    if e is not None and entry_of(e)["source"] == "CVC-FP":
        from fpeval.datasets import cvcfp
        return cvcfp.reference(nat, work, cfg)
    return reference_of(nat, work, cfg)


# ---------- synthetic renders ----------

def render(e):
    """Draw a render entry whole with synth.Renderer at the frame's px_per_m (fpeval.datasets.render.render_floor),
    unrotated, with the style overrides of its render spec; write the image, the label map (fpx.model classes) and the
    rooms (image px)."""
    import pickle
    import zlib
    from dataclasses import replace
    import cv2
    import numpy as np
    import shapely
    from fpeval import oracle
    from fpeval.datasets.render import render_floor
    from fpx import DEFAULT
    from synth import Renderer
    spec = e["render"]
    floors = REPO / spec["floors"]
    rec = next(r for r in (pickle.loads(zlib.decompress(b)) for b in pickle.loads(floors.read_bytes()))
               if str(r["floor_id"]) == spec["floor_id"])
    px_per_m = e["scale"]["px_per_m"]
    tf, shape, _ = oracle.frame(rec, replace(DEFAULT, px_per_m=px_per_m))
    renderer = Renderer(floors)
    base = renderer.style
    renderer.style = lambda rng, **force: base(rng, **{**spec["style"], **force})   # the spec's style choices win
    rng = np.random.default_rng(spec["seed"])
    img, lab, st, info = render_floor(renderer, rec, tf, shape, rng, px_per_m)
    ref = oracle.reference(rec, tf)
    IMAGES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CURATED / e["file"]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    label_path, rooms_path = render_files(e)
    cv2.imwrite(str(label_path), lab)
    rooms = [{"type": k, "geometry": shapely.geometry.mapping(shapely.set_precision(p, 0.01))} for k, p in ref["rooms"]]
    rooms_path.write_text(json.dumps(rooms), encoding="utf-8")
    style = {k: str(v) for k, v in st.items() if k in ("era", "walls", "doors", "windows", "font", "text")}
    print(f"{e['id']}: {img.shape[1]}x{img.shape[0]} px, {len(rooms)} rooms, style {style}, "
          f"elements {','.join(info['elements'])}", flush=True)
    return img, lab
