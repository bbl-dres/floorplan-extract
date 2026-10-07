"""Tests of the curated plan set (data/curated/plans.json, curated.py): the manifest validates against its own
vocabularies, every image exists, the validator catches broken entries, and the loader returns sane images, scales
and references for one plan per source (skipped where the source dataset is not on this machine)."""
import copy
import json

import numpy as np
import pytest

import curated

DATA = curated.read()
ENTRIES = curated.plans()
have_images = pytest.mark.skipif(not curated.IMAGES.exists(), reason="data/curated/images not on this machine")


def first(source):
    return next((e for e in ENTRIES if e["source"] == source), None)


def test_manifest_validates():
    errors, warnings, used = curated.validate()
    assert errors == []
    assert len(DATA["plans"]) >= 30
    assert sum(used["category"].values()) == len(ENTRIES)


def test_manifest_has_no_bbl_plans():
    for e in DATA["plans"]:                             # BBL plans only in the gitignored plans.local.json
        assert e["source"] != "BBL" and e["use"] != "BBL internal" and "data/bbl" not in e["origin"]


@have_images
def test_every_file_exists():
    for e in ENTRIES:
        assert (curated.CURATED / e["file"]).is_file(), e["file"]
        label, rooms = curated.render_files(e)
        if "render" in e:
            assert label.is_file() and rooms.is_file(), e["id"]
        if e["source"] == "FloorPlanCAD":                # its label map travels with the copy
            assert label.is_file(), e["id"]


def test_construction_and_fm_plans():
    by_cat = {}
    for e in DATA["plans"]:
        by_cat.setdefault(e["category"], []).append(e)
    assert len(by_cat["Construction drawings"]) >= 5 and len(by_cat["FM and safety plans"]) >= 3
    for e in DATA["plans"]:
        if e["source"] == "US federal drawings" and e["scale"]["px_per_m"]:   # 200 dpi render of a stated paper scale
            assert "200 dpi" in e["scale"]["method"] and 30 < e["scale"]["px_per_m"] < 150
        if e["source"] == "FloorPlanCAD":
            assert e["use"] == "benchmark only" and e["scale"]["px_per_m"] == 100


def test_validator_catches_broken_entries():
    data = copy.deepcopy(DATA)
    e = data["plans"][0]
    bad = {**copy.deepcopy(e), "id": "Bad Id", "era": "medieval", "wall_style": ["poché", "poché"],
           "use": "benchmark only", "extra": 1, "scale": {"px_per_m": None, "method": "known"}}
    bbl = {**copy.deepcopy(e), "id": "bbl-test", "source": "BBL", "licence": "BBL internal", "use": "BBL internal"}
    data["plans"] = [bad, bbl]
    errors, _, _ = curated.check(data)
    text = "\n".join(errors)
    for needle in ("id must be a short slug", "era 'medieval'", "duplicates", "unknown fields ['extra']",
                   "does not fit licence", "scale.method must start with 'unknown'", "plans.local.json only"):
        assert needle in text, needle
    errors, _, _ = curated.check({**data, "plans": []}, local=[bbl])              # allowed in the local file
    assert not [x for x in errors if "plans.local.json only" in x]


def test_dataset_keys():
    keys = {curated.dataset_key(e) for e in ENTRIES} - {None}
    assert ("cubicasa", "colorful/11260") in keys and ("commons", "c02") in keys
    assert any(k[0] == "cvcfp" for k in keys) and any(k[0] == "waffle" for k in keys)
    assert ("construction", "cp03") in keys and ("floorplancad", "0060-0042") in keys


SOURCES = sorted({e["source"] for e in ENTRIES})


@pytest.mark.parametrize("source", SOURCES)
def test_loader_one_plan_per_source(source):
    e = first(source)
    if not (curated.CURATED / e["file"]).exists():
        pytest.skip(f"{e['file']} not on this machine")
    if e["reference"]["content"] != "none" and not e["origin"].startswith("http") and "render" not in e \
            and not (curated.REPO / e["origin"]).exists():
        pytest.skip(f"source dataset of {e['id']} not on this machine")
    img, px_per_m, method, ref = curated.load(e)
    H, W = img.shape[:2]
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8 and min(H, W) > 100
    assert px_per_m is None or 10 < px_per_m < 1000
    assert method == e["scale"]["method"] and (px_per_m is None) == method.startswith("unknown")
    if e["reference"]["content"] == "none":
        assert ref is None
        return
    assert ref["kind"] in ("polygons", "masks")
    if ref["kind"] == "polygons":
        assert ref["walls"] and ref["rooms"]
        x0, y0, x1, y1 = np.array([p.bounds for p in ref["walls"]]).T
        assert x0.min() > -0.05 * W and y0.min() > -0.05 * H and x1.max() < 1.05 * W and y1.max() < 1.05 * H
    else:
        for k in ("walls", "doors", "windows"):
            assert ref[k].shape == (H, W) and ref[k].dtype == bool
        assert ref["walls"].mean() > 0.005                 # walls are drawn
        if "rooms" in ref:                               # render: rooms in image pixels
            assert ref["rooms"] and all(0 <= p.centroid.x <= W and 0 <= p.centroid.y <= H for p in ref["rooms"])


def test_bench_loader_label_map():
    e = first("FloorPlanCAD")
    if e is None or not (curated.CURATED / e["file"]).exists():
        pytest.skip("no FloorPlanCAD block on this machine")
    img, px, method, ref, masks = curated.bench_loader(e)
    assert px == 100 and ref is None
    assert {"walls", "doors", "windows", "columns", "stairs"} <= set(masks)
    assert all(m.shape == img.shape[:2] for m in masks.values()) and masks["walls"].any()


def test_bench_loader_render():
    e = first("IFC-Bench render") or first("Swiss Dwellings render")
    if e is None or not curated.render_files(e)[0].exists():
        pytest.skip("no render on this machine")
    img, px, method, ref, masks = curated.bench_loader(e)
    assert px == e["scale"]["px_per_m"] and masks is None
    assert set(ref) == {"walls", "doors", "windows", "rooms"} and ref["walls"] and ref["rooms"]
    rooms = json.loads(curated.render_files(e)[1].read_text(encoding="utf-8"))
    assert len(ref["rooms"]) == len(rooms)
