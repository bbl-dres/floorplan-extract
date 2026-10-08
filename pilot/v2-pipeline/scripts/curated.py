"""Curated plan collection (data/curated): validate the manifest, draw its synthetic renders, run the pipeline on it.

    python curated.py                       # validate plans.json (and plans.local.json if present), print a summary
    python curated.py render [ids]          # draw the synthetic renders of the manifest again (image, label map, rooms)
    python curated.py run ids [--out DIR] [--model PATH]   # stages 0-10 on curated plans ("all" or no ids: every plan;
                                            # default output data/curated-test, default model common.MODEL)

The manifest, its schema and the loaders are fpeval.datasets.curated. BBL plans go only into plans.local.json
(gitignored) and their outputs stay below the data folder. Exit code 1 when the manifest has errors.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from common import DATA
from fpeval.cli import cli, record_failure, setup, utf8_stdout
from fpeval.datasets import curated

TEST_OUT = DATA / "curated-test"


def summary(used, n):
    print(f"\n{n} plans")
    for f, counter in used.items():
        print(f"\n{f}")
        for k, v in counter.most_common():
            print(f"  {v:>3}  {k}")


def render_all(ids=None):
    for e in curated.plans(local=False):
        if "render" in e and (not ids or e["id"] in ids):
            curated.render(e)


def run(ids, out_dir=TEST_OUT, model_path=None, cfg=None):
    """Stages 0-10 on curated plans through bench.run (its scale cues, scoring and export), with outputs in out_dir
    instead of data/bench. Each <sheet>_bench.json gets the curated id, so export_viewer.py groups it by category."""
    import bench
    from common import MODEL
    from fpx import DEFAULT, pipeline
    from fpx.segment import load_model
    entries = curated.by_id()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench.SCALE_OUT = out_dir / "scale_eval"            # the OCR cache of the scale pre-pass goes with the outputs
    dataset = SimpleNamespace(native=lambda sid: curated.native(entries[sid]),
                              reference=lambda nat, work, cfg_: curated.reference(nat, work, cfg_, dataset.current),
                              image_path=lambda sid: curated.CURATED / entries[sid]["file"], current=None)
    model = load_model(Path(model_path) if model_path else MODEL)
    engine = pipeline.ocr_engine()
    if ids in ([], ["all"]):                              # every plan of the manifest (plus plans.local.json)
        ids = list(entries)
    failed = []
    for pid in ids:
        if entries[pid].get("_local") and not out_dir.resolve().is_relative_to(DATA.resolve()):
            raise SystemExit(f"{pid}: outputs of local (BBL) plans stay below {DATA} (gitignored)")
        dataset.current = entries[pid]
        try:
            res, sheet = bench.run("curated", pid, model, engine, cfg or DEFAULT, dataset=dataset, out_dir=out_dir)
        except Exception as ex:                          # keep going; the failures are listed at the end
            record_failure(failed, pid, ex)
            continue
        path = out_dir / f"{sheet.id}_bench.json"
        b = json.loads(path.read_text(encoding="utf-8"))
        b["curated_id"] = pid
        path.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
        s = res["scores"]
        shown = {k: s[k] for k in ("wall_iou", "door_iou", "window_iou", "rooms_matched", "rooms_ref") if k in s}
        print(f"{pid:32s} {res['px_per_m']:7.1f} px/m  {len(sheet.rooms):3d} rooms  {shown}  {res['seconds']} s", flush=True)
    if failed:
        (out_dir / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
        print(f"{len(failed)} plans failed (see {out_dir / 'failed.json'})")


def main(argv):
    utf8_stdout()
    if argv[:1] == ["render"]:
        render_all(set(argv[1:]))
        return 0
    if argv[:1] == ["run"]:
        ap = cli("stages 0-10 on curated plans", out=TEST_OUT, threads=12)
        ap.add_argument("ids", nargs="*", help='plan ids ("all" or none: every plan)')
        a = ap.parse_args(argv[1:])
        cfg = setup(a)
        run(a.ids, Path(a.out), a.model, cfg)
        return 0
    errors, warnings, used = curated.validate()
    for w in warnings:
        print(f"WARNING {w}")
    for x in errors:
        print(f"ERROR   {x}")
    summary(used, len(curated.plans()))
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
