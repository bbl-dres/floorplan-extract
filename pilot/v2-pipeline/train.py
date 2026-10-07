"""Train segmenter v2 on renderer-v2 samples: six classes (softmax) plus sigmoid heads for interior, room boundary,
void and text (fpx.model.HEADS), on one U-Net/ResNet-34.

    python train.py --iters 30000 --batch 16 --workers 22 --val 400 --every 2000   # GPU pod, 24 vCPU (public data only)
    python train.py --iters 20 --batch 2 --workers 0 --val 8       # CPU smoke test

Training samples are rendered on the fly from Swiss Dwellings training floors and IFC storeys (held-out IFC projects
excluded), so the model never sees the same image twice. Every worker draws from its own generator seeded with
(--seed, worker id), and the loader returns batches in a fixed order, so the sample stream of a run can be repeated.
Validation uses a fixed set rendered from the validation floors (other sites) and the held-out IFC projects, with all
v2 style elements. Rare classes: floors with columns are oversampled and crops centred on columns or stairs
(--column-floors, --focus-columns, --focus-stairs), and the class weights favour columns and stairs (--class-weights).
An exponential moving average of the weights (--ema) is evaluated and saved.

Writes to --out (default data/model-v2; the v1 model in data/model is left alone):
    segmenter.pt        best EMA checkpoint by mean IoU of the five element classes (v2 format, fpx.segment.load_model)
    segmenter_last.pt   EMA checkpoint at the last iteration
    metrics.json        best and last: per-class IoU overall, per style era, per source and per v2 style element; IoU of
                        the extra heads overall, per era and per source; plus the validation history
    train.log
"""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from common import CLASSES, DATA, HEADS
from fpx.segment import build_model, checkpoint, normalise
from synth import IGNORE, VERSION, Renderer

ELEMENTS = CLASSES[1:]


def sources(args, split):
    """Render sources for a split: Swiss Dwellings floors plus IFC storeys (projects in --ifc-val held out for validation)."""
    sd = DATA / ("floors-train.pkl" if split == "train" else "floors-val.pkl")
    ifc = Path(args.ifc) if args.ifc else DATA / "floors-ifc.pkl"
    if args.ifc_weight <= 0 or not ifc.exists():
        return [(sd, 1.0)]
    held = [p for p in args.ifc_val.split(",") if p]
    sel = {"exclude": held} if split == "train" else {"include": held}
    return [(sd, 1.0 - args.ifc_weight), (ifc, args.ifc_weight, sel)]


def renderer(args, split):
    return Renderer(sources(args, split), args.size, scale=(args.scale_min, args.scale_max), scale_core=args.scale_core,
                    column_floors=args.column_floors, focus=(("columns", args.focus_columns), ("stairs", args.focus_stairs)))


class Synth(IterableDataset):
    def __init__(self, args, seed):
        self.args, self.seed = args, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])     # no clock: a run can be repeated
        r = renderer(self.args, "train")
        failures = 0
        while True:
            try:
                img, lab, _, tg = r.sample(rng, targets=True)
            except Exception as ex:                      # a rare renderer failure must not end a GPU run
                failures += 1
                print(f"worker {info.id if info else 0}: sample skipped ({failures}): {type(ex).__name__}: {ex}", flush=True)
                if failures > 100:
                    raise
                continue
            yield (torch.from_numpy(img).permute(2, 0, 1).contiguous(), torch.from_numpy(lab),
                   torch.from_numpy(np.stack([tg[k] for k in HEADS])))


def worker_init(_):
    cv2.setNumThreads(1)                                 # one renderer per worker; no nested thread pools
    torch.set_num_threads(1)


def val_set(args, n):
    r = renderer(args, "val")
    xs, ys, ts, tags = [], [], [], []
    for i in range(n):
        img, lab, tag, tg = r.sample(np.random.default_rng(10_000 + i), targets=True)
        xs.append(torch.from_numpy(img).permute(2, 0, 1))
        ys.append(torch.from_numpy(lab))
        ts.append(torch.from_numpy(np.stack([tg[k] for k in HEADS])))
        tags.append(tag)
    return torch.stack(xs), torch.stack(ys), torch.stack(ts), tags


def loss_fn(logits, y, t, class_w, head_w, dice):
    """Weighted cross-entropy + Dice over the element classes; per head BCE + soft Dice on known pixels (t != IGNORE)."""
    k = len(CLASSES)
    cl = logits[:, :k]
    loss = F.cross_entropy(cl, y, weight=class_w) + dice(cl, y)
    parts = {"classes": float(loss)}
    for i, h in enumerate(HEADS[:logits.shape[1] - k]):
        w = head_w[h]
        valid = t[:, i] != IGNORE
        if w <= 0 or not valid.any():
            continue
        lg, tf = logits[:, k + i], (t[:, i] == 1).float()
        bce = F.binary_cross_entropy_with_logits(lg[valid], tf[valid])
        hl = bce
        if tf[valid].sum() > 0:
            p = torch.sigmoid(lg) * valid
            hl = hl + 1 - (2 * (p * tf).sum() + 1) / (p.sum() + (tf * valid).sum() + 1)
        loss = loss + w * hl
        parts[h] = float(hl)
    return loss, parts


@torch.no_grad()
def evaluate(model, X, Y, T, tags, device, heads, batch=16):
    """Per-class IoU (overall, per era, per source, per v2 element) and IoU of the heads at p > 0.5 (known pixels)."""
    model.eval()
    k = len(CLASSES)
    total = torch.zeros(k, k, dtype=torch.long)
    groups = {"by_era": {}, "by_source": {}, "by_element": {}}
    hstat = {"all": torch.zeros(len(heads), 2, dtype=torch.long)}
    hgroups = {"by_era": {}, "by_source": {}}
    for i in range(0, len(X), batch):
        x = normalise(X[i:i + batch].to(device)).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x).float()
        pred = out[:, :k].argmax(1).cpu()
        hp = (out[:, k:] > 0).cpu()
        for j in range(len(pred)):
            tg = tags[i + j]
            cm = torch.bincount(Y[i + j].long().flatten() * k + pred[j].flatten(), minlength=k * k).view(k, k)
            total += cm
            keys = {"by_era": [tg["era"]], "by_source": [tg["source"]], "by_element": [e for e in tg["elements"].split(",") if e]}
            for g, names in keys.items():
                for name in names:
                    groups[g][name] = groups[g].get(name, torch.zeros(k, k, dtype=torch.long)) + cm
            if heads:
                t = T[i + j, :len(heads)]
                valid = t != IGNORE
                tp = (t == 1)
                st = torch.stack([(hp[j] & tp & valid).flatten(1).sum(1), ((hp[j] | tp) & valid).flatten(1).sum(1)], 1)
                hstat["all"] += st
                for g in hgroups:
                    name = tg["era"] if g == "by_era" else tg["source"]
                    hgroups[g][name] = hgroups[g].get(name, torch.zeros(len(heads), 2, dtype=torch.long)) + st
    model.train()
    iou = lambda cm: dict(zip(CLASSES, (cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp(min=1)).tolist()))
    hiou = lambda st: {h: (float(st[i, 0]) / float(st[i, 1]) if st[i, 1] > 0 else None) for i, h in enumerate(heads)}
    out = {"iou": iou(total), **{g: {n: iou(cm) for n, cm in sorted(d.items())} for g, d in groups.items()}}
    if heads:
        out["heads"] = hiou(hstat["all"])
        out["heads_by_era"] = {n: hiou(s) for n, s in sorted(hgroups["by_era"].items())}
        out["heads_by_source"] = {n: hiou(s) for n, s in sorted(hgroups["by_source"].items())}
    out["mIoU_elements"] = float(np.mean([out["iou"][c] for c in ELEMENTS]))
    return out


@torch.no_grad()
def ema_update(ema_params, params, decay):
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, params, alpha=1 - decay)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val", type=int, default=400, help="number of fixed validation renders")
    ap.add_argument("--every", type=int, default=2000, help="validate (and maybe save) every N iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "model-v2"))
    ap.add_argument("--no-pretrained", action="store_true", help="random encoder initialisation instead of ImageNet weights")
    ap.add_argument("--heads", default=",".join(HEADS), help="extra heads, a prefix of " + ",".join(HEADS) + " ('none': v1-like)")
    ap.add_argument("--w-interior", type=float, default=0.5, help="loss weight of the interior head")
    ap.add_argument("--w-boundary", type=float, default=1.0, help="loss weight of the room-boundary head")
    ap.add_argument("--w-void", type=float, default=0.5, help="loss weight of the void head")
    ap.add_argument("--w-text", type=float, default=0.3, help="loss weight of the text head")
    ap.add_argument("--class-weights", default="0.5,1,3,3,5,2", help="cross-entropy weights: " + ",".join(CLASSES))
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of the weights (0: off)")
    ap.add_argument("--ifc", default="", help="IFC storeys pickle (default data/floors-ifc.pkl, skipped if missing)")
    ap.add_argument("--ifc-weight", type=float, default=0.1, help="share of samples drawn from IFC storeys")
    ap.add_argument("--ifc-val", default="wbdg_office,digital_hub", help="IFC projects held out for validation")
    ap.add_argument("--scale-min", type=float, default=0.6, help="smallest scale factor around 50 px/m")
    ap.add_argument("--scale-max", type=float, default=1.6, help="largest scale factor around 50 px/m")
    ap.add_argument("--scale-core", type=float, default=0.5, help="share of samples in the v1 range 0.8-1.25")
    ap.add_argument("--column-floors", type=float, default=0.25, help="share of samples from floors that have columns")
    ap.add_argument("--focus-columns", type=float, default=0.2, help="share of crops centred on a column")
    ap.add_argument("--focus-stairs", type=float, default=0.1, help="share of crops centred on a stair")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads of the training process (workers use one each)")
    ap.add_argument("--require-cuda", action="store_true", help="fail instead of falling back to the CPU (use on GPU pods)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a", buffering=1, encoding="utf-8")
    say = lambda *a: (print(*a, flush=True), print(*a, file=log))
    # containers often report the host's CPU count (e.g. 128) but grant a quota (e.g. 18): PyTorch's default thread
    # pool then oversubscribes the CPUs next to the renderer workers, and the training loop starves (14 s/it seen)
    torch.set_num_threads(args.threads)
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available (--require-cuda): check nvidia-smi; a failed GPU makes PyTorch fall back to the CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = [] if args.heads in ("", "none") else args.heads.split(",")
    assert heads == HEADS[:len(heads)], f"--heads must be a prefix of {HEADS}"
    head_w = {"interior": args.w_interior, "boundary": args.w_boundary, "void": args.w_void, "text": args.w_text}
    class_w = torch.tensor([float(v) for v in args.class_weights.split(",")])
    assert len(class_w) == len(CLASSES)
    say(f"device {device} {torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor()} "
        f"renderer {VERSION} heads {heads} args {vars(args)}")

    torch.manual_seed(args.seed)
    model = build_model(pretrained=not args.no_pretrained, heads=heads).to(device).to(memory_format=torch.channels_last)
    ema = model
    if args.ema > 0:
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    ema_state = [v for v in ema.state_dict().values() if v.dtype.is_floating_point]
    live_state = [v for v in model.state_dict().values() if v.dtype.is_floating_point]
    t0 = time.time()
    X, Y, T, tags = val_set(args, args.val)
    say(f"validation set: {len(X)} renders in {time.time() - t0:.0f} s; eras "
        f"{ {e: sum(t['era'] == e for t in tags) for e in sorted({t['era'] for t in tags})} }, "
        f"sources { {s: sum(t['source'] == s for t in tags) for s in sorted({t['source'] for t in tags})} }")
    # forkserver, not fork: the validation renders above start OpenCV's thread pool, and workers forked from that state
    # deadlock on its locks (seen on Linux with renderer 2.0). Windows always spawns.
    context = "forkserver" if args.workers > 0 and platform.system() != "Windows" else None
    loader = DataLoader(Synth(args, args.seed), batch_size=args.batch, num_workers=args.workers, worker_init_fn=worker_init,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                        prefetch_factor=2 if args.workers > 0 else None, multiprocessing_context=context)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=max(0.05, 5 / args.iters))
    dice = smp.losses.DiceLoss("multiclass", classes=list(range(1, len(CLASSES))), from_logits=True)
    cw = class_w.to(device)
    meta = lambda it, m: {"iter": it, "mIoU_elements": m["mIoU_elements"], "renderer": VERSION, "ema": args.ema,
                          "pretrained": not args.no_pretrained, "args": vars(args), "torch": str(torch.__version__),
                          "smp": str(smp.__version__), "date": time.strftime("%Y-%m-%d %H:%M")}
    best, best_m, history, it, it_last = -1.0, None, [], 0, 0
    t0 = t_last = time.time()
    t_data = 0.0
    batches = iter(loader)
    while it < args.iters:
        td = time.time()
        x, y, t = next(batches)
        t_data += time.time() - td
        x = normalise(x.to(device, non_blocking=True)).contiguous(memory_format=torch.channels_last)
        y, t = y.to(device, non_blocking=True).long(), t.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss, parts = loss_fn(logits.float(), y, t, cw, head_w, dice)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        it += 1
        if ema is not model:
            ema_update(ema_state, live_state, min(args.ema, (1 + it) / (10 + it)))
        if it % 100 == 0 or it == args.iters or it <= 3 or (it <= 100 and it % 10 == 0):
            n_it = it - it_last
            say(f"it {it} loss {loss.item():.4f} " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()) +
                f" lr {sched.get_last_lr()[0]:.2e} {(time.time() - t_last) / n_it:.3f} s/it (data wait {t_data / n_it:.3f} s/it)")
            t_last, t_data, it_last = time.time(), 0.0, it
        if it % args.every == 0 or it == args.iters:
            m = evaluate(ema, X, Y, T, tags, device, heads)
            history.append({"iter": it, "mIoU_elements": m["mIoU_elements"], "iou": m["iou"], "heads": m.get("heads")})
            say(f"val it {it} mIoU(elements) {m['mIoU_elements']:.3f} " + " ".join(f"{c} {v:.3f}" for c, v in m["iou"].items()) +
                ("" if not heads else " | heads " + " ".join(f"{h} {v:.3f}" if v is not None else f"{h} -" for h, v in m["heads"].items())))
            if m["mIoU_elements"] > best:
                best, best_m = m["mIoU_elements"], {"iter": it, **m}
                torch.save(checkpoint(ema, heads, **meta(it, m)), out / "segmenter.pt")
            if it == args.iters:
                torch.save(checkpoint(ema, heads, **meta(it, m)), out / "segmenter_last.pt")
            (out / "metrics.json").write_text(json.dumps({**best_m, "last": {"iter": it, **m}, "history": history,
                                                          "args": vars(args), "renderer": VERSION, "val_renders": len(X)},
                                                         indent=1), encoding="utf-8")
    say(f"done best mIoU(elements) {best:.3f} at it {best_m['iter']} in {(time.time() - t0) / 60:.1f} min")
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
