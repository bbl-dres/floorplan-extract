# Training Experiments E1–E3 (Renderer 3.0)

*Command lines for the RunPod runs of §6 item 1 of the [code review](../../../docs/reviews/2026-10-07-pilot-v2-code-review.md). Run from `pilot/v2-pipeline/code/` on the pod with the public data in `data/` (`sd_prepare.py` and `ifc_prepare.py` done, CUDA build of torch, [requirements.txt](requirements.txt)). Public data only on the pod; no BBL sheet.*

## What every run below gets from the renderer (3.0)

- **M1** `door()` takes the scale: two leaves from 1.5 m in plan metres (v2 compared pixels, so doors from 0.94 m got two leaves at scale 1.6). Casements: two sashes from 0.9 m, below that one sash narrower than the opening that opens 35–70°, frame and glass lines always at `thin_pen`, sash arcs 1 px or dashed, casement share 0.20 → 0.08. New fifth head **`swing`**: door leaf + swept sector as drawn (`--w-swing 0.5`); the pipeline does not read it yet.
- **M3** void target = `VOID, AIR, SHAFT, ELEVATOR, LIGHTWELL` (shafts and lifts carry the same cross); per-sample Dice for the heads, weight unchanged.
- **M8** the IFC share is the nominal `--ifc-weight` (the source is drawn once, the column floor searched inside it).
- **M4** `--negatives` switches the v3 negative set on (E3 only): dimension chains on the facades and interior strings, section markers, detail bubbles, axis bubbles on plain axes, door/window tags, level markers, furniture. All background; their lettering is in the text head.
- **M7** `--val-cache PATH` writes the 400 validation renders once (key: hash of `synth.py`, renderer options, `--val`) and reuses them; a file made with other settings is refused, so runs that share a path are scored on the same renders.

## Runs

Common options (24-vCPU pod with an RTX 4090; `--workers` = vCPUs − 2):

```
COMMON="--iters 30000 --batch 16 --workers 22 --val 400 --every 2000 --require-cuda"
```

| Run | Command |
|---|---|
| E1a, no heads | `python train.py $COMMON --heads none --out data/model-e1a --val-cache data/val-v3.pt` |
| E1b, heads down-weighted | `python train.py $COMMON --w-boundary 0.2 --w-interior 0.2 --out data/model-e1b --val-cache data/val-v3.pt` |
| **E2**, this renderer, five heads | `python train.py $COMMON --out data/model-e2 --val-cache data/val-v3.pt` |
| **E3**, E2 + negative set | `python train.py $COMMON --negatives --out data/model-e3 --val-cache data/val-v3neg.pt` |

- E1a, E1b and E2 share `data/val-v3.pt` (the key does not include `--heads` or the loss weights); E3 needs its own file because `--negatives` changes the renders. The first run of each writes it (~950 MB for 400 renders at 512 px: image, label and five target maps as uint8; 2–3 minutes); later runs read it.
- `--heads none` still renders the five targets (unused) so that the cache is the same.
- Each run writes `segmenter.pt` (best mean element IoU on the frozen set), `segmenter_last.pt`, `metrics.json` (with `val_key`) and `train.log` to `--out`. Synthetic validation does not select checkpoints for real sheets (MODEL_CARD), so score both checkpoints below.
- Smoke test before the long runs: `python train.py --iters 50 --batch 2 --workers 0 --val 8 --size 256 --every 25 --negatives --val-cache /tmp/val-smoke.pt --out /tmp/smoke` (CPU, about a minute).

## Evaluation per model

```
export FPX_DEVICE=cuda                      # inference on the GPU; the scripts default to the CPU
M=data/model-e2/segmenter.pt; T=e2          # likewise e1a, e1b, e3, and segmenter_last.pt as e2-last
python cubicasa_eval.py test --out data/cubicasa-$T --model $M
python harness.py render --out data/harness/render-$T.json --model $M
python harness.py cvcfp  --out data/harness/cvcfp-$T.json  --model $M
python harness.py waffle --out data/harness/waffle-$T.json --model $M
python fpcad_eval.py --model $M --out data/fpcad-$T --workers 8
```

What to read, against model v2 (MODEL_CARD tables):

- **E1 vs E2** (heads): CubiCasa door / window IoU (v2: 0.451 / 0.608), render harness rooms recall / precision; the void and swing head IoU on the frozen set (`metrics.json`, `heads`).
- **E2 vs v2** (M1, M3): CubiCasa door IoU and the share of ground-truth door pixels predicted as window (v1 → v2: 8.6 % → 18.6 %), door recall (0.643 → 0.524), CVC-FP door-as-window (0.016 → 0.13); void head IoU on the frozen set (v2: 0.26, wandering 0.14–0.36).
- **E3 vs E2** (M4): FloorPlanCAD wall F1 and precision (v2: 0.387 / 0.32) and door F1 (0.108), the WAFFLE false-wall share (11 % on the Aile Richelieu sheet), with no loss on CubiCasa and the render harness.

## Expected runtime and cost

- Training: v2 took 49 min on an RTX 4090 with 22 render workers. The v3 renderer draws about 10 % more per sample without the negatives and 25 % more with them (73 → 81–90 ms per 320 px sample on the development CPU), and the loop is data-bound at about 0.1 s per iteration, so **E1/E2 about 50–55 min, E3 about 55–65 min**, plus 2–3 min for the validation renders on the first run of each cache. About USD 0.6–0.8 per run at the 4090 rate; USD 3–4 for the four runs.
- Evaluation per model on the pod (GPU inference, CPU post-processing): `cubicasa_eval.py` 15–30 min (400 plans through the pipeline), `harness.py render` 10–15 min, `cvcfp` and `waffle` 5–10 min each, `fpcad_eval.py` 20–40 min (5,502 blocks, the door-width scale search re-segments each block, M6 of the review). About 1–2 hours per model; the harness caches segmenter outputs per model path.
