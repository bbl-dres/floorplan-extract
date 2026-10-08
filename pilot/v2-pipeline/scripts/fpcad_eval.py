"""FloorPlanCAD semantic symbol spotting with the pilot segmenter, after Fan et al. 2021 (ICCV), §6.1 and Tables 3 and 6.

    python fpcad_eval.py [--split test] [--limit N] [--seed 0] [--model PATH] [--out DIR] [--workers 4] [--threads 2]
                         [--line-px 1.25] [--scale-policy rescale|skip|none] [--overlays 5] [--resume]

FloorPlanCAD is CC BY-NC 4.0 (annotations; drawings stay with their owners): evaluation only, outputs stay in the
gitignored data folder (default data/fpcad/: rows_<tag>.jsonl, summary_<tag>.json, overlays/). The protocol, step
by step, is documented in fpeval.fpcad; the blocks, label ids, scale check, rendering and masks in
fpeval.datasets.fpcad.
"""
from fpeval.fpcad import main

if __name__ == "__main__":
    main()
