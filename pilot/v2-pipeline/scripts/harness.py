"""Evaluation harness on public data: the whole pipeline and the post-processing alone, on perfect labels, rendered
floors, CVC-FP scans and WAFFLE images (fpeval.harness).

    python harness.py oracle [--n N]                    # Swiss Dwellings test floors with perfect labels: post-processing alone
    python harness.py render [--n N] [--seed S] [--ocr [K]]   # the same floors rendered whole in random styles: full pipeline
    python harness.py cvcfp [--n N]                     # all CVC-FP scans, scale from the reference doors
    python harness.py waffle [--n N] [--scale proposed|reference]   # WAFFLE benchmark images (masks only)

Options: --config cfg.json, --out file, --ids a,b, --model path, --threads 8, --no-cache, --rerender. Writes
data/harness/<mode>.json (arguments, configuration, summary, groups, by_area_type, one row per sheet) and prints a
summary table. Modes, metrics and the cache: fpeval.harness; the scorer: fpeval.score.
"""
from fpeval.harness import main

if __name__ == "__main__":
    main()
