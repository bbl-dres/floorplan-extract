"""fpeval: what the benchmarks of pilot v2 share, so that every benchmark scores through one function.

    cli         the options every script shares (--model, --out, --config, --threads), console and thread setup,
                and the per-sheet loop's bookkeeping (a failure becomes a row, a skip a reason)
    raster      image loading, resampling to the working resolution with a white pad, class palette, overlays
    metrics     the metrics: rooms, walls, classes, openings, connectivity, regions; pooling over sheets
    oracle      Swiss Dwellings floors rasterised into perfect labels, and their own geometry as the reference
    score       score(sheet, ref, cfg): one row of metrics for a sheet whose stages have run, with the tolerances
    datasets/   one module per dataset, each returning the sheet at the working resolution and one Reference shape
    harness     the harness: context, segmenter cache, modes (oracle, render, cvcfp, waffle), output and tables
    fpcad       the FloorPlanCAD protocol (votes per primitive, F1 per category, PQ, report)

The scripts next to fpx/ (harness.py, cubicasa_eval.py, fpcad_eval.py, bench.py, curated.py, evaluate.py,
export_viewer.py, run_pipeline.py) are thin entry points over these modules.
"""
