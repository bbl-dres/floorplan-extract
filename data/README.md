# Data

Training and evaluation data for the extraction models. Everything in this folder is gitignored except this README and the curated manifest (`curated/README.md`, `curated/plans.json`).

The folders are split by permitted use (training or evaluation only), not by origin.

| Folder | Content | Use |
|---|---|---|
| `public/` | Licence-clean data that may be used to train a model BBL deploys: Swiss Dwellings v3.0.0 (CC BY 4.0), the 17 permissively licensed IFC-Bench models (CC BY 4.0, CC BY 3.0, MIT; `public/ifc-bench/`, see its SOURCE.md), OFL fonts for synthetic lettering (SIL OFL 1.1; `public/fonts/`); MLSTRUCT-FP once its terms are verified | Training |
| `benchmark/` | Evaluation-only data, never used for training: CubiCasa5K (CC BY-NC-SA 4.0), CVC-FP (CC BY-NC 4.0), the WAFFLE benchmark (per-image Wikimedia Commons licences), the Swiss Commons plans (public domain or CC0, but kept out of training on purpose so they stay a test of unseen drawing styles), FloorPlanCAD (CC BY-NC 4.0, drawings not owned by the authors; official test and train archives in `benchmark/floorplancad/`, real CAD construction drawings), and licence-checked construction drawing sets in `benchmark/construction-plans/`; later e.g. ArchCAD-400K (gated, after legal review) | Benchmarks, the viewer and model selection (the latter a grey zone for non-commercial data, see the [review §6.4](../docs/reviews/2026-10-07-pipeline-and-pilot-v2.md#64-questions-for-bbls-legal-service)) |
| `curated/` | 58 plans of maximum variety for the pilot v2 viewer (incl. construction drawings and FM-style plans), copied from `benchmark/`, Wikimedia Commons and renders of `public/` floors; `plans.json` (tracked) records licence, use, scale, reference and tags per plan; BBL plans only in the gitignored `plans.local.json` ([README](curated/README.md)) | Viewer and spot checks; evaluation only, as `benchmark/` |
| `bbl/` | BBL DWGs, PDFs and scans, plus renders and labels derived from them | Training and evaluation; non-public, process on Swiss infrastructure only |

- Keep a short `SOURCE.md` next to each dataset: origin URL, licence, version or download date.
- Labels follow the [pipeline data model](../docs/pipeline.md#4-data-model); the evaluation sample is described in [pipeline §8](../docs/pipeline.md#8-evaluation).
- Dataset descriptions, licences and recommended use: [public datasets](../docs/datasets-open.md); short summary in [literature review §7](../docs/literature-review.md#7-datasets).
