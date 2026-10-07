# Reports

Research reports behind the design documents and reviews: what was searched, what was found, and the evidence in full. They are kept as delivered, with a status line on top; their conclusions live in the [reviews](../reviews/README.md), the [literature review](../literature-review.md) and the [pipeline design](../pipeline.md). The newest report comes first.

| Date | Report | Question | Used in |
|---|---|---|---|
| 2026-10-07 | [Published baselines](2026-10-07-published-baselines.md) | Which published results on FloorPlanCAD, CubiCasa5K, WAFFLE and CVC-FP can our zero-shot numbers be compared with, under which protocol, and what must we do to make them comparable? | [Model card](../../pilot/v2-pipeline/MODEL_CARD.md#evaluation); `fpcad_eval.py`, `cubicasa_eval.py`, `harness.py` |
| 2026-10-07 | [Centre-line (BIM-style) reconstruction](2026-10-07-centre-line-bim-reconstruction.md) | Should walls be reduced to centre lines and rooms placed like in a BIM authoring tool, with openings on the wall axis and stairs as non-bounding objects? What does the literature say about producing such a graph, closing gaps, open plan, stairs and voids? | Analysis for the next pilot iteration; [pipeline.md](../pipeline.md) stages 3–6 |
| 2026-10-07 | [Construction drawings and FM floor plans](2026-10-07-construction-and-fm-drawings.md) | Do construction drawings and facility-management plans need their own data? Which datasets and licence-clean drawings exist, what do Swiss plans contain, and can BIM models generate construction-style drawings? | [Pipeline review §6](../reviews/2026-10-07-pipeline-target-inputs-masking-scale.md#6-drawing-types-construction-drawings-and-fm-plans); `data/benchmark/construction-plans/`, `data/benchmark/floorplancad/sample/` |
| 2026-10-07 | [Inputs, masking and scale: evidence brief](2026-10-07-pipeline-inputs-masking-scale-evidence.md) | How should the pipeline handle any upload (raster, PDF, DWG), isolate the drawing on a sheet, and find its scale? 49 new papers and 28 standards and tools | [Review of the pipeline as canonical target](../reviews/2026-10-07-pipeline-target-inputs-masking-scale.md); [pipeline.md](../pipeline.md) draft 0.3 |
| 2026-10-07 | [Missing papers](2026-10-07-missing-papers.md) | Which impactful papers does the project not yet have? | [Research index](../../research/README.md); [literature review](../literature-review.md) |

Reports may name sources that later turned out to be wrong or were corrected; the status line of each report lists the known corrections. Figures are the authors' own unless marked.
