# Floor Plan Extraction

<p align="center">
  <img src="assets/social-preview.jpg" width="100%" alt="Abstract architectural painting of fragmented floor-plan lines resolving into blue rooms and a connectivity graph on a cream background">
</p>

Exploring computer-vision extraction of structured floor plan data (rooms, room attributes, walls, openings, stairs, gross floor area) from heterogeneous plan archives for BBL's area management.

This repository brings together research, a target pipeline design and local pilot experiments. The pipeline is under development; the pilots document current findings and limitations.

## Documentation

- [Motivation and goals](docs/motivation-goals.md) — the area-management problem and project goals.
- [Pipeline design](docs/pipeline.md) — the target workflow, from uploaded plans to structured floor data and review.
- [Literature review and state of the art](docs/literature-review.md) — methods, evidence and remaining challenges.
- [Research sources](research/sources.md) — the shared bibliography for the documentation and pilots.

## Experiments

| Pilot | Focus |
|---|---|
| [Landgut Lohn · v1](pilot/landgut-lohn-og1/README.md) | Classical computer vision and a vision-language model on a first-floor plan |
| [Graph pipeline · v2](pilot/v2-pipeline/README.md) | A graph-based pipeline with a segmenter trained on style-randomised Swiss Dwellings renders |

Each pilot includes its own scripts, findings and instructions. Plan data and derived overlays stay local.

## Repository layout

| Folder | Content |
|---|---|
| [`docs/`](docs/) | [Motivation and goals](docs/motivation-goals.md), [pipeline design](docs/pipeline.md), [literature review and state of the art](docs/literature-review.md); solutions: [open](docs/solutions-open.md) and [closed](docs/solutions-closed.md); datasets: [open](docs/datasets-open.md), [open building documentation](docs/datasets-open-docu.md) and [closed](docs/datasets-closed.md); [reviews](docs/reviews/README.md); [research reports](docs/reports/README.md) |
| [`research/`](research/README.md) | The shared [source list](research/sources.md), an index of papers and practitioner sources ([papers.json](research/papers.json)), and scripts to download papers and convert them to Markdown |
| [`pilot/`](pilot/) | Experiments on real plans: [v1](pilot/landgut-lohn-og1/README.md) (classical CV and a VLM on Landgut Lohn, 1. OG) and [v2](pilot/v2-pipeline/README.md) (the graph pipeline with a segmenter trained on style-randomised Swiss Dwellings renders); scripts, findings and overlay viewers |
| [`data/`](data/README.md) | Training and evaluation data (local only) |
| [`assets/`](assets/) | README artwork and social preview |

## Data

Non-public BBL plans and everything derived from them (`pilot/**/data/`, `data/`), downloaded papers, their Markdown conversions and saved web pages stay local and are gitignored.
