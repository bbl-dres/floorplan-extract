# Curated Plans

A hand-picked set of 56 floor plans for the [pilot v2 viewer](../../pilot/v2-pipeline/viewer.html) and for spot checks of the pipeline. The plans are chosen for maximum variety, not as a random sample:
- drawing styles and inputs (scans, photo, born-digital, renders);
- eras and building types, weighted towards federal administrative and public buildings like BBL's portfolio;
- the drawings BBL actually receives: construction drawings (dimension chains, axes, tags, section markers, stage overlays) and FM-style plans (area stamps, safety and security overlays, few dimensions);
- the weak spots named in the [pilot README](../../pilot/v2-pipeline/README.md#findings) and the [review](../../docs/reviews/2026-10-07-pipeline-and-pilot-v2.md#24-benchmark-samples-in-the-viewer): hatching, ornament, columns, open plan, stamps, poché scans.

Only this README and `plans.json` are tracked. The images, the render label maps and `plans.local.json` stay local (gitignored).

**Core set.** 24 plans carry `"core": true`: the ones the [viewer](../../pilot/v2-pipeline/viewer.html) shows and compares across pipeline versions, so that the set stays constant. One plan per challenge, and only plans that may be shown publicly (`use` is never "benchmark only", since the viewer's data is tracked and published): the WAFFLE flats and mosque (baseline poché, breezeway, single-line walls, colour fills, column grid), six US federal construction and FM sheets (dimension chains and tags, blueprint, construction stages, hatched CAD survey, life-safety overlay, stamps in every office), the El Paso courthouse 1888, ten Swiss historical plans (Bundeshaus, Landesmuseum, Rathaus Bern, Schloss Wart, Grand Théâtre, Hallwyl survey, Netstal photo, Neumühle, the Vienna sanatorium, Globus-Heimeli, the Capuchin convent) and one Swiss Dwellings render. The other 32 plans serve spot checks (`curated.py run all`) and the local viewer export (`export_viewer.py --local`); the CVC-FP, CubiCasa5K and FloorPlanCAD plans among them stay out of the public viewer by their licence.

## Files

| File | Content |
|---|---|
| `plans.json` | The manifest: a `meta` branch that documents every field and the controlled vocabularies (as in [research/papers.json](../../research/papers.json)), then one entry per plan |
| `images/<id>.<ext>` | The copy of each plan: byte-identical to its origin, or the Commons file as downloaded; USACE sheets are their PDF pages rendered at 200 dpi (`origin` names the PDF and page; the reduced security exhibit at 600 dpi, because its lettering is 8 px at 200 dpi) |
| `images/<id>_label.png`, `images/<id>_rooms.json` | Renders: the renderer's label map (classes of `fpx.model`) and the floor's room polygons in image pixels. FloorPlanCAD blocks: their six-class label map (no rooms) |
| `plans.local.json` | BBL plans only, same schema; gitignored (see below) |

Per plan the manifest records:
- `source`, `origin` (the original file in `data/`, or the download URL) and `file`;
- `licence` exactly as the source states it (per image for Wikimedia Commons), `attribution` (author, date, page) and `use`, which says whether the image may ever be shown publicly;
- `scale`: px/m and how it is known, or null with `unknown: proposed by the pipeline` and the cues the sheet offers;
- `reference`: what ground truth exists and where it comes from;
- the tags `drawing_type` (construction drawing, FM plan, survey drawing, permit or design drawing, publication plan, marketing plan, diagram, synthetic), `input`, `wall_style`, `era`, `building_type`, `content`, `language` and `challenges`, plus `why` (one sentence) and `category` (the viewer group).

[`pilot/v2-pipeline/scripts/curated.py`](../../pilot/v2-pipeline/scripts/curated.py) validates the manifest against its own vocabularies and loads the plans:

```
cd pilot/v2-pipeline/scripts
python curated.py                    # validate plans.json (+ plans.local.json); exit 1 on errors; counts per tag
python curated.py render [ids]       # draw the synthetic renders again (image, label map, rooms)
python curated.py run ids|all [--model PATH]   # stages 0-10 through bench.py, outputs in data/curated-test
```

In code, `curated.load(id)` returns `(image, px_per_m or None, scale method, reference)`. `curated.bench_loader` has the signature of `bench.py`'s `LOADERS`.

## Categories

| Category | Plans | What they are |
|---|---|---|
| Construction drawings | 9 | USACE A-103 enlarged plan (dimension chains, door tags, section flags), a 1911 federal ink working drawing, a Navy hospital blueprint (inverted), the Stamford post office with 1940–41 construction stages, five FloorPlanCAD blocks (two schools, two hospital blocks, an office) |
| FM and safety plans | 4 | USACE life-safety plan, electronic-security plan (reduced print: its scale note is wrong), two headquarters floors with name and area stamps in every room |
| Administrative and public buildings | 8 | Bundeshaus 1902, Zürich post office, Bern council chamber, Landesmuseum, Grand Théâtre Genève, Göttingen observatory 1900, El Paso post office and courthouse 1888 (Supervising Architect), a HABS CAD survey of an administration building (2011) |
| Hospitals and hotels | 3 | Sporthotel 1900, Hans Auer's Vienna sanatorium 1888 (Allgemeine Bauzeitung plate, two floors), old Kantonsspital Zürich |
| Industry and transport | 2 | Orlando railway station (HABS measured drawing of the Amtrak era), Neumühle mill |
| Monuments and churches | 10 | Schloss Wart, Schloss Hallwyl survey, 17th-century Capuchin convent, Netstal church (photo), six WAFFLE plans (churches, castle, palace, mosque) |
| Housing | 16 | CubiCasa5K (2 per style group), CVC-FP (1 per subset), four WAFFLE flats and houses, Globus-Heimeli 1932 |
| Synthetic renders | 4 | KIT institute office floor, dental clinic, Digital Hub open-plan offices (IFC-Bench), a Swiss Dwellings apartment floor; whole floors at 50 px/m in contrasting styles, with complete label maps |

The construction drawings and FM plans are US federal (public domain) or Chinese (FloorPlanCAD, benchmark only): no Swiss or EU construction drawing with an open licence exists (see `data/benchmark/construction-plans/SOURCE.md`). Their scale is known where the sheet states one and the page size is known (USACE PDFs rendered at 200 dpi: 41–123 px/m) and for FloorPlanCAD (100 px/m); the photographed and Commons sheets leave it to the pipeline.

Resolution: the pipeline works at 50 px/m, and its OCR needs lettering of about 12 px or more. A plan whose native resolution is well below 25 px/m (upsampled more than twice) or whose room lettering is under 10 px does not work and is not kept. The review of October 2026 removed six such plans (Federal Supreme Court print at 18 px/m, Sanatorium Altein at 13 px/m, the Wanner station at under 10 px/m, the first Goetheanum at 14 px/m, the Geneva observatory at 475 px, the Matte power station at 534 px); the small WAFFLE diagrams stay because they come with ground truth and the pipeline scores them. The borderline sheets (Grand Théâtre at about 25 px/m, Capuchin convent at about 18 px/m, Landesmuseum at about 26 px/m, Kantonsspital at about 35 px/m, Pio Monte at 284 px) are kept for what they test, not for their resolution.

Gaps:
- only one plan drawn 1950–1989 (the Orlando station HABS sheet; public-domain plans of that period are rare);
- complex wall hatches (SIA material hatches, insulation bands) in only three plans; the renderer's material style is too faint on the IFC storeys to stand in, so add renders once the renderer draws Werkplan styles;
- no Swiss construction or CAFM plan: BBL's own plans fill these gaps locally (`plans.local.json`).

## Adding a Plan

1. **Copy the image.** Put it into `images/` as `<id>.<ext>`, where `<id>` is a short slug with a source prefix: `cc-`, `cvc-`, `wf-`, `cm-`, `syn-`, or `bbl-` for local plans. Keep the original bytes; for Commons, take the original, or the thumbnail rendition when the original is huge, and record what you took in `origin` and `scale.method`.
2. **Add an entry.** Fill every field of `meta.fields`. Tag values must come from `meta.vocabularies`; add a value there, with its description, only when no existing one fits. `why` is one sentence on what the plan adds to the set.
3. **Validate.** Run `python curated.py`; it must report 0 errors.
4. **Run the pipeline and export.** Run `bench.py` (or `python curated.py run <id>` for a quick test), then `python export_viewer.py`. The viewer groups curated sheets by `category` and shows tags, licence and `why`. It matches outputs to the manifest by `curated_id`, by source `curated` with sid = id, or by the dataset sample id (`cubicasa high_quality/8380`, `commons c02`, ...).

Downloads from Wikimedia Commons: at most one request every 30 s, with a descriptive user agent and no personal data.

## Licences

- **`use` follows from the licence**, and `curated.py` checks the pair:
  - `benchmark only`: CubiCasa5K (CC BY-NC-SA 4.0), CVC-FP (CC BY-NC 4.0) and FloorPlanCAD (CC BY-NC 4.0 for the annotations; the drawings stay with their owners). Evaluation only; never shown publicly, never used to train or select a deployed model without legal clearance (see the [review §6.4](../../docs/reviews/2026-10-07-pipeline-and-pilot-v2.md#64-questions-for-bbls-legal-service)).
  - `public domain or CC0`: may be shown and used freely. This includes the US federal drawings (17 U.S.C. 105, NARA and HABS); keep the USACE castle logo off public copies (a trademark).
  - `attribution`: CC BY or MIT; show with author, licence and source page. The renders are derived from CC BY 4.0 and MIT models.
  - `share-alike`: CC BY-SA or GPL; show with the credit, and adaptations such as overlays fall under the same licence.
- **WAFFLE images** carry the Commons licence of their source file. The licences of the ten WAFFLE plans here were read from their Commons pages in October 2026.
- **Public-domain plans stay out of training anyway.** Like all of `data/benchmark/`, they test unseen drawing styles.

## BBL Plans

BBL plans never go into `plans.json`. They go only into `plans.local.json`:
- same schema, `source` `BBL`, `use` `BBL internal`;
- gitignored, never published, never sent to any external service;
- internal file names and room data never reach a tracked file.

The file holds only `{"plans": [...]}`; the fields and vocabularies come from `plans.json`. A local entry looks like this (placeholders, not a real plan):

```json
{"plans": [{
  "id": "bbl-example-og1", "title": "<building>, 1st floor, survey 1950", "source": "BBL",
  "origin": "data/bbl/<file>", "file": "images/bbl-example-og1.png",
  "licence": "BBL internal", "attribution": {"author": null, "date": "1950", "page": null},
  "use": "BBL internal", "scale": {"px_per_m": null, "method": "unknown: proposed by the pipeline (scale note 1:50)"},
  "reference": {"content": "none", "source": ""}, "category": "Administrative and public buildings",
  "input": "scan 1-bit", "wall_style": ["poché"], "era": "1950–1989", "building_type": "office/administrative",
  "content": ["room stamps"], "language": ["de"], "challenges": ["small rooms"],
  "why": "A late hand-drafted federal office floor, the era the public set lacks."}]}
```

`curated.py` reads `plans.local.json` when it exists. It validates that file against the vocabularies of `plans.json`, and it rejects BBL entries in the tracked manifest. Their images go into `images/` like all others (gitignored). Pipeline outputs of local plans stay under `pilot/v2-pipeline/data/` (gitignored).
