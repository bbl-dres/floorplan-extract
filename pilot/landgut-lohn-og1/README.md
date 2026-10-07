# Pilot: Landgut Lohn, 1. OG main building

*October 2026. A first test of the extraction modes discussed in [motivation and goals](../../docs/motivation-goals.md), on one real historic BBL building with a curated reference.*

## Question

Can a VLM, classical CV, or a combination of both extract room outlines, room attributes and openings from (a) a clean CAD print and (b) a historical scan in a completely different drawing style?

## Data

Plan files and everything derived from them stay in `data/`, which is gitignored: the BBL archive plans are non-public.

| Sheet | Source | Style |
|---|---|---|
| S1 | `2051-AA-1. OG 2005-…-ARCH-1-OG-100.pdf` (BBL Planarchiv 2051 AA), rendered at 400 dpi; text layer ignored to simulate a scan | CAD print, 1:100 reduced to A4, outlined walls, door swings, room stamps with areas |
| S2 | `2051-AA-1. OG-54467-0056.tif`, 1-bit, 400 dpi | Historical, "Grundriss vom ersten Stock, Masstab 1:50", solid black walls, no door swings, calligraphic labels, no areas |
| S3 | `2051-AA-1. OG-54477-0066.tif` | Historical; not yet processed |

**Reference:** `plan_topology_og1-main.json` and `plan_calibration.json` from the Landgut Lohn reconstruction. It holds 15 rooms (name, stamp area, polygon), 25 walls, 37 openings (24 in exterior walls, 13 interior) and 3 tiled stoves, in calibrated plan metres. The reference itself was derived from the 2005 PDF and curated by hand.

This README contains room names and areas derived from the non-public plans. Check before publishing the repository.

## Method

- **Mode A (whole page):** the VLM (Claude, cloud) reads a 1500 px image of the whole sheet.
- **Mode B (native resolution):** the VLM reads a crop of the main building at 400 dpi.
- **Mode C (hybrid):** classical CV proposes room polygons. It keeps only wall-thick strokes, closes door gaps by erosion and regrows the regions with a watershed. The VLM then labels a numbered overlay of the candidates (names, merges, rejects), and the stamp area is cross-checked against the polygon area.

Run from `scripts/` in this order (Python 3.14, pymupdf, opencv, numpy, shapely, pillow):

| Script | Purpose |
|---|---|
| `00_reference_summary.py` | Reference rooms, stamp areas, exterior/interior openings |
| `01_render.py` | Render S1 at 400 dpi, crop the main building, previews of all sheets |
| `02_segment_2005.py` | Mode C geometry on S1: parameter sweep, best setting, numbered overlay |
| `03_score_hybrid_2005.py` | Mode C on S1 with the VLM labels; outward-offset test |
| `04_segment_scan_0056.py` | Mode C on S2: scale from the title block, comparison with 2005 |
| `05_export_viewer.py` | Package sheet images and results into `data/out/viewer-data.js` for the viewer |

The VLM readings (labels, merges, rejects, stamp areas) are recorded as data in scripts 03 and 04.

## Viewer

`viewer.html` overlays each original sheet with the extracted rooms, in plan metres (three.js, loaded from a CDN; needs internet).

1. Run `python scripts/05_export_viewer.py` (after scripts 01–04).
2. Open `viewer.html` directly in a browser. The data file embeds the images, so no web server is needed.

Controls:
- **Sheet:** the 2005 CAD print or the historical scan.
- **2D / 3D:** 3D extrudes rooms to 2.8 m and, with the reference on, the 2005 walls to 3.0 m (display heights only).
- **Colour rooms by:** room; confidence (QA signals available in production: the stamp-area check on S1, VLM review and outline regularity on S2); or accuracy against the reference (IoU, evaluation only).
- **Toggles:** extracted walls, room labels, 2005 reference, rejected fragments, the 10 cm wall-face snap (S1), plan opacity.
- **Stair void** (S1): the air space over the stair is shown separately in blue, as it is not floor area.
- **Hover** a room, wall or void for areas, IoU and the reason for its confidence.

The initial state can be set in the URL, e.g. `viewer.html#sheet=s2&view=3d&colour=confidence&ref=1`.

## Results: 2005 CAD print (S1)

| Task | Mode A (whole page) | Mode B (native crop) | Mode C (hybrid) |
|---|---|---|---|
| Room names | 15/15 | 15/15 | 15/15 |
| Stamp areas read | 14/15 | 14/15 (one hidden by a WC symbol) | – |
| Interior doors | not counted reliably | 12 + 1 uncertain (reference 13) | – |
| Openings in exterior walls | – | about 21–23 (reference 24) | – |
| Room polygons | – | – | 14/15 with IoU ≥ 0.5; mean IoU 0.76 after the VLM merges |
| Area error vs stamp | – | – | median 14.7%; 3.8% with a 0.10 m outward snap to the wall face |
| Fragments | – | – | 14 of 33 candidates rejected by the VLM (strips outside the facade) |
| Walls | – | – | 56.8 m² extracted vs 52.3 m² reference, IoU 0.61 |

Walls are the wall-pen strokes plus the enclosed strips between double wall outlines (narrower than about 0.8 m). False positives are tiled stoves, fireplaces and stair stringers, which are drawn with the same heavy pen.

Large rooms reach IoU 0.85–0.94 but come out 5–14% too small: the segmentation stops at the inner edge of the wall pen. Bathrooms lose up to 40% around fixtures, and the stair with its air space needs merging. Windows, balcony doors and interior doors use the same swing symbol, so they can only be told apart by position (exterior wall or not).

## Results: historical scan (S2)

- **Scale:** the title block's 1:50 at 400 dpi gives a building of 22.68 × 15.06 m, against 22.80 × 14.82 m in the reference (−0.5% / +1.6%).
- **Names:** all 11 calligraphic labels were read: Toilette, Diensten-Zimmer, Antichambre, Herren-Zimmer, Halle, Vestibule, Bad, Zimmer (2×), Boudoir, Schlafzimmer.
- **Geometry, unchanged rooms:** six rooms match their 2005 counterparts with IoU 0.82–0.90: Diensten-Zimmer → Kreidolf-Zimmer, Toilette → Bad/WC NW, Halle → Balkon, Zimmer → Biedermeier-Zimmer, Schlafzimmer → Damenzimmer, unlabelled room → Bad/WC SW.
- **Segmentation errors:** rooms joined by wide openings without door leaves merge: stair, vestibule and corridors become one 53.5 m² region, and the middle Zimmer merges with the Boudoir.
- **Walls:** 56 solid wall blocks totalling 26.1 m² vs 52.3 m² in the 2005 reference (IoU 0.36). Window openings are drawn as gaps, so wall segments between openings are separate blocks.
- **Changes in the building:** the 2005 Churchill-Zimmer (38.6 m²) covers the historical Herren-Zimmer (24.1 m²) and adjacent space, likely the Antichambre. The historical Bad (9.6 m²) sits where 2005 has two bathrooms. The scan shows an earlier state of the building.

## Findings

1. **Semantics work.** At native resolution the VLM read every room name, including calligraphy, and 14 of 15 stamp areas. The one miss was a stamp overprinted by a symbol. This is the part the commercial tool in the BBL test (AmpliFY) failed.
2. **Geometry needs pixel-precise CV plus snapping.** Even simple classical CV gives usable outlines (IoU 0.76 on the CAD print, 0.82–0.90 for unchanged rooms on the scan), but areas are only reporting-grade after snapping to wall faces (14.7% → 3.8% median error).
3. **The stamp-area cross-check works as QA.** It flagged 12 of 15 rooms before snapping, which matched the real errors.
4. **Labelling numbered overlays is cheap and effective.** One VLM pass fixed merges, rejected 14 fragments and named every room.
5. **Openings without door leaves break room separation.** Detecting doors and passages is needed both for room connections and for clean room outlines.
6. **The title-block scale was reliable here** (within 1.6%).
7. **Older plans show older buildings.** Plan-vs-reality checks are essential for historical sheets.
8. **Stairs break into several regions.** Flights, landings and the void (Luftraum) come out as separate fragments and have to be merged into one stair object, with the void kept apart because it is not floor area. The stamp "17.98" is printed inside the void, but the reference stair polygon is 23.16 m². The 5.2 m² difference is probably the void, so the stamp likely excludes it (to verify).

## Caveats

- One building, one floor, two sheets. Parameters were tuned on the test sheets themselves, with no held-out set, so the results are optimistic.
- The VLM had already seen the 2005 room names (from the reference and an earlier AmpliFY screenshot), so the S1 readings may be optimistic. The historical names were not known beforehand.
- The VLM was a cloud model. The plans were sent to Anthropic's API at the user's request; a production pipeline would use a self-hosted model.
- Even the reference deviates from its own stamp areas (up to +29% for the stair with its air space), so area comparisons carry this uncertainty.

## Next steps

Continued in [pilot v2](../v2-pipeline/README.md): the graph pipeline below, with a segmenter trained on style-randomised public data.


The pilot points to a structural, graph-based pipeline instead of segmenting rooms directly:

1. **Walls:** wall mask → centre lines (skeleton) → wall graph with junctions and thickness.
2. **Openings:** doors, windows and open passages as gaps between collinear wall ends, classified by symbol and position (exterior wall or not).
3. **Stairs and voids:** detect tread patterns; merge flights and landings into one stair object; keep voids apart.
4. **Rooms:** close the openings and take the faces of the planar wall graph as room outlines, snapped to wall faces.
5. **Room attributes:** read the room stamps (OCR/VLM) and assign them to faces; cross-check stamp areas.
6. **Room connections:** doors and passages become edges of the room-connectivity graph.

Literature to obtain and check before building it (titles from memory, verify):
- Macé et al., "A system to detect rooms in architectural floor plan images" (DAS 2010).
- Ahmed et al., "Automatic room detection and room labeling from architectural floor plans" (DAS 2012).
- de las Heras et al., notation-invariant structural floor plan recognition (IJDAR 2014) and the CVC-FP/SGT dataset paper (IJDAR 2015).
- Dodge et al., "Parsing floor plan images" (MVA 2017).
- Liu et al., "Raster-to-Vector: Revisiting Floorplan Transformation" (ICCV 2017), and Hu et al., Raster-to-Graph (CGF/EG 2024).

Further steps:
- Process scan S3 and the other floors; tune on some sheets and test on others.
- Replace classical CV with a trained detector/segmenter (e.g. RF-DETR) on style-randomised renders.
- Repeat the labelling step with a self-hosted VLM (e.g. Qwen3-VL) and compare it with the cloud VLM.
- Use the 2005 DXF (in `data/inputs`) as an exact vector reference (needs ezdxf).
