# Demo sheet

The sheet behind the "try a demo floor plan" link of the upload screen. `demo.json` names the file and carries its
title, source and licence; the server serves it through `GET/POST /api/demo` (see `../server.py`).

| File | What | Source | Licence |
|---|---|---|---|
| `cp-usace-bde-hq-1f.png` | Brigade headquarters, Small Brigade First Floor Plan, USACE standard design Rev 7.0 (22 June 2026), 1/8" = 1'-0" on a 34 x 22 in sheet, rendered at 200 dpi (6800 x 4400 px) | U.S. Army Corps of Engineers, Savannah District, Center of Standardization: [standard design PDF](https://mrsi.erdc.dren.mil/content/cos/sas/bn-bde-hq/Library/Standard%20Designs/BDE-BN%20HQ%20Standard%20Design%20Rev%207.0%20dated%2020260622.pdf) | Public domain (17 U.S.C. 105, work of the U.S. Government) |

It is plan `cp-usace-bde-hq-1f` of the curated set (`data/curated/plans.json`), chosen because every room carries a
name and area stamp (the closest public analogue to an FM area plan), the scale is read from the dimension strings,
and the pipeline finds the rooms cleanly. The whole sheet takes about 1.5 minutes on a laptop, mostly OCR.
