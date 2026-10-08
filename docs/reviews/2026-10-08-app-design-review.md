# Design Review of the Workflow App

*8 October 2026 · [app/](../../app/README.md), the local prototype of the [UX study](../wireframes/261007_Viewer%20and%20Workflow%20UX%20study.html). Reviewed screen by screen as a design pass: what each step shows, consistency of type, spacing and components, and where styles were written by hand instead of coming from tokens. Findings first, then what was changed; the open items close the document.*

## 1. Findings

### 1.1 Steps show more than the decision they ask for

| # | Screen | Finding | Decision |
|---|---|---|---|
| F1 | Step 1, building area | Besides the plan and the area handles the panel showed a "Detected on this sheet" card (input class, resolution, sheet size, title, title block, loader notes) and the drawing list with checkboxes and kind pills, even for a sheet with one drawing. None of it is needed to confirm the area; the resolution reappears in step 2, where it matters. | **Done.** The panel holds the step title, one sentence, and the two buttons. The drawing list appears only when the sheet has more than one drawing (then it decides which one to extract). The detected-sheet card is gone; its one useful line, the resolution and its source, lives in step 2. |
| F2 | Step 2, scale | Already reduced to the plan, the cues on the canvas and the figures that decide the scale. The heading was centred at 19 px while every other step heading is left-aligned at 20 px (an inline style). | **Done.** Same heading as the other steps. |
| F3 | Step 3, run | Was the storeys step with naming, ordering and a run button; the user removed the storey feature (earlier today). The row component kept a four-column grid meant for grip and name input. | **Done.** Plain rows (drawing, status, leave out); the dashed "add" button is a button component now (`.btn-dashed`), not a one-off. |
| F4 | Upload | The pipeline check printed the whole absolute path of the pipeline folder in the summary line. | **Done.** The last three path segments, the full path on hover. |

### 1.2 Consistency

| # | Finding | Decision |
|---|---|---|
| F5 | **Headings on five scales:** panel 20 px, upload 22 px, results head 17 px, section 15 px, inspector 15 px, plus the inline 19 px of step 2; uppercase labels in two near-identical variants (eyebrow .08em accent, section title .06em muted). | **Done.** A type scale in tokens (11, 12, 13, 14, 15, 17, 20, 22 px) with one role each; one uppercase label style with two colours (`.eyebrow`, `.section-title`). |
| F6 | **Button heights** 28, 32, 38, 40, 44 px, two of them inline. | **Done.** Sizes sm/md/lg/xl (28/32/40/44) as tokens; the 38 px "Choose files" button is a `btn-lg`. |
| F7 | **Surfaces:** card, file row, drawing row, storey row, review row, inspector, step card, modal and offline card each declared border, radius and background on their own, with radii 8, 10 and 12 px mixed. | **Done.** One surface rule, radii from four tokens (6/8/10/12), rows share the grid-row pattern. |
| F8 | **Colours written by hand:** 15 hex values and 3 rgba shadows in the stylesheet besides the tokens (#9aa1ab, #b9c4d6, #b4bac3, #c9cdd3, #eef0f2, the badge and box colours, the log and tooltip ink), the accent hover as a bare oklch, and 10 more in the script (confidence and layer colours, the health dot, the needs-review dots, the stamp deviation, the "not extracted" grey). The canvas colours were a second copy of the legend colours. | **Done.** Semantic tokens (success, warning, danger, info, each with bg/text/border where used), neutral tokens (ink, text-disabled, icon-muted, border-dashed, bg-track, surface, on-accent), shadow tokens, and the canvas reads its colours from the tokens (`T('--walls')`), so legend and drawing cannot drift. |
| F9 | **Spacing** by hand: gaps and paddings of 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28 px; 52 inline `style:` attributes in the script for margins, flex rows, widths and weights. | **Done.** A 4 px spacing scale (`--sp-1` … `--sp-7`), utility classes for the layout cases the script needs (`stack`, `row`, `push-end`, `center`, `clip`, `help`, `strong`, `mt-*`), component classes for the rest (`tooltip`, `input-dpi`, `ctl-range`, `pct`, `btn-sm`, `value-accent`, `title-md`, `warn-box.compact`, `spinner.on-primary`, `section.first`, `loader-notes`). Zero static inline styles remain; the two dynamic ones are data (a layer's colour swatch, the confidence bar width). The 6 and 10 px gaps inside dense rows stay as they are: they are component-internal, not layout. |
| F10 | **Focus states** existed for inputs only; buttons and rows had none. | **Done.** `:focus-visible` ring from a token on every focusable element. |
| F11 | The results panel padding (24/28) differed from the step panel (28); the stage overlays used three shadow strengths with no rule. | **Done.** Panel paddings on the spacing scale; shadows sm (pills, round buttons), md (stage toolbar), lg (modal), tip (tooltip). |

### 1.3 Structure

| # | Finding | Decision |
|---|---|---|
| F12 | One stylesheet mixed tokens, components and screens; the tokens were only colours. | **Done.** `static/css/tokens.css` (colour, type, space, radius, shadow, control sizes, layout widths) and `static/css/styles.css` (base, utilities, components, screens), nothing else writes a value. |
| F13 | The server served `static/<name>` only, flattening paths, so a `css/` folder could not be served; the handler also had no guard against `..`. | **Done.** Any path below `static/` is served, resolved and checked to stay inside the folder (a traversal attempt answers 404). |

## 2. What the user sees now

- Step 1: the sheet with the detected regions and the area handles, a one-sentence instruction, "Skip this sheet" and "Confirm area". The drawing list only when there is a choice.
- Step 2: unchanged in content, the heading now aligned with the other steps.
- Step 3: the confirmed drawings as a list with status, the run button and the note. *(Later the same day the step was folded into the results screen: the scale confirmation starts the extraction, the results screen shows the progress.)*
- Results: unchanged in layout; the confidence legend, the needs-review dots and the canvas use the same tokens.
- Everywhere: one type scale, one spacing scale, one surface style, focus rings.

## 3. Open Items

- The language switch and the settings button in the top bar are decoration from the study (disabled); either wire them or remove them before a wider test.
- The region overlays of step 1 (notes, legend, caption, north arrow) label every box; on dense sheets the labels compete with the plan. A hover-only label would calm the view.
- The results screen keeps two data-driven inline styles (layer swatch colour, confidence bar width); a CSS custom property per element would make even those token-based.
- Dark mode is not covered by the tokens yet (the pilot viewer has one); the token file is the place to add it.
- The code review that follows this pass covers the script's structure (one 1,000-line file), robustness and the server.
