# three.js (vendored)

three.js **0.186.1**, MIT licence (`LICENSE`), from the npm package via jsDelivr (`https://cdn.jsdelivr.net/npm/three@0.186.1/`):

| File | Origin |
|---|---|
| `three.module.min.js` | `build/three.module.js`, minified by jsDelivr (Terser; the header says so); it imports `./three.core.js` |
| `three.core.js` | `build/three.core.js`, minified by jsDelivr the same way (fetched as `three.core.min.js`, stored under the name the module build imports) |
| `OrbitControls.js` | `examples/jsm/controls/OrbitControls.js`, as published; it imports the bare specifier `three` |

`index.html` maps `three` to `three.module.min.js` with an import map. Only `../../view3d.js` (the 3D view of the
results) uses the library, and it loads on the first switch to 3D, so the rest of the app stays free of it.

To update: fetch the same three files of a newer release, keep the names above, and change the version here and in
`docs/reviews/2026-10-08-app-ux-review-2.md`.
