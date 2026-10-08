/* Results (board 1i of the UX study): layers left, figures and rooms right, both panels collapsible; a room pinned on the
   canvas opens a card. While the extraction runs the same screen shows the work in progress: the drawing at its working
   resolution, and walls, openings, stairs and rooms appearing on it as the pipeline finds them. Finished results can
   be seen in 3D (view3d.js, three.js vendored, loaded on demand). */
import { ACCENT, LAYER_COLOURS, NEUTRAL, PAL, T, confColour, fmt, getJSON, h, put, stageInfo } from '../core.js';
import { J, P, bboxOf, busy, drawingRows, keyOf, packagesOf, polygonOf, ppmOf, refreshJob, saveProject, screenName, startPolling } from '../state.js';
import { badge, beforeUnmount, errorBox, focusHeading, go, mount, notify, post, projectName, setDownloads, useViewer } from '../ui.js';
import { drawPolyWithHoles, hexA, inPoly, labelPoint, scaled } from '../viewer.js';

// --- state kept across renders (and, for the panels and layers, across sessions) ---------------------------------------
const RV = { colour: 'room', layers: { walls: true, doors: true, windows: true, passages: true, columns: true, stairs: true, rooms: true, labels: true, graph: false, gf: true, overlay: false },
  selected: null, opacity: 1, shownKey: null, left: true, right: true, sort: 'confidence', view: '2d', tab: 'rooms' };
try { Object.assign(RV, JSON.parse(localStorage.getItem('fpx.results') || '{}'), { selected: null, shownKey: null }); } catch (e) { /* defaults */ }
function remember() { try { localStorage.setItem('fpx.results', JSON.stringify({ colour: RV.colour, layers: RV.layers, opacity: RV.opacity, left: RV.left, right: RV.right, sort: RV.sort, view: RV.view, tab: RV.tab })); } catch (e) { /* private mode */ } }
const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const GROUPS = [
  ['Rooms', [['rooms', 'Room fill'], ['labels', 'Room labels'], ['graph', 'Room connections']]],
  ['Structure', [['walls', 'Walls'], ['doors', 'Doors'], ['windows', 'Windows'], ['passages', 'Passages'], ['columns', 'Columns'], ['stairs', 'Stairs and voids']]],
  ['Analysis', [['gf', 'GF outline'], ['overlay', 'Segmentation overlay']]],
];
const EXTRACT_STAGES = 12;                                   // from the text OCR to the export

let live = null;                                             // the live poll of the drawing on show: { key, timer, last, work }
function stopLive() { if (live && live.timer) clearInterval(live.timer); live = null; }
let v3 = null;                                               // the 3D view on show (view3d.js), or null
function stop3D() { if (v3) { v3.destroy(); v3 = null; } }

/** The step cursor of a drawing row (for the area and scale steps). */
function cursorOf(row) { return { sheet: P.sheets.indexOf(row.id), pkg: packagesOf(row.job).indexOf(row.pkg), drawing: row.pkg.drawings.indexOf(row.dr) }; }
/** Queue the extraction of the row's sheet and show its progress. */
async function startRun(row) {
  if (!await post(`/api/sheets/${row.id}/run`, {})) return;
  await refreshJob(row.id);
  P.resultsKey = row.key; saveProject(); renderResults();
}
function scaleText(r) {
  const ppm = ppmOf(r.job, r.pkg, r.dr); const dpi = r.conf.dpi || r.pkg.summary.dpi;
  if (!ppm) return 'no scale';
  return dpi ? fmt.scaleN(dpi / 0.0254 / ppm).replace(/ /g, '') + (r.conf.scale_source === 'measured' ? ' (measured)' : '') : `${ppm.toFixed(0)} px/m`;
}
/** The state of a drawing's extraction as a badge: its stage while it runs, finished with its time, failed; null before the run. */
function runStatus(r) {
  const j = r.job; const key = keyOf(r.pkg, r.dr);
  if (busy(j)) { const p = j.progress; const mine = p && (p.key === key || p.key === r.pkg.id); return badge(mine ? 'run' : 'wait', mine ? stageInfo(p.stage).label : j.status === 'running' ? 'waiting' : j.status); }
  if (r.res && r.res.outputs) return badge('ok', `Finished · ${fmt.secs(r.res.times && r.res.times.total)}`);
  if (r.res && (r.res.error || r.res.skipped)) return badge('err', 'Failed', r.res.error || r.res.skipped);
  if (j.status === 'error') return badge('err', 'Error');
  return null;
}

// --- the scene: everything drawn on the canvas, in working px of the working image -----------------------------------
function emptyScene() { return { frame: 'sheet', walls: [], openings: [], stairs: [], voids: [], columns: [], rooms: [], gf: [], graph: [], t0: {} }; }
const ringsOf = (g) => !g ? [] : g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
/** The final results as a scene: plan metres -> working px of the working image. */
function sceneFromResults(D) {
  const ppm = D.px_per_m, ex = D.extent;
  const W = (X, Y) => [(X - ex[0]) * ppm, (ex[3] - Y) * ppm];
  const polys = (g) => ringsOf(g).map((poly) => poly.map((ring) => ring.map(([X, Y]) => W(X, Y))));
  const s = emptyScene(); s.frame = 'work'; s.done = true;
  s.walls = D.walls.flatMap((w) => polys(w));
  s.columns = D.columns.flatMap((o) => polys(o));
  s.stairs = D.stairs.flatMap((o) => polys(o));
  s.voids = D.voids.map((v) => ({ ...v, polys: polys(v.geometry) }));
  s.openings = D.openings.map((o) => ({ ...o, polys: polys(o.geometry) }));
  s.rooms = D.rooms.map((r, i) => { const p = polys(r.geometry); return { ...r, i, polys: p, colour: PAL[i % PAL.length], centre: labelPoint(p[0] ? p[0][0] : []) }; });
  s.gf = D.floor.gf ? polys(D.floor.gf) : [];
  s.graph = D.connectivity;
  return s;
}
/** A live record (bridge._live) into the scene; layers that just appeared get their fade-in start. */
function applyLive(s, L) {
  const now = performance.now();
  const one = (rings) => rings.map((r) => [r]);
  const set = (layer, value, has) => { if (has && !s.t0[layer]) s.t0[layer] = now; s[layer] = value; };
  s.frame = 'work';
  set('walls', one(L.walls || []), (L.walls || []).length);
  set('openings', (L.openings || []).map((o) => ({ kind: o.kind, polys: one(o.rings || []) })), (L.openings || []).length);
  set('stairs', one(L.stairs || []), (L.stairs || []).length);
  set('rooms', (L.rooms || []).map((r, i) => ({ i, name: r.name, polys: one(r.rings || []), colour: PAL[i % PAL.length], confidence: 'medium', centre: labelPoint(r.rings && r.rings[0] ? r.rings[0] : []) })), (L.rooms || []).length);
  set('gf', one(L.gf || []), (L.gf || []).length);
}
function alphaOf(s, layer) { if (s.done || REDUCED || !s.t0[layer]) return 1; return Math.min(1, (performance.now() - s.t0[layer]) / 700); }
function animate(viewer, ms = 800) { const t0 = performance.now(); const step = () => { viewer.draw(); if (performance.now() - t0 < ms) requestAnimationFrame(step); }; requestAnimationFrame(step); }
const kindOn = (o) => (o.kind === 'window' && RV.layers.windows) || ((o.kind === 'door' || o.kind === 'exterior door') && RV.layers.doors) || ((o.kind === 'passage' || o.kind === 'interior opening') && RV.layers.passages);
const kindColour = (o) => o.kind === 'window' ? LAYER_COLOURS.windows : (o.kind || '').includes('door') ? LAYER_COLOURS.doors : LAYER_COLOURS.passages;
const roomColour = (r) => RV.colour === 'confidence' ? confColour(r.confidence) : r.colour;
const roomName = (r) => r.name || (r.small_region ? 'small region' : '');

// --- the screen -----------------------------------------------------------------------------------------------------
async function renderResults({ changed } = {}) {
  const rows = drawingRows();
  if (!rows.length) return go('upload');
  const row = rows.find((r) => r.key === P.resultsKey) || rows.find((r) => r.res && r.res.outputs) || rows[0];
  // a poll: only this row's job matters; a finished view on show stays (it would lose selection and zoom), and a
  // running one only refreshes its progress panel
  if (changed) {
    if (!changed.includes(row.id)) return;
    if (RV.shownKey === row.key && !busy(row.job)) return;
    if (live && live.key === row.key && busy(row.job)) { live.drawRight(); return; }
  }
  P.resultsKey = row.key; saveProject();
  stopLive(); stop3D();
  const done = !busy(row.job) && !!(row.res && row.res.outputs);
  RV.shownKey = done ? row.key : null; RV.selected = null;

  // --- layout: layers | canvas | figures and rooms, both sides collapsible
  const left = h('aside', { class: 'side left', hidden: !RV.left, 'aria-label': 'Layers' });
  const stage = h('main', { class: 'stage grid-bg' });
  const right = h('aside', { class: 'side right', hidden: !RV.right, 'aria-label': 'Results' });
  const wrap = h('div', { class: 'results' + (RV.left ? '' : ' no-left') + (RV.right ? '' : ' no-right') }, left, stage, right);
  mount(wrap, projectName());
  beforeUnmount(() => { stopLive(); stop3D(); });
  const viewer = useViewer(stage, done ? 'The extracted plan: walls, openings, stairs and rooms over the drawing' : 'The drawing while it is extracted');
  const toggle = (side) => {
    RV[side] = !RV[side]; remember();
    (side === 'left' ? left : right).hidden = !RV[side];
    wrap.classList.toggle('no-' + side, !RV[side]);
    edges[side].textContent = side === 'left' ? (RV.left ? '‹' : '›') : (RV.right ? '›' : '‹');
    edges[side].setAttribute('aria-expanded', String(RV[side]));
    edges[side].setAttribute('aria-label', `${RV[side] ? 'Hide' : 'Show'} the ${side === 'left' ? 'layers' : 'results'} panel`);
  };
  const edges = {
    left: h('button', { class: 'edge-toggle edge-left', 'aria-expanded': String(RV.left), 'aria-label': `${RV.left ? 'Hide' : 'Show'} the layers panel`, onclick: () => toggle('left') }, RV.left ? '‹' : '›'),
    right: h('button', { class: 'edge-toggle edge-right', 'aria-expanded': String(RV.right), 'aria-label': `${RV.right ? 'Hide' : 'Show'} the results panel`, onclick: () => toggle('right') }, RV.right ? '›' : '‹'),
  };
  stage.append(edges.left, edges.right);
  const opacity = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: RV.opacity, class: 'ctl-range', 'aria-label': 'Plan opacity', oninput: (e) => { RV.opacity = +e.target.value; viewer.imgAlpha = RV.opacity; pct.textContent = `${Math.round(RV.opacity * 100)}%`; remember(); viewer.draw(); if (v3) v3.setOpacity(RV.opacity); } });
  const pct = h('span', { class: 'pct' }, `${Math.round(RV.opacity * 100)}%`);
  const viewSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'View' },
    h('button', { class: RV.view === '2d' || !done ? 'on' : '', 'aria-pressed': String(RV.view === '2d' || !done), onclick: () => setView('2d') }, '2D'),
    h('button', { class: RV.view === '3d' && done ? 'on' : '', 'aria-pressed': String(RV.view === '3d' && done), disabled: !done, title: done ? 'Walls, openings and stairs as prisms over the drawing (heights are nominal)' : 'After the extraction', onclick: () => setView('3d') }, '3D'));
  stage.append(h('div', { class: 'stage-bottom' }, viewSeg, h('div', { class: 'vsep' }), h('div', { class: 'row' }, h('span', { class: 'muted' }, 'Plan'), opacity, pct),
    h('div', { class: 'vsep' }), h('button', { class: 'btn-link text', title: 'Fit the drawing into the view', onclick: () => { if (v3) v3.fit(); else { viewer.fit(); viewer.draw(); } } }, 'Fit')));
  const popup = h('div', { class: 'popup', hidden: true, role: 'dialog', 'aria-label': 'Room' });
  stage.append(popup);
  viewer.imgAlpha = RV.opacity;

  // --- data: the results when done, the live record while it runs
  let D = null, scene = emptyScene(), overlayImg = null;
  const f = row.pkg.preview.scale;
  if (done) {
    try { D = await getJSON(`/api/sheets/${row.id}/results?key=${encodeURIComponent(keyOf(row.pkg, row.dr))}`); }
    catch (e) { right.append(h('div', { class: 'error-box' }, e.message)); return; }
    if (D.error) { right.append(h('div', { class: 'error-box' }, D.error)); return; }
    scene = sceneFromResults(D);
    const home = scene.gf.length ? bboxOf(scene.gf[0][0]) : scene.walls.length ? bboxOf(scene.walls.flatMap((p) => p[0])) : null;
    viewer.load(`/api/sheets/${row.id}/file/${D.files.work}`, home && { ...home, pad: 60 }).catch((e) => right.prepend(h('div', { class: 'error-box' }, e.message)));
    setDownloads(downloads(row, D));
  } else {
    const pb = bboxOf(scaled(polygonOf(row.job, row.pkg, row.dr), f));              // the sheet, framed on the drawing, until the working image exists
    viewer.load(`/api/sheets/${row.id}/file/${row.pkg.preview.file}`, { ...pb, pad: 40 }).catch(() => {});
    setDownloads(null);
  }
  const review = D && D.review && D.review.rooms ? D.review.rooms : {};
  const needs = D ? needsReview(D, scene) : [];

  // --- 2D | 3D: the canvas stays as it is; the 3D view (three.js, loaded on first use) sits over it
  async function setView(mode) {
    if (mode === '3d' && !done) return;
    RV.view = mode; remember();
    for (const b of viewSeg.children) { const on = b.textContent === (mode === '3d' ? '3D' : '2D'); b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); }
    if (mode === '2d') { stop3D(); viewer.canvas.hidden = false; viewer.resize(); drawPopup(); return; }
    let mod;
    try { mod = await import('../view3d.js'); } catch (e) { RV.view = '2d'; remember(); notify(`The 3D view could not be loaded: ${e.message}`); return setView('2d'); }
    if (RV.view !== '3d' || !scene.done) return;
    stop3D(); viewer.canvas.hidden = true;
    v3 = mod.create3D(stage, scene, D.px_per_m, {
      imageUrl: `/api/sheets/${row.id}/file/${D.files.work}`, size: D.work_px || [viewer.img ? viewer.img.width : 1000, viewer.img ? viewer.img.height : 1000],
      colours: { walls: T('--walls-3d', '#d6d9df'), doors: LAYER_COLOURS.doors, windows: LAYER_COLOURS.windows, passages: LAYER_COLOURS.passages, columns: LAYER_COLOURS.columns, stairs: LAYER_COLOURS.stairs, voids: LAYER_COLOURS.voids, gf: LAYER_COLOURS.gf },
      roomColour, layers: RV.layers, opacity: RV.opacity, onSelect: (id) => select(id), onRender: () => placePopup(viewer) });
    v3.setLayers(RV.layers); v3.select(RV.selected);
  }

  // --- canvas
  viewer.overlay = (c, v) => {
    if (scene.frame === 'sheet') { v.poly(scaled(polygonOf(row.job, row.pkg, row.dr), f), { stroke: ACCENT, width: 1.5, dash: [6, 4] }); return; }
    const layer = (name, on, draw) => { if (!on) return; const a = alphaOf(scene, name); if (a <= 0) return; c.globalAlpha = a; draw(); c.globalAlpha = 1; };
    if (RV.layers.overlay && overlayImg && overlayImg.complete) { const [x, y] = v.toScreen(0, 0); c.globalAlpha = 0.85; c.drawImage(overlayImg, x, y, overlayImg.width * v.view.s, overlayImg.height * v.view.s); c.globalAlpha = 1; }
    layer('walls', RV.layers.walls, () => { for (const poly of scene.walls) drawPolyWithHoles(c, v, poly, hexA(LAYER_COLOURS.walls, .85), null); });
    layer('columns', RV.layers.columns, () => { for (const poly of scene.columns) drawPolyWithHoles(c, v, poly, LAYER_COLOURS.columns, null); });
    layer('stairs', RV.layers.stairs, () => { for (const poly of scene.stairs) drawPolyWithHoles(c, v, poly, hexA(LAYER_COLOURS.stairs, .35), LAYER_COLOURS.stairs); for (const o of scene.voids) for (const poly of o.polys) drawPolyWithHoles(c, v, poly, hexA(LAYER_COLOURS.voids, .3), LAYER_COLOURS.voids, [6, 4]); });
    layer('rooms', RV.layers.rooms, () => { for (const r of scene.rooms) { const col = roomColour(r); for (const poly of r.polys) drawPolyWithHoles(c, v, poly, hexA(col, .38), col, null, 1.5); } });
    layer('openings', true, () => { for (const o of scene.openings) { if (!kindOn(o)) continue; const col = kindColour(o); for (const poly of o.polys) drawPolyWithHoles(c, v, poly, hexA(col, .75), col); } });
    layer('gf', RV.layers.gf, () => { for (const poly of scene.gf) drawPolyWithHoles(c, v, poly, null, LAYER_COLOURS.gf, [8, 5], 2); });
    if (RV.layers.graph && scene.done) {
      const byId = Object.fromEntries(scene.rooms.map((r) => [r.id, r])); const op = Object.fromEntries(scene.openings.map((o) => [o.id, o]));
      for (const e of scene.graph) { const a = byId[e.a], b = byId[e.b], o = op[e.opening]; if (!a || !b || !a.centre || !b.centre) continue; const mid = o && o.polys[0] ? labelPoint(o.polys[0][0]) : [(a.centre[0] + b.centre[0]) / 2, (a.centre[1] + b.centre[1]) / 2]; v.poly([a.centre, mid, b.centre], { stroke: LAYER_COLOURS.graph, width: 1.5, close: false }); }
    }
    if (hovered && hovered !== RV.selected) { const r = scene.rooms.find((x) => x.id === hovered); if (r) for (const poly of r.polys) v.poly(poly[0], { stroke: hexA(ACCENT, .7), width: 2 }); }
    if (RV.selected) { const r = scene.rooms.find((x) => x.id === RV.selected); if (r) for (const poly of r.polys) { v.poly(poly[0], { stroke: 'rgba(255,255,255,.95)', width: 7 }); v.poly(poly[0], { stroke: ACCENT, width: 2.5 }); } }
    if (RV.layers.labels && scene.done && v.view.s * (D ? D.px_per_m : 1) > 12) for (const r of scene.rooms) {
      if (!r.centre) continue; const [sx, sy] = v.toScreen(r.centre[0], r.centre[1]);
      c.textAlign = 'center'; c.textBaseline = 'middle'; c.lineWidth = 3; c.strokeStyle = 'rgba(255,255,255,.9)';
      c.font = `600 11px 'IBM Plex Sans', system-ui, sans-serif`; c.strokeText(roomName(r), sx, sy - 6); c.fillStyle = '#1a1d21'; c.fillText(roomName(r), sx, sy - 6);
      if (r.area_net != null) { c.font = `10px 'IBM Plex Sans', system-ui, sans-serif`; c.strokeText(fmt.m2(r.area_net), sx, sy + 7); c.fillStyle = '#646b74'; c.fillText(fmt.m2(r.area_net), sx, sy + 7); }
    }
    placePopup(v);
  };
  let hovered = null;
  // the room under a point (a hidden room layer cannot be picked)
  const roomAt = (ip) => { if (!RV.layers.rooms) return null; for (let i = scene.rooms.length - 1; i >= 0; i--) { const r = scene.rooms[i]; for (const poly of r.polys) if (inPoly(ip, poly[0]) && !poly.slice(1).some((hole) => inPoly(ip, hole))) return r; } return null; };
  viewer.onClick = (ip) => { if (!scene.done) return; const r = roomAt(ip); select(r ? r.id : null); };
  viewer.tool = { hover: (ip, sp, v) => { if (!scene.done) return; const r = roomAt(ip); const id = r ? r.id : null; v.canvas.style.cursor = r ? 'pointer' : 'grab'; if (id !== hovered) { hovered = id; v.draw(); } } };

  // --- the pinned room card on the canvas
  /** The selected room's box on screen (2D: the canvas transform; 3D: the projected outline), or null. */
  function roomBox(r) {
    const pts = r.polys[0] ? r.polys[0][0] : [];
    const sp = v3 ? pts.map((q) => v3.projectPoint(q[0], q[1])).filter(Boolean) : pts.map(([x, y]) => viewer.toScreen(x, y));
    return sp.length ? bboxOf(sp) : null;
  }
  /** The card sits above the room, else beside it (right, then left), else below: never over the room itself. */
  function placePopup(v) {
    if (popup.hidden) return;
    const r = scene.rooms.find((x) => x.id === RV.selected);
    const b = r && roomBox(r);
    if (!b) { popup.hidden = true; return; }
    const w = popup.offsetWidth || 250, hh = popup.offsetHeight || 120, gap = 10, m = 8;
    const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    let left, top;
    if (b.y0 - gap - hh >= m) { top = b.y0 - gap - hh; left = cx - w / 2; }
    else if (b.x1 + gap + w <= v.w - m) { left = b.x1 + gap; top = cy - hh / 2; }
    else if (b.x0 - gap - w >= m) { left = b.x0 - gap - w; top = cy - hh / 2; }
    else { top = b.y1 + gap; left = cx - w / 2; }
    popup.style.left = `${Math.max(m, Math.min(v.w - w - m, left))}px`;
    popup.style.top = `${Math.max(m, Math.min(v.h - hh - m, top))}px`;
  }
  function drawPopup() {
    const r = scene.rooms.find((x) => x.id === RV.selected);
    if (!r) { popup.hidden = true; return; }
    const dev = r.area_deviation_pct;
    put(popup, h('div', { class: 'head' }, h('b', null, roomName(r) || '(no stamp)'), h('button', { class: 'btn-x', 'aria-label': 'Close', onclick: () => select(null) }, '×')),
      h('div', { class: 'kv-grid small' },
        h('span', null, 'Net area'), h('span', null, fmt.m2(r.area_net)),
        h('span', null, 'Stamp'), h('span', null, r.area_stamp != null ? [fmt.m2(r.area_stamp), ' ', h('span', { class: Math.abs(dev || 0) > 5 ? 'danger-text' : 'muted' }, dev != null ? `(${fmt.pct(dev)})` : '')] : 'none read'),
        r.number ? h('span', null, 'Number') : null, r.number ? h('span', null, r.number) : null,
        h('span', null, 'Confidence'), h('span', { class: 'conf-' + r.confidence + ' strong' }, r.confidence)));   // the reasons and the check box live in the Rooms and Issues tabs
    popup.hidden = false;
    placePopup(viewer);
  }
  function select(id) {
    RV.selected = id; drawPopup(); drawRooms(); viewer.draw(); if (v3) v3.select(id);
    if (id) { const el = right.querySelector(`[data-id="${id}"]`); if (el) el.scrollIntoView({ block: 'nearest' }); }
  }
  async function setChecked(r, checked) {
    review[r.id] = Object.assign(review[r.id] || {}, { checked });
    await post(`/api/sheets/${row.id}/review`, { key: keyOf(row.pkg, row.dr), rooms: { [r.id]: { checked } } });
    drawFigures(); drawRooms(); drawPopup();
  }
  document.onkeydown = (e) => { if (e.key === 'Escape' && screenName === 'results' && RV.selected) select(null); };

  // --- left: layers
  const counts = () => ({ walls: scene.walls.length, doors: scene.openings.filter((o) => (o.kind || '').includes('door')).length, windows: scene.openings.filter((o) => o.kind === 'window').length,
    passages: scene.openings.filter((o) => o.kind === 'passage' || o.kind === 'interior opening').length, columns: scene.columns.length, stairs: scene.stairs.length + scene.voids.length,
    rooms: scene.rooms.length, labels: null, graph: scene.graph.length, gf: scene.gf.length, overlay: null });
  function drawLeft() {
    const n = counts();
    const all = (on) => { for (const k of Object.keys(RV.layers)) if (k !== 'overlay') RV.layers[k] = on; remember(); drawLeft(); viewer.draw(); if (v3) v3.setLayers(RV.layers); };
    const conf = { high: 0, medium: 0, low: 0 }; for (const r of scene.rooms) conf[r.confidence] = (conf[r.confidence] || 0) + 1;
    put(left,
      h('div', { class: 'side-head' }, h('h2', null, 'Layers'), h('span', { class: 'row' }, h('button', { class: 'btn-link small', onclick: () => all(true) }, 'Show all'), h('button', { class: 'btn-link small', onclick: () => all(false) }, 'Hide all'))),
      h('div', { class: 'stack' }, h('div', { class: 'section-title' }, 'Colour rooms by'),
        h('div', { class: 'seg', role: 'group', 'aria-label': 'Colour rooms by' },
          h('button', { class: RV.colour === 'room' ? 'on' : '', 'aria-pressed': String(RV.colour === 'room'), onclick: () => { RV.colour = 'room'; remember(); drawLeft(); drawRooms(); viewer.draw(); if (v3) v3.setRoomColours(roomColour); } }, 'Room'),
          h('button', { class: RV.colour === 'confidence' ? 'on' : '', 'aria-pressed': String(RV.colour === 'confidence'), onclick: () => { RV.colour = 'confidence'; remember(); drawLeft(); drawRooms(); viewer.draw(); if (v3) v3.setRoomColours(roomColour); } }, 'Confidence')),
        RV.colour === 'confidence' ? h('div', { class: 'legend-rows' },
          h('div', null, h('span', { class: 'swatch c-success' }), h('span', { title: 'Stamp area within 5 %', class: 'help' }, 'High'), h('span', { class: 'n' }, conf.high)),
          h('div', null, h('span', { class: 'swatch c-warning' }), h('span', { title: 'No stamp to check against', class: 'help' }, 'Medium'), h('span', { class: 'n' }, conf.medium)),
          h('div', null, h('span', { class: 'swatch c-danger' }), h('span', { title: 'Area off by more than 15 %, several stamps, or tiny', class: 'help' }, 'Low'), h('span', { class: 'n' }, conf.low))) : null),
      GROUPS.map(([name, items]) => h('details', { class: 'layer-group', open: true }, h('summary', null, name),
        h('div', { class: 'layers' }, items.map(([key, label]) => layerRow(key, label, n[key]))))));
  }
  function layerRow(key, name, n) {
    const colour = LAYER_COLOURS[key] || (key === 'overlay' ? NEUTRAL : null);
    const cls = key === 'rooms' ? (RV.colour === 'confidence' ? 'sw-conf' : 'sw-rooms') : key === 'labels' ? 'sw-outline' : key === 'gf' ? 'sw-dashed' : '';
    return h('label', null, h('input', { type: 'checkbox', checked: RV.layers[key], onchange: (e) => { RV.layers[key] = e.target.checked; remember();
      if (key === 'overlay' && e.target.checked && !overlayImg && D) { overlayImg = new Image(); overlayImg.onload = () => viewer.draw(); overlayImg.src = `/api/sheets/${row.id}/file/${D.files.overlay}`; }
      viewer.draw(); if (v3) v3.setLayers(RV.layers); } }),
      h('span', { class: 'swatch ' + cls, style: colour && !cls ? `background:${colour}` : '' }), name, n != null ? h('span', { class: 'n' }, n) : null);
  }

  // --- right: the progress while it runs, the figures and rooms when done
  const figures = h('div'), roomList = h('div', { class: 'room-table' }), issues = h('div', { class: 'issues-list' });
  const tabPanel = h('div', { role: 'tabpanel' });
  const tabBtn = (key, label) => h('button', { role: 'tab', 'aria-selected': String(RV.tab === key), onclick: () => { RV.tab = key; remember(); drawTabs(); } }, label);
  const tabs = h('div', { class: 'tabs', role: 'tablist', 'aria-label': 'Rooms and issues' });
  function drawTabs() {
    put(tabs, tabBtn('rooms', `Rooms · ${scene.rooms.length}`), tabBtn('issues', `Issues · ${D ? needs.length + D.qa.length : 0}`));
    put(tabPanel, RV.tab === 'issues' ? issues : roomList);
  }
  function drawRight() {
    const j = J[row.id] || row.job, running = busy(j);           // the fresh job record: the poll replaces J[id]
    if (!done) {
      const failed = !running && (j.status === 'error' || !!(row.res && (row.res.error || row.res.skipped)));
      const p = j.progress; const info = p && p.stage && (p.key === keyOf(row.pkg, row.dr) || p.key === row.pkg.id) ? stageInfo(p.stage) : null;
      const k = info && info.index ? info.index : 0;
      put(right,
        h('div', { class: 'side-head' }, h('h2', null, running ? 'Extracting…' : failed ? 'Extraction failed' : 'Ready to extract'), running ? badge('run', 'Running') : failed ? badge('err', 'Failed') : null),
        running ? h('div', { class: 'stack-sm', role: 'status', 'aria-live': 'polite' },
          h('div', { class: 'row kv' }, h('span', null, info ? info.label : p && p.stage ? stageInfo(p.stage).label : 'Waiting for the worker…'), h('span', { class: 'muted small' }, k ? `${k} of ${EXTRACT_STAGES}` : '')),
          h('div', { class: 'progress-line', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': EXTRACT_STAGES, 'aria-valuenow': k }, h('i', { style: `width:${Math.round(100 * k / EXTRACT_STAGES)}%` }))) : null,
        h('p', { class: 'muted small pretty' }, running ? 'Walls, openings, stairs and rooms appear on the plan as they are found. About a minute per sheet on this machine, mostly text recognition.'
          : failed ? 'See the message below. Go back to correct the area or the scale, or run it again.' : 'The extraction starts when the scale is confirmed. Start it here if it has not.'),
        rows.length > 1 ? h('div', { class: 'run-list' }, rows.map((r) => h('div', { class: 'run-row' + (r.key === row.key ? ' on' : ''), onclick: () => { P.resultsKey = r.key; saveProject(); renderResults(); } },
          h('div', { class: 'f' }, h('div', { class: 'n' }, r.job.name + (r.dr.title ? ` · ${r.dr.title}` : '')), h('div', { class: 'm' }, `${scaleText(r)} · ${r.conf.polygon_px ? 'area set by hand' : 'area confirmed'}`, runStatus(r)))))) : null,
        !running && j.status === 'error' && j.error ? errorBox(j) : null,
        !running && row.res && (row.res.error || row.res.skipped) ? h('div', { class: 'error-box' }, row.res.error || row.res.skipped) : null,
        h('div', { class: 'foot' }, h('button', { class: 'btn btn-pill btn-xl', onclick: () => go('scale', cursorOf(row)) }, 'Back to scale'),
          !running ? h('button', { class: 'btn btn-primary btn-pill btn-xl', onclick: () => startRun(row) }, row.res ? 'Run again' : 'Run extraction') : null));
      return;
    }
    const pick = rows.length > 1 ? h('select', { class: 'select', 'aria-label': 'Drawing', onchange: (e) => { P.resultsKey = e.target.value; saveProject(); renderResults(); } },
      rows.map((r, i) => h('option', { value: r.key, selected: r.key === row.key }, (r.label || r.dr.title || `Drawing ${i + 1}`) + ' · ' + r.job.name))) : null;
    put(right,
      h('div', { class: 'side-head' }, h('h2', { class: 'clip' }, row.label || row.dr.title || row.job.name), badge('ok', 'Finished')),
      h('div', { class: 'row row-md small' }, h('button', { class: 'btn-link', title: 'Back to the area and scale steps of this sheet', onclick: () => go('scale', cursorOf(row)) }, '‹ Area and scale'),
        h('button', { class: 'btn-link', title: 'Run the extraction of this drawing again, e.g. after changing the area or the scale', onclick: () => startRun(row) }, 'Run again')),
      pick, figures, tabs, tabPanel);
    drawFigures(); drawRooms(); drawIssues(); drawTabs();
  }
  function drawFigures() {
    if (!D) return;
    const total = scene.rooms.reduce((a, r) => a + (r.area_net || 0), 0), high = scene.rooms.filter((r) => r.confidence === 'high').reduce((a, r) => a + (r.area_net || 0), 0);
    const share = total ? Math.round(100 * high / total) : 0, checked = scene.rooms.filter((r) => review[r.id] && review[r.id].checked).length;
    const t = D.times || {}; const total_s = t.total != null ? t.total : Object.values(t).reduce((a, b) => a + (+b || 0), 0);
    const cons = (D.scale && D.scale.consensus) || {}; const px = D.scale && D.scale.confirmed ? D.scale.confirmed.px_per_m : cons.px_per_m;
    const sc = px ? `${px.toFixed(1)} px/m · ${D.scale.confirmed ? 'confirmed' : cons.confidence || 'proposed'}` : '–';
    put(figures, h('div', { class: 'kv-grid' },
      h('span', null, 'Rooms'), h('span', null, `${scene.rooms.length}${checked ? ` · ${checked} checked` : ''}`),
      h('span', null, 'Net room area'), h('span', null, fmt.m2(D.floor.sum_net)),
      h('span', null, 'Gross floor area'), h('span', null, fmt.m2(D.floor.gf_area)),
      h('span', null, 'Openings'), h('span', null, scene.openings.length),
      h('span', null, 'Needs review'), h('span', { class: needs.length ? 'danger-text strong' : '' }, needs.length),
      h('span', null, 'Confidence'), h('span', null, h('span', { class: 'bar' }, h('i', { style: `width:${share}%` })), ' ', h('span', { title: 'Share of room area with high confidence', class: 'help' }, `${share} %`)),
      h('span', null, 'Scale'), h('span', { class: 'small' }, sc),
      h('span', null, 'Processing time'), h('span', null, fmt.secs(total_s))));
  }
  function drawRooms() {
    if (!D) return;
    const order = { low: 0, medium: 1, high: 2 };
    const sorted = [...scene.rooms].sort((a, b) => RV.sort === 'area' ? (b.area_net || 0) - (a.area_net || 0) : (order[a.confidence] ?? 3) - (order[b.confidence] ?? 3) || (b.area_net || 0) - (a.area_net || 0));
    const sortBtn = (key, label) => h('button', { class: 'btn-link text' + (RV.sort === key ? ' strong' : ''), 'aria-pressed': String(RV.sort === key), title: `Sort by ${label.toLowerCase()}`, onclick: () => { RV.sort = key; remember(); drawRooms(); } }, label, RV.sort === key ? ' ↑' : '');
    put(roomList, h('div', { class: 'room-row head' }, h('span', { title: 'Checked against the sheet' }, '✓'), h('span'), h('span', null, 'Room'), h('span', { class: 'ar' }, sortBtn('area', 'Area')), h('span', { class: 'cf' }, sortBtn('confidence', 'Conf.'))),
      sorted.map((r) => h('div', { class: 'room-row' + (r.id === RV.selected ? ' on' : ''), 'data-id': r.id, tabindex: 0, role: 'button', 'aria-pressed': String(r.id === RV.selected),
        onclick: () => select(r.id), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(r.id); } } },
        h('input', { type: 'checkbox', 'aria-label': `${roomName(r) || 'room'} checked`, checked: !!(review[r.id] && review[r.id].checked), onclick: (e) => e.stopPropagation(), onchange: (e) => setChecked(r, e.target.checked) }),
        h('span', { class: 'swatch', style: `background:${roomColour(r)}` }), h('span', { class: 'nm' }, roomName(r) || '(no stamp)'),
        h('span', { class: 'ar' }, fmt.n(r.area_net)), h('span', { class: 'cf conf-' + r.confidence }, r.confidence))));
  }
  function drawIssues() {
    if (!D) return;
    put(issues,
      needs.length ? h('div', { class: 'section-title' }, 'Needs review') : null,
      needs.length ? h('div', { class: 'review-list' }, needs.map((n) => h('div', { class: 'review-row', tabindex: n.room ? 0 : null, role: n.room ? 'button' : null, onclick: () => n.room && select(n.room), onkeydown: (e) => { if (n.room && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); select(n.room); } } }, h('span', { class: 'dot10 ' + n.cls }), h('div', null, h('div', { class: 't' }, n.title, ' ', h('span', null, n.id)), h('div', { class: 'd' }, n.text))))) : null,
      h('div', { class: 'section-title mt-2' }, `All checks · ${D.qa.length}`),
      h('div', { class: 'qa-list' }, D.qa.map((q) => h('div', { class: 'qa-row' }, h('span', { class: 'sev sev-' + q.severity }, q.severity), h('span', null, h('b', null, q.check), q.element ? ` · ${q.element}` : '', ': ', q.message)))));
  }

  drawLeft(); drawRight(); focusHeading();            // the panels are filled after the mount
  if (done && RV.view === '3d') setView('3d');
  if (!done) {
    live = { key: row.key, timer: null, last: null, work: false, drawRight };
    const tick = async () => {
      if (!live || live.key !== row.key) return;
      let L; try { L = await getJSON(`/api/sheets/${row.id}/live?key=${encodeURIComponent(keyOf(row.pkg, row.dr))}`); } catch (e) { return; }
      if (!live || live.key !== row.key || !L || !L.stage || L.updated === live.last) return;
      live.last = L.updated;
      if (!live.work && L.work) { live.work = true; viewer.load(`/api/sheets/${row.id}/file/${L.work}`, { x0: 0, y0: 0, x1: L.size[0], y1: L.size[1], pad: 40 }).catch(() => {}); }
      applyLive(scene, L); drawLeft(); animate(viewer);
    };
    tick(); live.timer = setInterval(tick, 1500);
    startPolling();
  }
}
/** The rooms, voids and checks that ask for a look: low confidence or a review note, voids deducted from GF, high QA findings. */
function needsReview(D, scene) {
  const out = [];
  for (const r of scene.rooms) if (r.confidence === 'low' || r.review) out.push({ cls: r.confidence === 'low' ? 'c-danger' : 'c-warning', title: roomName(r) || '(no stamp)', id: r.id, room: r.id, text: r.review ? `${r.review.guess || 'small region'}: ${r.review.reason || ''}` : (r.reasons || [])[0] || 'low confidence' });
  for (const v of scene.voids) if (v.gf_deducted) out.push({ cls: 'c-info', title: v.label || 'void', id: v.id || 'void', text: `${fmt.m2(v.area)} deducted from GF (above the 5 m² rule)` });
  for (const q of D.qa) if (q.severity === 'high' && !/^r\d/.test(q.element || '')) out.push({ cls: 'c-danger', title: q.check, id: q.element, text: q.message });
  return out;
}
/** The export files of a finished drawing for the Download menu (all of them exist on the server). */
function downloads(row, D) {
  const file = (name) => `/api/sheets/${row.id}/file/${name}?download=1`;
  const stem = D.files.json.replace(/\.json$/, '');
  return [
    { label: 'JSON', note: 'pipeline output, plan metres', href: file(D.files.json), name: D.files.json },
    { label: 'Excel', note: 'room list, floor figures, openings, QA', href: file(`${stem}.xlsx`), name: `${stem}.xlsx` },
    { label: 'DXF', note: 'CAD-Richtlinie BBL layers', href: file(D.files.dxf), name: D.files.dxf },
    { label: 'IFC', note: 'spaces, walls, doors, windows, stairs, slab', href: file(`${stem}.ifc`), name: `${stem}.ifc` },
    { label: 'CSV', note: 'room list with review state', href: `/api/sheets/${row.id}/rooms.csv?key=${encodeURIComponent(keyOf(row.pkg, row.dr))}`, name: `${stem}-rooms.csv` },
    { label: 'PNG', note: 'plan with the segmentation overlay', href: file(D.files.overlay), name: D.files.overlay },
  ];
}

export { RV, renderResults, runStatus, scaleText, stop3D, stopLive };
