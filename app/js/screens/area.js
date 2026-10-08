/* Step 1, building area (board 1d): the detected outline with handles; also the waiting screen and the walk to the next package. */
import { ACCENT, NEUTRAL, REGION_COLOURS, api, h, postJSON, put } from '../core.js';
import { J, P, bboxOf, firstUnconfirmed, keyOf, packagesOf, polygonOf, ppmOf, refreshJob, selectedDrawings } from '../state.js';
import { busyScreen, go, loadPreview, mount, post, projectName, stepHead, useViewer } from '../ui.js';
import { RectTool, scaled } from '../viewer.js';

// --- Step 1: building area (board 1d) ------------------------------------------------------------------------------

function renderArea() {
  const id = P.sheets[P.cursor.sheet]; const job = J[id];
  if (!job || !job.analysis) return renderWaiting(job);
  const pk = packagesOf(job); const pi = Math.min(P.cursor.pkg, pk.length - 1); const pkg = pk[pi];
  const sel = new Set(selectedDrawings(job, pkg).map((d) => d.id));
  const rects = {};                                  // drawing id -> editable rect in sheet px
  for (const d of pkg.drawings) rects[d.id] = bboxOf(polygonOf(job, pkg, d));
  let current = pkg.drawings.find((d) => sel.has(d.id)) || pkg.drawings[0];
  const f = pkg.preview.scale;
  const stage = h('main', { class: 'stage grid-bg' });
  const top = h('div', { class: 'stage-top' },
    h('div', { class: 'pill' }, h('span', { class: 'muted' }, `Sheet ${P.cursor.sheet + 1} of ${P.sheets.length}`), h('b', null, job.name), pk.length > 1 ? h('span', { class: 'muted' }, `· page ${pi + 1} of ${pk.length}`) : null),
    h('button', { class: 'btn btn-pill btn-lg', onclick: () => { if (!current) return; rects[current.id] = bboxOf(current.polygon_px); tool.r = { ...rects[current.id] }; viewer.draw(); } }, 'Reset to detected'));
  stage.append(top, h('div', { class: 'stage-note' }, 'Drag the handles or the area · scroll to zoom · drag the sheet to pan'));
  const list = h('div', { class: 'drawing-list' });
  const panel = h('aside', { class: 'panel' },
    stepHead(1, 'Step 1 of 2 · Building area', 'Confirm the building area'),
    h('p', { class: 'muted pretty' }, 'The outline was detected from the drawing. Drag the handles so the whole building is inside and the title block, legend and dimension chains stay out.'),
    pkg.drawings.length > 1 ? list : null,            // which drawing to extract: only a question when the sheet has several
    pkg.drawings.length === 0 ? h('div', { class: 'warn-box' }, 'No drawing was found on this sheet. It can only be skipped.') : null,
    h('div', { class: 'foot' },
      h('button', { class: 'btn btn-lg', onclick: () => confirmArea(job, pkg, rects, new Set()) }, 'Skip this sheet'),
      h('button', { class: 'btn btn-primary btn-lg', disabled: !pkg.drawings.length, onclick: () => confirmArea(job, pkg, rects, sel) }, 'Confirm area')));
  mount(h('div', { class: 'step' }, stage, panel), projectName());
  const viewer = useViewer(stage);
  const tool = new RectTool(current ? scaleRect(rects[current.id], f) : { x0: 0, y0: 0, x1: 0, y1: 0 }, (r) => { if (current) rects[current.id] = scaleRect(r, 1 / f); drawList(); });
  if (current) viewer.tool = tool;
  function drawList() {
    put(list, h('div', { class: 'section-title' }, `${pkg.drawings.length} drawing${pkg.drawings.length === 1 ? '' : 's'} on this sheet`));
    for (const d of pkg.drawings) {
      const r = rects[d.id]; const ppm = ppmOf(job, pkg, d);
      const size = ppm ? `${((r.x1 - r.x0) / ppm).toFixed(1)} × ${((r.y1 - r.y0) / ppm).toFixed(1)} m` : `${Math.round(r.x1 - r.x0)} × ${Math.round(r.y1 - r.y0)} px`;
      const cb = h('input', { type: 'checkbox', checked: sel.has(d.id), onclick: (e) => { e.stopPropagation(); }, onchange: (e) => { e.target.checked ? sel.add(d.id) : sel.delete(d.id); drawList(); viewer.draw(); } });
      list.append(h('div', { class: 'drawing-row' + (d === current ? ' on' : ''), onclick: () => { current = d; tool.r = scaleRect(rects[d.id], f); viewer.tool = tool; drawList(); viewer.draw(); } },
        cb, h('div', { class: 'clip' }, h('div', { class: 't' }, d.title || `${d.kind} ${d.id}`), h('div', { class: 's' }, `${d.id} · ${size}${d.storey ? ' · ' + d.storey : ''}${d.kind_confidence === 'low' ? ' · kind assumed' : ''}`)),
        h('span', { class: 'kind' }, d.kind)));
    }
  }
  drawList();
  viewer.overlay = (c, v) => {
    for (const r of pkg.regions) {
      if (r.class === 'drawing' || r.class === 'frame' || !r.polygon_px || r.polygon_px.length < 3) continue;
      const col = REGION_COLOURS[r.class] || NEUTRAL;
      v.poly(scaled(r.polygon_px, f), { stroke: col, width: 1, dash: [4, 3], fill: col + '14' });
      const b = bboxOf(r.polygon_px); v.label(b.x0 * f, b.y0 * f, r.class, { colour: col, size: 10, dy: -2 });
    }
    for (const d of pkg.drawings) {
      const pts = scaled(d.polygon_px, f);
      if (d === current) v.poly(pts, { stroke: ACCENT, width: 1, dash: [5, 4] });
      else { v.poly(pts, { stroke: sel.has(d.id) ? ACCENT : NEUTRAL, width: 1.5, dash: [6, 4] }); const b = bboxOf(pts); v.label(b.x0, b.y0, `${d.kind}${sel.has(d.id) ? '' : ' (not extracted)'}`, { colour: sel.has(d.id) ? ACCENT : '#646b74' }); }
    }
    if (current) {
      const r = rects[current.id]; const ppm = ppmOf(job, pkg, current);
      const size = ppm ? `${((r.x1 - r.x0) / ppm).toFixed(1)} × ${((r.y1 - r.y0) / ppm).toFixed(1)} m` : `${Math.round(r.x1 - r.x0)} × ${Math.round(r.y1 - r.y0)} px`;
      v.label(r.x0 * f, r.y0 * f, `Detected building · ${size}`, { dy: -14 });
    }
  };
  loadPreview(viewer, job, pkg, panel);
}
function scaleRect(r, k) { return { x0: r.x0 * k, y0: r.y0 * k, x1: r.x1 * k, y1: r.y1 * k }; }
async function confirmArea(job, pkg, rects, sel) {
  const body = {};
  for (const d of pkg.drawings) {
    const r = rects[d.id]; const det = bboxOf(d.polygon_px);
    const changed = ['x0', 'y0', 'x1', 'y1'].some((k) => Math.abs(r[k] - det[k]) > 0.5);
    body[keyOf(pkg, d)] = { extract: sel.has(d.id), area_confirmed: true,
      polygon_px: changed ? [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]].map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]) : null };
  }
  const r = await post(`/api/sheets/${job.id}/confirm`, { confirm: body }); if (!r) return; job.confirm = r.confirm;
  // step 2 starts now: the scale cues of the selected drawings, read inside the confirmed area, its caption band and the title block
  const keys = pkg.drawings.filter((d) => sel.has(d.id)).map((d) => keyOf(pkg, d));
  if (keys.length) { try { await postJSON(`/api/sheets/${job.id}/scale`, { keys, doors: false }); } catch (e) { /* busy: the scale step shows the state */ } await refreshJob(job.id); }
  const first = selectedDrawings(job, pkg)[0];
  if (first) go('scale', { ...P.cursor, drawing: pkg.drawings.indexOf(first) });
  else nextPackage();
}
function nextPackage() {
  const job = J[P.sheets[P.cursor.sheet]];
  if (P.cursor.pkg + 1 < packagesOf(job).length) return go('area', { ...P.cursor, pkg: P.cursor.pkg + 1, drawing: 0 });
  const c = firstUnconfirmed();
  if (c) return go(c.scale ? 'scale' : 'area', c);
  go('results');
}
/** A step opened before its sheet is analysed (a deep link, a reload): the busy screen with the way back. */
function renderWaiting(job) {
  if (!job) return go('upload');
  busyScreen([job], { title: projectName() || 'New project', back: () => go('upload') });
}

export { confirmArea, nextPackage, renderArea, renderWaiting, scaleRect };
