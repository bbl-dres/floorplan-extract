/* Step 2, plan scale (boards 2a-2c): proposal, cues, resolution, measuring a known distance. */
import { $, ACCENT, AMBER, DPI_SOURCE, T, UNITS, api, fmt, h, postJSON, put } from '../core.js';
import { nextPackage, renderWaiting } from './area.js';
import { J, P, bboxOf, busy, confOf, keyOf, packagesOf, polygonOf, refreshJob, saveProject, screenName, selectedDrawings, sheetConfirmed, startPolling } from '../state.js';
import { errorBox, go, loadPreview, mount, notify, post, previewUrl, projectName, stepHead, useViewer } from '../ui.js';
import { MeasureTool, scaled } from '../viewer.js';

// --- Step 2: plan scale (boards 2a, 2b, 2c) ------------------------------------------------------------------------

const SCALE_REQUESTED = new Set();
/** Queue step 2 for one drawing (doors: the door-width search with the segmenter), once per session; true when queued now. */
function requestScale(job, pkg, dr, doors = false) {
  const tag = `${job.id}|${keyOf(pkg, dr)}|${doors ? 'doors' : 'cues'}`;
  if (SCALE_REQUESTED.has(tag)) return false;
  SCALE_REQUESTED.add(tag);
  postJSON(`/api/sheets/${job.id}/scale`, { keys: [keyOf(pkg, dr)], doors }).then(() => refreshJob(job.id))
    .then(() => { if (screenName === 'scale') renderScale(); })
    .catch((e) => { SCALE_REQUESTED.delete(tag); notify(e.message); });
  return true;
}
/** The scale step while the server reads the cues: the drawing on the stage, a spinner in the panel. */
function renderScaleWait(job, pkg, dr) {
  const doors = !!(job.progress && job.progress.doors);
  const stage = h('main', { class: 'stage grid-bg' });
  const panel = h('aside', { class: 'panel' }, stepHead(2, 'Step 2 of 2 · Scale', 'Check the detected scale'),
    h('div', { class: 'row row-md' }, h('span', { class: 'spinner' }),
      h('span', null, doors ? 'Searching the drawing for doors with the segmenter…' : 'Reading the scale cues: title block, caption, scale bar and the dimension strings inside the area…')),
    h('p', { class: 'muted small pretty' }, doors ? 'About 20 seconds.' : 'About 5 to 20 seconds. The cues are read where they sit, not on the whole sheet.'),
    job.error ? errorBox(job) : null,
    h('div', { class: 'foot' }, h('button', { class: 'btn btn-pill btn-xl', onclick: () => go('area', { ...P.cursor }) }, 'Back')));
  mount(h('div', { class: 'step' }, stage, panel), projectName());
  const viewer = useViewer(stage); const f = pkg.preview.scale;
  viewer.overlay = (c, v) => { v.poly(scaled(polygonOf(job, pkg, dr), f), { stroke: ACCENT, width: 1.5, dash: [6, 4] }); };
  loadPreview(viewer, job, pkg, panel);
  startPolling();
}
function scaleCase(dr, conf) {
  const s = dr.scale || {}; const agree = (s.agreeing || []).map((a) => a.cue);
  if (!s.px_per_m) return 'none';
  if (agree.includes('viewport') || agree.includes('model_units')) return 'exact';
  if (agree.includes('scale_note') || agree.includes('sheet_scale_note')) return 'note';
  if (agree.includes('dimension_strings')) return 'dims';
  if (agree.includes('scale_bar') || agree.includes('linked')) return 'bar';
  if (agree.includes('door_widths')) return 'doors';
  return 'other';
}
const CASE_TEXT = {
  exact: ['The scale is exact: it comes from the CAD file (a layout viewport or model units).', null],
  note: ['A scale was read in the notes region. For an image, the real size of a pixel also depends on the resolution, so check both values before continuing.', null],
  dims: ['The scale was measured from the dimension strings on the drawing (highlighted): the numbers on the dimension lines against the distance between their ticks.', null],
  bar: ['No scale text was found on this drawing. A graphical scale bar was found instead (highlighted) and its length was measured.',
    'Warning: this scale comes from a drawn bar, which is rounded on print. Check it carefully before confirming.'],
  doors: ['No scale note, scale bar or dimension string was read. The scale was estimated from the width of the detected doors.',
    'Warning: an estimate from door widths is rough (20 % or more). Measure a known distance if any dimension is readable.'],
  other: ['The scale comes from a weak cue. Check it before continuing.', null],
  none: ['No scale cue was found on this drawing.', 'Measure a known distance (a dimension string, a scale bar or a known room width) to set the scale. Without it the drawing cannot be extracted.'],
};

function renderScale() {
  const id = P.sheets[P.cursor.sheet]; const job = J[id];
  if (!job || !job.analysis) return renderWaiting(job);
  const pk = packagesOf(job); const pkg = pk[Math.min(P.cursor.pkg, pk.length - 1)];
  const dr = pkg.drawings[Math.min(P.cursor.drawing, pkg.drawings.length - 1)];
  if (!dr) return go('area', { ...P.cursor, drawing: 0 });        // a page without drawings has no scale to check
  // step 2 runs on the server after the area is confirmed; until its record is there, the step waits (and asks for
  // it once when nothing is running, e.g. a job analysed before this step existed)
  if (!dr.scale && (busy(job) || requestScale(job, pkg, dr))) return renderScaleWait(job, pkg, dr);
  const conf = confOf(job, pkg, dr); const s = dr.scale || {}; const f = pkg.preview.scale;
  const sel = selectedDrawings(job, pkg); const di = sel.indexOf(dr);
  const kind = scaleCase(dr, conf);
  const noteN = s.note && s.note.scale;
  const noteBased = kind === 'note' && !(s.agreeing || []).some((a) => ['dimension_strings', 'scale_bar', 'linked', 'viewport', 'model_units'].includes(a.cue));
  // state of the step
  const st = { mode: conf.measured ? 'measured' : 'check', dpi: conf.dpi || pkg.summary.dpi || null, dpiChanged: !!conf.dpi,
    measured: conf.measured || null, pts: [], distance: conf.measured ? conf.measured.distance : '', unit: conf.measured ? conf.measured.unit : 'm' };
  const ppmProposal = s.px_per_m || null;
  function ppmNow() {
    if (st.mode === 'measure' || st.mode === 'measured') {
      const px = st.mode === 'measured' ? st.measured.px : st.pts.length === 2 ? Math.hypot(st.pts[1][0] - st.pts[0][0], st.pts[1][1] - st.pts[0][1]) / f : null;
      const m = parseFloat(String(st.distance).replace(',', '.')) * (UNITS[st.unit] || 1);
      return px && m > 0 ? px / m : null;
    }
    if (st.dpiChanged && noteBased && noteN && st.dpi) return st.dpi / 0.0254 / noteN;
    return conf.px_per_m && conf.scale_source !== 'measured' ? conf.px_per_m : ppmProposal;
  }
  const N = (ppm) => ppm && st.dpi ? st.dpi / 0.0254 / ppm : null;
  const stage = h('main', { class: 'stage grid-bg' });
  const topPill = h('div', { class: 'pill' });
  const btnReset = h('button', { class: 'btn btn-round', title: 'Start the measurement over', 'aria-label': 'Start the measurement over', onclick: () => { if (tool) tool.reset(); } }, '↻');
  const btnHelp = h('button', { class: 'btn btn-round strong', 'aria-label': 'Help', onclick: () => { $('#help').hidden = false; } }, '?');
  stage.append(h('div', { class: 'stage-top' }, topPill, btnReset, btnHelp));
  const panel = h('aside', { class: 'panel' });
  mount(h('div', { class: 'step' }, stage, panel), projectName());
  const viewer = useViewer(stage);
  let tool = null;
  const measure = new MeasureTool((pts) => { st.pts = pts.map((p) => [p[0] / f, p[1] / f]); renderPanel(); });
  measure.pxScale = 1 / f;
  viewer.overlay = (c, v) => {
    v.poly(scaled(polygonOf(job, pkg, dr), f), { stroke: ACCENT, width: 1.5, dash: [6, 4] });
    for (const b of (s.cue_boxes || [])) {
      const [x0, y0, x1, y1] = b.box;
      v.rect({ x0: x0 * f, y0: y0 * f, x1: x1 * f, y1: y1 * f }, { stroke: AMBER, width: 3, fill: 'rgba(217,154,30,.08)' });
      if (b.cue !== 'dimension_strings') v.label(x0 * f, y0 * f, b.label, { colour: T('--warning-text-strong', '#8a5a00'), dy: -8 });
    }
    if (st.mode === 'measured' && st.measured && st.measured.p1) {
      const p = scaled([st.measured.p1, st.measured.p2], f);
      v.poly(p, { stroke: ACCENT, width: 3, close: false }); p.forEach(([x, y]) => v.handle(x, y, 7));
      v.label((p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2, `${Math.round(st.measured.px)} px`, { bg: ACCENT, align: 'center', dy: -10 });
    }
  };
  // the view frames the drawing and its scale cues (at most 3x closer than the whole sheet), also after a resize or Fit
  const b = bboxOf(polygonOf(job, pkg, dr));
  const cues = (s.cue_boxes || []).map((c) => c.box);
  const bb = bboxOf([[b.x0, b.y0], [b.x1, b.y1], ...cues.flatMap((c) => [[c[0], c[1]], [c[2], c[3]]])]);
  viewer.load(previewUrl(job, pkg), { x0: bb.x0 * f, y0: bb.y0 * f, x1: bb.x1 * f, y1: bb.y1 * f, pad: 60, maxZoom: 3 }).catch((e) => panel.prepend(errorBox(job, e.message)));

  function renderPanel() {
    const ppm = ppmNow(); const n = N(ppm);
    const measuring = st.mode === 'measure';
    topPill.className = 'pill' + (measuring ? ' accent' : '');
    if (measuring) {
      const step = st.pts.length === 0 ? 'Click the first point.' : st.pts.length === 1 ? 'Click the second point.' : 'Enter the real distance.';
      put(topPill, h('span', { class: 'accent-text' }, 'Measure a known distance'), h('b', null, step));
    } else put(topPill, h('span', { class: 'muted' }, 'Reviewing floor plan'), h('b', null, dr.title || job.name), sel.length > 1 ? h('span', { class: 'muted' }, `· drawing ${di + 1} of ${sel.length}`) : null);
    btnReset.hidden = !measuring;
    const head = stepHead(2, 'Step 2 of 2 · Scale', 'Check the detected scale');
    const dpiRow = () => {
      const src = DPI_SOURCE[pkg.summary.dpi_source] || pkg.summary.dpi_source;
      const inp = h('input', { type: 'number', min: 10, max: 2400, step: 1, value: st.dpi || '', class: 'input-dpi', onchange: (e) => { const v = parseFloat(e.target.value); if (v > 0) { st.dpi = v; st.dpiChanged = true; renderPanel(); } } });
      const change = h('button', { class: 'btn btn-pill btn-sm', onclick: () => { put(right, inp, h('span', { class: 'muted small' }, 'dpi')); inp.focus(); } }, 'Change');
      const right = h('span', { class: 'row row-md' }, h('span', { class: 'big' }, st.dpi ? `${Math.round(st.dpi)} dpi` : 'unknown'), change);
      return h('div', { class: 'kv kv-center' }, h('span', { class: 'stack-0' }, 'Resolution', h('span', { class: 'muted small' }, st.dpiChanged ? 'set by hand' : src)), right);
    };
    const pixelRow = (sep) => h('div', { class: 'kv' + (sep ? ' sep' : '') }, h('span', null, 'Pixel size'), h('span', null, ppm ? `${ppm.toFixed(1)} px per metre` : '–', ppm ? h('span', { class: 'muted' }, ` · ${(1000 / ppm).toFixed(1)} mm per pixel`) : null));
    let body;
    if (measuring) {
      const stepN = st.pts.length < 2 ? st.pts.length + 1 : 3;
      const stepText = ['Click the first point.', 'Click the second point.', 'Enter the real distance.'][stepN - 1];
      const inp = h('input', { type: 'text', class: 'input-xl', value: st.distance, placeholder: 'e.g. 22.55', oninput: (e) => { st.distance = e.target.value; renderPanel(); } });
      const unit = h('select', { class: 'input-xl select-unit', onchange: (e) => { st.unit = e.target.value; renderPanel(); } }, Object.keys(UNITS).map((u) => h('option', { value: u, selected: u === st.unit }, u)));
      const px = st.pts.length === 2 ? Math.hypot(st.pts[1][0] - st.pts[0][0], st.pts[1][1] - st.pts[0][1]) : null;
      body = [
        h('div', { class: 'kv kv-center' }, h('span', { class: 'title-md' }, 'Measure a known distance'), h('button', { class: 'btn-link', onclick: () => { st.mode = conf.measured ? 'measured' : 'check'; viewer.tool = null; viewer.canvas.style.cursor = 'grab'; renderPanel(); viewer.draw(); } }, 'Cancel')),
        h('p', { class: 'muted small pretty tight' }, 'For an image, the scale can only be corrected by measuring a known distance: a dimension string, a scale bar or a known room width.'),
        h('div', { class: 'info-box' }, h('span', { class: 'num' }, String(stepN)), h('span', { class: 'strong' }, stepText)),
        h('div', { class: 'measure-grid' }, inp, unit),
        h('div', { class: 'scale-rows' },
          h('div', { class: 'kv' }, h('span', null, 'Measured'), h('span', null, px ? `${Math.round(px)} px on the sheet` : '–')),
          h('div', { class: 'kv' }, h('span', null, 'Calculated scale'), h('span', { class: 'value-accent' }, n ? fmt.scaleN(n) : ppm ? `${ppm.toFixed(1)} px/m` : '–')),
          dpiRow(), pixelRow(false)),
      ];
      const focusInput = stepN === 3 && document.activeElement !== inp;
      put(panel, head, body, foot(!!ppm));
      if (focusInput) inp.focus();
      return;
    }
    let [text, warn] = CASE_TEXT[kind];
    const dnote = (s.disagreeing || []).find((a) => a.cue === 'scale_note' || a.cue === 'sheet_scale_note');
    if (dnote && kind !== 'note') {
      const nN = s.note && s.note.scale ? `1:${s.note.scale}` : 'a scale note';
      text = `${nN} was read on the sheet, but the resolution of this image is not verified, so the note cannot be converted exactly: at ${st.dpi ? Math.round(st.dpi) + ' dpi' : 'the assumed dpi'} it gives ${dnote.px_per_m.toFixed(1)} px/m, ${dnote.ratio}x the ${kind === 'doors' ? 'door-width estimate' : 'measured value'} shown below.`;
      warn = kind === 'doors' ? 'Warning: the door-width estimate is rough and disagrees with the note. Measure a known distance (the scale bar, a dimension) or set the resolution if you know it.' : warn;
    }
    const rows = [];
    if (st.mode === 'measured') {
      rows.push(h('div', { class: 'kv' }, h('span', null, 'Measured'), h('span', null, `${Math.round(st.measured.px)} px = ${st.measured.distance} ${st.measured.unit}`)));
      rows.push(h('div', { class: 'kv' }, h('span', null, 'Calculated scale'), h('span', { class: 'big accent-text' }, n ? fmt.scaleN(n) : '–')));
    } else {
      if (kind === 'bar') { const ev = (s.agreeing || []).find((a) => a.cue === 'scale_bar' || a.cue === 'linked'); rows.push(h('div', { class: 'kv' }, h('span', null, 'Scale bar'), h('span', { class: 'small' }, ev ? ev.evidence.replace(/\(fit residual.*\)/, '').slice(0, 60) : ''))); }
      if (kind === 'dims') { const ev = (s.agreeing || []).find((a) => a.cue === 'dimension_strings'); rows.push(h('div', { class: 'kv' }, h('span', null, 'Dimension strings'), h('span', { class: 'small' }, ev ? ev.evidence.split(';')[0].slice(0, 70) : ''))); }
      const shown = !st.dpiChanged && s.scale && Math.abs((ppm || 0) - (ppmProposal || 0)) < 1e-6 ? s.scale.replace(':', ' : ') : fmt.scaleN(n);
      rows.push(h('div', { class: 'kv' }, h('span', null, 'Scale'), h('span', { class: 'big' }, n || s.scale ? shown : ppm ? `${ppm.toFixed(1)} px/m` : '–')));
    }
    rows.push(dpiRow(), pixelRow(true));
    const agree = s.agreeing || [], disagree = s.disagreeing || [];
    const cueList = h('details', null, h('summary', null, `Cues: ${agree.length} agree, ${disagree.length} disagree · confidence `, h('b', { class: 'conf-' + (st.mode === 'measured' ? 'confirmed' : s.confidence || 'none') }, st.mode === 'measured' ? 'measured by hand' : s.confidence || 'none')),
      h('div', { class: 'cues mt-2' },
        agree.map((a) => h('div', { class: 'cue agree' }, h('span', { class: 'mark' }, '✓'), h('div', null, h('b', null, a.cue.replace(/_/g, ' ')), ` ${a.px_per_m.toFixed(1)} px/m`, h('div', { class: 'ev' }, a.evidence)))),
        disagree.map((a) => h('div', { class: 'cue disagree' }, h('span', { class: 'mark' }, '!'), h('div', null, h('b', null, a.cue.replace(/_/g, ' ')), ` ${a.px_per_m.toFixed(1)} px/m (${a.ratio}x)`, h('div', { class: 'ev' }, a.evidence)))),
        (s.flags || []).map((fl) => h('div', { class: 'cue' }, h('span', { class: 'sev sev-' + fl.severity }, fl.severity[0]), h('div', { class: 'ev' }, fl.message)))));
    body = [h('p', { class: 'muted pretty' }, st.mode === 'measured' ? 'The scale was set by measuring a known distance on the drawing.' : text),
      warn && st.mode !== 'measured' ? h('div', { class: 'warn-box pretty' }, warn) : null,
      h('div', { class: 'scale-rows' }, rows), cueList,
      kind === 'none' && !(s.agreeing || []).concat(s.disagreeing || []).some((a) => a.cue === 'door_widths')
        ? h('button', { class: 'btn btn-pill btn-lg', title: 'Runs the segmenter on the drawing and estimates the scale from the width of the detected doors (rough: 20 % or more)', onclick: () => { if (requestScale(job, pkg, dr, true)) renderScale(); } }, 'Estimate from door widths') : null,
      h('button', { class: 'btn btn-pill btn-lg strong', onclick: () => { st.mode = 'measure'; st.pts = []; measure.pts = []; viewer.tool = measure; tool = measure; renderPanel(); viewer.draw(); } }, st.mode === 'measured' ? 'Measure again' : 'Correct by measuring')];
    put(panel, head, body, foot(!!ppm));
  }
  function foot(ok) {
    return h('div', { class: 'foot' },
      h('button', { class: 'btn btn-pill btn-xl', onclick: () => go('area', { ...P.cursor }) }, 'Back'),
      h('button', { class: 'btn btn-primary btn-pill btn-xl', disabled: !ok, onclick: confirmScale }, 'Confirm scale'));
  }
  async function confirmScale() {
    const ppm = ppmNow(); if (!ppm) return;
    const body = { scale_confirmed: true, px_per_m: Math.round(ppm * 1000) / 1000, dpi: st.dpiChanged ? st.dpi : null };
    if (st.mode === 'measure' || st.mode === 'measured') {
      const px = st.mode === 'measured' ? st.measured.px : Math.hypot(st.pts[1][0] - st.pts[0][0], st.pts[1][1] - st.pts[0][1]);
      const p1 = st.mode === 'measured' ? st.measured.p1 : st.pts[0], p2 = st.mode === 'measured' ? st.measured.p2 : st.pts[1];
      body.scale_source = 'measured';
      body.measured = { px: Math.round(px * 10) / 10, distance: parseFloat(String(st.distance).replace(',', '.')), unit: st.unit, p1: p1.map((v) => Math.round(v * 10) / 10), p2: p2.map((v) => Math.round(v * 10) / 10) };
    } else { body.scale_source = st.dpiChanged ? 'dpi changed' : 'proposal'; body.measured = null; }
    const r = await post(`/api/sheets/${job.id}/confirm`, { confirm: { [keyOf(pkg, dr)]: body } }); if (!r) return; job.confirm = r.confirm;
    const next = sel[di + 1];
    if (next) return go('scale', { ...P.cursor, drawing: pkg.drawings.indexOf(next) });
    if (sheetConfirmed(job)) {                             // the last scale of the sheet: the extraction starts now
      if (await post(`/api/sheets/${job.id}/run`, {})) await refreshJob(job.id);
      P.resultsKey = `${job.id}|${keyOf(pkg, dr)}`; saveProject();
    }
    nextPackage();
  }
  renderPanel();
}

export { CASE_TEXT, renderScale, renderScaleWait, requestScale, scaleCase };
