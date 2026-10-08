/* Constants, design-token colours for the canvas, DOM and fetch helpers, number formats. */

const PAL = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#86bcb6'];
// canvas colours are read from the design tokens (static/css/tokens.css), so the legend and the drawing agree
const T = (name, fallback) => (getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback);
const CONF = { high: T('--success', '#2e9e5b'), medium: T('--warning', '#d99a1e'), low: T('--danger', '#d0433b'), confirmed: T('--accent-canvas', '#2f6ea6'), none: T('--danger', '#d0433b') };
const ACCENT = T('--accent-canvas', '#2f6ea6');   // the accent as hex: canvas strokes and alpha fills need it
const AMBER = T('--warning', '#d99a1e');
const NEUTRAL = T('--neutral-canvas', '#8a8f99');
const LAYER_COLOURS = Object.fromEntries(['walls', 'doors', 'windows', 'passages', 'columns', 'stairs', 'voids', 'graph', 'gf'].map((k) => [k, T(`--${k}`, '#777')]));
const REGION_COLOURS = { 'title block': '#e67800', legend: '#00a0a0', 'scale bar': '#c800c8', 'scale note': '#c80078',
  'north arrow': '#7800dc', notes: '#787800', 'key plan': '#dc0000', caption: '#0078ff', other: '#5a5a5a', frame: '#969696' };
const UNITS = { m: 1, cm: 0.01, mm: 0.001, ft: 0.3048, in: 0.0254 };
const DPI_SOURCE = { render: 'from the PDF render', scan: 'from the scan header', paper: 'from the paper size (assumed)',
  metadata: 'from file metadata (unverified)', default: 'screen default, not trusted', unknown: 'unknown' };
const PKEY = 'fpx-workflow-project';

// ---------------------------------------------------------------------------------------------------------------------
// Small helpers

const $ = (sel, el = document) => el.querySelector(sel);
const put = (el, ...kids) => el.replaceChildren(...kids.flat(Infinity).filter((k) => k != null && k !== false));

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style') el.style.cssText = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
  return data;
}
const getJSON = (p) => api(p);
const postJSON = (p, body) => api(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

const fmt = {
  m2: (v) => v == null ? '–' : `${(+v).toFixed(1)} m²`,
  m: (v) => v == null ? '–' : `${(+v).toFixed(1)} m`,
  n: (v, d = 1) => v == null ? '–' : (+v).toFixed(d),
  pct: (v) => v == null ? '' : `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(0)} %`,
  scaleN: (N) => N == null ? '–' : `1 : ${N >= 20 ? Math.round(N) : N.toFixed(1)}`,
  secs: (s) => s == null ? '–' : s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`,
  mb: (b) => `${(b / 1e6).toFixed(1)} MB`,
};

function confColour(c) { return CONF[c] || NEUTRAL; }

/** The pipeline's stage labels in plain words, with the position of the extraction stages (1..12) for a progress bar. */
const STAGES = [
  ['1 sheet text', 'Reading the sheet text', 0], ['1b layout', 'Finding the drawings', 0], ['1c door', 'Searching doors for the scale', 0], ['1c scale', 'Reading the scale cues', 0],
  ['2 text (OCR', 'Reading the text in the drawing', 1], ['0 triage', 'Checking the drawing style', 2], ['2 text layer', 'Placing the text', 3],
  ['3 segmentation', 'Finding walls and openings', 4], ['3 walls', 'Tracing the walls', 5], ['4 openings', 'Doors and windows', 6],
  ['5 stairs', 'Stairs and voids', 7], ['6 rooms', 'Rooms', 8], ['7 room', 'Room names and areas', 9], ['8 derived', 'Areas and connections', 10],
  ['9 scale', 'Checks', 11], ['10 export', 'Writing the exports', 12]];
function stageInfo(stage) {
  const s = String(stage || '');
  const hit = STAGES.find(([prefix]) => s.startsWith(prefix));
  return hit ? { label: hit[1], index: hit[2] } : { label: s || 'Working…', index: 0 };
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

export { $, ACCENT, AMBER, CONF, DPI_SOURCE, LAYER_COLOURS, NEUTRAL, PAL, PKEY, REGION_COLOURS, T, UNITS, api, confColour, esc, fmt, getJSON, h, postJSON, put, stageInfo };
