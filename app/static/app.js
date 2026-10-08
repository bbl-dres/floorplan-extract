/* Floor plan workflow app: upload -> building area -> scale -> storeys and run -> results and export.
   Follows the boards of docs/wireframes/261007_Viewer and Workflow UX study.html (1c, 1d, 2a-2c, 1f, 1g/1h). */
'use strict';

const PAL = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#86bcb6'];
const CONF = { high: '#2e9e5b', medium: '#d99a1e', low: '#d0433b', confirmed: '#2f6ea6', none: '#d0433b' };
const ACCENT = '#2f6ea6';              // canvas colour for oklch(0.55 0.12 230)
const AMBER = '#d99a1e';
const LAYER_COLOURS = { walls: '#2b3a55', doors: '#d0433b', windows: '#2f8fd8', passages: '#f08a24', columns: '#6b7280',
  stairs: '#c23fbf', voids: '#1f78b4', graph: '#0f766e', gf: '#111827' };
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

function confColour(c) { return CONF[c] || '#8a93a3'; }

// ---------------------------------------------------------------------------------------------------------------------
// Project state (which sheet jobs belong to the project, in which order, and where the user is)

let P = loadProject();
const J = {};                          // job id -> job record from the server
let screenName = null;
let pollTimer = null;
let results = { key: null, data: null };

function loadProject() {
  try {
    const p = JSON.parse(localStorage.getItem(PKEY) || 'null');
    if (p && Array.isArray(p.sheets)) return Object.assign({ name: '', sheets: [], order: [], screen: 'upload', cursor: { sheet: 0, pkg: 0, drawing: 0 }, resultsKey: null }, p);
  } catch (e) { /* a broken entry: start fresh */ }
  return { name: '', sheets: [], order: [], screen: 'upload', cursor: { sheet: 0, pkg: 0, drawing: 0 }, resultsKey: null };
}
function saveProject() { try { localStorage.setItem(PKEY, JSON.stringify(P)); } catch (e) { /* private window */ } }

async function refreshJob(id) {
  try {
    J[id] = await getJSON(`/api/sheets/${id}`);
  } catch (e) {
    if (/not found/.test(e.message)) { P.sheets = P.sheets.filter((s) => s !== id); saveProject(); delete J[id]; }
    else J[id] = J[id] || { id, status: 'error', error: e.message, log: [], confirm: {} };
  }
  return J[id];
}

async function refreshAll() { await Promise.all(P.sheets.map(refreshJob)); }

function busy(job) { return ['analysing', 'queued', 'running'].includes(job.status); }

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    const ids = P.sheets.filter((id) => !J[id] || busy(J[id]));
    if (!ids.length) return;
    const before = ids.map((id) => J[id] && (J[id].status + '|' + JSON.stringify(J[id].progress) + '|' + (J[id].log || []).length));
    await Promise.all(ids.map(refreshJob));
    const after = ids.map((id) => J[id] && (J[id].status + '|' + JSON.stringify(J[id].progress) + '|' + (J[id].log || []).length));
    if (before.join() !== after.join()) onJobsChanged();
  }, 1500);
}

// ---------------------------------------------------------------------------------------------------------------------
// Access to the analysis

const keyOf = (pkg, dr) => `${pkg.id}/${dr.id}`;
const packagesOf = (job) => (job && job.analysis && job.analysis.packages) || [];
const confOf = (job, pkg, dr) => (job.confirm && job.confirm[keyOf(pkg, dr)]) || {};
function ppmOf(job, pkg, dr) { const c = confOf(job, pkg, dr); return c.px_per_m || (dr.scale && dr.scale.px_per_m) || null; }
function selectedDrawings(job, pkg) { return pkg.drawings.filter((d) => { const c = confOf(job, pkg, d); return c.extract === undefined ? d.kind === 'floor plan' : !!c.extract; }); }
function polygonOf(job, pkg, dr) { const c = confOf(job, pkg, dr); return c.polygon_px || dr.polygon_px; }
function bboxOf(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  return { x0, y0, x1, y1 };
}
function storeyLabel(job, pkg, dr) { const c = confOf(job, pkg, dr); return c.storey || dr.storey || ''; }

/** The storeys of the project: every drawing selected for extraction, in the user's order. */
function storeyRows() {
  const rows = [];
  for (const id of P.sheets) {
    const job = J[id];
    if (!job || !job.analysis) continue;
    for (const pkg of packagesOf(job)) for (const dr of selectedDrawings(job, pkg)) {
      const key = `${id}|${keyOf(pkg, dr)}`;
      const res = job.results && job.results[keyOf(pkg, dr)];
      rows.push({ key, id, job, pkg, dr, conf: confOf(job, pkg, dr), res, label: storeyLabel(job, pkg, dr) });
    }
  }
  const order = new Map(P.order.map((k, i) => [k, i]));
  rows.sort((a, b) => (order.has(a.key) ? order.get(a.key) : 1e9) - (order.has(b.key) ? order.get(b.key) : 1e9));
  P.order = rows.map((r) => r.key);
  return rows;
}

function firstUnconfirmed() {
  for (let si = 0; si < P.sheets.length; si++) {
    const job = J[P.sheets[si]];
    if (!job || !job.analysis) continue;
    const pk = packagesOf(job);
    for (let pi = 0; pi < pk.length; pi++) {
      const anyUnconfirmed = pk[pi].drawings.some((d) => !confOf(job, pk[pi], d).area_confirmed);
      if (anyUnconfirmed) return { sheet: si, pkg: pi, drawing: 0 };
      const sel = selectedDrawings(job, pk[pi]);
      const di = sel.findIndex((d) => !confOf(job, pk[pi], d).scale_confirmed);
      if (di >= 0) return { sheet: si, pkg: pi, drawing: pk[pi].drawings.indexOf(sel[di]), scale: true };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------------------------------------------------
// Canvas viewer: an image with pan and zoom, overlays drawn in screen space, an optional tool

class Viewer {
  constructor(stage) {
    this.stage = stage;
    this.canvas = h('canvas');
    stage.append(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.img = null; this.imgAlpha = 1; this.view = { s: 1, x: 0, y: 0 }; this.fitted = false; this.fitPad = 48;
    this.overlay = () => {}; this.tool = null; this.onClick = null;
    this.w = 0; this.h = 0; this.dpr = window.devicePixelRatio || 1;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(stage);
    let down = null, moved = false, panning = false, dragTool = false;
    const pos = (e) => { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    this.canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const [sx, sy] = pos(e); const ip = this.toImage(sx, sy);
      down = { sx, sy, x: this.view.x, y: this.view.y }; moved = false; panning = false; dragTool = false;
      if (this.tool && this.tool.down && this.tool.down(ip, [sx, sy], this)) dragTool = true;
      else panning = true;
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointermove', (e) => {
      const [sx, sy] = pos(e); const ip = this.toImage(sx, sy);
      if (down) {
        if (Math.hypot(sx - down.sx, sy - down.sy) > 3) moved = true;
        if (dragTool) { this.tool.move(ip, [sx, sy], this); this.draw(); }
        else if (panning && moved) { this.view.x = down.x + sx - down.sx; this.view.y = down.y + sy - down.sy; this.canvas.classList.add('grabbing'); this.draw(); }
      } else if (this.tool && this.tool.hover) {
        this.tool.hover(ip, [sx, sy], this);
      }
    });
    const up = (e) => {
      if (!down) return;
      const [sx, sy] = pos(e); const ip = this.toImage(sx, sy);
      if (dragTool && this.tool.up) this.tool.up(ip, [sx, sy], this);
      if (!moved) {
        if (this.tool && this.tool.click) this.tool.click(ip, [sx, sy], this);
        else if (this.onClick) this.onClick(ip, [sx, sy], this);
      }
      down = null; this.canvas.classList.remove('grabbing'); this.draw();
    };
    this.canvas.addEventListener('pointerup', up);
    this.canvas.addEventListener('pointercancel', up);
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const [sx, sy] = pos(e);
      const f = Math.exp(-e.deltaY * 0.0015);
      const s = Math.min(40, Math.max(0.02, this.view.s * f));
      const k = s / this.view.s;
      this.view.x = sx - (sx - this.view.x) * k; this.view.y = sy - (sy - this.view.y) * k; this.view.s = s;
      this.draw();
    }, { passive: false });
  }
  destroy() { this.ro.disconnect(); }
  resize() {
    const r = this.stage.getBoundingClientRect();
    this.w = r.width; this.h = r.height; this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.w * this.dpr)); this.canvas.height = Math.max(1, Math.round(this.h * this.dpr));
    if (!this.fitted) this.fit();
    this.draw();
  }
  load(url) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => { this.img = im; this.fitted = false; this.fit(); this.draw(); resolve(im); };
      im.onerror = () => reject(new Error(`image ${url} could not be loaded`));
      im.src = url;
    });
  }
  fit() {
    if (!this.img || !this.w) return;
    const s = Math.min((this.w - 2 * this.fitPad) / this.img.width, (this.h - 2 * this.fitPad) / this.img.height);
    this.view = { s, x: (this.w - this.img.width * s) / 2, y: (this.h - this.img.height * s) / 2 };
    this.fitted = true;
  }
  toScreen(x, y) { return [x * this.view.s + this.view.x, y * this.view.s + this.view.y]; }
  toImage(sx, sy) { return [(sx - this.view.x) / this.view.s, (sy - this.view.y) / this.view.s]; }
  draw() {
    const c = this.ctx;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);
    if (this.img) {
      const [x, y] = this.toScreen(0, 0);
      c.save();
      c.shadowColor = 'rgba(0,0,0,.12)'; c.shadowBlur = 12; c.shadowOffsetY = 2;
      c.fillStyle = '#fff'; c.fillRect(x, y, this.img.width * this.view.s, this.img.height * this.view.s);
      c.restore();
      c.globalAlpha = this.imgAlpha;
      c.imageSmoothingEnabled = this.view.s < 1.5;
      c.drawImage(this.img, x, y, this.img.width * this.view.s, this.img.height * this.view.s);
      c.globalAlpha = 1;
    }
    this.overlay(c, this);
    if (this.tool && this.tool.draw) this.tool.draw(c, this);
  }
  // overlay helpers (points in image px)
  poly(pts, { stroke, fill, width = 1.5, dash, close = true } = {}) {
    if (!pts || pts.length < 2) return;
    const c = this.ctx;
    c.beginPath();
    pts.forEach(([x, y], i) => { const [sx, sy] = this.toScreen(x, y); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); });
    if (close) c.closePath();
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = width; c.setLineDash(dash || []); c.stroke(); c.setLineDash([]); }
  }
  rect(b, opts) { this.poly([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], opts); }
  label(x, y, text, { colour = ACCENT, bg = null, size = 11, dy = -6, align = 'left' } = {}) {
    const c = this.ctx; const [sx, sy] = this.toScreen(x, y);
    c.font = `600 ${size}px 'IBM Plex Sans', system-ui, sans-serif`; c.textAlign = align; c.textBaseline = 'bottom';
    if (bg) { const w = c.measureText(text).width; const lx = align === 'center' ? sx - w / 2 - 6 : sx - 2; c.fillStyle = bg; c.beginPath(); c.roundRect(lx, sy + dy - size - 6, w + 12, size + 8, 4); c.fill(); c.fillStyle = '#fff'; c.fillText(text, sx + (align === 'center' ? 0 : 4), sy + dy - 1); }
    else { c.lineWidth = 3; c.strokeStyle = 'rgba(255,255,255,.9)'; c.strokeText(text, sx, sy + dy); c.fillStyle = colour; c.fillText(text, sx, sy + dy); }
  }
  handle(x, y, r = 6, colour = ACCENT) {
    const c = this.ctx; const [sx, sy] = this.toScreen(x, y);
    c.beginPath(); c.arc(sx, sy, r + 2, 0, Math.PI * 2); c.fillStyle = '#fff'; c.fill();
    c.beginPath(); c.arc(sx, sy, r, 0, Math.PI * 2); c.fillStyle = colour; c.fill();
  }
}

/** Building-area tool: a rectangle with corner and edge handles, movable from inside. */
class RectTool {
  constructor(rect, onChange) { this.r = rect; this.onChange = onChange; this.drag = null; }
  handles() {
    const r = this.r, mx = (r.x0 + r.x1) / 2, my = (r.y0 + r.y1) / 2;
    return [['nw', r.x0, r.y0], ['ne', r.x1, r.y0], ['sw', r.x0, r.y1], ['se', r.x1, r.y1], ['n', mx, r.y0], ['s', mx, r.y1], ['w', r.x0, my], ['e', r.x1, my]];
  }
  hit(sp, v) {
    for (const [id, x, y] of this.handles()) { const [sx, sy] = v.toScreen(x, y); if (Math.hypot(sx - sp[0], sy - sp[1]) <= 10) return id; }
    const [ix, iy] = v.toImage(sp[0], sp[1]);
    if (ix > this.r.x0 && ix < this.r.x1 && iy > this.r.y0 && iy < this.r.y1) return 'move';
    return null;
  }
  hover(ip, sp, v) {
    const id = this.hit(sp, v);
    v.canvas.style.cursor = !id ? 'grab' : id === 'move' ? 'move' : { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', w: 'ew-resize', e: 'ew-resize' }[id];
  }
  down(ip, sp, v) { const id = this.hit(sp, v); if (!id) return false; this.drag = { id, start: ip, r0: { ...this.r } }; return true; }
  move(ip) {
    if (!this.drag) return;
    const { id, start, r0 } = this.drag; const dx = ip[0] - start[0], dy = ip[1] - start[1]; const r = { ...r0 };
    if (id === 'move') { r.x0 += dx; r.x1 += dx; r.y0 += dy; r.y1 += dy; }
    else { if (id.includes('w')) r.x0 = ip[0]; if (id.includes('e')) r.x1 = ip[0]; if (id.includes('n')) r.y0 = ip[1]; if (id.includes('s')) r.y1 = ip[1]; }
    this.r = { x0: Math.min(r.x0, r.x1), y0: Math.min(r.y0, r.y1), x1: Math.max(r.x0, r.x1), y1: Math.max(r.y0, r.y1) };
  }
  up() { this.drag = null; this.onChange(this.r); }
  draw(c, v) {
    v.rect(this.r, { stroke: ACCENT, width: 2, fill: 'rgba(47,110,166,.05)' });
    for (const [id, x, y] of this.handles()) {
      if (id.length === 2) v.handle(x, y, 6);
      else { const [sx, sy] = v.toScreen(x, y); c.fillStyle = ACCENT; c.beginPath(); c.roundRect(sx - (id === 'n' || id === 's' ? 12 : 4), sy - (id === 'n' || id === 's' ? 4 : 12), id === 'n' || id === 's' ? 24 : 8, id === 'n' || id === 's' ? 8 : 24, 4); c.fill(); }
    }
  }
}

/** Two-point measurement: click twice, the distance in image px is shown; a third click starts over. */
class MeasureTool {
  constructor(onChange) { this.pts = []; this.onChange = onChange; this.cursor = null; }
  click(ip) { if (this.pts.length >= 2) this.pts = []; this.pts.push(ip); this.onChange(this.pts); }
  hover(ip, sp, v) { this.cursor = ip; v.canvas.style.cursor = 'crosshair'; if (this.pts.length === 1) v.draw(); }
  reset() { this.pts = []; this.onChange(this.pts); }
  draw(c, v) {
    const pts = this.pts.length === 1 && this.cursor ? [this.pts[0], this.cursor] : this.pts;
    if (pts.length === 2) {
      v.poly(pts, { stroke: ACCENT, width: 3, close: false });
      const d = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
      v.label((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2, `${Math.round(d * (this.pxScale || 1))} px`, { bg: ACCENT, align: 'center', dy: -10 });
    }
    for (const p of this.pts) v.handle(p[0], p[1], 7);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Screens

let viewer = null;
function screenEl() { return $('#screen'); }
function mount(node, crumb) {
  if (viewer) { viewer.destroy(); viewer = null; }
  const s = screenEl(); put(s, node);
  const c = $('#crumb'); c.hidden = !crumb; $('#crumb-text').textContent = crumb || '';
  $('#btn-start-over').hidden = !P.sheets.length;
}
function go(name, cursor) {
  if (cursor) P.cursor = Object.assign({ sheet: 0, pkg: 0, drawing: 0 }, cursor);
  P.screen = name; screenName = name; saveProject();
  ({ upload: renderUpload, area: renderArea, scale: renderScale, storeys: renderStoreys, results: renderResults }[name] || renderUpload)();
}
function onJobsChanged() {
  if (screenName === 'upload') renderUpload({ keepInput: true });
  else if (screenName === 'storeys') renderStoreys({ patch: true });
  else if (screenName === 'area' || screenName === 'scale') {
    const job = J[P.sheets[P.cursor.sheet]];
    if (job && !busy(job)) go(screenName);
  }
}
function projectName() {
  if (P.name) return P.name;
  const rows = storeyRows();
  if (rows.length) { const j = rows[0].job; return (j.name || '').replace(/\.[^.]+$/, ''); }
  const id = P.sheets[0]; return id && J[id] ? (J[id].name || '').replace(/\.[^.]+$/, '') : '';
}
function errorBox(job, text) {
  const pre = h('pre', { hidden: true }, (job.log || []).slice(-25).join('\n'));
  return h('div', { class: 'error-box' },
    h('div', null, h('b', null, 'Error: '), text || job.error || 'unknown error'),
    h('div', null, h('button', { class: 'btn-link', onclick: () => { pre.hidden = !pre.hidden; } }, 'Show log'),
      ' · ', h('a', { href: `/api/sheets/${job.id}/log`, target: '_blank' }, 'full log')), pre);
}

// --- Upload (board 1c) ---------------------------------------------------------------------------------------------

let health = null;
async function renderUpload({ keepInput } = {}) {
  const adding = storeyRows().length > 0;
  const input = h('input', { type: 'file', multiple: true, hidden: true, accept: '.pdf,.png,.jpg,.jpeg,.tif,.tiff,.dxf,.dwg', onchange: (e) => uploadFiles([...e.target.files]) });
  const drop = h('div', { class: 'dropzone', onclick: () => input.click(),
    ondragover: (e) => { e.preventDefault(); drop.classList.add('over'); }, ondragleave: () => drop.classList.remove('over'),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove('over'); uploadFiles([...e.dataTransfer.files]); } },
    h('div', { class: 'arrow' }, '↑'), h('div', { class: 'title' }, 'Drop floor plans here'),
    h('div', { class: 'muted small' }, 'PDF, JPG, PNG, TIFF, DXF · up to 300 MB each · several storeys at once'),
    h('button', { class: 'btn btn-primary', style: 'margin-top:6px;height:38px;padding:0 18px', onclick: (e) => { e.stopPropagation(); input.click(); } }, 'Choose files'));
  const list = h('div', { class: 'file-list' });
  for (const id of P.sheets) {
    const job = J[id]; if (!job) continue;
    const rows = packagesOf(job);
    const n = rows.reduce((a, p) => a + p.drawings.length, 0);
    const st = job.status;
    const sub = st === 'analysing' ? (job.log && job.log.length ? job.log[job.log.length - 1].slice(9) : 'analysing…')
      : st === 'error' ? job.error : st === 'ready' ? `analysed · ${rows.length} page${rows.length === 1 ? '' : 's'}, ${n} drawing${n === 1 ? '' : 's'}`
      : st === 'done' ? 'extracted' : st;
    const badge = st === 'analysing' ? h('span', { class: 'badge badge-run' }, h('span', { class: 'spinner' }), 'Analysing')
      : st === 'error' ? h('span', { class: 'badge badge-err' }, h('span', { class: 'dot' }), 'Error')
      : st === 'ready' ? h('span', { class: 'badge badge-ok' }, h('span', { class: 'dot' }), 'Ready')
      : h('span', { class: 'badge badge-wait' }, h('span', { class: 'dot' }), st);
    list.append(h('div', { class: 'file-row' },
      h('div', { style: 'min-width:0' }, h('div', { class: 'name' }, job.name), h('div', { class: 'sub', title: sub }, sub)),
      h('div', { class: 'actions' }, badge,
        st === 'error' ? h('button', { class: 'btn', onclick: () => alert((job.log || []).join('\n')) }, 'Log') : null,
        h('button', { class: 'btn-x', title: 'Remove', onclick: () => removeSheet(id) }, '×'))));
  }
  const anyReady = P.sheets.some((id) => J[id] && (J[id].status === 'ready' || J[id].status === 'done'));
  const anyBusy = P.sheets.some((id) => J[id] && busy(J[id]));
  const cont = anyReady && !anyBusy ? h('button', { class: 'btn btn-primary btn-lg', onclick: continueFromUpload }, 'Continue to step 1 ›') : null;
  const steps = h('div', { class: 'steps3' },
    stepCard('Step 1', 'Building area', 'Confirm the detected outline, leave out the title block.'),
    stepCard('Step 2', 'Scale', 'Read from dimension strings or the title block; measure if neither exists.'),
    stepCard('Step 3', 'Storeys', 'Order the sheets, then run the extraction.'));
  const status = h('div', { class: 'status-line' }, h('span', { class: 'spinner' }), 'Checking the local pipeline…');
  const main = h('main', { class: 'upload grid-bg' }, h('div', { class: 'upload-inner' },
    h('div', { style: 'display:flex;flex-direction:column;gap:4px' }, h('h2', null, adding ? 'Add a storey' : 'New project'),
      h('p', { class: 'muted pretty' }, adding ? 'Upload another sheet of the same building. It goes through the same two checks, then joins the storeys list.'
        : 'Upload one sheet per storey. Vector PDFs and scans both work; scans should be 300 dpi or better.')),
    drop, input, P.sheets.length ? list : null, cont, adding ? h('button', { class: 'btn btn-lg', onclick: () => go('storeys') }, '‹ Back to the storeys') : steps, status));
  mount(main, adding ? projectName() : null);
  startPolling();
  if (!health) { try { health = await getJSON('/api/health'); } catch (e) { health = { ok: false, items: { server: { ok: false, detail: e.message } } }; } }
  if (screenName !== 'upload') return;
  put(status, h('span', { class: 'dot10', style: `background:${health.ok ? '#2e9e5b' : '#d0433b'}` }),
    health.ok ? 'Processing runs on this machine. Plans are not uploaded anywhere.' : 'Something the pipeline needs is missing on this machine:');
  const rows = Object.entries(health.items || {}).map(([k, v]) => h('div', { class: 'row' }, h('span', { class: v.ok ? 'ok' : 'bad' }, v.ok ? '✓' : '✗'), h('b', null, k), h('span', { class: 'muted' }, v.detail)));
  const det = h('details', null, h('summary', null, `Pipeline check (${health.pipeline_dir || ''})`), h('div', { class: 'health' }, rows));
  if (!health.ok) det.open = true;
  status.after(det);
}
function stepCard(step, name, desc) { return h('div', { class: 'step-card' }, h('div', { class: 'eyebrow' }, step), h('div', { class: 'name' }, name), h('div', { class: 'desc' }, desc)); }

async function uploadFiles(files) {
  for (const f of files) {
    try {
      const r = await api(`/api/upload?name=${encodeURIComponent(f.name)}`, { method: 'POST', body: f, headers: { 'Content-Type': 'application/octet-stream' } });
      P.sheets.push(r.id); saveProject();
      await refreshJob(r.id);
    } catch (e) {
      alert(`${f.name}: ${e.message}`);
    }
  }
  renderUpload();
}
async function removeSheet(id) {
  if (!confirm('Remove this sheet and its files?')) return;
  try { await api(`/api/sheets/${id}`, { method: 'DELETE' }); } catch (e) { /* already gone */ }
  P.sheets = P.sheets.filter((s) => s !== id); delete J[id]; saveProject();
  renderUpload();
}
function continueFromUpload() {
  const c = firstUnconfirmed();
  if (!c) return go('storeys');
  go(c.scale ? 'scale' : 'area', c);
}
// after an upload finishes analysing, move on by itself when nothing else is pending
let autoAdvanced = new Set();
setInterval(() => {
  if (screenName !== 'upload') return;
  const pending = P.sheets.filter((id) => J[id] && busy(J[id]));
  const fresh = P.sheets.filter((id) => J[id] && J[id].status === 'ready' && !autoAdvanced.has(id));
  if (!pending.length && fresh.length) { fresh.forEach((id) => autoAdvanced.add(id)); continueFromUpload(); }
}, 800);

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
    h('div', { class: 'progress-bars' }, h('span', { class: 'on' }), h('span'), h('span')),
    h('div', null, h('div', { class: 'eyebrow' }, 'Step 1 of 3 · Building area'), h('h2', null, 'Confirm the building area')),
    h('p', { class: 'muted pretty' }, 'The outline was detected from the drawing. Drag the handles so the whole building is inside and the title block, legend and dimension chains stay out.'),
    detectedCard(job, pkg), list,
    pkg.drawings.length === 0 ? h('div', { class: 'warn-box' }, 'No drawing was found on this sheet. It can only be skipped.') : null,
    h('div', { class: 'foot' },
      h('button', { class: 'btn btn-lg', onclick: () => confirmArea(job, pkg, rects, new Set()) }, 'Skip this sheet'),
      h('button', { class: 'btn btn-primary btn-lg', disabled: !pkg.drawings.length, onclick: () => confirmArea(job, pkg, rects, sel) }, 'Confirm area')));
  mount(h('div', { class: 'step' }, stage, panel), projectName());
  viewer = new Viewer(stage);
  const tool = new RectTool(current ? scaleRect(rects[current.id], f) : { x0: 0, y0: 0, x1: 0, y1: 0 }, (r) => { if (current) rects[current.id] = scaleRect(r, 1 / f); drawList(); });
  if (current) viewer.tool = tool;
  function drawList() {
    put(list, h('div', { class: 'section-title' }, `${pkg.drawings.length} drawing${pkg.drawings.length === 1 ? '' : 's'} on this sheet`));
    for (const d of pkg.drawings) {
      const r = rects[d.id]; const ppm = ppmOf(job, pkg, d);
      const size = ppm ? `${((r.x1 - r.x0) / ppm).toFixed(1)} × ${((r.y1 - r.y0) / ppm).toFixed(1)} m` : `${Math.round(r.x1 - r.x0)} × ${Math.round(r.y1 - r.y0)} px`;
      const cb = h('input', { type: 'checkbox', checked: sel.has(d.id), onclick: (e) => { e.stopPropagation(); }, onchange: (e) => { e.target.checked ? sel.add(d.id) : sel.delete(d.id); drawList(); viewer.draw(); } });
      list.append(h('div', { class: 'drawing-row' + (d === current ? ' on' : ''), onclick: () => { current = d; tool.r = scaleRect(rects[d.id], f); viewer.tool = tool; drawList(); viewer.draw(); } },
        cb, h('div', { style: 'min-width:0' }, h('div', { class: 't' }, d.title || `${d.kind} ${d.id}`), h('div', { class: 's' }, `${d.id} · ${size}${d.storey ? ' · ' + d.storey : ''}${d.kind_confidence === 'low' ? ' · kind assumed' : ''}`)),
        h('span', { class: 'kind' }, d.kind)));
    }
  }
  drawList();
  viewer.overlay = (c, v) => {
    for (const r of pkg.regions) {
      if (r.class === 'drawing' || r.class === 'frame' || !r.polygon_px || r.polygon_px.length < 3) continue;
      const col = REGION_COLOURS[r.class] || '#777';
      v.poly(r.polygon_px.map(([x, y]) => [x * f, y * f]), { stroke: col, width: 1, dash: [4, 3], fill: col + '14' });
      const b = bboxOf(r.polygon_px); v.label(b.x0 * f, b.y0 * f, r.class, { colour: col, size: 10, dy: -2 });
    }
    for (const d of pkg.drawings) {
      const pts = d.polygon_px.map(([x, y]) => [x * f, y * f]);
      if (d === current) v.poly(pts, { stroke: ACCENT, width: 1, dash: [5, 4] });
      else { v.poly(pts, { stroke: sel.has(d.id) ? ACCENT : '#8a8f99', width: 1.5, dash: [6, 4] }); const b = bboxOf(pts); v.label(b.x0, b.y0, `${d.kind}${sel.has(d.id) ? '' : ' (not extracted)'}`, { colour: sel.has(d.id) ? ACCENT : '#646b74' }); }
    }
    if (current) {
      const r = rects[current.id]; const ppm = ppmOf(job, pkg, current);
      const size = ppm ? `${((r.x1 - r.x0) / ppm).toFixed(1)} × ${((r.y1 - r.y0) / ppm).toFixed(1)} m` : `${Math.round(r.x1 - r.x0)} × ${Math.round(r.y1 - r.y0)} px`;
      v.label(r.x0 * f, r.y0 * f, `Detected building · ${size}`, { dy: -14 });
    }
  };
  viewer.load(`/api/sheets/${id}/file/${pkg.preview.file}`).catch((e) => panel.prepend(errorBox(job, e.message)));
}
function scaleRect(r, k) { return { x0: r.x0 * k, y0: r.y0 * k, x1: r.x1 * k, y1: r.y1 * k }; }
function detectedCard(job, pkg) {
  const s = pkg.summary; const lay = pkg.layout;
  const title = (pkg.regions.find((r) => r.class === 'notes' && r.role === 'sheet title') || {}).text;
  const dpi = s.dpi ? `${Math.round(s.dpi)} dpi · ${DPI_SOURCE[s.dpi_source] || s.dpi_source}` : 'unknown';
  const paper = s.paper_mm ? `${Math.round(s.paper_mm[0])} × ${Math.round(s.paper_mm[1])} mm` : `${s.raster_px[0]} × ${s.raster_px[1]} px`;
  const warn = (s.warnings || []).concat(pkg.layout_flags || []);
  return h('div', { class: 'card' }, h('div', { class: 'section-title' }, 'Detected on this sheet'),
    h('div', { class: 'kv' }, h('span', null, 'Input class'), h('span', null, s.input_class + (s.text_runs && s.text_runs.native ? ', native text' : ''))),
    h('div', { class: 'kv' }, h('span', null, 'Resolution'), h('span', null, dpi)),
    h('div', { class: 'kv' }, h('span', null, 'Sheet'), h('span', null, paper + (pkg.deskew && pkg.deskew.applied ? ` · deskewed ${pkg.deskew.deg}°` : ''))),
    title ? h('div', { class: 'kv' }, h('span', null, 'Sheet title'), h('span', null, title.slice(0, 60))) : null,
    lay.title_block ? h('div', { class: 'kv' }, h('span', null, 'Title block'), h('span', null, 'found' + (lay.sheet_scale_notes.length ? ` · scale ${lay.sheet_scale_notes.map((n) => '1:' + n).join(', ')}` : ''))) : null,
    warn.length ? h('details', null, h('summary', null, `${warn.length} note${warn.length === 1 ? '' : 's'} from the loader`), h('ul', { style: 'margin:4px 0 0;padding-left:16px;font-size:12px;color:#646b74' }, warn.map((w) => h('li', null, w.message)))) : null);
}
async function confirmArea(job, pkg, rects, sel) {
  const body = {};
  for (const d of pkg.drawings) {
    const r = rects[d.id]; const det = bboxOf(d.polygon_px);
    const changed = ['x0', 'y0', 'x1', 'y1'].some((k) => Math.abs(r[k] - det[k]) > 0.5);
    body[keyOf(pkg, d)] = { extract: sel.has(d.id), area_confirmed: true,
      polygon_px: changed ? [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]].map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]) : null };
  }
  try { const r = await postJSON(`/api/sheets/${job.id}/confirm`, { confirm: body }); job.confirm = r.confirm; } catch (e) { return alert(e.message); }
  const first = selectedDrawings(job, pkg)[0];
  if (first) go('scale', { ...P.cursor, drawing: pkg.drawings.indexOf(first) });
  else nextPackage();
}
function nextPackage() {
  const job = J[P.sheets[P.cursor.sheet]];
  if (P.cursor.pkg + 1 < packagesOf(job).length) return go('area', { ...P.cursor, pkg: P.cursor.pkg + 1, drawing: 0 });
  const c = firstUnconfirmed();
  if (c) return go(c.scale ? 'scale' : 'area', c);
  go('storeys');
}
function renderWaiting(job) {
  const box = h('div', { class: 'upload grid-bg' }, h('div', { class: 'upload-inner' }, h('div', { class: 'card' },
    job && job.status === 'error' ? errorBox(job) : h('div', { style: 'display:flex;gap:10px;align-items:center' }, h('span', { class: 'spinner' }), `Analysing ${job ? job.name : 'the sheet'}… `, h('span', { class: 'muted' }, job && job.log && job.log.length ? job.log[job.log.length - 1].slice(9) : '')),
    h('div', null, h('button', { class: 'btn', onclick: () => go('upload') }, '‹ Back to the upload')))));
  mount(box, projectName());
}

// --- Step 2: plan scale (boards 2a, 2b, 2c) ------------------------------------------------------------------------

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
  const btnReset = h('button', { class: 'btn btn-round', title: 'Start the measurement over', onclick: () => { if (tool) tool.reset(); } }, '↻');
  const btnHelp = h('button', { class: 'btn btn-round', style: 'font-weight:600;font-size:15px', onclick: () => { $('#help').hidden = false; } }, '?');
  stage.append(h('div', { class: 'stage-top' }, topPill, btnReset, btnHelp));
  const panel = h('aside', { class: 'panel' });
  mount(h('div', { class: 'step' }, stage, panel), projectName());
  viewer = new Viewer(stage);
  let tool = null;
  const measure = new MeasureTool((pts) => { st.pts = pts.map((p) => [p[0] / f, p[1] / f]); renderPanel(); });
  measure.pxScale = 1 / f;
  viewer.overlay = (c, v) => {
    v.poly(polygonOf(job, pkg, dr).map(([x, y]) => [x * f, y * f]), { stroke: ACCENT, width: 1.5, dash: [6, 4] });
    for (const b of (s.cue_boxes || [])) {
      const [x0, y0, x1, y1] = b.box;
      v.rect({ x0: x0 * f, y0: y0 * f, x1: x1 * f, y1: y1 * f }, { stroke: AMBER, width: 3, fill: 'rgba(217,154,30,.08)' });
      if (b.cue !== 'dimension_strings') v.label(x0 * f, y0 * f, b.label, { colour: '#8a5a00', dy: -8 });
    }
    if (st.mode === 'measured' && st.measured && st.measured.p1) {
      const p = [st.measured.p1, st.measured.p2].map(([x, y]) => [x * f, y * f]);
      v.poly(p, { stroke: ACCENT, width: 3, close: false }); p.forEach(([x, y]) => v.handle(x, y, 7));
      v.label((p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2, `${Math.round(st.measured.px)} px`, { bg: ACCENT, align: 'center', dy: -10 });
    }
  };
  viewer.load(`/api/sheets/${id}/file/${pkg.preview.file}`).then(() => {
    const b = bboxOf(polygonOf(job, pkg, dr));
    const cues = (s.cue_boxes || []).map((c) => c.box);
    const all = [[b.x0, b.y0], [b.x1, b.y1], ...cues.flatMap((c) => [[c[0], c[1]], [c[2], c[3]]])];
    const bb = bboxOf(all); const pad = 60;
    const sc = Math.min((viewer.w - 2 * pad) / ((bb.x1 - bb.x0) * f), (viewer.h - 2 * pad) / ((bb.y1 - bb.y0) * f), viewer.view.s * 3);
    viewer.view = { s: sc, x: (viewer.w - (bb.x0 + bb.x1) * f * sc) / 2, y: (viewer.h - (bb.y0 + bb.y1) * f * sc) / 2 }; viewer.draw();
  }).catch((e) => panel.prepend(errorBox(job, e.message)));

  function renderPanel() {
    const ppm = ppmNow(); const n = N(ppm);
    const measuring = st.mode === 'measure';
    topPill.className = 'pill' + (measuring ? ' accent' : '');
    if (measuring) {
      const step = st.pts.length === 0 ? 'Click the first point.' : st.pts.length === 1 ? 'Click the second point.' : 'Enter the real distance.';
      put(topPill, h('span', { style: 'color:oklch(0.42 0.11 230)' }, 'Measure a known distance'), h('b', null, step));
    } else put(topPill, h('span', { class: 'muted' }, 'Reviewing floor plan'), h('b', null, dr.title || job.name), sel.length > 1 ? h('span', { class: 'muted' }, `· drawing ${di + 1} of ${sel.length}`) : null);
    btnReset.hidden = !measuring;
    const head = [h('div', { class: 'progress-bars' }, h('span', { class: 'on' }), h('span', { class: 'on' }), h('span')),
      h('div', null, h('div', { class: 'eyebrow' }, 'Step 2 of 3 · Plan scale'), h('h2', { style: 'text-align:center;font-size:19px' }, 'Check the detected scale'))];
    const dpiRow = () => {
      const src = DPI_SOURCE[pkg.summary.dpi_source] || pkg.summary.dpi_source;
      const inp = h('input', { type: 'number', min: 10, max: 2400, step: 1, value: st.dpi || '', style: 'width:90px;height:32px', onchange: (e) => { const v = parseFloat(e.target.value); if (v > 0) { st.dpi = v; st.dpiChanged = true; renderPanel(); } } });
      const change = h('button', { class: 'btn btn-pill', style: 'height:28px;padding:0 10px;font-size:12px', onclick: () => { put(right, inp, h('span', { class: 'muted small' }, 'dpi')); inp.focus(); } }, 'Change');
      const right = h('span', { style: 'display:flex;align-items:center;gap:10px' }, h('span', { class: 'big' }, st.dpi ? `${Math.round(st.dpi)} dpi` : 'unknown'), change);
      return h('div', { class: 'kv', style: 'align-items:center' }, h('span', { style: 'display:flex;flex-direction:column' }, 'Resolution', h('span', { class: 'muted small' }, st.dpiChanged ? 'set by hand' : src)), right);
    };
    const pixelRow = (sep) => h('div', { class: 'kv' + (sep ? ' sep' : '') }, h('span', null, 'Pixel size'), h('span', null, ppm ? `${ppm.toFixed(1)} px per metre` : '–', ppm ? h('span', { class: 'muted' }, ` · ${(1000 / ppm).toFixed(1)} mm per pixel`) : null));
    let body;
    if (measuring) {
      const stepN = st.pts.length < 2 ? st.pts.length + 1 : 3;
      const stepText = ['Click the first point.', 'Click the second point.', 'Enter the real distance.'][stepN - 1];
      const inp = h('input', { type: 'text', class: 'input-xl', value: st.distance, placeholder: 'e.g. 22.55', oninput: (e) => { st.distance = e.target.value; renderPanel(); } });
      const unit = h('select', { class: 'input-xl', style: 'padding:0 8px', onchange: (e) => { st.unit = e.target.value; renderPanel(); } }, Object.keys(UNITS).map((u) => h('option', { value: u, selected: u === st.unit }, u)));
      const px = st.pts.length === 2 ? Math.hypot(st.pts[1][0] - st.pts[0][0], st.pts[1][1] - st.pts[0][1]) : null;
      body = [
        h('div', { class: 'kv', style: 'align-items:center' }, h('span', { style: 'font-weight:600;font-size:15px;color:var(--text)' }, 'Measure a known distance'), h('button', { class: 'btn-link', style: 'color:var(--muted)', onclick: () => { st.mode = conf.measured ? 'measured' : 'check'; viewer.tool = null; viewer.canvas.style.cursor = 'grab'; renderPanel(); viewer.draw(); } }, 'Cancel')),
        h('p', { class: 'muted small pretty', style: 'margin-top:-6px' }, 'For an image, the scale can only be corrected by measuring a known distance: a dimension string, a scale bar or a known room width.'),
        h('div', { class: 'info-box' }, h('span', { class: 'num' }, String(stepN)), h('span', { style: 'font-weight:600' }, stepText)),
        h('div', { class: 'measure-grid' }, inp, unit),
        h('div', { class: 'scale-rows' },
          h('div', { class: 'kv' }, h('span', null, 'Measured'), h('span', null, px ? `${Math.round(px)} px on the sheet` : '–')),
          h('div', { class: 'kv' }, h('span', null, 'Calculated scale'), h('span', { style: 'font-weight:600;color:var(--accent-text);font-size:18px' }, n ? fmt.scaleN(n) : ppm ? `${ppm.toFixed(1)} px/m` : '–')),
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
      rows.push(h('div', { class: 'kv' }, h('span', null, 'Calculated scale'), h('span', { class: 'big', style: 'color:var(--accent-text)' }, n ? fmt.scaleN(n) : '–')));
    } else {
      if (kind === 'bar') { const ev = (s.agreeing || []).find((a) => a.cue === 'scale_bar' || a.cue === 'linked'); rows.push(h('div', { class: 'kv' }, h('span', null, 'Scale bar'), h('span', { class: 'small' }, ev ? ev.evidence.replace(/\(fit residual.*\)/, '').slice(0, 60) : ''))); }
      if (kind === 'dims') { const ev = (s.agreeing || []).find((a) => a.cue === 'dimension_strings'); rows.push(h('div', { class: 'kv' }, h('span', null, 'Dimension strings'), h('span', { class: 'small' }, ev ? ev.evidence.split(';')[0].slice(0, 70) : ''))); }
      const shown = !st.dpiChanged && s.scale && Math.abs((ppm || 0) - (ppmProposal || 0)) < 1e-6 ? s.scale.replace(':', ' : ') : fmt.scaleN(n);
      rows.push(h('div', { class: 'kv' }, h('span', null, 'Scale'), h('span', { class: 'big' }, n || s.scale ? shown : ppm ? `${ppm.toFixed(1)} px/m` : '–')));
    }
    rows.push(dpiRow(), pixelRow(true));
    const agree = s.agreeing || [], disagree = s.disagreeing || [];
    const cueList = h('details', null, h('summary', null, `Cues: ${agree.length} agree, ${disagree.length} disagree · confidence `, h('b', { class: 'conf-' + (st.mode === 'measured' ? 'confirmed' : s.confidence || 'none') }, st.mode === 'measured' ? 'measured by hand' : s.confidence || 'none')),
      h('div', { class: 'cues', style: 'margin-top:8px' },
        agree.map((a) => h('div', { class: 'cue agree' }, h('span', { class: 'mark' }, '✓'), h('div', null, h('b', null, a.cue.replace(/_/g, ' ')), ` ${a.px_per_m.toFixed(1)} px/m`, h('div', { class: 'ev' }, a.evidence)))),
        disagree.map((a) => h('div', { class: 'cue disagree' }, h('span', { class: 'mark' }, '!'), h('div', null, h('b', null, a.cue.replace(/_/g, ' ')), ` ${a.px_per_m.toFixed(1)} px/m (${a.ratio}x)`, h('div', { class: 'ev' }, a.evidence)))),
        (s.flags || []).map((fl) => h('div', { class: 'cue' }, h('span', { class: 'sev sev-' + fl.severity }, fl.severity[0]), h('div', { class: 'ev' }, fl.message)))));
    body = [h('p', { class: 'muted pretty' }, st.mode === 'measured' ? 'The scale was set by measuring a known distance on the drawing.' : text),
      warn && st.mode !== 'measured' ? h('div', { class: 'warn-box pretty' }, warn) : null,
      h('div', { class: 'scale-rows' }, rows), cueList,
      h('button', { class: 'btn btn-pill btn-lg', style: 'font-weight:600', onclick: () => { st.mode = 'measure'; st.pts = []; measure.pts = []; viewer.tool = measure; tool = measure; renderPanel(); viewer.draw(); } }, st.mode === 'measured' ? 'Measure again' : 'Correct by measuring')];
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
    try { const r = await postJSON(`/api/sheets/${job.id}/confirm`, { confirm: { [keyOf(pkg, dr)]: body } }); job.confirm = r.confirm; } catch (e) { return alert(e.message); }
    const next = sel[di + 1];
    if (next) go('scale', { ...P.cursor, drawing: pkg.drawings.indexOf(next) });
    else nextPackage();
  }
  renderPanel();
}

// --- Step 3: storeys and run (board 1f) ----------------------------------------------------------------------------

let storeyView = { index: 0 };
function renderStoreys({ patch } = {}) {
  const rows = storeyRows();
  if (patch && screenName === 'storeys' && $('#storey-list')) { fillStoreyList(rows); updateRunState(rows); return; }
  storeyView.index = Math.min(storeyView.index, Math.max(0, rows.length - 1));
  const stage = h('main', { class: 'stage grid-bg' });
  const pill = h('div', { class: 'pill', style: 'padding:0 6px;gap:10px' });
  stage.append(h('div', { class: 'stage-top' }, pill));
  const listEl = h('div', { class: 'storey-list', id: 'storey-list' });
  const runBtn = h('button', { class: 'btn btn-primary btn-xl', id: 'btn-run', disabled: !rows.length, onclick: runExtraction }, 'Run extraction');
  const runNote = h('div', { class: 'muted small', id: 'run-note', style: 'text-align:center' }, 'About 1 minute per sheet on this machine, mostly OCR.');
  const n = rows.length;
  const panel = h('aside', { class: 'panel' },
    h('div', { class: 'progress-bars' }, h('span', { class: 'on' }), h('span', { class: 'on' }), h('span', { class: 'on' })),
    h('div', null, h('div', { class: 'eyebrow' }, 'Step 3 of 3 · Storeys'), h('h2', null, n ? 'Ready to extract' : 'No storey yet')),
    h('p', { class: 'muted pretty' }, n === 1 ? 'One sheet is confirmed. Add more storeys of the same building, or run the extraction now.' : n ? `${n} drawings are confirmed. Order them from the lowest storey up, name them, then run the extraction.` : 'Every drawing was skipped. Upload a sheet or go back and select a drawing.'),
    h('div', { style: 'display:flex;flex-direction:column;gap:8px' }, h('div', { class: 'section-title' }, `${n} storey${n === 1 ? '' : 's'}`), listEl,
      h('button', { class: 'add-storey', onclick: () => go('upload') }, '+ Add storey')),
    h('div', { id: 'run-errors' }),
    h('div', { style: 'margin-top:auto;display:flex;flex-direction:column;gap:8px' }, runBtn, runNote));
  mount(h('div', { class: 'step' }, stage, panel), projectName());
  viewer = new Viewer(stage);
  function showStorey(i) {
    storeyView.index = i; const r = rows[i];
    put(pill, h('button', { class: 'btn-x', onclick: () => showStorey((i - 1 + rows.length) % rows.length) }, '‹'),
      h('b', null, r ? (r.label || r.dr.title || r.dr.id) : '–'), h('span', { class: 'muted' }, r ? `${r.job.name} · ${scaleText(r)}` : 'no storey'),
      h('button', { class: 'btn-x', onclick: () => showStorey((i + 1) % rows.length) }, '›'));
    if (!r) { viewer.img = null; viewer.draw(); return; }
    const f = r.pkg.preview.scale;
    viewer.overlay = (c, v) => { v.poly(polygonOf(r.job, r.pkg, r.dr).map(([x, y]) => [x * f, y * f]), { stroke: ACCENT, width: 1.5, dash: [6, 4] }); };
    viewer.load(`/api/sheets/${r.id}/file/${r.pkg.preview.file}`).catch(() => {});
    fillStoreyList(rows);
  }
  fillStoreyList(rows); updateRunState(rows); showStorey(storeyView.index);
  startPolling();

  function fillStoreyList(rows) {
    listEl.replaceChildren();
    rows.forEach((r, i) => {
      const inp = h('input', { type: 'text', value: r.label, placeholder: 'Storey', title: 'Storey name (EG, 1. OG, ...)', onchange: async (e) => { r.conf.storey = e.target.value; try { await postJSON(`/api/sheets/${r.id}/confirm`, { confirm: { [keyOf(r.pkg, r.dr)]: { storey: e.target.value } } }); } catch (err) { alert(err.message); } } });
      const grip = h('span', { class: 'grip' }, h('button', { title: 'Move up', onclick: () => moveStorey(i, -1) }, '▲'), h('button', { title: 'Move down', onclick: () => moveStorey(i, 1) }, '▼'));
      const meta = [scaleText(r), r.conf.polygon_px ? 'area set by hand' : 'area confirmed', r.pkg.summary.input_class];
      const status = runStatus(r);
      listEl.append(h('div', { class: 'storey-row' + (i === storeyView.index ? ' on' : ''), onclick: (e) => { if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') showStorey(i); } },
        grip, inp, h('div', { class: 'f' }, h('div', { class: 'n' }, r.job.name + (packagesOf(r.job).length > 1 ? ` · p. ${r.pkg.index + 1}` : '') + (r.dr.title ? ` · ${r.dr.title}` : '')), h('div', { class: 'm' }, meta.join(' · '), status)),
        h('button', { class: 'btn-x', title: 'Remove from the extraction', onclick: () => removeStorey(r) }, '×')));
    });
  }
  function moveStorey(i, d) { const j = i + d; if (j < 0 || j >= P.order.length) return; [P.order[i], P.order[j]] = [P.order[j], P.order[i]]; saveProject(); storeyView.index = j; fillStoreyList(storeyRows()); }
  async function removeStorey(r) {
    try { const res = await postJSON(`/api/sheets/${r.id}/confirm`, { confirm: { [keyOf(r.pkg, r.dr)]: { extract: false } } }); r.job.confirm = res.confirm; } catch (e) { return alert(e.message); }
    P.order = P.order.filter((k) => k !== r.key); saveProject(); renderStoreys();
  }
  function updateRunState(rows) {
    const jobs = [...new Set(rows.map((r) => r.job))];
    const running = jobs.some((j) => busy(j));
    const allDone = rows.length && rows.every((r) => r.res && r.res.outputs);
    const errs = jobs.filter((j) => j.status === 'error' && j.error);
    const errBox = $('#run-errors'); if (errBox) put(errBox, ...errs.map((j) => errorBox(j)));
    if (running) { runBtn.disabled = true; put(runBtn, h('span', { class: 'spinner', style: 'border-color:rgba(255,255,255,.4);border-top-color:#fff' }), 'Extracting…'); const cur = jobs.find((j) => j.progress); runNote.textContent = cur && cur.progress ? `${cur.name}: ${cur.progress.stage}` : 'waiting for the worker…'; }
    else if (allDone) { runBtn.disabled = false; runBtn.textContent = 'View results ›'; runBtn.onclick = () => go('results'); runNote.textContent = 'Extraction finished. Run it again after changing an area or a scale.'; if (storeyView.autoResults) { storeyView.autoResults = false; go('results'); } }
    else { runBtn.disabled = !rows.length; runBtn.textContent = rows.some((r) => r.res) ? 'Run extraction again' : 'Run extraction'; runBtn.onclick = runExtraction; runNote.textContent = 'About 1 minute per sheet on this machine, mostly OCR.'; }
  }
  async function runExtraction() {
    const jobs = [...new Set(storeyRows().map((r) => r.job))];
    storeyView.autoResults = true;
    for (const j of jobs) {
      if (busy(j)) continue;
      try { await postJSON(`/api/sheets/${j.id}/run`, {}); await refreshJob(j.id); } catch (e) { alert(`${j.name}: ${e.message}`); }
    }
    renderStoreys({ patch: true });
  }
}
function scaleText(r) {
  const ppm = ppmOf(r.job, r.pkg, r.dr); const dpi = r.conf.dpi || r.pkg.summary.dpi;
  if (!ppm) return 'no scale';
  return dpi ? fmt.scaleN(dpi / 0.0254 / ppm).replace(/ /g, '') + (r.conf.scale_source === 'measured' ? ' (measured)' : '') : `${ppm.toFixed(0)} px/m`;
}
function runStatus(r) {
  const j = r.job; const key = keyOf(r.pkg, r.dr);
  if (busy(j)) { const p = j.progress; const mine = p && p.key === key; return h('span', { class: 'badge ' + (mine ? 'badge-run' : 'badge-wait') }, mine ? h('span', { class: 'spinner' }) : h('span', { class: 'dot' }), mine ? p.stage : j.status === 'running' ? 'waiting' : 'queued'); }
  if (r.res && r.res.outputs) return h('span', { class: 'badge badge-ok' }, h('span', { class: 'dot' }), `Finished · ${fmt.secs(r.res.times && r.res.times.total)}`);
  if (r.res && (r.res.error || r.res.skipped)) return h('span', { class: 'badge badge-err', title: r.res.error || r.res.skipped }, h('span', { class: 'dot' }), 'Failed');
  if (j.status === 'error') return h('span', { class: 'badge badge-err' }, h('span', { class: 'dot' }), 'Error');
  return null;
}

// --- Results and export (boards 1g, 1h) ----------------------------------------------------------------------------

const RV = { colour: 'confidence', layers: { walls: true, doors: true, windows: true, passages: true, columns: true, stairs: true, rooms: true, labels: true, graph: false, gf: true, overlay: false }, selected: null, opacity: 1 };

async function renderResults() {
  const rows = storeyRows().filter((r) => r.res && r.res.outputs);
  if (!rows.length) return go('storeys');
  let row = rows.find((r) => r.key === P.resultsKey) || rows[0];
  P.resultsKey = row.key; saveProject();
  const left = h('aside', { class: 'side left' });
  const stage = h('main', { class: 'stage grid-bg' });
  const right = h('aside', { class: 'side right' });
  mount(h('div', { class: 'results' }, left, stage, right), projectName());
  viewer = new Viewer(stage);
  const opacity = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: RV.opacity, style: 'width:110px', oninput: (e) => { RV.opacity = +e.target.value; viewer.imgAlpha = RV.opacity; pct.textContent = `${Math.round(RV.opacity * 100)}%`; viewer.draw(); } });
  const pct = h('span', { style: 'width:34px' }, `${Math.round(RV.opacity * 100)}%`);
  stage.append(h('div', { class: 'stage-bottom' },
    h('div', { class: 'seg' }, h('button', { class: 'on' }, 'Top'), h('button', { disabled: true, title: '3D is not part of this prototype' }, '3D')),
    h('div', { class: 'vsep' }), h('div', { style: 'display:flex;align-items:center;gap:8px' }, h('span', { class: 'muted' }, 'Plan'), opacity, pct),
    h('div', { class: 'vsep' }), h('button', { class: 'btn-link', style: 'color:var(--text)', onclick: () => { viewer.fit(); viewer.draw(); } }, 'Reset view')));
  const hover = h('div', { style: 'position:absolute;pointer-events:none;background:#1a1d21;color:#fff;border-radius:6px;padding:6px 10px;font-size:12px;line-height:1.35;box-shadow:0 4px 14px rgba(0,0,0,.18);white-space:nowrap;z-index:3', hidden: true });
  stage.append(hover);
  let D = null, overlayImg = null;
  try { D = await getJSON(`/api/sheets/${row.id}/results?key=${encodeURIComponent(keyOf(row.pkg, row.dr))}`); }
  catch (e) { right.append(h('div', { class: 'error-box' }, e.message)); return; }
  if (D.error) { right.append(h('div', { class: 'error-box' }, D.error)); return; }
  results = { key: row.key, data: D };
  const ppm = D.px_per_m, ex = D.extent;                       // plan metres -> working px
  const W = (X, Y) => [(X - ex[0]) * ppm, (ex[3] - Y) * ppm];
  const rings = (g) => !g ? [] : g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
  const toPx = (ring) => ring.map(([X, Y]) => W(X, Y));
  const rooms = D.rooms.map((r, i) => ({ ...r, i, rings: rings(r.geometry).map((poly) => poly.map(toPx)), colour: PAL[i % PAL.length] }));
  for (const r of rooms) r.centre = labelPoint(r.rings[0] ? r.rings[0][0] : []);
  const geom = (list, key) => list.map((o) => ({ ...o, rings: rings(key ? o[key] : o).map((poly) => poly.map(toPx)) }));
  const openings = geom(D.openings, 'geometry'), walls = geom(D.walls), stairs = geom(D.stairs), voids = geom(D.voids, 'geometry'), columns = geom(D.columns);
  const gf = D.floor.gf ? rings(D.floor.gf).map((poly) => poly.map(toPx)) : [];
  const review = D.review && D.review.rooms ? D.review.rooms : {};
  const counts = { high: 0, medium: 0, low: 0 }; for (const r of rooms) counts[r.confidence] = (counts[r.confidence] || 0) + 1;
  const roomColour = (r) => RV.colour === 'confidence' ? confColour(r.confidence) : r.colour;
  const nKind = (k) => openings.filter((o) => o.kind === k || (k === 'door' && o.kind === 'exterior door') || (k === 'passage' && o.kind === 'interior opening')).length;

  // --- left: storeys, colour, layers
  function drawLeft() {
    put(left, 
      h('div', { class: 'storey-pick' }, h('div', { class: 'section-title', style: 'margin-bottom:4px' }, 'Storeys'),
        rows.map((r, i) => h('button', { class: r.key === row.key ? 'on' : '', onclick: () => { P.resultsKey = r.key; saveProject(); renderResults(); } }, h('span', null, r.label || r.dr.title || `Storey ${i + 1}`), h('span', { class: 'z' }, scaleText(r))))),
      h('div', { style: 'display:flex;flex-direction:column;gap:6px' }, h('div', { class: 'section-title' }, 'Colour rooms by'),
        h('div', { class: 'seg' }, h('button', { class: RV.colour === 'room' ? 'on' : '', onclick: () => { RV.colour = 'room'; drawLeft(); viewer.draw(); } }, 'Room'), h('button', { class: RV.colour === 'confidence' ? 'on' : '', onclick: () => { RV.colour = 'confidence'; drawLeft(); viewer.draw(); } }, 'QA confidence')),
        RV.colour === 'confidence' ? h('div', { class: 'legend-rows' },
          h('div', null, h('span', { class: 'swatch', style: 'background:#2e9e5b' }), h('span', { title: 'Stamp area within 5 %', style: 'cursor:help' }, 'High'), h('span', { class: 'n' }, counts.high)),
          h('div', null, h('span', { class: 'swatch', style: 'background:#d99a1e' }), h('span', { title: 'No stamp to check against', style: 'cursor:help' }, 'Medium'), h('span', { class: 'n' }, counts.medium)),
          h('div', null, h('span', { class: 'swatch', style: 'background:#d0433b' }), h('span', { title: 'Area off by more than 15 %, several stamps, or tiny', style: 'cursor:help' }, 'Low'), h('span', { class: 'n' }, counts.low))) : null),
      h('div', { class: 'layers' }, h('div', { class: 'section-title', style: 'margin-bottom:4px' }, 'Layers'),
        layerRow('walls', 'Walls', LAYER_COLOURS.walls, walls.length), layerRow('doors', 'Doors', LAYER_COLOURS.doors, nKind('door')),
        layerRow('windows', 'Windows', LAYER_COLOURS.windows, nKind('window')), layerRow('passages', 'Passages', LAYER_COLOURS.passages, nKind('passage')),
        layerRow('columns', 'Columns', LAYER_COLOURS.columns, columns.length), layerRow('stairs', 'Stairs and voids', LAYER_COLOURS.stairs, stairs.length + voids.length),
        layerRow('rooms', 'Rooms', null, rooms.length, RV.colour === 'confidence' ? 'sw-conf' : 'sw-rooms'), layerRow('labels', 'Room labels', null, null, 'sw-outline'),
        layerRow('graph', 'Room connections', LAYER_COLOURS.graph, D.connectivity.length), layerRow('gf', 'GF outline', null, gf.length, 'sw-dashed'),
        layerRow('overlay', 'Segmentation (pipeline overlay)', '#8a8f99', null)),
      h('div', { class: 'muted small', style: 'margin-top:auto;line-height:1.5' }, 'Plan data stays local (', h('code', null, 'app/data/'), ', gitignored).'));
  }
  function layerRow(key, name, colour, n, cls) {
    return h('label', null, h('input', { type: 'checkbox', checked: RV.layers[key], onchange: (e) => { RV.layers[key] = e.target.checked; if (key === 'overlay' && e.target.checked && !overlayImg) { overlayImg = new Image(); overlayImg.onload = () => viewer.draw(); overlayImg.src = `/api/sheets/${row.id}/file/${D.files.overlay}`; } viewer.draw(); } }),
      h('span', { class: 'swatch ' + (cls || ''), style: colour ? `background:${colour}` : '' }), name, n != null ? h('span', { class: 'n' }, n) : null);
  }

  // --- canvas
  viewer.overlay = (c, v) => {
    if (RV.layers.overlay && overlayImg && overlayImg.complete) { const [x, y] = v.toScreen(0, 0); c.globalAlpha = 0.85; c.drawImage(overlayImg, x, y, overlayImg.width * v.view.s, overlayImg.height * v.view.s); c.globalAlpha = 1; }
    if (RV.layers.walls) for (const w of walls) for (const poly of w.rings) drawPolyWithHoles(c, v, poly, 'rgba(43,58,85,.85)', null);
    if (RV.layers.columns) for (const o of columns) for (const poly of o.rings) drawPolyWithHoles(c, v, poly, LAYER_COLOURS.columns, null);
    if (RV.layers.stairs) { for (const o of stairs) for (const poly of o.rings) drawPolyWithHoles(c, v, poly, 'rgba(194,63,191,.35)', LAYER_COLOURS.stairs); for (const o of voids) for (const poly of o.rings) drawPolyWithHoles(c, v, poly, 'rgba(31,120,180,.3)', LAYER_COLOURS.voids, [5, 4]); }
    if (RV.layers.rooms) for (const r of rooms) { const col = roomColour(r); for (const poly of r.rings) drawPolyWithHoles(c, v, poly, hexA(col, .38), col, null, 1.5); }
    for (const o of openings) {
      const on = (o.kind === 'window' && RV.layers.windows) || ((o.kind === 'door' || o.kind === 'exterior door') && RV.layers.doors) || ((o.kind === 'passage' || o.kind === 'interior opening') && RV.layers.passages);
      if (!on) continue;
      const col = o.kind === 'window' ? LAYER_COLOURS.windows : o.kind.includes('door') ? LAYER_COLOURS.doors : LAYER_COLOURS.passages;
      for (const poly of o.rings) drawPolyWithHoles(c, v, poly, hexA(col, .75), col);
    }
    if (RV.layers.gf) for (const poly of gf) drawPolyWithHoles(c, v, poly, null, LAYER_COLOURS.gf, [8, 5], 2);
    if (RV.layers.graph) {
      const byId = Object.fromEntries(rooms.map((r) => [r.id, r])); const op = Object.fromEntries(openings.map((o) => [o.id, o]));
      for (const e of D.connectivity) { const a = byId[e.a], b = byId[e.b], o = op[e.opening]; if (!a || !b) continue; const mid = o && o.rings[0] ? labelPoint(o.rings[0][0]) : [(a.centre[0] + b.centre[0]) / 2, (a.centre[1] + b.centre[1]) / 2]; v.poly([a.centre, mid, b.centre], { stroke: LAYER_COLOURS.graph, width: 2, close: false }); for (const p of [a.centre, b.centre]) { const [sx, sy] = v.toScreen(p[0], p[1]); c.fillStyle = LAYER_COLOURS.graph; c.beginPath(); c.arc(sx, sy, 3.5, 0, Math.PI * 2); c.fill(); } }
    }
    if (RV.selected) { const r = rooms.find((x) => x.id === RV.selected); if (r) for (const poly of r.rings) { v.poly(poly[0], { stroke: 'rgba(255,255,255,.95)', width: 7 }); v.poly(poly[0], { stroke: ACCENT, width: 2.5 }); } }
    if (RV.layers.labels && v.view.s * ppm > 12) for (const r of rooms) { if (!r.centre) continue; const name = r.name || (r.small_region ? 'small region' : ''); const [sx, sy] = v.toScreen(r.centre[0], r.centre[1]); c.textAlign = 'center'; c.textBaseline = 'middle'; c.lineWidth = 3; c.strokeStyle = 'rgba(255,255,255,.9)'; if (name) { c.font = `600 11px 'IBM Plex Sans', system-ui, sans-serif`; c.strokeText(name, sx, sy - 6); c.fillStyle = '#1a1d21'; c.fillText(name, sx, sy - 6); } c.font = `11px 'IBM Plex Sans', system-ui, sans-serif`; const t2 = `${fmt.m2(r.area_net)}${r.area_stamp != null ? ' · st. ' + r.area_stamp : ''}`; c.strokeText(t2, sx, sy + (name ? 7 : 0)); c.fillStyle = '#646b74'; c.fillText(t2, sx, sy + (name ? 7 : 0)); }
  };
  const roomAt = (ip) => { for (let i = rooms.length - 1; i >= 0; i--) { const r = rooms[i]; for (const poly of r.rings) if (inPoly(ip, poly[0]) && !poly.slice(1).some((hole) => inPoly(ip, hole))) return r; } return null; };
  viewer.onClick = (ip) => { const r = roomAt(ip); select(r ? r.id : null); };
  viewer.tool = { hover: (ip, sp, v) => { const r = roomAt(ip); if (!r) { hover.hidden = true; v.canvas.style.cursor = 'grab'; return; } v.canvas.style.cursor = 'pointer'; hover.hidden = false; hover.innerHTML = `<b>${esc(r.name || '(no stamp)')}</b> · ${fmt.m2(r.area_net)} · ${esc(r.confidence)}<br><span style="opacity:.7">Click to pin</span>`; hover.style.left = `${sp[0] + 14}px`; hover.style.top = `${sp[1] + 14}px`; } };
  viewer.imgAlpha = RV.opacity;
  viewer.load(`/api/sheets/${row.id}/file/${D.files.work}`).then(() => { if (gf.length) fitTo(gf[0][0]); }).catch((e) => right.prepend(h('div', { class: 'error-box' }, e.message)));
  function fitTo(ring) { const b = bboxOf(ring); const pad = 60; const s = Math.min((viewer.w - 2 * pad) / (b.x1 - b.x0), (viewer.h - 2 * pad) / (b.y1 - b.y0)); viewer.view = { s, x: (viewer.w - (b.x0 + b.x1) * s) / 2, y: (viewer.h - (b.y0 + b.y1) * s) / 2 }; viewer.fitted = true; viewer.draw(); }

  // --- right: extraction, needs review, rooms, export
  const inspector = h('div');
  const roomList = h('div', { class: 'room-table' });
  function select(id) { RV.selected = id; drawInspector(); drawRoomList(); viewer.draw(); if (id) { const r = rooms.find((x) => x.id === id); const el = roomList.querySelector(`[data-id="${id}"]`); if (el) el.scrollIntoView({ block: 'nearest' }); if (r && r.centre) { const [sx, sy] = viewer.toScreen(r.centre[0], r.centre[1]); if (sx < 0 || sy < 0 || sx > viewer.w || sy > viewer.h) { viewer.view.x += viewer.w / 2 - sx; viewer.view.y += viewer.h / 2 - sy; viewer.draw(); } } } }
  async function setChecked(r, checked) {
    review[r.id] = Object.assign(review[r.id] || {}, { checked });
    try { await postJSON(`/api/sheets/${row.id}/review`, { key: keyOf(row.pkg, row.dr), rooms: { [r.id]: { checked } } }); } catch (e) { alert(e.message); }
    drawStats(); drawRoomList(); drawInspector();
  }
  function drawInspector() {
    const r = rooms.find((x) => x.id === RV.selected);
    if (!r) { inspector.replaceChildren(); return; }
    const dev = r.area_deviation_pct;
    put(inspector, h('div', { class: 'inspector' },
      h('div', { class: 'head' }, h('b', null, r.name || '(no stamp)'), h('button', { class: 'btn-x', onclick: () => select(null) }, '×')),
      h('div', { class: 'kv-grid', style: 'font-size:12px' },
        h('span', null, 'Net area'), h('span', null, fmt.m2(r.area_net)), h('span', null, 'Gross area'), h('span', null, fmt.m2(r.area_gross)),
        h('span', null, 'Stamp'), h('span', null, r.area_stamp != null ? [fmt.m2(r.area_stamp), ' ', h('span', { style: `color:${Math.abs(dev || 0) > 5 ? '#d0433b' : '#646b74'}` }, dev != null ? `(${fmt.pct(dev)})` : '')] : 'none read'),
        r.number ? h('span', null, 'Number') : null, r.number ? h('span', null, r.number) : null,
        h('span', null, 'Usage'), h('span', null, r.usage || '–'), h('span', null, 'Confidence'), h('span', { class: 'conf-' + r.confidence, style: 'font-weight:600' }, r.confidence),
        h('span', null, 'Neighbours'), h('span', null, (r.neighbours || []).length ? r.neighbours.map((n) => (rooms.find((x) => x.id === n) || {}).name || n).join(', ') : '–')),
      (r.reasons || []).length ? h('ul', null, r.reasons.map((x) => h('li', null, x))) : null,
      r.review ? h('div', { class: 'warn-box', style: 'padding:8px 10px;font-size:12px' }, `${r.review.guess || 'small region'}: ${r.review.reason || ''}`) : null,
      h('label', { style: 'display:flex;gap:8px;align-items:center;font-weight:600' }, h('input', { type: 'checkbox', checked: !!(review[r.id] && review[r.id].checked), onchange: (e) => setChecked(r, e.target.checked) }), 'Checked against the sheet')));
  }
  function drawRoomList() {
    const sorted = [...rooms].sort((a, b) => ({ low: 0, medium: 1, high: 2 }[a.confidence] ?? 3) - ({ low: 0, medium: 1, high: 2 }[b.confidence] ?? 3) || (b.area_net || 0) - (a.area_net || 0));
    put(roomList, h('div', { class: 'room-row head' }, h('span', null, '✓'), h('span'), h('span', null, `Rooms · ${rooms.length}`), h('span', { class: 'ar' }, 'Net'), h('span', { class: 'cf' }, 'Conf.')),
      sorted.map((r) => h('div', { class: 'room-row' + (r.id === RV.selected ? ' on' : ''), 'data-id': r.id, onclick: () => select(r.id) },
        h('input', { type: 'checkbox', checked: !!(review[r.id] && review[r.id].checked), onclick: (e) => e.stopPropagation(), onchange: (e) => setChecked(r, e.target.checked) }),
        h('span', { class: 'swatch', style: `background:${roomColour(r)}` }), h('span', { class: 'nm', style: r.id === RV.selected ? 'font-weight:600' : '' }, r.name || (r.small_region ? 'small region' : '(no stamp)')),
        h('span', { class: 'ar' }, fmt.n(r.area_net)), h('span', { class: 'cf conf-' + r.confidence }, r.confidence))));
  }
  const stats = h('div');
  function drawStats() {
    const total = rooms.reduce((a, r) => a + (r.area_net || 0), 0);
    const high = rooms.filter((r) => r.confidence === 'high').reduce((a, r) => a + (r.area_net || 0), 0);
    const share = total ? Math.round(100 * high / total) : 0;
    const checked = rooms.filter((r) => review[r.id] && review[r.id].checked).length;
    const t = D.times || {}; const total_s = t.total != null ? t.total : Object.values(t).reduce((a, b) => a + (+b || 0), 0);
    put(stats, h('div', { class: 'stat-head' }, h('h2', null, 'Extraction'), h('span', { class: 'badge badge-ok' }, h('span', { class: 'dot' }), 'Finished')),
      h('div', { class: 'kv-grid', style: 'margin-top:10px' },
        h('span', null, 'Confidence'), h('span', null, h('span', { class: 'bar' }, h('i', { style: `width:${share}%` })), ' ', h('span', { title: 'Share of room area with high confidence', style: 'cursor:help' }, `${share} %`)),
        h('span', null, 'Rooms'), h('span', null, `${rooms.length}${checked ? ` · ${checked} checked` : ''}`),
        h('span', null, 'Net room area'), h('span', null, fmt.m2(D.floor.sum_net)), h('span', null, 'Gross floor area'), h('span', null, fmt.m2(D.floor.gf_area)),
        h('span', null, 'Openings'), h('span', null, openings.length), h('span', null, 'Scale'), h('span', { class: 'small' }, (D.scale && D.scale.confirmed) ? `${D.scale.confirmed.px_per_m.toFixed(1)} px/m · ${D.scale.confirmed.source}` : (D.scale && D.scale.value) || '–'),
        h('span', null, 'Drawing style'), h('span', null, (D.triage && D.triage.graphical_style) || '–'), h('span', null, 'Processing time'), h('span', null, fmt.secs(total_s))));
  }
  const needs = [];
  for (const r of rooms) if (r.confidence === 'low' || r.review) needs.push({ col: r.confidence === 'low' ? '#d0433b' : '#d99a1e', title: r.name || (r.small_region ? 'small region' : '(no stamp)'), id: r.id, text: r.review ? `${r.review.guess || 'small region'}: ${r.review.reason || ''}` : (r.reasons || [])[0] || 'low confidence', room: r.id });
  for (const v of voids) if (v.gf_deducted) needs.push({ col: '#1f78b4', title: v.label || 'void', id: 'void', text: `${fmt.m2(v.area)} deducted from GF (above the 5 m² rule)` });
  for (const q of D.qa) if (q.severity === 'high' && !/^r\d/.test(q.element || '')) needs.push({ col: '#d0433b', title: q.check, id: q.element, text: q.message });
  const qaList = h('details', null, h('summary', null, `All QA issues · ${D.qa.length}`), h('div', { class: 'qa-list', style: 'margin-top:6px' }, D.qa.map((q) => h('div', { class: 'qa-row' }, h('span', { class: 'sev sev-' + q.severity }, q.severity), h('span', null, h('b', null, `${q.check} ${q.element}`), ' ', q.message)))));
  const dl = (name, label) => h('a', { class: 'btn', href: `/api/sheets/${row.id}/file/${name}?download=1`, download: name }, label || 'Download');
  right.append(stats,
    h('div', { class: 'section' }, h('div', { class: 'head' }, h('h2', null, 'Needs review'), h('span', { class: 'muted small' }, `${needs.filter((n) => n.room).length} rooms, ${needs.filter((n) => !n.room).length} other`)),
      needs.length ? h('div', { class: 'review-list' }, needs.slice(0, 12).map((n) => h('div', { class: 'review-row', onclick: () => n.room && select(n.room) }, h('span', { class: 'dot10', style: `background:${n.col}` }), h('div', null, h('div', { class: 't' }, n.title, ' ', h('span', null, n.id)), h('div', { class: 'd' }, n.text)), h('span', { class: 'muted' }, n.room ? '›' : '')))) : h('div', { class: 'muted small' }, 'Nothing flagged. Check the rooms against the sheet all the same.'),
      qaList, inspector, roomList),
    h('div', { class: 'section' }, h('h2', null, 'Export'),
      h('div', { class: 'export-row' }, h('span', { class: 'f' }, 'JSON'), h('span', { class: 'd' }, 'Pipeline output (plan metres)'), dl(D.files.json)),
      h('div', { class: 'export-row' }, h('span', { class: 'f' }, 'DXF'), h('span', { class: 'd' }, 'CAD-Richtlinie BBL layers'), dl(D.files.dxf)),
      h('div', { class: 'export-row' }, h('span', { class: 'f' }, 'PNG'), h('span', { class: 'd' }, 'Plan with overlay'), h('a', { class: 'btn', href: `/api/sheets/${row.id}/file/${D.files.overlay}`, download: D.files.overlay }, 'Download')),
      h('div', { class: 'export-row' }, h('span', { class: 'f' }, 'CSV'), h('span', { class: 'd' }, 'Room list with review state'), h('a', { class: 'btn', href: `/api/sheets/${row.id}/rooms.csv?key=${encodeURIComponent(keyOf(row.pkg, row.dr))}` }, 'Download')),
      h('div', { class: 'export-row' }, h('span', { class: 'f off' }, 'DWG'), h('span', { class: 'd' }, 'Not yet available'), h('button', { class: 'btn', disabled: true }, 'Soon')),
      h('div', { class: 'export-row' }, h('span', { class: 'f' }, 'XLSX'), h('span', { class: 'd' }, 'Room list, floor figures, openings, QA (Excel)'), dl(D.files.json.replace(/\.json$/, '.xlsx'))),
      h('div', { class: 'export-row' }, h('span', { class: 'f' }, 'IFC'), h('span', { class: 'd' }, 'IFC 4.3: spaces with net/gross quantities, walls, doors, windows, stairs, slab (nominal heights)'), dl(D.files.json.replace(/\.json$/, '.ifc'))),
      h('div', { class: 'export-row' }, h('span', { class: 'f off' }, 'PDF'), h('span', { class: 'd' }, 'Not yet available'), h('button', { class: 'btn', disabled: true }, 'Soon'))),
    h('div', { class: 'section', style: 'border-top:0;padding-top:0' }, h('button', { class: 'btn', onclick: () => go('storeys') }, '‹ Storeys and run')));
  drawLeft(); drawStats(); drawRoomList(); drawInspector();
}
function drawPolyWithHoles(c, v, poly, fill, stroke, dash, width = 1.5) {
  c.beginPath();
  for (const ring of poly) { ring.forEach(([x, y], i) => { const [sx, sy] = v.toScreen(x, y); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); }); c.closePath(); }
  if (fill) { c.fillStyle = fill; c.fill('evenodd'); }
  if (stroke) { c.strokeStyle = stroke; c.lineWidth = width; c.setLineDash(dash || []); c.stroke(); c.setLineDash([]); }
}
function hexA(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
function inPoly(p, ring) { let inside = false; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) { const [xi, yi] = ring[i], [xj, yj] = ring[j]; if ((yi > p[1]) !== (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) inside = !inside; } return inside; }
function labelPoint(ring) {
  if (!ring || ring.length < 3) return null;
  let A = 0, cx = 0, cy = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) { const f = ring[j][0] * ring[i][1] - ring[i][0] * ring[j][1]; A += f; cx += (ring[j][0] + ring[i][0]) * f; cy += (ring[j][1] + ring[i][1]) * f; }
  if (Math.abs(A) < 1e-9) return ring[0];
  const c = [cx / (3 * A), cy / (3 * A)];
  if (inPoly(c, ring)) return c;
  const b = bboxOf(ring); const mid = [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];   // L-shapes: scan the middle row for the widest inside run
  let best = null, run = null;
  for (let x = b.x0; x <= b.x1; x += (b.x1 - b.x0) / 60) { const ins = inPoly([x, mid[1]], ring); if (ins && !run) run = [x, x]; else if (ins) run[1] = x; else if (run) { if (!best || run[1] - run[0] > best[1] - best[0]) best = run; run = null; } }
  if (run && (!best || run[1] - run[0] > best[1] - best[0])) best = run;
  return best ? [(best[0] + best[1]) / 2, mid[1]] : mid;
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// ---------------------------------------------------------------------------------------------------------------------
// Boot

async function boot() {
  if (location.protocol === 'file:') { $('#offline').hidden = false; return; }
  $('#app').hidden = false;
  $('#btn-help').onclick = () => { $('#help').hidden = false; };
  $('#help-close').onclick = () => { $('#help').hidden = true; };
  $('#help').onclick = (e) => { if (e.target === $('#help')) $('#help').hidden = true; };
  $('#btn-start-over').onclick = () => { if (!confirm('Start a new project? The processed sheets stay in app/data until you remove them from the upload list.')) return; P = { name: '', sheets: [], order: [], screen: 'upload', cursor: { sheet: 0, pkg: 0, drawing: 0 }, resultsKey: null }; saveProject(); go('upload'); };
  // deep link: ?sheets=<id>,<id>&screen=area|scale|storeys|results&pkg=0&drawing=0&sheet=0 (also used for screenshots)
  const q = new URLSearchParams(location.search);
  if (q.get('sheets')) {
    P.sheets = q.get('sheets').split(',').filter(Boolean); P.order = []; P.resultsKey = q.get('key') || null;
    P.screen = q.get('screen') || 'upload'; P.cursor = { sheet: +(q.get('sheet') || 0), pkg: +(q.get('pkg') || 0), drawing: +(q.get('drawing') || 0) };
    saveProject(); history.replaceState(null, '', location.pathname);
  }
  try { await refreshAll(); } catch (e) { /* the server is checked per request */ }
  startPolling();
  const first = P.sheets.length ? P.screen : 'upload';
  if (first === 'results' && !storeyRows().some((r) => r.res && r.res.outputs)) return go('storeys');
  if ((first === 'area' || first === 'scale') && !(J[P.sheets[P.cursor.sheet]] || {}).analysis) return go('upload');
  go(first);
}
document.addEventListener('DOMContentLoaded', boot);
