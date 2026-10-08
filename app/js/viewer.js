/* Canvas viewer with pan and zoom, the building-area and measure tools, polygon helpers. */
import { ACCENT, h } from './core.js';
import { bboxOf } from './state.js';

// ---------------------------------------------------------------------------------------------------------------------
// Canvas viewer: an image with pan and zoom, overlays drawn in screen space, an optional tool

class Viewer {
  constructor(stage) {
    this.stage = stage;
    this.canvas = h('canvas');
    stage.append(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.img = null; this.imgAlpha = 1; this.view = { s: 1, x: 0, y: 0 }; this.fitted = false; this.fitPad = 48;
    this.home = null;                      // { x0, y0, x1, y1, pad, maxZoom }: what fit() frames instead of the whole image
    this.canvas.setAttribute('role', 'img');
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
  setLabel(text) { this.canvas.setAttribute('aria-label', text || ''); }
  resize() {
    const r = this.stage.getBoundingClientRect();
    this.w = r.width; this.h = r.height; this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.w * this.dpr)); this.canvas.height = Math.max(1, Math.round(this.h * this.dpr));
    if (!this.fitted) this.fit();
    this.draw();
  }
  /** Load an image; home (optional) is the part of it the view frames at first and on Fit: {x0, y0, x1, y1, pad, maxZoom}. */
  load(url, home = null) {
    this.home = home;
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => { this.img = im; this.fitted = false; this.fit(); this.draw(); resolve(im); };
      im.onerror = () => reject(new Error(`image ${url} could not be loaded`));
      im.src = url;
    });
  }
  /** Frame the home box, else the whole image. Does nothing until the stage has a size: resize() calls it then. */
  fit() {
    if (!this.img || !this.w || !this.h) return;
    const whole = Math.min((this.w - 2 * this.fitPad) / this.img.width, (this.h - 2 * this.fitPad) / this.img.height);
    const b = this.home;
    if (b && b.x1 > b.x0 && b.y1 > b.y0) {
      const pad = b.pad == null ? this.fitPad : b.pad;
      let s = Math.min((this.w - 2 * pad) / (b.x1 - b.x0), (this.h - 2 * pad) / (b.y1 - b.y0));
      if (b.maxZoom) s = Math.min(s, whole * b.maxZoom);
      this.view = { s, x: (this.w - (b.x0 + b.x1) * s) / 2, y: (this.h - (b.y0 + b.y1) * s) / 2 };
    } else {
      this.view = { s: whole, x: (this.w - this.img.width * whole) / 2, y: (this.h - this.img.height * whole) / 2 };
    }
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

/** Points of a polygon scaled by f (sheet px -> preview px and back). */
const scaled = (pts, f) => pts.map(([x, y]) => [x * f, y * f]);

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

export { MeasureTool, RectTool, Viewer, drawPolyWithHoles, hexA, inPoly, labelPoint, scaled };
