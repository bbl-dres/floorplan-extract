/* Screen mounting and routing, the current viewer, shared panel pieces (error box, project name). */
import { $, api, h, postJSON, put } from './core.js';
import { J, P, busy, drawingRows, resetProject, saveProject, screenName, setScreen, startPolling } from './state.js';
import { Viewer } from './viewer.js';

// ---------------------------------------------------------------------------------------------------------------------
// Screens

let current = null;                      // the canvas viewer of the screen on show; the next mount destroys it
function useViewer(stage, label) { if (current) current.destroy(); current = new Viewer(stage); current.setLabel(label || 'The sheet'); return current; }
const SCREENS = {};                      // name -> render function, filled by app.js (no module cycle with the screens)
function register(name, fn) { SCREENS[name] = fn; }
function screenEl() { return $('#screen'); }
const onUnmount = [];                    // what a screen wants torn down when the next one mounts (e.g. the 3D view)
function beforeUnmount(fn) { onUnmount.push(fn); }
function mount(node, crumb) {
  if (current) { current.destroy(); current = null; }
  while (onUnmount.length) { try { onUnmount.pop()(); } catch (e) { /* already gone */ } }
  const s = screenEl(); put(s, node);
  const c = $('#crumb'); c.hidden = !crumb; $('#crumb-text').textContent = crumb || '';
  $('#btn-start-over').hidden = !P.sheets.length;
  setDownloads(null);
  document.onkeydown = null;
  focusHeading();
}
/** The screen's first heading takes focus: a screen reader announces the new screen, keyboard users start at its top. */
function focusHeading() {
  const head = screenEl().querySelector('h2');
  if (head) { head.tabIndex = -1; head.focus({ preventScroll: true }); }
}
function go(name, cursor) {
  if (cursor) P.cursor = Object.assign({ sheet: 0, pkg: 0, drawing: 0 }, cursor);
  P.screen = name; setScreen(name); saveProject();
  (SCREENS[name] || SCREENS.upload)();
}
/** A notice at the bottom of the window (kind err | ok | info), announced to assistive technology; closes by itself. */
function notify(text, kind = 'err', ms = 8000) {
  let box = $('#toasts');
  if (!box) { box = h('div', { id: 'toasts', class: 'toasts', role: 'status', 'aria-live': 'polite' }); document.body.append(box); }
  const t = h('div', { class: `toast toast-${kind}` }, h('span', null, text), h('button', { class: 'btn-x', 'aria-label': 'Dismiss', onclick: () => t.remove() }, '×'));
  box.append(t);
  setTimeout(() => t.remove(), ms);
  return t;
}
/** POST a JSON body; on failure the message is shown as a notice and null comes back (the callers carry on). */
async function post(path, body) {
  try { return await postJSON(path, body); } catch (e) { notify(e.message); return null; }
}
/** The screen shown while sheets are analysed: the spinner with each sheet's last log line, the way out. Used by the
    upload page (with the abort) and by a step opened before its sheet is analysed (with the way back). */
function busyScreen(jobs, { title = 'New project', text, abort, back } = {}) {
  const line = (job) => job.log && job.log.length ? job.log[job.log.length - 1].slice(9) : 'starting…';
  const main = h('main', { class: 'upload grid-bg' }, h('div', { class: 'upload-inner' },
    h('div', { class: 'stack-sm' }, h('h2', null, title), h('p', { class: 'muted pretty' }, text || 'Reading the sheet: the drawings, title block and notes are found first, without text recognition. A few seconds.')),
    h('div', { class: 'card busy-card', role: 'status', 'aria-live': 'polite' },
      jobs.map((job) => job.status === 'error' ? errorBox(job) : h('div', { class: 'row row-md' }, h('span', { class: 'spinner' }), h('div', { class: 'clip' }, h('div', { class: 'name' }, job.name), h('div', { class: 'sub muted small' }, line(job))))),
      h('div', { class: 'row' }, abort ? h('button', { class: 'btn btn-lg', onclick: abort }, 'Abort and start again') : null,
        back ? h('button', { class: 'btn btn-lg', onclick: back }, '‹ Back to the upload') : null))));
  mount(main, null);
  startPolling();
}
/** Abort: the sheets are removed (a busy one is discarded by the server when its worker is done) and the project starts empty. */
async function abortProject() {
  const ids = [...P.sheets];
  resetProject();
  for (const id of ids) { try { await api(`/api/sheets/${id}`, { method: 'DELETE' }); } catch (e) { /* already gone */ } }
  go('upload');
}
/** The Download menu of the header: items [{label, note, href, name}] or null to hide it. */
function setDownloads(items) {
  const wrap = $('#download-menu'); if (!wrap) return;
  const list = wrap.querySelector('.menu-list'), btn = wrap.querySelector('button');
  wrap.hidden = !items || !items.length;
  list.hidden = true; btn.setAttribute('aria-expanded', 'false');
  put(list, (items || []).map((it) => h('a', { role: 'menuitem', href: it.href, download: it.name }, h('b', null, it.label), it.note ? h('span', { class: 'muted small' }, it.note) : null)));
}
/** Wire the header once: the Download menu opens on click, closes on Escape, on a click outside and after a choice. */
function initChrome() {
  const wrap = $('#download-menu'); if (!wrap) return;
  const list = wrap.querySelector('.menu-list'), btn = wrap.querySelector('button');
  const close = () => { list.hidden = true; btn.setAttribute('aria-expanded', 'false'); };
  btn.addEventListener('click', () => { list.hidden = !list.hidden; btn.setAttribute('aria-expanded', String(!list.hidden)); if (!list.hidden) { const a = list.querySelector('a'); if (a) a.focus(); } });
  list.addEventListener('click', (e) => { if (e.target.closest('a')) close(); });
  document.addEventListener('click', (e) => { if (!wrap.contains(e.target)) close(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !list.hidden) { close(); btn.focus(); } });
}
// a poll saw jobs change: refresh the list screens in place; a step screen is re-rendered only when its own sheet
// changed (another sheet's progress must not reset handles or measurement points)
function onJobsChanged(changed) {
  if (screenName === 'upload') SCREENS.upload({ keepInput: true });
  else if (screenName === 'results') SCREENS.results({ changed });
  else if (screenName === 'area' || screenName === 'scale') {
    const id = P.sheets[P.cursor.sheet], job = J[id];
    if (changed.includes(id) && job && !busy(job)) go(screenName);
  }
}
function projectName() {
  if (P.name) return P.name;
  const rows = drawingRows();
  if (rows.length) { const j = rows[0].job; return (j.name || '').replace(/\.[^.]+$/, ''); }
  const id = P.sheets[0]; return id && J[id] ? (J[id].name || '').replace(/\.[^.]+$/, '') : '';
}
/** A status badge: kind ok | warn | err | run | wait ('run' shows a spinner instead of the dot). */
function badge(kind, text, title) {
  return h('span', { class: `badge badge-${kind}`, title }, kind === 'run' ? h('span', { class: 'spinner' }) : h('span', { class: 'dot' }), text);
}
/** The head of a step panel: the two step bars (steps up to `step` on; 3 = both, the results), the eyebrow and the title. */
function stepHead(step, eyebrow, title) {
  return [h('div', { class: 'progress-bars', role: 'img', 'aria-label': step >= 3 ? 'Both steps done' : `Step ${step} of 2` }, [1, 2].map((k) => h('span', { class: k <= step ? 'on' : '' }))),
    h('div', null, h('div', { class: 'eyebrow' }, eyebrow), h('h2', null, title))];
}
const previewUrl = (job, pkg) => `/api/sheets/${job.id}/file/${pkg.preview.file}`;
/** The sheet preview into a viewer; when it fails, an error box at the top of the panel (when one is given). */
function loadPreview(viewer, job, pkg, panel) {
  return viewer.load(previewUrl(job, pkg)).catch((e) => { if (panel) panel.prepend(errorBox(job, e.message)); });
}
function errorBox(job, text) {
  const pre = h('pre', { hidden: true }, (job.log || []).slice(-25).join('\n'));
  return h('div', { class: 'error-box' },
    h('div', null, h('b', null, 'Error: '), text || job.error || 'unknown error'),
    h('div', null, h('button', { class: 'btn-link', onclick: () => { pre.hidden = !pre.hidden; } }, 'Show log'),
      ' · ', h('a', { href: `/api/sheets/${job.id}/log`, target: '_blank' }, 'full log')), pre);
}

export { SCREENS, abortProject, badge, beforeUnmount, busyScreen, errorBox, focusHeading, go, initChrome, loadPreview, mount, notify, onJobsChanged, post, previewUrl, projectName, register, screenEl, setDownloads, stepHead, useViewer };
