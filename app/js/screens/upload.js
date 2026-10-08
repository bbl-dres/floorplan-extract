/* Upload screen (board 1c): drop zone, sheet list, pipeline check. */
import { api, getJSON, h, put } from '../core.js';
import { J, P, busy, drawingRows, firstUnconfirmed, packagesOf, refreshJob, saveProject, screenName, startPolling } from '../state.js';
import { abortProject, badge, busyScreen, go, mount, notify, projectName } from '../ui.js';

// --- Upload (board 1c) ---------------------------------------------------------------------------------------------

let health = null, demo;                 // demo: undefined until asked, then the server's demo record or null
async function renderUpload({ keepInput } = {}) {
  const adding = drawingRows().length > 0;
  if (demo === undefined) { try { demo = (await getJSON('/api/demo')).demo; } catch (e) { demo = null; } if (screenName !== 'upload') return; }
  // sheets being analysed: no drop zone, a spinner and the way out
  const pending = P.sheets.filter((id) => J[id] && busy(J[id]));
  if (!adding && pending.length && !P.sheets.some((id) => J[id] && !busy(J[id]))) { autoAdvanced.clear(); return busyScreen(pending.map((id) => J[id]), { abort: abortProject }); }
  const input = h('input', { type: 'file', multiple: true, hidden: true, accept: '.pdf,.png,.jpg,.jpeg,.tif,.tiff,.dxf,.dwg', onchange: (e) => uploadFiles([...e.target.files]) });
  const drop = h('div', { class: 'dropzone', onclick: () => input.click(),
    ondragover: (e) => { e.preventDefault(); drop.classList.add('over'); }, ondragleave: () => drop.classList.remove('over'),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove('over'); uploadFiles([...e.dataTransfer.files]); } },
    h('div', { class: 'arrow' }, '↑'), h('div', { class: 'title' }, 'Drop floor plans here'),
    h('div', { class: 'muted small' }, 'PDF, JPG, PNG, TIFF, DXF · up to 300 MB each · several sheets at once'),
    h('button', { class: 'btn btn-primary btn-lg', onclick: (e) => { e.stopPropagation(); input.click(); } }, 'Choose files'),
    demo && !adding ? h('button', { class: 'btn-link small', title: `${demo.source || ''}${demo.licence ? ' · ' + demo.licence : ''}`, onclick: (e) => { e.stopPropagation(); startDemo(); } }, `or try a demo floor plan: ${demo.title}`) : null);
  const list = h('div', { class: 'file-list' });
  for (const id of P.sheets) {
    const job = J[id]; if (!job) continue;
    const rows = packagesOf(job);
    const n = rows.reduce((a, p) => a + p.drawings.length, 0);
    const st = job.status;
    const sub = st === 'analysing' || st === 'scaling' ? (job.log && job.log.length ? job.log[job.log.length - 1].slice(9) : `${st}…`)
      : st === 'error' ? job.error : st === 'ready' ? `analysed · ${rows.length} page${rows.length === 1 ? '' : 's'}, ${n} drawing${n === 1 ? '' : 's'}`
      : st === 'done' ? 'extracted' : st;
    const state = st === 'analysing' ? badge('run', 'Analysing') : st === 'scaling' ? badge('run', 'Reading scale') : st === 'error' ? badge('err', 'Error') : st === 'ready' ? badge('ok', 'Ready') : badge('wait', st);
    list.append(h('div', { class: 'file-row' },
      h('div', { class: 'clip' }, h('div', { class: 'name' }, job.name), h('div', { class: 'sub', title: sub }, sub)),
      h('div', { class: 'actions' }, state,
        st === 'error' ? h('a', { class: 'btn', href: `/api/sheets/${id}/log`, target: '_blank', rel: 'noopener' }, 'Log') : null,
        h('button', { class: 'btn-x', title: 'Remove', 'aria-label': `Remove ${job.name}`, onclick: () => removeSheet(id) }, '×'))));
  }
  const anyReady = P.sheets.some((id) => J[id] && (J[id].status === 'ready' || J[id].status === 'done'));
  const anyBusy = P.sheets.some((id) => J[id] && busy(J[id]));
  const cont = anyReady && !anyBusy ? h('button', { class: 'btn btn-primary btn-lg', onclick: continueFromUpload }, 'Continue to step 1 ›') : null;
  const steps = h('div', { class: 'steps3' },
    stepCard('Step 1', 'Building area', 'Confirm the detected outline, leave out the title block.'),
    stepCard('Step 2', 'Scale', 'Read from dimension strings or the title block; measure if neither exists.'),
    stepCard('Then', 'Results', 'The extraction runs by itself; review rooms and areas, download the exports.'));
  const status = h('div', { class: 'status-line' }, h('span', { class: 'spinner' }), 'Checking the local pipeline…');
  const main = h('main', { class: 'upload grid-bg' }, h('div', { class: 'upload-inner' },
    h('div', { class: 'stack-sm' }, h('h2', null, adding ? 'Add a sheet' : 'New project'),
      h('p', { class: 'muted pretty' }, adding ? 'Upload another sheet of the same building. It goes through the same two checks, then its extraction runs and its results join the others.'
        : 'Upload one sheet per floor plan. Vector PDFs and scans both work; scans should be 300 dpi or better.')),
    drop, input, P.sheets.length ? list : null, cont, adding ? h('button', { class: 'btn btn-lg', onclick: () => go('results') }, '‹ Back to the results') : steps, status));
  mount(main, adding ? projectName() : null);
  startPolling();
  if (!health) { try { health = await getJSON('/api/health'); } catch (e) { health = { ok: false, items: { server: { ok: false, detail: e.message } } }; } }
  if (screenName !== 'upload') return;
  put(status, h('span', { class: 'dot10 ' + (health.ok ? 'c-success' : 'c-danger') }),
    health.ok ? 'Processing runs on this machine. Plans are not uploaded anywhere.' : 'Something the pipeline needs is missing on this machine:');
  const rows = Object.entries(health.items || {}).map(([k, v]) => h('div', { class: 'row' }, h('span', { class: v.ok ? 'ok' : 'bad' }, v.ok ? '✓' : '✗'), h('b', null, k), h('span', { class: 'muted' }, v.detail)));
  const det = h('details', null, h('summary', { title: health.pipeline_dir || '' }, `Pipeline check (${(health.pipeline_dir || '').split(/[\\/]/).slice(-3).join('/')})`), h('div', { class: 'health' }, rows));
  if (!health.ok) det.open = true;
  status.after(det);
}
function stepCard(step, name, desc) { return h('div', { class: 'step-card' }, h('div', { class: 'eyebrow' }, step), h('div', { class: 'name' }, name), h('div', { class: 'desc' }, desc)); }
/** The demo sheet: the server creates the job from its bundled copy; the rest is the normal flow. */
async function startDemo() {
  try {
    const r = await api('/api/demo', { method: 'POST' });
    P.sheets.push(r.id); saveProject();
    await refreshJob(r.id);
  } catch (e) { return notify(e.message); }
  renderUpload();
}

async function uploadFiles(files) {
  for (const f of files) {
    try {
      const r = await api(`/api/upload?name=${encodeURIComponent(f.name)}`, { method: 'POST', body: f, headers: { 'Content-Type': 'application/octet-stream' } });
      P.sheets.push(r.id); saveProject();
      await refreshJob(r.id);
      renderUpload();                                      // the spinner shows while the next files upload
    } catch (e) {
      // a TypeError from fetch is a network failure: the page was not opened from the app server (python app/server.py,
      // http://127.0.0.1:8765/) but from disk or another local server, whose answer to the upload closes the connection
      const hint = e instanceof TypeError ? ' – no app server behind this page: start `python app/server.py` and open the address it prints' : '';
      notify(`${f.name}: ${e.message}${hint}`);
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
  if (!c) return go('results');
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

export { autoAdvanced, continueFromUpload, health, removeSheet, renderUpload, stepCard, uploadFiles };
