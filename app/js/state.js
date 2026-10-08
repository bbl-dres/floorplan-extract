/* Project state (which sheet jobs, where the user is), the job records from the server, polling, access to the analysis. */
import { PKEY, api, getJSON } from './core.js';

// ---------------------------------------------------------------------------------------------------------------------
// Project state (which sheet jobs belong to the project, in which order, and where the user is)

const P = loadProject();                // mutated in place; resetProject() for a new project
const J = {};                          // job id -> job record from the server
let screenName = null;                   // the screen on show; ui.go() sets it through setScreen (an import cannot be assigned)
function setScreen(name) { screenName = name; }
let pollTimer = null;

function loadProject() {
  try {
    const p = JSON.parse(localStorage.getItem(PKEY) || 'null');
    if (p && Array.isArray(p.sheets)) return Object.assign({ name: '', sheets: [], screen: 'upload', cursor: { sheet: 0, pkg: 0, drawing: 0 }, resultsKey: null }, p);
  } catch (e) { /* a broken entry: start fresh */ }
  return { name: '', sheets: [], screen: 'upload', cursor: { sheet: 0, pkg: 0, drawing: 0 }, resultsKey: null };
}
function saveProject() { try { localStorage.setItem(PKEY, JSON.stringify(P)); } catch (e) { /* private window */ } }
function resetProject() { Object.assign(P, { name: '', sheets: [], screen: 'upload', cursor: { sheet: 0, pkg: 0, drawing: 0 }, resultsKey: null }); saveProject(); }

/** The job record from the server; the revision the client holds goes along, and an unchanged job costs one small reply. */
async function refreshJob(id) {
  try {
    const have = J[id];
    const r = await getJSON(`/api/sheets/${id}${have && have.rev != null ? `?rev=${have.rev}` : ''}`);
    if (!r.unchanged) J[id] = r;
  } catch (e) {
    if (/not found/.test(e.message)) { P.sheets = P.sheets.filter((s) => s !== id); saveProject(); delete J[id]; }
    else J[id] = J[id] || { id, status: 'error', error: e.message, log: [], confirm: {} };
  }
  return J[id];
}

async function refreshAll() { await Promise.all(P.sheets.map(refreshJob)); }

function busy(job) { return ['analysing', 'scaling', 'queued', 'running'].includes(job.status); }

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    const ids = P.sheets.filter((id) => !J[id] || busy(J[id]));
    if (!ids.length) return;
    const before = ids.map((id) => J[id] && (J[id].status + '|' + JSON.stringify(J[id].progress) + '|' + (J[id].log || []).length));
    await Promise.all(ids.map(refreshJob));
    const after = ids.map((id) => J[id] && (J[id].status + '|' + JSON.stringify(J[id].progress) + '|' + (J[id].log || []).length));
    const changed = ids.filter((id, k) => before[k] !== after[k]);
    if (changed.length && jobsListener) jobsListener(changed);
  }, 1500);
}
let jobsListener = null;                 // set by the app: what to re-render when jobs change (ui.onJobsChanged)
function onJobs(fn) { jobsListener = fn; }

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
function drawingLabel(job, pkg, dr) { const c = confOf(job, pkg, dr); return c.storey || dr.storey || ''; }

/** The drawings of the project selected for extraction, in sheet, page and drawing order. */
function drawingRows() {
  const rows = [];
  for (const id of P.sheets) {
    const job = J[id];
    if (!job || !job.analysis) continue;
    for (const pkg of packagesOf(job)) for (const dr of selectedDrawings(job, pkg)) {
      const key = `${id}|${keyOf(pkg, dr)}`;
      const res = job.results && job.results[keyOf(pkg, dr)];
      rows.push({ key, id, job, pkg, dr, conf: confOf(job, pkg, dr), res, label: drawingLabel(job, pkg, dr) });
    }
  }
  return rows;
}

/** True when every selected drawing of the sheet has a confirmed scale (the extraction can start). */
function sheetConfirmed(job) {
  return packagesOf(job).every((pk) => selectedDrawings(job, pk).every((d) => confOf(job, pk, d).scale_confirmed));
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

export { J, P, bboxOf, busy, confOf, drawingLabel, drawingRows, firstUnconfirmed, jobsListener, keyOf, loadProject, onJobs, packagesOf, pollTimer, polygonOf, ppmOf, refreshAll, refreshJob, resetProject, saveProject, screenName, selectedDrawings, setScreen, sheetConfirmed, startPolling };
