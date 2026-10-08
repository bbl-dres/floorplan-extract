/* Floor plan workflow app: upload -> building area -> scale -> results and export (the extraction starts when the scale is confirmed).
   Follows the boards of docs/wireframes/261007_Viewer and Workflow UX study.html (1c, 1d, 2a-2c, 1g/1h). Entry module: registers the screens and boots. */
import { renderArea } from './screens/area.js';
import { $ } from './core.js';
import { renderResults } from './screens/results.js';
import { renderScale } from './screens/scale.js';
import { renderUpload } from './screens/upload.js';
import { J, P, onJobs, refreshAll, resetProject, saveProject, startPolling } from './state.js';
import { go, initChrome, onJobsChanged, register } from './ui.js';

for (const [name, fn] of Object.entries({ upload: renderUpload, area: renderArea, scale: renderScale, results: renderResults })) register(name, fn);
for (const old of ['storeys', 'run']) register(old, () => go('results'));   // the run step of earlier saved projects and links
onJobs(onJobsChanged);

async function boot() {
  if (location.protocol === 'file:') { $('#offline').hidden = false; return; }
  $('#app').hidden = false;
  const openHelp = () => { $('#help').hidden = false; $('#help-close').focus(); };
  const closeHelp = () => { $('#help').hidden = true; $('#btn-help').focus(); };
  $('#btn-help').onclick = openHelp;
  $('#help-close').onclick = closeHelp;
  $('#help').onclick = (e) => { if (e.target === $('#help')) closeHelp(); };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('#help').hidden) closeHelp(); });
  initChrome();
  $('#btn-start-over').onclick = () => { if (!confirm('Start a new project? The processed sheets stay in app/data until you remove them from the upload list.')) return; resetProject(); go('upload'); };
  // deep link: ?sheets=<id>,<id>&screen=area|scale|results&key=<id>|<package>/<drawing>&pkg=0&drawing=0&sheet=0 (also used for screenshots)
  const q = new URLSearchParams(location.search);
  if (q.get('sheets')) {
    P.sheets = q.get('sheets').split(',').filter(Boolean); P.resultsKey = q.get('key') || null;
    P.screen = q.get('screen') || 'upload'; P.cursor = { sheet: +(q.get('sheet') || 0), pkg: +(q.get('pkg') || 0), drawing: +(q.get('drawing') || 0) };
    saveProject(); history.replaceState(null, '', location.pathname);
  }
  try { await refreshAll(); } catch (e) { /* the server is checked per request */ }
  startPolling();
  const first = P.sheets.length ? P.screen : 'upload';
  if ((first === 'area' || first === 'scale') && !(J[P.sheets[P.cursor.sheet]] || {}).analysis) return go('upload');
  go(first);
}
document.addEventListener('DOMContentLoaded', boot);
