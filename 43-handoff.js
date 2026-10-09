/* ============================================================
   F13LD.synth · 43-handoff.js
   Hand the selected candidate's recipe on: F13LD.mesh (?r=), F13LD.lab
   (#r=, as F13LD.tpms sends it), the clipboard, and F13LD.queue (the shared
   queue client, mounted as the header's split Queue pill).
   ============================================================ */
'use strict';

function openInMesh(r){
  if(!r) return;
  window.open(MESH_URL + '?r=' + encodeURIComponent(JSON.stringify(r.recipe)), '_blank');
}
function openInLab(r){
  if(!r) return;
  window.open(LAB_URL + '#r=' + encodeURIComponent(JSON.stringify(r.recipe)), '_blank');
}
function copyRecipe(r, btn){
  if(!r) return;
  const txt = JSON.stringify(r.recipe, null, 2);
  const label = btn && btn.querySelector('span');
  const done = () => { if(!label) return; const o = label.textContent; label.textContent = 'Copied'; setTimeout(() => { label.textContent = o; }, 1400); };
  if(navigator.clipboard) navigator.clipboard.writeText(txt).then(done).catch(() => window.prompt('Copy the recipe:', txt));
  else window.prompt('Copy the recipe:', txt);
}

function renderHeaderPills(){
  const r = selectedResult();
  ['hdrMesh', 'hdrLab'].forEach(id => { document.getElementById(id).disabled = !r; });
  document.querySelectorAll('[data-selno]').forEach(e => { e.textContent = r ? '#' + (SYNTH.sel + 1) : ''; });
  const q = document.querySelector('#hdrRight .f13ldq-main');
  if(q) q.title = r ? `Add candidate #${SYNTH.sel + 1} to F13LD.queue — a list you can open on any device` : 'Synthesize first, then add a candidate to F13LD.queue';
}
function wireHandoff(){
  document.getElementById('hdrMesh').addEventListener('click', () => openInMesh(selectedResult()));
  document.getElementById('hdrLab').addEventListener('click', () => openInLab(selectedResult()));
  loadQueueClient(0);
}

// The queue client sets its own button text; Synth swaps in drawn icons.
function mountQueue(){
  if(!window.F13LDQueue) return;
  const fb = document.getElementById('qFallback'); if(fb) fb.remove();
  F13LDQueue.mountButton({ mount: '#hdrRight', anchor: '#hdrMesh', position: 'before', className: 'pill', text: 'Queue',
    getRecipe: () => { const r = selectedResult(); return r ? r.recipe : null; }, tool: 'f13ld.synth' });
  const main = document.querySelector('#hdrRight .f13ldq-main'), caret = document.querySelector('#hdrRight .f13ldq-caret');
  if(main) main.innerHTML = glyph('queue') + 'Queue';
  if(caret){ caret.innerHTML = glyph('chev'); caret.title = 'Choose which queue'; }
  renderHeaderPills();
}
function mountQueueFallback(){
  if(document.getElementById('qFallback')) return;
  const b = document.createElement('button'); b.type = 'button'; b.id = 'qFallback'; b.className = 'pill';
  b.innerHTML = glyph('queue') + 'Queue'; b.title = 'F13LD.queue is offline — reload the page to try again';
  b.addEventListener('click', () => alert('Add to queue is offline — couldn’t reach the queue service. Reload the page to try again.'));
  document.getElementById('hdrRight').insertBefore(b, document.getElementById('hdrMesh'));
}
function loadQueueClient(retry){
  const sc = document.createElement('script');
  sc.src = 'https://mshomper.github.io/f13ld.queue/f13ld-queue.js';
  sc.onload = mountQueue;
  sc.onerror = () => { mountQueueFallback(); if(retry < 1) setTimeout(() => loadQueueClient(retry + 1), 4000); };
  document.head.appendChild(sc);
}
