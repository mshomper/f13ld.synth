/* ============================================================
   F13LD.synth · worker/search-worker.js
   Runs SynthSearch.run() for one share of a search.

   Messages in:
     {type:'init', init}          init = the context 20-predictor.js builds
     {type:'run', id, req}        one explore or refine job
   Messages out:
     {type:'ready'} · {type:'result', id, result} · {type:'error', id, message}

   The single-file preview build inlines 10/11/12 ahead of this file and
   starts the worker from a Blob, so importScripts is skipped there.
   ============================================================ */
'use strict';
if(typeof SynthSearch === 'undefined') importScripts('../10-encoding.js', '../11-forest.js', '../12-search-core.js');

let ctx = null;
self.onmessage = function(e){
  const m = e.data;
  try {
    if(m.type === 'init'){ ctx = SynthSearch.makeContext(m.init); self.postMessage({ type: 'ready' }); return; }
    if(m.type === 'run'){
      if(!ctx) throw new Error('search worker used before init');
      self.postMessage({ type: 'result', id: m.id, result: SynthSearch.run(ctx, m.req) });
    }
  } catch(err){
    self.postMessage({ type: 'error', id: m.id, message: (err && err.message) || String(err) });
  }
};
