/* ============================================================
   F13LD.synth · 36-state.js
   What the UI modules share: the current results, the selection, the
   tile order and the search in progress.
   ============================================================ */
'use strict';

const SYNTH = {
  results: [],        // Predictor.toResult() objects, best match first
  sel: -1,            // index into results
  sortKey: 'match',   // strip order: match | conf | light
  running: false,
  stopFlag: false,
  lastReq: null,      // targets/weights the results were scored against
  intentDirty: false, // pads changed since the last search
  depth: DEFAULT_DEPTH
};

function selectedResult(){ return SYNTH.sel >= 0 ? SYNTH.results[SYNTH.sel] : null; }

function selectCandidate(i){
  if(i < 0 || i >= SYNTH.results.length) return;
  SYNTH.sel = i;
  if(typeof renderStrip === 'function') renderStrip();
  if(typeof renderInspector === 'function') renderInspector();
  if(typeof mapRender === 'function') mapRender();
  if(typeof renderHeaderPills === 'function') renderHeaderPills();
}
