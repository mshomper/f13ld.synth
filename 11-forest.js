/* ============================================================
   F13LD.synth · 11-forest.js
   Random-forest inference on flat typed arrays.

   Every tree of a forest is packed into one buffer of 12-byte node
   records (indices already offset), plus a list of root nodes. That makes it cheap to copy into search workers and fast to walk.

   predictStats() returns the mean (the forest's prediction, identical to
   the per-tree walk the old single-file build used) and the spread of the
   individual trees. The spread is the confidence signal: trees that agree
   mean the candidate sits where the training data is dense.
   Worker-safe: no DOM access.
   ============================================================ */
'use strict';

const SynthForest = (function(){

  // trees: [{feature, threshold, left, right, value}] as train_synth.py writes them.
  // Each node is one 12-byte record in a single buffer:
  //   int32  [3n]   split feature, or -1 at a leaf
  //   float32[3n+1] split threshold, or the leaf value
  //   int32  [3n+2] right child (absolute index)
  // The left child is always n+1 (scikit-learn builds trees depth-first;
  // pack() checks this and throws if a bundle ever breaks it).
  function pack(trees){
    let n = 0;
    for(const t of trees) n += t.feature.length;
    const buf = new ArrayBuffer(n * 12);
    const I = new Int32Array(buf), F = new Float32Array(buf);
    const roots = new Int32Array(trees.length);
    let off = 0;
    trees.forEach((t, ti) => {
      roots[ti] = off;
      const m = t.feature.length;
      for(let i = 0; i < m; i++){
        const k = (off + i) * 3;
        if(t.left[i] === -1){ I[k] = -1; F[k+1] = t.value[i]; continue; }
        if(t.left[i] !== i + 1) throw new Error('Model bundle tree is not depth-first ordered; Synth cannot pack it.');
        I[k] = t.feature[i]; F[k+1] = t.threshold[i]; I[k+2] = t.right[i] + off;
      }
      off += m;
    });
    return { buf, I, F, roots };
  }

  function predict(P, x){
    const I = P.I, F = P.F, R = P.roots;
    let s = 0;
    for(let k = 0; k < R.length; k++){
      let n = R[k], f;
      while((f = I[n*3]) !== -1) n = (x[f] <= F[n*3+1]) ? n + 1 : I[n*3+2];
      s += F[n*3+1];
    }
    return s / R.length;
  }

  function predictStats(P, x){
    const I = P.I, F = P.F, R = P.roots;
    let s = 0, s2 = 0;
    for(let k = 0; k < R.length; k++){
      let n = R[k], f;
      while((f = I[n*3]) !== -1) n = (x[f] <= F[n*3+1]) ? n + 1 : I[n*3+2];
      const v = F[n*3+1];
      s += v; s2 += v * v;
    }
    const N = R.length, mean = s / N;
    return { mean, std: Math.sqrt(Math.max(s2 / N - mean * mean, 0)) };
  }

  // Rebuild views after a structured-clone hop into a worker.
  function revive(P){ return { buf: P.buf, I: new Int32Array(P.buf), F: new Float32Array(P.buf), roots: P.roots }; }

  // The whole model: validity classifier + one entry per output metric.
  // metricsModel entries are {kind:'trained', forest} or {kind:'derived', op, inputs, floor}.
  function packModel(bundle){
    return {
      validity: bundle.validity_model ? pack(bundle.validity_model.trees) : null,
      metrics: bundle.metrics_model.map(m => (m && m.kind === 'derived')
        ? { kind: 'derived', op: m.op, inputs: m.inputs, floor: m.floor }
        : { kind: 'trained', forest: pack(m.trees) }),
      outputMetrics: bundle.output_metrics.slice()
    };
  }

  // A copy of the model that posts cleanly to a worker (buffers + roots only).
  function toMessage(model){
    const strip = P => P ? { buf: P.buf, roots: P.roots } : null;
    return { validity: strip(model.validity), outputMetrics: model.outputMetrics,
      metrics: model.metrics.map(m => m.kind === 'trained' ? { kind: 'trained', forest: strip(m.forest) } : m) };
  }
  function fromMessage(msg){
    return { validity: msg.validity ? revive(msg.validity) : null, outputMetrics: msg.outputMetrics,
      metrics: msg.metrics.map(m => m.kind === 'trained' ? { kind: 'trained', forest: revive(m.forest) } : m) };
  }

  function applyDerived(spec, pred){
    const vals = spec.inputs.map(k => pred[k]);
    if(spec.op === 'max_over_min'){
      const floor = spec.floor != null ? spec.floor : 0.01;
      return Math.max.apply(null, vals) / Math.max(Math.min.apply(null, vals), floor);
    }
    throw new Error('Unknown derived op: ' + spec.op);
  }

  // → { pred:{metric:value}, spread:{metric:std} }
  function predictAll(model, x){
    const pred = {}, spread = {};
    const M = model.outputMetrics;
    for(let i = 0; i < M.length; i++){
      const m = model.metrics[i];
      if(m.kind !== 'trained') continue;
      const r = predictStats(m.forest, x);
      pred[M[i]] = r.mean; spread[M[i]] = r.std;
    }
    for(let i = 0; i < M.length; i++){
      const m = model.metrics[i];
      if(m.kind === 'derived') pred[M[i]] = applyDerived(m, pred);
    }
    return { pred, spread };
  }

  function predictValidity(model, x){ return model.validity ? predict(model.validity, x) : 1; }

  return { pack, predict, predictStats, packModel, toMessage, fromMessage, predictAll, predictValidity };
})();

if(typeof module !== 'undefined') module.exports = SynthForest;
