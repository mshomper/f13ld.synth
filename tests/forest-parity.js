// node tests/forest-parity.js — the packed forest must predict exactly what
// the v0.2 single-file per-tree walker predicted, for every seed and metric.
'use strict';
const fs = require('fs'), path = require('path');
const SF = require('../11-forest.js');
const b = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, '../weights/tpms.json')));
// v0.2 reference walker, copied verbatim from the old index.html
class TreeWalker { constructor(t){ this.feature=Int32Array.from(t.feature); this.threshold=Float32Array.from(t.threshold); this.left=Int32Array.from(t.left); this.right=Int32Array.from(t.right); this.value=Float32Array.from(t.value); }
  predict(x){ let n=0; while(this.left[n]!==-1) n=(x[this.feature[n]]<=this.threshold[n])?this.left[n]:this.right[n]; return this.value[n]; } }
class RandomForest { constructor(trees){ this.trees=trees.map(t=>new TreeWalker(t)); this.invN=1/Math.max(this.trees.length,1); }
  predict(x){ let s=0; for(let i=0;i<this.trees.length;i++) s+=this.trees[i].predict(x); return s*this.invN; } }
const model = SF.packModel(b);
const refV = b.validity_model ? new RandomForest(b.validity_model.trees) : null;
const refM = b.metrics_model.map(m => m.kind === 'derived' ? null : new RandomForest(m.trees));
let worst = 0, t0 = Date.now();
for(const s of b.seed_samples){
  const x = Float32Array.from(s), r = SF.predictAll(model, x);
  b.output_metrics.forEach((k, i) => { if(refM[i]) worst = Math.max(worst, Math.abs(refM[i].predict(x) - r.pred[k])); });
  if(refV) worst = Math.max(worst, Math.abs(refV.predict(x) - SF.predictValidity(model, x)));
}
console.log(`forest parity: worst difference ${worst.toExponential(2)} over ${b.seed_samples.length} seeds (${Date.now()-t0} ms)`);
process.exit(worst < 1e-9 ? 0 : 1);
