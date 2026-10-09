// node tests/roundtrip.js [bundle]  — every training seed must survive
// decode → encode bit for bit, and decode → recipe must keep every field.
'use strict';
const fs = require('fs'), path = require('path');
const SE = require('../10-encoding.js');
const b = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, '../weights/tpms.json')));
const enc = SE.create(b.encoding, b.meta.feature_dim);
let bad = 0, sk = {};
b.seed_samples.forEach((s, i) => {
  const x = Float32Array.from(s), d = enc.decode(x), y = enc.encode(d);
  if(!enc.sameVector(x, y)){ bad++; if(bad < 4){ for(let k=0;k<x.length;k++) if(x[k]!==y[k]) console.log('seed',i,'slot',k,x[k],y[k]); } }
  const m = SE.matchSkeleton(d.terms, enc.maxTerms).join('|') || '(none)';
  sk[m] = (sk[m] || 0) + 1;
});
console.log(`round trip: ${b.seed_samples.length - bad}/${b.seed_samples.length} seeds exact`);
console.log('preset skeletons found:', sk);
process.exit(bad ? 1 : 0);
