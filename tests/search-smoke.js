// node tests/search-smoke.js — runs one explore + refine search on the main
// thread code path with the default pad intent, and measures how far the
// v0.2 build's scored vectors were from the recipes it actually sent.
'use strict';
const fs = require('fs'), path = require('path');
const SE = require('../10-encoding.js'), SF = require('../11-forest.js'), SS = require('../12-search-core.js');
const b = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, '../weights/tpms.json')));
const enc = SE.create(b.encoding, b.meta.feature_dim);
const msg = SF.toMessage(SF.packModel(b));
let t0 = Date.now();
const table = SS.buildSeedTable(b, enc, SF.fromMessage(msg));
console.log(`seed table: ${table.seeds.length} seeds, presets`, table.presets, `(${Date.now()-t0} ms)`);
const ctx = SS.makeContext({ encoding: b.encoding, featureDim: b.meta.feature_dim, model: msg,
  outputMetrics: b.output_metrics, sigmas: SS.sigmasFromBundle(b), seeds: table.seeds, ranges: table.ranges, spreadRef: table.spreadRef });
// default intent: Mechanics × Pore pad on require at the centre
const targets = { ex_norm: 0.2525, ey_norm: 0.2525, ez_norm: 0.2525, pore_size_p50_norm: 0.27 };
const weights = { ex_norm: 1, ey_norm: 0.7, ez_norm: 0.7, pore_size_p50_norm: 1 };
const req = { targets, weights, connectivity: 3, presetKey: null, keep: 48 };
t0 = Date.now();
const ex = SS.run(ctx, Object.assign({ phase: 'explore', count: 3000, rngSeed: 7 }, req));
const tEx = Date.now() - t0; t0 = Date.now();
const parents = ex.candidates.slice(0, 16).map(c => ({ design: c.design, seedIndex: c.seedIndex }));
const rf = SS.run(ctx, Object.assign({ phase: 'refine', parents, count: 120, rngSeed: 8 }, req));
const tRf = Date.now() - t0;
const final = SS.pickFinal(ex.candidates.concat(rf.candidates), 8, 2);
console.log(`explore 3000: ${tEx} ms · refine ${parents.length}×120: ${tRf} ms`);
console.log('explore stats', ex.stats);
console.log('top explore score', ex.candidates[0] && (ex.candidates[0].score*100).toFixed(1)+'%', '→ after refine', (final[0].score*100).toFixed(1)+'%');
final.forEach((c, i) => console.log(`#${i+1} score ${(c.score*100).toFixed(0)}% validity ${(c.validity*100).toFixed(0)}% conf ${SS.confidenceLabel(c.spreadRatio)} (${c.spreadRatio.toFixed(2)}) ${c.design.mode} seed ${c.seedIndex} ${c.presetKey}`));
// every final candidate re-scores identically from its recipe
let mismatch = 0;
for(const c of final){
  const rec = SE.toRecipe(c.design, { presetKey: c.presetKey });
  const back = { mode: rec.geometry.mode, cell_scale: rec.geometry.cell_scale,
    wall_thickness: rec.geometry.wall_thickness || 0, pipe_radius: rec.geometry.pipe_radius || 0, offset: rec.geometry.offset || 0,
    phase_shift: rec.geometry.phase_shift || {x:0,y:0,z:0}, normal_weights: rec.geometry.normal_weights || {wx:1,wy:1,wz:1},
    terms: rec.surface.terms.map(t => ({ coef: t.coef, phase_shift: t.phase_shift || {x:0,y:0,z:0}, factors: t.factors })) };
  if(!enc.sameVector(enc.encode(c.design), enc.encode(back))) mismatch++;
}
console.log(`recipe → features: ${final.length - mismatch}/${final.length} identical to what was scored`);

// v0.2 gap: blurred vector score vs the recipe v0.2 decoded and sent
function z(pred){ let s=0,w=0; for(const k in weights){ const zz=Math.min(Math.abs((pred[k]-targets[k])/ctx.sigmas[k]),3); s+=zz*zz*weights[k]; w+=weights[k]; } return Math.exp(-(s/w)/6); }
const R = SS.rng(3), lo = b.input_norm.lo, hi = b.input_norm.hi; let gaps = [];
for(let i = 0; i < 400; i++){
  const s = b.seed_samples[Math.floor(R()*200)], x = new Float32Array(s.length);
  for(let k=0;k<s.length;k++){ const span=Math.max(hi[k]-lo[k],1e-8); x[k]=Math.max(lo[k],Math.min(hi[k],s[k]+R.gauss()*span*0.05)); }
  const d = enc.decode(x);                               // snap, as v0.2's decodeRecipe did …
  d.normal_weights = {wx:1,wy:1,wz:1};                   // … which dropped normal weights,
  d.terms = d.terms.filter(t => t.factors.length).map(t => ({ coef: t.coef, phase_shift:{x:0,y:0,z:0},   // per-term phases, constants,
    factors: t.factors.map(f => ({ trig: f.trig, fx: Math.max(1,Math.min(3,Math.round(f.fx))), fy: Math.max(1,Math.min(3,Math.round(f.fy))), fz: Math.max(1,Math.min(3,Math.round(f.fz))) })) }));
  if(!d.terms.length) continue;
  const a = z(SF.predictAll(ctx.model, x).pred), c = z(SF.predictAll(ctx.model, enc.encode(d)).pred);
  gaps.push(Math.abs(a - c) * 100);
}
gaps.sort((p,q)=>p-q);
console.log(`v0.2 gap between scored vector and sent recipe (score points): median ${gaps[gaps.length>>1].toFixed(1)}, 90th pct ${gaps[Math.floor(gaps.length*0.9)].toFixed(1)}, max ${gaps[gaps.length-1].toFixed(1)}`);
process.exit(mismatch ? 1 : 0);
