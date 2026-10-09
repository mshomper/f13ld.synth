// node tests/mesh-parity.js [path to f13ld.mesh] — builds Synth results with
// F13LD.mesh's own TPMS field code (worker/m20-sdf-noise-tpms.js) and checks
// the solid volume fraction Mesh would build against Synth's prediction.
// Needs the f13ld.mesh repo checked out beside this one (or its path).
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const SE = require('../10-encoding.js'), SF = require('../11-forest.js'), SS = require('../12-search-core.js');
const meshDir = process.argv[2] || path.join(__dirname, '../../f13ld.mesh');
const src = fs.readFileSync(path.join(meshDir, 'worker/m20-sdf-noise-tpms.js'), 'utf8');
const sandbox = { Math, registerSDF(){}, console };
vm.runInNewContext(src + '\nthis.buildTPMSSDF = buildTPMSSDF;', sandbox);
const b = JSON.parse(fs.readFileSync(path.join(__dirname, '../weights/tpms.json')));
const enc = SE.create(b.encoding, b.meta.feature_dim), msg = SF.toMessage(SF.packModel(b));
const table = SS.buildSeedTable(b, enc, SF.fromMessage(msg));
const ctx = SS.makeContext({ encoding: b.encoding, featureDim: b.meta.feature_dim, model: msg, outputMetrics: b.output_metrics,
  sigmas: SS.sigmasFromBundle(b), seeds: table.seeds, ranges: table.ranges, spreadRef: table.spreadRef });
// a spread of intents so all three modes show up
const intents = [
  { ex_norm: 0.25, ey_norm: 0.25, ez_norm: 0.25, pore_size_p50_norm: 0.27 },
  { ex_norm: 0.03, ey_norm: 0.03, ez_norm: 0.03, pore_size_p50_norm: 0.35 },
  { volume_fraction: 20, keff_avg_norm: 0.1 },
];
const N = 36, rows = [];
intents.forEach((targets, ii) => {
  const weights = Object.fromEntries(Object.keys(targets).map(k => [k, 1]));
  const r = SS.run(ctx, { parents: ctx.seeds.map(s => ({ design: s.design, seedIndex: s.index })), count: 800, strength: 0.06, rngSeed: 11 + ii, targets, weights, connectivity: null, keep: 6 });
  for(const c of r.candidates.slice(0, 4)){
    const recipe = SE.toRecipe(c.design, { presetKey: c.presetKey });
    const sdf = sandbox.buildTPMSSDF(recipe);
    // one tile: world [-5,5] spans cell_scale periods; sample cell centres
    let inside = 0;
    for(let i = 0; i < N; i++) for(let j = 0; j < N; j++) for(let k = 0; k < N; k++){
      const p = [-5 + (i + .5) * 10 / N, -5 + (j + .5) * 10 / N, -5 + (k + .5) * 10 / N];
      if(sdf(p) < 0) inside++;
    }
    rows.push({ mode: c.design.mode, preset: c.presetKey, built: 100 * inside / (N*N*N), predicted: c.pred.volume_fraction });
  }
});
let worst = 0;
for(const r of rows){
  const d = r.built - r.predicted; worst = Math.max(worst, Math.abs(d));
  console.log(`${r.mode.padEnd(8)} ${r.preset.padEnd(14)} built ${r.built.toFixed(1).padStart(5)}%  predicted ${r.predicted.toFixed(1).padStart(5)}%  diff ${d >= 0 ? '+' : ''}${d.toFixed(1)}`);
}
console.log(`volume fraction, Mesh build vs Synth prediction: worst |diff| ${worst.toFixed(1)} points over ${rows.length} designs`);
