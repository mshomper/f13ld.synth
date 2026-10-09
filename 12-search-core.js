/* ============================================================
   F13LD.synth · 12-search-core.js
   The inverse search: grow real designs from training seeds, score the
   exact design, keep the best, refine around them.

   Candidates are built by mutating a decoded seed design the way
   F13LD.sweep varies a recipe (continuous nudges within the range the
   training data covers, occasional sin/cos swaps and frequency changes),
   then encoded. The vector the forest scores is therefore always a real
   design, never a blend of half-on terms or fractional mode switches.

   Runs the same in a search worker and on the main thread.
   Needs 10-encoding.js and 11-forest.js. No DOM access.
   ============================================================ */
'use strict';

const SynthSearch = (function(){
  const SE = (typeof SynthEncoding !== 'undefined') ? SynthEncoding : require('./10-encoding.js');
  const SF = (typeof SynthForest !== 'undefined') ? SynthForest : require('./11-forest.js');

  const TWO_PI = 2 * Math.PI;
  const VALIDITY_FLOOR = 0.3;   // candidates the classifier rates below this are dropped
  const EXPLORE_STRENGTH = 0.06, REFINE_STRENGTH = 0.02;

  // Small seeded generator so a worker's share of a search is reproducible.
  function rng(seed){
    let a = seed >>> 0;
    const next = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    next.gauss = () => {
      if(spare !== null){ const v = spare; spare = null; return v; }
      const u = Math.max(next(), 1e-12), v = next(), r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(TWO_PI * v); return r * Math.cos(TWO_PI * v);
    };
    return next;
  }

  const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;
  const r3 = v => SE.round(v, 3), r4 = v => SE.round(v, 4);

  // ── Seed table (built once on the main thread, posted to workers) ─────────
  // Per-mode ranges keep every mutation inside what the training data covers.
  function buildSeedTable(bundle, enc, model){
    const seedPresets = Array.isArray(bundle.seed_presets) ? bundle.seed_presets : null;
    const truncatedAt = (bundle.encoding.term_semantics === 'on_terms') ? null : enc.maxTerms;
    const customGroups = {};
    const seeds = bundle.seed_samples.map((s, i) => {
      const x = Float32Array.from(s);
      const design = enc.decode(x);
      let presetKey = seedPresets ? SE.canonicalPreset(seedPresets[i]) : null;
      if(!presetKey){
        const m = SE.matchSkeleton(design.terms, truncatedAt);
        if(m.length) presetKey = m.join('|');
        else {
          const sk = design.mode + ':' + SE.skeletonOf(design.terms);
          if(!(sk in customGroups)) customGroups[sk] = 'custom-' + (Object.keys(customGroups).length + 1);
          presetKey = customGroups[sk];
        }
      }
      return { index: i, design, presetKey };
    });

    const ranges = {};
    const grow = (r, k, v) => { if(!(k in r)) r[k] = [v, v]; else { r[k][0] = Math.min(r[k][0], v); r[k][1] = Math.max(r[k][1], v); } };
    for(const s of seeds){
      const d = s.design, r = ranges[d.mode] || (ranges[d.mode] = {});
      grow(r, 'cell_scale', d.cell_scale); grow(r, 'wall_thickness', d.wall_thickness);
      grow(r, 'pipe_radius', d.pipe_radius); grow(r, 'offset', d.offset);
      grow(r, 'nw', d.normal_weights.wx); grow(r, 'nw', d.normal_weights.wy); grow(r, 'nw', d.normal_weights.wz);
      for(const t of d.terms) if(t.factors.length) grow(r, 'coef', t.coef);
    }

    // Reference tree spread per metric: the median over training seeds.
    // A candidate's spread divided by this says how far it has wandered
    // from where the forest was trained.
    const spreadRef = {};
    const all = {};
    for(const s of seeds){
      const r = SF.predictAll(model, enc.encode(s.design));
      for(const k in r.spread) (all[k] || (all[k] = [])).push(r.spread[k]);
    }
    for(const k in all){ const a = all[k].sort((p, q) => p - q); spreadRef[k] = Math.max(a[a.length >> 1], 1e-6); }

    const presets = {};
    for(const s of seeds) presets[s.presetKey] = (presets[s.presetKey] || 0) + 1;
    return { seeds, ranges, spreadRef, presets };
  }

  // ── Mutation ──────────────────────────────────────────────────────────────
  function isTrivialPairShift(p){
    const w = v => ((v % 1) + 1) % 1;
    const x = w(p.x), y = w(p.y), z = w(p.z);
    return (x === 0 && y === 0 && z === 0) || (x === 0.5 && y === 0.5 && z === 0.5);
  }

  function mutate(seedDesign, R, strength, range){
    const d = SE.cloneDesign(seedDesign);
    const k = strength / EXPLORE_STRENGTH;              // discrete-change rate scales with strength
    const span = key => { const r = range[key]; return r ? Math.max(r[1] - r[0], 0.05) : 0.1; };
    const lim = (key, v) => { const r = range[key]; return r ? clamp(v, r[0], r[1]) : v; };

    // Cell scale lives on a quarter grid in the training data.
    if(R() < 0.12 * k){
      d.cell_scale = lim('cell_scale', d.cell_scale + (R() < 0.5 ? -0.25 : 0.25));
    }
    if(d.mode === 'shell'){
      d.wall_thickness = r3(lim('wall_thickness', d.wall_thickness + R.gauss() * strength * span('wall_thickness')));
      const nw = d.normal_weights;
      const j = v => lim('nw', v + R.gauss() * strength * span('nw'));
      let wx = j(nw.wx), wy = j(nw.wy), wz = j(nw.wz);
      const mean = (wx + wy + wz) / 3;                   // Sweep keeps the mean at 1
      d.normal_weights = { wx: r4(wx / mean), wy: r4(wy / mean), wz: r4(wz / mean) };
    } else if(d.mode === 'solid'){
      d.offset = r3(lim('offset', d.offset + R.gauss() * strength * span('offset')));
    } else if(d.mode === 'pi-tpms'){
      d.pipe_radius = r3(lim('pipe_radius', d.pipe_radius + R.gauss() * strength * span('pipe_radius')));
      if(R() < 0.25 * k){
        const ax = ['x', 'y', 'z'][Math.floor(R() * 3)];
        const ps = { x: d.phase_shift.x, y: d.phase_shift.y, z: d.phase_shift.z };
        ps[ax] = clamp(ps[ax] + (R() < 0.5 ? -0.125 : 0.125), 0, 1);
        if(!isTrivialPairShift(ps)) d.phase_shift = ps;
      }
    }

    const isPI = d.mode === 'pi-tpms';
    for(const t of d.terms){
      if(!t.factors.length) continue;                    // additive constant: leave as Sweep does
      let c = t.coef + R.gauss() * strength * 1.0;
      c = clamp(c, -1, 1);
      if(Math.abs(c) < 0.02) c = c < 0 ? -0.02 : 0.02;
      t.coef = r3(c);
      if(!isPI){
        const p = t.phase_shift;
        if(p.x || p.y || p.z){
          const wrap = v => r4(((v % TWO_PI) + TWO_PI) % TWO_PI);
          t.phase_shift = { x: wrap(p.x + R.gauss() * strength * Math.PI),
                            y: wrap(p.y + R.gauss() * strength * Math.PI),
                            z: wrap(p.z + R.gauss() * strength * Math.PI) };
        }
        for(const f of t.factors){
          if(R() < 0.06 * k) f.trig = f.trig.startsWith('sin') ? f.trig.replace('sin', 'cos') : f.trig.replace('cos', 'sin');
          if(R() < 0.05 * k){
            const key = 'f' + f.trig.charAt(4);          // the frequency that axis actually uses
            f[key] = 1 + Math.floor(R() * 3);
          }
        }
      }
    }
    return d;
  }

  // ── Scoring ───────────────────────────────────────────────────────────────
  // Weighted RMS of per-metric z-scores, each measured in the model's own
  // residual sigma; |z| capped at 3 so one far-off metric can't dominate.
  // Score = exp(-z²/6): ~85% at 1σ, ~50% at 2σ.
  function scoreCandidate(ctx, req, d, stats){
    const { enc, model } = ctx;
    if(SE.isDegenerate(d)){ stats.degenerate++; return null; }
    const x = enc.encode(d);
    const validity = SF.predictValidity(model, x);
    if(validity < VALIDITY_FLOOR){ stats.validity++; return null; }
    const { pred, spread } = SF.predictAll(model, x);
    if(req.connectivity != null && pred.directionality != null){
      const v = pred.directionality;
      const bucket = v < 0.5 ? 1 : (v < 0.833 ? 2 : 3);
      stats.buckets[bucket]++;
      if(bucket !== req.connectivity){ stats.connectivity++; return null; }
    }
    let z2 = 0, tw = 0, sr = 0, sw = 0;
    const zPerMetric = {};
    for(const m of ctx.outputMetrics){
      const w = req.weights[m] || 0;
      if(!w || req.targets[m] == null) continue;
      const z = (pred[m] - req.targets[m]) / ctx.sigmas[m];
      zPerMetric[m] = z;
      const zc = Math.min(Math.abs(z), 3);
      z2 += zc * zc * w; tw += w;
      if(spread[m] != null && ctx.spreadRef[m]){ sr += (spread[m] / ctx.spreadRef[m]) * w; sw += w; }
    }
    const zRms = tw > 0 ? Math.sqrt(z2 / tw) : 0;
    const score = Math.exp(-(zRms * zRms) / 6);
    const spreadRatio = sw > 0 ? sr / sw : 1;
    const conf = 1 / (1 + 0.35 * Math.max(0, spreadRatio - 1));
    return { design: d, validity, pred, spread, zPerMetric, zRms, score, spreadRatio,
             rank: score * Math.sqrt(validity) * conf };
  }

  // Keep the best `keep` by rank, one entry per distinct design.
  function topK(list, keep){
    const seen = new Set(), out = [];
    list.sort((a, b) => b.rank - a.rank);
    for(const c of list){
      const key = SE.designKey(c.design);
      if(seen.has(key)) continue;
      seen.add(key); out.push(c);
      if(out.length >= keep) break;
    }
    return out;
  }

  // One unit of work. req:
  //   targets, weights, connectivity, presetKey (null = any)
  //   phase 'explore': count candidates grown from random seeds
  //   phase 'refine' : parents [{design, seedIndex}], count per parent
  function run(ctx, req){
    const R = rng(req.rngSeed || 1);
    const stats = { scanned: 0, validity: 0, degenerate: 0, connectivity: 0, buckets: { 1: 0, 2: 0, 3: 0 } };
    const out = [];
    const push = (c, seedIndex) => { if(c){ c.seedIndex = seedIndex; c.presetKey = ctx.seeds[seedIndex].presetKey; out.push(c); } };
    if(req.phase === 'refine'){
      for(const p of req.parents){
        const seed = ctx.seeds[p.seedIndex];
        const range = ctx.ranges[p.design.mode] || {};
        for(let i = 0; i < req.count; i++){
          stats.scanned++;
          push(scoreCandidate(ctx, req, mutate(p.design, R, REFINE_STRENGTH, range), stats), seed.index);
        }
      }
    } else {
      const pool = req.presetKey ? ctx.seeds.filter(s => s.presetKey === req.presetKey) : ctx.seeds;
      if(!pool.length) return { candidates: [], stats };
      for(let i = 0; i < req.count; i++){
        stats.scanned++;
        const seed = pool[Math.floor(R() * pool.length)];
        const range = ctx.ranges[seed.design.mode] || {};
        // A small share are the training seeds themselves: real, solved designs.
        const d = (R() < 0.05) ? SE.cloneDesign(seed.design) : mutate(seed.design, R, EXPLORE_STRENGTH, range);
        push(scoreCandidate(ctx, req, d, stats), seed.index);
      }
    }
    return { candidates: topK(out, req.keep || 48), stats };
  }

  // Context from the init message. init.model is SynthForest.toMessage()
  // output; fromMessage() only rebuilds views, so on the main thread this
  // shares the same buffers rather than copying them.
  function makeContext(init){
    const enc = SE.create(init.encoding, init.featureDim);
    const model = SF.fromMessage(init.model);
    return { enc, model, outputMetrics: init.outputMetrics, sigmas: init.sigmas,
             seeds: init.seeds, ranges: init.ranges, spreadRef: init.spreadRef };
  }

  // Per-metric residual sigma: the trainer's held-out value when the bundle
  // has it, otherwise range/4 · sqrt(1 − R²) as the v0.2 build estimated it.
  function sigmasFromBundle(bundle){
    const out = {}, ev = bundle.eval || {}, sig = ev.metrics_test_sigma || {}, r2s = ev.metrics_test_r2 || {};
    for(const k of bundle.output_metrics){
      if(sig[k] != null && isFinite(sig[k])) { out[k] = Math.max(sig[k], 1e-6); continue; }
      const r2 = r2s[k] != null ? r2s[k] : 0.5, rg = bundle.output_ranges[k];
      const span = Math.max((rg.max - rg.min) || 1, 1e-8);
      out[k] = Math.max((span / 4) * Math.sqrt(Math.max(1 - r2, 0.05)), 1e-6);
    }
    return out;
  }

  // Final list: best first, at most `perSeed` results grown from one seed so
  // eight cards are not eight near-copies of the same design.
  function pickFinal(cands, n, perSeed){
    const ranked = topK(cands, cands.length), per = {}, out = [], skipped = [];
    for(const c of ranked){
      per[c.seedIndex] = (per[c.seedIndex] || 0) + 1;
      if(per[c.seedIndex] > perSeed){ skipped.push(c); continue; }
      out.push(c);
      if(out.length >= n) break;
    }
    // Few distinct seeds (a narrow preset filter): top up with the best of the rest.
    for(const c of skipped){ if(out.length >= n) break; out.push(c); }
    return out.sort((a, b) => b.rank - a.rank);
  }

  function mergeStats(a, b){
    if(!a) return JSON.parse(JSON.stringify(b));
    for(const k of ['scanned', 'validity', 'degenerate', 'connectivity']) a[k] += b[k];
    for(const k of [1, 2, 3]) a.buckets[k] += b.buckets[k];
    return a;
  }

  // Confidence label from the tree-spread ratio.
  function confidenceLabel(ratio){ return ratio <= 1.25 ? 'high' : ratio <= 2 ? 'medium' : 'low'; }

  return { rng, buildSeedTable, mutate, scoreCandidate, topK, run, makeContext,
           sigmasFromBundle, pickFinal, mergeStats, confidenceLabel,
           EXPLORE_STRENGTH, REFINE_STRENGTH, VALIDITY_FLOOR };
})();

if(typeof module !== 'undefined') module.exports = SynthSearch;
