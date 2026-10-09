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

  // ── Derived metrics (v0.5.0) ─────────────────────────────────────────────
  // Computed from the forest's predictions, never trained on their own:
  //   stiff_main   stiffness along the design's stiffest axis, whichever it is
  //   stiff_ratio  the other two axes' mean stiffness as a percent of it
  //                (100 = the same every way, low = one-axis)
  //   porosity     100 − volume fraction (percent open)
  // Spreads (for confidence) come from the axes they are built from.
  const DERIVED_METRICS = ['stiff_main', 'stiff_ratio', 'porosity'];
  const AXES = ['ex_norm', 'ey_norm', 'ez_norm'];
  function mainAxis(pred){
    let m = 0;
    for(let i = 1; i < 3; i++) if(pred[AXES[i]] > pred[AXES[m]]) m = i;
    return m;
  }
  function addDerived(pred, spread){
    if(pred.ex_norm != null && pred.ey_norm != null && pred.ez_norm != null){
      const m = mainAxis(pred), o1 = AXES[(m + 1) % 3], o2 = AXES[(m + 2) % 3];
      const main = pred[AXES[m]];
      pred.stiff_main = main;
      pred.stiff_ratio = main > 1e-6 ? 100 * Math.max(0, (pred[o1] + pred[o2]) / 2) / main : 100;
      if(spread){ spread.stiff_main = spread[AXES[m]]; spread.stiff_ratio = (spread[o1] + spread[o2]) / 2; }
    }
    if(pred.volume_fraction != null){
      pred.porosity = 100 - pred.volume_fraction;
      if(spread && spread.volume_fraction != null) spread.porosity = spread.volume_fraction;
    }
    return pred;
  }
  // Residual sigma for derived metrics. The ratio's depends on where the
  // target is (a ratio of two soft axes is loose), so the pads also send a
  // per-search value (req.sigmas, from ratioSigma); this is the fallback.
  function derivedSigmas(sig){
    if(sig.ex_norm && sig.ey_norm && sig.ez_norm){
      sig.stiff_main = (sig.ex_norm + sig.ey_norm + sig.ez_norm) / 3;
      sig.stiff_ratio = 25;
    }
    if(sig.volume_fraction) sig.porosity = sig.volume_fraction;
    return sig;
  }
  // First-order error of off/main at a target (ratio in percent).
  function ratioSigma(sigAxis, mainTarget, ratioTarget){
    const r = ratioTarget / 100, m = Math.max(mainTarget, 1e-3);
    return Math.max(3, Math.min(60, 100 * sigAxis * Math.sqrt(0.5 + r * r) / m));
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
      addDerived(r.pred, r.spread);
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
    addDerived(pred, spread);
    if(req.connectivity != null && pred.directionality != null){
      const v = pred.directionality;
      const bucket = v < 0.5 ? 1 : (v < 0.833 ? 2 : 3);
      stats.buckets[bucket]++;
      if(bucket !== req.connectivity){ stats.connectivity++; return null; }
    }
    const m = match(ctx, req, pred, spread);
    return finish(d, validity, pred, spread, m.zPerMetric, m.zRms, m.score, m.spreadRatio);
  }
  function match(ctx, req, pred, spread){
    let z2 = 0, tw = 0, sr = 0, sw = 0;
    const zPerMetric = {};
    for(const m of ctx.scoreMetrics){
      const w = req.weights[m] || 0;
      if(!w || req.targets[m] == null) continue;
      const z = (pred[m] - req.targets[m]) / ((req.sigmas && req.sigmas[m]) || ctx.sigmas[m]);
      zPerMetric[m] = z;
      const zc = Math.min(Math.abs(z), 3);
      z2 += zc * zc * w; tw += w;
      if(spread[m] != null && ctx.spreadRef[m]){ sr += (spread[m] / ctx.spreadRef[m]) * w; sw += w; }
    }
    const zRms = tw > 0 ? Math.sqrt(z2 / tw) : 0;
    return { zPerMetric, zRms, score: Math.exp(-(zRms * zRms) / 6), spreadRatio: sw > 0 ? sr / sw : 1 };
  }

  // Shape check (Matt, 2026-10-09). The solid fraction measured from the
  // shape itself is exact; the model's is a guess. A gap past 2σ means the
  // forest is wrong about this design, so its other predictions are suspect
  // too: confidence drops one level (to low past 3σ) and the rank takes the
  // same cut a low-confidence design gets. The measured value replaces the
  // prediction, so a volume-fraction target is scored on the real shape.
  const SHAPE_CHECK = { warnZ: 2, badZ: 3, cut: [1, 0.8, 0.56] };
  const CONF_LEVELS = ['high', 'medium', 'low'];
  function shapeCheck(ctx, req, c, measuredVF){
    const predicted = c.pred.volume_fraction, sig = ctx.sigmas.volume_fraction;
    if(measuredVF == null || !isFinite(measuredVF) || predicted == null || !sig) return c;
    const gapZ = (measuredVF - predicted) / sig, a = Math.abs(gapZ);
    const level = a > SHAPE_CHECK.badZ ? 2 : a > SHAPE_CHECK.warnZ ? 1 : 0;
    const pred = addDerived(Object.assign({}, c.pred, { volume_fraction: measuredVF }), null);
    const m = match(ctx, req, pred, c.spread);
    const out = finish(c.design, c.validity, pred, c.spread, m.zPerMetric, m.zRms, m.score, c.spreadRatio);
    const base = CONF_LEVELS.indexOf(confidenceLabel(c.spreadRatio));
    const now = level === 0 ? base : level === 1 ? Math.min(2, base + 1) : 2;
    out.rank *= now === base ? 1 : now === 2 ? SHAPE_CHECK.cut[2] : SHAPE_CHECK.cut[1];
    out.confidence = CONF_LEVELS[now];
    out.shape = { measured: measuredVF, predicted, gapZ, level };
    out.seedIndex = c.seedIndex; out.presetKey = c.presetKey;
    return out;
  }

  // Rank = match × √validity × confidence factor. Wider searches drift toward
  // designs where the forest disagrees with itself (likely model error), so
  // the factor is flat inside the training spread and falls off steeply past
  // it, with an extra cut for low confidence (Matt, 2026-10-09).
  function confidenceFactor(ratio){
    let f = ratio <= 1.25 ? 1 : 1 / (1 + 0.8 * (ratio - 1.25));
    if(ratio > 2) f *= 0.7;
    return f;
  }
  function finish(d, validity, pred, spread, zPerMetric, zRms, score, spreadRatio){
    const conf = confidenceFactor(spreadRatio);
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

  // ── One job (a worker's share of a round) ─────────────────────────────────
  // req:
  //   targets, weights, connectivity
  //   parents   [{design, seedIndex, seed?}]  seed:true → also score it unchanged
  //   count     designs to grow from the parents
  //   strength  mutation step (fraction of each parameter's training range)
  //   grid      {kx, ky, x0, x1, y0, y1, n}  map cells kept for diversity
  //   ptsKeys   metrics to report for every scored design (the result map)
  //   keep      best-by-rank to return
  // Returns the best `keep`, the best design per map cell, the map points
  // (ptsKeys values per scored design, flat Float32Array) and counts.
  function run(ctx, req){
    const R = rng(req.rngSeed || 1);
    const stats = { scanned: 0, validity: 0, degenerate: 0, connectivity: 0, buckets: { 1: 0, 2: 0, 3: 0 } };
    // '_match' and '_conf' report the design's overall match and tree-spread
    // ratio, so the map can show how good each point is on every target.
    const keys = req.ptsKeys || [], nk = keys.length;
    const src = keys.map(k => k === '_match' ? 1 : k === '_conf' ? 2 : 0);
    const pts = new Float32Array((req.count + req.parents.length) * nk);
    let np = 0;
    const out = [], cells = new Map(), g = req.grid;
    const push = (c, seedIndex) => {
      if(!c) return;
      c.seedIndex = seedIndex; c.presetKey = ctx.seeds[seedIndex].presetKey;
      out.push(c);
      for(let k = 0; k < nk; k++) pts[np * nk + k] = src[k] === 1 ? c.score : src[k] === 2 ? c.spreadRatio : c.pred[keys[k]];
      np++;
      if(g){
        const gx = Math.floor((c.pred[g.kx] - g.x0) / (g.x1 - g.x0) * g.n);
        const gy = Math.floor((c.pred[g.ky] - g.y0) / (g.y1 - g.y0) * g.n);
        if(gx >= 0 && gx < g.n && gy >= 0 && gy < g.n){
          const cell = gy * g.n + gx, o = cells.get(cell);
          if(!o || c.rank > o.rank) cells.set(cell, c);
        }
      }
    };
    const P = req.parents;
    if(!P.length) return { candidates: [], cells: [], pts: new Float32Array(0), stats };
    for(const p of P){
      if(!p.seed) continue;
      stats.scanned++;
      push(scoreCandidate(ctx, req, SE.cloneDesign(p.design), stats), p.seedIndex);
    }
    for(let i = 0; i < req.count; i++){
      stats.scanned++;
      const p = P[Math.floor(R() * P.length)];
      push(scoreCandidate(ctx, req, mutate(p.design, R, req.strength, ctx.ranges[p.design.mode] || {}), stats), p.seedIndex);
    }
    return { candidates: topK(out, req.keep || 48), cells: [...cells.entries()],
             pts: pts.slice(0, np * nk), stats };
  }

  // ── Coordinator: rounds until the time budget, a plateau or Stop ──────────
  // Each round grows designs from (a) the best found so far and (b) the best
  // design in every filled cell of the map, so it both closes in on the
  // target and keeps alternatives spread across the map. The mutation step
  // starts wide and narrows each round. Runs anywhere: opts.runJobs does the
  // work (a worker pool, or the main thread), opts.onRound reports progress.
  //   opts: {ctx, req, seconds, workers, perWorker, plateauRounds, runJobs, onRound, shouldStop, now, rand}
  const ROUND = { start: 0.12, floor: 0.025, decay: 0.78, parents: 48, eliteKeep: 64,
                  plateauRounds: 3, plateauGain: 0.005, minRounds: 4 };
  async function coordinate(opts){
    const { ctx, req } = opts;
    const now = opts.now || (() => Date.now());
    const rand = opts.rand || Math.random;
    const t0 = now(), budget = opts.seconds * 1000, W = Math.max(1, opts.workers || 1);
    const pool = req.presetKey ? ctx.seeds.filter(s => s.presetKey === req.presetKey) : ctx.seeds;
    if(!pool.length) return { final: [], stats: null, rounds: 0, scanned: 0, reason: 'empty-preset' };
    let elites = [], stats = null, history = [], round = 0, reason = 'time';
    const archive = new Map();
    let parents = pool.map(s => ({ design: s.design, seedIndex: s.index, seed: true }));
    while(true){
      const strength = Math.max(ROUND.floor, ROUND.start * Math.pow(ROUND.decay, round));
      const jobs = [];
      for(let j = 0; j < W; j++){
        // Seeds are scored unchanged once, split across the jobs.
        const mine = round === 0 ? parents.map((p, i) => i % W === j ? p : Object.assign({}, p, { seed: false })) : parents;
        jobs.push(Object.assign({}, req, { parents: mine, count: opts.perWorker, strength,
          rngSeed: ((rand() * 2147483647) | 0) + 1, keep: 48 }));
      }
      const results = await opts.runJobs(jobs);
      let fresh = [], pts = [];
      for(const r of results){
        fresh = fresh.concat(r.candidates);
        for(const [cell, c] of r.cells){ const o = archive.get(cell); if(!o || c.rank > o.rank) archive.set(cell, c); }
        pts.push(r.pts);
        stats = mergeStats(stats, r.stats);
      }
      elites = topK(elites.concat(fresh), ROUND.eliteKeep);
      const final = pickFinal(elites.concat([...archive.values()]), 8, 2, ctx.sigmas, ctx.ranges);
      const mean = final.length ? final.reduce((a, c) => a + c.score, 0) / final.length : 0;
      history.push(mean);
      round++;
      const elapsed = now() - t0;
      if(opts.onRound) opts.onRound({ round, elapsed, scanned: stats.scanned, final, mean, pts, strength, cells: archive.size });
      if(opts.shouldStop && opts.shouldStop()){ reason = 'stopped'; break; }
      if(elapsed >= budget){ reason = 'time'; break; }
      const patience = opts.plateauRounds || ROUND.plateauRounds;
      if(round >= ROUND.minRounds && strength <= ROUND.floor + 1e-9 && history.length > patience){
        const before = history[history.length - 1 - patience];
        if(mean - before < ROUND.plateauGain){ reason = 'settled'; break; }
      }
      // Next parents: half from the best so far, half from the map's cells.
      const cellsArr = [...archive.values()];
      const best = elites.slice(0, Math.ceil(ROUND.parents / 2));
      const spread = [];
      for(let i = 0; i < ROUND.parents - best.length && cellsArr.length; i++) spread.push(cellsArr[Math.floor(rand() * cellsArr.length)]);
      parents = best.concat(spread).map(c => ({ design: c.design, seedIndex: c.seedIndex }));
    }
    const all = elites.concat([...archive.values()]);
    const final = pickFinal(all, 8, 2, ctx.sigmas, ctx.ranges);
    // A longer list for the shape check, so a design it rules out has a replacement.
    const shortlist = pickFinal(all, 16, 2, ctx.sigmas, ctx.ranges);
    return { final, shortlist, stats, rounds: round, scanned: stats ? stats.scanned : 0, reason, seconds: (now() - t0) / 1000 };
  }

  // Context from the init message. init.model is SynthForest.toMessage()
  // output; fromMessage() only rebuilds views, so on the main thread this
  // shares the same buffers rather than copying them.
  function makeContext(init){
    const enc = SE.create(init.encoding, init.featureDim);
    const model = SF.fromMessage(init.model);
    const sigmas = derivedSigmas(Object.assign({}, init.sigmas));
    const scoreMetrics = init.outputMetrics.concat(DERIVED_METRICS.filter(k => sigmas[k]));
    return { enc, model, outputMetrics: init.outputMetrics, scoreMetrics, sigmas,
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
    return derivedSigmas(out);
  }

  // Final list: best first, at most `perSeed` results grown from one seed so
  // eight cards are not eight near-copies of the same design. With sigmas,
  // a further result from the same seed is kept only if some prediction
  // differs from that seed's earlier picks by NEAR_COPY_Z sigma or more.
  // Two designs from one seed also count as near-copies when their one-cell
  // shape barely differs: same term structure, and every continuous setting
  // within NEAR_COPY_SHAPE (20%) of its training range. cell_scale is left out (it
  // only sets how many repeats fit in the cell; the preview shows one).
  const NEAR_COPY_Z = 0.5, NEAR_COPY_SHAPE = 0.2;
  function shapeDistance(a, b, range){
    if(a.mode !== b.mode || a.terms.length !== b.terms.length) return Infinity;
    range = range || {};
    const span = k => { const r = range[k]; return r ? Math.max(r[1] - r[0], 0.02) : 1; };
    const ang = (p, q) => { const d = Math.abs(((p - q) % (2 * Math.PI) + 3 * Math.PI) % (2 * Math.PI) - Math.PI); return d / Math.PI; };
    let m = Math.max(Math.abs((a.wall_thickness || 0) - (b.wall_thickness || 0)) / span('wall_thickness'),
                     Math.abs((a.offset || 0) - (b.offset || 0)) / span('offset'),
                     Math.abs((a.pipe_radius || 0) - (b.pipe_radius || 0)) / span('pipe_radius'));
    if(a.normal_weights && b.normal_weights) for(const k of ['wx', 'wy', 'wz'])
      m = Math.max(m, Math.abs(a.normal_weights[k] - b.normal_weights[k]) / span('nw'));
    if(a.phase_shift && b.phase_shift) for(const k of ['x', 'y', 'z'])
      m = Math.max(m, Math.abs(a.phase_shift[k] - b.phase_shift[k]));
    for(let i = 0; i < a.terms.length; i++){
      const s = a.terms[i], t = b.terms[i];
      if(s.factors.length !== t.factors.length) return Infinity;
      for(let j = 0; j < s.factors.length; j++){
        const f = s.factors[j], g = t.factors[j];
        if(f.trig !== g.trig || f.fx !== g.fx || f.fy !== g.fy || f.fz !== g.fz) return Infinity;
      }
      m = Math.max(m, Math.abs(s.coef - t.coef) / 2);
      if(s.phase_shift && t.phase_shift) for(const k of ['x', 'y', 'z']) m = Math.max(m, ang(s.phase_shift[k], t.phase_shift[k]));
    }
    return m;
  }
  function pickFinal(cands, n, perSeed, sigmas, ranges){
    const ranked = topK(cands, cands.length), bySeed = {}, out = [], skipped = [], copies = [];
    const metrics = sigmas ? Object.keys(sigmas).filter(k => DERIVED_METRICS.indexOf(k) < 0) : [];
    const nearCopy = (a, b) => shapeDistance(a.design, b.design, ranges && ranges[a.design.mode]) < NEAR_COPY_SHAPE ||
      metrics.every(k => a.pred[k] == null || b.pred[k] == null || Math.abs(a.pred[k] - b.pred[k]) < NEAR_COPY_Z * sigmas[k]);
    for(const c of ranked){
      const mine = bySeed[c.seedIndex] || (bySeed[c.seedIndex] = []);
      if(mine.length >= perSeed){ skipped.push(c); continue; }
      if(sigmas && mine.some(o => nearCopy(o, c))){ copies.push(c); continue; }
      mine.push(c); out.push(c);
      if(out.length >= n) break;
    }
    // Few distinct seeds (a narrow preset filter): top up with the best of the rest.
    for(const c of skipped.concat(copies)){ if(out.length >= n) break; out.push(c); }
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

  return { rng, buildSeedTable, mutate, scoreCandidate, topK, run, coordinate, makeContext,
           sigmasFromBundle, pickFinal, mergeStats, confidenceLabel, confidenceFactor, shapeCheck, SHAPE_CHECK,
           addDerived, mainAxis, ratioSigma, shapeDistance, DERIVED_METRICS,
           EXPLORE_STRENGTH, REFINE_STRENGTH, VALIDITY_FLOOR, ROUND };
})();

if(typeof module !== 'undefined') module.exports = SynthSearch;
