/* ============================================================
   F13LD.synth · 20-predictor.js
   Loads a model bundle, builds the search context, runs searches on a
   pool of workers (main thread if workers are unavailable).
   ============================================================ */
'use strict';

const Predictor = {
  bundle: null, meta: null, metricsR2: null, sigmas: null, enc: null,
  seedTable: null, initMsg: null, ctx: null,
  loaded: false, loading: false, loadError: null,
  pool: null,

  async loadFamily(family){
    this.loaded = false; this.loading = true; this.loadError = null;
    const t0 = performance.now();
    try {
      const bundle = await fetchFirst(WEIGHTS_URLS(family));
      const enc = SynthEncoding.create(bundle.encoding, bundle.meta.feature_dim);
      const packed = SynthForest.packModel(bundle);
      const msg = SynthForest.toMessage(packed);
      const table = SynthSearch.buildSeedTable(bundle, enc, SynthForest.fromMessage(msg));
      this.sigmas = SynthSearch.sigmasFromBundle(bundle);
      this.initMsg = { encoding: bundle.encoding, featureDim: bundle.meta.feature_dim, model: msg,
        outputMetrics: bundle.output_metrics, sigmas: this.sigmas,
        seeds: table.seeds, ranges: table.ranges, spreadRef: table.spreadRef };
      this.ctx = SynthSearch.makeContext(this.initMsg);
      // The trees now live in packed buffers; drop the parsed JSON copies.
      bundle.metrics_model = null; bundle.validity_model = null;
      this.bundle = bundle; this.meta = bundle.meta; this.enc = enc; this.seedTable = table;
      this.metricsR2 = (bundle.eval && bundle.eval.metrics_test_r2) || {};
      this.loaded = true;
      console.info(`[F13LD.synth] Predictor loaded in ${((performance.now()-t0)/1000).toFixed(1)}s · ${bundle.meta.family} v${bundle.meta.version} · ${bundle.meta.n_valid} valid samples · ${table.seeds.length} seeds`);
      this.pool = SearchPool.create(this.initMsg);
    } catch(e){
      this.loadError = e.message || String(e);
      console.warn(`[F13LD.synth] No predictor for ${family}: ${this.loadError}`);
    } finally { this.loading = false; }
  },

  // Metrics this bundle predicts that Synth knows how to show.
  shownMetrics(){
    if(!this.bundle) return Object.keys(METRIC_DEFS);
    return this.bundle.output_metrics.filter(k => METRIC_DEFS[k]);
  },

  presetOptions(){
    if(!this.seedTable) return [];
    return Object.entries(this.seedTable.presets)
      .map(([key, n]) => ({ key, n, label: presetDisplayLabel(key) }))
      .sort((a, b) => b.n - a.n);
  },

  // req: {targets, weights, connectivity, presetKey, grid, ptsKeys}
  // opts: {depth: 'quick'|'wide'|'deep', onRound(info), shouldStop()}
  async inverseSearch(req, opts){
    if(!this.loaded) return { results: [], reason: 'no-model' };
    opts = opts || {};
    const depth = SEARCH_DEPTHS[opts.depth] || SEARCH_DEPTHS[DEFAULT_DEPTH];
    const W = this.pool ? this.pool.size : 1;
    const res = await SynthSearch.coordinate({
      ctx: this.ctx, req, seconds: depth.seconds, workers: W,
      perWorker: this.pool ? SEARCH_PER_WORKER : SEARCH_PER_WORKER_MAIN, plateauRounds: depth.patience,
      runJobs: jobs => this.runJobs(jobs),
      onRound: opts.onRound, shouldStop: opts.shouldStop,
      now: () => performance.now()
    });
    if(res.final.length){
      const top = res.final[0];
      const zStr = Object.entries(top.zPerMetric).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .map(([k, z]) => `${k}: ${z >= 0 ? '+' : ''}${z.toFixed(2)}σ`).join('  ·  ');
      console.info(`[F13LD.synth] Top candidate · match ${(top.score*100).toFixed(0)}% · zRMS ${top.zRms.toFixed(2)} · ${zStr}`);
    }
    if(res.stats){
      const st = res.stats;
      console.info(`[F13LD.synth] Search (${depth.label}): ${st.scanned.toLocaleString()} designs in ${res.rounds} rounds on ${W} ${this.pool ? 'workers' : 'thread (main)'} · ${res.seconds.toFixed(1)}s · ended: ${res.reason} · ${st.validity} validity-rejected · ${st.degenerate} degenerate · ${st.connectivity} off-connectivity`);
    }
    return { results: res.final.map(c => this.toResult(c)), reason: res.final.length ? 'ok' : (res.reason === 'empty-preset' ? 'empty-preset' : 'none'),
             stats: res.stats, rounds: res.rounds, seconds: res.seconds, ended: res.reason, workers: this.pool ? W : 0 };
  },

  async runJobs(jobs){
    if(this.pool){
      try { return await this.pool.runAll(jobs); }
      catch(e){ console.warn('[F13LD.synth] Worker search failed, falling back to the main thread:', e.message); this.pool.terminate(); this.pool = null; }
    }
    // Main-thread fallback: one job at a time, yielding between them.
    const out = [];
    for(const job of jobs){
      out.push(SynthSearch.run(this.ctx, job));
      await new Promise(r => setTimeout(r, 0));
    }
    return out;
  },

  toResult(c){
    const conf = SynthSearch.confidenceLabel(c.spreadRatio);
    const presetKey = c.presetKey;
    const recipe = SynthEncoding.toRecipe(c.design, {
      presetKey: presetKey.indexOf('|') < 0 ? presetKey : null,
      presetLabel: presetDisplayLabel(presetKey),
      synth: {
        tool_version: F13LD_SYNTH_VERSION,
        model: { family: this.meta.family, version: this.meta.version, trained_at: this.meta.trained_at, n_valid: this.meta.n_valid },
        seed_index: c.seedIndex, seed_preset: presetKey,
        score: +c.score.toFixed(4), validity: +c.validity.toFixed(4), confidence: conf,
        predicted: Object.fromEntries(Object.entries(c.pred).map(([k, v]) => [k, +v.toFixed(5)]))
      }
    });
    return { metrics: c.pred, spread: c.spread, zRms: c.zRms, score: c.score, validity: c.validity,
             zPerMetric: c.zPerMetric, spreadRatio: c.spreadRatio, confidence: conf,
             presetKey, seedIndex: c.seedIndex, design: c.design, recipe };
  }
};

function presetDisplayLabel(key){
  if(/^custom-\d+$/.test(key)) return 'custom set ' + key.split('-')[1];
  return SynthEncoding.presetLabel(key);
}

async function fetchFirst(urls){
  let lastErr = null;
  for(const url of urls){
    try {
      const res = await fetch(url, { cache: 'default' });
      if(!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
      return await res.json();
    } catch(e){ lastErr = e; }
  }
  throw lastErr || new Error('no model URL');
}

// ── Worker pool ─────────────────────────────────────────────────────────────
const SearchPool = {
  create(initMsg){
    if(typeof Worker === 'undefined') return null;
    const cores = navigator.hardwareConcurrency || 4;
    const phone = window.matchMedia && window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    const size = Math.max(1, Math.min(phone ? 2 : 15, cores - 1));   // all threads but one
    const workers = [];
    try {
      for(let i = 0; i < size; i++) workers.push(makeSearchWorker());
    } catch(e){
      console.info('[F13LD.synth] Workers unavailable, searching on the main thread:', e.message);
      workers.forEach(w => w.terminate());
      return null;
    }
    let nextId = 1;
    const pending = new Map();
    const ready = workers.map(w => new Promise((resolve, reject) => {
      w.onmessage = e => {
        const m = e.data;
        if(m.type === 'ready') return resolve();
        const p = pending.get(m.id);
        if(!p) return;
        pending.delete(m.id);
        if(m.type === 'error') p.reject(new Error(m.message)); else p.resolve(m.result);
      };
      w.onerror = e => { reject(new Error(e.message || 'worker error')); for(const p of pending.values()) p.reject(new Error(e.message || 'worker error')); pending.clear(); };
      w.postMessage({ type: 'init', init: initMsg });
    }));
    const allReady = Promise.all(ready);
    console.info(`[F13LD.synth] Search pool: ${size} workers`);
    return {
      size,
      async runAll(jobs){
        await allReady;
        return Promise.all(jobs.map((req, i) => new Promise((resolve, reject) => {
          const id = nextId++;
          pending.set(id, { resolve, reject });
          workers[i % size].postMessage({ type: 'run', id, req });
        })));
      },
      terminate(){ workers.forEach(w => w.terminate()); }
    };
  }
};

// The single-file preview build sets SYNTH_WORKER_SOURCE so the worker can
// start from a Blob; the site loads worker/search-worker.js normally.
function makeSearchWorker(){
  if(typeof SYNTH_WORKER_SOURCE === 'string'){
    const url = URL.createObjectURL(new Blob([SYNTH_WORKER_SOURCE], { type: 'text/javascript' }));
    return new Worker(url);
  }
  return new Worker('worker/search-worker.js');
}
