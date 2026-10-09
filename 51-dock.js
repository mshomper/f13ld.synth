/* ============================================================
   F13LD.synth · 51-dock.js
   Dock: Configure (drawer), tags, the status chip (F13LD.lab's "Field"
   mark ripples while searching), the depth toggle and Synthesize / Stop.
   runSearch() drives a search and feeds the map as rounds come back.
   ============================================================ */
'use strict';

// ── Configure drawer ──
function toggleDrawer(force){
  const d = document.getElementById('drawer'), b = document.getElementById('cfgBtn');
  const open = force != null ? force : d.hidden;
  d.hidden = !open; b.setAttribute('aria-expanded', open ? 'true' : 'false');
  if(open){ syncMaterialCard(); renderModelDrawer(); }
}
function wireDrawer(){
  document.getElementById('cfgBtn').addEventListener('click', () => toggleDrawer());
  document.addEventListener('keydown', e => { if(e.key === 'Escape') toggleDrawer(false); });
  document.querySelector('.main').addEventListener('pointerdown', () => toggleDrawer(false));
}

// ── Tags ──
function renderDockTags(){
  const m = materialById(globalInputs.material_id);
  const E = globalInputs.modulus_gpa, cs = globalInputs.cell_size_mm;
  document.getElementById('dockTags').innerHTML =
    `<button type="button" class="dtag" data-open title="Material — change in Configure">${icon2('material')}<b>${materialShort(m)}</b>${E ? E + ' GPa' : ''}</button>` +
    `<button type="button" class="dtag" data-open title="Unit cell size — change in Configure">${icon2('cell')}<b>cell</b>${cs ? cs + ' mm' : '—'}</button>`;
  document.querySelectorAll('#dockTags [data-open]').forEach(b => b.addEventListener('click', () => toggleDrawer(true)));
}
function icon2(name){ return `<span class="ico ico-sm">${icon(name)}</span>`; }

// ── Depth ──
function buildDepthSeg(){
  try { const d = localStorage.getItem(DEPTH_STORAGE_KEY); if(SEARCH_DEPTHS[d]) SYNTH.depth = d; } catch(e){}
  const seg = document.getElementById('depthSeg');
  seg.innerHTML = Object.entries(SEARCH_DEPTHS).map(([k, d]) => `<button type="button" data-d="${k}" title="${d.tip}" class="${k === SYNTH.depth ? 'on' : ''}">${d.label}</button>`).join('');
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    SYNTH.depth = b.dataset.d;
    seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    try { localStorage.setItem(DEPTH_STORAGE_KEY, SYNTH.depth); } catch(e){}
  }));
}

// ── Status chip: the F13LD mark (F13LD.lab 52-status "Field") ──
const STATUS = { svg: null, raf: 0, t0: 0, reduce: false };
function statusInit(){
  const host = document.getElementById('stMark');
  host.innerHTML = '<svg viewBox="0 0 100 100" width="20" height="20" aria-hidden="true"><defs><clipPath id="stClip"><path d="M50,5 C91,5 95,9 95,50 C95,91 91,95 50,95 C9,95 5,91 5,50 C5,9 9,5 50,5Z"/></clipPath></defs>' +
    '<path d="M50,5 C91,5 95,9 95,50 C95,91 91,95 50,95 C9,95 5,91 5,50 C5,9 9,5 50,5Z" fill="#111e13" stroke="#1D9E75" stroke-width="5"/>' +
    '<g clip-path="url(#stClip)" fill="none"><path class="s1" d="M0,31 Q25,21 50,31 Q75,41 100,31" stroke="#2c4e30" stroke-width="3"/><path class="s2" d="M0,69 Q25,59 50,69 Q75,79 100,69" stroke="#2c4e30" stroke-width="3"/>' +
    '<path class="s3" d="M31,0 Q21,25 31,50 Q41,75 31,100" stroke="#2c4e30" stroke-width="3"/><path class="s4" d="M69,0 Q59,25 69,50 Q79,75 69,100" stroke="#2c4e30" stroke-width="3"/>' +
    '<path class="h" d="M0,50 Q25,40 50,50 Q75,60 100,50" stroke="#c8f542" stroke-width="6"/><path class="v" d="M50,0 Q40,25 50,50 Q60,75 50,100" stroke="#c8f542" stroke-width="6"/>' +
    '<circle class="node" cx="50" cy="50" r="7" fill="#c8f542"/></g><path d="M50,5 C91,5 95,9 95,50 C95,91 91,95 50,95 C9,95 5,91 5,50 C5,9 9,5 50,5Z" fill="none" stroke="#1D9E75" stroke-width="5"/></svg>';
  STATUS.svg = host.querySelector('svg');
  STATUS.reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}
function statusRipple(t){
  const svg = STATUS.svg; if(!svg) return;
  let a = 10 * Math.cos(t), b = 10 * Math.cos(t + 1.3), s = 10 * Math.cos(t * 0.6 + 0.7);
  if(t === 0){ a = 10; b = 10; s = 10; }
  const hz = (y, m) => `M0,${y} Q25,${y - m} 50,${y} Q75,${y + m} 100,${y}`, vt = (x, m) => `M${x},0 Q${x - m},25 ${x},50 Q${x + m},75 ${x},100`;
  const q = c => svg.querySelector(c);
  q('.h').setAttribute('d', hz(50, a)); q('.v').setAttribute('d', vt(50, b));
  q('.s1').setAttribute('d', hz(31, s)); q('.s2').setAttribute('d', hz(69, s)); q('.s3').setAttribute('d', vt(31, s)); q('.s4').setAttribute('d', vt(69, s));
  q('.node').setAttribute('r', t === 0 ? '7' : (7 + 1.6 * Math.sin(t * 2)).toFixed(2));
}
function statusSpin(on){
  cancelAnimationFrame(STATUS.raf); STATUS.raf = 0;
  if(!on || STATUS.reduce){ statusRipple(0); return; }
  STATUS.t0 = 0;
  const loop = ts => { if(!STATUS.t0) STATUS.t0 = ts; statusRipple((ts - STATUS.t0) / 520); STATUS.raf = requestAnimationFrame(loop); };
  STATUS.raf = requestAnimationFrame(loop);
}
function statusSet(state, html, progress){
  const st = document.getElementById('st');
  st.className = 'st ' + (state || '');
  document.getElementById('stTx').innerHTML = html;
  if(progress != null) document.getElementById('stBar').style.width = (Math.max(0, Math.min(1, progress)) * 100).toFixed(1) + '%';
}

// ── Run / Stop ──
function setRunButton(running){
  const b = document.getElementById('runBtn');
  b.classList.toggle('stop', running);
  b.innerHTML = glyph(running ? 'stop' : 'play') + `<span>${running ? 'Stop' : 'Synthesize'}</span>`;
  b.title = running ? 'Stop the search and keep what it has found' : 'Search for designs that match your intent';
}
function wireRun(){ document.getElementById('runBtn').addEventListener('click', runSearch); }

async function runSearch(){
  if(SYNTH.running){ SYNTH.stopFlag = true; statusSet('run', 'stopping…'); return; }
  if(!Predictor.loaded) return;
  const pad = firstActivePad();
  if(!pad){ statusSet('warn', 'Turn a pad to prefer or require first'); return; }
  toggleDrawer(false);
  const { targets, weights, sigmas } = buildTargetsFromPads();
  const req = {
    targets, weights, sigmas, connectivity: connectivityState, presetKey: presetState || null,
    grid: { kx: pad.xMetric, ky: pad.yMetric, x0: MAP_RANGES[pad.xMetric][0], x1: MAP_RANGES[pad.xMetric][1], y0: MAP_RANGES[pad.yMetric][0], y1: MAP_RANGES[pad.yMetric][1], n: 24 },
    ptsKeys: MAP_PTS_KEYS
  };
  SYNTH.running = true; SYNTH.stopFlag = false; SYNTH.results = []; SYNTH.sel = -1; SYNTH.intentDirty = false; SYNTH.emptyText = '';
  SYNTH.lastReq = { targets, weights, sigmas };
  setRunButton(true); statusSpin(true);
  mapReset(); mapSetTab(PAD_DEFS.indexOf(pad));
  renderStrip(); renderInspector(); renderHeaderPills();
  const depth = SEARCH_DEPTHS[SYNTH.depth];
  document.getElementById('nScored').textContent = 'searching…';
  statusSet('run', 'starting…', 0); document.getElementById('st').title = '';
  let res;
  try {
    res = await Predictor.inverseSearch(req, {
      depth: SYNTH.depth,
      shouldStop: () => SYNTH.stopFlag,
      onCheck: () => statusSet('run', 'checking the shapes of the best designs…', 1),
      onRound: info => {
        for(const p of info.pts) mapAddPoints(p);
        MAP.round = info.round;
        mapSetGoals(info.final.map(c => c.pred));
        const best = info.final.length ? Math.max(...info.final.map(c => c.score)) : 0;
        statusSet('run', `round ${info.round} · <b>${info.scanned.toLocaleString()}</b> designs · best ${Math.round(best * 100)}%`, info.elapsed / (depth.seconds * 1000));
        document.getElementById('nScored').textContent = `${info.scanned.toLocaleString()} scored`;
      }
    });
  } catch(e){
    console.error('[F13LD.synth] Search failed:', e);
    res = { results: [], reason: 'error', error: e };
  }
  SYNTH.running = false; setRunButton(false); statusSpin(false);
  SYNTH.results = res.results || [];
  if(SYNTH.results.length){
    SYNTH.sel = 0;
    mapSetGoals(SYNTH.results.map(r => r.metrics));
    const ended = { settled: 'settled', time: 'time limit', stopped: 'stopped' }[res.ended] || res.ended;
    const best = Math.max(...SYNTH.results.map(r => r.score));
    const off = SYNTH.results.filter(r => r.shape && r.shape.level).length;
    // Out of reach: no result within REACH_Z on every target it was given.
    const reach = SYNTH.results.some(r => Object.values(r.zPerMetric).every(z => Math.abs(z) <= REACH_Z));
    statusSet(reach ? 'done' : 'warn', `<b>${res.stats.scanned.toLocaleString()} designs</b> · ${res.rounds} rounds · ${res.seconds.toFixed(1)} s · ${ended} · best ${Math.round(best * 100)}%` +
      (off ? ` · ${off} shape${off > 1 ? 's' : ''} differ from the model` : '') +
      (reach ? '' : ' · <b>target out of reach</b>, closest shown'), 1);
    document.getElementById('st').title = reach ? '' : `No design the model knows gets within ${REACH_Z}σ on every target you set. The eight shown are the closest; try moving a target toward where the map is bright.`;
    document.getElementById('nScored').textContent = `8 of ${res.stats.scanned.toLocaleString()} scored`;
  } else {
    const why = res.reason === 'error' ? 'search failed — see the console'
      : res.reason === 'empty-preset' ? 'no training designs for that preset'
      : res.stats && res.stats.connectivity > res.stats.scanned * 0.9 ? 'nothing matched the connectivity filter'
      : 'no candidates passed the filters';
    SYNTH.emptyText = why.charAt(0).toUpperCase() + why.slice(1) + '.';
    statusSet('warn', why);
    document.getElementById('nScored').textContent = 'no results';
  }
  renderStrip(); renderInspector(); mapRender(); renderHeaderPills();
  if(SYNTH.results.length) renderThumbnails();
}

// Pads changed after a search: the map's target moves live; the results
// were scored against the old intent, so say so.
function onIntentChanged(live){
  mapRender();
  if(SYNTH.results.length && !SYNTH.running && !SYNTH.intentDirty && !live){
    SYNTH.intentDirty = true;
    statusSet('warn', 'intent changed · Synthesize to update');
  }
}
