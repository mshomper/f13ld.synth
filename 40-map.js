/* ============================================================
   F13LD.synth · 40-map.js
   The result map — and the search's loading display. Every scored design
   lands as a faint point as its round comes back; the eight best are
   numbered rings that glide to where the latest round put them; the
   dashed ring is 1σ (the model's residual) around the pad's target.
   A point's brightness is its match on every target, including the ones
   the two axes cannot show; low-confidence points are drawn faint.
   Tabs switch the axes between the three pads. Click a ring to select.

   Points are drawn once into an offscreen layer as they arrive, so a
   frame only composites the layer and the rings.
   ============================================================ */
'use strict';

// The metrics every scored design reports for the map (12-search-core ptsKeys),
// plus its overall match and tree-spread ratio.
const MAP_KEYS = ['ex_norm', 'pore_size_p50_norm', 'stiff_main', 'stiff_ratio', 'porosity', 'pore_size_cv'];
const MAP_PTS_KEYS = MAP_KEYS.concat(['_match', '_conf']);
const MAP_TAB_LABELS = { mech_pore: 'Stiffness × Pore', direction: 'Main × Off-axis', porosity_pores: 'Porosity × Pore spread' };

const MAP = {
  tab: 0, pts: new Float32Array(0), n: 0, layer: null, dirtyFrom: 0,
  rings: [], goals: [], anim: 0, round: 0, legendExtra: ''
};

function mapReset(){
  MAP.n = 0; MAP.dirtyFrom = 0; MAP.rings = []; MAP.goals = []; MAP.round = 0; MAP.legendExtra = '';
  if(MAP.layer) MAP.layer.getContext('2d').clearRect(0, 0, MAP.layer.width, MAP.layer.height);
  mapRender();
}
function mapAddPoints(chunk){
  if(!chunk || !chunk.length) return;
  const nk = MAP_PTS_KEYS.length, need = MAP.n * nk + chunk.length;
  if(need > MAP.pts.length){
    const np = new Float32Array(Math.max(need, MAP.pts.length * 2, nk * 4096));
    np.set(MAP.pts.subarray(0, MAP.n * nk)); MAP.pts = np;
  }
  MAP.pts.set(chunk, MAP.n * nk);
  if(MAP.dirtyFrom == null) MAP.dirtyFrom = MAP.n;
  MAP.n += chunk.length / nk;
}
// Ring targets: one per result, in best-match order. Rings glide there.
function mapSetGoals(preds, instant){
  MAP.goals = preds.map(p => p);
  while(MAP.rings.length < MAP.goals.length){
    const g = MAP.goals[MAP.rings.length];
    MAP.rings.push(Object.assign({}, g));
  }
  MAP.rings.length = MAP.goals.length;
  if(instant){ MAP.rings = MAP.goals.map(g => Object.assign({}, g)); mapRender(); return; }
  if(!MAP.anim) MAP.anim = requestAnimationFrame(mapStep);
}
function mapStep(){
  MAP.anim = 0;
  let moving = false;
  MAP.rings.forEach((r, i) => {
    const g = MAP.goals[i]; if(!g) return;
    for(const k of MAP_KEYS){
      const d = g[k] - r[k];
      if(Math.abs(d) > 1e-5){ r[k] += d * 0.16; moving = true; } else r[k] = g[k];
    }
  });
  mapRender();
  if(moving) MAP.anim = requestAnimationFrame(mapStep);
}

function mapAxes(){
  const def = PAD_DEFS[MAP.tab];
  const kx = def.xMetric, ky = def.yMetric;
  return { def, kx, ky, xi: MAP_PTS_KEYS.indexOf(kx), yi: MAP_PTS_KEYS.indexOf(ky), xr: MAP_RANGES[kx], yr: MAP_RANGES[ky] };
}
function mapGeom(cv){
  const w = cv.clientWidth, h = cv.clientHeight, pl = 52, pr = 18, pt = 38, pb = 44;
  const A = mapAxes();
  const cl = (v, r) => Math.max(r[0], Math.min(r[1], v));
  return { w, h, pl, pr, pt, pb, A,
    sx: v => pl + (cl(v, A.xr) - A.xr[0]) / (A.xr[1] - A.xr[0]) * (w - pl - pr),
    sy: v => h - pb - (cl(v, A.yr) - A.yr[0]) / (A.yr[1] - A.yr[0]) * (h - pt - pb) };
}

function mapRender(){
  const cv = document.getElementById('mapCv');
  if(!cv || !cv.clientWidth || !cv.clientHeight) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2), G = mapGeom(cv);
  const W = Math.round(G.w * dpr), H = Math.round(G.h * dpr);
  if(cv.width !== W || cv.height !== H){ cv.width = W; cv.height = H; }
  const x = cv.getContext('2d');
  x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, G.w, G.h);
  const pw = G.w - G.pl - G.pr, ph = G.h - G.pt - G.pb;
  x.fillStyle = '#080b12'; x.fillRect(G.pl, G.pt, pw, ph);
  x.strokeStyle = 'rgba(255,255,255,.07)'; x.lineWidth = 1; x.strokeRect(G.pl + .5, G.pt + .5, pw - 1, ph - 1);

  // point layer (rebuilt on resize or tab change)
  if(!MAP.layer || MAP.layer.width !== W || MAP.layer.height !== H || MAP.layerTab !== MAP.tab){
    MAP.layer = MAP.layer || document.createElement('canvas');
    MAP.layer.width = W; MAP.layer.height = H; MAP.layerTab = MAP.tab; MAP.dirtyFrom = 0;
  }
  if(MAP.dirtyFrom != null){
    // Brightness = match on every target (score³ so 1σ reads bright and 2σ
    // faint), cut to a third for low confidence. Points are bucketed so each
    // bucket is one fillStyle.
    const L = MAP.layer.getContext('2d'), nk = MAP_PTS_KEYS.length, mi = MAP_PTS_KEYS.indexOf('_match'), ci = MAP_PTS_KEYS.indexOf('_conf');
    L.setTransform(dpr, 0, 0, dpr, 0, 0);
    if(MAP.dirtyFrom === 0) L.clearRect(0, 0, G.w, G.h);
    L.globalCompositeOperation = 'lighter';
    const base = 0.34 * Math.min(1, (pw * ph) / (620 * 420)), NB = 8, buckets = Array.from({ length: NB }, () => []);
    for(let i = MAP.dirtyFrom; i < MAP.n; i++){
      const sc = MAP.pts[i * nk + mi], ratio = MAP.pts[i * nk + ci];
      const b = 0.1 + 0.9 * sc * sc * sc * (ratio > 2 ? 0.33 : 1);
      buckets[Math.min(NB - 1, Math.floor(b * NB))].push(i);
    }
    buckets.forEach((list, k) => {
      if(!list.length) return;
      L.fillStyle = `rgba(79,184,201,${(base * (k + 0.5) / NB).toFixed(4)})`;
      for(const i of list) L.fillRect(G.sx(MAP.pts[i * nk + G.A.xi]) - 1, G.sy(MAP.pts[i * nk + G.A.yi]) - 1, 2.2, 2.2);
    });
    MAP.dirtyFrom = null;
  }
  x.setTransform(1, 0, 0, 1, 0, 0); x.drawImage(MAP.layer, 0, 0); x.setTransform(dpr, 0, 0, dpr, 0, 0);

  // target + 1σ ring (only when this pad is on)
  const def = G.A.def;
  if(padState[def.id] && padState[def.id].mode !== 'off'){
    const tvx = padPosToMetricValue(def, 'x'), tvy = padPosToMetricValue(def, 'y');
    const sg = Object.assign({}, Predictor.sigmas || {}, buildTargetsFromPads().sigmas);
    const tx = G.sx(tvx), ty = G.sy(tvy);
    const rx = sg[G.A.kx] ? Math.abs(G.sx(tvx + sg[G.A.kx]) - tx) : 0, ry = sg[G.A.ky] ? Math.abs(G.sy(tvy + sg[G.A.ky]) - ty) : 0;
    const col = padState[def.id].mode === 'require' ? '255,182,112' : '200,245,66';
    if(rx > 1 && ry > 1){ x.setLineDash([3, 4]); x.strokeStyle = `rgba(${col},.65)`; x.lineWidth = 1.2; x.beginPath(); x.ellipse(tx, ty, rx, ry, 0, 0, 6.2832); x.stroke(); x.setLineDash([]); }
    x.strokeStyle = `rgb(${col})`; x.lineWidth = 2; x.beginPath(); x.arc(tx, ty, 6, 0, 6.2832); x.stroke();
  }

  // the eight
  x.font = '600 11px "IBM Plex Mono", monospace'; x.textAlign = 'center'; x.textBaseline = 'middle';
  const order = MAP.rings.map((_, i) => i).filter(i => i !== SYNTH.sel).concat(SYNTH.sel >= 0 && SYNTH.sel < MAP.rings.length ? [SYNTH.sel] : []);
  order.forEach(i => {
    const q = MAP.rings[i];
    const on = i === SYNTH.sel && !SYNTH.running, X = G.sx(q[G.A.kx]), Y = G.sy(q[G.A.ky]), r = on ? 11 : 9;
    x.beginPath(); x.arc(X, Y, r, 0, 6.2832); x.fillStyle = on ? '#c8f542' : '#0c1a24'; x.fill();
    x.strokeStyle = on ? '#c8f542' : '#4FB8C9'; x.lineWidth = 1.2; x.stroke();
    x.fillStyle = on ? '#0a1206' : '#7BD3E3'; x.fillText(String(i + 1), X, Y + .5);
  });

  // axes: names + range ends in display units
  const end = (key, v) => { const r = displayMetric(key, v, 2); return r.v + (r.unit && r.unit !== 'norm' ? ' ' + r.unit : ''); };
  x.font = '13px "IBM Plex Mono", monospace'; x.fillStyle = '#7BD3E3'; x.textAlign = 'center';
  x.fillText(def.xLabel.toLowerCase() + ' →', G.pl + pw / 2, G.h - 14);
  x.save(); x.translate(16, G.pt + ph / 2); x.rotate(-Math.PI / 2); x.fillStyle = '#FFB670'; x.fillText(def.yLabel.toLowerCase() + ' →', 0, 0); x.restore();
  x.font = '10.5px "IBM Plex Mono", monospace'; x.fillStyle = '#5f6582';
  x.textAlign = 'left'; x.fillText(end(G.A.kx, G.A.xr[0]), G.pl, G.h - pb2(G));
  x.textAlign = 'right'; x.fillText(end(G.A.kx, G.A.xr[1]), G.pl + pw, G.h - pb2(G));
  x.save(); x.translate(G.pl - 8, G.h - G.pb); x.rotate(-Math.PI / 2); x.textAlign = 'left'; x.fillText(end(G.A.ky, G.A.yr[0]), 0, 0); x.restore();
  x.save(); x.translate(G.pl - 8, G.pt); x.rotate(-Math.PI / 2); x.textAlign = 'right'; x.fillText(end(G.A.ky, G.A.yr[1]), 0, 0); x.restore();

  const lg = document.getElementById('mapLegend');
  if(lg) lg.innerHTML = MAP.n
    ? `<b>${MAP.n.toLocaleString()}</b> designs passed the filters${MAP.round ? ' · round <b>' + MAP.round + '</b>' : ''}${MAP.legendExtra}<br>brighter = closer on every target · dashed ring: 1σ`
    : 'no search yet · Synthesize to fill the map';
}
function pb2(G){ return G.pb - 14; }

function buildMapTabs(){
  const seg = document.getElementById('mapAxSeg');
  seg.innerHTML = PAD_DEFS.map((d, i) => `<button type="button" data-i="${i}" class="${i === MAP.tab ? 'on' : ''}">${MAP_TAB_LABELS[d.id] || d.title}</button>`).join('');
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => mapSetTab(+b.dataset.i)));
}
function mapSetTab(i){
  MAP.tab = i;
  document.querySelectorAll('#mapAxSeg button').forEach(b => b.classList.toggle('on', +b.dataset.i === i));
  mapRender();
}
function wireMap(){
  const cv = document.getElementById('mapCv');
  cv.addEventListener('click', e => {
    if(SYNTH.running || !MAP.rings.length) return;
    const G = mapGeom(cv), r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 16;
    MAP.rings.forEach((q, i) => { const d = Math.hypot(G.sx(q[G.A.kx]) - mx, G.sy(q[G.A.ky]) - my); if(d < bd){ bd = d; best = i; } });
    if(best >= 0) selectCandidate(best);
  });
  cv.addEventListener('mousemove', e => {
    if(!MAP.rings.length){ cv.style.cursor = 'default'; return; }
    const G = mapGeom(cv), r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    cv.style.cursor = MAP.rings.some(q => Math.hypot(G.sx(q[G.A.kx]) - mx, G.sy(q[G.A.ky]) - my) < 12) ? 'pointer' : 'default';
  });
  if(typeof ResizeObserver !== 'undefined') new ResizeObserver(() => mapRender()).observe(document.getElementById('mapBody'));
  buildMapTabs();
}
