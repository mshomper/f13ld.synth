/* ============================================================
   F13LD.synth · 42-inspector.js
   The selected candidate: 3-D preview (25-raymarch), match / validity /
   confidence, one bar per target (target line, prediction, ±1σ band),
   the other predictions with the model's fit for each, and hand-offs.
   ============================================================ */
'use strict';

let VIEWER = null;
const CONF_TEXT = {
  high:   'The forest\'s trees agree on this design about as well as on its training data.',
  medium: 'The trees disagree more than usual here. Treat the numbers as direction.',
  low:    'The trees disagree strongly: this design is far from the training data. Check it in F13LD.lab.'
};

// Confidence note, with the shape check's finding when it moved the label.
function confText(r){
  const sh = r.shape;
  if(!sh || !sh.level) return CONF_TEXT[r.confidence];
  return `Measured from the shape, this design is ${sh.measured.toFixed(1)}% solid; the model expected ${sh.predicted.toFixed(1)}%. ` +
    'The model is wrong about this design, so its other numbers are suspect too. Check it in F13LD.lab.';
}
const measuredMark = (key, r) => (key === 'volume_fraction' || key === 'porosity') && r.shape ? ' <small title="measured from the shape, not predicted">measured</small>' : '';
// Row label; main-axis stiffness also says which axis is the main one.
function metricLabel(key, r){
  const m = METRIC_DEFS[key], base = m[resolveValue(1, m.norm_kind).isResolved ? 'labelAbs' : 'label'];
  return key === 'stiff_main' ? `${base} (${'XYZ'[SynthSearch.mainAxis(r.metrics)]})` : base;
}

function ensureViewer(){
  if(VIEWER) return VIEWER;
  const host = document.getElementById('viewHost');
  VIEWER = new SynthViewer(host);
  document.querySelectorAll('#tileSeg button').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('#tileSeg button').forEach(x => x.classList.toggle('on', x === b));
    VIEWER.setTiles(+b.dataset.t); renderViewLabel();
  }));
  const clipRow = document.getElementById('clipRow'), clipPos = document.getElementById('clipPos');
  host.querySelectorAll('[data-clip]').forEach(b => b.addEventListener('click', () => {
    const a = +b.dataset.clip, on = VIEWER.view.clipAxis === a ? 0 : a;
    host.querySelectorAll('[data-clip]').forEach(x => x.classList.toggle('on', +x.dataset.clip === on));
    clipRow.hidden = !on; if(on) clipPos.value = 0;
    VIEWER.setClip(on, 0);
  }));
  clipPos.addEventListener('input', () => VIEWER.setClip(VIEWER.view.clipAxis, +clipPos.value));
  document.getElementById('viewReset').addEventListener('click', () => VIEWER.reset());
  // F13LD.queue thumbnails use the preview when there is one.
  window.F13LD_snapshot = function(){
    try {
      const s = VIEWER.canvas, k = 256 / Math.max(s.width, s.height), o = document.createElement('canvas');
      o.width = Math.round(s.width * k); o.height = Math.round(s.height * k);
      o.getContext('2d').drawImage(s, 0, 0, o.width, o.height);
      return o.toDataURL('image/webp', 0.72);
    } catch(e){ return null; }
  };
  return VIEWER;
}
function renderViewLabel(){
  const r = selectedResult(); if(!r) return;
  const cs = globalInputs.cell_size_mm, t = VIEWER ? VIEWER.tiles : 1;
  document.getElementById('vLabel').innerHTML = `<b>${r.design.mode}</b> · ${t === 1 ? 'one cell' : '2 × 2 × 2 cells'}${cs ? ' · cell <b>' + cs + ' mm</b>' : ''} · as Mesh and Lab build it`;
}

function bulletRow(key, r, req){
  const m = METRIC_DEFS[key], pred = r.metrics[key], target = req.targets[key], sig = (req.sigmas && req.sigmas[key]) || Predictor.sigmas[key];
  const lo = m.min, hi = m.max, s = v => 4 + (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * 292;
  const z = (pred - target) / sig, c = dotColorForZ(z), d = displayMetric(key, pred);
  return `<div class="nm">${metricLabel(key, r)}</div>
    <div class="val" style="color:${c}">${d.v}<small>${d.unit}</small>${measuredMark(key, r)} <small>${z >= 0 ? '+' : ''}${z.toFixed(1)}σ</small></div>
    <svg viewBox="0 0 300 16" preserveAspectRatio="none" aria-hidden="true"><rect x="4" y="7" width="292" height="2" rx="1" fill="rgba(255,255,255,.1)"/>
     <rect x="${s(pred - sig)}" y="4" width="${Math.max(2, s(pred + sig) - s(pred - sig))}" height="8" rx="2" fill="${c}" opacity=".22"/>
     <path d="M${s(target)} 1V15" stroke="#FFB670" stroke-width="2"/><circle cx="${s(pred)}" cy="8" r="4" fill="${c}"/></svg>`;
}
function fitBar(r2){
  if(r2 == null) return '<span></span>';
  const cls = r2 >= 0.7 ? '' : r2 >= 0.4 ? 'med' : 'low';
  return `<span class="fit ${cls}" title="model fit on held-out designs: R² ${r2.toFixed(2)}"><i style="width:${Math.max(4, r2 * 100).toFixed(0)}%"></i></span>`;
}

function renderInspector(){
  const r = selectedResult();
  document.getElementById('inspEmpty').hidden = !!r;
  document.getElementById('inspMain').hidden = !r;
  if(!r){ document.getElementById('iSeed').textContent = ''; return; }
  ensureViewer();
  if(VIEWER.design !== r.design) VIEWER.setDesign(r.design);
  renderViewLabel();
  const i = SYNTH.sel;
  document.getElementById('iSeed').textContent = 'grown from training seed ' + r.seedIndex;
  document.getElementById('iRank').textContent = '#' + (i + 1);
  const cc = CONF_COLOR[r.confidence];
  document.getElementById('iTags').innerHTML = `<span class="tag">${presetDisplayLabel(r.presetKey)}</span><span class="tag">${r.design.mode}</span>` +
    `<span class="tag" style="color:${cc};border-color:${cc}66" title="${confText(r)}"><span class="dot" style="background:${cc}"></span>${r.confidence} confidence</span>` +
    (r.shape && r.shape.level ? `<span class="tag" style="color:var(--warn);border-color:var(--warn)" title="${confText(r)}">shape differs from model</span>` : '');
  document.getElementById('iScore').textContent = Math.round(r.score * 100) + '%';
  const v = document.getElementById('iValid'); v.textContent = Math.round(r.validity * 100) + '%'; v.style.color = r.validity < 0.85 ? 'var(--warn)' : '';
  const ic = document.getElementById('iConf'); ic.textContent = r.confidence; ic.style.color = cc; ic.title = confText(r);

  const req = SYNTH.lastReq || { targets: {}, weights: {} };
  const shown = Predictor.shownMetrics();
  const targeted = Object.keys(METRIC_DEFS).filter(k => req.weights[k] > 0 && req.targets[k] != null && r.metrics[k] != null);
  document.getElementById('iBul').innerHTML = targeted.map(k => bulletRow(k, r, req)).join('') ||
    '<div class="nm" style="grid-column:1/-1;color:var(--t3)">no pad was on</div>';
  const R2 = Predictor.metricsR2 || {};
  document.getElementById('iOth').innerHTML = shown.filter(k => !targeted.includes(k)).map(k => {
    const m = METRIC_DEFS[k], d = displayMetric(k, r.metrics[k]);
    const label = resolveValue(1, m.norm_kind).isResolved ? m.labelAbs : m.label;
    return `<span class="nm ${R2[k] != null && R2[k] < 0.5 ? 'rough' : ''}">${label}</span>${fitBar(R2[k])}<span class="v">${d.v}<small>${d.unit}</small>${measuredMark(k, r)}</span>`;
  }).join('');
  renderHeaderPills();
}
function wireInspector(){
  document.getElementById('iMesh').addEventListener('click', () => openInMesh(selectedResult()));
  document.getElementById('iLab').addEventListener('click', () => openInLab(selectedResult()));
  document.getElementById('iCopy').addEventListener('click', e => copyRecipe(selectedResult(), e.currentTarget));
}
