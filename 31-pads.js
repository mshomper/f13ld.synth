/* ============================================================
   F13LD.synth · 31-pads.js
   Design intent: the preset filter, the connectivity filter and the three
   X/Y pads (off / prefer / require, drag or number entry), and turning
   them into search targets. Results are shown on the result map, not on
   the pads (Matt, 2026-10-09).
   ============================================================ */
'use strict';

const padState = {};   // padId -> { x, y, mode, precisionMode }

function loadPadState() {
  for (const def of PAD_DEFS) padState[def.id] = { x: def.defaultPos.x, y: def.defaultPos.y, mode: def.defaultMode, precisionMode: false };
  try {
    const saved = JSON.parse(localStorage.getItem(PADS_STORAGE_KEY) || 'null');
    if (saved && typeof saved === 'object') for (const def of PAD_DEFS) if (saved[def.id]) Object.assign(padState[def.id], saved[def.id]);
  } catch(e) {}
}
function persistPadState() { try { localStorage.setItem(PADS_STORAGE_KEY, JSON.stringify(padState)); } catch(e) {} }

// ── Connectivity: candidates must be connected along this many axes ──
let connectivityState = 3;
function loadConnectivityState() {
  try { const v = JSON.parse(localStorage.getItem(CONNECTIVITY_STORAGE_KEY) || 'null'); if ([1,2,3].includes(v)) connectivityState = v; } catch(e) {}
}
function wireConnectivity() {
  const seg = document.getElementById('connSeg');
  const sync = () => seg.querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.v === connectivityState));
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    connectivityState = +b.dataset.v; sync();
    try { localStorage.setItem(CONNECTIVITY_STORAGE_KEY, JSON.stringify(connectivityState)); } catch(e) {}
  }));
  sync();
}

// ── Preset: grow candidates only from training seeds of this preset ──
// The list comes from the loaded bundle: retrained bundles carry each seed's
// sweep preset; older ones are grouped by term skeleton (some presets share
// one and show together).
let presetState = '';
function loadPresetState(){ try { presetState = localStorage.getItem(PRESET_STORAGE_KEY) || ''; } catch(e){} }
function buildPresetDropdown(){
  const sel = document.getElementById('presetSel');
  const opts = Predictor.presetOptions();
  sel.innerHTML = '<option value="">any preset · ' + opts.reduce((a, o) => a + o.n, 0) + ' seeds</option>' +
    opts.map(o => `<option value="${o.key}">${o.label} · ${o.n} seeds</option>`).join('');
  if (!opts.some(o => o.key === presetState)) presetState = '';
  sel.value = presetState;
  sel.onchange = () => { presetState = sel.value; try { localStorage.setItem(PRESET_STORAGE_KEY, presetState); } catch(e){} };
}

// ── Pad geometry ──
function padPosToMetricValue(def, axis) {
  const m = METRIC_DEFS[axis === 'x' ? def.xMetric : def.yMetric];
  return m.min + padState[def.id][axis] * (m.max - m.min);
}
function metricValueToPadPos(def, axis, value) {
  const m = METRIC_DEFS[axis === 'x' ? def.xMetric : def.yMetric];
  return Math.max(0, Math.min(1, (value - m.min) / (m.max - m.min)));
}
function padTargetText(def, axis) {
  const key = axis === 'x' ? def.xMetric : def.yMetric;
  const r = displayMetric(key, padPosToMetricValue(def, axis));
  return r.v + (r.unit ? ' ' + r.unit : '');
}

const PAD_W = 300, PAD_H = 150, PAD_P = 20;
function padSVG(def) {
  const s = padState[def.id], W = PAD_W, H = PAD_H, P = PAD_P;
  const tx = P + s.x * (W - 2 * P), ty = H - P - s.y * (H - 2 * P);
  const col = s.mode === 'require' ? '#FFB670' : '#c8f542';
  const corner = (txt, x, y, a) => { const [q, r] = txt.split(' · '); return `<text x="${x}" y="${y}" text-anchor="${a}" font-family="IBM Plex Mono" font-size="8.5" fill="#5f6582"><tspan x="${x}">${q}</tspan><tspan x="${x}" dy="10">${r}</tspan></text>`; };
  const c = def.cornerLabels;
  return `<svg class="pad-svg" viewBox="0 0 ${W} ${H}" data-pad="${def.id}" role="img" aria-label="${def.xLabel} and ${def.yLabel} target">
   <rect x="${P}" y="${P}" width="${W - 2 * P}" height="${H - 2 * P}" fill="#080b12" stroke="rgba(255,255,255,.07)"/>
   <path d="M${W / 2} ${P}V${H - P}M${P} ${H / 2}H${W - P}" stroke="rgba(255,255,255,.05)"/>
   ${corner(c.tl, P + 5, P + 11, 'start')}${corner(c.tr, W - P - 5, P + 11, 'end')}${corner(c.bl, P + 5, H - P - 15, 'start')}${corner(c.br, W - P - 5, H - P - 15, 'end')}
   <text x="${W / 2}" y="${H - 5}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="#7BD3E3">${def.xLabel.toLowerCase()} →</text>
   <text x="9" y="${H / 2}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="#FFB670" transform="rotate(-90 9 ${H / 2})">${def.yLabel.toLowerCase()} →</text>
   <g class="pad-target"><circle cx="${tx}" cy="${ty}" r="11" fill="none" stroke="${col}" stroke-width="1.4" opacity=".45"/><circle cx="${tx}" cy="${ty}" r="5" fill="#0a1820" stroke="${col}" stroke-width="2"/>
   <path d="M${tx - 16} ${ty}H${tx - 9}M${tx + 9} ${ty}H${tx + 16}M${tx} ${ty - 16}V${ty - 9}M${tx} ${ty + 9}V${ty + 16}" stroke="${col}" stroke-width="1" opacity=".6"/></g>
  </svg>`;
}

function renderPads() {
  const box = document.getElementById('pads');
  if (!box) return;
  box.innerHTML = PAD_DEFS.map(def => {
    const s = padState[def.id], on = s.mode !== 'off';
    const xd = METRIC_DEFS[def.xMetric], yd = METRIC_DEFS[def.yMetric];
    const body = !on ? '' : s.precisionMode
      ? `<div class="pad-prec">
           <label>${def.xLabel}<input type="number" step="any" data-prec="${def.id}" data-axis="x" value="${precValue(def, 'x')}"></label>
           <label>${def.yLabel}<input type="number" step="any" data-prec="${def.id}" data-axis="y" value="${precValue(def, 'y')}"></label>
         </div><div class="pad-ro"><span>${unitText(xd)}</span><span>${unitText(yd)}</span></div>`
      : padSVG(def) + `<div class="pad-ro"><span>target <b data-ro="${def.id}x">${padTargetText(def, 'x')}</b></span><span><b data-ro="${def.id}y">${padTargetText(def, 'y')}</b></span></div>`;
    return `<div class="pad ${s.mode === 'require' ? 'req' : s.mode === 'prefer' ? 'pref' : 'off'}">
      <div class="pad-h"><div class="pad-t"><span class="x">${def.xLabel}</span><span class="vs"> × </span><span class="y">${def.yLabel}</span></div>
        ${on ? `<button type="button" class="pad-num ${s.precisionMode ? 'on' : ''}" data-num="${def.id}" title="${s.precisionMode ? 'Back to the pad' : 'Type exact values'}">${glyph(s.precisionMode ? 'pad' : 'numpad')}</button>` : ''}
        <div class="seg" data-padmode="${def.id}">
          <button type="button" data-m="off" class="${s.mode === 'off' ? 'on' : ''}">off</button><button type="button" data-m="prefer" class="${s.mode === 'prefer' ? 'on' : ''}">prefer</button><button type="button" data-m="require" class="req ${s.mode === 'require' ? 'on' : ''}">require</button>
        </div></div>
      ${body}</div>`;
  }).join('');
  const n = PAD_DEFS.filter(d => padState[d.id].mode !== 'off').length;
  document.getElementById('nActive').textContent = n + ' pad' + (n === 1 ? '' : 's') + ' on';

  box.querySelectorAll('[data-padmode] button').forEach(b => b.addEventListener('click', () => {
    padState[b.parentNode.dataset.padmode].mode = b.dataset.m; persistPadState(); renderPads();
    if (typeof onIntentChanged === 'function') onIntentChanged();
  }));
  box.querySelectorAll('[data-num]').forEach(b => b.addEventListener('click', () => {
    const s = padState[b.dataset.num]; s.precisionMode = !s.precisionMode; persistPadState(); renderPads();
  }));
  box.querySelectorAll('input[data-prec]').forEach(inp => inp.addEventListener('change', () => {
    const def = PAD_DEFS.find(d => d.id === inp.dataset.prec), axis = inp.dataset.axis;
    const key = axis === 'x' ? def.xMetric : def.yMetric, v = parseFloat(inp.value);
    if (!isFinite(v)) return;
    padState[def.id][axis] = metricValueToPadPos(def, axis, unresolveValue(v, METRIC_DEFS[key].norm_kind));
    persistPadState(); renderPads();
    if (typeof onIntentChanged === 'function') onIntentChanged();
  }));
  box.querySelectorAll('svg[data-pad]').forEach(svg => {
    const def = PAD_DEFS.find(d => d.id === svg.dataset.pad);
    let drag = false;
    const move = e => {
      const r = svg.getBoundingClientRect();
      const s = padState[def.id];
      s.x = Math.max(0, Math.min(1, ((e.clientX - r.left) * PAD_W / r.width - PAD_P) / (PAD_W - 2 * PAD_P)));
      s.y = Math.max(0, Math.min(1, 1 - ((e.clientY - r.top) * PAD_H / r.height - PAD_P) / (PAD_H - 2 * PAD_P)));
      // cheap update: move the target, refresh the readout
      const tx = PAD_P + s.x * (PAD_W - 2 * PAD_P), ty = PAD_H - PAD_P - s.y * (PAD_H - 2 * PAD_P);
      svg.querySelector('.pad-target').setAttribute('transform', `translate(${tx - (PAD_P + svg._x0 * (PAD_W - 2 * PAD_P))} ${ty - (PAD_H - PAD_P - svg._y0 * (PAD_H - 2 * PAD_P))})`);
      const rx = document.querySelector(`[data-ro="${def.id}x"]`), ry = document.querySelector(`[data-ro="${def.id}y"]`);
      if (rx) rx.textContent = padTargetText(def, 'x'); if (ry) ry.textContent = padTargetText(def, 'y');
      if (typeof onIntentChanged === 'function') onIntentChanged(true);
    };
    svg._x0 = padState[def.id].x; svg._y0 = padState[def.id].y;
    svg.addEventListener('pointerdown', e => { drag = true; svg.setPointerCapture(e.pointerId); move(e); e.preventDefault(); });
    svg.addEventListener('pointermove', e => { if (drag) move(e); });
    const end = () => { if (!drag) return; drag = false; persistPadState(); renderPads(); if (typeof onIntentChanged === 'function') onIntentChanged(); };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
  });
}
function precValue(def, axis) {
  const key = axis === 'x' ? def.xMetric : def.yMetric;
  const r = resolveValue(padPosToMetricValue(def, axis), METRIC_DEFS[key].norm_kind);
  return +r.value.toFixed(r.unit === 'µm' ? 0 : 3);
}
function unitText(m) { const r = resolveValue(1, m.norm_kind); return r.isResolved ? r.unit : (m.norm_kind !== 'none' ? 'normalized' : (m.unit || '')); }

// ── Targets for the search ──
// Each on pad sets its X and Y targets; the stiffness pad also sets Ey and Ez
// (stiffness as a whole), slightly weaker.
function buildTargetsFromPads() {
  const targets = {}, weights = {};
  for (const def of PAD_DEFS) {
    const s = padState[def.id];
    if (s.mode === 'off') continue;
    const w = MODE_WEIGHTS[s.mode];
    targets[def.xMetric] = padPosToMetricValue(def, 'x'); weights[def.xMetric] = Math.max(weights[def.xMetric] || 0, w);
    targets[def.yMetric] = padPosToMetricValue(def, 'y'); weights[def.yMetric] = Math.max(weights[def.yMetric] || 0, w);
    if (def.coupledMetrics) for (const c of def.coupledMetrics) { targets[c] = targets[def.xMetric]; weights[c] = Math.max(weights[c] || 0, w * 0.7); }
  }
  weights.surface_complexity = 0;
  return { targets, weights };
}
function firstActivePad() { return PAD_DEFS.find(d => padState[d.id].mode !== 'off') || null; }
