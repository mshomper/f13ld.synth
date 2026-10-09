/* ============================================================
   F13LD.synth · 31-pads.js
   The three X/Y design-intent pads, the connectivity filter, the preset
   filter, and turning pad state into search targets.
   ============================================================ */
'use strict';

const padState = {};   // padId -> { x, y, mode, precisionMode }

// ============================================================
// PADS
// ============================================================
function loadPadState() {
  // Initialize with defaults first
  for (const def of PAD_DEFS) {
    padState[def.id] = { x: def.defaultPos.x, y: def.defaultPos.y, mode: def.defaultMode, precisionMode: false };
  }
  // Then overlay any saved state
  try {
    const saved = JSON.parse(localStorage.getItem(PADS_STORAGE_KEY) || 'null');
    if (saved && typeof saved === 'object') {
      for (const def of PAD_DEFS) {
        if (saved[def.id]) Object.assign(padState[def.id], saved[def.id]);
      }
    }
  } catch(e) {}
}
function persistPadState() { try { localStorage.setItem(PADS_STORAGE_KEY, JSON.stringify(padState)); } catch(e) {} }

// Connectivity filter: which directionality bucket (1, 2, or 3 connected axes)
// the search results MUST match. Single-select — always exactly one bucket
// active, since "force N-axis connectivity" is a single constraint.
let connectivityState = 3;  // default: force fully connected
function loadConnectivityState() {
  try {
    const saved = JSON.parse(localStorage.getItem(CONNECTIVITY_STORAGE_KEY) || 'null');
    if ([1,2,3].includes(saved)) connectivityState = saved;
  } catch(e) {}
}
function persistConnectivityState() {
  try { localStorage.setItem(CONNECTIVITY_STORAGE_KEY, JSON.stringify(connectivityState)); } catch(e) {}
}
function toggleConnectivity(n) {
  // Single-select: clicking sets the active bucket. No deselect — exactly one
  // constraint is always in effect, matching "force N-axis connectivity" intent.
  if (![1,2,3].includes(n)) return;
  connectivityState = n;
  persistConnectivityState();
  syncConnectivityToggleUI();
}
function syncConnectivityToggleUI() {
  const tg = document.getElementById('connectivityToggle');
  if (!tg) return;
  tg.querySelectorAll('button').forEach(b => {
    b.classList.toggle('active', parseInt(b.dataset.axes) === connectivityState);
  });
}

// Map pad's normalized [0,1] position → metric value in metric's native units
// (which for normalized metrics is the model's predicted normalized space)
function padPosToMetricValue(padDef, axis) {
  const metricKey = axis === 'x' ? padDef.xMetric : padDef.yMetric;
  const m = METRIC_DEFS[metricKey];
  const pos = padState[padDef.id][axis];
  // Map pad pos [0,1] → metric range [min, max] in normalized units.
  // For metrics with norm_kind !== 'none', that's the predictor's space directly.
  return m.min + pos * (m.max - m.min);
}

function metricValueToPadPos(padDef, axis, value) {
  const metricKey = axis === 'x' ? padDef.xMetric : padDef.yMetric;
  const m = METRIC_DEFS[metricKey];
  return Math.max(0, Math.min(1, (value - m.min) / (m.max - m.min)));
}

function buildPadRack() {
  const rack = document.getElementById('padRack'); rack.innerHTML = '';
  for (const def of PAD_DEFS) {
    const card = document.createElement('div');
    card.className = 'pad-card';
    card.id = `pad-${def.id}`;
    card.innerHTML = `
      <div class="pad-head">
        <div class="pad-title" onclick="togglePrecision('${def.id}')">
          <span class="pad-title-x">${def.xLabel}</span><span class="pad-title-vs"> × </span><span class="pad-title-y">${def.yLabel}</span>
          <span class="pad-title-flip" title="Switch to number entry"><svg width="13" height="11" viewBox="0 0 13 11" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 3h10M8.5 0.8 11 3 8.5 5.2"/><path d="M12 8H2M4.5 5.8 2 8l2.5 2.2"/></svg></span>
        </div>
        <div class="pad-toggle" id="toggle-${def.id}">
          <button data-mode="off">off</button>
          <button data-mode="prefer">prefer</button>
          <button data-mode="require" class="require">require</button>
        </div>
      </div>
      <div class="pad-svg-wrap" id="svg-${def.id}"></div>
      <div class="pad-precision">
        <div class="pad-prec-cell">
          <label>${def.xLabel}</label>
          <input type="number" id="prec-${def.id}-x" step="0.01">
          <span class="pp-units" id="prec-${def.id}-x-units"></span>
        </div>
        <div class="pad-prec-cell">
          <label>${def.yLabel}</label>
          <input type="number" id="prec-${def.id}-y" step="0.01">
          <span class="pp-units" id="prec-${def.id}-y-units"></span>
        </div>
      </div>
      <div class="pad-readout" id="readout-${def.id}">
        <div class="ro-x"><span class="ro-lbl">${def.xLabel}</span><span class="ro-val">—</span></div>
        <div class="ro-y"><span class="ro-lbl">${def.yLabel}</span><span class="ro-val">—</span></div>
      </div>
    `;
    rack.appendChild(card);

    // Wire toggle buttons
    const toggle = card.querySelector(`#toggle-${def.id}`);
    toggle.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        padState[def.id].mode = mode;
        persistPadState();
        renderPadCard(def.id);
      });
    });

    // Wire precision inputs
    document.getElementById(`prec-${def.id}-x`).addEventListener('input', e => updatePrecisionInput(def.id, 'x', e.target.value));
    document.getElementById(`prec-${def.id}-y`).addEventListener('input', e => updatePrecisionInput(def.id, 'y', e.target.value));

    renderPadCard(def.id);
    buildPadSVG(def.id);
  }
}

function renderPadCard(padId) {
  const def = PAD_DEFS.find(d => d.id === padId);
  const state = padState[padId];
  const card = document.getElementById(`pad-${padId}`);
  card.classList.toggle('disabled', state.mode === 'off');
  card.classList.toggle('require', state.mode === 'require');
  card.classList.toggle('precision', state.precisionMode);

  // Toggle highlight
  const tg = card.querySelector(`.pad-toggle`);
  tg.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.mode === state.mode));

  // Readout
  updatePadReadout(padId);

  // Precision input values
  const xVal = padPosToMetricValue(def, 'x');
  const yVal = padPosToMetricValue(def, 'y');
  const xRes = resolveValue(xVal, METRIC_DEFS[def.xMetric].norm_kind);
  const yRes = resolveValue(yVal, METRIC_DEFS[def.yMetric].norm_kind);
  document.getElementById(`prec-${padId}-x`).value = formatVal(xRes.isResolved ? xRes.value : xVal, METRIC_DEFS[def.xMetric].decimals);
  document.getElementById(`prec-${padId}-y`).value = formatVal(yRes.isResolved ? yRes.value : yVal, METRIC_DEFS[def.yMetric].decimals);
  document.getElementById(`prec-${padId}-x-units`).textContent = xRes.isResolved ? xRes.unit : (METRIC_DEFS[def.xMetric].norm_kind !== 'none' ? 'normalized' : METRIC_DEFS[def.xMetric].unit);
  document.getElementById(`prec-${padId}-y-units`).textContent = yRes.isResolved ? yRes.unit : (METRIC_DEFS[def.yMetric].norm_kind !== 'none' ? 'normalized' : METRIC_DEFS[def.yMetric].unit);
}

function togglePrecision(padId) {
  padState[padId].precisionMode = !padState[padId].precisionMode;
  persistPadState();
  renderPadCard(padId);
  // When leaving precision mode, the SVG was display:none — rebuild it to ensure
  // the dot reflects any value changes made in precision inputs.
  if (!padState[padId].precisionMode) buildPadSVG(padId);
}

function updatePrecisionInput(padId, axis, value) {
  const def = PAD_DEFS.find(d => d.id === padId);
  const metricKey = axis === 'x' ? def.xMetric : def.yMetric;
  const m = METRIC_DEFS[metricKey];
  let v = parseFloat(value);
  if (!isFinite(v)) return;
  // If the displayed value is in physical units, convert back to normalized first
  const isShowingPhys = m.norm_kind !== 'none' && resolveValue(1, m.norm_kind).isResolved;
  if (isShowingPhys) v = unresolveValue(v, m.norm_kind);
  padState[padId][axis] = metricValueToPadPos(def, axis, v);
  persistPadState();
  updatePadReadout(padId);
}

function updatePadReadout(padId) {
  const def = PAD_DEFS.find(d => d.id === padId);
  const state = padState[padId];
  const readout = document.getElementById(`readout-${padId}`);
  if (state.mode === 'off') { readout.querySelector('.ro-x .ro-val').textContent = '—'; readout.querySelector('.ro-y .ro-val').textContent = '—'; return; }
  const xVal = padPosToMetricValue(def, 'x');
  const yVal = padPosToMetricValue(def, 'y');
  const xRes = resolveValue(xVal, METRIC_DEFS[def.xMetric].norm_kind);
  const yRes = resolveValue(yVal, METRIC_DEFS[def.yMetric].norm_kind);
  const xDecs = METRIC_DEFS[def.xMetric].decimals;
  const yDecs = METRIC_DEFS[def.yMetric].decimals;
  const xDisplay = xRes.isResolved ? `${formatVal(xRes.value, xDecs)} ${xRes.unit}` : `${formatVal(xVal, xDecs)}`;
  const yDisplay = yRes.isResolved ? `${formatVal(yRes.value, yDecs)} ${yRes.unit}` : `${formatVal(yVal, yDecs)}`;
  readout.querySelector('.ro-x .ro-val').innerHTML = xDisplay;
  readout.querySelector('.ro-y .ro-val').innerHTML = yDisplay;
}

function buildPadSVG(padId) {
  const def = PAD_DEFS.find(d => d.id === padId);
  const wrap = document.getElementById(`svg-${padId}`);
  const W = 300, H = 168, P = 24;

  // Build the SVG once with stable identifiers for the dot, so subsequent
  // moves can update cx/cy without destroying pointer listeners.
  wrap.innerHTML = `
    <svg class="pad-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line class="pad-grid-line" x1="${P}" y1="${H/2}" x2="${W-P}" y2="${H/2}"/>
      <line class="pad-grid-line" x1="${W/2}" y1="${P}" x2="${W/2}" y2="${H-P}"/>
      <rect x="${P}" y="${P}" width="${W-2*P}" height="${H-2*P}" fill="none" stroke="#1a2030" stroke-width="0.5"/>
      <text class="pad-corner-label" x="${P+4}" y="${P+9}" text-anchor="start"><tspan x="${P+4}">${def.cornerLabels.tl.split(' · ')[0]}</tspan><tspan x="${P+4}" dy="10">${def.cornerLabels.tl.split(' · ')[1]}</tspan></text>
      <text class="pad-corner-label" x="${W-P-4}" y="${P+9}" text-anchor="end"><tspan x="${W-P-4}">${def.cornerLabels.tr.split(' · ')[0]}</tspan><tspan x="${W-P-4}" dy="10">${def.cornerLabels.tr.split(' · ')[1]}</tspan></text>
      <text class="pad-corner-label" x="${P+4}" y="${H-P-13}" text-anchor="start"><tspan x="${P+4}">${def.cornerLabels.bl.split(' · ')[0]}</tspan><tspan x="${P+4}" dy="10">${def.cornerLabels.bl.split(' · ')[1]}</tspan></text>
      <text class="pad-corner-label" x="${W-P-4}" y="${H-P-13}" text-anchor="end"><tspan x="${W-P-4}">${def.cornerLabels.br.split(' · ')[0]}</tspan><tspan x="${W-P-4}" dy="10">${def.cornerLabels.br.split(' · ')[1]}</tspan></text>
      <text class="pad-axis-label" x="${W/2}" y="${H-6}" text-anchor="middle">→ ${def.xLabel.toUpperCase()}</text>
      <text class="pad-axis-label" x="14" y="${H/2+3}" text-anchor="start" transform="rotate(-90, 14, ${H/2+3})">→ ${def.yLabel.toUpperCase()}</text>
      <circle class="pad-dot-glow" id="dotglow-${padId}" cx="0" cy="0" r="11"/>
      <circle class="pad-dot" id="dot-${padId}" cx="0" cy="0" r="5"/>
    </svg>
  `;

  const svg = wrap.querySelector('svg');

  // Pointer event handling — works for both single-click placement and click-drag.
  // Critical: we don't rebuild the SVG during drag, only update dot position.
  let dragging = false;
  const updateFromPointer = (e) => {
    const rect = svg.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (W / rect.width);
    const cy = (e.clientY - rect.top) * (H / rect.height);
    const xN = Math.max(0, Math.min(1, (cx - P) / (W - 2*P)));
    const yN = Math.max(0, Math.min(1, 1 - (cy - P) / (H - 2*P)));
    padState[padId].x = xN;
    padState[padId].y = yN;
    updatePadDot(padId);
    updatePadReadout(padId);
  };
  svg.addEventListener('pointerdown', e => {
    if (padState[padId].mode === 'off') return;
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    updateFromPointer(e);
    e.preventDefault();
  });
  svg.addEventListener('pointermove', e => {
    if (dragging) updateFromPointer(e);
  });
  const endDrag = (e) => {
    if (dragging) {
      dragging = false;
      try { svg.releasePointerCapture(e.pointerId); } catch(_) {}
      persistPadState();
    }
  };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);

  // Initial dot placement
  updatePadDot(padId);
}

// Cheap update — just moves the dot, doesn't touch listeners.
function updatePadDot(padId) {
  const state = padState[padId];
  const W = 300, H = 168, P = 24;
  const px = P + state.x * (W - 2*P);
  const py = P + (1 - state.y) * (H - 2*P);
  const dot = document.getElementById(`dot-${padId}`);
  const glow = document.getElementById(`dotglow-${padId}`);
  if (dot) { dot.setAttribute('cx', px); dot.setAttribute('cy', py); }
  if (glow) { glow.setAttribute('cx', px); glow.setAttribute('cy', py); }
}

// ============================================================
// SEARCH ORCHESTRATION
// ============================================================
function buildTargetsFromPads() {
  // For each ON pad, contribute its X/Y metric targets and weights.
  // For pad 1 (mech×pore), the X stiffness is also broadcast to ey/ez (coupledMetrics).
  const targets = {};   // metric_key (normalized) -> target value (normalized)
  const weights = {};   // metric_key -> 0..1 weight
  for (const def of PAD_DEFS) {
    const s = padState[def.id];
    if (s.mode === 'off') continue;
    const w = MODE_WEIGHTS[s.mode];
    // X
    targets[def.xMetric] = padPosToMetricValue(def, 'x');
    weights[def.xMetric] = Math.max(weights[def.xMetric] || 0, w);
    // Y
    targets[def.yMetric] = padPosToMetricValue(def, 'y');
    weights[def.yMetric] = Math.max(weights[def.yMetric] || 0, w);
    // Coupled (stiffness pad broadcasts X-target to Ey/Ez)
    if (def.coupledMetrics) {
      for (const c of def.coupledMetrics) {
        targets[c] = targets[def.xMetric];
        weights[c] = Math.max(weights[c] || 0, w * 0.7);  // slightly weaker on coupled
      }
    }
  }
  // surface_complexity always weight 0 (dropped from controls)
  weights.surface_complexity = 0;
  return { targets, weights };
}

// ── Preset filter ──────────────────────────────────────────────────────────
// Candidates are grown only from training seeds of the chosen preset. The
// list comes from the loaded bundle: each seed carries the preset of the
// sweep it came from (newer bundles), or is matched by its term skeleton
// (older bundles, where some presets share a skeleton and show together).
let presetState = '';
function loadPresetState(){ try { presetState = localStorage.getItem(PRESET_STORAGE_KEY) || ''; } catch(e){} }
function persistPresetState(){ try { localStorage.setItem(PRESET_STORAGE_KEY, presetState); } catch(e){} }
function buildPresetDropdown(){
  const sel = document.getElementById('presetSel');
  const opts = Predictor.presetOptions();
  sel.innerHTML = '<option value="">any preset</option>' +
    opts.map(o => `<option value="${o.key}">${o.label} · ${o.n} seeds</option>`).join('');
  if(!opts.some(o => o.key === presetState)) presetState = '';
  sel.value = presetState;
}
function onPresetChange(){ presetState = document.getElementById('presetSel').value; persistPresetState(); }

async function runSearch() {
  if (!Predictor.loaded) { alert('Predictor not loaded yet — wait for the model to finish loading.'); return; }
  const allOff = PAD_DEFS.every(d => padState[d.id].mode === 'off');
  if (allOff) { alert('Set at least one pad to "prefer" or "require" to define design intent.'); return; }
  const btn = document.getElementById('searchBtn');
  btn.disabled = true;
  const meta = document.getElementById('resultsMeta');
  meta.innerHTML = '<span class="spinner"></span>starting search…';
  document.getElementById('resultsBody').innerHTML = '';

  const { targets, weights } = buildTargetsFromPads();
  try {
    const res = await Predictor.inverseSearch(
      { targets, weights, connectivity: connectivityState, presetKey: presetState || null },
      text => { meta.innerHTML = `<span class="spinner"></span>${text}…`; });
    renderResults(res, targets);
  } catch(e){
    console.error('[F13LD.synth] Search failed:', e);
    meta.textContent = 'search failed';
    document.getElementById('resultsBody').innerHTML = `<div class="engine-empty">Search failed: ${String(e.message || e).replace(/</g,'&lt;')}</div>`;
  } finally { btn.disabled = false; }
}
