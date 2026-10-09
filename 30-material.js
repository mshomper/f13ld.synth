/* ============================================================
   F13LD.synth · 30-material.js
   Material card (in Configure) — F13LD.lab's AM library, shared with
   F13LD.vault through localStorage — and normalized ↔ physical unit
   conversion for display. The material never changes the search: the
   model works in normalized units (E/Es, k/ks, pore/cell).
   ============================================================ */
'use strict';

function formatVal(v, d) { if (v == null || !isFinite(v)) return '—'; return v.toFixed(d ?? 2); }

// Old v0.2 ids (shared with F13LD.vault) → the Lab library's closest entry.
const LEGACY_MATERIAL_IDS = { ti6al4v: 'ti64-g5-lpbf-hip' };

let globalInputs = { cell_size_mm: 2, material_id: SYNTH_DEFAULT_MATERIAL, modulus_gpa: null, density_gcc: null, thermal_k_wmk: null, ref_stress_mpa: null };

function materialById(id){ return SYNTH_MATERIALS.find(m => m.id === (LEGACY_MATERIAL_IDS[id] || id)) || null; }
function materialLabel(m){ return m ? m.name + ' · ' + m.condition : 'Custom'; }
function materialShort(m){ return m ? m.name.replace(/\s*\(.*\)\s*/, ' ').trim() : 'Custom'; }

function loadGlobalInputs() {
  try {
    const saved = JSON.parse(localStorage.getItem(GLOBAL_INPUTS_STORAGE_KEY) || 'null');
    if (saved && typeof saved === 'object') Object.keys(globalInputs).forEach(k => { if (k in saved && saved[k] != null) globalInputs[k] = saved[k]; });
  } catch(e) {}
  if (LEGACY_MATERIAL_IDS[globalInputs.material_id]) globalInputs.material_id = LEGACY_MATERIAL_IDS[globalInputs.material_id];
  // A fresh browser: fill the selected material's values once.
  const m = materialById(globalInputs.material_id);
  if (m && globalInputs.modulus_gpa == null) applyMaterial(m.id, true);
}
function persistGlobalInputs() { try { localStorage.setItem(GLOBAL_INPUTS_STORAGE_KEY, JSON.stringify(globalInputs)); } catch(e) {} }

function applyMaterial(id, quiet) {
  const m = materialById(id); if (!m) return;
  globalInputs.material_id = m.id; globalInputs.modulus_gpa = m.E; globalInputs.density_gcc = m.rho;
  globalInputs.thermal_k_wmk = m.k;
  persistGlobalInputs();
  if (!quiet) onMaterialChanged();
}

function buildMaterialSelect() {
  const sel = document.getElementById('matSel');
  const groups = {};
  SYNTH_MATERIALS.forEach(m => { (groups[m.group] ||= []).push(m); });
  sel.innerHTML = Object.entries(groups).map(([g, list]) =>
    `<optgroup label="${g}">` + list.map(m => `<option value="${m.id}">${materialLabel(m)}</option>`).join('') + '</optgroup>').join('') +
    '<option value="custom">Custom…</option>';
  sel.addEventListener('change', () => {
    if (sel.value === 'custom') { globalInputs.material_id = 'custom'; persistGlobalInputs(); onMaterialChanged(); }
    else applyMaterial(sel.value);
  });
  const map = { ref_modulus_gpa: 'modulus_gpa', ref_density_gcc: 'density_gcc', ref_thermal_k_wmk: 'thermal_k_wmk', ref_cell_size_mm: 'cell_size_mm' };
  Object.entries(map).forEach(([id, fld]) => {
    document.getElementById(id).addEventListener('input', e => {
      const v = parseFloat(e.target.value);
      globalInputs[fld] = isFinite(v) && v > 0 ? v : null;
      if (fld !== 'cell_size_mm') {
        const m = materialById(globalInputs.material_id);
        if (!m || m.E !== globalInputs.modulus_gpa || m.rho !== globalInputs.density_gcc || m.k !== globalInputs.thermal_k_wmk) globalInputs.material_id = 'custom';
      }
      persistGlobalInputs(); onMaterialChanged(true);
    });
  });
  syncMaterialCard();
}
function syncMaterialCard() {
  document.getElementById('matSel').value = materialById(globalInputs.material_id) ? materialById(globalInputs.material_id).id : 'custom';
  const put = (id, v) => { const el = document.getElementById(id); if (document.activeElement !== el) el.value = v ?? ''; };
  put('ref_modulus_gpa', globalInputs.modulus_gpa); put('ref_density_gcc', globalInputs.density_gcc);
  put('ref_thermal_k_wmk', globalInputs.thermal_k_wmk); put('ref_cell_size_mm', globalInputs.cell_size_mm);
}
// Everything that shows physical units redraws.
function onMaterialChanged(fromField) {
  if (!fromField) syncMaterialCard();
  if (typeof renderPads === 'function') renderPads();
  if (typeof renderInspector === 'function') renderInspector();
  if (typeof renderDockTags === 'function') renderDockTags();
}

// Resolution helpers — normalized ↔ physical at display
function resolveValue(rawNorm, normKind) {
  if (rawNorm == null || !isFinite(rawNorm) || !normKind || normKind === 'none') return { value:rawNorm, unit:'', isResolved:false };
  switch (normKind) {
    case 'cell_length': { const cs = globalInputs.cell_size_mm; if (cs == null || cs <= 0) return { value:rawNorm, unit:'', isResolved:false }; return { value:rawNorm * cs * 1000, unit:'µm', isResolved:true }; }
    case 'modulus_es':  { const Es = globalInputs.modulus_gpa; if (Es == null || Es <= 0) return { value:rawNorm, unit:'', isResolved:false }; return { value:rawNorm * Es, unit:'GPa', isResolved:true }; }
    case 'thermal_ks':  { const ks = globalInputs.thermal_k_wmk; if (ks == null || ks <= 0) return { value:rawNorm, unit:'', isResolved:false }; return { value:rawNorm * ks, unit:'W/mK', isResolved:true }; }
    default: return { value:rawNorm, unit:'', isResolved:false };
  }
}
function unresolveValue(physVal, normKind) {
  if (physVal == null || !isFinite(physVal) || !normKind || normKind === 'none') return physVal;
  const r = resolveValue(1, normKind);
  return r.isResolved ? physVal / r.value : physVal;
}
// "12.3 GPa" or "0.112 norm" for a metric value
function displayMetric(key, norm, digits) {
  const m = METRIC_DEFS[key];
  const r = resolveValue(norm, m.norm_kind);
  const d = digits != null ? digits : m.decimals;
  if (r.isResolved) return { v: formatVal(r.value, r.unit === 'µm' ? 0 : d), unit: r.unit };
  return { v: formatVal(norm, d), unit: m.norm_kind !== 'none' ? 'norm' : (m.unit || '') };
}
