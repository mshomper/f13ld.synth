/* ============================================================
   F13LD.synth · 30-material.js
   Material card (shared with F13LD.vault through localStorage) and
   normalized ↔ physical unit conversion for display.
   ============================================================ */
'use strict';

function formatVal(v, d) { if (v == null || !isFinite(v)) return '—'; return v.toFixed(d ?? 2); }

let globalInputs = { cell_size_mm:null, material_id:'ti6al4v', modulus_gpa:null, density_gcc:null, thermal_k_wmk:null, ref_stress_mpa:null };

// ============================================================
// MATERIAL CARD — mirrors Vault's globalInputs (shared localStorage)
// ============================================================
function loadGlobalInputs() {
  try {
    const saved = JSON.parse(localStorage.getItem(GLOBAL_INPUTS_STORAGE_KEY) || 'null');
    if (saved && typeof saved === 'object') Object.keys(globalInputs).forEach(k => { if (k in saved) globalInputs[k] = saved[k]; });
  } catch(e) {}
}
function persistGlobalInputs() { try { localStorage.setItem(GLOBAL_INPUTS_STORAGE_KEY, JSON.stringify(globalInputs)); } catch(e) {} }
function applyMaterialPreset(matId) {
  const p = MATERIAL_PRESETS[matId]; if (!p) return;
  globalInputs.material_id = matId; globalInputs.modulus_gpa = p.E; globalInputs.density_gcc = p.rho; globalInputs.thermal_k_wmk = p.k;
  persistGlobalInputs(); syncMaterialCardToInputs();
}
function syncMaterialCardToInputs() {
  document.getElementById('ref_modulus_gpa').value   = globalInputs.modulus_gpa   ?? '';
  document.getElementById('ref_density_gcc').value   = globalInputs.density_gcc   ?? '';
  document.getElementById('ref_thermal_k_wmk').value = globalInputs.thermal_k_wmk ?? '';
  document.getElementById('ref_stress_mpa').value    = globalInputs.ref_stress_mpa ?? '';
  document.getElementById('ref_cell_size_mm').value  = globalInputs.cell_size_mm  ?? '';
  document.getElementById('materialSel').value       = globalInputs.material_id   ?? 'custom';
}
function buildMaterialDropdown() {
  const sel = document.getElementById('materialSel'); sel.innerHTML = '';
  const groups = {};
  Object.entries(MATERIAL_PRESETS).forEach(([k,v]) => { (groups[v.group] ||= []).push([k,v]); });
  for (const [g, items] of Object.entries(groups)) {
    const og = document.createElement('optgroup'); og.label = g;
    for (const [k,v] of items) { const o = document.createElement('option'); o.value = k; o.textContent = v.label; og.appendChild(o); }
    sel.appendChild(og);
  }
  const cu = document.createElement('option'); cu.value = 'custom'; cu.textContent = 'Custom…'; sel.appendChild(cu);
}
function onMaterialChange() {
  const m = document.getElementById('materialSel').value;
  if (m === 'custom') { globalInputs.material_id = 'custom'; persistGlobalInputs(); }
  else applyMaterialPreset(m);
}
function clearMaterialCard() {
  globalInputs.modulus_gpa = null; globalInputs.density_gcc = null; globalInputs.thermal_k_wmk = null;
  globalInputs.ref_stress_mpa = null; globalInputs.cell_size_mm = null; globalInputs.material_id = 'custom';
  persistGlobalInputs(); syncMaterialCardToInputs();
}
function wireMaterialInputs() {
  const map = { ref_modulus_gpa:'modulus_gpa', ref_density_gcc:'density_gcc', ref_thermal_k_wmk:'thermal_k_wmk', ref_stress_mpa:'ref_stress_mpa', ref_cell_size_mm:'cell_size_mm' };
  Object.entries(map).forEach(([id, fld]) => {
    document.getElementById(id).addEventListener('input', e => {
      globalInputs[fld] = e.target.value === '' ? null : (parseFloat(e.target.value) || null);
      const m = matchMaterialPreset(); globalInputs.material_id = m || 'custom';
      document.getElementById('materialSel').value = globalInputs.material_id;
      persistGlobalInputs();
    });
  });
}
function matchMaterialPreset() {
  for (const [k,p] of Object.entries(MATERIAL_PRESETS))
    if (p.E === globalInputs.modulus_gpa && p.rho === globalInputs.density_gcc && p.k === globalInputs.thermal_k_wmk) return k;
  return null;
}

// Resolution helpers (ported from Vault) — convert normalized↔physical at display
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
  switch (normKind) {
    case 'cell_length': { const cs = globalInputs.cell_size_mm; if (cs == null || cs <= 0) return physVal; return physVal / (cs * 1000); }
    case 'modulus_es':  { const Es = globalInputs.modulus_gpa; if (Es == null || Es <= 0) return physVal; return physVal / Es; }
    case 'thermal_ks':  { const ks = globalInputs.thermal_k_wmk; if (ks == null || ks <= 0) return physVal; return physVal / ks; }
    default: return physVal;
  }
}
