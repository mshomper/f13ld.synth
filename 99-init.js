/* ============================================================
   F13LD.synth · 99-init.js
   Boot.
   ============================================================ */
'use strict';

// ============================================================
// BOOT
// ============================================================
async function boot() {
  loadGlobalInputs();
  // Fresh browser: the default material is selected but its values were never
  // filled in, so nothing resolved to physical units. Fill them once.
  if (MATERIAL_PRESETS[globalInputs.material_id] && globalInputs.modulus_gpa == null) applyMaterialPreset(globalInputs.material_id);
  loadPadState();
  loadConnectivityState();
  buildMaterialDropdown();
  syncMaterialCardToInputs();
  wireMaterialInputs();
  buildPadRack();
  syncConnectivityToggleUI();

  loadPresetState();
  refreshModelStatus();

  // Load Vault and Predictor in parallel — both are needed for full status display.
  await Promise.allSettled([
    Vault.loadCounts().catch(e => { console.warn('[F13LD.synth] Vault unreachable:', e.message); }),
    Predictor.loadFamily('tpms').then(() => { if (Predictor.loaded) buildPresetDropdown(); refreshModelStatus(); })
  ]);

  refreshModelStatus();
}
boot();
