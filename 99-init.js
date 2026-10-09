/* ============================================================
   F13LD.synth · 99-init.js
   Boot: restore state, build the UI, load the model and Vault counts.
   ============================================================ */
'use strict';

async function boot() {
  paintIcons(document);
  loadGlobalInputs();
  loadPadState();
  loadConnectivityState();
  loadPresetState();
  buildMaterialSelect();
  wireConnectivity();
  renderPads();
  wireMap();
  wireSort();
  wireInspector();
  wireHandoff();
  wireHeader();
  wireDrawer();
  buildDepthSeg();
  statusInit();
  wireRun();
  renderDockTags();
  renderStrip();
  renderInspector();
  renderModelChip();
  const first = firstActivePad();
  if (first) mapSetTab(PAD_DEFS.indexOf(first)); else mapRender();

  await Promise.allSettled([
    Vault.loadCounts().catch(e => { console.warn('[F13LD.synth] Vault unreachable:', e.message); }).then(renderModelChip),
    Predictor.loadFamily('tpms').then(() => {
      renderModelChip();
      if (Predictor.loaded) {
        buildPresetDropdown();
        document.getElementById('runBtn').disabled = false;
        const W = Predictor.pool ? Predictor.pool.size : 0;
        statusSet('', `ready · ${Predictor.seedTable.seeds.length} training designs · ${W ? W + ' workers' : 'page thread'}`);
      } else {
        statusSet('warn', 'model unavailable');
      }
      mapRender();
    })
  ]);
  renderModelChip();
}
boot();
