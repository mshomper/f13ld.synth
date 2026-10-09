/* ============================================================
   F13LD.synth · 50-status.js
   Model status bar: model version, fit and Vault lineage.
   ============================================================ */
'use strict';

// ============================================================
// MODEL STATUS BAR
// ============================================================
function refreshModelStatus() {
  const family = document.getElementById('familySel').value;
  const status = document.getElementById('modelStatus');
  if (Predictor.loading) {
    status.innerHTML = `<div class="ms-item"><span class="ms-lbl">predictor</span><span class="ms-val"><span class="spinner"></span>loading ${family}…</span></div><div class="ms-spacer"></div>`;
    return;
  }
  if (!Predictor.loaded) {
    const reason = Predictor.loadError ? Predictor.loadError.slice(0, 30) : 'not yet trained';
    status.innerHTML = `<div class="ms-item empty"><span class="ms-lbl">predictor</span><span class="ms-val">${family}: ${reason}</span></div><div class="ms-item warn"><span class="ms-lbl">status</span><span class="ms-val">model unavailable</span></div><div class="ms-spacer"></div>`;
    return;
  }
  const meta = Predictor.meta;
  const date = meta.trained_at ? meta.trained_at.slice(0, 10) : 'unknown';
  const meanR2 = (Predictor.bundle.eval && Predictor.bundle.eval.mean_r2) || 0;
  const repoUrl = `https://github.com/mshomper/f13ld.synth/blob/main/weights/${meta.family}.json`;

  // Vault lineage: total community data, trained-on subset, new-since-training delta
  let vaultBlocks = '';
  if (vaultCounts) {
    const totalForFam = vaultCounts.byFamily[family] || 0;
    const newSince = Vault.countNewSince(family, meta.trained_at);
    const stale = newSince > 200;
    vaultBlocks = `
      <div class="ms-item"><span class="ms-lbl">vault</span><span class="ms-val">${totalForFam} ${family} designs</span></div>
      <div class="ms-item ${stale ? 'warn' : ''}"><span class="ms-lbl">since training</span><span class="ms-val">+${newSince}${stale ? ' · retrain warranted' : ''}</span></div>`;
  } else {
    vaultBlocks = `<div class="ms-item empty"><span class="ms-lbl">vault</span><span class="ms-val">connecting…</span></div>`;
  }

  status.innerHTML = `
    <div class="ms-item ok"><span class="ms-lbl">model</span><span class="ms-val"><a href="${repoUrl}" target="_blank">${meta.family} · v${meta.version}</a></span></div>
    <div class="ms-item"><span class="ms-lbl">trained</span><span class="ms-val">${date} · ${meta.n_valid} valid · R² ${meanR2.toFixed(2)}</span></div>
    ${vaultBlocks}
    <div class="ms-spacer"></div>`;
}

function onFamilyChange() { refreshModelStatus(); }
