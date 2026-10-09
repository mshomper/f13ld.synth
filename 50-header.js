/* ============================================================
   F13LD.synth · 50-header.js
   Model chip in the header, and the model / search columns of the
   Configure drawer: per-metric fit, training lineage against F13LD.vault.
   ============================================================ */
'use strict';

const R2_LABELS = { volume_fraction: 'Volume fraction', ex_norm: 'Stiffness Ex', ey_norm: 'Stiffness Ey', ez_norm: 'Stiffness Ez',
  anisotropy: 'Anisotropy', pore_size_p50_norm: 'Median pore', pore_size_cv: 'Pore size CV', keff_avg_norm: 'Thermal cond.',
  surface_complexity: 'Surface complexity', directionality: 'Connected axes', gxy_norm: 'Shear Gxy', gxz_norm: 'Shear Gxz', gyz_norm: 'Shear Gyz' };

function vaultNewSince(){
  if(!vaultCounts || !Predictor.meta) return null;
  return Vault.countNewSince(document.getElementById('familySel').value, Predictor.meta.trained_at);
}
function renderModelChip(){
  const tx = document.getElementById('modelChipTx');
  if(Predictor.loading){ tx.textContent = 'loading model…'; return; }
  if(!Predictor.loaded){ tx.innerHTML = `<b>model</b> <span class="err">unavailable${Predictor.loadError ? ' · ' + Predictor.loadError.slice(0, 40) : ''}</span>`; return; }
  const m = Predictor.meta, r2 = (Predictor.bundle.eval && Predictor.bundle.eval.mean_r2) || 0;
  const fresh = vaultNewSince();
  tx.innerHTML = `<b>${m.family} model</b><span class="mc-d"> trained ${String(m.trained_at || '').slice(0, 10)} · ${m.n_valid.toLocaleString()} designs ·</span> R² ${r2.toFixed(2)}` +
    (fresh == null ? '' : fresh > 200 ? ` · <span class="fresh">+${fresh} new in Vault</span>` : ` · +${fresh} new in Vault`);
}
function renderModelDrawer(){
  const list = document.getElementById('r2List'), lin = document.getElementById('lineage');
  if(!Predictor.loaded){ list.innerHTML = '<p class="dr-note">No model loaded.</p>'; lin.innerHTML = ''; return; }
  const R2 = Predictor.metricsR2;
  list.innerHTML = Object.keys(R2).sort((a, b) => R2[b] - R2[a]).map(k => {
    const r = R2[k], col = r >= 0.7 ? '#5BB892' : r >= 0.4 ? '#4FB8C9' : '#ffa028';
    return `<div class="r2row"><span>${R2_LABELS[k] || k}</span><span class="bar"><i style="width:${Math.max(2, r * 100).toFixed(0)}%;background:${col}"></i></span><b>${r.toFixed(2)}</b></div>`;
  }).join('');
  const m = Predictor.meta, fresh = vaultNewSince();
  lin.innerHTML = `<span class="tag">trained ${String(m.trained_at || '').slice(0, 10)}</span><span class="tag">${m.n_valid.toLocaleString()} valid designs</span>` +
    `<span class="tag">bundle v${m.version}</span>` +
    (fresh == null ? '<span class="tag">Vault offline</span>' : `<span class="tag ${fresh > 200 ? 'warn' : ''}">+${fresh} since${fresh > 200 ? ' · retrain warranted' : ''}</span>`);
  const W = Predictor.pool ? Predictor.pool.size : 0;
  document.getElementById('searchNote').innerHTML =
    `Each click runs rounds: new designs are grown from the best so far and from the best in every filled part of the map, ` +
    `with the step narrowing each round. It stops at the depth's time limit, when the best eight stop improving, or when you press Stop. ` +
    `Candidates far from the training data are ranked down.<br><br>` +
    `${Predictor.seedTable.seeds.length} training designs to grow from · ${W ? W + ' search workers' : 'searching on the page thread'}.`;
}
function wireHeader(){
  document.getElementById('modelChip').addEventListener('click', () => toggleDrawer(true));
}
