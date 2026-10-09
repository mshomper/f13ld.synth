/* ============================================================
   F13LD.synth · 40-results.js
   Result cards and hand-offs to F13LD.mesh and F13LD.lab.
   ============================================================ */
'use strict';

const lastResults = { synth: [] };

function renderResults({ results, reason, stats, seconds, workers }, targets) {
  lastResults.synth = results;
  const meta = document.getElementById('resultsMeta');
  const body = document.getElementById('resultsBody');
  if (reason === 'no-model') {
    meta.innerHTML = 'predictor pending';
    body.innerHTML = `<div class="engine-empty predictor-pending">Predictor not yet loaded.<br><span class="empty-detail">Once weights/tpms.json ships in the repo, this section activates.</span></div>`;
    return;
  }
  if (results.length === 0) {
    meta.innerHTML = 'no candidates survived the filters';
    const why = stats && stats.connectivity > stats.scanned * 0.9
      ? 'Almost every design was outside the chosen connectivity. Try another axis count or preset.'
      : 'No candidates met the validity threshold. Try widening pad targets or another preset.';
    body.innerHTML = `<div class="engine-empty">${why}</div>`;
    return;
  }
  meta.innerHTML = `<b>${results.length} candidates</b> · ranked by score × validity × confidence`;
  body.innerHTML = '';
  const list = document.createElement('div'); list.className = 'results-list';
  results.forEach((r, i) => list.appendChild(buildResultCard(r, i+1, targets)));
  body.appendChild(list);
}

// Small drawn markers (no glyph characters): filled ring for targeted metrics.
const MARK_TARGETED = '<svg width="10" height="10" viewBox="0 0 10 10" aria-label="targeted"><circle cx="5" cy="5" r="3.6" fill="none" stroke="#c8f542" stroke-width="1.3"/><circle cx="5" cy="5" r="1.6" fill="#c8f542"/></svg>';
const MARK_PLAIN    = '<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="1.3" fill="#5a6680"/></svg>';
const CONF_TITLE = {
  high:   'The forest\'s trees agree on this design about as well as on its training data.',
  medium: 'The trees disagree more than usual. Treat the predictions as direction.',
  low:    'The trees disagree strongly: this design is far from the training data. Verify in F13LD.lab before relying on it.'
};

function buildResultCard(r, rank, targets) {
  const card = document.createElement('div');
  card.className = 'result-card';
  const presetTag = `<span class="rc-source-tag preset" title="Grown from a training seed of this preset">${escapeHtml(presetDisplayLabel(r.presetKey))}</span>`;
  const confTag = `<span class="rc-source-tag conf-${r.confidence}" title="${CONF_TITLE[r.confidence]} Spread ratio ${r.spreadRatio.toFixed(2)}.">${r.confidence} confidence</span>`;

  let metricsHTML = '';
  for (const key of Predictor.shownMetrics()) {
    const m = METRIC_DEFS[key];
    const pNorm = r.metrics[key];
    const tNorm = targets[key];
    const targeted = tNorm != null;
    const pRes = resolveValue(pNorm, m.norm_kind);
    const r2 = Predictor.metricsR2 ? Predictor.metricsR2[key] : null;
    const r2Class = r2 == null ? '' : (r2 >= 0.7 ? 'high' : r2 >= 0.4 ? 'med' : 'low');
    const r2Tag = r2 != null ? `<span class="m-r2 ${r2Class}">R² ${r2.toFixed(2)}</span>` : '';
    const zVal = (r.zPerMetric && r.zPerMetric[key] != null) ? r.zPerMetric[key] : null;
    const zTag = zVal != null ? sigmaBadgeHTML(zVal) : '';
    const rough = r2 != null && r2 < 0.5 ? 'rough' : '';
    const label = pRes.isResolved ? m.labelAbs : m.label;
    const predDisplay = pNorm == null ? '—' : (pRes.isResolved ? formatVal(pRes.value, m.decimals) : formatVal(pNorm, m.decimals));
    const unitDisplay = pRes.isResolved ? pRes.unit : (m.norm_kind !== 'none' ? 'norm' : (m.unit || ''));
    metricsHTML += `<div class="rc-metric ${rough}"><span class="m-marker">${targeted ? MARK_TARGETED : MARK_PLAIN}</span><span class="m-name">${label} ${r2Tag}</span><span class="m-z-col">${zTag}</span><span class="m-vals"><span class="m-pred">${predDisplay}</span>${unitDisplay?`<span class="m-units">${unitDisplay}</span>`:''}</span></div>`;
  }

  card.innerHTML = `
    <div class="rc-head">
      <div class="rc-head-left">
        <span class="rc-rank-num">#${rank}</span>
        ${presetTag}
        ${confTag}
      </div>
      <div class="rc-scores">
        <div class="rc-score-block"><span class="lbl">score</span>${bellSparklineSVG(r.zRms)}</div>
        <div class="rc-score-block"><span class="lbl">validity</span><span class="val ${r.validity < 0.85 ? 'warn' : 'valid'}">${(r.validity*100).toFixed(0)}%</span></div>
      </div>
    </div>
    <div class="rc-metrics">${metricsHTML}</div>
    <div class="rc-actions">
      <button class="rc-btn rc-btn-mesh" onclick="handoffToMesh(${rank})">Open in Mesh</button>
      <button class="rc-btn rc-btn-lab" onclick="handoffToLab(${rank})" title="Run a full solve on this exact recipe">Open in Lab</button>
      <button class="rc-btn rc-btn-vault" onclick="saveSynthToVault(${rank})">Save to Vault</button>
      <button class="rc-btn rc-btn-copy" onclick="copyRecipe(${rank}, this)" title="Copy the recipe JSON">Copy recipe</button>
    </div>`;
  return card;
}

function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]); }

function handoffToMesh(rank) {
  const r = lastResults.synth[rank - 1];
  if (!r || !r.recipe) { alert('No recipe available.'); return; }
  // Match F13LD.mesh's URL ingest format exactly: plain encodeURIComponent of
  // JSON.stringify(recipe). NOT base64-url. Mesh decodes via:
  //   JSON.parse(decodeURIComponent(searchParams.get('r')))
  try {
    const json = JSON.stringify(r.recipe);
    const url = `${MESH_URL}?r=${encodeURIComponent(json)}`;
    window.open(url, '_blank');
  } catch (e) {
    console.error('Mesh handoff failed:', e);
    alert('Could not encode recipe: ' + e.message);
  }
}

// F13LD.lab (v0.14.0+) reads the same recipe after "#r=", as F13LD.tpms sends it.
function handoffToLab(rank) {
  const r = lastResults.synth[rank - 1];
  if (!r || !r.recipe) { alert('No recipe available.'); return; }
  window.open(`${LAB_URL}#r=${encodeURIComponent(JSON.stringify(r.recipe))}`, '_blank');
}

function copyRecipe(rank, btn) {
  const r = lastResults.synth[rank - 1];
  if (!r) return;
  const txt = JSON.stringify(r.recipe, null, 2);
  const done = () => { const o = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => btn.textContent = o, 1400); };
  if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done).catch(() => alert(txt));
  else alert(txt);
}

function saveSynthToVault(rank) {
  const r = lastResults.synth[rank - 1];
  if (!r) return;
  alert(`Synth result #${rank} would be ingested into F13LD.vault as a candidate.\n\nScore: z = ${r.zRms.toFixed(2)}σ (${(r.score*100).toFixed(0)}%)\nValidity: ${(r.validity*100).toFixed(0)}%\nMode: ${r.recipe.geometry.mode}\n${r.recipe.surface.terms.length} active terms`);
}
