/* ============================================================
   F13LD.synth · 41-strip.js
   The eight candidates as tiles: a raymarched thumbnail, rank, match and a
   match bar. Sort by best match, most confident or lightest; the numbers
   always stay the best-match rank (the same numbers as on the map).
   ============================================================ */
'use strict';

const CONF_ORDER = { high: 0, medium: 1, low: 2 };
const CONF_COLOR = { high: '#5BB892', medium: '#4FB8C9', low: '#ffa028' };

function stripOrder(){
  const idx = SYNTH.results.map((_, i) => i), R = SYNTH.results;
  if(SYNTH.sortKey === 'conf') idx.sort((a, b) => (CONF_ORDER[R[a].confidence] - CONF_ORDER[R[b].confidence]) || (R[b].score - R[a].score));
  else if(SYNTH.sortKey === 'light') idx.sort((a, b) => (R[a].metrics.volume_fraction - R[b].metrics.volume_fraction));
  return idx;
}
function renderStrip(){
  const el = document.getElementById('strip');
  if(!SYNTH.results.length){
    el.innerHTML = `<div class="strip-empty">${SYNTH.running ? 'Searching… the best eight appear here when the search settles.' : (SYNTH.emptyText || 'Set your intent, then Synthesize.')}</div>`;
    return;
  }
  el.innerHTML = stripOrder().map(i => {
    const r = SYNTH.results[i], col = dotColorForZ(r.zRms);
    const img = r.thumb ? `<img class="thumb" alt="" src="${r.thumb}">` : `<div class="thumb"></div>`;
    return `<button type="button" class="cand ${i === SYNTH.sel ? 'on' : ''}" data-i="${i}" title="#${i + 1} · ${presetDisplayLabel(r.presetKey)} · ${r.design.mode} · ${r.confidence} confidence">
      ${img}<div class="cand-h">#${i + 1}<span class="cdot" style="background:${CONF_COLOR[r.confidence]}"></span><span class="sc">${Math.round(r.score * 100)}%</span></div>
      <div class="sbar"><i style="width:${(r.score * 100).toFixed(0)}%;background:${col}"></i></div></button>`;
  }).join('');
  el.querySelectorAll('.cand').forEach(b => b.addEventListener('click', () => selectCandidate(+b.dataset.i)));
}
function wireSort(){
  document.querySelectorAll('#sortSeg button').forEach(b => b.addEventListener('click', () => {
    SYNTH.sortKey = b.dataset.s;
    document.querySelectorAll('#sortSeg button').forEach(x => x.classList.toggle('on', x === b));
    renderStrip();
  }));
}
// Thumbnails render one at a time after a search; each tile fills in as it lands.
function renderThumbnails(){
  const list = SYNTH.results;
  synthThumbnails(list.map(r => r.design), 168, (i, url) => {
    if(SYNTH.results !== list || !url) return;
    list[i].thumb = url;
    const b = document.querySelector(`.cand[data-i="${i}"] .thumb`);
    if(b){ if(b.tagName === 'IMG') b.src = url; else b.outerHTML = `<img class="thumb" alt="" src="${url}">`; }
  });
}
