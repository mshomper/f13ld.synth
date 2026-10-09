/* ============================================================
   F13LD.synth · 03-icons.js
   Drawn icons (no glyph characters), in F13LD.lab / F13LD.sweep style:
   ICONS  40 × 40 tiles, 2.6 strokes, a neon accent node (.acc)
   GLYPHS 16 × 16 line glyphs for buttons, currentColor
   paintIcons(root) fills every [data-ico] and svg[data-gl] under root.
   ============================================================ */
'use strict';

const IC_SV = ' fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"';
const ICONS = {
  intent:   '<circle cx="20" cy="20" r="14"' + IC_SV + '/><path d="M20 3 V10 M20 30 V37 M3 20 H10 M30 20 H37"' + IC_SV + ' opacity=".7"/><circle class="acc" cx="20" cy="20" r="4"/>',
  cell:     '<path d="M20 4 L34 12 V28 L20 36 L6 28 V12 Z"' + IC_SV + '/><path d="M20 12 L27 16 V24 L20 28 L13 24 V16 Z"' + IC_SV + ' stroke-width="2" opacity=".55"/><circle class="acc" cx="20" cy="20" r="3.2"/>',
  map:      '<path d="M6 34 V6 M6 34 H34"' + IC_SV + '/><circle cx="14" cy="26" r="2" fill="currentColor"/><circle cx="20" cy="18" r="2" fill="currentColor" opacity=".6"/><circle cx="28" cy="22" r="2" fill="currentColor" opacity=".6"/><circle class="acc" cx="24" cy="12" r="3.6"/>',
  inspect:  '<circle cx="17" cy="17" r="10"' + IC_SV + '/><path d="M24.5 24.5 L34 34"' + IC_SV + '/><circle class="acc" cx="17" cy="17" r="3.4"/>',
  ranks:    '<path d="M5 35 H35"' + IC_SV + ' opacity=".6"/><path d="M8 35 V25 H15 V35 M16.5 35 V17 H23.5 V35 M25 35 V11 H32 V35"' + IC_SV + '/><circle class="acc" cx="28.5" cy="5.5" r="3.1"/>',
  sliders:  '<path d="M5 11 H9.5 M18.5 11 H35 M5 29 H21.5 M30.5 29 H35"' + IC_SV + '/><circle cx="14" cy="11" r="4.5"' + IC_SV + '/><circle class="acc" cx="26" cy="29" r="4"/>',
  material: '<path d="M20 4 L34 12 V28 L20 36 L6 28 V12 Z"' + IC_SV + '/><path d="M6 12 L20 20 L34 12 M20 20 V36"' + IC_SV + ' opacity=".7"/><circle class="acc" cx="20" cy="20" r="3.4"/>',
  model:    '<path d="M20 5 V13 M20 13 L10 23 M20 13 L30 23 M10 23 L5 33 M10 23 L15 33 M30 23 L25 33 M30 23 L35 33"' + IC_SV + '/><circle class="acc" cx="20" cy="13" r="3.4"/>'
};
const GL_S = ' fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"';
const GLYPHS = {
  play:  '<path d="M4.5 2.8 L12.8 8 L4.5 13.2 Z" fill="currentColor"/>',
  stop:  '<path d="M4 4 H12 V12 H4 Z" fill="currentColor"/>',
  chev:  '<path d="M4.5 6.2 L8 9.7 L11.5 6.2"' + GL_S + '/>',
  copy:  '<path d="M5.5 5.5 H12.5 V13 H5.5 Z"' + GL_S + '/><path d="M3.5 10.5 V3 H10"' + GL_S + '/>',
  tick:  '<path d="M3.5 8.4 L6.7 11.5 L12.5 4.8"' + GL_S + ' stroke-width="1.9"/>',
  reset: '<path d="M3.6 8 a4.4 4.4 0 1 0 1.3 -3.1 M3.4 2.6 V5.3 H6.1"' + GL_S + '/>',
  mesh:  '<path d="M8 1.8 L13.4 4.9 V11.1 L8 14.2 L2.6 11.1 V4.9 Z"' + GL_S + '/><path d="M2.6 4.9 L8 8 L13.4 4.9 M8 8 V14.2"' + GL_S + ' opacity=".6"/>',
  lab:   '<path d="M6 2 H10 M6.6 2 V6.4 L2.8 12.6 A1 1 0 0 0 3.7 14 H12.3 A1 1 0 0 0 13.2 12.6 L9.4 6.4 V2"' + GL_S + '/><path d="M4.6 10 H11.4"' + GL_S + ' opacity=".6"/>',
  queue: '<path d="M2.5 4 H10 M2.5 8 H10 M2.5 12 H7"' + GL_S + '/><path d="M12 9.5 V14.5 M9.5 12 H14.5"' + GL_S + '/>',
  ax1:   '<path d="M2 8 H14 M11.5 5.5 L14 8 L11.5 10.5"' + GL_S + '/>',
  ax2:   '<path d="M3 13 H14 M11.5 10.5 L14 13 L11.5 15.5 M3 13 V2 M0.5 4.5 L3 2 L5.5 4.5"' + GL_S + '/>',
  ax3:   '<path d="M6 10 H15 M6 10 V1 M6 10 L1 15"' + GL_S + '/><circle cx="6" cy="10" r="1.3" fill="currentColor"/>',
  numpad:'<path d="M2.5 4 H13.5 V12 H2.5 Z"' + GL_S + '/><path d="M5 7 H5.5 M7.75 7 H8.25 M10.5 7 H11 M5.5 9.6 H10.5"' + GL_S + '/>',
  pad:   '<path d="M2.5 2.5 H13.5 V13.5 H2.5 Z"' + GL_S + '/><circle cx="9.5" cy="6.5" r="1.8" fill="currentColor"/>'
};
function icon(name){ return '<svg viewBox="0 0 40 40" aria-hidden="true">' + (ICONS[name] || '') + '</svg>'; }
function glyph(name, cls){ return '<svg class="g' + (cls ? ' ' + cls : '') + '" viewBox="0 0 16 16" aria-hidden="true">' + (GLYPHS[name] || '') + '</svg>'; }
function paintIcons(root){
  (root || document).querySelectorAll('[data-ico]').forEach(e => { e.innerHTML = icon(e.dataset.ico); });
  (root || document).querySelectorAll('svg[data-gl]').forEach(e => { e.innerHTML = GLYPHS[e.dataset.gl] || ''; e.setAttribute('aria-hidden', 'true'); });
}
