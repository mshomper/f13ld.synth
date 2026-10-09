/* ============================================================
   F13LD.synth · 41-score-viz.js
   Score bell sparkline and per-metric sigma badges.
   ============================================================ */
'use strict';

// Traffic-light color ramp for the score sparkline dot. Anchors:
//   z=0.0 -> #22c55e (green)        z=1.5 -> #f59e0b (amber)
//   z=0.5 -> #84cc16 (lime)         z=2.0 -> #f97316 (orange)
//   z=1.0 -> #eab308 (yellow)       z>=3.0 -> #dc2626 (red)
function dotColorForZ(z) {
  const STOPS = [
    [0.0, [0x22, 0xc5, 0x5e]],
    [0.5, [0x84, 0xcc, 0x16]],
    [1.0, [0xea, 0xb3, 0x08]],
    [1.5, [0xf5, 0x9e, 0x0b]],
    [2.0, [0xf9, 0x73, 0x16]],
    [3.0, [0xdc, 0x26, 0x26]]
  ];
  const az = Math.min(Math.max(Math.abs(z) || 0, 0), 3);
  for (let i = 1; i < STOPS.length; i++) {
    if (az <= STOPS[i][0]) {
      const [z0, c0] = STOPS[i-1], [z1, c1] = STOPS[i];
      const t = (az - z0) / (z1 - z0);
      const r = Math.round(c0[0] + (c1[0]-c0[0])*t);
      const g = Math.round(c0[1] + (c1[1]-c0[1])*t);
      const b = Math.round(c0[2] + (c1[2]-c0[2])*t);
      return `rgb(${r},${g},${b})`;
    }
  }
  return 'rgb(220,38,38)';
}

// Per-metric signed sigma badge. + means predictor overshoots target, − means
// undershoots. Clamped to ±3σ for display. Border + text both colored by |z|
// using the same ramp as the bell dot, so per-metric and aggregate read alike.
function sigmaBadgeHTML(z) {
  const color = dotColorForZ(z);
  let disp;
  if (z > 3) disp = '>+3σ';
  else if (z < -3) disp = '<−3σ';
  else disp = `${z >= 0 ? '+' : ''}${z.toFixed(1)}σ`;
  return `<span class="m-z" style="color:${color};border-color:${color}">${disp}</span>`;
}

// Bell-curve sparkline SVG. Curve y=exp(-x^2/6) sampled at 13 points, x in [-3,3]
// mapped to screen [4,106]. Dot positioned at (zRms, exp(-zRms^2/6)) on the curve.
// Polyline is precomputed in viewBox units; only the dot moves with zRms.
function bellSparklineSVG(zRms) {
  const z = Math.min(Math.max(Math.abs(zRms) || 0, 0), 3);
  const xScr = 4 + (z + 3) * 17;
  const yScr = 27 - 24 * Math.exp(-(z*z)/6);
  const color = dotColorForZ(z);
  return `<svg width="110" height="30" viewBox="0 0 110 30" class="rc-bell" aria-label="match score, ${z.toFixed(2)} sigma from target">`
    + `<polyline points="4,22 12.5,19 21,15 29.5,10 38,7 46.5,4 55,3 63.5,4 72,7 80.5,10 89,15 97.5,19 106,22" fill="none" stroke="#6b7a9a" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`
    + `<circle cx="${xScr.toFixed(2)}" cy="${yScr.toFixed(2)}" r="4" fill="${color}"/>`
    + `</svg>`;
}
