/* ============================================================
   F13LD.synth · 01-defs.js
   Material presets, metric definitions and the three design-intent pads.
   ============================================================ */
'use strict';

// Material library — mirrors Vault
const MATERIAL_PRESETS = {
  ti6al4v:   { label:'Ti-6Al-4V',     group:'Titanium',    E:110, rho:4.43, k:7.0 },
  ss316l:    { label:'316L SS',       group:'Stainless',   E:195, rho:8.00, k:14  },
  ss174ph:   { label:'17-4PH SS',     group:'Stainless',   E:197, rho:7.80, k:18  },
  h13:       { label:'H13 Tool Steel',group:'Tool steel',  E:210, rho:7.80, k:28.6},
  al6061:    { label:'Al 6061',       group:'Aluminum',    E:69,  rho:2.70, k:167 },
  alsi10mg:  { label:'AlSi10Mg',      group:'Aluminum',    E:70,  rho:2.67, k:130 },
  in718:     { label:'Inconel 718',   group:'Nickel',      E:200, rho:8.19, k:11.4},
  peek:      { label:'PEEK',          group:'Polymer',     E:3.6, rho:1.32, k:0.25},
  pekk:      { label:'PEKK',          group:'Polymer',     E:4.5, rho:1.30, k:0.22},
};

// Output metrics — the 9 things the predictor produces.
// surface_complexity is dropped from user controls (per Matt) but still displayed on cards.
const METRIC_DEFS = {
  volume_fraction:    { label:'Volume fraction',    labelAbs:'Volume fraction',    unit:'%',     decimals:1, min:2,    max:60,    norm_kind:'none' },
  ex_norm:            { label:'Stiffness Ex / Es',  labelAbs:'Stiffness Ex',       unit:'GPa',   decimals:2, min:0.005,max:0.5,   norm_kind:'modulus_es' },
  ey_norm:            { label:'Stiffness Ey / Es',  labelAbs:'Stiffness Ey',       unit:'GPa',   decimals:2, min:0.005,max:0.5,   norm_kind:'modulus_es' },
  ez_norm:            { label:'Stiffness Ez / Es',  labelAbs:'Stiffness Ez',       unit:'GPa',   decimals:2, min:0.005,max:0.5,   norm_kind:'modulus_es' },
  anisotropy:         { label:'Anisotropy',         labelAbs:'Anisotropy',         unit:'×',     decimals:2, min:1,    max:5,     norm_kind:'none' },
  pore_size_p50_norm: { label:'Median pore φ / cell', labelAbs:'Median pore φ',      unit:'µm',    decimals:2, min:0.04, max:0.5,   norm_kind:'cell_length' },
  pore_size_cv:       { label:'Pore size CV',       labelAbs:'Pore size CV',       unit:'',      decimals:2, min:0.4,  max:1.4,   norm_kind:'none' },
  keff_avg_norm:      { label:'Thermal cond / ks',  labelAbs:'Thermal cond avg',   unit:'W/mK',  decimals:3, min:0.005,max:0.4,   norm_kind:'thermal_ks' },
  surface_complexity: { label:'Surface complexity', labelAbs:'Surface complexity', unit:'',      decimals:2, min:1,    max:1.5,   norm_kind:'none' },
  directionality:     { label:'Directionality',     labelAbs:'Axes percolated',    unit:'/3',    decimals:2, min:0.33, max:1.0,   norm_kind:'none' },
  // Shear moduli — shown on cards only when the loaded bundle predicts them
  // (bundles trained on F13LD.sweep v0.24+ data, which reports shear).
  gxy_norm:           { label:'Shear Gxy / Es',     labelAbs:'Shear Gxy',          unit:'GPa',   decimals:2, min:0.001,max:0.2,   norm_kind:'modulus_es' },
  gxz_norm:           { label:'Shear Gxz / Es',     labelAbs:'Shear Gxz',          unit:'GPa',   decimals:2, min:0.001,max:0.2,   norm_kind:'modulus_es' },
  gyz_norm:           { label:'Shear Gyz / Es',     labelAbs:'Shear Gyz',          unit:'GPa',   decimals:2, min:0.001,max:0.2,   norm_kind:'modulus_es' },
  // Derived (12-search-core addDerived): built from the predictions above.
  stiff_main:         { label:'Main-axis stiffness / Es', labelAbs:'Main-axis stiffness', unit:'GPa', decimals:2, min:0.005, max:0.5, norm_kind:'modulus_es', derived:true },
  stiff_ratio:        { label:'Off-axis share',     labelAbs:'Off-axis share',     unit:'%',     decimals:0, min:10,   max:100,   norm_kind:'none', derived:true },
  porosity:           { label:'Porosity',           labelAbs:'Porosity',           unit:'%',     decimals:1, min:40,   max:98,    norm_kind:'none', derived:true },
};

// Result-map axis ranges (normalized units). Fixed, so the map does not
// rescale while a search fills it; points beyond the edge sit on the edge.
const MAP_RANGES = {
  ex_norm: [0, 0.4], pore_size_p50_norm: [0, 0.8],
  stiff_main: [0, 0.55], stiff_ratio: [0, 100],
  porosity: [45, 100], pore_size_cv: [0.4, 1.5]
};

// Three pads, each defining an X/Y pair plus optional auxiliary metrics
// computed from pad position (e.g. pad 1 sets Ex≈Ey≈Ez to its X value).
// v0.5.0 (Matt, 2026-10-09): pads follow the three things that vary on
// their own in the training data — overall stiffness/density (pad 1),
// direction (pad 2) and pore uniformity (pad 3). Anisotropy (fit 0.11) and
// thermal (follows volume fraction at 0.97) are no longer pad axes; both
// are still shown as predictions.
const PAD_DEFS = [
  {
    id: 'mech_pore', title: 'Mechanics × Pore',
    xMetric: 'ex_norm',          xLabel: 'Stiffness',
    yMetric: 'pore_size_p50_norm',   yLabel: 'Pore size',
    cornerLabels: { tl:'compliant · large pores', tr:'stiff · large pores', bl:'compliant · tight pores', br:'stiff · tight pores' },
    // When pad active, drive ey/ez along with ex (sets stiffness as a whole, not just one axis)
    coupledMetrics: ['ey_norm', 'ez_norm'],
    // Pad range (overrides METRIC_DEFS min/max). Even stiffness past ~0.2 Es
    // is out of reach for every confident design in the 2026-10-09 data.
    xRange: [0.005, 0.3],
    defaultPos: { x: 0.45, y: 0.5 },
    defaultMode: 'require'
  },
  {
    id: 'direction', title: 'Main axis × Off-axis',
    xMetric: 'stiff_main',       xLabel: 'Main axis',
    yMetric: 'stiff_ratio',      yLabel: 'Off-axis',
    cornerLabels: { tl:'compliant · even', tr:'stiff · even', bl:'compliant · one-axis', br:'stiff · one-axis' },
    // When on, its stiffness targets replace pad 1's (pad 1 then sets pore size only)
    overridesStiffness: true,
    defaultPos: { x: 0.5, y: 0.6 },
    defaultMode: 'off'
  },
  {
    id: 'porosity_pores', title: 'Porosity × Pore spread',
    xMetric: 'porosity',         xLabel: 'Porosity',
    yMetric: 'pore_size_cv',     yLabel: 'Pore spread',
    xRange: [50, 99], yRange: [0.5, 1.4],    // where the training designs sit (p2–p98)
    cornerLabels: { tl:'dense · mixed pores', tr:'open · mixed pores', bl:'dense · even pores', br:'open · even pores' },
    defaultPos: { x: 0.6, y: 0.2 },
    defaultMode: 'off'
  }
];

// Mode → weight contribution
const MODE_WEIGHTS = { off: 0.0, prefer: 0.5, require: 1.0 };
