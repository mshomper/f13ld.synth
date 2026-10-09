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
  volume_fraction:    { label:'Volume fraction',    labelAbs:'Volume fraction',    unit:'%',     decimals:1, min:2,    max:30,    norm_kind:'none' },
  ex_norm:            { label:'Stiffness Ex / Es',  labelAbs:'Stiffness Ex',       unit:'GPa',   decimals:2, min:0.005,max:0.5,   norm_kind:'modulus_es' },
  ey_norm:            { label:'Stiffness Ey / Es',  labelAbs:'Stiffness Ey',       unit:'GPa',   decimals:2, min:0.005,max:0.5,   norm_kind:'modulus_es' },
  ez_norm:            { label:'Stiffness Ez / Es',  labelAbs:'Stiffness Ez',       unit:'GPa',   decimals:2, min:0.005,max:0.5,   norm_kind:'modulus_es' },
  anisotropy:         { label:'Anisotropy',         labelAbs:'Anisotropy',         unit:'×',     decimals:2, min:1,    max:5,     norm_kind:'none' },
  pore_size_p50_norm: { label:'Median pore φ / cell', labelAbs:'Median pore φ',      unit:'µm',    decimals:2, min:0.04, max:0.5,   norm_kind:'cell_length' },
  pore_size_cv:       { label:'Pore size CV',       labelAbs:'Pore size CV',       unit:'',      decimals:2, min:0.4,  max:0.9,   norm_kind:'none' },
  keff_avg_norm:      { label:'Thermal cond / ks',  labelAbs:'Thermal cond avg',   unit:'W/mK',  decimals:3, min:0.005,max:0.4,   norm_kind:'thermal_ks' },
  surface_complexity: { label:'Surface complexity', labelAbs:'Surface complexity', unit:'',      decimals:2, min:1,    max:1.5,   norm_kind:'none' },
  directionality:     { label:'Directionality',     labelAbs:'Axes percolated',    unit:'/3',    decimals:2, min:0.33, max:1.0,   norm_kind:'none' },
  // Shear moduli — shown on cards only when the loaded bundle predicts them
  // (bundles trained on F13LD.sweep v0.24+ data, which reports shear).
  gxy_norm:           { label:'Shear Gxy / Es',     labelAbs:'Shear Gxy',          unit:'GPa',   decimals:2, min:0.001,max:0.2,   norm_kind:'modulus_es' },
  gxz_norm:           { label:'Shear Gxz / Es',     labelAbs:'Shear Gxz',          unit:'GPa',   decimals:2, min:0.001,max:0.2,   norm_kind:'modulus_es' },
  gyz_norm:           { label:'Shear Gyz / Es',     labelAbs:'Shear Gyz',          unit:'GPa',   decimals:2, min:0.001,max:0.2,   norm_kind:'modulus_es' },
};

// Three pads, each defining an X/Y pair plus optional auxiliary metrics
// computed from pad position (e.g. pad 1 sets Ex≈Ey≈Ez to its X value).
const PAD_DEFS = [
  {
    id: 'mech_pore', title: 'Mechanics × Pore',
    xMetric: 'ex_norm',          xLabel: 'Stiffness',
    yMetric: 'pore_size_p50_norm',   yLabel: 'Pore size',
    cornerLabels: { tl:'compliant · large pores', tr:'stiff · large pores', bl:'compliant · tight pores', br:'stiff · tight pores' },
    // When pad active, drive ey/ez along with ex (sets stiffness as a whole, not just one axis)
    coupledMetrics: ['ey_norm', 'ez_norm'],
    defaultPos: { x: 0.5, y: 0.5 },
    defaultMode: 'require'
  },
  {
    id: 'aniso_dist', title: 'Anisotropy × Pore Distribution',
    xMetric: 'anisotropy',       xLabel: 'Anisotropy',
    yMetric: 'pore_size_cv',     yLabel: 'Pore size CV',
    cornerLabels: { tl:'isotropic · varied', tr:'directional · varied', bl:'isotropic · uniform', br:'directional · uniform' },
    defaultPos: { x: 0.0, y: 0.0 },  // bottom-left → isotropic + uniform (foam-like)
    defaultMode: 'off'
  },
  {
    id: 'mass_thermal', title: 'Mass Efficiency × Thermal',
    xMetric: 'volume_fraction',  xLabel: 'Volume fraction',
    yMetric: 'keff_avg_norm',    yLabel: 'Thermal cond.',
    cornerLabels: { tl:'lightweight · conductive', tr:'dense · conductive', bl:'lightweight · insulating', br:'dense · insulating' },
    defaultPos: { x: 0.3, y: 0.3 },
    defaultMode: 'off'  // Hidden value-add — toggled in only when thermal matters
  }
];

// Mode → weight contribution
const MODE_WEIGHTS = { off: 0.0, prefer: 0.5, require: 1.0 };
