/* ============================================================
   F13LD.synth · 00-config.js
   Version, endpoints and storage keys. Bump F13LD_SYNTH_VERSION and the
   header label in index.html (.fh-version) on every release.
   ============================================================ */
'use strict';

const F13LD_SYNTH_VERSION = '0.3.0';

// Model bundles. The relative path is what GitHub Pages serves; the absolute
// one lets the single-file preview build (opened outside the site) load the
// same bundle.
const WEIGHTS_URLS = family => [
  `weights/${family}.json`,
  `https://mshomper.github.io/f13ld.synth/weights/${family}.json`
];

// F13LD.vault — same Supabase project as the Vault explorer. The anon key is
// public by design.
const SUPABASE_URL = 'https://axinljpecycnvfncyhfs.supabase.co';
const SUPABASE_KEY = 'sb_publishable_DAlrNLqbUZiwkaA6wPSMIw_YUNY85LX';

const MESH_URL = 'https://mshomper.github.io/f13ld.mesh/';
const LAB_URL  = 'https://mshomper.github.io/f13ld.lab/';

const GLOBAL_INPUTS_STORAGE_KEY = 'f13ld.vault.globalInputs.v1';   // shared with F13LD.vault
const PADS_STORAGE_KEY          = 'f13ld.synth.pads.v1';
const CONNECTIVITY_STORAGE_KEY  = 'f13ld_synth_connectivity_v1';
const PRESET_STORAGE_KEY        = 'f13ld.synth.preset.v1';

// Search budget per click. Explore is split across the worker pool; the best
// distinct candidates are then refined with smaller nudges.
const SEARCH_BUDGET = { explore: 4000, refineParents: 16, refinePerParent: 150, results: 8, perSeed: 2, keepPerJob: 48 };
