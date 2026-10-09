/* ============================================================
   F13LD.synth · 00-config.js
   Version, endpoints and storage keys. Bump F13LD_SYNTH_VERSION and the
   header label in index.html (.fh-version) on every release.
   ============================================================ */
'use strict';

const F13LD_SYNTH_VERSION = '0.5.0';

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

// Search depth: a time budget per click. The search runs in rounds and
// stops early once the best eight stop improving; Stop ends it any time.
const SEARCH_DEPTHS = {
  quick: { label: 'Quick', seconds: 1.2,  patience: 2, tip: 'Quick search: about a second' },
  wide:  { label: 'Wide',  seconds: 5,    patience: 3, tip: 'Wide search: up to about 5 seconds, stops early when results settle' },
  deep:  { label: 'Deep',  seconds: 15,   patience: 6, tip: 'Deep search: up to about 15 seconds, for hard targets' }
};
const DEFAULT_DEPTH = 'wide';
const SEARCH_PER_WORKER = 600;       // designs per worker per round
const SEARCH_PER_WORKER_MAIN = 300;  // main-thread fallback (keeps the page responsive)
const REACH_Z = 1.5;                 // a result within this many σ on every target counts as reaching it
const DEPTH_STORAGE_KEY = 'f13ld.synth.depth.v1';
