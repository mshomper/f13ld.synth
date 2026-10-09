// node tools/sync-materials.js [path to f13ld.lab]
// Writes 02-materials.js from F13LD.lab's AM material library
// (15c-materials.js), keeping only what Synth shows: name, condition,
// family, modulus, density, thermal conductivity, yield.
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const labDir = process.argv[2] || path.join(__dirname, '../../f13ld.lab');
const src = fs.readFileSync(path.join(labDir, '15c-materials.js'), 'utf8');
const ctx = {}; vm.runInNewContext(src + '\nthis.M = F13LD_MATERIALS;', ctx);
const rows = ctx.M.filter(m => m && m.id && m.Es_MPa).map(m => ({
  id: m.id, name: m.name, condition: m.condition, process: m.process, group: m.family,
  E: +(m.Es_MPa / 1000).toFixed(1), rho: +(m.rho_kgm3 / 1000).toFixed(2),
  k: m.ks_WmK == null ? null : +m.ks_WmK, sigY: m.sigY0_MPa == null ? null : +m.sigY0_MPa
}));
const out = `/* ============================================================
   F13LD.synth · 02-materials.js   (generated — do not edit by hand)
   From F13LD.lab's AM material library (15c-materials.js, reviewed by
   Matt 2026-09-29). Regenerate: node tools/sync-materials.js ../f13ld.lab
   E GPa · rho g/cc · k W/mK (null: no data) · sigY MPa
   ============================================================ */
'use strict';
const SYNTH_MATERIALS = ${JSON.stringify(rows, null, 0).replace(/\},\{/g, '},\n  {').replace(/^\[\{/, '[\n  {').replace(/\}\]$/, '}\n]')};
const SYNTH_DEFAULT_MATERIAL = 'ti64-g5-lpbf-hip';   // F13LD.lab's default
`;
fs.writeFileSync(path.join(__dirname, '..', '02-materials.js'), out);
console.log(`wrote 02-materials.js (${rows.length} materials)`);
