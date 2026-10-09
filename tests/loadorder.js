// node tests/loadorder.js — the numbered scripts index.html loads share one
// global scope: no top-level name may be declared twice, and every file must
// parse. (Runtime references are checked by tests/page-check.py.)
'use strict';
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const files = [...html.matchAll(/<script defer src="([^"]+)"/g)].map(m => m[1]);
const seen = {}; let bad = 0;
for(const f of files){
  const src = fs.readFileSync(path.join(root, f), 'utf8');
  try { new Function(src); } catch(e){ console.log(`${f}: ${e.message}`); bad++; }
  for(const m of src.matchAll(/^(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)/gm)){
    if(seen[m[1]]){ console.log(`${m[1]} declared in ${seen[m[1]]} and ${f}`); bad++; }
    else seen[m[1]] = f;
  }
}
console.log(`load order: ${files.length} scripts, ${Object.keys(seen).length} globals, ${bad} problems`);
process.exit(bad ? 1 : 0);
