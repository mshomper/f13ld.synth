// node tools/build-single.js [out.html]
// Builds one self-contained HTML file from index.html: stylesheet and the
// numbered scripts inlined, the search worker started from a Blob. For
// click-testing outside the site (e.g. opening the file in the Claude app).
// The model loads from the live site, since the bundle is too big to inline.
'use strict';
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..');
const out = process.argv[2] || path.join(root, 'dist', 'f13ld-synth-preview.html');
const read = f => fs.readFileSync(path.join(root, f), 'utf8');
const safe = s => s.replace(/<\/script/gi, '<\\/script');

let html = read('index.html');
html = html.replace('<link rel="stylesheet" href="synth.css">', () => `<style>\n${read('synth.css')}\n</style>`);
const files = [...html.matchAll(/<script defer src="([^"]+)"><\/script>\n?/g)];
const workerSrc = ['10-encoding.js', '11-forest.js', '12-search-core.js', 'worker/search-worker.js'].map(read).join('\n;\n');
let inline = `<script>\nconst SYNTH_WORKER_SOURCE = ${JSON.stringify(workerSrc).replace(/<\/script/gi, '<\\/script')};\n</script>\n`;
for(const m of files) inline += `<script>\n/* ${m[1]} */\n${safe(read(m[1]))}\n</script>\n`;
html = html.replace(files.map(m => m[0]).join(''), () => inline);
if(/<script defer src=/.test(html)) throw new Error('a script tag was not inlined');
html = html.replace('<span class="fh-version">', '<span class="fh-version" title="single-file preview build">preview · ');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, html);
console.log(`wrote ${out} (${(html.length / 1024).toFixed(0)} KB)`);
