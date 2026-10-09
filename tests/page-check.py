"""Dev-only page check: serves the repo, loads Synth in headless Chromium
(software WebGL), runs a search, a preset-filtered search and a stopped
search, checks the shape check measured every result, checks the preview and thumbnails render, checks the phone layout
has no sideways overflow, and saves screenshots.

    python3 tests/page-check.py [port] [out-dir]

Vault and Google Fonts are blocked in the check. Timings here are the
session VM on 1-2 software threads, not a real machine."""
import sys, os, threading, http.server, functools, time
from playwright.sync_api import sync_playwright

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, 'tests')
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Q, directory=root))
threading.Thread(target=srv.serve_forever, daemon=True).start()
logs, problems = [], []

DONE = "!SYNTH.running && (SYNTH.results.length > 0 || /warn/.test(document.getElementById('st').className))"

def run(pg):
    pg.wait_for_function("typeof Predictor !== 'undefined' && (Predictor.loaded || !!Predictor.loadError)", timeout=60000)
    print('preset options:', pg.eval_on_selector_all('#presetSel option', 'os => os.map(o => o.textContent)'))
    print('depth default:', pg.eval_on_selector('#depthSeg .on', 'b => b.textContent'))
    t = time.time(); pg.click('#runBtn')
    pg.wait_for_timeout(1500)
    pg.screenshot(path=os.path.join(out, 'page-searching.png'))
    pg.wait_for_function(DONE, timeout=120000)
    print(f'search: {time.time()-t:.1f}s · status: {pg.inner_text("#stTx")}')
    n = pg.eval_on_selector_all('.cand', 'c => c.length'); print('tiles:', n)
    if n != 8: problems.append(f'expected 8 tiles, got {n}')
    pg.wait_for_function("document.querySelectorAll('.cand img.thumb').length === 8", timeout=120000)
    pg.wait_for_timeout(1500)
    lit = pg.evaluate("""(() => { const c = document.querySelector('#viewHost canvas.rm-canvas'); if(!c) return -1;
        const o = document.createElement('canvas'); o.width = c.width; o.height = c.height; const x = o.getContext('2d'); x.drawImage(c, 0, 0);
        const d = x.getImageData(0, 0, o.width, o.height).data; let lit = 0; for(let i = 0; i < d.length; i += 4) if(d[i+1] > 90) lit++; return lit / (d.length / 4); })()""")
    print(f'preview: {lit*100:.1f}% of pixels show surface')
    sh = pg.evaluate("SYNTH.results.map(r => r.shape ? [+r.shape.measured.toFixed(1), +r.shape.predicted.toFixed(1), r.shape.level, r.confidence] : null)")
    print('shape check (measured, predicted, level, confidence):', sh)
    if any(x is None for x in sh): problems.append('shape check did not measure every result')
    flagged = pg.evaluate("SYNTH.results.findIndex(r => r.shape && r.shape.level)")
    if flagged >= 0:
        pg.click(f'.cand[data-i="{flagged}"]'); pg.wait_for_timeout(800)
        pg.screenshot(path=os.path.join(out, 'page-shape-flag.png'))
    if lit < 0.02: problems.append('preview looks empty')
    pg.screenshot(path=os.path.join(out, 'page-check.png'))
    # select via a tile, sort, map ring numbers stay consistent
    pg.click('.cand[data-i="3"]'); pg.wait_for_timeout(600)
    print('selected:', pg.inner_text('#iRank'), '· header pill:', pg.inner_text('#hdrMesh'))
    pg.click('#sortSeg button[data-s="light"]'); pg.wait_for_timeout(300)
    print('lightest order:', pg.eval_on_selector_all('.cand', 'c => c.map(e => e.dataset.i).join(",")'))
    # recipe re-encodes to the scored features
    ok = pg.evaluate("""(() => { const r = SYNTH.results[0], enc = Predictor.enc, g = r.recipe.geometry;
        const back = { mode: g.mode, cell_scale: g.cell_scale, wall_thickness: g.wall_thickness || 0, pipe_radius: g.pipe_radius || 0, offset: g.offset || 0,
          phase_shift: g.phase_shift || {x:0,y:0,z:0}, normal_weights: g.normal_weights || {wx:1,wy:1,wz:1},
          terms: r.recipe.surface.terms.map(t => ({ coef: t.coef, phase_shift: t.phase_shift || {x:0,y:0,z:0}, factors: t.factors })) };
        return enc.sameVector(enc.encode(r.design), enc.encode(back)); })()""")
    print('recipe matches scored design:', ok)
    if not ok: problems.append('recipe does not re-encode to the scored design')
    # preset filter
    opts = pg.eval_on_selector_all('#presetSel option', 'os => os.map(o => o.value)')
    key = 'gyroid' if 'gyroid' in opts else opts[1]
    pg.select_option('#presetSel', key); pg.click('#runBtn')
    pg.wait_for_function(DONE, timeout=120000)
    presets = pg.evaluate("[...new Set(SYNTH.results.map(r => r.presetKey))]")
    print(f'preset {key}:', presets)
    if presets != [key]: problems.append(f'preset filter leaked: {presets}')
    pg.select_option('#presetSel', '')
    # Stop mid-search keeps results
    pg.click('#depthSeg button[data-d="deep"]'); pg.click('#runBtn'); pg.wait_for_timeout(2500); pg.click('#runBtn')
    pg.wait_for_function(DONE, timeout=120000)
    print('after Stop:', pg.inner_text('#stTx'), '· tiles', pg.eval_on_selector_all('.cand', 'c => c.length'))
    pg.click('#depthSeg button[data-d="wide"]')
    # drawer
    pg.click('#cfgBtn'); pg.wait_for_timeout(300); pg.screenshot(path=os.path.join(out, 'page-drawer.png')); pg.click('#cfgBtn')

with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}'))
    pg.on('pageerror', lambda e: (logs.append(f'[pageerror] {e}'), problems.append(str(e))))
    pg.route('**/supabase.co/**', lambda r: r.abort())
    pg.route('**/fonts.googleapis.com/**', lambda r: r.abort())
    pg.goto(f'http://127.0.0.1:{port}/index.html')
    try:
        run(pg)
        # phone: nothing may scroll sideways
        ph = b.new_page(viewport={'width': 390, 'height': 844})
        ph.on('pageerror', lambda e: problems.append('phone: ' + str(e)))
        ph.route('**/supabase.co/**', lambda r: r.abort()); ph.route('**/fonts.googleapis.com/**', lambda r: r.abort())
        ph.goto(f'http://127.0.0.1:{port}/index.html')
        ph.wait_for_function("typeof Predictor !== 'undefined' && Predictor.loaded", timeout=60000)
        ph.click('#runBtn'); ph.wait_for_function(DONE, timeout=120000); ph.wait_for_timeout(1500)
        over = ph.evaluate("[document.documentElement.scrollWidth - document.documentElement.clientWidth, ...[...document.querySelectorAll('.app *')].filter(e => e.scrollWidth > e.clientWidth + 1 && !['visible','clip'].includes(getComputedStyle(e).overflowX) && !e.matches('.stx,.badge,.model-chip,.dtag,.vlabel,.dtags,.nm')).map(e => e.className || e.tagName)]")
        print('phone overflow:', over)
        if over[0] > 0 or len(over) > 1: problems.append(f'phone overflow {over}')
        ph.screenshot(path=os.path.join(out, 'page-phone.png'), full_page=True)
    finally:
        print('\n'.join(l for l in logs if 'Failed to load resource' not in l and 'ERR_' not in l))
        b.close(); srv.shutdown()
print('PROBLEMS:' if problems else 'no problems', *problems, sep='\n  ')
sys.exit(1 if problems else 0)
