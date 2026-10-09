"""Dev-only page check: serves the repo, loads Synth in headless Chromium,
runs one search with the saved pad state, prints the console and saves a
screenshot. Usage: python3 tests/page-check.py [port] [out.png]"""
import sys, threading, http.server, functools, os, time
from playwright.sync_api import sync_playwright
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
shot = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, 'tests', 'page-check.png')
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
H = functools.partial(Q, directory=root)
srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
logs = []
def _run(pg):
    pg.wait_for_function("typeof Predictor !== 'undefined' && (Predictor.loaded || !!Predictor.loadError)", timeout=40000)
    print('preset options:', pg.eval_on_selector_all('#presetSel option', 'os => os.map(o => o.textContent)'))
    t = time.time()
    pg.click('#searchBtn')
    pg.wait_for_function("document.querySelectorAll('.result-card').length > 0 || /fail|no candidates/.test(document.getElementById('resultsMeta').textContent)", timeout=180000)
    print(f'search wall time {time.time()-t:.1f}s · cards', pg.eval_on_selector_all('.result-card', 'c => c.length'))
    print('results meta:', pg.inner_text('#resultsMeta'))
    pg.screenshot(path=shot, full_page=True)
    print('material E_s field:', pg.input_value('#ref_modulus_gpa'))
    # Preset filter: every card must come from the chosen preset's seeds.
    pg.select_option('#presetSel', 'gyroid')
    pg.evaluate("document.getElementById('resultsBody').innerHTML=''")
    pg.click('#searchBtn')
    pg.wait_for_function("document.querySelectorAll('.result-card').length > 0 || /fail|no candidates/.test(document.getElementById('resultsMeta').textContent)", timeout=180000)
    tags = pg.eval_on_selector_all('.rc-source-tag.preset', 'ts => ts.map(t => t.textContent)')
    print('gyroid filter tags:', sorted(set(tags)), len(tags))
    rec = pg.evaluate("JSON.stringify(lastResults.synth[0].recipe)")
    print('first recipe:', rec[:900])

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1400, 'height': 1000})
    pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}'))
    pg.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
    pg.route('**/supabase.co/**', lambda r: r.abort())        # no Vault from the test machine
    pg.route('**/fonts.googleapis.com/**', lambda r: r.abort())
    pg.goto(f'http://127.0.0.1:{port}/index.html')
    try:
      _run(pg)
    finally:
      print('\n'.join(logs)); b.close(); srv.shutdown()
