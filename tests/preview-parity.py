"""Dev-only check that the 3-D preview builds the shape F13LD.mesh builds.

For a spread of real designs (training seeds, all three modes), the
preview's own field shader (25-raymarch.js rmFieldGLSL) is evaluated on the
GPU at the centres of a 36³ grid over one cell, and the solid fraction is
compared with F13LD.mesh's TPMS field code (worker/m20-sdf-noise-tpms.js)
on the same grid in Node.

    python3 tests/preview-parity.py [path to f13ld.mesh]
"""
import sys, os, json, subprocess, threading, http.server, functools
from playwright.sync_api import sync_playwright

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mesh = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, '..', 'f13ld.mesh'))
N = 36

# Designs: decoded seeds, spread over modes and presets, as Synth recipes.
designs = json.loads(subprocess.check_output(['node', '-e', f"""
const fs=require('fs'),SE=require('{root}/10-encoding.js');
const b=JSON.parse(fs.readFileSync('{root}/weights/tpms.json'));const enc=SE.create(b.encoding,b.meta.feature_dim);
const by={{}};b.seed_samples.forEach(s=>{{const d=enc.decode(Float32Array.from(s));(by[d.mode]||(by[d.mode]=[])).push(d);}});
const pick=[];for(const m in by)for(let i=0;i<4&&i<by[m].length;i++)pick.push(by[m][Math.floor(i*by[m].length/4)]);
console.log(JSON.stringify(pick.map(d=>({{design:d,recipe:SE.toRecipe(d,{{}})}}))));"""]))

# Mesh: solid fraction on the grid. Mesh's world [-5,5] spans cell_scale periods,
# so sample the world box that holds exactly one period: [-5,5]/cell_scale.
mesh_frac = json.loads(subprocess.check_output(['node', '-e', f"""
const fs=require('fs'),vm=require('vm');const sb={{Math,registerSDF(){{}},console}};
vm.runInNewContext(fs.readFileSync('{mesh}/worker/m20-sdf-noise-tpms.js','utf8')+';this.B=buildTPMSSDF;',sb);
const D={json.dumps(designs)};const N={N};
console.log(JSON.stringify(D.map(x=>{{const sdf=sb.B(x.recipe),L=5/x.recipe.geometry.cell_scale;let n=0;
for(let i=0;i<N;i++)for(let j=0;j<N;j++)for(let k=0;k<N;k++){{const p=[-L+(i+.5)*2*L/N,-L+(j+.5)*2*L/N,-L+(k+.5)*2*L/N];if(sdf(p)<0)n++;}}
return n/(N*N*N);}})));"""]))

class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 8797), functools.partial(Q, directory=root))
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    pg = b.new_page()
    pg.goto('http://127.0.0.1:8797/tests/README.md')
    for f in ['24-f13-shade.js', '25-raymarch.js']:
        pg.add_script_tag(url=f'http://127.0.0.1:8797/{f}')
    gpu = pg.evaluate("""([D, N]) => D.map(x => {
      const c = document.createElement('canvas'); c.width = N; c.height = N * N;
      const gl = c.getContext('webgl2');
      const fs = '#version 300 es\\nprecision highp float;out vec4 o;\\n' + rmFieldGLSL(x.design).glsl +
        '\\nvoid main(){ float N=' + N.toFixed(1) + '; vec2 f=floor(gl_FragCoord.xy); float i=f.x, j=mod(f.y,N), k=floor(f.y/N);' +
        ' vec3 p=-3.14159265+(vec3(i,j,k)+0.5)*6.2831853/N; o=vec4(implicit(p)<0.0?1.0:0.0,0.0,0.0,1.0); }';
      const sh = (t, s) => { const h = gl.createShader(t); gl.shaderSource(h, s); gl.compileShader(h); if(!gl.getShaderParameter(h, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(h)); return h; };
      const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, RM_VERT)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(pr); gl.useProgram(pr);
      const bf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
      const l = gl.getAttribLocation(pr, 'p'); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0);
      gl.viewport(0, 0, N, N * N); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      const px = new Uint8Array(N * N * N * 4); gl.readPixels(0, 0, N, N * N, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let n = 0; for(let q = 0; q < px.length; q += 4) if(px[q] > 127) n++; return n / (N * N * N);
    })""", [designs, N])
    b.close()
srv.shutdown()
worst = 0
for x, m, g in zip(designs, mesh_frac, gpu):
    d = abs(m - g) * 100; worst = max(worst, d)
    print(f"{x['design']['mode']:8} preview {g*100:5.1f}%  mesh {m*100:5.1f}%  diff {d:4.1f}")
print(f'solid fraction, preview vs F13LD.mesh: worst diff {worst:.1f} points over {len(designs)} designs')
sys.exit(0 if worst < 1.5 else 1)
