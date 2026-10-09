/* ============================================================
   F13LD.synth · 25-raymarch.js
   3-D preview of a candidate: a raymarched implicit surface built from the
   exact design Synth scored, in F13LD.tpms's shader form, so what you see
   is what F13LD.mesh and F13LD.lab will build.

   Field  φ(p) = Σ coef · Π trig(f · (p_axis + term phase))   (F13LD.mesh's
   evalTermsList: the per-term phase is added before the frequency).
   Modes (normalization on, as the recipe says):
     solid    φ − offset
     shell    |φ − offset| / max(|∇φ|, 0.08) − wall · (n·w)   n = |∇φ| unit,
              w = normal weights (F13LD.mesh's anisotropic shell)
     pi-tpms  angle-corrected distance to {φ = 0} ∩ {φ(p + 2π·shift) = 0}
              − pipe radius (F13LD.tpms piField)
   One cell = one 2π period; 2×2×2 shows two periods per axis.
   Shading: F13LD-SHADE (24-f13-shade.js), the suite's shared look.

   SynthViewer(host)   interactive viewer (orbit, zoom, X/Y/Z section)
   synthThumbnails()   small renders of several designs (one GL context)
   ============================================================ */
'use strict';

const RM_VERT = '#version 300 es\nin vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}';

function rmFloat(v){ const s = (+v).toFixed(6).replace(/0+$/, '').replace(/\.$/, '.0'); return s.indexOf('.') < 0 ? s + '.0' : s; }

// GLSL for the design's field: base(p), gradB(p), implicit(p) (< 0 inside).
// Returns {glsl, feature}. Also used by tests/preview-parity.py.
function rmFieldGLSL(d){
  const axisVar = { x: 'gx', y: 'gy', z: 'gz' };
  const lines = d.terms.map(t => {
    const ph = t.phase_shift || { x: 0, y: 0, z: 0 };
    if(!t.factors.length) return '  v+=' + rmFloat(t.coef) + ';';
    const parts = t.factors.map(f => {
      const ax = f.trig.charAt(4), fn = f.trig.slice(0, 3);
      const freq = f['f' + ax], phase = ph[ax] || 0;
      return fn + '(' + rmFloat(freq) + '*(' + axisVar[ax] + '+' + rmFloat(phase) + '))';
    });
    return '  v+=' + rmFloat(t.coef) + '*' + parts.join('*') + ';';
  }).join('\n');
  const base = 'float base(vec3 p){float gx=p.x,gy=p.y,gz=p.z;float v=0.0;\n' + lines + '\n  return v;}\n' +
    'vec3 gradB(vec3 p){float e=0.012;return vec3(base(p+vec3(e,0,0))-base(p-vec3(e,0,0)),base(p+vec3(0,e,0))-base(p-vec3(0,e,0)),base(p+vec3(0,0,e))-base(p-vec3(0,0,e)))/(2.0*e);}';
  let implicit, feature;
  if(d.mode === 'pi-tpms'){
    const TP = 2 * Math.PI, s = d.phase_shift;
    const dv = 'vec3(' + rmFloat(s.x * TP) + ',' + rmFloat(s.y * TP) + ',' + rmFloat(s.z * TP) + ')';
    implicit = 'float implicit(vec3 p){vec3 q=p+' + dv + ';vec3 gA=gradB(p),gB=gradB(q);' +
      'float mA=max(length(gA),0.08),mB=max(length(gB),0.08);float dA=base(p)/mA,dB=base(q)/mB;' +
      'float c=clamp(dot(gA,gB)/(mA*mB),-0.95,0.95);float num=dA*dA-2.0*c*dA*dB+dB*dB;' +
      'return sqrt(max(num,0.0)/(1.0-c*c))-' + rmFloat(d.pipe_radius) + ';}';
    feature = d.pipe_radius;
  } else if(d.mode === 'shell'){
    const w = d.normal_weights || { wx: 1, wy: 1, wz: 1 };
    implicit = 'float implicit(vec3 p){vec3 g=gradB(p);float m=max(length(g),0.08);' +
      'vec3 n=abs(g)/max(length(g),1e-6);float wt=' + rmFloat(d.wall_thickness) + '*dot(n,vec3(' + rmFloat(w.wx) + ',' + rmFloat(w.wy) + ',' + rmFloat(w.wz) + '));' +
      'return abs(base(p)-(' + rmFloat(d.offset) + '))/m-wt;}';
    feature = d.wall_thickness;
  } else {
    implicit = 'float implicit(vec3 p){return base(p)-(' + rmFloat(d.offset) + ');}';
    feature = Math.PI * 0.25;
  }
  return { glsl: base + '\n' + implicit, feature };
}

// Full fragment shader for one design. tiles = periods shown per axis.
function rmBuildFrag(d, tiles, quality){
  const H = rmFloat(Math.PI * tiles);
  const F = rmFieldGLSL(d), feature = F.feature;
  const steps = quality === 'low' ? 96 : 220;
  return [
    '#version 300 es', 'precision highp float;', 'out vec4 fragColor;',
    'uniform vec2 res;uniform mat3 rot;uniform float zoom;uniform float uClipAxis;uniform float uClipPos;',
    F.glsl,
    'float boxSDF(vec3 p){vec3 d=abs(p)-' + H + ';return length(max(d,0.0))+min(max(d.x,max(d.y,d.z)),0.0);}',
    'float sceneSDF(vec3 p){float s=max(implicit(p),boxSDF(p));if(uClipAxis>0.5){float c=(uClipAxis<1.5)?p.x:((uClipAxis<2.5)?p.y:p.z);s=max(s,c-uClipPos*' + H + ');}return s;}',
    'vec3 nrm(vec3 p,float e){return normalize(vec3(sceneSDF(p+vec3(e,0,0))-sceneSDF(p-vec3(e,0,0)),sceneSDF(p+vec3(0,e,0))-sceneSDF(p-vec3(0,e,0)),sceneSDF(p+vec3(0,0,e))-sceneSDF(p-vec3(0,0,e))));}',
    'float f13Map(vec3 p){return 0.6*sceneSDF(p);}',
    F13_SHADE_GLSL,
    'void main(){',
    '  vec2 uv=(gl_FragCoord.xy-res*0.5)/min(res.x,res.y);',
    '  vec3 ro=rot*vec3(0.0,0.0,zoom);vec3 rd=normalize(rot*vec3(uv.x,uv.y,-1.6));',
    '  float H=' + H + ';',
    '  float r=clamp(length(uv)*1.1,0.0,1.0);vec3 bgCol=mix(vec3(0.07),vec3(0.03),r*r);vec4 bg=vec4(bgCol,1.0);',
    '  float featureSize=' + rmFloat(feature) + ';float camScale=clamp(zoom/(16.0*H/3.14159),0.15,1.0);',
    '  float thresh=featureSize*0.008*camScale;float maxStep=featureSize*0.25*camScale;float nrmE=featureSize*0.06*camScale;',
    '  vec3 iv=vec3(1.0)/rd;vec3 tb=(-vec3(H)-ro)*iv,tt=(vec3(H)-ro)*iv;vec3 tmi=min(tb,tt),tma=max(tb,tt);',
    '  float tEn=max(max(tmi.x,tmi.y),tmi.z),tEx=min(min(tma.x,tma.y),tma.z);',
    '  if(tEn>tEx||tEx<0.0){fragColor=bg;return;}',
    '  float t=max(tEn,0.001);bool hit=false;',
    '  for(int i=0;i<' + steps + ';i++){vec3 p=ro+rd*t;float d=sceneSDF(p);if(d<thresh){hit=true;break;}if(t>tEx+0.01)break;t+=clamp(d*0.9,thresh*0.5,maxStep);}',
    '  if(!hit){fragColor=bg;return;}',
    '  vec3 pos=ro+rd*t,n=nrm(pos,nrmE);if(dot(n,-rd)<0.0)n=-n;',
    '  bool isCut=implicit(pos)<sceneSDF(pos)-1e-5;',
    '  vec3 col=f13Shade(vec3(0.369,0.792,0.647),pos,n,rd,rot,isCut,6.2832,nrmE,1.6*H);',
    '  if(uClipAxis>0.5){vec3 ax=vec3(uClipAxis<1.5?1.0:0.0,(uClipAxis>1.5&&uClipAxis<2.5)?1.0:0.0,uClipAxis>2.5?1.0:0.0);',
    '    if(abs(dot(pos,ax)-uClipPos*H)<0.02*H&&abs(dot(n,ax))>0.7){vec3 ac=uClipAxis<1.5?vec3(0.1333,0.8275,0.9333):uClipAxis<2.5?vec3(0.9098,0.4745,0.9765):vec3(0.9804,0.8,0.0824);col=mix(col,ac,0.30);}}',
    '  col=mix(bgCol,col,exp(-max(t-tEn,0.0)*0.010));',
    '  fragColor=vec4(clamp(col,0.0,1.0),1.0);',
    '}'
  ].join('\n');
}

// Orbit as yaw/pitch → camera basis (columns: right, up, back).
function rmRotation(yaw, pitch){
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return new Float32Array([cy, 0, -sy,  sy * sp, cp, cy * sp,  sy * cp, -sp, cy * cp]);
}

function rmContext(canvas){
  const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if(!gl) return null;
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  return gl;
}
function rmProgram(gl, frag){
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram(), vs = sh(gl.VERTEX_SHADER, RM_VERT), fs = sh(gl.FRAGMENT_SHADER, frag);
  gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
  gl.deleteShader(vs); gl.deleteShader(fs);
  if(!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(p));
  const loc = gl.getAttribLocation(p, 'p');
  return { p, loc, U: { res: gl.getUniformLocation(p, 'res'), rot: gl.getUniformLocation(p, 'rot'), zoom: gl.getUniformLocation(p, 'zoom'),
    clipAxis: gl.getUniformLocation(p, 'uClipAxis'), clipPos: gl.getUniformLocation(p, 'uClipPos') } };
}
function rmDraw(gl, prog, w, h, view, applyView){
  gl.viewport(0, 0, w, h);
  gl.useProgram(prog.p);
  gl.enableVertexAttribArray(prog.loc); gl.vertexAttribPointer(prog.loc, 2, gl.FLOAT, false, 0, 0);
  gl.uniform2f(prog.U.res, w, h);
  gl.uniformMatrix3fv(prog.U.rot, false, rmRotation(view.yaw, view.pitch));
  gl.uniform1f(prog.U.zoom, view.zoom);
  gl.uniform1f(prog.U.clipAxis, view.clipAxis || 0);
  gl.uniform1f(prog.U.clipPos, view.clipPos || 0);
  applyView(gl, prog.p);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
const RM_DEFAULT_VIEW = { yaw: 0.75, pitch: 0.45 };
const rmDefaultZoom = tiles => 16 * tiles;   // F13LD.tpms frames one π-half-cell at zoom 16

// ── Interactive viewer ──────────────────────────────────────────────────────
function SynthViewer(host){
  this.host = host;
  this.canvas = document.createElement('canvas');
  this.canvas.className = 'rm-canvas';
  host.appendChild(this.canvas);
  this.gl = rmContext(this.canvas);
  this.design = null; this.prog = null; this.tiles = 1;
  this.view = { yaw: RM_DEFAULT_VIEW.yaw, pitch: RM_DEFAULT_VIEW.pitch, zoom: rmDefaultZoom(1), clipAxis: 0, clipPos: 0 };
  this.moving = false; this.raf = 0;
  const self = this;
  this.viewMenu = f13ViewInit({ tool: 'synth', host, canvas: this.canvas, redraw: () => self.render(), moving: () => self.moving });
  // The shared menu labels its button with a glyph; Synth draws its icons.
  const vb = host.querySelector('.f13v > button');
  if(vb) vb.innerHTML = '<svg class="g" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 2.5 A5.5 5.5 0 0 1 8 13.5 Z" fill="currentColor"/></svg> view';
  if(!this.gl){ host.classList.add('rm-nogl'); return; }
  let drag = false, lx = 0, ly = 0, pinch = null;
  const c = this.canvas;
  c.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; c.setPointerCapture(e.pointerId); self.moving = true; });
  c.addEventListener('pointermove', e => {
    if(!drag || pinch) return;
    self.view.yaw -= (e.clientX - lx) * 0.01;
    self.view.pitch = Math.max(-1.45, Math.min(1.45, self.view.pitch + (e.clientY - ly) * 0.01));
    lx = e.clientX; ly = e.clientY; self.render();
  });
  const end = () => { if(drag){ drag = false; self.moving = false; self.render(); } };
  c.addEventListener('pointerup', end); c.addEventListener('pointercancel', end);
  c.addEventListener('wheel', e => { e.preventDefault(); self.zoomBy(Math.exp(e.deltaY * 0.0012)); }, { passive: false });
  const td = t => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
  c.addEventListener('touchstart', e => { if(e.touches.length >= 2) pinch = { d0: td(e.touches) || 1, z0: self.view.zoom }; }, { passive: true });
  c.addEventListener('touchmove', e => { if(pinch && e.touches.length >= 2){ e.preventDefault(); self.view.zoom = self.clampZoom(pinch.z0 * pinch.d0 / (td(e.touches) || 1)); self.render(); } }, { passive: false });
  c.addEventListener('touchend', e => { if(e.touches.length < 2) pinch = null; });
  if(typeof ResizeObserver !== 'undefined') new ResizeObserver(() => self.render()).observe(host);
}
SynthViewer.prototype.clampZoom = function(z){ const z0 = rmDefaultZoom(this.tiles); return Math.max(z0 * 0.45, Math.min(z0 * 2.2, z)); };
SynthViewer.prototype.zoomBy = function(f){ this.view.zoom = this.clampZoom(this.view.zoom * f); this.moving = true; this.render(); clearTimeout(this._zt); this._zt = setTimeout(() => { this.moving = false; this.render(); }, 200); };
SynthViewer.prototype.setDesign = function(d){
  this.design = d; this.compile();
};
SynthViewer.prototype.setTiles = function(n){
  const ratio = this.view.zoom / rmDefaultZoom(this.tiles);
  this.tiles = n; this.view.zoom = rmDefaultZoom(n) * ratio; this.compile();
};
SynthViewer.prototype.setClip = function(axis, pos){ this.view.clipAxis = axis; this.view.clipPos = pos; this.render(); };
SynthViewer.prototype.reset = function(){
  this.view.yaw = RM_DEFAULT_VIEW.yaw; this.view.pitch = RM_DEFAULT_VIEW.pitch; this.view.zoom = rmDefaultZoom(this.tiles); this.render();
};
SynthViewer.prototype.compile = function(){
  if(!this.gl || !this.design) return;
  try {
    const p = rmProgram(this.gl, rmBuildFrag(this.design, this.tiles, 'high'));
    if(this.prog) this.gl.deleteProgram(this.prog.p);
    this.prog = p;
  } catch(e){ console.error('[F13LD.synth] preview shader failed:', e.message); this.prog = null; }
  this.render();
};
SynthViewer.prototype.render = function(){
  if(!this.gl || !this.prog) return;
  cancelAnimationFrame(this.raf);
  this.raf = requestAnimationFrame(() => {
    const r = this.host.getBoundingClientRect();
    if(r.width < 4 || r.height < 4) return;
    const scale = this.moving ? 0.5 : Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.round(r.width * scale), h = Math.round(r.height * scale);
    if(this.canvas.width !== w || this.canvas.height !== h){ this.canvas.width = w; this.canvas.height = h; }
    rmDraw(this.gl, this.prog, w, h, this.view, this.viewMenu.apply);
  });
};

// ── Thumbnails ──────────────────────────────────────────────────────────────
// One hidden GL context renders every thumbnail and hands back image URLs,
// so eight cards never hold eight live GL contexts. Returns a promise that
// resolves per design via onEach(i, url) and then with the full list.
let RM_THUMB = null;
function synthThumbnails(designs, size, onEach){
  if(!RM_THUMB){
    const c = document.createElement('canvas');
    RM_THUMB = { canvas: c, gl: rmContext(c), view: f13ViewInit({ tool: 'synth' }) };
  }
  const T = RM_THUMB, urls = [];
  if(!T.gl) return Promise.resolve(urls);
  T.canvas.width = size; T.canvas.height = size;
  return designs.reduce((p, d, i) => p.then(() => new Promise(res => {
    setTimeout(() => {
      try {
        const prog = rmProgram(T.gl, rmBuildFrag(d, 1, 'low'));
        rmDraw(T.gl, prog, size, size, { yaw: RM_DEFAULT_VIEW.yaw, pitch: RM_DEFAULT_VIEW.pitch, zoom: rmDefaultZoom(1) * 1.05 }, T.view.apply);
        urls[i] = T.canvas.toDataURL('image/png');
        T.gl.deleteProgram(prog.p);
      } catch(e){ urls[i] = null; console.warn('[F13LD.synth] thumbnail failed:', e.message); }
      if(onEach) onEach(i, urls[i]);
      res();
    }, 0);
  })), Promise.resolve()).then(() => urls);
}

// ── Shape check: solid fraction measured from the shape ─────────────────────
// The design's own field (the one tests/preview-parity.py checks against
// F13LD.mesh) is sampled at the centres of an N³ grid over one cell on the
// GPU; one pixel per sample. Returns percent solid, or null without WebGL2.
let RM_MEASURE = null;
function rmMeasureSolid(design, N){
  N = N || 48;
  if(!RM_MEASURE){ const c = document.createElement('canvas'); RM_MEASURE = { canvas: c, gl: rmContext(c) }; }
  const M = RM_MEASURE, gl = M.gl;
  if(!gl) return null;
  M.canvas.width = N; M.canvas.height = N * N;
  const fs = '#version 300 es\nprecision highp float;out vec4 o;\n' + rmFieldGLSL(design).glsl +
    '\nvoid main(){ float N=' + rmFloat(N) + '; vec2 f=floor(gl_FragCoord.xy); float i=f.x, j=mod(f.y,N), k=floor(f.y/N);' +
    ' vec3 p=-3.14159265+(vec3(i,j,k)+0.5)*6.2831853/N; o=vec4(implicit(p)<0.0?1.0:0.0,0.0,0.0,1.0); }';
  const prog = rmProgram(gl, fs);
  gl.useProgram(prog.p);
  gl.enableVertexAttribArray(prog.loc);
  gl.vertexAttribPointer(prog.loc, 2, gl.FLOAT, false, 0, 0);
  gl.viewport(0, 0, N, N * N);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  const px = new Uint8Array(N * N * N * 4);
  gl.readPixels(0, 0, N, N * N, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.deleteProgram(prog.p);
  let n = 0;
  for(let q = 0; q < px.length; q += 4) if(px[q] > 127) n++;
  return 100 * n / (N * N * N);
}
// All designs, yielding between them so the page stays live.
async function rmMeasureSolidAll(designs, N){
  const out = [];
  for(const d of designs){
    try { out.push(rmMeasureSolid(d, N)); } catch(e){ out.push(null); console.warn('[F13LD.synth] shape check failed:', e.message); }
    await new Promise(r => setTimeout(r, 0));
  }
  return out;
}
