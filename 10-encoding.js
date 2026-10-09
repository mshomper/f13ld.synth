/* ============================================================
   F13LD.synth · 10-encoding.js
   Design ⇄ feature vector ⇄ F13LD recipe.

   A "design" is the plain object Synth searches over. It is exactly what
   the predictor scores and exactly what the recipe carries, so the shape
   you open in F13LD.mesh or F13LD.lab is the shape that earned the score.

     { mode, cell_scale, wall_thickness, pipe_radius, offset,
       phase_shift: {x,y,z},           PI-TPMS pair offset, eighths of a cell
       normal_weights: {wx,wy,wz},     shell only (1,1,1 elsewhere)
       terms: [{ coef, phase_shift:{x,y,z}, factors:[{trig,fx,fy,fz}] }] }

   encode() mirrors train_synth.py encode_design_unified() slot for slot.
   The layout is read from the bundle's `encoding` block (modes, trig axes,
   max terms, max factors), so a bundle trained with a larger term cap
   loads without editing this file. Worker-safe: no DOM access.
   ============================================================ */
'use strict';

const SynthEncoding = (function(){

  // Preset skeletons — the axis pattern of each term, which is what survives
  // Sweep's jitter (it swaps sin/cos and redraws frequencies but keeps which
  // axes each term multiplies). Constants (zero-factor terms) are ignored.
  // Terms come from F13LD.tpms PRESETS and Sweep's shared preset table.
  const PRESET_SKELETONS = {
    gyroid:         { label:'gyroid',           axes:['xy','yz','xz'] },
    schwarzP:       { label:'Schwarz P',        axes:['x','y','z'] },
    schwarzD:       { label:'Schwarz D',        axes:['xy','xz','xz','yz'] },
    neovius:        { label:'Neovius',          axes:['x','y','z','xyz'] },
    iwp:            { label:'I-WP',             axes:['xy','yz','xz','x','y','z'] },
    fks:            { label:'Fischer-Koch S',   axes:['xyz','xyz','xyz'] },
    splitP:         { label:'split-P',          axes:['xyz','xyz','xyz'] },
    frd:            { label:'F-RD',             axes:['xyz','xyz','xyz','xy','yz','xz'] },
    gyroidHarmonic: { label:'gyroid-harmonic',  axes:['xy','yz','xz','xy','yz','xz'] },
    primitiveC:     { label:'primitive-C (G6)', axes:['x','y','z','x','y','z'] },
    octo:           { label:'octo (G8)',        axes:['x','y','z','xy','yz','xz'] },
    pHarmonic:      { label:'P-harmonic',       axes:['x','y','z','x','y','z'] },
    lidinoid:       { label:'lidinoid',         axes:['xyz','xyz','xyz','xy','yz','xz','x','y','z'] },
  };

  // Names Vault rows have used for presets → canonical key.
  const PRESET_ALIASES = {
    'gyroid':'gyroid', 'schwarzp':'schwarzP', 'schwarz p':'schwarzP', 'primitive':'schwarzP',
    'schwarzd':'schwarzD', 'schwarz d':'schwarzD', 'diamond':'schwarzD', 'neovius':'neovius',
    'iwp':'iwp', 'i-wp':'iwp', 'fks':'fks', 'fischer-koch s':'fks', 'fischerkochs':'fks',
    'splitp':'splitP', 'split-p':'splitP', 'frd':'frd', 'f-rd':'frd',
    'gyroidharmonic':'gyroidHarmonic', 'gyroid-harmonic':'gyroidHarmonic',
    'primitivec':'primitiveC', 'primitive-c (g6)':'primitiveC', 'octo':'octo', 'octo (g8)':'octo',
    'pharmonic':'pHarmonic', 'p-harmonic':'pHarmonic', 'lidinoid':'lidinoid',
  };
  function canonicalPreset(name){
    if(name == null) return null;
    const k = String(name).trim().toLowerCase();
    return PRESET_ALIASES[k] || (PRESET_SKELETONS[name] ? name : null);
  }
  function presetLabel(key){
    if(PRESET_SKELETONS[key]) return PRESET_SKELETONS[key].label;
    if(typeof key === 'string' && key.indexOf('|') >= 0)
      return key.split('|').map(presetLabel).join(' / ');
    return key || 'custom';
  }

  function skeletonOf(terms){
    return terms.filter(t => t.factors.length > 0)
      .map(t => Array.from(new Set(t.factors.map(f => f.trig.charAt(4)))).sort().join(''))
      .sort().join(',');
  }
  // Preset keys whose skeleton matches. A truncated design (old 6-term cap)
  // matches a preset whose first terms it shares.
  function matchSkeleton(terms, truncatedAt){
    const sk = skeletonOf(terms);
    const out = [];
    for(const [key, p] of Object.entries(PRESET_SKELETONS)){
      let axes = p.axes;
      if(truncatedAt && terms.length >= truncatedAt) axes = axes.slice(0, truncatedAt);
      if(axes.slice().sort().join(',') === sk) out.push(key);
    }
    return out;
  }

  const round = (v, d) => { const m = Math.pow(10, d); return Math.round(v * m) / m; };
  const f32 = v => Math.fround(v);

  function create(encoding, featureDim){
    const modes = encoding.modes, axes = encoding.trig_axes;
    const maxTerms = encoding.max_terms, maxFactors = encoding.max_factors;
    const factorStride = 5 + axes.length;           // active, fx, fy, fz, isCos, axis one-hot
    const termStride = 5 + maxFactors * factorStride; // active, coef, phase xyz, factors
    const baseOff = modes.length + 4 + 3 + 3;         // mode, 4 scalars, pair phase, normal weights
    const dim = baseOff + maxTerms * termStride;
    if(featureDim != null && featureDim !== dim)
      throw new Error('Model bundle feature_dim is '+featureDim+' but its encoding block implies '+dim+'. The bundle and trainer are out of step.');

    function encode(d){
      const x = new Float32Array(dim);
      let o = 0;
      for(const m of modes) x[o++] = (d.mode === m) ? 1 : 0;
      x[o++] = d.cell_scale || 0;
      x[o++] = d.wall_thickness || 0;
      x[o++] = d.pipe_radius || 0;
      x[o++] = d.offset || 0;
      const ps = d.phase_shift || {};
      x[o++] = ps.x || 0; x[o++] = ps.y || 0; x[o++] = ps.z || 0;
      const nw = d.normal_weights || {};
      x[o++] = nw.wx != null ? nw.wx : 1; x[o++] = nw.wy != null ? nw.wy : 1; x[o++] = nw.wz != null ? nw.wz : 1;
      const terms = d.terms;
      for(let t = 0; t < maxTerms; t++){
        const to = baseOff + t * termStride;
        if(t >= terms.length) continue;               // zero-padded inactive term
        const term = terms[t], tp = term.phase_shift || {};
        x[to] = 1; x[to+1] = term.coef; x[to+2] = tp.x || 0; x[to+3] = tp.y || 0; x[to+4] = tp.z || 0;
        for(let f = 0; f < maxFactors && f < term.factors.length; f++){
          const fo = to + 5 + f * factorStride, fac = term.factors[f];
          const axis = fac.trig.charAt(4);
          x[fo] = 1; x[fo+1] = fac.fx; x[fo+2] = fac.fy; x[fo+3] = fac.fz;
          x[fo+4] = fac.trig.startsWith('cos') ? 1 : 0;
          for(let a = 0; a < axes.length; a++) x[fo+5+a] = (axes[a] === axis) ? 1 : 0;
        }
      }
      return x;
    }

    // Inverse of encode for vectors encode produced (training seeds). Values
    // are rounded to the precision the trainer stored them at (4 dp) so that
    // encode(decode(seed)) reproduces the seed bit for bit.
    function decode(x){
      let o = 0, mi = 0, best = -Infinity;
      for(let i = 0; i < modes.length; i++){ if(x[i] > best){ best = x[i]; mi = i; } }
      o = modes.length;
      const r4 = v => round(v, 4);
      const d = {
        mode: modes[mi],
        cell_scale: r4(x[o]), wall_thickness: r4(x[o+1]), pipe_radius: r4(x[o+2]), offset: r4(x[o+3]),
        phase_shift: { x: r4(x[o+4]), y: r4(x[o+5]), z: r4(x[o+6]) },
        normal_weights: { wx: r4(x[o+7]), wy: r4(x[o+8]), wz: r4(x[o+9]) },
        terms: []
      };
      for(let t = 0; t < maxTerms; t++){
        const to = baseOff + t * termStride;
        if(!(x[to] > 0.5)) continue;
        const term = { coef: r4(x[to+1]), phase_shift: { x: r4(x[to+2]), y: r4(x[to+3]), z: r4(x[to+4]) }, factors: [] };
        for(let f = 0; f < maxFactors; f++){
          const fo = to + 5 + f * factorStride;
          if(!(x[fo] > 0.5)) continue;
          let ai = 0, ab = -Infinity;
          for(let a = 0; a < axes.length; a++){ if(x[fo+5+a] > ab){ ab = x[fo+5+a]; ai = a; } }
          term.factors.push({ trig: (x[fo+4] > 0.5 ? 'cos' : 'sin') + '(' + axes[ai] + ')',
                              fx: r4(x[fo+1]), fy: r4(x[fo+2]), fz: r4(x[fo+3]) });
        }
        d.terms.push(term);
      }
      return d;
    }

    // Bit-exact comparison against a float32 vector.
    function sameVector(a, b){
      if(a.length !== b.length) return false;
      for(let i = 0; i < a.length; i++) if(f32(a[i]) !== f32(b[i])) return false;
      return true;
    }

    return { modes, axes, maxTerms, maxFactors, termStride, factorStride, baseOff, dim,
             encode, decode, sameVector };
  }

  function cloneDesign(d){
    return {
      mode: d.mode, cell_scale: d.cell_scale, wall_thickness: d.wall_thickness,
      pipe_radius: d.pipe_radius, offset: d.offset,
      phase_shift: { x: d.phase_shift.x, y: d.phase_shift.y, z: d.phase_shift.z },
      normal_weights: { wx: d.normal_weights.wx, wy: d.normal_weights.wy, wz: d.normal_weights.wz },
      terms: d.terms.map(t => ({ coef: t.coef,
        phase_shift: { x: t.phase_shift.x, y: t.phase_shift.y, z: t.phase_shift.z },
        factors: t.factors.map(f => ({ trig: f.trig, fx: f.fx, fy: f.fy, fz: f.fz })) }))
    };
  }

  // Stable identity for de-duplicating candidates.
  function designKey(d){ return JSON.stringify(d); }

  // ── Recipe in F13LD.tpms v1.1.0 export shape ──────────────────────────────
  // Normalization flags are written explicitly: F13LD.mesh and F13LD.sweep
  // read a missing flag as "on", F13LD.lab reads it as "off". Sweep's solver
  // characterized the training designs normalized, so Synth says so.
  function toRecipe(d, ctx){
    ctx = ctx || {};
    const pi = d.mode === 'pi-tpms', shell = d.mode === 'shell';
    const presetKey = ctx.presetKey && PRESET_SKELETONS[ctx.presetKey] ? ctx.presetKey : 'custom';
    const terms = d.terms.map(t => {
      const out = { on: true, coef: t.coef,
        factors: t.factors.map(f => ({ trig: f.trig, fx: f.fx, fy: f.fy, fz: f.fz })) };
      const p = t.phase_shift;
      if(p && (p.x || p.y || p.z)) out.phase_shift = { x: p.x, y: p.y, z: p.z };
      return out;
    });
    return {
      meta: {
        version: '1.1.0',
        timestamp: new Date().toISOString(),
        preset: ctx.presetLabel || presetLabel(presetKey),
        source: 'f13ld.synth',
        synth: ctx.synth || null
      },
      surface: { type: 'terms', preset: presetKey, terms },
      surface_b: null,
      geometry: {
        offset:          pi ? null : d.offset,
        cell_scale:      d.cell_scale,
        mode:            d.mode,
        wall_thickness:  shell ? d.wall_thickness : null,
        shell_normalize: shell ? true : null,
        pipe_radius:     pi ? d.pipe_radius : null,
        pi_normalize:    pi ? true : null,
        phase_shift:     pi ? { x: d.phase_shift.x, y: d.phase_shift.y, z: d.phase_shift.z } : null,
        field_b_freq:    pi ? 1 : null,
        field_b_scale:   pi ? 1 : null,
        normal_weights:  shell ? { wx: d.normal_weights.wx, wy: d.normal_weights.wy, wz: d.normal_weights.wz } : null,
        gradient: { enabled: false }   // key kept for F13LD.mesh compatibility, as F13LD.tpms does
      }
    };
  }

  // A design with no surface left (no terms, or all coefficients ~0) gives
  // phi = 0 everywhere and renders as a solid cube.
  function isDegenerate(d){
    let s = 0;
    for(const t of d.terms) s += Math.abs(t.coef);
    return d.terms.length === 0 || s < 1e-3;
  }

  return { create, toRecipe, cloneDesign, designKey, isDegenerate, round,
           canonicalPreset, presetLabel, skeletonOf, matchSkeleton, PRESET_SKELETONS };
})();

if(typeof module !== 'undefined') module.exports = SynthEncoding;
