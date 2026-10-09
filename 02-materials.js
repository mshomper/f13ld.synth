/* ============================================================
   F13LD.synth · 02-materials.js   (generated — do not edit by hand)
   From F13LD.lab's AM material library (15c-materials.js, reviewed by
   Matt 2026-09-29). Regenerate: node tools/sync-materials.js ../f13ld.lab
   E GPa · rho g/cc · k W/mK (null: no data) · sigY MPa
   ============================================================ */
'use strict';
const SYNTH_MATERIALS = [
  {"id":"ti64-g23-lpbf-asbuilt","name":"Ti-6Al-4V ELI (Grade 23)","condition":"As-built (no HT)","process":"LPBF (SLM 280/500, 30 um, 400 W)","group":"Titanium","E":115,"rho":4.43,"k":5.4,"sigY":1123},
  {"id":"ti64-g5-lpbf-sr","name":"Ti-6Al-4V (Grade 5)","condition":"Stress relieved / heat treated 800 C 2 h","process":"LPBF (EOS M290, 40 um)","group":"Titanium","E":112.5,"rho":4.41,"k":6.7,"sigY":990},
  {"id":"ti64-g5-lpbf-hip","name":"Ti-6Al-4V (Grade 5)","condition":"HIP","process":"LPBF (3D Systems DMP)","group":"Titanium","E":112.5,"rho":4.42,"k":6.7,"sigY":920},
  {"id":"ti64-g23-lpbf-annealed","name":"Ti-6Al-4V ELI (Grade 23)","condition":"Annealed (per Renishaw)","process":"LPBF (Renishaw RenAM 500, 60 um)","group":"Titanium","E":117,"rho":4.4,"k":6.6,"sigY":956.5},
  {"id":"ti64-g23-lpbf-hip","name":"Ti-6Al-4V ELI (Grade 23)","condition":"HIP 920 C / 1000 bar / 2 h","process":"LPBF (SLM 280, 30 um, 400 W)","group":"Titanium","E":124,"rho":4.43,"k":6.6,"sigY":878},
  {"id":"ti64-g5-ebm-asbuilt","name":"Ti-6Al-4V (Grade 5)","condition":"As-built","process":"EB-PBF (Arcam Q10plus, 70 um)","group":"Titanium","E":113.8,"rho":4.43,"k":6.7,"sigY":896},
  {"id":"cpti-g2-lpbf-asbuilt","name":"CP-Ti Grade 2","condition":"As-built","process":"LPBF (EOS M290 400 W)","group":"Titanium","E":105,"rho":4.51,"k":16.4,"sigY":560},
  {"id":"cpti-g2-lpbf-ht","name":"CP-Ti Grade 2","condition":"Heat treated 700 C / 1.5-2 h, Ar","process":"LPBF (EOS M290 / M404)","group":"Titanium","E":105,"rho":4.51,"k":16.4,"sigY":437.5},
  {"id":"cpti-g1-lpbf-sr","name":"CP-Ti Grade 1","condition":"Stress relieved","process":"LPBF (3D Systems ProX DMP)","group":"Titanium","E":112.5,"rho":4.51,"k":16,"sigY":380},
  {"id":"ti6al7nb-lpbf-asbuilt","name":"Ti-6Al-7Nb","condition":"As-built","process":"LPBF (research, Hein et al. 2022)","group":"Titanium","E":105,"rho":4.52,"k":6.7,"sigY":940},
  {"id":"ti6al7nb-lpbf-sr","name":"Ti-6Al-7Nb","condition":"Stress relief 600 C / 4 h (HT3)","process":"LPBF (research, Hein et al. 2022)","group":"Titanium","E":116,"rho":4.52,"k":6.7,"sigY":1045},
  {"id":"ti2448-lpbf-asbuilt","name":"Ti-24Nb-4Zr-8Sn (beta Ti, Ti2448)","condition":"As-built","process":"LPBF (DMG Mori LT12, Z-loaded)","group":"Titanium","E":49,"rho":0,"k":null,"sigY":490},
  {"id":"ss316l-lpbf-asbuilt","name":"316L stainless","condition":"As-built","process":"LPBF (EOS M290, 40 um)","group":"Stainless steel","E":180,"rho":7.97,"k":15.3,"sigY":510},
  {"id":"ss316l-lpbf-annealed","name":"316L stainless","condition":"Full anneal","process":"LPBF (3D Systems DMP Flex/Factory 350)","group":"Stainless steel","E":180,"rho":8,"k":16.3,"sigY":345},
  {"id":"ss174-lpbf-h900","name":"17-4PH stainless","condition":"H900 (per EOS HT)","process":"LPBF (EOS M290, 40 um)","group":"Stainless steel","E":193,"rho":7.75,"k":18.3,"sigY":1240},
  {"id":"ss174-lpbf-asbuilt","name":"17-4PH stainless","condition":"As-built","process":"LPBF (research, Li et al.)","group":"Stainless steel","E":193,"rho":7.75,"k":18.3,"sigY":784},
  {"id":"ss155-lpbf-h900","name":"15-5PH stainless (EOS PH1)","condition":"H900 modified","process":"LPBF (EOS M290)","group":"Stainless steel","E":190,"rho":7.7,"k":17.8,"sigY":1325},
  {"id":"ss155-lpbf-asbuilt","name":"15-5PH stainless (EOS PH1)","condition":"As-built","process":"LPBF (EOS M290)","group":"Stainless steel","E":190,"rho":7.7,"k":18.3,"sigY":977.5},
  {"id":"ms1-lpbf-aged","name":"Maraging steel 1.2709 (EOS MS1)","condition":"Aged 490 C / 6 h","process":"LPBF (EOS M290, 40 um)","group":"Stainless steel","E":190,"rho":8.05,"k":20,"sigY":2015},
  {"id":"in718-lpbf-asbuilt","name":"Inconel 718","condition":"As-built","process":"LPBF (EOS M290, 40 um)","group":"Nickel superalloy","E":200,"rho":8.19,"k":11.1,"sigY":725},
  {"id":"in718-lpbf-sta","name":"Inconel 718","condition":"Solution + aged (AMS 5662-type)","process":"LPBF (EOS M290, 40 um)","group":"Nickel superalloy","E":200,"rho":8.19,"k":11.4,"sigY":1192.5},
  {"id":"in718-lpbf-hip-sta","name":"Inconel 718","condition":"HIP + solution + aged","process":"LPBF (Nikon SLM NXG 600, vertical)","group":"Nickel superalloy","E":200,"rho":8.2,"k":11.4,"sigY":985},
  {"id":"in625-lpbf-sr","name":"Inconel 625","condition":"Stress relieved 870 C","process":"LPBF (EOS M290, 40 um)","group":"Nickel superalloy","E":209,"rho":8.44,"k":9.8,"sigY":660},
  {"id":"hx-lpbf-asbuilt","name":"Hastelloy X","condition":"As-built","process":"LPBF (EOS M290 400 W)","group":"Nickel superalloy","E":185,"rho":8.2,"k":9.2,"sigY":587.5},
  {"id":"hx-lpbf-hip","name":"Hastelloy X","condition":"HIP 1177 C","process":"LPBF (Velo3D, vertical)","group":"Nickel superalloy","E":159,"rho":8.22,"k":9.2,"sigY":325},
  {"id":"h282-lpbf-ht","name":"Haynes 282","condition":"Heat treated (EOS option 1)","process":"LPBF (EOS M290, 40 um)","group":"Nickel superalloy","E":218,"rho":8.3,"k":10.2,"sigY":710.5},
  {"id":"alsi10mg-lpbf-asbuilt","name":"AlSi10Mg","condition":"As-built","process":"LPBF (3D Systems DMP)","group":"Aluminium","E":71,"rho":2.68,"k":125,"sigY":245},
  {"id":"alsi10mg-lpbf-sr","name":"AlSi10Mg","condition":"Stress relieved","process":"LPBF (3D Systems DMP)","group":"Aluminium","E":73,"rho":2.68,"k":165,"sigY":185},
  {"id":"alsi10mg-lpbf-t6","name":"AlSi10Mg","condition":"T6","process":"LPBF (GKN Additive, vertical bars, as-built surface)","group":"Aluminium","E":80.4,"rho":2.67,"k":140,"sigY":228.3},
  {"id":"scalmalloy-lpbf-aged","name":"Scalmalloy (Al-Mg-Sc-Zr)","condition":"Aged 325 C / 4 h","process":"LPBF (3D Systems DMP, 30 um)","group":"Aluminium","E":69,"rho":2.67,"k":97.5,"sigY":490},
  {"id":"a20x-lpbf-t7","name":"A20X / A205 (Al-Cu-Ag-TiB2)","condition":"T7 (solution + age)","process":"LPBF (Colibrium M2 Series 5, 400 W)","group":"Aluminium","E":74.5,"rho":2.85,"k":130,"sigY":397.5},
  {"id":"al6061ram2-lpbf-t6","name":"A6061-RAM2 (6061 + reactive additive)","condition":"Modified T6","process":"LPBF (3D Systems DMP Flex 350, XY)","group":"Aluminium","E":69,"rho":2.7,"k":162,"sigY":260},
  {"id":"cocrmo-lpbf-asbuilt","name":"CoCrMo (F75-type, EOS MP1)","condition":"As-built","process":"LPBF (EOS M290, 40 um)","group":"Cobalt-chrome","E":180.5,"rho":8.3,"k":14,"sigY":940},
  {"id":"cocrmo-lpbf-ht","name":"CoCrMo (F75-type, EOS MP1)","condition":"Stress relieved + solution annealed","process":"LPBF (EOS M290, 40 um)","group":"Cobalt-chrome","E":206.5,"rho":8.3,"k":14,"sigY":635},
  {"id":"cocrmo-lpbf-hip","name":"CoCrMo (ASTM F75, 3DS LaserForm CoCrF75)","condition":"HIP","process":"LPBF (3D Systems DMP)","group":"Cobalt-chrome","E":225,"rho":8.35,"k":14,"sigY":492.5},
  {"id":"ta-lpbf-asbuilt","name":"Tantalum (unalloyed)","condition":"As-built","process":"LPBF (research, Kustas et al. 2025)","group":"Refractory","E":186,"rho":16.68,"k":45,"sigY":477.8},
  {"id":"nb-lpbf-asbuilt","name":"Niobium (unalloyed)","condition":"As-built","process":"LPBF (research, Griemsmann et al. 2021)","group":"Refractory","E":105,"rho":8.58,"k":53.7,"sigY":324},
  {"id":"grcop42-lpbf-hip","name":"GRCop-42 (Cu-Cr-Nb)","condition":"HIP","process":"LPBF (Velo3D, vertical)","group":"Copper","E":115,"rho":8.79,"k":340,"sigY":185.8},
  {"id":"cucrzr-lpbf-ht","name":"CuCrZr","condition":"Tensile-optimized heat treatment","process":"LPBF (EOS M400-1)","group":"Copper","E":122.5,"rho":8.84,"k":315,"sigY":502.5},
  {"id":"pa12-sls","name":"PA12 (EOS PA 2200)","condition":"As-sintered, dry","process":"SLS (EOS)","group":"Polymer","E":1.6,"rho":0.93,"k":0.26,"sigY":null},
  {"id":"pa12-mjf","name":"PA12 (HP 3D HR PA 12)","condition":"As-printed","process":"MJF (HP)","group":"Polymer","E":1.9,"rho":1.01,"k":0.26,"sigY":null},
  {"id":"peek-sls","name":"PEEK (EOS PEEK HP3)","condition":"As-sintered (x/y)","process":"HT-SLS (EOS P 800)","group":"Polymer","E":4.3,"rho":1.31,"k":0.29,"sigY":null},
  {"id":"peek-fff","name":"PEEK (Victrex AM 450 FIL)","condition":"As-printed, XY","process":"FFF (heated chamber >=150 C)","group":"Polymer","E":3.5,"rho":1.3,"k":0.29,"sigY":70},
  {"id":"pekk-sls","name":"PEKK (Arkema Kepstan, research)","condition":"As-sintered","process":"HT-SLS (Benedetti et al. 2019)","group":"Polymer","E":4.3,"rho":1.27,"k":null,"sigY":null},
  {"id":"niti-lpbf","name":"NiTi (Nitinol)","condition":"Any (superelastic or shape-memory)","process":"LPBF","group":"Shape memory","E":78,"rho":6.45,"k":18,"sigY":null}
];
const SYNTH_DEFAULT_MATERIAL = 'ti64-g5-lpbf-hip';   // F13LD.lab's default
