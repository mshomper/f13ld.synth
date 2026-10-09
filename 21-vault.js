/* ============================================================
   F13LD.synth · 21-vault.js
   F13LD.vault row counts for the data-lineage status bar.
   ============================================================ */
'use strict';

// Vault state — counts and lineage only (Synth doesn't query individual designs)
let vaultCounts = null;

// ============================================================
// VAULT CLIENT — paginated load → in-memory counts.
// Used to show data lineage (community data total vs. trained-on count
// vs. new-since-training delta) and for Save-to-Vault candidate submission.
// ============================================================
const Vault = {
  async loadCounts() {
    const fetchOnce = async (offset, chunk) => {
      const url = `${SUPABASE_URL}/rest/v1/f13ld_designs?select=family,created_at,solver_validity,degenerate&order=created_at.desc&offset=${offset}&limit=${chunk}`;
      const res = await fetch(url, {
        headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` },
        mode: 'cors', cache: 'no-store', credentials: 'omit'
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
      return res.json();
    };
    let all = [], offset = 0, chunk = 1000;
    while (true) {
      let rows;
      try { rows = await fetchOnce(offset, chunk); }
      catch(e) { await new Promise(r => setTimeout(r, 800)); rows = await fetchOnce(offset, chunk); }
      all = all.concat(rows);
      if (rows.length < chunk) break;
      offset += chunk;
    }
    const usable = all.filter(d => d.degenerate !== true && d.solver_validity !== 'invalid');
    const byFamily = {};
    for (const d of usable) byFamily[d.family] = (byFamily[d.family] || 0) + 1;
    vaultCounts = { total: usable.length, allRaw: all.length, byFamily, designs: usable };
    console.info(`[F13LD.synth] Vault loaded: ${usable.length} usable / ${all.length} total`);
    return vaultCounts;
  },
  countNewSince(family, sinceDateIso) {
    if (!vaultCounts) return null;
    const since = new Date(sinceDateIso).getTime();
    return vaultCounts.designs.filter(d => d.family === family && new Date(d.created_at).getTime() > since).length;
  }
};
