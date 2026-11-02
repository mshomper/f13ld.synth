# 0011 Suite-wide code is copied into each tool, not loaded from one shared place

- Status: found
- Date: 2026-11-02
- Lenses: architecture
- Deliberate: yes
- Holding up: strained. The "◐ view" glyph in the shared F13LD-VIEW block breaks the no-glyph rule; Synth patches it after init and NEXT-UP says the real fix has to go into every tool.
- In the repo's log: CLAUDE.md Code layout; docs/NEXT-UP.md Ideas: shared F13LD-VIEW glyph

## Context

24-f13-shade.js carries the F13LD-SHADE / F13LD-VIEW blocks byte-identical with other F13LD tools (CLAUDE.md); 02-materials.js is generated from F13LD.lab by tools/sync-materials.js. By contrast the F13LD.queue client is loaded at runtime (43-handoff.js loadQueueClient).

## Options

### A: Copy per tool, kept identical by convention (current) (recommended)

Each tool works standalone; sync by hand or script.
- Pro: No cross-site runtime dependency
- Pro: A tool can pin a known-good copy
- Con: Every fix has to be repeated in each tool
- Con: Drift is only caught if someone compares
- Effort: In place

### B: Load shared blocks from one URL

Serve shade, view and materials from a suite origin, as the queue client already is.
- Pro: One fix reaches every tool
- Con: A broken shared file breaks every tool at once
- Con: Needs versioned URLs
- Effort: Medium

### C: Sync script for every shared block

Generalize tools/sync-materials.js to shade/view, with a byte-compare check.
- Pro: Keeps standalone tools, removes hand copying
- Con: Still a manual step per tool
- Effort: Small

## Recommendation

A (no confidence given). CLAUDE.md requires 24-f13-shade.js to stay byte-identical with the other tools and 02-materials.js to be regenerated from Lab.
