# 0010 Numbered classic scripts in one global scope, no build step, hosted on GitHub Pages

- Status: decided
- Date: 2026-11-02, decided 2026-11-02
- Lenses: architecture
- Deliberate: yes
- Holding up: fine
- In the repo's log: README.md Repo structure; CLAUDE.md Code layout
- Settles: front-end build = numbered classic scripts, no bundler

## Decision

**A: Numbered classic scripts (current).** Shared global scope, load order by file number.

## Context

index.html loads 00- to 99- scripts in order; each may only use names from same- or lower-numbered files (tests/loadorder.js). The worker importScripts 10/11/12. tools/build-single.js makes a one-file preview. Same layout as F13LD.mesh, lab and sweep.

## Options

### A: Numbered classic scripts (current) (chosen) (recommended)

Shared global scope, load order by file number.
- Pro: Matches the rest of the suite
- Pro: No tooling; edit and reload
- Pro: Worker reuses the same files
- Con: Global namespace collisions possible
- Con: Load order is a convention checked by a test
- Effort: In place

### B: Native ES modules, no bundler

import/export with module workers.
- Pro: Explicit dependencies, no load-order rule
- Pro: Matches the convention another project settled
- Con: Diverges from the other F13LD tools
- Con: Single-file build needs rework
- Effort: Medium

## Recommendation

A (no confidence given). README "Repo structure" and CLAUDE.md "Code layout" state it; tests/loadorder.js enforces it. Note: ~/crux/conventions.md settles "front-end build" as plain ES modules for AcuLoad, so confirming this records a cross-project difference.
