# 0013 Synth reads Vault but never writes candidates back

- Status: found
- Date: 2026-11-02
- Lenses: data-model, research
- Deliberate: yes
- Holding up: fine
- Open item in the repo's own log
- In the repo's log: README.md The data lineage loop: Save-to-Vault mocked pending schema decision; docs/NEXT-UP.md Ideas: Save to Vault is still a placeholder

## Context

21-vault.js only counts Vault rows for the model chip (public anon key in 00-config.js). The Save to Vault button described in the README is a placeholder "pending a schema decision on candidate tagging"; NEXT-UP still lists it.

## Options

### A: Read-only (current) (recommended)

Vault holds FEA-characterized designs only.
- Pro: Training data stays clean of model guesses
- Pro: No write credentials in a static page
- Con: The lineage loop in the README is not closed
- Con: Good candidates are lost unless sent through Queue or Mesh by hand
- Effort: In place

### B: Write candidates tagged as model-generated

Insert rows marked unvalidated, excluded from training until a sweep characterizes them.
- Pro: Closes the loop; seeds future sweeps
- Con: Needs a Vault schema field and a trainer filter
- Con: Write access from a public page
- Effort: Medium

### C: Route candidates through F13LD.queue to Sweep

Use the existing Queue hand-off as the only path back.
- Pro: No Vault schema change; only validated results reach Vault
- Con: Slower loop
- Effort: Small

## Recommendation

A (no confidence given). 21-vault.js is read-only; README "The data lineage loop" says the button is mocked pending a schema decision.
