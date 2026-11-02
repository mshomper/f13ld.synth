# 0005 Retraining is a manual run on Matt's machine, with weights committed to the repo

- Status: decided
- Date: 2026-11-02, decided 2026-11-02
- Lenses: architecture, research
- Deliberate: yes
- Holding up: fine
- In the repo's log: README.md Retraining: Pattern A

## Decision

**A: Manual local retrain, weights in git (current).** Matt runs the trainer when the Vault drift counter says so, then commits.

## Context

docs/HOW-TO-retrain-synth.md has Matt run train_synth.py --from-vault locally and commit weights/tpms.json; the page picks it up on next load. The README calls this Pattern A and plans a GitHub Actions retrain (Pattern B). Filters such as --max-norm 1.3 and --solver-version decide which Vault rows train.

## Options

### A: Manual local retrain, weights in git (current) (chosen) (recommended)

Matt runs the trainer when the Vault drift counter says so, then commits.
- Pro: Full control and a human looks at every R² before it ships
- Pro: Rollback is a git checkout
- Con: Retraining cannot happen in a session VM
- Con: Large binary-ish file in every retrain commit
- Effort: In place

### B: GitHub Actions retrain opening a PR (Pattern B)

A button runs the trainer on Actions against Vault and opens a PR with the new weights.
- Pro: Repeatable, logged, anyone can trigger it
- Pro: Review still happens on the PR
- Con: Needs Vault credentials in Actions secrets
- Con: Still commits the weights unless paired with release assets
- Effort: Medium

### C: Weights as release assets or object storage

Keep the trainer manual or automated but publish weights outside git history.
- Pro: Repo stays small
- Con: Page needs a versioned URL; rollback is less obvious
- Effort: Small to medium

## Recommendation

A (no confidence given). README "Retraining" names Pattern A as today's state; docs/HOW-TO-retrain-synth.md and CLAUDE.md say retraining runs on Matt's machine.
