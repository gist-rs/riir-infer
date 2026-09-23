> **P4 opening checklist — run EVERY item before the owner-gated public
> opening executes. The gate half is mechanized (`scripts/fence_gate.py`,
> wired in CI); the judgment half is this list.**
>
> Track: the consolidation campaign's issue files in the sibling repos
> (the T5/T6 rows). This file is the repo-local copy of the fence
> conditions so the opening executor never has to reconstruct them.

## The fence (mechanized — CI red blocks the opening)

- [ ] `scripts/fence_gate.py` green on HEAD (path-dep allowlist:
      `../katgpt-rs` only; zero foreign `riir_*` tokens in code).
- [ ] `cargo tree` shows zero `riir-*` packages from any feature
      combination.
- [ ] CI green: check + clippy `-D warnings` + tests, workspace-wide.

## Sanitized docs (judgment — verify each by reading)

- [ ] README states scope + usage only: no workspace narrative, no
      sibling-repo names, no perf-league internals, no internal issue or
      plan numbers.
- [ ] BOUNDARY.md public-safe: owns / does-not-own / allowlist stand on
      their own without referencing private siblings beyond the
      `../katgpt-rs` dep rows.
- [ ] No doc quotes the private commercial-strategy research; the
      licensing note stands alone (MIT).
- [ ] rust-toolchain.toml, CI lanes, and the vendored patch provenance
      notes name upstream projects only (no internal session refs).
- [ ] `cargo about generate` licenses page produced and committed.

## History + repo posture

- [ ] Git history carries no workspace narrative in commit messages
      (fresh-history law from the carve).
- [ ] `publish = false` stays until the owner flips it; the opening
      commit is the owner's, not an agent's.
- [ ] Trained weights: none in-tree (loaders only) — verify no
      `*.gguf`/`*.safetensors` artifacts are tracked.

## The gate that never loosens

- [ ] Any new dependency (crates.io or sibling) lands together with a
      BOUNDARY.md allowlist row and, for sibling deps, the fence gate's
      allowlist updated in the same commit.
