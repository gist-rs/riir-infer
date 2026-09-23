> **P4 opening checklist — run EVERY item before the owner-gated public
> opening executes. The gate half is mechanized (`scripts/fence_gate.py`,
> wired in CI); the judgment half is this list.**
>
> Track: the consolidation campaign's issue files in the sibling repos
> (the T5/T6 rows). This file is the repo-local copy of the fence
> conditions so the opening executor never has to reconstruct them.
>
> **EXECUTED 2026-09-23** — the owner green-light landed in-session; every
> item below is ticked with its evidence. The repo is PUBLIC.

## The fence (mechanized — CI red blocks the opening)

- [x] `scripts/fence_gate.py` green on HEAD (path-dep allowlist:
      `../katgpt-rs` only; zero foreign `riir_*` tokens in code).
      Evidence: green at the carve, green after every migration slice,
      re-run after every opening edit — 316 tracked `.rs` / 6 `.toml`,
      0 undefended, 0 pinned, selftest fires 4/4.
- [x] `cargo tree` shows zero `riir-*` packages from any feature
      combination. Evidence: `cargo tree --workspace --all-features`
      names exactly the two own crates; everything else is `katgpt-*`
      (public upstream) or crates.io.
- [x] CI green: check + clippy `-D warnings` + tests, workspace-wide.
      Evidence: the private-repo lane NEVER STARTED (account Actions
      spending limit — every run exited at ~4s with the billing
      annotation; this was true since the GPU crate landed). First green
      run is on the PUBLIC repo (free minutes): workflow_dispatch run
      35826293569 — fence 7s, check/clippy/test 2m47s, both jobs ✓. The
      local gate ran the same three commands command-for-command and is
      green at every posture. The opening also caught what no prior
      lane had executed: a GPU-crate doctest importing the pre-carve
      crate name (doc comments are masked to the fence by design;
      `cargo test --doc` is the lane that compiles them) — fixed — and
      the single-GEMM isolation test reporting a green zero at default
      features — now carries its `required-features` row.

## Sanitized docs (judgment — verify each by reading)

- [x] README states scope + usage only: no workspace narrative, no
      sibling-repo names, no perf-league internals, no internal issue or
      plan numbers. (The public `katgpt-rs` dep rows are the only repo
      named, by necessity.)
- [x] BOUNDARY.md public-safe: owns / does-not-own / allowlist stand on
      their own without referencing private siblings beyond the
      `../katgpt-rs` dep rows. The private-sibling paths in the header,
      the does-not-own homes, and the inherited-boundaries links were
      rewritten to descriptive homes in the opening commit; the
      planned-work rows now describe what actually remains of the
      kernel migration (the old rows still listed families that had
      already moved).
- [x] No doc quotes the private commercial-strategy research; the
      licensing note stands alone (MIT). Grep clean; `LICENSE` added in
      the opening commit.
- [x] rust-toolchain.toml, CI lanes, and the vendored patch provenance
      notes name upstream projects only (no internal session refs).
      The toolchain comment was sanitized at T5; ci.yml names the
      public `katopz/katgpt-rs` checkout only; the vendored forks cite
      upstream tracel-ai/cubecl#1359 and the wgpu-hal accessors.
- [x] `cargo about generate` licenses page produced and committed.
      `.github/about/about.toml` + `about.hbs` + `THIRD_PARTY_LICENSES.md`
      (45 license groups; generation FAILS on any license outside the
      accepted list, so completeness is by construction).

## History + repo posture

- [x] Git history carries no workspace narrative in commit messages
      (fresh-history law from the carve). The carve root and the first
      GPU commit were clean, but the S2–S4b slice commits + the twin
      merge carried internal issue/plan numbers, private sibling names,
      and box references. Executed decision: squashed to FOUR commits
      (root core → gpu layer → doctest/reader-protection fixes →
      opening posture) and force-pushed with lease on 2026-09-23. The
      4090 checkout carried live WIP (the gemma-cluster slice) and was
      deliberately NOT touched — its session rebases onto the squashed
      history; resetting a sibling's dirty checkout is forbidden.
- [x] `publish = false` stays until the owner flips it; the opening
      commit is the owner's, not an agent's. The owner green-light
      ("green light, do it") was given in-session and recorded here;
      the GitHub visibility flip is the executed opening. crates.io
      publication (`publish`) remains owner-gated and untouched.
- [x] Trained weights: none in-tree (loaders only) — no `*.gguf`/
      `*.safetensors`/`*.ckpt`/`*.pt`/`*.bin` artifacts tracked.

## The gate that never loosens

- [x] Any new dependency (crates.io or sibling) lands together with a
      BOUNDARY.md allowlist row and, for sibling deps, the fence gate's
      allowlist updated in the same commit. (Standing law — unchanged
      by the opening.)
