# Plan 001 — EXL3 T7b: the Metal/wgpu dequant GPU arm (Issue 001)

**Status:** CLOSED as TWIN-DUPLICATE — 2026-09-24 (M3 session).

## Verdict (the Batch-169 twin rule)

While this session built its T7b kernel on the M3, a sibling session landed
its OWN T7b in parallel: `af881ac` ("feat(001): T7b GPU arm — CubeCL EXL3
dequant, decode bit-exact + 71-87x wall vs CPU") plus T7c (`e80c051`,
`9499e40` — the fused trellis GEMV lane), all on the 4090, and a docs sync
(`0089033`). Origin's commit is canonical; the duplicate's work died in the
working tree per the owner's TWIN call (kept untracked, then deleted).

**What this session's duplicate contributed that the canonical one also
measured independently (cross-validation, not new code):**
- Decode stage bit-exactness reproduced on M3 Metal/wgpu-msl (K=4 int +
  K=2.5 half) — independent confirmation of the canonical kernel's
  contract on a SECOND CubeCL runtime.
- The Hadamard-stage divergence mechanism identified independently:
  wgpu-hal compiles MSL with `fast_math_enabled = true` by default, so
  Apple's compiler contracts `acc += a·b` into FMA at every accumulation —
  matching the canonical module's documented "both CubeCL backends
  contract a*b+c" note (measured here: 2–10 ulps on 75–90% of elements,
  max |Δ| 9.5e-7 at value scale 2–3). The canonical gates (rel-Fro 1e-5)
  cover the same class.
- The same LCG fixtures, LUT upload, and tensor-core perm math were derived
  here from the pin — no conflicts with the canonical record.

## The clobber this session found and repaired (the real finding)

The sibling's `3ce2d75` ("chore: rustfmt sweep") accidentally removed the
`pub mod exl3_dequant_cubecl;` declaration from
`crates/riir-infer-gpu/src/lib.rs` AND the `exl3_gpu` feature row from
`crates/riir-infer-gpu/Cargo.toml` — landing the T7b module ORPHANED at
HEAD: the file existed but compiled to nothing, `--list` showed 0 of its
tests, and the feature row was gone from the manifest. This session's own
duplicate rows had (unwittingly) replaced the sibling's in the same files,
which is how the working tree read as "mine" while HEAD read as theirs.

**Repair (this session's commit):** restore the declaration + feature row
verbatim from `af881ac`; delete the duplicate module + this plan's code.
Validated on M3 Metal: the canonical module compiles again and **7/7 of its
non-CUDA tests pass** (`exl3_dequant_cubecl::tests::*`, 3 ignored are the
CUDA-only rows) — the clobber fix is proven by the orphaned tests actually
running again.

## Lessons (for the shared-worktree ledger)

1. A rustfmt/formatting sweep commit that touches `lib.rs`/`Cargo.toml` in
   a repo where ANOTHER session has uncommitted rows in those exact files
   will silently drop the sibling's lines (the sweep's tooling read the
   working tree). Sweep commits should `git diff --stat` their manifest +
   module-tree files against HEAD and name every deletion before landing.
2. "Module file exists + feature exists in Cargo.toml" is not "module is
   compiled": the only check that catches an orphaned module is
   `cargo test --features X --lib <module>:: --list` (or any compile of the
   feature posture). The green-zero rule, one level down.
3. Two sessions building the same issue's next task in parallel is the
   second TWIN in this workspace inside a week (Batch-169 was the first).
   The issue file's task rows are the claim mechanism — annotate the row
   you are taking BEFORE writing code.
