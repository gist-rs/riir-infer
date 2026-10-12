# Issue 042 — the cubecl #1806 idle-server park fix at our vendored pin (the pin-bump lane)

**Status:** LANDED this box — gates green (see Progress); the laya cubecl_ops_smoke leg is Windows-link-blocked (pre-existing, unrelated) and rides the M3 lane.

## Why now

The idle loop queued this lane across cycles 38–41 ("the cubecl pin-bump re-gate
waits on their WIP clearing"): cubecl #1806 (`perf(channel): park the idle device
server and wake it when a buffer fills`, upstream `6bba5fdf0`, merged 2026-10-11)
fixes the wgpu-posture device server's idle loop — after the spin (8192) and yield
(64) budgets exhaust, the server slept in fixed 150 µs steps, so a host that
submits a few kernels, waits, then submits again finds the server asleep each time
and every buffer it fills waits out the rest of a sleep. Upstream measured ~20%
decode on an M2 Pro (small model, wgpu posture). The blocker was the 038
sibling's WIP holding riir-infer; that lane landed in full (bench 035 + the
step-4 standby disposition at `4db9e15`) and the tree is clean — the lane is
executable.

**No published release carries the fix**: cubecl-common 0.11.0 final was
published 2026-10-06T15:35Z, #1806 merged 2026-10-11T14:22Z — the fix exists only
on main. A pin-bump to 0.11.0 would pull two months of unrelated movement and
still not carry it. The execution is therefore the established vendor pattern
(the `vendor/cubecl-runtime-0.11.0-pre.2` #1359 precedent): vendor the published
`cubecl-common-0.11.0-pre.2` source and apply the upstream patch verbatim.

## The change

- [x] Vendor `cubecl-common-0.11.0-pre.2` (published crate from
      static.crates.io, minus `Cargo.lock` + `.cargo_vcs_info.json` per the
      runtime-vendor convention) under `vendor/`.
- [x] Apply the upstream #1806 patch verbatim (`git apply -p3` of the commit's
      `.patch` — all 8 hunks cleanly at offset 21; the custom_channel module is
      byte-identical between pre.2 and main at every hunk context).
- [x] Standalone validation: the patched crate's own lib suite **95/95 PASS**
      (flushing-at-limit, shutdown drain, the panic battery — the exact surfaces
      the patch touches).
- [x] `[patch.crates-io]` row + provenance note in the root manifest.
- [x] Re-gate riir-infer: `cargo check` (default) + the riir-infer-gpu
      cubecl-posture compile + clippy at the posture.
      **DONE 2026-10-13**: `cargo check -p riir-infer-gpu --features
      cubecl_runtime` Finished 1m17s (`cargo tree` confirms the vendored path
      copy resolves); clippy `--all-targets` at the posture clean (only the two
      standing infra warnings — wgpu-hal hardlink + katgpt-percepta, the
      documented noise class). **Plus**: the full lib suite at the posture —
      `cargo test -p riir-infer-gpu --features cubecl_runtime --lib` =
      **243 passed / 0 failed / 1 ignored** (490.99s; every test boots device
      runners through the patched channel).
- [-] Re-gate the cubecl posture on the GPU: `cargo test --release -p
      riir-infer-laya --features laya-riir-cubecl --test cubecl_ops_smoke`.
      **WINDOWS-BLOCKED (pre-existing, NOT this patch)**: every laya-lane exe
      fails to link on this box — LNK2038 `RuntimeLibrary` mismatch
      `MT_StaticRelease` (libesaxx_rs, via `tokenizers v0.22.2`) vs
      `MD_DynamicRelease` (libsentencepiece_sys, via `riir-infer-core`).
      Deterministic (survives `cargo clean -p` of both sys crates + rebuild);
      cubecl-common is pure Rust and cannot affect C++ CRT selection; the
      reflex CUDA lane ran green from REFLEX's own target/lock. The smoke's
      canonical posture is the M3 anyway (Bench 006 home) — **the M3 lane owes
      the smoke re-run + the reflex G5 re-run at both postures** (the standing
      "any lane dep bump re-runs G5" duty — this vendor bump counts).
- [x] Mirror the row in riir-ai (Cargo.toml patch table, pointing at
      `../riir-infer/vendor/…` — the Issue-1046 canonical-fork-home pattern,
      NOT a local twin) — riir-gpu is the other wgpu-posture consumer.
      **DONE**: `cargo check -p riir-gpu --features cubecl_runtime` Finished
      1m50s clean from this box.
- [x] Mirror the patch row in riir-refine (points at `../riir-infer/vendor/…`
      with the existing cubecl-runtime/wgpu-hal rows) + resolution check.
      **DONE**: `cargo metadata --features ternary_inference` resolves clean
      (compile-level proof rides on infer+ai checking the same vendored tree).
- [x] BOUNDARY.md vendor-row update (riir-infer; riir-ai's BOUNDARY carries no
      vendor enumeration — its rows live in the manifest's provenance
      comments, matching the Issue-1046 precedent) + commits + pushes.

## Notes

- The fix is HOST-side (channel server thread) — it cannot change GPU
  numerics; the re-gate is for compile + liveness (the park/unpark SeqCst
  handshake) not correctness of results. The league lanes are cudarc — no
  scored cell moves, no re-pin duty (consistent with the cycle-37 grading).
- Kill-switch posture: none needed (behavioral supersession of a sleep with a
  bounded park of the same timeout); if a pathology appears, removing the
  patch row restores the registry crate.
- Removal trigger: the first cubecl release whose version is > 0.11.0 (or a
  0.11.0-pre.x published after 2026-10-11) — bump the pin and drop all three
  patch rows + vendor trees in one commit.
