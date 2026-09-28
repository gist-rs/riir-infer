# Plan 613 — owner-gate pickup prep (refs riir-ai 1016)

**Status:** OPEN — prep tasks only; D9/D10 are recorded, not worked.

Refs: `../riir-ai/.issues/1016_workspace_owner_gated_decisions_summary.md` (riir-ai commit e8cbd91ff) · Issue 025 (local lane) · owner direction 2026-09-28 (4 repos deferred, no mainnet).

- [x] D7: delete the dead `gpu_transpose` module + repoint the BOUNDARY.md "GPU transpose kernel" ownership line at `transpose_cubecl.rs` in the SAME commit. Code task — dedicated follow-up commit, keep tests green. (EXECUTED 2026-09-28; see `.issues/025` D7 row for the record.)
- [ ] D8: widen the BOUNDARY allowlist for the audio-lane PoC by reusing the `objc2-core-ml` allowlist row; drift-row discipline per BOUNDARY.md (issue 015 owns the PoC record).
- [-] D9: research 327–332 routing to public katgpt-rs — deferred with T7/S8 (`.issues/1004_corpus_follows_op_layer_migration.md:34`).
- Record-only: D10 — S6b training-families home closes as by-design unless a consumer pull appears (`.issues/1003_riir_infer_carve_remains_4090.md:5`).
