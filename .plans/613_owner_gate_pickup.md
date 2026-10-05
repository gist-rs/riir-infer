# Plan 613 — owner-gate pickup prep (refs riir-ai 1016)

**Status:** CLOSED 2026-10-05 — the owner-gate backlog is dispositioned per the delegated verdict (session f9134c0d); the sibling pickup issue is removed (number stays spent); final dispositions live in this repo's HISTORY.md §2026-09-28 (Issue 025) + §2026-10-01 (Issues 998+1004) and the workspace record in riir-ai HISTORY.md §2026-10-05.

Refs: `../riir-ai/.issues/1016_workspace_owner_gated_decisions_summary.md` (riir-ai commit e8cbd91ff; tracker removed 2026-10-05 — surviving record: riir-ai HISTORY.md §2026-10-05) · ~~Issue 025~~ (removed 2026-09-28 per the noise-reduction rule; records: HISTORY.md §2026-09-28 + git history) · owner direction 2026-09-28 (4 repos deferred, no mainnet).

- [x] D7: delete the dead `gpu_transpose` module + repoint the BOUNDARY.md "GPU transpose kernel" ownership line at `transpose_cubecl.rs` in the SAME commit. Code task — dedicated follow-up commit, keep tests green. (EXECUTED 2026-09-28; see `.issues/025` D7 row for the record.)
- [x] D8: widen the BOUNDARY allowlist for the audio-lane PoC by reusing the `objc2-core-ml` allowlist row; drift-row discipline per BOUNDARY.md (issue 015 owns the PoC record). (EXECUTED 2026-09-28; see `.issues/025` D8 row.)
- [-] D9: research 327–332 routing to public katgpt-rs — deferred with T7/S8 (`.issues/1004_corpus_follows_op_layer_migration.md:34`). **Resolved 2026-10-01 — `.issues/1004` closed with the corpus-follows triggers read NEGATIVE (HISTORY.md §2026-10-01); no routing owed.**
- Record-only: D10 — S6b training-families home closes as by-design unless a consumer pull appears (`.issues/1003_riir_infer_carve_remains_4090.md:5`). (RECORDED 2026-09-28 — disposition ratified into 1003's status; see `.issues/025` D10 row.)
