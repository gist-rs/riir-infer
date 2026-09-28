# Issue 025 — owner-gate pickup: D7/D8 prep; D9/D10 recorded

**Status:** OPEN — pickup tasks from the workspace owner-gated summary (riir-ai 1016); execution gates per item; prep/recording tasks are agent-pickupable.

Master: `../riir-ai/.issues/1016_workspace_owner_gated_decisions_summary.md` (riir-ai commit e8cbd91ff).
Owner direction 2026-09-28: riir-mmorpg-examples / seal-online-remaster / seal-game-editor / sealm-toolkit are DEFERRED; ALL mainnet actions are ON HOLD.

- [ ] D7 — delete the dead `gpu_transpose` module (`.issues/017_gpu_transpose_module_has_no_callers.md:64`; `transpose_cubecl.rs` from Issue 572 covers the reachable use) AND update the BOUNDARY.md "GPU transpose kernel" ownership line to point at `transpose_cubecl.rs` in the SAME commit. Code task — dedicated follow-up commit, keep tests green. AGENT-PREP.
- [ ] D8 — audio-lane PoC BOUNDARY widening (`.issues/015_fluidaudio_coreml_audio_lane_poc.md:25`); rec: reuse the `objc2-core-ml` allowlist row. Contract change — issue 015 already exists; drift-row discipline per BOUNDARY.md. AGENT-PREP.
- [-] D9 — research 327–332 routing to public katgpt-rs (`.issues/1004_corpus_follows_op_layer_migration.md:34`) — deferred with T7/S8.
- D10 — RECORD-ONLY: S6b training-families long-term home (`.issues/1003_riir_infer_carve_remains_4090.md:5`) — close as by-design unless a consumer pull appears.
