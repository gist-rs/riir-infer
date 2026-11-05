# HISTORY.md — riir-infer

Compacted 2026-10-30: every `##` heading kept verbatim, one compact entry per record (≤2 short lines); the full pre-compaction text is in git history. Operational rules live in `AGENTS.md`; removed issue files: git history.

## 2026-11-05 — Issue 1164 CLOSED + removed: the T5 mixed-mode mask PROVEN at model level — Cell D re-cell reads fnv 0/6 (was 6/6 vacuous), anchor intact

- The closure bar the fix `a9aeedd` (any → all predicate, 2026-11-04) named: re-run bench 1157's Cell D through the same harness (`bench_1156_t6_league_cells`, name-filtered, one process, release, amo QUIET, preflight PROVENANCE quoted, runtime 3902 s, test ok). Result: **D_mixed085_heads0-4 fnv==dense 0/3 at BOTH lengths** (the vacuity is gone — the mask genuinely routes at model level); walls between C and dense (1.0243× @16K / 1.0408× @32K, report-only posture unchanged); **argmax-flips 0/3** (mixed quality = dense at argmax); **G1 anchor τ=1.0 ≡ dense held 6/6** (the identity contract proves the pipeline stayed sound); engagement counters moved, regime refusals 0; **Cell C revalidated** (1.0333×/1.0667×, 0 flips — the 1165 promotion posture intact on the fixed tree). Dated row on riir-ai bench 1157 (where the vacuity was born). Claude verdict not needed — the issue's own closure bar was this cell; claim zed-ablate@amo folded (claimed + re-celled + closed in one session, epoch 1794247200).

## 2026-11-05 — Issues 1170 + 1171 + 1172 CLOSED + removed: the dense-lane prefill campaign's three landed lanes — half-B past the fork (bench 941), tensor-flash attention (942), the GDN-bucket census (943)
## 2026-11-05 — Issue 1150 CLOSED + removed: the Metal dense-qwen3.8 forward arc complete — fork-parity delivered, the league posture is the default, G2 table recorded (Benches i1150 + 938–946)

- The arc's stale "IN PROGRESS" status line stopped at the bimodality question; every task was already done. Full landing sequence, each with its record: decode+parity G1 (i1150, 10-28) → prefill unlock dense-f16 matmul2d (i1150, 10-29) → stage-timers + ragged-tail bench fix (`306540e`/`5b41121`) → bimodality RESOLVED as posture-split not process variance (the audit, i1150 addendum, bmo exclusive `01a1a479-5f4f`) → GDN chunked route parity + the shared `cum_decay` floor bug fixed (bench 938) → Q6K packed tail 13× (939) → relaxed_precision 2× (940) → half-B past the fork (941) → tensor-flash attention (942) → GDN-bucket census (943) → fused norm family closed-negative (944) → free-running ablation matrix (945) → the league env set promoted to the METAL DEFAULT, bare boot 966 tok/s vs the 7 tok/s incumbent (946). **G2 same-model positioning table recorded 11-05 at the promoted posture** (exclusive amo job `0bf2c246`, cross-box-labeled vs the bmo-recorded three-way): ours beats the fork at every cell ≥2.5k (1.08–1.77×), flat 900–970 where uzu collapses long-P, 0.64–0.69× MLX through 10k — the fork-parity bar DELIVERED; MLX named the remaining gap. Residues (cross-backend cudarc parity G1 cell on the 4090 + the parked dense-f16 decode GEMV lever) filed as **riir-infer Issue 1177**. Claude verdict round 1 AGREE on the closure batch. Session zed-ablate@amo.

## 2026-11-05 — Issues 1170 + 1171 + 1172 CLOSED + removed: the dense-lane prefill campaign's three landed lanes — half-B past the fork (bench 941), tensor-flash attention (942), the GDN-bucket census (943)

- **1170 (dense-lane GEMM engine to MLX-class)** — lever 1 LANDED (bench 941, zed-amo-i1170-halfb@amo): the staged-activation half×half matmul2d over f16 twins (`RIIR_QWEN38_DENSE_MM2D_F16ACT`, promote-only then) answered the G2 physics question YES at kernel level (1.31–1.46×) and went past the fork at the short cells (pp512 640 / pp2560 725 / pp4954 580 tok/s); the arm is now part of the league Metal default (bench 946). Lever 2 (dense-f16 decode GEMV, ~8× off bandwidth, explicitly parked non-serving) survives as the pointer in riir-infer Issue 1177. All acceptance-gate residues cross-filed there.
- **1171 (tensor-flash prefill attention)** — both halves LANDED (bench 942, zed-amo-i1171-attn@amo): the long-p attention census + the Issue-1146 matmul2d attention wired into the dense lane (`RIIR_QWEN38_ATTN_TENSOR`, hosts on the dense context); past the fork at its own league cell; part of the league Metal default (bench 946). Bench 945's free-running attribution later corrected the record's StageTimers shares with dated ⚠ rows (non-GEMM ~131 ms/chunk, small kernels ≤1–2%).
- **1172 (GDN-bucket GPU-timestamp census)** — instrument + census LANDED (bench 943, zed-986a7538@amo): GPU counter-sample buffers on the shared in-order queue, hazard-pinned marker passes, zero in-pass syncs; pinned the non-GEMM share as real kernel wall with a per-kernel launch floor. Bench 945's ablation matrix later REFUTED the per-section floor reading (marker-boundary instrumentation cost; free-running small kernels 0.0–0.1 ms/layer) with dated ⚠ rows — the record's census tables stand as the serialized-instrument view.
- Claude verdict round 1 AGREE (2026-11-05) on the closure batch (1150 closes separately once its G2 cells record; 1108/1117/1164/1167/1168 stay open with reasons).

## 2026-11-05 — Issue 1176 CLOSED + removed: the league env set promoted to the METAL DEFAULT — bare boot 966 tok/s (was ~7), retention clean 272/272 (Bench 946)

- The five-env league posture (`DENSE_MM2D` / `LMHEAD_Q6K` / `DENSE_MM2D_RELAXED` / `DENSE_MM2D_F16ACT` / `ATTN_TENSOR`) flipped DEMOTE-ONLY on macOS+metal_tensor_gemm (exact `=0` each restores the incumbent; promote-only `=1` elsewhere; dependent arms refuse loud naming both fixes; demote banner added; the relaxed selector in `gemm_dense_f16_metal_zero_copy.rs` is the 5th site, module-only so no platform split inside). Verdict round 1 AGREE, 4 conditions honored: (1) per-family retention — 6 families (prose/math/code/multilingual/structured/needle-recall) × greedy walks vs the all-`=0` bit-exact incumbent: **zero first divergence, 272/272 top-1 agreement** — no family drops, the lossy-arm fallback NOT invoked; (2) caller/test audit — the forward's one non-bench caller is none (bench bin only); (3) loud per-platform banners; (4) refusal wording. G1: bare pp5120 **966 tok/s** vs LEAGUE 957 (±0.5–1% spread, byte-identical walks prose 48/48 + math 48/48) vs ALL0 **7 tok/s** (755 s/pass, bench 944's 736 s reproduced; pp512 75.7 s = the class is shape-flat). The ~100× bare-boot trap retired; the incumbent stays one documented `=0` away. Commit `d19223c` (on the wip `438eeab` dial flips). Record: `.benchmarks/946_issue1176_default_posture_promotion.md`. Lease folded: zed-ablate@amo, epoch 1793804639.

## 2026-11-05 — Issue 1165 CLOSED + removed: the tau=0.85 envelope-sparse prefill promotion LANDED (Metal/wgpu, >=16K) — P1-P6 all done, bench 1158 PASS

- Every task was already `[x]` (the Status line was stale — the promotion landed 2026-11-02: P1+P2+P3 `657b9caf` (default-on >=16384, demote-only `RIIR_PREFILL_SPARSE=0` byte-identical, default tau 0.9->0.85, the embedded digest-pinned v2 manifest posture gate with loud dense fallback + refusal counter) and P5+P6 `56025324` (the default-posture re-cell BIT-EXACT vs explicit tau0.85 at 16K+32K, bench 1158; the 11-site dense-pin survey over ai/train landed in the same commit family). Claude owner-proxy verdict AGREE 2026-11-02 (4 conditions, all honored). P3's env-FNV deviation honest-recorded (the demote path IS the G3 anchor). Claim line closed: zed-7710a782's expired lease (epoch 1793685200) — the work landed under zed-7710a782 + zed-0ca3214b before expiry mattered. Residual watch: the verdict's non-blocking follow-ups (bmo re-measure, mixed-corpus speed cells) stay with their own lanes.

## 2026-11-04 — Issue 1175 CLOSED + removed: the zero-length-binding launcher audit — 2 live sites fixed, the fused-twin offset bug found, 1 cross-repo site fixed in riir-ai

- Audit: 1102 `BufferArg::from_raw_parts` sites; exactly 2 literal-zero binds (both `RmsNormQkFusedCubeCL`'s unused slots — safe by accident, the kernel routes by `params[3]` and never length-reads). Fixed to one-element γ views (offset 0, in-bounds by construction) + the module-header law in `norms_cubecl.rs` (never 0-length; explicit params flag, never slot length). Audit BONUS: the 3 fused twin launchers rode the TWIN layout offset (`r0·dim·2`) on their twin-less γ fallback — past γ's end for any chunked twin-less launch (in-bounds only because the G1 shapes never chunked); deduped into the shared `optional_twin_binding` helper (offset 0 on the fallback). Cross-repo: riir-ai `riir-gpu/lora_fused_cubecl.rs` `launch_fused_b` bound `packed_b` len 0 ("size not needed" — backwards); fixed to the honest full length, **sibling commit `84239dbd2`** (behavior-identical, 4/4 lora tests green; cherry-picked in a detached worktree past a concurrent sibling push). Second-order check: ~15 length-reading kernels all launch with loop-guarded nonzero lengths — no other zero-length-capable site. Validation: fused G1 3/3 bitwise (twin-less arms), norms 15/15, full gpu lib **314/0** on Metal. Near-miss recorded: the brace surgery initially dropped two `r0 += rc;` loop advances — clippy's `while_immutable_condition` caught both before commit (the compile gate earns its keep). Session zed-ablate@amo, epoch 1793804639.

## 2026-11-04 — Issue 1174 CLOSED + removed: the free-running ablation instrument landed — the wall is GEMM-owned (~61%), the census non-GEMM share was marker-boundary cost (Bench 945)

- Instrument (`e59d59d`): `qwen38_ablate.rs` (27 launch-site classes, census-section aligned; all/csv parse, refuse-loud unknowns; 4 unit tests incl. the census-mapping drift pin) + 26 geometry-preserving skip gates in `qwen38_dense_cubecl.rs` (`RIIR_QWEN38_ABLATE` env with loud construction-time refusals of every launch-set-changing posture: census / stage timers / FUSE_* / GDN chunked) + `qwen38_wgpu_bench --ablate-matrix` (interleaved base/ablated pairs per class — sequential arms measure the box). Matrix at the promoted league posture, amo: base wall median 526.3 ms/chunk (973 tok/s, reproduces 944's baseline twice); Σ single-class 503.5 vs joint removal 498.3 → **+1.0% = additive within noise**; GEMM family ~321 ms ≈ 61% of the wall (gateup 135.6 + down 76.8 + qkv 62.0 + ssm_out 21.6 + attn_qkv 16.7 + attn_out 8.1; MLP pair alone 40%); GDN dataflow trio scan/conv/expand 127 ms = 24%; ALL floor **28 ms/chunk (5.3%) measured directly** (the ALL cell's pair deltas are structurally drift-inflated — near-idle pass → GPU clock drop inference — and must not be read as the floor); the small-kernel family the 943 census priced at ~0.5-0.7 ms/section measures 0.0-0.1 ms/layer free-running (5-50× boundary inflation; net 3.4%) — **943's "~403 ms non-GEMM share is REAL" is REFUTED per section; launch reduction is closed as a lever family** (measured twice: 944 + 945). Mechanism for the census-GEMM-LOW pattern: verified per-dispatch wrapper cost (client.flush + own MTLCommandBuffer + dedicated output + full P×M f32 blit copy-back, `gemm_dense_f16_metal_zero_copy.rs`) + a census time-moved pattern (GEMM section low ↔ next small section high); residue honestly UNSETTLED — the MLP share reads "GEMM-class work", kernel-vs-wrapper split = the direct-store GEMM cell, adjacent to the 1167 lane (deliberately not filed). Round-1 verdict fix en route: the gateup site's input cast moved outside its gate (double-count; re-read 138.45 → 135.56, consistent with the removed cast). Benches 942/943 carry dated ⚠ rows pointing here. Claude verdict round 1 REVISE (4 blocks) → all applied + confirmation run (`scan` 55.66 vs 54.41 reproduced; base identical) → round 2 AGREE, `0de2cc2`. Record: `.benchmarks/945_issue1174_free_running_ablation_matrix.md`. Lease folded: zed-ablate@amo, epoch 1793804657, expires 2026-11-06T15:04:17Z.

## 2026-11-04 — Issue 1173 CLOSED-NEGATIVE + removed: the fused norm+gating family measured sub-bar — the launch-floor lever REFUTED at e2e (Bench 944); the census's per-section walls are marker-boundary artifacts

- Three batched fused kernels (F1 norm_ssm+z_gate, F2 norm+f16-twin, F3 residual+norm, incl. the F3 cross-layer arm) landed BIT-IDENTICAL (G1 `tests/qwen38_fused_norm_gating_g1.rs` 3/3 GPU-vs-GPU bitwise f32+f16+twin-less; e2e greedy walks digit-exact on the sequential AND batched paths, `5bb84f1`), then measured FAR under the pre-registered ≥ +2%/fusion bar: steady medians pp5120 baseline ~971 / F1 ~976.5 / F1+F2 ~981 / F1+F2+F3 ~978.5 tok/s — per-fusion deltas within the ±0.5% spread, family ≤ ~1% (never default; arms stay opt-in `RIIR_QWEN38_FUSE_SSM_ZGATE/_NORM_CAST/_RESID_NORM`, exact-literal-1 promote-only). Mechanism finding: the census rows do NOT collapse under fusion (norm_ssm+z_gate 1.12 ms fused = 1.12 unfused) — 943's per-section "launch floor" and 942's StageTimers non-GEMM share are serialized-instrument boundaries, not removable wall; free-running marginal launch cost ≈ 0.02–0.05 ms/launch; the ~403 ms/chunk share has NO free-running owner → issue **1174** (the non-serialized attribution instrument) must land before the next lever. Instrument law landed en route: ZERO-LENGTH buffer bindings are clamped to the WHOLE buffer by the wgpu/Metal layer (the twin slot binds a one-element γ view + a params[3] flag, never slot length — pinned by the G1 twin-less arms) → issue **1175** (the launcher audit; `RmsNormQkFusedCubeCL` safe by accident). The bare-default ~7 tok/s vs ~971 promoted posture gap → issue **1176** (promotion-debt defect). F4 deferred with a corrected premise (conv1d intervenes between beta/decay and expand). Benches 942/943 annotated in place (append-only dated rows). Claude verdict round 1 REVISE → all four revisions applied → round 2 AGREE, closed `c9245db`. Record: `.benchmarks/944_issue1173_fused_norm_gating_ab.md`. Session zed-1ee7a33e@amo, epoch 1793797957.

## 2026-11-04 — Issue 1169 CLOSED-NEGATIVE + removed: the weight-delta transport PoC's Phase-0 kill branch FIRED — real checkpoints are dense in byte space, no protocol gets built

- T0.2 measured 3/3 real plan-452 checkpoint pairs: flipped-byte 92.7–94.4%, RLE/varint mask payload 1.58–1.60× zstd(full artifact) at pinned levels 3+19 — the mask is LARGER than the whole artifact; 0/3 on the pre-registered ≥2-sparse bar. The XOR/RLE codec + epoch-refusal vocabulary sketches live in the issue text if a sparse-regime source ever materializes (per-step RL checkpoints are the only plausible class — Research 415's value-space sparsity does not survive byte-space f32 mantissa churn on re-train artifacts). **riir-train Issue 667 closed-NEGATIVE via the same table** (`6c133454`) — one measurement killed both speculative protocols. Records: `.benchmarks/i1169_phase0/`. Session zed-agent@amo, epoch 1793697600 (lease `5f7d574`).

## 2026-11-02 — Issue 1166 CLOSED + removed: the tile-geometry residual answered NEGATIVE — the promoted ktile64 is the composed-neighborhood optimum (Bench 936)

- The second-generation sweep (`tests/issue1166_tile_geometry_sweep.rs`, `d9f8d9d`/`6941f77`): 1113's winner crossed with the axes 1113 varied (ktile128; 64×256 and 128×128 AT K64; 8sg/2sg; K32 floor) — G1 worst_rel=0.000e0 all 7 configs; G2 interleaved rotating-start medians on bmo EXCLUSIVE (the first attempt died exit 101: sentencepiece-sys needs cmake and rig job shells lack `/opt/homebrew/bin` — the script exports it now; 1167's census hit the same wall). Verdict: ktile128 noise-band 0.998–1.015× (fails the ≥1.5% e2e + no-long-P bar), wide tiles ~0.56–0.74× (occupancy), simdgroup count not a lever (8sg ≈ 1.0, 2sg ~0.65), K-tile a weak axis (K32 ~0.99). The 1146 pp4096 residual is NOT tile geometry; the surviving ground is Issue 1167 (packed-byte dequant load, H5 cooperative-tensor, census re-run — lease `2c053ec`). Rungs 2–4 were reassigned to 1167 per the lane partition (`23b407f`). AGENTS.md lane row amended. Record: Bench 936 + `i1166_tiles/sweep_run.log`. Session zed-amo-1166-tiles, epoch 1793712000 (claim `f97dcbc`).

## 2026-11-02 — Issue 1146 CLOSED + removed: the 1-box prefill throughput gap — 0.13–0.16× → 0.76×/0.93×/1.00× of llama.cpp (parity at pp32k) in one day-arc of measured, GOAT-gated rungs

- Arc (every rung measured before promotion, per-flag discipline): root cause the dist `gpu` row omitted the prefill-lane features (`412c887`, dispatch arithmetic to the digit, Bench 914) → width re-curve 128→512 1.39–1.42× 1-box (919) → tensor zero-copy GEMM PROMOTED `54f0dce` (920; + the wgpu lazy zero-init cold-pool corruption fix `3bbf275`) → H4 hoisted-load dequant PROMOTED `29be9a0` (921, bit-exact) → relaxed matmul2d PROMOTED dist-scoped (923, 1.96–2.19×, per-family KL max 3.83e-6) → census: attention 37% of the promoted pp4096 frame (925) → **ladder 4(c) matmul2d flash-attention PROMOTED dist-scoped `3499ec4` (Bench 934: 1.046×/1.483×/1.953× at pp512/pp4096/pp32768 — wins GROW with length, the 4(a) cmma long-context rejection inverted; KL max 6.67e-7 nats, top-1 16/16 ×4, e2e G1 identical; demote `RIIR_DIST_PREFILL_ATTN_TENSOR=0`; prefill-only by construction — one call site in the prefill ladder, decode never routes through it)** → embed-batched measured NULL, stays opt-in → pp32k settle: tensor 1.22× FASTER relaxed-OFF, guard-clean (the `401af78` containment-fix claim proven) → **2-box TB5 width leg (Bench 935, `0e63401`): the wire curve CROSSES in (1024, 2048) — w128 owns ≤1k via small-frame pipeline overlap (975 tok/s at pp512, faster than 1-box: both GPUs parallel), w512 owns ≥2k via wide-frame GEMM; the dist GPU default landed PROMPT-LENGTH-CONDITIONAL (128 < 2048 ≤ 512, the `min_batched` precedent; `--prefill-chunk` overrides; both branches functionally verified on the live wire)**. The family row lives in AGENTS.md's lane table (four promoted arms + demote literals). Residual (pp4096 0.93×) is kernel-internal follow-up — a new-kernel lane (matmul2d tile geometry at prefill shapes; H5 cooperative-tensor absolute-win class), not an armed cell. Records: Benches 911–925, 934, 935; artifact dirs `.benchmarks/i1146_*/`. Session zed-amo-1146-finish, epochs 1793612565 + 1793636000 (claims `3f4f828`, `e65acdb`).

## 2026-11-02 — Issue 1163 CLOSED + removed: the T4.4 regime capture wrote 0 rows at every posture — the post-warmup setter was a swallowed OnceLock no-op (found STATICALLY; the filed read_one race never ran)

- Root cause: the dump runner warms up DISARMED → the first `regime_capture_armed()` seeds `CAPTURE_PATH` (`OnceLock<Option<PathBuf>>`) with None from the unset env → `set_regime_capture_path` fails forever (already initialized, `let _ =` swallowed) → armed() false all run → 0 rows → the `total>0` panic ~3 h later. A OnceLock<Option<T>> holding None is INITIALIZED — the type carried the trap; 4 dead runs (~12 GPU-h) and a wrong first suspect (the silent read_one Err — that block never executed) explained by one mechanism. Fix riir-infer `77ed153` (Mutex+one-shot env seed, setter overrides anytime and fences the seed; REGIME_CAPTURE_COMPILED posture const; append self-heals the parent dir + counts failures; boundary read sync-fenced retry + counts failures; regression tests cfg-exclusive) + runner half riir-ai `e4644a2d6` (ms-scale posture assert; an ARMED SMOKE fails a dead seam in seconds; per-item rows>0 asserts; failures==0 gate; the test-NAME FILTER load-bearing — unfiltered --ignored also ran the bench_1152 sweep compiled into the same binary, the ~2.5 h per dead run). Validated END-TO-END: the first post-fix dump went green (56 rows = 14 items × 4 chunk rows, all 7 families, failures 0, 953 s), then the T4.4 freeze + calibrate landed in the same pass — fixtures/regime_centroids.json + BLAKE3 fb319c6f… (dim 10240), CALIBRATED gain 10.0/margin 0.20 (the tested pair at which zero retrieval-family rows take the loose floor; induction routes flow-side and bands AMBIGUOUS-conservative; continuation misroutes sparse-side — the safe direction). Full record: bench 933. Cell E may now arm (posture in bench 933). Lesson: /tmp evidence on amo is one power-cycle from gone (/tmp/i1156_t44 wiped mid-morning; the static root cause needed none of it). Session zed-amo-i638t1c, epoch 1793606100.

## 2026-11-01 — Issue 1162 CLOSED + removed: the lone `ternary_gemv` combo failed to compile (4 errors, pre-existing) — an ungated method whose targets + only caller are `ternary_gemm_batched`-gated

- Repair `2c694cb`: `prefill_project_pair` gated `ternary_gemm_batched` (the cfg its targets `PrefillSite`/`prefill_project` and its only caller `prefill_tokens_chunk_once` already carry; the module gate in lib.rs stays untouched — gating the whole module would compile OUT the gemv path). Validated clippy-clean at lone-gemv (was 4 errors), gemm_batched-only, the capture triple, default, dq_phase_bench. Residue refinement in the same commit: the 1161 `regime_capture_append` gate (`all(cubecl_runtime, ternary_gemv)`) matched the module but not the CALLER (the Issue-1156-T4.4 capture block inside `prefill_tokens_chunk_once`, cfg `all(cubecl_runtime, ternary_gemm_batched, ternary_attention_batched_prefill)`) — re-gated to the caller's exact cfg, verified by error injection (0 hits = block cfg'd out) + a minimal-replica crate. Pre-existing recorded: the ane_zc_channel_split-ALONE posture carries 2 dead-code warnings (metal_tensor_attn + try_prefill_attn_metal_tensor, consumers behind metal_tensor_gemm/macos). Session zed-m3-idle1031b, epoch 1793488531.

## 2026-11-01 — Issue 1161 CLOSED + removed: the 7 dead-code warnings at the cubecl_runtime-only posture (reflex's laya-riir-cubecl composition) — root cause: the callers' WHOLE MODULE is cfg'd out

- `ternary_deltanet_gpu_forward` is gated `all(cubecl_runtime, ternary_gemv)` (lib.rs:696), so at cubecl_runtime-only every call site (REACH marks, dense_pair_wg_size, regime_capture_append) compiles to nothing while the callee modules compile — rustc was right, "the callers are compiled" was the false premise. Repair `38df722` (the crate precedent, deltanet_rotation_cudarc.rs:1893, split by test consumption): the four not-test-consumed items extend their cfg to `all(..., ternary_gemv)`; the two test-consumed dense-pair fns take `cfg_attr(not(all(test, cubecl_runtime)), allow(dead_code))`. Consumer-edge fix rejected — the laya lane consumes the op layer (plan 611 T7), not the whole-forward module. Validated: clippy 0 warnings at cubecl_runtime-only (303/0 lib tests there), production clippy clean, reflex `--all-targets --all-features` clean (the originating posture). Adjacent pre-existing finding filed as issue 1162 (the ternary_gemv-ALONE combo fails to compile at HEAD — a ternary_gemm_batched cfg mismatch, no live consumer). Session zed-m3-idle-i1161, epoch 1793485379.

## 2026-10-31 — riir-ai Plan 607 follow-up: the promoted `RIIR_GDN_REC_MR` re-pinned through the formal league harness + the share-model corrected — 0.2714×/0.2792×/0.2732× (pp2048/16k/32k), FNV pins EXACT ×9, +7.7% window question closed (drift), rec share at the league chunk 5.09%/3.35% vs the serving-chunk census's 2.16%

Formal bout on bmo at tuple riir-ai `ded79a301` + this repo `6116c47` + katgpt-rs `2bbaf2640`, fork `73bfe6cc48`, verified-idle, canaries in-band: pp2048 289.69/1067.25, pp16k 254.04/909.87, pp32k 210.42/770.33 — the armed class (D4: 290.06/254.81/209.94) reproduces in a fresh window; the pre-promotion seeds (266.46/239.95) were window-depressed. The bench_1146 census (4 cells, bmo, full logs): at the LEAGUE chunk (single 2048-wide frame) gdn_rec = 288.85 ms = 5.09% (MR=0) / 187.12 ms = 3.35% (MR=4); measured rec-kernel saving −35.2% (isolated predicted −32.2%); e2e census +1.73% ≈ 5.09%×35.2% = 1.79%. The lost D2 strip cross-check recovered: gdn_rec −2.16 ms under MR at chunk 128 (predicted ~2 ms), census e2e +0.61% (D2 read +0.60%). Record: riir-ai league doc 10-31 pm rows + HISTORY.

## 2026-10-31 — Issue 1160 CLOSED + removed: soft-token feedback decode (LIFT/MoI class) — G0 COLLAPSE on Bonsai-2 PQ2_0

- Pre-registered G0 fired: pooled median decode top-1 mass 0.976 ≥ 0.9 over 512 greedy Metal decode steps (arith/G1b headline class 0.999; the pilot's top-128 τ=1.5 soft token puts 0.985 of its weight on the argmax) ⇒ `E^T s` ≈ the argmax embedding where the gain was claimed; closed without the 18-config sweep. Prose/zh are not spiky (median 0.48–0.68) — the reopen condition, with dense-lane and trained-channel (riir-train Plan 344) triggers, is in [Bench 932](.benchmarks/932_i1160_g0_soft_token_collapse.md). Probe kept: `tests/issue1160_g0_collapse_probe.rs` (`#[ignore]`, required-features row at birth). Source: riir-train Research 487.

## 2026-10-31 — riir-ai Plan 607 E1 LANDED:  promoted DEFAULT-ON at planes=8 (Metal/CubeCL prefill; the CUDA lane default-on since Bench 896) — riir-ai Plan 607 D4's armed-posture paired-window A/B on bmo at the m5ultra league shapes: +1.24%/+1.37%/+1.20% (pp2048/16k/32k, n=6/arm/shape), 18/18 armed passes above incumbent, manifest FNV birth pins EXACT under the armed kernel; delegated-owner verdict AGREE; env now DEMOTE-ONLY (0/off/false = incumbent byte-exact; garbage fails open to the DEFAULT arm); caveats recorded (e2e exceeds the +0.38% pre-registered ceiling — the A2 census share under-counts; M3-box e2e unmeasured per the box-state binding)

## 2026-10-30 — Issue 1157 CLOSED + removed: regime_gate inline sigmoid delegated to the substrate — bit-preserving, pinned

- `crates/riir-infer-gpu/src/regime_gate.rs` `fn sigmoid` is now a thin delegation to `katgpt_core::exact_sigmoid` (the bit-stable two-branch reference form, body-identical — NOT the Cephes `sigmoid`, whose ±40 clamp / ~1.2e-7 drift a gate decision must not inherit); delegation pinned bit-for-bit against the frozen legacy body over a fixed grid (`sigmoid_delegation_matches_frozen_legacy_body`, this commit). Copy-gate class finding from the substrate-first audit; 20/20 regime_gate tests green, clippy clean.

## 2026-10-30 — AGENTS.md compaction: the Role section's lane-landing narratives moved here (switches + postures stay in AGENTS.md's lane table)

- GDN L8 eager prefill (plan 621): PROMOTED DEFAULT-ON 2026-10-24 — riir-ai Bench 993 +3.22% / Bench 997 +2.76% pp2048, pp4096 +2.62% post-promotion, G1 bytewise; the ±8.4% band clause superseded by measurement.
- GEMV WG occupancy (riir-ai Issue 1114): default-on Metal 2026-10-24, CUDA 2026-10-25 (Bench 887, Issue 1115); +4.88/+5.52% e2e on the m5ultra. Fused fold-tail/SwiGLU-rotate kernels (Issue 1113): −160 launches/token, +1.40% e2e below bar → opt-in (Bench 995).
- GDN delta-roll (Issue 049, 2026-10-20): rank-1 unwind recovery ~64× vs snapshots, G1 drift 6.3e-6 (Bench 885). Stability-depth readout (Issue 1106, 2026-10-22). TOKENGATE T3.4 four-arm GOAT FAIL (bench 889); break-even ceiling ~1.5–1.8× at K≤0.25 (bench 891).
- Plan 623 multi-host: Phase 1 2026-10-26 (G1 63/63 qwen splits); verbs FFI `226cace`+`16fb64a` — RTT p50 7.7 µs, 9.368 GiB/s (Bench 927); Bench 928 `570ee79`+`f9c4803`; T3.4 async-finish `9f3c0f7` demoted promote-only (`6a1c524`, −48.5% on wire); T3.5 decode 52.3–52.6 tok/s vs llama.cpp-RPC 48.6 (Bench 929).
- Issue 1124 GLU-fused prefill FFN `1320dda`; T1.5 folded-lane wiring `c60b34d`; promoted `57fa7a7` (verdict `9ca951de`, +6.31/+6.72% pp2048); T3 mined via refine `d871a0fc`.
- Decode-hold rungs: core-fused `4132d94` (G2 negative), beta-expand `8aabe55` (+0.6–1.0%), quantize census falsifier `a2ebb37`. Dense-pair (issue 1129) `c400dbc` +9.88/+9.07% decode, CUDA re-cell +1.54% (Bench 1014, test head `0e509e4`). Prework-fused `e2edf61`+`2581fdd` (riir-ai `d25c7c460`), CUDA N/A proof `69e7d8e`. Qwen3.8 dense mm2d 50.22× kernel-level (Issue 1150).

## 2026-10-29 — Issue 654 promotion e2e EXECUTED + the gate REFUSED: the CUDA-placed head measured, the roll-chain loss priced structural (Bench 931)

The owner-GO'd promotion rung ("654 lgtm", claimed zed-m3-i654-promo) executed on the 4090 primary platform per the pinned posture-B gate — promotion REFUSED, the arm stays opt-in default-off: every replicated cell loses to plain decode (k=2: 0.846×/0.946× paired ratios; k=4: 0.745× — riir-train Bench 682's …)
Commits: `06c184a`, `4c77521`, `7e20540`, `b769ccc`.

## 2026-10-29 — Bench 929 follow-up: the plan-623 launch-context discriminator RE-TESTED — class refuted, harness spawn-retry landed

The Bench-929 PROVENANCE discriminator ("detached-matrix spawns EHOSTUNREACH 6/6 while foreground connects, root cause not found") was re-tested on demand (zed-m3-idle1029b): the launch-context class is REFUTED as a deterministic posture property — the exact matrix launch shape …

## 2026-10-29 — Issue 1155 CLOSED + removed: the exact slot-addressable GDN state PoC — hypothesis CONFIRMED (claim `89b4880`, tests `36c71f6`, addendum `5552964`, hash re-pin `074fc8f`)

The Qwenputer identity PROVED on our kernels (the divagr18/qwenputer distill, Research 018): one-hot keys address one state column through the REAL `gated_deltanet_step_inplace` bit-exactly when payloads are bounded integers in the ±2^24 grid + gate=1 + β=1 (G1a PASS incl.)

## 2026-10-29 — Issue 1150 follow-up LANDED: the dense-f16 matmul2d prefill arm — the Metal scalar-GEMM wall broken on the dense lane (A/B 50.22× kernel-level, claim `2c6dfaa`+`049d3ce`, this commit)

The Issue-1150 session record's named follow-up (all in-crate CubeCL GEMMs at ~110 GFLOP/s on Metal — the wall the Bonsai league's 0.249× prefill row shares): `gemm_dense_f16_metal_tensor.metal` + `gemm_dense_f16_metal_zero_copy.rs` — the dense twin of the ternary zero-copy lane (riir-ai Issue 663 T5 …)

## 2026-10-29 — Plan 623 Phase 3 COMPLETE: T3.5 scoreboard LANDED + the T3.4 arm wire-refuted and demoted (Bench 929, issue 1153 `ed10f14`+record, issue 1154 `6a1c524`)

The Phase-3 scoreboard on the amo⇄bmo TB wire (zed-m3-i971-housekeeping, foreground-chunked cells, bmo carrying the disclosed mammoth-capture co-tenant): decode row — ours 2-box s1 52.28–52.59 tok/s BEATS llama.cpp-RPC 2-box 48.6 (+7.6%; the Issue-1141..1146 production rungs flipped T2.7's −20% …)
Commits: `6a1c524`.

## 2026-10-29 — Plan 623 Phase 3 T3.3 CLOSED + T3.2's rkey half CLOSED: the MTU ladder is neutral, RDMA WRITE refused-by-silence (Bench 928, T3.2/T3.3 `570ee79`+`5304716`, bench `f9c4803`)

`verbs_write` + `verbs_metal` (feature `rdma_metal`, metal 0.31 — one workspace version, the BOUNDARY.md metal row) + the verbs-module surface they needed: `create_qp_typed`/`create_rc_qp`, `post_rdma_write`/`post_rdma_read` (the WR opcode + `IbvRdmaPart` path), `wr/wc_opcode::RDMA_READ` pins, and …

## 2026-10-29 — Plan 623 Phase 3 T3.4 LANDED: the decode-feed async-finish arm (issue 1151, T3.4 `9f3c0f7`)

> ⚠ **1151 is DOUBLY HELD** — this is the SECOND holder (`1151_plan623_t34_gpu_comm_sync`, filed `148f22d` reading the counter's value 1151 instead of value+1 — the counter already stood at 1151 after `559be00` — removed `726cec7`). The first holder is *Issue 1151 — the production posture single home* below. Both files are deleted, so neither can be renumbered; cite this one as **Plan 623 T3.4 (the issue-1151 GPU↔comm sync unit)**, never bare "Issue 1151".

`riir-infer-gpu` re-exports `{block_on, DynFut}` + `read_hidden_state_fut()` — the staged-copy read future ('static via a client clone, resolving to the f32 vector), created at ENQUEUE time so the copy encodes FIFO-after this stage's chain and its resolution IS the chain-done signal value …

## 2026-10-29 — Plan 623 Phase 3 T3.1 LANDED: the verbs FFI lane — first RDMA traffic on the amo⇄bmo TB bridge (Bench 927, T3.1 `226cace`+`16fb64a`, bench `322e843`)

`crates/riir-infer-dist/src/verbs/{ffi,mod}.rs` behind the opt-in `rdma` feature (macOS-only by contract — the `ibv_context_ops` vtable layout is Apple's librdma.dylib header's; rdma-core's evolved vtable is not claimed): the SDK header mirrored repr(C) with layout pins held to the C compiler's …

## 2026-10-28 — Issue 1151 CLOSED + removed: the promoted serving posture has a single home — the `production` aggregate + `PRODUCTION_RUNGS` + the dist posture floor (fix `80ff665`, close `b5acf92`)

> ⚠ **1151 is DOUBLY HELD in this repo** — this is the FIRST holder (`1151_production_posture_aggregate_single_home`, filed `559be00`, removed `f6b3b6a`), and it KEEPS the number (first allocation; katgpt-rs `citation_weight.py` awards it 11–5). The second is *Plan 623 T3.4 — the GPU↔comm sync unit* above (`1151_plan623_t34_gpu_comm_sync`, filed `148f22d`, removed `726cec7`). Cite this one as **Issue 1151 (production posture)**.

The miss class fired THREE times in nine days — each a consumer's hand-typed serving-feature list omitting a rung the league/engine posture carried: 1141 (dist ran the bare decode kernels, 37.24 → 21.4x ms/token after repair, 1.74×), 1146 (chunked prefill + batched attention omitted from dist's …)
Commits: `b5acf92`.

## 2026-10-28 — Issue 1129 CLOSED + removed: the cuda_backend CubeCL decode A/B validates the dense-pair default (riir-ai Bench 1014, the 4090 lane)

The remaining scope (the CudaRuntime arm of the DEFAULT-ON dense-pair fusion — the unmeasured-default class; the scored 4090 decode lane is cudarc, whose K2 fuses the pair) measured + closed from the amo idle lane over SSH while amo held zero-cargo under the Plan-337 T5 posture.
Commits: `cbcdb5a`, `0e509e4`.

## 2026-10-28 — Plan 623 T2.5: the boot-probe planner is fully wire-automatic (the probe EXCHANGE + its residue)

The EXCHANGE (zed-amo-i623-exchange, `22eb298`): the head fetches the tail's boot-probe record over the wire and composes `plan_split` live — no operator hand-carries JSON between hosts.
Commits: `a260b62`.

## 2026-10-28 — the grid-reselection drafter arm LANDED (katgpt-rs 966/plan-632; the arc closed NEGATIVE on the lane read, the arm stays opt-in)

`lattice_walk_reselect` (feature `grid_reselect`, `c04e3fa`): the keyed-walk body with the pick replaced by the prefix-first scored law `argmax_b [ln softmax(s/T) + λ·ln P(b|a)]` over each slot's own selector grid — re-scored along the re-selected prefix (`chain_step_scores` conditions on …)

## 2026-10-27 — Issues 1145 + 1139 CLOSED + removed: the M5 Ultra decode-gap campaign closes at 1.04× vs llama.cpp (Bench 907: multi-cube norm+rotate PROMOTED DEFAULT-ON, +2.83%)

Issue 1145 (the campaign's last routed rung, filed from the 1139 T4 census): one launch of `dim/hblock` cubes (5 at Bonsai-2's 5120/1024) replaces the mlp4-norm + r4-staging PAIR at each decode norm site — every cube redundantly runs the incumbent's Phase-1 mlp4 hoist + tree over the WHOLE vector …
Commits: `d1b6156`.

## 2026-10-27 — Plan 623 Phase 4 (TP): T4.1/T4.2 cell built + correct, gate REFUTED by measurement (Bench 906); Issue 1109 CLOSED + removed

First real TP cell over jaccl (`scripts/plan623_mlx_tp2_bench.py`): column/row sharding of the Bonsai-2 pack runtime, both ranks run all 64 layers, `all_sum` after row mats + vocab-split lm_head `all_gather`, greedy lockstep with NO token exchange.

## 2026-10-27 — Issue 1127 CLOSED + removed: the decode-hold escalation ladder — T1–T4 all executed with records; no owner gate remains (verdict, this session)

All four tasks closed with measured records: T1 attention-chain rung opt-in +0.77% (Bench 900); T2 census-negative (Bench 902); T3 measured-negative at the bit-identical boundary (Bench 903); T4 league re-cell EXECUTED (Bench 1008, riir-ai `e3ffd3004`) and adjudicated (`efd9332`, reviewer …)
Commits: `9ca951de`, `57fa7a7`, `c7ebe93`, `df9c464`.

## 2026-10-27 — Issue 1133: GDN conv+beta/decay+expand merge rung MEASURED-NULL (Bench 905: G1 exact everywhere, G2 2/4 pairs, pooled ≈0% ± 0.5%) — shipped opt-in, NOT promoted (`a4da7be` + the rung commit)

The last unmerged launch-bound boundary in the armed GDN decode chain: `conv_beta_decay_expand_fused_f32` folds conv1d + the Bench-899 beta/decay+expand pair into ONE launch (48 layers × 1 node ≈ 62 µs/token dispatch predicted ≈ +0.57%).
Commits: `fe06c7e`.

## 2026-10-26 — Issue 1122: history-fed n-gram drafter probe CLOSED-NEGATIVE (Bench 895: re-roll α 0.063 vs breakeven 0.58 — T1 wiring refused) — file removed

The NgramRL class (draft decode call N from the model's own same-prompt outputs) probed BEFORE wiring per the Issue-665 honest-assessment pattern.

## 2026-10-26 — Issue 1121: StabilityTracker first-ordinal pin CLOSED — landed `c34b954` — file removed

Hardening pin (Issue 1106's `stability.rs`): the first observed stage is now `debug_assert_eq!(ordinal, 0)` — a partial/offset capture feed (e.g. starting at `AfterLayer(30)`) passed both shipped guards (width + strictly-increasing) and reported INDEX-space depth, silently wrong by the offset.
Commits: `22afc87`.

## 2026-10-25 — Issue 1118 Finding 1: BPE merge ranks re-keyed to packed token-id pairs — LANDED (Bench 894: G1 bit-identical, G2 2.03× encode, G4 zero-alloc probes) — `ad7b5a0`+`54a6616`+`ddc7105`+`c5032f6`

The merge loop allocated two heap Strings per adjacent pair per merge iteration (its own comment admitted it, deferring "a future optimization").
Commits: `7b2fed9`, `addd0b4`, `0e9cc4ce`.

## 2026-10-24 — Issue 1112: arm8 whole-prefill CUDA crash on unfolded (pre-rotation) synth shapes CLOSED — root cause + fix + regression GREEN (`0522312` + `441b78f` + `63ab2d2`) — file removed

The crasher was never a kernel: the synth Bonsai-2-style file is UNFOLDED with DENSE a/b, `upload_layer_weights` stores the dense f32 mirrors beside EMPTY dummy `TernaryGroupWeights::new(0, 0)` handles, and the whole-prefill PLAIN lane (`rotation.is_none()`) mirrored the dummies into 0-word CUDA …
Commits: `1b5b623`, `d53c8f3`, `63ab2d2`, `e90d775`, `441b78f`.

## 2026-10-13 — Issue 042: cubecl #1806 idle-server park fix vendored CLOSED — landed `fa86871` (mirrors: riir-ai `4b86d2003`, riir-refine `f0718269`) — file removed

The idle-queue's "cubecl pin-bump re-gate" lane (queued cycles 38–41, blocked on the 038 sibling's WIP — cleared when their lane landed in full at `4db9e15`).
Commits: `6bba5fdf0`, `58e7689`, `0c5ed0fe7`.

## 2026-10-10 — Issue 037: wgpu full-suite device-loss cascade CLOSED — root cause + VRAM admission gate landed — `3564a72` (+ upstream draft `dad3e7c`) — file removed

The "which test kills the device" question was a category error: no single killer test exists — the full `cubecl_runtime` suite needs ~16 GiB contiguous commit and the loss fired at the box's GLOBAL VRAM commit position (trainer 7.6 GiB resident; VRAM trajectory 9.0 → 23.8 GiB at the loss second) …
Commits: `02e7a65`.

## 2026-10-09 — the elementwise params[0]=n root fix + the vendored metal adapter E0308 — `02e7a65`

Retired katgpt-rs Issue 928's two len_derived stability rows: sigmoid_f32 / silu_f32 bound by an EXPLICIT params[0]=n (the copy_f32_grid2d idiom), never input.len(); the new oversized-binding sentinel test proves [n, n+pad) untouched.]
Commits: `3564a72`.

## 2026-10-09 — Issue 036: LittleBit init-only sub-1-bit PTQ CLOSED — NO-GO for the format+kernel lane (Bench 034; T4 closed-by-verdict) — file removed

The paper's unmeasured point (Dual-SVID init-only, no QAT) measured end-to-end on gemma-2-2b-it (the T0 host decision `8d3fd7e`: local PPL bin over existing substrate — `forward_gemma2_f16` + `nll` + SentencePiece; wikitext-2-raw test sha256 `aca2f467…`, 49,152 tokens; anchor 18.67 validates the …)
Commits: `b235f4e`.

## 2026-10-07 — Issue 035: FA-constrained dLLM decoding POC CLOSED — NOT PROMOTED (G2 FAIL, sequential lane) — `7d90dfd` (+ file removed this commit)

P3 executed on the 4090 box (quiet: desktop-only GPU consumers, 26.6/33.3 GB RAM free; real-vocab arm = gemma-2-2b 256k, `../riir-train/data/gemma-2-2b-it-f16.gguf`).
Commits: `f4ffa93`, `7bda765a`.

## 2026-10-06 — Issue 028 T4 S2: the q4 weight arm — the container loads and runs the dual-format pair — (this commit)

`ProjWeights { Ternary(TernaryGroupWeights), Q4K(Vec<BlockQ4K>, rows, cols) }` at the 10 per-layer projection fields; `bitlinear` dispatches the Q4_K arm to fused `gemv_q4_k_row` (rayon across 16-row chunks, bit-stable), GPU hooks refuse q4 LOUD, loader `load_proj` storage-type dispatch.

## 2026-10-06 — Issue 028 T4 S1: the Q4_K prefill pack exists — plus the science premise settled — `94a4e7c`

`Ternary-Bonsai-2-27B-Q4_K.pf.gguf`, 14.43 GB, blake3 `0a7a5605fadc37b482c0fddf6c28334244eea818b1716461bddb412f1a28309c`; the science premise: the bonsai is TRAINED TERNARY, so the PQ2_0 wire is exact up to f16 group scales — requant carries only ε(Q4_K) (`UnsupportedFourthState` is the structural …)

## 2026-10-06 — the GPU crate is clippy `-D` clean at every posture — `e993eff`

Pre-existing warnings gone; the rotation quantize pair (`quantize_rotate_q8`/`_permute_rotate_q8` + `*_warp` fields) is test-consumed only — allow gate `not(all(test, feature = "prefill_q8_act"))` (the first `not(any(test, …))` was wrong in exactly the firing posture); `DQ_FQ_CUDA_SRC` is dead only …

## 2026-10-06 — Issue 035 P2.6 LANDED: real-vocabulary compile + the row-fold — the honest G2 regime — `ddcaeb0`

Trie-stepped subset construction in `fa_schema.rs` (256k vocab compiles the fn-call grammar in 60s; toy 0.13s, object 1.8s) + one vocab-wide exp row per position in `fa_posterior.rs` (fn-call draw 11.79s → 1.98s at L=256).

## 2026-10-06 — Issue 035 P2.5 LANDED: DFA minimization — the G2 ratio drops ~4.4× — `9e94684`+`672e2b0`

`src/fa_minimize.rs` (UNGATED, 7 tests): trim + block-granularity partition refinement applied ALWAYS inside `fa_schema::compile` (language-preserving by construction); empty-language schemas now fail at COMPILE time (`SchemaError::InvalidSchema`).

## 2026-10-06 — Issue 035 P2 LANDED: the schema front end + the honest overhead table — `1254264`+`5b5c462`

`src/fa_schema.rs` (UNGATED, 12 tests): JSON-schema subset (six types · properties/required · items · enum · anyOf) → Thompson NFA → on-demand subset graph → token `Automaton`; three grammar-soundness laws pinned (fresh ws wrappers, per-(mask,member) member chains, acceptance = walk ∧ final).

## 2026-10-06 — Issue 022 T5.4 CLOSED: the equal-FLOP skip-class gate — the lane wins, honestly — `8bd5c2d`+this commit

Record `.benchmarks/022_t54_skip_arms.md` (instrument `src/bin/twt_skip_amputation.rs` at `a7b080c`): twt 0.8919 > hydra 0.8642 > random floor 0.8437 >> ShortGPT 0.0320 at the equal 3-layer cut — selection disagreement total, ShortGPT's BI greedy 81 points below random (a dense-model removal …)

## 2026-10-06 — Issue 035 P1 LANDED: the `fa_constraint` decode wiring — exact joint x→0 per denoising step — `304876f`+`f4ffa93`

P0's sampler wired into the `gemma2_d2f` denoising loop behind opt-in `fa_constraint` (implies `gemma2_d2f` ⇒ `dllm`): ONE exact joint draw per step; commit axes `RawTop1` / `ConstrainedMarginal` (marginal log rides `SamplerFeatures::marginal_log`, the 7-param `ConstrainedSampler` — the 6-slot …)

## 2026-10-01 — riir-ai Issue 1004 R2 LANDED: the fused GDN prework (conv1d + SiLU + q/k L2-norm + head expansion, ONE dispatch) — `929d31e`

Feature `deltanet_prework_fused` (toggle `set_prefill_prework_fused`): 2 dispatches/layer vs the shipping chain's ~69; stage-isolated 7.29× @2048 / 9.13× @4096; G1 bit-identical (logits FNV `0ec2396fd4627f29` @2048 / `f35280cb95f5d306` @4096).

## 2026-10-01 — Issue 026 CLOSED (hygiene): the DQ phase-matrix instrument landed and ran — every axis INADMISSIBLE at the frozen corpora

Plan 614's instrument (`dq_fakequant` + the `dq_phase_matrix` runner; record `.benchmarks/023_dq_phase_matrix.md`) ran to EXIT0: base arith 0.9583, NIAH pooled 0.9896 — both OUTSIDE the pre-registered 0.25–0.95 window, so T3's damage-ratio assertions receive NO GATE; T5's promotion rule (any future …)
Commits: `5d0592f`, `4f68e5d`.

## 2026-10-01 — Issue 032 DONE: Metal CubeCL prefill computes Hadamard-folded (Bonsai-2) models (`5d0592f`)

Batched twins of all 8 decode rotation sites in the CubeCL prefill body + `GemvBatchedCubeCL::launch_token_grid`; G1 vs folded decode-eager top-1 equal, worst rel err ≤ 1.5e-5; pre-rotation Q2_0 pin `99a0733c45a0e663` @2048 exact; first Bonsai-2 M3 prefill 6.44× TTFT (100.9 tok/s @2048); test …
Commits: `5f075ff`, `7e99f34`.

## 2026-09-30 — reflex Issue 054 Part 2 REFUTED for this lane: the prefix-state handoff lead — laya is ModernBERT (bidirectional), not GDN; the coupling gate landed

The premise named the wrong architecture: laya checkpoints are ModernBERT-large / mmBERT-base (no causal mask; the state also sits after the per-question head span — different RoPE offset per question).

## 2026-09-30 — Issue 013 T2 EXECUTED: the K=V+ λ ladder on gemma-2 — G-A PASS vs a catastrophic baseline; K=V+ not viable standalone; G-D overfit; NIAH build crash (029)

Record `.benchmarks/012_t2_kv_ladder_gate.md` (+ `012_t2_run.log`, table `012_kv_table_residual.bin` sha `3a0f5333d6bafd63…`, reused by T3): f16 6.0907 (== T1, cross-run determinism PASS) vs k-0.00 catastrophic (tax +30,948,056%); k-1.00 λ*=1 G-A PASS (3344× recovery) but still ×92.6 off f16 → K=V+ …

## 2026-09-30 — Issue 013 T3 EXECUTED: P3 V-cache reconstruction — G1+G3 PASS at every λ; the 50.0% bytes/token law holds; G2 +7.9–8.8% recorded (the promotable P-half)

Chained run rc 0 (waited 399 min for T2, BLAKE3-verified its table artifact). Record `.benchmarks/013_t3_p3_reconstruct_gate.md` (+ `013_t3_reconstruct_report.md` written incrementally, `013_t3_run.log`): G3 max |Δlogit| 1.254e-4 (the rotation-rounding class); G1 PASS 0 flips in 2046 scored …

## 2026-10-05 — Issue 013 T4 scaling half EXECUTED: the naive read-path cost explodes with context (1.08→1.79→3.82×); the pre-registered rule fired HOLD — katgpt-core `fitted_v_reconstruct` stays opt-in

Bench 016 (`.benchmarks/016_t4_g2_scaling_hold.md`, cells `016a`/`016b`, log `016_t4_g2_run.log`; cache-sizing fix `2300457`): 1.079× @129 → 1.789× @1025 → 3.821× @4097 window edge ≫ the ≤1.20 promotion bar → katgpt-rs promotion HELD (`682936a1c` there; HISTORY § Issue 883 amended); quality …

## 2026-10-05 — Issue 013 COMPLETE (T4 kernel half + the promotion EXECUTED): both levers landed as one read path, the window-edge bar met — `fitted_v_reconstruct` PROMOTED to katgpt-core default

Bench 017 (`.benchmarks/017_t4_deferred_lever.md`, cells `017a`/`017b`, log `017_t4_deferred_run.log`): the fused deferred-restore attention kernel + the bitwise per-position (sin,cos) table landed as ONE read path (`fe75497`, extended by katgpt-rs `618d7ec71`'s `FittedTokenTable::row_index_of` + …)
Commits: `34145bfd5`.

## 2026-10-05 — Issue 033 COMPLETE: the KV-axis extension arm landed and ran — the KV-store axis is NULL (a4-class KV storage accuracy-free), the activation axis dominates; instrument extended with the KV arm + hardened corpora

Bench 033 (`.benchmarks/033_kv_axis_extension/`, exe at `ad683a6`): the runner gained the KV-STORE axis (`DQ_KV_GRIDS` — every stored K/V row rounded at the CUDA `kv_fill` prefill + CubeCL `forward_attention_layer_gpu` decode sites) + `DQ_ARITH_HARD=1` / `DQ_NI_NEEDLES` corpus hardening.

## 2026-09-27 — Issue 014 RESOLVED: activation-aware ternary scale fit — SPLIT BY LANE (born-ternary clean negative; dense-parent mechanism transfer)

From katgpt-rs Issue 886 (substrate `0fb2254d9`, katgpt-rs Bench 896). Resolution `3c2b569`; lanes `03681e9` (`act_taps`), `fc31777` (`act_retention_walk` bin), `28b1566` (dense-parent PTQ); benches `005_act_diagonal_bonsai_first_slice.md` (T1, feature `act_diagonal_calibration` over the …)

## 2026-09-27 — Issue 020 RESOLVED: the len_derived CAPACITY finding was the classifier's, not ours — and both fixed temp paths are gone

Finding 1: CAPACITY at `encoder_lane_cubecl.rs` was a classifier false positive, closed in the katgpt-rs instrument (`43c14d754`, the shared `length_from_handle_size_method` rule) — no riir-infer code changed.
Commits: `20ea589`, `337da41`.

## 2026-09-26 — Issue 019 RESOLVED: the c0dfa06 Metal barrier REVERTED — A alone was the wobble, and serial dispatches never needed barriers

Necessity conceded on the timing re-check (the "clean ×3 with barriers" triplet was confounded by `2a34bd3` in the working tree); premise refuted on the code: the fork's `begin_compute_pass` never sets `MTLDispatchType` — Metal defaults to SERIAL, memory visibility implied (the durable map if a …)
Commits: `4205c12`.

## 2026-09-26 — Plan 611 DONE: the T7 op-layer unification verdict (Bench 006) — riir-reflex Issue 008 closed

One portable CubeCL `Backend` over this repo's op layer (S1–S4), A/B'd against the hand lanes (S5; rules pre-registered at `fac0dfd`): CubeCL 5.1–8.4× slower than Metal, 0/12 wins (Bench 006, results `ac85c8e`) — the hand Metal lane stays the macOS default; the CubeCL arm KEPT opt-in …
Commits: `2a34bd3`, `4205c12`.

## 2026-09-26 — Issue 018 CLOSED: the Cubecl-posture drift wobble was a softmax write-after-read race (fix `2a34bd3`)

Both CubeCL softmax kernels (`softmax_f32`, `softmax_rows_inplace_f32`) read the max from smem slot 0 then wrote the sum over it with NO barrier; one `sync_cube()` after the read (+ chunked-launch fix: rows past `MAX_WG_X` re-normalized row 0 — the chunk's first row now rides `params[1]`).
Commits: `ccb5bd0`, `270740c`.

## 2026-09-26 — the riir-ai carve docs adopted into this repo (Issues 998 + 1003, Plan 610, Proposal 041, Benches 870 + 871, doc 002)

Moved home with numbers verbatim (every "riir-ai Issue 998 / Plan 610 / Proposal 041" citation resolves to the same document): `.issues/998_riir_infer_repo_promotion.md` + `.issues/1003_riir_infer_carve_remains_4090.md` (OPEN), `.plans/610_riir_infer_gpu_carve_slice1.md` (IN PROGRESS, S8) …

## 2026-09-25 — Issue 001 CLOSED: EXL3 (trellis-coded) weight format — reader complete, fused-GEMV lane closed at this tier

Full record (§1–§17 numbers unchanged) in `.docs/001_exl3_trellis_format_support.md`. Banked (opt-in `exl3`): CPU ref + LUT+rayon fast arm (`725a8a5`, 10.6–11.4×), pack loader (`bd93a7b`), CubeCL GPU decode v1/v2 (Bench 002, 71–87× wall on the 4090), sound bench harness (`9989e9f`), whole-pack …
Commits: `425e0c5`, `55c154f`.

## 2026-09-25 — T10 rung 2 (Metal): the attn_rope hoist pre-pass — default-off, promotion probe pending

`attn_rope` derives Q/K rope ONCE per layer into packed `[2, seq, d]` device scratch (`5ef7442`); opt-in `LAYA_METAL_ROPE_HOIST=1`, the in-kernel rope arm stays default and bit-identical; promotion via quiet-box position-balanced A/B (instrument archived: riir-reflex …)

## 2026-09-25 — two silent MiniCPM5 (llama-arch GGUF) defects, found bringing it up as Issue 011's long-context fixture

RoPE pairing `0b26b9a`: llama.cpp stores llama Q/K rows interleaved; this crate's rotate-half RoPE never un-permuted — `rope::unpermute_interleaved_rows` in `load_llama_weights_gguf` (ppl 259.93 → 69.0547 == HF fp32; gemma2/qwen2 are NEOX, untouched).
Commits: `175f67f`.

## 2026-09-25 — Issue 010 CLOSED (REFUTED): Gemma-2 has no QK-norm; the 288-tensor fixture is upstream-faithful

The premise was false: upstream `Gemma2Attention` defines no `q_norm`/`k_norm` — it bounds scores with `attn_logit_softcapping` (tanh, cap 50), already applied here (`attention_head_softcap`); per-head QK-norm arrived in Gemma-3. 288 = 11 tensors/layer × 26 + 2 globals is standard …
Commits: `95f52fb`.

## 2026-09-25 — Issue 009: software-pipelined narrow staging CLOSED — NEGATIVE at the gate, and the probe's tiny-op floor exposed

`sgemm_narrow_pipe` built, measured, reverted: gate NO-GO (probe −1.5%; forward-level paired env-flip −0.8..−1.6%, reproducible 3/3, under the ≥3% bar; nvcc SINKS register loads — the `asm volatile("" ::: "memory")` fence is load-bearing).

## 2026-09-25 — Issue 008: the narrow reg4 rung CLOSED — NEGATIVE, the narrow zone is TLP-bound, not bandwidth-bound

`sgemm_narrow_reg4` (4×4 fragments, 32×16 warp tile → 128 threads): every true narrow-served row regressed +41..+177% (`sgemm_shape_timing`, the `LAYA_CUDA_REG4` axis; the wide/xwide rows re-measured the 007 rung at −2.6..−20% both runs — the consistency check).

## 2026-09-25 — Issue 007: the register-blocking sgemm rung CLOSED — 4×4 fragments, −9..−21 % kernel on every wide/xwide shape

`sgemm_wide_reg4` + `sgemm_xwide_reg4` (16 accumulators, 2 B smem/FMA vs 3, ceiling 50%; same grids so block-fit cliffs carry over; k-ascending per-output accumulation).

## 2026-09-25 — Issue 006: the float4 sgemm rung CLOSED — every instance's B loads collapsed, −7..−17 % on every suite

The packed zone's problem was LOAD-ISSUE THROUGHPUT (12-16.5 TF/s = 15-20% of peak; 6 smem loads per 8 FMAs with the four B loads contiguous).

## 2026-09-25 — Issue 005: CUDA graphs CLOSED — NEGATIVE, the lane is GPU-bound (submit fully hidden)

Instrument ships (`LAYA_CUDA_STATS=1`, zero cost unset): `gpu == wall ±0.3%` on every row of every checkpoint — the whole CPU submit path (launches ~15 µs/call, uploads, ~0.2 ms allocs) executes inside the GPU window; a graph replay removes work that is already free (~1-2% bounded, under the noise …)

## 2026-09-25 — Issue 004: the sgemm tile ladder CLOSED — block-fit floors, the narrow single-question win

Three instances (`sgemm_narrow` 32×64×64, `sgemm_wide`, `sgemm_xwide` 64×128×32) picked by BLOCK-FIT on the SM count — the M3 `m < 256` rule does NOT transfer to the 128-SM 4090 (measured cliff: narrow −15.7% at n=2048 exactly 128 blocks, +46% at n=2560/160 blocks).

## 2026-09-25 — Issue 003 CLOSED: CUDA flash attention — the packed-path zeros defect fixed + the fused rung

v1's trait-default attention sliced host memory at non-zero offsets → `(ptr,len)` chain MISS → stale-zeros upload (typed_decisions 0.7445→0.2690; the evidence was the published bench, reflex `.benchmarks/026_4090windows_cuda` vs `018_4090windows_run`).

## 2026-09-25 — CalibrationTables promoted substrate-side (883 P0 Kimi fixture rider)

`katgpt_core::fitted_anchor_table::LayeredVkCalibration` (+ `VkLayerTables`) is now the ONE builder for both 883 P0 fixtures — this repo's gemma-2 dashboard and katgpt-rs's new Kimi-K3 dashboard (katgpt-rs Bench 889: ρ(V)≈ρ(K)≈ρ(V−K) per MLA layer, the "coupled through one latent" signature; KDA …)

## 2026-09-24 — Issue 002 CLOSED: the CUDA backend for the laya lane (the 4090 bench row, 17–60× the CPU posture)

Landed `9b52cb1`/`99f156e`→`e99d767`: `laya-riir-cuda` — cudarc 0.19 (`driver`+`nvrtc`, `cuda-13030`+`fallback-dynamic-loading`, target-scoped `not(macos)`), CUDA C → PTX at construction (NVRTC, sm_89); the Metal architecture ported verbatim (weight cache, epoch-keyed chain slots, `download_into` …)

## 2026-09-23 — crates.io publication: keep `publish = false` until the vendor patches upstream (owner-gates menu v2 row 2)

HARD blocker: both `vendor/` forks are load-bearing via `[patch.crates-io]` (`cubecl-runtime` — the #1359 drop-queue fix; `wgpu-hal` — the VRAM accessors) and a `[patch.crates-io]` section does not survive publication.

## 2026-09-25 — Issue 020 T6 CLOSED NEGATIVE: the encoder's host side is 1–1.6% of forward wall (measured before building)

Instrument `crates/riir-infer-laya/tests/metal_host_gpu_split.rs` (`#[ignore]`, `required-features = ["laya-riir-metal"]`, measurement-only) splits `Encoder::forward_packed` into `enq` (the whole host side) vs `sync` (`download_into`): seq188 0.80/48.7 ms, seq512 1.45/141.6, packed2x256 1.37/135.2 …

## 2026-09-25 — the narrow sgemm's shape is the measured local optimum (T7 occupancy axis refuted, both arms)

Two challengers behind a temporary `LAYA_METAL_SGEMM_VAR` flag (from the consumer's `sgemm_shape_timing` probe, reflex `ebe667e`): bk32 lost 17–33%, bn32 15–45% — the barrier cost beats the 2–3× co-residency gain; at BK 64 a 2-TG fit breaks the bank-conflict padding or the 8×8 block structure.

## 2026-09-25 — the sgemm MMA-roofline probe: the narrow instance is staging-bandwidth-bound (reflex Issue 020 T7 follow-up)

`crates/riir-infer-laya/examples/sgemm_roofline.rs` (measurement-only, `[[example]]` required-features row per the T1.1e law): the shipped narrow kernel verbatim vs an MMA-only twin (staged once, no re-staging, position-balanced).

## 2026-09-25 — f16-B staging REFUTED at kernel level (the roofline probe's own follow-up arm)

The `sgemm_hb` arm (`device const half*` B, RNE `f32_to_f16` seed): flat ±2% on every cell — B (12.6 MB f32) FITS L2 (~32 MB), so the binding cost is the L1/threadgroup-issue path, which halving bytes does not relieve; the f16-B backend rung is dead before being built.

## 2026-09-27 — Issue 021 CLOSED: the cuda CLS-row corruption — a chain-cache prefix-match

## alias, not a kernel race (the fused-GLU temp's slot vs the hidden that reused its address)

Fixed at the root: `chain_buf`/`chain_slot_for` now EVICT same-pointer different-length entries on bind; plus `download_into`'s `memcpy_dtoh` is `cuMemcpyDtoHAsync` (stream-ordered but ASYNC) — a trailing sync now guards the host read.
Commits: `c64d0b1`, `9de953c`.

## 2026-09-27 — Issue 011 CLOSED: `row_logit_floor` model-bound G1 complete — needle@64K PASS at every arm

T3c (MiniCPM5-1B at 64K real dilution, log `/tmp/ri011run/t6_minicpm64k.log`): PASS at every arm b8/b6/b6s0/b4 — 3/3 seq-exact, 0.00% top-1 flips over 12 scored tokens, m_Y preserved within 0.0007 (same top head L15H7), base ppl 1.0362. Standing: T2 PASS 8/6-bit (gemma-2 4096), T3 PASS 64K, the T4 …

## 2026-09-28 — Issue 017 (gpu_transpose dead module) + Issue 023 (fence F2 self-alias FP): both closed

017: `gpu_transpose.rs` + `src/kernels/transpose.wgsl` deleted (zero callers then known), `transpose_cubecl.rs` (Issue 572) covers the reachable use, BOUNDARY.md repointed, the riir-ai re-export dropped (`6c6bf169b`).
Commits: `04bd961b`, `3bed93f`, `734ef14`, `8330e196b`.

## 2026-09-28 — Issue 025 (owner-gate pickup) closed: D7/D8 executed, D9/D10 recorded

D7: gpu_transpose deleted + BOUNDARY.md repointed (record above). D8: the audio-lane BOUNDARY widening landed — an AUDIO Owns row (loader/serving-scoped, published CoreML bundles on `laya-riir-ane`) + the `objc2-core-ml` allowlist condition named (no new dep).

## 2026-09-29 — Issue 022 Phase 3 complete (audition + zero-training surrogate); Issue 024 closed measured-N/A both mechanisms

022 P3 (`27c9f86`): `src/twt/audition.rs` (mean + LaCo RDSC merge operators, the per-channel α/β branch-correction fit, BLAKE3-pinned selection) + `examples/twt_laya_audition.rs`; the apply path proven BIT-IDENTICAL against the parent forward every run.

## 2026-09-29 — Issue 022 Phase 4 LANDED (re-ternarization arms + κ budget + collapsed-GGUF writer)

`baeb686` behind `twt_collapse`: `src/twt/ternarize.rs` (three deterministic arms; PRE-REGISTERED budget κ = 2.0, τ_code = 1; arm B = integer CODE vote, arm C divides by the f16-ROUNDED scale and is BIT-EXACT on ternary input) + `src/twt/collapse_writer.rs` (GGUF v3 streamed from the parent mmap …)

## 2026-09-28 — Issue 015 audio PoC DEFERRED (owner: until M5 Ultra) + 023's missed hunk landed

Issue 015 stays OPEN, T1–T6 deferred (`- [-]`), the turnkey recon recorded in-issue (exact bundle `silero-vad-unified-256ms-v6.2.1.mlmodelc`, the wire contract from FluidAudio's `VadManager` @ `20d4f0bd`; the audio lane bypasses the laya digest/manifest coupling and REPORTS the plan verdict).
Commits: `e8e17b8`, `61ebf9c`.

## 2026-09-29 — Issue 022 T5.0 LANDED: the passthrough-collapsed checkpoint + the first real GOAT numbers (coarse grid FAIL; fine end measured)

Loader prerequisites fail-closed (`318b9fa`/`ceb96c3`): the writer REFUSES a qwen35 collapse without `twt.layer_types` (U8 `DeltaNetLayerType`) and refuses a stale `qwen35.nextn_predict_layers`; the loader (`qwen35_deltanet_config_from_gguf_metadata`, now `pub`) replaces derived types on the key …
Commits: `9905c7c`.

## 2026-09-29 — Issue 022 T5.0 COMPLETE: the agreement cliff mapped — 4.7% depth cut PASSES the GOAT bar, quality parity holds to 11%

Bench `.benchmarks/022_t5_collapsed_goat_agreement.md`: ε=0.01 (61/64 blocks, 4.7% cut) agreement 0.9486 — the FIRST pass of the pre-registered ≥0.9 bar; ε=0.015 → 0.8955; ε=0.02 → 0.5247; ε=0.03 → 0.0301; the grid points all FAIL.

## 2026-09-29 — Issue 022 T5.1 lane (1) COMPLETE: gemma-2 f16 control — clean negative at every real depth cut; the fine-end bracket is structurally empty

Instrument `0e0436c` (`PostLayerHook` capture seam on `forward_gemma2_f16_tapped`; `examples/twt_gemma2_profile` behind `twt_gemma2`; arch arms in `twt_collapse_emit` + `twt_goat_agreement`), verdict `55fa831`; bench `.benchmarks/015_t51_gemma2_control_goat.md`.

## 2026-09-29 — two Windows batch traps hit by the T2/T3/T5.1 schtask runners (recorded; the deleted wait-loop script's durable note)

(1) A `Start-Process`-launched `.cmd` inherits the MSYS PATH — `tasklist | find` resolves `/usr/bin/find` and the wait-loop wedges forever; the schtask launchers are immune (clean system env → `C:\Windows\System32\find.exe`).

## 2026-09-29 — Research 004: Disaggregated Quantization distilled (arXiv:26.26333) — three issues filed, riir-train Plan 430 routed

Distill of arXiv:2609.26333 (DQ/QADD), deepening the riir-clippy arxiv-walk-217 row: Issues 026 (phase-isolated quant sensitivity bench), 027 (T0 encoder-only asymmetric Q2_0 + Lloyd-Max grids), 028 (dual-PTQ resident disaggregated serving) filed; track (c) → riir-train Plan 430 (QADD prefiller …)

## 2026-10-01 — Plan 614 LANDED: the DQ phase matrix ran to EXIT0; every axis INADMISSIBLE at the frozen corpora; the instrument (and its defect chain) is the deliverable

v2 `23bbff5`+`3cf9ae9`, EXIT0 (log `F:/wt/dq614-matrix2.log`, report `F:/wt/dq614-matrix2/dq_phase_matrix.md`, record `.benchmarks/023_dq_phase_matrix.md`, `.highwater` 15→23): all four axes INADMISSIBLE (base arith 0.9583 — the `a07fffd` `reset_state` fix lifted the genuine rate; NIAH pooled …)

## 2026-10-01 — Issue 027 CLOSED measured-negative + removed: the offline LUT grid lane (Plan 615) — weight-space wins were real, the 2-bit class is function-destroying on dense artifacts

`2cb807d`: T0's asymmetric encoder (`quantize_row_q2_0_asymmetric`) +4.9 dB/family and Q2_0A's Lloyd-Max grids +0.5–1.2 dB/family — but MiniCPM5-1B ppl 6.8e5–7.0e6 vs base 49.8 at EVERY reachable granularity; the pre-registered class gate fired.

## 2026-10-01 — Issues 998 + 1004 CLOSED (hygiene): the D4 re-narrowing dissolved under the ratified no-migration architecture; the corpus-follows triggers resolved negative

998: the D4 re-narrowing is resolved-dissolved — P3/T7 landed as the ENCODER-lane unification under its pre-registered NO-DELETION verdict and D10 (owner-gate, riir-ai 1016) ratified the training families stay riir-gpu-side BY DESIGN; every widening's consumption edge is live, reversal would break …

## 2026-10-05 — Issue 031 CLOSED measured-non-repro + removed: the dq614 FATAL teardown hang did not reproduce across 6 arms; the A/B instrument is landed as durable

Instrument: `crates/riir-infer-gpu/src/bin/dq614_teardown_repro.rs` (`required-features = ["ternary_gemv_cuda_raw"]`, whole-file `#![cfg]`, dev-profile refusal) with arms `--exit plain|graceful|hard` × `--sticky` × `--alloc-mb` × `--iters` × `--no-cuda`; no-op repro knobs `DQ614_FORCE_FATAL=1` / …

## 2026-10-10 — Issue 040 CLOSED landed + removed: the weight-histogram MDL description-length audit (GGUF lane); the EXL3 rows ride the loader lane

`483f2ec`: `quant::desc_len` (feature `desc_len`) + the `desc_len_audit` report bin — per-tensor entropy / floor / slack sorted by HONEST slack, with the two-part honesty law (`N·H(p)` prices the weight VALUES only; the codebook + block-format side-info is priced separately, advisory rows sink).

## 2026-10-11 — Issue 1003 CLOSED (owner-gate record superseded): the D10 disposition stands — the 69 training families stay riir-gpu-side BY DESIGN; the live homes are the riir-ai BOUNDARY.md D4 row + riir-ai HISTORY:360

This closure SUPERSEDES the 2026-10-01 sentence in the Issue-998 closure row ("Issue 1003 stays OPEN deliberately") — that sentence predates the owner's 2026-10-05 rule ("rm owner-gate issues; ask Claude to judge; defer→HISTORY+close", riir-ai HISTORY:360) and the Claude-adjudicated verdict of …
Commits: `65a7268779`, `7e0e7d74`, `498b1732b9`.

## 2026-10-11 — Issue 028 CLOSED (PoC gate fired NULL, container shelved): the dual-PTQ disaggregated serving record; T5 stays deferred with its trigger

T1–T3 landed `09d0dd5`: the `DisaggregatedTernaryWeights` container (`load_pair` / `.pf` one-file / `from_single`), the ownership-preserving in-process `PhaseHandoff`, the escape-set law (`ssm_a`/`ssm_dt.bias`/`ssm_alpha`/`ssm_beta` bit-shared between copies, loud refusal naming divergence), the …

## 2026-10-11 — katgpt-rs Plan 622 Phase 7 engine lane LANDED (doc-sync row): the scale-invariant attention vehicle on the packed Bonsai ternary decode path; GPU legs open, GPU-window-gated

Landed `d1b9bd5` (session katopz-m3-622-continue) + follow-ups `71a0845` (loop_probe stale layer_idx repair) / `47f8f6f` (quiet the posture-dependent unused layer_idx warning) / `9f3b27b` (the harness compiles at its OWN required-features posture — the P43 class: single-feature clippy runs skipped …)

## 2026-10-13 — Issue 041 CLOSED (T1–T5 all landed): the prefix-cache snapshot submit burst — 96 submits → 2 launches per direction, the serving-lane TTFT lane

Landed `971b842` (T1+T2: the pointer-table copy path — module-local `TableCopyEngine` in `qwen38_prefix_cache.rs`, dual device pointer tables (live-state freshness-checked via a host address compare + per-entry tables), ONE `prefix_table_copy` launch per class per direction, opt-in …)
Commits: `48c2948`, `bb2a4b5`.

## 2026-10-13 — Issue 1005 CLOSED (T1–T8 all landed; defers carry their triggers): the eDLM/Drex inference lane — GGUF arch, block-causal mask, pointer head, marker encode, real-artifact parity, GPU forward, KV reuse, CUDA graphs

Phase 1–3 complete across four sessions: T1–T3 (the `edlm` GGUF arch + segment block-causal mask + row-form fallback), T4–T6 (marker encode/pack + the pointer head + T6 parity PASSED on the real `drex-dlm-Q8_0.gguf` — 87 tokens EXACT vs their published count, worst drift 0.0103, all three argmaxes …)

## 2026-10-13 — Issue 039 CLOSED measured-non-repro + removed: the wgpu `--lib` device-loss tail did NOT reproduce at HEAD on a clean exclusive box (green ×2 parallel + ×1 serial); the CUDA-posture leg surfaced and fixed a posture-biased TEST instead

The free window opened (verified idle: 5% util, 376 MiB, no compute apps) and the full recipe ran from a fresh isolated worktree at the then-tip `fbd6e6c` (`E:\git\wt039infer`, isolated `CARGO_TARGET_DIR`): wgpu posture (parallel) 258/0 in 520.27s; wgpu posture again (the issue's own twice-in-a-row …)
Commits: `450cdfc`.

## 2026-10-22 — Issue 1107 CLOSED negative-audit + removed: NO riir-infer path right-pads rows into a shared recurrent (GDN) prefill — the TokenGate pads-to-front law has nothing to repair here; the one multi-sequence surface (laya `forward_packed`) is varlen-packed by construction, and the riir-ai HLA note does not fire either

Audit per the issue's task 3 (a negative result is a valid close; not left open as a maybe). Paths checked, each at its own entry point's signature and launch geometry: (1) CPU `prefill_qwen_deltanet` / `_into` / `_chunk_into` (`src/deltanet/forward.rs`) — `prompt_tokens: &[usize]` single slice …

## 2026-10-23 — Issue 1111 CLOSED + removed: `forward_qwen_deltanet_ternary_hidden` returns the head basis (`R·h` on a Hadamard-folded file) — KEPT and documented, width fixed, pinned on the synthetic folded fixture

Filed from riir-train Plan 461 T2.0 (P3 measured the returned vector = the FORWARD-ROTATED final-norm state on `Ternary-Bonsai-2-27B-PQ2_0.gguf`; the doc claimed the primal state and D = 12288).
Commits: `4ede8e7b4`.

## 2026-10-25 — Issue 1113 CLOSED + removed: the mma `kv_f16_to_f32` +112 re-bias — fix + both pins EXECUTED on the 4090 (the compile-in-context window the M3 could not reach)

The one-line re-bias (`(exp + 112u) << 23`) and the two device-free pins landed 2026-10-24 (zed-m3-docsync-956-close) from the M3, where the module compiles to nothing (`#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]`); the M3's scratch `rustc` harness validated the pin …
Commits: `f72eff6`.

## 2026-10-25 — Issue 1114 CLOSED + removed: the Issue-994 KV guard default is backend-keyed (Metal measured clean to 128K)

The single-buffer KV guard's 208 MiB default is the 4090/`wgpu<spirv>` bisect (riir-ai Issue 994). On the M5 Ultra under `wgpu<msl>` the 64K and 128K rungs both passed their anchor-ref gates against an independent 32K artifact (riir-train Bench 657), so the guard refused working configurations …
Commits: `cd748a6`, `38612c5`.

## 2026-10-25 — Issue 1110 CLOSED + removed: the β-reopen probe — ROW DIES AT PROBE COST (Bench 890, renumbered from a colliding 888 — see the record's numbering note): the only real gemma2 checkpoint is a plain IT model, the d2f decode is a structural 1-step `<eos>` collapse, the drift census has zero substrate

T2 executed on the 4090 (CUDA posture `gemma2_d2f,cuda_backend`, detached runs, GPU 44 °C / 5 % util, no compute consumers).
Commits: `b2fe307`.

## 2026-10-26 — Issue 1119 CLOSED + removed: the zero-copy cache-split handle gap — fix `7b2fed9`, acceptance = the i1122 gate1 cell re-run

The riir-ai Issue 1122 gate1 cell (`gate1_site_isolated_tax`) ran bit-identical simdgroup code with 0 site-timing rows and n=0 site dispatches — the G1-instrument illusion class.
Commits: `7b2fed9`.

## 2026-10-25 — Issue 1115 CLOSED + removed: the GEMV occupancy wg rule default-on EVERYWHERE (the 4090 CudaRuntime A/B measured the platform split away)

Bench 887: prefill-tail lm_head A/B on the 4090 CudaRuntime — G1 FNV-identical ×6+6 rounds (pp512 + pp4096, Bonsai-2 vocab 248320 → the 128-wg bucket), reach exact (+1/±0 across 12/12 rounds via the counter), G3 1.0013×/1.0000× sub-noise → per the issue's own done-when …
Commits: `f4e6369c`.

## 2026-10-26 — Issue 015 CLOSED + removed: the silero-vad audio lane PoC — T6 verdict finalized (quiet-ish-box latency re-run, replicated ×2)

The PoC itself LANDED 2026-10-23 (T0–T3: `vad-ane` feature on `riir-infer-audio`, digest-walled loads both postures via `riir-infer-ane`'s `load_model_precompiled` + `predict_f32_multi`, 5-test gate suite green, the ANE finding — all-CPU placement on M5 Ultra/macOS 27, strict ANE gate unusable as a …)

## 2026-10-26 — Issue 1123 CLOSED + removed: the issue028 dual-format q4-prefill gate red on develop — the TEST was stale, not the loader (the TernaryQ2 promotion outran the battery)

Filed 2026-10-26 from the plan-623 Phase 1 validation sweep: `dual_format_q4_prefill_pair_loads_and_runs` tripped `matches!(dec.layers[0].in_proj_qkv, ProjWeights::Ternary(_))` at pristine `7a01f05` (filer's detached-worktree run verified it was NOT a 623 regression).
Commits: `5ad32ba`, `c787460`, `11a30c6`.

## 2026-10-26 — Issue 1125 CLOSED negative-already-shipped + removed: the Bench-1004 fused-quantize rung — the #395 mechanism IS the shipped K1–K5 posture; the census falsifier discharges it dynamically (0 standalone quantize launches on the 4090)

Filed from riir-ai Bench 1004 (the 0.9730× decode RED, first sub-parity decode cell of the scored era; the opponent's +3.84% jump rode prism b10826's #395/#401/#402 FWHT/q8_1 fusion quartet).
Commits: `a2ebb37`, `2e3889e`, `1320dda`.

## 2026-10-26 — Issue 1126 CLOSED measured-negative + removed: the GDN core fusion rung (Bench 898 −3.5%/−2.1%, G1 exact everywhere, opt-in `RIIR_GDN_CORE_FUSED`) and its named 4→3 follow-up (Bench 899 direction-positive 3/3 pairs, opt-in `RIIR_GDN_BETA_EXPAND_FUSED` — NOT promoted, league re-cell owner-gated)

The #398 conv-chain twin implemented as `recurrence_f32_fused_expand_hd{64,128,256}` — ONE launch replacing beta_decay + expand_and_l2_normalize + recurrence_fused, the Issue-742 body shared verbatim via `recurrence_fused_body_impl<HD, EXPAND_INLINE>` (`4132d94`).
Commits: `8aabe55`.

## 2026-10-26 — Issue 1127 T1 LANDED (Bench 900): the attention-chain fusion rung — 5→1 launches, G1 bit-identical incl. the graph lane, G2 4/5 paired windows positive (armed median +0.77%), opt-in `RIIR_ATTN_CHAIN_FUSED` — + Issue 1131 filed: the GDN rotation FWHT geometry rung (the census rows outside T3's family)

The T1 rung (`attn_chain_fused_f32` + the Issue-618 devpos twin for the graph-capture lane, `8d51f21`+`1d0291d`) merges split_qg + q/k-norm + RoPE + kv-append into ONE launch per attention layer — every body verbatim, ordering confined to block-local barriers, read-barrier-write rope staging …
Commits: `687f5669`, `63617c9`, `86e419d`.

## 2026-10-28 — Issue 1147 CLOSED + removed: the T2.4 kernel micro-batching rung — fused-batch v0 LANDED (`03f0fe6`, Bench 912) then DEMOTED same-day on the 2-box wire refutation (Bench 913: armed 46.9–52.6 vs incumbent 99.2–108.5 aggregate decode window, −52% — the BatchN wave forfeits the head⇄tail GPU overlap that IS the 2.37× bubble fill)

`pair_decode` ships default-OFF (`RIIR_DIST_PAIR=1` arms it for loopback rigs only, where the +39% was real and stands).

## 2026-10-28 — Issue 1130 CLOSED G2-FAIL + removed: the Metal concat-epilogue split rung (folded GDN qkv|z Split2 into the ternary concat GEMV epilogue)

Rung `b635a43` (`gemv_ternary_plane_rowtiled8_f16scale_split` + `GemvTernaryCubeCL::launch_split`); G1 to_bits EXACT 13/13 on amo + m3 (`9f1b2ba`).
Commits: `445cc4e5a`, `760265d`.

## 2026-10-28 — Issue 1131 CLOSED + removed: the T6 GDN rotation-FWHT geometry rung (CUDA) re-priced sub-bar, re-pin adjudicated NO-GO

Filed `86e419d` (Bench 902 rows `gdn_input_rotabq` + `gdn_out_norm_gate_permute_rotq`, 564 µs/token).
Commits: `8d204ba`, `194b952`.

## 2026-10-28 — Issue 1132 CENSUS COMPLETE + removed: the Metal per-dispatch floor census (whole-graph launch attribution on the folded decode)

Instrument `3d95ff9`: `DECODE_STAGE_MASK` 11→22 bits (per-launch-site refinement classes + the previously-ungated decode tail; mask=ALL bit-identical) + riir-ai `b864dc936` Phase E — the census rides the stage-instrument plumbing, no xctrace.
Commits: `772c8bf`, `1a28511`, `800d7b4`, `c38d28f`.

## 2026-10-28 — Issue 1133 CLOSED measured-null + removed: GDN conv+beta/decay+expand merge rung (row above, 2026-10-27)

Rung `0eaebad`, record Bench 905 (`.benchmarks/905_i1133_conv_merge_ab.md`): G1 exact everywhere, G2 2/4 pairs, pooled ≈0% ± 0.5%; ships opt-in `RIIR_GDN_CONV_MERGE=1`, default OFF.
Commits: `537ca18`.

## 2026-10-28 — Issue 1134 CLOSED G2-FAIL + removed: the norm-pair epilogue fold rung (input_norm + mid_norm)

Donated kernel `ResidualAddRmsNormRotateCubeCL` (`cbb5bf1`; G1 to_bits 3/3 + REACH PASS) is an e2e LOSS in 7/7 A/B runs (riir-ai `bench_1137_normrot_e2e_ab`): −2.96/−2.23 … −8.90/−9.48%.
Commits: `b02f379`, `4aed461`, `f44c740`, `ffb07bf`.

## 2026-10-28 — Issue 1135 CLOSED ceiling-gate NULL + removed: the attention proj+decode launch-cluster rung

Three verified-idle interleaved windows on amo (riir-ai `bench_1138_attn_proj_decode_strip_ceiling` + raw log `.benchmarks/1138_issue1135_attn_cluster_strip_ceiling_amo.raw.log`): proj 32 la/tok × 13.4-20.5 µs (pool 0.43-0.66 ms/tok); the only legal shape (648-F9 wq+wkv row-concat, −16 la/tok) …
Commits: `eafd97d`.

## 2026-10-28 — Issue 1136 CLOSED + removed: the gdn_out_gemv rung — ceiling real, G1 structurally dead; GDN chain launch budget CLOSED BY MEASUREMENT

Ceiling (riir-ai `bench_1139_gdn_outproj_strip_ceiling`, raw log `.benchmarks/1139_issue1136_gdn_outproj_strip_ceiling_amo.raw.log`): +0.524/+0.592 ms/tok (+2.16/+2.46%) @ 48 la/tok. Commits: `afc1f43`.

## 2026-10-28 — Issue 1137 CLOSED NULL + removed: the FFN decode-GEMV ALU-chain rung (and its hand-MSL lane i1138, Bench 1149)

Fork mapping study (`0cc47fa`: `kernel_mul_mv_tq2_0_f32`, PR-26980 algebra) found its in-tree twin `gemv_ternary_fma_cubecl` (Plan 563 / Bench 847, `RIIR_GEMV_FMA` default-OFF); riir-ai `bench_1141_fma_aspect_probe` on M5 Ultra: FMA loses ~3× on every aspect (gate/up 156.8 vs 47.9 µs, down 178.3 vs …)
Commits: `6277cd2`, `194b952`; lane i1138 closed MEASURED-NEGATIVE in riir-ai `.benchmarks/1149_i1138_field_e2e_ab_amo.md`.

## 2026-11-02 — Issue 1152 CLOSED + removed: verbs_metal fw gate + the amo RTR EINVAL — the MTLBuffer leg measured CLOSED on macOS 27.2 (`ebe0b11`+`429e085`)

The T3.2 MTLBuffer leg (Plan 623) was code-complete since `570ee79` but gated twice: amo's Application Firewall silently dropped TCP SYNs to unsigned listeners (no GUI session to accept the prompt), and an amo-side provider regression EINVAL'd every `ibv_modify_qp(RTR)` (`mask 0x101181 rc=14`, reproduced by verbs_ping byte-for-byte, no port/GID/link signature — driver-internal provider state). Both cleared on owner acts: the `socketfilterfw --add/--unblockapp` rows (survived the macOS 27.2 update, matched the rebuilt binaries' linker-adhoc CDHash) and the 11-02 on-site power-cycle (amo self-shutdown 11-01). All rows PASS both directions: REGISTRATION 216.9/80.7 µs, GPU→wire every-byte-0xC7, wire→GPU gpu_sum==cpu_sum, canary RTT p50 6.375 µs / BW 9.380 GiB/s. Two probe fixes landed (`ebe0b11`): verbs_metal's missing wire→GPU pattern recv (UC RQ FIFO — latent since `570ee79`, first execution ever) + verbs_ping's recv-window default 32→4 (macOS 27.2 caps 256 KiB windows at 4–7; depth×SGE-len-shaped, not bytes — 32×64 KiB green at the same GiB/s). Full rows: `.benchmarks/928_plan623_t32_rkey_mtu.md` Addendum 2. Issue file removed per the noise rule; the full text lives in git history.

## 2026-10-31 — Issue 1159 CLOSED + removed: non-aligned prefill chunk slow path — the chunked conv1d's p % 64 gate (`6a889df`)

- Located by the 1146 stage census at one wide frame (tensor-zerocopy, incumbent prework): deltanet Δ 7370 ms at p=3000 vs 3524 at p=4096; projections linear in p. Cause: `can_chunk_conv1d` required `p % 64 == 0`, so a ragged chunk ran p per-token conv1d dispatches per layer. Fix: the final chunk is short (kernel's partial-chunk carry merge already G1-pinned). p=3000 13571 → 8804 ms (217.8 → 340.8 tok/s, above p=4096's 331); final logits bit-identical to the fused-prework arm at p=3000 and p=4033. Mac serving (riir-gpu default `deltanet_prework_fused`) was never affected; the riir-ai Plan 644 gate posture was (its 1.059× @3000 is against the slow baseline — riir-ai Issue 1147 B6).

## 2026-10-31 — Issue 1158 CLOSED + removed: mask-input drafter refinement audit (JetSpec++ pass-1 collapse class)

Inventory of every `mask_token_id` reader, two classes, never pooled. **HAZARD (mask-input block drafters, single-pass J0 by contract):** riir-infer `qwen38_dflash2.rs` (`draft_block_hidden` input = anchor + all-mask; `lattice_walk_reselect` re-scores the J0 grid, no re-forward) + `speculative_decode/dspark_drafter.rs`; riir-ai `riir-gpu/src/dflash_config.rs` + DFlash2 consumers (benches 746/747/759/779/989, plan614 lanes/spec/floorgap/premise_966, plan632, issue1142, riir-engine bench_694); riir-train `dflash_trainer.rs`, `dataloader.rs::batches_dflash` (THE fully-noised recipe), `dflash2_chat_head_train.rs`, `chat_dflash2_probe`. No refinement-pass knob exists anywhere in this class today. **NOT the class (dLLM/D2F multi-step denoisers, trained on partial ratios — J≥1 inputs in distribution):** katgpt-forward `d2f`, riir-infer `gemma2_d2f*` + `issue1110_reopen_probe`, riir-train `dllm/*`, `distill_tsd/gdsd/harness`, `opsd_loop` (the filing's trainer list misfiled these three — `noise_min/max_ratio` corruption), plan357/389/409/410/433/437 rigs, `bench_dllm_kernels`, riir-ai `goat_250`; katgpt-rs `funcattn_g6_*` are MLM tests. Guards (doc-only, name this issue): `draft_block_hidden`, DSpark noise-block fill, `DFlashConfig.mask_token_id`, `batches_dflash`; cliff detector pinned at the verify seam (`speculative_decode/mod.rs` — J1 τ below J0 ⇒ refuse the arm). Fix vehicle unchanged: riir-train Plan 376 T3. Commits: riir-infer `1f85c99`, riir-ai `e6a2cd1e0`, riir-train `bcd9e30e`.

## 2026-11-02 — Issue 1156 CLOSED + removed: the sparse-prefill league lane — T1–T6 all DONE, τ=0.85 promoted-by-verdict (bench 1157; promotion lane = issue 1165)

Full arc: T2 block-sparse arm (τ=1.0 bit-identity contract, bench 1151); T3 the ΩMSR per-family sweep — blanket promotion REFUSED on the mean scorer (bench 1152), T3-v2 the envelope scorer re-open CONFIRMED — 7/7 families floored (bench 1153, manifest v2); T4 the modelless regime gate + T4.4 the centroid freeze/calibrate (bench 933, gain 10.0 / margin 0.20); T5 fused mixed-mode (kernel G1 green); T6 the league cells (bench 1157, riir-ai `5a09b20d0`): Cell C τ=0.85 = 1.0310× @16K / 1.0565× @32K verified-idle, 0 flips, anchor FNV 6/6 — the only cell clearing the pre-registered bars; τ=0.9 demoted at these shapes; the gate belt clean but overhead-negative on a homogeneous stream; Cell D vacuous → issue 1164. Claude owner-proxy verdict AGREE → the P1–P5 promotion lane = issue 1165 (Metal, ≥16K, load-time verify fail-safe, demote-literal byte-identity). Lease folded: zed-amo-i638t1b → zed-amo-i638t1c-lineage zed-7710a782, work complete.

## 2026-11-05 — Plan 485 P2 landed in riir-train: fixed-workspace `_into` dispatch variants (`f7b0e2d`) + the pre-existing gate red fixed (`5142561`)

- **P2 (`f7b0e2d`):** nine `GpuGemmaCubeCL` dispatchers gained `_into` twins that launch the SAME kernel into a caller-owned output handle — gemv_tiled, gemm, add, gemv_batched_backward (the backward stretch's kernel), geglu, rmsnorm_batched, norm_residual_batched, rope_batched, attention_fused; feature parity with their wrappers (`gpu_training_resident` where gated, geglu ungated). Incumbents became allocate-then-delegate wrappers — one launch site per kernel, zero behavior change. The twins carry the plan-485-P3 capture contract in their docs: `Graph::replay` bakes the raw pointer, so the caller keeps the handle alive until the last replay and refreshes inputs via same-stream writes only. Consumers: riir-train's plan 485 P3 capture lane.
- **Gate red (`5142561`):** the standing clippy gate (all-targets `-D warnings`) was RED at HEAD before any 485 work — 10 findings: regime_capture_write_failed/read_failed ungated while their callers sit behind the issue-1162 `all(...)` cfg (gated to the caller's cfg, the documented class); sparse_omega_msr's temp_manifest_paths test helper gated to its caller; 3 dead test locals dropped; 4 needless borrows + 1 unnecessary_min_or_max via `cargo clippy --fix` (the cargo-refine corpus had no rule — 0-edit miss logged for the improvement queue). GREEN at default and gpu_training_resident postures; 42 CPU-only lib tests pass.

## Lessons

- Check the reference implementation, not a remembered standard, before filing a fixture-integrity finding (Issue 010).
- A G1 that stops below the production chunk size certifies a size nobody ships (Issue 032).
- A dead-module grep must cover every repo that path-depends on the crate, at `--all-features` (Issue 017).
- In shared checkouts, commit via pathspec (`git commit -- <paths>`), never the bare index.
- A run lane needs ONE owner at a time; the handoff summary's active-plan section must name the owning session (bench 023).
- A shape-widening edit must move the host bind and the kernel binding together (T10 rung 2).
- Tiny-op kernel probes read a ~90% submission floor — narrow-zone rungs must gate at FORWARD level (Issue 009).
- Bandwidth-bound zones want register blocking; latency-bound zones want cp.async/double-buffering — the rooflines diverge, so the rungs must too (Issues 007/008).
- The sgemm block-fit floor is MEASURED per GPU (SM count), never ported from another box (Issue 004).
- Trajectory (argmax-agreement) divergence overstates functional damage — carry the hit-rate column beside any agreement claim (Issue 022 T5.0).
- A tail-read of never-written buffer regions is only a valid bounds check if the buffer STARTS zero on EVERY runtime — `client.empty()` promises nothing on the CUDA runtime (Issue 039).
- Cosine/state redundancy ≠ functional redundancy, on ternary and dense alike (Issue 022 T5.0/T5.1).
- Weight-space dB wins mean nothing without the model-level gate — the class verdict can kill every arm (Issue 027).
- Windows batch: no literal `)` in echoes inside `( ... )` blocks; never hand-launch a wait-loop `.cmd` via `Start-Process` from MSYS; prefer `schtasks` wrappers.
- Feature-gated targets carry their `required-features` row at birth — a whole-file `#![cfg]` target without one prints a green zero forever (the T1.1e law).
- `[patch.crates-io]` does not survive crates.io publication — `publish = false` until the vendor forks land upstream.
