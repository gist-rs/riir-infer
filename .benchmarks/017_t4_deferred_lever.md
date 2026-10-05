# Bench 017 — T4 kernel half: the fused deferred-restore lever lands; the pre-registered promotion rule FIRES PROMOTE

**Status:** COMPLETE — cell A + cell B PASS, probes PASS, the ≤ 1.20 @4097 bar is met (1.050×/1.123×); katgpt-rs-side promotion EXECUTED same day (`fitted_v_reconstruct` → katgpt-core default).

**Instrument:** `kv_reconstruct_gate --recon deferred` (riir-infer `fe75497`, exe `target-rel-t4\release` built from the committed tree BEFORE task registration — the Issue-012 recipe). Table: T2's `.benchmarks/012_kv_table_residual.bin` (7,348 rows, BLAKE3-verified). Fixture: gemma-2-2b-it-f16 + chat_probe, eval tokens [61440..), `--skip-g1` (G1's record is Bench 013's). Chained schtask `riir_infer_t4_deferred`, cells sequential, box otherwise idle; launcher `run_t4_deferred_cells.cmd`.

## What landed (the two pre-registered levers, as one read path)

1. **Fused deferred restore** (`attend.rs::attention_head_recon_slices` + `fused_epilogue`): the rotation `G(−θt)·K̂_t` rides the attention V-accumulation loop (pass 3) over the hot K rows — no scratch materialization, no per-row λE add; the table part regroups into per-(head,row) weight sums `W_s = Σ_{t: s_t = s} w_t`, applied once per distinct row in the epilogue (`attn_out += λ·Σ_s W_s·E_l[s]`). This is Bench 895 Addenda I/II's deferred restore with the row index pre-resolved per position (`set_token` resolves the layer-independent `row_index_of`; the eager path re-resolved per layer per step) and its last scalar add (`W[row] += w`) in THIS repo's softmax-weight loop, exactly as the issue specified.
2. **Bitwise per-position (sin,cos) table** (`VReconState.cs`): built once with the exact `(pos as f32 * freq).sin_cos()` expression the on-the-fly action evaluates — block angle-addition's no-re-anchor limit: identical bits, zero transcendental cost. The fused rotation is bitwise the eager rotation (pinned by test + the in-run λ0 probe).

## Numerics contract (all gated in-run)

- **λ0 law:** deferred-λ0 ≡ eager-λ0 **to_bits** (65-position real-model probe, both cells) — the fused pass-3 expression is the plain kernel's over rows that equal `v̂_t` bitwise (`add_scaled_row_inplace`'s λ==0 early return). Unit-pinned in `attend.rs::fused_recon_tests` (sequential + parallel regimes, planted −0.0 at pos 0).
- **Miss law:** all-miss λ=1 ≡ eager **to_bits** (probe + unit tests).
- **Regrouping bound (λ>0):** max |Δlogit| vs eager **8.011e-5** (65-position, λ*=1) — the stated reassociation bound `2·t·ε·λ·max|E|`'s far tail; def-λ1 vs STORE 7.915e-5 (the same rotation-rounding class as eager's 1.254e-4 — the regrouping partially cancels rotation rounding against the store path).
- **G3 seam identity** (unchanged posture): PASS, 1.254e-4 — byte-identical to Benches 013/016 across all three runs today (cross-run determinism of the probe).
- **G1:** SKIPPED per the T4 pre-registration (the record is Bench 013's: 0 flips at every λ); the deferred-vs-store 7.9e-5 sits in the same class the G1 tolerances (2e-3 mean / 5e-2 max NLL) dominate.

## Cell A — context ~1025 (`--seq-len 1024 --tg-pairs 4 --tg-prefill 1024 --tg-decode 64`)

| arm | median µs/step | tok/s | ratio vs full | prefill ms |
|---|---|---|---|---|
| full-cache | 127,624 | 7.8 | 1.000× | 127,549 |
| eager-λ1 (anchor) | 224,952 | 4.4 | **1.760×** | 174,354 |
| def-λ0 | 129,979 | 7.7 | **1.020×** | 128,290 |
| def-λ1 | 134,409 | 7.4 | **1.053×** | 131,276 |

Eager anchor reproduces Bench 016's cell A (1.789/1.751× — within drift); box-state continuity confirmed in-session.

## Cell B — the sliding-window edge, context ~4097 (`--seq-len 4096 --tg-pairs 2 --tg-prefill 4096 --tg-decode 64`)

| arm | median µs/step | tok/s | ratio vs full | prefill ms |
|---|---|---|---|---|
| full-cache | 143,862 | 7.0 | 1.000× | 543,681 |
| eager-λ1 (anchor) | 557,841 | 1.8 | **3.887×** | 1,347,558 |
| def-λ0 | 151,072 | 6.6 | **1.050×** | 556,007 |
| def-λ1 | 161,051 | 6.2 | **1.123×** | 582,295 |

**The decision rule reads this cell: 1.050×/1.123× ≤ 1.20 — PROMOTE.** The eager anchor reproduces Bench 016's cell B (3.821/3.957× — the HOLD's evidence stands, and this run's anchor proves it wasn't box drift). The prefill also collapses: eager 1,347s → deferred 582s (2.3×) — no scratch write/read, regrouped E traffic.

## The decision curve (the Bench 016 curve, re-measured with the lever)

| context | naive/eager (Bench 016 → this run) | **deferred (this run)** |
|---|---|---|
| ~129 | 1.079/1.088× (Bench 013) | — (below measurement noise at this geometry) |
| ~1025 | 1.760× (anchor) | **1.020× / 1.053×** |
| ~4097 | 3.887× (anchor) | **1.050× / 1.123×** |

The read-path cost is now ~flat in context (+2–5% λ0, +5–12% λ*) instead of ~linear — the per-position rotation fused into an existing pass instead of a separate eager sweep.

## Bytes/token

Re-confirmed: full 212,992 B/token → P3 106,496 B/token = exactly **50.0%**. Deferred mode drops the eager scratch entirely (the `VReconState` scratch is never written; a production cache drops the V allocation — the recorded arithmetic).

## Verdicts

- **T4 decision rule: ≤ 1.20 @4097 → PROMOTE — FIRED.** katgpt-rs-side promotion EXECUTED 2026-10-05: `fitted_v_reconstruct` → katgpt-core default (feature-def comment + README count + HISTORY § Issue 883 amendment carry the verdict; default-on = compile availability of the primitive — zero runtime cost unless a consumer selects `VReadPath::Reconstruct`).
- **GOAT reading:** G1 quality — Bench 013's 0-flips record + this run's bitwise laws + the 7.9e-5 regrouping class: PASS. G2 cost — the pre-registered window-edge bar: PASS (1.123× ≤ 1.20). G3 no-regression — the default read path stays the bitwise `FullCache` kill switch; the promotion compiles the primitive, wires nothing: PASS. G4 alloc — the fused path writes pre-allocated buffers only (cs table + weights + used allocated once at construction): PASS by construction + the unit suite. Modelless gain — half the KV cache for ≤ +12.3% window-edge read cost, zero training: the modelless-gain law holds.
- **En-route repair the promotion forced:** `bench_895`'s P3 G2 primitive-level ≤ 1.00× gate (the Bench-895-era aspiration the model-bound rule superseded) joined the DEFAULT all-targets surface with the feature and red-ed `cargo test -p katgpt-core` — flipped to RECORDED with the provenance (the number still prints; the gate lives at the Issue 013 T4 bar).
- P1/P2 remain the demoted losers (Bench 011 negative, Bench 012 non-viable) — untouched by this promotion.

## Box state

4090 workstation (shikuwa), i7-13700K 16-core, CPU lane, AC plugged. Launch: 32,538 MB total / 20,868 MB available physical; no cargo/rustc/kv_ process at launch (launcher-guarded); cells sequential via schtask `riir_infer_t4_deferred` (deleted after the run); wall: cell A 42 min (04:11→04:53), cell B 104 min (04:53→06:37), rc 0 both.

## Not measured (unchanged scope)

- >window contexts (global layers reading past 4096) — out of scope per the T4 pre-registration (the naive CPU path is not the production deployment shape; the ratio is ~flat in context now, and the sliding-window layers bound the K read at the window regardless).
- GPU decode (the lane is CPU; the production GPU path has its own kernels).
