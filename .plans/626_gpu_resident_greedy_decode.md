# Plan 626 — GPU-resident greedy decode: the queue-never-drains token loop (Issue 1168 T3)

**Status:** PLANNED — T1/T1b measured (Issue 1168); implementation unclaimed.

## Premise (measured, Issue 1168)

The decode league lane pays **17.79 ms/token** (P1: enqueue-front 2.455 +
GPU-serialized read 15.328) against a measured deep-queue floor of **15.32
ms/token = 65.30 tok/s** (P2) — the floor alone reads ~1.0× vs the Bench-1161
same-window opponent (63.5–63.9) and closes the decode hold. The per-token
`read_one` sync drains the queue; the next token's enqueue front then runs on an
idle GPU and its read blocks for the full GPU token. The only VALUE-dependent
CPU step is the input-embedding upload (set_input = 0.006 ms); every dispatch is
shape-only. P3 (sync every 2nd token = 16.61) confirms the mass scales with
sync frequency. The readback itself is CHEAP (0.17–0.25 ms) — T2 is closed
refuted.

## Design

**Substrate-first (verified 2026-11-03, this session): both kernels ALREADY EXIST
in-tree — the arc is a WIRING plan, not a kernel plan.**

- **Argmax-on-device: EXISTS** — `sampling_cubecl.rs` `argmax_f32` (parallel
  reduction → u32, 4-byte download) + launcher, plus `temperature_argmax_f32`.
  `buffer.rs` already documents the "GPU-resident argmax: download only 4 bytes
  instead of full logits" intent. T1 = verify the tie rule (first-index; the
  file's -1e30 note) against the CPU reference on all-tie + single-max +
  248320-shape fixtures, and pin it with a test.
- **Row-gather-on-device: EXISTS** — `deltanet_tree_verify_cubecl.rs`
  `row_gather_f32` + `gather` launcher (the spec-verify lane gathers rows on
  GPU today); multi-row variant `gather_rows_f32` in `encoder_lane_cubecl.rs`.
  T2 = verify an f16-table arm (the GGUF `token_embd` is f16; the incumbent CPU
  path uploads exact f16→f32 rows — the gather must produce bit-identical f32,
  cast in-kernel) or, if the f16 arm is not worth a kernel, upload the table
  pre-cast to f32 once (2.4 GB → 4.8 GB f32... measure; the f16-in-kernel cast
  is the memory-honest arm) and wire `row_gather_f32` against it with the
  bit-identity test.

Keep the queue deep across the token boundary by making the greedy token chain
fully device-side:

```
host loop (greedy lane, opt-in):
    wait K-harvest fence (every K tokens: read K ids, detokenize, drain-check)
    enqueue: [embed-gather(id_buf → input_row)] → [norm] → [64 layers] →
             [lm_head] → [argmax → id_buf[next]] → ...
    # NO per-token host sync; the id never reaches the host mid-stream
```

1. **Argmax-on-device: CONSUME** (`sampling_cubecl.rs`) — reduction over
   `logits[0..vocab]` writing a single `u32` id into a device buffer.
   **First-index tie rule** (the determinism law — Issue 940): pin against the
   CPU reference (all-tie and single-max fixtures) before wiring.
2. **Embed-gather: CONSUME + f16 arm** (`deltanet_tree_verify_cubecl.rs`
   `row_gather_f32`, or a small f16-table twin if the cast must happen
   in-kernel) — copies row `id` of the token-embedding table into the input row
   buffer, same layout the incumbent's CPU row-upload produces. The table
   uploads ONCE at construction (`token_embd` f16, 248320×5120×2 B ≈ 2.4 GB —
   unified memory, no VRAM pressure class on M-Ultra; the VRAM pre-flight row
   documents it). f32-table variant behind the same kernel (cast at upload).
3. **The resident loop** (`ternary_deltanet_gpu_forward.rs`, new mode beside
   `forward_token`): `resident_enqueue(&mut self, next_is_fresh: bool)` +
   `resident_harvest(&mut self, k: usize) -> Vec<u32>`. The incumbent lane stays
   byte-identical (all existing entry points untouched); the resident lane is a
   SEPARATE entry point so the sync-per-token contract (Issue 860's
   consume-fresh-input law) keeps one owner per posture.
   - KV/rope/pos bookkeeping: pos increments per enqueued token (the incumbent's
     host-side pos state moves to a device-mirrored counter read at harvest).
   - `CUBECL_WGPU_MAX_TASKS`: one token = 1127 tasks; the 1024 cap throttles
     ~1 token deep — the queue-depth invariant holds by construction (the
     regulate class, Bench 763 docstring); K-harvest every 8 tokens keeps the
     host out of the way (harvest = 1 fenced 32 B read ≈ 0.2 ms amortized).
   - Non-greedy sampling (top-k/top-p/temperature): FALLS BACK to the incumbent
     per-token sync lane — the resident lane serves greedy-only (documented
     loud; the serving router picks per request).
4. **Posture**: opt-in `RIIR_DECODE_RESIDENT=1` (PROMOTE-ONLY, exact-literal
   arms; unset/anything-else = incumbent byte-identical). Launch counter =
   vacuous guard (Bench 996 law): `RESIDENT_LAUNCHES` increments only on the
   resident path.

## Tasks

- [ ] T1: CONSUME `sampling_cubecl::argmax_f32` — verify + pin the tie rule (all-tie, single-max, 248320-shape) against the CPU reference; wire the id into a device-side buffer the next token's gather reads.
- [ ] T2: CONSUME `row_gather_f32` (or its f16-table twin) — bit-identity vs the CPU row-upload path (f16 table, 1e6 random ids); table upload-once at construction.
- [ ] T3: resident loop mode + harvest seam + env arm + counter; incumbent lane untouched (byte-identical boot assert).
- [ ] T4: greedy bit-identity e2e — resident greedy walk == incumbent greedy walk (token-id sequence EXACT) on the 27B; G2 A/B via the bench_763 posture (P1-shape vs resident) + the league re-cell (bench_768/831 tg/ctx) before any promotion.
- [ ] T5: promotion verdict — GOAT bars + reviewer/owner verdict; demote check = env literal restores the incumbent lane byte-identically.

## Bars (preregistered)

- G1: greedy token-id walk EXACT vs incumbent (FNV-class walk pin, ≥3 walks).
- G2: harness P1-shape ≥ −2.0 ms/token (17.79 → ≤15.8) AND the league decode
  re-cell improves every row; the promotion target is the league ratio ≥ 0.97×.
- G4: no new steady-state hot-path allocations (harvest buffers pre-allocated;
  the rotation-pool discipline — Issue 994 guards live).
- The 604-era blocker (CubeCL handle churn under sustained create/drop) is
  guarded by `pool_poison.rs`; the resident loop REDUCES handle churn (no
  per-token read allocations). If the vendor surface misbehaves, patches stay
  IN-REPO (`vendor/cubecl-runtime-0.11.0-pre.2`) and are owner-gated.

## Boundary

All kernels + the loop live in riir-infer-gpu (the inference substrate). The
serving-router integration (greedy→resident, sampled→incumbent) is the
consumer's (riir-ai engine) follow-up, filed there when the substrate lands.
