# Plan 626 — GPU-resident greedy decode: the queue-never-drains token loop (Issue 1168 T3)

**Status:** T1/T1b measured (Issue 1168); T1/T2 VERIFIED in-tree 2026-11-03; T3–T5 implementation claimed.
**Lease closed:** (handed over unexpired to zed-5ae93349@amo — their T3/T3a/T4 implementation WIP (+683 lines: argmax_f32_last twin, resident ring + begin/enqueue/harvest, indirect dequant kernel, 2 G1 test files) was already mid-flight in this shared tree; session zed-dd837b96, 1793688133, expiry was 2026-11-05T06:42Z)
**Claimed:** zed-5ae93349@amo (amo), 1793688980, expires 2026-11-05T06:42Z.

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

- [-] T1: CONSUME `sampling_cubecl::argmax_f32` — **VERIFIED 2026-11-03 (this session): the tie rule is already pinned by the in-tree tests** (`test_argmax_basic` / `_single_element` / `_all_equal` — tie→first-index 0 — / `_large_vocab` 256128, all with the 0xDEAD_BEEF sentinel green-zero guard); 4/4 PASS on this box (`cargo test -p riir-infer-gpu --release --features cubecl_runtime --lib sampling`). Kernel reduction is strict-greater ascending in phase 1 + lower-paired-index in the smem tree — the CPU first-index rule holds by construction. REMAINING (T3's wiring): the id must land in a device-side buffer the next token's gather reads (the launcher writes a 1-u32 handle already — wire, don't rebuild).
- [-] T2: **DONE IN-TREE — verified 2026-11-03 (this session), no new kernel needed.** The GPU-side embedding lookup is ALREADY the shipping `set_input_token` path (`ternary_deltanet_gpu_forward.rs:11369`): `DequantWteRowCubeCL::launch` dequantizes the wte row from the ternary bit-planes + f32 scales STRAIGHT INTO `self.x` (Issue 604 T8 G4 — zero CPU alloc, zero GPU alloc per token; the wte ternary handle pre-uploads in `new()`), + the folded-rotation inverse step. There is no f16 table and no CPU row-upload in the decode path — plan 626's f16-gather arm is MOOT. The per-token CPU→GPU value surface is ONE u32 scalar (`token as u32`). Bit-identity oracle: the kernel is the incumbent (not a variant) — T4's greedy-walk EXACT covers it.
- [-] T3: the resident loop — narrowed by T2 to TWO pieces: **CODE LANDED 2026-11-03 (this session), compiles clean at the full Metal posture, clippy-clean; behavioral proof = T4**: (a) an INDIRECT variant of the dequant kernel (`dequant_wte_row_f32` reads `row_idx` from a device id-buffer instead of the launch scalar — the `dequant_wte_rows_f32` multi-row variant at :321 is the in-tree precedent for buffer-fed row indices, bit-identical per row), (b) the resident entry points: `resident_enqueue` (indirect-dequant + the 1127 shape-only dispatches + argmax→ping-pong id_buf) and `resident_harvest` (K-token fenced id readback for detokenization + drain checks). Ping-pong id buffers A/B (in-order queue orders argmax-write → next-dequant-read correctly); greedy-only contract (sampled paths keep the incumbent lane).
- [x] T3a (the zed-7228dbf2 tie-rule observation, ACCEPTED 2026-11-03) — **LANDED 2026-11-03 (this session): `ArgmaxLastCubeCL` + `argmax_f32_last`, 11/11 sampling tests green on Metal. TWO MEASURED CORRECTIONS to the premise below: (1) the butterfly trap — the 256-wide smem tree pairs (tid,tid+128)...(tid,tid+2),(tid,tid+1), so slot order at the final step is NOT index order (a right-side champion migrates into a left slot; [1,5,5,2] puts (5.0,idx2) into slot 0 at the step-2 pair (0,2)); a value-only >= merge therefore misorders ties (returned idx1 on that vector) — the twin's phase-2 merge is the FULL lexicographic comparison (greater value, OR equal value AND higher index; nested plain ifs, cubecl-safe), which is genuinely tree-order-independent. (2) the incumbent's first-index claim is itself tree-order-luck on arbitrary cross-thread ties ([1,5,5,2] returns 2 — neither first nor last; measured); its pinned tests cover the cases where the luck holds (all-equal -> 0, single-max). NOT changed (pinned semantics, 940 law) — but any consumer relying on first-index-on-arbitrary-ties from ArgmaxCubeCL should know. Both facts pinned in sampling_cubecl tests (`test_argmax_tie_rules_diverge_exactly_at_the_tie` adjacent-slots fixture = the clean rule divergence; `test_argmax_last_is_tree_order_independent_on_mid_vector_tie` = the butterfly trap, asserted). Original premise text: the resident lane's argmax must be a **host-rule twin** — the league harness argmax (`bench_831_decode_long`/`bench_809`) is `max_by(total_cmp)` = LAST-index-on-ties, while the in-tree `argmax_f32` pins FIRST-index (4/4 sentinel tests, untouched). At an exact-f32 tie the walks diverge and the G1 EXACT bar breaks. So T3 adds `argmax_f32_last` (or an armed variant): same reduction shape, merge rule = lexicographic max over (value, index) — prefer higher value, on equal value prefer HIGHER index (a total order, so any reduction tree reproduces max_by exactly). Known sub-case documented: total_cmp orders −0.0 < +0.0 (the GPU `==` merge treats them equal) — the tie fixture uses finite non-zero values; NaN logits are upstream-screened (finite.rs) and out of scope. T4's identity test carries the tie-forcing fixture: both rules on one synthetic tie vector assert the divergence is EXACTLY the tie token (kernel first-index vs host last-index) — the class is tested, never latent. Real-logits walks then agree because ties are measure-zero; degenerate ternary loops (repeated-token contexts) are the realistic tie source, so the fixture is not paranoia.
- [x] T4 (G1) — **GREEN 2026-11-03 (zed-5ae93349, amo Metal, 27B folded)**: 3 seeds x 24-token resident walk == incumbent walk token-id EXACT, reach delta == WALK asserted, reset_state disarm law loud-verified (`plan626_resident_greedy_g1`, 12.2 s). ONE REAL BUG found + fixed by the heavy e2e: the id ring's 4-byte slot stride put the argmax sub-handles at non-32-byte bind offsets — wgpu `min_storage_buffer_offset_alignment = 32` refuses the bind on the first unaligned slot (measured `Buffer offset 16774148 ... limit 32`; slot 0 worked, the first wrap died) — slots are 32 bytes wide now (RING_SLOT_WORDS = 8); the hermetic kernels could never catch it (offset-0 pool bases). This is exactly why the heavy arm exists.
- [-] T4 (G2) — the A/B INSTRUMENT landed (`plan626_resident_g2_ab`, ABBA-interleaved, reach-asserted, load-gated; verdict printed, never asserted). FIRST RUN (amo 11-03, quiet box, PROVISIONAL-POSTURE): incumbent 35.51 vs resident 35.56 (1024-block) / incumbent 36.49 vs resident 35.19 (native-block) — ratio 0.965-1.001, resident >= incumbent weakly, BUT the ABSOLUTE does not reproduce the P1 reference (both lanes ~2x the 17.79 census; block_size clamp ruled out: native reads the same). The G2 bar (<=15.8) is UNADJUDICATED — the cell's posture diverges from the bench_763 census posture somewhere (candidates: stream/task caps, engine-level path vs substrate entry, feature arms, harvest cadence k). NEXT: replicate the exact census posture (or drive bench_763 itself through the resident seam), THEN the league re-cell (bench_768/831 tg/ctx) — the consumer-side step that needs the serving router. A G2 number without the posture match is not a measurement (the box-state law, one layer up).
- [ ] T5: promotion verdict — GOAT bars + reviewer/owner verdict; demote check = env literal restores the incumbent byte-identical path.

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
