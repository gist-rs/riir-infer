# Plan 620: Plan 622 Phase 7 — the CUDA si hook + the GPU G3 lane

**Status:** Active — Phases A–C code LANDED 2026-10-11 (kernel hook + both
twins via one shared helper, session plumbing + the no-recapture arm law,
the GPU harness bin); **Phase A bit-identity gate PASSED 2026-10-12**
(`.benchmarks/872`, three parts green — flag-0 byte-identical on the real
checkpoint); **Phase D LANDED 2026-10-11** (61eb815, the FA prefill hook —
see its checkboxes); the retrieval-lane residue CLOSED (0929fcb) and the
si-posture gate fix (9e3c9e1) landed 2026-10-12. OPEN: the G3 GPU run
(base-arm cross-check vs bench 936's CPU base arm — the GPU is idle, the
harness is ready incl. the retrieval lane) + the G2 pp2048/tg128 cells,
then plan 622's League rows + the owner-gated promotion decision.

Source survey (this session, read-only): the per-token CUDA decode for the
Bonsai-2 hybrid EXISTS — `TernaryDeltanetGpuForwardCudarc`
(`crates/riir-infer-gpu/src/ternary_deltanet_gpu_forward_cudarc.rs`), entry
`forward_token_graph` (L2416, full logits/token, ~90 tok/s measured class) and
`forward_token` eager (L1594). The league lane (riir-gpu `bench_734` tg128)
path-deps this crate and drives the SAME kernels — one hook covers both.
riir-engine is a re-export façade (no CUDA lane of its own; `channel_aware.rs`
is katgpt-rs CPU x86_64) — corrected premises from the survey.

## Non-negotiable laws

- **Flag-off = bit-identical** (the whole Plan 622 posture law): the si branch
  enters the kernel behind a runtime flag (`si_enable` scalar); a flag-0 launch
  must reproduce the incumbent kernel's bytes (G3-pinned before anything lands).
- **The transform is the CPU reference's expression order**:
  `score = (pos == 0) ? raw : a[dist] * (raw * inv_sigma) + m[dist]` with
  `dist = query_pos − pos`, `query_pos = n_positions − 1` (decode). The
  `si_head` reference is `riir-infer/src/transformer/attention_si.rs:128`.
- **Sink carve-out** (paper App. J): key 0 keeps its raw logit on BOTH the
  armed and probe postures; the σ probe never observes key 0.
- **Graph mode = no recapture**: session state rides DEVICE POINTERS (the
  `devpos` pattern, attention.rs L547-556) — arming/disarming between graph
  replays swaps pointee CONTENT, never the graph.
- **One source, two builds**: `attention_decode_f32` and `_devpos` are twins
  ("Body is IDENTICAL", attention.rs L618-628) — edit via the established
  `#define` hatch pattern (the Issue-753 `KV_F16` hatch, attention.rs L69-79),
  never a copy-paste pair.
- **Boundary**: riir-infer has zero riir-* deps; `riir-infer-gpu` already deps
  `katgpt-core` (its Cargo.toml L34-35) — the LUT/calibration substrate
  (`katgpt_core::scale_invariant`) is directly consumable. The G3 GPU harness
  is a bin INSIDE riir-infer-gpu (the root crate cannot dep the GPU crate —
  the `multicol_ab` example shape).
- **Coordinate with the 4090 agent**: the gpu crate carries their in-flight
  WIP (edlm_cubecl / multicol / dq lanes). Stage from a clean worktree on the
  4090 (`git worktree add`), never their dirty main checkout; disjoint files
  (this plan touches `cudarc_kernels/attention.rs` +
  `ternary_deltanet_gpu_forward_cudarc.rs` + the gpu Cargo.toml + one new bin).

## Phase A — the decode-kernel hook

- [x] `cudarc_kernels/attention.rs`: extend `attention_decode_f32` (score site
      L443-446) with the si branch — new params: `lut` (global f32\*, the
      interleaved (a_t, m_t) pairs, ≤16.6 KB at block 2080), `inv_sigma`
      (global f32\*, one entry per q-head of THIS layer — the launch is
      per-layer, grid = one block per q-head), `si_enable` (i32 scalars via the
      params pattern). `dist = n_positions - 1 - pos` (pos is in-kernel, L435).
      Apply BEFORE the tile-max pass (L454) — the max must track the
      TRANSFORMED row (the online-softmax law; same as the CPU `si_head`).
      LANDED as one shared `si_transform_score` device helper serving BOTH
      twins (the transform exists exactly once); `si_enabled` is a DEVICE
      POINTER (not a scalar) so a captured graph arms by CONTENT — and the
      expression order is pinned with `__fmul_rn`/`__fadd_rn` (each Rust op
      rounds once; nvrtc FMA contraction would skip the product's rounding).
- [x] Twin the edit into `attention_decode_f32_devpos` (L618+) through the
      shared-body hatch — the devpos variant reads `n_positions` from
      `*pos_dev + 1`; everything else identical.
- [x] Rust launchers: `launch_attention_decode` + `launch_attention_decode_devpos`
      gain one `si: Option<&SiDecodeLayerArgs>` arg; `None` pushes the
      struct-owned dummies (`si_dummy_lut = [1.0, 0.0]`, a 64-float neutral
      inv_sigma runway never read while armed = 0, `si_dummy_enabled = [0]`)
      — the kernel branch is dead, bit-identical incumbent. Callers threaded:
      the hybrid forward (eager + devpos + capture/patch/profiled lanes), the
      Qwen3.8 dense forward passes `None` (the si lane is the Bonsai hybrid
      decode), and the TRAINING forward passes `None` (trained-window
      forwards must reproduce the backbone the gradients were built on).
- [x] Split/gqa/mma variants are NOT on this decode path — leave them
      untouched (record why: consumed by qwen38 f16-KV gates + prefill only).
- [x] Bit-identity gate: a flag-0 A/B on the real PQ2_0 checkpoint — 64
      greedy tokens + a 128-token prefill, logits byte-equal vs the incumbent
      build (the G3 posture pin; run on the 4090, GPU-exclusive preflight).
      INSTRUMENT READY: `si_flag0_probe` (the recording bin) + the in-harness
      `--dump-logits`; three-part procedure: (1) `si_flag0_probe` stdout
      byte-compare pre-hook vs post-hook builds, (2) the in-build unit gate
      `test_attention_decode_si_flag_off_bit_identity_and_affine` (bit-exact
      disarmed-vs-dummy + identity-LUT wiring + the affine value check),
      (3) the existing matches_cpu tests' max_diff equality across builds.
      **GATE PASSED 2026-10-12 (`.benchmarks/872`)** — all three parts: (1)
      probe stdout BYTE-identical (fc /b clean, MD5 a1f7caa5… both sides,
      PRE = 4801622 + probe-file overlay, POST = aeae7d2); (2) the unit gate
      green at POST (affine check 1.7695e-8, one-ULP class); (3) the decode
      twins' max_diff values IDENTICAL across builds (2.980232e-8 / 5.960464e-8
      / 0.0). Two probe fixes landed en route: the feature set widened to
      `riir-infer-core/bonsai2_hadamard` (the loader loud-refuses the
      Hadamard-folded league GGUF without it — same set both sides,
      comparability preserved) and the cache-window clamp (aeae7d2: the
      262144-block workspace OOMed a 24 GB card; the harness's
      row_logit_floor shape, math-neutral inside the window, applied
      identically at both commits). Schtask `si620flag0`, log
      `E:\tmp\si620-flag0.log`.

## Phase B — session plumbing (`TernaryDeltanetGpuForwardCudarc`)

- [x] `SiGpuSession` (`si_gpu.rs`) behind the `scale_invariant_attn` feature
      on the gpu crate (new feature row forwarding
      `katgpt-core/scale_invariant_attn`; implies `ternary_gemv_cuda_raw` —
      the cudarc family it lives in).
- [x] LUT upload: `ScaleInvariantLut::build(tau, block_size)` → the
      interleaved `am` pairs as one 16.6 KB `CudaSlice` (uploaded once per
      session via the public `pair()` accessor — no katgpt-core
      reach-through; broadcast reads are LUT-local).
- [x] inv_sigma per (layer, head): the calibration table slot =
      `layer_idx * n_head + head` (the CPU harness contract) → per-layer
      launches pass their own n_head-float window (`SiDecodeArgs::window_for`
      — an Arc-backed `CudaView`, no alloc); `σ = 1.0` neutral slots are
      legal (GDN slots never launch attention).
- [x] Arming/disarming = `enabled` CONTENT flip (`arm_si`) — NO recapture;
      attach/detach change the baked pointers and invalidate the graph (the
      `set_lora` precedent). The `si_session_args!` macro keeps the 7 forward
      call sites one-line under both feature postures (the borrow is scoped
      to the field, not `&self` — the split-borrow law).
- [x] The σ probe arm: CPU probe (the existing `run_sigma_probe` shape) →
      frozen `HeadCalibration` sidecar (BLAKE3-checked, `to_sidecar`/
      `from_sidecar` already shipped in katgpt-core) → GPU armed. The CPU bin
      gained `--save-cal`/`--probe-only` (the sidecar producer); the GPU
      harness consumes it (`--cal-sidecar`) and quotes digest + host beside
      every row. Never per-head score downloads.

## Phase C — the GPU G3 harness (bin in riir-infer-gpu)

- [x] `si_zero_shot_ppl_gpu` bin (the root `si_zero_shot_ppl.rs` port):
      same arms (base / si / si-cal / si-pprope), same 16 log-rungs 128→2048,
      same 128-token scored spans, same corpus + tokenizer paths — but the
      feed loop drives `forward_token_graph` (~90 tok/s ⇒ ~5 min/arm).
      ⚠ DIVERGENCE (measured premise): the pp-RoPE arm SKIPs on the GPU lane
      — the rope kernels compute `inv_freq` INLINE from `theta_base`; there
      is no device rope table for the pp mask to ride (the plan's "if the GPU
      forward reads rope from device tables" condition is FALSE). The CPU
      run's pre-recorded expected-FAIL row stands as the demonstration; a GPU
      pp lane would need a masked-table rope kernel — its own slice if ever
      needed. RESIDUE CLOSED (0929fcb, 2026-10-12): the `--retrieval` needle
      probe is ported — second pass after the G3 table, the CPU printer's row
      shape, arm-major grouping (documented divergence), first live exercise
      rides the GPU G3 run.
      ⚠ FIRST RUN BLOCKED (2026-10-12, schtask `si620g3gpu`): the sidecar
      stage is SOLVED (the σ probe runs only under `--arms si-cal` — main.rs
      L456 `arms.contains(&Arm::SiCal)`; `--probe-only` then stops before the
      arms; the on-box sidecar digest `95ef…` matches the CPU run's 4090
      digest exactly — deterministic cross-run per host) and the GPU forward
      boots (16.6s) + the LUT + sidecar load — then the FIRST arm's first
      `forward_token_graph` dies `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
      operation not permitted when stream is capturing` (the log holds no
      `# arm base done` — the base arm's first capture already fires it).
      Suspect NARROWED (code read, this session): the capture window is
      `ensure_graph_captured` L2591-2639 — begin_capture(GLOBAL) →
      `si_session_args!` (borrow-only) → `forward_from_x_devpos` →
      per-attention-layer `forward_attention_layer_devpos` → `window_for`
      (CudaView borrow, documented no-alloc) → the launcher (None arm borrows
      struct-owned dummies — no per-call alloc). The dq fake-quant block is
      cfg'd out at this posture. The "operation not permitted when stream is
      capturing" class = an op illegal mid-capture: sync memcpy, allocation,
      or LAZY MODULE LOAD. **Working hypothesis: the lazy-load class** — the
      G3 harness constructs the forward via `TernaryDeltanetGpuForwardCudarc::new()`
      while the GREEN-on-this-box flag0 probe uses `new_graph_ready()` (the
      name says it: everything primed); `ensure_graph_captured` may be the
      first-ever exercise of `new()` + capture at this kernel set, and a
      first-use module load (or other lazy init) inside the window kills it.
      NEXT SLICE: (a) switch the harness to `new_graph_ready` and re-run the
      schtask (bins + sidecar `E:\tmp\si622-cal-4090.bin` all warm — the
      reproduction is one command), (b) if that fixes it, decide whether
      `ensure_graph_captured` should warm/prime defensively for every `new()`
      consumer, (c) then the G3 rows land (base-arm cross-check vs bench
      936's base pooled 7.2449 within the f32-posture envelope).
      ⚠ The bin requires ONLY `cuda_graphs_forward` + the folded loader — NOT
      `scale_invariant_attn` — so the `base` arm (and `--dump-logits`)
      runs on an UNHOOKED build: the cross-build byte-compare is executable
      without cherry-picks.
- [x] Verdict printing identical to the CPU harness (pooled ppl, worst
      per-rung Δ, the tolerance verdict) — the two harnesses' rows are
      directly comparable.
- [x] Cross-host disclosure in the output: the calibration digest is
      per-host (AVX2 vs NEON dot products differ in the f32 LSBs — measured
      this session: M3 `1879…` vs 4090 `95ef…`, σ stats identical to 4
      decimals). The harness prints the sidecar digest + the host-frozen
      note; quote both beside every row.

## Phase D — the G2 prefill cell (SEPARATE unit, lands after A–C)

- [ ] `att_pf_fa_body` (prefill_cuda_attention_fa.rs:409) — the si affine
      beside the causal-mask loop (L536-542, which already derives per-element
      key pos `kv` and query pos in-register); σ_h from the head lane of the
      documented fragment algebra. Scale is folded into Q at staging (L464-466)
      → fragments are post-scale logits = the `logit_scale = 1.0` entry space.
- [ ] Scope: the FA-engaged default posture only (p ≥ 128, `fa_engaged()`);
      the kill-switch fallback arms stay unhooked + bit-identical-off, noted.
- [ ] Then the G2 cells: pp2048/tg128 ±3% (si-ON vs flag-off) via the league
      harnesses (`bench_734`/`bench_772`) on the 4090, GPU-exclusive, with
      box-state provenance per the league law.

## Explicitly out of scope (here)

- Metal (`riir-engine` league path on the M3): the Metal/CUDA parity leg of
  G2 needs the SAME hook in the engine's Metal attention — its own unit,
  after D's CUDA numbers exist to be paritied against.
- Training-track pp-RoPE quality → riir-train Plan 449 (unchanged).
- The CPU G3 run's rows (in flight on the 4090) — they land independently;
  this plan's GPU re-run is the FAST re-gate lane + the τ-sweep enabler
  (Phase 5's App-H driver needs 3 more full runs — infeasible on CPU).

## Validation bar (before any row is published)

`cargo clippy --all-targets --features scale_invariant_attn -D warnings` on
riir-infer-gpu at the gpu-crate posture — LANDED 2026-10-11 at THREE postures
on the M3 via the x86_64-pc-windows-gnu typecheck lane (the cudarc family is
`not(macos)`-gated; no linking needed): default-off, `cuda_graphs_forward`
(+bonsai2_hadamard — the harness posture), and the full
`cuda_graphs_forward,scale_invariant_attn` si posture, `--lib --bins
--tests`, all green. The CPU-side `si_zero_shot_ppl` bin green at
`scale_invariant_attn,bonsai2_hadamard`. ~~REMAINING: the Phase-A bit-identity
A/B on the real checkpoint + the G3 GPU run's base-arm cross-check against
the CPU base arm~~ **BOTH DONE: the flag-0 A/B three-part GREEN (bench 872,
2026-10-12) and the G3 GPU run + base-arm cross-check PASS at 0.07% (bench
873, 2026-10-13 — base pooled 7.2499 GPU vs 7.2449 CPU; si/si-cal FAILs
reproduce; no promotion, two-lane-consistent).** Box work remaining: Phase D
(G2 prefill cells) only.

## The harness constructor law (bench 873, 2026-10-13)

**A graph-lane consumer constructs via `new_graph_ready` — `new()` is the
eager reference lane, and its cudarc event tracking (default ON) cannot cross
a capture window** (per-arg wait/record event pairs →
STREAM_CAPTURE_UNSUPPORTED on the first captured launch). The G3 harness's
`new()` cost the first GPU attempt (b51769c's "si-gated" attribution was this
delta wearing an si costume — flag0's probe used `new_graph_ready` and
captured the same body green); fixed in the same breath as bench 873's run.

## Phase-A bit-identity protocol (the recording instrument, landed 4801622)

The A/B instrument is committed: **`si_flag0_probe`** (bin,
`ternary_gemv_cuda_raw` + `cuda_graphs_forward` required-features, no si
surface — it compiles unchanged at any commit). The gate it serves:

1. **Pre-hook recording** — at a PRE-hook sha (e.g. `4801622` itself; the
   window never closes — a worktree reproduces the incumbent build at any
   time, the hook landing on develop does not consume it):

   ```sh
   git -C E:/git/riir-infer worktree add E:/git/wt-si620pre 4801622
   cd E:/git/wt-si620pre && CARGO_TARGET_DIR=E:/tmp/tgt-si620pre \
     cargo build --release -p riir-infer-gpu \
     --features ternary_gemv_cuda_raw,cuda_graphs_forward --bin si_flag0_probe
   E:/tmp/tgt-si620pre/release/si_flag0_probe.exe --gguf <league gguf> \
     > E:/tmp/si620-flag0-prehook.txt 2>&1
   ```

2. **Post-hook recording** — the SAME commands at the hook commit (a second
   worktree or the same one re-checked out; separate CARGO_TARGET_DIR per
   checkout) → `E:/tmp/si620-flag0-posthook.txt`.

3. **The gate: `diff` the two stdout streams — must be BYTE-identical**
   (same box, same env, no knobs touched). Every line is load-bearing:
   the eager prefill digests @16/@128, the eager greedy ids + rolled digest,
   the graph-lane twins (the devpos kernels the G3 harness drives), and the
   `eager==graph ids:` line (its pre/post VALUE must match too, whatever it
   records). A divergence in ANY line = the hook broke flag-0 — fix before
   landing, never after.

Notes: the two runs are correctness recordings — GPU-exclusivity is not
load-bearing for the bytes, but the box-state law prefers a quiet box; the
build competes for CPU with anything else on the box (bounded `-j` when the
4090 carries the CPU G3 run). The gguf path is printed in the output —
record which checkpoint the digests pin.
