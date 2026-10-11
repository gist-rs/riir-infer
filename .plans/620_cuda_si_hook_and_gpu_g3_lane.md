# Plan 620: Plan 622 Phase 7 — the CUDA si hook + the GPU G3 lane

**Status:** Active — the survey is complete (this document IS the spec, anchored
file:line); the hook + GPU harness are the open work. The CPU G3 zero-shot run
is in flight on the 4090 (schtasks `si622run`, log `E:\tmp\si622-run.log`,
worktree `E:\git\wt-si622` @ the si lane commit) — its rows land regardless;
this plan is the FAST lane (GPU G3 ≈ 5 min/arm vs the CPU's ~2 h/arm) and the
G2 engine cells' vehicle.

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

- [ ] `cudarc_kernels/attention.rs`: extend `attention_decode_f32` (score site
      L443-446) with the si branch — new params: `lut` (global f32\*, the
      interleaved (a_t, m_t) pairs, ≤16.6 KB at block 2080), `inv_sigma`
      (global f32\*, one entry per q-head of THIS layer — the launch is
      per-layer, grid = one block per q-head), `si_enable` (i32 scalars via the
      params pattern). `dist = n_positions - 1 - pos` (pos is in-kernel, L435).
      Apply BEFORE the tile-max pass (L454) — the max must track the
      TRANSFORMED row (the online-softmax law; same as the CPU `si_head`).
- [ ] Twin the edit into `attention_decode_f32_devpos` (L618+) through the
      shared-body hatch — the devpos variant reads `n_positions` from
      `*pos_dev + 1`; everything else identical.
- [ ] Rust launchers: `launch_attention_decode` (ternary_deltanet_gpu_forward_cudarc.rs
      L4722-4734) + `launch_attention_decode_devpos` (L5439-5443) gain the
      three args, threaded from a new session field.
- [ ] Split/gqa/mma variants are NOT on this decode path — leave them
      untouched (record why: consumed by qwen38 f16-KV gates + prefill only).
- [ ] Bit-identity gate: a flag-0 A/B on the real PQ2_0 checkpoint — 64
      greedy tokens + a 128-token prefill, logits byte-equal vs the incumbent
      build (the G3 posture pin; run on the 4090, GPU-exclusive preflight).

## Phase B — session plumbing (`TernaryDeltanetGpuForwardCudarc`)

- [ ] `SiGpuSession { lut: CudaSlice<f32>, inv_sigma: CudaSlice<f32> per layer
      or one slice + offsets, enabled: bool }` behind the
      `scale_invariant_attn` feature on the gpu crate (new feature row
      forwarding `katgpt-core/scale_invariant_attn`; the deltanet stack is
      already in its dep tree).
- [ ] LUT upload: `ScaleInvariantLut::build(tau, block_size)` → the
      interleaved `am` pairs as one 16.6 KB `CudaSlice` (uploaded once per
      session; broadcast reads are LUT-local).
- [ ] inv_sigma per (layer, head): the calibration table slot =
      `layer_idx * n_head + head` (the CPU harness contract) → per-layer
      launches pass their own n_head-float window; `σ = 1.0` neutral slots are
      legal (GDN slots never launch attention).
- [ ] Arming/disarming = `enabled` flip + (if the LUT pointer is baked into a
      captured graph) the devpos pointee swap — NO recapture.
- [ ] The σ probe arm: CPU probe (the existing `run_sigma_probe` shape) →
      frozen `HeadCalibration` sidecar (BLAKE3-checked, `to_sidecar`/
      `from_sidecar` already shipped in katgpt-core) → GPU armed. Never
      per-head score downloads.

## Phase C — the GPU G3 harness (bin in riir-infer-gpu)

- [ ] `si_zero_shot_ppl_gpu` bin (the root `si_zero_shot_ppl.rs` port):
      same arms (base / si / si-cal / si-pprope), same 16 log-rungs 128→2048,
      same 128-token scored spans, same corpus + tokenizer paths — but the
      feed loop drives `forward_token_graph` (~90 tok/s ⇒ ~5 min/arm) and the
      pp-RoPE arm consumes `pp_rope_table_from_marks` (gguf_loader.rs:2167)
      — NOTE: the pp table must reach the GPU rope path; if the GPU forward
      reads rope from device tables, upload the masked table (masked lanes =
      exact identity by construction, Phase 4).
- [ ] Verdict printing identical to the CPU harness (pooled ppl, worst
      per-rung Δ, the pre-recorded pp-RoPE FAIL) — the two harnesses' rows
      must be directly comparable.
- [ ] Cross-host disclosure in the output: the calibration digest is
      per-host (AVX2 vs NEON dot products differ in the f32 LSBs — measured
      this session: M3 `1879…` vs 4090 `95ef…`, σ stats identical to 4
      decimals). Quote the digest + host beside every row.

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
riir-infer-gpu at the gpu-crate posture; the Phase-A bit-identity A/B on the
real checkpoint; the G3 GPU run's base arm pooled-ppl cross-checked against
the CPU run's base arm (same corpus, same rungs — the numbers must agree
within the f32-posture envelope; a divergence beyond that indicts the
instrument, not the checkpoint — Issue-022 T3.1 law).
