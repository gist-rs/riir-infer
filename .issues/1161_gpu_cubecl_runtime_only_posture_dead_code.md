# Issue 1161 — riir-infer-gpu: 7 dead-code warnings at the cubecl_runtime-only posture (reflex's laya-riir-cubecl composition)

**Status:** open — filed 2026-10-31 by the riir-refine idle clippy sweep (session zed-m3-idle-ev190); found via `riir-reflex` `--all-features` clippy (the only posture in the workspace that compiles this combo).

## The posture hole

`riir-reflex` feature `laya-riir-cubecl` pulls `riir-infer-laya/laya-riir-cubecl` → `riir-infer-gpu` with **only `cubecl_runtime`** enabled (riir-infer-laya Cargo.toml:63). At that posture the riir-infer-gpu lib compiles the gemv/norms/regime items below, but rustc reports them dead — while every riir-infer-own posture (default + the ternary feature stack) uses them, so this repo's own gates are green. The warnings also survive the consumer's `-D warnings` because `--` lint caps apply to workspace members, not path deps — reflex's guard cannot see them either. **No gate in the workspace covers this composition today.** (The reflex Cargo.toml's own comment notes the lane-611 wiring: "the metal/cuda lanes never pulled the gpu tree, which is why reflex never needed these before" — the cubecl lane is the first consumer of this shape.)

```
cargo clippy --all-targets --all-features   # in riir-reflex, 2026-10-31, macOS, post 6754a0f
warning: function `dense_pair_wg_env` is never used        gemv_cubecl.rs:601
warning: function `dense_pair_wg_size` is never used       gemv_cubecl.rs:612
warning: static `NORM_ROTATE_FUSED_REACHED` is never used  norms_cubecl.rs:1871
warning: function `norm_rotate_fused_mark_reached` is never used  norms_cubecl.rs:1908
warning: static `NORM_ROTATE_MULTI_REACHED` is never used  norms_cubecl.rs:1940
warning: function `norm_rotate_multi_mark_reached` is never used  norms_cubecl.rs:1979
warning: function `regime_capture_append` is never used    regime_gate.rs:789
```

## The puzzle for the owning lane (why this is filed, not blind-fixed)

The obvious consumers are NOT cfg'd out: e.g. `ternary_deltanet_gpu_forward.rs:5907/6010` call `norm_rotate_fused_mark_reached()` inside `forward_from_x_body_range`, which sits in the `#[cfg(feature = "cubecl_runtime")] impl TernaryDeltanetGpuForward` (line 3421) — compiled at this posture. Yet rustc reports the callees dead. Either (a) the live decode path at cubecl_runtime-only resolves through a SECOND cfg'd copy of the forward body that doesn't call these, or (b) the whole `forward_from_x_body_range` chain is dead from a cfg'd-out root and rustc reported only the leaf items (the roots' warnings may be suppressed by existing allows). Adjudicate which, then pick the repair:

1. **Gate the items to the features that arm their consumers** (the crate precedent: `#[cfg_attr(not(all(test, feature = "...")), allow(dead_code))]` — deltanet_rotation_cudarc.rs:1893), keyed on the actual consumer-arming feature set; or
2. **Gate the whole unused-at-this-posture surface** behind the ternary feature umbrella so the cubecl_runtime-only posture compiles none of it; or
3. If the reflex composition itself is wrong (cubecl_runtime-only is not a supported posture of riir-infer-gpu), fix the CONSUMER edge: `riir-infer-laya/laya-riir-cubecl` should forward the minimal feature set that keeps riir-infer-gpu's lib warning-free.

Whichever lands: the class generalizes — any future consumer pulling riir-infer-gpu with a subset feature set will re-meet it. A one-line check in the consumer lane (`riir-reflex` clippy at `laya-riir-cubecl` with dep warnings surfaced) would pin the posture; consider a `RUSTFLAGS="-D warnings"` probe or a workspace-member shim if it should stay covered permanently.

## Non-goals

No judgment on the reflex lane's own feature choices (lane 611's wiring is deliberate per the Cargo.toml record); the finding is only that the composed posture emits dead-code warnings nothing owns.
