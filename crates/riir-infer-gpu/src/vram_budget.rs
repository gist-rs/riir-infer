//! Issue 994 — refuse-to-run VRAM budget for the GPU-resident ternary
//! forward (the silent-corruption hardening).
//!
//! ## The hazard (measured, riir-train Bench 600)
//!
//! On the 24 GB 4090 at 64K-class `block_size`, `TernaryDeltanetGpuForward`'s
//! constructor allocations (~15.8 GB: weights ~7.2 GB, attention KV ~8.6 GB,
//! plus working buffers) exhaust the cubecl-wgpu memory pool at the WDDM margin.
//! The Issue 714 vendor patch keeps the server alive through a failed reserve
//! (correct for multi-test processes) — but the failure then rides the
//! per-stream error sink, which nothing drains during construction. The
//! forward completes on the main thread with **self-consistent, silently
//! wrong activations**: bit-identical across two in-process prefills,
//! catastrophically wrong vs an external reference (cos 0.18–0.42). Only the
//! ladder anchor-ref gate (riir-train Issue 452 T3) caught it.
//!
//! ## The hardening — three independent prongs
//!
//! 1. **Pre-flight capacity estimate** ([`budget_verdict`] /
//!    [`check_forward_budget`]): compute the constructor working set from
//!    `Config` + the weight slabs and REFUSE (before any allocation) when it
//!    exceeds the adapter budget minus a margin. Catches gross over-commits
//!    (e.g. `block_size` 262,144 → KV alone ~34 GB) in microseconds instead
//!    of after partial allocation.
//! 2. **Pre-flight single-buffer KV guard** (same entry point): the
//!    per-layer attention KV cache is the only constructor buffer that grows
//!    with `block_size`, and the ≥52K-position class is broken on the
//!    cubecl/wgpu stack — bisected on the 4090 (block 49,216: BIT-EXACT
//!    clean; block 57,408: device lost during construction; block 65,472:
//!    DETERMINISTIC silent corruption — cos 0.419531 / top-1 0.0150,
//!    reproduced with ZERO pool reserve failures, refuting pool exhaustion
//!    as the mechanism; block 65,600: same corruption + reserve failures).
//!    The gate refuses when the per-layer KV bytes exceed the bisected
//!    default (208 MiB ⇒ ~53,248 positions at kvd 1024), between the
//!    measured-clean ceiling and the measured-broken floor. Env override:
//!    `RIIR_GPU_KV_LAYER_MAX_BYTES`.
//! 3. **Construction flush-gate** ([`drain_construction_errors`]): at the end
//!    of construction, `client.flush()` drains the per-stream error sink and
//!    returns `Err` when ANY construction-time allocation/write/launch failed
//!    (the Issue 714 non-fatal-OOM routing). The constructor turns that into
//!    a loud panic — no silently-corrupted forward can leave construction
//!    through the OOM class.
//!
//! ## Budget sources, in precedence order
//!
//! 1. `RIIR_GPU_VRAM_BUDGET_BYTES` — explicit override (set when the query is
//!    unavailable or the box shares the GPU with a known baseline).
//! 2. Adapter query: vendored `wgpu-hal` `Adapter::total_video_memory_bytes()`
//!    (DXGI `DedicatedVideoMemory` on DX12, device-local heaps on Vulkan,
//!    `recommendedMaxWorkingSetSize` on Metal), probed per backend from the
//!    wgpu adapter `CubeCLContext` retained at init (Issue 657's `init_setup`).
//! 3. Unknown (`None`) → pre-flight skipped (logged once); the flush-gate
//!    still protects.
//!
//! Margin: `RIIR_GPU_VRAM_MARGIN_BYTES` overrides; default
//! `max(budget / 4, 2 GiB)`. The measured 64K case sat at ~70% of the usable
//! ceiling and still thrashed — the margin exists to refuse "near the
//! ceiling", not only "over" it; margin-thrash inside the margin band remains
//! the flush-gate's job (see Bench 600's arithmetic).
//!
//! Kill switch: `RIIR_GPU_VRAM_CHECK=0` disables the pre-flight (the
//! repo-wide inverted-kill-switch convention). The flush-gate is NOT
//! switchable — it is correctness, not tuning. Default on.
//!
//! ## Vendor mechanism constraints (2026-09-22, CPU-only analysis — inputs for
//! whoever chases the zero-failure corruption mode; tracked at riir-train
//! Issue 452 T4/T5, whose Bench 600 addendum 2 owns the phenomenology)
//!
//! Facts read off the vendored stack that any mechanism theory must respect:
//!
//! 1. **wgpu's default size limits are NOT in play.** cubecl-wgpu's
//!    `request_device` requests `adapter.limits()` (vendor
//!    `cubecl-wgpu-0.11.0-pre.2/src/backend/wgsl.rs:35-42`), not
//!    `Limits::default()` — so the 256 MiB `max_buffer_size` / 128 MiB
//!    binding defaults never fire, and a per-buffer size-validation failure
//!    is ruled out at these KV sizes (192–256 MiB).
//! 2. **The memory-pool ladder geometry is derived from the binding limit.**
//!    cubecl-runtime builds `MemoryDeviceProperties.max_page_size` from
//!    `device.limits().max_storage_buffer_binding_size`
//!    (`cubecl-wgpu …/src/runtime.rs:306`), and the default SubSlices pool
//!    ladder divides that by 4 repeatedly down to 32 MiB
//!    (`cubecl-runtime …/memory_management/memory_manage.rs` `build_pools`),
//!    with `max_slice_size = page / 2^base`. A ~200 MiB KV cache buffer
//!    therefore lands in a sliced page whose co-tenancy (how many 192.2 vs
//!    224.25 vs 255.83 MiB slices share one page) changes across the bisect
//!    gap — a concrete, checkable geometric suspect for the deterministic
//!    corruption mode.
//! 3. **On Vulkan + 64-bit shader indexing the binding limit is `u64::MAX`**
//!    (`cubecl-wgpu …/src/backend/vulkan.rs:113`), degenerating that ladder
//!    (max_page/4^k starting at ~2^64); on a ~4 GiB-class limit the ladder
//!    yields 1 GiB / 256 MiB / 64 MiB pages. Which of the two ladders the
//!    bisect box actually ran is UNMEASURED — printing `device.limits()` +
//!    the resolved pool layout at the three bisect block sizes is the
//!    cheapest discriminating experiment (E1). E2: re-run block 65,472
//!    under an explicit ExclusivePages / single-big-pool config (the
//!    `configure_memory_pools` seam) — corruption vanishing there would
//!    localize the mechanism to sliced-page co-tenancy rather than the
//!    driver.
//!
//! Note also: the failed-reserve line Bench 600 addendum 2 captured
//! (`285212672` bytes inside a `1582170112`-byte page) is consistent with a
//! `SlicedPages` sub-allocation, i.e. the overt-OOM mode also routes through
//! this ladder.

use crate::cubecl_runtime::CubeCLContext;
use riir_infer_core::deltanet::ternary_weights::{GateProjWeights, QwenDeltaNetTernaryWeights};
use riir_infer_core::types::{Config, DeltaNetLayerType};

/// Kill switch for the pre-flight check (the flush-gate is NOT switchable —
/// it is the correctness backstop). `=0` disables, per the repo-wide
/// inverted-kill-switch convention (`RIIR_PREFILL_REC_MR=0` et al.).
pub const ENV_CHECK_DISABLE: &str = "RIIR_GPU_VRAM_CHECK";
/// Explicit budget override, in bytes.
pub const ENV_BUDGET_BYTES: &str = "RIIR_GPU_VRAM_BUDGET_BYTES";
/// Explicit margin override, in bytes.
pub const ENV_MARGIN_BYTES: &str = "RIIR_GPU_VRAM_MARGIN_BYTES";

/// Default margin floor: the pool-page/driver/GUI-baseline slack plus the
/// feature-dependent weight-side duplicates (concat staging buffers, spec
/// pools) that the estimator deliberately does not itemize.
pub const DEFAULT_MARGIN_FLOOR_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Default per-layer KV-cache refusal bound (bytes). Bisected on the
/// 4090/wgpu-spirv stack (Issue 994, 2026-09-22): block 49,216 (192.2 MiB)
/// measured BIT-EXACT clean; block 57,408 (224.5 MiB) lost the device during
/// construction; block 65,472 (255.8 MiB) silently corrupted (cos 0.419531 /
/// top-1 0.0150, deterministic, zero pool failures). The default sits between
/// the measured-clean ceiling and the measured-broken floor — 208 MiB ⇒
/// ~53,248 positions at kvd 1024. Override with `RIIR_GPU_KV_LAYER_MAX_BYTES`
/// (raise ONLY with a fresh bisect + external-reference validation on the
/// target stack).
pub const DEFAULT_KV_LAYER_MAX_BYTES: u64 = 208 * 1024 * 1024;

/// Env override for the per-layer KV guard, in bytes.
pub const ENV_KV_LAYER_MAX_BYTES: &str = "RIIR_GPU_KV_LAYER_MAX_BYTES";

/// The per-layer attention KV cache bytes for a config (kvd = `n_kv_head` ×
/// `head_dim`, f32, one cache).
pub fn kv_layer_bytes(config: &Config) -> u64 {
    let kvd = config.n_kv_head as u64 * config.head_dim as u64;
    config.block_size as u64 * kvd * 4
}

// ── Issue 994 / riir-train Issue 452 T4 — the SubSlices pool-ladder derivation
//    (experiment E1: print device limits + the resolved pool layout; see the
//    vendor-constraints notes in this module's header). Replicates
//    `vendor/cubecl-runtime-0.11.0-pre.2/src/memory_management/memory_manage.rs`
//    `build_pools`'s SubSlices arm and `SlicedPool::accept`'s first-fit order,
//    so a size can be mapped to the page class it will slice — WITHOUT
//    initializing the pools (CPU-only arithmetic over the adapter limits
//    cubecl-wgpu derives `MemoryDeviceProperties` from:
//    `max_page_size = max_storage_buffer_binding_size`,
//    `alignment = min_uniform_buffer_offset_alignment`).

/// One rung of the `SubSlices` pool ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolRung {
    /// The pool's page size in bytes.
    pub page_size: u64,
    /// The pool's `max_slice_size` — the first-fit selector.
    pub max_slice_size: u64,
}

/// The dynamic `SubSlices` pools `build_pools` creates, in the order
/// `reserve` scans them (ascending page size, then the binding-limit-sized
/// terminal pool). The tiny `ExclusivePages` pool for sub-alignment
/// allocations is omitted (it accepts only size 0).
///
/// Loop shape (verbatim from the vendored source): start at `max_page`, and
/// while the PREVIOUS page is ≥ 32 MiB, quarter it, align UP to
/// `alignment`, and give it `max_slice_size = page / 2^base` with `base`
/// counting from 1 — so the final iteration may dip below 32 MiB.
pub fn subslices_ladder(max_page: u64, alignment: u64) -> Vec<PoolRung> {
    let mut max_sizes = Vec::new();
    let mut page_sizes = Vec::new();
    let mut current = max_page;
    let mut base: u32 = 1;
    while current >= 32 * 1024 * 1024 {
        current /= 4;
        current = current.next_multiple_of(alignment.max(1));
        max_sizes.push(current / u64::from(2u32.pow(base)));
        page_sizes.push(current);
        base += 1;
    }
    max_sizes.reverse();
    page_sizes.reverse();
    let mut rungs: Vec<PoolRung> = page_sizes
        .into_iter()
        .zip(max_sizes)
        .map(|(page_size, max_slice_size)| PoolRung {
            page_size,
            max_slice_size,
        })
        .collect();
    let terminal = max_page / alignment.max(1) * alignment.max(1);
    rungs.push(PoolRung {
        page_size: terminal,
        max_slice_size: terminal,
    });
    rungs
}

/// Which ladder rung an allocation of `size` bytes slices, replicating
/// `reserve`'s first-fit over `SlicedPool::accept`: the first rung whose
/// `max_slice_size` fits, or whose page is within 20% of the size
/// (unbounded pools accept near-page strays). `None` = no rung fits
/// (`BufferTooBig`).
pub fn select_pool_rung(rungs: &[PoolRung], size: u64) -> Option<usize> {
    rungs.iter().position(|r| {
        r.max_slice_size >= size
            || match r.page_size.checked_sub(size) {
                Some(diff) => diff * 5 < r.page_size,
                None => false,
            }
    })
}

/// How many slices of `size` bytes co-tenant one page of `rung` (the
/// geometry the sliced pool carves when a rung serves same-sized buffers —
/// the KV caches at a given block size).
pub fn slices_per_page(rung: PoolRung, size: u64) -> u64 {
    rung.page_size / size.max(1)
}

/// GPU bytes a [`TernaryHandle`](crate::gemv_ternary_cubecl::TernaryHandle)
/// upload of `w` occupies: both u32-cast bit planes (same byte count as the
/// u64 sources), the f32-decoded group scales (2× the f16 source), and the
/// raw f16 scales (resident by default since Bench 771; kill-switch builds
/// skip 2 bytes/scale — inside the margin).
fn ternary_handle_bytes(w: &katgpt_core::TernaryGroupWeights) -> u64 {
    let bits = (w.pos_bits.len() + w.neg_bits.len()) as u64 * 8;
    let scales = w.group_scale.len() as u64 * (4 + 2);
    bits + scales
}

fn gate_proj_bytes(g: &GateProjWeights) -> u64 {
    match g.as_ternary() {
        Some(t) => ternary_handle_bytes(t),
        // Dense a/b upload as f32 (Plan 602 B2).
        None => g.as_dense().map_or(0, |(data, _, _)| data.len() as u64 * 4),
    }
}

/// Exact GPU bytes of the weight upload side of the constructor (every
/// ternary projection once, the actual gate-proj variant, dense f32 slabs).
///
/// Deliberately NOT itemized (inside the margin): feature-gated duplicates of
/// the input projections (the Issue 642 F3 concat staging copy, empty
/// `TernaryHandle` dummies), the tiny per-layer scalars (`a_log/dt_bias`,
/// < 1 MB total on Bonsai-27B), and the Metal/rs mirror caches (macOS-only,
/// opt-in at construction).
pub fn ternary_weights_gpu_bytes(weights: &QwenDeltaNetTernaryWeights) -> u64 {
    let mut total = ternary_handle_bytes(&weights.wte) + ternary_handle_bytes(&weights.lm_head);
    total += weights.final_norm.len() as u64 * 4;
    for l in &weights.layers {
        for p in l.projections() {
            total += ternary_handle_bytes(p);
        }
        for g in l.gate_projections() {
            total += gate_proj_bytes(g);
        }
        total += l.attn_q_norm.len() as u64 * 4;
        total += l.attn_k_norm.len() as u64 * 4;
        total += l.conv1d_weight.len() as u64 * 4;
        total += l.linear_norm.len() as u64 * 4;
        total += l.input_norm.len() as u64 * 4;
        total += l.post_attn_norm.len() as u64 * 4;
    }
    total
}

/// The constructor's persistent-activation working set, mirroring the
/// allocation arithmetic of `TernaryDeltanetGpuForward::new_with_rotation_policy`
/// (f32 element counts × 4 bytes).
pub fn forward_activation_bytes(config: &Config, layer_types: &[DeltaNetLayerType]) -> u64 {
    const F32: u64 = 4;
    let n = config.n_embd as u64;
    let n_v_heads = config.deltanet_linear_n_value_heads as u64;
    let head_dim = config.deltanet_linear_head_dim as u64;
    let n_k_heads = config.deltanet_linear_n_heads as u64;
    let q_dim = n_k_heads * head_dim;
    let v_dim = n_v_heads * head_dim;
    let qkv_dim = q_dim + q_dim + v_dim; // q + k + v (k_dim == q_dim here)
    let z_dim = v_dim;
    let conv_dim = qkv_dim;
    let kernel_size = config.deltanet_conv_kernel_size as u64;
    let mlp = config.mlp_hidden as u64;
    let vocab = config.vocab_size as u64;

    let mut total = 0u64;
    // Scalar activation buffers.
    total += 4 * n; // x, norm_x, tmp, ffn_out
    total += qkv_dim;
    total += 3 * n_v_heads * head_dim; // qkv_expanded
    total += z_dim;
    total += qkv_dim + z_dim + 2 * n_v_heads; // input_proj_out (Issue 642 F3)
    total += 4 * n_v_heads; // a_raw, b_raw, beta, decay
    total += n_v_heads * head_dim; // recurrent_out
    total += n; // rot_scratch (Plan 602 B3)
    total += n_v_heads * head_dim; // permute_tmp
    total += 3 * mlp; // ffn_gate, ffn_up, ffn_hidden
    total += 2 * mlp; // ffn_gate_up (Issue 642 F2)
    total += vocab; // logits

    // Attention-side buffers.
    let q_dim_attn = config.n_head as u64 * config.head_dim as u64;
    let kvd_attn = config.n_kv_head as u64 * config.head_dim as u64;
    total += 2 * q_dim_attn; // attn_qg
    total += 3 * q_dim_attn; // attn_q, attn_gate, attn_out
    total += 2 * kvd_attn; // attn_kv (Issue 648 F9 fused)
    let attn_split_max = (config.block_size.div_ceil(config.head_dim) as u64).min(
        crate::ternary_deltanet_gpu_forward::ATTN_SPLIT_DECODE_MAX_SPLITS_DEFAULT as u64,
    );
    total += config.n_head as u64 * attn_split_max * (config.head_dim as u64 + 2);

    // Per-layer persistent state (DeltaNet layers).
    let state_dim = n_v_heads * head_dim * head_dim;
    let n_gdn = layer_types
        .iter()
        .filter(|&&t| t == DeltaNetLayerType::DeltaNet)
        .count() as u64;
    total += n_gdn * (state_dim + conv_dim * kernel_size);

    // Attention KV caches — the block_size-proportional term (Issue 864's
    // warn sits on the same arithmetic).
    let n_attn = layer_types.len() as u64 - n_gdn;
    total += 2 * n_attn * config.block_size as u64 * kvd_attn;

    // Speculative-decode pools (Issue 665 P2 backups + Issue 727 H6 logits
    // pool) — only resident when the feature compiled them in.
    #[cfg(feature = "speculative_decode")]
    {
        total += n_gdn * (state_dim + conv_dim * kernel_size);
        total +=
            crate::ternary_deltanet_gpu_forward::TernaryDeltanetGpuForward::SPEC_MAX_K as u64 * vocab;
    }

    total * F32
}

/// The full constructor working-set estimate: weights + activations.
pub fn forward_working_set_bytes(config: &Config, weights: &QwenDeltaNetTernaryWeights) -> u64 {
    let layer_types = if weights.layer_types.is_empty() {
        vec![DeltaNetLayerType::Attention; config.n_layer]
    } else {
        weights.layer_types.clone()
    };
    ternary_weights_gpu_bytes(weights) + forward_activation_bytes(config, &layer_types)
}

/// The budget decision. Pure over its inputs; this form exists so tests can
/// pin the arithmetic without touching the process environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetVerdict {
    pub working_set_bytes: u64,
    pub budget_bytes: Option<u64>,
    pub margin_bytes: u64,
    pub allowed: bool,
}

impl BudgetVerdict {
    /// The refusal message (empty when allowed or when no budget is known).
    pub fn refusal_reason(&self) -> Option<String> {
        if self.allowed {
            return None;
        }
        let budget = self.budget_bytes?;
        Some(format!(
            "working set {:.2} GiB exceeds budget {:.2} GiB minus margin {:.2} GiB (allowed \u{2264} {:.2} GiB)",
            self.working_set_bytes as f64 / GIB_F64,
            budget as f64 / GIB_F64,
            self.margin_bytes as f64 / GIB_F64,
            budget.saturating_sub(self.margin_bytes) as f64 / GIB_F64,
        ))
    }
}

const GIB_F64: f64 = 1024.0 * 1024.0 * 1024.0;

/// Decide on a working set against a budget, with the default margin rule
/// (`max(budget/4, 2 GiB)`).
pub fn budget_verdict(working_set_bytes: u64, budget_bytes: Option<u64>) -> BudgetVerdict {
    let Some(budget) = budget_bytes else {
        return BudgetVerdict {
            working_set_bytes,
            budget_bytes: None,
            margin_bytes: 0,
            allowed: true,
        };
    };
    let margin = (budget / 4).max(DEFAULT_MARGIN_FLOOR_BYTES);
    let allowed = working_set_bytes.saturating_add(margin) <= budget;
    BudgetVerdict {
        working_set_bytes,
        budget_bytes: Some(budget),
        margin_bytes: margin,
        allowed,
    }
}

/// `=0` disables, per the repo-wide inverted-kill-switch convention.
fn preflight_disabled() -> bool {
    std::env::var(ENV_CHECK_DISABLE).is_ok_and(|v| v == "0")
}

fn parse_env_bytes(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok())
}

/// The pre-flight gate for `TernaryDeltanetGpuForward` construction.
///
/// Returns `Err(reason)` when the constructor should refuse (the caller
/// panics with the reason — the Issue-994 loud refusal). `Ok(None)` = check
/// disabled or no budget known, pre-flight skipped (the flush-gate still
/// protects); `Ok(Some(verdict))` = decided (always `allowed` on the Ok path).
pub fn check_forward_budget(
    ctx: &CubeCLContext,
    config: &Config,
    weights: &QwenDeltaNetTernaryWeights,
) -> Result<Option<BudgetVerdict>, String> {
    if preflight_disabled() {
        return Ok(None);
    }
    let budget = parse_env_bytes(ENV_BUDGET_BYTES).or_else(|| ctx.total_video_memory());
    let working_set = forward_working_set_bytes(config, weights);
    let mut verdict = budget_verdict(working_set, budget);
    if let Some(margin) = parse_env_bytes(ENV_MARGIN_BYTES) {
        verdict.margin_bytes = margin;
        verdict.allowed =
            working_set.saturating_add(margin) <= verdict.budget_bytes.unwrap_or(u64::MAX);
    }
    if let Some(reason) = verdict.refusal_reason() {
        return Err(format!(
            "{reason}. Dominant terms scale with `config.block_size` (the attention KV cache) — \
             clamp `config.block_size` to the real sequence bound BEFORE construction (the \
             riir-train driver pattern, Issue 510 T1/T2 / Issue 864), or move to a host with \
             more VRAM, or raise the budget via {ENV_BUDGET_BYTES} if you accept the risk. \
             (riir-ai Issue 994 refuse-to-run gate.)"
        ));
    }
    // Prong 2 — the single-buffer KV guard. The per-layer KV cache is the
    // only constructor buffer that grows with block_size, and the ≥52K-
    // position class is broken on this stack (bisected: 49,216 clean /
    // 57,408 device-lost / 65,472+ silent corruption, the latter reproduced
    // with ZERO pool failures — pool exhaustion refuted as the mechanism).
    let kv_bytes = kv_layer_bytes(config);
    let kv_max = parse_env_bytes(ENV_KV_LAYER_MAX_BYTES).unwrap_or(DEFAULT_KV_LAYER_MAX_BYTES);
    if kv_bytes > kv_max {
        let kvd = config.n_kv_head as u64 * config.head_dim as u64;
        let block_max = kv_max / (kvd * 4);
        return Err(format!(
            "per-layer attention KV cache {:.2} GiB (block_size {} × kvd {} × f32) exceeds the \
             bisected safe bound {:.2} GiB — the ≥52K-position class is broken on the \
             cubecl/wgpu stack (riir-ai Issue 994: block 57,408 lost the device during \
             construction; block 65,472+ produced DETERMINISTIC silent corruption, cos \
             0.419531 / top-1 0.0150, with zero allocation failures). Clamp `config.block_size` \
             ≤ {block_max} or raise {ENV_KV_LAYER_MAX_BYTES} ONLY with a fresh bisect + \
             external-reference validation. (riir-ai Issue 994 single-buffer KV gate.)",
            kv_bytes as f64 / GIB_F64,
            config.block_size,
            kvd,
            kv_max as f64 / GIB_F64,
        ));
    }
    Ok(Some(verdict))
}

/// The construction flush-gate: drain the client's stream error sink and
/// surface ANY construction-time failure (allocation, write, launch).
///
/// This is the prong that catches the measured silent-corruption mechanism:
/// the Issue 714 vendor patch routes a failed device-memory reserve into the
/// per-stream sink (keeping the server alive — right for multi-test
/// processes), but nothing drained the sink during construction, so the
/// forward ran on unbound buffers and produced self-consistent garbage
/// (Bench 600). One flush at the end of construction turns that into a loud
/// panic naming the drained errors.
pub fn drain_construction_errors(client: &crate::ActiveComputeClient) -> Result<(), String> {
    match client.flush() {
        Ok(()) => Ok(()),
        Err(e) => Err(format!(
            "construction-time GPU failure drained from the stream error sink: {e}. \
             A failed allocation/write/launch during construction leaves buffers unbound \
             while the forward still completes — the SELF-CONSISTENT SILENT-CORRUPTION class \
             of riir-ai Issue 994 (measured: cos 0.18-0.42 vs an external reference, bit-exact \
             in-process). Refusing to hand out a corrupted forward. If this is a deliberate \
             over-subscription test, disable the pre-flight with {ENV_CHECK_DISABLE}=0 — this \
             drain gate is correctness and stays on."
        )),
    }
}

#[cfg(test)]
mod ladder_tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    /// The clean 4-GiB-limit case: pages quarter down the ladder, slices
    /// halve per base, the terminal rung is the binding limit itself.
    #[test]
    fn ladder_4gib_power_of_two() {
        let rungs = subslices_ladder(4 * 1024 * MIB, 256);
        assert_eq!(
            rungs,
            vec![
                PoolRung { page_size: 16 * MIB, max_slice_size: MIB },
                PoolRung { page_size: 64 * MIB, max_slice_size: 8 * MIB },
                PoolRung { page_size: 256 * MIB, max_slice_size: 64 * MIB },
                PoolRung { page_size: 1024 * MIB, max_slice_size: 512 * MIB },
                PoolRung { page_size: 4096 * MIB, max_slice_size: 4096 * MIB },
            ]
        );
    }

    /// The back-solved 4090 binding limit (E1's hypothesis): the observed
    /// failed page 1,582,170,112 = limit/4 exactly, and the observed failing
    /// reserve 285,212,672 (the 4096×17,408×4 FFN chunk buffer) selects that
    /// rung — reproducing Bench 604 run C's panic arithmetic. Rungs are in
    /// `reserve`'s scan order (ascending), so the limit/4 rung is
    /// `rungs[len - 2]`.
    #[test]
    fn ladder_backsolved_4090_limit_reproduces_the_failure_page() {
        let limit = 6_328_680_448u64; // = 4 × 1,582,170,112
        let rungs = subslices_ladder(limit, 256);
        assert_eq!(
            rungs,
            vec![
                PoolRung { page_size: 24_721_408, max_slice_size: 1_545_088 },
                PoolRung { page_size: 98_885_632, max_slice_size: 12_360_704 },
                PoolRung { page_size: 395_542_528, max_slice_size: 98_885_632 },
                PoolRung { page_size: 1_582_170_112, max_slice_size: 791_085_056 },
                PoolRung { page_size: limit, max_slice_size: limit },
            ]
        );

        // The FFN chunk buffer (4096 tokens × 17,408 mlp × f32) selects the
        // limit/4 rung — its new-page allocation is exactly the observed
        // 1,582,170,112-byte device allocation that OOM'd.
        let ffn_chunk = 4096u64 * 17_408 * 4;
        assert_eq!(ffn_chunk, 285_212_672);
        assert_eq!(select_pool_rung(&rungs, ffn_chunk), Some(3));

        // The bisect-point KV caches also slice the limit/4 rung, with
        // co-tenancy dropping across the corruption gap: 49,216 (clean) → 7
        // slices/page; 57,408 (device lost) → 6; 65,472 (silent corruption)
        // and 65,600 → 5.
        for (block, expected_per_page) in [(49_216u64, 7u64), (57_408, 6), (65_472, 5), (65_600, 5)] {
            let kv = block * 1024 * 4; // kvd 1024, f32
            assert_eq!(select_pool_rung(&rungs, kv), Some(3), "block {block}");
            assert_eq!(slices_per_page(rungs[3], kv), expected_per_page, "block {block}");
        }
    }

    /// Near-page acceptance: a size within 20% of a rung's page routes to
    /// that rung even when no `max_slice` fits it — and a size ABOVE a rung's
    /// page never near-page-accepts it.
    #[test]
    fn select_pool_near_page_rule() {
        let rungs = subslices_ladder(4 * 1024 * MIB, 256);
        // 250 MiB is within 20% of the 256-MiB page → that rung, as a
        // near-page stray (its 64-MiB max_slice alone would not fit it).
        let sel = select_pool_rung(&rungs, 250 * MIB).unwrap();
        assert_eq!(rungs[sel].page_size, 256 * MIB);
        // 100 MiB exceeds the 256-MiB rung's max_slice AND is not near-page
        // → the 1-GiB rung via max_slice (512 MiB).
        let sel = select_pool_rung(&rungs, 100 * MIB).unwrap();
        assert_eq!(rungs[sel].page_size, 1024 * MIB);
        // 3.9 GiB: exceeds every intermediate max_slice; within 20% of the
        // 4-GiB terminal page → the terminal rung.
        let sel = select_pool_rung(&rungs, 3_900 * MIB).unwrap();
        assert_eq!(rungs[sel].page_size, 4096 * MIB);
        // Above the binding limit → no rung (BufferTooBig).
        assert!(select_pool_rung(&rungs, 5 * 1024 * MIB).is_none());
    }
}
