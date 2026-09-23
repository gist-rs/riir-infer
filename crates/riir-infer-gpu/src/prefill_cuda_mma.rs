//! Issue 734 Arm 6 e2e wiring (Bench 720) — dispatch `prefill_project`
//! through the raw-CUDA `mma.sync.m16n8k32.s8` i8 GEMM (Bench 719's
//! 2.45-2.49× kernel) via a HOST ROUND-TRIP bridge: read the CubeCL input
//! handle → DMA to host → upload to a CUDA staging buffer → quantize +
//! GEMM on the cudarc stream → read back → `client.write` into the CubeCL
//! output handle. This is the exact `prefill_project_metal` precedent
//! (Plan 534 T5), applied to the CUDA side.
//!
//! **Why a host round trip:** the CubeCL stack runs on wgpu/Vulkan and the
//! mma kernel on raw CUDA; sharing device memory between them requires
//! allocation-time `VK_KHR_external_memory` flags the CubeCL pool does not
//! set, so there is no zero-copy route without forking allocation. The
//! round-trip cost is therefore PART OF THE MEASUREMENT — this unit's G2
//! question is exactly "does the 2.45× kernel survive the bus?" (the
//! block-granularity / zero-copy follow-ups are armed on the answer).
//!
//! **Correctness (G1):** the div-form (`QuantDiv::Full`) and fold-form
//! (`FoldMode::Fused`) selectors root-caused in Bench 719 must reproduce
//! the shipping psplit bits BIT-EXACTLY on real activations — pinned by
//! the Bench-710 full-model FNV/argmax gates in `bench_734_arm6_e2e`.
//!
//! **Weights:** mirrored lazily once per `TernaryHandle`
//! (`cuda_mma_cache`, the Issue 727 H3 pattern) — GPU→host→CUDA, ~0.28 B
//! per ternary param (~7.6 GB for the full 27B model; per-weight failure
//! degrades that weight back to CubeCL permanently).
//!
//! **Knob:** `RIIR_PREFILL_CUDA_MMA` env ("1"/"true"/"on") or
//! [`set_prefill_use_cuda_mma`] — explicit env wins over the setter; the
//! default is OFF (the round-trip form is a measurement unit, not a
//! promotion candidate).

use std::sync::{Arc, Mutex, OnceLock};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::cubecl_runtime::ActiveRuntime;
use crate::gemm_ternary_i8_mma_cuda_raw::{GemmI8MmaScratch, GemmI8MmaError, GemmTernaryI8MmaCuda};
// FoldMode is consumed only by the `prefill_mmq_v2` q8 launch arms below;
// cfg-matched so feature slices without the arm (e.g. riir-train's dllm
// dependency slice) don't see an unused import.
#[cfg(feature = "prefill_mmq_v2")]
use crate::gemm_ternary_i8_mma_cuda_raw::FoldMode;
use crate::gemv_ternary_cubecl::TernaryHandle;

// ---------------------------------------------------------------------------
// Knob
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_CUDA_MMA: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn prefill_use_cuda_mma() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("RIIR_PREFILL_CUDA_MMA")
            .ok()
            .map(|s| matches!(s.trim(), "1" | "true" | "on"))
    });
    env.unwrap_or_else(|| PREFILL_CUDA_MMA.load(std::sync::atomic::Ordering::Relaxed))
}

/// Force-enable/disable the raw-CUDA mma prefill arm (overrides the
/// DEFAULT-OFF state; an explicit `RIIR_PREFILL_CUDA_MMA` env value wins
/// over both). Issue 734 Arm 6 A/B knob — public for the e2e benches.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_use_cuda_mma(on: bool) {
    PREFILL_CUDA_MMA.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Per-phase timing trace (`RIIR_PREFILL_CUDA_MMA_TRACE=1`) — the Bench-720
/// breakdown probe: how much of the arm's e2e loss is the CubeCL readback
/// (`read`), the bus (`up`/`down`), the CUDA compute (`quant+gemm`), and the
/// writeback (`write`). Diagnoses serialization-vs-bus for the follow-up
/// arms (block granularity / zero-copy).
fn trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| {
        std::env::var("RIIR_PREFILL_CUDA_MMA_TRACE").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    })
}

// ---------------------------------------------------------------------------
// Weight mirror
// ---------------------------------------------------------------------------

/// CUDA-side mirror of one ternary projection (pos/neg bit-plane words +
/// f32 group scales), uploaded once from the CubeCL handles.
///
/// `pos`/`neg` are `None` iff the mirror was built on the format-rung route
/// (Plan 572: `allow_packed_route` + the fmt knobs on at build time) — the
/// v8 GEMM consumes `packed` alone and the pair upload is skipped (halves
/// the weight VRAM). Dequant consumers ([`launch_dequant_wte_batch`]) and
/// the bitplane fallback need the pair — build those mirrors with
/// `allow_packed_route = false`.
pub struct CudaMmaWeightCache {
    // pub (not pub(crate)) until S4 homes prefill_cuda_full into this crate:
    // its embed launch reads the bitplane pair + scale directly, and the
    // carve left that file in riir-ai (E0616 x3 at prefill_cuda_full.rs:2447).
    // Read-only cross-crate access; S4 may narrow back when the file moves.
    pub pos: Option<CudaSlice<u32>>,
    pub neg: Option<CudaSlice<u32>>,
    pub scale: CudaSlice<f32>,
    /// The format-rung mirror (Plan 572; feature `prefill_mmq_v2` compiles
    /// it, `RIIR_PREFILL_MMQ_FMT` selects the rung — DEFAULT 3 since Issue
    /// 917, `0` the kill-switch). `None` keeps the cache byte-identical to
    /// the pre-rung surface.
    #[cfg(feature = "prefill_mmq_v2")]
    pub(crate) packed: Option<CudaSlice<u32>>,
}

/// Issue 884 T2b rung 4 (Plan 572) — the format-rung ingest transform:
/// dual bitplanes → packed-2-bit Q2_0-code words. The code table is
/// `riir_infer_core::quant::q2_0`'s (`block_q2_0`, the PrismML fork's MMQ
/// format) via the inverse of `repack_q2_0_to_ternary_group`: 00 = −1,
/// 01 = 0, 10 = +1; the (pos,neg) = (1,1) bit case folds to the zero code
/// exactly as the kernel's masked-SWAR decode folds it. Code 3 (Q2_0's
/// fourth state) is never emitted — bitplanes cannot express it.
///
/// Output: `u32[m * n/16]`, k-contiguous LSB-first — the 2-bit code of
/// weight k lives at bits `[2k mod 32, +2)` of word `k/16`. Size-neutral
/// vs the bitplane pair (2 bits/weight either way); `group_scale` is
/// untouched. The kernel-side consumer is `GEMM_BODY_V8_Q8`'s PRMT decode
/// (8 ALU-pipe ops per 8 weights post-903-T2a — the fork's decode class, B881).
#[cfg(feature = "prefill_mmq_v2")]
pub(crate) fn pack_bitplanes_to_q2(pos: &[u32], neg: &[u32], m: usize, n: usize) -> Vec<u32> {
    debug_assert_eq!(pos.len(), neg.len(), "bitplane word counts");
    // (pos_bit, neg_bit) -> Q2_0 code: (1,0)=2 (+1), (0,1)=0 (−1),
    // (0,0)=1 (0), (1,1)=1 (fold — the non-disjoint bit case decodes to 0
    // on the shipping path too).
    let code = |p: u32, n: u32| -> u32 { match (p, n) { (1, 0) => 2, (0, 1) => 0, _ => 1 } };
    let wpr = n / 32;
    let packed_wpr = n / 16;
    let mut out = vec![0u32; m * packed_wpr];
    for row in 0..m {
        for w in 0..wpr {
            let (pw, nw) = (pos[row * wpr + w], neg[row * wpr + w]);
            // 32 codes -> 8 bytes -> 2 packed words (16 weights each).
            for half in 0..2u32 {
                let mut acc = 0u32;
                for i in 0..16u32 {
                    let bit = 1u32 << (half * 16 + i);
                    let c = code(u32::from(pw & bit != 0), u32::from(nw & bit != 0));
                    acc |= c << (2 * i);
                }
                out[row * packed_wpr + w * 2 + half as usize] = acc;
            }
        }
    }
    out
}

/// The prefill GEMM behind the weight cache (Plan 572 + the Plan-573/574 mode
/// axis): the format-rung arm ([`
/// crate::gemm_ternary_i8_mma_cuda_raw::mmq_fmt_enabled`] + a packed mirror
/// built at cache time) dispatches by `RIIR_PREFILL_MMQ_FMT` mode — default
/// `3` since Issue 917 (the v10t L2-traffic-repair twin: smem-staged group
/// scales + transposed coalesced epilogue, ncu-guided, the measured winner);
/// `1` the v8t global-A PRMT-decode kernel, `2` the v9t staged-packed kernel,
/// `0` the kill-switch (unpacked).
/// Otherwise this is [`GemmTernaryI8MmaCuda::launch_prefill_gemm`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill_gemm_cached(
    mma: &GemmTernaryI8MmaCuda,
    stream: &CudaStream,
    cache: &CudaMmaWeightCache,
    scratch: &GemmI8MmaScratch,
    out: &CudaSlice<f32>,
    m: usize,
    n: usize,
    p: usize,
) -> Result<(), GemmI8MmaError> {
    #[cfg(feature = "prefill_mmq_v2")]
    if crate::gemm_ternary_i8_mma_cuda_raw::q8_act_enabled()
        && crate::gemm_ternary_i8_mma_cuda_raw::mmq_v2_enabled()
        && crate::gemm_ternary_i8_mma_cuda_raw::mmq_fmt_enabled()
        && let Some(packed) = &cache.packed
    {
        return match crate::gemm_ternary_i8_mma_cuda_raw::mmq_fmt_mode() {
            3 => mma.launch_gemm_q8_v10t(stream, packed, &cache.scale, scratch, out, m, n, p, FoldMode::Fused),
            2 => mma.launch_gemm_q8_v9t(stream, packed, &cache.scale, scratch, out, m, n, p, FoldMode::Fused),
            _ => mma.launch_gemm_q8_v8t(stream, packed, &cache.scale, scratch, out, m, n, p, FoldMode::Fused),
        };
    }
    let (Some(pos), Some(neg)) = (&cache.pos, &cache.neg) else {
        return Err(GemmI8MmaError::Launch(
            "fmt-armed weight mirror carries packed codes only — the bitplane \
             fallback requires the mirror to be rebuilt with the fmt knob off"
                .into(),
        ));
    };
    mma.launch_prefill_gemm(stream, pos, neg, &cache.scale, scratch, out, m, n, p)
}

/// Issue 902 T1 — the fused gate+up pair dispatch: one v11gu/v11gut launch
/// computing BOTH same-shape projections from one B (activation) tile stage,
/// when [`crate::gemm_ternary_i8_mma_cuda_raw::mmq_gu_mode`] arms it and both
/// mirrors carry packed codes. Otherwise (and on every non-fmt route) this is
/// exactly two [`launch_prefill_gemm_cached`] calls — the caller cannot tell
/// the arms apart beyond the launch count.
#[allow(clippy::too_many_arguments)]
pub fn launch_prefill_gemm_pair_cached(
    mma: &GemmTernaryI8MmaCuda,
    stream: &CudaStream,
    cache0: &CudaMmaWeightCache,
    cache1: &CudaMmaWeightCache,
    scratch: &GemmI8MmaScratch,
    out0: &CudaSlice<f32>,
    out1: &CudaSlice<f32>,
    m: usize,
    n: usize,
    p: usize,
) -> Result<(), GemmI8MmaError> {
    #[cfg(feature = "prefill_mmq_v2")]
    if crate::gemm_ternary_i8_mma_cuda_raw::q8_act_enabled()
        && crate::gemm_ternary_i8_mma_cuda_raw::mmq_v2_enabled()
        && crate::gemm_ternary_i8_mma_cuda_raw::mmq_fmt_enabled()
        && crate::gemm_ternary_i8_mma_cuda_raw::mmq_gu_mode() != 0
        && let (Some(packed0), Some(packed1)) = (&cache0.packed, &cache1.packed)
    {
        return match crate::gemm_ternary_i8_mma_cuda_raw::mmq_gu_mode() {
            4 => mma.launch_gemm_q8_v11gq_pair(
                stream, packed0, &cache0.scale, packed1, &cache1.scale, scratch, out0, out1, m,
                n, p, FoldMode::Fused,
            ),
            3 => mma.launch_gemm_q8_v11gs_pair(
                stream, packed0, &cache0.scale, packed1, &cache1.scale, scratch, out0, out1, m,
                n, p, FoldMode::Fused,
            ),
            2 => mma.launch_gemm_q8_v11gut_pair(
                stream, packed0, &cache0.scale, packed1, &cache1.scale, scratch, out0, out1, m,
                n, p, FoldMode::Fused,
            ),
            _ => mma.launch_gemm_q8_v11gu_pair(
                stream, packed0, &cache0.scale, packed1, &cache1.scale, scratch, out0, out1, m,
                n, p, FoldMode::Fused,
            ),
        };
    }
    launch_prefill_gemm_cached(mma, stream, cache0, scratch, out0, m, n, p)?;
    launch_prefill_gemm_cached(mma, stream, cache1, scratch, out1, m, n, p)
}

pub fn build_weight_cache(
    client: &ComputeClient<ActiveRuntime>,
    stream: &Arc<CudaStream>,
    w: &TernaryHandle,
    allow_packed_route: bool,
) -> Result<CudaMmaWeightCache, String> {
    let pos_bytes = client
        .read_one(w.pos_bits_u32.clone())
        .map_err(|e| format!("read pos_bits: {e:?}"))?;
    let neg_bytes = client
        .read_one(w.neg_bits_u32.clone())
        .map_err(|e| format!("read neg_bits: {e:?}"))?;
    let scale_bytes = client
        .read_one(w.group_scale_f32.clone())
        .map_err(|e| format!("read group_scale: {e:?}"))?;
    let pos = u32::from_bytes(&pos_bytes);
    let neg = u32::from_bytes(&neg_bytes);
    let scale = f32::from_bytes(&scale_bytes);
    debug_assert_eq!(pos.len(), w.m * w.blocks64 * 2, "pos word count");
    debug_assert_eq!(neg.len(), w.m * w.blocks64 * 2, "neg word count");
    debug_assert_eq!(scale.len(), w.m * w.groups_per_row, "scale count");
    // The format-rung route (Plan 572): when the GEMM consumer will run the
    // v8 packed kernel, upload ONLY the packed mirror — the bitplane pair is
    // never read and the pair upload (~2/3 of the armed cache) is skipped.
    // Dequant consumers (wte) build with `allow_packed_route = false` and
    // always carry the pair.
    #[cfg(feature = "prefill_mmq_v2")]
    let packed_route =
        allow_packed_route && crate::gemm_ternary_i8_mma_cuda_raw::mmq_fmt_route_enabled();
    #[cfg(not(feature = "prefill_mmq_v2"))]
    let _ = allow_packed_route; // the param only gates the mmq_v2 branch below
    #[cfg(feature = "prefill_mmq_v2")]
    if packed_route {
        let packed_words = pack_bitplanes_to_q2(pos, neg, w.m, w.n);
        let packed = stream
            .clone_htod(&packed_words)
            .map_err(|e| format!("upload packed: {e}"))?;
        let scale_dev = stream
            .clone_htod(scale)
            .map_err(|e| format!("upload scale: {e}"))?;
        return Ok(CudaMmaWeightCache {
            pos: None,
            neg: None,
            scale: scale_dev,
            packed: Some(packed),
        });
    }
    let pos = stream
        .clone_htod(pos)
        .map_err(|e| format!("upload pos: {e}"))?;
    let neg = stream
        .clone_htod(neg)
        .map_err(|e| format!("upload neg: {e}"))?;
    let scale = stream
        .clone_htod(scale)
        .map_err(|e| format!("upload scale: {e}"))?;
    Ok(CudaMmaWeightCache {
        pos: Some(pos),
        neg: Some(neg),
        scale,
        #[cfg(feature = "prefill_mmq_v2")]
        packed: None,
    })
}

// ---------------------------------------------------------------------------
// The stack (context + stream + kernels + grow-only staging)
// ---------------------------------------------------------------------------

/// Grow-only device staging, reused across every `prefill_project` call.
/// Kernels and copies are bounded to exact `[0..len)` prefixes via
/// `try_slice` views, so an oversized buffer is safe (stale tails are
/// never read — the kernels derive all extents from the `(n, p)` args).
struct Bufs {
    in_dev: Option<CudaSlice<f32>>,
    out_dev: Option<CudaSlice<f32>>,
    out_host: Vec<f32>,
    /// `(words, p, scratch)` — allocated for at least this many q-words
    /// and tokens.
    scratch: Option<(usize, usize, GemmI8MmaScratch)>,
}

struct Stack {
    kernels: GemmTernaryI8MmaCuda,
    stream: Arc<CudaStream>,
    bufs: Mutex<Bufs>,
}

static STACK: OnceLock<Option<Arc<Stack>>> = OnceLock::new();

fn stack() -> Option<Arc<Stack>> {
    STACK
        .get_or_init(|| match build_stack() {
            Ok(s) => Some(Arc::new(s)),
            Err(e) => {
                eprintln!(
                    "[734-arm6] CUDA mma stack init failed ({e}) — prefill stays on CubeCL"
                );
                None
            }
        })
        .clone()
}

fn build_stack() -> Result<Stack, String> {
    let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
    let stream = ctx.new_stream().map_err(|e| e.to_string())?;
    let kernels = GemmTernaryI8MmaCuda::new(ctx).map_err(|e| e.to_string())?;
    Ok(Stack {
        kernels,
        stream,
        bufs: Mutex::new(Bufs {
            in_dev: None,
            out_dev: None,
            out_host: Vec::new(),
            scratch: None,
        }),
    })
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Run one `prefill_project` projection on the raw-CUDA mma kernel.
/// Returns `false` (→ the caller falls through to the CubeCL path) on any
/// init/read/alloc/launch failure — degradation is bit-safe: both paths
/// compute the same values.
///
/// Requires `n % 128 == 0` (the GROUP_COLS contract) and `p >= 1` —
/// checked by the caller (the `prefill_project` arm).
#[allow(clippy::too_many_lines)]
pub fn dispatch(
    client: &ComputeClient<ActiveRuntime>,
    w: &TernaryHandle,
    input: &Handle,
    output: &Handle,
    p: usize,
) -> bool {
    if p == 0 {
        return true;
    }
    let Some(stack) = stack() else { return false };
    let (m, n) = (w.m, w.n);
    let trace = trace_enabled();
    let t_read = std::time::Instant::now();

    // Weight mirror — lazy, once per handle (the Issue 727 H3 pattern).
    let cache = w.cuda_mma_cache.get_or_init(|| {
        match build_weight_cache(client, &stack.stream, w, true) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                eprintln!(
                    "[734-arm6] weight mirror failed for m={m} n={n} ({e}) — \
                     this weight stays on CubeCL"
                );
                None
            }
        }
    });
    let Some(cache) = cache else { return false };

    // 1) Read the input activations back to host (DMA; syncs CubeCL).
    let Ok(input_bytes) = client.read_one(input.clone()) else {
        return false;
    };
    let input_f32 = f32::from_bytes(&input_bytes);
    debug_assert_eq!(input_f32.len(), n * p, "prefill_project input shape");
    let read_us = t_read.elapsed().as_micros();

    let in_len = n * p;
    let out_len = m * p;
    let words = p * (n / 4);

    let Ok(mut bufs) = stack.bufs.lock() else { return false };

    // 2) Grow-only staging (replace when too small; prefix views below).
    if bufs.in_dev.as_ref().is_none_or(|s| s.len() < in_len) {
        match stack.stream.alloc_zeros::<f32>(in_len) {
            Ok(s) => bufs.in_dev = Some(s),
            Err(_) => return false,
        }
    }
    if bufs.out_dev.as_ref().is_none_or(|s| s.len() < out_len) {
        match stack.stream.alloc_zeros::<f32>(out_len) {
            Ok(s) => bufs.out_dev = Some(s),
            Err(_) => return false,
        }
    }
    if bufs
        .scratch
        .as_ref()
        .is_none_or(|(cw, cp, _)| *cw < words || *cp < p)
    {
        match stack.kernels.alloc_scratch(&stack.stream, n, p) {
            Ok(s) => bufs.scratch = Some((words, p, s)),
            Err(_) => return false,
        }
    }
    if bufs.out_host.len() < out_len {
        bufs.out_host.resize(out_len, 0.0);
    }

    let Bufs {
        in_dev,
        out_dev,
        out_host,
        scratch,
    } = &mut *bufs;
    let Some(in_dev) = in_dev.as_mut() else { return false };
    let Some(out_dev) = out_dev.as_mut() else { return false };
    let Some((_, _, scratch)) = scratch.as_mut() else { return false };

    // 3) Upload the activations into the input staging prefix (scoped view:
    //    the mutable borrow ends with the block, re-freeing `in_dev` for the
    //    quantize launch).
    let t_up = std::time::Instant::now();
    {
        let Some(mut in_view) = in_dev.try_slice_mut(0..in_len) else { return false };
        if stack.stream.memcpy_htod(input_f32, &mut in_view).is_err() {
            return false;
        }
    }
    let up_us = t_up.elapsed().as_micros();

    // 4) Quantize + GEMM. `QuantDiv::Full` + `FoldMode::Fused` are the
    //    Bench-719 bit-identity selectors vs the Vulkan shipping path
    //    (the NVIDIA SPIR-V consumer's OpFDiv lowering + the driver's
    //    outer-fold FMA contraction).
    let t_gemm = std::time::Instant::now();
    if stack
        .kernels
        .launch_prefill_quantize(&stack.stream, in_dev, scratch, n, p)
        .is_err()
    {
        return false;
    }
    if launch_prefill_gemm_cached(
        &stack.kernels,
        &stack.stream,
        cache,
        scratch,
        out_dev,
        m,
        n,
        p,
    )
    .is_err()
    {
        return false;
    }
    let gemm_us = t_gemm.elapsed().as_micros();

    // 5) Read back the exact output prefix (memcpy_dtoh copies the whole
    //    bound range — the view keeps it at p*m).
    let t_down = std::time::Instant::now();
    let Some(out_view) = out_dev.try_slice(0..out_len) else { return false };
    if stack
        .stream
        .memcpy_dtoh(&out_view, &mut out_host[..out_len])
        .is_err()
    {
        return false;
    }
    let down_us = t_down.elapsed().as_micros();

    // 6) Write into the CubeCL output handle (the metal-path precedent).
    let t_write = std::time::Instant::now();
    client.write(
        output,
        cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&out_host[..out_len]).to_vec()),
    );
    if trace {
        eprintln!(
            "[720-trace] m={m} n={n} p={p}: read {read_us}us up {up_us}us gemm {gemm_us}us down {down_us}us write {}us",
            t_write.elapsed().as_micros()
        );
    }
    true
}

#[cfg(all(test, feature = "prefill_mmq_v2"))]
mod fmt_rung_tests {
    use super::*;

    /// The transform's decode reference: expand the packed codes back to
    /// sign values and compare against the bitplane semantics (pos − neg,
    /// with the (1,1) fold to 0) — the exact arithmetic the shipping kernel
    /// decode (`V6_A_STORE` / `v7_expand`) applies, and the contract the
    /// PRMT table must reproduce (the device-side gate pins the kernel).
    fn packed_weight(packed: &[u32], packed_wpr: usize, row: usize, k: usize) -> i32 {
        let w = packed[row * packed_wpr + k / 16];
        let code = (w >> (2 * (k % 16))) & 0x3;
        match code {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => panic!("code 3 must never be emitted (Q2_0 fourth state)"),
        }
    }

    #[test]
    fn pack_matches_bitplane_semantics() {
        let (m, n) = (3usize, 256usize);
        let wpr = n / 32;
        // Adversarial fixtures: non-disjoint planes (the (1,1) fold),
        // all-zero, all-pos, all-neg, and a wrung LCG.
        let pos_bits: Vec<u32> = (0..m * wpr)
            .map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) as u32 | (i as u32 / 7))
            .collect();
        let neg_bits: Vec<u32> = (0..m * wpr)
            .map(|i| (i as u64).wrapping_mul(0xBF58476D1CE4E5B9) as u32 | 0x5555_5555)
            .collect();
        let packed = pack_bitplanes_to_q2(&pos_bits, &neg_bits, m, n);
        assert_eq!(packed.len(), m * (n / 16), "size-neutral: 2 bits/weight");
        let packed_wpr = n / 16;
        for row in 0..m {
            for k in 0..n {
                let w = k / 32;
                let bit = 1u32 << (k % 32);
                let expect = match (
                    pos_bits[row * wpr + w] & bit != 0,
                    neg_bits[row * wpr + w] & bit != 0,
                ) {
                    (true, false) => 1,
                    (false, true) => -1,
                    // (1,1) folds to 0 — the masked-SWAR kernel semantics.
                    _ => 0,
                };
                assert_eq!(
                    packed_weight(&packed, packed_wpr, row, k),
                    expect,
                    "row {row} k {k}: packed code must decode to the \
                     bitplane weight (incl. the (1,1) fold)"
                );
            }
        }
    }

    #[test]
    fn pack_extremes_and_code3_never() {
        let (m, n) = (2usize, 128usize);
        let wpr = n / 32;
        let packed_wpr = n / 16;
        // All-zero planes -> every code = 1 (zero weight).
        let packed = pack_bitplanes_to_q2(&vec![0u32; m * wpr], &vec![0u32; m * wpr], m, n);
        assert!(packed.iter().all(|&w| w == 0x5555_5555), "all codes 01");
        // All-neg -> every code = 0 (-1).
        let packed = pack_bitplanes_to_q2(
            &vec![0u32; m * wpr],
            &vec![u32::MAX; m * wpr],
            m,
            n,
        );
        assert!(packed.iter().all(|&w| w == 0), "all codes 00");
        // All-pos -> every code = 2 (+1).
        let packed = pack_bitplanes_to_q2(
            &vec![u32::MAX; m * wpr],
            &vec![0u32; m * wpr],
            m,
            n,
        );
        assert!(packed.iter().all(|&w| w == 0xAAAA_AAAA), "all codes 10");
        // Fully non-disjoint ((1,1) everywhere) -> every code = 1 (the fold).
        let packed = pack_bitplanes_to_q2(
            &vec![u32::MAX; m * wpr],
            &vec![u32::MAX; m * wpr],
            m,
            n,
        );
        assert!(packed.iter().all(|&w| w == 0x5555_5555));
        // And the decode reference agrees on the folded fixture.
        for row in 0..m {
            for k in 0..n {
                assert_eq!(packed_weight(&packed, packed_wpr, row, k), 0);
            }
        }
    }

    /// Offline model of the kernel's PRMT decode (v8_decode_pair) — lets the
    /// selector math be validated against the SWAR reference without a
    /// device. PTX prmt: per output byte j, selector nibble s[4j+3:4j];
    /// s[1:0] = byte index (0-3 from a, 4-7 from b), s[2] = source select,
    /// s[3] = msb-replicate.
    fn prmt(a: u32, b: u32, s: u32) -> u32 {
        let pool = [
            a & 0xFF,
            (a >> 8) & 0xFF,
            (a >> 16) & 0xFF,
            (a >> 24) & 0xFF,
            b & 0xFF,
            (b >> 8) & 0xFF,
            (b >> 16) & 0xFF,
            (b >> 24) & 0xFF,
        ];
        let mut d = 0u32;
        for j in 0..4u32 {
            let nib = (s >> (4 * j)) & 0xF;
            let byte = pool[(nib & 7) as usize];
            let byte = if nib & 8 != 0 {
                if byte & 0x80 != 0 { 0xFF } else { 0x00 }
            } else {
                byte
            };
            d |= byte << (8 * j);
        }
        d
    }

    fn decode_pair(l0: u32, l2: u32, sel: u32) -> (u32, u32) {
        let w = prmt(l0, l2, sel) & 0x0000_FFFF;
        let t1 = w & 0x3333_3333;
        let e = prmt(0x0001_00FF, 0, t1);
        let t2 = (w >> 2) & 0x3333_3333;
        let o = prmt(0x0001_00FF, 0, t2);
        (prmt(e, o, 0x5140), prmt(e, o, 0x7362))
    }

    fn v7_expand(tp: u32, up: u32) -> u32 {
        let t = (tp * 0x0020_4081) & 0x0101_0101;
        let u = ((up * 0x0020_4081) & 0x0101_0101) * 0xFF;
        t | u
    }

    #[test]
    fn decode_model_matches_swar_reference() {
        let _wpr = 8usize; // n = 256
        for grp in 0..2usize {
            for ks in 0..4usize {
                for t in 0..4u32 {
                    let pw = ((grp * 7 + ks * 13 + t as usize) as u32).wrapping_mul(0x9E37_79B9);
                    let nw = ((grp * 5 + ks * 3 + t as usize * 17) as u32).wrapping_mul(0xBF58_476D);
                    // The bitplane SWAR reference (v7's exact decode).
                    let tp0 = (pw >> (t * 4)) & 0xF;
                    let tn0 = (nw >> (t * 4)) & 0xF;
                    let a0_ref = v7_expand(tp0 & !tn0, tn0 & !tp0);
                    let tp4 = (pw >> (16 + t * 4)) & 0xF;
                    let tn4 = (nw >> (16 + t * 4)) & 0xF;
                    let a2_ref = v7_expand(tp4 & !tn4, tn4 & !tp4);
                    // The packed words through the kernel's word indexing.
                    let l0 = pack_bitplanes_to_q2(&[pw], &[nw], 1, 32)[0];
                    let l1 = pack_bitplanes_to_q2(&[pw], &[nw], 1, 32)[1];
                    let (a0, a2) = decode_pair(l0, l1, 0x40 + t * 0x11);
                    assert_eq!(a0, a0_ref, "a0 grp {grp} ks {ks} t {t}");
                    assert_eq!(a2, a2_ref, "a2 grp {grp} ks {ks} t {t}");
                }
            }
        }
    }

    #[test]
    fn pack_q2_0_bridge_inverse() {
        // The bridge (riir_infer_core q2_0 repack) maps code 0 -> neg bit,
        // code 2 -> pos bit. Round-trip: bitplanes -> packed -> (the
        // bridge's own decode table) must agree bit-for-bit.
        let (m, n) = (1usize, 512usize);
        let wpr = n / 32;
        let pos_bits: Vec<u32> = (0..wpr).map(|i| (i as u32).wrapping_mul(0x1234_5679)).collect();
        let neg_bits: Vec<u32> = (0..wpr).map(|i| (i as u32).wrapping_mul(0x9ABC_DEF1)).collect();
        let packed = pack_bitplanes_to_q2(&pos_bits, &neg_bits, m, n);
        for k in 0..n {
            let code = (packed[k / 16] >> (2 * (k % 16))) & 0x3;
            let p = (pos_bits[k / 32] >> (k % 32)) & 1;
            let ng = (neg_bits[k / 32] >> (k % 32)) & 1;
            // repack_q2_0_to_ternary_group's table: 0 -> neg, 2 -> pos,
            // 1 -> neither. (3 rejected — see pack_bitplanes_to_q2's doc.)
            assert_eq!(code, match (p, ng) { (1, 0) => 2, (0, 1) => 0, _ => 1 });
        }
    }
}
