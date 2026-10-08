//! Issue 1005 T8 — the GPU eDLM forward (`edlm_gpu` feature): f16-resident
//! weights, multi-query GQA flash attention with per-row visibility bounds,
//! parity vs the CPU T7 cached path as the gate (tolerance class, laya-G5
//! style: argmax exact + disclosed drift — never bit-identity).
//!
//! # The decomposition (T7's, unchanged)
//!
//! eDLM never needs a general bool mask on the GPU. The T7 state-prefix
//! decomposition maps every phase to "query row `i` attends keys `0..t_n(i)`":
//!
//! - **state prefill**: `t_n(i) = sl` (state_bidir) or `i + 1` (causal) over
//!   the state's own keys — `branch_mask` over an all-state segment;
//! - **branch continuations**: keys `[state KV | own branch KV]`,
//!   `t_n(i) = sl + i + 1` — `branch_continuation_allow` (all state keys +
//!   causal within the branch; state keys precede every branch query).
//!
//! One kernel shape serves both ([`EdlmAttnMultiCubeCL`]).
//!
//! # Weights (the upload-path decision)
//!
//! Q8_0 → f32 EXACT on the host (the same `dequant_f16_to_f32` the CPU
//! streaming path reads) → round-to-nearest f16 → f16-resident buffers
//! (~15.7 GB on the 8B release: fits the 24 GB 4090 and unified-memory M3).
//! Every GEMM accumulates f32 (the f16 GEMV family's law), so the only
//! numerics faces vs the CPU f32 path are (1) the f16 weight rounding
//! (rel ≤ 2^-11 per weight) and (2) the attention kernel's online-softmax
//! accumulation order. Quant-resident GPU dequant kernels are the documented
//! fallback if the tolerance gate ever reds — a bigger kernel with no
//! measurement-lane need (this lane is CC BY-NC, local bench only).
//!
//! # Sync posture (v1, the `llama_cubecl` hybrid shape)
//!
//! GPU: the GEMMs ([`MatmulF16bCubeCL`]), the multi-row RMSNorm, the
//! attention. CPU (between readbacks): per-row split, qk-norm, RoPE, SwiGLU,
//! residuals, the pointer head — the exact core helpers, never copies. Four
//! readbacks per layer. The GPU-resident elementwise fold is the perf
//! follow-up; correctness is the v1 gate.
//!
//! # State cache posture (v1)
//!
//! The state K/V lives CPU-side after the prefill (it was on the host for the
//! upload anyway) and each branch re-uploads the combined `[state | branch]`
//! cache per layer. The GPU-resident KV carry is the perf follow-up.
//!
//! # Out of scope (v1, disclosed)
//!
//! - The packed-mask forward (option isolation) — rows/cached semantics only,
//!   the CPU cached path's own caveat.
//! - CUDA graphs / batched multi-branch dispatch.
//!
//! The CMMA tensor-core GEMM (the first perf follow-up) is IN: the four
//! projections dispatch through [`crate::matmul_f16b_cmma_cubecl`] unless
//! `EDLM_GPU_CMMA=0` (the scalar kernel stays the kill-switch posture; the
//! GOAT evidence: `tests/edlm_matmul_cmma_goat.rs` + the issue row).

#[cfg(feature = "edlm_gpu")]
use crate::cubecl_runtime::{
    ActiveRuntime, CubeCLContext, assert_binding_derives_units, create_f32, create_u32, read_f32,
};
use crate::matmul_f16b_cubecl::MatmulF16bCubeCL;
#[cfg(feature = "edlm_gpu")]
use cubecl::prelude::*;
#[cfg(feature = "edlm_gpu")]
use cubecl::server::Handle;
#[cfg(feature = "edlm_gpu")]
use half::f16 as half_f16;
#[cfg(feature = "edlm_gpu")]
use riir_infer_core::rope::{RopeFreqTable, apply_rope_with_freq};
#[cfg(feature = "edlm_gpu")]
use riir_infer_core::transformer::edlm::{
    BranchRow, EdlmGgufModel, EdlmLayerKv, EdlmPointerHead, EdlmStateKv, EdlmWeights,
    PackedEncoding, qk_norm_inplace,
};
#[cfg(feature = "edlm_gpu")]
use riir_infer_core::types::{Config, kv_dim, rmsnorm_with_gamma_eps, swiglu};

// ── kernels ─────────────────────────────────────────────────────────

/// Multi-row RMSNorm: `out[r, i] = x[r, i] / sqrt(mean(x[r,:]^2) + eps) ·
/// gamma[i]` — the layer-norm shape (one normed stream feeding the QKV and
/// gate/up GEMMs for the whole sequence).
///
/// Grid `(rows, 1, 1)`, 256-thread cubes striding the row (n is unbounded —
/// the while-loop stride covers any n), the eps riding `params[0]` (the
/// crate's params-buffer discipline: no u32→f32 cast issues, no 5th array).
/// The sum-of-squares order differs from the CPU sequential sum —
/// tolerance-class, never bit-identity.
#[cfg(feature = "edlm_gpu")]
#[cube(launch_unchecked)]
fn edlm_rmsnorm_rows_f32(x: &[f32], gamma: &[f32], params: &[f32], out: &mut [f32]) {
    let n = gamma.len() as u32;
    let row = CUBE_POS_X;
    let base = (row * n) as usize;
    let tid = UNIT_POS;

    let mut local = f32::new(0.0f32);
    let mut i = tid;
    while i < n {
        let v = x[base + i as usize];
        local += v * v;
        i += 256u32;
    }

    let mut red = Shared::<[f32]>::new_slice(256usize);
    red[tid as usize] = local;
    sync_cube();
    if tid < 128u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 128u32) as usize];
    }
    sync_cube();
    if tid < 64u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 64u32) as usize];
    }
    sync_cube();
    if tid < 32u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid < 1u32 {
        red[0usize] = red[0usize] + red[1usize];
    }
    sync_cube();

    let sum_sq = red[0usize];
    let inv_rms = f32::new(1.0f32) / (sum_sq / (n as f32) + params[0usize]).sqrt();
    let mut i = tid;
    while i < n {
        out[base + i as usize] = x[base + i as usize] * inv_rms * gamma[i as usize];
        i += 256u32;
    }
}

/// Multi-query GQA flash attention with per-row visibility bounds — the eDLM
/// eligibility law (see the module doc). One cube per (head, query): grid
/// `(n_head, seq_q, 1)`, 128-thread cubes == head_dim (the 8B release shape,
/// asserted at the launcher). `params = [n_head, n_kv_head, head_dim, scale]`.
///
/// Online softmax over key tiles of 128 (the `attention_decode` family's
/// shape): each thread owns one key per tile and computes the full smem-Q dot
/// against its global K row; the tile max/sum ride unrolled cube reductions;
/// the output dimension `tid` accumulates its own weighted V sum, rescaled by
/// the running-max correction each tile, normalized by the final running sum.
/// Accumulation order differs from the CPU plain softmax — tolerance-class.
#[cfg(feature = "edlm_gpu")]
#[cube(launch_unchecked)]
fn edlm_attn_multi_f32(q: &[f32], kv: &[f32], t_n: &[u32], params: &[f32], out: &mut [f32]) {
    let n_head = params[0usize] as u32;
    let n_kv = params[1usize] as u32;
    let hd = params[2usize] as u32;
    let scale = params[3usize];
    let kvd = n_kv * hd;
    let n_pos = (kv.len() as u32) / (2u32 * kvd);
    // The combined cache is [keys(n_pos·kvd) | values(n_pos·kvd)] — the V of
    // key j lives at kv_half + j·kvd (the attention_decode family's layout).
    let kv_half = (kv.len() as u32) / 2u32;

    let h = CUBE_POS_X;
    let qi = CUBE_POS_Y;
    let kvh = h * n_kv / n_head;
    let tid = UNIT_POS;

    let mut q_smem = Shared::<[f32]>::new_slice(128usize);
    let mut w = Shared::<[f32]>::new_slice(128usize);
    let mut red = Shared::<[f32]>::new_slice(128usize);
    q_smem[tid as usize] = q[((qi * n_head + h) * hd + tid) as usize];
    sync_cube();

    let vis = t_n[qi as usize];
    let big_neg = f32::new(-1.0e30f32);
    let mut m = big_neg;
    let mut ssum = f32::new(0.0f32);
    let mut acc = f32::new(0.0f32);

    let mut j0 = 0u32;
    while j0 < vis {
        // ── score: thread tid owns key j0 + tid (full smem-Q dot) ──
        let j = j0 + tid;
        let mut s = big_neg;
        if j < vis && j < n_pos {
            let kro = (j * kvd + kvh * hd) as usize;
            let mut dot = f32::new(0.0f32);
            let mut t = 0u32;
            while t < hd {
                dot += q_smem[t as usize] * kv[kro + t as usize];
                t += 1u32;
            }
            s = dot * scale;
        }

        // ── tile max ──
        red[tid as usize] = s;
        sync_cube();
        let mut off = 64u32;
        while off >= 1u32 {
            if tid < off {
                let other = red[(tid + off) as usize];
                if other > red[tid as usize] {
                    red[tid as usize] = other;
                }
            }
            sync_cube();
            off /= 2u32;
        }
        let tile_max = red[0usize];

        // ── tile-local weights + sum (0 for masked keys — the guard, not the
        //    exp, keeps them out) ──
        let mut my_exp = f32::new(0.0f32);
        if j < vis && j < n_pos {
            my_exp = f32::exp(s - tile_max);
        }
        w[tid as usize] = my_exp;
        red[tid as usize] = my_exp;
        sync_cube();
        let mut off = 64u32;
        while off >= 1u32 {
            if tid < off {
                red[tid as usize] = red[tid as usize] + red[(tid + off) as usize];
            }
            sync_cube();
            off /= 2u32;
        }
        let tile_sum = red[0usize];

        // ── weighted V accumulation for MY output dimension (tid) ──
        // Keys j0..j0+bound, bound = min(vis − j0, hd); vis ≤ n_pos (the
        // launcher's t_n law) keeps the V reads in range.
        let mut tile_val = f32::new(0.0f32);
        let lane_count = vis - j0;
        let mut bound = hd;
        if lane_count < hd {
            bound = lane_count;
        }
        let v_base = (kvh * hd) as usize;
        let mut jj = 0u32;
        while jj < bound {
            tile_val += w[jj as usize]
                * kv[(kv_half + (j0 + jj) * kvd) as usize + v_base + tid as usize];
            jj += 1u32;
        }

        // ── online softmax update (the family's correction form) ──
        let mut new_max = m;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(m - new_max);
        let curr_corr = f32::exp(tile_max - new_max);
        ssum = ssum * prev_corr + tile_sum * curr_corr;
        acc = acc * prev_corr + tile_val * curr_corr;
        m = new_max;

        sync_cube();
        j0 += 128u32;
    }

    out[((qi * n_head + h) * hd + tid) as usize] = acc / ssum;
}

// ── launchers ───────────────────────────────────────────────────────

/// Parameters for [`EdlmAttnMultiCubeCL::launch`] — the attention geometry of
/// one dispatch.
#[cfg(feature = "edlm_gpu")]
#[derive(Clone, Copy, Debug)]
pub struct EdlmAttnMultiParams {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Key positions in the combined `[keys | values]` buffer.
    pub n_positions: usize,
    /// Query rows.
    pub seq_q: usize,
    /// `1 / sqrt(head_dim)` — the caller computes it exactly like the CPU
    /// path (`1.0 / (hd as f32).sqrt()`).
    pub scale: f32,
}

/// Launcher for [`edlm_attn_multi_f32`].
#[cfg(feature = "edlm_gpu")]
pub struct EdlmAttnMultiCubeCL;

#[cfg(feature = "edlm_gpu")]
impl EdlmAttnMultiCubeCL {
    /// Launch the multi-query attention: `out[sq, n_head·hd]` from
    /// `q[sq, n_head·hd]`, `kv[2·n_pos·kvd]` (`[keys | values]`), per-row
    /// visibility `t_n[sq]` (query `i` attends keys `0..t_n(i)`).
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q: Handle,
        kv: Handle,
        t_n: Handle,
        out: Handle,
        p: &EdlmAttnMultiParams,
    ) {
        assert_eq!(
            p.head_dim, 128,
            "the attention cube is head_dim 128 (the 8B release shape); \
             a different head_dim needs a kernel variant"
        );
        assert!(
            p.n_head > 0 && p.n_kv_head > 0,
            "head counts must be positive"
        );
        assert_eq!(
            p.n_head % p.n_kv_head,
            0,
            "GQA mapping needs n_head % n_kv_head == 0"
        );
        assert!(p.seq_q > 0 && p.n_positions > 0, "empty attention dispatch");
        assert!(
            p.scale.is_finite() && p.scale > 0.0,
            "scale must be positive"
        );
        let kvd = p.n_kv_head * p.head_dim;
        // The kernel derives n_pos from the bound kv buffer (the `.issues/515`
        // class guard); the other shapes from the params. t_n's own bounds
        // (vis ≤ n_pos) are the caller's law — the model pass constructs them
        // from the visibility table, the tests pin the violation.
        assert_binding_derives_units(&kv, 2 * kvd, p.n_positions, "EdlmAttnMulti kv");
        assert_binding_derives_units(&q, p.n_head * p.head_dim, p.seq_q, "EdlmAttnMulti q");
        assert_binding_derives_units(&t_n, 1, p.seq_q, "EdlmAttnMulti t_n");
        assert_binding_derives_units(&out, p.n_head * p.head_dim, p.seq_q, "EdlmAttnMulti out");

        let params_h = create_f32(
            client,
            &[
                p.n_head as f32,
                p.n_kv_head as f32,
                p.head_dim as f32,
                p.scale,
            ],
        );

        // SAFETY: buffer sizes asserted above.
        unsafe {
            edlm_attn_multi_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(p.n_head as u32, p.seq_q as u32, 1),
                CubeDim::new_1d(128),
                BufferArg::from_raw_parts(q, p.seq_q * p.n_head * p.head_dim),
                BufferArg::from_raw_parts(kv, 2 * p.n_positions * kvd),
                BufferArg::from_raw_parts(t_n, p.seq_q),
                BufferArg::from_raw_parts(params_h, 4),
                BufferArg::from_raw_parts(out, p.seq_q * p.n_head * p.head_dim),
            );
        }
    }
}

/// Launcher for [`edlm_rmsnorm_rows_f32`].
#[cfg(feature = "edlm_gpu")]
pub struct EdlmRmsNormRowsCubeCL;

#[cfg(feature = "edlm_gpu")]
impl EdlmRmsNormRowsCubeCL {
    /// Launch the multi-row RMSNorm: `out[rows, n]` from `x[rows, n]` and
    /// `gamma[n]` at `eps`.
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x: Handle,
        gamma: Handle,
        out: Handle,
        rows: usize,
        n: usize,
        eps: f32,
    ) {
        assert!(rows > 0 && n > 0, "rmsnorm dims must be positive");
        assert_binding_derives_units(&x, n, rows, "EdlmRmsNormRows x");
        assert_binding_derives_units(&gamma, n, 1, "EdlmRmsNormRows gamma");
        assert_binding_derives_units(&out, n, rows, "EdlmRmsNormRows out");
        let params_h = create_f32(client, &[eps]);

        // SAFETY: buffer sizes asserted above.
        unsafe {
            edlm_rmsnorm_rows_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x, rows * n),
                BufferArg::from_raw_parts(gamma, n),
                BufferArg::from_raw_parts(params_h, 1),
                BufferArg::from_raw_parts(out, rows * n),
            );
        }
    }
}

// ── the model ───────────────────────────────────────────────────────

/// One layer's GPU-resident weights (f16) + the host-side qk-norm gammas the
/// hybrid pipeline applies between readbacks.
#[cfg(feature = "edlm_gpu")]
struct EdlmGpuLayerWeights {
    /// `[(n_head + 2·n_kv_head)·hd, n_embd]` f16 — the q|k|v row-concat of
    /// attn_q/attn_k/attn_v (one GEMM, one readback).
    qkv: Handle,
    /// `[n_embd, n_head·hd]` f16 (attn_output).
    wo: Handle,
    /// `[2·mlp_hidden, n_embd]` f16 — the gate|up row-concat (one GEMM, one
    /// readback).
    gateup: Handle,
    /// `[n_embd, mlp_hidden]` f16 (ffn_down).
    down: Handle,
    /// `[n_embd]` f32 (attn_norm — the GPU rmsnorm's gamma).
    attn_norm: Handle,
    /// `[n_embd]` f32 (ffn_norm — the GPU rmsnorm's gamma).
    post_attn_norm: Handle,
    /// `[head_dim]` f32 host copy (attn_q_norm — CPU qk-norm between readbacks).
    q_norm: Vec<f32>,
    /// `[head_dim]` f32 host copy (attn_k_norm).
    k_norm: Vec<f32>,
}

/// The GPU eDLM model: f16-resident weights on the CubeCL runtime + the CPU
/// state-prefix cache (T7's `EdlmStateKv`), serving the SAME API shape the
/// CPU streaming lane does (`state_prefill` → `forward_branches`).
///
/// Construct via [`EdlmGpuModel::open`] (GGUF, one layer dequanted at a
/// time) or [`EdlmGpuModel::from_weights`] (in-memory — the parity-test
/// lane). Both share the f16 upload conversion.
#[cfg(feature = "edlm_gpu")]
pub struct EdlmGpuModel {
    client: ComputeClient<ActiveRuntime>,
    pub config: Config,
    /// Host embedding table (rows are gathered CPU-side and uploaded — the
    /// full table never goes VRAM; only the sequence's rows are needed).
    wte: Vec<f32>,
    final_norm: Vec<f32>,
    pub pointer: Option<EdlmPointerHead>,
    layers: Vec<EdlmGpuLayerWeights>,
    freq: RopeFreqTable,
    scale: f32,
    /// The cached state prefix (CPU K/V + final-normed state hiddens) —
    /// filled by [`EdlmGpuModel::state_prefill`], consumed by
    /// [`EdlmGpuModel::forward_branches`].
    state: Option<EdlmStateKv>,
    /// The GEMM posture: `true` routes the four per-layer projections
    /// through the cooperative-matrix (tensor-core) kernel
    /// ([`crate::MatmulF16bCmmaCubeCL`]), `false` keeps the scalar tiled
    /// kernel. Resolved once at construction from `EDLM_GPU_CMMA`
    /// (`"0"` = scalar; unset/anything else = the measured default),
    /// overridable per-instance by the test seam.
    use_cmma: bool,
}

/// Convert a host f32 slice to f16 bytes (round-to-nearest — the documented
/// numerics face of the f16 residency decision).
#[cfg(feature = "edlm_gpu")]
fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let f16v: Vec<half_f16> = if data.len() < 4096 {
        data.iter().map(|&v| half_f16::from_f32(v)).collect()
    } else {
        use rayon::prelude::*;
        data.par_iter().map(|&v| half_f16::from_f32(v)).collect()
    };
    bytemuck::cast_slice::<half_f16, u8>(&f16v).to_vec()
}

/// The env kill-switch for the tensor-core GEMM posture, read ONCE per
/// process. `EDLM_GPU_CMMA=0` restores the scalar tiled kernel; unset or
/// any other value takes the measured default (CMMA — the GOAT lane's
/// verdict lives in `tests/edlm_matmul_cmma_goat.rs` + the issue row).
#[cfg(feature = "edlm_gpu")]
fn cmma_env_default() -> bool {
    match std::env::var("EDLM_GPU_CMMA") {
        Ok(v) => v != "0",
        Err(_) => true,
    }
}

#[cfg(feature = "edlm_gpu")]
impl EdlmGpuModel {
    /// Open an `edlm`-arch GGUF and upload every layer as f16 (one layer
    /// dequanted at a time — the streaming loader's host-footprint law; the
    /// Q8_0 dequant is exact, the f16 conversion is the rounding face).
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let core = EdlmGgufModel::open(path).map_err(|e| e.to_string())?;
        Self::from_core(core)
    }

    /// In-memory constructor — the parity-test lane: the SAME `EdlmWeights`
    /// the CPU oracle runs, uploaded f16.
    pub fn from_weights(weights: &EdlmWeights, config: &Config) -> Result<Self, String> {
        if config.head_dim != 128 {
            return Err(format!(
                "EdlmGpuModel needs head_dim 128 (the attention cube's shape), got {}",
                config.head_dim
            ));
        }
        let ctx = CubeCLContext::new().map_err(|e| format!("CubeCL context: {e:?}"))?;
        let client = ctx.client();

        let layers = weights
            .layers
            .iter()
            .map(|l| {
                let mut qkv = l.base.attn_wq.clone();
                qkv.extend_from_slice(&l.base.attn_wk);
                qkv.extend_from_slice(&l.base.attn_wv);
                let mut gateup = l.base.gate_proj.clone();
                gateup.extend_from_slice(&l.base.up_proj);
                Ok(EdlmGpuLayerWeights {
                    qkv: upload_f16(&client, &qkv)?,
                    wo: upload_f16(&client, &l.base.attn_wo)?,
                    gateup: upload_f16(&client, &gateup)?,
                    down: upload_f16(&client, &l.base.down_proj)?,
                    attn_norm: create_f32(&client, &l.base.input_norm),
                    post_attn_norm: create_f32(&client, &l.base.post_attn_norm),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            client,
            config: config.clone(),
            wte: weights.wte.clone(),
            final_norm: weights.final_norm.clone(),
            pointer: weights.pointer.clone(),
            layers,
            freq: RopeFreqTable::new(config.rope_theta, config.head_dim),
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            state: None,
            use_cmma: cmma_env_default(),
        })
    }

    fn from_core(core: EdlmGgufModel) -> Result<Self, String> {
        let config = core.config.clone();
        if config.head_dim != 128 {
            return Err(format!(
                "EdlmGpuModel needs head_dim 128 (the attention cube's shape), got {}",
                config.head_dim
            ));
        }
        let ctx = CubeCLContext::new().map_err(|e| format!("CubeCL context: {e:?}"))?;
        let client = ctx.client();

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let deq = |name: String| -> Result<Vec<f32>, String> {
                core.gguf
                    .dequant_f16_to_f32(&name)
                    .map_err(|e| format!("tensor {name}: {e}"))
            };
            let wq = deq(format!("blk.{i}.attn_q.weight"))?;
            let wk = deq(format!("blk.{i}.attn_k.weight"))?;
            let wv = deq(format!("blk.{i}.attn_v.weight"))?;
            let wo = deq(format!("blk.{i}.attn_output.weight"))?;
            let gate = deq(format!("blk.{i}.ffn_gate.weight"))?;
            let up = deq(format!("blk.{i}.ffn_up.weight"))?;
            let down = deq(format!("blk.{i}.ffn_down.weight"))?;
            let attn_norm = deq(format!("blk.{i}.attn_norm.weight"))?;
            let post_attn_norm = deq(format!("blk.{i}.ffn_norm.weight"))?;
            let q_norm = deq(format!("blk.{i}.attn_q_norm.weight"))?;
            let k_norm = deq(format!("blk.{i}.attn_k_norm.weight"))?;
            if q_norm.len() != config.head_dim || k_norm.len() != config.head_dim {
                return Err(format!(
                    "layer {i}: QK-norm gamma length {}/{} != head_dim {}",
                    q_norm.len(),
                    k_norm.len(),
                    config.head_dim
                ));
            }
            let mut qkv = wq;
            qkv.extend_from_slice(&wk);
            qkv.extend_from_slice(&wv);
            let mut gateup = gate;
            gateup.extend_from_slice(&up);
            layers.push(EdlmGpuLayerWeights {
                qkv: upload_f16(&client, &qkv)?,
                wo: upload_f16(&client, &wo)?,
                gateup: upload_f16(&client, &gateup)?,
                down: upload_f16(&client, &down)?,
                attn_norm: create_f32(&client, &attn_norm),
                post_attn_norm: create_f32(&client, &post_attn_norm),
                q_norm,
                k_norm,
            });
        }

        let freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        Ok(Self {
            client,
            config,
            wte: core.wte,
            final_norm: core.final_norm,
            pointer: core.pointer,
            layers,
            freq,
            scale,
            state: None,
            use_cmma: cmma_env_default(),
        })
    }

    /// The runtime's name — provenance for every latency figure quoted
    /// beside this lane (box state is part of the claim).
    pub fn runtime_name(&self) -> &'static str {
        ActiveRuntime::name(&self.client)
    }

    /// The GEMM posture this instance dispatches (provenance for parity
    /// rows: the two postures are tolerance-equivalent, never identical).
    pub fn matmul_posture(&self) -> &'static str {
        if self.use_cmma {
            "cmma"
        } else {
            "scalar"
        }
    }

    /// Test seam: pin the GEMM posture explicitly (the env default is
    /// process-global; parity tests need BOTH postures in one process).
    #[cfg(test)]
    fn with_matmul_posture(mut self, use_cmma: bool) -> Self {
        self.use_cmma = use_cmma;
        self
    }

    /// The one dispatch site for the lane's four per-layer projections.
    /// Shape law (the GOAT lane's interleaved medians, wgpu-spirv, 4090):
    /// cmma loses at M=8 (0.60× — staging ALU over dead tile rows) and wins
    /// from M≈16 (1.09×) upward (1.51× at 18, 3.4× at 87, 8× at 512) — so
    /// short branch rows stay scalar and everything prefill-class goes
    /// tensor-core. `EDLM_GPU_CMMA=0` forces scalar for ALL m.
    fn matmul_f16b(
        &self,
        a: Handle,
        b: Handle,
        out: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        if self.use_cmma && m >= 16 {
            crate::MatmulF16bCmmaCubeCL::launch::<ActiveRuntime>(&self.client, a, b, out, m, n, p);
        } else {
            MatmulF16bCubeCL::launch::<ActiveRuntime>(&self.client, a, b, out, m, n, p);
        }
    }

    /// Release fully-free GPU pool pages back to the driver (the Issue-712
    /// law: freed slices stay committed inside their pages otherwise).
    pub fn memory_cleanup(&self) {
        self.client.memory_cleanup();
    }

    /// Run the state tokens only and capture their per-layer K/V + the
    /// final-normed state hiddens — the T7 prefill, on the GPU. The state's
    /// own eligibility is `branch_mask` over an all-state segment: causal,
    /// widened to state↔state bidirectional when `state_bidir`.
    pub fn state_prefill(
        &mut self,
        state_ids: &[usize],
        state_pos: &[usize],
        state_bidir: bool,
    ) -> Result<(), String> {
        let sl = state_ids.len();
        assert_eq!(state_pos.len(), sl, "state_pos must match state_ids");
        assert!(
            sl > 0 && sl <= self.config.block_size,
            "state len {sl} out of range"
        );
        let (h, layers) = self.forward_rows(state_ids, state_pos, None, state_bidir)?;
        let mut hidden = h;
        let n = self.config.n_embd;
        for chunk in hidden.chunks_exact_mut(n) {
            rmsnorm_with_gamma_eps(chunk, &self.final_norm, self.config.rms_norm_eps);
        }
        self.state = Some(EdlmStateKv {
            state_len: sl,
            kvd: kv_dim(&self.config),
            state_ids: state_ids.to_vec(),
            layers,
            hidden,
        });
        Ok(())
    }

    /// The cached state prefix — parity inspection + the pointer head's
    /// state-side read (same shape the CPU lane returns).
    pub fn state_kv(&self) -> Result<&EdlmStateKv, String> {
        self.state
            .as_ref()
            .ok_or_else(|| "no state cache: call state_prefill first".to_string())
    }

    /// Branch rows continuing the cached state prefix — the T7 continuation,
    /// on the GPU. Output = the CPU `forward_edlm_branches_cached` shape:
    /// per-row `[(state hiddens | branch hiddens)]` post-final-norm, the
    /// pointer head reads markers from it directly.
    pub fn forward_branches(
        &mut self,
        enc: &PackedEncoding,
        rows: &[BranchRow],
    ) -> Result<Vec<Vec<f32>>, String> {
        let cache = self
            .state
            .as_ref()
            .ok_or_else(|| "no state cache: call state_prefill first".to_string())?;
        cache.check_matches(&self.config, &enc.ids[..enc.state_len]).map_err(|e| e.to_string())?;
        let n = self.config.n_embd;
        let sl = enc.state_len;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let ids = &enc.ids[row.start..row.end];
            let pos = &enc.pos[row.start..row.end];
            let (h, _) = self.forward_rows(ids, pos, Some(&cache.layers), false)?;
            let mut hidden = h;
            for chunk in hidden.chunks_exact_mut(n) {
                rmsnorm_with_gamma_eps(chunk, &self.final_norm, self.config.rms_norm_eps);
            }
            let mut full = Vec::with_capacity((sl + row.branch_len()) * n);
            full.extend_from_slice(&cache.hidden);
            full.extend_from_slice(&hidden);
            out.push(full);
        }
        Ok(out)
    }

    /// One pass over `ids` (query rows) with optional state-KV seeding.
    /// Returns the raw (pre-final-norm) hiddens `[sq · n_embd]` and — when
    /// `seed` is None (the state pass) — THIS pass's own per-layer K/V.
    ///
    /// Eligibility: with a seed, query `i` attends `0..sl + i + 1` over
    /// `[state | own]` (the continuation law); without one, all `sq` keys
    /// when `state_bidir` (the state's bidir block) else `i + 1` (causal).
    fn forward_rows(
        &self,
        ids: &[usize],
        pos: &[usize],
        seed: Option<&[EdlmLayerKv]>,
        state_bidir: bool,
    ) -> Result<(Vec<f32>, Vec<EdlmLayerKv>), String> {
        let config = self.config.clone();
        let n = config.n_embd;
        let hd = config.head_dim;
        let q_dim = config.n_head * hd;
        let kvd = kv_dim(&config);
        let n_kv = config.n_kv_head;
        let mlp = config.mlp_hidden;
        let n_layer = config.n_layer;
        let eps = config.rms_norm_eps as f32;
        let sq = ids.len();
        assert_eq!(pos.len(), sq, "pos must match ids");
        assert!(
            sq > 0 && sq <= config.block_size,
            "row len {sq} out of range"
        );

        let sl = seed.map_or(0, |s| {
            let first = s.first().map_or(0, |l| l.k.len() / kvd);
            for (li, l) in s.iter().enumerate() {
                assert_eq!(l.k.len(), first * kvd, "seed layer {li} length drift");
            }
            first
        });
        let total = sl + sq;

        // The visibility law (the module doc's t_n table).
        let t_n: Vec<u32> = if state_bidir && sl == 0 {
            vec![total as u32; sq]
        } else {
            (0..sq).map(|i| (sl + i + 1) as u32).collect()
        };
        let attn_params = EdlmAttnMultiParams {
            n_head: config.n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: total,
            seq_q: sq,
            scale: self.scale,
        };

        // Embedding rows on the host (no wte upload — only these rows exist
        // on the device).
        let mut h = vec![0.0f32; sq * n];
        for (r, &id) in ids.iter().enumerate() {
            let off = id * n;
            h[r * n..(r + 1) * n].copy_from_slice(&self.wte[off..off + n]);
        }

        // Reused staging (allocated once per pass, sized to the pass).
        let lq = q_dim + 2 * kvd;
        let mut q_rope = vec![0.0f32; sq * q_dim];
        let mut keys = vec![0.0f32; total * kvd];
        let mut values = vec![0.0f32; total * kvd];
        let mut kv_capture: Vec<EdlmLayerKv> = Vec::with_capacity(n_layer);
        let capture = seed.is_none();

        for li in 0..n_layer {
            let lw = &self.layers[li];

            // ── attention block ──
            let xr = h.clone();
            let h_h = create_f32(&self.client, &h);
            let hn1 = self.client.empty(sq * n * core::mem::size_of::<f32>());
            EdlmRmsNormRowsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                h_h,
                lw.attn_norm.clone(),
                hn1.clone(),
                sq,
                n,
                eps,
            );
            let qkv_out = self.client.empty(sq * lq * core::mem::size_of::<f32>());
            self.matmul_f16b(hn1, lw.qkv.clone(), qkv_out.clone(), sq, n, lq);
            let qkv_h = read_f32(&self.client, qkv_out).map_err(|e| e.to_string())?;

            // Host small-ops: split, per-head qk-norm, RoPE — the EXACT core
            // helpers the CPU path runs (parity carries; no re-derivation).
            for r in 0..sq {
                let row = &qkv_h[r * lq..(r + 1) * lq];
                let mut q_row = row[..q_dim].to_vec();
                qk_norm_inplace(
                    &mut q_row,
                    &lw.q_norm,
                    config.n_head,
                    hd,
                    config.rms_norm_eps,
                );
                apply_rope_with_freq(&mut q_row, &mut [], pos[r], hd, self.freq.as_slice());
                q_rope[r * q_dim..(r + 1) * q_dim].copy_from_slice(&q_row);

                let mut k_row = row[q_dim..q_dim + kvd].to_vec();
                qk_norm_inplace(&mut k_row, &lw.k_norm, n_kv, hd, config.rms_norm_eps);
                apply_rope_with_freq(&mut k_row, &mut [], pos[r], hd, self.freq.as_slice());
                keys[(sl + r) * kvd..(sl + r + 1) * kvd].copy_from_slice(&k_row);

                // V: untouched by qk-norm and RoPE.
                values[(sl + r) * kvd..(sl + r + 1) * kvd]
                    .copy_from_slice(&row[q_dim + kvd..lq]);
            }
            if let Some(seed_layers) = seed {
                keys[..sl * kvd].copy_from_slice(&seed_layers[li].k);
                values[..sl * kvd].copy_from_slice(&seed_layers[li].v);
            }

            let mut kv_combined = Vec::with_capacity(2 * total * kvd);
            kv_combined.extend_from_slice(&keys);
            kv_combined.extend_from_slice(&values);
            let kv_h = create_f32(&self.client, &kv_combined);
            let q_h = create_f32(&self.client, &q_rope);
            let tn_h = create_u32(&self.client, &t_n);
            let attn_h = self.client.empty(sq * q_dim * core::mem::size_of::<f32>());
            EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
                &self.client,
                q_h,
                kv_h,
                tn_h,
                attn_h.clone(),
                &attn_params,
            );

            let wo_out = self.client.empty(sq * n * core::mem::size_of::<f32>());
            self.matmul_f16b(attn_h, lw.wo.clone(), wo_out.clone(), sq, q_dim, n);
            let wo_h = read_f32(&self.client, wo_out).map_err(|e| e.to_string())?;
            for i in 0..sq * n {
                h[i] = xr[i] + wo_h[i];
            }

            // ── MLP block ──
            let h2_h = create_f32(&self.client, &h);
            let hn2 = self.client.empty(sq * n * core::mem::size_of::<f32>());
            EdlmRmsNormRowsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                h2_h,
                lw.post_attn_norm.clone(),
                hn2.clone(),
                sq,
                n,
                eps,
            );
            let gu_out = self.client.empty(sq * 2 * mlp * core::mem::size_of::<f32>());
            self.matmul_f16b(hn2, lw.gateup.clone(), gu_out.clone(), sq, n, 2 * mlp);
            let gu = read_f32(&self.client, gu_out).map_err(|e| e.to_string())?;
            let mut mlp_in = vec![0.0f32; sq * mlp];
            for r in 0..sq {
                let row = &gu[r * 2 * mlp..(r + 1) * 2 * mlp];
                swiglu(
                    &mut mlp_in[r * mlp..(r + 1) * mlp],
                    &row[..mlp],
                    &row[mlp..2 * mlp],
                );
            }
            let mi_h = create_f32(&self.client, &mlp_in);
            let down_out = self.client.empty(sq * n * core::mem::size_of::<f32>());
            self.matmul_f16b(mi_h, lw.down.clone(), down_out.clone(), sq, mlp, n);
            let d_h = read_f32(&self.client, down_out).map_err(|e| e.to_string())?;
            for i in 0..sq * n {
                h[i] += d_h[i];
            }

            if capture {
                kv_capture.push(EdlmLayerKv {
                    k: keys[..sq * kvd].to_vec(),
                    v: values[..sq * kvd].to_vec(),
                });
            }
        }

        Ok((h, kv_capture))
    }
}

/// Upload a host f32 slice as an f16 device buffer (round-to-nearest).
#[cfg(feature = "edlm_gpu")]
fn upload_f16(client: &ComputeClient<ActiveRuntime>, data: &[f32]) -> Result<Handle, String> {
    let bytes = f32_to_f16_bytes(data);
    Ok(client.create_from_slice(&bytes))
}

#[cfg(all(test, feature = "edlm_gpu"))]
mod tests;
