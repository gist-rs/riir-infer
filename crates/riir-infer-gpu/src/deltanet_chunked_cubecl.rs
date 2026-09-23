//! DeltaNet chunkwise parallel prefill kernels (Issue 652 / Plan 533).
//!
//! Implements the chunkwise parallel algorithm (Yang & Wang 2024) for the
//! DeltaNet recurrence on GPU. The sequential per-token prefill path dispatches
//! P recurrence + P conv1d kernels per layer (4096 sequential dispatches at
//! P=128, 32 layers) — this module collapses those to O(P/C) chunk-boundary
//! transitions + intra-chunk batched matmuls.
//!
//! ## Phases (Plan 533)
//!
//! - **Phase 1** (this file, initial): chunked conv1d. Proves the chunked-
//!   dispatch pattern on the simpler state-carrying op. Reduces P conv1d
//!   dispatches to ceil(P/C).
//! - **Phase 2-4** (TODO): chunked recurrence — intra-chunk parallel matmul +
//!   inter-chunk state transition. The big lever (the recurrence is the
//!   structural limit identified in Issue 637 T5).

use cubecl::prelude::*;
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Phase 1 — Chunked conv1d
// ---------------------------------------------------------------------------

/// CubeCL kernel: depthwise conv1d + SiLU over a chunk of `c` tokens in one dispatch.
///
/// The single-token kernel (`deltanet_conv1d_f32`) shifts the conv_state left by
/// 1, appends the new input, convolves, and applies SiLU — all in-place on the
/// input buffer. Processing C tokens sequentially requires C dispatches because
/// each token reads the previous token's conv_state.
///
/// The chunked kernel processes C tokens in one dispatch. The key correctness
/// constraint: token t's convolution needs the RAW (pre-SiLU) inputs of tokens
/// t-1, t-2, t-3. If those were overwritten in-place (as the single-token kernel
/// does), the convolution would read SiLU'd values (incorrect). So the chunked
/// kernel uses SEPARATE read-only input + write-only output buffers.
///
/// ## Layout
///
/// - `input`: `[c, conv_dim]` row-major — READ-ONLY raw inputs for the C tokens.
/// - `output`: `[c, conv_dim]` row-major — WRITE-ONLY SiLU convolution outputs.
/// - `conv_weight`: `[conv_dim, kernel_size]` row-major.
/// - `carry`: `[conv_dim, kernel_size - 1]` — the last `kernel_size - 1` RAW
///   inputs from the previous chunk (or the initial conv_state, which holds raw
///   inputs). READ-ONLY here: the carry update runs as a SEPARATE dispatch
///   after this kernel (see `deltanet_conv1d_carry_update_f32`, Issue 673).
/// - `params`: `[c_f32, conv_dim_f32, kernel_size_f32]`.
///
/// ## Dispatch
///
/// One thread per (token, channel) pair: `CubeCount::Static(ceil(c * conv_dim /
/// 256), 1, 1)`, `CubeDim::new_1d(256)`.
///
/// ## Correctness
///
/// Bit-identical to `c` sequential `deltanet_conv1d_f32` calls when the carry is
/// correctly initialized. The arithmetic per output element is identical (same
/// convolution sum + same SiLU); only the data sourcing (chunk window vs shifted
/// state) differs. The carry update is deliberately NOT done in this kernel:
/// within one dispatch there is no ordering between the conv threads that READ
/// the carry (tokens `[0, ks-1)`) and the update threads that WRITE it, so an
/// in-kernel update is a data race (Issue 673).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_conv1d_chunked_f32(
    input: &[f32],
    output: &mut [f32],
    conv_weight: &[f32],
    carry: &[f32],
    params: &[f32],
) {
    let c = params[0usize] as usize;
    let conv_dim = params[1usize] as usize;
    let kernel_size = params[2usize] as usize;
    // params[3] = carry_stride: the per-channel stride in the carry buffer.
    //   - kernel_size - 1: the original chunked-carry layout (last ks-1 raw inputs).
    //   - kernel_size: the conv_state layout (full ks-wide sliding window; the
    //     chunked kernel reads/writes positions 1..ks-1, leaving position 0
    //     untouched — it holds a stale value that the next decode's shift
    //     overwrites before use). Set via carry_idx_offset below.
    let carry_stride = params[3usize] as usize;
    // params[4] = carry_idx_offset: 0 for the original layout, 1 for conv_state.
    let carry_idx_offset = params[4usize] as usize;
    let ks_m1 = kernel_size - 1;
    let idx = ABSOLUTE_POS;

    // idx decomposes into (token, channel): idx = token * conv_dim + channel.
    let total = c * conv_dim;
    if idx >= total {
        terminate!();
    }

    let token = idx / conv_dim;
    let ch = idx % conv_dim;
    let weight_off = ch * kernel_size;
    let carry_off = ch * carry_stride + carry_idx_offset;

    // Build the convolution window for this token: kernel_size samples ending at
    // `token`. Samples before token 0 come from the carry (previous chunk's tail).
    //
    // For kernel_size=4, token t needs samples at positions [t-3, t-2, t-1, t].
    // The carry holds positions [-3, -2, -1] (the previous chunk's last 3 tokens).
    // Position p >= 0 is `input[p * conv_dim + ch]`.
    //
    // The convolution: out = Σ_{k=0}^{ks-1} window[ks-1-k] * weight[ch, k]
    // (weight[0] multiplies the oldest sample, weight[ks-1] the newest — matching
    // the single-token kernel where conv_state[0] is the oldest after the shift).
    let mut sum = f32::new(0.0f32);
    for k in 0..kernel_size {
        // The k-th sample from oldest: position = token - ks_m1 + k.
        // k=0 → oldest (position token - ks_m1); k=ks_m1 → newest (position token).
        let sample_pos_signed = token as i64 - ks_m1 as i64 + k as i64;
        let val = if sample_pos_signed < 0 {
            // From carry: position -(ks_m1) maps to carry index 0, position -1
            // maps to carry index ks_m1-1. So carry index = sample_pos + ks_m1.
            carry[carry_off + (sample_pos_signed + ks_m1 as i64) as usize]
        } else {
            let sample_pos = sample_pos_signed as usize;
            input[sample_pos * conv_dim + ch]
        };
        sum += val * conv_weight[weight_off + k];
    }

    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    let neg_sum = f32::new(0.0f32) - sum;
    let exp_neg = neg_sum.exp();
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + exp_neg);
    output[idx] = sum * sig;
}

/// Carry update for the chunked conv1d: shift the carry left by `c` and append
/// the chunk's last `min(c, ks-1)` raw inputs (Issue 673 Bug C fix).
///
/// Correct merge (matching the CPU reference):
///   new_carry = [old_carry[c..ks-1], new_inputs[0..c]]
/// — for full chunks (`c >= ks-1`) the whole carry is the last `ks-1` inputs;
///   for partial chunks (`c < ks-1`) the old carry's tail shifts to the front.
///
/// Dispatched as a SEPARATE dispatch AFTER `deltanet_conv1d_chunked_f32`:
/// within the conv kernel's dispatch there is no ordering between the conv
/// threads READING the carry and update threads WRITING it — an in-kernel
/// update is a data race. Dispatch order on the compute stream guarantees the
/// conv reads complete first.
///
/// One thread per CHANNEL: each thread owns its channel's carry row
/// exclusively, doing the left-shift serially in-thread (ascending `j` reads
/// `j+c` before any later iteration's write clobbers it — program order within
/// a thread is respected; no cross-thread carry access).
///
/// ## Layout (same buffers as the conv kernel)
/// - `input`: `[c, conv_dim]` — READ-ONLY raw inputs.
/// - `carry`: `[conv_dim, carry_stride]` — IN-OUT. `carry_stride`/
///   `carry_idx_offset` select the compact (`ks-1`, 0) or conv_state
///   (`ks`, 1) layout, exactly as in the conv kernel.
/// - `params`: `[c_f32, conv_dim_f32, kernel_size_f32, carry_stride_f32,
///   carry_idx_offset_f32]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_conv1d_carry_update_f32(
    input: &[f32],
    carry: &mut [f32],
    params: &[f32],
) {
    let c = params[0usize] as usize;
    let conv_dim = params[1usize] as usize;
    let kernel_size = params[2usize] as usize;
    let carry_stride = params[3usize] as usize;
    let carry_idx_offset = params[4usize] as usize;
    let ks_m1 = kernel_size - 1;
    let ch = ABSOLUTE_POS;

    if ch >= conv_dim {
        terminate!();
    }

    let carry_off = ch * carry_stride + carry_idx_offset;

    for j in 0..ks_m1 {
        let src = j + c;
        if src < ks_m1 {
            // Old carry slot still in the window — shifts left by c.
            carry[carry_off + j] = carry[carry_off + src];
        } else {
            // New input token (src - ks_m1 = j + c - (ks-1) ∈ [0, c)).
            let token = src - ks_m1;
            carry[carry_off + j] = input[token * conv_dim + ch];
        }
    }
}

/// Launcher for the chunked conv1d kernel (Phase 1, Plan 533).
///
/// # Buffer contract
///
/// Unlike the single-token `DeltanetConv1dCubeCL` which operates in-place on
/// `input` (reading the raw input, writing SiLU back), the chunked kernel
/// requires SEPARATE input and output buffers because token t's convolution
/// needs the raw (pre-SiLU) inputs of tokens t-1, t-2, t-3.
///
/// - `input_handle`: `[c, conv_dim]` — READ-ONLY raw inputs for the C tokens.
/// - `output_handle`: `[c, conv_dim]` — SiLU convolution outputs (written).
/// - `conv_weight_handle`: `[conv_dim, kernel_size]` — depthwise weights.
/// - `carry_handle`: `[conv_dim, carry_stride]` — IN-OUT. On entry: the last
///   `kernel_size - 1` raw inputs from the previous chunk (or initial state).
///   On exit: the last `min(c, kernel_size - 1)` raw inputs of THIS chunk for
///   the next chunk (partial chunks shift the old carry's tail to the front —
///   Issue 673 Bug C). When `carry_stride == kernel_size - 1` +
///   `carry_idx_offset == 0`, this is the compact chunked-carry layout. When
///   `carry_stride == kernel_size` + `carry_idx_offset == 1`, this matches the
///   conv_state sliding-window layout used by the decode path — positions
///   1..ks-1 are read/written, position 0 is untouched (stale, shifted out on
///   next decode).
///
/// Issues TWO ordered dispatches (Issue 673): (1) the conv kernel, which only
/// READS the carry, and (2) the carry-update kernel, which rewrites it. This
/// removes the intra-dispatch read/write race the single-kernel design had and
/// fixes the partial-chunk merge.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetChunkedConv1dCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetChunkedConv1dCubeCL {
/// Launch the chunked conv1d kernel.
///
/// # Safety
///
/// Caller must ensure:
/// - `input_handle` has `c * conv_dim` f32 elements (raw inputs, read-only).
/// - `output_handle` has `c * conv_dim` f32 elements (SiLU outputs, written).
/// - `conv_weight_handle` has `conv_dim * kernel_size` f32 elements.
/// - `carry_handle` has `conv_dim * carry_stride` f32 elements.
/// - `carry_stride` is `kernel_size - 1` (compact) or `kernel_size` (conv_state).
/// - `carry_idx_offset` is 0 (compact) or 1 (conv_state).
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
pub unsafe fn launch<R: Runtime>(
    client: &ComputeClient<R>,
    input_handle: Handle,
    output_handle: Handle,
    conv_weight_handle: Handle,
    carry_handle: Handle,
    c: usize,
    conv_dim: usize,
    kernel_size: usize,
    carry_stride: usize,
    carry_idx_offset: usize,
) {
    let params: [f32; 5] = [
        c as f32,
        conv_dim as f32,
        kernel_size as f32,
        carry_stride as f32,
        carry_idx_offset as f32,
    ];
    let params_handle = client.create_from_slice(f32::as_bytes(&params));
    let total = c * conv_dim;
    let wg = 256usize;
    let n_wg = total.div_ceil(wg).max(1) as u32;

    unsafe {
        deltanet_conv1d_chunked_f32::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_wg, 1, 1),
            CubeDim::new_1d(wg as u32),
            BufferArg::from_raw_parts(input_handle.clone(), total),
            BufferArg::from_raw_parts(output_handle, total),
            BufferArg::from_raw_parts(conv_weight_handle, conv_dim * kernel_size),
            BufferArg::from_raw_parts(carry_handle.clone(), conv_dim * carry_stride),
            BufferArg::from_raw_parts(params_handle.clone(), 5),
        );

        // Carry update as a separate ordered dispatch (Issue 673 Bug C + the
        // intra-dispatch carry read/write race): the conv kernel only READS
        // the carry; this dispatch rewrites it AFTER those reads complete.
        // One thread per channel — the per-channel left-shift is serial
        // in-thread, so no cross-thread carry access.
        let n_wg_carry = (conv_dim as u32).div_ceil(wg as u32).max(1);
        deltanet_conv1d_carry_update_f32::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_wg_carry, 1, 1),
            CubeDim::new_1d(wg as u32),
            BufferArg::from_raw_parts(input_handle, total),
            BufferArg::from_raw_parts(carry_handle, conv_dim * carry_stride),
            BufferArg::from_raw_parts(params_handle, 5),
        );
    }
}
}

// ---------------------------------------------------------------------------
// Phase 2 — Chunked recurrence: cumulative decay + weighted V + state contribution
// ---------------------------------------------------------------------------
//
// The DeltaNet recurrence per head: S_t = α_t·S_{t-1} + β_t·v_t⊗k_t^T
//
// Unrolling over a chunk [0..C-1] with boundary state S_{-1}:
//   S_t = (Π_{i=0}^{t} α_i)·S_{-1} + Σ_{j=0}^{t} (Π_{i=j+1}^{t} α_i)·β_j·v_j⊗k_j^T
//
// The chunk-end state (for the next chunk's boundary):
//   S_{C-1} = total_decay·S_{-1} + ΔS
// where total_decay = Π_{i=0}^{C-1} α_i and
//   ΔS = Σ_{j=0}^{C-1} (Π_{i=j+1}^{C-1} α_i)·β_j·v_j⊗k_j^T
//       = V_weighted^T @ K   (a [d,C]×[C,d]→[d,d] matmul)
// where V_weighted[j,:] = (Π_{i=j+1}^{C-1} α_i)·β_j·v_j[:]
//
// The output for token t:
//   o_t = q_t^T·S_t / sqrt(d)
//       = (Π_{i=0}^{t} α_i)·q_t^T·S_{-1}/sqrt(d) + Σ_{j≤t} decay_tj·β_j·(q_t·v_j)·k_j/sqrt(d)
//
// The first term is the cross-chunk contribution (Q @ S_boundary).
// The second term is the intra-chunk contribution (causal-weighted QV·K).
// ---------------------------------------------------------------------------

/// Compute per-token cumulative decay factors for the chunk-end state transition.
///
/// For each token j in [0, C-1], computes:
///   decay_to_end[j] = Π_{i=j+1}^{C-1} α_i   (the decay from j+1 to the chunk end)
///   total_decay = Π_{i=0}^{C-1} α_i          (the full chunk decay, written to output[C])
///
/// Layout:
/// - `alpha`: `[n_head, C]` — per-head per-token decay scalars (α_t = exp(-softplus(...))).
/// - `decay_to_end`: `[n_head, C+1]` — per-head decay factors. The last element
///   per head (index C) holds total_decay = Π_{i=0}^{C-1} α_i.
/// - `params`: `[n_head_f32, C_f32]`.
///
/// This is a reverse cumulative product per head. Implemented as a single
/// elementwise dispatch — each thread computes one decay_to_end value by looping
/// over the relevant α's. For C=64 this is 64 multiplies per thread, acceptable
/// for the one-per-chunk-call cost.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_chunk_decay_f32(
    alpha: &[f32],
    decay_to_end: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * (c + 1);
    if idx >= total {
        terminate!();
    }

    let h = idx / (c + 1);
    let j = idx % (c + 1);
    let alpha_base = h * c;

    // decay_to_end[h, j] for j < C = Π_{i=j+1}^{C-1} α[h, i]
    // decay_to_end[h, C] = Π_{i=0}^{C-1} α[h, i] = total_decay
    let mut product = f32::new(1.0f32);
    if j < c {
        // Product from i=j+1 to C-1
        let mut i = j + 1;
        while i < c {
            product *= alpha[alpha_base + i];
            i += 1;
        }
    } else {
        // j == C: total product from i=0 to C-1
        let mut i = 0;
        while i < c {
            product *= alpha[alpha_base + i];
            i += 1;
        }
    }
    decay_to_end[idx] = product;
}

/// Launcher for the chunk decay kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetChunkDecayCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetChunkDecayCubeCL {
    /// Compute per-head decay_to_end factors for state transition.
    ///
    /// # Safety
    /// - `alpha_handle`: `n_head * C` f32 elements.
    /// - `decay_to_end_handle`: `n_head * (C + 1)` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        alpha_handle: Handle,
        decay_to_end_handle: Handle,
        n_head: usize,
        c: usize,
    ) {
        let params: [f32; 2] = [n_head as f32, c as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * (c + 1);
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_chunk_decay_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(alpha_handle, n_head * c),
                BufferArg::from_raw_parts(decay_to_end_handle, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Split variant of chunk decay: writes per-token decay [n_head, C] and total
/// decay [n_head] to SEPARATE buffers (no readback needed for production use).
///
/// This avoids the [n_head, C+1] packed layout that the original
/// `DeltanetChunkDecayCubeCL` produces, which requires a CPU readback + re-upload
/// to extract the two components. The split variant writes directly to the
/// layouts expected by `DeltanetDeltaSCubeCL` (reads [n_head, C]) and
/// `DeltanetStateTransitionCubeCL` (reads [n_head]).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_chunk_decay_split_f32(
    alpha: &[f32],
    decay_to_end: &mut [f32],
    total_decay: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * (c + 1);
    if idx >= total {
        terminate!();
    }

    let h = idx / (c + 1);
    let j = idx % (c + 1);
    let alpha_base = h * c;

    let mut product = f32::new(1.0f32);
    if j < c {
        let mut i = j + 1;
        while i < c {
            product *= alpha[alpha_base + i];
            i += 1;
        }
        decay_to_end[h * c + j] = product;
    } else {
        // j == C: total product from i=0 to C-1
        let mut i = 0;
        while i < c {
            product *= alpha[alpha_base + i];
            i += 1;
        }
        total_decay[h] = product;
    }
}

/// Launcher for the split chunk decay kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetChunkDecaySplitCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetChunkDecaySplitCubeCL {
    /// Compute per-head decay_to_end [n_head, C] and total_decay [n_head].
    ///
    /// # Safety
    /// - `alpha_handle`: `n_head * C` f32 elements.
    /// - `decay_to_end_handle`: `n_head * C` f32 elements.
    /// - `total_decay_handle`: `n_head` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        alpha_handle: Handle,
        decay_to_end_handle: Handle,
        total_decay_handle: Handle,
        n_head: usize,
        c: usize,
    ) {
        let params: [f32; 2] = [n_head as f32, c as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * (c + 1);
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_chunk_decay_split_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(alpha_handle, n_head * c),
                BufferArg::from_raw_parts(decay_to_end_handle, n_head * c),
                BufferArg::from_raw_parts(total_decay_handle, n_head),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Compute V_weighted[h, j, :] = decay_to_end[h, j] * beta[h, j] * V[h, j, :]
///
/// This is the weighted V matrix for the ΔS = V_weighted^T @ K computation.
/// Also used (with a different decay matrix) for the intra-chunk output computation.
///
/// Layout:
/// - `v`: `[n_head, C, d]` — the V projections for C tokens.
/// - `beta`: `[n_head, C]` — per-head per-token beta scalars.
/// - `decay_to_end`: `[n_head, C]` — from `DeltanetChunkDecayCubeCL` (first C entries per head).
/// - `v_weighted`: `[n_head, C, d]` — output.
/// - `params`: `[n_head_f32, C_f32, d_f32]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_weighted_v_f32(
    v: &[f32],
    beta: &[f32],
    decay_to_end: &[f32],
    v_weighted: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * c * d;
    if idx >= total {
        terminate!();
    }

    // idx decomposes into (head, token, dim): idx = head * C * d + token * d + dim
    let head = idx / (c * d);
    let rem = idx % (c * d);
    let token = rem / d;

    let beta_val = beta[head * c + token];
    let decay_val = decay_to_end[head * c + token];
    v_weighted[idx] = decay_val * beta_val * v[idx];
}

/// Launcher for the weighted V kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetWeightedVCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetWeightedVCubeCL {
    /// Compute V_weighted = decay_to_end ⊙ beta ⊙ V.
    ///
    /// # Safety
    /// All handles must have the correct element counts (see kernel doc).
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        v_handle: Handle,
        beta_handle: Handle,
        decay_to_end_handle: Handle,
        v_weighted_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_weighted_v_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(v_handle, total),
                BufferArg::from_raw_parts(beta_handle, n_head * c),
                BufferArg::from_raw_parts(decay_to_end_handle, n_head * c),
                BufferArg::from_raw_parts(v_weighted_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// State transition: S_next[h] = total_decay[h] * S_prev[h] + delta_s[h]
///
/// Layout:
/// - `s_prev`: `[n_head, d, d]` — the boundary state entering this chunk.
/// - `delta_s`: `[n_head, d, d]` — the intra-chunk state contribution (V_w^T @ K).
/// - `total_decay`: `[n_head]` — Π α for the chunk (from DeltanetChunkDecayCubeCL, index C).
/// - `s_next`: `[n_head, d, d]` — output: the state for the next chunk boundary.
/// - `params`: `[n_head_f32, d_sq_f32]` where d_sq = d*d.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_state_transition_f32(
    s_prev: &[f32],
    delta_s: &[f32],
    total_decay: &[f32],
    s_next: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let d_sq = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * d_sq;
    if idx >= total {
        terminate!();
    }

    let head = idx / d_sq;
    let td = total_decay[head];
    s_next[idx] = td * s_prev[idx] + delta_s[idx];
}

/// Launcher for the state transition kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetStateTransitionCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetStateTransitionCubeCL {
    /// Compute S_next = total_decay * S_prev + delta_s.
    ///
    /// # Safety
    /// All handles must have the correct element counts.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        s_prev_handle: Handle,
        delta_s_handle: Handle,
        total_decay_handle: Handle,
        s_next_handle: Handle,
        n_head: usize,
        d: usize,
    ) {
        let d_sq = d * d;
        let params: [f32; 2] = [n_head as f32, d_sq as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * d_sq;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_state_transition_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(s_prev_handle, total),
                BufferArg::from_raw_parts(delta_s_handle, total),
                BufferArg::from_raw_parts(total_decay_handle, n_head),
                BufferArg::from_raw_parts(s_next_handle, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3 — Intra-chunk output (causal-weighted QV·K)
// ---------------------------------------------------------------------------
//
// The intra-chunk output for token t:
//   intra_t = Σ_{j=0}^{t} inter_decay[t,j] · β_j · (q_t·v_j) · k_j
//
// where inter_decay[t,j] = Π_{i=j+1}^{t} α_i (the cumulative decay from j+1 to t).
//
// This is decomposed into:
// 1. `deltanet_intra_attn_f32`: compute attn[h,t,j] = (q_t·v_j) · inter_decay[t,j] · β_j
//    for j ≤ t (causal), 0 for j > t. Writes a [H, C, C] matrix.
// 2. MatmulCubeCL (per head, or a batched variant): intra[h,t,:] = attn[h,t,:] @ K[h,:,:]
//
// For simplicity + correctness, the intra output kernel below fuses both steps:
// one cube per (head, token), each cube computes the full d-dim output for that
// token by looping over j=0..t and accumulating decay-weighted (q_t·v_j)·k_j.
// ---------------------------------------------------------------------------

/// Compute the intra-chunk output for all tokens in a chunk.
///
/// For token t in the chunk, computes:
///   intra[h, t, :] = Σ_{j=0}^{t} inter_decay[h,t,j] · β[h,j] · (q[h,t,:] · v[h,j,:]) · k[h,j,:]
///
/// where inter_decay[h,t,j] = Π_{i=j+1}^{t} α[h,i] (cumulative decay).
///
/// ## Dispatch
///
/// One cube per (head, token): `CubeCount::Static(n_head, C, 1)`.
/// Each cube uses `d` threads (one per output dimension). For d=128 and plane=32,
/// each lane handles d/PLANE = 4 dimensions.
///
/// ## Accumulation pattern
///
/// For each j from 0 to t (inclusive):
///   1. Compute running_decay *= α[j] (the decay accumulated through position j).
///      inter_decay[t,j] = running_decay AFTER multiplying by α[j+1..t].
///      Wait — let me be precise: inter_decay[t,j] = Π_{i=j+1}^{t} α[i].
///      For j=t, inter_decay = 1 (empty product).
///      For j=t-1, inter_decay = α[t].
///      For j=t-2, inter_decay = α[t-1]·α[t].
///   So we can compute it by accumulating from j=t down to j=0:
///      running = 1; for j=t downto 0: weight = running; running *= α[j+1]... NO.
//
//   Let me think again. inter_decay[t,j] = Π_{i=j+1}^{t} α[i].
//   For a fixed t, as j decreases from t to 0:
//     j=t: inter_decay = 1
//     j=t-1: inter_decay = α[t]
//     j=t-2: inter_decay = α[t-1]·α[t]
//     j=0: inter_decay = α[1]·α[2]·...·α[t]
//
//   So we can compute it incrementally: start with weight=1 at j=t, then for
//   each j from t-1 downto 0: weight *= α[j+1].
//   That is: weight starts at 1. For j from t down to 0:
//     contribution_j = weight * β[j] * (q_t · v_j) * k_j
//     if j > 0: weight *= α[j]  (prepares weight for j-1)
//
//   Wait: inter_decay[t, j-1] = Π_{i=j}^{t} α[i] = α[j] * Π_{i=j+1}^{t} α[i] = α[j] * inter_decay[t,j].
//   So: weight_{j-1} = α[j] * weight_j. That means after processing j, multiply weight by α[j].
//   For j=t: weight=1. After processing j=t, weight = α[t] = inter_decay[t, t-1]. ✓
//   After processing j=t-1: weight = α[t-1] * α[t] = inter_decay[t, t-2]. ✓
//
//   So the loop is: weight = 1; for j from t downto 0:
//     weight_j = inter_decay[t,j] = weight (before the multiply)
//     process contribution with weight_j
//     weight *= α[j]   (if j > 0; for j=0 this is total_decay which we don't need here)
///
/// ## Plane cooperation
///
/// Each lane owns `COLS_PER_LANE = d / PLANE` output dimensions. The q_t·v_j dot
/// product is computed via `plane_sum` across the plane (each lane contributes its
/// slice of the dot product). The k_j accumulation is per-lane (each lane
/// accumulates into its own output dimensions).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_intra_output_f32(
    q: &[f32],       // [n_head, C, d]
    k: &[f32],       // [n_head, C, d]
    v: &[f32],       // [n_head, C, d]
    alpha: &[f32],   // [n_head, C]
    beta: &[f32],    // [n_head, C]
    output: &mut [f32], // [n_head, C, d]
    params: &[f32],  // [n_head_f32, C_f32, d_f32]
) {
    let n_head = params[0usize] as u32;
    let c = params[1usize] as u32;
    let d = params[2usize] as u32;
    let head = CUBE_POS_X;
    let token = CUBE_POS_Y;
    let lane = UNIT_POS_PLANE;
    let plane_dim = PLANE_DIM;

    if head >= n_head || token >= c {
        terminate!();
    }

    let q_base = (head * c + token) * d;
    let head_alpha_base = head * c;

    // Each lane's output accumulators (d/PLANE = 4 dimensions for d=128, PLANE=32).
    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);

    // Loop over j from token downto 0, accumulating inter_decay incrementally.
    // weight starts at 1 (inter_decay[t,t] = empty product = 1).
    let mut weight = f32::new(1.0f32);

    // Iterate j = token, token-1, ..., 0 via step = 0..=token, j = token - step.
    for step in 0u32..=token {
        let j = token - step;
        let kv_base = (head * c + j) * d;
        let beta_j = beta[(head_alpha_base + j) as usize];

        // Compute q_t · v_j (plane-cooperative dot product).
        let qv0 = q[(q_base + lane) as usize] * v[(kv_base + lane) as usize];
        let qv1 = q[(q_base + lane + plane_dim) as usize] * v[(kv_base + lane + plane_dim) as usize];
        let qv2 = q[(q_base + lane + 2u32 * plane_dim) as usize] * v[(kv_base + lane + 2u32 * plane_dim) as usize];
        let qv3 = q[(q_base + lane + 3u32 * plane_dim) as usize] * v[(kv_base + lane + 3u32 * plane_dim) as usize];
        let qv_dot = qv0 + qv1 + qv2 + qv3;
        let q_t_dot_v_j = plane_sum(qv_dot);

        // The contribution: weight * beta_j * q_t_dot_v_j * k_j[:]
        let scale = weight * beta_j * q_t_dot_v_j;

        acc0 += scale * k[(kv_base + lane) as usize];
        acc1 += scale * k[(kv_base + lane + plane_dim) as usize];
        acc2 += scale * k[(kv_base + lane + 2u32 * plane_dim) as usize];
        acc3 += scale * k[(kv_base + lane + 3u32 * plane_dim) as usize];

        // Update weight for j-1: inter_decay[t, j-1] = α[j] * inter_decay[t, j]
        if j > 0u32 {
            weight *= alpha[(head_alpha_base + j) as usize];
        }
    }

    // Write output (each lane writes its 4 dimensions).
    let out_base = (head * c + token) * d;
    output[(out_base + lane) as usize] = acc0;
    output[(out_base + lane + plane_dim) as usize] = acc1;
    output[(out_base + lane + 2u32 * plane_dim) as usize] = acc2;
    output[(out_base + lane + 3u32 * plane_dim) as usize] = acc3;
}

/// Launcher for the intra-chunk output kernel.
///
/// Requires `d == PLANE * 4` (128 on Metal/CUDA where PLANE=32). Falls back to
/// sequential per-token recurrence otherwise.
#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
#[allow(unused_imports, reason = "Plane trait needed for plane_sum() resolution")]
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetIntraOutputCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetIntraOutputCubeCL {
    pub const PLANE: usize = 32;
    pub const COLS_PER_LANE: usize = 4;

    #[inline]
    #[must_use]
    pub fn supports(d: usize) -> bool {
        d == Self::PLANE * Self::COLS_PER_LANE
    }

    /// Launch the intra-chunk output kernel.
    ///
    /// # Safety
    /// - `d` must satisfy [`Self::supports`].
    /// - Q, K, V: `n_head * C * d` f32 elements each.
    /// - alpha, beta: `n_head * C` f32 elements each.
    /// - output: `n_head * C * d` f32 elements.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        k_handle: Handle,
        v_handle: Handle,
        alpha_handle: Handle,
        beta_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        debug_assert!(
            Self::supports(d),
            "intra-output kernel requires d == {}", Self::PLANE * Self::COLS_PER_LANE
        );
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * d;
        unsafe {
            deltanet_intra_output_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, c as u32, 1),
                CubeDim::new_1d(Self::PLANE as u32),
                BufferArg::from_raw_parts(q_handle, total),
                BufferArg::from_raw_parts(k_handle, total),
                BufferArg::from_raw_parts(v_handle, total),
                BufferArg::from_raw_parts(alpha_handle, n_head * c),
                BufferArg::from_raw_parts(beta_handle, n_head * c),
                BufferArg::from_raw_parts(output_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3b — Cross-chunk contribution (Q @ S_boundary) + forward decay
// ---------------------------------------------------------------------------
//
// The cross-chunk contribution for token t:
//   cross_t = (Π_{i=0}^{t} α_i) · q_t^T · S_{-1}
//
// where S_{-1} is the boundary state entering this chunk.
//
// decay_to_t[h, t] = Π_{i=0}^{t} α[h, i]  (forward cumulative product)
// cross_t = decay_to_t · (q_t^T @ S_{-1})
// ---------------------------------------------------------------------------

/// Compute forward cumulative decay: `decay_to_t[h, t] = Π_{i=0}^{t} α[h, i]`.
///
/// Layout:
/// - `alpha`: `[n_head, C]`
/// - `decay_to_t`: `[n_head, C]` — output. decay_to_t[h, 0] = α[h, 0],
///   decay_to_t[h, t] = decay_to_t[h, t-1] * α[h, t].
/// - `params`: `[n_head_f32, C_f32]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_forward_decay_f32(
    alpha: &[f32],
    decay_to_t: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * c;
    if idx >= total {
        terminate!();
    }

    let h = idx / c;
    let t = idx % c;
    let alpha_base = h * c;

    // Forward cumulative product: Π_{i=0}^{t} α[h, i]
    let mut product = f32::new(1.0f32);
    for i in 0..=t {
        product *= alpha[alpha_base + i];
    }
    decay_to_t[idx] = product;
}

/// Launcher for the forward cumulative decay kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetForwardDecayCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetForwardDecayCubeCL {
    /// Launch the forward cumulative decay kernel.
    ///
    /// # Safety
    ///
    /// Caller must ensure: `alpha_handle` and `decay_to_t_handle` each have
    /// `n_head * c` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        alpha_handle: Handle,
        decay_to_t_handle: Handle,
        n_head: usize,
        c: usize,
    ) {
        let params: [f32; 2] = [n_head as f32, c as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_forward_decay_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(alpha_handle, total),
                BufferArg::from_raw_parts(decay_to_t_handle, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Cross-chunk contribution: `cross[h, t, :] = decay_to_t[h, t] · q[h, t, :]^T · S[h, :, :]`
///
/// Structure mirrors `deltanet_delta_s_f32` (per-lane full column ownership,
/// Issue 673 Bug A fix): one cube per (head, token), 32 lanes, each lane owns
/// `COLS_PER_LANE = d / PLANE = 4` output columns and loops over ALL d rows
/// for each, accumulating the complete per-column dot in registers. NO
/// cross-lane reduction — `plane_sum` is wrong here because each lane's
/// partial covers a different column (the pre-fix kernel reduced partials of
/// 32 different columns into a mod-32 staircase; see Issue 673).
///
/// Total S traffic: each of the d·d S elements is read exactly once per cube
/// (4 reads × 128 rows × 32 lanes = 16384), the minimum for a correct matvec.
///
/// Layout:
/// - `q`: `[n_head, C, d]`
/// - `s_boundary`: `[n_head, d, d]`
/// - `decay_to_t`: `[n_head, C]`
/// - `output`: `[n_head, C, d]` — cross-chunk contribution (ADDED to intra-chunk output).
/// - `params`: `[n_head_f32, C_f32, d_f32]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_cross_chunk_f32(
    q: &[f32],
    s_boundary: &[f32],
    decay_to_t: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as u32;
    let c = params[1usize] as u32;
    let d = params[2usize] as u32;
    let head = CUBE_POS_X;
    let token = CUBE_POS_Y;
    let lane = UNIT_POS_PLANE;
    let plane_dim = PLANE_DIM;

    if head >= n_head || token >= c {
        terminate!();
    }

    let q_base = (head * c + token) * d;
    let decay = decay_to_t[(head * c + token) as usize];

    // output[h, t, col] = decay_to_t · Σ_r q[h,t,r] · S[h,r,col]
    //
    // Issue 673 Bug A fix: per-lane full column ownership (delta_s structure).
    // Each lane owns 4 columns {lane, lane+32, lane+64, lane+96} and loops
    // over ALL d rows for each — the per-column dot is complete within one
    // lane, so no cross-lane reduction is needed (or valid: plane_sum here
    // summed partials of 32 DIFFERENT columns, producing a mod-32 staircase
    // instead of a matvec).
    //
    // S[r, col] is at index head*d*d + r*d + col; q[r] is at q_base + r.
    let s_base = head * d * d;

    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);
    for r in 0u32..d {
        let q_r = q[(q_base + r) as usize];
        let s_row = s_base + r * d;
        acc0 += q_r * s_boundary[(s_row + lane) as usize];
        acc1 += q_r * s_boundary[(s_row + lane + plane_dim) as usize];
        acc2 += q_r * s_boundary[(s_row + lane + 2u32 * plane_dim) as usize];
        acc3 += q_r * s_boundary[(s_row + lane + 3u32 * plane_dim) as usize];
    }

    // Write output scaled by decay — each lane scatters its 4 completed columns.
    let out_base = (head * c + token) * d;
    output[(out_base + lane) as usize] = decay * acc0;
    output[(out_base + lane + plane_dim) as usize] = decay * acc1;
    output[(out_base + lane + 2u32 * plane_dim) as usize] = decay * acc2;
    output[(out_base + lane + 3u32 * plane_dim) as usize] = decay * acc3;
}

/// Launcher for the cross-chunk contribution kernel.
///
/// Requires `d == PLANE * 4` (128 on Metal/CUDA).
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetCrossChunkCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetCrossChunkCubeCL {
    pub const PLANE: usize = 32;
    pub const COLS_PER_LANE: usize = 4;

    #[inline]
    #[must_use]
    pub fn supports(d: usize) -> bool {
        d == Self::PLANE * Self::COLS_PER_LANE
    }

    /// Launch the cross-chunk contribution kernel.
    ///
    /// # Safety
    /// - `d` must satisfy [`Self::supports`].
    /// - Q: `n_head * C * d`, S_boundary: `n_head * d * d`,
    ///   decay_to_t: `n_head * C`, output: `n_head * C * d`.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        s_boundary_handle: Handle,
        decay_to_t_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        debug_assert!(Self::supports(d), "cross-chunk kernel requires d == 128");
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * d;
        unsafe {
            deltanet_cross_chunk_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, c as u32, 1),
                CubeDim::new_1d(Self::PLANE as u32),
                BufferArg::from_raw_parts(q_handle, total),
                BufferArg::from_raw_parts(s_boundary_handle, n_head * d * d),
                BufferArg::from_raw_parts(decay_to_t_handle, n_head * c),
                BufferArg::from_raw_parts(output_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2b — Intra-chunk state contribution (ΔS)
// ---------------------------------------------------------------------------
//
// ΔS[h] = Σ_{j=0}^{C-1} decay_to_end[h, j] · β[h, j] · v[h, j, :] ⊗ k[h, j, :]^T
//
// This is a [d, d] matrix per head, accumulated from C rank-1 updates.
// One cube per (head, row), each cube computes one row of the d×d ΔS matrix.
// Each lane owns COLS_PER_LANE = 4 columns and loops over the C tokens.
// ---------------------------------------------------------------------------

/// Compute ΔS = Σ_j decay_to_end[j] · β[j] · v[j, :] ⊗ k[j, :]^T
///
/// Layout:
/// - `k`: `[n_head, C, d]`
/// - `v`: `[n_head, C, d]`
/// - `beta`: `[n_head, C]`
/// - `decay_to_end`: `[n_head, C]` (from `DeltanetChunkDecayCubeCL`, first C entries per head)
/// - `delta_s`: `[n_head, d, d]` — output
/// - `params`: `[n_head_f32, C_f32, d_f32]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_delta_s_f32(
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    decay_to_end: &[f32],
    delta_s: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as u32;
    let c = params[1usize] as u32;
    let d = params[2usize] as u32;
    let head = CUBE_POS_X;
    let row = CUBE_POS_Y;
    let lane = UNIT_POS_PLANE;
    let plane_dim = PLANE_DIM;

    if head >= n_head || row >= d {
        terminate!();
    }

    let head_base = head * c;
    let ds_base = (head * d * d + row * d) as usize;

    // Each lane owns 4 columns: lane, lane+32, lane+64, lane+96.
    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);

    // Loop over C tokens, accumulating rank-1 updates into this row.
    for j in 0u32..c {
        let kv_base = (head * c + j) * d;
        let v_row = v[(kv_base + row) as usize]; // v[j, row]
        let scale = decay_to_end[(head_base + j) as usize] * beta[(head_base + j) as usize] * v_row;

        // ΔS[row, col] += scale * k[j, col] for each col.
        acc0 += scale * k[(kv_base + lane) as usize];
        acc1 += scale * k[(kv_base + lane + plane_dim) as usize];
        acc2 += scale * k[(kv_base + lane + 2u32 * plane_dim) as usize];
        acc3 += scale * k[(kv_base + lane + 3u32 * plane_dim) as usize];
    }

    delta_s[ds_base + lane as usize] = acc0;
    delta_s[ds_base + (lane + plane_dim) as usize] = acc1;
    delta_s[ds_base + (lane + 2u32 * plane_dim) as usize] = acc2;
    delta_s[ds_base + (lane + 3u32 * plane_dim) as usize] = acc3;
}

/// Launcher for the ΔS (intra-chunk state contribution) kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetDeltaSCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetDeltaSCubeCL {
    pub const PLANE: usize = 32;
    pub const COLS_PER_LANE: usize = 4;

    #[inline]
    #[must_use]
    pub fn supports(d: usize) -> bool {
        d == Self::PLANE * Self::COLS_PER_LANE
    }

    /// Launch the ΔS kernel.
    ///
    /// # Safety
    /// - `d` must satisfy [`Self::supports`].
    /// - K, V: `n_head * C * d`. beta, decay_to_end: `n_head * C`.
    /// - delta_s: `n_head * d * d`.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        k_handle: Handle,
        v_handle: Handle,
        beta_handle: Handle,
        decay_to_end_handle: Handle,
        delta_s_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        debug_assert!(Self::supports(d), "delta_s kernel requires d == 128");
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total_hcd = n_head * c * d;
        unsafe {
            deltanet_delta_s_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, d as u32, 1),
                CubeDim::new_1d(Self::PLANE as u32),
                BufferArg::from_raw_parts(k_handle, total_hcd),
                BufferArg::from_raw_parts(v_handle, total_hcd),
                BufferArg::from_raw_parts(beta_handle, n_head * c),
                BufferArg::from_raw_parts(decay_to_end_handle, n_head * c),
                BufferArg::from_raw_parts(delta_s_handle, n_head * d * d),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4 — Wiring helpers: layout transpose for production prefill
// ---------------------------------------------------------------------------
//
// The production prefill stores QKVX in token-major [P, 3*n_head*d] and
// beta/alpha in [P, n_head]. The chunked recurrence kernels expect head-major
// [n_head, C, d] and [n_head, C]. These three kernels bridge the layout gap.
//
// All three are simple elementwise transposes — one dispatch per chunk, O(C * n_head * d)
// work. Negligible vs the O(C * d²) recurrence.
// ---------------------------------------------------------------------------

/// Extract Q, K, V from token-major qkvx layout to head-major [n_head, C, d].
///
/// Input `qkvx_chunk`: [C, 3*n_head*d] — the qkvx slice for C tokens.
///   For token t, head h, dim i:
///   Q: qkvx_chunk[t * 3*n_head*d + 0*n_head*d + h*d + i]
///   K: qkvx_chunk[t * 3*n_head*d + 1*n_head*d + h*d + i]
///   V: qkvx_chunk[t * 3*n_head*d + 2*n_head*d + h*d + i]
///
/// Output: q, k, v each [n_head, C, d].
///   out[h * C * d + t * d + i]
///
/// Dispatch: one thread per (head, token, dim). Total = n_head * C * d.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_extract_qkv_chunk_f32(
    qkvx: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    params: &[f32], // [n_head, C, d, v_dim]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let v_dim = params[3usize] as usize; // n_head * d
    let idx = ABSOLUTE_POS;
    let total = n_head * c * d;
    if idx >= total {
        terminate!();
    }

    // idx decomposes into (head, token, dim).
    let head = idx / (c * d);
    let rem = idx % (c * d);
    let token = rem / d;
    let dim = rem % d;

    let qkvx_stride = 3 * v_dim; // 3 * n_head * d per token
    let head_offset = head * d + dim;

    let t_base = token * qkvx_stride;
    q_out[idx] = qkvx[t_base + head_offset];              // Q block at offset 0
    k_out[idx] = qkvx[t_base + v_dim + head_offset];      // K block at offset v_dim
    v_out[idx] = qkvx[t_base + 2 * v_dim + head_offset];  // V block at offset 2*v_dim
}

/// Launcher for the QKV extraction kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetExtractQkvChunkCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetExtractQkvChunkCubeCL {
    /// Extract Q/K/V from token-major to head-major layout.
    ///
    /// # Safety
    /// - `qkvx_handle`: C * 3 * n_head * d f32 elements.
    /// - q_out, k_out, v_out: n_head * C * d f32 elements each.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkvx_handle: Handle,
        q_out_handle: Handle,
        k_out_handle: Handle,
        v_out_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let v_dim = n_head * d;
        let params: [f32; 4] = [n_head as f32, c as f32, d as f32, v_dim as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_extract_qkv_chunk_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(qkvx_handle, c * 3 * v_dim),
                BufferArg::from_raw_parts(q_out_handle, total),
                BufferArg::from_raw_parts(k_out_handle, total),
                BufferArg::from_raw_parts(v_out_handle, total),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

/// Extract scalars (beta, alpha) from token-major [C, n_head] to head-major [n_head, C].
///
/// Dispatch: one thread per (head, token). Total = n_head * C.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_extract_scalars_chunk_f32(
    src: &[f32],
    dst: &mut [f32],
    params: &[f32], // [n_head, C]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * c;
    if idx >= total {
        terminate!();
    }

    // Head-major output: dst[head * C + token]
    let head = idx / c;
    let token = idx % c;

    // Token-major input: src[token * n_head + head]
    dst[idx] = src[token * n_head + head];
}

/// Launcher for the scalars extraction kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetExtractScalarsChunkCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetExtractScalarsChunkCubeCL {
    /// Transpose [C, n_head] → [n_head, C].
    ///
    /// # Safety
    /// - `src_handle`: C * n_head f32 elements.
    /// - `dst_handle`: n_head * C f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        dst_handle: Handle,
        n_head: usize,
        c: usize,
    ) {
        let params: [f32; 2] = [n_head as f32, c as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_extract_scalars_chunk_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(src_handle, c * n_head),
                BufferArg::from_raw_parts(dst_handle, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Scatter combined output from head-major [n_head, C, d] to token-major [C, n_head*d].
///
/// output_token[t * v_dim + h * d + i] = (intra[h * C * d + t * d + i] + cross[h * C * d + t * d + i]) * scale
///
/// Dispatch: one thread per (token, head, dim). Total = C * n_head * d.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_scatter_output_chunk_f32(
    intra: &[f32],
    cross: &[f32],
    out: &mut [f32],
    params: &[f32], // [n_head, C, d, scale]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let scale = params[3usize];
    let idx = ABSOLUTE_POS;
    let total = c * n_head * d;
    if idx >= total {
        terminate!();
    }

    // idx decomposes into (token, head, dim) for token-major output.
    let token = idx / (n_head * d);
    let rem = idx % (n_head * d);
    let head = rem / d;
    let dim = rem % d;

    // Head-major input offset: h * C * d + t * d + i
    let hcd_off = head * c * d + token * d + dim;
    out[idx] = (intra[hcd_off] + cross[hcd_off]) * scale;
}

/// Launcher for the scatter-output kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetScatterOutputChunkCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetScatterOutputChunkCubeCL {
    /// Combine intra+cross, scale by 1/sqrt(d), scatter to token-major.
    ///
    /// # Safety
    /// - intra, cross: n_head * C * d f32 elements each.
    /// - out: C * n_head * d f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        intra_handle: Handle,
        cross_handle: Handle,
        out_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let scale = 1.0f32 / (d as f32).sqrt();
        let params: [f32; 4] = [n_head as f32, c as f32, d as f32, scale];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = c * n_head * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_scatter_output_chunk_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(intra_handle, n_head * c * d),
                BufferArg::from_raw_parts(cross_handle, n_head * c * d),
                BufferArg::from_raw_parts(out_handle, total),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};

    /// Tolerance for chunked vs sequential f32 comparison. The arithmetic is
    /// identical (same convolution sum + same SiLU); the only difference is data
    /// sourcing. Bit-identity is expected but we allow a tiny tolerance for f32
    /// exp() implementation differences across the two code paths.
    const TOL: f32 = 1e-6;

    /// CPU reference: run the sequential conv1d (matching `deltanet_conv1d_f32`)
    /// for `c` tokens, carrying the conv_state across tokens.
    ///
    /// The single-token kernel semantics:
    /// 1. Shift conv_state left by 1: conv_state[ch, k] = conv_state[ch, k+1]
    /// 2. Append raw input: conv_state[ch, ks-1] = input_raw[ch]
    /// 3. Conv: out = Σ_k conv_state[ch, k] * weight[ch, k]
    /// 4. SiLU: output[ch] = out * sigmoid(out)
    ///
    /// The chunked kernel uses a carry of the last `ks-1` RAW inputs (not the
    /// full conv_state which is `ks` wide). The conversion: the carry holds
    /// conv_state[ch, 0..ks-1] (the oldest `ks-1` samples); the newest sample
    /// (conv_state[ch, ks-1]) is the current token's raw input, which the chunked
    /// kernel reads from the input buffer.
    fn cpu_sequential_conv1d_chunked(
        input_raw: &[f32],          // [c, conv_dim] raw inputs
        conv_weight: &[f32],        // [conv_dim, kernel_size]
        carry_in_out: &mut [f32],   // [conv_dim, kernel_size-1] IN-OUT
        c: usize,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let ks_m1 = kernel_size - 1;
        let mut output = vec![0.0f32; c * conv_dim];

        // The carry holds the last `ks-1` raw inputs from BEFORE this chunk:
        // carry[ch * ks_m1 + j] = raw input at position -(ks_m1) + j.
        // For ks=4: carry[0]=raw[-3], carry[1]=raw[-2], carry[2]=raw[-1].
        //
        // The convolution for token t (chunk-relative index) needs samples at
        // positions [t-3, t-2, t-1, t] (for ks=4). Position p < 0 → carry,
        // position p >= 0 → input_raw[p].
        //
        // The weight indexing: weight[ch, 0] multiplies the oldest sample
        // (position t-ks_m1), weight[ch, ks-1] multiplies the newest (position t).
        // This matches the GPU kernel exactly.
        for t in 0..c {
            for ch in 0..conv_dim {
                let mut sum = 0.0f32;
                for k in 0..kernel_size {
                    let sample_pos_signed = t as i64 - ks_m1 as i64 + k as i64;
                    let val = if sample_pos_signed < 0 {
                        let carry_idx =
                            (sample_pos_signed + ks_m1 as i64) as usize;
                        carry_in_out[ch * ks_m1 + carry_idx]
                    } else {
                        let sample_pos = sample_pos_signed as usize;
                        input_raw[sample_pos * conv_dim + ch]
                    };
                    sum += val * conv_weight[ch * kernel_size + k];
                }
                // SiLU
                let sig = 1.0 / (1.0 + (-sum).exp());
                output[t * conv_dim + ch] = sum * sig;
            }
        }

        // Update carry: the last `ks_m1` raw inputs of THIS chunk for the next
        // chunk. These are tokens [c - ks_m1, c - 1] (the last ks_m1 tokens).
        // carry[ch * ks_m1 + j] = input_raw[(c - ks_m1 + j) * conv_dim + ch].
        // If c < ks_m1 (partial chunk smaller than carry), the new carry is a
        // mix of old carry + chunk inputs — but for simplicity we require c >= ks_m1
        // (the production chunk size 64 >> 3).
        if c >= ks_m1 {
            for ch in 0..conv_dim {
                for j in 0..ks_m1 {
                    carry_in_out[ch * ks_m1 + j] =
                        input_raw[(c - ks_m1 + j) * conv_dim + ch];
                }
            }
        } else {
            // Partial chunk smaller than carry: merge old carry + new inputs.
            // This shifts the carry window by c positions.
            for ch in 0..conv_dim {
                let mut new_carry = vec![0.0f32; ks_m1];
                // Old carry slots that are still valid: indices [c..ks_m1)
                for j in c..ks_m1 {
                    new_carry[j - c] = carry_in_out[ch * ks_m1 + j];
                }
                // New inputs fill the remaining slots: [ks_m1 - c..ks_m1)
                for j in 0..c {
                    new_carry[ks_m1 - c + j] = input_raw[j * conv_dim + ch];
                }
                carry_in_out[ch * ks_m1..ch * ks_m1 + ks_m1]
                    .copy_from_slice(&new_carry);
            }
        }
        output
    }

    /// G1: chunked conv1d output matches sequential reference at C=8, conv_dim=16, ks=4.
    #[test]
    fn test_chunked_conv1d_g1_basic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let c: usize = 8;
        let conv_dim: usize = 16;
        let kernel_size: usize = 4;
        let ks_m1 = kernel_size - 1;

        // Synthetic raw inputs: input_raw[t, ch] = 0.1 * (t as f32) + 0.01 * (ch as f32)
        let input_raw: Vec<f32> = (0..c * conv_dim)
            .map(|i| {
                let t = i / conv_dim;
                let ch = i % conv_dim;
                0.1 * (t as f32) + 0.01 * (ch as f32)
            })
            .collect();

        // Depthwise weights: conv_weight[ch, k] = 0.1 * (ch as f32) + 0.01 * (k as f32)
        let conv_weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|i| {
                let ch = i / kernel_size;
                let k = i % kernel_size;
                0.1 * (ch as f32) + 0.01 * (k as f32)
            })
            .collect();

        // Initial carry (last ks-1 raw inputs from a "previous chunk").
        // Use zeros as the initial conv_state equivalent (matches production init).
        let carry_init: Vec<f32> = vec![0.0f32; conv_dim * ks_m1];

        // ── GPU chunked path ──
        let input_handle = client.create_from_slice(f32::as_bytes(&input_raw));
        let conv_weight_handle = client.create_from_slice(f32::as_bytes(&conv_weight));
        let carry_handle = client.create_from_slice(f32::as_bytes(&carry_init));
        let output_handle = client.empty(c * conv_dim * std::mem::size_of::<f32>());

        // Compact carry layout: stride = ks-1, offset = 0.
        let carry_stride = ks_m1;
        let carry_idx_offset = 0;
        unsafe {
            DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                conv_weight_handle,
                carry_handle.clone(),
                c,
                conv_dim,
                kernel_size,
                carry_stride,
                carry_idx_offset,
            );
        }

        let gpu_output_bytes = client.read_one(output_handle).expect("read output");
        let gpu_output = f32::from_bytes(&gpu_output_bytes);
        let gpu_carry_bytes = client.read_one(carry_handle).expect("read carry");
        let gpu_carry = f32::from_bytes(&gpu_carry_bytes);

        // ── CPU sequential reference ──
        let mut cpu_carry = carry_init.clone();
        let cpu_output =
            cpu_sequential_conv1d_chunked(&input_raw, &conv_weight, &mut cpu_carry, c, conv_dim, kernel_size);

        // ── Compare outputs ──
        assert_eq!(gpu_output.len(), cpu_output.len());
        let mut max_diff = 0.0f32;
        for (i, (&g, &c_ref)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c_ref).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < TOL,
                "chunked conv1d output[{i}] GPU={g:.8} CPU={c_ref:.8} diff={diff:.2e} (tol {TOL:.2e})"
            );
        }
        println!("test_chunked_conv1d_g1_basic: max output diff = {max_diff:.2e}");

        // ── Compare carry (the last ks-1 raw inputs for the next chunk) ──
        assert_eq!(gpu_carry.len(), cpu_carry.len());
        for (i, (&g, &c_ref)) in gpu_carry.iter().zip(cpu_carry.iter()).enumerate() {
            let diff = (g - c_ref).abs();
            assert!(
                diff < TOL,
                "chunked conv1d carry[{i}] GPU={g:.8} CPU={c_ref:.8} diff={diff:.2e}"
            );
        }
    }

    /// G1: chunked conv1d with a non-zero initial carry (simulates chunk N > 0).
    #[test]
    fn test_chunked_conv1d_g1_nonzero_carry() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let c: usize = 16;
        let conv_dim: usize = 8;
        let kernel_size: usize = 4;
        let ks_m1 = kernel_size - 1;

        let input_raw: Vec<f32> = (0..c * conv_dim)
            .map(|i| {
                let t = i / conv_dim;
                let ch = i % conv_dim;
                0.05 * (t as f32) - 0.02 * (ch as f32) + 0.5
            })
            .collect();

        let conv_weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|i| 0.1 + 0.01 * (i as f32))
            .collect();

        // Non-zero carry: simulate a previous chunk's last ks-1 raw inputs.
        let carry_init: Vec<f32> = (0..conv_dim * ks_m1)
            .map(|i| 0.3 - 0.01 * (i as f32))
            .collect();

        // ── GPU chunked path ──
        let input_handle = client.create_from_slice(f32::as_bytes(&input_raw));
        let conv_weight_handle = client.create_from_slice(f32::as_bytes(&conv_weight));
        let carry_handle = client.create_from_slice(f32::as_bytes(&carry_init));
        let output_handle = client.empty(c * conv_dim * std::mem::size_of::<f32>());

        // Compact carry layout: stride = ks-1, offset = 0.
        let carry_stride = ks_m1;
        let carry_idx_offset = 0;
        unsafe {
            DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                conv_weight_handle,
                carry_handle.clone(),
                c,
                conv_dim,
                kernel_size,
                carry_stride,
                carry_idx_offset,
            );
        }

        let gpu_output_bytes = client.read_one(output_handle).expect("read output");
        let gpu_output = f32::from_bytes(&gpu_output_bytes);
        let gpu_carry_bytes = client.read_one(carry_handle).expect("read carry");
        let gpu_carry = f32::from_bytes(&gpu_carry_bytes);

        // ── CPU sequential reference ──
        let mut cpu_carry = carry_init.clone();
        let cpu_output =
            cpu_sequential_conv1d_chunked(&input_raw, &conv_weight, &mut cpu_carry, c, conv_dim, kernel_size);

        // ── Compare outputs ──
        let mut max_diff = 0.0f32;
        for (i, (&g, &c_ref)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c_ref).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < TOL,
                "chunked conv1d nonzero-carry output[{i}] GPU={g:.8} CPU={c_ref:.8} diff={diff:.2e}"
            );
        }
        println!("test_chunked_conv1d_g1_nonzero_carry: max output diff = {max_diff:.2e}");

        // ── Compare carry ──
        for (i, (&g, &c_ref)) in gpu_carry.iter().zip(cpu_carry.iter()).enumerate() {
            let diff = (g - c_ref).abs();
            assert!(
                diff < TOL,
                "chunked conv1d nonzero-carry carry[{i}] GPU={g:.8} CPU={c_ref:.8} diff={diff:.2e}"
            );
        }
    }

    /// G1: multi-chunk — run 4 chunks sequentially, verify the carry is correctly
    /// propagated across chunk boundaries. This is the real prefill pattern.
    #[test]
    fn test_chunked_conv1d_g1_multi_chunk() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let total_tokens: usize = 64;
        let chunk_size: usize = 16;
        let conv_dim: usize = 8;
        let kernel_size: usize = 4;
        let ks_m1 = kernel_size - 1;
        let n_chunks = total_tokens.div_ceil(chunk_size);

        // Full sequence of raw inputs
        let input_raw: Vec<f32> = (0..total_tokens * conv_dim)
            .map(|i| {
                let t = i / conv_dim;
                let ch = i % conv_dim;
                0.01 * (t as f32) + 0.001 * (ch as f32) - 0.1
            })
            .collect();

        let conv_weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|i| 0.05 * ((i as f32) + 1.0))
            .collect();

        // ── GPU: process chunk by chunk ──
        let conv_weight_handle = client.create_from_slice(f32::as_bytes(&conv_weight));
        let carry_handle = client.create_from_slice(f32::as_bytes(&vec![0.0f32; conv_dim * ks_m1]));
        let mut gpu_all_outputs = vec![0.0f32; total_tokens * conv_dim];

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total_tokens);
            let this_c = end - start;

            let chunk_input = &input_raw[start * conv_dim..end * conv_dim];
            let input_handle = client.create_from_slice(f32::as_bytes(chunk_input));
            let output_handle = client.empty(this_c * conv_dim * std::mem::size_of::<f32>());

            // Compact carry layout: stride = ks-1, offset = 0.
            unsafe {
                DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                    &client,
                    input_handle,
                    output_handle.clone(),
                    conv_weight_handle.clone(),
                    carry_handle.clone(),
                    this_c,
                    conv_dim,
                    kernel_size,
                    ks_m1,
                    0,
                );
            }

            let out_bytes = client.read_one(output_handle).expect("read chunk output");
            let out = f32::from_bytes(&out_bytes);
            gpu_all_outputs[start * conv_dim..end * conv_dim].copy_from_slice(out);
        }

        // ── CPU: sequential reference over the full sequence ──
        let mut cpu_carry = vec![0.0f32; conv_dim * ks_m1];
        let cpu_output =
            cpu_sequential_conv1d_chunked(&input_raw, &conv_weight, &mut cpu_carry, total_tokens, conv_dim, kernel_size);

        // ── Compare full outputs ──
        let mut max_diff = 0.0f32;
        for (i, (&g, &c_ref)) in gpu_all_outputs.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c_ref).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < TOL,
                "multi-chunk conv1d output[{i}] GPU={g:.8} CPU={c_ref:.8} diff={diff:.2e}"
            );
        }
        println!("test_chunked_conv1d_g1_multi_chunk: max output diff = {max_diff:.2e} across {total_tokens} tokens / {n_chunks} chunks");
    }

    // ── Issue 658 Phase 2: conv_state layout G1 test ─────────────────────

    /// G1: chunked conv1d with conv_state layout (stride=ks, offset=1) produces
    /// the SAME output as the compact layout (stride=ks-1, offset=0), and the
    /// conv_state buffer ends up in the correct format for the decode path.
    ///
    /// This is the layout used by `prefill_with_layer_capture` — the chunked
    /// kernel reads/writes conv_state positions 1..ks-1, leaving position 0
    /// untouched (stale, shifted out on the next decode before use).
    #[test]
    fn test_chunked_conv1d_g1_conv_state_layout() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let c: usize = 16;
        let conv_dim: usize = 8;
        let kernel_size: usize = 4;
        let ks_m1 = kernel_size - 1;

        // Synthetic raw inputs.
        let input_raw: Vec<f32> = (0..c * conv_dim)
            .map(|i| {
                let t = i / conv_dim;
                let ch = i % conv_dim;
                0.1 * (t as f32) + 0.01 * (ch as f32) + 0.3
            })
            .collect();

        let conv_weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|i| 0.05 + 0.01 * (i as f32))
            .collect();

        // ── Run with compact layout (stride=ks-1, offset=0) ──
        let input_h1 = client.create_from_slice(f32::as_bytes(&input_raw));
        let weight_h = client.create_from_slice(f32::as_bytes(&conv_weight));
        let carry_h1 = client.create_from_slice(f32::as_bytes(&vec![0.0f32; conv_dim * ks_m1]));
        let out_h1 = client.empty(c * conv_dim * std::mem::size_of::<f32>());
        unsafe {
            DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                &client, input_h1, out_h1.clone(), weight_h.clone(),
                carry_h1.clone(), c, conv_dim, kernel_size, ks_m1, 0,
            );
        }
        let out1_bytes = client.read_one(out_h1).expect("read");
        let out1 = f32::from_bytes(&out1_bytes);
        let carry1_bytes = client.read_one(carry_h1).expect("read");
        let carry1 = f32::from_bytes(&carry1_bytes);

        // ── Run with conv_state layout (stride=ks, offset=1) ──
        // The conv_state is initialized to zeros (position 0 is never read by the
        // chunked kernel; positions 1..ks-1 are the carry).
        let input_h2 = client.create_from_slice(f32::as_bytes(&input_raw));
        let conv_state_init: Vec<f32> = vec![0.0f32; conv_dim * kernel_size];
        let conv_state_h = client.create_from_slice(f32::as_bytes(&conv_state_init));
        let out_h2 = client.empty(c * conv_dim * std::mem::size_of::<f32>());
        unsafe {
            DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                &client, input_h2, out_h2.clone(), weight_h, conv_state_h.clone(),
                c, conv_dim, kernel_size, kernel_size, 1,
            );
        }
        let out2_bytes = client.read_one(out_h2).expect("read");
        let out2 = f32::from_bytes(&out2_bytes);
        let conv_state_bytes = client.read_one(conv_state_h).expect("read");
        let conv_state_post = f32::from_bytes(&conv_state_bytes);

        // ── Compare outputs: must be identical ──
        assert_eq!(out1.len(), out2.len());
        let mut max_diff = 0.0f32;
        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < TOL,
                "conv_state layout output[{i}] mismatch: compact={a:.8} conv_state={b:.8} diff={diff:.2e}"
            );
        }
        println!("test_chunked_conv1d_g1_conv_state_layout: max output diff = {max_diff:.2e}");

        // ── Verify conv_state positions 1..ks-1 match compact carry 0..ks-2 ──
        for ch in 0..conv_dim {
            for j in 0..ks_m1 {
                let compact_val = carry1[ch * ks_m1 + j];
                let conv_state_val = conv_state_post[ch * kernel_size + j + 1];
                let diff = (compact_val - conv_state_val).abs();
                assert!(
                    diff < TOL,
                    "conv_state[{ch},{}] = {conv_state_val:.8} vs compact carry[{ch},{j}] = {compact_val:.8} diff = {diff:.2e}",
                    j + 1
                );
            }
        }
    }

    /// Issue 673 Bug C regression: partial chunk (`c < ks-1`) carry merge.
    ///
    /// The pre-fix kernel wrote the c new inputs to carry slots 0..c-1
    /// UNSHIFTED, leaving the old carry's tail entries at their old positions —
    /// corrupting the final partial chunk's carry into decode. The correct merge
    /// is `[old_carry[c..ks-1], new_inputs[0..c]]` (what the CPU reference and
    /// now the carry-update kernel do). This test uses a NON-ZERO initial carry
    /// with values disjoint from the input range so the mis-ordering is detected
    /// at full magnitude. Covers both carry layouts (compact + conv_state).
    #[test]
    fn test_chunked_conv1d_issue673_partial_chunk() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let conv_dim: usize = 8;
        let kernel_size: usize = 4;
        let ks_m1 = kernel_size - 1;

        // ── Compact layout, c = 2 (< ks_m1 = 3) ──
        let c: usize = 2;

        // Non-zero carry (values ~1.x) disjoint from the inputs (~2.x-3.x).
        let carry_init: Vec<f32> = (0..conv_dim * ks_m1)
            .map(|i| {
                let ch = i / ks_m1;
                let j = i % ks_m1;
                1.0 + 0.1 * (ch as f32) + 0.01 * (j as f32)
            })
            .collect();
        let input_raw: Vec<f32> = (0..c * conv_dim)
            .map(|i| {
                let t = i / conv_dim;
                let ch = i % conv_dim;
                2.0 + (t as f32) + 0.05 * (ch as f32)
            })
            .collect();
        let conv_weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|i| 0.05 + 0.01 * (i as f32))
            .collect();

        // CPU reference (its c < ks_m1 branch does the correct merge).
        let mut cpu_carry = carry_init.clone();
        let cpu_output =
            cpu_sequential_conv1d_chunked(&input_raw, &conv_weight, &mut cpu_carry, c, conv_dim, kernel_size);

        let input_h = client.create_from_slice(f32::as_bytes(&input_raw));
        let weight_h = client.create_from_slice(f32::as_bytes(&conv_weight));
        let carry_h = client.create_from_slice(f32::as_bytes(&carry_init));
        let out_h = client.empty(c * conv_dim * std::mem::size_of::<f32>());
        unsafe {
            DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                &client, input_h, out_h.clone(), weight_h.clone(), carry_h.clone(),
                c, conv_dim, kernel_size, ks_m1, 0,
            );
        }
        let out_bytes = client.read_one(out_h).expect("read output");
        let gpu_out = f32::from_bytes(&out_bytes);
        let carry_bytes = client.read_one(carry_h).expect("read carry");
        let gpu_carry = f32::from_bytes(&carry_bytes);

        // Output equality (the conv reads are unaffected by the carry fix).
        for (i, (&g, &r)) in gpu_out.iter().zip(cpu_output.iter()).enumerate() {
            assert!(
                (g - r).abs() < TOL,
                "partial-chunk output[{i}] GPU={g:.8} CPU={r:.8}"
            );
        }

        // Carry equality — the load-bearing assert for Bug C.
        for (i, (&g, &r)) in gpu_carry.iter().zip(cpu_carry.iter()).enumerate() {
            assert!(
                (g - r).abs() < TOL,
                "partial-chunk carry[{i}] GPU={g:.8} CPU={r:.8} \
                 (Issue 673 Bug C: expected [old_carry[c..ks-1], new_inputs[0..c]])"
            );
        }

        // ── conv_state layout, c = 1 (the production partial-chunk shape) ──
        let c1: usize = 1;
        let input_raw1: Vec<f32> = (0..c1 * conv_dim)
            .map(|i| {
                let t = i / conv_dim;
                let ch = i % conv_dim;
                2.0 + (t as f32) + 0.05 * (ch as f32)
            })
            .collect();
        let mut cpu_carry1 = carry_init.clone();
        let _ = cpu_sequential_conv1d_chunked(
            &input_raw1, &conv_weight, &mut cpu_carry1, c1, conv_dim, kernel_size,
        );

        // conv_state: positions 1..ks-1 carry the compact carry; position 0 is
        // a stale marker that must remain untouched.
        let mut conv_state_init = vec![999.0f32; conv_dim * kernel_size];
        for ch in 0..conv_dim {
            for j in 0..ks_m1 {
                conv_state_init[ch * kernel_size + j + 1] = carry_init[ch * ks_m1 + j];
            }
        }
        let input_h1 = client.create_from_slice(f32::as_bytes(&input_raw1));
        let state_h = client.create_from_slice(f32::as_bytes(&conv_state_init));
        let out_h1 = client.empty(c1 * conv_dim * std::mem::size_of::<f32>());
        unsafe {
            DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                &client, input_h1, out_h1.clone(), weight_h, state_h.clone(),
                c1, conv_dim, kernel_size, kernel_size, 1,
            );
        }
        let state_bytes = client.read_one(state_h).expect("read conv_state");
        let gpu_state = f32::from_bytes(&state_bytes);

        for ch in 0..conv_dim {
            // Stale position 0 untouched.
            assert_eq!(
                gpu_state[ch * kernel_size].to_bits(),
                999.0f32.to_bits(),
                "conv_state[{ch}, 0] must stay untouched"
            );
            for j in 0..ks_m1 {
                let got = gpu_state[ch * kernel_size + j + 1];
                let want = cpu_carry1[ch * ks_m1 + j];
                assert!(
                    (got - want).abs() < TOL,
                    "conv_state[{ch}, {}] = {got:.8} vs expected {want:.8} \
                     (Issue 673 Bug C partial-chunk merge)",
                    j + 1
                );
            }
        }
    }

    /// CPU reference: sequential DeltaNet recurrence for C tokens from a boundary state.
    ///
    /// S_t = α_t · S_{t-1} + β_t · v_t ⊗ k_t^T
    /// o_t = q_t^T · S_t / sqrt(d)
    ///
    /// Returns (outputs [C, d per head], final_state [d*d per head]).
    fn cpu_sequential_recurrence(
        q: &[f32],           // [n_head, C, d]
        k: &[f32],           // [n_head, C, d]
        v: &[f32],           // [n_head, C, d]
        alpha: &[f32],       // [n_head, C]
        beta: &[f32],        // [n_head, C]
        s_init: &[f32],      // [n_head, d, d]
        n_head: usize,
        c: usize,
        d: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = s_init.to_vec();
        let mut outputs = vec![0.0f32; n_head * c * d];
        let scale = 1.0 / (d as f32).sqrt();

        for h in 0..n_head {
            for t in 0..c {
                let s_base = h * d * d;
                let qkv_base = (h * c + t) * d;
                let a = alpha[h * c + t];
                let b = beta[h * c + t];

                // S = α·S + β·v⊗k^T
                for row in 0..d {
                    let v_row = v[qkv_base + row];
                    for col in 0..d {
                        state[s_base + row * d + col] =
                            a * state[s_base + row * d + col] + b * v_row * k[qkv_base + col];
                    }
                }

                // o_t = q_t^T · S_t / sqrt(d)
                for col in 0..d {
                    let mut dot = 0.0f32;
                    for row in 0..d {
                        dot += q[qkv_base + row] * state[s_base + row * d + col];
                    }
                    outputs[(h * c + t) * d + col] = dot * scale;
                }
            }
        }
        (outputs, state)
    }

    /// G1: full chunked recurrence pipeline matches sequential reference.
    ///
    /// Tests the complete Phase 2-3 pipeline:
    /// 1. ChunkDecay (decay_to_end + total_decay)
    /// 2. DeltaS (intra-chunk state contribution)
    /// 3. StateTransition (S_next = total_decay * S_init + ΔS)
    /// 4. ForwardDecay (decay_to_t for cross-chunk)
    /// 5. IntraOutput (causal-weighted QV·K)
    /// 6. CrossChunk (decay_to_t · Q @ S_init)
    /// 7. Output = (IntraOutput + CrossChunk) / sqrt(d)
    #[test]
    fn test_chunked_recurrence_g1_full_pipeline() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 4; // small for test speed
        let c: usize = 8;
        let d: usize = 128;
        assert!(DeltanetIntraOutputCubeCL::supports(d), "test requires d=128");
        let scale = 1.0 / (d as f32).sqrt();

        // Synthetic data with small magnitudes (to stay in linear regime of f32).
        // Q/K/V: [n_head, C, d], distinct per head+token+dim.
        let make_hcd = |offset: f32| -> Vec<f32> {
            (0..n_head * c * d)
                .map(|i| {
                    let h = i / (c * d);
                    let rem = i % (c * d);
                    let t = rem / d;
                    let dim = rem % d;
                    offset + 0.001 * (h as f32) + 0.0001 * (t as f32) + 0.00001 * (dim as f32)
                })
                .collect()
        };
        let q = make_hcd(0.01);
        let k = make_hcd(0.02);
        let v = make_hcd(0.03);

        // alpha ∈ (0, 1): use 0.9 + small variation.
        let alpha: Vec<f32> = (0..n_head * c)
            .map(|i| 0.9 + 0.001 * (i as f32 % 10.0))
            .collect();
        // beta ∈ (0, 1): use 0.5 uniform.
        let beta: Vec<f32> = vec![0.5; n_head * c];

        // Initial state: small random-ish values.
        let s_init: Vec<f32> = (0..n_head * d * d)
            .map(|i| 0.0001 * ((i as f32 % 100.0) - 50.0))
            .collect();

        // ── CPU sequential reference ──
        let (cpu_outputs, cpu_final_state) =
            cpu_sequential_recurrence(&q, &k, &v, &alpha, &beta, &s_init, n_head, c, d);

        // ── GPU chunked pipeline ──
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let k_h = client.create_from_slice(f32::as_bytes(&k));
        let v_h = client.create_from_slice(f32::as_bytes(&v));
        let alpha_h = client.create_from_slice(f32::as_bytes(&alpha));
        let beta_h = client.create_from_slice(f32::as_bytes(&beta));
        let s_init_h = client.create_from_slice(f32::as_bytes(&s_init));

        // Step 1: ChunkDecay → decay_to_end [n_head, C+1] (first C are per-token, last is total).
        let decay_to_end_h = client.empty(n_head * (c + 1) * std::mem::size_of::<f32>());
        unsafe {
            DeltanetChunkDecayCubeCL::launch::<ActiveRuntime>(
                &client, alpha_h.clone(), decay_to_end_h.clone(), n_head, c,
            );
        }

        // Step 2: DeltaS → delta_s [n_head, d, d].
        // decay_to_end for DeltaS is the first C entries per head (index 0..C, not C+1).
        // The DeltaS kernel reads decay_to_end[h, j] for j in 0..C, which is at index h*(C+1)+j.
        // But the kernel assumes [n_head, C] layout. We need to extract the first C entries.
        // For simplicity, create a separate buffer with just the first C entries per head.
        // Actually — the decay_to_end buffer is [n_head, C+1], but DeltaS expects [n_head, C].
        // The memory layout: decay_to_end[h*(C+1) .. h*(C+1)+C] are the per-token values,
        // decay_to_end[h*(C+1)+C] is total_decay. The DeltaS kernel reads at head*C + j,
        // which would read from the WRONG offset (it would read total_decay of head h-1).
        // FIX: we need a separate [n_head, C] buffer for the per-token decay_to_end.
        // Let's read back decay_to_end, extract the first C per head, re-upload.
        let decay_to_end_bytes = client.read_one(decay_to_end_h.clone()).expect("read decay_to_end");
        let decay_to_end = f32::from_bytes(&decay_to_end_bytes);
        let mut decay_to_end_c = vec![0.0f32; n_head * c];
        for h in 0..n_head {
            for j in 0..c {
                decay_to_end_c[h * c + j] = decay_to_end[h * (c + 1) + j];
            }
        }
        let total_decay: Vec<f32> = (0..n_head).map(|h| decay_to_end[h * (c + 1) + c]).collect();
        let decay_to_end_c_h = client.create_from_slice(f32::as_bytes(&decay_to_end_c));
        let total_decay_h = client.create_from_slice(f32::as_bytes(&total_decay));

        let delta_s_h = client.empty(n_head * d * d * std::mem::size_of::<f32>());
        unsafe {
            DeltanetDeltaSCubeCL::launch::<ActiveRuntime>(
                &client, k_h.clone(), v_h.clone(), beta_h.clone(),
                decay_to_end_c_h.clone(), delta_s_h.clone(), n_head, c, d,
            );
        }

        // Step 3: StateTransition → s_next [n_head, d, d].
        let s_next_h = client.empty(n_head * d * d * std::mem::size_of::<f32>());
        unsafe {
            DeltanetStateTransitionCubeCL::launch::<ActiveRuntime>(
                &client, s_init_h.clone(), delta_s_h.clone(), total_decay_h.clone(),
                s_next_h.clone(), n_head, d,
            );
        }

        // Step 4: ForwardDecay → decay_to_t [n_head, C].
        let decay_to_t_h = client.empty(n_head * c * std::mem::size_of::<f32>());
        unsafe {
            DeltanetForwardDecayCubeCL::launch::<ActiveRuntime>(
                &client, alpha_h.clone(), decay_to_t_h.clone(), n_head, c,
            );
        }

        // Step 5: IntraOutput → intra_out [n_head, C, d].
        let intra_out_h = client.empty(n_head * c * d * std::mem::size_of::<f32>());
        unsafe {
            DeltanetIntraOutputCubeCL::launch::<ActiveRuntime>(
                &client, q_h.clone(), k_h.clone(), v_h.clone(),
                alpha_h.clone(), beta_h.clone(), intra_out_h.clone(), n_head, c, d,
            );
        }

        // Step 6: CrossChunk → cross_out [n_head, C, d].
        let cross_out_h = client.empty(n_head * c * d * std::mem::size_of::<f32>());
        unsafe {
            DeltanetCrossChunkCubeCL::launch::<ActiveRuntime>(
                &client, q_h.clone(), s_init_h.clone(), decay_to_t_h.clone(),
                cross_out_h.clone(), n_head, c, d,
            );
        }

        // Step 7: Read back + combine: output = (intra + cross) * scale.
        let intra_bytes = client.read_one(intra_out_h).expect("read intra");
        let intra = f32::from_bytes(&intra_bytes);
        let cross_bytes = client.read_one(cross_out_h).expect("read cross");
        let cross = f32::from_bytes(&cross_bytes);
        let gpu_outputs: Vec<f32> = intra.iter().zip(cross.iter()).map(|(&i, &c_v)| (i + c_v) * scale).collect();

        // Read back final state.
        let s_next_bytes = client.read_one(s_next_h).expect("read s_next");
        let gpu_final_state = f32::from_bytes(&s_next_bytes);

        // ── Compare outputs ──
        // Tolerance: the chunked path uses different reduction orders (plane_sum vs
        // sequential accumulation), so we expect FP-equivalence, not bit-identity.
        // The existing recurrence kernel uses 1e-4 tolerance for the same reason.
        const REC_TOL: f32 = 1e-3;
        let mut max_out_diff = 0.0f32;
        for (&g, &c_ref) in gpu_outputs.iter().zip(cpu_outputs.iter()) {
            let diff = (g - c_ref).abs();
            if diff > max_out_diff {
                max_out_diff = diff;
            }
        }
        println!("test_chunked_recurrence_g1_full_pipeline: max output diff = {max_out_diff:.2e} (tol {REC_TOL:.0e})");
        assert!(
            max_out_diff < REC_TOL,
            "chunked recurrence output max diff {max_out_diff:.2e} exceeds tol {REC_TOL:.0e}"
        );

        // ── Compare final state ──
        let mut max_state_diff = 0.0f32;
        for (&g, &c_ref) in gpu_final_state.iter().zip(cpu_final_state.iter()) {
            let diff = (g - c_ref).abs();
            if diff > max_state_diff {
                max_state_diff = diff;
            }
        }
        println!("test_chunked_recurrence_g1_full_pipeline: max state diff = {max_state_diff:.2e} (tol {REC_TOL:.0e})");
        assert!(
            max_state_diff < REC_TOL,
            "chunked recurrence final state max diff {max_state_diff:.2e} exceeds tol {REC_TOL:.0e}"
        );
    }

    /// Issue 673 Bug A regression: the cross-chunk kernel must compute a true
    /// per-column matvec `out[h,t,col] = decay[h,t] · Σ_r q[h,t,r]·S[h,r,col]`.
    ///
    /// The pre-fix kernel used `plane_sum` across lanes whose partials covered
    /// DIFFERENT columns — producing a mod-32 staircase reduction whose output
    /// is block-constant with period 32. The G1 pipeline test above masks this
    /// because its synthetic magnitudes (q≈0.01, s≈±0.005) put the error ~20×
    /// under REC_TOL. This test amplifies magnitudes (q ∈ ±0.5, S ∈ ±0.25) so
    /// the staircase error is O(0.1-1) — far above tolerance — and additionally
    /// asserts the output is NOT block-constant (bit-identical positions 0/1
    /// per (head, token) row is the buggy kernel's fingerprint).
    #[test]
    fn test_cross_chunk_issue673_amplified_matvec() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 2;
        let c: usize = 4;
        let d: usize = 128;
        assert!(DeltanetCrossChunkCubeCL::supports(d), "test requires d=128");

        // Deterministic generic patterns (no structure shared between q and S
        // — a staircase-vs-matvec difference shows up at full magnitude).
        let q: Vec<f32> = (0..n_head * c * d)
            .map(|i| (((i * 37) % 101) as f32 - 50.0) / 100.0) // ±0.5
            .collect();
        let s: Vec<f32> = (0..n_head * d * d)
            .map(|i| (((i * 53) % 211) as f32 - 105.0) / 420.0) // ±0.25
            .collect();
        let decay_to_t: Vec<f32> = (0..n_head * c)
            .map(|i| 0.3 + 0.1 * ((i % 5) as f32))
            .collect();

        // CPU reference matvec.
        let mut cpu = vec![0.0f32; n_head * c * d];
        for h in 0..n_head {
            for t in 0..c {
                for col in 0..d {
                    let mut dot = 0.0f32;
                    for r in 0..d {
                        dot += q[(h * c + t) * d + r] * s[h * d * d + r * d + col];
                    }
                    cpu[(h * c + t) * d + col] = decay_to_t[h * c + t] * dot;
                }
            }
        }

        // GPU.
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let s_h = client.create_from_slice(f32::as_bytes(&s));
        let decay_h = client.create_from_slice(f32::as_bytes(&decay_to_t));
        let out_h = client.empty(n_head * c * d * std::mem::size_of::<f32>());
        unsafe {
            DeltanetCrossChunkCubeCL::launch::<ActiveRuntime>(
                &client, q_h, s_h, decay_h, out_h.clone(), n_head, c, d,
            );
        }
        let out_bytes = client.read_one(out_h).expect("read cross output");
        let gpu = f32::from_bytes(&out_bytes);

        // Primary check: matches the CPU matvec.
        let mut max_diff = 0.0f32;
        for (&g, &r) in gpu.iter().zip(cpu.iter()) {
            let diff = (g - r).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        println!("test_cross_chunk_issue673_amplified_matvec: max diff = {max_diff:.2e}");
        assert!(
            max_diff < 1e-3,
            "cross-chunk output diverges from CPU matvec: max diff {max_diff:.2e} \
             (Issue 673 Bug A staircase reduction?)"
        );

        // Secondary check: not block-constant. The pre-fix kernel writes the
        // plane-uniform dot0 to all 32 lanes of a block — positions 0 and 1 are
        // bit-identical. A correct matvec on generic S never is.
        for h in 0..n_head {
            for t in 0..c {
                let base = (h * c + t) * d;
                assert_ne!(
                    gpu[base].to_bits(),
                    gpu[base + 1].to_bits(),
                    "cross-chunk output is block-constant (Issue 673 Bug A)"
                );
            }
        }
    }
}
