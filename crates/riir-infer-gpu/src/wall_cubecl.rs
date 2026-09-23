//! CubeCL Wall Attention kernels.
//!
//! GPU-accelerated Wall Attention primitives for prefix-state management
//! and Q/K rescaling. Wall Attention maintains per-head-dimension prefix
//! states that accumulate gate values across the sequence, then rescales
//! Q and K vectors to modulate attention.
//!
//! # Kernels
//!
//! | Kernel | Algorithm | Dispatch |
//! |--------|-----------|----------|
//! | `wall_gate_project_f32` | GEMV + sigmoid gate projection | `ceil(gate_proj_dim/64)` WG × 64 threads |
//! | `wall_prefix_decode_f32` | Prefix state update (single step) | `ceil(head_dim/256)` WG × 256 threads |
//! | `wall_prefix_prefill_f32` | Prefix state cumulative sum (full sequence) | `ceil(head_dim/256)` WG × 256 threads |
//! | `wall_rescale_f32` | Q/K rescaling by exp(±prefix) | `ceil((q_dim+kv_dim)/256)` WG × 256 threads |
//! | `wall_rescale_from_combined_f32` | Q/K rescaling from fused QKV buffer | `ceil(section_len/256)` WG × 256 threads |
//!
//! # CubeCL v0.10 Constraints
//!
//! - 3-4 `Array<f32>` parameters per kernel (within project limit).
//! - No conditional expressions as values — use `if { }` statements.
//! - `ABSOLUTE_POS` is `usize`, `UNIT_POS` is `u32` — cast appropriately.
//! - `f32::new(literal)` for constants in `#[cube]` context.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.
//! - Use `while` loops instead of `for` loops.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Kernel 1: Gate projection (GEMV + sigmoid + clamp)
// ---------------------------------------------------------------------------

/// CubeCL Wall Attention gate projection kernel.
///
/// Computes the gate vector for a single decode step:
/// ```text
/// logit = Σ_j w_g[i * d_model + j] * hidden[j] + bias
/// gate[i] = min(sigmoid(logit), gate_max)
/// ```
///
/// Each thread handles one gate element. The dot product is accumulated
/// with a while loop over d_model dimensions (matching attention_decode_f32
/// dot-product pattern).
///
/// ## Parameter Layout
///
/// - `hidden`: `[f32; d_model]` — input hidden state
/// - `w_g`: `[f32; gate_proj_dim * d_model]` — gate projection weights (row-major)
/// - `gate`: `[f32; gate_proj_dim]` — output gate values
/// - `params`: `[f32; 4]` — `[d_model, gate_proj_dim, bias, gate_max]`
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(gate_proj_dim/64), 1, 1)`, `CubeDim::new_1d(64)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn wall_gate_project_f32(
    hidden: &[f32],
    w_g: &[f32],
    gate: &mut [f32],
    params: &[f32],
) {
    // params[0] = d_model, params[1] = gate_proj_dim, params[2] = bias, params[3] = gate_max
    let d_model = params[0usize] as u32;
    let gate_proj_dim = params[1usize] as u32;
    let bias = params[2usize];
    let gate_max = params[3usize];

    let tid = ABSOLUTE_POS;
    let gate_proj_dim_usize = gate_proj_dim as usize;

    if tid >= gate_proj_dim_usize {
        terminate!();
    }

    let tid_u32 = tid as u32;

    // Dot product: row i of w_g (length d_model) dotted with hidden
    let row_offset = tid_u32 * d_model;
    let mut dot = f32::new(0.0f32);
    let mut j = 0u32;
    while j < d_model {
        dot += w_g[(row_offset + j) as usize] * hidden[j as usize];
        j += 1u32;
    }

    let logit = dot + bias;

    // sigmoid(x) = 1.0 / (1.0 + exp(-x))
    let one = f32::new(1.0f32);
    let sig = one / (one + f32::exp(-logit));

    // gate[i] = min(sigmoid(logit), gate_max)
    let mut result = sig;
    if sig > gate_max {
        result = gate_max;
    }
    gate[tid] = result;
}

// ---------------------------------------------------------------------------
// Kernel 2: Prefix decode (single-step update)
// ---------------------------------------------------------------------------

/// CubeCL Wall Attention prefix decode kernel.
///
/// Updates the prefix state for a single decode step:
/// ```text
/// prefix_curr[tid] = prefix_prev[tid] + log_gate[tid]
/// ```
///
/// This is a simple elementwise addition applied to the per-head-dimension
/// prefix accumulation state.
///
/// ## Parameter Layout
///
/// - `log_gate`: `[f32; head_dim]` — log gate values for this step
/// - `prefix_prev`: `[f32; head_dim]` — previous prefix state
/// - `prefix_curr`: `[f32; head_dim]` — updated prefix state
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(head_dim/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn wall_prefix_decode_f32(
    log_gate: &[f32],
    prefix_prev: &[f32],
    prefix_curr: &mut [f32],
) {
    let n = log_gate.len();
    let tid = ABSOLUTE_POS;

    if tid >= n {
        terminate!();
    }

    prefix_curr[tid] = prefix_prev[tid] + log_gate[tid];
}

// ---------------------------------------------------------------------------
// Kernel 3: Prefix prefill (sequential cumulative sum)
// ---------------------------------------------------------------------------

/// CubeCL Wall Attention prefix prefill kernel.
///
/// Computes cumulative prefix sums across a full sequence during prefill:
/// ```text
/// running = 0.0
/// for t in 0..seq_len:
///     running += log_gates[t * head_dim + d]
///     prefix_sums[t * head_dim + d] = running
/// ```
///
/// Each thread handles one head_dim element (dimension `d`) and scans
/// all `seq_len` positions sequentially. This is efficient because:
/// - head_dim (e.g. 256) threads run in parallel
/// - Each thread does O(seq_len) sequential work (no sync needed)
///
/// ## Parameter Layout
///
/// - `log_gates`: `[f32; seq_len * head_dim]` — log gate values for all positions
/// - `prefix_sums`: `[f32; seq_len * head_dim]` — output cumulative sums
/// - `params`: `[f32; 2]` — `[seq_len, head_dim]`
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(head_dim/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn wall_prefix_prefill_f32(
    log_gates: &[f32],
    prefix_sums: &mut [f32],
    params: &[f32],
) {
    // params[0] = seq_len, params[1] = head_dim
    let seq_len = params[0usize] as u32;
    let head_dim = params[1usize] as u32;

    let tid = ABSOLUTE_POS;
    let head_dim_usize = head_dim as usize;

    if tid >= head_dim_usize {
        terminate!();
    }

    let d = tid as u32;

    let mut running = f32::new(0.0f32);
    let mut t = 0u32;
    while t < seq_len {
        running += log_gates[(t * head_dim + d) as usize];
        prefix_sums[(t * head_dim + d) as usize] = running;
        t += 1u32;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Q/K rescaling
// ---------------------------------------------------------------------------

/// CubeCL Wall Attention Q/K rescaling kernel.
///
/// Rescales Q and K vectors using the accumulated prefix states:
/// ```text
/// total = q_dim + kv_dim
/// if tid < q_dim:
///     d = tid % head_dim
///     q[tid] *= exp(prefix_qk[d])              // prefix_qk[d] = prefix_q[d]
/// else:
///     k_idx = tid - q_dim
///     d = k_idx % head_dim
///     k[k_idx] *= exp(-prefix_qk[head_dim + d]) // prefix_qk[head_dim+d] = prefix_k[d]
/// ```
///
/// This applies the Wall Attention modulation: Q is scaled up by the
/// prefix state and K is scaled down, implementing a form of attention
/// gating based on the accumulated prefix information.
///
/// ## Parameter Layout (4 arrays to stay within CubeCL v0.10 limit)
///
/// - `q`: `[f32; q_dim]` — query vector (n_heads * head_dim), mutated in-place
/// - `k`: `[f32; kv_dim]` — key vector (n_kv_heads * head_dim), mutated in-place
/// - `prefix_qk`: `[f32; 2 * head_dim]` — concatenated `[prefix_q | prefix_k]`
/// - `params`: `[f32; 3]` — `[q_dim, kv_dim, head_dim]`
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil((q_dim+kv_dim)/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn wall_rescale_f32(
    q: &mut [f32],
    k: &mut [f32],
    prefix_qk: &[f32],
    params: &[f32],
) {
    // params[0] = q_dim (n_heads * head_dim)
    // params[1] = kv_dim (n_kv_heads * head_dim)
    // params[2] = head_dim
    // prefix_qk layout: [prefix_q(head_dim) | prefix_k(head_dim)]
    let q_dim = params[0usize] as u32;
    let kv_dim = params[1usize] as u32;
    let head_dim = params[2usize] as u32;

    let total = (q_dim + kv_dim) as usize;
    let tid = ABSOLUTE_POS;

    if tid >= total {
        terminate!();
    }

    let tid_u32 = tid as u32;

    if tid_u32 < q_dim {
        // This thread handles a Q element
        let d = tid_u32 - (tid_u32 / head_dim) * head_dim;
        q[tid] = q[tid] * f32::exp(prefix_qk[d as usize]);
    }
    if tid_u32 >= q_dim {
        // This thread handles a K element
        let k_idx = tid_u32 - q_dim;
        let d = k_idx - (k_idx / head_dim) * head_dim;
        k[k_idx as usize] = k[k_idx as usize] * f32::exp(-prefix_qk[(head_dim + d) as usize]);
    }
}

// ---------------------------------------------------------------------------
// Launcher structs
// ---------------------------------------------------------------------------

/// CubeCL Wall Attention gate projection launcher.
#[cfg(feature = "cubecl_runtime")]
pub struct WallGateProjectCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl WallGateProjectCubeCL {
    /// Launch gate projection kernel.
    ///
    /// Dispatch: `ceil(gate_proj_dim/64)` workgroups of 64 threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `hidden_handle`: `d_model` f32 elements
    /// - `w_g_handle`: `gate_proj_dim * d_model` f32 elements
    /// - `gate_handle`: `gate_proj_dim` f32 elements
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        hidden_handle: Handle,
        hidden_len: usize,
        w_g_handle: Handle,
        w_g_len: usize,
        gate_handle: Handle,
        gate_len: usize,
        d_model: usize,
        gate_proj_dim: usize,
    ) {
        let cube_dim = 64u32;
        let num_wg = (gate_proj_dim as u32).div_ceil(cube_dim).max(1);
        let bias: f32 = 0.0;
        let gate_max: f32 = 1.0;
        let params: [f32; 4] = [d_model as f32, gate_proj_dim as f32, bias, gate_max];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            wall_gate_project_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(hidden_handle, hidden_len),
                BufferArg::from_raw_parts(w_g_handle, w_g_len),
                BufferArg::from_raw_parts(gate_handle, gate_len),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

/// CubeCL Wall Attention prefix decode launcher.
#[cfg(feature = "cubecl_runtime")]
pub struct WallPrefixDecodeCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl WallPrefixDecodeCubeCL {
    /// Launch prefix decode kernel (single-step prefix update).
    ///
    /// Dispatch: `ceil(head_dim/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `log_gate_handle`: `head_dim` f32 elements
    /// - `prefix_prev_handle`: `head_dim` f32 elements
    /// - `prefix_curr_handle`: `head_dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        log_gate_handle: Handle,
        prefix_prev_handle: Handle,
        prefix_curr_handle: Handle,
        head_dim: usize,
    ) {
        let num_wg = (head_dim as u32).div_ceil(256u32).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            wall_prefix_decode_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(log_gate_handle, head_dim),
                BufferArg::from_raw_parts(prefix_prev_handle, head_dim),
                BufferArg::from_raw_parts(prefix_curr_handle, head_dim),
            );
        }
    }
}

/// CubeCL Wall Attention prefix prefill launcher.
#[cfg(feature = "cubecl_runtime")]
pub struct WallPrefixPrefillCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl WallPrefixPrefillCubeCL {
    /// Launch prefix prefill kernel (sequential cumulative sum across sequence).
    ///
    /// Dispatch: `ceil(head_dim/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `log_gates_handle`: `seq_len * head_dim` f32 elements
    /// - `prefix_sums_handle`: `seq_len * head_dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        log_gates_handle: Handle,
        log_gates_len: usize,
        prefix_sums_handle: Handle,
        prefix_sums_len: usize,
        seq_len: usize,
        head_dim: usize,
    ) {
        let num_wg = (head_dim as u32).div_ceil(256u32).max(1);
        let params: [f32; 2] = [seq_len as f32, head_dim as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            wall_prefix_prefill_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(log_gates_handle, log_gates_len),
                BufferArg::from_raw_parts(prefix_sums_handle, prefix_sums_len),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// CubeCL Wall Attention Q/K rescaling launcher.
#[cfg(feature = "cubecl_runtime")]
pub struct WallRescaleCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl WallRescaleCubeCL {
    /// Launch Q/K rescaling kernel.
    ///
    /// Dispatch: `ceil((q_dim+kv_dim)/256)` workgroups of 256 threads.
    ///
    /// The kernel uses a concatenated `[prefix_q | prefix_k]` buffer to stay
    /// within the 4-array CubeCL v0.10 parameter limit. The caller must provide
    /// `prefix_qk_handle` of length `2 * head_dim` with layout:
    /// `[prefix_q(head_dim) | prefix_k(head_dim)]`.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `q_handle`: `q_dim` f32 elements (n_heads * head_dim)
    /// - `k_handle`: `kv_dim` f32 elements (n_kv_heads * head_dim)
    /// - `prefix_qk_handle`: `2 * head_dim` f32 elements (concatenated)
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        q_len: usize,
        k_handle: Handle,
        k_len: usize,
        prefix_qk_handle: Handle,
        prefix_qk_len: usize,
        q_dim: usize,
        kv_dim: usize,
        head_dim: usize,
    ) {
        let total = q_dim + kv_dim;
        let num_wg = (total as u32).div_ceil(256u32).max(1);
        let params: [f32; 3] = [q_dim as f32, kv_dim as f32, head_dim as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            wall_rescale_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(q_handle, q_len),
                BufferArg::from_raw_parts(k_handle, k_len),
                BufferArg::from_raw_parts(prefix_qk_handle, prefix_qk_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: Q/K rescaling from combined buffer
// ---------------------------------------------------------------------------

/// CubeCL Wall Attention Q/K rescaling kernel (from combined buffer).
///
/// Rescales a sub-section of a fused QKV buffer using the accumulated prefix
/// states. This is the "from combined" variant of [`wall_rescale_f32`], analogous
/// to how `rope_from_combined_f32` relates to `rope_f32`.
///
/// For a section at `combined[section_offset..section_offset+section_len]`:
/// ```text
/// if !is_k:
///     d = tid % head_dim
///     output[tid] = combined[section_offset + tid] * exp(prefix_qk[d])
/// if is_k:
///     d = tid % head_dim
///     output[tid] = combined[section_offset + tid] * exp(-prefix_qk[head_dim + d])
/// ```
///
/// ## Parameter Layout (4 arrays — CubeCL v0.10 limit)
///
/// - `combined`: `[f32; total_qkv]` — full QKV buffer (read-only)
/// - `prefix_qk`: `[f32; 2 * head_dim]` — concatenated `[prefix_q | prefix_k]` (read-only)
/// - `output`: `[f32; section_len]` — rescaled output
/// - `params`: `[f32; 5]` — `[total_qkv, section_offset, section_len, head_dim, is_k]`
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(section_len/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn wall_rescale_from_combined_f32(
    combined: &[f32],
    prefix_qk: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    // params[0] = total_qkv (unused in kernel, for buffer sizing)
    // params[1] = section_offset
    // params[2] = section_len
    // params[3] = head_dim
    // params[4] = is_k (0.0 = Q section, 1.0 = K section)
    let section_offset = params[1usize] as u32;
    let section_len = params[2usize] as u32;
    let head_dim = params[3usize] as u32;
    let is_k_raw = params[4usize];
    let zero = f32::new(0.0f32);

    let tid = ABSOLUTE_POS;
    let section_len_usize = section_len as usize;

    if tid >= section_len_usize {
        terminate!();
    }

    let tid_u32 = tid as u32;

    // d = tid % head_dim
    let head_idx = tid_u32 / head_dim;
    let d = tid_u32 - head_idx * head_dim;

    // Read from combined buffer at section_offset + tid
    let src_idx = (section_offset + tid_u32) as usize;
    let val = combined[src_idx];

    // Branch on is_k to select scaling direction
    if is_k_raw == zero {
        // Q section: multiply by exp(prefix_qk[d])
        output[tid] = val * f32::exp(prefix_qk[d as usize]);
    }
    if is_k_raw != zero {
        // K section: multiply by exp(-prefix_qk[head_dim + d])
        output[tid] = val * f32::exp(-prefix_qk[(head_dim + d) as usize]);
    }
}

/// CubeCL Wall Attention Q/K rescaling-from-combined launcher.
///
/// Wraps the `wall_rescale_from_combined_f32` kernel for rescaling a sub-section
/// of the combined QKV buffer produced by the fused triple QKV GEMV.
#[cfg(feature = "cubecl_runtime")]
pub struct WallRescaleFromCombinedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl WallRescaleFromCombinedCubeCL {
    /// Launch rescaling kernel on a sub-section of a combined QKV buffer.
    ///
    /// Reads from `combined[section_offset..section_offset+section_len]`,
    /// applies Wall Attention rescaling, and writes to `output[section_len]`.
    ///
    /// When `is_k` is false, applies Q scaling: `output = combined * exp(prefix_qk)`.
    /// When `is_k` is true, applies K scaling: `output = combined * exp(-prefix_qk)`.
    ///
    /// Dispatch: `ceil(section_len/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `combined` must have at least `section_offset + section_len` f32 elements
    /// - `prefix_qk_handle` must have `2 * head_dim` f32 elements
    ///   (concatenated `[prefix_qk(head_dim) | prefix_k(head_dim)]`)
    /// - `output_handle` must have `section_len` f32 elements
    /// - `section_offset` must be aligned to `head_dim` boundary
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        combined_handle: Handle,
        total_qkv: usize,
        prefix_qk_handle: Handle,
        prefix_qk_len: usize,
        output_handle: Handle,
        section_offset: usize,
        section_len: usize,
        head_dim: usize,
        is_k: bool,
    ) {
        let num_wg = (section_len as u32).div_ceil(256u32).max(1);
        let is_k_f32 = if is_k { 1.0f32 } else { 0.0f32 };
        let params: [f32; 5] = [
            total_qkv as f32,
            section_offset as f32,
            section_len as f32,
            head_dim as f32,
            is_k_f32,
        ];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            wall_rescale_from_combined_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(combined_handle, total_qkv),
                BufferArg::from_raw_parts(prefix_qk_handle, prefix_qk_len),
                BufferArg::from_raw_parts(output_handle, section_len),
                BufferArg::from_raw_parts(params_handle, 5),
            );
        }
    }
}
