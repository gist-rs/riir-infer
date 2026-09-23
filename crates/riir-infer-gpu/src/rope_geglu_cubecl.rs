//! CubeCL RoPE and GeGLU kernels (Plan 106 T2.13).
//!
//! GPU-accelerated Rotary Position Embedding and GeGLU activation for Gemma 2 decode.
//! Replaces CPU fallback ops to eliminate sync points in the hybrid forward pass.
//!
//! # Kernels
//!
//! | Kernel | Algorithm | Dispatch | SharedMemory |
//! |--------|-----------|----------|--------------|
//! | `rope_f32` | Interleaved rotation with precomputed cos/sin | `ceil(n/256)` WG × 256 threads | None |
//! | `geglu_f32` | Element-wise gate * GELU(gate) * up | `ceil(n/256)` WG × 256 threads | None |
//!
//! # CubeCL v0.10 Constraints
//!
//! - 3 Array parameters for both kernels (within 3-4 limit).
//! - No conditional expressions as values — use `if { }` statements.
//! - `ABSOLUTE_POS` is `usize`, `UNIT_POS` is `u32` — cast appropriately.
//! - `f32::new(literal)` for constants in `#[cube]` context.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.
//!
//! # RoPE cos/sin Table
//!
//! The `rope_f32` kernel takes a precomputed cos/sin table of length `head_dim`:
//! - `cos_sin[2*d]   = cos(pos * freq[d])`
//! - `cos_sin[2*d+1] = sin(pos * freq[d])`
//!
//! where `freq[d] = 1 / theta^(2*d / head_dim)`.
//!
//! Use [`precompute_rope_cos_sin`] to generate this table on CPU.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// RoPE-from-combined kernel (for fused QKV)
// ---------------------------------------------------------------------------

/// CubeCL RoPE kernel for reading from a sub-section of a combined QKV buffer.
///
/// Applies rotary position embedding to a contiguous section within a larger
/// combined `[Q | K | V]` buffer, writing the RoPE'd result to a separate output.
///
/// This is used after the fused triple QKV GEMV (`gemv_qkv_plane_f16`) which
/// produces a combined `[Q(q_dim) | K(kv_dim) | V(kv_dim)]` output. We need
/// separate Q and K outputs with RoPE applied for downstream attention and
/// KV store operations.
///
/// ## Algorithm
///
/// Matches the existing `rope_f32` interleaved rotation exactly:
/// - For each thread `idx` in `[0, section_len)`:
///   - Compute position within head using `head_dim`
///   - Look up cos/sin from precomputed table at pair index `d`
///   - Even element: `output[idx] = x * cos - x_next * sin`
///   - Odd element:  `output[idx] = x_prev * sin + x * cos`
///
/// ## Parameter Layout
///
/// - `input_combined`: `[f32; total_qkv]` — combined QKV buffer
/// - `cos_sin`: `[f32; head_dim]` — precomputed cos/sin table
/// - `output`: `[f32; section_len]` — RoPE'd output for this section
/// - `params`: `[f32; 3]` — `[section_offset, section_len, head_dim]`
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(section_len/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rope_from_combined_f32(
    input_combined: &[f32],
    cos_sin: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    // params[0] = section_offset (byte offset into combined buffer where this section starts)
    // params[1] = section_len   (number of elements in this section)
    // params[2] = head_dim
    let section_offset = params[0usize] as u32;
    let section_len = params[1usize] as u32;
    let head_dim_u32 = params[2usize] as u32;

    let tid = ABSOLUTE_POS;
    let section_len_usize = section_len as usize;

    if tid >= section_len_usize {
        terminate!();
    }

    let tid_u32 = tid as u32;

    // Position within head — same logic as rope_f32
    let head_idx = tid_u32 / head_dim_u32;
    let within_head = tid_u32 - head_idx * head_dim_u32;
    let pair_d = within_head / 2u32;
    let is_odd = within_head - pair_d * 2u32;

    // cos at cos_sin[2*d], sin at cos_sin[2*d + 1]
    let cos_val = cos_sin[(2u32 * pair_d) as usize];
    let sin_val = cos_sin[(2u32 * pair_d + 1u32) as usize];

    // Read from combined buffer at section_offset + tid
    let src_idx = (section_offset + tid_u32) as usize;
    let x = input_combined[src_idx];

    // Even element of pair: output = x * cos - x_next * sin
    if is_odd == 0u32 {
        let x_next = input_combined[(section_offset + tid_u32 + 1u32) as usize];
        output[tid] = x * cos_val - x_next * sin_val;
    }
    // Odd element of pair: output = x_prev * sin + x * cos
    if is_odd == 1u32 {
        let x_prev = input_combined[(section_offset + tid_u32 - 1u32) as usize];
        output[tid] = x_prev * sin_val + x * cos_val;
    }
}

// ---------------------------------------------------------------------------
// RoPE kernel
// ---------------------------------------------------------------------------

/// CubeCL RoPE (Rotary Position Embedding) kernel, pairing-generic.
///
/// Applies rotary position embedding to Q or K vectors via output array.
/// Each thread handles one element, reading its pair partner for the rotation.
///
/// ## Pairing conventions (Issue 435)
///
/// The partner of an element is `pair_stride` slots away. Both conventions
/// rotate the same `head_dim / 2` angle pairs — they differ only in *which*
/// two components form each pair:
///
/// | `pair_stride`  | Convention                       | Pairs                       |
/// |----------------|----------------------------------|-----------------------------|
/// | `1`            | [`RopePairing::Interleaved`]     | `(0,1), (2,3), …`           |
/// | `head_dim / 2` | [`RopePairing::RotateHalf`]      | `(0, D/2), (1, D/2+1), …`   |
///
/// For pair index `d` at rotation angle `pos * freq[d]`, with `lo`/`hi` the two
/// paired components:
/// - `output[lo] = input[lo] * cos - input[hi] * sin`
/// - `output[hi] = input[lo] * sin + input[hi] * cos`
///
/// `RotateHalf` is what HuggingFace's `rotate_half` (and therefore
/// `riir_infer_core::rope`) implements, and is the convention the Gemma-2 GGUF
/// weights were trained under. Getting it wrong does not crash — it silently
/// scrambles attention, which is exactly the failure Issue 435 recorded
/// (step-1 CE 6.33 on GPU vs 0.79 on CPU for the same sample).
///
/// ## Parameter Layout
///
/// - `input`: `[f32; n_heads * head_dim]` — Q or K vector.
/// - `cos_sin`: `[f32; head_dim]` — precomputed cos/sin table, indexed by pair
///   index `d` (`cos` at `2d`, `sin` at `2d+1`) under **both** conventions.
/// - `params`: `[f32; 1]` — `[pair_stride]`.
/// - `output`: `[f32; n_heads * head_dim]` — RoPE-applied output.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rope_f32(
    input: &[f32],
    cos_sin: &[f32],
    params: &[f32],
    output: &mut [f32],
) {
    let n = input.len();
    let head_dim_u32 = cos_sin.len() as u32;
    let pair_stride = params[0usize] as u32;
    let tid = ABSOLUTE_POS;

    if tid >= n {
        terminate!();
    }

    let tid_u32 = tid as u32;

    // Position within head — avoids % operator using subtraction trick:
    // within_head = tid - (tid / head_dim) * head_dim
    let head_idx = tid_u32 / head_dim_u32;
    let within_head = tid_u32 - head_idx * head_dim_u32;

    // Pairing-generic decode: each group spans `2 * pair_stride` components and
    // holds `pair_stride` consecutive pairs. `is_hi` selects which half of the
    // group this element sits in; `pair_d` is its rotation-angle index.
    //   stride = 1        → group = 2,        pair_d = within_head / 2
    //   stride = head_dim/2 → group = head_dim, pair_d = within_head % (head_dim/2)
    let group = 2u32 * pair_stride;
    let group_idx = within_head / group;
    let within_group = within_head - group_idx * group;
    let is_hi = within_group / pair_stride;
    let pair_d = group_idx * pair_stride + within_group - is_hi * pair_stride;

    // cos at cos_sin[2*d], sin at cos_sin[2*d + 1]
    let cos_val = cos_sin[(2u32 * pair_d) as usize];
    let sin_val = cos_sin[(2u32 * pair_d + 1u32) as usize];

    let x = input[tid];

    // Low element of pair: output = x * cos - x_hi * sin
    if is_hi == 0u32 {
        let x_hi = input[(tid_u32 + pair_stride) as usize];
        output[tid] = x * cos_val - x_hi * sin_val;
    }
    // High element of pair: output = x_lo * sin + x * cos
    if is_hi == 1u32 {
        let x_lo = input[(tid_u32 - pair_stride) as usize];
        output[tid] = x_lo * sin_val + x * cos_val;
    }
}

// ---------------------------------------------------------------------------
// Batched RoPE kernel (Plan 482 T5) — applies RoPE to [seq_len × n_heads × head_dim]
// ---------------------------------------------------------------------------

/// CubeCL batched RoPE kernel — applies position-dependent rotation to all
/// positions in a sequence in one dispatch.
///
/// Input layout: `[seq_len, n_heads, head_dim]` row-major.
/// Cos/sin table: `[seq_len, head_dim]` — pre-computed for ALL positions.
///
/// Each thread handles one element. The position is derived from the element
/// index: `pos = tid / (n_heads * head_dim)`.
///
/// Pairing is selected by `pair_stride` exactly as in [`rope_f32`] — see that
/// kernel's docs for the `Interleaved` vs `RotateHalf` table (Issue 435).
///
/// ## Parameter Layout
///
/// - `input`: `[f32; seq_len * n_heads * head_dim]`
/// - `cos_sin_all`: `[f32; seq_len * head_dim]` — cos/sin for all positions
/// - `params`: `[f32; 3]` — `[head_dim, row_stride, pair_stride]` where
///   `row_stride = n_heads * head_dim`
/// - `output`: `[f32; seq_len * n_heads * head_dim]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rope_batched_f32(
    input: &[f32],
    cos_sin_all: &[f32],
    params: &[f32],
    output: &mut [f32],
) {
    let head_dim_u32 = params[0usize] as u32;
    let row_stride = params[1usize] as u32; // n_heads * head_dim
    let pair_stride = params[2usize] as u32;
    let n = input.len();
    let tid = ABSOLUTE_POS;

    if tid >= n {
        terminate!();
    }

    let tid_u32 = tid as u32;

    // Which position (row) and offset within the row
    let pos = tid_u32 / row_stride;
    let within_row = tid_u32 - pos * row_stride;

    // Position within head (same decode as rope_f32)
    let head_idx = within_row / head_dim_u32;
    let within_head = within_row - head_idx * head_dim_u32;
    let group = 2u32 * pair_stride;
    let group_idx = within_head / group;
    let within_group = within_head - group_idx * group;
    let is_hi = within_group / pair_stride;
    let pair_d = group_idx * pair_stride + within_group - is_hi * pair_stride;

    // Look up cos/sin from the per-position table
    let cos_sin_offset = pos * head_dim_u32;
    let cos_val = cos_sin_all[(cos_sin_offset + 2u32 * pair_d) as usize];
    let sin_val = cos_sin_all[(cos_sin_offset + 2u32 * pair_d + 1u32) as usize];

    let x = input[tid];

    // Low element of pair: output = x * cos - x_hi * sin
    if is_hi == 0u32 {
        let x_hi = input[(tid_u32 + pair_stride) as usize];
        output[tid] = x * cos_val - x_hi * sin_val;
    }
    // High element of pair: output = x_lo * sin + x * cos
    if is_hi == 1u32 {
        let x_lo = input[(tid_u32 - pair_stride) as usize];
        output[tid] = x_lo * sin_val + x * cos_val;
    }
}

// ---------------------------------------------------------------------------
// GeGLU kernel
// ---------------------------------------------------------------------------

/// CubeCL GeGLU activation kernel.
///
/// Computes `output[i] = GELU_tanh(gate[i]) * up[i]` element-wise.
///
/// The full GeGLU formula: `GELU(gate) * up` where:
/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// Note: `GELU(x)` already includes the `x` factor, so we do NOT multiply by `gate` again.
///
/// ## Parameter Layout
///
/// - `gate`: `[f32; n]` — gate projection output.
/// - `up`: `[f32; n]` — up projection output.
/// - `output`: `[f32; n]` — GeGLU result.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn geglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    let tid = ABSOLUTE_POS;

    if tid >= n {
        terminate!();
    }

    let g = gate[tid];
    let u = up[tid];

    // GELU tanh approximation (inline):
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let sqrt_2_over_pi = f32::new(0.797_884_6_f32);
    let coeff = f32::new(0.044715f32);
    let half = f32::new(0.5f32);
    let one = f32::new(1.0f32);

    let x_cubed = g * g * g;
    let inner = sqrt_2_over_pi * (g + coeff * x_cubed);
    let tanh_val = f32::tanh(inner);
    let gelu = half * g * (one + tanh_val);

    output[tid] = gelu * u;
}

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------

/// Which two components of a head form each rotation pair (Issue 435).
///
/// Both conventions rotate the same `head_dim / 2` angles and share the same
/// cos/sin table layout; they differ only in the pairing, so a mismatch between
/// the forward kernel and the weights' training convention is silent — it
/// degrades quality instead of erroring.
///
/// Pick the one the checkpoint was trained under:
///
/// - [`RopePairing::RotateHalf`] — HuggingFace `rotate_half`, GGUF NEOX-style,
///   and what `riir_infer_core::rope` (the CPU reference for every model in this
///   workspace) implements. **Gemma-2 needs this.**
/// - [`RopePairing::Interleaved`] — adjacent `(2d, 2d+1)` pairs, the GPT-NeoX
///   "normal" mode. Kept because the LLaMA CubeCL path ships against it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopePairing {
    /// Adjacent pairs: `(0,1), (2,3), …`.
    Interleaved,
    /// Half-split pairs: `(0, D/2), (1, D/2+1), …`.
    RotateHalf,
}

impl RopePairing {
    /// Distance between the two components of a pair, in f32 slots.
    #[inline]
    pub fn stride(self, head_dim: usize) -> usize {
        match self {
            Self::Interleaved => 1,
            Self::RotateHalf => head_dim / 2,
        }
    }
}

/// Precompute cos/sin table for RoPE at a given position.
///
/// Returns `Vec<f32>` of length `head_dim`:
/// - `table[2*d]   = cos(pos * freq[d])`
/// - `table[2*d+1] = sin(pos * freq[d])`
///
/// where `freq[d] = 1 / theta^(2*d / head_dim)`.
///
/// Pass this table as the `cos_sin` parameter to [`RopeCubeCL::launch`].
#[cfg(feature = "cubecl_runtime")]
pub fn precompute_rope_cos_sin(pos: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut table = vec![0.0f32; head_dim];
    for d in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        table[2 * d] = angle.cos();
        table[2 * d + 1] = angle.sin();
    }
    table
}

/// Position-keyed cache for the RoPE cos/sin table + its GPU handle.
///
/// The cos/sin values depend only on `(pos, head_dim, theta)`. During a single
/// forward pass `pos` is identical across all layers and across Q/K, so the
/// table is computed and uploaded once and then reused `n_layer * 2` times.
///
/// Without this cache each `dispatch_rope_gpu` call recomputes `head_dim / 2`
/// `powf` calls **and** re-uploads a fresh GPU buffer. For Gemma 2 (26 layers,
/// `head_dim = 256`) that is `26 * 2 = 52` redundant `powf` batches plus 52
/// redundant GPU buffer allocations per generated token.
///
/// The cache stores a single `(pos, table, handle)` entry. On a cache hit the
/// stored GPU [`Handle`] is cloned — `Handle` is reference-counted so the clone
/// is cheap and shares the underlying GPU allocation.
#[cfg(feature = "cubecl_runtime")]
pub struct RopeCosSinCache {
    pos: usize,
    table: Vec<f32>,
    handle: Option<Handle>,
}

#[cfg(feature = "cubecl_runtime")]
impl RopeCosSinCache {
    /// Create an empty cache (no position computed yet).
    pub fn new() -> Self {
        Self {
            pos: usize::MAX,
            table: Vec::new(),
            handle: None,
        }
    }

    /// Return a cloned GPU handle for the cos/sin table at `pos`.
    ///
    /// Recomputes the table and re-uploads only when `pos` changes; otherwise
    /// clones the cached handle (cheap — `Handle` is reference-counted).
    pub fn get_or_compute(
        &mut self,
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> Handle {
        if self.pos != pos || self.handle.is_none() {
            self.table = precompute_rope_cos_sin(pos, head_dim, theta);
            self.handle = Some(client.create_from_slice(f32::as_bytes(&self.table)));
            self.pos = pos;
        }
        self.handle.as_ref().expect("handle populated above").clone()
    }
}

#[cfg(feature = "cubecl_runtime")]
impl Default for RopeCosSinCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Launcher structs
// ---------------------------------------------------------------------------

/// CubeCL RoPE launcher.
///
/// Wraps the `rope_f32` kernel with precomputed cos/sin table.
#[cfg(feature = "cubecl_runtime")]
pub struct RopeCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in T2.13+ forward pass wiring
impl RopeCubeCL {
    /// Launch RoPE kernel: applies rotary position embedding to Q or K.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `input_handle`: `n_heads * head_dim` f32 elements
    /// - `cos_sin_handle`: `head_dim` f32 elements (from [`precompute_rope_cos_sin`])
    /// - `output_handle`: `n_heads * head_dim` f32 elements
    ///
    /// `head_dim` must be even and > 0.
    ///
    /// `pairing` must match the convention the weights were trained under —
    /// see [`RopePairing`].
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        cos_sin_handle: Handle,
        output_handle: Handle,
        n: usize,
        head_dim: usize,
        pairing: RopePairing,
    ) {
        let num_wg = (n as u32).div_ceil(256u32).max(1);
        let params: [f32; 1] = [pairing.stride(head_dim) as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            rope_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(cos_sin_handle, head_dim),
                BufferArg::from_raw_parts(params_handle, 1),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

/// CubeCL RoPE-from-combined launcher.
///
/// Wraps the `rope_from_combined_f32` kernel for applying RoPE to a sub-section
/// of the combined QKV buffer produced by the fused triple QKV GEMV.
#[cfg(feature = "cubecl_runtime")]
pub struct RopeFromCombinedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in fused QKV forward pass wiring
impl RopeFromCombinedCubeCL {
    /// Launch RoPE kernel on a sub-section of a combined QKV buffer.
    ///
    /// **Interleaved only.** Unlike [`RopeCubeCL`] / [`RopeBatchedCubeCL`] this
    /// kernel was not made pairing-generic in the Issue 435 fix, because it has
    /// no live call sites. Anything wiring it into a rotate-half model (Gemma-2
    /// included) must give it the `pair_stride` treatment first — see
    /// [`RopePairing`].
    ///
    /// Reads from `input_combined[section_offset..section_offset+section_len]`,
    /// applies interleaved RoPE, and writes to `output[section_len]`.
    ///
    /// # Safety
    ///
    /// - `input_combined` must have at least `section_offset + section_len` f32 elements
    /// - `cos_sin_handle` must have `head_dim` f32 elements
    /// - `output_handle` must have `section_len` f32 elements
    /// - `head_dim` must be even and > 0
    /// - `section_offset` must be aligned to `head_dim` boundary
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_combined: Handle,
        total_qkv: usize,
        cos_sin_handle: Handle,
        output_handle: Handle,
        section_offset: usize,
        section_len: usize,
        head_dim: usize,
    ) {
        let num_wg = (section_len as u32).div_ceil(256u32).max(1);
        let params: [f32; 3] = [section_offset as f32, section_len as f32, head_dim as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            rope_from_combined_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_combined, total_qkv),
                BufferArg::from_raw_parts(cos_sin_handle, head_dim),
                BufferArg::from_raw_parts(output_handle, section_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// CubeCL GeGLU launcher.
///
/// Wraps the `geglu_f32` kernel for element-wise GeGLU activation.
#[cfg(feature = "cubecl_runtime")]
pub struct GegluCubeCL;

/// CubeCL batched RoPE launcher (Plan 482 T5).
///
/// Applies RoPE to `[seq_len × n_heads × head_dim]` in one dispatch.
#[cfg(feature = "cubecl_runtime")]
pub struct RopeBatchedCubeCL;

/// Pre-compute cos/sin table for ALL positions in a sequence.
///
/// Returns `Vec<f32>` of length `seq_len * head_dim`:
/// For position `pos`, the table is at offset `pos * head_dim`:
/// - `table[pos * head_dim + 2*d]   = cos(pos * freq[d])`
/// - `table[pos * head_dim + 2*d+1] = sin(pos * freq[d])`
///
/// where `freq[d] = 1 / theta^(2*d / head_dim)`.
///
/// `d` is the **pair index**, not a component index, so this table is shared by
/// both [`RopePairing`] conventions — only the kernel's pairing changes.
#[cfg(feature = "cubecl_runtime")]
pub fn precompute_rope_cos_sin_batched(
    seq_len: usize,
    head_dim: usize,
    theta: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut table = vec![0.0f32; seq_len * head_dim];
    for pos in 0..seq_len {
        for d in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            table[pos * head_dim + 2 * d] = angle.cos();
            table[pos * head_dim + 2 * d + 1] = angle.sin();
        }
    }
    table
}

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in Plan 482 batched forward
impl RopeBatchedCubeCL {
    /// Launch batched RoPE kernel.
    ///
    /// Applies position-dependent rotary embedding to all positions in one dispatch.
    ///
    /// Dispatch: `ceil(total / 256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `seq_len * n_heads * head_dim` f32 elements
    /// - `cos_sin_handle`: `seq_len * head_dim` f32 elements (from [`precompute_rope_cos_sin_batched`])
    /// - `output_handle`: `seq_len * n_heads * head_dim` f32 elements
    ///
    /// `pairing` must match the convention the weights were trained under —
    /// see [`RopePairing`].
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        cos_sin_handle: Handle,
        output_handle: Handle,
        total: usize,         // seq_len * n_heads * head_dim
        head_dim: usize,
        n_heads: usize,
        pairing: RopePairing,
    ) {
        let row_stride = (n_heads * head_dim) as f32;
        let params: &[f32] = &[
            head_dim as f32,
            row_stride,
            pairing.stride(head_dim) as f32,
        ];
        let params_handle = client.create_from_slice(f32::as_bytes(params));
        let cos_sin_len = (total / (n_heads * head_dim)) * head_dim; // seq_len * head_dim
        let num_wg = (total as u32).div_ceil(256u32).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            rope_batched_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, total),
                BufferArg::from_raw_parts(cos_sin_handle, cos_sin_len),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(output_handle, total),
            );
        }
    }
}

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in T2.13+ forward pass wiring
impl GegluCubeCL {
    /// Launch GeGLU kernel: `output[i] = GELU(gate[i]) * up[i]`.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `gate_handle`: `n` f32 elements
    /// - `up_handle`: `n` f32 elements
    /// - `output_handle`: `n` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        gate_handle: Handle,
        up_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        let num_wg = (n as u32).div_ceil(256u32).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            geglu_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(gate_handle, n),
                BufferArg::from_raw_parts(up_handle, n),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use crate::cubecl_runtime::ActiveRuntime;

    use crate::cubecl_runtime::CubeCLContext;

    use super::*;

    // ── CPU reference implementations ──

    /// CPU reference RoPE for either pairing convention (Issue 435).
    ///
    /// Under `RotateHalf` this is `riir_infer_core::rope::apply_rope_with_freq`
    /// written out; under `Interleaved` it is the historical GPU convention.
    fn apply_rope_cpu(
        data: &[f32],
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        theta: f32,
        pairing: RopePairing,
    ) -> Vec<f32> {
        let mut result = data.to_vec();
        let half_dim = head_dim / 2;
        let stride = pairing.stride(head_dim);
        for head in 0..n_heads {
            let base = head * head_dim;
            for d in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                // Interleaved: (2d, 2d+1). RotateHalf: (d, d + half_dim).
                let idx0 = match pairing {
                    RopePairing::Interleaved => base + 2 * d,
                    RopePairing::RotateHalf => base + d,
                };
                let idx1 = idx0 + stride;
                let x0 = result[idx0];
                let x1 = result[idx1];
                result[idx0] = x0 * cos_val - x1 * sin_val;
                result[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
        result
    }

    /// CPU reference GELU tanh approximation.
    fn gelu_tanh_cpu(x: f32) -> f32 {
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// CPU reference GeGLU: `GELU(gate) * up`.
    ///
    /// Note: `gelu_tanh_cpu(g)` already includes the `g` factor,
    /// so we do NOT multiply by `g` again.
    fn geglu_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| gelu_tanh_cpu(g) * u)
            .collect()
    }

    // ── RoPE tests ──

    /// Both pairings, so every RoPE test covers the convention Gemma-2 needs
    /// (`RotateHalf`) and the one LLaMA still ships (`Interleaved`).
    const PAIRINGS: [RopePairing; 2] = [RopePairing::Interleaved, RopePairing::RotateHalf];

    /// Run the single-position RoPE kernel and read the result back.
    fn run_rope_gpu(
        client: &ComputeClient<ActiveRuntime>,
        input: &[f32],
        cos_sin: &[f32],
        head_dim: usize,
        pairing: RopePairing,
    ) -> Vec<f32> {
        let n = input.len();
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let cos_sin_handle = client.create_from_slice(f32::as_bytes(cos_sin));
        let output_handle = client.empty(core::mem::size_of_val(input));

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RopeCubeCL::launch::<ActiveRuntime>(
                client,
                input_handle,
                cos_sin_handle,
                output_handle.clone(),
                n,
                head_dim,
                pairing,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        f32::from_bytes(&bytes).to_vec()
    }

    fn assert_close(expected: &[f32], got: &[f32], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
        for (i, (&exp, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (exp - g).abs() < 1e-5,
                "{what} element {i}: expected {exp}, got {g}"
            );
        }
    }

    /// Verify RoPE at pos=0 is identity (cos=1, sin=0 → no rotation).
    #[test]
    fn test_rope_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let head_dim = 16usize;
        let n_heads = 2usize;
        let n = n_heads * head_dim;
        let theta = 10000.0f32;
        let pos = 0usize;

        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let cos_sin = precompute_rope_cos_sin(pos, head_dim, theta);

        for pairing in PAIRINGS {
            let output = run_rope_gpu(&client, &input, &cos_sin, head_dim, pairing);
            // pos=0 is identity under either pairing.
            assert_close(&input, &output, &format!("{pairing:?}"));
            let expected = apply_rope_cpu(&input, pos, head_dim, n_heads, theta, pairing);
            assert_close(&expected, &output, &format!("{pairing:?}"));
        }
    }

    /// Verify RoPE with known values matches CPU reference.
    #[test]
    fn test_rope_known_values() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let head_dim = 4usize;
        let n_heads = 2usize;
        let theta = 10000.0f32;
        let pos = 1usize;

        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cos_sin = precompute_rope_cos_sin(pos, head_dim, theta);

        for pairing in PAIRINGS {
            let output = run_rope_gpu(&client, input, &cos_sin, head_dim, pairing);
            let expected = apply_rope_cpu(input, pos, head_dim, n_heads, theta, pairing);
            assert_close(&expected, &output, &format!("{pairing:?}"));
        }

        // The two conventions must actually differ at pos > 0 — otherwise this
        // test would pass even if `pair_stride` were ignored by the kernel.
        let interleaved = run_rope_gpu(&client, input, &cos_sin, head_dim, RopePairing::Interleaved);
        let rotate_half = run_rope_gpu(&client, input, &cos_sin, head_dim, RopePairing::RotateHalf);
        assert!(
            interleaved
                .iter()
                .zip(&rotate_half)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "Interleaved and RotateHalf produced identical output — pair_stride is being ignored"
        );
    }

    /// The GPU kernel must agree with `riir_infer_core`'s CPU RoPE — the reference
    /// the Gemma-2 training path is validated against (Issue 435).
    #[test]
    fn test_rope_matches_riir_engine_rotate_half() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let head_dim = 256usize;
        let n_heads = 4usize;
        let n = n_heads * head_dim;
        let theta = 10000.0f32;

        for pos in [0usize, 1, 7, 42, 511] {
            let input: Vec<f32> = (0..n).map(|i| ((i + pos) as f32 * 0.013).sin()).collect();
            let cos_sin = precompute_rope_cos_sin(pos, head_dim, theta);

            let got = run_rope_gpu(&client, &input, &cos_sin, head_dim, RopePairing::RotateHalf);

            // riir-engine rotates Q and K together; pass the same buffer as both
            // and keep the Q half.
            let mut q = input.clone();
            let mut k = input.clone();
            let freq = riir_infer_core::rope::RopeFreqTable::new(theta, head_dim);
            riir_infer_core::rope::apply_rope_with_freq(&mut q, &mut k, pos, head_dim, freq.as_slice());

            assert_close(&q, &got, &format!("pos={pos}"));
        }
    }

    /// Verify RoPE with Gemma 2 realistic dimensions (head_dim=256, n_heads=8).
    #[test]
    fn test_rope_gemma2_dims() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let head_dim = 256usize;
        let n_heads = 8usize;
        let n = n_heads * head_dim;
        let theta = 10000.0f32;
        let pos = 42usize;

        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let cos_sin = precompute_rope_cos_sin(pos, head_dim, theta);

        for pairing in PAIRINGS {
            let output = run_rope_gpu(&client, &input, &cos_sin, head_dim, pairing);
            let expected = apply_rope_cpu(&input, pos, head_dim, n_heads, theta, pairing);
            assert_eq!(output.len(), n);
            for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
                assert!(
                    (exp - got).abs() < 1e-4,
                    "{pairing:?} element {i}: expected {exp}, got {got}"
                );
            }
        }
    }

    // ── GeGLU tests ──

    /// Verify GeGLU with known values matches CPU reference.
    #[test]
    fn test_geglu_basic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n = 8usize;
        let gate: &[f32] = &[1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -2.0, 0.1];
        let up: &[f32] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let gate_handle = client.create_from_slice(f32::as_bytes(gate));
        let up_handle = client.create_from_slice(f32::as_bytes(up));
        let output_handle = client.empty(n * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            GegluCubeCL::launch::<ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        let expected = geglu_cpu(gate, up);

        assert_eq!(output.len(), n);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }

    /// Verify GeGLU with zero inputs produces zero output.
    #[test]
    fn test_geglu_zeros() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n = 256usize;
        let gate: Vec<f32> = vec![0.0f32; n];
        let up: Vec<f32> = vec![0.0f32; n];

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));
        let up_handle = client.create_from_slice(f32::as_bytes(&up));
        let output_handle = client.empty(n * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            GegluCubeCL::launch::<ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), n);
        for (i, &got) in output.iter().enumerate() {
            assert!(got.abs() < 1e-6, "element {i}: expected 0.0, got {got}");
        }
    }

    /// Verify GeGLU with non-trivial up projection scales the output.
    #[test]
    fn test_geglu_scaled_up() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n = 16usize;
        let gate: Vec<f32> = vec![1.0f32; n];
        let up: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));
        let up_handle = client.create_from_slice(f32::as_bytes(&up));
        let output_handle = client.empty(n * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            GegluCubeCL::launch::<ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        let expected = geglu_cpu(&gate, &up);

        assert_eq!(output.len(), n);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-4,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }
}
