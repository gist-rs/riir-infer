//! Fused ternary FFN GPU dispatch (Issue 600).
//!
//! Chains the 3 FFN ternary GEMVs + SwiGLU activation into a single GPU
//! command buffer, eliminating the per-matvec CPU↔GPU sync that made the
//! naive Issue 599 integration slower than CPU parallel.
//!
//! # Operation
//!
//! ```text
//! gate = ffn_gate @ x    (ternary GEMV, m=mlp_dim, n=n_embd)
//! up   = ffn_up   @ x    (ternary GEMV, m=mlp_dim, n=n_embd)
//! inter = silu(gate) * up  (SwiGLU element-wise)
//! out   = ffn_down @ inter (ternary GEMV, m=n_embd, n=mlp_dim)
//! ```
//!
//! All 4 dispatches are queued without `read_one`. Only the final output is
//! read back — ONE GPU sync point instead of 4.
//!
//! # Why this is different from Bench 437's failed deferred-readback
//!
//! Bench 437 kept Q/K/V/gate/up outputs ALL alive simultaneously (5 concurrent
//! buffers), which increased Metal memory pressure → 16% slower.
//!
//! This fused dispatch has a **linear dependency chain**: gate_buf + up_buf are
//! consumed by SwiGLU → inter_buf → consumed by down GEMV. At most 2 intermediate
//! buffers are live at any point. The GPU command scheduler can reuse buffer
//! memory between dependent dispatches.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::{GemvTernaryCubeCL, TernaryHandle};

// ---------------------------------------------------------------------------
// SwiGLU element-wise kernel (local — avoids deltanet_inference dep)
// ---------------------------------------------------------------------------

/// SwiGLU element-wise kernel: `output[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
///
/// This is the same math as `deltanet_cubecl::deltanet_gating_f32` but kept local
/// to avoid pulling in the `deltanet_inference` feature (which depends on
/// riir-engine). The fused FFN dispatch only needs cubecl_runtime + ternary_gemv.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn swiglu_elementwise(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    let n = params[0usize] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= n {
        terminate!();
    }

    let g = gate[idx];
    // SiLU: g * sigmoid(g) = g / (1 + exp(-g))
    let neg_g = f32::new(0.0f32) - g;
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
    output[idx] = g * sig * up[idx];
}

// ---------------------------------------------------------------------------
// Pre-uploaded FFN weight handles
// ---------------------------------------------------------------------------

/// Pre-uploaded GPU weight handles for one layer's FFN.
///
/// Construct once at model load; reuse across all decode steps.
/// The three projections share the same `n_embd` input dimension (for gate/up)
/// or `mlp_dim` (for down).
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct TernaryFfnHandles {
    /// `ffn_gate` projection: [mlp_dim × n_embd].
    pub gate: TernaryHandle,
    /// `ffn_up` projection: [mlp_dim × n_embd].
    pub up: TernaryHandle,
    /// `ffn_down` projection: [n_embd × mlp_dim].
    pub down: TernaryHandle,
}

#[cfg(feature = "cubecl_runtime")]
impl TernaryFfnHandles {
    /// Upload all 3 FFN projections from CPU ternary weights.
    ///
    /// `gate` and `up` must have the same shape (both [mlp_dim × n_embd]).
    /// `down` must be [n_embd × mlp_dim].
    pub fn from_weights(
        client: &ComputeClient<ActiveRuntime>,
        gate: &katgpt_core::TernaryGroupWeights,
        up: &katgpt_core::TernaryGroupWeights,
        down: &katgpt_core::TernaryGroupWeights,
    ) -> Self {
        debug_assert_eq!(gate.rows, up.rows, "gate and up must have same output dim");
        debug_assert_eq!(gate.cols, up.cols, "gate and up must have same input dim");
        debug_assert_eq!(down.cols, gate.rows, "down input dim must match gate/up output dim");
        debug_assert_eq!(down.rows, gate.cols, "down output dim must match gate/up input dim");

        Self {
            gate: TernaryHandle::from_weights(client, gate),
            up: TernaryHandle::from_weights(client, up),
            down: TernaryHandle::from_weights(client, down),
        }
    }

    /// Input dimension (n_embd) — shared by gate + up.
    pub fn n_embd(&self) -> usize {
        self.gate.n
    }

    /// MLP intermediate dimension.
    pub fn mlp_dim(&self) -> usize {
        self.gate.m
    }
}

/// Fused ternary FFN GPU dispatcher.
///
/// Chains gate GEMV + up GEMV + SwiGLU + down GEMV in a single command buffer.
/// The intermediate buffers (gate, up, inter) live on GPU only — no CPU readback
/// until the final output.
#[cfg(feature = "cubecl_runtime")]
pub struct TernaryFfnFused;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl TernaryFfnFused {
    /// Dispatch the fused FFN: `out = ffn_down @ swilu(ffn_gate @ x) * (ffn_up @ x)`.
    ///
    /// Queues 4 GPU dispatches (gate GEMV, up GEMV, SwiGLU, down GEMV) without
    /// any `read_one` call. Returns the output Handle for the caller to read
    /// back (potentially batching with other readbacks).
    ///
    /// # Buffer requirements
    ///
    /// - `x_handle`: [n_embd] f32 elements
    /// - Returns: Handle pointing to [n_embd] f32 elements (the FFN output)
    ///
    /// # Safety
    ///
    /// - `x_handle` must have `handles.n_embd()` f32 elements
    /// - All TernaryHandles in `handles` must have been created from the same `client`
    pub unsafe fn dispatch_ffn_noread(
        client: &ComputeClient<ActiveRuntime>,
        handles: &TernaryFfnHandles,
        x_handle: Handle,
    ) -> Handle {
        let mlp_dim = handles.mlp_dim();
        let n_embd = handles.n_embd();

        // Allocate intermediate buffers (empty = uninitialized, will be written by kernels).
        let gate_handle = client.empty(mlp_dim * core::mem::size_of::<f32>());
        let up_handle = client.empty(mlp_dim * core::mem::size_of::<f32>());
        let inter_handle = client.empty(mlp_dim * core::mem::size_of::<f32>());
        let out_handle = client.empty(n_embd * core::mem::size_of::<f32>());

        // Dispatch 1: gate = ffn_gate @ x
        // Dispatch 2: up = ffn_up @ x
        // These are independent (both read x) but will execute sequentially in the
        // command queue. Both write to separate buffers.
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                client,
                &handles.gate,
                x_handle.clone(),
                gate_handle.clone(),
            );
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                client,
                &handles.up,
                x_handle,
                up_handle.clone(),
            );
        }

        // Dispatch 3: inter = silu(gate) * up
        // Inline SwiGLU: output[i] = gate[i] * sigmoid(gate[i]) * up[i]
        let swiglu_params: [f32; 1] = [mlp_dim as f32];
        let swiglu_params_handle = client.create_from_slice(<f32 as CubeElement>::as_bytes(&swiglu_params));
        let swiglu_wg = 128u32;
        let swiglu_num_wg = (mlp_dim as u32).div_ceil(swiglu_wg);
        unsafe {
            swiglu_elementwise::launch_unchecked::<ActiveRuntime>(
                client,
                CubeCount::Static(swiglu_num_wg, 1, 1),
                CubeDim::new_1d(swiglu_wg),
                BufferArg::from_raw_parts(gate_handle, mlp_dim),
                BufferArg::from_raw_parts(up_handle, mlp_dim),
                BufferArg::from_raw_parts(inter_handle.clone(), mlp_dim),
                BufferArg::from_raw_parts(swiglu_params_handle, 1),
            );
        }

        // Dispatch 4: out = ffn_down @ inter
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                client,
                &handles.down,
                inter_handle,
                out_handle.clone(),
            );
        }

        out_handle
    }

    /// Dispatch the fused FFN and read back the output in one sync.
    ///
    /// Convenience wrapper around `dispatch_ffn_noread` + `read_one`.
    /// Writes the result into `output` (must have `n_embd` elements).
    ///
    /// # Safety
    ///
    /// Same as `dispatch_ffn_noread`. Additionally, `output.len()` must be ≥ `n_embd`.
    pub unsafe fn dispatch_ffn_with_readback(
        client: &ComputeClient<ActiveRuntime>,
        handles: &TernaryFfnHandles,
        x: &[f32],
        output: &mut [f32],
    ) {
        let n_embd = handles.n_embd();
        debug_assert_eq!(x.len(), n_embd, "input x must have n_embd elements");
        debug_assert_eq!(output.len(), n_embd, "output must have n_embd elements");

        let x_handle = client.create_from_slice(<f32 as CubeElement>::as_bytes(x));
        let out_handle = unsafe { Self::dispatch_ffn_noread(client, handles, x_handle) };

        let bytes = client.read_one(out_handle).unwrap();
        let src = bytemuck::cast_slice::<u8, f32>(&bytes);
        output[..src.len()].copy_from_slice(src);
    }
}

// ---------------------------------------------------------------------------
// GPU FFN hook impl (Issue 601)
// ---------------------------------------------------------------------------

/// Reference to one layer's 3 FFN projections — for [`GpuTernaryFfn::preupload_layers`].
pub struct LayerFfnWeightsRef<'a> {
    pub gate: &'a katgpt_core::TernaryGroupWeights,
    pub up: &'a katgpt_core::TernaryGroupWeights,
    pub down: &'a katgpt_core::TernaryGroupWeights,
}

/// GPU implementation of [`katgpt_core::TernaryFfnHook`].
///
/// Wraps `TernaryFfnFused::dispatch_ffn_with_readback` with a handle cache keyed
/// by the gate weight's pointer identity (each layer has a unique gate allocation).
/// Pre-upload all layer FFN weights at model load via [`Self::preupload_model`].
///
/// # Usage
///
/// ```rust,ignore
/// let ctx = GpuContext::new()?;
/// let client = ctx.cubecl_client();
/// let ffn_hook = GpuTernaryFfn::new(client);
/// ffn_hook.preupload_model(&weights);
/// forward_qwen_deltanet_ternary_with_hook(
///     &mut x, &weights, &mut cache, token, pos, &config,
///     &mut scratch, &rope_freq, None,
///     None,                 // per-matvec hook (Issue 599 path)
///     None,                 // fused input projections (Issue 602)
///     Some(&ffn_hook),      // fused FFN (Issue 601)
/// );
/// ```
/// Pointer-identity cache key half: (pos_bits ptr, neg_bits ptr) of one
/// weight allocation.
#[cfg(feature = "cubecl_runtime")]
fn ptr_key(w: &katgpt_core::TernaryGroupWeights) -> (usize, usize) {
    (w.pos_bits.as_ptr() as usize, w.neg_bits.as_ptr() as usize)
}

/// Handle-cache key: the pointer identity of ALL THREE weight allocations
/// (Issue 674 B2 class — a hit implies the same up/down, not just gate).
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FfnCacheKey {
    gate: (usize, usize),
    up: (usize, usize),
    down: (usize, usize),
}

#[cfg(feature = "cubecl_runtime")]
impl FfnCacheKey {
    fn new(
        gate: &katgpt_core::TernaryGroupWeights,
        up: &katgpt_core::TernaryGroupWeights,
        down: &katgpt_core::TernaryGroupWeights,
    ) -> Self {
        Self {
            gate: ptr_key(gate),
            up: ptr_key(up),
            down: ptr_key(down),
        }
    }
}

#[cfg(feature = "cubecl_runtime")]
pub struct GpuTernaryFfn {
    client: ComputeClient<ActiveRuntime>,
    /// Cached FFN handles, keyed by the pointer identity of ALL THREE weight
    /// allocations (gate, up, down — pos+neg each; Issue 674 B2 class).
    handles: papaya::HashMap<FfnCacheKey, TernaryFfnHandles>,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuTernaryFfn {
    /// Create a GPU FFN hook with the given CubeCL client.
    pub fn new(client: ComputeClient<ActiveRuntime>) -> Self {
        Self {
            client,
            handles: papaya::HashMap::builder().build(),
        }
    }

    /// Pre-upload a single layer's FFN weights.
    pub fn preupload(
        &self,
        gate: &katgpt_core::TernaryGroupWeights,
        up: &katgpt_core::TernaryGroupWeights,
        down: &katgpt_core::TernaryGroupWeights,
    ) {
        let key = FfnCacheKey::new(gate, up, down);
        if self.handles.pin().get(&key).is_none() {
            let h = TernaryFfnHandles::from_weights(&self.client, gate, up, down);
            self.handles.pin().insert(key, h);
        }
    }

    /// Invalidate ALL cached handles.
    ///
    /// Call on model drop/reload: pointer identity is only stable while the
    /// allocations live, and allocator address reuse can otherwise make a new
    /// model's weights hash to an old key and silently reuse the previous
    /// model's device handles (Issue 674 B1 class).
    pub fn clear(&self) {
        self.handles.pin().clear();
    }

    /// Invalidate one layer's cached handles (the entry keyed by this weight
    /// set). Use for a partial weight swap.
    pub fn invalidate(
        &self,
        gate: &katgpt_core::TernaryGroupWeights,
        up: &katgpt_core::TernaryGroupWeights,
        down: &katgpt_core::TernaryGroupWeights,
    ) {
        let key = FfnCacheKey::new(gate, up, down);
        self.handles.pin().remove(&key);
    }

    /// Pre-upload all per-layer FFN weights from a ternary model.
    /// Call once after loading the model, before the first forward pass.
    ///
    /// Accepts any iterator yielding (gate, up, down) weight references — works
    /// with `QwenDeltaNetTernaryWeights.layers` or any other ternary model.
    pub fn preupload_layers<
        'w,
        L: IntoIterator<Item = &'w LayerFfnWeightsRef<'w>>,
    >(
        &self,
        layers: L,
    ) {
        for l in layers {
            self.preupload(l.gate, l.up, l.down);
        }
    }
}

#[cfg(feature = "cubecl_runtime")]
impl katgpt_core::TernaryFfnHook for GpuTernaryFfn {
    fn ffn(
        &self,
        gate_w: &katgpt_core::TernaryGroupWeights,
        up_w: &katgpt_core::TernaryGroupWeights,
        down_w: &katgpt_core::TernaryGroupWeights,
        x: &[f32],
        out: &mut [f32],
    ) {
        let key = FfnCacheKey::new(gate_w, up_w, down_w);
        let pin = self.handles.pin();
        let handle = pin.get(&key);
        if let Some(h) = handle { unsafe {
                TernaryFfnFused::dispatch_ffn_with_readback(&self.client, h, x, out);
            } } else {
                // On-the-fly upload (first call without preupload).
                let h = TernaryFfnHandles::from_weights(&self.client, gate_w, up_w, down_w);
                pin.insert(key, h.clone());
                unsafe {
                    TernaryFfnFused::dispatch_ffn_with_readback(&self.client, &h, x, out);
                }
            }
    }
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::context::GpuContext;
    use katgpt_core::{TernaryGroupWeights, simd_ternary_group_matvec};

    /// Build random ternary weights for testing via quantize_from_f32.
    fn make_ternary_weights(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut state = hasher.finish();

        let dense: Vec<f32> = (0..rows * cols)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let bits = state as i32;
                (bits as f32) / (i32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        TernaryGroupWeights::quantize_from_f32(&dense, rows, cols)
    }

    /// CPU reference FFN: gate → SwiGLU(gate, up) → down.
    fn cpu_ffn(
        gate_w: &TernaryGroupWeights,
        up_w: &TernaryGroupWeights,
        down_w: &TernaryGroupWeights,
        x: &[f32],
    ) -> Vec<f32> {
        let mlp_dim = gate_w.rows;
        let n_embd = gate_w.cols;

        let mut gate_out = vec![0.0f32; mlp_dim];
        let mut up_out = vec![0.0f32; mlp_dim];
        let mut inter = vec![0.0f32; mlp_dim];

        simd_ternary_group_matvec(gate_w, &x[..n_embd], &mut gate_out);
        simd_ternary_group_matvec(up_w, &x[..n_embd], &mut up_out);

        // SwiGLU: inter = silu(gate) * up = gate * sigmoid(gate) * up
        for i in 0..mlp_dim {
            let g = gate_out[i];
            let sig = 1.0 / (1.0 + (-g).exp());
            inter[i] = g * sig * up_out[i];
        }

        let mut out = vec![0.0f32; n_embd];
        simd_ternary_group_matvec(down_w, &inter, &mut out);
        out
    }

    #[test]
    fn test_fused_ffn_matches_cpu_small() {
        let n_embd = 256;
        let mlp_dim = 512;

        let gate_w = make_ternary_weights(mlp_dim, n_embd, 1);
        let up_w = make_ternary_weights(mlp_dim, n_embd, 2);
        let down_w = make_ternary_weights(n_embd, mlp_dim, 3);

        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let cpu_out = cpu_ffn(&gate_w, &up_w, &down_w, &x);

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let handles = TernaryFfnHandles::from_weights(&client, &gate_w, &up_w, &down_w);

        let mut gpu_out = vec![0.0f32; n_embd];
        unsafe {
            TernaryFfnFused::dispatch_ffn_with_readback(&client, &handles, &x, &mut gpu_out);
        }

        // Compare: tolerance accounts for ternary GEMV reduction-order differences + SwiGLU
        for (i, (cpu, gpu)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let denom = cpu.abs().max(1.0);
            let rel_err = (cpu - gpu).abs() / denom;
            assert!(
                rel_err < 0.05,
                "FFN mismatch at {i}: cpu={cpu:.6} gpu={gpu:.6} rel_err={rel_err:.6}"
            );
        }
    }

    #[test]
    fn test_fused_ffn_matches_cpu_medium() {
        let n_embd = 512;
        let mlp_dim = 1024;

        let gate_w = make_ternary_weights(mlp_dim, n_embd, 10);
        let up_w = make_ternary_weights(mlp_dim, n_embd, 20);
        let down_w = make_ternary_weights(n_embd, mlp_dim, 30);

        let x: Vec<f32> = (0..n_embd)
            .map(|i| ((i as u32).wrapping_mul(1103515245) as f32 % 2.0) - 1.0)
            .collect();

        let cpu_out = cpu_ffn(&gate_w, &up_w, &down_w, &x);

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let handles = TernaryFfnHandles::from_weights(&client, &gate_w, &up_w, &down_w);

        let mut gpu_out = vec![0.0f32; n_embd];
        unsafe {
            TernaryFfnFused::dispatch_ffn_with_readback(&client, &handles, &x, &mut gpu_out);
        }

        for (i, (cpu, gpu)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let denom = cpu.abs().max(1.0);
            let rel_err = (cpu - gpu).abs() / denom;
            assert!(
                rel_err < 0.05,
                "FFN mismatch at {i}: cpu={cpu:.6} gpu={gpu:.6} rel_err={rel_err:.6}"
            );
        }
    }

    #[test]
    fn test_fused_ffn_matches_cpu_bonsai_shape() {
        // Bonsai-27B FFN shape scaled down for test speed: n_embd=512, mlp_dim=2048.
        // Full 5120×17408 is validated in the bench harness (tests/bench_600_*).
        let n_embd = 512;
        let mlp_dim = 2048;

        let gate_w = make_ternary_weights(mlp_dim, n_embd, 100);
        let up_w = make_ternary_weights(mlp_dim, n_embd, 200);
        let down_w = make_ternary_weights(n_embd, mlp_dim, 300);

        let x: Vec<f32> = (0..n_embd)
            .map(|i| ((i as u32).wrapping_mul(2654435761) as f32 % 2.0) - 1.0)
            .collect();

        let cpu_out = cpu_ffn(&gate_w, &up_w, &down_w, &x);

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let handles = TernaryFfnHandles::from_weights(&client, &gate_w, &up_w, &down_w);

        let mut gpu_out = vec![0.0f32; n_embd];
        unsafe {
            TernaryFfnFused::dispatch_ffn_with_readback(&client, &handles, &x, &mut gpu_out);
        }

        // At this scale, rel_err tolerance is higher due to accumulated reduction-order
        // differences across 17408-dim dot products.
        let mut max_rel = 0.0f32;
        for (cpu, gpu) in cpu_out.iter().zip(gpu_out.iter()) {
            let denom = cpu.abs().max(1.0);
            let rel_err = (cpu - gpu).abs() / denom;
            max_rel = max_rel.max(rel_err);
        }
        eprintln!("Bonsai FFN max rel_err: {max_rel:.6}");
        assert!(max_rel < 0.1, "Bonsai FFN rel_err too high: {max_rel:.6}");
    }

    /// Issue 674 B1/B2 class regression: the cache key covers ALL THREE
    /// weight allocations (same gate, different up = distinct entry) and
    /// `clear()` invalidates the cache so a model reload re-uploads instead
    /// of reusing stale handles via allocator address reuse.
    #[test]
    fn test_ffn_cache_keys_all_three_weights_and_clear() {
        use katgpt_core::TernaryFfnHook;

        let n_embd = 256;
        let mlp_dim = 512;

        let gate_w = make_ternary_weights(mlp_dim, n_embd, 41);
        let up1_w = make_ternary_weights(mlp_dim, n_embd, 42);
        let up2_w = make_ternary_weights(mlp_dim, n_embd, 43);
        let down_w = make_ternary_weights(n_embd, mlp_dim, 44);
        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let hook = GpuTernaryFfn::new(client);

        // B2 class: same gate/down, different up — two distinct allocations
        // must be two distinct cache entries (before the fix, the second
        // preupload was a silent no-op hit and up2 callers got up1 handles).
        hook.preupload(&gate_w, &up1_w, &down_w);
        hook.preupload(&gate_w, &up2_w, &down_w);
        assert_eq!(
            hook.handles.pin().len(),
            2,
            "same-gate/different-up must be two entries (3-pair key)"
        );

        // The up2 caller gets up2-correct output through the hook path.
        let cpu_up2 = cpu_ffn(&gate_w, &up2_w, &down_w, &x);
        let mut gpu_out = vec![0.0f32; n_embd];
        TernaryFfnHook::ffn(&hook, &gate_w, &up2_w, &down_w, &x, &mut gpu_out);
        let mut max_rel = 0.0f32;
        for (cpu, gpu) in cpu_up2.iter().zip(gpu_out.iter()) {
            let denom = cpu.abs().max(1.0);
            max_rel = max_rel.max((cpu - gpu).abs() / denom);
        }
        assert!(max_rel < 0.05, "up2 through 3-pair-keyed cache rel_err: {max_rel:.6}");

        // B1 class: clear empties the cache; a same-key lookup re-uploads.
        hook.clear();
        assert_eq!(hook.handles.pin().len(), 0, "clear() must empty the cache");
        let mut gpu_out2 = vec![0.0f32; n_embd];
        TernaryFfnHook::ffn(&hook, &gate_w, &up1_w, &down_w, &x, &mut gpu_out2);
        assert_eq!(
            hook.handles.pin().len(),
            1,
            "post-clear lookup must re-upload (no stale-handle reuse)"
        );
    }
}
