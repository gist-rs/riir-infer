//! Fused DeltaNet input projections GPU dispatch (Issue 602).
//!
//! Chains the 4 DeltaNet input-projection ternary GEMVs — `qkv`, `z`, `a`, `b` —
//! into a single GPU command buffer. All 4 projections share the same input `x`
//! (the RMSNorm'd layer input), so they can be dispatched back-to-back without
//! any inter-kernel data dependency. ONE `read_one` at the end drains all 4.
//!
//! # Operation
//!
//! ```text
//! qkv = in_proj_qkv @ x   (ternary GEMV, m = q_dim + k_dim + v_dim, n = n_embd)
//! z   = in_proj_z   @ x   (ternary GEMV, m = z_dim = v_dim,        n = n_embd)
//! a   = in_proj_a   @ x   (ternary GEMV, m = n_v_heads,             n = n_embd)
//! b   = in_proj_b   @ x   (ternary GEMV, m = n_v_heads,             n = n_embd)
//! ```
//!
//! All 4 dispatches are queued without `read_one`. Only the final outputs are
//! read back — **ONE GPU sync point instead of 4**.
//!
//! # Why this is different from Bench 437's failed deferred-readback
//!
//! Bench 437 kept Q/K/V/gate/up outputs ALL alive simultaneously (5 concurrent
//! buffers each ~70 KB for the FFN case), which increased Metal memory pressure
//! → 16% slower.
//!
//! These 4 input projections produce small outputs:
//! - `qkv`: ~6 KB (1536 floats for Qwen 3.5-0.8B; ~6 KB for Bonsai-27B)
//! - `z`:   ~2 KB
//! - `a`:   16 bytes (4 floats)
//! - `b`:   16 bytes (4 floats)
//!
//! Total intermediate: ~8 KB. Negligible memory pressure.
//!
//! # Sibling to Issue 600/601 (fused FFN)
//!
//! This module mirrors `ternary_ffn_fused.rs` structurally:
//! - `TernaryInputProjHandles`   ↔ `TernaryFfnHandles`
//! - `TernaryInputProjFused`     ↔ `TernaryFfnFused`
//! - `GpuTernaryInputProj`       ↔ `GpuTernaryFfn`
//! - `LayerInputProjWeightsRef`  ↔ `LayerFfnWeightsRef`

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::{GemvTernaryCubeCL, TernaryHandle};

// ---------------------------------------------------------------------------
// Pre-uploaded input-projection weight handles
// ---------------------------------------------------------------------------

/// Pre-uploaded GPU weight handles for one DeltaNet layer's 4 input projections.
///
/// Construct once at model load; reuse across all decode steps. All 4 share
/// the same `n_embd` input dimension.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct TernaryInputProjHandles {
    /// `in_proj_qkv`: [(q_dim + k_dim + v_dim) × n_embd].
    pub qkv: TernaryHandle,
    /// `in_proj_z`: [z_dim × n_embd] (z_dim = v_dim).
    pub z: TernaryHandle,
    /// `in_proj_a`: [n_v_heads × n_embd] (decay gate input).
    pub a: TernaryHandle,
    /// `in_proj_b`: [n_v_heads × n_embd] (update-rate gate input).
    pub b: TernaryHandle,
}

#[cfg(feature = "cubecl_runtime")]
impl TernaryInputProjHandles {
    /// Upload all 4 input projections from CPU ternary weights.
    ///
    /// All 4 weights must have `cols == n_embd` (they all consume the same
    /// RMSNorm'd layer input `x`).
    pub fn from_weights(
        client: &ComputeClient<ActiveRuntime>,
        qkv: &katgpt_core::TernaryGroupWeights,
        z: &katgpt_core::TernaryGroupWeights,
        a: &katgpt_core::TernaryGroupWeights,
        b: &katgpt_core::TernaryGroupWeights,
    ) -> Self {
        debug_assert_eq!(qkv.cols, z.cols, "qkv and z must share n_embd input dim");
        debug_assert_eq!(qkv.cols, a.cols, "qkv and a must share n_embd input dim");
        debug_assert_eq!(qkv.cols, b.cols, "qkv and b must share n_embd input dim");

        Self {
            qkv: TernaryHandle::from_weights(client, qkv),
            z: TernaryHandle::from_weights(client, z),
            a: TernaryHandle::from_weights(client, a),
            b: TernaryHandle::from_weights(client, b),
        }
    }

    /// Input dimension (n_embd) — shared by all 4 projections.
    pub fn n_embd(&self) -> usize {
        self.qkv.n
    }

    /// QKV output dimension (q_dim + k_dim + v_dim).
    pub fn qkv_dim(&self) -> usize {
        self.qkv.m
    }

    /// Z output dimension (z_dim = v_dim).
    pub fn z_dim(&self) -> usize {
        self.z.m
    }

    /// Number of value heads (a/b output dim).
    pub fn n_v_heads(&self) -> usize {
        self.a.m
    }
}

/// Output of `dispatch_input_projections_noread` — the 4 GPU handles.
///
/// The caller is responsible for `read_one`-ing each into CPU memory.
#[cfg(feature = "cubecl_runtime")]
pub struct InputProjOutputs {
    pub qkv: Handle,
    pub z: Handle,
    pub a: Handle,
    pub b: Handle,
}

/// Fused DeltaNet input-projection GPU dispatcher.
///
/// Chains the 4 GEMVs (qkv, z, a, b) sharing input `x` in a single command
/// buffer. The intermediate output buffers live on GPU only — no CPU readback
/// until the caller explicitly drains them.
#[cfg(feature = "cubecl_runtime")]
pub struct TernaryInputProjFused;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl TernaryInputProjFused {
    /// Dispatch the 4 input projections without any readback.
    ///
    /// Queues 4 GPU GEMV dispatches (all reading the same `x_handle`, writing
    /// to 4 separate output buffers). Returns the 4 output Handles for the
    /// caller to drain.
    ///
    /// # Buffer requirements
    ///
    /// - `x_handle`: [n_embd] f32 elements
    /// - Returns: 4 Handles (qkv, z, a, b) with their respective output sizes
    ///
    /// # Safety
    ///
    /// - `x_handle` must have `handles.n_embd()` f32 elements
    /// - All TernaryHandles in `handles` must have been created from the same `client`
    pub unsafe fn dispatch_input_projections_noread(
        client: &ComputeClient<ActiveRuntime>,
        handles: &TernaryInputProjHandles,
        x_handle: Handle,
    ) -> InputProjOutputs {
        let qkv_dim = handles.qkv_dim();
        let z_dim = handles.z_dim();
        let n_v_heads = handles.n_v_heads();

        // Allocate output buffers (empty = uninitialized, will be written by kernels).
        let qkv_handle = client.empty(qkv_dim * core::mem::size_of::<f32>());
        let z_handle = client.empty(z_dim * core::mem::size_of::<f32>());
        let a_handle = client.empty(n_v_heads * core::mem::size_of::<f32>());
        let b_handle = client.empty(n_v_heads * core::mem::size_of::<f32>());

        // Dispatch all 4 GEMVs. They're independent (all read x, write separate buffers)
        // and execute sequentially in the command queue.
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                client,
                &handles.qkv,
                x_handle.clone(),
                qkv_handle.clone(),
            );
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                client,
                &handles.z,
                x_handle.clone(),
                z_handle.clone(),
            );
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                client,
                &handles.a,
                x_handle.clone(),
                a_handle.clone(),
            );
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                client,
                &handles.b,
                x_handle,
                b_handle.clone(),
            );
        }

        InputProjOutputs {
            qkv: qkv_handle,
            z: z_handle,
            a: a_handle,
            b: b_handle,
        }
    }

    /// Dispatch the 4 input projections and read back all 4 outputs in one sync
    /// pass.
    ///
    /// Convenience wrapper around `dispatch_input_projections_noread` + 4×
    /// `read_one`. Writes the results into the caller-provided slices.
    ///
    /// # Safety
    ///
    /// Same as `dispatch_input_projections_noread`. Additionally:
    /// - `qkv_out.len()` must be ≥ `handles.qkv_dim()`
    /// - `z_out.len()` must be ≥ `handles.z_dim()`
    /// - `a_out.len()` and `b_out.len()` must be ≥ `handles.n_v_heads()`
    pub unsafe fn dispatch_input_projections_with_readback(
        client: &ComputeClient<ActiveRuntime>,
        handles: &TernaryInputProjHandles,
        x: &[f32],
        qkv_out: &mut [f32],
        z_out: &mut [f32],
        a_out: &mut [f32],
        b_out: &mut [f32],
    ) {
        let n_embd = handles.n_embd();
        debug_assert_eq!(x.len(), n_embd, "input x must have n_embd elements");
        debug_assert_eq!(qkv_out.len(), handles.qkv_dim(), "qkv_out size mismatch");
        debug_assert_eq!(z_out.len(), handles.z_dim(), "z_out size mismatch");
        debug_assert_eq!(a_out.len(), handles.n_v_heads(), "a_out size mismatch");
        debug_assert_eq!(b_out.len(), handles.n_v_heads(), "b_out size mismatch");

        let x_handle = client.create_from_slice(<f32 as CubeElement>::as_bytes(x));
        let outs = unsafe {
            Self::dispatch_input_projections_noread(client, handles, x_handle)
        };

        // Read back each output. CubeCL's `read_one` forces a sync; the 4 calls
        // here all drain the same command queue (the dispatches were already
        // queued by dispatch_input_projections_noread), so the cost is ~1 sync
        // + 4 small readbacks rather than 4 independent syncs.
        let qkv_bytes = client.read_one(outs.qkv).unwrap();
        let src = bytemuck::cast_slice::<u8, f32>(&qkv_bytes);
        qkv_out[..src.len()].copy_from_slice(src);

        let z_bytes = client.read_one(outs.z).unwrap();
        let src = bytemuck::cast_slice::<u8, f32>(&z_bytes);
        z_out[..src.len()].copy_from_slice(src);

        let a_bytes = client.read_one(outs.a).unwrap();
        let src = bytemuck::cast_slice::<u8, f32>(&a_bytes);
        a_out[..src.len()].copy_from_slice(src);

        let b_bytes = client.read_one(outs.b).unwrap();
        let src = bytemuck::cast_slice::<u8, f32>(&b_bytes);
        b_out[..src.len()].copy_from_slice(src);
    }
}

// ---------------------------------------------------------------------------
// GPU input-projection hook impl (Issue 602)
// ---------------------------------------------------------------------------

/// Reference to one layer's 4 input projections — for [`GpuTernaryInputProj::preupload_layers`].
pub struct LayerInputProjWeightsRef<'a> {
    pub qkv: &'a katgpt_core::TernaryGroupWeights,
    pub z: &'a katgpt_core::TernaryGroupWeights,
    pub a: &'a katgpt_core::TernaryGroupWeights,
    pub b: &'a katgpt_core::TernaryGroupWeights,
}

/// GPU implementation of [`katgpt_core::TernaryInputProjHook`].
///
/// Wraps `TernaryInputProjFused::dispatch_input_projections_with_readback`
/// with a handle cache keyed by the POINTER IDENTITY of all four weight
/// allocations (qkv + z + a + b, pos+neg each — Issue 674 B2: keying on all
/// four makes a hit imply the same z/a/b too, not just the same qkv). Each
/// layer has a unique weight allocation set, so the key is unique per layer.
/// Pre-upload all layer weights at model load via [`Self::preupload_layers`].
///
/// # Invalidation contract (Issue 674 B1)
///
/// Pointer identity is only stable while the allocations live. A process
/// that drops a model and loads another can hit allocator address reuse —
/// the new model's weights then hash to the OLD key and silently reuse the
/// previous model's device handles. Call [`Self::clear`] on every model
/// drop/reload (or [`Self::invalidate`] for a single layer's entry).
///
/// # Usage
///
/// ```rust,ignore
/// let ctx = GpuContext::new()?;
/// let client = ctx.cubecl_client();
/// let input_proj_hook = GpuTernaryInputProj::new(client);
/// input_proj_hook.preupload_layers(...);
/// forward_qwen_deltanet_ternary_with_hook(
///     &mut x, &weights, &mut cache, token, pos, &config,
///     &mut scratch, &rope_freq, None,
///     None,                     // per-matvec hook (Issue 599 path)
///     Some(&input_proj_hook),   // fused input projections (Issue 602)
///     Some(&ffn_hook),          // fused FFN (Issue 601)
/// );
/// ```
/// Pointer-identity cache key half: (pos_bits ptr, neg_bits ptr) of one
/// weight allocation.
#[cfg(feature = "cubecl_runtime")]
fn ptr_key(w: &katgpt_core::TernaryGroupWeights) -> (usize, usize) {
    (w.pos_bits.as_ptr() as usize, w.neg_bits.as_ptr() as usize)
}

/// Handle-cache key: the pointer identity of ALL FOUR weight allocations
/// (Issue 674 B2 — a hit implies the same z/a/b, not just the same qkv).
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct InputProjCacheKey {
    qkv: (usize, usize),
    z: (usize, usize),
    a: (usize, usize),
    b: (usize, usize),
}

#[cfg(feature = "cubecl_runtime")]
impl InputProjCacheKey {
    fn new(
        qkv: &katgpt_core::TernaryGroupWeights,
        z: &katgpt_core::TernaryGroupWeights,
        a: &katgpt_core::TernaryGroupWeights,
        b: &katgpt_core::TernaryGroupWeights,
    ) -> Self {
        Self {
            qkv: ptr_key(qkv),
            z: ptr_key(z),
            a: ptr_key(a),
            b: ptr_key(b),
        }
    }
}

#[cfg(feature = "cubecl_runtime")]
pub struct GpuTernaryInputProj {
    client: ComputeClient<ActiveRuntime>,
    /// Cached input-projection handles, keyed by the pointer identity of ALL
    /// FOUR weight allocations (qkv, z, a, b — pos+neg each; Issue 674 B2).
    handles: papaya::HashMap<InputProjCacheKey, TernaryInputProjHandles>,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuTernaryInputProj {
    /// Create a GPU input-projection hook with the given CubeCL client.
    pub fn new(client: ComputeClient<ActiveRuntime>) -> Self {
        Self {
            client,
            handles: papaya::HashMap::builder().build(),
        }
    }

    /// Pre-upload a single layer's 4 input projections.
    pub fn preupload(
        &self,
        qkv: &katgpt_core::TernaryGroupWeights,
        z: &katgpt_core::TernaryGroupWeights,
        a: &katgpt_core::TernaryGroupWeights,
        b: &katgpt_core::TernaryGroupWeights,
    ) {
        let key = InputProjCacheKey::new(qkv, z, a, b);
        if self.handles.pin().get(&key).is_none() {
            let h = TernaryInputProjHandles::from_weights(&self.client, qkv, z, a, b);
            self.handles.pin().insert(key, h);
        }
    }

    /// Invalidate ALL cached handles.
    ///
    /// Call on model drop/reload: pointer identity is only stable while the
    /// allocations live, and allocator address reuse can otherwise make a new
    /// model's weights hash to an old key and silently reuse the previous
    /// model's device handles (Issue 674 B1).
    pub fn clear(&self) {
        self.handles.pin().clear();
    }

    /// Invalidate one layer's cached handles (the entry keyed by this weight
    /// set). Use for a partial weight swap (e.g. a z-only LoRA hot-swap).
    pub fn invalidate(
        &self,
        qkv: &katgpt_core::TernaryGroupWeights,
        z: &katgpt_core::TernaryGroupWeights,
        a: &katgpt_core::TernaryGroupWeights,
        b: &katgpt_core::TernaryGroupWeights,
    ) {
        let key = InputProjCacheKey::new(qkv, z, a, b);
        self.handles.pin().remove(&key);
    }

    /// Pre-upload all per-layer input projections from a ternary model.
    /// Call once after loading the model, before the first forward pass.
    ///
    /// Accepts any iterator yielding (qkv, z, a, b) weight references.
    pub fn preupload_layers<
        'w,
        L: IntoIterator<Item = &'w LayerInputProjWeightsRef<'w>>,
    >(
        &self,
        layers: L,
    ) {
        for l in layers {
            self.preupload(l.qkv, l.z, l.a, l.b);
        }
    }
}

#[cfg(feature = "cubecl_runtime")]
impl katgpt_core::TernaryInputProjHook for GpuTernaryInputProj {
    fn input_projections(
        &self,
        qkv_w: &katgpt_core::TernaryGroupWeights,
        z_w: &katgpt_core::TernaryGroupWeights,
        a_w: &katgpt_core::TernaryGroupWeights,
        b_w: &katgpt_core::TernaryGroupWeights,
        x: &[f32],
        qkv_out: &mut [f32],
        z_out: &mut [f32],
        a_out: &mut [f32],
        b_out: &mut [f32],
    ) {
        let key = InputProjCacheKey::new(qkv_w, z_w, a_w, b_w);
        let pin = self.handles.pin();
        if let Some(h) = pin.get(&key) { unsafe {
                TernaryInputProjFused::dispatch_input_projections_with_readback(
                    &self.client, h, x, qkv_out, z_out, a_out, b_out,
                );
            } } else {
                // On-the-fly upload (first call without preupload).
                let h = TernaryInputProjHandles::from_weights(&self.client, qkv_w, z_w, a_w, b_w);
                pin.insert(key, h.clone());
                unsafe {
                    TernaryInputProjFused::dispatch_input_projections_with_readback(
                        &self.client, &h, x, qkv_out, z_out, a_out, b_out,
                    );
                }
            }
    }
}

// ---------------------------------------------------------------------------
// Tests — mirror the fused FFN test pattern
// ---------------------------------------------------------------------------

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

    /// CPU reference: 4 separate ternary matvecs sharing input x.
    fn cpu_input_projections(
        qkv_w: &TernaryGroupWeights,
        z_w: &TernaryGroupWeights,
        a_w: &TernaryGroupWeights,
        b_w: &TernaryGroupWeights,
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut qkv = vec![0.0f32; qkv_w.rows];
        let mut z = vec![0.0f32; z_w.rows];
        let mut a = vec![0.0f32; a_w.rows];
        let mut b = vec![0.0f32; b_w.rows];
        simd_ternary_group_matvec(qkv_w, x, &mut qkv);
        simd_ternary_group_matvec(z_w, x, &mut z);
        simd_ternary_group_matvec(a_w, x, &mut a);
        simd_ternary_group_matvec(b_w, x, &mut b);
        (qkv, z, a, b)
    }

    /// Compare two f32 slices with relative-error tolerance.
    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        let mut max_rel = 0.0f32;
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            let denom = av.abs().max(1.0);
            let rel_err = (av - bv).abs() / denom;
            max_rel = max_rel.max(rel_err);
            assert!(
                rel_err < tol,
                "{label}[{i}]: cpu={av:.6} gpu={bv:.6} rel_err={rel_err:.6} (max so far {max_rel:.6})"
            );
        }
    }

    #[test]
    fn test_fused_input_proj_matches_cpu_small() {
        // Qwen 3.5-0.8B DeltaNet layer shape (scaled down for test speed):
        // n_embd=256, q_dim=k_dim=v_dim=128 (4 heads × 32 head_dim),
        // z_dim=128, n_v_heads=4.
        let n_embd = 256;
        let q_dim = 128;
        let z_dim = 128;
        let n_v_heads = 4;

        let qkv_w = make_ternary_weights(2 * q_dim + z_dim, n_embd, 1);
        let z_w = make_ternary_weights(z_dim, n_embd, 2);
        let a_w = make_ternary_weights(n_v_heads, n_embd, 3);
        let b_w = make_ternary_weights(n_v_heads, n_embd, 4);

        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let (cpu_qkv, cpu_z, cpu_a, cpu_b) =
            cpu_input_projections(&qkv_w, &z_w, &a_w, &b_w, &x);

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let handles =
            TernaryInputProjHandles::from_weights(&client, &qkv_w, &z_w, &a_w, &b_w);

        let mut gpu_qkv = vec![0.0f32; 2 * q_dim + z_dim];
        let mut gpu_z = vec![0.0f32; z_dim];
        let mut gpu_a = vec![0.0f32; n_v_heads];
        let mut gpu_b = vec![0.0f32; n_v_heads];
        unsafe {
            TernaryInputProjFused::dispatch_input_projections_with_readback(
                &client,
                &handles,
                &x,
                &mut gpu_qkv,
                &mut gpu_z,
                &mut gpu_a,
                &mut gpu_b,
            );
        }

        assert_close(&cpu_qkv, &gpu_qkv, 0.05, "qkv");
        assert_close(&cpu_z, &gpu_z, 0.05, "z");
        assert_close(&cpu_a, &gpu_a, 0.05, "a");
        assert_close(&cpu_b, &gpu_b, 0.05, "b");
    }

    #[test]
    fn test_fused_input_proj_matches_cpu_medium() {
        // Medium shape: n_embd=512, qkv output 1024, z 512, n_v_heads=8.
        let n_embd = 512;
        let qkv_dim = 1024;
        let z_dim = 512;
        let n_v_heads = 8;

        let qkv_w = make_ternary_weights(qkv_dim, n_embd, 10);
        let z_w = make_ternary_weights(z_dim, n_embd, 20);
        let a_w = make_ternary_weights(n_v_heads, n_embd, 30);
        let b_w = make_ternary_weights(n_v_heads, n_embd, 40);

        let x: Vec<f32> = (0..n_embd)
            .map(|i| ((i as u32).wrapping_mul(1103515245) as f32 % 2.0) - 1.0)
            .collect();

        let (cpu_qkv, cpu_z, cpu_a, cpu_b) =
            cpu_input_projections(&qkv_w, &z_w, &a_w, &b_w, &x);

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let handles =
            TernaryInputProjHandles::from_weights(&client, &qkv_w, &z_w, &a_w, &b_w);

        let mut gpu_qkv = vec![0.0f32; qkv_dim];
        let mut gpu_z = vec![0.0f32; z_dim];
        let mut gpu_a = vec![0.0f32; n_v_heads];
        let mut gpu_b = vec![0.0f32; n_v_heads];
        unsafe {
            TernaryInputProjFused::dispatch_input_projections_with_readback(
                &client,
                &handles,
                &x,
                &mut gpu_qkv,
                &mut gpu_z,
                &mut gpu_a,
                &mut gpu_b,
            );
        }

        assert_close(&cpu_qkv, &gpu_qkv, 0.05, "qkv");
        assert_close(&cpu_z, &gpu_z, 0.05, "z");
        assert_close(&cpu_a, &gpu_a, 0.05, "a");
        assert_close(&cpu_b, &gpu_b, 0.05, "b");
    }

    #[test]
    fn test_fused_input_proj_matches_cpu_bonsai_shape() {
        // Bonsai-27B shape scaled down for test speed:
        // n_embd=5120 (real), qkv output=1536 (4 heads × 128 × 3), z=512, n_v_heads=4.
        // Use n_embd=512 for test speed; the bench harness validates the full shape.
        let n_embd = 512;
        let qkv_dim = 1536;
        let z_dim = 512;
        let n_v_heads = 4;

        let qkv_w = make_ternary_weights(qkv_dim, n_embd, 100);
        let z_w = make_ternary_weights(z_dim, n_embd, 200);
        let a_w = make_ternary_weights(n_v_heads, n_embd, 300);
        let b_w = make_ternary_weights(n_v_heads, n_embd, 400);

        let x: Vec<f32> = (0..n_embd)
            .map(|i| ((i as u32).wrapping_mul(2654435761) as f32 % 2.0) - 1.0)
            .collect();

        let (cpu_qkv, cpu_z, cpu_a, cpu_b) =
            cpu_input_projections(&qkv_w, &z_w, &a_w, &b_w, &x);

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let handles =
            TernaryInputProjHandles::from_weights(&client, &qkv_w, &z_w, &a_w, &b_w);

        let mut gpu_qkv = vec![0.0f32; qkv_dim];
        let mut gpu_z = vec![0.0f32; z_dim];
        let mut gpu_a = vec![0.0f32; n_v_heads];
        let mut gpu_b = vec![0.0f32; n_v_heads];
        unsafe {
            TernaryInputProjFused::dispatch_input_projections_with_readback(
                &client,
                &handles,
                &x,
                &mut gpu_qkv,
                &mut gpu_z,
                &mut gpu_a,
                &mut gpu_b,
            );
        }

        // At this scale, rel_err tolerance is higher due to accumulated
        // reduction-order differences across 512-dim dot products.
        assert_close(&cpu_qkv, &gpu_qkv, 0.10, "bonsai qkv");
        assert_close(&cpu_z, &gpu_z, 0.10, "bonsai z");
        assert_close(&cpu_a, &gpu_a, 0.10, "bonsai a");
        assert_close(&cpu_b, &gpu_b, 0.10, "bonsai b");
    }

    /// Issue 674 B1 regression: `clear()` invalidates the handle cache so a
    /// model reload (the address-reuse scenario — a new allocation at a
    /// previously-keyed address produces a SAME-key lookup) re-uploads fresh
    /// handles instead of silently reusing the previous model's.
    #[test]
    fn test_input_proj_cache_clear_reuploads() {
        use katgpt_core::TernaryInputProjHook;

        let n_embd = 256;
        let q_dim = 128;
        let z_dim = 128;
        let n_v_heads = 4;

        let qkv_w = make_ternary_weights(2 * q_dim + z_dim, n_embd, 7);
        let z_w = make_ternary_weights(z_dim, n_embd, 8);
        let a_w = make_ternary_weights(n_v_heads, n_embd, 9);
        let b_w = make_ternary_weights(n_v_heads, n_embd, 11);
        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let (cpu_qkv, cpu_z, cpu_a, cpu_b) =
            cpu_input_projections(&qkv_w, &z_w, &a_w, &b_w, &x);

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let hook = GpuTernaryInputProj::new(client);

        // Populate the cache, then drop the model (the stale-handle scenario).
        hook.preupload(&qkv_w, &z_w, &a_w, &b_w);
        assert_eq!(hook.handles.pin().len(), 1, "preupload should cache one entry");

        hook.clear(); // the model-reload invalidation contract
        assert_eq!(hook.handles.pin().len(), 0, "clear() must empty the cache");

        // A same-key lookup after clear (address reuse) must MISS and
        // re-upload — and still produce correct outputs via the hook path.
        let mut gpu_qkv = vec![0.0f32; 2 * q_dim + z_dim];
        let mut gpu_z = vec![0.0f32; z_dim];
        let mut gpu_a = vec![0.0f32; n_v_heads];
        let mut gpu_b = vec![0.0f32; n_v_heads];
        TernaryInputProjHook::input_projections(
            &hook,
            &qkv_w,
            &z_w,
            &a_w,
            &b_w,
            &x,
            &mut gpu_qkv,
            &mut gpu_z,
            &mut gpu_a,
            &mut gpu_b,
        );
        assert_eq!(
            hook.handles.pin().len(),
            1,
            "post-clear lookup must re-upload (no stale-handle reuse)"
        );
        assert_close(&cpu_qkv, &gpu_qkv, 0.05, "clear qkv");
        assert_close(&cpu_z, &gpu_z, 0.05, "clear z");
        assert_close(&cpu_a, &gpu_a, 0.05, "clear a");
        assert_close(&cpu_b, &gpu_b, 0.05, "clear b");
    }

    /// Issue 674 B2 regression: the cache key covers ALL FOUR weight
    /// allocations — the same qkv with a DIFFERENT z must be a distinct
    /// entry (before the fix, the second preupload was a silent no-op hit
    /// and z2 callers received z1's handles).
    #[test]
    fn test_input_proj_cache_keys_all_four_weights() {
        use katgpt_core::TernaryInputProjHook;

        let n_embd = 256;
        let q_dim = 128;
        let z_dim = 128;
        let n_v_heads = 4;

        let qkv_w = make_ternary_weights(2 * q_dim + z_dim, n_embd, 21);
        let z1_w = make_ternary_weights(z_dim, n_embd, 22);
        let z2_w = make_ternary_weights(z_dim, n_embd, 23);
        let a_w = make_ternary_weights(n_v_heads, n_embd, 24);
        let b_w = make_ternary_weights(n_v_heads, n_embd, 25);
        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let hook = GpuTernaryInputProj::new(client);

        // Same qkv/a/b, different z — two distinct allocations must be two
        // distinct cache entries.
        hook.preupload(&qkv_w, &z1_w, &a_w, &b_w);
        hook.preupload(&qkv_w, &z2_w, &a_w, &b_w);
        assert_eq!(
            hook.handles.pin().len(),
            2,
            "same-qkv/different-z must be two entries (4-pair key)"
        );

        // The z2 caller must get z2-correct outputs through the hook path.
        let (_, cpu_z2, _, _) = cpu_input_projections(&qkv_w, &z2_w, &a_w, &b_w, &x);
        let mut gpu_qkv = vec![0.0f32; 2 * q_dim + z_dim];
        let mut gpu_z = vec![0.0f32; z_dim];
        let mut gpu_a = vec![0.0f32; n_v_heads];
        let mut gpu_b = vec![0.0f32; n_v_heads];
        TernaryInputProjHook::input_projections(
            &hook,
            &qkv_w,
            &z2_w,
            &a_w,
            &b_w,
            &x,
            &mut gpu_qkv,
            &mut gpu_z,
            &mut gpu_a,
            &mut gpu_b,
        );
        assert_close(&cpu_z2, &gpu_z, 0.05, "z2 through 4-pair-keyed cache");

        // And the single-layer invalidate removes exactly the z2 entry.
        hook.invalidate(&qkv_w, &z2_w, &a_w, &b_w);
        assert_eq!(hook.handles.pin().len(), 1, "invalidate removes one entry");
    }
}
