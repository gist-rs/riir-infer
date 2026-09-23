//! Issue 666 — GPU Q+V LoRA decode kernel (cudarc / CUDA path).
//!
//! Applies a rank-`r` Q+V LoRA correction to the compact QKV buffer of one
//! DeltaNet layer during decode. The math mirrors
//! [`riir_infer_core::deltanet::QvLora::forward_qv_delta`] exactly:
//!
//! ```text
//! ax_q[r]   = A_q[r × n_embd] @ norm_x[n_embd]
//! ax_v[r]   = A_v[r × n_embd] @ norm_x[n_embd]
//! delta_q[q_dim] = scale · B_q[q_dim × r] @ ax_q[r]
//! delta_v[v_dim] = scale · B_v[v_dim × r] @ ax_v[r]
//! qkv[0..q_dim]              += delta_q
//! qkv[q_dim+k_dim..]         += delta_v
//! ```
//!
//! Total ~115K FLOPs at Bonsai-27B dims (rank 8, n_embd 5120, q_dim=v_dim=2048)
//! — negligible vs the ~10^11 FLOP backbone forward. Unblocks serving the
//! trained Arm B LoRA adapter ([Plan 334] / [Issue 641] T6 checkpoint) at full
//! GPU decode speed (~66 tok/s) instead of falling back to CPU decode (1.76 tok/s).
//!
//! [Plan 334]: ../../.plans/334_single_layer_lora_accuracy_parity_bonsai_metal.md
//! [Issue 641]: ../../.issues/641_gpu_lora_forward_backward_substrate_gap.md

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

use super::CudarcKernelError;

// ─────────────────────────────────────────────────────────────────────────
// CUDA source
// ─────────────────────────────────────────────────────────────────────────

const LORA_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// Phase 1 — down-projection: ax = A @ norm_x
// One block of `rank` threads; each thread computes one ax[r] via a
// dot-product reduction over n_embd. rank is small (8 for Arm B), so a
// single warp with strided accumulation is sufficient.
// ---------------------------------------------------------------------------
extern "C" __global__ void lora_down_project_f32(
    const float* __restrict__ a_mat,    // [rank × n_embd] row-major
    const float* __restrict__ norm_x,   // [n_embd]
    float* __restrict__ ax_out,         // [rank]
    const int rank,
    const int n_embd)
{
    const int r = threadIdx.x;
    if (r >= rank) return;

    const float* row = a_mat + (r * n_embd);
    float sum = 0.0f;
    for (int j = 0; j < n_embd; j++) {
        sum += row[j] * norm_x[j];
    }
    ax_out[r] = sum;
}

// ---------------------------------------------------------------------------
// Phase 2 — up-projection + add: qkv_slice[i] += scale · B[i,:] @ ax
// One block of `out_dim` threads; each thread computes one output element.
// `k_offset` is the byte offset into qkv where the slice starts (in f32 units).
// ---------------------------------------------------------------------------

// Apply the Q delta: qkv[0..q_dim] += scale · B_q @ ax_q
extern "C" __global__ void lora_up_project_add_f32(
    const float* __restrict__ b_mat,    // [out_dim × rank] row-major
    const float* __restrict__ ax,       // [rank]
    float* __restrict__ qkv,            // [conv_dim] — Q or V slice starts at `slice_off`
    const int out_dim,
    const int rank,
    const int slice_off,                // offset into qkv (f32 units)
    const float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;

    const float* row = b_mat + (i * rank);
    float sum = 0.0f;
    for (int r = 0; r < rank; r++) {
        sum += row[r] * ax[r];
    }
    qkv[slice_off + i] += scale * sum;
}
"#;

// ─────────────────────────────────────────────────────────────────────────
// Compiled kernel module
// ─────────────────────────────────────────────────────────────────────────

/// Compiled CUDA functions for the Q+V LoRA decode path.
pub struct LoraDecodeKernels {
    down_project: CudaFunction,
    up_project_add: CudaFunction,
    _module: Arc<CudaModule>,
}

impl LoraDecodeKernels {
    /// Compile the LoRA kernels via nvrtc (sm_89 for Ada / RTX 4090).
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, CudarcKernelError> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            LORA_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| CudarcKernelError::Compile(e.to_string()))?;

        let down_project = module
            .load_function("lora_down_project_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let up_project_add = module
            .load_function("lora_up_project_add_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;

        Ok(Self {
            down_project,
            up_project_add,
            _module: module,
        })
    }

    /// Apply the Q+V LoRA correction to `qkv` in-place.
    ///
    /// Reads `norm_x[n_embd]`, computes the LoRA delta via two-phase matvec,
    /// and adds the delta to the Q slice `qkv[0..q_dim]` and the V slice
    /// `qkv[q_dim+k_dim .. q_dim+k_dim+v_dim]`.
    ///
    /// Uses `ax_q` / `ax_v` as scratch (rank-sized, pre-allocated).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_qv_apply(
        &self,
        stream: &CudaStream,
        norm_x: &CudaSlice<f32>,
        qkv: &mut CudaSlice<f32>,
        ax_q: &mut CudaSlice<f32>,
        ax_v: &mut CudaSlice<f32>,
        a_q: &CudaSlice<f32>,
        b_q: &CudaSlice<f32>,
        a_v: &CudaSlice<f32>,
        b_v: &CudaSlice<f32>,
        n_embd: usize,
        q_dim: usize,
        k_dim: usize,
        v_dim: usize,
        rank: usize,
        scale: f32,
    ) -> Result<(), CudarcKernelError> {
        let rank_i32 = rank as i32;
        let n_embd_i32 = n_embd as i32;
        let q_dim_i32 = q_dim as i32;
        let v_dim_i32 = v_dim as i32;
        let block_rank = rank.max(1) as u32;

        // Phase 1a: ax_q = A_q @ norm_x
        unsafe {
            stream
                .launch_builder(&self.down_project)
                .arg(a_q)
                .arg(norm_x)
                .arg(&mut *ax_q)
                .arg(&rank_i32)
                .arg(&n_embd_i32)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_rank, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }

        // Phase 1b: ax_v = A_v @ norm_x
        unsafe {
            stream
                .launch_builder(&self.down_project)
                .arg(a_v)
                .arg(norm_x)
                .arg(&mut *ax_v)
                .arg(&rank_i32)
                .arg(&n_embd_i32)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_rank, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }

        // Phase 2a: qkv[0..q_dim] += scale · B_q @ ax_q
        let grid_q = (q_dim as u32).div_ceil(256).max(1);
        let zero_off: i32 = 0;
        unsafe {
            stream
                .launch_builder(&self.up_project_add)
                .arg(b_q)
                .arg(&*ax_q)
                .arg(&mut *qkv)
                .arg(&q_dim_i32)
                .arg(&rank_i32)
                .arg(&zero_off)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (grid_q, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }

        // Phase 2b: qkv[q_dim+k_dim .. ] += scale · B_v @ ax_v
        let v_off_i32: i32 = (q_dim + k_dim) as i32;
        let grid_v = (v_dim as u32).div_ceil(256).max(1);
        unsafe {
            stream
                .launch_builder(&self.up_project_add)
                .arg(b_v)
                .arg(&*ax_v)
                .arg(qkv)
                .arg(&v_dim_i32)
                .arg(&rank_i32)
                .arg(&v_off_i32)
                .arg(&scale)
                .launch(LaunchConfig {
                    grid_dim: (grid_v, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// QvLoraGpuCudarc — uploaded weights + scratch (one target layer)
// ─────────────────────────────────────────────────────────────────────────

/// GPU-resident Q+V LoRA adapter for one DeltaNet layer (decode path).
///
/// Holds the 4 LoRA weight matrices uploaded once + rank-sized scratch buffers
/// reused per token. Construct via [`QvLoraGpuCudarc::from_qv_lora`] from a
/// CPU [`QvLora`](riir_infer_core::deltanet::QvLora) (e.g. a loaded checkpoint).
///
/// All buffers are pre-allocated — zero per-token allocation after construction
/// (GOAT G4).
pub struct QvLoraGpuCudarc {
    pub(crate) a_q: CudaSlice<f32>,
    pub(crate) b_q: CudaSlice<f32>,
    pub(crate) a_v: CudaSlice<f32>,
    pub(crate) b_v: CudaSlice<f32>,
    pub(crate) scale: f32,
    pub(crate) rank: usize,
    pub(crate) n_embd: usize,
    pub(crate) q_dim: usize,
    pub(crate) v_dim: usize,
}

impl QvLoraGpuCudarc {
    /// Upload a CPU [`QvLora`](riir_infer_core::deltanet::QvLora) to the GPU.
    ///
    /// Copies all 4 weight matrices + allocates rank-sized scratch. Called
    /// once at adapter-load time (e.g. from a `.bin` checkpoint); the buffers
    /// persist for the lifetime of the forward pass.
    pub fn from_qv_lora(
        stream: &Arc<CudaStream>,
        lora: &riir_infer_core::deltanet::qv_lora::QvLora,
    ) -> Result<Self, CudarcKernelError> {
        let upload_err = |e: cudarc::driver::DriverError| CudarcKernelError::Launch(e.to_string());
        let a_q = stream.clone_htod(&lora.a_q).map_err(upload_err)?;
        let b_q = stream.clone_htod(&lora.b_q).map_err(upload_err)?;
        let a_v = stream.clone_htod(&lora.a_v).map_err(upload_err)?;
        let b_v = stream.clone_htod(&lora.b_v).map_err(upload_err)?;

        Ok(Self {
            a_q,
            b_q,
            a_v,
            b_v,
            scale: lora.scale(),
            rank: lora.rank,
            n_embd: lora.n_embd,
            q_dim: lora.q_dim,
            v_dim: lora.v_dim,
        })
    }

    /// Issue 504 T1 (Plan 370 confirm) — in-place weight refresh for the
    /// CLOSED-LOOP TRAINING loop: the optimizer mutates the CPU adapter every
    /// step, and the training forward must see the CURRENT weights. Copies
    /// the 4 matrices into the existing device buffers (no realloc, no graph
    /// invalidation — the buffer addresses are unchanged) and refreshes the
    /// host-side scale.
    ///
    /// # Errors
    ///
    /// Returns [`CudarcKernelError::InvalidArg`] when the adapter's shapes
    /// don't match what this slot was built with (a rank/dim change needs a
    /// fresh slot — the rank scratch is sized at attach time; detach + attach
    /// instead of updating).
    pub fn update_from(
        &mut self,
        stream: &Arc<CudaStream>,
        lora: &riir_infer_core::deltanet::qv_lora::QvLora,
    ) -> Result<(), CudarcKernelError> {
        if lora.rank != self.rank
            || lora.n_embd != self.n_embd
            || lora.q_dim != self.q_dim
            || lora.v_dim != self.v_dim
        {
            return Err(CudarcKernelError::InvalidArg(format!(
                "update_from: adapter shape mismatch (rank {}/{} n_embd {}/{} q_dim {}/{} v_dim {}/{}); \
                 re-attach instead of updating when the shape changes",
                lora.rank, self.rank, lora.n_embd, self.n_embd,
                lora.q_dim, self.q_dim, lora.v_dim, self.v_dim,
            )));
        }
        let upload_err = |e: cudarc::driver::DriverError| CudarcKernelError::Launch(e.to_string());
        stream.memcpy_htod(&lora.a_q, &mut self.a_q).map_err(upload_err)?;
        stream.memcpy_htod(&lora.b_q, &mut self.b_q).map_err(upload_err)?;
        stream.memcpy_htod(&lora.a_v, &mut self.a_v).map_err(upload_err)?;
        stream.memcpy_htod(&lora.b_v, &mut self.b_v).map_err(upload_err)?;
        self.scale = lora.scale();
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Tests — cross-validate against the CPU reference (forward_qv_delta)
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn get_ctx() -> Option<Arc<CudaContext>> {
        CudaContext::new(0).ok()
    }

    /// G1 — the GPU LoRA kernel must match the CPU `QvLora::forward_qv_delta`
    /// reference bit-closely (within f32 matvec tolerance).
    ///
    /// Builds a tiny QvLora (rank 4, n_embd 32, q_dim 16, v_dim 16), runs both
    /// paths on random `norm_x`, and compares the resulting Q+V slices of qkv.
    #[test]
    fn test_lora_gpu_matches_cpu_reference() {
        let Some(ctx) = get_ctx() else {
            eprintln!("skipping — no CUDA device");
            return;
        };
        let stream: Arc<CudaStream> = ctx.new_stream().unwrap();

        // Small dims for a fast test.
        let rank = 4;
        let n_embd = 32;
        let q_dim = 16;
        let v_dim = 16;
        let k_dim = q_dim; // DeltaNet: k_dim == q_dim
        let conv_dim = q_dim + k_dim + v_dim;
        let alpha = 8.0;

        let mut rng = riir_infer_core::types::Rng::new(42);
        let lora = riir_infer_core::deltanet::qv_lora::QvLora::new(rank, n_embd, q_dim, v_dim, alpha, &mut rng);

        // Random norm_x + initial qkv.
        let norm_x: Vec<f32> = (0..n_embd).map(|_| rng.normal() * 0.1).collect();
        let qkv_init: Vec<f32> = (0..conv_dim).map(|_| rng.normal() * 0.5).collect();

        // ── CPU reference ──
        let mut qkv_cpu = qkv_init.clone();
        let mut delta_q = vec![0.0f32; q_dim];
        let mut delta_v = vec![0.0f32; v_dim];
        let mut ax_q = vec![0.0f32; rank];
        let mut ax_v = vec![0.0f32; rank];
        lora.forward_qv_delta(&norm_x, &mut delta_q, &mut delta_v, &mut ax_q, &mut ax_v);
        for i in 0..q_dim {
            qkv_cpu[i] += delta_q[i];
        }
        let v_off = q_dim + k_dim;
        for i in 0..v_dim {
            qkv_cpu[v_off + i] += delta_v[i];
        }

        // ── GPU path ──
        let kernels = LoraDecodeKernels::new(ctx).expect("compile kernels");
        let gpu_lora = QvLoraGpuCudarc::from_qv_lora(&stream, &lora).expect("upload lora");
        let norm_x_gpu = stream.clone_htod(&norm_x).unwrap();
        let mut qkv_gpu = stream.clone_htod(&qkv_init).unwrap();
        // Scratch buffers (allocated once, reused per token in production).
        let mut ax_q_gpu = stream.alloc_zeros::<f32>(rank).unwrap();
        let mut ax_v_gpu = stream.alloc_zeros::<f32>(rank).unwrap();

        kernels
            .launch_qv_apply(
                &stream,
                &norm_x_gpu,
                &mut qkv_gpu,
                &mut ax_q_gpu,
                &mut ax_v_gpu,
                &gpu_lora.a_q,
                &gpu_lora.b_q,
                &gpu_lora.a_v,
                &gpu_lora.b_v,
                n_embd,
                q_dim,
                k_dim,
                v_dim,
                rank,
                gpu_lora.scale,
            )
            .expect("launch lora");

        stream.synchronize().unwrap();
        let mut qkv_result = vec![0.0f32; conv_dim];
        stream.memcpy_dtoh(&qkv_gpu, &mut qkv_result).unwrap();

        // ── Compare (B=0 init → delta=0 → should match exactly) ──
        let mut max_diff = 0.0f32;
        for i in 0..conv_dim {
            let d = (qkv_cpu[i] - qkv_result[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("[lora-gpu-test] max_diff with B=0 init: {max_diff} (expected ~0)");

        // Now test with non-zero B to exercise the kernel.
        let mut lora_nz = lora.clone();
        for i in 0..lora_nz.b_q.len() {
            lora_nz.b_q[i] = rng.normal() * 0.3;
        }
        for i in 0..lora_nz.b_v.len() {
            lora_nz.b_v[i] = rng.normal() * 0.3;
        }

        // Re-run CPU reference with non-zero B.
        let mut qkv_cpu2 = qkv_init.clone();
        let mut delta_q2 = vec![0.0f32; q_dim];
        let mut delta_v2 = vec![0.0f32; v_dim];
        let mut ax_q2 = vec![0.0f32; rank];
        let mut ax_v2 = vec![0.0f32; rank];
        lora_nz.forward_qv_delta(
            &norm_x,
            &mut delta_q2,
            &mut delta_v2,
            &mut ax_q2,
            &mut ax_v2,
        );
        for i in 0..q_dim {
            qkv_cpu2[i] += delta_q2[i];
        }
        for i in 0..v_dim {
            qkv_cpu2[v_off + i] += delta_v2[i];
        }

        // Re-run GPU with non-zero B.
        let gpu_lora_nz = QvLoraGpuCudarc::from_qv_lora(&stream, &lora_nz).expect("upload lora nz");
        let mut qkv_gpu2 = stream.clone_htod(&qkv_init).unwrap();
        kernels
            .launch_qv_apply(
                &stream,
                &norm_x_gpu,
                &mut qkv_gpu2,
                &mut ax_q_gpu,
                &mut ax_v_gpu,
                &gpu_lora_nz.a_q,
                &gpu_lora_nz.b_q,
                &gpu_lora_nz.a_v,
                &gpu_lora_nz.b_v,
                n_embd,
                q_dim,
                k_dim,
                v_dim,
                rank,
                gpu_lora_nz.scale,
            )
            .expect("launch lora nz");

        stream.synchronize().unwrap();
        let mut qkv_result2 = vec![0.0f32; conv_dim];
        stream.memcpy_dtoh(&qkv_gpu2, &mut qkv_result2).unwrap();

        let mut max_diff2 = 0.0f32;
        for i in 0..conv_dim {
            let d = (qkv_cpu2[i] - qkv_result2[i]).abs();
            if d > max_diff2 {
                max_diff2 = d;
            }
        }
        eprintln!("[lora-gpu-test] max_diff with non-zero B: {max_diff2}");
        assert!(
            max_diff2 < 1e-4,
            "GPU LoRA kernel diverges from CPU reference: max_diff = {max_diff2}"
        );
    }
}
