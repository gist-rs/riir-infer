//! Issue 654 promotion e2e — the BMTH (Bonsai MTP Head) CUDA scorer
//! (cudarc + nvrtc).
//!
//! The trained draft head is a support × d f32 table over raw residual rows
//! (`riir-infer-dist::spec_mtp::BmthHead`); the CPU fold is a sequential
//! ascending-c dot per slot (~6.2 ms single-thread at the Bonsai-2 chat
//! shape, support 2077 × d 7168). This module places the SAME scoring on
//! the GPU: one block per support row (block-strided dot + shared reduce +
//! bias), then the Issue-697 CPU-exact first-index argmax over the scores.
//! Per score: one H2D row upload, two kernel launches, one 8-byte readback
//! — sub-millisecond against a decode-step budget, which is the 654 gate's
//! GPU-placement precondition ("a synchronous CPU head makes the e2e
//! measure PLACEMENT, not the drafter").
//!
//! ## Numerics disclosure (deliberate, local-only)
//!
//! The GPU block reduce folds in a different order than the CPU sequential
//! ascending-c fold, so a near-tie argmax may differ from the CPU scorer.
//! That is safe by construction: the head only PROPOSES drafts — acceptance
//! is adjudicated model-side on exact verify rows (the spec lane's lossless
//! G1 is drafter-agnostic), so a different draft changes efficiency, never
//! correctness. The head score crosses no sync/quorum boundary — the
//! Issue-940 determinism law's LOCAL classification, recorded here.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};

use super::ElementwiseKernels;

const BMTH_CUDA_SRC: &str = r#"
extern "C" __global__ void bmth_scores_f32(
    const float* __restrict__ w,       // [support * d] row-major
    const float* __restrict__ x,       // [d]
    const float* __restrict__ bias,    // [support]
    const int d,
    float* __restrict__ scores)        // [support]
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const float* wr = w + (size_t)row * (size_t)d;
    float acc = 0.0f;
    for (int c = tid; c < d; c += stride) {
        acc += wr[c] * x[c];
    }
    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[row] = bias[row] + sm[0];
}
"#;

/// Device-resident BMTH scorer: weights uploaded once, per-`score` traffic
/// is one 4·d-byte H2D upload + one 8-byte D2H readback. Zero steady-state
/// host allocation (the staging slices are built at construction).
pub struct BmthHeadCuda {
    stream: Arc<CudaStream>,
    w: cudarc::driver::safe::CudaSlice<f32>,
    bias: cudarc::driver::safe::CudaSlice<f32>,
    x: cudarc::driver::safe::CudaSlice<f32>,
    scores: cudarc::driver::safe::CudaSlice<f32>,
    argmax_res: cudarc::driver::safe::CudaSlice<u64>,
    scores_kernel: CudaFunction,
    /// The Issue-697 CPU-exact first-index argmax, reused (compiled once
    /// here, never re-owned — one nvrtc module for the whole scorer).
    ew: ElementwiseKernels,
    d: usize,
    support: usize,
    _module: Arc<CudaModule>,
}

const ZERO_U64: [u64; 1] = [0];

impl BmthHeadCuda {
    /// Build on CUDA device `device_idx`: compile the scores kernel, upload
    /// the head table. `w` is support × d row-major (the BMTH artifact's
    /// raw-feature layout — the bake already folded standardization in).
    pub fn new_device(
        device_idx: usize,
        w: &[f32],
        bias: &[f32],
        d: usize,
        support: usize,
    ) -> Result<Self, String> {
        if d == 0 || support == 0 {
            return Err(format!(
                "bmth cuda: degenerate shape d {d} support {support}"
            ));
        }
        if w.len() != support * d {
            return Err(format!(
                "bmth cuda: w len {} != support {support} × d {d}",
                w.len()
            ));
        }
        if bias.len() != support {
            return Err(format!(
                "bmth cuda: bias len {} != support {support}",
                bias.len()
            ));
        }
        let ctx = CudaContext::new(device_idx)
            .map_err(|e| format!("bmth cuda: device {device_idx} init failed: {e}"))?;
        let stream: Arc<CudaStream> = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            BMTH_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("bmth cuda: nvrtc compile failed: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("bmth cuda: module load failed: {e}"))?;
        let scores_kernel = module
            .load_function("bmth_scores_f32")
            .map_err(|e| format!("bmth cuda: kernel load failed: {e}"))?;
        let ew = ElementwiseKernels::new(ctx.clone())
            .map_err(|e| format!("bmth cuda: elementwise kernels: {e}"))?;

        let mut w_dev = stream.alloc_zeros::<f32>(support * d).map_err(device_err)?;
        stream.memcpy_htod(w, &mut w_dev).map_err(device_err)?;
        let mut bias_dev = stream.alloc_zeros::<f32>(support).map_err(device_err)?;
        stream
            .memcpy_htod(bias, &mut bias_dev)
            .map_err(device_err)?;
        let x_dev = stream.alloc_zeros::<f32>(d).map_err(device_err)?;
        let scores_dev = stream.alloc_zeros::<f32>(support).map_err(device_err)?;
        let argmax_res = stream.alloc_zeros::<u64>(1).map_err(device_err)?;
        Ok(Self {
            stream,
            w: w_dev,
            bias: bias_dev,
            x: x_dev,
            scores: scores_dev,
            argmax_res,
            scores_kernel,
            ew,
            d,
            support,
            _module: module,
        })
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn support(&self) -> usize {
        self.support
    }

    /// Score one raw residual row: per-slot `bias + Σ w·x` on device, then
    /// the CPU-exact first-index argmax. Returns the winning SLOT (the
    /// caller maps it through the head's support-id table). Synchronizes
    /// before the readback — the drafter's roll chain needs the answer now.
    pub fn score(&mut self, x: &[f32]) -> Result<usize, String> {
        if x.len() != self.d {
            return Err(format!("bmth cuda: row width {} != d {}", x.len(), self.d));
        }
        self.stream
            .memcpy_htod(ZERO_U64.as_slice(), &mut self.argmax_res)
            .map_err(device_err)?;
        self.stream
            .memcpy_htod(x, &mut self.x)
            .map_err(device_err)?;
        let d_i32 = self.d as i32;
        let cfg = LaunchConfig {
            grid_dim: (self.support as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * core::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.scores_kernel)
                .arg(&self.w)
                .arg(&self.x)
                .arg(&self.bias)
                .arg(&d_i32)
                .arg(&mut self.scores)
                .launch(cfg)
                .map_err(|e| format!("bmth cuda: scores launch failed: {e}"))?;
        }
        self.ew
            .launch_argmax_first(&self.stream, &self.scores, self.support, &self.argmax_res)
            .map_err(|e| format!("bmth cuda: argmax launch failed: {e}"))?;
        self.stream.synchronize().map_err(device_err)?;
        let packed = self
            .stream
            .clone_dtoh(&self.argmax_res)
            .map_err(device_err)?[0];
        Ok(!(packed as u32) as usize)
    }
}

fn device_err(e: cudarc::driver::DriverError) -> String {
    format!("bmth cuda: device error: {e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in [-1, 1) — no external RNG dep.
    fn lcg_f32(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*state >> 40) as u32; // 24 bits of mantissa
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    /// CPU sequential ascending-c fold — the reference semantics.
    fn cpu_argmax(w: &[f32], x: &[f32], bias: &[f32], d: usize, support: usize) -> usize {
        let mut best = f32::NEG_INFINITY;
        let mut best_i = 0usize;
        for v in 0..support {
            let wr = &w[v * d..(v + 1) * d];
            let mut s = bias[v];
            for c in 0..d {
                s += wr[c] * x[c];
            }
            if s > best {
                best = s;
                best_i = v;
            }
        }
        best_i
    }

    #[test]
    fn gpu_argmax_matches_cpu_on_separated_margins() {
        let Ok(ctx) = CudaContext::new(0) else {
            // No CUDA device on this box — the 4090 executes this test
            // (the crate's own cudarc tests skip the same way).
            return;
        };
        let _ = &ctx;
        let (d, support) = (96usize, 37usize);
        let mut st = 0x654u64;
        let w: Vec<f32> = (0..support * d).map(|_| lcg_f32(&mut st)).collect();
        let bias: Vec<f32> = (0..support).map(|_| lcg_f32(&mut st)).collect();
        let mut dev = BmthHeadCuda::new_device(0, &w, &bias, d, support).expect("bmth cuda build");
        // Several rows — the fold-order delta is ~1e-5; random draws
        // separate the top-2 by far more on every one of these.
        for _ in 0..8 {
            let x: Vec<f32> = (0..d).map(|_| lcg_f32(&mut st)).collect();
            let got = dev.score(&x).expect("gpu score");
            let want = cpu_argmax(&w, &x, &bias, d, support);
            assert_eq!(got, want, "gpu argmax slot != cpu argmax slot");
        }
    }

    #[test]
    fn wrong_width_row_refuses() {
        let Ok(_ctx) = CudaContext::new(0) else {
            return;
        };
        let (d, support) = (32usize, 8usize);
        let w = vec![0.5f32; support * d];
        let bias = vec![0.0f32; support];
        let mut dev = BmthHeadCuda::new_device(0, &w, &bias, d, support).expect("build");
        let err = dev.score(&vec![0.0f32; d + 1]).expect_err("must refuse");
        assert!(err.contains("row width"), "refusal names the width: {err}");
    }
}
