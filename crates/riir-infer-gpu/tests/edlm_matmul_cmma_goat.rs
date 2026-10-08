//! The eDLM f16-B GEMM GOAT lane (Issue 1005 T8 follow-up): the scalar tiled
//! kernel vs the cooperative-matrix (tensor-core) kernel.
//!
//! Two halves:
//! - **always-on** — real-shape-class parity for BOTH kernels vs the CPU
//!   reference (the tiny unit tests in `matmul_f16b_cmma_cubecl` cover the
//!   small-shape correctness; this pins the large-N accumulation depth and
//!   the multi-row bounds arms at the lane's real dimensions).
//! - **`#[ignore]` GOAT** — the interleaved A/B timing table (the
//!   sequential-A/B law: arms alternate in pairs, median per arm — never two
//!   sequential runs). Run:
//!   `cargo test --release -p riir-infer-gpu --features edlm_gpu
//!    --test edlm_matmul_cmma_goat -- --ignored --nocapture`
//!
//! Box state is part of every number: quote the `PROVENANCE:` line + the
//! host's GPU-exclusivity state (this lane prints the runtime name; take
//! `nvidia-smi` beside the run).

#![cfg(feature = "edlm_gpu")]
#![allow(clippy::too_many_arguments)]

use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext, create_f32, read_f32};
use riir_infer_gpu::{Handle, MatmulF16bCmmaCubeCL, MatmulF16bCubeCL};
use cubecl::Runtime as _;
use half::f16 as half_f16;

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// CPU reference on the f16-rounded operands both kernels consume (the
/// scalar kernel rounds A in-register — the same grid — so this oracle
/// differs from either kernel only in accumulation ORDER).
fn cpu_matmul(a: &[f32], b_f16: &[half_f16], m: usize, n: usize, p: usize) -> Vec<f32> {
    let a16: Vec<half_f16> = a.iter().map(|&v| half_f16::from_f32(v)).collect();
    let mut out = vec![0.0f32; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut acc = 0.0f32;
            let a_row = &a16[i * n..(i + 1) * n];
            let b_row = &b_f16[j * n..(j + 1) * n];
            for k in 0..n {
                acc += f32::from(a_row[k]) * f32::from(b_row[k]);
            }
            out[i * p + j] = acc;
        }
    }
    out
}

fn run_kernel(
    use_cmma: bool,
    ctx: &CubeCLContext,
    a_h: Handle,
    b_h: Handle,
    out_h: Handle,
    m: usize,
    n: usize,
    p: usize,
) {
    let cl = ctx.client();
    if use_cmma {
        MatmulF16bCmmaCubeCL::launch::<ActiveRuntime>(&cl, a_h, b_h, out_h, m, n, p);
    } else {
        MatmulF16bCubeCL::launch::<ActiveRuntime>(&cl, a_h, b_h, out_h, m, n, p);
    }
}

/// Both kernels agree with the CPU oracle at the lane's REAL dimension
/// classes (large N accumulation depth + multi-row bounds), CPU cost kept
/// ~sub-second: the m=8 arm carries the full N=4096 depth; the m=87 arm
/// checks multi-row bounds at a shortened N.
#[test]
fn real_shape_parity_both_kernels() {
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let cl = ctx.client();

    for (m, n, p) in [(8usize, 4096usize, 6144usize), (87, 512, 24576)] {
        let mut r = Lcg(0xBEEF_1005);
        let a: Vec<f32> = (0..m * n).map(|_| r.next_f32() * 0.5).collect();
        let b_f32: Vec<f32> = (0..p * n).map(|_| r.next_f32() * 0.5).collect();
        let b_f16: Vec<half_f16> = b_f32.iter().map(|&v| half_f16::from_f32(v)).collect();
        let want = cpu_matmul(&a, &b_f16, m, n, p);
        let denom = want.iter().map(|w| w.abs()).fold(0.0f32, f32::max);

        let a_h = create_f32(&cl, &a);
        let b_h = cl.create_from_slice(bytemuck::cast_slice::<half_f16, u8>(&b_f16));

        for use_cmma in [false, true] {
            let out_h = cl.empty(m * p * core::mem::size_of::<f32>());
            run_kernel(use_cmma, &ctx, a_h.clone(), b_h.clone(), out_h.clone(), m, n, p);
            let got = read_f32(&cl, out_h).expect("read");
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            let rel = worst / denom.max(1e-9);
            assert!(
                rel < 3e-3,
                "parity ({}): [{m}x{n}]x[{p}x{n}] rel {rel} (abs {worst})",
                if use_cmma { "cmma" } else { "scalar" }
            );
            println!(
                "parity {} [{m}x{n}]x[{p}x{n}]: rel {rel:.2e}",
                if use_cmma { "cmma  " } else { "scalar" }
            );
        }
    }
}

/// One timed chunk: K back-to-back launches on REUSED buffers, one
/// readback at the end (the lane's real pipeline runs 4 launches between
/// readbacks — this measures sustained kernel time, not per-launch sync).
fn timed_chunk(
    use_cmma: bool,
    ctx: &CubeCLContext,
    a_h: Handle,
    b_h: Handle,
    out_h: Handle,
    m: usize,
    n: usize,
    p: usize,
    launches: usize,
) -> std::time::Duration {
    let start = std::time::Instant::now();
    for _ in 0..launches {
        run_kernel(use_cmma, ctx, a_h.clone(), b_h.clone(), out_h.clone(), m, n, p);
    }
    let cl = ctx.client();
    let _ = read_f32(&cl, out_h).expect("read");
    start.elapsed()
}

/// The GOAT: interleaved A/B over the lane's four per-layer GEMM shapes at
/// four sequence lengths. PRINTS the table + verdict; asserts nothing about
/// perf (a loaded-box bar rots) — promotion evidence is the printed medians,
/// recorded in the issue row + commit.
#[test]
#[ignore]
fn goat_interleaved_ab() {
    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    println!("PROVENANCE: runtime {}", ActiveRuntime::name(&ctx.client()));

    // The 8B release's per-layer GEMM faces (n_embd 4096, qkv 6144,
    // gate|up 24576, down 12288→4096) at four sequence lengths: a short
    // branch row (8), the published sample (87), a mid state (512), a long
    // state (2048).
    let shapes: Vec<(usize, usize, usize)> = vec![
        (8, 4096, 6144),
        (16, 4096, 6144),
        (18, 4096, 6144),
        (32, 4096, 6144),
        (87, 4096, 6144),
        (87, 4096, 4096),
        (87, 4096, 24576),
        (87, 12288, 4096),
        (512, 4096, 6144),
        (2048, 4096, 6144),
    ];

    const PAIRS: usize = 15;
    const LAUNCHES: usize = 8;
    const WARMUP: usize = 2;

    println!(
        "| {:>18} | {:>10} | {:>10} | {:>7} | {:>8} |",
        "shape (m×n×p)", "scalar µs", "cmma µs", "speedup", "cmma TF"
    );
    let mut all_wins = true;
    for &(m, n, p) in &shapes {
        let mut r = Lcg(0xD00D_1005);
        let a: Vec<f32> = (0..m * n).map(|_| r.next_f32() * 0.5).collect();
        let b_f32: Vec<f32> = (0..p * n).map(|_| r.next_f32() * 0.5).collect();
        let b_f16: Vec<half_f16> = b_f32.iter().map(|&v| half_f16::from_f32(v)).collect();
        let cl = ctx.client();
        let a_h = create_f32(&cl, &a);
        let b_h = cl.create_from_slice(bytemuck::cast_slice::<half_f16, u8>(&b_f16));
        let out_h = cl.empty(m * p * core::mem::size_of::<f32>());

        // Warmup BOTH arms (compilation + clocks) before any timing.
        for _ in 0..WARMUP {
            timed_chunk(false, &ctx, a_h.clone(), b_h.clone(), out_h.clone(), m, n, p, LAUNCHES);
            timed_chunk(true, &ctx, a_h.clone(), b_h.clone(), out_h.clone(), m, n, p, LAUNCHES);
        }

        // Interleaved pairs — median per arm (the sequential-A/B law).
        let mut scalar: Vec<u128> = Vec::with_capacity(PAIRS);
        let mut cmma: Vec<u128> = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            scalar.push(
                timed_chunk(false, &ctx, a_h.clone(), b_h.clone(), out_h.clone(), m, n, p, LAUNCHES)
                    .as_micros(),
            );
            cmma.push(
                timed_chunk(true, &ctx, a_h.clone(), b_h.clone(), out_h.clone(), m, n, p, LAUNCHES)
                    .as_micros(),
            );
        }
        scalar.sort_unstable();
        cmma.sort_unstable();
        let s_med = scalar[PAIRS / 2] as f64;
        let c_med = cmma[PAIRS / 2] as f64;
        let speedup = s_med / c_med;
        let tflops = (2.0 * m as f64 * n as f64 * p as f64) / (c_med * 1e-6) / 1e12;
        if speedup < 1.0 {
            all_wins = false;
        }
        println!(
            "| {:>18} | {:>10.1} | {:>10.1} | {:>6.2}x | {:>8.2} |",
            format!("{m}×{n}×{p}"),
            s_med / LAUNCHES as f64,
            c_med / LAUNCHES as f64,
            speedup,
            tflops
        );
    }
    println!(
        "VERDICT: {}",
        if all_wins {
            "cmma wins every cell — promote candidate"
        } else {
            "mixed — read the table per shape before any promotion"
        }
    );
}
