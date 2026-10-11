//! `multicol_ab` — Issue 038's owed paired bench (step 3): the n∈[3,8]
//! decode band, `gemv_ternary_cuda_raw` (the incumbent n=1 handler, called
//! once per token) vs `gemv_ternary_multicol_cuda_raw` (one call for the
//! whole batch) vs `gemm_ternary_i8_mma_cuda_raw` (the shipping prefill
//! quantize+GEMM pair), at Bonsai-2 real layer shapes.
//!
//! Protocol (plan 612 G2 runbook, adapted to three kernel arms): R reps per
//! (shape, n) with the arm ORDER ROTATED per rep (rep r starts at arm
//! r % 3) so position bias cancels; verdict = median of same-rep PAIRED
//! ratios (multicol / n1_loop and multicol / mma), never cross-window
//! comparisons.
//!
//! ⚠ What the numbers MEAN: these are HANDLER-CONTRACT wall times — each
//! arm times what a driver pays calling that arm's public API today,
//! including the GEMV arms' per-call host quantize + upload and the MMA
//! arm's f32 upload → device quantize → GEMM → download sequence (the
//! shipping prefill dispatch steps 3-5). The pure-kernel claim is upstream
//! ncu's (this port is bit-identical to the n=1 kernel by construction);
//! the wired driver A/B (GPU-resident activations, step 4) is the GOAT
//! surface. Read these numbers as the standalone-handler comparison the
//! issue specifies, not as end-to-end decode latency.
//!
//! Box-state discipline: this binary prints the config only — the RUNNER
//! quotes `scripts/bench_preflight`-class provenance in the bench record
//! (GPU-exclusive window, power state, load) per the Issue-021 law.
//! Run `--release` only.
//!
//! ```text
//! cargo run --release -p riir-infer-gpu --features ternary_gemv_cuda_raw \
//!   --example multicol_ab [--reps 30] [--warmup 3] [--shapes big,small] \
//!   [--n 3,4,5,6,7,8]
//! ```

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use katgpt_core::TernaryGroupWeights;

use riir_infer_gpu::gemm_ternary_i8_mma_cuda_raw::GemmTernaryI8MmaCuda;
use riir_infer_gpu::gemv_ternary_cuda_raw::TernaryGemmCudaRaw;
use riir_infer_gpu::gemv_ternary_multicol_cuda_raw::TernaryGemmMultiColCudaRaw;

/// The token band the multicol kernel owns. (The module constants are the
/// authority; this array mirrors them for arg validation + iteration.)
const N_BAND: [usize; 6] = [3, 4, 5, 6, 7, 8];

const BIG: (usize, usize) = (5120, 17408); // Bonsai-2 FFN down_proj class
const SMALL: (usize, usize) = (1536, 4096); // sanity-small layer

#[derive(Clone, Copy)]
enum Arm {
    N1Loop,
    Multicol,
    Mma,
}

impl Arm {
    fn slot(self) -> usize {
        match self {
            Arm::N1Loop => 0,
            Arm::Multicol => 1,
            Arm::Mma => 2,
        }
    }
}

/// Deterministic weights: every code appears, rows are phase-rotated so no
/// two rows are identical; scales drawn per group from a fixed LCG in
/// [0.4, 0.8].
fn make_weights(m: usize, n: usize) -> TernaryGroupWeights {
    let mut w = TernaryGroupWeights::new(m, n);
    for row in 0..m {
        for col in 0..n {
            let val: i8 = match (col + row) % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            };
            w.set(row, col, val);
        }
    }
    let mut state: u32 = 0x5EED_0380 ^ (m as u32) ^ (n as u32).rotate_left(13);
    for s in &mut w.group_scale {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = ((state >> 16) as f32) / 65535.0;
        *s = f16::from_f32(0.4 + 0.4 * unit);
    }
    w
}

/// LCG-deterministic activations in [-1, 1) (same generator family as the
/// module tests).
fn lcg_inputs(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0
        })
        .collect()
}

struct MmaArm {
    kernels: GemmTernaryI8MmaCuda,
    stream: Arc<CudaStream>,
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn quartiles(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[s.len() / 4], s[(3 * s.len()) / 4])
}

fn main() {
    let mut reps = 30usize;
    let mut warmup = 3usize;
    let mut shape_filter = String::new();
    let mut n_list: Vec<usize> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reps" => reps = args.next().and_then(|v| v.parse().ok()).unwrap_or(reps),
            "--warmup" => warmup = args.next().and_then(|v| v.parse().ok()).unwrap_or(warmup),
            "--shapes" => shape_filter = args.next().unwrap_or_default(),
            "--n" => {
                if let Some(v) = args.next() {
                    n_list = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                }
            }
            other => {
                eprintln!("unknown arg {other:?} — see the module doc for usage");
                std::process::exit(2);
            }
        }
    }
    if n_list.is_empty() {
        n_list = N_BAND.to_vec();
    }
    for &n in &n_list {
        assert!(
            (3..=8).contains(&n),
            "n={n} outside the multicol band [3,8] — the kernel refuses it by design"
        );
    }
    let shapes: Vec<(&str, usize, usize)> = [("big", BIG.0, BIG.1), ("small", SMALL.0, SMALL.1)]
        .into_iter()
        .filter(|(name, _, _)| shape_filter.is_empty() || shape_filter.contains(name))
        .collect();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("# multicol_ab — issue 038 step 3 (paired bench, plan-612 G2 runbook adaptation)");
    println!(
        "# unix={now} · reps={reps} warmup={warmup} shapes={} · arm order rotated per rep (position-balanced)",
        shapes.iter().map(|s| s.0).collect::<Vec<_>>().join(",")
    );
    println!("# HANDLER-CONTRACT wall times (host quantize + transfers included per arm's own API) — see module doc");

    // ---- device + handlers ----
    let ctx = CudaContext::new(0).unwrap_or_else(|e| {
        eprintln!(
            "FATAL: no CUDA device 0 ({e}) — this bench needs a free, exclusive GPU window"
        );
        std::process::exit(1);
    });
    let mut h_n1 = TernaryGemmCudaRaw::new().expect("n1 handler init");
    let mut h_mc = TernaryGemmMultiColCudaRaw::new().expect("multicol handler init");
    let mma_arm = {
        let kernels = GemmTernaryI8MmaCuda::new(ctx.clone()).expect("mma kernels init");
        let stream = ctx.new_stream().expect("mma stream");
        MmaArm { kernels, stream }
    };

    // verdict-block rows, printed at the end
    let mut verdict_rows: Vec<String> = Vec::new();

    for (shape_name, m, n) in shapes {
        let w = make_weights(m, n);
        let weight_bytes = w.pos_bits.len() * 8 + w.neg_bits.len() * 8 + w.group_scale.len() * 2;

        let idx_n1 = h_n1.upload_weights(&w).expect("n1 upload");
        let idx_mc = h_mc.upload_weights(&w).expect("mc upload");

        // MMA mirrors: bitplane pair as u32 words (the cache layout's own
        // word-count law: m * blocks64 * 2 per plane) + f32 scales.
        let to_u32_words = |bits: &[u64]| -> Vec<u32> {
            let mut out = Vec::with_capacity(bits.len() * 2);
            for &word in bits {
                out.push(word as u32);
                out.push((word >> 32) as u32);
            }
            out
        };
        let pos_dev: CudaSlice<u32> = mma_arm.stream.clone_htod(&to_u32_words(&w.pos_bits)).unwrap();
        let neg_dev: CudaSlice<u32> = mma_arm.stream.clone_htod(&to_u32_words(&w.neg_bits)).unwrap();
        let scale_dev: CudaSlice<f32> = mma_arm
            .stream
            .clone_htod(&w.group_scale.iter().map(|s| s.to_f32()).collect::<Vec<_>>())
            .unwrap();

        // Activations for the max n once; per-n prefixes reuse it.
        let max_p = *n_list.last().unwrap();
        let xs_all = lcg_inputs(0xCAFE_0380 ^ n as u32, max_p * n);
        let mut in_dev: CudaSlice<f32> = mma_arm
            .stream
            .clone_htod(&xs_all[..max_p * n].to_vec())
            .unwrap();
        let out_dev: CudaSlice<f32> = mma_arm.stream.alloc_zeros(max_p * m).unwrap();
        let mut out_host = vec![0f32; max_p * m];

        for &tokens in &n_list {
            let xs = &xs_all[..tokens * n];
            let mut buf_a = vec![0f32; tokens * m];
            let mut buf_b = vec![0f32; tokens * m];
            let scratch = mma_arm
                .kernels
                .alloc_scratch(&mma_arm.stream, n, tokens)
                .expect("mma scratch");

            // Arm bodies — each is the arm's full public-API contract.
            let mut run_n1 = || {
                for tok in 0..tokens {
                    h_n1.forward(
                        idx_n1,
                        &xs[tok * n..(tok + 1) * n],
                        &mut buf_a[tok * m..(tok + 1) * m],
                    )
                    .expect("n1 forward");
                }
            };
            let mut run_mc = || {
                h_mc.forward_multicol(idx_mc, xs, tokens, &mut buf_b)
                    .expect("multicol forward");
            };
            let mut run_mma = || {
                let in_len = tokens * n;
                {
                    let mut in_view = in_dev.try_slice_mut(0..in_len).expect("in view");
                    mma_arm
                        .stream
                        .memcpy_htod(xs, &mut in_view)
                        .expect("mma upload");
                }
                mma_arm
                    .kernels
                    .launch_prefill_quantize(&mma_arm.stream, &in_dev, &scratch, n, tokens)
                    .expect("mma quantize");
                mma_arm
                    .kernels
                    .launch_prefill_gemm(
                        &mma_arm.stream,
                        &pos_dev,
                        &neg_dev,
                        &scale_dev,
                        &scratch,
                        &out_dev,
                        m,
                        n,
                        tokens,
                    )
                    .expect("mma gemm");
                let view = out_dev.try_slice(0..tokens * m).expect("out view");
                mma_arm
                    .stream
                    .memcpy_dtoh(&view, &mut out_host[..tokens * m])
                    .expect("mma download");
            };

            // Warmup.
            for _ in 0..warmup {
                run_n1();
                run_mc();
                run_mma();
            }

            let mut times: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for rep in 0..reps {
                // Order rotation: rep r starts at arm r % 3.
                let order: [Arm; 3] = match rep % 3 {
                    0 => [Arm::N1Loop, Arm::Multicol, Arm::Mma],
                    1 => [Arm::Multicol, Arm::Mma, Arm::N1Loop],
                    _ => [Arm::Mma, Arm::N1Loop, Arm::Multicol],
                };
                for arm in order {
                    let t0 = Instant::now();
                    match arm {
                        Arm::N1Loop => run_n1(),
                        Arm::Multicol => run_mc(),
                        Arm::Mma => run_mma(),
                    }
                    times[arm.slot()].push(t0.elapsed().as_secs_f64() * 1e6);
                }
            }

            let meds: Vec<f64> = times.iter().map(|t| median(t)).collect();
            let (p25_n1, p75_n1) = quartiles(&times[0]);
            let (p25_mc, p75_mc) = quartiles(&times[1]);
            let (p25_mma, p75_mma) = quartiles(&times[2]);
            let gbps = |i: usize| weight_bytes as f64 / 1e9 / (meds[i] / 1e6);
            let r_mc_n1: Vec<f64> =
                (0..reps).map(|r| times[1][r] / times[0][r]).collect();
            let r_mc_mma: Vec<f64> =
                (0..reps).map(|r| times[1][r] / times[2][r]).collect();

            println!(
                "\n## {shape_name} m={m} n={n} tokens={tokens} · weight {:.2} MiB",
                weight_bytes as f64 / (1024.0 * 1024.0)
            );
            println!("| arm | median µs | p25 | p75 | weight GB/s |");
            println!("|---|---|---|---|---|");
            println!(
                "| n1_loop | {:.1} | {:.1} | {:.1} | {:.0} |",
                meds[0], p25_n1, p75_n1, gbps(0)
            );
            println!(
                "| multicol | {:.1} | {:.1} | {:.1} | {:.0} |",
                meds[1], p25_mc, p75_mc, gbps(1)
            );
            println!(
                "| mma | {:.1} | {:.1} | {:.1} | {:.0} |",
                meds[2], p25_mma, p75_mma, gbps(2)
            );
            println!(
                "paired same-rep ratios: mc/n1 median {:.3} · mc/mma median {:.3}",
                median(&r_mc_n1),
                median(&r_mc_mma)
            );
            verdict_rows.push(format!(
                "- {shape_name} n={tokens}: mc/n1 {:.3} (mc {:.0}µs vs n1 {:.0}µs) · mc/mma {:.3}",
                median(&r_mc_n1),
                meds[1],
                meds[0],
                median(&r_mc_mma),
            ));
        }

        drop(pos_dev);
        drop(neg_dev);
        drop(scale_dev);
        drop(in_dev);
        drop(out_dev);
    }

    println!("\n# verdict summary (median same-rep paired ratios)");
    for row in &verdict_rows {
        println!("{row}");
    }
}
