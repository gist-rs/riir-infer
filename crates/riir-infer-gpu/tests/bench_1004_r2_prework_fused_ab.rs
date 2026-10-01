//! riir-ai Issue 1004 R2 — stage-isolated A/B: the fused GDN prework
//! (conv1d + SiLU + q/k L2-norm + head expansion, ONE dispatch) vs the
//! shipping chain (per-64 chunked conv1d + the batched
//! expand-and-L2-normalize) at the Bonsai-2 27B GDN dims
//! (n_k 16 × n_v 48 × head_dim 128, kernel_size 4).
//!
//! Prework-only timing (the projections and the beta/decay dispatch are
//! identical in both arms and omitted). P = 2048 and 4096 by default.
//! INTERLEAVED pairs with alternating order; the median of the per-pair
//! `fused / shipping` ratios is the verdict, so box drift lands on both arms
//! of a pair. Every fused run's output + carried conv state is checked
//! bit-for-bit against the shipping chain's, so a fast wrong arm cannot win.
//!
//! A PROVENANCE line (power source, power mode, load average) prints beside
//! the numbers — the G2 box-state law.
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/b1004r2 cargo test -p riir-infer-gpu --release \
//!     --features deltanet_prework_fused \
//!     --test bench_1004_r2_prework_fused_ab -- --ignored --nocapture
//! ```
//!
//! Env: `B1004_PAIRS` (default 11), `B1004_P` (comma list, default
//! `2048,4096`).
#![cfg(feature = "deltanet_prework_fused")]

use std::time::Instant;

use cubecl::prelude::*;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::deltanet_chunked_cubecl::DeltanetChunkedConv1dCubeCL;
use riir_infer_gpu::deltanet_cubecl::ExpandAndL2NormalizeHeadsBatchedCubeCL;
use riir_infer_gpu::deltanet_prework_fused_cubecl::DeltanetPreworkFusedCubeCL;

const N_K: usize = 16;
const N_V: usize = 48;
const HEAD_DIM: usize = 128;
const KS: usize = 4;
const CONV_DIM: usize = (2 * N_K + N_V) * HEAD_DIM;
const QKVX_DIM: usize = 3 * N_V * HEAD_DIM;
const PREFILL_CHUNK_SIZE: usize = 64; // the production conv chunk

fn lcg_fill(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lo + (hi - lo) * ((x >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn sh(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().replace('\n', " "))
        .unwrap_or_else(|_| "unavailable".into())
}

struct Bufs {
    input: cubecl::server::Handle,
    weight: cubecl::server::Handle,
    carry0: Vec<f32>,
}

/// One timed run of an arm; returns (seconds, expanded bits, carry bits).
fn run_arm(
    client: &ComputeClient<ActiveRuntime>,
    b: &Bufs,
    p: usize,
    fused: bool,
) -> (f64, Vec<u32>, Vec<u32>) {
    let carry = client.create_from_slice(f32::as_bytes(&b.carry0));
    let expanded = client.empty(p * QKVX_DIM * 4);
    pollster::block_on(client.sync()).expect("sync before");
    let t = Instant::now();
    if fused {
        unsafe {
            DeltanetPreworkFusedCubeCL::launch::<ActiveRuntime>(
                client,
                b.input.clone(),
                expanded.clone(),
                b.weight.clone(),
                carry.clone(),
                p,
                CONV_DIM,
                KS,
                N_K,
                N_V,
                HEAD_DIM,
            );
        }
    } else {
        // The shipping chain, in production dispatch order: per-64 chunked
        // conv (which issues its own ordered carry updates) into a separate
        // intermediate buffer, then the batched expand over the whole P.
        let conv_out = client.empty(p * CONV_DIM * 4);
        let mut t0 = 0usize;
        while t0 < p {
            let c_len = PREFILL_CHUNK_SIZE.min(p - t0);
            let in_chunk = b
                .input
                .clone()
                .offset_start((t0 * CONV_DIM * 4) as u64)
                .offset_end(((p - t0 - c_len) * CONV_DIM * 4) as u64);
            let out_chunk = conv_out
                .clone()
                .offset_start((t0 * CONV_DIM * 4) as u64)
                .offset_end(((p - t0 - c_len) * CONV_DIM * 4) as u64);
            unsafe {
                DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                    client,
                    in_chunk,
                    out_chunk,
                    b.weight.clone(),
                    carry.clone(),
                    c_len,
                    CONV_DIM,
                    KS,
                    KS,
                    1,
                );
            }
            t0 += c_len;
        }
        unsafe {
            ExpandAndL2NormalizeHeadsBatchedCubeCL::launch::<ActiveRuntime>(
                client,
                conv_out,
                expanded.clone(),
                N_K,
                N_V,
                HEAD_DIM,
                p,
            );
        }
    }
    pollster::block_on(client.sync()).expect("sync after");
    let secs = t.elapsed().as_secs_f64();
    let bits = |h: cubecl::server::Handle| -> Vec<u32> {
        f32::from_bytes(&client.read_one_unchecked(h))
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    (secs, bits(expanded), bits(carry))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[test]
#[ignore = "GPU A/B bench — run with --ignored --nocapture in --release"]
fn bench_1004_r2_prework_fused_ab() {
    let pairs: usize = std::env::var("B1004_PAIRS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11);
    let ps: Vec<usize> = std::env::var("B1004_P")
        .unwrap_or_else(|_| "2048,4096".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert!(pairs >= 3, "need at least 3 pairs for a median");
    assert!(
        DeltanetPreworkFusedCubeCL::supports(HEAD_DIM, N_K, N_V),
        "bench shape must be supported"
    );

    println!(
        "PROVENANCE: power [{}] powermode [{}] loadavg [{}] pairs={pairs}",
        sh("pmset", &["-g", "batt"]),
        sh("sh", &["-c", "pmset -g | grep -i powermode"]),
        sh("sysctl", &["-n", "vm.loadavg"]),
    );

    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    for &p in &ps {
        let b = Bufs {
            input: client.create_from_slice(f32::as_bytes(&lcg_fill(
                0x1004 ^ p as u64,
                p * CONV_DIM,
                -0.35,
                0.35,
            ))),
            weight: client.create_from_slice(f32::as_bytes(&lcg_fill(
                0x57 ^ p as u64,
                CONV_DIM * KS,
                -0.12,
                0.12,
            ))),
            carry0: lcg_fill(0xC0 ^ p as u64, CONV_DIM * KS, -0.3, 0.3),
        };

        // Warm both arms once (pipeline compile + first-touch), take the
        // reference bits, and pin bit-identity before any timing claim.
        let (_, ref_exp, ref_carry) = run_arm(&client, &b, p, false);
        assert!(ref_exp.iter().any(|&x| x != 0), "reference output all zero");
        let (_, f_exp, f_carry) = run_arm(&client, &b, p, true);
        assert_eq!(f_exp, ref_exp, "P={p}: fused expanded output differs");
        assert_eq!(f_carry, ref_carry, "P={p}: fused carry differs");

        let mut ratios = Vec::with_capacity(pairs);
        let mut fused_ms = Vec::with_capacity(pairs);
        let mut ship_ms = Vec::with_capacity(pairs);
        for i in 0..pairs {
            // Alternate the order per pair; bit-check EVERY fused run.
            let (t_ship, t_fused) = if i % 2 == 0 {
                let a = run_arm(&client, &b, p, false);
                let b_ = run_arm(&client, &b, p, true);
                assert_eq!(b_.1, ref_exp, "P={p} pair {i}: fused output drifted");
                (a.0, b_.0)
            } else {
                let b_ = run_arm(&client, &b, p, true);
                assert_eq!(b_.1, ref_exp, "P={p} pair {i}: fused output drifted");
                (run_arm(&client, &b, p, false).0, b_.0)
            };
            ratios.push(t_fused / t_ship);
            fused_ms.push(t_fused * 1e3);
            ship_ms.push(t_ship * 1e3);
        }
        let (lo, hi) = {
            let mut r = ratios.clone();
            r.sort_by(f64::total_cmp);
            (r[0], r[r.len() - 1])
        };
        println!(
            "P={p:>6}  fused median {:>8.2} ms vs shipping median {:>8.2} ms  ratio fused/shipping median {:.3} (min {:.3} max {:.3}) → speedup {:.3}x",
            median(fused_ms),
            median(ship_ms),
            median(ratios.iter().copied().collect()),
            lo,
            hi,
            1.0 / median(ratios.iter().copied().collect()),
        );
    }
}
