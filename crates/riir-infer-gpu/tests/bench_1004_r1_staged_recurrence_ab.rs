//! riir-ai Issue 1004 R1 — stage-isolated A/B: the threadgroup-staged
//! multi-token DeltaNet recurrence vs the shipping multi-token kernel.
//!
//! Recurrence-only timing at the Bonsai-2 27B GDN shape (48 value heads ×
//! head_dim 128), P = 2048 and 16384. Each `(rows, tb)` staging shape is
//! measured in INTERLEAVED pairs against the shipping kernel (the order
//! alternates per pair) and reported as the median of the per-pair
//! `staged / shipping` time ratios, so box drift lands on both arms of a pair.
//! Every staged run's output is also checked bit-for-bit against the
//! shipping kernel's, so a fast wrong arm cannot win.
//!
//! A PROVENANCE line (power source, load average) prints beside the numbers.
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/b1004 cargo test -p riir-infer-gpu --release \
//!     --features deltanet_recurrence_smem_staged \
//!     --test bench_1004_r1_staged_recurrence_ab -- --ignored --nocapture
//! ```
//!
//! Env: `B1004_PAIRS` (default 9), `B1004_P` (comma list, default
//! `2048,16384`).
#![cfg(feature = "deltanet_recurrence_smem_staged")]

use std::time::Instant;

use cubecl::prelude::*;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL;
use riir_infer_gpu::deltanet_recurrence_staged_cubecl::DeltanetRecurrenceStagedCubeCL;

const N_HEAD: usize = 48;
const HEAD_DIM: usize = 128;
const SHAPES: &[(usize, usize)] = &[
    (4, 16),
    (8, 8),
    (8, 16),
    (8, 24),
    (16, 8),
    (16, 16),
    (32, 8),
];

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
    qkvx: cubecl::server::Handle,
    beta: cubecl::server::Handle,
    decay: cubecl::server::Handle,
    state0: Vec<f32>,
}

/// One timed run of an arm; returns (seconds, output bits).
fn run_arm(
    client: &ComputeClient<ActiveRuntime>,
    b: &Bufs,
    p: usize,
    staged: Option<(usize, usize)>,
) -> (f64, Vec<u32>) {
    let state = client.create_from_slice(f32::as_bytes(&b.state0));
    let output = client.empty(p * N_HEAD * HEAD_DIM * 4);
    pollster::block_on(client.sync()).expect("sync before");
    let t = Instant::now();
    unsafe {
        match staged {
            None => DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                client,
                b.qkvx.clone(),
                b.beta.clone(),
                b.decay.clone(),
                state,
                output.clone(),
                N_HEAD,
                HEAD_DIM,
                p,
            ),
            Some((rows, tb)) => DeltanetRecurrenceStagedCubeCL::launch_shape::<ActiveRuntime>(
                client,
                b.qkvx.clone(),
                b.beta.clone(),
                b.decay.clone(),
                state,
                output.clone(),
                N_HEAD,
                HEAD_DIM,
                p,
                rows,
                tb,
            ),
        }
    }
    pollster::block_on(client.sync()).expect("sync after");
    let secs = t.elapsed().as_secs_f64();
    let bits = f32::from_bytes(&client.read_one_unchecked(output))
        .iter()
        .map(|v| v.to_bits())
        .collect();
    (secs, bits)
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
fn bench_1004_r1_staged_recurrence_ab() {
    let pairs: usize = std::env::var("B1004_PAIRS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9);
    let ps: Vec<usize> = std::env::var("B1004_P")
        .unwrap_or_else(|_| "2048,16384".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert!(pairs >= 3, "need at least 3 pairs for a median");
    for &(r, tb) in SHAPES {
        assert!(
            DeltanetRecurrenceStagedCubeCL::valid_shape(r, tb),
            "shape ({r},{tb}) invalid"
        );
    }

    println!(
        "PROVENANCE: power [{}] powermode [{}] loadavg [{}] pairs={pairs}",
        sh("pmset", &["-g", "batt"]),
        sh("sh", &["-c", "pmset -g | grep -i powermode"]),
        sh("sysctl", &["-n", "vm.loadavg"]),
    );

    let ctx = CubeCLContext::new().expect("CubeCL should initialize");
    let client = ctx.client();

    for &p in &ps {
        let v_dim = N_HEAD * HEAD_DIM;
        let b = Bufs {
            qkvx: client.create_from_slice(f32::as_bytes(&lcg_fill(
                0x1004 ^ p as u64,
                p * 3 * v_dim,
                -0.18,
                0.18,
            ))),
            beta: client.create_from_slice(f32::as_bytes(&lcg_fill(
                0xB ^ p as u64,
                p * N_HEAD,
                0.05,
                0.95,
            ))),
            decay: client.create_from_slice(f32::as_bytes(&lcg_fill(
                0xD ^ p as u64,
                p * N_HEAD,
                0.80,
                0.999,
            ))),
            state0: lcg_fill(0x5, N_HEAD * HEAD_DIM * HEAD_DIM, -0.05, 0.05),
        };

        // Warm every arm once (pipeline compile + first-touch), and take the
        // reference bits.
        let (_, ref_bits) = run_arm(&client, &b, p, None);
        assert!(
            ref_bits.iter().any(|&x| x != 0),
            "reference output all zero"
        );
        for &s in SHAPES {
            let (_, bits) = run_arm(&client, &b, p, Some(s));
            assert!(
                bits == ref_bits,
                "P={p} shape {s:?}: staged output differs from shipping"
            );
        }

        let mut base_all = Vec::new();
        for &s in SHAPES {
            let mut ratios = Vec::with_capacity(pairs);
            let mut staged_t = Vec::with_capacity(pairs);
            for i in 0..pairs {
                let (ta, tb_) = if i % 2 == 0 {
                    let a = run_arm(&client, &b, p, None).0;
                    (a, run_arm(&client, &b, p, Some(s)).0)
                } else {
                    let bb = run_arm(&client, &b, p, Some(s)).0;
                    (run_arm(&client, &b, p, None).0, bb)
                };
                ratios.push(tb_ / ta);
                staged_t.push(tb_);
                base_all.push(ta);
            }
            let (lo, hi) = {
                let mut r = ratios.clone();
                r.sort_by(f64::total_cmp);
                (r[0], r[r.len() - 1])
            };
            println!(
                "P={p:>6} rows={:>2} tb={:>2}  staged median {:>8.2} ms  ratio staged/shipping median {:.3} (min {:.3} max {:.3}) → speedup {:.3}x",
                s.0,
                s.1,
                median(staged_t) * 1e3,
                median(ratios.clone()),
                lo,
                hi,
                1.0 / median(ratios),
            );
        }
        println!(
            "P={p:>6} shipping median {:>8.2} ms over {} runs",
            median(base_all) * 1e3,
            SHAPES.len() * pairs
        );
    }
}
