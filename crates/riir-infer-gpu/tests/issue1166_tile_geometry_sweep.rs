//! Issue 1166 rung A — the SECOND-generation tile-geometry sweep for the
//! ternary matmul2d prefill GEMM.
//!
//! The Bench-1113 cell-2 sweep (7 configs) promoted `ktile64_64x128x4sg`
//! (Plan 641 T4, riir-ai Benches 1009/1010) — but its search space never
//! composed the winner with the axes it varied independently: K-tile 128,
//! NRB 256 and NRA 128 AT K-tile 64, and 8 simdgroups. This instrument
//! sweeps that unsearched space with the PROMOTED INCUMBENT as the
//! reference arm (the simdgroup-vs-tensor question is settled; the demote
//! K32 arm rides as the floor reference). G1 = scale-relative 1e-5 against
//! a CPU dequant-GEMV reference (config-independent truth, no shared GPU
//! path) + the never-written sentinel; G2 = interleaved rotating-start
//! timing, median over rounds, per-launch ms.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1166 cargo test -p riir-infer-gpu --release --features \
//!   metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,\
//!   ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,\
//!   riir-infer-core/bonsai2_hadamard \
//!   --test issue1166_tile_geometry_sweep -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The `[[test]]` row's required-features is the FULL feature set above — a
//! shorter `--features` list makes cargo silently skip the target (a green
//! zero that ran nothing).
//!
//! Env: `BENCH1166_SHAPES` (`MxN,MxN,…`, default the three Bonsai-2
//! prefill bottlenecks), `BENCH1166_PS` (default `512,2048,4096` — the
//! dist w512 frame + the league rows), `BENCH1166_ROUNDS` (4),
//! `BENCH1166_LAUNCHES` (2).

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use katgpt_core::TernaryGroupWeights;
use riir_infer_gpu::gemm_ternary_metal_tensor::{MetalTensorGemm, MetalTileConfig, MetalWeightCache};

/// G1 scale-relative bound — bench_1113's bar (bench_645 lineage).
const G1_REL_LIMIT: f64 = 1e-5;

/// Sentinel written into output buffers before launch (a kernel that fails
/// to write an element is caught, never inherited).
const SENTINEL: f32 = -98765.4;

/// The three Bonsai-2 prefill bottlenecks (Bench 925 census: FFN + DeltaNet
/// projections = 63% of the promoted pp4096 frame): `(name, M, N)` where
/// the GEMM is `Y[P×M] = dequant(W[M×N]) @ X[N per token]`.
const DEFAULT_SHAPES: &[(&str, usize, usize)] = &[
    ("ffn_gate/up", 17408, 5120),
    ("ffn_down", 5120, 17408),
    ("attn_q", 12288, 5120),
];

/// P = 512 (the dist w512 frame this issue's residual was measured at) +
/// the league rows.
const DEFAULT_PS: &[usize] = &[512, 2048, 4096];

/// The sweep. First entry = the PROMOTED incumbent (the reference ratios
/// read against it); last = the K32 demote (the floor reference). The four
/// middle entries are the unsearched ktile64 compositions (Bench-1113
/// cell-2 varied the axes but never crossed them with K-tile 64; NRB 256
/// and NRA 128 were only measured at K32, and 8 simdgroups never at all).
const SWEEP: &[(&str, MetalTileConfig)] = &[
    (
        "ktile64_64x128x4sg",
        MetalTileConfig { nra: 64, nrb: 128, nk_chunk: 4, nsg: 4 },
    ),
    (
        "ktile128_64x128x4sg",
        MetalTileConfig { nra: 64, nrb: 128, nk_chunk: 8, nsg: 4 },
    ),
    (
        "ktile64_64x256x4sg",
        MetalTileConfig { nra: 64, nrb: 256, nk_chunk: 4, nsg: 4 },
    ),
    (
        "ktile64_128x128x4sg",
        MetalTileConfig { nra: 128, nrb: 128, nk_chunk: 4, nsg: 4 },
    ),
    (
        "ktile64_64x128x8sg",
        MetalTileConfig { nra: 64, nrb: 128, nk_chunk: 4, nsg: 8 },
    ),
    (
        "ktile64_64x128x2sg",
        MetalTileConfig { nra: 64, nrb: 128, nk_chunk: 4, nsg: 2 },
    ),
    (
        "k32_64x128x4sg",
        MetalTileConfig { nra: 64, nrb: 128, nk_chunk: 2, nsg: 4 },
    ),
];

// ---------------------------------------------------------------------------
// Fixtures (bench_645's deterministic LCG + realistic ternary fill)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// Realistic ternary fill matching the measured Bonsai distribution
/// (35.1% −1 / 29.9% 0 / 35.0% +1) — the same fill bench_1113 used.
fn build(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
    let mut w = TernaryGroupWeights::new(rows, cols);
    let mut rng = Lcg(seed);
    for r in 0..rows {
        for c in 0..cols {
            let v = match rng.next_u64() % 1000 {
                0..=350 => -1i8,
                351..=649 => 0,
                _ => 1,
            };
            if v != 0 {
                w.set(r, c, v);
            }
        }
    }
    w
}

fn random_input(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len)
        .map(|_| ((rng.next_u64() >> 40) as f32 / 8_388_608.0) - 1.0)
        .collect()
}

// ---------------------------------------------------------------------------
// CPU dequant-GEMV reference — config-independent truth, no shared GPU path
// ---------------------------------------------------------------------------

/// `want[t * rows + r] = sum_e dequant(w[r][e]) * x[t * cols + e]`, decoded
/// straight from the bit-plane layout (`TernaryGroupWeights` doc: pos/neg
/// words `[rows * blocks64]`, one f16 scale per 128-element group per row).
fn cpu_reference(w: &TernaryGroupWeights, x: &[f32], p: usize) -> Vec<f32> {
    let (rows, cols) = (w.rows, w.cols);
    let group_size = (cols / w.groups_per_row).max(1);
    let mut out = vec![0.0f32; p * rows];
    for r in 0..rows {
        let pos = &w.pos_bits[r * w.blocks64..(r + 1) * w.blocks64];
        let neg = &w.neg_bits[r * w.blocks64..(r + 1) * w.blocks64];
        let scales = &w.group_scale[r * w.groups_per_row..(r + 1) * w.groups_per_row];
        for t in 0..p {
            let x_row = &x[t * cols..(t + 1) * cols];
            let mut acc = 0.0f32;
            for b in 0..w.blocks64 {
                let pw = pos[b];
                let nw = neg[b];
                if pw == 0 && nw == 0 {
                    continue;
                }
                for lane in 0..64 {
                    let e = b * 64 + lane;
                    if e >= cols {
                        break;
                    }
                    let bit = 1u64 << lane;
                    let sign = if pw & bit != 0 {
                        1.0f32
                    } else if nw & bit != 0 {
                        -1.0f32
                    } else {
                        continue;
                    };
                    acc += sign * scales[e / group_size].to_f32() * x_row[e];
                }
            }
            out[t * rows + r] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Metal arm plumbing (bench_1113's shape — this file never names metal-rs)
// ---------------------------------------------------------------------------

struct MetalArmWeights {
    cache: MetalWeightCache,
    blocks64: usize,
    groups_per_row: usize,
    m: usize,
    n: usize,
}

fn metal_weights(gemm: &MetalTensorGemm, w: &TernaryGroupWeights) -> MetalArmWeights {
    let pos_u32: Vec<u32> = w
        .pos_bits
        .iter()
        .flat_map(|&v| [v as u32, (v >> 32) as u32])
        .collect();
    let neg_u32: Vec<u32> = w
        .neg_bits
        .iter()
        .flat_map(|&v| [v as u32, (v >> 32) as u32])
        .collect();
    let scale_f32: Vec<f32> = w.group_scale.iter().map(|s| s.to_f32()).collect();
    MetalArmWeights {
        cache: MetalWeightCache::upload(gemm, &pos_u32, &neg_u32, &scale_f32),
        blocks64: w.blocks64,
        groups_per_row: w.groups_per_row,
        m: w.rows,
        n: w.cols,
    }
}

/// One raw-Metal cooperative launch (self-synchronizing: the launch fences
/// on `wait_until_completed`, so a timed region cannot exclude the work).
fn launch_metal(gemm: &MetalTensorGemm, cache: &MetalArmWeights, input: &[f32], out_len: usize) {
    let input_buf = gemm.upload_f32(input);
    let output_buf = gemm.create_output(out_len);
    gemm.launch(
        &cache.cache.pos,
        &cache.cache.neg,
        &cache.cache.scale,
        &input_buf,
        &output_buf,
        cache.blocks64 as u32,
        cache.groups_per_row as u32,
        cache.n as u32,
        cache.m as u32,
        input.len() as u32 / cache.n as u32,
    );
}

// ---------------------------------------------------------------------------
// PROVENANCE (the box-state law — print, never gate)
// ---------------------------------------------------------------------------

fn loadavg_1m() -> Option<f64> {
    let out = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("vm.loadavg")
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace().nth(1)?.parse().ok()
}

fn print_provenance(when: &str) {
    match loadavg_1m() {
        Some(load) => {
            println!("PROVENANCE ({when}): loadavg1m={load:.2}");
            if load > 6.0 {
                println!(
                    "⚠ LOADED BOX (load {load:.2} > 6.0, the bench_preflight bar): the \
                     TIMING rows below are NOT quotable — G1 rows remain valid."
                );
            }
        }
        None => println!("PROVENANCE ({when}): loadavg unavailable"),
    }
}

// ---------------------------------------------------------------------------
// G1 — every compiled config must match the CPU reference on ragged shapes
// ---------------------------------------------------------------------------

/// (rows, cols, p) — ragged coverage on every axis (edge tiles exercise the
/// cooperative-tensor bounds handling per config), plus one real projection.
const G1_CASES: &[(usize, usize, usize)] = &[
    (64, 256, 3),
    (48, 5120, 5),
    (130, 384, 9),
    (33, 192, 3),
    (256, 640, 16),
    (1024, 5120, 8),
];

#[test]
#[ignore = "requires macOS Metal + a GPU; run explicitly with --release"]
fn g1_sweep_configs_match_cpu_reference() {
    let mut any_compiled = false;
    let mut failed_configs: Vec<&str> = Vec::new();
    for &(name, cfg) in SWEEP {
        let gemm = match MetalTensorGemm::new_with_config(cfg) {
            Ok(g) => g,
            Err(e) => {
                println!("  G1 {name:<20} COMPILE_FAIL — {e}");
                continue;
            }
        };
        any_compiled = true;

        let mut worst = (0.0f64, String::new());
        let mut failed = false;
        for &(rows, cols, p) in G1_CASES {
            let w = build(rows, cols, 0x1166 ^ rows as u64 ^ (cols as u64) << 8);
            let input = random_input(p * cols, 0xBEEF ^ p as u64);
            let want = cpu_reference(&w, &input, p);

            let cache = metal_weights(&gemm, &w);
            let input_buf = gemm.upload_f32(&input);
            let output_buf = gemm.create_output(p * rows);
            gemm.launch(
                &cache.cache.pos,
                &cache.cache.neg,
                &cache.cache.scale,
                &input_buf,
                &output_buf,
                cache.blocks64 as u32,
                cache.groups_per_row as u32,
                cache.n as u32,
                cache.m as u32,
                p as u32,
            );
            let got = gemm.read_f32(&output_buf, p * rows);

            let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
            for (i, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
                assert_ne!(
                    *g, SENTINEL,
                    "{name} rows={rows} cols={cols} p={p}: element {i} never written"
                );
                let rel = ((g - wv).abs() / scale) as f64;
                if rel > worst.0 {
                    worst = (rel, format!("rows={rows} cols={cols} p={p} idx={i}"));
                }
                if rel >= G1_REL_LIMIT {
                    failed = true;
                }
            }
        }
        println!(
            "  G1 {name:<20} {}  worst_rel={:.3e} at [{}]",
            if failed { "FAIL" } else { "PASS" },
            worst.0,
            worst.1
        );
        if failed {
            failed_configs.push(name);
        }
    }
    assert!(
        any_compiled,
        "every sweep config failed to compile — the instrument calibrated nothing"
    );
    assert!(
        failed_configs.is_empty(),
        "G1 FAILED for: {failed_configs:?} (limit {G1_REL_LIMIT}) — the timed sweep must not \
         run on a config whose output is wrong"
    );
}

// ---------------------------------------------------------------------------
// G2 — the interleaved geometry sweep, incumbent as the reference arm
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires macOS Metal + a GPU; run explicitly with --release"]
fn g2_ktile64_composed_geometry_sweep() {
    print_provenance("start");

    let shapes: Vec<(String, usize, usize)> = std::env::var("BENCH1166_SHAPES")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| {
                    let (m, n) = s.split_once('x').expect("shape as MxN");
                    (format!("{m}x{n}"), m.parse().unwrap(), n.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| {
            DEFAULT_SHAPES
                .iter()
                .map(|&(n, m, k)| (n.to_string(), m, k))
                .collect()
        });
    let ps: Vec<usize> = std::env::var("BENCH1166_PS")
        .ok()
        .map(|v| v.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or_else(|| DEFAULT_PS.to_vec());
    let rounds: usize = std::env::var("BENCH1166_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let launches: usize = std::env::var("BENCH1166_LAUNCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    // Compile every config up front; COMPILE_FAIL rows drop out of the sweep.
    let mut configs: Vec<(&str, MetalTensorGemm)> = Vec::new();
    for &(name, cfg) in SWEEP {
        match MetalTensorGemm::new_with_config(cfg) {
            Ok(g) => configs.push((name, g)),
            Err(e) => println!("  SWEEP {name:<20} COMPILE_FAIL — {e}"),
        }
    }
    println!(
        "\nG2 sweep: {} compiled config(s) · rounds={rounds} launches={launches} · reference = {}",
        configs.len(),
        SWEEP[0].0
    );

    for shape_ref in shapes.iter() {
        let (shape_name, m, n) = (shape_ref.0.as_str(), shape_ref.1, shape_ref.2);
        let w = build(m, n, 0x1166 ^ m as u64 ^ (n as u64) << 8);
        let caches: Vec<MetalArmWeights> =
            configs.iter().map(|(_, g)| metal_weights(g, &w)).collect();

        for &p in &ps {
            let input = random_input(p * n, 0xBEEF);

            // Warm every arm once (JIT/pool growth lands here, not in round 0).
            for ((_, gemm), cache) in configs.iter().zip(&caches) {
                launch_metal(gemm, cache, &input, p * m);
            }

            let n_arms = configs.len();
            let mut times = vec![vec![0.0f64; n_arms]; rounds];
            for (r, row) in times.iter_mut().enumerate().take(rounds) {
                for k in 0..n_arms {
                    let ai = (r + k) % n_arms; // rotating start arm
                    let (_, gemm) = &configs[ai];
                    let cache = &caches[ai];
                    let t0 = std::time::Instant::now();
                    for _ in 0..launches {
                        launch_metal(gemm, cache, &input, p * m);
                    }
                    let dt = t0.elapsed().as_secs_f64() / launches as f64;
                    assert!(dt > 0.0, "{shape_name} P={p} arm {}: zero-length timing — the loud-zero guard", configs[ai].0);
                    row[ai] = dt;
                }
            }

            let median = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            let base = median(times.iter().map(|row| row[0]).collect());
            println!(
                "\n── {shape_name} (m={m} n={n}) P={p} — incumbent {} {:.3} ms/launch ──",
                SWEEP[0].0,
                base * 1e3
            );
            for (ci, (name, _)) in configs.iter().enumerate() {
                let med = median(times.iter().map(|row| row[ci]).collect());
                println!(
                    "  {name:<20} {:>9.3} ms/launch   vs incumbent {:>6.3}×",
                    med * 1e3,
                    base / med
                );
            }
        }
    }

    print_provenance("end");
}
