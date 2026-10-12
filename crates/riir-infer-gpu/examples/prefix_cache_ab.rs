//! `prefix_cache_ab` — Issue 041 T3: the prefix-cache submit-burst
//! wall-time A/B at the real Bonsai-2 27B GDN state shape (48 recurrent
//! layers × 786,432 f32 = 3.0 MiB each + 48 conv layers × 40,960 f32 =
//! 160 KB each ≈ 158.9 MB per checkpoint).
//!
//! Arms: the shipping `memcpy_dtod` loop (96 driver submits per direction)
//! vs the issue-041 pointer-table copy (2 launches per direction), selected
//! PER OP through the production env gate (`QWEN38_PREFIX_TABLE_COPY`) so
//! both postures run in one process on one stream. Arm order ROTATES per
//! rep (plan-612 G2 pairing); verdict = median of same-rep PAIRED ratios
//! (table / dtod), never cross-window comparisons.
//!
//! What is timed (the serving shape): a WARM refresh-insert (state →
//! existing entry — the steady serving posture once the checkpoint exists)
//! and a RESTORE (entry → live state — the TTFT-critical path the issue
//! scopes FIRST: the only regime where the win is visible, a short-suffix
//! fill's ~10 ms budget). Each op is timed twice: enqueue-only (the host
//! driver-submit burst the serving loop's CPU pays) and sync-inclusive
//! (device completion). The FRESH insert (the T4 arena pool's surface) is
//! report-only context — a fresh cache pays 2 arena allocs cold (the
//! pre-T4 path paid 96 per-layer allocs every fresh insert); the table
//! arm's first cache additionally includes the lazy engine build (nvrtc
//! compile) and is disclosed as such.
//!
//! Restores target the live state object itself (the production shape: one
//! serving session's state) so the table arm's state-table freshness probe
//! stays warm — no per-op table htods, exactly as production.
//!
//! Box-state discipline: this binary prints config + results only — the
//! RUNNER quotes provenance in the bench record (Issue-021 law). Run
//! `--release` only.
//!
//! ```text
//! cargo run --release -p riir-infer-gpu --features ternary_gemv_cuda_raw \
//!   --example prefix_cache_ab [--reps 41] [--warmup 5]
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream};

use riir_infer_gpu::qwen38_dense_cudarc::Qwen38DecodeState;
use riir_infer_gpu::qwen38_prefix_cache::{
    Qwen38PrefixCache, submit_probe_counts, submit_probe_reset,
};

/// The Bonsai-2 27B GDN geometry (the `qwen38_dense_cudarc` config parse):
/// `n_v_heads 48 × head_k_dim 128 × head_v_dim 128` per recurrent layer;
/// `l_qkv_out (2×16×128 + 48×128 = 10240) × conv_kernel 4` per conv layer.
const REC_LAYERS: usize = 48;
const REC_LEN: usize = 48 * 128 * 128; // 786,432 f32 = 3.0 MiB
const CONV_LAYERS: usize = 48;
const CONV_LEN: usize = (2 * 16 * 128 + 48 * 128) * 4; // 40,960 f32 = 160 KB

const TOKENS: [u32; 8] = [11, 22, 33, 44, 55, 66, 77, 88];
const TABLE_ENV: &str = "QWEN38_PREFIX_TABLE_COPY";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Dtod,
    Table,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::Dtod => "dtod",
            Self::Table => "table",
        }
    }

    /// The production gate is consulted per call (`TableCopyMode::Env`
    /// re-reads the env on every insert/restore), so per-op toggling
    /// selects the path without process restarts. The example is
    /// single-threaded — the 2024-edition `unsafe` on `set_var` is
    /// sound here.
    fn set_env(self) {
        unsafe {
            match self {
                Self::Table => std::env::set_var(TABLE_ENV, "1"),
                Self::Dtod => std::env::remove_var(TABLE_ENV),
            }
        }
    }
}

/// LCG fill — deterministic, every element in [-1, 1) (the test module's
/// fill, at the 27B shape).
fn fill_class(
    stream: &Arc<CudaStream>,
    layers: usize,
    len: usize,
    seed: u64,
) -> Vec<CudaSlice<f32>> {
    let mut lcg = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::with_capacity(layers);
    for _ in 0..layers {
        let host: Vec<f32> = (0..len)
            .map(|_| {
                lcg = lcg
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((lcg >> 33) as f32) / (1u64 << 31) as f32 - 1.0
            })
            .collect();
        let mut dev = stream.alloc_zeros::<f32>(len).expect("alloc");
        stream.memcpy_htod(&host, &mut dev).expect("htod");
        out.push(dev);
    }
    out
}

fn make_state(stream: &Arc<CudaStream>, seed: u64) -> Qwen38DecodeState {
    Qwen38DecodeState {
        keys: Vec::new(),
        values: Vec::new(),
        recurrent: fill_class(stream, REC_LAYERS, REC_LEN, seed),
        conv: fill_class(stream, CONV_LAYERS, CONV_LEN, seed.wrapping_add(1)),
    }
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// One rep's four timings for one arm.
#[derive(Clone, Copy)]
struct Sample {
    enq_ins: Duration,
    sync_ins: Duration,
    enq_res: Duration,
    sync_res: Duration,
}

fn main() {
    let mut reps = 41usize;
    let mut warmup = 5usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reps" => reps = args.next().and_then(|v| v.parse().ok()).unwrap_or(reps),
            "--warmup" => warmup = args.next().and_then(|v| v.parse().ok()).unwrap_or(warmup),
            other => panic!("unknown arg {other} (expected --reps N / --warmup N)"),
        }
    }

    let ctx = CudaContext::new(0).unwrap_or_else(|e| panic!("CUDA context (the 4090 lane): {e}"));
    let stream = ctx.new_stream().expect("stream");

    let ckpt_mib = (REC_LAYERS * REC_LEN + CONV_LAYERS * CONV_LEN) * 4 / (1024 * 1024);
    println!("prefix_cache_ab — issue 041 T3 (submit-burst wall-time A/B)");
    println!(
        "shape: rec {REC_LAYERS}×{REC_LEN} f32 + conv {CONV_LAYERS}×{CONV_LEN} f32 \
         = {ckpt_mib} MiB per checkpoint"
    );
    println!("config: reps={reps} warmup={warmup} (order-rotated paired reps)");

    let mut state = make_state(&stream, 0xC0FF_EE01);
    let mut c_dtod = Qwen38PrefixCache::new(1);
    let mut c_tab = Qwen38PrefixCache::new(1);

    // ── warmup: engine build + fresh insert + refresh + restore per arm ──
    for arm in [Arm::Dtod, Arm::Table] {
        arm.set_env();
        let c = match arm {
            Arm::Dtod => &mut c_dtod,
            Arm::Table => &mut c_tab,
        };
        c.insert(&stream, &state, &TOKENS)
            .expect("warm fresh insert");
        for _ in 0..warmup {
            c.insert(&stream, &state, &TOKENS).expect("warm refresh");
            let matched = c
                .restore(&stream, &mut state, &TOKENS)
                .expect("warm restore");
            assert_eq!(matched, Some(TOKENS.len()));
        }
    }
    stream.synchronize().expect("warmup sync");

    // ── submit-count probe: one warm refresh + restore per arm ──
    submit_probe_reset();
    for arm in [Arm::Dtod, Arm::Table] {
        arm.set_env();
        let c = match arm {
            Arm::Dtod => &mut c_dtod,
            Arm::Table => &mut c_tab,
        };
        c.insert(&stream, &state, &TOKENS).expect("probe refresh");
        c.restore(&stream, &mut state, &TOKENS)
            .expect("probe restore");
    }
    let counts = submit_probe_counts();
    println!(
        "submit probe (warm refresh+restore, both arms): \
         dtod_memcpies={} snap_allocs={} table_launches={} table_htods={} \
         (expected [96+96, 0, 2+2, 0])",
        counts[0], counts[1], counts[2], counts[3]
    );

    // ── fresh insert, report-only (T4's surface; table arm pays the lazy
    //    engine build — nvrtc compile — inside this number, disclosed) ──
    for arm in [Arm::Dtod, Arm::Table] {
        arm.set_env();
        submit_probe_reset();
        let mut fresh = Qwen38PrefixCache::new(1);
        let t = Instant::now();
        fresh
            .insert(&stream, &state, &TOKENS)
            .expect("fresh insert");
        let wall = t.elapsed();
        stream.synchronize().expect("fresh sync");
        let c = submit_probe_counts();
        println!(
            "fresh insert ({}) — wall {:?} | dtod={} allocs={} launches={} htods={} \
             (T4: 2 arena allocs cold / 0 pool hit; the table arm's engine build rides this number)",
            arm.label(),
            wall,
            c[0],
            c[1],
            c[2],
            c[3]
        );
    }

    // ── the timed paired phase ──
    let mut s_dtod: Vec<Sample> = Vec::with_capacity(reps);
    let mut s_tab: Vec<Sample> = Vec::with_capacity(reps);
    for rep in 0..reps {
        let order: [Arm; 2] = if rep % 2 == 0 {
            [Arm::Dtod, Arm::Table]
        } else {
            [Arm::Table, Arm::Dtod]
        };
        for arm in order {
            arm.set_env();
            let c = match arm {
                Arm::Dtod => &mut c_dtod,
                Arm::Table => &mut c_tab,
            };
            let t0 = Instant::now();
            c.insert(&stream, &state, &TOKENS)
                .expect("timed refresh insert");
            let enq_ins = t0.elapsed();
            stream.synchronize().expect("timed sync ins");
            let sync_ins = t0.elapsed();

            let t1 = Instant::now();
            let matched = c
                .restore(&stream, &mut state, &TOKENS)
                .expect("timed restore");
            let enq_res = t1.elapsed();
            stream.synchronize().expect("timed sync res");
            let sync_res = t1.elapsed();
            assert_eq!(matched, Some(TOKENS.len()));

            let s = Sample {
                enq_ins,
                sync_ins,
                enq_res,
                sync_res,
            };
            match arm {
                Arm::Dtod => s_dtod.push(s),
                Arm::Table => s_tab.push(s),
            }
        }
    }

    // ── verdict: per-arm medians + same-rep paired ratios ──
    let med = |get: fn(&Sample) -> Duration, v: &[Sample]| -> Duration {
        let mut vals: Vec<Duration> = v.iter().map(get).collect();
        median(&mut vals)
    };
    let us = |d: Duration| d.as_nanos() as f64 / 1000.0;
    let m_dtod_ins = med(|s| s.enq_ins, &mut s_dtod);
    let m_dtod_sync_ins = med(|s| s.sync_ins, &mut s_dtod);
    let m_dtod_res = med(|s| s.enq_res, &mut s_dtod);
    let m_dtod_sync_res = med(|s| s.sync_res, &mut s_dtod);
    let m_tab_ins = med(|s| s.enq_ins, &mut s_tab);
    let m_tab_sync_ins = med(|s| s.sync_ins, &mut s_tab);
    let m_tab_res = med(|s| s.enq_res, &mut s_tab);
    let m_tab_sync_res = med(|s| s.sync_res, &mut s_tab);

    println!("\narm   | enq_ins µs | sync_ins µs | enq_res µs | sync_res µs");
    println!(
        "dtod  | {:10.1} | {:11.1} | {:10.1} | {:11.1}",
        us(m_dtod_ins),
        us(m_dtod_sync_ins),
        us(m_dtod_res),
        us(m_dtod_sync_res)
    );
    println!(
        "table | {:10.1} | {:11.1} | {:10.1} | {:11.1}",
        us(m_tab_ins),
        us(m_tab_sync_ins),
        us(m_tab_res),
        us(m_tab_sync_res)
    );

    // Same-rep paired ratios (rep i pairs across arms — order-rotated).
    let paired = |get: fn(&Sample) -> Duration| -> f64 {
        let mut rs: Vec<f64> = (0..reps)
            .map(|i| us(get(&s_tab[i])) / us(get(&s_dtod[i])).max(1.0))
            .collect();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rs[rs.len() / 2]
    };
    println!(
        "\npaired median table/dtod: enq_ins {:.3}× | sync_ins {:.3}× | enq_res {:.3}× | sync_res {:.3}×",
        paired(|s| s.enq_ins),
        paired(|s| s.sync_ins),
        paired(|s| s.enq_res),
        paired(|s| s.sync_res)
    );
    println!(
        "context: one direction moves {ckpt_mib} MiB; short-suffix TTFT budget ~10 ms \
         → table sync_res = {:.2}% of budget (dtod {:.2}%)",
        us(m_tab_sync_res) / 10_000.0,
        us(m_dtod_sync_res) / 10_000.0
    );
}
