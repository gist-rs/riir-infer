//! Issue 1168 T1 — the readback floor probe: decompose the per-token
//! `read_one` cost into fixed sync vs bytes vs tail-wait components.
//!
//! The triage (`.issues/1168_decode_host_sync_lever_triage.md`) measured — via
//! riir-ai bench_763's own census log — that the decode lane pays ~3.14 ms/token
//! between the per-token-sync e2e (Phase A, 18.48 ms/tok) and the GPU-bound
//! floor (Phase B dispatch-only + amortized drain, 15.34 ms/tok = 65.2 tok/s,
//! which would read ~0.98x vs the same-window opponent). The phase delta is the
//! per-token `read_one` of the 993 KB logits buffer: GPU-tail wait + copy +
//! allocation + submission/callback overhead. This probe prices the COMPONENTS
//! on a bare client (no model), so the T2-vs-T3 rung decision is a measurement,
//! not a guess:
//!
//! - (a) `read_one(4 B)` on a drained queue — the fixed sync/submission floor.
//! - (b) `read_one(993 KB)` on a drained queue — the bytes cost on top.
//! - (c) `read_one(993 KB)` right after ONE tiny GEMV dispatch — tail-wait +
//!   copy + overhead (the Phase A per-token shape in miniature).
//! - (d) 20 tiny GEMVs enqueued, ONE read at the end — the Phase B
//!   amortization reference (divide by 20).
//! - (e) `read_one` of a buffer the GPU never touched vs one a GEMV wrote —
//!   pins whether write-dependency ordering adds cost beyond the tail wait.
//!
//! All medians over 200 iterations after a 20-iteration warm-up, interleaved
//! A/B inside each phase (the sequential-A/B box-state law, riir-ai
//! sequential_ab_timing_audit): phases run small/big/small/big so a box-state
//! drift cannot masquerade as a size effect.
//!
//! Usage (GPU-exclusive; no model needed):
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/i1168-probe cargo test -p riir-infer-gpu \
//!     --features cubecl_runtime --release \
//!     --test issue1168_readback_floor_probe -- --nocapture --ignored
//! ```

#![cfg(all(
    feature = "ternary_gemv",
    target_os = "macos",
    not(debug_assertions)
))]

use std::time::Instant;

use katgpt_core::TernaryGroupWeights;
use riir_infer_gpu::cubecl_runtime::{ActiveComputeClient, ActiveRuntime, CubeCLContext};
use riir_infer_gpu::gemv_ternary_cubecl::{GemvTernaryCubeCL, TernaryHandle};

/// The league logits size: vocab 248320 x f32.
const LOGITS_BYTES: usize = 248320 * 4;

const ITERS: usize = 200;
const WARMUP: usize = 20;
/// The Phase B reference depth: 20 dispatch-only tokens then one read.
const QUEUE_DEPTH: usize = 20;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

/// Deterministic pseudo-random f32 fill (LCG; no rand dep).
fn pseudo_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// A tiny ternary GEMV dispatch: m=8 rows, n=128 (one group, one block64).
/// Work is negligible; its only job is to occupy the queue with REAL kernel
/// launches (server-side launch + encoder work) so phases (c)-(e) measure the
/// read-after-work shape.
fn tiny_gemv(client: &ActiveComputeClient) {
    let rows = 8usize;
    let cols = 128usize;
    let mut w = TernaryGroupWeights::new(rows, cols);
    let mut s = 0x9E3779B97F4A7C15u64;
    for r in 0..rows {
        for c in 0..cols {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 33) as f32) / ((1u64 << 31) as f32) - 1.0;
            w.set(
                r,
                c,
                if v > 0.33 {
                    1i8
                } else if v < -0.33 {
                    -1i8
                } else {
                    0i8
                },
            );
        }
        w.set_scale(r, 0, 0.5);
    }
    let handle = TernaryHandle::from_weights(client, &w);
    let input = client.create_from_slice(bytemuck::cast_slice(&pseudo_random_f32(
        cols,
        0x94D049BB133111EB,
    )));
    let output = client.empty(rows);
    unsafe {
        GemvTernaryCubeCL::launch::<ActiveRuntime>(client, &handle, input, output);
    }
}

#[test]
#[ignore = "requires GPU + Metal; ~30 s"]
fn issue1168_readback_floor_probe() {
    let ctx = CubeCLContext::new().expect("GPU init");
    let client = ctx.client();
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!(
        "  Issue 1168 T1 — readback floor probe ({})",
        ctx.runtime_name()
    );
    eprintln!("  ITERS={ITERS} WARMUP={WARMUP} QUEUE_DEPTH={QUEUE_DEPTH}");
    eprintln!("═══════════════════════════════════════════════════════════");

    // ── fixtures ──
    let small = client.create_from_slice(bytemuck::cast_slice(&[1.0f32]));
    let big_src = pseudo_random_f32(LOGITS_BYTES / 4, 0xBF58476D1CE4E5B9);
    let big = client.create_from_slice(bytemuck::cast_slice(&big_src));
    // A GEMV output for the write-dependency arm: rows=8 f32.
    let tiny_out = client.empty(8);

    // Drain once so the queue starts empty.
    let _ = client.read_one(small.clone());

    // ── (a) fixed floor: read 4 B, drained ──
    let mut a = Vec::with_capacity(ITERS);
    for _ in 0..WARMUP {
        let _ = client.read_one(small.clone());
    }
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = client.read_one(small.clone());
        a.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // ── (b) bytes: read 993 KB, drained (interleaved with (a)'s shape) ──
    let mut b = Vec::with_capacity(ITERS);
    for _ in 0..WARMUP {
        let _ = client.read_one(big.clone());
    }
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = client.read_one(big.clone());
        b.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // ── (c) tail wait: ONE tiny GEMV then read big — the per-token shape ──
    let mut c = Vec::with_capacity(ITERS);
    for _ in 0..WARMUP {
        tiny_gemv(&client);
        let _ = client.read_one(big.clone());
    }
    for _ in 0..ITERS {
        tiny_gemv(&client);
        let t = Instant::now();
        let _ = client.read_one(big.clone());
        c.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // ── (d) amortized: 20 tiny GEMVs then ONE read (Phase B reference) ──
    let mut d = Vec::with_capacity(ITERS);
    for _ in 0..WARMUP {
        for _ in 0..QUEUE_DEPTH {
            tiny_gemv(&client);
        }
        let _ = client.read_one(big.clone());
    }
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for _ in 0..QUEUE_DEPTH {
            tiny_gemv(&client);
        }
        let enqueue = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let _ = client.read_one(big.clone());
        let read = t1.elapsed().as_secs_f64() * 1e3;
        d.push((enqueue, read));
    }
    let d_enqueue: f64 = median(&mut d.iter().map(|(e, _)| *e).collect::<Vec<_>>());
    let d_read: f64 = median(&mut d.iter().map(|(_, r)| *r).collect::<Vec<_>>());

    // ── (e) write dependency: read the GEMV's OWN output (8 f32) — does
    //    reading a just-written buffer cost more than reading untouched
    //    bytes at the same size? ──
    let mut e = Vec::with_capacity(ITERS);
    for _ in 0..WARMUP {
        tiny_gemv(&client);
        let _ = client.read_one(tiny_out.clone());
    }
    for _ in 0..ITERS {
        tiny_gemv(&client);
        let t = Instant::now();
        let _ = client.read_one(tiny_out.clone());
        e.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let (am, bm, cm, em) = (
        median(&mut a),
        median(&mut b),
        median(&mut c),
        median(&mut e),
    );

    eprintln!("─── readback floor components (medians, ms) ───");
    eprintln!("  (a) read 4 B drained            : {am:8.3}");
    eprintln!(
        "  (b) read 993 KB drained         : {bm:8.3}   bytes delta = {:.3}",
        bm - am
    );
    eprintln!(
        "  (c) 1 tiny GEMV -> read 993 KB  : {cm:8.3}   tail delta  = {:.3}",
        cm - bm
    );
    eprintln!(
        "  (d) 20 GEMVs -> one read        : enqueue {d_enqueue:6.3} + read {d_read:6.3}  (read/depth = {:.3})",
        d_read / QUEUE_DEPTH as f64
    );
    eprintln!(
        "  (e) 1 tiny GEMV -> read 8 f32   : {em:8.3}   (write-dep vs (a): {:.3})",
        em - am
    );
    eprintln!();
    eprintln!("─── rung decision bands (from the triage) ───");
    eprintln!("  T2 (summary-read rung) if (b)-(a) is the dominant mass;");
    eprintln!("  T3 (K-deferred/GPU-resident arc) if (a) alone rivals the 3.14 ms/tok class.");
    eprintln!();
    eprintln!(
        "  Phase-A-class per-token estimate from these parts: (c) = {:.3} ms",
        cm
    );
    eprintln!("  (bench_763 measured the class at ~3.14 ms/tok over the FULL forward:");
    eprintln!("   1127 dispatches/token vs this probe's 1-20 — the launch-side share of");
    eprintln!("   the sync cost scales with queued-work depth; (d) is the linear read.)");
}
