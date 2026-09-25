//! sgemm MMA-roofline probe (measurement only, never a gate) — the f16-axis
//! discriminator for the T7 follow-up rungs (reflex issue 020): is the
//! narrow instance MMA-issue-bound or staging/bandwidth-bound?
//!
//! Three arms over the SAME tile geometry (BM=32, BN=64, BK=64, 512
//! threads, the shipped narrow instance's exact shape):
//! 1. `sgemm` — a verbatim copy of the shipped narrow kernel (source:
//!    `src/laya/riir/metal.rs` `MSL_SGEMM_NARROW` at riir-infer `9a4e17a`;
//!    drift-checked against the CPU triple loop on the ragged-edge cell, so
//!    a copy drift fails loud).
//! 2. `sgemm_mma_roofline` — the SAME inner k-chunk (same three
//!    simdgroup_loads + the same two MMAs), staged ONCE before the loop,
//!    then `reps` iterations with NO re-staging and NO barriers. FLOPs are
//!    matched to arm 1 by reps = k/BK. Whether the invariant threadgroup
//!    loads hoist out of the rep loop or not, the number is the empirical
//!    ceiling for everything after staging — the gap to arm 1 is the
//!    staging+barrier share.
//!
//! Verdict grammar: roofline ≈ shipped ⇒ MMA-issue-bound ⇒ the f16 lever is
//! the MMA itself (f16 simdgroup MMA, the big numeric rung); roofline ≫
//! shipped ⇒ staging/bandwidth-bound ⇒ the f16 lever is B-operand bytes
//! (f16 staging/device reads, the smaller numeric rung).
//!
//! Position-balanced: each cell runs R rounds, alternating which arm is
//! encoded first (the cold-GPU sequencing trap, `.issues/015`); per round
//! each arm encodes 3 dispatches back-to-back in ONE command buffer and
//! waits once (the forward's own pipelined posture — per-op commit+wait
//! measures submission overhead, not the kernel).
//!
//! Run:
//!   cargo run --release -p riir-infer-laya --features laya-riir-metal \
//!       --example sgemm_roofline

#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use objc2::rc::autoreleasepool;
use std::time::Instant;

const RESOURCE_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::HazardTrackingModeTracked);

const MSL_HEAD: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"#;

/// Verbatim copy of the shipped narrow instance (riir-infer `9a4e17a`,
/// `MSL_SGEMM_NARROW`) — the CPU divergence check on cell 1 is what keeps
/// this copy honest.
const MSL_NARROW_COPY: &str = r#"
constant uint BM = 32u;
constant uint BN = 64u;
constant uint BK = 64u;
constant uint TAS = 65u;
constant uint TBS = 65u;

kernel void sgemm(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_rs [[buffer(6)]],
    constant uint& a_cs [[buffer(7)]],
    constant uint& b_rs [[buffer(8)]],
    constant uint& b_cs [[buffer(9)]],
    constant uint& a_bs [[buffer(10)]],
    constant uint& b_bs [[buffer(11)]],
    constant uint& c_bs [[buffer(12)]],
    threadgroup float* raw [[threadgroup(13)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + 32u * TAS;
    threadgroup float* edge = raw;
    const uint m0 = gtp.y * BM;
    const uint n0 = gtp.x * BN;
    device const float* A = a + gtp.z * a_bs;
    device const float* B = b + gtp.z * b_bs;
    device float* C = out + gtp.z * c_bs;

    const uint sg = lid >> 5u;
    const uint lane = lid & 31u;
    const uint sgr = sg >> 2u;
    const uint sgc = sg & 3u;

    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);

    for (uint t = 0u; t < k; t += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 4u; ++q) {
            const uint idx = lid + q * 512u;
            const uint r = idx >> 6u;
            const uint c = idx & 63u;
            const uint gr = m0 + r;
            const uint ac = t + c;
            ta[r * TAS + c] = (gr < m && ac < k) ? A[gr * a_rs + ac * a_cs] : 0.0f;
        }
        for (uint q = 0u; q < 8u; ++q) {
            const uint idx = lid + q * 512u;
            const uint kk = idx >> 6u;
            const uint col = idx & 63u;
            const uint bc = t + kk;
            if (b_cs == 1u) {
                tb[kk * TBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + n0 + col] : 0.0f;
            } else {
                tb[kk * TBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + (n0 + col) * b_cs] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_float8x8 fa, fb0, fb1;
            simdgroup_load(fa, ta + sgr * 8u * TAS + kk, TAS);
            simdgroup_load(fb0, tb + kk * TBS + sgc * 8u, TBS);
            simdgroup_load(fb1, tb + kk * TBS + (sgc + 4u) * 8u, TBS);
            simdgroup_multiply_accumulate(acc0, fa, fb0, acc0);
            simdgroup_multiply_accumulate(acc1, fa, fb1, acc1);
        }
    }

    if ((m0 + BM <= m) && (n0 + BN <= n)) {
        simdgroup_store(acc0, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u), n);
        simdgroup_store(acc1, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u + 4u * 8u), n);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, edge + sg * 128u, 8u);
        simdgroup_store(acc1, edge + sg * 128u + 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 2u; ++q) {
            const uint e = lane + q * 32u;
            const uint er = e >> 3u;
            const uint ec = e & 7u;
            const uint gr = m0 + sgr * 8u + er;
            const uint gc0 = n0 + sgc * 8u + ec;
            const uint gc1 = gc0 + 4u * 8u;
            if (gr < m && gc0 < n) { C[gr * n + gc0] = edge[sg * 128u + e]; }
            if (gr < m && gc1 < n) { C[gr * n + gc1] = edge[sg * 128u + 64u + e]; }
        }
    }
}
"#;

/// The MMA-only twin: same inner k-chunk, staged once, `reps` iterations of
/// the 8-wide chunk with no device traffic and no barriers. Buffers are
/// allocated tile-padded (no edge path); `n` is only needed for the store.
const MSL_ROOFLINE: &str = r#"
constant uint RBM = 32u;
constant uint RBN = 64u;
constant uint RBK = 64u;
constant uint RTAS = 65u;
constant uint RTBS = 65u;

kernel void sgemm_mma_roofline(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& reps [[buffer(4)]],
    threadgroup float* raw [[threadgroup(5)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + RBM * RTAS;
    const uint m0 = gtp.y * RBM;
    const uint n0 = gtp.x * RBN;

    for (uint q = 0u; q < 4u; ++q) {
        const uint idx = lid + q * 512u;
        const uint r = idx >> 6u;
        const uint c = idx & 63u;
        ta[r * RTAS + c] = a[(m0 + r) * 64u + c];
    }
    for (uint q = 0u; q < 8u; ++q) {
        const uint idx = lid + q * 512u;
        const uint kk = idx >> 6u;
        const uint col = idx & 63u;
        tb[kk * RTBS + col] = b[kk * 64u + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = lid >> 5u;
    const uint sgr = sg >> 2u;
    const uint sgc = sg & 3u;
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);
    for (uint rep = 0u; rep < reps; ++rep) {
        for (uint kk = 0u; kk < RBK; kk += 8u) {
            simdgroup_float8x8 fa, fb0, fb1;
            simdgroup_load(fa, ta + sgr * 8u * RTAS + kk, RTAS);
            simdgroup_load(fb0, tb + kk * RTBS + sgc * 8u, RTBS);
            simdgroup_load(fb1, tb + kk * RTBS + (sgc + 4u) * 8u, RTBS);
            simdgroup_multiply_accumulate(acc0, fa, fb0, acc0);
            simdgroup_multiply_accumulate(acc1, fa, fb1, acc1);
        }
    }
    simdgroup_store(acc0, out + (m0 + sgr * 8u) * n + (n0 + sgc * 8u), n);
    simdgroup_store(acc1, out + (m0 + sgr * 8u) * n + (n0 + sgc * 8u + 4u * 8u), n);
}
"#;

/// The f16-B twin: the narrow instance with the B operand resident as f16
/// (element units — b_rs/b_cs index the half array), converted to f32 at
/// staging. Everything after the staging (tile layout, barriers, MMA) is
/// bit-identical to arm 1; only the device-side B bytes halve. This is the
/// probe arm for the f16-B rung the roofline verdict quantified: if this
/// wins ≥ 1.3× the backend rung (f16 transposed weight cache + dispatch +
/// Issue-750-T3 retention walk) is justified.
const MSL_HB: &str = r#"
kernel void sgemm_hb(
    device const float* a [[buffer(0)]],
    device const half* bh [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_rs [[buffer(6)]],
    constant uint& a_cs [[buffer(7)]],
    constant uint& b_rs [[buffer(8)]],
    constant uint& b_cs [[buffer(9)]],
    constant uint& a_bs [[buffer(10)]],
    constant uint& b_bs [[buffer(11)]],
    constant uint& c_bs [[buffer(12)]],
    threadgroup float* raw [[threadgroup(13)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + 32u * TAS;
    threadgroup float* edge = raw;
    const uint m0 = gtp.y * BM;
    const uint n0 = gtp.x * BN;
    device const float* A = a + gtp.z * a_bs;
    device const half* B = bh + gtp.z * b_bs;
    device float* C = out + gtp.z * c_bs;

    const uint sg = lid >> 5u;
    const uint lane = lid & 31u;
    const uint sgr = sg >> 2u;
    const uint sgc = sg & 3u;

    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);

    for (uint t = 0u; t < k; t += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 4u; ++q) {
            const uint idx = lid + q * 512u;
            const uint r = idx >> 6u;
            const uint c = idx & 63u;
            const uint gr = m0 + r;
            const uint ac = t + c;
            ta[r * TAS + c] = (gr < m && ac < k) ? A[gr * a_rs + ac * a_cs] : 0.0f;
        }
        for (uint q = 0u; q < 8u; ++q) {
            const uint idx = lid + q * 512u;
            const uint kk = idx >> 6u;
            const uint col = idx & 63u;
            const uint bc = t + kk;
            if (b_cs == 1u) {
                tb[kk * TBS + col] =
                    (bc < k && n0 + col < n) ? float(B[bc * b_rs + n0 + col]) : 0.0f;
            } else {
                tb[kk * TBS + col] =
                    (bc < k && n0 + col < n) ? float(B[bc * b_rs + (n0 + col) * b_cs]) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_float8x8 fa, fb0, fb1;
            simdgroup_load(fa, ta + sgr * 8u * TAS + kk, TAS);
            simdgroup_load(fb0, tb + kk * TBS + sgc * 8u, TBS);
            simdgroup_load(fb1, tb + kk * TBS + (sgc + 4u) * 8u, TBS);
            simdgroup_multiply_accumulate(acc0, fa, fb0, acc0);
            simdgroup_multiply_accumulate(acc1, fa, fb1, acc1);
        }
    }

    if ((m0 + BM <= m) && (n0 + BN <= n)) {
        simdgroup_store(acc0, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u), n);
        simdgroup_store(acc1, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u + 4u * 8u), n);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, edge + sg * 128u, 8u);
        simdgroup_store(acc1, edge + sg * 128u + 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 2u; ++q) {
            const uint e = lane + q * 32u;
            const uint er = e >> 3u;
            const uint ec = e & 7u;
            const uint gr = m0 + sgr * 8u + er;
            const uint gc0 = n0 + sgc * 8u + ec;
            const uint gc1 = gc0 + 4u * 8u;
            if (gr < m && gc0 < n) { C[gr * n + gc0] = edge[sg * 128u + e]; }
            if (gr < m && gc1 < n) { C[gr * n + gc1] = edge[sg * 128u + 64u + e]; }
        }
    }
}
"#;

/// (m, k, n, label) — the shipped kernel's real geometries; the roofline
/// twin runs reps = k/64 at the same grid. Cell 1 is the CPU-verified one
/// (ragged m exercises the edge path); cells 2-3 are exact-tile.
const CELLS: &[(usize, usize, usize, &str)] = &[
    (317, 3072, 1024, "encoder QKV @ m317 (ragged-edge, CPU-verified)"),
    (1024, 3072, 1024, "packed scale @ m1024"),
    (1024, 8192, 1024, "deep-k @ m1024 (staging share stressed)"),
];

const ROUNDS: usize = 10;
const DISPATCHES_PER_ROUND: usize = 3;
const NARROW_STAGING_BYTES: u64 = 24_960;

/// f32 → f16 round-to-nearest-even (the conversion Metal's own
/// `float<->half` casts use, so the probe's seed matches the kernel's view).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let man = bits & 0x007f_ffff;
    if exp > 15 {
        return sign | 0x7c00; // inf (no NaN seeds in this probe)
    }
    if exp >= -14 {
        // normal: round the mantissa to 10 bits with RNE
        let shift = 13;
        let mut half_man = man >> shift;
        let rem = man & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && (half_man & 1) == 1) {
            half_man += 1;
        }
        let mut e = exp + 15;
        let mut hm = half_man;
        if hm == 0x400 {
            hm = 0;
            e += 1;
        }
        if e >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((e as u16) << 10) | hm as u16;
    }
    // subnormal / zero
    if exp < -25 {
        return sign;
    }
    let man = man | 0x0080_0000;
    let shift = (-exp - 14 + 13) as u32;
    let mut hm = man >> shift;
    let rem = man & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    if rem > halfway || (rem == halfway && (hm & 1) == 1) {
        hm += 1;
    }
    sign | hm as u16
}

fn median(samples: &mut [u128]) -> f64 {
    samples.sort();
    samples[samples.len() / 2] as f64 / 1000.0 // ns → µs
}

fn load_avg() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime failed".into())
}

fn main() {
    println!("box: {}", load_avg());
    let Some(device) = Device::system_default() else {
        panic!("no Metal device");
    };
    let queue = device.new_command_queue();
    let msl = format!("{MSL_HEAD}{MSL_NARROW_COPY}{MSL_ROOFLINE}{MSL_HB}");
    let lib = device
        .new_library_with_source(&msl, &CompileOptions::new())
        .expect("MSL compile");
    let mk_pipe = |name: &str| {
        let f = lib.get_function(name, None).expect(name);
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline")
    };
    let narrow_pipe = mk_pipe("sgemm");
    let roof_pipe = mk_pipe("sgemm_mma_roofline");
    let hb_pipe = mk_pipe("sgemm_hb");

    for &(m, k, n, label) in CELLS {
        let grid = MTLSize {
            width: (n as u64).div_ceil(64),
            height: (m as u64).div_ceil(32),
            depth: 1,
        };
        let tpg = MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        };
        // Tile-padded FLOPs (both arms compute the padded grid).
        let pm = grid.height * 32;
        let pn = grid.width * 64;
        let flops = (2 * pm * pn * k as u64) as f64;

        // ── arm 1: shipped narrow (copy) ──
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect();
        // Row-major A [m][k] (a_rs=k, a_cs=1) and row-major B [k][n]
        // (b_rs=n, b_cs=1 — the kernel's coalesced special case).
        let w: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
        let mut got = vec![0f32; m * n];
        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * w[kk * n + j];
                }
                want[i * n + j] = s;
            }
        }
        let a_buf = device.new_buffer_with_data(
            a.as_ptr().cast(),
            (a.len() * 4) as u64,
            RESOURCE_OPTIONS,
        );
        let w_buf = device.new_buffer_with_data(
            w.as_ptr().cast(),
            (w.len() * 4) as u64,
            RESOURCE_OPTIONS,
        );
        let out_buf = device.new_buffer((got.len() * 4) as u64, RESOURCE_OPTIONS);
        {
            let cb = queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder().to_owned();
            enc.set_compute_pipeline_state(&narrow_pipe);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            for (i, v) in [m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0, 0]
                .iter()
                .enumerate()
            {
                enc.set_bytes(i as u64 + 3, 4, std::ptr::from_ref(v).cast());
            }
            enc.set_threadgroup_memory_length(13, NARROW_STAGING_BYTES);
            enc.dispatch_thread_groups(grid, tpg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let size = got.len() * 4;
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents().cast::<u8>(),
                got.as_mut_ptr().cast::<u8>(),
                size,
            );
        }
        let max_want = want.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let max_diff = want
            .iter()
            .zip(got.iter())
            .map(|(c, d)| (c - d).abs())
            .fold(0.0f32, f32::max);
        let rel = max_diff / max_want.max(1e-30);
        if rel > 1e-3 {
            eprintln!(
                "DEBUG {m}x{k}x{n}: rel {rel:e} max_diff {max_diff:e} max_want {max_want:e}\n  got[:4]  = {:?}\n  want[:4] = {:?}\n  got[470..474] = {:?}",
                &got[..4],
                &want[..4],
                &got[470..474],
            );
            panic!("cell {m}x{k}x{n}: narrow COPY diverged rel {rel:e}");
        }
        let bit = max_diff == 0.0;

        // ── f16-B arm: same weights as u16 (RNE), verify against narrow ──
        let w_h: Vec<u16> = w.iter().copied().map(f32_to_f16).collect();
        let wh_buf = device.new_buffer_with_data(
            w_h.as_ptr().cast(),
            (w_h.len() * 2) as u64,
            RESOURCE_OPTIONS,
        );
        let mut got_h = vec![0f32; m * n];
        {
            let cb = queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder().to_owned();
            enc.set_compute_pipeline_state(&hb_pipe);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&wh_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            for (i, v) in [m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0, 0]
                .iter()
                .enumerate()
            {
                enc.set_bytes(i as u64 + 3, 4, std::ptr::from_ref(v).cast());
            }
            enc.set_threadgroup_memory_length(13, NARROW_STAGING_BYTES);
            enc.dispatch_thread_groups(grid, tpg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents().cast::<u8>(),
                got_h.as_mut_ptr().cast::<u8>(),
                size,
            );
        }
        let hb_diff = got
            .iter()
            .zip(got_h.iter())
            .map(|(c, d)| (c - d).abs())
            .fold(0.0f32, f32::max);
        let hb_rel = hb_diff / max_want.max(1e-30);
        if hb_rel > 2e-2 {
            panic!(
                "cell {m}x{k}x{n}: f16-B arm diverged from narrow rel {hb_rel:e}"
            );
        }

        // ── roofline buffers (padded, seeded) ──
        let ra: Vec<f32> = (0..pm * 64).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect();
        let rb: Vec<f32> = (0..64 * 64).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
        let ra_buf = device.new_buffer_with_data(
            ra.as_ptr().cast(),
            (ra.len() * 4) as u64,
            RESOURCE_OPTIONS,
        );
        let rb_buf = device.new_buffer_with_data(
            rb.as_ptr().cast(),
            (rb.len() * 4) as u64,
            RESOURCE_OPTIONS,
        );
        let rout_buf = device.new_buffer(pm * pn * 4, RESOURCE_OPTIONS);
        let reps = (k / 64) as u32;
        let pn32 = pn as u32;

        // ── position-rotated timed rounds: 3 arms (narrow / hb / roofline) ──
        const ARMS: usize = 3;
        let mut t_narrow: Vec<u128> = Vec::with_capacity(ROUNDS);
        let mut t_hb: Vec<u128> = Vec::with_capacity(ROUNDS);
        let mut t_roof: Vec<u128> = Vec::with_capacity(ROUNDS);
        for r in 0..ROUNDS {
            for pos in 0..ARMS {
                // rotation: which arm goes first moves each round
                let arm = (pos + r) % ARMS;
                let t_n = &mut t_narrow;
                let t_h = &mut t_hb;
                let t_r = &mut t_roof;
                autoreleasepool(|_pool| {
                    let cb = queue.new_command_buffer().to_owned();
                    let enc = cb.new_compute_command_encoder().to_owned();
                    let t0 = Instant::now();
                    for _ in 0..DISPATCHES_PER_ROUND {
                        match arm {
                            0 => {
                                enc.set_compute_pipeline_state(&narrow_pipe);
                                enc.set_buffer(0, Some(&a_buf), 0);
                                enc.set_buffer(1, Some(&w_buf), 0);
                                enc.set_buffer(2, Some(&out_buf), 0);
                                for (i, v) in
                                    [m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0, 0]
                                        .iter()
                                        .enumerate()
                                {
                                    enc.set_bytes(i as u64 + 3, 4, std::ptr::from_ref(v).cast());
                                }
                                enc.set_threadgroup_memory_length(13, NARROW_STAGING_BYTES);
                            }
                            1 => {
                                enc.set_compute_pipeline_state(&hb_pipe);
                                enc.set_buffer(0, Some(&a_buf), 0);
                                enc.set_buffer(1, Some(&wh_buf), 0);
                                enc.set_buffer(2, Some(&out_buf), 0);
                                for (i, v) in
                                    [m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0, 0]
                                        .iter()
                                        .enumerate()
                                {
                                    enc.set_bytes(i as u64 + 3, 4, std::ptr::from_ref(v).cast());
                                }
                                enc.set_threadgroup_memory_length(13, NARROW_STAGING_BYTES);
                            }
                            _ => {
                                enc.set_compute_pipeline_state(&roof_pipe);
                                enc.set_buffer(0, Some(&ra_buf), 0);
                                enc.set_buffer(1, Some(&rb_buf), 0);
                                enc.set_buffer(2, Some(&rout_buf), 0);
                                for (i, v) in [pn32, reps].iter().enumerate() {
                                    enc.set_bytes(i as u64 + 3, 4, std::ptr::from_ref(v).cast());
                                }
                                enc.set_threadgroup_memory_length(5, NARROW_STAGING_BYTES);
                            }
                        }
                        enc.dispatch_thread_groups(grid, tpg);
                    }
                    enc.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    let dt = t0.elapsed().as_nanos() / DISPATCHES_PER_ROUND as u128;
                    match arm {
                        0 => t_n.push(dt),
                        1 => t_h.push(dt),
                        _ => t_r.push(dt),
                    }
                });
            }
        }
        let us_n = median(&mut t_narrow);
        let us_h = median(&mut t_hb);
        let us_r = median(&mut t_roof);
        let tf_n = flops / (us_n * 1e-6) / 1e12;
        let tf_h = flops / (us_h * 1e-6) / 1e12;
        let tf_r = flops / (us_r * 1e-6) / 1e12;
        println!(
            "cell {m}x{k}x{n}: narrow p50={us_n:>8.1}us ({tf_n:.2} TF/s{}) | f16B p50={us_h:>8.1}us ({tf_h:.2} TF/s, {:+.1}%) | roofline p50={us_r:>8.1}us ({tf_r:.2} TF/s)  # {label}",
            if bit { ", bit-identical" } else { ", matches" },
            (us_h / us_n - 1.0) * 100.0,
        );
        // Keep the buffers alive past the closure borrows.
        std::hint::black_box((&a_buf, &w_buf, &out_buf, &ra_buf, &rb_buf, &rout_buf, &wh_buf));
    }
    println!("done: {}", load_avg());
}
