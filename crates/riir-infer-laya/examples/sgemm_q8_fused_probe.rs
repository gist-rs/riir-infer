//! The fused-Q8 staging pricing probe (Plan 616 Phase 2, measurement
//! only, never a gate) — the option-(i) A/B the plan asks for: the
//! device-resident fused `sgemm_q8` vs today's Phase 1 posture (MPS over
//! the widened F32 `Wᵀ`) on exactly the dense cells Phase 0 priced
//! (`sgemm_mps_probe`'s four projection shapes × m 106–1700).
//!
//! Three arms per cell, position-balanced over 15 rounds:
//! 1. `mps` — Phase 1: `MPSMatrixMultiplication` over the F32 `Wᵀ` (B
//!    row-major [k, n]).
//! 2. `q8` — Phase 2: the fused `sgemm_q8` narrow instance over the RAW
//!    blocked bytes (native [n, k] layout, in-staging dequant) — the
//!    resident posture's unsplit dispatch (the dense cells are unsplit
//!    under both rules).
//! 3. `narrow` — the f32 narrow instance over the F32 `Wᵀ`: isolates the
//!    fused staging's cost (q8 vs narrow) from the dispatch change (narrow
//!    vs MPS). Phase 0 measured mps/narrow 0.584–0.783 geo-mean here.
//!
//! Correctness: q8 and narrow must be BIT-identical (the staged tile
//! values are the host widen's exactly — the Phase 2 bit-identity claim,
//! checked per cell); MPS vs narrow at the probe's rel 1e-4 (the Phase 0
//! posture). The weight bytes are SERIALIZED fake-quant of the fill, so
//! the q8 carrier decodes to exactly the f32 B's values (the converter's
//! own round-trip law).
//!
//! PROVENANCE (quote in the record — the Issue-021 law): run
//! `../riir-reflex/scripts/bench_preflight.sh` beside it.
//!
//! Run:
//!   cargo run --release -p riir-infer-laya --features laya-riir-metal \
//!       --example sgemm_q8_fused_probe

#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
// The legacy `objc` macros probe `feature = "cargo-clippy"`.
#![allow(unexpected_cfgs)]

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::runtime::{Class, NO, Object};
use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, CommandBufferRef, CompileOptions, Device, MTLResourceOptions, MTLSize};
use objc2::rc::autoreleasepool;
use riir_infer_laya::laya::riir::fake_quant::{
    BLOCK, fake_quant_q8, q8_quant_of, q8_scale_bits, q8_scale_f32,
};
use std::time::Instant;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

const RESOURCE_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::HazardTrackingModeTracked);

/// `MPSDataTypeFloat32` = `MPSDataTypeFloatBit | 32`.
const MPS_F32: u32 = 0x1000_0000 | 32;

/// The f32 narrow instance (verbatim `sgemm_mps_probe` copy) — arm 3.
const MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
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
    device const float* A = a;
    device const float* B = b;
    device float* C = out;
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
            tb[kk * TBS + col] = (bc < k && n0 + col < n) ? B[bc * b_rs + n0 + col] : 0.0f;
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

// The Phase 2 fused narrow instance: the same tile math, the B tile
// staged from the RAW blocked bytes (k-fastest lanes, one block per
// warp). Verbatim the backend's `sgemm_q8` body.
kernel void sgemm_q8(
    device const float* a [[buffer(0)]],
    device const uint8_t* q8 [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_rs [[buffer(6)]],
    constant uint& a_cs [[buffer(7)]],
    threadgroup float* raw [[threadgroup(8)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + 32u * TAS;
    threadgroup float* edge = raw;
    const uint m0 = gtp.y * BM;
    const uint n0 = gtp.x * BN;
    device const float* A = a;
    device float* C = out;
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
            const uint col = idx >> 6u;
            const uint kk = idx & 63u;
            const uint bc = t + kk;
            float v = 0.0f;
            if (bc < k && n0 + col < n) {
                const uint e = (n0 + col) * k + bc;
                device const uint8_t* bp = q8 + (uint64_t)(e >> 5) * 34u;
                const ushort bits = (ushort)bp[0] | ((ushort)bp[1] << 8);
                v = float(as_type<half>(bits)) * float((int8_t)bp[2u + (e & 31u)]);
            }
            tb[kk * TBS + col] = v;
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

const NARROW_STAGING_BYTES: u64 = ((32 * 65 + 64 * 65) * 4) as u64;

/// (k, n, label) — the encoder projections at the english geometry
/// (d = 1024, i = 2624).
const SHAPES: &[(usize, usize, &str)] = &[
    (1024, 3072, "qkv"),
    (1024, 1024, "o"),
    (1024, 5248, "mlp-up"),
    (2624, 1024, "mlp-down"),
];
const MS: &[usize] = &[106, 188, 317, 512, 895, 1700];
const ROUNDS: usize = 15;
const DISPATCHES_PER_ROUND: usize = 3;

/// One `MPSMatrix` view over `buf` (`rows × cols` f32, dense rows).
unsafe fn mps_matrix(buf: &Buffer, rows: usize, cols: usize) -> *mut Object {
    let desc_cls = Class::get("MPSMatrixDescriptor").expect("MPSMatrixDescriptor");
    let mat_cls = Class::get("MPSMatrix").expect("MPSMatrix");
    unsafe {
        let desc: *mut Object = msg_send![desc_cls,
            matrixDescriptorWithRows: rows as u64
            columns: cols as u64
            rowBytes: (cols * 4) as u64
            dataType: MPS_F32];
        let mat: *mut Object = msg_send![mat_cls, alloc];
        let buf_ptr = buf.as_ptr().cast::<Object>();
        msg_send![mat, initWithBuffer: buf_ptr descriptor: desc]
    }
}

unsafe fn mps_gemm(device: &Device, m: usize, k: usize, n: usize) -> *mut Object {
    let cls = Class::get("MPSMatrixMultiplication").expect("MPSMatrixMultiplication");
    unsafe {
        let g: *mut Object = msg_send![cls, alloc];
        let dev = device.as_ptr().cast::<Object>();
        msg_send![g, initWithDevice: dev
            transposeLeft: NO
            transposeRight: NO
            resultRows: m as u64
            resultColumns: n as u64
            interiorColumns: k as u64
            alpha: 1.0f64
            beta: 0.0f64]
    }
}

/// Serialize fake-quantized values into the raw Q8_0 block layout (the
/// converter's own block loop) AND return the decoded values — the f32
/// carrier arms (narrow + the CPU anchor) must stage the DECODED values,
/// exactly what the Q8 carrier decodes to (the identity gate's Dense
/// twin law; a RAW-fill f32 arm would compare different VALUES, not
/// different carriers).
fn q8_fixture(data: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut q = data.to_vec();
    fake_quant_q8(&mut q);
    let mut out = Vec::with_capacity(q.len() / BLOCK * 34 + 34);
    for block in q.chunks(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let bits = q8_scale_bits(amax);
        let d = q8_scale_f32(bits);
        out.extend_from_slice(&bits.to_le_bytes());
        for &x in block {
            out.push(q8_quant_of(x, d) as u8);
        }
    }
    (out, q)
}

fn median_us(samples: &mut [u128]) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1000.0
}

fn load_avg() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime failed".into())
}

fn main() {
    println!("box: {}", load_avg());
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(MSL, &CompileOptions::new())
        .expect("MSL compile");
    let mut pipes = Vec::new();
    for name in ["sgemm", "sgemm_q8"] {
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let p = device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline");
        pipes.push(p);
    }
    let narrow = &pipes[0];
    let fused = &pipes[1];

    let mut geo_mps = Vec::new();
    let mut geo_fused = Vec::new();
    for &(k, n, label) in SHAPES {
        for &m in MS {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 2_654_435_761) % 1000) as f32 / 1000.0 - 0.5)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 40_503) % 997) as f32 / 997.0 - 0.5)
                .collect();
            // B in BOTH carriers: the f32 carrier is the DECODED values
            // (what the Q8 carrier decodes to — the identity law), and
            // the probe stages its TRANSPOSE (row-major [k, n] Wᵀ, the
            // Phase 1 posture's binding).
            let (q8, bq) = q8_fixture(&b);
            let b_t: Vec<f32> = {
                let mut t = vec![0f32; k * n];
                for (row, src) in bq.chunks_exact(k).enumerate() {
                    for (kk, v) in src.iter().enumerate() {
                        t[kk * n + row] = *v;
                    }
                }
                t
            };
            let a_buf = device.new_buffer_with_data(
                a.as_ptr().cast(),
                (a.len() * 4) as u64,
                RESOURCE_OPTIONS,
            );
            let b_buf = device.new_buffer_with_data(
                b_t.as_ptr().cast(),
                (b_t.len() * 4) as u64,
                RESOURCE_OPTIONS,
            );
            let q_buf =
                device.new_buffer_with_data(q8.as_ptr().cast(), q8.len() as u64, RESOURCE_OPTIONS);
            let o_mps = device.new_buffer((m * n * 4) as u64, RESOURCE_OPTIONS);
            let o_nar = device.new_buffer((m * n * 4) as u64, RESOURCE_OPTIONS);
            let o_q8 = device.new_buffer((m * n * 4) as u64, RESOURCE_OPTIONS);
            let (ma, mb, mc, gemm) = unsafe {
                (
                    mps_matrix(&a_buf, m, k),
                    mps_matrix(&b_buf, k, n),
                    mps_matrix(&o_mps, m, n),
                    mps_gemm(&device, m, k, n),
                )
            };
            let uargs_n = [
                m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0, 0,
            ];
            let uargs_q = [m as u32, n as u32, k as u32, k as u32, 1];
            let grid = MTLSize {
                width: n.div_ceil(64) as u64,
                height: m.div_ceil(32) as u64,
                depth: 1,
            };
            let tpg = MTLSize {
                width: 512,
                height: 1,
                depth: 1,
            };
            let enc_mps = |cb: &CommandBufferRef| {
                let cbp = cb.as_ptr().cast::<Object>();
                for _ in 0..DISPATCHES_PER_ROUND {
                    unsafe {
                        let () = msg_send![gemm, encodeToCommandBuffer: cbp
                            leftMatrix: ma
                            rightMatrix: mb
                            resultMatrix: mc];
                    }
                }
            };
            let enc_kernel = |cb: &CommandBufferRef,
                              p: &metal::ComputePipelineState,
                              out: &Buffer,
                              bufs: &[(&Buffer, u64)],
                              uargs: &[u32],
                              tg_index: u64| {
                let enc = cb.new_compute_command_encoder();
                for _ in 0..DISPATCHES_PER_ROUND {
                    enc.set_compute_pipeline_state(p);
                    for (i, (b, off)) in bufs.iter().enumerate() {
                        enc.set_buffer(i as u64, Some(b), *off);
                    }
                    let ob = bufs.len() as u64;
                    enc.set_buffer(ob, Some(out), 0);
                    for (i, v) in uargs.iter().enumerate() {
                        enc.set_bytes(ob + 1 + i as u64, 4, std::ptr::from_ref(v).cast());
                    }
                    enc.set_threadgroup_memory_length(tg_index, NARROW_STAGING_BYTES);
                    enc.dispatch_thread_groups(grid, tpg);
                }
                enc.end_encoding();
            };
            // Correctness + warm-up: narrow and q8 must be BIT-identical
            // (the staged tile values are the host widen's exactly); MPS
            // at the Phase 0 rel budget. Anchored against a CPU reference
            // on sampled rows so an all-zero pair cannot pass.
            autoreleasepool(|_| {
                let cb = queue.new_command_buffer();
                enc_kernel(
                    cb,
                    narrow,
                    &o_nar,
                    &[(&a_buf, 0), (&b_buf, 0)],
                    &uargs_n,
                    13,
                );
                enc_kernel(cb, fused, &o_q8, &[(&a_buf, 0), (&q_buf, 0)], &uargs_q, 8);
                enc_mps(cb);
                cb.commit();
                cb.wait_until_completed();
            });
            let (mut dmax, mut wmax) = (0f32, 0f32);
            let mut bit = true;
            unsafe {
                let pn = std::slice::from_raw_parts(o_nar.contents().cast::<f32>(), m * n);
                let pq = std::slice::from_raw_parts(o_q8.contents().cast::<f32>(), m * n);
                let pm = std::slice::from_raw_parts(o_mps.contents().cast::<f32>(), m * n);
                for ((x, y), z) in pn.iter().zip(pq).zip(pm) {
                    dmax = dmax.max((x - z).abs());
                    wmax = wmax.max(x.abs());
                    bit &= x.to_bits() == y.to_bits();
                }
                for i in (0..m).step_by(m.div_ceil(7)) {
                    for j in (0..n).step_by(n.div_ceil(9)) {
                        let mut s = 0f64;
                        for kk in 0..k {
                            // b_t is the Wᵀ the two device arms stage
                            // (row-major [k, n]); the weight `b` is
                            // [n, k] and indexes differently.
                            s += f64::from(a[i * k + kk]) * f64::from(b_t[kk * n + j]);
                        }
                        let rel = ((s as f32) - pm[i * n + j]).abs() / (s.abs() as f32).max(1.0);
                        assert!(rel < 1e-3, "{label} m{m}: MPS vs CPU rel {rel:e}");
                    }
                }
            }
            assert!(
                wmax > 1.0,
                "{label} m{m}: outputs are ~zero (wmax {wmax}) — nothing computed"
            );
            let rel = dmax / wmax.max(1e-30);
            assert!(rel < 1e-4, "{label} m{m}: MPS vs narrow rel {rel:e}");
            assert!(
                bit,
                "{label} m{m}: the fused-q8 arm MUST be bit-identical to the f32 narrow arm"
            );
            let (mut tm, mut tq, mut tn) = (
                Vec::with_capacity(ROUNDS),
                Vec::with_capacity(ROUNDS),
                Vec::with_capacity(ROUNDS),
            );
            for r in 0..ROUNDS {
                for pos in 0..3 {
                    let arm = (pos + r) % 3;
                    autoreleasepool(|_| {
                        let cb = queue.new_command_buffer();
                        let t0 = Instant::now();
                        match arm {
                            0 => enc_mps(cb),
                            1 => enc_kernel(
                                cb,
                                fused,
                                &o_q8,
                                &[(&a_buf, 0), (&q_buf, 0)],
                                &uargs_q,
                                8,
                            ),
                            _ => enc_kernel(
                                cb,
                                narrow,
                                &o_nar,
                                &[(&a_buf, 0), (&b_buf, 0)],
                                &uargs_n,
                                13,
                            ),
                        }
                        cb.commit();
                        cb.wait_until_completed();
                        let dt = t0.elapsed().as_nanos() / DISPATCHES_PER_ROUND as u128;
                        match arm {
                            0 => tm.push(dt),
                            1 => tq.push(dt),
                            _ => tn.push(dt),
                        }
                    });
                }
            }
            let (um, uq, un) = (median_us(&mut tm), median_us(&mut tq), median_us(&mut tn));
            geo_mps.push(um / un);
            geo_fused.push(uq / un);
            println!(
                "{label:>8} m{m:>4} k{k} n{n}: mps {um:>8.1} | fused-q8 {uq:>8.1} | narrow {un:>8.1} us | mps/narrow {:.3} fused/narrow {:.3} fused/mps {:.3} | bit{bonus}",
                um / un,
                uq / un,
                uq / um,
                bonus = if bit { "-identical" } else { "" },
            );
            std::hint::black_box((&a_buf, &b_buf, &q_buf, &o_nar, &o_q8, &o_mps));
        }
    }
    let gm = |v: &[f64]| (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp();
    println!("geo-mean per m (over the four shapes):");
    for &m in MS {
        let r: Vec<f64> = geo_mps
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| i % MS.len() == MS.iter().position(|&x| x == m).unwrap())
            .map(|(_, x)| x)
            .collect();
        let f: Vec<f64> = geo_fused
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| i % MS.len() == MS.iter().position(|&x| x == m).unwrap())
            .map(|(_, x)| x)
            .collect();
        println!(
            "m{m:>4}: mps/narrow {:.3} · fused-q8/narrow {:.3} · fused-q8/mps {:.3}",
            gm(&r),
            gm(&f),
            gm(&f) / gm(&r)
        );
    }
    println!("done: {}", load_avg());
}
