//! MPS-vs-narrow sgemm pricing probe (measurement only, never a gate) —
//! reflex issue 020's last open axis: the typed_decisions trio loses on the
//! encoder GEMM at big m, where torch/MPS "runs the same FLOPs and still wins
//! ~1.4× on GEMM throughput". T7 + the roofline probe closed that axis among
//! OUR instances; the issue named the reopen as a Metal-stack change. The
//! cheapest such change is to call Apple's own `MPSMatrixMultiplication` —
//! the kernel family torch's MPS backend dispatches — for the big-m calls.
//! Price it before building it.
//!
//! Two arms over the real encoder projection shapes at the packed/long m the
//! losing suites run (typed 5-q packed Σseq ≈ 900–1700; long single cases
//! 188–512):
//! 1. `sgemm` — the shipped narrow instance (verbatim copy, see
//!    `sgemm_roofline.rs`; at these m the band rule never picks xwide and
//!    the split rule never splits, so narrow IS the shipped kernel here).
//! 2. `MPSMatrixMultiplication` — fp32 in, fp32 out, B = Wᵀ row-major
//!    [k, n] (the SAME binding the backend already caches), no transpose.
//!
//! Correctness: every cell checks MPS against narrow (rel max-abs), so a
//! descriptor/stride slip fails loud instead of timing garbage.
//!
//! Position-balanced: arm order alternates per round; each arm encodes
//! `DISPATCHES_PER_ROUND` GEMMs into ONE command buffer and waits once (the
//! forward's pipelined posture). Ratios only — the box is shared.
//!
//! Run:
//!   cargo run --release -p riir-infer-laya --features laya-riir-metal \
//!       --example sgemm_mps_probe

#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
// The legacy `objc` macros probe `feature = "cargo-clippy"`.
#![allow(unexpected_cfgs)]

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::runtime::{Class, NO, Object};
use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, CommandBufferRef, CompileOptions, Device, MTLResourceOptions, MTLSize};
use objc2::rc::autoreleasepool;
use std::time::Instant;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

const RESOURCE_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::HazardTrackingModeTracked);

/// `MPSDataTypeFloat32` = `MPSDataTypeFloatBit | 32`.
const MPS_F32: u32 = 0x1000_0000 | 32;

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
    let f = lib.get_function("sgemm", None).expect("sgemm");
    let narrow = device
        .new_compute_pipeline_state_with_function(&f)
        .expect("pipeline");
    let mut geo = Vec::new();
    for &(k, n, label) in SHAPES {
        for &m in MS {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 2_654_435_761) % 1000) as f32 / 1000.0 - 0.5)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 40_503) % 997) as f32 / 997.0 - 0.5)
                .collect();
            let a_buf =
                device.new_buffer_with_data(a.as_ptr().cast(), (a.len() * 4) as u64, RESOURCE_OPTIONS);
            let b_buf =
                device.new_buffer_with_data(b.as_ptr().cast(), (b.len() * 4) as u64, RESOURCE_OPTIONS);
            let o_nar = device.new_buffer((m * n * 4) as u64, RESOURCE_OPTIONS);
            let o_mps = device.new_buffer((m * n * 4) as u64, RESOURCE_OPTIONS);
            let (ma, mb, mc, gemm) = unsafe {
                (
                    mps_matrix(&a_buf, m, k),
                    mps_matrix(&b_buf, k, n),
                    mps_matrix(&o_mps, m, n),
                    mps_gemm(&device, m, k, n),
                )
            };
            let uargs = [m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0, 0];
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
            let enc_narrow = |cb: &CommandBufferRef| {
                let enc = cb.new_compute_command_encoder();
                for _ in 0..DISPATCHES_PER_ROUND {
                    enc.set_compute_pipeline_state(&narrow);
                    enc.set_buffer(0, Some(&a_buf), 0);
                    enc.set_buffer(1, Some(&b_buf), 0);
                    enc.set_buffer(2, Some(&o_nar), 0);
                    for (i, v) in uargs.iter().enumerate() {
                        enc.set_bytes(i as u64 + 3, 4, std::ptr::from_ref(v).cast());
                    }
                    enc.set_threadgroup_memory_length(13, NARROW_STAGING_BYTES);
                    enc.dispatch_thread_groups(grid, tpg);
                }
                enc.end_encoding();
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
            // Correctness + warm-up (pipeline compile / MPS kernel pick).
            autoreleasepool(|_| {
                let cb = queue.new_command_buffer();
                enc_narrow(cb);
                enc_mps(cb);
                cb.commit();
                cb.wait_until_completed();
            });
            let (mut dmax, mut wmax) = (0f32, 0f32);
            let mut bit = true;
            unsafe {
                let pn = std::slice::from_raw_parts(o_nar.contents().cast::<f32>(), m * n);
                let pm = std::slice::from_raw_parts(o_mps.contents().cast::<f32>(), m * n);
                for (x, y) in pn.iter().zip(pm) {
                    dmax = dmax.max((x - y).abs());
                    wmax = wmax.max(x.abs());
                    bit &= x.to_bits() == y.to_bits();
                }
            }
            // Anchor both arms to a CPU reference on sampled rows, so a
            // "bit-identical" pair of all-zero outputs cannot pass.
            let mut cpu_rel = 0f32;
            unsafe {
                let pm = std::slice::from_raw_parts(o_mps.contents().cast::<f32>(), m * n);
                for i in (0..m).step_by(m.div_ceil(7)) {
                    for j in (0..n).step_by(n.div_ceil(9)) {
                        let mut s = 0f64;
                        for kk in 0..k {
                            s += f64::from(a[i * k + kk]) * f64::from(b[kk * n + j]);
                        }
                        cpu_rel = cpu_rel.max(((s as f32) - pm[i * n + j]).abs() / (s.abs() as f32).max(1.0));
                    }
                }
            }
            assert!(wmax > 1.0, "{label} m{m}: outputs are ~zero (wmax {wmax}) — nothing computed");
            assert!(cpu_rel < 1e-3, "{label} m{m}: MPS vs CPU rel {cpu_rel:e}");
            let rel = dmax / wmax.max(1e-30);
            assert!(rel < 1e-4, "{label} m{m}: MPS diverged from narrow rel {rel:e}");
            let (mut tn, mut tm) = (Vec::with_capacity(ROUNDS), Vec::with_capacity(ROUNDS));
            for r in 0..ROUNDS {
                for pos in 0..2 {
                    let arm = (pos + r) % 2;
                    autoreleasepool(|_| {
                        let cb = queue.new_command_buffer();
                        let t0 = Instant::now();
                        if arm == 0 {
                            enc_narrow(cb);
                        } else {
                            enc_mps(cb);
                        }
                        cb.commit();
                        cb.wait_until_completed();
                        let dt = t0.elapsed().as_nanos() / DISPATCHES_PER_ROUND as u128;
                        if arm == 0 { tn.push(dt) } else { tm.push(dt) }
                    });
                }
            }
            let (un, um) = (median_us(&mut tn), median_us(&mut tm));
            let flops = (2 * m * n * k) as f64;
            let ratio = um / un;
            geo.push((label, m, ratio));
            println!(
                "{label:>8} m{m:>4} k{k} n{n}: narrow {un:>8.1}us ({:.2} TF/s) | mps {um:>8.1}us ({:.2} TF/s) | mps/narrow {ratio:.3} | rel {rel:.1e} cpu {cpu_rel:.1e} wmax {wmax:.1}{}",
                flops / (un * 1e-6) / 1e12,
                flops / (um * 1e-6) / 1e12,
                if bit { " bit-identical" } else { "" },
            );
            std::hint::black_box((&a_buf, &b_buf, &o_nar, &o_mps));
        }
    }
    for &m in MS {
        let r: Vec<f64> = geo.iter().filter(|g| g.1 == m).map(|g| g.2).collect();
        let gm = r.iter().map(|x| x.ln()).sum::<f64>() / r.len() as f64;
        println!("m{m:>4}: geo-mean mps/narrow {:.3}", gm.exp());
    }
    println!("done: {}", load_avg());
}
