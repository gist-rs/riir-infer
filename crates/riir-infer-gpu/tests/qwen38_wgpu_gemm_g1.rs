//! Issue 1150 G1 — the tiled Q4_K × f32 GEMM unit gate.
//!
//! Kernel vs a CPU reference on random shapes: correctness of the dequant +
//! tiling + accumulation across m-edge (48 not divisible by 32), p-edge, and
//! the n=5120 production column count. Tiny tensors — runs on any Metal box.

#![cfg(feature = "qwen38_wgpu")]

use bytemuck::Zeroable;
use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, quantize_row_q4_k};
use riir_infer_gpu::gemm_q4k_tile_cubecl::{GatherRowsF32CubeCL, GemmQ4KTileCubeCL};
use riir_infer_gpu::{cubecl_runtime::CubeCLContext, gemv_q4k_cubecl::Q4KHandle};

fn cpu_gemm_q4k(blocks: &[BlockQ4K], x: &[f32], m: usize, n: usize, p: usize) -> Vec<f32> {
    let bpr = n / QK_K;
    let mut w = vec![0.0f32; m * n];
    let mut row = vec![0.0f32; n];
    for r in 0..m {
        riir_infer_core::quant::q4k::dequantize_row_q4_k(&blocks[r * bpr..(r + 1) * bpr], &mut row);
        w[r * n..(r + 1) * n].copy_from_slice(&row);
    }
    let mut out = vec![0.0f32; p * m];
    for t in 0..p {
        for r in 0..m {
            let mut dot = 0.0f32;
            for c in 0..n {
                dot += w[r * n + c] * x[t * n + c];
            }
            out[t * m + r] = dot;
        }
    }
    out
}

fn quantize_matrix(data: &[f32], m: usize, n: usize) -> Vec<BlockQ4K> {
    assert!(n.is_multiple_of(QK_K));
    let bpr = n / QK_K;
    let mut blocks = vec![BlockQ4K::zeroed(); m * bpr];
    let mut row = vec![0.0f32; n];
    for r in 0..m {
        row.copy_from_slice(&data[r * n..(r + 1) * n]);
        quantize_row_q4_k(&row, &mut blocks[r * bpr..(r + 1) * bpr]);
    }
    blocks
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(14_426_950_408_889_407);
    ((*seed >> 33) as f32 / (u64::MAX >> 33) as f32) * 2.0 - 1.0
}

fn run_case(m: usize, n: usize, p: usize) {
    let mut seed: u64 = 0x1150_1150_1150_1150;
    let w: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
    let blocks = quantize_matrix(&w, m, n);
    let expect = cpu_gemm_q4k(&blocks, &x, m, n, p);

    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let handle = Q4KHandle::from_blocks(&client, &blocks, m, n);
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
    unsafe {
        GemmQ4KTileCubeCL::launch::<riir_infer_gpu::cubecl_runtime::ActiveRuntime>(
            &client,
            &handle,
            x_h,
            out_h.clone(),
            p,
        );
    }
    let got_bytes = client.read_one(out_h).expect("read");
    let got: &[f32] = bytemuck::cast_slice(&got_bytes);

    // Q4_K quantization error dominates: compare against the DEQUANTIZED-weight
    // reference with a tolerance scaled to the q4k grid (d·sc·0.5 per element,
    // accumulated over n). The kernel and the CPU reference dequantize
    // IDENTICALLY (same block math), so the residual is fp reassociation only.
    let mut max_abs = 0.0f32;
    for i in 0..p * m {
        max_abs = max_abs.max((got[i] - expect[i]).abs());
    }
    let scale = expect.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
    let rel = max_abs / scale;
    assert!(
        rel < 2e-4,
        "gemm_q4k_tile m={m} n={n} p={p}: max_abs {max_abs:.3e} rel {rel:.3e}"
    );
    println!("[g1] m={m} n={n} p={p}: max_abs {max_abs:.3e} rel {rel:.3e} OK");
}

#[test]
fn gemm_q4k_tile_edges_and_production_n() {
    // m-edge: 48 rows < tile 32? no — 48 = 32 + 16 (partial second tile).
    run_case(48, 256, 32);
    // p-edge: exactly one tile; two tiles.
    run_case(64, 512, 32);
    run_case(64, 512, 64);
    // Production column count n=5120 (80 KTILEs), partial m.
    run_case(100, 5120, 64);
    // FFN shape, small p.
    run_case(17408, 5120, 32);
}

#[test]
fn gather_rows_basic() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let n = 8usize;
    let src: Vec<f32> = (0..5 * n).map(|i| i as f32).collect();
    let src_h = client.create_from_slice(bytemuck::cast_slice(&src));
    let idx: Vec<u32> = vec![3, 0, 4];
    let idx_h = client.create_from_slice(bytemuck::cast_slice(&idx));
    let out_h = client.create_from_slice(&vec![0u8; 3 * n * 4]);
    unsafe {
        GatherRowsF32CubeCL::launch::<riir_infer_gpu::cubecl_runtime::ActiveRuntime>(
            &client,
            src_h,
            idx_h,
            out_h.clone(),
            5,
            3,
            n,
        );
    }
    let got_bytes = client.read_one(out_h).expect("read");
    let got: &[f32] = bytemuck::cast_slice(&got_bytes);
    for (i, &s) in idx.iter().enumerate() {
        for c in 0..n {
            let want = (s as f32) * n as f32 + c as f32;
            assert_eq!(got[i * n + c], want, "row {i} col {c}");
        }
    }
}
