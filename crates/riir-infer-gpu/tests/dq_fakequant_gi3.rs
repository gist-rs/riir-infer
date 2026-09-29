//! Plan 614 / Issue 026 — G-i3 for the CubeCL lane: the on-device fake-quant
//! kernels must equal the host reference BIT-EXACTLY on seeded tensors, both
//! grids + the A8 control, including the pinned edge cases (zero block, A2
//! argmax TIE with lowest-index-wins, sub-f16-scale guard, ragged tail,
//! multi-row isolation).
//!
//! Runs on whatever backend CubeCL selects on this box (Metal on the M3, CUDA
//! on the 4090) — the per-lane half of G-i3's "two kernels, one reference"
//! law (the CUDA/cudarc prefill-lane kernel gets its own arm on the 4090).
//!
//! gpu-less boxes: SKIPS loud (no silent green).

#![cfg(feature = "dq_phase_bench")]

use riir_infer_gpu::cubecl_runtime::{read_f32, ActiveComputeClient};
use riir_infer_gpu::dq_fakequant::{host_quant_dequant, DqGrid};
use riir_infer_gpu::dq_fakequant_cubecl::test_support_launch;

use cubecl::prelude::*;

use riir_infer_gpu::CubeCLContext;

/// Deterministic LCG (no RNG dep; identical on every box).
fn lcg(seed: u64) -> impl Iterator<Item = f32> {
    let mut s = seed;
    std::iter::from_fn(move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = ((s >> 40) as i64 as f64 / (1u64 << 24) as f64) as f32;
        Some((f - 0.5) * 6.0) // ~Uniform(-3, 3) — activation-scale values
    })
}

fn run_grid(
    client: &ActiveComputeClient,
    buf: &[f32],
    dim: usize,
    rows: usize,
    grid: DqGrid,
) -> Vec<f32> {
    let handle = client.create_from_slice(f32::as_bytes(buf));
    test_support_launch(client, &handle, dim, rows, grid);
    read_f32(client, handle).expect("read back")
}

fn check(grid: DqGrid, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{grid:?}: length");
    let mut bad = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if g.to_bits() != e.to_bits() {
            bad += 1;
            if first.is_none() {
                first = Some((i, g, e));
            }
        }
    }
    assert_eq!(
        bad, 0,
        "{grid:?}: {bad}/{} mismatched, first at {first:?}",
        got.len()
    );
}

#[test]
fn cubecl_fakequant_matches_host_reference_all_grids() {
    let ctx = match CubeCLContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no CubeCL device on this box ({e})");
            return;
        }
    };
    let client = ctx.client();
    // Seeded tensors: 8 rows × 3584 (the model's hidden dim) per grid.
    let dim = 3584usize;
    let rows = 8usize;
    for (grid, seed) in [
        (DqGrid::A2, 0xD00D1u64),
        (DqGrid::A4, 0xD00D2u64),
        (DqGrid::A8, 0xD00D3u64),
    ] {
        let buf: Vec<f32> = lcg(seed).take(dim * rows).collect();
        let mut expected = vec![0f32; buf.len()];
        host_quant_dequant(grid, &buf, &mut expected, dim);
        let got = run_grid(&client, &buf, dim, rows, grid);
        check(grid, &got, &expected);
    }
}

#[test]
fn cubecl_fakequant_edge_cases() {
    let ctx = match CubeCLContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no CubeCL device on this box ({e})");
            return;
        }
    };
    let client = ctx.client();
    // One row, dim = 128: [zero-prefix block | the A2 TIE (indices 3 & 7 at
    // ∓2.0 — lowest index wins the sign) | on-grid + near-max values].
    let dim = 128usize;
    let mut row = vec![0f32; dim];
    row[3] = -2.0; // the LOWEST-index max — ITS sign must steer d
    row[7] = 2.0; // the tie (higher index — must lose the argmax)
    row[40] = 0.5;
    row[41] = -1.9;
    for grid in [DqGrid::A2, DqGrid::A4, DqGrid::A8] {
        let mut expected = vec![0f32; dim];
        host_quant_dequant(grid, &row, &mut expected, dim);
        let got = run_grid(&client, &row, dim, 1, grid);
        check(grid, &got, &expected);
    }
}

#[test]
fn cubecl_fakequant_subf16_scale_block() {
    let ctx = match CubeCLContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no CubeCL device on this box ({e})");
            return;
        }
    };
    let client = ctx.client();
    // amax ≈ 2e-8 → amax/2 lands in the f16-subnormal range: the guard
    // branch must agree with the host reference bit-exactly.
    let dim = 128usize;
    let mut row = vec![1e-8f32; dim];
    row[0] = 2e-8;
    row[1] = -1.5e-8;
    for grid in [DqGrid::A2, DqGrid::A8] {
        let mut expected = vec![0f32; dim];
        host_quant_dequant(grid, &row, &mut expected, dim);
        let got = run_grid(&client, &row, dim, 1, grid);
        check(grid, &got, &expected);
    }
}

#[test]
fn cubecl_fakequant_ragged_tail_multi_row() {
    let ctx = match CubeCLContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no CubeCL device on this box ({e})");
            return;
        }
    };
    let client = ctx.client();
    // dim NOT a multiple of either block: the per-row tail is its own block
    // and rows never share statistics (row 1's huge tail value must not
    // affect row 2's grid).
    let dim = 100usize;
    let rows = 3usize;
    let mut buf: Vec<f32> = lcg(0x5EED).take(dim * rows).collect();
    // Overwrite row 1's last element with an outlier.
    buf[dim..2 * dim].last_mut().unwrap().clone_from(&1e3f32);
    for grid in [DqGrid::A2, DqGrid::A4, DqGrid::A8] {
        let mut expected = vec![0f32; buf.len()];
        host_quant_dequant(grid, &buf, &mut expected, dim);
        let got = run_grid(&client, &buf, dim, rows, grid);
        check(grid, &got, &expected);
    }
}
