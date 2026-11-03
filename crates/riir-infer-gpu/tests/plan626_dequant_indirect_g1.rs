//! Plan 626 T3(a) — G1 for the INDIRECT wte dequant kernel
//! (`dequant_wte_row_indirect_f32`): the resident greedy lane's embed reads
//! its row id from a DEVICE ring slot instead of the launch scalar, so the
//! token never round-trips through the host (Issue 1168: the per-token sync
//! is the decode lane's 17.79 → 15.32 ms/token lever).
//!
//! The bar is BIT-IDENTITY, not tolerance: the indirect body is the scalar
//! [`DequantWteRowCubeCL`] per-element arithmetic verbatim with
//! `row_idx = id_buf[id_slot]`, so for every row `r` and every slot holding
//! `r`, the two kernels must produce identical f32 bits (to_bits equality,
//! the Issue 1129 G1 law). Every row of a deterministic ternary table is
//! walked through both kernels — not just one, so a row-indexing bug in the
//! indirect variant cannot hide behind a lucky row.
//!
//! Hermetic: a 3×128 ternary table, no model file, ~1 s — runs in the
//! default `cargo test -p riir-infer-gpu --features cubecl_runtime,ternary_gemv`
//! lane, GPU-present (skips loud without one — a deferral, never a green).
//!
//! ```bash
//! cargo test -p riir-infer-gpu --features cubecl_runtime,ternary_gemv \
//!     --test plan626_dequant_indirect_g1 -- --nocapture
//! ```

#![cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]

use cubecl::prelude::*;

use katgpt_core::TernaryGroupWeights;
use riir_infer_gpu::cubecl_runtime::{ActiveComputeClient, ActiveRuntime, CubeCLContext};
use riir_infer_gpu::ternary_deltanet_gpu_forward::{
    DequantWteRowCubeCL, DequantWteRowIndirectCubeCL,
};

const ROWS: usize = 3;
const COLS: usize = 128; // one ternary group (GROUP_SIZE_DW), two u64 blocks

/// Deterministic ternary table exercising all three values (+1 / 0 / −1) in
/// every row, with distinct per-row scale so a row mix-up reads loud.
fn build_table() -> TernaryGroupWeights {
    let mut w = TernaryGroupWeights::new(ROWS, COLS);
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for r in 0..ROWS {
        for c in 0..COLS {
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
        // Distinct per-row scales: a scalar/indirect row mismatch would land
        // on a different scale group and fail to_bits immediately.
        w.set_scale(r, 0, 0.25 * (r as f32 + 1.0));
    }
    w
}

#[test]
fn plan626_dequant_indirect_is_bit_identical_to_scalar() {
    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed) — hermetic arm deferred, never green");
        return;
    };
    let client: ActiveComputeClient = ctx.client();

    let table = build_table();
    let handle = riir_infer_gpu::gemv_ternary_cubecl::TernaryHandle::from_weights(&client, &table);
    let blocks64 = handle.blocks64 as u32;
    let groups_per_row = handle.groups_per_row as u32;
    let n = COLS as u32;
    assert_eq!(handle.m, ROWS, "table row count");
    assert_eq!(blocks64, 2, "128 cols = two u64 blocks per row");

    // The device id buffer holds every row id at slot `r` — the resident
    // ring's shape in miniature (id_buf[slot], a scalar slot arg).
    let ids: Vec<u32> = (0..ROWS as u32).collect();
    let id_buf = client.create_from_slice(u32::as_bytes(&ids));

    for r in 0..ROWS as u32 {
        let out_scalar = client.empty(COLS * core::mem::size_of::<f32>());
        unsafe {
            DequantWteRowCubeCL::launch::<ActiveRuntime>(
                &client,
                handle.pos_bits_u32.clone(),
                handle.neg_bits_u32.clone(),
                handle.group_scale_f32.clone(),
                out_scalar.clone(),
                r,
                blocks64,
                groups_per_row,
                n,
            );
        }

        let out_indirect = client.empty(COLS * core::mem::size_of::<f32>());
        unsafe {
            DequantWteRowIndirectCubeCL::launch::<ActiveRuntime>(
                &client,
                handle.pos_bits_u32.clone(),
                handle.neg_bits_u32.clone(),
                handle.group_scale_f32.clone(),
                id_buf.clone(),
                out_indirect.clone(),
                r, // id_slot — the ring slot naming row `ids[r] == r`
                ROWS as u32,
                blocks64,
                groups_per_row,
                n,
            );
        }

        let a = client.read_one(out_scalar).expect("read scalar out");
        let b = client.read_one(out_indirect).expect("read indirect out");
        let a = f32::from_bytes(&a);
        let b = f32::from_bytes(&b);
        assert_eq!(a.len(), COLS, "scalar out length");
        assert_eq!(b.len(), COLS, "indirect out length");
        for c in 0..COLS {
            assert_eq!(
                a[c].to_bits(),
                b[c].to_bits(),
                "row {r} col {c}: indirect dequant diverged from the scalar kernel \
                 (scalar {} vs indirect {})",
                a[c],
                b[c]
            );
        }
    }
    eprintln!("[plan626-dequant-indirect-g1] {ROWS} rows × {COLS} cols bit-identical ✓");
}
