//! Issue 1173 — the fused norm+gating family G1: each fused kernel vs the
//! UNFUSED launch sequence it replaces, GPU-vs-GPU on ActiveRuntime,
//! BITWISE on both the f32 and f16 outputs.
//!
//! The fusions are launch-mechanics changes, never numerics changes: every
//! kernel copies its host kernel's phases verbatim (the same strided x²
//! accumulation, the same 256-slot pairwise reduction tree, the same
//! left-to-right multiply order, `half::f16::cast_from` for the twin).
//! A single ULP of drift here fails loud — the fused arm's byte-identity
//! claim (the GOAT G1 bar) rests on this file.
//!
//! Arms:
//! - F1: `RmsNormZgateBatchedF16CubeCL` vs `RmsNormBatched` →
//!   `DeltanetZGating` (→ `CastF32ToF16`) — rec_n_b + twin bit-exact.
//! - F2: `RmsNormTwinBatchedCubeCL` vs `RmsNormBatched` → `CastF32ToF16`
//!   — norm_out + twin bit-exact.
//! - F3: `ResidualRmsNormTwinBatchedCubeCL` vs `ResidualAdd` →
//!   `RmsNormBatched` → `CastF32ToF16` — x (in-place), norm_out + twin
//!   bit-exact.
//!
//! Shapes cover: dim < 256 (idle lanes in the reduction), dim = 128 and
//! 96 (the non-power-of-two width), odd row counts, and the twin-less
//! posture (None → the fused kernel must still produce the byte-identical
//! f32 output).
//!
//! #![cfg] green-zero rule: the file carries its required-features row in
//! Cargo.toml ([[test]] qwen38_fused_norm_gating_g1).

#![cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]

use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::deltanet_cubecl::DeltanetZGatingCubeCL;
use riir_infer_gpu::elementwise_cubecl::CastF32ToF16CubeCL;
use riir_infer_gpu::norms_cubecl::{ResidualAddCubeCL, RmsNormBatchedCubeCL};
use riir_infer_gpu::norms_fused_batched_cubecl::{
    ResidualRmsNormTwinBatchedCubeCL, RmsNormTwinBatchedCubeCL, RmsNormZgateBatchedF16CubeCL,
};

/// Minimal deterministic PRNG (xorshift) — the crate's fixture discipline.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn read_f32(client: &ComputeClient<ActiveRuntime>, h: &Handle, n: usize) -> Vec<f32> {
    client
        .read_one(h.clone())
        .map(|b| bytemuck::cast_slice(&b)[..n].to_vec())
        .unwrap_or_default()
}

/// The f16 twin read as raw BITS — the byte-identity contract is on the
/// encoded half pattern, not on a widening round-trip.
fn read_f16_bits(client: &ComputeClient<ActiveRuntime>, h: &Handle, n: usize) -> Vec<u16> {
    client
        .read_one(h.clone())
        .map(|b| bytemuck::cast_slice(&b)[..n].to_vec())
        .unwrap_or_default()
}

fn fixture(rows: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed);
    // Wide-range activations (the norm must see non-unit variance), z in a
    // sigmoid-spread range (positive and negative), gamma near 1 ± spread.
    let input: Vec<f32> = (0..rows * dim)
        .map(|_| (rng.next_f32() - 0.5) * 8.0)
        .collect();
    let z: Vec<f32> = (0..rows * dim)
        .map(|_| (rng.next_f32() - 0.5) * 6.0)
        .collect();
    let gamma: Vec<f32> = (0..dim).map(|_| 0.5 + rng.next_f32()).collect();
    (input, z, gamma)
}

fn eps() -> f32 {
    1e-5
}

#[test]
fn f1_zgate_fused_is_bitwise_the_unfused_pair() {
    let client = CubeCLContext::new().expect("gpu context").client();
    for (rows, dim) in [(72usize, 128usize), (9, 96), (5, 60), (3, 256)] {
        let (input, z, gamma) = fixture(rows, dim, 0x1173_0001);
        let n = rows * dim;
        let eps = eps();

        // Arm A — the unfused sequence: norm → zgate (in-place) → cast.
        let in_a = client.create_from_slice(bytemuck::cast_slice(&input));
        let out_a = client.create_from_slice(&vec![0u8; n * 4]);
        let out_a_h = client.create_from_slice(&vec![0u8; n * 2]);
        unsafe {
            RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                in_a.clone(),
                client.create_from_slice(bytemuck::cast_slice(&gamma)),
                out_a.clone(),
                rows,
                dim,
                eps,
            );
            DeltanetZGatingCubeCL::launch::<ActiveRuntime>(
                &client,
                out_a.clone(),
                client.create_from_slice(bytemuck::cast_slice(&z)),
                n,
            );
            CastF32ToF16CubeCL::launch::<ActiveRuntime>(&client, out_a.clone(), out_a_h.clone(), n);
        }

        // Arm B — the fused kernel (twin present).
        let in_b = client.create_from_slice(bytemuck::cast_slice(&input));
        let out_b = client.create_from_slice(&vec![0u8; n * 4]);
        let out_b_h = client.create_from_slice(&vec![0u8; n * 2]);
        unsafe {
            RmsNormZgateBatchedF16CubeCL::launch::<ActiveRuntime>(
                &client,
                in_b.clone(),
                client.create_from_slice(bytemuck::cast_slice(&gamma)),
                client.create_from_slice(bytemuck::cast_slice(&z)),
                out_b.clone(),
                Some(out_b_h.clone()),
                rows,
                dim,
                eps,
            );
        }

        let a = read_f32(&client, &out_a, n);
        let b = read_f32(&client, &out_b, n);
        assert_eq!(
            a, b,
            "F1 f32 mismatch at rows={rows} dim={dim}: the fused kernel drifted from the unfused pair"
        );
        let ha = read_f16_bits(&client, &out_a_h, n);
        let hb = read_f16_bits(&client, &out_b_h, n);
        assert_eq!(
            ha, hb,
            "F1 f16 twin mismatch at rows={rows} dim={dim}: the twin rounding drifted"
        );

        // Arm C — the fused kernel TWIN-LESS (the no-half-B posture): the
        // f32 output must stay byte-identical with the twin slot unbound.
        let in_c = client.create_from_slice(bytemuck::cast_slice(&input));
        let out_c = client.create_from_slice(&vec![0u8; n * 4]);
        unsafe {
            RmsNormZgateBatchedF16CubeCL::launch::<ActiveRuntime>(
                &client,
                in_c,
                client.create_from_slice(bytemuck::cast_slice(&gamma)),
                client.create_from_slice(bytemuck::cast_slice(&z)),
                out_c.clone(),
                None,
                rows,
                dim,
                eps,
            );
        }
        let c = read_f32(&client, &out_c, n);
        assert_eq!(
            a, c,
            "F1 twin-less posture drifted at rows={rows} dim={dim}"
        );
    }
}

#[test]
fn f2_norm_twin_fused_is_bitwise_norm_then_cast() {
    let client = CubeCLContext::new().expect("gpu context").client();
    for (rows, dim) in [(7usize, 100usize), (16, 128), (4, 61)] {
        let (input, _z, gamma) = fixture(rows, dim, 0x1173_0002);
        let n = rows * dim;
        let eps = eps();

        let in_a = client.create_from_slice(bytemuck::cast_slice(&input));
        let out_a = client.create_from_slice(&vec![0u8; n * 4]);
        let out_a_h = client.create_from_slice(&vec![0u8; n * 2]);
        unsafe {
            RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                in_a.clone(),
                client.create_from_slice(bytemuck::cast_slice(&gamma)),
                out_a.clone(),
                rows,
                dim,
                eps,
            );
            CastF32ToF16CubeCL::launch::<ActiveRuntime>(&client, out_a.clone(), out_a_h.clone(), n);
        }

        let in_b = client.create_from_slice(bytemuck::cast_slice(&input));
        let out_b = client.create_from_slice(&vec![0u8; n * 4]);
        let out_b_h = client.create_from_slice(&vec![0u8; n * 2]);
        unsafe {
            RmsNormTwinBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                in_b,
                client.create_from_slice(bytemuck::cast_slice(&gamma)),
                out_b.clone(),
                Some(out_b_h.clone()),
                rows,
                dim,
                eps,
            );
        }

        let a = read_f32(&client, &out_a, n);
        let b = read_f32(&client, &out_b, n);
        assert_eq!(a, b, "F2 f32 mismatch at rows={rows} dim={dim}");
        let ha = read_f16_bits(&client, &out_a_h, n);
        let hb = read_f16_bits(&client, &out_b_h, n);
        assert_eq!(ha, hb, "F2 f16 twin mismatch at rows={rows} dim={dim}");
    }
}

#[test]
fn f3_residual_norm_fused_is_bitwise_add_norm_cast() {
    let client = CubeCLContext::new().expect("gpu context").client();
    for (rows, dim) in [(6usize, 128usize), (11, 96), (2, 300)] {
        let (x0, _z, gamma) = fixture(rows, dim, 0x1173_0003);
        let mut rng = Rng(0x1173_0004);
        let residual: Vec<f32> = (0..rows * dim)
            .map(|_| (rng.next_f32() - 0.5) * 4.0)
            .collect();
        let n = rows * dim;
        let eps = eps();

        // Arm A — the unfused sequence: residual add (in-place on x) → norm
        // → cast.
        let x_a = client.create_from_slice(bytemuck::cast_slice(&x0));
        let res_a = client.create_from_slice(bytemuck::cast_slice(&residual));
        let out_a = client.create_from_slice(&vec![0u8; n * 4]);
        let out_a_h = client.create_from_slice(&vec![0u8; n * 2]);
        unsafe {
            ResidualAddCubeCL::launch::<ActiveRuntime>(&client, x_a.clone(), res_a, x_a.clone(), n);
            RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                x_a.clone(),
                client.create_from_slice(bytemuck::cast_slice(&gamma)),
                out_a.clone(),
                rows,
                dim,
                eps,
            );
            CastF32ToF16CubeCL::launch::<ActiveRuntime>(&client, out_a.clone(), out_a_h.clone(), n);
        }

        // Arm B — the fused kernel (x updated in place + norm + twin).
        let x_b = client.create_from_slice(bytemuck::cast_slice(&x0));
        let res_b = client.create_from_slice(bytemuck::cast_slice(&residual));
        let out_b = client.create_from_slice(&vec![0u8; n * 4]);
        let out_b_h = client.create_from_slice(&vec![0u8; n * 2]);
        unsafe {
            ResidualRmsNormTwinBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                x_b.clone(),
                res_b,
                client.create_from_slice(bytemuck::cast_slice(&gamma)),
                out_b.clone(),
                Some(out_b_h.clone()),
                rows,
                dim,
                eps,
            );
        }

        let xa = read_f32(&client, &x_a, n);
        let xb = read_f32(&client, &x_b, n);
        assert_eq!(
            xa, xb,
            "F3 in-place x mismatch at rows={rows} dim={dim}: the residual add drifted"
        );
        let a = read_f32(&client, &out_a, n);
        let b = read_f32(&client, &out_b, n);
        assert_eq!(a, b, "F3 norm_out f32 mismatch at rows={rows} dim={dim}");
        let ha = read_f16_bits(&client, &out_a_h, n);
        let hb = read_f16_bits(&client, &out_b_h, n);
        assert_eq!(ha, hb, "F3 f16 twin mismatch at rows={rows} dim={dim}");
    }
}
