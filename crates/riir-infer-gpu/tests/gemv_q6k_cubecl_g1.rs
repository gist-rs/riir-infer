//! Issue 1150 follow-up 3 — the Q6_K tail-GEMV unit gate: the CubeCL/Metal
//! kernel vs the CPU canonical `gemv_q6_k` (riir-infer-core) on realistic
//! quantized rows.
//!
//! Two arms: a wide smoke (32 rows × 1024 cols, quantize-then-GEMV — the
//! real distribution chain) and a sign-exhaustive arm (every scale byte and
//! qh bit-field pattern exercised via all-ones/all-max fixtures — the i8
//! sign-extension and the 2-bit qh unpack are the two places a bit bug
//! hides). TOL 1e-4 relative — the re-association band (per-lane block
//! partials + one plane tree vs the CPU's sequential row walk).
//!
//! #![cfg] green-zero rule: the file carries its required-features row in
//! Cargo.toml ([[test]] gemv_q6k_cubecl_g1).

#![cfg(all(feature = "qwen38_wgpu", target_os = "macos"))]

use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;
use riir_infer_core::quant::q6k::{BlockQ6K, QK_K, gemv_q6_k, quantize_row_q6_k};
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::gemv_q6k_cubecl::{GemvQ6KCubeCL, repack_q6k_payload};

/// The re-association tolerance (the chunkwise family's band).
const TOL: f32 = 1e-4;

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn read(client: &ComputeClient<ActiveRuntime>, h: &Handle, n: usize) -> Vec<f32> {
    client
        .read_one(h.clone())
        .map(|b| bytemuck::cast_slice(&b)[..n].to_vec())
        .unwrap_or_default()
}

fn run_case(client: &ComputeClient<ActiveRuntime>, blocks: &[BlockQ6K], x: &[f32], m: usize) {
    let n = x.len();
    let (payload, d_sc) = repack_q6k_payload(blocks, m);
    let (w_h, d_h, x_h, out_h) = (
        client.create_from_slice(bytemuck::cast_slice(&payload)),
        client.create_from_slice(bytemuck::cast_slice(&d_sc)),
        client.create_from_slice(bytemuck::cast_slice(x)),
        client.create_from_slice(&vec![0u8; m * 4]),
    );
    unsafe {
        GemvQ6KCubeCL::launch::<ActiveRuntime>(&client, w_h, d_h, x_h, out_h.clone(), m, n);
    }
    let got = read(&client, &out_h, m);
    let want = gemv_q6_k(blocks, x, m, n);
    let mut max_rel = 0.0f32;
    for (g, r) in got.iter().zip(&want) {
        let denom = r.abs().max(1e-3);
        max_rel = max_rel.max(((g - r) / denom).abs());
    }
    println!("q6k gemv G1: m={m} n={n} max_rel {max_rel:.2e} (TOL {TOL})");
    assert!(
        max_rel < TOL,
        "q6k gemv rel err {max_rel} >= {TOL} — bit-layout or scale-sign bug"
    );
}

#[test]
fn q6k_gemv_matches_cpu_on_quantized_rows() {
    let client = CubeCLContext::new().expect("gpu context").client();
    let m = 32usize;
    let n = 1024usize; // 4 super-blocks per row

    let mut rng = Rng(0x1150_0003);
    let mut x = vec![0.0f32; n];
    for v in x.iter_mut() {
        // Realistic activation scale (the normed tail row: O(1) values).
        *v = rng.next_f32() * 2.0 - 1.0;
    }
    let mut blocks = vec![
        BlockQ6K {
            ql: [0; QK_K / 2],
            qh: [0; QK_K / 4],
            scales: [0; QK_K / 16],
            d: 0,
        };
        m * (n / QK_K)
    ];
    for (r, row) in blocks.chunks_mut(n / QK_K).enumerate() {
        let mut w = vec![0.0f32; n];
        for v in w.iter_mut() {
            // Per-row weight distribution (lm_head-ish: small weights, one
            // f16 super-scale per block).
            *v = (rng.next_f32() * 2.0 - 1.0) * (0.01 + 0.001 * (r as f32));
        }
        quantize_row_q6_k(&w, row);
    }
    run_case(&client, &blocks, &x, m);
}

#[test]
fn q6k_gemv_sign_and_bitfield_exhaustive() {
    // All-ones ql nibbles + all-max qh fields + max i8 scales (sign bit set)
    // — every extraction path (low/high nibble × 4 qh bit positions × i8
    // sign) hits its extreme. The value model w = d·sc·(q−32) with q=127,
    // sc=−128 → w = d·(−128)·95 — large magnitudes make a misread LOUD.
    let client = CubeCLContext::new().expect("gpu context").client();
    let m = 4usize;
    let n = QK_K; // one super-block per row
    let mut blocks = vec![
        BlockQ6K {
            ql: [0xFF; QK_K / 2],
            qh: [0xFF; QK_K / 4],
            scales: [-128i8; QK_K / 16],
            d: half::f16::from_f32(0.01).to_bits(),
        };
        m
    ];
    // Row 1 flips every sign: scales +127, qh zero, ql alternating nibbles.
    blocks[1].qh = [0x00; QK_K / 4];
    blocks[1].scales = [127i8; QK_K / 16];
    // Row 2: qh = 0b01010101 (alternating 2-bit fields 1,1,1,1), ql 0x0F/0xF0.
    blocks[2].qh = [0b0101_0101u8; QK_K / 4];
    for (i, b) in blocks[2].ql.iter_mut().enumerate() {
        *b = if i % 2 == 0 { 0x0F } else { 0xF0 };
    }
    // Row 3: minimum everything.
    blocks[3].ql = [0x00; QK_K / 2];
    blocks[3].scales = [-128i8; QK_K / 16];

    let mut x = vec![0.0f32; n];
    let mut rng = Rng(0x1150_0004);
    for v in x.iter_mut() {
        *v = rng.next_f32() * 2.0 - 1.0;
    }
    run_case(&client, &blocks, &x, m);
}
