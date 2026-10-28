//! Issue 1146 ladder 3 — G1 for the batched prefill embed.
//!
//! `DequantWteRowsCubeCL` (ONE launch, `p` rows) must equal `p` launches of
//! `DequantWteRowCubeCL` (the incumbent per-token arm) BIT-FOR-BIT, on a
//! synthetic ternary table with a non-multiple-of-256 width (the tail
//! workgroup's `col >= n` guard) and repeated / extreme token ids.

use riir_infer_gpu::{
    CubeCLContext, DequantWteRowCubeCL, DequantWteRowsCubeCL, create_f32, create_u32, read_f32,
};

/// Deterministic xorshift — no RNG dep, fixed seed.
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

#[test]
fn batched_embed_bit_identical_to_per_token() {
    let ctx = CubeCLContext::new().expect("CubeCL init");
    let client = ctx.client();

    // n = 5 groups of 128 minus 64: 576 cols → 3 workgroups of 256, ragged tail.
    let rows: u32 = 97;
    let n: u32 = 576;
    let blocks64 = n.div_ceil(64);
    let groups = n.div_ceil(128);

    let mut s = 0x1146_u64 ^ 0x9E37_79B9_7F4A_7C15;
    let words = (rows * blocks64 * 2) as usize;
    let mut pos = vec![0u32; words];
    let mut neg = vec![0u32; words];
    for i in 0..words {
        let a = xorshift(&mut s) as u32;
        let b = xorshift(&mut s) as u32;
        pos[i] = a & !b; // disjoint planes: ternary {-1, 0, +1}
        neg[i] = b & !a;
    }
    let scales: Vec<f32> = (0..rows * groups)
        .map(|_| (xorshift(&mut s) % 10_000) as f32 * 1e-4 + 1e-3)
        .collect();

    let pos_h = create_u32(&client, &pos);
    let neg_h = create_u32(&client, &neg);
    let scale_h = create_f32(&client, &scales);

    let toks: Vec<u32> = vec![0, 5, 5, 96, 13, 0, 42, 96, 1, 77, 3];
    let p = toks.len() as u32;
    let f = core::mem::size_of::<f32>();

    // Incumbent: one launch per token into its own output buffer.
    let mut want = Vec::with_capacity((p * n) as usize);
    for &t in &toks {
        let out = client.empty(n as usize * f);
        unsafe {
            DequantWteRowCubeCL::launch(
                &client,
                pos_h.clone(),
                neg_h.clone(),
                scale_h.clone(),
                out.clone(),
                t,
                blocks64,
                groups,
                n,
            );
        }
        want.extend(read_f32(&client, out).expect("read per-token row"));
    }

    // Batched: one launch, p rows.
    let toks_h = create_u32(&client, &toks);
    let out_b = client.empty((p * n) as usize * f);
    unsafe {
        DequantWteRowsCubeCL::launch(
            &client,
            pos_h,
            neg_h,
            scale_h,
            toks_h,
            out_b.clone(),
            rows,
            p,
            blocks64,
            groups,
            n,
        );
    }
    let got = read_f32(&client, out_b).expect("read batched");

    assert_eq!(got.len(), want.len());
    let nonzero = want.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > want.len() / 4,
        "fixture degenerate: {nonzero} nonzero"
    );
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "row {} col {}: batched {g} vs per-token {w}",
            i / n as usize,
            i % n as usize
        );
    }
}
