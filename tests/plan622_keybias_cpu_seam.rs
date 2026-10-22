//! Plan 622 Phase 1 T1.1 acceptance — the per-key × per-head additive
//! attention-logit bias (CPU reference).
//!
//! Two laws, both at the `attention_heads_parallel_bias` level (the seam every
//! CPU forward routes through):
//!
//! 1. **None-identity**: `None` and an all-zero bias produce bit-identical
//!    outputs — the bias path never perturbs the incumbent arithmetic (no
//!    `+ 0.0`: that would flip `-0.0`).
//! 2. **It binds**: a non-zero bias moves the softmax weights toward the
//!    biased key and the output follows — the seam is live, not a no-op.

#![cfg(not(target_arch = "wasm32"))]

use riir_infer_core::transformer::{
    KeyBiasLayout, attention_heads_parallel, attention_heads_parallel_bias,
};

fn reference_out(
    softcap: f32,
    n_head: usize,
    n_kv: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
) -> Vec<f32> {
    // Deterministic q / k / v — small primes scaled so scores land in a
    // numerically interesting range (no saturation, no denormals).
    let q: Vec<f32> = (0..n_head * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.25)
        .collect();
    let key: Vec<f32> = (0..t_n * kv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
        .collect();
    let value: Vec<f32> = (0..t_n * kv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();
    let mut out = vec![0.0f32; n_head * hd];
    let mut scores = vec![0.0f32; n_head * 256];
    let scale = 1.0 / (hd as f32).sqrt();
    unsafe {
        attention_heads_parallel(
            &q,
            &key,
            &value,
            &mut out,
            &mut scores,
            n_head,
            n_kv,
            kv_dim,
            hd,
            t_n,
            scale,
            softcap,
            256,
        );
    }
    out
}

fn biased_out(
    bias: Option<(&[f32], KeyBiasLayout)>,
    softcap: f32,
    n_head: usize,
    n_kv: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
) -> Vec<f32> {
    let q: Vec<f32> = (0..n_head * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.25)
        .collect();
    let key: Vec<f32> = (0..t_n * kv_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
        .collect();
    let value: Vec<f32> = (0..t_n * kv_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();
    let mut out = vec![0.0f32; n_head * hd];
    let mut scores = vec![0.0f32; n_head * 256];
    let scale = 1.0 / (hd as f32).sqrt();
    unsafe {
        attention_heads_parallel_bias(
            &q,
            &key,
            &value,
            &mut out,
            &mut scores,
            n_head,
            n_kv,
            kv_dim,
            hd,
            t_n,
            scale,
            softcap,
            256,
            bias,
        );
    }
    out
}

fn assert_same_shape(n_head: usize, n_kv: usize, kv_dim: usize, hd: usize, t_n: usize) {
    assert_eq!(kv_dim, n_kv * hd);
    let _ = (n_head, t_n);
}

#[test]
fn none_and_all_zero_bias_are_bit_identical_both_softcap_arms() {
    let (n_head, n_kv, hd, t_n) = (4usize, 2usize, 16usize, 37usize);
    let kv_dim = n_kv * hd;
    assert_same_shape(n_head, n_kv, kv_dim, hd, t_n);

    for softcap in [0.0f32, 30.0] {
        let reference = reference_out(softcap, n_head, n_kv, kv_dim, hd, t_n);
        let none = biased_out(None, softcap, n_head, n_kv, kv_dim, hd, t_n);
        assert_eq!(reference.len(), none.len());
        for (r, n) in reference.iter().zip(&none) {
            assert_eq!(
                r.to_bits(),
                n.to_bits(),
                "softcap={softcap}: None != incumbent"
            );
        }

        let zeros = vec![0.0f32; n_head * t_n];
        let zero = biased_out(
            Some((&zeros, KeyBiasLayout { head_stride: t_n })),
            softcap,
            n_head,
            n_kv,
            kv_dim,
            hd,
            t_n,
        );
        for (r, z) in reference.iter().zip(&zero) {
            assert_eq!(
                r.to_bits(),
                z.to_bits(),
                "softcap={softcap}: zero-bias != incumbent"
            );
        }
    }
}

#[test]
fn nonzero_bias_binds_and_routes_toward_the_biased_key() {
    let (n_head, n_kv, hd, t_n) = (2usize, 1usize, 16usize, 9usize);
    let kv_dim = n_kv * hd;
    let softcap = 0.0f32;

    let reference = reference_out(softcap, n_head, n_kv, kv_dim, hd, t_n);
    // Large positive bias on head 0, key slot 3; zero elsewhere. The softmax
    // must shift weight toward key 3 and the output must move.
    let mut bias = vec![0.0f32; n_head * t_n];
    bias[3] = 12.0;
    let out = biased_out(
        Some((&bias, KeyBiasLayout { head_stride: t_n })),
        softcap,
        n_head,
        n_kv,
        kv_dim,
        hd,
        t_n,
    );
    let moved: usize = reference
        .iter()
        .zip(&out)
        .filter(|(r, o)| r.to_bits() != o.to_bits())
        .count();
    // Head 0 occupies out[0..hd]; head 1 (bias row zero) must be untouched.
    assert!(
        out[..hd]
            .iter()
            .zip(&reference[..hd])
            .all(|(o, r)| o.to_bits() != r.to_bits()),
        "head 0 output must move under a +12 logit bias on key 3"
    );
    assert!(
        out[hd..]
            .iter()
            .zip(&reference[hd..])
            .all(|(o, r)| o.to_bits() == r.to_bits()),
        "head 1 (zero bias row) must stay bit-identical"
    );
    assert!(
        moved >= hd,
        "at least head 0's {hd} outputs move; got {moved}"
    );
}

#[test]
fn per_head_rows_are_independent_gqa() {
    // GQA: 4 Q heads over 2 KV groups; the bias is per Q HEAD. Distinct bias
    // rows must produce distinct outputs per head even when heads share a KV
    // group — the layout contract (head_stride rows, slot index).
    let (n_head, n_kv, hd, t_n) = (4usize, 2usize, 16usize, 13usize);
    let kv_dim = n_kv * hd;
    let softcap = 0.0f32;
    let reference = reference_out(softcap, n_head, n_kv, kv_dim, hd, t_n);

    let mut bias = vec![0.0f32; n_head * t_n];
    bias[2] = 9.0; // head 0 (row 0), key slot 2
    bias[3 * t_n + 5] = -9.0; // head 3 (same KV group as head 1? irrelevant — per Q head)
    let out = biased_out(
        Some((&bias, KeyBiasLayout { head_stride: t_n })),
        softcap,
        n_head,
        n_kv,
        kv_dim,
        hd,
        t_n,
    );
    for h in 0..n_head {
        let (o, r) = (&out[h * hd..(h + 1) * hd], &reference[h * hd..(h + 1) * hd]);
        let moved = o.iter().zip(r).any(|(a, b)| a.to_bits() != b.to_bits());
        match h {
            0 | 3 => assert!(moved, "head {h} carries a non-zero bias row and must move"),
            _ => assert!(
                !moved,
                "head {h} has a zero bias row and must be bit-identical"
            ),
        }
    }
}
