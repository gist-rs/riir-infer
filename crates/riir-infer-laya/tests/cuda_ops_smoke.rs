//! CUDA-vs-Cpu op equivalence (`.issues/002`) — the `metal_ops_smoke`
//! pattern on the non-macOS half. Runs only under `laya-riir-cuda` on a
//! non-macOS host; the [[test]] row keeps the green-zero rule honest. The
//! forward G5 gate (consumer-side `laya_riir_parity`) is the standing
//! correctness authority; this file pins OP-level equivalence so a bad
//! kernel is diagnosable without re-deriving it from a red G5.
//!
//! Every test holds a file-wide GPU lock for its body and opens each
//! independent arm with `begin_pass()` — the consumer repo's issue-015
//! epoch contract: the chain cache keys device slots by `(ptr, len,
//! EPOCH)`, and the epoch only moves inside `begin_pass`, so a standalone
//! op arm that never begins a pass could silently hit a recycled
//! address's stale slot.

#![cfg(all(not(target_os = "macos"), feature = "laya-riir-cuda"))]

use std::sync::{Mutex, MutexGuard, OnceLock};

use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::cuda::Cuda;

/// Serialize test BODIES (at most ONE live CUDA context + stream at a
/// time); poison-recovering so one panic does not cascade.
fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lcg() -> impl FnMut() -> f32 {
    let mut s = 0x1234_5678u32;
    move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f32) / 8_388_608.0 - 1.0
    }
}

fn vec_of(n: usize) -> Vec<f32> {
    let mut r = lcg();
    (0..n).map(|_| r()).collect()
}

fn report(name: &str, a: &[f32], b: &[f32], tol: f32) {
    let max = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = a.iter().fold(0.0f32, |acc, v| acc.max(*v)).abs().max(1e-9);
    println!("{name}: max abs {max:.4e} · scale {scale:.3e} · tol {tol:.1e}");
    assert!(
        max <= tol * scale.max(1e-3),
        "{name}: DIVERGED max {max:.4e}"
    );
}

/// Sync a written slice's device result out (src = the written slice,
/// out = a fresh host buffer — the forward body's exact shape).
fn sync_out(g: &Cuda, buf: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; buf.len()];
    g.download_into(buf, &mut out);
    out
}

#[test]
fn cuda_ops_match_cpu_op_by_op() {
    let _gpu = gpu_lock();
    let g = Cuda::new().expect("cuda backend");
    let c = Cpu;

    // GEMM — the stride shapes at forward-like + ragged sizes, including
    // m/n tails past the 64-tile, a k tail past BK 32, and the m=1 head
    // shapes (the act head's [1, 257]×[257, 129]-class calls).
    for (mm, k, n) in [
        (25usize, 768usize, 2304usize),
        (7, 64, 33),
        (1, 257, 129),
        (300, 100, 700),
        (317, 1024, 5248), // the encoder Wi shape at the bench seq
        (100, 2624, 1024), // the encoder mlp-Wo shape
    ] {
        // Each arm consumes fresh host fixtures → its own pass (the epoch
        // contract; see the module doc).
        g.begin_pass();
        let a = vec_of(mm * k);
        let b = vec_of(k * n);
        let mut dc = vec![0f32; mm * n];
        let mut dg = vec![0f32; mm * n];
        c.matmul(&a, 0, mm, k, &b, 0, n, &mut dc, 0);
        g.matmul(&a, 0, mm, k, &b, 0, n, &mut dg, 0);
        report(
            &format!("matmul {mm}x{k}x{n}"),
            &dc,
            &sync_out(&g, &dg),
            1e-3,
        );

        let w = vec_of(n * k);
        let mut wc = vec![0f32; mm * n];
        let mut wg = vec![0f32; mm * n];
        c.matmul_w(&a, mm, k, &w, n, &mut wc);
        g.matmul_w(&a, mm, k, &w, n, &mut wg);
        report(
            &format!("matmul_w {mm}x{k}x{n}"),
            &wc,
            &sync_out(&g, &wg),
            1e-3,
        );
    }
    {
        let (mm, hd) = (17usize, 64usize);
        let q = vec_of(mm * hd);
        let k = vec_of(mm * hd);
        let mut sc = vec![0f32; mm * mm];
        let mut sg = vec![0f32; mm * mm];
        c.matmul_kt(&q, 0, mm, hd, &k, 0, &mut sc, 0);
        g.begin_pass();
        g.matmul_kt(&q, 0, mm, hd, &k, 0, &mut sg, 0);
        report("matmul_kt", &sc, &sync_out(&g, &sg), 1e-3);
    }

    // The batched-head attention ops — ragged seq on purpose (37: every
    // tile edge; 300: the wide batched path with a ragged n tail).
    for seq in [37usize, 300usize] {
        let heads = if seq == 300 { 2 } else { 4 };
        let hd = 64usize;
        g.begin_pass();
        let q = vec_of(heads * seq * hd);
        let k = vec_of(heads * seq * hd);
        let mut sc = vec![0f32; heads * seq * seq];
        let mut sg = vec![0f32; heads * seq * seq];
        c.matmul_kt_heads(&q, &k, heads, seq, hd, &mut sc);
        g.matmul_kt_heads(&q, &k, heads, seq, hd, &mut sg);
        report(
            &format!("matmul_kt_heads seq{seq}"),
            &sc,
            &sync_out(&g, &sg),
            1e-3,
        );

        let v = vec_of(heads * seq * hd);
        let probs = vec_of(heads * seq * seq);
        let mut cc = vec![0f32; heads * seq * hd];
        let mut cg = vec![0f32; heads * seq * hd];
        c.matmul_heads(&probs, &v, heads, seq, seq, hd, &mut cc);
        g.matmul_heads(&probs, &v, heads, seq, seq, hd, &mut cg);
        report(
            &format!("matmul_heads seq{seq}"),
            &cc,
            &sync_out(&g, &cg),
            1e-3,
        );
    }
    {
        // The sliding-window mask broadcast (heads slabs of the parent).
        let (seq, heads) = (37usize, 4usize);
        g.begin_pass();
        let mask = vec_of(seq * seq);
        let probs = vec_of(heads * seq * seq);
        let mut xc = probs.clone();
        let mut xg = probs.clone();
        c.add_mask_broadcast(&mut xc, &mask, heads);
        g.add_mask_broadcast(&mut xg, &mask, heads);
        report("add_mask_broadcast", &xc, &sync_out(&g, &xg), 1e-6);
    }

    // Elementwise.
    let n = 1000;
    g.begin_pass();
    let x = vec_of(n);
    let y = vec_of(n);
    let mut xc = x.clone();
    let mut xg = x.clone();
    let xc_len = xc.len();
    c.add(&mut xc, 0, &y, 0, xc_len);
    let xg_len = xg.len();
    g.add(&mut xg, 0, &y, 0, xg_len);
    report("add", &xc, &sync_out(&g, &xg), 1e-6);

    let (d, rows) = (64usize, 15usize);
    g.begin_pass();
    let xb = vec_of(rows * d);
    let bias = vec_of(d);
    let mut xc = xb.clone();
    let mut xg = xb.clone();
    c.add_bias_row(&mut xc, d, &bias);
    g.add_bias_row(&mut xg, d, &bias);
    report("add_bias_row", &xc, &sync_out(&g, &xg), 1e-6);

    let mut xc = xb.clone();
    let mut xg = xb.clone();
    c.scale(&mut xc, 0.125);
    g.scale(&mut xg, 0.125);
    report("scale", &xc, &sync_out(&g, &xg), 1e-6);

    let mut xc = xb.clone();
    let mut xg = xb.clone();
    c.relu(&mut xc);
    g.relu(&mut xg);
    report("relu", &xc, &sync_out(&g, &xg), 1e-6);

    let mut xc = xb.clone();
    let mut xg = xb.clone();
    c.gelu_erf(&mut xc);
    g.gelu_erf(&mut xg);
    report("gelu_erf (libm vs CUDA erff)", &xc, &sync_out(&g, &xg), 1e-4);

    let (r2, i_sz) = (5usize, 32usize);
    g.begin_pass();
    let fused = vec_of(r2 * 2 * i_sz);
    let mut oc = vec![0f32; r2 * i_sz];
    let mut og = vec![0f32; r2 * i_sz];
    c.glu_gelu_gate(&fused, r2, i_sz, &mut oc);
    g.glu_gelu_gate(&fused, r2, i_sz, &mut og);
    report("glu_gelu_gate", &oc, &sync_out(&g, &og), 1e-4);

    // LN + softmax (reduction order differs by design — loose tol).
    let w = vec_of(d);
    g.begin_pass();
    let mut oc = vec![0f32; rows * d];
    let mut og = vec![0f32; rows * d];
    let mut sq = Vec::new();
    c.layer_norm_nobias_into(&xb, &w, 1e-5, d, &mut sq, &mut oc);
    let mut sq2 = Vec::new();
    g.layer_norm_nobias_into(&xb, &w, 1e-5, d, &mut sq2, &mut og);
    report("layer_norm", &oc, &sync_out(&g, &og), 1e-3);

    let mut xc = xb.clone();
    let mut xg = xb.clone();
    c.softmax_rows(&mut xc, d);
    g.softmax_rows(&mut xg, d);
    report("softmax_rows", &xc, &sync_out(&g, &xg), 1e-3);

    // rope / split / merge / gather (exact-ish data movement).
    let (seq, heads, hd) = (9usize, 4usize, 64usize);
    g.begin_pass();
    let q = vec_of(heads * seq * hd);
    let (cos, sin) = riir_infer_laya::laya::riir::ops::rope_tables(seq, hd, 160_000.0);
    let mut qc = q.clone();
    let mut qg = q.clone();
    c.apply_rope(&mut qc, seq, heads, hd, &cos, &sin);
    g.apply_rope(&mut qg, seq, heads, hd, &cos, &sin);
    report("apply_rope", &qc, &sync_out(&g, &qg), 1e-4);

    let row_stride = 3 * 384;
    let src = vec_of(seq * row_stride);
    let mut oc = vec![0f32; heads * seq * hd];
    let mut og = vec![0f32; heads * seq * hd];
    c.split_heads(&src, row_stride, 384, seq, heads, hd, &mut oc);
    g.split_heads(&src, row_stride, 384, seq, heads, hd, &mut og);
    report("split_heads", &oc, &sync_out(&g, &og), 1e-6);

    let mut oc2 = vec![0f32; seq * heads * hd];
    let mut og2 = vec![0f32; seq * heads * hd];
    // merge consumes the SYNCED split output (og's own host bytes are a
    // stale handle under the lazy-sync contract — sync_out returns a copy).
    let og_synced = sync_out(&g, &og);
    c.merge_heads(&oc, seq, heads, hd, &mut oc2);
    g.merge_heads(&og_synced, seq, heads, hd, &mut og2);
    report("merge_heads", &oc2, &sync_out(&g, &og2), 1e-6);

    let markers = [3usize, 1, 7, 0];
    let mut oc3 = vec![0f32; markers.len() * d];
    let mut og3 = vec![0f32; markers.len() * d];
    c.gather_rows(&xb, d, &markers, &mut oc3);
    g.gather_rows(&xb, d, &markers, &mut og3);
    report("gather_rows", &oc3, &sync_out(&g, &og3), 1e-6);

    // copy_into (the layer-0 identity path).
    g.begin_pass();
    let src2 = vec_of(64);
    let mut dc2 = vec![0f32; 64];
    let mut dg2 = vec![0f32; 64];
    c.copy_into(&src2, &mut dc2);
    g.copy_into(&src2, &mut dg2);
    report("copy_into", &dc2, &sync_out(&g, &dg2), 1e-6);

    // download_into across syncs: the epoch bump must not lose the buffer,
    // and a recycled dst slice must serve its OWN second write.
    g.begin_pass();
    let a0 = vec_of(16); // m*k = 2*8
    let w0 = vec_of(16); // n*k = 2*8
    let mut dg = vec![0f32; 4]; // m*n
    g.matmul_w(&a0, 2, 8, &w0, 2, &mut dg);
    let d1 = sync_out(&g, &dg); // sync 1
    let mut dg2 = vec![0f32; 4];
    g.matmul_w(&a0, 2, 8, &w0, 2, &mut dg2);
    let d2 = sync_out(&g, &dg2); // sync 2 — recycled dst slice
    assert_eq!(d1, d2, "recycled dst slice diverged across syncs");
    println!("cross-sync download: ok");

    // The prefix download (the head's CLS row — a leading d of a seq·d
    // slot; matched by base pointer + sufficient extent).
    g.begin_pass();
    let full = vec_of(9 * 64);
    let mut slot = vec![0f32; 9 * 64];
    g.copy_into(&full, &mut slot);
    let mut cls = vec![0f32; 64];
    g.download_into(&slot[..64], &mut cls);
    assert_eq!(cls, &full[..64], "prefix download diverged");
    println!("prefix download: ok");
}

/// The fused-attention seam (.issues/003): `attention_forward` equivalence
/// vs the CPU default op sequence, full attention, ragged + padded tile
/// sizes (9/37 walk every tile edge; 129 = FBQ·4+1; 1 = the single-key
/// row) — the metal_ops_smoke mirrors.
#[test]
fn cuda_fused_attention_matches_cpu_full() {
    let _gpu = gpu_lock();
    let g = Cuda::new().expect("cuda backend");
    let c = Cpu;
    let hd = 64usize;
    for (seq, heads) in [(1usize, 4usize), (9, 4), (37, 4), (64, 2), (129, 2)] {
        g.begin_pass();
        let d = heads * hd;
        let qkv = vec_of(seq * 3 * d);
        let (cos, sin) = riir_infer_laya::laya::riir::ops::rope_tables(seq, hd, 160_000.0);
        let scale = (hd as f32).sqrt().recip();
        let mut sa = riir_infer_laya::laya::riir::backend::AttnScratch::default();
        let mut sb = riir_infer_laya::laya::riir::backend::AttnScratch::default();
        let mut oc = vec![0f32; seq * d];
        let mut og = vec![0f32; seq * d];
        c.attention_forward(
            &qkv, 0, &cos, &sin, 0, scale, seq, heads, hd, usize::MAX, None, &mut sa, &mut oc, 0,
        );
        g.attention_forward(
            &qkv, 0, &cos, &sin, 0, scale, seq, heads, hd, usize::MAX, None, &mut sb, &mut og, 0,
        );
        report(
            &format!("attn full {seq}x{heads}"),
            &oc,
            &sync_out(&g, &og),
            1e-3,
        );
    }
}

/// Sliding-window equivalence: the CPU lane consumes the additive mask
/// tensor, the fused kernel predicates on the window — the SAME allowed
/// set must come out. Window 8 at seq 64, window 4 ragged at seq 37,
/// window 16 at seq 130.
#[test]
fn cuda_fused_attention_matches_cpu_sliding() {
    let _gpu = gpu_lock();
    let g = Cuda::new().expect("cuda backend");
    let c = Cpu;
    let hd = 64usize;
    for (seq, heads, window) in [(64usize, 4usize, 8usize), (37, 4, 4), (130, 2, 16)] {
        g.begin_pass();
        let d = heads * hd;
        let qkv = vec_of(seq * 3 * d);
        let (cos, sin) = riir_infer_laya::laya::riir::ops::rope_tables(seq, hd, 160_000.0);
        let scale = (hd as f32).sqrt().recip();
        // the encoder's mask: [seq, seq], f32::MIN outside the window
        let mut mask = vec![f32::MIN; seq * seq];
        for qi in 0..seq {
            let lo = qi.saturating_sub(window);
            let hi = (qi + window).min(seq - 1);
            for kv in lo..=hi {
                mask[qi * seq + kv] = 0.0;
            }
        }
        let mut sa = riir_infer_laya::laya::riir::backend::AttnScratch::default();
        let mut sb = riir_infer_laya::laya::riir::backend::AttnScratch::default();
        let mut oc = vec![0f32; seq * d];
        let mut og = vec![0f32; seq * d];
        c.attention_forward(
            &qkv, 0, &cos, &sin, 0, scale, seq, heads, hd, window, Some(&mask), &mut sa, &mut oc, 0,
        );
        g.attention_forward(
            &qkv, 0, &cos, &sin, 0, scale, seq, heads, hd, window, Some(&mask), &mut sb, &mut og, 0,
        );
        report(
            &format!("attn slide {seq}x{heads} w{window}"),
            &oc,
            &sync_out(&g, &og),
            1e-3,
        );
    }
}

/// The PACKED offsets (.issues/003's founding defect): a two-sequence
/// packed dispatch (non-zero qkv/rope/out offsets) must equal the same
/// two sequences dispatched at offset zero — the exact shape the
/// trait-default path silently fed zeros at v1. Also exercises the
/// sliding window at a packed offset.
#[test]
fn cuda_fused_attention_packed_offsets_match_singles() {
    let _gpu = gpu_lock();
    let g = Cuda::new().expect("cuda backend");
    let c = Cpu;
    let hd = 64usize;
    let heads = 2usize;
    let d = heads * hd;
    let (s1, s2, window) = (21usize, 33usize, 12usize);
    let total = s1 + s2;
    g.begin_pass();
    let qkv = vec_of(total * 3 * d);
    let (cos, sin) = riir_infer_laya::laya::riir::ops::rope_tables(total, hd, 160_000.0);
    let scale = (hd as f32).sqrt().recip();
    let mut mask1 = vec![f32::MIN; s1 * s1];
    for qi in 0..s1 {
        let lo = qi.saturating_sub(window);
        let hi = (qi + window).min(s1 - 1);
        for kv in lo..=hi {
            mask1[qi * s1 + kv] = 0.0;
        }
    }
    let mut mask2 = vec![f32::MIN; s2 * s2];
    for qi in 0..s2 {
        let lo = qi.saturating_sub(window);
        let hi = (qi + window).min(s2 - 1);
        for kv in lo..=hi {
            mask2[qi * s2 + kv] = 0.0;
        }
    }

    // CPU reference per sequence (the mask-consuming default sequence).
    let mut oc = vec![0f32; total * d];
    let mut sa = riir_infer_laya::laya::riir::backend::AttnScratch::default();
    c.attention_forward(
        &qkv[..s1 * 3 * d], 0, &cos, &sin, 0, scale, s1, heads, hd, window, Some(&mask1), &mut sa,
        &mut oc[..s1 * d], 0,
    );
    c.attention_forward(
        &qkv[s1 * 3 * d..], 0, &cos[s1 * hd..], &sin[s1 * hd..], 0, scale, s2, heads, hd, window,
        Some(&mask2), &mut sa, &mut oc[s1 * d..], 0,
    );

    // CUDA packed: ONE parent dispatch per sequence at its offsets (the
    // whole `out` parent every time, offset at bind — the encoder's exact
    // call shape, so every sequence shares ONE chain slot).
    let mut og = vec![0f32; total * d];
    let mut sb = riir_infer_laya::laya::riir::backend::AttnScratch::default();
    g.attention_forward(
        &qkv, 0, &cos, &sin, 0, scale, s1, heads, hd, window, Some(&mask1), &mut sb, &mut og, 0,
    );
    g.attention_forward(
        &qkv, s1 * 3 * d, &cos, &sin, s1, scale, s2, heads, hd, window, Some(&mask2), &mut sb,
        &mut og, s1 * d,
    );
    report("attn packed offsets", &oc, &sync_out(&g, &og), 1e-3);
}
