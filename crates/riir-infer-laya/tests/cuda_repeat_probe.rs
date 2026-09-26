//! Issue 021 T1 — the CUDA repeat probe: every kernel at banking77's real
//! shapes (the long-sequence ~317-token, 77-way class — ModernBERT-large
//! english geometry: d 1024, intermediate 2624, 16 heads × hd 64, sliding
//! window 64), run REPEATEDLY with identical inputs and bit-compared
//! against run 0. The harness repeat check fired once at this suite
//! (riir-reflex Bench 052 §4090); this probe is the kernel-level form of
//! the same question: is any single op scheduling-nondeterministic at
//! these shapes, or is the flip above the op layer?
//!
//! Protocol per arm: the body closure restores the pristine input bytes →
//! `begin_pass()` (the epoch contract — fresh uploads + fresh slots, the
//! harness's real per-forward cadence) → op → `download_into` → exact
//! bit-compare vs the golden (call 0). A mismatch prints the arm, the
//! repeat index, and the max abs delta.
//!
//! Runs only under `laya-riir-cuda` on a non-macOS host; the [[test]] row
//! keeps the green-zero rule honest (the `cuda_ops_smoke` posture).

#![cfg(all(not(target_os = "macos"), feature = "laya-riir-cuda"))]

use std::sync::{Mutex, MutexGuard, OnceLock};

use riir_infer_laya::laya::riir::backend::{AttnScratch, Backend};
use riir_infer_laya::laya::riir::cuda::Cuda;

/// Serialize test bodies (at most ONE live CUDA context + stream at a
/// time); poison-recovering so one panic does not cascade.
fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Deterministic input stream (the `cuda_ops_smoke` LCG).
fn lcg(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f32) / 8_388_608.0 - 1.0
    }
}

fn vec_of(n: usize, seed: u32) -> Vec<f32> {
    let mut r = lcg(seed);
    (0..n).map(|_| r()).collect()
}

fn download(g: &Cuda, src: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    g.download_into(src, &mut out);
    out
}

/// The verdict: first differing element (index, |a-b|) between two runs.
fn first_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32)> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x.to_bits() != y.to_bits())
        .map(|i| (i, (a[i] - b[i]).abs()))
}

/// Run `body` once for the golden, then REPEATS times, bit-comparing each
/// result. The body owns its whole reset→pass→op→download cycle.
fn repeat_arm(
    rows: &mut Vec<String>,
    failed: &mut bool,
    name: &str,
    repeats: usize,
    mut body: impl FnMut() -> Vec<f32>,
) {
    let golden = body();
    let mut bad = 0usize;
    let mut first: Option<(usize, usize, f32)> = None; // (repeat, idx, |Δ|)
    for rep in 1..=repeats {
        let got = body();
        if let Some((idx, delta)) = first_diff(&golden, &got) {
            bad += 1;
            if first.is_none() {
                first = Some((rep, idx, delta));
            }
        }
    }
    match first {
        None => rows.push(format!("✓ {name}: {repeats} repeats bit-stable")),
        Some((rep, idx, delta)) => {
            *failed = true;
            rows.push(format!(
                "✗ {name}: {bad}/{repeats} repeats DIVERGED — first at rep \
                 {rep}, idx {idx} (of {}), max |Δ| {delta:.3e}",
                golden.len()
            ));
        }
    }
}

const SEQ: usize = 317;
const D: usize = 1024;
const I_SZ: usize = 2624;
const HEADS: usize = 16;
const HD: usize = 64;
const WINDOW: usize = 64;
const REPEATS: usize = 200;

#[test]
fn cuda_ops_repeat_bit_stable_at_banking77_shapes() {
    let _gpu = gpu_lock();
    let g = Cuda::new().expect("cuda backend");
    let mut rows: Vec<String> = Vec::new();
    let mut failed = false;

    // ── matmul_w — the projection family ────────────────────────────────
    // qkv proj: [317,1024] × w[3072,1024] — xwide_reg4 at this m.
    {
        let a = vec_of(SEQ * D, 1);
        let w = vec_of(3 * D * D, 2);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; SEQ * 3 * D];
        repeat_arm(
            &mut rows,
            &mut failed,
            "matmul_w qkv [317,1024]×[3072,1024]",
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, SEQ, D, &w, 3 * D, &mut dst);
                download(&g, &dst)
            },
        );
    }
    // fused gate+up: [317,1024] × w[5248,1024] — xwide_reg4.
    {
        let a = vec_of(SEQ * D, 3);
        let w = vec_of(2 * I_SZ * D, 4);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; SEQ * 2 * I_SZ];
        repeat_arm(
            &mut rows,
            &mut failed,
            "matmul_w gate/up [317,1024]×[5248,1024]",
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, SEQ, D, &w, 2 * I_SZ, &mut dst);
                download(&g, &dst)
            },
        );
    }
    // down proj: [317,2624] × w[1024,2624] — wide at this m.
    {
        let a = vec_of(SEQ * I_SZ, 5);
        let w = vec_of(D * I_SZ, 6);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; SEQ * D];
        repeat_arm(
            &mut rows,
            &mut failed,
            "matmul_w down [317,2624]×[1024,2624]",
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, SEQ, I_SZ, &w, D, &mut dst);
                download(&g, &dst)
            },
        );
    }
    // ladder-boundary shapes: narrow (m 200), narrow tail (m 33), xwide
    // row tail (m 321), xwide col tail (n 2080) — the smoke file's grid.
    for (mm, nn, tag) in [
        (200usize, 1024usize, "narrow m=200"),
        (33, 1024, "narrow tail m=33"),
        (321, 2624, "xwide row tail m=321"),
        (317, 2080, "xwide col tail n=2080"),
    ] {
        let a = vec_of(mm * D, 7);
        let w = vec_of(nn * D, 8);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; mm * nn];
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("matmul_w {tag} [{mm},{D}]×[{nn},{D}]"),
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, mm, D, &w, nn, &mut dst);
                download(&g, &dst)
            },
        );
    }

    // ── flash attention — sliding + full, the banking77 windows ────────
    for (win, tag) in [(WINDOW, "sliding w=64"), (usize::MAX, "full")] {
        let qkv = vec_of(SEQ * 3 * D, 9);
        let mut cos = vec![0f32; SEQ * HD];
        let mut sin = vec![0f32; SEQ * HD];
        {
            let mut r = lcg(10);
            for c in cos.iter_mut() {
                *c = r();
            }
            for s in sin.iter_mut() {
                *s = r();
            }
        }
        let mut out = vec![0f32; SEQ * D];
        let mut scratch = AttnScratch::default();
        let mut wq = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("flash_attn seq={SEQ} h={HEADS} hd={HD} {tag}"),
            REPEATS,
            || {
                wq = qkv.clone();
                out.iter_mut().for_each(|v| *v = 0.0);
                scratch = AttnScratch::default();
                g.begin_pass();
                g.attention_forward(
                    &wq,
                    0,
                    &cos,
                    &sin,
                    0,
                    1.0 / (HD as f32).sqrt(),
                    SEQ,
                    HEADS,
                    HD,
                    win,
                    None,
                    &mut scratch,
                    &mut out,
                    0,
                );
                download(&g, &out)
            },
        );
    }

    // ── batched head GEMMs (the reference attention's score + mix) ──────
    {
        let q = vec_of(HEADS * SEQ * HD, 11);
        let k = vec_of(HEADS * SEQ * HD, 12);
        let mut wq = Vec::new();
        let mut dst = vec![0f32; HEADS * SEQ * SEQ];
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("matmul_kt_heads h={HEADS} m={SEQ} hd={HD}"),
            60, // 16×317×317 dst — fewer repeats keep the run short
            || {
                wq = q.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_kt_heads(&wq, &k, HEADS, SEQ, HD, &mut dst);
                download(&g, &dst)
            },
        );
    }
    {
        let a = vec_of(HEADS * SEQ * HD, 13);
        let b = vec_of(HEADS * HD * HD, 14);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; HEADS * SEQ * HD];
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("matmul_heads h={HEADS} m={SEQ} k={HD} n={HD}"),
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_heads(&wa, &b, HEADS, SEQ, HD, HD, &mut dst);
                download(&g, &dst)
            },
        );
    }

    // ── reductions: ln_rows + softmax_rows ──────────────────────────────
    {
        let x = vec_of(SEQ * D, 15);
        let w: Vec<f32> = (0..D).map(|i| 0.5f32 + (i % 7) as f32 * 0.1).collect();
        let mut out = vec![0f32; SEQ * D];
        let mut wx = Vec::new();
        let mut sq = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("ln_rows rows={SEQ} d={D}"),
            REPEATS,
            || {
                wx = x.clone();
                out.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.layer_norm_nobias_into(&wx, &w, 1e-12, D, &mut sq, &mut out);
                download(&g, &out)
            },
        );
    }
    {
        let rows_n = HEADS * SEQ; // the reference-attention softmax shape
        let n = SEQ;
        let x = vec_of(rows_n * n, 16);
        let mut wx = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("softmax_rows rows={rows_n} n={n}"),
            60,
            || {
                wx = x.clone();
                g.begin_pass();
                g.softmax_rows(&mut wx, n);
                download(&g, &wx)
            },
        );
    }

    // ── elementwise + gather + head splits ──────────────────────────────
    {
        let q = vec_of(HEADS * SEQ * HD, 17);
        let cos = vec_of(SEQ * HD, 18);
        let sin = vec_of(SEQ * HD, 19);
        let mut wq = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("rope [{HEADS},{SEQ},{HD}]"),
            REPEATS,
            || {
                wq = q.clone();
                g.begin_pass();
                g.apply_rope(&mut wq, SEQ, HEADS, HD, &cos, &sin);
                download(&g, &wq)
            },
        );
    }
    {
        let fused = vec_of(SEQ * 2 * I_SZ, 20);
        let mut wf = Vec::new();
        let mut out = vec![0f32; SEQ * I_SZ];
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("glu_gelu_gate rows={SEQ} i_sz={I_SZ}"),
            REPEATS,
            || {
                wf = fused.clone();
                out.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.glu_gelu_gate(&wf, SEQ, I_SZ, &mut out);
                download(&g, &out)
            },
        );
    }
    {
        let x = vec_of(SEQ * I_SZ, 21);
        let mut wx = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("gelu_erf len={}", SEQ * I_SZ),
            REPEATS,
            || {
                wx = x.clone();
                g.begin_pass();
                g.gelu_erf(&mut wx);
                download(&g, &wx)
            },
        );
    }
    {
        let x = vec_of(SEQ * D, 22);
        let y = vec_of(SEQ * D, 23);
        let mut wx = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("add len={}", SEQ * D),
            REPEATS,
            || {
                wx = x.clone();
                g.begin_pass();
                g.add(&mut wx, 0, &y, 0, SEQ * D);
                download(&g, &wx)
            },
        );
    }
    {
        let x = vec_of(SEQ * 3 * D, 24);
        let bias = vec_of(3 * D, 25);
        let mut wx = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("add_bias_row len={} d={}", SEQ * 3 * D, 3 * D),
            REPEATS,
            || {
                wx = x.clone();
                g.begin_pass();
                g.add_bias_row(&mut wx, 3 * D, &bias);
                download(&g, &wx)
            },
        );
    }
    {
        let x = vec_of(4 * D, 26);
        let rows_idx: Vec<usize> = {
            let mut r = lcg(27);
            (0..SEQ).map(|_| (r().abs() * 4.0) as usize % 4).collect()
        };
        let mut out = vec![0f32; SEQ * D];
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("gather_rows d={D} rows={SEQ}"),
            REPEATS,
            || {
                out.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.gather_rows(&x, D, &rows_idx, &mut out);
                download(&g, &out)
            },
        );
    }
    {
        let src = vec_of(SEQ * 3 * D, 28);
        let mut out = vec![0f32; HEADS * SEQ * HD];
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("split_heads seq={SEQ} h={HEADS} hd={HD}"),
            REPEATS,
            || {
                out.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.split_heads(&src, 3 * D, 0, SEQ, HEADS, HD, &mut out);
                download(&g, &out)
            },
        );
    }
    {
        let src = vec_of(HEADS * SEQ * HD, 29);
        let mut out = vec![0f32; SEQ * D];
        repeat_arm(
            &mut rows,
            &mut failed,
            &format!("merge_heads seq={SEQ} h={HEADS} hd={HD}"),
            REPEATS,
            || {
                out.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.merge_heads(&src, SEQ, HEADS, HD, &mut out);
                download(&g, &out)
            },
        );
    }
    {
        let a = vec_of(SEQ * D, 30);
        let b = vec_of(D * I_SZ, 31);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; SEQ * I_SZ];
        repeat_arm(
            &mut rows,
            &mut failed,
            "matmul [317,1024]×[1024,2624] (plain)",
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul(&wa, 0, SEQ, D, &b, 0, I_SZ, &mut dst, 0);
                download(&g, &dst)
            },
        );
    }

    // ── the HEAD scorer chain at banking77's 77-way shape ────────────────
    // reads_of: gather → scorer LN → Linear[d,d] → GELU → Linear[d,1] →
    // softmax32(host). m = k_opts = 77 is UNIQUE to banking77 (the noul
    // pools and the other suites never present 77 options); n = 1 is the
    // narrow instance's degenerate column tile. Both arms mirror head.rs
    // exactly (the marker gather draws from a 16-row pool like the case
    // pack, the scorer LN runs rows=77 d=1024).
    {
        let x = vec_of(16 * D, 32);
        let rows_idx: Vec<usize> = {
            let mut r = lcg(33);
            (0..77).map(|_| (r().abs() * 16.0) as usize % 16).collect()
        };
        let mut out = vec![0f32; 77 * D];
        repeat_arm(
            &mut rows,
            &mut failed,
            "head gather_rows d=1024 rows=77",
            REPEATS,
            || {
                out.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.gather_rows(&x, D, &rows_idx, &mut out);
                download(&g, &out)
            },
        );
    }
    {
        let x = vec_of(77 * D, 34);
        let w: Vec<f32> = (0..D).map(|i| 0.5f32 + (i % 5) as f32 * 0.2).collect();
        let mut out = vec![0f32; 77 * D];
        let mut wx: Vec<f32> = Vec::new();
        let mut sq = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            "head scorer ln_rows rows=77 d=1024",
            REPEATS,
            || {
                wx = x.clone();
                out.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.layer_norm_nobias_into(&wx, &w, 1e-12, D, &mut sq, &mut out);
                download(&g, &out)
            },
        );
    }
    {
        let a = vec_of(77 * D, 35);
        let w = vec_of(D * D, 36);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; 77 * D];
        repeat_arm(
            &mut rows,
            &mut failed,
            "head s1 matmul_w [77,1024]×[1024,1024]",
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, 77, D, &w, D, &mut dst);
                download(&g, &dst)
            },
        );
    }
    {
        let a = vec_of(77 * D, 37);
        let w = vec_of(D, 38);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; 77];
        repeat_arm(
            &mut rows,
            &mut failed,
            "head s3 matmul_w [77,1024]×[1024,1] (n=1)",
            REPEATS,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, 77, D, &w, 1, &mut dst);
                download(&g, &dst)
            },
        );
    }
    {
        let x = vec_of(77 * D, 39);
        let mut wx: Vec<f32> = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            "head gelu len=77*1024",
            REPEATS,
            || {
                wx = x.clone();
                g.begin_pass();
                g.gelu_erf(&mut wx);
                download(&g, &wx)
            },
        );
    }

    {
        let x = vec_of(77 * D, 39);
        let mut wx: Vec<f32> = Vec::new();
        repeat_arm(
            &mut rows,
            &mut failed,
            "head gelu len=77*1024",
            REPEATS,
            || {
                wx = x.clone();
                g.begin_pass();
                g.gelu_erf(&mut wx);
                download(&g, &wx)
            },
        );
    }
    // ── the ACT head chain at its exact shapes (Issue 021's live fire) ──
    // act_of: matmul_w [1,1028]×a0w[256,1028] → bias → gelu → matmul_w
    // [1,256]×a2w[2,256] → bias → download[2]. The harness fire shows the
    // act output materially different between identical calls while the
    // scorer logits stay exact — these two GEMM shapes were the one
    // unprobed surface (m=1, k=1028 with a 4-past-BK tail, n=2).
    {
        let a = vec_of(D + 4, 40);
        let w = vec_of(256 * (D + 4), 41);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; 256];
        repeat_arm(
            &mut rows,
            &mut failed,
            "act a0 matmul_w [1,1028]×[256,1028]",
            2000,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, 1, D + 4, &w, 256, &mut dst);
                download(&g, &dst)
            },
        );
    }
    {
        let a = vec_of(256, 42);
        let w = vec_of(2 * 256, 43);
        let mut wa = Vec::new();
        let mut dst = vec![0f32; 2];
        repeat_arm(
            &mut rows,
            &mut failed,
            "act a2 matmul_w [1,256]×[2,256]",
            2000,
            || {
                wa = a.clone();
                dst.iter_mut().for_each(|v| *v = 0.0);
                g.begin_pass();
                g.matmul_w(&wa, 1, 256, &w, 2, &mut dst);
                download(&g, &dst)
            },
        );
    }

    for row in &rows {
        println!("{row}");
    }
    assert!(!failed, "cuda repeat probe: at least one arm DIVERGED");
}
