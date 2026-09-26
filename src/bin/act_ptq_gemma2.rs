//! act_ptq_gemma2 — riir-infer Issue 014 T2(b): the dense-parent PTQ lane on
//! the house fixture (gemma-2-2b-it f16).
//!
//! This is the regime where katgpt-rs Bench 896's synthetic gains are
//! predicted to transfer, if they transfer anywhere: real f32-effective
//! parents (the f16 checkpoint), real activations (the diagonal + held-out
//! vectors collected by `forward_gemma2_f16_act_tapped`), the REAL quantizer
//! (`TernaryGroupWeights` at g128 — the same payload the Bonsai lane ships).
//! The metric is Bench 896's own G1 — held-out output reconstruction error
//! `E‖(W−Ŵ)x‖² / E‖Wx‖²` — at layer level. A full-model ternary gemma
//! forward does not exist; the layer-level read is the honest scope of this
//! lane, and it is stated as such in the record.
//!
//! Arms (mirroring the Bonsai walk minus the born-ternary-only control):
//! | arm | rule | diagonal |
//! |---|---|---|
//! | `mean_abs` | `quantize_from_f32` (activation-blind baseline) | — |
//! | `wma_ex2` | `WeightedMeanAbs` | per-tap `E[x²]` |
//! | `ws_ex2` | `WeightedSearch` | per-tap `E[x²]` |
//! | `ws_uniform` | `WeightedSearch` | uniform (blind-search control) |
//! | `zeroqat` | mean-abs codes + multiplier GD (default knobs) | per-tap `E[x²]` |
//!
//! The reference is the f16 parent itself. On DENSE parents (unlike the
//! born-ternary lane) the layer-local surrogate is non-degenerate: the
//! requant carries real quantization error the fits can redistribute.
//!
//! MEASUREMENT-ONLY (Issue 014 P0 law): nothing is written back to the
//! checkpoint; the requant payloads live and die inside the evaluation.
//!
//! Usage:
//! ```text
//! act_ptq_gemma2 <model.gguf> <corpus-dir-or-txt>
//!     [--max-tokens N] [--seq-len N] [--eval-vecs N] [--stride N]
//!     [--min-pos N] [--report PATH] [--artifact PATH]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use half::f16;
use katgpt_core::TernaryGroupWeights;
use katgpt_core::act_channel_moments::{ActChannelDiagonal, ActChannelMoments};
use katgpt_types::ternary_group_act_aware::ActAwareScaleFit;
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::ForwardContext;
use riir_infer_core::transformer::gemma2_act_tap::{
    GemmaActCapture, forward_gemma2_f16_act_tapped,
};
use riir_infer_core::types::kv_dim;

/// `TernaryGroupWeights::GROUP_SIZE` (the dequant path's own convention).
const GROUP: usize = 128;

/// riir-train `ZeroQatConfig::default()` — the comparator's knobs, verbatim.
const ZQ_STEPS: usize = 100;
const ZQ_LR: f32 = 0.01;
const ZQ_EPS: f32 = 0.01;
const ZQ_MIN_M: f32 = 1e-6;
const ZQ_MAX_M: f32 = 1e6;

const ARMS: [&str; 5] = ["mean_abs", "wma_ex2", "ws_ex2", "ws_uniform", "zeroqat"];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: act_ptq_gemma2 <model.gguf> <corpus-dir-or-txt> [--max-tokens N] \
             [--seq-len N] [--eval-vecs N] [--stride N] [--min-pos N] \
             [--report PATH] [--artifact PATH]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let mut max_tokens: usize = 8_192;
    let mut seq_len: usize = 512;
    let mut eval_vecs: usize = 48;
    let mut stride: usize = 16;
    let mut min_pos: usize = 64;
    let mut report_path: Option<PathBuf> = None;
    let mut artifact_path: Option<PathBuf> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().context("--max-tokens N")?;
                i += 2;
            }
            "--seq-len" => {
                seq_len = args[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--eval-vecs" => {
                eval_vecs = args[i + 1].parse().context("--eval-vecs N")?;
                i += 2;
            }
            "--stride" => {
                stride = args[i + 1].parse().context("--stride N")?;
                i += 2;
            }
            "--min-pos" => {
                min_pos = args[i + 1].parse().context("--min-pos N")?;
                i += 2;
            }
            "--report" => {
                report_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--artifact" => {
                artifact_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if seq_len > 4096 {
        bail!("--seq-len must stay <= 4096 (gemma-2 sliding window; see the module doc)");
    }
    if stride == 0 {
        bail!("--stride must be >= 1");
    }

    // ── Load model + tokenizer from ONE open GGUF (the vk shape) ──
    let t0 = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    println!(
        "# act_ptq_gemma2: gemma-2 f16 | layers={} n_embd={} mlp_hidden={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.mlp_hidden,
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    drop(gguf);

    // ── Corpus → tokens ──
    let t1 = Instant::now();
    let text = load_corpus_text(&corpus_path)?;
    let all_tokens = tok.encode(&text);
    println!(
        "# corpus: {} chars → {} tokens ({:.1}s)",
        text.len(),
        all_tokens.len(),
        t1.elapsed().as_secs_f32()
    );
    let take = all_tokens.len().min(max_tokens);
    let tokens: Vec<usize> = all_tokens[..take].to_vec();

    // ── The tap widths (4 distinct inputs per layer) ──
    let n = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let widths: Vec<usize> = (0..config.n_layer)
        .flat_map(|_| [n, q_dim, n, config.mlp_hidden])
        .collect();
    let mut moments = ActChannelMoments::new(&widths);
    let mut capture = GemmaActCapture::new(config.n_layer, eval_vecs, stride, min_pos);

    // ── The tap pass (chunked causal forward) ──
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let t2 = Instant::now();
    for chunk in tokens.chunks(seq_len) {
        cache.reset();
        for (pos, &token) in chunk.iter().enumerate() {
            forward_gemma2_f16_act_tapped(
                &mut ctx,
                &weights,
                &mut cache,
                &mut moments,
                &mut capture,
                token,
                pos,
                &config,
            );
        }
    }
    let pass_s = t2.elapsed().as_secs_f32();
    println!(
        "# tap pass: {} tokens in {:.0}s ({:.0} tok/s) | captured {} pools",
        tokens.len(),
        pass_s,
        tokens.len() as f32 / pass_s.max(1e-6),
        capture.pools.iter().filter(|p| !p.is_empty()).count(),
    );

    // ── Freeze the diagonal (+ optional artifact) ──
    let diagonal = moments.freeze();
    assert!(diagonal.verify());
    if let Some(p) = artifact_path {
        std::fs::write(&p, diagonal.to_bytes())
            .with_context(|| format!("write artifact {}", p.display()))?;
        println!(
            "# diagonal artifact: {} (commitment {})",
            p.display(),
            hex(&diagonal.commitment())
        );
    }

    // The held-out pools must be FULL everywhere (deterministic capture, the
    // corpus is long enough at defaults) — a short pool means the eval
    // support shrank; report it, never fail silently.
    let min_pool = capture.pools.iter().map(|p| p.len()).min().unwrap_or(0);
    println!(
        "# eval pools: min {min_pool} / target {eval_vecs} vectors per tap",
    );
    if min_pool == 0 {
        bail!("capture produced empty pools — corpus too short for --min-pos/--stride");
    }

    // ── PTQ + per-layer evaluation ──
    // acc[arm]: (Σ num, Σ den) over all layers/tensors/eval vectors.
    let mut acc: Vec<(f64, f64)> = vec![(0.0, 0.0); ARMS.len()];
    // per family: attn = wq+wk+wv+wo, mlp = gate+up+down
    let mut acc_attn: Vec<(f64, f64)> = vec![(0.0, 0.0); ARMS.len()];
    let mut acc_mlp: Vec<(f64, f64)> = vec![(0.0, 0.0); ARMS.len()];
    let mut worst: Vec<(f64, String)> = vec![(0.0, String::new()); ARMS.len()];

    // The uniform diagonal per width (built lazily).
    let mut uniform: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();

    let t3 = Instant::now();
    for (l, layer) in weights.layers.iter().enumerate() {
        let base = l * 4;
        let tensors: [(&str, &[f16], usize, usize, usize); 7] = [
            ("attn_wq", &layer.attn_wq, q_dim, n, base),
            (
                "attn_wk",
                &layer.attn_wk,
                kv_dim(&config),
                n,
                base,
            ),
            (
                "attn_wv",
                &layer.attn_wv,
                kv_dim(&config),
                n,
                base,
            ),
            ("attn_wo", &layer.attn_wo, n, q_dim, base + 1),
            ("gate_proj", &layer.gate_proj, config.mlp_hidden, n, base + 2),
            ("up_proj", &layer.up_proj, config.mlp_hidden, n, base + 2),
            (
                "down_proj",
                &layer.down_proj,
                n,
                config.mlp_hidden,
                base + 3,
            ),
        ];
        for (name, wf16, rows, cols, tap) in tensors {
            let pool = &capture.pools[tap];
            // The f32 parent (row-major [rows × cols], the matmul layout).
            let parent: Vec<f32> = wf16.iter().map(|&v| v.to_f32()).collect();
            // The reference outputs (one per eval vector).
            let refs: Vec<Vec<f32>> = pool
                .iter()
                .map(|x| f32_matvec(&parent, rows, cols, x))
                .collect();
            // Per-arm payloads + errors.
            for (ai, arm) in ARMS.iter().enumerate() {
                let t_arm = Instant::now();
                let payload = requantize(arm, &parent, rows, cols, &diagonal, tap, &mut uniform)?;
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for (x, y_ref) in pool.iter().zip(&refs) {
                    let mut y = vec![0.0f32; rows];
                    katgpt_core::simd_ternary_group_matvec_parallel(
                        &payload,
                        x,
                        &mut y,
                    );
                    for (ye, yr) in y.iter().zip(y_ref) {
                        let d = f64::from(ye - yr);
                        num += d * d;
                        den += f64::from(*yr) * f64::from(*yr);
                    }
                }
                acc[ai].0 += num;
                acc[ai].1 += den;
                if name != "down_proj" && name != "gate_proj" && name != "up_proj" {
                    acc_attn[ai].0 += num;
                    acc_attn[ai].1 += den;
                } else {
                    acc_mlp[ai].0 += num;
                    acc_mlp[ai].1 += den;
                }
                let rel = if den > 0.0 { (num / den).sqrt() } else { f64::NAN };
                if rel > worst[ai].0 {
                    worst[ai] = (rel, format!("l{l:02}.{name}"));
                }
                let _ = t_arm.elapsed();
            }
        }
        println!("# layer {l} done ({:.0}s cumulative)", t3.elapsed().as_secs_f32());
    }

    // ── Report ──
    let mut out = String::new();
    out.push_str("# Issue 014 T2(b) — dense-parent PTQ lane: gemma-2 f16 (the house fixture)\n\n");
    out.push_str(&format!(
        "slice: {} tokens (seq_len {seq_len}) | eval: {} vectors/tap (stride {stride}, min_pos {min_pos}) | metric: held-out E‖(W−Ŵ)x‖²/E‖Wx‖² (sqrt)\n\n",
        tokens.len(),
        min_pool,
    ));
    out.push_str("| arm | overall | attn family | mlp family | worst tensor |\n|---|---|---|---|---|\n");
    for (ai, arm) in ARMS.iter().enumerate() {
        let (num, den) = acc[ai];
        let (an, ad) = acc_attn[ai];
        let (mn, md) = acc_mlp[ai];
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {} ({:.4}) |\n",
            arm,
            (num / den).sqrt(),
            (an / ad).sqrt(),
            (mn / md).sqrt(),
            worst[ai].1,
            worst[ai].0,
        ));
    }
    out.push_str(
        "\nMEASUREMENT-ONLY (Issue 014 P0 law): layer-level reconstruction metric on \
         real parents + real activations (Bench 896's G1, model-bound). No full-model \
         ternary gemma forward exists; the retention walk is the Bonsai lane's gate.\n",
    );
    print!("{out}");
    if let Some(p) = report_path {
        std::fs::write(&p, &out).with_context(|| format!("write report {}", p.display()))?;
        println!("# report written: {}", p.display());
    }
    Ok(())
}

/// `y[r] = Σ_c W[r][c] · x[c]` on the f32 parent (plain scalar; the eval
/// runs a few dozen vectors, not a serving path).
fn f32_matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for (&wv, &xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        y[r] = acc;
    }
    y
}

/// Build one arm's ternary payload from the f32 parent.
fn requantize(
    arm: &str,
    parent: &[f32],
    rows: usize,
    cols: usize,
    diagonal: &ActChannelDiagonal,
    tap: usize,
    uniform: &mut std::collections::HashMap<usize, Vec<f32>>,
) -> Result<TernaryGroupWeights> {
    match arm {
        "mean_abs" => Ok(TernaryGroupWeights::quantize_from_f32(parent, rows, cols)),
        "wma_ex2" | "ws_ex2" => {
            let fit = if arm == "wma_ex2" {
                ActAwareScaleFit::WeightedMeanAbs
            } else {
                ActAwareScaleFit::WeightedSearch
            };
            Ok(TernaryGroupWeights::quantize_from_f32_act_aware(
                parent,
                rows,
                cols,
                diagonal.mean_sq(tap),
                fit,
            ))
        }
        "ws_uniform" => {
            let d = uniform.entry(cols).or_insert_with(|| vec![1.0f32; cols]);
            Ok(TernaryGroupWeights::quantize_from_f32_act_aware(
                parent,
                rows,
                cols,
                d,
                ActAwareScaleFit::WeightedSearch,
            ))
        }
        "zeroqat" => Ok(zeroqat_refit(parent, rows, cols, diagonal.mean_sq(tap))),
        other => bail!("unknown arm {other}"),
    }
}

/// The ZeroQAT-class arm at riir-train default knobs: codes fixed at the
/// mean-abs requant's, scales GD-refined (multiplier parameterization) on
/// the diagonal-weighted reconstruction loss — an exact parabola per group.
fn zeroqat_refit(
    parent: &[f32],
    rows: usize,
    cols: usize,
    diag_slice: &[f32],
) -> TernaryGroupWeights {
    let mut rq = TernaryGroupWeights::quantize_from_f32(parent, rows, cols);
    for r in 0..rows {
        for g in 0..rq.groups_per_row {
            let g_start = g * GROUP;
            let g_end = (g_start + GROUP).min(cols);
            let h = &diag_slice[g_start..g_end];
            let hmax = h.iter().copied().fold(0.0f32, f32::max);
            let s_rq = f32::from(rq.group_scale[r * rq.groups_per_row + g]);
            let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
            for (j, &wij) in parent[r * cols + g_start..r * cols + g_end].iter().enumerate() {
                let u = if hmax > 0.0 { h[j] / hmax } else { 1.0 };
                let q = ternary_sign(&rq, r, g_start + j) as f32;
                a += u * wij * wij;
                b += u * wij * q;
                c += u * q * q;
            }
            let l = |m: f32| {
                let sm = s_rq * m;
                a - 2.0 * b * sm + c * sm * sm
            };
            let mut m = 1.0f32;
            for _ in 0..ZQ_STEPS {
                let grad = (l(m + ZQ_EPS) - l(m - ZQ_EPS)) / (2.0 * ZQ_EPS);
                m = (m - ZQ_LR * grad).clamp(ZQ_MIN_M, ZQ_MAX_M);
            }
            rq.group_scale[r * rq.groups_per_row + g] = f16::from_f32(m * s_rq);
        }
    }
    rq
}

#[inline]
fn ternary_sign(w: &TernaryGroupWeights, row: usize, col: usize) -> i8 {
    let idx = row * w.blocks64 + (col >> 6);
    let mask = 1u64 << (col & 63);
    if w.pos_bits[idx] & mask != 0 {
        1
    } else if w.neg_bits[idx] & mask != 0 {
        -1
    } else {
        0
    }
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}
