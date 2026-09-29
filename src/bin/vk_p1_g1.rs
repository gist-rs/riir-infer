//! vk_p1_g1 — riir-infer Issue 013 T1's model-bound G1 harness (katgpt-rs
//! Issue 883 P1, Research 587): the fitted-token-value table on
//! gemma-2-2b — **PPL at matched bits with and without mean removal** —
//! plus the cache-level quant-MSE replay that validates Gate 1's
//! prediction without cross-arm trajectory divergence.
//!
//! Protocol + gates are PRE-REGISTERED in `.issues/013_*.md` (added
//! 2026-09-28, before this bin's first run; the tolerances do not move
//! after numbers exist):
//!
//! - **Gate 1** (prediction): the measured per-layer V quant-MSE drop
//!   `1 − MSE_mean/MSE_plain` — both quantizations replayed from the SAME
//!   f16-trajectory K/V captures — vs the prediction `1 − m·ρ_l(V)`; mean
//!   absolute error ≤ 0.10 over layers. ρ_l(V) from THIS run's calibration
//!   slice (a held-out prediction, not the Bench 004 numbers).
//! - **Gate 2** (quality, hard): PPL(mean-removed) ≤ PPL(plain) per bits.
//! - **Gate 3** (pilot slice): NLL share by calibration-frequency band
//!   (0–1k / 1k–8k / tail); the full per-family walk stays deferred.
//! - **Gate 4** (pilot slice): mean NLL over chunk positions 0–7 (the
//!   sink proxy); the kv_sink_window check stays deferred.
//! - Wiring check **G0**: the mean arm's K quant-MSE must equal the plain
//!   arm's bitwise (the decorator passes keys through).
//!
//! MEASUREMENT LANE (the P0 law carries): the bin makes no serving claim —
//! Gate 2 decides quality. A measured null (the table refunds nothing on
//! gemma-2) is a legitimate recorded negative.
//!
//! Usage:
//! ```text
//! vk_p1_g1 <gguf> <corpus-dir-or-txt> [--cal-tokens N] [--eval-tokens N]
//!          [--seq-len N] [--bits 2,4] [--top-k N] [--lambda-js F]
//!          [--tile-size N] [--report PATH]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use katgpt_core::fitted_value_table::{FittedTokenTable, MeanRemovedValueCache, VkSignal};
use katgpt_kv::kvarn::kv_cache::{KVarNConfig, KVarNKVCache};
use katgpt_transformer::MultiLayerKVCache;
use katgpt_types::QuantizedKVCache;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::ForwardContext;
use riir_infer_core::transformer::gemma2_calibration::{
    CalibrationTables, forward_gemma2_f16_tapped, load_gemma2_f16_direct,
};
use riir_infer_core::transformer::gemma2_quantized::{
    MirrorRefresh, QuantizedKvMirror, forward_gemma2_f16_qkv,
};
use riir_infer_core::transformer::{forward_gemma2_f16, NoHook};
use riir_infer_core::types::kv_dim;

/// Chunk positions ≤ this count as "early" (the Gate-4 sink proxy).
const EARLY_POS: usize = 8;

/// Per-arm PPL accounting: total NLL, calibration-frequency-band NLL
/// (Gate 3), and early-position NLL (Gate 4).
#[derive(Default)]
struct PplAcc {
    nll: f64,
    n: usize,
    band_nll: [f64; 3],
    band_n: [usize; 3],
    early_nll: f64,
    early_n: usize,
}

impl PplAcc {
    fn score(&mut self, logits: &[f32], target: usize, rank: &[u32], pos: usize) {
        let l = nll(logits, target);
        self.nll += l;
        self.n += 1;
        let band = match rank.get(target).copied().unwrap_or(u32::MAX) {
            r if r < 1_000 => 0,
            r if r < 8_000 => 1,
            _ => 2,
        };
        self.band_nll[band] += l;
        self.band_n[band] += 1;
        if pos < EARLY_POS {
            self.early_nll += l;
            self.early_n += 1;
        }
    }

    fn ppl(&self) -> f64 {
        (self.nll / self.n.max(1) as f64).exp()
    }
}

/// Per-bits MSE-replay accumulators (the Gate-1 instruments).
struct MseAcc {
    /// K quant SSE per layer, plain arm (aggregate over kv_dim).
    sse_k_plain: Vec<f64>,
    /// K quant SSE per layer, mean arm — must equal `sse_k_plain` (G0).
    sse_k_mean: Vec<f64>,
    /// V quant SSE per layer per kv-head, plain arm.
    sse_v_plain: Vec<Vec<f64>>,
    /// V quant SSE per layer per kv-head, mean arm.
    sse_v_mean: Vec<Vec<f64>>,
}

fn nll(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&l| (l as f64 - m).exp()).sum();
    m + z.ln() - logits[target] as f64
}

fn sq_diff_sum(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum()
}

fn kvarn_config(
    n_layer: usize,
    kvd: usize,
    max_seq_len: usize,
    bits: u8,
    tile_size: usize,
) -> KVarNConfig {
    KVarNConfig {
        n_layers: n_layer,
        kv_dim: kvd,
        max_seq_len,
        bits,
        tile_size,
        ..KVarNConfig::default()
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: vk_p1_g1 <model.gguf> <corpus-dir-or-txt> [--cal-tokens N] \
             [--eval-tokens N] [--seq-len N] [--bits 2,4] [--top-k N] \
             [--lambda-js F] [--tile-size N] [--report PATH]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let (mut cal_tokens, mut eval_tokens, mut seq_len) = (60_000usize, 12_288usize, 1024usize);
    let mut bits_list: Vec<u8> = vec![2, 4];
    let (mut top_k, mut lambda_js, mut tile_size) = (8192usize, 0.0f32, 128usize);
    let mut report_path: Option<PathBuf> = None;
    let mut i = 3;
    while i < args.len() {
        let v = args.get(i + 1).context("flag needs a value")?;
        match args[i].as_str() {
            "--cal-tokens" => cal_tokens = v.parse()?,
            "--eval-tokens" => eval_tokens = v.parse()?,
            "--seq-len" => seq_len = v.parse()?,
            "--bits" => {
                bits_list = v.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?
            }
            "--top-k" => top_k = v.parse()?,
            "--lambda-js" => lambda_js = v.parse()?,
            "--tile-size" => tile_size = v.parse()?,
            "--report" => report_path = Some(PathBuf::from(v)),
            other => bail!("unknown arg {other}"),
        }
        i += 2;
    }
    if seq_len > 4096 {
        bail!("--seq-len must stay <= 4096 (gemma-2 sliding window; see vk_calibration)");
    }

    // ── Load model + tokenizer from ONE open GGUF ──
    let t0 = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let mut config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    let bos = tok.bos_id();
    println!(
        "# vk_p1_g1: gemma-2 f16 | layers={} n_embd={} n_head={} n_kv_head={} head_dim={} kv_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_kv_head,
        config.head_dim,
        kv_dim(&config),
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    drop(gguf);

    // ── Corpus → held-out split ──
    let text = load_corpus_text(&corpus_path)?;
    let all = tok.encode(&text);
    let cal_end = cal_tokens.min(all.len());
    let eval_end = (cal_tokens.saturating_add(eval_tokens)).min(all.len());
    if eval_end - cal_end < seq_len {
        bail!(
            "eval slice ({} tokens) shorter than one chunk ({seq_len}) — fetch more corpus",
            eval_end - cal_end
        );
    }
    let cal: Vec<usize> = all[..cal_end].to_vec();
    let eval: Vec<usize> = all[cal_end..eval_end].to_vec();
    println!(
        "# corpus: {} chars → {} tokens | cal slice {} | eval slice {} (held-out)",
        text.len(),
        all.len(),
        cal.len(),
        eval.len()
    );

    // Eval chunks: [BOS] + seq_len corpus tokens (the row_logit_floor T2
    // convention), cache+mirror reset per chunk.
    let chunks: Vec<Vec<usize>> = eval
        .chunks(seq_len)
        .map(|c| {
            let mut s = Vec::with_capacity(c.len() + 1);
            s.push(bos);
            s.extend_from_slice(c);
            s
        })
        .collect();
    let longest = chunks.iter().map(Vec::len).max().unwrap_or(0);
    config.block_size = longest;
    let scored_total: usize = chunks.iter().map(|c| c.len() - 1).sum();
    println!(
        "# eval: {} chunks × ≤{} tokens ([BOS]+chunk) | {} scored | pre-registered gates: G1 MAE≤0.10, G2 ppl(mean)≤ppl(plain)",
        chunks.len(),
        longest,
        scored_total
    );

    let kvd = kv_dim(&config);
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;

    // Frequency rank map (the Gate-3 band keying) + tracked mass (the
    // Gate-1 prediction's m), both from the CAL slice only.
    let mut counts = vec![0u64; config.vocab_size];
    for &t in &cal {
        counts[t] += 1;
    }
    let mut order: Vec<u32> = (0..config.vocab_size as u32).collect();
    order.sort_unstable_by_key(|&t| std::cmp::Reverse(counts[t as usize]));
    let mut rank = vec![u32::MAX; config.vocab_size];
    for (r, &t) in order.iter().enumerate() {
        if counts[t as usize] > 0 {
            rank[t as usize] = r as u32;
        }
    }
    let total_cal: u64 = counts.iter().sum();
    let tracked: u64 = {
        let mut c = counts.clone();
        c.sort_unstable_by(|a, b| b.cmp(a));
        c.iter().take(top_k).sum()
    };
    let m_cov = tracked as f64 / total_cal as f64;
    println!(
        "# prediction: top_k={} tracks {m_cov:.4} of cal mass → predicted per-layer drop = 1 − m·ρ_l(V)",
        cal_tables_top_k(top_k)
    );

    // ── Phase A — calibration pass (the tapped forward, no lm_head) ──
    let mut cal_tables = CalibrationTables::from_counts(n_layer, kvd, counts, top_k);
    let mut ctx = ForwardContext::new(&config);
    let mut cal_cache = MultiLayerKVCache::new(&config);
    let t_cal = Instant::now();
    for chunk in cal.chunks(seq_len) {
        cal_cache.reset();
        for (pos, &token) in chunk.iter().enumerate() {
            forward_gemma2_f16_tapped(
                &mut ctx,
                &weights,
                &mut cal_cache,
                &mut cal_tables,
                token,
                pos,
                &config,
                &mut NoHook,
            );
        }
    }
    println!(
        "# phase A (calibration): {} tokens in {:.0}s ({:.0} tok/s)",
        cal.len(),
        t_cal.elapsed().as_secs_f32(),
        cal.len() as f32 / t_cal.elapsed().as_secs_f32().max(1e-6)
    );
    let rho_v: Vec<f64> = cal_tables
        .layers
        .iter()
        .map(|lt| f64::from(lt.v.r_squared().aggregate))
        .collect();
    let rho_head: Vec<Vec<f64>> = cal_tables
        .layers
        .iter()
        .map(|lt| {
            (0..n_kv)
                .map(|h| f64::from(lt.v.r_squared().aggregate_over(h * hd, (h + 1) * hd)))
                .collect()
        })
        .collect();
    let table = FittedTokenTable::from_calibration(&cal_tables, VkSignal::ValueMean, lambda_js);
    println!(
        "# frozen table: {} rows × width {} ({:.2} GiB) | λ_js={lambda_js}",
        table.rows(),
        table.width(),
        table.bytes() as f64 / (1 << 30) as f64
    );

    // ── Phase B/C — the f16 control arm (PPL + the K/V captures) ──
    let mut cap_k = vec![0.0f32; n_layer * longest * kvd];
    let mut cap_v = vec![0.0f32; n_layer * longest * kvd];
    let mut mse: Vec<MseAcc> = (0..bits_list.len())
        .map(|_| MseAcc {
            sse_k_plain: vec![0.0; n_layer],
            sse_k_mean: vec![0.0; n_layer],
            sse_v_plain: vec![vec![0.0; n_kv]; n_layer],
            sse_v_mean: vec![vec![0.0; n_kv]; n_layer],
        })
        .collect();
    // Replay caches: one plain + one decorated per bits, reused across
    // chunks via reset() (post-896 reset re-arms the quantized flags).
    let mut replay_plain: Vec<KVarNKVCache> = bits_list
        .iter()
        .map(|&b| KVarNKVCache::with_config(&kvarn_config(n_layer, kvd, longest, b, tile_size)))
        .collect();
    let mut replay_mean: Vec<MeanRemovedValueCache<KVarNKVCache>> = bits_list
        .iter()
        .map(|&b| {
            let inner = KVarNKVCache::with_config(&kvarn_config(n_layer, kvd, longest, b, tile_size));
            MeanRemovedValueCache::new(inner, &table, longest)
        })
        .collect();
    let mut row = vec![0.0f32; kvd];

    let mut f16_acc = PplAcc::default();
    let mut f16_cache = MultiLayerKVCache::new(&config);
    let t_f16 = Instant::now();
    for chunk in &chunks {
        f16_cache.reset();
        let t_n = chunk.len();
        for (pos, &token) in chunk.iter().enumerate() {
            let logits = forward_gemma2_f16(&mut ctx, &weights, &mut f16_cache, token, pos, &config);
            if pos + 1 < t_n {
                f16_acc.score(logits, chunk[pos + 1], &rank, pos);
            }
        }
        // Capture the f16 trajectory's K/V (exact, the replay's source).
        for l in 0..n_layer {
            let base = l * longest * kvd;
            cap_k[base..base + t_n * kvd].copy_from_slice(&f16_cache.layers[l].key[..t_n * kvd]);
            cap_v[base..base + t_n * kvd].copy_from_slice(&f16_cache.layers[l].value[..t_n * kvd]);
        }
        // Replay per bits: SAME V values through both quantizations.
        for (bi, _bits) in bits_list.iter().enumerate() {
            replay_plain[bi].reset();
            replay_mean[bi].reset();
            for (pos, &tok_id) in chunk.iter().enumerate() {
                replay_mean[bi].set_token(pos, tok_id as u32);
                for l in 0..n_layer {
                    let off = (l * longest + pos) * kvd;
                    let (k, v) = (&cap_k[off..off + kvd], &cap_v[off..off + kvd]);
                    replay_plain[bi].store_key(l, pos, k);
                    replay_plain[bi].store_value(l, pos, v);
                    replay_mean[bi].store_key(l, pos, k);
                    replay_mean[bi].store_value(l, pos, v);
                }
            }
            let acc = &mut mse[bi];
            for l in 0..n_layer {
                for pos in 0..t_n {
                    let off = (l * longest + pos) * kvd;
                    replay_plain[bi].dequantize_value_into(l, pos, &mut row);
                    for (h, s) in acc.sse_v_plain[l].iter_mut().enumerate() {
                        *s += sq_diff_sum(&cap_v[off + h * hd..off + (h + 1) * hd], &row[h * hd..(h + 1) * hd]);
                    }
                    replay_mean[bi].dequantize_value_into(l, pos, &mut row);
                    for (h, s) in acc.sse_v_mean[l].iter_mut().enumerate() {
                        *s += sq_diff_sum(&cap_v[off + h * hd..off + (h + 1) * hd], &row[h * hd..(h + 1) * hd]);
                    }
                    replay_plain[bi].dequantize_key_into(l, pos, &mut row);
                    acc.sse_k_plain[l] += sq_diff_sum(&cap_k[off..off + kvd], &row);
                    replay_mean[bi].dequantize_key_into(l, pos, &mut row);
                    acc.sse_k_mean[l] += sq_diff_sum(&cap_k[off..off + kvd], &row);
                }
            }
        }
    }
    let f16_s = t_f16.elapsed().as_secs_f32();
    println!(
        "# phase B f16 control: {} forwards in {:.0}s ({:.0} tok/s) + replays",
        f16_acc.n + chunks.len(),
        f16_s,
        (f16_acc.n + chunks.len()) as f32 / f16_s.max(1e-6)
    );

    // ── Phase B — the quant arms ──
    let refresh = MirrorRefresh {
        tile_size,
        cache_max_seq: longest,
    };
    struct QuantResult {
        bits: u8,
        mean: bool,
        acc: PplAcc,
    }
    let mut quant_results: Vec<QuantResult> = Vec::new();
    for &bits in &bits_list {
        for mean in [false, true] {
            let cfg = kvarn_config(n_layer, kvd, longest, bits, tile_size);
            let mut mirror = QuantizedKvMirror::new(&config, longest);
            let mut acc = PplAcc::default();
            let t = Instant::now();
            if mean {
                let inner = KVarNKVCache::with_config(&cfg);
                let mut cache = MeanRemovedValueCache::new(inner, &table, longest);
                for chunk in &chunks {
                    cache.reset();
                    mirror.reset();
                    for (pos, &token) in chunk.iter().enumerate() {
                        cache.set_token(pos, token as u32);
                        let logits = forward_gemma2_f16_qkv(
                            &mut ctx, &weights, &mut cache, &mut mirror, token, pos, &config,
                            Some(&refresh),
                        );
                        if pos + 1 < chunk.len() {
                            acc.score(logits, chunk[pos + 1], &rank, pos);
                        }
                    }
                }
            } else {
                let mut cache = KVarNKVCache::with_config(&cfg);
                for chunk in &chunks {
                    cache.reset();
                    mirror.reset();
                    for (pos, &token) in chunk.iter().enumerate() {
                        let logits = forward_gemma2_f16_qkv(
                            &mut ctx, &weights, &mut cache, &mut mirror, token, pos, &config,
                            Some(&refresh),
                        );
                        if pos + 1 < chunk.len() {
                            acc.score(logits, chunk[pos + 1], &rank, pos);
                        }
                    }
                }
            }
            let secs = t.elapsed().as_secs_f32();
            println!(
                "# phase B {}{bits}: ppl runs in {:.0}s ({:.0} tok/s) | ppl {:.4}",
                if mean { "mean" } else { "plain" },
                secs,
                (acc.n + chunks.len()) as f32 / secs.max(1e-6),
                acc.ppl()
            );
            quant_results.push(QuantResult { bits, mean, acc });
        }
    }

    // ── Gates + report ──
    let mut out = String::new();
    out.push_str("# Issue 013 T1 pilot — fitted-token-value P1 on gemma-2-2b (pre-registered gates)\n\n");
    out.push_str(&format!(
        "fixture: {} | corpus: {} | cal {} tok / eval {} tok (held-out) | seq [BOS]+{} | λ_js={lambda_js} | top_k={top_k} (m={m_cov:.4}) | bits {bits_list:?} | tile {tile_size}\n\n",
        gguf_path.display(),
        corpus_path.display(),
        cal.len(),
        eval.len(),
        seq_len,
    ));
    out.push_str("## Phase A — calibration slice (the Gate-1 predictor)\n\n| layer | ρ_l(V) | predicted drop 1−m·ρ |\n|---|---|---|\n");
    for (l, r) in rho_v.iter().enumerate() {
        out.push_str(&format!("| {l} | {r:.4} | {:.4} |\n", 1.0 - m_cov * r));
    }
    out.push_str(&format!(
        "\nper-head ρ_l(V) (layer 0): {} | (layer {}): {}\n",
        rho_head[0]
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
            .join(" "),
        n_layer / 2,
        rho_head[n_layer / 2]
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    ));

    // Gate 1 + G0.
    out.push_str("\n## Gate 1 — measured V quant-MSE drop vs prediction (same-trajectory replay)\n\n| bits | layer | MSE plain | MSE mean | measured drop | predicted | miss |\n|---|---|---|---|---|---|---|\n");
    for (bi, &bits) in bits_list.iter().enumerate() {
        let acc = &mse[bi];
        let mut mae = 0.0f64;
        let mut rows = String::new();
        for (l, (sp, sm)) in acc.sse_v_plain.iter().zip(acc.sse_v_mean.iter()).enumerate() {
            let mp = sp.iter().sum::<f64>();
            let mm = sm.iter().sum::<f64>();
            let drop = if mp > 1e-12 { 1.0 - mm / mp } else { 0.0 };
            let pred = 1.0 - m_cov * rho_v[l];
            let miss = drop - pred;
            mae += miss.abs();
            rows.push_str(&format!(
                "| {bits} | {l} | {mp:.4e} | {mm:.4e} | {drop:.4} | {pred:.4} | {miss:+.4} |\n"
            ));
        }
        mae /= n_layer as f64;
        out.push_str(&rows);
        out.push_str(&format!(
            "\n**G1 bits={bits}: MAE = {mae:.4} — {}** (pre-registered bar ≤ 0.10)\n",
            if mae <= 0.10 { "PASS" } else { "FAIL" }
        ));
        // G0: K identical between arms.
        let g0_max: f64 = acc
            .sse_k_plain
            .iter()
            .zip(acc.sse_k_mean.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        out.push_str(&format!(
            "\nG0 wiring check bits={bits}: Σ|K SSE plain − mean| = {g0_max:e} — {}\n",
            if g0_max == 0.0 { "bitwise OK" } else { "MISMATCH (wiring bug)" }
        ));
    }

    // Gate 2 + 3 + 4 table.
    out.push_str("\n## Gate 2/3/4 — PPL + band shares + early-position NLL\n\n| arm | ppl | Δ vs plain | band 0–1k share | band 1k–8k share | band tail share | early(0..7) mean NLL |\n|---|---|---|---|---|---|---|\n");
    let mut arm_rows: Vec<(String, &PplAcc)> = vec![("f16".to_string(), &f16_acc)];
    for q in &quant_results {
        let name = format!(
            "{}{}",
            if q.mean { "mean" } else { "plain" },
            q.bits
        );
        arm_rows.push((name, &q.acc));
    }
    let ppl_of = |name: &str| -> f64 {
        arm_rows
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(f64::NAN, |(_, a)| a.ppl())
    };
    for (name, acc) in &arm_rows {
        let tot = acc.nll.max(1e-12);
        let delta = if let Some(bits) = name
            .strip_prefix("mean")
            .and_then(|s| s.parse::<u8>().ok())
        {
            let plain = ppl_of(&format!("plain{bits}"));
            acc.ppl() - plain
        } else {
            f64::NAN
        };
        let verdict2 = match name.strip_prefix("mean") {
            Some(bits_str) => {
                let bits: u8 = bits_str.parse().unwrap_or(0);
                let plain = ppl_of(&format!("plain{bits}"));
                if acc.ppl() <= plain { "G2 PASS" } else { "G2 FAIL" }
            }
            None => "control",
        };
        out.push_str(&format!(
            "| {name} ({verdict2}) | {:.4} | {delta:+.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            acc.ppl(),
            acc.band_nll[0] / tot,
            acc.band_nll[1] / tot,
            acc.band_nll[2] / tot,
            if acc.early_n > 0 { acc.early_nll / acc.early_n as f64 } else { f64::NAN }
        ));
    }
    out.push_str("\nG3/G4 are pilot-slice readouts (the full per-family retention walk + the kv_sink_window check stay deferred; see the issue). A 2-bit PPL regression beside a Gate-1 pass is the recorded absmax-caveat confirmation, not a contradiction.\n");
    out.push_str("\nMEASUREMENT LANE (P0 law): no serving claim — Gate 2 decides. Box state recorded beside the numbers in the bench note.\n");

    print!("{out}");
    if let Some(p) = report_path {
        std::fs::write(&p, &out).with_context(|| format!("write report {}", p.display()))?;
        eprintln!("# report written: {}", p.display());
    }
    Ok(())
}

/// The CalibrationTables builder clamps top_k to the seen vocabulary;
/// report the requested dial as-typed (the run's coverage line is the
/// effective value).
fn cal_tables_top_k(top_k: usize) -> usize {
    top_k
}
