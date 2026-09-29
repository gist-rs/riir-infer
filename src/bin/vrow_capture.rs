//! `vrow_capture` — katgpt-rs Issue 907's fixture half: dump REAL post-W_V
//! value rows from gemma-2-2b-it f16 decode so a katgpt-kv bench can replay
//! them through the KVarN bit arms (the Issue-907 ladder-inversion
//! attribution: b2/b3/b4 are DIFFERENT quantizers via the `with_config`
//! machinery split — skip-varn + grouped-4 RTN at b≤2 vs per-tile var-norm
//! at b≥3 — and the measured inversion must be attributed to machinery vs
//! width on REAL rows).
//!
//! The T1 tap law (Issue 883 trap 1 / `forward_gemma2_f16_tapped`): V is
//! tapped after W_V, exactly where the cache path consumes it. This bin
//! walks the same chat_probe stream the T1/T2 instruments use (the anomaly
//! was measured on that stream — same posture, same rows), captures every
//! layer's V row per position, and dumps a headered binary:
//!
//! ```text
//! magic "VROW001" | u32 n_layers | u32 kv_dim | u32 rows_per_layer |
//! blake3-hex corpus (64 bytes ASCII) | rows layer-major (f32 LE)
//! ```
//!
//! Fixture is GITIGNORED (`.raw/vrow/`); the consumer bench pins the file's
//! BLAKE3 and skips loud without it — the gitignored-fixture law.
//!
//! ```text
//! cargo run --release -p riir-infer-core --features vk_calibration --bin vrow_capture -- \
//!   --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf \
//!   --corpus ../riir-train/data/chat_probe \
//!   --tokens 2048 --seq-len 1024 --out .raw/vrow/gemma2_vrows.bin
//! ```

use std::io::Write as _;

use anyhow::{Context, Result, bail};
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::gemma2_calibration::{
    forward_gemma2_f16_tapped, CalibrationTables,
};
use riir_infer_core::transformer::{ForwardContext, NoHook};
use riir_infer_core::types::kv_dim;
use katgpt_transformer::MultiLayerKVCache;

fn main() -> Result<()> {
    let mut gguf_path = String::from("../riir-train/data/gemma-2-2b-it-f16.gguf");
    let mut corpus_path = String::from("../riir-train/data/chat_probe");
    let mut tokens_n = 2048usize;
    let mut seq_len = 1024usize;
    let mut out_path = String::from(".raw/vrow/gemma2_vrows.bin");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf_path = args.next().context("--gguf needs a path")?,
            "--corpus" => corpus_path = args.next().context("--corpus needs a path")?,
            "--tokens" => tokens_n = args.next().context("--tokens needs N")?.parse()?,
            "--seq-len" => seq_len = args.next().context("--seq-len needs N")?.parse()?,
            "--out" => out_path = args.next().context("--out needs a path")?,
            other => bail!("unknown arg {other}"),
        }
    }
    assert!(
        (2..=4096).contains(&seq_len),
        "--seq-len must be in [2, 4096] (the in-repo stack is exact only below the sliding window)"
    );

    // ── model + tokenizer (the kv_plus_ladder shape) ──
    let t0 = std::time::Instant::now();
    let gguf = GgufFile::open(std::path::Path::new(&gguf_path)).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let mut config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let corpus_text = load_corpus_text(std::path::Path::new(&corpus_path))
        .with_context(|| format!("read corpus {corpus_path}"))?;
    let corpus_blake3 = blake3::Hasher::new()
        .update(corpus_text.as_bytes())
        .finalize()
        .to_hex()
        .to_string();
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    drop(gguf);
    let n_layers = config.n_layer;
    let kvd = kv_dim(&config);
    println!(
        "# vrow_capture: gemma-2 f16 | layers={n_layers} kv_dim={kvd} vocab={} | load {:.1}s",
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );

    // ── the frozen token stream, chunked (the T1 posture) ──
    let all = tok.encode(&corpus_text);
    let take = all.len().min(tokens_n);
    let tokens: Vec<usize> = all[..take].to_vec();
    let chunks: Vec<Vec<usize>> = tokens.chunks(seq_len).map(<[usize]>::to_vec).collect();
    let rows_total = tokens.len();
    println!(
        "# corpus: {} chars → {} tokens → {} chunks × ≤{seq_len}",
        corpus_text.len(),
        tokens.len(),
        chunks.len()
    );

    // block_size sizes the KV rows + score scratch (the row_logit_floor law).
    config.block_size = seq_len;

    // The V/K tables are the tapped forward's contract (lane baggage; the
    // capture reads rows directly from the layer caches).
    let mut token_counts = vec![0u64; config.vocab_size];
    for &t in &tokens {
        token_counts[t] += 1;
    }
    let mut tables = CalibrationTables::from_counts(n_layers, kvd, token_counts, 1);

    // ── walk + capture: V rows straight out of each layer cache after the
    //    forward (the cache row at `pos` IS the post-W_V V for that token —
    //    the T1 tap law, no second copy) ──
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&out_path).with_context(|| format!("create {out_path}"))?,
    );
    out.write_all(b"VROW001")?;
    out.write_all(&(n_layers as u32).to_le_bytes())?;
    out.write_all(&(kvd as u32).to_le_bytes())?;
    out.write_all(&(rows_total as u32).to_le_bytes())?;
    // corpus BLAKE3 as 64 ASCII bytes
    let mut blake_bytes = [0u8; 64];
    blake_bytes.copy_from_slice(corpus_blake3.as_bytes());
    out.write_all(&blake_bytes)?;

    let mut row_buf = vec![0.0f32; kvd];
    let mut captured = 0usize;
    let t1 = std::time::Instant::now();
    for (ci, chunk) in chunks.iter().enumerate() {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        for (pos, &token) in chunk.iter().enumerate() {
            forward_gemma2_f16_tapped(
                &mut ctx, &weights, &mut cache, &mut tables, token, pos, &config, &mut NoHook,
            );
            // layer-major: for row `pos`, every layer's V row. The dump is
            // therefore [layer][pos-within-chunk] per chunk — the consumer
            // reads rows_per_layer × n_layers rows in file order.
            for layer in 0..n_layers {
                let off = pos * kvd;
                row_buf.copy_from_slice(&cache.layers[layer].value[off..off + kvd]);
                out.write_all(bytemuck::cast_slice::<f32, u8>(&row_buf))?;
            }
            captured += 1;
        }
        if (ci + 1) % 2 == 0 || ci + 1 == chunks.len() {
            eprintln!(
                "[vrow] chunk {}/{} | {captured} rows captured | {:.2} tok/s",
                ci + 1,
                chunks.len(),
                captured as f32 / t1.elapsed().as_secs_f32().max(1e-6)
            );
        }
    }
    out.flush()?;
    let meta = std::fs::metadata(&out_path)?;
    println!(
        "# wrote {out_path} ({} bytes) | layers={n_layers} kv_dim={kvd} rows_per_layer={rows_total} | corpus_blake3 {corpus_blake3}",
        meta.len()
    );
    Ok(())
}
