//! Issue 1146 — the prefill frame census (Metal/CubeCL, real Bonsai-2).
//!
//! Strip-down attribution over one 1-box chunked prefill: each posture masks
//! exactly one [`PREFILL_STAGE_NAMES`] bit off; the per-frame wall's drop vs
//! the ALL baseline is that class's frame cost. This is the decode census's
//! Δwall law (Bench 1005/1006, the Issue-1132 instrument's prefill twin) —
//! a stripped run produces garbage logits and that is fine: correctness is
//! irrelevant when profiling, the wall is still valid, and every posture
//! runs the SAME frame sequence (same widths, same positions) so the
//! KV-growth term and warm-up amortization cancel in the Δ.
//!
//! Postures: the three coarse stages (deltanet / attention / ffn), the
//! seven deltanet-chain sub-bits (gdn_norm .. gdn_residual), and the two
//! standalone tails (embed / tail). G-C1 additivity: the coarse strips
//! PARTITION the frame, so `deltanetΔ + attentionΔ + ffnΔ + embedΔ + tailΔ`
//! must land near the ALL wall — the printed residue is scratch alloc +
//! the per-frame sync + the truly unattributed.
//!
//! The Issue-1122 projection site timing ([`PREFILL_SITE_TIMING`]) is armed
//! during the ALL posture only; its rows are submission-side walls for the
//! Metal-tensor projection sites — secondary evidence, never the record
//! (a queued dispatch's wall is not its exec time; the strip Δs are).
//!
//! Bit-identity: the ALL posture runs twice + once before any mask was ever
//! touched; the final chunk's logits must be bit-exact across the three —
//! the mask machinery itself must not move a token.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1146 cargo test -p riir-infer-gpu --release \
//!   --features cubecl_runtime,ternary_gemm_batched,multi_host,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test bench_1146_prefill_frame_census -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` overrides the sibling-layout default (the same
//! override the plan-623 G1 gate uses).

#![cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "multi_host"
))]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::ternary_deltanet_gpu_forward::{
    PREFILL_STAGE_ALL, PREFILL_STAGE_ATTENTION, PREFILL_STAGE_BETA, PREFILL_STAGE_DELTANET,
    PREFILL_STAGE_EMBED, PREFILL_STAGE_FFN, PREFILL_STAGE_GDN_NORM, PREFILL_STAGE_GDN_POST,
    PREFILL_STAGE_GDN_PREWORK, PREFILL_STAGE_GDN_REC, PREFILL_STAGE_GDN_RESIDUAL,
    PREFILL_STAGE_GDN_ROTIN, PREFILL_STAGE_TAIL, prefill_site_timing_reset,
    prefill_site_timing_snapshot, set_prefill_site_timing_armed, set_prefill_stage_mask,
};

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

/// Chunk width — the shipped serving width (Bench 911's per-frame floor
/// posture). The wide-chunk comparison re-runs the ALL posture at 4× this.
const CHUNK: usize = 128;
/// Frames per pass (total prompt = CHUNK × FRAMES tokens).
const FRAMES: usize = 8;
/// Untimed compile pass per posture (kernel JIT + cache warm).
const WARMUP_PASSES: usize = 1;
/// Timed passes per posture; the per-frame wall is the pass total / FRAMES.
const TIMED_PASSES: usize = 4;

struct Posture {
    name: &'static str,
    /// Bits cleared for this posture (0 = the ALL baseline).
    strip: u32,
}

fn postures() -> Vec<Posture> {
    vec![
        Posture {
            name: "ALL",
            strip: 0,
        },
        Posture {
            name: "-deltanet",
            strip: 1 << PREFILL_STAGE_DELTANET,
        },
        Posture {
            name: "-attention",
            strip: 1 << PREFILL_STAGE_ATTENTION,
        },
        Posture {
            name: "-ffn",
            strip: 1 << PREFILL_STAGE_FFN,
        },
        Posture {
            name: "-gdn_norm",
            strip: 1 << PREFILL_STAGE_GDN_NORM,
        },
        Posture {
            name: "-gdn_rotin",
            strip: 1 << PREFILL_STAGE_GDN_ROTIN,
        },
        Posture {
            name: "-gdn_prework",
            strip: 1 << PREFILL_STAGE_GDN_PREWORK,
        },
        Posture {
            name: "-gdn_beta",
            strip: 1 << PREFILL_STAGE_GDN_BETA,
        },
        Posture {
            name: "-gdn_rec",
            strip: 1 << PREFILL_STAGE_GDN_REC,
        },
        Posture {
            name: "-gdn_post",
            strip: 1 << PREFILL_STAGE_GDN_POST,
        },
        Posture {
            name: "-gdn_residual",
            strip: 1 << PREFILL_STAGE_GDN_RESIDUAL,
        },
        Posture {
            name: "-embed",
            strip: 1 << PREFILL_STAGE_EMBED,
        },
        Posture {
            name: "-tail",
            strip: 1 << PREFILL_STAGE_TAIL,
        },
    ]
}

fn tokens(count: usize, bos: usize, vocab: usize) -> Vec<usize> {
    (0..count)
        .map(|i| {
            if i == 0 {
                bos
            } else {
                (bos + i * 7919) % vocab
            }
        })
        .collect()
}

/// One pass over the whole prompt in CHUNK-wide frames. Returns the pass wall
/// and the FINAL frame's logits bits (identity evidence; empty when a stripped
/// tail produced none worth comparing — the caller only compares ALL runs).
fn run_pass(
    fwd: &mut TernaryDeltanetGpuForward,
    toks: &[usize],
    chunk: usize,
    n_layer: usize,
) -> (Duration, Vec<u8>) {
    let prompt_len = toks.len();
    let mut wall = Duration::ZERO;
    let mut last_bits = Vec::new();
    for (base, chunk_toks) in toks.chunks(chunk).enumerate() {
        let is_final = base + chunk_toks.len() == prompt_len;
        let t0 = Instant::now();
        let out = fwd.prefill_range_chunk(
            chunk_toks,
            None,
            base * chunk,
            prompt_len,
            LayerRange::new(0, n_layer as u32),
            is_final,
        );
        // The sync: on non-final frames the logits buffer is stale but the
        // read forces the queue to drain — the same cost every posture pays,
        // so it cancels in the Δ. On the final frame it is the real readback.
        let logits = fwd.read_logits();
        wall += t0.elapsed();
        if is_final {
            last_bits = out
                .iter()
                .chain(logits.iter())
                .flat_map(|f| f.to_bits().to_le_bytes())
                .collect();
        }
    }
    (wall, last_bits)
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn bench_1146_prefill_frame_census() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let (config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    assert!(
        weights.rotation.is_some(),
        "{}: not a Hadamard-folded file",
        model.display()
    );
    let prompt_len = CHUNK * FRAMES;
    let mut config = config;
    config.block_size = prompt_len + 1;
    let n_layer = config.n_layer;
    let bos = config.vocab_size / 2;
    let toks = tokens(prompt_len, bos, config.vocab_size);
    println!(
        "i1146 census: n_layer {n_layer}, prompt {prompt_len} = {FRAMES}×{CHUNK}, \
         postures {}",
        postures().len()
    );

    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    // ── Bit-identity anchor: the never-touched run, before any mask call. ──
    fwd.reset_state();
    let (w0, bits_never) = run_pass(&mut fwd, &toks, CHUNK, n_layer);
    println!(
        "never-touched ALL pass: {:?} ({:.1} ms/frame)",
        w0,
        w0.as_secs_f64() * 1e3 / FRAMES as f64
    );

    // The all-bits baseline + the strips. First ALL run doubles as the
    // second identity anchor (the mask setter to ALL is a no-op vs never).
    let mut results: Vec<(&'static str, f64)> = Vec::new();
    let mut all_bits_first = Vec::new();
    for posture in postures() {
        set_prefill_stage_mask(PREFILL_STAGE_ALL & !posture.strip);
        if posture.strip == 0 {
            // Site timing rides the ALL posture only (submission-side rows —
            // secondary evidence; the strips are the record).
            prefill_site_timing_reset();
            set_prefill_site_timing_armed(true);
        }
        fwd.reset_state();
        for _ in 0..WARMUP_PASSES {
            run_pass(&mut fwd, &toks, CHUNK, n_layer);
        }
        let mut total = Duration::ZERO;
        let mut last_bits = Vec::new();
        for _ in 0..TIMED_PASSES {
            fwd.reset_state();
            let (w, bits) = run_pass(&mut fwd, &toks, CHUNK, n_layer);
            total += w;
            last_bits = bits;
        }
        if posture.strip == 0 {
            set_prefill_site_timing_armed(false);
            all_bits_first = last_bits;
        }
        let per_frame = total.as_secs_f64() * 1e3 / (TIMED_PASSES * FRAMES) as f64;
        results.push((posture.name, per_frame));
    }
    set_prefill_stage_mask(PREFILL_STAGE_ALL);

    // ── Third identity anchor: ALL again AFTER every strip ran. ──
    fwd.reset_state();
    let (_, bits_after) = run_pass(&mut fwd, &toks, CHUNK, n_layer);
    assert!(
        bits_never == all_bits_first && all_bits_first == bits_after,
        "i1146: the mask machinery moved a token — bit-identity anchors disagree"
    );
    println!("bit-identity: never-touched == ALL-first == ALL-after ✓");

    // ── The wide-chunk comparison (the issue's "wider chunks" question):
    //    same total prompt, 4× the frame width, ALL posture. ──
    set_prefill_stage_mask(PREFILL_STAGE_ALL);
    fwd.reset_state();
    for _ in 0..WARMUP_PASSES {
        run_pass(&mut fwd, &toks, CHUNK * 4, n_layer);
    }
    let mut wide = Duration::ZERO;
    for _ in 0..TIMED_PASSES {
        fwd.reset_state();
        let (w, _) = run_pass(&mut fwd, &toks, CHUNK * 4, n_layer);
        wide += w;
    }
    let wide_per_tok = wide.as_secs_f64() * 1e3 / (TIMED_PASSES * prompt_len) as f64;

    // ── The table. Δ = ALL − strip = the class's frame cost. ──
    let all_ms = results[0].1;
    println!(
        "\n=== i1146 prefill frame census (ms/frame, {}×{}) ===",
        FRAMES, CHUNK
    );
    for (name, ms) in &results {
        if *name == "ALL" {
            println!("{:>14}  {:8.2}  (baseline)", name, ms);
        } else {
            println!("{:>14}  {:8.2}  Δ {:8.2}", name, ms, all_ms - ms);
        }
    }
    let coarse: f64 = results
        .iter()
        .filter(|(n, _)| matches!(*n, "-deltanet" | "-attention" | "-ffn" | "-embed" | "-tail"))
        .map(|(_, ms)| all_ms - ms)
        .sum();
    println!(
        "G-C1 coarse additivity: {:.2} of {:.2} ms/frame attributed; residue {:.2} \
         (scratch alloc + sync + unattributed)",
        coarse,
        all_ms,
        all_ms - coarse
    );
    println!(
        "wide-chunk control: 4×{} frames read {:8.3} ms/token vs {:8.3} at {}",
        CHUNK * 4,
        wide_per_tok,
        all_ms / CHUNK as f64,
        CHUNK
    );

    let rows = prefill_site_timing_snapshot();
    if !rows.is_empty() {
        println!("\nprojection site timing (submission-side, ALL posture):");
        for (name, nanos, count, max_nanos) in rows {
            println!(
                "  {:>14}  {:9.2} ms/frame-total  n {:5}  max {:7.3} ms",
                name,
                nanos as f64 / 1e6 / TIMED_PASSES as f64,
                count / TIMED_PASSES as u64,
                max_nanos as f64 / 1e6
            );
        }
    }
}
