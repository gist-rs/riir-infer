//! Issue 1179 diag probe: load the vendor draft, run ONE block forward on
//! synthetic embeddings (no KV), print max|v| per stage — finds where inf/NaN
//! enters.

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_gpu::qwen38_specdec::{DFlashCpu, StFile, parse_vendor_config};

fn stats(v: &[f32]) -> (f32, f32, usize) {
    let mut mx = 0.0f32;
    let mut nan = 0usize;
    for &x in v {
        if x.is_nan() {
            nan += 1;
        } else {
            mx = mx.max(x.abs());
        }
    }
    // crude inf count
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    (mx, inf as f32, nan)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 0.0 || nb <= 0.0 { 0.0 } else { dot / (na * nb) }
}

fn main() {
    let spec = StFile::open(std::path::Path::new("/Users/katopz/uzu27b/speculator/model.safetensors")).unwrap();
    let cfg = parse_vendor_config(&std::fs::read_to_string("/Users/katopz/uzu27b/speculator/config.json").unwrap()).unwrap();
    let mut draft = DFlashCpu::load(&spec, &cfg).unwrap();
    let gg = GgufFile::open(std::path::Path::new("/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf")).unwrap();
    let t_prev = gg.dequant_tensor_row("token_embd.weight", 79, 5120).unwrap();
    let mask = gg.dequant_tensor_row("token_embd.weight", 248070, 5120).unwrap();
    println!("emb t_prev max|.|={:?} mask max={:?}", stats(&t_prev).0, stats(&mask).0);
    println!("t_prev nan={} mask nan={}", stats(&t_prev).2, stats(&mask).2);
    // Norm-scales NaN check: reload just that tensor from the artifact.
    let scales = spec
        .f32_vec_pub("speculator.draft_model.layers.0.pre_mixer_norm.scales")
        .unwrap();
    let nan_scales = scales.iter().filter(|v| v.is_nan()).count();
    let mx = scales.iter().fold(0.0f32, |m, v| if v.is_nan() { m } else { m.max(v.abs()) });
    println!(
        "pre_mixer_norm.scales: len={} nan={} max|.|={mx} first6={:?}",
        scales.len(),
        nan_scales,
        &scales[..6]
    );

    // ── fold-verification: vendor step path vs the folded W_orig ──
    // (exposed through a pub probe fn in the module)
    let ok = riir_infer_gpu::qwen38_specdec::verify_fold(&spec, &cfg);
    println!("fold verification: {ok}");
    let hidden_trace = draft.draft_traced(&t_prev, &mask, true).unwrap();
    let hid = &hidden_trace.0;
    let per_row: Vec<(f32, usize)> = (0..16)
        .map(|r| {
            let row = &hid[r * 5120..(r + 1) * 5120];
            let (mx, _, nan) = stats(row);
            (mx, nan)
        })
        .collect();
    for (r, (mx, nan)) in per_row.iter().enumerate() {
        println!("  out row {r}: max={mx} nan={nan}");
    }
    let hidden_trace = draft.draft_traced(&t_prev, &mask, true).unwrap();
    for l in &hidden_trace.1 {
        println!("  {l}");
    }
    let hidden = hidden_trace.0;
    let (mx, inf, nan) = stats(&hidden);
    println!("draft_hidden max|.|={mx} inf={inf} nan={nan}");
    for (r, row) in hidden.chunks_exact(5120).take(3).enumerate() {
        let (mx, inf, nan) = stats(row);
        println!("  row {r}: max={mx:.4} inf={inf} nan={nan} first3={:?}", &row[..3]);
    }
}
