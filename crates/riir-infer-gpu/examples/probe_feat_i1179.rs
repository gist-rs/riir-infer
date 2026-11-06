//! Issue 1179 diag: capture the LAST layer's (63) residual at a prompt's
//! final position and reconstruct logits through final-norm + lm_head —
//! must match the forward's own logits if the capture semantics are right.

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;

fn main() {
    let gguf_path = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";
    let mut fwd = Qwen38WgpuForward::new(std::path::Path::new(gguf_path), 8192, 512).unwrap();
    let gg = GgufFile::open(std::path::Path::new(gguf_path)).unwrap();
    let tok = BpeTokenizer::from_gguf(&gg).unwrap();
    let prompt_text = std::fs::read_to_string("../riir-ai/scripts/uzu_t2_prompt.txt").unwrap();
    let prompt: Vec<u32> = tok.encode(&prompt_text).into_iter().map(|t| t as u32).collect();
    println!("prompt {} tokens", prompt.len());

    let vocab = fwd.config.vocab_size;
    let e = fwd.config.n_embd;
    let eps = fwd.config.rms_norm_eps;
    let mut lm_head = vec![0.0f32; vocab * e];
    for v in 0..vocab {
        let row = gg.dequant_tensor_row("output.weight", v, e).unwrap();
        lm_head[v * e..(v + 1) * e].copy_from_slice(&row);
    }
    let out_norm = gg.dequant_tensor_row("output_norm.weight", 0, e).unwrap();

    // Prefill WITHOUT capture, then forward the last token again? No —
    // prefill consumes the prompt; the LAST position's logits come from the
    // prefill itself. For the capture test: prefill all but the last token,
    // then forward_token_capture the LAST token at its position.
    let (_, logits) = fwd.ingest_capture(&prompt[..prompt.len() - 1], 0, &[]).unwrap();
    let _ = logits;
    let last = *prompt.last().unwrap();
    let pos = prompt.len() - 1;
    let (_, logits63, caps) = fwd.forward_token_capture(last, pos, &[63]).unwrap();
    let feat = &caps[0];
    let sum_sq: f32 = feat.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sum_sq / e as f32 + eps).sqrt();
    let normed: Vec<f32> = feat.iter().zip(out_norm.iter()).map(|(&f, &w)| f * inv * w).collect();
    let mut top: Vec<(u32, f32)> = Vec::new();
    for (v, lrow) in lm_head.chunks_exact(e).enumerate().take(vocab) {
        let mut acc = 0.0f32;
        for i in 0..e { acc += lrow[i] * normed[i]; }
        if top.len() < 5 {
            top.push((v as u32, acc));
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else if acc > top[4].1 {
            top[4] = (v as u32, acc);
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    let mut actual: Vec<(u32, f32)> = logits63.iter().copied().enumerate().map(|(i, v)| (i as u32, v)).collect();
    actual.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("reconstructed-from-capture top5 = {:?}", &top[..5]);
    println!("forward's own logits   top5 = {:?}", &actual[..5]);
    // cosine between the feat-normed direction and the forward's normed x
    // (the forward's norm_x is internal; compare the reconstructed logits
    // vector cosine instead)
    let dot: f32 = top.iter().zip(actual.iter()).map(|(a, b)| a.1 * b.1).sum();
    println!("(top5 comparison — equal lists ⇒ capture semantics correct)");
}
