//! Issue 1179 bisect step 1: synthesize the dense mini-target safetensors
//! (the GGUF's logical embedding + lm_head as BF16 matrices) for the uzu-side
//! probe, then read back the header to verify.

use std::path::Path;

fn main() {
    let gg = riir_infer_core::gguf_loader::GgufFile::open(Path::new(
        "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf",
    ))
    .unwrap();
    let vocab = 248320usize;
    let dim = 5120usize;

    // token_embd: Q4_K in this GGUF — dequant rows to f32 then to bf16
    // (the earlier bulk-copy assumption assumed BF16; the header says
    // Q4_K — row dequant is the loader's own path, same as the harness's).
    let mut emb_bytes = Vec::with_capacity(vocab * dim * 2);
    for v in 0..vocab {
        let row = gg.dequant_tensor_row("token_embd.weight", v, dim).unwrap();
        for x in row {
            emb_bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
        }
        if v % 100_000 == 0 && v > 0 {
            println!("emb {v}/{vocab}");
        }
    }
    // output.weight: Q6_K — dequant rows to f32 then to bf16.
    let mut out_bytes = Vec::with_capacity(vocab * dim * 2);
    for v in 0..vocab {
        let row = gg.dequant_tensor_row("output.weight", v, dim).unwrap();
        for x in row {
            out_bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
        }
        if v % 100_000 == 0 && v > 0 {
            println!("lm_head {v}/{vocab}");
        }
    }

    // Assemble the safetensors file: header + two tensors.
    let mut header = serde_json::Map::new();
    header.insert(
        "input_embedding.weights".to_string(),
        serde_json::json!({"dtype": "BF16", "shape": [vocab as u64, dim as u64], "data_offsets": [0, emb_bytes.len() as u64]}),
    );
    let out_start = emb_bytes.len() as u64;
    header.insert(
        "output_embedding.weights".to_string(),
        serde_json::json!({"dtype": "BF16", "shape": [vocab as u64, dim as u64], "data_offsets": [out_start, out_start + out_bytes.len() as u64]}),
    );
    // Specs: dense full precision (no quant, no incoherence) — the lookup
    // gathers raw rows, the readout is a plain matmul (exactly the harness's
    // own consumption of the GGUF tables).
    let mut meta = serde_json::Map::new();
    meta.insert(
        "input_embedding.spec".to_string(),
        serde_json::Value::String(
            serde_json::json!({"type": "FullPrecisionSpec", "layout": "input_output"}).to_string(),
        ),
    );
    meta.insert(
        "output_embedding.spec".to_string(),
        serde_json::Value::String(
            serde_json::json!({"type": "FullPrecisionSpec", "layout": "output_input"}).to_string(),
        ),
    );
    header.insert("__metadata__".to_string(), serde_json::Value::Object(meta));

    let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = Vec::new();
    file.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_json);
    file.extend_from_slice(&emb_bytes);
    file.extend_from_slice(&out_bytes);
    std::fs::write("/tmp/i1179_mini_target.safetensors", &file).unwrap();
    println!(
        "mini target written: {} bytes (emb {} + out {})",
        file.len(),
        emb_bytes.len(),
        out_bytes.len()
    );
}
