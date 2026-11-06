//! Issue 1179 bisect step 2: run MY CPU DFlash draft on the SAME inputs the
//! uzu probe used (the 4 dumped feature rows as the KV, token 198 as t_prev)
//! and print the argmax chain — compare with the uzu ground truth
//! [8404 ×8, 12 ×6].

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_gpu::qwen38_specdec::{DFlashCpu, StFile, parse_vendor_config};

fn main() {
    let spec = StFile::open(std::path::Path::new("/Users/katopz/uzu27b/speculator/model.safetensors")).unwrap();
    let cfg = parse_vendor_config(&std::fs::read_to_string("/Users/katopz/uzu27b/speculator/config.json").unwrap()).unwrap();
    let mut draft = DFlashCpu::load(&spec, &cfg).unwrap();
    let gg = GgufFile::open(std::path::Path::new("/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf")).unwrap();

    // Same KV: the 4 dumped feature rows (verify rows 0..3), packed.
    let dump = std::fs::read("/tmp/i1179_cycle0_dump.f32").unwrap();
    let f32s: Vec<f32> = bytemuck::cast_slice(&dump).to_vec();
    let layers = 5usize;
    let rows = 4usize;
    let d = 5120usize;
    let mut packed = vec![0.0f32; rows * d * layers];
    for li in 0..layers {
        for r in 0..rows {
            for i in 0..d {
                packed[(r * layers + li) * d + i] = f32s[(li * rows + r) * d + i];
            }
        }
    }
    draft.accept(&packed, rows).unwrap();
    println!("[cpu-probe] kv ctx_len = {}", draft.context_length());

    // Same t_prev: token 198, logical row (no embedding transform — matches
    // the dense mini-target table uzu gathered).
    let t_prev = gg.dequant_tensor_row("token_embd.weight", 198, d).unwrap();
    let mask = gg.dequant_tensor_row("token_embd.weight", 248070, d).unwrap();
    let hidden = draft.draft(&t_prev, &mask).unwrap();

    // Same readout: the GGUF lm_head rows (bf16 in the mini target; f32 here).
    let mut lm_head = vec![0.0f32; 248320 * d];
    for v in 0..248320 {
        let row = gg.dequant_tensor_row("output.weight", v, d).unwrap();
        lm_head[v * d..(v + 1) * d].copy_from_slice(&row);
    }
    let mut chain = Vec::new();
    for r in 1..16 {
        let row = &hidden[r * d..(r + 1) * d];
        let mut best = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for v in 0..248320 {
            let lrow = &lm_head[v * d..(v + 1) * d];
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += lrow[i] * row[i];
            }
            if acc > bv {
                bv = acc;
                best = v as u32;
            }
        }
        chain.push(best);
    }
    println!("[cpu-probe] chain tokens: {:?}", chain);
    println!("[cpu-probe] expected (uzu): [8404 x8, 12 x6]");
}
