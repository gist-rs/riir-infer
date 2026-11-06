use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_gpu::qwen38_specdec::{DFlashCpu, StFile, parse_vendor_config};

fn main() {
    let spec = StFile::open(std::path::Path::new("/Users/katopz/uzu27b/speculator/model.safetensors")).unwrap();
    let cfg = parse_vendor_config(&std::fs::read_to_string("/Users/katopz/uzu27b/speculator/config.json").unwrap()).unwrap();
    let mut draft = DFlashCpu::load(&spec, &cfg).unwrap();
    let gg = GgufFile::open(std::path::Path::new("/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf")).unwrap();
    let t_prev = gg.dequant_tensor_row("token_embd.weight", 198, 5120).unwrap();
    let mask = gg.dequant_tensor_row("token_embd.weight", 248070, 5120).unwrap();
    let (_, lines) = draft.draft_traced(&t_prev, &mask, true).unwrap();
    for l in lines {
        println!("{l}");
    }
}
// (rope dump appended by a separate example below)
