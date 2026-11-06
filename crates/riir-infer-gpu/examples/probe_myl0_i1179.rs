//! Issue 1179 bisect: MY CPU layer-0 on the uzu probe's exact input —
//! compare with /tmp/i1179_uzu_layer0_{hidden,shortcut}.f32.

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_gpu::qwen38_specdec::{DFlashCpu, StFile, parse_vendor_config};

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
    let t_prev = gg.dequant_tensor_row("token_embd.weight", 198, 5120).unwrap();
    let mask = gg.dequant_tensor_row("token_embd.weight", 248070, 5120).unwrap();
    let (my_h, my_s, my_h2) = draft.dump_layer0(&t_prev, &mask, false).unwrap();


    let uh: Vec<f32> = bytemuck::cast_slice(&std::fs::read("/tmp/i1179_uzu_layer0_hidden.f32").unwrap()).to_vec();
    let us: Vec<f32> = bytemuck::cast_slice(&std::fs::read("/tmp/i1179_uzu_layer0_shortcut.f32").unwrap()).to_vec();
    let max = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    println!("L0 hidden [value|gate]: my max={:.2} uzu max={:.2} cos={:.4}", max(&my_h), max(&uh), cosine(&my_h, &uh));
    println!("L0 hidden [gate|value]: my max={:.2} cos={:.4}", max(&my_h2), cosine(&my_h2, &uh));
    println!("L0 shortcut: my max={:.2} uzu max={:.2} cos={:.4}", max(&my_s), max(&us), cosine(&my_s, &us));
    println!("uzu hidden row0 first8 = {:?}", &uh[..8]);
    println!("my  hidden row0 first8 = {:?}", &my_h[..8]);
    std::fs::write("/tmp/i1179_my_layer0_hidden.f32", bytemuck::cast_slice(&my_h)).unwrap();
    std::fs::write("/tmp/i1179_my_layer0_shortcut.f32", bytemuck::cast_slice(&my_s)).unwrap();
    std::fs::write("/tmp/i1179_my_layer0_h2.f32", bytemuck::cast_slice(&my_h2)).unwrap();
    println!("h2 dumped (/tmp/i1179_my_layer0_h2.f32) max={:.3}", max(&my_h2));
}
