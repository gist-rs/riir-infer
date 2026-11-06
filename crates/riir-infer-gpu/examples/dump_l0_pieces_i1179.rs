//! Dump my L0 conv weights + folded kernel_projection for the uzu probe.

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_gpu::qwen38_specdec::{DFlashCpu, StFile, parse_vendor_config};

fn main() {
    let spec = StFile::open(std::path::Path::new("/Users/katopz/uzu27b/speculator/model.safetensors")).unwrap();
    let cfg = parse_vendor_config(&std::fs::read_to_string("/Users/katopz/uzu27b/speculator/config.json").unwrap()).unwrap();
    let mut draft = DFlashCpu::load(&spec, &cfg).unwrap();
    let kp = draft.dump_l0_mixer_kernel_proj().to_vec();
    std::fs::write("/tmp/i1179_l0_kernel_proj.f32", bytemuck::cast_slice(&kp)).unwrap();
    println!("kernel proj dumped ({} f32)", kp.len());
    let gg = GgufFile::open(std::path::Path::new("/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf")).unwrap();
    // Conv weights via the artifact (BF16 [5120, 2]).
    let pre = spec.f32_vec_pub("speculator.draft_model.layers.0.mixer_conv.pre_conv.weights").unwrap();
    let post = spec.f32_vec_pub("speculator.draft_model.layers.0.mixer_conv.post_conv.weights").unwrap();
    let mpre = spec.f32_vec_pub("speculator.draft_model.layers.0.mlp_conv.pre_conv.weights").unwrap();
    let mpost = spec.f32_vec_pub("speculator.draft_model.layers.0.mlp_conv.post_conv.weights").unwrap();
    let mut all = Vec::new();
    all.extend_from_slice(bytemuck::cast_slice(&pre));
    all.extend_from_slice(bytemuck::cast_slice(&post));
    all.extend_from_slice(bytemuck::cast_slice(&mpre));
    all.extend_from_slice(bytemuck::cast_slice(&mpost));
    std::fs::write("/tmp/i1179_conv_weights.f32", &all).unwrap();
    println!("conv weights dumped (4 x [5120,2] f32)");
    let _ = gg;
}
