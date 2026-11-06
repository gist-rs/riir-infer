fn main() {
    // MY rope tables for positions 0..16, dim 128, base 1e7 (half-split).
    let (cos, sin) = riir_infer_gpu::qwen38_specdec::rope_tables_for_probe(&(0..16).collect::<Vec<_>>());
    std::fs::write("/tmp/i1179_my_rope_cos.f32", bytemuck::cast_slice(&cos)).unwrap();
    std::fs::write("/tmp/i1179_my_rope_sin.f32", bytemuck::cast_slice(&sin)).unwrap();
    println!("my rope dumped");
}
