//! `laya-quant8 <checkpoint-dir>` — the Q8 artifact converter (instinct
//! issue 018 Lane D2a, the STORAGE tier): reads `<dir>/model.safetensors`
//! (F16), writes `<dir>/derived/model.q8.safetensors` (>=2D tensors as
//! Q8_0 blocks, 1D norms/biases carried F16) + the BLAKE3 sidecar.
//!
//! The converter REFUSES to emit an artifact whose read-back is not
//! byte-identical to the in-memory fake-quant of the same weights — the
//! adoption's no-numerics-change proof runs on the real bytes here, so
//! the Bench-0046 probe reads carry as the D2a cell re-seat.
//!
//! Serve with `LAYA_WEIGHTS_VARIANT=q8` (the loader verifies the sidecar
//! and refuses loud when the artifact is missing — never auto-derived at
//! load).

use riir_infer_laya::laya::riir::q8_artifact::convert_checkpoint;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: laya-quant8 <checkpoint-dir>   (the dir holding model.safetensors)");
        std::process::exit(2);
    }
    let dir = std::path::PathBuf::from(&args[1]);
    if !dir.join("model.safetensors").is_file() {
        eprintln!(
            "⛔ {}: no model.safetensors — not a checkpoint dir",
            dir.display()
        );
        std::process::exit(2);
    }
    let t = std::time::Instant::now();
    match convert_checkpoint(&dir) {
        Ok(rep) => {
            println!("✅ converted {}", dir.display());
            println!(
                "  tensors {} (quantized {}, skipped/named {})",
                rep.tensors,
                rep.quantized_tensors,
                rep.skipped_tensors.len()
            );
            println!("  elements {}", rep.quantized_elements);
            println!(
                "  bytes {} F16 -> {} Q8 ({:.1}% — the D2a estimate)",
                rep.f16_bytes,
                rep.q8_bytes,
                100.0 * rep.q8_bytes as f64 / rep.f16_bytes as f64
            );
            println!("  blake3 {}", rep.digest);
            println!("  in {:?}", t.elapsed());
            println!("serve with LAYA_WEIGHTS_VARIANT=q8");
        }
        Err(e) => {
            eprintln!("⛔ conversion refused: {e}");
            std::process::exit(1);
        }
    }
}
