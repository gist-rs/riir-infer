//! `laya-quant4 <checkpoint-dir>` — the Q4 artifact converter (Plan 616
//! Phase 3, the Q4/PQ2 seam): reads `<dir>/model.safetensors` (F16),
//! writes `<dir>/derived/model.q4.safetensors` (>=2D tensors as Q4_0
//! blocks — one f16 scale + 16 packed-nibble bytes per 32 weights, 1D
//! norms/biases carried F16) + the BLAKE3 sidecar.
//!
//! The converter REFUSES to emit an artifact whose read-back is not
//! byte-identical to the in-memory fake-quant Q4 of the same weights —
//! the format's no-drift proof runs on the real bytes here.
//!
//! ⚠ The Q4 grid's weight error is ~15× the Q8 grid's; its RETENTION is
//! D1-priced separately (instinct issue 018's own law) before
//! `LAYA_WEIGHTS_VARIANT=q4` serves anything. This bin ships the format
//! seam, never an adoption.
//!
//! Serve (once retention passes) with `LAYA_WEIGHTS_VARIANT=q4` (the
//! loader verifies the sidecar and refuses loud when the artifact is
//! missing — never auto-derived at load).

use riir_infer_laya::laya::riir::q4_artifact::convert_checkpoint;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: laya-quant4 <checkpoint-dir>   (the dir holding model.safetensors)");
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
                "  bytes {} F16 -> {} Q4 ({:.1}%)",
                rep.f16_bytes,
                rep.q4_bytes,
                100.0 * rep.q4_bytes as f64 / rep.f16_bytes as f64
            );
            println!("  blake3 {}", rep.digest);
            println!("  in {:?}", t.elapsed());
            println!("⚠ Q4 retention is D1-priced separately — measure before serving");
            println!("serve (once retention passes) with LAYA_WEIGHTS_VARIANT=q4");
        }
        Err(e) => {
            eprintln!("⛔ conversion refused: {e}");
            std::process::exit(1);
        }
    }
}
