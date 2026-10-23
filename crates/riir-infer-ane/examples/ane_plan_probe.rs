//! The "does anything place on ANE on this box" instrument (issue 015
//! T1's follow-up): walk the compute plan of any compiled `.mlmodelc`
//! directory and print the placement counts. Read-only — no digest
//! wall, no `MLModel` load. The M5-Ultra all-CPU verdict on the silero
//! bundle needed controls that are not the silero graph; point this at
//! system bundles (`/System/Library/**/*.mlmodelc`) or lane bundles.
//!
//! ```text
//! cargo run --release -p riir-infer-ane --example ane_plan_probe -- \
//!   /path/to/bundle.mlmodelc [cpuAndNe|cpuAndGpu|cpuOnly] [function]
//! ```

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use riir_infer_ane::{ComputeUnits, probe_compute_plan};
    use std::path::PathBuf;

    let Some(path) = std::env::args().nth(1) else {
        eprintln!(
            "usage: ane_plan_probe <bundle.mlmodelc> [cpuAndNe|cpuAndGpu|cpuOnly] [function]"
        );
        std::process::exit(2);
    };
    let units = match std::env::args().nth(2).as_deref() {
        Some("cpuAndGpu") => ComputeUnits::CpuAndGpu,
        Some("cpuOnly") => ComputeUnits::CpuOnly,
        _ => ComputeUnits::CpuAndNe,
    };
    let function = std::env::args().nth(3);

    let placement = probe_compute_plan(&PathBuf::from(&path), units, function.as_deref())?;
    println!(
        "{path}: units={units:?} device={} ane={} gpu={} cpu={} transitions={} passes_strict={}",
        placement.device_ops,
        placement.ane_ops,
        placement.gpu_ops,
        placement.cpu_ops,
        placement.transitions,
        placement.passes(),
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ane_plan_probe: macOS-only");
    std::process::exit(3);
}
