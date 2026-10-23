//! The G3 latency posture (`.issues/015`): per-chunk p50/p99 at the
//! 256 ms cadence over the real model, BOTH load postures (A: the
//! published pre-compiled `.mlmodelc`; B: the locally compiled
//! `.mlpackage`), box state printed beside every number. Warm-up
//! excluded; nearest-rank percentiles with the tail support disclosed
//! (the percentile-index law — `idx = ceil(p·n) − 1`, so p99 at n=1000
//! carries 10 samples).
//!
//! ```text
//! cargo run --release -p riir-infer-audio --features vad-ane \
//!   --example vad_timing -- [chunks=1000]
//! ```

#[cfg(all(feature = "vad-ane", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use riir_infer_audio::vad::{CHUNK_SAMPLES, SileroVad};

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    println!(
        "chunks={n} chunk_ms={} (256 ms cadence)",
        CHUNK_SAMPLES as f32 / 16.0
    );

    for (posture, seated) in [
        (
            "A (published pre-compiled .mlmodelc v6.2.1)",
            SileroVad::load_from_default_dir().map(|o| o.map(|s| (s.0, s.1))),
        ),
        (
            "B (locally compiled .mlpackage v6.0.0)",
            SileroVad::load_package_from_default_dir().map(|o| o.map(|s| (s.0, s.1))),
        ),
    ] {
        let Some((mut vad, placement)) = seated? else {
            println!("{posture}: SKIP LOUD — bundle absent (scripts/fetch_silero_vad.sh)");
            continue;
        };
        println!(
            "{posture}: device={} ane={} gpu={} cpu={} transitions={} passes_strict={}",
            placement.device_ops,
            placement.ane_ops,
            placement.gpu_ops,
            placement.cpu_ops,
            placement.transitions,
            placement.passes(),
        );

        // Deterministic warm-up (excluded): the synth stream, 20 chunks.
        for i in 0..20 {
            let p = vad.process(&riir_infer_audio::vad::synth::speech_shaped(i as f64))?;
            std::hint::black_box(p);
        }

        let mut samples: Vec<f32> = Vec::with_capacity(n);
        let mut probs: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let chunk = riir_infer_audio::vad::synth::speech_shaped(i as f64);
            let t = std::time::Instant::now();
            let p = vad.process(&chunk)?;
            let dt = t.elapsed();
            samples.push(dt.as_secs_f32() * 1e3);
            probs.push(p);
        }

        samples.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f64| -> (f32, usize) {
            let idx = ((p * n as f64).ceil() as usize)
                .saturating_sub(1)
                .min(n - 1);
            (samples[idx], n - idx)
        };
        let (p50, sup50) = pct(0.50);
        let (p99, sup99) = pct(0.99);
        let max = samples[n - 1];
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        println!(
            "latency_ms: p50={p50:.3} (tail {sup50}) p99={p99:.3} (tail {sup99}) max={max:.3} mean={mean:.3}"
        );
        println!(
            "realtime: rtf_p50={:.5} rtf_p99={:.5} (chunk_budget=256 ms)",
            p50 / 256.0,
            p99 / 256.0
        );
        println!(
            "probs over the modulated stream: min={:.4} max={:.4} last={:.4}",
            probs.iter().cloned().fold(f32::INFINITY, f32::min),
            probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            probs[n - 1],
        );
    }

    // Box state beside every number (the latency-provenance law; the
    // authoritative wrapper is ../riir-reflex/scripts/bench_preflight.sh
    // — quote its PROVENANCE line in the record).
    println!(
        "box: available_parallelism={:?}",
        std::thread::available_parallelism()
    );
    Ok(())
}

#[cfg(not(all(feature = "vad-ane", target_os = "macos")))]
fn main() {
    eprintln!("vad_timing: the vad-ane lane is macOS-only");
    std::process::exit(3);
}
