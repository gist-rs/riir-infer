//! The PoC gate suite (`.issues/015` G1–G3) over the REAL downloaded
//! bundle: load, the synthetic verdicts, and state-carry determinism.
//! Every test is `#[ignore]`d (real-model material), executed
//! explicitly:
//!
//! ```text
//! cargo test -p riir-infer-audio --features vad-ane \
//!   --test silero_vad_poc -- --ignored --nocapture
//! ```
//!
//! Skip law: without the bundle every test prints SKIP LOUD naming the
//! fetch remedy and returns — a deferral, never a green.

#![cfg(all(feature = "vad-ane", target_os = "macos"))]

use riir_infer_audio::vad::{self, SileroVad, VadPlacement};

fn seat_or_skip(tag: &str) -> Option<(SileroVad, VadPlacement)> {
    match SileroVad::load_from_default_dir() {
        Ok(Some(seated)) => Some(seated),
        Ok(None) => {
            let dir = vad::models_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "($HOME unset)".into());
            println!(
                "SKIP LOUD [{tag}]: no bundle under {dir} — run scripts/fetch_silero_vad.sh (a deferral, never a green)"
            );
            None
        }
        Err(e) => panic!("{tag}: the bundle was found and REFUSED: {e}"),
    }
}

fn seat_package_or_skip(tag: &str) -> Option<(SileroVad, VadPlacement)> {
    match SileroVad::load_package_from_default_dir() {
        Ok(Some(seated)) => Some(seated),
        Ok(None) => {
            println!(
                "SKIP LOUD [{tag}]: no {name} under the models root — run scripts/fetch_silero_vad.sh",
                name = vad::PACKAGE_NAME
            );
            None
        }
        Err(e) => panic!("{tag}: the package was found and REFUSED: {e}"),
    }
}

/// Render one short spoken sentence through macOS `say` → mono 16 kHz
/// f32 — the REAL-speech positive control (TTS, local, deterministic
/// voice; no fixture files, no test-row data). Returns chunks of
/// [`vad::CHUNK_SAMPLES`].
fn tts_speech_chunks() -> Option<Vec<Vec<f32>>> {
    let tmp = std::env::temp_dir().join(format!(
        "riir_audio_tts_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&tmp).ok()?;
    let aiff = tmp.join("utt.aiff");
    let wav = tmp.join("utt.wav");
    let said = std::process::Command::new("say")
        .args([
            "-v",
            "Samantha",
            "-o",
            aiff.to_str()?,
            "The quick brown fox jumps over the lazy dog while five weary miners rest beside the river.",
        ]
        .iter()
        .copied())
        .output();
    let said = said.ok()?;
    if !said.status.success() {
        eprintln!(
            "SKIP LOUD [tts]: `say` failed ({}) — the positive control is deferred, never a green",
            String::from_utf8_lossy(&said.stderr).trim()
        );
        let _ = std::fs::remove_dir_all(&tmp);
        return None;
    }
    let conv = std::process::Command::new("afconvert")
        .args(
            [
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                "-c",
                "1",
                aiff.to_str()?,
                wav.to_str()?,
            ]
            .iter()
            .copied(),
        )
        .output()
        .ok()?;
    if !conv.status.success() {
        eprintln!(
            "SKIP LOUD [tts]: afconvert failed ({}) — positive control deferred",
            String::from_utf8_lossy(&conv.stderr).trim()
        );
        let _ = std::fs::remove_dir_all(&tmp);
        return None;
    }
    let bytes = std::fs::read(&wav).ok()?;
    let _ = std::fs::remove_dir_all(&tmp);
    // WAVE: 44-byte canonical header, then LE i16 mono samples.
    let data = bytes.get(44..)?;
    let samples: Vec<f32> = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| i16::from_le_bytes([p[0], p[1]]) as f32 / 32768.0)
        .collect();
    let chunks: Vec<Vec<f32>> = samples
        .chunks(vad::CHUNK_SAMPLES)
        .filter(|c| c.len() == vad::CHUNK_SAMPLES)
        .map(|c| c.to_vec())
        .collect();
    if chunks.is_empty() {
        return None;
    }
    Some(chunks)
}

/// G1 — the model loads through the host's precompiled path; the plan
/// verdict is REPORTED (the strict `passes()` law is expected to refuse
/// this graph's cast transitions — T1's recorded prediction).
#[test]
#[ignore = "real-model material — run with --ignored (scripts/fetch_silero_vad.sh first)"]
fn g1_load_and_placement_verdict() {
    let Some((vad, placement)) = seat_or_skip("g1") else {
        return;
    };
    println!(
        "placement: device={} ane={} gpu={} cpu={} transitions={} passes_strict={}",
        placement.device_ops,
        placement.ane_ops,
        placement.gpu_ops,
        placement.cpu_ops,
        placement.transitions,
        placement.passes(),
    );
    // The load SUCCEEDED (G1); whatever the walk said is the record.
    let _ = vad;
}

/// G2 — the NEGATIVE-control + positive-control framing (re-registered
/// after the first read): the synthetic classes must all sit BELOW 0.1
/// (a VAD must not false-trigger on silence, tones, or speech-shaped
/// noise) while real speech (G2c) clears 0.5.
///
/// ⛔ RE-REGISTRATION DISCLOSED: the first read's pre-registered law
/// (ordinal silence < tone ≤ speech-shaped) was FALSIFIED — tone read
/// BELOW the silence baseline (0.0171 vs 0.0308). Recorded as-is in
/// `.issues/015`; the ordinal law was a wrong prior about the model's
/// internal baseline, not a pipeline fact (G2c proves the pipeline).
#[test]
#[ignore = "real-model material — run with --ignored (scripts/fetch_silero_vad.sh first)"]
fn g2_synthetic_negatives_stay_low() {
    let Some((mut vad, _)) = seat_or_skip("g2") else {
        return;
    };

    // Silence: the freshly-seated zero stream, settled over 8 chunks.
    let mut p_silence = 0.0;
    for _ in 0..8 {
        p_silence = vad.process(&vad::synth::silence()).expect("silence chunk");
    }

    // Tone: reset first so the classes don't share recurrence state.
    vad.reset();
    let mut p_tone = 0.0;
    for _ in 0..8 {
        p_tone = vad.process(&vad::synth::tone()).expect("tone chunk");
    }

    // Speech-shaped: a 12-chunk stream (3 s) so the envelope cycles.
    vad.reset();
    let mut p_speech_shaped = 0.0;
    for i in 0..12 {
        p_speech_shaped = vad
            .process(&vad::synth::speech_shaped(i as f64 * 0.256))
            .expect("speech-shaped chunk");
    }

    println!(
        "negatives: silence={p_silence:.4} tone={p_tone:.4} speech_shaped={p_speech_shaped:.4}"
    );
    assert!(
        p_silence < 0.1,
        "settled silence ({p_silence}) must read near-zero (T1 falsifiable prediction d — HELD)"
    );
    assert!(p_tone < 0.1, "tone ({p_tone}) must not false-trigger");
    assert!(
        p_speech_shaped < 0.1,
        "speech-shaped noise ({p_speech_shaped}) must not false-trigger — silero is \
         speech-specific; the near-floor read is the EXPECTED posture (T1's recorded risk), \
         the real-speech separation is G2c's"
    );
}

/// G1b — the A/B placement question posture A's all-CPU verdict
/// raised: does LOCALLY compiling the published `.mlpackage` restore
/// ANE preference on this chip? The verdict is REPORTED either way —
/// both postures are the PoC's answer, never smoothed.
#[test]
#[ignore = "real-model material — run with --ignored (scripts/fetch_silero_vad.sh first)"]
fn g1b_local_compile_placement_verdict() {
    let Some((vad, placement)) = seat_package_or_skip("g1b") else {
        return;
    };
    println!(
        "posture B (locally compiled .mlpackage): device={} ane={} gpu={} cpu={} transitions={} passes_strict={}",
        placement.device_ops,
        placement.ane_ops,
        placement.gpu_ops,
        placement.cpu_ops,
        placement.transitions,
        placement.passes(),
    );
    let _ = vad;
}

/// G2c — the REAL-speech positive control (macOS `say` TTS): if the
/// lane works end to end, real speech must clear the published working
/// point (silero's own threshold is 0.5) while the synthetic classes
/// sit near the ~0.03 baseline. This is the test the G2 synthetic
/// read's near-floor probabilities made necessary — the synthetic
/// classes alone cannot separate pipeline-correct from pipeline-broken.
#[test]
#[ignore = "real-model material — run with --ignored (scripts/fetch_silero_vad.sh first)"]
fn g2c_real_speech_positive_control() {
    let Some((mut vad, _)) = seat_or_skip("g2c") else {
        return;
    };
    let Some(chunks) = tts_speech_chunks() else {
        return; // SKIP LOUD already printed
    };
    let probs: Vec<f32> = chunks
        .iter()
        .map(|c| vad.process(c).expect("tts chunk"))
        .collect();
    let peak = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let tail = *probs.last().expect("non-empty");
    println!(
        "real speech: {} chunks, peak={peak:.4} last={tail:.4} series={:?}",
        chunks.len(),
        probs.iter().map(|p| format!("{p:.3}")).collect::<Vec<_>>()
    );
    assert!(
        peak > 0.5,
        "real speech must clear the published VAD working point (peak {peak}) — \
         below it, the input pipeline or state carry is broken, not the classes"
    );
}

/// The streaming-state law: two identical 16-chunk streams from reset
/// answer bit-identically — the recurrence carry is deterministic, no
/// hidden per-call state.
#[test]
#[ignore = "real-model material — run with --ignored (scripts/fetch_silero_vad.sh first)"]
fn state_carry_is_deterministic_across_reset() {
    let Some((mut vad, _)) = seat_or_skip("carry") else {
        return;
    };
    let run = |v: &mut SileroVad| -> Vec<f32> {
        v.reset();
        (0..16)
            .map(|i| {
                v.process(&vad::synth::speech_shaped(i as f64 * 0.256))
                    .expect("chunk")
            })
            .collect()
    };
    let a = run(&mut vad);
    let b = run(&mut vad);
    assert_eq!(a, b, "identical streams from reset must be bit-identical");
}
