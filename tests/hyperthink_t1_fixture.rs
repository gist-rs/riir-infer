//! Fixture gate for the Issue 920 T1 bias-delta sidecar (`hyperthink_t1`
//! lane: the gemma-2-2b delta-content capture).
//!
//! The committed `tests/fixtures/hyperthink/*` artifacts are the BLAKE3-pinned
//! provenance of the T1 run (the full 2,000-probe capture on
//! gemma-2-2b-it-f16). The pins are data: a sidecar edit or a truncated
//! re-write reds here. No semantic re-derivation — the
//! `hyperthink_t1_delta_census` bin owns the measurement; this gate owns the
//! bytes plus the pre-registered premise verdict they record.
//!
//! DEFERRED to the M5 box (2026-10-07): the M3 full capture measured
//! ~36 s/probe (≈ 20 h projected) and was killed at probe ~425/2000; the
//! partial `.raw` artifacts were discarded (the run is deterministic — it
//! restarts from scratch). Until the sidecar + its `.blake3` pin land in
//! `tests/fixtures/hyperthink/`, both tests SKIP LOUD (a deferral, never a
//! green zero) and go live on their own once the rerun's artifacts are
//! copied in.

use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hyperthink")
}

/// Loud-skip guard: the sidecar is absent until the M5 census rerun lands
/// its artifacts — a deferral is printed, never a silent green zero.
fn fixtures_present() -> bool {
    let dir = fixture_dir();
    let sidecar = dir.join("gemma-2-2b-it-f16.t1_bias_delta.json");
    let pin = dir.join("gemma-2-2b-it-f16.t1_bias_delta.json.blake3");
    if sidecar.is_file() && pin.is_file() {
        return true;
    }
    eprintln!(
        "SKIP (T1 deferred to the M5 box): sidecar/pin absent under {} — \
         rerun hyperthink_t1_delta_census there, then copy both files in",
        dir.display()
    );
    false
}

/// The sidecar's blake3 pin rides beside it (the 919 convention).
#[test]
fn t1_sidecar_matches_its_blake3_pin() {
    if !fixtures_present() {
        return;
    }
    let dir = fixture_dir();
    let pin = std::fs::read_to_string(dir.join("gemma-2-2b-it-f16.t1_bias_delta.json.blake3"))
        .expect("blake3 pin present");
    let digest = pin
        .split_whitespace()
        .next()
        .expect("pin names a digest");
    let data =
        std::fs::read(dir.join("gemma-2-2b-it-f16.t1_bias_delta.json")).expect("sidecar present");
    assert_eq!(
        blake3::hash(&data).to_hex().as_str(),
        digest,
        "T1 sidecar drifted from its BLAKE3 pin"
    );
}

/// The sidecar is the FULL pre-registered run (never a pilot): 2,000 probes,
/// the probe + prompt-c pins, the six-site/K-excluded protocol echo, and the
/// pre-registered premise verdict block — whatever it says.
#[test]
fn t1_sidecar_is_the_full_pre_registered_run() {
    if !fixtures_present() {
        return;
    }
    let j = std::fs::read_to_string(
        fixture_dir().join("gemma-2-2b-it-f16.t1_bias_delta.json"),
    )
    .expect("sidecar present");
    for needle in [
        "\"lane\": \"issue920_t1_bias_delta\"",
        "\"probes\": 2000",
        "\"k_site\": \"excluded by construction (softmax shift-invariance, the paper's own exclusion)\"",
        "\"premise_verdict\":",
        "\"c1_pass\":",
        "\"c2_pass\":",
        "\"overall\":",
    ] {
        assert!(j.contains(needle), "sidecar missing disclosure: {needle}");
    }
    // The probe fixture's blake3 must be present and well-formed — the
    // sidecar is only meaningful against the pinned probe file.
    let probes_pin_idx = j
        .find("\"probes_blake3\": \"")
        .expect("probes blake3 pinned");
    let rest = &j[probes_pin_idx + "\"probes_blake3\": \"".len()..];
    let hex = rest.split('"').next().expect("hex digest");
    assert_eq!(hex.len(), 64, "blake3 hex is 64 chars");
}
