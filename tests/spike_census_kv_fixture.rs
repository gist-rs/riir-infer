//! Fixture gate for the Issue 919 T3 KV-diagonal sidecars (both cells).
//!
//! The committed `tests/fixtures/spike_census_kv/*.kvdiag.json` artifacts are
//! the BLAKE3-pinned provenance of the cell runs (gemma-2-2b cell 1
//! `47efae7`, Bonsai-27B cell 2). The pins are data: a sidecar edit or a
//! truncated re-write reds here. No semantic re-derivation — the harness
//! bins own the measurement; this gate owns the bytes.

use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/spike_census_kv")
}

#[test]
fn kvdiag_sidecars_match_their_blake3_pins() {
    let dir = fixture_dir();
    let mut checked = 0usize;
    let entries = std::fs::read_dir(&dir).expect("fixture dir readable");
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("blake3") {
            continue;
        }
        let pin = std::fs::read_to_string(&path).expect("blake3 pin readable");
        let mut parts = pin.split_whitespace();
        let digest = parts.next().unwrap_or_else(|| panic!("pin {} malformed", path.display()));
        let name = parts.next().expect("pin names its sidecar");
        let sidecar = dir.join(name);
        let data = std::fs::read(&sidecar).unwrap_or_else(|e| panic!("sidecar {name}: {e}"));
        let actual = blake3::hash(&data).to_hex();
        assert_eq!(
            actual.as_str(),
            digest,
            "sidecar {name} drifted from its BLAKE3 pin"
        );
        checked += 1;
    }
    assert!(
        checked >= 2,
        "expected both cell sidecars pinned (gemma + bonsai), found {checked}"
    );
}

#[test]
fn bonsai_sidecar_discloses_the_hybrid_shape() {
    let p = fixture_dir().join("Ternary-Bonsai-2-27B-PQ2_0.kvdiag.json");
    let j = std::fs::read_to_string(&p).expect("bonsai sidecar present");
    for needle in [
        "\"attention_layers\": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]",
        "\"rows_observed\": 31488",
        "never-read dummies",
    ] {
        assert!(j.contains(needle), "sidecar missing disclosure: {needle}");
    }
}
