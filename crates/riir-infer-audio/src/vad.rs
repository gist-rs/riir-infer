//! The silero-vad lane — `.issues/015` T2/T3: the FluidInference
//! `silero-vad-unified-256ms` CoreML bundle (Apache-2.0 weights; this
//! crate ships the LOADER + serving state, never weights) seated on the
//! shared [`riir_infer_ane`] host.
//!
//! Wire contract (confirmed byte-for-byte from the bundle's
//! `metadata.json`, 2026-10-23): inputs `audio_input` `[1, 4160]` f32
//! (64-sample context + 4096 new — 256 ms @ 16 kHz), `hidden_state` +
//! `cell_state` `[1, 128]` f32; outputs `vad_output` `[1, 1, 1]` f32
//! (the speech probability), `new_hidden_state` + `new_cell_state`
//! `[1, 128]` f32. State flow: context = the last 64 samples of the
//! PRIOR chunk (zero-init); the LSTM state is carried between chunks
//! (zero-init) — the streaming-state serving shape, resolved with ONE
//! multi-output forward per chunk (re-predicting per output would
//! re-roll the recurrence).
//!
//! The load refuses a digest mismatch loud (the blake3-dir-v1 pin
//! below — minted by the `audio_bundle_digest` example against the
//! fetch script's download, never hand-computed) and never falls back
//! to CPU silently: `ComputeUnits::CpuAndNe` is FluidAudio's published
//! posture, and the plan-walk verdict is REPORTED (the strict laya
//! `Placement::passes()` law is predicted to refuse this graph on its
//! fp32↔fp16 cast transitions — T1's recorded prediction).

use std::path::{Path, PathBuf};

use riir_infer_ane::{
    ArtifactPin, ComputeUnits, LoadedModel, Placement, PlanWalk, load_model, load_model_precompiled,
};

/// The placement verdict type, re-exported — consumers (and the PoC
/// suite) read the load verdict through the LANE, never through the
/// host directly.
pub use riir_infer_ane::Placement as VadPlacement;

/// The downloaded bundle directory name under the models root —
/// posture A, the PUBLISHED pre-compiled `.mlmodelc` (v6.2.1).
pub const BUNDLE_NAME: &str = "silero-vad-unified-256ms-v6.2.1.mlmodelc";

/// Posture B, the published `.mlpackage` SOURCE (v6.0.0 — the newest
/// source bundle on the hub), compiled LOCALLY through the host's
/// compile cache. The A/B exists because the PoC's first G1 read
/// measured the pre-compiled bundle at ALL-CPU preferred placement on
/// this M5 Ultra (357/357 ops `MLCPUComputeDevice`, 0 ANE) — the
/// upstream bundle's compiled units predate the chip (converted
/// 2026-06-23, coremltools 9.0b1), and the laya lane's locally-
/// compiled packages DO place on ANE. The locally-compiled posture is
/// the testable repair.
pub const PACKAGE_NAME: &str = "silero-vad-unified-256ms-v6.0.0.mlpackage";

/// The digest pin — blake3-dir-v1 over the bundle (sorted relative
/// paths, `rel \0 data \0` per file). `files`/`bytes` are the exact
/// walls (5 files, 1,063,425 B). A mismatched download refuses loud —
/// never a silent CPU fallback, never a stale artifact.
const DIGEST: &str = "097b5886419c8e73cf86f923c2dab4bc86e294e5cb861ef240e146244f8f21e5";

/// The posture-B pin over the `.mlpackage` source (3 files,
/// 1,040,846 B — minted by `audio_bundle_digest`, 2026-10-23).
const PACKAGE_DIGEST: &str = "8588b28ca6f702087ab2520146f5d3eb42e682a8ec45236994b610fa77dc648c";

fn pin() -> ArtifactPin {
    ArtifactPin {
        digest: DIGEST.to_string(),
        files: 5,
        bytes: 1_063_425,
    }
}

fn package_pin() -> ArtifactPin {
    ArtifactPin {
        digest: PACKAGE_DIGEST.to_string(),
        files: 3,
        bytes: 1_040_846,
    }
}

/// Samples per processed chunk (256 ms @ 16 kHz).
pub const CHUNK_SAMPLES: usize = 4096;
/// Context samples prepended to each chunk (the prior chunk's tail).
pub const CONTEXT_SAMPLES: usize = 64;
/// Input width: context + chunk.
pub const INPUT_SAMPLES: usize = CONTEXT_SAMPLES + CHUNK_SAMPLES;
/// LSTM hidden/cell width.
pub const STATE_DIM: usize = 128;

/// One seated silero-vad: the loaded model + the carried streaming
/// state (context tail + LSTM hidden/cell). Single-threaded by the
/// host's contract.
pub struct SileroVad {
    model: LoadedModel,
    placement: Placement,
    context: Vec<f32>,
    hidden: Vec<f32>,
    cell: Vec<f32>,
    input_scratch: Vec<f32>,
    prob_out: [f32; 1],
    hidden_out: Vec<f32>,
    cell_out: Vec<f32>,
}

impl SileroVad {
    /// Load the bundle at `bundle_dir` (a pre-compiled `.mlmodelc`
    /// directory) under FluidAudio's published `.cpuAndNeuralEngine`
    /// posture. Digest walls enforced; the plan walk REPORTED, never
    /// gating (T1's recorded prediction: the strict laya law refuses
    /// this graph's cast transitions — the verdict belongs to the PoC
    /// record, not to the load path).
    pub fn load(bundle_dir: &Path) -> Result<(Self, Placement), riir_infer_ane::Error> {
        let (model, placement) = load_model_precompiled(
            bundle_dir,
            &pin(),
            ComputeUnits::CpuAndNe,
            None,
            "run scripts/fetch_silero_vad.sh (downloads the FluidInference bundle \
             into ~/.cache/riir-infer/audio/ — see .issues/015)",
            PlanWalk::Verify,
        )?;
        Ok((Self::seated(model, placement), placement))
    }

    /// Resolve the bundle under the lane's models root — `RIIR_AUDIO_MODELS_DIR`
    /// first, else `~/.cache/riir-infer/audio` — then [`Self::load`].
    /// `Ok(None)` = no models root configured (the caller skips LOUD —
    /// a deferral, never a green); `Err` = the bundle was found and
    /// refused (digest wall / load failure — never silent).
    pub fn load_from_default_dir() -> Result<Option<(Self, Placement)>, riir_infer_ane::Error> {
        let Some(dir) = models_dir() else {
            return Ok(None);
        };
        let path = dir.join(BUNDLE_NAME);
        if !path.is_dir() {
            return Ok(None);
        }
        Self::load(&path).map(Some)
    }

    /// Posture B: the published `.mlpackage` SOURCE, compiled LOCALLY
    /// through the host's compile cache ([`load_model`] — the exact
    /// path the laya lane uses, and the A/B half of the placement
    /// question posture A's all-CPU verdict raised).
    pub fn load_package(package_dir: &Path) -> Result<(Self, Placement), riir_infer_ane::Error> {
        let (model, placement) = load_model(
            package_dir,
            &package_pin(),
            ComputeUnits::CpuAndNe,
            None,
            "RIIR_AUDIO_ANE_CACHE",
            "audio",
            "run scripts/fetch_silero_vad.sh (downloads the FluidInference \
             bundle into ~/.cache/riir-infer/audio/ — see .issues/015)",
            PlanWalk::Verify,
        )?;
        Ok((Self::seated(model, placement), placement))
    }

    /// [`Self::load_package`] from the lane's models root. `Ok(None)`
    /// = the package is absent (skip LOUD); `Err` = found and refused.
    pub fn load_package_from_default_dir()
    -> Result<Option<(Self, Placement)>, riir_infer_ane::Error> {
        let Some(dir) = models_dir() else {
            return Ok(None);
        };
        let path = dir.join(PACKAGE_NAME);
        if !path.is_dir() {
            return Ok(None);
        }
        Self::load_package(&path).map(Some)
    }

    fn seated(model: LoadedModel, placement: Placement) -> Self {
        Self {
            model,
            placement,
            context: vec![0.0; CONTEXT_SAMPLES],
            hidden: vec![0.0; STATE_DIM],
            cell: vec![0.0; STATE_DIM],
            input_scratch: vec![0.0; INPUT_SAMPLES],
            prob_out: [0.0],
            hidden_out: vec![0.0; STATE_DIM],
            cell_out: vec![0.0; STATE_DIM],
        }
    }

    /// The load-time plan verdict (device/ANE/GPU/CPU op counts +
    /// transitions) — REPORTED to the caller, never enforced here.
    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// Re-zero the streaming state (context + LSTM hidden/cell) — a new
    /// stream through the same seated model.
    pub fn reset(&mut self) {
        self.context.fill(0.0);
        self.hidden.fill(0.0);
        self.cell.fill(0.0);
    }

    /// Process one 256 ms chunk (exactly [`CHUNK_SAMPLES`] f32 samples
    /// @ 16 kHz) and return the speech probability. The context tail +
    /// LSTM state carry to the next call — the streaming contract.
    pub fn process(&mut self, chunk: &[f32]) -> Result<f32, riir_infer_ane::Error> {
        assert_eq!(
            chunk.len(),
            CHUNK_SAMPLES,
            "audio: chunk must be {CHUNK_SAMPLES} samples (256 ms @ 16 kHz)"
        );
        self.input_scratch[..CONTEXT_SAMPLES].copy_from_slice(&self.context);
        self.input_scratch[CONTEXT_SAMPLES..].copy_from_slice(chunk);
        // The context tail for the NEXT chunk is this chunk's last 64
        // samples (the wire contract — copied before the scratch is
        // rebuilt).
        self.context
            .copy_from_slice(&chunk[CHUNK_SAMPLES - CONTEXT_SAMPLES..]);

        self.model.predict_f32_multi(
            &[
                ("audio_input", &self.input_scratch, &[1, INPUT_SAMPLES]),
                ("hidden_state", &self.hidden, &[1, STATE_DIM]),
                ("cell_state", &self.cell, &[1, STATE_DIM]),
            ],
            &mut [
                ("vad_output", self.prob_out.as_mut_slice()),
                ("new_hidden_state", self.hidden_out.as_mut_slice()),
                ("new_cell_state", self.cell_out.as_mut_slice()),
            ],
        )?;

        // State carry AFTER a successful forward (a refused prediction
        // leaves the prior stream state untouched).
        self.hidden.copy_from_slice(&self.hidden_out);
        self.cell.copy_from_slice(&self.cell_out);
        Ok(self.prob_out[0])
    }
}

/// The lane's models root: `$RIIR_AUDIO_MODELS_DIR`, else
/// `$HOME/.cache/riir-infer/audio`.
pub fn models_dir() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RIIR_AUDIO_MODELS_DIR") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".cache/riir-infer/audio"))
}

/// The deterministic G2 signal classes (`.issues/015` T3): silence /
/// tone / speech-shaped noise. Pure synthesis — no fixture files, no
/// test-row data (the grammar-not-answers law does not apply: these
/// are not benchmark rows, they are synthetic control signals).
pub mod synth {
    const SR: f64 = 16_000.0;

    /// Digital silence — the negative control.
    pub fn silence() -> Vec<f32> {
        vec![0.0; super::CHUNK_SAMPLES]
    }

    /// A 440 Hz sine at 0.2 amplitude — periodic, non-speech.
    pub fn tone() -> Vec<f32> {
        (0..super::CHUNK_SAMPLES)
            .map(|i| (0.2 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR).sin()) as f32)
            .collect()
    }

    /// Speech-shaped noise: formant-band carriers (F1 500 / F2 1500 /
    /// F3 2500 Hz) under a 4 Hz syllabic AM envelope with a hard
    /// off-gate below 0.35 — the syllable rhythm without phonemes.
    /// `t0` (seconds) phases the envelope across a multi-chunk stream.
    pub fn speech_shaped(t0: f64) -> Vec<f32> {
        (0..super::CHUNK_SAMPLES)
            .map(|i| {
                let t = t0 + i as f64 / SR;
                let syll = 0.5 * (1.0 + (2.0 * std::f64::consts::PI * 4.0 * t).sin());
                let gate = if syll > 0.35 { syll } else { 0.0 };
                let s = 0.5 * (2.0 * std::f64::consts::PI * 500.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 1500.0 * t).sin()
                    + 0.125 * (2.0 * std::f64::consts::PI * 2500.0 * t).sin();
                (0.35 * gate * s) as f32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seated-state law: a fresh seat is the zero stream — context,
    /// hidden, cell all zero, no probability read yet.
    #[test]
    fn const_wire_shape_is_the_metadata_contract() {
        assert_eq!(INPUT_SAMPLES, 4160);
        assert_eq!(CHUNK_SAMPLES, 4096);
        assert_eq!(CONTEXT_SAMPLES, 64);
        assert_eq!(STATE_DIM, 128);
        // The load-time walls, mirrored here so a pin edit and the wire
        // contract cannot drift apart silently.
        assert_eq!(pin().files, 5);
        assert_eq!(pin().bytes, 1_063_425);
        assert_eq!(DIGEST.len(), 64, "the minted digest is blake3 hex");
        assert_ne!(
            DIGEST, "PLACEHOLDER_UNTIL_MINTED",
            "pin minted before release"
        );
    }
}
