//! Plan 623 T5.1d — the trained MTP-head drafter: [`DraftSource`] over the
//! BMTH v1 artifact bytes (riir-train Issue 654 / Bench 677).
//!
//! The artifact (written by riir-train's `bonsai_mtp_head_fit`) is a
//! class-balanced ridge head over the folded runtime's post-final-layer
//! residual: `argmax_v(W[v]·x + b[v])` predicts the model's own greedy next
//! token from ONE matvec (support × d) + a first-index argmax — no lm_head,
//! no final norm, no basis reconstruction (the bake law: a linear head's
//! argmax is invariant to the final RMSNorm scale, gamma, and the Hadamard
//! basis).
//!
//! **Basis disclosure (measured, not assumed).** The training residual is
//! the POST-FINAL stream (`read_residual_x`, all 64 layers); the serving
//! wire carries the HEAD STAGE's own layer-window output — at a 2-stage
//! split `< n_layer` that is a mid-pipeline activation, a different
//! representation the head was not fit on. The runner-level seat is
//! width-checked (`MtpHeadDrafter::seat`), and the in-posture acceptance is
//! a MEASUREMENT (`examples/mtp_head_acceptance.rs` + the scoreboard row),
//! never an inherited claim: the fit split read 0.7929 per-slot on the
//! post-final basis; the split-`s` consumer reads whatever it reads.
//!
//! Wire format (BMTH v1, all little-endian — mirrors the riir-train
//! writer's `write_artifact` byte for byte):
//!
//! ```text
//! magic "BMTH" | version u32 = 1 | d u32 | vocab u32 | support u32
//! w:     support × d f32   (row-major, RAW-feature baked)
//! bias:  support     f32
//! ids:   support     u32   (the vocab id each slot predicts)
//! ```
//!
//! Exactly `20 + support·(d+2)·4` bytes — trailing junk refuses, as does a
//! corrupt id (≥ vocab) or a sidecar digest mismatch on
//! [`BmthHead::load_verified`].

use std::path::Path;

use crate::spec_fill::{DraftCtx, DraftSource};

/// BMTH (Bonsai MTP Head) artifact magic — must match the riir-train writer.
pub const BMTH_MAGIC: [u8; 4] = *b"BMTH";
/// BMTH wire version this reader consumes.
pub const BMTH_VERSION: u32 = 1;

/// The trained head: a support × d linear table over raw residual rows.
pub struct BmthHead {
    d: usize,
    vocab: u32,
    support: usize,
    /// support × d, row-major, RAW-feature baked (no standardization at
    /// the consumer — the bake already folded mu/rstd in).
    w: Vec<f32>,
    bias: Vec<f32>,
    ids: Vec<u32>,
    /// Reused per-`score` scratch (support wide) — the serving hot path
    /// allocates nothing.
    scores: Vec<f32>,
    /// The CUDA-placed scorer (the 654 gate's GPU-placement precondition
    /// on the 4090). `None` = the CPU sequential fold. Cfg'd out on macOS
    /// (the host module is CUDA-only); `place_on_cuda_device` refuses loud
    /// when the feature is off — an armed arm that silently fell back to
    /// CPU would make the e2e measure PLACEMENT, not the drafter.
    #[cfg(all(feature = "spec_mtp_cuda", not(target_os = "macos")))]
    device: Option<riir_infer_gpu::BmthHeadCuda>,
}

impl BmthHead {
    /// Parse the artifact bytes. Every corruption class refuses loud — a
    /// silent misread would draft plausible garbage for the whole run.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 20 {
            return Err("bmth: shorter than the 20-byte header".to_string());
        }
        let hdr: [u8; 20] = bytes[..20].try_into().expect("len checked above");
        if hdr[..4] != BMTH_MAGIC {
            return Err(format!(
                "bmth: bad magic {:?} (expected {BMTH_MAGIC:?})",
                &hdr[..4]
            ));
        }
        let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        if version != BMTH_VERSION {
            return Err(format!("bmth: version {version} != {BMTH_VERSION}"));
        }
        let d = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let vocab = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]);
        let support = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]) as usize;
        if d == 0 || vocab == 0 || support == 0 {
            return Err(format!(
                "bmth: degenerate shape d {d} vocab {vocab} support {support}"
            ));
        }
        let need = 20 + support * (d + 2) * 4;
        if bytes.len() != need {
            return Err(format!(
                "bmth: len {} != expected {need} (d {d}, vocab {vocab}, support {support})",
                bytes.len()
            ));
        }
        let mut at = 20;
        let w: Vec<f32> = bytes[at..at + support * d * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        at += support * d * 4;
        let bias: Vec<f32> = bytes[at..at + support * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        at += support * 4;
        let ids: Vec<u32> = bytes[at..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        if let Some(bad) = ids.iter().find(|&&id| id >= vocab) {
            return Err(format!("bmth: support id {bad} >= vocab {vocab}"));
        }
        Ok(Self {
            d,
            vocab,
            support,
            w,
            bias,
            ids,
            scores: vec![0.0; support],
            #[cfg(all(feature = "spec_mtp_cuda", not(target_os = "macos")))]
            device: None,
        })
    }

    /// Read the artifact file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("bmth read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    /// Read the artifact + verify it against the `.blake3` sidecar the
    /// riir-train writer leaves beside it (hex, optional trailing
    /// newline). A mismatch refuses — a drifted head drafts drivel.
    pub fn load_verified(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("bmth read {}: {e}", path.display()))?;
        let sidecar = Path::with_extension(path, "blake3");
        let want = std::fs::read_to_string(&sidecar)
            .map_err(|e| format!("bmth sidecar {}: {e}", sidecar.display()))?;
        let got = blake3::hash(&bytes).to_string();
        let want = want.trim();
        if got != want {
            return Err(format!(
                "bmth: digest mismatch — file {got} != sidecar {want} ({})",
                sidecar.display()
            ));
        }
        Self::from_bytes(&bytes)
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    pub fn support(&self) -> usize {
        self.support
    }

    /// Is `id` one of the support slots' predicted vocab ids? (Diagnostic:
    /// an argmax outside the support can NEVER be drafted by the head.)
    pub fn contains(&self, id: u32) -> bool {
        self.ids.contains(&id)
    }

    /// Score one raw residual row: per-slot `bias[v] + Σ_c w[v][c]·x[c]`,
    /// ascending-c sequential fold (the trainer's dot order — consumer and
    /// fit eval stay bit-comparable), first-index argmax over the support
    /// slots. Returns the predicted vocab id. With the head placed on the
    /// CUDA device ([`Self::place_on_cuda_device`]) the fold runs on the
    /// GPU and the argmax is the CPU-exact first-index kernel — the fold
    /// ORDER differs (block reduce vs sequential), which is safe because a
    /// draft only PROPOSES: acceptance is adjudicated model-side on exact
    /// verify rows, so numerics change efficiency, never correctness
    /// (LOCAL classification per the Issue-940 determinism law).
    pub fn score(&mut self, x: &[f32]) -> Result<u32, String> {
        if x.len() != self.d {
            return Err(format!("bmth: row width {} != d {}", x.len(), self.d));
        }
        #[cfg(all(feature = "spec_mtp_cuda", not(target_os = "macos")))]
        if let Some(dev) = &mut self.device {
            let slot = dev.score(x)?;
            return Ok(self.ids[slot]);
        }
        for v in 0..self.support {
            let wr = &self.w[v * self.d..(v + 1) * self.d];
            let mut s = self.bias[v];
            for c in 0..self.d {
                s += wr[c] * x[c];
            }
            self.scores[v] = s;
        }
        Ok(self.ids[crate::argmax_first(&self.scores) as usize])
    }

    /// Place the scoring on a CUDA device (the 654 promotion gate's
    /// GPU-placement precondition: "the head as a GPU matvec or at least
    /// overlapped with verify"). Idempotent. Refuses loud — never a
    /// silent CPU fallback — when the feature is off, the target is
    /// macOS, or no CUDA device answers.
    #[cfg(all(feature = "spec_mtp_cuda", not(target_os = "macos")))]
    pub fn place_on_cuda_device(&mut self, device_idx: usize) -> Result<(), String> {
        if self.device.is_some() {
            return Ok(());
        }
        self.device = Some(riir_infer_gpu::BmthHeadCuda::new_device(
            device_idx,
            &self.w,
            &self.bias,
            self.d,
            self.support,
        )?);
        Ok(())
    }

    /// Same refusal, non-CUDA postures (feature off or macOS build): the
    /// harness calls this unconditionally on `--head-device cuda`, and the
    /// refusal is the contract — the gate counts no CPU-fallback read.
    #[cfg(not(all(feature = "spec_mtp_cuda", not(target_os = "macos"))))]
    pub fn place_on_cuda_device(&mut self, _device_idx: usize) -> Result<(), String> {
        let _ = (&self.w, &self.bias, self.d, self.support);
        Err(
            "bmth cuda: built without spec_mtp_cuda (or on macOS) — rebuild with \
             --features spec_mtp_cuda on a CUDA box; refusing, never a silent CPU fallback"
                .to_string(),
        )
    }

    /// Is the scorer placed on the CUDA device? (The harness prints this
    /// beside every row — the posture disclosure.)
    #[cfg(all(feature = "spec_mtp_cuda", not(target_os = "macos")))]
    pub fn on_cuda_device(&self) -> bool {
        self.device.is_some()
    }

    /// Non-CUDA postures: always false.
    #[cfg(not(all(feature = "spec_mtp_cuda", not(target_os = "macos"))))]
    pub fn on_cuda_device(&self) -> bool {
        let _ = (&self.w, &self.bias);
        false
    }
}

/// The trained-head drafter (T5.1d). Seated against a stage width via
/// [`MtpHeadDrafter::seat`]; proposes through the runner's roll seam:
///
/// ```text
/// d_0      = pending                       (the echo — always accepted)
/// d_j      = argmax(head(row_{j-1}))       (the trained prediction)
/// row_j    = roll(d_j, cursor + j)         (the stage's own layer window)
/// ```
///
/// K rolls produce K drafts AND the complete verify payload — the drafter's
/// rolls ARE the runner's rows, no second forward.
pub struct MtpHeadDrafter {
    head: BmthHead,
    row: Vec<f32>,
    error: Option<String>,
}

impl MtpHeadDrafter {
    /// Seat the head against a stage's hidden width. A width mismatch
    /// refuses at the seat — never mid-run.
    pub fn seat(head: BmthHead, hidden_dim: usize) -> Result<Self, String> {
        if head.d() != hidden_dim {
            return Err(format!(
                "bmth head d {} != stage hidden_dim {hidden_dim} — wrong head for this stage",
                head.d()
            ));
        }
        Ok(Self {
            row: vec![0.0; hidden_dim],
            head,
            error: None,
        })
    }

    pub fn head(&self) -> &BmthHead {
        &self.head
    }

    /// The first error the drafter hit (a roll row of the wrong width —
    /// unreachable through [`MtpHeadDrafter::seat`], checked anyway), if
    /// any. The drafter degrades to the lossless echo when it cannot
    /// score; the harness surfaces this instead of swallowing it.
    pub fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    fn score_row(&mut self) -> u32 {
        match self.head.score(&self.row) {
            Ok(id) => id,
            Err(e) => {
                if self.error.is_none() {
                    self.error = Some(e);
                }
                // Degradation is lossless: the echo drafts still verify.
                0
            }
        }
    }
}

impl DraftSource for MtpHeadDrafter {
    /// Residual-less fallback: the echo (repeat-pending). Lossless, no
    /// gain — the runner seats this drafter through `propose_with_roll`,
    /// never through here.
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32> {
        let last = ctx.committed.last().copied().unwrap_or(0);
        vec![last; ctx.k]
    }

    fn propose_with_roll(
        &mut self,
        ctx: &DraftCtx<'_>,
        roll: &mut dyn FnMut(u32, usize, &mut [f32]),
    ) -> Vec<u32> {
        let base = ctx.committed.len().saturating_sub(1);
        let pending = ctx.committed.last().copied().unwrap_or(0);
        let mut drafts = Vec::with_capacity(ctx.k);
        drafts.push(pending);
        // Roll 0: the pending token at the cursor — row_0 feeds d_1's score.
        roll(pending, base, &mut self.row);
        for j in 1..ctx.k {
            let id = self.score_row();
            drafts.push(id);
            // Roll j: d_j at cursor + j — completes the verify rows.
            roll(id, base + j, &mut self.row);
        }
        drafts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build BMTH v1 bytes (the riir-train writer's exact layout).
    fn build_bytes(d: usize, vocab: u32, w: &[f32], bias: &[f32], ids: &[u32]) -> Vec<u8> {
        let support = ids.len();
        let mut b = Vec::new();
        b.extend_from_slice(&BMTH_MAGIC);
        b.extend_from_slice(&BMTH_VERSION.to_le_bytes());
        b.extend_from_slice(&(d as u32).to_le_bytes());
        b.extend_from_slice(&vocab.to_le_bytes());
        b.extend_from_slice(&(support as u32).to_le_bytes());
        for v in w {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in bias {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in ids {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn sample_head() -> (BmthHead, Vec<f32>, Vec<f32>, Vec<u32>) {
        // d=8, vocab=64, support=4. Row v of W is the unit vector e_v
        // scaled: slot v scores highest when x[v] is largest.
        let d = 8usize;
        let vocab = 64u32;
        let mut w = vec![0f32; 4 * d];
        for v in 0..4 {
            w[v * d + v] = 1.0;
        }
        let bias = vec![0.5f32, -0.25, 0.125, 0.0];
        let ids = vec![10u32, 20, 30, 40];
        let head = BmthHead::from_bytes(&build_bytes(d, vocab, &w, &bias, &ids)).unwrap();
        (head, w, bias, ids)
    }

    #[test]
    fn loads_and_scores_the_written_layout() {
        let (mut head, w, bias, ids) = sample_head();
        assert_eq!(head.d(), 8);
        assert_eq!(head.vocab(), 64);
        assert_eq!(head.support(), 4);
        // x = e_2 · 3.0: slot 2 must win (w[2]·x = 3.0 + bias 0.125).
        let mut x = [0f32; 8];
        x[2] = 3.0;
        assert_eq!(head.score(&x).unwrap(), ids[2]);
        // A near-tie that bias must decide: x = e_0 · 0.5 vs slot 3 with
        // bias 0.0 — slot 0 = 0.5 + 0.5 = 1.0, slot 3 = 0.0 → slot 0.
        let mut y = [0f32; 8];
        y[0] = 0.5;
        assert_eq!(head.score(&y).unwrap(), ids[0]);
        // And the recorded weights really round-trip.
        assert_eq!(head.w[..3], w[..3]);
        assert_eq!(head.bias, bias);
    }

    #[test]
    fn argmax_tie_goes_to_the_first_slot() {
        // Two identical rows → equal scores → the SMALLER slot index wins
        // (first-index rule, same as the trainer's argmax_first).
        let d = 4usize;
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let bias = vec![0.0, 0.0];
        let ids = vec![7u32, 9];
        let mut head = BmthHead::from_bytes(&build_bytes(d, 16, &w, &bias, &ids)).unwrap();
        assert_eq!(head.score(&[1.0, 0.0, 0.0, 0.0]).unwrap(), 7);
    }

    #[test]
    fn loader_refuses_every_corruption_class() {
        let (head, w, bias, ids) = sample_head();
        let good = build_bytes(8, 64, &w, &bias, &ids);
        assert!(BmthHead::from_bytes(&good).is_ok());
        drop(head);

        // Bad magic.
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(BmthHead::from_bytes(&bad).is_err());
        // Bad version.
        let mut bad = good.clone();
        bad[4] = 9;
        assert!(BmthHead::from_bytes(&bad).is_err());
        // Truncated body.
        assert!(BmthHead::from_bytes(&good[..good.len() - 4]).is_err());
        // Trailing junk refuses (strict length).
        let mut bad = good.clone();
        bad.extend_from_slice(&[0u8; 4]);
        assert!(BmthHead::from_bytes(&bad).is_err());
        // Degenerate shapes.
        assert!(BmthHead::from_bytes(&build_bytes(0, 64, &[], &bias, &ids)).is_err());
        assert!(BmthHead::from_bytes(&build_bytes(8, 0, &w, &bias, &ids)).is_err());
        assert!(BmthHead::from_bytes(&build_bytes(8, 64, &w, &bias, &[])).is_err());
        // A support id outside the vocab refuses.
        assert!(BmthHead::from_bytes(&build_bytes(8, 64, &w, &bias, &[64u32, 1, 2, 3])).is_err());
        // Width mismatch at score time refuses.
        let (mut head, _, _, _) = sample_head();
        assert!(head.score(&[0.0; 7]).is_err());
    }

    #[test]
    fn load_verified_checks_the_sidecar() {
        let (_, w, bias, ids) = sample_head();
        let bytes = build_bytes(8, 64, &w, &bias, &ids);
        let dir = std::env::temp_dir().join(format!("bmth_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("head_v1.bin");
        std::fs::write(&path, &bytes).unwrap();
        let sidecar = Path::with_extension(&path, "blake3");
        std::fs::write(&sidecar, format!("{}\n", blake3::hash(&bytes))).unwrap();
        assert!(BmthHead::load_verified(&path).is_ok());
        // A corrupt sidecar refuses.
        std::fs::write(&sidecar, format!("{}\n", "0".repeat(64))).unwrap();
        assert!(BmthHead::load_verified(&path).is_err());
        // A missing sidecar refuses.
        std::fs::remove_file(&sidecar).unwrap();
        assert!(BmthHead::load_verified(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn drafter_echoes_then_scores_through_the_roll_seam() {
        // Scripted roll table: (pos) → row. Row j drives d_{j+1}'s score.
        let (_, w, bias, ids) = sample_head();
        let head = BmthHead::from_bytes(&build_bytes(8, 64, &w, &bias, &ids)).unwrap();
        let mut drafter = MtpHeadDrafter::seat(head, 8).unwrap();
        let committed = [5u32, 6, 7];
        let k = 3;
        // Roll (pending@base) → x favors slot 1 → d_1 = ids[1] = 20.
        // Roll (20@base+1) → x favors slot 3 → d_2 = ids[3] = 40.
        // The third roll (40@base+2) only completes the verify rows.
        let mut calls: Vec<(u32, usize)> = Vec::new();
        {
            let mut rows: Vec<Vec<f32>> = Vec::new();
            let mut r0 = [0f32; 8];
            r0[1] = 2.0;
            rows.push(r0.to_vec());
            let mut r1 = [0f32; 8];
            r1[3] = 1.5;
            rows.push(r1.to_vec());
            rows.push(vec![0f32; 8]);
            let mut ri = 0usize;
            let mut roll = |_tok: u32, pos: usize, out: &mut [f32]| {
                calls.push((_tok, pos));
                out.copy_from_slice(&rows[ri]);
                ri += 1;
            };
            let drafts = drafter.propose_with_roll(
                &DraftCtx {
                    committed: &committed,
                    k,
                },
                &mut roll,
            );
            assert_eq!(drafts, vec![7, 20, 40], "echo, then the two scores");
        }
        assert_eq!(
            calls,
            vec![(7, 2), (20, 3), (40, 4)],
            "k rolls at cursor + j: the verify rows are complete"
        );
        assert!(drafter.take_error().is_none());
    }

    #[test]
    fn seat_refuses_a_foreign_width() {
        let (_, w, bias, ids) = sample_head();
        let head = BmthHead::from_bytes(&build_bytes(8, 64, &w, &bias, &ids)).unwrap();
        assert!(MtpHeadDrafter::seat(head, 16).is_err());
    }

    #[test]
    fn wrong_width_row_degrades_to_the_lossless_echo_and_records() {
        // Defense-in-depth: a row whose width disagrees with d (reachable
        // only by bypassing `seat`) must record the error and degrade to a
        // lossless draft — never panic, never draft from a misread row.
        let (_, w, bias, ids) = sample_head();
        let head = BmthHead::from_bytes(&build_bytes(8, 64, &w, &bias, &ids)).unwrap();
        let mut drafter = MtpHeadDrafter {
            head,
            row: vec![0.0; 4], // a foreign-width roll landed here
            error: None,
        };
        let mut roll = |_tok: u32, _pos: usize, _out: &mut [f32]| {};
        let drafts = drafter.propose_with_roll(
            &DraftCtx {
                committed: &[1u32, 2],
                k: 2,
            },
            &mut roll,
        );
        assert_eq!(
            drafts,
            vec![2, 0],
            "d_0 echo holds; the broken row scores id 0"
        );
        assert!(
            drafter.take_error().is_some(),
            "the width error is recorded"
        );
    }

    #[test]
    fn real_artifact_loads_and_scores_in_shape() {
        // The 60.7 MB serving artifact when present (amo keeps a copy at
        // the riir-train data pin) — header, digest, and one score at the
        // seated width. Skips loud without it.
        let path =
            std::path::Path::new("../../riir-train/data/bonsai_mtp/head/bonsai_mtp_head_v1.bin");
        if !path.exists() {
            eprintln!("skip: no bonsai_mtp_head_v1.bin on this box");
            return;
        }
        let mut head = BmthHead::load_verified(path).expect("artifact + sidecar");
        assert_eq!(head.d(), 5120);
        assert_eq!(head.vocab(), 248320);
        assert_eq!(head.support(), 2963);
        let x = vec![0f32; 5120];
        let id = head.score(&x).unwrap();
        assert!(id < 248320);
    }
}
