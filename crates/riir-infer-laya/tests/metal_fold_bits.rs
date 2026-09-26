//! The fold rungs' bit-identity gate (reflex issue 020 T11, the last open
//! lever): the residual fold (`LAYA_METAL_FOLD_RES`) and the GLU fold
//! (`LAYA_METAL_FOLD_GLU`) must answer BIT-IDENTICALLY to the unfused
//! streams they replace — at the raw `f32` bits, on the REAL english
//! checkpoint, at loop AND packed AND mixed-plan shapes.
//!
//! Why bits and not the G5 drift budget: the fold is supposed to be
//! bit-identical BY CONSTRUCTION (the same k-ascending slice chains, then
//! one add — IEEE addition commutes; the GLU epilogue is the glu kernel's
//! own expression order). A drift gate would accept a fold that silently
//! reordered the arithmetic; this gate refuses it. The knob-off and
//! mixed-plan arms run the unfused stream, so every row below must be an
//! EXACT equality — a single differing bit is a red, never a tolerance.
//!
//! Reach is asserted, not assumed (the `splitk_dispatches` law): the fold
//! backend must dispatch fold epilogues at the all-split shapes and ZERO
//! at the unsplit shapes; the off backend must dispatch zero everywhere.
//!
//! Posture: every backend here is pinned `with_mps(false)` — the fold's
//! subject is the split-K reduce under [`SplitRule::DEFAULT`]'s reach (the
//! MPS posture's `SplitRule::WITH_MPS` splits only m ≤ 32, so the reach
//! arithmetic below is DEFAULT's by construction, reflex issue 020 T13b).
//!
//! Run: `cargo test --release -p riir-infer-laya --features laya-riir-metal
//! --test metal_fold_bits` (skips LOUD without the checkpoint weights —
//! never a silent pass; an absent subject must not read as green).
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::config::{Checkpoint, load_checkpoint_configs};
use riir_infer_laya::laya::riir::backend::Backend;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::metal::Metal;
use riir_infer_laya::laya::riir::weights as ckpt_weights;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

/// Shapes that cover every fold posture on the shipped `SplitRule`
/// (`max_row_tiles: 3` → all four encoder GEMMs split while m ≤ 96; the
/// d×d / MLP-down shapes stay split to m 128 / 256):
/// - 24 / 54: everything splits (the small-m band the rung targets).
/// - 80: the sweep's qkv/MLP-up margin band.
/// - 140 / 188 / 317: attn-out still splits at 140 (m ≤ 128), MLP-down
///   at 188 (m ≤ 256) — the PARTIALLY-split shapes; 317 nothing splits.
/// - 512: nothing splits (the fold arm must equal the unfused stream by
///   taking the staged fallback, never by changing kernels).
const LOOP_SEQS: &[usize] = &[24, 54, 80, 140, 188, 317, 512];

/// Packed pairs: both-split ([30, 61] — 91 rows), one-splits-one-doesn't
/// ([24, 200] — the mixed plan the per-row split rule produces), and an
/// all-unsplit pair ([200, 220] — the fallback under a packed plan).
const PACKED_CASES: &[&[usize]] = &[&[30, 61], &[24, 200], &[200, 220]];

const VOCAB: usize = 50368;

fn ids_for(len: usize, seed: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (((i + seed) * 7919) % VOCAB) as u32)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn load_encoder() -> Encoder {
    let root = weights_root();
    if std::env::var_os("LAYA_WEIGHTS_DIR").is_none()
        && !root.join("english").join("model.safetensors").exists()
    {
        eprintln!(
            "SKIP LOUD: no english checkpoint under {:?} — the fold bit gate \
             needs the real weights, never a green zero",
            root
        );
        std::process::exit(0);
    }
    let dir = ensure_checkpoint(&root, Checkpoint::English).expect("checkpoint present");
    let name = Checkpoint::English.subfolder();
    let (_agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name).expect("configs");
    let mut raw = ckpt_weights::load(&dir.join("model.safetensors"), name).expect("weights");
    Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder")
}

#[test]
fn fold_arms_are_bit_identical_to_the_unfused_stream() {
    let enc = load_encoder();
    let off = Metal::new().expect("metal").with_mps(false).with_folds(false, false);
    let res_on = Metal::new().expect("metal").with_mps(false).with_folds(true, false);
    let glu_on = Metal::new().expect("metal").with_mps(false).with_folds(false, true);
    let both_on = Metal::new().expect("metal").with_mps(false).with_folds(true, true);
    for b in [&off, &res_on, &glu_on, &both_on] {
        enc.warm(b);
    }

    for &seq in LOOP_SEQS {
        let ids = ids_for(seq, 3);
        // Warm both arms at this shape first (kernel pipelines + scratch
        // pools are shape-dependent; the comparison is of steady state).
        for b in [&off, &res_on, &glu_on, &both_on] {
            b.begin_pass();
            enc.forward(b, &ids).expect("warm forward");
        }
        // Reach baseline — the counters are cumulative, so every shape
        // judges its own DELTA against this snapshot.
        let before = (
            res_on.fold_dispatches(),
            glu_on.fold_dispatches(),
            off.fold_dispatches(),
        );
        let reference = {
            off.begin_pass();
            let h = enc.forward(&off, &ids).expect("off forward");
            let mut out = vec![0f32; h.len()];
            off.download_into(&h, &mut out);
            bits(&out)
        };
        for (label, b) in [("res", &res_on), ("glu", &glu_on), ("both", &both_on)] {
            b.begin_pass();
            let h = enc.forward(b, &ids).expect("fold forward");
            let mut out = vec![0f32; h.len()];
            b.download_into(&h, &mut out);
            assert_eq!(
                bits(&out),
                reference,
                "seq {seq}: fold {label} diverges from the unfused stream at the \
                 raw-bit level — the fold reordered arithmetic, which its \
                 bit-identity construction forbids"
            );
        }
        // Reach, both directions (per-shape DELTA over the snapshot):
        // the fold backends must dispatch fold epilogues at the all-split
        // shapes and none once nothing splits; the off backend must
        // dispatch none anywhere.
        let (res_d, glu_d, off_d) = (
            res_on.fold_dispatches() - before.0,
            glu_on.fold_dispatches() - before.1,
            off.fold_dispatches() - before.2,
        );
        if seq <= 96 {
            assert!(
                res_d > 0 && glu_d > 0,
                "seq {seq}: all-split shape dispatched no fold epilogue — the \
                 fold arm would be passing on the plain kernel"
            );
        }
        if seq >= 317 {
            assert_eq!(
                res_d, 0,
                "seq {seq}: unsplit shape dispatched a fold epilogue — the \
                 all-split predicate is wider than the split rule"
            );
            assert_eq!(
                glu_d, 0,
                "seq {seq}: unsplit shape dispatched a GLU fold epilogue"
            );
        }
        assert_eq!(off_d, 0, "the fold-off backend dispatched a fold epilogue");
    }
}

#[test]
fn fold_arms_are_bit_identical_under_packed_plans() {
    let enc = load_encoder();
    let off = Metal::new().expect("metal").with_mps(false).with_folds(false, false);
    let both_on = Metal::new().expect("metal").with_mps(false).with_folds(true, true);
    enc.warm(&off);
    enc.warm(&both_on);

    for seqs in PACKED_CASES {
        let total: usize = seqs.iter().sum();
        let ids = ids_for(total, 11);
        for b in [&off, &both_on] {
            b.begin_pass();
            enc.forward_packed(b, &ids, seqs).expect("warm packed");
        }
        let reference = {
            off.begin_pass();
            let h = enc.forward_packed(&off, &ids, seqs).expect("off packed");
            let mut out = vec![0f32; h.len()];
            off.download_into(&h, &mut out);
            bits(&out)
        };
        both_on.begin_pass();
        let h = enc
            .forward_packed(&both_on, &ids, seqs)
            .expect("fold packed");
        let mut out = vec![0f32; h.len()];
        both_on.download_into(&h, &mut out);
        assert_eq!(
            bits(&out),
            reference,
            "packed {seqs:?}: the fold arm diverges from the unfused stream at \
             the raw-bit level (covers the mixed-plan staged fallback and the \
             per-row split plan)"
        );
    }
}
