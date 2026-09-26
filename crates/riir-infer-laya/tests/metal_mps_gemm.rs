//! The MPS GEMM arm (reflex issue 020 T13): the bit gate and the paired A/B.
//!
//! `Metal::with_mps(true)` routes every unsplit batch-1 dense GEMM through
//! Apple's `MPSMatrixMultiplication` (the kernel family the torch MPS
//! oracle runs). The pricing probe (`examples/sgemm_mps_probe.rs`) measured
//! it bit-identical to the narrow instance on all 24 encoder cells, so the
//! gate here is RAW BITS, not the G5 drift budget: a drift gate would
//! accept an MPS kernel pick that silently reordered the reduction.
//!
//! Reach is asserted both ways (the `splitk_dispatches` law): the MPS
//! backend must dispatch MPS GEMMs once any encoder projection is unsplit
//! (m ≥ 97 under the shipped `SplitRule`), and none while everything
//! splits; the off backend must dispatch none.
//!
//! ```sh
//! # the gate (skips LOUD without the checkpoint)
//! cargo test --release -p riir-infer-laya --features laya-riir-metal \
//!     --test metal_mps_gemm
//! # the paired A/B (measurement only)
//! cargo test --release -p riir-infer-laya --features laya-riir-metal \
//!     --test metal_mps_gemm -- --ignored --nocapture
//! ```
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::config::{Checkpoint, load_checkpoint_configs};
use riir_infer_laya::laya::riir::backend::Backend;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::metal::Metal;
use riir_infer_laya::laya::riir::weights as ckpt_weights;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

/// Loop shapes: all-split (24/54/80), the first unsplit band (106/140),
/// partially split (188), nothing split (317/512).
const LOOP_SEQS: &[usize] = &[24, 54, 80, 106, 140, 188, 317, 512];

/// Packed plans: both-split, mixed (one segment splits, one does not), all
/// unsplit, and the typed_decisions case shape (5 questions × ~179 tokens).
const PACKED_CASES: &[&[usize]] = &[
    &[30, 61],
    &[24, 200],
    &[200, 220],
    &[179, 179, 179, 179, 179],
];

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
            "SKIP LOUD: no english checkpoint under {root:?} — the MPS bit gate \
             needs the real weights, never a green zero"
        );
        std::process::exit(0);
    }
    let dir = ensure_checkpoint(&root, Checkpoint::English).expect("checkpoint present");
    let name = Checkpoint::English.subfolder();
    let (_agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name).expect("configs");
    let mut raw = ckpt_weights::load(&dir.join("model.safetensors"), name).expect("weights");
    Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder")
}

fn run(enc: &Encoder, b: &Metal, ids: &[u32], seqs: Option<&[usize]>) -> Vec<f32> {
    b.begin_pass();
    let h = match seqs {
        Some(s) => enc.forward_packed(b, ids, s).expect("packed forward"),
        None => enc.forward(b, ids).expect("forward"),
    };
    let mut out = vec![0f32; h.len()];
    b.download_into(&h, &mut out);
    out
}

fn backends() -> (Metal, Metal) {
    let off = Metal::new().expect("metal").with_mps(false);
    let on = Metal::new().expect("metal").with_mps(true);
    assert!(
        on.mps_active(),
        "MPSMatrixMultiplication did not resolve on this macOS host — the arm \
         cannot be gated here, and a skipped arm must not read as green"
    );
    (off, on)
}

#[test]
fn mps_arm_is_bit_identical_on_loop_shapes() {
    let enc = load_encoder();
    let (off, on) = backends();
    enc.warm(&off);
    enc.warm(&on);
    for &seq in LOOP_SEQS {
        let ids = ids_for(seq, 3);
        run(&enc, &off, &ids, None);
        run(&enc, &on, &ids, None);
        let before = (on.mps_dispatches(), off.mps_dispatches());
        let reference = bits(&run(&enc, &off, &ids, None));
        let got = bits(&run(&enc, &on, &ids, None));
        assert_eq!(
            got, reference,
            "seq {seq}: the MPS arm diverges from the narrow instance at the \
             raw-bit level"
        );
        let (on_d, off_d) = (on.mps_dispatches() - before.0, off.mps_dispatches() - before.1);
        if seq <= 96 {
            assert_eq!(on_d, 0, "seq {seq}: all-split shape dispatched an MPS GEMM");
        } else {
            assert!(
                on_d > 0,
                "seq {seq}: unsplit shape dispatched no MPS GEMM — the arm would \
                 be passing on the narrow kernel"
            );
        }
        assert_eq!(off_d, 0, "the MPS-off backend dispatched an MPS GEMM");
    }
}

#[test]
fn mps_arm_is_bit_identical_under_packed_plans() {
    let enc = load_encoder();
    let (off, on) = backends();
    enc.warm(&off);
    enc.warm(&on);
    for seqs in PACKED_CASES {
        let total: usize = seqs.iter().sum();
        let ids = ids_for(total, 11);
        run(&enc, &off, &ids, Some(seqs));
        run(&enc, &on, &ids, Some(seqs));
        let reference = bits(&run(&enc, &off, &ids, Some(seqs)));
        let got = bits(&run(&enc, &on, &ids, Some(seqs)));
        assert_eq!(
            got, reference,
            "packed {seqs:?}: the MPS arm diverges from the narrow instance at \
             the raw-bit level (covers the per-row split plan's unsplit runs)"
        );
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Paired, position-balanced whole-forward A/B (the fold A/B's law): per
/// round both backends run the SAME ids, order alternating; the verdict is
/// the median of per-round on/off ratios. Record box state beside it
/// (reflex `scripts/bench_preflight.sh`).
#[test]
#[ignore = "measurement-only A/B (issue 020 T13 MPS GEMM arm) — run with --ignored --nocapture on a quiet box"]
fn t13_mps_paired_ab() {
    const ROUNDS: usize = 24;
    let enc = load_encoder();
    let (off, on) = backends();
    enc.warm(&off);
    enc.warm(&on);
    println!(
        "t13 MPS A/B: MPS GEMM arm vs narrow · {ROUNDS} paired rounds/shape · judge \
         on the ratios' medians, never the absolute p50s"
    );
    let shapes: Vec<(String, Vec<usize>)> = [54usize, 80, 106, 140, 188, 317, 512]
        .iter()
        .map(|s| (format!("loop {s}"), vec![*s]))
        .chain(PACKED_CASES.iter().map(|s| (format!("packed {s:?}"), s.to_vec())))
        .collect();
    for (label, seqs) in &shapes {
        let total: usize = seqs.iter().sum();
        let ids = ids_for(total, 5);
        let packed = (seqs.len() > 1).then_some(seqs.as_slice());
        let fwd = |b: &Metal| {
            let t = std::time::Instant::now();
            run(&enc, b, &ids, packed);
            t.elapsed().as_secs_f64() * 1e3
        };
        for _ in 0..2 {
            fwd(&off);
            fwd(&on);
        }
        let before = on.mps_dispatches();
        let (mut a, mut b, mut r) = (Vec::new(), Vec::new(), Vec::new());
        for round in 0..ROUNDS {
            let (x, y) = if round % 2 == 0 {
                let x = fwd(&off);
                (x, fwd(&on))
            } else {
                let y = fwd(&on);
                (fwd(&off), y)
            };
            a.push(x);
            b.push(y);
            r.push(y / x);
        }
        let per_fwd = (on.mps_dispatches() - before) / ROUNDS as u64;
        let mut rs = r.clone();
        rs.sort_by(f64::total_cmp);
        println!(
            "{label:>28}: off p50 {:7.3} ms · on p50 {:7.3} ms · paired on/off median {:.3} \
             (IQR {:.3}–{:.3}) · wins {}/{ROUNDS} · mps gemms/fwd {per_fwd}",
            median(a),
            median(b),
            median(r.clone()),
            rs[ROUNDS / 4],
            rs[3 * ROUNDS / 4],
            r.iter().filter(|v| **v < 1.0).count(),
        );
    }
}
