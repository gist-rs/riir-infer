//! `twt_laya_audition` — the Issue 022 Phase-3 driver (feature
//! `twt_profile`): the surrogate-pool audition + the T3.3 branch
//! corrections over a laya checkpoint, on the CPU backend (the lane's
//! f32 deployment arithmetic — the born-quantized rule belongs to the
//! Bonsai ternary lane, not this one).
//!
//! Per pre-registered ε-grid partition, per block, the candidate pool is
//! {each member layer passthrough} ∪ {mean merge} ∪ {RDSC merge} — the
//! merges ONLY for type-homogeneous blocks: the Phase-2 DP carries NO
//! type constraint, and averaging Q/K trained against different RoPE
//! thetas — or a full-attention with a sliding-window layer — is
//! incoherent (the T2.1 tier-(i) rule). A mixed block's pool is
//! passthrough-only, recorded as such. Norm params are ISOLATED per the
//! pinned menu (v1: merged mean, recorded in the artifact); a member
//! without `attn_norm` (layer 0) propagates `None` — never an invented
//! norm.
//!
//! Protocol (pinned BEFORE the first measurement — this file's git
//! history is the pre-registration):
//! - rows split by stride-2 interleave in prompt-row order: EVEN rows →
//!   fit, ODD rows → held-out (both halves see every prompt);
//! - T3.2 selection metric = mean-sq mapping error on FIT rows only;
//! - T3.3 corrections fit on FIT rows only; the held-out half is touched
//!   exactly once, by the decomposition readout;
//! - the decomposition is guarded at +5% (a correction that worsens
//!   held-out error beyond that is a fit pathology and the driver
//!   refuses to record it as a measurement); the REDUCTION FRACTION
//!   itself is the recorded No-GD boundary datum — reported, never gated
//!   here (the zero-training-vs-riir-train-423 track split belongs to
//!   Phase 5's budgets);
//! - the instrument proves itself EVERY run: the parity arm replays one
//!   real layer through this driver's apply path and requires
//!   BIT-IDENTITY with the parent forward's captured state — a
//!   forward-body drift indicts the instrument, not the checkpoint.
//!
//! ```text
//! cargo run --release -p riir-infer-core --features twt_profile \
//!   --example twt_laya_audition -- --checkpoint english \
//!   --calib /path/to/prompts.txt [--eps all|0.05|...] [--max-rows 1536] \
//!   [--no-parity] [--out .raw/twt/english_audition.json]
//! ```

use std::collections::HashMap;

use riir_infer_core::twt::{
    blake3_of, mean_sq_err, merge_mean, merge_rdsc, minmax_partition, selection_pin,
    CandRow, CorrectionFit, SelectionPin, SMatrixBuilder, PRE_REGISTERED_EPS_GRID,
};
use riir_infer_laya::laya::config::{Checkpoint, EncoderConfig};
use riir_infer_laya::laya::riir::backend::{AttnScratch, Backend, Cpu};
use riir_infer_laya::laya::riir::encoder::{CaptureStage, Encoder, LayerWeights};
use riir_infer_laya::laya::riir::ops;
use riir_infer_laya::laya::riir::weights as lane_weights;
use riir_infer_laya::laya::tokenize::Tok;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

/// A merged operator's owned weights (the passthrough candidates borrow
/// the encoder's layers instead — no 1.3 GB clone set).
struct MergedWeights {
    attn_norm: Option<Vec<f32>>,
    wqkv: Vec<f32>,
    wo: Vec<f32>,
    wi: Vec<f32>,
    mlp_wo: Vec<f32>,
    mlp_norm: Vec<f32>,
}

/// One candidate's weight source.
enum CandW {
    /// Passthrough of member layer `li` (absolute index).
    Member(usize),
    /// A merged operator (only built for homogeneous blocks).
    Merged(Box<MergedWeights>),
}

/// The apply-path scratch (the forward's `Scratch` roles for one layer;
/// grow-only across applies — the layer loop must not allocate).
struct ApplyScratch {
    x: Vec<f32>,
    qkv: Vec<f32>,
    attn: AttnScratch,
    merged: Vec<f32>,
    xn: Vec<f32>,
    act: Vec<f32>,
    sq: Vec<f32>,
}

impl ApplyScratch {
    fn new() -> Self {
        Self {
            x: Vec::new(),
            qkv: Vec::new(),
            attn: AttnScratch::default(),
            merged: Vec::new(),
            xn: Vec::new(),
            act: Vec::new(),
            sq: Vec::new(),
        }
    }
}

/// One layer's application to the packed residual stream `h`
/// (`[total × d]`) — the forward's per-layer body, verbatim op order
/// (`Encoder::forward_packed_impl`): norm (or identity copy) → Wqkv →
/// per-sequence fused attention (rope + scale + window + mask) → Wo
/// residual accumulate → mlp norm → fused Wi GLU → mlp Wo residual
/// accumulate. The parity arm asserts bit-identity against the real
/// forward every run, so drift here is caught, not trusted.
#[allow(clippy::too_many_arguments)]
fn apply_layer(
    b: &Cpu,
    h: &mut [f32],
    layer_norm: Option<&[f32]>,
    wqkv: &[f32],
    wo: &[f32],
    wi: &[f32],
    mlp_wo: &[f32],
    mlp_norm: &[f32],
    sliding: bool,
    seqs: &[usize],
    total: usize,
    cfg: &EncoderConfig,
    rope: &(Vec<f32>, Vec<f32>),
    masks: &[Option<Vec<f32>>],
    sc: &mut ApplyScratch,
) {
    let d = cfg.hidden;
    let hd = cfg.head_dim();
    let heads = cfg.heads;
    let i_sz = cfg.intermediate;
    let eps = cfg.eps;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let window = cfg.sliding_window();

    // Pre-size the out-buffers the ops assert on (the forward's
    // `Scratch::reset` role): the norms write into `x`/`xn`, the fused
    // attention into `merged`.
    sc.x.resize(total * d, 0.0);
    sc.xn.resize(total * d, 0.0);
    sc.merged.resize(total * d, 0.0);

    // x = attn_norm(h) — or h itself (the layer-0 identity path, the
    // same copy the forward makes).
    match layer_norm {
        Some(w) => b.layer_norm_nobias_into(h, w, eps, d, &mut sc.sq, &mut sc.x),
        None => {
            sc.x.clear();
            sc.x.extend_from_slice(h);
        }
    }

    sc.qkv.clear();
    sc.qkv.resize(total * 3 * d, 0.0);
    b.matmul_w(&sc.x, total, d, wqkv, 3 * d, &mut sc.qkv);
    let mut off = 0usize;
    for (si, &seq) in seqs.iter().enumerate() {
        let mask = if sliding { masks.get(si).and_then(Option::as_deref) } else { None };
        b.attention_forward(
            &sc.qkv,
            off * 3 * d,
            &rope.0,
            &rope.1,
            off,
            scale,
            seq,
            heads,
            hd,
            if sliding { window } else { usize::MAX },
            mask,
            &mut sc.attn,
            &mut sc.merged,
            off * d,
        );
        off += seq;
    }
    b.matmul_w_accum(&sc.merged, total, d, wo, d, h);

    b.layer_norm_nobias_into(h, mlp_norm, eps, d, &mut sc.sq, &mut sc.xn);
    sc.act.clear();
    sc.act.resize(total * i_sz, 0.0);
    b.matmul_w_glu(&sc.xn, total, d, wi, i_sz, &mut sc.act);
    b.matmul_w_accum(&sc.act, total, i_sz, mlp_wo, d, h);
}

/// Window masks per sequence — the forward's builder, verbatim.
fn build_masks(b: &Cpu, seqs: &[usize], cfg: &EncoderConfig) -> Vec<Option<Vec<f32>>> {
    let window = cfg.sliding_window();
    if !b.needs_window_mask(cfg.head_dim()) {
        return Vec::new();
    }
    seqs.iter()
        .map(|&seq| {
            if seq > 1 && window < seq - 1 {
                let mut m = vec![f32::MIN; seq * seq];
                for qi in 0..seq {
                    let lo = qi.saturating_sub(window);
                    let hi = (qi + window).min(seq - 1);
                    for kv in lo..=hi {
                        m[qi * seq + kv] = 0.0;
                    }
                }
                Some(m)
            } else {
                None
            }
        })
        .collect()
}

fn main() {
    let mut checkpoint = Checkpoint::English;
    let mut calib = String::new();
    let mut eps_arg = "all".to_string();
    let mut max_rows = 1536usize;
    let mut parity = true;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--checkpoint" => {
                let v = args.next().expect("--checkpoint needs a value");
                checkpoint = Checkpoint::ALL
                    .into_iter()
                    .find(|c| c.subfolder() == v)
                    .unwrap_or_else(|| panic!("unknown checkpoint {v:?} (english|multilingual|typed)"));
            }
            "--calib" => calib = args.next().expect("--calib needs a path"),
            "--eps" => eps_arg = args.next().expect("--eps needs all|<float>"),
            "--max-rows" => max_rows = args.next().expect("--max-rows needs N").parse().unwrap(),
            "--no-parity" => parity = false,
            "--out" => out = Some(args.next().expect("--out needs a path")),
            other => panic!("unknown arg {other}"),
        }
    }
    if calib.is_empty() {
        eprintln!(
            "usage: twt_laya_audition --checkpoint <english|multilingual|typed> --calib <prompts.txt> [--eps all|F] [--max-rows N] [--no-parity] [--out P]"
        );
        std::process::exit(2);
    }
    let corpus_bytes = std::fs::read(&calib).unwrap_or_else(|e| panic!("read {calib}: {e}"));
    let prompts: Vec<&str> = std::str::from_utf8(&corpus_bytes)
        .expect("calibration corpus must be UTF-8")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if prompts.is_empty() {
        panic!("calibration corpus {calib} has no prompt lines");
    }

    // Checkpoint: locate → verify → parse → load (the Phase-1 posture).
    let ckpt_name = checkpoint.subfolder();
    let dir = ensure_checkpoint(&weights_root(), checkpoint)
        .unwrap_or_else(|e| panic!("checkpoint fetch: {e}"));
    let (_agent_cfg, cfg) = riir_infer_laya::laya::config::load_checkpoint_configs(&dir, ckpt_name)
        .unwrap_or_else(|e| panic!("configs: {e}"));
    let mut map: HashMap<String, lane_weights::Weights> =
        lane_weights::load(&dir.join("model.safetensors"), ckpt_name)
            .unwrap_or_else(|e| panic!("weights: {e}"));
    let encoder =
        Encoder::from_map(&mut map, cfg.clone(), ckpt_name).unwrap_or_else(|e| panic!("encoder: {e}"));
    let tok = Tok::from_dir(&dir, ckpt_name).unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let backend = Cpu;
    encoder.warm(&backend);

    let n_layers = cfg.layers;
    let d = cfg.hidden;

    // ── the parity arm: the instrument proves itself before any number ──
    // Two layers, one per attention type — layer 5 (sliding; the mask
    // + slide-theta path) and layer 6 (full attention; the full-theta
    // no-mask path). Never layer 0 (no attn_norm). Each: capture the
    // states around the REAL forward's layer, apply that layer's own
    // weights through THIS driver's apply path, require bit-identity.
    if parity {
        let ids = tok
            .encode(prompts[0])
            .unwrap_or_else(|e| panic!("parity prompt encode: {e}"));
        assert!(ids.len() >= 8, "parity prompt too short");
        let len = ids.len();
        let mut stages: HashMap<usize, Vec<f32>> = HashMap::new();
        encoder
            .forward_capture(&backend, &ids, &[len], &mut |stage, row, state| {
                if let CaptureStage::AfterLayer(li) = stage {
                    let slot = stages.entry(li).or_insert_with(|| vec![0f32; len * d]);
                    slot[row * d..row * d + d].copy_from_slice(&state[..d]);
                }
            })
            .unwrap_or_else(|e| panic!("parity forward: {e}"));
        for li in [5usize, 6] {
            let h_prev = stages.get(&(li - 1)).expect("parity stage n-1");
            let h_true = stages.get(&li).expect("parity stage n");
            let lw = encoder.layer_weights(li).expect("parity layer present");
            let rope = ops::rope_tables(len, cfg.head_dim(), cfg.theta_for(li));
            let masks = build_masks(&backend, &[len], &cfg);
            let mut h = h_prev.clone();
            let mut sc = ApplyScratch::new();
            apply_layer(
                &backend, &mut h, lw.attn_norm, lw.wqkv, lw.wo, lw.wi, lw.mlp_wo,
                lw.mlp_norm, lw.sliding, &[len], len, &cfg, &rope, &masks, &mut sc,
            );
            let diffs = h
                .iter()
                .zip(h_true)
                .filter(|(a, c)| a.to_bits() != c.to_bits())
                .count();
            assert_eq!(
                diffs, 0,
                "parity FAILED on layer {li}: the apply path diverged from the parent forward \
                 on {diffs} elements — the audition would measure a different operator; \
                 fix the instrument first"
            );
            eprintln!(
                "[twt-audition] parity: layer-{li} apply is BIT-IDENTICAL to the parent forward ({len} rows × {d})"
            );
        }
    }

    // ── capture all stages over the calibration corpus ──
    // store[stage][row × d]: stage 0 = embedding, stage s+1 = after layer
    // s. Rows land prompt-major in capture order (the stride-2 split is
    // index-based over that order).
    let mut prompts_ok: Vec<Vec<u32>> = Vec::new();
    let mut rows_total = 0usize;
    'corpus: for prompt in &prompts {
        let Ok(ids) = tok.encode(prompt) else { continue };
        if ids.len() < 2 {
            continue;
        }
        if rows_total + ids.len() > max_rows {
            break 'corpus;
        }
        rows_total += ids.len();
        prompts_ok.push(ids);
    }
    assert!(rows_total >= 8, "calibration too small: {rows_total} rows");
    let mut store = vec![vec![0f32; rows_total * d]; n_layers + 1];
    let mut row_base = 0usize;
    for ids in &prompts_ok {
        let len = ids.len();
        encoder
            .forward_capture(&backend, ids, &[len], &mut |stage, row, state| {
                let s = match stage {
                    CaptureStage::Embedding => 0,
                    CaptureStage::AfterLayer(li) => li + 1,
                };
                let off = (row_base + row) * d;
                store[s][off..off + d].copy_from_slice(&state[..d]);
            })
            .unwrap_or_else(|e| panic!("capture failed: {e}"));
        row_base += len;
    }
    eprintln!(
        "[twt-audition] captured {n_layers}+1 stages over {rows_total} rows ({} prompts, dim {d})",
        prompts_ok.len()
    );

    // ── the S matrix ONCE (cosine meter, the Phase-1 posture), then the
    //    partitions over the pre-registered grid ──
    let mut nb = SMatrixBuilder::new(n_layers, d);
    row_base = 0usize;
    for ids in &prompts_ok {
        let len = ids.len();
        nb.begin_forward(len).unwrap();
        for row in 0..len {
            for li in 0..n_layers {
                let off = (row_base + row) * d;
                nb.push(li, row, &store[li + 1][off..off + d]).unwrap();
            }
        }
        nb.end_forward().unwrap();
        row_base += len;
    }
    let sm = nb.finalize().unwrap();
    let s = sm.cosine();

    let grid: Vec<f32> = if eps_arg == "all" {
        PRE_REGISTERED_EPS_GRID.to_vec()
    } else {
        vec![eps_arg.parse().expect("--eps must be all or a float")]
    };
    let mut partitions = Vec::with_capacity(grid.len());
    for &eps in &grid {
        let part = minmax_partition(s, eps).unwrap_or_else(|e| panic!("partition ε={eps}: {e}"));
        partitions.push((eps, part));
    }

    // ── protocol setup ──
    let fit_rows = rows_total.div_ceil(2); // even-index rows
    let held_rows = rows_total - fit_rows;
    let fit_idx: Vec<usize> = (0..rows_total).step_by(2).collect();
    let held_idx: Vec<usize> = (1..rows_total).step_by(2).collect();
    assert_eq!(fit_idx.len(), fit_rows);
    eprintln!(
        "[twt-audition] protocol: fit {fit_rows} (even rows) / held-out {held_rows} (odd rows), stride-2 interleave"
    );

    // Per-prompt row ranges for the apply loop.
    let mut prompt_ranges: Vec<(usize, usize)> = Vec::with_capacity(prompts_ok.len());
    let mut acc = 0usize;
    for ids in &prompts_ok {
        prompt_ranges.push((acc, acc + ids.len()));
        acc += ids.len();
    }

    // Shared geometry caches — rope tables and masks depend only on
    // (sliding, len); every candidate reuses them.
    let mut rope_cache: HashMap<(bool, usize), (Vec<f32>, Vec<f32>)> = HashMap::new();
    let mut mask_cache: HashMap<usize, Vec<Option<Vec<f32>>>> = HashMap::new();
    let mut sur = vec![0f32; rows_total * d];
    let mut sc = ApplyScratch::new();

    let mut art = serde_json::json!({
        "checkpoint": ckpt_name,
        "n_layers": n_layers,
        "dim": d,
        "rows_total": rows_total,
        "fit_rows": fit_rows,
        "held_rows": held_rows,
        "split": "stride2_even_fit_odd_heldout",
        "selection_metric": "mean_sq_err on fit rows",
        "norm_menu_pinned": "merged_mean (v1; block-final/audition-best alternatives recorded, not run)",
        "merges_on_mixed_type_blocks": "refused (pool passthrough-only; recorded per block)",
        "corpus_blake3": blake3_of(&corpus_bytes),
        "parity_proven_this_run": parity,
        "audition_arithmetic": "f32 CPU (the laya lane's deployment posture; the born-quantized audition rule belongs to the Bonsai ternary lane)",
        "pre_registration": "selection on fit rows only; corrections fit on fit rows; held-out touched once by the decomposition; the reduction fraction is the recorded No-GD boundary datum (Phase 5 owns the track split); decomposition guarded at +5% (a worse-than-that correction is a fit pathology, refused)",
        "partitions": [],
    });
    let partitions_out = art["partitions"].as_array_mut().unwrap();

    for (eps, part) in &partitions {
        let mut blocks_out = Vec::with_capacity(part.len());
        for block in part {
            let (s, e) = (block.start, block.end);
            let k = block.len();
            let types: Vec<bool> = (s..e).map(|li| cfg.sliding[li]).collect();
            let homogeneous = types.iter().all(|&t| t == types[0]);
            let h_in: &[f32] = &store[s][..rows_total * d];
            let h_e: &[f32] = &store[e][..rows_total * d];

            // Candidate table (canonical order: members ascending, then
            // mean, then rdsc — the pin's hash assumes this order).
            let mut cand_ids: Vec<String> = (s..e).map(|li| format!("member:{li}")).collect();
            let mut cand_w: Vec<CandW> = (s..e).map(CandW::Member).collect();
            if homogeneous {
                let members: Vec<LayerWeights<'_>> =
                    (s..e).map(|li| encoder.layer_weights(li).unwrap()).collect();
                let any_no_norm = members.iter().any(|m| m.attn_norm.is_none());
                // Explicit per-tensor merges — no fn-pointer indirection
                // (the HRTB coercion through a generic merge fn fights the
                // LayerWeights lifetimes for nothing).
                macro_rules! merged {
                    ($field:ident, $op:ident) => {
                        $op(members.iter().map(|m| m.$field)).unwrap()
                    };
                }
                let mw = MergedWeights {
                    attn_norm: if any_no_norm {
                        None // layer-0 quirk propagates, never invented
                    } else {
                        Some(merge_mean(members.iter().map(|m| m.attn_norm.unwrap())).unwrap())
                    },
                    wqkv: merged!(wqkv, merge_mean),
                    wo: merged!(wo, merge_mean),
                    wi: merged!(wi, merge_mean),
                    mlp_wo: merged!(mlp_wo, merge_mean),
                    mlp_norm: merged!(mlp_norm, merge_mean),
                };
                cand_ids.push("mean".into());
                cand_w.push(CandW::Merged(Box::new(mw)));
                let rw = MergedWeights {
                    attn_norm: if any_no_norm {
                        None
                    } else {
                        Some(merge_rdsc(members.iter().map(|m| m.attn_norm.unwrap())).unwrap())
                    },
                    wqkv: merged!(wqkv, merge_rdsc),
                    wo: merged!(wo, merge_rdsc),
                    wi: merged!(wi, merge_rdsc),
                    mlp_wo: merged!(mlp_wo, merge_rdsc),
                    mlp_norm: merged!(mlp_norm, merge_rdsc),
                };
                cand_ids.push("rdsc".into());
                cand_w.push(CandW::Merged(Box::new(rw)));
            }

            // Per candidate: ONE apply over all rows (prompt-major),
            // sliced to fit/held-out for the metric + corrections.
            let mut rows_out = Vec::with_capacity(cand_w.len());
            for (ci, cw) in cand_w.iter().enumerate() {
                for &(r0, r1) in &prompt_ranges {
                    let len = r1 - r0;
                    let sliding = match cw {
                        CandW::Member(li) => cfg.sliding[*li],
                        CandW::Merged(_) => types[0], // homogeneous only
                    };
                    let rope = rope_cache
                        .entry((sliding, len))
                        .or_insert_with(|| {
                            let theta = if sliding {
                                cfg.rope_theta_slide
                            } else {
                                cfg.rope_theta_full
                            };
                            ops::rope_tables(len, cfg.head_dim(), theta)
                        });
                    let masks = mask_cache
                        .entry(len)
                        .or_insert_with(|| build_masks(&backend, &[len], &cfg));
                    let mut h = vec![0f32; len * d];
                    h.copy_from_slice(&h_in[r0 * d..r1 * d]);
                    match cw {
                        CandW::Member(li) => {
                            let lw = encoder.layer_weights(*li).unwrap();
                            apply_layer(
                                &backend, &mut h, lw.attn_norm, lw.wqkv, lw.wo, lw.wi,
                                lw.mlp_wo, lw.mlp_norm, lw.sliding, &[len], len, &cfg,
                                rope, masks, &mut sc,
                            );
                        }
                        CandW::Merged(w) => {
                            apply_layer(
                                &backend, &mut h, w.attn_norm.as_deref(), &w.wqkv, &w.wo,
                                &w.wi, &w.mlp_wo, &w.mlp_norm, sliding, &[len], len, &cfg,
                                rope, masks, &mut sc,
                            );
                        }
                    }
                    sur[r0 * d..r1 * d].copy_from_slice(&h);
                }

                // Fit-half gather (even rows).
                let gather = |idx: &[usize]| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
                    let cap = idx.len() * d;
                    let mut a = Vec::with_capacity(cap);
                    let mut b = Vec::with_capacity(cap);
                    let mut c = Vec::with_capacity(cap);
                    for &r in idx {
                        a.extend_from_slice(&h_in[r * d..(r + 1) * d]);
                        b.extend_from_slice(&sur[r * d..(r + 1) * d]);
                        c.extend_from_slice(&h_e[r * d..(r + 1) * d]);
                    }
                    (a, b, c)
                };
                let (f_in, f_sur, f_e) = gather(&fit_idx);
                let err_fit = mean_sq_err(&f_sur, &f_e, fit_rows, d);
                let corr = CorrectionFit::fit(&f_in, &f_sur, &f_e, fit_rows, d)
                    .unwrap_or_else(|e| panic!("block {s}..{e} candidate {}: fit: {e}", cand_ids[ci]));

                // Held-out decomposition (the protocol's one touch). Each
                // arm gets its OWN least-squares fit on the fit rows:
                // +α = the through-origin fit (β=0 model), +αβ = the joint
                // fit. Re-using the joint α for the α-only arm mis-prices
                // the rung (measured: +12.7% on block 14..18 from the
                // joint slope, its own through-origin fit improves).
                let (o_in, o_sur, o_e) = gather(&held_idx);
                let err_raw = mean_sq_err(&o_sur, &o_e, held_rows, d);
                let corr_a =
                    CorrectionFit::fit_alpha_only(&f_in, &f_sur, &f_e, fit_rows, d)
                        .unwrap_or_else(|e| panic!("block {s}..{e} candidate {}: α-only fit: {e}", cand_ids[ci]));
                let mut corr_a_buf = vec![0f32; held_rows * d];
                let mut corr_ab = vec![0f32; held_rows * d];
                corr_a.apply(&o_in, &o_sur, &mut corr_a_buf);
                corr.apply(&o_in, &o_sur, &mut corr_ab);
                let err_a = mean_sq_err(&corr_a_buf, &o_e, held_rows, d);
                let err_ab = mean_sq_err(&corr_ab, &o_e, held_rows, d);
                let alpha_mean: f64 = corr.alpha.iter().map(|&a| a as f64).sum::<f64>() / d as f64;
                let beta_mean: f64 = corr.beta.iter().map(|&b| b as f64).sum::<f64>() / d as f64;

                rows_out.push(serde_json::json!({
                    "id": cand_ids[ci],
                    "err_fit": err_fit,
                    "err_raw_held": err_raw,
                    "err_alpha_held": err_a,
                    "err_alpha_beta_held": err_ab,
                    "alpha_mean": alpha_mean,
                    "beta_mean": beta_mean,
                    "degenerate_channels": corr.degenerate_channels,
                }));
            }

            // Selection (fit rows only) + the cross-checked pin.
            let cand_rows: Vec<CandRow> = cand_ids
                .iter()
                .zip(rows_out.iter())
                .map(|(id, r)| CandRow {
                    id: id.clone(),
                    err_fit: r["err_fit"].as_f64().unwrap(),
                })
                .collect();
            let pin: SelectionPin =
                selection_pin(&cand_rows).unwrap_or_else(|e| panic!("block {s}..{e}: selection: {e}"));

            // The winner's decomposition, guarded at +5% — RELATIVE to a
            // NON-ZERO raw error. A singleton passthrough's raw error is
            // EXACTLY 0 (the candidate IS the parent's layer), and the
            // fit's f64 rounding leaves ~1e-13 noise — a relative guard
            // would refuse an exact block on its own rounding (measured:
            // block 0..1 at ε=1.20). Guard only blocks with real error;
            // a zero-raw block's reduction is 0 by definition.
            let wr = rows_out.iter().find(|r| r["id"] == pin.winner).unwrap();
            let (raw, ea, eab) = (
                wr["err_raw_held"].as_f64().unwrap(),
                wr["err_alpha_held"].as_f64().unwrap(),
                wr["err_alpha_beta_held"].as_f64().unwrap(),
            );
            if raw > 1e-9 {
                assert!(
                    ea <= raw * 1.05 && eab <= raw * 1.05,
                    "block {s}..{e}: correction WORSENED held-out error beyond the +5% guard \
                     (raw {raw} → α {ea} → αβ {eab}) — fit pathology, refused"
                );
            }
            let reduction = if raw > 0.0 { (1.0 - eab / raw).clamp(0.0, 1.0) } else { 0.0 };

            blocks_out.push(serde_json::json!({
                "start": s, "end": e, "k": k,
                "homogeneous": homogeneous,
                "type": if homogeneous { if types[0] { "sliding" } else { "full" } } else { "mixed" },
                "pool": if homogeneous {
                    "members+mean+rdsc"
                } else {
                    "members (passthrough-only: mixed types — merge incoherent across thetas/mask)"
                },
                "winner": pin.winner,
                "table_blake3": pin.table_blake3,
                "winner_held_reduction_alpha_beta": reduction,
                "winner_alpha_mean": wr["alpha_mean"],
                "winner_beta_mean": wr["beta_mean"],
                "candidates": rows_out,
            }));
        }
        partitions_out.push(serde_json::json!({ "eps": eps, "blocks": blocks_out }));
        eprintln!("[twt-audition] ε={eps}: {} blocks auditioned", part.len());
    }

    println!(
        "twt_laya_audition: {} partitions auditioned over {rows_total} rows ({} prompts, parity={})",
        partitions.len(),
        prompts_ok.len(),
        parity
    );
    for p in partitions_out.iter() {
        println!("ε={}:", p["eps"]);
        for blk in p["blocks"].as_array().unwrap() {
            println!(
                "  [{:>2},{:>2}) {:>5} → {} (held reduction αβ {:.3})",
                blk["start"].as_u64().unwrap(),
                blk["end"].as_u64().unwrap(),
                blk["type"].as_str().unwrap(),
                blk["winner"].as_str().unwrap(),
                blk["winner_held_reduction_alpha_beta"].as_f64().unwrap(),
            );
        }
    }

    if let Some(path) = out {
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(&path, serde_json::to_string_pretty(&art).unwrap())
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
        eprintln!("[twt-audition] artifact written to {path}");
    }
}
