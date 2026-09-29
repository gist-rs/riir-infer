//! Issue 022 Phase 5 — the Bonsai GDN apply-path audition (T5.0's gate:
//! passthrough-only failed quality at real depth cuts, which JUSTIFIES
//! this instrument; the merge question finally gets its test).
//!
//! What it does, in order:
//!
//! 1. Loads the PARENT ternary weights once and the Phase-1 profile
//!    artifact's S matrix, then partitions the main stack with the
//!    TYPE-AWARE min-max DP (`minmax_partition_typed`) — every
//!    multi-layer block is homogeneous, so every block is a REAL merge
//!    candidate. The unconstrained DP's blocks mixed GDN and attention
//!    layers (the interval-4 rhythm puts an attention layer inside almost
//!    every span), which made merges impossible there — the merge room
//!    the T3.2 laya record predicted for this lane lives behind the type
//!    constraint, not despite it.
//! 2. Runs a capture pass: the parent forward over `--rows` contiguous
//!    calibration rows, snapshotting the post-layer residual at every
//!    block boundary (the h the block must reproduce) — plus the first
//!    merge block's FIRST-member slot, the parity witness.
//! 3. PREFLIGHT (after row 0, before the remaining rows buy anything):
//!    the parity block's first member is replayed through
//!    `qwen_deltanet_ternary_layer_body` over the captured boundary h and
//!    MUST reproduce the parent's own post-member hidden bit-exactly —
//!    the apply path IS the forward's code, but this proves the state
//!    handling (fresh candidate slot vs the parent's slot) is equivalent
//!    too. A nonzero diff aborts the run.
//! 4. Auditions per block: the pool is {each member passthrough} ∪
//!    {sign_majority (arm B — the integer code vote, op-independent)}
//!    ∪ {mean:source_quant, rdsc:source_quant (arm C over the two merge
//!    operators)}. GDN gate/decay params (a_log, dt_bias, conv1d) are
//!    NEVER averaged (exp/sigmoid distortion — T3.1): they copy from the
//!    block's minimax-medoid member; RMSNorm gammas merge with the mean.
//!    Winner = argmin of Σ‖f_cand(h_in) − h_e‖² over rows × positions,
//!    evaluated in deployment arithmetic (born-ternary: arm B/C output
//!    runs through the stock ternary ops).
//! 5. Writes the selection table (JSON; the consumer pins its blake3)
//!    that `twt_collapse_emit --selection` turns into a collapsed GGUF.
//!
//! The candidate's GDN/attention state is ITS OWN: a merged layer must
//! carry its recurrence from the sequence start (the zero-training
//! surrogate semantics), so each candidate replay starts from a reset
//! slot and consumes the captured boundary h per position.
//!
//! ```text
//! cargo run --release -p riir-infer-core --features twt_bonsai,twt_collapse \
//!   --example twt_bonsai_audition -- \
//!   --parent ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --profile .raw/twt/bonsai_ultrachat_profile.json \
//!   --corpus .raw/twt_bonsai_calib.txt \
//!   --eps 0.1 --rows 8 --row-len 256 \
//!   --out .raw/twt/bonsai_audition_e01_typed.json
//! ```

use std::time::Instant;

use anyhow::{Context, Result, bail};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, effective_rotary_dim,
};
use riir_infer_core::deltanet::rotation::rotate_inverse_inplace;
use riir_infer_core::deltanet::ternary_forward::{
    forward_qwen_deltanet_ternary_with_hook, qwen_deltanet_ternary_layer_body,
};
use riir_infer_core::deltanet::ternary_weights::{
    DeltaNetTernaryLayerWeights, GateProjWeights, QwenDeltaNetTernaryWeights,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::twt::audition::{merge_mean, merge_rdsc};
use riir_infer_core::twt::partition::{Block, minmax_partition_typed, partition_worst};
use riir_infer_core::twt::smatrix::SMatrix;
use riir_infer_core::twt::ternarize::{Materialized, arm_sign_majority, arm_source_quant};
use riir_infer_core::types::DeltaNetLayerType;

use katgpt_core::TernaryGroupWeights;

/// The merge operators (the pool's f̄ axis; arm B ignores the op — its
/// integer vote reads the members directly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MergeOp {
    Mean,
    Rdsc,
}

impl MergeOp {
    fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Rdsc => "rdsc",
        }
    }
    fn merge(self, refs: &[&[f32]]) -> Result<Vec<f32>> {
        match self {
            Self::Mean => Ok(merge_mean(refs.iter().copied())?),
            Self::Rdsc => Ok(merge_rdsc(refs.iter().copied())?),
        }
    }
}

/// The materialization arms auditioned for merged operators. Arm A
/// (dense f16) is deliberately ABSENT: the collapsed checkpoint must stay
/// loadable by the ternary pipeline, so only ternary-emit-able arms are
/// deployable candidates here (arm A remains T4.2's operator-level
/// reference, not a pool member).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmChoice {
    SignMajority,
    SourceQuant,
}

impl ArmChoice {
    fn name(self) -> &'static str {
        match self {
            Self::SignMajority => "sign_majority",
            Self::SourceQuant => "source_quant",
        }
    }
}

/// A candidate's weights: a parent member by reference (zero-copy) or an
/// owned merged construction.
enum CandWeights<'a> {
    Member(&'a DeltaNetTernaryLayerWeights),
    Owned(Box<DeltaNetTernaryLayerWeights>),
}

impl CandWeights<'_> {
    fn w(&self) -> &DeltaNetTernaryLayerWeights {
        match self {
            Self::Member(w) => w,
            Self::Owned(w) => w,
        }
    }
}

#[derive(serde::Serialize)]
struct CandRec {
    id: String,
    err_sq: f64,
}

#[derive(serde::Serialize)]
struct BlockRec {
    start: usize,
    end: usize,
    kind: &'static str,
    n_members: usize,
    candidates: Vec<CandRec>,
    winner: String,
    /// `mean` | `rdsc` — the merge operator (merged winners only).
    op: Option<String>,
    /// `member` | `sign_majority` | `source_quant` — the emitted arm.
    arm: String,
    /// The dense-field (ssm_alpha/beta) merge operator (merged winners).
    dense_op: String,
    /// The member whose never-averaged params (a_log/dt_bias/conv1d) the
    /// merged construction copies — the block's minimax medoid.
    params_member: usize,
}

#[derive(serde::Serialize)]
struct Selection {
    eps: f32,
    partition_mode: &'static str,
    rows: usize,
    row_len: usize,
    corpus_file: String,
    corpus_blake3_file: String,
    profile_corpus_blake3: String,
    parity: ParityRec,
    blocks: Vec<BlockRec>,
}

#[derive(serde::Serialize)]
struct ParityRec {
    layer: usize,
    block: [usize; 2],
    positions: usize,
    max_abs_diff: f64,
}

/// Minimax medoid member of a block (the same rule the passthrough emit
/// uses: the member whose worst cosine distance to the rest of its block
/// is smallest; ties → lowest index). Deterministic from S alone — the
/// params_member anchor is identical on the audition and emit sides.
fn medoid(s: &SMatrix, b: Block) -> usize {
    if b.len() == 1 {
        return b.start;
    }
    let mut best = b.start;
    let mut best_worst = f32::INFINITY;
    for m in b.start..b.end {
        let w = (b.start..b.end)
            .filter(|&k| k != m)
            .map(|k| s.get(m, k))
            .fold(0.0f32, f32::max);
        if w < best_worst {
            best_worst = w;
            best = m;
        }
    }
    best
}

/// Dequantize one ternary projection field across the block's members and
/// materialize the requested arm.
fn merged_ternary_field(
    members: &[&DeltaNetTernaryLayerWeights],
    get: impl Fn(&DeltaNetTernaryLayerWeights) -> &TernaryGroupWeights,
    op: MergeOp,
    arm: ArmChoice,
) -> Result<TernaryGroupWeights> {
    let w0 = get(members[0]);
    let (rows, cols) = (w0.rows, w0.cols);
    let dense: Vec<Vec<f32>> = members
        .iter()
        .map(|l| QwenDeltaNetTernaryWeights::dequant_proj_to_dense(get(l)))
        .collect();
    for d in &dense {
        assert_eq!(d.len(), rows * cols, "member field shape drift");
    }
    let refs: Vec<&[f32]> = dense.iter().map(|v| v.as_slice()).collect();
    let out = match arm {
        ArmChoice::SignMajority => arm_sign_majority(&refs, rows, cols)?,
        ArmChoice::SourceQuant => {
            let merged = op.merge(&refs)?;
            arm_source_quant(&merged, rows, cols)?
        }
    };
    match out {
        Materialized::Ternary(w) => Ok(*w),
        Materialized::DenseF16(_) => bail!("arm A is not a deployable audition candidate"),
    }
}

/// Merge one dense `GateProjWeights` field across members (ssm_alpha/beta,
/// the Bonsai-2 dense escape set — linear projections, so a merge is
/// coherent; unlike the never-averaged scalar decay params).
fn merged_gate_field(
    members: &[&DeltaNetTernaryLayerWeights],
    get: impl Fn(&DeltaNetTernaryLayerWeights) -> &GateProjWeights,
    dense_op: MergeOp,
) -> Result<GateProjWeights> {
    let mut rows = 0usize;
    let mut cols = 0usize;
    let mut dense: Vec<Vec<f32>> = Vec::with_capacity(members.len());
    for l in members {
        let (v, r, c) = match get(l) {
            GateProjWeights::Ternary(w) => (
                QwenDeltaNetTernaryWeights::dequant_proj_to_dense(w),
                w.rows,
                w.cols,
            ),
            GateProjWeights::Dense(v, r, c) => (v.clone(), *r, *c),
        };
        if rows == 0 {
            rows = r;
            cols = c;
        } else {
            assert_eq!((r, c), (rows, cols), "gate field shape drift");
        }
        dense.push(v);
    }
    let refs: Vec<&[f32]> = dense.iter().map(|v| v.as_slice()).collect();
    let merged = dense_op.merge(&refs)?;
    Ok(GateProjWeights::Dense(merged, rows, cols))
}

/// Merge one dense param vector with the mean (the RMSNorm-gamma menu;
/// T3.1 pins merged_mean for norms — the laya precedent).
fn merged_mean_f32(
    members: &[&DeltaNetTernaryLayerWeights],
    get: impl Fn(&DeltaNetTernaryLayerWeights) -> &[f32],
) -> Result<Vec<f32>> {
    let refs: Vec<&[f32]> = members.iter().map(|l| get(l)).collect();
    Ok(merge_mean(refs)?)
}

/// Build one merged candidate layer. `params_member` carries the
/// never-averaged GDN scalars; `dense_op` merges ssm_alpha/beta (mean
/// under arm B — its vote reads member codes, so the mean is the natural
/// dense twin — and the winning op under arm C).
fn build_merged_candidate(
    is_gdn: bool,
    members: &[&DeltaNetTernaryLayerWeights],
    op: MergeOp,
    arm: ArmChoice,
    params_member: &DeltaNetTernaryLayerWeights,
) -> Result<DeltaNetTernaryLayerWeights> {
    let dense_op = match arm {
        ArmChoice::SignMajority => MergeOp::Mean,
        ArmChoice::SourceQuant => op,
    };
    let empty = || TernaryGroupWeights::new(0, 0);
    if is_gdn {
        Ok(DeltaNetTernaryLayerWeights {
            attn_wq: empty(),
            attn_wk: empty(),
            attn_wv: empty(),
            attn_wo: empty(),
            in_proj_qkv: merged_ternary_field(members, |l| &l.in_proj_qkv, op, arm)?,
            in_proj_a: merged_gate_field(members, |l| &l.in_proj_a, dense_op)?,
            in_proj_b: merged_gate_field(members, |l| &l.in_proj_b, dense_op)?,
            in_proj_z: merged_ternary_field(members, |l| &l.in_proj_z, op, arm)?,
            out_proj: merged_ternary_field(members, |l| &l.out_proj, op, arm)?,
            gate_proj: merged_ternary_field(members, |l| &l.gate_proj, op, arm)?,
            up_proj: merged_ternary_field(members, |l| &l.up_proj, op, arm)?,
            down_proj: merged_ternary_field(members, |l| &l.down_proj, op, arm)?,
            attn_q_norm: vec![],
            attn_k_norm: vec![],
            // Never averaged (exp/sigmoid distortion — T3.1): from the
            // block's minimax medoid member.
            conv1d_weight: params_member.conv1d_weight.clone(),
            a_log: params_member.a_log.clone(),
            dt_bias: params_member.dt_bias.clone(),
            linear_norm: merged_mean_f32(members, |l| &l.linear_norm)?,
            input_norm: merged_mean_f32(members, |l| &l.input_norm)?,
            post_attn_norm: merged_mean_f32(members, |l| &l.post_attn_norm)?,
        })
    } else {
        Ok(DeltaNetTernaryLayerWeights {
            attn_wq: merged_ternary_field(members, |l| &l.attn_wq, op, arm)?,
            attn_wk: merged_ternary_field(members, |l| &l.attn_wk, op, arm)?,
            attn_wv: merged_ternary_field(members, |l| &l.attn_wv, op, arm)?,
            attn_wo: merged_ternary_field(members, |l| &l.attn_wo, op, arm)?,
            in_proj_qkv: empty(),
            in_proj_a: GateProjWeights::empty(),
            in_proj_b: GateProjWeights::empty(),
            in_proj_z: empty(),
            out_proj: empty(),
            gate_proj: merged_ternary_field(members, |l| &l.gate_proj, op, arm)?,
            up_proj: merged_ternary_field(members, |l| &l.up_proj, op, arm)?,
            down_proj: merged_ternary_field(members, |l| &l.down_proj, op, arm)?,
            attn_q_norm: merged_mean_f32(members, |l| &l.attn_q_norm)?,
            attn_k_norm: merged_mean_f32(members, |l| &l.attn_k_norm)?,
            conv1d_weight: vec![],
            a_log: vec![],
            dt_bias: vec![],
            linear_norm: vec![],
            input_norm: merged_mean_f32(members, |l| &l.input_norm)?,
            post_attn_norm: merged_mean_f32(members, |l| &l.post_attn_norm)?,
        })
    }
}

/// The h_in for a block at position p: the parent's post-embedding hidden
/// for a block starting at layer 0 (reproduced with the same lookup +
/// inverse fold the forward runs), else the captured pre-block boundary.
fn feed_h_in(
    x: &mut [f32],
    n: usize,
    weights: &QwenDeltaNetTernaryWeights,
    start_is_zero: bool,
    token: usize,
    cap: &[f32],
) {
    if start_is_zero {
        weights.dequant_wte_row_into(token, &mut x[..n]);
        if let Some(rot) = weights.rotation.as_ref()
            && rot.inverse_embedding
        {
            rotate_inverse_inplace(&mut x[..n], rot.signs_for_width(n), rot.block_size);
        }
    } else {
        x[..n].copy_from_slice(&cap[..n]);
    }
}

/// Run one candidate over one row, accumulating Σ‖f(h_in) − h_e‖². `he`
/// indexes the block's own captured boundary; `hin` the preceding one.
#[allow(clippy::too_many_arguments)]
fn eval_candidate_row(
    cand: &DeltaNetTernaryLayerWeights,
    is_gdn: bool,
    start_is_zero: bool,
    tokens: &[usize],
    hin: &[f32],
    he: &[f32],
    x: &mut [f32],
    n: usize,
    cache: &mut HybridCache,
    scratch: &mut HybridForwardScratch,
    config: &riir_infer_core::types::Config,
    rope_freq: &RopeFreqTable,
    rotation: Option<&riir_infer_core::deltanet::rotation::TernaryRotationConfig>,
    weights: &QwenDeltaNetTernaryWeights,
    row_len: usize,
) -> f64 {
    let mut err = 0.0f64;
    cache.reset();
    for p in 0..row_len {
        feed_h_in(x, n, weights, start_is_zero, tokens[p], &hin[p * n..]);
        qwen_deltanet_ternary_layer_body(
            x,
            cand,
            is_gdn,
            &mut cache.deltanet_state.recurrent_states[0],
            &mut cache.deltanet_state.conv_states[0],
            &mut cache.kv_cache.layers[0],
            p,
            config,
            scratch,
            rope_freq,
            rotation,
            None,
            None,
            None,
        );
        for i in 0..n {
            let d = x[i] - he[p * n + i];
            err += (d * d) as f64;
        }
    }
    err
}

fn main() -> Result<()> {
    let mut parent_path = None;
    let mut profile_path = None;
    let mut corpus_path = None;
    let mut out_path = None;
    let mut eps = 0.1f32;
    let mut rows = 8usize;
    let mut row_len = 256usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--parent" => parent_path = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--profile" => {
                profile_path = Some(std::path::PathBuf::from(args.next().expect("path")))
            }
            "--corpus" => corpus_path = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--out" => out_path = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--eps" => eps = args.next().expect("v").parse()?,
            "--rows" => rows = args.next().expect("v").parse()?,
            "--row-len" => row_len = args.next().expect("v").parse()?,
            other => bail!("unknown arg {other}"),
        }
    }
    let parent_path = parent_path.context("--parent is required")?;
    let profile_path = profile_path.context("--profile is required")?;
    let corpus_path = corpus_path.context("--corpus is required")?;
    let out_path = out_path.context("--out is required")?;
    assert!(rows >= 1 && row_len >= 2, "need rows ≥ 1, row_len ≥ 2");
    eprintln!(
        "[audition] box state: sibling agents may be loading this box — wall times are load-\
         contaminated; the SELECTION is deterministic (f64 sums in fixed order) and is the claim"
    );

    // ── 1. parent + types + typed partition ──
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&parent_path)
        .with_context(|| format!("load {}", parent_path.display()))?;
    eprintln!(
        "[audition] parent loaded in {:.1}s ({} layers, rotation {})",
        t0.elapsed().as_secs_f32(),
        config.n_layer,
        weights.rotation.is_some(),
    );
    let n_layer = config.n_layer;
    let n = config.n_embd;
    let parent_types: Vec<DeltaNetLayerType> = weights.layer_types.clone();
    let types: Vec<bool> = parent_types
        .iter()
        .map(|&t| t == DeltaNetLayerType::DeltaNet)
        .collect();

    let profile_text = std::fs::read_to_string(&profile_path)
        .with_context(|| format!("read {}", profile_path.display()))?;
    let profile: serde_json::Value = serde_json::from_str(&profile_text)?;
    let p_layers = profile["n_layers"].as_u64().context("n_layers")? as usize;
    assert_eq!(p_layers, n_layer, "profile S is for a different layer count");
    let entries = profile["S_upper"].as_array().context("S_upper")?;
    let s = SMatrix::from_fn(n_layer, |i, j| {
        entries
            .iter()
            .find_map(|e| {
                let a = e[0].as_u64().unwrap() as usize;
                let b = e[1].as_u64().unwrap() as usize;
                ((a, b) == (i, j)).then(|| e[2].as_f64().unwrap() as f32)
            })
            .unwrap_or_else(|| panic!("S_upper missing entry ({i},{j})"))
    });
    let blocks = minmax_partition_typed(&s, eps, &types).context("typed partition")?;
    let worst = partition_worst(&s, &blocks);
    let merge_block_idxs: Vec<usize> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.len() >= 2)
        .map(|(i, _)| i)
        .collect();
    eprintln!(
        "[audition] ε={eps} TYPED: {} blocks (depth {:.1}%), worst {worst:.4}, {} mergeable: {}",
        blocks.len(),
        100.0 * blocks.len() as f32 / n_layer as f32,
        merge_block_idxs.len(),
        merge_block_idxs
            .iter()
            .map(|&i| format!("[{},{})", blocks[i].start, blocks[i].end))
            .collect::<Vec<_>>()
            .join(" "),
    );
    assert!(!merge_block_idxs.is_empty(), "no mergeable blocks at this ε");
    let nb = blocks.len();

    // ── 2. corpus + tokens ──
    let text = load_corpus_text(&corpus_path)?;
    let tok = {
        let gguf = GgufFile::open(&parent_path).context("re-open parent for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer")?
    };
    let all = tok.encode(&text);
    assert!(
        all.len() >= rows * row_len,
        "corpus too short: {} tokens < {} rows × {row_len}",
        all.len(),
        rows
    );
    let corpus_blake3 = riir_infer_core::twt::blake3_of(text.as_bytes());

    // ── 3. setup (ONE all-GDN cache: slot 0 carries a real GDN state AND a
    //    real KV — MultiLayerKVCache sizes every layer regardless of type;
    //    the all-GDN type list is what keeps recurrent_states[0] real) ──
    config.block_size = row_len;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let all_gdn = vec![DeltaNetLayerType::DeltaNet; n_layer];
    let mut cache = HybridCache::with_layer_types(&config, &all_gdn);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x_full = vec![0.0f32; config.vocab_size.max(n)];
    let mut x_cand = vec![0.0f32; n];
    let mut capture = vec![vec![0.0f32; n]; n_layer];

    // h_e store: cap[((row*nb + bi) * row_len + p) * n + i]
    let mut cap = vec![0.0f32; rows * nb * row_len * n];
    let parity_bi = merge_block_idxs[0];
    let parity_layer = blocks[parity_bi].start;
    let mut parity_cap = vec![0.0f32; row_len * n];

    // ── 4. capture pass ──
    let t_cap = Instant::now();
    for row in 0..rows {
        cache.reset();
        for p in 0..row_len {
            let token = all[row * row_len + p];
            forward_qwen_deltanet_ternary_with_hook(
                &mut x_full,
                &weights,
                &mut cache,
                token,
                p,
                &config,
                &mut scratch,
                &rope_freq,
                Some(&mut capture),
                None,
                None,
                None,
            );
            for (bi, b) in blocks.iter().enumerate() {
                let off = ((row * nb + bi) * row_len + p) * n;
                cap[off..off + n].copy_from_slice(&capture[b.end - 1]);
            }
            if row == 0 {
                let off = p * n;
                parity_cap[off..off + n].copy_from_slice(&capture[parity_layer]);
            }
        }
        eprintln!(
            "[audition] capture row {}/{} done ({:.0}s elapsed)",
            row + 1,
            rows,
            t_cap.elapsed().as_secs_f32(),
        );

        // ── PREFLIGHT after row 0 ──
        if row == 0 {
            let b = blocks[parity_bi];
            let member = &weights.layers[b.start];
            let is_gdn = types[b.start];
            let mut max_diff = 0.0f64;
            // The parent's row-0 replay started from a FRESH state (the
            // capture loop's reset) — the preflight must too, or the
            // candidate carries row 0's final recurrence into position 0.
            cache.reset();
            for p in 0..row_len {
                let token = all[p];
                let hin: &[f32] = if b.start == 0 {
                    &[]
                } else {
                    // row 0, the block BEFORE the parity block
                    let prev_bi = parity_bi - 1;
                    &cap[(prev_bi * row_len + p) * n..]
                };
                feed_h_in(&mut x_cand, n, &weights, b.start == 0, token, hin);
                qwen_deltanet_ternary_layer_body(
                    &mut x_cand,
                    member,
                    is_gdn,
                    &mut cache.deltanet_state.recurrent_states[0],
                    &mut cache.deltanet_state.conv_states[0],
                    &mut cache.kv_cache.layers[0],
                    p,
                    &config,
                    &mut scratch,
                    &rope_freq,
                    weights.rotation.as_ref(),
                    None,
                    None,
                    None,
                );
                for i in 0..n {
                    max_diff = max_diff.max((x_cand[i] - parity_cap[p * n + i]).abs() as f64);
                }
            }
            eprintln!(
                "[audition] PREFLIGHT parity (layer {parity_layer} of block [{},{}) vs parent \
                 capture, {row_len} positions): max |Δ| = {max_diff:e}",
                b.start, b.end
            );
            if max_diff != 0.0 {
                bail!(
                    "apply-path parity FAILED (max |Δ| = {max_diff:e}) — the audition instrument \
                     does not mirror the forward; fix the instrument, never the checkpoint"
                );
            }
        }
    }
    let parity_diff = 0.0f64; // asserted above — recorded in the artifact

    // ── 5. audition (block-outer, in file order) ──
    let t_aud = Instant::now();
    let mut cand_total = 0usize;
    for &bi in &merge_block_idxs {
        cand_total += blocks[bi].len() + 3;
    }
    let mut cand_done = 0usize;
    let mut n_merged_winners = 0usize;
    let mut block_recs: Vec<BlockRec> = Vec::with_capacity(nb);
    for (bi, b) in blocks.iter().enumerate() {
        if b.len() == 1 {
            block_recs.push(BlockRec {
                start: b.start,
                end: b.end,
                kind: if types[b.start] { "deltanet" } else { "attention" },
                n_members: 1,
                candidates: vec![CandRec { id: format!("member:{}", b.start), err_sq: 0.0 }],
                winner: format!("member:{}", b.start),
                op: None,
                arm: "member".to_owned(),
                dense_op: "mean".to_owned(),
                params_member: b.start,
            });
            continue;
        }
        let k = b.len();
        let is_gdn = types[b.start];
        let members: Vec<&DeltaNetTernaryLayerWeights> =
            (b.start..b.end).map(|l| &weights.layers[l]).collect();
        let params_member = medoid(&s, *b);

        // pool: k members + 3 merged constructions
        let mut pool: Vec<(String, CandWeights)> = Vec::with_capacity(k + 3);
        for l in b.start..b.end {
            pool.push((format!("member:{l}"), CandWeights::Member(&weights.layers[l])));
        }
        for (op, arm) in [
            (MergeOp::Mean, ArmChoice::SignMajority),
            (MergeOp::Mean, ArmChoice::SourceQuant),
            (MergeOp::Rdsc, ArmChoice::SourceQuant),
        ] {
            let t_build = Instant::now();
            let layer = build_merged_candidate(
                is_gdn,
                &members,
                op,
                arm,
                &weights.layers[params_member],
            )
            .with_context(|| format!("build {op:?}/{arm:?} for [{},{})", b.start, b.end))?;
            eprintln!(
                "[audition] built {}:{} for [{},{}) in {:.1}s",
                op.name(),
                arm.name(),
                b.start,
                b.end,
                t_build.elapsed().as_secs_f32()
            );
            let id = match arm {
                ArmChoice::SignMajority => "sign_majority".to_owned(),
                ArmChoice::SourceQuant => format!("{}:source_quant", op.name()),
            };
            pool.push((id, CandWeights::Owned(Box::new(layer))));
        }

        // evaluate every candidate
        let mut recs: Vec<CandRec> = Vec::with_capacity(pool.len());
        let hin_base = if b.start == 0 { 0usize } else { bi - 1 };
        for (id, cand) in &pool {
            let t_c = Instant::now();
            let mut err = 0.0f64;
            for row in 0..rows {
                let tokens = &all[row * row_len..row * row_len + row_len];
                let hin = &cap[(row * nb + hin_base) * row_len * n..];
                let he = &cap[(row * nb + bi) * row_len * n..];
                err += eval_candidate_row(
                    cand.w(),
                    is_gdn,
                    b.start == 0,
                    tokens,
                    hin,
                    he,
                    &mut x_cand,
                    n,
                    &mut cache,
                    &mut scratch,
                    &config,
                    &rope_freq,
                    weights.rotation.as_ref(),
                    &weights,
                    row_len,
                );
            }
            recs.push(CandRec { id: id.clone(), err_sq: err });
            cand_done += 1;
            eprintln!(
                "[audition] [{},{}] {id} err {err:.4e} ({:.1}s, {cand_done}/{cand_total})",
                b.start,
                b.end,
                t_c.elapsed().as_secs_f32()
            );
        }

        // winner: argmin err (ties → lowest index: total-order scan)
        let mut best = 0usize;
        for (i, r) in recs.iter().enumerate() {
            if r.err_sq < recs[best].err_sq {
                best = i;
            }
        }
        let winner_id = recs[best].id.clone();
        // the pool order is members ascending then (mean,B),(mean,C),(rdsc,C)
        let (op_name, arm_name, dense_op_name) = match best.cmp(&k) {
            std::cmp::Ordering::Less => (None, "member".to_owned(), "mean".to_owned()),
            _ => {
                let (op, arm) = [
                    (MergeOp::Mean, ArmChoice::SignMajority),
                    (MergeOp::Mean, ArmChoice::SourceQuant),
                    (MergeOp::Rdsc, ArmChoice::SourceQuant),
                ][best - k];
                (
                    Some(op.name().to_owned()),
                    arm.name().to_owned(),
                    match arm {
                        ArmChoice::SignMajority => MergeOp::Mean.name().to_owned(),
                        ArmChoice::SourceQuant => op.name().to_owned(),
                    },
                )
            }
        };
        if best >= k {
            n_merged_winners += 1;
        }
        eprintln!(
            "[audition] block [{},{}) WINNER {winner_id} (err {:.4e})",
            b.start, b.end, recs[best].err_sq
        );
        block_recs.push(BlockRec {
            start: b.start,
            end: b.end,
            kind: if is_gdn { "deltanet" } else { "attention" },
            n_members: k,
            candidates: recs,
            winner: winner_id,
            op: op_name,
            arm: arm_name,
            dense_op: dense_op_name,
            params_member,
        });
    }
    eprintln!(
        "[audition] audition done in {:.0}s — {n_merged_winners}/{} mergeable blocks picked a MERGE",
        t_aud.elapsed().as_secs_f32(),
        merge_block_idxs.len()
    );

    let sel = Selection {
        eps,
        partition_mode: "typed",
        rows,
        row_len,
        corpus_file: corpus_path.display().to_string(),
        corpus_blake3_file: corpus_blake3,
        profile_corpus_blake3: profile["corpus_blake3"].as_str().unwrap_or("?").to_owned(),
        parity: ParityRec {
            layer: parity_layer,
            block: [blocks[parity_bi].start, blocks[parity_bi].end],
            positions: row_len,
            max_abs_diff: parity_diff,
        },
        blocks: block_recs,
    };
    let json = serde_json::to_string_pretty(&sel)?;
    std::fs::write(&out_path, &json).with_context(|| format!("write {}", out_path.display()))?;
    println!("# twt_bonsai_audition — selection written to {}", out_path.display());
    println!("# ε={eps} typed partition: {nb} blocks, {} mergeable", merge_block_idxs.len());
    println!("block\tspan\tk\twinner\terr_sq\tparams_member");
    for r in &sel.blocks {
        let werr = r
            .candidates
            .iter()
            .find(|c| c.id == r.winner)
            .map(|c| c.err_sq)
            .unwrap_or(f64::NAN);
        println!(
            "[{0},{1}).\t{2}\t{3}\t{4}\t{werr:.4e}",
            r.start, r.end, r.n_members, r.winner, r.params_member
        );
    }
    Ok(())
}
