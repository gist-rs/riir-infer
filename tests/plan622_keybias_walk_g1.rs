//! Plan 622 Phase 1 T1.3 — the per-key-bias G1 byte-identity WALK (CPU arms).
//!
//! Protocol adapted from the Bench-948-class retention walk (riir-gpu
//! `bench_948_attn_fa_retention_walk.rs`): a multi-window walk over the
//! 6-family real-text corpus (code / math / prose / structured /
//! multilingual / induction-recall) at multiple sequence lengths — but the
//! gate here is STRONGER than the 948 walk's 0-argmax-flip bar, because the
//! axis under test is an OPT-IN additive surface, not a tolerance-class
//! kernel: **zero bit differences**, every family, every length, every
//! softcap posture, every dispatch arm.
//!
//! Arms per cell: the incumbent kernel (`attention_heads_parallel`, the
//! pre-Plan-622 arithmetic, still shipped beside the seam) vs the bias arm
//! at `None` (dispatch identity — the add is skipped) vs the bias arm at
//! all-zero (arithmetic identity — `+ 0.0`). All three must be bit-equal.
//! The `-0.0` edge is why `None` and zero are DISTINCT postures (`-0.0 +
//! 0.0 = +0.0`); the generators exclude exact-zero inputs so the walk
//! pins the arithmetic identity on its domain, and the module doc on
//! `KeyBiasLayout` carries the standing law.
//!
//! Coverage (the T1.0 inventory rows 1–3, CPU):
//! - row 1 — the score-accumulate site itself (`attention_heads_parallel*`),
//!   all three dispatch regions: sequential (`t < 512`), the rayon fan-out
//!   (`t ≥ 512`), and the `n_head <= 1` branch; both softcap postures
//!   (the no-softcap and post-softcap head kernels).
//! - row 2 — the threading through a full attention layer
//!   (`forward_attention_layer`: gated-Q proj → QK-norm → partial RoPE →
//!   cache → heads → gate → o_proj) as a decode walk, plus the LAYOUT law:
//!   a bias on cache slot `s` moves exactly the positions `p ≥ s` and
//!   leaves every earlier position bit-identical.
//! - rows 2-tail/5-ref — the full prefill (`prefill_qwen_deltanet_into`:
//!   batched Phase A + per-position Phase B + batched MLP tail) at a
//!   synthetic hybrid [DeltaNet, Attention, Attention] model — the bias
//!   rides the attention layers and the GDN layer is untouched (its output
//!   is part of the byte-equal logits).
//!
//! NOT in this file (named, never silent): the CUDA arms — cudarc
//! decode+verify (row 5), prefill FA + canonical (rows 6–7) — join this
//! harness from the 4090 session (T1.2 lands first; the same
//! incumbent-vs-None-vs-zero protocol runs over the real league model).
//! The ternary model-level arms ride the synthetic Bonsai-2 GGUF path
//! (`tests/common/synth_bonsai2.rs`) in that same session; the bias SITE
//! those paths route through is the row-1 kernel covered here.

#![cfg(all(
    feature = "tokengate",
    feature = "deltanet_inference",
    not(target_arch = "wasm32")
))]

use riir_infer_core::deltanet::forward::{
    forward_attention_layer, prefill_qwen_deltanet_into, AttentionLayerScratch, HybridCache,
    HybridForwardScratch, PrefillContext,
};
use riir_infer_core::deltanet::weights::{DeltaNetLayerWeights, Proj, QwenDeltaNetWeights};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::transformer::{
    attention_heads_parallel, attention_heads_parallel_bias, KVCache, KeyBiasLayout,
};
use riir_infer_core::types::{Config, DeltaNetLayerType};

// ─── Determinism: FNV-1a seeding + splitmix64 stream ───────────────────────

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// splitmix64 — the walk's only randomness; a fixed seed rebuilds a cell
/// byte-for-byte on any box (the corpus seal is this file's git commit).
struct SplitMix(u64);

impl SplitMix {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let v = ((z ^ (z >> 31)) >> 40) as f32; // top 24 bits → [0, 2^24)
        v / 8_388_608.0 - 1.0 // [-1, 1)
    }

    fn stream_into(&mut self, dst: &mut [f32]) {
        for v in dst.iter_mut() {
            let x = self.next_f32();
            // No exact-zero elements: the -0.0 edge stays a named posture
            // distinction, never a walk flake.
            *v = if x == 0.0 { 1.0e-3 } else { x };
        }
    }

    fn stream_vec(&mut self, len: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        self.stream_into(&mut v);
        v
    }

    /// RMSNorm gamma rows: near-1, never zero.
    fn gamma_vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| 1.0 + 0.05 * self.next_f32()).collect()
    }
}

// ─── The 948-class family corpus (adapted; short seeds, same six classes) ──

#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Code,
    Math,
    Prose,
    Structured,
    Multilingual,
    Induction,
}

const FAMILIES: [Family; 6] = [
    Family::Code,
    Family::Math,
    Family::Prose,
    Family::Structured,
    Family::Multilingual,
    Family::Induction,
];

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::Code => "code",
            Family::Math => "math",
            Family::Prose => "prose",
            Family::Structured => "structured",
            Family::Multilingual => "multilingual",
            Family::Induction => "induction",
        }
    }

    fn seed_text(self) -> &'static str {
        match self {
            // Small-magnitude lattice code: cancellation-prone dot products.
            Family::Code => {
                "pub fn whole_prefill_inner(ctx: &Ctx, tokens: &[usize]) -> Vec<f32> \
                 { let p = tokens.len(); causal_attention(&q, &k, &v, &mut out, p, n_head) }"
            }
            // Wider magnitudes: softmax saturation pressure at long spans.
            Family::Math => {
                "a reservoir holds 12450 cubic meters at dawn; outflow exceeds inflow \
                 by 3.2 percent per hour; the recurrence V(t+1) = V(t) - k*V(t) + S(t) \
                 solves to k = 0.0297 per hour and the check lands within 0.6 percent"
            }
            // Unit-normalized narrative prose rows.
            Family::Prose => {
                "The lighthouse keeper had kept the log for forty-one years and the \
                 entries had never once mentioned the sea; he dipped the pen and \
                 wrote: calm at last, gulls returned, the light burning clean."
            }
            // Mixed-magnitude structured rows (alternating scale by slot).
            Family::Structured => {
                "{\"service\": \"kat-ledger\", \"version\": 3, \"tiers\": \
                   {\"plasma\": {\"write\": \"in-memory\"}, \"hot\": {\"write\": \"wal\"}}, \
                   \"epoch\": {\"number\": 2958, \"pending\": 154}}"
            }
            // Exact duplicate key rows (softmax tie ordering under the bias).
            Family::Multilingual => {
                "東京の朝は早い、駅前の小さな喫茶店でホットコーヒーを頼む。 \
                 เรือใบใหญ่ลอยอยู่บนทะเลกลางคืน Le vent soufflait fort sur la place \
                 déserte ce soir Der Zug hielt mit zwanzig Minuten Verspätung"
            }
            // Near-one-hot keys: the long-range KV-recall stressor.
            Family::Induction => {
                "alpha->4f7b beta->9c1d gamma->2e8a delta->7b3f epsilon->6a2c \
                 zeta->1d9e query: what follows gamma? answer 2e8a"
            }
        }
    }
}

/// One walk cell's q / key / value tensors, family-shaped.
///
/// The families stress different numeric paths of the SAME identity law:
/// lattice quantization (cancellation), wide magnitudes (saturation),
/// unit rows (balanced), alternating scale (mixed), duplicate keys (exact
/// softmax ties), near-one-hot keys (argmax-critical sparse dots).
fn fill_family_inputs(
    family: Family,
    seed: u64,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    t_n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kv_dim = n_kv * hd;
    let mut sm = SplitMix(seed);
    let q = sm.stream_vec(n_head * hd);
    let mut key = sm.stream_vec(t_n * kv_dim);
    let mut value = sm.stream_vec(t_n * kv_dim);

    match family {
        Family::Code => {
            // Quarter-lattice magnitudes: cancellation-prone dots. Only an
            // exactly-zero lattice point is nudged off zero (the no-zero law).
            for v in key.iter_mut().chain(value.iter_mut()) {
                let q = (*v * 4.0).round() / 4.0;
                *v = if q == 0.0 { 1.0e-3 } else { q };
            }
        }
        Family::Math => {
            for v in key.iter_mut().chain(value.iter_mut()) {
                *v *= 2.0;
            }
        }
        Family::Prose => {
            for row in key.chunks_mut(kv_dim) {
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0e-6);
                for x in row.iter_mut() {
                    *x *= 0.5 / norm;
                }
            }
        }
        Family::Structured => {
            for (t, row) in key.chunks_mut(kv_dim).enumerate() {
                let s = if t % 2 == 0 { 4.0 } else { 0.25 };
                for x in row.iter_mut() {
                    *x *= s;
                }
            }
        }
        Family::Multilingual => {
            // Exact duplicate key rows: row t copies row t-4 for t % 8 == 7 —
            // equal scores → equal exp → the tie ordering must survive both
            // inert arms identically.
            for t in (8..t_n).step_by(8) {
                let (src, dst) = (t - 4, t);
                key.copy_within(src * kv_dim..src * kv_dim + kv_dim, dst * kv_dim);
            }
        }
        Family::Induction => {
            // Near-one-hot keys: one ±1 pivot per key row (hash-derived
            // position), everything else zero. Dots reduce to single q
            // elements — nonzero by the generator's no-zero law.
            let mut h = fnv1a("induction-pivots".as_bytes());
            for row in key.chunks_mut(kv_dim) {
                h = h.wrapping_mul(0x100_0000_01b3).wrapping_add(0x9e37);
                let pivot = (h % kv_dim as u64) as usize;
                let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
                for x in row.iter_mut() {
                    *x = 0.0;
                }
                row[pivot] = sign;
            }
        }
    }
    (q, key, value)
}

// ─── League geometries × multi-window lengths ──────────────────────────────

struct Geom {
    name: &'static str,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    lengths: &'static [usize],
}

const LENGTHS_LEAGUE_A: [usize; 4] = [37, 128, 289, 512];
const LENGTHS_LEAGUE_B: [usize; 3] = [37, 128, 257];
const LENGTHS_SINGLE_HEAD: [usize; 2] = [5, 37];

const GEOMS: [Geom; 3] = [
    // The documented qwen3.5-family serving geometry (head_dim 256, 4:1 GQA).
    Geom {
        name: "league-a-qwen35-family-8q2kv-hd256",
        n_head: 8,
        n_kv: 2,
        hd: 256,
        lengths: &LENGTHS_LEAGUE_A,
    },
    // The 27B-class 4:1 GQA geometry.
    Geom {
        name: "league-b-27b-class-32q8kv-hd128",
        n_head: 32,
        n_kv: 8,
        hd: 128,
        lengths: &LENGTHS_LEAGUE_B,
    },
    // The `n_head <= 1` sequential branch.
    Geom {
        name: "single-head-branch",
        n_head: 1,
        n_kv: 1,
        hd: 64,
        lengths: &LENGTHS_SINGLE_HEAD,
    },
];

/// Row stride for the score scratch (and the bias rows in the binds cell):
/// ≥ the largest walk length.
const SCORE_STRIDE: usize = 512;

fn assert_bits_eq(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: elem {i} differs ({x:?} vs {y:?})"
        );
    }
}

// ─── Row 1: the heads walk ─────────────────────────────────────────────────

#[test]
fn walk_none_and_zero_bias_are_bit_identical_to_the_incumbent() {
    for geom in &GEOMS {
        for &t_n in geom.lengths {
            for family in FAMILIES {
                let seed = fnv1a(geom.name.as_bytes())
                    ^ fnv1a(family.seed_text().as_bytes())
                    ^ (t_n as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                let (q, key, value) =
                    fill_family_inputs(family, seed, geom.n_head, geom.n_kv, geom.hd, t_n);
                let kv_dim = geom.n_kv * geom.hd;
                let scale = 1.0 / (geom.hd as f32).sqrt();
                let cells = format!("{}/{}/t={t_n}", geom.name, family.name());

                for softcap in [0.0f32, 30.0] {
                    let mut out_ref = vec![0.0f32; geom.n_head * geom.hd];
                    let mut scores_ref = vec![0.0f32; geom.n_head * SCORE_STRIDE];
                    unsafe {
                        attention_heads_parallel(
                            &q,
                            &key,
                            &value,
                            &mut out_ref,
                            &mut scores_ref,
                            geom.n_head,
                            geom.n_kv,
                            kv_dim,
                            geom.hd,
                            t_n,
                            scale,
                            softcap,
                            SCORE_STRIDE,
                        );
                    }

                    // None = dispatch identity: the add is skipped.
                    let mut out_none = vec![0.0f32; geom.n_head * geom.hd];
                    let mut scores_none = vec![0.0f32; geom.n_head * SCORE_STRIDE];
                    unsafe {
                        attention_heads_parallel_bias(
                            &q,
                            &key,
                            &value,
                            &mut out_none,
                            &mut scores_none,
                            geom.n_head,
                            geom.n_kv,
                            kv_dim,
                            geom.hd,
                            t_n,
                            scale,
                            softcap,
                            SCORE_STRIDE,
                            None,
                        );
                    }
                    assert_bits_eq(&out_ref, &out_none, &format!("{cells}: None vs incumbent"));

                    // All-zero = arithmetic identity (+0.0 on a nonzero domain).
                    let zeros = vec![0.0f32; geom.n_head * t_n];
                    let mut out_zero = vec![0.0f32; geom.n_head * geom.hd];
                    let mut scores_zero = vec![0.0f32; geom.n_head * SCORE_STRIDE];
                    unsafe {
                        attention_heads_parallel_bias(
                            &q,
                            &key,
                            &value,
                            &mut out_zero,
                            &mut scores_zero,
                            geom.n_head,
                            geom.n_kv,
                            kv_dim,
                            geom.hd,
                            t_n,
                            scale,
                            softcap,
                            SCORE_STRIDE,
                            Some((&zeros, KeyBiasLayout { head_stride: t_n })),
                        );
                    }
                    assert_bits_eq(&out_ref, &out_zero, &format!("{cells}: zero vs incumbent"));
                }
            }
        }
    }
}

/// The seam BINDS at a league shape: a +12 logit bias on head 0's slot 3
/// moves head 0 and leaves every other head bit-identical (per-head rows
/// are independent under GQA at the production geometry, not just the
/// T1.1 fixture's 4×2).
#[test]
fn walk_binds_at_league_shape_and_respects_per_head_rows() {
    let geom = &GEOMS[0];
    let t_n = 128usize;
    let (q, key, value) = fill_family_inputs(
        Family::Induction,
        fnv1a(b"binds-cell"),
        geom.n_head,
        geom.n_kv,
        geom.hd,
        t_n,
    );
    let kv_dim = geom.n_kv * geom.hd;
    let scale = 1.0 / (geom.hd as f32).sqrt();
    let softcap = 0.0f32;

    let mut out_ref = vec![0.0f32; geom.n_head * geom.hd];
    let mut scores = vec![0.0f32; geom.n_head * SCORE_STRIDE];
    unsafe {
        attention_heads_parallel(
            &q, &key, &value, &mut out_ref, &mut scores, geom.n_head, geom.n_kv, kv_dim,
            geom.hd, t_n, scale, softcap, SCORE_STRIDE,
        );
    }

    let mut bias = vec![0.0f32; geom.n_head * t_n];
    bias[3] = 12.0; // head 0 (row 0), key slot 3
    let mut out_biased = vec![0.0f32; geom.n_head * geom.hd];
    let mut scores_b = vec![0.0f32; geom.n_head * SCORE_STRIDE];
    unsafe {
        attention_heads_parallel_bias(
            &q, &key, &value, &mut out_biased, &mut scores_b, geom.n_head, geom.n_kv, kv_dim,
            geom.hd, t_n, scale, softcap, SCORE_STRIDE,
            Some((&bias, KeyBiasLayout { head_stride: t_n })),
        );
    }

    let hd = geom.hd;
    assert!(
        out_biased[..hd]
            .iter()
            .zip(&out_ref[..hd])
            .all(|(b, r)| b.to_bits() != r.to_bits()),
        "head 0 output must move under a +12 logit bias"
    );
    for h in 1..geom.n_head {
        let (b, r) = (&out_biased[h * hd..(h + 1) * hd], &out_ref[h * hd..(h + 1) * hd]);
        assert_bits_eq(r, b, &format!("head {h} (zero bias row) vs incumbent"));
    }
}

// ─── Row 2: the layer-level threading walk ─────────────────────────────────

/// League-family-shaped layer config (4:1 GQA, head_dim 128): big enough to
/// exercise the full threading at a production ratio, small enough for a
/// dev-profile test run. The row-1 walk carries the head_dim-256 geometry.
fn layer_walk_config() -> Config {
    Config {
        vocab_size: 97,
        block_size: 200,
        n_embd: 256,
        n_head: 8,
        head_dim: 128,
        mlp_hidden: 512,
        n_layer: 1,
        n_kv_head: 2,
        use_rope: true,
        rope_dimension_count: 32, // partial RoPE, 0.25 factor
        ..Default::default()
    }
}

fn synth_attention_layer(cfg: &Config, seed: u64) -> DeltaNetLayerWeights {
    let mut sm = SplitMix(seed);
    let n = cfg.n_embd;
    let q_dim = cfg.n_head * cfg.head_dim;
    let kvd = cfg.n_kv_head * cfg.head_dim;
    let mlp = cfg.mlp_hidden;
    let mut wq = sm.stream_vec(2 * q_dim * n);
    // Keep the output-gate half (odd per-head row blocks of the interleaved
    // `[q(hd), gate(hd)]*n_head` layout) in fast_sigmoid's unclamped range:
    // a ±1 stream over 256 dims reaches the |x| > 40 exact-0.0/1.0 clamp,
    // and a clamped-to-0 gate annihilates its head's attention output —
    // the bias could not propagate. Real checkpoints sit far from the
    // clamp; the walk tests the bias math, not the sigmoid tail.
    let hd = cfg.head_dim;
    for block in (0..2 * cfg.n_head).step_by(2) {
        let gate_off = (block + 1) * hd * n;
        for v in &mut wq[gate_off..gate_off + hd * n] {
            *v *= 0.1;
        }
    }
    // Keep the decode-walk loop gain below 1: a synthetic ±1 `attn_wo` has
    // trained-weight-scale nothing, and the per-position feedback
    // (x ← o_proj(attn_out·gate)) grows x ~×10³/step — inf by pos 7, NaN by
    // pos 8 (both bias arms NaN-identical, which reads as "no movement").
    // Real checkpoints carry trained (small-norm) output projections; the
    // attention path the bias lives in is norm-invariant (q/k/v/gate all
    // read the RMS-normalized input), so this scaling changes only the
    // fixture's magnitude stability, never the law under test.
    let mut wo = sm.stream_vec(n * q_dim);
    for v in &mut wo {
        *v *= 0.01;
    }
    DeltaNetLayerWeights {
        attn_wq: Proj::dense(wq, 2 * q_dim, n),
        attn_wk: Proj::dense(sm.stream_vec(kvd * n), kvd, n),
        attn_wv: Proj::dense(sm.stream_vec(kvd * n), kvd, n),
        attn_wo: Proj::dense(wo, n, q_dim),
        attn_q_norm: sm.gamma_vec(cfg.head_dim),
        attn_k_norm: sm.gamma_vec(cfg.head_dim),
        in_proj_qkv: Proj::empty(),
        in_proj_a: Proj::empty(),
        in_proj_b: Proj::empty(),
        in_proj_z: Proj::empty(),
        out_proj: Proj::empty(),
        conv1d_weight: Vec::new(),
        a_log: Vec::new(),
        dt_bias: Vec::new(),
        linear_norm: Vec::new(),
        gate_proj: Proj::dense(sm.stream_vec(mlp * n), mlp, n),
        up_proj: Proj::dense(sm.stream_vec(mlp * n), mlp, n),
        down_proj: Proj::dense(sm.stream_vec(n * mlp), n, mlp),
        input_norm: sm.gamma_vec(n),
        post_attn_norm: sm.gamma_vec(n),
    }
}

/// Decode-walk law, at the layer level:
/// 1. `None` ≡ all-zero bias over the WHOLE 193-position trajectory
///    (bit-exact after every position, not just the last).
/// 2. Slot scoping: a +12 bias on head 0's cache slot 160 moves exactly the
///    positions `p ≥ 160` and leaves positions `p < 160` bit-identical —
///    the slot-indexed layout contract, threaded through proj → norm →
///    RoPE → cache → heads → gate → o_proj.
#[test]
fn layer_walk_none_zero_bit_identity_and_slot_scoping() {
    let cfg = layer_walk_config();
    let layer = synth_attention_layer(&cfg, fnv1a(b"layer-walk-weights"));
    let rope = RopeFreqTable::new(cfg.rope_theta, cfg.head_dim);
    let t_walk = 193usize; // < block_size (200); spans sequential dispatch only
    let n = cfg.n_embd;

    let mut sm = SplitMix(fnv1a(b"layer-walk-x0"));
    let x0 = sm.stream_vec(n);

    let run_arm = |bias: Option<(&Vec<f32>, KeyBiasLayout)>| -> Vec<Vec<f32>> {
        let mut cache = KVCache::new(&cfg);
        let mut scratch = AttentionLayerScratch::new(&cfg);
        let mut traj = Vec::with_capacity(t_walk);
        let mut x = x0.clone();
        for pos in 0..t_walk {
            forward_attention_layer(
                &mut x,
                &layer,
                &mut cache,
                pos,
                &cfg,
                &rope,
                &mut scratch,
                bias.as_ref().map(|(d, l)| (&d[..], *l)),
            );
            traj.push(x.clone());
        }
        traj
    };

    let none_traj = run_arm(None);
    let zeros = vec![0.0f32; cfg.n_head * cfg.block_size];
    let zero_traj = run_arm(Some((
        &zeros,
        KeyBiasLayout {
            head_stride: cfg.block_size,
        },
    )));

    // Law 1: the whole trajectory is bit-identical.
    assert_eq!(none_traj.len(), zero_traj.len());
    for (p, (a, b)) in none_traj.iter().zip(&zero_traj).enumerate() {
        assert_bits_eq(a, b, &format!("layer walk pos {p}: None vs zero"));
    }

    // Law 2: slot-scoped bias.
    let slot = 160usize;
    let mut bound = vec![0.0f32; cfg.n_head * cfg.block_size];
    bound[slot] = 12.0; // head 0, cache slot 160
    let bound_traj = run_arm(Some((
        &bound,
        KeyBiasLayout {
            head_stride: cfg.block_size,
        },
    )));

    for p in 0..slot {
        assert_bits_eq(
            &none_traj[p],
            &bound_traj[p],
            &format!("layer walk pos {p} (< slot {slot}): must be untouched"),
        );
    }
    for p in slot..t_walk {
        let max_diff = none_traj[p]
            .iter()
            .zip(&bound_traj[p])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.0,
            "layer walk pos {p} (≥ slot {slot}): head 0's bias must reach the output"
        );
    }
}

// ─── Rows 2-tail/5-ref: the full-prefill logits walk ────────────────────────

/// Synthetic hybrid model: [DeltaNet, Attention, Attention] — the bias must
/// ride both attention layers and leave the GDN layer's contribution
/// bit-identical, through the batched Phase A + per-position Phase B +
/// batched MLP tail + lm_head.
fn hybrid_config() -> Config {
    Config {
        vocab_size: 64,
        block_size: 128,
        n_embd: 128,
        n_head: 4,
        n_kv_head: 2,
        head_dim: 32,
        mlp_hidden: 256,
        n_layer: 3,
        use_rope: true,
        rope_dimension_count: 8,
        deltanet_linear_n_heads: 2,
        deltanet_linear_n_value_heads: 2,
        deltanet_linear_head_dim: 8,
        deltanet_conv_kernel_size: 4,
        ..Default::default()
    }
}

fn synth_hybrid_weights(cfg: &Config, seed: u64) -> QwenDeltaNetWeights {
    let mut sm = SplitMix(seed);
    let n = cfg.n_embd;
    let q_dim = cfg.n_head * cfg.head_dim;
    let kvd = cfg.n_kv_head * cfg.head_dim;
    let mlp = cfg.mlp_hidden;

    let attention_layer = |sm: &mut SplitMix| {
        let mut wq = sm.stream_vec(2 * q_dim * n);
        // Same gate-range law as `synth_attention_layer`: keep the gate half
        // out of fast_sigmoid's exact-0/1 clamp so the bias propagates.
        let hd = cfg.head_dim;
        for block in (0..2 * cfg.n_head).step_by(2) {
            let gate_off = (block + 1) * hd * n;
            for v in &mut wq[gate_off..gate_off + hd * n] {
                *v *= 0.1;
            }
        }
        // Same loop-gain law: trained-scale output projection (the prefill's
        // residual stream is additive across 3 layers × 64 positions — an
        // untrained ±1 attn_wo still stays bounded here, but keep the
        // fixture magnitudes comparable to the decode walk).
        let mut wo = sm.stream_vec(n * q_dim);
        for v in &mut wo {
            *v *= 0.01;
        }
        DeltaNetLayerWeights {
        attn_wq: Proj::dense(wq, 2 * q_dim, n),
        attn_wk: Proj::dense(sm.stream_vec(kvd * n), kvd, n),
        attn_wv: Proj::dense(sm.stream_vec(kvd * n), kvd, n),
        attn_wo: Proj::dense(wo, n, q_dim),
        attn_q_norm: sm.gamma_vec(cfg.head_dim),
        attn_k_norm: sm.gamma_vec(cfg.head_dim),
        in_proj_qkv: Proj::empty(),
        in_proj_a: Proj::empty(),
        in_proj_b: Proj::empty(),
        in_proj_z: Proj::empty(),
        out_proj: Proj::empty(),
        conv1d_weight: Vec::new(),
        a_log: Vec::new(),
        dt_bias: Vec::new(),
        linear_norm: Vec::new(),
        gate_proj: Proj::dense(sm.stream_vec(mlp * n), mlp, n),
        up_proj: Proj::dense(sm.stream_vec(mlp * n), mlp, n),
        down_proj: Proj::dense(sm.stream_vec(n * mlp), n, mlp),
        input_norm: sm.gamma_vec(n),
        post_attn_norm: sm.gamma_vec(n),
        }
    };

    let n_k = cfg.deltanet_linear_n_heads;
    let n_v = cfg.deltanet_linear_n_value_heads;
    let lhd = cfg.deltanet_linear_head_dim;
    let k = cfg.deltanet_conv_kernel_size;
    let qkv_dim = (n_k * 2 + n_v) * lhd;
    let v_dim = n_v * lhd;
    let deltanet_layer = DeltaNetLayerWeights {
        attn_wq: Proj::empty(),
        attn_wk: Proj::empty(),
        attn_wv: Proj::empty(),
        attn_wo: Proj::empty(),
        attn_q_norm: Vec::new(),
        attn_k_norm: Vec::new(),
        in_proj_qkv: Proj::dense(sm.stream_vec(qkv_dim * n), qkv_dim, n),
        in_proj_a: Proj::dense(sm.stream_vec(n_k * n), n_k, n),
        in_proj_b: Proj::dense(sm.stream_vec(n_k * n), n_k, n),
        in_proj_z: Proj::dense(sm.stream_vec(v_dim * n), v_dim, n),
        out_proj: Proj::dense(sm.stream_vec(n * v_dim), n, v_dim),
        conv1d_weight: sm.stream_vec(qkv_dim * k), // [conv_dim, kernel] — conv_dim = qkv_dim
        // Decay in (0, 1): a = exp(a_log) < 1 keeps the recurrence bounded.
        a_log: vec![-0.5; n_v],
        dt_bias: sm.stream_vec(n_k * 2 + n_v).iter().map(|x| x * 0.05).collect(),
        linear_norm: sm.gamma_vec(lhd),
        gate_proj: Proj::dense(sm.stream_vec(mlp * n), mlp, n),
        up_proj: Proj::dense(sm.stream_vec(mlp * n), mlp, n),
        down_proj: Proj::dense(sm.stream_vec(n * mlp), n, mlp),
        input_norm: sm.gamma_vec(n),
        post_attn_norm: sm.gamma_vec(n),
    };

    let layer_types = vec![
        DeltaNetLayerType::DeltaNet,
        DeltaNetLayerType::Attention,
        DeltaNetLayerType::Attention,
    ];
    QwenDeltaNetWeights {
        wte: sm.stream_vec(cfg.vocab_size * n),
        final_norm: sm.gamma_vec(n),
        lm_head: Proj::dense(sm.stream_vec(cfg.vocab_size * n), cfg.vocab_size, n),
        layers: vec![deltanet_layer, attention_layer(&mut sm), attention_layer(&mut sm)],
        layer_types,
    }
}

#[test]
fn prefill_walk_none_and_zero_bias_logits_bit_identical() {
    let cfg = hybrid_config();
    let weights = synth_hybrid_weights(&cfg, fnv1a(b"prefill-walk-weights"));
    let rope = RopeFreqTable::new(cfg.rope_theta, cfg.head_dim);
    let v = cfg.vocab_size;

    let run_arm = |tokens: &[usize], bias: Option<(&Vec<f32>, KeyBiasLayout)>| -> Vec<f32> {
        let layer_types = weights.layer_types.clone();
        let mut cache = HybridCache::with_layer_types(&cfg, &layer_types);
        let mut scratch = HybridForwardScratch::new(&cfg);
        let mut ctx = PrefillContext::new(&cfg, 128);
        let mut logits = vec![0.0f32; v];
        prefill_qwen_deltanet_into(
            &weights,
            &cfg,
            &mut cache,
            tokens,
            &mut scratch,
            &rope,
            &mut ctx,
            &mut logits,
            bias.as_ref().map(|(d, l)| (&d[..], *l)),
        );
        logits
    };

    for family in [Family::Code, Family::Induction] {
        for &t_n in &[1usize, 37, 64] {
            let mut sm = SplitMix(
                fnv1a(family.seed_text().as_bytes())
                    ^ (t_n as u64).wrapping_mul(0x2545_F491_4F6C_DD1D),
            );
            let tokens: Vec<usize> = (0..t_n)
                .map(|_| (sm.next_f32().abs() * v as f32) as usize % v)
                .collect();
            let cells = format!("prefill {}/t={t_n}", family.name());

            let none_logits = run_arm(&tokens, None);
            let zeros = vec![0.0f32; cfg.n_head * t_n];
            let zero_logits = run_arm(
                &tokens,
                Some((&zeros, KeyBiasLayout { head_stride: t_n })),
            );
            assert_bits_eq(&none_logits, &zero_logits, &format!("{cells}: None vs zero"));

            // The seam binds through the full prefill — EXCEPT at t_n == 1,
            // where the law is the OPPOSITE: a single-key softmax is
            // bias-invariant (σ of one logit is 1.0 whatever the logit), so
            // the bias must NOT move the logits. Assert that law at t=1 and
            // the movement law everywhere else.
            let slot = t_n / 2;
            let mut bound = vec![0.0f32; cfg.n_head * t_n];
            bound[slot] = 9.0;
            let bound_logits = run_arm(
                &tokens,
                Some((&bound, KeyBiasLayout { head_stride: t_n })),
            );
            let moved = none_logits
                .iter()
                .zip(&bound_logits)
                .any(|(a, b)| a.to_bits() != b.to_bits());
            if t_n == 1 {
                assert!(
                    !moved,
                    "{cells}: a single-key softmax must be bias-invariant"
                );
            } else {
                assert!(
                    moved,
                    "{cells}: a mid-span head-0 bias must move the logits"
                );
            }
        }
    }
}
