//! G1 correctness gate for `forward_ternary` (Plan 333 T2.2).
//!
//! The load-bearing test is [`g1_ternary_matches_dense_equivalent`]: a ternary
//! model and its **materialized dense f32 equivalent** must produce the same
//! logits. That isolates the one thing this file actually introduces — the
//! `BitLinear` wiring — from the kernel (gated in katgpt-rs Issue 578) and from
//! the transformer shape (gated by `forward_llama`'s own tests).
//!
//! If the wiring transposes a matrix, drops a group scale, feeds the wrong
//! residual, or slices a buffer to the wrong length, the two diverge.

use super::*;
use crate::llama_layer::{LlamaLayerWeights, LlamaTransformerWeights};
use crate::ternary_layer::{TernaryLayerWeights, TernaryTransformerWeights};
use katgpt_core::types::{Config, GROUP_SIZE, ModelArchitecture};

// ── Fixtures ────────────────────────────────────────────────────────────────

/// Deterministic LCG — no `rand` dep, reproducible across runs.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 33) as f32) / (1u64 << 31) as f32 - 1.0
    }

    /// Ternary `{-1, 0, +1}` — roughly 1/3 each, matching real ternary sparsity.
    fn next_ternary(&mut self) -> i8 {
        match self.next_u64() % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        }
    }
}

/// `n_embd = 128` so `cols` is an exact multiple of `GROUP_SIZE` on the
/// square projections, and `mlp_hidden = 256` so `down_proj` spans **two**
/// groups per row — a single-group fixture would not catch a `group_scale`
/// indexing bug.
fn tiny_config() -> Config {
    Config {
        vocab_size: 32,
        block_size: 32,
        n_embd: 128,
        n_head: 4,
        n_kv_head: 2,
        head_dim: 32,
        n_layer: 2,
        mlp_hidden: 256,
        rms_norm_eps: 1e-6,
        model_arch: ModelArchitecture::Ternary,
        tied_embeddings: false,
        use_rope: true,
        post_norm: false,
        ..Default::default()
    }
}

/// Random ternary weights with a random per-group scale.
fn random_ternary(rng: &mut Lcg, rows: usize, cols: usize) -> TernaryGroupWeights {
    let mut w = TernaryGroupWeights::new(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            w.set(r, c, rng.next_ternary());
        }
        // Keep scales small and positive — a plausible quantization scale, and
        // far enough from f16 subnormals that the dense mirror is exact.
        for g in 0..w.groups_per_row {
            w.set_scale(r, g, 0.05 + 0.1 * (rng.next_f32() + 1.0));
        }
    }
    w
}

/// Materialize the exact dense f32 matrix the ternary container represents.
///
/// `scale_at` returns the f16 scale widened to f32, so this mirror is exact:
/// any logit difference between the two forward passes comes from float
/// **accumulation order**, never from a value mismatch.
fn densify(w: &TernaryGroupWeights) -> Vec<f32> {
    let mut dense = vec![0.0f32; w.rows * w.cols];
    for r in 0..w.rows {
        for c in 0..w.cols {
            let scale = w.scale_at(r, c / GROUP_SIZE);
            dense[r * w.cols + c] = f32::from(w.get(r, c)) * scale;
        }
    }
    dense
}

/// A ternary model plus its bit-exact dense mirror (same embeddings, norms,
/// and LM head — only the projection container differs).
fn make_pair(seed: u64) -> (Config, TernaryTransformerWeights, LlamaTransformerWeights) {
    let config = tiny_config();
    let n = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let kvd = config.n_kv_head * config.head_dim;
    let mlp = config.mlp_hidden;
    let vocab = config.vocab_size;

    let mut rng = Lcg::new(seed);

    let mut wte = vec![0.0f32; vocab * n];
    for x in wte.iter_mut() {
        *x = rng.next_f32();
    }
    let mut lm_head = vec![0.0f32; vocab * n];
    for x in lm_head.iter_mut() {
        *x = rng.next_f32();
    }
    let mut final_norm = vec![0.0f32; n];
    for x in final_norm.iter_mut() {
        *x = 0.5 + 0.5 * (rng.next_f32() + 1.0);
    }

    let mut bit_layers = Vec::with_capacity(config.n_layer);
    let mut dense_layers = Vec::with_capacity(config.n_layer);
    for _ in 0..config.n_layer {
        let attn_wq = random_ternary(&mut rng, q_dim, n);
        let attn_wk = random_ternary(&mut rng, kvd, n);
        let attn_wv = random_ternary(&mut rng, kvd, n);
        let attn_wo = random_ternary(&mut rng, n, q_dim);
        let gate_proj = random_ternary(&mut rng, mlp, n);
        let up_proj = random_ternary(&mut rng, mlp, n);
        let down_proj = random_ternary(&mut rng, n, mlp);

        let mut input_norm = vec![0.0f32; n];
        let mut post_attn_norm = vec![0.0f32; n];
        for x in input_norm.iter_mut().chain(post_attn_norm.iter_mut()) {
            *x = 0.5 + 0.5 * (rng.next_f32() + 1.0);
        }

        dense_layers.push(LlamaLayerWeights {
            attn_wq: densify(&attn_wq),
            attn_wk: densify(&attn_wk),
            attn_wv: densify(&attn_wv),
            attn_wo: densify(&attn_wo),
            gate_proj: densify(&gate_proj),
            up_proj: densify(&up_proj),
            down_proj: densify(&down_proj),
            input_norm: input_norm.clone(),
            post_attn_norm: post_attn_norm.clone(),
        });
        bit_layers.push(TernaryLayerWeights {
            attn_wq,
            attn_wk,
            attn_wv,
            attn_wo,
            gate_proj,
            up_proj,
            down_proj,
            input_norm,
            post_attn_norm,
        });
    }

    let ternary = TernaryTransformerWeights {
        wte: wte.clone(),
        lm_head: lm_head.clone(),
        final_norm: final_norm.clone(),
        layers: bit_layers,
    };
    let dense = LlamaTransformerWeights {
        wte,
        lm_head,
        final_norm,
        layers: dense_layers,
    };
    (config, ternary, dense)
}

// ── G1: the ternary path matches its dense equivalent ───────────────────────

/// The gate. Same weights, two containers, same logits.
///
/// Tolerance is on **relative** error: the ternary kernel accumulates
/// per-128-group then sums, while `matmul` runs a straight SIMD dot product,
/// so the two differ in float rounding but nothing else. `1e-4` relative is
/// ~3 orders of magnitude tighter than any wiring bug would survive (a
/// transposed matrix or a dropped scale moves logits by O(1)).
#[test]
fn g1_ternary_matches_dense_equivalent() {
    let (config, ternary, dense) = make_pair(0xB17E);

    let mut bit_ctx = ForwardContext::new(&config);
    let mut bit_cache = MultiLayerKVCache::new(&config);
    let mut dense_ctx = ForwardContext::new(&config);
    let mut dense_cache = MultiLayerKVCache::new(&config);

    // Multi-position: a wiring bug in the KV-cache write shows up only once
    // attention has more than one position to attend over.
    for pos in 0..8 {
        let token = (pos * 5 + 3) % config.vocab_size;
        let bit_logits =
            forward_ternary(&mut bit_ctx, &ternary, &mut bit_cache, token, pos, &config).to_vec();
        let dense_logits = forward_llama(
            &mut dense_ctx,
            &dense,
            &mut dense_cache,
            token,
            pos,
            &config,
        )
        .to_vec();

        assert_eq!(bit_logits.len(), config.vocab_size);
        for (i, (b, d)) in bit_logits.iter().zip(&dense_logits).enumerate() {
            assert!(
                b.is_finite(),
                "ternary logit {i} at pos {pos} is not finite"
            );
            let denom = d.abs().max(1.0);
            let rel = (b - d).abs() / denom;
            assert!(
                rel < 1e-4,
                "pos {pos} logit {i}: ternary {b} vs dense {d} (rel {rel:e})"
            );
        }
    }
}

/// Logits stay finite and non-degenerate across a long decode — catches a
/// residual/normalization wiring bug that only compounds over positions.
#[test]
fn forward_ternary_multi_token_stable() {
    let (config, weights, _) = make_pair(7);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    for pos in 0..config.block_size {
        let token = pos % config.vocab_size;
        let logits = forward_ternary(&mut ctx, &weights, &mut cache, token, pos, &config);
        assert!(logits.iter().all(|l| l.is_finite()), "NaN/Inf at pos {pos}");
        let spread = logits.iter().fold(f32::MIN, |a, &b| a.max(b))
            - logits.iter().fold(f32::MAX, |a, &b| a.min(b));
        assert!(spread > 0.0, "logits collapsed to a constant at pos {pos}");
    }
}

/// `generate_ternary` runs prefill + decode without panicking and keeps the
/// prompt as a prefix.
#[test]
fn generate_ternary_extends_prompt() {
    let (config, weights, _) = make_pair(11);
    let mut rng = Rng::new(1234);
    let prompt = [1usize, 2, 3];

    let out = generate_ternary(&weights, &config, &mut rng, &prompt, 8);

    assert_eq!(&out[..prompt.len()], &prompt);
    assert!(out.len() > prompt.len(), "no tokens generated");
    assert!(out.iter().all(|&t| t < config.vocab_size));
}

// ── Weight-container accounting ─────────────────────────────────────────────

/// Every projection satisfies the bit-plane invariant, and the footprint is
/// the promised 2.125 bits/weight (2 bit-planes + one f16 per 128 weights).
#[test]
fn ternary_weights_invariants_and_footprint() {
    let (config, weights, _) = make_pair(3);

    assert!(
        weights.invariants_hold(),
        "pos_bits & neg_bits must be disjoint in every projection"
    );

    let params = weights.ternary_param_count();
    let n = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let kvd = config.n_kv_head * config.head_dim;
    let mlp = config.mlp_hidden;
    let per_layer = q_dim * n + 2 * (kvd * n) + n * q_dim + 2 * (mlp * n) + n * mlp;
    assert_eq!(params, config.n_layer * per_layer);

    // 2.125 bits/weight exactly — every `cols` here is a multiple of 128, so
    // there is no partial-block padding to account for.
    let bits_per_weight = weights.ternary_bytes() as f64 * 8.0 / params as f64;
    assert!(
        (bits_per_weight - 2.125).abs() < 1e-9,
        "expected 2.125 bits/weight, got {bits_per_weight}"
    );
}
