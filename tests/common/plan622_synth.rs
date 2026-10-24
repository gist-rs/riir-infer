//! Plan 622 shared synthetic-hybrid kit (T2.1's pruned-prefill G1 lane).
//!
//! Mirrors the in-file kit of `tests/plan622_keybias_walk_g1.rs` (the T1.3
//! walk): same `hybrid_config`, same `synth_hybrid_weights` construction
//! order and fixture laws (gate-range ×0.1, attn_wo ×0.01, `a_log = -0.5`,
//! dt_bias ×0.05), same SplitMix determinism. NOT yet consumed by the walk
//! file — its own copy stays until its lane touches it (avoiding cross-lane
//! test churn); if you edit a fixture law here, the walk file's copy is the
//! one to keep in sync.

#![allow(dead_code)]

use riir_infer_core::deltanet::weights::{DeltaNetLayerWeights, Proj, QwenDeltaNetWeights};
use riir_infer_core::types::{Config, DeltaNetLayerType};

// ─── Determinism: FNV-1a seeding + splitmix64 stream ───────────────────────

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// splitmix64 — a fixed seed rebuilds a cell byte-for-byte on any box.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let v = ((z ^ (z >> 31)) >> 40) as f32; // top 24 bits → [0, 2^24)
        v / 8_388_608.0 - 1.0 // [-1, 1)
    }

    pub fn stream_into(&mut self, dst: &mut [f32]) {
        for v in dst.iter_mut() {
            let x = self.next_f32();
            // No exact-zero elements: the -0.0 edge stays a named posture.
            *v = if x == 0.0 { 1.0e-3 } else { x };
        }
    }

    pub fn stream_vec(&mut self, len: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        self.stream_into(&mut v);
        v
    }

    /// RMSNorm gamma rows: near-1, never zero.
    pub fn gamma_vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| 1.0 + 0.05 * self.next_f32()).collect()
    }
}

// ─── The synthetic hybrid model ────────────────────────────────────────────

pub fn hybrid_config() -> Config {
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
        rope_dimension_count: 8, // partial RoPE — position errors are visible
        deltanet_linear_n_heads: 2,
        deltanet_linear_n_value_heads: 2,
        deltanet_linear_head_dim: 8,
        deltanet_conv_kernel_size: 4,
        ..Default::default()
    }
}

pub fn synth_hybrid_weights(cfg: &Config, seed: u64) -> QwenDeltaNetWeights {
    let mut sm = SplitMix(seed);
    let n = cfg.n_embd;
    let q_dim = cfg.n_head * cfg.head_dim;
    let kvd = cfg.n_kv_head * cfg.head_dim;
    let mlp = cfg.mlp_hidden;

    let attention_layer = |sm: &mut SplitMix| {
        let mut wq = sm.stream_vec(2 * q_dim * n);
        // Gate-range law: keep the gate half out of fast_sigmoid's exact-0/1
        // clamp so score movement propagates.
        let hd = cfg.head_dim;
        for block in (0..2 * cfg.n_head).step_by(2) {
            let gate_off = (block + 1) * hd * n;
            for v in &mut wq[gate_off..gate_off + hd * n] {
                *v *= 0.1;
            }
        }
        // Loop-gain law: trained-scale output projection.
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
        conv1d_weight: sm.stream_vec(qkv_dim * k),
        // Decay in (0, 1): a = exp(a_log) < 1 keeps the recurrence bounded.
        a_log: vec![-0.5; n_v],
        dt_bias: sm
            .stream_vec(n_k * 2 + n_v)
            .iter()
            .map(|x| x * 0.05)
            .collect(),
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
        layers: vec![
            deltanet_layer,
            attention_layer(&mut sm),
            attention_layer(&mut sm),
        ],
        layer_types,
    }
}

// ─── Shared assertions ─────────────────────────────────────────────────────

pub fn assert_bits_eq(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: elem {i} differs ({x:?} vs {y:?})"
        );
    }
}
