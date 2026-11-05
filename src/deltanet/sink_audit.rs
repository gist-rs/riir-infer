//! The sink-audit census lane (katgpt-rs Issue 981 T1/T2, `sink_audit`).
//!
//! One pass per checkpoint, offline: run the hybrid decode forward over the
//! first N tokens, capture each FULL-ATTENTION layer's post-RoPE Q rows at
//! the moment the layer runs, then harvest the per-head softmax mass those
//! queries put on the first N positions — through
//! [`katgpt_core::kv_sink_window`]'s audit core, which owns the profile, the
//! chosen cut points, and the policy selector.
//!
//! # Scope
//!
//! Full-attention layers ONLY ([`DeltaNetLayerType::Attention`]). GDN layers
//! have no softmax sink; a blended measurement dilutes the signal the verdict
//! reads.
//!
//! # Determinism (G1)
//!
//! The census drives the incumbent decode forward token-by-token (pos 0..N,
//! layers ascending), then harvests (layers ascending, heads ascending,
//! queries ascending) with sequential f32 rows accumulating into f64 — one
//! fixed order, no RNG, no wall clock. Same checkpoint + corpus + N → the
//! same profile bits (the bin pins this with a BLAKE3 sidecar).
//!
//! # Sync boundary
//!
//! None — an offline diagnostic. Nothing here is synced, and nothing here
//! runs in a serving loop (the capture rides an `Option` that is `None`
//! everywhere else, cfg-gated so a flag-off build compiles the field itself
//! out).
//!
//! # What the audit is NOT
//!
//! It does not gate the forward (Plan 287's classifier stays the in-engine
//! lane), and it does not touch the token-gate reserve (a correctness margin,
//! not an eviction budget — Issue 981 T2's exclusion, recorded with reason).

use crate::deltanet::forward::{HybridCache, HybridForwardScratch, forward_qwen_deltanet};
use crate::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use crate::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use crate::deltanet::weights::QwenDeltaNetWeights;
use crate::rope::RopeFreqTable;
use crate::types::{Config, DeltaNetLayerType};
use katgpt_core::kv_sink_window::{SinkAuditProfile, accumulate_head_mass_into, finalize_profile};

/// The Q-row capture riding [`HybridForwardScratch`] (the `scale_invariant`
/// precedent: session state beside the buffers the kernels already take).
///
/// Armed ONLY by the census pass; every other construction leaves it `None`,
/// and the field is cfg-gated so a flag-off build never carries it.
pub struct SinkAuditQCapture {
    /// Slot `s` holds `[N × q_dim]` post-RoPE Q rows for full-attention
    /// layer `attn_layers[s]` (parallel arrays — `q_rows.len() ==
    /// attn_layers.len()`).
    pub q_rows: Vec<Vec<f32>>,
    /// The `layer_idx` values that are full-attention layers, ascending.
    pub attn_layers: Vec<usize>,
    /// `layer_idx → slot` (None for GDN layers), built once.
    slot_of_layer: Vec<Option<usize>>,
    /// The position the caller is forwarding this step; the body's capture
    /// writes row `pos` of the matching layer.
    pub pos: usize,
    /// `N` — the capture width every row buffer was sized for.
    pub n_positions: usize,
}

impl SinkAuditQCapture {
    /// Size the capture for one census pass: `n_positions` rows per
    /// full-attention layer, `q_dim = n_head * head_dim` wide.
    pub fn new(config: &Config, layer_types: &[DeltaNetLayerType], n_positions: usize) -> Self {
        let q_dim = config.n_head * config.head_dim;
        let mut attn_layers = Vec::new();
        let mut slot_of_layer = vec![None; layer_types.len()];
        for (idx, lt) in layer_types.iter().enumerate() {
            if *lt == DeltaNetLayerType::Attention {
                slot_of_layer[idx] = Some(attn_layers.len());
                attn_layers.push(idx);
            }
        }
        let q_rows = attn_layers
            .iter()
            .map(|_| vec![0.0f32; n_positions * q_dim])
            .collect();
        Self {
            q_rows,
            attn_layers,
            slot_of_layer,
            pos: 0,
            n_positions,
        }
    }

    /// Capture one layer's Q rows for the current position. Called from the
    /// hybrid body AFTER [`forward_attention_layer`-equivalent] ran — the
    /// attention step reads `q_buf` last at step 5 and never writes it again,
    /// so the buffer still holds post-RoPE Q here.
    pub(crate) fn capture(&mut self, layer_idx: usize, q_buf: &[f32]) {
        if let Some(slot) = self.slot_of_layer[layer_idx] {
            let row_off = self.pos * q_buf.len();
            let dst = &mut self.q_rows[slot][row_off..row_off + q_buf.len()];
            dst.copy_from_slice(q_buf);
        }
    }
}

/// The census result: the audit profile plus the surface the claims are
/// scoped to.
#[derive(Debug)]
pub struct SinkCensus {
    pub profile: SinkAuditProfile,
    /// The census window N (== `profile.positions_scanned`).
    pub n_positions: usize,
    /// The tokens actually consumed (the first N of the encoded corpus).
    pub tokens_used: usize,
}

/// Run the census over the first `n_positions` tokens of `tokens`.
///
/// Drives the incumbent decode forward (no census-specific arithmetic in the
/// model), harvests the softmax mass per (full-attention layer, head, query),
/// and finalizes the audit profile. `m_act` is the mean per-layer max V-row
/// L2 norm over the window (the paper's secondary signal; disclosed, never
/// verdict-bearing — the fitted_v_gate norm class).
pub fn run_census(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    tokens: &[usize],
    rope_freq: &RopeFreqTable,
    n_positions: usize,
) -> Result<SinkCensus, String> {
    if tokens.len() < n_positions {
        return Err(format!(
            "corpus encodes {} tokens < the census window {n_positions}",
            tokens.len()
        ));
    }
    let census_config = census_posture(config, n_positions);
    let mut cache = HybridCache::with_layer_types_bounded(
        &census_config,
        &weights.layer_types,
        n_positions,
    );
    let mut scratch = HybridForwardScratch::new(&census_config);
    arm_capture(&mut scratch, config, &weights.layer_types, n_positions);

    let mut logits = vec![0.0f32; config.vocab_size];
    let tokens_used = n_positions;
    for (pos, &token) in tokens.iter().take(n_positions).enumerate() {
        set_capture_pos(&mut scratch, pos);
        forward_qwen_deltanet(
            &mut logits,
            weights,
            &mut cache,
            token,
            pos,
            &census_config,
            &mut scratch,
            rope_freq,
        );
    }
    finish_census(&mut scratch, &cache, &census_config, n_positions, tokens_used)
}

/// The TERNARY census (Issue 981 follow-up (b)): the same pass over the
/// flagship's serving body — `QwenDeltaNetTernaryWeights` +
/// [`forward_qwen_deltanet_ternary`]. The dense loader mis-shapes the PQ2_0
/// ternary file (in_proj_qkv row overrun), so the Bonsai-2-27B census MUST
/// run here. The capture seam, the harvest, and the profile arithmetic are
/// the shared code below — the ternary body writes f32 post-RoPE Q/K/V into
/// the same buffers the dense body does (the Issue-594 gate applies at the
/// output, after the attention read, exactly as the audit premise requires).
///
/// A folded (Hadamard) file needs the `bonsai2_hadamard` feature at the
/// RUN's build (the loader refuses loud otherwise); this lane itself is
/// rotation-agnostic.
pub fn run_census_ternary(
    weights: &QwenDeltaNetTernaryWeights,
    config: &Config,
    tokens: &[usize],
    rope_freq: &RopeFreqTable,
    n_positions: usize,
) -> Result<SinkCensus, String> {
    if tokens.len() < n_positions {
        return Err(format!(
            "corpus encodes {} tokens < the census window {n_positions}",
            tokens.len()
        ));
    }
    let census_config = census_posture(config, n_positions);
    let mut cache = HybridCache::with_layer_types_bounded(
        &census_config,
        &weights.layer_types,
        n_positions,
    );
    let mut scratch = HybridForwardScratch::new(&census_config);
    arm_capture(&mut scratch, config, &weights.layer_types, n_positions);

    let mut logits = vec![0.0f32; config.vocab_size];
    let tokens_used = n_positions;
    for (pos, &token) in tokens.iter().take(n_positions).enumerate() {
        set_capture_pos(&mut scratch, pos);
        forward_qwen_deltanet_ternary(
            &mut logits,
            weights,
            &mut cache,
            token,
            pos,
            &census_config,
            &mut scratch,
            rope_freq,
        );
    }
    finish_census(&mut scratch, &cache, &census_config, n_positions, tokens_used)
}

/// The census posture shared by both bodies: the cache is sized to exactly
/// the census window (the Issue-929 lesson: never the model's 128k-class
/// block_size — Bonsai-2 would demand >100 GiB).
fn census_posture(config: &Config, n_positions: usize) -> Config {
    let mut census_config = config.clone();
    census_config.block_size = n_positions;
    census_config
}

/// Arm the Q capture on the scratch (the census owns the only armed copy).
fn arm_capture(
    scratch: &mut HybridForwardScratch,
    config: &Config,
    layer_types: &[DeltaNetLayerType],
    n_positions: usize,
) {
    scratch.sink_audit_q = Some(SinkAuditQCapture::new(config, layer_types, n_positions));
}

fn set_capture_pos(scratch: &mut HybridForwardScratch, pos: usize) {
    if let Some(cap) = scratch.sink_audit_q.as_mut() {
        cap.pos = pos;
    }
}

/// Take the capture back out, run the shared harvest, and assemble the
/// [`SinkCensus`].
fn finish_census(
    scratch: &mut HybridForwardScratch,
    cache: &HybridCache,
    config: &Config,
    n_positions: usize,
    tokens_used: usize,
) -> Result<SinkCensus, String> {
    let capture = scratch
        .sink_audit_q
        .take()
        .ok_or_else(|| "census capture vanished".to_string())?;
    let profile = harvest_profile(cache, &capture, config, n_positions);
    Ok(SinkCensus {
        profile,
        n_positions,
        tokens_used,
    })
}

/// The shared harvest — identical for the dense and ternary bodies (both
/// write f32 post-RoPE K/V into the same `KVCache` layout). Sequential f32
/// rows accumulating into f64, one fixed order (layers ascending, heads
/// ascending, queries ascending); the audit core owns the arithmetic.
fn harvest_profile(
    cache: &HybridCache,
    capture: &SinkAuditQCapture,
    config: &Config,
    n_positions: usize,
) -> SinkAuditProfile {
    let q_dim = config.n_head * config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let head_dim = config.head_dim;
    let kv_per_head = n_head / n_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // ── Harvest: (layers ascending, heads ascending, queries ascending) ──
    let mut acc = vec![0.0f64; n_positions];
    let mut row: Vec<f32> = Vec::new();
    for (slot, &layer_idx) in capture.attn_layers.iter().enumerate() {
        let layer_cache = &cache.kv_cache.layers[layer_idx];
        let q_rows = &capture.q_rows[slot];
        for qi in 0..n_positions {
            for h in 0..n_head {
                let kvh = h / kv_per_head;
                let q_row = &q_rows[qi * q_dim..(qi + 1) * q_dim];
                let q_head = &q_row[h * head_dim..(h + 1) * head_dim];
                // This head's K rows for positions 0..=qi, strided by kvd.
                let k_base = layer_cache.key.as_slice();
                let mut head_keys = Vec::with_capacity((qi + 1) * head_dim);
                for ki in 0..=qi {
                    let off = ki * kvd + kvh * head_dim;
                    head_keys.extend_from_slice(&k_base[off..off + head_dim]);
                }
                accumulate_head_mass_into(
                    q_head,
                    &head_keys,
                    qi + 1,
                    head_dim,
                    scale,
                    &mut row,
                    &mut acc,
                );
            }
        }
    }
    let first_n_profile = finalize_profile(&acc, capture.attn_layers.len(), n_head, n_positions);

    // ── M-Act: mean per-layer max V-row L2 norm (disclosed, not verdict-bearing)
    let mut layer_maxes = Vec::with_capacity(capture.attn_layers.len());
    for &layer_idx in &capture.attn_layers {
        let v = cache.kv_cache.layers[layer_idx].value.as_slice();
        let mut max_sq = 0.0f64;
        for pos in 0..n_positions {
            let row_v = &v[pos * kvd..(pos + 1) * kvd];
            let mut sq = 0.0f64;
            for &x in row_v {
                sq += f64::from(x) * f64::from(x);
            }
            if sq > max_sq {
                max_sq = sq;
            }
        }
        layer_maxes.push(max_sq.sqrt());
    }
    let m_act = if layer_maxes.is_empty() {
        0.0
    } else {
        layer_maxes.iter().sum::<f64>() / layer_maxes.len() as f64
    };

    SinkAuditProfile {
        f_attn_token0: first_n_profile[0],
        first_n_profile,
        m_act,
        layers_scanned: capture.attn_layers.len(),
        heads_scanned: n_head,
        positions_scanned: n_positions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::weights::QwenDeltaNetWeights;
    use crate::types::DeltaNetLayerType;
    use katgpt_core::kv_sink_window::{SinkAuditVerdict, SinkWindowPolicy};

    /// A tiny hybrid config (4 layers: GDN, ATTN, GDN, ATTN) with dims small
    /// enough to run on a test box.
    fn tiny_config() -> Config {
        Config {
            n_layer: 4,
            n_embd: 64,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 16,
            mlp_hidden: 128,
            vocab_size: 128,
            block_size: 32,
            layer_types: vec![
                DeltaNetLayerType::DeltaNet,
                DeltaNetLayerType::Attention,
                DeltaNetLayerType::DeltaNet,
                DeltaNetLayerType::Attention,
            ],
            deltanet_linear_head_dim: 8,
            deltanet_linear_n_heads: 2,
            deltanet_linear_n_value_heads: 2,
            deltanet_conv_kernel_size: 4,
            ..Config::default()
        }
    }

    /// ALL-ZERO weights ⇒ every logit 0 ⇒ every softmax row uniform ⇒ the
    /// analytic profile: position k's mean mass is
    /// `(1/(N-k)) * Σ_{qi=k..N-1} 1/(qi+1)`, and m_act = 0 (zero V rows).
    #[test]
    fn census_on_zero_weights_matches_the_analytic_uniform_profile() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).collect();

        let census = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(census.n_positions, n);
        assert_eq!(census.profile.positions_scanned, n);
        // Scope: only the two ATTENTION layers, 4 heads.
        assert_eq!(census.profile.layers_scanned, 2);
        assert_eq!(census.profile.heads_scanned, 4);

        for (k, &mass) in census.profile.first_n_profile.iter().enumerate() {
            let want: f64 = (k..n).map(|qi| 1.0 / (qi + 1) as f64).sum::<f64>() / (n - k) as f64;
            assert!(
                (mass - want).abs() < 1e-6,
                "profile[{k}] = {mass} want {want}"
            );
        }
        assert!((census.profile.f_attn_token0 - census.profile.first_n_profile[0]).abs() < 1e-15);
        assert_eq!(census.profile.m_act, 0.0, "zero V rows carry zero norms");

        // A TINY window reads SinkHeavy by construction: uniform softmax
        // concentrates on early positions because every query sees few of
        // them — profile[0] = H(N)/N ≈ 0.34 at N=8, above the heavy cut. The
        // classifier is correct; the census window is what scales this away
        // (H(512)/512 ≈ 0.012 → SinkFree on a real window).
        assert_eq!(census.profile.verdict(), SinkAuditVerdict::SinkHeavy);
        let incumbent = SinkWindowPolicy::new(4, 256);
        assert_eq!(
            SinkWindowPolicy::from_audit(SinkAuditVerdict::SinkHeavy, &census.profile, incumbent),
            katgpt_core::kv_sink_window::AuditedSinkPolicy::KeepIncumbentSinkHeavy,
            "heavy keeps the incumbent pin"
        );
    }

    /// G1 at the lane level: the same census twice is bit-identical.
    #[test]
    fn census_is_deterministic() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % config.vocab_size).collect();

        let a = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        let b = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(a.profile.first_n_profile, b.profile.first_n_profile);
        assert_eq!(a.profile.f_attn_token0, b.profile.f_attn_token0);
        assert_eq!(a.profile.m_act, b.profile.m_act);
    }

    /// A corpus shorter than the window refuses loud, never silently scans.
    #[test]
    fn short_corpus_refuses() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let err = run_census(&weights, &config, &[1, 2, 3], &rope, 8).unwrap_err();
        assert!(err.contains("census window"), "{err}");
    }

    /// The TERNARY census on all-zero ternary weights meets the SAME analytic
    /// uniform profile as the dense census — the shared-harvest identity. The
    /// config is all-attention (GDN ternary projections demand 128-multiple
    /// group dims the tiny dense config cannot satisfy) with 128-multiple
    /// ternary dims (`QwenDeltaNetTernaryWeights::zeros`' contract).
    #[cfg(feature = "deltanet_ternary_inference")]
    fn tiny_ternary_config() -> Config {
        Config {
            n_layer: 4,
            n_embd: 128,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 64,
            mlp_hidden: 128,
            vocab_size: 128,
            block_size: 32,
            layer_types: vec![DeltaNetLayerType::Attention; 4],
            ..Config::default()
        }
    }

    #[cfg(feature = "deltanet_ternary_inference")]
    #[test]
    fn ternary_census_on_zero_weights_matches_the_analytic_uniform_profile() {
        let config = tiny_ternary_config();
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).collect();

        let census = run_census_ternary(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(census.n_positions, n);
        assert_eq!(census.profile.positions_scanned, n);
        // Scope: every layer is full-attention here — all four scanned.
        assert_eq!(census.profile.layers_scanned, 4);
        assert_eq!(census.profile.heads_scanned, 2);

        for (k, &mass) in census.profile.first_n_profile.iter().enumerate() {
            let want: f64 = (k..n).map(|qi| 1.0 / (qi + 1) as f64).sum::<f64>() / (n - k) as f64;
            assert!(
                (mass - want).abs() < 1e-6,
                "profile[{k}] = {mass} want {want}"
            );
        }
        assert!((census.profile.f_attn_token0 - census.profile.first_n_profile[0]).abs() < 1e-15);
        assert_eq!(census.profile.m_act, 0.0, "zero V rows carry zero norms");
        assert_eq!(census.profile.verdict(), SinkAuditVerdict::SinkHeavy);
    }

    /// G1 at the ternary lane level: the same census twice is bit-identical.
    #[cfg(feature = "deltanet_ternary_inference")]
    #[test]
    fn ternary_census_is_deterministic() {
        let config = tiny_ternary_config();
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % config.vocab_size).collect();

        let a = run_census_ternary(&weights, &config, &tokens, &rope, n).expect("census runs");
        let b = run_census_ternary(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(a.profile.first_n_profile, b.profile.first_n_profile);
        assert_eq!(a.profile.f_attn_token0, b.profile.f_attn_token0);
        assert_eq!(a.profile.m_act, b.profile.m_act);
    }
}
