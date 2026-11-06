//! The sink-audit census on a DENSE transformer (riir-infer Issue 1180 T1,
//! the real ungated G2 arm for katgpt-rs Issue 981).
//!
//! The hybrid census (`deltanet::sink_audit`) captures post-RoPE Q rows and
//! re-scores them after the pass. The dense bodies (gemma-2 f16, llama)
//! already carry an off-hot-path re-scorer: [`AttnSpanProbe`] runs one extra
//! Q·K pass per armed row inside `attend_row`, with the forward's own scale
//! and softcap, so the distribution it reads is the one the forward USES.
//! Arming it on every row with a leading profile of width N gives the
//! census accumulator directly: `profile[k] = Σ_{layer, head, row ≥ k} p`.
//! [`finalize_profile`] turns it into the same [`SinkAuditProfile`] the
//! hybrid lane emits, so the verdict, `from_audit`, and the report bytes are
//! shared code.
//!
//! Every layer is scanned: these bodies are full causal attention at every
//! layer for `pos < block_size` (the gemma-2 f16 forward attends `0..=pos`
//! with no sliding window), and the census sizes `block_size` to N.

use katgpt_core::kv_sink_window::{SinkAuditProfile, finalize_profile};
use katgpt_transformer::MultiLayerKVCache;

use super::ForwardContext;
use super::attention_probe::AttnSpanProbe;
use crate::deltanet::sink_audit::{SinkCensus, mean_layer_max_v_norm};
use crate::types::{Config, kv_dim};

/// Run the census over the first `n_positions` of `tokens` through
/// `forward(ctx, cache, token, pos, config)` — one decode step of the
/// caller's dense body. The caller owns token 0 (prepend BOS where the
/// serving path does: the sink usually IS that token).
pub fn run_census_dense<F>(
    config: &Config,
    tokens: &[usize],
    n_positions: usize,
    mut forward: F,
) -> Result<SinkCensus, String>
where
    F: FnMut(&mut ForwardContext, &mut MultiLayerKVCache, usize, usize, &Config),
{
    if n_positions == 0 {
        return Err("census window must be positive".into());
    }
    if tokens.len() < n_positions {
        return Err(format!(
            "corpus encodes {} tokens < the census window {n_positions}",
            tokens.len()
        ));
    }
    // The Issue-929 bounded-cache lesson: size to the census, never the
    // model's advertised context.
    let mut census_config = config.clone();
    census_config.block_size = n_positions;

    let mut ctx = ForwardContext::new(&census_config);
    let mut probe = AttnSpanProbe::new(config.n_layer, config.n_head, n_positions)
        .with_leading_profile(n_positions);
    probe.armed = true;
    ctx.attn_probe = Some(probe);
    let mut cache = MultiLayerKVCache::new(&census_config);

    for (pos, &token) in tokens.iter().take(n_positions).enumerate() {
        if let Some(p) = ctx.attn_probe.as_mut() {
            p.rows += 1;
        }
        forward(&mut ctx, &mut cache, token, pos, &census_config);
    }

    let probe = ctx
        .attn_probe
        .take()
        .ok_or_else(|| "census probe vanished".to_string())?;
    if probe.rows != n_positions as u64 {
        return Err(format!(
            "probe saw {} rows, expected {n_positions}",
            probe.rows
        ));
    }
    // Loud-zero defence: a body that never reaches the probe leaves the
    // profile at 0 — a FALSE SinkFree. Each observed (layer, head, row) adds
    // a full softmax row (width N covers every key), so the total must be
    // layers·heads·N.
    let expected = (config.n_layer * config.n_head * n_positions) as f64;
    let total: f64 = probe.profile.iter().sum();
    if (total - expected).abs() > 1e-3 * expected {
        return Err(format!(
            "probe mass {total:.3} != layers·heads·N {expected} — the forward \
             did not route every layer through the probed attention"
        ));
    }
    let first_n_profile =
        finalize_profile(&probe.profile, config.n_layer, config.n_head, n_positions);
    let m_act = mean_layer_max_v_norm(
        cache.layers.iter().map(|l| l.value.as_slice()),
        n_positions,
        kv_dim(config),
    );
    Ok(SinkCensus {
        profile: SinkAuditProfile {
            f_attn_token0: first_n_profile[0],
            first_n_profile,
            m_act,
            layers_scanned: config.n_layer,
            heads_scanned: config.n_head,
            positions_scanned: n_positions,
        },
        n_positions,
        tokens_used: n_positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::attention_probe::ProbeShape;

    fn tiny_config() -> Config {
        Config {
            n_layer: 2,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 4,
            n_embd: 8,
            mlp_hidden: 16,
            vocab_size: 8,
            block_size: 64,
            ..Config::default()
        }
    }

    /// A stand-in forward that routes every layer through the probe with a
    /// ZERO query: each row is uniform over `0..=pos`, so position k's mean
    /// is `mean_{q ≥ k} 1/(q+1)` — the analytic profile.
    fn uniform_forward(
        ctx: &mut ForwardContext,
        _cache: &mut MultiLayerKVCache,
        _tok: usize,
        pos: usize,
        cfg: &Config,
    ) {
        let hd = cfg.head_dim;
        let kvd = cfg.n_kv_head * hd;
        let q = vec![0.0f32; cfg.n_head * hd];
        let keys = vec![0.0f32; (pos + 1) * kvd];
        let probe = ctx.attn_probe.as_mut().expect("armed");
        for layer in 0..cfg.n_layer {
            probe.observe_layer(
                layer,
                &q,
                &keys,
                ProbeShape {
                    n_head: cfg.n_head,
                    n_kv_head: cfg.n_kv_head,
                    kv_dim: kvd,
                    head_dim: hd,
                    t_n: pos + 1,
                    scale: 0.5,
                    softcap: 0.0,
                },
                None,
            );
        }
    }

    #[test]
    fn dense_census_matches_the_analytic_uniform_profile() {
        let cfg = tiny_config();
        let n = 16;
        let tokens: Vec<usize> = (0..n).collect();
        let c = run_census_dense(&cfg, &tokens, n, uniform_forward).expect("census");
        for (k, &m) in c.profile.first_n_profile.iter().enumerate() {
            let want: f64 = (k..n).map(|q| 1.0 / (q + 1) as f64).sum::<f64>() / (n - k) as f64;
            assert!((m - want).abs() < 1e-6, "k={k} got {m} want {want}");
        }
        assert_eq!(c.profile.layers_scanned, 2);
        assert_eq!(c.profile.positions_scanned, n);
    }

    #[test]
    fn dense_census_is_deterministic() {
        let cfg = tiny_config();
        let tokens: Vec<usize> = (0..12).collect();
        let a = run_census_dense(&cfg, &tokens, 12, uniform_forward).unwrap();
        let b = run_census_dense(&cfg, &tokens, 12, uniform_forward).unwrap();
        assert_eq!(a.profile.first_n_profile, b.profile.first_n_profile);
    }

    /// A body that never reaches the probe must refuse, not read SinkFree.
    #[test]
    fn unprobed_forward_refuses_loud() {
        let cfg = tiny_config();
        let tokens: Vec<usize> = (0..8).collect();
        let err = run_census_dense(&cfg, &tokens, 8, |_, _, _, _, _| {}).unwrap_err();
        assert!(err.contains("did not route"), "{err}");
    }

    #[test]
    fn short_corpus_refuses() {
        let cfg = tiny_config();
        assert!(run_census_dense(&cfg, &[0, 1], 8, uniform_forward).is_err());
    }
}
