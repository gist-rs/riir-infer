//! Extension tests: prefill, domain latent, shared KV, cluster LM head,
//! Gemma 2 block-causal.
//!
//! Hoisted out of `tests.rs` (Plan 302) to keep both files under the
//! 2048-line ceiling.

#![allow(unnameable_test_items)]
#![allow(dead_code)]

use super::*;

fn small_target_2layer() -> Config {
    let mut c = Config::small_target();
    c.n_layer = 2;
    c
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_generate_with_prefill_domain_latent() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);

    // Create a non-zero domain latent
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);

    let prompt: Vec<usize> = (0..4).collect();

    // Generate without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut prefill1 = PrefillContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let mut rng1 = Rng::new(42);
    let generated1 = generate_with_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights,
        &mut cache1,
        &config,
        &mut rng1,
        &prompt,
        10,
        &crate::types::LoraPair::none(),
        None,
    );

    // Generate with domain latent (same seed)
    let mut ctx2 = ForwardContext::new(&config);
    let mut prefill2 = PrefillContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let mut rng2 = Rng::new(42);
    let generated2 = generate_with_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights,
        &mut cache2,
        &config,
        &mut rng2,
        &prompt,
        10,
        &crate::types::LoraPair::none(),
        Some(&dl),
    );

    // Outputs should differ — domain latent modulates K/V at mid-layer
    assert_ne!(
        generated1, generated2,
        "domain latent should change generation output"
    );
}

#[test]
fn test_forward_prefill_multilayer_logits_finite() {
    let config = small_target_2layer();
    config.validate().unwrap();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens: Vec<usize> = (0..8).collect();
    #[cfg(not(feature = "domain_latent"))]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    assert_eq!(logits.len(), config.vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(
            l.is_finite(),
            "multilayer prefill logit {i} not finite: {l}"
        );
    }
}

#[test]
fn test_forward_prefill_multilayer_cache_populated() {
    let config = small_target_2layer();
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens: Vec<usize> = (0..4).collect();
    #[cfg(not(feature = "domain_latent"))]
    forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    // Both layers should have K/V populated
    for layer in 0..2 {
        for p in 0..4 {
            let off = p * kvd;
            let key_sum: f32 = cache.layers[layer].key[off..off + kvd].iter().sum();
            let val_sum: f32 = cache.layers[layer].value[off..off + kvd].iter().sum();
            assert!(
                key_sum != 0.0,
                "layer {layer} K cache at pos {p} should be populated"
            );
            assert!(
                val_sum != 0.0,
                "layer {layer} V cache at pos {p} should be populated"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Domain Latent injection (Plan 038)
// -----------------------------------------------------------------------

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_changes_logits() {
    let config = small_target_2layer(); // 2 layers, mid-layer = layer 1
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);

    // Without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

    // With domain latent (non-zero embedding)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        None,
        Some(&dl),
    );

    // Logits should differ — domain latent modulates K/V at mid-layer
    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff, "domain latent should change logits");
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_zero_embedding_same_logits() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);

    // Without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

    // With zero domain latent — should be identical
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let dl = crate::types::DomainLatent::zeros(kvd);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        None,
        Some(&dl),
    );

    for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "zero domain latent should not change logits, diff at {i}: {diff}"
        );
    }
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_prefill_changes_logits() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let tokens: Vec<usize> = (0..4).collect();

    // Without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut prefill1 = PrefillContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights,
        &mut cache1,
        &tokens,
        &config,
        None,
        None,
    );

    // With domain latent
    let mut ctx2 = ForwardContext::new(&config);
    let mut prefill2 = PrefillContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.3; kvd]);
    let logits2 = forward_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights,
        &mut cache2,
        &tokens,
        &config,
        None,
        Some(&dl),
    );

    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff, "domain latent in prefill should change logits");
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_prefill_then_decode() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.2; kvd]);

    // Prefill with domain latent
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let prompt: Vec<usize> = (0..3).collect();
    let logits_prefill = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &prompt,
        &config,
        None,
        Some(&dl),
    );
    assert_eq!(logits_prefill.len(), config.vocab_size);
    for (i, &l) in logits_prefill.iter().enumerate() {
        assert!(
            l.is_finite(),
            "prefill with domain_latent logit {i} not finite: {l}"
        );
    }

    // Decode with domain latent (position 3)
    let logits_decode = forward_base(
        &mut ctx,
        &weights,
        &mut cache,
        0,
        3,
        &config,
        None,
        Some(&dl),
    );
    assert_eq!(logits_decode.len(), config.vocab_size);
    for (i, &l) in logits_decode.iter().enumerate() {
        assert!(
            l.is_finite(),
            "decode after prefill with domain_latent logit {i} not finite: {l}"
        );
    }
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_forward_with_domain_latent_wrapper() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.1; kvd]);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let logits = forward_with_domain_latent(
        &mut ctx,
        &weights,
        &mut cache,
        0,
        0,
        &config,
        None,
        Some(&dl),
    );
    assert_eq!(logits.len(), config.vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit {i} not finite: {l}");
    }
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_with_lora_changes_logits() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let rank = 4;
    let in_dim = config.n_embd;
    let out_dim = config.n_embd;

    let lora = crate::types::LoraAdapter {
        a: vec![0.1f32; rank * in_dim],
        b: vec![0.1f32; out_dim * rank],
        rank,
        alpha: 8.0,
        in_dim,
        out_dim,
    };
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);

    // With both lora + domain_latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(
        &mut ctx1,
        &weights,
        &mut cache1,
        0,
        0,
        &config,
        Some(&lora),
        Some(&dl),
    );

    // With lora only (no domain_latent)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        Some(&lora),
        None,
    );

    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(
        any_diff,
        "domain_latent + lora should differ from lora-only"
    );
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_with_lora_prefill_pipeline() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let rank = 4;
    let in_dim = config.n_embd;
    let out_dim = config.n_embd;

    let lora = crate::types::LoraAdapter {
        a: vec![0.1f32; rank * in_dim],
        b: vec![0.1f32; out_dim * rank],
        rank,
        alpha: 8.0,
        in_dim,
        out_dim,
    };
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
    let tokens: Vec<usize> = (0..3).collect();

    // Pipeline 1: prefill + decode with both lora + dl
    let mut ctx1 = ForwardContext::new(&config);
    let mut prefill1 = PrefillContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let _ = forward_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights,
        &mut cache1,
        &tokens,
        &config,
        Some(&lora),
        Some(&dl),
    );
    let logits1 = forward_base(
        &mut ctx1,
        &weights,
        &mut cache1,
        0,
        tokens.len(),
        &config,
        Some(&lora),
        Some(&dl),
    );

    // Pipeline 2: prefill + decode with lora only
    let mut ctx2 = ForwardContext::new(&config);
    let mut prefill2 = PrefillContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let _ = forward_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights,
        &mut cache2,
        &tokens,
        &config,
        Some(&lora),
        None,
    );
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        tokens.len(),
        &config,
        Some(&lora),
        None,
    );

    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(
        any_diff,
        "prefill+decode with lora+dl should differ from lora-only pipeline"
    );
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_zero_with_lora_same_as_lora_only() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let rank = 4;
    let in_dim = config.n_embd;
    let out_dim = config.n_embd;

    let lora = crate::types::LoraAdapter {
        a: vec![0.1f32; rank * in_dim],
        b: vec![0.1f32; out_dim * rank],
        rank,
        alpha: 8.0,
        in_dim,
        out_dim,
    };
    let dl_zero = crate::types::DomainLatent::zeros(kvd);

    // With zero domain_latent + lora
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(
        &mut ctx1,
        &weights,
        &mut cache1,
        0,
        0,
        &config,
        Some(&lora),
        Some(&dl_zero),
    );

    // With lora only (no domain_latent)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        Some(&lora),
        None,
    );

    for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "zero domain_latent + lora should match lora-only, diff at {i}: {diff}"
        );
    }
}

// ── Shared KV Cache (Phase 3, Plan 055) ─────────────────────

#[test]
fn test_preload_kv_cache_dimension_mismatch() {
    // bpe: n_kv_head=4, head_dim=8 → kv_dim=32
    // bpe_draft: n_kv_head=2, head_dim=8 → kv_dim=16
    let target_config = Config::bpe();
    let draft_config = Config::bpe_draft();

    let target_cache = MultiLayerKVCache::new(&target_config);
    let mut draft_cache = MultiLayerKVCache::new(&draft_config);

    // Preload should silently skip (kv_dim mismatch)
    preload_kv_cache(
        &mut draft_cache,
        &target_cache,
        1,
        &target_config,
        &draft_config,
    );

    // Draft cache should remain all zeros
    for layer in &draft_cache.layers {
        assert!(
            layer.key.iter().all(|&v| v == 0.0),
            "draft cache key should remain zero on dim mismatch"
        );
        assert!(
            layer.value.iter().all(|&v| v == 0.0),
            "draft cache value should remain zero on dim mismatch"
        );
    }
}

#[test]
fn test_preload_kv_cache_matching_dims() {
    // Same config for both → kv_dim matches
    let config = Config::small_target();
    let kvd = crate::types::kv_dim(&config);

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Populate target cache at pos 0 and pos 1
    let mut target_cache = MultiLayerKVCache::new(&config);
    let mut target_ctx = ForwardContext::new(&config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 1, 1, &config);

    // Create empty draft cache
    let mut draft_cache = MultiLayerKVCache::new(&config);

    // Preload positions [0..2) from target
    preload_kv_cache(&mut draft_cache, &target_cache, 2, &config, &config);

    // Verify draft cache has target's KV for positions 0 and 1
    for (layer_idx, draft_layer) in draft_cache.layers.iter().enumerate() {
        let target_layer = &target_cache.layers[layer_idx];
        let copy_len = 2 * kvd;
        for i in 0..copy_len {
            assert_eq!(
                draft_layer.key[i], target_layer.key[i],
                "draft key mismatch at layer {layer_idx}, idx {i}"
            );
            assert_eq!(
                draft_layer.value[i], target_layer.value[i],
                "draft value mismatch at layer {layer_idx}, idx {i}"
            );
        }
    }
}

#[test]
fn test_preload_kv_cache_zero_pos() {
    let config = Config::small_target();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut target_cache = MultiLayerKVCache::new(&config);
    let mut target_ctx = ForwardContext::new(&config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);

    let mut draft_cache = MultiLayerKVCache::new(&config);

    // Preload with pos=0 copies nothing (no positions to share)
    preload_kv_cache(&mut draft_cache, &target_cache, 0, &config, &config);

    // Draft cache should remain all zeros
    for layer in &draft_cache.layers {
        assert!(
            layer.key.iter().all(|&v| v == 0.0),
            "draft cache should remain zero with pos=0"
        );
    }
}

#[test]
fn test_preload_kv_cache_fewer_draft_layers() {
    // Target: 2 layers, Draft: 1 layer — only layer 0 shared
    let target_config = Config {
        n_layer: 2,
        ..Config::small_target()
    };
    let draft_config = Config {
        n_layer: 1,
        ..Config::small_target()
    };

    let kvd = crate::types::kv_dim(&target_config);
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);

    let mut target_cache = MultiLayerKVCache::new(&target_config);
    let mut target_ctx = ForwardContext::new(&target_config);
    let _ = forward(
        &mut target_ctx,
        &target_weights,
        &mut target_cache,
        0,
        0,
        &target_config,
    );

    let mut draft_cache = MultiLayerKVCache::new(&draft_config);

    preload_kv_cache(
        &mut draft_cache,
        &target_cache,
        1,
        &target_config,
        &draft_config,
    );

    // Draft has 1 layer, only layer 0 should be copied
    assert_eq!(draft_cache.layers.len(), 1);
    let draft_layer = &draft_cache.layers[0];
    let target_layer = &target_cache.layers[0];
    for i in 0..kvd {
        assert_eq!(
            draft_layer.key[i], target_layer.key[i],
            "layer 0 key should be copied"
        );
        assert_eq!(
            draft_layer.value[i], target_layer.value[i],
            "layer 0 value should be copied"
        );
    }
}

/// T14: Verify hybrid behavior — drafter forwards with preloaded target KV.
/// Past positions [0..pos) read from preloaded target KV,
/// new position [pos] computed by drafter and written to its own cache.
#[test]
fn test_preload_kv_cache_hybrid_forward() {
    let config = Config::small_target();
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Build target KV cache for positions 0 and 1
    let mut target_cache = MultiLayerKVCache::new(&config);
    let mut target_ctx = ForwardContext::new(&config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 1, 1, &config);

    // Preload target KV [0..2) into draft cache
    let mut draft_cache = MultiLayerKVCache::new(&config);
    preload_kv_cache(&mut draft_cache, &target_cache, 2, &config, &config);

    // Drafter forwards at pos=2 with preloaded KV — should produce valid logits
    let mut draft_ctx = ForwardContext::new(&config);
    let logits = forward(&mut draft_ctx, &weights, &mut draft_cache, 2, 2, &config);

    // Logits must be finite (no NaN/Inf from garbage KV)
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] not finite: {v}");
    }

    // Draft cache now has: [0..2) from target, [2] from drafter
    for layer in &draft_cache.layers {
        // Position 2 should have non-zero KV (written by drafter)
        let pos2_off = 2 * kvd;
        let has_nonzero = layer.key[pos2_off..pos2_off + kvd]
            .iter()
            .any(|&v| v != 0.0);
        assert!(has_nonzero, "drafter should have written KV at pos 2");
    }
}

// --- T15–T19: Clustered LM Head Tests ---

#[test]
fn test_cluster_map_round_robin() {
    // 10 tokens, cluster_size=3 → 4 clusters: [0,1,2], [3,4,5], [6,7,8], [9]
    let map = cluster_map_round_robin(10, 3);
    assert_eq!(map.len(), 4);
    assert_eq!(map[0], vec![0, 1, 2]);
    assert_eq!(map[1], vec![3, 4, 5]);
    assert_eq!(map[2], vec![6, 7, 8]);
    assert_eq!(map[3], vec![9]);
}

#[test]
fn test_cluster_map_round_robin_exact_division() {
    // 8 tokens, cluster_size=4 → 2 clusters
    let map = cluster_map_round_robin(8, 4);
    assert_eq!(map.len(), 2);
    assert_eq!(map[0], vec![0, 1, 2, 3]);
    assert_eq!(map[1], vec![4, 5, 6, 7]);
}

#[test]
fn test_standard_lm_head_matches_matmul() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let n = config.n_embd;

    let mut logits_matmul = vec![0.0f32; config.vocab_size];
    let mut logits_standard = vec![0.0f32; config.vocab_size];
    let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Issue 019 Phase B.3: `standard_lm_head` was de-forked to the
    // `katgpt_forward` canonical, which uses `matmul_parallel` (auto-falls-
    // back to serial below the 512-row threshold; Config::micro() has
    // vocab_size=27 so the serial path is taken and drift is 0). The
    // reference here is `matmul_parallel` to match the canonical's actual
    // primitive — using bare `matmul` would test an implementation detail
    // (which matmul fn is selected) rather than behavior (correctness).
    types::matmul_parallel(
        &mut logits_matmul,
        &weights.lm_head,
        &hidden,
        config.vocab_size,
        n,
    );
    standard_lm_head(
        &mut logits_standard,
        &hidden,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    for i in 0..config.vocab_size {
        let diff = (logits_matmul[i] - logits_standard[i]).abs();
        assert!(diff < 1e-6, "standard_lm_head differs at {i}: {diff}");
    }
}

#[test]
fn test_clustered_lm_head_only_cluster_tokens_finite() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    let n = config.n_embd;
    let cluster_size = 16;

    let cluster_map = cluster_map_round_robin(config.vocab_size, cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * n).map(|_| rng.normal()).collect();

    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map.clone());

    let mut logits = vec![0.0f32; config.vocab_size];
    let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let num_clusters = cluster_map.len();
    let mut cluster_scores_buf = vec![0.0f32; num_clusters];
    let mut cluster_indexed_buf = Vec::with_capacity(num_clusters);
    let mut cluster_selected_buf = Vec::with_capacity(num_clusters);
    clustered_lm_head(
        &mut logits,
        &hidden,
        &weights.lm_head,
        weights.mtp_cluster_classifier.as_ref().unwrap(),
        weights.mtp_cluster_map.as_ref().unwrap(),
        config.vocab_size,
        n,
        1,
        &mut cluster_scores_buf,
        &mut cluster_indexed_buf,
        &mut cluster_selected_buf,
    );

    // Find winning cluster (the one with finite logits)
    let winning = cluster_map
        .iter()
        .find(|tokens| tokens.iter().all(|&t| logits[t].is_finite()))
        .expect("one cluster should have finite logits");

    // Cluster tokens: finite. Others: -inf
    let cluster_set: std::collections::HashSet<usize> = winning.iter().copied().collect();
    for (i, &logit) in logits.iter().enumerate() {
        if cluster_set.contains(&i) {
            assert!(logit.is_finite(), "token {i} in cluster should be finite");
        } else {
            assert_eq!(logit, f32::NEG_INFINITY, "token {i} should be -inf");
        }
    }
}

#[test]
fn test_clustered_lm_head_logits_match_standard() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    let n = config.n_embd;
    let cluster_size = 16;

    let cluster_map = cluster_map_round_robin(config.vocab_size, cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * n).map(|_| rng.normal()).collect();

    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map.clone());

    let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Standard logits
    let mut logits_std = vec![0.0f32; config.vocab_size];
    standard_lm_head(
        &mut logits_std,
        &hidden,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    // Clustered logits
    let mut logits_clust = vec![0.0f32; config.vocab_size];
    let num_clusters = cluster_map.len();
    let mut cluster_scores_buf = vec![0.0f32; num_clusters];
    let mut cluster_indexed_buf = Vec::with_capacity(num_clusters);
    let mut cluster_selected_buf = Vec::with_capacity(num_clusters);
    clustered_lm_head(
        &mut logits_clust,
        &hidden,
        &weights.lm_head,
        weights.mtp_cluster_classifier.as_ref().unwrap(),
        weights.mtp_cluster_map.as_ref().unwrap(),
        config.vocab_size,
        n,
        1,
        &mut cluster_scores_buf,
        &mut cluster_indexed_buf,
        &mut cluster_selected_buf,
    );

    // Find winning cluster
    let winning = cluster_map
        .iter()
        .find(|tokens| tokens.iter().all(|&t| logits_clust[t].is_finite()))
        .expect("one cluster should win");

    // Clustered logits for winning tokens should match standard exactly
    for &t in winning {
        let diff = (logits_clust[t] - logits_std[t]).abs();
        assert!(diff < 1e-5, "logit[{t}] mismatch: diff={diff}");
    }
}

#[test]
fn test_forward_base_clustered_dispatch() {
    // Config::bpe() has vocab=4096, threshold=4096 → 4096 >= 4096 activates
    // Use topk=1 so only 1 cluster is selected (produces -inf for non-cluster tokens)
    let mut config = Config::bpe();
    config.mtp_cluster_topk = 1;
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);

    let cluster_map = cluster_map_round_robin(config.vocab_size, config.mtp_cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * config.n_embd)
        .map(|_| rng.normal())
        .collect();
    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    // Clustered path active: some -inf, some finite
    let inf_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
    let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
    assert!(inf_count > 0, "should have -inf logits (clustered path)");
    assert!(
        finite_count > 0,
        "should have finite logits (cluster tokens)"
    );
    assert_eq!(inf_count + finite_count, config.vocab_size);
}

#[test]
fn test_forward_base_standard_fallback_no_weights() {
    // Config::micro() has threshold=usize::MAX → never activates clustered path
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    // Standard path: all finite, no -inf
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] should be finite: {v}");
    }
}

#[test]
fn test_cluster_map_from_embeddings_fallback() {
    let wte = vec![0.0f32; 100 * 32];
    let map = cluster_map_from_embeddings(&wte, 100, 32, 25);
    let expected = cluster_map_round_robin(100, 25);
    assert_eq!(map, expected);
}

// ── Gemma 2 Block-Causal Forward Tests (Plan 108 T1) ─────────────

/// Helper: micro Gemma2 config for block-causal testing.
#[cfg(feature = "dllm")]
fn gemma2_micro_dllm_config() -> Config {
    Config {
        model_arch: crate::types::ModelArchitecture::Gemma2,
        rms_norm_eps: 1e-6,
        rms_norm_offset: true,
        tied_embeddings: true,
        use_rope: true,
        rope_theta: 10000.0,
        post_norm: true,
        attn_logit_softcapping: 50.0,
        final_logit_softcapping: 30.0,
        d2f_block_size: 4,
        attention_mode: crate::types::AttentionMode::Bidirectional,
        ..Config::micro_dllm()
    }
}

/// Helper: create random Gemma2 weights for testing.
#[cfg(feature = "dllm")]
fn gemma2_test_weights(
    config: &Config,
    rng: &mut Rng,
) -> crate::gemma_layer::GemmaTransformerWeights {
    use crate::gemma_layer::GemmaLayerWeights;

    let n = config.n_embd;
    let kv_dim = config.n_kv_head * config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let mlp = config.mlp_hidden;
    let vocab = config.vocab_size;
    let scale = 0.01;

    let wte: Vec<f32> = (0..vocab * n).map(|_| rng.normal() * scale).collect();
    let final_norm: Vec<f32> = vec![1.0; n]; // gamma=1 (identity after +1 offset)

    let layers: Vec<_> = (0..config.n_layer)
        .map(|_| GemmaLayerWeights {
            attn_wq: (0..q_dim * n).map(|_| rng.normal() * scale).collect(),
            attn_wk: (0..kv_dim * n).map(|_| rng.normal() * scale).collect(),
            attn_wv: (0..kv_dim * n).map(|_| rng.normal() * scale).collect(),
            attn_wo: (0..n * q_dim).map(|_| rng.normal() * scale).collect(),
            gate_proj: (0..mlp * n).map(|_| rng.normal() * scale).collect(),
            up_proj: (0..mlp * n).map(|_| rng.normal() * scale).collect(),
            down_proj: (0..n * mlp).map(|_| rng.normal() * scale).collect(),
            input_norm: vec![1.0; n],
            post_attn_norm: vec![1.0; n],
            pre_mlp_norm: vec![1.0; n],
            post_mlp_norm: vec![1.0; n],
        })
        .collect();

    crate::gemma_layer::GemmaTransformerWeights {
        wte,
        final_norm,
        layers,
        #[cfg(feature = "delta_routing")]
        delta_routing_query: vec![vec![0.0f32; config.n_embd]; config.n_layer],
        #[cfg(feature = "delta_routing")]
        delta_routing_norm: vec![vec![1.0f32; config.n_embd]; config.n_layer],
    }
}

#[cfg(feature = "dllm")]
#[test]
fn test_forward_gemma2_block_causal_logits_finite() {
    let config = gemma2_micro_dllm_config();
    let mut rng = Rng::new(42);
    let weights = gemma2_test_weights(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);

    let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let prompt_len = 4;
    let block_size = 2;
    let mut all_logits = vec![0.0f32; tokens.len() * config.vocab_size];

    forward_gemma2_block_causal(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        prompt_len,
        block_size,
        &config,
        &mut all_logits,
    );

    // All logits should be finite
    for (i, &v) in all_logits.iter().enumerate() {
        assert!(v.is_finite(), "all_logits[{i}] should be finite: {v}");
    }

    // Last position's logits (returned slice content) should be finite
    let last_off = (tokens.len() - 1) * config.vocab_size;
    let result = &all_logits[last_off..last_off + config.vocab_size];
    assert_eq!(result.len(), config.vocab_size);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "result[{i}] should be finite: {v}");
    }
}

#[cfg(feature = "dllm")]
#[test]
fn test_forward_gemma2_block_causal_prompt_attends_all() {
    let config = gemma2_micro_dllm_config();
    let mut rng = Rng::new(42);
    let weights = gemma2_test_weights(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);

    // 8 tokens: 4 prompt + 4 generation
    let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let prompt_len = 4;
    let block_size = 2;

    let mut all_logits = vec![0.0f32; tokens.len() * config.vocab_size];

    forward_gemma2_block_causal(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        prompt_len,
        block_size,
        &config,
        &mut all_logits,
    );

    // Verify block_causal_t_n gives prompt_len for prompt positions
    for p in 0..prompt_len {
        let t_n = block_causal_t_n(p, prompt_len, block_size, tokens.len());
        assert_eq!(
            t_n, prompt_len,
            "prompt position {p} should attend to all {prompt_len} prompt tokens, got {t_n}"
        );
    }

    // Verify prompt logits differ from generation logits
    // (prompt positions attend to more tokens → different outputs)
    let prompt_logits_start = 0;
    let gen_logits_start = prompt_len * config.vocab_size;
    let mut prompt_differs = false;
    for i in 0..config.vocab_size {
        if (all_logits[prompt_logits_start + i] - all_logits[gen_logits_start + i]).abs() > 1e-6 {
            prompt_differs = true;
            break;
        }
    }
    assert!(prompt_differs, "prompt and generation logits should differ");
}

#[cfg(feature = "dllm")]
#[test]
fn test_forward_gemma2_block_causal_shape() {
    let config = gemma2_micro_dllm_config();
    let mut rng = Rng::new(42);
    let weights = gemma2_test_weights(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);

    let tokens = vec![0, 1, 2, 3];
    let prompt_len = 2;
    let block_size = 2;
    let mut all_logits = vec![0.0f32; tokens.len() * config.vocab_size];

    forward_gemma2_block_causal(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        prompt_len,
        block_size,
        &config,
        &mut all_logits,
    );

    // all_logits should have seq_len * vocab_size entries
    assert_eq!(all_logits.len(), tokens.len() * config.vocab_size);

    // Last position's logits should be finite
    let last_off = (tokens.len() - 1) * config.vocab_size;
    let result = &all_logits[last_off..last_off + config.vocab_size];
    assert_eq!(result.len(), config.vocab_size);
    assert!(result.iter().all(|v| v.is_finite()));
}
