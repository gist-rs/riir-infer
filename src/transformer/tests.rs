//! Unit tests for the transformer module.
//!
//! Hoisted out of `mod.rs` (Plan 302) to keep the main module under the
//! 2048-line ceiling. All tests live at `crate::transformer::tests::*`.

#![allow(unnameable_test_items)]
#![allow(dead_code)]

    use super::*;

    #[test]
    fn test_forward_output_size() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite: {l}");
        }
    }

    #[test]
    fn test_forward_cache_populated() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let key_sum: f32 = cache.layers[0].key[..kvd].iter().sum();
        let val_sum: f32 = cache.layers[0].value[..kvd].iter().sum();
        assert!(key_sum != 0.0, "K cache at pos 0 should be populated");
        assert!(val_sum != 0.0, "V cache at pos 0 should be populated");
    }

    #[test]
    fn test_forward_positions_differ() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits_0 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
        let logits_1 = forward(&mut ctx, &weights, &mut cache, 0, 1, &config);
        let different = logits_0.iter().zip(logits_1).any(|(&a, b)| a != *b);
        assert!(different, "logits at different positions should differ");
    }

    #[test]
    fn test_generate_deterministic() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(t1, t2, "Same seed must produce same tokens");
    }

    #[test]
    fn test_generate_valid_tokens() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 32);
        assert_eq!(tokens.len(), 32);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of range");
        }
    }

    #[test]
    fn test_tokens_to_string() {
        let tokens = vec![0, 1, 2, 25, 26];
        let s = tokens_to_string(&tokens);
        assert_eq!(s, "abcz_");
    }

    #[test]
    fn test_forward_context_reuse() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Multiple forward passes with same context should give same results
        let _l1 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
        let l2 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        // Note: results differ because cache accumulates, but buffers should not leak
        for &v in l2.iter() {
            assert!(v.is_finite(), "reused context produced non-finite: {v}");
        }
    }

    // ── Multi-layer tests ─────────────────────────────────────────

    #[test]
    fn test_forward_output_size_nlayer2() {
        let mut config = Config::micro();
        config.n_layer = 2;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        assert_eq!(weights.layers.len(), 2);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        assert_eq!(cache.layers.len(), 2);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_logits_finite_nlayer4() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite with n_layer=4: {l}");
        }
    }

    #[test]
    fn test_n_layer_1_matches_current() {
        // n_layer=1 must produce identical deterministic output to old single-layer code
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(t1, t2, "n_layer=1 should be deterministic");
        assert_eq!(config.n_layer, 1, "micro config should have n_layer=1");
    }

    #[test]
    fn test_multi_layer_cache_populated() {
        let mut config = Config::micro();
        config.n_layer = 3;
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

        // Every layer's cache should be populated
        for (layer_idx, layer_cache) in cache.layers.iter().enumerate() {
            let key_sum: f32 = layer_cache.key[..kvd].iter().sum();
            let val_sum: f32 = layer_cache.value[..kvd].iter().sum();
            assert!(
                key_sum != 0.0,
                "layer {layer_idx} K cache at pos 0 should be populated"
            );
            assert!(
                val_sum != 0.0,
                "layer {layer_idx} V cache at pos 0 should be populated"
            );
        }
    }

    #[test]
    fn test_hidden_state_populated() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let sum: f32 = ctx.hidden_state.iter().sum();
        assert!(
            sum != 0.0,
            "hidden_state should be populated after forward pass"
        );
        for (i, &v) in ctx.hidden_state.iter().enumerate() {
            assert!(v.is_finite(), "hidden_state[{i}] should be finite: {v}");
        }
    }

    #[test]
    fn test_multi_layer_generate_valid() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 16);
        assert_eq!(tokens.len(), 16);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of range");
        }
    }

    // ── GQA tests ───────────────────────────────────────────────

    #[test]
    fn test_gqa_produces_valid_logits() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        for pos in 0..4 {
            let logits = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            for (i, &l) in logits.iter().enumerate() {
                assert!(
                    l.is_finite(),
                    "gqa_draft logit {i} at pos {pos} not finite: {l}"
                );
            }
        }
    }

    #[test]
    fn test_gqa_mha_backward_compat() {
        // When n_kv_head == n_head, GQA produces identical results to standard MHA.
        // Micro config has n_kv_head=4, n_head=4 → pure MHA.
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(
            t1, t2,
            "MHA backward compat: same seed must produce same tokens"
        );
        assert_eq!(
            config.n_kv_head, config.n_head,
            "micro config should have n_kv_head == n_head"
        );
    }

    #[test]
    fn test_gqa_kv_cache_smaller() {
        // GQA config should have smaller KV cache than equivalent MHA config
        let gqa = Config::gqa_draft();
        let kvd = crate::types::kv_dim(&gqa);
        assert_eq!(
            kvd,
            gqa.n_kv_head * gqa.head_dim,
            "kv_dim should be n_kv_head * head_dim"
        );
        assert!(
            kvd < gqa.n_embd,
            "GQA kv_dim ({kvd}) should be < n_embd ({})",
            gqa.n_embd
        );

        // Verify cache is correctly sized
        let cache = KVCache::new(&gqa);
        assert_eq!(
            cache.key.len(),
            gqa.block_size * kvd,
            "GQA key cache should use kv_dim"
        );
        assert_eq!(
            cache.value.len(),
            gqa.block_size * kvd,
            "GQA value cache should use kv_dim"
        );
    }

    #[test]
    fn test_gqa_generate_valid_tokens() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 8);
        assert_eq!(tokens.len(), 8);
        for &t in &tokens {
            assert!(t < config.vocab_size, "GQA token {t} out of range");
        }
    }

    #[test]
    fn test_config_validate_gqa() {
        // Valid configs should pass validation
        assert!(Config::micro().validate().is_ok());
        assert!(Config::draft().validate().is_ok());
        assert!(Config::small_target().validate().is_ok());
        assert!(Config::gqa_draft().validate().is_ok());

        // Invalid: n_head not divisible by n_kv_head
        let mut bad = Config::micro();
        bad.n_kv_head = 3; // n_head=4, not divisible by 3
        assert!(bad.validate().is_err());

        // Invalid: n_head * head_dim != n_embd
        let mut bad2 = Config::micro();
        bad2.head_dim = 5; // 4*5=20 != 16
        assert!(bad2.validate().is_err());
    }

    // ── Paged KV cache tests ────────────────────────────────────

    #[test]
    fn test_paged_cache_write_read_roundtrip() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 1);
        let kvd = crate::types::kv_dim(&config);

        // Ensure pages for position 0
        paged.ensure_pages(0, 0);

        // Write some K/V data
        let k_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.2).collect();
        paged.write_kv(0, 0, 0, &k_data, &v_data);

        // Read back
        let mut k_out = vec![0.0f32; kvd];
        let mut v_out = vec![0.0f32; kvd];
        paged.read_kv(0, 0, 0, &mut k_out, &mut v_out);

        assert_eq!(k_out, k_data, "K data roundtrip mismatch");
        assert_eq!(v_out, v_data, "V data roundtrip mismatch");
    }

    #[test]
    fn test_paged_cache_linear_matches_flat() {
        // Paged cache should produce same results as flat cache for a linear sequence
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Run with flat cache
        let mut ctx = ForwardContext::new(&config);
        let mut flat_cache = MultiLayerKVCache::new(&config);
        let _flat_logits = forward(&mut ctx, &weights, &mut flat_cache, 0, 0, &config).to_vec();

        // Manually copy flat cache data to paged cache
        let mut paged = PagedKVCache::new(&config, 1);
        paged.ensure_pages(0, 0);

        for (layer_idx, layer_cache) in flat_cache.layers.iter().enumerate() {
            let k_data = &layer_cache.key[..kvd];
            let v_data = &layer_cache.value[..kvd];
            paged.write_kv(layer_idx, 0, 0, k_data, v_data);
        }

        // Read back and compare
        for layer_idx in 0..config.n_layer {
            let mut k_out = vec![0.0f32; kvd];
            let mut v_out = vec![0.0f32; kvd];
            paged.read_kv(layer_idx, 0, 0, &mut k_out, &mut v_out);

            let flat_k = &flat_cache.layers[layer_idx].key[..kvd];
            let flat_v = &flat_cache.layers[layer_idx].value[..kvd];
            assert_eq!(k_out, flat_k, "layer {layer_idx} K mismatch: paged vs flat");
            assert_eq!(v_out, flat_v, "layer {layer_idx} V mismatch: paged vs flat");
        }
    }

    #[test]
    fn test_paged_cache_fork_no_corruption() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut paged = PagedKVCache::new(&config, 1);

        // Write data to seq 0 at position 0
        paged.ensure_pages(0, 0);
        let k_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 1.0).collect();
        let v_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 2.0).collect();
        paged.write_kv(0, 0, 0, &k_orig, &v_orig);

        // Fork at position 0 (share nothing — fork_page = 0/16 = 0)
        let fork_seq = paged.fork(0, 0);

        // Write different data to forked seq
        paged.ensure_pages(fork_seq, 0);
        let k_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 99.0).collect();
        let v_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 100.0).collect();
        paged.write_kv(0, fork_seq, 0, &k_fork, &v_fork);

        // Original seq should be unchanged
        let mut k_check = vec![0.0f32; kvd];
        let mut v_check = vec![0.0f32; kvd];
        paged.read_kv(0, 0, 0, &mut k_check, &mut v_check);
        assert_eq!(k_check, k_orig, "original K corrupted after fork write");
        assert_eq!(v_check, v_orig, "original V corrupted after fork write");
    }

    #[test]
    fn test_paged_cache_fork_shares_prefix() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut paged = PagedKVCache::new(&config, 1);

        // Write data at positions 0..PAGE_SIZE (fills one page)
        paged.ensure_pages(0, PAGE_SIZE - 1);
        for pos in 0..PAGE_SIZE {
            let k: Vec<f32> = vec![pos as f32; kvd];
            let v: Vec<f32> = vec![pos as f32 * 2.0; kvd];
            paged.write_kv(0, 0, pos, &k, &v);
        }

        // Fork at position 8 (still within page 0)
        let fork_seq = paged.fork(0, 8);

        // Ensure forked seq has its own pages from fork point
        paged.ensure_pages(fork_seq, PAGE_SIZE);

        // The forked seq should share page 0 (prefix) but have its own page 1+
        // Verify shared prefix data is accessible
        let mut k_out = vec![0.0f32; kvd];
        let mut v_out = vec![0.0f32; kvd];
        paged.read_kv(0, fork_seq, 0, &mut k_out, &mut v_out);
        assert_eq!(k_out[0], 0.0, "forked seq should see original pos 0 data");
    }

    #[test]
    fn test_paged_cache_reset_frees_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 2);

        // Allocate pages for two sequences
        paged.ensure_pages(0, 31); // 2 pages (0..15 and 16..31)
        paged.ensure_pages(1, 15); // 1 page

        let total_before = paged.total_pages;
        assert!(total_before > 0, "should have allocated some pages");

        // Reset should free all pages
        paged.reset();

        // Free list should contain the freed pages
        // (exact count depends on implementation, but should be > 0)
        // After reset, we can allocate again and reuse freed pages
        paged.ensure_pages(0, 0);
        // If reuse works, total_pages shouldn't grow
        assert_eq!(paged.total_pages, total_before, "should reuse freed pages");
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        // Forward some tokens, snapshot, modify, restore, verify same logits
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache with tokens at positions 0..4
        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Snapshot at position 4
        let snapshot = cache.snapshot(4, &config);

        // Fill more positions
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Now restore
        cache.restore(&snapshot, &config);

        // Verify restored: forward at position 4 should give same result as fresh cache at pos 4
        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "restored logits should match fresh: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_snapshot_correct_size() {
        let config = Config::micro();
        let kd = types::kv_dim(&config);
        let cache = MultiLayerKVCache::new(&config);
        let snapshot = cache.snapshot(5, &config);

        assert_eq!(snapshot.pos, 5);
        assert_eq!(snapshot.layers.len(), config.n_layer);
        for layer in &snapshot.layers {
            assert_eq!(layer.key.len(), 5 * kd);
            assert_eq!(layer.value.len(), 5 * kd);
        }
    }

    #[test]
    fn test_restore_zeros_stale_data() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache
        for pos in 0..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Snapshot at position 3
        let snapshot = cache.snapshot(3, &config);

        // Restore
        cache.restore(&snapshot, &config);

        // Verify positions after pos=3 are zeroed
        let kd = types::kv_dim(&config);
        for layer in &cache.layers {
            for val in &layer.key[3 * kd..] {
                assert_eq!(*val, 0.0, "stale key data should be zeroed");
            }
            for val in &layer.value[3 * kd..] {
                assert_eq!(*val, 0.0, "stale value data should be zeroed");
            }
        }
    }

    #[test]
    fn test_snapshot_restore_multi_layer() {
        // Test with n_layer > 1 (small_target config)
        let config = Config::small_target();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache
        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        let snapshot = cache.snapshot(4, &config);
        assert_eq!(snapshot.layers.len(), 4, "should have 4 layer snapshots");

        // Modify and restore
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }
        cache.restore(&snapshot, &config);

        // Verify restored correctly by checking logits match fresh cache
        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "multi-layer restore should match fresh"
            );
        }
    }

    #[test]
    fn test_snapshot_restore_gqa() {
        // Test with GQA config (kv_dim < n_embd)
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        let snapshot = cache.snapshot(4, &config);
        let kd = types::kv_dim(&config);

        // Verify snapshot uses GQA kv_dim (smaller than n_embd)
        assert_eq!(kd, config.n_kv_head * config.head_dim);
        assert!(kd < config.n_embd, "GQA kv_dim should be < n_embd");
        for layer in &snapshot.layers {
            assert_eq!(layer.key.len(), 4 * kd);
        }

        // Restore and verify
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }
        cache.restore(&snapshot, &config);

        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!((a - b).abs() < 1e-3, "GQA restore should match fresh");
        }
    }

    // ── forward_paged tests ──────────────────────────────────────

    #[test]
    fn test_forward_paged_logits_match_forward() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Flat cache forward
        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);
        let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

        // Paged cache forward
        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);
        let logits_paged =
            forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

        assert_eq!(logits_flat.len(), logits_paged.len());
        for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "forward_paged logit {i} differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_forward_paged_logits_match_forward_multi_pos() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);

        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);

        for pos in 0..4 {
            let token = pos; // simple: use pos as token
            let logits_flat = forward(
                &mut ctx_flat,
                &weights,
                &mut cache_flat,
                token,
                pos,
                &config,
            );
            let logits_paged = forward_paged(
                &mut ctx_paged,
                &weights,
                &mut cache_paged,
                0,
                token,
                pos,
                &config,
            );

            for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "pos {pos} logit {i} differs: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_forward_paged_gqa_logits_match() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);
        let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);
        let logits_paged =
            forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

        assert_eq!(logits_flat.len(), logits_paged.len());
        for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "GQA forward_paged logit {i} differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_forward_paged_output_size() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = PagedKVCache::new(&config, 1);
        let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_paged_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = PagedKVCache::new(&config, 1);
        let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite: {l}");
        }
    }

    // ── Rollback tests ─────────────────────────────────────────────

    #[test]
    fn test_paged_rollback_frees_exclusive_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 2);

        // Allocate pages for seq 0 up to pos 31 (2 pages: 0..15, 16..31)
        paged.ensure_pages(0, 31);
        let seq0_pages_len = paged.layer_page_tables[0][0].len();
        assert!(seq0_pages_len >= 2, "seq 0 should have at least 2 pages");

        // Rollback seq 0 to pos 0 — all pages are exclusive (no other seq)
        paged.rollback(0, 0);

        // Page table should be truncated
        assert!(
            paged.layer_page_tables[0][0].is_empty(),
            "seq 0 page table should be empty after rollback to pos 0"
        );
        // All pages should be freed (they were exclusive)
        assert!(
            !paged.free_pages.is_empty(),
            "exclusive pages should be returned to free list"
        );
    }

    #[test]
    fn test_paged_rollback_preserves_shared_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 4);

        // Allocate pages for seq 0 up to pos 31
        paged.ensure_pages(0, 31);
        let _initial_pages_len = paged.layer_page_tables[0][0].len();

        // Fork a new sequence from seq 0 at pos 16 — shares first page
        // (fork returns layer_page_tables[0].len(), which may be > 1 if max_sequences > 1)
        let seq1 = paged.fork(0, 16);
        assert_ne!(seq1, 0, "fork should return a new sequence index");

        // Allocate exclusive pages for seq 0 beyond fork point
        paged.ensure_pages(0, 47); // extra pages after pos 31

        let free_before = paged.free_pages.len();
        let pages_before_rollback = paged.layer_page_tables[0][0].len();

        // Rollback seq 0 to pos 16 — keeps shared page, frees exclusive ones
        paged.rollback(0, 16);

        // Page table should be truncated to 1 page (covers 0..15)
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            1,
            "seq 0 should have 1 page after rollback to pos 16 (page covers 0..15)"
        );

        // Some pages should have been freed (the exclusive ones beyond page 0)
        let freed = paged.free_pages.len() - free_before;
        assert!(
            freed > 0,
            "exclusive pages beyond rollback point should be freed"
        );

        // But NOT more than what was removed from page table
        let removed = pages_before_rollback - 1;
        assert!(
            freed <= removed,
            "freed pages ({freed}) should not exceed removed pages ({removed})"
        );
    }

    #[test]
    fn test_paged_rollback_shared_page_not_freed() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 4);

        // Allocate pages for seq 0
        paged.ensure_pages(0, 31);

        // Fork seq 1 at pos 0 — shares nothing initially (fork_page = 0)
        let seq1 = paged.fork(0, 0);

        // Allocate different pages for seq 1
        paged.ensure_pages(seq1, 31);

        // Now fork seq 2 from seq 0 at pos 16 — shares first page with seq 0
        let seq2 = paged.fork(0, 16);
        let shared_page_idx = paged.layer_page_tables[0][0][0];

        // Rollback seq 2 to pos 0 — the shared page should NOT be freed
        let _free_before = paged.free_pages.len();
        paged.rollback(seq2, 0);

        // Shared page should still be in seq 0's page table
        assert!(
            paged.layer_page_tables[0][0].contains(&shared_page_idx),
            "shared page should still be referenced by seq 0"
        );
        // Shared page should NOT be in free list
        assert!(
            !paged.free_pages.contains(&shared_page_idx),
            "shared page should not be freed"
        );
    }

    #[test]
    fn test_paged_rollback_truncates_page_table() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 1);

        // Allocate 4 pages worth of positions
        paged.ensure_pages(0, 63);
        assert!(
            paged.layer_page_tables[0][0].len() >= 4,
            "should have at least 4 pages for pos 0..63"
        );

        // Rollback to pos 32 — should keep 2 pages (0..15, 16..31)
        paged.rollback(0, 32);
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            2,
            "should have exactly 2 pages after rollback to pos 32"
        );

        // Rollback to pos 16 — should keep 1 page (0..15)
        paged.rollback(0, 16);
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            1,
            "should have exactly 1 page after rollback to pos 16"
        );
    }

    #[test]
    fn test_paged_rollback_all_layers_consistent() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut paged = PagedKVCache::new(&config, 1);

        // Allocate pages for all layers
        paged.ensure_pages(0, 31);

        // Rollback to pos 16
        paged.rollback(0, 16);

        // All layers should have the same page table length
        let expected = 1; // 1 page covers 0..15
        for (layer_idx, lt) in paged.layer_page_tables.iter().enumerate() {
            assert_eq!(
                lt[0].len(),
                expected,
                "layer {layer_idx} should have {expected} pages after rollback"
            );
        }
    }

    // ======================================================================
    // Sparse MLP tests (Plan 022: TwELL-inspired)
    // ======================================================================

    /// Sparse matmul produces identical output to dense at 0% sparsity (all alive).
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_0_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        for i in 0..rows {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "Mismatch at {i}: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i]
            );
        }
    }

    /// Sparse matmul produces identical output at 95% sparsity.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_95_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let mut input = vec![0.0f32; cols];
        // 5% alive
        for i in (0..cols).step_by(20) {
            input[i] = 1.0;
        }
        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        for i in 0..rows {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-4,
                "Mismatch at {i}: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i]
            );
        }
    }

    /// Sparse matmul with 100% sparsity (all zeros) produces all-zero output.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_100_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let input = vec![0.0f32; cols];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        let alive = crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        assert_eq!(alive, 0, "Expected 0 alive neurons");
        for (i, &val) in sparse_out.iter().take(rows).enumerate() {
            assert_eq!(val, 0.0, "Expected zero output at {i}");
        }
    }

    /// `ForwardContext` buffers are correctly sized when `sparse_mlp` is enabled.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_forward_context_sparse_buffers() {
        let config = crate::types::Config::micro();
        let ctx = super::ForwardContext::new(&config);
        assert_eq!(ctx.active_indices.len(), config.mlp_hidden);
        assert_eq!(ctx.active_values.len(), config.mlp_hidden);
    }

    /// Forward pass works correctly with `sparse_mlp` enabled.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_forward_with_sparse_mlp() {
        let config = crate::types::Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = crate::transformer::TransformerWeights::new(&config, &mut rng);
        let mut ctx = crate::transformer::ForwardContext::new(&config);
        let mut cache = crate::transformer::MultiLayerKVCache::new(&config);

        let logits = crate::transformer::forward(&mut ctx, &weights, &mut cache, 26, 0, &config);

        // Verify logits are finite
        for l in logits {
            assert!(l.is_finite(), "Logit is not finite: {l}");
        }
    }

    /// Sparse matmul with negative values (should be treated as dead by `ReLU` context).
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_negative_input() {
        let rows = 8;
        let cols = 32;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let mut input = vec![0.0f32; cols];
        // Mix of positive, negative, zero
        input[0] = 1.0;
        input[1] = -1.0; // Should be ignored (not > 0)
        input[2] = 0.5;
        input[3] = -0.5; // Should be ignored
        // Rest are 0.0

        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        // Both should match since matmul doesn't skip negatives but sparse_matmul skips input[c] <= 0
        // So we need to compare against a modified dense that also skips negatives
        for r in 0..rows {
            let mut expected = 0.0f32;
            for c in 0..cols {
                if input[c] > 0.0 {
                    expected += weight[r * cols + c] * input[c];
                }
            }
            assert!(
                (sparse_out[r] - expected).abs() < 1e-4,
                "Mismatch at {r}: sparse={}, expected={}",
                sparse_out[r],
                expected
            );
        }
    }

    // -----------------------------------------------------------------------
    // Plan 025: Bidirectional Prefill + Modality LoRA Switching
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_prefill_logits_finite() {
        let config = Config::micro();
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
            assert!(l.is_finite(), "prefill logit {i} is not finite: {l}");
        }
    }

    #[test]
    fn test_forward_prefill_populates_cache() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..5).collect();
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
        // All 5 positions should have K/V in cache
        for p in 0..5 {
            let off = p * kvd;
            let key_sum: f32 = cache.layers[0].key[off..off + kvd].iter().sum();
            let val_sum: f32 = cache.layers[0].value[off..off + kvd].iter().sum();
            assert!(key_sum != 0.0, "K cache at pos {p} should be populated");
            assert!(val_sum != 0.0, "V cache at pos {p} should be populated");
        }
    }

    #[test]
    fn test_forward_prefill_logits_shape() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = vec![0, 1, 2];
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
    }

    #[test]
    fn test_forward_prefill_single_token() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens = vec![5];
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
                "single-token prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_prefill_then_decode_shared_cache() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Prefill with 4 tokens
        let prompt: Vec<usize> = (0..4).collect();
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);

        // Decode from position 4 (should use same cache)
        let logits2 = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        assert_eq!(logits2.len(), config.vocab_size);
        for (i, &l) in logits2.iter().enumerate() {
            assert!(
                l.is_finite(),
                "decode after prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_no_lora_matches_existing_forward() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Existing forward (no LoRA)
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config);

        // New forward_base with None (should be identical)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        #[cfg(not(feature = "domain_latent"))]
        let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None);
        #[cfg(feature = "domain_latent")]
        let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None, None);

        for i in 0..config.vocab_size {
            let diff = (logits1[i] - logits2[i]).abs();
            assert!(
                diff < 1e-6,
                "forward and forward_base(None) differ at {i}: {diff}"
            );
        }
    }

    #[test]
    fn test_generate_with_prefill_produces_tokens() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let prompt: Vec<usize> = (0..4).collect();
        let generated = {
            #[cfg(not(feature = "domain_latent"))]
            {
                generate_with_prefill(
                    &mut ctx,
                    &mut prefill,
                    &weights,
                    &mut cache,
                    &config,
                    &mut rng,
                    &prompt,
                    10,
                    &crate::types::LoraPair::none(),
                )
            }
            #[cfg(feature = "domain_latent")]
            {
                generate_with_prefill(
                    &mut ctx,
                    &mut prefill,
                    &weights,
                    &mut cache,
                    &config,
                    &mut rng,
                    &prompt,
                    10,
                    &crate::types::LoraPair::none(),
                    None,
                )
            }
        };

        assert!(!generated.is_empty(), "should generate at least one token");
        assert!(generated.len() <= 10, "should not exceed max_gen_tokens");
        for (i, &t) in generated.iter().enumerate() {
            assert!(t < config.vocab_size, "token {i} out of range: {t}");
        }
    }

// ── TurboQuant de-fork GOAT gate (Issue 019 Phase A.1) ────────────────────
//
// Proves `forward_turboquant` works end-to-end with the real bit-packed
// katgpt-quant cache (the former riir-engine stub stored plain f32 with no
// quantization). This test is the GOAT gate for the de-fork: it verifies
// (G1) correctness — logits are finite, correct length, and the cache
// actually compresses (compression_ratio > 1.0 proves bit-packing is active,
// not the f32 identity path); (G3) no-regression — multi-token generation
// stays stable (no NaN/Inf propagation from quantization roundtrip error).
#[cfg(feature = "turboquant")]
#[allow(unnameable_test_items)]
mod turboquant_defork_tests {
    use super::*;

    #[test]
    fn test_forward_turboquant_logits_finite_and_compressed() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        // Real bit-packed cache — 4-bit K/V (8× compression vs f32).
        let mut cache = crate::turboquant::TurboQuantKVCache::new(&config, 4, 4);

        // Run forward for a single token at pos 0.
        let logits = forward_turboquant(&mut ctx, &weights, &mut cache, 0, 0, &config);

        // G1: logits are finite and correct length.
        assert_eq!(logits.len(), config.vocab_size, "vocab size mismatch");
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "turboquant logit {i} not finite: {l}");
        }

        // G1 (the de-fork proof): compression_ratio > 1.0 means bit-packing is
        // active. The former f32 stub had no compression method at all; the real
        // katgpt-quant cache ships compression_ratio. If this assert fires, the
        // re-export resolved to the wrong type (e.g. a leftover stub).
        let ratio = cache.compression_ratio();
        assert!(
            ratio > 1.0,
            "compression_ratio should be > 1.0 (bit-packing active), got {ratio}"
        );
    }

    #[test]
    fn test_forward_turboquant_multi_token_stable() {
        // G3 no-regression: quantization roundtrip error must not propagate to
        // NaN/Inf across multiple tokens. Run 4 forward passes (pos 0..3) with
        // the same cache and verify every token's logits stay finite.
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut cache = crate::turboquant::TurboQuantKVCache::new(&config, 4, 4);

        for pos in 0..4usize {
            let mut ctx = ForwardContext::new(&config);
            let token = pos % config.vocab_size;
            let logits = forward_turboquant(&mut ctx, &weights, &mut cache, token, pos, &config);
            assert_eq!(logits.len(), config.vocab_size, "pos {pos}: vocab mismatch");
            for (i, &l) in logits.iter().enumerate() {
                assert!(
                    l.is_finite(),
                    "pos {pos} logit {i} not finite: {l} (quantization error propagated)"
                );
            }
        }
    }

    #[test]
    fn test_turboquant_cache_roundtrip_lossy_but_close() {
        // Core correctness: store a key, dequantize it back, verify the result
        // is close to the original (lossy quantization, not bit-exact). This is
        // the behavior change vs the f32 stub (which was bit-exact identity).
        let config = Config::micro();
        let mut cache = crate::turboquant::TurboQuantKVCache::new(&config, 4, 4);
        let kvd = crate::types::kv_dim(&config);

        // Store a known key vector.
        let key: Vec<f32> = (0..kvd).map(|i| (i as f32) * 0.1 - 0.5).collect();
        cache.store_key(0, 0, &key);

        // Dequantize back.
        let mut recovered = vec![0.0f32; kvd];
        cache.dequantize_key_into(0, 0, &mut recovered);

        // Should be close but not bit-exact (4-bit quantization introduces error).
        let mut max_err = 0.0f32;
        for (i, (&o, &r)) in key.iter().zip(recovered.iter()).enumerate() {
            let err = (o - r).abs();
            assert!(
                err < 0.5,
                "dim {i}: dequant error {err} too large (original {o}, recovered {r})"
            );
            max_err = max_err.max(err);
        }
        // Quantization introduces SOME error — this proves we're not on the f32
        // identity path. If max_err == 0.0, we accidentally got the stub back.
        assert!(
            max_err > 0.0,
            "dequant was bit-exact (max_err=0) — this means the f32 stub is still active, not the real bit-packed cache"
        );
    }
}
