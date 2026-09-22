    use super::*;
    use crate::deltanet::forward::HybridForwardScratch;
    use crate::deltanet::weights::QwenDeltaNetWeights;
    use crate::rope::RopeFreqTable;
    use crate::types::Config;
    use katgpt_core::gdn_tree_verify::{GdnTreeVerifier, build_topology};

    /// Build a chain tree topology: node 0 is root, node i's parent is i-1.
    fn chain_topology(t: usize, alpha: f32) -> TreeTopology {
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let alphas = vec![alpha; t];
        build_topology(&parents, &alphas)
    }

    /// Fill weights with deterministic non-zero values (LCG).
    fn fill_weights_seeded(weights: &mut QwenDeltaNetWeights, seed: u32) {
        let mut state = seed;
        let next = |state: &mut u32| -> f32 {
            *state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (*state as f32) / (u32::MAX as f32) * 0.1 - 0.05 // [-0.05, 0.05)
        };
        for layer in &mut weights.layers {
            for w in layer.in_proj_qkv.dense_data_mut() {
                *w = next(&mut state);
            }
            for w in layer.in_proj_a.dense_data_mut() {
                *w = next(&mut state);
            }
            for w in layer.in_proj_b.dense_data_mut() {
                *w = next(&mut state);
            }
            for w in layer.in_proj_z.dense_data_mut() {
                *w = next(&mut state);
            }
            for w in layer.out_proj.dense_data_mut() {
                *w = next(&mut state);
            }
            for w in &mut layer.conv1d_weight {
                *w = next(&mut state);
            }
            for w in &mut layer.a_log {
                *w = next(&mut state).abs() + 0.1; // ensure positive
            }
            for w in &mut layer.dt_bias {
                *w = next(&mut state);
            }
            layer.linear_norm.fill(1.0); // gamma = 1 for simplicity
            for w in layer.gate_proj.dense_data_mut() {
                *w = next(&mut state);
            }
            for w in layer.up_proj.dense_data_mut() {
                *w = next(&mut state);
            }
            for w in layer.down_proj.dense_data_mut() {
                *w = next(&mut state);
            }
            layer.input_norm.fill(1.0);
            layer.post_attn_norm.fill(1.0);
        }
        weights.final_norm.fill(1.0);
    }

    /// Run sequential forward for T tokens and return per-token logits.
    fn sequential_forward(
        weights: &QwenDeltaNetWeights,
        config: &Config,
        cache: &mut crate::deltanet::HybridCache,
        tokens: &[usize],
        start_pos: usize,
    ) -> Vec<Vec<f32>> {
        let n = config.n_embd;
        let v = config.vocab_size;
        let buf_size = n.max(v);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut scratch = HybridForwardScratch::new(config);
        let mut x = vec![0.0f32; buf_size];

        let mut results = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let logits = crate::deltanet::forward_qwen_deltanet(
                &mut x,
                weights,
                cache,
                token,
                pos,
                config,
                &mut scratch,
                &rope_freq,
            );
            results.push(logits.to_vec());
        }
        results.shrink_to_fit();
        results
    }

    /// Test: pure-DeltaNet chain tree (3 nodes) matches sequential forward.
    #[test]
    fn test_chain_matches_sequential_all_deltanet() {
        use crate::types::DeltaNetLayerType::*;
        let n_layer = 2;
        let layer_types = vec![DeltaNet; n_layer];
        let config = Config::qwen_deltanet(n_layer, layer_types.clone());

        // Use a smaller config for test speed
        let config = Config {
            n_embd: 128,
            n_head: 4,
            head_dim: 32,
            n_kv_head: 4,
            mlp_hidden: 256,
            vocab_size: 64,
            block_size: 512,
            deltanet_linear_head_dim: 32,
            deltanet_linear_n_heads: 4,
            deltanet_linear_n_value_heads: 4,
            deltanet_conv_kernel_size: 4,
            deltanet_state_dim: 4 * 32 * 32,
            ..config
        };

        let mut weights = QwenDeltaNetWeights::zeros(&config);
        fill_weights_seeded(&mut weights, 42);

        // Sequential forward: 3 tokens at positions 0, 1, 2
        let tokens = vec![1, 2, 3];
        let mut cache_seq = crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let seq_logits = sequential_forward(&weights, &config, &mut cache_seq, &tokens, 0);

        // Tree forward: 3-node chain
        let t = 3;
        let mut topo = chain_topology(t, 0.9);
        let mut cache_tree =
            crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut verifier = GdnTreeVerifier::new(t, 32, 32);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x_scratch = vec![0.0f32; config.n_embd.max(config.vocab_size)];

        let tree_logits = forward_tree_qwen_deltanet(
            &mut x_scratch,
            &weights,
            &mut cache_tree,
            &mut topo,
            &tokens,
            0,
            &config,
            &rope_freq,
            &mut verifier,
        );

        // Compare per-node logits (topo-indexed = original for a chain)
        let v = config.vocab_size;
        for k in 0..t {
            let max_diff: f32 = (0..v)
                .map(|i| {
                    (tree_logits[k * v + i] - seq_logits[k][i]).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "node {k}: max diff {max_diff} exceeds tolerance"
            );
        }
    }

    /// Test: hybrid model (mixed Attention + `DeltaNet`) chain tree matches sequential.
    #[test]
    fn test_hybrid_chain_matches_sequential() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let n_layer = layer_types.len();
        let config = Config::qwen_deltanet(n_layer, layer_types.clone());

        // Smaller config for test speed
        let config = Config {
            n_embd: 128,
            n_head: 4,
            head_dim: 32,
            n_kv_head: 4,
            mlp_hidden: 256,
            vocab_size: 64,
            block_size: 512,
            deltanet_linear_head_dim: 32,
            deltanet_linear_n_heads: 4,
            deltanet_linear_n_value_heads: 4,
            deltanet_conv_kernel_size: 4,
            deltanet_state_dim: 4 * 32 * 32,
            ..config
        };

        let mut weights = QwenDeltaNetWeights::zeros(&config);
        // Fill attention weights too
        let mut state = 99u32;
        let next = |state: &mut u32| -> f32 {
            *state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (*state as f32) / (u32::MAX as f32) * 0.1 - 0.05
        };
        for layer in &mut weights.layers {
            for w in layer.attn_wq.dense_data_mut() { *w = next(&mut state); }
            for w in layer.attn_wk.dense_data_mut() { *w = next(&mut state); }
            for w in layer.attn_wv.dense_data_mut() { *w = next(&mut state); }
            for w in layer.attn_wo.dense_data_mut() { *w = next(&mut state); }
        }
        fill_weights_seeded(&mut weights, 42);

        let tokens = vec![1, 2, 3];

        // Sequential
        let mut cache_seq = crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let seq_logits = sequential_forward(&weights, &config, &mut cache_seq, &tokens, 0);

        // Tree
        let t = 3;
        let mut topo = chain_topology(t, 0.9);
        let mut cache_tree =
            crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut verifier = GdnTreeVerifier::new(t, 32, 32);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x_scratch = vec![0.0f32; config.n_embd.max(config.vocab_size)];

        let tree_logits = forward_tree_qwen_deltanet(
            &mut x_scratch,
            &weights,
            &mut cache_tree,
            &mut topo,
            &tokens,
            0,
            &config,
            &rope_freq,
            &mut verifier,
        );

        let v = config.vocab_size;
        for k in 0..t {
            let max_diff: f32 = (0..v)
                .map(|i| (tree_logits[k * v + i] - seq_logits[k][i]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "hybrid node {k}: max diff {max_diff} exceeds tolerance"
            );
        }
    }

    /// Test: pure-DeltaNet branching tree (A→B→D, A→C) matches per-branch sequential.
    #[test]
    fn test_branching_matches_per_branch_all_deltanet() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, DeltaNet];
        let config = Config::qwen_deltanet(layer_types.len(), layer_types.clone());

        let config = Config {
            n_embd: 128,
            n_head: 4,
            head_dim: 32,
            n_kv_head: 4,
            mlp_hidden: 256,
            vocab_size: 64,
            block_size: 512,
            deltanet_linear_head_dim: 32,
            deltanet_linear_n_heads: 4,
            deltanet_linear_n_value_heads: 4,
            deltanet_conv_kernel_size: 4,
            deltanet_state_dim: 4 * 32 * 32,
            ..config
        };

        let mut weights = QwenDeltaNetWeights::zeros(&config);
        fill_weights_seeded(&mut weights, 77);

        // Branching tree: A(0)→B(1)→D(3), A(0)→C(2)
        // parents: [MAX, 0, 0, 1]
        let parents = [usize::MAX, 0, 0, 1];
        let alphas = [0.9f32; 4];
        let mut topo = build_topology(&parents, &alphas);
        let token_ids = [10usize, 20, 30, 40]; // A, B, C, D

        // Sequential per branch:
        // Branch 1: A→B→D (tokens 10, 20, 40)
        let mut cache_b1 = crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let seq_b1 = sequential_forward(&weights, &config, &mut cache_b1, &[10, 20, 40], 0);

        // Branch 2: A→C (tokens 10, 30) — fresh cache (A is recomputed)
        let mut cache_b2 = crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let seq_b2 = sequential_forward(&weights, &config, &mut cache_b2, &[10, 30], 0);

        // Tree forward
        let mut cache_tree =
            crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut verifier = GdnTreeVerifier::new(4, 32, 32);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x_scratch = vec![0.0f32; config.n_embd.max(config.vocab_size)];

        let tree_logits = forward_tree_qwen_deltanet(
            &mut x_scratch,
            &weights,
            &mut cache_tree,
            &mut topo,
            &token_ids,
            0,
            &config,
            &rope_freq,
            &mut verifier,
        );

        let v = config.vocab_size;
        // Map topo nodes to original: find A, B, C, D in topo order
        // topo_order maps topo_k → original. For this tree, root A is topo 0.
        // BFS: A(0), B(1), C(2), D(3) — but topo order depends on BFS queue order.
        // Let's just find by matching: topo node k's token = token_ids[topo.topo_order[k]]
        let find_topo = |token: usize| -> usize {
            (0..4).find(|&k| token_ids[topo.topo_order[k]] == token).unwrap()
        };
        let k_a = find_topo(10);
        let k_b = find_topo(20);
        let k_c = find_topo(30);
        let k_d = find_topo(40);

        // Branch 1: A (node 0 of branch), B (node 1), D (node 2)
        for (tree_k, seq_logits_k) in [(k_a, &seq_b1[0]), (k_b, &seq_b1[1]), (k_d, &seq_b1[2])] {
            let max_diff: f32 = (0..v)
                .map(|i| (tree_logits[tree_k * v + i] - seq_logits_k[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "branch1 tree_k={tree_k}: max diff {max_diff} exceeds tolerance"
            );
        }

        // Branch 2: A (node 0), C (node 1) — A should match branch 1's A
        for (tree_k, seq_logits_k) in [(k_a, &seq_b2[0]), (k_c, &seq_b2[1])] {
            let max_diff: f32 = (0..v)
                .map(|i| (tree_logits[tree_k * v + i] - seq_logits_k[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "branch2 tree_k={tree_k}: max diff {max_diff} exceeds tolerance"
            );
        }
    }

    // ── Commit path tests (T4.3c.2) ──

    /// Compare two `HybridCaches` for equality (`DeltaNet` recurrent state, `conv_state`,
    /// and KV cache) within a tolerance.
    fn assert_caches_match(a: &crate::deltanet::HybridCache, b: &crate::deltanet::HybridCache, tol: f32) {
        // DeltaNet recurrent state
        assert_eq!(
            a.deltanet_state.recurrent_states.len(),
            b.deltanet_state.recurrent_states.len(),
            "recurrent_states layer count mismatch"
        );
        for (i, (sa, sb)) in a
            .deltanet_state
            .recurrent_states
            .iter()
            .zip(b.deltanet_state.recurrent_states.iter())
            .enumerate()
        {
            assert_eq!(sa.len(), sb.len(), "layer {i}: recurrent state len mismatch");
            if sa.is_empty() {
                continue; // Attention layer
            }
            let max_diff: f32 = sa
                .iter()
                .zip(sb.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < tol,
                "layer {i}: recurrent state max diff {max_diff} exceeds tol {tol}"
            );
        }
        // Conv state
        for (i, (sa, sb)) in a
            .deltanet_state
            .conv_states
            .iter()
            .zip(b.deltanet_state.conv_states.iter())
            .enumerate()
        {
            assert_eq!(sa.len(), sb.len(), "layer {i}: conv state len mismatch");
            if sa.is_empty() {
                continue;
            }
            let max_diff: f32 = sa
                .iter()
                .zip(sb.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < tol,
                "layer {i}: conv state max diff {max_diff} exceeds tol {tol}"
            );
        }
        // KV cache
        assert_eq!(
            a.kv_cache.layers.len(),
            b.kv_cache.layers.len(),
            "kv_cache layer count mismatch"
        );
        for (i, (la, lb)) in a.kv_cache.layers.iter().zip(b.kv_cache.layers.iter()).enumerate() {
            assert_eq!(la.key.len(), lb.key.len(), "layer {i}: kv key len mismatch");
            assert_eq!(la.value.len(), lb.value.len(), "layer {i}: kv value len mismatch");
            let max_diff_k: f32 = la
                .key
                .iter()
                .zip(lb.key.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let max_diff_v: f32 = la
                .value
                .iter()
                .zip(lb.value.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff_k < tol,
                "layer {i}: kv key max diff {max_diff_k} exceeds tol {tol}"
            );
            assert!(
                max_diff_v < tol,
                "layer {i}: kv value max diff {max_diff_v} exceeds tol {tol}"
            );
        }
    }

    /// Test: commit after tree forward (pure-DeltaNet chain) produces the same
    /// recurrent state + conv state as pure sequential decode.
    #[test]
    fn test_commit_chain_matches_sequential_all_deltanet() {
        use crate::types::DeltaNetLayerType::*;
        let n_layer = 2;
        let layer_types = vec![DeltaNet; n_layer];
        let config = Config::qwen_deltanet(n_layer, layer_types.clone());
        let config = Config {
            n_embd: 128,
            n_head: 4,
            head_dim: 32,
            n_kv_head: 4,
            mlp_hidden: 256,
            vocab_size: 64,
            block_size: 512,
            deltanet_linear_head_dim: 32,
            deltanet_linear_n_heads: 4,
            deltanet_linear_n_value_heads: 4,
            deltanet_conv_kernel_size: 4,
            deltanet_state_dim: 4 * 32 * 32,
            ..config
        };

        let mut weights = QwenDeltaNetWeights::zeros(&config);
        fill_weights_seeded(&mut weights, 99);

        let tokens = vec![5, 6, 7];

        // Ground truth: pure sequential decode
        let mut cache_seq = crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let _ = sequential_forward(&weights, &config, &mut cache_seq, &tokens, 0);

        // Tree forward (read-only on recurrent state, then commit)
        let t = tokens.len();
        let mut topo = chain_topology(t, 0.9);
        let mut cache_tree =
            crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut verifier = GdnTreeVerifier::new(t, 32, 32);
        let mut x_scratch = vec![0.0f32; config.n_embd.max(config.vocab_size)];

        let _tree_logits = forward_tree_qwen_deltanet(
            &mut x_scratch,
            &weights,
            &mut cache_tree,
            &mut topo,
            &tokens,
            0,
            &config,
            &rope_freq,
            &mut verifier,
        );

        // Commit the accepted path (all tokens in the chain)
        let mut scratch_commit = HybridForwardScratch::new(&config);
        commit_tree_qwen_deltanet(
            &weights,
            &mut cache_tree,
            &tokens,
            0,
            &config,
            &mut scratch_commit,
            &rope_freq,
        );

        // Verify cache states match (DeltaNet recurrent + conv)
        assert_caches_match(&cache_seq, &cache_tree, 1e-5);
    }

    /// Test: commit after tree forward (hybrid DeltaNet+Attention chain) produces
    /// the same recurrent state, conv state, AND KV cache as pure sequential decode.
    #[test]
    fn test_commit_hybrid_chain_matches_sequential() {
        use crate::types::DeltaNetLayerType::*;
        // 4 layers: DeltaNet, Attention, DeltaNet, Attention
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let config = Config::qwen_deltanet(layer_types.len(), layer_types.clone());
        let config = Config {
            n_embd: 128,
            n_head: 4,
            head_dim: 32,
            n_kv_head: 4,
            mlp_hidden: 256,
            vocab_size: 64,
            block_size: 512,
            deltanet_linear_head_dim: 32,
            deltanet_linear_n_heads: 4,
            deltanet_linear_n_value_heads: 4,
            deltanet_conv_kernel_size: 4,
            deltanet_state_dim: 4 * 32 * 32,
            ..config
        };

        let mut weights = QwenDeltaNetWeights::zeros(&config);
        fill_weights_seeded(&mut weights, 123);

        let tokens = vec![11, 22, 33];

        // Ground truth: pure sequential decode
        let mut cache_seq = crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let _ = sequential_forward(&weights, &config, &mut cache_seq, &tokens, 0);

        // Tree forward (Attention layers leave garbage KV; DeltaNet state unchanged)
        let t = tokens.len();
        let mut topo = chain_topology(t, 0.9);
        let mut cache_tree =
            crate::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut verifier = GdnTreeVerifier::new(t, 32, 32);
        let mut x_scratch = vec![0.0f32; config.n_embd.max(config.vocab_size)];

        let _tree_logits = forward_tree_qwen_deltanet(
            &mut x_scratch,
            &weights,
            &mut cache_tree,
            &mut topo,
            &tokens,
            0,
            &config,
            &rope_freq,
            &mut verifier,
        );

        // Commit
        let mut scratch_commit = HybridForwardScratch::new(&config);
        commit_tree_qwen_deltanet(
            &weights,
            &mut cache_tree,
            &tokens,
            0,
            &config,
            &mut scratch_commit,
            &rope_freq,
        );

        // Verify ALL cache states match (DeltaNet recurrent + conv + KV cache)
        assert_caches_match(&cache_seq, &cache_tree, 1e-5);
    }

    // ── T4.3c.3: speculative step tests ──

    /// Build a small hybrid target config (matching the commit test dims).
    fn hybrid_test_config(layer_types: Vec<DeltaNetLayerType>) -> Config {
        use crate::types::DeltaNetLayerType::*;
        let _ = (DeltaNet, Attention); // silence unused import in closures
        let config = Config::qwen_deltanet(layer_types.len(), layer_types.clone());
        Config {
            n_embd: 128,
            n_head: 4,
            head_dim: 32,
            n_kv_head: 4,
            mlp_hidden: 256,
            vocab_size: 64,
            block_size: 512,
            deltanet_linear_head_dim: 32,
            deltanet_linear_n_heads: 4,
            deltanet_linear_n_value_heads: 4,
            deltanet_conv_kernel_size: 4,
            deltanet_state_dim: 4 * 32 * 32,
            ..config
        }
    }

    /// Build a standard-transformer drafter config with the same vocab as the
    /// target. The drafter can be small — it just needs to produce marginals.
    fn draft_test_config(vocab_size: usize) -> Config {
        let config = Config::micro();
        Config {
            vocab_size,
            draft_lookahead: 4,
            tree_budget: 16,
            block_size: 512,
            ..config
        }
    }

    /// Smoke test: the speculative step must return at least one accepted token.
    #[test]
    fn test_speculative_step_qwen_deltanet_returns_tokens() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet];
        let target_config = hybrid_test_config(layer_types.clone());
        let draft_config = draft_test_config(target_config.vocab_size);

        let mut rng = Rng::new(42);
        let draft_weights = TransformerWeights::new(&draft_config, &mut rng);
        let mut target_weights = QwenDeltaNetWeights::zeros(&target_config);
        fill_weights_seeded(&mut target_weights, 99);

        let mut draft_sctx = SpeculativeContext::new(&draft_config);
        let mut tree_builder = TreeBuilder::new(&draft_config);
        let mut target_cache = HybridCache::with_layer_types(&target_config, &layer_types);
        let hd = target_config.deltanet_linear_head_dim;
        let mut verifier = GdnTreeVerifier::new(64, hd, hd);
        let mut target_scratch = HybridForwardScratch::new(&target_config);
        let rope_freq = RopeFreqTable::new(target_config.rope_theta, target_config.head_dim);

        let (accepted, len) = speculative_step_qwen_deltanet_tree(
            &mut draft_sctx,
            &mut tree_builder,
            &draft_weights,
            &draft_config,
            &target_weights,
            &mut target_cache,
            &target_config,
            &mut verifier,
            &mut target_scratch,
            &rope_freq,
            1, // token must be < min(draft_vocab, target_vocab) = 64
            0,
            &mut rng,
        );

        assert!(!accepted.is_empty(), "must accept at least one token");
        assert_eq!(len, accepted.len());
    }

    /// Same seed must produce same accepted tokens.
    #[test]
    fn test_speculative_step_qwen_deltanet_deterministic() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention];
        let target_config = hybrid_test_config(layer_types.clone());
        let draft_config = draft_test_config(target_config.vocab_size);

        let run = || {
            let mut rng = Rng::new(42);
            let draft_weights = TransformerWeights::new(&draft_config, &mut rng);
            let mut target_weights = QwenDeltaNetWeights::zeros(&target_config);
            fill_weights_seeded(&mut target_weights, 77);

            let mut draft_sctx = SpeculativeContext::new(&draft_config);
            let mut tree_builder = TreeBuilder::new(&draft_config);
            let mut target_cache = HybridCache::with_layer_types(&target_config, &layer_types);
            let hd = target_config.deltanet_linear_head_dim;
            let mut verifier = GdnTreeVerifier::new(64, hd, hd);
            let mut target_scratch = HybridForwardScratch::new(&target_config);
            let rope_freq = RopeFreqTable::new(target_config.rope_theta, target_config.head_dim);

            let mut rng2 = Rng::new(42);
            speculative_step_qwen_deltanet_tree(
                &mut draft_sctx,
                &mut tree_builder,
                &draft_weights,
                &draft_config,
                &target_weights,
                &mut target_cache,
                &target_config,
                &mut verifier,
                &mut target_scratch,
                &rope_freq,
                1, // token must be < min(draft_vocab, target_vocab) = 64
                0,
                &mut rng2,
            )
        };

        let (a1, _) = run();
        let (a2, _) = run();
        assert_eq!(a1, a2, "same seed must produce same accepted tokens");
    }

    /// The accepted tokens must advance the target cache identically to a
    /// pure sequential decode of the same tokens. This is the end-to-end
    /// correctness guarantee: after the speculative step, the cache state
    /// must be bit-equivalent to `sequential_forward(accepted_tokens)`.
    #[test]
    fn test_speculative_step_cache_matches_sequential() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let target_config = hybrid_test_config(layer_types.clone());
        let draft_config = draft_test_config(target_config.vocab_size);

        let mut rng = Rng::new(42);
        let draft_weights = TransformerWeights::new(&draft_config, &mut rng);
        let mut target_weights = QwenDeltaNetWeights::zeros(&target_config);
        fill_weights_seeded(&mut target_weights, 55);

        // Run the speculative step (this commits accepted tokens into target_cache)
        let mut draft_sctx = SpeculativeContext::new(&draft_config);
        let mut tree_builder = TreeBuilder::new(&draft_config);
        let mut target_cache = HybridCache::with_layer_types(&target_config, &layer_types);
        let hd = target_config.deltanet_linear_head_dim;
        let mut verifier = GdnTreeVerifier::new(64, hd, hd);
        let mut target_scratch = HybridForwardScratch::new(&target_config);
        let rope_freq = RopeFreqTable::new(target_config.rope_theta, target_config.head_dim);

        let (accepted, _len) = speculative_step_qwen_deltanet_tree(
            &mut draft_sctx,
            &mut tree_builder,
            &draft_weights,
            &draft_config,
            &target_weights,
            &mut target_cache,
            &target_config,
            &mut verifier,
            &mut target_scratch,
            &rope_freq,
            1, // token must be < min(draft_vocab, target_vocab) = 64
            0,
            &mut rng,
        );

        assert!(!accepted.is_empty());

        // Ground truth: pure sequential decode of the same accepted tokens
        // from the same starting state.
        let mut cache_seq = HybridCache::with_layer_types(&target_config, &layer_types);
        let _ = sequential_forward(
            &target_weights,
            &target_config,
            &mut cache_seq,
            &accepted,
            0,
        );

        // The committed cache must match the sequential decode cache.
        // Tolerance 1e-5 — same as the T4.3c.2 commit tests.
        assert_caches_match(&cache_seq, &target_cache, 1e-5);
    }

    // ── Plan 434: Weaver call-site wiring tests ──

    /// Build a zero-weight `WeaverCorrector` whose K exceeds `vocab_size`, so
    /// `correct_marginals_with_scratch` takes the k > `vocab_size` early-return
    /// path → marginals unchanged. Mirrors the Plan 433 zero-weight test.
    #[cfg(feature = "weaver_runtime")]
    fn make_zero_weaver(
        n_embd: usize,
        vocab_size: usize,
        draft_lookahead: usize,
    ) -> (
        katgpt_speculative::weaver::WeaverCorrector,
        katgpt_speculative::weaver::WeaverScratch,
    ) {
        use katgpt_speculative::weaver::{WeaverConfig, WeaverCorrector, WeaverScratch, WeaverWeights};
        let weaver_cfg = WeaverConfig {
            hidden_dim: n_embd,
            n_heads: 4,
            k_candidates: vocab_size + 100, // K > V → early return, marginals unchanged
            n_layer: 1,
            d_ff: n_embd * 2,
            rms_eps: 1e-6,
            max_depth: draft_lookahead,
        };
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(weaver_cfg.clone()));
        let scratch = WeaverScratch::new(&weaver_cfg);
        (corrector, scratch)
    }

    /// T4: zero-weight Weaver (K > V early-return) must produce the same
    /// accepted tokens as the base path. Same seed, same inputs, same RNG
    /// path → deterministic match.
    #[cfg(feature = "weaver_runtime")]
    #[test]
    fn test_spec_step_deltanet_weaver_matches_base_zero_weight() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet];
        let target_config = hybrid_test_config(layer_types.clone());
        let draft_config = draft_test_config(target_config.vocab_size);
        let n_embd = target_config.n_embd;
        let vocab = target_config.vocab_size;

        let run = |use_weaver: bool| -> (Vec<usize>, usize) {
            let mut rng = Rng::new(42);
            let draft_weights = TransformerWeights::new(&draft_config, &mut rng);
            let mut target_weights = QwenDeltaNetWeights::zeros(&target_config);
            fill_weights_seeded(&mut target_weights, 99);

            let mut draft_sctx = SpeculativeContext::new(&draft_config);
            let mut tree_builder = TreeBuilder::new(&draft_config);
            let mut target_cache =
                HybridCache::with_layer_types(&target_config, &layer_types);
            let hd = target_config.deltanet_linear_head_dim;
            let mut verifier = GdnTreeVerifier::new(64, hd, hd);
            let mut target_scratch = HybridForwardScratch::new(&target_config);
            let rope_freq =
                RopeFreqTable::new(target_config.rope_theta, target_config.head_dim);
            let mut rng2 = Rng::new(42);

            if use_weaver {
                let (weaver, mut wscratch) = make_zero_weaver(
                    n_embd,
                    vocab,
                    draft_config.draft_lookahead,
                );
                let mut h_dflash_captured =
                    vec![0.0f32; draft_config.draft_lookahead * draft_config.n_embd];
                speculative_step_qwen_deltanet_tree_with_weaver(
                    &mut draft_sctx,
                    &mut tree_builder,
                    &draft_weights,
                    &draft_config,
                    &target_weights,
                    &mut target_cache,
                    &target_config,
                    &mut verifier,
                    &mut target_scratch,
                    &rope_freq,
                    1,
                    0,
                    &mut rng2,
                    &mut h_dflash_captured,
                    &weaver,
                    &mut wscratch,
                )
            } else {
                speculative_step_qwen_deltanet_tree(
                    &mut draft_sctx,
                    &mut tree_builder,
                    &draft_weights,
                    &draft_config,
                    &target_weights,
                    &mut target_cache,
                    &target_config,
                    &mut verifier,
                    &mut target_scratch,
                    &rope_freq,
                    1,
                    0,
                    &mut rng2,
                )
            }
        };

        let (accepted_base, len_base) = run(false);
        let (accepted_weaver, len_weaver) = run(true);
        assert_eq!(
            len_base, len_weaver,
            "zero-weight Weaver must not change accepted count"
        );
        assert_eq!(
            accepted_base, accepted_weaver,
            "zero-weight Weaver (K > V early-return) must produce bit-identical \
             accepted tokens to the base path"
        );
    }

    /// T5: cold-start (no prior commit → `target_scratch.hidden_copy` is zeros)
    /// must not panic and must produce at least one finite accepted token.
    #[cfg(feature = "weaver_runtime")]
    #[test]
    fn test_spec_step_deltanet_weaver_cold_start_no_panic() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention];
        let target_config = hybrid_test_config(layer_types.clone());
        let draft_config = draft_test_config(target_config.vocab_size);
        let n_embd = target_config.n_embd;
        let vocab = target_config.vocab_size;

        let mut rng = Rng::new(7);
        let draft_weights = TransformerWeights::new(&draft_config, &mut rng);
        let mut target_weights = QwenDeltaNetWeights::zeros(&target_config);
        fill_weights_seeded(&mut target_weights, 33);

        let mut draft_sctx = SpeculativeContext::new(&draft_config);
        let mut tree_builder = TreeBuilder::new(&draft_config);
        let mut target_cache = HybridCache::with_layer_types(&target_config, &layer_types);
        let hd = target_config.deltanet_linear_head_dim;
        let mut verifier = GdnTreeVerifier::new(64, hd, hd);
        // Fresh scratch — hidden_copy is zero-init (cold-start case).
        let mut target_scratch = HybridForwardScratch::new(&target_config);
        let rope_freq = RopeFreqTable::new(target_config.rope_theta, target_config.head_dim);

        let (weaver, mut wscratch) =
            make_zero_weaver(n_embd, vocab, draft_config.draft_lookahead);
        let mut h_dflash_captured =
            vec![0.0f32; draft_config.draft_lookahead * draft_config.n_embd];

        let (accepted, len) = speculative_step_qwen_deltanet_tree_with_weaver(
            &mut draft_sctx,
            &mut tree_builder,
            &draft_weights,
            &draft_config,
            &target_weights,
            &mut target_cache,
            &target_config,
            &mut verifier,
            &mut target_scratch,
            &rope_freq,
            2,
            0,
            &mut rng,
            &mut h_dflash_captured,
            &weaver,
            &mut wscratch,
        );

        assert!(!accepted.is_empty(), "must accept at least one token");
        assert_eq!(len, accepted.len());
        for &tok in &accepted {
            assert!(tok < vocab, "accepted token {tok} must be < vocab {vocab}");
        }
    }
