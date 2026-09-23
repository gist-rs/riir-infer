    use super::*;
    use crate::test_gpu_support::{gpu_release_pages, heavy_model_test_gate};

    // ── Config tests ───────────────────────────────────────────────

    #[test]
    fn test_d2f_config_defaults() {
        let config = Gemma2D2fConfig::default();
        assert_eq!(config.block_size, 16);
        assert_eq!(config.denoise_steps, 8);
        assert!((config.confidence_threshold - 0.7).abs() < 1e-5);
        assert!((config.temperature - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_d2f_config_presets() {
        let quality = Gemma2D2fConfig::quality();
        assert_eq!(quality.denoise_steps, 12);
        assert!((quality.confidence_threshold - 0.8).abs() < 1e-5);

        let speed = Gemma2D2fConfig::speed();
        assert_eq!(speed.denoise_steps, 4);
        assert!((speed.confidence_threshold - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_d2f_config_with_block_size() {
        let config = Gemma2D2fConfig::default().with_block_size(32);
        assert_eq!(config.block_size, 32);
        assert_eq!(config.denoise_steps, 8);
    }

    #[test]
    fn test_d2f_config_clone() {
        let config = Gemma2D2fConfig::default();
        let mut cloned = config;
        cloned.block_size = 32;
        assert_eq!(config.block_size, 16);
        assert_eq!(cloned.block_size, 32);
    }

    // ── Block state tests ───────────────────────────────────────────

    #[test]
    fn test_block_state_transitions() {
        let fully = D2fBlockState::FullyActivated;
        assert!(fully.is_fully_activated());

        let semi = D2fBlockState::SemiActivated {
            step: 3,
            confidence: 0.5,
        };
        assert!(!semi.is_fully_activated());
    }

    // ── Block-causal params tests ───────────────────────────────────

    #[test]
    fn test_block_causal_params_default() {
        let params = AttentionBlockCausalParams::default();
        assert_eq!(params.n_head, 8);
        assert_eq!(params.n_kv_head, 4);
        assert_eq!(params.head_dim, 256);
        assert_eq!(params.block_size, 16);
        assert!((params.softcap - 50.0).abs() < 1e-6);
        assert!((params.scale - 0.0625).abs() < 1e-6);
    }

    #[test]
    fn test_block_causal_t_n() {
        let mut params = AttentionBlockCausalParams {
            pos: 0,
            prompt_len: 4,
            n_positions: 8,
            ..Default::default()
        };

        // Prompt position: attends to all prompt positions.
        assert_eq!(params.t_n(), 4);

        params.pos = 3;
        assert_eq!(params.t_n(), 4);

        // Generation position in first block.
        params.pos = 4;
        params.block_size = 2;
        assert_eq!(params.t_n(), 6); // prompt(4) + block(2) = 6

        params.pos = 5;
        assert_eq!(params.t_n(), 6); // Same block.

        // Generation position in second block.
        params.pos = 6;
        assert_eq!(params.t_n(), 8); // prompt(4) + 2*block(2) = 8
    }

    // ── Softmax tests ───────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1.0, got {sum}"
        );
        for (i, &p) in logits.iter().enumerate() {
            assert!(p > 0.0, "prob[{i}] should be positive, got {p}");
        }
    }

    #[test]
    fn test_softmax_edge_cases() {
        // Empty input.
        let mut empty: Vec<f32> = vec![];
        softmax(&mut empty);

        // Single element.
        let mut single = vec![5.0f32];
        softmax(&mut single);
        assert!(
            (single[0] - 1.0).abs() < 1e-5,
            "single element softmax should be 1.0"
        );

        // All equal.
        let mut equal = vec![1.0f32, 1.0, 1.0];
        softmax(&mut equal);
        for (i, &p) in equal.iter().enumerate() {
            assert!(
                (p - 1.0 / 3.0).abs() < 1e-5,
                "equal logits should give uniform probs, prob[{i}] = {p}"
            );
        }

        // Large values (numerical stability).
        let mut large = vec![10000.0f32, 10001.0, 10002.0];
        softmax(&mut large);
        let sum: f32 = large.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "large value softmax should sum to 1.0, got {sum}"
        );
    }

    // ── Sampling tests ──────────────────────────────────────────────

    #[test]
    fn test_sample_from_probs_valid() {
        let mut rng = fastrand::Rng::new();
        let probs = vec![0.1f32, 0.2, 0.3, 0.4];
        for _ in 0..100 {
            let idx = sample_from_probs(&probs, &mut rng);
            assert!(idx < probs.len(), "sampled index {idx} out of range");
        }
    }

    #[test]
    fn test_sample_from_probs_distribution() {
        let mut rng = fastrand::Rng::new();
        let probs = vec![0.0f32, 0.0, 1.0, 0.0]; // All mass on index 2.
        for _ in 0..50 {
            let idx = sample_from_probs(&probs, &mut rng);
            assert_eq!(idx, 2, "should always sample index 2");
        }
    }

    #[test]
    fn test_sample_from_probs_single_element() {
        let mut rng = fastrand::Rng::new();
        let probs = vec![1.0f32];
        for _ in 0..20 {
            let idx = sample_from_probs(&probs, &mut rng);
            assert_eq!(idx, 0, "should always sample index 0");
        }
    }

    // ── Confidence + sampling integration ───────────────────────────

    #[test]
    fn test_sample_with_confidence_dominant_token() {
        let mut rng = fastrand::Rng::new();
        let mut logits = vec![0.0f32; 10];
        logits[3] = 10.0;

        let mut scratch = Vec::with_capacity(logits.len());
        let (token, confidence) = sample_with_confidence(&logits, 1.0, 999, &mut rng, &mut scratch);
        assert_eq!(token, 3, "should sample highest logit token");
        assert!(
            confidence > 0.99,
            "confidence should be high, got {confidence}"
        );
    }

    #[test]
    fn test_sample_with_confidence_mask_suppression() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut logits = vec![0.0f32; 10];
        logits[5] = 10.0;
        logits[3] = 20.0; // Dominant after mask suppression (exp(20) >> exp(10)).

        let mut scratch = Vec::with_capacity(logits.len());
        let (token, _) = sample_with_confidence(&logits, 1.0, 5, &mut rng, &mut scratch);
        assert_ne!(token, 5, "should not sample mask token");
        assert_eq!(token, 3, "should sample next highest token");
    }

    #[test]
    fn test_sample_with_confidence_temperature() {
        let mut rng = fastrand::Rng::new();
        let logits = vec![0.0f32, 5.0f32];

        // High temperature → more uniform.
        let mut count_0 = 0usize;
        let mut scratch = Vec::with_capacity(logits.len());
        for _ in 0..1000 {
            let (token, _) = sample_with_confidence(&logits, 10.0, 999, &mut rng, &mut scratch);
            if token == 0 {
                count_0 += 1;
            }
        }
        assert!(
            count_0 > 20 && count_0 < 500,
            "high temperature should spread probability, got {count_0}/1000 for token 0"
        );
    }

    #[test]
    fn test_sample_with_confidence_uniform_logits() {
        let mut rng = fastrand::Rng::new();
        let uniform = vec![1.0f32; 5];

        let mut scratch = Vec::with_capacity(uniform.len());
        let (_, confidence) = sample_with_confidence(&uniform, 1.0, 999, &mut rng, &mut scratch);
        assert!(
            (confidence - 0.2).abs() < 0.01,
            "uniform logits should give ~0.2 confidence, got {confidence}"
        );
    }

    // ── GPU Integration tests ───────────────────────────────────────
    // These tests require GPU runtime and Config::gemma2_2b() dimensions.
    // Run with: cargo test -p riir-gpu --features gemma2_d2f -- --nocapture

    /// Create minimal test weights with Gemma2 2B correct shapes but random data.
    /// NOT numerically correct — just for structure/binding validation.
    /// NOTE: Creates ~2.3 GB of weight data (256K × 2304 vocab embedding).
    fn create_test_weights(config: &Config) -> GemmaTransformerWeights {
        use riir_infer_core::gemma_layer::GemmaLayerWeights;
        use riir_infer_core::types::Rng;

        let mut rng = Rng::new(42);
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kv_dim = config.n_kv_head * config.head_dim;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;
        let scale = 0.01;

        let wte: Vec<f32> = (0..vocab * n).map(|_| rng.normal() * scale).collect();
        let final_norm: Vec<f32> = vec![1.0; n];

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

        GemmaTransformerWeights {
            wte,
            final_norm,
            layers,
            #[cfg(feature = "delta_routing")]
            delta_routing_query: vec![vec![0.0f32; config.n_embd]; config.n_layer],
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: vec![vec![1.0f32; config.n_embd]; config.n_layer],
        }
    }

    /// Verify block_causal_forward produces correct output shape [seq_len][vocab_size].
    ///
    /// Uses Gemma 2 2B config (requires ~2.3 GB test weights).
    /// Run with: cargo test -p riir-gpu --features gemma2_d2f test_block_causal_forward_shape
    #[test]
    fn test_block_causal_forward_shape() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let d2f_config = Gemma2D2fConfig {
            block_size: 4,
            ..Default::default()
        };

        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, d2f_config);

        // 3 prompt tokens + 4 generation tokens = 7 total.
        let tokens: Vec<usize> = vec![42, 7, 123, 1, 2, 3, 4];
        let prompt_len = 3;

        let logits = d2f.block_causal_forward(&tokens, prompt_len);

        // Verify output shape: [seq_len][vocab_size].
        assert_eq!(logits.len(), tokens.len(), "seq_len mismatch");
        for (i, pos_logits) in logits.iter().enumerate() {
            assert_eq!(
                pos_logits.len(),
                config.vocab_size,
                "vocab_size mismatch at pos {i}"
            );
        }

        // Verify logits are finite (not NaN or Inf).
        for (i, pos_logits) in logits.iter().enumerate() {
            let n_nan = pos_logits.iter().filter(|x| x.is_nan()).count();
            let n_inf = pos_logits.iter().filter(|x| x.is_infinite()).count();
            assert_eq!(n_nan, 0, "NaN logits at pos {i}");
            assert_eq!(n_inf, 0, "Inf logits at pos {i}");
        }

        println!(
            "block_causal_forward: {} positions × {} vocab = OK",
            tokens.len(),
            config.vocab_size,
        );

        drop(d2f);
        gpu_release_pages(&client);
    }

    /// Verify block_causal_forward with prompt-only (no generation tokens).
    #[test]
    fn test_block_causal_forward_prompt_only() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let d2f_config = Gemma2D2fConfig {
            block_size: 4,
            ..Default::default()
        };

        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, d2f_config);

        let tokens: Vec<usize> = vec![10, 20, 30];
        let prompt_len = 3;

        let logits = d2f.block_causal_forward(&tokens, prompt_len);

        assert_eq!(logits.len(), 3);
        for (i, pos_logits) in logits.iter().enumerate() {
            assert_eq!(
                pos_logits.len(),
                config.vocab_size,
                "vocab mismatch at pos {i}"
            );
        }

        println!(
            "block_causal_forward (prompt-only): 3 positions × {} vocab = OK",
            config.vocab_size
        );

        drop(d2f);
        gpu_release_pages(&client);
    }

    /// Verify KV cache reset: two identical calls produce identical results.
    #[test]
    fn test_block_causal_forward_cache_reset() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let d2f_config = Gemma2D2fConfig {
            block_size: 2,
            ..Default::default()
        };

        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, d2f_config);

        let tokens: Vec<usize> = vec![1, 2, 3, 4];
        let prompt_len = 2;

        let logits1 = d2f.block_causal_forward(&tokens, prompt_len);
        let logits2 = d2f.block_causal_forward(&tokens, prompt_len);

        for pos in 0..tokens.len() {
            let max_diff = logits1[pos]
                .iter()
                .zip(logits2[pos].iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "Cache reset failure at pos {pos}: max_diff={max_diff:.8}"
            );
        }

        println!("block_causal_forward cache reset: identical results across 2 calls ✓");

        drop(d2f);
        gpu_release_pages(&client);
    }

    /// Verify different tokens produce different logits.
    #[test]
    fn test_block_causal_forward_token_dependent() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let d2f_config = Gemma2D2fConfig {
            block_size: 2,
            ..Default::default()
        };

        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, d2f_config);

        let tokens_a: Vec<usize> = vec![1, 2];
        let tokens_b: Vec<usize> = vec![99, 100];
        let prompt_len = 1;

        let logits_a = d2f.block_causal_forward(&tokens_a, prompt_len);
        let logits_b = d2f.block_causal_forward(&tokens_b, prompt_len);

        // Position 0 logits should differ (different input tokens).
        let max_diff = logits_a[0]
            .iter()
            .zip(logits_b[0].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Token-dependent output expected: max_diff={max_diff:.8}"
        );

        println!("block_causal_forward token-dependent: max_diff={max_diff:.6} ✓");

        drop(d2f);
        gpu_release_pages(&client);
    }

    // ── D2F Decode Loop Integration Tests (Plan 108 T4) ────────────

    /// D2F decode produces non-mask tokens with correct shape.
    ///
    /// Uses Gemma 2 2B config (requires ~2.3 GB test weights + GPU runtime).
    /// Run with: cargo test -p riir-gpu --features gemma2_d2f test_d2f_decode_produces_valid_tokens
    #[test]
    fn test_d2f_decode_produces_valid_tokens() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let decode_config = Gemma2D2fConfig {
            block_size: 4,
            denoise_steps: 8,
            confidence_threshold: 0.1, // Low threshold for random model.
            temperature: 1.0,
            sampler: None,
            sc_config: D2fScConfig::default(),
        };

        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
        let mut rng = fastrand::Rng::with_seed(42);

        let prompt = vec![42, 7, 123];
        let mask_token_id = 0; // Use 0 as mask (avoid vocab_size-1 which may be valid).

        let result = d2f_decode_gemma2(&mut d2f, &prompt, mask_token_id, &decode_config, &mut rng);

        // Result should have prompt + block_size tokens.
        assert_eq!(
            result.tokens.len(),
            prompt.len() + decode_config.block_size,
            "should have prompt + block tokens"
        );

        // Prompt tokens should be unchanged.
        for (i, &prompt_tok) in prompt.iter().enumerate() {
            assert_eq!(
                result.tokens[i], prompt_tok,
                "prompt token at pos {i} should be unchanged"
            );
        }

        // Non-prompt tokens should be valid vocab indices.
        let vocab = config.vocab_size;
        for pos in prompt.len()..result.tokens.len() {
            assert!(
                result.tokens[pos] < vocab,
                "token at pos {pos} should be valid vocab index, got {}",
                result.tokens[pos]
            );
        }

        println!(
            "d2f_decode: {} tokens, {} steps, converged={}",
            result.tokens.len(),
            result.steps_used,
            result.converged
        );

        drop(d2f);
        gpu_release_pages(&client);
    }

    /// D2F decode converges (all positions unmasked) within denoise_steps
    /// when confidence threshold is low enough.
    ///
    /// Uses Gemma 2 2B config (requires ~2.3 GB test weights + GPU runtime).
    /// Run with: cargo test -p riir-gpu --features gemma2_d2f test_d2f_decode_converges
    #[test]
    fn test_d2f_decode_converges() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let decode_config = Gemma2D2fConfig {
            block_size: 4,
            denoise_steps: 20,          // Plenty of steps.
            confidence_threshold: 0.01, // Very low threshold → always unmask.
            temperature: 1.0,
            sampler: None,
            sc_config: D2fScConfig::default(),
        };

        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
        let mut rng = fastrand::Rng::with_seed(42);

        let mask_token_id = 0;
        let prompt = vec![42, 7];
        let result = d2f_decode_gemma2(&mut d2f, &prompt, mask_token_id, &decode_config, &mut rng);

        assert!(
            result.converged,
            "should converge with very low threshold, steps_used={}, state={:?}",
            result.steps_used, result.state
        );
        assert_eq!(
            result.state,
            D2fBlockState::FullyActivated,
            "state should be FullyActivated"
        );
        assert!(
            result.steps_used <= decode_config.denoise_steps,
            "steps_used should not exceed max"
        );

        // No token should be the mask token (all unmasked).
        for (pos, &tok) in result.tokens.iter().enumerate() {
            assert_ne!(
                tok, mask_token_id,
                "position {pos} should not be mask token"
            );
        }

        // Diagnostics should be populated.
        assert!(result.steps_used > 0, "should use at least 1 step");
        assert_eq!(
            result.confidence.len(),
            result.tokens.len(),
            "confidence should have entry per token"
        );
        assert_eq!(
            result.confidence_history.len(),
            result.steps_used,
            "confidence_history should have entry per step"
        );

        println!(
            "d2f_decode_converges: {} steps, confidence_history={:?}",
            result.steps_used, result.confidence_history
        );

        drop(d2f);
        gpu_release_pages(&client);
    }

    // ── DiffusionSampler Tests (Plan 108 T5) ──────────────────────

    #[test]
    fn test_sampler_features_from_logits() {
        // Peaked distribution: token 0 dominates.
        let logits = vec![10.0, 1.0, 0.5, 0.1];
        let features = SamplerFeatures::from_logits(&logits, 2, 8, 3, 16, 999);

        assert!(
            features.top1_prob > 0.9,
            "top1_prob should be dominant: {}",
            features.top1_prob
        );
        assert!(
            features.margin > 0.5,
            "margin should be large: {}",
            features.margin
        );
        assert!(
            features.top3_mass > 0.99,
            "top3_mass should be near 1.0: {}",
            features.top3_mass
        );
        assert!(
            features.entropy < 0.5,
            "entropy should be low: {}",
            features.entropy
        );
        assert!(
            (features.step_norm - 0.25).abs() < 1e-5,
            "step_norm should be 2/8"
        );
        assert!(
            (features.pos_norm - 3.0 / 16.0).abs() < 1e-5,
            "pos_norm should be 3/16"
        );
    }

    #[test]
    fn test_sampler_features_excludes_mask() {
        let logits = vec![10.0, 5.0, 1.0, 0.5];
        let features = SamplerFeatures::from_logits(&logits, 0, 4, 0, 4, 0);
        assert!(features.top1_prob < 1.0, "should not include mask token");
    }

    #[test]
    fn test_sampler_features_uniform_logits() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let features = SamplerFeatures::from_logits(&logits, 0, 4, 0, 4, 999);
        assert!(
            (features.top1_prob - 0.25).abs() < 0.01,
            "uniform top1≈0.25: {}",
            features.top1_prob
        );
        assert!(
            features.margin < 0.01,
            "uniform margin≈0: {}",
            features.margin
        );
        assert!(
            (features.entropy - 4.0f32.ln()).abs() < 0.01,
            "uniform entropy=ln(4): {}",
            features.entropy
        );
    }

    #[test]
    fn test_sampler_untrained_predicts_half() {
        let sampler = DiffusionSampler::untrained();
        let features = SamplerFeatures::from_logits(&[1.0, 2.0, 3.0], 0, 4, 0, 4, 999);
        let p = sampler.predict(&features);
        assert!(
            (p - 0.5).abs() < 1e-6,
            "untrained should predict 0.5, got {p}"
        );
        assert!((0.0..=1.0).contains(&p), "prediction should be in [0,1]");
    }

    #[test]
    fn test_sampler_decide_with_trained_weights() {
        let accept_sampler = DiffusionSampler::from_weights([2.0; 6], 1.0);
        let good_features = SamplerFeatures {
            top1_prob: 0.9,
            margin: 0.5,
            top3_mass: 0.95,
            entropy: 0.3,
            step_norm: 0.5,
            pos_norm: 0.5,
        };
        assert!(
            accept_sampler.decide(&good_features, 0.5),
            "should accept with positive weights"
        );
        let reject_sampler = DiffusionSampler::from_weights([-2.0; 6], -5.0);
        assert!(
            !reject_sampler.decide(&good_features, 0.5),
            "should reject with negative weights"
        );
    }

    #[test]
    fn test_sampler_predict_bounded() {
        let sampler = DiffusionSampler::from_weights([100.0; 6], 100.0);
        let features = SamplerFeatures {
            top1_prob: 1.0,
            margin: 1.0,
            top3_mass: 1.0,
            entropy: 0.0,
            step_norm: 1.0,
            pos_norm: 1.0,
        };
        let p = sampler.predict(&features);
        assert!(p <= 1.0, "predict should be ≤ 1.0, got {p}");
        assert!(p >= 0.0, "predict should be ≥ 0.0, got {p}");
    }

    /// D2F decode with sampler produces valid output.
    /// Run with: cargo test -p riir-gpu --features gemma2_d2f test_d2f_decode_with_sampler
    #[test]
    fn test_d2f_decode_with_sampler() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);
        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let sampler = DiffusionSampler::from_weights([1.0; 6], 1.0);
        let decode_config = Gemma2D2fConfig {
            block_size: 4,
            denoise_steps: 8,
            confidence_threshold: 0.5,
            temperature: 1.0,
            sampler: Some(sampler),
            sc_config: D2fScConfig::default(),
        };
        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
        let mut rng = fastrand::Rng::with_seed(42);
        let prompt = vec![42, 7];
        let result = d2f_decode_gemma2(&mut d2f, &prompt, 0, &decode_config, &mut rng);
        assert_eq!(
            result.tokens.len(),
            prompt.len() + decode_config.block_size,
            "prompt + block tokens"
        );
        for (i, &t) in prompt.iter().enumerate() {
            assert_eq!(result.tokens[i], t, "prompt token at pos {i} unchanged");
        }
        for pos in prompt.len()..result.tokens.len() {
            assert!(
                result.tokens[pos] < config.vocab_size,
                "valid vocab at pos {pos}"
            );
        }
        println!(
            "d2f_decode_with_sampler: {} tokens, {} steps, converged={}",
            result.tokens.len(),
            result.steps_used,
            result.converged
        );

        drop(d2f);
        gpu_release_pages(&client);
    }

    // ── Self-Conditioning tests (Plan 250 T1-T4) ─────────────────

    #[test]
    fn test_sc_config_default_is_disabled() {
        let config = Gemma2D2fConfig::default();
        assert!(!config.sc_config.enabled, "SC should be disabled by default");
    }

    #[test]
    fn test_sc_config_in_d2f_config() {
        let config = Gemma2D2fConfig {
            sc_config: D2fScConfig::enabled(),
            ..Default::default()
        };
        assert!(config.sc_config.enabled);
    }

    /// SC-enabled decode should run without panicking.
    /// W_sc is lazily allocated as identity-padded on first SC step.
    #[test]
    fn test_d2f_decode_with_sc_enabled_runs() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let decode_config = Gemma2D2fConfig {
            block_size: 4,
            denoise_steps: 8,
            confidence_threshold: 0.1,
            temperature: 1.0,
            sampler: None,
            sc_config: D2fScConfig::enabled(),
        };

        let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
        let mut rng = fastrand::Rng::with_seed(42);

        let prompt = vec![1usize, 2, 3];
        let mask_token_id = config.vocab_size - 1;
        let result = d2f_decode_gemma2(&mut d2f, &prompt, mask_token_id, &decode_config, &mut rng);

        // Should produce valid tokens and not panic.
        assert_eq!(result.tokens.len(), prompt.len() + decode_config.block_size);
        for pos in prompt.len()..result.tokens.len() {
            assert!(result.tokens[pos] < config.vocab_size);
        }

        // W_sc should have been lazily allocated.
        assert!(d2f.w_sc_ref().is_some(), "W_sc should be allocated after SC decode");

        drop(d2f);
        gpu_release_pages(&client);
    }

    /// SC-enabled decode with identity W_sc should produce the SAME tokens
    /// as SC-disabled decode (zero behavioral change when untrained).
    /// This is the GOAT gate G4 property: "feature disabled = zero overhead".
    #[test]
    fn test_d2f_decode_sc_identity_matches_disabled() {
        let _heavy = heavy_model_test_gate();
        let ctx = crate::GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let prompt = vec![1usize, 2, 3];
        let mask_token_id = config.vocab_size - 1;

        // Run without SC.
        //
        // Scoped + shared weights: two full F32 instances live at once
        // ≈ 21 GB committed — the 24 GB heap OOMs (Issue 712). The weights are
        // seed-deterministic and identical for both runs, so one Vec serves
        // both instances; the instances themselves never overlap.
        let weights1 = create_test_weights(&config);
        let decode_config_disabled = Gemma2D2fConfig {
            block_size: 4,
            denoise_steps: 8,
            confidence_threshold: 0.1,
            temperature: 1.0,
            sampler: None,
            sc_config: D2fScConfig::disabled(),
        };
        let result_disabled = {
            let mut d2f1 =
                GpuGemmaCubeCLD2F::new(client.clone(), &weights1, &config, decode_config_disabled);
            let mut rng1 = fastrand::Rng::with_seed(42);
            d2f_decode_gemma2(&mut d2f1, &prompt, mask_token_id, &decode_config_disabled, &mut rng1)
        };

        // Run with SC enabled (W_sc = [I|0], identity-padded).
        let decode_config_enabled = Gemma2D2fConfig {
            sc_config: D2fScConfig::enabled(),
            ..decode_config_disabled
        };
        let result_enabled = {
            let mut d2f2 =
                GpuGemmaCubeCLD2F::new(client.clone(), &weights1, &config, decode_config_enabled);
            let mut rng2 = fastrand::Rng::with_seed(42);
            d2f_decode_gemma2(&mut d2f2, &prompt, mask_token_id, &decode_config_enabled, &mut rng2)
        };
        gpu_release_pages(&client);

        // With identity W_sc, SC has zero effect: tokens should match exactly.
        assert_eq!(
            result_disabled.tokens, result_enabled.tokens,
            "SC with identity W_sc should produce identical tokens to SC-disabled"
        );
        assert_eq!(
            result_disabled.steps_used, result_enabled.steps_used,
            "SC with identity W_sc should use same number of steps"
        );
    }
