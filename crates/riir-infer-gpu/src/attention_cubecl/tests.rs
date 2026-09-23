    use crate::cubecl_runtime::ActiveRuntime;

    use crate::cubecl_runtime::CubeCLContext;

    use super::*;

    /// Gemma 2 2B test constants.
    const N_HEAD: usize = 8;
    const N_KV_HEAD: usize = 4;
    const HEAD_DIM: usize = 256;
    const SOFTCAP: f32 = 50.0;
    const SCALE: f32 = 0.0625; // 1/√256

    // -----------------------------------------------------------------------
    // fold_factor unit tests (Plan 179 D1)
    // -----------------------------------------------------------------------

    // NOTE: gated to match `fold_factor` definition (Issue 377) — without `fold_dispatch`
    // the function does not exist, so the tests cannot compile.
    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn test_fold_factor_single_token() {
        assert_eq!(fold_factor(8, 1, 256), 1);
    }

    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn test_fold_factor_gemma2_spec4() {
        // Gemma 2 2B: n_head=8, spec_len=4 → max_fold=min(4,256/8)=4, largest div of 4 ≤ 4 = 4
        assert_eq!(fold_factor(8, 4, 256), 4);
    }

    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn test_fold_factor_no_benefit() {
        // Already fills workgroup: n_head=256, seq_len_q=2 → max_fold=min(2,256/256)=1
        assert_eq!(fold_factor(256, 2, 256), 1);
    }

    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn test_fold_factor_prime_seq() {
        // seq_len_q is prime (7), n_head=8 → max_fold=min(7,256/8)=7, but 7 divides 7
        assert_eq!(fold_factor(8, 7, 256), 7);
    }

    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn test_fold_factor_no_divisor() {
        // seq_len_q=6, n_head=128 → max_fold=min(6,256/128)=2, largest divisor of 6 ≤ 2 = 2
        assert_eq!(fold_factor(128, 6, 256), 2);
    }

    /// CPU reference implementation: batch softmax attention with softcapping.
    ///
    /// Computes the same result as the GPU kernel but using a simpler batch
    /// softmax approach. Mathematically equivalent to the online softmax
    /// (up to floating-point ordering differences).
    fn attention_decode_cpu(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_positions: usize,
        softcap: f32,
        scale: f32,
    ) -> Vec<f32> {
        let kv_stride = n_kv_head * head_dim;
        let mut output = vec![0.0f32; n_head * head_dim];

        for head_idx in 0..n_head {
            let head_off = head_idx * head_dim;
            let kv_group = head_idx * n_kv_head / n_head;
            let kv_off = kv_group * head_dim;

            // Compute softcapped scores for all positions.
            let mut scores = vec![0.0f32; n_positions];
            for (p, s) in scores.iter_mut().enumerate() {
                let k_base = p * kv_stride + kv_off;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query[head_off + d] * keys[k_base + d];
                }
                let raw = dot * scale;
                *s = softcap * (raw / softcap).tanh();
            }

            // Batch softmax: max → exp → sum → normalize.
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            // Weighted sum of values.
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (p, es) in exp_scores.iter().enumerate() {
                    let v_base = p * kv_stride + kv_off;
                    val += es * values[v_base + d];
                }
                output[head_off + d] = val / sum_exp;
            }
        }

        output
    }

    /// Verify GPU attention against CPU reference.
    ///
    /// Creates a combined KV buffer from separate key/value arrays,
    /// uploads to GPU, launches kernel, and compares results.
    fn verify_attention(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
        tolerance: f32,
    ) {
        let expected = attention_decode_cpu(
            query,
            keys,
            values,
            N_HEAD,
            N_KV_HEAD,
            HEAD_DIM,
            n_positions,
            SOFTCAP,
            SCALE,
        );

        let q_len = N_HEAD * HEAD_DIM;

        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        // Create combined KV buffer: keys || values
        let combined_kv = params.combine_kv(keys, values);

        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_handle = client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionCubeCL::launch::<ActiveRuntime>(
            client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        let bytes = client
            .read_one(attn_out_handle)
            .expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), q_len, "output length mismatch");
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < tolerance,
                "element {i}: expected {exp}, got {got}, err = {err}"
            );
        }
        println!(
            "attention (n_pos={n_positions}, heads={N_HEAD}/{N_KV_HEAD}, dim={HEAD_DIM}): max_error = {max_err}"
        );
    }

    /// Generate deterministic test data using sin/cos patterns.
    fn make_test_data(n_positions: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = N_HEAD * HEAD_DIM;
        let kv_len = n_positions * N_KV_HEAD * HEAD_DIM;

        let query: Vec<f32> = (0..q_len)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let keys: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.02).cos() * 0.3)
            .collect();
        let values: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.03).sin() * 0.2)
            .collect();

        (query, keys, values)
    }

    /// Single KV position: output should be the value vector (softmax of 1 element = 1.0).
    #[test]
    fn test_attention_single_position() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 1;
        let (query, keys, values) = make_test_data(n_positions);
        verify_attention(&client, &query, &keys, &values, n_positions, 1e-3);
    }

    /// Small context (4 positions): fits in one tile, tests basic multi-position softmax.
    #[test]
    fn test_attention_small_context() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 4;
        let (query, keys, values) = make_test_data(n_positions);
        verify_attention(&client, &query, &keys, &values, n_positions, 1e-2);
    }

    /// Partial tile (200 positions): less than one full 256-position tile.
    /// Tests the n_in_tile boundary and valid_pos guards.
    #[test]
    fn test_attention_partial_tile() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 200;
        let (query, keys, values) = make_test_data(n_positions);
        verify_attention(&client, &query, &keys, &values, n_positions, 1e-2);
    }

    /// Two tiles (300 positions): requires online softmax update across tiles.
    /// Tests the critical online softmax correction factors.
    #[test]
    fn test_attention_two_tiles() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 300;
        let (query, keys, values) = make_test_data(n_positions);
        verify_attention(&client, &query, &keys, &values, n_positions, 1e-2);
    }

    /// Exactly one full tile (256 positions): edge case where n_in_tile == cube_size.
    #[test]
    fn test_attention_exact_tile() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 256;
        let (query, keys, values) = make_test_data(n_positions);
        verify_attention(&client, &query, &keys, &values, n_positions, 1e-2);
    }

    /// Realistic Gemma 2 decode dimensions (512 positions).
    /// Tests with a representative KV cache size for short-context decode.
    #[test]
    fn test_attention_gemma2_realistic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 512;
        let (query, keys, values) = make_test_data(n_positions);
        verify_attention(&client, &query, &keys, &values, n_positions, 1e-2);
    }

    // -----------------------------------------------------------------------
    // Parametric-path parity tests (Bench 701)
    //
    // `AttentionCubeCL::launch` dispatches to the HARDCODED Gemma-2 2B kernel
    // only when (n_head, n_kv_head, head_dim, softcap) == (8, 4, 256, 50.0).
    // Every OTHER shape — including all Gemma-4 Q4_K training attention
    // (test config 4/2, real 12B 16/8 sliding) — routes to the PARAMETRIC
    // `attention_decode_llama_f32` kernel, which had ZERO test coverage
    // before this. The coverage hole hid the `get_window` interleaved-layout
    // bug (riir-ai gemma4_q4k_train) for 10 days.
    // -----------------------------------------------------------------------

    /// Parametric CPU reference (same math as `attention_decode_cpu`, explicit
    /// shape parameters instead of the Gemma-2 constants).
    #[allow(clippy::too_many_arguments)]
    fn attention_decode_cpu_parametric(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_positions: usize,
        softcap: f32,
        scale: f32,
    ) -> Vec<f32> {
        let kv_stride = n_kv_head * head_dim;
        let mut output = vec![0.0f32; n_head * head_dim];

        for head_idx in 0..n_head {
            let head_off = head_idx * head_dim;
            let kv_group = head_idx * n_kv_head / n_head;
            let kv_off = kv_group * head_dim;

            let mut scores = vec![0.0f32; n_positions];
            for (p, s) in scores.iter_mut().enumerate() {
                let k_base = p * kv_stride + kv_off;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query[head_off + d] * keys[k_base + d];
                }
                let raw = dot * scale;
                *s = if softcap > 0.0 {
                    softcap * (raw / softcap).tanh()
                } else {
                    raw
                };
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (p, es) in exp_scores.iter().enumerate() {
                    let v_base = p * kv_stride + kv_off;
                    val += es * values[v_base + d];
                }
                output[head_off + d] = val / sum_exp;
            }
        }
        output
    }

    /// Verify the PARAMETRIC kernel path (non-Gemma-2 shapes) against the CPU
    /// reference at an explicit shape. Uses sin/cos test data scaled to the
    /// given head count so GQA grouping is exercised.
    fn verify_attention_parametric(
        client: &ComputeClient<ActiveRuntime>,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_positions: usize,
        softcap: f32,
        scale: f32,
        tolerance: f32,
    ) {
        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        let query: Vec<f32> = (0..q_len)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let keys: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.02).cos() * 0.3)
            .collect();
        let values: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.03).sin() * 0.2)
            .collect();

        let expected = attention_decode_cpu_parametric(
            &query, &keys, &values, n_head, n_kv_head, head_dim, n_positions,
            softcap, scale,
        );

        let params = AttentionParams {
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
            softcap,
            scale,
        };
        let combined_kv = params.combine_kv(&keys, &values);

        let query_handle = client.create_from_slice(f32::as_bytes(&query));
        let kv_handle = client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionCubeCL::launch::<ActiveRuntime>(
            client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        let bytes = client
            .read_one(attn_out_handle)
            .expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), q_len, "output length mismatch");
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < tolerance,
                "element {i}: expected {exp}, got {got}, err = {err}"
            );
        }
        println!(
            "attention parametric (n_pos={n_positions}, heads={n_head}/{n_kv_head}, \
             dim={head_dim}, softcap={softcap}, scale={scale}): max_error = {max_err}"
        );
    }

    /// Gemma-4 Q4_K training shape (the Issue 436 G1 fixture): 4 q-heads,
    /// 2 kv-heads, scale=1.0. Exercises the parametric kernel + 2:1 GQA.
    #[test]
    fn test_attention_parametric_gemma4_q4k_shape() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // n_pos=1 is layout-trivial; 4 and 8 require correct [keys||values]
        // separation (an interleaved buffer would mix v-rows into the scores).
        for n_positions in [1usize, 4, 8] {
            verify_attention_parametric(
                &client, 4, 2, 256, n_positions, 50.0, 1.0, 1e-3,
            );
        }
    }

    /// Real Gemma-4 12B sliding-layer shape: 16 q-heads, 8 kv-heads,
    /// head_dim=256, scale=1.0 — the production parametric consumer.
    #[test]
    fn test_attention_parametric_gemma4_12b_sliding() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        verify_attention_parametric(
            &client, 16, 8, 256, 300, 50.0, 1.0, 1e-2,
        );
    }

    /// MiniCPM-style shape: 16 heads, 2 kv-heads (8:1 GQA), head_dim=128,
    /// NO softcap — exercises the parametric kernel's softcap-bypass branch.
    #[test]
    fn test_attention_parametric_no_softcap() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        verify_attention_parametric(
            &client, 16, 2, 128, 128, 0.0, 0.088_388_35, 1e-2,
        );
    }

    // -----------------------------------------------------------------------
    // Block-causal attention tests (Plan 108 T2)
    // -----------------------------------------------------------------------

    /// CPU reference: block-causal attention for a single query position.
    ///
    /// Computes the same result as the GPU block-causal kernel but using
    /// batch softmax. Only attends to positions `0..t_n` where
    /// `t_n = block_causal_t_n(pos, prompt_len, block_size, n_positions)`.
    #[cfg(feature = "gemma2_d2f")]
    fn attention_block_causal_cpu(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_positions: usize,
        pos: usize,
        prompt_len: usize,
        block_size: usize,
        softcap: f32,
        scale: f32,
    ) -> Vec<f32> {
        let kv_stride = n_kv_head * head_dim;
        let mut output = vec![0.0f32; n_head * head_dim];

        // Compute block-causal attention boundary
        let t_n = if pos < prompt_len {
            prompt_len
        } else {
            let block_idx = (pos - prompt_len) / block_size;
            let block_end = prompt_len + (block_idx + 1) * block_size;
            block_end.min(n_positions)
        };

        for head_idx in 0..n_head {
            let head_off = head_idx * head_dim;
            let kv_group = head_idx * n_kv_head / n_head;
            let kv_off = kv_group * head_dim;

            // Compute softcapped scores for attended positions only.
            let mut scores = vec![0.0f32; t_n];
            for (p, s) in scores.iter_mut().enumerate() {
                let k_base = p * kv_stride + kv_off;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query[head_off + d] * keys[k_base + d];
                }
                let raw = dot * scale;
                *s = softcap * (raw / softcap).tanh();
            }

            // Batch softmax: max → exp → sum → normalize.
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();

            // Weighted sum of values.
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (p, es) in exp_scores.iter().enumerate() {
                    let v_base = p * kv_stride + kv_off;
                    val += es * values[v_base + d];
                }
                output[head_off + d] = val / sum_exp;
            }
        }

        output
    }

    /// Verify GPU block-causal attention against CPU reference.
    ///
    /// Creates combined KV buffer, uploads to GPU, launches kernel,
    /// and compares results within tolerance.
    #[cfg(feature = "gemma2_d2f")]
    fn verify_block_causal_attention(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
        pos: usize,
        prompt_len: usize,
        block_size: usize,
        tolerance: f32,
    ) {
        let expected = attention_block_causal_cpu(
            query,
            keys,
            values,
            N_HEAD,
            N_KV_HEAD,
            HEAD_DIM,
            n_positions,
            pos,
            prompt_len,
            block_size,
            SOFTCAP,
            SCALE,
        );

        let q_len = N_HEAD * HEAD_DIM;

        let params = AttentionBlockCausalParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
            pos,
            prompt_len,
            block_size,
        };

        // Create combined KV buffer: keys || values
        let combined_kv = params.combine_kv(keys, values);

        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_handle = client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionCubeCL::launch_block_causal::<ActiveRuntime>(
            client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        let bytes = client
            .read_one(attn_out_handle)
            .expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), q_len, "output length mismatch");
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < tolerance,
                "element {i}: expected {exp}, got {got}, err = {err}"
            );
        }
        println!(
            "block-causal attention (n_pos={n_positions}, pos={pos}, prompt_len={prompt_len}, block_size={block_size}, t_n={}): max_error = {max_err}",
            params.t_n()
        );
    }

    /// Single block: prompt_len=0, block_size >= n_positions → all positions attend to all.
    /// Equivalent to bidirectional attention (no causal mask).
    #[cfg(feature = "gemma2_d2f")]
    #[test]
    fn test_attention_block_causal_single_block() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 8;
        let (query, keys, values) = make_test_data(n_positions);

        // prompt_len=0, block_size=8: t_n = min(0 + 1*8, 8) = 8 for all positions
        // All positions see all others → equivalent to full (bidirectional) attention
        for pos in 0..n_positions {
            verify_block_causal_attention(
                &client,
                &query,
                &keys,
                &values,
                n_positions,
                pos,
                0, // prompt_len
                8, // block_size = n_positions
                1e-2,
            );
        }
    }

    /// Prompt-only: all positions are prompt tokens, attend to all prompt tokens.
    #[cfg(feature = "gemma2_d2f")]
    #[test]
    fn test_attention_block_causal_prompt_only() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 16;
        let (query, keys, values) = make_test_data(n_positions);

        // All positions are prompt (pos < prompt_len): t_n = prompt_len for all
        for pos in 0..8 {
            verify_block_causal_attention(
                &client,
                &query,
                &keys,
                &values,
                n_positions,
                pos,
                8, // prompt_len — all test positions are prompt
                4, // block_size (irrelevant for prompt positions)
                1e-2,
            );
        }
    }

    /// Block-causal: prompt + generation blocks, across-block causal masking.
    #[cfg(feature = "gemma2_d2f")]
    #[test]
    fn test_attention_block_causal_generation() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 16;
        let prompt_len = 4;
        let block_size = 4;
        let (query, keys, values) = make_test_data(n_positions);

        // Generation block 0 (positions 4-7): attends to prompt(4) + block 0(4) = 8
        verify_block_causal_attention(
            &client,
            &query,
            &keys,
            &values,
            n_positions,
            4,
            prompt_len,
            block_size,
            1e-2,
        );
        verify_block_causal_attention(
            &client,
            &query,
            &keys,
            &values,
            n_positions,
            7,
            prompt_len,
            block_size,
            1e-2,
        );

        // Generation block 1 (positions 8-11): attends to prompt(4) + block 0(4) + block 1(4) = 12
        verify_block_causal_attention(
            &client,
            &query,
            &keys,
            &values,
            n_positions,
            8,
            prompt_len,
            block_size,
            1e-2,
        );
        verify_block_causal_attention(
            &client,
            &query,
            &keys,
            &values,
            n_positions,
            11,
            prompt_len,
            block_size,
            1e-2,
        );

        // Generation block 2 (positions 12-15): attends to all 16 positions
        verify_block_causal_attention(
            &client,
            &query,
            &keys,
            &values,
            n_positions,
            15,
            prompt_len,
            block_size,
            1e-2,
        );
    }

    /// Partial tile: t_n doesn't align to tile boundary.
    /// Tests the valid_pos guards within a tile.
    #[cfg(feature = "gemma2_d2f")]
    #[test]
    fn test_attention_block_causal_partial_tile() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 300;
        let prompt_len = 5;
        let block_size = 7;
        let (query, keys, values) = make_test_data(n_positions);

        // Position 10: gen_offset=5, block_idx=0, block_end=5+7=12, t_n=12
        verify_block_causal_attention(
            &client,
            &query,
            &keys,
            &values,
            n_positions,
            10,
            prompt_len,
            block_size,
            1e-2,
        );

        // Position 100: gen_offset=95, block_idx=13, block_end=5+14*7=103, t_n=103
        verify_block_causal_attention(
            &client,
            &query,
            &keys,
            &values,
            n_positions,
            100,
            prompt_len,
            block_size,
            1e-2,
        );
    }

    // -----------------------------------------------------------------------
    // GOAT proof: fold dispatch (Plan 179 D1, T8)
    // -----------------------------------------------------------------------

    // G1: `fold_factor()` correctness — unit tests pass for all combos.
    // Already covered by test_fold_factor_* above.

    /// Run baseline GPU attention decode and return output vector.
    ///
    /// Only called from `fold_dispatch` tests; `#[allow(dead_code)]` silences
    /// the warning in builds without that feature.
    #[allow(dead_code)]
    fn run_attention_decode(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        combined: &[f32],
        n_positions: usize,
    ) -> Vec<f32> {
        let q_len = N_HEAD * HEAD_DIM;
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_handle = client.create_from_slice(f32::as_bytes(combined));
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionCubeCL::launch::<ActiveRuntime>(
            client,
            query_handle,
            kv_handle,
            out_handle.clone(),
            &params,
        );

        let bytes = client
            .read_one(out_handle)
            .expect("should read baseline output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// G2: Folded dispatch produces same output as baseline (seq_len_q=1 → fold=1).
    /// When fold_factor returns 1, the folded kernel should behave identically to
    /// the baseline kernel for a single query token.
    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn goat_fold_dispatch_seq1_matches_baseline() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 128;
        let (query, keys, values) = make_test_data(n_positions);
        let combined = super::AttentionParams {
            n_positions,
            ..Default::default()
        }
        .combine_kv(&keys, &values);

        // Baseline: single-token decode via attention_decode_f32
        let baseline_out = run_attention_decode(&client, &query, &combined, n_positions);

        // Folded: single-token decode via attention_decode_folded_f32
        let params = super::AttentionParams {
            n_positions,
            ..Default::default()
        };
        let q_len = N_HEAD * HEAD_DIM;
        let q_handle = client.create_from_slice(f32::as_bytes(&query));
        let _kv_len = 2 * n_positions * N_KV_HEAD * HEAD_DIM;
        let kv_handle = client.create_from_slice(f32::as_bytes(&combined));
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        super::AttentionCubeCL::launch_folded::<ActiveRuntime>(
            &client,
            q_handle,
            kv_handle,
            out_handle.clone(),
            &params,
            1, // seq_len_q = 1
        );

        let folded_bytes = client
            .read_one(out_handle)
            .expect("should read folded output");
        let folded_out = f32::from_bytes(&folded_bytes).to_vec();

        // Verify: outputs match within FP tolerance
        let mut max_diff = 0.0f32;
        for (i, (a, b)) in baseline_out.iter().zip(folded_out.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-3,
                "GOAT FAIL G2: folded seq1 mismatch at [{i}]: baseline={a}, folded={b}, diff={diff}"
            );
        }
        eprintln!("  G2 PASS: max_diff={max_diff:.6e} (seq_len_q=1)");
    }

    /// G2 extended: Folded dispatch produces correct output for seq_len_q=4.
    /// Each of 4 query tokens should produce the same output as running
    /// the baseline kernel 4 times independently.
    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn goat_fold_dispatch_seq4_correctness() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 128;
        let (query_single, keys, values) = make_test_data(n_positions);
        let combined = super::AttentionParams {
            n_positions,
            ..Default::default()
        }
        .combine_kv(&keys, &values);

        // Baseline: run single-token kernel once to get expected output per token
        let expected_out = run_attention_decode(&client, &query_single, &combined, n_positions);

        // Folded: create batched query with 4 identical tokens
        let seq_len_q = 4usize;
        let mut batched_query = Vec::with_capacity(seq_len_q * N_HEAD * HEAD_DIM);
        for _ in 0..seq_len_q {
            batched_query.extend_from_slice(&query_single);
        }
        let batched_out_len = seq_len_q * N_HEAD * HEAD_DIM;

        let params = super::AttentionParams {
            n_positions,
            ..Default::default()
        };
        let q_handle = client.create_from_slice(f32::as_bytes(&batched_query));
        let _kv_len = 2 * n_positions * N_KV_HEAD * HEAD_DIM;
        let kv_handle = client.create_from_slice(f32::as_bytes(&combined));
        let out_handle = client.empty(batched_out_len * core::mem::size_of::<f32>());

        super::AttentionCubeCL::launch_folded::<ActiveRuntime>(
            &client,
            q_handle,
            kv_handle,
            out_handle.clone(),
            &params,
            seq_len_q,
        );

        let folded_bytes = client
            .read_one(out_handle)
            .expect("should read folded output");
        let folded_out = f32::from_bytes(&folded_bytes).to_vec();

        // Verify: each of 4 token outputs matches the single-token baseline
        let mut max_diff = 0.0f32;
        for tok in 0..seq_len_q {
            let tok_offset = tok * N_HEAD * HEAD_DIM;
            for (i, (a, b)) in expected_out
                .iter()
                .zip(folded_out[tok_offset..].iter())
                .enumerate()
            {
                let diff = (a - b).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                assert!(
                    diff < 1e-3,
                    "GOAT FAIL G2: folded seq4 tok={tok} idx={i}: expected={a}, got={b}, diff={diff}"
                );
            }
        }
        eprintln!("  G2 PASS: max_diff={max_diff:.6e} (seq_len_q={seq_len_q})");
    }

    /// G3: No NaN — decode steps with folded dispatch, all logits finite.
    ///
    /// Simulates growing context (KV positions 1,2,4,8,...,1024) to stress the
    /// online softmax accumulation in the folded kernel. Uses power-of-2 steps
    /// to cover tile boundary transitions efficiently.
    ///
    /// For the full 1000-step run, use:
    /// ```sh
    /// cargo test -p riir-gpu --features "cubecl_runtime,fold_dispatch" --lib -- goat_no_nan --release
    /// ```
    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn goat_fold_dispatch_no_nan() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let seq_len_q = 4usize;
        // Power-of-2 positions + boundaries: covers tile transitions efficiently
        let n_positions_list: &[usize] = &[
            1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 767, 768, 999,
            1000, 1023, 1024,
        ];
        let max_n_pos = *n_positions_list.last().unwrap();
        let q_dim = N_HEAD * HEAD_DIM; // 2048
        let kv_stride = N_KV_HEAD * HEAD_DIM; // 1024

        // Pre-allocate KV cache for max positions (keys + values)
        let kv_total = 2 * max_n_pos * kv_stride;
        let kv_data: Vec<f32> = (0..kv_total)
            .map(|i| ((i as f32) * 0.02).cos() * 0.3)
            .collect();

        // Fixed query (4 tokens, identical)
        let single_q: Vec<f32> = (0..q_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let batched_query: Vec<f32> = single_q.repeat(seq_len_q);

        let mut nan_count = 0usize;
        let mut inf_count = 0usize;

        for &n_positions in n_positions_list {
            let combined = super::AttentionParams {
                n_positions,
                ..Default::default()
            }
            .combine_kv(
                &kv_data[..n_positions * kv_stride],
                &kv_data[n_positions * kv_stride..2 * n_positions * kv_stride],
            );

            let params = super::AttentionParams {
                n_positions,
                ..Default::default()
            };
            let q_handle = client.create_from_slice(f32::as_bytes(&batched_query));
            let kv_handle = client.create_from_slice(f32::as_bytes(&combined));
            let out_len = seq_len_q * N_HEAD * HEAD_DIM;
            let out_handle = client.empty(out_len * core::mem::size_of::<f32>());

            super::AttentionCubeCL::launch_folded::<ActiveRuntime>(
                &client,
                q_handle,
                kv_handle,
                out_handle.clone(),
                &params,
                seq_len_q,
            );

            let out_bytes = client.read_one(out_handle).expect("should read output");
            let output = f32::from_bytes(&out_bytes);

            for (i, &val) in output.iter().enumerate() {
                if val.is_nan() {
                    nan_count += 1;
                    if nan_count <= 5 {
                        eprintln!("  G3 FAIL: NaN at n_pos={n_positions} idx={i}");
                    }
                } else if val.is_infinite() {
                    inf_count += 1;
                    if inf_count <= 5 {
                        eprintln!("  G3 FAIL: Inf at n_pos={n_positions} idx={i}");
                    }
                }
            }

            // Early exit on first failure
            if nan_count > 0 || inf_count > 0 {
                panic!(
                    "GOAT FAIL G3: non-finite outputs at n_pos={n_positions} (NaN={nan_count}, Inf={inf_count})"
                );
            }
        }

        eprintln!(
            "  G3 PASS: all outputs finite across {} context sizes (max={max_n_pos})",
            n_positions_list.len()
        );
    }

    // -----------------------------------------------------------------------
    // Issue 515 T2 — refusal pins for the guards added beyond T4's `launch`
    // one (launch_folded / launch_block_causal / q8kv). A guard that stops
    // firing is indistinguishable from a repo with no such bug, so each new
    // wiring gets the same catch_unwind pin the T4 wiring has in
    // riir-train-gpu's `attention_kv_binding_must_be_trimmed_to_live_range`.
    // -----------------------------------------------------------------------

    /// Shared fixture: Gemma2-constant-shaped KV where the buffer holds MORE
    /// positions than `params.n_positions` declares — the compact_temp class.
    /// Returns (q, kv_padded, kv_exact, trim_bytes).
    /// Gated by the union of its callers' features — under default features
    /// both call sites compile away and an ungated fixture reads as dead.
    #[cfg(any(feature = "fold_dispatch", feature = "gemma2_d2f"))]
    fn padded_kv_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>, u64) {
        let n_positions = 1usize;
        let padded_positions = 8usize;
        let kv_stride = N_KV_HEAD * HEAD_DIM;
        let q: Vec<f32> = (0..N_HEAD * HEAD_DIM)
            .map(|i| ((i as f32 % 7.0) * 0.1) - 0.3)
            .collect();
        let kv_exact: Vec<f32> = (0..2 * n_positions * kv_stride)
            .map(|i| ((i as f32 % 5.0) * 0.2) - 0.4)
            .collect();
        let mut kv_padded = vec![0.0f32; 2 * padded_positions * kv_stride];
        kv_padded[..kv_exact.len()].copy_from_slice(&kv_exact);
        let trim_bytes =
            ((kv_padded.len() - kv_exact.len()) * core::mem::size_of::<f32>()) as u64;
        (q, kv_padded, kv_exact, trim_bytes)
    }

    fn guard_refuses(f: impl FnOnce()) -> bool {
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err();
        std::panic::set_hook(prev_hook);
        refused
    }

    #[cfg(feature = "fold_dispatch")]
    #[test]
    fn launch_folded_refuses_oversized_kv_binding() {
        let ctx = match CubeCLContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP: CubeCL init failed: {e:?}");
                return;
            }
        };
        let client = ctx.client();
        let (q, kv_padded, _, _) = padded_kv_fixture();
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions: 1,
            softcap: SOFTCAP,
            scale: SCALE,
        };
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let kv_h = client.create_from_slice(f32::as_bytes(&kv_padded));
        let out_h = client.empty(q.len() * core::mem::size_of::<f32>());
        let refused = guard_refuses(|| {
            AttentionCubeCL::launch_folded::<ActiveRuntime>(
                &client, q_h, kv_h, out_h, &params, 1,
            );
        });
        assert!(
            refused,
            "launch_folded accepted an oversized kv binding — its 515 guard is not wired"
        );
    }

    #[cfg(feature = "gemma2_d2f")]
    #[test]
    fn launch_block_causal_refuses_oversized_kv_binding() {
        let ctx = match CubeCLContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP: CubeCL init failed: {e:?}");
                return;
            }
        };
        let client = ctx.client();
        let (q, kv_padded, _, _) = padded_kv_fixture();
        let params = AttentionBlockCausalParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions: 1,
            softcap: SOFTCAP,
            scale: SCALE,
            pos: 0,
            prompt_len: 1,
            block_size: 8,
        };
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let kv_h = client.create_from_slice(f32::as_bytes(&kv_padded));
        let out_h = client.empty(q.len() * core::mem::size_of::<f32>());
        let refused = guard_refuses(|| {
            AttentionCubeCL::launch_block_causal::<ActiveRuntime>(
                &client, q_h, kv_h, out_h, &params,
            );
        });
        assert!(
            refused,
            "launch_block_causal accepted an oversized kv binding — its 515 guard is not wired"
        );
    }

    #[test]
    fn q8kv_launch_refuses_oversized_qs_binding() {
        use crate::attention_q8kv_cubecl::AttentionQ8KVCubeCL;
        let ctx = match CubeCLContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP: CubeCL init failed: {e:?}");
                return;
            }
        };
        let client = ctx.client();
        // Q8_0 geometry at Gemma2 constants: qs stride 256 u32s, scale
        // stride 32 f32s per position.
        let kv_stride_q8 = N_KV_HEAD * 8 * 8;
        let kv_scale_stride = N_KV_HEAD * 8;
        let live_positions = 1usize;
        let padded_positions = 8usize;
        let q: Vec<f32> = (0..N_HEAD * HEAD_DIM)
            .map(|i| ((i as f32 % 7.0) * 0.1) - 0.3)
            .collect();
        let qs_exact: Vec<f32> = (0..2 * live_positions * kv_stride_q8)
            .map(|i| ((i as f32 % 3.0) * 0.5) - 0.5)
            .collect();
        let mut qs_padded = vec![0.0f32; 2 * padded_positions * kv_stride_q8];
        qs_padded[..qs_exact.len()].copy_from_slice(&qs_exact);
        let scales: Vec<f32> = vec![0.5; 2 * live_positions * kv_scale_stride];

        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions: live_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let qs_h = client.create_from_slice(f32::as_bytes(&qs_padded));
        let scales_h = client.create_from_slice(f32::as_bytes(&scales));
        let out_h = client.empty(q.len() * core::mem::size_of::<f32>());
        let refused = guard_refuses(|| {
            AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
                &client, q_h, qs_h, scales_h, out_h, &params,
            );
        });
        assert!(
            refused,
            "q8kv launch accepted an oversized kv_qs binding — its 515 guard is not wired"
        );
    }
