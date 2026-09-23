    use super::*;
    use crate::context::GpuContext;
    use crate::test_gpu_support::{gpu_release_pages, heavy_model_test_gate};

    /// Verify CubeCL forward pass initializes and produces correct-shaped output.
    ///
    /// Uses the smallest viable test to verify end-to-end wiring without
    /// requiring the full Gemma 2 2B model (too large for CI).
    #[test]
    fn test_cubecl_helpers_rmsnorm() {
        let mut data = vec![3.0f32, 4.0, 0.0, 0.0];
        let gamma = vec![1.0f32, 1.0, 1.0, 1.0];
        // mean(x²) = (9 + 16) / 4 = 6.25, rsqrt(6.25 + eps) ≈ 0.4
        rmsnorm_gamma(&mut data, &gamma, 4, 1e-6);
        let expected_scale = 1.0 / 6.25f32.sqrt();
        assert!(
            (data[0] - 3.0 * expected_scale).abs() < 1e-5,
            "RMSNorm element 0: expected {}, got {}",
            3.0 * expected_scale,
            data[0]
        );
        assert!(
            (data[1] - 4.0 * expected_scale).abs() < 1e-5,
            "RMSNorm element 1: expected {}, got {}",
            4.0 * expected_scale,
            data[1]
        );
    }

    #[test]
    fn test_cubecl_helpers_geglu() {
        let gate = vec![1.0f32, 2.0, -1.0];
        let up = vec![1.0f32, 1.0, 1.0];
        let mut out = vec![0.0f32; 3];
        geglu(&gate, &up, &mut out);
        // GELU(1) ≈ 0.8413, so out[0] ≈ 1.0 * 0.8413 * 1.0 ≈ 0.8413
        assert!(
            (out[0] - 0.8413).abs() < 0.01,
            "GeGLU[0]: expected ~0.8413, got {}",
            out[0]
        );
        // GELU(-1) ≈ -0.1587, so out[2] ≈ -1.0 * -0.1587 * 1.0 ≈ 0.1587
        assert!(
            out[2].abs() < 0.5,
            "GeGLU[2]: expected near 0, got {}",
            out[2]
        );
    }

    #[test]
    fn test_cubecl_helpers_softcap() {
        // tanh saturates slowly: tanh(100/50)=tanh(2)≈0.964 → 48.2, not 50.
        // Use large values so tanh saturates to ~1.0.
        let mut data = vec![10000.0f32, -10000.0, 0.0];
        softcap(&mut data, 50.0);
        assert!(
            (data[0] - 50.0).abs() < 0.01,
            "softcap large positive: expected ~50, got {}",
            data[0]
        );
        assert!(
            (data[1] - (-50.0)).abs() < 0.01,
            "softcap large negative: expected ~-50, got {}",
            data[1]
        );
        assert!(
            data[2].abs() < 0.01,
            "softcap zero: expected ~0, got {}",
            data[2]
        );
    }

    #[test]
    fn test_cubecl_helpers_rope() {
        // Test RoPE with simple values
        let mut data = vec![1.0f32, 0.0, 0.0, 1.0]; // 1 head, head_dim=4 (2 pairs)
        apply_rope(&mut data, 0, 4, 1, 10000.0);
        // At pos=0, all angles are 0, cos=1, sin=0 → no change
        assert!(
            (data[0] - 1.0).abs() < 1e-6,
            "RoPE pos=0 should not change data"
        );
        assert!(data[1].abs() < 1e-6, "RoPE pos=0 should zero sin component");
    }

    /// `apply_rope` must agree with `riir_engine`'s CPU RoPE at pos > 0.
    ///
    /// Issue 435: the pos=0 test above passes under **either** pairing
    /// convention, because RoPE at position 0 is the identity. That is how an
    /// interleaved-vs-rotate-half mismatch survived here — this test is the
    /// one that actually pins the convention.
    #[test]
    fn test_cubecl_helpers_rope_matches_riir_engine() {
        let head_dim = 64usize;
        let n_heads = 3usize;
        let theta = 10000.0f32;
        let freq = riir_infer_core::rope::RopeFreqTable::new(theta, head_dim);

        for pos in [1usize, 5, 97] {
            let input: Vec<f32> =
                (0..n_heads * head_dim).map(|i| ((i + pos) as f32 * 0.017).cos()).collect();

            let mut got = input.clone();
            apply_rope(&mut got, pos, head_dim, n_heads, theta);

            let mut q = input.clone();
            let mut k = input.clone();
            riir_infer_core::rope::apply_rope_with_freq(&mut q, &mut k, pos, head_dim, freq.as_slice());

            for (i, (&exp, &g)) in q.iter().zip(got.iter()).enumerate() {
                assert!(
                    (exp - g).abs() < 1e-5,
                    "pos={pos} element {i}: riir-engine {exp}, apply_rope {g}"
                );
            }
        }
    }

    #[test]
    fn test_cubecl_kv_cache() {
        let mut cache = CpuKVCache::new(2, 4);
        assert_eq!(cache.n_positions(0), 0);

        cache.store(0, 0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.n_positions(0), 1);
        assert_eq!(cache.keys[0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.values[0], &[5.0, 6.0, 7.0, 8.0]);

        cache.store(0, 1, &[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0]);
        assert_eq!(cache.n_positions(0), 2);

        let combined = cache.get_combined_kv(0, 2);
        assert_eq!(
            combined,
            &[
                1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );
    }

    /// Verify CubeCL GEMV dispatch produces correct results.
    #[test]
    fn test_cubecl_dispatch_gemv() {
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();

        // Simple 2×2 GEMV: output = weight @ input
        let weight: &[f32] = &[1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let input: &[f32] = &[1.0, 1.0];
        // Expected: [1+2, 3+4] = [3, 7]

        let weight_handle = client.create_from_slice(f32::as_bytes(weight));
        let fwd = FakeGpuGemmaCubeCL {
            client: client.clone(),
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
        };

        let result = fwd.dispatch_gemv(&weight_handle, input, 2, 2);
        assert_eq!(result.len(), 2, "GEMV output should have 2 elements");
        assert!(
            (result[0] - 3.0).abs() < 1e-5,
            "GEMV[0]: expected 3.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 7.0).abs() < 1e-5,
            "GEMV[1]: expected 7.0, got {}",
            result[1]
        );
    }

    /// Minimal struct to test dispatch_gemv without full model weights.
    struct FakeGpuGemmaCubeCL {
        client: ComputeClient<ActiveRuntime>,
        gemv_autotune: crate::gemv_autotune::GemvAutotune,
    }

    impl FakeGpuGemmaCubeCL {
        fn dispatch_gemv(&self, weight: &Handle, input: &[f32], m: usize, n: usize) -> Vec<f32> {
            let input_handle = self.client.create_from_slice(f32::as_bytes(input));
            let output_handle = self.client.empty(m * core::mem::size_of::<f32>());

            unsafe {
                self.gemv_autotune.launch::<ActiveRuntime>(
                    &self.client,
                    weight.clone(),
                    input_handle,
                    output_handle.clone(),
                    m,
                    n,
                );
            }

            let bytes = self
                .client
                .read_one(output_handle)
                .expect("read should succeed");
            f32::from_bytes(&bytes).to_vec()
        }

        fn dispatch_gemv_q4k(&self, weight: &Q4KHandle, input: &[f32]) -> Vec<f32> {
            let input_handle = self.client.create_from_slice(f32::as_bytes(input));
            let output_handle = self.client.empty(weight.m * core::mem::size_of::<f32>());

            unsafe {
                GemvQ4KCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    weight,
                    input_handle,
                    output_handle.clone(),
                );
            }

            let bytes = self
                .client
                .read_one(output_handle)
                .expect("read should succeed");
            f32::from_bytes(&bytes).to_vec()
        }
    }

    /// Verify CubeCL Q4_K GEMV dispatch produces results matching f32 GEMV
    /// within Q4_K quantization tolerance.
    #[test]
    fn test_cubecl_dispatch_gemv_q4k() {
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();

        // 2×256 matrix with sine wave weights
        let m = 2;
        let n = 256;
        let weight: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.1).sin() * 2.0).collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

        // F32 reference GEMV
        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let fwd = FakeGpuGemmaCubeCL {
            client: client.clone(),
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
        };
        let f32_result = fwd.dispatch_gemv(&weight_handle, &input, m, n);

        // Q4_K GEMV
        let mut all_blocks = Vec::new();
        let blocks_per_row = n.div_ceil(QK_K);
        let padded_n = blocks_per_row * QK_K;
        for row in 0..m {
            let mut padded_row = vec![0.0f32; padded_n];
            padded_row[..n].copy_from_slice(&weight[row * n..(row + 1) * n]);
            let start = all_blocks.len();
            all_blocks.resize(start + blocks_per_row, BlockQ4K::zeroed());
            quantize_row_q4_k(&padded_row, &mut all_blocks[start..]);
        }
        let q4k_handle = Q4KHandle::from_blocks(&client, &all_blocks, m, padded_n);

        let mut padded_input = input.clone();
        if padded_input.len() < padded_n {
            padded_input.resize(padded_n, 0.0);
        }

        let q4k_result = fwd.dispatch_gemv_q4k(&q4k_handle, &padded_input);

        assert_eq!(
            q4k_result.len(),
            m,
            "Q4_K GEMV output should have {m} elements"
        );

        // Q4_K quantization tolerance: ~0.5 per element × 256 ≈ up to ~128,
        // but errors cancel → expect < 10 for random-ish data
        let mut max_err = 0.0f32;
        for (&f32_val, &q4k_val) in f32_result.iter().zip(q4k_result.iter()) {
            let err = (f32_val - q4k_val).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 10.0, "Q4_K vs F32 max error too large: {max_err}");
    }

    /// Verify Q4_K full forward pass produces logits with correct shape.
    ///
    /// Uses `Config::gemma2_2b()` with random weights (same pattern as F32 test).
    /// Validates Q4_K quantize→upload→dequant+GEMV→attention wiring end-to-end.
    /// NOT numerically correct — just for structure/binding validation.
    #[test]
    fn test_cubecl_forward_q4k() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        // Use different tokens at pos=0 and pos=1 so that K/V differ across positions.
        // Same token at both positions would produce identical K/V (same GEMV input),
        // making attention output = V regardless of position (softmax over identical scores).
        let token0 = 42;
        let token1 = 7;

        // Create Q4_K forward pass (quantizes all projections to Q4_K on upload).
        // `client.clone()` (not move): the page release at the end of this test
        // reuses the client (Issue 712).
        let mut fwd = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);

        // Forward pass at position 0
        let logits = fwd.forward(token0, 0);

        assert_eq!(
            logits.len(),
            config.vocab_size,
            "Q4_K forward should produce {} logits",
            config.vocab_size
        );

        // Logits should be finite (no NaN/Inf from Q4_K dequant)
        for (i, &logit) in logits.iter().enumerate() {
            assert!(
                logit.is_finite(),
                "Logit[{i}] should be finite, got {logit}"
            );
        }

        // Forward pass at position 1 with different token (different K/V → different logits)
        let logits1 = fwd.forward(token1, 1);
        assert_eq!(
            logits1.len(),
            config.vocab_size,
            "Second forward should produce {} logits",
            config.vocab_size
        );

        let max_diff = logits
            .iter()
            .zip(logits1.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Different tokens at pos=0 vs pos=1 produce different K/V, so logits must differ
        let mut any_diff = false;
        for (&l0, &l1) in logits.iter().zip(logits1.iter()) {
            if (l0 - l1).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "Position 1 logits should differ from position 0 (max_diff={max_diff:.6})"
        );

        // Release the instance's pages before later tests in this process
        // allocate (Issue 712).
        drop(fwd);
        gpu_release_pages(&client);
    }

    /// Verify `forward_gpu` (GPU-resident, 1 sync/layer) produces same logits
    /// as `forward` (CPU-hybrid, 4 syncs/layer).
    ///
    /// Tests the T2.14 GPU-side RMSNorm, RoPE, GeGLU, and residual add kernels
    /// produce numerically equivalent results to the CPU fallback.
    #[test]
    fn test_cubecl_forward_gpu_matches_forward() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);

        let token0 = 42;
        let token1 = 7;
        let token2 = 123;

        // Run CPU-hybrid forward (4 sync/layer) for positions 0..3.
        //
        // The instance is scoped so its ~12 GB of GPU weight/KV handles DROP
        // before the GPU-resident instance is constructed: two full F32
        // instances live at once overcommit the 24 GB heap — the marginal pool
        // page for the autotune benchmark tensors then fails, the DSD-thread
        // panic silently drops launches, and the parity assert fails on
        // garbage logits (Issue 712). Dropped handle slices are reused by the
        // next constructor's reserves (pool `try_reserve` coalesces), so no
        // new pages are allocated either.
        let (logits_cpu_0, logits_cpu_1, logits_cpu_2) = {
            let mut fwd_cpu = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            (
                fwd_cpu.forward(token0, 0),
                fwd_cpu.forward(token1, 1),
                fwd_cpu.forward(token2, 2),
            )
        };

        // Run GPU-resident forward (1 sync/layer) for same positions.
        let (logits_gpu_0, logits_gpu_1, logits_gpu_2) = {
            let mut fwd_gpu = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            (
                fwd_gpu.forward_gpu(token0, 0),
                fwd_gpu.forward_gpu(token1, 1),
                fwd_gpu.forward_gpu(token2, 2),
            )
        };

        // Both instances dropped: hand their pages back so later tests in
        // this process start from a clean heap (Issue 712).
        gpu_release_pages(&client);

        // Verify shapes
        assert_eq!(logits_gpu_0.len(), config.vocab_size);
        assert_eq!(logits_gpu_1.len(), config.vocab_size);
        assert_eq!(logits_gpu_2.len(), config.vocab_size);

        // GPU-resident should produce very similar results to CPU-hybrid.
        // Tolerance accounts for floating-point differences in RMSNorm
        // (shared memory reduction vs CPU sequential sum) and GeGLU (tanh approx).
        let max_diff_0 = logits_cpu_0
            .iter()
            .zip(logits_gpu_0.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let max_diff_1 = logits_cpu_1
            .iter()
            .zip(logits_gpu_1.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let max_diff_2 = logits_cpu_2
            .iter()
            .zip(logits_gpu_2.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        // Allow reasonable tolerance for floating-point divergence
        // across 26 layers of GPU kernels vs CPU computation.
        // Initial threshold: 0.1 (can tighten as we validate).
        let tolerance = 0.1;
        assert!(
            max_diff_0 < tolerance,
            "forward_gpu vs forward pos=0 max_diff={max_diff_0:.6}"
        );
        assert!(
            max_diff_1 < tolerance,
            "forward_gpu vs forward pos=1 max_diff={max_diff_1:.6}"
        );
        assert!(
            max_diff_2 < tolerance,
            "forward_gpu vs forward pos=2 max_diff={max_diff_2:.6}"
        );

        println!(
            "forward_gpu vs forward max diffs: pos0={max_diff_0:.6}, pos1={max_diff_1:.6}, pos2={max_diff_2:.6}"
        );
    }

    /// Test Q4K GPU-resident forward matches CPU-hybrid forward (reproduces Issue 016).
    ///
    /// The Q4K GPU-resident path had a correctness bug causing token 234936 repetition.
    /// This test compares Q4K `forward_gpu` vs `forward` to catch any divergence.
    #[test]
    fn test_cubecl_q4k_forward_gpu_matches_forward() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);

        let token0 = 42;
        let token1 = 7;

        // Run CPU-hybrid forward for positions 0..2, then DROP the instance
        // before the GPU-resident one is constructed — never hold two model
        // instances live simultaneously (Issue 712: Q4K pairs fit, but the
        // same ordering keeps the suite's committed-page reuse clean).
        let (logits_cpu_0, logits_cpu_1) = {
            let mut fwd_cpu = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);
            (fwd_cpu.forward(token0, 0), fwd_cpu.forward(token1, 1))
        };

        // Run GPU-resident forward for same positions
        let (logits_gpu_0, logits_gpu_1) = {
            let mut fwd_gpu = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);
            (fwd_gpu.forward_gpu(token0, 0), fwd_gpu.forward_gpu(token1, 1))
        };

        gpu_release_pages(&client);

        // Verify shapes
        assert_eq!(
            logits_gpu_0.len(),
            config.vocab_size,
            "pos0 GPU logits length"
        );
        assert_eq!(
            logits_gpu_1.len(),
            config.vocab_size,
            "pos1 GPU logits length"
        );

        // Check all logits are finite (no NaN/Inf from Q4K dequant)
        for (i, &l) in logits_gpu_0.iter().enumerate() {
            assert!(l.is_finite(), "pos0 logit[{i}] not finite: {l}");
        }
        for (i, &l) in logits_gpu_1.iter().enumerate() {
            assert!(l.is_finite(), "pos1 logit[{i}] not finite: {l}");
        }

        // GPU-resident should produce similar results to CPU-hybrid.
        // Tolerance is wider than F32 due to Q4K quantization noise accumulation.
        let max_diff_0 = logits_cpu_0
            .iter()
            .zip(logits_gpu_0.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let max_diff_1 = logits_cpu_1
            .iter()
            .zip(logits_gpu_1.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        // Q4K quantization + GPU floating point differences across 26 layers
        let tolerance = 1.0;
        assert!(
            max_diff_0 < tolerance,
            "Q4K forward_gpu vs forward pos=0 max_diff={max_diff_0:.6}"
        );
        assert!(
            max_diff_1 < tolerance,
            "Q4K forward_gpu vs forward pos=1 max_diff={max_diff_1:.6}"
        );

        println!(
            "Q4K forward_gpu vs forward max diffs: pos0={max_diff_0:.6}, pos1={max_diff_1:.6}"
        );

        // Verify token diversity: argmax of pos1 logits should differ from pos0
        // (same token repeating suggests a bug in the pipeline)
        let top0 = logits_gpu_0
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map(|(i, _)| i)
            .unwrap();
        let top1 = logits_gpu_1
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map(|(i, _)| i)
            .unwrap();
        println!("Q4K GPU top tokens: pos0={top0}, pos1={top1}");
    }

    /// Test Q4K GPU-resident sequential decode (Issue 016 regression).
    ///
    /// Simulates the real inference loop: position 0 produces a token,
    /// then that token feeds into position 1 via forward_gpu.
    /// Checks that the GPU KV cache persists correctly between calls
    /// and that logits change between positions.
    #[test]
    fn test_cubecl_q4k_gpu_sequential_decode() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);

        // `client.clone()` (not move): the CPU-hybrid comparison instance below
        // reuses the same client after this GPU instance is dropped (Issue 712).
        let mut fwd = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);
        let bos_token = 2; // Standard BOS token

        // Position 0: BOS token
        let logits_p0 = fwd.forward_gpu(bos_token, 0);
        assert_eq!(logits_p0.len(), config.vocab_size, "vocab size");
        for (i, &l) in logits_p0.iter().enumerate() {
            assert!(l.is_finite(), "pos0 logit[{i}] not finite: {l}");
        }

        let top_p0 = argmax_of(&logits_p0);
        println!("Q4K GPU decode: pos=0 token={bos_token} → top={top_p0}");

        // Position 1: feed back the argmax token (simulates greedy decode)
        let logits_p1 = fwd.forward_gpu(top_p0, 1);
        assert_eq!(logits_p1.len(), config.vocab_size, "vocab size");
        for (i, &l) in logits_p1.iter().enumerate() {
            assert!(l.is_finite(), "pos1 logit[{i}] not finite: {l}");
        }

        let top_p1 = argmax_of(&logits_p1);
        println!("Q4K GPU decode: pos=1 token={top_p0} → top={top_p1}");

        // Position 2: continue decoding
        let logits_p2 = fwd.forward_gpu(top_p1, 2);
        assert_eq!(logits_p2.len(), config.vocab_size, "vocab size");
        for (i, &l) in logits_p2.iter().enumerate() {
            assert!(l.is_finite(), "pos2 logit[{i}] not finite: {l}");
        }

        let top_p2 = argmax_of(&logits_p2);
        println!("Q4K GPU decode: pos=2 token={top_p1} → top={top_p2}");

        // The GPU-resident instance's logits are already extracted — drop it
        // before constructing the CPU-hybrid comparison instance so the two
        // are never live simultaneously (Issue 712).
        drop(fwd);

        // The logits at different positions MUST differ (proves KV cache is working)
        let max_diff_01 = logits_p0
            .iter()
            .zip(logits_p1.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_diff_12 = logits_p1
            .iter()
            .zip(logits_p2.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Q4K GPU sequential logit diffs: p0↔p1={max_diff_01:.6}, p1↔p2={max_diff_12:.6}");

        // Compare: CPU-hybrid sequential decode to see if it also degenerates
        let (logits_cpu_0, logits_cpu_1, logits_cpu_2) = {
            let mut fwd_cpu = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);
            let logits_cpu_0 = fwd_cpu.forward(bos_token, 0);
            let top_cpu_0 = argmax_of(&logits_cpu_0);
            let logits_cpu_1 = fwd_cpu.forward(top_cpu_0, 1);
            let top_cpu_1 = argmax_of(&logits_cpu_1);
            let logits_cpu_2 = fwd_cpu.forward(top_cpu_1, 2);
            let top_cpu_2 = argmax_of(&logits_cpu_2);
            println!(
                "Q4K CPU decode: pos=0 → top={top_cpu_0}, pos=1 → top={top_cpu_1}, pos=2 → top={top_cpu_2}"
            );
            (logits_cpu_0, logits_cpu_1, logits_cpu_2)
        };

        gpu_release_pages(&client);

        let cpu_gpu_diff_0 = logits_cpu_0
            .iter()
            .zip(logits_p0.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let cpu_gpu_diff_1 = logits_cpu_1
            .iter()
            .zip(logits_p1.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let cpu_gpu_diff_2 = logits_cpu_2
            .iter()
            .zip(logits_p2.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        println!(
            "Q4K sequential CPU vs GPU diffs: p0={cpu_gpu_diff_0:.6}, p1={cpu_gpu_diff_1:.6}, p2={cpu_gpu_diff_2:.6}"
        );

        // Key check: GPU-resident and CPU-hybrid should match at each position.
        // If both degenerate (same token repeat), it's a weight quality issue, not a pipeline bug.
        // If CPU-hybrid produces diverse tokens but GPU doesn't, it's a pipeline bug.
        assert!(
            cpu_gpu_diff_0 < 1.0,
            "pos0 CPU vs GPU too large: {cpu_gpu_diff_0:.6}"
        );
        assert!(
            cpu_gpu_diff_1 < 1.0,
            "pos1 CPU vs GPU too large: {cpu_gpu_diff_1:.6}"
        );
        assert!(
            cpu_gpu_diff_2 < 1.0,
            "pos2 CPU vs GPU too large: {cpu_gpu_diff_2:.6}"
        );

        // Token diversity is expected but not enforced with random weights (degenerate by design)
        // With real weights, the CPU-hybrid path produces diverse tokens.
        // This test mainly checks CPU/GPU consistency.
    }

    /// Argmax helper for tests.
    fn argmax_of(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Create minimal test weights with correct shapes but random data.
    /// NOT numerically correct — just for structure/binding validation.
    fn create_test_weights(config: &Config) -> GemmaTransformerWeights {
        use riir_infer_core::gemma_layer::GemmaLayerWeights;
        use riir_infer_core::types::Rng;

        let mut rng = Rng::new(42);
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kv_dim = config.n_kv_head * config.head_dim;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;

        // Small scale to keep logits reasonable
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

        GemmaTransformerWeights {
            wte,
            final_norm,
            layers,
            // S5: gated like the struct fields (riir-gpu's default always
            // carried delta_routing, so the bare-cubecl posture never
            // compiled these tests at the old home).
            #[cfg(feature = "delta_routing")]
            delta_routing_query: vec![vec![0.0f32; config.n_embd]; config.n_layer],
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: vec![vec![1.0f32; config.n_embd]; config.n_layer],
        }
    }

    /// GOAT proof (T4): GPU argmax produces identical results to CPU argmax.
    ///
    /// Verifies correctness across multiple vocab sizes and edge cases.
    /// This is the critical correctness gate for GPU-resident sampling.
    #[cfg(feature = "gpu_decode_fusion")]
    #[test]
    fn test_goat_gpu_argmax_correctness() {
        use crate::cubecl_runtime::CubeCLContext;

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Helper: run CPU argmax.
        // Issue 718: FIRST-index tie-break (strict `>`) — matches the
        // ArgmaxCubeCL kernel spec (pinned by sampling_cubecl's own
        // test_argmax_all_equal) and the engine's CPU decode paths
        // (swir_validation argmax_u32, ict_runtime argmax_u8, itself
        // test-pinned as argmax_breaks_ties_by_lowest_index). Rust's
        // `max_by` returns the LAST tied index and is NOT the decode
        // convention — the previous oracle failed all_equal 0 vs 99.
        fn cpu_argmax(data: &[f32]) -> usize {
            let mut best_idx = 0usize;
            let mut best_val = data[0];
            for (i, &v) in data.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            best_idx
        }

        // Helper: run GPU argmax
        fn gpu_argmax(client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>, data: &[f32]) -> usize {
            let input_handle = client.create_from_slice(f32::as_bytes(data));
            let output_handle = client.empty(core::mem::size_of::<u32>());
            // SAFETY: input_handle contains data.len() f32 elements,
            // output_handle is pre-allocated with 4 bytes (1 u32).
            unsafe {
                ArgmaxCubeCL::launch::<ActiveRuntime>(
                    client,
                    input_handle,
                    output_handle.clone(),
                    data.len(),
                );
            }
            let bytes = client.read_one(output_handle).expect("GPU read failed");
            u32::from_bytes(&bytes)[0] as usize
        }

        // Test 1: Basic ascending
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(gpu_argmax(&client, &data), cpu_argmax(&data), "ascending");

        // Test 2: Basic descending
        let data: Vec<f32> = (0..10).rev().map(|i| i as f32).collect();
        assert_eq!(gpu_argmax(&client, &data), cpu_argmax(&data), "descending");

        // Test 3: All equal
        let data = vec![5.0f32; 100];
        assert_eq!(gpu_argmax(&client, &data), cpu_argmax(&data), "all_equal");

        // Test 4: Max at first position
        let mut data = vec![0.0f32; 1000];
        data[0] = 999.0;
        assert_eq!(gpu_argmax(&client, &data), cpu_argmax(&data), "max_first");

        // Test 5: Max at last position
        let mut data = vec![0.0f32; 1000];
        data[999] = 999.0;
        assert_eq!(gpu_argmax(&client, &data), cpu_argmax(&data), "max_last");

        // Test 6: All negative
        let data: Vec<f32> = (-100..0).map(|i| i as f32).collect();
        assert_eq!(
            gpu_argmax(&client, &data),
            cpu_argmax(&data),
            "all_negative"
        );

        // Test 7: Small differences (1e-7)
        let mut data = vec![0.0f32; 256];
        data[128] = 1.0;
        data[129] = 1.0 + 1e-7;
        assert_eq!(gpu_argmax(&client, &data), cpu_argmax(&data), "small_diff");

        // Test 8: Gemma 2 vocab size (256128) — max at random position
        let mut data = vec![0.1f32; 256128];
        data[100000] = 99.0;
        assert_eq!(
            gpu_argmax(&client, &data),
            cpu_argmax(&data),
            "gemma2_vocab"
        );

        // Test 9: Gemma 2 vocab size — max at last position
        let mut data = vec![0.1f32; 256128];
        data[256127] = 99.0;
        assert_eq!(
            gpu_argmax(&client, &data),
            cpu_argmax(&data),
            "gemma2_vocab_last"
        );

        // Test 10: Multiple random arrays at vocab size
        let mut rng = fastrand::Rng::new();
        for trial in 0..10 {
            let data: Vec<f32> = (0..256128).map(|_| rng.f32()).collect();
            let gpu = gpu_argmax(&client, &data);
            let cpu = cpu_argmax(&data);
            assert_eq!(gpu, cpu, "random trial {trial}: gpu={gpu} cpu={cpu}");
        }
    }

    /// Plan 171 Phase 4 T20: GOAT proof — CubeCL `forward_gpu()` (1 sync) matches
    /// `forward()` (104 syncs) output within FP32 tolerance.
    ///
    /// This is the critical correctness gate for sync elimination.
    /// Both methods use identical weights, so the outputs should match closely.
    /// The tolerance accounts for floating-point ordering differences between
    /// GPU kernel execution (parallel reductions) and CPU sequential computation.
    ///
    /// Gates the full sync-eliminated path behind `gpu_decode_fusion`.
    #[cfg(feature = "gpu_decode_fusion")]
    #[test]
    fn test_goat_forward_gpu_matches_forward() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);

        // ── F32 path ──────────────────────────────────────────────
        //
        // Phases are SEQUENCED (all CPU-hybrid positions, drop, then all
        // GPU-resident positions) instead of interleaved per position: two
        // full F32 instances live at once ≈ 21 GB committed — the 24 GB heap
        // OOMs, the DSD-thread panics silently drop launches, and the parity
        // assert fails on garbage logits (Issue 712). Dropped slices are
        // reused by the next constructor's reserves, so peak = ONE instance.

        let tokens = [42usize, 7, 123, 999, 0];
        let tolerance = 0.1;

        let logits_cpu_all: Vec<Vec<f32>> = {
            let mut fwd_cpu = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            tokens
                .iter()
                .enumerate()
                .map(|(pos, &token)| fwd_cpu.forward(token, pos))
                .collect()
        };
        gpu_release_pages(&client);

        let logits_gpu_all: Vec<Vec<f32>> = {
            let mut fwd_gpu = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            tokens
                .iter()
                .enumerate()
                .map(|(pos, &token)| fwd_gpu.forward_gpu(token, pos))
                .collect()
        };
        gpu_release_pages(&client);

        for (pos, ((logits_cpu, logits_gpu), &token)) in logits_cpu_all
            .iter()
            .zip(logits_gpu_all.iter())
            .zip(tokens.iter())
            .enumerate()
        {
            assert_eq!(logits_gpu.len(), config.vocab_size, "pos={pos} vocab size");

            // Check all logits are finite
            for (i, &l) in logits_gpu.iter().enumerate() {
                assert!(l.is_finite(), "pos={pos} logit[{i}] not finite: {l}");
            }

            let max_diff = logits_cpu
                .iter()
                .zip(logits_gpu.iter())
                .map(|(&c, &g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < tolerance,
                "F32 GOAT failed at pos={pos} token={token}: max_diff={max_diff:.6} > {tolerance}"
            );
            println!("GOAT F32 pos={pos} token={token}: max_diff={max_diff:.6} ✓");
        }

        // ── Q4K path ──────────────────────────────────────────
        // Same sequencing (Issue 712): never two instances live at once.

        let logits_cpu_q4k_all: Vec<Vec<f32>> = {
            let mut fwd_cpu_q4k = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);
            tokens
                .iter()
                .enumerate()
                .map(|(pos, &token)| fwd_cpu_q4k.forward(token, pos))
                .collect()
        };
        gpu_release_pages(&client);

        let logits_gpu_q4k_all: Vec<Vec<f32>> = {
            let mut fwd_gpu_q4k = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);
            tokens
                .iter()
                .enumerate()
                .map(|(pos, &token)| fwd_gpu_q4k.forward_gpu(token, pos))
                .collect()
        };
        gpu_release_pages(&client);

        let q4k_tolerance = 1.0; // Wider due to Q4K quantization noise

        for (pos, ((logits_cpu, logits_gpu), &token)) in logits_cpu_q4k_all
            .iter()
            .zip(logits_gpu_q4k_all.iter())
            .zip(tokens.iter())
            .enumerate()
        {
            assert_eq!(
                logits_gpu.len(),
                config.vocab_size,
                "Q4K pos={pos} vocab size"
            );

            for (i, &l) in logits_gpu.iter().enumerate() {
                assert!(l.is_finite(), "Q4K pos={pos} logit[{i}] not finite: {l}");
            }

            let max_diff = logits_cpu
                .iter()
                .zip(logits_gpu.iter())
                .map(|(&c, &g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_diff < q4k_tolerance,
                "Q4K GOAT failed at pos={pos} token={token}: max_diff={max_diff:.6} > {q4k_tolerance}"
            );
            println!("GOAT Q4K pos={pos} token={token}: max_diff={max_diff:.6} ✓");
        }

        println!(
            "\n=== GOAT Proof Passed: forward_gpu() (1 sync) matches forward() (104 syncs) ==="
        );
    }

    /// Plan 171 Phase 5 T23: Full pipeline GOAT proof — greedy decode.
    ///
    /// The definitive correctness test for the entire GPU decode fusion pipeline.
    /// Generates 8 tokens using two paths and verifies bit-identical argmax:
    ///
    /// 1. **Baseline**: `forward()` (CPU-hybrid, 104 syncs) + CPU argmax per token
    /// 2. **GPU-fused**: `generate_gpu()` (GPU-resident, 1 sync) + GPU argmax per token
    ///
    /// If this passes, the entire pipeline is correct:
    /// - GPU argmax matches CPU argmax (Phase 1)
    /// - GPU-resident hidden states match CPU-hybrid hidden states (Phase 4)
    /// - Autoregressive token feeding is correct across both paths
    #[cfg(feature = "gpu_decode_fusion")]
    #[test]
    fn test_goat_full_pipeline_decode() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);

        // ── Baseline: CPU-hybrid forward() + CPU argmax ──────────────────
        // ── Baseline: CPU-hybrid forward() + CPU argmax ──────────
        // Scoped so the instance drops before the GPU-fused one is
        // constructed — two full F32 instances at once overcommit the 24 GB
        // heap (Issue 712).
        let prompt = vec![42usize, 7, 123];
        let max_tokens = 8;

        let baseline_tokens: Vec<usize> = {
            let mut fwd_baseline = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            let mut baseline_tokens: Vec<usize> = Vec::new();

            // Prefill: process prompt tokens, get logits, CPU argmax
            let mut last_logits = Vec::new();
            for (i, &token) in prompt.iter().enumerate() {
                last_logits = fwd_baseline.forward(token, i);
            }
            let first_token = argmax_of(&last_logits);
            if first_token != 1 {
                baseline_tokens.push(first_token);
            }

            // Decode: CPU argmax per token
            for _ in 1..max_tokens {
                if baseline_tokens.is_empty() {
                    break;
                }
                let pos = prompt.len() + baseline_tokens.len() - 1;
                let logits = fwd_baseline.forward(*baseline_tokens.last().unwrap(), pos);
                let token = argmax_of(&logits);
                if token == 1 {
                    break;
                }
                baseline_tokens.push(token);
            }
            baseline_tokens
        };
        gpu_release_pages(&client);

        // ── GPU-fused: generate_gpu() + GPU argmax ───────────
        let gpu_tokens = {
            let mut fwd_gpu = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            fwd_gpu.generate_gpu(&prompt, max_tokens)
        };
        gpu_release_pages(&client);

        // ── Verify bit-identical greedy decode ──────────────────────────
        assert_eq!(
            baseline_tokens.len(),
            gpu_tokens.len(),
            "Token count mismatch: baseline={}, gpu={}",
            baseline_tokens.len(),
            gpu_tokens.len()
        );

        for (i, (b, g)) in baseline_tokens.iter().zip(gpu_tokens.iter()).enumerate() {
            assert_eq!(
                b, g,
                "Token mismatch at position {i}: baseline={b}, gpu={g}"
            );
        }

        println!(
            "\n=== GOAT Full Pipeline Proof Passed: {} tokens match bit-identically ===",
            baseline_tokens.len()
        );
        for (i, &t) in gpu_tokens.iter().enumerate() {
            println!("  token[{i}] = {t}");
        }
    }

    /// Plan 171 Phase 6 T34: GOAT proof — speculative decode with early-exit draft.
    ///
    /// Verifies that speculative decoding produces the same greedy output as
    /// non-speculative `generate_gpu()`. Two correctness properties are checked:
    ///
    /// 1. **Early-exit draft produces finite logits** — `forward_gpu_logits_handle_max_layer`
    ///    with `max_layer < n_layer` produces valid (not NaN/Inf) logits.
    /// 2. **Speculative decode tokens are all full-model predictions** — since both
    ///    use greedy argmax, speculative decode should produce the same tokens
    ///    as the non-speculative path when the draft model agrees.
    ///
    /// With random weights, the early-exit draft will have low acceptance rate,
    /// but the verify loop guarantees correctness: every output token comes from
    /// the full model's argmax.
    #[cfg(feature = "gpu_decode_fusion")]
    #[test]
    fn test_goat_early_exit_draft() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);

        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);

        // ── Part 1: Early-exit draft produces finite logits ──────────────
        //
        // Reuse a single instance for all max_layer values to conserve GPU memory.
        // Each call re-processes from scratch at pos=1.
        {
            let mut fwd = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            let token = 42usize;

            // Warmup to trigger autotune
            let _ = fwd.forward_gpu(token, 0);

            for max_layer in [1, 4, 8, 13, 26] {
                let logits_handle = fwd.forward_gpu_logits_handle_max_layer(token, 1, max_layer);
                let logits_handle_copy = logits_handle.clone();
                let logits = fwd.read_handle(&logits_handle_copy);

                assert_eq!(
                    logits.len(),
                    config.vocab_size,
                    "max_layer={max_layer} vocab size"
                );
                for (i, &l) in logits.iter().enumerate() {
                    assert!(
                        l.is_finite(),
                        "max_layer={max_layer} logit[{i}] not finite: {l}"
                    );
                }
                println!(
                    "  Early-exit max_layer={max_layer}: all {} logits finite ✓",
                    logits.len()
                );
            }
        }
        gpu_release_pages(&client);

        // ── Part 2: Speculative decode tokens are all full-model predictions ──
        //
        // With speculative decoding, every accepted token has been verified by
        // the full model. Every rejected token is replaced by the full model's
        // prediction. So all output tokens should match the greedy decode.
        //
        // We create instances one at a time to avoid GPU memory exhaustion.
        let prompt = vec![42usize, 7, 123];
        let max_tokens = 16;

        // Greedy baseline (one instance)
        let greedy_tokens = {
            let mut fwd_greedy = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            fwd_greedy.generate_gpu(&prompt, max_tokens)
        };
        gpu_release_pages(&client);

        // Test a single representative speculative config to avoid resource pressure
        {
            let mut fwd_spec = GpuGemmaCubeCL::new(client.clone(), &weights, &config);
            let result = fwd_spec.generate_gpu_speculative(
                &prompt, max_tokens, 4, // draft_lookahead
                4, // draft_layers
            );

            // All output tokens must match greedy decode
            // (since verify uses full model argmax for every token)
            assert_eq!(
                result.tokens.len(),
                greedy_tokens.len(),
                "token count mismatch: spec={}, greedy={}",
                result.tokens.len(),
                greedy_tokens.len()
            );

            for (i, (s, g)) in result.tokens.iter().zip(greedy_tokens.iter()).enumerate() {
                assert_eq!(s, g, "token mismatch at pos {i}: spec={s}, greedy={g}");
            }

            println!(
                "  Speculative draft_layers=4 lookahead=4: \
                 {} tokens match greedy ✓ (acceptance={:.1}%, rounds={})",
                result.tokens.len(),
                result.acceptance_rate() * 100.0,
                result.speculation_rounds
            );
        }
        gpu_release_pages(&client);

        println!("\n=== GOAT Early-Exit Draft Proof Passed: speculative matches greedy decode ===");
    }

    // ── Plan 171 T28 GOAT Proof: Q4_K fused QKV ──────────────────────

    /// GOAT proof: Q4_K fused triple QKV produces same output as 3 separate GEMV dispatches.
    ///
    /// Creates a Q4_K model, runs forward through one layer using:
    /// 1. Separate Q, K, V dispatches (3 dispatches)
    /// 2. Fused QKV dispatch (1 dispatch)
    ///    Compares logits for exact match.
    #[cfg(feature = "gpu_decode_fusion")]
    #[test]
    fn test_goat_q4k_fused_qkv() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);
        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let model = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);

        let q_dim = config.n_head * config.head_dim;
        let kv_dim = config.n_kv_head * config.head_dim;
        let n = config.n_embd;

        // Create a test input vector
        let input: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) / 10.0).collect();
        let input_handle = client.create_from_slice(f32::as_bytes(&input));

        let CubeCLWeightFormat::Q4K(w) = &model.weights else { panic!("test requires Q4_K weight format") };

        // Path 1: Separate Q4_K GEMV dispatches (3 dispatches)
        let q = model.dispatch_gemv_q4k_gpu(&w.layers[0].attn_wq, input_handle.clone());
        let k = model.dispatch_gemv_q4k_gpu(&w.layers[0].attn_wk, input_handle.clone());
        let v = model.dispatch_gemv_q4k_gpu(&w.layers[0].attn_wv, input_handle.clone());

        let q_data = model.read_handle(&q);
        let k_data = model.read_handle(&k);
        let v_data = model.read_handle(&v);

        // Path 2: Fused Q4_K QKV GEMV dispatch (1 dispatch)
        let qkv_handle = w.layers[0]
            .qkv_combined
            .as_ref()
            .expect("fused QKV handle should be built");
        let fused = model.dispatch_gemv_qkv_q4k_gpu(qkv_handle, input_handle.clone());
        let fused_data = model.read_handle(&fused);

        // Compare: fused output is [Q | K | V]
        let total = q_dim + 2 * kv_dim;
        assert_eq!(fused_data.len(), total, "fused output length mismatch");

        let mut max_err = 0.0f32;
        for i in 0..q_dim {
            let err = (fused_data[i] - q_data[i]).abs();
            max_err = max_err.max(err);
            assert!(
                err < 0.01,
                "Q mismatch at {i}: separate={}, fused={}, err={err}",
                q_data[i],
                fused_data[i]
            );
        }
        for i in 0..kv_dim {
            let err = (fused_data[q_dim + i] - k_data[i]).abs();
            max_err = max_err.max(err);
            assert!(
                err < 0.01,
                "K mismatch at {i}: separate={}, fused={}, err={err}",
                k_data[i],
                fused_data[q_dim + i]
            );
        }
        for i in 0..kv_dim {
            let err = (fused_data[q_dim + kv_dim + i] - v_data[i]).abs();
            max_err = max_err.max(err);
            assert!(
                err < 0.01,
                "V mismatch at {i}: separate={}, fused={}, err={err}",
                v_data[i],
                fused_data[q_dim + kv_dim + i]
            );
        }

        println!("GOAT T28 Q4_K fused QKV: max_err={max_err:.6} across {total} elements ✓");

        drop(model);
        gpu_release_pages(&client);
    }

    // ── Plan 171 T29 GOAT Proof: Q4_K fused GeGLU ────────────────────

    /// GOAT proof: Q4_K fused gate+up+GeGLU produces same output as separate dispatches.
    #[cfg(feature = "gpu_decode_fusion")]
    #[test]
    fn test_goat_q4k_fused_geglu() {
        let _heavy = heavy_model_test_gate();
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();
        // Start from a clean pool: earlier (parallel, non-gated) tests can
        // leave fully-free-but-committed pages behind (Issue 712).
        gpu_release_pages(&client);
        let config = Config::gemma2_2b();
        let weights = create_test_weights(&config);
        let model = GpuGemmaCubeCL::new_q4k(client.clone(), &weights, &config);

        let mlp = config.mlp_hidden;
        let n = config.n_embd;

        // Create a test input vector
        let input: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 10.0).collect();
        let input_handle = client.create_from_slice(f32::as_bytes(&input));

        let CubeCLWeightFormat::Q4K(w) = &model.weights else { panic!("test requires Q4_K weight format") };

        // Path 1: Separate Q4_K GEMV + GeGLU (3 dispatches)
        let gate = model.dispatch_gemv_q4k_gpu(&w.layers[0].gate_proj, input_handle.clone());
        let up = model.dispatch_gemv_q4k_gpu(&w.layers[0].up_proj, input_handle.clone());
        let separate = model.dispatch_geglu_gpu(gate, up, mlp);
        let separate_data = model.read_handle(&separate);

        // Path 2: Fused Q4_K GeGLU (1 dispatch)
        let geglu_handle = w.layers[0]
            .gate_up_combined
            .as_ref()
            .expect("fused GeGLU handle should be built");
        let fused = model.dispatch_gemv_geglu_q4k_gpu(geglu_handle, input_handle, mlp);
        let fused_data = model.read_handle(&fused);

        // Compare
        assert_eq!(fused_data.len(), separate_data.len());
        let mut max_err = 0.0f32;
        for i in 0..mlp {
            let err = (fused_data[i] - separate_data[i]).abs();
            max_err = max_err.max(err);
            assert!(
                err < 0.1,
                "GeGLU mismatch at {i}: separate={}, fused={}, err={err}",
                separate_data[i],
                fused_data[i]
            );
        }

        println!("GOAT T29 Q4_K fused GeGLU: max_err={max_err:.6} across {mlp} elements ✓");

        drop(model);
        gpu_release_pages(&client);
    }

    /// Issue 714 probe: a trivial fresh-client dispatch appended AFTER the
    /// heavy GOAT tests, so a Vulkan "Parent device is lost" event — which
    /// surfaces LAZILY on whatever API call happens next — is observed at a
    /// known, blameless location instead of silently poisoning the suite
    /// tail.
    ///
    /// `VK_ERROR_DEVICE_LOST` is reported asynchronously (wgpu#5132/#6229):
    /// the faulting operation may have happened much earlier, so a probe
    /// failure localizes the fault to the PRECEDING work (the heavy gemma2
    /// GOAT tests, or drops/frees racing in-flight submits at their
    /// teardown) — never to the probe itself.
    ///
    /// Meaningful only under `--test-threads=1` (libtest guarantees no
    /// ordering under parallelism; the `heavy_model_test_gate` guard still
    /// forces the probe behind every gated heavy test even in parallel
    /// runs). Harness: `scripts/issue_714_device_lost_repro.sh`
    /// (`gemma2` | `full` | `probe`).
    #[test]
    fn test_issue714_device_lost_probe() {
        let _heavy = heavy_model_test_gate();

        // Stage 1/3 — context init (adapter request): a lost adapter
        // surfaces here.
        let ctx = GpuContext::new().unwrap_or_else(|e| {
            panic!("Issue 714 probe stage 1/3 (context init): device lost? {e:?}")
        });

        // Stage 2/3 — fresh client from the process-shared CubeCL server
        // (Issue 676 `OnceLock`: one wgpu device per process — one loss
        // kills every later init; a fresh process recovers).
        let client = ctx.cubecl_client();

        // Stage 3/3 — trivial dispatch + readback: the same 2×2 GEMV
        // autotune path `test_cubecl_dispatch_gemv` uses. A submit-time
        // or read-time loss surfaces here.
        let weight: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let input: &[f32] = &[1.0, 1.0];
        let weight_handle = client.create_from_slice(f32::as_bytes(weight));
        let fwd = FakeGpuGemmaCubeCL {
            client: client.clone(),
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
        };
        let result = fwd.dispatch_gemv(&weight_handle, input, 2, 2);
        assert_eq!(result.len(), 2, "probe GEMV output should have 2 elements");
        assert!(
            (result[0] - 3.0).abs() < 1e-5,
            "probe GEMV[0]: expected 3.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 7.0).abs() < 1e-5,
            "probe GEMV[1]: expected 7.0, got {}",
            result[1]
        );

        println!("Issue 714 probe: device alive after preceding suite work");
    }
