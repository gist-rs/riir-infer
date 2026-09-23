//! GPU forward and generate methods for `GpuGemmaCubeCL`.
//!
//! Extracted from `mod.rs` for file hygiene (Issue 003).
//!
//! Rust allows multiple `impl` blocks for the same struct in different files,
//! so these methods extend `GpuGemmaCubeCL` defined in `mod.rs`.

use super::*;

#[allow(dead_code)] // forward_layer_gpu is all-GPU forward scaffolding (Issue 429 clippy).
impl GpuGemmaCubeCL {
    // ── GPU-resident forward pass (T2.14) ──────────────────────────
    /// Run forward pass with GPU-resident hidden state + GPU-resident KV cache (1 final sync).
    ///
    /// This is the fully GPU-resident version of [`forward`](Self::forward) that keeps
    /// both hidden state and KV cache on GPU throughout all layers. The only sync is
    /// the final logits read.
    ///
    /// # Sync Points (1 final only)
    ///
    /// | Sync | What | Why |
    /// |------|------|-----|
    /// | Final | Read logits to CPU | Apply logit softcapping on CPU |
    ///
    /// Total: 1 sync (final logits read) vs 104 (4/layer) in [`forward`](Self::forward).
    /// Improvement: ~100× fewer syncs vs CPU-hybrid, ~27× fewer vs GPU-resident+CPU-KV.
    ///
    /// # T19 Verification: Sync Elimination Confirmed
    ///
    /// `forward_layer_gpu()` contains ZERO `read_handle()` calls on the GPU KV cache
    /// hot path. The only `read_handle` in the GPU KV cache path is the final logits
    /// read in this method (step 5 below). The F16 branch has a CPU KV cache fallback
    /// that calls `read_handle`, but this is only reached when `gpu_kv_cache` is `None`.
    ///
    /// All operations use CubeCL GPU kernels — no CPU fallbacks in the hot path:
    /// - RMSNorm: `dispatch_rmsnorm_gpu` (RmsNormCubeCL)
    /// - RoPE: `dispatch_rope_gpu` / `dispatch_rope_from_combined_gpu` (RopeCubeCL)
    /// - GEMV: `dispatch_gemv_gpu` / `dispatch_gemv_f16_gpu` / `dispatch_gemv_q4k_gpu`
    /// - Attention: `dispatch_attention_gpu` (AttentionCubeCL with GPU KV cache)
    /// - KV Store: `KvStoreCubeCL` / `KvStoreKRopeVCombinedCubeCL` (GPU-resident)
    /// - GeGLU: `dispatch_geglu_gpu` / `dispatch_gemv_geglu_gpu` (fused)
    /// - Residual: `dispatch_norm_residual_gpu` (NormResidualCubeCL, fused)
    ///
    /// # GPU Kernels Used (T2.12–T2.14)
    ///
    /// | Operation | Kernel | Source |
    /// |-----------|--------|--------|
    /// | RMSNorm | `rmsnorm_f32` | `norms_cubecl.rs` (T2.12) |
    /// | RoPE | `rope_f32` | `rope_geglu_cubecl.rs` (T2.13) |
    /// | GeGLU | `geglu_f32` | `rope_geglu_cubecl.rs` (T2.13) |
    /// | Residual add | `residual_add_f32` | `norms_cubecl.rs` (T2.12) |
    /// | GEMV | `gemv_plane_f32` | `gemv_cubecl.rs` (T2.3) |
    /// | Attention | `attention_decode_f32` | `attention_cubecl.rs` (T2.5) |
    pub fn forward_gpu(&mut self, token: usize, pos: usize) -> Vec<f32> {
        // Plan 409 Phase 3: `delta_routing` early-return makes the GPU-resident
        // loop unreachable when delta_routing is enabled. The allow is for that case.
        #![allow(unreachable_code)]
        let n = self.config.n_embd;
        #[allow(unused_variables)]
        let vocab = self.config.vocab_size;
        let embed_scale = (n as f32).sqrt();
        #[allow(unused_variables)]
        let eps = self.config.rms_norm_eps as f32;

        // 1. Embedding lookup on CPU (single row, scaled by sqrt(n_embd))
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i] * embed_scale;
        }
        // Upload to GPU
        // `mut` is required when `delta_routing` is OFF (the for loop + rmsnorm
        // reassign this binding). When `delta_routing` is ON, the early return
        // makes the reassignments dead code, so clippy under `--all-features`
        // flags `mut` as unused — suppress that here.
        #[cfg_attr(feature = "delta_routing", allow(unused_mut))]
        let mut hidden_handle = self.client.create_from_slice(f32::as_bytes(&hidden));

        // 2. Process all layers.
        //
        // Plan 409 Phase 3 fix (2026-07-09): When delta_routing is enabled, the
        // fully-GPU path (`forward_layer_gpu`) cannot apply delta routing without
        // a per-layer CPU sync (delta accumulation needs the hidden state on CPU).
        // That would make it as slow as the hybrid path. So we fall back to the
        // hybrid `forward()` which correctly applies delta routing.
        //
        // When delta_routing is DISABLED, the fully-GPU path is correct (no delta
        // routing to apply) and faster (0 sync per layer).
        #[cfg(feature = "delta_routing")]
        {
            // Delta routing requires CPU-side delta accumulation — use hybrid path.
            // Read the embedding handle to CPU and delegate to forward().
            // forward() owns the Wall gate step on this path (Issue 957
            // single-step rule — stepping here too would advance the prefix
            // twice per token).
            drop(hidden_handle);
            return self.forward(token, pos);
        }
        #[cfg(feature = "wall_attention")]
        self.wall_step_if_active(&hidden);
        #[cfg(not(feature = "delta_routing"))]
        for layer_idx in 0..self.config.n_layer {
            hidden_handle = self.forward_layer_gpu(hidden_handle, layer_idx, pos);
        }

        // 3. Final RMSNorm on GPU
        hidden_handle =
            self.dispatch_rmsnorm_gpu(hidden_handle, &self.gpu_norm_gammas.final_norm, n, eps);

        // 4. LM head GEMV (tied wte): logits = wte @ hidden
        //
        // Issue 429 T4: when `lm_head_cpu` is enabled, download the hidden
        // handle to CPU and run the lm_head GEMV on CPU. This adds one sync
        // (hidden download) but eliminates the 256000-row SPIR-V/Vulkan
        // precision gap. In `forward_gpu()` this is the only extra cost —
        // the GPU lm_head GEMV would download logits anyway (step 5).
        #[cfg(feature = "lm_head_cpu")]
        let mut logits = {
            let hidden_cpu = self.read_handle(&hidden_handle);
            let mut out = vec![0.0f32; vocab];
            katgpt_core::simd::simd_matmul_rows_parallel(
                &mut out,
                &self.wte_cpu,
                &hidden_cpu,
                vocab,
                n,
            );
            out
        };
        // Issue 936: `mut` dropped on this arm — the only post-binding mutation
        // (`softcap`) is on the `lm_head_cpu` arm's binding; this one is never
        // re-borrowed mutably after the block (warns on lanes without lm_head_cpu).
        #[cfg(not(feature = "lm_head_cpu"))]
        let logits = {
            let logits_handle = match &self.weights {
                CubeCLWeightFormat::F32(w) => {
                    self.dispatch_gemv_gpu(&w.wte, hidden_handle, vocab, n)
                }
                CubeCLWeightFormat::F16(w) => {
                    self.dispatch_gemv_f16_gpu(&w.wte, hidden_handle, vocab, n)
                }
                CubeCLWeightFormat::Q4K(w) => self.dispatch_gemv_q4k_gpu(&w.wte, hidden_handle),
            };

            // 5. Final sync: read logits to CPU + softcap
            let mut logits = self.read_handle(&logits_handle);
            if self.config.final_logit_softcapping > 0.0 {
                softcap(&mut logits, self.config.final_logit_softcapping);
            }
            logits
        };

        #[cfg(feature = "lm_head_cpu")]
        if self.config.final_logit_softcapping > 0.0 {
            softcap(&mut logits, self.config.final_logit_softcapping);
        }

        logits
    }

    /// Run a forward pass and return the final-layer hidden state (1 sync, no LM head).
    ///
    /// This is the embedding-extraction path: it returns the same vector as
    /// `forward_trace_layers().1.last()` (post-last-layer, **before** the final
    /// RMSNorm) but skips both the per-layer CPU downloads and the
    /// `vocab_size × n_embd` LM-head GEMV, neither of which an embedding
    /// consumer needs.
    pub fn forward_hidden(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let embed_scale = (n as f32).sqrt();

        let mut embed = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in embed.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i] * embed_scale;
        }

        #[cfg(feature = "wall_attention")]
        self.wall_step_if_active(&embed);

        // delta_routing accumulates deltas on the CPU, so the GPU-resident
        // layer loop cannot apply it — mirror forward_gpu's hybrid fallback.
        #[cfg(feature = "delta_routing")]
        {
            let mut hidden = embed;
            for layer_idx in 0..self.config.n_layer {
                hidden = self.forward_layer(hidden, layer_idx, pos);
            }
            hidden
        }
        #[cfg(not(feature = "delta_routing"))]
        {
            let mut hidden_handle = self.client.create_from_slice(f32::as_bytes(&embed));
            for layer_idx in 0..self.config.n_layer {
                hidden_handle = self.forward_layer_gpu(hidden_handle, layer_idx, pos);
            }
            self.read_handle(&hidden_handle)
        }
    }

    /// Fully GPU-resident forward pass returning logits Handle (no download).
    ///
    /// Same as [`forward_gpu`](Self::forward_gpu) but returns the GPU Handle
    /// containing logits instead of downloading to CPU. Used by
    /// [`generate_gpu`](Self::generate_gpu) for GPU-side argmax sampling.
    ///
    /// **Note**: Logit softcapping is NOT applied. Since softcap
    /// (`tanh(x/cap) * cap`) is monotonically increasing, it doesn't
    /// change the argmax result, so it can be safely skipped for
    /// greedy decoding.
    #[cfg(feature = "gpu_decode_fusion")]
    fn forward_gpu_logits_handle(&mut self, token: usize, pos: usize) -> Handle {
        self.forward_gpu_logits_handle_max_layer(token, pos, self.config.n_layer)
    }

    /// Fully GPU-resident forward pass with early exit, returning logits Handle.
    ///
    /// Runs only `max_layer` transformer layers (instead of all `n_layer`),
    /// then applies final RMSNorm and LM head. Used for speculative decoding
    /// where the first N layers serve as a fast draft model.
    ///
    /// **KV cache behavior:** Writes KV entries for layers `0..max_layer` only.
    /// The remaining layers `max_layer..n_layer` are untouched — their KV
    /// caches retain entries from prior full-model passes.
    ///
    /// **Warning:** The hidden state after `max_layer` layers is a lower-quality
    /// representation. The draft predictions will have lower accuracy than
    /// the full model. This is intentional — the verify pass re-runs the full
    /// model to confirm.
    ///
    /// # Arguments
    ///
    /// * `token` — Input token ID
    /// * `pos` — Sequence position
    /// * `max_layer` — Number of transformer layers to run (1..n_layer).
    ///   Must be > 0 and <= `config.n_layer`.
    #[cfg(feature = "gpu_decode_fusion")]
    pub(super) fn forward_gpu_logits_handle_max_layer(
        &mut self,
        token: usize,
        pos: usize,
        max_layer: usize,
    ) -> Handle {
        // `delta_routing` early-return (Issue 718) makes the GPU-resident
        // loop unreachable when the feature is enabled — mirror forward_gpu's
        // allow for that case.
        #![allow(unreachable_code)]
        assert!(
            max_layer > 0 && max_layer <= self.config.n_layer,
            "max_layer ({max_layer}) must be in 1..={}",
            self.config.n_layer
        );
        let n = self.config.n_embd;
        #[allow(unused_variables)]
        let vocab = self.config.vocab_size;
        let embed_scale = (n as f32).sqrt();
        #[allow(unused_variables)]
        let eps = self.config.rms_norm_eps as f32;

        // 1. Embedding lookup on CPU (single row, scaled by sqrt(n_embd))
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i] * embed_scale;
        }
        // Upload to GPU
        // `mut` is required when `delta_routing` is OFF (the layer loop
        // reassigns this binding); when ON the early return makes the
        // reassignments dead code — same suppression as forward_gpu.
        #[cfg_attr(feature = "delta_routing", allow(unused_mut))]
        let mut hidden_handle = self.client.create_from_slice(f32::as_bytes(&hidden));

        // Wall Attention gate step (CPU-side, Plan 193 T3) — MOVED below the
        // delta_routing delegation: forward() owns the step on that path
        // (Issue 957 single-step rule).

        // Issue 718: delta_routing requires CPU-side delta accumulation — the
        // GPU-resident layer loop below cannot apply it (same constraint as
        // `forward_gpu`, which delegates to the hybrid `forward()`).
        // Previously this method had NO fallback, so `generate_gpu()` decoded
        // through a layer stack that never applied delta routing whenever the
        // feature was enabled — which it is by DEFAULT in riir-gpu (added to
        // defaults in eddac8c50 for struct-layout sync). The GOAT gate
        // (test_goat_full_pipeline_decode) caught it as a 3.06-logit divergence
        // once the cubecl_runtime,gpu_decode_fusion combo compiled again.
        // Delegate to the hybrid forward() and re-upload the logits to
        // preserve the Handle return type (mirrors the lm_head_cpu pattern).
        // NOTE: this ignores `max_layer` — the speculative-decode draft loses
        // its early-exit speedup under delta_routing but stays CORRECT (the
        // verify loop re-runs the full model).
        #[cfg(feature = "delta_routing")]
        {
            drop(hidden_handle);
            let logits = self.forward(token, pos);
            return self.client.create_from_slice(f32::as_bytes(&logits));
        }

        // Wall Attention gate step — fully-GPU arm only (Plan 193 T3; the
        // delegation above lets forward() own the step).
        #[cfg(feature = "wall_attention")]
        self.wall_step_if_active(&hidden);

        // 2. Process layers 0..max_layer (GPU-resident, 0 sync per layer)
        for layer_idx in 0..max_layer {
            hidden_handle = self.forward_layer_gpu(hidden_handle, layer_idx, pos);
        }

        // 3. Final RMSNorm on GPU
        hidden_handle =
            self.dispatch_rmsnorm_gpu(hidden_handle, &self.gpu_norm_gammas.final_norm, n, eps);

        // 4. LM head GEMV (tied wte): logits = wte @ hidden
        // Return handle without downloading — caller will run argmax on GPU
        //
        // Issue 429 T4: when `lm_head_cpu` is enabled, compute logits on CPU
        // then re-upload to a GPU handle. This preserves the Handle return type
        // (needed by gpu_decode_fusion's GPU-side argmax) at the cost of one
        // hidden download + one logits upload. Note: this combination
        // (lm_head_cpu + gpu_decode_fusion) partially defeats decode fusion's
        // 4-byte-download optimization — but correct logits > fast garbage.
        #[cfg(feature = "lm_head_cpu")]
        {
            let hidden_cpu = self.read_handle(&hidden_handle);
            let mut logits_cpu = vec![0.0f32; vocab];
            katgpt_core::simd::simd_matmul_rows_parallel(
                &mut logits_cpu,
                &self.wte_cpu,
                &hidden_cpu,
                vocab,
                n,
            );
            self.client.create_from_slice(f32::as_bytes(&logits_cpu))
        }
        #[cfg(not(feature = "lm_head_cpu"))]
        match &self.weights {
            CubeCLWeightFormat::F32(w) => self.dispatch_gemv_gpu(&w.wte, hidden_handle, vocab, n),
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_gemv_f16_gpu(&w.wte, hidden_handle, vocab, n)
            }
            CubeCLWeightFormat::Q4K(w) => self.dispatch_gemv_q4k_gpu(&w.wte, hidden_handle),
        }
    }

    /// GPU-side argmax: finds the index of the maximum logit without downloading.
    ///
    /// Downloads only 4 bytes (1 u32 = token ID) instead of
    /// `vocab_size * 4` bytes (~1 MB for Gemma 2 2B).
    ///
    /// # Panics
    ///
    /// Panics if the GPU read fails.
    #[cfg(feature = "gpu_decode_fusion")]
    fn gpu_argmax_from_handle(&self, logits_handle: Handle) -> usize {
        let output_handle = self.client.empty(core::mem::size_of::<u32>());
        // SAFETY: logits_handle contains vocab_size f32 elements,
        // output_handle is pre-allocated with 4 bytes (1 u32).
        unsafe {
            ArgmaxCubeCL::launch::<ActiveRuntime>(
                &self.client,
                logits_handle,
                output_handle.clone(),
                self.config.vocab_size,
            );
        }
        let bytes = self
            .client
            .read_one(output_handle)
            .unwrap_or_else(|e| panic!("GPU argmax read failed: {e}"));
        let result = u32::from_bytes(&bytes);
        result[0] as usize
    }

    /// GPU-resident autoregressive generation with GPU-side argmax (Plan 171 T3).
    ///
    /// Uses [`forward_gpu`](Self::forward_gpu) for prefill (downloads logits for
    /// the first generated token) and `forward_gpu_logits_handle` + GPU argmax
    /// for the decode loop. Each decode token downloads only 4 bytes (token ID)
    /// instead of ~1 MB (full logits).
    ///
    /// # Performance
    ///
    /// | Phase | Tokens | Download/Token | Total Download |
    /// |-------|--------|----------------|----------------|
    /// | Prefill | `prompt_len` | ~1 MB | ~`prompt_len` MB |
    /// | Decode | `max_tokens` | 4 B | ~`max_tokens * 4` B |
    ///
    /// For a typical prompt of 50 tokens + 500 generated tokens:
    /// - Old: 550 × 1 MB = 550 MB downloaded
    /// - New: 50 × 1 MB + 500 × 4 B ≈ 50 MB downloaded (10× reduction)
    ///
    /// # Arguments
    ///
    /// - `prompt_tokens` — input token IDs (must be non-empty).
    /// - `max_tokens` — maximum number of tokens to generate (excluding prompt).
    ///
    /// # Returns
    ///
    /// Generated token IDs (excluding prompt tokens). Stops on EOS (token 1).
    #[cfg(feature = "gpu_decode_fusion")]
    pub fn generate_gpu(&mut self, prompt_tokens: &[usize], max_tokens: usize) -> Vec<usize> {
        if prompt_tokens.is_empty() {
            return Vec::new();
        }
        let mut tokens = Vec::with_capacity(max_tokens);
        let eos_token = 1; // Gemma 2 EOS

        // ── Prefill: process prompt tokens ──
        // For intermediate tokens, we eat the full logits download (prefill is
        // one-time cost, not per-token bottleneck). For the last prompt token,
        // we also eat the download and use CPU argmax since it's just one token.
        //
        // Future optimization: add forward_gpu variant that skips logits download
        // for intermediate tokens (would need a "run layers only" mode).
        let mut last_logits = Vec::new();
        for (i, &token) in prompt_tokens.iter().enumerate() {
            last_logits = self.forward_gpu(token, i);
        }

        // First generated token: CPU argmax on last prefill logits
        // (we already downloaded them, so use them directly)
        let first_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i);

        if first_token == eos_token {
            return tokens;
        }
        tokens.push(first_token);

        // ── Decode loop: GPU forward + GPU argmax (4-byte download per token) ──
        for _ in 1..max_tokens {
            let last_token = *tokens.last().unwrap();
            let pos = prompt_tokens.len() + tokens.len() - 1;

            let logits_handle = self.forward_gpu_logits_handle(last_token, pos);
            let next_token = self.gpu_argmax_from_handle(logits_handle);

            if next_token == eos_token {
                break;
            }
            tokens.push(next_token);
        }

        tokens
    }

    /// GPU-resident speculative decoding with early-exit draft (Plan 171 T34).
    ///
    /// Uses the first `draft_layers` transformer layers as a fast draft model
    /// to predict K tokens, then verifies each with the full model. Accepted
    /// draft tokens cost `draft_layers/26` of a full forward pass; rejected
    /// tokens are re-computed by the verifier.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// loop:
    ///   1. Draft: run layers 0..draft_layers for K steps, argmax each
    ///   2. Verify: run full model at each draft position, compare argmax
    ///   3. Accept matching prefix, reject divergence point
    ///   4. Continue from first rejected position (or after all accepted)
    /// ```
    ///
    /// # Performance Model
    ///
    /// | Acceptance Rate | Effective Speedup (K=4) |
    /// |-----------------|----------------------|
    /// | 80%             | ~2.0×                |
    /// | 70%             | ~1.7×                |
    /// | 50%             | ~1.3×                |
    ///
    /// Draft cost per token: `draft_layers / n_layer` of full forward.
    /// For `draft_layers=4, n_layer=26`: draft is ~15% of full forward.
    /// Net per speculative round: `K * 0.15 + 1` full forward equivalents.
    ///
    /// # Arguments
    ///
    /// * `prompt_tokens` — Input token IDs (must be non-empty).
    /// * `max_tokens` — Maximum tokens to generate (excluding prompt).
    /// * `draft_lookahead` — Number of draft tokens per speculation round (K).
    ///   Typical values: 3-5. Higher K benefits from higher acceptance rates.
    /// * `draft_layers` — Number of transformer layers for draft model.
    ///   Must be > 0 and < `config.n_layer`. Typical: 4 (first ~15% of model).
    ///
    /// # Returns
    ///
    /// [`SpeculativeResult`] with generated tokens and acceptance statistics.
    #[cfg(feature = "gpu_decode_fusion")]
    pub fn generate_gpu_speculative(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        draft_lookahead: usize,
        draft_layers: usize,
    ) -> SpeculativeResult {
        assert!(!prompt_tokens.is_empty(), "prompt_tokens must be non-empty");
        assert!(draft_lookahead > 0, "draft_lookahead must be > 0");
        assert!(
            draft_layers > 0 && draft_layers < self.config.n_layer,
            "draft_layers ({draft_layers}) must be in 1..{}",
            self.config.n_layer - 1
        );

        let eos_token = 1; // Gemma 2 EOS
        let mut tokens: Vec<usize> = Vec::with_capacity(max_tokens);
        let mut total_draft_tokens = 0usize;
        let mut total_accepted = 0usize;
        let mut speculation_rounds = 0usize;

        // ── Prefill: process prompt tokens through full model ──
        let mut last_logits = Vec::new();
        for (i, &token) in prompt_tokens.iter().enumerate() {
            last_logits = self.forward_gpu(token, i);
        }

        // First generated token: CPU argmax on last prefill logits
        let first_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i);

        if first_token == eos_token || max_tokens == 0 {
            return SpeculativeResult {
                tokens: Vec::new(),
                total_draft_tokens: 0,
                total_accepted: 0,
                speculation_rounds: 0,
            };
        }
        tokens.push(first_token);

        // ── Speculative decode loop ──
        while tokens.len() < max_tokens {
            speculation_rounds += 1;
            let base_pos = prompt_tokens.len() + tokens.len() - 1;

            // ── Phase 1: Draft K tokens using early-exit model ──
            let mut draft_tokens: Vec<usize> = Vec::with_capacity(draft_lookahead);

            for (draft_pos, k) in (base_pos..).zip(0..draft_lookahead) {
                let input_token = if k == 0 {
                    *tokens.last().unwrap()
                } else {
                    *draft_tokens.last().unwrap()
                };

                let logits_handle =
                    self.forward_gpu_logits_handle_max_layer(input_token, draft_pos, draft_layers);
                let draft_token = self.gpu_argmax_from_handle(logits_handle);

                if draft_token == eos_token {
                    break;
                }
                draft_tokens.push(draft_token);
            }

            total_draft_tokens += draft_tokens.len();

            if draft_tokens.is_empty() {
                // Draft produced EOS — generate one fallback token from full model
                let logits_handle =
                    self.forward_gpu_logits_handle(*tokens.last().unwrap(), base_pos);
                let next = self.gpu_argmax_from_handle(logits_handle);
                if next == eos_token {
                    break;
                }
                tokens.push(next);
                continue;
            }

            // ── Phase 2: Verify draft tokens against full model ──
            //
            // Run the full model at each draft position. The KV cache for
            // layers 0..draft_layers already has entries from the draft pass;
            // the verify pass overwrites them. Layers draft_layers..n_layer
            // are populated only by the verify pass (they have prefix entries
            // from the prefill and prior accepted tokens).
            //
            // We verify one token at a time to stop at the first rejection.
            // This means we run the full model sequentially at positions
            // base_pos+1..base_pos+1+n_accepted.
            let mut n_accepted_this_round = 0;

            for (k, &draft_tok) in draft_tokens.iter().enumerate() {
                let verify_pos = base_pos + 1 + k;

                // Run full model at verify_pos. Input is the *accepted* token at
                // the previous position.
                let input_token = if k == 0 {
                    *tokens.last().unwrap()
                } else {
                    draft_tokens[k - 1] // Previous draft token (already verified)
                };

                let logits_handle = self.forward_gpu_logits_handle(input_token, verify_pos);
                let verify_token = self.gpu_argmax_from_handle(logits_handle);

                if verify_token == eos_token {
                    // Verifier hit EOS — stop generation
                    // (Don't add EOS to output per convention)
                    break;
                }

                if verify_token == draft_tok {
                    // Draft prediction confirmed
                    n_accepted_this_round += 1;
                    total_accepted += 1;
                    tokens.push(verify_token);

                    if tokens.len() >= max_tokens {
                        break;
                    }
                } else {
                    // Rejection: take the verifier's token instead
                    tokens.push(verify_token);
                    break;
                }
            }

            // If all draft tokens were accepted and we still have budget,
            // also take the verifier's bonus token at position base_pos + 1 + K
            // (the token AFTER the last draft position).
            if n_accepted_this_round == draft_tokens.len()
                && tokens.len() < max_tokens
                && n_accepted_this_round > 0
            {
                let bonus_pos = base_pos + 1 + draft_tokens.len();
                let last_accepted = *tokens.last().unwrap();
                let logits_handle = self.forward_gpu_logits_handle(last_accepted, bonus_pos);
                let bonus_token = self.gpu_argmax_from_handle(logits_handle);

                if bonus_token != eos_token {
                    tokens.push(bonus_token);
                }
            }

            if tokens.len() >= max_tokens {
                break;
            }
        }

        SpeculativeResult {
            tokens,
            total_draft_tokens,
            total_accepted,
            speculation_rounds,
        }
    }

    /// Single Gemma 2 transformer layer with GPU-resident hidden state + KV cache (0 syncs).
    ///
    /// # Compute Pass Structure (fully GPU-resident, no CPU sync)
    ///
    /// ```text
    /// [GPU] RMSNorm(hidden, input_norm)
    /// [GPU] GEMV Q + K + V → q, k, v handles
    /// [GPU] RoPE(q, k) → q_rope, k_rope handles
    /// [GPU] KV Store(k_rope, v → gpu_cache) → updated cache
    /// [GPU] KV Compact(gpu_cache → compact_temp) → compact view for attention
    /// [GPU] Attention(q_rope, compact_temp) → attn_out handle
    /// [GPU] GEMV Wo(attn_out) → wo handle
    /// [GPU] RMSNorm(wo, post_attn_norm)
    /// [GPU] residual add(wo_normed, residual) → hidden handle
    /// [GPU] RMSNorm(hidden, pre_mlp_norm)
    /// [GPU] GEMV gate + up → gate, up handles
    /// [GPU] GeGLU(gate, up) → mlp_hidden handle
    /// [GPU] GEMV down(mlp_hidden) → down handle
    /// [GPU] RMSNorm(down, post_mlp_norm)
    /// [GPU] residual add(down_normed, residual2) → hidden_out handle
    /// ```
    fn forward_layer_gpu(&mut self, hidden_handle: Handle, layer_idx: usize, pos: usize) -> Handle {
        let n = self.config.n_embd;
        let q_dim = self.config.n_head * self.config.head_dim;
        let kv_dim = self.config.n_kv_head * self.config.head_dim;
        let mlp = self.config.mlp_hidden;
        let eps = self.config.rms_norm_eps as f32;
        let norms = &self.gpu_norm_gammas.layers[layer_idx];

        // Save residual for post-norm add
        let residual_handle = hidden_handle.clone();

        // ── Pass A: RMSNorm + QKV + Positional Encoding (RoPE or Wall) ──

        let normed_handle = self.dispatch_rmsnorm_gpu(hidden_handle, &norms.input_norm, n, eps);

        // Wall prefix buffer (only computed when Wall is active).
        #[cfg(feature = "wall_attention")]
        let wall_prefix = self.wall_state.as_ref().and_then(|s| {
            if s.is_active() {
                Some(s.prefix_qk_buffer())
            } else {
                None
            }
        });

        // GPU: QKV GEMVs + Positional Encoding + KV Store
        // F16 path uses fused triple QKV GEMV (1 dispatch instead of 3) +
        // offset-aware encoding + combined KV store.
        let (q_rope_handle, _gpu_kv_stored) = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                let q = self.dispatch_gemv_gpu(
                    &w.layers[layer_idx].attn_wq,
                    normed_handle.clone(),
                    q_dim,
                    n,
                );
                let k = self.dispatch_gemv_gpu(
                    &w.layers[layer_idx].attn_wk,
                    normed_handle.clone(),
                    kv_dim,
                    n,
                );
                let v =
                    self.dispatch_gemv_gpu(&w.layers[layer_idx].attn_wv, normed_handle, kv_dim, n);

                // Branch: Wall rescale vs RoPE rotation
                let (q_pe, k_pe) = {
                    #[cfg(feature = "wall_attention")]
                    if let Some(ref prefix) = wall_prefix {
                        self.dispatch_wall_rescale_gpu(q, k, prefix, q_dim, kv_dim)
                    } else {
                        (
                            self.dispatch_rope_gpu(q, pos, self.config.n_head),
                            self.dispatch_rope_gpu(k, pos, self.config.n_kv_head),
                        )
                    }
                    #[cfg(not(feature = "wall_attention"))]
                    {
                        (
                            self.dispatch_rope_gpu(q, pos, self.config.n_head),
                            self.dispatch_rope_gpu(k, pos, self.config.n_kv_head),
                        )
                    }
                };
                // Store K, V in GPU KV cache
                if let Some(ref gpu_cache) = self.gpu_kv_cache {
                    unsafe {
                        KvStoreCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            k_pe.clone(),
                            v,
                            gpu_cache.cache_handle(layer_idx).clone(),
                            gpu_cache.kv_stride(),
                            pos,
                            gpu_cache.block_size(),
                        );
                    }
                }
                (q_pe, self.gpu_kv_cache.is_some())
            }
            CubeCLWeightFormat::F16(w) => {
                // Fused triple QKV GEMV: 1 dispatch instead of 3 (saves 2 dispatches/layer)
                let total_qkv = q_dim + 2 * kv_dim;
                let qkv_combined = self.dispatch_gemv_qkv_f16_gpu(
                    &w.layers[layer_idx].qkv_combined,
                    normed_handle,
                    q_dim,
                    kv_dim,
                    n,
                );

                // Offset-aware positional encoding on Q section (offset=0, len=q_dim)
                let q_pe = {
                    #[cfg(feature = "wall_attention")]
                    if let Some(ref prefix) = wall_prefix {
                        self.dispatch_wall_from_combined_gpu(
                            qkv_combined.clone(),
                            total_qkv,
                            prefix,
                            0,
                            q_dim,
                            false,
                        )
                    } else {
                        self.dispatch_rope_from_combined_gpu(
                            qkv_combined.clone(),
                            total_qkv,
                            0,
                            q_dim,
                            pos,
                            self.config.n_head,
                        )
                    }
                    #[cfg(not(feature = "wall_attention"))]
                    {
                        self.dispatch_rope_from_combined_gpu(
                            qkv_combined.clone(),
                            total_qkv,
                            0,
                            q_dim,
                            pos,
                            self.config.n_head,
                        )
                    }
                };

                // Offset-aware positional encoding on K section (offset=q_dim, len=kv_dim)
                let k_pe = {
                    #[cfg(feature = "wall_attention")]
                    if let Some(ref prefix) = wall_prefix {
                        self.dispatch_wall_from_combined_gpu(
                            qkv_combined.clone(),
                            total_qkv,
                            prefix,
                            q_dim,
                            kv_dim,
                            true,
                        )
                    } else {
                        self.dispatch_rope_from_combined_gpu(
                            qkv_combined.clone(),
                            total_qkv,
                            q_dim,
                            kv_dim,
                            pos,
                            self.config.n_kv_head,
                        )
                    }
                    #[cfg(not(feature = "wall_attention"))]
                    {
                        self.dispatch_rope_from_combined_gpu(
                            qkv_combined.clone(),
                            total_qkv,
                            q_dim,
                            kv_dim,
                            pos,
                            self.config.n_kv_head,
                        )
                    }
                };

                // Store K (positionally encoded) and V (from combined buffer) into KV cache
                if let Some(ref gpu_cache) = self.gpu_kv_cache {
                    // GPU KV cache: store directly from handles (zero CPU sync)
                    unsafe {
                        KvStoreKRopeVCombinedCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            k_pe,
                            kv_dim,
                            qkv_combined,
                            total_qkv,
                            gpu_cache.cache_handle(layer_idx).clone(),
                            gpu_cache.kv_stride(),
                            pos,
                            gpu_cache.block_size(),
                            q_dim,
                        );
                    }
                } else {
                    // CPU KV cache fallback: read combined buffer, extract K and V sections
                    let k_pe_data = self.read_handle(&k_pe);
                    let qkv_all = self.read_handle(&qkv_combined);
                    let v_start = q_dim + kv_dim;
                    let v_data = &qkv_all[v_start..v_start + kv_dim];
                    #[cfg(feature = "q8_kv_cache")]
                    if let Some(ref mut q8_cache) = self.kv_cache_q8 {
                        q8_cache.store(layer_idx, pos, &k_pe_data, v_data);
                    } else {
                        self.kv_cache.store(layer_idx, pos, &k_pe_data, v_data);
                    }
                    #[cfg(not(feature = "q8_kv_cache"))]
                    self.kv_cache.store(layer_idx, pos, &k_pe_data, v_data);
                }
                (q_pe, self.gpu_kv_cache.is_some())
            }
            CubeCLWeightFormat::Q4K(w) => {
                // Try fused triple QKV GEMV first (Plan 171 T28), fallback to separate
                if let Some(ref qkv_handle) = w.layers[layer_idx].qkv_combined {
                    let total_qkv = q_dim + 2 * kv_dim;
                    let qkv_combined = self.dispatch_gemv_qkv_q4k_gpu(qkv_handle, normed_handle);
                    let q_pe = {
                        #[cfg(feature = "wall_attention")]
                        if let Some(ref prefix) = wall_prefix {
                            self.dispatch_wall_from_combined_gpu(
                                qkv_combined.clone(),
                                total_qkv,
                                prefix,
                                0,
                                q_dim,
                                false,
                            )
                        } else {
                            self.dispatch_rope_from_combined_gpu(
                                qkv_combined.clone(),
                                total_qkv,
                                0,
                                q_dim,
                                pos,
                                self.config.n_head,
                            )
                        }
                        #[cfg(not(feature = "wall_attention"))]
                        {
                            self.dispatch_rope_from_combined_gpu(
                                qkv_combined.clone(),
                                total_qkv,
                                0,
                                q_dim,
                                pos,
                                self.config.n_head,
                            )
                        }
                    };
                    let k_pe = {
                        #[cfg(feature = "wall_attention")]
                        if let Some(ref prefix) = wall_prefix {
                            self.dispatch_wall_from_combined_gpu(
                                qkv_combined.clone(),
                                total_qkv,
                                prefix,
                                q_dim,
                                kv_dim,
                                true,
                            )
                        } else {
                            self.dispatch_rope_from_combined_gpu(
                                qkv_combined.clone(),
                                total_qkv,
                                q_dim,
                                kv_dim,
                                pos,
                                self.config.n_kv_head,
                            )
                        }
                        #[cfg(not(feature = "wall_attention"))]
                        {
                            self.dispatch_rope_from_combined_gpu(
                                qkv_combined.clone(),
                                total_qkv,
                                q_dim,
                                kv_dim,
                                pos,
                                self.config.n_kv_head,
                            )
                        }
                    };
                    // Store K, V in GPU KV cache
                    if let Some(ref gpu_cache) = self.gpu_kv_cache {
                        unsafe {
                            KvStoreKRopeVCombinedCubeCL::launch::<ActiveRuntime>(
                                &self.client,
                                k_pe.clone(),
                                kv_dim,
                                qkv_combined,
                                total_qkv,
                                gpu_cache.cache_handle(layer_idx).clone(),
                                gpu_cache.kv_stride(),
                                pos,
                                gpu_cache.block_size(),
                                q_dim,
                            );
                        }
                    }
                    (q_pe, self.gpu_kv_cache.is_some())
                } else {
                    // Fallback: 3 separate Q4_K GEMV dispatches
                    let q = self
                        .dispatch_gemv_q4k_gpu(&w.layers[layer_idx].attn_wq, normed_handle.clone());
                    let k = self
                        .dispatch_gemv_q4k_gpu(&w.layers[layer_idx].attn_wk, normed_handle.clone());
                    let v = self.dispatch_gemv_q4k_gpu(&w.layers[layer_idx].attn_wv, normed_handle);
                    let (q_pe, k_pe) = {
                        #[cfg(feature = "wall_attention")]
                        if let Some(ref prefix) = wall_prefix {
                            self.dispatch_wall_rescale_gpu(q, k, prefix, q_dim, kv_dim)
                        } else {
                            (
                                self.dispatch_rope_gpu(q, pos, self.config.n_head),
                                self.dispatch_rope_gpu(k, pos, self.config.n_kv_head),
                            )
                        }
                        #[cfg(not(feature = "wall_attention"))]
                        {
                            (
                                self.dispatch_rope_gpu(q, pos, self.config.n_head),
                                self.dispatch_rope_gpu(k, pos, self.config.n_kv_head),
                            )
                        }
                    };
                    // Store K, V in GPU KV cache
                    if let Some(ref gpu_cache) = self.gpu_kv_cache {
                        unsafe {
                            KvStoreCubeCL::launch::<ActiveRuntime>(
                                &self.client,
                                k_pe.clone(),
                                v,
                                gpu_cache.cache_handle(layer_idx).clone(),
                                gpu_cache.kv_stride(),
                                pos,
                                gpu_cache.block_size(),
                            );
                        }
                    }
                    (q_pe, self.gpu_kv_cache.is_some())
                }
            }
        };

        // ── Pass B: Attention + Wo + RMSNorm + residual ──────────────

        // GPU: Attention
        let attn_out_handle = self.dispatch_attention_gpu(q_rope_handle, layer_idx, pos);

        // GPU: Wo GEMV (uses attn_out directly, no intermediate sync)
        let wo_handle = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_gemv_gpu(&w.layers[layer_idx].attn_wo, attn_out_handle, n, q_dim)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_gemv_f16_gpu(&w.layers[layer_idx].attn_wo, attn_out_handle, n, q_dim)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_gemv_q4k_gpu(&w.layers[layer_idx].attn_wo, attn_out_handle)
            }
        };

        // GPU: Fused RMSNorm + ResidualAdd (1 dispatch instead of 2)
        let hidden_handle = self.dispatch_norm_residual_gpu(
            wo_handle,
            &norms.post_attn_norm,
            residual_handle,
            n,
            eps,
        );

        // ── Pass C: MLP ──────────────────────────────────────────────

        let residual2_handle = hidden_handle.clone();

        // GPU: RMSNorm(hidden, pre_mlp_norm)
        let normed2 = self.dispatch_rmsnorm_gpu(hidden_handle, &norms.pre_mlp_norm, n, eps);

        // GPU: Gate + Up GEMVs + GeGLU (F32 and F16 use fused single dispatch, Q4K uses separate)
        let mlp_handle = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                // Fused gate+up GEMV + GeGLU in single dispatch (saves 2 dispatches/layer)
                self.dispatch_gemv_geglu_gpu(
                    &w.layers[layer_idx].gate_proj,
                    &w.layers[layer_idx].up_proj,
                    normed2,
                    mlp,
                    n,
                )
            }
            CubeCLWeightFormat::F16(w) => {
                // Fused gate+up GEMV + GeGLU in single dispatch (saves 2 dispatches/layer)
                self.dispatch_gemv_geglu_f16_gpu(
                    &w.layers[layer_idx].gate_proj,
                    &w.layers[layer_idx].up_proj,
                    normed2,
                    mlp,
                    n,
                )
            }
            CubeCLWeightFormat::Q4K(w) => {
                // Try fused gate+up+GeGLU first (Plan 171 T29), fallback to separate
                if let Some(ref geglu_handle) = w.layers[layer_idx].gate_up_combined {
                    self.dispatch_gemv_geglu_q4k_gpu(geglu_handle, normed2, mlp)
                } else {
                    let gate =
                        self.dispatch_gemv_q4k_gpu(&w.layers[layer_idx].gate_proj, normed2.clone());
                    let up = self.dispatch_gemv_q4k_gpu(&w.layers[layer_idx].up_proj, normed2);
                    self.dispatch_geglu_gpu(gate, up, mlp)
                }
            }
        };

        // GPU: Down GEMV
        let down_handle = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_gemv_gpu(&w.layers[layer_idx].down_proj, mlp_handle, n, mlp)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_gemv_f16_gpu(&w.layers[layer_idx].down_proj, mlp_handle, n, mlp)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_gemv_q4k_gpu(&w.layers[layer_idx].down_proj, mlp_handle)
            }
        };

        // GPU: Fused RMSNorm + ResidualAdd (1 dispatch instead of 2)
        self.dispatch_norm_residual_gpu(down_handle, &norms.post_mlp_norm, residual2_handle, n, eps)
    }

    // ── Sync helpers ────────────────────────────────────────────────

    /// Read a CubeCL Handle to CPU as `Vec<f32>`.
    ///
    /// Blocks until the GPU kernel producing this handle's data completes.
    /// On Apple Silicon unified memory, this is a fast memory-mapped read.
    ///
    /// # Panics
    ///
    /// Panics if the GPU read fails (indicates a programming bug or
    /// GPU device loss, not a normal runtime condition).
    pub fn read_handle(&self, handle: &Handle) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(handle.clone())
            .unwrap_or_else(|e| panic!("CubeCL buffer read failed: {e}"));
        f32::from_bytes(&bytes).to_vec()
    }
}
