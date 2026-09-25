//! CubeCL dispatch helper methods for `GpuGemmaCubeCL`.
//!
//! Extracted from `mod.rs` for file hygiene (Issue 003). These are the low-level
//! CubeCL kernel dispatch helpers that launch GEMV, attention, RoPE, GeGLU,
//! and residual-add kernels on the GPU.
//!
//! Rust allows multiple `impl` blocks for the same struct in different files,
//! so these methods simply extend `GpuGemmaCubeCL` defined in `mod.rs`.

use super::*;

#[allow(dead_code)] // GPU dispatch scaffolding — called under additional features (Issue 429 clippy).
impl GpuGemmaCubeCL {
    // ── CubeCL dispatch helpers ─────────────────────────────────────

    /// Launch CubeCL GEMV: `output[M] = weight[M,N] @ input[N]`.
    ///
    /// Creates input handle from CPU data, allocates output handle,
    /// launches kernel, reads result back to CPU.
    pub fn dispatch_gemv(
        &self,
        weight: &Handle,
        input: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        let input_handle = self.client.create_from_slice(f32::as_bytes(input));
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());

        // SAFETY: weight has M×N elements, input has N elements,
        // output is pre-allocated with M elements.
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

        self.read_handle(&output_handle)
    }

    /// Launch CubeCL f16 weight GEMV: `output[M] = weight_f16[M,N] @ input[N]`.
    ///
    /// Creates input handle from CPU data, allocates output handle,
    /// launches f16→f32 kernel, reads result back to CPU.
    pub(super) fn dispatch_gemv_f16(&self, weight: &F16Handle, input: &[f32]) -> Vec<f32> {
        let input_handle = self.client.create_from_slice(f32::as_bytes(input));
        let output_handle = self.client.empty(weight.m * core::mem::size_of::<f32>());

        // SAFETY: F16Handle has correct m, n and weight buffer is m*n f16 elements.
        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight.weight.clone(),
                input_handle,
                output_handle.clone(),
                weight.m,
                weight.n,
            );
        }

        self.read_handle(&output_handle)
    }

    /// Launch batched Q/K/V GEMVs (3 launches, 1 sync).
    ///
    /// All three GEMVs share the same input (hidden state).
    /// CubeCL batches them into a single GPU submission.
    ///
    /// Returns `(q, k, v)` vectors.
    pub fn dispatch_qkv(
        &self,
        layer_weights: &CubeCLLayerWeights,
        hidden: &[f32],
        q_dim: usize,
        kv_dim: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let q_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let k_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());
        let v_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());

        // SAFETY: All buffer sizes match the (m, n) dimensions.
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wq.clone(),
                input_handle.clone(),
                q_handle.clone(),
                q_dim,
                n_embd,
            );
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wk.clone(),
                input_handle.clone(),
                k_handle.clone(),
                kv_dim,
                n_embd,
            );
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wv.clone(),
                input_handle,
                v_handle.clone(),
                kv_dim,
                n_embd,
            );
        }

        // Single sync: read all outputs (first read triggers batch execution)
        let q = self.read_handle(&q_handle);
        let k = self.read_handle(&k_handle);
        let v = self.read_handle(&v_handle);

        (q, k, v)
    }

    /// Launch batched attention + Wo GEMV (2 launches, 1 sync).
    ///
    /// The attention kernel writes to attn_out_handle, which is passed
    /// directly as input to the Wo GEMV without CPU round-trip.
    /// CubeCL's stream scheduler ensures correct ordering.
    ///
    /// Returns Wo output `[n_embd]`.
    pub(super) fn dispatch_attention_wo(
        &self,
        layer_weights: &CubeCLLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
    ) -> Vec<f32> {
        let n_positions = pos + 1;
        let q_dim = self.config.n_head * self.config.head_dim;
        let trace_attn =
            layer_idx == 0 && pos <= 1 && std::env::var("RIIR_GPU_TRACE").as_deref() == Ok("1");

        // Build combined KV buffer from CPU cache
        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, n_positions);
        if trace_attn {
            let kv_stride = self.kv_cache.kv_stride;
            let kv_half = combined_kv.len() / 2;
            let kv_n_pos = kv_half / kv_stride;
            let keys_norm: f32 = combined_kv[..kv_half]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let vals_norm: f32 = combined_kv[kv_half..]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let k_first4: Vec<f32> = combined_kv[..4].to_vec();
            let v_first4: Vec<f32> = combined_kv[kv_half..kv_half + 4].to_vec();
            let k_pos1_first4: Vec<f32> = if kv_n_pos > 1 {
                combined_kv[kv_stride..kv_stride + 4].to_vec()
            } else {
                vec![]
            };
            println!(
                "  combined_kv: len={} kv_half={} n_pos={} keys_norm={:.4} vals_norm={:.4}",
                combined_kv.len(),
                kv_half,
                kv_n_pos,
                keys_norm,
                vals_norm,
            );
            println!("    k_pos0_first4={k_first4:?}");
            println!("    v_pos0_first4={v_first4:?}");
            if !k_pos1_first4.is_empty() {
                println!("    k_pos1_first4={k_pos1_first4:?}");
            }
        }

        // Create handles
        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        // Launch attention
        let mut params = self.attn_params;
        params.n_positions = n_positions;
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        if trace_attn {
            // Debug path: sync after attention to inspect attn_out
            let attn_out = self.read_handle(&attn_out_handle);
            let attn_norm: f32 = attn_out.iter().map(|x| x * x).sum::<f32>().sqrt();
            let attn_first4: Vec<f32> = attn_out[..4].to_vec();
            let attn_min = attn_out.iter().cloned().fold(f32::INFINITY, f32::min);
            let attn_max = attn_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "    attn_out: len={} [{:+.4}, {:+.4}] norm={:.4} first4={attn_first4:?}",
                attn_out.len(),
                attn_min,
                attn_max,
                attn_norm,
            );

            // Re-upload attn_out for Wo GEMV
            let attn_rehandle = self.client.create_from_slice(f32::as_bytes(&attn_out));
            unsafe {
                self.gemv_autotune.launch::<ActiveRuntime>(
                    &self.client,
                    layer_weights.attn_wo.clone(),
                    attn_rehandle,
                    wo_out_handle.clone(),
                    n_embd,
                    q_dim,
                );
            }
        } else {
            // Fused path: attention → Wo without intermediate sync
            // SAFETY: attn_wo is [n_embd, q_dim], attn_out is [q_dim], wo_out is [n_embd]
            unsafe {
                self.gemv_autotune.launch::<ActiveRuntime>(
                    &self.client,
                    layer_weights.attn_wo.clone(),
                    attn_out_handle, // Passed directly from attention output
                    wo_out_handle.clone(),
                    n_embd,
                    q_dim,
                );
            }
        }

        // Final sync: read wo_out
        self.read_handle(&wo_out_handle)
    }

    /// **Plan 409 Phase 3:** Trace version of `dispatch_attention_wo` that
    /// returns BOTH `attn_out` (q_dim) and `wo_out` (n_embd) separately, so
    /// the isolation test can determine which of the two diverges.
    ///
    /// Same algorithm as `dispatch_attention_wo`, but ALWAYS syncs after
    /// attention (never uses the fused path).
    pub(super) fn dispatch_attention_wo_trace(
        &self,
        layer_weights: &CubeCLLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
    ) -> Vec<f32> {
        let n_positions = pos + 1;
        let q_dim = self.config.n_head * self.config.head_dim;

        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, n_positions);

        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        let mut params = self.attn_params;
        params.n_positions = n_positions;
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // ALWAYS read attn_out (trace mode)
        let attn_out = self.read_handle(&attn_out_handle);

        // Re-upload for Wo GEMV
        let attn_rehandle = self.client.create_from_slice(f32::as_bytes(&attn_out));
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wo.clone(),
                attn_rehandle,
                wo_out_handle.clone(),
                n_embd,
                q_dim,
            );
        }

        let wo_out = self.read_handle(&wo_out_handle);

        // Store attn_out in the KV cache's debug slot (repurpose: print magnitude)
        let attn_norm: f32 = attn_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        let wo_norm: f32 = wo_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "    [trace L{layer_idx} P{pos}] attn_out_norm={attn_norm:.4} wo_out_norm={wo_norm:.4}"
        );

        wo_out
    }

    /// **Plan 410 Phase 3A.1:** Dispatch attention + Wo for the active weight format.
    ///
    /// This is the original `forward_layer` dispatch logic factored out so that
    /// both the LoRA-active path (which needs the split variant) and the normal
    /// path can call it without duplicating the format-match logic.
    pub fn dispatch_attention_wo_by_format(
        &self,
        q: &[f32],
        layer_idx: usize,
        pos: usize,
        n: usize,
    ) -> Vec<f32> {
        match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                #[cfg(feature = "q8_kv_cache")]
                if self.kv_cache_q8.is_some() {
                    return self.dispatch_attention_wo_q8kv_f32(
                        &w.layers[layer_idx],
                        q,
                        layer_idx,
                        pos,
                        n,
                    );
                }
                self.dispatch_attention_wo(&w.layers[layer_idx], q, layer_idx, pos, n)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_attention_wo_f16(&w.layers[layer_idx], q, layer_idx, pos, n)
            }
            CubeCLWeightFormat::Q4K(w) => {
                #[cfg(feature = "q8_kv_cache")]
                if self.kv_cache_q8.is_some() {
                    return self.dispatch_attention_wo_q8kv_q4k(
                        &w.layers[layer_idx],
                        q,
                        layer_idx,
                        pos,
                        n,
                    );
                }
                self.dispatch_attention_wo_q4k(&w.layers[layer_idx], q, layer_idx, pos, n)
            }
        }
    }

    /// **Plan 410 Phase 3A.1:** Split attention + Wo dispatch that returns BOTH
    /// `attn_out` (q_dim) and `wo_out` (n_embd), for the O-LoRA insertion point.
    ///
    /// Same as `dispatch_attention_wo_trace` but returns the `attn_out` vector
    /// instead of discarding it. Used when LoRA is active so the O-projection
    /// LoRA delta can be applied with `attn_out` as input. F32 only — LoRA
    /// training uses unquantized weights.
    #[cfg_attr(not(feature = "gemma_lora"), allow(dead_code))]
    pub fn dispatch_attention_wo_split(
        &self,
        layer_weights: &CubeCLLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let n_positions = pos + 1;
        let q_dim = self.config.n_head * self.config.head_dim;

        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, n_positions);

        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        let mut params = self.attn_params;
        params.n_positions = n_positions;
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // Sync attn_out (needed for O-LoRA input)
        let attn_out = self.read_handle(&attn_out_handle);

        // Re-upload for Wo GEMV
        let attn_rehandle = self.client.create_from_slice(f32::as_bytes(&attn_out));
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wo.clone(),
                attn_rehandle,
                wo_out_handle.clone(),
                n_embd,
                q_dim,
            );
        }

        let wo_out = self.read_handle(&wo_out_handle);
        (attn_out, wo_out)
    }

    /// Launch batched gate + up GEMVs (2 launches, 1 sync).
    ///
    /// Both share the same input (hidden state after pre-MLP RMSNorm).
    /// Returns `(gate, up)` vectors, each of length `mlp_hidden`.
    pub fn dispatch_gate_up(
        &self,
        layer_weights: &CubeCLLayerWeights,
        hidden: &[f32],
        mlp: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let gate_handle = self.client.empty(mlp * core::mem::size_of::<f32>());
        let up_handle = self.client.empty(mlp * core::mem::size_of::<f32>());

        // SAFETY: gate/up weights are [mlp, n_embd], hidden is [n_embd]
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.gate_proj.clone(),
                input_handle.clone(),
                gate_handle.clone(),
                mlp,
                n_embd,
            );
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.up_proj.clone(),
                input_handle,
                up_handle.clone(),
                mlp,
                n_embd,
            );
        }

        // Single sync for gate + up
        let gate = self.read_handle(&gate_handle);
        let up = self.read_handle(&up_handle);

        (gate, up)
    }

    // ── Q4_K dispatch helpers ────────────────────────────────────────

    /// Launch CubeCL Q4_K dequant+GEMV: `output[M] = dequant_q4k(weight) @ input[N]`.
    ///
    /// Uses the Q4_K handle's stored dimensions. Input length must equal
    /// `handle.n` and output length will be `handle.m`.
    pub(super) fn dispatch_gemv_q4k(&self, weight: &Q4KHandle, input: &[f32]) -> Vec<f32> {
        let input_handle = self.client.create_from_slice(f32::as_bytes(input));
        let output_handle = self.client.empty(weight.m * core::mem::size_of::<f32>());

        // SAFETY: Q4KHandle has correct m, n and buffer sizes.
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight,
                input_handle,
                output_handle.clone(),
            );
        }

        self.read_handle(&output_handle)
    }

    /// Launch batched Q/K/V Q4_K dequant+GEMVs (3 launches, 1 sync).
    ///
    /// All three GEMVs share the same input (hidden state).
    /// Returns `(q, k, v)` vectors.
    pub(super) fn dispatch_qkv_q4k(
        &self,
        layer_weights: &CubeCLQ4KLayerWeights,
        hidden: &[f32],
        _q_dim: usize,
        _kv_dim: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Note: Q4_K handles store their own dimensions, but we verify
        // the hidden input matches the expected n_embd.
        debug_assert_eq!(hidden.len(), n_embd);

        // dispatch_gemv_q4k allocates output handles internally.
        // Q4KHandle stores m/n dimensions — no external size params needed.
        let q = self.dispatch_gemv_q4k(&layer_weights.attn_wq, hidden);
        let k = self.dispatch_gemv_q4k(&layer_weights.attn_wk, hidden);
        let v = self.dispatch_gemv_q4k(&layer_weights.attn_wv, hidden);

        (q, k, v)
    }

    /// Launch attention + Wo Q4_K GEMV (batched, attn_out → Wo, 1 sync).
    ///
    /// Mirrors the F32 `dispatch_attention_wo` — passes `attn_out_handle` directly
    /// to the Wo Q4_K GEMV without reading to CPU. This avoids the stale-data bug
    /// that occurs when reading an `empty`-allocated handle after kernel write.
    pub(super) fn dispatch_attention_wo_q4k(
        &self,
        layer_weights: &CubeCLQ4KLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
    ) -> Vec<f32> {
        let n_positions = pos + 1;
        let q_dim = self.config.n_head * self.config.head_dim;

        // Build combined KV buffer from CPU cache
        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, n_positions);

        // Create handles
        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        // Launch attention (same for both F32 and Q4_K — attention doesn't use weight GEMV)
        let mut params = self.attn_params;
        params.n_positions = n_positions;
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // Launch Wo Q4_K GEMV (uses attn_out directly, no intermediate sync!)
        // SAFETY: attn_wo Q4KHandle has m=n_embd, n=q_dim, attn_out is [q_dim]
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.attn_wo,
                attn_out_handle,
                wo_out_handle.clone(),
            );
        }

        // Single sync for attention + Wo
        self.read_handle(&wo_out_handle)
    }

    /// Launch batched gate + up Q4_K dequant+GEMVs.
    ///
    /// Both share the same input (hidden state after pre-MLP RMSNorm).
    /// Returns `(gate, up)` vectors, each of length `mlp_hidden`.
    pub(super) fn dispatch_gate_up_q4k(
        &self,
        layer_weights: &CubeCLQ4KLayerWeights,
        hidden: &[f32],
        _mlp: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(hidden.len(), n_embd);

        // dispatch_gemv_q4k allocates output handles internally.
        // Q4KHandle stores m/n dimensions — gate/up output is mlp-sized.
        let gate = self.dispatch_gemv_q4k(&layer_weights.gate_proj, hidden);
        let up = self.dispatch_gemv_q4k(&layer_weights.up_proj, hidden);

        (gate, up)
    }

    // ── F16 weight dispatch helpers (Plan 106 T2.10) ────────────────

    /// Launch batched Q/K/V f16 weight GEMVs (3 launches, 1 sync).
    ///
    /// All three GEMVs share the same input (hidden state).
    /// Returns `(q, k, v)` vectors.
    pub(super) fn dispatch_qkv_f16(
        &self,
        layer_weights: &CubeCLF16LayerWeights,
        hidden: &[f32],
        _q_dim: usize,
        _kv_dim: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        debug_assert_eq!(hidden.len(), n_embd);

        let q = self.dispatch_gemv_f16(&layer_weights.attn_wq, hidden);
        let k = self.dispatch_gemv_f16(&layer_weights.attn_wk, hidden);
        let v = self.dispatch_gemv_f16(&layer_weights.attn_wv, hidden);

        (q, k, v)
    }

    /// Launch attention + Wo f16 weight GEMV (batched, attn_out → Wo, 1 sync).
    ///
    /// Mirrors the F32 `dispatch_attention_wo` — passes `attn_out_handle` directly
    /// to the Wo f16 GEMV without reading to CPU.
    pub(super) fn dispatch_attention_wo_f16(
        &self,
        layer_weights: &CubeCLF16LayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
    ) -> Vec<f32> {
        let n_positions = pos + 1;
        let q_dim = self.config.n_head * self.config.head_dim;

        // Build combined KV buffer from CPU cache
        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, n_positions);

        // Create handles
        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        // Launch attention (same for all weight formats — attention doesn't use weight GEMV)
        let mut params = self.attn_params;
        params.n_positions = n_positions;
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // Launch Wo f16 GEMV (uses attn_out directly, no intermediate sync!)
        // SAFETY: attn_wo F16Handle has m=n_embd, n=q_dim, attn_out is [q_dim]
        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wo.weight.clone(),
                attn_out_handle,
                wo_out_handle.clone(),
                layer_weights.attn_wo.m,
                layer_weights.attn_wo.n,
            );
        }

        // Single sync for attention + Wo
        self.read_handle(&wo_out_handle)
    }

    /// Launch batched gate + up f16 weight GEMVs.
    ///
    /// Both share the same input (hidden state after pre-MLP RMSNorm).
    /// Returns `(gate, up)` vectors, each of length `mlp_hidden`.
    pub(super) fn dispatch_gate_up_f16(
        &self,
        layer_weights: &CubeCLF16LayerWeights,
        hidden: &[f32],
        _mlp: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(hidden.len(), n_embd);

        // dispatch_gemv_f16 allocates output handles internally.
        // F16Handle stores m/n dimensions — gate/up output is mlp-sized.
        let gate = self.dispatch_gemv_f16(&layer_weights.gate_proj, hidden);
        let up = self.dispatch_gemv_f16(&layer_weights.up_proj, hidden);

        (gate, up)
    }

    // ── Q8_0 KV attention dispatch (Plan 106 T2.9) ────────────────

    /// Launch attention + Wo with Q8_0 inline KV dequant + f32 Wo GEMV.
    ///
    /// Same algorithm as `dispatch_attention_wo` but reads Q8_0 packed KV
    /// from `CpuKVCacheQ8` and uses `AttentionQ8KVCubeCL` for inline dequant.
    #[cfg(feature = "q8_kv_cache")]
    pub(super) fn dispatch_attention_wo_q8kv_f32(
        &self,
        layer_weights: &CubeCLLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
    ) -> Vec<f32> {
        let q_dim = self.config.n_head * self.config.head_dim;

        // Build combined Q8_0 KV buffers from quantized cache
        let q8kv = self
            .kv_cache_q8
            .as_ref()
            .expect("Q8 KV cache required")
            .get_combined_q8kv(layer_idx);

        // Create handles
        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_qs_handle = self.client.create_from_slice(u32::as_bytes(&q8kv.kv_qs));
        let kv_scales_handle = self
            .client
            .create_from_slice(f32::as_bytes(&q8kv.kv_scales));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        // Launch Q8_0 attention
        let mut params = self.attn_params;
        params.n_positions = pos + 1;
        AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            attn_out_handle.clone(),
            &params,
        );

        // Launch f32 Wo GEMV (uses attn_out directly, no intermediate sync)
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wo.clone(),
                attn_out_handle,
                wo_out_handle.clone(),
                n_embd,
                q_dim,
            );
        }

        self.read_handle(&wo_out_handle)
    }

    /// Launch attention + Wo with Q8_0 inline KV dequant + Q4_K Wo GEMV.
    ///
    /// Combines Q8_0 KV attention with Q4_K dequant+GEMV for Wo projection.
    #[cfg(feature = "q8_kv_cache")]
    pub(super) fn dispatch_attention_wo_q8kv_q4k(
        &self,
        layer_weights: &CubeCLQ4KLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
    ) -> Vec<f32> {
        let q_dim = self.config.n_head * self.config.head_dim;

        // Build combined Q8_0 KV buffers from quantized cache
        let q8kv = self
            .kv_cache_q8
            .as_ref()
            .expect("Q8 KV cache required")
            .get_combined_q8kv(layer_idx);

        // Create handles
        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_qs_handle = self.client.create_from_slice(u32::as_bytes(&q8kv.kv_qs));
        let kv_scales_handle = self
            .client
            .create_from_slice(f32::as_bytes(&q8kv.kv_scales));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        // Launch Q8_0 attention
        let mut params = self.attn_params;
        params.n_positions = pos + 1;
        AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            attn_out_handle.clone(),
            &params,
        );

        // Launch Q4_K Wo GEMV (uses attn_out directly)
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.attn_wo,
                attn_out_handle,
                wo_out_handle.clone(),
            );
        }

        self.read_handle(&wo_out_handle)
    }

    // ── GPU-resident dispatch helpers (T2.14) ──────────────────────
    // These methods work with GPU handles only — no CPU sync.
    // Used by forward_gpu / forward_layer_gpu.

    /// Launch CubeCL GEMV entirely on GPU (handle-to-handle, no CPU sync).
    ///
    /// Returns output handle for chaining into subsequent GPU kernels.
    pub fn dispatch_gemv_gpu(
        &self,
        weight: &Handle,
        input_handle: Handle,
        m: usize,
        n: usize,
    ) -> Handle {
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());
        // SAFETY: weight has M×N elements, input has N elements.
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
        output_handle
    }

    /// Launch CubeCL GEMV using the **tiled** variant only (handle-to-handle, no CPU sync).
    ///
    /// Used by the GPU-resident training forward. The tiled variant is now correct
    /// for all dimensions after the Plan 481 fix (early `terminate!()` was breaking
    /// cooperative shared-memory loading for small `m`).
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_gemv_tiled_gpu(
        &self,
        weight: &Handle,
        input_handle: Handle,
        m: usize,
        n: usize,
    ) -> Handle {
        use crate::gemv_cubecl::GemvCubeCL;
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());
        // SAFETY: weight has M×N elements, input has N elements.
        unsafe {
            GemvCubeCL::launch_tiled::<ActiveRuntime>(
                &self.client,
                weight.clone(),
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }
        output_handle
    }

    /// Launch CubeCL GEMM (matrix-matrix multiply) entirely on GPU (handle-to-handle, no CPU sync).
    ///
    /// Computes `C[M,P] = A[M,N] × B^T[P,N]` where B is stored `[P,N]` row-major.
    ///
    /// For transformer training, this maps to `Y = X @ W^T` where:
    /// - A = X (input batch), `[seq_len, in_dim]` → M=seq_len, N=in_dim
    /// - B = W (weight), `[out_dim, in_dim]` → P=out_dim
    /// - C = Y (output batch), `[seq_len, out_dim]`
    ///
    /// This is the batched equivalent of `dispatch_gemv_tiled_gpu` — instead of
    /// processing one position at a time (M=1 GEMV), it processes the full
    /// sequence as a single GEMM dispatch. Reduces ~55K per-position GEMV
    /// dispatches to ~182 batched GEMM dispatches (7 weights × 26 layers).
    ///
    /// Plan 482 T2.
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_gemm_gpu(
        &self,
        weight_handle: &Handle,
        input_handle: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) -> Handle {
        use crate::matmul_cubecl::MatmulCubeCL;
        let output_handle = self.client.empty(m * p * core::mem::size_of::<f32>());
        // SAFETY: weight_handle has P×N elements, input_handle has M×N elements,
        // output_handle has M×P elements.
        unsafe {
            MatmulCubeCL::launch_tiled::<ActiveRuntime>(
                &self.client,
                input_handle,
                weight_handle.clone(),
                output_handle.clone(),
                m,
                n,
                p,
            );
        }
        output_handle
    }

    /// Transpose an `[rows x cols]` f32 handle into a `[cols x rows]` one,
    /// entirely on the device (riir-train Issue 572).
    ///
    /// `dst` must already be allocated with at least `rows * cols` f32
    /// elements; it is fully overwritten, so a reused scratch handle needs no
    /// clearing. Nothing is read back and no CPU sync is forced — the write is
    /// ordered before any later dispatch that reads `dst` on the same client.
    ///
    /// This exists so the training lane can hold ONE transposed scratch buffer
    /// per projection role instead of a transposed copy of every weight: the
    /// pre-transposed handles cost a second 9.74 GB of device memory at
    /// Gemma-2-2B f32, which does not fit beside the weights on a 24 GB card.
    ///
    /// Deliberately NOT gated on `gpu_training_resident` (unlike its GEMV
    /// neighbour): both Gemma-2 backward routes — the resident batched one and
    /// the hybrid per-position one — build their transposed handles through the
    /// same constructor, and a feature-gated dispatch would fork that
    /// constructor into two code paths for no behavioural reason.
    pub fn dispatch_transpose_gpu(
        &self,
        src_handle: &Handle,
        dst_handle: &Handle,
        rows: usize,
        cols: usize,
    ) {
        use crate::transpose_cubecl::TransposeCubeCL;
        // SAFETY: both handles are asserted by the caller to hold at least
        // rows x cols f32 elements; the kernel takes its shape from params and
        // derives nothing from either buffer's declared length.
        unsafe {
            TransposeCubeCL::launch::<ActiveRuntime>(
                &self.client,
                src_handle.clone(),
                dst_handle.clone(),
                rows,
                cols,
            );
        }
    }

    /// Launch batched plane GEMV for backward weight-transposed multiply.
    ///
    /// Issue 424 fix: replaces `dispatch_gemm_gpu` in the batched backward.
    /// The GEMM tiled kernel accumulates sequentially (depth n=2304), while
    /// the Plane kernel tree-reduces across 32 lanes (depth n/32 + log2(32)
    /// ≈ 77). The 30× better numerical accuracy prevents gradient divergence
    /// at 26-layer scale.
    ///
    /// Computes `output[batch, out_dim] = input[batch, in_dim] @ weight[out_dim, in_dim]^T`
    /// — same math as `dispatch_gemm_gpu(weight, input, batch, in_dim, out_dim)`.
    ///
    /// Requires plane (subgroup) support. On Metal this is always available
    /// (subgroup size 32).
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_gemv_batched_backward_gpu(
        &self,
        weight_handle: &Handle,
        input_handle: Handle,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Handle {
        use crate::gemv_cubecl::GemvBatchedCubeCL;
        let output_handle = self
            .client
            .empty(batch * out_dim * core::mem::size_of::<f32>());
        // SAFETY: weight_handle has out_dim × in_dim elements,
        // input_handle has batch × in_dim elements,
        // output_handle has batch × out_dim elements.
        unsafe {
            GemvBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight_handle.clone(),
                input_handle,
                output_handle.clone(),
                batch,
                in_dim,
                out_dim,
            );
        }
        output_handle
    }

    /// Launch CubeCL element-wise add entirely on GPU (handle-to-handle, no CPU sync).
    ///
    /// Computes `output[n] = a[n] + b[n]`. Used by the all-GPU LoRA path to
    /// combine the base GEMV output with the LoRA delta.
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_add_gpu(&self, a_handle: &Handle, b_handle: Handle, n: usize) -> Handle {
        use crate::gemv_cubecl::AddCubeCL;
        let output_handle = self.client.empty(n * core::mem::size_of::<f32>());
        // SAFETY: a and b have `n` elements each.
        unsafe {
            AddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                a_handle.clone(),
                b_handle,
                output_handle.clone(),
                n,
            );
        }
        output_handle
    }

    /// Launch CubeCL Q4_K dequant+GEMV entirely on GPU (handle-to-handle).
    pub(super) fn dispatch_gemv_q4k_gpu(&self, weight: &Q4KHandle, input_handle: Handle) -> Handle {
        let output_handle = self.client.empty(weight.m * core::mem::size_of::<f32>());
        // SAFETY: Q4KHandle has correct m, n and buffer sizes.
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight,
                input_handle,
                output_handle.clone(),
            );
        }
        output_handle
    }

    /// Launch fused triple QKV GEMV on GPU (Q4_K weights, handle-to-handle).
    ///
    /// Computes all Q/K/V projections in a single GPU dispatch, replacing
    /// 3 separate Q4_K GEMV dispatches (Plan 171 T28).
    ///
    /// Returns a combined handle with `q_dim + 2 * kv_dim` f32 elements:
    /// `[Q; q_dim | K; kv_dim | V; kv_dim]`.
    pub(super) fn dispatch_gemv_qkv_q4k_gpu(
        &self,
        handle: &crate::gemv_qkv_q4k_cubecl::Q4KQKVHandle,
        input_handle: Handle,
    ) -> Handle {
        let total_rows = handle.total_rows();
        let output_handle = self.client.empty(total_rows * core::mem::size_of::<f32>());
        // SAFETY: Q4KQKVHandle has correct buffer sizes for fused QKV.
        unsafe {
            GemvQkvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                handle,
                input_handle,
                output_handle.clone(),
            );
        }
        output_handle
    }

    /// Launch fused dual GEMV + GeGLU on GPU (Q4_K weights, handle-to-handle).
    ///
    /// Computes `output = GELU(gate_GEMV) * up_GEMV` in a single GPU dispatch,
    /// replacing 3 separate dispatches (gate GEMV + up GEMV + GeGLU).
    /// Plan 171 T29.
    pub(super) fn dispatch_gemv_geglu_q4k_gpu(
        &self,
        handle: &crate::gemv_geglu_q4k_cubecl::Q4KGegluHandle,
        input_handle: Handle,
        m: usize,
    ) -> Handle {
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());
        // SAFETY: Q4KGegluHandle has correct buffer sizes for fused GeGLU.
        unsafe {
            GemvGegluQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                handle,
                input_handle,
                output_handle.clone(),
            );
        }
        output_handle
    }

    /// Launch CubeCL f16 weight GEMV entirely on GPU (handle-to-handle, no CPU sync).
    ///
    /// f16 weights are cast to f32 on-the-fly during the dot product.
    /// Returns output handle for chaining into subsequent GPU kernels.
    pub(super) fn dispatch_gemv_f16_gpu(
        &self,
        weight: &F16Handle,
        input_handle: Handle,
        m: usize,
        n: usize,
    ) -> Handle {
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());
        // SAFETY: F16Handle has correct m, n and weight buffer is m*n f16 elements.
        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight.weight.clone(),
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }
        output_handle
    }

    /// Launch CubeCL RMSNorm entirely on GPU (handle-to-handle).
    ///
    /// Computes `output[i] = input[i] * inv_rms * gamma[i]` on GPU.
    pub fn dispatch_rmsnorm_gpu(
        &self,
        input_handle: Handle,
        gamma: &Handle,
        dim: usize,
        eps: f32,
    ) -> Handle {
        let output_handle = self.client.empty(dim * core::mem::size_of::<f32>());
        // SAFETY: input and gamma have `dim` elements, output pre-allocated.
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                input_handle,
                gamma.clone(),
                output_handle.clone(),
                dim,
                eps,
            );
        }
        output_handle
    }

    /// Launch CubeCL RoPE entirely on GPU (handle-to-handle).
    ///
    /// Uses the cached cos/sin table + GPU handle from [`rope_cos_sin_cache`]:
    /// within a single forward pass `pos` is constant across layers and Q/K,
    /// so the `head_dim / 2` `powf` calls and the GPU buffer upload happen only
    /// once per token instead of once per `(layer, Q/K)` pair.
    pub fn dispatch_rope_gpu(
        &self,
        input_handle: Handle,
        pos: usize,
        n_heads: usize,
    ) -> Handle {
        let head_dim = self.config.head_dim;
        let n = n_heads * head_dim;
        let cos_sin_handle = self.rope_cos_sin_cache.borrow_mut().get_or_compute(
            &self.client,
            pos,
            head_dim,
            self.config.rope_theta,
        );
        let output_handle = self.client.empty(n * core::mem::size_of::<f32>());
        // SAFETY: input has n elements, cos_sin has head_dim elements.
        unsafe {
            RopeCubeCL::launch::<ActiveRuntime>(
                &self.client,
                input_handle,
                cos_sin_handle,
                output_handle.clone(),
                n,
                head_dim,
                crate::rope_geglu_cubecl::RopePairing::RotateHalf,
            );
        }
        output_handle
    }

    /// Launch CubeCL GeGLU entirely on GPU (handle-to-handle).
    ///
    /// Computes `output[i] = gate[i] * GELU(gate[i]) * up[i]` on GPU.
    pub fn dispatch_geglu_gpu(
        &self,
        gate_handle: Handle,
        up_handle: Handle,
        n: usize,
    ) -> Handle {
        let output_handle = self.client.empty(n * core::mem::size_of::<f32>());
        // SAFETY: gate and up have `n` elements each.
        unsafe {
            GegluCubeCL::launch::<ActiveRuntime>(
                &self.client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                n,
            );
        }
        output_handle
    }

    /// Launch fused dual GEMV + GeGLU entirely on GPU (F32 weights, handle-to-handle).
    ///
    /// Computes `output[row] = GELU(dot(weight_gate_row, input)) * dot(weight_up_row, input)`
    /// in a single GPU dispatch, replacing 3 separate dispatches (gate GEMV + up GEMV + GeGLU).
    ///
    /// Saves 2 dispatches per layer × 26 layers = 52 dispatches per decode token.
    pub(super) fn dispatch_gemv_geglu_gpu(
        &self,
        weight_gate: &Handle,
        weight_up: &Handle,
        input_handle: Handle,
        m: usize,
        n: usize,
    ) -> Handle {
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());
        // SAFETY: weight_gate and weight_up have m*n f32 elements, input has n f32 elements.
        unsafe {
            GemvGegluCubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight_gate.clone(),
                weight_up.clone(),
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }
        output_handle
    }

    /// Launch fused dual GEMV + GeGLU entirely on GPU (F16 weights, handle-to-handle).
    ///
    /// Computes `output[row] = GELU(dot(weight_gate_row, input)) * dot(weight_up_row, input)`
    /// in a single GPU dispatch, replacing 3 separate dispatches (gate GEMV + up GEMV + GeGLU).
    ///
    /// Saves 2 dispatches per layer × 26 layers = 52 dispatches per decode token.
    pub(super) fn dispatch_gemv_geglu_f16_gpu(
        &self,
        weight_gate: &F16Handle,
        weight_up: &F16Handle,
        input_handle: Handle,
        m: usize,
        n: usize,
    ) -> Handle {
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());
        // SAFETY: weight_gate and weight_up have m*n f16 elements, input has n f32 elements.
        unsafe {
            GemvGegluF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight_gate.weight.clone(),
                weight_up.weight.clone(),
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }
        output_handle
    }

    /// Launch fused triple QKV GEMV on GPU (F16 weights, handle-to-handle).
    ///
    /// Computes `output[row] = dot(weight_qkv_row, input)` for all Q/K/V rows
    /// in a single GPU dispatch. The output is a combined `[Q | K | V]` buffer.
    ///
    /// Saves 2 dispatches per layer × 26 layers = 52 dispatches per decode token.
    ///
    /// Returns a combined handle with `q_dim + 2 * kv_dim` f32 elements.
    pub(super) fn dispatch_gemv_qkv_f16_gpu(
        &self,
        weight_qkv: &F16Handle,
        input_handle: Handle,
        q_dim: usize,
        kv_dim: usize,
        n: usize,
    ) -> Handle {
        let total_rows = q_dim + 2 * kv_dim;
        let output_handle = self.client.empty(total_rows * core::mem::size_of::<f32>());
        let params: [f32; 3] = [q_dim as f32, kv_dim as f32, n as f32];
        let params_handle = self.client.create_from_slice(f32::as_bytes(&params));

        // SAFETY: weight_qkv has (q_dim + 2*kv_dim) * n f16 elements,
        // input has n f32 elements, output has total_rows f32 elements.
        unsafe {
            GemvQkvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight_qkv.weight.clone(),
                input_handle,
                output_handle.clone(),
                params_handle,
                q_dim,
                kv_dim,
                n,
            );
        }
        output_handle
    }

    /// Launch RoPE on a sub-section of a combined QKV buffer (handle-to-handle).
    ///
    /// Reads from `combined[offset..offset+section_len]`, applies RoPE, writes
    /// to a new output handle of size `section_len`.
    pub(super) fn dispatch_rope_from_combined_gpu(
        &self,
        combined_handle: Handle,
        total_qkv: usize,
        section_offset: usize,
        section_len: usize,
        pos: usize,
        n_heads: usize,
    ) -> Handle {
        let _ = n_heads; // Consistency with dispatch_rope_gpu; kernel computes heads internally
        let head_dim = self.config.head_dim;
        let cos_sin_handle = self.rope_cos_sin_cache.borrow_mut().get_or_compute(
            &self.client,
            pos,
            head_dim,
            self.config.rope_theta,
        );
        let output_handle = self.client.empty(section_len * core::mem::size_of::<f32>());
        // SAFETY: combined has total_qkv elements, cos_sin has head_dim elements,
        // output has section_len elements, section_offset is head_dim-aligned.
        unsafe {
            RopeFromCombinedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                combined_handle,
                total_qkv,
                cos_sin_handle,
                output_handle.clone(),
                section_offset,
                section_len,
                head_dim,
            );
        }
        output_handle
    }

    // ── Wall Attention dispatch methods (Plan 193 T3) ────────────────────
    //
    // Wall rescaling replaces RoPE rotation with elementwise Q/K modulation.
    // These methods are behind #[cfg(feature = "wall_attention")] — zero cost
    // when the feature is disabled.

    /// Launch CubeCL Wall Q/K rescaling on GPU (handle-to-handle).
    ///
    /// Applies `q[d] *= exp(prefix_q[d])` and `k[d] *= exp(-prefix_k[d])`
    /// for each head dimension. Replaces `dispatch_rope_gpu` in Wall mode.
    ///
    /// The prefix buffer contains concatenated `[prefix_q(head_dim) | prefix_k(head_dim)]`.
    #[cfg(feature = "wall_attention")]
    pub(super) fn dispatch_wall_rescale_gpu(
        &self,
        q_handle: Handle,
        k_handle: Handle,
        prefix_qk: &[f32], // length 2 * head_dim
        q_dim: usize,
        kv_dim: usize,
    ) -> (Handle, Handle) {
        let head_dim = self.config.head_dim;
        let prefix_qk_handle = self.client.create_from_slice(f32::as_bytes(prefix_qk));

        // SAFETY: q has q_dim elements, k has kv_dim elements, prefix_qk has 2*head_dim elements.
        // Kernel modifies q and k in-place.
        unsafe {
            WallRescaleCubeCL::launch::<ActiveRuntime>(
                &self.client,
                q_handle.clone(),
                q_dim,
                k_handle.clone(),
                kv_dim,
                prefix_qk_handle,
                prefix_qk.len(),
                q_dim,
                kv_dim,
                head_dim,
            );
        }
        // Kernel operates in-place — return the same handles (CubeCL copy-on-write safe).
        (q_handle, k_handle)
    }

    /// Launch CubeCL Wall rescaling on a sub-section of combined QKV buffer.
    ///
    /// Replaces `dispatch_rope_from_combined_gpu` in Wall mode.
    /// Reads from `combined[offset..offset+section_len]`, applies rescaling,
    /// writes to new output handle.
    #[cfg(feature = "wall_attention")]
    pub(super) fn dispatch_wall_from_combined_gpu(
        &self,
        combined_handle: Handle,
        total_qkv: usize,
        prefix_qk: &[f32], // length 2 * head_dim
        section_offset: usize,
        section_len: usize,
        is_k: bool,
    ) -> Handle {
        let head_dim = self.config.head_dim;
        let prefix_qk_handle = self.client.create_from_slice(f32::as_bytes(prefix_qk));
        let output_handle = self.client.empty(section_len * core::mem::size_of::<f32>());

        // SAFETY: combined has total_qkv elements, prefix_qk has 2*head_dim elements,
        // output has section_len elements.
        unsafe {
            WallRescaleFromCombinedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                combined_handle,
                total_qkv,
                prefix_qk_handle,
                prefix_qk.len(),
                output_handle.clone(),
                section_offset,
                section_len,
                head_dim,
                is_k,
            );
        }
        output_handle
    }

    /// Whether Wall attention is active (has weights loaded).
    #[cfg(feature = "wall_attention")]
    #[allow(dead_code)] // WIP: Wall gate query; step_cpu path is the active consumer.
    fn wall_active(&self) -> bool {
        self.wall_state.as_ref().is_some_and(|s| s.is_active())
    }

    /// Launch CubeCL residual add entirely on GPU (handle-to-handle).
    ///
    /// Computes `output[i] = a[i] + b[i]` on GPU.
    ///
    /// Kept as fallback — `forward_layer_gpu` now uses `dispatch_norm_residual_gpu` instead.
    #[allow(dead_code)]
    pub fn dispatch_residual_add_gpu(
        &self,
        a_handle: Handle,
        b_handle: Handle,
        n: usize,
    ) -> Handle {
        let output_handle = self.client.empty(n * core::mem::size_of::<f32>());
        // SAFETY: a and b have `n` elements each.
        unsafe {
            ResidualAddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                a_handle,
                b_handle,
                output_handle.clone(),
                n,
            );
        }
        output_handle
    }

    /// Launch fused CubeCL RMSNorm + ResidualAdd on GPU (handle-to-handle).
    ///
    /// Computes `output[i] = rmsnorm(input)[i] + residual[i]` in a single dispatch.
    /// Replaces two separate dispatches (rmsnorm + residual_add) with one fused kernel.
    /// Saves 1 GPU dispatch per call (2 per layer = 52 total for 26 layers).
    pub fn dispatch_norm_residual_gpu(
        &self,
        input_handle: Handle,
        gamma: &Handle,
        residual_handle: Handle,
        dim: usize,
        eps: f32,
    ) -> Handle {
        use crate::epilogue::NormResidualCubeCL;

        let output_handle = self.client.empty(dim * core::mem::size_of::<f32>());
        // SAFETY: input, gamma, residual have `dim` elements, output pre-allocated.
        unsafe {
            NormResidualCubeCL::launch::<ActiveRuntime>(
                &self.client,
                input_handle,
                gamma.clone(),
                residual_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }
        output_handle
    }

    /// Launch flash attention on GPU (handle-to-handle, no CPU sync).
    ///
    /// Handles Q8_0 KV cache dispatch when available, falls back to f32 KV.
    /// Returns attn_out handle (NOT Wo output — caller chains Wo separately).
    pub fn dispatch_attention_gpu(
        &self,
        query_handle: Handle,
        layer_idx: usize,
        pos: usize,
    ) -> Handle {
        let q_dim = self.config.n_head * self.config.head_dim;
        let n_positions = pos + 1;
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());

        let mut params = self.attn_params;
        params.n_positions = n_positions;

        // Use GPU-resident KV cache if available (zero-copy, no re-upload)
        if let Some(ref gpu_cache) = self.gpu_kv_cache {
            // SAFETY: pos < block_size guaranteed by caller.
            let (compact_handle, n_pos) =
                unsafe { gpu_cache.compact_for_attention(&self.client, layer_idx, pos) };
            params.n_positions = n_pos;
            AttentionCubeCL::launch::<ActiveRuntime>(
                &self.client,
                query_handle,
                compact_handle,
                attn_out_handle.clone(),
                &params,
            );
            return attn_out_handle;
        }

        // Use Q8 KV cache if available (CPU fallback)
        #[cfg(feature = "q8_kv_cache")]
        if let Some(ref q8_cache) = self.kv_cache_q8 {
            let q8kv = q8_cache.get_combined_q8kv(layer_idx);
            let kv_qs_handle = self.client.create_from_slice(u32::as_bytes(&q8kv.kv_qs));
            let kv_scales_handle = self
                .client
                .create_from_slice(f32::as_bytes(&q8kv.kv_scales));
            AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
                &self.client,
                query_handle,
                kv_qs_handle,
                kv_scales_handle,
                attn_out_handle.clone(),
                &params,
            );
            return attn_out_handle;
        }

        // f32 KV path (CPU fallback — re-uploads combined KV each call)
        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, n_positions);
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );
        attn_out_handle
    }

    // ── Plan 482: Batched dispatch methods (handle-to-handle, no CPU sync) ──

    /// Launch batched CubeCL RMSNorm for `[seq_len × dim]` (Plan 482 T4).
    ///
    /// Normalizes each row independently. One workgroup per row.
    /// Replaces `seq_len` separate `dispatch_rmsnorm_gpu` calls.
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_rmsnorm_batched_gpu(
        &self,
        input_handle: Handle,
        gamma: &Handle,
        seq_len: usize,
        dim: usize,
        eps: f32,
    ) -> Handle {
        use crate::norms_cubecl::RmsNormBatchedCubeCL;
        let output_handle = self
            .client
            .empty(seq_len * dim * core::mem::size_of::<f32>());
        // SAFETY: input has seq_len*dim elements, gamma has dim elements.
        unsafe {
            RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                input_handle,
                gamma.clone(),
                output_handle.clone(),
                seq_len,
                dim,
                eps,
            );
        }
        output_handle
    }

    /// Launch batched fused RMSNorm + ResidualAdd for `[seq_len × dim]` (Plan 482 T4).
    ///
    /// For each row: `output[r, j] = rmsnorm(input)[r, j] + residual[r, j]`.
    /// One workgroup per row. Replaces `seq_len` separate `dispatch_norm_residual_gpu` calls.
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_norm_residual_batched_gpu(
        &self,
        input_handle: Handle,
        gamma: &Handle,
        residual_handle: Handle,
        seq_len: usize,
        dim: usize,
        eps: f32,
    ) -> Handle {
        use crate::epilogue::NormResidualBatchedCubeCL;
        let output_handle = self
            .client
            .empty(seq_len * dim * core::mem::size_of::<f32>());
        // SAFETY: input, residual have seq_len*dim elements, gamma has dim elements.
        unsafe {
            NormResidualBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                input_handle,
                gamma.clone(),
                residual_handle,
                output_handle.clone(),
                seq_len,
                dim,
                eps,
            );
        }
        output_handle
    }

    /// Launch batched CubeCL RoPE for `[seq_len × n_heads × head_dim]` (Plan 482 T5).
    ///
    /// Applies position-dependent rotary embedding to all positions in one dispatch.
    /// Uses a pre-computed `[seq_len × head_dim]` cos/sin table.
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_rope_batched_gpu(
        &self,
        input_handle: Handle,
        cos_sin_handle: Handle,
        seq_len: usize,
        n_heads: usize,
    ) -> Handle {
        use crate::rope_geglu_cubecl::{RopeBatchedCubeCL, RopePairing};
        let head_dim = self.config.head_dim;
        let total = seq_len * n_heads * head_dim;
        let output_handle = self.client.empty(total * core::mem::size_of::<f32>());
        // SAFETY: input has total elements, cos_sin has seq_len*head_dim elements.
        unsafe {
            RopeBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                input_handle,
                cos_sin_handle,
                output_handle.clone(),
                total,
                head_dim,
                n_heads,
                RopePairing::RotateHalf,
            );
        }
        output_handle
    }

    /// Launch fused causal attention entirely on GPU (Issue 430 Option B).
    ///
    /// Replaces the CPU attention path — 2 syncs per layer under the current
    /// batched forward (QKV batched readback + attn_out upload; see
    /// `resident.rs` — the "4 sync points" figure was the pre-batching count)
    /// with a single GPU dispatch. Computes QK^T + softcap + softmax +
    /// weighted-sum in one kernel.
    ///
    /// See `attention_causal_fused_cubecl::CausalAttentionFusedCubeCL` for
    /// the kernel contract + constraints (seq_len ≤ 256, power of 2; head_dim ≤ 256).
    ///
    /// # Safety
    ///
    /// - `q_handle`: `seq_len * q_dim` f32 elements (post-RoPE Q)
    /// - `k_handle`: `seq_len * kv_dim` f32 elements (post-RoPE K)
    /// - `v_handle`: `seq_len * kv_dim` f32 elements (pre-RoPE V)
    /// - Returns: `seq_len * q_dim` f32 element handle (attn_out)
    #[cfg(feature = "gpu_training_resident")]
    pub fn dispatch_attention_fused_gpu(
        &self,
        q_handle: Handle,
        k_handle: Handle,
        v_handle: Handle,
        seq_len: usize,
    ) -> Handle {
        use crate::attention_causal_fused_cubecl::CausalAttentionFusedCubeCL;
        let n_head = self.config.n_head;
        let n_kv_head = self.config.n_kv_head;
        let head_dim = self.config.head_dim;
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv_head * head_dim;
        let softcap = self.config.attn_logit_softcapping;
        let output_handle =
            self.client.empty(seq_len * q_dim * core::mem::size_of::<f32>());
        // SAFETY: q/k/v handles have correct sizes (caller contract).
        unsafe {
            CausalAttentionFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                q_handle,
                k_handle,
                v_handle,
                output_handle.clone(),
                n_head,
                n_kv_head,
                head_dim,
                seq_len,
                q_dim,
                kv_dim,
                softcap,
            );
        }
        output_handle
    }
}
