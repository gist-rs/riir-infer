//! CubeCL dispatch helper methods for `GpuGemma4CubeCL`.
//!
//! These are the low-level CubeCL kernel dispatch helpers that launch GEMV and
//! attention kernels on the GPU. Mirrors `gemma2_cubecl::dispatch` but with
//! per-layer-type dimension parameters (Q/K/V sizes vary between Sliding and
//! Full attention layers).

use super::*;
use super::weight_buffers::Gemma4CubeCLLayerWeights;

#[allow(dead_code)] // Forward-path scaffolding — exercised under GPU tests.
impl GpuGemma4CubeCL {
    // ── CubeCL dispatch helpers ─────────────────────────────────────

    /// Launch CubeCL GEMV: `output[M] = weight[M,N] @ input[N]`.
    ///
    /// Creates input handle from CPU data, allocates output handle, launches
    /// kernel via the autotune cache, reads result back to CPU.
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

    /// Launch batched Q/K/V GEMVs (3 launches, 1 sync).
    ///
    /// All three GEMVs share the same input (the RMSNormed hidden state).
    /// CubeCL batches them into a single GPU submission.
    ///
    /// Returns `(q, k, v)` vectors. Dimensions are per-layer-type:
    /// `q_dim` = `n_head * head_dim` (sliding) or `n_head * global_head_dim` (full);
    /// `kv_dim` = `n_kv_head * head_dim` (sliding) or `n_global_kv_head * global_head_dim` (full).
    pub fn dispatch_qkv(
        &self,
        layer_weights: &Gemma4CubeCLLayerWeights,
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

        // Single sync: read all outputs (first read triggers batch execution).
        let q = self.read_handle(&q_handle);
        let k = self.read_handle(&k_handle);
        let v = self.read_handle(&v_handle);

        (q, k, v)
    }

    /// Launch batched gate + up GEMVs (2 launches, 1 sync).
    ///
    /// Returns `(gate, up)` vectors, each of length `mlp_hidden`.
    pub fn dispatch_gate_up(
        &self,
        layer_weights: &Gemma4CubeCLLayerWeights,
        hidden: &[f32],
        mlp_hidden: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let gate_handle = self.client.empty(mlp_hidden * core::mem::size_of::<f32>());
        let up_handle = self.client.empty(mlp_hidden * core::mem::size_of::<f32>());

        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.gate_proj.clone(),
                input_handle.clone(),
                gate_handle.clone(),
                mlp_hidden,
                n_embd,
            );
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.up_proj.clone(),
                input_handle,
                up_handle.clone(),
                mlp_hidden,
                n_embd,
            );
        }

        let gate = self.read_handle(&gate_handle);
        let up = self.read_handle(&up_handle);
        (gate, up)
    }

    /// Launch attention + Wo GEMV for a single layer (2 launches, 1 sync).
    ///
    /// Builds the combined KV window buffer from the CPU cache, uploads Q + KV,
    /// launches the attention kernel (writes `attn_out`), then launches the Wo
    /// GEMV consuming `attn_out` directly (no intermediate CPU sync).
    ///
    /// Per-layer-type parameters:
    /// - `head_dim`: sliding uses `config.head_dim`, full uses `config.global_head_dim`.
    /// - `n_kv_head`: sliding uses `config.n_kv_head`, full uses `config.n_global_kv_head`.
    /// - `t_start`, `n_pos`: the attended window (sliding = `[pos-sw+1, pos]`,
    ///   full = `[0, pos]`).
    ///
    /// Returns Wo output `[n_embd]`.
    ///
    /// # Gemma-4 attention specifics
    ///
    /// - `scale = 1.0` (NO `1/sqrt(head_dim)` pre-scaling, per llama.cpp).
    /// - `softcap = config.attn_logit_softcapping` (50.0 for Gemma-4).
    /// - QK-Norm is applied BEFORE this call (on CPU, after the QKV GEMV).
    pub fn dispatch_attention_wo(
        &self,
        layer_weights: &Gemma4CubeCLLayerWeights,
        query: &[f32],
        layer_idx: usize,
        t_start: usize,
        n_pos: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        n_embd: usize,
        q_dim: usize,
    ) -> Vec<f32> {
        // Build combined KV window from CPU cache.
        let combined_kv = self
            .kv_cache
            .get_combined_kv_window(layer_idx, t_start, n_pos);

        // Create handles.
        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        // Launch attention. Gemma-4 uses scale=1.0 (no 1/sqrt(head_dim)).
        let params = AttentionParams {
            n_head,
            n_kv_head,
            head_dim,
            n_positions: n_pos,
            softcap: self.config.attn_logit_softcapping,
            scale: 1.0, // Gemma-4: no pre-attn scaling (per llama.cpp)
        };
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // Fused path: attention → Wo without intermediate sync.
        // SAFETY: attn_wo is [n_embd, q_dim], attn_out is [q_dim], wo_out is [n_embd].
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

        // Final sync: read wo_out.
        self.read_handle(&wo_out_handle)
    }

    /// Read a CubeCL handle back to CPU as `Vec<f32>`.
    ///
    /// Wrapper around `client.read_one` for the synchronous hybrid forward path.
    /// Each call is one GPU→CPU sync point. Panics if the GPU read fails
    /// (indicates a programming bug or GPU device loss).
    pub fn read_handle(&self, handle: &Handle) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(handle.clone())
            .unwrap_or_else(|e| panic!("CubeCL buffer read failed: {e}"));
        f32::from_bytes(&bytes).to_vec()
    }
}
