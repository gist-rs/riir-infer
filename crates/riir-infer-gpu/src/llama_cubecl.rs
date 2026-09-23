//! CubeCL integration for LLaMA GPU forward pass (MiniCPM5-1B inference).
//!
//! Hybrid CPU/CubeCL decode forward pass for LLaMA-family inference:
//! - **CubeCL (GPU)**: GEMV (Q/K/V, Wo, gate/up/down, lm_head) + flash attention
//! - **CPU fallback**: RMSNorm, RoPE, SwiGLU, residual add
//!
//! # Architecture Differences from Gemma 2
//!
//! | Aspect | Gemma 2 | LLaMA (MiniCPM5-1B) |
//! |--------|---------|---------------------|
//! | Embedding | sqrt(n_embd) scaling | No scaling |
//! | Norm gamma | +1 offset pre-applied | No offset |
//! | MLP activation | GeGLU (GELU * up) | SwiGLU (SiLU(gate) * up) |
//! | Norm gammas per layer | 4 | 2 (input_norm, post_attn_norm) |
//! | Norm placement | Post-norm | Pre-norm (direct add) |
//! | Logit softcap | Yes | None |
//! | LM head | Tied to wte | Separate weight matrix |
//!
//! # Sync Points (4 per layer)
//!
//! | Sync | GPU Kernels | CPU Ops After Sync |
//! |------|------------|-------------------|
//! | 1 | Q + K + V GEMVs | RoPE(Q), RoPE(K), store K/V in cache |
//! | 2 | Attention + Wo GEMV | Direct add (no post-norm) |
//! | 3 | Gate + Up GEMVs | SwiGLU |
//! | 4 | Down GEMV | Direct add (no post-norm) |

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;

use crate::attention_cubecl::{AttentionCubeCL, AttentionParams};
use riir_infer_core::llama_layer::LlamaTransformerWeights;
use riir_infer_core::types::Config;

// Re-use shared CPU helpers and GPU types from gemma2_cubecl.
#[cfg(feature = "cubecl_runtime")]
use crate::epilogue::{NormResidualCubeCL, SwigluCubeCL};
use crate::gemma2_cubecl::{CpuKVCache, GpuKVCache, apply_rope, rmsnorm_gamma};
#[cfg(feature = "cubecl_runtime")]
use crate::gemma2_cubecl::KvStoreCubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::norms_cubecl::{ResidualAddCubeCL, RmsNormCubeCL};
#[cfg(feature = "cubecl_runtime")]
use crate::rope_geglu_cubecl::RopeCubeCL;

// ── CPU fallback: SwiGLU activation ─────────────────────────────────

/// SiLU (Sigmoid Linear Unit) activation: `x / (1 + exp(-x))`.
#[cfg(feature = "cubecl_runtime")]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU activation: `out[i] = SiLU(gate[i]) * up[i]`.
///
/// LLaMA uses SwiGLU instead of Gemma 2's GeGLU:
/// - SiLU activation (not GELU) on the gate projection
/// - Multiplied element-wise by the up projection
#[cfg(feature = "cubecl_runtime")]
fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = silu(g) * u;
    }
}

// ── Weight Buffers ─────────────────────────────────────────────────

/// Per-layer CubeCL GEMV weight handles for LLaMA inference (f32).
#[cfg(feature = "cubecl_runtime")]
pub(crate) struct LlamaCubeCLLayerWeights {
    pub(crate) attn_wq: Handle, // [q_dim, n_embd] where q_dim = n_head * head_dim
    pub(crate) attn_wk: Handle, // [kv_dim, n_embd]
    pub(crate) attn_wv: Handle, // [kv_dim, n_embd]
    pub(crate) attn_wo: Handle, // [n_embd, q_dim]
    pub(crate) gate_proj: Handle, // [mlp_hidden, n_embd]
    pub(crate) up_proj: Handle, // [mlp_hidden, n_embd]
    pub(crate) down_proj: Handle, // [n_embd, mlp_hidden]
}

/// All CubeCL GEMV weight handles for LLaMA inference.
#[cfg(feature = "cubecl_runtime")]
pub(crate) struct LlamaCubeCLWeightBuffers {
    pub(crate) layers: Vec<LlamaCubeCLLayerWeights>,
    /// Separate lm_head weight (NOT tied to wte like Gemma 2).
    pub(crate) lm_head: Handle, // [vocab_size, n_embd]
}

#[cfg(feature = "cubecl_runtime")]
impl LlamaCubeCLWeightBuffers {
    /// Upload all GEMV weights to CubeCL's buffer pool.
    pub(crate) fn from_weights(
        client: &ComputeClient<ActiveRuntime>,
        weights: &LlamaTransformerWeights,
    ) -> Self {
        let lm_head = client.create_from_slice(f32::as_bytes(&weights.lm_head));

        let layers = weights
            .layers
            .iter()
            .map(|l| LlamaCubeCLLayerWeights {
                attn_wq: client.create_from_slice(f32::as_bytes(&l.attn_wq)),
                attn_wk: client.create_from_slice(f32::as_bytes(&l.attn_wk)),
                attn_wv: client.create_from_slice(f32::as_bytes(&l.attn_wv)),
                attn_wo: client.create_from_slice(f32::as_bytes(&l.attn_wo)),
                gate_proj: client.create_from_slice(f32::as_bytes(&l.gate_proj)),
                up_proj: client.create_from_slice(f32::as_bytes(&l.up_proj)),
                down_proj: client.create_from_slice(f32::as_bytes(&l.down_proj)),
            })
            .collect();

        Self { layers, lm_head }
    }
}

// ── Norm Gammas (CPU-side) ─────────────────────────────────────────

/// Per-layer LLaMA RMSNorm gamma vectors.
///
/// LLaMA has 2 norm gammas per layer (vs Gemma 2's 4):
/// - `input_norm`: applied before attention QKV projection
/// - `post_attn_norm`: applied before MLP gate/up projection (called ffn_norm in LLaMA naming)
///
/// No +1 offset is applied to these gammas (unlike Gemma 2).
#[cfg(feature = "cubecl_runtime")]
pub(crate) struct LlamaLayerNormGammas {
    pub(crate) input_norm: Vec<f32>,     // [n_embd]
    pub(crate) post_attn_norm: Vec<f32>, // [n_embd]
}

/// All LLaMA RMSNorm gamma vectors.
#[cfg(feature = "cubecl_runtime")]
pub(crate) struct LlamaNormGammas {
    pub(crate) layers: Vec<LlamaLayerNormGammas>,
    pub(crate) final_norm: Vec<f32>, // [n_embd]
}

#[cfg(feature = "cubecl_runtime")]
impl LlamaNormGammas {
    /// Extract norm gammas from LLaMA transformer weights.
    ///
    /// LLaMA gamma is used directly without +1 offset (unlike Gemma 2
    /// which pre-applies +1 during weight loading).
    pub(crate) fn from_weights(weights: &LlamaTransformerWeights) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|l| LlamaLayerNormGammas {
                input_norm: l.input_norm.clone(),
                post_attn_norm: l.post_attn_norm.clone(),
            })
            .collect();

        Self {
            layers,
            final_norm: weights.final_norm.clone(),
        }
    }
}

// ── GpuLlamaCubeCL ─────────────────────────────────────────────────

/// CubeCL-accelerated LLaMA forward pass for MiniCPM5-1B inference.
///
/// Hybrid CPU/CubeCL decode path:
/// - GPU: GEMV (Q/K/V, Wo, gate/up/down, lm_head) + flash attention
/// - CPU: RMSNorm, RoPE, SwiGLU, residual add
///
/// # Differences from GpuGemmaCubeCL
///
/// - No embedding scaling (Gemma 2 scales by sqrt(n_embd))
/// - SwiGLU instead of GeGLU (SiLU activation, not GELU)
/// - Separate lm_head (not tied to wte)
/// - 2 norm gammas per layer (not 4)
/// - Pre-norm architecture (direct residual add, no post-norm)
/// - No logit softcapping
///
/// Per-layer GPU RMSNorm gamma handles for LLaMA.
///
/// LLaMA has 2 norms per layer (vs Gemma 2's 4).
#[cfg(feature = "cubecl_runtime")]
pub(crate) struct LlamaGpuLayerNormGammas {
    pub(crate) input_norm: Handle,
    pub(crate) post_attn_norm: Handle,
}

/// All GPU-resident RMSNorm gamma handles for LLaMA inference.
#[cfg(feature = "cubecl_runtime")]
pub(crate) struct LlamaGpuNormGammaHandles {
    pub(crate) layers: Vec<LlamaGpuLayerNormGammas>,
    pub(crate) final_norm: Handle,
}

#[cfg(feature = "cubecl_runtime")]
impl LlamaGpuNormGammaHandles {
    /// Upload all RMSNorm gamma vectors to CubeCL GPU buffers.
    pub(crate) fn upload(
        client: &ComputeClient<ActiveRuntime>,
        weights: &LlamaTransformerWeights,
    ) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|l| LlamaGpuLayerNormGammas {
                input_norm: client.create_from_slice(f32::as_bytes(&l.input_norm)),
                post_attn_norm: client.create_from_slice(f32::as_bytes(&l.post_attn_norm)),
            })
            .collect();

        let final_norm = client.create_from_slice(f32::as_bytes(&weights.final_norm));

        Self { layers, final_norm }
    }
}

pub struct GpuLlamaCubeCL {
    /// CubeCL compute client sharing the same wgpu Device/Queue as GpuContext.
    pub(crate) client: ComputeClient<ActiveRuntime>,
    /// Model configuration.
    pub(crate) config: Config,
    /// CubeCL GEMV weight handles (uploaded once, cloned per launch).
    pub(crate) weights: LlamaCubeCLWeightBuffers,
    /// CPU RMSNorm gamma vectors (used for CPU-side RMSNorm fallback).
    pub(crate) norm_gammas: LlamaNormGammas,
    /// GPU RMSNorm gamma handles (uploaded once, used for GPU-side RMSNorm).
    gpu_norm_gammas: LlamaGpuNormGammaHandles,
    /// CPU KV cache (grows with each position).
    /// Used by CPU-hybrid `forward()` path.
    pub(crate) kv_cache: CpuKVCache,
    /// GPU-resident KV cache — pre-allocated combined `[keys || values]` buffers.
    /// Used by `forward_gpu()` path. Eliminates per-layer K/V sync.
    gpu_kv_cache: Option<GpuKVCache>,
    /// CPU embedding weights (for embedding lookup, no sqrt scaling).
    pub(crate) wte_cpu: Vec<f32>,
    /// Attention parameters.
    attn_params: AttentionParams,
    /// GEMV autotune cache.
    pub(crate) gemv_autotune: crate::gemv_autotune::GemvAutotune,
    /// Cached RoPE cos/sin table + GPU handle, keyed on `pos`.
    ///
    /// Within a single forward pass `pos` is constant across layers and Q/K,
    /// so the table is computed once and reused `n_layer * 2` times instead of
    /// recomputing `head_dim / 2` `powf` calls + a GPU buffer upload per call.
    /// Wrapped in `RefCell` because `dispatch_rope_gpu` takes `&self`.
    #[cfg(feature = "cubecl_runtime")]
    rope_cos_sin_cache: std::cell::RefCell<crate::rope_geglu_cubecl::RopeCosSinCache>,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuLlamaCubeCL {
    /// Initialize CubeCL-accelerated LLaMA forward pass.
    ///
    /// Uploads all GEMV weights to CubeCL's GPU buffer pool and clones
    /// norm gammas to CPU. The client should share the same wgpu Device/Queue
    /// as the GpuContext (via `init_device`).
    pub fn new(
        client: ComputeClient<ActiveRuntime>,
        weights: &LlamaTransformerWeights,
        config: &Config,
    ) -> Self {
        let cubecl_weights = LlamaCubeCLWeightBuffers::from_weights(&client, weights);
        let norm_gammas = LlamaNormGammas::from_weights(weights);
        let gpu_norm_gammas = LlamaGpuNormGammaHandles::upload(&client, weights);
        let kv_stride = config.n_kv_head * config.head_dim;
        let kv_cache = CpuKVCache::new(config.n_layer, kv_stride);
        let gpu_kv_cache = Some(GpuKVCache::new(
            &client,
            config.n_layer,
            kv_stride,
            config.block_size,
        ));

        let attn_params = AttentionParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0, // Set per attention dispatch.
            softcap: 0.0,   // LLaMA has no attention logit softcapping.
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        };

        Self {
            client,
            config: config.clone(),
            weights: cubecl_weights,
            norm_gammas,
            gpu_norm_gammas,
            kv_cache,
            gpu_kv_cache,
            wte_cpu: weights.wte.clone(),
            attn_params,
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
            #[cfg(feature = "cubecl_runtime")]
            rope_cos_sin_cache: std::cell::RefCell::new(
                crate::rope_geglu_cubecl::RopeCosSinCache::new(),
            ),
        }
    }

    /// Run full LLaMA forward pass for a single token at the given position.
    ///
    /// Returns logits vector of length `vocab_size`.
    ///
    /// # Differences from Gemma 2 `forward()`
    ///
    /// - No embedding sqrt scaling
    /// - Separate lm_head GEMV (not tied to wte)
    /// - No logit softcapping
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let vocab = self.config.vocab_size;

        // 1. Embedding lookup on CPU (NO sqrt scaling, unlike Gemma 2)
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i];
        }

        // 2. Process all layers
        for layer_idx in 0..self.config.n_layer {
            hidden = self.forward_layer(hidden, layer_idx, pos);
        }

        // 3. Final RMSNorm
        let eps = self.config.rms_norm_eps as f32;
        rmsnorm_gamma(&mut hidden, &self.norm_gammas.final_norm, n, eps);

        // 4. LM head (separate weight, NOT tied to wte): logits = lm_head @ hidden
        // 5. NO logit softcapping (LLaMA doesn't use softcapping)
        self.dispatch_gemv(&self.weights.lm_head, &hidden, vocab, n)
    }

    /// Single LLaMA transformer layer (hybrid CPU/CubeCL dispatch).
    ///
    /// # Compute Pass Structure (4 sync points)
    ///
    /// ```text
    /// [CPU] Save residual = hidden
    /// [CPU] RMSNorm(hidden, input_norm) — NO +1 offset
    /// [GPU] GEMV Q + K + V → q, k, v           ← Sync 1
    /// [CPU] RoPE(q)
    /// [CPU] RoPE(k)
    /// [CPU] Store k, v in CPU KV cache
    /// [GPU] Attention + Wo → wo_out              ← Sync 2
    /// [CPU] hidden = wo_out + residual           ← NO post-norm, direct add
    /// [CPU] residual2 = hidden
    /// [CPU] RMSNorm(hidden, post_attn_norm) — NO +1 offset
    /// [GPU] GEMV gate + up → gate, up            ← Sync 3
    /// [CPU] SwiGLU(gate, up) → mlp_hidden       ← SiLU, NOT GELU
    /// [GPU] GEMV down → down_out                 ← Sync 4
    /// [CPU] hidden = down_out + residual2        ← NO post-norm, direct add
    /// ```
    fn forward_layer(&mut self, mut hidden: Vec<f32>, layer_idx: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let q_dim = self.config.n_head * self.config.head_dim;
        let kv_dim = self.config.n_kv_head * self.config.head_dim;
        let mlp = self.config.mlp_hidden;
        let eps = self.config.rms_norm_eps as f32;
        let norms = &self.norm_gammas.layers[layer_idx];

        // Save residual for attention output add (pre-norm: direct add, no post-norm)
        let residual = hidden.clone();

        // ── Pass A: QKV + RoPE ─────────────────────────────────────

        // CPU: RMSNorm (input_norm) — gamma has NO +1 offset
        rmsnorm_gamma(&mut hidden, &norms.input_norm, n, eps);

        // GPU: QKV GEMVs (batched, 1 sync for all 3) → Sync 1
        let (mut q, mut k, v) =
            self.dispatch_qkv(&self.weights.layers[layer_idx], &hidden, q_dim, kv_dim, n);

        // CPU: RoPE on both Q and K
        apply_rope(
            &mut q,
            pos,
            self.config.head_dim,
            self.config.n_head,
            self.config.rope_theta,
        );
        apply_rope(
            &mut k,
            pos,
            self.config.head_dim,
            self.config.n_kv_head,
            self.config.rope_theta,
        );

        // CPU: Store K, V in KV cache
        self.kv_cache.store(layer_idx, pos, &k, &v);

        // ── Pass B: Attention + Wo + residual ──────────────────────

        // GPU: Attention + Wo (batched, 1 sync) → Sync 2
        let wo_out =
            self.dispatch_attention_wo(&self.weights.layers[layer_idx], &q, layer_idx, pos, n);

        // CPU: Direct residual add (NO post-norm — LLaMA is pre-norm architecture)
        let mut hidden = wo_out;
        for (h, r) in hidden.iter_mut().zip(residual.iter()) {
            *h += r;
        }

        // Save residual2 for MLP output add
        let residual2 = hidden.clone();

        // ── Pass C: MLP ────────────────────────────────────────────

        // CPU: RMSNorm (post_attn_norm / ffn_norm) — NO +1 offset
        rmsnorm_gamma(&mut hidden, &norms.post_attn_norm, n, eps);

        // GPU: Gate + Up GEMVs (batched, 1 sync) → Sync 3
        let (gate, up) = self.dispatch_gate_up(&self.weights.layers[layer_idx], &hidden, mlp, n);

        // CPU: SwiGLU (SiLU activation, NOT GELU)
        let mut mlp_hidden = vec![0.0f32; mlp];
        swiglu(&gate, &up, &mut mlp_hidden);

        // GPU: Down GEMV (1 sync) → Sync 4
        let down_out = self.dispatch_gemv(
            &self.weights.layers[layer_idx].down_proj,
            &mlp_hidden,
            n,
            mlp,
        );

        // CPU: Direct residual add (NO post-norm)
        let mut hidden = down_out;
        for (h, r) in hidden.iter_mut().zip(residual2.iter()) {
            *h += r;
        }

        hidden
    }

    // ── GEMV dispatch helpers ────────────────────────────────────────

    /// Launch CubeCL f32 GEMV: `output[M] = weight[M,N] @ input[N]`.
    fn dispatch_gemv(&self, weight: &Handle, input: &[f32], m: usize, n: usize) -> Vec<f32> {
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
    /// All three GEMVs share the same input (hidden state).
    /// CubeCL batches them into a single GPU submission.
    ///
    /// Returns `(q, k, v)` vectors.
    fn dispatch_qkv(
        &self,
        layer_weights: &LlamaCubeCLLayerWeights,
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
    ///
    /// Returns Wo output `[n_embd]`.
    fn dispatch_attention_wo(
        &self,
        layer_weights: &LlamaCubeCLLayerWeights,
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

        // Launch attention (LLaMA has no softcapping)
        let mut params = self.attn_params;
        params.n_positions = n_positions;
        AttentionCubeCL::launch::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

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

        // Final sync: read wo_out
        self.read_handle(&wo_out_handle)
    }

    /// Launch batched gate + up GEMVs (2 launches, 1 sync).
    ///
    /// Both share the same input (hidden state after post_attn_norm RMSNorm).
    ///
    /// Returns `(gate, up)` vectors, each of length `mlp_hidden`.
    fn dispatch_gate_up(
        &self,
        layer_weights: &LlamaCubeCLLayerWeights,
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

    /// Read a CubeCL Handle back to CPU as Vec<f32>.
    fn read_handle(&self, handle: &Handle) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(handle.clone())
            .unwrap_or_else(|e| panic!("CubeCL buffer read failed: {e}"));
        f32::from_bytes(&bytes).to_vec()
    }

    // ── GPU-resident dispatch helpers ─────────────────────────────────
    // These methods work with GPU handles only — no CPU sync.
    // Used by forward_gpu / forward_layer_gpu.

    /// Launch CubeCL GEMV entirely on GPU (handle-to-handle, no CPU sync).
    fn dispatch_gemv_gpu(
        &self,
        weight: &Handle,
        input_handle: Handle,
        m: usize,
        n: usize,
    ) -> Handle {
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
        output_handle
    }

    /// Launch CubeCL RMSNorm entirely on GPU (handle-to-handle).
    fn dispatch_rmsnorm_gpu(
        &self,
        input_handle: Handle,
        gamma: &Handle,
        dim: usize,
        eps: f32,
    ) -> Handle {
        let output_handle = self.client.empty(dim * core::mem::size_of::<f32>());
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
    ///
    /// Pairing (Issue 435): [`RopePairing::RotateHalf`], matching both this
    /// struct's own hybrid `forward()` — which rotates on CPU via the shared
    /// `apply_rope` — and the CPU reference `riir_infer_core::transformer::
    /// forward_llama`, which uses `apply_rope_with_freq`. Measured against that
    /// reference on MiniCPM5-1B by `test_llama_cubecl_vs_cpu`, not assumed.
    fn dispatch_rope_gpu(&self, input_handle: Handle, pos: usize, n_heads: usize) -> Handle {
        let head_dim = self.config.head_dim;
        let n = n_heads * head_dim;
        let cos_sin_handle = self
            .rope_cos_sin_cache
            .borrow_mut()
            .get_or_compute(&self.client, pos, head_dim, self.config.rope_theta);
        let output_handle = self.client.empty(n * core::mem::size_of::<f32>());
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

    /// Launch CubeCL ResidualAdd entirely on GPU (handle-to-handle).
    fn dispatch_residual_add_gpu(&self, a_handle: Handle, b_handle: Handle, n: usize) -> Handle {
        let output_handle = self.client.empty(n * core::mem::size_of::<f32>());
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

    /// Launch CubeCL SwiGLU entirely on GPU (handle-to-handle).
    fn dispatch_swiglu_gpu(&self, gate_handle: Handle, up_handle: Handle, n: usize) -> Handle {
        let output_handle = self.client.empty(n * core::mem::size_of::<f32>());
        unsafe {
            SwigluCubeCL::launch::<ActiveRuntime>(
                &self.client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                n,
            );
        }
        output_handle
    }

    /// Launch fused RMSNorm + ResidualAdd on GPU.
    #[allow(dead_code)] // WIP: GPU fused norm-residual; CPU path is currently dispatched.
    fn dispatch_norm_residual_gpu(
        &self,
        input_handle: Handle,
        gamma: &Handle,
        residual_handle: Handle,
        dim: usize,
        eps: f32,
    ) -> Handle {
        let output_handle = self.client.empty(dim * core::mem::size_of::<f32>());
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

    /// GPU: Attention with GPU-resident or CPU KV cache.
    fn dispatch_attention_gpu(&self, query_handle: Handle, layer_idx: usize, pos: usize) -> Handle {
        let q_dim = self.config.n_head * self.config.head_dim;
        let n_positions = pos + 1;
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());

        let mut params = self.attn_params;
        params.n_positions = n_positions;

        if let Some(ref gpu_cache) = self.gpu_kv_cache {
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
        } else {
            let combined_kv = self.kv_cache.get_combined_kv(layer_idx, n_positions);
            let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
            AttentionCubeCL::launch::<ActiveRuntime>(
                &self.client,
                query_handle,
                kv_handle,
                attn_out_handle.clone(),
                &params,
            );
        }

        attn_out_handle
    }

    // ── GPU-resident forward pass ─────────────────────────────────────

    /// Run forward pass with GPU-resident hidden state + GPU KV cache (1 final sync).
    ///
    /// This is the fully GPU-resident version that keeps both hidden state
    /// and KV cache on GPU throughout all layers. The only sync is the
    /// final logits read.
    ///
    /// Total: 1 sync (final logits read) vs 96 (4/layer) in `forward`.
    pub fn forward_gpu(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let vocab = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;

        // 1. Embedding lookup on CPU (no sqrt scaling), upload to GPU
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i];
        }
        let mut hidden_handle = self.client.create_from_slice(f32::as_bytes(&hidden));

        // 2. Process all layers (GPU-resident, 0 sync per layer with GPU KV cache)
        for layer_idx in 0..self.config.n_layer {
            hidden_handle = self.forward_layer_gpu(hidden_handle, layer_idx, pos);
        }

        // 3. Final RMSNorm on GPU
        hidden_handle =
            self.dispatch_rmsnorm_gpu(hidden_handle, &self.gpu_norm_gammas.final_norm, n, eps);

        // 4. LM head GEMV (separate weight, NOT tied to wte)
        let logits_handle = self.dispatch_gemv_gpu(&self.weights.lm_head, hidden_handle, vocab, n);

        // 5. Final sync: read logits to CPU (no softcapping for LLaMA)
        self.read_handle(&logits_handle)
    }

    /// Single LLaMA transformer layer with GPU-resident hidden state + KV cache (0 syncs).
    ///
    /// LLaMA pre-norm architecture:
    /// ```text
    /// residual = hidden
    /// [GPU] RMSNorm(hidden, input_norm)
    /// [GPU] GEMV Q + K + V → q, k, v handles
    /// [GPU] RoPE(q, k) → q_rope, k_rope handles
    /// [GPU] KV Store(k_rope, v → gpu_cache)
    /// [GPU] Attention(q_rope, cache) → attn_out handle
    /// [GPU] GEMV Wo(attn_out) → wo handle
    /// [GPU] hidden = wo + residual              (direct add, no post-norm)
    /// residual2 = hidden
    /// [GPU] RMSNorm(hidden, post_attn_norm)
    /// [GPU] GEMV gate + up → gate, up handles
    /// [GPU] SwiGLU(gate, up) → mlp_hidden handle
    /// [GPU] GEMV down(mlp_hidden) → down handle
    /// [GPU] hidden = down + residual2           (direct add, no post-norm)
    /// ```
    fn forward_layer_gpu(&mut self, hidden_handle: Handle, layer_idx: usize, pos: usize) -> Handle {
        let n = self.config.n_embd;
        let q_dim = self.config.n_head * self.config.head_dim;
        let kv_dim = self.config.n_kv_head * self.config.head_dim;
        let mlp = self.config.mlp_hidden;
        let eps = self.config.rms_norm_eps as f32;
        let gpu_norms = &self.gpu_norm_gammas.layers[layer_idx];
        let layer_weights = &self.weights.layers[layer_idx];

        // Save residual for attention output add (pre-norm: direct add)
        let residual_handle = hidden_handle.clone();

        // ── Pass A: RMSNorm + QKV + RoPE + KV Store ──────────────────

        let normed_handle = self.dispatch_rmsnorm_gpu(hidden_handle, &gpu_norms.input_norm, n, eps);

        // GPU: QKV GEMVs
        let q = self.dispatch_gemv_gpu(&layer_weights.attn_wq, normed_handle.clone(), q_dim, n);
        let k = self.dispatch_gemv_gpu(&layer_weights.attn_wk, normed_handle.clone(), kv_dim, n);
        let v = self.dispatch_gemv_gpu(&layer_weights.attn_wv, normed_handle, kv_dim, n);

        // GPU: RoPE on Q and K
        let q_rope = self.dispatch_rope_gpu(q, pos, self.config.n_head);
        let k_rope = self.dispatch_rope_gpu(k, pos, self.config.n_kv_head);

        // GPU: Store K, V in GPU KV cache (zero-copy, no CPU sync)
        if let Some(ref gpu_cache) = self.gpu_kv_cache {
            unsafe {
                KvStoreCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    k_rope.clone(),
                    v,
                    gpu_cache.cache_handle(layer_idx).clone(),
                    gpu_cache.kv_stride(),
                    pos,
                    gpu_cache.block_size(),
                );
            }
        }

        // ── Pass B: Attention + Wo + residual ─────────────────────────

        // GPU: Attention
        let attn_out_handle = self.dispatch_attention_gpu(q_rope, layer_idx, pos);

        // GPU: Wo GEMV
        let wo_handle = self.dispatch_gemv_gpu(&layer_weights.attn_wo, attn_out_handle, n, q_dim);

        // GPU: Direct residual add (LLaMA pre-norm: no post-norm before add)
        let hidden_handle = self.dispatch_residual_add_gpu(wo_handle, residual_handle, n);

        // Save residual2 for MLP output add
        let residual2_handle = hidden_handle.clone();

        // ── Pass C: MLP ──────────────────────────────────────────────

        // GPU: RMSNorm (post_attn_norm / ffn_norm)
        let normed2 = self.dispatch_rmsnorm_gpu(hidden_handle, &gpu_norms.post_attn_norm, n, eps);

        // GPU: Gate + Up GEMVs
        let gate = self.dispatch_gemv_gpu(&layer_weights.gate_proj, normed2.clone(), mlp, n);
        let up = self.dispatch_gemv_gpu(&layer_weights.up_proj, normed2, mlp, n);

        // GPU: SwiGLU (SiLU activation, not GELU)
        let mlp_hidden = self.dispatch_swiglu_gpu(gate, up, mlp);

        // GPU: Down GEMV
        let down_handle = self.dispatch_gemv_gpu(&layer_weights.down_proj, mlp_hidden, n, mlp);

        // GPU: Direct residual add (no post-norm)
        self.dispatch_residual_add_gpu(down_handle, residual2_handle, n)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_zeros() {
        let gate = vec![0.0f32; 8];
        let up = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];
        swiglu(&gate, &up, &mut out);
        // SiLU(0) = 0 / (1 + 1) = 0.0
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_swiglu_positive() {
        let gate = vec![1.0f32; 4];
        let up = vec![2.0f32; 4];
        let mut out = vec![0.0f32; 4];
        swiglu(&gate, &up, &mut out);
        // SiLU(1.0) = 1.0 / (1 + exp(-1)) ≈ 0.7311
        // SwiGLU = SiLU(1.0) * 2.0 ≈ 1.4621
        let silu_1 = 1.0 / (1.0 + (-1.0f32).exp());
        let expected = silu_1 * 2.0;
        for v in &out {
            assert!((v - expected).abs() < 1e-4, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_swiglu_negative() {
        let gate = vec![-1.0f32; 4];
        let up = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        swiglu(&gate, &up, &mut out);
        // SiLU(-1.0) = -1.0 / (1 + exp(1)) ≈ -0.2689
        let silu_neg1 = -1.0 / (1.0 + 1.0f32.exp());
        for v in &out {
            assert!(
                (v - silu_neg1).abs() < 1e-4,
                "expected {silu_neg1}, got {v}"
            );
        }
    }

    #[test]
    fn test_silu_symmetry() {
        // SiLU(x) should have the same sign as x
        for x in [-5.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0] {
            let s = silu(x);
            if x > 0.0 {
                assert!(s > 0.0, "SiLU({x}) = {s}, expected positive");
            } else if x < 0.0 {
                assert!(s < 0.0, "SiLU({x}) = {s}, expected negative");
            } else {
                assert_eq!(s, 0.0, "SiLU(0) should be 0");
            }
        }
    }
}
