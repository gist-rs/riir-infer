//! Gemma-2 V/K calibration lane — Issue 883 P0 (the model-side half of the
//! shared fitted-anchor-table substrate; katgpt-rs `.issues/883`, Research
//! 587).
//!
//! One offline pass over a frozen checkpoint with **V/K taps at every
//! layer** feeds [`katgpt_core::fitted_anchor_table::StreamingMeanTable`]
//! per (layer, signal ∈ {V, K, V−K}) — the closed-form estimator
//! `E_l[s] = mean(V_t − K_t | s_t = s)` (pre-RoPE K tap, post-W_V V tap),
//! the value-mean twin `E^V_l[s]`, and the R² dashboard (per-layer
//! token-explained fractions ρ_l — the go/no-go for P1's mean-removed V
//! quant, P2's fitted K=V+ retrofit, P3's cache halving).
//!
//! **Tap-point law (883 trap 1)**: K is tapped EXACTLY where the cache
//! path would consume it — after the W_K projection, BEFORE
//! [`crate::rope::apply_rope_with_freq`]; V after W_V. Gemma-2 carries no
//! QK-norm (upstream-faithful; see the architecture note in the bin), so
//! there is no post-QK-norm wrinkle for this family. The trap re-arms for
//! gemma-3/4-class stacks, where q/k norm is present.
//!
//! **No lm_head**: the calibration forward stops after the final RMSNorm —
//! the vocab projection is ~20% of per-token FLOPs and the dashboard never
//! reads logits (the `forward_gemma2_f16` structure minus steps 3–4; the
//! correspondence is pinned by the `forward_gemma2_f16` doc + test on the
//! shared per-layer shape).
//!
//! Measurement-only (the issue's own law): no quality claim at P0. The
//! tables are `StreamingMeanTable` rows — James–Stein shrinkage at
//! finalize, tail-lump lower-bound coverage, all in the shared substrate.

use anyhow::Context as _;
use katgpt_core::fitted_anchor_table::StreamingMeanTable;

use super::attention_heads_parallel;
use super::{ForwardContext, RAYON_MLP_THRESHOLD, RAYON_QKV_THRESHOLD};
use crate::gemma_layer::{GemmaLayerWeightsF16, GemmaTransformerWeightsF16};
use crate::gguf_loader::GgufFile;
use crate::types::{self, Config};
use katgpt_transformer::MultiLayerKVCache;

/// Per-layer triplet of streaming tables (V, K, V−K) — one
/// [`StreamingMeanTable`] per signal, `top_k` tracked token rows + the
/// tail lump. Constructed once; `observe` is the calibration loop's hot
/// path (alloc-free, the substrate's Bench 886 G4b).
pub struct VkLayerTables {
    pub v: StreamingMeanTable,
    pub k: StreamingMeanTable,
    pub vk: StreamingMeanTable,
}

/// The full calibration state: one table triplet per layer + the
/// token→row map (`u32::MAX` = untracked → tail) built from the corpus
/// frequency pre-pass + the V−K scratch (owned here so the observe path
/// never borrows a ForwardContext buffer whose sizing is a LoRA concern).
pub struct CalibrationTables {
    pub layers: Vec<VkLayerTables>,
    /// vocab-size map: token id → tracked row index, or `u32::MAX`.
    pub row_of_token: Vec<u32>,
    /// corpus frequency counts per token id (the Zipf shape read).
    pub token_counts: Vec<u64>,
    pub top_k: usize,
    vk_scratch: Vec<f32>,
}

impl CalibrationTables {
    /// Build tables for `n_layer` layers of `kv_dim`-wide rows, tracking
    /// the `top_k` most frequent tokens of `token_counts` (the frequency
    /// pre-pass output; ties broken by token id for determinism).
    #[must_use]
    pub fn from_counts(n_layer: usize, kv_dim: usize, token_counts: Vec<u64>, top_k: usize) -> Self {
        let vocab = token_counts.len();
        let mut order: Vec<u32> = (0..vocab as u32).filter(|&t| token_counts[t as usize] > 0).collect();
        order.sort_unstable_by(|a, b| {
            token_counts[*b as usize]
                .cmp(&token_counts[*a as usize])
                .then_with(|| a.cmp(b))
        });
        let top_k = top_k.min(order.len());
        let mut row_of_token = vec![u32::MAX; vocab];
        for (row, &tok) in order[..top_k].iter().enumerate() {
            row_of_token[tok as usize] = row as u32;
        }
        let layers = (0..n_layer)
            .map(|_| VkLayerTables {
                v: StreamingMeanTable::new(top_k, kv_dim),
                k: StreamingMeanTable::new(top_k, kv_dim),
                vk: StreamingMeanTable::new(top_k, kv_dim),
            })
            .collect();
        Self {
            layers,
            row_of_token,
            token_counts,
            top_k,
            vk_scratch: vec![0.0; kv_dim],
        }
    }

    /// Observe one token's per-layer tap pair. `k_vec` MUST be the
    /// pre-RoPE K (the tap-point law); `v_vec` the post-W_V V. The V−K
    /// residual is computed into the owned scratch — the retrofit table's
    /// exact future input.
    pub fn observe_layer(&mut self, layer: usize, token: usize, k_vec: &[f32], v_vec: &[f32]) {
        let kvd = self.vk_scratch.len();
        for i in 0..kvd {
            self.vk_scratch[i] = v_vec[i] - k_vec[i];
        }
        let t = &mut self.layers[layer];
        let row = self.row_of_token[token] as usize;
        if row != u32::MAX as usize {
            t.v.observe(row, v_vec);
            t.k.observe(row, k_vec);
            t.vk.observe(row, &self.vk_scratch);
        } else {
            t.v.observe_tail(v_vec);
            t.k.observe_tail(k_vec);
            t.vk.observe_tail(&self.vk_scratch);
        }
    }
}

/// The calibration forward: the `forward_gemma2_f16` layer stack (f16
/// weights, causal, per-token) **minus the final norm/lm_head/softcap**
/// (the dashboard never reads logits; the vocab projection is ~20% of
/// per-token FLOPs) **plus V/K taps** fed to `tables` between the QKV
/// projections and RoPE.
///
/// Sequences must stay ≤ 4096 positions (below gemma-2's sliding window —
/// this stack does not implement SWA rotation, so longer sequences would
/// tap a cache the production SWA path never builds; the bin chunks at
/// 1024).
///
/// Returns the pre-final-norm hidden state slice (the layer stack's raw
/// output — calibration callers consume K/V via `tables`, this is for
/// continuity checks only).
pub fn forward_gemma2_f16_tapped(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    tables: &mut CalibrationTables,
    token: usize,
    pos: usize,
    config: &Config,
) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] * sqrt(n_embd)  (f16 → f32)
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) =
                (*weights.wte.get_unchecked(tok_off + i)).to_f32() * embed_scale;
        }
    }

    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;

    // The V−K scratch lives in ctx.lora_buf (lora_rank ≥ kv_dim for every
    // gemma-2 config; reusing the pre-allocated buffer keeps the observe
    // path alloc-free — the substrate's G4 law).
    debug_assert!(ctx.lora_buf.len() >= kvd, "lora_buf must cover kv_dim");

    // 2. Layer loop (structure correspondence: forward_gemma2_f16 steps
    //    a–o, verbatim, with the tap inserted between d and e).
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. pre-attn RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // d. QKV projections (rayon at threshold — the production shape)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| types::matmul_f16(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| types::matmul_f16(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| types::matmul_f16(v_buf, wv, x_in, kvd, n));
            });
        } else {
            types::matmul_f16(q_buf, wq, x_in, q_dim, n);
            types::matmul_f16(k_buf, wk, x_in, kvd, n);
            types::matmul_f16(v_buf, wv, x_in, kvd, n);
        }

        // ── THE TAP (883 trap 1): pre-RoPE K, post-W_V V ──────────────
        tables.observe_layer(layer_idx, token, &ctx.k[..kvd], &ctx.v[..kvd]);

        // e. RoPE
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // f. cache K,V
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // g. attention + softcapping (parallel heads — the production shape)
        let attn_softcap = config.attn_logit_softcapping;
        ctx.attn_out[..q_dim].fill(0.0);
        unsafe {
            attention_heads_parallel(
                &ctx.q,
                &layer_cache.key,
                &layer_cache.value,
                &mut ctx.attn_out,
                &mut ctx.head_scores,
                config.n_head,
                n_kv,
                kvd,
                hd,
                t_n,
                scale,
                attn_softcap,
                config.block_size,
            );
        }

        // h. output projection
        types::matmul_f16(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // i. post-attn norm + residual
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.post_attn_norm, config.rms_norm_eps);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // j. residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        // k. pre-MLP norm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.pre_mlp_norm, config.rms_norm_eps);

        // l. GeGLU
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        let mlp_hidden = config.mlp_hidden;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| types::matmul_f16(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| types::matmul_f16(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            types::matmul_f16(gate_buf, wg, x_in, mlp_hidden, n);
            types::matmul_f16(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // m. down projection
        types::matmul_f16_parallel(&mut ctx.x, &layer_weights.down_proj, &ctx.hidden, n, mlp_hidden);

        // o. post-MLP norm + residual
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.post_mlp_norm, config.rms_norm_eps);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
}

/// Load Gemma-2 f16 weights DIRECTLY from the GGUF's F16 tensors (no f32
/// intermediate — 5.2 GiB peak instead of 10.4; the standard
/// `load_gemma2_weights_gguf` f32 path would double-peak past this box's
/// free RAM beside the 2.6 GiB table set).
///
/// F16-type tensors only: a quantized gemma-2 GGUF must go through the
/// f32 dequant path (not this loader).
pub fn load_gemma2_f16_direct(gguf: &GgufFile, config: &Config) -> anyhow::Result<GemmaTransformerWeightsF16> {
    let read_f16 = |name: &str| -> anyhow::Result<Vec<half::f16>> {
        let slice = gguf
            .tensor_slice(name)
            .context(format!("tensor '{name}' data out of bounds"))?;
        #[cfg(target_endian = "little")]
        {
            let bits = bytemuck::cast_slice::<_, u16>(slice);
            Ok(bits.iter().map(|&b| half::f16::from_bits(b)).collect())
        }
        #[cfg(target_endian = "big")]
        {
            let _ = slice;
            anyhow::bail!("big-endian host: use the f32 dequant path");
        }
    };
    let read_norm = |name: &str| -> anyhow::Result<Vec<f32>> { gguf.dequant_f16_to_f32(name) };

    let wte = read_f16("token_embd.weight")?;
    let final_norm = read_norm("output_norm.weight")?;
    let mut layers = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let f = |suffix: &str| format!("blk.{i}.{suffix}");
        layers.push(GemmaLayerWeightsF16 {
            attn_wq: read_f16(&f("attn_q.weight"))?,
            attn_wk: read_f16(&f("attn_k.weight"))?,
            attn_wv: read_f16(&f("attn_v.weight"))?,
            attn_wo: read_f16(&f("attn_output.weight"))?,
            gate_proj: read_f16(&f("ffn_gate.weight"))?,
            up_proj: read_f16(&f("ffn_up.weight"))?,
            down_proj: read_f16(&f("ffn_down.weight"))?,
            input_norm: read_norm(&f("attn_norm.weight"))?,
            post_attn_norm: read_norm(&f("post_attention_norm.weight"))?,
            pre_mlp_norm: read_norm(&f("ffn_norm.weight"))?,
            post_mlp_norm: read_norm(&f("post_ffw_norm.weight"))?,
        });
    }
    Ok(GemmaTransformerWeightsF16 {
        wte,
        final_norm,
        layers,
    })
}
