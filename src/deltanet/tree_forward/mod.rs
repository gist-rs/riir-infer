//! Tree-structured forward pass for hybrid `QwenDeltaNet` models (Plan 424 T4.3c).
//!
//! Processes ALL tree nodes through the full transformer stack in one pass,
//! using GDN rollback-free tree verification ([`katgpt_core::gdn_tree_verify`])
//! at each `DeltaNet` (linear recurrent) layer. Attention layers use per-branch
//! sequential KV-rollback (simplest correct approach for branching trees).
//!
//! # Architecture
//!
//! For a draft tree with T nodes:
//! 1. Embed all T nodes: `x[i] = wte[token_ids[i]]` (no position embedding for Qwen3.5)
//! 2. Per layer:
//!    - Save residual, input RMSNorm (all T nodes)
//!    - If DeltaNet layer: `forward_tree_deltanet_layer` — projects QKV/Z/a/b per
//!      node, computes tree-structured conv1d windows, per-head gates, then calls
//!      [`verify_gdn_tree`] per head (read-only — S₀ not modified)
//!    - If Attention layer: `forward_tree_attention_per_branch` — processes each
//!      root-to-leaf path sequentially through [`forward_attention_layer`]
//!      (KV cache is overwritten per branch; accepted path replayed at commit)
//!    - Residual add (all T nodes)
//!    - Save MLP residual, post-attn RMSNorm, SwiGLU MLP, residual add (all T nodes)
//! 3. Final `RMSNorm` + LM head (all T nodes) → per-node logits `[T * vocab_size]`
//!
//! # Cross-repo primitive consumption
//!
//! The delta-rule recurrence in `gated_deltanet_step_inplace` uses the SAME
//! math as the katgpt-core primitive:
//! ```text
//! S_new = α·S_old + β·k⊗(v − α·S_old·k)   [riir-ai: S is d_v × d_k]
//! ```
//! The katgpt-core primitive expects S₀ in `[d_k × d_v]` layout (transposed vs
//! riir-ai). State is transposed before each verify/commit call.
//!
//! Per-head α/β (`QwenDeltaNet` computes decay/beta per `v_head` per token) requires
//! per-head topology (`cumulative_log_decay` differs). The ancestor bits and topo
//! order are shared; only `cumulative_log_decay` is recomputed per head.
//!
//! # Convention alignment (T4.3b heritage)
//!
//! The tree verify primitive and `gated_deltanet_step_inplace` use the same
//! post-update readout convention: `output = S_new · q / √d_k`. The primitive's
//! internal `1/√dₖ` scale matches riir-ai's `scale = 1/√(key_dim)`.

use katgpt_core::gdn_tree_verify::{
    GdnLayerParams, GdnTreeVerifier, TreeTopology, build_topology_from_tree_nodes,
    verify_gdn_tree,
};
use katgpt_core::speculative::sampling::{sample_from_distribution, sample_residual_distribution_into};
use katgpt_core::speculative::types::{TreeNode, TreePath};
use katgpt_core::traits::NoPruner;
use katgpt_core::{Rng, softmax_scaled};
use katgpt_speculative::dd_tree::TreeBuilder;

use crate::deltanet::forward::{
    forward_attention_layer, forward_qwen_deltanet, l2_normalize, softplus, HybridCache,
    HybridForwardScratch,
};
use crate::deltanet::weights::{DeltaNetLayerWeights, QwenDeltaNetWeights};
use crate::dflash::dflash_predict_with;
#[cfg(feature = "weaver_runtime")]
use crate::dflash::dflash_predict_with_weaver;
use crate::rope::RopeFreqTable;
use crate::spec_types::SpeculativeContext;
use crate::transformer::TransformerWeights;
use crate::types::{Config, DeltaNetLayerType, rmsnorm_with_gamma_eps, swiglu};

/// Upper bound on the `DeltaNet` conv1d kernel size. Qwen3.5 ships with K=4;
/// this bound covers any realistic variant and lets the per-node ancestor
/// LUT live on the stack (no heap allocation in the conv1d hot loop).
const MAX_CONV_KERNEL_SIZE: usize = 8;

/// Recompute `cumulative_log_decay` for a topology using a different set of α values.
///
/// Called per-head when α differs across heads. The ancestor bits and topo order
/// are unchanged; only the log-decay accumulators are recomputed.
fn recompute_cumulative_log_decay(topo: &mut TreeTopology, alphas: &[f32]) {
    for k in 0..topo.n_nodes {
        let orig = topo.topo_order[k];
        let p = topo.parent[k];
        let log_alpha = (alphas[orig] as f64).ln();
        if p != usize::MAX {
            topo.cumulative_log_decay[k] = topo.cumulative_log_decay[p] + log_alpha;
        } else {
            topo.cumulative_log_decay[k] = log_alpha;
        }
    }
}

/// Transpose a `[d_v × d_k]` state (riir-ai layout) to `[d_k × d_v]` (katgpt-core layout).
///
/// `dst[m * d_v + d] = src[d * d_k + m]`
fn transpose_state(src: &[f32], dst: &mut [f32], d_k: usize, d_v: usize) {
    for m in 0..d_k {
        for d in 0..d_v {
            dst[m * d_v + d] = src[d * d_k + m];
        }
    }
}


/// Tree-structured forward for one `DeltaNet` (linear recurrent) layer.
///
/// Projects QKV/Z/a/b for all T nodes, computes tree-structured conv1d windows,
/// per-head gates, then calls [`verify_gdn_tree`] per head. The recurrent state
/// S₀ is **read-only** — use [`commit_tree_deltanet_layer`] to write the accepted path.
///
/// # Arguments
/// * `x` — `[T * n_embd]` hidden states (node-major, topo-indexed). Modified in-place
///   with the layer output.
/// * `layer` — `DeltaNet` layer weights.
/// * `state` — `[n_v_heads * key_dim * val_dim]` recurrent state (**read-only**).
/// * `conv_state` — `[conv_dim * kernel_size]` committed conv sliding window (**read-only**).
/// * `topo` — Tree topology (mutated: `cumulative_log_decay` recomputed per head).
/// * `config` — Model config.
/// * `verifier` — Pre-allocated tree verify scratch.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn forward_tree_deltanet_layer(
    x: &mut [f32],
    layer: &DeltaNetLayerWeights,
    state: &[f32],
    conv_state: &[f32],
    topo: &mut TreeTopology,
    config: &Config,
    verifier: &mut GdnTreeVerifier,
) {
    let t = topo.n_nodes;
    let n_embd = config.n_embd;
    let n_k_heads = config.deltanet_linear_n_heads;
    let n_v_heads = config.deltanet_linear_n_value_heads;
    let key_dim = config.deltanet_linear_head_dim;
    let val_dim = config.deltanet_linear_head_dim;
    let kernel_size = config.deltanet_conv_kernel_size;

    let q_dim = n_k_heads * key_dim;
    let k_dim = n_k_heads * key_dim;
    let v_dim = n_v_heads * val_dim;
    let qkv_dim = q_dim + k_dim + v_dim;
    let z_dim = v_dim;
    let conv_dim = qkv_dim;
    let state_dim_per_head = key_dim * val_dim;

    // ── 1. Project raw QKV, Z, a, b for all T nodes ──
    // raw_qkv: [T * qkv_dim], indexed by topo node k
    // raw_qkv is stored in TOPO order (for ancestor walking during conv1d)
    let mut raw_qkv = vec![0.0f32; t * qkv_dim];
    let mut z_all = vec![0.0f32; t * z_dim];
    let mut a_raw_all = vec![0.0f32; t * n_v_heads];
    let mut b_raw_all = vec![0.0f32; t * n_v_heads];

    for k in 0..t {
        let x_k = &x[k * n_embd..(k + 1) * n_embd];
        layer
            .in_proj_qkv
            .matvec(x_k, &mut raw_qkv[k * qkv_dim..(k + 1) * qkv_dim]);
        layer
            .in_proj_z
            .matvec(x_k, &mut z_all[k * z_dim..(k + 1) * z_dim]);
        layer.in_proj_a.matvec(
            x_k,
            &mut a_raw_all[k * n_v_heads..(k + 1) * n_v_heads],
        );
        layer.in_proj_b.matvec(
            x_k,
            &mut b_raw_all[k * n_v_heads..(k + 1) * n_v_heads],
        );
    }

    // ── 2. Compute conv1d output for all T nodes (tree-structured windows) ──
    // conv_out: [T * qkv_dim], stored in TOPO order
    let mut conv_out = vec![0.0f32; t * qkv_dim];

    for k in 0..t {
        let depth_k = topo.depth(k);

        // Build the conv window for node k:
        // window[ch * kernel_size + ki] for ki in 0..kernel_size
        // The window contains the last K raw QKV values along node k's branch:
        //   [committed_tail..., root_raw, ..., parent_raw, k_raw]
        // committed_tail: (K-1-depth_k) values from committed conv_state
        // tree part: (depth_k+1) values from branch (root to k inclusive)

        // Precompute ancestor node index per `ki` slot. The ancestor depends
        // only on (k, ki) — NOT on channel `ch` — so computing it inside the
        // channel loop (conv_dim ×) was pure waste. Slots in the committed
        // tail ([0, kernel_size-1-depth_k)) are unused (they read conv_state);
        // fill them with `k` as a harmless sentinel.
        let mut ancestor_lut = [k; MAX_CONV_KERNEL_SIZE];
        for ki in 0..kernel_size {
            if ki >= kernel_size - 1 - depth_k {
                let tree_pos = ki - (kernel_size - 1 - depth_k);
                let steps_up = depth_k - tree_pos;
                let mut ancestor = k;
                for _ in 0..steps_up {
                    ancestor = topo.parent[ancestor];
                }
                ancestor_lut[ki] = ancestor;
            }
        }
        let committed_count = kernel_size - 1 - depth_k;

        for ch in 0..conv_dim {
            let cw = ch * kernel_size;
            let mut sum = 0.0f32;
            // Committed-tail part: reads conv_state (channel-dependent offset).
            for ki in 0..committed_count {
                sum += conv_state[cw + depth_k + ki] * layer.conv1d_weight[cw + ki];
            }
            // Tree-branch part: reads raw_qkv at precomputed ancestor.
            for ki in committed_count..kernel_size {
                let ancestor = ancestor_lut[ki];
                sum += raw_qkv[ancestor * conv_dim + ch] * layer.conv1d_weight[cw + ki];
            }
            // SiLU activation. fast_sigmoid is the Cephes polynomial used
            // everywhere else in riir-engine (see commit 0470434).
            let sig = crate::simd::fast_sigmoid(sum);
            conv_out[k * conv_dim + ch] = sum * sig;
        }
    }

    // ── 3. Compute gates (beta, decay) for all T nodes per v_head ──
    // beta_all: [n_v_heads * T], decay_all: [n_v_heads * T] — indexed [h * T + k] (topo k)
    let mut beta_all = vec![0.0f32; n_v_heads * t];
    let mut decay_all = vec![0.0f32; n_v_heads * t];

    for k in 0..t {
        for h in 0..n_v_heads {
            let b_raw = b_raw_all[k * n_v_heads + h];
            let a_raw = a_raw_all[k * n_v_heads + h];
            beta_all[h * t + k] = crate::simd::fast_sigmoid(b_raw);
            let a_val = a_raw + layer.dt_bias[h];
            // Issue 594: `a_log` is the GGUF `ssm_a` tensor, which the
            // converter already negated + exp'd (`-exp(A_log_raw)`). Do NOT
            // re-apply `-exp()`. See forward.rs step 6 for the full note.
            let g = layer.a_log[h] * softplus(a_val);
            decay_all[h * t + k] = g.exp();
        }
    }

    // ── 4. Split conv_out into Q/K/V, expand Q/K heads, L2-normalize ──
    // Store per-head Q/K/V in ORIGINAL node index order (for the primitive).
    // q_normed_all: [n_v_heads * T * key_dim] — [h * (T*key_dim) + orig * key_dim + d]
    // k_normed_all: same layout
    // v_all:        [n_v_heads * T * val_dim]
    let mut q_normed_all = vec![0.0f32; n_v_heads * t * key_dim];
    let mut k_normed_all = vec![0.0f32; n_v_heads * t * key_dim];
    let mut v_all = vec![0.0f32; n_v_heads * t * val_dim];

    let repeat_factor = n_v_heads / n_k_heads;

    for k in 0..t {
        let orig = topo.topo_order[k];
        let conv_k = &conv_out[k * conv_dim..(k + 1) * conv_dim];
        let (q_conv, rest) = conv_k.split_at(q_dim);
        let (k_conv, v_conv) = rest.split_at(k_dim);

        // Expand Q/K heads to v_heads and L2-normalize per head.
        for h in 0..n_v_heads {
            let src_head_off = (h / repeat_factor) * key_dim;
            // Q
            let q_off = h * (t * key_dim) + orig * key_dim;
            q_normed_all[q_off..q_off + key_dim]
                .copy_from_slice(&q_conv[src_head_off..src_head_off + key_dim]);
            l2_normalize(&mut q_normed_all[q_off..q_off + key_dim]);
            // K
            let k_off = h * (t * key_dim) + orig * key_dim;
            k_normed_all[k_off..k_off + key_dim]
                .copy_from_slice(&k_conv[src_head_off..src_head_off + key_dim]);
            l2_normalize(&mut k_normed_all[k_off..k_off + key_dim]);
            // V (no expansion needed — V already has n_v_heads heads)
            let v_off = h * (t * val_dim) + orig * val_dim;
            v_all[v_off..v_off + val_dim]
                .copy_from_slice(&v_conv[h * val_dim..(h + 1) * val_dim]);
        }
    }

    // ── 5. Per-head tree verify ──
    // recurrent_output_all: [n_v_heads * T * val_dim] — per-head per-node output
    // stored in TOPO order (verify_gdn_tree returns topo-indexed output)
    let mut recurrent_output_all = vec![0.0f32; n_v_heads * t * val_dim];

    let mut s0_transposed = vec![0.0f32; state_dim_per_head];

    for h in 0..n_v_heads {
        // Recompute topology cumulative_log_decay for this head's alphas
        let decays_h = &decay_all[h * t..(h + 1) * t];
        let betas_h = &beta_all[h * t..(h + 1) * t];
        recompute_cumulative_log_decay(topo, decays_h);

        // Transpose state for this head
        let state_h = &state[h * state_dim_per_head..(h + 1) * state_dim_per_head];
        transpose_state(state_h, &mut s0_transposed, key_dim, val_dim);

        // Build params (indexed by original node index)
        let q_h = &q_normed_all[h * (t * key_dim)..(h + 1) * (t * key_dim)];
        let k_h = &k_normed_all[h * (t * key_dim)..(h + 1) * (t * key_dim)];
        let v_h = &v_all[h * (t * val_dim)..(h + 1) * (t * val_dim)];

        let params = GdnLayerParams {
            keys: k_h,
            values: v_h,
            queries: q_h,
            alphas: decays_h,
            betas: betas_h,
        };

        let out_h = verify_gdn_tree(verifier, topo, &params, &s0_transposed, key_dim, val_dim);

        // Copy to recurrent_output_all (topo-indexed)
        recurrent_output_all[h * (t * val_dim)..(h + 1) * (t * val_dim)]
            .copy_from_slice(&out_h);
    }

    // ── 6. Per-node: RMSNorm + SiLU gate + output projection ──
    // Hoist recurrent_buf outside the per-node loop — it's fully overwritten
    // each iteration via copy_from_slice, so allocating per node is pure waste.
    let mut recurrent_buf = vec![0.0f32; v_dim];
    for k in 0..t {
        // Gather per-head recurrent output for node k (topo-indexed)
        for h in 0..n_v_heads {
            let out_off = h * (t * val_dim) + k * val_dim;
            recurrent_buf[h * val_dim..(h + 1) * val_dim]
                .copy_from_slice(&recurrent_output_all[out_off..out_off + val_dim]);
        }

        // RMSNorm with linear_norm — PER HEAD (Issue 594). `ssm_norm` is one
        // `[val_dim]` gamma shared across heads; `recurrent_buf` is
        // `n_v_heads * val_dim`. See `forward::forward_deltanet_layer` step 10.
        for h in 0..n_v_heads {
            let off = h * val_dim;
            rmsnorm_with_gamma_eps(
                &mut recurrent_buf[off..off + val_dim],
                &layer.linear_norm,
                config.rms_norm_eps,
            );
        }

        // SiLU gate: output *= silu(z)
        for i in 0..z_dim {
            let z_val = z_all[k * z_dim + i];
            let sig = crate::simd::fast_sigmoid(z_val);
            recurrent_buf[i] *= z_val * sig;
        }

        // Output projection: x[k * n_embd..] = out_proj * recurrent_buf
        layer.out_proj.matvec(
            &recurrent_buf,
            &mut x[k * n_embd..(k + 1) * n_embd],
        );
    }
}

/// Tree-structured attention forward for one Attention layer (per-branch sequential).
///
/// Processes each root-to-leaf path sequentially through [`forward_attention_layer`].
/// The KV cache is overwritten per branch; the accepted path must be replayed at
/// commit time via [`commit_tree_attention_layer`].
///
/// For shared ancestors (nodes on multiple branches), the attention output is
/// recomputed for each branch but produces the same result (same hidden state,
/// same committed KV prefix). This is redundant but correct.
///
/// # Arguments
/// * `x` — `[T * n_embd]` hidden states (topo-indexed). Modified in-place per node.
/// * `layer` — Attention layer weights.
/// * `cache` — KV cache for this layer (K/V at positions >= `base_pos` are overwritten).
/// * `topo` — Tree topology.
/// * `base_pos` — Position of the root node.
/// * `config` — Model config.
/// * `rope_freq` — `RoPE` frequency table.
#[allow(clippy::too_many_arguments)]
fn forward_tree_attention_per_branch(
    x: &mut [f32],
    layer: &DeltaNetLayerWeights,
    cache: &mut crate::transformer::KVCache,
    topo: &TreeTopology,
    base_pos: usize,
    config: &Config,
    rope_freq: &RopeFreqTable,
) {
    let n = config.n_embd;
    let t = topo.n_nodes;

    // Find leaves: nodes with no children. Single-pass O(T) — mark every
    // node that appears as someone's parent, then leaves = unmarked nodes.
    // (The previous closure scanned all T parents per node → O(T²).)
    let mut has_child = vec![false; t];
    for c in 0..t {
        let p = topo.parent[c];
        if p != usize::MAX {
            has_child[p] = true;
        }
    }
    let leaves: Vec<usize> = (0..t).filter(|&k| !has_child[k]).collect();

    let mut attn_scratch = crate::deltanet::forward::AttentionLayerScratch::new(config);

    for &leaf in &leaves {
        // Reconstruct path from root to leaf (topo indices, root first)
        let mut path: Vec<usize> = Vec::with_capacity(t);
        let mut cur = leaf;
        loop {
            path.push(cur);
            if topo.parent[cur] == usize::MAX {
                break;
            }
            cur = topo.parent[cur];
        }
        path.reverse();

        // Process path tokens sequentially
        for (i, &node_k) in path.iter().enumerate() {
            let pos = base_pos + i;
            forward_attention_layer(
                &mut x[node_k * n..(node_k + 1) * n],
                layer,
                cache,
                pos,
                config,
                rope_freq,
                &mut attn_scratch,
            );
        }
    }
}

// ── Commit path (T4.3c.2) ──
//
// After forward_tree_qwen_deltanet produces per-node logits for rejection
// sampling, the accepted path must be committed to the HybridCache. We use
// sequential replay (matching katgpt-rs speculative_step_gdn_tree, which
// also chose `commit_accepted_path_sequential` over the primitive-based
// commit_gdn2_tree_layer). The accepted path is short (typically 1-4 tokens),
// so re-forward cost is negligible, and correctness is guaranteed by
// construction — it IS the sequential decode path.

/// Tree-structured forward for a hybrid `QwenDeltaNet` model.
///
/// Processes all T tree nodes simultaneously. `DeltaNet` layers use rollback-free
/// tree verification; Attention layers use per-branch sequential KV-rollback.
///
/// # Arguments
/// * `x` — Scratch buffer `[max(n_embd, vocab_size)]` (reused as in `forward_qwen_deltanet`).
///   NOT used — tree forward allocates its own `[T * n_embd]` buffer internally.
/// * `weights` — Model weights.
/// * `cache` — Hybrid cache. `DeltaNet` recurrent state is **read-only** (S₀ not
///   modified). KV cache for Attention layers is overwritten at positions >=
///   `base_pos` — the accepted path must be replayed at commit time.
/// * `topo` — Tree topology.
/// * `token_ids` — Token ID per tree node (topo-indexed: `token_ids[topo_order_inv]`).
///   Actually indexed by ORIGINAL node index — same as the topology's original indices.
/// * `base_pos` — Position of the root node.
/// * `config` — Model config.
/// * `rope_freq` — `RoPE` frequency table.
/// * `verifier` — Pre-allocated tree verify scratch (sized for `max_t >= T`).
///
/// # Returns
/// Per-node logits `[T * vocab_size]`, **topo-indexed** (node k = topo node k).
///
/// # Panics
/// Panics if `token_ids.len() != topo.n_nodes` or if the verifier is undersized.
#[allow(clippy::too_many_arguments)]
pub fn forward_tree_qwen_deltanet(
    _x: &mut [f32], // unused — kept for API parity with forward_qwen_deltanet
    weights: &QwenDeltaNetWeights,
    cache: &mut HybridCache,
    topo: &mut TreeTopology,
    token_ids: &[usize],
    base_pos: usize,
    config: &Config,
    rope_freq: &RopeFreqTable,
    verifier: &mut GdnTreeVerifier,
) -> Vec<f32> {
    let t = topo.n_nodes;
    let n = config.n_embd;
    let vocab = config.vocab_size;
    assert_eq!(token_ids.len(), t, "token_ids length must match topology node count");

    // ── Hidden states: [T * n_embd], topo-indexed ──
    let mut x = vec![0.0f32; t * n];

    // ── 1. Embed all tree nodes ──
    for k in 0..t {
        let orig = topo.topo_order[k];
        let token = token_ids[orig];
        let tok_off = token * n;
        x[k * n..(k + 1) * n].copy_from_slice(&weights.wte[tok_off..tok_off + n]);
    }

    // ── 2. Layer loop with hybrid dispatch ──
    // Pre-allocate residual scratch once and reuse across all layers — the
    // previous `flat_map(|k| x[k*n..].to_vec()).collect()` allocated T+1
    // heap buffers per layer (T `.to_vec()` intermediates + 1 collect).
    let mut residual = vec![0.0f32; t * n];
    let mut mlp_residual = vec![0.0f32; t * n];
    // MLP scratch (gate/up/down) — same size every layer, hoist out of the loop.
    let mlp_hidden = config.mlp_hidden;
    let mut gate_buf = vec![0.0f32; mlp_hidden];
    let mut up_buf = vec![0.0f32; mlp_hidden];
    let mut hidden_buf = vec![0.0f32; mlp_hidden];

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

        // a. Save residual (per node) — in-place copy, zero alloc.
        for k in 0..t {
            residual[k * n..(k + 1) * n].copy_from_slice(&x[k * n..(k + 1) * n]);
        }

        // b. Input RMSNorm (per node)
        for k in 0..t {
            rmsnorm_with_gamma_eps(
                &mut x[k * n..(k + 1) * n],
                &layer_weights.input_norm,
                config.rms_norm_eps,
            );
        }

        // c. Layer-specific forward
        if is_linear {
            forward_tree_deltanet_layer(
                &mut x,
                layer_weights,
                &cache.deltanet_state.recurrent_states[layer_idx],
                &cache.deltanet_state.conv_states[layer_idx],
                topo,
                config,
                verifier,
            );
        } else {
            // Attention: per-branch sequential (KV cache overwritten per branch).
            // The KV cache at positions >= base_pos is garbage after this call —
            // the accepted path must be replayed via forward_attention_layer at commit.
            forward_tree_attention_per_branch(
                &mut x,
                layer_weights,
                &mut cache.kv_cache.layers[layer_idx],
                topo,
                base_pos,
                config,
                rope_freq,
            );
        }

        // d. Residual add (per node) — flat iteration over `t*n` elements
        // enables LLVM to lower to a single SIMD axpy loop (vs the nested
        // k,i loop, which emits a bounds check per `k`).
        for (x, &r) in x.iter_mut().zip(&residual[..t * n]) {
            *x += r;
        }

        // e. Save MLP residual (per node) — in-place copy, zero alloc.
        for k in 0..t {
            mlp_residual[k * n..(k + 1) * n].copy_from_slice(&x[k * n..(k + 1) * n]);
        }

        // f. Post-attention RMSNorm (per node)
        for k in 0..t {
            rmsnorm_with_gamma_eps(
                &mut x[k * n..(k + 1) * n],
                &layer_weights.post_attn_norm,
                config.rms_norm_eps,
            );
        }

        // g. SwiGLU MLP (per node)
        for k in 0..t {
            layer_weights
                .gate_proj
                .matvec(&x[k * n..(k + 1) * n], &mut gate_buf);
            layer_weights
                .up_proj
                .matvec(&x[k * n..(k + 1) * n], &mut up_buf);
            swiglu(&mut hidden_buf, &gate_buf, &up_buf);
            layer_weights
                .down_proj
                .matvec(&hidden_buf, &mut x[k * n..(k + 1) * n]);
        }

        // h. Residual add (per node) — flat iteration (see step d above).
        for (x, &r) in x.iter_mut().zip(&mlp_residual[..t * n]) {
            *x += r;
        }
    }

    // ── 3. Final RMSNorm (per node) ──
    for k in 0..t {
        rmsnorm_with_gamma_eps(&mut x[k * n..(k + 1) * n], &weights.final_norm, config.rms_norm_eps);
    }

    // ── 4. LM head (per node) ──
    let mut logits = vec![0.0f32; t * vocab];
    let mut hidden_copy = vec![0.0f32; n];
    for k in 0..t {
        hidden_copy[..n].copy_from_slice(&x[k * n..(k + 1) * n]);
        weights.lm_head.matvec(
            &hidden_copy[..n],
            &mut logits[k * vocab..(k + 1) * vocab],
        );
    }

    logits
}

/// Commit the accepted path after tree verification: sequential replay (T4.3c.2).
///
/// After [`forward_tree_qwen_deltanet`] produces per-node logits for rejection
/// sampling, the accepted path must be committed to the [`HybridCache`] to
/// advance the recurrent state (`DeltaNet` layers) and fix the KV cache
/// (Attention layers — positions >= `base_pos` are garbage after tree verify).
///
/// This function replays the accepted tokens through [`forward_qwen_deltanet`]
/// sequentially. This matches the katgpt-rs `speculative_step_gdn_tree`
/// pattern, which also commits via sequential replay
/// (`commit_accepted_path_sequential`) rather than the primitive-based
/// `commit_gdn2_tree_layer`. The accepted path is short (typically 1-4
/// tokens), so the re-forward cost is negligible, and correctness is
/// guaranteed by construction — it IS the sequential decode path.
///
/// # Cache state after this call
///
/// - **`DeltaNet` layers:** recurrent state and `conv_state` advanced as if the
///   accepted tokens had been decoded sequentially from `base_pos`.
/// - **Attention layers:** KV cache at positions `[base_pos, base_pos + len)`
///   holds the accepted tokens' K/V (overwriting any garbage from tree verify).
///
/// # Arguments
/// * `weights` — Model weights.
/// * `cache` — Hybrid cache (mutated in-place).
/// * `accepted_tokens` — Accepted token IDs, root first.
/// * `base_pos` — Position of the first accepted token.
/// * `config` — Model config.
/// * `scratch` — Pre-allocated forward scratch.
/// * `rope_freq` — `RoPE` frequency table.
///
/// # Panics
/// Panics if `accepted_tokens` is empty.
pub fn commit_tree_qwen_deltanet(
    weights: &QwenDeltaNetWeights,
    cache: &mut HybridCache,
    accepted_tokens: &[usize],
    base_pos: usize,
    config: &Config,
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
) {
    assert!(
        !accepted_tokens.is_empty(),
        "commit_tree_qwen_deltanet: accepted_tokens must be non-empty"
    );
    let buf_size = config.n_embd.max(config.vocab_size);
    let mut x = vec![0.0f32; buf_size];

    for (i, &token) in accepted_tokens.iter().enumerate() {
        let pos = base_pos + i;
        forward_qwen_deltanet(
            &mut x,
            weights,
            cache,
            token,
            pos,
            config,
            scratch,
            rope_freq,
        );
    }
}

// ── Speculative step (T4.3c.3) ──
//
// Full hybrid QwenDeltaNet speculative tree decode pipeline:
//   1. Draft marginals via DFlash (standard transformer drafter)
//   2. Build DDTree from marginals
//   3. `forward_tree_qwen_deltanet` — verify all nodes in one pass
//   4. p/q rejection sampling along best path
//   5. `commit_tree_qwen_deltanet` — commit accepted tokens via sequential replay
//   6. Return accepted tokens + count
//
// Design: drafter is a standard `TransformerWeights` (DFlash); target is a
// `QwenDeltaNetWeights` (hybrid DeltaNet+Attention). This matches the
// speculative_step_gdn_tree pattern from katgpt-rs, where the drafter and
// target are different model types. The drafter produces marginals for tree
// construction + p/q rejection; the target verifies via tree forward.

// Note: path-prefix → topo-index lookup was previously an O(T) linear scan
// (`find_topo_node_for_path`); it is now a precomputed HashMap built once per
// spec step (see `path_to_topo` in `spec_step_deltanet_post_draft`).

/// Encode a path prefix at a given depth, following the `DDTree` convention:
/// one `u32` slot per level, root at slot 0 (`TreePath::push`).
fn encode_path_prefix(path: &[usize], up_to_depth: usize) -> TreePath {
    path.iter()
        .take(up_to_depth + 1)
        .enumerate()
        .fold(TreePath::default(), |acc, (d, &tok)| acc.push(tok as u32, d))
}

/// Extract candidate verification paths from a `DDTree` (top-3 root branches).
///
/// This is the riir-ai local copy of `katgpt_forward::step::extract_ddtree_paths`,
/// kept here to avoid the `katgpt-forward` → `katgpt-transformer` heavy dep
/// chain and because the algorithm is small (O(N) single pass). Each branch
/// follows the best child at subsequent depths.
fn extract_ddtree_paths(tree: &[TreeNode]) -> Vec<Vec<usize>> {
    use std::collections::HashMap;
    if tree.is_empty() {
        return Vec::new();
    }

    let mut max_depth: usize = 0;
    let mut roots: Vec<&TreeNode> = Vec::new();
    let mut child_index: HashMap<(usize, TreePath), &TreeNode> = HashMap::new();

    for node in tree.iter() {
        if node.depth > max_depth {
            max_depth = node.depth;
        }
        if node.depth == 0 {
            roots.push(node);
        } else {
            let key = (node.depth, node.parent_path.parent(node.depth));
            child_index
                .entry(key)
                .and_modify(|existing| {
                    if node.score > existing.score {
                        *existing = node;
                    }
                })
                .or_insert(node);
        }
    }

    roots.sort_by(|a, b| b.score.total_cmp(&a.score));
    roots.truncate(3);

    let mut paths = Vec::with_capacity(roots.len());
    for root in roots {
        let mut path = vec![root.token_idx];
        let mut current_path = root.parent_path;
        for depth in 1..=max_depth {
            match child_index.get(&(depth, current_path)) {
                Some(node) => {
                    path.push(node.token_idx);
                    current_path = node.parent_path;
                }
                None => break,
            }
        }
        paths.push(path);
    }
    paths
}

/// Hybrid `QwenDeltaNet` speculative tree decode step (Plan 424 T4.3c.3).
///
/// Orchestrates the full speculative pipeline for a hybrid `QwenDeltaNet` target
/// model with a standard transformer `DFlash` drafter:
///
/// 1. Draft marginals via [`dflash_predict_with`] (standard transformer drafter)
/// 2. Build `DDTree` via [`TreeBuilder::build`]
/// 3. Verify all nodes via [`forward_tree_qwen_deltanet`] (read-only on `DeltaNet`
///    recurrent state; Attention KV cache left garbage at positions ≥ `pos`)
/// 4. p/q rejection sampling along the best path using target logits vs drafter
///    marginals
/// 5. Commit accepted tokens via [`commit_tree_qwen_deltanet`] (sequential
///    replay — advances DeltaNet state, fixes Attention KV cache)
///
/// # Arguments
/// * `draft_sctx` — Drafter speculative context (marginals buffer + scratch).
/// * `tree_builder` — Pre-allocated `DDTree` builder.
/// * `draft_weights` / `draft_config` — Draft model (standard transformer).
/// * `target_weights` — Target model (hybrid `QwenDeltaNet`, for verification).
/// * `target_cache` — Target hybrid cache (mutated by the commit step).
/// * `target_config` — Target model config.
/// * `verifier` — Pre-allocated tree verify scratch.
/// * `target_scratch` — Pre-allocated forward scratch for the commit replay.
/// * `rope_freq` — `RoPE` frequency table (target).
/// * `token` / `pos` — Current token and position.
/// * `rng` — Random number generator (for rejection sampling).
///
/// # Returns
/// `(accepted_tokens, num_accepted)` — `accepted_tokens` includes the bonus
/// token if all draft tokens are accepted. At least one token is always
/// returned (the fallback).
///
/// # Panics
/// Panics if the drafter produces zero marginals AND no fallback is possible,
/// or if the verifier scratch is undersized for the tree.
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_qwen_deltanet_tree(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &QwenDeltaNetWeights,
    target_cache: &mut HybridCache,
    target_config: &Config,
    verifier: &mut GdnTreeVerifier,
    target_scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Draft marginals via DFlash
    dflash_predict_with(draft_sctx, draft_weights, draft_config, token, pos);

    spec_step_deltanet_post_draft(
        draft_sctx.steps_populated,
        &draft_sctx.marginals_flat,
        tree_builder,
        draft_config,
        target_weights,
        target_cache,
        target_config,
        verifier,
        target_scratch,
        rope_freq,
        token,
        pos,
        rng,
    )
}

/// Weaver-corrected sibling of `speculative_step_qwen_deltanet_tree` (Plan 434).
///
/// Drops in `dflash_predict_with_weaver` for the draft step, then delegates
/// to the same post-draft pipeline (`DDTree` build → tree forward → p/q reject
/// → commit). Callers allocate `h_dflash_captured` once sized
/// `[draft_config.draft_lookahead * draft_config.n_embd]` and `weaver_scratch`
/// once via `WeaverScratch::new(&weaver.config)`.
///
/// See `dflash_predict_with_weaver` for the no-harm contract: zero-weight
/// Weaver weights leave the marginals bit-identical to the base path.
#[cfg(feature = "weaver_runtime")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_qwen_deltanet_tree_with_weaver(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &QwenDeltaNetWeights,
    target_cache: &mut HybridCache,
    target_config: &Config,
    verifier: &mut GdnTreeVerifier,
    target_scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    h_dflash_captured: &mut [f32],
    weaver: &katgpt_speculative::weaver::WeaverCorrector,
    weaver_scratch: &mut katgpt_speculative::weaver::WeaverScratch,
) -> (Vec<usize>, usize) {
    // 1. Draft marginals via DFlash + Weaver correction.
    //    h_verifier = verifier's preserved hidden state from the prior commit
    //    (scratch.hidden_copy is the aliasing-avoidance copy made before the
    //    lm_head matmul in forward_qwen_deltanet). On cold-start it is zeros,
    //    which Weaver's no-harm contract handles (zero hidden → zero residual).
    //    embedding = target_weights.wte ([vocab_size, n_embd] row-major).
    let n_embd = target_config.n_embd;
    let h_verifier = &target_scratch.hidden_copy[..n_embd];
    let embedding = &target_weights.wte;
    let _ = dflash_predict_with_weaver(
        draft_sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        h_dflash_captured,
        weaver,
        h_verifier,
        embedding,
        weaver_scratch,
    );

    spec_step_deltanet_post_draft(
        draft_sctx.steps_populated,
        &draft_sctx.marginals_flat,
        tree_builder,
        draft_config,
        target_weights,
        target_cache,
        target_config,
        verifier,
        target_scratch,
        rope_freq,
        token,
        pos,
        rng,
    )
}

/// Post-draft pipeline shared by the base and Weaver-corrected spec step
/// variants (Plan 434 T1). Consumes the already-populated marginals + step
/// count and runs: `DDTree` build → tree forward → p/q reject → commit.
#[allow(clippy::too_many_arguments)]
fn spec_step_deltanet_post_draft(
    steps_populated: usize,
    marginals_flat: &[f32],
    tree_builder: &mut TreeBuilder,
    draft_config: &Config,
    target_weights: &QwenDeltaNetWeights,
    target_cache: &mut HybridCache,
    target_config: &Config,
    verifier: &mut GdnTreeVerifier,
    target_scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    let vocab_size = target_config.vocab_size;

    // Build marginals view (same as speculative_step_rollback_with)
    let mut marginals_buf: [&[f32]; 64] = [&[]; 64];
    let count = steps_populated.min(64);
    for (i, slot) in marginals_buf.iter_mut().enumerate().take(count) {
        let start = i * vocab_size;
        let end = start + vocab_size;
        *slot = if end <= marginals_flat.len() && i < steps_populated {
            &marginals_flat[start..end]
        } else {
            &[]
        };
    }
    let marginals = &marginals_buf[..count];

    // 2. Build DDTree
    let tree = tree_builder.build(marginals, draft_config, &NoPruner, false);
    let tree_owned: Vec<TreeNode> = tree.to_vec(); // detach from builder (it reuses buffers)

    if tree_owned.is_empty() {
        let fallback = sample_from_distribution(
            marginals.first().copied().unwrap_or(&[1.0]),
            rng,
        );
        return (vec![fallback], 1);
    }

    // 3. Build tree topology from DDTree nodes.
    // The scalar alpha is a placeholder — `forward_tree_deltanet_layer` calls
    // `recompute_cumulative_log_decay` per head with the actual per-head
    // decays derived from the layer's `a_log`/`in_proj_a` projections, so this
    // initial value is overwritten. Use 0.99 (same default as katgpt-rs).
    let (mut topo, token_ids) = build_topology_from_tree_nodes(&tree_owned, 0.99);

    // 4. Forward all tree nodes through the target (read-only verify).
    // DeltaNet recurrent state is read-only; Attention KV cache at positions
    // ≥ pos is garbage (overwritten per branch). The commit step fixes both.
    let buf_size = target_config.n_embd.max(target_config.vocab_size);
    let mut x_scratch = vec![0.0f32; buf_size];
    let tree_logits = forward_tree_qwen_deltanet(
        &mut x_scratch,
        target_weights,
        target_cache,
        &mut topo,
        &token_ids,
        pos,
        target_config,
        rope_freq,
        verifier,
    );

    // 5. Extract candidate paths (top-3 root branches) and p/q reject.
    let paths = extract_ddtree_paths(&tree_owned);

    // Build a lookup map (depth, parent_path) -> topo index, so the rejection
    // loop below is O(1) per lookup instead of O(T) linear scan per (path,
    // depth). The (depth, parent_path) pair is unique per tree node by the
    // DDTree path-encoding invariant.
    use std::collections::HashMap;
    let path_to_topo: HashMap<(usize, TreePath), usize> = (0..topo.n_nodes)
        .map(|k| {
            let orig = topo.topo_order[k];
            let node = &tree_owned[orig];
            ((node.depth, node.parent_path), k)
        })
        .collect();

    if paths.is_empty() {
        // Fallback: sample from the root node's logits
        let root_logits = &tree_logits[0..vocab_size];
        let mut probs = vec![0.0f32; vocab_size];
        probs[..vocab_size].copy_from_slice(root_logits);
        softmax_scaled(&mut probs, 1.0 / target_config.temperature);
        let fallback = sample_from_distribution(&probs, rng);
        return (vec![fallback], 1);
    }

    // 6. Try each candidate path with p/q rejection sampling.
    // Hoist a single vocab_size probs buffer outside the path/depth loops —
    // the previous code did `node_logits.to_vec()` per (path, depth), each
    // cloning ~vocab_size (~150K for Qwen) floats. Qwen3.5-B = 151K f32 ≈
    // 600KB allocated+freed per rejection step.
    let mut probs_buf: Vec<f32> = vec![0.0f32; vocab_size];
    let mut residual_buf: Vec<f32> = Vec::new();

    for path in &paths {
        let mut accepted = Vec::with_capacity(path.len());
        let mut all_accepted = true;

        for (depth, &draft_tok) in path.iter().enumerate() {
            let current_path_prefix = encode_path_prefix(path, depth);

            // Find the topo node matching this path prefix and use its logits.
            // O(1) HashMap lookup replaces the O(T) linear scan.
            let Some(&topo_k) = path_to_topo.get(&(depth, current_path_prefix))
            else {
                // No matching tree node — can't verify this token
                all_accepted = false;
                break;
            };

            let node_logits = &tree_logits[topo_k * vocab_size..(topo_k + 1) * vocab_size];
            probs_buf[..vocab_size].copy_from_slice(node_logits);
            softmax_scaled(&mut probs_buf, 1.0 / target_config.temperature);

            let q_dist = marginals.get(depth).copied().unwrap_or(&[]);
            let q_i = q_dist.get(draft_tok).copied().unwrap_or(0.0);
            let p_i = probs_buf.get(draft_tok).copied().unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };

            if rng.uniform() <= acceptance_prob {
                accepted.push(draft_tok);
            } else {
                residual_buf.clear();
                residual_buf.resize(vocab_size, 0.0);
                let replacement =
                    sample_residual_distribution_into(&probs_buf, q_dist, &mut residual_buf, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        // Bonus token if all accepted
        if all_accepted && !accepted.is_empty() {
            let last_depth = path.len() - 1;
            let last_prefix = encode_path_prefix(path, last_depth);
            if let Some(&topo_k) = path_to_topo.get(&(last_depth, last_prefix)) {
                let bonus_logits = &tree_logits[topo_k * vocab_size..(topo_k + 1) * vocab_size];
                probs_buf[..vocab_size].copy_from_slice(bonus_logits);
                softmax_scaled(&mut probs_buf, 1.0 / target_config.temperature);
                let bonus = sample_from_distribution(&probs_buf, rng);
                accepted.push(bonus);
            }
        }

        if !accepted.is_empty() {
            // 7. Commit the accepted path (exclude bonus from the committed
            //    prefix — the bonus token becomes the next prompt for the
            //    drafter but its state is committed by the next step's commit).
            //
            //    Actually, matching katgpt-rs: commit ALL accepted tokens
            //    INCLUDING the bonus (the bonus is a target-verified token
            //    from the last node's logits). The commit replays them through
            //    forward_qwen_deltanet sequentially, advancing the cache state
            //    for all of them.
            commit_tree_qwen_deltanet(
                target_weights,
                target_cache,
                &accepted,
                pos,
                target_config,
                target_scratch,
                rope_freq,
            );

            let len = accepted.len();
            return (accepted, len);
        }
    }

    // All paths exhausted: forward current token through target, sample
    let n = target_config.n_embd;
    let buf = n.max(vocab_size);
    let mut x_fallback = vec![0.0f32; buf];
    let logits = forward_qwen_deltanet(
        &mut x_fallback,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
        target_scratch,
        rope_freq,
    );
    probs_buf[..vocab_size].copy_from_slice(&logits[..vocab_size]);
    softmax_scaled(&mut probs_buf, 1.0 / target_config.temperature);
    let fallback = sample_from_distribution(&probs_buf, rng);
    (vec![fallback], 1)
}


// ── Tests ────────────────────────────────────────────────────────────────────
//
// Extracted to `tests.rs` per Issue 530 — the inline test block was 952
// lines, pushing tree_forward.rs over the 2048 soft limit. Parent declares
// the test mod via #[path] so use super::* continues to resolve all items.

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
