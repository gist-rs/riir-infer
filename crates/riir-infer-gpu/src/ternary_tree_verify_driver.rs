//! Tree-masked batched verify driver (Issue 721 T4 + T4a) —
//! `forward_tree_verify`.
//!
//! Drives a whole speculative draft **tree** through the GPU-resident
//! Ternary-Bonsai forward in one dispatch chain per layer:
//!
//! - **DeltaNet layers (48/64)**: the masked-solve kernels
//!   ([`crate::deltanet_tree_verify_cubecl`]) with T-row batched projections
//!   ([`crate::GemmTernaryBatchedCubeCL`]). Weights read once per layer
//!   regardless of T — the wall-clock prize.
//! - **Attention layers (16/64)**: ancestor-masked batched attention
//!   (Issue 721 T4a — [`crate::QwenAttentionTreeGatedCubeCL`]): batched
//!   Q/KV projections, per-node-position RoPE
//!   ([`crate::QwenRopePartialTreeCubeCL`]), then ONE attention dispatch per
//!   layer whose K/V set is [committed cache prefix ∪ ancestor-or-self tree
//!   nodes]. Verify never writes the KV cache — the commit replay owns those
//!   positions.
//! - **Commit**: sequential replay of the accepted path through the existing
//!   decode forward — identical to the CPU oracle's `commit_tree_qwen_deltanet`
//!   design ("correctness guaranteed by construction — it IS the sequential
//!   decode path"). Zero new commit machinery.
//!
//! The verify pass is **read-only** on ALL state: DeltaNet recurrent + conv
//! state AND the attention KV caches (positions ≥ `base_pos` are simply not
//! written — the commit replay fills the accepted path).

#![allow(clippy::too_many_arguments)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use riir_infer_core::types::DeltaNetLayerType;

use crate::cubecl_runtime::ActiveRuntime;
use crate::deltanet_tree_verify_cubecl::{
    GatingConcatBatchedCubeCL, RowGatherScatterCubeCL, Split4BatchedCubeCL, TreeBuildRhsCubeCL,
    TreeBuildXYCubeCL, TreeComputeOutCubeCL, TreeConv1dGatherCubeCL,
    TreeCumulativeLogDecayCubeCL, TreeForwardSubCubeCL, TreeVerifyPlan,
};
use crate::elementwise_cubecl::CopyCubeCL;
use crate::gemm_ternary_batched_cubecl::GemmTernaryBatchedCubeCL;
use crate::norms_cubecl::{
    ResidualAddCubeCL, RmsNormBatchedCubeCL, RmsNormQkFusedCubeCL, RmsNormZgateFusedCubeCL,
};
use crate::qwen_attention_cubecl::{
    QwenAttentionTreeGatedCubeCL, QwenRopePartialTreeCubeCL, QwenSplitKvBatchedCubeCL,
    QwenSplitQgBatchedCubeCL,
};
use crate::ternary_deltanet_gpu_forward::{DequantWteRowCubeCL, TernaryDeltanetGpuForward};

/// Persistent T-row buffers for tree verify, allocated once at `t_max`
/// (G4: steady-state verify performs no allocation).
pub struct TreeVerifyGpuBuffers {
    pub t_max: usize,
    // Topology uploads (rewritten per cycle; sized at t_max)
    pub parent: Handle,   // [t_max] u32
    pub conv_lut: Handle, // [t_max × K] u32
    pub anc_lo: Handle,   // [t_max] u32
    pub anc_hi: Handle,   // [t_max] u32
    // Hidden-state axis [t_max × n]
    pub x: Handle,
    pub norm_x: Handle,
    pub resid: Handle, // reused for input + mlp residual saves
    pub tmp: Handle,   // layer output before residual add
    pub ffn_out: Handle,
    // DeltaNet projections [t_max × proj_dim]
    pub input_proj_out: Handle,
    pub qkv_raw: Handle, // [t_max × conv_dim] — pre-conv projections
    pub z: Handle,       // [t_max × z_dim]
    pub a_raw: Handle,   // [t_max × n_v]
    pub b_raw: Handle,   // [t_max × n_v]
    pub conv_out: Handle,     // [t_max × conv_dim] — silu conv output
    pub qkv_expanded: Handle, // [t_max × 3·n_v·d]
    pub beta: Handle,         // [t_max × n_v]
    pub decay: Handle,        // [t_max × n_v]
    pub cld: Handle,          // [t_max × n_v]
    pub recurrent_out: Handle, // [t_max × n_v·d] — token-major solve output
    // Masked-solve scratch
    pub x_mat: Handle, // [n_v × t_max × t_max]
    pub y_mat: Handle, // [n_v × t_max × t_max]
    pub rhs: Handle,   // [n_v × t_max × d]
    pub u: Handle,     // [n_v × t_max × d]
    // FFN [t_max × ·]
    pub ffn_gate_up: Handle, // [t_max × 2·mlp]
    pub ffn_hidden: Handle,  // [t_max × mlp]
    // Attention-layer scratch (Issue 721 T4a — batched tree attention)
    pub attn_qg: Handle,    // [t_max × 2·q_dim] — interleaved [q, gate] per head
    pub attn_kv: Handle,    // [t_max × 2·kvd] — concatenated [K, V]
    pub attn_q: Handle,     // [t_max × q_dim]
    pub attn_gate: Handle,  // [t_max × q_dim]
    pub attn_k: Handle,     // [t_max × kvd] (post-RoPE)
    pub attn_v: Handle,     // [t_max × kvd]
    pub attn_out: Handle,   // [t_max × q_dim] (gated)
    pub attn_pos: Handle,   // u32 [t_max] — RoPE positions = base_pos + depth
    /// Single-row scratch for the attention KV commit gather (T6 fast commit).
    pub attn_k_row: Handle, // [kvd]
    pub attn_v_row: Handle, // [kvd]
    // Output [t_max × vocab]
    pub logits: Handle,
    /// Scratch row for the wte dequant → scatter bridge (n elements).
    pub wte_row: Handle,
}

impl TernaryDeltanetGpuForward {
    /// Allocate (or grow) the tree buffers for `t` nodes.
    fn ensure_tree_buffers(&mut self, t: usize) {
        let need_alloc = match &self.tree_buffers {
            None => true,
            Some(b) => b.t_max < t,
        };
        if !need_alloc {
            return;
        }
        let t_max = t;
        let n = self.config.n_embd;
        let n_v = self.config.deltanet_linear_n_value_heads;
        let n_k = self.config.deltanet_linear_n_heads;
        let d = self.config.deltanet_linear_head_dim;
        let k = self.config.deltanet_conv_kernel_size;
        let mlp = self.config.mlp_hidden;
        let vocab = self.config.vocab_size;
        let conv_dim = 2 * (n_k * d) + n_v * d;
        let z_dim = n_v * d;
        let proj_dim = conv_dim + z_dim + 2 * n_v;
        let q_dim = self.config.n_head * self.config.head_dim;
        let kvd = self.config.n_kv_head * self.config.head_dim;
        let f32b = |len: usize| self.client.empty(len * core::mem::size_of::<f32>());
        let u32b = |len: usize| self.client.empty(len * core::mem::size_of::<u32>());

        self.tree_buffers = Some(TreeVerifyGpuBuffers {
            t_max,
            parent: u32b(t_max),
            conv_lut: u32b(t_max * k),
            anc_lo: u32b(t_max),
            anc_hi: u32b(t_max),
            x: f32b(t_max * n),
            norm_x: f32b(t_max * n),
            resid: f32b(t_max * n),
            tmp: f32b(t_max * n),
            ffn_out: f32b(t_max * n),
            input_proj_out: f32b(t_max * proj_dim),
            qkv_raw: f32b(t_max * conv_dim),
            z: f32b(t_max * z_dim),
            a_raw: f32b(t_max * n_v),
            b_raw: f32b(t_max * n_v),
            conv_out: f32b(t_max * conv_dim),
            qkv_expanded: f32b(t_max * 3 * n_v * d),
            beta: f32b(t_max * n_v),
            decay: f32b(t_max * n_v),
            cld: f32b(t_max * n_v),
            recurrent_out: f32b(t_max * n_v * d),
            x_mat: f32b(n_v * t_max * t_max),
            y_mat: f32b(n_v * t_max * t_max),
            rhs: f32b(n_v * t_max * d),
            u: f32b(n_v * t_max * d),
            ffn_gate_up: f32b(t_max * 2 * mlp),
            ffn_hidden: f32b(t_max * mlp),
            attn_qg: f32b(t_max * 2 * q_dim),
            attn_kv: f32b(t_max * 2 * kvd),
            attn_q: f32b(t_max * q_dim),
            attn_gate: f32b(t_max * q_dim),
            attn_k: f32b(t_max * kvd),
            attn_v: f32b(t_max * kvd),
            attn_out: f32b(t_max * q_dim),
            attn_pos: u32b(t_max),
            attn_k_row: f32b(kvd),
            attn_v_row: f32b(kvd),
            logits: f32b(t_max * vocab),
            wte_row: f32b(n),
        });
    }

    /// Verify a speculative draft tree — per-node logits, one batched read.
    ///
    /// # Arguments
    /// * `tokens` — token ids per node, **topo-ordered** (parents precede
    ///   children), aligned with `plan`.
    /// * `plan` — the precomputed topology artifacts ([`TreeVerifyPlan`]).
    ///
    /// # Returns
    /// Per-node logits `[T][vocab]`, topo-indexed (row k = node k).
    ///
    /// # State contract (mirrors the CPU oracle)
    /// * DeltaNet recurrent + conv state: **read-only**.
    /// * Attention KV caches: **read-only** — verify never writes them (the
    ///   ancestor-masked kernel reads the committed prefix + tree K/V from
    ///   the batched projection output). The accepted path is committed via
    ///   [`Self::commit_tree_verify`], whose sequential replay fills positions
    ///   `[base_pos, base_pos + len)`.
    /// * `self.pos`: unchanged.
    ///
    /// # Panics
    /// Panics if `tokens.len() != plan.t`.
    pub fn forward_tree_verify(
        &mut self,
        tokens: &[usize],
        plan: &TreeVerifyPlan,
    ) -> Vec<Vec<f32>> {
        assert_eq!(tokens.len(), plan.t, "tokens must align with the plan");
        // Plan 602 B3 — the tree-verify batched lane has no rotation wiring
        // (its concat GEMM consumes `norm_x` directly); running folded
        // weights unrotated is the silent-garbage class. Refuse LOUD — the
        // cudarc precedent for replay-style lanes (the per-token
        // `forward_speculative_verify` DOES carry the rotation via
        // `forward_dispatch_only`).
        if self.rotation.is_some() {
            panic!(
                "Bonsai-2 folded tree-verify on the CubeCL forward: not wired \
                 (Plan 602 B3) — use forward_speculative_verify (the per-token \
                 verify carries the rotation)"
            );
        }
        // Issue 965 flush-before-read: verify reads the CubeCL state handles
        // (recurrent + KV), which a graph-armed prefill leaves behind the
        // cudarc mirrors until the spec flush.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_flush(self);
        let t = plan.t;
        self.ensure_tree_buffers(t);

        let n = self.config.n_embd;
        let eps = self.config.rms_norm_eps as f32;
        let mlp = self.config.mlp_hidden;
        let vocab = self.config.vocab_size;
        let kernel_size = self.config.deltanet_conv_kernel_size;

        // ── 0. Upload the topology artifacts ──
        {
            let bufs = self.tree_buffers.as_ref().expect("tree buffers");
            let t_max = bufs.t_max;
            let parent_pad = pad_u32(&plan.parent, t_max, u32::MAX);
            let lut_pad = pad_u32(&plan.conv_lut, t_max * kernel_size, 0);
            let lo_pad = pad_u32(&plan.anc_lo, t_max, 0);
            let hi_pad = pad_u32(&plan.anc_hi, t_max, 0);
            // RoPE positions: base_pos + depth (T4a — the attention layers'
            // per-node rotations). Tree rows are topo-indexed; positions are
            // depth-indexed — the upload carries the mapping.
            let base_pos = self.pos;
            let pos_pad: Vec<u32> = (0..t_max)
                .map(|k| {
                    if k < t {
                        (base_pos + plan.depth[k]) as u32
                    } else {
                        0
                    }
                })
                .collect();
            self.client.write(
                &bufs.parent,
                cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&parent_pad).to_vec()),
            );
            self.client.write(
                &bufs.conv_lut,
                cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&lut_pad).to_vec()),
            );
            self.client.write(
                &bufs.anc_lo,
                cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&lo_pad).to_vec()),
            );
            self.client.write(
                &bufs.anc_hi,
                cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&hi_pad).to_vec()),
            );
            self.client.write(
                &bufs.attn_pos,
                cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(&pos_pad).to_vec()),
            );
        }

        // ── 1. Embed all T nodes: dequant wte row → scratch → scatter ──
        {
            let (pos_bits, neg_bits, group_scale) = {
                let w = &self.wte_handle;
                (
                    w.pos_bits_u32.clone(),
                    w.neg_bits_u32.clone(),
                    w.group_scale_f32.clone(),
                )
            };
            let blocks64 = self.wte_handle.blocks64 as u32;
            let groups_per_row = self.wte_handle.groups_per_row as u32;
            for (k, &tok) in tokens.iter().enumerate() {
                let bufs = self.tree_buffers.as_ref().expect("tree buffers");
                unsafe {
                    DequantWteRowCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        pos_bits.clone(),
                        neg_bits.clone(),
                        group_scale.clone(),
                        bufs.wte_row.clone(),
                        tok as u32,
                        blocks64,
                        groups_per_row,
                        n as u32,
                    );
                    RowGatherScatterCubeCL::scatter::<ActiveRuntime>(
                        &self.client,
                        bufs.wte_row.clone(),
                        bufs.x.clone(),
                        n,
                        k,
                    );
                }
            }
        }

        // ── 2. Layer loop ──
        for layer_idx in 0..self.layers.len() {
            let is_deltanet = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

            // a. resid = x
            {
                let bufs = self.tree_buffers.as_ref().expect("tree buffers");
                unsafe {
                    CopyCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        bufs.x.clone(),
                        bufs.resid.clone(),
                        t * n,
                    );
                }
            }

            // b. norm_x = rmsnorm(x, input_norm) over T rows
            {
                let layer_w = &self.layers[layer_idx];
                let bufs = self.tree_buffers.as_ref().expect("tree buffers");
                unsafe {
                    RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        bufs.x.clone(),
                        layer_w.input_norm.clone(),
                        bufs.norm_x.clone(),
                        t,
                        n,
                        eps,
                    );
                }
            }

            // c. Layer forward → tmp rows
            if is_deltanet {
                self.tree_deltanet_layer(layer_idx, t);
            } else {
                self.tree_attention_layer_batched(layer_idx, t);
            }

            // d. x += tmp
            {
                let bufs = self.tree_buffers.as_ref().expect("tree buffers");
                unsafe {
                    ResidualAddCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        bufs.x.clone(),
                        bufs.tmp.clone(),
                        bufs.x.clone(),
                        t * n,
                    );
                }
            }

            // e. resid = x; norm_x = rmsnorm(x, post_attn_norm)
            {
                let layer_w = &self.layers[layer_idx];
                let bufs = self.tree_buffers.as_ref().expect("tree buffers");
                unsafe {
                    CopyCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        bufs.x.clone(),
                        bufs.resid.clone(),
                        t * n,
                    );
                    RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        bufs.x.clone(),
                        layer_w.post_attn_norm.clone(),
                        bufs.norm_x.clone(),
                        t,
                        n,
                        eps,
                    );
                }
            }

            // f. FFN: gate_up GEMM → SwiGLU → down GEMM
            {
                let layer_w = &self.layers[layer_idx];
                let bufs = self.tree_buffers.as_ref().expect("tree buffers");
                unsafe {
                    GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        &layer_w.gate_up_proj,
                        bufs.norm_x.clone(),
                        bufs.ffn_gate_up.clone(),
                        t,
                    );
                    GatingConcatBatchedCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        bufs.ffn_gate_up.clone(),
                        bufs.ffn_hidden.clone(),
                        mlp,
                        t,
                    );
                    GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        &layer_w.down_proj,
                        bufs.ffn_hidden.clone(),
                        bufs.ffn_out.clone(),
                        t,
                    );
                }
            }

            // g. x += ffn_out
            {
                let bufs = self.tree_buffers.as_ref().expect("tree buffers");
                unsafe {
                    ResidualAddCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        bufs.x.clone(),
                        bufs.ffn_out.clone(),
                        bufs.x.clone(),
                        t * n,
                    );
                }
            }
        }

        // ── 3. Final norm + lm head over T rows ──
        {
            let bufs = self.tree_buffers.as_ref().expect("tree buffers");
            unsafe {
                RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    bufs.x.clone(),
                    self.final_norm.clone(),
                    bufs.norm_x.clone(),
                    t,
                    n,
                    eps,
                );
                GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &self.lm_head,
                    bufs.norm_x.clone(),
                    bufs.logits.clone(),
                    t,
                );
            }
        }

        // ── 4. One read of the whole [T × vocab] logits buffer ──
        // The buffer is sized at t_max (reused across cycles with varying t —
        // the bench's per-cycle trees); only the first t rows are valid.
        let bufs = self.tree_buffers.as_ref().expect("tree buffers");
        let bytes = self
            .client
            .read_one(bufs.logits.clone())
            .expect("read tree logits");
        let flat = f32::from_bytes(&bytes);
        assert!(
            flat.len() >= t * vocab,
            "logits buffer ({}) smaller than t*vocab ({t}*{vocab})",
            flat.len()
        );
        flat[..t * vocab]
            .chunks(vocab)
            .map(<[f32]>::to_vec)
            .collect()
    }

    /// Batched tree forward for one DeltaNet layer (the masked-solve path).
    fn tree_deltanet_layer(&mut self, layer_idx: usize, t: usize) {
        let n_v = self.config.deltanet_linear_n_value_heads;
        let n_k = self.config.deltanet_linear_n_heads;
        let d = self.config.deltanet_linear_head_dim;
        let k = self.config.deltanet_conv_kernel_size;
        let conv_dim = 2 * (n_k * d) + n_v * d;
        let z_dim = n_v * d;
        let eps = self.config.rms_norm_eps as f32;

        let state = self.deltanet_states[layer_idx]
            .as_ref()
            .expect("state for DeltaNet layer")
            .clone();
        let conv_state = self.conv_states[layer_idx]
            .as_ref()
            .expect("conv_state for DeltaNet layer")
            .clone();
        let layer_w = &self.layers[layer_idx];
        let bufs = self.tree_buffers.as_ref().expect("tree buffers");

        unsafe {
            // 1. Batched input projections (qkv|z|a|b) — one weight read.
            GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_w.in_proj_concat,
                bufs.norm_x.clone(),
                bufs.input_proj_out.clone(),
                t,
            );
            // 2. Per-row split.
            Split4BatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.input_proj_out.clone(),
                bufs.qkv_raw.clone(),
                bufs.z.clone(),
                bufs.a_raw.clone(),
                bufs.b_raw.clone(),
                conv_dim,
                z_dim,
                n_v,
                n_v,
                t,
            );
            // 3. Tree conv1d gather (read-only conv_state).
            TreeConv1dGatherCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.qkv_raw.clone(),
                layer_w.conv1d_weight.clone(),
                conv_state,
                bufs.conv_lut.clone(),
                bufs.conv_out.clone(),
                t,
                conv_dim,
                k,
            );
            // 4. Beta/decay.
            crate::deltanet_cubecl::DeltanetBetaDecayBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.a_raw.clone(),
                bufs.b_raw.clone(),
                layer_w.a_log.clone(),
                layer_w.dt_bias.clone(),
                bufs.beta.clone(),
                bufs.decay.clone(),
                n_v,
                t,
            );
            // 5. Expand + L2-normalize heads.
            crate::deltanet_cubecl::ExpandAndL2NormalizeHeadsBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.conv_out.clone(),
                bufs.qkv_expanded.clone(),
                n_k,
                n_v,
                d,
                t,
            );
            // 6. Per-head cumulative log-decay.
            TreeCumulativeLogDecayCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.decay.clone(),
                bufs.parent.clone(),
                bufs.cld.clone(),
                n_v,
                t,
            );
            // 7-10. The masked solve.
            TreeBuildXYCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.qkv_expanded.clone(),
                bufs.beta.clone(),
                bufs.cld.clone(),
                bufs.anc_lo.clone(),
                bufs.anc_hi.clone(),
                bufs.x_mat.clone(),
                bufs.y_mat.clone(),
                n_v,
                t,
                d,
            );
            TreeBuildRhsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.qkv_expanded.clone(),
                bufs.beta.clone(),
                bufs.cld.clone(),
                state.clone(),
                bufs.rhs.clone(),
                n_v,
                t,
                d,
            );
            TreeForwardSubCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.x_mat.clone(),
                bufs.rhs.clone(),
                bufs.u.clone(),
                n_v,
                t,
                d,
            );
            TreeComputeOutCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.qkv_expanded.clone(),
                bufs.y_mat.clone(),
                bufs.u.clone(),
                bufs.cld.clone(),
                state,
                bufs.recurrent_out.clone(),
                n_v,
                t,
                d,
            );
            // 11. Per-head RMSNorm + z-gate (the kernel is row-indexed —
            // dispatching T·n_v rows batches it with zero kernel changes).
            RmsNormZgateFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.recurrent_out.clone(),
                layer_w.linear_norm.clone(),
                bufs.z.clone(),
                t * n_v,
                d,
                eps,
            );
            // 12. Batched out projection.
            GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_w.out_proj,
                bufs.recurrent_out.clone(),
                bufs.tmp.clone(),
                t,
            );
        }
    }

    /// Ancestor-masked batched attention for one attention layer (Issue 721
    /// T4a) — 8 dispatches for ALL T nodes, replacing the per-branch bridge's
    /// ~(#node-visits × 8) dispatches.
    ///
    /// Pipeline (batched over T topo rows; the committed KV prefix and the
    /// tree's ancestor-or-self K/V set are consumed by ONE attention
    /// dispatch — verify never writes the KV caches):
    ///
    /// 1. `qg = W_qg @ norm_x` (batched ternary GEMM)
    /// 2. `kv = W_kv @ norm_x` (batched ternary GEMM)
    /// 3. split QG → Q + gate
    /// 4. split KV → K + V
    /// 5. fused per-head Q/K RMSNorm (row-indexed kernel — T·n_head rows)
    /// 6. partial RoPE at per-node positions (`base_pos + depth`, uploaded)
    /// 7. [`QwenAttentionTreeGatedCubeCL`] — online-softmax attention over
    ///    [committed cache 0..base_pos] ∪ [ancestor-or-self tree rows]
    /// 8. `tmp = W_o @ attn_out` (batched ternary GEMM)
    fn tree_attention_layer_batched(&mut self, layer_idx: usize, t: usize) {
        let n_head = self.config.n_head;
        let n_kv = self.config.n_kv_head;
        let hd = self.config.head_dim;
        let kvd = n_kv * hd;
        let eps = self.config.rms_norm_eps as f32;
        let rotary_dim = if self.config.rope_dimension_count > 0 {
            self.config.rope_dimension_count
        } else {
            hd // full rotation
        };
        let theta_base = self.config.rope_theta;
        // Committed KV prefix: positions 0..base_pos were written by the
        // decode path before verify; verify reads them and writes nothing.
        let n_committed = self.pos;

        let bufs = self.tree_buffers.as_ref().expect("tree buffers");
        let layer_w = &self.layers[layer_idx];
        let wq = layer_w.attn_wq.as_ref().expect("attn_wq for Attention layer");
        let wkv = layer_w
            .attn_wkv
            .as_ref()
            .expect("attn_wkv for Attention layer");
        let wo = layer_w
            .attn_wo
            .as_ref()
            .expect("attn_wo for Attention layer");
        let q_norm = layer_w
            .attn_q_norm
            .as_ref()
            .expect("attn_q_norm for Attention layer");
        let k_norm = layer_w
            .attn_k_norm
            .as_ref()
            .expect("attn_k_norm for Attention layer");
        let key_cache = self.kv_key_caches[layer_idx]
            .as_ref()
            .expect("key cache for Attention layer");
        let value_cache = self.kv_value_caches[layer_idx]
            .as_ref()
            .expect("value cache for Attention layer");

        unsafe {
            // 1–2. Batched projections from the T-row normed hidden states.
            GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                wq,
                bufs.norm_x.clone(),
                bufs.attn_qg.clone(),
                t,
            );
            GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                wkv,
                bufs.norm_x.clone(),
                bufs.attn_kv.clone(),
                t,
            );
            // 3–4. Splits.
            QwenSplitQgBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.attn_qg.clone(),
                bufs.attn_q.clone(),
                bufs.attn_gate.clone(),
                hd,
                n_head,
                t,
            );
            QwenSplitKvBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.attn_kv.clone(),
                bufs.attn_k.clone(),
                bufs.attn_v.clone(),
                kvd,
                t,
            );
            // 5. Fused Q/K per-head RMSNorm — the kernel is row-indexed, so
            //    T·n_head / T·n_kv rows serve T nodes with no changes.
            RmsNormQkFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.attn_q.clone(),
                bufs.attn_k.clone(),
                q_norm.clone(),
                k_norm.clone(),
                t * n_head,
                t * n_kv,
                hd,
                eps,
            );
            // 6. Partial RoPE at each node's uploaded position.
            QwenRopePartialTreeCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.attn_q.clone(),
                bufs.attn_k.clone(),
                bufs.attn_pos.clone(),
                rotary_dim,
                hd,
                n_head,
                n_kv,
                theta_base,
                t,
            );
            // 7. Ancestor-masked gated attention: committed prefix + tree.
            QwenAttentionTreeGatedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                bufs.attn_q.clone(),
                bufs.attn_k.clone(),
                bufs.attn_v.clone(),
                key_cache.clone(),
                value_cache.clone(),
                bufs.attn_gate.clone(),
                bufs.anc_lo.clone(),
                bufs.anc_hi.clone(),
                bufs.attn_out.clone(),
                hd,
                n_head,
                n_kv,
                t,
                n_committed,
            );
            // 8. Batched out-projection → tmp rows (the caller adds resid).
            GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                wo,
                bufs.attn_out.clone(),
                bufs.tmp.clone(),
                t,
            );
        }
    }

    /// Commit the accepted path after tree verification: sequential replay.
    ///
    /// Mirrors the CPU oracle's `commit_tree_qwen_deltanet` — re-forward each
    /// accepted token through the normal decode path, which advances the
    /// DeltaNet recurrent/conv state and overwrites the per-branch attention
    /// KV garbage at positions `[base_pos, base_pos + len)`.
    ///
    /// Returns the final logits (the prediction for the next position).
    pub fn commit_tree_verify(
        &mut self,
        weights: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
        accepted_tokens: &[usize],
    ) -> Vec<f32> {
        assert!(
            !accepted_tokens.is_empty(),
            "commit requires at least one accepted token"
        );
        let mut logits = Vec::new();
        for &tok in accepted_tokens {
            self.set_input_token(weights, tok);
            logits = self.forward_token();
        }
        logits
    }

    /// Fast state commit for the accepted tree path (Issue 721 T6): rank-1
    /// replay from the verify's OWN tree-buffer rows — **zero weight reads**.
    ///
    /// Per DeltaNet layer, per path node (root→leaf order): conv-state
    /// advance ([`DeltanetConv1dCubeCL`] on the node's raw qkv row — the same
    /// kernel the decode path runs per token) + recurrent-state update
    /// ([`DeltanetRecurrenceCubeCL`] on the node's expanded qkv/beta/decay
    /// rows, gathered from the tree buffers). Per attention layer: KV-cache
    /// append of the node's post-RoPE K + raw V at `base_pos + step` via
    /// [`QwenKvCacheAppendCubeCL`].
    ///
    /// This replaces `(1 + path_len)` full decode forwards (~43 ms each on
    /// Metal — weight-read bound) with `path_len × (6·n_deltanet + 3·n_attn)`
    /// small dispatches (~1–3 ms total) — the commit half of the issue's
    /// ~1.06 forwards/token economics.
    ///
    /// Numerical contract: the state after this commit matches the
    /// sequential-replay commit ([`Self::commit_tree_verify`]) within the
    /// batched-vs-single GEMM summation-order tolerance (the G1 class) — the
    /// replayed rows come from the verify's batched kernels; the state-update
    /// kernels are the decode path's own. The next `forward_tree_verify`
    /// builds on this state exactly as it would on the sequential commit.
    ///
    /// # Panics
    /// Panics if `path_nodes` is empty or contains an out-of-range index.
    /// `debug_assert`s head_dim == 128 (the legacy recurrence kernel's
    /// hard-coded smem tree — the same constraint the decode path carries).
    pub fn commit_tree_verify_fast(&mut self, path_nodes: &[usize]) {
        assert!(
            !path_nodes.is_empty(),
            "fast commit requires a non-empty accepted path"
        );
        // Issue 965 flush-before-commit: the commit reads the verify buffers
        // AND the CubeCL state handles (recurrent/conv/KV) it advances.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_flush(self);
        let n_v = self.config.deltanet_linear_n_value_heads;
        let n_k = self.config.deltanet_linear_n_heads;
        let d = self.config.deltanet_linear_head_dim;
        let hd = self.config.head_dim;
        let n_kv = self.config.n_kv_head;
        let kvd = n_kv * hd;
        let k = self.config.deltanet_conv_kernel_size;
        let conv_dim = 2 * (n_k * d) + n_v * d;
        let base_pos = self.pos;
        let bufs = self.tree_buffers.as_ref().expect("tree buffers");

        for layer_idx in 0..self.layers.len() {
            let is_deltanet = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
            let layer_w = &self.layers[layer_idx];
            for (step, &node) in path_nodes.iter().enumerate() {
                let pos = base_pos + step;
                if is_deltanet {
                    unsafe {
                        // Conv-state advance: the node's RAW pre-conv row →
                        // the decode conv1d kernel (shift + append + dot+silu
                        // into scratch — only the state write matters here).
                        RowGatherScatterCubeCL::gather::<ActiveRuntime>(
                            &self.client,
                            bufs.qkv_raw.clone(),
                            self.qkv.clone(),
                            conv_dim,
                            node,
                        );
                        let conv_state = self.conv_states[layer_idx]
                            .as_ref()
                            .expect("conv_state for DeltaNet layer");
                        crate::deltanet_cubecl::DeltanetConv1dCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            self.qkv.clone(),
                            layer_w.conv1d_weight.clone(),
                            conv_state.clone(),
                            conv_dim,
                            k,
                        );
                        // Rank-1 recurrent update from the verify's rows.
                        RowGatherScatterCubeCL::gather::<ActiveRuntime>(
                            &self.client,
                            bufs.qkv_expanded.clone(),
                            self.qkv_expanded.clone(),
                            3 * n_v * d,
                            node,
                        );
                        RowGatherScatterCubeCL::gather::<ActiveRuntime>(
                            &self.client,
                            bufs.beta.clone(),
                            self.beta_buf.clone(),
                            n_v,
                            node,
                        );
                        RowGatherScatterCubeCL::gather::<ActiveRuntime>(
                            &self.client,
                            bufs.decay.clone(),
                            self.decay_buf.clone(),
                            n_v,
                            node,
                        );
                        let state = self.deltanet_states[layer_idx]
                            .as_ref()
                            .expect("state for DeltaNet layer");
                        crate::deltanet_cubecl::DeltanetRecurrenceCubeCL::launch_with_gpu_handles::<
                            ActiveRuntime,
                        >(
                            &self.client,
                            self.qkv_expanded.clone(),
                            self.beta_buf.clone(),
                            self.decay_buf.clone(),
                            state.clone(),
                            self.recurrent_out.clone(),
                            n_v,
                            d,
                        );
                    }
                } else {
                    unsafe {
                        // KV-cache append: post-RoPE K + raw V rows at pos.
                        RowGatherScatterCubeCL::gather::<ActiveRuntime>(
                            &self.client,
                            bufs.attn_k.clone(),
                            bufs.attn_k_row.clone(),
                            kvd,
                            node,
                        );
                        RowGatherScatterCubeCL::gather::<ActiveRuntime>(
                            &self.client,
                            bufs.attn_v.clone(),
                            bufs.attn_v_row.clone(),
                            kvd,
                            node,
                        );
                        let key_cache = self.kv_key_caches[layer_idx]
                            .as_ref()
                            .expect("key cache for Attention layer");
                        let value_cache = self.kv_value_caches[layer_idx]
                            .as_ref()
                            .expect("value cache for Attention layer");
                        crate::qwen_attention_cubecl::QwenKvCacheAppendCubeCL::launch::<
                            ActiveRuntime,
                        >(
                            &self.client,
                            bufs.attn_k_row.clone(),
                            bufs.attn_v_row.clone(),
                            key_cache.clone(),
                            value_cache.clone(),
                            kvd,
                            pos,
                        );
                    }
                }
            }
        }
        self.pos = base_pos + path_nodes.len();
    }
}

/// Pad a u32 slice to `len` (per-cycle uploads write the full t_max buffers).
fn pad_u32(src: &[u32], len: usize, fill: u32) -> Vec<u32> {
    let mut out = src.to_vec();
    out.resize(len, fill);
    out
}
