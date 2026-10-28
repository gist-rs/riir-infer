//! Dense qwen3.8 (qwen35 arch, GDN hybrid) forward on the CubeCL/wgpu runtime —
//! the Metal twin of `qwen38_dense_cudarc` (Issue 1150).
//!
//! Why: uzu and MLX are Apple-silicon-only; the only same-model comparisons our
//! engine can ever run against them require a Metal qwen3.8 path. The CUDA
//! forward (cfg not(macos)) is the bit-parity REFERENCE, not the port target —
//! this module composes the existing runtime-generic kernels (GDN, attention,
//! norms) with a new Q4_K tiled GEMM (`gemm_q4k_tile_cubecl`), accepting an
//! f32-activation GEMM class (no int8 activation quant) documented as a
//! numerics divergence from the cudarc W4A8 path.
//!
//! Config parse is deliberately self-contained (mirrors
//! `Qwen38DenseConfig::from_gguf`, same `qwen35.*` keys) so the 9k-line CUDA
//! lane is untouched — the DRY fold (shared common module) is Issue 1150's
//! follow-up once both paths are verified.
//!
//! Weight residency: layer matrices stay packed Q4_K (`Q4KHandle`, dequant
//! in-kernel); `token_embd` (Q4_K) and `output` lm_head (Q6_K) dequant to f32
//! on the host at load (5 GB each — the ternary lane's lm_head precedent) and
//! serve through the f32 `GemvCubeCL` / gather kernels.
//!
//! # Numerics class
//!
//! All activations f32; attention scale 1/√256 applied after the dot; GDN read
//! scale 1/√128; argmax first-index. GEMV/GEMM = f32 × dequant (MORE precise
//! than the cudarc int8-activation path, never bit-equal to it). End-to-end
//! acceptance = greedy-argmax parity vs llama.cpp on the same GGUF + the GEMM
//! unit G1, not bit-parity with the CUDA lane.
//!
//! # Prefill state contract
//!
//! `ingest` walks `self.chunk`-token passes (multiple of 64): the chunked
//! conv1d (sliced 64-token sub-chunks — the public launcher is conv + ordered
//! carry-update, Issue 673), the multi-token recurrence and the KV fill advance
//! every state buffer exactly as sequential decode would; the ragged tail
//! (< chunk) runs through the decode path. Chunked-conv input and output are
//! SEPARATE buffers — an in-place launch races the conv window reads against
//! the output writes (the ternary lane's swap pattern).

use std::path::Path;

use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;

use riir_infer_core::gguf_loader::{GgmlType, GgufFile};

use crate::cubecl_runtime::ActiveRuntime;
use crate::gemv_q4k_cubecl::Q4KHandle;

/// Layer kinds: GDN (deltanet) vs full attention — pattern from
/// `qwen35.full_attention_interval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen38WgpuLayerType {
    Deltanet,
    Attention,
}

/// Mirrors `Qwen38DenseConfig` (qwen38_dense_cudarc.rs:2201) — same `qwen35.*`
/// metadata keys, self-contained by design (module doc).
#[derive(Clone, Debug)]
pub struct Qwen38WgpuConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub mlp_hidden: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub d_inner: usize,
    pub conv_kernel: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub interval: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<Qwen38WgpuLayerType>,
}

impl Qwen38WgpuConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .architecture()
            .ok_or("config: missing GGUF architecture")?
            .to_string();
        if arch != "qwen35" {
            return Err(format!("config: expected qwen35 arch, got {arch}"));
        }
        let need = |key: &str| -> Result<u64, String> {
            gguf.metadata_u64(key)
                .ok_or_else(|| format!("config: missing metadata {key}"))
        };
        let n_embd = need("qwen35.embedding_length")? as usize;
        let block_count = need("qwen35.block_count")? as usize;
        let nextn = need("qwen35.nextn_predict_layers").unwrap_or(0) as usize;
        let n_layer = block_count - nextn;
        let n_head = need("qwen35.attention.head_count")? as usize;
        let n_kv_head = need("qwen35.attention.head_count_kv")? as usize;
        let head_dim = need("qwen35.attention.key_length")? as usize;
        let rms_norm_eps = gguf
            .metadata_f64("qwen35.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-6) as f32;
        let mlp_hidden = need("qwen35.feed_forward_length")? as usize;
        let n_v_heads = need("qwen35.ssm.time_step_rank")? as usize;
        let n_k_heads = need("qwen35.ssm.group_count")? as usize;
        let head_k_dim = need("qwen35.ssm.state_size")? as usize;
        let d_inner = need("qwen35.ssm.inner_size")? as usize;
        let conv_kernel = need("qwen35.ssm.conv_kernel").unwrap_or(4) as usize;
        let rotary = need("qwen35.rope.dimension_count").unwrap_or(0) as usize;
        let rotary_dim = if rotary == 0 { head_dim } else { rotary };
        let rope_theta = gguf
            .metadata_f64("qwen35.rope.freq_base")
            .unwrap_or(10_000.0) as f32;
        let interval = need("qwen35.full_attention_interval").unwrap_or(4) as usize;
        let head_v_dim = d_inner / n_v_heads;

        let mut layer_types = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layer_types.push(if (i + 1) % interval == 0 {
                Qwen38WgpuLayerType::Attention
            } else {
                Qwen38WgpuLayerType::Deltanet
            });
        }

        let vocab_size = gguf
            .tensor_info("token_embd.weight")
            .ok_or("config: missing token_embd.weight")?
            .shape
            .first()
            .copied()
            .ok_or("config: token_embd.weight has no dims")?;

        Ok(Self {
            n_embd,
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            rms_norm_eps,
            mlp_hidden,
            n_k_heads,
            n_v_heads,
            head_k_dim,
            head_v_dim,
            d_inner,
            conv_kernel,
            rotary_dim,
            rope_theta,
            interval,
            vocab_size,
            layer_types,
        })
    }

    /// GDN input-projection width: `[Q(n_k·head_k) | K(n_k·head_k) | V(n_v·head_v)]`.
    pub fn l_qkv_out(&self) -> usize {
        2 * self.n_k_heads * self.head_k_dim + self.n_v_heads * self.head_v_dim
    }
    /// Attention KV width per side (K or V): `n_kv_head · head_dim`.
    pub fn kvd(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
}

/// Per-layer weights: Q4_K handles for the big matrices, f32 handles for the
/// small vectors. `Clone` so forward loops can take an owned copy by index
/// (avoids holding `&self.layers` across `&mut self` layer calls).
#[derive(Clone)]
pub struct Qwen38WgpuLayerWeights {
    pub kind: Qwen38WgpuLayerType,
    pub attn_norm: Handle,
    pub post_attention_norm: Handle,
    pub ffn_gate: Q4KHandle,
    pub ffn_up: Q4KHandle,
    pub ffn_down: Q4KHandle,
    // GDN-only
    pub qkv: Option<Q4KHandle>,
    pub z: Option<Q4KHandle>,
    pub ssm_alpha: Option<Q4KHandle>,
    pub ssm_beta: Option<Q4KHandle>,
    pub ssm_conv1d: Option<Handle>,
    pub ssm_a: Option<Handle>,
    pub ssm_dt_bias: Option<Handle>,
    pub ssm_norm: Option<Handle>,
    pub ssm_out: Option<Q4KHandle>,
    // Attention-only
    pub attn_q: Option<Q4KHandle>,
    pub attn_k: Option<Q4KHandle>,
    pub attn_v: Option<Q4KHandle>,
    pub attn_output: Option<Q4KHandle>,
    pub attn_q_norm: Option<Handle>,
    pub attn_k_norm: Option<Handle>,
}

/// The forward. Construct with [`Qwen38WgpuForward::new`]; ingest prompts with
/// [`ingest`](Self::ingest); decode with [`forward_token`](Self::forward_token).
pub struct Qwen38WgpuForward {
    client: ComputeClient<ActiveRuntime>,
    pub config: Qwen38WgpuConfig,
    layers: Vec<Qwen38WgpuLayerWeights>,
    wte: Handle,
    lm_head: Handle,
    output_norm: Handle,
    /// Prompt-ingest chunk (tokens per batched pass); multiple of 64.
    chunk: usize,
    max_ctx: usize,
    // Scratch — batched (chunk) set.
    x_b: Handle,
    norm_x_b: Handle,
    tmp_b: Handle,
    qkv_b: Handle,
    conv_out_b: Handle,
    qkvx_exp_b: Handle,
    rec_b: Handle,
    rec_n_b: Handle,
    z_b: Handle,
    beta_b: Handle,
    decay_b: Handle,
    a_b: Handle,
    b_b: Handle,
    qg_b: Handle,
    q_b: Handle,
    gate_b: Handle,
    attn_out_b: Handle,
    k_b: Handle,
    v_b: Handle,
    gateup_b: Handle,
    up_b: Handle,
    hidden_b: Handle,
    logits_b: Handle,
    // Decode scratch.
    x: Handle,
    norm_x: Handle,
    tmp: Handle,
    qkv: Handle,
    qkv_exp: Handle,
    rec: Handle,
    z: Handle,
    beta: Handle,
    decay: Handle,
    a_raw: Handle,
    b_raw: Handle,
    qg: Handle,
    q: Handle,
    gate: Handle,
    attn_out: Handle,
    k: Handle,
    v: Handle,
    mlp_gate: Handle,
    mlp_up: Handle,
    mlp_hidden: Handle,
    logits: Handle,
    // State.
    gdn_states: Vec<Option<Handle>>,
    conv_states: Vec<Option<Handle>>,
    kv_key_caches: Vec<Option<Handle>>,
    kv_value_caches: Vec<Option<Handle>>,
    pos: usize,
}

fn f32_from_gguf(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| format!("weights: missing {name}"))?;
    let bytes = gguf
        .tensor_slice(name)
        .ok_or_else(|| format!("weights: no data for {name}"))?;
    let n = info.n_elements();
    match info.ggml_type {
        GgmlType::F32 => Ok(bytemuck::cast_slice(bytes)[..n].to_vec()),
        GgmlType::F16 => {
            let pairs: &[u16] = bytemuck::cast_slice(bytes);
            Ok(pairs[..n]
                .iter()
                .map(|&b| f32::from(half::f16::from_bits(b)))
                .collect())
        }
        GgmlType::BF16 => {
            let pairs: &[u16] = bytemuck::cast_slice(bytes);
            Ok(pairs[..n]
                .iter()
                .map(|&b| f32::from(half::bf16::from_bits(b)))
                .collect())
        }
        other => Err(format!(
            "weights: {name} is {other:?}, expected F32/F16/BF16"
        )),
    }
}

fn upload_f32(client: &ComputeClient<ActiveRuntime>, v: &[f32]) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(v))
}

fn q4k_from_gguf(
    client: &ComputeClient<ActiveRuntime>,
    gguf: &GgufFile,
    name: &str,
    m: usize,
    n: usize,
) -> Result<Q4KHandle, String> {
    let blocks = gguf
        .q4k_tensor_blocks(name)
        .map_err(|e| format!("weights: {name}: {e}"))?;
    let expect = m * (n / 256);
    if blocks.len() != expect {
        return Err(format!(
            "weights: {name} has {} blocks, expected {expect} ({m}×{n})",
            blocks.len()
        ));
    }
    Ok(Q4KHandle::from_blocks(client, blocks, m, n))
}

/// Size (f32 elements) of every state buffer, per layer kind.
struct StateDims {
    gdn: usize,
    conv: usize,
    kv: usize,
}

impl Qwen38WgpuForward {
    /// Open the GGUF, build the config, upload all weights (host-dequants the
    /// embedding + lm_head to f32 — ~10 GB of host work + buffers), allocate
    /// zeroed state and scratch for `chunk`-token ingests.
    pub fn new(gguf_path: &Path, ctx_len: usize, chunk: usize) -> Result<Self, String> {
        assert!(chunk.is_multiple_of(64), "chunk must be a multiple of 64");
        let gguf = GgufFile::open(gguf_path).map_err(|e| format!("open: {e}"))?;
        let config = Qwen38WgpuConfig::from_gguf(&gguf)?;
        let client = crate::cubecl_runtime::CubeCLContext::new()
            .map_err(|e| format!("gpu context: {e}"))?
            .client();

        let e = config.n_embd;
        let vocab = config.vocab_size;
        let l_qkv = config.l_qkv_out();
        let kvd = config.kvd();
        let n_v = config.n_v_heads;
        let hd_v = config.head_v_dim;
        let hd = config.head_dim;

        // Global tensors — embedding + lm_head host-dequant to f32.
        let wte_blocks = gguf
            .q4k_tensor_blocks("token_embd.weight")
            .map_err(|err| format!("token_embd: {err}"))?;
        let mut wte = vec![0.0f32; vocab * e];
        {
            let bpr = e / 256;
            let mut row = vec![0.0f32; e];
            for r in 0..vocab {
                riir_infer_core::quant::q4k::dequantize_row_q4_k(
                    &wte_blocks[r * bpr..(r + 1) * bpr],
                    &mut row,
                );
                wte[r * e..(r + 1) * e].copy_from_slice(&row);
            }
        }
        let wte_h = upload_f32(&client, &wte);
        let big_gb = (wte.len() + vocab * e) as f64 * 4.0 / 1e9;
        drop(wte);

        let lm_blocks = gguf
            .q6k_tensor_blocks("output.weight")
            .map_err(|err| format!("output.weight: {err}"))?;
        let mut lm = vec![0.0f32; vocab * e];
        {
            let bpr = e / 256;
            let mut row = vec![0.0f32; e];
            for r in 0..vocab {
                riir_infer_core::quant::q6k::dequantize_row_q6_k(
                    &lm_blocks[r * bpr..(r + 1) * bpr],
                    &mut row,
                );
                lm[r * e..(r + 1) * e].copy_from_slice(&row);
            }
        }
        let lm_h = upload_f32(&client, &lm);
        drop(lm);

        let output_norm = upload_f32(&client, &f32_from_gguf(&gguf, "output_norm.weight")?);

        // Per-layer weights.
        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let kind = config.layer_types[i];
            let gdn = kind == Qwen38WgpuLayerType::Deltanet;
            let attn = !gdn;
            let g = |name: String, m: usize, n: usize| -> Result<Q4KHandle, String> {
                q4k_from_gguf(&client, &gguf, &name, m, n)
            };
            let f = |name: &str| -> Result<Handle, String> {
                Ok(upload_f32(&client, &f32_from_gguf(&gguf, name)?))
            };
            layers.push(Qwen38WgpuLayerWeights {
                kind,
                attn_norm: f(&format!("blk.{i}.attn_norm.weight"))?,
                post_attention_norm: f(&format!("blk.{i}.post_attention_norm.weight"))?,
                ffn_gate: g(format!("blk.{i}.ffn_gate.weight"), config.mlp_hidden, e)?,
                ffn_up: g(format!("blk.{i}.ffn_up.weight"), config.mlp_hidden, e)?,
                ffn_down: g(format!("blk.{i}.ffn_down.weight"), e, config.mlp_hidden)?,
                qkv: gdn
                    .then(|| g(format!("blk.{i}.attn_qkv.weight"), l_qkv, e))
                    .transpose()?,
                z: gdn
                    .then(|| g(format!("blk.{i}.attn_gate.weight"), n_v * hd_v, e))
                    .transpose()?,
                ssm_alpha: gdn
                    .then(|| g(format!("blk.{i}.ssm_alpha.weight"), n_v, e))
                    .transpose()?,
                ssm_beta: gdn
                    .then(|| g(format!("blk.{i}.ssm_beta.weight"), n_v, e))
                    .transpose()?,
                ssm_conv1d: gdn
                    .then(|| f(&format!("blk.{i}.ssm_conv1d.weight")))
                    .transpose()?,
                ssm_a: gdn.then(|| f(&format!("blk.{i}.ssm_a"))).transpose()?,
                ssm_dt_bias: gdn
                    .then(|| f(&format!("blk.{i}.ssm_dt.bias")))
                    .transpose()?,
                ssm_norm: gdn
                    .then(|| f(&format!("blk.{i}.ssm_norm.weight")))
                    .transpose()?,
                ssm_out: gdn
                    .then(|| g(format!("blk.{i}.ssm_out.weight"), e, n_v * hd_v))
                    .transpose()?,
                attn_q: attn
                    .then(|| g(format!("blk.{i}.attn_q.weight"), 2 * config.n_head * hd, e))
                    .transpose()?,
                attn_k: attn
                    .then(|| g(format!("blk.{i}.attn_k.weight"), kvd, e))
                    .transpose()?,
                attn_v: attn
                    .then(|| g(format!("blk.{i}.attn_v.weight"), kvd, e))
                    .transpose()?,
                attn_output: attn
                    .then(|| g(format!("blk.{i}.attn_output.weight"), e, config.n_head * hd))
                    .transpose()?,
                attn_q_norm: attn
                    .then(|| f(&format!("blk.{i}.attn_q_norm.weight")))
                    .transpose()?,
                attn_k_norm: attn
                    .then(|| f(&format!("blk.{i}.attn_k_norm.weight")))
                    .transpose()?,
            });
        }
        drop(gguf);

        // Scratch — batched (chunk) set.
        let batch = |n_f32: usize| client.create_from_slice(&vec![0u8; n_f32 * 4]);
        let x_b = batch(chunk * e);
        let norm_x_b = batch(chunk * e);
        let tmp_b = batch(chunk * e);
        let qkv_b = batch(chunk * l_qkv);
        let conv_out_b = batch(chunk * l_qkv);
        let qkvx_exp_b = batch(chunk * 3 * n_v * hd_v);
        let rec_b = batch(chunk * n_v * hd_v);
        let rec_n_b = batch(chunk * n_v * hd_v);
        let z_b = batch(chunk * n_v * hd_v);
        let beta_b = batch(chunk * n_v);
        let decay_b = batch(chunk * n_v);
        let a_b = batch(chunk * n_v);
        let b_b = batch(chunk * n_v);
        let qg_b = batch(chunk * 2 * config.n_head * hd);
        let q_b = batch(chunk * config.n_head * hd);
        let gate_b = batch(chunk * config.n_head * hd);
        let attn_out_b = batch(chunk * config.n_head * hd);
        let k_b = batch(chunk * kvd);
        let v_b = batch(chunk * kvd);
        let gateup_b = batch(chunk * config.mlp_hidden);
        let up_b = batch(chunk * config.mlp_hidden);
        let hidden_b = batch(chunk * config.mlp_hidden);
        let logits_b = batch(vocab);

        // Decode set.
        let x = batch(e);
        let norm_x = batch(e);
        let tmp = batch(e);
        let qkv = batch(l_qkv);
        let qkv_exp = batch(3 * n_v * hd_v);
        let rec = batch(n_v * hd_v);
        let z = batch(n_v * hd_v);
        let beta = batch(n_v);
        let decay = batch(n_v);
        let a_raw = batch(n_v);
        let b_raw = batch(n_v);
        let qg = batch(2 * config.n_head * hd);
        let q = batch(config.n_head * hd);
        let gate = batch(config.n_head * hd);
        let attn_out = batch(config.n_head * hd);
        let k = batch(kvd);
        let v = batch(kvd);
        let mlp_gate = batch(config.mlp_hidden);
        let mlp_up = batch(config.mlp_hidden);
        let mlp_hidden_h = batch(config.mlp_hidden);
        let logits = batch(vocab);

        // State — wgpu buffers are zero-initialized (WebGPU spec); these
        // allocations are the zero_state of record at boot.
        let dims = StateDims {
            gdn: n_v * config.head_k_dim * hd_v,
            conv: l_qkv * config.conv_kernel,
            kv: ctx_len * kvd,
        };
        let mut gdn_states = Vec::with_capacity(config.n_layer);
        let mut conv_states = Vec::with_capacity(config.n_layer);
        let mut kv_key_caches = Vec::with_capacity(config.n_layer);
        let mut kv_value_caches = Vec::with_capacity(config.n_layer);
        for kind in &config.layer_types {
            match kind {
                Qwen38WgpuLayerType::Deltanet => {
                    gdn_states.push(Some(batch(dims.gdn)));
                    conv_states.push(Some(batch(dims.conv)));
                    kv_key_caches.push(None);
                    kv_value_caches.push(None);
                }
                Qwen38WgpuLayerType::Attention => {
                    gdn_states.push(None);
                    conv_states.push(None);
                    kv_key_caches.push(Some(batch(dims.kv)));
                    kv_value_caches.push(Some(batch(dims.kv)));
                }
            }
        }

        let gdn_count = config
            .layer_types
            .iter()
            .filter(|k| **k == Qwen38WgpuLayerType::Deltanet)
            .count();
        println!(
            "[qwen38-wgpu] loaded: {} layers ({gdn_count} GDN / {} attn), vocab {vocab}, \
             wte+lm_head f32 {big_gb:.1} GB, ctx {ctx_len}, chunk {chunk}",
            config.n_layer,
            config.n_layer - gdn_count,
        );

        Ok(Self {
            client,
            config,
            layers,
            wte: wte_h,
            lm_head: lm_h,
            output_norm,
            chunk,
            max_ctx: ctx_len,
            x_b,
            norm_x_b,
            tmp_b,
            qkv_b,
            conv_out_b,
            qkvx_exp_b,
            rec_b,
            rec_n_b,
            z_b,
            beta_b,
            decay_b,
            a_b,
            b_b,
            qg_b,
            q_b,
            gate_b,
            attn_out_b,
            k_b,
            v_b,
            gateup_b,
            up_b,
            hidden_b,
            logits_b,
            x,
            norm_x,
            tmp,
            qkv,
            qkv_exp,
            rec,
            z,
            beta,
            decay,
            a_raw,
            b_raw,
            qg,
            q,
            gate,
            attn_out,
            k,
            v,
            mlp_gate,
            mlp_up,
            mlp_hidden: mlp_hidden_h,
            logits,
            gdn_states,
            conv_states,
            kv_key_caches,
            kv_value_caches,
            pos: 0,
        })
    }

    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Full context reset (KV, GDN recurrent S, conv states — all re-zeroed).
    pub fn zero_state(&mut self) {
        let dims = StateDims {
            gdn: self.config.n_v_heads * self.config.head_k_dim * self.config.head_v_dim,
            conv: self.config.l_qkv_out() * self.config.conv_kernel,
            kv: self.max_ctx * self.config.kvd(),
        };
        let mut zeros_gdn = vec![0u8; dims.gdn * 4];
        let mut zeros_conv = vec![0u8; dims.conv * 4];
        let mut zeros_kv = vec![0u8; dims.kv * 4];
        for s in self.gdn_states.iter_mut() {
            if s.is_some() {
                *s = Some(self.client.create_from_slice(&zeros_gdn));
            }
        }
        for s in self.conv_states.iter_mut() {
            if s.is_some() {
                *s = Some(self.client.create_from_slice(&zeros_conv));
            }
        }
        for s in self.kv_key_caches.iter_mut() {
            if s.is_some() {
                *s = Some(self.client.create_from_slice(&zeros_kv));
            }
        }
        for s in self.kv_value_caches.iter_mut() {
            if s.is_some() {
                *s = Some(self.client.create_from_slice(&zeros_kv));
            }
        }
        zeros_gdn.fill(1);
        zeros_conv.fill(1);
        zeros_kv.fill(1);
        let _ = (zeros_gdn, zeros_conv, zeros_kv);
        self.pos = 0;
    }

    fn argmax_first(v: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i;
            }
        }
        best as u32
    }

    fn read_logits(&self, handle: &Handle) -> Vec<f32> {
        self.client
            .read_one(handle.clone())
            .map(|b| bytemuck::cast_slice(&b).to_vec())
            .unwrap_or_default()
    }

    /// One decode step: token at `pos`, returns the greedy argmax.
    pub fn forward_token(&mut self, token: u32, pos: usize) -> Result<u32, String> {
        if pos + 1 > self.max_ctx {
            return Err(format!("pos {} exceeds ctx {}", pos, self.max_ctx));
        }
        let vocab = self.config.vocab_size;
        let e = self.config.n_embd;
        let eps = self.config.rms_norm_eps;

        // x = wte[token] — per-token gather through a fresh 1-entry index.
        let idx = self.client.create_from_slice(bytemuck::bytes_of(&token));
        unsafe {
            crate::gemm_q4k_tile_cubecl::GatherRowsF32CubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.wte.clone(),
                idx,
                self.x.clone(),
                vocab,
                1,
                e,
            );
        }

        for li in 0..self.layers.len() {
            let lw = self.layers[li].clone();
            match lw.kind {
                Qwen38WgpuLayerType::Deltanet => self.decode_gdn_layer(li, &lw, eps),
                Qwen38WgpuLayerType::Attention => self.decode_attn_layer(li, &lw, pos, eps),
            }
        }

        // Tail: norm + lm_head + argmax.
        unsafe {
            crate::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                self.output_norm.clone(),
                self.norm_x.clone(),
                e,
                eps,
            );
            crate::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.lm_head.clone(),
                self.norm_x.clone(),
                self.logits.clone(),
                vocab,
                e,
            );
        }
        let logits = self.read_logits(&self.logits);
        self.pos = pos + 1;
        Ok(Self::argmax_first(&logits))
    }

    fn decode_gdn_layer(&mut self, li: usize, lw: &Qwen38WgpuLayerWeights, eps: f32) {
        let c = &self.config;
        let e = c.n_embd;
        let l_qkv = c.l_qkv_out();
        let n_v = c.n_v_heads;
        let hd_v = c.head_v_dim;
        let state = self.gdn_states[li].clone().expect("gdn state");
        let conv_state = self.conv_states[li].clone().expect("conv state");
        let qkv = lw.qkv.clone().expect("qkv");
        let z_w = lw.z.clone().expect("z");
        let alpha = lw.ssm_alpha.clone().expect("alpha");
        let beta_w = lw.ssm_beta.clone().expect("beta");
        let conv_w = lw.ssm_conv1d.clone().expect("conv1d");
        let ssm_a = lw.ssm_a.clone().expect("ssm_a");
        let dt_bias = lw.ssm_dt_bias.clone().expect("dt_bias");
        let ssm_norm = lw.ssm_norm.clone().expect("ssm_norm");
        let ssm_out = lw.ssm_out.clone().expect("ssm_out");

        unsafe {
            crate::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                lw.attn_norm.clone(),
                self.norm_x.clone(),
                e,
                eps,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &qkv,
                self.norm_x.clone(),
                self.qkv.clone(),
                1,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &z_w,
                self.norm_x.clone(),
                self.z.clone(),
                1,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &alpha,
                self.norm_x.clone(),
                self.a_raw.clone(),
                1,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &beta_w,
                self.norm_x.clone(),
                self.b_raw.clone(),
                1,
            );
            crate::deltanet_cubecl::DeltanetBetaDecayCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.a_raw.clone(),
                self.b_raw.clone(),
                ssm_a,
                dt_bias,
                self.beta.clone(),
                self.decay.clone(),
                n_v,
            );
            crate::deltanet_cubecl::DeltanetConv1dCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.qkv.clone(),
                conv_w,
                conv_state,
                l_qkv,
                c.conv_kernel,
            );
            crate::deltanet_cubecl::ExpandAndL2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.qkv.clone(),
                self.qkv_exp.clone(),
                c.n_k_heads,
                n_v,
                hd_v,
            );
            crate::deltanet_cubecl::DeltanetRecurrenceRowParCubeCL::launch_with_gpu_handles::<
                ActiveRuntime,
            >(
                &self.client,
                self.qkv_exp.clone(),
                self.beta.clone(),
                self.decay.clone(),
                state,
                self.rec.clone(),
                n_v,
                hd_v,
            );
            crate::norms_cubecl::RmsNormZgateFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.rec.clone(),
                ssm_norm,
                self.z.clone(),
                n_v,
                hd_v,
                eps,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &ssm_out,
                self.rec.clone(),
                self.tmp.clone(),
                1,
            );
            crate::norms_cubecl::ResidualAddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                self.tmp.clone(),
                self.x.clone(),
                e,
            );
            self.decode_mlp(lw, eps);
        }
    }

    fn decode_attn_layer(&mut self, li: usize, lw: &Qwen38WgpuLayerWeights, pos: usize, eps: f32) {
        let c = &self.config;
        let e = c.n_embd;
        let hd = c.head_dim;
        let kvd = c.kvd();
        let key_cache = self.kv_key_caches[li].clone().expect("key cache");
        let value_cache = self.kv_value_caches[li].clone().expect("value cache");
        let wq = lw.attn_q.clone().expect("attn_q");
        let wk = lw.attn_k.clone().expect("attn_k");
        let wv = lw.attn_v.clone().expect("attn_v");
        let wo = lw.attn_output.clone().expect("attn_output");
        let q_norm = lw.attn_q_norm.clone().expect("q_norm");
        let k_norm = lw.attn_k_norm.clone().expect("k_norm");

        unsafe {
            crate::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                lw.attn_norm.clone(),
                self.norm_x.clone(),
                e,
                eps,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wq,
                self.norm_x.clone(),
                self.qg.clone(),
                1,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wk,
                self.norm_x.clone(),
                self.k.clone(),
                1,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wv,
                self.norm_x.clone(),
                self.v.clone(),
                1,
            );
            crate::qwen_attention_cubecl::QwenSplitQgCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.qg.clone(),
                self.q.clone(),
                self.gate.clone(),
                hd,
                c.n_head,
            );
            crate::norms_cubecl::RmsNormQkFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.q.clone(),
                self.k.clone(),
                q_norm,
                k_norm,
                c.n_head,
                c.n_kv_head,
                hd,
                eps,
            );
            crate::qwen_attention_cubecl::QwenRopePartialCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.q.clone(),
                self.k.clone(),
                pos,
                c.rotary_dim,
                hd,
                c.n_head,
                c.n_kv_head,
                c.rope_theta,
            );
            crate::qwen_attention_cubecl::QwenKvCacheAppendCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.k.clone(),
                self.v.clone(),
                key_cache.clone(),
                value_cache.clone(),
                kvd,
                pos,
            );
            crate::qwen_attention_cubecl::QwenAttentionDecodeGatedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.q.clone(),
                key_cache,
                value_cache,
                self.gate.clone(),
                self.attn_out.clone(),
                hd,
                c.n_head,
                c.n_kv_head,
                pos + 1,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wo,
                self.attn_out.clone(),
                self.tmp.clone(),
                1,
            );
            crate::norms_cubecl::ResidualAddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                self.tmp.clone(),
                self.x.clone(),
                e,
            );
            self.decode_mlp(lw, eps);
        }
    }

    /// SwiGLU MLP (both layer kinds): norm → gate/up GEMVs → silu(gate)·up →
    /// down GEMV → residual.
    fn decode_mlp(&mut self, lw: &Qwen38WgpuLayerWeights, eps: f32) {
        let e = self.config.n_embd;
        let mlp = self.config.mlp_hidden;
        let gate = &lw.ffn_gate;
        let up = &lw.ffn_up;
        let down = &lw.ffn_down;
        unsafe {
            crate::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                lw.post_attention_norm.clone(),
                self.norm_x.clone(),
                e,
                eps,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                gate,
                self.norm_x.clone(),
                self.mlp_gate.clone(),
                1,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                up,
                self.norm_x.clone(),
                self.mlp_up.clone(),
                1,
            );
            crate::deltanet_cubecl::DeltanetGatingCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.mlp_gate.clone(),
                self.mlp_up.clone(),
                self.mlp_hidden.clone(),
                mlp,
            );
            crate::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                down,
                self.mlp_hidden.clone(),
                self.tmp.clone(),
                1,
            );
            crate::norms_cubecl::ResidualAddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                self.tmp.clone(),
                self.x.clone(),
                e,
            );
        }
    }

    /// Prompt ingest: batched chunk passes of `self.chunk` tokens (state-safe),
    /// ragged tail through the decode path. Returns the last row's greedy
    /// argmax (the decode seed).
    pub fn ingest(&mut self, tokens: &[u32], base_pos: usize) -> Result<u32, String> {
        if base_pos + tokens.len() > self.max_ctx {
            return Err(format!(
                "ingest {} tokens at {} exceeds ctx {}",
                tokens.len(),
                base_pos,
                self.max_ctx
            ));
        }
        let mut p = 0usize;
        while tokens.len() - p >= self.chunk {
            self.ingest_chunk_gpu(&tokens[p..p + self.chunk], base_pos + p);
            p += self.chunk;
        }
        let mut last = 0u32;
        for (i, &t) in tokens[p..].iter().enumerate() {
            last = self.forward_token(t, base_pos + p + i)?;
        }
        self.pos = base_pos + tokens.len();
        Ok(last)
    }

    /// One batched pass over exactly `self.chunk` tokens (a multiple of 64 —
    /// GEMM tiles 32, chunked conv walks 64-token sub-chunks).
    fn ingest_chunk_gpu(&mut self, tokens: &[u32], base_pos: usize) {
        let c = &self.config;
        let e = c.n_embd;
        let vocab = c.vocab_size;
        let p = self.chunk;

        // x_b <- wte[tokens].
        let idx_handle = self
            .client
            .create_from_slice(bytemuck::cast_slice::<u32, u8>(tokens));
        unsafe {
            crate::gemm_q4k_tile_cubecl::GatherRowsF32CubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.wte.clone(),
                idx_handle,
                self.x_b.clone(),
                c.vocab_size,
                p,
                e,
            );
        }

        let eps = eps_of(c);
        for li in 0..self.layers.len() {
            let lw = self.layers[li].clone();
            match lw.kind {
                Qwen38WgpuLayerType::Deltanet => self.prefill_gdn_layer(li, &lw, eps),
                Qwen38WgpuLayerType::Attention => self.prefill_attn_layer(li, &lw, base_pos, eps),
            }
        }

        // Last-row logits: gather row p-1, norm, f32 GEMV, (argmax by caller).
        let last_idx = self
            .client
            .create_from_slice(bytemuck::bytes_of(&(p as u32 - 1)));
        unsafe {
            crate::gemm_q4k_tile_cubecl::GatherRowsF32CubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x_b.clone(),
                last_idx,
                self.norm_x_b.clone(),
                p,
                1,
                e,
            );
            crate::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.norm_x_b.clone(),
                self.output_norm.clone(),
                self.tmp_b.clone(),
                e,
                eps,
            );
            crate::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.lm_head.clone(),
                self.tmp_b.clone(),
                self.logits_b.clone(),
                vocab,
                e,
            );
        }
    }

    fn prefill_gdn_layer(&mut self, li: usize, lw: &Qwen38WgpuLayerWeights, eps: f32) {
        let c = &self.config;
        let e = c.n_embd;
        let l_qkv = c.l_qkv_out();
        let n_v = c.n_v_heads;
        let hd_v = c.head_v_dim;
        let p = self.chunk;
        let state = self.gdn_states[li].clone().expect("gdn state");
        let conv_state = self.conv_states[li].clone().expect("conv state");
        let qkv = lw.qkv.clone().expect("qkv");
        let z_w = lw.z.clone().expect("z");
        let alpha = lw.ssm_alpha.clone().expect("alpha");
        let beta_w = lw.ssm_beta.clone().expect("beta");
        let conv_w = lw.ssm_conv1d.clone().expect("conv1d");
        let ssm_a = lw.ssm_a.clone().expect("ssm_a");
        let dt_bias = lw.ssm_dt_bias.clone().expect("dt_bias");
        let ssm_norm = lw.ssm_norm.clone().expect("ssm_norm");
        let ssm_out = lw.ssm_out.clone().expect("ssm_out");

        unsafe {
            crate::norms_cubecl::RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x_b.clone(),
                lw.attn_norm.clone(),
                self.norm_x_b.clone(),
                p,
                e,
                eps,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &qkv,
                self.norm_x_b.clone(),
                self.qkv_b.clone(),
                p,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &z_w,
                self.norm_x_b.clone(),
                self.z_b.clone(),
                p,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &alpha,
                self.norm_x_b.clone(),
                self.a_b.clone(),
                p,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &beta_w,
                self.norm_x_b.clone(),
                self.b_b.clone(),
                p,
            );
            crate::deltanet_cubecl::DeltanetBetaDecayBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.a_b.clone(),
                self.b_b.clone(),
                ssm_a,
                dt_bias,
                self.beta_b.clone(),
                self.decay_b.clone(),
                n_v,
                p,
            );
            // Chunked conv1d+SiLU over sliced 64-token sub-chunks (state-safe:
            // the public launcher is conv + ordered carry-update). Input and
            // output are SEPARATE buffers — in-place races the window reads.
            let f4 = 4u64;
            let mut t0 = 0usize;
            while t0 < p {
                let in_chunk = self
                    .qkv_b
                    .clone()
                    .offset_start(t0 as u64 * l_qkv as u64 * f4);
                let out_chunk = self
                    .conv_out_b
                    .clone()
                    .offset_start(t0 as u64 * l_qkv as u64 * f4);
                crate::deltanet_chunked_cubecl::DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    in_chunk,
                    out_chunk,
                    conv_w.clone(),
                    conv_state.clone(),
                    64,
                    l_qkv,
                    c.conv_kernel,
                    c.conv_kernel, // carry_stride = ks (conv_state layout)
                    1,             // carry_idx_offset = 1 (skip position 0)
                );
                t0 += 64;
            }
            crate::deltanet_cubecl::ExpandAndL2NormalizeHeadsBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.conv_out_b.clone(),
                self.qkvx_exp_b.clone(),
                c.n_k_heads,
                n_v,
                hd_v,
                p,
            );
            crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles::<
                ActiveRuntime,
            >(
                &self.client,
                self.qkvx_exp_b.clone(),
                self.beta_b.clone(),
                self.decay_b.clone(),
                state,
                self.rec_b.clone(),
                n_v,
                hd_v,
                p,
            );
            crate::norms_cubecl::RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.rec_b.clone(),
                ssm_norm,
                self.rec_n_b.clone(),
                p * n_v,
                hd_v,
                eps,
            );
            crate::deltanet_cubecl::DeltanetZGatingCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.rec_n_b.clone(),
                self.z_b.clone(),
                p * n_v * hd_v,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &ssm_out,
                self.rec_n_b.clone(),
                self.tmp_b.clone(),
                p,
            );
            crate::norms_cubecl::ResidualAddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x_b.clone(),
                self.tmp_b.clone(),
                self.x_b.clone(),
                p * e,
            );
            self.prefill_mlp(lw, p, eps);
        }
    }

    fn prefill_attn_layer(
        &mut self,
        li: usize,
        lw: &Qwen38WgpuLayerWeights,
        base_pos: usize,
        eps: f32,
    ) {
        let c = &self.config;
        let e = c.n_embd;
        let hd = c.head_dim;
        let kvd = c.kvd();
        let p = self.chunk;
        let key_cache = self.kv_key_caches[li].clone().expect("key cache");
        let value_cache = self.kv_value_caches[li].clone().expect("value cache");
        let wq = lw.attn_q.clone().expect("attn_q");
        let wk = lw.attn_k.clone().expect("attn_k");
        let wv = lw.attn_v.clone().expect("attn_v");
        let wo = lw.attn_output.clone().expect("attn_output");
        let q_norm = lw.attn_q_norm.clone().expect("q_norm");
        let k_norm = lw.attn_k_norm.clone().expect("k_norm");

        unsafe {
            crate::norms_cubecl::RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x_b.clone(),
                lw.attn_norm.clone(),
                self.norm_x_b.clone(),
                p,
                e,
                eps,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wq,
                self.norm_x_b.clone(),
                self.qg_b.clone(),
                p,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wk,
                self.norm_x_b.clone(),
                self.k_b.clone(),
                p,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wv,
                self.norm_x_b.clone(),
                self.v_b.clone(),
                p,
            );
            crate::qwen_attention_cubecl::QwenSplitQgBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.qg_b.clone(),
                self.q_b.clone(),
                self.gate_b.clone(),
                hd,
                c.n_head,
                p,
            );
            crate::norms_cubecl::RmsNormQkFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.q_b.clone(),
                self.k_b.clone(),
                q_norm,
                k_norm,
                p * c.n_head,
                p * c.n_kv_head,
                hd,
                eps,
            );
            crate::qwen_attention_cubecl::QwenRopePartialBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.q_b.clone(),
                self.k_b.clone(),
                c.rotary_dim,
                hd,
                c.n_head,
                c.n_kv_head,
                c.rope_theta,
                p,
                base_pos,
            );
            crate::qwen_attention_cubecl::QwenKvCacheFillSplitBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.k_b.clone(),
                self.v_b.clone(),
                key_cache.clone(),
                value_cache.clone(),
                kvd,
                p,
                base_pos,
            );
            let (kv_read, vv_read) = if base_pos > 0 {
                (key_cache.clone(), value_cache.clone())
            } else {
                (self.k_b.clone(), self.v_b.clone())
            };
            crate::qwen_attention_cubecl::QwenAttentionPrefillTiledCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.q_b.clone(),
                kv_read,
                vv_read,
                self.gate_b.clone(),
                self.attn_out_b.clone(),
                hd,
                c.n_head,
                c.n_kv_head,
                p,
                base_pos,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &wo,
                self.attn_out_b.clone(),
                self.tmp_b.clone(),
                p,
            );
            crate::norms_cubecl::ResidualAddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x_b.clone(),
                self.tmp_b.clone(),
                self.x_b.clone(),
                p * e,
            );
            self.prefill_mlp(lw, p, eps);
        }
    }

    /// SwiGLU MLP (batched).
    fn prefill_mlp(&mut self, lw: &Qwen38WgpuLayerWeights, p: usize, eps: f32) {
        let e = self.config.n_embd;
        let mlp = self.config.mlp_hidden;
        let gate = &lw.ffn_gate;
        let up = &lw.ffn_up;
        let down = &lw.ffn_down;
        unsafe {
            crate::norms_cubecl::RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x_b.clone(),
                lw.post_attention_norm.clone(),
                self.norm_x_b.clone(),
                p,
                e,
                eps,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                gate,
                self.norm_x_b.clone(),
                self.gateup_b.clone(),
                p,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                up,
                self.norm_x_b.clone(),
                self.up_b.clone(),
                p,
            );
            crate::deltanet_cubecl::DeltanetGatingCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.gateup_b.clone(),
                self.up_b.clone(),
                self.hidden_b.clone(),
                p * mlp,
            );
            crate::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                &self.client,
                down,
                self.hidden_b.clone(),
                self.tmp_b.clone(),
                p,
            );
            crate::norms_cubecl::ResidualAddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x_b.clone(),
                self.tmp_b.clone(),
                self.x_b.clone(),
                p * e,
            );
        }
    }
}

fn eps_of(c: &Qwen38WgpuConfig) -> f32 {
    c.rms_norm_eps
}
