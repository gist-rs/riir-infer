//! eDLM (Drex DLM) segment-structured scoring lane — riir-infer Issue 1005.
//!
//! Loads and serves `nace-ai/drex-dlm` (Efficient-DLM-8B backbone, GGUF arch
//! key `edlm`): a Qwen3-family transformer scored through a segment-based
//! block-causal mask instead of plain causality. One forward pass produces
//! hidden states; a pointer head reads per-option probabilities off the
//! `<decide>`/`</opt>` marker positions.
//!
//! Clean-room from the wire + published mechanics (their code is MIT —
//! reference reading only; weights CC BY-NC 4.0 — local bench/comparison
//! serving only, never a product lane, never a distill teacher).
//!
//! Phase 1 (this module):
//! - T1 `load_edlm_weights_gguf` — `edlm` GGUF arch support (Qwen3 tensor
//!   naming, `attn_{q,k}_norm` QK-RMSNorm carried, `pointer.*` head tensors,
//!   NEOX RoPE — NO Q/K row permutation, unlike the `llama` arch loader).
//! - T2 `branch_mask` — segment mask: `attend(i,j) iff j<=i and
//!   (seg[j]==0 or seg[j]==seg[i])`, plus `state_bidir` (state↔state both
//!   directions) and the option-isolation variant; pads (`seg < 0`) are keys
//!   to nobody and produce garbage queries (never read). Generalizes
//!   `attention::block_causal_t_n` from a prefix length to arbitrary
//!   per-(query, key) eligibility.
//! - T3 `rows_of` + `forward_edlm_rows` — row-form fallback: every question
//!   runs as a self-contained causal row `state + branch` (same positions as
//!   the packed form), the mask-parity escape for over-long packed sequences.
//!   Parity with the packed form holds by construction (the KV set of a
//!   branch query is identical in both forms — state never sees branches) and
//!   is pinned by a test on both `state_bidir` postures.
//!
//! Phase 2/3 (separate units): marker encode/pack (T4), pointer-head module
//! assembly + end-to-end parity vs their published outputs (T5/T6), state
//! prefix KV reuse + GPU forward (T7/T8).
//!
//! Reference distillation: the subagent note on `nace-ai/drex-dlm` @ `6c63df2`
//! (`code/kev/model.py::branch_mask_batch` + `rows_of` + `PointerHead`) and
//! the `nace-ai/llama.cpp` branch `edlm` (`src/models/edlm.cpp` — the C++
//! mask predicate is the same law; `pointer.temperature.weight` is the GGUF
//! tensor name, NE 1). Every name/key below is header-verified against the
//! real `drex-dlm-Q8_0.gguf` (2026-10-08).

use crate::llama_layer::LlamaLayerWeights;
use crate::rope::RopeFreqTable;
use anyhow::{bail, Result};

/// Segment id: option-isolation "no option" (state / instruction token).
pub const OPT_NONE: i32 = -1;
/// Segment id: option-isolation `<decide>` marker (sees every span).
pub const OPT_DECIDE: i32 = -2;

// ── T2 — the branch mask ────────────────────────────────────────────

/// Build the eDLM branch attention mask.
///
/// `seg[t]`: `0` = state, `k ≥ 1` = question-`k` branch, `< 0` = pad.
/// Writes row-major `out[i * l + j] = true` iff query `i` may attend key `j`:
///
/// - causal: `j <= i`
/// - keys are the state (`seg[j] == 0`) or the query's own segment
/// - `state_bidir`: state↔state attends both directions (the released
///   posture; the state prefix KV cache stays exact because state still never
///   sees branches)
/// - `opts` (option isolation, same length as `seg`): a key inside an option
///   span (`opt >= 0`) is visible only to the decide marker
///   (`opt == OPT_DECIDE`) or the same span — spans never see each other
/// - pads (`seg < 0`) are visible to nobody; every row keeps its diagonal
///   (softmax never runs on an all-blocked row; pad rows produce garbage that
///   is never read)
///
/// This is the additive-mask law their Python (`branch_mask_batch`,
/// blocked = `finfo(dtype).min`, allowed = 0) and their C++ server
/// (`-INFINITY`, allowed = 0) both realize; the bool form feeds
/// `attention_head_masked` which skips ineligible keys outright.
pub fn branch_mask(seg: &[i32], state_bidir: bool, opts: Option<&[i32]>, out: &mut [bool]) {
    let l = seg.len();
    assert_eq!(
        out.len(),
        l * l,
        "branch_mask out buffer must be l*l ({l}*{l}), got {}",
        out.len()
    );
    if let Some(o) = opts {
        assert_eq!(
            o.len(),
            l,
            "opts must have the same length as seg ({} vs {l})",
            o.len()
        );
    }

    for i in 0..l {
        for j in 0..l {
            let allow = if i == j {
                // Diagonal always — every row keeps a finite softmax.
                true
            } else {
                let (gi, gj) = (seg[i], seg[j]);
                let real = gi >= 0 && gj >= 0;
                let causal = j <= i;
                let key_is_state = gj == 0;
                let same_seg = gj == gi;
                // Causal base, then the state_bidir disjunct OR-ed in
                // independently of causality (state↔state is bidirectional),
                // then the option-isolation conjunct — the reference order
                // (`branch_mask_batch`: base → state_bidir OR → opts AND).
                let base = real && causal && (key_is_state || same_seg);
                let bidir = state_bidir && real && gi == 0 && gj == 0;
                let mut allow = base || bidir;
                if let Some(o) = opts
                    && allow
                {
                    let (oi, oj) = (o[i], o[j]);
                    let key_is_option = oj >= 0;
                    let query_is_decide = oi == OPT_DECIDE;
                    let same_option = oj == oi;
                    allow = !key_is_option || query_is_decide || same_option;
                }
                allow
            };
            out[i * l + j] = allow;
        }
    }
}

/// Build the row-form mask for one `state + branch` causal row
/// (`rows_of` output): causal triangle + the full state block bidirectional
/// when `state_bidir` (state keys are fully visible either way for branch
/// queries — they precede the branch; the flag only widens state↔state).
///
/// No pads: rows are exact slices, so the mask is exact too.
pub fn row_branch_mask(state_len: usize, branch_len: usize, state_bidir: bool) -> Vec<bool> {
    let l = state_len + branch_len;
    let mut out = vec![false; l * l];
    for i in 0..l {
        for j in 0..l {
            let causal = j <= i;
            let state_pair = state_bidir && i < state_len && j < state_len;
            out[i * l + j] = causal || state_pair;
        }
    }
    out
}

// ── T3 — the row form ───────────────────────────────────────────────

/// A packed eDLM encoding: token ids, per-token position ids (branch positions
/// RESTART at `state_len` — RoPE phases are branch-local, duplicated across
/// questions by design), segment ids, and the marker readout indices.
///
/// Constructed by hand in Phase 1 (the marker encode/pack is T4); the mask,
/// row-form, and parity machinery below are already its consumers.
#[derive(Debug, Clone, Default)]
pub struct PackedEncoding {
    pub ids: Vec<usize>,
    pub pos: Vec<usize>,
    pub seg: Vec<i32>,
    /// Token count of the state segment (`seg[..state_len]` all `0`).
    pub state_len: usize,
    /// Absolute index of each question's decide marker (its branch's last token).
    pub decide_idx: Vec<usize>,
    /// Absolute index of each question's `</opt>` key positions.
    pub opt_idx: Vec<Vec<usize>>,
}

impl PackedEncoding {
    /// Assert the structural invariants the mask/row machinery relies on.
    pub fn validate(&self) -> Result<()> {
        let l = self.ids.len();
        if self.pos.len() != l || self.seg.len() != l {
            bail!("ids/pos/seg length mismatch: {l}/{}/{}", self.pos.len(), self.seg.len());
        }
        if self.state_len == 0 || self.state_len > l {
            bail!("state_len {} out of range for l {l}", self.state_len);
        }
        if self.seg[..self.state_len].iter().any(|&g| g != 0) {
            bail!("seg[..state_len] is not all zeros");
        }
        Ok(())
    }
}

/// One question's branch as a self-contained row: `state ++ branch`.
///
/// `decide`/`opts` are offsets WITHIN the branch; add `state_len` for
/// row-relative readout positions. `start..end` index the packed encoding.
#[derive(Debug, Clone)]
pub struct BranchRow {
    pub start: usize,
    pub end: usize,
    pub decide: usize,
    pub opts: Vec<usize>,
}

impl BranchRow {
    /// Branch token count (excluding the state prefix).
    pub fn branch_len(&self) -> usize {
        self.end - self.start
    }
}

/// Split a packed encoding into per-question rows.
///
/// Row `k` = `state ++ branch_k`: the branch slice runs from the question's
/// first segment token to its decide marker (inclusive). Positions carry over
/// unchanged — the branch already restarts at `state_len` in the packed form,
/// so packed and row forms agree position-for-position and the KV set of a
/// branch query is identical in both (parity by construction; pinned by
/// `packed_rows_parity`).
pub fn rows_of(enc: &PackedEncoding) -> Result<Vec<BranchRow>> {
    enc.validate()?;
    let mut rows = Vec::with_capacity(enc.decide_idx.len());
    for (k, &decide_abs) in enc.decide_idx.iter().enumerate() {
        let start = if k == 0 {
            enc.state_len
        } else {
            enc.decide_idx[k - 1] + 1
        };
        let end = decide_abs + 1;
        if end > enc.seg.len() || start >= end {
            bail!("branch {k}: empty or out-of-range slice [{start}, {end})");
        }
        if enc.seg[start] != (k as i32) + 1 || enc.seg[end - 1] != (k as i32) + 1 {
            bail!(
                "branch {k}: layout mismatch — seg[start]={} seg[end-1]={}, expected {}",
                enc.seg[start],
                enc.seg[end - 1],
                (k as i32) + 1
            );
        }
        let opts = enc
            .opt_idx
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("branch {k}: missing opt_idx row"))?
            .iter()
            .map(|&o| {
                if o < start || o >= end {
                    bail!("branch {k}: opt index {o} outside [{start}, {end})");
                }
                Ok(o - start)
            })
            .collect::<Result<Vec<usize>>>()?;
        rows.push(BranchRow {
            start,
            end,
            decide: decide_abs - start,
            opts,
        });
    }
    Ok(rows)
}

// ── T1 — weights ────────────────────────────────────────────────────

/// The eDLM pointer head (their `PointerHead`, 256-dim default).
///
/// `z_j = (W_k·h_j + b_k) · (W_q·h_decide + b_q) / sqrt(dp)`, probabilities =
/// softmax per question over the option keys; `temperature` divides the
/// logits BEFORE the softmax (eval only; training always saw T=1 — argmax
/// unchanged by construction). The GGUF carries `pointer.q.weight` F16,
/// `pointer.{q,k}.bias` F32, `pointer.temperature.weight` F32 {1}.
#[derive(Debug, Clone, Default)]
pub struct EdlmPointerHead {
    /// `[dp, d]` row-major (our matmul convention).
    pub q: Vec<f32>,
    pub q_bias: Vec<f32>,
    /// `[dp, d]` row-major.
    pub k: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub dp: usize,
    /// Input hidden dim `d` (4096 on the 8B release).
    pub d: usize,
    pub temperature: f32,
}

impl EdlmPointerHead {
    /// Per-question option probabilities: `h_decide` `[d]`,
    /// `h_opts` `[n_opts * d]` concatenated in option order.
    ///
    /// `z_j = (W_k·h_j + b_k) · (W_q·h_decide + b_q) / sqrt(dp)`, then
    /// softmax over the option keys (plain `exp`, the reference numerics —
    /// not the decode exp-table path).
    pub fn question_probs(&self, h_decide: &[f32], h_opts: &[f32], n_opts: usize) -> Vec<f32> {
        assert_eq!(h_decide.len(), self.d, "h_decide must be [d]");
        assert_eq!(h_opts.len(), n_opts * self.d, "h_opts must be [n_opts * d]");
        assert!(n_opts > 0);
        let mut qv = vec![0.0f32; self.dp];
        let mut kv = vec![0.0f32; self.dp];
        crate::types::matmul(&mut qv, &self.q, h_decide, self.dp, self.d);
        for (qv_i, qb) in qv.iter_mut().zip(self.q_bias.iter()) {
            *qv_i += qb;
        }
        let scale = 1.0 / (self.dp as f32).sqrt();
        let t = if self.temperature > 0.0 {
            self.temperature
        } else {
            1.0
        };
        let mut z = vec![0.0f32; n_opts];
        let mut max_z = f32::NEG_INFINITY;
        for j in 0..n_opts {
            crate::types::matmul(
                &mut kv,
                &self.k,
                &h_opts[j * self.d..(j + 1) * self.d],
                self.dp,
                self.d,
            );
            let mut dot = 0.0f32;
            for i in 0..self.dp {
                dot += (kv[i] + self.k_bias[i]) * qv[i];
            }
            z[j] = dot * scale / t;
            max_z = max_z.max(z[j]);
        }
        // Softmax (stable).
        let mut sum = 0.0f32;
        for v in z.iter_mut() {
            *v = (*v - max_z).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in z.iter_mut() {
            *v *= inv;
        }
        z
    }
}

/// One eDLM transformer layer: the llama-family projections plus the Qwen3
/// QK-RMSNorm gammas (`attn_q_norm` / `attn_k_norm`, `[head_dim]`, applied
/// per head AFTER projection, BEFORE RoPE).
pub struct EdlmLayerWeights {
    pub base: LlamaLayerWeights,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// Full eDLM weight set (dequantized f32). No KV cache — the lane scores one
/// full sequence per forward. `lm_head` is optional: the scoring readout is
/// the pointer head over hidden states, never text logits (the GGUF carries
/// `output.weight`, but a tied-emissions conversion may omit it).
pub struct EdlmWeights {
    pub wte: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f32>>,
    pub layers: Vec<EdlmLayerWeights>,
    pub pointer: Option<EdlmPointerHead>,
}

/// `Config` from `edlm.*` GGUF metadata (prefix mirrors
/// `qwen2_config_from_gguf_metadata`; header-verified defaults).
fn edlm_config_from_gguf_metadata(gguf: &crate::gguf_loader::GgufFile) -> Result<crate::types::Config> {
    let prefix = "edlm.";
    let context_length = gguf
        .metadata_u64(&format!("{prefix}context_length"))
        .unwrap_or(32768) as usize;
    let n_embd = gguf
        .metadata_u64(&format!("{prefix}embedding_length"))
        .unwrap_or(4096) as usize;
    let n_layer = gguf
        .metadata_u64(&format!("{prefix}block_count"))
        .unwrap_or(36) as usize;
    let mlp_hidden = gguf
        .metadata_u64(&format!("{prefix}feed_forward_length"))
        .unwrap_or(12288) as usize;
    let n_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count"))
        .unwrap_or(32) as usize;
    let n_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count_kv"))
        .unwrap_or(8) as usize;
    let rms_norm_eps = gguf
        .metadata_f64(&format!("{prefix}attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length"))
        .unwrap_or((n_embd / n_head) as u64) as usize;
    let rope_theta = gguf
        .metadata_f64(&format!("{prefix}rope.freq_base"))
        .unwrap_or(1_000_000.0) as f32;

    let vocab_size = gguf
        .tensor_info("token_embd.weight")
        .and_then(|info| info.shape.last().copied())
        .unwrap_or(151_936);

    let mut config = crate::types::Config::micro();
    config.vocab_size = vocab_size;
    config.block_size = context_length;
    config.n_embd = n_embd;
    config.n_head = n_head;
    config.head_dim = head_dim;
    config.mlp_hidden = mlp_hidden;
    config.n_layer = n_layer;
    config.n_kv_head = n_kv_head;
    config.rms_norm_eps = rms_norm_eps;
    config.rope_theta = rope_theta;
    // The lane's forward is `forward_edlm_packed`, invoked explicitly — the
    // enum is a dispatch hint for OTHER code paths only. `Llama` is the
    // closest surface (RoPE-family, SwiGLU, no embedding scaling), the same
    // call the qwen2 loader makes. A dedicated variant would be an
    // upstream (katgpt-types) change — not this unit's blast radius.
    config.model_arch = crate::types::ModelArchitecture::Llama;
    config.use_rope = true;
    config.tied_embeddings = false;
    config.rms_norm_offset = false;
    config.post_norm = false;
    config.attn_logit_softcapping = 0.0;
    config.final_logit_softcapping = 0.0;
    Ok(config)
}

/// Load eDLM weights from a GGUF file (arch `edlm`).
///
/// Tensor map (header-verified against `drex-dlm-Q8_0.gguf`):
/// `token_embd.weight`, `output_norm.weight`, `output.weight` (optional),
/// per layer `blk.{N}.attn_norm.weight`, `blk.{N}.attn_{q,k,v}.weight`,
/// `blk.{N}.attn_output.weight`, `blk.{N}.attn_{q,k}_norm.weight`,
/// `blk.{N}.ffn_{norm,gate,up,down}.weight`; pointer
/// `pointer.{q,k}.weight` (F16) + `pointer.{q,k}.bias` (F32) +
/// `pointer.temperature.weight` (F32, NE 1; absent ⇒ 1.0).
///
/// RoPE is NEOX (rotate-half) — the fork maps `LLM_ARCH_EDLM` to
/// `LLAMA_ROPE_TYPE_NEOX` — so Q/K rows are stored in HF order and are NOT
/// unpermuted (the `qwen2` loader posture, not the `llama` one).
///
/// Supports every quant `GgufFile::dequant_f16_to_f32` handles (the release
/// GGUF is Q8_0 with F16 pointer weights / F32 norms).
pub fn load_edlm_weights_gguf(path: &std::path::Path) -> Result<(crate::types::Config, EdlmWeights)> {
    let gguf = crate::gguf_loader::GgufFile::open(path)?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "edlm" {
        bail!("expected edlm architecture, got '{arch}'");
    }
    let config = edlm_config_from_gguf_metadata(&gguf)?;
    let n_layer = config.n_layer;

    let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;
    let lm_head = gguf.try_dequant_f16_to_f32("output.weight")?;

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let base = LlamaLayerWeights {
            attn_wq: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_q.weight"))?,
            attn_wk: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_k.weight"))?,
            attn_wv: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_v.weight"))?,
            attn_wo: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_output.weight"))?,
            gate_proj: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_gate.weight"))?,
            up_proj: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_up.weight"))?,
            down_proj: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_down.weight"))?,
            input_norm: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_norm.weight"))?,
            post_attn_norm: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_norm.weight"))?,
        };
        let q_norm = gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_q_norm.weight"))?;
        let k_norm = gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_k_norm.weight"))?;
        if q_norm.len() != config.head_dim || k_norm.len() != config.head_dim {
            bail!(
                "layer {i}: QK-norm gamma length {}/{} != head_dim {}",
                q_norm.len(),
                k_norm.len(),
                config.head_dim
            );
        }
        layers.push(EdlmLayerWeights {
            base,
            q_norm,
            k_norm,
        });
    }

    let pointer = if gguf.tensor_info("pointer.q.weight").is_some() {
        let q = gguf.dequant_f16_to_f32("pointer.q.weight")?;
        let k = gguf.dequant_f16_to_f32("pointer.k.weight")?;
        let q_bias = gguf.try_dequant_f16_to_f32("pointer.q.bias")?.unwrap_or_default();
        let k_bias = gguf.try_dequant_f16_to_f32("pointer.k.bias")?.unwrap_or_default();
        let temperature = gguf
            .try_dequant_f16_to_f32("pointer.temperature.weight")?
            .and_then(|v| v.first().copied())
            .unwrap_or(1.0);
        if q.len() != k.len() || q.len() % config.n_embd != 0 {
            bail!(
                "pointer weight shape mismatch: q {} k {} (n_embd {})",
                q.len(),
                k.len(),
                config.n_embd
            );
        }
        let dp = q.len() / config.n_embd;
        Some(EdlmPointerHead {
            q,
            q_bias,
            k,
            k_bias,
            dp,
            d: config.n_embd,
            temperature,
        })
    } else {
        None
    };

    Ok((
        config,
        EdlmWeights {
            wte,
            final_norm,
            lm_head,
            layers,
            pointer,
        },
    ))
}

// ── the forward ─────────────────────────────────────────────────────

/// Full-sequence eDLM scoring forward (Qwen3 blocks, NO KV cache).
///
/// `ids`/`pos`/`allow` describe the sequence (`allow` row-major
/// `[l * l]` — `branch_mask` or `row_branch_mask` output). Returns the
/// post-final-norm hidden states `[l * n_embd]`; the pointer head reads
/// marker positions from this buffer (no lm_head — the lane never generates
/// text).
pub fn forward_edlm_packed(
    weights: &EdlmWeights,
    config: &crate::types::Config,
    ids: &[usize],
    pos: &[usize],
    allow: &[bool],
) -> Vec<f32> {
    let l = ids.len();
    assert_eq!(pos.len(), l, "pos must match ids");
    assert_eq!(allow.len(), l * l, "allow must be l*l");
    assert!(l > 0 && l <= config.block_size, "l {l} out of range");
    assert_eq!(weights.layers.len(), config.n_layer, "layer count mismatch");
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let n_kv = config.n_kv_head;
    assert_eq!(n % config.n_head, 0, "n_embd must divide by n_head");

    let freq = RopeFreqTable::new(config.rope_theta, hd);
    let scale = 1.0 / (hd as f32).sqrt();

    let mut x = vec![0.0f32; n];
    let mut xr = vec![0.0f32; n];
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kvd];
    let mut v = vec![0.0f32; kvd];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut scores = vec![0.0f32; l];
    let mut hidden = vec![0.0f32; l * n];
    let mut gate = vec![0.0f32; config.mlp_hidden];
    let mut up = vec![0.0f32; config.mlp_hidden];
    let mut mlp_out = vec![0.0f32; config.mlp_hidden];
    let mut key_cache = vec![0.0f32; l * kvd];
    let mut value_cache = vec![0.0f32; l * kvd];

    // Embedding (no scaling, no wpe — RoPE carries position).
    for (p, &id) in ids.iter().enumerate() {
        let dst = &mut hidden[p * n..(p + 1) * n];
        let src = &weights.wte[id * n..(id + 1) * n];
        dst.copy_from_slice(src);
    }

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        // Phase A: K/V for every position (mask-independent).
        for (p, &_id) in ids.iter().enumerate() {
            x.copy_from_slice(&hidden[p * n..(p + 1) * n]);
            crate::types::rmsnorm_with_gamma_eps(&mut x, &layer.base.input_norm, config.rms_norm_eps);
            crate::types::matmul(&mut k, &layer.base.attn_wk, &x, kvd, n);
            crate::types::matmul(&mut v, &layer.base.attn_wv, &x, kvd, n);
            // K: per-KV-head QK-norm, then RoPE at this position (rotate the
            // K buffer only — an empty second slice applies zero K heads).
            qk_norm_inplace(&mut k, &layer.k_norm, n_kv, hd, config.rms_norm_eps);
            crate::rope::apply_rope_with_freq(&mut k, &mut [], pos[p], hd, freq.as_slice());
            key_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&k);
            value_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&v);
        }

        // Phase B: Q per position + masked GQA attention.
        for (p, &_id) in ids.iter().enumerate() {
            x.copy_from_slice(&hidden[p * n..(p + 1) * n]);
            // Residual saved PRE-norm — the attention block adds back the
            // un-normed stream (the forward_llama residual law).
            xr.copy_from_slice(&x);
            crate::types::rmsnorm_with_gamma_eps(&mut x, &layer.base.input_norm, config.rms_norm_eps);
            crate::types::matmul(&mut q, &layer.base.attn_wq, &x, q_dim, n);
            // Q: per-head QK-norm, then RoPE (q-only — empty second slice).
            qk_norm_inplace(&mut q, &layer.q_norm, config.n_head, hd, config.rms_norm_eps);
            crate::rope::apply_rope_with_freq(&mut q, &mut [], pos[p], hd, freq.as_slice());

            attn_out[..n].fill(0.0);
            for h in 0..config.n_head {
                let kv_group = h * n_kv / config.n_head;
                unsafe {
                    super::attention::attention_head_masked(
                        &q,
                        &key_cache,
                        &value_cache,
                        &mut attn_out,
                        &mut scores,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        l,
                        scale,
                        allow,
                        p,
                    );
                }
            }

            crate::types::matmul(&mut x, &layer.base.attn_wo, &attn_out, n, q_dim);
            for i in 0..n {
                x[i] += xr[i];
            }

            // MLP: SwiGLU (out = silu(gate) ⊙ up, then down-proj).
            xr.copy_from_slice(&x);
            crate::types::rmsnorm_with_gamma_eps(&mut x, &layer.base.post_attn_norm, config.rms_norm_eps);
            crate::types::matmul(&mut gate, &layer.base.gate_proj, &x, config.mlp_hidden, n);
            crate::types::matmul(&mut up, &layer.base.up_proj, &x, config.mlp_hidden, n);
            crate::types::swiglu(&mut mlp_out, &gate, &up);
            crate::types::matmul(&mut x, &layer.base.down_proj, &mlp_out, n, config.mlp_hidden);
            for i in 0..n {
                x[i] += xr[i];
            }

            hidden[p * n..(p + 1) * n].copy_from_slice(&x);
        }
        let _ = layer_idx;
    }

    // Final norm on every position's hidden state (the pointer head reads
    // marker positions from the normed buffer; no lm_head — the lane never
    // generates text).
    for p in 0..l {
        crate::types::rmsnorm_with_gamma_eps(
            &mut hidden[p * n..(p + 1) * n],
            &weights.final_norm,
            config.rms_norm_eps,
        );
    }
    hidden
}

/// Per-head RMSNorm in place over a concatenated multi-head buffer.
fn qk_norm_inplace(buf: &mut [f32], gamma: &[f32], n_heads: usize, head_dim: usize, eps: f64) {
    debug_assert_eq!(buf.len(), n_heads * head_dim);
    debug_assert_eq!(gamma.len(), head_dim);
    for h in 0..n_heads {
        let off = h * head_dim;
        let head = &mut buf[off..off + head_dim];
        let mut sum_sq = 0.0f32;
        for &v in head.iter() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps as f32).sqrt();
        for (v, &g) in head.iter_mut().zip(gamma.iter()) {
            *v *= inv_rms * g;
        }
    }
}

/// Row-form forward: every question runs as a self-contained
/// `state ++ branch` causal row (mask from [`row_branch_mask`]).
///
/// `state_bidir` must MATCH the packed posture being mirrored — parity holds
/// in both postures, never across them (with the flag off the packed state is
/// a causal prefix of itself, and the row mask must say the same).
///
/// Returns per-question hidden states `[row_len * n_embd]` — slice branch
/// tokens at `[(state_len + off) * n_embd ..]` with the row's offsets.
pub fn forward_edlm_rows(
    weights: &EdlmWeights,
    config: &crate::types::Config,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    state_bidir: bool,
) -> Vec<Vec<f32>> {
    let sl = enc.state_len;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let bl = row.branch_len();
        let mut ids = Vec::with_capacity(sl + bl);
        let mut pos = Vec::with_capacity(sl + bl);
        ids.extend_from_slice(&enc.ids[..sl]);
        pos.extend_from_slice(&enc.pos[..sl]);
        ids.extend_from_slice(&enc.ids[row.start..row.end]);
        pos.extend_from_slice(&enc.pos[row.start..row.end]);
        let allow = row_branch_mask(sl, bl, state_bidir);
        out.push(forward_edlm_packed(weights, config, &ids, &pos, &allow));
    }
    out
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const S: i32 = 0;

    fn seg_state_branches(state: usize, branches: &[usize]) -> Vec<i32> {
        let mut seg = vec![S; state];
        for (k, &bl) in branches.iter().enumerate() {
            seg.extend(std::iter::repeat(k as i32 + 1).take(bl));
        }
        seg
    }

    #[test]
    fn branch_mask_four_quadrant_law() {
        // state(2) | q1(3) | q2(3)
        let seg = seg_state_branches(2, &[3, 3]);
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, true, None, &mut m);

        let at = |i: usize, j: usize| m[i * l + j];

        // state↔state: bidirectional (state_bidir = true).
        assert!(at(0, 1) && at(1, 0), "state pair must attend both ways");
        // state ↛ any branch (either direction).
        for i in 0..2 {
            for j in 2..l {
                assert!(!at(i, j), "state query {i} must not see branch key {j}");
            }
        }
        for i in 2..l {
            for j in 0..2 {
                assert!(at(i, j), "branch query {i} must see state key {j}");
            }
        }
        // q1 (2..5) sees only itself; q2 (5..8) sees state + own prefix.
        assert!(!at(3, 4) && at(4, 3), "q1 causal within own branch");
        assert!(!at(6, 2) && !at(6, 4), "q2 must not see q1 tokens");
        assert!(at(6, 5) && at(7, 5), "q2 causal within own branch");
        // Diagonal always.
        for i in 0..l {
            assert!(at(i, i), "diagonal must survive");
        }
    }

    #[test]
    fn branch_mask_state_bidir_off_prefixes_the_state() {
        let seg = seg_state_branches(2, &[2]);
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, false, None, &mut m);
        // Without state_bidir the state is a causal prefix of itself.
        assert!(!m[0 * l + 1], "state query 0 must not see future state key 1");
        assert!(m[1 * l + 0], "state query 1 sees earlier state key 0");
        // Branch queries still see the whole state (it precedes them).
        assert!(m[2 * l + 0] && m[2 * l + 1]);
    }

    #[test]
    fn branch_mask_pads_are_invisible_keys_and_dead_queries() {
        let seg = vec![S, 1, -1, -1];
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, true, None, &mut m);
        let at = |i: usize, j: usize| m[i * l + j];
        // Pads as keys: visible to nobody (a pad query's own diagonal excepted).
        for i in 0..l {
            for j in [2usize, 3] {
                if i != j {
                    assert!(!at(i, j), "pad key {j} must be blocked for query {i}");
                }
            }
        }
        // Pads as queries: only their own diagonal survives (never read).
        for j in 0..l {
            assert_eq!(at(2, j), j == 2, "pad query 2 only diagonal");
            assert_eq!(at(3, j), j == 3, "pad query 3 only diagonal");
        }
    }

    #[test]
    fn branch_mask_option_isolation() {
        // state | instruction | opt0(2) | opt1(1) | decide — one question.
        let seg = vec![S, 1, 1, 1, 1, 1];
        let opts = vec![OPT_NONE, OPT_NONE, 0, 0, 1, OPT_DECIDE];
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, true, Some(&opts), &mut m);
        let at = |i: usize, j: usize| m[i * l + j];
        // opt0 tokens see state + instruction + their own EARLIER span token
        // (spans stay causal within themselves — the isolation conjunct only
        // narrows, never widens the causal base).
        assert!(at(2, 0) && at(2, 1) && !at(2, 3) && at(3, 2) && !at(2, 4));
        // opt1 sees state + instruction + itself, NOT opt0.
        assert!(at(4, 0) && at(4, 1) && !at(4, 2) && !at(4, 3));
        // decide sees everything in its question.
        for j in 0..l {
            assert!(at(5, j), "decide must see key {j}");
        }
        // instruction is not an option key: visible to everyone (causal permitting).
        assert!(at(4, 1));
        // diagonal survives isolation.
        for i in 0..l {
            assert!(at(i, i));
        }
    }

    #[test]
    fn branch_mask_rejects_bad_buffers() {
        let seg = vec![S, 1];
        let mut m = vec![false; 4];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            branch_mask(&seg, true, None, &mut m[..3])
        }));
        assert!(r.is_err(), "l*l buffer required");
        let opts = vec![OPT_NONE];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            branch_mask(&seg, true, Some(&opts), &mut m)
        }));
        assert!(r.is_err(), "opts must match seg length");
    }

    fn sample_encoding() -> PackedEncoding {
        // state(5) | q1: [instr(1), opt(2), decide] | q2: [instr(1), opt(3), opt(1), decide]
        let mut ids = Vec::new();
        let mut pos = Vec::new();
        let mut seg = Vec::new();
        // state
        for i in 0..5 {
            ids.push(i);
            pos.push(i);
            seg.push(S);
        }
        let mut decide_idx = Vec::new();
        let mut opt_idx = Vec::new();
        // q1 branch: len 4
        for (bi, _) in (0..3).enumerate() {
            ids.push(10 + bi);
            pos.push(5 + bi);
            seg.push(1);
        }
        ids.push(63); // the decide marker (any in-vocab id — markers are T4)
        pos.push(5 + 3);
        seg.push(1);
        decide_idx.push(ids.len() - 1);
        opt_idx.push(vec![ids.len() - 2]);
        // q2 branch: len 6
        for bi in 0..5 {
            ids.push(20 + bi);
            pos.push(5 + bi);
            seg.push(2);
        }
        ids.push(62); // q2's decide marker
        pos.push(5 + 5);
        seg.push(2);
        decide_idx.push(ids.len() - 1);
        opt_idx.push(vec![ids.len() - 4, ids.len() - 2]);
        PackedEncoding {
            ids,
            pos,
            seg,
            state_len: 5,
            decide_idx,
            opt_idx,
        }
    }

    #[test]
    fn rows_of_slices_and_offsets() {
        let enc = sample_encoding();
        let rows = rows_of(&enc).expect("rows");
        assert_eq!(rows.len(), 2);
        // q1: branch [5, 9), decide offset 3, opt offset 2.
        assert_eq!(rows[0].start, 5);
        assert_eq!(rows[0].end, 9);
        assert_eq!(rows[0].decide, 3);
        assert_eq!(rows[0].opts, vec![2]);
        // q2: branch [9, 15), decide offset 5, opts 2 and 4.
        assert_eq!(rows[1].start, 9);
        assert_eq!(rows[1].end, 15);
        assert_eq!(rows[1].decide, 5);
        assert_eq!(rows[1].opts, vec![2, 4]);
    }

    #[test]
    fn rows_of_refuses_layout_mismatch() {
        let mut enc = sample_encoding();
        enc.seg[9] = 1; // q2's first token claims q1's segment
        let r = rows_of(&enc);
        assert!(r.is_err(), "layout mismatch must refuse");
        let mut enc = sample_encoding();
        enc.opt_idx[1][0] = 3; // opt index outside q2's slice
        let r = rows_of(&enc);
        assert!(r.is_err(), "opt outside slice must refuse");
    }

    // Deterministic tiny random weights (LCG) for the forward parity test.
    fn tiny_config() -> crate::types::Config {
        let mut c = crate::types::Config::micro();
        c.vocab_size = 64;
        c.block_size = 256;
        // Qwen3 shape law: n_embd == n_head * head_dim (no q_dim != n_embd).
        c.n_embd = 32;
        c.n_head = 4;
        c.n_kv_head = 2;
        c.head_dim = 8;
        c.mlp_hidden = 64;
        c.n_layer = 2;
        c.rms_norm_eps = 1e-6;
        c.rope_theta = 1_000_000.0;
        c.use_rope = true;
        c.tied_embeddings = false;
        c.rms_norm_offset = false;
        c.post_norm = false;
        c.attn_logit_softcapping = 0.0;
        c.final_logit_softcapping = 0.0;
        c
    }

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn mat(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
            (0..rows * cols).map(|_| self.next_f32() * scale).collect()
        }
        fn gamma(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| 1.0 + 0.1 * self.next_f32()).collect()
        }
    }

    fn tiny_weights(config: &crate::types::Config) -> EdlmWeights {
        let mut r = Lcg(0x5EED_1005);
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(EdlmLayerWeights {
                base: LlamaLayerWeights {
                    attn_wq: r.mat(config.n_head * config.head_dim, config.n_embd, 0.2),
                    attn_wk: r.mat(config.n_kv_head * config.head_dim, config.n_embd, 0.2),
                    attn_wv: r.mat(config.n_kv_head * config.head_dim, config.n_embd, 0.2),
                    attn_wo: r.mat(config.n_embd, config.n_head * config.head_dim, 0.2),
                    gate_proj: r.mat(config.mlp_hidden, config.n_embd, 0.2),
                    up_proj: r.mat(config.mlp_hidden, config.n_embd, 0.2),
                    down_proj: r.mat(config.n_embd, config.mlp_hidden, 0.2),
                    input_norm: r.gamma(config.n_embd),
                    post_attn_norm: r.gamma(config.n_embd),
                },
                q_norm: r.gamma(config.head_dim),
                k_norm: r.gamma(config.head_dim),
            });
        }
        EdlmWeights {
            wte: r.mat(config.vocab_size, config.n_embd, 0.5),
            final_norm: r.gamma(config.n_embd),
            lm_head: None,
            layers,
            pointer: None,
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn parity_case(state_bidir: bool) -> f32 {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let l = enc.ids.len();
        let mut allow = vec![false; l * l];
        branch_mask(&enc.seg, state_bidir, None, &mut allow);
        let packed = forward_edlm_packed(&weights, &config, &enc.ids, &enc.pos, &allow);

        let rows = rows_of(&enc).expect("rows");
        let row_hiddens = forward_edlm_rows(&weights, &config, &enc, &rows, state_bidir);

        let n = config.n_embd;
        let sl = enc.state_len;
        let mut worst = 0.0f32;
        // State positions: identical (state never sees branches in either form).
        for p in 0..sl {
            let row0 = &row_hiddens[0];
            let d = max_abs_diff(
                &packed[p * n..(p + 1) * n],
                &row0[p * n..(p + 1) * n],
            );
            worst = worst.max(d);
        }
        // Branch positions: packed branch token (abs p) == row token (sl + off).
        for (k, row) in rows.iter().enumerate() {
            let rh = &row_hiddens[k];
            for off in 0..row.branch_len() {
                let d = max_abs_diff(
                    &packed[(row.start + off) * n..(row.start + off + 1) * n],
                    &rh[(sl + off) * n..(sl + off + 1) * n],
                );
                worst = worst.max(d);
            }
        }
        worst
    }

    #[test]
    fn packed_rows_parity_bidir() {
        let worst = parity_case(true);
        assert!(
            worst < 1e-5,
            "packed-vs-row parity exceeded tolerance: {worst}"
        );
    }

    #[test]
    fn packed_rows_parity_no_bidir() {
        let worst = parity_case(false);
        assert!(
            worst < 1e-5,
            "packed-vs-row parity (no state_bidir) exceeded tolerance: {worst}"
        );
    }

    /// Diagnostic: single-question packed-vs-row must be exact — pins whether
    /// a divergence is within-branch (forward bug) or cross-question (mask bug).
    #[test]
    fn parity_single_question_diagnostic() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        // state(3) | q1(3): ids 0..3 state, 3..6 branch; decide = 5.
        let enc = PackedEncoding {
            ids: vec![0, 1, 2, 10, 11, 63],
            pos: vec![0, 1, 2, 3, 4, 5],
            seg: vec![0, 0, 0, 1, 1, 1],
            state_len: 3,
            decide_idx: vec![5],
            opt_idx: vec![vec![4]],
        };
        let n = config.n_embd;
        let l = enc.ids.len();
        let mut allow = vec![false; l * l];
        branch_mask(&enc.seg, true, None, &mut allow);
        let packed = forward_edlm_packed(&weights, &config, &enc.ids, &enc.pos, &allow);
        let rows = rows_of(&enc).expect("rows");
        let row_hiddens = forward_edlm_rows(&weights, &config, &enc, &rows, true);
        let rh = &row_hiddens[0];
        let sl = enc.state_len;
        for p in 0..l {
            let d = max_abs_diff(
                &packed[p * n..(p + 1) * n],
                &rh[p * n..(p + 1) * n],
            );
            println!("pos {p}: diff {d}");
        }
        let _ = sl;
        let worst = max_abs_diff(&packed, rh);
        assert!(worst < 1e-5, "single-question parity: {worst}");
    }

    /// Diagnostic: two questions — prints per-position packed-vs-row diffs to
    /// pin which branch diverges (row k shares the state prefix with row 0,
    /// so any nonzero is a mask/cross-question defect, not fp noise).
    #[test]
    fn parity_two_question_diagnostic() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let n = config.n_embd;
        let l = enc.ids.len();
        let mut allow = vec![false; l * l];
        branch_mask(&enc.seg, true, None, &mut allow);
        let packed = forward_edlm_packed(&weights, &config, &enc.ids, &enc.pos, &allow);
        let rows = rows_of(&enc).expect("rows");
        let row_hiddens = forward_edlm_rows(&weights, &config, &enc, &rows, true);
        let sl = enc.state_len;
        for (k, row) in rows.iter().enumerate() {
            let rh = &row_hiddens[k];
            // state prefix of every row vs packed state
            for p in 0..sl {
                let d = max_abs_diff(&packed[p * n..(p + 1) * n], &rh[p * n..(p + 1) * n]);
                println!("row{k} state pos {p}: {d}");
            }
            for off in 0..row.branch_len() {
                let d = max_abs_diff(
                    &packed[(row.start + off) * n..(row.start + off + 1) * n],
                    &rh[(sl + off) * n..(sl + off + 1) * n],
                );
                println!("row{k} branch off {off}: {d}");
            }
        }
        // Eligible-key audit: state query 0 and q2's first branch token (abs 9).
        let elig = |q: usize| -> Vec<usize> {
            (0..l).filter(|&j| allow[q * l + j]).collect()
        };
        println!("packed state q0 eligible: {:?}", elig(0));
        println!("packed q2off0 (abs 9) eligible: {:?}", elig(9));
        let rall = row_branch_mask(sl, rows[1].branch_len(), true);
        let rl = sl + rows[1].branch_len();
        let elig_r = |q: usize| -> Vec<usize> { (0..rl).filter(|&j| rall[q * rl + j]).collect() };
        println!("row q2off0 eligible: {:?}", elig_r(sl));
        // Contamination audit: row0-state vs row1-state must be BIT-IDENTICAL
        // (state never sees either branch); and packed-state vs a standalone
        // state-only forward (l = sl, full bidir) must be too.
        let d_rows = max_abs_diff(
            &row_hiddens[0][..sl * n],
            &row_hiddens[1][..sl * n],
        );
        println!("row0-state vs row1-state: {d_rows}");
        let mut s_allow = vec![false; sl * sl];
        branch_mask(&enc.seg[..sl], true, None, &mut s_allow);
        let standalone = forward_edlm_packed(&weights, &config, &enc.ids[..sl], &enc.pos[..sl], &s_allow);
        let d_packed = max_abs_diff(&packed[..sl * n], &standalone);
        println!("packed-state vs standalone-state: {d_packed}");
        let d_row0 = max_abs_diff(&row_hiddens[0][..sl * n], &standalone);
        println!("row0-state vs standalone-state: {d_row0}");
    }

    #[test]
    fn qk_norm_matches_reference_formula() {
        let mut buf = vec![0.5, -1.25, 2.0, 0.125, 3.0, -0.5, 1.5, -2.5];
        let gamma = vec![1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.05, 0.95];
        let expected: Vec<f32> = {
            let mut e = buf.clone();
            let sum_sq: f32 = e.iter().map(|v| v * v).sum();
            let inv = 1.0 / (sum_sq / 8.0 + 1e-6).sqrt();
            for (v, &g) in e.iter_mut().zip(gamma.iter()) {
                *v *= inv * g;
            }
            e
        };
        qk_norm_inplace(&mut buf, &gamma, 1, 8, 1e-6);
        assert!(max_abs_diff(&buf, &expected) < 1e-7);
    }

    #[test]
    fn loader_env_gated_header_check() {
        let path = match std::env::var("EDLM_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!("SKIP: EDLM_GGUF unset — set it to drex-dlm-Q8_0.gguf for the header check");
                return;
            }
        };
        let gguf = crate::gguf_loader::GgufFile::open(std::path::Path::new(&path))
            .expect("open EDLM_GGUF");
        assert_eq!(gguf.architecture(), Some("edlm"));
        let config = edlm_config_from_gguf_metadata(&gguf).expect("config");
        assert_eq!(config.n_layer, 36);
        assert_eq!(config.n_embd, 4096);
        assert_eq!(config.n_head, 32);
        assert_eq!(config.n_kv_head, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.mlp_hidden, 12288);
        assert_eq!(config.vocab_size, 151_936);
        assert_eq!(config.block_size, 32768);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert_eq!(config.rope_theta, 1_000_000.0);
        // Pointer tensors present; temperature is F32 {1}.
        assert!(gguf.tensor_info("pointer.q.weight").is_some());
        assert!(gguf.tensor_info("pointer.k.weight").is_some());
        let temp = gguf
            .try_dequant_f16_to_f32("pointer.temperature.weight")
            .expect("temperature dequant")
            .expect("temperature present");
        assert_eq!(temp.len(), 1);
        // Norm tensors are F32 — cheap dequant smoke.
        let gamma = gguf
            .dequant_f16_to_f32("blk.0.attn_q_norm.weight")
            .expect("q norm");
        assert_eq!(gamma.len(), 128);
        assert!(gamma.iter().all(|v| v.is_finite()));
    }
}
