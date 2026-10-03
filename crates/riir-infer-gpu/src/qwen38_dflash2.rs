//! Issue 742 T3 — the DFlash2 trained block drafter (Qwen3.8-27B target),
//! ported from the llama.cpp PR #27342 reference implementation
//! (`F:/llama.cpp-pr/src/models/dflash.cpp` + the driver in
//! `common/speculative.cpp` `common_speculative_impl_draft_dflash`).
//!
//! CPU-only, eval-first (the acceptance harness is the consumer; the GPU
//! integration into the verify loop is a later unit). Weights load from the
//! z-lab BF16 safetensors (`z-lab/Qwen3.8-27B-DFlash2`, 3.85 GB — the
//! trained reference; the analogalok Q2_K GGUF is the quantized incumbent
//! column, measured acceptance-neutral per its HF card).
//!
//! # The contract (every piece anchored to the PR source)
//!
//! **Config** (z-lab `config.json`, cross-checked against the analogalok
//! GGUF metadata `dflash.*`): 5 layers, hidden 5120, 32 Q / 8 KV heads,
//! head_dim 128, FFN 17408 (SwiGLU), sliding_window 2048 (all layers),
//! **non-causal** attention (the trained semantics — the checkpoint's own
//! `config.json` carries `"is_causal": false`, honored by both z-lab
//! reference impls and the llama.cpp fork; the causal reading of the torch
//! fallback branch was implemented + REFUTED in Issue 989 — knob
//! `QWEN38_DFLASH2_CAUSAL=1`), RoPE **NEOX** half-split full-head 128 @
//! theta 1e7 (`LLM_ARCH_DFLASH` returns NEOX for the non-DSV4 backbones
//! per llama.cpp's `llama_model_rope_type`; the GGUF carries no
//! rope.dimension_count so the key_length default applies), rms_eps 1e-6
//! (HF config; NOTE the analogalok GGUF wrote 0.0 — a converter artifact
//! we do NOT replicate), block_size 8, conv kernel 2 / group 16,
//! selector rank 256 / top_k 16, `mask_token_id` 248070.
//!
//! **Encoder** (`graph<true>`): input = the target's residual stream at 5
//! tap points CONCATENATED — `[5 × 5120 = 25600]` — projected by
//! `fc.weight` `[5120, 25600]` then RMSNorm(`hidden_norm`) → the fused
//! feature `inp_g` `[5120]`.
//!
//! **Tap points**: llama.cpp extracts `embeddings_layer_inp` at the GGUF's
//! `dflash.target_layers = [6,20,34,48,62]` — the INPUT of those layers =
//! the OUTPUT of 0-indexed layers `[5,19,33,47,61]` = the post-MLP
//! residual stream the qwen38 forward's `forward_token_capture` taps.
//! (HF's config lists `[5,19,33,47,61]`; the GGUF's +1 shift is the same
//! tap expressed as the next layer's input.)
//!
//! **KV injection** (decoder `ubatch.embd` branch): per committed position
//! `p`, per drafter layer: `K = rope(k_norm(k_proj · inp_g))` (GQA 8
//! heads), `V = v_proj · inp_g`, written at ring slot `p % 2048`.
//!
//! **Noise block** (decoder token batch): tokens `[anchor, MASK × 7]` at
//! positions `n..n+7` (the anchor = the last committed token, at its OWN
//! position — llama.cpp's unified cache OVERWRITES the injected KV at `n`
//! with the anchor's token-embedding KV; this port excludes the ring's
//! injected entry at `n` and substitutes the block KV, identical
//! semantics). Embeddings come from the TARGET's `token_embd` (shared
//! weights — the drafter ships none). Per layer: `h = rmsnorm(x,
//! attn_norm)` → DFlash2 dynamic conv (side 0) → GQA attention over
//! `[ring window ∪ block KV]`, NON-CAUSAL (future keys visible —
//! `causal_attn=false` in the driver), SWA mask `key_pos >= query_pos −
//! (2048−1)` (`is_masked_swa` STANDARD: masked iff `q−k >= n_swa`),
//! scale 1/√128, q/k per-head RMSNorm then full-head RoPE → output conv
//! (side 1) → residual from the PRE-attn `x` → `rmsnorm(ffn_norm)` →
//! conv (side 0) → SwiLLM gate/up → down → conv (side 1) → residual.
//! Final `rmsnorm(output_norm)` → the TARGET's `lm_head` (shared) →
//! logits `[8, vocab]`.
//!
//! **The conv** (`build_dflash2_conv`): per token `i` (block-local) and
//! channel `c` (group `g = c/16` of 320):
//! `out[i][c] = Σ_t (dyn[i][g][t][s] + base[s][t][c]) · x[i−t][c]`, `t ∈
//! {0,1}`, zero-padded at block start (`x[<0] = 0`). `dyn` = the 1280-dim
//! `kernel_projection` output laid out `[320 groups, 2 taps, 2 sides]`
//! (ggml row-major ne order — `reshape_4d(dynamic, n_groups, kernel, 2,
//! tokens)`); side 0 = input conv, side 1 = output conv.
//!
//! **Selector lattice** (`build_post_sampling`, DFlash2 only): per draft
//! position `pos ∈ 1..=7`: top-16 candidates from the position's logits +
//! their unary logits; `code = sel_hidden · h_pos` (`[256]`);
//! `score(prev, succ) = succ_code · (prev_code ⊙ code)`; `score +=
//! unary(succ)`. Position 1's predecessor = the ANCHOR's
//! `predecessor_codebook` row; positions > 1 chain through the chosen
//! candidate at the previous position (the driver's single-chain greedy
//! walk — equivalent to reading the lattice row of the chosen
//! predecessor).
//!
//! **Chain walk** (the driver's greedy path): at each position take the
//! argmax over the 16 successor scores; confidence = `1/Σ_k exp(s_k −
//! s_max)` (the softmax probability of the argmax — the PR tip's own
//! p_min math); the `p_min` gate stops the draft at the first position
//! below the threshold. The harness evaluates n-max × p-min grids offline
//! from ONE walk (the walk carries per-position confidences).
//!
//! **Confidence temperature** (Bench 747 — the p-min calibration
//! residual): the PR's walk has TWO p-min branches keyed on the draft
//! temperature (`dp.temperature` = the SERVER request's sampling temp,
//! `server-context.cpp:2944`): greedy (≤ 0) compares `1/Σexp(s_k−s_max)`;
//! sampled (> 0) compares the CHOSEN candidate's probability under
//! `softmax(s/T)`. The T3.0 incumbent sweep ran `--temp 0.6` → the
//! SAMPLED branch — the incumbent's p-min 0.82 gate compared against
//! T=0.6-sharpened confidences, while our offline grid compared against
//! the T=1 greedy formula. `DraftWalk::confidence(pos, T)` is the one
//! source of truth for BOTH regimes (identical to the greedy chain
//! confidence at T = 1); the gate direction (`conf < p_min` → stop) is
//! the PR's and never changed — the Bench-746 "inverted calibration"
//! was a metric-semantics + temperature-regime mismatch, not a sign bug.

use std::collections::VecDeque;
use std::path::Path;

use katgpt_core::{keyed_gumbel_noise, truncation_keep_mask};
use rayon::prelude::*;

/// The drafter config (z-lab `config.json` values; see the module header).
#[derive(Debug, Clone)]
pub struct DFlash2Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub block_size: usize,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub rms_eps: f32,
    pub vocab_size: usize,
    pub mask_token_id: u32,
}

impl Default for DFlash2Config {
    fn default() -> Self {
        Self {
            n_layer: 5,
            n_embd: 5120,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            n_ff: 17408,
            block_size: 8,
            conv_kernel: 2,
            conv_group: 16,
            selector_rank: 256,
            selector_top_k: 16,
            rope_theta: 1.0e7,
            sliding_window: 2048,
            // z-lab config.json `rms_norm_eps: 1e-06` (the analogalok GGUF
            // wrote 0.0 — converter artifact, not replicated).
            rms_eps: 1.0e-6,
            vocab_size: 248_320,
            mask_token_id: 248_070,
        }
    }
}

/// One drafter layer's weights, all f32 row-major `[out][in]` (HF Linear
/// convention: `y = W · x`).
#[derive(Debug, Default, Clone)]
pub struct DFlash2Layer {
    pub attn_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    /// HF `base_kernel` `[side][tap][channel]` — same memory layout as the
    /// GGUF's `[channel, tap, side]` ne order (channel fastest in both).
    pub attn_conv_base: Vec<f32>,
    pub attn_conv_proj: Vec<f32>,
    pub ffn_conv_base: Vec<f32>,
    pub ffn_conv_proj: Vec<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct DFlash2Weights {
    pub fc: Vec<f32>,
    pub hidden_norm: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub sel_hidden: Vec<f32>,
    pub sel_prev: Vec<f32>,
    pub sel_next: Vec<f32>,
    pub layers: Vec<DFlash2Layer>,
}

// ── minimal safetensors reader (BF16 → f32; no new deps) ──

struct SafetensorsFile {
    file: std::fs::File,
    index: std::collections::HashMap<String, (String, Vec<usize>, u64)>,
}

impl SafetensorsFile {
    fn open(path: &Path) -> Result<Self, String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| format!("read header len: {e}"))?;
        let hlen = u64::from_le_bytes(len_buf) as usize;
        let mut hdr = vec![0u8; hlen];
        file.read_exact(&mut hdr)
            .map_err(|e| format!("read header: {e}"))?;
        let parsed: serde_json::Value =
            serde_json::from_slice(&hdr).map_err(|e| format!("parse header json: {e}"))?;
        let obj = parsed
            .as_object()
            .ok_or_else(|| "header not an object".to_string())?;
        let data_base = 8 + hlen as u64;
        let mut index = std::collections::HashMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{name}: no dtype"))?
                .to_string();
            let shape: Vec<usize> = meta
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("{name}: no shape"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let beg = meta
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            index.insert(name.clone(), (dtype, shape, data_base + beg));
        }
        Ok(Self { file, index })
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        self.index
            .get(name)
            .map(|(_, s, _)| s.clone())
            .ok_or_else(|| format!("tensor {name} not in file"))
    }

    /// Read one tensor as f32 (BF16 widened by bit shift; F32 passed through).
    fn read_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        use std::io::{Read, Seek, SeekFrom};
        let (dtype, shape, off) = self
            .index
            .get(name)
            .cloned()
            .ok_or_else(|| format!("tensor {name} not in file"))?;
        let n: usize = shape.iter().product();
        let mut out = Vec::with_capacity(n);
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| format!("seek {name}: {e}"))?;
        match dtype.as_str() {
            "BF16" => {
                let mut raw = vec![0u8; n * 2];
                self.file
                    .read_exact(&mut raw)
                    .map_err(|e| format!("read {name}: {e}"))?;
                // bf16 -> f32: the 16 bits become the HIGH half of the f32.
                // from_le_bytes([0, 0, lo, hi]) places lo at byte 2 and hi at
                // byte 3 — exactly `bf16_bits << 16` with NO further shift.
                out.extend(
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| f32::from_bits(u32::from_le_bytes([0, 0, c[0], c[1]]))),
                );
            }
            "F32" => {
                let mut raw = vec![0u8; n * 4];
                self.file
                    .read_exact(&mut raw)
                    .map_err(|e| format!("read {name}: {e}"))?;
                out.extend(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
                );
            }
            other => return Err(format!("{name}: unsupported dtype {other}")),
        }
        Ok(out)
    }
}

/// Load the z-lab DFlash2 drafter from `model.safetensors` (every shape
/// VALIDATED against the config before reading).
pub fn load_dflash2_safetensors(
    path: &Path,
    cfg: &DFlash2Config,
) -> Result<DFlash2Weights, String> {
    let mut st = SafetensorsFile::open(path)?;
    let expect = |st: &SafetensorsFile, name: &str, want: &[usize]| -> Result<(), String> {
        let got = st.shape(name)?;
        if got != want {
            return Err(format!("{name}: shape {got:?} != expected {want:?}"));
        }
        Ok(())
    };
    let e = cfg.n_embd;
    let ff = cfg.n_ff;
    let qd = cfg.n_head * cfg.head_dim;
    let kvd = cfg.n_kv_head * cfg.head_dim;
    let rank = cfg.selector_rank;
    let v = cfg.vocab_size;
    let inp = 5 * e;

    expect(&st, "fc.weight", &[e, inp])?;
    expect(&st, "hidden_norm.weight", &[e])?;
    // HF name `norm.weight` (the GGUF calls it `output_norm.weight`).
    expect(&st, "norm.weight", &[e])?;
    expect(
        &st,
        "candidate_selector.hidden_projection.weight",
        &[rank, e],
    )?;
    expect(&st, "candidate_selector.predecessor_codebook", &[v, rank])?;
    expect(&st, "candidate_selector.successor_codebook", &[v, rank])?;

    let mut layers = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let p = |s: &str| format!("layers.{i}.{s}");
        expect(&st, &p("input_layernorm.weight"), &[e])?;
        expect(&st, &p("self_attn.q_proj.weight"), &[qd, e])?;
        expect(&st, &p("self_attn.k_proj.weight"), &[kvd, e])?;
        expect(&st, &p("self_attn.v_proj.weight"), &[kvd, e])?;
        expect(&st, &p("self_attn.o_proj.weight"), &[e, qd])?;
        expect(&st, &p("self_attn.q_norm.weight"), &[cfg.head_dim])?;
        expect(&st, &p("self_attn.k_norm.weight"), &[cfg.head_dim])?;
        expect(&st, &p("post_attention_layernorm.weight"), &[e])?;
        expect(&st, &p("mlp.gate_proj.weight"), &[ff, e])?;
        expect(&st, &p("mlp.up_proj.weight"), &[ff, e])?;
        expect(&st, &p("mlp.down_proj.weight"), &[e, ff])?;
        expect(&st, &p("attention_conv.base_kernel"), &[2, 2, e])?;
        expect(
            &st,
            &p("attention_conv.kernel_projection.weight"),
            &[1280, e],
        )?;
        expect(&st, &p("mlp_conv.base_kernel"), &[2, 2, e])?;
        expect(&st, &p("mlp_conv.kernel_projection.weight"), &[1280, e])?;
        layers.push(DFlash2Layer {
            attn_norm: st.read_f32(&p("input_layernorm.weight"))?,
            q_proj: st.read_f32(&p("self_attn.q_proj.weight"))?,
            k_proj: st.read_f32(&p("self_attn.k_proj.weight"))?,
            v_proj: st.read_f32(&p("self_attn.v_proj.weight"))?,
            o_proj: st.read_f32(&p("self_attn.o_proj.weight"))?,
            q_norm: st.read_f32(&p("self_attn.q_norm.weight"))?,
            k_norm: st.read_f32(&p("self_attn.k_norm.weight"))?,
            ffn_norm: st.read_f32(&p("post_attention_layernorm.weight"))?,
            gate_proj: st.read_f32(&p("mlp.gate_proj.weight"))?,
            up_proj: st.read_f32(&p("mlp.up_proj.weight"))?,
            down_proj: st.read_f32(&p("mlp.down_proj.weight"))?,
            attn_conv_base: st.read_f32(&p("attention_conv.base_kernel"))?,
            attn_conv_proj: st.read_f32(&p("attention_conv.kernel_projection.weight"))?,
            ffn_conv_base: st.read_f32(&p("mlp_conv.base_kernel"))?,
            ffn_conv_proj: st.read_f32(&p("mlp_conv.kernel_projection.weight"))?,
        });
    }

    Ok(DFlash2Weights {
        fc: st.read_f32("fc.weight")?,
        hidden_norm: st.read_f32("hidden_norm.weight")?,
        output_norm: st.read_f32("norm.weight")?,
        sel_hidden: st.read_f32("candidate_selector.hidden_projection.weight")?,
        sel_prev: st.read_f32("candidate_selector.predecessor_codebook")?,
        sel_next: st.read_f32("candidate_selector.successor_codebook")?,
        layers,
    })
}

// ── small math helpers (self-contained — no engine convention ambiguity) ──

#[inline]
fn rmsnorm_into(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * gamma[i];
    }
}

#[inline]
fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `dst[o] = Σ_i w[o·in + i] · x[i]` — one GEMV row-block.
#[inline]
fn gemv_into(w: &[f32], x: &[f32], dst: &mut [f32]) {
    let inn = x.len();
    for (o, d) in dst.iter_mut().enumerate() {
        let row = &w[o * inn..(o + 1) * inn];
        let mut acc = 0.0f32;
        for (i, &v) in x.iter().enumerate() {
            acc += row[i] * v;
        }
        *d = acc;
    }
}

/// rayon-parallel `gemv_into` (output rows split across cores - the fc
/// encoder GEMV is 131M MAC/position and the injection loop would be
/// minutes single-threaded).
fn gemv_into_par(w: &[f32], x: &[f32], dst: &mut [f32]) {
    let inn = x.len();
    assert_eq!(w.len(), dst.len() * inn);
    dst.par_iter_mut().enumerate().for_each(|(o, d)| {
        let row = &w[o * inn..(o + 1) * inn];
        let mut acc = 0.0f32;
        for (i, &v) in x.iter().enumerate() {
            acc += row[i] * v;
        }
        *d = acc;
    });
}

/// Top-k indices by value, descending (earlier index wins ties — the
/// first-index argmax convention). O(n) scan with bounded insert.
fn top_k_desc(vals: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = Vec::with_capacity(k + 1);
    let mut best: Vec<f32> = Vec::with_capacity(k + 1);
    for (i, &v) in vals.iter().enumerate() {
        if best.len() < k || v > *best.last().expect("non-empty") {
            // insert into the descending list after all entries >= v
            // (ties keep the EARLIER index first).
            let pos = best.partition_point(|&b| b >= v);
            best.insert(pos, v);
            idx.insert(pos, i);
            if best.len() > k {
                best.pop();
                idx.pop();
            }
        }
    }
    idx
}

/// The per-draft result of ONE noise-block forward + lattice + walk.
#[derive(Debug, Default, Clone)]
pub struct DraftWalk {
    /// `(token, confidence)` per draft position 1..=7 (block index). The
    /// confidence is the PR's GREEDY formula (`1/Σexp(s_k−s_max)` at
    /// T=1) — see [`DraftWalk::confidence`] for the temperature regimes.
    pub chain: Vec<(u32, f32)>,
    /// Raw top-1-from-logits per position (the DFlash1-style ablation).
    pub logits_top1: Vec<u32>,
    /// Top-16 candidate ids per position (index 0 = block pos 1).
    pub candidates: Vec<Vec<u32>>,
    /// The 16 selector scores per position (dot + unary, candidate order
    /// — the lattice row segment the driver argmaxes). The ONE source of
    /// truth every p-min confidence derives from (Bench 747).
    pub scores: Vec<Vec<f32>>,
}

impl DraftWalk {
    /// The PR-27342 p-min confidence at `pos` under `temperature` —
    /// the walk's one source of truth for the gate, covering BOTH of the
    /// driver's branches (`common_speculative_impl_draft_dflash`):
    ///
    /// - `temperature <= 0.0` (greedy, the PR's else-branch): the argmax's
    ///   `1 / Σ_k exp(s_k − s_max)` — numerically identical to
    ///   `chain[pos].1` and to `temperature == 1.0`.
    /// - `temperature > 0.0` (the PR's sampled branch): the CHOSEN
    ///   candidate's probability under `softmax(s/T)`. The incumbent
    ///   T3.0 sweep ran the server at `--temp 0.6` (`server-context.cpp`
    ///   passes the request's sampling temp to the drafter), so ITS p-min
    ///   0.82 gate compared against this T-sharpened quantity.
    ///
    /// The gate direction is the PR's own and lives at the consumer:
    /// `confidence(pos, T) < p_min` → stop drafting (never `>`).
    ///
    /// Divergence note (deterministic offline grids): the PR's sampled
    /// branch SAMPLES the chain at T; this walk keeps the deterministic
    /// argmax choice and reports its T-scaled probability — an upper
    /// bound on the sampled candidate's confidence at the same position.
    pub fn confidence(&self, pos: usize, temperature: f32) -> f32 {
        let scores = &self.scores[pos];
        assert!(!scores.is_empty(), "walk position {pos} carries no scores");
        let tok = self.chain[pos].0;
        let idx = self.candidates[pos]
            .iter()
            .position(|&t| t == tok)
            .unwrap_or_else(|| panic!("chain token {tok} missing from candidates at {pos}"));
        let t = if temperature > 0.0 { temperature } else { 1.0 };
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &s in scores {
            sum += ((s - max) / t).exp();
        }
        ((scores[idx] - max) / t).exp() / sum
    }
}

/// The stateful drafter: feature ring + injection + block drafting.
pub struct DFlash2Drafter {
    pub cfg: DFlash2Config,
    pub w: DFlash2Weights,
    /// Per layer: K/V ring `[2048][n_kv_head*head_dim]`.
    ring_k: Vec<Vec<f32>>,
    ring_v: Vec<Vec<f32>>,
    /// Positions currently stored in the ring (one shared occupancy — every
    /// layer injects at the same positions; ascending).
    ring_pos: VecDeque<usize>,
    /// RoPE base frequencies `[head_dim/2]` (angle = pos · freq[i]).
    rope_freqs: Vec<f32>,
}

impl DFlash2Drafter {
    /// Reset the feature ring for a fresh sequence (the KV slots need no
    /// clearing — occupancy is tracked by `ring_pos` alone and slots are
    /// rewritten on injection).
    pub fn reset_ring(&mut self) {
        self.ring_pos.clear();
    }

    pub fn new(cfg: DFlash2Config, w: DFlash2Weights) -> Self {
        let hd = cfg.head_dim;
        let mut rope_freqs = vec![0.0f32; hd / 2];
        for (i, f) in rope_freqs.iter_mut().enumerate() {
            *f = cfg.rope_theta.powf(-(2.0 * i as f32) / hd as f32);
        }
        let n_kv = cfg.n_kv_head * hd;
        let swa = cfg.sliding_window;
        let ring_k = vec![vec![0.0f32; swa * n_kv]; cfg.n_layer];
        let ring_v = vec![vec![0.0f32; swa * n_kv]; cfg.n_layer];
        Self {
            cfg,
            w,
            ring_k,
            ring_v,
            ring_pos: VecDeque::with_capacity(swa + 8),
            rope_freqs,
        }
    }

    /// Encode one position's fused target features (the 5×5120 concat, tap
    /// layers in ascending order) and inject per-layer K/V at `pos`.
    /// Positions must be injected in ascending order (contiguous prefix).
    pub fn inject_position(&mut self, features_5x: &[f32], pos: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let hd = cfg.head_dim;
        let kvh = cfg.n_kv_head;
        assert_eq!(
            features_5x.len(),
            5 * e,
            "features must be the 5x5120 concat"
        );
        if self.ring_pos.back() >= Some(&pos) {
            return Err(format!(
                "inject: pos {pos} not beyond ring back {:?} (must ascend)",
                self.ring_pos.back()
            ));
        }

        // Encoder: fc (5120×25600 GEMV) + hidden_norm.
        let mut inp_g = vec![0.0f32; e];
        gemv_into_par(&self.w.fc, features_5x, &mut inp_g);
        let mut inp_normed = vec![0.0f32; e];
        rmsnorm_into(&inp_g, &self.w.hidden_norm, cfg.rms_eps, &mut inp_normed);
        let inp_g = inp_normed;

        // Per-layer K/V injection (k_norm per head, then RoPE at `pos`).
        let n_kv = kvh * hd;
        let mut k = vec![0.0f32; n_kv];
        let mut v = vec![0.0f32; n_kv];
        let mut normed = vec![0.0f32; hd];
        for (li, layer) in self.w.layers.iter().enumerate() {
            gemv_into(&layer.k_proj, &inp_g, &mut k);
            gemv_into(&layer.v_proj, &inp_g, &mut v);
            for h in 0..kvh {
                let seg = &mut k[h * hd..(h + 1) * hd];
                rmsnorm_into(seg, &layer.k_norm, cfg.rms_eps, &mut normed);
                apply_rope_in_place(&mut normed, pos, &self.rope_freqs);
                seg.copy_from_slice(&normed);
            }
            let slot = pos % cfg.sliding_window;
            self.ring_k[li][slot * n_kv..(slot + 1) * n_kv].copy_from_slice(&k);
            self.ring_v[li][slot * n_kv..(slot + 1) * n_kv].copy_from_slice(&v);
        }

        if self.ring_pos.len() == cfg.sliding_window {
            self.ring_pos.pop_front();
        }
        self.ring_pos.push_back(pos);
        Ok(())
    }

    /// Draft at anchor position `n` (the anchor token at its OWN position).
    ///
    /// `embed_row(token) -> [n_embd]` supplies the TARGET's token embedding
    /// (shared weights); `lm_head_rows(hidden_rows, n_rows) -> logits`
    /// supplies the TARGET's output projection (shared). The block's
    /// post-output_norm rows are piped straight into it (no second norm —
    /// the PR-27342 graph contract).
    /// Phase 1 of the two-phase eval path: the noise-block forward through
    /// the 5 layers + final norm. Returns the `[block_size][n_embd]`
    /// post-output_norm rows (the lm_head input; row 0 = the anchor slot).
    /// Read-only on the drafter — the harness runs many anchors in
    /// parallel via rayon (teacher-forced drafts are independent).
    #[allow(clippy::too_many_lines)]
    pub fn draft_block_hidden(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let hd = cfg.head_dim;
        let nh = cfg.n_head;
        let kvh = cfg.n_kv_head;
        let grp = nh / kvh;
        let bs = cfg.block_size;
        let swa = cfg.sliding_window;
        let n_groups = e / cfg.conv_group;
        let projected = 2 * cfg.conv_kernel * n_groups;

        // ── the noise block input: anchor + masks, target embeddings ──
        let mut x = vec![0.0f32; bs * e];
        let anchor_emb = embed_row(anchor_token);
        let mask_emb = embed_row(cfg.mask_token_id);
        x[..e].copy_from_slice(&anchor_emb);
        for i in 1..bs {
            x[i * e..(i + 1) * e].copy_from_slice(&mask_emb);
        }

        // ── the layer stack ──
        let mut h = vec![0.0f32; bs * e];
        let mut hc = vec![0.0f32; bs * e];
        let mut dyn_coeff = vec![0.0f32; bs * projected];
        let mut q = vec![0.0f32; bs * nh * hd];
        let mut k = vec![0.0f32; bs * kvh * hd];
        let mut vv = vec![0.0f32; bs * kvh * hd];
        let mut attn_out = vec![0.0f32; bs * nh * hd];
        let mut ao = vec![0.0f32; bs * e];
        let mut attn_final = vec![0.0f32; bs * e];
        let mut ffn_inp = vec![0.0f32; bs * e];
        let mut hf = vec![0.0f32; bs * e];
        let mut dynf = vec![0.0f32; bs * projected];
        let mut hfc = vec![0.0f32; bs * e];
        let mut mid = vec![0.0f32; bs * cfg.n_ff];
        let mut down_out = vec![0.0f32; bs * e];
        let mut ffn_final = vec![0.0f32; bs * e];
        let mut normed = vec![0.0f32; hd];
        let mut ovec = vec![0.0f32; hd];

        for li in 0..cfg.n_layer {
            let layer = &self.w.layers[li];

            // h = rmsnorm(x, attn_norm); attn dynamic coefficients; side-0 conv.
            for i in 0..bs {
                rmsnorm_into(
                    &x[i * e..(i + 1) * e],
                    &layer.attn_norm,
                    cfg.rms_eps,
                    &mut h[i * e..(i + 1) * e],
                );
                gemv_into(
                    &layer.attn_conv_proj,
                    &h[i * e..(i + 1) * e],
                    &mut dyn_coeff[i * projected..(i + 1) * projected],
                );
            }
            apply_conv(
                &dyn_coeff,
                &layer.attn_conv_base,
                0,
                bs,
                e,
                cfg,
                &h,
                &mut hc,
            );

            // Q/K/V (block-local).
            for i in 0..bs {
                let hseg = &hc[i * e..(i + 1) * e];
                gemv_into(&layer.q_proj, hseg, &mut q[i * nh * hd..(i + 1) * nh * hd]);
                gemv_into(
                    &layer.k_proj,
                    hseg,
                    &mut k[i * kvh * hd..(i + 1) * kvh * hd],
                );
                gemv_into(
                    &layer.v_proj,
                    hseg,
                    &mut vv[i * kvh * hd..(i + 1) * kvh * hd],
                );
            }

            // per-head norms + RoPE at the block positions.
            for i in 0..bs {
                let p = anchor_pos + i;
                for hh in 0..nh {
                    let base = i * nh * hd + hh * hd;
                    let seg = &mut q[base..base + hd];
                    rmsnorm_into(seg, &layer.q_norm, cfg.rms_eps, &mut normed);
                    apply_rope_in_place(&mut normed, p, &self.rope_freqs);
                    seg.copy_from_slice(&normed);
                }
                for hh in 0..kvh {
                    let base = i * kvh * hd + hh * hd;
                    let seg = &mut k[base..base + hd];
                    rmsnorm_into(seg, &layer.k_norm, cfg.rms_eps, &mut normed);
                    apply_rope_in_place(&mut normed, p, &self.rope_freqs);
                    seg.copy_from_slice(&normed);
                }
            }

            // Attention over [ring window ∪ block KV], non-causal + SWA
            // (the TRAINED semantics — config.json is_causal:false, Issue
            // 989's falsification record; QWEN38_DFLASH2_CAUSAL=1 arms the
            // refuted causal arm).
            let causal = dflash2_causal_block();
            let ring_k = &self.ring_k[li];
            let ring_v = &self.ring_v[li];
            let scale = 1.0f32 / (hd as f32).sqrt();
            for i in 0..bs {
                let qp = anchor_pos + i;
                let lo = qp.saturating_sub(swa - 1);
                for hh in 0..nh {
                    let kvh_idx = hh / grp;
                    let qb = i * nh * hd + hh * hd;
                    let qseg = &q[qb..qb + hd];

                    // Pass 1: max unscaled score over visible keys.
                    // Issue 989 (Bench 942): ring keys at positions BEYOND the
                    // anchor are FUTURE committed tokens — the reference never
                    // holds them at draft time (its cache is `[0..start)`), and
                    // attending them flips 99.5% of chains and destroys
                    // acceptance (measured 0.474 vs 1.647 mean prefix — the
                    // bench-746 offline Phase-B shape). Masked here so the
                    // offline rayon harness (full stream pre-injected) matches
                    // the live loop's semantics; a no-op for the live ring.
                    let mut max_s = f32::NEG_INFINITY;
                    for bj in 0..bs {
                        if causal && bj > i {
                            continue;
                        }
                        if anchor_pos + bj < lo {
                            continue;
                        }
                        let base = bj * kvh * hd + kvh_idx * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += qseg[d] * k[base + d];
                        }
                        max_s = max_s.max(dot);
                    }
                    for &rp in self.ring_pos.iter() {
                        if rp == anchor_pos || rp < lo || rp > anchor_pos {
                            continue;
                        }
                        let base = (rp % swa) * kvh * hd + kvh_idx * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += qseg[d] * ring_k[base + d];
                        }
                        max_s = max_s.max(dot);
                    }

                    // Pass 2: softmax weights + V accumulation.
                    let mut sum = 0.0f32;
                    ovec.iter_mut().for_each(|v| *v = 0.0);
                    for bj in 0..bs {
                        if causal && bj > i {
                            continue;
                        }
                        if anchor_pos + bj < lo {
                            continue;
                        }
                        let base = bj * kvh * hd + kvh_idx * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += qseg[d] * k[base + d];
                        }
                        let w = ((dot - max_s) * scale).exp();
                        sum += w;
                        for d in 0..hd {
                            ovec[d] += w * vv[base + d];
                        }
                    }
                    for &rp in self.ring_pos.iter() {
                        if rp == anchor_pos || rp < lo || rp > anchor_pos {
                            continue;
                        }
                        let slot = rp % swa;
                        let base = slot * kvh * hd + kvh_idx * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += qseg[d] * ring_k[base + d];
                        }
                        let w = ((dot - max_s) * scale).exp();
                        sum += w;
                        for d in 0..hd {
                            ovec[d] += w * ring_v[base + d];
                        }
                    }
                    let ob = i * nh * hd + hh * hd;
                    for d in 0..hd {
                        attn_out[ob + d] = ovec[d] / sum;
                    }
                }
            }

            // o_proj + side-1 conv + residual from the PRE-attention x.
            for i in 0..bs {
                gemv_into(
                    &layer.o_proj,
                    &attn_out[i * nh * hd..(i + 1) * nh * hd],
                    &mut ao[i * e..(i + 1) * e],
                );
            }
            apply_conv(
                &dyn_coeff,
                &layer.attn_conv_base,
                1,
                bs,
                e,
                cfg,
                &ao,
                &mut attn_final,
            );
            for j in 0..bs * e {
                ffn_inp[j] = attn_final[j] + x[j];
            }

            // FFN: norm → side-0 conv → SwiGLU → down → side-1 conv → residual.
            for i in 0..bs {
                rmsnorm_into(
                    &ffn_inp[i * e..(i + 1) * e],
                    &layer.ffn_norm,
                    cfg.rms_eps,
                    &mut hf[i * e..(i + 1) * e],
                );
                gemv_into(
                    &layer.ffn_conv_proj,
                    &hf[i * e..(i + 1) * e],
                    &mut dynf[i * projected..(i + 1) * projected],
                );
            }
            apply_conv(&dynf, &layer.ffn_conv_base, 0, bs, e, cfg, &hf, &mut hfc);
            for i in 0..bs {
                let hseg = &hfc[i * e..(i + 1) * e];
                let mseg = &mut mid[i * cfg.n_ff..(i + 1) * cfg.n_ff];
                for (o, row) in layer.gate_proj.chunks(e).enumerate() {
                    let mut g = 0.0f32;
                    let mut u = 0.0f32;
                    let urow = &layer.up_proj[o * e..(o + 1) * e];
                    for (j, &val) in hseg.iter().enumerate() {
                        g += row[j] * val;
                        u += urow[j] * val;
                    }
                    mseg[o] = silu(g) * u;
                }
            }
            for i in 0..bs {
                gemv_into(
                    &layer.down_proj,
                    &mid[i * cfg.n_ff..(i + 1) * cfg.n_ff],
                    &mut down_out[i * e..(i + 1) * e],
                );
            }
            apply_conv(
                &dynf,
                &layer.ffn_conv_base,
                1,
                bs,
                e,
                cfg,
                &down_out,
                &mut ffn_final,
            );
            for j in 0..bs * e {
                x[j] = ffn_final[j] + ffn_inp[j];
            }
        }

        // Final norm → t_embd.
        let mut t_embd = vec![0.0f32; bs * e];
        for i in 0..bs {
            rmsnorm_into(
                &x[i * e..(i + 1) * e],
                &self.w.output_norm,
                cfg.rms_eps,
                &mut t_embd[i * e..(i + 1) * e],
            );
        }

        t_embd
    }

    /// Phase 2 of the two-phase eval path: the selector lattice + greedy
    /// chain walk over the block's lm_head logits. `hidden_rows` is the
    /// `draft_block_hidden` output; `logits` the target's lm_head rows for
    /// the MASK positions 1..block_size-1 (`[block_size-1][vocab]`
    /// flattened, in block order).
    pub fn lattice_walk(
        &self,
        anchor_token: u32,
        hidden_rows: &[f32],
        logits: &[f32],
    ) -> DraftWalk {
        let (candidates, unary, logits_top1, codes) = self.walk_rows(hidden_rows, logits);

        // Chain walk with per-position confidence (the greedy driver path).
        let mut chain: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        let mut walk_scores: Vec<Vec<f32>> = Vec::with_capacity(candidates.len());
        let mut pred_tok = anchor_token;
        for i in 0..candidates.len() {
            let scores = self.chain_step_scores(pred_tok, &candidates[i], &unary[i], &codes[i]);
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for (ki, &s) in scores.iter().enumerate() {
                if s > best_s {
                    best_s = s;
                    best = ki;
                }
            }
            let mut sum = 0.0f32;
            for &s in scores.iter() {
                sum += (s - best_s).exp();
            }
            chain.push((candidates[i][best], 1.0 / sum));
            walk_scores.push(scores);
            pred_tok = candidates[i][best];
        }

        DraftWalk {
            chain,
            logits_top1,
            candidates,
            scores: walk_scores,
        }
    }

    /// The per-row phase both chain walks share — asserts + the top-k /
    /// `sel_hidden` computation, moved verbatim out of `lattice_walk` (the
    /// rayon knob and row order unchanged, so the greedy path's values are
    /// bit-identical to the pre-extraction code).
    #[allow(clippy::type_complexity)]
    fn walk_rows(
        &self,
        hidden_rows: &[f32],
        logits: &[f32],
    ) -> (Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<u32>, Vec<Vec<f32>>) {
        /// One row's walk inputs: (candidates, unary, top1, hidden code).
        type WalkRow = (Vec<u32>, Vec<f32>, u32, Vec<f32>);

        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let bs = cfg.block_size;
        assert_eq!(hidden_rows.len(), bs * e);
        let n_out = bs - 1;
        assert_eq!(logits.len(), n_out * cfg.vocab_size);

        let top_k = cfg.selector_top_k;
        let rank = cfg.selector_rank;
        // Per-row phase (top-k over each 151k row + the sel_hidden gemv) —
        // the 7 rows are INDEPENDENT. Bench 759 §The mechanism, IDENTIFIED:
        // single-threaded, this phase is CPU-core-placement-sensitive on a
        // hybrid CPU (8.3 ms Thread-Director-placed on E-cores vs 4.7 on
        // P-cores — the whole G2 PASS/FAIL swing). Under rayon it is
        // placement-insensitive and faster than either single-threaded
        // placement. Bit-identical by construction: per-row computation
        // unchanged, rows collected in order. DFLASH2_WALK_PAR=0 restores
        // the sequential path (the A/B knob).
        let par = std::env::var("DFLASH2_WALK_PAR").as_deref() != Ok("0");
        let mut candidates: Vec<Vec<u32>> = Vec::with_capacity(n_out);
        let mut unary: Vec<Vec<f32>> = Vec::with_capacity(n_out);
        let mut logits_top1: Vec<u32> = Vec::with_capacity(n_out);
        let mut codes: Vec<Vec<f32>> = Vec::with_capacity(n_out);
        if par {
            let per_row: Vec<WalkRow> = (0..n_out)
                .into_par_iter()
                .map(|i| {
                    let lrow = &logits[i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
                    let idx = top_k_desc(lrow, top_k);
                    let cands: Vec<u32> = idx.iter().map(|&t| t as u32).collect();
                    let un: Vec<f32> = idx.iter().map(|&t| lrow[t]).collect();
                    let top1 = idx[0] as u32;
                    let hseg = &hidden_rows[(i + 1) * e..(i + 2) * e];
                    let mut c = vec![0.0f32; rank];
                    gemv_into(&self.w.sel_hidden, hseg, &mut c);
                    (cands, un, top1, c)
                })
                .collect();
            for (cands, un, top1, c) in per_row {
                candidates.push(cands);
                unary.push(un);
                logits_top1.push(top1);
                codes.push(c);
            }
        } else {
            for i in 0..n_out {
                let lrow = &logits[i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
                let idx = top_k_desc(lrow, top_k);
                candidates.push(idx.iter().map(|&t| t as u32).collect());
                unary.push(idx.iter().map(|&t| lrow[t]).collect());
                logits_top1.push(idx[0] as u32);
            }
            for i in 0..n_out {
                let hseg = &hidden_rows[(i + 1) * e..(i + 2) * e];
                let mut c = vec![0.0f32; rank];
                gemv_into(&self.w.sel_hidden, hseg, &mut c);
                codes.push(c);
            }
        }
        (candidates, unary, logits_top1, codes)
    }

    /// One chain step's selector scores: `dot(sel_next[cand],
    /// sel_prev[pred] ⊙ code) + unary` per candidate — shared verbatim by
    /// the greedy and keyed chain walks (identical arithmetic and order,
    /// so the greedy path's scores are bit-identical to the pre-extraction
    /// code).
    fn chain_step_scores(
        &self,
        pred_tok: u32,
        candidates: &[u32],
        unary: &[f32],
        code: &[f32],
    ) -> Vec<f32> {
        let rank = self.cfg.selector_rank;
        let pcode = &self.w.sel_prev[pred_tok as usize * rank..(pred_tok as usize + 1) * rank];
        let mut scores = vec![0.0f32; candidates.len()];
        for (ki, &tok) in candidates.iter().enumerate() {
            let ncode = &self.w.sel_next[tok as usize * rank..(tok as usize + 1) * rank];
            let mut dot = 0.0f32;
            for r in 0..rank {
                dot += ncode[r] * (pcode[r] * code[r]);
            }
            scores[ki] = dot + unary[ki];
        }
        scores
    }

    /// The keyed-sampled chain walk — the exact-sampling posture (Plan 614
    /// Phase 1, the drafter half of a LOSSLESS SAMPLED-TARGET production
    /// lane). At each draft position `anchor_pos + 1 + i` the picked
    /// candidate is the keyed-Gumbel argmax over the row's selector
    /// scores, with the noise keyed by `(seed, absolute position,
    /// candidate token)` — the keyed stream the target's verify loop
    /// samples with once Phase 2 wires the target side
    /// (`katgpt_core::keyed_gumbel_max_sample`; no target-side caller
    /// exists yet — the joint key-convention test below pins the pair).
    /// A draft token then equals the target's own keyed sample exactly
    /// when the two score orderings agree after the shared noise, and the
    /// pick is invariant to evaluation order and batch composition by
    /// construction — "a draft is accepted exactly when it equals the
    /// token serial decoding samples there" (the byte-exact contract).
    ///
    /// Scope, stated honestly (verdict round 1): this does NOT lift the
    /// Bench-746/747 greedy-stream 1.92 — that comparison ran greedy
    /// drafter vs greedy target, already the most favourable regime, and
    /// the residual analysis there stands. What it buys: the production
    /// lane's target decodes SAMPLED (chat T ≈ 0.6), where a greedy chain
    /// accepts only at the target's per-position mode probability; under
    /// the shared keyed stream acceptance approaches 1 as the drafter's
    /// distribution converges to the target's (shared-Gumbel coupling:
    /// P(match) ≥ (1−TV)/(1+TV), exact 1 at equality).
    ///
    /// Truncation contract (Phase 2 must honor): the deployment sampler is
    /// truncated (Qwen3 recommends top-p 0.95 / top-k 20 at T = 0.6), so
    /// the target must mask its logits BEFORE the keyed argmax, and the
    /// drafter should apply the same mask to its candidate set — a pick
    /// outside the target's nucleus is a guaranteed miss. Masking first
    /// also bounds the target side's cost to the survivors.
    ///
    /// `temperature` scales the scores (`s/T`) exactly as the target's
    /// sampler scales its logits; `temperature <= 0` delegates to the
    /// greedy chain walk (bit-identical to [`Self::lattice_walk`] — the
    /// tested posture). `chain[i].1` carries the chosen candidate's
    /// `softmax(s/T)` probability, which is exactly what
    /// [`DraftWalk::confidence`] returns for the picked token — the p-min
    /// gate reads it unchanged (NOTE for the A/B: the incumbent's p-min
    /// 0.82 was calibrated on argmax-pick confidences; keyed picks are
    /// often non-argmax, so the gate truncates more and the threshold must
    /// be re-calibrated in-run).
    pub fn lattice_walk_keyed(
        &self,
        anchor_token: u32,
        hidden_rows: &[f32],
        logits: &[f32],
        seed: u64,
        anchor_pos: usize,
        temperature: f32,
    ) -> DraftWalk {
        self.lattice_walk_keyed_impl(
            anchor_token,
            hidden_rows,
            logits,
            seed,
            anchor_pos,
            temperature,
            None,
        )
    }

    /// The keyed walk under the DEPLOYMENT TRUNCATION — the Plan-614
    /// Phase-2 contract's drafter half: "the drafter should apply the same
    /// mask to its candidate set — a pick outside the target's nucleus is a
    /// guaranteed miss." Per block row the drafter builds
    /// [`katgpt_core::truncation_keep_mask`] over its OWN lm_head logits
    /// (the same `(top_k, top_p)` the target's sampler uses — the mask
    /// semantics live in ONE place), then the keyed pick consults only the
    /// SURVIVING candidates, and the recorded confidence is the chosen
    /// token's `softmax(s/T)` over the survivors — the proposal
    /// distribution the p-min gate reads. The pick can never empty:
    /// candidate 0 is the row's logits top-1 and the mask always keeps the
    /// first survivor (the sole exception is a non-finite top-1 — the mask
    /// drops `+inf` while `top_k_desc` ranks it — where the walk falls back
    /// to candidate 0 with confidence 0.0, which the p-min gate then cuts;
    /// real model rows are finite). The untruncated posture (`None`/`None`)
    /// delegates bit-identically to [`Self::lattice_walk_keyed`], and
    /// `temperature <= 0` delegates to the greedy walk like every sampled
    /// arm (greedy never leaves the top-`selector_top_k`, so masking has no
    /// greedy meaning).
    ///
    /// Cost note: one full-vocab sort per row beyond `walk_rows`' selection
    /// — the arm's measurement posture accepts it; a rank-only keep-count
    /// helper can replace it if production adopts the mask (the two orders
    /// agree on finite rows — same ties-earlier comparator).
    pub fn lattice_walk_keyed_masked(
        &self,
        anchor_token: u32,
        hidden_rows: &[f32],
        logits: &[f32],
        seed: u64,
        anchor_pos: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> DraftWalk {
        if temperature <= 0.0 {
            return self.lattice_walk(anchor_token, hidden_rows, logits);
        }
        if top_k.is_none() && top_p.is_none() {
            return self.lattice_walk_keyed(
                anchor_token,
                hidden_rows,
                logits,
                seed,
                anchor_pos,
                temperature,
            );
        }
        let n_out = self.cfg.block_size - 1;
        let vocab = self.cfg.vocab_size;
        debug_assert_eq!(logits.len(), n_out * vocab);
        let mut keep = Vec::with_capacity(n_out * vocab);
        for i in 0..n_out {
            keep.extend(truncation_keep_mask(
                &logits[i * vocab..(i + 1) * vocab],
                top_k,
                top_p,
            ));
        }
        self.lattice_walk_keyed_impl(
            anchor_token,
            hidden_rows,
            logits,
            seed,
            anchor_pos,
            temperature,
            Some(&keep),
        )
    }

    /// The keyed walk's shared body: `keep == None` is the untruncated
    /// [`Self::lattice_walk_keyed`]; `Some(flat [n_out * vocab])` filters
    /// both the pick and the confidence's softmax to the surviving
    /// candidates (the masked arm's contract above). The unmasked path's
    /// arithmetic is BIT-IDENTICAL to the pre-extraction walk: same
    /// candidate order, same comparison chain, same max/sum accumulation
    /// order.
    fn lattice_walk_keyed_impl(
        &self,
        anchor_token: u32,
        hidden_rows: &[f32],
        logits: &[f32],
        seed: u64,
        anchor_pos: usize,
        temperature: f32,
        keep: Option<&[bool]>,
    ) -> DraftWalk {
        if temperature <= 0.0 {
            return self.lattice_walk(anchor_token, hidden_rows, logits);
        }
        let (candidates, unary, logits_top1, codes) = self.walk_rows(hidden_rows, logits);
        let vocab = self.cfg.vocab_size;

        let mut chain: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        let mut walk_scores: Vec<Vec<f32>> = Vec::with_capacity(candidates.len());
        let mut pred_tok = anchor_token;
        for i in 0..candidates.len() {
            let scores = self.chain_step_scores(pred_tok, &candidates[i], &unary[i], &codes[i]);
            // The keyed pick: temperature-scaled scores + the shared keyed
            // stream (candidate TOKEN keys the noise, never the candidate
            // index — drafter and target must agree on the same token's
            // noise value at the same position). The masked arm consults
            // survivor keys only (mask → scale → keyed argmax, the
            // Phase-2 order).
            let position = (anchor_pos + 1 + i) as u64;
            let row_keep = keep.map(|k| &k[i * vocab..(i + 1) * vocab]);
            let survives = |ki: usize| -> bool {
                row_keep.is_none_or(|rk| rk[candidates[i][ki] as usize])
            };
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            let mut any = false;
            for ki in 0..candidates[i].len() {
                if !survives(ki) {
                    continue;
                }
                let s = scores[ki] / temperature
                    + keyed_gumbel_noise(seed, position, candidates[i][ki]);
                if !any || s > best_s {
                    best_s = s;
                    best = ki;
                    any = true;
                }
            }
            // The chosen candidate's softmax(s/T) probability over the
            // PROPOSAL distribution — survivors only when masked (the
            // unmasked path folds every candidate, bit-identical to the
            // pre-extraction walk). `DraftWalk::confidence` remains the
            // UNMASKED readback — the masked arm's chain confidence is
            // deliberately the gated distribution's.
            let mut max = f32::NEG_INFINITY;
            for (ki, &s) in scores.iter().enumerate() {
                if survives(ki) {
                    max = max.max(s);
                }
            }
            let mut sum = 0.0f32;
            for (ki, &s) in scores.iter().enumerate() {
                if survives(ki) {
                    sum += ((s - max) / temperature).exp();
                }
            }
            let (tok, conf) = if any {
                (
                    candidates[i][best],
                    ((scores[best] - max) / temperature).exp() / sum,
                )
            } else {
                // Unreachable on finite rows (candidate 0 = the logits
                // top-1 always survives); the non-finite-top-1 fallback is
                // confidence 0.0 so the p-min gate cuts the chain here.
                (candidates[i][0], 0.0)
            };
            chain.push((tok, conf));
            walk_scores.push(scores);
            pred_tok = tok;
        }

        DraftWalk {
            chain,
            logits_top1,
            candidates,
            scores: walk_scores,
        }
    }

    /// The INDEPENDENT-sampled chain walk — the incumbent llama.cpp posture
    /// (the PR's sampled branch: each draft token is a fresh categorical
    /// draw from `softmax(scores/T)` over the row's candidates, random
    /// private to the drafter, sharing NOTHING with the target's sampling
    /// stream). Plan-614 Phase-1 box-4's control arm: against a keyed-
    /// sampled target stream its per-position hit chance is the plain
    /// `sum p_target·p_drafter` collision mass — the coupling lower bound
    /// `(1−TV)/(1+TV)` the keyed arm rides is exactly what it lacks.
    ///
    /// RNG: a SplitMix64 finalizer over `(rng_seed, step, draw_index)` —
    /// the keyed noise's own finalizer shape (full avalanche, open-interval
    /// u via 53 bits), so the arm is deterministic under a seed yet has no
    /// positional agreement with the target's keyed stream even at the
    /// same seed: the keys differ by construction (no target seed/position
    /// enters). `temperature <= 0` delegates to the greedy walk
    /// (bit-identical to [`Self::lattice_walk`]).
    pub fn lattice_walk_independent(
        &self,
        anchor_token: u32,
        hidden_rows: &[f32],
        logits: &[f32],
        rng_seed: u64,
        temperature: f32,
    ) -> DraftWalk {
        if temperature <= 0.0 {
            return self.lattice_walk(anchor_token, hidden_rows, logits);
        }
        #[inline(always)]
        fn fin(z: u64) -> u64 {
            let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        // One uniform per draw, advanced within the step: u(open (0,1))
        // from 53 bits — the keyed construction without a key.
        let draw = |step: u64, k: u64| -> f32 {
            let z = fin(fin(rng_seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15)) ^ k);
            let inv = 1.0f64 / (1u64 << 53) as f64;
            (((z >> 11) as f64) * inv + inv * 0.5) as f32
        };

        let (candidates, unary, logits_top1, codes) = self.walk_rows(hidden_rows, logits);
        let mut chain: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        let mut walk_scores: Vec<Vec<f32>> = Vec::with_capacity(candidates.len());
        let mut pred_tok = anchor_token;
        for i in 0..candidates.len() {
            let scores = self.chain_step_scores(pred_tok, &candidates[i], &unary[i], &codes[i]);
            // Categorical pick over softmax(scores/T) via one uniform on
            // the cumulative distribution (candidate order fixed = the
            // top-k-desc order; f32 accumulation is deterministic).
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let mut probs = vec![0.0f32; scores.len()];
            for (pi, &s) in scores.iter().enumerate() {
                probs[pi] = ((s - max) / temperature).exp();
                sum += probs[pi];
            }
            let u = draw(i as u64, 0) * sum;
            let mut acc = 0.0f32;
            let mut best = scores.len() - 1;
            for (pi, &w) in probs.iter().enumerate() {
                acc += w;
                if u < acc {
                    best = pi;
                    break;
                }
            }
            let conf = probs[best] / sum;
            chain.push((candidates[i][best], conf));
            walk_scores.push(scores);
            pred_tok = candidates[i][best];
        }

        DraftWalk {
            chain,
            logits_top1,
            candidates,
            scores: walk_scores,
        }
    }

    /// The one-shot composition (the future integrated loop's entry): the
    /// two phases around a caller-supplied lm_head.
    #[allow(clippy::type_complexity)]
    pub fn draft_block(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
        lm_head_rows: &dyn Fn(&[f32], usize) -> Result<Vec<f32>, String>,
    ) -> Result<DraftWalk, String> {
        let (hidden, logits) = self.draft_block_rows(anchor_token, anchor_pos, embed_row, lm_head_rows)?;
        Ok(self.lattice_walk(anchor_token, &hidden, &logits))
    }

    /// The keyed-sampled composition — [`Self::draft_block`] under the
    /// exact-sampling posture: the chain walk keyed by
    /// `(seed, absolute position, candidate token)` at the deployment
    /// `temperature` (see [`Self::lattice_walk_keyed`]). This is the entry
    /// point the verify loop calls when the target decodes with the keyed
    /// sampler — the drafted block is then proposed in the target's own
    /// sampling stream. `temperature <= 0` is the greedy posture,
    /// bit-identical to [`Self::draft_block`].
    #[allow(clippy::type_complexity)]
    pub fn draft_block_keyed(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
        lm_head_rows: &dyn Fn(&[f32], usize) -> Result<Vec<f32>, String>,
        seed: u64,
        temperature: f32,
    ) -> Result<DraftWalk, String> {
        let (hidden, logits) = self.draft_block_rows(anchor_token, anchor_pos, embed_row, lm_head_rows)?;
        Ok(self.lattice_walk_keyed(
            anchor_token,
            &hidden,
            &logits,
            seed,
            anchor_pos,
            temperature,
        ))
    }

    /// The masked keyed composition — [`Self::draft_block_keyed`] plus the
    /// drafter's own-nucleus truncation ([`Self::lattice_walk_keyed_masked`]).
    /// `None`/`None` truncation delegates bit-identically to
    /// [`Self::draft_block_keyed`].
    #[allow(clippy::type_complexity)]
    pub fn draft_block_keyed_masked(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
        lm_head_rows: &dyn Fn(&[f32], usize) -> Result<Vec<f32>, String>,
        seed: u64,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<DraftWalk, String> {
        let (hidden, logits) = self.draft_block_rows(anchor_token, anchor_pos, embed_row, lm_head_rows)?;
        Ok(self.lattice_walk_keyed_masked(
            anchor_token,
            &hidden,
            &logits,
            seed,
            anchor_pos,
            temperature,
            top_k,
            top_p,
        ))
    }

    /// The two phases' shared body: the noise-block forward + the MASK-row
    /// extraction + the target lm_head (moved verbatim out of
    /// `draft_block`). Returns `(hidden_rows, logits)`.
    #[allow(clippy::type_complexity)]
    fn draft_block_rows(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
        lm_head_rows: &dyn Fn(&[f32], usize) -> Result<Vec<f32>, String>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let bs = self.cfg.block_size;
        let e = self.cfg.n_embd;
        let hidden = self.draft_block_hidden(anchor_token, anchor_pos, embed_row);
        let n_out = bs - 1;
        let mut rows = vec![0.0f32; n_out * e];
        for i in 1..bs {
            rows[(i - 1) * e..i * e].copy_from_slice(&hidden[i * e..(i + 1) * e]);
        }
        let logits = lm_head_rows(&rows, n_out)?;
        Ok((hidden, logits))
    }
}
/// The DFlash2 dynamic conv (`build_dflash2_conv`): per block token `i`,
/// channel `c` (group `g = c/16`):
/// `out[i][c] = Σ_t (dyn[i][g + n_groups·(t + kernel·side)] + base[side·(kernel·e) + t·e + c]) · src[i−t][c]`
/// with `src[<0] = 0` (zero-padded block start).
/// Issue 989 — the intra-block attention mask contract. The trained
/// semantics are **NON-CAUSAL**: the z-lab checkpoint's own `config.json`
/// carries `"is_causal": false`, and BOTH reference implementations honor
/// it (torch `model.py` reads `config.is_causal` FIRST — the
/// `layer_type == "sliding_attention"` fallback only fires when the field
/// is ABSENT; MLX `model_mlx.py` `is_causal=cfg.get("is_causal")` → False →
/// `create_causal_mask` never applied, `block = key >= ctx_len` with no
/// `key <= query` term). The llama.cpp fork's "cache-aware, non-causal
/// attention" comment (prismml dflash.cpp) matches. A causal reading of
/// the torch fallback branch was implemented, A/B'd (2.648 → 2.612 — no
/// acceptance gain), and REFUTED by the config field; the knob is retained
/// as the falsification record. Default OFF (non-causal, the trained
/// semantics); `QWEN38_DFLASH2_CAUSAL=1` enables the causal mask (the
/// refuted-hypothesis arm).
pub fn dflash2_causal_block() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("QWEN38_DFLASH2_CAUSAL")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

fn apply_conv(
    dyn_coeff: &[f32],
    base: &[f32],
    side: usize,
    bs: usize,
    e: usize,
    cfg: &DFlash2Config,
    src: &[f32],
    dst: &mut [f32],
) {
    let gsz = cfg.conv_group;
    let n_groups = e / gsz;
    let kernel = cfg.conv_kernel;
    let stride = 2 * kernel * n_groups;
    for i in 0..bs {
        let dseg = &dyn_coeff[i * stride..(i + 1) * stride];
        for c in 0..e {
            let g = c / gsz;
            let mut acc = 0.0f32;
            for t in 0..kernel {
                if i >= t {
                    let di = g + n_groups * (t + kernel * side);
                    let bi = side * (kernel * e) + t * e + c;
                    acc += (dseg[di] + base[bi]) * src[(i - t) * e + c];
                }
            }
            dst[i * e + c] = acc;
        }
    }
}

/// RoPE NEOX (half-split pairing — `LLM_ARCH_DFLASH` returns NEOX for the
/// non-DSV4 backbones per llama.cpp's `llama_model_rope_type`): pair
/// `(i, i + d/2)` rotates by `angle = pos · freq[i]`, `i < d/2`.
fn apply_rope_in_place(v: &mut [f32], pos: usize, freqs: &[f32]) {
    let np = freqs.len();
    for i in 0..np {
        let ang = pos as f32 * freqs[i];
        let (s, c) = (ang.sin(), ang.cos());
        let a = v[i];
        let b = v[i + np];
        v[i] = a * c - b * s;
        v[i + np] = a * s + b * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::keyed_gumbel_max_sample;

    #[test]
    fn rope_identity_at_zero_and_pairwise_rotation() {
        // NEOX half-split: 4 dims, 2 pairs — (0,2) and (1,3).
        let freqs = vec![1.0f32, 0.5];
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        apply_rope_in_place(&mut v, 0, &freqs);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
        // pos=1: pair (0,2) rotates by 1 rad (freq[0] = theta^0 = 1).
        let mut v2 = vec![1.0f32, 0.0, 0.0, 9.0];
        apply_rope_in_place(&mut v2, 1, &freqs);
        assert!((v2[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((v2[2] - 1.0f32.sin()).abs() < 1e-6);
        // pair (1,3) rotates by 0.5 rad: a=0, b=9 → v1 = -9·sin, v3 = 9·cos.
        assert!((v2[1] - (-9.0 * 0.5f32.sin())).abs() < 1e-6);
        assert!((v2[3] - 9.0 * 0.5f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_matches_reference_formula() {
        let x = [3.0f32, -4.0, 5.0, 12.0];
        let gamma = [1.0f32, 0.5, 2.0, 1.0];
        let mut out = [0.0f32; 4];
        rmsnorm_into(&x, &gamma, 1e-6, &mut out);
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let inv = 1.0 / (ss + 1e-6).sqrt();
        for i in 0..4 {
            assert!((out[i] - x[i] * inv * gamma[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn top_k_desc_orders_and_ties() {
        let vals = vec![1.0f32, 5.0, 5.0, 0.5, 3.0];
        let idx = top_k_desc(&vals, 3);
        assert_eq!(idx.len(), 3);
        // ties: earlier index wins → 5.0 at idx 1 before idx 2.
        assert_eq!(idx[0], 1);
        assert_eq!(idx[1], 2);
        assert_eq!(idx[2], 4);
    }

    #[test]
    fn conv_zero_pad_and_tap_semantics() {
        // kernel=2, group=16 → n_groups=2 at e=32.
        let cfg = DFlash2Config {
            n_embd: 32,
            conv_group: 16,
            conv_kernel: 2,
            ..Default::default()
        };
        let bs = 4;
        let e = 32;
        // dyn all zero; base side 0 tap 0 = 1, tap 1 = 0 → identity on tap 0.
        let mut base = vec![0.0f32; 2 * 2 * e];
        for b in base.iter_mut().take(e) {
            *b = 1.0;
        }
        let dyn_coeff = vec![0.0f32; bs * (2 * 2 * (e / 16))];
        let src: Vec<f32> = (0..bs * e).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; bs * e];
        apply_conv(&dyn_coeff, &base, 0, bs, e, &cfg, &src, &mut dst);
        assert_eq!(dst, src);
        // tap 1 = 1 instead → out[i] = src[i] + src[i-1] (0 at i=0).
        let mut base2 = vec![0.0f32; 2 * 2 * e];
        for b in base2.iter_mut().take(2 * e) {
            *b = 1.0;
        }
        let mut dst2 = vec![0.0f32; bs * e];
        apply_conv(&dyn_coeff, &base2, 0, bs, e, &cfg, &src, &mut dst2);
        for i in 0..bs {
            for c in 0..e {
                let want = src[i * e + c] + if i >= 1 { src[(i - 1) * e + c] } else { 0.0 };
                assert_eq!(dst2[i * e + c], want);
            }
        }
    }

    /// `out×in` identity matrix (row-major) — the crafted-weight tests'
    /// projection helper.
    fn identity_m(out: usize, inn: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; out * inn];
        for i in 0..out.min(inn) {
            m[i * inn + i] = 1.0;
        }
        m
    }

    #[test]
    fn causal_block_attention_pins_mask_row_visibility() {
        // Issue 989 — the TRAINED semantics are NON-CAUSAL: the z-lab
        // checkpoint's config.json carries "is_causal": false (honored by
        // both reference impls — the torch fallback branch that reads
        // causal from layer_types never fires — and by the llama.cpp
        // fork). This pins it end-to-end through `draft_block_hidden`
        // with crafted weights: q_proj = 0 makes every dot 0 → uniform
        // attention over the VISIBLE keys — non-causally ALL block keys —
        // so EVERY row's attention output is the mean of all 4 v's. (The
        // causal arm, `QWEN38_DFLASH2_CAUSAL=1`, would give row i the
        // mean over rows 0..=i — the falsified reading this test guards
        // against regressing INTO silently.)
        let cfg = DFlash2Config {
            n_layer: 1,
            n_embd: 4,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 2,
            n_ff: 4,
            block_size: 4,
            conv_kernel: 1,
            conv_group: 2,
            selector_rank: 2,
            selector_top_k: 2,
            rope_theta: 1.0e4,
            sliding_window: 8,
            rms_eps: 1.0e-6,
            vocab_size: 32,
            mask_token_id: 1,
        };
        let e = cfg.n_embd;
        let n_kv = cfg.n_kv_head * cfg.head_dim;
        let n_q = cfg.n_head * cfg.head_dim;
        // attn_conv: identity on tap 0 both sides (kernel=1 → base is
        // [side][channel]); everything else zero → hc = h, conv side-1 = id.
        let mut layer = DFlash2Layer::default();
        layer.attn_norm = vec![1.0; e];
        layer.ffn_norm = vec![1.0; e];
        layer.q_norm = vec![1.0; cfg.head_dim];
        layer.k_norm = vec![1.0; cfg.head_dim];
        layer.attn_conv_base = {
            let mut b = vec![0.0f32; 2 * cfg.conv_kernel * e];
            for (i, v) in b.iter_mut().enumerate() {
                if i < e || (cfg.conv_kernel * e..cfg.conv_kernel * e + e).contains(&i) {
                    *v = 1.0;
                }
            }
            b
        };
        layer.ffn_conv_base = layer.attn_conv_base.clone();
        // k_proj / v_proj / o_proj: identity (v_j = hc_j[..2]; both heads
        // share the kv head → identical per-head outputs).
        layer.k_proj = identity_m(n_kv, e);
        layer.v_proj = identity_m(n_kv, e);
        layer.o_proj = identity_m(e, n_q);
        // q_proj ZERO → q = 0 → uniform attention over visible keys.
        layer.q_proj = vec![0.0; n_q * e];
        // dynamic conv coefficients: conv_proj = 0 → dyn = 0 (identity base
        // only). Size = projected × e.
        let projected = 2 * cfg.conv_kernel * (e / cfg.conv_group);
        layer.attn_conv_proj = vec![0.0; projected * e];
        layer.ffn_conv_proj = vec![0.0; projected * e];
        // FFN zero → mid = silu(0)·0 = 0 → down = 0.
        layer.gate_proj = vec![0.0; cfg.n_ff * e];
        layer.up_proj = vec![0.0; cfg.n_ff * e];
        layer.down_proj = vec![0.0; e * cfg.n_ff];
        let w = DFlash2Weights {
            fc: vec![0.0; e * 5 * e],
            hidden_norm: vec![1.0; e],
            output_norm: vec![1.0; e],
            sel_hidden: vec![0.0; cfg.selector_rank * e],
            sel_prev: vec![0.0; cfg.vocab_size * cfg.selector_rank],
            sel_next: vec![0.0; cfg.vocab_size * cfg.selector_rank],
            layers: vec![layer],
        };
        let drafter = DFlash2Drafter::new(cfg.clone(), w);
        let anchor: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let mask: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 {
                anchor.clone()
            } else {
                mask.clone()
            }
        };
        let t = drafter.draft_block_hidden(0, 0, &embed);

        // Expected under NON-CAUSAL (the trained semantics): every row's
        // attention = mean(v_a, v_m, v_m, v_m); then + x_i (residual) →
        // rmsnorm.
        let rms = |x: &[f32]| -> Vec<f32> {
            let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (ss + cfg.rms_eps).sqrt();
            x.iter().map(|v| v * inv).collect::<Vec<_>>()
        };
        let h_a = rms(&anchor);
        let h_m = rms(&mask);
        let v_a = [h_a[0], h_a[1]];
        let v_m = [h_m[0], h_m[1]];
        let mean0 = (v_a[0] + 3.0 * v_m[0]) / 4.0;
        let mean1 = (v_a[1] + 3.0 * v_m[1]) / 4.0;
        for i in 0..cfg.block_size {
            // o_proj identity over [head0_v, head1_v] (same v twice).
            let ao = [mean0, mean1, mean0, mean1];
            let x_i = if i == 0 { &anchor } else { &mask };
            let mut pre = vec![0.0f32; e];
            for c in 0..e {
                pre[c] = ao[c] + x_i[c];
            }
            let want = rms(&pre);
            for c in 0..e {
                let diff = (t[i * e + c] - want[c]).abs();
                assert!(
                    diff < 1e-5,
                    "row {i} dim {c}: got {} want {} (non-causal trained mask violated?)",
                    t[i * e + c],
                    want[c]
                );
            }
        }
    }

    #[test]
    fn walk_confidence_temperature_contract() {
        // Hand-built walk: one position, candidates [10, 11, 12] with
        // selector scores [2, 0, -1] — the argmax (10) is unique.
        let mut w = DraftWalk::default();
        w.candidates.push(vec![10, 11, 12]);
        w.scores.push(vec![2.0, 0.0, -1.0]);
        w.chain
            .push((10, 1.0 / (1.0 + (-2.0f32).exp() + (-3.0f32).exp())));
        // Greedy (T <= 0) == the chain confidence == softmax at T = 1.
        let g = w.confidence(0, 0.0);
        assert!((g - w.chain[0].1).abs() < 1e-6);
        assert!((w.confidence(0, 1.0) - g).abs() < 1e-6);
        // Sampled branch: T < 1 sharpens the argmax's probability, T > 1
        // flattens it (the incumbent sweep ran T = 0.6).
        assert!(w.confidence(0, 0.6) > g);
        assert!(w.confidence(0, 2.0) < g);
        // Hand-computed T = 0.6 value (the PR's exp((s − max)/T) form).
        let want06 = 1.0f32 / (1.0 + (-2.0f32 / 0.6).exp() + (-3.0f32 / 0.6).exp());
        assert!((w.confidence(0, 0.6) - want06).abs() < 1e-6);
        // Non-argmax chosen token (the sampled branch can pick it): its
        // OWN probability — exercises the candidate-index lookup.
        w.chain[0] = (11, 0.0);
        let c11 = w.confidence(0, 1.0);
        let want11 = (-2.0f32).exp() / (1.0 + (-2.0f32).exp() + (-3.0f32).exp());
        assert!((c11 - want11).abs() < 1e-6);
        assert!(c11 < g);
        // Bounds + finiteness across regimes.
        w.chain[0] = (10, g);
        for t in [0.0f32, 0.6, 1.0, 2.0] {
            let c = w.confidence(0, t);
            assert!(c.is_finite() && c > 0.0 && c <= 1.0 + 1e-6);
        }
    }

    /// The synthetic drafter for the keyed-walk tests: the causal fixture's
    /// all-zero SELECTOR weights (sel_hidden/sel_prev/sel_next = 0), so the
    /// chain scores collapse to the candidates' own lm_head logits (`unary`)
    /// — full control of the walk from the crafted logits alone. Zero
    /// weights are legal here because the walk never reads missing weights.
    fn zero_selector_drafter() -> DFlash2Drafter {
        let cfg = DFlash2Config {
            n_layer: 1,
            n_embd: 4,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 2,
            n_ff: 4,
            block_size: 4,
            conv_kernel: 1,
            conv_group: 2,
            selector_rank: 2,
            selector_top_k: 2,
            rope_theta: 1.0e4,
            sliding_window: 8,
            rms_eps: 1.0e-6,
            vocab_size: 32,
            mask_token_id: 1,
        };
        let e = cfg.n_embd;
        let n_kv = cfg.n_kv_head * cfg.head_dim;
        let n_q = cfg.n_head * cfg.head_dim;
        let mut layer = DFlash2Layer::default();
        layer.attn_norm = vec![1.0; e];
        layer.ffn_norm = vec![1.0; e];
        layer.q_norm = vec![1.0; cfg.head_dim];
        layer.k_norm = vec![1.0; cfg.head_dim];
        layer.attn_conv_base = {
            let mut b = vec![0.0f32; 2 * cfg.conv_kernel * e];
            for (i, v) in b.iter_mut().enumerate() {
                if i < e || (cfg.conv_kernel * e..cfg.conv_kernel * e + e).contains(&i) {
                    *v = 1.0;
                }
            }
            b
        };
        layer.ffn_conv_base = layer.attn_conv_base.clone();
        layer.k_proj = identity_m(n_kv, e);
        layer.v_proj = identity_m(n_kv, e);
        layer.o_proj = identity_m(e, n_q);
        layer.q_proj = vec![0.0; n_q * e];
        let projected = 2 * cfg.conv_kernel * (e / cfg.conv_group);
        layer.attn_conv_proj = vec![0.0; projected * e];
        layer.ffn_conv_proj = vec![0.0; projected * e];
        layer.gate_proj = vec![0.0; cfg.n_ff * e];
        layer.up_proj = vec![0.0; cfg.n_ff * e];
        layer.down_proj = vec![0.0; e * cfg.n_ff];
        let w = DFlash2Weights {
            fc: vec![0.0; e * 5 * e],
            hidden_norm: vec![1.0; e],
            output_norm: vec![1.0; e],
            sel_hidden: vec![0.0; cfg.selector_rank * e],
            sel_prev: vec![0.0; cfg.vocab_size * cfg.selector_rank],
            sel_next: vec![0.0; cfg.vocab_size * cfg.selector_rank],
            layers: vec![layer],
        };
        DFlash2Drafter::new(cfg, w)
    }

    /// Three lm_head rows (block_size 4 → n_out 3) over vocab 32, top-2
    /// candidates each, with a TINY gap on row 2 (0.5 vs 0.49) so the keyed
    /// noise flips that pick at a measurable seed fraction.
    fn crafted_logits() -> Vec<f32> {
        let v = 32usize;
        let mut lg = vec![0.0f32; 3 * v];
        let mut set = |row: usize, a: u32, va: f32, b: u32, vb: f32| {
            lg[row * v + a as usize] = va;
            lg[row * v + b as usize] = vb;
        };
        set(0, 5, 2.0, 9, 1.9);
        set(1, 3, 1.0, 7, 0.9);
        set(2, 1, 0.5, 2, 0.49);
        lg
    }

    #[test]
    fn keyed_walk_greedy_posture_is_bit_identical_to_the_greedy_walk() {
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        let greedy = drafter.draft_block(0, 100, &embed, &lm_head).expect("draft");
        for t in [0.0f32, -1.0] {
            let keyed = drafter
                .draft_block_keyed(0, 100, &embed, &lm_head, 7, t)
                .expect("draft keyed");
            assert_eq!(keyed.chain, greedy.chain, "T={t} must delegate to greedy");
            assert_eq!(keyed.scores, greedy.scores);
            assert_eq!(keyed.candidates, greedy.candidates);
            assert_eq!(keyed.logits_top1, greedy.logits_top1);
        }
    }

    #[test]
    fn keyed_walk_matches_the_keyed_argmax_composition() {
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        let (seed, t) = (0xD1A_5EED_u64, 0.6f32);
        let walk = drafter
            .draft_block_keyed(0, 100, &embed, &lm_head, seed, t)
            .expect("draft keyed");
        // Independent recomposition of the contract: with the zero selector,
        // scores == the candidates' logits; the pick is the keyed-Gumbel
        // argmax over the candidate TOKENS (not indices) at absolute
        // positions 101..103; confidence is the chosen's softmax(s/T).
        let cands = [
            vec![5u32, 9],
            vec![3, 7],
            vec![1, 2],
        ];
        let rows = [
            [2.0f32, 1.9],
            [1.0, 0.9],
            [0.5, 0.49],
        ];
        for i in 0..3 {
            let scores = rows[i]; // zero selector → unary only (sel_prev is
            // zero, so the chain step's dot term is 0 for ANY pred — the
            // recomposition legitimately ignores the chain history)
            let position = (100 + 1 + i) as u64;
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for (k, &tok) in cands[i].iter().enumerate() {
                let s = scores[k] / t + keyed_gumbel_noise(seed, position, tok);
                if s > best_s {
                    best_s = s;
                    best = k;
                }
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|&s| ((s - max) / t).exp()).sum();
            let conf = ((scores[best] - max) / t).exp() / sum;
            assert_eq!(walk.chain[i].0, cands[i][best], "row {i} keyed pick");
            assert!((walk.chain[i].1 - conf).abs() < 1e-6, "row {i} confidence");
            assert_eq!(walk.scores[i], scores.to_vec(), "row {i} scores");
            assert_eq!(walk.candidates[i], cands[i], "row {i} candidates");
        }
    }

    #[test]
    fn keyed_walk_matches_the_target_keyed_sampler_under_a_masked_shared_distribution() {
        // The Phase-2 target half's contract, pinned from the drafter side:
        // when the target's full-vocab logits equal the drafter's candidate
        // scores everywhere the drafter can propose (every other token at
        // −inf — the truncation mask), the keyed walk's pick at each block
        // row must equal `keyed_gumbel_max_sample(logits/T, seed, position)`
        // at that row's produced-token position — 100% match, by
        // construction IDENTICAL computation. This fails if anyone moves
        // the drafter's position convention off `anchor_pos + 1 + i` (the
        // produced token's position, never the input position whose logits
        // produced it): an off-by-one keeps the output lossless but
        // silently decouples the shared stream, and the only symptom would
        // be an acceptance drop no correctness test catches.
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        let cands = [vec![5u32, 9], vec![3, 7], vec![1, 2]];
        let rows = [[2.0f32, 1.9], [1.0, 0.9], [0.5, 0.49]];
        let t = 0.6f32;
        // 64 seeds, not one: a single seed can coincide across ±1 position
        // offsets (measured: one seed misses a −1 mutant 12.3% of the time;
        // the planted-mutation canary in the landing commit saw seed
        // 0xFEED_5EED give identical picks at all three offsets). A 64-seed
        // sweep misses the mutant with probability ~0.123⁶⁴ — effectively 0.
        for seed in 0..64u64 {
            let walk = drafter
                .draft_block_keyed(0, 100, &embed, &lm_head, seed, t)
                .expect("draft keyed");
            for i in 0..3 {
                let position = (100 + 1 + i) as u64;
                // The masked target distribution: −inf everywhere except
                // the drafter's own candidates at their scores.
                let mut full = [f32::NEG_INFINITY; 32];
                for (k, &tok) in cands[i].iter().enumerate() {
                    full[tok as usize] = rows[i][k];
                }
                let scaled: Vec<f32> = full.iter().map(|&l| l / t).collect();
                let target_tok = keyed_gumbel_max_sample(&scaled, seed, position);
                assert_eq!(
                    walk.chain[i].0, target_tok,
                    "seed {seed} row {i}: drafter pick diverged from the target \
                     keyed sampler — key convention or arithmetic drifted"
                );
            }
        }
    }

    #[test]
    fn keyed_walk_is_deterministic_and_actually_samples() {
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        let a = drafter
            .draft_block_keyed(0, 100, &embed, &lm_head, 42, 0.6)
            .expect("a");
        let b = drafter
            .draft_block_keyed(0, 100, &embed, &lm_head, 42, 0.6)
            .expect("b");
        assert_eq!(a.chain, b.chain, "same seed must reproduce the walk");
        // It ACTUALLY samples: with the 0.1-gap row 0 (2.0 vs 1.9 at T=0.6)
        // and the 0.01-gap row 2, the keyed noise flips picks for a large
        // fraction of seeds. The greedy pick at every row is candidates[0].
        let greedy: Vec<u32> = vec![5, 3, 1];
        let mut flipped = 0u32;
        for seed in 0..200u64 {
            let w = drafter
                .draft_block_keyed(0, 100, &embed, &lm_head, seed, 0.6)
                .expect("w");
            if w.chain.iter().zip(&greedy).any(|(c, g)| c.0 != *g) {
                flipped += 1;
            }
        }
        assert!(
            flipped > 40,
            "keyed walk should diverge from greedy on a large seed fraction: {flipped}/200"
        );
    }

    #[test]
    fn masked_keyed_walk_none_truncation_is_bit_identical_to_the_keyed_walk() {
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        for seed in 0..32u64 {
            let keyed = drafter
                .draft_block_keyed(0, 100, &embed, &lm_head, seed, 0.6)
                .expect("keyed");
            let masked = drafter
                .draft_block_keyed_masked(0, 100, &embed, &lm_head, seed, 0.6, None, None)
                .expect("masked");
            assert_eq!(masked.chain, keyed.chain, "seed {seed}: None/None must delegate");
            assert_eq!(masked.scores, keyed.scores);
            assert_eq!(masked.candidates, keyed.candidates);
            assert_eq!(masked.logits_top1, keyed.logits_top1);
        }
    }

    #[test]
    fn masked_keyed_walk_never_picks_outside_the_survivor_nucleus() {
        // top_k = Some(1) keeps the row's logits top-1 ONLY (the mask's
        // first-survivor rule) — so every seed's pick on every row must be
        // candidates[0], i.e. exactly the untruncated walk's logits_top1,
        // and the single-survivor softmax is confidence 1.0. This pins the
        // FILTER (a drifted implementation that ignores the mask keeps
        // sampling the second candidate on flip-prone seeds — the crafted
        // rows flip on >20% of seeds per the determinism test above).
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        let untrunc_top1 = vec![5u32, 3, 1]; // candidates[0] per row
        let mut conf_seen = 0usize;
        for seed in 0..200u64 {
            let w = drafter
                .draft_block_keyed_masked(0, 100, &embed, &lm_head, seed, 0.6, Some(1), None)
                .expect("masked");
            let picks: Vec<u32> = w.chain.iter().map(|&(t, _)| t).collect();
            assert_eq!(picks, untrunc_top1, "seed {seed}: pick outside the k=1 nucleus");
            for &(_, c) in &w.chain {
                assert!((c - 1.0).abs() < 1e-6, "single-survivor confidence must be 1.0");
                conf_seen += 1;
            }
            // The walk still records the FULL candidate set + scores
            // (diagnostics, not filtered).
            assert_eq!(w.candidates, [vec![5u32, 9], vec![3, 7], vec![1, 2]]);
        }
        assert_eq!(conf_seen, 600);
    }

    #[test]
    fn masked_keyed_walk_top_p_survivor_mass_cuts_the_candidate_set() {
        // The arm's real bite: top-p OVER THE SURVIVORS can keep FEWER
        // candidates than the walk proposes — row 0 of the crafted logits
        // (2.0 vs 1.9, rest 0.0) puts only ~16.8% of the finite mass on
        // token 5, so top_p = 0.1 keeps ONLY token 5 and that row's pick
        // must equal it for every seed, while the UNMASKED keyed walk
        // flips row 0 on a large seed fraction (the same-seed disagreement
        // count is the mask actually firing, not a delegation accident;
        // rows 1–2 keep both candidates — their share thresholds are
        // higher — and stay unasserted).
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        let mut disagreements = 0u32;
        for seed in 0..200u64 {
            let masked = drafter
                .draft_block_keyed_masked(0, 100, &embed, &lm_head, seed, 0.6, None, Some(0.1))
                .expect("masked");
            assert_eq!(masked.chain[0].0, 5, "seed {seed}: row 0 must keep only its top-1");
            assert!(
                (masked.chain[0].1 - 1.0).abs() < 1e-6,
                "single-survivor row confidence must be 1.0"
            );
            let keyed = drafter
                .draft_block_keyed(0, 100, &embed, &lm_head, seed, 0.6)
                .expect("keyed");
            if keyed.chain[0].0 != 5 {
                disagreements += 1;
            }
        }
        assert!(
            disagreements > 20,
            "the mask must actually change picks on a real seed fraction: {disagreements}/200"
        );
    }

    #[test]
    fn independent_walk_greedy_posture_is_bit_identical_to_the_greedy_walk() {
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let lg = crafted_logits();
        let lm_head = |_rows: &[f32], _n: usize| -> Result<Vec<f32>, String> { Ok(lg.clone()) };
        let greedy = drafter.draft_block(0, 100, &embed, &lm_head).expect("draft");
        for t in [0.0f32, -1.0] {
            let ind = drafter
                .lattice_walk_independent(0, &drafter_draft_hidden(&drafter, 0, 100, &embed), &lg, 7, t);
            assert_eq!(ind.chain, greedy.chain, "T={t} must delegate to greedy");
            assert_eq!(ind.scores, greedy.scores);
        }
    }

    /// The zero-selector drafter's noise-block hidden for the walk-only
    /// tests (draft_block_keyed hides this seam; the independent walk takes
    /// hidden rows directly like `lattice_walk` does).
    fn drafter_draft_hidden(
        drafter: &DFlash2Drafter,
        anchor: u32,
        anchor_pos: usize,
        embed: &dyn Fn(u32) -> Vec<f32>,
    ) -> Vec<f32> {
        drafter.draft_block_hidden(anchor, anchor_pos, embed)
    }

    #[test]
    fn independent_walk_recomposes_the_categorical_pick_exactly() {
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let hidden = drafter.draft_block_hidden(0, 100, &embed);
        let lg = crafted_logits();
        let (seed, t) = (0xBEEF_u64, 0.6f32);
        let walk = drafter.lattice_walk_independent(0, &hidden, &lg, seed, t);
        // Independent recomposition: with the zero selector, scores == the
        // candidates' logits; the pick is the categorical draw from
        // softmax(scores/T) via the walk's own uniform stream (the same
        // splitmix finalizer over (seed, step, 0) — duplicated HERE on
        // purpose, the test-side recomposition convention).
        let cands = [vec![5u32, 9], vec![3, 7], vec![1, 2]];
        let rows = [[2.0f32, 1.9], [1.0, 0.9], [0.5, 0.49]];
        let fin = |mut z: u64| -> u64 {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for i in 0..3 {
            let scores = rows[i];
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let probs: Vec<f32> = scores
                .iter()
                .map(|&s| {
                    let w = ((s - max) / t).exp();
                    sum += w;
                    w
                })
                .collect();
            let z = fin(fin(seed ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))) ^ 0);
            let inv = 1.0f64 / (1u64 << 53) as f64;
            let u = (((z >> 11) as f64) * inv + inv * 0.5) as f32;
            let target = u * sum;
            let mut acc = 0.0f32;
            let mut best = scores.len() - 1;
            for (pi, &w) in probs.iter().enumerate() {
                acc += w;
                if target < acc {
                    best = pi;
                    break;
                }
            }
            assert_eq!(walk.chain[i].0, cands[i][best], "row {i} categorical pick");
            assert!(
                (walk.chain[i].1 - probs[best] / sum).abs() < 1e-6,
                "row {i} confidence == the picked softmax prob"
            );
        }
        // Determinism under the seed.
        let again = drafter.lattice_walk_independent(0, &hidden, &lg, seed, t);
        assert_eq!(again.chain, walk.chain);
        // And a different seed is a different stream (the finalizer avalanche).
        assert!(
            drafter
                .lattice_walk_independent(0, &hidden, &lg, seed + 1, t)
                .chain
                .iter()
                .zip(&walk.chain)
                .any(|(a, b)| a.0 != b.0),
            "seed+1 should move at least one pick on these tiny gaps"
        );
    }

    #[test]
    fn independent_walk_actually_samples_not_greedy() {
        let drafter = zero_selector_drafter();
        let anchor = vec![1.0f32, 0.0, 0.0, 0.0];
        let mask = vec![0.0f32, 1.0, 1.0, 1.0];
        let embed = |tok: u32| -> Vec<f32> {
            if tok == 0 { anchor.clone() } else { mask.clone() }
        };
        let hidden = drafter.draft_block_hidden(0, 100, &embed);
        let lg = crafted_logits();
        // Greedy picks candidates[0] at every row. The independent arm at
        // T=0.6 draws from softmax — row 0's p(first) is only ~0.54, so the
        // greedy-token rate across seeds must land far below 100% (a
        // greedy-posture bug pins it at exactly 200/200).
        let mut greedy_first = 0u32;
        for seed in 0..200u64 {
            let w = drafter.lattice_walk_independent(0, &hidden, &lg, seed, 0.6);
            if w.chain[0].0 == 5 {
                greedy_first += 1;
            }
        }
        assert!(
            greedy_first < 160,
            "independent walk collapsed to greedy: {greedy_first}/200 picked the argmax"
        );
        assert!(greedy_first > 40, "sampler degenerate: only {greedy_first}/200 argmax picks");
    }
}
