//! The riir-owned decision head — the candle port (`crate::laya::head`) is
//! the normative math: +`type_emb[qtype`], two pre-norm encoder layers (torch
//! `in_proj` Q|K|V rows, `need_weights` 1/√hd q-scaling, `ReLU` FF), marker
//! gather, scorer LN→Linear→GELU→Linear, act head from the CLS row +
//! [top1, top1−top2, entropy/ln(max(k,2)), k/255], softmax32.
//!
//! ⚠ The head FF activation is `ReLU` (torch's `nn.TransformerEncoderLayer`
//! default), the scorer's own GELU is genuinely gelu — the same split the
//! candle port records.
//!
//! Temperatures/`softmax32` come from the shared substrate
//! ([`super::super::temps`]) — NOT copied.

use std::collections::HashMap;

use super::super::temps::softmax32;
use super::super::{LayaError, Result};
use super::backend::Backend;
use super::weights::Weights;

/// One pre-norm transformer-encoder layer of the head (torch
/// `nn.TransformerEncoderLayer`, `norm_first`, `batch_first`, `ReLU` FF).
struct HeadLayer {
    /// `self_attn.in_proj_weight` [3d, d] — torch layout: Q rows, then K
    /// rows, then V rows.
    in_proj_w: Vec<f32>,
    in_proj_b: Vec<f32>,
    /// `self_attn.out_proj`.
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    n1w: Vec<f32>,
    n1b: Vec<f32>,
    n2w: Vec<f32>,
    n2b: Vec<f32>,
    /// `linear1` [4d, d] (`ReLU` between).
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    /// `linear2` [d, 4d].
    l2w: Vec<f32>,
    l2b: Vec<f32>,
}

/// The decision head (our tensors).
pub struct Head {
    layers: Vec<HeadLayer>,
    /// `type_emb.weight` [3, d].
    type_emb: Vec<f32>,
    /// `scorer.0` `LayerNorm`.
    s0w: Vec<f32>,
    s0b: Vec<f32>,
    /// `scorer.1` Linear(d→d) + GELU.
    s1w: Vec<f32>,
    s1b: Vec<f32>,
    /// `scorer.3` Linear(d→1).
    s3w: Vec<f32>,
    s3b: Vec<f32>,
    /// `act_head.0` Linear(d+4→256) + GELU.
    a0w: Vec<f32>,
    a0b: Vec<f32>,
    /// `act_head.2` Linear(256→2).
    a2w: Vec<f32>,
    a2b: Vec<f32>,
    eps: f32,
    /// The head's hidden size (the encoder's `d` — the pinned checkpoints
    /// share one geometry for both stacks).
    d: usize,
}

/// Per-question forward outputs (raw — the agent turns these into answers).
pub struct HeadOutput {
    /// Per-marker scorer logits (length = marker count, all valid — the
    /// port runs one question per forward, so no −1e4 fill occurs).
    pub logits: Vec<f32>,
    /// `softmax(act_logits)` [2].
    pub act_probabilities: Vec<f32>,
}

/// The head's op buffers, owned by the CALLER and reused across layers and
/// questions.
///
/// Two reasons this lives outside [`Head::forward`] instead of as `vec!`
/// locals:
///
/// 1. **Chain-key uniqueness (the packed-path aliasing hazard).** The Metal
///    chain cache keys device buffers by `(host ptr, len, epoch)`, and one
///    epoch covers a whole packed case. Per-question host buffers that are
///    freed and re-allocated per question deterministically malloc-reuse
///    addresses — two logical buffers then share one key within the epoch,
///    and a `chain_buf` upload or a prefix-matched download serves the
///    previous question's device bytes. Measured: a shared CLS row and a
///    stale `act_in` for same-shape question pairs (raw-bit divergence from
///    the loop path). With one scratch per question allocated UP FRONT and
///    kept alive for the whole case, every logical buffer has its own
///    address for the entire epoch and the collision class cannot fire.
/// 2. **Allocation churn (reflex issue 020 T6's head half).** `forward`
///    used to allocate ~20 zeroed buffers per layer per question; the
///    scratch touches each page once and reuses it across layers and
///    questions.
///
/// `fit` never reallocates at a constant length, so a buffer's address is
/// stable from its first use on. Every buffer that reaches a backend op
/// must be a scratch field — pure-host scratch (`softmax32` temps, the
/// returned logits) stays local.
pub struct HeadScratch {
    pub(crate) sq: Vec<f32>,
    pub(crate) nx: Vec<f32>,
    pub(crate) qkv: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) scores: Vec<f32>,
    pub(crate) ctx: Vec<f32>,
    pub(crate) merged: Vec<f32>,
    pub(crate) attn_out: Vec<f32>,
    pub(crate) nx2: Vec<f32>,
    pub(crate) ff: Vec<f32>,
    pub(crate) ff2: Vec<f32>,
    pub(crate) rows: Vec<f32>,
    pub(crate) s: Vec<f32>,
    pub(crate) s1: Vec<f32>,
    pub(crate) logits_buf: Vec<f32>,
    pub(crate) cls: Vec<f32>,
    pub(crate) act_in: Vec<f32>,
    pub(crate) a: Vec<f32>,
    pub(crate) act_logits_buf: Vec<f32>,
}

impl HeadScratch {
    /// Resize to `n` when needed; a constant length keeps both the capacity
    /// AND the address (never reallocates), which is the chain-key
    /// guarantee above.
    fn fit(buf: &mut Vec<f32>, n: usize) {
        if buf.len() != n {
            buf.clear();
            buf.resize(n, 0.0);
        }
    }

    /// One fresh scratch per question. The packed path allocates ALL of a
    /// case's scratches before its first forward (see the struct doc); the
    /// per-question loop path allocates one per call — its epoch is per
    /// question, so address reuse across calls cannot alias anything.
    pub fn new() -> Self {
        Self {
            sq: Vec::new(),
            nx: Vec::new(),
            qkv: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            scores: Vec::new(),
            ctx: Vec::new(),
            merged: Vec::new(),
            attn_out: Vec::new(),
            nx2: Vec::new(),
            ff: Vec::new(),
            ff2: Vec::new(),
            rows: Vec::new(),
            s: Vec::new(),
            s1: Vec::new(),
            logits_buf: Vec::new(),
            cls: Vec::new(),
            act_in: Vec::new(),
            a: Vec::new(),
            act_logits_buf: Vec::new(),
        }
    }
}

impl Default for HeadScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Head {
    /// Pre-place every weight slice this forward hands to a backend op on
    /// the device (riir-reflex Issue 020 T1) — the `matmul_w` weights, the
    /// `add_bias_row` biases and the `layer_norm_nobias_into` norms.
    /// `type_emb` is warmed ROW BY ROW: `forward` passes one of its rows to
    /// `add_bias_row`, and a row's `(ptr, len)` key differs per `qtype`.
    pub fn warm(&self, b: &dyn Backend) {
        let d = self.d;
        for layer in &self.layers {
            b.warm_weight(&layer.n1w);
            b.warm_weight(&layer.n1b);
            b.warm_weight_2d(&layer.in_proj_w, 3 * d, d);
            b.warm_weight(&layer.in_proj_b);
            b.warm_weight_2d(&layer.out_w, d, d);
            b.warm_weight(&layer.out_b);
            b.warm_weight(&layer.n2w);
            b.warm_weight(&layer.n2b);
            b.warm_weight_2d(&layer.l1w, 4 * d, d);
            b.warm_weight(&layer.l1b);
            b.warm_weight_2d(&layer.l2w, d, 4 * d);
            b.warm_weight(&layer.l2b);
        }
        for row in self.type_emb.chunks_exact(self.d) {
            b.warm_weight(row);
        }
        b.warm_weight(&self.s0w);
        b.warm_weight(&self.s0b);
        b.warm_weight_2d(&self.s1w, d, d);
        b.warm_weight(&self.s1b);
        b.warm_weight_2d(&self.s3w, 1, d);
        b.warm_weight(&self.s3b);
        b.warm_weight_2d(&self.a0w, 256, d + 4);
        b.warm_weight(&self.a0b);
        b.warm_weight_2d(&self.a2w, 2, 256);
        b.warm_weight(&self.a2b);
    }

    /// Assemble from a parsed safetensors map (weights are REMOVED — the
    /// encoder consumed its names first).
    pub fn from_map(
        map: &mut HashMap<String, Weights>,
        ckpt: &'static str,
        d: usize,
        eps: f32,
    ) -> Result<Self> {
        let missing = |name: &str| LayaError::Pin {
            checkpoint: ckpt,
            file: name.to_string(),
            detail: "head tensor missing from checkpoint".into(),
        };
        let mut take = |name: &str| -> Result<Vec<f32>> {
            map.remove(name)
                .map(|w| w.data)
                .ok_or_else(|| missing(name))
        };
        let mut layers = Vec::with_capacity(2);
        for idx in 0..2 {
            layers.push(HeadLayer {
                in_proj_w: take(&format!("head.layers.{idx}.self_attn.in_proj_weight"))?,
                in_proj_b: take(&format!("head.layers.{idx}.self_attn.in_proj_bias"))?,
                out_w: take(&format!("head.layers.{idx}.self_attn.out_proj.weight"))?,
                out_b: take(&format!("head.layers.{idx}.self_attn.out_proj.bias"))?,
                n1w: take(&format!("head.layers.{idx}.norm1.weight"))?,
                n1b: take(&format!("head.layers.{idx}.norm1.bias"))?,
                n2w: take(&format!("head.layers.{idx}.norm2.weight"))?,
                n2b: take(&format!("head.layers.{idx}.norm2.bias"))?,
                l1w: take(&format!("head.layers.{idx}.linear1.weight"))?,
                l1b: take(&format!("head.layers.{idx}.linear1.bias"))?,
                l2w: take(&format!("head.layers.{idx}.linear2.weight"))?,
                l2b: take(&format!("head.layers.{idx}.linear2.bias"))?,
            });
        }
        Ok(Self {
            layers,
            type_emb: take("type_emb.weight")?,
            s0w: take("scorer.0.weight")?,
            s0b: take("scorer.0.bias")?,
            s1w: take("scorer.1.weight")?,
            s1b: take("scorer.1.bias")?,
            s3w: take("scorer.3.weight")?,
            s3b: take("scorer.3.bias")?,
            a0w: take("act_head.0.weight")?,
            a0b: take("act_head.0.bias")?,
            a2w: take("act_head.2.weight")?,
            a2b: take("act_head.2.bias")?,
            eps,
            d,
        })
    }

    /// Forward the encoded sequence through the head.
    ///
    /// `h` is the encoder's `last_hidden_state` flat `[seq, d]` (consumed
    /// in place — the residual stream starts there); `qtype` selects the
    /// type-embedding row; `markers` are the option `[MASK]` positions;
    /// `sc` is the caller-owned scratch (see [`HeadScratch`] for why it is
    /// not internal). ONE forward body for every backend (`.issues/005`).
    /// Host reads of device results go through [`Backend::download_into`] —
    /// exactly three per forward (logits, CLS row, act logits); everything
    /// else stays device-side under Metal.
    pub fn forward(
        &self,
        b: &dyn Backend,
        h: &mut [f32],
        qtype: usize,
        markers: &[usize],
        sc: &mut HeadScratch,
    ) -> Result<HeadOutput> {
        let d = self.d;
        let seq = h.len() / d;
        let heads = d / 64; // both pinned geometries: head_dim 64
        let hd = d / heads;
        let scale = 1.0f32 / (hd as f32).sqrt();

        // x = h + type_emb[qtype] (broadcast over positions) — in place on
        // the encoder's residual stream.
        let x: &mut [f32] = h;
        let trow = self
            .type_emb
            .get(qtype * d..(qtype + 1) * d)
            .ok_or_else(|| LayaError::Config {
                checkpoint: "head",
                detail: format!("qtype {qtype} outside the 3-row type embedding"),
            })?;
        b.add_bias_row(x, d, trow);

        for layer in &self.layers {
            // Pre-norm MHA block (no key-padding mask: one unpadded row).
            // ops::layer_norm's op order = nobias LN + per-row bias add.
            HeadScratch::fit(&mut sc.nx, seq * d);
            b.layer_norm_nobias_into(x, &layer.n1w, self.eps, d, &mut sc.sq, &mut sc.nx);
            b.add_bias_row(&mut sc.nx, d, &layer.n1b);
            HeadScratch::fit(&mut sc.qkv, seq * 3 * d);
            b.matmul_w(&sc.nx, seq, d, &layer.in_proj_w, 3 * d, &mut sc.qkv);
            b.add_bias_row(&mut sc.qkv, 3 * d, &layer.in_proj_b);
            HeadScratch::fit(&mut sc.q, seq * d);
            HeadScratch::fit(&mut sc.k, seq * d);
            HeadScratch::fit(&mut sc.v, seq * d);
            b.split_heads(&sc.qkv, 3 * d, 0, seq, heads, hd, &mut sc.q);
            b.split_heads(&sc.qkv, 3 * d, d, seq, heads, hd, &mut sc.k);
            b.split_heads(&sc.qkv, 3 * d, 2 * d, seq, heads, hd, &mut sc.v);
            // torch's need_weights path scales q by sqrt(1/head_dim) before
            // the matmul. All heads in ONE backend op (one dispatch under
            // Metal; the CPU lane loops per head identically to v1).
            b.scale(&mut sc.q, scale);
            HeadScratch::fit(&mut sc.scores, heads * seq * seq);
            b.matmul_kt_heads(&sc.q, &sc.k, heads, seq, hd, &mut sc.scores);
            b.softmax_rows(&mut sc.scores, seq);
            HeadScratch::fit(&mut sc.ctx, heads * seq * hd);
            b.matmul_heads(&sc.scores, &sc.v, heads, seq, seq, hd, &mut sc.ctx);
            HeadScratch::fit(&mut sc.merged, seq * d);
            b.merge_heads(&sc.ctx, seq, heads, hd, &mut sc.merged);
            HeadScratch::fit(&mut sc.attn_out, seq * d);
            b.matmul_w(&sc.merged, seq, d, &layer.out_w, d, &mut sc.attn_out);
            b.add_bias_row(&mut sc.attn_out, d, &layer.out_b);
            b.add(x, 0, &sc.attn_out, 0, x.len());

            // Pre-norm FF block — ReLU (torch's default activation).
            HeadScratch::fit(&mut sc.nx2, seq * d);
            b.layer_norm_nobias_into(x, &layer.n2w, self.eps, d, &mut sc.sq, &mut sc.nx2);
            b.add_bias_row(&mut sc.nx2, d, &layer.n2b);
            HeadScratch::fit(&mut sc.ff, seq * 4 * d);
            b.matmul_w(&sc.nx2, seq, d, &layer.l1w, 4 * d, &mut sc.ff);
            b.add_bias_row(&mut sc.ff, 4 * d, &layer.l1b);
            b.relu(&mut sc.ff);
            HeadScratch::fit(&mut sc.ff2, seq * d);
            b.matmul_w(&sc.ff, seq, 4 * d, &layer.l2w, d, &mut sc.ff2);
            b.add_bias_row(&mut sc.ff2, d, &layer.l2b);
            b.add(x, 0, &sc.ff2, 0, x.len());
        }

        // Gather the marker rows: [k, d].
        let k_opts = markers.len();
        HeadScratch::fit(&mut sc.rows, k_opts * d);
        b.gather_rows(x, d, markers, &mut sc.rows);

        // scorer: LN → Linear(d,d) → GELU → Linear(d,1).
        HeadScratch::fit(&mut sc.s, k_opts * d);
        b.layer_norm_nobias_into(&sc.rows, &self.s0w, self.eps, d, &mut sc.sq, &mut sc.s);
        b.add_bias_row(&mut sc.s, d, &self.s0b);
        HeadScratch::fit(&mut sc.s1, k_opts * d);
        b.matmul_w(&sc.s, k_opts, d, &self.s1w, d, &mut sc.s1);
        b.add_bias_row(&mut sc.s1, d, &self.s1b);
        b.gelu_erf(&mut sc.s1);
        HeadScratch::fit(&mut sc.logits_buf, k_opts); // [k, 1] row-major IS [k]
        b.matmul_w(&sc.s1, k_opts, d, &self.s3w, 1, &mut sc.logits_buf);
        b.add_bias_row(&mut sc.logits_buf, 1, &self.s3b);
        let mut logits = vec![0f32; k_opts];
        b.download_into(&sc.logits_buf, &mut logits);

        // act head feats — detached probs of the raw logits (inference: the
        // detach is a no-op numerically), entropy over max(k, 2).
        let p = softmax32(&logits);
        let k_eff = std::cmp::max(k_opts, 2) as f64;
        let mut ent = 0f64;
        for pi in &p {
            let cl = (*pi as f64).max(1e-9);
            ent -= cl * cl.ln();
        }
        ent /= k_eff.ln();
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top1 = *sorted.first().unwrap_or(&0.0);
        let top2 = sorted.get(1).copied().unwrap_or(0.0);
        let feats = [top1, top1 - top2, ent as f32, k_eff as f32 / 255.0];

        // pooled = CLS row (position 0) + feats → act logits → softmax.
        // The CLS row is a DEVICE result — sync it out; the features were
        // computed from the already-downloaded logits.
        HeadScratch::fit(&mut sc.cls, d);
        b.download_into(&x[..d], &mut sc.cls);
        HeadScratch::fit(&mut sc.act_in, d + 4);
        sc.act_in[..d].copy_from_slice(&sc.cls);
        sc.act_in[d..].copy_from_slice(&feats);
        HeadScratch::fit(&mut sc.a, 256);
        b.matmul_w(&sc.act_in, 1, d + 4, &self.a0w, 256, &mut sc.a);
        b.add_bias_row(&mut sc.a, 256, &self.a0b);
        b.gelu_erf(&mut sc.a);
        HeadScratch::fit(&mut sc.act_logits_buf, 2);
        b.matmul_w(&sc.a, 1, 256, &self.a2w, 2, &mut sc.act_logits_buf);
        b.add_bias_row(&mut sc.act_logits_buf, 2, &self.a2b);
        let mut act_logits = vec![0f32; 2];
        b.download_into(&sc.act_logits_buf, &mut act_logits);
        let act_probabilities = softmax32(&act_logits);

        Ok(HeadOutput {
            logits,
            act_probabilities,
        })
    }
}
