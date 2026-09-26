//! The riir-owned `ModernBERT` encoder forward — the candle port
//! (`crate::laya::encoder`) is the normative math; this module carries the
//! same op order, same shapes and the same GEMM call shapes onto the
//! flat-`Vec<f32>` ops in [`super::ops`], so the G5 drift budget is spent
//! only on the reduction-order differences, not the matmuls or the gelu.
//!
//! Geometry facts the port compiles against (`.docs/laya_reference_pin.md`):
//! `RoPE` is the ONLY position signal; NO bias tensors anywhere in the
//! encoder; layer 0 has `mlp_norm` only (the embeddings norm feeds attention
//! directly); full attention every `global_attn_every_n_layers`-th layer,
//! sliding window otherwise; per-layer-type `RoPE` theta.

use std::collections::HashMap;

use super::super::config::EncoderConfig;
use super::super::{LayaError, Result};
use super::backend::{AttnScratch, Backend};
use super::ops;
use super::weights::Weights;

/// One encoder layer's weights (all f32, bias-free — taken out of the
/// parsed map, never cloned).
struct Layer {
    /// `attn_norm` — `None` for layer 0 (the reference's `nn.Identity`).
    attn_norm: Option<Vec<f32>>,
    /// `attn.Wqkv.weight` [3d, d].
    wqkv: Vec<f32>,
    /// `attn.Wo.weight` [d, d].
    wo: Vec<f32>,
    /// `mlp.Wi.weight` [2I, d] — fused (input, gate).
    wi: Vec<f32>,
    /// `mlp.Wo.weight` [d, I].
    mlp_wo: Vec<f32>,
    /// `mlp_norm.weight` [d].
    mlp_norm: Vec<f32>,
    /// Sliding-window layer?
    sliding: bool,
}

/// The loaded `ModernBERT` encoder (our tensors).
pub struct Encoder {
    cfg: EncoderConfig,
    /// The checkpoint this encoder serves (error rendering).
    ckpt: &'static str,
    /// `embeddings.tok_embeddings.weight` [vocab, d].
    tok_emb: Vec<f32>,
    /// `embeddings.norm.weight` [d].
    emb_norm: Vec<f32>,
    layers: Vec<Layer>,
    /// `final_norm.weight` [d].
    final_norm: Vec<f32>,
}

/// Per-forward scratch — allocated once at the sequence's sizes, reused
/// across all layers (the layer loop must not allocate).
struct Scratch {
    x: Vec<f32>,
    qkv: Vec<f32>,
    attn: AttnScratch,
    merged: Vec<f32>,
    xn: Vec<f32>,
    act: Vec<f32>,
    sq: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            x: Vec::new(),
            qkv: Vec::new(),
            attn: AttnScratch::default(),
            merged: Vec::new(),
            xn: Vec::new(),
            act: Vec::new(),
            sq: Vec::new(),
        }
    }

    fn reset(&mut self, n: usize) {
        self.x.resize(n, 0.0);
        self.merged.resize(n, 0.0);
        self.xn.resize(n, 0.0);
        // qkv (seq·3d) and the attention scratch (split thirds + the score
        // parent) are sized inside the backend's `attention_forward`. The
        // GEMM stagings the fold rungs consumed (`attn_out`/`fused`/
        // `mlp_out`) are gone with them (reflex issue 020 T11): the
        // residual adds and the GLU ride the projection ops, and the
        // Metal knob-off arm stages on its own grow-only device buffer.
    }
}

impl Encoder {
    /// Pre-place every weight slice this forward hands to a backend op on
    /// the device (riir-reflex Issue 020 T1).
    ///
    /// The set is EXACTLY the slices that reach a weight-consuming op —
    /// `layer_norm_nobias_into`'s `w` and `matmul_w`'s `w`. `tok_emb` is
    /// deliberately ABSENT: the token gather is host-side (`forward`'s
    /// embedding loop), so uploading the vocab-sized table would spend
    /// device memory on bytes no kernel ever binds.
    pub fn warm(&self, b: &dyn Backend) {
        let d = self.cfg.hidden;
        let i_sz = self.cfg.intermediate;
        b.warm_weight(&self.emb_norm);
        for layer in &self.layers {
            if let Some(w) = &layer.attn_norm {
                b.warm_weight(w);
            }
            // Shapes MIRROR `forward`'s own `matmul_w` calls — a warm at
            // the wrong shape would build a transpose the hot path then
            // misses on, so these are asserted inside `weight_t_buf`
            // (`n · k == len`) rather than trusted.
            b.warm_weight_2d(&layer.wqkv, 3 * d, d);
            b.warm_weight_2d(&layer.wo, d, d);
            b.warm_weight_2d(&layer.wi, 2 * i_sz, d);
            b.warm_weight_2d(&layer.mlp_wo, d, i_sz);
            b.warm_weight(&layer.mlp_norm);
        }
        b.warm_weight(&self.final_norm);
    }

    /// Assemble from a parsed safetensors map (weights are REMOVED — the
    /// map is split between encoder and head with no second copy; the
    /// unused `temperature` tensor is what legitimately remains).
    pub fn from_map(
        map: &mut HashMap<String, Weights>,
        cfg: EncoderConfig,
        ckpt: &'static str,
    ) -> Result<Self> {
        let missing = |name: &str| LayaError::Pin {
            checkpoint: ckpt,
            file: name.to_string(),
            detail: "tensor missing from checkpoint".into(),
        };
        // The embedding first — its declared shape is checked against the
        // config vocab (rows AND width both pinned by the shape + the
        // hidden-dependent takes below).
        let vocab_name = "encoder.embeddings.tok_embeddings.weight";
        let tok_w = map.remove(vocab_name).ok_or_else(|| missing(vocab_name))?;
        if tok_w.shape.first().copied() != Some(cfg.vocab) {
            return Err(LayaError::Config {
                checkpoint: ckpt,
                detail: format!(
                    "tok_embeddings rows {:?} != config vocab {}",
                    tok_w.shape.first(),
                    cfg.vocab
                ),
            });
        }
        let tok_emb = tok_w.data;
        let mut take = |name: &str| -> Result<Vec<f32>> {
            map.remove(name)
                .map(|w| w.data)
                .ok_or_else(|| missing(name))
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for idx in 0..cfg.layers {
            let attn_norm = if idx == 0 {
                None
            } else {
                Some(take(&format!("encoder.layers.{idx}.attn_norm.weight"))?)
            };
            layers.push(Layer {
                attn_norm,
                wqkv: take(&format!("encoder.layers.{idx}.attn.Wqkv.weight"))?,
                wo: take(&format!("encoder.layers.{idx}.attn.Wo.weight"))?,
                wi: take(&format!("encoder.layers.{idx}.mlp.Wi.weight"))?,
                mlp_wo: take(&format!("encoder.layers.{idx}.mlp.Wo.weight"))?,
                mlp_norm: take(&format!("encoder.layers.{idx}.mlp_norm.weight"))?,
                sliding: cfg.sliding[idx],
            });
        }
        Ok(Self {
            cfg,
            ckpt,
            tok_emb,
            emb_norm: take("encoder.embeddings.norm.weight")?,
            layers,
            final_norm: take("encoder.final_norm.weight")?,
        })
    }

    /// Forward `input_ids` (one unpadded sequence — the reference capture's
    /// batch shape) to `last_hidden_state` as a flat `[seq, d]` row-major
    /// buffer. ONE forward body for every backend (`.issues/005`): the op
    /// order lives here and nowhere else. Under the Metal backend the
    /// returned Vec is a HANDLE — ops run device-side and its host bytes
    /// are stale until [`Backend::download_into`] syncs; every consumer
    /// below reads it only through backend ops.
    pub fn forward(&self, b: &dyn Backend, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_packed(b, input_ids, std::slice::from_ref(&input_ids.len()))
    }

    /// The head dimension this encoder dispatches attention at (the agent
    /// feeds it to [`Backend::supports_packed_attention`]).
    pub fn head_dim(&self) -> usize {
        self.cfg.head_dim()
    }

    /// Packed varlen forward (reflex issue 020 T5): `input_ids` is the
    /// CONCATENATION of `seqs.len()` sequences with lengths `seqs`
    /// (`Σ seqs == input_ids.len()`).
    ///
    /// Every row-wise op and every GEMM runs ONCE over all `total` rows —
    /// the batch's GEMMs launch `Σseq` rows per dispatch instead of one
    /// dispatch per question — while attention is dispatched PER SEQUENCE
    /// at its packed offset, so each sequence sees exactly the rows,
    /// positions and windows the single-sequence [`Self::forward`] gave
    /// it. Per-row results are therefore bit-identical to the sequential
    /// forwards (the packed-equivalence gate holds that, CPU by
    /// construction, Metal by offset-bound kernels).
    ///
    /// `seqs` with one entry IS [`Self::forward`] (the same code, zeroed
    /// offsets).
    pub fn forward_packed(
        &self,
        b: &dyn Backend,
        input_ids: &[u32],
        seqs: &[usize],
    ) -> Result<Vec<f32>> {
        let total = input_ids.len();
        debug_assert_eq!(seqs.iter().sum::<usize>(), total, "packed seqs sum");
        // Per-question row segments for the backend's kernel choice (the
        // packed ≡ loop law under split-K); cleared on every exit path.
        let _segments = RowSegments::set(b, seqs);
        let d = self.cfg.hidden;
        let hd = self.cfg.head_dim();
        let heads = self.cfg.heads;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let i_sz = self.cfg.intermediate;
        let eps = self.cfg.eps;

        // Embeddings: host-side token gather (a ~total·d memcpy off the
        // agent-owned table — cheap, and it keeps the vocab-sized table out
        // of the device flow entirely) + LayerNorm (the norm feeds layer 0's
        // attention directly — the layer-0 quirk).
        let mut gathered = vec![0f32; total * d];
        for (s, id) in input_ids.iter().enumerate() {
            let row = (*id as usize) * d;
            let Some(src) = self.tok_emb.get(row..row + d) else {
                return Err(LayaError::Config {
                    checkpoint: self.ckpt,
                    detail: format!("token id {id} outside the embedding table"),
                });
            };
            gathered[s * d..s * d + d].copy_from_slice(src);
        }
        let mut h = vec![0f32; total * d];
        let mut sq = Vec::new();
        b.layer_norm_nobias_into(&gathered, &self.emb_norm, eps, d, &mut sq, &mut h);

        let mut sc = Scratch::new();
        sc.reset(total * d);
        let mut rope_full: Option<(Vec<f32>, Vec<f32>)> = None;
        let mut rope_slide: Option<(Vec<f32>, Vec<f32>)> = None;

        // Sliding-window additive masks, ONE PER SEQUENCE ([seq, seq] each,
        // built once, used by every sliding layer) — skipped when the
        // backend predicates the window in-kernel (the fused dispatch's
        // own gate, mirrored per sequence: `window < seq − 1`).
        let window = self.cfg.sliding_window();
        let masks: Vec<Option<Vec<f32>>> = if b.needs_window_mask(hd) {
            seqs.iter()
                .map(|&seq| {
                    if seq > 1 && window < seq - 1 {
                        let mut m = vec![f32::MIN; seq * seq];
                        for qi in 0..seq {
                            let lo = qi.saturating_sub(window);
                            let hi = (qi + window).min(seq - 1);
                            for kv in lo..=hi {
                                m[qi * seq + kv] = 0.0;
                            }
                        }
                        Some(m)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        for layer in &self.layers {
            // x = attn_norm(h) — or h itself on layer 0 (the identity path,
            // copied DEVICE-side: the residual stream is device-current
            // under Metal and a host copy would read stale bytes).
            match &layer.attn_norm {
                Some(w) => {
                    b.layer_norm_nobias_into(&h, w, eps, d, &mut sc.sq, &mut sc.x);
                }
                None => b.copy_into(&h, &mut sc.x),
            }

            // Attention: ONE Wqkv GEMM over all packed rows, then the
            // backend's fused attention block PER SEQUENCE at its packed
            // offset (rope + q-scale + score + mask + softmax + value mix
            // + head merge in ONE op per sequence — the sequence sees its
            // own rows and window exactly as the unbatched forward).
            sc.qkv.resize(total * 3 * d, 0.0);
            b.matmul_w(&sc.x, total, d, &layer.wqkv, 3 * d, &mut sc.qkv);
            let rope = if layer.sliding {
                rope_slide.get_or_insert_with(|| {
                    self.rope_tables_for(seqs, hd, self.cfg.rope_theta_slide)
                })
            } else {
                rope_full
                    .get_or_insert_with(|| self.rope_tables_for(seqs, hd, self.cfg.rope_theta_full))
            };
            let mut off = 0usize;
            for (si, &seq) in seqs.iter().enumerate() {
                let mask = if layer.sliding {
                    masks.get(si).and_then(Option::as_deref)
                } else {
                    None
                };
                b.attention_forward(
                    &sc.qkv,
                    off * 3 * d,
                    &rope.0,
                    &rope.1,
                    off,
                    scale,
                    seq,
                    heads,
                    hd,
                    if layer.sliding { window } else { usize::MAX },
                    mask,
                    &mut sc.attn,
                    &mut sc.merged,
                    off * d,
                );
                off += seq;
            }
            b.matmul_w_accum(&sc.merged, total, d, &layer.wo, d, &mut h);

            // MLP: fused Wi → gelu(input) · gate → Wo — whole packed
            // buffers, unchanged op order. The GLU epilogue rides the
            // projection op (reflex issue 020 T11: the fold rungs — the
            // Metal lane folds them into the split-K epilogue when the
            // whole call splits; both arms bit-identical to the unfused
            // pair by construction).
            b.layer_norm_nobias_into(&h, &layer.mlp_norm, eps, d, &mut sc.sq, &mut sc.xn);
            sc.act.resize(total * i_sz, 0.0);
            b.matmul_w_glu(&sc.xn, total, d, &layer.wi, i_sz, &mut sc.act);
            b.matmul_w_accum(&sc.act, total, i_sz, &layer.mlp_wo, d, &mut h);
        }

        let mut out = vec![0f32; total * d];
        b.layer_norm_nobias_into(&h, &self.final_norm, eps, d, &mut sc.sq, &mut out);
        Ok(out)
    }
}

impl Encoder {
    /// The packed forward's rope tables: one `[rows, hd]` pair per theta.
    /// A single sequence gets the plain [`ops::rope_tables`] (no copy); a
    /// batch gets the per-row packed concatenation.
    fn rope_tables_for(&self, seqs: &[usize], hd: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
        if seqs.len() == 1 {
            ops::rope_tables(seqs[0], hd, theta)
        } else {
            ops::rope_tables_packed(seqs, hd, theta)
        }
    }

    /// DEBUG PROBE (issue 016; retained as the 018 stability+bisect
    /// instrument): the single-sequence [`Self::forward`] op stream with a
    /// per-op sink. Every sink call receives `(tag, host-current bytes)`
    /// for the op's primary output (one `download_into` per call — a
    /// device sync per op, debug only). The op order mirrors
    /// `forward_packed` at `seqs = [len]` exactly; running it against two
    /// backends and zipping the tags localizes the first divergent op.
    #[cfg(feature = "laya-riir-cubecl")]
    pub fn forward_probe(
        &self,
        b: &dyn Backend,
        input_ids: &[u32],
        sink: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<()> {
        fn dl(b: &dyn Backend, src: &[f32], buf: &mut Vec<f32>) {
            buf.clear();
            buf.resize(src.len(), 0.0);
            b.download_into(src, buf);
        }
        /// Issue 018 lever 1 (deep mode): ALSO sink the attention block's
        /// device-written internals (q/k/v splits, the score parent, the
        /// per-head context) and the host-authored mask/rope slots AS THE
        /// DEVICE HOLDS THEM. `LAYA_PROBE_DEEP=1` arms it. This is the
        /// granularity step that names a KERNEL rather than an op: the
        /// attention composes ~10 kernels between the op-level sinks, and
        /// a wobble fire inside that window only says "attention" — the
        /// first divergent INTERNAL tag (`scores` after a clean `q` = the
        /// score GEMM; `mask` ≠ host bytes = the upload path; …) pins the
        /// kernel. Cost: ~15 MB of extra readback per layer per pass.
        fn deep_probes() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("LAYA_PROBE_DEEP").is_ok_and(|v| v == "1")
            })
        }
        let total = input_ids.len();
        let _segments = RowSegments::set(b, std::slice::from_ref(&total));
        let d = self.cfg.hidden;
        let hd = self.cfg.head_dim();
        let heads = self.cfg.heads;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let i_sz = self.cfg.intermediate;
        let eps = self.cfg.eps;

        let mut gathered = vec![0f32; total * d];
        for (s, id) in input_ids.iter().enumerate() {
            let row = (*id as usize) * d;
            let Some(src) = self.tok_emb.get(row..row + d) else {
                return Err(LayaError::Config {
                    checkpoint: self.ckpt,
                    detail: format!("token id {id} outside the embedding table"),
                });
            };
            gathered[s * d..s * d + d].copy_from_slice(src);
        }
        let mut h = vec![0f32; total * d];
        let mut sq = Vec::new();
        b.layer_norm_nobias_into(&gathered, &self.emb_norm, eps, d, &mut sq, &mut h);
        let mut pb = Vec::new();
        dl(b, &h, &mut pb);
        sink("emb", &pb);

        let mut sc = Scratch::new();
        sc.reset(total * d);
        let rope_full = self.rope_tables_for(std::slice::from_ref(&total), hd, self.cfg.rope_theta_full);
        let rope_slide =
            self.rope_tables_for(std::slice::from_ref(&total), hd, self.cfg.rope_theta_slide);
        let window = self.cfg.sliding_window();
        let mask: Option<Vec<f32>> = if b.needs_window_mask(hd) && total > 1 && window < total - 1 {
            let mut m = vec![f32::MIN; total * total];
            for qi in 0..total {
                let lo = qi.saturating_sub(window);
                let hi = (qi + window).min(total - 1);
                for kv in lo..=hi {
                    m[qi * total + kv] = 0.0;
                }
            }
            Some(m)
        } else {
            None
        };

        for (li, layer) in self.layers.iter().enumerate() {
            match &layer.attn_norm {
                Some(w) => b.layer_norm_nobias_into(&h, w, eps, d, &mut sc.sq, &mut sc.x),
                None => b.copy_into(&h, &mut sc.x),
            }
            dl(b, &sc.x, &mut pb);
            sink(&format!("L{li}.x"), &pb);

            sc.qkv.resize(total * 3 * d, 0.0);
            b.matmul_w(&sc.x, total, d, &layer.wqkv, 3 * d, &mut sc.qkv);
            dl(b, &sc.qkv, &mut pb);
            sink(&format!("L{li}.qkv"), &pb);

            let rope = if layer.sliding { &rope_slide } else { &rope_full };
            b.attention_forward(
                &sc.qkv,
                0,
                &rope.0,
                &rope.1,
                0,
                scale,
                total,
                heads,
                hd,
                if layer.sliding { window } else { usize::MAX },
                if layer.sliding { mask.as_deref() } else { None },
                &mut sc.attn,
                &mut sc.merged,
                0,
            );
            dl(b, &sc.merged, &mut pb);
            sink(&format!("L{li}.attn"), &pb);

            if deep_probes() {
                dl(b, &sc.attn.q, &mut pb);
                sink(&format!("L{li}.q"), &pb);
                dl(b, &sc.attn.k, &mut pb);
                sink(&format!("L{li}.k"), &pb);
                dl(b, &sc.attn.v, &mut pb);
                sink(&format!("L{li}.v"), &pb);
                dl(b, &sc.attn.scores, &mut pb);
                sink(&format!("L{li}.scores"), &pb);
                dl(b, &sc.attn.ctx, &mut pb);
                sink(&format!("L{li}.ctx"), &pb);
                if layer.sliding
                    && let Some(m) = mask.as_deref()
                {
                    dl(b, m, &mut pb);
                    sink(&format!("L{li}.mask"), &pb);
                }
                dl(b, &rope.0, &mut pb);
                sink(&format!("L{li}.cos"), &pb);
                dl(b, &rope.1, &mut pb);
                sink(&format!("L{li}.sin"), &pb);
            }

            b.matmul_w_accum(&sc.merged, total, d, &layer.wo, d, &mut h);
            dl(b, &h, &mut pb);
            sink(&format!("L{li}.h_attn"), &pb);

            b.layer_norm_nobias_into(&h, &layer.mlp_norm, eps, d, &mut sc.sq, &mut sc.xn);
            sc.act.resize(total * i_sz, 0.0);
            b.matmul_w_glu(&sc.xn, total, d, &layer.wi, i_sz, &mut sc.act);
            dl(b, &sc.act, &mut pb);
            sink(&format!("L{li}.act"), &pb);
            b.matmul_w_accum(&sc.act, total, i_sz, &layer.mlp_wo, d, &mut h);
            dl(b, &h, &mut pb);
            sink(&format!("L{li}.h_mlp"), &pb);
        }

        let mut out = vec![0f32; total * d];
        b.layer_norm_nobias_into(&h, &self.final_norm, eps, d, &mut sc.sq, &mut out);
        dl(b, &out, &mut pb);
        sink("final", &pb);
        Ok(())
    }
}

/// Scope guard for [`Backend::set_row_segments`]: set on entry, cleared on
/// drop — an early `?` return must not leave the hint describing the next
/// forward's GEMMs.
struct RowSegments<'a>(&'a dyn Backend);

impl<'a> RowSegments<'a> {
    fn set(b: &'a dyn Backend, seqs: &[usize]) -> Self {
        b.set_row_segments(seqs);
        Self(b)
    }
}

impl Drop for RowSegments<'_> {
    fn drop(&mut self) {
        self.0.set_row_segments(&[]);
    }
}
