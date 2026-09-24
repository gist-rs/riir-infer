//! The **DSpark drafter** — CPU reference port of the llama.cpp contract.
//!
//! Extracted from `tests/bench_696_issue717_g2full_trained_dspark.rs` (Plan 343
//! T1.5). It lived inside that test, which meant the only way to measure a
//! drafter against the dspark floor (1.689 tok/cyc, Issue 717 G2) was to be that
//! test: `src/` referenced dspark only in `gguf_loader`. Any other consumer had
//! to re-port ~500 lines or quote a number it could not reproduce.
//!
//! # The contract this ports
//!
//! Mirrors llama.cpp `src/models/dflash.cpp::build_dspark_markov_head` +
//! `common/speculative.cpp::draft_dflash` with `is_dspark=true`:
//!
//! * **encoder** — concat of 5 target layers `[5 × n_embd]` → `fc [25600 → 5120]`
//!   → `RMSNorm(hidden_norm)` = `inp_g`.
//! * **KV injection** — at a committed position, `K = rope(k_norm(wk · inp_g))`,
//!   `V = wv · inp_g`, with **no** `attn_norm` (the `graph<false>` embd branch).
//! * **noise block** — `[anchor, MASK×3]` with non-causal attention over the
//!   committed injected KV plus the block's own K/V.
//! * **markov head** — `col_i = base_i + B · A[prev_i]`, chained across the 4
//!   block positions.
//!
//! `log_snr_fc1`/`fc2` are deliberately **not** used at inference — llama.cpp's
//! dspark path does not consult them, so porting them would diverge from the
//! contract being measured against.
//!
//! # Why the math helpers are local
//!
//! `rmsnorm_inplace`, `rope_neox`, `matvec`, `q4_1_row_dot` and `half_f32` stay
//! module-private on purpose. `q4_1_row_dot`/`half_f32` encode the **Q4_1 GGUF
//! row layout** this drafter's `lm_head`/`markov_b` are stored in, which no other
//! caller in this crate needs. The crate does carry other `rmsnorm` variants
//! (`cpu_reference`, `gemma2_cubecl`, `gemma4_cubecl`) with differing signatures;
//! consolidating them is a separate cleanup and NOT folded in here, because
//! swapping a normalization under a numerics port is exactly how a
//! bit-comparison stops meaning anything.

use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use riir_infer_core::gguf_loader::GgufFile;

#[inline]
fn rmsnorm_inplace(x: &mut [f32], gamma: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    for (xi, &g) in x.iter_mut().zip(gamma) {
        *xi = *xi * inv * g;
    }
}

/// NEOX-style RoPE over `rot_dim` (pairs at (i, i + rot_dim/2)), in place.
/// `theta_i = base^(-2i/rot_dim)`.
#[inline]
fn rope_neox(x: &mut [f32], pos: usize, rot_dim: usize, base: f32) {
    let half = rot_dim / 2;
    for i in 0..half {
        let freq = base.powf(-2.0 * i as f32 / rot_dim as f32);
        let ang = pos as f32 * freq;
        let (s, c) = (ang.sin(), ang.cos());
        let xi = x[i];
        let xj = x[i + half];
        x[i] = xi * c - xj * s;
        x[i + half] = xi * s + xj * c;
    }
}

/// y = W·x for a row-major f32 weight `W` of shape [out, in] (flattened).
#[inline]
fn matvec(w: &[f32], x: &[f32], out: &mut [f32]) {
    let inp = x.len();
    for (o, yo) in out.iter_mut().enumerate() {
        let row = &w[o * inp..(o + 1) * inp];
        let mut acc = 0.0f32;
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        *yo = acc;
    }
}

/// Q4_1 row·x: row `r` of a [rows, 5120] Q4_1 tensor (20 B per 32 weights).
fn q4_1_row_dot(slice: &[u8], row_idx: usize, cols: usize, x: &[f32]) -> f32 {
    const BLOCK_BYTES: usize = 20;
    const QK: usize = 32;
    let blocks_per_row = cols / QK;
    let start = row_idx * blocks_per_row * BLOCK_BYTES;
    let row = &slice[start..start + blocks_per_row * BLOCK_BYTES];
    let mut acc = 0.0f32;
    // Dequantize 32 weights per block straight into the dot.
    for (b, chunk) in row.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let d = half_f32(chunk[0], chunk[1]);
        let m = half_f32(chunk[2], chunk[3]);
        let qs = &chunk[4..20];
        let base = b * QK;
        for (j, &q) in qs.iter().enumerate() {
            let lo = d * f32::from(q & 0x0F) + m;
            let hi = d * f32::from(q >> 4) + m;
            acc += lo * x[base + j];
            acc += hi * x[base + j + QK / 2];
        }
    }
    acc
}

/// IEEE 754 half → f32 (bit manipulation — the `half` crate is an optional
/// dep of riir-gpu, not enabled for this test target).
#[inline]
fn half_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = f32::from(bits & 0x03FF);
    if exp == 0 {
        sign * mant * 2.0f32.powi(-24) // subnormal
    } else if exp == 31 {
        f32::NAN // inf/nan — not expected in weights
    } else {
        sign * (1.0 + mant / 1024.0) * 2.0f32.powi(exp - 15)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ───────────────────────────────────────────────────────────────────────────
// The DSpark drafter (CPU reference port of the llama.cpp contract)
// ───────────────────────────────────────────────────────────────────────────

pub struct DrafterLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_norm: Vec<f32>, // [head_dim] — llama.cpp attn_q_norm is [n_embd_head]
    k_norm: Vec<f32>,
    wq: Vec<f32>, // [5120, 5120]
    wk: Vec<f32>, // [512,, 5120]
    wv: Vec<f32>, // [512, 5120]
    wo: Vec<f32>, // [5120, 5120]
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

pub struct DsparkDrafter {
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_ff: usize,
    rope_base: f32,
    eps: f32,
    mask_token: usize,
    rank: usize,
    layers: Vec<DrafterLayer>,
    /// fc [25600 → 5120] + hidden_norm — the encoder fusion of target layers.
    enc_fc: Vec<f32>,
    enc_norm: Vec<f32>,
    /// final output_norm.
    out_norm: Vec<f32>,
    /// Markov head: A [vocab, 256] (BF16), B [vocab, 256] (Q4_1).
    markov_a: Vec<f32>,
    markov_b: Vec<f32>,
    /// conf_proj [5376] + bias.
    conf_w: Vec<f32>,
    conf_b: f32,
    /// lm_head raw Q4_1 bytes (dequantized inline per GEMV) + vocab.
    lm_raw: Vec<u8>,
    vocab: usize,
    /// The GGUF handle (mmap owner + row dequant for wte).
    gguf: GgufFile,
    /// Cached MASK embedding row.
    mask_row: Vec<f32>,
}

/// Drafter KV cache: [layer][pos] → K/V rows (kv_heads × head_dim = 512 f32).
pub struct DrafterCache {
    k: Vec<Vec<f32>>, // [n_layer][max_pos * 512]
    v: Vec<Vec<f32>>,
    len: usize, // number of valid positions
}

pub struct BlockOut {
    /// base logits [4][vocab] (pre-markov).
    base_logits: Vec<Vec<f32>>,
    /// final-norm hidden [4][5120] (conf head input).
    hidden: Vec<Vec<f32>>,
}

impl DsparkDrafter {
    pub fn load(path: &Path) -> Self {
        let g = GgufFile::open(path).expect("open dspark gguf");
        let get_u = |k: &str| g.metadata_u64(k).unwrap_or_else(|| panic!("meta {k}"));
        let n_embd = get_u("dspark.embedding_length") as usize;
        let n_head = get_u("dspark.attention.head_count") as usize;
        let n_kv_head = get_u("dspark.attention.head_count_kv") as usize;
        let head_dim = get_u("dspark.attention.key_length") as usize;
        let n_ff = get_u("dspark.feed_forward_length") as usize;
        let n_layer = get_u("dspark.block_count") as usize;
        let mask_token = get_u("dspark.dspark.mask_token_id") as usize;
        let rank = get_u("dspark.dspark.markov_rank") as usize;
        let rope_base = g.metadata_f64("dspark.rope.freq_base").unwrap_or(1e7) as f32;
        let eps = g.metadata_f64("dspark.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
        let vocab = g.metadata_u64("dspark.vocab_size").unwrap_or(248320) as usize;
        println!(
            "[dspark] arch: {n_layer} layers, n_embd={n_embd}, heads={n_head}/{n_kv_head} \
             (dim {head_dim}), ffn={n_ff}, rank={rank}, mask={mask_token}, vocab={vocab}, \
             rope_base={rope_base}"
        );

        let t0 = Instant::now();
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let dq = |name: String| g.dequant_f16_to_f32(&name).unwrap();
            layers.push(DrafterLayer {
                attn_norm: dq(format!("blk.{l}.attn_norm.weight")),
                ffn_norm: dq(format!("blk.{l}.ffn_norm.weight")),
                q_norm: dq(format!("blk.{l}.attn_q_norm.weight")),
                k_norm: dq(format!("blk.{l}.attn_k_norm.weight")),
                wq: dq(format!("blk.{l}.attn_q.weight")),
                wk: dq(format!("blk.{l}.attn_k.weight")),
                wv: dq(format!("blk.{l}.attn_v.weight")),
                wo: dq(format!("blk.{l}.attn_output.weight")),
                gate: dq(format!("blk.{l}.ffn_gate.weight")),
                up: dq(format!("blk.{l}.ffn_up.weight")),
                down: dq(format!("blk.{l}.ffn_down.weight")),
            });
        }
        let enc_fc = g.dequant_f16_to_f32("dspark.fc.weight").unwrap();
        let enc_norm = g.dequant_f16_to_f32("dspark.hidden_norm.weight").unwrap();
        let out_norm = g.dequant_f16_to_f32("output_norm.weight").unwrap();
        let markov_a = g.dequant_f16_to_f32("dspark.markov_head_a.weight").unwrap();
        let markov_b = g.dequant_f16_to_f32("dspark.markov_head_b.weight").unwrap();
        let conf_w = g.dequant_f16_to_f32("dspark.confidence_head.weight").unwrap();
        let conf_b = g.dequant_f16_to_f32("dspark.confidence_head.bias").unwrap()[0];
        let lm_raw = g.tensor_slice("output.weight").expect("output.weight").to_vec();
        let mask_row = g
            .dequant_tensor_row("token_embd.weight", mask_token, n_embd)
            .unwrap();
        println!("[dspark] weights dequantized in {:?} (host RSS note: ~4 GB)", t0.elapsed());
        assert_eq!(enc_fc.len(), 5 * n_embd * n_embd, "fc is [5*n_embd, n_embd]");
        assert_eq!(markov_a.len(), vocab * rank);
        assert_eq!(markov_b.len(), vocab * rank);
        assert_eq!(conf_w.len(), n_embd + rank);
        Self {
            n_embd,
            n_head,
            n_kv_head,
            head_dim,
            n_ff,
            rope_base,
            eps,
            mask_token,
            rank,
            layers,
            enc_fc,
            enc_norm,
            out_norm,
            markov_a,
            markov_b,
            conf_w,
            conf_b,
            lm_raw,
            vocab,
            gguf: g,
            mask_row,
        }
    }

    /// Allocate a KV cache sized for this drafter and `max_pos` positions.
    ///
    /// Exists so callers do not have to know the KV row width
    /// (`n_kv_head * head_dim`) or the layer count. The extracted test built this
    /// struct literally from private fields, which is precisely the detail that
    /// belongs with the drafter rather than with every consumer.
    pub fn new_cache(&self, max_pos: usize) -> DrafterCache {
        let row = self.n_kv_head * self.head_dim;
        DrafterCache {
            k: vec![vec![0.0f32; max_pos * row]; self.layers.len()],
            v: vec![vec![0.0f32; max_pos * row]; self.layers.len()],
            len: 0,
        }
    }

    /// The MASK token id the noise block is built from.
    #[inline]
    pub fn mask_token(&self) -> usize {
        self.mask_token
    }

    /// Drafter vocabulary size.
    #[inline]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Number of drafter layers.
    #[inline]
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    /// The ENCODER: fused target features [5×n_embd] → inp_g [n_embd].
    pub fn encode(&self, feats: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_embd];
        matvec(&self.enc_fc, feats, &mut out);
        let mut o = out;
        rmsnorm_inplace(&mut o, &self.enc_norm, self.eps);
        o
    }

    /// KV INJECTION at committed position `p` (graph<false> embd branch):
    /// K = rope(k_norm_perhead(wk·inp_g)), V = wv·inp_g. No attn_norm.
    pub fn inject(&self, cache: &mut DrafterCache, inp_g: &[f32], p: usize) {
        let kv = self.n_kv_head * self.head_dim;
        for (li, lw) in self.layers.iter().enumerate() {
            let mut k = vec![0.0f32; kv];
            matvec(&lw.wk, inp_g, &mut k);
            let mut v = vec![0.0f32; kv];
            matvec(&lw.wv, inp_g, &mut v);
            // per-head RMSNorm with k_norm (head-local, like attn_k_norm).
            for h in 0..self.n_kv_head {
                let sl = &mut k[h * self.head_dim..(h + 1) * self.head_dim];
                rmsnorm_inplace(sl, &lw.k_norm, self.eps);
                rope_neox(sl, p, self.head_dim, self.rope_base);
            }
            let off = p * kv;
            cache.k[li][off..off + kv].copy_from_slice(&k);
            cache.v[li][off..off + kv].copy_from_slice(&v);
        }
        cache.len = cache.len.max(p + 1);
    }

    /// The NOISE BLOCK forward: tokens `[anchor, mask×3]` at positions
    /// `[n..n+3]`, non-causal attention over `[0..=n+3]` (committed injected
    /// KV + the block's own K/V). Returns base logits + normed hiddens.
    pub fn forward_block(&self, cache: &DrafterCache, anchor: usize, n: usize) -> BlockOut {
        const BLOCK: usize = 4;
        let d = self.n_embd;
        let kv = self.n_kv_head * self.head_dim;
        let q_dim = self.n_head * self.head_dim;

        // Embeddings: anchor row + mask rows.
        let mut x = vec![vec![0.0f32; d]; BLOCK];
        let anchor_row = self
            .gguf
            .dequant_tensor_row("token_embd.weight", anchor, d)
            .expect("anchor embd row");
        x[0].copy_from_slice(&anchor_row);
        for xi in x.iter_mut().skip(1) {
            xi.copy_from_slice(&self.mask_row);
        }

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let group = self.n_head / self.n_kv_head; // 10

        for (li, lw) in self.layers.iter().enumerate() {
            // h = rmsnorm(x, attn_norm)
            let mut h = vec![vec![0.0f32; d]; BLOCK];
            for (hi, xi) in h.iter_mut().zip(x.iter()) {
                hi.copy_from_slice(xi);
                rmsnorm_inplace(hi, &lw.attn_norm, self.eps);
            }
            // Q/K/V projections per block position.
            let mut qs = vec![vec![0.0f32; q_dim]; BLOCK];
            let mut ks = vec![vec![0.0f32; kv]; BLOCK];
            let mut vs = vec![vec![0.0f32; kv]; BLOCK];
            for i in 0..BLOCK {
                matvec(&lw.wq, &h[i], &mut qs[i]);
                matvec(&lw.wk, &h[i], &mut ks[i]);
                matvec(&lw.wv, &h[i], &mut vs[i]);
            }
            // q/k per-head norm + rope (block positions n..n+3).
            for i in 0..BLOCK {
                let pos = n + i;
                for qh in 0..self.n_head {
                    let sl = &mut qs[i][qh * self.head_dim..(qh + 1) * self.head_dim];
                    rmsnorm_inplace(sl, &lw.q_norm, self.eps);
                    rope_neox(sl, pos, self.head_dim, self.rope_base);
                }
                for kh in 0..self.n_kv_head {
                    let sl = &mut ks[i][kh * self.head_dim..(kh + 1) * self.head_dim];
                    rmsnorm_inplace(sl, &lw.k_norm, self.eps);
                    rope_neox(sl, pos, self.head_dim, self.rope_base);
                }
            }
            // Write the block's K/V into scratch attention range (append-only
            // semantics: gather keys from cache[0..n] + block[n..n+4)).
            // Scratch key buffer: [total][kv] (cache holds committed + we
            // overlay the block's own K/V at [n..n+4)).
            let total = n + BLOCK;
            let mut keys = vec![0.0f32; total * kv];
            let mut vals = vec![0.0f32; total * kv];
            keys[..n * kv].copy_from_slice(&cache.k[li][..n * kv]);
            vals[..n * kv].copy_from_slice(&cache.v[li][..n * kv]);
            for i in 0..BLOCK {
                keys[(n + i) * kv..(n + i + 1) * kv].copy_from_slice(&ks[i]);
                vals[(n + i) * kv..(n + i + 1) * kv].copy_from_slice(&vs[i]);
            }
            // Attention per q head (non-causal: all `total` keys).
            let mut wo_in = vec![vec![0.0f32; q_dim]; BLOCK];
            for i in 0..BLOCK {
                for qh in 0..self.n_head {
                    let kvh = qh / group;
                    let q = &qs[i][qh * self.head_dim..(qh + 1) * self.head_dim];
                    let mut scores = vec![0.0f32; total];
                    let mut maxv = f32::NEG_INFINITY;
                    for t in 0..total {
                        let key = &keys[t * kv + kvh * self.head_dim
                            ..t * kv + (kvh + 1) * self.head_dim];
                        let mut s = 0.0f32;
                        for (qa, kb) in q.iter().zip(key) {
                            s += qa * kb;
                        }
                        s *= scale;
                        scores[t] = s;
                        maxv = maxv.max(s);
                    }
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - maxv).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    let dst = &mut wo_in[i][qh * self.head_dim..(qh + 1) * self.head_dim];
                    for t in 0..total {
                        let wgt = scores[t] * inv;
                        let vrow = &vals[t * kv + kvh * self.head_dim
                            ..t * kv + (kvh + 1) * self.head_dim];
                        for (dv, vv) in dst.iter_mut().zip(vrow) {
                            *dv += wgt * vv;
                        }
                    }
                }
            }
            // out proj + residual (llama.cpp: ffn_inp = wo(attn) + inpL).
            for i in 0..BLOCK {
                let mut o = vec![0.0f32; d];
                matvec(&lw.wo, &wo_in[i], &mut o);
                for (xi, ov) in x[i].iter_mut().zip(o) {
                    *xi += ov;
                }
            }
            // FFN: rmsnorm + SwiGLU + down + residual.
            let mut h2 = vec![0.0f32; d];
            let mut gate = vec![0.0f32; self.n_ff];
            let mut up = vec![0.0f32; self.n_ff];
            let mut mid = vec![0.0f32; self.n_ff];
            for xrow in x.iter_mut().take(BLOCK) {
                h2.copy_from_slice(xrow);
                rmsnorm_inplace(&mut h2, &lw.ffn_norm, self.eps);
                matvec(&lw.gate, &h2, &mut gate);
                matvec(&lw.up, &h2, &mut up);
                for (m, (gv, uv)) in mid.iter_mut().zip(gate.iter().zip(up.iter())) {
                    *m = gv * uv / (1.0 + (-gv).exp());
                }
                let mut dn = vec![0.0f32; d];
                matvec(&lw.down, &mid, &mut dn);
                for (xi, dv) in xrow.iter_mut().zip(dn) {
                    *xi += dv;
                }
            }
        }

        // Final norm + lm_head (per position; rayon over vocab rows).
        let mut hidden = vec![vec![0.0f32; d]; BLOCK];
        for (hi, xi) in hidden.iter_mut().zip(x.iter()) {
            hi.copy_from_slice(xi);
            rmsnorm_inplace(hi, &self.out_norm, self.eps);
        }
        let lm = &self.lm_raw;
        let base_logits: Vec<Vec<f32>> = hidden
            .par_iter()
            .map(|h| {
                (0..self.vocab)
                    .into_par_iter()
                    .map(|v| q4_1_row_dot(lm, v, d, h))
                    .collect()
            })
            .collect();
        BlockOut { base_logits, hidden }
    }

    /// The markov head chain (build_dspark_markov_head port).
    /// Returns (draft tokens [4], per-position confidence).
    pub fn markov_chain(
        &self,
        out: &BlockOut,
        anchor: usize,
        use_base: bool,
        use_bias: bool,
    ) -> (Vec<usize>, Vec<f32>) {
        let r = self.rank;
        let a = &self.markov_a;
        let b = &self.markov_b;
        let mut prev = anchor;
        let mut drafts = Vec::with_capacity(4);
        let mut confs = Vec::with_capacity(4);
        let argmax = |col: &[f32]| -> usize {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (v, &c) in col.iter().enumerate() {
                if c > best_v {
                    best_v = c;
                    best = v;
                }
            }
            best
        };
        for i in 0..4 {
            let a_row = &a[prev * r..(prev + 1) * r];
            // col[v] = (base ? base[i][v] : 0) + (bias ? B[v]·A[prev] : 0)
            let best = match (use_base, use_bias) {
                (false, _) => {
                    // bias-only (or nothing): must materialize the bias vector
                    if use_bias {
                        let col: Vec<f32> = (0..self.vocab)
                            .into_par_iter()
                            .map(|v| {
                                let brow = &b[v * r..(v + 1) * r];
                                let mut bias = 0.0f32;
                                for (bb, aa) in brow.iter().zip(a_row) {
                                    bias += bb * aa;
                                }
                                bias
                            })
                            .collect();
                        argmax(&col)
                    } else {
                        unreachable!("at least one of base/bias")
                    }
                }
                (true, false) => argmax(&out.base_logits[i]), // base only
                (true, true) => {
                    let base = &out.base_logits[i];
                    let col: Vec<f32> = (0..self.vocab)
                        .into_par_iter()
                        .map(|v| {
                            let brow = &b[v * r..(v + 1) * r];
                            let mut bias = 0.0f32;
                            for (bb, aa) in brow.iter().zip(a_row) {
                                bias += bb * aa;
                            }
                            base[v] + bias
                        })
                        .collect();
                    argmax(&col)
                }
            };
            // confidence: sigmoid(conf_w · [hidden_i; A[prev]] + b)
            let h = &out.hidden[i];
            let mut cacc = self.conf_b;
            for (wv, hv) in self.conf_w[..self.n_embd].iter().zip(h) {
                cacc += wv * hv;
            }
            for (wv, av) in self.conf_w[self.n_embd..].iter().zip(a_row) {
                cacc += wv * av;
            }
            confs.push(sigmoid(cacc));
            drafts.push(best);
            prev = best;
        }
        (drafts, confs)
    }
}
