//! Issue 1179 — the qwen3.8 specdec lane's shared half: the uzu DFlash
//! speculator consumed as a FROZEN vendor artifact (Path A of
//! `.issues/1179_qwen38_specdec_verify_lane.md`).
//!
//! **Provenance + posture.** The artifact is trymirai's
//! `trymirai/Qwen3.8-27B-M` `speculator/` publication (apache-2.0):
//! `model.safetensors` + `config.json` + `shapes.json`, blake3-pinned at
//! load (A1/A8 — weights-from-disk, never committed to any repo). The
//! consuming format was read from the MIT uzu engine source at pin
//! `1866d189` (`encodable_block/dflash.rs`, `weaver.rs`,
//! `speculators/dflash_tfm.rs`, the CPU kernels under `backends/cpu/kernel/`).
//! No training happens here — a loader reading a frozen vendor artifact
//! serves the modelless mandate.
//!
//! **Format (all read from the pin):** every draft Linear is stored as
//! uint4 affine (group 32 along the input dim, per-(group, out) bf16 scale,
//! zero-point nibble-packed 2-per-byte along the output dim) plus
//! incoherence processing (HybridSpec, block 32, input_output): the runtime
//! math is `y = D_out·H32·Ŵ·H32·D_in·x` per 32-block. This loader FOLDS the
//! basis change into the weight at load time (`W_orig = D_out H Ŵ H D_in`,
//! H32 the normalized Walsh–Hadamard, orthogonal symmetric), so the runtime
//! draft is plain f32 matmuls — mathematically identical to the vendor
//! path up to the vendor's int8 KV rounding (orthogonal-basis internal,
//! score-preserving in exact arithmetic; disclosed in the T1 record).
//!
//! **Draft semantics (uzu-exact):** block of `block_size` rows — row 0 =
//! the last accepted target token's embedding (from the TARGET's table),
//! rows 1.. = the mask token; rows are bidirectionally visible within the
//! block (is_causal=false), the block attends the draft KV cache (the
//! projected target hidden states of every accepted token, appended by
//! [`DFlashCpu::accept`]) under the ±window/2 sliding rule; RoPE half-split
//! base 1e7 over the full 128-dim head at ABSOLUTE draft-context positions;
//! output norm (ShortcutMode::Add) normalizes residual + branch; the
//! readout rows 1.. predict the tokens at the next positions through the
//! TARGET's lm_head (not reproduced here — the harness owns it).
//!
//! T1 runs the ARGMAX-chain tree shape (uzu `DFlashTfmTreeConstructionMethod::Argmax`):
//! one block forward, argmax per readout row, the 15 chain candidates
//! verified by the target in one keyed pass (`qwen38_spec::keyed_accept`).

#![cfg(feature = "qwen38_specdec")]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use half::bf16;

// ── safetensors reader (the vendor artifact's container) ────────────────

pub struct StFile {
    data: Vec<u8>,
    tensors: BTreeMap<String, StInfo>,
}

struct StInfo {
    dtype: String,
    shape: Vec<u64>,
    start: u64,
    end: u64,
}

impl StFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("st open {path:?}: {e}"))?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)
            .map_err(|e| format!("st header len: {e}"))?;
        let header_len = u64::from_le_bytes(len_buf);
        let mut header = vec![0u8; header_len as usize];
        f.read_exact(&mut header)
            .map_err(|e| format!("st header: {e}"))?;
        let header: serde_json::Value =
            serde_json::from_slice(&header).map_err(|e| format!("st header json: {e}"))?;
        let mut data = Vec::new();
        f.read_to_end(&mut data)
            .map_err(|e| format!("st data: {e}"))?;
        let mut tensors = BTreeMap::new();
        for (k, v) in header.as_object().ok_or("st header not an object")? {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"]
                .as_str()
                .ok_or_else(|| format!("st {k}: dtype"))?
                .to_string();
            let shape = v["shape"]
                .as_array()
                .ok_or_else(|| format!("st {k}: shape"))?
                .iter()
                .map(|x| x.as_u64().ok_or_else(|| format!("st {k}: shape elem")))
                .collect::<Result<Vec<_>, _>>()?;
            let offs = v["data_offsets"]
                .as_array()
                .ok_or_else(|| format!("st {k}: offsets"))?;
            let start = offs[0].as_u64().ok_or_else(|| format!("st {k}: off0"))?;
            let end = offs[1].as_u64().ok_or_else(|| format!("st {k}: off1"))?;
            tensors.insert(
                k.clone(),
                StInfo {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        let _ = header_len;
        Ok(Self { data, tensors })
    }

    pub fn f32_vec_pub(&self, name: &str) -> Result<Vec<f32>, String> {
        self.f32_vec(name)
    }

    fn bytes(&self, name: &str) -> Result<&[u8], String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("st: missing tensor {name}"))?;
        // `self.data` was read starting at the data section (byte 8 +
        // header_len), so tensor offsets index it DIRECTLY — no extra
        // data_offset shift (the double offset read garbage: NaN scales,
        // found by the probe-vs-python cross-read).
        let s = info.start as usize;
        let e = info.end as usize;
        self.data
            .get(s..e)
            .ok_or_else(|| format!("st: {name} out of bounds"))
    }

    fn shape(&self, name: &str) -> Result<Vec<u64>, String> {
        Ok(self
            .tensors
            .get(name)
            .ok_or_else(|| format!("st: missing tensor {name}"))?
            .shape
            .clone())
    }

    fn f32_vec(&self, name: &str) -> Result<Vec<f32>, String> {
        let bytes = self.bytes(name)?;
        match self.tensors.get(name).map(|i| i.dtype.as_str()) {
            Some("F32") => Ok(bytemuck::cast_slice(bytes).to_vec()),
            Some("BF16") => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(bf16::from_le_bytes(*b)))
                .collect()),
            Some("F16") => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(half::f16::from_le_bytes(*b)))
                .collect()),
            other => Err(format!("st: {name} dtype {other:?} not a float type")),
        }
    }

    fn i32_signs(&self, name: &str) -> Result<Vec<f32>, String> {
        let bytes = self.bytes(name)?;
        let raw = bytemuck::cast_slice::<_, i32>(bytes);
        Ok(raw.iter().map(|&s| s as f32).collect())
    }

    fn u8_vec(&self, name: &str) -> Result<&[u8], String> {
        self.bytes(name)
    }
}

// ── the uint4 affine dequant + incoherence fold ─────────────────────────

/// A draft Linear in the ORIGINAL basis: row-major [out][in] f32; the
/// runtime is `y[row] = Σ_i w[row][i] · x[i]`.
pub struct W4 {
    pub w: Vec<f32>,
    pub out: usize,
    pub inn: usize,
}

impl W4 {
    /// `y` for one row of x — the caller batches over block rows.
    #[inline]
    pub fn row(&self, x: &[f32]) -> Vec<f32> {
        y_view(self, x)
    }

    /// Many rows at once (x rows are contiguous [rows][in]).
    pub fn rows(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n * self.out];
        for r in 0..n {
            let xr = &x[r * self.inn..(r + 1) * self.inn];
            for o in 0..self.out {
                let wr = &self.w[o * self.inn..(o + 1) * self.inn];
                let mut acc = 0.0f32;
                for i in 0..self.inn {
                    acc += wr[i] * xr[i];
                }
                y[r * self.out + o] = acc;
            }
        }
        y
    }
}

/// Normalized 32-wide Walsh–Hadamard (orthogonal, symmetric).
fn hadamard32() -> [[f32; 32]; 32] {
    let mut h = [[0.0f32; 32]; 32];
    h[0][0] = 1.0;
    let mut size = 1usize;
    while size < 32 {
        for r in 0..size {
            for c in 0..size {
                let v = h[r][c];
                h[r][c + size] = v;
                h[r + size][c] = v;
                h[r + size][c + size] = -v;
            }
        }
        size *= 2;
    }
    let s = 1.0 / (32.0f32).sqrt();
    for row in &mut h {
        for v in row.iter_mut() {
            *v *= s;
        }
    }
    h
}

/// Fold the vendor's incoherence basis change into `w` (in place):
/// `w ← D_out · H32 · w · H32 · D_in`, blockwise along each axis (the
/// runtime path's input prep is `H32(signs_in ⊙ x)`, the output side
/// `signs_out ⊙ H32(·)` — uzu `rht_wrapper.rs` + `activation_transform.rs`
/// at the pin; this load-time fold is the exact algebraic inverse).
fn fold_incoherence(w: &mut [f32], out: usize, inn: usize, in_signs: &[f32], out_signs: &[f32]) {
    assert!(
        inn.is_multiple_of(32) && out.is_multiple_of(32),
        "fold: dims must be multiples of 32"
    );
    let h = hadamard32();
    // Right side: per 32-column block, w_blk ← w_blk · H32, then column l
    // scales by in_signs[block*32 + l]. (H symmetric: (w·H)[j] = Σ_k w[k]·H[j][k].)
    for blk in 0..inn / 32 {
        let c0 = blk * 32;
        for o in 0..out {
            let row = &mut w[o * inn..(o + 1) * inn];
            let old: [f32; 32] = row[c0..c0 + 32].try_into().expect("32 cols");
            for j in 0..32 {
                let mut acc = 0.0f32;
                for k in 0..32 {
                    acc += old[k] * h[j][k];
                }
                row[c0 + j] = acc * in_signs[c0 + j];
            }
        }
    }
    // Left side: per 32-row block, w_blk ← H32 · w_blk ACROSS ALL COLUMNS
    // (the row-mix is independent of the column index; the first draft of
    // this loop only transformed columns 0..32 and corrupted every Linear —
    // found by verify_fold, max|vendor − folded| = 6.26 → 1e-4 after).
    {
        let mut block = vec![0.0f32; 32 * inn];
        for blk in 0..out / 32 {
            let r0 = blk * 32;
            for r in 0..32 {
                block[r * inn..(r + 1) * inn]
                    .copy_from_slice(&w[(r0 + r) * inn..(r0 + r + 1) * inn]);
            }
            for r in 0..32 {
                for c in 0..inn {
                    let mut acc = 0.0f32;
                    for k in 0..32 {
                        acc += h[r][k] * block[k * inn + c];
                    }
                    w[(r0 + r) * inn + c] = acc;
                }
            }
        }
    }
    // Output signs (D_out) — one full-row pass.
    for r in 0..out {
        let s = out_signs[r];
        if s != 1.0 {
            for c in 0..inn {
                w[r * inn + c] *= s;
            }
        }
    }
}

/// Dequantize one uint4-affine vendor Linear (group 32 along `inn`,
/// per-(group, out) bf16 scale, zero-point nibble-packed along `out`) and
/// fold the incoherence basis into the ORIGINAL-basis weight.
fn load_linear(st: &StFile, prefix: &str, out: usize, inn: usize) -> Result<W4, String> {
    let codes = st.u8_vec(&format!("{prefix}.weights.quantized.weights"))?;
    let scales = st.f32_vec(&format!("{prefix}.weights.quantized.scales"))?;
    let zps = st.u8_vec(&format!("{prefix}.weights.quantized.zero_points"))?;
    let in_signs = st.i32_signs(&format!("{prefix}.weights.incoherence_signs.input_signs"))?;
    let out_signs = st.i32_signs(&format!("{prefix}.weights.incoherence_signs.output_signs"))?;
    // Shapes from the pin: codes [out, inn/2], scales [inn/32, out],
    // zero_points [inn/32, out/2].
    let codes_shape = st.shape(&format!("{prefix}.weights.quantized.weights"))?;
    if codes_shape != [out as u64, (inn / 2) as u64] {
        return Err(format!(
            "{prefix}: codes shape {:?} != [{out}, {}]",
            codes_shape,
            inn / 2
        ));
    }
    let mut w = vec![0.0f32; out * inn];
    for o in 0..out {
        for i in 0..inn {
            let byte = codes[o * (inn / 2) + i / 2];
            let nib = if i.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let g = i / 32;
            let scale = scales[g * out + o];
            let zp_flat = g * (out / 2) + o / 2;
            let zp_byte = zps[zp_flat];
            let zp = if o.is_multiple_of(2) {
                zp_byte & 0x0F
            } else {
                zp_byte >> 4
            };
            w[o * inn + i] = scale * (nib as f32 - zp as f32);
        }
    }
    fold_incoherence(&mut w, out, inn, &in_signs, &out_signs);
    Ok(W4 { w, out, inn })
}

/// Dequantize one uint4-affine vendor Linear (group 32 along `inn`,
/// per-(group, out) bf16 scale, zero-point nibble-packed along `out`) and
/// fold the incoherence basis into the ORIGINAL-basis weight.
pub type LinearRawParts = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
pub type Layer0Dump = (Vec<f32>, Vec<f32>, Vec<f32>);

pub fn load_linear_raw(
    st: &StFile,
    prefix: &str,
    out: usize,
    inn: usize,
) -> Result<LinearRawParts, String> {
    let codes = st
        .u8_vec(&format!("{prefix}.weights.quantized.weights"))?
        .to_vec();
    let scales = st.f32_vec(&format!("{prefix}.weights.quantized.scales"))?;
    let zps = st
        .u8_vec(&format!("{prefix}.weights.quantized.zero_points"))?
        .to_vec();
    let in_signs = st.i32_signs(&format!("{prefix}.weights.incoherence_signs.input_signs"))?;
    let out_signs = st.i32_signs(&format!("{prefix}.weights.incoherence_signs.output_signs"))?;
    let mut w = vec![0.0f32; out * inn];
    for o in 0..out {
        for i in 0..inn {
            let byte = codes[o * (inn / 2) + i / 2];
            let nib = if i.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let g = i / 32;
            let scale = scales[g * out + o];
            let zp_flat = g * (out / 2) + o / 2;
            let zp_byte = zps[zp_flat];
            let zp = if o.is_multiple_of(2) {
                zp_byte & 0x0F
            } else {
                zp_byte >> 4
            };
            w[o * inn + i] = scale * (nib as f32 - zp as f32);
        }
    }
    Ok((w, scales, in_signs, out_signs))
}

/// Diag: verify the load-time incoherence fold against the vendor's runtime
/// step path (signs⊙x → H32 → Ŵ → H32 → signs_out) on random inputs.
pub fn verify_fold(st: &StFile, cfg: &VendorConfig) -> String {
    let d = cfg.draft.model_dim;
    let prefix = "speculator.draft_model.layers.0.mixer.qkvg_projection";
    let (w_hat, _scales, in_signs, out_signs) = load_linear_raw(
        st,
        prefix,
        DRAFT_HEADS * DRAFT_HEAD_DIM + 2 * DRAFT_GROUPS * DRAFT_HEAD_DIM,
        d,
    )
    .expect("raw");
    let folded = load_linear(
        st,
        prefix,
        DRAFT_HEADS * DRAFT_HEAD_DIM + 2 * DRAFT_GROUPS * DRAFT_HEAD_DIM,
        d,
    )
    .expect("folded");
    let mut max_err = 0.0f32;
    for trial in 0..3 {
        let seed = (trial + 1) as f32 * 0.1234567;
        let x: Vec<f32> = (0..d)
            .map(|i| ((i * 37 + trial * 911) % 251) as f32 / 251.0 * seed - 0.3)
            .collect();
        // vendor step path
        let mut xp: Vec<f32> = x.clone();
        for blk in 0..d / 32 {
            let c0 = blk * 32;
            let mut stripe: [f32; 32] = [0.0; 32];
            for l in 0..32 {
                stripe[l] = xp[c0 + l] * in_signs[c0 + l];
            }
            // hadamard_transform (walsh then 1/sqrt(32))
            let mut s = 1usize;
            while s < 32 {
                for lane in 0..32 {
                    if lane & s == 0 {
                        let a = stripe[lane];
                        let b = stripe[lane | s];
                        stripe[lane] = a + b;
                        stripe[lane | s] = a - b;
                    }
                }
                s <<= 1;
            }
            let sc = 1.0 / (32.0f32).sqrt();
            for l in 0..32 {
                xp[c0 + l] = stripe[l] * sc;
            }
        }
        let out_dim = folded.out;
        let mut y_hat = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += w_hat[o * d + i] * xp[i];
            }
            y_hat[o] = acc;
        }
        // output: signs_out ⊙ H32 per block
        let mut y_vendor = y_hat.clone();
        for blk in 0..out_dim / 32 {
            let r0 = blk * 32;
            let mut stripe: [f32; 32] = [0.0; 32];
            stripe.copy_from_slice(&y_vendor[r0..r0 + 32]);
            let mut s = 1usize;
            while s < 32 {
                for lane in 0..32 {
                    if lane & s == 0 {
                        let a = stripe[lane];
                        let b = stripe[lane | s];
                        stripe[lane] = a + b;
                        stripe[lane | s] = a - b;
                    }
                }
                s <<= 1;
            }
            let sc = 1.0 / (32.0f32).sqrt();
            for l in 0..32 {
                y_vendor[r0 + l] = stripe[l] * sc * out_signs[r0 + l];
            }
        }
        // folded path
        let y_folded = folded.row(&x);
        for o in 0..out_dim {
            max_err = max_err.max((y_vendor[o] - y_folded[o]).abs());
        }
    }
    format!("max|vendor - folded| = {max_err:.6}")
}

/// A plain BF16 [out][in]-shaped or [d, k] vendor matrix (the convs).
fn load_bf16(st: &StFile, name: &str) -> Result<Vec<f32>, String> {
    st.f32_vec(name)
}

// ── the DFlash draft model (uzu config exactly) ─────────────────────────

pub const DRAFT_LAYERS: usize = 5;
pub const DRAFT_HEADS: usize = 32;
pub const DRAFT_GROUPS: usize = 8;
pub const DRAFT_HEAD_DIM: usize = 128;
pub const DRAFT_WINDOW: usize = 4095;
pub const DRAFT_SCALE: f32 = 0.08838835;
pub const DRAFT_EPS: f32 = 1e-6;
pub const DRAFT_ROPE_BASE: f32 = 10_000_000.0;

pub struct DFlashLayerWeights {
    pub pre_mixer_norm: Vec<f32>,
    pub pre_mlp_norm: Vec<f32>,
    pub mixer_conv_pre: Vec<f32>, // [5120, 2]
    pub mixer_conv_post: Vec<f32>,
    pub mixer_kernel_proj: W4, // 5120 → 2·C (C = 2·(5120/16) = 640)
    pub qkvg: W4,              // 5120 → 6144
    pub query_norm: Vec<f32>,  // 128
    pub key_norm: Vec<f32>,    // 128
    pub attn_out: W4,          // 4096 → 5120
    pub mlp_conv_pre: Vec<f32>,
    pub mlp_conv_post: Vec<f32>,
    pub mlp_kernel_proj: W4,
    pub mlp_up: W4,   // 5120 → 34816 (gate | up)
    pub mlp_down: W4, // 17408 → 5120
}

pub struct DFlashWeights {
    pub model_dim: usize,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub context_projection: W4, // model_dim·L → model_dim
    pub context_norm: Vec<f32>,
    pub state_kv_projection: W4, // model_dim → layers·2·groups·head_dim
    pub output_norm: Vec<f32>,
    pub layers: Vec<DFlashLayerWeights>,
}

/// Per-token draft coefficients for the separable causal convs (pre + post
/// columns packed in one kernel-projection row, the vendor layout).
struct ConvCoefs {
    /// [rows][2·C]
    raw: Vec<f32>,
    coef_count: usize,
}

impl ConvCoefs {
    fn cols(&self, row: usize, post: bool) -> &[f32] {
        let off = if post { self.coef_count } else { 0 };
        &self.raw
            [row * 2 * self.coef_count + off..row * 2 * self.coef_count + off + self.coef_count]
    }
}

fn conv_forward(
    x: &mut Vec<f32>,
    rows: usize,
    dim: usize,
    weights: &[f32], // [dim, K]
    k: usize,
    coefs: &ConvCoefs,
    post: bool,
) {
    // uzu separable_causal_conv kernel at the pin: out[t,c] = Σ_j
    // x[t-j, c]·(weights[c, K-1-j] + coef[t, j·G + c/G]), j over
    // 0..min(K, t+1). Coefficient column offset picks pre (0) or post (C).
    let groups = dim / 16; // conv_group_size 16
    let mut out = vec![0.0f32; rows * dim];
    for t in 0..rows {
        let coef = coefs.cols(t, post);
        for c in 0..dim {
            let g = c / 16;
            let mut acc = 0.0f32;
            for j in 0..k.min(t + 1) {
                let w = weights[c * k + (k - 1 - j)];
                let delta = coef[j * groups + g];
                acc += x[(t - j) * dim + c] * (w + delta);
            }
            out[t * dim + c] = acc;
        }
    }
    *x = out;
}

fn rmsnorm(x: &mut [f32], rows: usize, dim: usize, scales: &[f32]) {
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / dim as f32 + DRAFT_EPS).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * inv * scales[i];
        }
    }
}

pub fn rope_tables_for_probe(positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
    rope_tables(positions)
}

fn rope_tables(positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
    // uzu PrecalculatedRoPE (UnscaledRoPEConfig, head_dim 128): half-split
    // pairing (c, c+64), inv_freq = base^(-2c/128) over pairs c = 0..64,
    // duplicated across both halves.
    let dim = DRAFT_HEAD_DIM;
    let half = dim / 2;
    let mut cos = vec![0.0f32; positions.len() * dim];
    let mut sin = vec![0.0f32; positions.len() * dim];
    for (ti, &pos) in positions.iter().enumerate() {
        for p in 0..half {
            let freq = DRAFT_ROPE_BASE.powf(-(2.0 * p as f32) / dim as f32);
            let ang = pos as f32 * freq;
            let (s, c) = ang.sin_cos();
            cos[ti * dim + p] = c;
            cos[ti * dim + p + half] = c;
            sin[ti * dim + p] = s;
            sin[ti * dim + p + half] = s;
        }
    }
    (cos, sin)
}

fn apply_rope(h: &mut [f32], rows: usize, cos: &[f32], sin: &[f32]) {
    let dim = DRAFT_HEAD_DIM;
    let half = dim / 2;
    for r in 0..rows {
        for c in 0..dim {
            let paired = if c < half { c + half } else { c - half };
            let signed = if c < half { -1.0 } else { 1.0 };
            let v = h[r * dim + c];
            let pv = signed * h[r * dim + paired];
            h[r * dim + c] = v * cos[r * dim + c] + pv * sin[r * dim + c];
        }
    }
}

/// The CPU f32 DFlash draft: weights + the accept-side KV state.
pub struct DFlashCpu {
    pub w: DFlashWeights,
    /// draft KV per layer: [ctx][2·groups·head_dim] (K | V, head-major
    /// [groups][head_dim] per side).
    kv: Vec<Vec<f32>>,
    ctx_len: usize,
    capacity: usize,
}

impl DFlashCpu {
    pub fn load(st: &StFile, cfg: &VendorConfig) -> Result<Self, String> {
        let d = cfg.draft.model_dim;
        let prefix = "speculator.draft_model";
        let context_projection = load_linear(
            st,
            &format!("{prefix}.context_projection"),
            d,
            d * cfg.draft.target_layer_ids.len(),
        )?;
        let context_norm = st.f32_vec(&format!("{prefix}.context_norm.scales"))?;
        let layer_kv_dim = 2 * DRAFT_GROUPS * DRAFT_HEAD_DIM;
        let state_kv_projection = load_linear(
            st,
            &format!("{prefix}.state_kv_projection"),
            DRAFT_LAYERS * layer_kv_dim,
            d,
        )?;
        let output_norm = st.f32_vec(&format!("{prefix}.output_norm.scales"))?;
        let mut layers = Vec::new();
        for li in 0..cfg.draft.layer_count {
            let lp = format!("{prefix}.layers.{li}");
            let coef_count = cfg.draft.conv_kernel_size * (d / cfg.draft.conv_group_size);
            layers.push(DFlashLayerWeights {
                pre_mixer_norm: st.f32_vec(&format!("{lp}.pre_mixer_norm.scales"))?,
                pre_mlp_norm: st.f32_vec(&format!("{lp}.pre_mlp_norm.scales"))?,
                mixer_conv_pre: load_bf16(st, &format!("{lp}.mixer_conv.pre_conv.weights"))?,
                mixer_conv_post: load_bf16(st, &format!("{lp}.mixer_conv.post_conv.weights"))?,
                mixer_kernel_proj: load_linear(
                    st,
                    &format!("{lp}.mixer_conv.kernel_projection"),
                    2 * coef_count,
                    d,
                )?,
                qkvg: load_linear(
                    st,
                    &format!("{lp}.mixer.qkvg_projection"),
                    DRAFT_HEADS * DRAFT_HEAD_DIM + 2 * DRAFT_GROUPS * DRAFT_HEAD_DIM,
                    d,
                )?,
                query_norm: st.f32_vec(&format!("{lp}.mixer.query_norm.scales"))?,
                key_norm: st.f32_vec(&format!("{lp}.mixer.key_norm.scales"))?,
                attn_out: load_linear(
                    st,
                    &format!("{lp}.mixer.out_projection"),
                    d,
                    DRAFT_HEADS * DRAFT_HEAD_DIM,
                )?,
                mlp_conv_pre: load_bf16(st, &format!("{lp}.mlp_conv.pre_conv.weights"))?,
                mlp_conv_post: load_bf16(st, &format!("{lp}.mlp_conv.post_conv.weights"))?,
                mlp_kernel_proj: load_linear(
                    st,
                    &format!("{lp}.mlp_conv.kernel_projection"),
                    2 * coef_count,
                    d,
                )?,
                mlp_up: load_linear(
                    st,
                    &format!("{lp}.mlp.up_projection"),
                    2 * cfg.draft.hidden_dim,
                    d,
                )?,
                mlp_down: load_linear(
                    st,
                    &format!("{lp}.mlp.down_projection"),
                    d,
                    cfg.draft.hidden_dim,
                )?,
            });
        }
        Ok(Self {
            w: DFlashWeights {
                model_dim: d,
                block_size: cfg.draft.block_size,
                mask_token_id: cfg.draft.mask_token_id,
                target_layer_ids: cfg.draft.target_layer_ids.clone(),
                context_projection,
                context_norm,
                state_kv_projection,
                output_norm,
                layers,
            },
            kv: (0..DRAFT_LAYERS).map(|_| Vec::new()).collect(),
            ctx_len: 0,
            capacity: 0,
        })
    }

    pub fn context_length(&self) -> usize {
        self.ctx_len
    }

    pub fn reset(&mut self) {
        for kv in self.kv.iter_mut() {
            kv.clear();
        }
        self.ctx_len = 0;
        self.capacity = 0;
    }

    /// uzu encode_accept: append the ACCEPTED tokens' target features to
    /// the draft KV. `feats_rows` = one packed [model_dim·L] row per token
    /// (the captured target layers concatenated in target_layer_ids order).
    ///
    /// The projected per-layer rows go through the attention PREPARE before
    /// caching (uzu `append_projected_kv` → `prepare`): K is RMS-normed
    /// per head (key_norm) and RoPE'd at the token's ABSOLUTE draft
    /// position; V is cached raw (no value norm in this config). The draft
    /// attention then reads the prefix rows as-is.
    pub fn accept(&mut self, feats_rows: &[f32], n: usize) -> Result<(), String> {
        let d = self.w.model_dim;
        let captured = self.w.target_layer_ids.len();
        let layer_kv_dim = 2 * DRAFT_GROUPS * DRAFT_HEAD_DIM;
        let kv_dim = DRAFT_GROUPS * DRAFT_HEAD_DIM;
        if self.ctx_len + n > self.capacity {
            self.capacity = (self.ctx_len + n).next_power_of_two().max(1024);
            for kv in self.kv.iter_mut() {
                kv.resize(self.capacity * layer_kv_dim, 0.0);
            }
        }
        let positions: Vec<usize> = (self.ctx_len..self.ctx_len + n).collect();
        let (cos, sin) = rope_tables(&positions);
        for t in 0..n {
            let row = &feats_rows[t * d * captured..(t + 1) * d * captured];
            let mut normed = self.w.context_projection.row(row);
            rmsnorm(&mut normed, 1, d, &self.w.context_norm);
            let kv_all = self.w.state_kv_projection.row(&normed);
            for (li, kv) in self.kv.iter_mut().enumerate() {
                let src = &kv_all[li * layer_kv_dim..(li + 1) * layer_kv_dim];
                let dst = &mut kv[self.ctx_len * layer_kv_dim..(self.ctx_len + 1) * layer_kv_dim];
                dst[..kv_dim].copy_from_slice(&src[..kv_dim]);
                dst[kv_dim..].copy_from_slice(&src[kv_dim..]);
                // K: per-head RMS norm (key_norm) + PER-HEAD rope at this
                // position (the rope is 128-dim per head — the same bug the
                // draft block had, fixed here too).
                let k_row = &mut dst[..kv_dim];
                for head in 0..DRAFT_GROUPS {
                    head_rmsnorm(
                        &mut k_row[head * DRAFT_HEAD_DIM..(head + 1) * DRAFT_HEAD_DIM],
                        &self.w.layers[li].key_norm,
                    );
                    apply_rope(
                        &mut k_row[head * DRAFT_HEAD_DIM..(head + 1) * DRAFT_HEAD_DIM],
                        1,
                        &cos[t * DRAFT_HEAD_DIM..(t + 1) * DRAFT_HEAD_DIM],
                        &sin[t * DRAFT_HEAD_DIM..(t + 1) * DRAFT_HEAD_DIM],
                    );
                }
                // V: raw (value_norm absent in this config).
            }
            self.ctx_len += 1;
        }
        Ok(())
    }

    /// Diag: the folded mixer kernel_projection weights of layer 0 (for the
    /// uzu-probe conv glue).
    pub fn dump_l0_mixer_kernel_proj(&self) -> &[f32] {
        &self.w.layers[0].mixer_kernel_proj.w
    }

    /// Diag: run ONLY layer 0 (uzu-probe twin — the same input: block
    /// embeddings, rope 0..16, stateless no-KV) and return (hidden,
    /// shortcut) as f32. `mlp_gate_first` swaps the fused-up halves (the
    /// [gate|value] convention) for the variant test.
    pub fn dump_layer0(
        &mut self,
        t_prev_emb: &[f32],
        mask_emb: &[f32],
        mlp_gate_first: bool,
    ) -> Result<Layer0Dump, String> {
        let d = self.w.model_dim;
        let rows = self.w.block_size;
        let mut hidden = vec![0.0f32; rows * d];
        hidden[..d].copy_from_slice(t_prev_emb);
        for r in 1..rows {
            hidden[r * d..(r + 1) * d].copy_from_slice(mask_emb);
        }
        let mut shortcut = hidden.clone();
        let positions: Vec<usize> = (0..rows).collect();
        let (cos, sin) = rope_tables(&positions);
        let coef_count = 2 * (d / 16);
        let lw = &self.w.layers[0];
        // L0 pre_mixer_norm: Copy.
        shortcut.copy_from_slice(&hidden);
        let mut h = hidden.clone();
        rmsnorm(&mut h, rows, d, &lw.pre_mixer_norm);
        let mut coefs_raw = lw.mixer_kernel_proj.rows(&h, rows);
        let coefs = ConvCoefs {
            raw: std::mem::take(&mut coefs_raw),
            coef_count,
        };
        conv_forward(&mut h, rows, d, &lw.mixer_conv_pre, 2, &coefs, false);
        h = self.attention_block(0, &h, rows, &cos, &sin);
        conv_forward(&mut h, rows, d, &lw.mixer_conv_post, 2, &coefs, true);
        // pre_mlp_norm (Add).
        for i in 0..h.len() {
            shortcut[i] = h[i] + shortcut[i];
        }
        let mut h2 = shortcut.clone();
        rmsnorm(&mut h2, rows, d, &lw.pre_mlp_norm);
        let mut coefs_raw2 = lw.mlp_kernel_proj.rows(&h2, rows);
        let coefs2 = ConvCoefs {
            raw: std::mem::take(&mut coefs_raw2),
            coef_count,
        };
        conv_forward(&mut h2, rows, d, &lw.mlp_conv_pre, 2, &coefs2, false);
        let up = lw.mlp_up.rows(&h2, rows);
        let hidden_dim = lw.mlp_up.out / 2;
        let mut act = vec![0.0f32; rows * hidden_dim];
        for r in 0..rows {
            for i in 0..hidden_dim {
                let (v, g) = if mlp_gate_first {
                    (
                        up[r * 2 * hidden_dim + hidden_dim + i],
                        up[r * 2 * hidden_dim + i],
                    )
                } else {
                    (
                        up[r * 2 * hidden_dim + i],
                        up[r * 2 * hidden_dim + hidden_dim + i],
                    )
                };
                act[r * hidden_dim + i] = g / (1.0 + (-g).exp()) * v;
            }
        }
        let mut h3 = lw.mlp_down.rows(&act, rows);
        conv_forward(&mut h3, rows, d, &lw.mlp_conv_post, 2, &coefs2, true);
        Ok((h3, shortcut, h2))
    }

    /// uzu encode_draft (Argmax chain shape): ONE block forward over
    /// `block_size` rows (row 0 = `t_prev_emb`, the rest = the mask row);
    /// returns the output-normed hidden [block][d] (the caller reads out
    /// rows 1.. through the target lm_head).
    pub fn draft(&mut self, t_prev_emb: &[f32], mask_emb: &[f32]) -> Result<Vec<f32>, String> {
        self.draft_traced(t_prev_emb, mask_emb, false)
            .map(|(h, _)| h)
    }

    /// [`Self::draft`] with a per-stage max/nan trace (diag-only; prints).
    pub fn draft_traced(
        &mut self,
        t_prev_emb: &[f32],
        mask_emb: &[f32],
        trace: bool,
    ) -> Result<(Vec<f32>, Vec<String>), String> {
        let mut trace_lines = Vec::new();
        let d = self.w.model_dim;
        let b = self.w.block_size;
        let rows = b;
        let mut hidden = vec![0.0f32; rows * d];
        hidden[..d].copy_from_slice(t_prev_emb);
        for r in 1..rows {
            hidden[r * d..(r + 1) * d].copy_from_slice(mask_emb);
        }
        let mut shortcut = hidden.clone();
        // Block positions = absolute draft-context positions.
        let positions: Vec<usize> = (self.ctx_len..self.ctx_len + rows).collect();
        let (cos, sin) = rope_tables(&positions);
        let coef_count = 2 * (d / 16);
        let stage = |name: &str, v: &[f32], lines: &mut Vec<String>| {
            if trace {
                let mut mx = 0.0f32;
                let mut nan = 0usize;
                for &x in v {
                    if x.is_nan() {
                        nan += 1;
                    } else {
                        mx = mx.max(x.abs());
                    }
                }
                lines.push(format!("{name}: max={mx:.4} nan={nan} len={}", v.len()));
            }
        };
        stage("emb", &hidden, &mut trace_lines);

        for (li, lw) in self.w.layers.iter().enumerate() {
            // pre_mixer_norm: layer 0 = Copy, layers 1+ = Add (uzu
            // transformer_layer.rs at the pin).
            if li == 0 {
                shortcut.copy_from_slice(&hidden);
            } else {
                for i in 0..hidden.len() {
                    shortcut[i] = hidden[i] + shortcut[i];
                }
            }
            let mut h = hidden.clone();
            rmsnorm(&mut h, rows, d, &lw.pre_mixer_norm);
            stage(&format!("L{li} premix_norm"), &h, &mut trace_lines);

            let mut coefs_raw = lw.mixer_kernel_proj.rows(&h, rows);
            stage(&format!("L{li} mixer_coefs"), &coefs_raw, &mut trace_lines);
            let coefs = ConvCoefs {
                raw: std::mem::take(&mut coefs_raw),
                coef_count,
            };
            conv_forward(&mut h, rows, d, &lw.mixer_conv_pre, 2, &coefs, false);
            stage(&format!("L{li} pre_conv"), &h, &mut trace_lines);

            h = self.attention_block(li, &h, rows, &cos, &sin);
            stage(&format!("L{li} attention"), &h, &mut trace_lines);
            conv_forward(&mut h, rows, d, &lw.mixer_conv_post, 2, &coefs, true);
            stage(&format!("L{li} post_conv"), &h, &mut trace_lines);

            // pre_mlp_norm (Add): s = h + shortcut; shortcut = s; norm(s).
            for i in 0..h.len() {
                shortcut[i] = h[i] + shortcut[i];
            }
            let mut h2 = shortcut.clone();
            rmsnorm(&mut h2, rows, d, &lw.pre_mlp_norm);
            stage(&format!("L{li} premlp_norm"), &h2, &mut trace_lines);

            let mut coefs_raw2 = lw.mlp_kernel_proj.rows(&h2, rows);
            let coefs2 = ConvCoefs {
                raw: std::mem::take(&mut coefs_raw2),
                coef_count,
            };
            conv_forward(&mut h2, rows, d, &lw.mlp_conv_pre, 2, &coefs2, false);
            // DenseMLP: fused up = [value | gate] — the uzu gated_act_mul
            // FP path runs the INTERLEAVED variant (value_operand=None ⇒
            // interleaved: gate at base+H+gated, value at base+gated), so
            // silu(gate)·value reads the SECOND half as the gate. (uzu
            // gate_act_mul.rs at the pin; the GGUF [gate|up] convention
            // does not transfer to the mirai-m export.)
            let up = lw.mlp_up.rows(&h2, rows);
            let hidden_dim = lw.mlp_up.out / 2;
            let mut act = vec![0.0f32; rows * hidden_dim];
            for r in 0..rows {
                for i in 0..hidden_dim {
                    let v = up[r * 2 * hidden_dim + i];
                    let g = up[r * 2 * hidden_dim + hidden_dim + i];
                    act[r * hidden_dim + i] = g / (1.0 + (-g).exp()) * v;
                }
            }
            let mut h3 = lw.mlp_down.rows(&act, rows);
            stage(&format!("L{li} mlp_down"), &h3, &mut trace_lines);
            conv_forward(&mut h3, rows, d, &lw.mlp_conv_post, 2, &coefs2, true);
            stage(&format!("L{li} mlp_postconv"), &h3, &mut trace_lines);
            hidden = h3;
        }
        // output_norm (Add): s = hidden + shortcut; norm(s).
        for i in 0..hidden.len() {
            hidden[i] += shortcut[i];
        }
        rmsnorm(&mut hidden, rows, d, &self.w.output_norm);
        stage("out_norm", &hidden, &mut trace_lines);
        Ok((hidden, trace_lines))
    }

    /// One draft layer's attention over [prefix KV; block] — bidirectional
    /// within the block (is_causal=false), the prefix visible under the
    /// ±window/2 rule, GQA 32q/8kv, per-head q/k RMS norms, half-split rope
    /// (uzu `should_use_key` + `attention_prepare` at the pin).
    fn attention_block(
        &self,
        li: usize,
        h: &[f32],
        rows: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Vec<f32> {
        let lw = &self.w.layers[li];
        let hd = DRAFT_HEAD_DIM;
        let qh = DRAFT_HEADS;
        let kvh = DRAFT_GROUPS;
        let layer_kv_dim = 2 * kvh * hd;
        let gqa = qh / kvh;
        let prefix = self.ctx_len;
        let half_win = DRAFT_WINDOW / 2;

        let qkvg = lw.qkvg.rows(h, rows);
        let q_dim = qh * hd;
        let kv_dim = kvh * hd;
        // Per-head q/k RMS norms, then rope (uzu prepare order: norm → rope).
        // qkvg is ROW-major [rows][q|k|v] (W4::rows) — slice PER ROW; the
        // original section-major cut (q = all rows' q) was the verdict-1 root
        // cause: row 0's q correct, every other row a q/k/v mix, k/v entirely
        // rows 10-15's bytes.
        let mut q = vec![0.0f32; rows * q_dim];
        let mut k = vec![0.0f32; rows * kv_dim];
        let mut v = vec![0.0f32; rows * kv_dim];
        for r in 0..rows {
            let src = &qkvg[r * (q_dim + 2 * kv_dim)..(r + 1) * (q_dim + 2 * kv_dim)];
            q[r * q_dim..(r + 1) * q_dim].copy_from_slice(&src[..q_dim]);
            k[r * kv_dim..(r + 1) * kv_dim].copy_from_slice(&src[q_dim..q_dim + kv_dim]);
            v[r * kv_dim..(r + 1) * kv_dim].copy_from_slice(&src[q_dim + kv_dim..]);
        }
        for r in 0..rows {
            for head in 0..qh {
                head_rmsnorm(
                    &mut q[r * q_dim + head * hd..r * q_dim + (head + 1) * hd],
                    &lw.query_norm,
                );
            }
            for head in 0..kvh {
                head_rmsnorm(
                    &mut k[r * kv_dim + head * hd..r * kv_dim + (head + 1) * hd],
                    &lw.key_norm,
                );
            }
        }
        // RoPE is PER-HEAD (128 dims): heads 1..31 of each row were left
        // un-roped when the whole [rows][q_dim] buffer was passed in once —
        // the bisect's smoking gun (their attention 42.75 vs mine 77.28 on
        // identical input).
        for r in 0..rows {
            let (rc, rs) = (&cos[r * hd..(r + 1) * hd], &sin[r * hd..(r + 1) * hd]);
            for head in 0..qh {
                apply_rope(
                    &mut q[r * q_dim + head * hd..r * q_dim + (head + 1) * hd],
                    1,
                    rc,
                    rs,
                );
            }
            for head in 0..kvh {
                apply_rope(
                    &mut k[r * kv_dim + head * hd..r * kv_dim + (head + 1) * hd],
                    1,
                    rc,
                    rs,
                );
            }
        }

        let mut out = vec![0.0f32; rows * q_dim];
        for r in 0..rows {
            let q_pos = self.ctx_len + r;
            for head in 0..qh {
                let kv_head = head / gqa;
                let qrow = &q[r * q_dim + head * hd..r * q_dim + (head + 1) * hd];
                // Online softmax over prefix + suffix keys.
                let mut max_s = f32::NEG_INFINITY;
                let mut keys: Vec<(usize, f32)> = Vec::new();
                for key in 0..prefix + rows {
                    let (k_pos, krow): (i64, &[f32]) = if key < prefix {
                        (
                            key as i64,
                            &self.kv[li][key * layer_kv_dim + kv_head * hd
                                ..key * layer_kv_dim + (kv_head + 1) * hd],
                        )
                    } else {
                        let sr = key - prefix;
                        (
                            (self.ctx_len + sr) as i64,
                            &k[sr * kv_dim + kv_head * hd..sr * kv_dim + (kv_head + 1) * hd],
                        )
                    };
                    // is_causal=false sliding window: |Δpos| ≤ window/2.
                    if (q_pos as i64 - k_pos).abs() > half_win as i64 {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for c in 0..hd {
                        dot += qrow[c] * krow[c];
                    }
                    let s = dot * DRAFT_SCALE;
                    keys.push((key, s));
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut denom = 0.0f32;
                let mut acc = vec![0.0f32; hd];
                for &(key, s) in &keys {
                    let w = (s - max_s).exp();
                    denom += w;
                    let krow: &[f32] = if key < prefix {
                        &self.kv[li][key * layer_kv_dim + kvh * hd + kv_head * hd
                            ..key * layer_kv_dim + kvh * hd + (kv_head + 1) * hd]
                    } else {
                        let sr = key - prefix;
                        &v[sr * kv_dim + kv_head * hd..sr * kv_dim + (kv_head + 1) * hd]
                    };
                    for c in 0..hd {
                        acc[c] += w * krow[c];
                    }
                }
                if denom > 0.0 {
                    for v in acc.iter_mut() {
                        *v /= denom;
                    }
                }
                out[r * q_dim + head * hd..r * q_dim + (head + 1) * hd].copy_from_slice(&acc);
            }
        }
        lw.attn_out.rows(&out, rows)
    }
}

fn head_rmsnorm(head: &mut [f32], scales: &[f32]) {
    let sum: f32 = head.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sum / head.len() as f32 + DRAFT_EPS).sqrt();
    for (i, v) in head.iter_mut().enumerate() {
        *v = *v * inv * scales[i];
    }
}

/// The vendor input-embedding's output incoherence transform (uzu
/// `input_embedding_lookup` at the pin: per 32-block, dequant row →
/// hadamard32 → × factors[lane], factors = incoherence_signs.output_signs
/// of `decoder.embedding.input_embedding` in the vendor checkpoint). The
/// GGUF rows are the LOGICAL table; the draft consumes the TRANSFORMED
/// lookup rows, so every row passed to [`DFlashCpu::draft`] must run
/// through this first.
pub fn apply_embedding_output_incoherence(row: &mut [f32], signs: &[f32]) {
    assert_eq!(
        row.len() % 32,
        0,
        "embedding transform: dim must be a multiple of 32"
    );
    assert_eq!(row.len(), signs.len(), "embedding transform: signs length");
    for blk in 0..row.len() / 32 {
        let c0 = blk * 32;
        let mut block = [0.0f32; 32];
        block.copy_from_slice(&row[c0..c0 + 32]);
        // walsh in-place, then 1/sqrt(32) — uzu hadamard_transform.
        let mut s = 1usize;
        while s < 32 {
            for lane in 0..32 {
                if lane & s == 0 {
                    let a = block[lane];
                    let b = block[lane | s];
                    block[lane] = a + b;
                    block[lane | s] = a - b;
                }
            }
            s <<= 1;
        }
        let sc = 1.0 / (32.0f32).sqrt();
        for l in 0..32 {
            row[c0 + l] = block[l] * sc * signs[c0 + l];
        }
    }
}

/// The vendor `speculator/config.json` subset the loader consumes.
#[derive(Clone)]
pub struct VendorConfig {
    pub draft: DraftCfg,
}

#[derive(Clone)]
pub struct DraftCfg {
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub layer_count: usize,
    pub conv_kernel_size: usize,
    pub conv_group_size: usize,
}

pub fn parse_vendor_config(json: &str) -> Result<VendorConfig, String> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("speculator config json: {e}"))?;
    let draft = &v["speculator_config"]["draft_config"];
    let g = |obj: &serde_json::Value, key: &str| -> Result<f64, String> {
        obj[key]
            .as_f64()
            .ok_or_else(|| format!("speculator config: missing {key}"))
    };
    let layer_count = draft["layer_configs"]
        .as_array()
        .ok_or("speculator config: layer_configs")?
        .len();
    Ok(VendorConfig {
        draft: DraftCfg {
            model_dim: g(draft, "model_dim")? as usize,
            hidden_dim: g(draft, "hidden_dim")? as usize,
            block_size: g(draft, "block_size")? as usize,
            mask_token_id: g(draft, "mask_token_id")? as u32,
            target_layer_ids: draft["target_layer_ids"]
                .as_array()
                .ok_or("speculator config: target_layer_ids")?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as usize)
                        .ok_or("target_layer_ids elem")
                })
                .collect::<Result<Vec<_>, _>>()?,
            layer_count,
            conv_kernel_size: draft["layer_configs"][0]["mixer_conv_config"]["conv_kernel_size"]
                .as_u64()
                .ok_or("speculator config: conv_kernel_size")?
                as usize,
            conv_group_size: draft["layer_configs"][0]["mixer_conv_config"]["conv_group_size"]
                .as_u64()
                .ok_or("speculator config: conv_group_size")? as usize,
        },
    })
}

fn y_view(l: &W4, x: &[f32]) -> Vec<f32> {
    l.w.chunks_exact(l.inn)
        .map(|wr| wr.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum())
        .collect()
}
