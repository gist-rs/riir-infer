//! Ternary-weight structs for the hybrid DeltaNet/Attention model (Issue 594).
//!
//! Mirrors [`super::weights::QwenDeltaNetWeights`] — the architecture and
//! forward-pass shape are identical. The only difference is the **weight
//! container**: the 12 projection matrices per layer (4 attention + 5
//! `DeltaNet` + 3 MLP) plus the global `wte` and `lm_head` are
//! [`TernaryGroupWeights`] (bit-plane packed, 2.125 bits/w) instead of dense
//! `Vec<f32>`. `RMSNorm` gammas, `conv1d_weight`, `a_log`, `dt_bias`, and
//! `linear_norm` stay dense f32 — they are F32 in the GGUF file and are not
//! projection matvecs.
//!
//! ## Why `wte` and `lm_head` are ternary here (unlike `TernaryLayerWeights`)
//!
//! Ternary-Bonsai-27B stores `token_embd.weight` and `output.weight` as `Q2_0`
//! (type 42), `[5120 × 248320]` each. Dequantizing either to f32 costs ~5.1 GB
//! — together that's ~10 GB, which would make the model *larger* than the 16 GB
//! dense version and defeat the entire 6.67 GiB memory win. So both stay
//! [`TernaryGroupWeights`]:
//! - `lm_head` is a plain matvec → [`simd_ternary_group_matvec`] in one call.
//! - `wte` is a row lookup → [`Self::dequant_wte_row_into`] reads the bit-planes
//!   for one row (the public-field layout makes this possible without a
//!   katgpt-rs change).
//!
//! ## Naming
//!
//! Named `ternary`, not `bitnet` (consistent with [`crate::ternary_layer`]):
//! the format is `Q2_0_g128`, not `BitNet`'s `i2_s`, and the model is `qwen35`.

use crate::types::Config;
use katgpt_core::TernaryGroupWeights;

/// The `DeltaNet` gate projections `in_proj_a` / `in_proj_b` (Issue 980).
///
/// Pre-rotation Bonsai-27B stores them ternary (`Q2_0`). Bonsai 2's folded
/// files carry them DENSE (BF16, type 30, `[n_v_heads × n_embd]`) — the
/// whitepaper's full-precision escape set, neither rotated nor quantized.
/// The field keeps both representations; the forward dispatches per layer.
#[derive(Debug, Clone)]
pub enum GateProjWeights {
    Ternary(TernaryGroupWeights),
    /// Dense `[rows × cols]` row-major (rows = `n_v_heads`, cols = `n_embd`).
    Dense(Vec<f32>, usize, usize),
}

impl GateProjWeights {
    /// Empty ternary placeholder (the unused layer-type's field).
    pub fn empty() -> Self {
        Self::Ternary(TernaryGroupWeights::new(0, 0))
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Ternary(w) => w.rows,
            Self::Dense(_, rows, _) => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Ternary(w) => w.cols,
            Self::Dense(_, _, cols) => *cols,
        }
    }

    /// The ternary arm, if this projection is ternary.
    ///
    /// The fused input-proj hook (Issue 602) passes a/b through as ternary
    /// weights; a rotated or dense-a/b layer bypasses that path.
    pub fn as_ternary(&self) -> Option<&TernaryGroupWeights> {
        match self {
            Self::Ternary(w) => Some(w),
            Self::Dense(_, _, _) => None,
        }
    }

    /// The dense arm's row-major f32 data + shape, if this projection is
    /// dense (Issue 980 T4-ALT — the whole-prefill cudarc lane's
    /// `gemm_dense_ab_batched` consumes it on the PRIMAL normed input).
    pub fn as_dense(&self) -> Option<(&[f32], usize, usize)> {
        match self {
            Self::Ternary(_) => None,
            Self::Dense(data, rows, cols) => Some((data.as_slice(), *rows, *cols)),
        }
    }

    /// `y[..rows] = W @ x[..cols]` — the exact dispatch the forward uses.
    ///
    /// The Ternary arm calls [`katgpt_core::simd_ternary_group_matvec_parallel`]
    /// — the SAME entry point the old `bitlinear` path used — so an old-file
    /// (ternary a/b) forward stays byte-identical after the enum change
    /// (below 256 rows that kernel delegates to serial internally; calling
    /// the parallel entry point preserves the dispatch verbatim).
    pub fn matvec_into(&self, y: &mut [f32], x: &[f32]) {
        match self {
            Self::Ternary(w) => {
                debug_assert_eq!(x.len(), w.cols);
                debug_assert_eq!(y.len(), w.rows);
                katgpt_core::simd_ternary_group_matvec_parallel(w, &x[..w.cols], &mut y[..w.rows]);
            }
            Self::Dense(data, rows, cols) => {
                debug_assert_eq!(x.len(), *cols);
                debug_assert_eq!(y.len(), *rows);
                for (r, yr) in y[..*rows].iter_mut().enumerate() {
                    let row = &data[r * cols..(r + 1) * cols];
                    let mut acc = 0.0f32;
                    for (&wv, &xv) in row.iter().zip(x[..*cols].iter()) {
                        acc += wv * xv;
                    }
                    *yr = acc;
                }
            }
        }
    }

    /// G1 corruption check: ternary arm satisfies the bit-plane invariant;
    /// dense arm is trivially valid (any f32 bits decode).
    pub fn invariants_hold(&self) -> bool {
        match self {
            Self::Ternary(w) => w.rows == 0 || w.invariant_holds(),
            Self::Dense(_, _, _) => true,
        }
    }

    /// Element count (0 for the empty placeholder). The Dense arm counts its
    /// own elements; the Ternary arm counts weight elements (not bit-plane
    /// bytes) so callers can fold it into `ternary_param_count`.
    pub fn param_count(&self) -> usize {
        match self {
            Self::Ternary(w) => w.rows * w.cols,
            Self::Dense(_, rows, cols) => rows * cols,
        }
    }
}

/// Issue 028 T4 S2 — the 10 per-layer projection container: the decode
/// copy's ternary bit-planes OR the q4-class prefill pack's `Q4_K` blocks.
///
/// The disaggregated container (Issue 028) stores one decode copy (ternary)
/// and one prefill copy whose per-layer projections are `Q4_K`; the globals
/// (`wte`/`lm_head`) and the a/b gate projections stay OUTSIDE this enum —
/// the pack byte-copies them (ternary / dense respectively), so they keep
/// their own types and every matvec on them is unchanged.
///
/// The forward dispatches per arm in ONE place ([`bitlinear`] in
/// `ternary_forward` — plus [`Self::matvec_into`], the same dispatch for
/// direct callers): the Ternary arm is the pre-existing SIMD kernel path,
/// the Q4K arm is a fused q4 row-dequant GEMV (`gemv_q4_k_row`) — the q4
/// blocks are never materialized as f32.
#[derive(Debug, Clone)]
pub enum ProjWeights {
    Ternary(TernaryGroupWeights),
    /// `Q4_K` blocks, row-major: `rows` rows of `cols / QK_K` 144-byte
    /// super-blocks (the GGUF wire layout, kept quant-resident).
    Q4K(Vec<crate::quant::q4k::BlockQ4K>, usize, usize),
}

impl ProjWeights {
    /// Empty placeholder (the unused layer-type's field).
    pub fn empty() -> Self {
        Self::Ternary(TernaryGroupWeights::new(0, 0))
    }

    /// Empty-shaped ternary of the given dims (the `zeros` test constructor).
    pub fn ternary(rows: usize, cols: usize) -> Self {
        Self::Ternary(TernaryGroupWeights::new(rows, cols))
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Ternary(w) => w.rows,
            Self::Q4K(_, rows, _) => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Ternary(w) => w.cols,
            Self::Q4K(_, _, cols) => *cols,
        }
    }

    /// The ternary arm, if this projection is ternary. The GPU/act-tap hook
    /// paths consume `&TernaryGroupWeights` — a q4 projection has no such
    /// payload and refuses there (fail-closed, never a silent hook drop).
    pub fn as_ternary(&self) -> Option<&TernaryGroupWeights> {
        match self {
            Self::Ternary(w) => Some(w),
            Self::Q4K(..) => None,
        }
    }

    /// Mutable ternary arm (the refit/requant walkers).
    pub fn as_ternary_mut(&mut self) -> Option<&mut TernaryGroupWeights> {
        match self {
            Self::Ternary(w) => Some(w),
            Self::Q4K(..) => None,
        }
    }

    /// `y[..rows] = W @ x[..cols]` — the exact dispatch the forward uses.
    ///
    /// Ternary: [`katgpt_core::simd_ternary_group_matvec_parallel`] (the
    /// pre-existing entry point). Q4K: per-row fused dequant-dot
    /// ([`crate::quant::q4k::gemv_q4_k_row`], LUT variant behind the
    /// `simd_lut_q4k` feature), rows rayon-parallel at ≥ 256 rows — the
    /// same threshold the ternary parallel kernel uses. Row-independent
    /// work, so parallelism never reorders a row's own summation.
    pub fn matvec_into(&self, y: &mut [f32], x: &[f32]) {
        match self {
            Self::Ternary(w) => {
                debug_assert_eq!(x.len(), w.cols);
                debug_assert_eq!(y.len(), w.rows);
                katgpt_core::simd_ternary_group_matvec_parallel(w, &x[..w.cols], &mut y[..w.rows]);
            }
            Self::Q4K(blocks, rows, cols) => {
                debug_assert_eq!(x.len(), *cols);
                debug_assert_eq!(y.len(), *rows);
                matvec_q4k(y, blocks, *rows, *cols, x);
            }
        }
    }

    /// Dequantize to dense f32 row-major (the bulk readback; the dense
    /// training container and the audit tools consume this).
    pub fn dequant_to_dense(&self) -> Vec<f32> {
        match self {
            Self::Ternary(w) => QwenDeltaNetTernaryWeights::dequant_proj_to_dense(w),
            Self::Q4K(blocks, rows, cols) => {
                if *rows == 0 {
                    return Vec::new();
                }
                use crate::quant::q4k::{QK_K, dequantize_row_q4_k};
                let nb = cols / QK_K;
                let mut out = vec![0.0f32; rows * cols];
                for r in 0..*rows {
                    dequantize_row_q4_k(&blocks[r * nb..(r + 1) * nb], &mut out[r * cols..(r + 1) * cols]);
                }
                out
            }
        }
    }

    /// G1 corruption check: the ternary arm satisfies the bit-plane
    /// invariant; the Q4K arm is structurally checked (block count must
    /// match `rows × cols/QK_K` — a mis-sized read would be a silent
    /// out-of-bounds in the matvec otherwise).
    pub fn invariants_hold(&self) -> bool {
        match self {
            Self::Ternary(w) => w.rows == 0 || w.invariant_holds(),
            Self::Q4K(blocks, rows, cols) => {
                use crate::quant::q4k::QK_K;
                *rows == 0 || blocks.len() == rows * (cols / QK_K)
            }
        }
    }

    /// Element count (0 for the empty placeholder) — weight elements, not
    /// storage bytes, in both arms.
    pub fn param_count(&self) -> usize {
        self.rows() * self.cols()
    }

    /// Storage bytes this projection occupies (bit-planes + f16 scales, or
    /// q4 super-blocks).
    pub fn bytes(&self) -> usize {
        match self {
            Self::Ternary(w) => {
                if w.rows == 0 {
                    return 0;
                }
                let planes = 2 * w.rows * w.blocks64 * std::mem::size_of::<u64>();
                let scales = w.rows * w.groups_per_row * std::mem::size_of::<half::f16>();
                planes + scales
            }
            Self::Q4K(blocks, _, _) => blocks.len() * std::mem::size_of::<crate::quant::q4k::BlockQ4K>(),
        }
    }
}

/// The Q4K matvec body: per-row fused dequant-dot, rayon across rows at the
/// ternary kernel's ≥ 256-row threshold (row-independent work — bit-stable
/// under any worker count).
fn matvec_q4k(y: &mut [f32], blocks: &[crate::quant::q4k::BlockQ4K], rows: usize, cols: usize, x: &[f32]) {
    use crate::quant::q4k::{QK_K, gemv_q4_k_row};
    assert!(cols.is_multiple_of(QK_K), "q4 projection cols {cols} not a multiple of {QK_K}");
    let nb = cols / QK_K;
    debug_assert_eq!(blocks.len(), rows * nb, "q4 block count mismatch");
    if rows >= 256 {
        use rayon::prelude::*;
        const ROWS_PER_CHUNK: usize = 16;
        y[..rows]
            .par_chunks_mut(ROWS_PER_CHUNK)
            .enumerate()
            .for_each(|(c, chunk)| {
                let r0 = c * ROWS_PER_CHUNK;
                for (i, yr) in chunk.iter_mut().enumerate() {
                    let r = r0 + i;
                    *yr = gemv_q4_k_row(&blocks[r * nb..(r + 1) * nb], &x[..cols]);
                }
            });
    } else {
        for r in 0..rows {
            y[r] = gemv_q4_k_row(&blocks[r * nb..(r + 1) * nb], &x[..cols]);
        }
    }
}

/// Per-layer ternary weights for a hybrid DeltaNet/Attention model.
///
/// Fields unused by a layer type are empty ([`TernaryGroupWeights::new`] with
/// `rows=0, cols=0`), matching the empty-`Vec` convention of
/// [`super::weights::DeltaNetLayerWeights`].
///
/// **Linear attention (`DeltaNet`) layers** use: `in_proj_qkv`, `in_proj_a`,
/// `in_proj_b`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`.
///
/// **Full attention layers** use: `attn_wq` (gated: `2*q_dim`), `attn_wk`,
/// `attn_wv`, `attn_wo`, `gate_proj`, `up_proj`, `down_proj`.
#[derive(Clone)]
pub struct DeltaNetTernaryLayerWeights {
    // --- Full attention projections (ternary; empty for DeltaNet layers) ---
    /// Gated Q projection `[2*q_dim, n_embd]` (Issue 594).
    ///
    /// Qwen3.5 gated attention: `attn_wq @ x` produces `[2*q_dim]` laid out as
    /// `[q(hd), gate(hd)]` per head interleaved. The forward splits this into
    /// q and gate, then multiplies the attention output by `sigmoid(gate)`.
    pub attn_wq: ProjWeights,
    pub attn_wk: ProjWeights,
    pub attn_wv: ProjWeights,
    pub attn_wo: ProjWeights,

    // --- Linear attention (DeltaNet) projections; a/b are Issue 980's
    // GateProjWeights (ternary in pre-rotation files, dense BF16 in Bonsai 2) ---
    pub in_proj_qkv: ProjWeights,
    pub in_proj_a: GateProjWeights,
    pub in_proj_b: GateProjWeights,
    pub in_proj_z: ProjWeights,
    pub out_proj: ProjWeights,

    // --- SwiGLU MLP (both layer types) ---
    pub gate_proj: ProjWeights,
    pub up_proj: ProjWeights,
    pub down_proj: ProjWeights,

    // --- Dense fields (stay Vec<f32> — F32 in GGUF, not projections) ---
    /// QK-norm gamma for Q (per-head, [`head_dim`]). Full-attn layers only.
    pub attn_q_norm: Vec<f32>,
    /// QK-norm gamma for K (per-head, [`head_dim`]). Full-attn layers only.
    pub attn_k_norm: Vec<f32>,
    /// Conv1d weight [`n_total_heads`, `conv_kernel_dim`]. `DeltaNet` layers only.
    pub conv1d_weight: Vec<f32>,
    /// Log-decay parameter [`n_key_heads`]. `DeltaNet` layers only.
    pub a_log: Vec<f32>,
    /// dt bias [`n_total_heads`]. `DeltaNet` layers only.
    pub dt_bias: Vec<f32>,
    /// `RMSNorm` gamma for the linear-attention output [`n_total_heads`].
    /// `DeltaNet` layers only.
    pub linear_norm: Vec<f32>,
    /// Pre-attention `RMSNorm` gamma [`n_embd`]. Both layer types.
    pub input_norm: Vec<f32>,
    /// Post-attention `RMSNorm` gamma [`n_embd`]. Both layer types.
    pub post_attn_norm: Vec<f32>,
}

impl DeltaNetTernaryLayerWeights {
    /// The 10 per-layer projections in forward-pass order.
    ///
    /// For a `DeltaNet` layer, the attention fields are empty (rows=0); for an
    /// attention layer, the `DeltaNet` fields are empty. Empty weights pass
    /// every check trivially, so this can iterate all 10 unconditionally.
    ///
    /// Issue 980: `in_proj_a`/`in_proj_b` are NOT in this array — they are
    /// [`GateProjWeights`] (ternary OR dense). Reach them directly, or via
    /// [`Self::gate_projections`].
    pub fn projections(&self) -> [&ProjWeights; 10] {
        [
            &self.attn_wq,
            &self.attn_wk,
            &self.attn_wv,
            &self.attn_wo,
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.out_proj,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ]
    }

    /// The two Issue-980 gate projections.
    pub fn gate_projections(&self) -> [&GateProjWeights; 2] {
        [&self.in_proj_a, &self.in_proj_b]
    }

    /// G1 corruption check: every non-empty projection satisfies its arm's
    /// invariant (ternary bit-plane conflict check; q4 structural block
    /// count). The [`GateProjWeights`] fields check their own arm (Dense is
    /// trivially valid).
    ///
    /// A loader that mis-parses `Q2_0_g128` typically violates this, so it is
    /// the cheapest post-load sanity gate available. Empty projections (the
    /// unused layer-type's fields) pass trivially.
    pub fn invariants_hold(&self) -> bool {
        self.projections().iter().all(|w| {
            // Empty weights (rows=0) pass — the unused layer-type's fields.
            w.rows() == 0 || w.invariants_hold()
        }) && self.in_proj_a.invariants_hold()
            && self.in_proj_b.invariants_hold()
    }
}

/// All ternary weights for a hybrid DeltaNet/Attention model.
///
/// `wte` and `lm_head` are [`TernaryGroupWeights`] (not dense f32) because
/// Ternary-Bonsai-27B stores them as `Q2_0` — see the module docs for why
/// dequantizing would defeat the memory win.
pub struct QwenDeltaNetTernaryWeights {
    /// Embedding table [`vocab_size`, `n_embd`] as ternary bit-planes.
    /// Row lookup via [`Self::dequant_wte_row_into`].
    pub wte: TernaryGroupWeights,
    /// Final `RMSNorm` gamma [`n_embd`]. Dense f32.
    pub final_norm: Vec<f32>,
    /// LM head [`vocab_size`, `n_embd`] as ternary bit-planes.
    /// Logits via `simd_ternary_group_matvec(&wte, &x, &mut logits)`.
    pub lm_head: TernaryGroupWeights,
    pub layers: Vec<DeltaNetTernaryLayerWeights>,
    /// Per-layer type map (cached for dispatch).
    pub layer_types: Vec<crate::types::DeltaNetLayerType>,
    /// Hadamard-folded basis metadata (Issue 980). `None` = pre-rotation
    /// file — the forward then runs the exact pre-Bonsai-2 path (G3). The
    /// loader only populates this under the `bonsai2_hadamard` feature;
    /// a folded file REFUSES to load without it.
    pub rotation: Option<crate::deltanet::rotation::TernaryRotationConfig>,
}

impl QwenDeltaNetTernaryWeights {
    /// G1 corruption check across every layer + the global ternary tensors.
    pub fn invariants_hold(&self) -> bool {
        self.wte.rows == 0
            || self.wte.invariant_holds()
                && (self.lm_head.rows == 0 || self.lm_head.invariant_holds())
                && self.layers.iter().all(|l| l.invariants_hold())
    }

    /// Create zero-initialized ternary weights for testing.
    ///
    /// **Do NOT use for inference** — projections are all-zero, which produces
    /// garbage logits. This exists for tests that verify the forward-path
    /// wiring (no panics, correct slice lengths) without needing the real
    /// 7.1 GB model.
    ///
    /// Requires all projection dims to be multiples of 128 (the ternary
    /// group size). [`Config::qwen_deltanet`] satisfies this for the default
    /// dims (`n_embd=2048`, `mlp_hidden=8192`, `head_dim=128`, `vocab_size=151936`).
    pub fn zeros(config: &Config) -> Self {
        use crate::types::DeltaNetLayerType;

        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kvd = config.n_kv_head * config.head_dim;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;

        let layer_types = if config.layer_types.is_empty() {
            vec![DeltaNetLayerType::Attention; config.n_layer]
        } else {
            config.layer_types.clone()
        };

        let linear_key_head_dim = config.deltanet_linear_head_dim;
        let linear_num_key_heads = config.deltanet_linear_n_heads;
        let linear_num_value_heads = config.deltanet_linear_n_value_heads;
        let conv_ks = config.deltanet_conv_kernel_size;

        // Dim conventions match `forward_deltanet_layer_ternary` (the forward
        // pass), NOT the dense `QwenDeltaNetWeights::zeros` (which uses a
        // different QKV split that only agrees when n_k == n_v). The forward
        // computes: q_dim = n_k*hd, k_dim = n_k*hd, v_dim = n_v*hd.
        let linear_qkv_out = linear_num_key_heads * linear_key_head_dim * 2
            + linear_num_value_heads * linear_key_head_dim;
        // The forward reads a/b outputs as n_v_heads each (scratch.ab_raw splits at n_v_heads).
        let linear_a_out = linear_num_value_heads;
        let linear_b_out = linear_num_value_heads;
        let linear_z_out = linear_num_value_heads * linear_key_head_dim;
        let linear_out_in = linear_num_value_heads * linear_key_head_dim;
        let conv_dim = linear_num_key_heads * linear_key_head_dim * 2
            + linear_num_value_heads * linear_key_head_dim;

        let empty = || ProjWeights::empty();

        let layers: Vec<DeltaNetTernaryLayerWeights> = layer_types
            .iter()
            .map(|&lt| {
                let is_linear = lt == DeltaNetLayerType::DeltaNet;
                DeltaNetTernaryLayerWeights {
                    // Full attention (empty for DeltaNet layers)
                    attn_wq: if is_linear {
                        empty()
                    } else {
                        ProjWeights::ternary(2 * q_dim, n)
                    },
                    attn_wk: if is_linear {
                        empty()
                    } else {
                        ProjWeights::ternary(kvd, n)
                    },
                    attn_wv: if is_linear {
                        empty()
                    } else {
                        ProjWeights::ternary(kvd, n)
                    },
                    // Issue 594: `q_dim`, not `kvd` — `attn_wo` consumes the
                    // full gated-attention output width. Matches the GGUF
                    // loader's `blk.N.attn_output.weight` = [5120 x 6144].
                    attn_wo: if is_linear {
                        empty()
                    } else {
                        ProjWeights::ternary(n, q_dim)
                    },

                    // Linear attention (empty for attn layers)
                    in_proj_qkv: if is_linear {
                        ProjWeights::ternary(linear_qkv_out, n)
                    } else {
                        empty()
                    },
                    in_proj_a: if is_linear {
                        GateProjWeights::Ternary(TernaryGroupWeights::new(linear_a_out, n))
                    } else {
                        GateProjWeights::empty()
                    },
                    in_proj_b: if is_linear {
                        GateProjWeights::Ternary(TernaryGroupWeights::new(linear_b_out, n))
                    } else {
                        GateProjWeights::empty()
                    },
                    in_proj_z: if is_linear {
                        ProjWeights::ternary(linear_z_out, n)
                    } else {
                        empty()
                    },
                    out_proj: if is_linear {
                        ProjWeights::ternary(n, linear_out_in)
                    } else {
                        empty()
                    },

                    // MLP (both types)
                    gate_proj: ProjWeights::ternary(mlp, n),
                    up_proj: ProjWeights::ternary(mlp, n),
                    down_proj: ProjWeights::ternary(n, mlp),

                    // Dense fields
                    attn_q_norm: if is_linear {
                        Vec::new()
                    } else {
                        vec![1.0f32; config.head_dim]
                    },
                    attn_k_norm: if is_linear {
                        Vec::new()
                    } else {
                        vec![1.0f32; config.head_dim]
                    },
                    conv1d_weight: if is_linear {
                        vec![0.0f32; conv_dim * conv_ks]
                    } else {
                        Vec::new()
                    },
                    a_log: if is_linear {
                        vec![0.0f32; linear_num_key_heads]
                    } else {
                        Vec::new()
                    },
                    dt_bias: if is_linear {
                        vec![0.0f32; linear_num_value_heads]
                    } else {
                        Vec::new()
                    },
                    // PER-HEAD gamma (Issue 594): `ssm_norm` in the real GGUF
                    // is `[head_dim]`, shared across heads — NOT
                    // `n_v_heads * head_dim`. The old sizing matched a buggy
                    // whole-buffer norm and hid an out-of-bounds read.
                    linear_norm: if is_linear {
                        vec![1.0f32; linear_key_head_dim]
                    } else {
                        Vec::new()
                    },
                    input_norm: vec![1.0f32; n],
                    post_attn_norm: vec![1.0f32; n],
                }
            })
            .collect();

        Self {
            wte: TernaryGroupWeights::new(vocab, n),
            final_norm: vec![1.0f32; n],
            lm_head: TernaryGroupWeights::new(vocab, n),
            layers,
            layer_types,
            rotation: None,
        }
    }

    /// Dequantize one row of `wte` into `out` (embedding lookup).
    ///
    /// Reads the bit-planes directly from the public fields — no katgpt-rs
    /// change needed. The per-128-weight f16 group scale is applied to each
    /// reconstructed ternary value. `out.len()` must equal `wte.cols`.
    ///
    /// This is the ternary analog of `x[..n].copy_from_slice(&wte[token*n..])`
    /// in the dense forward pass.
    #[allow(clippy::needless_range_loop)]
    pub fn dequant_wte_row_into(&self, row_idx: usize, out: &mut [f32]) {
        dequant_group_row_into(&self.wte, row_idx, out);
    }

    /// Dequantize one row of the LM head into `out` — the logit-lens reads
    /// the top-token row (Issue 022 T5.4's hydra arm consumes the lens DE
    /// against the parent's own argmax). Same bit-plane layout as
    /// [`Self::dequant_wte_row_into`]: `lm_head` is the identical
    /// [`TernaryGroupWeights`] type.
    pub fn dequant_lm_head_row_into(&self, row_idx: usize, out: &mut [f32]) {
        dequant_group_row_into(&self.lm_head, row_idx, out);
    }

    /// Total parameter count across all projections + global tensors (weight
    /// elements, not storage bytes — both arms count their own elements).
    pub fn ternary_param_count(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .flat_map(|l| l.projections())
            .map(|w| w.param_count())
            .sum::<usize>()
            + self
                .layers
                .iter()
                .flat_map(|l| l.gate_projections())
                .map(|w| w.param_count())
                .sum::<usize>();
        layers + self.wte.rows * self.wte.cols + self.lm_head.rows * self.lm_head.cols
    }

    /// Bytes occupied by the weight containers: ternary bit-planes plus one
    /// f16 scale per 128-weight group (2.125 bits/weight at cols % 128 == 0),
    /// or q4 super-blocks (4.5 bpw) for a q4-arm projection. A Dense gate
    /// projection counts as raw f32 (its storage class).
    pub fn ternary_bytes(&self) -> usize {
        fn gate_tensor_bytes(w: &GateProjWeights) -> usize {
            match w {
                GateProjWeights::Ternary(t) => t_bytes(t),
                GateProjWeights::Dense(data, _, _) => data.len() * std::mem::size_of::<f32>(),
            }
        }
        fn t_bytes(w: &TernaryGroupWeights) -> usize {
            if w.rows == 0 {
                return 0;
            }
            let planes = 2 * w.rows * w.blocks64 * std::mem::size_of::<u64>();
            let scales = w.rows * w.groups_per_row * std::mem::size_of::<half::f16>();
            planes + scales
        }
        let layers: usize = self
            .layers
            .iter()
            .flat_map(|l| l.projections())
            .map(|w| w.bytes())
            .sum::<usize>()
            + self
                .layers
                .iter()
                .flat_map(|l| l.gate_projections())
                .map(gate_tensor_bytes)
                .sum::<usize>();
        layers + t_bytes(&self.wte) + t_bytes(&self.lm_head)
    }

    /// Dequantize a ternary projection to dense f32 row-major data.
    ///
    /// Reconstructs `value[row][col] = ternary_sign(row, col) * group_scale[row, group(col)]`
    /// for all `rows * cols` elements. Empty projections (`rows == 0`) return `vec![]`.
    ///
    /// This is the bulk form of [`Self::dequant_wte_row_into`] (which does one row
    /// at a time for the embedding lookup). It exists for Issue 445's last-N-layer
    /// training path: the top-N layers + `lm_head` are dequantized to f32 at load
    /// time so they can receive gradients via the existing dense backward, while
    /// the bottom `(n_layer - N)` layers stay ternary (frozen).
    pub fn dequant_proj_to_dense(w: &TernaryGroupWeights) -> Vec<f32> {
        if w.rows == 0 {
            return Vec::new();
        }
        let mut out = vec![0.0f32; w.rows * w.cols];
        for r in 0..w.rows {
            let pos_base = r * w.blocks64;
            let neg_base = r * w.blocks64;
            let scale_base = r * w.groups_per_row;
            for c in 0..w.cols {
                let word = c / 64;
                let bit = c % 64;
                let mask = 1u64 << bit;
                let is_pos = (w.pos_bits[pos_base + word] & mask) != 0;
                let is_neg = (w.neg_bits[neg_base + word] & mask) != 0;
                let ternary: f32 = match (is_pos, is_neg) {
                    (true, false) => 1.0,
                    (false, true) => -1.0,
                    _ => 0.0,
                };
                // GROUP_SIZE = 128 (katgpt-types::GROUP_SIZE).
                let group = c / 128;
                let scale = w.group_scale[scale_base + group].to_f32();
                out[r * w.cols + c] = ternary * scale;
            }
        }
        out
    }

    /// Convert the ternary model to a mixed dense container for last-N-layer training
    /// (riir-train Issue 445 / Plan 333 T3.0).
    ///
    /// Layers `0..(n_layer - n_trainable)` keep their ternary projections (wrapped
    /// as [`Proj::Ternary`] — frozen, forward-only). Layers
    /// `(n_layer - n_trainable)..n_layer` + `lm_head` are **dequantized** to
    /// dense f32 (wrapped as [`Proj::Dense`] — receive gradients during training).
    ///
    /// The returned [`QwenDeltaNetWeights`] is runnable through the **existing**
    /// [`forward_qwen_deltanet`](super::forward::forward_qwen_deltanet) because
    /// every projection call dispatches via [`Proj::matvec`], which handles both
    /// arms. No new forward function is needed.
    ///
    /// Memory at the documented defaults:
    /// - N=4: ~11.2 GB dequantized (4 layers + `lm_head`) + 6.67 GB ternary base = ~18 GB
    /// - N=8: ~17.3 GB dequantized + 6.67 GB ternary base = ~24 GB
    ///
    /// `wte` is dequantized eagerly (one-time cost at load). The dense
    /// [`QwenDeltaNetWeights`] has `wte: Vec<f32>`, so the ternary bit-plane
    /// `wte` cannot stay ternary in the returned struct. At Qwen3.5's shape
    /// `[248320 × 5120]`, that's a 5.08 GB `Vec<f32>`.
    pub fn to_hybrid_dense(&self, n_trainable: usize) -> super::weights::QwenDeltaNetWeights {
        use super::weights::{DeltaNetLayerWeights, Proj, QwenDeltaNetWeights};
        let n_layer = self.layers.len();
        assert!(
            n_trainable <= n_layer,
            "n_trainable {n_trainable} exceeds n_layer {n_layer}"
        );
        let first_trainable = n_layer - n_trainable;

        // Dequantize wte into a dense Vec<f32> (one-time cost).
        let n = self.wte.cols;
        let vocab = self.wte.rows;
        let mut wte_dense = vec![0.0f32; vocab * n];
        for row in 0..vocab {
            self.dequant_wte_row_into(row, &mut wte_dense[row * n..(row + 1) * n]);
        }

        // lm_head: dequantize to Proj::Dense (trainable).
        let lm_head_rows = self.lm_head.rows;
        let lm_head_cols = self.lm_head.cols;
        let lm_head_dense = Self::dequant_proj_to_dense(&self.lm_head);
        let lm_head = Proj::dense(lm_head_dense, lm_head_rows, lm_head_cols);

        // Per-layer conversion.
        let layers: Vec<DeltaNetLayerWeights> = self
            .layers
            .iter()
            .enumerate()
            .map(|(idx, twl)| {
                let trainable = idx >= first_trainable;
                convert_ternary_layer_to_proj_layer(twl, trainable)
            })
            .collect();

        QwenDeltaNetWeights {
            wte: wte_dense,
            final_norm: self.final_norm.clone(),
            lm_head,
            layers,
            layer_types: self.layer_types.clone(),
        }
    }
}

/// Shared row dequant for the two [`TernaryGroupWeights`] tables (the wte
/// lookup and the LM head's logit-lens reads). Bit-planes + per-128-weight
/// f16 group scales, the layout `dequant_wte_row_into` documented.
/// `out.len()` must equal `w.cols`.
#[allow(clippy::needless_range_loop)]
fn dequant_group_row_into(w: &TernaryGroupWeights, row_idx: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), w.cols, "out slice must match w.cols");
    debug_assert!(row_idx < w.rows, "row_idx out of range");

    let blocks64 = w.blocks64;
    let groups_per_row = w.groups_per_row;
    let base = row_idx * blocks64;
    let scale_base = row_idx * groups_per_row;

    for c in 0..w.cols {
        let word = c / 64;
        let bit = c % 64;
        let mask = 1u64 << bit;

        let is_pos = (w.pos_bits[base + word] & mask) != 0;
        let is_neg = (w.neg_bits[base + word] & mask) != 0;

        let ternary: f32 = match (is_pos, is_neg) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        };

        // GROUP_SIZE = 128 (katgpt-types::GROUP_SIZE). Hardcoded to avoid
        // the binary_plasma/ternary_group_scale feature-export ambiguity.
        let group = c / 128;
        let scale = w.group_scale[scale_base + group].to_f32();
        out[c] = ternary * scale;
    }
}

/// Convert one ternary layer to a dense-container layer.
///
/// `trainable` controls whether the projections become [`Proj::Dense`] (top-N
/// layers — dequantized, receive gradients) or stay [`Proj::Ternary`] (bottom
/// layers — frozen, forward-only via the ternary SIMD kernel).
///
/// Dense f32 fields (`input_norm`, `post_attn_norm`, `conv1d_weight`, `a_log`,
/// `dt_bias`, `linear_norm`, `attn_q_norm`, `attn_k_norm`) are cloned verbatim
/// from the ternary layer — they are f32 in both structs.
fn convert_ternary_layer_to_proj_layer(
    twl: &DeltaNetTernaryLayerWeights,
    trainable: bool,
) -> super::weights::DeltaNetLayerWeights {
    use super::weights::{DeltaNetLayerWeights, Proj};

    /// Convert one projection: `Ternary` (frozen) or `Dense` (dequantized,
    /// trainable).
    ///
    /// Issue 028 T4 S2: a q4-arm projection dequantizes in both postures —
    /// the dense container has no q4 arm (it is the TRAINING container; its
    /// Ternary arm is the ternary kernel's container).
    fn convert_proj(w: &ProjWeights, trainable: bool) -> Proj {
        match w {
            ProjWeights::Ternary(t) => convert_ternary_ref(t, trainable),
            ProjWeights::Q4K(_, rows, cols) => {
                if *rows == 0 {
                    Proj::empty()
                } else {
                    let data = w.dequant_to_dense();
                    Proj::dense(data, *rows, *cols)
                }
            }
        }
    }

    /// The ternary-arm body of [`convert_proj`] (shared with
    /// `convert_gate_proj`, whose `GateProjWeights::Ternary` arm arrives as
    /// a bare [`TernaryGroupWeights`]).
    fn convert_ternary_ref(t: &TernaryGroupWeights, trainable: bool) -> Proj {
        if trainable {
            let data = QwenDeltaNetTernaryWeights::dequant_proj_to_dense(t);
            Proj::dense(data, t.rows, t.cols)
        } else if t.rows == 0 {
            Proj::empty()
        } else {
            #[cfg(feature = "deltanet_ternary_inference")]
            {
                Proj::Ternary(t.clone())
            }
            #[cfg(not(feature = "deltanet_ternary_inference"))]
            {
                // Unreachable: this function is only called when the
                // ternary feature is active (the ternary weights struct
                // doesn't exist otherwise).
                unreachable!("convert_proj without deltanet_ternary_inference")
            }
        }
    }

    /// Issue 980: the a/b fields are `GateProjWeights` — a Dense arm is
    /// already dense (clone verbatim, trainable or not: the dequant stage
    /// carries no extra information for an already-dense projection); a
    /// Ternary arm follows the shared `convert_proj` logic.
    fn convert_gate_proj(w: &super::ternary_weights::GateProjWeights, trainable: bool) -> Proj {
        match w {
            super::ternary_weights::GateProjWeights::Dense(data, rows, cols) => {
                Proj::dense(data.clone(), *rows, *cols)
            }
            super::ternary_weights::GateProjWeights::Ternary(t) => convert_ternary_ref(t, trainable),
        }
    }

    DeltaNetLayerWeights {
        attn_wq: convert_proj(&twl.attn_wq, trainable),
        attn_wk: convert_proj(&twl.attn_wk, trainable),
        attn_wv: convert_proj(&twl.attn_wv, trainable),
        attn_wo: convert_proj(&twl.attn_wo, trainable),
        in_proj_qkv: convert_proj(&twl.in_proj_qkv, trainable),
        in_proj_a: convert_gate_proj(&twl.in_proj_a, trainable),
        in_proj_b: convert_gate_proj(&twl.in_proj_b, trainable),
        in_proj_z: convert_proj(&twl.in_proj_z, trainable),
        out_proj: convert_proj(&twl.out_proj, trainable),
        gate_proj: convert_proj(&twl.gate_proj, trainable),
        up_proj: convert_proj(&twl.up_proj, trainable),
        down_proj: convert_proj(&twl.down_proj, trainable),
        attn_q_norm: twl.attn_q_norm.clone(),
        attn_k_norm: twl.attn_k_norm.clone(),
        conv1d_weight: twl.conv1d_weight.clone(),
        a_log: twl.a_log.clone(),
        dt_bias: twl.dt_bias.clone(),
        linear_norm: twl.linear_norm.clone(),
        input_norm: twl.input_norm.clone(),
        post_attn_norm: twl.post_attn_norm.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty weights (the unused layer-type's fields) must pass invariants
    /// trivially — this is the guard for the hybrid layer dispatch.
    #[test]
    fn test_empty_projection_passes_invariant() {
        let empty = TernaryGroupWeights::new(0, 0);
        assert_eq!(empty.rows, 0);
        // invariant_holds on empty should pass (no bits to conflict).
        // TernaryGroupWeights::invariant_holds may or may not handle rows=0;
        // our layer-level check skips empty weights explicitly.
        let layer = DeltaNetTernaryLayerWeights {
            attn_wq: ProjWeights::empty(),
            attn_wk: ProjWeights::empty(),
            attn_wv: ProjWeights::empty(),
            attn_wo: ProjWeights::empty(),
            in_proj_qkv: ProjWeights::empty(),
            in_proj_a: GateProjWeights::empty(),
            in_proj_b: GateProjWeights::empty(),
            in_proj_z: ProjWeights::empty(),
            out_proj: ProjWeights::empty(),
            gate_proj: ProjWeights::empty(),
            up_proj: ProjWeights::empty(),
            down_proj: ProjWeights::empty(),
            attn_q_norm: Vec::new(),
            attn_k_norm: Vec::new(),
            conv1d_weight: Vec::new(),
            a_log: Vec::new(),
            dt_bias: Vec::new(),
            linear_norm: Vec::new(),
            input_norm: Vec::new(),
            post_attn_norm: Vec::new(),
        };
        assert!(layer.invariants_hold());
    }

    /// Issue 028 T4 S2 — the q4 arm dispatches and reads back exactly: the
    /// matvec equals a dequantize-then-scalar-dot reference, and
    /// `dequant_to_dense` equals the per-row reference. The quantization
    /// MATH is pinned by the q4k module's own tests; this pins the
    /// ProjWeights arm (row indexing, parallel/serial chunking, ownership)
    /// plus a loose q4-vs-ternary magnitude sanity (the two formats' dots
    /// approximate the same source — never a bit identity).
    #[test]
    fn q4k_arm_matvec_and_dequant_match_reference() {
        use crate::quant::q4k::{QK_K, dequantize_row_q4_k, quantize_row_q4_k};
        use bytemuck::Zeroable;

        // 320 rows (crosses the 256-row parallel threshold with a ragged
        // 16-row tail) × 512 cols (2 super-blocks per row).
        let (rows, cols) = (320usize, 512usize);
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 2.0)
            .collect();
        let nb = cols / QK_K;
        let mut blocks = vec![crate::quant::q4k::BlockQ4K::zeroed(); rows * nb];
        for r in 0..rows {
            quantize_row_q4_k(&src[r * cols..(r + 1) * cols], &mut blocks[r * nb..(r + 1) * nb]);
        }
        // Independent reference payload (w owns its own copy).
        let blocks_ref = blocks.clone();
        let w = ProjWeights::Q4K(blocks, rows, cols);

        // Invariants + accessors.
        assert!(w.invariants_hold());
        assert_eq!((w.rows(), w.cols()), (rows, cols));
        assert_eq!(w.param_count(), rows * cols);
        assert_eq!(
            w.bytes(),
            rows * nb * std::mem::size_of::<crate::quant::q4k::BlockQ4K>()
        );
        assert!(w.as_ternary().is_none());

        // Independent dense reference (per-row dequant) + per-row dot ref.
        let x: Vec<f32> = (0..cols).map(|i| (i % 31) as f32 / 31.0 - 0.5).collect();
        let ref_dot = |r: usize| -> f32 {
            let mut row = vec![0.0f32; cols];
            dequantize_row_q4_k(
                &blocks_ref[r * nb..(r + 1) * nb],
                &mut row,
            );
            row.iter().zip(&x).map(|(wv, xv)| wv * xv).sum()
        };

        // matvec == dequant-then-dot reference (both chunk postures).
        let mut y = vec![0.0f32; rows];
        w.matvec_into(&mut y, &x);
        for (r, got) in y.iter().enumerate() {
            let want = ref_dot(r);
            let scale = want.abs().max(1.0);
            assert!(
                (*got - want).abs() <= 1e-3 * scale,
                "row {r}: q4 matvec {got} vs reference {want}"
            );
        }

        // dequant_to_dense == the same per-row reference.
        let mut dense = vec![0.0f32; rows * cols];
        for r in 0..rows {
            dequantize_row_q4_k(
                &blocks_ref[r * nb..(r + 1) * nb],
                &mut dense[r * cols..(r + 1) * cols],
            );
        }
        assert_eq!(w.dequant_to_dense(), dense);

        // q4K arm vs the TERNARY arm quantized from the same source: both
        // approximate the same f32 dot with O(quant step) per-element noise.
        // On quant-hostile fixtures a single row's ternary noise can reach
        // ~20% of its dot (measured: this fixture's row 0 is 19% off), so
        // the cross-format check is AGGREGATE — the per-row dispatch truth
        // is already pinned exactly by the dequant-reference above; a
        // row-shift/transpose bug here would blow the aggregate up.
        let t = katgpt_core::TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let mut y_t = vec![0.0f32; rows];
        katgpt_core::simd_ternary_group_matvec_parallel(&t, &x, &mut y_t);
        let sum_q4: f32 = y.iter().sum();
        let sum_t: f32 = y_t.iter().sum();
        let denom = y.iter().map(|v| v.abs()).sum::<f32>().max(1.0);
        assert!(
            (sum_q4 - sum_t).abs() <= 0.05 * denom,
            "aggregate q4 vs ternary dots diverge: {sum_q4} vs {sum_t} (Σ|y| {denom})"
        );
    }

    /// The Ternary arm of the enum dispatches to the same kernel as before
    /// the enum existed (the old-file byte-identity contract).
    #[test]
    fn ternary_arm_matvec_is_the_parallel_kernel() {
        let (rows, cols) = (384usize, 256usize);
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| if i % 3 == 0 { 0.7 } else { -0.4 })
            .collect();
        let t = katgpt_core::TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let w = ProjWeights::Ternary(t.clone());

        let x: Vec<f32> = (0..cols).map(|i| (i as f32 / cols as f32) - 0.5).collect();
        let mut a = vec![0.0f32; rows];
        let mut b = vec![0.0f32; rows];
        katgpt_core::simd_ternary_group_matvec_parallel(&t, &x, &mut a);
        w.matvec_into(&mut b, &x);
        assert_eq!(a, b, "enum ternary arm must be the identical kernel");
    }

    /// A mixed-arm layer (a q4 in_proj_qkv beside ternary siblings) passes
    /// the layer-level G1 check — the disaggregated prefill copy's shape.
    #[test]
    fn mixed_arm_layer_passes_invariants() {
        use crate::quant::q4k::{QK_K, quantize_row_q4_k};
        use bytemuck::Zeroable;
        let (rows, cols) = (8usize, 256usize);
        let src = vec![0.5f32; rows * cols];
        let nb = cols / QK_K;
        let mut blocks = vec![crate::quant::q4k::BlockQ4K::zeroed(); rows * nb];
        quantize_row_q4_k(&src, &mut blocks);

        let mut layer = {
            let cfg = crate::types::Config::qwen_deltanet(
                2,
                vec![crate::types::DeltaNetLayerType::DeltaNet],
            );
            QwenDeltaNetTernaryWeights::zeros(&cfg).layers[0].clone()
        };
        layer.in_proj_qkv = ProjWeights::Q4K(blocks, rows, cols);
        assert!(layer.invariants_hold());

        // A wrong block count must fail the structural check.
        layer.in_proj_qkv = ProjWeights::Q4K(
            vec![crate::quant::q4k::BlockQ4K::zeroed(); rows * nb - 1],
            rows,
            cols,
        );
        assert!(!layer.invariants_hold());
    }
}
