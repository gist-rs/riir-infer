//! Hadamard-folded (rotated) ternary weight support — Bonsai 2 (Issue 980).
//!
//! Bonsai-2 27B stores ternary weights in a rotated basis:
//! `W' = W·R` with `R = diag(S)·(1/√n)·H_n` applied blockwise (block 1024,
//! unscaled Walsh–Hadamard `H_n`, per-dimension sign vector `S`). At runtime
//! every folded matmul computes `y = W'·(R⁻¹·x)`; because `R⁻¹ = Rᵀ` and the
//! transformed input re-maps to the primal basis, the matmul output is in the
//! primal basis and no output-side transform is needed.
//!
//! Fork contract (`PrismML` `llama.cpp` pin `7dffb158d`, confirmed by source
//! read 2026-09-19):
//!
//! - **Folded matmul** (`build_lora_mm`): `x' = (1/√n)·H_n·(S⊙x)` per block —
//!   sign multiply FIRST, then the blockwise Hadamard (the fork builds an
//!   explicit `±1/√n` matrix and lets backends substitute an FWHT via
//!   `GGML_HINT_SRC0_IS_HADAMARD`). Applications are memoized per activation
//!   tensor, so a shared input (e.g. the normed residual feeding qkv + z) is
//!   rotated once.
//! - **Embedding inverse** (`build_inp_embd`): the token-embedding table rows
//!   are stored latent — folded with the SAME forward transform,
//!   `z = (1/√n)·H_n·(S⊙e)` — so after lookup the primal embedding is
//!   restored by the INVERSE map `e = S⊙((1/√n)·H_n·z)`: Hadamard first,
//!   sign SECOND (fork `build_inp_embd`: `h = s * (H z)`). Composing the two
//!   gives `S·H·H·S = I` on the block.
//! - **`gdn_v_grouped`** (`llama-model.cpp:2080`): when set, the `ssm_out`
//!   input arrives in tiled `[hd, nk, rep]` head order and must be permuted
//!   to the grouped `[hd, rep, nk]` order the fold was computed in, before
//!   signs + rotation. With `n_v = ssm_dt_rank`, `n_k = ssm_n_group`:
//!   `hd = in_dim / n_v`, `rep = n_v / n_k` (requires `n_v % n_k == 0`).
//! - Sign vectors are keyed by the matmul INPUT width (`prism.hadamard.
//!   sign_widths` = `[5120, 6144, 17408]` for Bonsai-2: n_embd / value_dim /
//!   ffn). `sign_values` is their concatenation in `sign_widths` order.
//!
//! Refusal posture: a file declaring `prism.hadamard` that we cannot honor
//! exactly (version, transform, axis, sign mode, widths, geometry) is
//! rejected LOUDLY at load — never run folded weights unrotated (stock
//! llama.cpp garbage class).
//!
//! Normalization note: the FWHT here scales each butterfly by `1/√2`, so a
//! 1024-block transform carries the full `1/√1024` — the same unitary map as
//! the fork's explicit matrix (up to float rounding). `katgpt-kv` ships a
//! slice-shaped twin (`kvarn::hadamard`); it is deliberately not consumed
//! here (new cross-repo dep for one function) and `katgpt-core::meld`'s
//! const-generic `[f32; D]` shape would force per-block copies on the hot
//! path. The GPU lane carries its own CUDA twin of this kernel.

use anyhow::{bail, Context, Result};

/// Parsed `prism.hadamard.*` metadata + the runtime rotation tables.
///
/// Stored on [`crate::deltanet::ternary_weights::QwenDeltaNetTernaryWeights`]
/// as `Option<TernaryRotationConfig>`; `None` = pre-rotation file (the old
/// Bonsai-27B lane, byte-identical behavior).
#[derive(Debug, Clone)]
pub struct TernaryRotationConfig {
    /// Block size of the blockwise Hadamard (1024 for Bonsai-2). Power of 2;
    /// every folded matmul input width must be a multiple of this.
    pub block_size: usize,
    /// Sign vectors keyed by matmul input width (`+1`/`-1` validated at load).
    /// The forward lookup is `signs_for_width(w)`.
    pub signs: Vec<(usize, Vec<i8>)>,
    /// `prism.hadamard.gdn_v_grouped` — the `ssm_out` input needs the
    /// tiled→grouped V-head permute before signs + rotation.
    pub gdn_v_grouped: bool,
    /// `ssm_dt_rank` (`n_v`) and `ssm_n_group` (`n_k`), needed to derive the
    /// permute geometry when [`Self::gdn_v_grouped`] is set.
    pub gdn_v_heads: usize,
    pub gdn_k_groups: usize,
    /// Widths of the embedding-inverse transform: `token_embd.weight` rows
    /// are latent and get the inverse rotation after lookup.
    pub inverse_embedding: bool,
}

impl TernaryRotationConfig {
    /// Sign vector for a folded matmul's input width.
    ///
    /// `None` = no sign stage (the GGUF's `sign_mode = identity` case).
    /// Errors only for `sign_mode = explicit` files are raised at load; here a
    /// missing width is a hard slice-index bug, so callers resolve widths
    /// through the loader-validated set.
    pub fn signs_for_width(&self, width: usize) -> Option<&[i8]> {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, v)| v.as_slice())
    }

    /// Rotated input width for scratch sizing: the widest folded matmul input.
    /// The forward computes it from config dims (`n_embd` / `v_dim` / mlp), so
    /// this is only a load-time cross-check.
    pub fn max_folded_width(&self) -> usize {
        self.signs.iter().map(|(w, _)| *w).max().unwrap_or(0)
    }
}

/// In-place unitary Walsh–Hadamard transform on one power-of-2 block.
///
/// Each butterfly multiplies by `1/√2`, so the whole transform applies
/// `1/√n` — the same map as the fork's explicit `±1/√n` Hadamard matrix.
/// Zero-allocation, `O(n log n)`. Non-power-of-2 lengths are a caller bug
/// (the loader refuses non-power-of-2 block sizes).
#[inline]
pub(crate) fn fwht_block_inplace(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "FWHT block must be power-of-2, got {n}");
    if n <= 1 {
        return;
    }
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    // Split borrow: LLVM proves the halves non-aliasing, enabling the same
    // auto-vectorization the katgpt-kv twin gets from pointer arithmetic.
    for step in (0..n.trailing_zeros()).map(|s| 1usize << (s + 1)) {
        let half = step / 2;
        for block in x.chunks_exact_mut(step) {
            let (lo, hi) = block.split_at_mut(half);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (av, bv) = (*a, *b);
                *a = (av + bv) * inv_sqrt2;
                *b = (av - bv) * inv_sqrt2;
            }
        }
    }
}

/// Forward rotation on a folded matmul's input, in place:
/// `x ← (1/√n)·H_n·(S⊙x)` per `block_size` block (sign flip FIRST).
///
/// `signs` is the width-keyed sign vector for THIS width (`None` = identity
/// sign mode). Width must be a multiple of `block_size` (validated at load).
pub fn rotate_forward_inplace(x: &mut [f32], signs: Option<&[i8]>, block_size: usize) {
    debug_assert!(
        x.len().is_multiple_of(block_size),
        "folded input width {} not a multiple of block {block_size}",
        x.len()
    );
    for (bi, block) in x.chunks_exact_mut(block_size).enumerate() {
        if let Some(s) = signs {
            let s = &s[bi * block_size..(bi + 1) * block_size];
            for (v, &sign) in block.iter_mut().zip(s.iter()) {
                *v *= sign as f32;
            }
        }
        fwht_block_inplace(block);
    }
}

/// Inverse rotation for a latent-basis lookup result, in place:
/// `z ← S⊙((1/√n)·H_n·z)` per block — Hadamard FIRST, sign SECOND.
///
/// Used ONLY for the token-embedding table (`inverse_weight_names`), whose
/// rows are stored as `S⊙(H_n·e/√n)`.
pub fn rotate_inverse_inplace(z: &mut [f32], signs: Option<&[i8]>, block_size: usize) {
    debug_assert!(
        z.len().is_multiple_of(block_size),
        "inverse width {} not a multiple of block {block_size}",
        z.len()
    );
    for (bi, block) in z.chunks_exact_mut(block_size).enumerate() {
        fwht_block_inplace(block);
        if let Some(s) = signs {
            let s = &s[bi * block_size..(bi + 1) * block_size];
            for (v, &sign) in block.iter_mut().zip(s.iter()) {
                *v *= sign as f32;
            }
        }
    }
}

/// `gdn_v_grouped` feature permute for the `ssm_out` input, in place.
///
/// The recurrence emits V in tiled `[hd, nk, rep]` order
/// (`f = hd + hd_len·nk + hd_len·n_k·rep`); the fold was computed in grouped
/// `[hd, rep, nk]` order (`f = hd + hd_len·rep + hd_len·rep_max·nk`). `tmp`
/// must be at least `x.len()` (the caller's rotation scratch serves).
pub fn permute_gdn_v_grouped_inplace(x: &mut [f32], tmp: &mut [f32], n_v: usize, n_k: usize) {
    let len = x.len();
    debug_assert!(tmp.len() >= len, "permute scratch too small");
    debug_assert!(
        n_k > 0 && n_v.is_multiple_of(n_k) && len.is_multiple_of(n_v),
        "gdn permute geometry invalid: len {len} n_v {n_v} n_k {n_k}"
    );
    let hd = len / n_v;
    let rep = n_v / n_k;
    tmp[..len].copy_from_slice(x);
    // The vector is one segment of n_v heads × hd: the tiled head
    // (nk, r) at offset (r·n_k + nk)·hd moves to the grouped offset
    // (nk·rep + r)·hd — a whole-head-block gather, never a partial head.
    for nk in 0..n_k {
        for r in 0..rep {
            let src = (r * n_k + nk) * hd;
            let dst = (nk * rep + r) * hd;
            x[dst..dst + hd].copy_from_slice(&tmp[src..src + hd]);
        }
    }
}

/// Inverse of [`permute_gdn_v_grouped_inplace`] — grouped → tiled. G1
/// bisect knob (`RIIR_B2_PERM=2`); not on any production path.
pub fn permute_gdn_v_grouped_inverse_inplace(
    x: &mut [f32],
    tmp: &mut [f32],
    n_v: usize,
    n_k: usize,
) {
    let len = x.len();
    debug_assert!(tmp.len() >= len, "permute scratch too small");
    debug_assert!(
        n_k > 0 && n_v.is_multiple_of(n_k) && len.is_multiple_of(n_v),
        "gdn permute geometry invalid: len {len} n_v {n_v} n_k {n_k}"
    );
    let hd = len / n_v;
    let rep = n_v / n_k;
    tmp[..len].copy_from_slice(x);
    for nk in 0..n_k {
        for r in 0..rep {
            let src = (nk * rep + r) * hd;
            let dst = (r * n_k + nk) * hd;
            x[dst..dst + hd].copy_from_slice(&tmp[src..src + hd]);
        }
    }
}

// ── GGUF metadata parsing ───────────────────────────────────────────────

use crate::gguf_loader::GgufFile;

/// Folded-name allowlist check against OUR structural map (the fork's
/// `is_foldable_weight` posture, resolved per layer type).
///
/// - `output.weight` — the LM head (always folded when present).
/// - `DeltaNet` layers fold `attn_qkv`, `attn_gate`, `ssm_out`, `ffn_{down,gate,up}`.
/// - Full-attention layers fold `attn_{q,k,v,output}`, `ffn_{down,gate,up}`.
///
/// `in_proj_a`/`in_proj_b` (`ssm_alpha/ssm_beta`) are NOT foldable — they are
/// the dense escape set and consume the PRIMAL input.
pub fn is_known_folded_name(
    name: &str,
    n_layer: usize,
    layer_types: &[crate::types::DeltaNetLayerType],
) -> bool {
    use crate::types::DeltaNetLayerType;
    if name == "output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((idx, suffix)) = rest.split_once('.') else {
        return false;
    };
    let Ok(idx) = idx.parse::<usize>() else {
        return false;
    };
    if idx >= n_layer || idx >= layer_types.len() {
        return false;
    }
    let is_linear = layer_types[idx] == DeltaNetLayerType::DeltaNet;
    let (head, tail) = suffix
        .split_once('.')
        .map_or((suffix, None), |(h, t)| (h, Some(t)));
    if tail != Some("weight") {
        return false;
    }
    match head {
        "ffn_down" | "ffn_gate" | "ffn_up" => true,
        "attn_qkv" | "attn_gate" | "ssm_out" => is_linear,
        "attn_q" | "attn_k" | "attn_v" | "attn_output" => !is_linear,
        _ => false,
    }
}

/// Parse + validate `prism.hadamard.*` from an open GGUF.
///
/// Returns `Ok(None)` when the file declares no rotation (pre-rotation
/// Bonsai lane). Every unsupported combination is a LOUD error — a rotated
/// file we cannot honor must never load silently.
pub fn parse_prism_hadamard(gguf: &GgufFile) -> Result<Option<TernaryRotationConfig>> {
    let Some(version) = gguf.metadata_u64("prism.hadamard.version") else { return Ok(None) };
    if version != 1 {
        bail!("prism.hadamard.version {version} unsupported (only 1) — refusing to load a folded file we cannot honor");
    }

    let block_size = gguf
        .metadata_u64("prism.hadamard.block_size")
        .context("prism.hadamard present but block_size missing")? as usize;
    if block_size == 0 || !block_size.is_power_of_two() {
        bail!("prism.hadamard.block_size {block_size} invalid (must be a power of 2)");
    }

    let transform = gguf
        .metadata_string("prism.hadamard.transform")
        .context("prism.hadamard.transform missing")?;
    if transform != "normalized-sylvester-walsh-hadamard" {
        bail!("prism.hadamard.transform '{transform}' unsupported (only normalized-sylvester-walsh-hadamard)");
    }

    let axis = gguf
        .metadata_string("prism.hadamard.axis")
        .context("prism.hadamard.axis missing")?;
    if axis != "input-last-dimension" {
        bail!("prism.hadamard.axis '{axis}' unsupported (only input-last-dimension)");
    }

    let sign_mode = gguf
        .metadata_string("prism.hadamard.sign_mode")
        .context("prism.hadamard.sign_mode missing")?;
    let explicit = match sign_mode {
        "identity" => false,
        "explicit" => true,
        other => bail!("prism.hadamard.sign_mode '{other}' unsupported (identity|explicit)"),
    };

    let weight_names = gguf
        .metadata_array("prism.hadamard.weight_names")
        .context("prism.hadamard.weight_names missing")?;
    if weight_names.is_empty() {
        bail!("prism.hadamard.weight_names is empty");
    }

    // Every folded weight must (a) exist, (b) be 2D, (c) have an input dim
    // (shape[0] = ggml ne[0]) divisible by the block size. The foldable-kind
    // allowlist is enforced by the CALLER (it knows the structural map).
    for name_val in weight_names {
        let name = name_val
            .as_str()
            .with_context(|| format!("prism.hadamard weight name {name_val:?} not a string"))?;
        let info = gguf
            .tensor_info(name)
            .with_context(|| format!("prism.hadamard weight '{name}' not found in file"))?;
        anyhow::ensure!(
            info.shape.len() == 2,
            "prism.hadamard weight '{name}' has {} dims, expected 2D",
            info.shape.len()
        );
        let in_dim = info.shape[0];
        anyhow::ensure!(
            in_dim % block_size == 0,
            "prism.hadamard block {block_size} does not divide input dim {in_dim} of '{name}'"
        );
    }

    // Sign data: explicit mode requires sign_widths + sign_values covering
    // every distinct folded input width; identity mode must not carry them.
    let signs = if explicit {
        let widths = gguf
            .metadata_array("prism.hadamard.sign_widths")
            .context("sign_mode explicit but prism.hadamard.sign_widths missing")?;
        let values = gguf
            .metadata_array("prism.hadamard.sign_values")
            .context("sign_mode explicit but prism.hadamard.sign_values missing")?;
        let mut widths_us: Vec<usize> = Vec::with_capacity(widths.len());
        for w in widths {
            let w = w
                .as_u64()
                .with_context(|| format!("prism.hadamard sign width {w:?} not an integer"))?;
            anyhow::ensure!(w > 0, "prism.hadamard sign width must be positive");
            widths_us.push(w as usize);
        }
        let total: usize = widths_us.iter().sum();
        anyhow::ensure!(
            values.len() == total,
            "prism.hadamard sign_values length {} != sum(sign_widths) {total}",
            values.len()
        );
        let mut signs = Vec::with_capacity(widths_us.len());
        let mut off = 0usize;
        for w in &widths_us {
            let mut vec = vec![0i8; *w];
            for (i, slot) in vec.iter_mut().enumerate() {
                let v = values[off + i]
                    .as_f64()
                    .with_context(|| format!("prism.hadamard sign value {:?} not numeric", values[off + i]))?;
                anyhow::ensure!(
                    v == 1.0 || v == -1.0,
                    "prism.hadamard sign value {v} not ±1"
                );
                *slot = if v < 0.0 { -1 } else { 1 };
            }
            off += *w;
            signs.push((*w, vec));
        }
        signs
    } else {
        Vec::new()
    };

    let gdn_v_grouped = gguf.metadata_u64("prism.hadamard.gdn_v_grouped");
    let (gdn_v_heads, gdn_k_groups) = if gdn_v_grouped.unwrap_or(0) != 0 {
        // The permute geometry comes from the ssm hyperparameters the config
        // was derived from.
        let n_v = gguf
            .metadata_u64("qwen35.ssm.time_step_rank")
            .context("gdn_v_grouped set but qwen35.ssm.time_step_rank missing")? as usize;
        let n_k = gguf
            .metadata_u64("qwen35.ssm.group_count")
            .context("gdn_v_grouped set but qwen35.ssm.group_count missing")? as usize;
        anyhow::ensure!(
            n_k > 0 && n_v.is_multiple_of(n_k),
            "gdn_v_grouped geometry invalid: n_v {n_v} not divisible by n_k {n_k}"
        );
        (n_v, n_k)
    } else {
        (0, 0)
    };

    // Inverse (lookup-side) tables. Only the token embedding is verified by
    // the fork; anything else would load and silently stay rotated.
    let mut inverse_embedding = false;
    if let Some(inverses) = gguf.metadata_array("prism.hadamard.inverse_weight_names") {
        for name in inverses {
            let name = name
                .as_str()
                .with_context(|| format!("prism.hadamard inverse name {name:?} not a string"))?;
            if name != "token_embd.weight" {
                bail!("prism.hadamard inverse weight '{name}' unsupported (only token_embd.weight)");
            }
            inverse_embedding = true;
        }
    }

    Ok(Some(TernaryRotationConfig {
        block_size,
        signs,
        gdn_v_grouped: gdn_v_grouped.unwrap_or(0) != 0,
        gdn_v_heads,
        gdn_k_groups,
        inverse_embedding,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FWHT equals the fork's explicit ±1/√n Hadamard matrix multiply:
    /// entry (r, c) is `parity(r & c) · (-1)^... / √n` (llama-model.cpp:2000).
    /// Index loops keep the (row, col) semantics of the reference matrix
    /// literal — the thing under test.
    #[allow(clippy::needless_range_loop)]
    #[test]
    fn fwht_matches_explicit_hadamard_matrix() {
        for n in [2usize, 4, 8, 64, 256, 1024] {
            let scale = 1.0 / (n as f32).sqrt();
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 97) as f32 - 48.0).collect();
            let mut got = x.clone();
            fwht_block_inplace(&mut got);
            for r in 0..n {
                let mut want = 0.0f32;
                for c in 0..n {
                    let mut parity = r & c;
                    parity ^= parity >> 16;
                    parity ^= parity >> 8;
                    parity ^= parity >> 4;
                    parity ^= parity >> 2;
                    parity ^= parity >> 1;
                    want += x[c] * if parity & 1 == 1 { -scale } else { scale };
                }
                assert!(
                    (got[r] - want).abs() < 1e-3,
                    "n={n} row {r}: got {} want {want}",
                    got[r]
                );
            }
        }
    }

    /// Self-inverse (unitary): applying twice restores the input.
    #[test]
    fn fwht_is_self_inverse() {
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 31) as f32).collect();
        let mut buf = x.clone();
        fwht_block_inplace(&mut buf);
        fwht_block_inplace(&mut buf);
        for (a, b) in buf.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    /// The embedding-store contract: the table rows are latent because they
    /// were folded with the SAME forward transform (`z = (1/√n)·H·(S⊙e)`),
    /// so the lookup's inverse rotation (`S⊙(H·z/√n)`) recovers the primal
    /// embedding exactly.
    #[test]
    fn embedding_inverse_recovers_primal() {
        let n = 1024;
        let signs: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let e: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 23) as f32 - 11.0).collect();

        // Store: z = forward-rotate(e) — the latent (rotated) embedding row.
        let mut z = e.clone();
        rotate_forward_inplace(&mut z, Some(&signs), n);

        // Runtime inverse: e' = S⊙(H z).
        rotate_inverse_inplace(&mut z, Some(&signs), n);
        for (a, b) in z.iter().zip(e.iter()) {
            assert!((a - b).abs() < 1e-3, "got {a} want {b}");
        }
    }

    /// Forward transform is a pure rotation: ||Rx|| == ||x||.
    #[test]
    fn forward_rotation_preserves_norm() {
        let n = 2048;
        let block = 1024;
        let signs: Vec<i8> = (0..n).map(|i| if i % 5 == 0 { -1 } else { 1 }).collect();
        let x: Vec<f32> = (0..n).map(|i| ((i * 17) % 41) as f32).collect();
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut buf = x;
        rotate_forward_inplace(&mut buf, Some(&signs), block);
        let norm_after: f32 = buf.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() / norm_before < 1e-5);
    }

    /// The GDN V permute moves whole head segments: grouped[tiled head
    /// (nk, rep)] — head `rep*n_k + nk` must land at `nk*rep + rep`.
    #[test]
    fn gdn_v_permute_moves_head_segments() {
        let hd = 4usize;
        let n_k = 2usize;
        let rep = 3usize;
        let n_v = n_k * rep; // 6 heads
        let len = hd * n_v;
        let x: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let mut buf = x.clone();
        let mut tmp = vec![0.0f32; len];
        permute_gdn_v_grouped_inplace(&mut buf, &mut tmp, n_v, n_k);
        for nk in 0..n_k {
            for r in 0..rep {
                let src_head = r * n_k + nk;
                let dst_head = nk * rep + r;
                for d in 0..hd {
                    assert_eq!(
                        buf[dst_head * hd + d],
                        x[src_head * hd + d],
                        "head (nk={nk}, rep={r}) moved to the wrong slot"
                    );
                }
            }
        }
    }

    /// The folded-name allowlist resolves per layer type (full-attention
    /// interval 4 ⇒ layers 3/7/... are attention, the rest `DeltaNet`).
    #[test]
    fn folded_name_allowlist_matches_layer_types() {
        use crate::types::DeltaNetLayerType;
        let layer_types: Vec<DeltaNetLayerType> = (0..8)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    DeltaNetLayerType::Attention
                } else {
                    DeltaNetLayerType::DeltaNet
                }
            })
            .collect();
        let n = layer_types.len();
        assert!(is_known_folded_name("output.weight", n, &layer_types));
        assert!(is_known_folded_name("blk.0.attn_qkv.weight", n, &layer_types));
        assert!(is_known_folded_name("blk.0.ssm_out.weight", n, &layer_types));
        assert!(is_known_folded_name("blk.0.attn_gate.weight", n, &layer_types));
        assert!(is_known_folded_name("blk.3.attn_q.weight", n, &layer_types));
        assert!(is_known_folded_name("blk.3.attn_output.weight", n, &layer_types));
        assert!(is_known_folded_name("blk.3.ffn_down.weight", n, &layer_types));
        // Cross-type refusals: DeltaNet kinds on an attention layer and
        // vice versa.
        assert!(!is_known_folded_name("blk.3.attn_qkv.weight", n, &layer_types));
        assert!(!is_known_folded_name("blk.3.ssm_out.weight", n, &layer_types));
        assert!(!is_known_folded_name("blk.0.attn_q.weight", n, &layer_types));
        // Never-foldable names.
        assert!(!is_known_folded_name("token_embd.weight", n, &layer_types));
        assert!(!is_known_folded_name("blk.0.ssm_alpha.weight", n, &layer_types));
        assert!(!is_known_folded_name("blk.0.ssm_beta.weight", n, &layer_types));
        assert!(!is_known_folded_name("blk.64.ffn_up.weight", n, &layer_types));
        assert!(!is_known_folded_name("blk.0.bogus.weight", n, &layer_types));
    }
}
