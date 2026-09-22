//! Q+V `LoRA` adapter for ternary `DeltaNet` layers (Plan 334 T2.3 / Issue 448 T7).
//!
//! Rank-`r` `LoRA` on the compact Q and V projections of one `DeltaNet` layer.
//! This is the trainable parameter set for arms B (mid-layer) and C (full-layer)
//! of Plan 334. The frozen ternary backbone provides `norm_x` (the input to the
//! QKV projection); the `LoRA` adds a low-rank perturbation to the Q and V
//! portions of the compact QKV output.
//!
//! # Math
//!
//! Forward (the `LoRA` delta added to the frozen backbone's compact QKV):
//! ```text
//! ax_q = A_q @ norm_x          // [r]      (down-projection, Q)
//! δq   = α/r · B_q @ ax_q      // [q_dim]  (up-projection, Q)
//! ax_v = A_v @ norm_x          // [r]      (down-projection, V)
//! δv   = α/r · B_v @ ax_v      // [v_dim]  (up-projection, V)
//! qkv_compact_perturbed = qkv_compact_frozen + [δq | 0 | δv]
//! ```
//!
//! Backward (from [`riir_train_engine::deltanet::model_backward::LoraTargetGrad`]):
//! ```text
//! // grad_q_compact = dL/d(q_compact)  [q_dim]  (from model_backward)
//! // grad_v_compact = dL/d(v_compact)  [v_dim]  (from model_backward)
//!
//! grad_B_q[i,j] = scale · grad_q_compact[i] · ax_q[j]
//! grad_A_q[j,k] = scale · bta_q[j] · norm_x[k]   where bta_q = B_q^T @ grad_q_compact
//! grad_B_v[i,j] = scale · grad_v_compact[i] · ax_v[j]
//! grad_A_v[j,k] = scale · bta_v[j] · norm_x[k]   where bta_v = B_v^T @ grad_v_compact
//! ```
//!
//! # Initialization
//!
//! Standard `LoRA` init: `A ~ N(0, 1/√n_embd)`, `B = 0`. The adapter produces zero
//! output at step 0 (identity behavior), then learns.
//!
//! # Parameter layout (flat, for `AdamW`)
//!
//! ```text
//! [A_q (r × n_embd) | B_q (q_dim × r) | A_v (r × n_embd) | B_v (v_dim × r)]
//! ```
//! Total: `2·r·n_embd + r·(q_dim + v_dim)` params per layer.

#![allow(clippy::needless_range_loop)]

use crate::types::Rng;

// ─────────────────────────────────────────────────────────────────────────
// QvLora — one layer's Q+V LoRA adapter
// ─────────────────────────────────────────────────────────────────────────

/// Rank-`r` Q+V `LoRA` adapter for one `DeltaNet` layer.
///
/// `A_q` is `[r × n_embd]` (down-projection for Q), `B_q` is `[q_dim × r]`
/// (up-projection for Q). Similarly `A_v`, `B_v` for V.
///
/// `q_dim = n_k_heads · head_dim`, `v_dim = n_v_heads · head_dim`.
#[derive(Clone, Debug)]
pub struct QvLora {
    /// Down-projection for Q `[r × n_embd]` (row-major).
    pub a_q: Vec<f32>,
    /// Up-projection for Q `[q_dim × r]` (row-major).
    pub b_q: Vec<f32>,
    /// Down-projection for V `[r × n_embd]` (row-major).
    pub a_v: Vec<f32>,
    /// Up-projection for V `[v_dim × r]` (row-major).
    pub b_v: Vec<f32>,

    pub rank: usize,
    pub n_embd: usize,
    pub q_dim: usize,
    pub v_dim: usize,
    /// `LoRA` alpha (the adapter output is scaled by `alpha / rank`).
    pub alpha: f32,
}

impl QvLora {
    /// Create a new Q+V `LoRA` adapter with standard initialization.
    ///
    /// `A_q`, `A_v` are initialized with small Gaussian noise; `B_q`, `B_v` are
    /// zero — so the adapter produces zero output at step 0 (identity behavior).
    pub fn new(
        rank: usize,
        n_embd: usize,
        q_dim: usize,
        v_dim: usize,
        alpha: f32,
        rng: &mut Rng,
    ) -> Self {
        let std_a = 1.0 / (n_embd as f32).sqrt();
        let a_q = (0..rank * n_embd).map(|_| rng.normal() * std_a).collect();
        let b_q = vec![0.0; q_dim * rank];
        let a_v = (0..rank * n_embd).map(|_| rng.normal() * std_a).collect();
        let b_v = vec![0.0; v_dim * rank];

        Self {
            a_q,
            b_q,
            a_v,
            b_v,
            rank,
            n_embd,
            q_dim,
            v_dim,
            alpha,
        }
    }

    /// The `LoRA` scaling factor: `alpha / rank`.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Total parameter count (`A_q` + `B_q` + `A_v` + `B_v`).
    #[inline]
    pub fn n_params(&self) -> usize {
        2 * self.rank * self.n_embd + self.rank * (self.q_dim + self.v_dim)
    }

    /// Compute the Q+V `LoRA` deltas for one token.
    ///
    /// Writes `δq[i] = scale · Σ_r B_q[i,r] · ax_q[r]` into `delta_q`,
    /// and `δv[i] = scale · Σ_r B_v[i,r] · ax_v[r]` into `delta_v`.
    ///
    /// Also writes `ax_q = A_q @ norm_x` and `ax_v = A_v @ norm_x` into the
    /// scratch buffers (needed for the backward pass — caller should cache).
    ///
    /// `delta_q.len()` must equal `q_dim`; `delta_v.len()` must equal `v_dim`;
    /// `ax_q.len()` and `ax_v.len()` must equal `rank`.
    pub fn forward_qv_delta(
        &self,
        norm_x: &[f32],
        delta_q: &mut [f32],
        delta_v: &mut [f32],
        ax_q: &mut [f32],
        ax_v: &mut [f32],
    ) {
        debug_assert_eq!(norm_x.len(), self.n_embd);
        debug_assert_eq!(delta_q.len(), self.q_dim);
        debug_assert_eq!(delta_v.len(), self.v_dim);
        debug_assert_eq!(ax_q.len(), self.rank);
        debug_assert_eq!(ax_v.len(), self.rank);

        let scale = self.scale();
        let r = self.rank;
        let n = self.n_embd;

        // ax_q = A_q @ norm_x  [r]
        for ri in 0..r {
            let row = &self.a_q[ri * n..(ri + 1) * n];
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += row[j] * norm_x[j];
            }
            ax_q[ri] = sum;
        }

        // ax_v = A_v @ norm_x  [r]
        for ri in 0..r {
            let row = &self.a_v[ri * n..(ri + 1) * n];
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += row[j] * norm_x[j];
            }
            ax_v[ri] = sum;
        }

        // delta_q = scale * B_q @ ax_q  [q_dim]
        for i in 0..self.q_dim {
            let row = &self.b_q[i * r..(i + 1) * r];
            let mut sum = 0.0f32;
            for ri in 0..r {
                sum += row[ri] * ax_q[ri];
            }
            delta_q[i] = scale * sum;
        }

        // delta_v = scale * B_v @ ax_v  [v_dim]
        for i in 0..self.v_dim {
            let row = &self.b_v[i * r..(i + 1) * r];
            let mut sum = 0.0f32;
            for ri in 0..r {
                sum += row[ri] * ax_v[ri];
            }
            delta_v[i] = scale * sum;
        }
    }

    /// Apply the Q+V `LoRA` perturbation to a compact QKV buffer in-place.
    ///
    /// Layout of `qkv_compact`: `[q_dim | k_dim | v_dim]` where `k_dim` occupies
    /// the middle slice `[q_dim..q_dim+k_dim]`. Only the Q and V slices are
    /// perturbed; K is untouched.
    ///
    /// `k_dim = n_k_heads · head_dim` (= `q_dim` for Qwen3.5 `DeltaNet`, but passed
    /// explicitly for clarity).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_to_qkv_compact(
        &self,
        norm_x: &[f32],
        qkv_compact: &mut [f32],
        k_dim: usize,
        ax_q: &mut [f32],
        ax_v: &mut [f32],
        delta_q_scratch: &mut [f32],
        delta_v_scratch: &mut [f32],
    ) {
        debug_assert_eq!(qkv_compact.len(), self.q_dim + k_dim + self.v_dim);

        self.forward_qv_delta(norm_x, delta_q_scratch, delta_v_scratch, ax_q, ax_v);

        // Add δq to the Q slice [0..q_dim]
        for i in 0..self.q_dim {
            qkv_compact[i] += delta_q_scratch[i];
        }

        // Add δv to the V slice [q_dim+k_dim .. q_dim+k_dim+v_dim]
        let v_start = self.q_dim + k_dim;
        for i in 0..self.v_dim {
            qkv_compact[v_start + i] += delta_v_scratch[i];
        }
    }

    // ── Flat parameter view (for AdamW) ──

    /// Flatten all params into a single contiguous slice (for `AdamW`).
    ///
    /// Layout: `[A_q | B_q | A_v | B_v]` — matches the gradient layout produced
    /// by [`lora_qv_backward_flat`].
    pub fn flatten_into(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.n_params());
        let mut offset = 0;
        out[offset..offset + self.a_q.len()].copy_from_slice(&self.a_q);
        offset += self.a_q.len();
        out[offset..offset + self.b_q.len()].copy_from_slice(&self.b_q);
        offset += self.b_q.len();
        out[offset..offset + self.a_v.len()].copy_from_slice(&self.a_v);
        offset += self.a_v.len();
        out[offset..offset + self.b_v.len()].copy_from_slice(&self.b_v);
    }

    /// Unflatten from a contiguous slice back into the struct's fields.
    pub fn unflatten_from(&mut self, src: &[f32]) {
        debug_assert_eq!(src.len(), self.n_params());
        let mut offset = 0;
        let a_q_len = self.a_q.len();
        self.a_q.copy_from_slice(&src[offset..offset + a_q_len]);
        offset += a_q_len;
        let b_q_len = self.b_q.len();
        self.b_q.copy_from_slice(&src[offset..offset + b_q_len]);
        offset += b_q_len;
        let a_v_len = self.a_v.len();
        self.a_v.copy_from_slice(&src[offset..offset + a_v_len]);
        offset += a_v_len;
        let b_v_len = self.b_v.len();
        self.b_v.copy_from_slice(&src[offset..offset + b_v_len]);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Save / Load
// ─────────────────────────────────────────────────────────────────────────

/// Binary serialization format magic for Q+V `LoRA`.
const QV_LORA_MAGIC: &[u8; 4] = b"BQVL"; // Bonsai Q+V LoRA

/// Save a `QvLora` to disk. Format:
/// ```text
/// magic     : 4 bytes  = "BQVL"
/// version   : u32       = 1
/// rank      : u32
/// n_embd    : u32
/// q_dim     : u32
/// v_dim     : u32
/// alpha     : f32
/// A_q data  : rank * n_embd * 4 bytes
/// B_q data  : q_dim * rank * 4 bytes
/// A_v data  : rank * n_embd * 4 bytes
/// B_v data  : v_dim * rank * 4 bytes
/// ```
pub fn save_qv_lora(lora: &QvLora, path: &std::path::Path) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(QV_LORA_MAGIC)?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(lora.rank as u32).to_le_bytes())?;
    f.write_all(&(lora.n_embd as u32).to_le_bytes())?;
    f.write_all(&(lora.q_dim as u32).to_le_bytes())?;
    f.write_all(&(lora.v_dim as u32).to_le_bytes())?;
    f.write_all(&lora.alpha.to_le_bytes())?;
    f.write_all(bytemuck::cast_slice(&lora.a_q))?;
    f.write_all(bytemuck::cast_slice(&lora.b_q))?;
    f.write_all(bytemuck::cast_slice(&lora.a_v))?;
    f.write_all(bytemuck::cast_slice(&lora.b_v))?;
    Ok(())
}

/// Load a `QvLora` from disk (inverse of [`save_qv_lora`]).
pub fn load_qv_lora(path: &std::path::Path) -> anyhow::Result<QvLora> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == QV_LORA_MAGIC, "bad magic: expected BQVL, got {magic:?}");

    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    anyhow::ensure!(version == 1, "unsupported QvLora version: {version}");

    f.read_exact(&mut buf4)?;
    let rank = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4)?;
    let n_embd = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4)?;
    let q_dim = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4)?;
    let v_dim = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4)?;
    let alpha = f32::from_le_bytes(buf4);

    let read_vec = |f: &mut std::fs::File, n: usize| -> anyhow::Result<Vec<f32>> {
        let mut bytes = vec![0u8; n * 4];
        f.read_exact(&mut bytes)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    };

    let a_q = read_vec(&mut f, rank * n_embd)?;
    let b_q = read_vec(&mut f, q_dim * rank)?;
    let a_v = read_vec(&mut f, rank * n_embd)?;
    let b_v = read_vec(&mut f, v_dim * rank)?;

    Ok(QvLora {
        a_q,
        b_q,
        a_v,
        b_v,
        rank,
        n_embd,
        q_dim,
        v_dim,
        alpha,
    })
}

/// Binary serialization format magic for the Plan 334 training-driver
/// checkpoint ("`LoRA` arm-C collection" — written by
/// `bonsai_lora_accuracy_parity_arm_c.rs::LoraCollection::save`).
const LORC_MAGIC: &[u8; 4] = b"LORC";

/// A LORC v1 checkpoint: per-layer Q+V adapters sharing one rank/alpha.
///
/// The format does NOT store `n_embd` / `q_dim` / `v_dim` — they are implied
/// by the model config and must be passed to [`load_lorc`]. Arm B (single
/// target layer) is the production shape; Arm C checkpoints carry all
/// `DeltaNet` layers.
#[derive(Clone, Debug)]
pub struct LorcCheckpoint {
    /// Shared `LoRA` rank.
    pub rank: usize,
    /// Shared `LoRA` alpha.
    pub alpha: f32,
    /// `(layer_idx, adapter)` — one entry per `LoRA` target layer, in file order.
    pub layers: Vec<(usize, QvLora)>,
}

impl LorcCheckpoint {
    /// Consume into the single-layer (Arm B) adapter.
    ///
    /// Errors if the checkpoint holds anything other than exactly one target
    /// layer — the Issue 666 GPU decode path attaches one adapter at one layer.
    pub fn into_single_layer(mut self) -> anyhow::Result<(usize, QvLora)> {
        anyhow::ensure!(
            self.layers.len() == 1,
            "LORC checkpoint holds {} target layers; expected exactly 1 \
             (Arm B single-layer serving — the GPU decode path attaches one adapter)",
            self.layers.len()
        );
        Ok(self.layers.remove(0))
    }
}

/// Load a LORC v1 checkpoint written by the Plan 334 training driver.
///
/// `n_embd` / `q_dim` / `v_dim` come from the model config (`q_dim =
/// n_k_heads · head_dim`, `v_dim = n_v_heads · head_dim`) because the format
/// does not store them. Errors on magic/version mismatch, dimension mismatch,
/// truncated weight data, or trailing bytes.
pub fn load_lorc(
    path: &std::path::Path,
    n_embd: usize,
    q_dim: usize,
    v_dim: usize,
) -> anyhow::Result<LorcCheckpoint> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    anyhow::ensure!(
        &magic == LORC_MAGIC,
        "bad magic: expected LORC, got {magic:?}"
    );

    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    anyhow::ensure!(version == 1, "unsupported LORC version: {version}");

    f.read_exact(&mut buf4)?;
    let rank = u32::from_le_bytes(buf4) as usize;
    anyhow::ensure!(rank >= 1, "LORC rank must be >= 1, got {rank}");
    f.read_exact(&mut buf4)?;
    let alpha = f32::from_le_bytes(buf4);
    f.read_exact(&mut buf4)?;
    let n_layers = u32::from_le_bytes(buf4) as usize;
    anyhow::ensure!(n_layers >= 1, "LORC checkpoint holds no target layers");

    let mut layer_indices = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        f.read_exact(&mut buf4)?;
        layer_indices.push(u32::from_le_bytes(buf4) as usize);
    }

    let read_vec = |f: &mut std::fs::File, n: usize| -> anyhow::Result<Vec<f32>> {
        let mut bytes = vec![0u8; n * 4];
        f.read_exact(&mut bytes)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    };

    let mut layers = Vec::with_capacity(n_layers);
    for &idx in &layer_indices {
        let a_q = read_vec(&mut f, rank * n_embd)?;
        let b_q = read_vec(&mut f, q_dim * rank)?;
        let a_v = read_vec(&mut f, rank * n_embd)?;
        let b_v = read_vec(&mut f, v_dim * rank)?;
        layers.push((
            idx,
            QvLora {
                a_q,
                b_q,
                a_v,
                b_v,
                rank,
                n_embd,
                q_dim,
                v_dim,
                alpha,
            },
        ));
    }

    // Strict EOF check — a well-formed LORC file has no trailing bytes.
    let mut trailing = [0u8; 1];
    anyhow::ensure!(
        f.read(&mut trailing).unwrap_or(0) == 0,
        "LORC checkpoint has trailing bytes — dimension mismatch vs the model config?"
    );

    Ok(LorcCheckpoint {
        rank,
        alpha,
        layers,
    })
}

/// Save a LORC v1 checkpoint (byte-identical to the training driver's
/// `LoraCollection::save` for the same weights).
///
/// All adapters must share `rank` and `alpha` (the format stores them once).
pub fn save_lorc(
    path: &std::path::Path,
    layers: &[(usize, &QvLora)],
    rank: usize,
    alpha: f32,
) -> anyhow::Result<()> {
    use std::io::Write;
    anyhow::ensure!(!layers.is_empty(), "cannot save an empty LORC checkpoint");
    for (i, (_, l)) in layers.iter().enumerate() {
        anyhow::ensure!(
            l.rank == rank,
            "layer {i} rank {} != shared rank {rank}",
            l.rank
        );
        anyhow::ensure!(
            (l.alpha - alpha).abs() < f32::EPSILON,
            "layer {i} alpha {} != shared alpha {alpha}",
            l.alpha
        );
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(LORC_MAGIC)?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(rank as u32).to_le_bytes())?;
    f.write_all(&alpha.to_le_bytes())?;
    f.write_all(&(layers.len() as u32).to_le_bytes())?;
    for &(idx, _) in layers {
        f.write_all(&(idx as u32).to_le_bytes())?;
    }
    for &(_, l) in layers {
        f.write_all(bytemuck::cast_slice(&l.a_q))?;
        f.write_all(bytemuck::cast_slice(&l.b_q))?;
        f.write_all(bytemuck::cast_slice(&l.a_v))?;
        f.write_all(bytemuck::cast_slice(&l.b_v))?;
    }
    Ok(())
}

// (LORC tests live in the `tests` module at the bottom of this file, next to
// the QvLora tests.)

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> Rng {
        Rng::new(42)
    }

    /// Issue 668 — a LORC v1 file written by an INDEPENDENT byte-builder that
    /// mirrors the documented format must load back exactly. This pins the
    /// wire format against drift in either direction (loader or writer).
    #[test]
    fn lorc_format_roundtrip() {
        let rank = 8usize;
        let n_embd = 64usize;
        let q_dim = 32usize;
        let v_dim = 96usize;
        let alpha = 16.0f32;

        let mut rng = Rng::new(1234);
        let mut lora = QvLora::new(rank, n_embd, q_dim, v_dim, alpha, &mut rng);
        // Random-init leaves B zero — perturb so roundtrip checks real data.
        for v in lora.b_q.iter_mut() {
            *v = rng.normal() * 0.5;
        }
        for v in lora.b_v.iter_mut() {
            *v = rng.normal() * 0.5;
        }

        // Independent writer (mirrors `LoraCollection::save` byte layout).
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"LORC");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(rank as u32).to_le_bytes());
        bytes.extend_from_slice(&alpha.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one target layer
        bytes.extend_from_slice(&32u32.to_le_bytes()); // layer 32 (Arm B)
        for src in [&lora.a_q, &lora.b_q, &lora.a_v, &lora.b_v] {
            for v in src.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }

        let dir = std::env::temp_dir().join(format!("riir_lorc_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("format_pin.lorc");
        std::fs::write(&path, &bytes).unwrap();

        let ckpt = load_lorc(&path, n_embd, q_dim, v_dim).expect("load_lorc");
        assert_eq!(ckpt.rank, rank);
        assert_eq!(ckpt.alpha, alpha);
        assert_eq!(ckpt.layers.len(), 1);
        let (idx, loaded) = ckpt.into_single_layer().expect("into_single_layer");
        assert_eq!(idx, 32);
        assert_eq!(loaded.rank, rank);
        assert_eq!(loaded.n_embd, n_embd);
        assert_eq!(loaded.q_dim, q_dim);
        assert_eq!(loaded.v_dim, v_dim);
        assert_eq!(loaded.alpha, alpha);
        assert_eq!(loaded.a_q, lora.a_q);
        assert_eq!(loaded.b_q, lora.b_q);
        assert_eq!(loaded.a_v, lora.a_v);
        assert_eq!(loaded.b_v, lora.b_v);
    }

    /// Issue 668 — `save_lorc` → `load_lorc` roundtrip (multi-layer) + strict EOF.
    #[test]
    fn lorc_save_load_roundtrip() {
        let rank = 4usize;
        let n_embd = 64usize;
        let q_dim = 32usize;
        let v_dim = 96usize;
        let alpha = 8.0f32;

        let mut rng = Rng::new(99);
        let l0 = QvLora::new(rank, n_embd, q_dim, v_dim, alpha, &mut rng);
        let mut l1 = QvLora::new(rank, n_embd, q_dim, v_dim, alpha, &mut rng);
        for v in l1.b_q.iter_mut() {
            *v = rng.normal();
        }

        let dir = std::env::temp_dir().join(format!("riir_lorc_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.lorc");

        let layers: Vec<(usize, &QvLora)> = vec![(7, &l0), (32, &l1)];
        save_lorc(&path, &layers, rank, alpha).expect("save_lorc");
        let ckpt = load_lorc(&path, n_embd, q_dim, v_dim).expect("load_lorc");
        assert_eq!(ckpt.layers.len(), 2);
        assert_eq!(ckpt.layers[0].0, 7);
        assert_eq!(ckpt.layers[1].0, 32);
        assert_eq!(ckpt.layers[0].1.a_q, l0.a_q);
        assert_eq!(ckpt.layers[1].1.b_q, l1.b_q);

        // Dimension mismatch must be rejected (here: wrong q_dim → the strict
        // EOF check fires because the read consumed the wrong byte count).
        let bad = load_lorc(&path, n_embd, q_dim + 16, v_dim);
        assert!(bad.is_err(), "mismatched dims must fail");

        // Multi-layer checkpoints cannot serve as single-layer (Arm B).
        assert!(ckpt.clone().into_single_layer().is_err());
    }

    #[test]
    fn test_qv_lora_init_zero_output() {
        // At init, B = 0, so the LoRA delta should be zero.
        let mut rng = make_rng();
        let lora = QvLora::new(4, 16, 8, 12, 8.0, &mut rng);
        let norm_x = vec![1.0; 16];
        let mut delta_q = vec![0.0; 8];
        let mut delta_v = vec![0.0; 12];
        let mut ax_q = vec![0.0; 4];
        let mut ax_v = vec![0.0; 4];
        lora.forward_qv_delta(&norm_x, &mut delta_q, &mut delta_v, &mut ax_q, &mut ax_v);

        for &d in &delta_q {
            assert!(d.abs() < 1e-7, "Q delta should be zero at init, got {d}");
        }
        for &d in &delta_v {
            assert!(d.abs() < 1e-7, "V delta should be zero at init, got {d}");
        }
    }

    #[test]
    fn test_qv_lora_forward_correctness() {
        // With known A, B, norm_x, verify the delta matches manual computation.
        let mut rng = make_rng();
        let rank = 2;
        let n_embd = 4;
        let q_dim = 3;
        let v_dim = 5;
        let alpha = 4.0;
        let mut lora = QvLora::new(rank, n_embd, q_dim, v_dim, alpha, &mut rng);

        // Set B to non-zero so the delta is non-zero.
        for i in 0..lora.b_q.len() {
            lora.b_q[i] = (i as f32) * 0.1;
        }
        for i in 0..lora.b_v.len() {
            lora.b_v[i] = (i as f32) * 0.05;
        }

        let norm_x: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1];

        let mut delta_q = vec![0.0; q_dim];
        let mut delta_v = vec![0.0; v_dim];
        let mut ax_q = vec![0.0; rank];
        let mut ax_v = vec![0.0; rank];
        lora.forward_qv_delta(&norm_x, &mut delta_q, &mut delta_v, &mut ax_q, &mut ax_v);

        // Manual check: ax_q[0] = A_q[0,:] · norm_x
        let manual_ax_q0: f32 = lora.a_q[0..4]
            .iter()
            .zip(norm_x.iter())
            .map(|(a, x)| a * x)
            .sum();
        assert!((ax_q[0] - manual_ax_q0).abs() < 1e-6, "ax_q[0] mismatch");

        // delta_q[0] = scale * B_q[0,:] · ax_q
        let scale = alpha / rank as f32;
        let manual_dq0: f32 = lora.b_q[0..2]
            .iter()
            .zip(ax_q.iter())
            .map(|(b, a)| b * a)
            .sum::<f32>()
            * scale;
        assert!((delta_q[0] - manual_dq0).abs() < 1e-6, "delta_q[0] mismatch");
    }

    #[test]
    fn test_qv_lora_apply_to_qkv_compact() {
        let mut rng = make_rng();
        let rank = 2;
        let n_embd = 4;
        let q_dim = 3;
        let k_dim = 3; // Same as q_dim for Qwen3.5
        let v_dim = 5;
        let mut lora = QvLora::new(rank, n_embd, q_dim, v_dim, 4.0, &mut rng);

        // Set B to non-zero
        for i in 0..lora.b_q.len() {
            lora.b_q[i] = 0.1;
        }
        for i in 0..lora.b_v.len() {
            lora.b_v[i] = 0.2;
        }

        let norm_x = vec![0.5; n_embd];
        let conv_dim = q_dim + k_dim + v_dim;
        let mut qkv = vec![1.0; conv_dim];
        let qkv_orig = qkv.clone();

        let mut ax_q = vec![0.0; rank];
        let mut ax_v = vec![0.0; rank];
        let mut dq = vec![0.0; q_dim];
        let mut dv = vec![0.0; v_dim];
        lora.apply_to_qkv_compact(&norm_x, &mut qkv, k_dim, &mut ax_q, &mut ax_v, &mut dq, &mut dv);

        // K slice [q_dim..q_dim+k_dim] should be unchanged
        for i in q_dim..q_dim + k_dim {
            assert!((qkv[i] - qkv_orig[i]).abs() < 1e-10, "K should be unchanged");
        }

        // Q and V slices should be perturbed
        let scale = lora.scale();
        for i in 0..q_dim {
            let expected_delta = scale * 0.1 * ax_q.iter().sum::<f32>();
            assert!((qkv[i] - qkv_orig[i] - expected_delta).abs() < 1e-6, "Q perturbation wrong at {i}");
        }
    }

    #[test]
    fn test_flatten_unflatten_roundtrip() {
        let mut rng = make_rng();
        let lora = QvLora::new(4, 8, 6, 10, 8.0, &mut rng);
        let n = lora.n_params();
        let mut flat = vec![0.0; n];
        lora.flatten_into(&mut flat);

        // Mutate the flat buffer to verify unflatten reads it
        flat.fill(1.5);

        let mut lora2 = lora.clone();
        lora2.unflatten_from(&flat);
        // All params should be 1.5
        assert!(lora2.a_q.iter().all(|&v| v == 1.5));
        assert!(lora2.b_q.iter().all(|&v| v == 1.5));
        assert!(lora2.a_v.iter().all(|&v| v == 1.5));
        assert!(lora2.b_v.iter().all(|&v| v == 1.5));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut rng = make_rng();
        let lora = QvLora::new(4, 8, 6, 10, 8.0, &mut rng);
        let path = std::env::temp_dir().join(format!(
            "test_qv_lora_roundtrip_{}.bin",
            std::process::id()
        ));
        save_qv_lora(&lora, &path).unwrap();
        let loaded = load_qv_lora(&path).unwrap();
        assert_eq!(loaded.rank, lora.rank);
        assert_eq!(loaded.n_embd, lora.n_embd);
        assert_eq!(loaded.q_dim, lora.q_dim);
        assert_eq!(loaded.v_dim, lora.v_dim);
        assert_eq!(loaded.alpha, lora.alpha);
        assert_eq!(loaded.a_q, lora.a_q);
        assert_eq!(loaded.b_q, lora.b_q);
        assert_eq!(loaded.a_v, lora.a_v);
        assert_eq!(loaded.b_v, lora.b_v);
        let _ = std::fs::remove_file(&path);
    }

}
