//! `attention_to_answer` — the m_Y probe (katgpt-rs Issue 882 P4).
//!
//! `m_Y = (1/R) Σ_r Σ_{i∈Y} p_{r,i}`: the attention mass a query row puts on
//! a labeled answer span `Y`, averaged over the probed rows `R` — the
//! Research 586 / paper Table-3 instrument, on our models.
//!
//! The probe is OFF the hot path: the attention kernels are untouched. When
//! [`AttnSpanProbe::armed`] is set, the gemma-2 f16 forward calls
//! [`AttnSpanProbe::observe_layer`] once per layer after the K cache write, which re-scores
//! `Q·K` for every head over the current row (the same softcap), turns it
//! into the distribution the forward USES — the plain softmax, or the
//! floored + coded one when `ForwardContext::logit_floor` is set — and adds
//! the span's mass into a `[layer × head]` table. Cost: one extra QK pass
//! per probed position, nothing on unprobed ones.

use std::ops::Range;

/// The floor policy a probe can reproduce: the real policy under
/// `row_logit_floor`, an uninhabited type without it (so `None` is the only
/// value and the plain softmax the only path).
#[cfg(feature = "row_logit_floor")]
pub type ProbeFloor = super::attention_floor::RowLogitFloorPolicy;
/// See the `row_logit_floor` definition.
#[cfg(not(feature = "row_logit_floor"))]
#[derive(Debug, Clone, Copy)]
pub enum ProbeFloor {}

/// The m_Y accumulator (one per forward context).
#[derive(Debug, Clone, PartialEq)]
pub struct AttnSpanProbe {
    /// Key positions of the answer span `Y`.
    pub span: Range<usize>,
    /// Probe the NEXT forward call(s) while `true` (the caller arms it on
    /// the query rows it wants, e.g. question + answer positions).
    pub armed: bool,
    /// `Σ_r Σ_{i∈Y} p_{r,i}` per `(layer, head)`, row-major `[n_layer][n_head]`.
    pub mass: Vec<f64>,
    /// Probed rows per `(layer, head)` cell (a row is counted only once its
    /// context covers the span start).
    pub rows: u64,
    n_head: usize,
    /// One score row of scratch (`block_size`), owned so the forward can
    /// borrow the probe beside its own buffers.
    scratch: Vec<f32>,
}

impl AttnSpanProbe {
    /// An unarmed probe sized for `n_layer × n_head` rows of up to
    /// `block_size` keys.
    pub fn new(n_layer: usize, n_head: usize, block_size: usize) -> Self {
        Self {
            span: 0..0,
            armed: false,
            mass: vec![0.0; n_layer * n_head],
            rows: 0,
            n_head,
            scratch: vec![0.0; block_size],
        }
    }

    /// Clear the table (keeps the sizing).
    pub fn reset(&mut self) {
        self.mass.fill(0.0);
        self.rows = 0;
        self.armed = false;
    }

    /// Mean mass per layer (over heads and rows).
    pub fn layer_means(&self) -> Vec<f64> {
        let r = self.rows.max(1) as f64 * self.n_head as f64;
        self.mass
            .chunks(self.n_head)
            .map(|h| h.iter().sum::<f64>() / r)
            .collect()
    }

    /// `(layer, head, mean mass)` of the most answer-attending head.
    pub fn top_head(&self) -> (usize, usize, f64) {
        let r = self.rows.max(1) as f64;
        self.mass
            .iter()
            .enumerate()
            .fold((0, 0, f64::NEG_INFINITY), |b, (i, &m)| match m / r > b.2 {
                true => (i / self.n_head, i % self.n_head, m / r),
                false => b,
            })
    }

    /// Mean over every `(layer, head)` cell — the scalar `m_Y`.
    pub fn m_y(&self) -> f64 {
        let cells = self.mass.len().max(1) as f64;
        self.mass.iter().sum::<f64>() / (self.rows.max(1) as f64 * cells)
    }
}

/// The per-layer shape [`AttnSpanProbe::observe_layer`] needs.
#[derive(Debug, Clone, Copy)]
pub struct ProbeShape {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub t_n: usize,
    pub scale: f32,
    pub softcap: f32,
}

impl AttnSpanProbe {
    /// Add one layer's span mass for the current query row. The CALLER counts
    /// `rows` (once per probed position) and should arm only rows whose context
    /// covers the span; a row ending before the span start adds nothing here.
    ///
    /// # Panics
    /// If `t_n` exceeds the `block_size` the probe was built with.
    pub fn observe_layer(
        &mut self,
        layer: usize,
        q: &[f32],
        key_cache: &[f32],
        shape: ProbeShape,
        floor: Option<ProbeFloor>,
    ) {
        let ProbeShape {
            n_head,
            n_kv_head,
            kv_dim,
            head_dim,
            t_n,
            scale,
            softcap,
        } = shape;
        let lo = self.span.start.min(t_n);
        let hi = self.span.end.min(t_n);
        if lo >= hi {
            return;
        }
        let row = &mut self.scratch[..t_n];
        for h in 0..n_head {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let g = (h * n_kv_head / n_head) * head_dim;
            for (t, s) in row.iter_mut().enumerate() {
                let k_off = t * kv_dim + g;
                let dot =
                    crate::simd::simd_dot_f32(qh, &key_cache[k_off..k_off + head_dim], head_dim);
                *s = match softcap > 0.0 {
                    true => softcap * crate::simd::fast_tanh(dot * scale / softcap),
                    false => dot * scale,
                };
            }
            let z = exp_row(row, floor);
            let span_mass: f32 = row[lo..hi].iter().sum();
            self.mass[layer * n_head + h] += f64::from(span_mass / z);
        }
    }
}

/// Turn a score row into unnormalised softmax numerators in place and
/// return their sum — the plain softmax, or the floored + coded one.
fn exp_row(row: &mut [f32], floor: Option<ProbeFloor>) -> f32 {
    #[cfg(not(feature = "row_logit_floor"))]
    if let Some(never) = floor {
        match never {}
    }
    #[cfg(feature = "row_logit_floor")]
    if let Some(p) = floor.filter(|p| row.len() > p.n_sink) {
        use katgpt_core::row_logit_floor::{
            LogitCodec, floored_coded_exp_inplace, min_width_for_tv,
        };
        let width = min_width_for_tv(p.width_ctx.unwrap_or(row.len() - p.n_sink), p.tv);
        let mut lut = [0.0f32; 256];
        return floored_coded_exp_inplace(row, p.n_sink, width, &LogitCodec::new(p.bits), &mut lut)
            .1;
    }
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for s in row.iter_mut() {
        *s = (*s - m).exp();
        z += *s;
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A query identical to one key and orthogonal to the rest puts its
    /// mass on that key: the probe must read it off the span.
    #[test]
    fn span_mass_tracks_the_matching_key() {
        let (hd, t_n) = (4usize, 6usize);
        let mut k = vec![0.0f32; t_n * hd];
        for t in 0..t_n {
            k[t * hd + (t % hd)] = 1.0;
        }
        k[3 * hd..4 * hd].copy_from_slice(&[0.0, 0.0, 0.0, 9.0]);
        let q = [0.0f32, 0.0, 0.0, 9.0];
        let shape = ProbeShape {
            n_head: 1,
            n_kv_head: 1,
            kv_dim: hd,
            head_dim: hd,
            t_n,
            scale: 1.0,
            softcap: 0.0,
        };
        let mut probe = AttnSpanProbe::new(1, 1, t_n);
        probe.span = 3..4;
        probe.rows = 1;
        probe.observe_layer(0, &q, &k, shape, None);
        assert!(probe.m_y() > 0.99, "m_y {}", probe.m_y());
        // A span beyond the row contributes nothing.
        let mut p2 = AttnSpanProbe::new(1, 1, t_n);
        p2.span = 10..12;
        p2.observe_layer(0, &q, &k, shape, None);
        assert_eq!(p2.mass, vec![0.0]);
    }

    /// The span masses over a partition of the row sum to one per head.
    #[test]
    fn partition_masses_sum_to_one() {
        let (hd, t_n, nh) = (8usize, 40usize, 2usize);
        let lcg = |i: usize| (((i * 2_654_435_761) % 1_000_003) as f32 / 1_000_003.0) * 2.0 - 1.0;
        let k: Vec<f32> = (0..t_n * hd).map(|i| 2.0 * lcg(i + 3)).collect();
        let q: Vec<f32> = (0..nh * hd).map(|i| 2.0 * lcg(i + 77)).collect();
        let shape = ProbeShape {
            n_head: nh,
            n_kv_head: 1,
            kv_dim: hd,
            head_dim: hd,
            t_n,
            scale: 0.5,
            softcap: 50.0,
        };
        let mut total = [0.0f64; 2];
        for span in [0..13, 13..29, 29..40] {
            let mut p = AttnSpanProbe::new(1, nh, t_n);
            p.span = span;
            p.observe_layer(0, &q, &k, shape, None);
            total[0] += p.mass[0];
            total[1] += p.mass[1];
        }
        for t in total {
            assert!((t - 1.0).abs() < 1e-5, "{t}");
        }
    }

    /// Same partition law on the FLOORED distribution the forward uses.
    #[cfg(feature = "row_logit_floor")]
    #[test]
    fn floored_partition_masses_sum_to_one() {
        let (hd, t_n, nh) = (8usize, 40usize, 2usize);
        let lcg = |i: usize| (((i * 2_654_435_761) % 1_000_003) as f32 / 1_000_003.0) * 2.0 - 1.0;
        let k: Vec<f32> = (0..t_n * hd).map(|i| 2.0 * lcg(i + 3)).collect();
        let q: Vec<f32> = (0..nh * hd).map(|i| 2.0 * lcg(i + 77)).collect();
        let shape = ProbeShape {
            n_head: nh,
            n_kv_head: 1,
            kv_dim: hd,
            head_dim: hd,
            t_n,
            scale: 0.5,
            softcap: 50.0,
        };
        let mut total = [0.0f64; 2];
        for span in [0..13, 13..29, 29..40] {
            let mut p = AttnSpanProbe::new(1, nh, t_n);
            p.span = span;
            p.observe_layer(
                0,
                &q,
                &k,
                shape,
                Some(ProbeFloor {
                    n_sink: 2,
                    bits: 6,
                    tv: 1e-3,
                    width_ctx: Some(65_536),
                }),
            );
            total[0] += p.mass[0];
            total[1] += p.mass[1];
        }
        for t in total {
            assert!((t - 1.0).abs() < 1e-5, "{t}");
        }
    }
}
