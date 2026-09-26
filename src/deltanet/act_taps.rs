//! The Issue-014 activation tap plan, shared by the collector
//! (`act_diagonal_calibration`, T1) and the refit/walk instrument
//! (`act_retention_walk`, T2/T3).
//!
//! ONE mapping from the qwen35 hybrid ternary weights to (a) the distinct
//! linear-INPUT tensors the forward feeds its `bitlinear` matvecs and (b) the
//! per-call visit order. Both instruments must agree on it or the diagonal
//! slices mis-attribute — so the plan lives here, in the lib, and the
//! `TernaryMatvecHook` consumer asserts it every token (shape + call-count
//! asserts below are the tripwire).
//!
//! Per token (deterministic bitlinear call order):
//! - DeltaNet layer: `in_proj_qkv` → *attn_in*, `in_proj_z` → *attn_in*,
//!   `out_proj` → *layer_out*; then `gate_proj`/`up_proj` → *ffn_in*,
//!   `down_proj` → *swiglu*.
//! - Attention layer: `attn_wq`/`attn_wk`/`attn_wv` → *attn_in*,
//!   `attn_wo` → *layer_out*; then the same FFN trio.
//! - Final: `lm_head` → *final_in*.
//!
//! `attn_in` and `ffn_in` are DIFFERENT tensors (the FFN input is the
//! re-normed post-attention residual), hence two taps. Duplicated inputs
//! (qkv = z, wq = wk = wv, gate = up) fold into one tap per distinct tensor.
//! `in_proj_a`/`in_proj_b` carry no hook seam (`matvec_into` bypasses
//! `bitlinear`) and are the Issue-980 DENSE escape set on Bonsai-2 — never
//! ternary, out of refit scope by construction. `wte` is a row lookup, not a
//! matvec — no tap, out of refit scope.

use katgpt_core::act_channel_moments::{ActChannelDiagonal, ActChannelMoments};
use katgpt_core::{TernaryGroupWeights, TernaryMatvecHook, simd_ternary_group_matvec_parallel};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::ternary_weights::QwenDeltaNetTernaryWeights;
use crate::types::{Config, DeltaNetLayerType};

/// One refit-able ternary matvec site: the projection role + owning layer.
///
/// The refit iterates ALL of these (every tensor the forward matvecs); the
/// tap plan maps each to the distinct input tensor whose diagonal weights it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernarySite {
    InProjQkv(usize),
    InProjZ(usize),
    OutProj(usize),
    AttnWq(usize),
    AttnWk(usize),
    AttnWv(usize),
    AttnWo(usize),
    GateProj(usize),
    UpProj(usize),
    DownProj(usize),
    LmHead,
}

impl TernarySite {
    /// The owning layer (`Some(l)`; `None` for the head).
    pub fn layer(&self) -> Option<usize> {
        match self {
            TernarySite::LmHead => None,
            TernarySite::InProjQkv(l)
            | TernarySite::InProjZ(l)
            | TernarySite::OutProj(l)
            | TernarySite::AttnWq(l)
            | TernarySite::AttnWk(l)
            | TernarySite::AttnWv(l)
            | TernarySite::AttnWo(l)
            | TernarySite::GateProj(l)
            | TernarySite::UpProj(l)
            | TernarySite::DownProj(l) => Some(*l),
        }
    }

    /// Human projection name (`in_proj_qkv`, `attn_wo`, `lm_head`, ...).
    pub fn proj(&self) -> &'static str {
        match self {
            TernarySite::InProjQkv(_) => "in_proj_qkv",
            TernarySite::InProjZ(_) => "in_proj_z",
            TernarySite::OutProj(_) => "out_proj",
            TernarySite::AttnWq(_) => "attn_wq",
            TernarySite::AttnWk(_) => "attn_wk",
            TernarySite::AttnWv(_) => "attn_wv",
            TernarySite::AttnWo(_) => "attn_wo",
            TernarySite::GateProj(_) => "gate_proj",
            TernarySite::UpProj(_) => "up_proj",
            TernarySite::DownProj(_) => "down_proj",
            TernarySite::LmHead => "lm_head",
        }
    }

    /// The distinct input-tensor kind this site reads (`attn_in`, ...).
    pub fn tap_kind(&self) -> &'static str {
        match self {
            TernarySite::InProjQkv(_)
            | TernarySite::InProjZ(_)
            | TernarySite::AttnWq(_)
            | TernarySite::AttnWk(_)
            | TernarySite::AttnWv(_) => "attn_in",
            TernarySite::OutProj(_) | TernarySite::AttnWo(_) => "layer_out",
            TernarySite::GateProj(_) | TernarySite::UpProj(_) => "ffn_in",
            TernarySite::DownProj(_) => "swiglu",
            TernarySite::LmHead => "final_in",
        }
    }
}

/// A distinct linear-input tensor (one accumulator entry).
pub struct ActTap {
    /// Human label including the owning layer (`l03.attn_in`).
    pub label: String,
    /// Aggregate kind for the dashboard (`attn_in` / `layer_out` / ...).
    pub kind: &'static str,
    pub width: usize,
}

/// One expected matvec call site: which tap observes it, the (rows, cols) the
/// weights must carry (asserted every token), and WHICH tensor it is (the T2
/// refit mapping).
pub struct ActTapStep {
    pub site: TernarySite,
    pub tap: usize,
    pub rows: usize,
    pub cols: usize,
}

/// The walk of the weights in the forward's exact `bitlinear` call order,
/// deduplicated into one tap per distinct input tensor. Built once from the
/// weights; the hook asserts it per token; the refit consumes the same
/// `site → tap` mapping so the diagonal can never mis-attribute.
pub struct ActTapPlan {
    pub taps: Vec<ActTap>,
    pub steps: Vec<ActTapStep>,
    pub moment_widths: Vec<usize>,
}

impl ActTapPlan {
    /// The accumulator footprint: Σ|x|, Σx² as f64 + one u64 count per tap.
    pub fn moment_bytes(&self) -> usize {
        let total: usize = self.moment_widths.iter().sum();
        total * 16 + self.moment_widths.len() * 8
    }

    /// Walk the weights in the forward's exact `bitlinear` call order and
    /// deduplicate identical inputs into one tap per distinct tensor.
    pub fn build(config: &Config, weights: &QwenDeltaNetTernaryWeights) -> Self {
        let n = config.n_embd;
        let mut taps: Vec<ActTap> = Vec::new();
        let mut steps: Vec<ActTapStep> = Vec::new();
        let mut moment_widths: Vec<usize> = Vec::new();
        let push_tap = |taps: &mut Vec<ActTap>,
                            widths: &mut Vec<usize>,
                            label: String,
                            kind: &'static str,
                            width: usize| {
            taps.push(ActTap {
                label,
                kind,
                width,
            });
            widths.push(width);
            taps.len() - 1
        };
        for (l, layer) in weights.layers.iter().enumerate() {
            let gdn = weights.layer_types[l] == DeltaNetLayerType::DeltaNet;
            let attn_in = push_tap(
                &mut taps,
                &mut moment_widths,
                format!("l{l:02}.attn_in"),
                "attn_in",
                n,
            );
            let out_w = if gdn {
                &layer.out_proj
            } else {
                &layer.attn_wo
            };
            let layer_out = push_tap(
                &mut taps,
                &mut moment_widths,
                format!("l{l:02}.layer_out"),
                "layer_out",
                out_w.cols,
            );
            let ffn_in = push_tap(
                &mut taps,
                &mut moment_widths,
                format!("l{l:02}.ffn_in"),
                "ffn_in",
                n,
            );
            let swiglu = push_tap(
                &mut taps,
                &mut moment_widths,
                format!("l{l:02}.swiglu"),
                "swiglu",
                layer.down_proj.cols,
            );
            if gdn {
                for (site, w) in [
                    (TernarySite::InProjQkv(l), &layer.in_proj_qkv),
                    (TernarySite::InProjZ(l), &layer.in_proj_z),
                ] {
                    steps.push(ActTapStep {
                        site,
                        tap: attn_in,
                        rows: w.rows,
                        cols: w.cols,
                    });
                }
            } else {
                for (site, w) in [
                    (TernarySite::AttnWq(l), &layer.attn_wq),
                    (TernarySite::AttnWk(l), &layer.attn_wk),
                    (TernarySite::AttnWv(l), &layer.attn_wv),
                ] {
                    steps.push(ActTapStep {
                        site,
                        tap: attn_in,
                        rows: w.rows,
                        cols: w.cols,
                    });
                }
            }
            steps.push(ActTapStep {
                site: if gdn {
                    TernarySite::OutProj(l)
                } else {
                    TernarySite::AttnWo(l)
                },
                tap: layer_out,
                rows: out_w.rows,
                cols: out_w.cols,
            });
            // FFN trio (both layer types, identical order in the forward).
            steps.push(ActTapStep {
                site: TernarySite::GateProj(l),
                tap: ffn_in,
                rows: layer.gate_proj.rows,
                cols: layer.gate_proj.cols,
            });
            steps.push(ActTapStep {
                site: TernarySite::UpProj(l),
                tap: ffn_in,
                rows: layer.up_proj.rows,
                cols: layer.up_proj.cols,
            });
            steps.push(ActTapStep {
                site: TernarySite::DownProj(l),
                tap: swiglu,
                rows: layer.down_proj.rows,
                cols: layer.down_proj.cols,
            });
        }
        let head = push_tap(
            &mut taps,
            &mut moment_widths,
            "final_in".to_string(),
            "final_in",
            weights.lm_head.cols,
        );
        steps.push(ActTapStep {
            site: TernarySite::LmHead,
            tap: head,
            rows: weights.lm_head.rows,
            cols: weights.lm_head.cols,
        });
        Self {
            taps,
            steps,
            moment_widths,
        }
    }

    /// The `E[x²]` diagonal slice a site's refit consumes (its tap's
    /// `mean_sq` — Bench 896 found `E[x²]` the better weight in every
    /// non-control row; re-checking on real activations is T3's job, not the
    /// mapping's).
    pub fn diag_for<'a>(&self, diag: &'a ActChannelDiagonal, site: TernarySite) -> &'a [f32] {
        let step = self
            .steps
            .iter()
            .find(|s| s.site == site)
            .unwrap_or_else(|| panic!("act_taps: site {site:?} absent from the tap plan"));
        diag.mean_sq(step.tap)
    }
}

/// Visit every refit-able ternary matvec tensor, in `bitlinear` call order —
/// the SAME order [`ActTapPlan::build`] walks, so `steps[i].site` is the i-th
/// visited tensor (a shape assert on both sides pins the agreement).
pub fn for_each_ternary_site_mut(
    weights: &mut QwenDeltaNetTernaryWeights,
    mut f: impl FnMut(TernarySite, &mut TernaryGroupWeights),
) {
    for l in 0..weights.layers.len() {
        let gdn = weights.layer_types[l] == DeltaNetLayerType::DeltaNet;
        {
            let layer = &mut weights.layers[l];
            if gdn {
                f(TernarySite::InProjQkv(l), &mut layer.in_proj_qkv);
                f(TernarySite::InProjZ(l), &mut layer.in_proj_z);
            } else {
                f(TernarySite::AttnWq(l), &mut layer.attn_wq);
                f(TernarySite::AttnWk(l), &mut layer.attn_wk);
                f(TernarySite::AttnWv(l), &mut layer.attn_wv);
            }
        }
        {
            let layer = &mut weights.layers[l];
            f(
                if gdn {
                    TernarySite::OutProj(l)
                } else {
                    TernarySite::AttnWo(l)
                },
                if gdn {
                    &mut layer.out_proj
                } else {
                    &mut layer.attn_wo
                },
            );
        }
        let layer = &mut weights.layers[l];
        f(TernarySite::GateProj(l), &mut layer.gate_proj);
        f(TernarySite::UpProj(l), &mut layer.up_proj);
        f(TernarySite::DownProj(l), &mut layer.down_proj);
    }
    f(TernarySite::LmHead, &mut weights.lm_head);
}

/// The immutable twin of [`for_each_ternary_site_mut`] (same order).
pub fn for_each_ternary_site(
    weights: &QwenDeltaNetTernaryWeights,
    mut f: impl FnMut(TernarySite, &TernaryGroupWeights),
) {
    for l in 0..weights.layers.len() {
        let gdn = weights.layer_types[l] == DeltaNetLayerType::DeltaNet;
        let layer = &weights.layers[l];
        if gdn {
            f(TernarySite::InProjQkv(l), &layer.in_proj_qkv);
            f(TernarySite::InProjZ(l), &layer.in_proj_z);
        } else {
            f(TernarySite::AttnWq(l), &layer.attn_wq);
            f(TernarySite::AttnWk(l), &layer.attn_wk);
            f(TernarySite::AttnWv(l), &layer.attn_wv);
        }
        f(
            if gdn {
                TernarySite::OutProj(l)
            } else {
                TernarySite::AttnWo(l)
            },
            if gdn {
                &layer.out_proj
            } else {
                &layer.attn_wo
            },
        );
        f(TernarySite::GateProj(l), &layer.gate_proj);
        f(TernarySite::UpProj(l), &layer.up_proj);
        f(TernarySite::DownProj(l), &layer.down_proj);
    }
    f(TernarySite::LmHead, &weights.lm_head);
}

/// The collector's side-band observer: runs the REAL kernel (what the
/// unhooked path runs — the forward stays bit-identical) then feeds the
/// `ActChannelMoments` accumulator at the plan's tap. Call order is asserted
/// every token (begin/end bracket); a forward change trips it loudly.
pub struct ActTapHook<'a> {
    plan: &'a ActTapPlan,
    /// `Mutex` because `TernaryMatvecHook` hands out `&self`; the bitlinear
    /// call sites are sequential, so the lock is uncontended (~400/token).
    moments: Mutex<&'a mut ActChannelMoments>,
    call: AtomicUsize,
    steps_per_token: usize,
}

impl<'a> ActTapHook<'a> {
    pub fn new(plan: &'a ActTapPlan, moments: &'a mut ActChannelMoments) -> Self {
        Self {
            plan,
            moments: Mutex::new(moments),
            call: AtomicUsize::new(0),
            steps_per_token: plan.steps.len(),
        }
    }

    pub fn begin_token(&self) {
        self.call.store(0, Ordering::Relaxed);
    }

    pub fn end_token(&self) {
        let n = self.call.load(Ordering::Relaxed);
        assert_eq!(
            n, self.steps_per_token,
            "act_taps: the forward made {n} hook calls, the plan expected {} — \
             the call order changed under the collector",
            self.steps_per_token,
        );
    }
}

impl TernaryMatvecHook for ActTapHook<'_> {
    fn matvec(&self, w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
        // The REAL kernel, exactly what the unhooked path runs — the forward
        // stays bit-identical; the observation below is side-band.
        simd_ternary_group_matvec_parallel(w, &x[..w.cols], &mut y[..w.rows]);
        let i = self.call.fetch_add(1, Ordering::Relaxed);
        let step = self
            .plan
            .steps
            .get(i)
            .unwrap_or_else(|| panic!("act_taps: hook call {i} past the plan"));
        assert_eq!(
            (w.rows, w.cols),
            (step.rows, step.cols),
            "act_taps: matvec for {} at call {i} has shape {}x{}, the plan \
             expected {}x{} — tap attribution would be wrong",
            self.plan.taps[step.tap].label,
            w.rows,
            w.cols,
            step.rows,
            step.cols,
        );
        assert_eq!(
            x.len(),
            self.plan.taps[step.tap].width,
            "act_taps: observed input width {} != tap width {}",
            x.len(),
            self.plan.taps[step.tap].width
        );
        let mut m = self
            .moments
            .lock()
            .expect("act_taps: moments lock poisoned");
        m.observe(step.tap, &x[..step.cols]);
    }
}
