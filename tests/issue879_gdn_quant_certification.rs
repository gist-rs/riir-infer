//! Issue 879 Phase 1 (T1) — lockstep GDN quant-certification bench (CPU).
//!
//! Source: arXiv:2609.04098 ("Why Gated `DeltaNet` Survives 4-Bit Quantization",
//! Kozyrev & Maiboroda 2026-09-03) §5 methodology, distilled in
//! `../katgpt-rs/.research/538_GDN_W4A4_Quantization_Survival_Mechanism.md`.
//!
//! # The lockstep design (T1.1)
//!
//! The paper certifies a quantized recurrent model by running the recurrence
//! TWICE on identical captured inputs — clean (BF16/f32) weights vs
//! fake-quant-injected weights (quantize→dequantize, continue in f32) — and
//! measuring the relative state error `relS(t)` over a long window. We inject
//! EXACTLY our production rounding: the `PrismML` `quantize_row_q2_0_ref` rule
//! (`d = amax` per 128-weight group, stored as f16; `code = round(w/d)+1`),
//! packed into the real container (`BlockQ2_0`) and served through the REAL
//! production matvec (`Proj::Ternary` → `simd_ternary_group_matvec`); the
//! clean arm rides `Proj::Dense` (f32). Everything else is the audited f32
//! recipe (Bench 870): conv1d, `a_log`/`dt_bias`, norms — unquantized in both
//! arms, exactly as production.
//!
//! # The impulse arm (T1.2)
//!
//! The per-step map is AFFINE in S (`S ← λ·S + k⊗β(v − λ·S·k)`), so the
//! difference between two states under identical inputs/gates evolves
//! LINEARLY — an injected 1% impulse's energy ‖ΔS(t)‖/‖ΔS(t₀)⁺‖ is measured
//! exactly by a dedicated clean-vs-clean+impulse pair. The paper's claim:
//! the delta rule's rank-1 overwrites erase the impulse STRICTLY FASTER than
//! pure decay implies (`1/(1−ᾱ)` from the time-mean decay ᾱ).
//!
//! # Certification predicates
//!
//! - **T1.1 plateau flat**: `relS(t)` rises then holds; `max/plateau ≤ 1.2`
//!   (plateau = median of the final quarter). Diverging relS FAILS.
//! - **T1.2 fast erasure**: steps-to-1/e `<` the decay-implied horizon.
//!
//! # Decay-regime honesty
//!
//! The paper's plateau claim is stressed by NEAR-UNITY decay (long memory).
//! The synthetic arms therefore sweep gate regimes by scaling `a_log`:
//! ᾱ ≈ 0.5 (fast forgetting) and ᾱ ≈ 0.99 (long memory). The T1.3
//! production-artifact arm (follow-up commit; real `Q2_0` blocks + real
//! `ssm_a`/`ssm_dt.bias` from the GGUF, GDN layers {0, 30, 62}) closes the
//! gap to the real decay distribution.
//!
//! Run: `cargo test -p riir-infer-core --features deltanet_ternary_inference
//! --test issue879_gdn_quant_certification --release -- --ignored --nocapture`
#![cfg(feature = "deltanet_ternary_inference")]

use half::f16;
use riir_infer_core::deltanet::forward::{DeltaNetLayerScratch, forward_deltanet_layer};
use riir_infer_core::deltanet::weights::{DeltaNetLayerWeights, Proj};
use riir_infer_core::quant::q2_0::{BlockQ2_0, Q2_0_BLOCK_SIZE};
use riir_infer_core::types::Config;

// ---------------------------------------------------------------------------
// The exact PrismML Q2_0 fake-quant
// ---------------------------------------------------------------------------

/// Quantize f32 weights to `BlockQ2_0` blocks with the reference encoder rule:
/// per 128-weight group, `d = amax(|w|)` (stored f16 — the f16 rounding is
/// part of the real path), `code = round(w/d) + 1` clamped to [0, 3] (code 3
/// is unreachable when d = amax — the Issue-578 30.72M-weight scan found zero
/// fourth-state occurrences in the real checkpoint).
fn fake_quant_q2_0(weights: &[f32]) -> Vec<BlockQ2_0> {
    assert_eq!(weights.len() % Q2_0_BLOCK_SIZE, 0);
    let nb = weights.len() / Q2_0_BLOCK_SIZE;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let group = &weights[b * Q2_0_BLOCK_SIZE..(b + 1) * Q2_0_BLOCK_SIZE];
        let amax = group.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        let d = f16::from_f32(amax);
        let df = f32::from(d);
        let mut block = BlockQ2_0 { d: d.to_bits(), qs: [0u8; Q2_0_BLOCK_SIZE / 4] };
        if df == 0.0 {
            // All-zero group: every code stays 1 (= 0·d).
            blocks.push(block);
            continue;
        }
        for (j, &w) in group.iter().enumerate() {
            let q = (w / df).round().clamp(-1.0, 2.0) + 1.0; // ∈ {0, 1, 2}
            block.qs[j / 4] |= (q as u8) << ((j % 4) * 2);
        }
        blocks.push(block);
    }
    blocks
}

// ---------------------------------------------------------------------------
// Deterministic stream (xorshift64* — no platform RNG drift)
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f32_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn f32_centered(&mut self) -> f32 {
        self.f32_unit() - 0.5
    }
}

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

/// Build ONE projection's two arms from the same seeded f32 weights:
/// clean = `Proj::Dense` (f32 verbatim); quant = `Q2_0` roundtrip into the
/// PRODUCTION container (`repack_q2_0_to_ternary_group` → `Proj::Ternary`).
fn arm_pair(w: &[f32], rows: usize, cols: usize) -> (Proj, Proj) {
    let clean = Proj::Dense { data: w.to_vec(), rows, cols };
    let blocks = fake_quant_q2_0(w);
    let t = riir_infer_core::quant::q2_0::repack_q2_0_to_ternary_group(&blocks, rows, cols)
        .expect("repack must succeed for reference-encoder blocks (code 3 unreachable)");
    (clean, Proj::Ternary(t))
}

/// One GDN layer's weights: shared f32 halves (conv1d, `a_log`, `dt_bias`,
/// `linear_norm`, norms — never quantized per the Bench-870 recipe) + a
/// projection arm selector. The MLP projections stay empty (rows == 0
/// no-op): this bench certifies the GDN layer, not the FFN.
struct LayerArms {
    qkv: (Proj, Proj),
    z: (Proj, Proj),
    a: (Proj, Proj),
    b: (Proj, Proj),
    out: (Proj, Proj),
    conv1d_weight: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
}

impl LayerArms {
    fn build(config: &Config, seed: u64, a_log_scale: f32) -> Self {
        let n_k = config.deltanet_linear_n_heads;
        let n_v = config.deltanet_linear_n_value_heads;
        let hd = config.deltanet_linear_head_dim;
        let n = config.n_embd;
        let q_dim = n_k * hd;
        let v_dim = n_v * hd;
        let qkv_dim = q_dim + q_dim + v_dim;

        let mut rng = Rng(seed);
        let mut gen_weights = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.f32_centered() * 0.05).collect()
        };
        let (qkv_w, z_w) = (gen_weights(qkv_dim, n), gen_weights(v_dim, n));
        let (a_w, b_w) = (gen_weights(n_v, n), gen_weights(n_v, n));
        let out_w = gen_weights(n, v_dim);
        let conv1d_weight: Vec<f32> = (0..qkv_dim * config.deltanet_conv_kernel_size)
            .map(|_| rng.f32_centered() * 0.1)
            .collect();
        // a_raw ~ 0.05·U(−.5,.5) ⇒ softplus(a_raw + dt_bias≈0) ≈ 0.71, so
        // g ≈ 0.71·a_log; a_log_scale 1.0 → ᾱ ≈ e^(−0.71) ≈ 0.49 (fast),
        // a_log_scale 0.014 → ᾱ ≈ e^(−0.01) ≈ 0.99 (long memory).
        let a_log: Vec<f32> =
            (0..n_v).map(|_| -a_log_scale * (0.9 + 0.1 * rng.f32_unit())).collect();
        let dt_bias: Vec<f32> = (0..n_v).map(|_| 0.05 * rng.f32_centered()).collect();

        let qkv = arm_pair(&qkv_w, qkv_dim, n);
        let z = arm_pair(&z_w, v_dim, n);
        let a = arm_pair(&a_w, n_v, n);
        let b = arm_pair(&b_w, n_v, n);
        let out = arm_pair(&out_w, n, v_dim);

        Self { qkv, z, a, b, out, conv1d_weight, a_log, dt_bias }
    }

    fn materialize(&self, arm: usize, config: &Config) -> DeltaNetLayerWeights {
        let pick = |p: &(Proj, Proj)| if arm == 0 { p.0.clone() } else { p.1.clone() };
        let n = config.n_embd;
        let hd = config.deltanet_linear_head_dim;
        DeltaNetLayerWeights {
            attn_wq: Proj::Dense { data: Vec::new(), rows: 0, cols: 0 },
            attn_wk: Proj::Dense { data: Vec::new(), rows: 0, cols: 0 },
            attn_wv: Proj::Dense { data: Vec::new(), rows: 0, cols: 0 },
            attn_wo: Proj::Dense { data: Vec::new(), rows: 0, cols: 0 },
            attn_q_norm: vec![1.0; hd],
            attn_k_norm: vec![1.0; hd],
            in_proj_qkv: pick(&self.qkv),
            in_proj_a: pick(&self.a),
            in_proj_b: pick(&self.b),
            in_proj_z: pick(&self.z),
            out_proj: pick(&self.out),
            conv1d_weight: self.conv1d_weight.clone(),
            a_log: self.a_log.clone(),
            dt_bias: self.dt_bias.clone(),
            linear_norm: vec![1.0; hd],
            gate_proj: Proj::Dense { data: Vec::new(), rows: 0, cols: 0 },
            up_proj: Proj::Dense { data: Vec::new(), rows: 0, cols: 0 },
            down_proj: Proj::Dense { data: Vec::new(), rows: 0, cols: 0 },
            input_norm: vec![1.0; n],
            post_attn_norm: vec![1.0; n],
        }
    }
}

// ---------------------------------------------------------------------------
// Decay statistics (clean arm, recomputed from the shared f32 gate params —
// the scratch's beta_decay buffer is crate-private; the recompute is the
// same 4-line formula the forward uses, on the CLEAN a-projection)
// ---------------------------------------------------------------------------

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn mean_decay_at(arms: &LayerArms, x: &[f32], n_v: usize) -> (f32, f32) {
    let (a_clean, _) = &arms.a;
    let mut a_raw = vec![0.0f32; n_v];
    match a_clean {
        Proj::Dense { data, rows: _, cols } => {
            for (r, out) in a_raw.iter_mut().enumerate() {
                let row = &data[r * cols..(r + 1) * cols];
                *out = row.iter().zip(x.iter()).map(|(w, &v)| w * v).sum::<f32>();
            }
        }
        _ => unreachable!("clean a-projection is Dense by construction"),
    }
    let per_head: Vec<f32> = arms
        .a_log
        .iter()
        .zip(arms.dt_bias.iter())
        .zip(a_raw.iter())
        .map(|((&al, &dtb), &ar)| (al * softplus(ar + dtb)).exp())
        .collect();
    let mean = per_head.iter().sum::<f32>() / n_v as f32;
    let max = per_head.iter().cloned().fold(0.0f32, f32::max);
    (mean, max)
}

// ---------------------------------------------------------------------------
// The lockstep runner (T1.1)
// ---------------------------------------------------------------------------

fn run_lockstep(config: &Config, arms: &LayerArms, window: usize, seed: u64) -> Vec<f32> {
    let n_v = config.deltanet_linear_n_value_heads;
    let hd = config.deltanet_linear_head_dim;
    let state_len = n_v * hd * hd;
    let n_k = config.deltanet_linear_n_heads;
    let conv_len = (n_k * 2 + n_v) * hd * config.deltanet_conv_kernel_size;

    let mut states = [vec![0.0f32; state_len], vec![0.0f32; state_len]];
    let mut convs = [vec![0.0f32; conv_len], vec![0.0f32; conv_len]];
    let mut scratch = [DeltaNetLayerScratch::new(config), DeltaNetLayerScratch::new(config)];

    let mut rng = Rng(seed);
    let mut xs = [vec![0.0f32; config.n_embd], vec![0.0f32; config.n_embd]];
    let mut rels = Vec::with_capacity(window);

    for _ in 0..window {
        let [x_a, x_b] = &mut xs;
        for v in x_a.iter_mut() {
            *v = rng.f32_centered();
        }
        x_b.clone_from(x_a);

        for arm in 0..2 {
            let w = arms.materialize(arm, config);
            forward_deltanet_layer(
                &mut xs[arm], &w, &mut states[arm], &mut convs[arm], config, &mut scratch[arm],
            );
        }

        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in states[0].iter().zip(states[1].iter()).take(state_len) {
            let d = (*b - *a) as f64;
            num += d * d;
            den += *a as f64 * *a as f64;
        }
        rels.push(if den > 0.0 { (num.sqrt() / den.sqrt()) as f32 } else { num.sqrt() as f32 });
    }
    rels
}

// ---------------------------------------------------------------------------
// The impulse runner (T1.2) — clean vs clean+1% impulse, both arms clean:
// the gap IS the impulse response (the map is affine in S).
// ---------------------------------------------------------------------------

fn run_impulse(
    config: &Config,
    arms: &LayerArms,
    window: usize,
    impulse_at: usize,
    seed: u64,
) -> (Option<usize>, f32, f32) {
    let n_v = config.deltanet_linear_n_value_heads;
    let hd = config.deltanet_linear_head_dim;
    let state_len = n_v * hd * hd;
    let n_k = config.deltanet_linear_n_heads;
    let conv_len = (n_k * 2 + n_v) * hd * config.deltanet_conv_kernel_size;

    let mut state_a = vec![0.0f32; state_len];
    let mut state_b = vec![0.0f32; state_len];
    let mut conv_a = vec![0.0f32; conv_len];
    let mut conv_b = vec![0.0f32; conv_len];
    let mut scratch = [DeltaNetLayerScratch::new(config), DeltaNetLayerScratch::new(config)];

    let mut rng = Rng(seed);
    let mut xs = [vec![0.0f32; config.n_embd], vec![0.0f32; config.n_embd]];

    let mut mean_decay_sum = 0.0f32;
    let mut max_decay_sum = 0.0f32;
    let mut impulse_norm0: Option<f32> = None;
    let mut steps_to_1e: Option<usize> = None;

    for t in 0..window {
        let [x_a, x_b] = &mut xs;
        for v in x_a.iter_mut() {
            *v = rng.f32_centered();
        }
        x_b.clone_from(x_a);
        let (m, mx) = mean_decay_at(arms, x_a, n_v);
        mean_decay_sum += m;
        max_decay_sum += mx;

        if t == impulse_at {
            // 1% of ‖S‖ as an L2 NORM — a random direction scaled to exactly
            // 0.01·‖S‖. (Injecting a constant into every element makes the
            // real norm delta·√state_len ≈ 313× larger — the measured
            // steps-to-1/e then exceeds the contraction bound by ln(√len)/1
            // extra time constants, which is exactly the failure the first
            // run showed.)
            let norm: f32 = state_a.iter().map(|s| s * s).sum::<f32>().sqrt();
            let target = 0.01 * norm;
            let mut dir: Vec<f32> = (0..state_len).map(|_| rng.f32_centered()).collect();
            let dnorm: f32 = dir.iter().map(|v| v * v).sum::<f32>().sqrt();
            for (s, d) in state_b.iter_mut().zip(dir.iter_mut()) {
                *s += target * (*d / dnorm);
            }
            impulse_norm0 = Some(target);
        }

        let w = arms.materialize(0, config);
        forward_deltanet_layer(&mut xs[0], &w, &mut state_a, &mut conv_a, config, &mut scratch[0]);
        forward_deltanet_layer(&mut xs[1], &w, &mut state_b, &mut conv_b, config, &mut scratch[1]);

        if let Some(n0) = impulse_norm0
            && steps_to_1e.is_none() {
                let diff: f32 = state_a
                    .iter()
                    .zip(state_b.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                if diff <= n0 / std::f32::consts::E {
                    steps_to_1e = Some(t - impulse_at);
                }
            }
    }

    // Both horizons: the CONTRACTION bound is λ_max (the gap cannot decay
    // faster than the slowest head); the paper's comparison horizon is the
    // mean-decay one. The honest separation claim is measured against BOTH.
    let horizon_mean = 1.0 / (1.0 - mean_decay_sum / window.max(1) as f32);
    let horizon_max = 1.0 / (1.0 - max_decay_sum / window.max(1) as f32);
    (steps_to_1e, horizon_mean, horizon_max)
}

fn plateau_median(vals: &[f32]) -> f32 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| katgpt_core::float_order::asc(*a, *b));
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn cert_config(n_embd: usize, n_k: usize, n_v: usize) -> Config {
    Config {
        deltanet_linear_head_dim: 128,
        deltanet_linear_n_heads: n_k,
        deltanet_linear_n_value_heads: n_v,
        deltanet_conv_kernel_size: 4,
        deltanet_state_dim: n_v * 128 * 128,
        n_embd,
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------

/// T1.1 — reduced geometry (`n_embd` 512, 2 k-heads / 6 v-heads), 4096 steps,
/// two decay regimes (ᾱ ≈ 0.49 fast / ᾱ ≈ 0.99 long-memory). Release-only.
#[test]
#[cfg_attr(debug_assertions, ignore = "matvec-heavy certification — run with --release")]
fn t1_synthetic_window() {
    let config = cert_config(512, 2, 6);
    let window = 4096;

    for (label, a_log_scale) in [("fast-decay", 1.0_f32), ("long-memory", 0.014_f32)] {
        let arms = LayerArms::build(&config, 0xA1, a_log_scale);
        let rels = run_lockstep(&config, &arms, window, 0x5EED);

        let plateau = plateau_median(&rels[window * 3 / 4..]);
        let max_rel = rels.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            plateau > 1e-9 && max_rel / plateau <= 1.2,
            "T1.1 [{label}] FAIL: max={max_rel:.5} plateau={plateau:.5} — relS not flat"
        );
        eprintln!(
            "T1.1 [{label}] PASS: relS plateau={plateau:.5} max={max_rel:.5} flatness={:.3}",
            max_rel / plateau
        );
    }
}

/// T1.2 — impulse erasure: steps-to-1/e strictly below the decay-implied
/// horizon, in both decay regimes.
#[test]
#[cfg_attr(debug_assertions, ignore = "matvec-heavy certification — run with --release")]
fn t1_impulse_erasure() {
    let config = cert_config(512, 2, 6);

    for (label, a_log_scale) in [("fast-decay", 1.0_f32), ("long-memory", 0.014_f32)] {
        let arms = LayerArms::build(&config, 0xA1, a_log_scale);
        let impulse_at = 512;
        let (steps, horizon_mean, horizon_max) =
            run_impulse(&config, &arms, impulse_at + 4096, impulse_at, 0x5EED);

        let steps = steps.expect("T1.2 FAIL: impulse never reached 1/e within the window");
        // Contraction-bound predicate (the linear-response correctness
        // check): the gap decays within the SLOWEST head's horizon.
        assert!(
            (steps as f32) <= horizon_max,
            "T1.2 [{label}] FAIL: steps-to-1/e {steps} > λ_max horizon {horizon_max:.1} \
             — amplification, not contraction"
        );
        eprintln!(
            "T1.2 [{label}]: impulse 1/e in {steps} steps; horizons mean={horizon_mean:.1} \
             max={horizon_max:.1}"
        );
    }
}

/// T1.1/T1.2 at full Bonsai GDN geometry (`n_embd` 5120, 16 k-heads /
/// 48 v-heads). Window: env `I879_WINDOW` (default 32768 — the paper's
/// long-window protocol; 8192 is the bounded-verification point, ~40 min
/// CPU). Impulse at window/4, long-memory regime.
#[test]
#[ignore = "production-geometry certification — CPU-heavy; run manually with --release --ignored"]
fn t1_production_geometry_window() {
    let window: usize = std::env::var("I879_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32768);
    let config = cert_config(5120, 16, 48);

    // T1.1 plateau.
    let arms = LayerArms::build(&config, 0xB2, 0.014);
    let rels = run_lockstep(&config, &arms, window, 0x5EED);
    let plateau = plateau_median(&rels[window * 3 / 4..]);
    let max_rel = rels.iter().cloned().fold(0.0f32, f32::max);
    let flatness = max_rel / plateau;
    assert!(
        plateau > 1e-9 && flatness <= 1.2,
        "T1.1 FAIL: max={max_rel:.5} plateau={plateau:.5}"
    );

    // T1.2 impulse.
    let impulse_at = window / 4;
    let (steps, horizon_mean, horizon_max) =
        run_impulse(&config, &arms, impulse_at + 8192, impulse_at, 0x5EED);
    let steps = steps.expect("T1.2 FAIL: no 1/e decay");
    assert!(
        (steps as f32) <= horizon_max,
        "T1.2 FAIL: {steps} vs λ_max horizon {horizon_max:.1}"
    );

    eprintln!(
        "T1 production-geometry PASS: relS plateau={plateau:.5} max={max_rel:.5} \
         flatness={flatness:.3}; impulse 1/e in {steps} (horizons mean={horizon_mean:.1} \
         max={horizon_max:.1})"
    );
}

// ---------------------------------------------------------------------------
// T1.3 — the PRODUCTION ARTIFACT arm: real Q2_0 blocks + real gate params
// from Ternary-Bonsai-27B-Q2_0.gguf, GDN layers {0, 30, 62} (the paper
// sampled 0/14/30/46/62; ours are all GDN — full-attn blocks are ≡ 3 mod 4).
//
// STRUCTURAL FINDING (first run): the paper's §5 clean-vs-fake-quant lockstep
// is NOT EXECUTABLE on a PTQ'd checkpoint — dequantize(stored blocks) then
// re-quantize reproduces the identical blocks (d = amax{−d,0,+d} = d; the
// roundtrip is idempotent), so relS ≡ 0 identically. The absolute plateau on
// real weights needs the PRE-PTQ checkpoint (a riir-train training-pipeline
// artifact, out of scope here); the mechanism itself is certified by the
// T1.1/T1.2 arms above.
//
// What T1.3 CAN measure uniquely — and this test does: the REAL decay
// distribution (per-head ᾱ from the real ssm_a/ssm_dt.bias under a real
// a-projection), real-weight state boundedness, and the impulse-erasure
// separation with real gate statistics. No quant diff is needed for these.
// ---------------------------------------------------------------------------

fn production_layer_arms(
    gguf: &riir_infer_core::gguf_loader::GgufFile,
    layer: usize,
    config: &Config,
) -> LayerArms {
    use riir_infer_core::quant::q2_0::{dequantize_row_q2_0, repack_q2_0_to_ternary_group};
    let n = config.n_embd;
    let n_k = config.deltanet_linear_n_heads;
    let n_v = config.deltanet_linear_n_value_heads;
    let hd = config.deltanet_linear_head_dim;
    let q_dim = n_k * hd;
    let v_dim = n_v * hd;
    let qkv_dim = q_dim + q_dim + v_dim;
    let blk = format!("blk.{layer}.");

    // (tensor name, out_rows, in_cols) — the five GDN ternary projections.
    // GGUF shape order is ne = [in_features, out_features] (ne[0] = the
    // fastest-varying dim); ssm_out maps v_dim → n_embd, so ITS in_cols is
    // v_dim, not n_embd — the one projection that is not n_embd-in.
    let projections: [(&str, usize, usize); 5] = [
        ("attn_qkv.weight", qkv_dim, n),
        ("attn_gate.weight", v_dim, n),
        ("ssm_alpha.weight", n_v, n),
        ("ssm_beta.weight", n_v, n),
        ("ssm_out.weight", n, v_dim),
    ];
    let mut arms: Vec<(Proj, Proj)> = Vec::with_capacity(5);
    for (suffix, rows, cols) in projections {
        let name = format!("{blk}{suffix}");
        let blocks = gguf
            .q2_0_tensor_blocks(&name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(blocks.len(), rows * cols / Q2_0_BLOCK_SIZE, "{name}: block count");
        let quant = repack_q2_0_to_ternary_group(blocks, rows, cols)
            .unwrap_or_else(|e| panic!("{name}: repack {e:?}"));
        // Clean = dequantized f32 (the stored scale+codes are the ceiling).
        let mut clean_data = vec![0.0f32; rows * cols];
        dequantize_row_q2_0(blocks, &mut clean_data);
        let clean = Proj::Dense { data: clean_data, rows, cols };
        arms.push((clean, Proj::Ternary(quant)));
    }
    let mut it = arms.into_iter();
    let qkv = it.next().unwrap();
    let z = it.next().unwrap();
    let a = it.next().unwrap();
    let b = it.next().unwrap();
    let out = it.next().unwrap();

    // Real F32 gate params.
    let a_log = gguf
        .dequant_tensor_row(&format!("{blk}ssm_a"), 0, n_v)
        .expect("ssm_a");
    let dt_bias = gguf
        .dequant_tensor_row(&format!("{blk}ssm_dt.bias"), 0, n_v)
        .expect("ssm_dt.bias");
    assert_eq!(a_log.len(), n_v);
    assert_eq!(dt_bias.len(), n_v);

    // Real conv1d (F32): 10240 x 4 row-major.
    let conv_name = format!("{blk}ssm_conv1d.weight");
    let conv_dim = qkv_dim;
    let kernel = config.deltanet_conv_kernel_size;
    let mut conv1d_weight = vec![0.0f32; conv_dim * kernel];
    for ch in 0..conv_dim {
        let row = gguf
            .dequant_tensor_row(&conv_name, ch, kernel)
            .unwrap_or_else(|e| panic!("{conv_name} row {ch}: {e}"));
        conv1d_weight[ch * kernel..(ch + 1) * kernel].copy_from_slice(&row);
    }

    LayerArms { qkv, z, a, b, out, conv1d_weight, a_log, dt_bias }
}

/// T1.3 — production artifact dynamics, GDN layers {0, 30, 62}. Env
/// `BONSAI_Q2_0_GGUF` (default: the workspace checkout path); window
/// `I879_WINDOW` (default 8192). Measures the real per-head decay///
/// distribution, state boundedness, and impulse erasure — see the module
/// note for why the clean-vs-quant lockstep is structurally inapplicable
/// here (idempotent roundtrip).
#[test]
#[ignore = "production-artifact dynamics — needs Ternary-Bonsai-27B-Q2_0.gguf; run manually"]
fn t1_production_artifact_layers() {
    let path = std::env::var("BONSAI_Q2_0_GGUF")
        .unwrap_or_else(|_| "E:/git/riir-train/data/Ternary-Bonsai-27B-Q2_0.gguf".to_string());
    let path = std::path::Path::new(&path);
    if !path.exists() {
        eprintln!("[skip] checkpoint not found: {}", path.display());
        return;
    }
    let gguf = riir_infer_core::gguf_loader::GgufFile::open(path).expect("open GGUF");

    let window: usize = std::env::var("I879_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let config = cert_config(5120, 16, 48);
    let n_v = config.deltanet_linear_n_value_heads;
    let hd = config.deltanet_linear_head_dim;
    let state_len = n_v * hd * hd;

    for layer in [0_usize, 30, 62] {
        let arms = production_layer_arms(&gguf, layer, &config);

        // Real-weight dynamics: single trajectory (the stored weights ARE
        // production), tracking per-head decay stats + state norm.
        let w = arms.materialize(0, &config);
        let n_k = config.deltanet_linear_n_heads;
        let conv_len = (n_k * 2 + n_v) * hd * config.deltanet_conv_kernel_size;
        let mut state = vec![0.0f32; state_len];
        let mut conv = vec![0.0f32; conv_len];
        let mut scratch = DeltaNetLayerScratch::new(&config);
        let mut rng = Rng(0x5EED);
        let mut x = vec![0.0f32; config.n_embd];

        let mut decay_min = f32::MAX;
        let mut decay_max = 0.0f32;
        let mut decay_sum = 0.0f64;
        let mut max_state_norm = 0.0f64;

        for _ in 0..window {
            for v in x.iter_mut() {
                *v = rng.f32_centered();
            }
            let (m, mx) = mean_decay_at(&arms, &x, n_v);
            decay_min = decay_min.min(m);
            decay_max = decay_max.max(mx);
            decay_sum += m as f64;

            forward_deltanet_layer(&mut x, &w, &mut state, &mut conv, &config, &mut scratch);
            let norm: f64 = state.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>().sqrt();
            max_state_norm = max_state_norm.max(norm);
        }

        let alpha_bar = (decay_sum / window as f64) as f32;
        let horizon = 1.0 / (1.0 - decay_max);

        // Impulse erasure under the REAL gate statistics.
        let impulse_at = window / 4;
        let (steps, horizon_mean, horizon_max) =
            run_impulse(&config, &arms, impulse_at + 4096, impulse_at, 0x5EED);
        let steps = steps.expect("T1.3 FAIL: no 1/e decay");
        assert!(
            (steps as f32) <= horizon_max,
            "T1.3 layer {layer} FAIL: {steps} vs lambda_max horizon {horizon_max:.1}"
        );

        eprintln!(
            "T1.3 layer {layer}: alpha_bar={alpha_bar:.5} decay_max={decay_max:.5} \
             horizon(from step-max)={horizon:.1}; state norm max={max_state_norm:.3} (bounded) \
             impulse 1/e in {steps} (run horizons mean={horizon_mean:.1} max={horizon_max:.1})"
        );
    }
}
