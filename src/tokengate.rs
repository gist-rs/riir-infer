//! Plan 622 TokenGate lane switches (T1.5).
//!
//! Two-level gate over the prompt-skimming lane (per-key-bias attention +
//! exact pruned prefill + the deterministically-constructed gate overlay):
//!
//! - **Compile-time** — the `tokengate` cargo feature. OFF: the lane's
//!   serving code compiles away and [`runtime_enabled`] is a constant
//!   `false` that never reads the environment.
//! - **Runtime** — `RIIR_TOKENGATE` DEMOTES inside the feature (the
//!   kill-switch direction: an armed lane is disabled by env, never
//!   enabled by env — the `RIIR_MMORPG_EXAMPLES_GOAL_SALIENCE=0` /
//!   `LAYA_METAL_FLASH=0` posture). Demote literals: `0`, `false`, `off`
//!   (case-insensitive, trimmed); anything else — including unset —
//!   leaves the lane armed.
//!
//! The bias KERNEL capability (Phase 1's `*_bias` attention variants) is
//! deliberately NOT gated: it rides `Option` parameters and is a dispatch
//! identity at `None`. This feature gates the LANE — the serving quartet +
//! gate overlay (Phases 2–3) and the lane's harness targets (the
//! `required-features` rows). Phase 2's first consumer calls
//! [`runtime_enabled`] once at boot; callers must not read it per tick.

/// Compile-time-and-runtime gate read. Call ONCE at boot (the read is
/// cached; the feature-off arm folds to a constant `false` with no
/// environment access — the flag-off posture compiles the whole read away).
pub fn runtime_enabled() -> bool {
    #[cfg(feature = "tokengate")]
    {
        use std::sync::OnceLock;
        static ARMED: OnceLock<bool> = OnceLock::new();
        *ARMED.get_or_init(|| match std::env::var("RIIR_TOKENGATE") {
            Ok(v) => !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            ),
            Err(_) => true,
        })
    }
    #[cfg(not(feature = "tokengate"))]
    {
        false
    }
}

/// Plan 622 T2.1 — the scaffold-masked top-K selection (pure fn).
///
/// Scaffold = original positions `< n_sink` (the `kv_evict` sink convention,
/// `SinkWindowPolicy::default().n_sink`) PLUS the final position (the logits
/// source — always kept regardless of score). Among the remaining prunable
/// positions, the `k` highest scores survive; `k >=` prunable count keeps
/// everything. Ties break FIRST-INDEX (the determinism law); NaN scores sort
/// WORST — demoted to −inf in the sort key (NOT by `total_cmp` alone: IEEE
/// totalOrder puts NaN above +inf ascending, so a raw total_cmp descending
/// would rank NaN FIRST — pinned by the select_nan unit test). Writes
/// ASCENDING original indices into `out` (cleared first). One scratch
/// allocation for the prunable index sort (CPU reference — the G4 zero-alloc
/// law prices the serving path, not this lane's prune step).
#[cfg(feature = "tokengate")]
pub fn select_kept_into(scores: &[f32], n_sink: usize, k: usize, out: &mut Vec<usize>) {
    let t = scores.len();
    out.clear();
    // Scaffold: first `n_sink` positions + the final position (deduped below
    // — the two overlap when t <= n_sink).
    out.extend(0..n_sink.min(t));
    if t == 0 {
        return;
    }
    out.push(t - 1);

    // Prunable range: (n_sink .. t-1). Empty when the scaffold already covers
    // everything except (possibly) the last token.
    let prunable = n_sink..t.saturating_sub(1);
    let prunable_len = prunable.len();
    if k > 0 && prunable_len > 0 {
        let key = |s: f32| if s.is_nan() { f32::NEG_INFINITY } else { s };
        let mut ranked: Vec<usize> = prunable.collect();
        // Descending by score; equal scores keep ascending original index.
        // NaN demotes to −inf BEFORE the total_cmp (see the doc: a raw
        // total_cmp ranks NaN FIRST descending — the opposite of the law).
        ranked.sort_by(|&a, &b| {
            key(scores[b])
                .total_cmp(&key(scores[a]))
                .then_with(|| a.cmp(&b))
        });
        let take = k.min(prunable_len);
        out.extend_from_slice(&ranked[..take]);
    }

    out.sort_unstable();
    out.dedup();
}

/// Logit-space bias that makes a pad key dead under any finite real score:
/// `exp(-1e9 - max) == 0.0` exactly in f32, so the pad contributes `+0.0` to
/// the softmax denominator and `+0.0 * v_pad` to the numerator — the real
/// keys' sums are bitwise unchanged (a real key always supplies the max, and
/// `+0.0` additions cannot flip a non-negative partial sum).
pub const PAD_KEY_BIAS: f32 = -1.0e9;

/// Plan 622 T2.2 — the bucketed shape's pad-key bias rows (pure fn).
///
/// Writes head-major `[n_head][padded_len]` (the
/// `crate::transformer::KeyBiasLayout` contract with
/// `head_stride = padded_len`): slots `< real_len` get `0.0` (inert),
/// slots `>= real_len` get [`PAD_KEY_BIAS`] (dead key). This is the table the
/// fixed-shape arm (T2.4's graph-per-padded-length) relies on — its captured
/// kernel attends the FULL padded window per query, so the pads must be
/// masked in logit space, not by a per-query causal bound. The bucketed
/// CPU reference (`prefill_qwen_deltanet_pruned_into` with `bucket`) does NOT
/// consume this on its own scoring path — its real queries are causally
/// bounded below `real_len`, so pads are already unreachable and the bias
/// add would be dead weight; the law it embodies is pinned at kernel level
/// by `tests/plan622_bucketed_prefill_g1.rs` (full-window + pad bias ≡
/// causal window, bit-for-bit). When a caller bias table joins (Phase 3's
/// gate overlay, kept-order rows), the COMPOSED table must extend the same
/// way: gate rows for the kept slots, [`PAD_KEY_BIAS`] for the pads.
///
/// Zero-alloc (`*_into` form); `dst` is `n_head * padded_len`.
#[cfg(feature = "tokengate")]
pub fn pad_bias_rows_into(dst: &mut [f32], n_head: usize, real_len: usize, padded_len: usize) {
    assert!(
        dst.len() >= n_head * padded_len,
        "pad-bias buffer {} < {} rows",
        dst.len(),
        n_head * padded_len
    );
    assert!(real_len <= padded_len, "real rows exceed the padded width");
    for h in 0..n_head {
        let row = &mut dst[h * padded_len..][..padded_len];
        row[..real_len].fill(0.0);
        row[real_len..].fill(PAD_KEY_BIAS);
    }
}

/// Plan 622 T2.3 — the frozen prefill result a pruned DECODE consumes.
///
/// `full_rows` = the prefill's full prompt length T (the `<= L` band's row
/// count and the ORIGINAL-position clock); `kept_rows` = the `> L` band's
/// PACKED row count (`PrunedPrefillPlan::real_len`); `prune_at_layer` = L.
/// Invariant: the scaffold keeps the original final token, so the last row
/// of both bands is original position T-1 — a decode token's RoPE position
/// is `full_rows + step` on EVERY layer (one position clock), while the KV
/// write slot is band-local (`full_rows + step` at `<= L`, `kept_rows +
/// step` above). That slot/phase split is the decode half of the subset-
/// softmax exactness law (compaction moves storage, never phases — the
/// `kv_evict` law, same shape).
#[cfg(feature = "tokengate")]
pub struct PrunedDecodeLayout {
    pub prune_at_layer: usize,
    pub full_rows: usize,
    pub kept_rows: usize,
}

#[cfg(feature = "tokengate")]
impl PrunedDecodeLayout {
    /// Build from the prefill's own bookkeeping (`prompt_len` = the full T).
    pub fn from_plan(
        prompt_len: usize,
        plan: &crate::deltanet::forward::PrunedPrefillPlan,
    ) -> Self {
        Self {
            prune_at_layer: plan.prune_at_layer,
            full_rows: prompt_len,
            kept_rows: plan.real_len,
        }
    }
}

/// Plan 622 T2.3 — the gate FOLDED ONCE at prefill end into per-layer
/// decode-side surfaces: an additive per-key mask (the bias rows, PACKED
/// slot order) + the per-head logit scale `s_eff` (applied as a query
/// multiply — `(s·q)·k = s·(q·k)`, the bias rides AFTER the scale, so the
/// mask entries are unscaled logits). Dense layout, the T1.2 CUDA wire
/// shape: layer l's bias window starts at `l * n_head * head_stride`, rows
/// head-major; layer l's scales sit at `scale[l * n_head ..][..n_head]`.
/// Layers `<= prune_at_layer` never read their windows (the gate exists
/// above L only), and GDN layers consume neither surface.
///
/// CONTRACT (the constructor's — Phase 3's scorer builds this at prefill
/// end, once): `head_stride >= kept_rows + every decode step the caller
/// will run`, slots `< kept_rows` carry the prompt mask, slots `>=
/// kept_rows` are ZERO (decode-appended keys are always attended — bias
/// 0, they are never pruned) and scales are exactly `1.0` where the lane
/// has no opinion (the `≡ 1` whole-table check then skips the q-multiply
/// entirely — dispatch identity, and `q * 1.0` is a bitwise no-op besides).
#[cfg(feature = "tokengate")]
pub struct PrunedDecodeGate<'a> {
    pub bias: &'a [f32],
    pub head_stride: usize,
    pub scale: Option<&'a [f32]>,
}

/// Plan 622 T3.1 — the rank-ratio law `r = ln(max(2, m)) / ln(max(2, p))`:
/// the query's rank among KEPT tokens over its rank in the FULL prompt
/// (both 1-based; `m <= p` always — kept ⊆ full). The `max(2, ·)` guard is
/// SYMMETRIC on purpose: a denominator-only guard breaks the `r ≡ 1`
/// identity at rank 1 (ln 1 = 0 over ln 2 = 0 gives r = 0 for the very
/// token whose m == p) — the unit test pins the identity at p = 1 too.
/// `m == p` (the K = full posture) short-circuits to exactly 1.0 so the
/// overlay is an EXACT no-op there for any γ.
#[cfg(feature = "tokengate")]
#[inline]
pub fn rank_ratio(rank_kept: usize, rank_full: usize) -> f32 {
    debug_assert!(rank_kept >= 1 && rank_full >= 1, "ranks are 1-based");
    debug_assert!(
        rank_kept <= rank_full,
        "kept rank {} exceeds full rank {}",
        rank_kept,
        rank_full
    );
    if rank_kept == rank_full {
        return 1.0;
    }
    let m = rank_kept.max(2) as f64;
    let p = rank_full.max(2) as f64;
    (m.ln() / p.ln()) as f32
}

/// Plan 622 T3.1 — the rank-ratio cooling `s_eff = 1 + γ·(r − 1)` (the
/// paper's per-query temperature, the code form the research quotes). γ = 1
/// is the lane's fixed value (Plan 622 T3.1); any γ is accepted here and
/// the K = full posture returns exactly 1.0 FOR ANY γ (the identity law —
/// pinned by the unit test). Consumers: the `> L` decode gate's per-key
/// scale window (T2.3's `PrunedDecodeGate::scale`), keyed by the KEPT rank
/// of each packed slot.
#[cfg(feature = "tokengate")]
#[inline]
pub fn rank_cooling(rank_kept: usize, rank_full: usize, gamma: f32) -> f32 {
    1.0 + gamma * (rank_ratio(rank_kept, rank_full) - 1.0)
}

/// Plan 622 T3.1 — sequence-RMSNorm scorer normalization, fused with the
/// readout dot: `scores[i] = readout · (h_i / rms(h_i))`, one pure pass
/// over the layer-L residual rows `[rows, n]`. Per-TOKEN row norm (the
/// codebase's own `rmsnorm` reading — each token scale-invariant, so the
/// readout dot is a pure direction measure comparable across the sequence);
/// gamma-free (the modelless overlay carries no trained gains — a readout
/// row applies its own scale by construction). The alternative reading
/// (one global statistic over the whole [rows × n] buffer) is a named
/// choice for T3.4's ablation rows, not a second fn. Zero rows are safe
/// (inv_rms = 1/√eps, score 0 — pinned). f64 statistics, f32 readout;
/// the dot folds sequentially (the Issue-940 determinism law).
/// Zero-alloc (`*_into`); `dst` is `rows`, `readout` is `n`.
#[cfg(feature = "tokengate")]
pub fn seq_rmsnorm_readout_into(
    dst: &mut [f32],
    hidden: &[f32],
    readout: &[f32],
    n: usize,
    eps: f64,
) {
    let rows = dst.len();
    assert_eq!(hidden.len(), rows * n, "hidden rows mismatch");
    assert_eq!(readout.len(), n, "readout width mismatch");
    for i in 0..rows {
        let row = &hidden[i * n..(i + 1) * n];
        let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv_rms = 1.0 / (sum_sq / n as f64 + eps).sqrt();
        let mut acc = 0.0f32;
        for (v, w) in row.iter().zip(readout.iter()) {
            acc += ((*v as f64 * inv_rms) as f32) * w;
        }
        dst[i] = acc;
    }
}

#[cfg(test)]
mod tests {
    // Feature-off posture: the switch is the constant false, no env read.
    // (The feature-on arm's env behavior is a deployment switch, not a
    // unit-test surface — it is exercised by the walk harness boot at the
    // armed posture.)
    #[cfg(not(feature = "tokengate"))]
    #[test]
    fn feature_off_the_switch_is_constant_false() {
        assert!(!super::runtime_enabled());
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn feature_on_the_switch_is_a_bool() {
        // Value-agnostic at test time (the env may carry a demote literal on
        // a shared box): the only law unit-testable without env control is
        // that the read terminates and is stable across calls.
        let a = super::runtime_enabled();
        let b = super::runtime_enabled();
        assert_eq!(a, b);
    }

    // ---- select_kept_into (T2.1 selection laws) ------------------------

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_scaffold_survives_the_worst_scores() {
        // Scores poison the scaffold; it is kept anyway.
        let scores = vec![0.1, -1e9, -1e9, -1e9, -1e9, 0.5];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 4, 0, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 5]); // sinks + last
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_topk_over_prunable_first_index_on_ties() {
        // k=2 over prunable 4..5 (single token) → t=6, prunable = {4}.
        let scores = vec![0.0, 0.0, 0.0, 0.0, 9.0, 0.0];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 4, 1, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5]);

        // Ties: t=8, n_sink=2, k=2, all prunable equal → first-index wins
        // (2 and 3, never 6/7 despite identical scores).
        let scores = vec![0.0; 8];
        super::select_kept_into(&scores, 2, 2, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 7]);
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_k_at_least_prunable_keeps_everything() {
        let scores = vec![0.3, 0.1, 0.4, 0.1, 0.5, 0.9];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 4, usize::MAX, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5]);
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_nan_sorts_worst_and_tiny_prompts_dedup() {
        // NaN scores never beat a real score.
        let scores = vec![f32::NAN, f32::NAN, 0.2, f32::NAN, 0.1, f32::NAN];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 0, 1, &mut out);
        assert_eq!(out, vec![2, 5]); // scaffold(last=5) + best real (2)

        // t <= n_sink: scaffold is everything, last dedups, no panic.
        let scores = vec![1.0, 2.0, 3.0];
        super::select_kept_into(&scores, 8, 5, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn rank_ratio_is_exactly_one_when_ranks_match_including_rank_one() {
        // The identity law, every posture the asymmetric-guard variant would
        // have broken: m == p → exactly 1.0, INCLUDING p = 1 (ln 1 = 0 —
        // the raw formula gives 0/ln2 = 0 there; the symmetric guard + the
        // short-circuit are what make the claim true).
        for p in [1usize, 2, 3, 10, 24, 1000] {
            assert_eq!(super::rank_ratio(p, p), 1.0, "m == p at p={p}");
        }
        // And the cooling is therefore an exact no-op at K = full for ANY γ.
        for gamma in [0.0f32, 1.0, 2.5, -1.0, 100.0] {
            assert_eq!(super::rank_cooling(7, 7, gamma), 1.0);
        }
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn rank_ratio_cools_monotonically_and_saturates_at_the_rank_two_floor() {
        // Deeper in the kept order → cooler (r strictly decreases in m).
        let r1 = super::rank_ratio(2, 10);
        let r2 = super::rank_ratio(5, 10);
        let r3 = super::rank_ratio(9, 10);
        assert!(r1 < r2 && r2 < r3 && r3 < 1.0);
        // The rank-2 floor: m = 1 floors to 2 (ln 2 / ln 10), never 0.
        let floor = super::rank_ratio(1, 10);
        assert_eq!(floor, r1, "m=1 and m=2 share the max(2, ·) floor");
        let expected = (2.0f64.ln() / 10.0f64.ln()) as f32;
        assert_eq!(floor, expected);
        // Cooling at γ = 1: s = 1 + (r − 1) by the definition.
        let s = super::rank_cooling(2, 10, 1.0);
        assert_eq!(s, 1.0 + 1.0 * (r1 - 1.0));
        assert!(s < 1.0, "a pruned rank cools below 1");
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn seq_rmsnorm_readout_known_answer_and_zero_row_safety() {
        let eps = 1e-6f64;
        // Hand-computable: row [3, 4], rms = √((9+16)/2) = √12.5, readout
        // [1, 0] → score = 3/√12.5.
        let hidden = [3.0f32, 4.0];
        let readout = [1.0f32, 0.0];
        let mut scores = [0.0f32; 1];
        super::seq_rmsnorm_readout_into(&mut scores, &hidden, &readout, 2, eps);
        let expected = (3.0f64 / 12.5f64.sqrt()) as f32;
        assert!(
            (scores[0] - expected).abs() < 1e-6,
            "{} vs {expected}",
            scores[0]
        );

        // Row independence: a second row does not move the first (and the
        // second row's own score is its direction measure: [100, -100] has
        // rms 100 → [1, -1], readout [1, 0] → 1.0).
        let two_rows = [3.0, 4.0, 100.0, -100.0];
        let mut two_scores = [0.0f32; 2];
        super::seq_rmsnorm_readout_into(&mut two_scores, &two_rows, &readout, 2, eps);
        assert_eq!(two_scores[0].to_bits(), scores[0].to_bits());
        assert!((two_scores[1] - 1.0).abs() < 1e-6);

        // Zero row: inv_rms = 1/√eps, score 0, finite — no NaN.
        let zeros = [0.0f32; 4];
        super::seq_rmsnorm_readout_into(&mut two_scores, &zeros, &readout, 2, eps);
        assert_eq!(two_scores[0], 0.0);
        assert!(two_scores[0].is_finite());
    }
}
