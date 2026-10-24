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
}
