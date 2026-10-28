//! Issue 1151 — the promoted serving posture as DATA.
//!
//! [`PRODUCTION_RUNGS`] mirrors the `production` aggregate feature row in
//! this crate's Cargo.toml one-to-one: each entry names a rung and reports
//! — by `cfg!`, truthful for THIS compilation under cargo's feature
//! unification — whether the rung is compiled in. Consumers
//! (riir-infer-dist's `gpu` posture; the engine's serving lanes) forward
//! `riir-infer-gpu/production` and inherit every rung;
//! [`missing_production_rungs`] names any drift.
//!
//! Contract (the Cargo.toml row carries it too): joining the aggregate IS
//! the promotion act (GOAT-gated); opt-in rungs stay out; demotion stays
//! env kill-switches — cargo features are additive-only, the house
//! demote-only law. The in-module tests pin this manifest to the Cargo.toml
//! row in BOTH directions, so a rung added to one and forgotten in the
//! other reds at any `cargo test`.

/// Every promoted serving rung, paired with whether THIS compilation
/// selected it. Order follows the Cargo.toml `production` row.
pub const PRODUCTION_RUNGS: &[(&str, bool)] = &[
    ("ternary_gemv", cfg!(feature = "ternary_gemv")),
    ("ternary_gemm_batched", cfg!(feature = "ternary_gemm_batched")),
    ("ternary_gemm_simdgroup", cfg!(feature = "ternary_gemm_simdgroup")),
    (
        "ternary_deltanet_chunked_prefill",
        cfg!(feature = "ternary_deltanet_chunked_prefill"),
    ),
    (
        "ternary_attention_batched_prefill",
        cfg!(feature = "ternary_attention_batched_prefill"),
    ),
    (
        "deltanet_recurrence_rowpar",
        cfg!(feature = "deltanet_recurrence_rowpar"),
    ),
    (
        "deltanet_recurrence_parallel",
        cfg!(feature = "deltanet_recurrence_parallel"),
    ),
    ("delta_routing", cfg!(feature = "delta_routing")),
    ("params_handle_cache", cfg!(feature = "params_handle_cache")),
    ("metal_tensor_gemm", cfg!(feature = "metal_tensor_gemm")),
];

/// Promoted rungs the current compilation is MISSING. Empty iff the
/// `production` posture is fully compiled in.
pub fn missing_production_rungs() -> Vec<&'static str> {
    PRODUCTION_RUNGS
        .iter()
        .filter(|(_, selected)| !*selected)
        .map(|(name, _)| *name)
        .collect()
}

/// True iff every promoted serving rung is compiled in.
pub fn production_posture_complete() -> bool {
    PRODUCTION_RUNGS.iter().all(|(_, selected)| *selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The floor the manifest parser must never fall under — a parse that
    /// stops matching anything must RED, never pass vacuously.
    const MIN_MANIFEST_RUNGS: usize = 10;

    /// Parse the quoted rung names out of the Cargo.toml `production`
    /// row — bracket-slice, line-agnostic (single- or multi-line rows
    /// both parse; the row carries no nested brackets or inline
    /// comments). A missing/unterminated row fails LOUDLY here rather
    /// than returning a vacuous empty set.
    fn manifest_production_rungs() -> Vec<String> {
        let manifest = include_str!("../Cargo.toml");
        let trigger = "production = [";
        let start = manifest
            .find(trigger)
            .unwrap_or_else(|| panic!("production row missing from Cargo.toml"));
        let rest = &manifest[start + trigger.len()..];
        let end = rest
            .find(']')
            .unwrap_or_else(|| panic!("production row unterminated in Cargo.toml"));
        rest[..end]
            .split(',')
            .map(|piece| piece.trim().trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    #[test]
    fn manifest_and_const_agree_in_both_directions() {
        let manifest = manifest_production_rungs();
        assert!(
            manifest.len() >= MIN_MANIFEST_RUNGS,
            "production row parse floor tripped: parsed {} rungs, floor {}              — the parser or the row broke; never pass vacuously",
            manifest.len(),
            MIN_MANIFEST_RUNGS
        );
        let const_names: Vec<&str> = PRODUCTION_RUNGS.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            const_names.len(),
            manifest.len(),
            "const and Cargo.toml production row disagree on count: const              {const_names:?} vs manifest {manifest:?} (Issue 1151)"
        );
        for name in &const_names {
            assert!(
                manifest.iter().any(|m| m == name),
                "PRODUCTION_RUNGS names `{name}` but the Cargo.toml                  production row does not — the data manifest diverged from                  the aggregate (Issue 1151)"
            );
        }
        for m in &manifest {
            assert!(
                const_names.iter().any(|c| c == m),
                "Cargo.toml production row names `{m}` but PRODUCTION_RUNGS                  does not — a rung joined the aggregate without joining the                  data manifest (Issue 1151)"
            );
        }
    }

    #[test]
    fn rungs_manifest_is_not_drained() {
        assert!(
            PRODUCTION_RUNGS.len() >= MIN_MANIFEST_RUNGS,
            "PRODUCTION_RUNGS drained below the floor — the consumer-side              missing-checks would pass vacuously (Issue 1151)"
        );
        let mut names: Vec<&str> = PRODUCTION_RUNGS.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(
            names.len(),
            count,
            "PRODUCTION_RUNGS carries a duplicate name (Issue 1151)"
        );
    }
}
