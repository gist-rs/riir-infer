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
}
