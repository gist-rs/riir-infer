//! Wall Attention configuration — Issue 019 Phase C.1 de-fork.
//!
//! The local `WallConfig` struct was a v0.1.0 fork of the canonical
//! `katgpt_types::WallConfig` (Plan 173). It diverged in three ways:
//! - field rename `key_projected` (local) vs `use_key_projected` (canonical)
//! - extra `gate_proj_dim` field + `validate()` + `with_dims()` methods
//!   (now promoted upstream)
//! - extra `use_wall: bool` field (canonical uses `Option<WallConfig>` at
//!   the parent `Config.wall_config` level instead)
//!
//! All three are now reconciled: the canonical ships `gate_proj_dim`,
//! `validate`, `with_dims`, and serde derives; the `use_wall` field is
//! dropped (callers use the `Option<WallConfig>` pattern); the field
//! rename is consumed at every call site.

#[cfg(feature = "wall_attention")]
pub use katgpt_core::types::WallConfig;
