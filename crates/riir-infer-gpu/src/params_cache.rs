//! Issue 761 T3 — params-handle cache: kill the per-launch params
//! `create_from_slice` → `ScheduleTask::Write` → flush-submit storm.
//!
//! ## The measured problem (Bench 761 T1)
//!
//! M3 Metal decode is HOST-bound: the dispatch-only loop costs ~32.7 ms/token
//! of which only ~8-13% is the host blocked on GPU fences. The dominant cost
//! is a storm of **~577 `queue.submit`s per token**: every scalar-carrying
//! kernel launch helper uploads its params array with
//! `client.create_from_slice(f32::as_bytes(params))`, and each such upload is
//! a `ScheduleTask::Write` which forces a full stream flush at enqueue
//! (encoder finish + submit + memory_cleanup + release_uniforms ≈ 30-45 µs
//! each) — plus a heap `to_vec`, a pool reserve/free pair, and a staging copy.
//! ~95% of these params arrays are **byte-identical every token** (dims, eps,
//! inv_dim, caps — they only depend on model config).
//!
//! ## The fix
//!
//! Cache the params `Handle` keyed by `blake3(params_bytes)`. A hit returns a
//! clone of the persistent handle (refcount bump — no write, no Write task,
//! no flush, no allocation). A miss takes the old path once and inserts.
//! Params buffers are read-only kernel inputs (bound as storage-read), so
//! sharing one buffer between launch sites with identical bytes is safe.
//!
//! Volatile keys (pos/token-dependent: rope `pos`, kv-append `pos`, attention
//! `n_positions`, wte `row_idx`) miss every token by design — they keep the
//! legacy path (~4-6 Write tasks/token instead of ~577). The cache is capped
//! ([`PARAMS_CACHE_CAP`]); on overflow it clears wholesale (the bind-group
//! cache pattern — repopulation costs one legacy token).
//!
//! ## Gates
//!
//! - Feature `params_handle_cache` (default-off) compiles the cache in.
//! - Env `RIIR_PARAMS_CACHE=0|off|false|no` disables it at runtime (the A/B
//!   kill-switch for the GOAT bit-identity gate — one binary, two arms).
//! - Without the feature, [`params_handle`] is a pass-through to
//!   `client.create_from_slice` — default builds are behavior-identical.
//!
//! Correctness contract: the Handle a cache hit returns MUST carry the exact
//! bytes the caller passed (blake3 collision resistance ⇒ identical), and no
//! consumer may write to a params buffer (they are kernel inputs only —
//! enforced by the launch helpers, which bind them read-only).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::{ComputeClient, Runtime};
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Cache capacity before a wholesale clear. Sized so the ~380 static keys of
/// the Bonsai-27B decode path + ~6 volatile keys/token cover a ~600-token
/// generation before an evict (one legacy-cost token).
#[cfg(feature = "params_handle_cache")]
const PARAMS_CACHE_CAP: usize = 4096;

#[cfg(feature = "params_handle_cache")]
pub mod stats {
    /// Cache diagnostics (Issue 761 GOAT evidence): hits / misses / evictions.
    pub static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static EVICTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Reset the diagnostics counters.
    pub fn reset() {
        HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        MISSES.store(0, std::sync::atomic::Ordering::Relaxed);
        EVICTS.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(feature = "params_handle_cache")]
mod enabled {
    use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

    // Tri-state: 0 = follow env (cached on first read), 1 = force on, 2 = force off.
    // The override exists so one binary can A/B both arms for the GOAT gate
    // (Bench 761) without re-reading the env on the hot path.
    static OVERRIDE: AtomicU8 = AtomicU8::new(0);
    static ENV_ENABLED: AtomicBool = AtomicBool::new(false);
    static ENV_READ: AtomicBool = AtomicBool::new(false);

    fn env_enabled() -> bool {
        if !ENV_READ.swap(true, Ordering::Relaxed) {
            let enabled = !matches!(
                std::env::var("RIIR_PARAMS_CACHE")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "off" | "false" | "no"
            );
            ENV_ENABLED.store(enabled, Ordering::Relaxed);
        }
        ENV_ENABLED.load(Ordering::Relaxed)
    }

    pub fn cache_enabled() -> bool {
        match OVERRIDE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => env_enabled(),
        }
    }

    /// Force the cache on/off for benches + tests (`None` = follow env).
    pub fn set_override(enabled: Option<bool>) {
        OVERRIDE.store(
            match enabled {
                Some(true) => 1,
                Some(false) => 2,
                None => 0,
            },
            Ordering::Relaxed,
        );
    }
}

#[cfg(feature = "params_handle_cache")]
pub use enabled::set_override;

#[cfg(feature = "params_handle_cache")]
std::thread_local! {
    /// Thread-local because decode is single-threaded per client and the
    /// launch helpers are free functions without a cache parameter. Distinct
    /// threads get distinct caches (correct, at worst doubled memory).
    static PARAMS_CACHE: std::cell::RefCell<std::collections::HashMap<u64, Handle>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Return a GPU handle holding exactly `bytes`, cached when the
/// `params_handle_cache` feature is on (and the env kill-switch isn't).
///
/// Drop-in replacement for `client.create_from_slice(bytes)` at per-launch
/// params-upload sites. Without the feature this is a pass-through.
#[cfg(feature = "cubecl_runtime")]
pub fn params_handle<R: Runtime>(client: &ComputeClient<R>, bytes: &[u8]) -> Handle {
    #[cfg(not(feature = "params_handle_cache"))]
    {
        client.create_from_slice(bytes)
    }
    #[cfg(feature = "params_handle_cache")]
    {
        if bytes.is_empty() || !enabled::cache_enabled() {
            return client.create_from_slice(bytes);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(bytes);
        // blake3 Hash has no as_u64 in this version — take the first 8 bytes
        // of the digest (64-bit key; collision odds across ~4k live keys are
        // ~10⁻¹² — and a collision only causes a wrong-params READ, which the
        // G1 gate would surface immediately).
        let key = u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap());

        let cached = PARAMS_CACHE.with(|cache| cache.borrow().get(&key).cloned());
        if let Some(h) = cached {
            stats::HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return h;
        }
        stats::MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let h = client.create_from_slice(bytes);
        PARAMS_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= PARAMS_CACHE_CAP {
                cache.clear();
                stats::EVICTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            cache.insert(key, h.clone());
        });
        h
    }
}

/// Clear the params cache (test seam).
#[cfg(all(feature = "cubecl_runtime", feature = "params_handle_cache"))]
pub fn clear() {
    PARAMS_CACHE.with(|cache| cache.borrow_mut().clear());
}
