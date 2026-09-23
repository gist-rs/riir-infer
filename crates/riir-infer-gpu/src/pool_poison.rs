//! Issue 994 — GPU memory-pool poison detection and refusal.
//!
//! cubecl-wgpu's pool reserve panics on device-memory exhaustion
//! (`compute/server.rs` `initialize_memory`:
//! `panic!("failed to reserve {size} bytes of device memory: {err}")`).
//! Measured on the 4090 (riir-train Bench 600, cross-filed as issue 994):
//! at ≥~65K contexts the reserve panics land on background threads, are
//! absorbed, and the forward COMPLETES with deterministically corrupted
//! activations — no abort, plausible shapes, bit-stable across runs. A
//! failure this shape passes every in-process consistency check and is
//! only detectable against an external reference.
//!
//! This module converts that silent-corruption class into a loud refusal:
//!
//! 1. **Panic-hook chain** — a process hook matches the reserve-failure
//!    signature and records it, then delegates to the previous hook, so
//!    messages keep printing and any pre-existing hook keeps working.
//! 2. **wgpu uncaptured errors** — the CubeCL device's uncaptured-error
//!    handler (the Metal-side face of the same exhaustion class: wgpu's
//!    `create_buffer` is infallible and reports OOM through the handler)
//!    records the error first, then re-panics with wgpu's default phrase
//!    so the error site stays exactly as loud as the handler it replaces.
//!
//! [`poisoned()`] is the single read: `None` = healthy (one relaxed
//! atomic load); `Some(detail)` = a pool failure was observed since
//! process start. [`TernaryDeltanetGpuForward`] refuses at construction,
//! at every prefill-chunk and decode-funnel entry, and after every
//! result-producing unit — corrupted results can never leave the forward
//! as a successful return.
//!
//! Scope: the CubeCL/wgpu path. The cudarc twin surfaces OOM as typed
//! `Result`s (cudaMalloc error codes) and never takes this path. The
//! capacity fix for 64K+ contexts (paged/offloaded KV) is riir-train
//! Issue 452 T5 — this module owns the FAILURE MODE, not the capacity.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once, OnceLock};

/// cubecl-wgpu's own panic text (`initialize_memory`, cubecl-wgpu
/// `compute/server.rs`). Both fragments TOGETHER are specific enough that
/// an unrelated library's "failed to reserve …" cannot false-positive —
/// matching one alone would record thread-spawn and allocator messages
/// that have nothing to do with the GPU pool.
const RESERVE_SIG: (&str, &str) = ("failed to reserve", "bytes of device memory");

static HOOK_INSTALLED: Once = Once::new();
static POISONED: AtomicBool = AtomicBool::new(false);
static DETAIL: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn detail_slot() -> &'static Mutex<Option<String>> {
    DETAIL.get_or_init(|| Mutex::new(None))
}

fn matches_reserve_signature(msg: &str) -> bool {
    msg.contains(RESERVE_SIG.0) && msg.contains(RESERVE_SIG.1)
}

/// Install the panic-hook chain (idempotent; `Once`). Called from
/// `CubeCLContext::new_uncached` — i.e. before any GPU forward can exist —
/// and directly by the G1 gate. Chaining (take → wrap → set) preserves
/// whatever hook was installed before us.
pub fn ensure_hooks_installed() {
    HOOK_INSTALLED.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                String::new()
            };
            if matches_reserve_signature(&msg) {
                record(format!("cubecl-wgpu pool reserve panicked: {msg}"));
            }
            previous(info);
        }));
    });
}

/// Record a pool failure. First writer wins — the FIRST failure names the
/// state (later failures add nothing), because the first is the one whose
/// partial allocation the pool is now serving from.
pub(crate) fn record(detail: String) {
    POISONED.store(true, Ordering::Release);
    let mut slot = match detail_slot().lock() {
        Ok(guard) => guard,
        // A thread panicked while holding the slot: the data is still sound
        // (String), recover it rather than cascading a mutex poison into a
        // detection-path panic.
        Err(poisoned) => poisoned.into_inner(),
    };
    if slot.is_none() {
        *slot = Some(detail);
    }
}

/// `None` = healthy (one relaxed-order load on the hot path — checked at
/// every forward/prefill entry and exit). `Some(detail)` = a GPU pool
/// failure was observed since process start; every result this process
/// has produced since that failure is untrustworthy.
pub fn poisoned() -> Option<String> {
    if !POISONED.load(Ordering::Acquire) {
        return None;
    }
    let slot = match detail_slot().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    slot.clone()
}

/// wgpu uncaptured-error handler (wgpu 30: `Arc<dyn Fn(wgpu::Error)>`).
/// Records into the poison state FIRST — on the absorbed-background-thread
/// path the panic below dies with its thread, and the record is the only
/// thing that survives — then re-panics with wgpu's default phrase so the
/// error site keeps the exact loudness of the default handler this
/// replaces. Installed on the retained CubeCL device at context creation.
#[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
pub fn install_uncaptured_handler(device: &wgpu::Device) {
    let handler: std::sync::Arc<dyn wgpu::UncapturedErrorHandler> =
        std::sync::Arc::new(|err: wgpu::Error| {
            let detail = format!("wgpu uncaptured error: {err}");
            eprintln!("[issue 994] {detail}");
            record(detail);
            panic!("Handling wgpu errors as fatal by default");
        });
    device.on_uncaptured_error(handler);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests share the process-global poison state; each test resets it so
    /// ordering can never couple them.
    fn reset_poison_for_test() {
        POISONED.store(false, Ordering::Release);
        *detail_slot().lock().unwrap() = None;
    }

    #[test]
    fn reserve_signature_matches_the_cubecl_panic_text() {
        // The exact message shape from the 4090 specimen (issue 994 /
        // riir-train Bench 600): sizes differ, both fragments present.
        assert!(matches_reserve_signature(
            "failed to reserve 285212672 bytes of device memory: out of device memory allocating 1582170112 bytes"
        ));
        // Either fragment alone is NOT enough — a thread/allocator message
        // that says "failed to reserve" must not poison the GPU pool state.
        assert!(!matches_reserve_signature(
            "failed to reserve stack for thread spawn"
        ));
        assert!(!matches_reserve_signature(
            "allocation of N bytes of device memory rejected"
        ));
        // Ordinary panics never match.
        assert!(!matches_reserve_signature(
            "called `Option::unwrap()` on a `None` value"
        ));
        assert!(!matches_reserve_signature(""));
    }

    #[test]
    fn record_then_poisoned_roundtrips_and_first_writer_wins() {
        reset_poison_for_test();
        assert_eq!(poisoned(), None, "fresh state must read healthy");

        record("first failure detail".to_string());
        record("second failure detail".to_string());
        let detail = poisoned().expect("poisoned() must report after record()");
        assert!(
            detail.contains("first failure detail"),
            "first writer must win, got: {detail}"
        );
        reset_poison_for_test();
        assert_eq!(poisoned(), None, "reset must clear the state");
    }
}
