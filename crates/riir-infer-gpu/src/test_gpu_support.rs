//! GPU test support — shared helpers for heavy GPU test modules.
//!
//! Issue 712: the cubecl sliced pools only release a page back to the driver
//! (`vkFreeMemory`) on an **explicit** cleanup — freed *slices* stay committed
//! inside their pages otherwise. Tests that construct a full model instance
//! (~12 GB of weight/KV pages for gemma2_2b F32) must release those pages when
//! the instance drops, or later tests in the SAME process see a monotonically
//! growing committed footprint and OOM the 24 GB 4090 heap (whose DSD-thread
//! panics then silently drop launches and cascade).

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;
#[cfg(feature = "cubecl_runtime")]
use cubecl::client::ComputeClient;

/// Release fully-free GPU pool pages back to the driver (Issue 712).
///
/// Call AFTER dropping the model instance (handles must be free for the pages
/// to count as fully free). The tiny read round-trip forces the FIFO device
/// queue to process the cleanup before returning.
#[cfg(feature = "cubecl_runtime")]
pub(crate) fn gpu_release_pages(client: &ComputeClient<ActiveRuntime>) {
    client.memory_cleanup();
    let probe = client.empty(4);
    let _ = client.read_one(probe);
}

/// Process-wide serialization gate for tests that construct a FULL gemma2_2b
/// model instance (Issue 712).
///
/// One F32 instance holds ~8.5 GB of host weights plus ~12 GB of device
/// commit; libtest's default parallelism starts as many tests as there are
/// cores, and N concurrent full-model tests overcommit the box's commit limit
/// (32 GB RAM / 24 GB VRAM + pagefile). Measured on the 4090 (full
/// `cubecl_runtime` lib suite): at the suite tail ten gemma2_cubecl instances
/// were still live when the surviving tests' host allocations aborted the
/// process (`memory allocation of N bytes failed` -> STATUS_STACK_BUFFER_OVERRUN)
/// and light tests running beside the peak failed on garbage. Acquire this
/// gate for the WHOLE body of every full-model test: at most one heavy test
/// is live at a time (its trailing `gpu_release_pages` hands device pages
/// back before the next one starts) while light tests stay parallel.
#[cfg(all(test, feature = "cubecl_runtime"))]
static HEAVY_MODEL_TEST_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the process-wide full-model test gate (Issue 712). Hold the
/// returned guard for the whole test body:
/// `let _heavy = heavy_model_test_gate();`
#[cfg(all(test, feature = "cubecl_runtime"))]
pub(crate) fn heavy_model_test_gate() -> std::sync::MutexGuard<'static, ()> {
    HEAVY_MODEL_TEST_GATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
