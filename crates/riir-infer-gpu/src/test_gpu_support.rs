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

/// Issue 037: measured VRAM requirement for ONE full-model gemma2_2b test.
///
/// The issue-037 bisect (serialized full-suite run, 4090, co-resident 7.6 GiB
/// trainer) measured the first heavy test's working set at ~16.2 GiB
/// test-side, and the loss fired when GLOBAL occupancy hit 23.8 of 24.5 GiB
/// (16.9 GiB free was NOT enough). 18 GiB = the ~16.2 GiB working set plus
/// ~1.8 GiB staging/pool slack, clearing the measured failure point with
/// real margin; the env override exists for boxes where the measurement does
/// not transfer.
#[cfg(all(test, feature = "cubecl_runtime"))]
pub(crate) const HEAVY_TEST_MIN_FREE_BYTES: u64 = 18 * 1024 * 1024 * 1024;

/// Env override for [`heavy_model_vram_guard`]: set `1` to run the heavy
/// tests regardless of measured headroom (the historical behavior).
#[cfg(all(test, feature = "cubecl_runtime"))]
pub(crate) const ENV_FORCE_HEAVY_TESTS: &str = "RIIR_GPU_FORCE_HEAVY_TESTS";

/// Windows NVML global-occupancy probe: `nvidia-smi` reports TRUE global
/// memory used across ALL processes — the only Windows signal that sees a
/// co-resident CUDA trainer (measured, issue 037: BOTH WDDM `Budget` and
/// NVIDIA's `VK_EXT_memory_budget` report near-full budgets and process-local
/// usage while the box is actually 97% occupied — 22.8 GiB "free" measured
/// against 16.9 GiB truly free). Bounded, best-effort, fail-open: any spawn
/// or parse failure returns `None` and the caller falls back to the
/// in-process budget probes.
#[cfg(all(test, feature = "cubecl_runtime", target_os = "windows"))]
fn nvml_global_free_bytes() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // First line: "<used MiB>, <total MiB>" (a second GPU appends lines —
    // take the max-total line so a multi-GPU box gates on the GPU we'd use).
    let mut best: Option<(u64, u64)> = None;
    for line in text.lines() {
        let mut parts = line.split(',').map(str::trim);
        let (Some(used), Some(total)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Ok(used) = used.parse::<u64>() else { continue };
        let Ok(total) = total.parse::<u64>() else { continue };
        if best.is_none_or(|(bt, _)| total > bt) {
            best = Some((total, used));
        }
    }
    let (total, used) = best?;
    Some(total.saturating_sub(used) * 1024 * 1024)
}

/// VRAM-aware admission for one full-model test (Issue 037).
///
/// Returns the heavy gate (already holding [`HEAVY_MODEL_TEST_GATE`]) when
/// the box has enough AVAILABLE VRAM, or `None` after printing a LOUD skip
/// line. The skip exists because an OOM-class failure at the marginal
/// allocation is escalated by wgpu-core (`handle_hal_error` → `device.lose()`)
/// into a process-wide device loss, which the Issue-676 shared device then
/// cascades across every later GPU test — 16 victims measured. Skipping the
/// heavy tests keeps the other ~239 tests green on a co-resident box; run
/// the skipped tests on an exclusive window or force them with
/// `RIIR_GPU_FORCE_HEAVY_TESTS=1`.
///
/// Headroom source, in order: (1) Windows — the [`nvml_global_free_bytes`]
/// global-occupancy probe (the only signal that sees co-resident compute);
/// (2) the in-process budget probes
/// (`CubeCLContext::available_video_memory` — honest on Linux/Metal); both
/// `None` FAILS OPEN (historical behavior preserved, never silently narrowed
/// coverage on boxes nothing can read).
#[cfg(all(test, feature = "cubecl_runtime"))]
pub(crate) fn heavy_model_vram_guard(test: &str) -> Option<std::sync::MutexGuard<'static, ()>> {
    #[cfg(target_os = "windows")]
    let nvml = nvml_global_free_bytes();
    #[cfg(not(target_os = "windows"))]
    let nvml: Option<u64> = None;
    let available = nvml.or_else(|| {
        crate::cubecl_runtime::CubeCLContext::new()
            .ok()
            .and_then(|ctx| ctx.available_video_memory())
    });
    if std::env::var(ENV_FORCE_HEAVY_TESTS).as_deref() == Ok("1") {
        println!(
            "[heavy-test] RUN {test}: {ENV_FORCE_HEAVY_TESTS}=1 override — measured headroom \
             {measured}.",
            measured = available
                .map(|b| format!("{:.1} GiB", b as f64 / (1024.0 * 1024.0 * 1024.0)))
                .unwrap_or_else(|| "UNAVAILABLE".into()),
        );
        return Some(heavy_model_test_gate());
    }
    match available {
        Some(bytes) if bytes >= HEAVY_TEST_MIN_FREE_BYTES => {
            println!(
                "[heavy-test] RUN {test}: {:.1} GiB free VRAM measured (need {req:.1}).",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                req = HEAVY_TEST_MIN_FREE_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            Some(heavy_model_test_gate())
        }
        other => {
            let measured = other
                .map(|b| format!("{:.1} GiB", b as f64 / (1024.0 * 1024.0 * 1024.0)))
                .unwrap_or_else(|| "UNAVAILABLE (probe returned None — fail-open declined)".into());
            eprintln!(
                "[heavy-test] SKIP {test}: needs {req:.1} GiB free VRAM, measured {measured}. \
                 Co-resident GPU work exhausts the commit budget and an OOM here escalates to a \
                 process-wide device loss (issue 037). Run on an exclusive window or set \
                 {ENV_FORCE_HEAVY_TESTS}=1 to override.",
                req = HEAVY_TEST_MIN_FREE_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            None
        }
    }
}
