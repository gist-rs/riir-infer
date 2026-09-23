//! GEMV autotune — runtime benchmarking of plane vs tiled kernel variants.
//!
//! On first call for each unique (m, n) dimension pair, benchmarks both the
//! plane (subgroup) and tiled GEMV variants to determine which is faster on the
//! current device. Results are cached for the process lifetime.
//!
//! For Gemma 2 2B, there are ~6 unique (m, n) pairs across all GEMV operations:
//! - QKV:    (2048×2304), (1024×2304), (1024×2304)
//! - Wo:     (2304×2048)
//! - Gate:   (9216×2304)
//! - Up:     (9216×2304)
//! - Down:   (2304×9216)
//! - lm_head: (256000×2304)
//!
//! The one-time benchmark cost (~1-2s total) pays for itself across millions
//! of kernel launches during inference.
//!
//! The cache is a `papaya::HashMap` — lock-free, read-mostly fit. Each unique
//! (m, n) is benchmarked once then read on every subsequent launch.
//!
//! ```sh
//! RUST_LOG=info cargo test -p riir-gpu --features cubecl_runtime test_gemv_autotune -- --nocapture
//! ```

#[cfg(feature = "cubecl_runtime")]
use std::time::Instant;

#[cfg(feature = "cubecl_runtime")]
use cubecl::Runtime;
#[cfg(feature = "cubecl_runtime")]
use cubecl::client::ComputeClient;
#[cfg(feature = "cubecl_runtime")]
use cubecl::features::Plane;
#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::CubeElement;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use crate::gemv_cubecl::GemvCubeCL;

// ---------------------------------------------------------------------------
// GemvVariant
// ---------------------------------------------------------------------------

/// GEMV kernel variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GemvVariant {
    /// Plane (subgroup) cooperative dot product with `plane_sum()`.
    /// Best on Metal where subgroup size = 32.
    Plane,
    /// Shared memory tiling, one thread per row. Fallback.
    Tiled,
}

impl std::fmt::Display for GemvVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GemvVariant::Plane => write!(f, "plane"),
            GemvVariant::Tiled => write!(f, "tiled"),
        }
    }
}

// ---------------------------------------------------------------------------
// GemvAutotune
// ---------------------------------------------------------------------------

/// Autotune cache for GEMV kernel variant selection.
///
/// Caches the fastest variant per (m, n) dimension pair in a **process-global**
/// map: the fastest variant for a shape is a property of the device + shape,
/// not of the model instance. Every `GpuGemma*` runtime shares the single
/// `ActiveRuntime` device, so a process-wide cache means each unique (m, n) is
/// benchmarked (and its benchmark tensors allocated) **once per process** —
/// not once per model instance. Per-instance caches re-benchmarked every
/// construction, whose benchmark tensors (up to 256 MB per shape) stacked on
/// top of two live model instances' weights to tip the 24 GB 4090 heap into
/// WDDM-spill OOM (Issue 712).
/// Thread-safe via lock-free `papaya::HashMap` (read-mostly: each unique
/// (m, n) is benchmarked once on the slow path, then read on every
/// subsequent launch — the classic papaya fit per the global rule).
///
/// # Usage
///
/// ```ignore
/// let autotune = GemvAutotune::new();
/// // First call for (2048, 2304) in the process: benchmarks both variants (~100ms)
/// // Subsequent calls (any instance): returns cached variant (~ns)
/// autotune.launch::<ActiveRuntime>(&client, weight, input, output, 2048, 2304);
/// ```
pub struct GemvAutotune {
    /// Number of warmup iterations before timing (JIT compilation, cache priming).
    warmup_iters: usize,
    /// Number of timed iterations (median used for selection).
    bench_iters: usize,
}

/// Process-global autotune results, keyed by (m, n).
///
/// Shared by every `GemvAutotune` instance (all model runtimes use the one
/// `ActiveRuntime` device). See the struct doc for why the cache is global.
static GLOBAL_CACHE: std::sync::LazyLock<papaya::HashMap<(usize, usize), GemvVariant>> =
    std::sync::LazyLock::new(papaya::HashMap::new);

impl GemvAutotune {
    /// Create a new autotuner with default benchmark settings.
    ///
    /// Defaults: 3 warmup + 5 timed iterations per variant. Results land in
    /// the process-global cache (see [`GLOBAL_CACHE`]).
    pub fn new() -> Self {
        Self {
            warmup_iters: 3,
            bench_iters: 5,
        }
    }

    /// Create with custom benchmark settings.
    pub fn with_iters(warmup: usize, bench: usize) -> Self {
        Self {
            warmup_iters: warmup,
            bench_iters: bench,
        }
    }

    /// Select the best GEMV variant for the given dimensions.
    ///
    /// On first call for (m, n), benchmarks both variants and caches the result.
    /// Subsequent calls return the cached selection.
    ///
    /// **Diagnostic override (Plan 409):** `GEMV_FORCE_TILED=1` forces the
    /// tiled variant for all dimensions, bypassing the benchmark. Used to
    /// isolate whether the plane kernel has a correctness bug.
    pub fn select<R: Runtime>(&self, client: &ComputeClient<R>, m: usize, n: usize) -> GemvVariant {
        // Plan 409 diagnostic: force tiled variant to test plane-kernel bug hypothesis.
        if std::env::var("GEMV_FORCE_TILED").as_deref() == Ok("1") {
            return GemvVariant::Tiled;
        }

        let key = (m, n);

        // Fast path: cache hit (lock-free read via papaya guard).
        if let Some(variant) = GLOBAL_CACHE.pin().get(&key) {
            return *variant;
        }

        // Slow path: benchmark (write via papaya pin).
        let has_plane = client.features().plane.contains(Plane::Ops);
        let best = if has_plane {
            self.benchmark_variants::<R>(client, m, n)
        } else {
            println!("[gemv_autotune] ({m}×{n}): no plane support, using tiled");
            GemvVariant::Tiled
        };

        // Converge on the FIRST settled cache value, not our local benchmark
        // result (Issue 949). Under a parallel cold start (measured: 4 model
        // instances in one process racing the same (m, n)) several threads
        // benchmark concurrently and timing jitter can pick DIFFERENT winners
        // — with a plain `insert` the last writer flips the cache AFTER some
        // callers already dispatched their own winner, so call #1 used `plane`
        // while call #2's cache-hit returned `tiled`: a 1-ULP d1≠d2 drift that
        // broke `task_direction_is_deterministic`. `get_or_insert` is atomic
        // insert-if-absent: every racer dispatches the value now in the cache,
        // which never changes once set — per-process variant determinism.
        let settled = *GLOBAL_CACHE.pin().get_or_insert(key, best);
        if settled != best {
            println!(
                "[gemv_autotune] ({m}×{n}): local bench said {best}, parallel racer settled {settled} — using {settled}"
            );
        } else {
            println!("[gemv_autotune] ({m}×{n}): selected {best} variant");
        }

        settled
    }

    /// Benchmark plane vs tiled variants and return the faster one.
    ///
    /// Creates synthetic weight/input data, runs each variant N times,
    /// uses median duration for selection.
    fn benchmark_variants<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        m: usize,
        n: usize,
    ) -> GemvVariant {
        // Cap benchmark rows to fit within a safe GPU allocation budget. Both
        // GEMV kernels are row-parallel — the per-row cost (and thus the
        // plane-vs-tiled ordering) depends on n (reduction length), not m (row
        // count). Capping m while keeping n preserves variant ordering and
        // avoids wgpu OOM on large weight matrices (e.g., lm_head 256000×2304
        // ≈ 2.36 GB f32). See Issue 429 T5.
        let budget_bytes = bench_budget_bytes();
        let m_bench = cap_m(m, n, budget_bytes);
        let total_f32 = m_bench * n;
        if m_bench < m {
            println!(
                "[gemv_autotune] ({m}×{n}): capping benchmark rows {m}→{m_bench} ({budget_bytes}-byte budget, {total_f32} elems)"
            );
        }
        println!(
            "[gemv_autotune] ({m}×{n}): benchmarking plane vs tiled ({m_bench}×{n}, {total_f32} elements)..."
        );

        // Create synthetic test data — small non-zero values to exercise real arithmetic.
        let weight_data = vec![0.01f32; total_f32];
        let input_data = vec![0.5f32; n];
        let weight_handle = client.create_from_slice(f32::as_bytes(&weight_data));
        let input_handle = client.create_from_slice(f32::as_bytes(&input_data));

        // Warmup: JIT compile kernels and prime GPU caches.
        for _ in 0..self.warmup_iters {
            let out = client.empty(m_bench * core::mem::size_of::<f32>());
            // SAFETY: Handles have correct sizes for (m_bench, n).
            unsafe {
                GemvCubeCL::launch_plane::<R>(
                    client,
                    weight_handle.clone(),
                    input_handle.clone(),
                    out.clone(),
                    m_bench,
                    n,
                );
            }
            let _ = client.read_one(out);

            let out = client.empty(m_bench * core::mem::size_of::<f32>());
            unsafe {
                GemvCubeCL::launch_tiled::<R>(
                    client,
                    weight_handle.clone(),
                    input_handle.clone(),
                    out.clone(),
                    m_bench,
                    n,
                );
            }
            let _ = client.read_one(out);
        }

        // Time plane variant.
        let mut plane_durations = Vec::with_capacity(self.bench_iters);
        for _ in 0..self.bench_iters {
            let out = client.empty(m_bench * core::mem::size_of::<f32>());
            let start = Instant::now();
            // SAFETY: Handles have correct sizes for (m_bench, n).
            unsafe {
                GemvCubeCL::launch_plane::<R>(
                    client,
                    weight_handle.clone(),
                    input_handle.clone(),
                    out.clone(),
                    m_bench,
                    n,
                );
            }
            let _ = client.read_one(out);
            plane_durations.push(start.elapsed());
        }

        // Time tiled variant.
        let mut tiled_durations = Vec::with_capacity(self.bench_iters);
        for _ in 0..self.bench_iters {
            let out = client.empty(m_bench * core::mem::size_of::<f32>());
            let start = Instant::now();
            unsafe {
                GemvCubeCL::launch_tiled::<R>(
                    client,
                    weight_handle.clone(),
                    input_handle.clone(),
                    out.clone(),
                    m_bench,
                    n,
                );
            }
            let _ = client.read_one(out);
            tiled_durations.push(start.elapsed());
        }

        let plane_median = median_duration(&mut plane_durations);
        let tiled_median = median_duration(&mut tiled_durations);

        let best = if plane_median <= tiled_median {
            GemvVariant::Plane
        } else {
            GemvVariant::Tiled
        };

        println!(
            "[gemv_autotune] ({m}×{n}): plane={plane_median:?} tiled={tiled_median:?} → {best}"
        );

        best
    }

    /// Launch GEMV using the autotuned variant for this (m, n).
    ///
    /// Delegates to `GemvCubeCL::launch_plane` or `launch_tiled` based on
    /// the cached benchmark result.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: m × n f32 elements
    /// - `input_handle`: n f32 elements
    /// - `output_handle`: m f32 elements
    pub unsafe fn launch<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let variant = self.select::<R>(client, m, n);
        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            match variant {
                GemvVariant::Plane => {
                    GemvCubeCL::launch_plane::<R>(
                        client,
                        weight_handle,
                        input_handle,
                        output_handle,
                        m,
                        n,
                    );
                }
                GemvVariant::Tiled => {
                    GemvCubeCL::launch_tiled::<R>(
                        client,
                        weight_handle,
                        input_handle,
                        output_handle,
                        m,
                        n,
                    );
                }
            }
        }
    }

    /// Returns the number of cached dimension pairs (process-wide).
    #[inline]
    pub fn cache_size(&self) -> usize {
        GLOBAL_CACHE.len()
    }

    /// Clear the process-global autotune cache (useful for re-benchmarking).
    pub fn clear(&self) {
        GLOBAL_CACHE.pin().clear();
    }
}

impl Default for GemvAutotune {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cap the number of output rows for the benchmark to fit within a byte budget.
///
/// Both GEMV kernels are row-parallel — the per-row cost (and thus the
/// plane-vs-tiled ordering) depends on `n` (reduction length), not `m` (row
/// count). Capping `m` while keeping `n` preserves the variant ordering while
/// avoiding GPU OOM on large weight matrices (e.g., lm_head 256000×2304 ≈ 2.36 GB).
/// Returns `m` unchanged if it already fits within the budget.
///
/// See Issue 429 T5.
fn cap_m(m: usize, n: usize, budget_bytes: usize) -> usize {
    if n == 0 || m == 0 {
        return m;
    }
    let elem_size = core::mem::size_of::<f32>();
    let max_elems = budget_bytes / elem_size;
    let m_cap = max_elems / n;
    if m <= m_cap { m } else { m_cap.max(1) }
}

/// Budget for benchmark weight matrix allocation.
///
/// Default 256 MB (well within any modern GPU's buffer limits, large enough to
/// saturate GPU parallelism). Override via `GEMV_BENCH_BUDGET_MB` env var for
/// debugging on constrained devices.
fn bench_budget_bytes() -> usize {
    const DEFAULT_BYTES: usize = 256 * 1024 * 1024;
    match std::env::var("GEMV_BENCH_BUDGET_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(mb) => mb.saturating_mul(1024 * 1024),
        None => DEFAULT_BYTES,
    }
}

/// Compute median duration from a sorted list.
fn median_duration(times: &mut [std::time::Duration]) -> std::time::Duration {
    times.sort();
    let mid = times.len() / 2;
    if times.len().is_multiple_of(2) {
        (times[mid - 1] + times[mid]) / 2
    } else {
        times[mid]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_variant_display() {
        assert_eq!(format!("{}", GemvVariant::Plane), "plane");
        assert_eq!(format!("{}", GemvVariant::Tiled), "tiled");
    }

    #[test]
    fn test_median_duration_odd() {
        let mut times = vec![
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(30),
        ];
        assert_eq!(
            median_duration(&mut times),
            std::time::Duration::from_millis(20)
        );
    }

    #[test]
    fn test_median_duration_even() {
        let mut times = vec![
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(30),
            std::time::Duration::from_millis(40),
        ];
        assert_eq!(
            median_duration(&mut times),
            std::time::Duration::from_millis(25)
        );
    }

    #[test]
    fn test_autotune_default() {
        // Order-dependent otherwise: the cache is PROCESS-GLOBAL, so tests that
        // ran earlier in this binary (serialized alphabetical order — e.g. the
        // gemma2 forwards autotuning their GEMV shapes) legitimately populate
        // it before this test is reached. Clear first; the invariant under
        // test is "a default autotuner consults an empty cache", not "no
        // earlier test ever autotuned".
        let at = GemvAutotune::default();
        at.clear();
        assert_eq!(at.cache_size(), 0);
    }

    // --- cap_m tests (Issue 429 T5: wgpu OOM during GEMV autotune) ---

    #[test]
    fn test_cap_m_small_matrix_unchanged() {
        // In-layer GEMVs (2048×2304, 9216×2304, etc.) fit within 256 MB → no cap.
        let budget = 256 * 1024 * 1024;
        assert_eq!(cap_m(2048, 2304, budget), 2048);
        assert_eq!(cap_m(9216, 2304, budget), 9216);
        assert_eq!(cap_m(256000, 4, budget), 256000); // tiny n → fits easily
    }

    #[test]
    fn test_cap_m_lm_head_capped() {
        // lm_head 256000×2304 = 589,824,000 f32 = 2.36 GB → must be capped.
        let budget = 256 * 1024 * 1024; // 256 MB
        let m_capped = cap_m(256000, 2304, budget);
        // 256 MB / 4 bytes / 2304 = 29,127 rows (floor)
        assert_eq!(m_capped, 29_127);
        // The capped matrix must fit within the budget
        assert!(m_capped * 2304 * 4 <= budget);
        assert!(m_capped > 0);
        // And the original (uncapped) would exceed it
        assert!(256000 * 2304 * 4 > budget);
    }

    #[test]
    fn test_cap_m_zero_n_returns_m() {
        assert_eq!(cap_m(1000, 0, 1024), 1000);
    }

    #[test]
    fn test_cap_m_zero_m_returns_zero() {
        assert_eq!(cap_m(0, 2304, 1024), 0);
    }

    #[test]
    fn test_cap_m_guarantees_at_least_one_row() {
        // Even if n alone exceeds the budget, ensure at least 1 row.
        let budget = 1024; // tiny budget
        let m_capped = cap_m(1000, 999_999, budget);
        assert_eq!(m_capped, 1);
    }

    #[test]
    fn test_cap_m_exact_fit() {
        // m * n * 4 exactly equals budget → should NOT cap (m <= m_cap).
        let budget = 256 * 1024 * 1024;
        let max_elems = budget / 4;
        let n = 1024;
        let m = max_elems / n; // exact fit
        assert_eq!(cap_m(m, n, budget), m);
    }
}
