//! CubeCL GPU kernel runtime integration (Plan 106 T2.2).
//!
//! CubeCL compiles Rust `#[cube]` functions to Metal/WGSL/SPIR-V at runtime (JIT),
//! enabling portable high-performance GPU compute without hand-written WGSL shaders.
//!
//! ## Architecture
//!
//! ```text
//! Rust #[cube] kernel → CubeCL IR → Metal/WGSL/SPIR-V → wgpu → GPU
//! ```
//!
//! ## Device Sharing (T2.6)
//!
//! Currently creates its own wgpu device. T2.6 will share the device with
//! `GpuContext` via `cubecl::wgpu::init_device()` for zero-copy interop.

use std::sync::Arc;
use std::fmt;

/// CubeCL runtime error types.
#[derive(Debug)]
#[allow(dead_code)] // Used in T2.3+ (CubeCL GEMV kernel)
pub enum CubeCLError {
    /// Runtime initialization failed (no adapter, device creation error, etc.)
    InitFailed(String),
    /// Kernel compilation or execution error.
    KernelError(String),
}

impl fmt::Display for CubeCLError {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CubeCLError::InitFailed(msg) => write!(f, "CubeCL init failed: {msg}"),
            CubeCLError::KernelError(msg) => write!(f, "CubeCL kernel error: {msg}"),
        }
    }
}

impl std::error::Error for CubeCLError {}

// ---------------------------------------------------------------------------
// Feature-gated CubeCL imports
// ---------------------------------------------------------------------------

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

/// The CubeCL buffer handle, re-exported so downstream consumers of this
/// crate's launchers can name the type without a direct `cubecl` dep (plan
/// 611 S1b: the laya `CubeclBackend` residency tables key on it).
#[cfg(feature = "cubecl_runtime")]
pub use cubecl::server::Handle;

// wgpu types are needed for the default (non-CUDA) path in CubeCLContext::new()
// and in test code. Under `cuda_backend` on a NON-macOS target the wgpu init
// branch is compiled out; on macOS the wgpu path is ALWAYS active (Issue 949:
// `--all-features` enables `cuda_backend` everywhere, but macOS has no CUDA —
// cudarc's libcuda load panics — so the feature is inert there by cfg).
#[cfg(all(feature = "cubecl_runtime", any(not(feature = "cuda_backend"), target_os = "macos")))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

// ---------------------------------------------------------------------------
// Binding-size guard (Issue 639 T5)
// ---------------------------------------------------------------------------

/// Debug-only guard that a handle actually backs the `len` f32 elements a
/// launcher is about to bind.
///
/// `BufferArg::from_raw_parts(handle, len)` does not validate `len` against the
/// handle, so an **undersized** handle is straightforward out-of-bounds UB that
/// no test catches until it corrupts unrelated memory.
///
/// Only the undersized direction is asserted here, and an exact-equality
/// assert would false-positive on CubeCL's memory-pool bucket rounding, where
/// `client.empty(n)` legitimately returns a larger allocation.
///
/// **This comment used to add "the *oversized* direction ... is now fixed at
/// the source — the affected kernels take explicit dimension scalars". That
/// was true of the kernels Issue 639 looked at and false of the workspace.**
/// The attention family still derives `n_positions` from `kv.len()`, and on
/// 2026-09-05 that produced an identically-zero attention output in both
/// GPU-resident-KV paths (riir-train `.issues/511`, fixed in `3e00c93e0`; the
/// class is `.issues/515`). Use [`assert_binding_derives_units`] at any
/// launcher that binds a buffer to a kernel which derives a dimension from a
/// length — this guard cannot see that failure, because the binding is
/// oversized rather than undersized.
#[cfg(feature = "cubecl_runtime")]
#[inline]
pub fn debug_assert_binding_at_least(handle: &cubecl::server::Handle, len: usize, what: &str) {
    debug_assert!(
        handle.size_in_used() >= (len * core::mem::size_of::<f32>()) as u64,
        "{what}: binding declares {len} f32 ({} bytes) but the handle only backs \
         {} bytes — out-of-bounds read/write",
        len * core::mem::size_of::<f32>(),
        handle.size_in_used(),
    );
}

/// Encode a shape/offset as an f32 params element, refusing to round: f32
/// exactly represents integers up to 2²⁴, and the guard keeps that bound
/// loud instead of letting a huge packed offset or row length round
/// silently (plan 611 S2/S3 — the params-over-views discipline).
#[cfg(feature = "cubecl_runtime")]
#[inline]
pub(crate) fn f32_exact(v: usize) -> f32 {
    assert!(
        v <= (1usize << 24),
        "shape {v} exceeds the f32-exact bound 2^24"
    );
    v as f32
}

/// Assert that a kernel deriving a dimension from `x.len()` will derive the
/// value its launcher intends.
///
/// The sibling guard above asks "is the handle big enough". This one asks the
/// only question that matters for a `.len()`-deriving kernel: **given the
/// buffer we are about to bind, what will the shader compute?** In a `#[cube]`
/// kernel `x.len()` is the size of the BOUND BUFFER —
/// `BufferArg::from_raw_parts`'s `length` never reaches the shader — so a
/// pooled or reused buffer silently changes the kernel's idea of its own
/// shape.
///
/// Asserting the derived UNIT COUNT rather than the byte size is what makes
/// this false-positive-free: pool bucket rounding is only a finding when it
/// changes the quotient, and when it changes the quotient the kernel is
/// genuinely wrong.
///
/// **Not `debug_assert!`.** This repo's own rule is to run GPU gates under
/// `--release`, where a `debug_assert!` compiles to nothing — a guard that is
/// absent from the configuration its subject runs in is not a guard. The cost
/// is a field read, an integer divide and a compare, against a kernel dispatch
/// that is three orders of magnitude more expensive.
#[cfg(feature = "cubecl_runtime")]
#[inline]
pub fn assert_binding_derives_units(
    handle: &cubecl::server::Handle,
    elems_per_unit: usize,
    expected_units: usize,
    what: &str,
) {
    let elems = (handle.size_in_used() / core::mem::size_of::<f32>() as u64) as usize;
    let derived = elems.checked_div(elems_per_unit).unwrap_or(0);
    assert!(
        derived == expected_units,
        "{what}: the kernel derives its own dimension from the bound buffer, and \
         this binding backs {elems} f32 = {derived} unit(s) of {elems_per_unit}, \
         not the {expected_units} the launcher intends. Trim the handle to the \
         live range (`handle.offset_end(bytes)`) or pass the dimension \
         explicitly — see riir-train `.issues/515`."
    );
}

/// The host-read barrier shape the laya `CubeclBackend` uses (plan 611 S1b):
/// read one handle's full content and widen it to f32.
///
/// `ComputeClient::read_one` is synchronous — it drains the stream up to and
/// including every dispatch that produced the handle — so this is the ONE
/// blocking point per forward, exactly the Metal lane's `download_into`
/// sync. Keeping the `Bytes`-to-f32 conversion inside this crate means
/// consumers never name a `cubecl` type to read their results.
///
/// Returns the handle's live range as f32 (offset views read their view,
/// not the parent).
#[cfg(feature = "cubecl_runtime")]
pub fn read_f32<R: Runtime>(
    client: &ComputeClient<R>,
    handle: Handle,
) -> Result<Vec<f32>, cubecl::server::ServerError> {
    let bytes = client.read_one(handle)?;
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// Upload a host f32 slice into a fresh device handle — the create half of
/// [`read_f32`], kept beside it so consumers of this crate's CubeCL surface
/// never name a `cubecl` type for either direction (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub fn create_f32<R: Runtime>(client: &ComputeClient<R>, data: &[f32]) -> Handle {
    client.create_from_slice(f32::as_bytes(data))
}

/// Upload a u32 slice (the index-buffer shape — e.g. the encoder lane's
/// row-gather ids) so the laya backend never names a cubecl type.
pub fn create_u32<R: Runtime>(client: &ComputeClient<R>, data: &[u32]) -> Handle {
    client.create_from_slice(u32::as_bytes(data))
}

// ---------------------------------------------------------------------------
// Active runtime selection (Issue 442)
// ---------------------------------------------------------------------------
//
// The active CubeCL runtime is selected at compile time via the `cuda_backend`
// feature flag. On Windows/NVIDIA (4090), `cuda_backend` switches to native CUDA
// to bypass WDDM GPU clock throttling (Issue 441 T7). On macOS and other
// platforms, the default wgpu path is used (Metal on macOS, DX12/Vulkan elsewhere).
//
// Issue 949: every `cuda_backend` cfg in this crate is ANDed with
// `not(target_os = "macos")` — the feature may still be ENABLED on macOS
// (e.g. by `--workspace --all-features`, the Layer-3 lane), but it is INERT
// there: `ActiveRuntime`/`ActiveDevice` resolve to the wgpu types and every
// CUDA arm compiles away. macOS has no CUDA driver — cudarc's `libcuda`
// load panics at runtime — so a feature-enabled-but-inert posture is the
// only one that keeps `--all-features` green on the M3 box.
//
// Production code should use `ActiveRuntime` / `ActiveDevice` instead of
// `WgpuRuntime` / `WgpuDevice` so the backend is swappable. Test code may use
// either — tests that explicitly target the wgpu path should keep `WgpuRuntime`.

/// The active CubeCL runtime type.
///
/// - Default (`cubecl_runtime`): `WgpuRuntime` (Metal/WGSL/SPIR-V via wgpu)
/// - `cuda_backend` feature on non-macOS: `CudaRuntime` (native CUDA, bypasses
///   WDDM). On macOS the feature is INERT — `ActiveRuntime` stays `WgpuRuntime`
///   (Metal) even when enabled (Issue 949: `--all-features` must not panic on
///   a box with no CUDA driver).
#[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
pub type ActiveRuntime = cubecl::cuda::CudaRuntime;

/// The active CubeCL runtime type (wgpu variant).
#[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
pub type ActiveRuntime = WgpuRuntime;

/// The active CubeCL device type.
///
/// - Default: `WgpuDevice`
/// - `cuda_backend` on non-macOS: `CudaDevice` (inert on macOS, Issue 949)
#[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
pub type ActiveDevice = cubecl::cuda::CudaDevice;

/// The active CubeCL device type (wgpu variant).
#[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
pub type ActiveDevice = WgpuDevice;

/// Compute client parameterized by the active runtime.
pub type ActiveComputeClient = ComputeClient<ActiveRuntime>;

// ---------------------------------------------------------------------------
// CubeCL context
// ---------------------------------------------------------------------------

/// CubeCL GPU compute context.
///
/// Wraps the active CubeCL runtime for JIT-compiled GPU kernels.
/// - Default: wgpu runtime (Metal on Apple Silicon, WGSL/SPIR-V elsewhere).
/// - `cuda_backend` feature: native CUDA runtime (bypasses WDDM on Windows).
///
/// # Example
///
/// ```rust,ignore
/// let ctx = CubeCLContext::new()?;
/// let client = ctx.client();
/// let handle = client.empty(1024);
/// // ... launch kernel ...
/// ```
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct CubeCLContext {
    device: ActiveDevice,
    /// Shared wgpu device — retained from `init_setup` for zero-copy interop
    /// (Issue 657: wgpu MSL passthrough dispatch). `None` on CUDA backend or
    /// when CubeCL manages its own device internally.
    #[cfg(all(feature = "cubecl_runtime", any(not(feature = "cuda_backend"), target_os = "macos")))]
    wgpu_device: Option<Arc<wgpu::Device>>,
    /// Shared wgpu queue — same queue CubeCL submits to (shared via
    /// `init_setup`). Used by Issue 657's wgpu MSL passthrough dispatch.
    #[cfg(all(feature = "cubecl_runtime", any(not(feature = "cuda_backend"), target_os = "macos")))]
    wgpu_queue: Option<Arc<wgpu::Queue>>,
    /// Issue 994: the adapter's total video memory, probed once at init via
    /// the vendored wgpu-hal accessors (DXGI `DedicatedVideoMemory` /
    /// Vulkan device-local heaps / Metal `recommendedMaxWorkingSetSize`).
    /// `None` on the CUDA backend or when no probe matches — the VRAM
    /// pre-flight then skips (the construction flush-gate still protects).
    total_video_memory: Option<u64>,
}

/// Process-wide shared context (Issue 676).
///
/// CubeCL registers the wgpu server at the FIXED `WgpuDevice::DefaultDevice`
/// id — `DeviceHandle::insert` panics with "Service already initialized" on
/// any second registration in the same process. A context was therefore
/// implicitly process-lifetime ever since Issue 657 switched `new` from the
/// lazy `WgpuRuntime::client(&device)` load (which returned the SAME cached
/// server on every call) to the eager `init_setup` registration. Cache the
/// first context and hand out clones — restoring the pre-657 sharing
/// semantics (one server, one JIT kernel cache per process) while keeping
/// Issue 657's retained wgpu Device/Queue. Clones share the same underlying
/// compute client; there is no `Drop` deregistration in CubeCL, so this
/// changes no observable drop behavior.
#[cfg(feature = "cubecl_runtime")]
static SHARED_CUBECL_CONTEXT: std::sync::OnceLock<CubeCLContext> = std::sync::OnceLock::new();

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in T2.3+ (CubeCL GEMV kernel)
impl CubeCLContext {
    /// Create a new CubeCL context using the default GPU device.
    ///
    /// Initializes the active CubeCL runtime with default options.
    /// - Default: selects the wgpu adapter automatically (Metal on macOS).
    /// - `cuda_backend`: creates a `CudaDevice(0)` (primary GPU).
    ///
    /// The underlying server is created once per process; repeated calls
    /// return clones of the shared context (Issue 676).
    ///
    /// Issue 719: init is serialized under the crate-wide [`gpu_init_lock`]
    /// (shared with `GpuContext::new`) — `new_uncached` runs the same
    /// driver-level wgpu device-creation sequence, and concurrent execution
    /// of that sequence from multiple threads deadlocks non-deterministically
    /// on the 4090/Windows/Vulkan box (see `context.rs` for the measurement).
    /// The OnceLock alone serializes same-path callers but not cross-path
    /// races against a concurrent `GpuContext::new()` first call.
    pub fn new() -> Result<Self, CubeCLError> {
        // Fast path: no lock touch once initialized.
        if let Some(ctx) = SHARED_CUBECL_CONTEXT.get() {
            return Ok(ctx.clone());
        }
        let _guard = crate::context::gpu_init_lock();
        Ok(SHARED_CUBECL_CONTEXT
            .get_or_init(Self::new_uncached)
            .clone())
    }

    /// Construct + register a fresh context. Call ONCE per process (via
    /// [`Self::new`]) — a second call panics inside CubeCL's `DeviceHandle::insert`.
    fn new_uncached() -> Self {
        #[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
        {
            use cubecl::wgpu::{init_setup, AutoGraphicsApi};

            // Issue 649: set a high `CUBECL_WGPU_MAX_TASKS` default so the
            // entire decode forward (~835 dispatches) fits into a single
            // command buffer. CubeCL's default is 32 (→ ~26 submissions/token);
            // 1024 gives ~6% throughput gain. Only set if the user hasn't
            // already — env var override always wins.
            if std::env::var("CUBECL_WGPU_MAX_TASKS").is_err() {
                // SAFETY: single-threaded init, before any CubeCL client exists.
                // No other thread can read this env var concurrently.
                unsafe { std::env::set_var("CUBECL_WGPU_MAX_TASKS", "1024"); }
            }

            // Issue 657: use `init_setup` instead of `WgpuRuntime::client` so
            // we retain the wgpu Device + Queue for zero-copy interop. Both
            // paths initialize the ComputeClient for `DefaultDevice` — the
            // difference is `init_setup` returns the `WgpuSetup` with device +
            // queue clones, while `WgpuRuntime::client` discards them.
            let device = WgpuDevice::DefaultDevice;
            let setup = init_setup::<AutoGraphicsApi>(
                &device,
                cubecl::wgpu::RuntimeOptions::default(),
            );
            let client = WgpuRuntime::client(&device);
            let runtime_name = WgpuRuntime::name(&client);
            println!("CubeCL runtime initialized: {runtime_name}");
            // Issue 994: the pool-poison detector must be live before ANY
            // forward can exist — the context is the single construction
            // choke point, and the fast-path clone in `new()` can only run
            // after this uncached path ran once.
            crate::pool_poison::ensure_hooks_installed();
            crate::pool_poison::install_uncaptured_handler(&setup.device);
            apply_optional_memory_config(&client);
            // Issue 994: probe the adapter's video memory once (vendored
            // wgpu-hal accessors). Best-effort — `None` skips the pre-flight
            // and leaves the flush-gate as the backstop.
            let total_video_memory = adapter_total_video_memory(&setup.adapter);
            if total_video_memory.is_none() {
                eprintln!(
                    "[Issue 994] adapter video-memory probe returned None — the VRAM pre-flight \
                     will skip; the construction flush-gate still protects."
                );
            }
            Self {
                device,
                wgpu_device: Some(Arc::new(setup.device)),
                wgpu_queue: Some(Arc::new(setup.queue)),
                total_video_memory,
            }
        }
        #[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
        {
            let device = cubecl::cuda::CudaDevice::new(0);
            let client = cubecl::cuda::CudaRuntime::client(&device);
            let runtime_name = cubecl::cuda::CudaRuntime::name(&client);
            println!("CubeCL runtime initialized: {runtime_name}");
            apply_optional_memory_config(&client);
            // Issue 994: no wgpu adapter on the CUDA backend — the pre-flight
            // skips unless RIIR_GPU_VRAM_BUDGET_BYTES is set; the flush-gate
            // still protects (it is runtime-agnostic).
            Self { device, total_video_memory: None }
        }
    }

    /// Get the shared wgpu device (Issue 657). Returns `None` on CUDA backend.
    /// This is the same device CubeCL uses — sharing it enables zero-copy
    /// buffer interop between CubeCL and direct wgpu dispatch (e.g., the
    /// metal::tensor MSL passthrough path).
    #[cfg(all(feature = "cubecl_runtime", any(not(feature = "cuda_backend"), target_os = "macos")))]
    #[inline]
    pub fn wgpu_device(&self) -> Option<&Arc<wgpu::Device>> {
        self.wgpu_device.as_ref()
    }

    /// Get the shared wgpu queue (Issue 657). Returns `None` on CUDA backend.
    /// This is the same queue CubeCL submits to — dispatching wgpu compute
    /// passes on this queue preserves ordering with CubeCL dispatches.
    #[cfg(all(feature = "cubecl_runtime", any(not(feature = "cuda_backend"), target_os = "macos")))]
    #[inline]
    pub fn wgpu_queue(&self) -> Option<&Arc<wgpu::Queue>> {
        self.wgpu_queue.as_ref()
    }

    /// CUDA-backend twin of [`Self::wgpu_device`]: the CUDA runtime has no
    /// shared wgpu device. The accessor stays callable (returning `None`, as
    /// the doc promises) so metal-gated call sites compile under
    /// `--all-features` on macOS — their `None` arm already handles the
    /// "CUDA backend?" case.
    #[cfg(all(feature = "cubecl_runtime", feature = "cuda_backend", not(target_os = "macos")))]
    #[inline]
    pub fn wgpu_device(&self) -> Option<&Arc<wgpu::Device>> {
        None
    }

    /// CUDA-backend twin of [`Self::wgpu_queue`] — see [`Self::wgpu_device`].
    #[cfg(all(feature = "cubecl_runtime", feature = "cuda_backend", not(target_os = "macos")))]
    #[inline]
    pub fn wgpu_queue(&self) -> Option<&Arc<wgpu::Queue>> {
        None
    }

    /// Get a reference to the underlying CubeCL device.
    #[inline]
    pub fn device(&self) -> &ActiveDevice {
        &self.device
    }

    /// Get a compute client for launching kernels.
    ///
    /// The client is reference-counted. Multiple calls with the same device
    /// return clients backed by the same compute server.
    pub fn client(&self) -> ComputeClient<ActiveRuntime> {
        ActiveRuntime::client(&self.device)
    }

    /// Get the runtime backend name (e.g., `"wgpu<msl>"`, `"cuda"`).
    pub fn runtime_name(&self) -> &'static str {
        ActiveRuntime::name(&self.client())
    }

    /// Issue 994: the adapter's total video memory as probed at init
    /// (or `None` — CUDA backend / no matching probe; set
    /// `RIIR_GPU_VRAM_BUDGET_BYTES` to arm the pre-flight there).
    #[inline]
    pub fn total_video_memory(&self) -> Option<u64> {
        self.total_video_memory
    }
}

// ---------------------------------------------------------------------------
// Optional memory-config overrides (Issue 604 G2 crash workaround)
// ---------------------------------------------------------------------------
//
// CubeCL's dynamic memory pool has two distinct failure modes on the 4090
// (both tracked under Issue 614):
//
//   1. `server.rs:124` 1 GiB allocation failure in the dynamic pool, even
//      with 23+ GiB of free VRAM. Tracked upstream as #1456/#1468/#1417;
//      all three are fixed in cubecl 0.11.0-pre.2 (2026-08-10), which this
//      crate now uses (Issue 614 T3). **Verified fixed on the 4090** (Issue 614
//      T6, 2026-08-12): both `cuda_backend` and `wgpu<spirv>` paths complete
//      without `CUBECL_PERSISTENT_MODE`. The workaround env var is now
//      COUNTERPRODUCTIVE on 0.11 + WDDM (forces persistent pool → harder OOM
//      crash). `apply_optional_memory_config` below keeps it as an escape
//      hatch but warns when set.
//      ⚠ **COUNTER-EXAMPLE 2026-08-25 — mode 1 is NOT fully retired on
//      `wgpu<spirv>`.** riir-train Plan 343 T1.7's seam check aborted with
//      `memory allocation of 1073741824 bytes failed` on this 4090, immediately
//      after `CubeCL runtime initialized: wgpu<spirv>` and with **26.7 of
//      33.3 GB host RAM free** — so not exhaustion. 1 GiB / 4 B = 2^28 exactly,
//      i.e. a power-of-two POOL PAGE rather than any semantic buffer.
//      Isolated: the identical binary/model/corpus **without** the
//      `speculative_decode` feature exits 0. That feature adds
//      `SPEC_MAX_K` = 8 extra vocab-sized handles at CONSTRUCTION
//      (`ternary_deltanet_gpu_forward.rs:1694`, ~8 MB total at the 248,320
//      Bonsai vocab — note the doc there still says "~4 MB at 131k vocab"),
//      and those 8 allocations are apparently enough to push the dynamic pool
//      into requesting a new page. So T6's "verified fixed" holds for the
//      paths T6 exercised but does not generalize to this allocation pattern.
//      Untried escape hatches for the next run, in order: the
//      `CUBECL_MEMORY_CLEANUP_INTERVAL` lever below, then
//      `CUBECL_PERSISTENT_MODE=1` (expected to be counterproductive per T6,
//      but it is the documented hatch and this case is outside T6's evidence).
//   2. (Issue 604 / tracel-ai/cubecl#1359) drop_queue arithmetic overflow
//      on >4 GiB cumulative allocations. The 0.11 upgrade does NOT retire this:
//      `drop_queue/policy.rs` in 0.11.0-pre.2 is byte-identical to the 0.10.0
//      file (verified by diff — Issue 614 T4), so the fix is still ours. It is
//      the vendored cubecl-runtime patch at
//      `vendor/cubecl-runtime-0.11.0-pre.2/`, wired via
//      `[patch.crates-io]` in the workspace root `Cargo.toml`. That wiring is a
//      single line and went missing once already (Issue 614 T5): if Cargo.lock
//      shows a `source = "registry+..."` line under `[[package]] cubecl-runtime`,
//      the patch is NOT applied and this comment is lying. Applies to Metal as
//      well as CUDA — cubecl-runtime is backend-agnostic.
//
// Two opt-in env vars provide a workaround without upgrading:
//
//   CUBECL_PERSISTENT_MODE=1
//       Forces `MemoryAllocationMode::Persistent` — all allocations go
//       through the persistent pool, avoiding dynamic-pool fragmentation.
//       May increase peak VRAM usage (no recycling) but the forward pass
//       reuses pre-allocated handles, so the steady-state footprint is
//       bounded by the model size (~7.2 GB for Ternary-Bonsai-27B).
//
//   CUBECL_MEMORY_CLEANUP_INTERVAL=N
//       Calls `client.memory_cleanup()` every N tokens from the forward path.
//       Default 0 (disabled). A value of 10-50 releases reclaimable pool
//       slices periodically, preventing fragmentation from accumulating.
//       Read by `TernaryDeltanetGpuForward` on each `forward_token` call.
//
// Both are no-ops when unset. Production code path is unchanged.

/// Apply optional CubeCL memory configuration based on environment variables.
///
/// `CUBECL_PERSISTENT_MODE=1` forces `MemoryAllocationMode::Persistent`.
///
/// **Historical context (Issue 614):** this was a required workaround on
/// cubecl 0.10 + Windows WDDM to avoid the `server.rs:124` 1 GiB dynamic-pool
/// allocation abort. On cubecl 0.11.0-pre.2 (since Issue 614 T3), the original
/// bug is FIXED and this env var is now COUNTERPRODUCTIVE on WDDM — it forces
/// the persistent pool, which causes a HARDER crash (wgpu OOM + validation
/// error) than the default dynamic mode (which recovers). **Do not set on 0.11+**
/// (Issue 614 T6, verified 2026-08-12). Kept as an escape hatch for cubecl 0.10
/// fallback or non-WDDM environments where it might still help.
#[cfg(feature = "cubecl_runtime")]
fn apply_optional_memory_config(client: &ComputeClient<ActiveRuntime>) {
    if std::env::var("CUBECL_PERSISTENT_MODE")
        .ok()
        .filter(|v| !v.is_empty() && v != "0")
        .is_some()
    {
        // SAFETY: called once at init, before any kernel launches. The
        // "not thread safe" caveat applies to concurrent calls from
        // multiple threads — this is single-threaded init code.
        unsafe {
            client.allocation_mode(cubecl::MemoryAllocationMode::Persistent);
        }
        println!("CubeCL memory: Persistent mode (CUBECL_PERSISTENT_MODE set)");
        eprintln!("WARNING: CUBECL_PERSISTENT_MODE is counterproductive on cubecl 0.11+ WDDM (Issue 614 T6).");
    }
}

// ---------------------------------------------------------------------------
// GELU verification kernel (test-only)
// ---------------------------------------------------------------------------

/// GELU activation — element-wise kernel for integration testing.
///
/// Computes `GELU(x) = x * Φ(x)` where `Φ` is the Gaussian CDF.
/// Verifies the full CubeCL → Metal/WGSL compilation pipeline.
/// Uses concrete `f32` (not generic `F: Float`) to avoid type inference
/// issues in the `#[cube(launch_unchecked)]` macro expansion on v0.10.0.
#[cfg(all(feature = "cubecl_runtime", test))]
#[cube(launch_unchecked)]
fn gelu_verify(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar_verify(input[ABSOLUTE_POS]);
    }
}

/// Scalar GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
#[cfg(all(feature = "cubecl_runtime", test))]
#[allow(unstable_name_collisions)] // CubeCL's f32::erf() may collide with future std method
#[cube]
fn gelu_scalar_verify(x: f32) -> f32 {
    let sqrt2 = f32::new(comptime!(2.0f32.sqrt()));
    let tmp = x / sqrt2;
    x * (f32::erf(tmp) + f32::new(1.0f32)) / f32::new(2.0f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use cubecl::prelude::*;

    use super::{ActiveRuntime, CubeCLContext};

    /// Verify CubeCL initializes and creates a Metal device context.
    #[test]
    fn test_cubecl_init() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let name = ctx.runtime_name();
        println!("CubeCL runtime: {name}");
        assert!(!name.is_empty(), "runtime name should not be empty");
    }

    /// Issue 676 regression: a SECOND `CubeCLContext::new()` in the same
    /// process must not panic with "Service already initialized" (the
    /// fixed-id `init_setup` registration bug every lib-test suite hit).
    /// Both contexts must yield usable, identical compute clients.
    #[test]
    fn test_second_context_does_not_panic() {
        let a = CubeCLContext::new().expect("first context should initialize");
        let b = CubeCLContext::new().expect("second context should share the process-wide server");
        // Clones share one registration → same device id → same client handle.
        assert_eq!(a.device(), b.device());
    }

    /// Verify CubeCL GELU kernel compiles and produces correct results.
    ///
    /// Expected:
    /// - GELU(-1) ≈ -0.1587
    /// - GELU(0) = 0
    /// - GELU(1) ≈ 0.8413
    /// - GELU(5) ≈ 5.0
    #[test]
    fn test_cubecl_gelu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: &[f32] = &[-1.0, 0.0, 1.0, 5.0];
        let n = input.len();

        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(std::mem::size_of_val(input));

        unsafe {
            super::gelu_verify::launch_unchecked::<ActiveRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(n as u32),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle.clone(), n),
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert!(
            (output[0] - (-0.1587)).abs() < 0.01,
            "GELU(-1) expected ~-0.1587, got {}",
            output[0]
        );
        assert!(
            output[1].abs() < 0.01,
            "GELU(0) expected ~0.0, got {}",
            output[1]
        );
        assert!(
            (output[2] - 0.8413).abs() < 0.01,
            "GELU(1) expected ~0.8413, got {}",
            output[2]
        );
        assert!(
            (output[3] - 5.0).abs() < 0.01,
            "GELU(5) expected ~5.0, got {}",
            output[3]
        );

        println!("GELU results: {output:?}");
    }

    /// Verify buffer allocation and read-back roundtrip.
    #[test]
    fn test_cubecl_buffer_roundtrip() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let n = data.len();

        let handle = client.create_from_slice(f32::as_bytes(data));
        let bytes = client.read_one(handle).expect("should read buffer");
        let result = f32::from_bytes(&bytes);

        assert_eq!(result.len(), n);
        for (i, (&expected, &got)) in data.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - got).abs() < 1e-6,
                "element {i}: expected {expected}, got {got}"
            );
        }

        println!("Buffer roundtrip OK: {result:?}");
    }
}

/// Probe the adapter's total video memory via the vendored wgpu-hal
/// accessors (the workspace vendor patch).
///
/// Probes every backend compiled for this target (mirroring wgpu's default
/// backend set per platform); `as_hal` returns `None` for the backends the
/// adapter is not on, so the first `Some` wins.
pub fn adapter_total_video_memory(adapter: &wgpu::Adapter) -> Option<u64> {
    // SAFETY: read-only descriptor queries on the hal adapter; the returned
    // value is used immediately and no hal object escapes this function.
    unsafe {
        #[cfg(target_os = "windows")]
        {
            if let Some(bytes) = adapter
                .as_hal::<wgpu::hal::dx12::Api>()
                .and_then(|a| a.total_video_memory_bytes())
            {
                return Some(bytes);
            }
            if let Some(bytes) = adapter
                .as_hal::<wgpu::hal::vulkan::Api>()
                .and_then(|a| a.total_video_memory_bytes())
            {
                return Some(bytes);
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(bytes) = adapter
                .as_hal::<wgpu::hal::metal::Api>()
                .and_then(|a| a.total_video_memory_bytes())
            {
                return Some(bytes);
            }
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            if let Some(bytes) = adapter
                .as_hal::<wgpu::hal::vulkan::Api>()
                .and_then(|a| a.total_video_memory_bytes())
            {
                return Some(bytes);
            }
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", unix)))]
        {
            let _ = adapter;
        }
        None
    }
}
