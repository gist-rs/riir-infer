//! GPU context holding wgpu device, queue, and optional CubeCL integration.
//!
//! When the `cubecl_runtime` feature is enabled, the context initializes CubeCL's
//! compute server on the same wgpu device/queue via `init_device()` (Plan 106 T2.6).
//! Both WGSL and CubeCL kernels submit to the same `wgpu::Queue`, preserving
//! execution order without separate GPU contexts.

use std::sync::Arc;

use wgpu::{AdapterInfo, Device, Limits, Queue};
// Issue 892 T3: the adapter/device negotiation vocabulary is used only by the
// native-only `new_async`, so it must be gated with it — otherwise the wasm32
// `browser` lane reds on `unused_imports` under `-D warnings`.
#[cfg(not(target_arch = "wasm32"))]
use wgpu::{
    DeviceDescriptor, ExperimentalFeatures, Features, PowerPreference, RequestAdapterOptions,
};

/// GPU context holding device and queue.
///
/// Stores the wgpu device and queue, plus optional CubeCL integration
/// for JIT-compiled GPU kernels (Plan 106 T2.6).
///
/// When the `cubecl_runtime` feature is enabled, the context also initializes
/// a CubeCL compute server sharing the same wgpu device/queue via `init_device()`.
/// This enables interop between WGSL and CubeCL kernels on the same GPU — both
/// command streams submit to the same `wgpu::Queue`, preserving submission order.
///
/// # Buffer interop
///
/// CubeCL manages its own `wgpu::Buffer` pool internally. WGSL-managed buffers
/// and CubeCL-managed buffers coexist on the same device. Use `copy_buffer_to_buffer`
/// on the wgpu encoder to transfer data between the two pools at compute-pass
/// boundaries.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter_info: AdapterInfo,
    pub limits: Limits,
    /// Whether the device supports subgroup operations (subgroupAdd, etc.).
    /// On Apple M-series Metal, this enables simd_sum() for cooperative reductions.
    pub has_subgroups: bool,
    /// Number of token positions per command-encoder submission in the chunked
    /// forward/backward passes (see `GpuForwardPass::forward`, `GpuBackwardPass::backward_pass_impl`).
    ///
    /// Metal's driver stalls (or appears to deadlock) when too many compute passes
    /// are accumulated in a single encoder. The original empirical safe value was 2
    /// (≈48 compute passes per encoder at ~24 passes per position). Larger values
    /// reduce per-step `queue.submit` count — the dominant cost for tiny models
    /// trained with per-sample loops (Issue 016: 29 sec/step → target <2 sec/step).
    ///
    /// Tunable via `RIIR_GPU_CHUNK_SIZE` env var so users can experiment without
    /// recompiling. Defaults to 2 (the historical safe value). Override examples:
    /// - `RIIR_GPU_CHUNK_SIZE=8` — 4× fewer submits, ~4× faster step on Metal
    /// - `RIIR_GPU_CHUNK_SIZE=16` — 8× fewer submits; verify no deadlock on your GPU
    ///
    /// If a larger value triggers a Metal driver stall, fall back to a smaller one.
    /// The cost of overshoot is a hung training loop, not data corruption.
    pub chunk_size: usize,
    /// CubeCL device handle.
    /// - Default: shares the same wgpu device/queue via `cubecl::wgpu::init_device()`.
    /// - `cuda_backend` feature: independent `CudaDevice` (native CUDA path,
    ///   bypasses WDDM on Windows — Issue 442).
    ///
    /// Use `cubecl_client()` to obtain a `ComputeClient` for kernel launches.
    #[cfg(feature = "cubecl_runtime")]
    cubecl_device: crate::cubecl_runtime::ActiveDevice,
}

// CubeCL Runtime trait needed for WgpuRuntime::client().
#[cfg(feature = "cubecl_runtime")]
use cubecl::Runtime;

/// Process-wide serialization for GPU context INITIALIZATION (Issue 719).
///
/// Shared by [`GpuContext::new`] and [`CubeCLContext::new`](crate::cubecl_runtime::CubeCLContext::new).
/// Both init paths run the same driver-level sequence (wgpu
/// Instance/Adapter/Device creation + CubeCL server registration), and
/// CONCURRENT execution of that sequence from multiple threads deadlocks
/// non-deterministically on the 4090/Windows/Vulkan box — measured 4/7
/// threads wedged forever inside `adapter.request_device()` (another run:
/// 1/7 wedged inside `init_device`). The lib suite under default
/// parallelism froze at 101/604 tests exactly this way. Callers must NOT
/// re-enter any context constructor while holding this guard (none of the
/// init paths call back into them — verified by reading both).
///
/// Native only (Issue 892 T3): its sole callers are `GpuContext::new()` and
/// the `cubecl_runtime` server init, both of which are native-only.
#[cfg(not(target_arch = "wasm32"))]
static GPU_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the process-wide GPU init serialization lock (Issue 719). Poison
/// is de-fanged: a panicking init leaves no partial registration worth
/// blocking later callers over (the OnceLocks stay unset; the next caller
/// retries fresh).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn gpu_init_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

impl GpuContext {
    /// Create a new GPU context, selecting the best available adapter.
    /// Uses `pollster::block_on` for synchronous initialization.
    ///
    /// **Native only (Issue 892 T3).** In a browser `wgpu::Device` wraps
    /// `Rc<Cell<u32>>` and is therefore `!Send + !Sync`, so neither the
    /// process-global `OnceLock<GpuContext>` below nor `pollster::block_on`
    /// can exist there — WebGPU adapter/device requests are JS Promises and
    /// the page has one thread. `browser_gpu_init::new_gpu_context()` is the
    /// wasm32 constructor and its own docs already said so; the cfg was
    /// missing only because nothing in this repo compiled wasm32 (Issue 892).
    ///
    /// When `cubecl_runtime` feature is enabled, also initializes a CubeCL
    /// compute server on the same device for JIT-compiled kernels.
    ///
    /// **Issue 714: process-global cache.** Every call previously created a
    /// NEW wgpu Instance/Adapter/Device/Queue AND registered a NEW CubeCL
    /// server via `init_device` — and CubeCL has no Drop deregistration, so
    /// the server (and its wgpu Device Arc, and its memory pools) stays
    /// registered for the process lifetime. A 590-test suite created **46
    /// live Vulkan devices**, each pinning its test's pool high-water mark
    /// (~20 GB of un-reusable residue on the 24 GB 4090 → the full-suite
    /// OOM cascade). One shared context restores the Issue-676 sharing
    /// semantics for the wgpu-orchestration path: one device, one server,
    /// one pool whose freed slices are REUSED by later tests instead of
    /// stranded on a dead-but-alive device. Initialization failure is not
    /// cached — a later call retries fresh (an adapter may appear mid-run).
    ///
    /// **Issue 719: init is serialized under [`gpu_init_lock`].** The previous
    /// "race-benign" pattern (parallel first-callers each build a device
    /// before one wins the slot) is NOT benign on the 4090/Windows/Vulkan
    /// box: concurrent execution of the init sequence deadlocks
    /// non-deterministically — measured with 7 threads racing `new()`:
    /// 4 wedged forever inside `adapter.request_device()`, in another run 1
    /// wedged inside `init_device` (6/7 completed). The full lib suite under
    /// default parallelism froze at 101/604 tests exactly this way (every
    /// worker blocked behind first-call `GpuContext::new()`). The race also
    /// leaked every loser's server + Vulkan device for the process lifetime
    /// (CubeCL has no deregistration) — the partial reintroduction of the
    /// Issue-714 46-device leak. The mutex closes both: one racer runs
    /// `new_async()` alone; the rest fast-path on the OnceLock after the
    /// guard releases. Failure is still not cached (Err propagates with the
    /// lock released; the next caller retries fresh).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, GpuError> {
        static SHARED_GPU_CONTEXT: std::sync::OnceLock<GpuContext> = std::sync::OnceLock::new();
        if let Some(ctx) = SHARED_GPU_CONTEXT.get() {
            return Ok(ctx.clone());
        }
        let _guard = gpu_init_lock();
        // Double-check after the wait: the racer we queued behind may have
        // already initialized (or a later caller may have retried past an Err).
        if let Some(ctx) = SHARED_GPU_CONTEXT.get() {
            return Ok(ctx.clone());
        }
        let ctx = pollster::block_on(Self::new_async())?;
        Ok(SHARED_GPU_CONTEXT.get_or_init(|| ctx).clone())
    }

    /// Async GPU context initialization.
    ///
    /// Creates the wgpu instance, adapter, device, and queue. When CubeCL
    /// is enabled, transfers instance/adapter ownership to CubeCL via
    /// `WgpuSetup` + `init_device()`, while sharing device/queue via clone.
    #[cfg(not(target_arch = "wasm32"))]
    async fn new_async() -> Result<Self, GpuError> {
        // wgpu v29: InstanceDescriptor::default() removed — use new_without_display_handle()
        // for compute-only (no surface/rendering) contexts.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        // wgpu v25: request_adapter returns Result instead of Option
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                // wgpu v30: opt-in coarse limit buckets. We query real limits
                // from the adapter below, so keep the exact values.
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();

        // Use the adapter's actual limits (not downlevel defaults) to support
        // shaders with many storage bindings (e.g., qkv_projection has 7+).
        let adapter_limits = adapter.limits();

        // Issue 711: wgpu-30 gates pipeline-layout immediates (WebGPU push
        // constants — what cubecl-0.11's SPIR-V path uses for kernel scalar
        // params) behind Features::IMMEDIATES. Without it every
        // immediate-carrying kernel panics at create_pipeline_layout on the
        // DSD dispatch thread and the launch is silently dropped (output
        // stays zero — the Issue 711 zeros + kimi backward e-2 divergence on
        // this box). Intersection keeps adapters without the feature on the
        // old behavior.
        //
        // Issue 828: same silent-drop class for the cmma kernels. cubecl's
        // feature probe queries the PHYSICAL adapter (register_cmma), so the
        // MmaConfig list advertises cmma and the SPIR-V emits
        // OpCooperativeMatrix*KHR — but the LOGICAL device only gets
        // VK_KHR_cooperative_matrix when EXPERIMENTAL_COOPERATIVE_MATRIX is
        // requested (wgpu-hal adapter.rs pushes the extension solely on that
        // bit). Without it every cmma pipeline fails creation on the 4090's
        // Vulkan backend and the launch is silently dropped (bench_809's
        // max_abs 4.1e-1 = |oracle|). Experimental features also need the
        // descriptor's experimental_features gate (wgpu-core instance.rs
        // refuses the request otherwise). On Metal the bit is unset (no
        // extension concept; MSL simdgroup matrix needs none), so the
        // intersection is a no-op there. NOTE: requesting the bit only
        // enables the EXTENSION — whether the driver supports the kernel's
        // 8×8×8 f32 config is a separate shape question the 4090 probe
        // settles (see Issue 828).
        let required_features = {
            let base = adapter.features()
                & (Features::SUBGROUP
                    | Features::IMMEDIATES
                    | Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
            #[cfg(feature = "cubecl_runtime")]
            {
                base | (adapter.features() & Features::PASSTHROUGH_SHADERS)
            }
            #[cfg(not(feature = "cubecl_runtime"))]
            {
                base
            }
        };
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("riir-engine gpu device"),
                required_features,
                required_limits: adapter_limits,
                // Issue 828: EXPERIMENTAL_COOPERATIVE_MATRIX is in the
                // experimental mask — wgpu-core refuses the device request
                // without this gate (RequestDeviceError::ExperimentalFeatures
                // NotEnabled). The same gate the vendored cubecl request_device
                // carries for its own path.
                experimental_features: unsafe { ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::DeviceError(e.to_string()))?;

        // riir-train .issues/511 (the sixth probe): device loss was 100%
        // INVISIBLE in this stack — wgpu's error sink DROPS DeviceLost-class
        // errors by design ("surfaced via callback"), no callback was
        // registered, and the only downstream symptom is buffers coming back
        // as silent invalid error objects that die at `map_async` with a
        // generic validation error. That is the entire mechanism behind the
        // dllm-8/gdsd concurrency census rows. Registering the callback makes
        // the loss moment + its message visible ONCE, loudly, the instant it
        // happens — `lose()` fires this with the originating hal error string
        // (e.g. "Out of memory"), which names the real trigger.
        device.set_device_lost_callback(|reason, message| {
            eprintln!(
                "[riir-gpu] DEVICE LOST: reason={reason:?} message={message:?} — every \
                 buffer created after this point is a silent invalid error object"
            );
        });

        let limits = device.limits();
        // Issue 714: pin the negotiated binding limit per process — the
        // full-vocab f32 lm-head GEMV needs ≥2.36 GB in ONE storage bind;
        // runs that passed vs failed this historically differed in ways only
        // this line can distinguish.
        eprintln!(
            "[riir-gpu] device limits: max_storage_buffer_binding_size={}, max_buffer_size={}",
            limits.max_storage_buffer_binding_size, limits.max_buffer_size
        );
        let has_subgroups = device.features().contains(Features::SUBGROUP);

        // Issue 016: chunk_size tunes the per-encoder position count for the
        // chunked forward/backward passes. The historical safe default is 2;
        // users override via RIIR_GPU_CHUNK_SIZE to reduce queue.submit count
        // (the dominant per-step cost for tiny models with per-sample loops).
        let chunk_size = std::env::var("RIIR_GPU_CHUNK_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(2);
        if chunk_size != 2 {
            eprintln!(
                "[riir-gpu] RIIR_GPU_CHUNK_SIZE override: {chunk_size} (default=2). \
                 If Metal stalls/hangs, lower this value."
            );
        }

        // Initialize CubeCL device for JIT kernel compilation.
        //
        // Default path: shares the wgpu Device/Queue via `init_device()` — no
        // separate GPU context. Instance/Adapter ownership transfers to CubeCL's
        // internal WgpuServer. Enables interop between WGSL and CubeCL dispatches.
        //
        // `cuda_backend` path (Issue 442): creates an independent `CudaDevice`.
        // Native CUDA bypasses WDDM GPU clock throttling on Windows (Issue 441 T7).
        // The wgpu device is still created above (for any direct WGSL paths), but
        // CubeCL kernels dispatch through the CUDA runtime — the two don't share
        // buffers, which is fine because the training loop uses CubeCL exclusively.
        #[cfg(all(feature = "cubecl_runtime", any(not(feature = "cuda_backend"), target_os = "macos")))]
        let cubecl_device = {
            let setup = cubecl::wgpu::WgpuSetup {
                instance,
                adapter,
                device: device.clone(),
                queue: queue.clone(),
                backend: adapter_info.backend,
            };
            cubecl::wgpu::init_device(setup, Default::default())
        };

        #[cfg(all(feature = "cubecl_runtime", feature = "cuda_backend", not(target_os = "macos")))]
        let cubecl_device = {
            // Native CUDA path — device 0 is the primary GPU.
            // The CUDA runtime initializes lazily on first client() call.
            cubecl::cuda::CudaDevice::new(0)
        };

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits,
            has_subgroups,
            chunk_size,
            #[cfg(feature = "cubecl_runtime")]
            cubecl_device,
        })
    }

    /// Get a CubeCL compute client for launching JIT-compiled kernels.
    ///
    /// - Default: shares the same wgpu device and queue as this context,
    ///   enabling interop between WGSL and CubeCL dispatches on the same GPU.
    /// - `cuda_backend` feature: returns a native CUDA client (bypasses WDDM).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ctx = GpuContext::new()?;
    /// let client = ctx.cubecl_client();
    /// let handle = client.empty(1024);
    /// // ... launch CubeCL kernels ...
    /// ```
    #[cfg(feature = "cubecl_runtime")]
    pub fn cubecl_client(
        &self,
    ) -> cubecl::prelude::ComputeClient<crate::cubecl_runtime::ActiveRuntime> {
        crate::cubecl_runtime::ActiveRuntime::client(&self.cubecl_device)
    }

    /// Get the CubeCL device handle.
    ///
    /// Useful for advanced CubeCL operations that need the device directly,
    /// such as creating `BufferArg` from raw handles.
    #[cfg(feature = "cubecl_runtime")]
    #[inline]
    pub fn cubecl_device(&self) -> &crate::cubecl_runtime::ActiveDevice {
        &self.cubecl_device
    }
}

#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    DeviceError(String),
    ShaderError(String),
    BufferError(String),
    /// A backward pass bound to one `GpuForwardPass` was handed a different
    /// one (Issue 499): its per-layer bind groups capture the first forward's
    /// buffers and cannot be re-keyed, so a mismatched pairing would silently
    /// write gradients into the wrong instance. Reported instead of computed.
    ForwardMismatch { bound: u64, given: u64 },
}

impl std::fmt::Display for GpuError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "No suitable GPU adapter found"),
            GpuError::DeviceError(msg) => write!(f, "Device error: {msg}"),
            GpuError::ShaderError(msg) => write!(f, "Shader error: {msg}"),
            GpuError::BufferError(msg) => write!(f, "Buffer error: {msg}"),
            GpuError::ForwardMismatch { bound, given } => write!(
                f,
                "Backward pass is bound to forward #{bound} but was handed forward #{given} \
                 — a GpuBackwardPass captures one GpuForwardPass's buffers and cannot be shared"
            ),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_init() {
        match GpuContext::new() {
            Ok(ctx) => {
                println!("GPU adapter: {}", ctx.adapter_info.name);
                assert!(!ctx.adapter_info.name.is_empty());
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU adapter available — skipping GPU tests");
            }
            Err(e) => panic!("Unexpected GPU error: {e}"),
        }
    }

    /// Verify CubeCL initializes on the shared device.
    #[cfg(feature = "cubecl_runtime")]
    #[test]
    fn test_gpu_context_cubecl_shared() {
        let ctx = GpuContext::new().expect("GpuContext should initialize");
        let client = ctx.cubecl_client();
        let name = crate::cubecl_runtime::ActiveRuntime::name(&client);
        println!("CubeCL runtime on shared device: {name}");
        assert!(!name.is_empty(), "runtime name should not be empty");

        // Verify it's a Metal backend on macOS.
        #[cfg(target_os = "macos")]
        assert!(
            name.contains("msl") || name.contains("metal"),
            "Expected Metal backend on macOS, got: {name}"
        );
    }
}
