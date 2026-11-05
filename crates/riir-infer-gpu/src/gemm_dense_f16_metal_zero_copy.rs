//! Issue 1150 follow-up — Zero-copy raw-Metal **dense** GEMM (f16 AND Q4_K
//! weights) via the shared CubeCL device.
//!
//! The dense twin of [`crate::gemm_ternary_metal_zero_copy`]: the SAME
//! zero-copy raw-Metal dispatch choreography (shared device + queue via
//! `wgpu::Device::as_hal`, per-dispatch raw handle extraction, dedicated
//! output buffer + GPU→GPU copy-back, `ensure_wgpu_initialized` first-use
//! cure). Two compiled pipelines on one context:
//!
//! - `pipeline` — the matmul2d kernel whose A operand is the f16 weight
//!   matrix read STRAIGHT from device memory (no staging, no dequant).
//! - `q4k_pipeline` — plan-614 Phase-3 task 1: the staged-dequant matmul2d
//!   whose A operand is Q4_K blocks dequanted in-kernel to the threadgroup
//!   half tile (the ternary kernel's skeleton; ~4.5 bpw served in place —
//!   the f16 posture's 3.5× memory cost and its S-ratio poison avoided).
//!
//! # Why dense f16
//!
//! The Issue-1150 session record measured the dense-qwen3.8 Metal prefill
//! wall: every in-crate CubeCL tiled GEMM (q4k-tile / f32-tile / f16b-twin)
//! sits at ~110 GFLOP/s — the scalar-kernel wall the Bonsai league's 0.249×
//! Metal prefill row shares. The matmul2d cooperative-tensor path is the
//! measured unlock (the ternary lane's 3.72× kernel-level speedup, Bench
//! 645): dense weights pre-converted to f16 at load are the representation
//! that path consumes with zero per-dispatch conversion work.
//!
//! The cost is disclosed, not hidden: f16 is 16 bpw vs Q4_K's ~4.5 — the
//! F16 load posture trades ~3.5× weight memory (~54 GB for the 27B — M5
//! Ultra class, not the M3) and a bandwidth-bound decode for a tensor-core
//! prefill. The posture was opt-in (`RIIR_QWEN38_DENSE_MM2D=1`) until
//! issue 1176 (2026-11-05) made the league posture the METAL default
//! (demote-only: the exact literal `RIIR_QWEN38_DENSE_MM2D=0` restores the
//! incumbent packed-load route); the ~100× bare-incumbent route measured
//! in bench 944 is the trap the promotion retires. Both postures are
//! benched. The Q4_K pipeline (Bench 1162/1163:
//! the flip gate's S ratio needs the packed bytes; the dequant's ~3.4×
//! arithmetic intensity is the measured latency-hiding lever) is the
//! serving-byte alternative — verify-shape first, decode GEMV follow-up.
//!
//! # Feature gate
//!
//! `metal_tensor_gemm` + macOS-only — the same gate as the ternary module
//! (the vendored wgpu-hal fork exposing `raw_handle()`).

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use metal::foreign_types::ForeignTypeRef;
use metal::{
    Buffer, BufferRef, CommandQueueRef, CompileOptions, ComputePipelineState, DeviceRef,
    MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use wgpu::{Device as WgpuDevice, Queue as WgpuQueue};

use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use crate::cubecl_runtime::ActiveRuntime;
use crate::gemm_ternary_metal_zero_copy::{covered, insert_merged};

/// The Q4_K MSL source — the staged-dequant matmul2d (the ternary skeleton,
/// Q4_K nibble dequant in the staging phase).
const Q4K_MSL_SOURCE: &str = include_str!("gemm_q4k_metal_tensor.metal");

/// The dense-f16 MSL source — the matmul2d kernel with the A operand as a
/// strided device view over the f16 weights (no staging, no dequant).
const MSL_SOURCE: &str = include_str!("gemm_dense_f16_metal_tensor.metal");

/// The process-shared dispatch context — the ANE zero-copy precedent: one
/// install per process at forward load (the armed env), read by every F16
/// dispatch site without threading a parameter through the forward. (No
/// separate weight-cache struct: a dense weight is ONE CubeCL handle + its
/// shape — `QuantWeight::F16` stores exactly that; the ternary module needs
/// a cache because a ternary weight is three handles.)
static DENSE_MM: std::sync::OnceLock<Option<Arc<MetalDenseF16ZeroCopyGemm>>> =
    std::sync::OnceLock::new();

/// The half-B context slot (issue 1170): the SAME struct, its GEMM pipeline
/// compiled from the half×half variant source. Installed only when the
/// `RIIR_QWEN38_DENSE_MM2D_F16ACT=1` env is armed at forward load (the
/// promote-only posture); the dispatch sites select it via
/// [`gemm_mm_for_posture`] and the route feeds f16 activation twins.
static HALF_B_MM: std::sync::OnceLock<Option<Arc<MetalDenseF16ZeroCopyGemm>>> =
    std::sync::OnceLock::new();

/// Install the process-shared dense-f16 context. Called once at forward
/// load when the `RIIR_QWEN38_DENSE_MM2D=1` arm is set; a second install is
/// an idempotent no-op returning the first context (the OnceLock law).
pub fn install_dense_mm(
    gemm: MetalDenseF16ZeroCopyGemm,
) -> &'static Arc<MetalDenseF16ZeroCopyGemm> {
    DENSE_MM
        .get_or_init(|| Some(Arc::new(gemm)))
        .as_ref()
        .expect("dense-f16 context already installed")
}

/// The installed context, if the arm is armed. `None` = the incumbent
/// posture (the F16 variant is never constructed without an install).
pub fn dense_mm() -> Option<&'static Arc<MetalDenseF16ZeroCopyGemm>> {
    DENSE_MM.get().and_then(|o| o.as_ref())
}

/// Install the process-shared half-B context (issue 1170). Idempotent like
/// [`install_dense_mm`].
pub fn install_half_b_mm(
    gemm: MetalDenseF16ZeroCopyGemm,
) -> &'static Arc<MetalDenseF16ZeroCopyGemm> {
    HALF_B_MM
        .get_or_init(|| Some(Arc::new(gemm)))
        .as_ref()
        .expect("half-B context already installed")
}

/// The installed half-B context, if the F16ACT env armed it.
pub fn half_b_mm() -> Option<&'static Arc<MetalDenseF16ZeroCopyGemm>> {
    HALF_B_MM.get().and_then(|o| o.as_ref())
}

/// The context a prefill GEMM dispatches through: the half-B variant when
/// the F16ACT posture armed it (the route then feeds an f16 activation
/// twin), else the incumbent mixed path. ONE selection site — a dispatch
/// site never guesses.
pub fn gemm_mm_for_posture() -> Option<&'static Arc<MetalDenseF16ZeroCopyGemm>> {
    half_b_mm().or_else(dense_mm)
}

/// Zero-copy raw-Metal dense-f16 GEMM dispatch context (Issue 1150).
///
/// Created once per forward when the `RIIR_QWEN38_DENSE_MM2D=1` arm is set.
/// Holds the raw `id<MTLDevice>` + `id<MTLCommandQueue>` extracted from the
/// shared CubeCL wgpu device + queue, plus the compiled matmul2d
/// `ComputePipelineState`. Ordering with CubeCL is automatic (one queue);
/// the first-use zero-init hazard and the output-aliasing hazard are cured
/// exactly as the ternary module cures them.
pub struct MetalDenseF16ZeroCopyGemm {
    /// Raw `id<MTLDevice>` borrowed from the held wgpu Arcs.
    device: *mut metal::MTLDevice,
    /// Raw `id<MTLCommandQueue>` — the queue CubeCL submits to.
    queue: *mut metal::MTLCommandQueue,
    /// The compiled matmul2d pipeline (owns the library + function).
    pipeline: ComputePipelineState,
    /// The compiled decode-GEMV pipeline (same library, plain kernel — the
    /// matmul2d path cannot serve a (K × 1) activation view; measured: the
    /// cooperative load/store silently produce nothing at p=1).
    gemv_pipeline: ComputePipelineState,
    /// The compiled Q4_K staged-dequant matmul2d pipeline (plan-614 task 1;
    /// own library — the Q4K MSL is a separate source).
    q4k_pipeline: ComputePipelineState,
    /// Whether this context's matmul2d compiled with relaxed precision
    /// (the follow-up-4 selector). Fixed at construction; a getter, not a
    /// re-read, so the compiled source and the disclosure can never drift.
    relaxed_precision: bool,
    /// Dedicated output buffers keyed by byte size — the pool-aliasing cure
    /// (Metal hazard tracking silently drops the cooperative store when the
    /// output shares the CubeCL pool buffer with the operands). One GPU→GPU
    /// copy-back per dispatch, zero host round-trip.
    output_cache: Mutex<HashMap<u64, Buffer>>,
    /// First-use zero-init tracking, per pool buffer, as sorted merged
    /// `[start, end)` intervals — wgpu's lazy buffer clears behind the raw
    /// write (Issue 1146 root cause) are prevented by one CubeCL fill on
    /// first sight of an output range.
    wgpu_initialized: Mutex<HashMap<wgpu::Buffer, Vec<(u64, u64)>>>,
    /// First-use fills issued (`RIIR_ZC_FILL_TRACE=1` prints at powers of 2).
    init_fills: std::sync::atomic::AtomicU64,
    /// Total dispatches since creation (profiling).
    dispatch_count: std::sync::atomic::AtomicU64,
    /// The most recent committed command buffer, retained for
    /// [`Self::wait_last`] (instrument-grade sync for tests/timing cells).
    last_cmd_buffer: std::sync::Mutex<Option<metal::CommandBuffer>>,
    /// Hold the wgpu device + queue Arcs alive — the raw pointers point into
    /// their internals.
    _wgpu_device: Arc<WgpuDevice>,
    _wgpu_queue: Arc<WgpuQueue>,
}

// SAFETY: raw Objective-C object pointers shareable across threads; Metal
// command-queue submission is thread-safe; `ComputePipelineState` (metal-rs)
// is Send + Sync; the wgpu Arcs are Send + Sync.
unsafe impl Send for MetalDenseF16ZeroCopyGemm {}
unsafe impl Sync for MetalDenseF16ZeroCopyGemm {}

/// The tile geometry the dense pipeline compiles at — the ternary kernel's
/// proven fork geometry (64×128 tiles, 4 simdgroups, 32-element K tile).
/// The MSL's SWEEP_* defaults ARE this geometry, so the host constants and
/// the compiled kernel agree by construction. A config override would need
/// the `compose_msl_source` define-preamble treatment (not wired — YAGNI
/// until a calibration cell asks).
const DENSE_NRA: u64 = 64;
const DENSE_NRB: u64 = 128;
const DENSE_NSG: u64 = 4;

// ---------------------------------------------------------------------------
// Issue 1150 follow-up 4 — the dense-lane relaxed-precision selector
// ---------------------------------------------------------------------------
// The fork's tensor `kernel_mul_mm` compiles `matmul2d` with
// relaxed_precision = true; the dense host pinned FALSE (the bit-exact G1
// posture). Bench 923 measured 1.96–2.19× for exactly this flag on the
// ternary sibling of this kernel (same mixed half×float operand pair).
// Selector shape copied from the ternary module's, but SEPARATE from
// `RIIR_METAL_TENSOR_RELAXED`: the dist stage pins the ternary selector
// programmatically at stage load, and a shared knob would leak that pin
// into the dense lane's numerics in a co-hosted process. The dense lane
// owns its env (the lane's `RIIR_QWEN38_*` vocabulary).

/// 0 = unset (fall through to the env arm), 1 = off, 2 = on.
static PROD_RELAXED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static PROD_RELAXED_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Pin relaxed precision for every dense context constructed AFTER this call
/// (wins over `RIIR_QWEN38_DENSE_MM2D_RELAXED`). Test + programmatic arm.
pub fn set_dense_relaxed_precision(on: bool) {
    let v = match on {
        true => 2,
        false => 1,
    };
    PROD_RELAXED.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the next dense context compiles `matmul2d` with relaxed
/// precision: the explicit setter if pinned, else the env — DEMOTE-ONLY
/// since issue 1176 (2026-11-05): the league posture's
/// `RIIR_QWEN38_DENSE_MM2D_RELAXED=1` is the METAL default (bench 944's
/// promoted ids were measured with it armed), and the exact literal "0"
/// restores the pre-1176 incumbent bit-exact posture (the G1 parity
/// lanes' posture — the demotion is byte-identical to the old unset
/// behavior). Parsed once, the demotion announced on stderr; the armed
/// state is disclosed by the dense/half-B arm boot banners.
pub fn production_dense_relaxed_precision() -> bool {
    match PROD_RELAXED.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => *PROD_RELAXED_ENV.get_or_init(|| {
            let demoted = std::env::var("RIIR_QWEN38_DENSE_MM2D_RELAXED").as_deref() == Ok("0");
            if demoted {
                eprintln!(
                    "[Issue 1176] dense-f16 matmul2d relaxed_precision DEMOTED to false by \
                     RIIR_QWEN38_DENSE_MM2D_RELAXED=0 — the bit-exact incumbent"
                );
            }
            !demoted
        }),
    }
}

/// Compose the dense MSL: an explicit `#define SWEEP_RELAXED` on BOTH
/// branches (the bench-921 law — the compiled source string can never
/// disagree with the `.metal` fallback, which stays false).
pub fn compose_dense_msl_source(relaxed: bool, body: &str) -> String {
    let define = match relaxed {
        true => "#define SWEEP_RELAXED true\n",
        false => "#define SWEEP_RELAXED false\n",
    };
    format!("{define}{body}")
}

impl MetalDenseF16ZeroCopyGemm {
    /// Create the zero-copy dense-f16 context on the shared CubeCL Metal
    /// device at the process-selector posture (the env / setter arm). The
    /// explicit-posture constructor [`Self::new_with_relaxed`] is the test
    /// arm — parallel test threads must not mutate the global selector.
    pub fn new(wgpu_device: Arc<WgpuDevice>, wgpu_queue: Arc<WgpuQueue>) -> Result<Self, String> {
        Self::new_with_relaxed(
            wgpu_device,
            wgpu_queue,
            production_dense_relaxed_precision(),
        )
    }

    /// Create the zero-copy dense-f16 context with an EXPLICIT
    /// relaxed-precision posture. Returns `Err` when wgpu isn't Metal or MSL
    /// compilation fails (Metal 3.1+ required for `metal::tensor`).
    pub fn new_with_relaxed(
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
        relaxed: bool,
    ) -> Result<Self, String> {
        Self::new_with_gemm_source(wgpu_device, wgpu_queue, relaxed, MSL_SOURCE)
    }

    /// The alternate-MSL constructor — issue 1170's probe arm: the SAME
    /// choreography (dispatch is dtype-agnostic, same buffer/scalar
    /// bindings), the GEMM pipeline compiled from an alternate source (the
    /// half×half matmul2d variant). The gemv + q4k pipelines compile as
    /// usual; only the GEMM function swaps. Probe-only until the A/B
    /// promotes it — no route caller, no env.
    pub fn new_with_gemm_source(
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
        relaxed: bool,
        gemm_source: &str,
    ) -> Result<Self, String> {
        // Raw device + queue extraction — the T4 pattern (identical to the
        // ternary module's).
        let device_ptr: *mut metal::MTLDevice = unsafe {
            let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
            let dev = guard
                .as_ref()
                .ok_or("wgpu device is not Metal — dense-f16 zero-copy GEMM requires Metal")?;
            let proto = dev.raw_device();
            (&**proto) as *const _ as *mut metal::MTLDevice
        };
        let queue_ptr: *mut metal::MTLCommandQueue = unsafe {
            let guard = wgpu_queue.as_hal::<wgpu::hal::api::Metal>();
            let q = guard
                .as_ref()
                .ok_or("wgpu queue is not Metal — dense-f16 zero-copy GEMM requires Metal")?;
            let proto = q.as_raw();
            proto as *const _ as *mut metal::MTLCommandQueue
        };

        // Compile the dense matmul2d pipeline. The MSL's SWEEP_* defaults are
        // the fork geometry; the DENSE_* constants below mirror them (the
        // compile-time check is the G1 gate's edge-shape suite). relaxed
        // precision is the follow-up-4 selector (default OFF, the incumbent
        // bit-exact posture) — pinned in the field so the gate + bench logs
        // can prove which arm a context carries.
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
        let options = CompileOptions::new();
        let source = compose_dense_msl_source(relaxed, gemm_source);
        let library = device_ref
            .new_library_with_source(source.as_str(), &options)
            .map_err(|e| format!("dense-f16 MSL compilation failed: {e}"))?;
        let function = library
            .get_function("gemm_dense_f16_tensor", None)
            .map_err(|e| format!("Function 'gemm_dense_f16_tensor' not found: {e}"))?;
        let pipeline = device_ref
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("dense-f16 pipeline creation failed: {e}"))?;

        // The decode-GEMV function lives in the CANONICAL source; an
        // alternate GEMM source (the probe arm) compiles it separately —
        // one extra library compile at construction, probe-only cost. The
        // canonical path reuses the same library (pointer-equal sources).
        let gemv_library = if std::ptr::eq(gemm_source, MSL_SOURCE) {
            library.clone()
        } else {
            device_ref
                .new_library_with_source(MSL_SOURCE, &options)
                .map_err(|e| format!("dense-f16 gemv MSL compilation failed: {e}"))?
        };
        let gemv_function = gemv_library
            .get_function("gemv_dense_f16", None)
            .map_err(|e| format!("Function 'gemv_dense_f16' not found: {e}"))?;
        let gemv_pipeline = device_ref
            .new_compute_pipeline_state_with_function(&gemv_function)
            .map_err(|e| format!("dense-f16 gemv pipeline creation failed: {e}"))?;
        drop(library);
        drop(gemv_library);

        // The Q4_K pipeline: separate source, same compile path.
        let q4k_library = device_ref
            .new_library_with_source(Q4K_MSL_SOURCE, &options)
            .map_err(|e| format!("q4k MSL compilation failed: {e}"))?;
        let q4k_function = q4k_library
            .get_function("gemm_q4k_tensor", None)
            .map_err(|e| format!("Function 'gemm_q4k_tensor' not found: {e}"))?;
        let q4k_pipeline = device_ref
            .new_compute_pipeline_state_with_function(&q4k_function)
            .map_err(|e| format!("q4k pipeline creation failed: {e}"))?;
        drop(q4k_library);

        Ok(Self {
            device: device_ptr,
            queue: queue_ptr,
            pipeline,
            gemv_pipeline,
            q4k_pipeline,
            relaxed_precision: relaxed,
            output_cache: Mutex::new(HashMap::new()),
            wgpu_initialized: Mutex::new(HashMap::new()),
            init_fills: std::sync::atomic::AtomicU64::new(0),
            dispatch_count: std::sync::atomic::AtomicU64::new(0),
            last_cmd_buffer: std::sync::Mutex::new(None),
            _wgpu_device: wgpu_device,
            _wgpu_queue: wgpu_queue,
        })
    }

    /// Total dispatches since creation (profiling).
    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether this context compiled `matmul2d` with relaxed precision —
    /// the engagement-proof getter (a bench log or gate asserts the ARM it
    /// measured, never infers it from env).
    pub fn relaxed_precision(&self) -> bool {
        self.relaxed_precision
    }

    /// Issue-1146 first-use zero-init cure (same body as the ternary
    /// module's, on this context's own tracker): the first time an output
    /// range is seen, have CubeCL write it so wgpu marks the range
    /// initialized BEFORE the raw blit lands; a seen range is a set lookup.
    /// `pub(crate)` so the tensor-attention arm can share THIS tracker (one
    /// tracker per pool buffer — a second tracker would re-fill ranges the
    /// dense GEMM already marked); the [`TensorAttnHost`] impl below is the
    /// dispatch-site surface.
    pub(crate) fn ensure_wgpu_initialized(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        output: &Handle,
        len: usize,
    ) -> Result<(), String> {
        let managed = client
            .get_resource(output.clone())
            .map_err(|e| format!("get_resource failed: {e:?}"))?;
        let resource = managed.resource();
        let start = resource.offset;
        let end = start + (len * std::mem::size_of::<f32>()) as u64;
        let mut seen = self.wgpu_initialized.lock().unwrap();
        let ranges = seen.entry(resource.buffer.clone()).or_default();
        if covered(ranges, start, end) {
            return Ok(());
        }
        unsafe {
            crate::elementwise_cubecl::FillZerosCubeCL::launch::<ActiveRuntime>(
                client,
                output.clone(),
                len,
            );
        }
        let _ = client.flush();
        insert_merged(ranges, start, end);
        let n = self
            .init_fills
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if n.is_power_of_two() && std::env::var("RIIR_ZC_FILL_TRACE").as_deref() == Ok("1") {
            eprintln!("[Issue 1150] dense-f16 zero-copy first-use fills: {n}");
        }
        Ok(())
    }

    /// Dispatch the dense-f16 GEMM via raw Metal on the shared CubeCL queue.
    ///
    /// Computes `output[P × M] = W[M × K](f16) @ input[P × K](f32)^T`.
    /// Zero copy both directions: W and X are read from the CubeCL-managed
    /// buffers in place; Y is written to a dedicated Metal buffer and copied
    /// back GPU→GPU (the aliasing cure).
    ///
    /// The decode path rides [`Self::dispatch_gemv`] (the plain kernel) —
    /// the matmul2d cooperative path cannot serve a (K × 1) activation view.
    pub fn dispatch(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        w: &Handle,
        m: usize,
        n: usize,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> Result<(), String> {
        // Submit CubeCL's pending producers FIRST (the Issue-670 ordering
        // law): our raw commit must not enqueue ahead of the kernels writing
        // `input` on the shared queue. Also settles pool bookkeeping before
        // the raw-handle walk.
        let _ = client.flush();
        self.ensure_wgpu_initialized(client, output, p * m)?;

        // Raw handles, fresh per dispatch (pool may reallocate).
        let (w_ptr, w_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, w,
            )?;
        let (input_ptr, input_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, input,
            )?;
        let (output_ptr, output_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, output,
            )?;

        let w_ref: &BufferRef = unsafe { BufferRef::from_ptr(w_ptr) };
        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let cubecl_output_ref: &BufferRef = unsafe { BufferRef::from_ptr(output_ptr) };

        // Dedicated output buffer — the pool-aliasing cure (see field doc).
        let output_bytes = (p * m * std::mem::size_of::<f32>()) as u64;
        let dedicated_output = {
            let mut cache = self.output_cache.lock().unwrap();
            cache
                .entry(output_bytes)
                .or_insert_with(|| {
                    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(self.device) };
                    device_ref.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared)
                })
                .clone()
        };

        // Tile geometry — the compiled MSL defaults (the DENSE_* constants;
        // the host and the kernel agree by construction).
        let nra = DENSE_NRA;
        let nrb = DENSE_NRB;
        let nsg = DENSE_NSG;
        let num_threads = 32u64 * nsg;

        // Grid: one workgroup per output tile — `dispatch_thread_groups`, the
        // matmul2d kernel indexes via tgpig (the ternary module's law).
        let grid = MTLSize {
            width: (p as u64).div_ceil(nrb),
            height: (m as u64).div_ceil(nra),
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        let k: u32 = n as u32;
        let m: u32 = m as u32;
        let p_tokens: u32 = p as u32;

        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Buffer bindings (match gemm_dense_f16_metal_tensor.metal):
        //   0: weights_maj (read, CubeCL pool, f16)
        //   1: input_batch (read, CubeCL pool, f32)
        //   2: output_batch (write, DEDICATED Metal buffer)
        encoder.set_buffer(0, Some(w_ref), w_off);
        encoder.set_buffer(1, Some(input_ref), input_off);
        encoder.set_buffer(2, Some(&dedicated_output), 0);

        // Scalar bindings (3..5 match the MSL `constant uint&` declarations).
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        encoder.set_bytes(5, 4, &p_tokens as *const u32 as *const _);

        // Hazard declarations — weights + input read, output written (the
        // memory-less hazard-tracking race the ternary module measured on
        // M3 Max).
        encoder.use_resource(w_ref, MTLResourceUsage::Read);
        encoder.use_resource(input_ref, MTLResourceUsage::Read);
        encoder.use_resource(&dedicated_output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();

        // GPU→GPU copy-back within the same command buffer (ordered after the
        // compute pass) — zero host round-trip.
        let blit_encoder = cmd_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &dedicated_output,
            0,
            cubecl_output_ref,
            output_off,
            output_bytes,
        );
        blit_encoder.end_encoding();

        cmd_buffer.commit();
        *self.last_cmd_buffer.lock().expect("last_cmd_buffer lock") = Some(cmd_buffer.to_owned());

        // No wait_until_completed — in-order shared queue (the ternary
        // module's async-pipeline law; see its dispatch comment).
        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Dispatch the Q4_K staged-dequant GEMM (plan-614 Phase-3 task 1):
    /// `output[P × M] = dequant_q4k(W[M × K]) @ input[P × K](f32)^T` with the
    /// weights consumed as Q4_K blocks IN PLACE (no f16 resident copy).
    /// Same choreography as [`Self::dispatch`] (dedicated output + blit,
    /// first-use cure, hazard declarations); the tile geometry is the same
    /// fork geometry, and the staging phase adds one threadgroup-memory
    /// buffer (K_TILE × NRA halfs = 4 KB).
    ///
    /// Shape laws: `m` must be a multiple of the 64-row tile (the cooperative
    /// store skips tile-ragged m — the dense kernel's measured law), `n` a
    /// multiple of 256 (the Q4_K block), `p` ≥ 1 (a (K × 1) view is served:
    /// the staged-dequant path stages A independently of p — the p=1 G1 cell
    /// pins it).
    pub fn dispatch_q4k(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        w_q4k: &Handle,
        d_dmin: &Handle,
        m: usize,
        n: usize,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> Result<(), String> {
        // Submit CubeCL's pending producers FIRST (the Issue-670 ordering
        // law); also settles pool bookkeeping before the raw-handle walk.
        let _ = client.flush();
        self.ensure_wgpu_initialized(client, output, p * m)?;

        let (w_ptr, w_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, w_q4k,
            )?;
        let (dd_ptr, dd_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, d_dmin,
            )?;
        let (input_ptr, input_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, input,
            )?;
        let (output_ptr, output_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, output,
            )?;

        let w_ref: &BufferRef = unsafe { BufferRef::from_ptr(w_ptr) };
        let dd_ref: &BufferRef = unsafe { BufferRef::from_ptr(dd_ptr) };
        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let cubecl_output_ref: &BufferRef = unsafe { BufferRef::from_ptr(output_ptr) };

        let output_bytes = (p * m * std::mem::size_of::<f32>()) as u64;
        let dedicated_output = {
            let mut cache = self.output_cache.lock().unwrap();
            cache
                .entry(output_bytes)
                .or_insert_with(|| {
                    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(self.device) };
                    device_ref.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared)
                })
                .clone()
        };

        let nra = DENSE_NRA;
        let nrb = DENSE_NRB;
        let nsg = DENSE_NSG;
        let num_threads = 32u64 * nsg;
        let grid = MTLSize {
            width: (p as u64).div_ceil(nrb),
            height: (m as u64).div_ceil(nra),
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        let k: u32 = n as u32;
        let m_u: u32 = m as u32;
        let p_tokens: u32 = p as u32;
        // The staging tile: K_TILE × NRA halfs (32 × 64 × 2 = 4 KB at the
        // fork geometry) — mirrors gemm_ternary_metal_tensor's staging.
        let shmem_bytes: u64 = 32 * 64 * 2;

        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4k_pipeline);

        // Buffer bindings (match gemm_q4k_metal_tensor.metal):
        //   0: weight_q4k (read, CubeCL pool, packed Q4_K payloads)
        //   1: d_dmin (read, CubeCL pool, f32 block scales)
        //   2: input_batch (read, CubeCL pool, f32)
        //   3: output_batch (write, DEDICATED Metal buffer)
        encoder.set_buffer(0, Some(w_ref), w_off);
        encoder.set_buffer(1, Some(dd_ref), dd_off);
        encoder.set_buffer(2, Some(input_ref), input_off);
        encoder.set_buffer(3, Some(&dedicated_output), 0);

        // Scalar bindings (4..6 match the MSL `constant uint&` declarations).
        encoder.set_bytes(4, 4, &k as *const u32 as *const _);
        encoder.set_bytes(5, 4, &m_u as *const u32 as *const _);
        encoder.set_bytes(6, 4, &p_tokens as *const u32 as *const _);

        // Threadgroup memory for the dequanted weight tile (index 0).
        encoder.set_threadgroup_memory_length(0, shmem_bytes);

        encoder.use_resource(w_ref, MTLResourceUsage::Read);
        encoder.use_resource(dd_ref, MTLResourceUsage::Read);
        encoder.use_resource(input_ref, MTLResourceUsage::Read);
        encoder.use_resource(&dedicated_output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();

        // GPU→GPU copy-back within the same command buffer (ordered after the
        // compute pass) — zero host round-trip.
        let blit_encoder = cmd_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &dedicated_output,
            0,
            cubecl_output_ref,
            output_off,
            output_bytes,
        );
        blit_encoder.end_encoding();

        cmd_buffer.commit();
        *self.last_cmd_buffer.lock().expect("last_cmd_buffer lock") = Some(cmd_buffer.to_owned());

        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Dispatch the decode GEMV (`y[m] = W[m×n](f16) @ x[n]`) — the plain
    /// kernel (one simdgroup per row, coalesced f16 reads, one simd_sum),
    /// no tile constraints, direct write to the CubeCL output handle (normal
    /// store semantics — the dedicated-output cure is for the cooperative
    /// store only). The first-use zero-init cure + hazard declarations still
    /// apply (raw dispatch, wgpu tracking blind).
    pub fn dispatch_gemv(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        w: &Handle,
        m: usize,
        n: usize,
        input: &Handle,
        output: &Handle,
    ) -> Result<(), String> {
        let _ = client.flush();
        self.ensure_wgpu_initialized(client, output, m)?;

        let (w_ptr, w_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, w,
            )?;
        let (input_ptr, input_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, input,
            )?;
        let (output_ptr, output_off) =
            crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::extract_raw_buffer(
                client, output,
            )?;

        let w_ref: &BufferRef = unsafe { BufferRef::from_ptr(w_ptr) };
        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let output_ref: &BufferRef = unsafe { BufferRef::from_ptr(output_ptr) };

        // 128 threads = 4 simdgroups = 4 rows per workgroup.
        let grid = MTLSize {
            width: (m as u64).div_ceil(4),
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        let k: u32 = n as u32;
        let md: u32 = m as u32;

        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemv_pipeline);
        encoder.set_buffer(0, Some(w_ref), w_off);
        encoder.set_buffer(1, Some(input_ref), input_off);
        encoder.set_buffer(2, Some(output_ref), output_off);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &md as *const u32 as *const _);
        encoder.use_resource(w_ref, MTLResourceUsage::Read);
        encoder.use_resource(input_ref, MTLResourceUsage::Read);
        encoder.use_resource(output_ref, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();

        cmd_buffer.commit();
        *self.last_cmd_buffer.lock().expect("last_cmd_buffer lock") = Some(cmd_buffer.to_owned());
        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Device pointer (device-equality diagnostics).
    pub fn device_ptr(&self) -> usize {
        self.device as usize
    }

    /// Queue pointer (device-equality diagnostics).
    pub fn queue_ptr(&self) -> usize {
        self.queue as usize
    }

    /// Wait for the most recent raw dispatch (instrument-grade sync for
    /// tests/timing cells that read the output via a non-queued path).
    pub fn wait_last(&self) {
        if let Some(cb) = self
            .last_cmd_buffer
            .lock()
            .expect("last_cmd_buffer lock")
            .as_ref()
        {
            cb.wait_until_completed();
        }
    }
}

// SAFETY: delegation — the tensor-attention arm (issue 1171) shares this
// context's tracker + device/queue so the qwen38 pool has ONE first-use
// tracker across the dense GEMM and the attention arm.
impl crate::attention_prefill_metal_tensor::TensorAttnHost for MetalDenseF16ZeroCopyGemm {
    fn ensure_wgpu_initialized(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        out: &Handle,
        len: usize,
    ) -> Result<(), String> {
        self.ensure_wgpu_initialized(client, out, len)
    }
    fn device_ptr(&self) -> usize {
        self.device as usize
    }
    fn queue_ptr(&self) -> usize {
        self.queue as usize
    }
}

impl Drop for MetalDenseF16ZeroCopyGemm {
    fn drop(&mut self) {
        // The raw device + queue pointers are borrowed from the held wgpu
        // Arcs; Drop runs before they drop. Nothing to release.
    }
}
