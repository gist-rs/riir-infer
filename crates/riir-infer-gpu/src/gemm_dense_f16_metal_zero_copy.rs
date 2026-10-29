//! Issue 1150 follow-up — Zero-copy raw-Metal **dense-f16** GEMM via the
//! shared CubeCL device.
//!
//! The dense twin of [`crate::gemm_ternary_metal_zero_copy`]: the SAME
//! zero-copy raw-Metal dispatch choreography (shared device + queue via
//! `wgpu::Device::as_hal`, per-dispatch raw handle extraction, dedicated
//! output buffer + GPU→GPU copy-back, `ensure_wgpu_initialized` first-use
//! cure), driving a matmul2d kernel whose A operand is the f16 weight matrix
//! read STRAIGHT from device memory — no ternary bit-plane dequant, no
//! threadgroup staging, no staging barrier in the K loop.
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
//! prefill. The posture is opt-in (`RIIR_QWEN38_DENSE_MM2D=1`), never the
//! default, and both postures are benched.
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

impl MetalDenseF16ZeroCopyGemm {
    /// Create the zero-copy dense-f16 context on the shared CubeCL Metal
    /// device. Returns `Err` when wgpu isn't Metal or MSL compilation fails
    /// (Metal 3.1+ required for `metal::tensor`).
    pub fn new(wgpu_device: Arc<WgpuDevice>, wgpu_queue: Arc<WgpuQueue>) -> Result<Self, String> {
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
        // compile-time check is the G1 gate's edge-shape suite).
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
        let options = CompileOptions::new();
        let library = device_ref
            .new_library_with_source(MSL_SOURCE, &options)
            .map_err(|e| format!("dense-f16 MSL compilation failed: {e}"))?;
        let function = library
            .get_function("gemm_dense_f16_tensor", None)
            .map_err(|e| format!("Function 'gemm_dense_f16_tensor' not found: {e}"))?;
        let pipeline = device_ref
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("dense-f16 pipeline creation failed: {e}"))?;
        let gemv_function = library
            .get_function("gemv_dense_f16", None)
            .map_err(|e| format!("Function 'gemv_dense_f16' not found: {e}"))?;
        let gemv_pipeline = device_ref
            .new_compute_pipeline_state_with_function(&gemv_function)
            .map_err(|e| format!("dense-f16 gemv pipeline creation failed: {e}"))?;
        drop(library);

        Ok(Self {
            device: device_ptr,
            queue: queue_ptr,
            pipeline,
            gemv_pipeline,
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

    /// Issue-1146 first-use zero-init cure (same body as the ternary
    /// module's, on this context's own tracker): the first time an output
    /// range is seen, have CubeCL write it so wgpu marks the range
    /// initialized BEFORE the raw blit lands; a seen range is a set lookup.
    fn ensure_wgpu_initialized(
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

impl Drop for MetalDenseF16ZeroCopyGemm {
    fn drop(&mut self) {
        // The raw device + queue pointers are borrowed from the held wgpu
        // Arcs; Drop runs before they drop. Nothing to release.
    }
}
