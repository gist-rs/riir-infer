//! Issue 663 T5 — Zero-copy raw-Metal ternary GEMM via shared CubeCL device.
//!
//! This is the **third attempt** at capturing the 3.72× kernel-level matmul2d
//! speedup (Bench 645) end-to-end. The first two were refuted by host
//! round-trip / staging-copy overhead:
//!
//! - Issue 656 (0.87×): raw Metal kernel on a **separate** `Device::system_default()`
//!   — each of ~448 prefill projections forced 2 blocking syncs (CubeCL read →
//!   Metal upload → Metal run → Metal read → CubeCL write).
//! - Issue 657 (0.95×): same kernel dispatched via wgpu MSL passthrough on the
//!   shared CubeCL queue — per-dispatch staging copies + bind group rebuild
//!   negated the kernel gain.
//!
//! This module eliminates **both** overhead sources by using the raw Metal
//! dispatch path validated in the T4 test (`bench_663_zero_copy_interop.rs`):
//!
//! - **Same device + queue as CubeCL** — extracted via `wgpu::Device::as_hal::<Metal>`
//!   + `raw_device()` / `as_raw()` (no cross-queue sync, no separate context).
//! - **Zero-copy weight cache** — reuses the CubeCL-managed weight buffers
//!   directly (extracts raw `id<MTLBuffer>` via `get_resource` →
//!   `as_hal::<Metal>` → `raw_handle()`). No re-upload, no staging copy.
//! - **Zero-copy input/output** — each dispatch reads + writes the same GPU
//!   memory CubeCL manages. No staging buffers, no host round-trip.
//! - **Raw `metal::ComputePipelineState` dispatch** — no wgpu compute pass,
//!   no bind group, no command encoder allocations per dispatch.
//!
//! # Architecture
//!
//! ```text
//! Boot (once per weight matrix):
//!   CubeCL weight handles → extract raw id<MTLBuffer> → cache in ZeroCopyWeightCache
//!
//! Per dispatch (zero copy, zero round-trip):
//!   1. Extract raw id<MTLBuffer> for input + output CubeCL handles
//!   2. queue_ref.new_command_buffer() on the shared Metal queue
//!   3. encoder.set_buffer(0..4, raw handles) + set_bytes(5..9, scalars)
//!   4. encoder.dispatch_threads(grid, threadgroup)
//!   5. cmd_buffer.commit() — ordering with CubeCL is automatic (same queue)
//! ```
//!
//! # Feature gate
//!
//! `metal_tensor_gemm` + macOS-only. Requires the vendored wgpu-hal fork that
//! exposes `metal::Buffer::raw_handle()` (Issue 663 T1.5).

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
use crate::gemv_ternary_cubecl::TernaryHandle;

/// The raw MSL source — identical to Issue 656's matmul2d kernel (scalar
/// buffer bindings 5-9, not the packed-struct wgpu variant). Dispatched
/// directly via `metal::ComputePipelineState`.
const MSL_SOURCE: &str = include_str!("gemm_ternary_metal_tensor.metal");

/// Cached raw weight buffer info for one ternary weight matrix.
///
/// Stores the CubeCL handles + lazily-extracted raw buffer info. The raw
/// pointers are re-extracted on each dispatch (not cached across dispatches)
/// because CubeCL's memory pool can reallocate. However, the CubeCL handles
/// themselves are stable (reference-counted) and cheap to clone.
#[derive(Clone)]
pub struct ZeroCopyWeightCache {
    /// CubeCL handle for `pos_bits_u32`.
    pos: Handle,
    /// CubeCL handle for `neg_bits_u32`.
    neg: Handle,
    /// CubeCL handle for `group_scale_f32`.
    scale: Handle,
}

// SAFETY: `Handle` is `Send + Sync` (it wraps an Arc to CubeCL's server-side
// resource management). The cache can be shared across threads.
unsafe impl Send for ZeroCopyWeightCache {}
unsafe impl Sync for ZeroCopyWeightCache {}

/// Zero-copy raw-Metal ternary GEMM dispatch context (Issue 663 T5).
///
/// Created once at model load time. Holds the raw `id<MTLDevice>` +
/// `id<MTLCommandQueue>` extracted from the shared CubeCL wgpu device + queue,
/// plus the compiled `metal::tensor` matmul2d `ComputePipelineState`.
///
/// # Why this beats Issue 657
///
/// Issue 657 (`MetalTensorWgpuGemm`) used wgpu's compute pass API with
/// staging copies to work around wgpu's usage-scope validation. This module
/// skips wgpu entirely for the dispatch — it goes straight to Metal's
/// `MTLComputeCommandEncoder`, which has no usage-scope validation and can
/// bind the same buffer as both read + write within one dispatch (with
/// `use_resource(..., Read | Write)`).
///
/// # Why this beats Issue 656
///
/// Issue 656 (`MetalTensorGemm`) used a separate `Device::system_default()`
/// Metal context, forcing host round-trips for data exchange with CubeCL.
/// This module shares CubeCL's device + queue — the buffers ARE the same GPU
/// memory, no round-trip is possible.
pub struct MetalTensorZeroCopyGemm {
    /// Raw `id<MTLDevice>` extracted from the shared CubeCL wgpu device.
    /// Borrowed for the lifetime of the `WgpuDevice` Arc (held by `CubeCLContext`).
    device: *mut metal::MTLDevice,
    /// Raw `id<MTLCommandQueue>` extracted from the shared CubeCL wgpu queue.
    /// Same queue CubeCL submits to — ordering is automatic.
    queue: *mut metal::MTLCommandQueue,
    /// The compiled matmul2d pipeline. Owns the library + function internally.
    pipeline: ComputePipelineState,
    /// Cached dedicated output buffers, keyed by byte size. These are
    /// REQUIRED because CubeCL sub-allocates all handles within one pool
    /// buffer — when the output shares the same `id<MTLBuffer>` as the
    /// weights/input, Metal's hazard tracking silently fails the kernel's
    /// cooperative-tensor store. The dedicated output buffer breaks the
    /// aliasing; after the kernel writes, we copy back to the CubeCL handle.
    /// This is ONE GPU→GPU copy (not a host round-trip).
    output_cache: Mutex<HashMap<u64, Buffer>>,
    /// Issue 663 T5 profiling: total dispatch count (for overhead analysis).
    dispatch_count: std::sync::atomic::AtomicU64,
    /// Hold the wgpu device + queue Arcs alive — the raw Metal handles point
    /// into their internals. Dropping these would invalidate `device` + `queue`.
    _wgpu_device: Arc<WgpuDevice>,
    _wgpu_queue: Arc<WgpuQueue>,
}

// SAFETY: The raw `id` pointers are Objective-C object pointers that can be
// shared across threads. Metal's command queue submission is thread-safe.
// The `ComputePipelineState` (metal-rs) is `Send + Sync` (it wraps an `id`
// via foreign-types). The wgpu Arcs are `Send + Sync`.
unsafe impl Send for MetalTensorZeroCopyGemm {}
unsafe impl Sync for MetalTensorZeroCopyGemm {}

impl MetalTensorZeroCopyGemm {
    /// Create a new zero-copy GEMM context on the shared CubeCL Metal device.
    ///
    /// Takes the wgpu `Device` + `Queue` Arcs from `CubeCLContext`. Extracts
    /// the raw Metal handles (via the T4 pattern), compiles the matmul2d
    /// kernel, and returns a dispatch context. Returns `Err` if:
    /// - wgpu isn't using the Metal backend (non-macOS without the feature)
    /// - MSL compilation fails (missing `metal::tensor` support — Metal 3.1+)
    pub fn new(
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
    ) -> Result<Self, String> {
        // Extract the raw Metal device + queue (T4 pattern).
        let device_ptr: *mut metal::MTLDevice = unsafe {
            let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
            let dev = guard
                .as_ref()
                .ok_or("wgpu device is not Metal — zero-copy GEMM requires Metal backend")?;
            let proto = dev.raw_device(); // &Retained<ProtocolObject<dyn MTLDevice>>
            (&**proto) as *const _ as *mut metal::MTLDevice
        };

        let queue_ptr: *mut metal::MTLCommandQueue = unsafe {
            let guard = wgpu_queue.as_hal::<wgpu::hal::api::Metal>();
            let q = guard
                .as_ref()
                .ok_or("wgpu queue is not Metal — zero-copy GEMM requires Metal backend")?;
            let proto = q.as_raw(); // &ProtocolObject<dyn MTLCommandQueue>
            proto as *const _ as *mut metal::MTLCommandQueue
        };

        // Compile the matmul2d kernel on this device.
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
        let options = CompileOptions::new();
        let library = device_ref
            .new_library_with_source(MSL_SOURCE, &options)
            .map_err(|e| format!("MSL compilation failed: {e}"))?;
        let function = library
            .get_function("gemm_ternary_tensor", None)
            .map_err(|e| format!("Function 'gemm_ternary_tensor' not found: {e}"))?;
        let pipeline = device_ref
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline creation failed: {e}"))?;

        // Drop the library — the pipeline holds its own reference.
        drop(library);

        Ok(Self {
            device: device_ptr,
            queue: queue_ptr,
            pipeline,
            output_cache: Mutex::new(HashMap::new()),
            dispatch_count: std::sync::atomic::AtomicU64::new(0),
            _wgpu_device: wgpu_device,
            _wgpu_queue: wgpu_queue,
        })
    }

    /// Total dispatches since creation (profiling).
    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Cache CubeCL weight handles for zero-copy dispatch.
    ///
    /// Called once per weight matrix at model load time. Stores references to
    /// the CubeCL weight handles — the raw `id<MTLBuffer>` pointers are
    /// extracted fresh on every dispatch (not cached) because CubeCL's memory
    /// pool can reallocate/defragment between dispatches, invalidating cached
    /// raw pointers.
    pub fn cache_weights(
        &self,
        _client: &ComputeClient<ActiveRuntime>,
        w: &TernaryHandle,
    ) -> Result<ZeroCopyWeightCache, String> {
        Ok(ZeroCopyWeightCache {
            pos: w.pos_bits_u32.clone(),
            neg: w.neg_bits_u32.clone(),
            scale: w.group_scale_f32.clone(),
        })
    }

    /// Extract `(raw id<MTLBuffer>, offset)` from a CubeCL handle.
    ///
    /// Returns a raw pointer into the CubeCL-managed GPU memory. The pointer
    /// is valid for as long as the CubeCL handle is alive.
    fn extract_raw_buffer(
        client: &ComputeClient<ActiveRuntime>,
        handle: &Handle,
    ) -> Result<(*mut metal::MTLBuffer, u64), String> {
        // cuda_backend twin: the CubeCL resource is CUDA-shaped (no wgpu
        // buffer to extract a Metal handle from) — report unavailable
        // instead of failing to compile. Callers treat Err as "path
        // disabled" and fall through.
        #[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
        {
            let _ = (client, handle);
            Err(
                "zero-copy Metal extraction unavailable: cuda_backend replaces the wgpu CubeCL runtime"
                    .to_string(),
            )
        }
        #[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
        {
            let managed = client
                .get_resource(handle.clone())
                .map_err(|e| format!("get_resource failed: {e:?}"))?;
            let resource = managed.resource();
            let offset = resource.offset;
            let wgpu_buffer = &resource.buffer;

            // SAFETY: `as_hal` is unsafe because it exposes the raw hal buffer.
            // We only read the raw pointer (via `raw_handle()`) — no mutation.
            let id_ptr = unsafe {
                let guard = wgpu_buffer.as_hal::<wgpu::hal::api::Metal>();
                let hal_buf = guard
                    .as_ref()
                    .ok_or("CubeCL buffer is not Metal — zero-copy requires Metal backend")?;
                let proto = hal_buf.raw_handle(); // &ProtocolObject<dyn MTLBuffer>
                proto as *const _ as *mut metal::MTLBuffer
            };
            Ok((id_ptr, offset))
        }
    }

    /// Dispatch the ternary GEMM via raw Metal on the shared CubeCL queue.
    ///
    /// **Zero copy, zero host round-trip.** Reads input from the CubeCL input
    /// handle, writes output to the CubeCL output handle, dispatches the
    /// matmul2d kernel on the same Metal command queue CubeCL uses. No
    /// staging buffers, no bind groups, no host round-trip.
    ///
    /// # Arguments
    ///
    /// - `client`: the CubeCL compute client (for raw buffer extraction)
    /// - `weights`: the cached raw weight buffers (from `cache_weights`)
    /// - `w`: the ternary weight handle (for `m`, `n`, `blocks64`, etc.)
    /// - `input`: CubeCL handle holding `[P × N]` f32 input activations
    /// - `output`: CubeCL handle for `[P × M]` f32 output (will be written)
    /// - `p`: number of tokens (P dimension)
    pub fn dispatch(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        weights: &ZeroCopyWeightCache,
        w: &TernaryHandle,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> Result<(), String> {
        // Issue 670 T4: force CubeCL to submit any pending command buffers to
        // the shared Metal queue BEFORE we commit our raw command buffer.
        // This dispatch binds CubeCL pool buffers directly; if the producer
        // kernels writing `input` are still sitting in CubeCL's server
        // channel, our raw commit would be enqueued AHEAD of them and the
        // kernel would read stale input — deterministic-per-shape garbage
        // (the `needless_return` G1 failure in the production-scale re-bench).
        // The isolated Issue 663 bench masked this hazard by calling
        // `pollster::block_on(client.sync())` before extraction;
        // `flush()` gets the same submission ordering without a CPU-GPU sync
        // (it submits, it does not wait). Flush FIRST — also settles any
        // pending pool bookkeeping before `extract_raw_buffer` walks it.
        //
        // Issue 670 T5 (2026-08-14): a first-dispatch `client.sync()` variant
        // was tried and does NOT cure the residual first-prefill corruption
        // (case-order A/B: only the FIRST zerocopy prefill after init reads
        // wrong state; later prefills are bit-identical to simdgroup). The
        // corruption is not a submission-visibility hazard — see the issue's
        // T5 close-out for the pool-reservation characterization.
        let _ = client.flush();

        // Extract raw weight + input buffer handles fresh on every dispatch.
        let (pos_ptr, pos_off) = Self::extract_raw_buffer(client, &weights.pos)?;
        let (neg_ptr, neg_off) = Self::extract_raw_buffer(client, &weights.neg)?;
        let (scale_ptr, scale_off) = Self::extract_raw_buffer(client, &weights.scale)?;
        let (input_ptr, input_off) = Self::extract_raw_buffer(client, input)?;
        let (output_ptr, output_off) = Self::extract_raw_buffer(client, output)?;

        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let pos_ref: &BufferRef = unsafe { BufferRef::from_ptr(pos_ptr) };
        let neg_ref: &BufferRef = unsafe { BufferRef::from_ptr(neg_ptr) };
        let scale_ref: &BufferRef = unsafe { BufferRef::from_ptr(scale_ptr) };
        let cubecl_output_ref: &BufferRef = unsafe { BufferRef::from_ptr(output_ptr) };

        // Allocate or reuse a DEDICATED output buffer for the kernel write.
        //
        // CubeCL sub-allocates all handles within one `id<MTLBuffer>` pool
        // buffer. When the output handle shares the same pool buffer as the
        // weights/input (which it ALWAYS does in practice), Metal's hazard
        // tracking silently fails the cooperative-tensor store — the kernel
        // runs but writes nothing. The fix: write to a dedicated Metal
        // buffer (not from the CubeCL pool), then copy back to the CubeCL
        // output handle via `encoder.copy_from_buffer` (GPU→GPU, same
        // command buffer, zero host round-trip).
        //
        // This is the same pattern Issue 657 used for its staging buffers,
        // but ONLY for the output (not the input + weights) — reducing the
        // copy count from 2 (Issue 657) to 1.
        let output_bytes = (p * w.m * std::mem::size_of::<f32>()) as u64;
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

        // matmul2d tile constants (must match the MSL kernel).
        const NRA: u32 = 64; // features (M) per workgroup
        const NRB: u32 = 128; // tokens (P) per workgroup
        const NSG: u32 = 4; // simdgroups per workgroup
        const THREADS_PER_SG: u32 = 32;
        const NUM_THREADS: u32 = THREADS_PER_SG * NSG; // 128
        const THREADGROUP_MEM_BYTES: u64 = 32 * 64 * 2; // K_TILE × NRA × sizeof(half)

        // Grid: one workgroup per output tile.
        // NOTE: use `dispatch_thread_groups` (not `dispatch_threads`). The
        // matmul2d kernel indexes its output tile via `tgpig` (threadgroup
        // position in grid), so the grid must be in thread GROUPS, not
        // threads. `dispatch_threads` interprets the grid as thread count,
        // which dispatches far too many workgroups + causes out-of-bounds writes.
        // (Issue 656's `MetalTensorGemm::launch` uses `dispatch_thread_groups`.)
        let grid = MTLSize {
            width: (p as u64).div_ceil(NRB as u64), // token tiles
            height: (w.m as u64).div_ceil(NRA as u64), // feature tiles
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: NUM_THREADS as u64,
            height: 1,
            depth: 1,
        };

        // Scalar params (must match the MSL kernel's `constant uint&` bindings).
        let blocks64: u32 = w.blocks64 as u32;
        let groups_per_row: u32 = w.groups_per_row as u32;
        let n: u32 = w.n as u32;
        let m: u32 = w.m as u32;
        let p_tokens: u32 = p as u32;

        // Dispatch on the shared Metal queue.
        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Buffer bindings (match gemm_ternary_metal_tensor.metal):
        //   0: pos_bits_u32 (read, CubeCL pool)
        //   1: neg_bits_u32 (read, CubeCL pool)
        //   2: group_scale_f32 (read, CubeCL pool)
        //   3: input_batch (read, CubeCL pool)
        //   4: output_batch (write, DEDICATED Metal buffer)
        encoder.set_buffer(0, Some(pos_ref), pos_off);
        encoder.set_buffer(1, Some(neg_ref), neg_off);
        encoder.set_buffer(2, Some(scale_ref), scale_off);
        encoder.set_buffer(3, Some(input_ref), input_off);
        encoder.set_buffer(4, Some(&dedicated_output), 0);  // offset 0 — dedicated

        // Scalar bindings (5..9 match the MSL `constant uint&` declarations).
        encoder.set_bytes(5, 4, &blocks64 as *const u32 as *const _);
        encoder.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(7, 4, &n as *const u32 as *const _);
        encoder.set_bytes(8, 4, &m as *const u32 as *const _);
        encoder.set_bytes(9, 4, &p_tokens as *const u32 as *const _);

        // Threadgroup memory for the dequanted weight tile (index 0).
        encoder.set_threadgroup_memory_length(0, THREADGROUP_MEM_BYTES);

        // Declare resource usage so Metal's hazard tracking is correct. The
        // weights + input are read; the output is written. Without this,
        // memory-less hazard tracking on M3 Max can race with CubeCL's
        // in-flight writes to the same buffers.
        encoder.use_resource(pos_ref, MTLResourceUsage::Read);
        encoder.use_resource(neg_ref, MTLResourceUsage::Read);
        encoder.use_resource(scale_ref, MTLResourceUsage::Read);
        encoder.use_resource(input_ref, MTLResourceUsage::Read);
        encoder.use_resource(&dedicated_output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();

        // Copy the kernel's output from the dedicated buffer back to the
        // CubeCL output handle. This is a GPU→GPU copy within the same
        // command buffer — zero host round-trip. The copy is ordered after
        // the compute pass (same encoder sequence).
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

        // NOTE: we do NOT call `cmd_buffer.wait_until_completed()`. The command
        // buffer runs on the shared Metal queue — CubeCL's subsequent
        // dispatches (including the next prefill_project that reads this
        // output) are automatically ordered after it (Metal guarantees
        // in-order execution within one queue). Blocking here would serialize
        // the async pipeline, reintroducing the Issue 656 host-round-trip cost.
        //
        // The `client.flush()` at the TOP of this function ensures the input
        // buffer's producers are already SUBMITTED, so queue ordering makes
        // their writes visible to this kernel; the queue ordering likewise
        // makes the OUTPUT visible to the next consumer. The only caller that
        // needs explicit `wait_until_completed` is a test that reads the
        // buffer via a non-queued path (e.g., `client.read_one` which may map
        // without a GPU sync). Production code never does this within the
        // tick loop.

        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Convenience: get the device pointer (for T4b-style device-equality checks).
    pub fn device_ptr(&self) -> usize {
        self.device as usize
    }

    /// Convenience: get the queue pointer (for diagnostics).
    pub fn queue_ptr(&self) -> usize {
        self.queue as usize
    }
}

impl Drop for MetalTensorZeroCopyGemm {
    fn drop(&mut self) {
        // The raw device + queue pointers are borrowed from the wgpu Arcs
        // (`_wgpu_device` + `_wgpu_queue`). Dropping those Arcs would
        // invalidate the pointers — but Drop runs before the Arcs drop, so
        // the borrow is still valid here. We don't need to release anything
        // (foreign-types doesn't own these borrowed refs).
    }
}
