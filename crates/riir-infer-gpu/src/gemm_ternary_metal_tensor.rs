//! Raw Metal ternary GEMM using `metal::tensor` / `mpp::tensor_ops::matmul2d`.
//!
//! Issue 656 — **REFUTED end-to-end** (host-round-trip 0.87× regression,
//! Bench 666 interleaved re-measurement). The kernel itself is 3.72× faster
//! than simdgroup cmma in isolation (Bench 645), but the host-round-trip
//! integration (CubeCL → host read → Metal upload → launch → read → CubeCL
//! re-upload) forces 2 blocking syncs per projection × ~448 projections per
//! prefill, serializing the async pipeline. The wgpu-passthrough integration
//! (Issue 657) also failed at 0.95× (per-dispatch staging copy + command
//! buffer overhead). **Simdgroup cmma is the optimal GEMM kernel on M3 Max
//! Metal.** This module is kept as a negative-result artifact — the wiring
//! is correct, the kernel works, but no integration path captures the
//! kernel gain on M3 Max Metal.
//!
//! # Why the kernel is fast but the integration is slow
//!
//! The matmul2d kernel bypasses CubeCL to use Metal's cooperative-tensor
//! matmul API (`mpp::tensor_ops::matmul2d`) with 64×128 output tiles, 4
//! simdgroups — the same path llama.cpp uses. CubeCL's `cmma::execute` maps
//! to `simdgroup_matrix_8x8` (8×8×8 tiles, 1 simdgroup). But the per-projection
//! integration overhead (blocking fence + host/GPU data transfers for
//! host-round-trip, or staging copies + per-dispatch command buffer for
//! wgpu-passthrough) exceeds the kernel-level savings. Closing this gap
//! would require running the ENTIRE forward pass in raw Metal (eliminating
//! the CubeCL↔Metal switches entirely) — a major architectural change,
//! not an integration fix (see Issue 651 §"Update 2026-08-13").
//!
//! # Architecture
//!
//! - **A-matrix** (weights): dequanted from ternary bit-plane format into a
//!   threadgroup `half` staging buffer, then fed to `matmul2d` as a tensor.
//! - **B-matrix** (activations): read directly from device memory via
//!   `metal::tensor::tensor` — no shared-memory staging.
//! - **Tile**: 64 rows × 128 tokens output, 32-element K-tile, 4 simdgroups
//!   per workgroup (128 threads). Matches llama.cpp's proven dispatch.
//!
//! # References
//!
//! - [mikeroyal/Metal-Guide](https://github.com/mikeroyal/Metal-Guide) — comprehensive Metal dev guide
//! - [apple/metal-cpp](https://github.com/apple/metal-cpp) — Apple's official C++ wrapper for Metal framework
//! - [doom-fish/apple-metal-rs](https://github.com/doom-fish/apple-metal-rs) — alternative Rust Metal bindings
//! - llama.cpp reference: `prismml-llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` L9861 (`kernel_mul_mm`)
//! - wgpu MSL passthrough: `wgpu-hal-30.0.0/src/metal/device.rs` L1304 (`ShaderInput::Msl`)
//!
//! # Safety
//!
//! macOS-only (metal-rs). The raw MSL kernel is compiled at init time via
//! `device.new_library_with_source()`. Buffer bindings must match the kernel's
//! `device` buffer declarations. G1 correctness is validated against the CPU
//! reference in the test harness.

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use metal::{
    Buffer, CommandQueue, CompileOptions, Device, Library,
    MTLResourceOptions, MTLSize, ComputePipelineState,
};

// ---------------------------------------------------------------------------
// MSL kernel source — the raw Metal shader
// ---------------------------------------------------------------------------

/// The raw MSL source for the ternary GEMM kernel.
///
/// Currently a scalar reference kernel (1 thread per output element) that
/// validates the metal-rs pipeline end-to-end. The matmul2d-accelerated
/// version will replace the inner loop with `mpp::tensor_ops::matmul2d`.
const MSL_SOURCE: &str = include_str!("gemm_ternary_metal_tensor.metal");

// ---------------------------------------------------------------------------
// Ternary bit-plane format constants (must match gemv_ternary_cubecl.rs)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Ternary bit-plane format constants (must match gemv_ternary_cubecl.rs)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tile dimensions — currently 1 thread per output element (scalar reference).
// Will be increased once the matmul2d version is validated.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Metal tensor GEMM context
// ---------------------------------------------------------------------------

/// A Metal device + command queue + compiled kernel pipeline for the
/// ternary tensor GEMM.
pub struct MetalTensorGemm {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    _library: Library,
    /// Scratch buffer cache (T8 optimization). Keyed by byte size; reused
    /// across calls to avoid per-projection `new_buffer` allocation overhead.
    /// Stores both input + output scratch buffers.
    scratch: std::sync::Mutex<std::collections::HashMap<u64, Buffer>>,
}

impl MetalTensorGemm {
    /// Create a new Metal tensor GEMM context on the default Metal device.
    ///
    /// Compiles the raw MSL kernel at init time. Returns an error if the
    /// Metal device doesn't support `metal::tensor` (requires Metal 3.1+,
    /// macOS 14+).
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        // Check for metal::tensor support (MSL 3.1+, macOS 14+)
        // The compile will fail if the header isn't available.
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SOURCE, &options)
            .map_err(|e| format!("MSL compilation failed: {e}"))?;

        let function_name = "gemm_ternary_tensor";
        let function = library
            .get_function(function_name, None)
            .map_err(|e| format!("Function '{function_name}' not found: {e}"))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline creation failed: {e}"))?;

        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            pipeline,
            _library: library,
            scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Get or create a scratch buffer of `byte_len` bytes, reused across calls.
    /// Avoids per-projection Metal buffer allocation (T8 optimization).
    fn scratch_buffer(&self, byte_len: u64) -> Buffer {
        let mut cache = self.scratch.lock().unwrap();
        if let Some(buf) = cache.remove(&byte_len) {
            return buf;
        }
        self.device.new_buffer(
            byte_len,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    /// Upload ternary weights to Metal buffers.
    ///
    /// Returns `(pos_buf, neg_buf, scale_buf)` — three Metal buffers holding
    /// the bit-plane data, ready for the GEMM kernel.
    pub fn upload_weights(
        &self,
        pos_bits_u32: &[u32],
        neg_bits_u32: &[u32],
        group_scale_f32: &[f32],
    ) -> (Buffer, Buffer, Buffer) {
        let pos = self.new_buffer_from_slice(pos_bits_u32);
        let neg = self.new_buffer_from_slice(neg_bits_u32);
        let scale = self.new_buffer_from_slice(group_scale_f32);
        (pos, neg, scale)
    }

    fn new_buffer_from_slice<T>(&self, data: &[T]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            (std::mem::size_of_val(data)) as u64,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    /// Upload f32 activations to a Metal buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Buffer {
        self.new_buffer_from_slice(data)
    }

    /// Upload f32 activations to a reused scratch buffer (T8).
    /// Copies data into a cached buffer, avoiding per-call allocation.
    pub fn upload_f32_scratch(&self, data: &[f32]) -> Buffer {
        let byte_len = (data.len() * 4) as u64;
        let buf = self.scratch_buffer(byte_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                byte_len as usize,
            );
        }
        buf.did_modify_range(metal::NSRange::new(0, byte_len));
        buf
    }

    /// Create a zeroed output buffer.
    pub fn create_output(&self, len: usize) -> Buffer {
        self.device.new_buffer(
            (len * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    /// Get a reused output buffer of `len` f32 elements (T8).
    pub fn create_output_scratch(&self, len: usize) -> Buffer {
        let byte_len = (len * 4) as u64;
        self.scratch_buffer(byte_len)
    }

    /// Read an f32 buffer back to the host.
    ///
    /// # Safety
    ///
    /// The caller must ensure `buf` was synchronized (the command buffer
    /// committed + completed) before calling this. Managed-memory buffers
    /// require `didModifyRange` on the CPU side or `synchronize` on the GPU
    /// side; for our write-once weights + single-launch output buffers, the
    /// `wait_until_completed()` in `launch()` provides the sync fence.
    pub fn read_f32(&self, buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    /// Launch the ternary tensor GEMM.
    ///
    /// Computes `output[P × m] = dequant_ternary(weight[m × n]) @ input[P × n]^T`
    /// using the `metal::tensor` matmul2d API.
    ///
    /// Dispatch: 64×128 output tiles, 4 simdgroups (128 threads) per workgroup.
    /// Matches llama.cpp's `kernel_mul_mm` tile geometry. Edge tiles that exceed
    /// the matrix bounds are handled by the matmul2d tensor extents (bounds
    /// checking is built into the cooperative-tensor store/load).
    ///
    /// # Arguments
    /// - `pos_buf`, `neg_buf`, `scale_buf`: ternary weight buffers from `upload_weights`
    /// - `input_buf`: f32 activation buffer `[P × n]` row-major
    /// - `output_buf`: f32 output buffer `[P × m]` row-major (will be written)
    /// - `blocks64`: number of 64-element blocks per weight row
    /// - `groups_per_row`: number of 128-element scale groups per weight row
    /// - `n`: weight input dimension (K in GEMM terms)
    /// - `m`: weight output dimension (M in GEMM terms)
    /// - `p`: number of tokens (N in GEMM terms)
    pub fn launch(
        &self,
        pos_buf: &Buffer,
        neg_buf: &Buffer,
        scale_buf: &Buffer,
        input_buf: &Buffer,
        output_buf: &Buffer,
        blocks64: u32,
        groups_per_row: u32,
        n: u32,
        m: u32,
        p: u32,
    ) {
        // matmul2d tile constants (must match the MSL kernel)
        const NRA: u32 = 64;      // features (M) per workgroup
        const NRB: u32 = 128;     // tokens (P) per workgroup
        const NSG: u32 = 4;       // simdgroups per workgroup
        const THREADS_PER_SG: u32 = 32;
        const NUM_THREADS: u32 = THREADS_PER_SG * NSG;  // 128

        // Threadgroup memory for the dequanted A tile: K_TILE × NRA × sizeof(half)
        // K_TILE = 32, NRA = 64, sizeof(half) = 2 → 4096 bytes.
        // Must be explicitly allocated via set_threadgroup_memory_length — an
        // unsized `threadgroup char*` in the kernel defaults to zero bytes.
        const THREADGROUP_MEM_BYTES: u64 = 32 * 64 * 2;  // 4096

        // Grid: one workgroup per output tile
        let grid = MTLSize {
            width: p.div_ceil(NRB) as u64,   // token tiles
            height: m.div_ceil(NRA) as u64,  // feature tiles
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: NUM_THREADS as u64,
            height: 1,
            depth: 1,
        };

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Buffer bindings (must match the MSL kernel's buffer declarations)
        encoder.set_buffer(0, Some(pos_buf), 0);
        encoder.set_buffer(1, Some(neg_buf), 0);
        encoder.set_buffer(2, Some(scale_buf), 0);
        encoder.set_buffer(3, Some(input_buf), 0);
        encoder.set_buffer(4, Some(output_buf), 0);

        // Scalar bindings (set_bytes takes NSUInteger = u64)
        encoder.set_bytes(5, 4, blocks64.to_ne_bytes().as_ptr() as *const _);
        encoder.set_bytes(6, 4, groups_per_row.to_ne_bytes().as_ptr() as *const _);
        encoder.set_bytes(7, 4, n.to_ne_bytes().as_ptr() as *const _);
        encoder.set_bytes(8, 4, m.to_ne_bytes().as_ptr() as *const _);
        encoder.set_bytes(9, 4, p.to_ne_bytes().as_ptr() as *const _);

        // Allocate threadgroup memory for the dequanted weight tile (index 0 = [[threadgroup(0)]])
        encoder.set_threadgroup_memory_length(0, THREADGROUP_MEM_BYTES);

        encoder.use_resource(pos_buf, metal::MTLResourceUsage::Read);
        encoder.use_resource(neg_buf, metal::MTLResourceUsage::Read);
        encoder.use_resource(scale_buf, metal::MTLResourceUsage::Read);
        encoder.use_resource(input_buf, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buf, metal::MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }
}

// ---------------------------------------------------------------------------
// Metal weight cache — the Metal-side counterpart of a `TernaryHandle`
// ---------------------------------------------------------------------------

/// Three Metal buffers holding the ternary bit-plane data for one weight
/// matrix, pre-uploaded at model load time.
///
/// This is the Metal-side mirror of `TernaryHandle`'s three CubeCL handles.
/// Stored as `Option<MetalWeightCache>` on `TernaryHandle`; `None` when the
/// `metal_tensor_gemm` feature is off or on non-macOS.
#[derive(Clone)]
pub struct MetalWeightCache {
    /// Positive bit-plane buffer (u32, layout `[rows * blocks64 * 2]`).
    pub pos: Buffer,
    /// Negative bit-plane buffer (u32, layout `[rows * blocks64 * 2]`).
    pub neg: Buffer,
    /// Group scales buffer (f32, layout `[rows * groups_per_row]`).
    pub scale: Buffer,
}

impl MetalWeightCache {
    /// Upload ternary weight data to Metal buffers.
    ///
    /// The three slices must match the bit-plane layout used by
    /// [`TernaryHandle`]: pos/neg as u32 (u64→2×u32 cast), scales as f32
    /// (f16→f32 decoded).
    pub fn upload(
        gemm: &MetalTensorGemm,
        pos_bits_u32: &[u32],
        neg_bits_u32: &[u32],
        group_scale_f32: &[f32],
    ) -> Self {
        let (pos, neg, scale) = gemm.upload_weights(pos_bits_u32, neg_bits_u32, group_scale_f32);
        Self { pos, neg, scale }
    }
}
