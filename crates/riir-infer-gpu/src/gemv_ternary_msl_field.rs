//! Issue 1138 — the hand-MSL FFN decode-GEMV lane (i1137 reopen lever (a),
//! adjudicated GO 2026-10-27).
//!
//! Raw-Metal dispatch of the PR-26980 field algebra mapped onto our pos/neg
//! plane format. The mechanism, the byte layout, and the fork geometry are
//! documented in the paired `gemv_ternary_msl_field.metal`; the i1137
//! mapping study names the mechanism and the prize arithmetic (~53%-of-stream
//! ≈ +13% e2e; even 45% ≈ +8%, both orders over the +1.5% promote bar).
//!
//! Format: byte `c = (p − n) + 2 ∈ {1,2,3}` packed 4 weights/byte, sequential
//! (byte b ↔ values 4b..4b+3); `w = c − 2` value-exact; the (1,1) plane alias
//! encodes as c = 2 (also w = 0). Group scales ride as raw f16 bits — the
//! same values the incumbent f16scale arm dispatches.
//!
//! Dispatch gates (refusal is the caller's fallback signal, the Issue-1124
//! posture-gate shape): `cols % 256 == 0 && rows % 8 == 0`. Both league FFN
//! shapes and lm_head satisfy them.
//!
//! G1 class: NOT to_bits (per-half scale placement + accumulation order
//! differ from the incumbent) — the pre-authorized Bench-742 law governs:
//! unit max-abs band vs the shipping f16scale kernel + greedy-token
//! loop-stream identity e2e.
//!
//! Toggle: `RIIR_GEMV_TQ2_FIELD=1` arms (default OFF), `=0` is the demote
//! hatch; malformed values refuse LOUD. One-shot `[tq2_field] REACH:` line
//! on first dispatch (the Bench-768 vacuous-guard law).
//!
//! Two dispatch postures:
//! - **managed** (`new`/`new_with_shared_queue` + `enqueue`): own device +
//!   queue + managed weight buffers — the aspect-instrument posture
//!   (bench_1148).
//! - **zero-copy** (`new_zero_copy` + `dispatch_zero_copy`): the Issue-663
//!   T5 interop — shared CubeCL device + queue (raw pointers extracted once,
//!   kept valid by the wgpu Arcs; never wrapped in owning handles — the
//!   raw_device() pointer is borrowed, and an owning wrapper would
//!   over-release), weights uploaded as shared-mode buffers ONCE, y/out ride
//!   the CubeCL pool buffers (extracted fresh per dispatch — the pool can
//!   reallocate), dedicated shared-mode output + blit back (the
//!   pool-aliasing lesson), `client.flush()` before the raw commit (the
//!   Issue-670 T4 submission ordering), NO wait (shared-queue in-order
//!   execution). The e2e posture.

#![cfg(all(feature = "metal_tensor_gemm", feature = "ternary_gemv", target_os = "macos"))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use metal::foreign_types::ForeignTypeRef;
use metal::{
    Buffer, BufferRef, CommandQueue, CommandQueueRef, CompileOptions, ComputePipelineState, Device,
    DeviceRef, Library, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use wgpu::{Device as WgpuDevice, Queue as WgpuQueue};

use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use katgpt_core::TernaryGroupWeights;

use crate::cubecl_runtime::ActiveRuntime;

/// The raw MSL kernel (single source of truth for the algebra + layout docs).
const MSL_SOURCE: &str = include_str!("gemv_ternary_msl_field.metal");

// ---------------------------------------------------------------------------
// The packed field format (CPU side)
// ---------------------------------------------------------------------------

/// A ternary weight matrix repacked into the 4-per-byte field layout.
#[derive(Clone, Debug)]
pub struct TernaryFieldWeights {
    /// `[rows * cols/4]` packed field bytes, row-major, sequential.
    pub qbytes: Vec<u8>,
    /// `[rows * cols/128]` raw f16 group-scale bits, row-major (unchanged
    /// values from [`TernaryGroupWeights::group_scale`]).
    pub gscales_u16: Vec<u16>,
    pub rows: usize,
    pub cols: usize,
}

impl TernaryFieldWeights {
    /// Repack from the pos/neg plane format. Refuses misaligned geometries
    /// (the kernel has no bounds guards by design — dispatch gates instead).
    pub fn from_weights(w: &TernaryGroupWeights) -> Result<Self, String> {
        if !w.cols.is_multiple_of(256) {
            return Err(format!(
                "gemv_ternary_msl_field: cols {} not a multiple of 256 — refuse (posture gate)",
                w.cols
            ));
        }
        if !w.rows.is_multiple_of(8) {
            return Err(format!(
                "gemv_ternary_msl_field: rows {} not a multiple of 8 — refuse (posture gate)",
                w.rows
            ));
        }
        let row_bytes = w.cols / 4;
        let mut qbytes = vec![0u8; w.rows * row_bytes];
        for r in 0..w.rows {
            let row_base = r * row_bytes;
            for k in 0..w.blocks64 {
                let pos = w.pos_bits[r * w.blocks64 + k];
                let neg = w.neg_bits[r * w.blocks64 + k];
                // 64 weights → 16 bytes, 4 weights/byte, sequential.
                for i in 0..64usize {
                    let p = ((pos >> i) & 1) as u8;
                    let n = ((neg >> i) & 1) as u8;
                    let c = p + 2 - n; // (p − n) + 2 ∈ {1,2,3}; (1,1) → 2 = w 0
                    let idx = row_base + k * 16 + (i >> 2);
                    qbytes[idx] |= c << ((i & 3) * 2);
                }
            }
        }
        // f16 bits verbatim (bytemuck, the in-tree idiom — no half dep here).
        let gscales_u16 = bytemuck::cast_slice(&w.group_scale).to_vec();
        debug_assert_eq!(gscales_u16.len(), w.rows * w.groups_per_row);
        Ok(Self {
            qbytes,
            gscales_u16,
            rows: w.rows,
            cols: w.cols,
        })
    }

    /// Repack the gate/up CONCAT view (row-concat `[gate; up]`, the
    /// `TernaryHandle::from_two_weights` layout the decode FFN dispatches —
    /// SwiGLU reads gate rows `[0..mlp)`, up rows `[mlp..2mlp)`). Row-major
    /// planes concat by plain append, so the merged view is allocation-only.
    pub fn from_two_row_concat(
        gate: &TernaryGroupWeights,
        up: &TernaryGroupWeights,
    ) -> Result<Self, String> {
        assert_eq!(
            gate.cols, up.cols,
            "from_two_row_concat: gate/up col mismatch"
        );
        let mut merged = TernaryGroupWeights::new(gate.rows + up.rows, gate.cols);
        merged.pos_bits[..gate.pos_bits.len()].copy_from_slice(&gate.pos_bits);
        merged.pos_bits[gate.pos_bits.len()..].copy_from_slice(&up.pos_bits);
        merged.neg_bits[..gate.neg_bits.len()].copy_from_slice(&gate.neg_bits);
        merged.neg_bits[gate.neg_bits.len()..].copy_from_slice(&up.neg_bits);
        merged.group_scale[..gate.group_scale.len()].copy_from_slice(&gate.group_scale);
        merged.group_scale[gate.group_scale.len()..].copy_from_slice(&up.group_scale);
        Self::from_weights(&merged)
    }
}

// ---------------------------------------------------------------------------
// Toggle + REACH counter (the 768 vacuous-guard law)
// ---------------------------------------------------------------------------

/// 0 = unset (env decides), 1 = armed, 2 = demote hatch.
static FIELD_TOGGLE: AtomicU8 = AtomicU8::new(0);
static FIELD_LAUNCH_COUNT: AtomicUsize = AtomicUsize::new(0);

static FIELD_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn parse_field_env() -> bool {
    match std::env::var("RIIR_GEMV_TQ2_FIELD") {
        Ok(v) if v == "1" => true,
        Ok(v) if v == "0" => false,
        Ok(v) => {
            eprintln!(
                "[i1138] RIIR_GEMV_TQ2_FIELD={v:?} malformed (want \"0\" or \"1\"); default OFF"
            );
            false
        }
        Err(_) => false,
    }
}

/// The dispatch gate: explicit setter wins over `RIIR_GEMV_TQ2_FIELD`
/// (operator intent beats the env, the Issue-1122 precedence); default OFF.
pub fn gemv_use_field() -> bool {
    match FIELD_TOGGLE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => *FIELD_ENV.get_or_init(parse_field_env),
    }
}

/// Arm / demote the field lane (the explicit setter arm).
pub fn set_gemv_use_field(on: bool) {
    FIELD_TOGGLE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Dispatches actually issued through the field lane. The benches assert
/// this is 0 before any armed window (a silently no-op toggle must be
/// CAUGHT, not read as zero-effect) and that OFF arms add ZERO.
pub fn field_launch_count() -> usize {
    FIELD_LAUNCH_COUNT.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// The Metal context
// ---------------------------------------------------------------------------

enum Posture {
    /// Owned device + queue, managed weight buffers (the aspect rig).
    Managed { device: Device, queue: CommandQueue },
    /// The Issue-663 T5 interop: raw shared device/queue pointers (kept
    /// valid by the wgpu Arcs; NEVER wrapped in owning handles — the
    /// raw_device() pointer is borrowed, and an owning wrapper would
    /// over-release on drop).
    ZeroCopy(ZeroCopyState),
}

struct ZeroCopyState {
    /// Raw `id<MTLDevice>` — borrowed from the wgpu Arc below.
    device_ptr: *mut metal::MTLDevice,
    /// Raw `id<MTLCommandQueue>` — the queue CubeCL submits to; ordering
    /// with CubeCL dispatches is automatic (in-order queue).
    queue_ptr: *mut metal::MTLCommandQueue,
    /// Dedicated output buffers keyed by byte size — CubeCL sub-allocates
    /// all handles inside one pool `id<MTLBuffer>`; writing the kernel
    /// output straight into a pool alias of the weights/input silently
    /// fails hazard tracking. Write dedicated, blit back (one GPU→GPU copy).
    output_cache: Mutex<HashMap<u64, Buffer>>,
    /// Hold the wgpu device + queue Arcs alive — the raw pointers point
    /// into their internals.
    _wgpu_device: Arc<WgpuDevice>,
    _wgpu_queue: Arc<WgpuQueue>,
}

/// Compiled field-GEMM context for one matrix.
pub struct GemvTernaryFieldMsl {
    pipeline: ComputePipelineState,
    _library: Library,
    weights: TernaryFieldWeights,
    qs_buf: Buffer,
    gscales_buf: Buffer,
    posture: Posture,
    /// Dispatches through this context (both postures; profiling).
    dispatch_count: AtomicUsize,
}

// SAFETY: the raw `id` pointers are Objective-C object pointers; Metal
// command-queue submission is thread-safe. The metal-rs handle types are
// Send + Sync (the in-tree MetalTensorGemm/zero-copy precedents store the
// same set in shared contexts). `Handle` is Send+Sync per the CubeCL
// zero-copy precedent's SAFETY note.
unsafe impl Send for GemvTernaryFieldMsl {}
unsafe impl Sync for GemvTernaryFieldMsl {}

impl GemvTernaryFieldMsl {
    /// Managed posture on the system default device + private queue.
    pub fn new(w: &TernaryGroupWeights) -> Result<Self, String> {
        let device = Device::system_default().ok_or("gemv_ternary_msl_field: no Metal device")?;
        Self::new_with_shared_queue(w, device.new_command_queue())
    }

    /// Managed posture on an explicit queue (the aspect rig's shared queue).
    pub fn new_with_shared_queue(
        w: &TernaryGroupWeights,
        queue: CommandQueue,
    ) -> Result<Self, String> {
        let weights = TernaryFieldWeights::from_weights(w)?;
        let device = Device::system_default().ok_or("gemv_ternary_msl_field: no Metal device")?;
        let (library, pipeline) = compile_pipeline(&device)?;
        let (qs_buf, gscales_buf) = upload_weights_managed(&device, &weights);
        Ok(Self {
            pipeline,
            _library: library,
            weights,
            qs_buf,
            gscales_buf,
            posture: Posture::Managed { device, queue },
            dispatch_count: AtomicUsize::new(0),
        })
    }

    /// Zero-copy posture (the Issue-663 T5 interop): compile on the SHARED
    /// CubeCL device, upload weights as shared-mode buffers there, keep the
    /// wgpu Arcs that own the raw device/queue alive.
    pub fn new_zero_copy(
        w: &TernaryGroupWeights,
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
    ) -> Result<Self, String> {
        let weights = TernaryFieldWeights::from_weights(w)?;

        // Extract the raw shared device + queue (the T4 pattern, verbatim
        // from gemm_ternary_metal_zero_copy.rs).
        let device_ptr: *mut metal::MTLDevice = unsafe {
            let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
            let dev = guard
                .as_ref()
                .ok_or("wgpu device is not Metal — the field lane zero-copy requires Metal")?;
            let proto = dev.raw_device();
            (&**proto) as *const _ as *mut metal::MTLDevice
        };
        let queue_ptr: *mut metal::MTLCommandQueue = unsafe {
            let guard = wgpu_queue.as_hal::<wgpu::hal::api::Metal>();
            let q = guard
                .as_ref()
                .ok_or("wgpu queue is not Metal — the field lane zero-copy requires Metal")?;
            let proto = q.as_raw();
            proto as *const _ as *mut metal::MTLCommandQueue
        };

        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
        let (library, pipeline) = compile_pipeline(device_ref)?;
        let qs_buf = device_ref.new_buffer_with_data(
            weights.qbytes.as_ptr() as *const core::ffi::c_void,
            weights.qbytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let gscales_buf = device_ref.new_buffer_with_data(
            weights.gscales_u16.as_ptr() as *const core::ffi::c_void,
            (weights.gscales_u16.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            pipeline,
            _library: library,
            weights,
            qs_buf,
            gscales_buf,
            posture: Posture::ZeroCopy(ZeroCopyState {
                device_ptr,
                queue_ptr,
                output_cache: Mutex::new(HashMap::new()),
                _wgpu_device: wgpu_device,
                _wgpu_queue: wgpu_queue,
            }),
            dispatch_count: AtomicUsize::new(0),
        })
    }

    /// Zero-copy posture for the gate/up CONCAT view (`from_two_row_concat`
    /// + [`Self::new_zero_copy`]).
    pub fn new_zero_copy_concat(
        gate: &TernaryGroupWeights,
        up: &TernaryGroupWeights,
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
    ) -> Result<Self, String> {
        let weights = TernaryFieldWeights::from_two_row_concat(gate, up)?;
        Self::finish_zero_copy(weights, wgpu_device, wgpu_queue)
    }

    /// Zero-copy posture for a plain matrix (shared by the concat path).
    pub fn new_zero_copy_plain(
        w: &TernaryGroupWeights,
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
    ) -> Result<Self, String> {
        let weights = TernaryFieldWeights::from_weights(w)?;
        Self::finish_zero_copy(weights, wgpu_device, wgpu_queue)
    }

    fn finish_zero_copy(
        weights: TernaryFieldWeights,
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
    ) -> Result<Self, String> {
        let device_ptr: *mut metal::MTLDevice = unsafe {
            let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
            let dev = guard
                .as_ref()
                .ok_or("wgpu device is not Metal — the field lane zero-copy requires Metal")?;
            let proto = dev.raw_device();
            (&**proto) as *const _ as *mut metal::MTLDevice
        };
        let queue_ptr: *mut metal::MTLCommandQueue = unsafe {
            let guard = wgpu_queue.as_hal::<wgpu::hal::api::Metal>();
            let q = guard
                .as_ref()
                .ok_or("wgpu queue is not Metal — the field lane zero-copy requires Metal")?;
            let proto = q.as_raw();
            proto as *const _ as *mut metal::MTLCommandQueue
        };
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
        let (library, pipeline) = compile_pipeline(device_ref)?;
        let qs_buf = device_ref.new_buffer_with_data(
            weights.qbytes.as_ptr() as *const core::ffi::c_void,
            weights.qbytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let gscales_buf = device_ref.new_buffer_with_data(
            weights.gscales_u16.as_ptr() as *const core::ffi::c_void,
            (weights.gscales_u16.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            pipeline,
            _library: library,
            weights,
            qs_buf,
            gscales_buf,
            posture: Posture::ZeroCopy(ZeroCopyState {
                device_ptr,
                queue_ptr,
                output_cache: Mutex::new(HashMap::new()),
                _wgpu_device: wgpu_device,
                _wgpu_queue: wgpu_queue,
            }),
            dispatch_count: AtomicUsize::new(0),
        })
    }

    /// Matrix rows (output dim).
    pub fn rows(&self) -> usize {
        self.weights.rows
    }

    /// Matrix cols (input dim).
    pub fn cols(&self) -> usize {
        self.weights.cols
    }

    /// Upload bytes for the field format (parity assertion input: 4
    /// weights/byte + 2 bytes/scale-group vs 2 bit-planes + f16 scales —
    /// both are `rows*cols/4 + rows*(cols/128)*2`).
    pub fn upload_bytes(&self) -> u64 {
        self.qs_buf.length() + self.gscales_buf.length()
    }

    /// Create a managed f32 device buffer from host data (y vectors).
    pub fn buffer_from_f32(&self, data: &[f32]) -> Buffer {
        let Posture::Managed { device, .. } = &self.posture else {
            panic!("buffer_from_f32: managed posture only");
        };
        device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    /// Create a zeroed managed f32 output buffer.
    pub fn output_buffer(&self, len: usize) -> Buffer {
        let Posture::Managed { device, .. } = &self.posture else {
            panic!("output_buffer: managed posture only");
        };
        device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeManaged)
    }

    /// Read an f32 buffer back to host (call after [`Self::sync`]).
    pub fn read_f32(&self, buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    /// Enqueue ONE decode GEMV dispatch (`out = weights @ y`) on this
    /// context's managed queue and commit. NO sync — chains pipeline; sync
    /// with [`Self::sync`] (one wait at chain end) before reading.
    ///
    /// Grid = rows/8 threadgroups × 64 threads (the fork geometry). The
    /// posture gate was enforced at [`TernaryFieldWeights::from_weights`].
    pub fn enqueue(&self, y: &Buffer, out: &Buffer) {
        let count = FIELD_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        if count == 0 {
            eprintln!("[tq2_field] REACH: armed (first field-lane dispatch, i1138)");
        }

        let Posture::Managed { queue, .. } = &self.posture else {
            panic!("enqueue: managed posture only (zero-copy uses dispatch_zero_copy)");
        };
        let rows = self.weights.rows as u32;
        let cols = self.weights.cols as u32;

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_field_dispatch(
            enc,
            &self.pipeline,
            &self.qs_buf,
            &self.gscales_buf,
            (y as &BufferRef, 0),
            out,
            rows,
            cols,
        );
        enc.end_encoding();
        cmd.commit();
    }

    /// Block until every enqueued dispatch on this context's queue completed.
    pub fn sync(&self) {
        let Posture::Managed { queue, .. } = &self.posture else {
            panic!("sync: managed posture only");
        };
        let cmd = queue.new_command_buffer();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Zero-copy dispatch (the e2e posture): y read from the CubeCL pool
    /// handle, output written to a dedicated shared-mode buffer and blitted
    /// back into the CubeCL output handle — one GPU→GPU copy, no host
    /// round-trip, no staging of the weights. Ordered against CubeCL's own
    /// dispatches by the shared in-order queue (`client.flush()` first —
    /// the Issue-670 T4 law: producers must be SUBMITTED before the raw
    /// commit enqueues ahead of them).
    pub fn dispatch_zero_copy(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        input: &Handle,
        output: &Handle,
    ) -> Result<(), String> {
        let Posture::ZeroCopy(zc) = &self.posture else {
            return Err("dispatch_zero_copy: not a zero-copy context".to_string());
        };

        let count = FIELD_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        if count == 0 {
            eprintln!("[tq2_field] REACH: armed (first zero-copy field dispatch, i1138)");
        }

        // Issue-670 T4: submit CubeCL's pending work BEFORE our raw commit
        // (and settle pool bookkeeping before extraction walks it).
        let _ = client.flush();

        let (input_ptr, input_off) = extract_raw_buffer(client, input)?;
        let (output_ptr, output_off) = extract_raw_buffer(client, output)?;

        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let cubecl_output_ref: &BufferRef = unsafe { BufferRef::from_ptr(output_ptr) };

        // Dedicated output (shared mode), cached by byte size.
        let output_bytes = (self.weights.rows * 4) as u64;
        let dedicated_output = {
            let mut cache = zc.output_cache.lock().unwrap();
            cache
                .entry(output_bytes)
                .or_insert_with(|| {
                    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(zc.device_ptr) };
                    device_ref.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared)
                })
                .clone()
        };

        let rows = self.weights.rows as u32;
        let cols = self.weights.cols as u32;

        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(zc.queue_ptr) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let enc = cmd_buffer.new_compute_command_encoder();
        encode_field_dispatch(
            enc,
            &self.pipeline,
            &self.qs_buf,
            &self.gscales_buf,
            (input_ref, input_off),
            &dedicated_output,
            rows,
            cols,
        );
        // Hazard tracking: the pool buffer is read; the dedicated output is
        // written (the Issue-663 memory-less-hazard lesson).
        enc.use_resource(input_ref, MTLResourceUsage::Read);
        enc.use_resource(&dedicated_output, MTLResourceUsage::Write);
        enc.end_encoding();

        // Blit the kernel output into the CubeCL output handle — ordered
        // after the compute pass (same command buffer). The next CubeCL
        // consumer reads it via normal queue ordering; NO wait here (the
        // Issue-656 law: blocking would serialize the async pipeline).
        let blit = cmd_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &dedicated_output,
            0,
            cubecl_output_ref,
            output_off,
            output_bytes,
        );
        blit.end_encoding();

        cmd_buffer.commit();
        Ok(())
    }

    /// Dispatches through this context (profiling).
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count.load(Ordering::Relaxed)
    }
}

/// Shared encoder body for both postures (bindings match
/// `gemv_ternary_msl_field.metal`: 0 qs, 1 gscales, 2 y, 3 out, 4/5 scalars).
fn encode_field_dispatch(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    qs: &Buffer,
    gscales: &Buffer,
    y: (&BufferRef, u64),
    out: &Buffer,
    rows: u32,
    cols: u32,
) {
    let (y_ref, y_off) = y;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(qs), 0);
    enc.set_buffer(1, Some(gscales), 0);
    enc.set_buffer(2, Some(y_ref), y_off);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(4, 4, rows.to_ne_bytes().as_ptr() as *const _);
    enc.set_bytes(5, 4, cols.to_ne_bytes().as_ptr() as *const _);
    let grid = MTLSize {
        width: (rows / 8) as u64,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, threadgroup);
}

fn compile_pipeline(device: &DeviceRef) -> Result<(Library, ComputePipelineState), String> {
    let library = device
        .new_library_with_source(MSL_SOURCE, &CompileOptions::new())
        .map_err(|e| format!("gemv_ternary_msl_field: MSL compile failed: {e}"))?;
    let function = library
        .get_function("gemv_ternary_field_f32", None)
        .map_err(|e| format!("gemv_ternary_msl_field: function not found: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("gemv_ternary_msl_field: pipeline failed: {e}"))?;
    Ok((library, pipeline))
}

fn upload_weights_managed(device: &Device, weights: &TernaryFieldWeights) -> (Buffer, Buffer) {
    let qs_buf = device.new_buffer_with_data(
        weights.qbytes.as_ptr() as *const core::ffi::c_void,
        weights.qbytes.len() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let gscales_buf = device.new_buffer_with_data(
        weights.gscales_u16.as_ptr() as *const core::ffi::c_void,
        (weights.gscales_u16.len() * 2) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    (qs_buf, gscales_buf)
}

/// Extract `(raw id<MTLBuffer>, offset)` from a CubeCL handle — verbatim
/// the Issue-663 T5 mechanic (fresh per dispatch: the pool can reallocate).
fn extract_raw_buffer(
    client: &ComputeClient<ActiveRuntime>,
    handle: &Handle,
) -> Result<(*mut metal::MTLBuffer, u64), String> {
    let managed = client
        .get_resource(handle.clone())
        .map_err(|e| format!("get_resource failed: {e:?}"))?;
    let resource = managed.resource();
    let offset = resource.offset;
    let wgpu_buffer = &resource.buffer;
    let id_ptr = unsafe {
        let guard = wgpu_buffer.as_hal::<wgpu::hal::api::Metal>();
        let hal_buf = guard
            .as_ref()
            .ok_or("CubeCL buffer is not Metal — the field lane zero-copy requires Metal")?;
        let proto = hal_buf.raw_handle();
        proto as *const _ as *mut metal::MTLBuffer
    };
    Ok((id_ptr, offset))
}

// ---------------------------------------------------------------------------
// The per-model cache bundle (the forward's armed construction)
// ---------------------------------------------------------------------------

/// Field contexts for the decode FFN pair + lm_head, built ONCE at model
/// load when [`gemv_use_field`] is armed at construction (the env/setter is
/// read-once — the f16-scale posture; a mid-run flip cannot conjure caches
/// that were never built, and the vacuous counter catches the no-op).
///
/// Memory cost when armed: one field copy of the FFN + lm_head weights
/// (≈ rows*cols/4 bytes each) — ~2.9 GB on the league 27B posture. That is
/// the opt-in price of the lane until promotion retires the plane buffers.
pub struct FieldFfnCaches {
    /// Per-layer `gate_up_proj` (row-concat `[gate; up]`, m = 2·mlp).
    /// `None` = that matrix refused the posture gate (fail-open per matrix).
    pub gate_up: Vec<Option<Arc<GemvTernaryFieldMsl>>>,
    /// Per-layer `down_proj` (`None` = refused).
    pub down: Vec<Option<Arc<GemvTernaryFieldMsl>>>,
    /// The lm_head tail GEMV (`None` = refused or not supplied).
    pub lm_head: Option<Arc<GemvTernaryFieldMsl>>,
}

/// One layer's projection views for [`FieldFfnCaches::build`] — the caller
/// maps its CPU layer type to `&TernaryGroupWeights` views (`None` = not
/// ternary / not available → the lane demotes that matrix).
pub type FieldLayerViews<'a> = (
    Option<&'a TernaryGroupWeights>,
    Option<&'a TernaryGroupWeights>,
    Option<&'a TernaryGroupWeights>,
);

impl FieldFfnCaches {
    pub fn build<'a>(
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
        layers: impl Iterator<Item = FieldLayerViews<'a>>,
        lm_head: Option<&TernaryGroupWeights>,
    ) -> Self {
        let mut gate_up = Vec::new();
        let mut down = Vec::new();
        for (g, u, d) in layers {
            gate_up.push(match (g, u) {
                (Some(g), Some(u)) => GemvTernaryFieldMsl::new_zero_copy_concat(
                    g,
                    u,
                    wgpu_device.clone(),
                    wgpu_queue.clone(),
                )
                .ok()
                .map(Arc::new),
                _ => None,
            });
            down.push(match d {
                Some(d) => {
                    GemvTernaryFieldMsl::new_zero_copy_plain(
                        d,
                        wgpu_device.clone(),
                        wgpu_queue.clone(),
                    )
                    .ok()
                    .map(Arc::new)
                }
                None => None,
            });
        }
        let lm_head_ctx = lm_head.and_then(|w| {
            GemvTernaryFieldMsl::new_zero_copy_plain(w, wgpu_device.clone(), wgpu_queue.clone())
                .ok()
                .map(Arc::new)
        });
        Self {
            gate_up,
            down,
            lm_head: lm_head_ctx,
        }
    }
}
