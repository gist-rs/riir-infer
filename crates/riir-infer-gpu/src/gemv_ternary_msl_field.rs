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
//! unit max-abs band vs the shipping f16scale kernel.
//!
//! Toggle: `RIIR_GEMV_TQ2_FIELD=1` arms (default OFF), `=0` is the demote
//! hatch; malformed values refuse LOUD. One-shot `[tq2_field] REACH:` line
//! on first dispatch (the Bench-768 vacuous-guard law).
//!
//! This cycle ships the aspect-level instrument (G1 + the Bench-1141-method
//! A/B); the e2e zero-copy integration rides the Issue-663 T5 interop
//! pattern AFTER the aspect direction is measured on this box.

#![cfg(all(feature = "metal_tensor_gemm", feature = "ternary_gemv", target_os = "macos"))]

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

use katgpt_core::TernaryGroupWeights;

/// The raw MSL kernel (single source of truth for the algebra + layout docs).
const MSL_SOURCE: &str = include_str!("gemv_ternary_msl_field.metal");

// ---------------------------------------------------------------------------
// The packed field format (CPU side)
// ---------------------------------------------------------------------------

/// A ternary weight matrix repacked into the 4-per-byte field layout.
///
/// Owned by the Metal context ([`GemvTernaryFieldMsl`]); kept public for the
/// bench instruments that want to assert upload-byte parity.
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

/// Dispatches actually issued through [`GemvTernaryFieldMsl::enqueue`].
/// The bench asserts this is 0 before any armed window (a silently no-op
/// toggle must be CAUGHT, not read as zero-effect).
pub fn field_launch_count() -> usize {
    FIELD_LAUNCH_COUNT.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// The Metal context
// ---------------------------------------------------------------------------

/// Device + shared queue + compiled pipeline + uploaded field weights for
/// one matrix. One instance per weight; the cold-rotation rigs share ONE
/// queue across instances so the kernels serialize exactly like the CubeCL
/// arm's single queue (a private queue per instance would let cold-set
/// kernels overlap and distort the aspect curve).
pub struct GemvTernaryFieldMsl {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    _library: Library,
    weights: TernaryFieldWeights,
    qs_buf: Buffer,
    gscales_buf: Buffer,
}

impl GemvTernaryFieldMsl {
    /// Repack, compile the MSL at init time, upload the weight buffers on a
    /// PRIVATE queue (fine for single-context use; use
    /// [`Self::new_with_shared_queue`] when several contexts must
    /// serialize).
    pub fn new(w: &TernaryGroupWeights) -> Result<Self, String> {
        let device = Device::system_default().ok_or("gemv_ternary_msl_field: no Metal device")?;
        Self::new_with_shared_queue(w, device.new_command_queue())
    }

    /// Same, but dispatches ride the GIVEN queue (the bench's shared-queue
    /// posture).
    pub fn new_with_shared_queue(w: &TernaryGroupWeights, queue: CommandQueue) -> Result<Self, String> {
        let weights = TernaryFieldWeights::from_weights(w)?;
        let device = Device::system_default().ok_or("gemv_ternary_msl_field: no Metal device")?;
        let library = device
            .new_library_with_source(MSL_SOURCE, &CompileOptions::new())
            .map_err(|e| format!("gemv_ternary_msl_field: MSL compile failed: {e}"))?;
        let function = library
            .get_function("gemv_ternary_field_f32", None)
            .map_err(|e| format!("gemv_ternary_msl_field: function not found: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("gemv_ternary_msl_field: pipeline failed: {e}"))?;

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

        Ok(Self {
            device,
            queue,
            pipeline,
            _library: library,
            weights,
            qs_buf,
            gscales_buf,
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
        self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    /// Create a zeroed managed f32 output buffer.
    pub fn output_buffer(&self, len: usize) -> Buffer {
        self.device
            .new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeManaged)
    }

    /// Read an f32 buffer back to host (call after [`Self::sync`]).
    pub fn read_f32(&self, buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    /// Enqueue ONE decode GEMV dispatch (`out = weights @ y`) on this
    /// context's queue and commit. NO sync — chains pipeline; sync with
    /// [`Self::sync`] (one wait at chain end) before reading.
    ///
    /// Grid = rows/8 threadgroups × 64 threads (the fork geometry). The
    /// posture gate was enforced at [`TernaryFieldWeights::from_weights`].
    pub fn enqueue(&self, y: &Buffer, out: &Buffer) {
        let count = FIELD_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
        if count == 0 {
            eprintln!("[tq2_field] REACH: armed (first field-lane dispatch, i1138)");
        }

        let rows = self.weights.rows as u32;
        let cols = self.weights.cols as u32;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&self.qs_buf), 0);
        enc.set_buffer(1, Some(&self.gscales_buf), 0);
        enc.set_buffer(2, Some(y), 0);
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
        enc.end_encoding();
        cmd.commit();
    }

    /// Block until every enqueued dispatch on this context's queue completed.
    pub fn sync(&self) {
        let cmd = self.queue.new_command_buffer();
        cmd.commit();
        cmd.wait_until_completed();
    }
}
