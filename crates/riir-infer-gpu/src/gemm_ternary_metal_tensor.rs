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
//! # Calibration sweep (Issue 1113 lever-3 cell 2, 2026-10-24)
//!
//! The M5-Ultra prefill defense re-opened this kernel as the calibration
//! instrument: [`MetalTileConfig`] + [`MetalTensorGemm::new_with_config`]
//! compiles the SAME kernel at any tile geometry (the SWEEP_* define guards in
//! the .metal), so the per-SKU tile calibration the research note demands
//! (`.research/400` — "measure the smem/simdgroup GEMM shapes against the M5
//! GPU's limits FIRST, then design") runs against the live kernel instead of
//! a hard-coded 64×128. The M3-era e2e verdicts above were taken at P=128;
//! they do NOT answer the pp2048+ postures (per-projection overhead amortizes
//! ~16× differently) and they do not answer a different box.
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
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
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
// Tile configuration (Issue 1113 lever-3 calibration cell 2)
// ---------------------------------------------------------------------------

/// Compile-time tile geometry of the matmul2d kernel.
///
/// The default is the fork geometry (`kernel_mul_mm`: 64×128 output tile,
/// K-tile 32, 4 simdgroups). [`MetalTensorGemm::new_with_config`] compiles a
/// pipeline with any other geometry by prepending `SWEEP_*` defines to the
/// MSL source — see the guard block in `gemm_ternary_metal_tensor.metal`.
/// The sweep instrument (`bench_1113_prefill_tensor_sweep`) walks a config
/// list; configs the mpp framework refuses surface as a `COMPILE_FAIL` row,
/// never a panic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalTileConfig {
    /// Output features (M) per workgroup. smem cost: `K_TILE * nra * 2` bytes.
    pub nra: u32,
    /// Output tokens (N) per workgroup.
    pub nrb: u32,
    /// K-tile = `16 * nk_chunk` (SZ_SIMD = 16 is fixed by the hardware).
    pub nk_chunk: u32,
    /// Simdgroups per workgroup; threads = `32 * nsg`.
    pub nsg: u32,
}

impl Default for MetalTileConfig {
    fn default() -> Self {
        Self {
            nra: 64,
            nrb: 128,
            nk_chunk: 2,
            nsg: 4,
        }
    }
}

impl MetalTileConfig {
    /// The `#define` preamble for this config, prepended to the MSL source.
    /// Pub for the integration paths ([`crate::gemm_ternary_metal_zero_copy`]
    /// compiles the same source at the same config).
    pub fn defines(self) -> String {
        let Self {
            nra,
            nrb,
            nk_chunk,
            nsg,
        } = self;
        format!(
            "#define SWEEP_NRA {nra}\n#define SWEEP_NRB {nrb}\n#define SWEEP_NK_CHUNK {nk_chunk}\n#define SWEEP_NSG {nsg}\n"
        )
    }

    /// Threadgroup memory the dequanted A tile needs: `K_TILE * NRA` halfs.
    pub fn threadgroup_mem_bytes(self) -> u64 {
        let k_tile = 16 * self.nk_chunk;
        u64::from(k_tile * self.nra) * 2
    }

    /// Issue 1122 gate 2 — the Bench-1113 cell-2 sweep winner
    /// (`ktile64_64x128x4sg`: K-tile 64, 64×128 output tile, 4 simdgroups),
    /// leading at 3.02–3.30× per-launch over the incumbent K32 geometry. The
    /// Bench-998 integration cell ran the DEFAULT geometry only — this const
    /// is the integration target its closure never executed.
    pub const KTILE64: Self = Self {
        nra: 64,
        nrb: 128,
        nk_chunk: 4,
        nsg: 4,
    };
}

// ---------------------------------------------------------------------------
// Issue 1122 gate 2 — the production-geometry selector
// ---------------------------------------------------------------------------
// The integration paths (`MetalTensorZeroCopyGemm::new` + the forward's
// metal-rs context) compiled the kernel at the built-in K32 defaults only —
// the swept winner was unreachable from any e2e cell, which is exactly the
// gap Issue 1122 filed. One packed global + one env arm selects the geometry
// BEFORE context construction (the pipeline is compile-time; a mid-run
// change cannot re-geometry an existing context).

/// Packed geometry: `nra<<48 | nrb<<32 | nk_chunk<<16 | nsg` — 0 = unset
/// (fields are ≥ 1, so 0 is not a value).
static PROD_GEOM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn pack_geom(c: MetalTileConfig) -> u64 {
    (u64::from(c.nra) << 48)
        | (u64::from(c.nrb) << 32)
        | (u64::from(c.nk_chunk) << 16)
        | u64::from(c.nsg)
}

fn unpack_geom(v: u64) -> MetalTileConfig {
    MetalTileConfig {
        nra: (v >> 48) as u32,
        nrb: ((v >> 32) & 0xFFFF) as u32,
        nk_chunk: ((v >> 16) & 0xFFFF) as u32,
        nsg: (v & 0xFFFF) as u32,
    }
}

/// Pin the production geometry for every context constructed AFTER this call.
/// Explicit setter wins over the env arm (operator intent beats default
/// derivation); contexts already compiled are unaffected by construction.
pub fn set_metal_tensor_production_geometry(config: MetalTileConfig) {
    PROD_GEOM.store(pack_geom(config), std::sync::atomic::Ordering::Relaxed);
}

/// The `RIIR_METAL_TENSOR_GEOM` env arm, parsed once: `"nra,nrb,nk_chunk,nsg"`
/// (e.g. `"64,128,4,4"` = [`MetalTileConfig::KTILE64`], the promoted default;
/// `"64,128,2,4"` = the K32 demote). Malformed values
/// refuse LOUD at first read and are ignored — a silently-wrong geometry
/// would corrupt every number the cell records.
static PROD_GEOM_ENV: std::sync::OnceLock<Option<MetalTileConfig>> = std::sync::OnceLock::new();

/// The pure parse half of the env arm (unit-testable without touching the
/// process env or the [`PROD_GEOM_ENV`] OnceLock; pub so the engagement
/// cell can assert the selector honored the armed spec). `None` = malformed;
/// the caller prints the loud refusal naming the spec.
pub fn parse_prod_geom_spec(spec: &str) -> Option<MetalTileConfig> {
    let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return None;
    }
    let parse = |s: &str| s.parse::<u32>().ok().filter(|v| *v >= 1);
    match (
        parse(parts[0]),
        parse(parts[1]),
        parse(parts[2]),
        parse(parts[3]),
    ) {
        (Some(nra), Some(nrb), Some(nk_chunk), Some(nsg)) => Some(MetalTileConfig {
            nra,
            nrb,
            nk_chunk,
            nsg,
        }),
        _ => None,
    }
}

fn parse_prod_geom_env() -> Option<MetalTileConfig> {
    let spec = std::env::var("RIIR_METAL_TENSOR_GEOM").ok()?;
    let parsed = parse_prod_geom_spec(&spec);
    match parsed {
        Some(c) => {
            eprintln!(
                "[Issue 1122] production geometry = {}x{}, K-tile {}, {} simdgroups",
                c.nra,
                c.nrb,
                16 * c.nk_chunk,
                c.nsg
            );
            Some(c)
        }
        None => {
            eprintln!(
                "[Issue 1122] RIIR_METAL_TENSOR_GEOM={spec:?} malformed (want \"nra,nrb,nk_chunk,nsg\", every field >= 1); using built-in defaults"
            );
            None
        }
    }
}

/// The geometry the next integration context compiles with: the explicit
/// setter if pinned, else `RIIR_METAL_TENSOR_GEOM` (parsed once, now the
/// DEMOTE-ONLY arm), else the **PROMOTED ktile64 default**
/// ([`MetalTileConfig::KTILE64`] — Plan 641 T4, delegated owner verdict
/// session `d4135f15` criterion commit riir-ai `48c464e25`, promotion
/// recorded Bench 1009/1010: three replication windows every posture-read
/// ≥1.163× (pp2048 1.223/1.238/1.253, pp16k 1.203/1.217, pp32k 1.163/1.170)
/// plus the M3 pp2048 no-regression cell reading 1.258× — the gain holds on
/// BOTH Metal box classes). Demote to the incumbent K32 byte-path via
/// `RIIR_METAL_TENSOR_GEOM=64,128,2,4`.
pub fn production_geometry() -> MetalTileConfig {
    let pinned = PROD_GEOM.load(std::sync::atomic::Ordering::Relaxed);
    if pinned != 0 {
        return unpack_geom(pinned);
    }
    (*PROD_GEOM_ENV.get_or_init(parse_prod_geom_env)).unwrap_or(MetalTileConfig::KTILE64)
}

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
    /// The tile geometry this instance's pipeline was compiled with — the
    /// launch grid, threadgroup size, and smem length all derive from it.
    config: MetalTileConfig,
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
        Self::new_with_config(MetalTileConfig::default())
    }

    /// Compile the kernel at a custom [`MetalTileConfig`] (the calibration
    /// sweep's entry point; [`new`](Self::new) is this with the fork geometry).
    ///
    /// A geometry the mpp framework refuses fails HERE, at pipeline compile,
    /// with the compiler's message — the caller records it and moves on.
    pub fn new_with_config(config: MetalTileConfig) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        // Check for metal::tensor support (MSL 3.1+, macOS 14+)
        // The compile will fail if the header isn't available.
        let options = CompileOptions::new();
        let source = format!("{}{}", config.defines(), MSL_SOURCE);
        let library = device
            .new_library_with_source(source.as_str(), &options)
            .map_err(|e| format!("MSL compilation failed ({config:?}): {e}"))?;

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
            config,
            scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// The tile geometry this instance was compiled with.
    pub fn config(&self) -> MetalTileConfig {
        self.config
    }

    /// Get or create a scratch buffer of `byte_len` bytes, reused across calls.
    /// Avoids per-projection Metal buffer allocation (T8 optimization).
    fn scratch_buffer(&self, byte_len: u64) -> Buffer {
        let mut cache = self.scratch.lock().unwrap();
        if let Some(buf) = cache.remove(&byte_len) {
            return buf;
        }
        self.device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeManaged)
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
        self.device
            .new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeManaged)
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
    /// Dispatch: one workgroup per 64×128 output tile, 4 simdgroups (128
    /// threads) — or the geometry this instance was compiled with via
    /// [`new_with_config`](Self::new_with_config). Edge tiles that exceed
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
        // Tile geometry derives from the compiled config (the fork geometry
        // reproduces the historical constants exactly).
        let cfg = self.config;
        let nra = cfg.nra;
        let nrb = cfg.nrb;
        let num_threads = u64::from(32 * cfg.nsg);
        let threadgroup_mem_bytes = cfg.threadgroup_mem_bytes();

        // Grid: one workgroup per output tile
        let grid = MTLSize {
            width: p.div_ceil(nrb) as u64,  // token tiles
            height: m.div_ceil(nra) as u64, // feature tiles
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: num_threads,
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
        encoder.set_threadgroup_memory_length(0, threadgroup_mem_bytes);

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

#[cfg(test)]
mod production_geometry_tests {
    use super::*;

    #[test]
    fn env_spec_golden_parses_to_ktile64() {
        assert_eq!(
            parse_prod_geom_spec("64,128,4,4"),
            Some(MetalTileConfig::KTILE64)
        );
    }

    #[test]
    fn env_spec_tolerates_whitespace() {
        assert_eq!(
            parse_prod_geom_spec(" 64, 128 , 4,4 "),
            Some(MetalTileConfig::KTILE64)
        );
    }

    #[test]
    fn env_spec_malformed_refuses_none() {
        // wrong field count
        assert_eq!(parse_prod_geom_spec("64,128,4"), None);
        assert_eq!(parse_prod_geom_spec("64,128,4,4,2"), None);
        assert_eq!(parse_prod_geom_spec(""), None);
        // zero is not a value (fields are >= 1)
        assert_eq!(parse_prod_geom_spec("0,128,4,4"), None);
        assert_eq!(parse_prod_geom_spec("64,0,4,4"), None);
        // non-numeric
        assert_eq!(parse_prod_geom_spec("64,128,x,4"), None);
        assert_eq!(parse_prod_geom_spec("64,128,-4,4"), None);
    }

    #[test]
    fn k_tile_is_16_chunks_and_default_is_ktile64() {
        let kt64 = parse_prod_geom_spec("64,128,4,4").expect("golden spec");
        assert_eq!(16 * kt64.nk_chunk, 64);
        // Plan 641 T4 promotion: the env-unset fallback is the PROMOTED ktile64
        // (verdict session d4135f15, criterion commit riir-ai 48c464e25; three
        // windows + the M3 no-regression cell). The lib test process has no env
        // armed and no setter pinned, so this reads the fallback directly.
        assert_eq!(
            production_geometry(),
            MetalTileConfig::KTILE64,
            "the promoted default is ktile64 (Plan 641 T4); the env is DEMOTE-ONLY"
        );
        // the K32 demote spelling stays reachable and parses to the incumbent config
        assert_eq!(
            parse_prod_geom_spec("64,128,2,4"),
            Some(MetalTileConfig {
                nra: 64,
                nrb: 128,
                nk_chunk: 2,
                nsg: 4
            }),
            "the K32 demote via RIIR_METAL_TENSOR_GEOM stays byte-reachable"
        );
    }
}
