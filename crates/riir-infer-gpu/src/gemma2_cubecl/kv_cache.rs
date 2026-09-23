//! KV cache types for CubeCL Gemma 2 forward pass.
//!
//! Extracted from `gemma2_cubecl/mod.rs` to keep the main file under the
//! 2048-line guideline. Contains:
//! - `CpuKVCache` — f32 CPU-side KV cache
//! - `GpuKVCache` — GPU-resident pre-allocated KV cache
//! - `CpuKVCacheQ8` — Q8_0 quantized CPU KV cache
//! - `KvStoreCubeCL` / `KvStoreKRopeVCombinedCubeCL` / `KvCompactCubeCL` —
//!   CubeCL kernels for GPU KV store and compact operations

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;

#[cfg(feature = "q8_kv_cache")]
use crate::attention_q8kv_cubecl::Q8KVBuffers;

// ── CPU KV Cache ───────────────────────────────────────────────────

/// CPU-side KV cache for CubeCL attention dispatch.
///
/// Grows linearly with each position. At position `pos`, each layer's
/// cache contains `(pos + 1) * kv_stride` f32 elements.
///
/// Memory: `2 × (pos + 1) × kv_stride × n_layer × 4 bytes`.
/// For Gemma 2 2B at pos=2048: 2 × 2048 × 1024 × 26 × 4 ≈ 438 MB.
#[cfg(feature = "cubecl_runtime")]
pub struct CpuKVCache {
    /// Key cache per layer: `keys[layer].len() = (pos + 1) * kv_stride`.
    pub keys: Vec<Vec<f32>>,
    /// Value cache per layer: `values[layer].len() = (pos + 1) * kv_stride`.
    pub values: Vec<Vec<f32>>,
    /// Stride per position: `n_kv_head * head_dim`.
    pub kv_stride: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl CpuKVCache {
    pub fn new(n_layer: usize, kv_stride: usize) -> Self {
        Self {
            keys: vec![Vec::new(); n_layer],
            values: vec![Vec::new(); n_layer],
            kv_stride,
        }
    }

    /// Store K and V vectors for the current position in the given layer.
    ///
    /// Store K and V vectors for the given position in the given layer.
    ///
    /// Uses indexed writes at `pos * kv_stride` offset, overwriting any
    /// previous data for that position. This handles sequence restarts
    /// correctly (e.g., warmup followed by new sequence at pos=0).
    ///
    /// `k` and `v` must have exactly `kv_stride` elements.
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.kv_stride, "K length mismatch");
        debug_assert_eq!(v.len(), self.kv_stride, "V length mismatch");
        let offset = pos * self.kv_stride;
        let end = offset + self.kv_stride;
        // Resize vectors if needed to accommodate this position
        if end > self.keys[layer].len() {
            self.keys[layer].resize(end, 0.0);
            self.values[layer].resize(end, 0.0);
        }
        self.keys[layer][offset..end].copy_from_slice(k);
        self.values[layer][offset..end].copy_from_slice(v);
    }

    /// Build combined KV buffer for CubeCL attention: `[keys | values]`.
    ///
    /// Layout: `keys(n_pos × kv_stride) || values(n_pos × kv_stride)`.
    ///
    /// Only returns entries for positions `0..n_positions`, ignoring any
    /// stale data from previous sequences that may still be in the buffer.
    pub fn get_combined_kv(&self, layer: usize, n_positions: usize) -> Vec<f32> {
        let kv_len = n_positions * self.kv_stride;
        let keys = &self.keys[layer];
        let values = &self.values[layer];
        let mut combined = Vec::with_capacity(kv_len * 2);
        combined.extend_from_slice(&keys[..kv_len.min(keys.len())]);
        // Pad with zeros if cache doesn't have enough positions yet
        if kv_len > keys.len() {
            combined.resize(kv_len, 0.0);
        }
        combined.extend_from_slice(&values[..kv_len.min(values.len())]);
        if kv_len > values.len() {
            combined.resize(kv_len * 2, 0.0);
        }
        combined
    }

    /// Current number of cached positions for the given layer.
    #[allow(dead_code)]
    pub fn n_positions(&self, layer: usize) -> usize {
        self.keys[layer].len() / self.kv_stride
    }
}

// ── GPU KV Store Kernel ───────────────────────────────────────────

/// CubeCL kernel for storing K/V into a GPU-resident combined KV cache buffer.
///
/// Writes `k_src[kv_stride]` and `v_src[kv_stride]` into `cache` at position
/// offset `pos`. The cache layout is `[keys || values]` matching the attention
/// kernel's expected format:
///
/// ```text
/// cache = [keys(block_size × kv_stride) | values(block_size × kv_stride)]
///         └────── kv_half ──────────────┘
///
/// K write: cache[pos * kv_stride + tid] = k_src[tid]
/// V write: cache[kv_half + pos * kv_stride + tid] = v_src[tid]
/// ```
///
/// # Dispatch
///
/// | CubeDim       | CubeCount                  | Responsibility          |
/// |---------------|----------------------------|-------------------------|
/// | `new_1d(256)` | `(ceil(kv_stride/256),1,1)` | 1 thread per KV element |
///
/// For Gemma 2 2B (kv_stride=1024): 4 workgroups × 256 threads = 1024 threads.
/// Each thread writes 1 K element + 1 V element (2 f32 writes total).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn kv_store_f32(
    k_src: &[f32],
    v_src: &[f32],
    cache: &mut [f32],
    params: &[f32],
) {
    // params[0] = kv_stride, params[1] = pos, params[2] = kv_half
    // Cast f32 → u32 for indexing (CubeCL v0.10 workaround for scalar params)
    let kv_stride = params[0] as u32;
    let pos = params[1] as u32;
    let kv_half = params[2] as u32;

    let tid = ABSOLUTE_POS as u32;

    if tid < kv_stride {
        // Write K at position offset in keys section
        cache[(pos * kv_stride + tid) as usize] = k_src[tid as usize];
        // Write V at position offset in values section
        cache[(kv_half + pos * kv_stride + tid) as usize] = v_src[tid as usize];
    }
}

/// Launcher for the GPU KV store kernel.
///
/// Stores K and V vectors into a pre-allocated GPU-resident KV cache buffer
/// at the given position offset. No CPU↔GPU sync required.
#[cfg(feature = "cubecl_runtime")]
pub struct KvStoreCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl KvStoreCubeCL {
    /// Store K and V into a GPU-resident combined KV cache buffer.
    ///
    /// # Arguments
    ///
    /// * `client` — CubeCL compute client
    /// * `k_src` — GPU handle with `kv_stride` f32 elements (rope'd K)
    /// * `v_src` — GPU handle with `kv_stride` f32 elements (V)
    /// * `cache` — GPU handle with `2 * block_size * kv_stride` f32 elements
    /// * `kv_stride` — Elements per position: `n_kv_head * head_dim`
    /// * `pos` — Current position in the sequence (write offset)
    /// * `block_size` — Maximum sequence length (cache capacity per layer)
    ///
    /// # Safety
    ///
    /// - `k_src` must have at least `kv_stride` f32 elements
    /// - `v_src` must have at least `kv_stride` f32 elements
    /// - `cache` must have at least `2 * block_size * kv_stride` f32 elements
    /// - `pos` must be < `block_size`
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        k_src: Handle,
        v_src: Handle,
        cache: Handle,
        kv_stride: usize,
        pos: usize,
        block_size: usize,
    ) {
        let kv_half = block_size * kv_stride;
        let total_cache_len = kv_half * 2;
        let params: [f32; 3] = [kv_stride as f32, pos as f32, kv_half as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        let n_workgroups = kv_stride.div_ceil(256) as u32;

        // SAFETY: Caller guarantees correct buffer sizes.
        // Edition 2024: explicit unsafe block required inside unsafe fn.
        unsafe {
            kv_store_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_workgroups, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(k_src, kv_stride),
                BufferArg::from_raw_parts(v_src, kv_stride),
                BufferArg::from_raw_parts(cache, total_cache_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// CubeCL kernel for storing K/V into a GPU-resident KV cache buffer,
/// reading K from a separate RoPE'd handle and V from a combined QKV buffer.
///
/// This is used in the fused QKV path where:
/// - K has been extracted with RoPE applied into a separate handle
/// - V is still in the combined `[Q | K | V]` buffer at `v_offset`
///
/// ```text
/// cache = [keys(block_size × kv_stride) | values(block_size × kv_stride)]
///         └────── kv_half ──────────────┘
///
/// K write: cache[pos * kv_stride + tid] = k_rope[tid]
/// V write: cache[kv_half + pos * kv_stride + tid] = combined[v_offset + tid]
/// ```
///
/// # Dispatch
///
/// Same as `kv_store_f32`: `ceil(kv_stride/256)` workgroups × 256 threads.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn kv_store_k_rope_v_combined_f32(
    k_rope: &[f32],
    qkv_combined: &[f32],
    cache: &mut [f32],
    params: &[f32],
) {
    // params[0] = kv_stride, params[1] = pos, params[2] = kv_half
    // params[3] = v_offset (offset into combined buffer where V starts)
    let kv_stride = params[0usize] as u32;
    let pos = params[1usize] as u32;
    let kv_half = params[2usize] as u32;
    let v_offset = params[3usize] as u32;

    let tid = ABSOLUTE_POS as u32;

    if tid < kv_stride {
        // Write K (from separate RoPE'd handle) at position offset in keys section
        cache[(pos * kv_stride + tid) as usize] = k_rope[tid as usize];
        // Write V (from combined buffer) at position offset in values section
        cache[(kv_half + pos * kv_stride + tid) as usize] = qkv_combined[(v_offset + tid) as usize];
    }
}

/// Launcher for GPU KV store: K from separate RoPE'd handle, V from combined buffer.
///
/// Used by the fused QKV GEMV path where K is RoPE'd separately but V
/// remains in the combined `[Q | K | V]` buffer.
#[cfg(feature = "cubecl_runtime")]
pub struct KvStoreKRopeVCombinedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl KvStoreKRopeVCombinedCubeCL {
    /// Store K (RoPE'd, separate handle) and V (from combined buffer) into GPU KV cache.
    ///
    /// # Safety
    ///
    /// - `k_rope` must have at least `kv_stride` f32 elements
    /// - `qkv_combined` must have at least `v_offset + kv_stride` f32 elements
    /// - `cache` must have at least `2 * block_size * kv_stride` f32 elements
    /// - `pos` must be < `block_size`
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        k_rope: Handle,
        kv_stride_len: usize,
        qkv_combined: Handle,
        combined_len: usize,
        cache: Handle,
        kv_stride: usize,
        pos: usize,
        block_size: usize,
        q_dim: usize,
    ) {
        let kv_half = block_size * kv_stride;
        let total_cache_len = kv_half * 2;
        let v_offset = q_dim + kv_stride; // V starts after Q and K sections
        let params: [f32; 4] = [
            kv_stride as f32,
            pos as f32,
            kv_half as f32,
            v_offset as f32,
        ];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        let n_workgroups = kv_stride.div_ceil(256) as u32;

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            kv_store_k_rope_v_combined_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_workgroups, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(k_rope, kv_stride_len),
                BufferArg::from_raw_parts(qkv_combined, combined_len),
                BufferArg::from_raw_parts(cache, total_cache_len),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

// ── GPU KV Compact Kernel ──────────────────────────────────────────

/// CubeCL kernel for converting strided GPU KV cache to compact format for attention.
///
/// The GPU KV cache uses strided layout: `[keys(block_size × kv_stride) | values(block_size × kv_stride)]`.
/// The attention kernel expects compact layout: `[keys(n_pos × kv_stride) | values(n_pos × kv_stride)]`.
///
/// This kernel copies the first `n_positions` keys and values from the strided cache
/// into a compact buffer:
///
/// ```text
/// Strided: [K₀..K_{bs-1} | V₀..V_{bs-1}]   (bs = block_size)
///           └── kv_half ──┘
///
/// Compact: [K₀..K_{np-1} | V₀..V_{np-1}]   (np = n_positions)
///           └─ np*stride ─┘
/// ```
///
/// Each thread copies one key element AND one value element (2 reads, 2 writes).
///
/// # Dispatch
///
/// | CubeDim       | CubeCount                          | Responsibility           |
/// |---------------|------------------------------------|--------------------------|
/// | `new_1d(256)` | `(ceil(n_pos*stride/256), 1, 1)`  | 1 thread per KV position |
///
/// For Gemma 2 2B at pos=2048 (n_pos=2049, stride=1024): ~8K workgroups.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn kv_compact_f32(src: &[f32], dst: &mut [f32], params: &[f32]) {
    // params[0] = kv_stride, params[1] = n_positions, params[2] = block_size
    let kv_stride = params[0] as u32;
    let n_positions = params[1] as u32;
    let block_size = params[2] as u32;

    let src_kv_half = block_size * kv_stride;
    let dst_kv_half = n_positions * kv_stride;

    let tid = ABSOLUTE_POS as u32;

    // Each thread copies one key element and one value element
    if tid < n_positions * kv_stride {
        // Keys: contiguous in both layouts, offset 0
        dst[tid as usize] = src[tid as usize];
        // Values: from strided offset to compact offset
        dst[(dst_kv_half + tid) as usize] = src[(src_kv_half + tid) as usize];
    }
}

/// Launcher for the GPU KV compact kernel.
///
/// Converts a strided KV cache into the compact format expected by the
/// attention kernel. No CPU↔GPU sync required.
#[cfg(feature = "cubecl_runtime")]
pub struct KvCompactCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl KvCompactCubeCL {
    /// Convert strided KV cache to compact format for attention.
    ///
    /// # Arguments
    ///
    /// * `client` — CubeCL compute client
    /// * `src` — Strided cache handle (`2 * block_size * kv_stride` f32 elements)
    /// * `dst` — Compact output handle (`2 * n_positions * kv_stride` f32 elements)
    /// * `kv_stride` — Elements per position: `n_kv_head * head_dim`
    /// * `n_positions` — Number of valid positions (pos + 1)
    /// * `block_size` — Maximum sequence length (cache capacity)
    ///
    /// # Safety
    ///
    /// - `src` must have at least `2 * block_size * kv_stride` f32 elements
    /// - `dst` must have at least `2 * n_positions * kv_stride` f32 elements
    /// - `n_positions` must be <= `block_size`
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src: Handle,
        dst: Handle,
        kv_stride: usize,
        n_positions: usize,
        block_size: usize,
    ) {
        let params: [f32; 3] = [kv_stride as f32, n_positions as f32, block_size as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        let copy_elements = n_positions * kv_stride;
        let n_workgroups = copy_elements.div_ceil(256) as u32;

        let src_len = 2 * block_size * kv_stride;
        let dst_len = 2 * n_positions * kv_stride;

        // SAFETY: Caller guarantees correct buffer sizes.
        // Edition 2024: explicit unsafe block required inside unsafe fn.
        unsafe {
            kv_compact_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_workgroups, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(src, src_len),
                BufferArg::from_raw_parts(dst, dst_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ── GPU-Resident KV Cache ──────────────────────────────────────────

/// GPU-resident KV cache — pre-allocated CubeCL handles per layer.
///
/// Each layer gets a combined buffer in strided layout:
///
/// ```text
/// cache[layer] = [keys(block_size × kv_stride) | values(block_size × kv_stride)]
///                └──────── kv_half ─────────────┘
/// ```
///
/// The attention kernel expects compact layout:
/// ```text
/// compact = [keys(n_pos × kv_stride) | values(n_pos × kv_stride)]
/// ```
///
/// A `kv_compact_f32` kernel converts strided → compact before each attention call.
/// The `compact_temp` buffer is reused across layers (sequential processing).
///
/// # Memory (Gemma 2 2B)
///
/// - `kv_stride = n_kv_head × head_dim = 4 × 256 = 1024`
/// - `block_size = 8192` (max sequence length)
/// - Per layer cache: `2 × 8192 × 1024 × 4 = 64 MB`
/// - 26 layer caches: `26 × 64 MB = 1.66 GB`
/// - 1 compact temp: `64 MB` (reused across layers)
/// - **Total: ~1.72 GB**
///
/// This is allocated once at construction and reused for the entire
/// generation session. No CPU↔GPU sync needed for KV cache operations.
///
/// # Zero-Init
///
/// Buffers are zero-initialized via `client.empty()`, so attention
/// scores for uninitialized positions will be zero (effectively -∞
/// after softmax, causing no attention to empty slots).
#[cfg(feature = "cubecl_runtime")]
pub struct GpuKVCache {
    /// Combined KV cache handle per layer (strided layout).
    /// Each handle has `2 * block_size * kv_stride` f32 elements.
    caches: Vec<Handle>,
    /// Temp buffer for compact KV view (reused across layers).
    /// Pre-allocated at max size: `2 * block_size * kv_stride` f32 elements.
    /// After `kv_compact_f32`, contains `[keys(n_pos × stride) | values(n_pos × stride)]`.
    compact_temp: Handle,
    /// Stride per position: `n_kv_head * head_dim`.
    kv_stride: usize,
    /// Maximum sequence length (cache capacity per layer).
    block_size: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuKVCache {
    /// Create a new GPU-resident KV cache.
    ///
    /// Allocates `n_layer` combined KV cache buffers on the GPU, each
    /// sized for `block_size` positions with `kv_stride` elements per position.
    /// Also allocates one compact temp buffer (reused across layers).
    pub fn new(
        client: &ComputeClient<ActiveRuntime>,
        n_layer: usize,
        kv_stride: usize,
        block_size: usize,
    ) -> Self {
        let cache_elements = 2 * block_size * kv_stride;
        let cache_bytes = cache_elements * core::mem::size_of::<f32>();
        let caches = (0..n_layer)
            .map(|i| {
                log::info!(
                    "GPU KV cache L{i}: {cache_elements} f32 ({:.1} MB)",
                    cache_bytes as f64 / (1024.0 * 1024.0)
                );
                client.empty(cache_bytes)
            })
            .collect();

        let compact_temp = client.empty(cache_bytes);
        log::info!(
            "GPU KV compact temp: {cache_elements} f32 ({:.1} MB)",
            cache_bytes as f64 / (1024.0 * 1024.0)
        );

        Self {
            caches,
            compact_temp,
            kv_stride,
            block_size,
        }
    }

    /// Get the cache handle for a specific layer (for KV store writes).
    pub fn cache_handle(&self, layer: usize) -> &Handle {
        &self.caches[layer]
    }

    /// Produce a compact KV view for attention by running the compact kernel.
    ///
    /// Returns `(compact_handle, n_positions)` where:
    /// - `compact_handle` is a binding into `compact_temp` **trimmed to the
    ///   live range** `2 × n_positions × kv_stride`
    /// - `n_positions = pos + 1`
    ///
    /// The caller passes these to the attention kernel which expects
    /// `[keys(n_pos × stride) | values(n_pos × stride)]` layout.
    ///
    /// # The trim is load-bearing, not an optimization
    ///
    /// `compact_temp` is allocated once for `block_size` positions and reused
    /// across layers, so it is almost always LARGER than the live range. Both
    /// attention kernels derive
    ///
    /// ```text
    /// n_positions = kv.len() / 2 / kv_stride
    /// ```
    ///
    /// and `kv.len()` in the generated shader is the size of the **bound
    /// buffer** — `BufferArg::from_raw_parts`'s `length` metadata does not
    /// reach it. Handing the kernel the untrimmed handle therefore makes it
    /// believe there are `block_size` positions: it reads the value half from
    /// `block_size × kv_stride` instead of `n_positions × kv_stride`, lands in
    /// never-written memory, and writes an **identically zero** attention
    /// output — no panic, no NaN, just a silently dead attention block.
    ///
    /// Measured 2026-09-05 (riir-train `.issues/511`): the GPU-resident
    /// training forward produced `attn_out == 0` at every layer and position
    /// while K and V in the cache were correct. Trimming with `offset_end`
    /// reproduces the exact-size buffer bit-identically.
    ///
    /// The trim is computed from the handle's own size rather than from
    /// `block_size` so that a pool over-allocation cannot reintroduce the bug.
    ///
    /// # Safety
    ///
    /// Caller must ensure `pos < block_size`.
    pub unsafe fn compact_for_attention(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        layer_idx: usize,
        pos: usize,
    ) -> (Handle, usize) {
        let n_positions = pos + 1;
        // SAFETY: Caller ensures pos < block_size.
        // Edition 2024: explicit unsafe block required inside unsafe fn.
        unsafe {
            KvCompactCubeCL::launch::<ActiveRuntime>(
                client,
                self.caches[layer_idx].clone(),
                self.compact_temp.clone(),
                self.kv_stride,
                n_positions,
                self.block_size,
            );
        }
        let live_bytes =
            (2 * n_positions * self.kv_stride * core::mem::size_of::<f32>()) as u64;
        let trim_bytes = self.compact_temp.size().saturating_sub(live_bytes);
        (self.compact_temp.clone().offset_end(trim_bytes), n_positions)
    }

    /// KV stride: elements per position per K or V.
    #[inline]
    pub fn kv_stride(&self) -> usize {
        self.kv_stride
    }

    /// Maximum sequence length (cache capacity).
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

// ── CPU KV Cache (Q8_0 quantized) ──────────────────────────────────

/// Q8_0 quantized CPU-side KV cache for CubeCL attention dispatch.
///
/// Stores K/V in Q8_0 format (symmetric 8-bit quantization) for ~3.5×
/// memory reduction vs f32 KV cache. Inline dequantization happens in
/// the CubeCL attention kernel during Q·K scoring and value accumulation.
///
/// # Memory Layout
///
/// Each Q8_0 block: 32 values → 32 bytes (packed i8) + 4 bytes (f32 scale).
/// Per position per kv_head (head_dim=256): 8 blocks × (8 u32 + 1 f32).
///
/// ```text
/// f32 cache:   2 × (pos+1) × kv_stride × n_layer × 4 bytes
/// Q8_0 cache:  2 × (pos+1) × (kv_q8_stride×4 + kv_scale_stride×4) × n_layer
/// ```
///
/// For Gemma 2 2B at pos=2048:
/// - f32:   2 × 2048 × 1024 × 26 × 4 ≈ 438 MB
/// - Q8_0:  2 × 2048 × (256+32) × 26 × 4 ≈ 123 MB (3.56× reduction)
#[cfg(feature = "q8_kv_cache")]
pub struct CpuKVCacheQ8 {
    /// Key cache: packed i8 values as u32 words (8 u32s per Q8_0 block).
    /// Per layer: grows by `kv_q8_stride` u32s per position.
    keys_qs: Vec<Vec<u32>>,
    /// Key cache: per-block f32 scales (CPU pre-decoded f16→f32).
    /// Per layer: grows by `kv_scale_stride` f32s per position.
    keys_scales: Vec<Vec<f32>>,
    /// Value cache: packed i8 values as u32 words (same layout as keys_qs).
    values_qs: Vec<Vec<u32>>,
    /// Value cache: per-block f32 scales (same layout as keys_scales).
    values_scales: Vec<Vec<f32>>,
    /// Stride per position for qs: `n_kv_head × (head_dim / Q8_BLOCK_SIZE) × 8`.
    /// For Gemma 2 2B: 4 × 8 × 8 = 256 u32s.
    #[allow(dead_code)] // Used by methods, not directly accessed
    kv_q8_stride: usize,
    /// Stride per position for scales: `n_kv_head × (head_dim / Q8_BLOCK_SIZE)`.
    /// For Gemma 2 2B: 4 × 8 = 32 f32s.
    kv_scale_stride: usize,
    /// Number of KV heads.
    n_kv_head: usize,
    /// Head dimension (must be multiple of 32).
    head_dim: usize,
}

/// Q8_0 block size: 32 values per block.
#[cfg(feature = "q8_kv_cache")]
const Q8_BLOCK_SIZE: usize = 32;

#[cfg(feature = "q8_kv_cache")]
impl CpuKVCacheQ8 {
    /// Create a new Q8_0 KV cache.
    ///
    /// `head_dim` must be a multiple of 32 (Q8_0 block size).
    pub fn new(n_layer: usize, n_kv_head: usize, head_dim: usize) -> Self {
        assert!(
            head_dim.is_multiple_of(Q8_BLOCK_SIZE),
            "head_dim ({head_dim}) must be multiple of 32 (Q8_0 block size)"
        );
        let blocks_per_head = head_dim / Q8_BLOCK_SIZE;
        Self {
            keys_qs: vec![Vec::new(); n_layer],
            keys_scales: vec![Vec::new(); n_layer],
            values_qs: vec![Vec::new(); n_layer],
            values_scales: vec![Vec::new(); n_layer],
            kv_q8_stride: n_kv_head * blocks_per_head * 8, // 8 u32s per block (32 bytes)
            kv_scale_stride: n_kv_head * blocks_per_head,  // 1 f32 per block
            n_kv_head,
            head_dim,
        }
    }

    /// Quantize and store K, V vectors for the current position in the given layer.
    ///
    /// `k` and `v` must each have `n_kv_head × head_dim` elements.
    /// TODO: Use indexed writes at `pos * stride` offset (same fix as CpuKVCache).
    pub fn store(&mut self, layer: usize, _pos: usize, k: &[f32], v: &[f32]) {
        let kv_stride = self.n_kv_head * self.head_dim;
        debug_assert_eq!(k.len(), kv_stride, "K length mismatch");
        debug_assert_eq!(v.len(), kv_stride, "V length mismatch");

        let (k_qs, k_scales) = quantize_row_to_q8_packed(k);
        let (v_qs, v_scales) = quantize_row_to_q8_packed(v);

        self.keys_qs[layer].extend_from_slice(&k_qs);
        self.keys_scales[layer].extend_from_slice(&k_scales);
        self.values_qs[layer].extend_from_slice(&v_qs);
        self.values_scales[layer].extend_from_slice(&v_scales);
    }

    /// Build combined Q8_0 KV buffers for CubeCL attention dispatch.
    ///
    /// Returns `Q8KVBuffers` with combined layout:
    /// - `kv_qs`: `[keys_qs(n_pos × kv_q8_stride) || values_qs(n_pos × kv_q8_stride)]`
    /// - `kv_scales`: `[keys_scales(n_pos × kv_scale_stride) || values_scales(n_pos × kv_scale_stride)]`
    pub fn get_combined_q8kv(&self, layer: usize) -> Q8KVBuffers {
        let keys_qs = &self.keys_qs[layer];
        let values_qs = &self.values_qs[layer];
        let keys_scales = &self.keys_scales[layer];
        let values_scales = &self.values_scales[layer];

        let mut kv_qs = Vec::with_capacity(keys_qs.len() + values_qs.len());
        kv_qs.extend_from_slice(keys_qs);
        kv_qs.extend_from_slice(values_qs);

        let mut kv_scales = Vec::with_capacity(keys_scales.len() + values_scales.len());
        kv_scales.extend_from_slice(keys_scales);
        kv_scales.extend_from_slice(values_scales);

        // Issue 716: production q8 cache has no sink sidecar — empty by default
        // (the guard is opt-in via `Q8KVBuffers::quantize_kv_with_sink`).
        Q8KVBuffers {
            kv_qs,
            kv_scales,
            sink_kv: Vec::new(),
        }
    }

    /// Current number of cached positions for the given layer.
    #[allow(dead_code)]
    pub fn n_positions(&self, layer: usize) -> usize {
        self.keys_scales[layer].len() / self.kv_scale_stride
    }
}

/// Quantize a row of f32 values to packed Q8_0 format.
///
/// Returns `(qs: Vec<u32>, scales: Vec<f32>)` where:
/// - `qs`: packed i8 values, 8 u32s per block (32 bytes / 4)
/// - `scales`: one f32 scale per block (symmetric: `scale = max_abs / 127`)
///
/// Matches the GPU-side dequantization in `attention_decode_q8kv`.
#[cfg(feature = "q8_kv_cache")]
fn quantize_row_to_q8_packed(src: &[f32]) -> (Vec<u32>, Vec<f32>) {
    assert!(
        src.len().is_multiple_of(Q8_BLOCK_SIZE),
        "src length must be multiple of 32 (Q8_0 block size)"
    );
    let n_blocks = src.len() / Q8_BLOCK_SIZE;
    let mut qs = Vec::with_capacity(n_blocks * 8);
    let mut scales = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        let base = block_idx * Q8_BLOCK_SIZE;
        let block = &src[base..base + Q8_BLOCK_SIZE];

        // Symmetric quantization: scale = max_abs / 127
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        scales.push(scale);

        // Quantize each value to i8, pack 4 per u32 (little-endian)
        let mut words = [0u32; 8];
        for (i, &v) in block.iter().enumerate() {
            let q = if scale > 0.0 {
                (v / scale).round().clamp(-127.0, 127.0) as i8
            } else {
                0i8
            };
            let word_idx = i / 4;
            let byte_idx = i % 4;
            words[word_idx] |= (q as u8 as u32) << (byte_idx * 8);
        }
        qs.extend_from_slice(&words);
    }

    (qs, scales)
}
