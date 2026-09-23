//! CubeCL f16 weight GEMV kernel for decode-time matrix-vector multiplication (Plan 106 T2.10).
//!
//! Implements `output[M] = weight[M,N] @ input[N]` using CubeCL's `#[cube]` DSL
//! with f16 weight storage for 2× bandwidth savings on weight reads.
//!
//! Weight arrays store `half::f16` elements (2 bytes each); input/output remain f32.
//! Inside the kernel, f16 weights are cast to f32 via `f32::cast_from()` before
//! accumulation — matching the accumulation precision of the f32 GEMV while
//! halving weight memory bandwidth.
//!
//! Two variants:
//! - **Plane (subgroup)**: Cooperative dot product with `plane_sum()` reduction.
//!   Each plane/subgroup handles one output row. Best performance on Metal.
//! - **Shared memory tiling**: Fallback when planes unavailable.
//!   Input vector tiled into shared memory, one thread per output row.
//!
//! # Dispatch
//!
//! | Variant | CubeDim          | CubeCount              | Rows/cube |
//! |---------|------------------|------------------------|-----------|
//! | Plane   | `new_1d(256)`    | `ceil(m / 8), 1, 1`   | 8         |
//! | Tiled   | `new_1d(256)`    | `ceil(m / 256), 1, 1` | 256       |

#[cfg(feature = "cubecl_runtime")]
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use half::f16 as half_f16;

#[cfg(feature = "cubecl_runtime")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Plane (subgroup) GEMV — primary kernel, f16 weights
// ---------------------------------------------------------------------------

/// CubeCL GEMV kernel using plane (subgroup) cooperative dot product with f16 weights.
///
/// Each plane handles one output row. Lanes cooperatively compute the dot product:
/// lane `j` accumulates elements at indices `j, j + PLANE_DIM, j + 2*PLANE_DIM, ...`.
/// `plane_sum()` reduces partial sums — single `simd_sum()` on Metal.
///
/// Weight elements are read as f16 and cast to f32 for accumulation:
/// `partial += f32::cast_from(weight[k]) * input[k]`
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    feature = "gemv_fma_contract",
    cube(launch_unchecked, fast_math = FastMath::AllowContraction.into())
)]
#[cfg_attr(
    not(feature = "gemv_fma_contract"),
    cube(launch_unchecked)
)]
fn gemv_plane_f16_f32(weight: &[half_f16], input: &[f32], output: &mut [f32]) {
    let n = input.len() as u32;
    let m = output.len() as u32;

    // Each plane handles one output row.
    // row = global_thread_index / plane_size
    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let row_offset = row * n;

    // Cooperative dot product: lane j handles elements j, j + PLANE_DIM, ...
    // Adjacent lanes read contiguous weight elements → fully coalesced access.
    let mut partial = f32::new(0.0f32);

    let mut k = lane;
    while k < n {
        // Cast f16 weight → f32 for accumulation
        let w_f32 = f32::cast_from(weight[(row_offset + k) as usize]);
        partial += w_f32 * input[k as usize];
        k += PLANE_DIM;
    }

    // Hardware SIMD reduction: sum all lane partials in the plane.
    let result = plane_sum(partial);

    // Lane 0 writes the final result for this row.
    if lane == 0 {
        output[row as usize] = result;
    }
}

// ---------------------------------------------------------------------------
// Shared memory GEMV — fallback kernel (no subgroups required), f16 weights
// ---------------------------------------------------------------------------

/// CubeCL GEMV kernel using shared memory tiling with f16 weights (no subgroup required).
///
/// Each thread computes one output row (full dot product). The input vector
/// is tiled into shared memory so all threads share the same input tile,
/// improving memory access patterns.
///
/// Weight elements are read as f16 and cast to f32 for accumulation.
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    feature = "gemv_fma_contract",
    cube(launch_unchecked, fast_math = FastMath::AllowContraction.into())
)]
#[cfg_attr(
    not(feature = "gemv_fma_contract"),
    cube(launch_unchecked)
)]
fn gemv_tile_f16_f32(weight: &[half_f16], input: &[f32], output: &mut [f32]) {
    let n = input.len() as u32;
    let m = output.len() as u32;

    // Each thread handles one output row.
    let row = ABSOLUTE_POS as u32;

    if row >= m {
        terminate!();
    }

    let row_offset = row * n;
    let mut sum = f32::new(0.0f32);

    // Shared memory for input vector tile (256 elements = 1 KB).
    let mut tile_input = Shared::<[f32]>::new_slice(256usize);

    let tile_size = 256u32;
    let num_tiles = n.div_ceil(tile_size);

    let mut tile_idx = 0u32;
    while tile_idx < num_tiles {
        let tile_start = tile_idx * tile_size;

        // Cooperatively load tile of input into shared memory.
        // Thread t loads element tile_start + t (or 0 if out of bounds).
        let t = UNIT_POS;
        let input_idx = tile_start + t;
        if input_idx < n {
            tile_input[t as usize] = input[input_idx as usize];
        } else {
            tile_input[t as usize] = f32::new(0.0f32);
        }

        sync_cube();

        // Accumulate partial dot product from this tile.
        let mut k = 0u32;
        while k < tile_size {
            let weight_idx = tile_start + k;
            if weight_idx < n {
                // Cast f16 weight → f32 for accumulation
                let w_f32 = f32::cast_from(weight[(row_offset + weight_idx) as usize]);
                sum += w_f32 * tile_input[k as usize];
            }
            k += 1u32;
        }

        sync_cube();
        tile_idx += 1u32;
    }

    output[row as usize] = sum;
}

// ---------------------------------------------------------------------------
// F16Handle — f16 weight buffer wrapper
// ---------------------------------------------------------------------------

/// GPU buffer handle for f16 weight storage.
///
/// Stores weight matrix as `half::f16` (2 bytes per element) for 2× bandwidth
/// savings vs f32 during GEMV weight reads. Constructed from f32 data via
/// CPU-side conversion before GPU upload.
///
/// # Layout
///
/// Row-major `[m, n]` matrix stored as `Vec<half::f16>` (2 × m × n bytes).
#[cfg(feature = "cubecl_runtime")]
pub struct F16Handle {
    /// GPU buffer with f16 weight data (2 bytes per element).
    pub weight: Handle,
    /// Number of rows (output dimension).
    pub m: usize,
    /// Number of columns (input dimension).
    pub n: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl F16Handle {
    /// Create an `F16Handle` by converting f32 weights to f16 on CPU, then uploading to GPU.
    ///
    /// The conversion happens on CPU to avoid GPU-side f16 overhead during GEMV.
    /// The resulting GPU buffer stores `m * n` f16 elements (2 bytes each).
    pub fn from_f32<R: Runtime>(
        client: &ComputeClient<R>,
        weights_f32: &[f32],
        m: usize,
        n: usize,
    ) -> Self {
        assert_eq!(
            weights_f32.len(),
            m * n,
            "weights_f32 length ({}) must equal m * n ({m} * {n} = {})",
            weights_f32.len(),
            m * n
        );

        // Convert f32 → f16 on CPU
        // Use serial iteration for small weights — rayon task-splitting overhead
        // isn't worth it below ~8K elements. Large weight matrices benefit from
        // parallel conversion since this is potentially millions of elements.
        let weight_f16: Vec<half_f16> = if weights_f32.len() < 8192 {
            weights_f32.iter().map(|&x| half_f16::from_f32(x)).collect()
        } else {
            weights_f32
                .par_iter()
                .map(|&x| half_f16::from_f32(x))
                .collect()
        };

        // Upload f16 bytes to GPU (2 bytes per element)
        let weight_bytes = bytemuck::cast_slice::<half_f16, u8>(&weight_f16);
        let weight = client.create_from_slice(weight_bytes);

        Self { weight, m, n }
    }
}

// ---------------------------------------------------------------------------
// Public API: auto-selecting launcher
// ---------------------------------------------------------------------------

/// CubeCL f16 weight GEMV launcher with automatic plane/shared-memory selection.
///
/// Selects the plane (subgroup) kernel when the device supports it,
/// otherwise falls back to the shared-memory tiled kernel.
///
/// Weight data is stored as f16 (2 bytes per element) for reduced memory
/// bandwidth. Input and output remain f32.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvF16CubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvF16CubeCL {
    /// Launch f16 weight GEMV: `output[M] = weight_f16[M,N] @ input[N]`.
    ///
    /// Auto-selects plane or tiled kernel based on device capabilities.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - weight_handle: M×N f16 elements (2 × M × N bytes)
    /// - input_handle: N f32 elements
    /// - output_handle: M f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let has_plane = client.features().plane.contains(Plane::Ops);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            if has_plane {
                Self::launch_plane::<R>(client, weight_handle, input_handle, output_handle, m, n);
            } else {
                Self::launch_tiled::<R>(client, weight_handle, input_handle, output_handle, m, n);
            }
        }
    }

    /// Launch plane (subgroup) f16 weight GEMV kernel.
    ///
    /// Each plane handles one output row with cooperative dot product + `plane_sum()`.
    /// Workgroup: 256 threads → 8 planes (with plane_dim=32 on Metal).
    /// Dispatch: `ceil(m / 8)` workgroups.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: M×N f16 elements
    /// - `input_handle`: N f32 elements
    /// - `output_handle`: M f32 elements
    pub unsafe fn launch_plane<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Conservative: Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8
        let num_wg = (m as u32).div_ceil(rows_per_wg).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            gemv_plane_f16_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(weight_handle, m * n),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }

    /// Launch shared-memory tiled f16 weight GEMV kernel (fallback, no subgroups).
    ///
    /// Each thread computes one output row. Input tiled into shared memory.
    /// Workgroup: 256 threads → 256 rows per workgroup.
    /// Dispatch: `ceil(m / 256)` workgroups.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: M×N f16 elements
    /// - `input_handle`: N f32 elements
    /// - `output_handle`: M f32 elements
    pub unsafe fn launch_tiled<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let wg_size = 256u32;
        let num_wg = (m as u32).div_ceil(wg_size).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            gemv_tile_f16_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(weight_handle, m * n),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use cubecl::features::Plane;
    use crate::cubecl_runtime::ActiveRuntime;

    use crate::cubecl_runtime::CubeCLContext;

    use super::*;

    /// Helper: convert f32 slice to f16 bytes for GPU upload.
    fn f32_slice_to_f16_bytes(data: &[f32]) -> Vec<u8> {
        let f16: Vec<half_f16> = if data.len() < 4096 {
            data.iter().map(|&x| half_f16::from_f32(x)).collect()
        } else {
            data.par_iter().map(|&x| half_f16::from_f32(x)).collect()
        };
        bytemuck::cast_slice::<half_f16, u8>(&f16).to_vec()
    }

    /// Verify plane f16 GEMV with 4×4 identity matrix.
    #[test]
    fn test_gemv_f16_plane_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let m = 4usize;
        let n = 4usize;

        // Identity matrix (row-major)
        let weight_f32: &[f32] = &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let expected: &[f32] = &[1.0, 2.0, 3.0, 4.0];

        let weight_handle = client.create_from_slice(&f32_slice_to_f16_bytes(weight_f32));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvF16CubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 0.01,
                "element {i}: expected {exp}, got {got}"
            );
        }
        println!("plane f16 GEMV identity: {output:?}");
    }

    /// Verify plane f16 GEMV with a general 3×4 matrix.
    #[test]
    fn test_gemv_f16_plane_general() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let m = 3usize;
        let n = 4usize;
        let weight_f32: &[f32] = &[
            1.0, 2.0, 3.0, 4.0, // row 0: 1+4+9+16 = 30
            5.0, 6.0, 7.0, 8.0, // row 1: 5+12+21+32 = 70
            9.0, 10.0, 11.0, 12.0, // row 2: 9+20+33+48 = 110
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let expected: &[f32] = &[30.0, 70.0, 110.0];

        let weight_handle = client.create_from_slice(&f32_slice_to_f16_bytes(weight_f32));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvF16CubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 0.5,
                "element {i}: expected {exp}, got {got}"
            );
        }
        println!("plane f16 GEMV general: {output:?}");
    }

    /// Verify tiled f16 GEMV (shared memory fallback).
    #[test]
    fn test_gemv_f16_tiled() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 4usize;
        let n = 8usize;

        // 4×8 partial identity: row i picks input[i]
        let weight_f32: &[f32] = &[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected: &[f32] = &[1.0, 2.0, 3.0, 4.0];

        let weight_handle = client.create_from_slice(&f32_slice_to_f16_bytes(weight_f32));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvF16CubeCL::launch_tiled::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 0.01,
                "element {i}: expected {exp}, got {got}"
            );
        }
        println!("tiled f16 GEMV: {output:?}");
    }

    /// Verify auto-select picks plane or tiled based on device capability.
    #[test]
    fn test_gemv_f16_auto_select() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 4usize;
        let n = 4usize;

        // Scaled identity: 2 * I
        let weight_f32: &[f32] = &[
            2.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, 2.0,
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let expected: &[f32] = &[2.0, 4.0, 6.0, 8.0];

        let weight_handle = client.create_from_slice(&f32_slice_to_f16_bytes(weight_f32));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 0.01,
                "element {i}: expected {exp}, got {got}"
            );
        }

        let mode = if client.features().plane.contains(Plane::Ops) {
            "plane"
        } else {
            "tiled"
        };
        println!("auto-select f16 GEMV ({mode}): {output:?}");
    }

    /// Verify plane f16 GEMV with 256×256 diagonal matrix (typical LLM hidden dim).
    #[test]
    fn test_gemv_f16_plane_large() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let m = 256usize;
        let n = 256usize;

        // Diagonal matrix with values 1..=256
        let mut weight_f32 = vec![0.0f32; m * n];
        for i in 0..m.min(n) {
            weight_f32[i * n + i] = (i + 1) as f32;
        }
        // All-ones input → output[i] = (i+1)
        let input = vec![1.0f32; n];
        let expected: Vec<f32> = (0..m)
            .map(|i| if i < n { (i + 1) as f32 } else { 0.0 })
            .collect();

        let weight_handle = client.create_from_slice(&f32_slice_to_f16_bytes(&weight_f32));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvF16CubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(err < 0.5, "element {i}: expected {exp}, got {got}");
        }
        println!("large f16 GEMV ({m}×{n}): max_error = {max_err}");
    }

    /// Verify f16 GEMV matches f32 GEMV within f16 precision tolerance.
    ///
    /// f16 has ~3 decimal digits of precision. For typical LLM weight magnitudes
    /// (±10), the quantization error per element is ~0.01–0.05. Accumulated over
    /// N=16 elements, total error should stay within ~0.1–1.0.
    #[test]
    fn test_gemv_f16_matches_f32() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let m = 8usize;
        let n = 16usize;

        // General matrix with values representable in f16
        let weight_f32: Vec<f32> = (0..m * n).map(|i| ((i % 7) + 1) as f32 * 0.5).collect();
        let input: Vec<f32> = (0..n).map(|i| ((i % 5) + 1) as f32 * 0.25).collect();

        // Run f32 GEMV (reference)
        let weight_f32_handle = client.create_from_slice(f32::as_bytes(&weight_f32));
        let input_f32_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_f32_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            crate::gemv_cubecl::GemvCubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_f32_handle,
                input_f32_handle,
                output_f32_handle.clone(),
                m,
                n,
            );
        }

        let bytes_f32 = client
            .read_one(output_f32_handle)
            .expect("should read f32 output");
        let output_f32 = f32::from_bytes(&bytes_f32);

        // Run f16 GEMV
        let weight_f16_handle = client.create_from_slice(&f32_slice_to_f16_bytes(&weight_f32));
        let input_f16_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_f16_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvF16CubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_f16_handle,
                input_f16_handle,
                output_f16_handle.clone(),
                m,
                n,
            );
        }

        let bytes_f16 = client
            .read_one(output_f16_handle)
            .expect("should read f16 output");
        let output_f16 = f32::from_bytes(&bytes_f16);

        assert_eq!(output_f32.len(), m);
        assert_eq!(output_f16.len(), m);

        let mut max_err = 0.0f32;
        for (i, (&f32_val, &f16_val)) in output_f32.iter().zip(output_f16.iter()).enumerate() {
            let err = (f32_val - f16_val).abs();
            if err > max_err {
                max_err = err;
            }
            // f16 tolerance: ~0.1 per element × 16 accumulation = ~1.6 worst case
            assert!(
                err < 2.0,
                "element {i}: f32={f32_val}, f16={f16_val}, diff={err}"
            );
        }
        println!("f16 vs f32 GEMV ({m}×{n}): max_error = {max_err}");
    }
}
