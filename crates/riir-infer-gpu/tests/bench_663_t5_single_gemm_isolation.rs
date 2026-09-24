//! Issue 663 T5 — single-GEMM zero-copy dispatch isolation test.
//!
//! Isolates whether the G1 failure in `bench_663_t5_zerocopy_prefill_e2e.rs`
//! is in the dispatch path (raw Metal on shared CubeCL device) or in the
//! prefill integration. Runs ONE matmul2d GEMM via the zero-copy path with
//! known weights + input, compares against a CPU reference.
//!
//! If this passes, the dispatch path is correct and the prefill G1 failure
//! is an integration bug (buffer aliasing, stride mismatch, etc.). If this
//! fails, the dispatch path itself has a bug.
//!
//! ## Usage
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/p536 cargo test -p riir-infer-gpu \
//!     --features "cubecl_runtime ternary_gemm_batched metal_tensor_gemm" --release \
//!     --test bench_663_t5_single_gemm_isolation -- --nocapture --ignored
//! ```

// Issue 830: `not(feature = "cuda_backend")` on a CUDA host — this test
// reaches through `ComputeClient<ActiveRuntime>` to a raw
// `wgpu::hal::api::Metal` buffer. `cuda_backend` re-points `ActiveRuntime`
// at `CudaRuntime`, whose `GpuResource` has no `buffer`/`offset` fields, so
// the body is E0609 in any feature set that turns it on (e.g.
// `--all-features`) ON A CUDA HOST. On macOS the feature is inert (Issue 949)
// and the alias stays wgpu, so the test compiles there under `--all-features`.
// The Metal path is what is under test; excluding the CUDA arm is the honest
// gate.
#![cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    any(not(feature = "cuda_backend"), target_os = "macos"),
    target_os = "macos",
))]

use cubecl::client::ComputeClient;

use riir_infer_gpu::TernaryHandle;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};

/// Reference: scalar CPU ternary GEMM (u32 bit-plane indexing, matches the MSL kernel).
fn cpu_ternary_gemm(
    pos_bits_u64: &[u64],
    neg_bits_u64: &[u64],
    group_scale: &[half::f16],
    input: &[f32], // [P][K]
    m: usize,
    n: usize,
    p: usize,
) -> Vec<f32> {
    // Cast u64 bit-planes to u32 (interleaved low/high halves), matching the
    // CubeCL handle layout (cast_u64_to_u32).
    let pos_u32: Vec<u32> = pos_bits_u64
        .iter()
        .flat_map(|&w| [w as u32, (w >> 32) as u32])
        .collect();
    let neg_u32: Vec<u32> = neg_bits_u64
        .iter()
        .flat_map(|&w| [w as u32, (w >> 32) as u32])
        .collect();

    let blocks64 = n.div_ceil(64);
    let groups_per_row = n.div_ceil(128);
    let words_per_row = blocks64 * 2; // u32 words per weight row
    let mut out = vec![0.0f32; p * m];
    for tok in 0..p {
        for row in 0..m {
            let mut acc = 0.0f32;
            for col in 0..n {
                let word_idx = row * words_per_row + col / 32;
                let bit_pos = col % 32;
                let pos_bit = (pos_u32[word_idx] >> bit_pos) & 1;
                let neg_bit = (neg_u32[word_idx] >> bit_pos) & 1;
                let sign = pos_bit as f32 - neg_bit as f32;
                let scale = group_scale[row * groups_per_row + col / 128].to_f32();
                let w = sign * scale;
                let x = input[tok * n + col];
                acc += w * x;
            }
            out[tok * m + row] = acc;
        }
    }
    out
}

/// Extract raw (id<MTLBuffer>, offset) from a CubeCL handle.
fn extract_raw(
    client: &ComputeClient<ActiveRuntime>,
    handle: &cubecl::server::Handle,
) -> (*mut metal::MTLBuffer, u64) {
    let managed = client.get_resource(handle.clone()).expect("get_resource");
    let resource = managed.resource();
    let offset = resource.offset;
    let wgpu_buffer = &resource.buffer;
    let id_ptr = unsafe {
        let guard = wgpu_buffer.as_hal::<wgpu::hal::api::Metal>();
        let hal_buf = guard.as_ref().expect("Metal buffer");
        let proto = hal_buf.raw_handle();
        proto as *const _ as *mut metal::MTLBuffer
    };
    (id_ptr, offset)
}

/// Single-GEMM zero-copy isolation test.
#[test]
#[ignore = "requires Metal GPU + CubeCL runtime; run manually"]
fn single_gemm_zerocopy_vs_cpu() {
    // Small GEMM: M=64, K=256, P=4 (fits in one 64×128 output tile, one token tile).
    let m: usize = 64;
    let n: usize = 256;
    let p: usize = 4;

    // Generate deterministic test weights + input.
    let blocks64 = n.div_ceil(64);
    let groups_per_row = n.div_ceil(128);
    let pos_bits: Vec<u64> = (0..m * blocks64)
        .map(|i| (i as u64).wrapping_mul(0x123456789ABCDEF0))
        .collect();
    let neg_bits: Vec<u64> = (0..m * blocks64)
        .map(|i| (i as u64).wrapping_mul(0xFEDCBA9876543210))
        .collect();
    let group_scale: Vec<half::f16> = (0..m * groups_per_row)
        .map(|i| half::f16::from_f32(0.5 + 0.001 * (i as f32)))
        .collect();
    let input: Vec<f32> = (0..p * n)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();

    // CPU reference.
    let cpu_out = cpu_ternary_gemm(&pos_bits, &neg_bits, &group_scale, &input, m, n, p);
    eprintln!(
        "[isolation] CPU reference: first 8 = {:?}",
        &cpu_out[..8.min(cpu_out.len())]
    );

    // CubeCL setup.
    let ctx = CubeCLContext::new().expect("CubeCL init");
    let client: ComputeClient<ActiveRuntime> = ctx.client();
    let wgpu_device = ctx.wgpu_device().expect("wgpu device");
    let _wgpu_queue = ctx.wgpu_queue().expect("wgpu queue");

    // Upload weights + input to CubeCL.
    let weight_handle = TernaryHandle::from_raw(&client, &pos_bits, &neg_bits, &group_scale, m, n);

    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input).to_vec();
    let input_handle = client.empty(input_bytes.len());
    client.write(
        &input_handle,
        cubecl::bytes::Bytes::from_bytes_vec(input_bytes),
    );

    // Verify input data landed on GPU.
    pollster::block_on(client.sync()).expect("sync after write");
    let in_check = client.read_one(input_handle.clone()).expect("read input");
    let in_check_f32: &[f32] = bytemuck::cast_slice(&in_check[..]);
    eprintln!(
        "[diag] input readback: first 4 = {:?}, expected first 4 = {:?}",
        &in_check_f32[..4],
        &input[..4]
    );

    // Also verify weight data landed on GPU.
    let pos_check = client
        .read_one(weight_handle.pos_bits_u32.clone())
        .expect("read pos");
    eprintln!(
        "[diag] pos readback: first 16 bytes = {:?}",
        &pos_check[..16.min(pos_check.len())]
    );

    // Use a DEDICATED Metal buffer for output (not CubeCL pool) to test
    // whether the pool-buffer aliasing is the root cause of the all-zeros bug.
    let metal_device: &metal::DeviceRef = unsafe {
        let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
        let dev = guard.as_ref().expect("Metal device");
        let proto = dev.raw_device();
        metal::foreign_types::ForeignTypeRef::from_ptr(
            (&**proto) as *const _ as *mut metal::MTLDevice,
        )
    };
    let output_bytes = p * m * std::mem::size_of::<f32>();
    let dedicated_output = metal_device.new_buffer(
        output_bytes as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    eprintln!(
        "[diag] dedicated output buffer: {} bytes at ptr {:p}",
        output_bytes,
        metal::foreign_types::ForeignType::as_ptr(&dedicated_output)
    );

    // Also create a CubeCL output handle for comparison.
    let _output_handle = client.empty(output_bytes);

    // Dispatch to the DEDICATED Metal output buffer directly.
    // We can't use `gemm.dispatch()` for this — it expects CubeCL handles.
    // Instead, replicate the dispatch inline with the dedicated buffer.
    {
        use metal::foreign_types::ForeignTypeRef;
        use metal::{MTLResourceUsage, MTLSize};

        const NRA: u32 = 64;
        const NRB: u32 = 128;
        const NSG: u32 = 4;
        const NUM_THREADS: u32 = 32 * NSG;
        const TG_MEM: u64 = 32 * 64 * 2;

        let queue_ref = unsafe {
            metal::CommandQueueRef::from_ptr(
                ctx.wgpu_queue()
                    .unwrap()
                    .as_hal::<wgpu::hal::api::Metal>()
                    .as_ref()
                    .unwrap()
                    .as_raw() as *const _ as *mut metal::MTLCommandQueue,
            )
        };

        // Compile the kernel.
        let msl_source = include_str!("../src/gemm_ternary_metal_tensor.metal");
        let library = metal_device
            .new_library_with_source(msl_source, &metal::CompileOptions::new())
            .expect("MSL compile");
        let function = library
            .get_function("gemm_ternary_tensor", None)
            .expect("function");
        let pipeline = metal_device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

        // Extract raw weight + input buffers from CubeCL.
        let (pos_ptr, pos_off) = extract_raw(&client, &weight_handle.pos_bits_u32);
        let (neg_ptr, neg_off) = extract_raw(&client, &weight_handle.neg_bits_u32);
        let (scale_ptr, scale_off) = extract_raw(&client, &weight_handle.group_scale_f32);
        let (input_ptr, input_off) = extract_raw(&client, &input_handle);

        let pos_ref = unsafe { metal::BufferRef::from_ptr(pos_ptr) };
        let neg_ref = unsafe { metal::BufferRef::from_ptr(neg_ptr) };
        let scale_ref = unsafe { metal::BufferRef::from_ptr(scale_ptr) };
        let input_ref = unsafe { metal::BufferRef::from_ptr(input_ptr) };

        let grid = MTLSize {
            width: (p as u64).div_ceil(NRB as u64),
            height: (m as u64).div_ceil(NRA as u64),
            depth: 1,
        };
        let tg = MTLSize {
            width: NUM_THREADS as u64,
            height: 1,
            depth: 1,
        };

        let blocks64_v: u32 = weight_handle.blocks64 as u32;
        let gpr_v: u32 = weight_handle.groups_per_row as u32;
        let n_v: u32 = weight_handle.n as u32;
        let m_v: u32 = weight_handle.m as u32;
        let p_v: u32 = p as u32;

        let cmd = queue_ref.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(pos_ref), pos_off);
        enc.set_buffer(1, Some(neg_ref), neg_off);
        enc.set_buffer(2, Some(scale_ref), scale_off);
        enc.set_buffer(3, Some(input_ref), input_off);
        // Binding 4 = output = DEDICATED buffer, offset 0.
        enc.set_buffer(4, Some(&dedicated_output), 0);
        enc.set_bytes(5, 4, &blocks64_v as *const u32 as *const _);
        enc.set_bytes(6, 4, &gpr_v as *const u32 as *const _);
        enc.set_bytes(7, 4, &n_v as *const u32 as *const _);
        enc.set_bytes(8, 4, &m_v as *const u32 as *const _);
        enc.set_bytes(9, 4, &p_v as *const u32 as *const _);
        enc.set_threadgroup_memory_length(0, TG_MEM);
        enc.use_resource(pos_ref, MTLResourceUsage::Read);
        enc.use_resource(neg_ref, MTLResourceUsage::Read);
        enc.use_resource(scale_ref, MTLResourceUsage::Read);
        enc.use_resource(input_ref, MTLResourceUsage::Read);
        enc.use_resource(&dedicated_output, MTLResourceUsage::Write);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Read back via Metal (not CubeCL).
        let ptr = dedicated_output.contents() as *const f32;
        let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, p * m) }.to_vec();
        eprintln!(
            "[isolation] GPU output (dedicated): first 8 = {:?}",
            &gpu_out[..8.min(gpu_out.len())]
        );

        // Compare.
        assert_eq!(gpu_out.len(), cpu_out.len(), "output length mismatch");
        let mut max_abs = 0.0f32;
        let mut n_over = 0usize;
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let abs = (g - c).abs();
            if abs > 0.5 {
                n_over += 1;
                if n_over <= 5 {
                    eprintln!("  MISMATCH[{i}]: gpu={g:.4} cpu={c:.4} diff={abs:.4}");
                }
            }
            if abs > max_abs {
                max_abs = abs;
            }
        }
        eprintln!(
            "[isolation-dedicated] max_abs={max_abs:.4}, over={}/{} ({:.1}%)",
            n_over,
            gpu_out.len(),
            100.0 * n_over as f64 / gpu_out.len() as f64
        );

        if max_abs < 1.0 {
            eprintln!(
                "✅ single-GEMM DEDICATED output matches CPU reference (max_abs {max_abs:.4})"
            );
        } else {
            eprintln!(
                "❌ single-GEMM DEDICATED output MISMATCHES CPU reference (max_abs {max_abs:.4})"
            );
        }
    }
}
