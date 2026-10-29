//! Issue 1150 follow-up G1 — the dense-f16 matmul2d + decode-GEMV unit gate.
//!
//! Two kernels, one gate:
//! - `gemm_dense_f16_tensor` (the cooperative matmul2d prefill path) vs a CPU
//!   reference over the SAME f16-rounded weights — tile slicing at the
//!   production edge classes and the n=5120 production column count.
//! - `gemv_dense_f16` (the plain decode path — the matmul2d path cannot
//!   serve a (K × 1) activation view; measured: the cooperative load/store
//!   silently produce nothing at tiny p).
//!
//! Shape laws (measured, enforced by the dispatch guards + asserted here):
//! - matmul2d: `m` must be a multiple of the 64-row tile (the cooperative
//!   store SKIPS the tile when m and p are both tile-ragged); `p` must be ≥
//!   the 64-token ingest floor (production ingest chunks are multiples of
//!   64; decode rides the GEMV kernel).
//! - Tolerance: relative to the per-element accumulation MAGNITUDE `Σ|w·x|`
//!   (cancellation-free) — a pure relative band on a near-zero dot measures
//!   summation-order noise, not the kernel.

#![cfg(all(
    feature = "qwen38_wgpu",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]

use cubecl::client::ComputeClient;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::gemm_dense_f16_metal_zero_copy::MetalDenseF16ZeroCopyGemm;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(14_426_950_408_889_407);
    ((*seed >> 33) as f32 / (u64::MAX >> 33) as f32) * 2.0 - 1.0
}

fn f16_bits(wf: &[f32]) -> Vec<u16> {
    wf.iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect()
}

/// CPU reference over the f16-ROUNDED weights — the kernel and the reference
/// see identical weight values, so the band isolates summation-order drift
/// (f32 accumulate both sides) from any indexing/tiling defect. Returns the
/// dots and the per-element `Σ|w·x|` magnitudes.
fn cpu_gemm_f16(w16: &[u16], x: &[f32], m: usize, n: usize, p: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut out, mut mag) = (vec![0.0f32; p * m], vec![0.0f32; p * m]);
    for t in 0..p {
        for r in 0..m {
            let (mut dot, mut acc) = (0.0f32, 0.0f32);
            for c in 0..n {
                let term = f32::from(half::f16::from_bits(w16[r * n + c])) * x[t * n + c];
                dot += term;
                acc += term.abs();
            }
            out[t * m + r] = dot;
            mag[t * m + r] = acc;
        }
    }
    (out, mag)
}

fn run_case(
    mm: &MetalDenseF16ZeroCopyGemm,
    client: &ComputeClient<ActiveRuntime>,
    m: usize,
    n: usize,
    p: usize,
) {
    let mut seed: u64 =
        0xd00d_f160_0000_0000 ^ (m as u64) ^ ((n as u64) << 21) ^ ((p as u64) << 42);
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
    let w16 = f16_bits(&wf);
    let (expect, mag) = cpu_gemm_f16(&w16, &x, m, n, p);

    let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
    // Host uploads are DEFERRED writes (the issue-1146 attention-gate
    // hazard): a raw commit can land before a queued upload reaches the
    // pool. Flush orders kernel producers, not these — settle the fixture.
    pollster::block_on(client.sync()).expect("sync after uploads");

    mm.dispatch(client, &w_h, m, n, &x_h, &out_h, p)
        .unwrap_or_else(|e| panic!("dispatch {m}x{n} p{p}: {e}"));
    mm.wait_last();
    let got_bytes = client.read_one(out_h).expect("read");
    let got: &[f32] = bytemuck::cast_slice(&got_bytes);

    let mut max_rel = 0.0f64;
    for (i, (&g, (&e, &mg))) in got.iter().zip(expect.iter().zip(mag.iter())).enumerate() {
        let rel = ((g - e).abs() / mg.max(1e-9)) as f64;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < 2e-4,
            "case {m}x{n} p{p}: element {i} got {g} want {e} (rel-to-mag {rel:.3e})"
        );
    }
    println!("case {m}x{n} p{p}: max_rel-to-mag {max_rel:.3e}");
}

/// CPU GEMV reference over the f16-rounded weights (the decode shape).
fn cpu_gemv_f16(w16: &[u16], x: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut out, mut mag) = (vec![0.0f32; m], vec![0.0f32; m]);
    for r in 0..m {
        let (mut dot, mut acc) = (0.0f32, 0.0f32);
        for c in 0..n {
            let term = f32::from(half::f16::from_bits(w16[r * n + c])) * x[c];
            dot += term;
            acc += term.abs();
        }
        out[r] = dot;
        mag[r] = acc;
    }
    (out, mag)
}

fn run_gemv_case(
    mm: &MetalDenseF16ZeroCopyGemm,
    client: &ComputeClient<ActiveRuntime>,
    m: usize,
    n: usize,
) {
    let mut seed: u64 = 0xdead_beef_d160_0000 ^ (m as u64) ^ ((n as u64) << 21);
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
    let w16 = f16_bits(&wf);
    let (expect, mag) = cpu_gemv_f16(&w16, &x, m, n);

    let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_h = client.create_from_slice(&vec![0u8; m * 4]);
    pollster::block_on(client.sync()).expect("sync after uploads");

    mm.dispatch_gemv(client, &w_h, m, n, &x_h, &out_h)
        .unwrap_or_else(|e| panic!("gemv dispatch {m}x{n}: {e}"));
    mm.wait_last();
    let got_bytes = client.read_one(out_h).expect("read");
    let got: &[f32] = bytemuck::cast_slice(&got_bytes);

    let mut max_rel = 0.0f64;
    for (i, (&g, (&e, &mg))) in got.iter().zip(expect.iter().zip(mag.iter())).enumerate() {
        let rel = ((g - e).abs() / mg.max(1e-9)) as f64;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < 2e-4,
            "gemv case {m}x{n}: element {i} got {g} want {e} (rel-to-mag {rel:.3e})"
        );
    }
    println!("gemv case {m}x{n}: max_rel-to-mag {max_rel:.3e}");
}

#[test]
fn dense_f16_matmul2d_g1() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense-f16 zero-copy context");

    // The full matrix re-measured under the upload-sync discipline (the
    // pre-sync zero-writes were the deferred-upload race, not shape laws —
    // every class reads its OWN values, including m-edge, p-edge, the
    // combined ragged class, and p=1). The gate pins the full matrix so a
    // cooperative-tensor regression on ANY production shape class is loud.
    run_case(&mm, &client, 64, 64, 128); // exact 64x128 tiles
    run_case(&mm, &client, 64, 96, 100); // p-edge + k-edge (m exact)
    run_case(&mm, &client, 128, 64, 100); // p-edge, two m tiles
    run_case(&mm, &client, 130, 64, 128); // m-edge alone
    run_case(&mm, &client, 48, 64, 128); // m-edge, single tile
    run_case(&mm, &client, 48, 96, 100); // m AND p both tile-ragged
    run_case(&mm, &client, 512, 5120, 256); // production-shape column count
    run_case(&mm, &client, 512, 5120, 64); // ingest-floor p (production min)
    run_case(&mm, &client, 64, 5120, 64); // ingest floor, one m tile
    run_case(&mm, &client, 512, 5120, 1); // matmul2d at p=1 (parity class)

    // The guards considered during the race investigation came OUT: under
    // the upload-sync discipline every shape class is correct (measured
    // above). The race itself is the fixture's discipline, not the
    // kernel's — production reads x from queue-ordered kernels, never from
    // a just-uploaded host buffer.
}

#[test]
fn dense_f16_gemv_g1() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense-f16 zero-copy context");

    run_gemv_case(&mm, &client, 64, 64); // one workgroup exact
    run_gemv_case(&mm, &client, 1000, 5120); // m-edge (1000 = 250 workgroups exact)
    run_gemv_case(&mm, &client, 512, 5120); // production FFN-down shape
    run_gemv_case(&mm, &client, 3, 96); // m-edge ragged (3 rows, 1 workgroup)
}

/// Kernel-level A/B at the production FFN shape ([17408×5120]×512): the
/// incumbent Q4_K tiled CubeCL GEMM (the ~110 GFLOP/s scalar wall, session
/// record) vs the dense-f16 matmul2d. ABBA-interleaved pairs, median of
/// pair-ratios (the sequential-AB law — a ratio of two sequential arms
/// measures the box, interleaving measures the kernels). RELEASE + --ignored:
/// `cargo test -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm
/// --test qwen38_dense_f16_matmul2d_g1 dense_f16_vs_q4k_ab --release --
/// --ignored --nocapture`. Quote the box state beside the numbers.
#[test]
#[ignore]
fn dense_f16_vs_q4k_ab() {
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, quantize_row_q4_k};
    use riir_infer_gpu::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL;
    use riir_infer_gpu::gemv_q4k_cubecl::Q4KHandle;

    let (m, n, p) = (17_408usize, 5_120usize, 512usize);
    let mut seed: u64 = 0xabab_d160_0000_0000;
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
    let w16 = f16_bits(&wf);

    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense-f16 zero-copy context");

    // Incumbent arm: Q4_K blocks → the tiled CubeCL GEMM.
    let bpr = n / QK_K;
    let mut blocks = vec![BlockQ4K::zeroed(); m * bpr];
    let mut row = vec![0.0f32; n];
    for r in 0..m {
        row.copy_from_slice(&wf[r * n..(r + 1) * n]);
        quantize_row_q4_k(&row, &mut blocks[r * bpr..(r + 1) * bpr]);
    }
    let q4 = Q4KHandle::from_blocks(&client, &blocks, m, n);
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
    // Dense arm: f16 weights + the zero-copy context.
    let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
    pollster::block_on(client.sync()).expect("sync after uploads");

    // Warmup both arms (JIT + pool + dedicated buffers).
    for _ in 0..3 {
        unsafe {
            GemmQ4KTileCubeCL::launch::<ActiveRuntime>(&client, &q4, x_h.clone(), out_h.clone(), p);
        }
        mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, p)
            .expect("dense dispatch");
    }
    pollster::block_on(client.sync()).expect("warmup sync");

    // Throughput protocol: N back-to-back dispatches between two full syncs,
    // wall/N per arm — submission cost and GPU time both inside the clock,
    // the same shape and the same box state for both arms.
    const N: usize = 10;
    let time_q4 = || {
        pollster::block_on(client.sync()).expect("q4 pre-sync");
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            unsafe {
                GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                    &client,
                    &q4,
                    x_h.clone(),
                    out_h.clone(),
                    p,
                );
            }
        }
        pollster::block_on(client.sync()).expect("q4 post-sync");
        t0.elapsed().as_secs_f64() * 1000.0 / N as f64
    };
    let time_dense = || {
        pollster::block_on(client.sync()).expect("dense pre-sync");
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, p)
                .expect("dense dispatch");
        }
        mm.wait_last(); // raw-Metal submissions are invisible to client.sync()
        pollster::block_on(client.sync()).expect("dense post-sync");
        t0.elapsed().as_secs_f64() * 1000.0 / N as f64
    };

    // ABBA x2 rounds, median of the four per-arm readings.
    let mut q4s = Vec::new();
    let mut dens = Vec::new();
    for _ in 0..2 {
        q4s.push(time_q4());
        dens.push(time_dense());
        dens.push(time_dense());
        q4s.push(time_q4());
    }
    q4s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dens.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (q4_med, d_med) = (q4s[q4s.len() / 2], dens[dens.len() / 2]);
    let gflop = 2.0 * m as f64 * n as f64 * p as f64 / 1e9;
    // Loud-zero insurance at the exact timing shape: the timed loop wrote
    // out_h N times — read it back once and check row 0/token 0 against the
    // CPU dot over the same f16 weights (the timed_region_guard class: a
    // latency bar satisfied by absent work reads as an infinite speedup).
    let (expect0, _) = {
        let mut dot = 0.0f32;
        let mut acc = 0.0f32;
        for c in 0..n {
            let term = f32::from(half::f16::from_bits(w16[c])) * x[c];
            dot += term;
            acc += term.abs();
        }
        (dot, acc)
    };
    // Re-dispatch the dense arm LAST — the ABBA order leaves the q4 arm's
    // (quantized-weight) output in the shared handle otherwise, and the
    // check would compare against the wrong arm's numerics.
    mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, p)
        .expect("dense dispatch");
    mm.wait_last();
    let got_bytes = client.read_one(out_h.clone()).expect("read ab output");
    let got: &[f32] = bytemuck::cast_slice(&got_bytes);
    let rel = ((got[0] - expect0).abs() / expect0.abs().max(1e-6)) as f64;
    assert!(
        rel < 2e-4,
        "A/B dense arm loud-zero: got {} want {expect0} (rel {rel:.3e})",
        got[0]
    );
    println!(
        "A/B output verified: got {:.6} want {expect0:.6} (rel {rel:.3e})",
        got[0]
    );

    println!(
        "A/B @{m}x{n}xp{p} ({gflop:.1} GFLOP, N={N} back-to-back): q4-tile {q4_med:.2} ms \
         ({:.0} GFLOP/s) | dense-f16 mm2d {d_med:.2} ms ({:.0} GFLOP/s) | speedup {:.2}x",
        gflop / (q4_med / 1000.0),
        gflop / (d_med / 1000.0),
        q4_med / d_med
    );
}
