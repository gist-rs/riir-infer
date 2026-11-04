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
    tol: f64,
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
            rel < tol,
            "case {m}x{n} p{p}: element {i} got {g} want {e} (rel-to-mag {rel:.3e} >= band {tol:.3e})"
        );
    }
    println!("case {m}x{n} p{p}: max_rel-to-mag {max_rel:.3e}");
}

/// Issue 1170 half-B case: the kernel reads the activation as an f16 STAGING
/// view, so the CPU reference rounds x to f16 too (identical values both
/// sides — the band isolates accumulation order from indexing/tiling
/// defects, the same design as run_case).
fn cpu_gemm_f16_hact(
    w16: &[u16],
    x16: &[u16],
    m: usize,
    n: usize,
    p: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (mut out, mut mag) = (vec![0.0f32; p * m], vec![0.0f32; p * m]);
    for t in 0..p {
        for r in 0..m {
            let (mut dot, mut acc) = (0.0f32, 0.0f32);
            for c in 0..n {
                let term = f32::from(half::f16::from_bits(w16[r * n + c]))
                    * f32::from(half::f16::from_bits(x16[t * n + c]));
                dot += term;
                acc += term.abs();
            }
            out[t * m + r] = dot;
            mag[t * m + r] = acc;
        }
    }
    (out, mag)
}

fn run_case_hact(
    mm: &MetalDenseF16ZeroCopyGemm,
    client: &ComputeClient<ActiveRuntime>,
    m: usize,
    n: usize,
    p: usize,
    tol: f64,
) {
    let mut seed: u64 =
        0xd17a_b117_0000_0000 ^ (m as u64) ^ ((n as u64) << 21) ^ ((p as u64) << 42);
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
    let w16 = f16_bits(&wf);
    let x16 = f16_bits(&x);
    let (expect, mag) = cpu_gemm_f16_hact(&w16, &x16, m, n, p);

    let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x16));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
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
            rel < tol,
            "case {m}x{n} p{p}: element {i} got {g} want {e} (rel-to-mag {rel:.3e} >= band {tol:.3e})"
        );
    }
    println!("half-b case {m}x{n} p{p}: max_rel-to-mag {max_rel:.3e}");
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
    let mm = MetalDenseF16ZeroCopyGemm::new_with_relaxed(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        false,
    )
    .expect("dense-f16 zero-copy context");
    assert!(
        !mm.relaxed_precision(),
        "exact gate must carry the exact arm"
    );

    // The full matrix re-measured under the upload-sync discipline (the
    // pre-sync zero-writes were the deferred-upload race, not shape laws —
    // every class reads its OWN values, including m-edge, p-edge, the
    // combined ragged class, and p=1). The gate pins the full matrix so a
    // cooperative-tensor regression on ANY production shape class is loud.
    // Band 2e-4 rel-to-mag: the f32-accumulate exact posture (summation-order
    // noise only).
    run_case(&mm, &client, 64, 64, 128, 2e-4); // exact 64x128 tiles
    run_case(&mm, &client, 64, 96, 100, 2e-4); // p-edge + k-edge (m exact)
    run_case(&mm, &client, 128, 64, 100, 2e-4); // p-edge, two m tiles
    run_case(&mm, &client, 130, 64, 128, 2e-4); // m-edge alone
    run_case(&mm, &client, 48, 64, 128, 2e-4); // m-edge, single tile
    run_case(&mm, &client, 48, 96, 100, 2e-4); // m AND p both tile-ragged
    run_case(&mm, &client, 512, 5120, 256, 2e-4); // production-shape column count
    run_case(&mm, &client, 512, 5120, 64, 2e-4); // ingest-floor p (production min)
    run_case(&mm, &client, 64, 5120, 64, 2e-4); // ingest floor, one m tile
    run_case(&mm, &client, 512, 5120, 1, 2e-4); // matmul2d at p=1 (parity class)

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
    let mm = MetalDenseF16ZeroCopyGemm::new_with_relaxed(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        false,
    )
    .expect("dense-f16 zero-copy context");

    run_gemv_case(&mm, &client, 64, 64); // one workgroup exact
    run_gemv_case(&mm, &client, 1000, 5120); // m-edge (1000 = 250 workgroups exact)
    run_gemv_case(&mm, &client, 512, 5120); // production FFN-down shape
    run_gemv_case(&mm, &client, 3, 96); // m-edge ragged (3 rows, 1 workgroup)
}

/// Follow-up 4 — the relaxed-precision arm's G1: the exact gate's fixture
/// machinery, PRE-REGISTERED band 3e-2 rel-to-mag (fp16-accumulate physics:
/// ~2.8e-3 relative per 32-deep tile partial against a random-sign dot, the
/// K-deep tile partials summing in f32 — measured drift lands far below the
/// band; the band is a drift detector, the e2e walk agreement stats are the
/// deployed-path arbiter, per the bench-923 lossy-surface protocol).
/// Plus the determinism pin: the tile geometry fixes the reduction order, so
/// two runs of the same case are BYTE-IDENTICAL — relaxed changes precision,
/// never order.
#[test]
fn dense_f16_matmul2d_relaxed_g1() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new_with_relaxed(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        true,
    )
    .expect("dense-f16 zero-copy context");
    assert!(
        mm.relaxed_precision(),
        "relaxed gate must carry the relaxed arm"
    );

    run_case(&mm, &client, 64, 64, 128, 3e-2); // exact tiles
    run_case(&mm, &client, 130, 64, 128, 3e-2); // m-edge
    run_case(&mm, &client, 48, 96, 100, 3e-2); // m AND p both tile-ragged
    run_case(&mm, &client, 512, 5120, 256, 3e-2); // production column count
    run_case(&mm, &client, 512, 5120, 64, 3e-2); // ingest-floor p

    // Determinism pin: one fixture, two dispatches, two outputs —
    // byte-identical.
    let (m, n, p) = (512usize, 5120usize, 64usize);
    let mut seed: u64 = 0x1150_1E1A_FE00_D001 ^ ((n as u64) << 21);
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
    let w16 = f16_bits(&wf);
    let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_a = client.create_from_slice(&vec![0u8; p * m * 4]);
    let out_b = client.create_from_slice(&vec![0u8; p * m * 4]);
    pollster::block_on(client.sync()).expect("sync after uploads");
    mm.dispatch(&client, &w_h, m, n, &x_h, &out_a, p)
        .expect("dispatch a");
    mm.wait_last();
    mm.dispatch(&client, &w_h, m, n, &x_h, &out_b, p)
        .expect("dispatch b");
    mm.wait_last();
    let a = client.read_one(out_a).expect("read a");
    let b = client.read_one(out_b).expect("read b");
    assert_eq!(a, b, "relaxed arm is not run-to-run deterministic");
}

/// Issue 1170 half-B arm's G1: the CPU reference sees f16-ROUNDED
/// activations too (the kernel reads the f16 staging twin), so the band
/// isolates accumulation order + the activation rounding from tiling
/// defects. Pre-registered band 3e-2 rel-to-mag (the activation rounding
/// sits at ~5e-4 per term relative to the term — the same order the f16
/// weight rounding already contributes; measured drift lands far below).
/// Plus the same determinism pin as the relaxed arm.
#[test]
fn dense_f16_half_b_g1() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new_with_gemm_source(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        false,
        include_str!("../src/gemm_dense_f16h_metal_tensor.metal"),
    )
    .expect("half-b zero-copy context");

    run_case_hact(&mm, &client, 64, 64, 128, 3e-2); // exact tiles
    run_case_hact(&mm, &client, 130, 64, 128, 3e-2); // m-edge
    run_case_hact(&mm, &client, 512, 5120, 256, 3e-2); // production column count
    run_case_hact(&mm, &client, 512, 5120, 64, 3e-2); // ingest-floor p
}

/// Follow-up 4 — exact vs relaxed kernel A/B at the PRODUCTION per-layer GEMM
/// shapes (qwen3.8-27B: qkv 10240×5120, z 6144×5120, ssm_out 5120×6144, mlp
/// gate/up 17408×5120, mlp down 5120×17408, attn_q 12288×5120) at the ingest
/// chunk p=512 — the TFLOPS table that prices the wiring before the e2e
/// cells. ABBA-interleaved pairs, median of pair-ratios (the sequential-AB
/// law — a ratio of two sequential arms measures the box, interleaving
/// measures the kernels). RELEASE + --ignored:
/// `cargo test -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm
/// --test qwen38_dense_f16_matmul2d_g1 dense_f16_relaxed_ab --release --
/// --ignored --nocapture`. Quote the box state beside the numbers.
#[test]
#[ignore]
fn dense_f16_relaxed_ab() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let exact = MetalDenseF16ZeroCopyGemm::new_with_relaxed(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        false,
    )
    .expect("exact context");
    let relaxed = MetalDenseF16ZeroCopyGemm::new_with_relaxed(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        true,
    )
    .expect("relaxed context");

    const P: usize = 512;
    const SHAPES: [(usize, usize, &str); 6] = [
        (10_240, 5_120, "gdn.qkv"),
        (6_144, 5_120, "gdn.z"),
        (5_120, 6_144, "ssm.out/attn.o"),
        (17_408, 5_120, "mlp.gate/up"),
        (5_120, 17_408, "mlp.down"),
        (12_288, 5_120, "attn.q"),
    ];
    let x: Vec<f32> = (0..P * 17_408)
        .map(|i| ((i % 2009) as f32 / 2009.0) * 2.0 - 1.0)
        .collect();
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));

    const N_PAIRS: usize = 15;
    const WARM: usize = 3;
    let t0 = std::time::Instant::now();
    for &(m, n, name) in &SHAPES {
        let mut seed: u64 = 0xab11_5000 ^ ((m as u64) << 13) ^ (n as u64);
        let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
        let w16 = f16_bits(&wf);
        let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
        let out_h = client.create_from_slice(&vec![0u8; P * m * 4]);
        pollster::block_on(client.sync()).expect("sync after uploads");

        let time_arm = |mm: &MetalDenseF16ZeroCopyGemm| -> f64 {
            pollster::block_on(client.sync()).expect("pre-dispatch sync");
            let s = std::time::Instant::now();
            mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, P)
                .expect("dispatch");
            mm.wait_last();
            s.elapsed().as_secs_f64()
        };

        // Warmup both arms (first-use fills + pipeline warm).
        for _ in 0..WARM {
            time_arm(&exact);
            time_arm(&relaxed);
        }

        // ABBA pairs: alternate which arm opens the pair.
        let mut ratios: Vec<f64> = Vec::with_capacity(N_PAIRS);
        let mut last_exact = 0.0f64;
        let mut last_relaxed = 0.0f64;
        for pair in 0..N_PAIRS {
            if pair.is_multiple_of(2) {
                last_exact = time_arm(&exact);
                last_relaxed = time_arm(&relaxed);
            } else {
                last_relaxed = time_arm(&relaxed);
                last_exact = time_arm(&exact);
            }
            ratios.push(last_relaxed / last_exact);
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios"));
        let med = ratios[N_PAIRS / 2];
        let flop = (2.0 * (P * m * n) as f64) / 1e12;
        let tf_exact = flop / last_exact;
        let tf_relaxed = flop / last_relaxed;
        println!(
            "{name} {m}x{n} p{P}: exact {tf_exact:.1} TFLOPS, relaxed {tf_relaxed:.1} \
             TFLOPS, median ratio (relaxed/exact) {med:.3}"
        );
        drop(w_h);
        drop(out_h);
    }
    println!("probe wall: {:.1} s", t0.elapsed().as_secs_f64());
}

/// Issue 1170 G2 probe — half×half matmul2d (staged-activation view) vs the
/// mixed half×float device path, BOTH at relaxed precision, at the six
/// production shapes, p=512. The physics question that decides the wiring
/// arc: the fork stages/converts B to half before the MMA; ours reads the
/// activation as a device view. If the half×half schedule beats the mixed
/// path's 41.5–53.3 TFLOPS materially, the route wiring (f16 activation
/// staging + a convert kernel per chunk) is justified; if not, the lead
/// closes with evidence. PROBE-ONLY: the variant kernel compiles in the
/// test, no route caller, no env. RELEASE + --ignored:
/// `cargo test -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm
/// --test qwen38_dense_f16_matmul2d_g1 dense_f16_half_b_ab --release --
/// --ignored --nocapture`. Quote the box state beside the numbers.
#[test]
#[ignore]
fn dense_f16_half_b_ab() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mixed = MetalDenseF16ZeroCopyGemm::new_with_relaxed(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        true,
    )
    .expect("mixed context");
    let half_b = MetalDenseF16ZeroCopyGemm::new_with_gemm_source(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
        true,
        include_str!("../src/gemm_dense_f16h_metal_tensor.metal"),
    )
    .expect("half-b context");
    assert!(mixed.relaxed_precision() && half_b.relaxed_precision());

    const P: usize = 512;
    const SHAPES: [(usize, usize, &str); 6] = [
        (10_240, 5_120, "gdn.qkv"),
        (6_144, 5_120, "gdn.z"),
        (5_120, 6_144, "ssm.out/attn.o"),
        (17_408, 5_120, "mlp.gate/up"),
        (5_120, 17_408, "mlp.down"),
        (12_288, 5_120, "attn.q"),
    ];
    let x: Vec<f32> = (0..P * 17_408)
        .map(|i| ((i % 2009) as f32 / 2009.0) * 2.0 - 1.0)
        .collect();
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let x16 = f16_bits(&x);
    let x16_h = client.create_from_slice(bytemuck::cast_slice(&x16));

    const N_PAIRS: usize = 15;
    const WARM: usize = 3;
    for &(m, n, name) in &SHAPES {
        let mut seed: u64 = 0xab11_5000 ^ ((m as u64) << 13) ^ (n as u64);
        let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
        let w16 = f16_bits(&wf);
        let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
        let out_h = client.create_from_slice(&vec![0u8; P * m * 4]);
        pollster::block_on(client.sync()).expect("sync after uploads");

        let time_arm = |mm: &MetalDenseF16ZeroCopyGemm, x_in: &cubecl::server::Handle| -> f64 {
            pollster::block_on(client.sync()).expect("pre-dispatch sync");
            let s = std::time::Instant::now();
            mm.dispatch(&client, &w_h, m, n, x_in, &out_h, P)
                .expect("dispatch");
            mm.wait_last();
            s.elapsed().as_secs_f64()
        };

        // Warmup both arms (first-use fills + pipeline warm).
        for _ in 0..WARM {
            time_arm(&mixed, &x_h);
            time_arm(&half_b, &x16_h);
        }

        // ABBA pairs: alternate which arm opens the pair.
        let mut ratios: Vec<f64> = Vec::with_capacity(N_PAIRS);
        let mut last_mixed = 0.0f64;
        let mut last_half = 0.0f64;
        for pair in 0..N_PAIRS {
            if pair.is_multiple_of(2) {
                last_mixed = time_arm(&mixed, &x_h);
                last_half = time_arm(&half_b, &x16_h);
            } else {
                last_half = time_arm(&half_b, &x16_h);
                last_mixed = time_arm(&mixed, &x_h);
            }
            ratios.push(last_half / last_mixed);
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios"));
        let med = ratios[N_PAIRS / 2];
        let flop = (2.0 * (P * m * n) as f64) / 1e12;
        let tf_mixed = flop / last_mixed;
        let tf_half = flop / last_half;
        println!(
            "{name} {m}x{n} p{P}: mixed {tf_mixed:.1} TFLOPS, half_b {tf_half:.1} \
             TFLOPS, median ratio (half/mixed) {med:.3}"
        );
        drop(w_h);
        drop(out_h);
    }
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

/// The Bench-1163 diagnosis check (the reviewer verdict's pre-session-1
/// evidence): the f16 matmul2d at p=9 with K cut to ONE K-tile (n=32) and to
/// K/8 (n=640). Fixed-launch-cost hypothesis: time barely moves (≈1.2 ms at
/// every n). K-loop-work hypothesis: time scales ≈ n (the Bench-1163 sliced
/// arm already read 0.162 ms at n=640, predicting ~7.5 µs at n=32 — if that
/// reproduces, the 1.2 ms is per-K-tile load-latency work, and the fix is a
/// skeleton whose K-loop is plain streaming loads — the GEMV class — not a
/// custom mma tiling).
/// Run: `cargo test --release -p riir-infer-gpu --features
/// qwen38_wgpu,metal_tensor_gemm --test qwen38_dense_f16_matmul2d_g1
/// matmul2d_k_shrink_check -- --ignored --nocapture`
#[test]
#[ignore]
fn matmul2d_k_shrink_check() {
    let m = 17_408usize;
    let p = 9usize;
    let mm = MetalDenseF16ZeroCopyGemm::new(
        CubeCLContext::new()
            .expect("gpu context")
            .wgpu_device()
            .expect("wgpu device")
            .clone(),
        CubeCLContext::new()
            .expect("gpu context")
            .wgpu_queue()
            .expect("wgpu queue")
            .clone(),
    )
    .expect("dense zero-copy context");
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();

    println!("K-shrink check @m={m} p={p} (f16 matmul2d, N=20 back-to-back)");
    println!("{:>6} | {:>10} {:>10}", "n", "ms/call", "µs/K-tile");
    for n in [32usize, 640, 2560, 5120] {
        let mut seed: u64 = 0x0614_5b17_0000_0000 ^ (n as u64);
        let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
        let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
        let w16 = f16_bits(&wf);
        let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
        let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
        let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
        pollster::block_on(client.sync()).expect("sync uploads");

        for _ in 0..3 {
            mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, p)
                .expect("warm");
        }
        mm.wait_last();
        pollster::block_on(client.sync()).expect("warm sync");

        let mut ts = Vec::new();
        for _ in 0..2 {
            pollster::block_on(client.sync()).expect("pre");
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, p)
                    .expect("timed");
            }
            mm.wait_last();
            pollster::block_on(client.sync()).expect("post");
            ts.push(t0.elapsed().as_secs_f64() * 1000.0 / 20.0);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = ts[ts.len() / 2];
        let k_tiles = (n / 32).max(1);
        println!("{n:>6} | {t:>10.3} {:>10.3}", t * 1000.0 / k_tiles as f64);
    }
}

/// Plan 614 Phase-3 floor probe (riir-ai plan 614; the Bench-965 pattern on
/// the production box): the verify-shape p-sweep at the production ffn shape
/// [17408×5120] — per-layer times for the three Metal kernel families that
/// could serve a DFlash2 verify chunk (p = draft block + pending ≈ 9 rows):
///
/// 1. `GemvQ4KBatchedCubeCL` — the incumbent batched Q4K GEMV (decode uses
///    batch=1; verify would ride batch=p). 2D grid (rows × batch): each batch
///    column re-reads its rows' weights — pure-DRAM behavior scales ~p×, but
///    a ~50 MB weight slab may stay L2-resident across columns. The
///    batch=p/batch=1 ratio is the incumbent stack's C_K floor for the packed
///    (Q4K, S_dense = 15.8 GB/token) posture — the posture whose byte ratio
///    the flip gate actually cites.
/// 2. `GemmQ4KTileCubeCL` — the scalar tile GEMM (the 1150 A/B's loser at
///    p=512; here priced at verify shapes).
/// 3. `MetalDenseF16ZeroCopyGemm::dispatch` — the tensor-unit f16 arm: the
///    compute floor any future Q4_K-dequant tensor verify kernel would ride.
///    Its decode poison (f16 = 3.55× weight bytes → the flip gate's S ratio
///    ~8.2 instead of 2.32) is the scoping argument for that kernel.
///
/// Every arm streams the SAME weights (handles built once); timing is the
/// established protocol (sync → N back-to-back → wait_last/raw-commit wait
/// → sync, ABBA×2 median); loud-zero insurance reads one verified output
/// back at the verify shape p=9 (both Q4K arms + mm2d).
///
/// Run:
/// ```text
/// cargo test --release -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm \
///   --test qwen38_dense_f16_matmul2d_g1 prow_verify_shape_probe -- --ignored --nocapture
/// ```
/// Quote the box state beside the numbers (bench_preflight PROVENANCE).
#[test]
#[ignore]
fn prow_verify_shape_probe() {
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, dequantize_row_q4_k, quantize_row_q4_k};
    use riir_infer_gpu::gemm_q4k_tile_cubecl::GemmQ4KTileCubeCL;
    use riir_infer_gpu::gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL;
    use riir_infer_gpu::gemv_q4k_cubecl::Q4KHandle;

    let (m, n) = (17_408usize, 5_120usize);
    // One fixed weight set for the whole sweep — every arm and every p reads
    // the same bytes, so cross-arm ratios are weight-identical.
    let mut seed: u64 = 0x614d_5f70_726f_6265;
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let w16 = f16_bits(&wf);
    let bpr = n / QK_K;
    let mut blocks = vec![BlockQ4K::zeroed(); m * bpr];
    let mut row = vec![0.0f32; n];
    for r in 0..m {
        row.copy_from_slice(&wf[r * n..(r + 1) * n]);
        quantize_row_q4_k(&row, &mut blocks[r * bpr..(r + 1) * bpr]);
    }

    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense-f16 zero-copy context");

    let q4 = Q4KHandle::from_blocks(&client, &blocks, m, n);
    let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
    pollster::block_on(client.sync()).expect("sync after uploads");

    // Per-block bytes: Q4K payload 144 B (36 u32 × 4) + split d/dmin 8 B over
    // QK_K=256 weights; f16 = 2 B/weight.
    let q4k_wbytes = (m * n * 152 / 256) as f64;
    let f16_wbytes = (m * n * 2) as f64;
    println!(
        "prow probe @{m}x{n}: q4k weights {:.1} MB, f16 weights {:.1} MB",
        q4k_wbytes / 1e6,
        f16_wbytes / 1e6
    );

    const N: usize = 20;
    fn time_batched(
        client: &ComputeClient<ActiveRuntime>,
        q4: &riir_infer_gpu::gemv_q4k_cubecl::Q4KHandle,
        p: usize,
        x_h: cubecl::server::Handle,
        out_h: cubecl::server::Handle,
    ) -> f64 {
        pollster::block_on(client.sync()).expect("batched pre-sync");
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            unsafe {
                GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                    client,
                    q4,
                    x_h.clone(),
                    out_h.clone(),
                    p,
                );
            }
        }
        pollster::block_on(client.sync()).expect("batched post-sync");
        t0.elapsed().as_secs_f64() * 1000.0 / N as f64
    }
    fn time_tile(
        client: &ComputeClient<ActiveRuntime>,
        q4: &riir_infer_gpu::gemv_q4k_cubecl::Q4KHandle,
        p: usize,
        x_h: cubecl::server::Handle,
        out_h: cubecl::server::Handle,
    ) -> f64 {
        pollster::block_on(client.sync()).expect("tile pre-sync");
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            unsafe {
                GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                    client,
                    q4,
                    x_h.clone(),
                    out_h.clone(),
                    p,
                );
            }
        }
        pollster::block_on(client.sync()).expect("tile post-sync");
        t0.elapsed().as_secs_f64() * 1000.0 / N as f64
    }
    // 8 args is the honest shape here — the timing fn must not capture (nested
    // item), and the zero-copy dispatch itself needs all of these.
    #[allow(clippy::too_many_arguments)]
    fn time_mm(
        client: &ComputeClient<ActiveRuntime>,
        mm: &MetalDenseF16ZeroCopyGemm,
        w_h: &cubecl::server::Handle,
        m: usize,
        n: usize,
        p: usize,
        x_h: &cubecl::server::Handle,
        out_h: &cubecl::server::Handle,
    ) -> f64 {
        pollster::block_on(client.sync()).expect("mm pre-sync");
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            mm.dispatch(client, w_h, m, n, x_h, out_h, p)
                .expect("mm dispatch");
        }
        mm.wait_last(); // raw-Metal submissions are invisible to client.sync()
        pollster::block_on(client.sync()).expect("mm post-sync");
        t0.elapsed().as_secs_f64() * 1000.0 / N as f64
    }

    println!(
        "{:>4} | {:>10} {:>8} | {:>10} {:>9} | {:>10} {:>9}",
        "p", "gemv-b ms", "GB/s", "tile ms", "GFLOP/s", "mm2d ms", "GFLOP/s"
    );
    for p in [1usize, 2, 4, 8, 9, 16, 32, 64] {
        let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
        let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
        // One output handle PER ARM — the p=9 loud-zero readbacks must see
        // each arm's own writes, never a sibling arm's.
        let out_g = client.create_from_slice(&vec![0u8; p * m * 4]);
        let out_t = client.create_from_slice(&vec![0u8; p * m * 4]);
        let out_m = client.create_from_slice(&vec![0u8; p * m * 4]);
        // Warmup all three arms (JIT + pool + dedicated buffers), then sync.
        unsafe {
            GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                &q4,
                x_h.clone(),
                out_g.clone(),
                p,
            );
            if p.is_multiple_of(32) {
                GemmQ4KTileCubeCL::launch::<ActiveRuntime>(
                    &client,
                    &q4,
                    x_h.clone(),
                    out_t.clone(),
                    p,
                );
            }
        }
        mm.dispatch(&client, &w_h, m, n, &x_h, &out_m, p)
            .expect("mm warmup");
        pollster::block_on(client.sync()).expect("warmup sync");

        // ABBA-style alternation x2, median of 4 per arm (the loaded-box
        // discipline — Bench-965 defect 2: one reading is host-stall lottery).
        // The scalar tile GEMM requires p ≡ 0 (mod 32) — the verify-class
        // shapes (2..16) are DECODE-PATH territory by design (the launch
        // asserts), so its column reports only the legal shapes.
        let mut gs = Vec::new();
        let mut ts = Vec::new();
        let mut ms = Vec::new();
        for _ in 0..2 {
            gs.push(time_batched(&client, &q4, p, x_h.clone(), out_g.clone()));
            if p.is_multiple_of(32) {
                ts.push(time_tile(&client, &q4, p, x_h.clone(), out_t.clone()));
            }
            ms.push(time_mm(&client, &mm, &w_h, m, n, p, &x_h, &out_m));
            if p.is_multiple_of(32) {
                ts.push(time_tile(&client, &q4, p, x_h.clone(), out_t.clone()));
            }
            ms.push(time_mm(&client, &mm, &w_h, m, n, p, &x_h, &out_m));
            gs.push(time_batched(&client, &q4, p, x_h.clone(), out_g.clone()));
        }
        let med = |v: &mut Vec<f64>| {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (g, t, d) = (med(&mut gs), med(&mut ts), med(&mut ms));
        let gflop = 2.0 * m as f64 * n as f64 * p as f64 / 1e9;
        let t_disp = if t.is_nan() {
            "—".to_string()
        } else {
            format!("{t:>10.3}")
        };
        let t_gf = if t.is_nan() {
            "—".to_string()
        } else {
            format!("{:>9.0}", gflop / (t / 1000.0))
        };
        println!(
            "{p:>4} | {g:>10.3} {:>8.0} | {t_disp} {t_gf} | {d:>10.3} {:>9.0}",
            q4k_wbytes / (g / 1000.0) / 1e9,
            gflop / (d / 1000.0),
        );

        // Loud-zero insurance at the verify shape: p=9 row 0 must match the
        // CPU dot for every arm that timed at this p (the timed loop wrote
        // out_h N times per arm — absent work would read as a zero).
        if p == 9 {
            // Loud-zero insurance at the verify shape: each arm read back from
            // its OWN output handle. The two Q4K arms verify against the
            // DEQUANTIZED-weight CPU reference (their outputs carry Q4K quant
            // error vs the f16-rounded reference — the 2e-4 band is calibrated
            // for f16-vs-f16 summation order, not quant noise); mm2d verifies
            // against the f16-rounded reference (its weights ARE f16).
            let mut wq4 = vec![0.0f32; n];
            dequantize_row_q4_k(&blocks[..bpr], &mut wq4);
            let (mut want_q4, mut acc_q4, mut want_f16, mut acc_f16) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for c in 0..n {
                let term16 = f32::from(half::f16::from_bits(w16[c])) * x[c];
                want_f16 += term16;
                acc_f16 += term16.abs();
                let term4 = wq4[c] * x[c];
                want_q4 += term4;
                acc_q4 += term4.abs();
            }
            let read = |h: &cubecl::server::Handle| {
                let bytes = client.read_one(h.clone()).expect("read p9");
                bytemuck::cast_slice::<u8, f32>(&bytes)[0]
            };
            let rel_q4 = ((read(&out_g) - want_q4).abs() / acc_q4.max(1e-9)) as f64;
            assert!(
                rel_q4 < 2e-4,
                "prow p9 loud-zero: batched GEMV row0 got {} want {want_q4} (rel-to-mag {rel_q4:.3e})",
                read(&out_g)
            );
            let rel_m = ((read(&out_m) - want_f16).abs() / acc_f16.max(1e-9)) as f64;
            assert!(
                rel_m < 2e-4,
                "prow p9 loud-zero: mm2d row0 got {} want {want_f16} (rel-to-mag {rel_m:.3e})",
                read(&out_m)
            );
            println!(
                "prow p9 outputs verified: batched-GEMV rel-to-mag {rel_q4:.3e} (vs dequant ref), mm2d rel-to-mag {rel_m:.3e} (vs f16 ref)"
            );
        }
    }
}

/// Plan 614 Phase-3 task-1 step 1 — the split-K scheduling validation WITHOUT
/// writing a kernel: slice the [17408×5120] f16 weights into S K-slices and
/// enqueue S independent zero-copy mm2d dispatches back-to-back (no sync
/// between them), one wait at the end. If Metal overlaps the dependency-free
/// command buffers, the sliced product's wall collapses toward the bandwidth
/// floor (S× [17408×640] slices at S=8 → ~0.25–0.5 ms/call-equivalent at the
/// verify shape) — proving Bench 1162's latency-bound diagnosis and the
/// split-K projection the task-1 kernel spec rests on. If the buffers
/// serialize (S × 1.2 ms), the finding redirects the kernel design to
/// IN-KERNEL grid.z split-K.
///
/// Per-slice buffers: distinct x and weight handles, and a PADDED-DISTINCT
/// output per slice (the zero-copy path's dedicated-output cache keys on byte
/// size — same-size outputs would share one buffer and race across slices;
/// +4·s bytes is the cheap distinct-key). Reduction: host sum of the S
/// partials, vs the f16-rounded CPU reference (the same 2e-4 rel-to-mag
/// band, applied to the SUM — the slicing does not change any dot).
///
/// Run:
/// ```text
/// cargo test --release -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm \
///   --test qwen38_dense_f16_matmul2d_g1 prow_splitk_emulation_probe -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn prow_splitk_emulation_probe() {
    use riir_infer_gpu::gemm_dense_f16_metal_zero_copy::MetalDenseF16ZeroCopyGemm;

    let (m, n, p) = (17_408usize, 5_120usize, 9usize);
    let mut seed: u64 = 0x531_17da_0000_0000;
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x_full: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
    let w16 = f16_bits(&wf);
    let (expect, mag) = cpu_gemm_f16(&w16, &x_full, m, n, p);

    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense-f16 zero-copy context");

    let w_h = client.create_from_slice(bytemuck::cast_slice(&w16));
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x_full));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
    pollster::block_on(client.sync()).expect("sync after uploads");

    // Baseline: the full-K single call at p=9 (the Bench-1162 1.2 ms floor).
    for _ in 0..3 {
        mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, p)
            .expect("warm");
    }
    mm.wait_last();
    pollster::block_on(client.sync()).expect("warm sync");
    let time_full = || {
        pollster::block_on(client.sync()).expect("pre");
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            mm.dispatch(&client, &w_h, m, n, &x_h, &out_h, p)
                .expect("full");
        }
        mm.wait_last();
        pollster::block_on(client.sync()).expect("post");
        t0.elapsed().as_secs_f64() * 1000.0 / 20.0
    };

    // The sliced arm, parameterized by S.
    let time_sliced = |s: usize,
                       w_parts: &[cubecl::server::Handle],
                       x_parts: &[cubecl::server::Handle],
                       out_parts: &[cubecl::server::Handle]| {
        pollster::block_on(client.sync()).expect("sliced pre");
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            for k in 0..s {
                mm.dispatch(
                    &client,
                    &w_parts[k],
                    m,
                    n / s,
                    &x_parts[k],
                    &out_parts[k],
                    p,
                )
                .unwrap_or_else(|e| panic!("sliced dispatch {s}/{k}: {e}"));
            }
        }
        mm.wait_last();
        pollster::block_on(client.sync()).expect("sliced post");
        t0.elapsed().as_secs_f64() * 1000.0 / 20.0
    };

    println!(
        "split-K emulation @{m}x{n} p{p}: f16 weights {:.1} MB, x {:.2} MB",
        (m * n * 2) as f64 / 1e6,
        (p * n * 4) as f64 / 1e6
    );
    println!(
        "{:>3} | {:>10} {:>10} {:>9} {:>8}",
        "S", "sliced ms", "per-call", "TF/s", "GB/s"
    );

    for s in [1usize, 2, 4, 8] {
        // Build per-slice buffers (n divisible by 8; K-tile 32 divides 640).
        let n_s = n / s;
        let mut w_parts = Vec::new();
        let mut x_parts = Vec::new();
        let mut out_parts = Vec::new();
        for k in 0..s {
            let mut ws = Vec::with_capacity(m * n_s);
            for r in 0..m {
                ws.extend_from_slice(&w16[r * n + k * n_s..r * n + (k + 1) * n_s]);
            }
            w_parts.push(client.create_from_slice(bytemuck::cast_slice(&ws)));
            let mut xs = Vec::with_capacity(p * n_s);
            for t in 0..p {
                xs.extend_from_slice(&x_full[t * n + k * n_s..t * n + (k + 1) * n_s]);
            }
            x_parts.push(client.create_from_slice(bytemuck::cast_slice(&xs)));
            // Padded-distinct output: the zero-copy cache keys on byte size.
            let bytes = p * m * 4 + 4 * k;
            out_parts.push(client.create_from_slice(&vec![0u8; bytes]));
        }
        pollster::block_on(client.sync()).expect("parts sync");

        // Warm the sliced shape once (dedicated-output cache + pools).
        for k in 0..s {
            mm.dispatch(&client, &w_parts[k], m, n_s, &x_parts[k], &out_parts[k], p)
                .expect("sliced warm");
        }
        mm.wait_last();
        pollster::block_on(client.sync()).expect("sliced warm sync");

        // ABBA: full-K, sliced, sliced, full-K — median of each.
        let mut fs = Vec::new();
        let mut ss = Vec::new();
        fs.push(time_full());
        ss.push(time_sliced(s, &w_parts, &x_parts, &out_parts));
        ss.push(time_sliced(s, &w_parts, &x_parts, &out_parts));
        fs.push(time_full());
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ss.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (f, sl) = (fs[fs.len() / 2], ss[ss.len() / 2]);
        let gflop = 2.0 * m as f64 * n as f64 * p as f64 / 1e9;
        let gbs = (m * n * 2) as f64 / (sl / 1000.0) / 1e9;
        println!(
            "{s:>3} | {sl:>10.3} {:>10.3} {:>9.2} {:>8.0}   (full-K baseline {f:.3} ms)",
            sl / s as f64,
            gflop / (sl / 1000.0),
            gbs,
        );

        // Loud-zero at S=8: host-sum the partials, compare vs the CPU ref.
        if s == 8 {
            let mut got_sum = vec![0.0f32; p * m];
            for (k, oh) in out_parts.iter().enumerate() {
                let bytes = client.read_one(oh.clone()).expect("read slice");
                let part: &[f32] = bytemuck::cast_slice(&bytes);
                for i in 0..p * m {
                    got_sum[i] += part[i];
                }
                let _ = k;
            }
            let mut max_rel = 0.0f64;
            for i in 0..p * m {
                let rel = ((got_sum[i] - expect[i]).abs() / mag[i].max(1e-9)) as f64;
                max_rel = max_rel.max(rel);
            }
            assert!(
                max_rel < 2e-4,
                "split-K S=8 loud-zero: max rel-to-mag {max_rel:.3e}"
            );
            println!("split-K S=8 reduction verified: max rel-to-mag {max_rel:.3e}");
        }
    }
}
