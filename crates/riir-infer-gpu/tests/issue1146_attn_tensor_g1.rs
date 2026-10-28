//! Issue 1146 ladder 4(c) — G1 of the matmul2d flash-attention prefill
//! (`MetalTensorAttentionPrefill`) against the production macOS arm
//! (`QwenAttentionPrefillTiledM32CubeCL`), on synthetic tensors.
//!
//! A TOLERANCE gate: the tensor arm stages Q and P as half, so it is never
//! bit-identical. Per case it asserts, over every `(row, head)` output vector:
//! the relative error `max_abs / max|ref|` under `REL_BAR`, and the vector's
//! argmax agreeing on at least `ARGMAX_FLOOR` of them. The tensor arm runs
//! FIRST in every case, on fresh handles — its output takes the same
//! dedicated-buffer + blit-back route as the zero-copy GEMM, whose first-use
//! wgpu zero-init hazard only a cold dispatch exposes
//! (`issue1146_tensor_range_g1`'s ordering).
//!
//! Cases cover the minimum routable cache (64 rows), ragged tiles, chunked
//! prefill (`base_pos > 0`, a small chunk against a long cache), two GQA
//! ratios, and PEAKED scores (K scaled ×4) that exercise the lazy online
//! rescale (it moves the running max only past +8).
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1146a cargo test -p riir-infer-gpu --release \
//!   --features metal_tensor_gemm,cubecl_runtime,ternary_gemm_batched,ternary_attention_batched_prefill \
//!   --test issue1146_attn_tensor_g1 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `I1146A_LONG=1` adds a p=4096 case.

#![cfg(all(
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    target_os = "macos"
))]

use cubecl::client::ComputeClient;
use cubecl::server::Handle;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::qwen_attention_prefill_m32_cubecl::QwenAttentionPrefillTiledM32CubeCL;
use riir_infer_gpu::{MetalTensorAttentionPrefill, MetalTensorZeroCopyGemm};

const HD: usize = 256;
/// Relative error bar (`max_abs / max|ref|`): half Q/P carry ~1e-3 relative
/// rounding; 1e-2 leaves the margin a reassociated softmax needs.
const REL_BAR: f32 = 1e-2;
/// Fraction of `(row, head)` vectors whose argmax must agree.
const ARGMAX_FLOOR: f64 = 0.99;

/// Deterministic uniform `[-amp, amp)` (an LCG — a fixture, not a sampler).
fn fill(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn upload(client: &ComputeClient<ActiveRuntime>, v: &[f32]) -> Handle {
    let h = client.empty(v.len() * 4);
    client.write(
        &h,
        cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(v).to_vec()),
    );
    h
}

fn read(client: &ComputeClient<ActiveRuntime>, h: &Handle) -> Vec<f32> {
    let b = client.read_one(h.clone()).expect("read_one");
    bytemuck::cast_slice::<u8, f32>(&b[..]).to_vec()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

/// f64 CPU reference — the third opinion when the two GPU arms disagree.
#[allow(clippy::too_many_arguments, reason = "the attention handle set")]
fn cpu_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    n_head: usize,
    n_kv: usize,
    p: usize,
    base_pos: usize,
) -> Vec<f32> {
    let scale = 1.0 / (HD as f64).sqrt();
    let mut out = vec![0.0f32; p * n_head * HD];
    for i in 0..p {
        let q_abs = base_pos + i;
        for h in 0..n_head {
            let kvh = h * n_kv / n_head;
            let qo = (i * n_head + h) * HD;
            let scores: Vec<f64> = (0..=q_abs)
                .map(|t| {
                    let ko = (t * n_kv + kvh) * HD;
                    (0..HD)
                        .map(|d| q[qo + d] as f64 * k[ko + d] as f64)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for d in 0..HD {
                let o: f64 = w
                    .iter()
                    .enumerate()
                    .map(|(t, wt)| wt * v[(t * n_kv + kvh) * HD + d] as f64)
                    .sum::<f64>()
                    / z;
                let gate = 1.0 / (1.0 + (-(g[qo + d] as f64)).exp());
                out[qo + d] = (o * gate) as f32;
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

struct Case {
    label: &'static str,
    n_head: usize,
    n_kv: usize,
    p: usize,
    base_pos: usize,
    k_amp: f32,
}

#[test]
#[ignore = "requires the Metal GPU; run explicitly with --release"]
fn tensor_attention_matches_m32_within_tolerance() {
    let ctx = CubeCLContext::new().expect("GPU init");
    let client = ctx.client();
    let gemm = MetalTensorZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("zero-copy context");
    let attn = MetalTensorAttentionPrefill::new(&gemm).expect("tensor attention pipeline");

    let mut cases = vec![
        Case {
            label: "min-cache",
            n_head: 8,
            n_kv: 2,
            p: 64,
            base_pos: 0,
            k_amp: 1.0,
        },
        Case {
            label: "ragged",
            n_head: 8,
            n_kv: 2,
            p: 100,
            base_pos: 0,
            k_amp: 1.0,
        },
        Case {
            label: "tiny-chunk",
            n_head: 8,
            n_kv: 2,
            p: 4,
            base_pos: 60,
            k_amp: 1.0,
        },
        Case {
            label: "chunked",
            n_head: 8,
            n_kv: 2,
            p: 256,
            base_pos: 256,
            k_amp: 1.0,
        },
        Case {
            label: "gqa6",
            n_head: 24,
            n_kv: 4,
            p: 512,
            base_pos: 0,
            k_amp: 1.0,
        },
        Case {
            label: "amp2-s",
            n_head: 8,
            n_kv: 2,
            p: 128,
            base_pos: 0,
            k_amp: 2.0,
        },
        Case {
            label: "amp3-s",
            n_head: 8,
            n_kv: 2,
            p: 128,
            base_pos: 0,
            k_amp: 3.0,
        },
        Case {
            label: "amp4-s",
            n_head: 8,
            n_kv: 2,
            p: 128,
            base_pos: 0,
            k_amp: 4.0,
        },
        Case {
            label: "peaked",
            n_head: 24,
            n_kv: 4,
            p: 512,
            base_pos: 0,
            k_amp: 4.0,
        },
        Case {
            label: "long-cache",
            n_head: 24,
            n_kv: 4,
            p: 1000,
            base_pos: 3000,
            k_amp: 1.0,
        },
    ];
    if std::env::var("I1146A_LONG").as_deref() == Ok("1") {
        cases.push(Case {
            label: "pp4096",
            n_head: 24,
            n_kv: 4,
            p: 4096,
            base_pos: 0,
            k_amp: 1.0,
        });
    }

    // `I1146A_REPEAT=n` runs the whole case list n times (nondeterminism probe).
    let repeat: usize = std::env::var("I1146A_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let n_cases = cases.len();
    let cases: Vec<&Case> = (0..repeat).flat_map(|_| cases.iter()).collect();
    let mut failures = Vec::new();
    for (ci, c) in cases.iter().enumerate() {
        let ci = ci % n_cases;
        let rows = c.base_pos + c.p;
        let seed = 1000 * ci as u64;
        let q = fill(c.p * c.n_head * HD, seed + 1, 1.0);
        let k = fill(rows * c.n_kv * HD, seed + 2, c.k_amp);
        let v = fill(rows * c.n_kv * HD, seed + 3, 1.0);
        let g = fill(c.p * c.n_head * HD, seed + 4, 3.0);
        let (qh, kh, vh, gh) = (
            upload(&client, &q),
            upload(&client, &k),
            upload(&client, &v),
            upload(&client, &g),
        );
        let out_len = c.p * c.n_head * HD;
        let out_t = client.empty(out_len * 4);
        let out_r = client.empty(out_len * 4);
        // Host uploads are deferred writes; the dispatch's `flush()` orders
        // KERNEL producers (the production case), but a raw commit can land
        // before a queued `write` reaches the pool — reading the previous
        // case's data from reused memory. Sync so the fixture is settled
        // (`I1146A_NO_SYNC=1` drops it, to reproduce that hazard).
        if std::env::var("I1146A_NO_SYNC").as_deref() != Ok("1") {
            pollster::block_on(client.sync()).expect("sync after uploads");
        }

        // Tensor arm FIRST, cold.
        attn.dispatch(
            &gemm, &client, &qh, &kh, &vh, &gh, &out_t, c.n_head, c.n_kv, c.p, c.base_pos,
        )
        .expect("tensor dispatch");
        attn.wait_last();
        let got = read(&client, &out_t);

        unsafe {
            QwenAttentionPrefillTiledM32CubeCL::launch::<ActiveRuntime>(
                &client,
                qh.clone(),
                kh.clone(),
                vh.clone(),
                gh.clone(),
                out_r.clone(),
                HD,
                c.n_head,
                c.n_kv,
                c.p,
                c.base_pos,
            );
        }
        pollster::block_on(client.sync()).expect("sync");
        let want = read(&client, &out_r);

        assert!(
            got.iter().all(|x| x.is_finite()),
            "{}: non-finite tensor output",
            c.label
        );
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_ref = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let rel = max_abs / max_ref.max(f32::MIN_POSITIVE);
        let vectors = c.p * c.n_head;
        let agree = got
            .chunks(HD)
            .zip(want.chunks(HD))
            .filter(|(a, b)| argmax(a) == argmax(b))
            .count();
        let agree_frac = agree as f64 / vectors as f64;
        println!(
            "{:<10} heads {}/{} p {:>4} base {:>4} · max_abs {max_abs:.3e} · max|ref| {max_ref:.3e} · rel {rel:.3e} · argmax {agree}/{vectors}",
            c.label, c.n_head, c.n_kv, c.p, c.base_pos
        );
        // Third opinion on the cheap cases (causal cost ~ p·rows·heads·HD).
        if c.p * (c.base_pos + c.p) * c.n_head <= 8_000_000 {
            let cpu = cpu_ref(&q, &k, &v, &g, c.n_head, c.n_kv, c.p, c.base_pos);
            println!(
                "{:<10} vs f64 CPU · tensor max_abs {:.3e} · m32 max_abs {:.3e}",
                "",
                max_abs_diff(&got, &cpu),
                max_abs_diff(&want, &cpu)
            );
        }
        if rel > REL_BAR || agree_frac < ARGMAX_FLOOR {
            failures.push(format!(
                "{}: rel {rel:.3e}, argmax {agree}/{vectors}",
                c.label
            ));
        }
    }
    assert!(
        attn.dispatch_count() as usize == cases.len(),
        "vacuous guard: tensor arm did not run every case"
    );
    assert!(
        failures.is_empty(),
        "tensor attention outside tolerance: {failures:?}"
    );
}
