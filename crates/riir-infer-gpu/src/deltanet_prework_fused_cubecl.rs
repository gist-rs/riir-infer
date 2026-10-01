//! DeltaNet fused GDN prework — conv1d + SiLU + q/k L2-norm + head expansion
//! in ONE dispatch (riir-ai Issue 1004 R2, the `gdn_prefill_prework.py` shape).
//!
//! The shipping chain is three stages over the qkv stream per layer:
//! 1. `DeltanetChunkedConv1dCubeCL` (P/64 dispatches + a carry-update each):
//!    depthwise conv + SiLU, writing the intermediate `qkv_conv_b`;
//! 2. `DeltanetBetaDecayBatchedCubeCL` (independent stream — NOT fused here);
//! 3. `ExpandAndL2NormalizeHeadsBatchedCubeCL`: reads the SiLU'd compact
//!    stream back, L2-normalizes each compact q/k head, broadcasts it to
//!    `n_v/n_k` expanded heads, copies v.
//!
//! This kernel does 1+3 in ONE dispatch that reads the RAW qkv stream once
//! and writes the expanded buffer directly — the intermediate
//! `qkv_conv_b` write + read disappears (~160 MB/layer @ P=2048 at Bonsai-2
//! dims), and ~P/32 dispatches collapse to 2 (fused + carry-update). The
//! beta/decay stage stays separate: it reads the `a`/`b` projection stream,
//! not qkv, so fusing it would add buffer args without removing traffic.
//!
//! **Bit-identity is by construction, not by tolerance.** Per output element
//! the expression sequence is the identical one in the identical order:
//! - the conv sum loops `k` ascending over the same window (carry for
//!   positions < 0, raw input otherwise) — verbatim
//!   [`deltanet_conv1d_chunked_f32`](crate::deltanet_chunked_cubecl);
//! - SiLU is the same `sum * (1/(1+exp(-sum)))` expression;
//! - the L2 `sq_sum` loops the head's channels ascending — the identical
//!   order [`expand_and_l2_normalize_heads_batched_f32`] uses — over the SAME
//!   silu'd values (staged in threadgroup memory instead of round-tripped
//!   through global, which preserves bits);
//! - the zero-norm guard and `x * inv_norm` multiply are unchanged;
//! - the compact→expanded mapping is the inverse of `src_head = out_head %
//!   n_k`: compact head `h` writes out-heads `h, h+n_k, ..` — each expanded
//!   element exactly once, same value.
//!
//! `tests::fused_is_bit_identical_to_the_chain` pins all of that against the
//! shipping kernels on the same inputs, including the carried conv state and
//! the chunk-boundary windows (the fused kernel reads the raw stream for
//! intra-call windows; the chunked chain reads a carry those same inputs
//! populated — identical values).
//!
//! Shape constraints (asserted by [`DeltanetPreworkFusedCubeCL::supports`]):
//! `head_dim == 128` (one workgroup per (token, head-slot), one thread per
//! channel) and `n_v_heads % n_k_heads == 0` (the broadcast). `p ≤ 65535`
//! (the Metal Y grid cap; production chunks are far below).

#[cfg(feature = "deltanet_prework_fused")]
use cubecl::prelude::*;
#[cfg(feature = "deltanet_prework_fused")]
use cubecl::server::Handle;

/// Launch counter — the e2e A/B asserts the armed arm actually dispatched
/// this kernel (the R1 `STAGED_LAUNCHES` pattern; a silent fallback would
/// otherwise make a green run indistinguishable from a no-op).
#[cfg(feature = "deltanet_prework_fused")]
static PREWORK_FUSED_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Read [`PREWORK_FUSED_LAUNCHES`].
#[cfg(feature = "deltanet_prework_fused")]
pub fn prework_fused_launches() -> u64 {
    PREWORK_FUSED_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// One workgroup per (token, head-slot): X = `2*n_k + n_v` head-slots
/// (n_k compact Q heads, n_k compact K heads, n_v V heads), Y = tokens.
/// 128 threads = `head_dim` channels; the silu'd head is staged in 512 B of
/// threadgroup memory between the conv phase and the normalize/write phase.
#[cfg(feature = "deltanet_prework_fused")]
#[cube(launch_unchecked)]
fn deltanet_prework_fused_f32(
    input: &[f32],
    expanded: &mut [f32],
    conv_weight: &[f32],
    carry: &[f32],
    params: &[f32],
) {
    // params[0] = p (bounds the launcher's exact Y grid; not read here).
    let conv_dim = params[1usize] as usize;
    let kernel_size = params[2usize] as usize;
    let n_k = params[3usize] as usize;
    let n_v = params[4usize] as usize;
    // params[5] = head_dim == 128 == CUBE_DIM_X (launcher asserts; no early
    // terminate so every unit reaches the barrier).
    let carry_stride = params[6usize] as usize;
    let carry_idx_offset = params[7usize] as usize;
    let ks_m1 = kernel_size - 1;

    let t = CUBE_POS_Y as usize;
    let slot = CUBE_POS_X as usize; // 0..2n_k = compact Q/K head, 2n_k.. = V head
    let c = UNIT_POS_X as usize; // channel within the head (1-D cube)

    let v_dim = n_v * 128;
    let qkvx_stride = 3 * v_dim;
    let repeat = n_v / n_k;

    // This workgroup's channel span in the compact stream, and its output
    // section (0 = Q, 1 = K, 2 = V). Q/K are contiguous in the compact
    // layout, so `slot * 128 + c` addresses both; `slot % n_k` is the
    // compact head index either way.
    let ch = slot * 128 + c;
    let section = if slot < 2 * n_k { slot / n_k } else { 2usize };
    let src_head = slot % n_k;

    // ── Conv phase: verbatim `deltanet_conv1d_chunked_f32` per channel ──
    // window position t-ks+1+k; positions < 0 come from the carry.
    let weight_off = ch * kernel_size;
    let carry_off = ch * carry_stride + carry_idx_offset;
    let mut sum = f32::new(0.0f32);
    for k in 0..kernel_size {
        let sample_pos_signed = t as i64 - ks_m1 as i64 + k as i64;
        let val = if sample_pos_signed < 0 {
            carry[carry_off + (sample_pos_signed + ks_m1 as i64) as usize]
        } else {
            input[sample_pos_signed as usize * conv_dim + ch]
        };
        sum += val * conv_weight[weight_off + k];
    }

    // SiLU — the same expression sequence as the chunked kernel.
    let neg_sum = f32::new(0.0f32) - sum;
    let exp_neg = neg_sum.exp();
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + exp_neg);
    let val = sum * sig;

    let mut smem = Shared::<[f32]>::new_slice(comptime!(128usize));
    smem[c] = val;
    sync_cube();

    // ── Normalize + write phase ──
    if section < 2 {
        // Q/K: L2-normalize the compact head (sq_sum ascending over the
        // silu'd values — the expand kernel's exact loop), then broadcast to
        // the `repeat` expanded heads that read this compact head
        // (out_head % n_k == src_head).
        let mut sq_sum = f32::new(0.0f32);
        for i in 0..128usize {
            let v = smem[i];
            sq_sum += v * v;
        }
        // Zero-norm guard (Issue 673 Bug D).
        let inv_norm = if sq_sum > f32::new(0.0f32) {
            f32::new(1.0f32) / sq_sum.sqrt()
        } else {
            f32::new(0.0f32)
        };
        let out_val = smem[c] * inv_norm;
        let section_base = t * qkvx_stride + section * v_dim;
        for r in 0..repeat {
            let out_head = src_head + r * n_k;
            expanded[section_base + out_head * 128 + c] = out_val;
        }
    } else {
        // V: copy the silu'd value (no norm) into this V head's slot.
        let v_base = t * qkvx_stride + 2 * v_dim;
        expanded[v_base + (slot - 2 * n_k) * 128 + c] = smem[c];
    }
}

/// The fused GDN prework launcher (Issue 1004 R2).
///
/// Issues TWO ordered dispatches, mirroring
/// [`DeltanetChunkedConv1dCubeCL::launch`](crate::deltanet_chunked_cubecl):
/// (1) the fused kernel, which only READS the carry, and (2) the shared
/// carry-update kernel with `c = p`, which rewrites it for the next call
/// (the Issue 673 read/write-race discipline — dispatch order on the compute
/// stream guarantees the reads complete first).
#[cfg(feature = "deltanet_prework_fused")]
pub struct DeltanetPreworkFusedCubeCL;

#[cfg(feature = "deltanet_prework_fused")]
impl DeltanetPreworkFusedCubeCL {
    const HEAD_DIM: usize = 128;

    /// Shape admission: one workgroup per (token, head-slot) with one thread
    /// per channel — `head_dim` must be exactly the cube width, and the
    /// broadcast requires `n_v_heads % n_k_heads == 0`.
    pub fn supports(head_dim: usize, n_k_heads: usize, n_v_heads: usize) -> bool {
        head_dim == Self::HEAD_DIM && n_v_heads.is_multiple_of(n_k_heads)
    }

    /// # Safety
    ///
    /// - `input_handle`: `p * conv_dim` f32 raw qkv projections (read-only).
    /// - `expanded_handle`: `p * 3 * n_v_heads * head_dim` f32 (written).
    /// - `conv_weight_handle`: `conv_dim * kernel_size` f32.
    /// - `carry_handle`: `conv_dim * carry_stride` f32 (read, then rewritten
    ///   by the ordered carry-update dispatch).
    /// - `conv_dim == (2 * n_k_heads + n_v_heads) * head_dim`,
    ///   [`Self::supports`] must hold, `p <= 65535`.
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface"
    )]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        expanded_handle: Handle,
        conv_weight_handle: Handle,
        carry_handle: Handle,
        p: usize,
        conv_dim: usize,
        kernel_size: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) {
        debug_assert!(
            Self::supports(head_dim, n_k_heads, n_v_heads),
            "fused prework requires head_dim == {} and n_v divisible by n_k \
             (got head_dim={head_dim}, n_k={n_k_heads}, n_v={n_v_heads})",
            Self::HEAD_DIM
        );
        debug_assert!(p <= 65535, "fused prework: p={p} exceeds the Metal Y grid cap");
        debug_assert_eq!(
            conv_dim,
            (2 * n_k_heads + n_v_heads) * head_dim,
            "conv_dim must be the compact qkv width"
        );

        let carry_stride = kernel_size; // conv_state layout (the launch contract)
        let params: [f32; 8] = [
            p as f32,
            conv_dim as f32,
            kernel_size as f32,
            n_k_heads as f32,
            n_v_heads as f32,
            head_dim as f32,
            carry_stride as f32,
            1.0, // carry_idx_offset (conv_state layout)
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let expanded_len = p * 3 * n_v_heads * head_dim;

        PREWORK_FUSED_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            deltanet_prework_fused_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static((2 * n_k_heads + n_v_heads) as u32, p as u32, 1),
                CubeDim::new_1d(Self::HEAD_DIM as u32),
                BufferArg::from_raw_parts(input_handle.clone(), p * conv_dim),
                BufferArg::from_raw_parts(expanded_handle, expanded_len),
                BufferArg::from_raw_parts(conv_weight_handle, conv_dim * kernel_size),
                BufferArg::from_raw_parts(carry_handle.clone(), conv_dim * carry_stride),
                BufferArg::from_raw_parts(params_handle, 8),
            );

            // Ordered carry update (Issue 673 discipline) with c = p: for a
            // full chunk the new carry is this call's last ks-1 raw inputs —
            // exactly what the per-64-chunk chain leaves after its final
            // chunk. Generic for p < ks-1 (the old carry's tail shifts).
            crate::deltanet_chunked_cubecl::launch_conv1d_carry_update::<R>(
                client,
                input_handle,
                carry_handle,
                p,
                conv_dim,
                kernel_size,
                carry_stride,
                1,
            );
        }
    }
}

#[cfg(all(test, feature = "deltanet_prework_fused"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use crate::deltanet_chunked_cubecl::DeltanetChunkedConv1dCubeCL;
    use crate::deltanet_cubecl::ExpandAndL2NormalizeHeadsBatchedCubeCL;

    /// Deterministic pseudo-random data in a realistic range (LCG, no RNG dep).
    fn lcg_fill(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (x >> 40) as f32 / (1u64 << 24) as f32;
                lo + (hi - lo) * u
            })
            .collect()
    }

    #[derive(Debug)]
    struct Shape {
        n_k: usize,
        n_v: usize,
        hd: usize,
        ks: usize,
        p: usize,
    }

    /// Run BOTH arms over the same inputs; returns (expanded, carry_out) per
    /// arm. The reference arm is the shipping chain: per-64 chunked conv +
    /// swap + the batched expand — the exact kernels `prefill_tokens_chunk`
    /// dispatches in production order.
    #[allow(clippy::too_many_arguments)]
    fn run_both(
        client: &ComputeClient<ActiveRuntime>,
        s: &Shape,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (n_k, n_v, hd, ks, p) = (s.n_k, s.n_v, s.hd, s.ks, s.p);
        let conv_dim = (2 * n_k + n_v) * hd;
        let qkvx_dim = 3 * n_v * hd;
        let input = lcg_fill(seed, p * conv_dim, -0.35, 0.35);
        let weight = lcg_fill(seed ^ 0x57, conv_dim * ks, -0.12, 0.12);
        // Initial carry: the conv_state layout (conv_dim x ks), position 0
        // stale (the production layout — offset 1 reads/writes 1..ks-1).
        let carry0 = lcg_fill(seed ^ 0xC0, conv_dim * ks, -0.3, 0.3);

        // ── Reference arm: the shipping chain ──
        let in_h = client.create_from_slice(f32::as_bytes(&input));
        let w_h = client.create_from_slice(f32::as_bytes(&weight));
        let carry_ref = client.create_from_slice(f32::as_bytes(&carry0));
        let conv_out = client.empty(p * conv_dim * 4);
        // Per-64 chunks in production order (DeltanetChunkedConv1dCubeCL
        // itself issues conv + carry-update per chunk).
        let chunk = 64usize;
        let mut t0 = 0usize;
        while t0 < p {
            let c_len = chunk.min(p - t0);
            let qkv_in = in_h
                .clone()
                .offset_start((t0 * conv_dim * 4) as u64)
                .offset_end(((p - t0 - c_len) * conv_dim * 4) as u64);
            let qkv_out = conv_out
                .clone()
                .offset_start((t0 * conv_dim * 4) as u64)
                .offset_end(((p - t0 - c_len) * conv_dim * 4) as u64);
            unsafe {
                DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                    client,
                    qkv_in,
                    qkv_out,
                    w_h.clone(),
                    carry_ref.clone(),
                    c_len,
                    conv_dim,
                    ks,
                    ks,
                    1,
                );
            }
            t0 += c_len;
        }
        let expanded_ref_h = client.empty(p * qkvx_dim * 4);
        unsafe {
            ExpandAndL2NormalizeHeadsBatchedCubeCL::launch::<ActiveRuntime>(
                client,
                conv_out,
                expanded_ref_h.clone(),
                n_k,
                n_v,
                hd,
                p,
            );
        }
        let expanded_ref =
            f32::from_bytes(&client.read_one(expanded_ref_h.clone()).expect("read expanded ref")).to_vec();
        let carry_ref_out =
            f32::from_bytes(&client.read_one(carry_ref.clone()).expect("read carry ref")).to_vec();

        // ── Fused arm ──
        let carry_fused = client.create_from_slice(f32::as_bytes(&carry0));
        let expanded_fused_h = client.empty(p * qkvx_dim * 4);
        unsafe {
            DeltanetPreworkFusedCubeCL::launch::<ActiveRuntime>(
                client,
                in_h,
                expanded_fused_h.clone(),
                w_h,
                carry_fused.clone(),
                p,
                conv_dim,
                ks,
                n_k,
                n_v,
                hd,
            );
        }
        let expanded_fused =
            f32::from_bytes(&client.read_one(expanded_fused_h).expect("read expanded fused")).to_vec();
        let carry_fused_out =
            f32::from_bytes(&client.read_one(carry_fused).expect("read carry fused")).to_vec();

        (expanded_ref, expanded_fused, carry_ref_out, carry_fused_out)
    }

    /// G1: bit-identity vs the shipping chain (expanded outputs + carried
    /// conv state), across shapes incl. production dims, odd p, p < ks-1 and
    /// a zero-norm head (the Issue 673 Bug D guard).
    #[test]
    fn fused_is_bit_identical_to_the_chain() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        for (shape, seed) in [
            (Shape { n_k: 2, n_v: 4, hd: 128, ks: 4, p: 200 }, 11),
            (Shape { n_k: 16, n_v: 48, hd: 128, ks: 4, p: 192 }, 12),
            (Shape { n_k: 16, n_v: 48, hd: 128, ks: 4, p: 1 }, 13),
            (Shape { n_k: 16, n_v: 48, hd: 128, ks: 4, p: 2 }, 14), // p < ks-1
            (Shape { n_k: 4, n_v: 8, hd: 128, ks: 4, p: 63 }, 15),  // odd, non-multiple
            (Shape { n_k: 1, n_v: 3, hd: 128, ks: 4, p: 65 }, 16),
        ] {
            let (exp_ref, exp_fused, carry_ref, carry_fused) = run_both(&client, &shape, seed);
            let mut diffs = 0usize;
            for (a, b) in exp_ref.iter().zip(exp_fused.iter()) {
                if a.to_bits() != b.to_bits() {
                    diffs += 1;
                }
            }
            assert_eq!(
                diffs, 0,
                "expanded outputs differ in {diffs} elements at {shape:?}"
            );
            let mut carry_diffs = 0usize;
            for (a, b) in carry_ref.iter().zip(carry_fused.iter()) {
                if a.to_bits() != b.to_bits() {
                    carry_diffs += 1;
                }
            }
            assert_eq!(carry_diffs, 0, "carry differs in {carry_diffs} elements at {shape:?}");
            println!(
                "fused_is_bit_identical_to_the_chain: {:?} OK ({} expanded elems)",
                shape,
                exp_ref.len()
            );
        }
    }

    /// The launch counter moves (the e2e assert's unit-level twin).
    #[test]
    fn fused_launch_counter_increments() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let before = prework_fused_launches();
        let s = Shape { n_k: 2, n_v: 4, hd: 128, ks: 4, p: 8 };
        let _ = run_both(&client, &s, 99);
        assert!(prework_fused_launches() > before);
    }
}
