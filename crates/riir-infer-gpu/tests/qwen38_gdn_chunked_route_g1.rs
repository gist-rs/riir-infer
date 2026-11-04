//! Issue 1150 follow-up 2 — the chunked-GDN ROUTE-level G1: my repack +
//! window pipeline vs the incumbent multi-token kernel on the SAME random
//! inputs (GPU-vs-GPU on ActiveRuntime/wgpu).
//!
//! The chunkwise family's own in-module tests prove the kernels against a
//! CPU sequential reference at 1e-4 (synthetic). This test closes the gap
//! the family tests cannot see: the ROUTE's wiring — the token-major →
//! head-major repack layout, the per-window state carry, the scatter back
//! into token-major `rec_b` — against the SHIPPING kernel the route must
//! replace. A layout bug (transposed repack slice, wrong head offset,
//! mis-windowed state) fails LOUD here; a pass at the family's re-association
//! band means the e2e walk divergence recorded in the bench file is the
//! summation-order class, not a wiring defect.
//!
//! #![cfg] green-zero rule: the file carries its required-features row in
//! Cargo.toml ([[test]] qwen38_gdn_chunked_route_g1).

#![cfg(all(
    feature = "qwen38_wgpu",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]

use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL;
use riir_infer_gpu::qwen38_gdn_chunked_route::{GDN_CHUNK_C, GdnChunkedScratch, ingest_window};

/// The family's own re-association tolerance (its tests use 1e-4).
const TOL: f32 = 1e-4;

/// Decay profiles the route must be judged on:
/// - `FloorSafe`: α ∈ [0.93, 1.0) — 64-token cumulative ≥ e⁻⁵, never near
///   the family's −60 log-space floor. Any failure here is a WIRING bug.
/// - `FloorCrossing`: α ~ U(0, 0.5) — cumulative ≈ e⁻⁸⁹, crosses the −60
///   floor mid-window; the family's both-floored-ratio-reads-1 rule then
///   corrupts the intra-chunk weights (the doc's "behaviorally exact"
///   claim does not cover this profile). Recorded as the floor-class arm.
#[derive(Clone, Copy, PartialEq, Debug)]
enum DecayProfile {
    FloorSafe,
    FloorCrossing,
}

fn decay_for(profile: DecayProfile, r: f32) -> f32 {
    match profile {
        // α ∈ [0.93, 1.0) — slow heads (a_log·softplus ≈ −0.07/token)
        DecayProfile::FloorSafe => 0.93 + r * 0.07,
        // α ∈ (0, 0.5) — fast heads crossing the −60 floor mid-window
        DecayProfile::FloorCrossing => r * 0.5,
    }
}

/// Minimal deterministic PRNG (xorshift) — same fixture discipline as the
/// family's in-module tests.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn read(client: &ComputeClient<ActiveRuntime>, h: &Handle, n: usize) -> Vec<f32> {
    client
        .read_one(h.clone())
        .map(|b| bytemuck::cast_slice(&b)[..n].to_vec())
        .unwrap_or_default()
}

fn run_route_parity(profile: DecayProfile, windows: usize) -> (f32, f32) {
    let client = CubeCLContext::new().expect("gpu context").client();
    let h = 8usize; // qwen3.8-27B n_v_heads
    let d = 128usize; // head_v_dim (square state — the route's gate)
    let c = GDN_CHUNK_C;
    let p = windows * c;
    let v_dim = h * d;

    let mut rng = Rng(0x1150_1150 + windows as u64);
    let mut qkvx = vec![0.0f32; p * 3 * v_dim]; // token-major [p, 3*v_dim]
    let mut beta = vec![0.0f32; p * h]; // token-major [p, h]
    let mut decay = vec![0.0f32; p * h]; // token-major [p, h]
    for x in qkvx.iter_mut() {
        *x = rng.next_f32();
    }
    // Realistic ranges: β ∈ (0, 1], α per the profile under test; q/k/v get
    // L2-normalized heads below (the real path's ExpandAndL2NormalizeHeads).
    for i in 0..p * h {
        beta[i] = rng.next_f32();
        decay[i] = decay_for(profile, rng.next_f32());
    }

    // L2-normalize the q/k head blocks per token (the real path does this
    // BEFORE the recurrence; the family's own tests normalize too).
    for t in 0..p {
        let row = t * 3 * v_dim;
        for block in [0usize, 1usize] {
            for hh in 0..h {
                let off = row + block * v_dim + hh * d;
                let norm: f32 = qkvx[off..off + d].iter().map(|x| x * x).sum::<f32>().sqrt();
                let inv = 1.0 / norm.max(1e-8);
                for x in qkvx[off..off + d].iter_mut() {
                    *x *= inv;
                }
            }
        }
    }

    // Shared entry state (nonzero — exercises the S₀ terms).
    let mut state = vec![0.0f32; h * d * d];
    for x in state.iter_mut() {
        *x = rng.next_f32() * 0.1;
    }

    // Arm A — the incumbent multi-token kernel over the full p tokens.
    let (qkvx_a, beta_a, decay_a, state_a, out_a) = (
        client.create_from_slice(bytemuck::cast_slice(&qkvx)),
        client.create_from_slice(bytemuck::cast_slice(&beta)),
        client.create_from_slice(bytemuck::cast_slice(&decay)),
        client.create_from_slice(bytemuck::cast_slice(&state)),
        client.create_from_slice(&vec![0u8; p * v_dim * 4]),
    );
    unsafe {
        DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
            &client,
            qkvx_a,
            beta_a,
            decay_a,
            state_a.clone(),
            out_a.clone(),
            h,
            d,
            p,
        );
    }

    // Arm B — the route: repack + the chunkwise window pipeline per window.
    let (qkvx_b, beta_b, decay_b, state_b, out_b) = (
        client.create_from_slice(bytemuck::cast_slice(&qkvx)),
        client.create_from_slice(bytemuck::cast_slice(&beta)),
        client.create_from_slice(bytemuck::cast_slice(&decay)),
        client.create_from_slice(bytemuck::cast_slice(&state)),
        client.create_from_slice(&vec![0u8; p * v_dim * 4]),
    );
    let scratch = GdnChunkedScratch::new(&client, h, d);
    for w in 0..windows {
        unsafe {
            ingest_window(
                &client,
                &scratch,
                qkvx_b.clone().offset_start((w * c * 3 * v_dim * 4) as u64),
                beta_b.clone().offset_start((w * c * h * 4) as u64),
                decay_b.clone().offset_start((w * c * h * 4) as u64),
                state_b.clone(),
                out_b.clone().offset_start((w * c * v_dim * 4) as u64),
                h,
                d,
            );
        }
    }

    // Compare outputs + final states.
    let out_a = read(&client, &out_a, p * v_dim);
    let out_b = read(&client, &out_b, p * v_dim);
    let state_a = read(&client, &state_a, h * d * d);
    let state_b = read(&client, &state_b, h * d * d);

    let mut max_out = 0.0f32;
    for (a, b) in out_a.iter().zip(&out_b) {
        let denom = a.abs().max(1e-3);
        max_out = max_out.max(((a - b) / denom).abs());
    }
    let mut max_state = 0.0f32;
    for (a, b) in state_a.iter().zip(&state_b) {
        let denom = a.abs().max(1e-3);
        max_state = max_state.max(((a - b) / denom).abs());
    }
    println!(
        "route G1 [{profile:?}, {windows} windows]: max_out_rel {max_out:.2e} \
         max_state_rel {max_state:.2e} (TOL {TOL})"
    );
    (max_out, max_state)
}

#[test]
fn route_matches_multi_token_kernel_floor_safe() {
    // WIRING gate: floor-safe decay must pass at the family's band. A
    // failure here is the route's bug, never the numerics class.
    let (max_out, max_state) = run_route_parity(DecayProfile::FloorSafe, 1);
    assert!(
        max_out < TOL,
        "floor-safe output rel err {max_out} >= {TOL} — wiring bug"
    );
    assert!(
        max_state < TOL,
        "floor-safe state rel err {max_state} >= {TOL} — wiring bug"
    );
}

#[test]
fn route_two_windows_carry_matches_multi_token_floor_safe() {
    // The multi-window state-carry seam (the family tests only ever run ONE
    // window): the per-window in-place state_update feeding the next window.
    let (max_out, max_state) = run_route_parity(DecayProfile::FloorSafe, 2);
    assert!(
        max_out < TOL,
        "two-window output rel err {max_out} >= {TOL}"
    );
    assert!(
        max_state < TOL,
        "two-window state rel err {max_state} >= {TOL}"
    );
}

#[test]
fn route_floor_crossing_matches_multi_token() {
    // The floor-crossing profile (α ~ U(0, 0.5): cumulative ≈ e⁻⁸⁹, far
    // past the old −60 cumulative floor) must now match at the family band —
    // the per-step α clamp keeps the cumulative exact (the issue-1150 route
    // G1 measured 7.2e+1 here against the OLD floor; that class is dead).
    let (max_out, max_state) = run_route_parity(DecayProfile::FloorCrossing, 1);
    assert!(
        max_out < TOL,
        "floor-crossing output rel err {max_out} >= {TOL} — the ratio corruption is back"
    );
    assert!(
        max_state < TOL,
        "floor-crossing state rel err {max_state} >= {TOL}"
    );
}
