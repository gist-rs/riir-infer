//! Issue 726 T3 Phase B — the ANE block executor (full-width dispatch).
//!
//! Per eligible projection: read back the normed input (token-major
//! `[p × n]` f32), then per 2048-token block: pack to the ANE's
//! channel-major fp16 `[n × W]` layout, eval, unpack each output segment
//! from channel-major `[seg_dim × W]` fp16 to token-major `[W × seg_dim]`
//! f32, and write it into the target handle at the block's token offset
//! via `client.write` on an offset slice (the sanctioned Graph-input
//! refresh path — the write lands at the slice offset).
//!
//! Phase B scope: **full coverage only** (`f == 1.0 && tail == 0`) — the
//! ANE replaces the GPU projection for exact 2048-multiple prompts; every
//! other shape fail-opens to the GPU batched GEMM. The suffix-channel
//! split + tail complement are Phase C.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::bridge::BridgeKernel;

// ── Issue 726 diagnosis (S1): per-stage wall-clock accumulators ─────────────
// Decomposes the 7-8× in-pipeline vs P10-microbench gap. The seam is
// single-threaded serial by design (P12), so Relaxed atomics suffice;
// Instant pairs number in the hundreds per block — negligible vs ms-scale
// stages. Always-on under the feature (no flag to forget); reset/dump from
// the A/B harness brackets each arm.
pub static T_READ_NS: AtomicU64 = AtomicU64::new(0); // read_one: map+sync GPU→host
pub static T_READCOPY_NS: AtomicU64 = AtomicU64::new(0); // f32::from_bytes().to_vec()
pub static T_PACK_NS: AtomicU64 = AtomicU64::new(0);
pub static T_EVAL_NS: AtomicU64 = AtomicU64::new(0); // ANE bridge eval only
pub static T_UNPACK_NS: AtomicU64 = AtomicU64::new(0);
pub static T_WALLOC_NS: AtomicU64 = AtomicU64::new(0); // Bytes staging alloc+copy
pub static T_WSUBMIT_NS: AtomicU64 = AtomicU64::new(0); // client.write submit
pub static N_EVALS: AtomicU64 = AtomicU64::new(0);
pub static N_WRITES: AtomicU64 = AtomicU64::new(0);
pub static N_READS: AtomicU64 = AtomicU64::new(0);

fn bump(c: &AtomicU64, t: Instant) {
    c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Zero all stage timers + counters (call at arm start).
pub fn ane_stage_reset() {
    for c in [
        &T_READ_NS,
        &T_READCOPY_NS,
        &T_PACK_NS,
        &T_EVAL_NS,
        &T_UNPACK_NS,
        &T_WALLOC_NS,
        &T_WSUBMIT_NS,
        &N_EVALS,
        &N_WRITES,
        &N_READS,
        &N_EVAL_FAILS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    // Issue 769 T10 P5: the retry counters live in `bridge` (that is where the
    // retry is) but belong to the same per-arm epoch as everything above.
    super::bridge::eval_retry_reset();
}

/// One-line stage summary (call at arm end). Empty activity reports as the
/// GPU-arm control string.
pub fn ane_stage_dump() -> String {
    let ld = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let (r, rc, pk, ev, up, wa, ws) = (
        ld(&T_READ_NS),
        ld(&T_READCOPY_NS),
        ld(&T_PACK_NS),
        ld(&T_EVAL_NS),
        ld(&T_UNPACK_NS),
        ld(&T_WALLOC_NS),
        ld(&T_WSUBMIT_NS),
    );
    let (ne, nw, nr) = (ld(&N_EVALS), ld(&N_WRITES), ld(&N_READS));
    let total = r + rc + pk + ev + up + wa + ws;
    if total == 0 && ne == 0 {
        return "[ane-stage] no ANE stage activity (fail-open / GPU arm)".into();
    }
    let ms = |ns: u64| ns as f64 / 1e6;
    let per = |ns: u64, n: u64| if n > 0 { format!("~{:.1}", ms(ns) / n as f64) } else { "—".into() };
    // Issue 769 T10 P5: `N_EVAL_FAILS` was reset here and incremented by both
    // exec paths, and printed by NOTHING — so a genuine fail-open (an ANE block
    // silently served by GPU, which changes what the arm measures) was invisible
    // in the line every bench quotes. Retries are the same hazard one step
    // earlier: they succeed, so `fails` stays 0 while the `eval` share doubles
    // for those ops. Both are printed unconditionally, including the zeros — a
    // health field that only appears when unhealthy cannot be read as green.
    let (nf, nrt, nat) = (
        ld(&N_EVAL_FAILS),
        super::bridge::N_EVAL_RETRIES.load(Ordering::Relaxed),
        super::bridge::N_EVAL_ATTEMPTS.load(Ordering::Relaxed),
    );
    let retry_pct = if nat > 0 { nrt as f64 / nat as f64 * 100.0 } else { 0.0 };
    format!(
        "[ane-stage] total {:.1} ms — read(map+sync) {:.1} ({}× {}) · copy {:.1} · pack {:.1} · eval {:.1} ({}× {}, {:.0}%) · unpack {:.1} · walloc {:.1} · wsubmit {:.1} ({}× {}) · health[fail-open {} · eval retries {}/{} attempts ({:.1}%)]",
        ms(total),
        ms(r), nr, per(r, nr),
        ms(rc), ms(pk),
        ms(ev), ne, per(ev, ne), if total > 0 { ev as f64 / total as f64 * 100.0 } else { 0.0 },
        ms(up), ms(wa),
        ms(ws), nw, per(ws, nw),
        nf, nrt, nat, retry_pct,
    )
}

/// Pack one block of the token-major f32 input into the ANE's channel-major
/// fp16 layout: `x16[c * W + t] = fp16(normx[(block * W + t) * n + c])`.
pub fn pack_input_block(normx: &[f32], n: usize, block: usize, w: usize, x16: &mut [u16]) {
    debug_assert_eq!(x16.len(), n * w);
    let base = block * w;
    for t in 0..w {
        let row = &normx[(base + t) * n..(base + t + 1) * n];
        for (c, &v) in row.iter().enumerate() {
            x16[c * w + t] = half::f16::from_f32(v).to_bits();
        }
    }
}

/// Tile edge for the pack/unpack transpose loops (Bench 775 fix ladder #3:
/// the scalar loops run one strided store per element — 16× write
/// amplification at W=2048; the tiled form amortizes both sides to ~128B
/// runs). Bit-identical to the scalar form by construction — the same
/// per-element `f16::from_f32` conversion, only the loop order changes.
const TRANSPOSE_TILE: usize = 32;

/// Tiled pack (Plan 549 T1): reads `TC×4B` runs of consecutive channels per
/// token row, writes `TT×2B` linear runs per channel — both sides of the
/// transpose stay cache-linear. Same output bits as [`pack_input_block`].
pub fn pack_input_block_tiled(normx: &[f32], n: usize, block: usize, w: usize, x16: &mut [u16]) {
    debug_assert_eq!(x16.len(), n * w);
    let base = block * w;
    let tc = TRANSPOSE_TILE.min(n);
    let tt = (TRANSPOSE_TILE * 2).min(w);
    let mut c0 = 0;
    while c0 < n {
        let c_hi = (c0 + tc).min(n);
        let mut t0 = 0;
        while t0 < w {
            let t_hi = (t0 + tt).min(w);
            for t in t0..t_hi {
                let row = &normx[(base + t) * n..(base + t + 1) * n];
                for c in c0..c_hi {
                    x16[c * w + t] = half::f16::from_f32(row[c]).to_bits();
                }
            }
            t0 = t_hi;
        }
        c0 = c_hi;
    }
}

/// Unpack one output segment from channel-major fp16 `[seg_dim × W]` to
/// token-major f32 `[W × seg_dim]`: `out[t * seg_dim + c] =
/// f32(y16[(chan_off + c) * W + t])`.
pub fn unpack_segment_block(
    y16: &[u16],
    chan_off: usize,
    seg_dim: usize,
    w: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), w * seg_dim);
    for t in 0..w {
        for c in 0..seg_dim {
            out[t * seg_dim + c] = half::f16::from_bits(y16[(chan_off + c) * w + t]).to_f32();
        }
    }
}

/// Tiled unpack (Plan 549 T1) — the mirror of [`pack_input_block_tiled`].
/// Same output bits as [`unpack_segment_block`].
pub fn unpack_segment_block_tiled(
    y16: &[u16],
    chan_off: usize,
    seg_dim: usize,
    w: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), w * seg_dim);
    let tc = TRANSPOSE_TILE.min(seg_dim);
    let tt = (TRANSPOSE_TILE * 2).min(w);
    let mut c0 = 0;
    while c0 < seg_dim {
        let c_hi = (c0 + tc).min(seg_dim);
        let mut t0 = 0;
        while t0 < w {
            let t_hi = (t0 + tt).min(w);
            for t in t0..t_hi {
                let row = &mut out[t * seg_dim..(t + 1) * seg_dim];
                for c in c0..c_hi {
                    row[c] = half::f16::from_bits(y16[(chan_off + c) * w + t]).to_f32();
                }
            }
            t0 = t_hi;
        }
        c0 = c_hi;
    }
}

/// One full-width ANE projection over all blocks of a prompt.
///
/// `segments`: (target handle, segment channel offset in the fused output,
/// segment width). Writes each block's segment rows at token offset
/// `block * W` inside the target (token-major `[p × seg_dim]` handles).
///
/// Serial per-block submission (P12: no dual-projection overlap at real
/// dims; omlx #5). The input is read back ONCE per projection; blocks then
/// pack → eval → unpack → write in sequence. Staging buffers are allocated
/// per call (the prefill path's per-call scratch pattern — the decode
/// steady state stays alloc-free).
pub fn dispatch_full_width<R: Runtime>(
    client: &ComputeClient<R>,
    kernel: &BridgeKernel,
    input: &Handle,
    p: usize,
    n: usize,
    oc_total: usize,
    w: usize,
    segments: &[(Handle, usize, usize)],
) -> Result<(), String> {
    debug_assert!(
        p.is_multiple_of(w),
        "full-width dispatch requires exact blocks"
    );
    let blocks = p / w;

    // 1. Read the whole token-major input once.
    let normx = read_input_token_major(client, input)?;
    debug_assert_eq!(normx.len(), p * n);

    // 2. Per-block pack → eval → per-segment unpack + offset write.
    let mut x16 = vec![0u16; n * w];
    let mut y16 = vec![0u16; oc_total * w];
    let mut seg_staging: Vec<Vec<f32>> = segments
        .iter()
        .map(|&(_, _, seg_dim)| vec![0f32; w * seg_dim])
        .collect();
    for block in 0..blocks {
        let t = Instant::now();
        pack_input_block(&normx, n, block, w, &mut x16);
        bump(&T_PACK_NS, t);
        let t = Instant::now();
        kernel.eval(&x16, &mut y16)?;
        bump(&T_EVAL_NS, t);
        N_EVALS.fetch_add(1, Ordering::Relaxed);
        for (si, &(_, chan_off, seg_dim)) in segments.iter().enumerate() {
            let t = Instant::now();
            unpack_segment_block(&y16, chan_off, seg_dim, w, &mut seg_staging[si]);
            bump(&T_UNPACK_NS, t);
            // Write at the block's token offset: element offset
            // block * w * seg_dim (f32 → ×4 bytes).
            let t = Instant::now();
            let bytes = cubecl::bytes::Bytes::from_bytes_vec(
                seg_staging[si]
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect(),
            );
            bump(&T_WALLOC_NS, t);
            let target = segments[si]
                .0
                .clone()
                .offset_start((block * w * seg_dim * 4) as u64);
            let t = Instant::now();
            client.write(&target, bytes);
            bump(&T_WSUBMIT_NS, t);
            N_WRITES.fetch_add(1, Ordering::Relaxed);
        }
    }
    super::note_ane_dispatch();
    Ok(())
}

// ── Plan 549 T2: the split-overlap job API ─────────────────────────────────
//
// The omlx deployed shape: the ANE computes one segment of a fused projection
// while the GPU computes the complement CONCURRENTLY. Structure:
//
//   begin_split_overlapped  (main thread) — ONE dependency-bound `read_one`
//                            of the input (its poll drains only the producing
//                            kernels — the caller has NOT yet submitted the
//                            complement it wants to overlap with), then
//                            spawns a worker that packs → evals → unpacks
//                            every block into host staging.
//   [caller submits the GPU complement GEMMs — concurrent with the worker]
//   AneSplitJob::finish     (main thread) — joins the worker, then writes
//                            each segment/block into the target handles via
//                            `client.write`.
//
// Thread-safety by construction: the worker performs ZERO client calls (pure
// host + ANE bridge — `BridgeKernel` is Send+Sync, verified in P4), so no
// cross-thread cubecl use exists. Ordering: the consumer kernels submitted
// after `finish` returns see both the complement GEMMs and the segment
// writes — same-queue in-order execution.
//
// Fail-open is CLEAN: no writes happen before a successful join of ALL
// blocks, so an eval error anywhere leaves every output handle untouched —
// the seam may fail-open to the full GPU path with zero partial-state risk
// (an improvement over the Phase B mid-prompt panic policy).

/// Count of eval failures that fail-opened a split job (Issue 726
/// observability — the status=0x9 retry-recovered class should trend visible
/// here without killing dispatches).
pub static N_EVAL_FAILS: AtomicU64 = AtomicU64::new(0);

/// An in-flight split-overlap dispatch (see the module section above). Created
/// by [`begin_split_overlapped`]; complete it with [`Self::finish`].
pub struct AneSplitJob {
    handle: Option<std::thread::JoinHandle<Result<Vec<Vec<f32>>, String>>>,
    /// `(target handle, program-relative channel offset, segment width)` —
    /// program-RELATIVE offsets (the fused program's rows [0, oc_total) may
    /// be partially consumed; segments reference rows within it).
    segments: Vec<(Handle, usize, usize)>,
    w: usize,
    blocks: usize,
}

impl AneSplitJob {
    /// Assemble from an already-spawned worker (the Plan 550 GPU-IN hybrid —
    /// same staging contract, different producer). NOT part of the public
    /// dispatch flow; `begin_split_overlapped` remains the primary ctor.
    #[doc(hidden)]
    pub fn from_parts(
        handle: std::thread::JoinHandle<Result<Vec<Vec<f32>>, String>>,
        segments: Vec<(Handle, usize, usize)>,
        w: usize,
        blocks: usize,
    ) -> Self {
        Self {
            handle: Some(handle),
            segments,
            w,
            blocks,
        }
    }
}

/// Begin a split-overlap dispatch. Returns immediately after the input
/// readback + worker spawn; the caller should submit the GPU complement
/// NOW and then call [`AneSplitJob::finish`].
#[allow(clippy::too_many_arguments)]
pub fn begin_split_overlapped<R: Runtime>(
    client: &ComputeClient<R>,
    kernel: std::sync::Arc<BridgeKernel>,
    input: &Handle,
    p: usize,
    n: usize,
    oc_total: usize,
    w: usize,
    segments: &[(Handle, usize, usize)],
) -> Result<AneSplitJob, String> {
    debug_assert!(p.is_multiple_of(w), "split dispatch requires exact blocks");
    let blocks = p / w;
    // 1. The ONE dependency-bound read (before any complement submission —
    //    the ordering that keeps this read cheap; see the section doc).
    let normx = read_input_token_major(client, input)?;
    debug_assert_eq!(normx.len(), p * n);
    // 2. Worker: pure host + ANE — pack → eval → unpack per block. Only the
    //    segment METADATA crosses the thread boundary (Handles are not Send;
    //    they stay on the main thread in the job).
    let seg_meta: Vec<(usize, usize)> = segments.iter().map(|&(_, o, d)| (o, d)).collect();
    let seg_dims: Vec<usize> = seg_meta.iter().map(|&(_, d)| d).collect();
    let worker = std::thread::Builder::new()
        .name("ane-split".into())
        .spawn(move || {
            let mut x16 = vec![0u16; n * w];
            let mut y16 = vec![0u16; oc_total * w];
            // Per-segment staging across ALL blocks — `finish` writes each
            // segment as one block-strided sequence of offset writes.
            let mut staging: Vec<Vec<f32>> = seg_dims
                .iter()
                .map(|&d| vec![0f32; blocks * w * d])
                .collect();
            for block in 0..blocks {
                let t = Instant::now();
                pack_input_block_tiled(&normx, n, block, w, &mut x16);
                bump(&T_PACK_NS, t);
                let t = Instant::now();
                if let Err(e) = kernel.eval(&x16, &mut y16) {
                    bump(&T_EVAL_NS, t);
                    N_EVAL_FAILS.fetch_add(1, Ordering::Relaxed);
                    return Err(format!("ANE split eval failed (block {block}): {e}"));
                }
                bump(&T_EVAL_NS, t);
                N_EVALS.fetch_add(1, Ordering::Relaxed);
                for (si, &(chan_off, seg_dim)) in seg_meta.iter().enumerate() {
                    let t = Instant::now();
                    // Unpack straight into the segment's all-blocks staging at
                    // this block's token offset (block-strided layout).
                    let out = &mut staging[si][block * w * seg_dim..(block + 1) * w * seg_dim];
                    unpack_segment_block_tiled(&y16, chan_off, seg_dim, w, out);
                    bump(&T_UNPACK_NS, t);
                }
            }
            Ok(staging)
        })
        .map_err(|e| format!("ANE split worker spawn failed: {e}"))?;
    Ok(AneSplitJob {
        handle: Some(worker),
        segments: segments.to_vec(),
        w,
        blocks,
    })
}

impl AneSplitJob {
    /// Join the worker + write every segment/block into its target handle.
    /// On any error NO writes have been performed — the caller may fail-open
    /// to the full GPU path cleanly.
    pub fn finish<R: Runtime>(mut self, client: &ComputeClient<R>) -> Result<(), String> {
        let worker = self
            .handle
            .take()
            .expect("finish called twice on AneSplitJob");
        let staging = worker
            .join()
            .map_err(|_| "ANE split worker panicked".to_string())??;
        for (si, (target, _, seg_dim)) in self.segments.iter().enumerate() {
            for block in 0..self.blocks {
                let t = Instant::now();
                let src = &staging[si][block * self.w * seg_dim..(block + 1) * self.w * seg_dim];
                let bytes = cubecl::bytes::Bytes::from_bytes_vec(
                    src.iter().flat_map(|&v| v.to_le_bytes()).collect(),
                );
                bump(&T_WALLOC_NS, t);
                let dst = target
                    .clone()
                    .offset_start((block * self.w * seg_dim * 4) as u64);
                let t = Instant::now();
                client.write(&dst, bytes);
                bump(&T_WSUBMIT_NS, t);
                N_WRITES.fetch_add(1, Ordering::Relaxed);
            }
        }
        super::note_ane_dispatch();
        Ok(())
    }
}

impl Drop for AneSplitJob {
    fn drop(&mut self) {
        // A job dropped without `finish` (a seam bug) would detach the worker
        // writing into dropped staging — join defensively so the failure is
        // loud in debug + leak-free in release.
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
            debug_assert!(false, "AneSplitJob dropped without finish()");
        }
    }
}

/// Read the token-major f32 input back from a handle.
pub fn read_input_token_major<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
) -> Result<Vec<f32>, String> {
    let t = Instant::now();
    let bytes = client
        .read_one(input.clone())
        .map_err(|e| format!("ANE input readback failed: {e}"))?;
    bump(&T_READ_NS, t);
    N_READS.fetch_add(1, Ordering::Relaxed);
    let t = Instant::now();
    let v = f32::from_bytes(&bytes).to_vec();
    bump(&T_READCOPY_NS, t);
    Ok(v)
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    feature = "metal_tensor_gemm"
))]
mod tests {
    use super::*;

    /// Issue 769 T10 P5 regression guard.
    ///
    /// The defect this replaces was not a wrong number — it was a field that
    /// existed, was reset, was incremented, and was printed by NOTHING, so a
    /// fail-open (an ANE block silently served by GPU, which changes what the
    /// arm measures) never reached the `[ane-stage]` line every bench quotes.
    /// A format-string edit can silently reintroduce exactly that, so the
    /// PRESENCE of both health fields is pinned, in BOTH dumps, and pinned at
    /// the ZERO value — a field that shows up only when unhealthy is
    /// indistinguishable from a field that is gone.
    #[test]
    fn stage_dumps_carry_the_health_fields_even_when_zero() {
        ane_stage_reset();
        super::super::exec_zc::zc_stage_reset();

        // The non-ZC dump short-circuits to the GPU-arm control string when the
        // arm did nothing, so drive one counter to get the real format path.
        N_EVALS.store(1, Ordering::Relaxed);
        T_EVAL_NS.store(1_000_000, Ordering::Relaxed);
        let d = ane_stage_dump();
        assert!(d.contains("health[fail-open 0"), "ane-stage lost fail-open: {d}");
        assert!(d.contains("eval retries 0/0"), "ane-stage lost retries: {d}");

        let z = super::super::exec_zc::zc_stage_dump();
        assert!(z.contains("fails 0 retries 0/0"), "zc dump lost retries: {z}");

        // And the counters must be readable as a nonzero rate, or the field is
        // decoration. 1 retry of 2 attempts is 50.0%.
        super::super::bridge::N_EVAL_RETRIES.store(1, Ordering::Relaxed);
        super::super::bridge::N_EVAL_ATTEMPTS.store(2, Ordering::Relaxed);
        let z = super::super::exec_zc::zc_stage_dump();
        assert!(z.contains("retries 1/2 (50.0%)"), "zc retry rate wrong: {z}");
        let d = ane_stage_dump();
        assert!(d.contains("eval retries 1/2 attempts (50.0%)"), "ane retry rate wrong: {d}");

        // Leave the process's statics clean for the other tests in this binary.
        ane_stage_reset();
        super::super::exec_zc::zc_stage_reset();
    }

    #[test]
    fn tiled_pack_bit_identical_to_scalar() {
        // Plan 549 T1: the tiled transpose must produce the SAME BITS as the
        // scalar reference (same per-element conversion, loop order only).
        // Non-tile-multiple edges on both axes (n=70, w=100).
        let (n, w, blocks) = (70usize, 100usize, 3usize);
        let normx: Vec<f32> = (0..blocks * w * n)
            .map(|i| ((i as f32 * 0.37) % 21.0) - 10.5)
            .collect();
        let mut scalar = vec![0u16; n * w];
        let mut tiled = vec![0u16; n * w];
        for b in 0..blocks {
            pack_input_block(&normx, n, b, w, &mut scalar);
            pack_input_block_tiled(&normx, n, b, w, &mut tiled);
        }
        assert_eq!(scalar, tiled, "tiled pack bits must match the scalar form");
    }

    #[test]
    fn tiled_unpack_bit_identical_to_scalar() {
        let (oc, w) = (77usize, 98usize);
        let y16: Vec<u16> = (0..oc * w)
            .map(|i| half::f16::from_f32(((i as f32 * 0.11) % 17.0) - 8.0).to_bits())
            .collect();
        for (off, dim) in [(0usize, oc), (5usize, 40), (13usize, 1)] {
            let mut scalar = vec![0f32; w * dim];
            let mut tiled = vec![0f32; w * dim];
            unpack_segment_block(&y16, off, dim, w, &mut scalar);
            unpack_segment_block_tiled(&y16, off, dim, w, &mut tiled);
            assert_eq!(scalar, tiled, "tiled unpack bits must match (off={off})");
        }
    }

    #[test]
    fn pack_unpack_round_trip() {
        // Small-but-representative: n=6 channels, W=8 tokens, 1 block.
        let (n, w) = (6usize, 8usize);
        let normx: Vec<f32> = (0..n * w).map(|i| ((i as f32) * 0.25) - 3.0).collect();
        let mut x16 = vec![0u16; n * w];
        pack_input_block(&normx, n, 0, w, &mut x16);
        // Unpack the input back (same channel-major decode as a segment of
        // width n at offset 0) + compare with fp16 tolerance.
        let mut out = vec![0f32; w * n];
        unpack_segment_block(&x16, 0, n, w, &mut out);
        for t in 0..w {
            for c in 0..n {
                let orig = normx[t * n + c];
                let rt = out[t * n + c];
                assert!(
                    (orig - rt).abs() <= orig.abs() * 1e-3 + 1e-6,
                    "({t},{c}): {orig} vs {rt}"
                );
            }
        }
    }

    #[test]
    fn pack_block_offset_lands_in_second_window() {
        // 2 blocks: packing block 1 must read rows [W..2W).
        let (n, w) = (4usize, 4usize);
        let normx: Vec<f32> = (0..n * 2 * w).map(|i| i as f32).collect();
        let mut x16 = vec![0u16; n * w];
        pack_input_block(&normx, n, 1, w, &mut x16);
        // Channel 0 of block 1 = normx rows at token indices 4..8, col 0.
        for t in 0..w {
            let expect = normx[(w + t) * n];
            assert_eq!(half::f16::from_bits(x16[t]).to_f32(), expect, "t={t}");
        }
    }

    #[test]
    fn segment_channel_offset_selects_rows() {
        // y16 channel-major [oc=5 × W=3]; segment at offset 2 width 2.
        let (oc, w) = (5usize, 3usize);
        let y16: Vec<u16> = (0..oc * w)
            .map(|i| half::f16::from_f32(i as f32).to_bits())
            .collect();
        let mut out = vec![0f32; w * 2];
        unpack_segment_block(&y16, 2, 2, w, &mut out);
        // out[t*2 + c] = y16[(2+c)*w + t]
        for t in 0..w {
            for c in 0..2 {
                assert_eq!(out[t * 2 + c], ((2 + c) * w + t) as f32);
            }
        }
    }
}
