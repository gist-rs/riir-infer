//! Issue 1172 — the GDN-bucket census instrument's own gate.
//!
//! The instrument measures GPU wall with Metal counter-sample pass
//! attachments and hazard-pinned marker passes. ONE serial test (the
//! phases share the one queue — parallel test threads would interleave
//! markers and skew each other's walls), proving the measurement on three
//! axes:
//!
//! 1. **Sleep wall** — a CPU sleep between two markers must appear in the
//!    section wall (the NS_PER_TICK period law validated empirically,
//!    never assumed). Empty hazards are sound here: real gaps serialize an
//!    idle queue.
//! 2. **Production shape** — the core gate: markers hazard-pinned on the
//!    written buffer around a block of real rmsnorm launches must read the
//!    same wall as a CPU-sync reference of the SAME launches (the
//!    pipeline warmed first — a cold compile would inflate the reference
//!    ~6 ms). This pins the measured law that UNPINNED markers slide ahead
//!    of queued work while pinned ones bracket it honestly, and that the
//!    census wall scales linearly with the work.
//! 3. **Grouping** — a full 15-mark chunk group + the inter-chunk skip:
//!    the "end"-mark → next-"input_norm"-mark boundary is never a census
//!    section (a sleep across that boundary must be carried in
//!    `inter_chunk_ms`, never folded into a section).

#![cfg(all(
    feature = "qwen38_wgpu",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]

use std::sync::Arc;
use std::time::{Duration, Instant};

use riir_infer_gpu::qwen38_gdn_census::{GdnCensus, MARKS_PER_CHUNK};

type Client = cubecl::client::ComputeClient<riir_infer_gpu::cubecl_runtime::ActiveRuntime>;

fn census(layer: usize) -> (GdnCensus, Client) {
    let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new().expect("cubecl context");
    let dev: Arc<wgpu::Device> = ctx.wgpu_device().expect("metal wgpu device").clone();
    let queue: Arc<wgpu::Queue> = ctx.wgpu_queue().expect("metal wgpu queue").clone();
    let client = ctx.client();
    (
        GdnCensus::new(dev, queue, layer).expect("gdn census"),
        client,
    )
}

#[test]
fn census_instrument_g1() {
    // ── Phase 1: sleep walls (empty hazards; idle-queue gaps) ──────────
    {
        let (mut c, client) = census(0);
        c.mark(&client, "input_norm", &[]).expect("mark");
        std::thread::sleep(Duration::from_millis(8));
        c.mark(&client, "gemm_qkv_z_ab", &[]).expect("mark");
        std::thread::sleep(Duration::from_millis(2));
        c.mark(&client, "beta_decay", &[]).expect("mark");
        let rep = c.finish().expect("finish");
        assert!(rep.chunks == 0, "a 3-mark run is a partial group, not a chunk");
        assert_eq!(rep.rows.len(), 1, "one partial group row");
        let sleep_wall = rep.rows[0][0];
        assert!(
            (6.0..80.0).contains(&sleep_wall),
            "the 8 ms CPU sleep must read as the first section's GPU wall, got \
             {sleep_wall} ms — the NS_PER_TICK law or the boundary math is broken"
        );
        let small_gap = rep.rows[0][1];
        assert!(
            (1.0..20.0).contains(&small_gap),
            "the 2 ms gap must read near-2 ms, got {small_gap} ms"
        );
    }

    // ── Phase 2: hazard-pinned markers bracket real work ────────────────
    {
        let (mut c, client) = census(0);
        let src = client.create_from_slice(bytemuck::cast_slice(&vec![1.0f32; 512 * 2560]));
        let w = client.create_from_slice(bytemuck::cast_slice(&vec![1.0f32; 2560]));
        let dst = client.create_from_slice(bytemuck::cast_slice(&vec![0.0f32; 512 * 2560]));
        let launch = |client: &Client| {
            unsafe {
                riir_infer_gpu::norms_cubecl::RmsNormBatchedCubeCL::launch::<
                    riir_infer_gpu::cubecl_runtime::ActiveRuntime,
                >(client, src.clone(), w.clone(), dst.clone(), 512, 2560, 1e-6);
            }
        };
        // Warm the pipeline OUTSIDE any timed arm — a cold compile reads
        // ~6 ms and would poison the reference.
        launch(&client);
        pollster::block_on(client.sync()).expect("warm sync");

        // Reference: 10 launches timed with a CPU sync from an idle queue.
        pollster::block_on(client.sync()).expect("pre sync");
        let t0 = Instant::now();
        for _ in 0..10 {
            launch(&client);
        }
        pollster::block_on(client.sync()).expect("ref sync");
        let ref_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Census arm: hazard-pinned markers around the SAME launches on the
        // SAME buffers — marker b's ReadWrite on dst carries the RAW edge
        // against the block's last write, so it cannot retire early.
        c.mark(&client, "sandwich_a", std::slice::from_ref(&dst)).expect("mark a");
        for _ in 0..10 {
            launch(&client);
        }
        c.mark(&client, "sandwich_b", std::slice::from_ref(&dst)).expect("mark b");
        let rep = c.finish().expect("finish");
        let measured = rep.rows.last().expect("row")[0];
        // The sync reference carries host-side submission + poll overhead
        // the census (pure GPU wall) legitimately lacks (~0.3–0.6 ms at
        // n=10 — measured: an idle sync is 0.04 ms, per-launch CPU cost
        // adds the rest), so a modest census UNDER-read is expected. The
        // pathology this guards — unpinned markers sliding ahead of queued
        // work — read 0.05–0.5×; the 0.6 floor plus the linearity check
        // below catch it.
        assert!(
            measured > ref_ms * 0.6 && measured < ref_ms * 1.5,
            "census wall {measured} ms vs sync reference {ref_ms} ms — the \
             hazard-pinned markers do not bracket real GPU work honestly"
        );

        // Linearity: 30 launches ≈ 3× the 10-launch wall (the same hazard
        // chain, three times the work).
        pollster::block_on(client.sync()).expect("pre sync");
        c.mark(&client, "sandwich_c", std::slice::from_ref(&dst)).expect("mark c");
        for _ in 0..30 {
            launch(&client);
        }
        c.mark(&client, "sandwich_d", std::slice::from_ref(&dst)).expect("mark d");
        let rep30 = c.finish().expect("finish");
        let measured30 = rep30.rows.last().expect("row")[0];
        assert!(
            measured30 > measured * 2.0 && measured30 < measured * 4.0,
            "30-launch census wall {measured30} ms vs 10-launch {measured} ms — \
             the census does not scale with real work"
        );
    }

    // ── Phase 3: grouping + the inter-chunk skip ────────────────────────
    {
        let (mut c, client) = census(3);
        // A full chunk group with 2 ms gaps between marks (idle-queue
        // sound), then a sleep across the inter-chunk boundary ("end" →
        // next chunk's first mark), then two more marks. A MISSING skip
        // would fold the 6 ms inter-chunk wall into a section.
        for i in 0..MARKS_PER_CHUNK {
            c.mark(&client, ["input_norm", "scan", "end"][i % 3], &[])
                .expect("mark");
            std::thread::sleep(Duration::from_millis(2));
        }
        std::thread::sleep(Duration::from_millis(6));
        c.mark(&client, "input_norm", &[]).expect("mark");
        std::thread::sleep(Duration::from_millis(2));
        c.mark(&client, "scan", &[]).expect("mark");
        let rep = c.finish().expect("finish");
        assert_eq!(rep.chunks, 1, "17 marks = one full chunk group + 2 open marks");
        assert_eq!(rep.rows.len(), 2, "one full row + one single-section row");
        assert!(
            rep.inter_chunk_ms >= 5.0 && rep.inter_chunk_ms <= 80.0,
            "the 6 ms sleep sits on the inter-chunk boundary and must be \
             carried there, got {} ms",
            rep.inter_chunk_ms
        );
        assert!(
            rep.rows[0][0] < 12.0 && rep.rows[1][0] < 12.0,
            "only 2 ms gaps inside either group — first sections must stay \
             small: group0 {} ms, group1 {} ms",
            rep.rows[0][0],
            rep.rows[1][0]
        );
    }
}
