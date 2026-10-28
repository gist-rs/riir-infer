# Bench 927 — Plan 623 Phase 3 T3.1: verbs bring-up on the amo⇄bmo TB bridge (first RDMA traffic)

**Date:** 2026-10-29
**Lane:** Plan 623 Phase 3 — [`T3.1` verbs FFI + UC QP bring-up](../.plans/623_multi_host_pipeline_inference_lane.md), the T3.2 heap leg, the T3.3 small-message + striped rows.
**Commits:** riir-infer `226cace` (FFI lane) · `16fb64a` (UC RTS mask) · the two verbs_ping protocol fixes (header offset, recv ring).
**Provenance:** amo (M5 Ultra 96 GiB, macOS, idle, load 0.77) ⇄ bmo (M5 Ultra 96 GiB, load 0.45; **the mammoth teacher capture was running on bmo's GPU during every row — CPU-idle, no capture traffic crosses the TB wire**; every number below is CPU+wire only). Driver: `verbs_ping` release builds; amo built at HEAD, binary shipped to bmo over the bridge. Both sides enumerated `["rdma_en2","rdma_en13","rdma_en3","rdma_en4","rdma_en5","rdma_en6"]` (all six are bridge0 members); device auto-picked `rdma_en2`, port 1, **link_layer=thunderbolt**, active_mtu=4096, max_msg=16 MiB, gid_tbl_len=1024, max_qp_wr=4095, max_sge=1, max_cqe=4096.

## What ran

One UC QP pair (256/256 WRs, 224 B inline granted), endpoints exchanged over the TCP side channel
(192.168.10.1⇄.2, BLAKE3-keyed MAC both ways), both sides through the same INIT→RTR→RTS walk.
GIDs are NOT IPv4-mapped on this transport (link-local-style 64-bit GUIDs; `pick_gid` fell back to
index 0 — the fallback is the WORKING path here).

## Rows

| probe | shape | result |
|---|---|---|
| **RTT** | 64 B SEND/RECV ping-pong, inline, n=1800 (10% warmup discarded), content-verified every 256 | **p50 7.2–9.6 µs, p99 7.5–12.5 µs** across runs; the cleanest early run read p50 **7.709 µs** / p99 8.292 µs |
| **BW** | one-way striped push, windowed | 4 KiB stripes 4.55 GiB/s · 16 KiB 9.17 · 64 KiB 9.10 · 128 KiB **8.76** · 1 MiB 8.68 GiB/s |
| **BW sustained** | 1 GiB total, 128 KiB × window 24 | **9.368 GiB/s one-way** (114.6 ms) |
| **security negative** | host/peer with different keys | BOTH sides refuse at the endpoint exchange: `endpoint MAC mismatch — refusing unauthenticated RDMA endpoint`; zero QP traffic |

vs the incumbents on the same pair: TCP RTT @4 B was 94–96 µs (Bench 896) → **12×**; TCP bulk was
8.6 GB/s (Bench 896) → **1.09×**; jaccl pipelined all_sum rate was 16–22 µs (Bench 905) — the
verbs ping-pong RTT is BELOW that pipelined collective rate (different shape; transport-level
dominance, not a collective comparison).

## Provider laws discovered (the T3.2/T3.3 payload)

1. **UC RTS mask is STATE|SQ_PSN only.** `IBV_QP_TIMEOUT` on a UC QP → `ibv_modify_qp` EINVAL
   (mask 0x10201, rc=22). RC keeps TIMEOUT (type-aware mask in `Qp::to_rts`).
2. **Posted-recv budget: 4 MB OK, 5 MB → `ibv_post_recv` rc=-12 (ENOMEM).** Measured at
   16×256 KiB (4 MB) PASS vs 20×256 KiB (5 MB) FAIL; shape-independent (16 KiB×64, 128 KiB×24,
   1 MiB×4 all pass at ≤4 MB). The transport design law: the recv ring stays ≤4 MB posted;
   a 2.6 MB activation frame fits ONE posted recv at 4 MB budget — stripes only if frames grow.
   ⚠ First-run 32×256 KiB "failures" in the shape ladder were STALE-HOST contamination (a dying
   listener still bound; `pkill` + immediate rebind race) — the clean re-runs above are the record;
   the cliff bisect (24/20×256 KiB) was run clean.
3. **rkey granted 0x0** with LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ requested — remote-access keys
   are not exercised by this provider posture (or not surfaced); **SEND/RECV is the transport
   shape**. RDMA WRITE (the zero-copy one-way shape) is untested until rkey semantics are
   clarified — the open half of T3.2's zero-copy claim.
4. **Heap registration of 67 MB accepted** (lkey granted) — the T3.2 heap leg.
   The **MTLBuffer leg is OPEN** (needs a Metal-context probe on the boxes; contents() of a
   shared MTLBuffer is ordinary pages on Apple silicon, so acceptance is EXPECTED — measure, don't assume).
5. **UC credit flow control is real:** with the receiver torn down mid-push, the sender's SQ
   stalls with NO send completions (not error completions) — a dead peer reads as a poll
   timeout, never as a WC error. The transport's health check must be time-based, not WC-based.

## Traps paid (all fixed in the same session)

- Host read the bulk header from the bulk area while its recv buffer sat at offset 0 → parsed
  zeros → early exit → peer hit UC RNR (`post_send` rc=-12) → then the credit stall above.
- A 16-byte header can TRUNCATE on UC if the posted recv is shorter (truncate = QP error, not a
  short read) — recv length is now `max(rtt_msg, HDR_LEN)`.
- Pre-posting the full stripe count broke at the posted-recv budget — the host ring is
  window-cycled (slot k reused only after k's WC reaped).

## What this buys the plan

The Phase-2 2-box decode loss (−16..−26%, Bench 900/906/911) is TCP-shaped: the per-token hop
paid ~0.09–0.10 ms of TCP RTT against a ~14 ms step — but the PP sequencing loss was NEVER
bandwidth-bound. The RDMA hop at **7.7 µs** removes ~92% of the TCP hop latency; whether the
2-box decode row flips positive is **T3.4/T3.5's question** (GPU⇄comm sync + the real scoreboard
rows over `dist_stage`), not this bench's claim.

**Not claimed:** no serving integration yet (`verbs` is a probe lane); no RDMA WRITE row; no
MTU-override ladder (all rows ran at the active 4096); T3.5 scoreboard vs llama.cpp RPC-RDMA
(not built) stays open.

**Local (m3) validation:** dlopen + symbol resolution green (no RDMA device locally —
enumeration honest-loud); 6/6 layout pins vs the C compiler's measured offsets; 5/5 unit tests;
clippy -D clean at `rdma` and flag-off postures; 24 passed flag-off = byte-identical G3.
