# Bench 913 — Plan 623 / issue 1147: the 2-box wire confirmation cell REFUTES the fused decode batch (the loopback +39% does not transfer)

**Status:** RECORDED — the fused batch (BatchN, `pair_decode`) REFUTED on the serving topology; default DEMOTED to the incumbent cadence (`RIIR_DIST_PAIR=1` arms, loopback rigs only); G1 PASS on the wire ×2 streams (first wire G1)
**Session:** zed-amo-i623-mb (continuation, 2026-10-28)
**Date:** 2026-10-28
**Plan:** `.plans/623_multi_host_pipeline_inference_lane.md` T2.4 · `.issues/1147_t24_kernel_micro_batching.md`
**Supersedes (the posture, not the numbers):** the Bench-912 promotion was loopback-only — [Bench 912](912_plan623_issue1147_fused_decode_batch.md) stands as measured; this record adds the topology axis its confirmation cell owed

## Why this cell existed

Bench 912's +39% GOAT was measured on **bmo loopback** (head + tail on ONE box, one
shared CubeCL queue). Its own record named the owed forwarder: "the 109.5 2-box row is
Bench 908's wire topology — its confirmation cell waits on a quiet amo window." The
window opened (amo + bmo both verified idle, no compute consumers on either box) and
the cell ran.

## Provenance

- amo = head (M5 Ultra, load 1.5–1.9 during cells; the rethink serve resident at 0%
  CPU — no forwards in flight), bmo = tail (M5 Ultra, load ~1.0, no compute
  consumers) — both verified idle via `ps` before firing.
- Both binaries built at `fc28c98` (`dist_stage`, `--features gpu,bonsai2_hadamard`),
  sha256 prefix `d2a74b652e624bd0` **byte-identical on both boxes** (bmo synced by
  incremental bundle `^39af44c..develop` + ff-merge; its three sibling-local differing
  scripts moved aside to `scripts/.bmo_local_bak/`, never deleted).
- Model `Ternary-Bonsai-2-27B-PQ2_0.gguf` (amo `~/git/riir-train/data/`, bmo
  `~/models/`), split 32/32 GPU both stages, TB5 link (192.168.10.1 ⇄ 192.168.10.2),
  `--streams 2`, prompt `7,13,42,100`, `--max-new 32 --no-eos`, ctx 4096.
- Session key: fresh random 64-hex per battery (the t03-session convention).
- Rig: `.wt/i623_wire_cell/cell.sh` (tail on bmo via ssh, readiness = poll the tail
  log's `listening on` — never a TCP probe, the single-accept trap; landed as
  `scripts/plan623_wire_cell.sh` in the same commit).

## The cells — ABBA ×2 + a G1 rider

| cell | order | arm | decode window (agg) | per-stream |
|---|---|---|---|---|
| ab_p1_a | 1st | **armed** (BatchN) | 52.59 tok/s | 26.3 / 26.5 |
| ab_p0_b | 2nd | incumbent | 99.20 tok/s | 49.6 / 50.1 |
| ab_p1_b | 3rd | **armed** | 46.93 tok/s | 23.5 / 23.7 |
| ab_p0_a | 4th | incumbent | 108.46 tok/s | 54.2 / 54.8 |
| g1_p0 | 5th | incumbent + `--full-check` | 108.50 tok/s | 54.3 / 54.9 — **G1 PASS both streams** |

- Incumbent replicates Bench 908's 2-box row (~109.5) and the G1 rider is the FIRST
  wire-topology bit-identity pass: both streams' tokens byte-identical to the 1-box
  GPU oracle (45.42 tok/s incl. prefill) run on the head box.
- Armed = **46.9–52.6 aggregate (−52% vs incumbent 99.2–108.5)**, per-stream halved
  too. ABBA order rules out drift; the gap is 2× against ±10% cell noise.

## The mechanism (pinned in the code, not inferred)

`run_head_multi` computes the head stage SYNCHRONOUSLY per stream (`step()` — the
hidden IS the readback; the frame cannot be encoded before it completes):

- **Incumbent (`max_drain = 1`)**: one frame in flight per pass, streams rotate —
  while the tail's GPU runs stream A's chain, the head's GPU computes stream B's
  head-half. **The two GPUs overlap continuously — that overlap IS the 2.37×
  bubble fill** (Bench 908).
- **Armed (BatchN + drain-all)**: the head must finish BOTH head-halves before it
  can send either frame, and the tail runs BOTH tail-chains before replying. The
  pipeline collapses into whole waves: head(A)+head(B) → wire → tail(A)+tail(B) →
  wire → … **zero cross-box overlap — the pipeline's entire win forfeited**, hence
  ~2×.

On loopback there is no second GPU: the batch's win was real there (inter-chain
host-idle removal between the shared queue's chains, +39%) and stays available for
co-located rigs via `RIIR_DIST_PAIR=1`.

## Verdicts

1. **The fused decode batch is a single-GPU optimization** — loopback +39% (Bench
   912) stands for co-located stages; on the 2-box wire it is a −52% loss. Demoted:
   `pair_decode` now defaults OFF (`HeadOptions::default` + the bin's env parse);
   `RIIR_DIST_PAIR=1` arms it. The incumbent cadence is byte-identical to the
   pre-1147 path (s=1 never arms either way — `all_feed` requires `ready_ct > 1`).
2. **The kernel-level batch-2 rung (issue 1147's forwarder) is CLOSED for the wire
   topology**: under the incumbent cadence the tail never holds two ready decode
   chains (it replies each frame before the next arrives), so a grid-dim batch has
   no engagement window. The rung's remaining scope is the loopback rig only — the
   +39% bar would apply there, and the wave-vs-overlap law (this record) prices it:
   co-located stages share one GPU, so a batch that cuts launches per wave helps
   exactly where the wave already wins. Serving-path micro-batching via BatchN is
   not a direction on this topology; the continuous-batching ADMISSION seam (serve
   multi-tenancy) is unaffected.
3. **T2.7 scoreboard unchanged**: the s2 = 2.37× / s3 saturated rows ARE the
   incumbent cadence — this cell re-confirms them (108.5 vs 109.5 across days).

## Honesty notes

- Both boxes idle-verified at fire; the amo 1-min load read 1.5–1.9 during the
  battery (this session's own rig + ssh), not a compute consumer on the GPU.
- Cell noise ±10% (amo-side scheduling); the measured gap is 2× — the verdict does
  not sit on the noise.
- The armed arm's loopback G1 (Bench 912, ×2 builds) is untouched by this record;
  the wire G1 here is the incumbent arm.
- **Post-cell loopback smoke DISCARDED as contaminated (same session, ~05:05–05:13):
  post-demotion default checks on amo loopback read 78.55 (unset) / 78.39 (=0) /
  56.11 (=1) / 12.77 (=1, last) — and a `ps` spot-check then found a sibling lane's
  own `dist_stage` loopback cells live on amo (`/tmp/i1146-tgt-armed/…`, two
  processes). Concurrent loopback stages share one GPU; none of those reads is
  usable. The demotion does not rest on them: the wire cells above are the
  serving-topology evidence, and the default is the pre-1147 byte-identical path.
- **Bench 912's loopback arm classes carry an open question this record cannot
  close**: the contaminated amo reads happened to cross-match 912's bmo classes
  (this session's incumbent-cadence reads ≈ 912's armed 78; the one armed read ≈
  912's incumbent 56), and the sibling t03 anchor (env-unset → armed under the
  pre-rebase parse) read 56.70 ≈ 912's incumbent class. That pattern is consistent
  with 912's A/B rig having labeled its arms inverted — the +39% would then be the
  incumbent arm's own number — but the 912 rig is not inspectable (worktrees
  cleaned) and every read here is contaminated, so this is a HYPOTHESIS, not a
  finding. The lever: one verified-idle loopback A/B re-read (either box) with a
  BatchN-frame counter as the arm oracle — rate alone cannot discriminate once two
  sessions share a box. The demotion stands regardless (the wire refutation + the
  conservative default); only the loopback +39% attribution waits on that read.
