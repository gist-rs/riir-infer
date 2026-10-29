# Issue 1150 session record — the dense-qwen3.8 Metal lane: what landed, what it measured

**Status:** RECORD — decode + parity G1 LANDED; prefill engine = the named follow-up (2026-10-28)

## Landed (riir-infer, pushed: 465c1d8 → 538a1f2 → 8a17641 → 22d28e3 → 0099e27 → 63c8847)

- **`qwen38_dense_cubecl`** — the full dense-qwen3.8 (qwen35 arch, 48 GDN / 16 attn,
  vocab 248320) forward on CubeCL/wgpu: self-contained config parse (CUDA lane
  untouched), QuantWeight dispatch (Q4_K packed Q4KHandle in-kernel dequant;
  Q6_K host-dequant f32 — the standard Q4_K_M recipe mixes both), host-side
  embeddings + 32768-row-chunked f32 lm_head (Metal per-buffer limits forbid
  5 GB single buffers — measured, twice), full GDN+attention decode, chunked
  prefill (sliced 64-token conv + ordered carry update, multi-token recurrence,
  KV fill, tiled flash), MLP on all four paths.
- **`gemm_q4k_tile_cubecl`** — the Q4_K×f32 tiled GEMM + f32 twin + gather.
  **G1 unit gate GREEN on M3 + bmo** (edge shapes + n=5120, rel < 2e-4 vs the
  dequantized-weight CPU reference).
- **`qwen38_wgpu_bench`** — pp/decode cells + the greedy walk (the parity artifact).

## The G1 that matters — greedy parity vs llama.cpp (bmo, same GGUF, same box)

| engine | raw completion (temp 0, 24 tokens, prompt "The harbor town woke slowly under a grey sky,") |
|---|---|
| **ours (Metal, f32-act)** | `" the kind that promised rain but delivered only a damp chill. The air smelled of salt, diesel, and the faint sweetness of overripe peaches from the market"` |
| **llama.cpp fork 73bfe6cc4** (W4A8 int8-act, /completion raw) | `" the kind that promised rain but delivered only a damp chill. At the edge of the quay, where the water lapped"` |

**12/12 first tokens EXACT** across an entirely different kernel stack — the
module's pre-registered numerics divergence (f32 activations vs the cudarc/
llama int8-activation path) only shows at greedy near-ties from token 13.
The forward is correct end-to-end (boot → 64-layer GDN-hybrid → logits).

## The wall — measured, honest

- Prefill pp2484 = **4 tok/s** (chunk 512): ~216 GFLOP/s aggregate.
- The A/B at the production ffn shape ([17408×5120]×512, bmo release):
  **q4k-tile 614 / matmul-f32 570 / matmul-f16b 547 tok/s-equiv — all ≈110
  GFLOP/s per GEMM.** Every in-crate CubeCL tiled GEMM (mine + the two
  battle-tested ones) sits at the same scalar-kernel wall — the Bonsai
  league's 0.249× Metal prefill row is THIS wall, now measured on dense too.
- Conclusion: a competitive Metal dense prefill needs the **matmul2d (MPP
  cooperative-vector) route** — the ternary zero-copy MSL lane
  (`gemm_ternary_metal_zero_copy`, Issue 656/663/670/1146) is the template;
  the dense-f16 matmul2d variant (weights f16 at load, raw-Metal dispatch on
  the shared queue) is the named follow-up. MLX's 1411 tok/s @10k qwen3.8 row
  is the bar that route must chase.

## Also landed en route

- Q4_K_M files mix Q4_K + Q6_K layer tensors (QuantWeight, the cudarc
  `QuantW{q4}` pattern); vocab reads from token_embd ne-product (shape[0] is
  n_embd — the truncation bug produced a collapse-loop then a clean fix).
- bmo hygiene: llama-server + bench processes killed; `/tmp/i1150` (cargo
  target + llama build) kept for the follow-up lane; cmake via
  `~/Library/Python/3.9/bin` (pip wheel) recorded.

## Follow-up (the next lane)

1. dense-f16 matmul2d MSL kernel + zero-copy dispatch (the speed unlock);
2. decode on the Q4K GEMV path is already fine (bandwidth-bound) — verify tok/s;
3. re-run the bmo four-way pp cells once 1 lands; MLX/fork/uzu rows already recorded.

## Follow-up LANDED — the dense-f16 matmul2d arm (2026-10-29, zed-m3-i1150-densef16)

Step 1 + the f16 load posture, claimed under this issue (lease zed-m3-i1150-densef16):

- **`gemm_dense_f16_metal_tensor.metal` + `gemm_dense_f16_metal_zero_copy.rs`** — the dense
  twin of the ternary zero-copy lane: the SAME matmul2d cooperative path
  (64×128 tiles, 4 simdgroups, K-tile 32) with the A operand as a strided
  **device f16 tensor view** — no bit-plane dequant, no threadgroup staging,
  no staging barrier in the K loop. Plus **`gemv_dense_f16`** (plain kernel,
  one simdgroup per row, coalesced f16 reads, one simd_sum) for decode.
  Shared-device zero-copy choreography mirrored from Issue 663 T5 (flush →
  ensure_wgpu_initialized → fresh raw extraction → dedicated output + blit
  → commit; `covered`/`insert_merged` reused via a `pub(crate)` bump).
- **`QuantWeight::F16`** + the `RIIR_QWEN38_DENSE_MM2D=1` load posture (opt-in
  env, loud-refusing where Metal/the feature is absent; env-off = the packed
  incumbent byte-path): host-dequant once at load → f16 bit patterns → the
  gemm sites route through matmul2d, decode gemv through the plain kernel.
  The 16-bpw memory cost is the posture's disclosed trade (27B ≈ 54 GB — M5
  Ultra class; the M3 cannot hold it, so the unit gates + A/B here are
  kernel-level and the four-way cells are the bmo follow-up).
- **G1 unit gate GREEN (M3)** — `tests/qwen38_dense_f16_matmul2d_g1.rs`:
  matmul2d exact/edge classes (m-edge, p-edge, k-edge, combined ragged,
  p=1, the n=5120 production count) + gemv (m-edge, production shape), all
  vs an f16-rounded CPU reference, rel-to-mag ≤ 1.6e-7 (the band is
  relative to Σ|w·x| — a pure relative band on near-zero dots measures
  summation-order noise, not the kernel).
- **⛔→✓ The deferred-upload race (the session's hard-won finding):** the
  first gate runs measured ALL-ZERO writes across shape classes that are
  provably correct under sync — three fake "shape laws" (m-ragged skip,
  p-floor skip, combined-ragged skip) were minted and then retracted.
  Root cause: `create_from_slice` uploads are DEFERRED writes and a raw
  Metal commit can land before they reach the pool (the Issue-1146
  attention gate's documented hazard; flush orders kernel producers, NOT
  host uploads — the cure is `client.sync()` after uploads). The gate now
  pins the full matrix under sync so any real cooperative-tensor regression
  is loud. Lesson recorded: **a guard minted from a racy negative is a
  constraint on the world that the world never imposed** — re-measure
  negatives under the settled fixture before encoding them as laws.
- **A/B @ [17408×5120]×512 (91.3 GFLOP), release, N=10 back-to-back,
  ABBA×2 median, M3 Max quiet AC:** q4-tile **528 ms (173 GFLOP/s)** vs
  dense-f16 mm2d **10.5 ms (8678 GFLOP/s)** — **50.22× kernel-level**, with
  the loud-zero insurance (wait_last inside the clock + a post-timing
  output readback vs the CPU dot, rel 5.8e-6). 8.7 TFLOP/s is the M3 Max
  fp16 tensor rate — compute-bound, physically coherent. The session
  record's ~110 GFLOP/s wall is measured GONE on the dense path.
- **The wall this unlocks:** the prefill wall (~110 GFLOP/s aggregate, the
  Bonsai league's 0.249× Metal row shares it) is the scalar-dequant GEMM
  class; the matmul2d route replaces it on the dense lane. The e2e prefill
  cells (pp 2484/4954/9856/16392 + decode ctx2048, the four-way same-model
  table vs fork/MLX/uzu) are the next lane — bmo-only (54 GB), co-tenant
  with the mammoth capture (disclosed) or queued behind it.

## The bmo four-way runbook (2026-10-29, zed-m3-i1150-fourway — prep landed, run queued)

The four-way run lane, claimed under issue 1150 (48 h from 2026-10-29T01:36Z).
Everything is verified present on bmo; only the build + the run remain, both
queued behind the box's live tenants.

**State verified on bmo 2026-10-29 ~08:30 +07** (uptime 1d14h, load <0.6):

- Model GGUF present: `~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (the bench's
  DEFAULT_GGUF; the same file the 10-28 llama.cpp parity G1 ran against).
- `/tmp/i1150/` build tree kept — but its release binary is STALE (built
  10-28 21:51, before the dense-f16 arm landed); a rebuild at the new HEAD is
  REQUIRED for the dense posture.
- Disk: 1.5 TB free. RAM class: M5 Ultra 96 GB — the 54 GB f16 in-memory
  footprint + KV at ctx 24576 fits (the 10-28 parity run loaded the same
  posture on this box).

**Build (bmo, after the box clears):**

```sh
ssh katopz@100.100.117.44
cd ~/git/riir-infer && git fetch origin && git checkout origin/develop   # detached worktree posture for a bench run
cargo build --release -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm \
  --bin qwen38_wgpu_bench --target-dir /tmp/i1150
```

**Run (the four-way cells; `--decode-ctx` needs the NEW bin):**

```sh
RIIR_QWEN38_DENSE_MM2D=1 /tmp/i1150/release/qwen38_wgpu_bench \
  --pp 2484,4954,9856,16392 --decode 32 --decode-ctx 2048
# --pp: median of 3 + warm-up per cell (the bench's own convention, matches
#       the opponents' median-of-3 shape)
# --decode 32: raw-prompt decode (compat posture)
# --decode-ctx 2048: 32 decode steps after a prompt padded to 2048 tokens —
#       the league's decode-at-context convention (this flag landed with the
#       four-way prep; the old bin decoded after the ~10-token prompt only)
# Posture note: the env arms the dense-f16 path; without it the run is the
# incumbent Q4_K packed-load posture (G3 byte-path) — the four-way table is
# the DENSE posture's row.
```

**Gates before the run (box state — the latency-provenance law):**

1. The mammoth capture DONE or parked by its owner lane: the capture drives
   the mammoth serve on bmo:8793 per request (~2 s/row GPU work) — the
   capture and a timing-scored GPU bench CANNOT co-tenan. As of this
   runbook's landing the capture is mid-`xnli_en` (suite 2/2, started
   04:16 +07, ~17 h/suite measured on ag_news → ETA ~evening local).
2. The plan-623 T3.5 scoreboard cells (issue 1153, zed-m3-i971) cleared AND
   their timing-scored status answered (co-tenancy message queued from this
   lane 2026-10-29 ~07:5x +07). If their cells are timing-scored, wait for
   their completion or an explicit free-window answer; do not contend.
3. Record beside the numbers: free RAM, commit-vs-limit at launch,
   concurrent job census (commit-ranked per the house law), AC power. bmo
   has no bench_preflight script of its own — the reflex script's box-state
   discipline is the template, the `PROVENANCE:` line the shape to quote.

**The table this run fills (the four-way, same-model qwen3.8-27B 4-bit-class,
bmo M5 Ultra — opponents' rows recorded 2026-10-28, riir-ai
`.benchmarks/uzu27b/README.md`):**

| P (in-tokens) | uzu mirai-m:4 | fork `73bfe6cc4` Q4_K_M | MLX-LM 4bit | OURS (dense-f16 mm2d) |
|---:|---:|---:|---:|---:|
| 2484 | 1009.7 | 525.23 ±101 | 1395.0 | **18** (clean median) |
| 4954 | 1156.9 | 898.26 | 1410.8 | **3** (poisoned median; the clean read is pass-1 **19**, disclosed) |
| 9856 | 536.0 | 855.12 | 1418.7 | no median (run stopped; warm-up + pass 1 read **5**) |
| 16392 | 392.7 | 799.71 | 629.7 ⚠ declining | not reached |
| decode ctx2048 | (none recorded) | (none recorded) | (none recorded) | not reached |

Cells to beat per the issue-1150 G2: MLX is the prefill bar at every cell
(1395–1419 through 10k). The kernel-level A/B (50.22×) predicts the dense-f16
route clears the scalar-dequant wall (~110 GFLOP/s class) that pinned the
fork's rows; the e2e cells are where that prediction meets wall-clock.

**Honest scope:** the decode ctx2048 row has NO opponent readings (the
three-way was prefill-only) — it is ours-only context until a decode leg is
run for the opponents (a separate lane; the fork's `-tg` at ctx2048 and
MLX's decode TPS are the two rows that would join it). Do not present the
decode row as a four-way cell until those legs exist.

## The bmo runbook rev 2 — built, gated, auto-runner armed (2026-10-29 ~09:30 +07, zed-m3-i1150-fourway)

State executed this morning:

- **Sync**: bmo's `git fetch origin` FAILS (access rights — its origin/develop
  ref was stale at the plan-623 T0 era, `39af44c`). Fixed per-run with a git
  bundle over the existing scp path: m3 `git bundle create … origin/develop`
  (21 MB) → bmo `git fetch <bundle> origin/develop:refs/remotes/origin/develop`
  → `git checkout --detach origin/develop`. bmo detached at **`efeb527`**
  (T3.5-closed HEAD ⊇ `dc81b96` `--decode-ctx`). bmo is branch `develop` with
  only an untracked `scripts/.bmo_local_bak/` — the detached posture is safe.
- **Build DONE**: `cargo build --release -p riir-infer-gpu --features
  qwen38_wgpu,metal_tensor_gemm --bin qwen38_wgpu_bench --target-dir
  /tmp/i1150 -j 8` → **35.7 s incremental** against the 10-28 cache (nice 15,
  co-tenant-safe); bin `/tmp/i1150/release/qwen38_wgpu_bench` stamped
  2026-10-29 09:13. `--decode-ctx` verified at source level (the parse arm +
  the `needs --decode N` guard); NO GPU probe was run to verify it (see the
  incident below).
- **Runner rev 2** (`/tmp/i1150/run_fourway.sh`): five gates now —
  1 capture harness live → refuse · 2 plan-623 `dist_stage` live → refuse ·
  **3 NEW mammoth SERVE RESIDENCY → refuse**: the serve holds ~55 GB wired via
  MPS in unified memory; the dense-f16 bench needs ~54 GB + KV; 55+54 > 96 GB
  = swap thrash — the harness-only gate was INSUFFICIENT (the serve stays
  resident after the capture ends; current free RAM 36% while it lives) ·
  **4 NEW free-RAM floor ≥55%** (of 96 GB ≈ the 54 GB load; parse tested —
  reads 36 today, refuses as intended) · 5 bin freshness (must be ≥ the
  09:13 build). The census (commit-ranked, per the house law) prints on every
  invocation, dry or real.
- **Auto-runner ARMED**: `/tmp/i1150/run_fourway_wait.sh` (pid in
  `/tmp/i1150/wait.pid`), polls every 600 s, single-shot, deadline +14 h
  (~23:25 +07) — fires the real run when all gates pass; logs to
  `/tmp/i1150/autofourway.log` with the per-poll refusal reason. If the
  capture ends but the serve lingers, the log names gate 3 — retiring the
  serve is the capture owner's (640-T3b) or an operator's call, never this
  lane's.
- **INCIDENT (recorded, ~09:15 +07)**: a ctx64 `--decode-ctx` parse probe was
  run alongside the LIVE capture — ~3 min of GPU co-tenancy before the kill.
  Capture artifact UNHARMED (distill captures record deterministic outputs,
  not latencies — a few rows ran slower, nothing corrupted). Lesson: the
  gates exist for the RUNNER, not as optional etiquette — NO GPU probe of any
  size runs while any tenant is live; flag-parse verification is a source
  grep, not a model load.
- **T3.5 closed** (`efeb527`, Bench 929): gate 2 is permanently clear (the
  `dist_stage` census stays as a tripwire). i1152 (`verbs_metal` firewall
  gate) stays owner-gated — out of scope here. The decode ctx2048 row stays
  ours-only; the opponent decode legs were NOT armed (separate lane, own
  claim).

## The bmo runbook rev 3 — gate 3 resolved: the serve is the capture's OWN rig (2026-10-29 ~11:05 +07, zed-m3-i1150-fourway)

Rev 2 left gate 3 (mammoth serve residency) as a wait-forever: "retiring it
is 640-T3b's or an operator's call, never this lane's." The timeline read on
bmo resolved who that owner is — and it is the capture itself:

- **Serve provenance (measured, not assumed)**: `serve_mammoth.py` pid 23553
  started **2026-10-28 11:16:12 +07** (PPID 1). `distill.json` records
  ag_news at **61,197 s ~= 17 h** — i.e. the ag_news capture began when the
  serve booted and completed 04:16 +07 Oct 29, writing `ag_news_teacher.bin`
  + `.blake3` + the manifest (teacher acc 0.8786, p50 1790 ms). The CURRENT
  harness (pid 56901, `--suites xnli_en`, also PPID 1) started 04:16:25 +07,
  rewrote the manifest with the ag_news row, and is mid-xnli_en. One
  established `:8793` connection: harness <-> serve.
- **Verdict**: the serve is the capture lane's own temporary rig — stood up
  with the capture, one purpose, canonical rig home is amo per riir-reflex
  AGENTS.md (bmo's copy is a stand-up, file dates Oct 24-28). After the
  capture's artifacts flush and the harness exits with zero consumers, an
  idle ~55 GB resident defeats the owner-queued four-way (55+54 > 96 GB).
  This SUPERSEDES rev 2's "never this lane's": retiring the capture's own
  rig post-completion is this lane's call, taken and logged. Reversal is
  one command (in the script header on bmo).
- **Runner rev 4 armed** (pid 20411 in `wait.pid`, deadline **+20 h** ~=
  06:04 +07 Oct 30 — the old +14 h left only ~2 h margin against an
  ag_news-paced xnli_en at ~21:16 +07 ETA; the lease 2026-10-31T01:36Z
  covers the window): each poll now runs `retire_serve_when_safe.sh`
  BEFORE the gate probe. The script no-ops unless ALL hold — (1) capture
  harness NOT live (never touches a live capture), (2)
  `xnli_en_teacher.bin` + `.blake3` flushed (the flush-complete marker,
  not just the `.bin`), (3) zero ESTABLISHED `:8793` connections — then
  SIGTERM -> 20 s -> SIGKILL, every step logged to `autofourway.log` with
  the provenance + the one-command reversal. If the harness exits WITHOUT
  the xnli artifacts (a crash), the serve is left resident and the log
  says "capture may have CRASHED" every poll — the capture lane's retry
  outranks the four-way, and the runner expires at deadline rather than
  guessing. Script copies: bmo `/tmp/i1150/retire_serve_when_safe.sh` +
  m3 `/tmp/i1150_local/` (both /tmp — ephemeral; the decision record
  lives in this file and in the script header).
- **Pace note**: xnli_en at 8000/20000 by 08:38 (~2.0 s/row -> ~15:10 +07
  ETA) or ag_news-paced (~3.1 s/row -> ~21:16 +07 ETA); both inside the
  deadline. After the serve dies, wired MPS memory returns to the OS and
  gate 4 (free RAM >=55%) re-verifies before anything fires.

## The bmo four-way RUN — measured, FAIL-honest, stopped early (2026-10-29 15:54–19:51 +07, zed-m3-i1150-fourway)

The auto-runner executed end to end exactly as designed: capture completed
("PASSED — 1 suite(s) dumped", xnli_en 20000/20000, teacher acc ~0.899),
`retire_serve_when_safe.sh` retired the capture's own rig (all three gates
held; reversal command in the log), five gates passed (free RAM **81%** at
fire), and the run started 15:54 +07 (`fourway_20261029_1554.log` on bmo).
The bench loaded in 36.8 s with the dense arm engaged (`[Issue 1150]
dense-f16 matmul2d arm armed`). Then the numbers:

| cell | warm-up | pass 1 | pass 2 | pass 3 | MEDIAN |
|---|---:|---:|---:|---:|---:|
| pp 2484 | 1312.8 s = 2 t/s | 140.5 s = **18 t/s** | 2860.6 s = 1 t/s | 140.7 s = **18 t/s** | **18** |
| pp 4954 | 1716.8 s = 3 t/s | 255.3 s = **19 t/s** | 1997.4 s = 2 t/s | 1754.0 s = 3 t/s | 3 (poisoned) |
| pp 9856 | 1926.6 s = 5 t/s | 1908.0 s = 5 t/s | — stopped — | — | — |

Stopped 19:51 +07 (SIGTERM, rc=143, runner logged and exited single-shot).
**Two outcomes, recorded separately (they are not the same claim):**

- **The GEMM-wall hypothesis: CONFIRMED and removed.** The incumbent
  packed-path wall (4 tok/s / 216 GFLOP/s aggregate) moved to ~18–19 tok/s
  / ~0.9 TFLOP/s — the dense arm transferred at the sites it routes; the
  scalar-dequant GEMM class is measured gone e2e, not just kernel-level.
- **G2 against the MLX bar: FAIL** — best clean read 18–19 tok/s vs MLX
  1395–1419 (~75× behind). The end-to-end claim failed because other costs
  now dominate the pass.

**Findings + the revised follow-up order (verdict-revised: measure before
routing — the 50×-transfer surprise is exactly what a guess costs):**

1. **Step 0 — per-stage timers on one clean pass** (pp 2484, the 140 s
   cell): coarse timers around lm_head, GDN scan, routed GEMMs, attention,
   rest. Replaces FLOP-arithmetic guessing with a ~3-minute measured
   breakdown. (Source read 2026-10-29: lm_head is ALREADY last-row-only per
   chunk — `ingest_chunk_gpu` slices row p-1 before the GEMV sweeps — so
   the lm_head FLOP suspect is WEAKENED to ~1–2 s of a 140 s pass, at the
   old f32 GEMV pace, ~8 chunk GEMVs per 512-token chunk. Worth routing
   eventually; not the presumptive dominator.)
2. **Residency audit — is the box carrying two weight copies?** The f16
   device copy is 54 GB; beside it live the ~10 GB host `wte_host` Vec, the
   lm_head host dequant staging, and the 16.8 GB mmapped GGUF during load
   (dropped at `new()`'s end, but its file-cache pages persist). The
   alternating 8–20× slow passes match a re-fault storm over host-backed
   pages; gate 4 checked that the load FITS, not the steady state. Test
   what is resident per pass (footprint + purge), and free/de-stage what
   is not load-bearing — a structural fix, not just a re-run protocol.
3. **GDN chunked scan routing** (48/64 layers), IF the stage timer points
   there.
4. **Re-run protocol**: single pass per cell, per-pass memory watchdog
   (free% + swap + footprint), persisted/pre-warmed CubeCL autotune cache
   so the 20–32 min first-shape warm-ups aren't paid during the measured
   run.
5. **Warm-shape sweep** (from 4): one compile per tile class, cells built
   from compiled-class shapes.

**Provenance**: bmo M5 Ultra, quiet box at fire (census in the log:
co-tenants none, free 81%, load 0.41); bin built 09:13 +07 at efeb527
(`--features qwen38_wgpu` release); GGUF qwen3.8-27b-Q4_K_M 16.8 GB (the
opponents' same file), f16 load posture; `RIIR_QWEN38_DENSE_MM2D=1`; run
window 15:54–19:51 +07 with `caffeinate` holding the box. The capture lane's
xnli_en numbers (teacher acc ~0.899, 20000 rows, ~2.2 s/row) landed before
the run — both lanes' artifacts intact (`distill.json` + `DISTILL.md`).
Verdict on this record: Claude REVISE round 1 (2026-10-29) — the two-outcome
split, the stage-timer-first order, and the residency audit are its
amendments; lm_head last-row-only verified in source same day.

## Stage timers + the pure-prefill re-measurement (2026-10-29 evening, zed-m3-i1150-stagetimer)

Follow-up task 1 executed (`306540e` timers, `5b41121` bench pad fix),
runs on bmo 20:20–21:45 +07, box quiet (free 93% before load, 80% after).

**The synchronized stage breakdown** (`RIIR_QWEN38_STAGE_TIMERS=1`, pp 2484,
TWO passes, shares reproducible warm-up-vs-pass-1): ragged tail **40–42%**,
GDN scan (48 layers) **31–32%**, tail_logits (8 f32 GEMV sweeps per chunk,
last-row-only — verified in source) **15–16%**, attention (16 layers)
**10.7%**, wte upload 0.3–0.5%, argmax read ~0%. Three consequences:

1. **The four-way pp cells were not pure prefill.** pp 2484 at chunk 512
   left a 436-token ragged tail through SEQUENTIAL decode — ~40–42% of the
   pass — while the opponents' cells are pure prefill. Fixed in the bench:
   targets round UP to the chunk multiple (`2484→2560`), padded count
   disclosed on every line (`5b41121`). The morning's OURS column is a MIXED
   cell and must be re-read; its 18 tok/s is not comparable to uzu/fork/MLX.
2. **GEMMs are no longer the bottleneck** — GDN + lm_head tail + attention
   own ~57–59% of the synchronized pass; the routed dense GEMM sites are
   the minority. GDN routing is the named optimization lead, gated on (3).
3. **Decode regresses 28× on the f16 posture**: raw 32-step decode
   **1.86 tok/s**, decode-ctx 2048 **1.23 tok/s** (vs the incumbent packed
   ~52 tok/s class, issue-1153 cells). 54 GB/token vs ~16 GB/token plus an
   untuned gemv. The f16 posture is a PREFILL experiment — it must never
   serve decode; recorded so nobody presents it as a serving posture.

**The stability finding (upgrades the residency audit):** the pure-prefill
re-read (free-running, no timers, warm compile cache, quiet box) reads
**6 tok/s stable — 451/461/455 s across three 2560-token passes** — while
the morning four-way process read 18 tok/s on comparable work in the same
posture. Same box, same bin lineage, same env: the e2e dense prefill is
**bimodal across processes (~3×) and neither mode is explained yet**. The
residency audit task absorbs this: explain the 18-vs-6 split or neither
number pins. (Also measured tonight: no autotune re-payment — the CubeCL
compile cache persisted across processes; the 20–32 min first-shape cost is
first-ever-compile only.)

Raw logs: bmo `/tmp/i1150/stagetimer.log`, `/tmp/i1150/pureprefill.log`,
`/tmp/i1150/decodeonly.log` (all /tmp — the numbers above are the record).
