# Bench 929 — plan-623 T3.5: the Phase-3 scoreboard (decode, the T3.4 arm verdict, prefill, the decomposition)

**Date:** 2026-10-29 (00:33–01:58 UTC window)
**Lane:** [Plan 623](../.plans/623_multi_host_pipeline_inference_lane.md) T3.5 / [Issue 1153](../.issues/1153_plan623_t35_scoreboard.md) (file-level claim, 48 h)
**Session:** zed-m3-i971-housekeeping, epoch 1793262200
**Boxes:** amo (`Mac17,5`-class M5 Ultra 96 GiB, HEAD box) ⇄ bmo (`Mac17,15` M5 Ultra 96 GiB, tail) over the TB5 bridge (192.168.10.1⇄192.168.10.2, `bridge0`), amo=HEAD / bmo=TAIL (the firewall-clean direction per [Issue 1152](../.issues/1152_verbsmetal_firewall_gate.md))
**Code:** riir-infer `6b525ed` (the T3.4 async-finish arm + the step-timing instrument) on BOTH boxes; harness commits up to `94bed9f` (harness-only, binaries unchanged after the 06:48 `gpu,bonsai2_hadamard` rebuild on both boxes)
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf`, split 32/64, `--backend gpu`, CTX 4096 (p8192 skipped — harness pins the TAIL's ctx globally; the 512/2048 rows carry the verdict's ≥2-prompt-length requirement)

## PROVENANCE (box state, disclosed)

- **bmo co-tenant, whole window:** the riir-reflex mammoth capture lane resident (serve ~54 GiB MPS + bursty xnli capture driver + `caffeinate -dis`); capture bursts land as bmo load spikes between cells. amo: load 0.9–1.2 at start; amo 15-min load average read 10.09 at 01:58 (the matrix's own tail boots + the sibling lanes' churn), 1-min 3.51 settling.
- bmo memory at start: 37% free (the mammoth resident). All cells foreground-interleaved within each phase (the sequential-drift law: arms minutes apart drift +5–21% on a shared box — ABBA/ABA orders only).
- **The launch-context discriminator (measured, load-bearing for every row here):** cells run in the FOREGROUND of an ssh session. The same script, same binary, same args, same minute: foreground spawns connect; nohup-detached-matrix spawns failed `EHOSTUNREACH` 6/6 while manual probes (foreground AND nohup) connected 4/4 seconds later. Root cause NOT found (macOS per-app local-network/firewall posture of the reparented process tree is the suspect class); the discipline recorded: **plan-623 wire cells run foreground-chunked; a detached matrix run is not a valid instrument on these boxes until the discriminator is explained.**

## Row 1 — single-stream decode (the Proposal-042 G-gate rows)

| row | tok/s | source |
|---|---|---|
| ours 2-box s1 (demoted, the serving posture) | **52.28 / 52.59** (armed/demoted — wash at s1, ±0.6%) | Bench-929 cells s1_p1_a / s1_p0_b |
| ours 1-box oracle (in-run, amo, post-pipeline warm) | **56.29** | the 06:49 G1 flagship cell (same process as the wire row below) |
| llama.cpp RPC 1-box solo (amo/bmo) | 59.1 / 60.9 | Bench 900 |
| llama.cpp RPC 2-box layer-split | **48.6** | Bench 900 |
| mlx-socket PP | −26% class | Bench 904 |
| exo | N/A — cannot load PQ2_0 | Phase-0 note, unchanged |
| MLX jaccl | prose only (16–22 µs pipelined collective vs our 7.7 µs verbs ping-pong RTT, Bench 905); its serving forms are dead (ring deadlock, TP refuted Bench 906) | not re-measured |

**G-gate verdict ("PP ≤ 5% slower than 1-box"): FAIL by 2.1 pt** — ours s1 52.3–52.6 vs oracle 56.29 = **−6.7..−7.1%**. The decomposition (Row 4) localizes the 7% (+1.38 ms/token over the oracle's 17.77): the second hop RTT + the second box's dispatch overhead, not host-side serialization.
**vs the best incumbent:** ours 2-box 52.3 beats llama.cpp-RPC 2-box 48.6 by **+7.6%** at single-stream decode (both on the TB wire class, both layer-split). This is the scoreboard's like-for-like serving row and it flips T2.7's −20%-era reading: the Issue-1141..1146 production-posture rungs landed between the two measurements.

## Row 2 — the T3.4 async-finish arm on the real wire: REFUTED, demote

ABBA ×2 at s=2 (IDS4 prompts, 31 steps, tight spread):

| cell | arm | wall tok/s | decode window |
|---|---|---|---|
| ab_p1_a | armed | 34.91 | 51.24 |
| ab_p0_b | demoted | 53.67 | 99.99 |
| ab_p1_b | armed | 36.10 | 51.38 |
| ab_p0_a | demoted | 53.60 | 99.30 |

**Armed = −48.5% on the decode window** (51.3 vs 99.6 aggregate; per-stream 25.7 vs 50.2), reproducible, ±1% cell spread. Mechanism (read off the drain code): the deferred finish→send drain submits BOTH chains before ANY activation leaves the head — the tail idles through the head's second chain and the head idles through the tail's two forwards; the pass is fully serial. The incumbent inline cadence pipelines the two GPUs (the head computes stream B's head-half under the tail's stream-A forward — the Bench-913 win). **The T3.4 landing's +5% (M3 loopback rig) was a single-queue artifact: on one shared queue there is no cross-GPU pipeline to forfeit; the wire is the arm's refutation posture.** s=1 is a wash (the arm engages — every pass is all-feed — but has nothing to overlap and loses nothing).
**Action:** unit issue files the DEMOTION (the wire serving default returns to the inline cadence; the arm stays reachable opt-in for the loopback rig where it won). The all-feed gate does not save it — at s≥2 all-feed IS the steady state and it is exactly where the loss lives.

## Row 3 — prefill (the verdict fold: long-prompt warm cells, chunk re-cell)

Every cell `--full-check`; the in-run oracle prefill is the comparator (same process; warmth per row: the pipeline prefill runs first-touch per cell, the oracle after it — the JIT amortizes over the long prompt; deltas quoted are wall ratios inside one cell, cross-cell deltas disclosed as ±10% under the co-tenant load).

**2-box prefill wall, 2 streams, vs the in-run 1-box oracle prefill (2048-token prompt ×2):**

| chunk | 2-box wall s0/s1 | oracle prefill | aggregate ratio (4096/W ÷ 2048/W_o) |
|---|---|---|---|
| 128 | 3.90 / 4.05 s | 3.57 s | **1.76×** |
| 512 | 3.01 / 3.44 · 2.99 / 3.42 | 2.85 / 2.83 | **1.65–1.66×** |
| 2048 | 3.07 / 4.50 · 3.09 / 4.54 | 2.80 / 2.80 | **1.24×** |

**512-token prompt ×2:** c128 → 1.52×; c512 → **1.11×** (one chunk per stream = zero chunk-level pipelining; the two streams pipeline each other only).

**The falsifiable overlap bar (verdict round 1): PASSES at the incumbent chunk.** c128 = 1.76× aggregate ≈ 88% of the 2× ideal (bubble + per-chunk head-side gaps = the residual 12%); the single-chunk postures (c2048@p2048, c512@p512) land 1.1–1.25× — exactly the serial prediction, which is the bar's control arm. The overlap EXISTS; it is not free.

**The chunk re-cell verdict: exo's bubble heuristic does NOT transfer — the incumbent 128 stands.** exo pre-sizes `prefill_step_size // min(4, world_size)` = 2048 @ 2 ranks to amortize per-chunk overhead over the bubble; on OUR activation pipeline the boundary frame is cheap (2.6 MB @ 128 pos over an 8.6 GB/s bridge) and finer chunks pipeline BETTER (1.76× at c128 vs 1.24× at c2048). No default change (the deltas are robust in direction but ≤10% in magnitude at the incumbent posture; a promotion needs its own unit per the discipline).

## Row 4 — the decode-step decomposition (RIIR_DIST_STEP_TIMING, this unit's instrument)

s=2 demoted steady state, 64-step window (head and tail agree to 0.4 ms on the wall):

| bucket | head | tail |
|---|---|---|
| per-step wall | — | 19.05 ms/token (1181 ms / 62 tokens) |
| submit | ≈0 (<0.0005 ms) | — |
| finish (readback wait) | ≈0 | — |
| send (activation 20 KB) | ≈0 | 0.007 ms |
| recv (link wait) | **4.855 ms** | 3.508 ms |
| fwd (range forward + logits readback) | — | **14.826 ms** |

Head-local work is <1% of the budget; the tail's forward + logits readback is **78%**; the head's recv-wait (the head-half chain hidden under the tail's work — the pipelining working as designed) is 25%; the two sides sum to the wall. **Both T3.4 residues are DECLINED on the numbers:** the literal comm-THREAD split moves a <1% bucket; the tail logits async-readback arm chases ≤3% (the M3 idle readback probe class, ~0.35–0.4 ms inside the 14.8 ms fwd). Prefill-side tail decomposition (pfwd_s2): tail chunk compute ≈ 557 ms of a ~950 ms/stream head wall — the prefill residual lives in the head-side per-chunk cadence, the follow-up refinement axis if prefill ever matters enough to chase it.

## Row 5 — the incumbent adjudication (T3.5's literal ask)

- **llama.cpp "RPC-RDMA": N/A** — llama.cpp ships TCP-only RPC (no RDMA transport upstream; `transport-apple.cpp` is Apple-RDMA for the *client* leg but the 2-box serving row measured is TCP). Its best row stands: Bench 900's TCP layer-split 48.6 tok/s — which we beat by +7.6% (Row 1).
- **exo: N/A** — cannot load PQ2_0 (Phase-0). Its prefill DESIGN informed this bench (below).
- **MLX jaccl: stays prose.** Bench 905's 16–22 µs pipelined collective rate is not a serving row; its serving forms are measured-dead (ring deadlock at 2 nodes; TP-2 refuted at −69/−64%, Bench 906). Our verbs RTT (7.7 µs, Bench 927) remains a transport substrate result, not a serving row — the TCP serving rows above are the scoreboard's serving truth, and a verbs-`Transport` serving swap files WITH the measured gap only if a decomposition shows the link leg binding (Row 4 shows it does not, at this split/box class).

## The exo comparison (source-level, corrected per the Claude verdict round 1)

Read at the research pin `bc412e2` (`.raw/exo`, rm at lane end): `generate.py:163-278` + `auto_parallel.py:67-198`.

- exo's 2-rank pipeline prefill is a systolic schedule with **placeholder iterations that run NO model compute** (`generate.py:236-247` — the leading/trailing loops call only the progress callback; MLX collectives need matched send/recv per rank). **Total prefill compute is IDENTICAL to ours** (each rank: full prompt × own layer shard) — the earlier "ours pays half exo's compute" claim was wrong and is retracted; the honest difference is control/cadence: their blocking `recv_like` + lockstep slots vs our async chunk streaming. Same schedule, same (N+R−1)×chunk-time shape.
- Their chunk divisor (`generate.py:194`) is a bubble heuristic that does not transfer to our wire (Row 3: finer chunks win here).
- Decode: their per-token `all_gather` to every rank vs our routed TokenStream — a DECODE-only claim (T2.7-backed), not carried into the prefill comparison.
- `remote_prefill.py` (disaggregated prefill, msgpack KV chunks over TCP) = prior art for a future prefill/decode-split lane; out of scope here.

## Harness lessons paid for this session (each measured)

1. `set -u` on macOS bash 3.2 fired FOUR unset-expansion deaths (`I623_*_ENV`, `env_prefix`, empty `extra[@]`) — every expansion in a `set -u` harness needs `${VAR:-}` / `${arr[@]+...}`; the matrix burned three full passes on one-line deaths before the local end-to-end test discipline stuck.
2. Stale per-cell tail logs satisfy the readiness poll instantly → head connect-refused against a still-loading tail. Clear the log before boot (harness now does).
3. A stale listener holds the port and cascades every following cell — pre-kill per cell (harness now does).
4. macOS `seq -s,` emits a TRAILING comma → `--prompt` parse error "empty string" from the trailing empty element. Strip it at generation.
5. The launch-context discriminator (PROVENANCE): foreground cells are the only valid posture until explained. *(Superseded 2026-10-29 by the addendum below: the class was refuted on re-test; foreground stays the discipline for the sequential-drift law.)*

## Verdicts (one line each)

- Scoreboard decode row: **ours 2-box 52.3 beats llama.cpp-RPC 2-box 48.6 (+7.6%)**; the ≤5% PP-overhead gate **FAILS at −7.1%** (honest fail, localized by the decomposition).
- T3.4 arm: **wire-refuted (−48.5% window), demote filed.**
- Prefill: overlap bar **PASSES at c128 (1.76×)**; chunk default **stands**; exo's chunk heuristic **does not transfer**.
- Residues: comm-THREAD split **DECLINED** (<1%); tail-logits arm **DECLINED** (≤3%).
- RDMA serving swap: **not justified by this decomposition** (the link leg is not the binding cost at this split).

## Addendum (2026-10-29, ~10:00 +07, zed-m3-idle1029b) — the launch-context discriminator RE-TESTED: class refuted as deterministic; mechanism unproven; harness hardened

The PROVENANCE discriminator ("detached-matrix spawns EHOSTUNREACH 6/6 while foreground connects") was re-tested on demand the next morning, every leg, both boxes:

| probe | posture | result |
|---|---|---|
| amo detached shell → bmo TB:22 (ssh + raw TCP + prov-style ssh, the exact matrix launch shape `ssh amo "nohup … &"`) | fully reparented | **all errno 0, sub-ms** |
| amo detached **dist_stage head** (adhoc/linker-signed, the incident binary), full boot → connect 192.168.10.2:17632, no listener | fully reparented | **ECONNREFUSED (os error 61)** — path fine, correct no-listener errno |
| amo foreground dist_stage (unchanged control) | ssh session | connects (the 07:57+ rerun, all cells green) |
| m3 detached → tailscale IPs (100.111.186.76 / 100.100.117.44):22 | fully reparented | errno 0 — after a **6 s cold first-touch** (timeout), instant (0.07 s) on retry |
| amo provenance polls DURING the incident window (07:17–07:39 +07, in `~/git/riir-infer/.wt/i623_t35_matrix/provenance.log`) | ssh from m3 | **all 9 polls succeeded** — the TB link was up the whole window |

Findings, in strength order:

1. **The launch-context class is REFUTED as a deterministic posture property.** Detached shells, detached adhoc-signed dist_stage, and detached ssh all reach the TB subnet and the tailscale paths today. A detached matrix run is not invalid *by construction*.
2. **The errno signature belongs to the issue-1152 class.** amo's Application Firewall is ON (State = 1) and `dist_stage` is adhoc/linker-signed; issue 1152 measured EXACTLY `EHOSTUNREACH "No route to host"` from specific adhoc binaries on amo→bmo while nc/python3/ssh connect — a network-posture SYN drop wearing a routing errno. The discriminator is most plausibly the same dropper, context-attribution-dependent (a detached tree attributes differently than a session), but this is INFERENCE, not measurement — see (4).
3. **A cold-path failure mode is measured and now healed in the harness**: a first-touch connect on an idle path can stall seconds (m3→tailscale: 6 s timeout, then 0.07 s). `plan623_wire_cell.sh`'s tail-spawn ssh was single-shot with no ConnectTimeout — one cold instant killed the cell and the readiness loop could not heal a spawn that never delivered. The harness now carries `ConnectTimeout=10` on the delivery-critical ssh legs and retries the spawn leg up to 3× on dead-ssh + EMPTY-log (the never-delivered signature); a real boot failure (log has content) is surfaced immediately, never retried (smoke-tested with a bogus bin path: surfaces in seconds, no retry spin). The retry is on the SSH leg only — no model load happens on a failed spawn, so a retry costs seconds.
4. **The original incident is honest-unknown.** The failing run's artifacts are destroyed: the matrix `cells.log` never survived (absent), the failing head logs were overwritten by the 07:57+ foreground rerun (same cell names), no tail logs from the window survive on bmo, and `log show` on both boxes has no sshd visibility. The binary has been rebuilt twice since. The "macOS per-app local-network posture of the reparented tree" suspect in the original record stands as the named-but-unproven class, now linked to 1152.
5. **The discipline stands, with its reason corrected**: plan-623 wire cells stay foreground-chunked — for the sequential-drift law (the real perf reason, unchanged), and because a detached matrix requires a SIGNED binary to be trustworthy on these boxes (issue 1152's owner gate). A detached run may be attempted after dist_stage is signed; the hardened spawn retry makes transient cold-path failures self-heal either way.
