# Bench 928 — Plan 623 T3.2 rkey semantics + T3.3 MTU override ladder (amo⇄bmo TB bridge)

**Date:** 2026-10-29
**Lane:** Plan 623 Phase 3 — [T3.2's open half (rkey semantics + the RDMA-WRITE attempt)](../.plans/623_multi_host_pipeline_inference_lane.md) and T3.3's remaining legs (the `--mtu` override ladder + the posted-recv budget cliff vs MTU). Instruments landed in `570ee79` (`verbs_write` + `verbs_metal` + the RC/WRITE module surface).
**Provenance:** amo (M5 Ultra 96 GiB, load 0.8–1.4, no GPU work) ⇄ bmo (M5 Ultra 96 GiB, load 0.35–0.44; **bmo's GPU carried co-tenant work the whole window — the flash_cmma bench (`/tmp/i809m5`) + the mammoth teacher capture (MPS, bmo-local loopback; no capture traffic crosses the TB wire)** — CPU+wire rows unaffected, and the wired-memory posture is itself one of the findings below). Same `rdma_en2` port-1 posture as Bench 927 (active MTU 4096, max_msg 16 MiB). Targets/initiators ran fresh processes per cell; ports 4813–4820.

## T3.3 — the MTU override ladder (the plan's remaining legs)

`--mtu` override engages cleanly: the QP walks RTR/RTS at the negotiated enum (path_mtu=1024/2048 confirmed in the RTS line at both postures).

| row | MTU 4096 | MTU 2048 | MTU 1024 |
|---|---|---|---|
| RTT 64 B inline, p50 (dedicated cells) | 6.75–8.58 µs | 9.58 µs | 10.08 µs |
| RTT 64 B inline, p50 (interleaved cells, same MTUs) | 7.3–10.1 µs | 7.0–9.5 µs | 7.4–7.6 µs |
| BW 128 KiB stripes ×24 (63 MiB) | **9.342** GiB/s | **9.337** | **9.294** |
| BW 1 MiB stripes ×3 (63 MiB) | **9.229** GiB/s | **9.295** | **9.277** |

- **The override is performance-invisible.** −0.5% at MTU 1024 (both stripe shapes) — inside run noise. No reason to ever leave the active 4096; no cliff interaction.
- **RTT is MTU-insensitive** — a 64 B inline message is one packet at every MTU. Round 1's dedicated cells showed a monotone "MTU trend" (6.75 → 9.58 → 10.08); the repeat cells at FIXED MTU 4096 read 8.58 and 6.08 µs — the trend was per-run variance/order, **refuted by repetition**, not carried.

## T3.3 — the posted-recv budget cliff: MTU-INDEPENDENT, BOX-LOAD-DEPENDENT

The cliff bisect drove the host's pre-posted receive ring (`window × stripe`) across the boundary:

| shape (posted bytes) | MTU 4096 | MTU 2048 | MTU 1024 |
|---|---|---|---|
| 128 K×24 = 3.0 MB | PASS | PASS | PASS |
| 1 M×3 = 3.0 MB | PASS | PASS | PASS |
| 1 M×5 = 5.0 MB | FAIL −12 | FAIL −12 | FAIL −12 |
| 256 K×16 = 4.0 MB (927's PASS control) | **FAIL −12** | **FAIL −12** | **FAIL −12** |
| 256 K×13 = 3.25 MB | PASS | — | — |
| 256 K×14 = 3.5 MB | PASS | — | — |
| 128 K×28 = 3.5 MB / 28 WRs | PASS | — | — |
| 256 K×15 = 3.75 MB | PASS | — | — |

- **MTU-independence is clean**: identical outcomes at every MTU — the budget is not MTU-shaped.
- **The edge MOVED vs Bench 927** (same day, same binary lineage): 927 read 4 MB PASS / 5 MB FAIL; this window reads **3.75 MB PASS / 4.0 MB FAIL** — including a FRESH-quiescence 4 MB probe (8 s idle before the cell), which rules out process-churn leakage. The resident-load delta is the attributed cause: bmo held **4,252,674 pages wired (~66+ GB)** under the flash_cmma bench + mammoth capture, vs 927's quieter posture. Inference, labeled: same code, same boxes, different resident load, different cliff — mechanism (the provider's posted-recv pool sizing) unverified.
- **Bytes-shaped at the edge**: 3.5 MB over 28 WRs passes; 4.0 MB over 16 WRs fails — no per-WR-count wall at the edge.
- **THE LAW (corrects Bench 927's "the recv ring stays ≤4 MB"): never hard-code 4 MB as safe.** Size the recv ring ≤3 MB (passing in both measured postures with margin), or runtime-probe the budget before sizing a ring — the cliff moves with co-tenant wired memory. `ibv_post_recv` rc=-12 (ENOMEM) remains the loud failure signature.

## T3.2 — rkey semantics: CLOSED NEGATIVE at the provider

`verbs_write` (amo initiator ⇄ bmo target, port 4820), three QP postures:

1. **RC is refused at CREATE.** `ibv_create_qp(type 2)` returns NULL on both boxes — the TB provider does not build the RC type at all. RDMA READ is unreachable at any attr mask (the RTS TIMEOUT fallback in `Qp::to_rts` is kept for providers that accept the type; here it never fires).
2. **The rkey grant matrix — every combination reads rkey 0x0.** LW; LW|RR; LW|RW; LW|RW|RR (full); even the spec-invalid REMOTE_WRITE-only and REMOTE_READ-only registrations are ACCEPTED (not refused) — and every one of them grants lkey 0x01000000, **rkey 0x00000000**. The provider never emits a remote-access key in any posture.
3. **UC RDMA WRITE stalls in silence.** With the auto-walk's UC QP up (RTS at MTU 4096), the initiator posted the N-op RDMA-WRITE ladder at the target's (raddr, rkey=0x0): the first `post_send(RDMA_WRITE)` is ACCEPTED (rc=0) and the first `poll_cq` **times out at 10 s — no completion, no error WC**. The same QP's SEND ops complete normally around it (the bring-up endpoint exchange rode it). The provider accepts the WRITE opcode and never advances it: refused-by-silence.

**Verdict: the zero-copy one-way shape (RDMA WRITE) is unavailable in this provider posture. SEND/RECV stays the wire shape; the heap bounce per side is the measured cost. T3.2's zero-copy claim closes negative at the transport layer — not a defect in the lane.** (Consistent with the UC transport class: UC has no RDMA-READ responder by spec, and this provider appears to wire neither remote-access opcode despite accepting the WR.)

### The verbs_metal residue (MTLBuffer leg) — code-complete, blocked on an amo-local gate

`verbs_metal` (feature `rdma_metal`, metal 0.31 — one workspace version, BOUNDARY.md row) implements the leg: shared-storage `MTLBuffer` → `reg_mr(contents(), LW|RW|RR)` → GPU→wire (inline MSL `fill_bytes` compute fill, SEND, initiator verifies) → wire→GPU (pattern SENDed into the registered buffer, inline `sum_strided` kernel vs CPU sum). Code builds clippy-clean at both postures; 29 lib tests + 6/6 layout pins green.

**Blocked measuring it end-to-end by an amo-local macOS Application Firewall gate, outside the lane's reach:** the firewall is ENABLED on amo (state=1, blockall off, stealth off) and **silently drops the initiator's TCP SYNs when the LISTENING binary is an unsigned Rust executable** — while the identical port serves `nc`, `python3`, `ssh`, and `clang`-built binaries to the same peer (all proven connect+accept, rc=0, sub-ms). Both processes verified healthy mid-stall: target in `TcpListener::accept` (__accept), initiator in `read_exact` on the endpoint exchange — the connection never leaves the kernel. The `Application Support` relocation (the fw's approved-apps heuristic) and `/Users/Shared` (bmo's squash-sheet: every non-home write invisible to its own session) both bit; `sudo socketfilterfw` needs a password (not held).

**Unblock (owner, one line):** `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add <verbs_metal> --unblockapp <verbs_metal>` (or codesign the binary). The Metal-side premise was verified independently on amo meanwhile: a clang/ObjC probe created the shared MTLBuffer, blit-filled 0xAB, and read it back through `contents()` (`first=ab`) — pages-are-pages holds; the registration path is expected-accept the moment the wire reaches it.

## What this buys the plan

- T3.3 CLOSED: no MTU tuning lever exists (neutral everywhere) — the ladder was worth running to prove the negative; the recv-ring sizing law is now load-aware (≤3 MB or runtime probe), which matters for the Phase-2 activation-frame rings as much as the probes.
- T3.2's rkey half CLOSED: RDMA WRITE/READ are off the table on this provider — the Phase-3 design keeps SEND/RECV and prices the bounce as fixed overhead (already inside the 7.7 µs RTT / 9.3 GiB/s rows).
- T3.2's MTLBuffer half: one owner command from measurement; the lane is otherwise out of blockers.

## Addendum (2026-10-29 ~01:5xZ, zed-m3-idle1029a) — the remedy-3 bmo-target rows ran; REGISTRATION MEASURED + PASS; the byte-identity rows hit a NEW plane-level RTR regression

Posture per [issue 1152](../.issues/1152_verbsmetal_firewall_gate.md) remedy 3 (the T3.5 lane handed the rows over unclaimed after Bench 929; claim live on the issue). amo = initiator, **bmo = target/listener** (Metal side) — zero firewall interaction (amo initiates outbound only; bmo accepted the unsigned listener inbound first try, the dist_stage posture). Fresh builds both sides at `efeb527` content, `/tmp/vm_target` (kept for the re-run), release, `--features rdma_metal`; driver = the plan623_wire_cell.sh shape (ssh-held foreground listener, log-polled readiness, never a TCP probe); key minted per session; ports 4821/4822.

**Row 1 — REGISTRATION (the leg's primary question): MEASURED, PASS ×2.** `ibv_reg_mr` over a shared-storage `MTLBuffer.contents()` on bmo's M5 Ultra: **ACCEPTED** — run 1: `addr=0x10d2e0000 (contents=0x10d2e0000, same=true) len=1052672 lkey=0x01000000 rkey=0x00000000 in 105.708µs`; run 2 (gid pinned 0): `99.25µs`, same shape. Pages-are-pages now holds on the WIRE path too, not just the clang probe. rkey=0x0 re-confirms Bench 928's provider law (no remote keys under any access flags — SEND/RECV stays the wire shape).

**Rows 2–3 — GPU→wire / wire→GPU: BLOCKED, and NOT by the firewall — by a plane-level RTR regression that post-dates Bench 928.** Both runs failed identically: the bmo target completes its full bring-up (RTR→RTS, UC QP in RTS, path_mtu=4096, inline_cap=224; GPU fill committed 7.5–8.3 ms) and posts the SEND — which then times out (`poll_cq: completion timeout`, `verbs_metal.rs:388`) because the amo initiator never reached RTR: `ibv_modify_qp(mask 0x101181) rc=14` (EINVAL) at `verbs_metal.rs:160`.

**The regression is the PLANE, not the example — verbs_ping reproduces it byte-for-byte.** The known-good instrument (the binary that landed Bench 927/928's RTT/bandwidth rows) fails the SAME way today: endpoint exchange clean (MAC'd), caps granted identically (inline=224, send_wr=256), both GIDs well-formed and exchanged (`amo gid[0]=a638:196d:dbd9:457f:9af1:2f6e:68d7:1f9d`, `bmo gid[0]=bfc4:0173:c846:4fcf:ab96:60f0:3605:23ab`, both rdma_en2 port 1, TB link, MTU 4096 agreement) — then amo's `to_rtr` EINVALs (`verbs_ping.rs:257`, same mask, same rc) while **bmo's own RTR with amo's endpoint succeeds** (it reached RTS both examples, both runs). Asymmetry is per-BOX, not per-role: everything amo's provider accepts, bmo's accepts; the reverse direction chokes at RTR.

**Not the GID pick (tested):** `--gid-index 0` pinned on both sides — identical failure (the first run's unpicked `pick_gid(vp, None)` `any_mapped` theory died with the pin; verbs_ping passes `peer_v4` anyway and still fails).

**Not a reboot:** amo booted 10-27 17:31:35, bmo 10-27 17:44:04 — neither restarted since before Bench 928's working rows. The amo-side provider/link state changed under them (TB controller state, GID-table re-derivation, driver-internal cache — no further isolation available without GUI/owner access; `rdma_en2` query surfaces all ACTIVE/healthy).

**Consequences:** (1) Bench 928/929's landed rows remain valid — they ran while the plane was healthy, and today's failure is a NEW state, disclosed here; any re-run of those cells must re-verify the plane first (one verbs_ping RTT row is the cheapest canary). (2) The verbs_metal byte-identity rows wait on the amo-side plane healing (owner GUI session / TB unplug-replug / reboot are the candidate remeadies, owner-gated). (3) Issue 1152 STAYS OPEN — the amo-listener acceptance posture is unchanged, and the plane regression is one more datum for the owner's gate. (4) `/tmp/vm_target` kept on BOTH boxes (the rebuild is ~12 s warm; the binaries are 606 KB, built from `efeb527` content) — the re-run after the plane heals reuses them. `(bmo also now carries cmake via homebrew — the sentencepiece dep needed it; the build PATH needs /opt/homebrew/bin prepended over ssh.)`

**Hardening landed (2026-10-29, zed-m3-idle1029c, this commit):** the L75-noted `pick_gid(vp, None)` gap is CLOSED — verbs_metal's `bring_up` now derives `peer_v4` from the TCP peer and passes it, making the GID pick byte-identical to verbs_ping's differential posture (same-link match first; on TB's unmapped GIDs the behavior is UNCHANGED — index-0 fallback both ways, which is why all landed rows stand). Consequence for (4): the kept `/tmp/vm_target` binaries pre-date this — the plane-heal re-run must REBUILD (~12 s warm) to carry it, or disclose the old posture.

**Runway prepped (2026-10-29 ~11:0x local, zed-m3-i623-verbs-cont):** the consequence-(4) rebuild is DISCHARGED on BOTH boxes — amo and bmo checkouts at `76564d9` (bmo fast-forwarded from amo's checkout over the bridge: bmo has no GitHub ssh key, so it fetched `katopz@192.168.10.1:…` — disclosed provenance path), `/tmp/vm_target` verbs binaries rebuilt fresh (amo 10:55, bmo 11:01) at `rdma,rdma_metal`. The post-heal re-run needs NO build; the verbs_ping plane canary rides consequence (1).

**GID-table dating note (2026-10-29 ~10:30Z, zed-m3-idle1029c, read-only `ibv_devinfo -v -d rdma_en2` both boxes):** the GID tables are byte-STABLE through the regression window — amo `GID[0]=a638:196d:dbd9:457f:9af1:2f6e:68d7:1f9d` and bmo `GID[0]=bfc4:0173:c846:4fcf:ab96:60f0:3605:23ab` read IDENTICAL to the failing-era values (the ~06:00–09:30 verbs runs above), with `GID[1]` the `fe80::` link-local form on both (consistent with Bench 927's not-IPv4-mapped law; also note `pick_gid`'s v4-mapped match arm can never fire on this transport — the hardening above is posture-alignment, not a behavior change here, re-confirmed). The rederivation-between-working-era-and-06:00 half is UNDATEABLE: no verbs-era logs survive on either box's /tmp (checked — bmo's `/tmp/i623_wire_cells` holds only the 08:00+ T3.5 dist_stage tails, amo's `/tmp/plan623-t3` is a build dir; Bench 927's working-era runs predate every surviving verbs artifact), and Bench 927's record carries the port facts but no GID values. Net narrowing: port state PORT_ACTIVE (dig lane, ~10:00) + IP layer healthy continuously (dig lane) + GID table stable through the window (this probe) ⇒ the amo-side change carries NO observable port/GID/link/IP signature — the candidate surface is driver-internal provider state (attribute validation / QP-state machinery), which the owner's reboot resets and the post-heal verbs_ping canary then re-prices.

## Addendum 2 (2026-11-02, zed-m3-i1152-heal1102) — BOTH blockers cleared; the MTLBuffer leg CLOSED both directions on macOS 27.2

**The heal signals fired and the owner lifted the gate:** amo self-shutdown 11-01 and was power-cycled ON SITE 11-02 ~10:1x (the issue's named heal signal — the RTR EINVAL died with the old provider state); both boxes then took macOS **27.0.1 → 27.2** (Build 26B123) and rebooted again; the owner ran the `socketfilterfw --add/--unblockapp` lines for both verbs binaries (rows 19/20 in amo's app table — they SURVIVED the OS update and matched the rebuilt binaries' linker-adhoc CDHash class; no re-add needed after the second rebuild). bmo's firewall stays DISABLED (state=0, unchanged).

**Posture:** both binaries rebuilt from `ebe0b11` content (release, `--features rdma,rdma_metal`, `CARGO_TARGET_DIR=/tmp/vm_target`), built on m3 and scp'd to BOTH boxes (bmo's checkout is sibling WIP on branch `timer-build` — untouched; amo's checkout is the shared builder checkout at `77ed153`, also untouched). amo = `192.168.10.1`, bmo = `192.168.10.2`, `rdma_en2` port 1, active_mtu=5 (4096), key minted per session, ports 4810/4821.

**Rows (all MEASURED PASS on macOS 27.2):**

| row | bmo target (remedy 3) | amo target (the issue's acceptance clause) |
|---|---|---|
| REGISTRATION (reg_mr over `MTLBuffer.contents()`) | **ACCEPTED 216.9 µs**, `addr==contents`, lkey=0x01000000, rkey=0x0 | **ACCEPTED 80.7 µs**, same shape |
| GPU→wire (fill kernel → SEND → initiator verifies) | fill 3.745 ms, SEND 571.6 µs, **every byte == 0xc7 PASS** | fill 3.625 ms, SEND 551.5 µs, **every byte == 0xc7 PASS** |
| wire→GPU (pattern SEND → `sum_strided` kernel vs CPU sum) | 173.8 µs, **gpu_sum == cpu_sum == 133693440 PASS** | 172.9 µs, **PASS** (same sums) |
| VERDICT | **CLOSED — both directions byte-identical** | **CLOSED** (fw-gated listener accepted first try) |

Plane canary (verbs_ping, default posture): RTT n=1800 p50 **6.375 µs** p99 13.9 µs (27.0.1 league: 6.6–7.7 µs — slightly better); BW 64 MiB **9.380 GiB/s** one-way (vs 9.342 on 27.0.1 — inside noise); clean exits both sides.

**NEW provider finding — the 27.2 recv-window cap (fix landed `ebe0b11`):** the first post-update canary ENOMEMd (`ibv_post_recv rc=-12`) on bmo at the bulk window's FIRST 256 KiB post, deterministically ×2, on a FRESH boot — 27.2's provider caps the recv window at **4–7 × 256 KiB SGEs** (bisected: window 1/2/4 green at 8.54–9.39 GiB/s; window 8/16/32 rc=-12; and the cap follows DEPTH × SGE-len, not total bytes — `stripe 65536 × window 32` (same 2 MiB) runs **green at 9.388 GiB/s**). `verbs_ping`'s default window demoted **32 → 4** (perf-identical: 9.380 vs 9.388 GiB/s), flag retained for re-widening after a future re-measure; the ≥8×256 KiB failure shape supersedes nothing in the L39 law (≤3 MB total) — it is an ADDITIONAL, tighter, per-SGE-length constraint on 27.2.

**Latent verbs_metal protocol bug fixed (`ebe0b11`, would have eaten the wire→GPU rows):** the target posted only the 16 B marker recv — the initiator's 1 MiB pattern SEND (posted before the marker on the ordered SQ) would have truncated against it (UC RQ matches FIFO → QP error) and the marker behind it dropped; first execution ever (the leg was blocked since landing at `570ee79`). Fix: GUARD-sized pattern recv (wr 21) posted BEFORE the marker recv + wire-order drain of both CQEs. Worked first try.

**Control-plane note:** bmo's TB-bridge ssh flapped during setup (worked at 10:31, timed out 10:33 ×2, verbose-ok 10:35, timed out again 10:36; ICMP clean throughout at 0.6 ms; bmo sshd listening, fw disabled) — self-healed and stable since; the lane moved bmo control to Tailscale (stable all session). Not diagnosed further; worth remembering as a bmo sshd flakiness class on the bridge path.

**Consequences:** (1) T3.2's MTLBuffer leg **CLOSED** — registration + both byte-identity directions measured PASS on the amo⇄bmo TB bridge at macOS 27.2, the acceptance clause of [issue 1152](../.issues/1152_verbsmetal_firewall_gate.md) MET (amo listener accepted the bmo initiator with the fw rows in place) — issue closes with this addendum. (2) Phase 3's wire laws unchanged and re-confirmed on 27.2: SEND/RECV only (rkey=0x0 everywhere), MTU 4096 active, RTT ~6.4 µs / BW ~9.4 GiB/s envelope. (3) `verbs_ping` recv-window default is now 4 — any future lane sizing rings should re-read the L39 law + this addendum's 27.2 cap.
