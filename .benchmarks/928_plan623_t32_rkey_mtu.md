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
