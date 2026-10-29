# Issue 1152: verbs_metal end-to-end is gated by amo's macOS Application Firewall (unsigned-bin SYN drop)

**Status:** OPEN — mechanism fully measured 2026-10-29 ~00:2xZ (m3 idle lane, read-only ssh probes; no box writes, the amo⇄bmo lane belongs to the live T3.5 session): the posture row below pins amo's exact fw state, the differential is fully explained (Apple-signed builtins auto-allowed + past GUI accepts per-binary-path), and remedy 3 is upgraded to PREFERRED — the plan-623 wire-cell rig ITSELF already runs tail-on-bmo/head-on-amo live (`plan623_wire_cell.sh` L34–49: `--role tail --listen 192.168.10.2:$PORT` on bmo, `--connect` from amo), so the verbs_metal rows can ride the same posture with ZERO fw interaction. The one-line remedies for the owner are unchanged.
**Claimed:** zed-m3-idle1029a, 1793264000, expires 2026-10-31T01:30:00+00:00
  — the remedy-3 bmo-target rows: T3.5 lane (zed-m3-i971-housekeeping) handed the rows over unclaimed 2026-10-29 ~01:1xZ; deliverable = a Bench-928 addendum, issue STAYS OPEN for the owner gate
**Date:** 2026-10-29
**Lane:** Plan 623 T3.2's MTLBuffer leg ([Bench 928](../.benchmarks/928_plan623_t32_rkey_mtu.md)) — the code is landed (`570ee79`, `examples/verbs_metal.rs`, feature `rdma_metal`); the measurement is blocked by a box gate, not the lane.

## Finding

amo's macOS Application Firewall (ENABLED, state=1; blockall OFF, stealth OFF) **silently drops TCP SYNs from the TB bridge when the LISTENING process is an unsigned Rust binary**. Proven by differential on the same port, same listener, same peer (bmo):

| listener on amo:PORT | bmo connect result |
|---|---|
| `nc -l` | connect rc=0, accepted, sub-ms |
| `python3` socket server | connect rc=0, accepted, 0.0005 s |
| `ssh` (system, allowlisted) | connect rc=0 |
| `clang`-built ObjC probe | connect rc=0 |
| `verbs_metal` / `verbs_write` / unsigned Rust bin | **EHOSTUNREACH "No route to host" ×8 retries; listener's `accept` never returns** |

Both Rust processes were healthy mid-stall (sampled): target in `TcpListener::accept` (__accept), initiator in `read_exact` — the kernel holds the SYN. The fw's own app list shows amo's allowlist is per-binary-path (an old `link_probe` build is allowlisted; the probe binaries are not), and the `Application Support` relocation heuristic plus `/Users/Shared` (bmo's squash-sheet: non-home writes invisible to its own session) were both tried and bit.

## Why this is an issue and not a lane fix

`socketfilterfw --add/--unblockapp` requires sudo; the lane holds no password. The Metal-side premise was verified independently (clang/ObjC probe: shared MTLBuffer + blit fill reads back through `contents()`), so the only missing row is the wire test through the unsigned binary.

## Remedies (any one)

1. **Owner (one line):** `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add <bin> --unblockapp <bin>` for the shipped verbs binaries — or disable the fw for bridge0 (owner's box policy call).
2. **Sign the probe binaries** (codesign identity; the windows-build/biometric lanes have the recipe shape) — signed binaries pass the fw unsigned-app heuristic.
3. **Lane-side workaround if another window needs the row before the gate lifts:** run the TARGET role on bmo (amo initiates — the fw gate is amo's inbound direction only; every amo→bmo probe ran clean).

## Firewall posture — MEASURED 2026-10-29 (read-only, unprivileged ssh; amo macOS 27.0.1)

`socketfilterfw --getglobalstate --getblockall --getallowsigned --getstealthmode --listapps` all work WITHOUT sudo over ssh — future lanes can re-verify this posture read-only at any time.

- Firewall ENABLED (state=1) · blockall **disabled** · stealth off — rule-based mode, per-app rule table evaluated.
- **"Automatically allow built-in signed software" ENABLED** — every Apple-signed builtin passes inbound with no rule (the differential explained: `nc`, `ssh`, `/usr/bin/python3`, `ruby` …).
- **"Automatically allow downloaded signed software" ENABLED** — nominal quarantine+signed path; a locally built binary carries no quarantine bit, so this knob does not reach cargo output.
- The app table (16 rows, each `Allow incoming connections`) — **non-Apple binaries DO hold rules**: `link_probe` (`target/release/examples/link_probe`), the uv cpython 3.12, Xcode's Python.app, and `.raw/llama.cpp/build-rpc/bin/ggml-rpc-server`. `codesign -dv` on the allowlisted `link_probe`: `flags=0x20002(adhoc,linker-signed)` — the SAME signature class as cargo output. **An Apple-issued identity is NOT required for the allowlist path**; each row got there via one GUI prompt accept (or sudo) for that exact binary.
- The verbs binaries: NO table row → the fw's default for an unlisted app is a GUI "accept incoming connections?" prompt; in an unattended ssh session the prompt is never answered and the kernel silently drops the SYN (listener's `accept` never returns — exactly the measured stall).

## Remedy refinement (supersedes remedy 2's guess)

- **Remedy 2 as written is wrong in both directions.** An Apple identity is unnecessary (linker-adhoc binaries hold rules fine), and explicit ad-hoc codesign does NOT auto-allow either — the qBittorrent precedent (issue #3885, verified through macOS 14+) shows an explicitly ad-hoc-signed app still prompts ONCE; the signature's only value is making the accept STICKY (stable CDHash — the prompt stops repeating for that binary). The gate lifts with exactly one of:
  1. **One GUI accept per built verbs binary at amo's screen** (amo has a live GUI session): run the listener once, accept the prompt; the rule then holds for re-RUNS of the same bytes. A REBUILD changes the CDHash and re-prompts — accept on the final binary of the measurement session, or
  2. **Owner one-liner** (unchanged): `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add <bin> --unblockapp <bin>` — also per-binary-path; byte changes likely invalidate the row the same way, or
  3. **Remedy 3 — now PREFERRED** (was "lane-side workaround"): the wire-cell harness already proves the tail-on-bmo posture live on this bridge; run `verbs_metal --listen` on bmo and the initiator from amo (bmo has Metal; its inbound accepts amo today — the cells run green). Reuse the rig's readiness pattern: poll the tail's log for the listening line, never a TCP probe (the single-accept EOF trap is documented in the script header). Coordination notes: the T3.5 session owns the amo⇄bmo lane this window, and bmo carries the riir-train 640 T3b capture — a seconds-class buffer probe is negligible load, but pick a between-cells window and stay off their stage binaries (`dist_stage`) — build the probe to a separate target dir.

## Acceptance

`verbs_metal --listen` on amo accepts the bmo initiator; Bench 928's "unblocked" addendum records REGISTRATION/GPU→wire/wire→GPU rows; issue closes with the addendum ref.

## Remedy-3 run result (2026-10-29 ~01:5xZ, the claimed lane — issue STAYS OPEN)

The bmo-target rows ran ([Bench 928 addendum](../.benchmarks/928_plan623_t32_rkey_mtu.md)): **REGISTRATION row MEASURED PASS ×2** (shared-storage MTLBuffer pages registered as an RDMA MR on bmo's M5 Ultra, 105.7/99.25 µs, lkey granted, rkey=0x0 per the provider law) — the leg's primary question is answered. The GPU→wire / wire→GPU byte-identity rows are blocked by a NEW plane-level regression: **the amo initiator's `ibv_modify_qp(RTR)` EINVALs (`mask 0x101181 rc=14`), reproduced byte-for-byte by verbs_ping** — the instrument that landed Bench 927/928's rows fails identically today, bmo's RTR succeeds, neither box rebooted since the working rows: an amo-side provider/link state change, not a code defect and not the firewall. Owner-gated remedies (GUI session on amo / TB re-plug / reboot) now gate BOTH the amo-listener acceptance AND the plane heal; the acceptance clause is unchanged.

## Re-probe (2026-10-29 ~10:47 local, the claimed lane — plane STILL gated)

verbs_ping amo→bmo re-run post-hardening (`d4604b0`/`874330c` — amo checkout at `ae72224` already carries both; the binary used was the pre-existing instrument build, byte-identical for this path): **the regression reproduces byte-for-byte** — amo initiator panics `to RTR: ibv_modify_qp(mask 0x101181) rc=14` while the bmo host reaches RTS cleanly (RTS line in its log) — the differential is exactly as recorded, NOT a GID-table issue (both sides picked gid[0], the working fallback; agrees with the i928 dating probe, `cd0783f`). amo firewall still ENABLED (state=1). Nothing lane-side moved in ~9 h: the owner gate stands; no further probes until a heal signal (owner act or a box reboot) to avoid noise.

## Runway prepped (2026-10-29 ~11:0x local — the re-run needs NO build)

Both boxes' `/tmp/vm_target` verbs binaries rebuilt fresh at `76564d9` (amo 10:55, bmo 11:01; bmo's checkout fast-forwarded `efeb5273` → `76564d9` by fetching from amo's checkout over the bridge — bmo has no GitHub ssh key, provenance disclosed). The plane-heal re-run is now: one `verbs_ping` canary (consequence-1), then the remedy-3 rows directly. Claim stays live (expires 2026-10-31T01:30Z) for that re-run.
