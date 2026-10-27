# Issue 1109 — bmo cluster bring-up (TB5 link + tailnet) and the bmo routing of riir-rethink's open issues

**Status:** OPEN — filed 2026-10-22. Blocks [Plan 623](../.plans/623_multi_host_pipeline_inference_lane.md) Phase 0. Parent: [Proposal 042](../.proposals/042_multi_mac_distributed_inference.md). T0/T1/T2/T3 DONE 2026-10-26 — the issue stays OPEN on the owner-physical remainder ONLY (T4 RDMA, Phase 3+ prerequisite).

## On-box adjudication (amo side EXECUTED 2026-10-26 ~07:00+07 — zed-amo-idle-015; the item-9 desk research above was run against the live boxes)

| check (from amo) | result | reads as |
|---|---|---|
| `tailscale status` | bmo **offline, last seen 9h ago, tx 1248 rx 0**; m3-1 offline 7h; m3 offline 14h | bmo down since ~22:00 +07 10-25 — the CURRENT blocker, upstream of both research hypotheses |
| `tailscale ping 100.100.117.44` ×5 | all 5 TIMED OUT, "no reply" | the 10-22 DERP-relay posture (71–92 ms) is STALE — today there is no path at all |
| `tailscale netcheck` (amo) | **UDP: true**, IPv4 yes, IPv6 yes, DERP sin 43.5 ms | amo's tailnet posture HEALTHY — the T2 firewall hypothesis is NOT the live fault on this side |
| port 22 probe (100.100.117.44) | refused/unreachable | same upstream cause — T0 not adjudicable while bmo is down |
| `networksetup -listallnetworkservices` | "Thunderbolt Bridge" EXISTS + ENABLED on amo | amo's T1 half is pre-staged correctly |
| `ifconfig bridge0` | up, RUNNING, link-local `169.254.125.77/16` (unchanged from 10-22) | amo waiting for bmo's side to join |
| `ping6 ff02::1%bridge0` | only amo's own addresses reply (sub-ms) | no TB peer on the bridge — confirms the 10-22 no-peer read |
| `system_profiler SPThunderboltDataType` | ALL ports "**No device connected**" (the Mac Studio entries are amo's own buses) | **bmo is absent from the TB bus: powered off, deep-asleep, or unplugged** — the 10-22 "80 Gb/s negotiated" link no longer exists |

**Verdict: the blocker is UPSTREAM of the desk research's two hypotheses — bmo must be powered on / woken FIRST (owner-physical, supersedes T0 in sequence).** The T1 root-cause work (bmo bridge service) and T2 work (direct-path negotiation) are both unstartable until bmo is on the bus and the tailnet. amo's side is fully verified and pre-staged: when bmo returns, run the research's T1/T2 command sequences AS WRITTEN (they were sound — just not yet adjudicable), re-check `tailscale status` for `active; relay` → direct, then `tailscale ping` for `via <ip>`. The tx 1248 rx 0 tailnet row is the wake-confirmation signal to watch (rx > 0 = bmo answering again).

**Re-check 2026-10-26 ~09:15+07 (zed-amo-idle):** bmo still offline (last seen 12h ago, tailnet row unchanged) — the owner-physical gate stands; no state change since the adjudication.

**Re-check 2026-10-26 ~16:30+07 (zed-amo, readiness probe for Proposal 042):** bmo is BACK and the bridge is LIVE — biggest state jump since the 10-22 baseline:

| check (from amo) | result | reads as |
|---|---|---|
| `ping 192.168.10.2` | 2/2 replies, **0.53 ms avg** | the TB bridge now carries STATIC IPs — amo `192.168.10.1/24` (owner's canonical addressing per AGENTS.md, NOT the 10.0.0.x plan below), bmo `192.168.10.2` answering — **T1's verification goal (sub-ms bridge ping) MET at the network layer** |
| `system_profiler SPThunderboltDataType` | Bus 0: **Device connected, 80 Gb/s, peer `Mac17,15`** (buses 1–5 empty) | the ONE-cable TB5 link is enumerated again — matches the 10-22 record |
| `ssh katopz@192.168.10.2` / `ssh katop@192.168.10.2` | port answers (host key exchanged OK) but **publickey DENIED for both usernames** — BatchMode prompt → password | **T0 first half DONE (Remote Login ON), second half PENDING: amo's `~/.ssh/id_ed25519.pub` is not in bmo's authorized_keys** — every remote bmo step (T3 workspace, incumbent builds on bmo, any 2-box run) stays blocked on this one owner-physical act |
| tailscale (`/Applications/Tailscale.app/.../Tailscale status`) | bmo `active; relay "sin"; offline, last seen 4h ago, tx 1248 rx 0`; direct `ping 100.100.117.44` 100% loss | tailnet fallback lane still DOWN (T2 unstartable); does not block the bridge-primary Phase 0/2 posture |
| `rdma_ctl status` | `disabled` | unchanged — D2 still open, owner-physical (Phase 3+ only) |
| amo GPU | no elephant/recapture/capture processes live | the T0.3 quiet-GPU window may have arrived EARLY (plan note said ~10-27) — re-verify at bench time |

**Sequencing update:** the unblock chain shortens to ONE owner step — add amo's SSH key on bmo (T0 remainder) → T0.1/T0.2/T0.4 + T3 unblock immediately (bridge already live at 192.168.10.x). bmo's RAM remains UNMEASURED (needs the SSH) — the D3 capacity row cannot be chosen until then.

**Claim reshaped 2026-10-26:** the adjudication is recorded above; the issue stays OPEN with the owner-physical remainder (power bmo on → T0 → T1 → T2 → T3). No lease held.

## Measured state (from amo, 2026-10-22)

| fact | value |
|---|---|
| amo | Mac Studio M5 Ultra, 96 GiB (103079215104 B), macOS 27.0.1 (26A434) |
| TB link | ONE cable, peer `Mac17,15`, negotiated **80 Gb/s** (other 5 ports "Up to 120 Gb/s", unconnected) |
| amo `bridge0` | up, member `en2`, **link-local only** `169.254.125.77/16` — no static IP; broadcast ping + `ff02::1` got no peer reply |
| tailnet | amo `100.111.186.76`, bmo `100.100.117.44` — ICMP OK but **71–92 ms RTT** (DERP-relayed, not direct) |
| ssh bmo | **refused** on port 22 (Remote Login off) |
| RDMA | `rdma_ctl status` = **disabled** on amo; `ibv_devices` lists none |

## Tasks

- [x] **T0 (owner, physical, on bmo):** System Settings → General → Sharing → **Remote
  Login ON**; add amo's `~/.ssh/id_ed25519.pub` to bmo's `authorized_keys`.
  **DONE 2026-10-26 ~17:00 — owner added the key; passwordless SSH VERIFIED
  (`ssh katopz@192.168.10.2`). The ONE blocker on all 2-box work is cleared.
  CORRECTION ~18:30: the account's primary RecordName read `seal2` via whoami
  (an alias-shaped legacy name, home always `/Users/katopz`) — owner RENAMED it
  `seal2 → katopz` via `dscl` the same hour; fresh `whoami` = `katopz`. All
  tooling uses `katopz@192.168.10.2` only.**
- [x] **T1 (both boxes):** Thunderbolt Bridge static IPv4 — amo `10.0.0.1/24`, bmo
  `10.0.0.2/24` (`networksetup -setmanual "Thunderbolt Bridge" 10.0.0.X 255.255.255.0`);
  verify `ping 10.0.0.2` sub-ms; record bmo's chip/RAM/macOS build (incumbents' RDMA
  needs IDENTICAL OS builds). **DONE 2026-10-26 — sub-ms bridge ping live at the
  owner's canonical addressing `192.168.10.1/24 ⇄ 192.168.10.2` (NOT the 10.0.0.x
  plan; AGENTS.md canonical). bmo RECORD (via SSH, 17:00): `Mac17,15`, Apple M5
  Ultra, hw.memsize 103079215104 B = 96 GiB (SAME as amo), hw.ncpu 30, macOS
  27.0.1 BuildVersion `26A434` — IDENTICAL build to amo (the RDMA identical-OS
  requirement is MET). bmo `rdma_ctl status` = disabled (same as amo — T4/D2 both
  boxes still pending). bmo bridge0 static 192.168.10.2/24, status active.
  NOTE: bmo has NO cargo/rustc/cmake yet — T3 installs the toolchain.**
- [x] **T2:** Tailscale direct path: `tailscale ping bmo` until `via <ip>` (not DERP);
  70–90 ms relayed RTT makes the tailnet a fallback posture only.
  **DONE 2026-10-26 ~21:4x+07 (zed-amo, read-only diagnosis subagent): the direct
  path is LIVE BOTH WAYS — no owner step needed.** amo→bmo: `pong from bmo
  (100.100.117.44) via 192.168.1.93:41641 in 35ms` (first pong, incl. disco);
  bmo→amo: `pong from amo (100.111.186.76) via 192.168.1.92:41641 in 19ms`.
  `tailscale status` shows bmo `active; direct` (the earlier `offline, tx 1248
  rx 0` degenerate row is GONE — bmo's daemon self-recovered, no config delta
  visible). `netcheck` UDP:true + easy NAT (MappingVariesByDestIP: false) on
  BOTH boxes; the direct path rides the WiFi LAN (192.168.1.x), NOT the TB
  bridge — expected, and fine: the bridge (192.168.10.x) stays the primary
  raw-IP transport, the tailnet fallback is now sub-40 ms direct instead of
  DERP-relayed. (bmo netcheck printed one cosmetic `routerIP/FetchRIB: sysctl:
  cannot allocate memory` line; re-read only if netcheck later fails to report.)
  **TAILNET ARM MEASURED 2026-10-26 ~23:00 ([Bench 901](../.benchmarks/901_plan623_t02_tailnet_arm.md)):
  amo⇄bmo over 100.x direct = 0.025–0.028 GB/s, RTT ~9.4 ms @4B — ~340× bandwidth /
  ~100× RTT worse than the TB5 bridge. VERDICT: tailnet = control-plane/relay-join
  only, NEVER the activation data plane (3 RTTs ≈ 30 ms > the whole 17 ms decode
  step). T0.2 is now FULLY DONE both arms.**
- [x] **T3:** Workspace on bmo: clone the repos Plan 623 needs (`riir-infer`, `katgpt-rs`),
  same `rust-toolchain.toml`; rsync the D3 model artifacts once, BLAKE3-verified.
  **DONE 2026-10-26 ~18:30 (zed-amo):** rustup on bmo with the PINNED toolchain
  (`cargo/rustc 1.98.1` — matches `rust-toolchain.toml` exactly); repos rsynced
  `~/git/{riir-infer,katgpt-rs}` (target/.raw/.env excluded); the 6.7 GB
  `Ternary-Bonsai-2-27B-PQ2_0.gguf` at `~/models/` — **sha256-verified identical
  both boxes** (`3907dc16…2ec1`); amo's BUILT llama.cpp tree shipped to
  `~/git/riir-infer/.raw/llama.cpp` (same OS/arch → binaries run as-is; skips the
  missing-cmake problem). **RPC transport smoke PASSED end-to-end:**
  `ggml-rpc-server` on bmo (:50052, MTL0 79.6 GB free visible) + `llama-cli --rpc`
  on amo → handshake, buffer setup, and set_tensor over the bridge all green; the
  model LOAD then refused: pinned research clone `19ab5a5` cannot parse
  `output.weight` ggml type 142 (PQ2_0) — `invalid ggml type 142. should be in
  [0, 43)`. EXPECTED: the league fork `prismml-llama.cpp` @ `prism-b10787-73bfe6c`
  (`/Users/katopz/git/riir-ai/.raw/prismml-llama.cpp`, PQ2_0 in its loader) is the
  Bonsai-capable incumbent — T0.4's 2-box rows need it built `-DGGML_RPC=ON` on
  BOTH boxes. That is the T0.4 session's first step, recorded there.**
  **T0.4 llama.cpp-RPC arm EXECUTED 2026-10-26 (~22:00–23:00):** fork built on amo
  (`-DGGML_RPC=ON -DGGML_BLAS=OFF` — BLAS ON aborts the server on `RMS_NORM`),
  shipped as a rpath-surgered binary bundle (bmo has NO CLT — no native build, no
  install_name_tool; all load-command fixes happen on amo then ship). Full table +
  traps: [Bench 900](../.benchmarks/900_plan623_t04_llamacpp_rpc_2box.md). Headline:
  **2-box PP over TB5/TCP tg128 48.6 t/s = −20% vs bmo solo 60.9** — the incumbent
  loses single-stream decode on identical boxes; prefill ≈ free (−2.7%); `-sm tensor`
  over RPC unsupported (server crash). MLX/exo arms remain open in T0.4.**
- [x] **T4 (owner, physical — Proposal 042 D2): DONE 2026-10-27** — `rdma_ctl enable`
  from Recovery on BOTH boxes (owner act), `ibv_devices` shows 6× `rdma_en*` per box;
  TB bridge IPs intact post-reboot (bridge + RDMA coexist on the one cable's stack).
  First JACCL measurement: [Bench 905](../.benchmarks/905_plan623_phase3_jaccl_probe.md)
  (all_sum 64 MiB 18.5 GB/s, RTT-class 16-20 µs; p2p send/recv deadlocks on jaccl TOO —
  mlx 0.32.0 2-node p2p broken on both backends, upstream filing GO'd).

## Routing — "can bmo's GPU solve all of riir-rethink/.issues?"

No — **two of seven are GPU/hardware-gated; the rest are gated on owner calls or dead
lanes**, which a second GPU does not move. Per issue (read 2026-10-22):

| rethink issue | blocked on | what bmo does |
|---|---|---|
| 016 encoder class on GPU hosts | owner: "M5-Ultra-gated" serve seating (T5–T6) | **hardware gate MET** — bmo is the candidate serving host; the owner GO is still required |
| 024 Proposal 003 tracker | v1 hosted lane + remote-leg GOAT; re-fits gated on riir-train capacity | hosts `/v1/decide` for the remote-leg GOAT (`RIIR_RETHINK_ESC_REMOTE=http://192.168.10.2:…`, the T1 canonical addressing — an older draft of this row said 10.0.0.2, which was never deployed); re-fits via the deferred train plan |
| 025 v1 billing | M5-Ultra-gated deploy of the hosted lane | hosting only; owner deploy call remains |
| 031 egemma2-ft seating | T4 optional, owner-gated OOD probe | can run T4 (small encoder read) once the owner arms it |
| 032 SubQ Snap beat lane | teacher fine-tune = riir-train Issue 629 (4090-queued); T6 hard owner gate | candidate second node for the teacher lane — rides the deferred train plan (Proposal 042 §Train) |
| 034 NCM multi-depth | AUTO-HOLD (UltraBERT P0 died); needs a third encoder candidate + ESC re-arm | **not GPU-blocked** — bmo does not move it |
| 035 JDI re-bench release | T2 waits on 629; T4 needs the pplx 4090-NF4 tunnel (8793) | can host the pplx rig on MLX 4-bit (riir-reflex Bench 132 adapter exists) — changes the bench posture, so it is a recorded posture change, not a drop-in |

Each row is acted on in its OWN repo/issue; this table is the routing record, not the
work. riir-infer Issue 015 (the audio lane) is also unblocked by M5 hardware (recorded
there 2026-10-22).

## Unblock research (item-9 pass, 2026-10-26 — desk research from the M3 box, NOT executed; the commands below adjudicate on the boxes)

The two technical blockers (T1 no-peer-reply, T2 DERP-only) both have known root-cause classes. Canonical refs: Tailscale's NAT-traversal write-up ([how-nat-traversal-works](https://tailscale.com/blog/how-nat-traversal-works) — live link verified this pass; the kb/1022 and kb/1131 KB URLs served stale unrelated cached content through jina, so verify against live docs at execution) + Apple's Thunderbolt-bridge guide (`support.apple.com/guide/mac-help/connect-two-macs-with-a-thunderbolt-cable-mchl7cb91a48`).

**T1 root cause (most likely): bmo's Thunderbolt Bridge SERVICE is not active.** The measured signature — link negotiated 80 Gb/s but zero reply to broadcast ping AND `ff02::1` — means amo's side is up and bmo's bridge never answers ARP/NDP. On a fresh Mac the TB network service is absent/disabled until created; the cable alone brings the link up but not the network service. Fix on bmo, before T1's `-setmanual`:

```sh
networksetup -listallnetworkservices        # does "Thunderbolt Bridge" exist on bmo?
# absent → create it (interface name from `networksetup -listallhardwareports`, likely "Thunderbolt 2"):
networksetup -createnetworkservice "Thunderbolt Bridge" "Thunderbolt 2"
networksetup -setmanual "Thunderbolt Bridge" 10.0.0.2 255.255.255.0
```

amo mirrors with `10.0.0.1`. If the service exists but shows "Not Connected" in System Settings → Network, selecting it (or `networksetup -setnetworkserviceenabled "Thunderbolt Bridge" on`) activates it. Link-local (169.254.x.x) self-assignment only completes when BOTH bridges are active — the measured link-local-only state on amo is consistent with bmo's side never having joined.

**T2 root cause (most likely): inbound UDP blocked on one side (macOS application firewall), so hole-punching fails and the pair falls back to DERP.** Same-LAN tailnet nodes should go direct in seconds; DERP-relayed means at least one side's WireGuard UDP (default port 41641) never arrives. Diagnosis + fix, on BOTH boxes:

```sh
tailscale debug netcheck      # "UDP: false" or IPv4 false = the smoking gun on that box
tailscale ping 100.100.117.44 # repeat 3–5×; "via DERP" → "via <ip:port>" when direct lands
```

If netcheck shows UDP blocked: System Settings → Network → Firewall → allow Tailscale incoming (or `tailscale set --advertise-exit-node=...` is unrelated — do NOT touch; the fix is purely the firewall row). Amo↔bmo share the TB cable/LAN, so once both firewalls allow UDP the direct path is local-segment and the 71–92 ms DERP RTT collapses to sub-ms — which also makes the tailnet a viable Plan-623 transport fallback (T0.2's tailnet arm becomes worth measuring, not just recording).

**Sequencing note:** T0 (Remote Login + ssh key) remains the owner's physical first step — every remote remediation of T1/T2 needs a shell on bmo. T1 before T2: the TB bridge is the transport Plan 623 Phase 0 actually measures; the tailnet is the fallback posture.
