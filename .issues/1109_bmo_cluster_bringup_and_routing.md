# Issue 1109 — bmo cluster bring-up (TB5 link + tailnet) and the bmo routing of riir-rethink's open issues

**Status:** OPEN — filed 2026-10-22. Blocks [Plan 623](../.plans/623_multi_host_pipeline_inference_lane.md) Phase 0. Parent: [Proposal 042](../.proposals/042_multi_mac_distributed_inference.md).

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

- [ ] **T0 (owner, physical, on bmo):** System Settings → General → Sharing → **Remote
  Login ON**; add amo's `~/.ssh/id_ed25519.pub` to bmo's `authorized_keys`.
- [ ] **T1 (both boxes):** Thunderbolt Bridge static IPv4 — amo `10.0.0.1/24`, bmo
  `10.0.0.2/24` (`networksetup -setmanual "Thunderbolt Bridge" 10.0.0.X 255.255.255.0`);
  verify `ping 10.0.0.2` sub-ms; record bmo's chip/RAM/macOS build (incumbents' RDMA
  needs IDENTICAL OS builds).
- [ ] **T2:** Tailscale direct path: `tailscale ping bmo` until `via <ip>` (not DERP);
  70–90 ms relayed RTT makes the tailnet a fallback posture only.
- [ ] **T3:** Workspace on bmo: clone the repos Plan 623 needs (`riir-infer`, `katgpt-rs`),
  same `rust-toolchain.toml`; rsync the D3 model artifacts once, BLAKE3-verified.
- [ ] **T4 (owner, physical — Proposal 042 D2):** RDMA: `rdma_ctl enable` from Recovery
  Terminal on BOTH, reboot; `ibv_devices` shows `rdma_en*`. Only needed from Plan 623
  Phase 3; TB Bridge must be OFF on the RDMA port (use a second TB cable for the TCP
  bridge if both postures are wanted at once).

## Routing — "can bmo's GPU solve all of riir-rethink/.issues?"

No — **two of seven are GPU/hardware-gated; the rest are gated on owner calls or dead
lanes**, which a second GPU does not move. Per issue (read 2026-10-22):

| rethink issue | blocked on | what bmo does |
|---|---|---|
| 016 encoder class on GPU hosts | owner: "M5-Ultra-gated" serve seating (T5–T6) | **hardware gate MET** — bmo is the candidate serving host; the owner GO is still required |
| 024 Proposal 003 tracker | v1 hosted lane + remote-leg GOAT; re-fits gated on riir-train capacity | hosts `/v1/decide` for the remote-leg GOAT (`RIIR_RETHINK_ESC_REMOTE=http://10.0.0.2:…`); re-fits via the deferred train plan |
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
