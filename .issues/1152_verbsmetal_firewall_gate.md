# Issue 1152: verbs_metal end-to-end is gated by amo's macOS Application Firewall (unsigned-bin SYN drop)

**Status:** OPEN
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

## Acceptance

`verbs_metal --listen` on amo accepts the bmo initiator; Bench 928's "unblocked" addendum records REGISTRATION/GPU→wire/wire→GPU rows; issue closes with the addendum ref.
