# Bench 935 — Issue 1146: the 2-box TB5 prefill width leg (amo⇄bmo, ABAB) — the wire curve crosses in (1024, 2048)

**Status:** RECORD — the 2-box leg is RUN and the answer is **length-conditional**: w128 owns ≤1k (small-frame pipeline overlap), w512 owns ≥2k (wide-frame GEMM). The `dist_stage` GPU default is now **prompt-length-conditional (128 below 2048, 512 at/above)** — a flat flip would have regressed the ≤1k class by up to 25%. Functional check both branches on the live wire PASS.
**Date:** 2026-11-02 18:15–18:23 +07
**Session:** zed-amo-1146-finish, 1793636000 (claim `e65acdb`)
**Issue:** `.issues/1146_prefill_throughput_gap_1box.md` (closed + removed same day — the arc in `HISTORY.md`)
**Box:** head = amo (M5 Ultra, 30 cores, AC; the sibling `bonsai_mtp_head_fit` GPU tenant had just cleared — load 5.81 falling), tail = bmo (load 0.91); wire = the TB5 bridge `192.168.10.2`, `--split 32`, both stages GPU
**Build:** riir-infer `5602532` (clean detached worktree; katgpt-rs `6c7656193`), release, `-p riir-infer-dist --bin dist_stage --features gpu,bonsai2_hadamard` — one binary, sha-verified amo↔bmo identical (`8da68686…`)
**Rig:** `scripts/i1146_width_2box.sh` (head-local quiet gate + per-cell tail boot over ssh; ABAB per length; tail host overridden to the TB address — bmo's Tailscale is down, the wire itself is TB)

## The wire curve (timed reps; warm cells matched in every pair)

| prompt | w128 r1 / r2 | w512 r1 / r2 | w512 vs w128 |
|---|---:|---:|---:|
| pp512 | 0.525 / 0.532 s | 0.662 / 0.658 s | **0.79× — w128 wins** |
| pp1024 | 0.908 / 0.911 s | 0.937 / 0.925 s | **0.97× — w128, marginal** |
| pp2048 | 1.683 / 1.693 s | 1.475 / 1.476 s | **1.14× — w512 wins** |
| pp4096 | 3.328 / 3.334 s | 2.703 / 2.696 s | **1.23× — w512 wins** |

**The crossover sits in (1024, 2048)** — pinned by the two flanking lengths, both with tight
reps (±1%). The mechanism splits exactly as Bench 909 hypothesised: below the crossover the
small frames pipeline (the tail prefills frame *n* while the head streams *n+1*; pp512/w128
reads 975 tok/s — the 2-box split running FASTER than either 1-box cell because both GPUs
work in parallel), above it the wider frames' GEMM efficiency wins and the loss compounds
with length. The 1-box curve (Bench 919) showed no such crossover because loopback has no
serialisation term.

## The landed default

`dist_stage.rs` — the GPU `--prefill-chunk` default is prompt-length-conditional:

```rust
"gpu" => {
    const WIRE_WIDE_MIN: usize = 2048;
    if prompt.len() >= WIRE_WIDE_MIN { 512 } else { 128 }
}
```

- `--prefill-chunk` still overrides both. CPU backend unchanged (1).
- Precedent: this is the `HeadOptions::min_batched` shape (a prompt-length gate on the
  chunking posture, `runner.rs`), not a behaviour edge — frame sizing is config-class.
- `runner.rs` docs updated (`prefill_chunk` field + the `prefill_width` module paragraph,
  which cited Bench-909's flat "128 wins everywhere" — stale on the wire).

**Functional check (the conditional binary, live wire, no flag):** pp2048 default reads
**1.455 s** = the w512 signature (1.475/1.476, not w128's 1.683/1.693); pp512 default reads
**0.531 s** = the w128 signature (0.525/0.532, not w512's 0.658/0.662). Both branches engage.

## Honesty notes

- No G1 decode cells in this rig — width moves frame boundaries only, every chunked body is
  bit-identity-gated against its sequential twin, and Bench 919's cross-width G1 covered the
  1-box posture; the wire protocol is width-independent (the head sends the width in `Begin`).
- amo carried residual CPU load (5.81 falling from the sibling's fit) — ABAB interleaving
  exposes both arms to the same load, and the reps are ±1%, so the DELTAS are clean; the
  absolute numbers are ~10% pessimistic.
- The tail self-exits after one served stream — the rig boots a fresh tail per cell (both
  arms identically), so boot cost rides outside the timed window either way.

## Provenance

- Session logs: `/tmp/i1146w2/run1/` (pp512, pp4096), `/tmp/i1146w2/run2/` (pp1024, pp2048),
  functional-check logs `/tmp/i1146w2/cond_head_{512,2048}.log` (amo) + `/tmp/i1146w2/cond_tail.log`
  (bmo). Binary sha256 `8da68686…` verified amo↔bmo pre-run.
- The landing commit carries the conditional default + the `runner.rs` doc + a mechanical
  clippy collapse in the 1165 lane's `qwen_attention_prefill_sparse_cubecl.rs` (HEAD's
  `-D warnings` clippy was red there — `collapsible_if`, zero semantics).
