# Bench 946 — Issue 1176: the league env set promoted to the Metal default — bare boot = 966 tok/s (was ~7), byte-identity + per-family retention clean

**Status:** LANDED (2026-11-05, zed-ablate@amo). The five-env league
posture (`RIIR_QWEN38_DENSE_MM2D`, `RIIR_QWEN38_LMHEAD_Q6K`,
`RIIR_QWEN38_DENSE_MM2D_RELAXED`, `RIIR_QWEN38_DENSE_MM2D_F16ACT`,
`RIIR_QWEN38_ATTN_TENSOR`) flipped DEMOTE-ONLY on
macOS+metal_tensor_gemm (the exact literal `=0` each restores the
incumbent byte-identically; promote-only `=1` elsewhere — the forward
still constructs off-Metal; the CUDA twin is the separate
qwen38_dense_cudarc crate). Claude verdict round 1 AGREE (4 conditions,
all honored below); this record is the dial's GOAT cell — the arms keep
their own records (i1150/914–925/934/940/941/942).

## G1 cells (pp5120, release, `--features qwen38_wgpu,metal_tensor_gemm`, isolated target)

| cell | median | notes |
|---|---:|---|
| BARE default (no envs) | **966 tok/s** (889/950/966/972 warm-up+3) | the pre-1176 bare route ran **7 tok/s** (755 s/pass) — the promotion retires the ~100× trap |
| LEAGUE (the five `=1`, bench 944's posture) | 957 tok/s (866/888/957/966) | same class; run-to-run ordering within the ±0.5–1% spread |
| ALL0 (the five `=0`) | **7 tok/s** (755.09 s warm-up, 754.77 s pass) | the demote literals provably restore the incumbent route (bench 944's 736 s reproduced within noise) |
| ALL0 pp512 cross-check | 7 tok/s (75.71 s/pass) | the incumbent class is shape-flat here — not a pp5120-only reading |

Boot-banner identity: the BARE boot prints the identical arm set as the
explicit LEAGUE boot (dense armed league DEFAULT + tensor-flash + half-B
+ Q6_K packed tail, all `relaxed_precision=true`); the ALL0 boot prints
the dense DEMOTED banner and loads the packed route (11.7–11.9 s vs the
dense posture's 20–30 s).

## Walk identity + per-family retention (verdict condition 1)

Greedy-argmax walks, one process per cell, `--walk-tokens 48` (needle
32). **Identity:** BARE == LEAGUE byte-for-byte on the two families run
under both postures (prose 48/48, math 48/48). **Retention vs the
bit-exact incumbent (ALL0):** top-1 agreement **272/272 across all six
families — zero first divergence anywhere**:

| family | prompt | walk | matching prefix (BARE vs ALL0) |
|---|---|---:|---|
| prose | harbor-town continuation | 48 | **48/48** |
| math | prime-sequence continuation | 48 | **48/48** (continues 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 in both) |
| code | quicksort fn body | 48 | **48/48** |
| multilingual | Thai sentence (Bangkok/Tokyo capitals) | 48 | **48/48** |
| structured | JSON order record | 48 | **48/48** |
| long-context recall | needle `AMBER-LANTERN-7742` in ~675 tokens of filler | 32 | **32/32** (recites the watchword identically; even the shared trailing special tokens 159034/248044/248045×2 agree) |

No family drops — **the lossy-arm fallback (demote `_RELAXED`/`_F16ACT`
back to opt-in) is NOT invoked**; all five arms stand as the default.
This is a top-1/greedy read at pp≤675-prompt scale — the numerics delta
the arms carry (f16 weights, half×half GEMM, relaxed matmul2d) is
below the argmax decision floor on these families; it is NOT a KL/CE
distribution claim (the per-family KL instrument remains
`issue1146_relaxed_kl` for the ternary lane; the dense lane's 940
record carries its 2× precision cell). Honest scope: the retention cell
certifies the dial on THIS model (qwen3.8-27b-Q4_K_M) at THESE prompt
families; a different checkpoint re-runs the cell.

## Demoted-route guard (the trap stays reachable)

The incumbent route is one `RIIR_QWEN38_DENSE_MM2D=0` away and is now a
documented, banner-labeled posture (the parity lanes' bit-exact
anchor), not a silent default. Dependent-arm refusals name both fixes
(demote the dependent too, or remove the dense demotion).

## Provenance

- Commit: the 1176 flip on top of wip `438eeab` (this record lands with
  it). Build: release, `CARGO_TARGET_DIR=/tmp/i1176-target`,
  `--features qwen38_wgpu,metal_tensor_gemm`.
- Box: amo (M5 Ultra, 30 cores, macOS 27.2), AC power, load 3.25→2.60
  over the run, no GPU compute consumer (idle rethink serve + fleet-web
  CPU services only), 96 GB RAM. One process at a time (GPU-exclusive
  correctness cells).
- Model: `/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (64
  layers = 48 GDN / 16 attn). Logs: `/tmp/i1176/` (session-scoped;
  the tables above are the durable record).
- Postures: BARE = the five envs unset; LEAGUE = the five `=1`;
  ALL0 = the five `=0`. Env-parse sites: 4 in `qwen38_dense_cubecl.rs`
  + the relaxed selector in `gemm_dense_f16_metal_zero_copy.rs`
  (macOS+metal_tensor_gemm-only module — no platform split needed
  inside it).
