# Bench 938 — Issue 1150 follow-up 2: the GDN chunked-scan route on the dense Metal lane (verdict: parity; the family floor bug found + fixed)

**Status:** LANDED (2026-11-04, zed-986a7538@amo). The Issue-734-T4 chunkwise
family (`deltanet_delta_rule_chunked`, zero code consumers until now) wired
into `qwen38_dense_cubecl` behind `RIIR_QWEN38_GDN_CHUNKED=1` (promote-only).
**Perf verdict: PARITY at every measured cell — no gain, no regression.**
Along the way the route-level G1 caught a REAL numerics bug in the shared
family (the cumulative −60 log-space floor corrupting both-floored γ ratios,
max_out_rel 7.2e+1 on a floor-crossing fixture) — fixed with a per-step α
clamp; the e2e greedy walk is now **byte-identical** between the arms.

## Provenance

- Box: amo (M5 Ultra, macOS 27.2, 96 GB, **AC power**, load 2.27–2.46, up 2d;
  ~75 GB free at record time; the dflash2 sibling's GPU test loop was settling
  during the first runs — the final cells ran with zero other GPU processes).
- Code: `2a06318d6b58` (route `aa33282` + the cum-decay floor fix; the claim
  `63f784a`).
- Model: `~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (16.8 GB, the same file the
  bmo audit used — dense-f16 MM2D dequants it at load; both postures read the
  same GGUF).
- Posture: `RIIR_QWEN38_DENSE_MM2D=1` throughout (the fast dense-f16 arm).
  First-ever compile of the 9 route kernels on amo paid ~140 s ONCE in the
  first smoke (warm-up 183.6 s vs pass 164.3 s — read as 3 tok/s and almost
  recorded as a pathology); the CubeCL disk cache persisted it — every
  subsequent process free-runs at parity. **The lesson stands: never record
  the first-ever-compile process of a new kernel family.**
- Harness: `qwen38_wgpu_bench` (--pp/--passes/--walk; median of passes after
  warm-up; targets round up to the chunk multiple — pure prefill).

## Route G1 — the gate that caught the bug (`tests/qwen38_gdn_chunked_route_g1`)

GPU-vs-GPU: my route (token-major repack + 64-token chunkwise windows) vs the
incumbent `DeltanetRecurrenceMultiTokenCubeCL` on the same random fixtures,
TOL 1e-4 (the family's own band):

| fixture (h=8, d=128) | max_out_rel | max_state_rel | verdict |
|---|---|---|---|
| FloorSafe α∈[0.93,1), 1 window | **1.59e-5** | 6.0e-5 | PASS — wiring correct |
| FloorSafe, 2 windows (state carry) | **9.74e-6** | 7.5e-5 | PASS — carry correct |
| FloorCrossing α~U(0,0.5), 1 window — OLD floor | **7.22e+1** | 2.3e+2 | FAIL — the floor bug |
| FloorCrossing — per-step α clamp fix | **6.70e-6** | 7.7e-6 | PASS |

The family's own 3 in-module tests pass before AND after the fix (their
fixtures never cross the floor — that is exactly why the bug survived).

## The floor bug (now fixed in shared substrate)

`cum_decay` floored the CUMULATIVE log-decay at −60. Sound goal (guard
`ln(0)=−∞` when a step's α underflows to 0 in the beta-decay kernel) — wrong
instrument: once two window positions are BOTH floored, their ratio reads
`exp(−60−(−60)) = 1` where the true ratio is O(1) (e.g. 0.25) — an O(1) error
on every floored pair's intra-chunk weight. The module doc's "behaviorally
exact for every reachable input" claim was tested only on fixtures that never
cross the floor. Real qwen3.8 heads decay fast enough (α = exp(a_log·softplus)
per token; a_log ≈ −2-class heads cross e⁻⁶⁰ within a 64-token window) — the
e2e effect was the 64-token greedy walk DIVERGING at token 3 (recorded in the
pre-fix run: OFF "请帮我分析这段文字" vs ON "请帮我找出重复的句子").

**Fix:** clamp the PER-STEP α at 1e−30 before `ln` — a true-0 α encodes as a
−69 step (all cross-boundary ratios → 0, matching the sequential
forget-everything semantics), and the cumulative stays exact, so every ratio
is the true difference of logs. The floor-crossing fixture drops 7.2e+1 →
6.7e-6; the e2e walk md5 is byte-identical OFF vs ON. ⚠ Semantic choice,
deliberate: 1e−30 is not "a big enough number" to tune — a true-zero α MUST
encode as forget-everything (a −69-class log step); do not "fix" the clamp
back to a cumulative floor or raise it without re-running the route G1's
FloorCrossing arm (it reads 6.7e-6 at this clamp, 7.2e+1 at the old floor).

## Perf cells (warm compile cache; median tok/s, padded pure-prefill)

| cell | incumbent (env off) | chunked route | verdict |
|---|---|---|---|
| pp512 (1 chunk) | 21 (24.4–25.0 s) | 21 (24.5–24.9 s) | parity |
| pp2484→2560 (5 chunks) | 21 (120.2–121.8 s) | 22 (115.9–116.9 s) | parity (~±4%, one rounding step) |
| pp1024 @ chunk 1024 (16 windows/layer) | — | 21 (48.1 s, linear) | scaling clean |

Stage-timer bucket A/B @pp512 (same sync posture both arms): gdn **1.44 s
(5.9%) OFF vs 1.67 s (6.6%) ON**; attn 388/367 ms; **tail_logits 22.0 s
(90.5%) OFF / 22.8 s (90.3%) ON**; wte ~450 ms.

## The two findings that outlive the route verdict

1. **The stage-share premise was a bucket misread.** The audit's "GDN scan
   (48 layers) 31–32%" (bmo, pp2484, Q4_K incumbent posture) is dominated by
   the GDN-layer GEMMs (qkv/z/alpha/beta/ssm_out weight streaming), not the
   recurrence scan. On the dense posture those GEMMs collapsed ~3× and the
   recurrence itself is ~6% of a synchronized pass (1.4–1.7 s of 25 s at
   pp512). There was never a 31–32% scan lever to pull. The multi-token
   kernel's serial token chain is latency-adequate at n_v = 8 heads — the
   chunkwise decomposition's parallelism (n_head·C-shaped) is too thin here to
   beat it.
2. **The REAL lead on this lane is tail_logits** — 90% of the synchronized
   pass at pp512 (22 s: 8 f32 GEMV sweeps over 248,320-row lm_head for ONE
   row of logits per 512-token chunk) and 15–16% of the audit's pp2484 pass.
   The lm_head tail GEMV routing follow-up (issue 1150) is the lane's actual
   bottleneck, not GDN. ⚠ Bucket-naming caveat for the next reader: the
   stage-timer `gdn` bucket is the WHOLE GDN layer (norms + 5 GEMMs + conv +
   recurrence + gating), not the scan alone — the scan itself is ~6% of a
   synchronized pass on this posture. Read the bucket as `gdn_layer`.

## Route disposition

`RIIR_QWEN38_GDN_CHUNKED` stays promote-only, NEVER default (no perf gain to
promote — the GOAT gate's G2 fails on parity). The route stays LANDED as the
recorded, parity-proven opt-in: G1 clean at 1e-5-class across floor-safe AND
floor-crossing profiles, single and multi-window; argmax-identical e2e. It is
the ready-made arm for any future GDN lane with heads wide enough that
chunkwise parallelism pays, and the cum-decay fix hardens the family for that
day (it was silently wrong for fast-decay heads on ANY lane that ever adopted
it).

Session: zed-986a7538@amo, 1794465837.
