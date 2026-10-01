# Bench — T5.5 ε-sweep Pareto record + T5.3 tg/pp perf at the passing point (Issue 022)

**Status:** RECORD — m(ε) monotone across all 11 points (machined:
`twt_pareto_check`, 8 measured known answers hold); the cost axis is
measured per point (bytes/tensors from the deterministic emit); the perf
axis is measured at the quality-matched point (T5.3, this bench) and
per-point teacher-forced rates come from the agreement sweep log.

## What this is

The Pareto report T5.5 asks for: **(m, params, FLOPs, eval) per
pre-registered ε point**, with box-state provenance on every latency
number, plus the m(ε) monotonicity assertion. T5.3 adds the perf axis at
the **matched quality floor** — the point where the pre-registered
agreement bar PASSES (Bench 022: ε=0.01, agreement 0.9486) — measured as
real decode (tg) and prefill (pp) rates, not the agreement harness's
incidental wall.

## Instruments

- `examples/twt_pareto_check` (this landing) — re-runs the min-max DP at
  every ε over the profile artifact's stored S matrix; asserts m(ε)
  monotone and re-verifies the 8 measured counts as known answers. No
  GGUF written; the DP over a 64×64 matrix is milliseconds. **The
  monotonicity assertion is the Phase 2 bug detector the issue asks
  for.**
- `examples/twt_collapse_emit` — the deterministic emitter; its
  `# output: … (N bytes, M tensors)` line per ε is the cost-axis receipt
  (byte-copied members: kept payload bytes ARE the kept weights).
- `src/bin/twt_perf_tg_pp` (this landing) — T5.3: teacher-forced pp over
  the frozen corpus head (512 tokens — the sweep's chunk length), then a
  greedy free-decode tg loop (128 steps) off the pp tail; KV-per-token and
  fixed GDN state computed from the loaded config with the m/L laws
  asserted by integer cross-multiplication; box-state provenance printed.

## The Pareto table

Pre-registered coarse grid + the fine-end probes, ascending ε. Eval =
parent-vs-collapsed top-1 agreement (Bench 022); the collapsed hit-rate
(the model's own next-token accuracy) in parens. `kept-bytes%` =
collapsed file bytes / parent file bytes (7,206,168,928 B) — the
decode-FLOPs proxy: per decode token the forward touches every weight
once, so FLOPs/token ∝ payload bytes at a fixed quant class (the
byte-copied members never requant). The measured tok/s column is the
ground truth where it exists; the bytes column is the design-time proxy.
gdn/attn = kept layer composition (drives the KV/token column).

| ε | m | depth% | kept-bytes% (FLOPs proxy) | tensors | gdn/attn | eval (agreement) |
|---|---|---|---|---|---|---|
| 0.01 | 61 | 95.3% | 95.77% | 812 | 46/15 | **0.9486 PASS** (0.7505) |
| 0.015 | 57 | 89.1% | 90.01% | 753 | 41/16 | 0.8955 (0.7495) |
| 0.02 | 49 | 76.6% | 78.81% | 653 | 37/12 | 0.5247 (0.5076) |
| 0.03 | 35 | 54.7% | 58.72% | 451 | 21/14 | 0.0301 (0.0240) |
| 0.05 | 25 | 39.1% | 44.83% | 332 | 18/7 | 0.1945 (0.0042) |
| 0.1 | 14 | 21.9% | 29.29% | 187 | 10/4 | 0.0051 |
| 0.2 | 8 | 12.5% | 20.89% | 112 | 7/1 | 0.0029 (0.0022) |
| 0.3 | 5 | 7.8% | 16.61% | 70 | 4/1 | 0.0000 |
| 0.5 | 3 | 4.7% | 13.75% | 42 | — | not run (grid tail, fatal regime) |
| 0.8 | 2 | 3.1% | 12.27% | 25 | — | not run (grid tail, fatal regime) |
| 1.2 | 1 | 1.6% | 10.95% | 17 | — | not run (grid tail, fatal regime) |

Readings the table adds beyond Bench 022:

1. **kept-bytes% ≠ depth%** — the medoid winner is chosen by S-matrix
   distance, blind to layer size, and the surviving membership skews
   heavy: at ε=0.3 the depth cut is 92.2% but the byte cut is only
   83.4%. Every cost claim must name WHICH axis (depth, bytes, tensors);
   the agreement numbers price none of them.
2. **The ε=0.015 row keeps all 16 attention blocks** (only GDN layers
   merge) — so its KV/token is the parent's, and its near-miss quality
   (0.8955) is paid entirely in GDN capacity. The ε=0.03 row's attention
   count (14) exceeds ε=0.02's (12): per-type counts are NOT monotone in
   ε — only the total m is (asserted); merges regroup across types.
3. **m(ε) is monotone across all 11 points** (61 ≥ 57 ≥ 49 ≥ 35 ≥ 25 ≥
   14 ≥ 8 ≥ 5 ≥ 3 ≥ 2 ≥ 1) — the machined assertion holds; no Phase 2
   bug.

## T5.3 — tg/pp at the matched quality floor

The parent and the two bracket arms (ε=0.01 PASS, ε=0.015 near-miss),
same frozen corpus head (512-token pp prompt, 128-token greedy decode,
KV window = the run's own 640-token context):

| arm | layers (gdn/attn) | load | pp tok/s (512 tok) | tg tok/s (128 steps) | kv/token | gdn fixed state |
|---|---|---|---|---|---|---|
| parent (64L) | 64 (48/16) | 129.9s | 1.64 | 1.76 | 131,072 B | 158,859,264 B |
| collapsed ε=0.01 (**PASS 0.9486**) | 61 (46/15) | 129.6s | **1.82 (+10.8%)** | **1.76 (+0.0%)** | 122,880 B (−6.25%) | 152,240,128 B (−4.2%) |
| collapsed ε=0.015 (near-miss 0.8955) | 57 (41/16) | 125.8s | **2.08 (+26.8%)** | **2.13 (+21.0%)** | 131,072 B (±0) | 135,692,288 B (−14.6%) |

Raw record: `.raw/twt/t53_perf_tg_pp.json` (+ `.log`); prompt head = the
frozen corpus's first 512 tokens (`.raw/twt/audition_calib.txt`), decode
seed = the pp tail's argmax.

Free-decode divergence (recorded, never scored): the first generated
token where the collapsed greedy stream leaves the parent's — the
agreement bar's ~5% per-position divergence compounds chaotically in a
free decode; the tg RATES are the claim, the divergence position is
honesty.

The m/L laws held by assertion in-run (integer cross-multiplication, a
violation would have aborted):

- ε=0.01 (attn 15/16): kv/token 122,880 = 131,072 × 15/16 exactly; GDN
  152,240,128 = 158,859,264 × 46/48 exactly.
- ε=0.015 (attn 16/16 — the row keeps every attention block): kv/token
  131,072 == parent; GDN 135,692,288 = parent × 41/48 exactly.
- GDN state is FIXED (recurrence + conv), not per-token: the per-token
  growth is attention KV only — 80 MiB at the run's 640-token context
  (parent), 75 MiB at ε=0.01, 80 MiB at ε=0.015.

## Box state (per the G2 law)

- 2026-10-01T16:07:45Z start (tg/pp), 17:19–18:02Z (profiler floor), UTC.
  Apple M3 Max, 16 cores, macOS 26.6.2.
- `loadavg { 4.12 4.46 4.66 }` at start — moderate sibling load (the
  agreement sweep ran at 11–13; this is the quieter box). AC power,
  battery charged. `powermode 2` (High Power).
- The perf process itself ran ~9.4 cores. Rates are single-run readings
  under that load; the ±20%-class box caveat applies (katgpt-rs AGENTS.md
  §Docs gate), the RATIOS between arms taken back-to-back are the claim.

## Profiler samples/sec floor at L=64 / D=6144

Measured by running `examples/twt_bonsai_profile` on the league model
against the frozen audition corpus with the SVCCA row cap set to 6,144
(cap = retention, not a stop condition — the run was killed at 7,444 rows
once the rate was confirmed to HOLD 38% past the target window; no
artifact was written, the progress-log row counts + process wall are the
receipt, `.raw/twt/bonsai_profiler_floor_d6144.log`):

- **Steady capture rate 1.94 rows/s** (rows 1,141 → 7,444 over the
  observed window, ~9.6-core process), weight load 136 s.
- **D=6,144 capture ≈ 53 min compute + 2 min load ≈ 55 min wall** on
  this box state.
- **Floor (fires downward): a D=6,144 L=64 capture exceeding 2 h wall on
  an equivalent box state is a profiler regression** — 2.2× headroom
  over the measured wall, sized to absorb the documented ±20%-class box
  noise without crying wolf.
- Per-row cost ≈ 0.52 s (the S-matrix hook over the full 64-layer
  ternary forward at 5120 hidden); the fold itself is O(rows × L × dim)
  memcpy + the cosine pass at the end — the forward dominates, so the
  profiler floor tracks the forward's own box-state band.

## Verdict

- **At the pre-registered bar's passing point the depth cut buys almost
  no decode speed on CPU: +10.8% pp, +0.0% tg for a 4.7% depth / 4.2%
  byte cut.** The fixed per-token overheads (embedding, norms, the
  ~0.7–0.8 GB embedding/output tail every ε row still carries — the
  ε=1.2 one-block file is 789 MB total, dequant dispatch) dominate the
  marginal layer at 61/64 layers — the tg rate is floor-bound, not
  depth-bound, at this end of the curve.
- **The near-miss point is where the perf lives: ε=0.015 (10.9% depth
  cut) reads +26.8% pp / +21.0% tg — and its hit-rate is at PARITY with
  the parent (0.7495 vs 0.7478).** The whole tg speedup the lane can
  offer on this model class sits between 0% (the bar's PASS point) and
  ~21% (0.0045 under the trajectory-identity bar, functionally at
  parity). If a deployment prices next-token quality (hit) rather than
  trajectory identity (agreement), the 10.9% cut is the honest operating
  point; the pre-registered bar prices trajectory identity, and that
  choice costs the entire tg gain.
- **tg ≈ pp on this CPU lane** (parent 1.76 vs 1.64 — decode slightly
  faster; weights are page-warm after the pp pass). Unlike the GPU lane
  (decode is busy-bound, prefill compute-bound), the depth cut moves both
  axes proportionally on CPU — the GPU-side tg/pp split (the league
  perf axis) stays the 4090's question, unblocked and optional.
- **Free-decode divergence is immediate at both arms** (generated[3] at
  ε=0.01, generated[2] at ε=0.015): ~5% per-position disagreement
  compounds chaotically in a greedy stream. The rates are the claim;
  the generated ids are recorded in the JSON for reproducibility.
- League context: the CPU decode numbers are NOT the league number (that
  lane is 4090 GPU-busy-bound). What transfers is the RATIO structure:
  the passing point is perf-flat, the parity point carries the gain —
  the same trade the 4090 re-cell will read at its own numerics.

Session: riir-infer-022-t53-t55
