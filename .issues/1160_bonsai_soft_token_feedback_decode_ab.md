# Issue 1160: Training-free soft-token feedback decode A/B on Bonsai (LIFT/MoI class)

**Status:** Open — ready (GAIN filing, not owner-gated). Machine: **Metal-first** — the PQ2_0 wgpu/Metal decode lane is the serving default (riir-ai Bench 1011 lineage, ktile64+zerocopy), so G0 and the pilot arm run local; the full grid, if armed, is a long task → 4090 via issue per house rule. Verify the Metal lane decodes PQ2_0 before relying on it; fall back to the 4090 otherwise.

## Context

Source: [Research 487](../../riir-train/.research/487_LIFT_Teacher_Forced_Latent_Feedback_Pretraining.md) — arXiv:2609.38149 (LIFT trains a latent state channel; the training-free inference-only member of the same family is Mixture of Inputs, arXiv:2505.14827, NeurIPS 2025, and Soft Thinking).

**Question:** does feeding the model's own top-k output distribution back as the next-step input (a "soft token") improve frozen-model decode quality on `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` at ≤1% per-step decode cost — without any training?

**Why this could work where bench 668 failed:** different interface. Recirculation mixed INTERNAL hidden states across layers (11→4) — an interface the model never consumes natively; it failed 12/12 on our model. A soft token enters through the model's NATIVE input path (token-embedding space, same shape, same position, KV layout unchanged). MoI shows frozen dense models accept it — but its published gains are ACCURACY on generated answers (math/code), not perplexity.

**Why it could fail:** (a) PQ2 ternary distributions may be spiky — if top-1 carries most of the mass, the weighted mean E^⊤s collapses toward the argmax embedding, which is LIFT's own measured k=1 FAILURE mode (−1.16 ppl vs compute-matched; width is the mechanism, not presence); (b) the one family data point on our stack is negative (katgpt-rs bench 668); (c) MoI's long-generation drift failure mode.

## Method (decode-only; no training; no wire/vessel change)

At each decode step, after sampling x_{t+1} from logits o_t, compute s = softmax(top_k(o_t)/τ); the NEXT step consumes input embedding E^⊤s (**pure arm**) or (1−β)·E^⊤s + β·E[x_{t+1}] (**MoI arm**). Prefill untouched; KV shape unchanged; the emitted token path unchanged (only the input side changes).

- Env `RIIR_SOFT_TOKEN_FEEDBACK` promote-only (arms on exact `1`/`true`/`on`); unset = byte-identical incumbent (env-switch law). Feature-gated code; default-off until a GOAT says otherwise.
- Spiky-distribution guard: explicit tail constant on the top-k renormalization — never a silent floor (riir-train R486 demarcation).

### G0 — early collapse probe (minutes; run FIRST, local Metal)

Decode a small eval slice and record the median top-1 mass + entropy of the decode distributions. **Median top-1 mass ≥ 0.9 ⇒ the soft token ≈ the argmax embedding ⇒ close as k=1-collapse with the negative recorded, WITHOUT the sweep** (LIFT's D.2: the k=1 channel loses by 1.16 ppl). This converts the issue's main named risk into a check that costs minutes.

### Arms and sweep budget (cost stated up front)

Arms: pure (1) + MoI (β ∈ {0.5, 0.9}) = 3; k ∈ {16, 128, 1024} × τ ∈ {1.0, 1.5} ⇒ **18 configurations, each a SEQUENTIAL (one-token-at-a-time) decode on a 27B — never run all 18 blind.** **Pilot first:** (k=128, τ=1.5, β=0.9 MoI). The full grid runs only if the pilot clears G0 and moves G1b in the right direction.

## Gates (pre-registered; demote-on-wash)

- **G1a — sequential NLL (MoI arm only; a guard, not the headline).** ⚠ Teacher-forced parallel scoring (`si_zero_shot_ppl --golden-nll`) CANNOT score these arms as-is: the pure arm replaces the real token's embedding with the model's own prediction (the scored text is never seen — NLL bad by construction), and the MoI arm makes each position's input depend on the previous step's output (sequential scoring only). The wiring is NEW WORK — a named task, not an assumed lane: a one-token-at-a-time scoring loop; `golden_nll` today is a feature row (`golden_nll = ["katgpt-core/fixed_log_softmax"]`, Cargo.toml) wired into `si_zero_shot_ppl` for quant-variant comparison only. Bar: pilot sequential NLL ≤ baseline (tie allowed here; the headline is G1b).
- **G1b — accuracy on generated answers (the headline; both arms).** Fixed set of **N ≥ 200 procedural/arithmetic prompts (fixed seed list, set in advance — a McNemar test on a few dozen cannot reach significance)** (LIFT's largest-gain class), exact-match scoring, **paired McNemar vs baseline** (the Plan 480 gate pattern). Headline bar: best arm beats baseline exact-match at McNemar p < 0.05.
- **G2 — cost:** per-step decode time within 1% of baseline (the projection is one k×d matvec ≈ 0.02% of a 27B forward on paper — measure, don't assume; `bench_preflight` provenance line beside every number).
- **G3 — stability:** greedy decode on a fixed prompt set, 512+ tokens, no divergence/loops (MoI's drift failure mode).

## Rules

- Metal-first for G0 + pilot (after verifying PQ2_0 decodes on the Metal lane); 4090 for the full grid if armed. `CARGO_TARGET_DIR=/tmp/...` isolation; GPU-exclusivity check before any timed run (riir-ai house rule).
- Even a G1b win stays opt-in: serving is decode-throughput-first (perf league); promoting a quality mode into any default set is a separate owner call with its own GOAT.
- Demote-on-wash: a G0 collapse or a G1b tie closes this issue (same posture as bench 668's recorded negative); reopen only if a dense (non-ternary) consumer lane appears or the Plan-344 LIFT arm lands (a trained channel would change the prior).
- Any healer-caused wrong heal on this lane: file to riir-refine per house rule.
