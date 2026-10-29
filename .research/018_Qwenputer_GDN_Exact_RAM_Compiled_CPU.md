# Research 018 — Qwenputer: a RISC-V CPU compiled into Qwen3.5-4B's weights, GDN state as exact RAM
**Status:** Done — Gain verdict, 1 issue filed (no plan, no trainer filing). Verdict negotiated with the Claude reviewer (2 rounds; REVISE round 1 corrected the L1 exactness premise — see §2 and Issue 1155).

> **Sources:** blog <https://divagr.com/blog/qwenputer> (2026-10-22) · model card <https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer> (Apache-2.0; ModelScope mirror `Spartan1925/Qwen3.5-4B-Qwenputer`) · code `divagr18/qwenputer` **@ `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`** (MIT, shallow clone in `.raw/qwenputer`, removed post-verdict; quotes re-verified at this pin). Author: Divyansh Agrawal (`@software{agrawal2026qwenputer}`, v0.2.0).
> **Date:** 2026-10-29
> **Related Research:** katgpt-rs 621 (Handwired — exact compiled circuits; the law-set cousin), katgpt-rs 229 (ProgramAsWeights — the symbolic-first verdict), katgpt-rs 074 (Subterranean — training-based compilation, the opposite pole), katgpt-rs 575 (dQwen3.5 — the Qwen3.5 GDN-hybrid arch twin), katgpt-rs 070 (Gated DeltaNet 2), **this repo 007 (Triadic GDN — the GDN lane's state-capacity intel)**
> **Related Issues:** riir-infer Issue 1155 (`gdn_exact_slot_scratchpad_poc` — filed this session); katgpt-rs Issue 940 (float reduction-order determinism law — the fold-order constraint the POC inherits)
> **Classification:** Public

---

## TL;DR

Qwenputer is Qwen3.5-4B with a 32-bit RISC-V CPU (RV32IM + Zba + bignum coprocessor) **compiled** into 11 extra transformer layers ahead of Qwen's own 32 — **zero training: a compiler sets every computer weight**, Qwen's weights untouched. The CPU's RAM is the model's own **Gated DeltaNet state**: one-hot keys address one state cell exactly, and a value-domain contract (integer payloads on an exact grid, enforced by compile-time validators) makes each addressed cell an exact memory cell — registers/PC included. The base model drives the CPU through its own trained tool-call vocabulary (`<tool_call>` for `calc` — nothing external runs); an 11-layer forward computes the result; the model writes the `<tool_response>` digit by digit. Result: exact arithmetic (97.5–100% vs base 55–90%, matching/beating a real host Python tool), ~230,000 replayed output positions with **zero mismatches** vs a reference machine, 18M free-running instructions without error, and base-text KL 6e-4 (noise floor 5e-4). Bonus lane: an in-model filesystem (1,024 hash slots, 512 files ≈ 53K tokens) that **survives conversation save/load** — 90% cross-chat fact accuracy vs 30% for pasted history.

**Why it matters to THIS repo:** the trick lives entirely on our substrate. The deltanet lane owns GDN forwards/kernels/state (Bonsai-27B carries 48 GDN layers; qwen3.8 is the arch twin family); `delta_roll`'s `GdnStateJournal` already journals per-token (u, k, β, g) contexts and unwinds rank-1; the multi-host lane already ships `set_hidden_state`/`read_hidden_state` GPU seams. Qwenputer is the existence proof that **GDN state can be driven as exact, addressable, persistent RAM** — and the load-bearing lesson for us is that the exactness is a **domain contract enforced by validators**, not a property of the delta rule (§2 L1 — our own kernel does NOT get it from algebra; Issue 1155 proves exactly that split).

## 1. What ships (artifact + code, at the pin)

| Surface | Fact |
|---|---|
| Checkpoint | 18.8 GB bf16, **mostly zero padding**, ~1M nonzero computer weights; hidden 3968 = Qwen's 2560 + computer dims; vocab + 230,324 trace tokens; config-only changes — **loads as stock `Qwen3_5ForCausalLM`** |
| Computer | 11 compiled layers before Qwen's 32; RV32IM + Zba + bignum unit; passes official riscv-tests; programs match QEMU; 18M free-running instructions, zero errors |
| Driver | `RvChat`: boots firmware INTO the GDN state after the prompt; keeps CPU trace tokens out of Qwen's layers (`mute.py`); every token still comes from the weights |
| Executor | Triton batched executor, ~3,200 CPU trace tokens/s on an L4 (seconds per big calculation — not a speed play) |
| Benchmarks (greedy, 40/set) | Synthetic int arithmetic 75→**97.5**; DeepMind `mul_big` 55→**97.5** (Python tool 100); `div_big` 72.5→**97.5** (tool 95); `mul_div_multiple_longer` 90→**100**; `add_or_sub_big` 87.5→**97.5**; NUPA long-int 48.6→**100** (= Python tool). Exactness: ~230K replayed output positions vs reference machine, **0 mismatches**. Base-text KL 6e-4 vs noise floor 5e-4 |
| Filesystem (`rv-store` branch + second checkpoint) | internal put/append/get/list/delete — **no tool calls**; name hashed into 1,024 slots (linear probing), text in other heads at 8 tokens/row chained; ≤512 files ≈ 53K tokens; **persists across conversations via state save/load**; cross-chat facts 90% vs 30% (pasted history) and 83.3% vs 78.9% (Python-run memory tools) |
| Safety write-ups (blog follow-ups) | hybrid models can hide reasoning in RNN state evading CoT monitors; compiled weights as in-weight monitors the model cannot switch off |

## 2. The exactness construction laws (the transferable content; quotes at the pin)

The compiler (`chipformer/q35/`) refuses any program step that cannot be represented **exactly** in the running dtype. That discipline, not the CPU, is the transferable asset. ⚠ **Round-1 verdict correction:** the bit-exactness is NOT a property of the delta-rule algebra — it is a **value-domain contract enforced by validators**. Our own kernel proves the difference (see L1).

- **L1 — one-hot key + domain contract ⇒ exact cell overwrite.** `ir.py` L12-13: "keys must be one-hot (an exact overwrite of one state row) or zero"; enforced at compile time: `require(nz <= 1, "key must be one-hot", nz)` (L425). **Orientation:** one-hot `k = e_j` addresses one key **column** `j` across all value rows (our layout is `S[val_row, key_col]`, `forward.rs` L255/L262 — Qwenputer's "row" wording is transposed for our kernel). **And the algebra alone does NOT give bit-exactness:** our `gated_deltanet_step_inplace` (L250-258) computes the overwrite as `S[row,j] ← S[row,j] + b·(v[row] − S[row,j])` (the delta form). With b = 1 that is `a + (v − a)`, which is **not** `v` for arbitrary f32 — cancellation counterexample: a = 1e8, v = 1.0 → `1 − 1e8` rounds to `−1e8`, the sum reads `0.0`. Bit-exactness holds only when the payload lives on an **exact grid** (integers with |a|, |v| and |v−a| all ≤ 2^24 — every intermediate exactly representable in the f32 mantissa): `v − a` and `a + (v − a)` are then exactly representable and the round trip is exact. That grid is Qwenputer's L4/L6 contract, carried by `require(is_bf16(...))`/`silu_ok(...)` validators. The readout needs its own guard: the addressing dot sums `S[row,c]·k[c]` over ALL columns, so a non-finite value in any NON-addressed slot produces `inf·0 = NaN` inside `simd_dot_f32` and contaminates the read — the domain contract must include **finite-only state**.
- **L2 — exact 0/1 write gate via saturating sigmoid.** L443-445: `beta = sigmoid(256 g − 128)`: "exactly 1 at g = 1 and exactly 0 for integers g <= 0" — bf16 saturation turns a sigmoid into a hardness function. Same trick as katgpt-rs fast_sigmoid saturation, used here as an exactness device rather than a smoothness one.
- **L3 — decay as a three-value selector {keep, halve, clear}.** L42: `DECAY = {0.0: 1.0, 110.0: 0.5, 238.0: 0.0}  # a -> exp(g) with dt_bias = -110, A_log = 0` — the dt/A_log config is wired so `exp()` lands on exact values 1.0 / 0.5 / 0.0. Keep (1.0) and halve (0.5) are exact power-of-two scalings; clear (0.0) destroys the state irreversibly. Halving-counter rows (`A_BIT * 2^-n`, L100/L381) implement multi-step timing as repeated halving — **the gate machinery doubles as a bit-shift unit**, bounded by the f32 **exponent** range (exact while the payload stays normal, 2^-126; subnormals extend to 2^-149 with reduced guarantee) — an exponent bound, not a mantissa one.
- **L4 — silu-as-exact-ReLU by K-scaling.** L38: `K = 128  # silu gate scale: silu(K n) = K relu(n) for integers n` — at scaled integer range the sigmoid factor saturates to exactly 1, so the production activation computes exact integer ReLU. FFN arithmetic proceeds in exact integers inside a live model. This is the generator of the exact grid L1 needs.
- **L5 — ballast dims pin the norms.** `emit.py`: chip tokens carry a huge constant B in ballast dims so "with m·B²/hidden a power of 4, every RMSNorm on a chip token divides by exactly sqrt(m/hidden)·B" — residual scaling becomes an exact power of 2 and weights carry exact reciprocals. Norm layers stop being noise sources by construction.
- **L6 — compile-time require() gates = declarations + validators.** Every emitted step is checked: `require(is_bf16(a) & np.isin(a, list(DECAY)), …)`, `require(silu_ok(gate), "gate outside silu's exact range", …)`, one-hotness, beta shape. A program that cannot run exactly **does not compile**. This is Handwired's declaration/validator discipline (katgpt-rs Research 621 M1/M8) applied to a *production* transformer — independent rediscovery, i.e. convergence evidence that the law-set is the right import. **L6 is what makes L1 true**: the validators enforce the domain contract; the delta rule only supplies the addressing.
- **L7 — trace-token masking.** The CPU's execution trace lives in the 230K extra vocab tokens and is **muted from the base layers** (`q35/mute.py`) — the coprocessor's scratch work never perturbs the host's residual stream (hence KL 6e-4).
- **L8 — piggyback the trained interface.** The host model invokes the coprocessor through vocabulary it was already trained on (`<tool_call>` shape) — **no post-training anywhere** (KL vs base within noise). The integration cost lands in the driver, not in weights.

## 3. Path 0 / three-track routing (per-track verdicts, never pooled)

| Track | Verdict | Home |
|---|---|---|
| (a) modelless inference | **No file.** No modelless consumer: our delta-rule memory family already owns the slot-table-vs-delta-rule axis — `kron_memory`'s G0 measured the one-hot slot table WINNING at low load (bench 951), which is Qwenputer's regime choice confirmed in miniature. Forcing an open primitive would invent a consumer. | — |
| (b) self-adaptive runtime | **Fusion idea, novelty TBD** (F1/F2 below). | — |
| (c) model-based weights | **No training exists to file.** "None of their weights are learned: a compiler computes every one" — construction, not GD. The adjacent idea (compile circuits into OUR checkpoints) is **WATCH**: ternary/PQ2 quantization destroys every exactness contract above (cf. Research 538 GDN W4A4 survival); the dense qwen3.8 f16 lane could in principle host bf16-compiled layers, but no consumer or serving posture exists today. | — |
| **Substrate (this repo's MOAT)** | **Gain — this note + Issue 1155**: prove the exact-slot identity ON OUR KERNELS as the two-sided claim it actually is — in-domain bit-exactness (G1a) AND the negative control showing arbitrary-f32 payloads break it (G1b, so nobody records "the identity does not transfer" off a false premise) — plus the journal-restore hypothesis cell. | `.issues/1155_gdn_exact_slot_scratchpad_poc.md` |

No adversarial panel was run: the input is a zero-GD artifact + engineering write-up, not a training-adjacent paper — the panel trigger ("first-pass classification touches training") does not fire. Model-based coverage was checked directly (above): the only model-based angle needs no training and is recorded as WATCH.

## 4. Signal-diffs vs closest cousins (why none kills the note's job)

- **katgpt-rs 621 (Handwired)** — compiles programs into its OWN layered tensor runtime (affine/reglu/boolean/index), bit-exact, differentiable. Diff: Qwenputer embeds into a **production pretrained transformer** using its live GDN layers as RAM, and the host model *autonomously drives* the coprocessor through its trained interface — a consumer class Handwired does not have. 621's verdict (circuit port blocked by 229's symbolic-first ruling **absent a concrete consumer**) is **unchanged**: Qwenputer creates no symbolic-constraint consumer for us. L4–L6 here *are* 621's M1/M8 laws rediscovered on a different substrate — recorded as convergence, not new claims.
- **katgpt-rs 229 (ProgramAsWeights)** — spec→compile verdict, symbolic-first for enumerable-output specs. Unchanged; no new evidence against it.
- **katgpt-rs 074 (Subterranean)** — procedural compilation via full fine-tuning ("LoRA fails for procedural knowledge"). Qwenputer is the **no-training pole** of the same "procedure into weights" axis: where the procedure is a *digital circuit*, construction beats training entirely. Axis observation, no filing.
- **this repo 007 (Triadic GDN)** — GDN lane state-capacity intel (E× state, decode E× caveat). Orthogonal axis (capacity vs exactness); both notes feed the same lane.
- **katgpt-rs 575 (dQwen3.5)** — the Qwen3.5 arch twin (our league runs qwen3.8 GDN hybrids). Qwenputer is **not league-relevant**: 3,200 CPU tok/s on an L4 is a correctness play, not a decode-throughput play; the opponents table is untouched.
- **Percepta `transformer-vm` (katgpt-rs Research 031/032 — added by owner challenge, 2026-10-29; the closest cousin in our own corpus, missed by the original sweep).** End-to-end compiler that analytically builds a WASM VM into a standard autoregressive transformer's weights — zero training (`expr_to_tensor` populates weights from a MILP schedule), ReGLU/stepglu gate primitives, `fetch_sum` state-through-attention, Futamura specialization (`_cursor_lookup`), d_model=36 / 7 layers / ~1.5 MB, deterministic greedy execution. Same class as Qwenputer, so the not-Super-GOAT call is STRENGTHENED, not changed. The deltas Qwenputer holds: (i) **production pretrained host** — Qwen's 4B weights untouched (KL 6e-4), the LLM autonomously drives the coprocessor through its trained tool-call vocabulary, vs a purpose-built 1.5 MB transformer that is only a computer; (ii) **RAM substrate** — GDN linear-attention state (O(1)/head, one-hot row overwrite, persists across state save/load — the filesystem) vs soft-attention KV with a `HARD_K=1e10` hardmax approximation; (iii) **exactness validation** — 230K replayed positions + 18M free-running instructions + compile-time require() gates vs smoke-class tests (031: "no adversarial tests visible"). Convergences: ReGLU/stepglu ↔ L4 silu-K-scaling; the erase-on-slot-reuse −1.0 self-loop ↔ the delta-rule overwrite; both zero-training analytical construction. Workspace tie-in: our healer's `wasm_validator` lane already ships the percepta decoder + lowering (`percepta_wasm`), so this cousin was doubly findable — the sweep vocabulary was wrong, not the corpus.
- **kron_memory G0 (katgpt-rs bench 951)** — one-hot slot table beat the delta-rule cell at low load. Qwenputer is that result's in-model existence proof: exact addressable slots are the right memory regime when the workload is a program, fuzzy delta blending when it is association.

## 5. Verdict

**GAIN** — substrate note (this file) + Issue 1155 POC; fusion ideas recorded below; no plan, no trainer filing.

Why not Super-GOAT: the class is published and the headline artifact is itself public — Schlag/Irie/Schmidhuber 2021 (delta rule as fast-weight programming — the blog's own citation), RASP→Tracr (programs → exact hand-set transformer weights), The Neural Compiler (arXiv:2605.22498), Handwired (Research 621), **Percepta's transformer-vm (Research 031 — end-to-end WASM-VM→transformer-weights compiler, the closest in-corpus instance)**, and Qwenputer itself (the production-model instance). We cite the artifact; the workspace-level novelty is only the *bridge to our GDN substrate* — exactly what Issue 1155 prices, and it is a POC, not a claimed primitive.

**MOAT gate per domain:** katgpt-rs — laws already recorded via 621; no new file there (this note carries the crosswalk). riir-infer — **this note + Issue 1155** (GDN lane owns the surface). riir-train — nothing: zero training exists to file. League — no impact (correctness play, not speed). Healer (fusion priority #2, asked): zero documented healer gaps map to in-weights coprocessors — the 621 sweep already covered the circuit-class healer surfaces; non-actionable. Game runtime (fusion priority #1, asked): exact arithmetic in our stack stays **raw-side Rust** by the sync-boundary law (deterministic replay needs exact values, computed natively, never through a model) — no consumer; recorded.

## 6. Fusion (recorded, novelty TBD — no file, no issue beyond 1155)

- **F1 — persistent per-session GDN scratchpad.** Qwenputer's filesystem survives conversation save/load because GDN state is *state*. Our analog: a serving lane that freezes named GDN-state regions into a `MerkleFrozenEnvelope`-committed artifact and resumes them through the `GdnStateJournal`/disaggregated-handoff seams — durable in-state scratch across sessions, BLAKE3-checked like every freeze. Re-open trigger: any serving lane that needs durable in-state scratch (the TokenGate/GDN delta-roll consumer seams are the candidate doors).
- **F2 — in-weight monitors / RNN-hidden reasoning (safety watch).** The blog's follow-up claims (hybrids hiding reasoning in RNN state; in-weight monitors the model cannot switch off) map loosely onto rethink's ESC-enforcement line and riir-auth's behavioral-identity pillar. Watched, no consumer, no filing — neither surface asked for it.
- **F3 — sparse mostly-zero checkpoint loading.** 18.8 GB bf16 carrying ~1M nonzero weights is a loader/quant-format question (mmap + sparse segment tables). WATCH for `gguf_loader`/`safetensors_loader` if a Qwenputer-class checkpoint is ever consumed; no plans to consume one.

## 7. What's private vs open

Everything is public (MIT code, Apache-2.0 model); the note lives in the public substrate repo. Nothing touches riir-* private IP. Any future port of L1–L6 into our kernels lands HERE under a feature flag with Issue-940-conformant determinism pins (sequential fold order; first-index ties) BEFORE any published number — the laya-lane discipline.

## 8. Validation protocol (Issue 1155, if pulled)

1. **G1a — in-domain bit-exact gate (CPU):** drive `gated_deltanet_step_inplace` with gate=1, β=1, one-hot key column, payloads on the exact grid (bounded integers: |a|, |v| and |v−a| all ≤ 2^24) and finite-only state → assert every addressed cell round-trips bit-exactly and non-addressed cells are bit-unchanged, across dtype/shape batteries. **G1b — negative control (the load-bearing half):** the same battery with arbitrary-f32 payloads MUST produce mismatches (the cancellation class: a=1e8, v=1.0 → 0.0) — proving the domain contract is what carries exactness, and pre-empting the Issue-825 false-negative ("the identity does not transfer") that an unconstrained G1 would have recorded.
2. **Decay-selector semantics:** keep (1.0) and halve (0.5) are exact power-of-two state scalings on our kernel; clear (0.0) zeroes the state exactly (state destruction, forward-side). Halving-counter rows reach 2^-n exactly within the f32 exponent range (normals to 2^-126; subnormals to 2^-149 with reduced guarantee).
3. **G4 zero-alloc** posture on the hot path; **G3** byte-identical behavior with the demo flag off (the primitive is a proof, not a default).
4. **Journal interplay — HYPOTHESIS cell, scoped:** a pinned-slot region written in-domain and journaled + unwound via `GdnStateJournal` under **keep/halve** postures restores bit-exactly (the unwind `(S − δ·k)·(1/g)` is exact scaling plus the same-grid subtraction — law 2's "algebraically exact, NOT bit-exact" becomes bit-exact IN-DOMAIN; the claim is pre-registered as a hypothesis precisely so a failure is a result, not a surprise). **Clear is out of scope**: journal law 1 refuses g=0 windows ("irreversible by construction").
5. **Stretch (owner-gated):** one GPU-lane parity cell (wgpu) at the same shapes; and a ternary-quant survival probe documenting WHICH L1–L6 contracts die under PQ2 — feeding the Research 538 survival line.

## P0–P3

- **P0 (this commit):** this note + Issue 1155 + highwater bumps.
- **Addendum (2026-10-29, owner challenge):** the cousin sweep missed Percepta's `transformer-vm` (Research 031/032) — the closest in-corpus instance of the class. §4 carries the full signal-diff; the verdict is UNCHANGED and STRENGTHENED (Q1 was already NO on published prior art; transformer-vm is a stronger local citation). The surviving workspace-level claim sharpens: Qwenputer's delta vs transformer-vm is precisely production-host-model + linear-attention-state RAM + adversarially-validated exactness — the bridge Issue 1155 prices. Amendment rode the user challenge per the standing rule; verdict-closing negotiation had consumed its round cap, and this edit adds prior art AGREEING with the recorded verdict rather than changing it.
- **P1 (unclaimed):** Issue 1155 G1a/G1b/G4/G3 cells — cheap, test-level, no kernel changes expected.
- **P2 (trigger-gated):** journal-interplay hypothesis cell inside a `gdn_delta_roll` posture; F1 design note if a durable-scratch consumer appears.
- **P3 (owner-gated):** GPU parity cell; ternary-quant survival probe; any F1 implementation.

## Resolution addendum (2026-10-29, zed-amo-i1155-exactslot)

**P1 + P2 LANDED — the hypothesis CONFIRMED.** Commit `cdeeda9` (develop):
`tests/issue1155_exact_slot_gdn.rs` (7 tests, required-features
`deltanet_inference`, journal mod under `gdn_delta_roll`) +
`tests/issue1155_exact_slot_g4_alloc.rs` (G4 counting-allocator own binary).

- **G1a PASS:** one-hot column + gate=1 + β=1 + grid payloads round-trip
  bit-exactly on the REAL kernel; non-addressed cells bit-unchanged; sweep
  (1,16,8)/(2,64,16)/(8,128,128); grid edges ±2^23/±2^24/0; readout
  bit-exact at K ∈ {16,64} (scale = 2^-2/2^-3, the power-of-two leg).
- **G1b PASS (must-fail direction):** a=1e8, v=1 collapses to exactly 0.0;
  overflow rows poison via inf·0 = NaN across kd−1 columns; random
  arbitrary-f32 battery mismatches > 0 asserted.
- **Decay selectors:** keep bit-stable; halve exact down to 2^-149 where
  2^-149·0.5 ties-to-even into 0.0, and odd-mantissa subnormals round to
  even (3·2^-149 → 2^-148); clear lands on +0 UNIFORMLY — x·0's ±0 sign is
  rewritten by the delta=+0 addend in the same step, so sign(x) is not
  observable post-step.
- **Journal hypothesis CONFIRMED (the headline):** keep/halve in-domain
  unwind to EVERY target restores BIT-EXACTLY (schedule [1,0.5,1] at the
  default eps 0.25; 8×0.5 at eps 2^-8, the gate passing AT equality).
  delta_roll law 2's "algebraically exact, NOT bit-exact" is REFUTED
  IN-DOMAIN — the reassociation rounds nothing when every intermediate is
  on the exact grid and every scaling a power of two. The law's general
  case is unchanged (out-of-domain still drifts). Law 1's fail-closed g=0
  refusal re-verified (Err + state bits untouched).
- **G4 PASS:** 256 writes + records + unwind, 0 allocations (release).
- **G3 HOLDS:** zero production source touched (two test files + one
  [[test]] block).

Issue 1155 file REMOVED (resolve rule); this addendum is the record.
P3 (wgpu parity + PQ2 survival) stays owner-gated/deferred.
