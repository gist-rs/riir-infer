# AGENTS.md — riir-ai

The global `~/.agents/` rules apply; this file documents repo-local context
that supplements them.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](../../../../../BOUNDARY.md) is the authoritative per-repo contract: what this
repo **owns**, what it **does not own** (with the correct home for each), the
crate-granular **allowlist** of what it may depend on, links to the cross-repo
rules' one canonical home, and the **drift ledger** of known gaps. On any
conflict with prose in this file, BOUNDARY.md wins.

- **Domain test:** does this serve a **game runtime** concern (NPC cognition, game state sync, perception, emotion, spatial queries, engine inference used BY games)? NO → it belongs in another repo; file there.
- **Read it before** adding any dep, crate, module, System impl, or vocabulary
  type — and before assuming a concern is yours to implement.
- **Enforcement** is not prose: `../riir-ai/scripts/ci_boundary_contract.sh`
  fails on an undeclared cross-repo dep, on a drift row without its open issue,
  and on a contract row that no longer matches the measured graph. Run boundary
  checks VIA the `boundary-guard` skill, not as ad-hoc greps.
- **Found a violation?** File the issue FIRST (`.issues/NNN_boundary_*.md`), add
  the drift row, then fix. Closing the issue removes the row in the same commit.

## Domain Boundary Rules (non-negotiable)

**This repo is for game runtime + engine ONLY.** It is NOT a catch-all for
any Rust project that happens to use the inference substrates. Every new
crate, module, or feature must pass the domain test:

### The domain test (ask before adding anything new)

> Does this code serve a **game runtime** concern (NPC cognition, game
> state sync, perception, emotion, spatial queries, chain-validated game
> events, engine inference used BY games)?
>
> - **YES** → it belongs in riir-ai.
> - **NO** → it belongs in a different repo. File it there.

### What belongs in riir-ai

- `riir-engine` — inference engine (transformer, MCTS, HLA, perception, cognition)
- `riir-games-*` — game domains (civ, quest, shared vocabulary, combat, zone)
- `riir-wasm` — WASM validator runtime (game on-chain validation)
- `riir-router` — inference routing (serves game cognition)
- `riir-gpu` / `riir-gpu-async` — GPU inference kernels (game cognition hot path)
- `riir-net` — adaptive transport (game multiplayer sync)
- `riir-simloop` — game simulation loop driver
- `riir-rest` — REST API for game runtime queries
- `riir-agents` / `riir-rag` — ONLY if the task graph / retrieval serves a game
  runtime concern. Generic developer-tool agents do NOT belong here.

### What does NOT belong in riir-ai

| Concern | Correct repo | Why not riir-ai |
|---|---|---|
| **Developer tools** (clippy healing, codegen, lint fixing) | `riir-clippy` | Developer tooling ≠ game runtime. Borrows substrate *patterns* but has zero game-domain coupling. |
| **Training research** (LoRA SFT, GRPO, distillation) | `riir-train` | Training ≠ inference. The modelless-first mandate means riir-ai ships inference substrates; training experiments live separately. |
| **Chain consensus** (block production, slashing, economics) | `riir-chain` | Chain ≠ game. Game events flow through the chain, but chain mechanics are not game logic. |
| **Storage** (shard persistence, KV store, retrieval index) | `riir-neuron-db` | Storage ≠ game. NeuronShard is the freeze/thaw vessel, not game state. |
| **Game SDK facade** (vocabulary re-export for consumers) | `riir-game-sdk` | The facade depends UPward into riir-ai; riir-ai has zero code-level dep on the SDK. |
| **Game product domain** (mmorpg examples, seal remaster) | `riir-mmorpg-examples` / `seal-online-remaster` | Product ≠ engine. Consumers plug game logic via SDK traits. |

### The "pattern borrowing ≠ dependency" rule

A new project may borrow an **architectural pattern** from a riir-ai crate
(e.g., the drafter+pruner+corpus pattern from `quest_grammar`) without needing
a **code-level dependency** on that crate. If the pattern can be reimplemented
in ~200 LOC and the new project has no other game-domain coupling, it belongs
in a **new repo**, not as a new module in riir-ai.

**Canonical precedent (2026-08-10):** the clippy-healing pipeline (Proposal
034) borrows the `TernaryDraftModel` + `SealQuestCorpus` + `ConstraintPruner`
pattern from `quest_grammar` but has zero game-domain coupling. It ships in
`/git/riir-clippy` (new repo), NOT in `crates/riir-games-quest/`. The prior
draft of Proposal 034 that proposed `crates/riir-games-quest/src/clippy_healing/`
was a boundary violation — corrected before implementation.

### The prior rejection (Proposal 017) is a different case

Proposal 017 proposed extracting `riir-games` entirely out of riir-ai into a
separate repo. That was correctly **rejected** because `riir-games` has heavy
**bidirectional coupling** with `riir-engine` (NPC cognition dispatches into
engine runtimes; engine exports game-shaped types). The clippy healer is
fundamentally different: it has **zero** coupling with any game crate.

### When in doubt: run the boundary-guard skill

The `boundary-guard` skill (`.agents/skills/boundary-guard/`) enforces these
rules via grep checks. Run it before adding any new System impl, game logic
module, or vocabulary type to a consumer/src/ directory.

## Repo Layout

This is a **Cargo workspace** (21 in-tree crates). `crates/riir-ffi` is
**excluded** from the workspace because it depends on the sibling
`katgpt-rs` via a path outside this workspace (see root `Cargo.toml`
lines 28-37). It still builds standalone via
`cargo build --manifest-path crates/riir-ffi/Cargo.toml --features ...`,
and other workspace members can depend on it via `path =`.

| Crate | Role |
|---|---|
| `riir-engine` | Core engine: transformer, MCTS, Fourier, HLA, DeltaMemory, frame-sampling, per-NPC cognition runtimes (CLR, KARC, Committed π, ARG, CWM, CGSP), `poincare_bridge/` (Plan 497; opt-in `poincare_imagination` feature — FINAL REFUTE on quality axis per `.benchmarks/497_poincare_defend_wrong.md` §4; Phase 4 G9 cost gate PASS per `.benchmarks/497_poincare_mcts_g9.md` — ~30× rollout speedup; Phase 5 G10 coverage gate PASS per `.benchmarks/497_poincare_g10.md` — 100% coverage + ~91× cache lookup speedup; architectural + latency wins stand, stays opt-in permanently) + **mop_runtime** (Plan 538; opt-in — reward-free per-NPC path-entropy policy over the zone-KG abstraction; G1–G4+G8 ALL PASS per Benches 680/681; stays opt-in on the no-default-consumer rule) |
| `riir-data` | ConvexTok training corpus + LP graph pipeline; replay-to-SSD |
| `riir-validator-sdk` | WASM validator SDK (CLI + event export) |
| `riir-cognition-sdk` | Cognition module author SDK (Plan 385 Phase 2); mirrors `riir-validator-sdk` layout for update-shaped `cog_*` exports |
| `riir-wasm` | WASM validator runtime (LEAP strategy, MUSE skill lifecycle) |
| `riir-router` | Inference routing (hydra budget, dynamic pair, embedding, turso cold) |
| `riir-gpu` / `riir-gpu-async` | GPU inference kernels (CubeCL, fused layer dispatch, LoRA training-method research consumers) |
| `riir-games-shared` | Shared foundational types extracted from riir-games (Plan 484 Phase 0) + **vocabulary substrate moved here from `riir-game-sdk` (Issue 002, 2026-07-16)**: `types`, `payoff`, `game_traits`, `zone` (minus `market_bridge`), `spatial`, `entity`, `tick`, `heightfield`, `rules`, `quant_expert_route`. Layer 0 of the 4-layer split — no deps on `riir-games-*`, no dep on the SDK (the vocabulary back-edge was removed by Issue 002). Two zone submodules (`manifold_bridge`, `sleep_time_reload`) + `game_traits::grudge_field` pull in `riir-engine`/`katgpt-core` behind feature gates. |
| `riir-games-civ` | Civilization game domain extracted from riir-games (Plan 484 Phase 1 / Issue 002): civ engine, NPC CLR, social, MAG, LEO civ goals/networks, salience gate, sleep-time catalog. Layer 1 of the 4-layer split — deps on `riir-games-shared` only |
| `riir-games-quest` | Quest game domain extracted from riir-games (Plan 484 Phase 2 / Issue 002): quest game logic + modelless quest grammar pipeline. Layer 2 of the 4-layer split — deps on `riir-games-shared` + `riir-games-civ` |
| `riir-games` | Game integrations (Layer 3 of the split): dungeon, NPC, combat, LEO, PlasmaPath, zone `market_bridge`, crowd MCGS, **orchard** (apple foraging, Issue 490), **swarm** (batched 1000-NPC `ForagerSwarmSystem` + `ForagerAi` height-aware attraction brain + **pet AI substrate** `swarm::pet` (`PetAiConfig` + `ActivePetState` + `follow_hero`, Issue 589) + pet roster/taming (promoted to substrate by Plan 537 T4.2/T4.3, 2026-08-14 — `swarm/pet_roster.rs` + `swarm/taming.rs`) + monster/combat + **swarm deliberation** (`swarm/deliberation.rs` — `SwarmDeliberationSystem<S: ThreatSource>` stuck-NPC escape search, Plan 537 Phase 5; Phases 1–3 honestly deferred on the single-consumer rule) + **CLR collective threat emotion substrate** `tick_swarm_emotions_collective` + `tick_swarm_emotions_collective_with_aura` composition; Plan 515 extraction ~2388 LOC, Issue 557; CLR POC Issue 588, composition Plan 019) + **demonstration-coverage curiosity** (`swarm/coverage_curiosity.rs` — opt-in `demo_coverage_curiosity`, Issue 672 / Research 479: consistency-cluster-triggered ONE-exemplar seeking via dot+sigmoid coverage-target bias; falsifiable A/B PASSed, Bench 674; `ApplicationRecorder` + `drop_level` (Issue 059, 2026-08-14) bridge live consumers — task-grouped outcome windows + repair-invalidation — consumed end-to-end by the demonstration-teachable pets live A/B in riir-mmorpg-examples, Bench 013: targeted 1.000 vs generic 0.667, 32/32 seeds; `DemonstratedSkill::covered_mask`/`from_covered_mask` + `PetEntry.taught_mask` + pet-roster **wire v2** (Issue 060, same day — Warm-tier persistence follow-up: taught ranks survive pet switch + re-login through the wire → WAL → STAR chain; the BLAKE3 commitment covers the mask)), **motivation/feeling** (latent emotion types, Proposal 022 / Issue 493), **npc::imagination** (`ImaginedHlaMCTS` — Plan 497 Phase 4 consumer wrapper around `imagine_hla_into`; opt-in `poincare_imagination` feature; cost-axis win only, quality ABANDONED). Depends on `riir-games-shared` natively (same workspace) — **no longer depends on `riir-game-sdk`** (Issue 002, 2026-07-16, removed the vocabulary back-edge). Re-exports `riir-games-shared`, `riir-games-civ`, `riir-games-quest` for back-compat. The SDK depends UPward on this crate (optional `orchard` + `swarm` features) and re-exports its systems; `riir-game-sdk` is the dev entry point for *consumers* (riir-mmorpg-examples, formerly riir-mmorpg-examples, seal-online-remaster), not a vocabulary source. |
| `riir-rest` | REST API (RAG prefetch, P2P jury, neural probe) |
| `riir-examples` | Bin examples (bomber, g_zero, go, percepta) |
| `riir-tools` | CLI tools |
| `riir-poc` | Private R&D crate for "defend-wrong" POCs and negative controls (Issue 356) |
| `riir-core-wasm` | WASM-safe gateway re-exporting katgpt-core's plasma_path primitives for edge/browser consumers |
| `riir-mcp-client` | GM→Backend control plane leaf crate (Ed25519 GmSigner + GmEnvelope trait + WebTransport client); Plan 364 |
| `riir-net` | Adaptive transport substrate (Plan 494, Proposal 028) — `NetworkTransport` trait + DTOs + deterministic `select_tier` picker + `LocalTransport` (mpsc, default-on) + `WsTransport` (feature `ws`) + **`BinaryTransport` trait (Issue 559)** for opaque byte payloads + **`WasmWsTransport`** (feature `wasm`, wasm32-only `web_sys::WebSocket` client). SDK facade re-exports trait + types only; ~440 LOC deduplicated from riir-mmorpg-examples. Phase 3 (seal migration) honestly CLOSED — see `crates/riir-net/.docs/PHASE3_AUDIT.md`. **Issue 005 (2026-07-22):** added the hero-state overlay protocol (`PeerHeroState` DTO + `ClientMessage`/`ServerMessage` envelopes). **Issue 570 Part B (2026-07-31):** deleted the `PeerHeroState` surface as dead wire residue — production avatar sync has been the chain-committed binary `AvatarStateDelta` (Issue 559 / Issue 024) since 2026-07-22; no producer emitted `PeerHeroState` + no consumer read it. The JSON text path now carries only `ClientMessage::Input` + `ServerMessage::Snap`. **Issue 559 (2026-07-22):** added `BinaryTransport` trait (SRP split from `NetworkTransport`) + binary WS message support on `WsTransport` + `WasmWsTransport` for wasm32 — the wire layer for chain-committed `AvatarStateDelta` bytes.  **Issue 667 (2026-08-14):** WebTransport is the PRIMARY transport — `webtransport` feature (wtransport 0.6) ships `WebTransportTransport` (BinaryTransport client, length-prefixed uni-stream framing) + `hub::RelayCore` (transport-agnostic registry, Bytes currency) + `spawn_webtransport_listener` / `spawn_validated_relay_server_with_core` (dual-listen: TCP=WS + UDP=WT on the same port number, ONE shared core so WS + WT clients mirror to each other — the authority + native player use this; browser stays WS until P3 `web_sys::WebTransport`, Unity until P4 FFI). P1+P2 DONE (G1 byte-identity + G3 no-regression PASS; G2 netem + G4 datagram-path deferred to P5). **P3 DONE (same day):** `WasmWtTransport` (wasm32 `web_sys::WebTransport`, same framing; needs `--cfg web_sys_unstable_apis`) + `tier_fallback_chain` (auto-adaptive `[select_tier, WS, Local]` — consumer override `RIIR_MMORPG_EXAMPLES_TRANSPORT` / `?transport=`) + `GET /wt-cert-hash` (SHA-256 cert-DER pin the browser shell exports as `window.RIIR_WT_CERT_HASH` before wasm init; static-only hosts miss → WS, the intended CF fallback). ONE `webtransport` feature on both targets (deps resolve per-target; tokio/axum/wtransport moved to `[target.'cfg(not(wasm32))'.dependencies]`). **P3-fix (same evening, found by automating the browser badge test with Playwright — consumer `scripts/browser-badge-test/`):** a WT session constructed from the WASM call stack NEVER DIALS in Chromium (CDP: `webTransportCreated` but never `ConnectionEstablished`; options identity / construction timing / pending stream+datagram readers / option-object GC keepalives / launch flags all ruled out by a 9-probe isolation ladder) — fix: the page shell pre-constructs the session in pure JS after fetching the pin (`window.RIIR_WT_SESSION`, same-origin only) + the wasm adopts it via `WasmWtTransport::adopt` (`connect` self-construction kept for shell-less hosts; wasm handshake budget 1500 ms → 5 s). Browser e2e now Playwright-asserted end-to-end (`connected via WebTransport` console line + screenshot). **P5 SEND-SIDE DONE (same day; G2 netem deferred — needs Linux):** `BinaryTransport::publish_lossy_owned` (lossy-ELIGIBLE hint; reliable transports deliver reliably — strict upgrade, never an obligation) + `hub::OutboundFrame { data: Bytes, lossy: bool }` (fan-out preserves the delivery class; reliability never downgraded in transit — datagram-origin frames ride lossy, WS/stream-origin fan out reliable) + native client/relay + wasm send/receive datagram paths (fits `max_datagram_size` → raw datagram, no length prefix; oversize/error → length-prefixed stream fallback — an error is not a loss event) + `WireRouter::publish` magic routing (ZDLT zone → lossy; AVTD/IVTD/QLTD/WLTD/TRPP/TRCO/TRJR/STAQ/STAR/PRTD chain-committed → reliable stream; the SAME first-4-byte peek as `route()`, so publish + route can never disagree on a frame's class) + the authority connects to its own relay WT-first (without which the sole zone-delta publisher rode WS + the datagram path stayed dormant) + `WT_CONNECT_TIMEOUT` 5 s (UDP-blackhole handshake cap; `connect_with_timeout` test variant). 5 new tests. CF verdict: WS-over-edge is the only viable CF ingress; WebRTC is NOT a CF tier (Workers = Realtime/Calls media SFU only, no raw UDP; Containers get no public UDP either) — `WebRtc`/`P2pMesh` stay Phase-4 browser-P2P tiers, no impls (consumer sets `multiplayer: false` — hub-relay topology). CF e2e matrix + numbers: riir-mmorpg-examples `.benchmarks/012`. **G2 netem DONE + stream-serialization bug fixed (2026-08-15, Bench 676):** G2 measured via Docker Desktop's Linux kernel (`tc netem` on `lo` — macOS can't run netem but the container kernel can; probe `crates/riir-net/examples/netem_g2_probe.rs` + `scripts/g2_netem_docker.sh`) — **PASS: WT datagram p99 50.6 ms vs WS 288.2 ms under 1% loss (5.7×)**, 16 KB stream class 156 vs 950 ms (6.1×). The measurement caught a bug localhost hid: both WT outbound loops awaited each frame's stream lifecycle inline → ~1-2 RTT per frame serialized (5.5 s p50 backlog for reliable 512 B at 40 ms RTT). Fix: pipelined stream writes via spawned tasks bounded by `Semaphore(MAX_INFLIGHT_STREAM_WRITES = 64)` — the io_uring SQ-depth pattern (Research 323 vocabulary); reliable queues FIFO, lossy drops at submit when saturated, permit release = the CQE equivalent. **WS `TCP_NODELAY` follow-up DONE (same day, Bench 676 addendum):** client (`NativeWsTransport::connect` via `set_client_nodelay` — covers ws:// + wss://) + all four axum accept paths via axum's built-in `serve::ListenerExt::tap_io` — WS 0%-loss 512 B p50 93.1 → 47.92 ms (parity with WT), 1%-loss p99 288.2 → 193.5 ms; **G2 headline honestly recalibrated to 3.8×** (the 5.7× included the now-removed Nagle handicap). Remaining follow-ups in Issue 667: persistent-stream framing, Chromium upstream report (wasm writer pipelining audit PASSED 2026-08-15 — the browser send path already spawns per-frame `spawn_local` tasks bounded by `MAX_IN_FLIGHT_SENDS = 32`, no serialization to fix). **Static cache fix (2026-08-19, `27388728a`, mmorpg Issue 076 half):** both relay routers now layer `Cache-Control: no-cache` on every response — without it browsers applied HEURISTIC freshness to the ServeDir-served wasm bundle + JS glue and silently kept running a stale build after a rebuild; store-but-must-revalidate turns rebuilds into cheap 304s (ServeDir emits Last-Modified). Harmless on the WS handshake + `/wt-cert-hash`. Consumer-side half (the `?v=<sha>` cache-bust + build-version UI): riir-mmorpg-examples `ba8f6d1`. |
| `riir-agents` | Generic latent-space task reasoning agent (Plan 523). Operates on any task graph (TODO / training pipeline / game management / code refactor) via `dot(task_dir, context_dir) + sigmoid` latent scoring + strategy selection. Modelless baseline (Phase 1, default-on, permanent) constructs directions via BLAKE3-deterministic embedding. **Opt-in `gemma2_directions` feature (Phase 2)** loads Gemma 2 2B GGUF locally + extracts hidden-state embeddings as direction vectors — GOAT verdict **FAIL** per ``.benchmarks/571`` (v3, absolute gate): Gemma2 reads real semantic signal (overrides misleading priority metadata directionally) but not strongly enough to be correct (ranks a JSDoc pass above a critical security vuln); ~9 s/embedding is also unaffordable on a per-tick hot path (refutes Plan 523's "~100× faster" claim — measured speedup is ~1.4×). Stays opt-in as a reproducible negative-result artifact (mirrors Plan 410's Go-arena failure). Phases 3–5 deferred pending concrete consumers. **Opt-in `shared_context` feature (Plan 529, DeLM distillation — Research 337; implies `multi_agent`)** — async-claim verified shared-context coordination: `AsyncSharedContextCoordinator` (atomic CAS claims fresh `Ready→InProgress` / retry `Failed→InProgress`, barrier-free `run_async`), `SharedContextEntry` POD + modelless admission gate (verbatim ref-tag anchor match + BLAKE3 commitment recompute, zero LLM) + `SharedContextLog` (ArcSwap lock-free reads, encrypted Warm-tier write-through), Sheaf-ADMM conflict resolution via `run_consensus_over` (conflict winner admitted as an `EntryKind::Constraint`), ExperienceGraph avoid-set derivation (Failure entries → dual-stream TD projection → binding constraints per dispatch). **GOAT G1–G5 ALL PASS (Phase 6)**: failure-reuse 1.000 on 12/12 seeds (zero avoidable duplicate falls) vs round-barrier 0.870 driven by the real consensus coordinator; admission p99 9.9µs release; exactly 2 allocs/admit (G4 caught + fixed a per-admit growth-realloc bug). Default-off pending Phase 5 (distributed chain ingestion — riir-chain read-only for this workspace). |
| `riir-rag` | Latent-first retrieval facade for `riir-agents` (Plan 524 Phases 1-6 DONE; Phases 7-9 BLOCKED on Plan 318 C13). Composes the neuron-db substrate (`ShardIndex::retrieve_diverse` Clifford-wedge KNN + `Bm25Index` exact symbol + `LocalKvStore` chunk content) behind a pluggable `Chunker`/`Embedder`/`RetrievalProvider` trait surface + a `TokenBudgetPacker` that fits retrieved context into 256K. Three modelless retrieval axes: **latent** (8-D vector KNN, default), **lexical** (BM25 exact, default), **graph** (KG triple k-hop, opt-in `graph_rag`/`code_triples` — Plan 526 / Plan 325 Phase 4). Default `ModellessEmbedder` is deterministic (BLAKE3 + DFT + sigmoid, no weights). GOAT G1/G2 (**123µs** p50 @ N=1000, 1.6× headroom)/G3/G4 PASS modellessly (Bench 575); **G5 GraphRAG PASS** (Plan 526 — hybrid finds transitive callers at 2 graph hops with zero lexical overlap that pure latent+BM25 misses). `graph_rag`/`code_triples` stay opt-in (real dep cost: kg_triple + syn ~2MB); `kimi_k3_embedder`/`dense_two_stage` blocked on C13. See ``.docs/02_crates/riir_rag.md``. |

> **Moved to `riir-game-sdk` repo (2026-07-16):** `riir-viz` (Bevy visualization)
> and `riir-gm-tool` (GM tool binary) were moved to the `riir-game-sdk` repo
> to eliminate the back-edge (`riir-viz → riir-game-sdk` for GM dashboard
> types). The SDK repo is now a workspace: root facade crate +
> `crates/riir-viz` + `crates/riir-gm-tool`. riir-ai has zero code-level
> dependency on the SDK repo.

`riir-chain` and `riir-chaind` were **spun off** to their own repo
(`/git/riir-chain`). Consumers (riir-games, riir-examples, riir-game-sdk repo)
reference the chain lib via sibling path dep. See
`riir-chain/.plans/001_chain_spinoff.md` and
`riir-chain/.plans/002_chaind_spinoff.md`.

## Sibling-Repo Layout

Path dependencies assume this on-disk layout:

```
/git/riir-ai          ← this repo (workspace)
/git/katgpt-rs        ← katgpt-core (non-optional), katgpt-rs (optional)
/git/riir-chain       ← riir-chain (lib), riir-chaind (daemon)
/git/riir-neuron-db   ← leaf crate (re-exported by riir-chain)
/git/riir-train       ← training-method research (Issue 004 split)
/git/riir-game-sdk    ← workspace: facade crate + crates/riir-viz + crates/riir-gm-tool (Issue 002 + viz/gm-tool move 2026-07-16); dev entry point for consumers (riir-mmorpg-examples, seal-online-remaster)
```

If you move this repo, update path deps in every `Cargo.toml` that
references `../../../katgpt-rs`, `../../../riir-chain`, `../../../riir-game-sdk`, etc.

## Build Commands

```bash
# Workspace baseline (default features — the prod path)
cargo check --workspace

# Every feature at once — catches combo-only regressions
# (the `merkle_root` / `can_freeze` lesson class applied to a workspace)
cargo check --workspace --all-features

# Full CI guard (mirrors sibling repos, plus the Lean 4 proof gate)
./scripts/ci_feature_guard.sh

# Single crate
cargo check -p riir-engine --lib
cargo test -p riir-engine --test hla_bounds_spec_match
```

The workspace's feature matrix is too large for `cargo hack --each-feature`
(riir-engine alone has ~50 features). The CI script substitutes
`--all-features` for the combo-regression check — see
`scripts/ci_feature_guard.sh` for rationale.

## Static vs Dynamic Verification Split (the FV instance)

`riir-ai` carries the **fourth** Lean 4 formal-verification instance in the
7-repo stack: `riir-ai/.proofs/RiirAiProof` (Plan 353, Mathlib-required,
toolchain `leanprover/lean4:v4.32.0-rc1`).

### Why this repo gets *fewer* proofs (not more)

`riir-ai`'s selling point is **dynamic**: self-play, all-goals learning,
collapse-aware recovery, curiosity-driven exploration. These are runtime
behaviors validated empirically against real game sessions — Lean is a poor
fit. **Do not attempt to prove dynamic properties.**

The proof-able subset is the **static invariants the runtime depends on**
but doesn't produce:

| Invariant | Status | Where |
|---|---|---|
| **HLA scalar boundedness** — every sigmoid-derived scalar is in `(0,1)`; every clamped emotion scalar is in `[0,1]` | ✅ DONE (Plan 353) | `.proofs/RiirAiProof/Hla/Bounded.lean` (14 theorems) |
| **Freeze/thaw reader invariant** — readers never observe a torn adapter snapshot (memory-model theorem) | ✅ DONE (Issue 348 T2, 2026-06-30) | `.proofs/RiirAiProof/Runtime/FreezeThaw.lean` (2 theorems; depends only on `propext` after the Issue 354 torn-read fix made the invariant hold by construction) |
| **Bridge ordering on learned directions** — extends public sigmoid-monotonicity to private tuned direction vectors | ✅ DONE (Phase 5, 2026-06-30) | `.proofs/RiirAiProof/Phase5Specialization.lean` — covered by specialization of the public `action_bridge_ranking_preserved` (universally quantified over arbitrary directions; no new theorem needed). Cross-repo FV rollout complete; coordinator issue `katgpt-rs/.issues/012` removed (commit `a357e8a1`) |
| **HLA spec self-tests on concrete instances** — closes the spec-authority gap in the C3 FV pattern (Plan 441 convention) | ✅ DONE (Plan 490, 2026-07-15) | `.proofs/RiirAiProof/Hla/SpecTests.lean` (14 `example` proofs: 3 dot + 3 sigmoid + 5 clamp01 + 3 curiosity_drive_raw — the repo-specific extension pinning Rust-transcribed blend weights). Runtime deliberately skipped (structural/abstract, no external authority). GOAT G1–G4 PASS; axiom inventory unchanged |

### What's modeled vs what's assumed

- **HLA boundedness** is modeled over `ℝ` (infinite precision). The f32
  implementation saturates near `|x| > 17` (rounds to exactly `0.0` or
  `1.0`); the Lean theorem's strict `<` bound holds over `ℝ`, and the
  paired Rust spec-match test (`crates/riir-engine/tests/hla_bounds_spec_match.rs`)
  handles the f32 precision regime empirically with two complementary
  checks (strict `(0,1)` on `[-10,10]`, closed `[0,1]` on `[-50,50]`).
- **NaN handling** in Rust's `clamped()` (NaN → 0 before clamp) is **not
  modeled** in Lean — `ℝ` has no NaN. The spec-match test is the sole
  validator for that path.
- **Memory model** for freeze/thaw — Lean doesn't ship a default weak-memory
  model, so T2 ships under a sequential-consistency atomicity axiom
  (`arcswap_store_atomicity` in `Basic.lean`) rather than modeling ARMv8/rcpc
  literally. The Rust code uses weaker orderings; the
  `concurrent_lora_no_torn_read` stress test (100K iterations, shipped with
  the Issue 354 fix) guards that gap empirically. This is "option C" from
  the original Issue 348 §5 A/B/C trade-off; the issue file was removed once
  all eight tasks closed (commit `97902de7`).

### Adding a new theorem

If you add a new sigmoid-derived scalar to the runtime:

1. Add a matching theorem in `.proofs/RiirAiProof/Hla/Bounded.lean`
   mirroring `emotion_scalar_via_sigmoid_bounded` (A4) — the proof is one
   line (`sigmoid_bounded (dot q d)`).
2. If you add a field to `NpcEmotionScalars`, add a matching
   `clamped_<field>_bounded` theorem (mirror B2–B6) and extend
   `clamped_all_bounded`.
3. Run `cd .proofs && lake build` — must pass with no `sorry`.
4. Run `cargo test -p riir-engine --test hla_bounds_spec_match` — must pass.
5. Verify the axiom inventory: `cd .proofs && lake env lean PrintAxioms.lean`
   — must be `{propext, Classical.choice, Quot.sound}` only.

See `.proofs/README.md` §"Regenerating after Rust changes" for the full
protocol.

### Cross-repo FV coordination

- **Public sibling (extends):** `katgpt-rs/.proofs/KatgptProof` (Plan 293) —
  proves sigmoid ranking preservation on public direction vectors.
  `RiirAiProof` adds the boundedness complement and (Phase 5) extends
  ranking to the private tuned direction set.
- **Chain sibling:** `riir-chain/.proofs/RiirChainProof` (Plans 004 + 008 + 009).
- **Neuron-db sibling:** `riir-neuron-db/.proofs/NeuronDbProof` (Plans 007 + 008).
- **Coordinator:** `katgpt-rs/.issues/012_*` was **removed** (commit `a357e8a1`) once the cross-repo FV rollout completed — all four repos' proof instances are landed and their invariant tables are self-documenting.
- **This repo's issue tracker:** `riir-ai/.issues/348_*` was **removed** once
  all eight tasks (T1–T8) closed DONE on 2026-06-30 (commit `97902de7`). The
  FV instance is fully landed — see the invariant table above and
  `.proofs/README.md`.

Per coordinator rule C4: **private proofs stay private.** Lean files in
`riir-ai/.proofs/` are internal-only. Do not cross-port them into the
public `katgpt-rs` repo, even as "reference".

## GPU exclusivity during correctness gates (MANDATORY)

**Do not run a numerical correctness gate while another process is using the
GPU.** Check first; if a sibling agent is running GPU work, wait or coordinate.

This is not hygiene, it is a measured correctness-of-measurement requirement.
`Bench 649` measured, on one machine in one
process with a returning-idle control:

| GPU condition | prefill divergence rate |
|---|---:|
| idle | 3/74 = **4.05%** |
| second process saturating it | 25/50 = **50.00%** |

That is a **12.3× swing, χ²(1)=36.03, p≈2×10⁻⁹**, in a *correctness* rate — not
throughput — driven entirely by an unrelated process. Issue 642 separately
measured the throughput half at 14.06 → 0.16 tok/s.

**What it invalidates.** Six consecutive Issue 640 experiments (Benches 643–648)
produced mutually contradictory conclusions because each was measured against a
base rate that a *different process* was setting. Interleaving, control arms and
power gates were all added in response and none could have found the cause,
because the cause was outside the process being instrumented.

**How to apply.**

- **Unity Editor + Zed are EXEMPT from the exclusivity check** (owner's call,
  2026-08-14). An idle/lightly-active GUI editor does not saturate the GPU the
  way a second compute workload does — Bench 649's 50% divergence regime was
  measured with a second cargo process *saturating* the GPU, not an editor
  holding a window compositor. Do NOT block gate runs on Unity/Zed being open;
  do not defer correctness gates for them.
- The gate that REMAINS: before a gate run, `ps aux | grep -E "cargo|riir|katgpt" |
  grep -v grep` and check for another **compute** consumer (sibling cargo test,
  inference run, training job). `uptime` load average is **not** a proxy — the
  race fired at load 4.3 and stayed quiet at load 9.3.
- Record GPU-exclusivity in the benchmark doc alongside the numbers, the same way
  load average is already recorded ("Unity/Zed active" is acceptable + should be
  noted, per the exemption above).
- If exclusivity cannot be guaranteed against a *compute* workload, say so in
  the doc and treat the result as provisional. A gate measured under compute
  contention is not evidence.
- **Issue 711 RESOLVED (2026-08-16, same day): the kimi_k3 GPU backward
  "DX12 divergence" was a MISSING DEVICE FEATURE — `Features::IMMEDIATES`**
  (wgpu-30 gates pipeline-layout immediates, i.e. WebGPU push constants,
  behind it; cubecl-0.11's SPIR-V path passes kernel scalar params that way).
  riir-gpu's own `GpuContext` (crates/riir-gpu/src/context.rs) creates the
  wgpu device requesting only `SUBGROUP | PASSTHROUGH_SHADERS` and hands it
  to cubecl via `WgpuSetup` — every immediate-carrying pipeline creation
  panicked on a DSD dispatch thread and the launch was silently dropped
  (output stays zero / stale → the 2e-2..9e-2 "divergence" and the
  ~850 IMMEDIATES panics/run). Fix: request `IMMEDIATES` in the same
  adapter-features intersection. Post-fix (GPU-exclusive): kimi lib 7/7,
  sequence integration green, weaver 26/26, gemv 80/80, Layer 1.8 green on
  this box. **Backend truth: the 4090 runs cubecl through VULKAN/SPIR-V,
  not DX12** (wgpu `request_adapter` picks Vulkan for the 4090; the
  "DX12" attribution in the 2026-08-16 filing was wrong) — and note
  wgpu-hal only enables `VK_KHR_buffer_device_address` for ray-tracing
  features, so the custom BDA storage path in cubecl-wgpu runs on a
  BDA-less device here (upstream cubecl expects its own ash device
  creation for that path). The pre-migration bisect (junction worktrees
  at `709686047^`, de-vacuated harness) confirmed pre-migration GREEN at
  fp-noise (3.7e-9) on this box — the migration exposed the feature gap,
  it did not corrupt numerics. **The former Issue 712 follow-up is RESOLVED**
  (see the 712 row below — the alloc cliff). **Remaining follow-up: Issue 714**
  — one Vulkan "Parent device is lost" event kills the late suite tail; exists
  at clean HEAD (not the 712 fix — bisect matrix in the issue file). libtest CAPTURES stderr
  on passing tests — use `--nocapture` when tracing GPU allocation probes.
  `GEMV_FORCE_TILED=1` remains a valid workaround if plane benchmarking
  needs suppressing, but is no longer needed for correctness.
- This applies to every numerical gate in the repo, not just Issue 640 — any G1
  tolerance check, any bit-identity claim, any CRPS/coverage number.

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused** — even after a file is removed per the noise-reduction rule. Before
creating a new `.issues/` file, read `.issues/.highwater`, use `value + 1` as
the number, and write the new value back. The same rule applies to `.plans/`,
`.docs/`, `.benchmarks/`, and `.research/` — never recycle a number that git
history shows was already allocated.

**Collision precedent (Bench 675 + Bench 677, 2026-08-15 — TWO collisions same
night):** a stale-`.highwater` read dual-allocated 675 within 9 minutes (and
regressed the committed highwater 676→675). Resolution: renumber the file with
fewer live references (MOP → 677, the free gap), keeping the one whose number
is baked into a committed test filename + active consumer WIP (npc_episodic).
**Second collision:** 677 was then dual-allocated too — a concurrent session
wrote `677_episodic_curiosity_salience_gate_goat.md` (02:03) after the MOP
677 renumber landed (01:21), using a number cached at plan time and never
re-verified. Same tie-breaker applied: episodic keeps 677 (baked into
`bench_677_episodic_curiosity_goat.rs` + source comments), MOP moves to 679.
Rules: (1) before allocating a number, `ls` the target folder — `.highwater`
can be stale or regressed by a concurrent writer; (2) **re-scan the folder at
WRITE time, not just at allocation time** — concurrent agents commit during
long implementation phases and a cached allocation goes stale in under an
hour.

**Third collision (Bench 697, 2026-08-19):** dual-allocated across a rebase —
the 721 session cached 697 at plan time while the 726 T1 session landed
`697_issue726_t1_gpu_prefill_baseline.md` on origin first; the 721 session's
rebase over the sibling push integrated the file WITHOUT the collision being
noticed (allocation-time and write-time scans both predated the rebase).
Resolved per the tie-breaker: 721 keeps 697 (baked into the committed test
filename `bench_697_issue721_tree_g2.rs`), the 726 T1 doc renumbered → 698.
**New rule: re-scan the folder after every REBASE too** — a rebase imports
concurrent allocations exactly like a fresh write.

## Resolved + removed issues (2026-08-15 hygiene pass)

The following issue files were verifiably RESOLVED/CLOSED and removed per the
noise-reduction rule. Historical citations to these numbers remain valid (they
name a finding, not a path). Load-bearing verdicts preserved:

| Issue | Resolution + where the record lives |
|---|---|
| **606** ternary GEMV roofline | RESOLVED/SUPERSEDED — `launch_rowtiled8` is the `launch()` default on both backends; dp4a + cudarc answer the CUDA leg; Issue 628 closed the rest. Record: Bench 606 + ``.docs/09_performance/ternary_gpu_forward.md``; row in Issue 629's backlog table. |
| **608** dp4a CUDA kernel | RESOLVED — system thesis CONFIRMED on WDDM (CUDA Graphs 76.4 tok/s = 96.6% of llama.cpp, Bench 667); `cuda_graphs_forward` DEFAULT-ON (`46a2eca60`). |
| **613** match llama.cpp Metal ternary throughput | RESOLVED — T1/T2/T3 all DONE (Bench 610/611/606 §G5b; cudarc path Bench 639); Metal GEMV at ~23% roofline = llama.cpp parity. |
| **640** batched RMSNorm prefill nondeterminism | RESOLVED — root cause = cross-process GPU contention (Bench 668); encoded as the AGENTS.md GPU-exclusivity rule above. |
| **651** why not copy prismml shaders | RESOLVED — three walls (2 structural + Wall 3 answered by measurement: 656 0.87×, 657 0.95×, 663/670 zerocopy refuted at production scale). **Canonical record moved to ``.docs/09_performance/metal_parity_verdicts.md``.** |
| **659** decode-path stage breakdown | DONE via 661 T1 — `DECODE_STAGE_MASK` substrate + Bench 669 (FFN 32.3% / recurrence 5.2% / attention 3.6%). |
| **660** Metal ICBs via wgpu fork | CLOSED-DEFERRED — T1+T2 shipped (bind-group cache in `vendor/cubecl-wgpu-0.11.0-pre.2/`); ICB deferred at 1.3% ceiling (Bench 670). Revisit trigger: prefill with thousands of dispatches/token. Record: `metal_parity_verdicts.md`. |
| **661** DeltaNet recurrence kernel | CLOSED-DEFERRED — 5.2% ceiling; rowpar kernel already shipped (5.78×, Issue 619). Record: `metal_parity_verdicts.md`. |
| **662** attention decode kernel | DEFERRED — 3.6% ceiling at short context (Bench 669). Record: `metal_parity_verdicts.md`. |
| **664** cross-op fusion | CLOSED-DEFERRED — negative result measured twice (Bench 670); shipped-fusion list preserved in `metal_parity_verdicts.md`. |
| **667** WebTransport primary transport | RESOLVED — every claim re-verified 2026-08-15. Record: AGENTS.md §`riir-net` + riir-mmorpg-examples HISTORY.md §Issue 667 follow-ups + Bench 012. |
| **679** optimizer shared-uniform truncation | RESOLVED — the CRITICAL AdamW bug fixed in the original commit; B1-B10 fixed in the 2026-08-15 pass. Record: riir-clippy AGENTS.md Batch 31 + regression test. |
| **683** edge_lora adaptive-width construction storm | RESOLVED — all fixes landed; 122 edge_lora lib tests + 22 bench_298 gates pass. Record: riir-clippy Research 034. |
| **684** edge_lora topology-trainer hot-path allocs | RESOLVED + T3/T11 CLOSED (2026-08-15, `b5ec1903d`) — T11 probe (`probe_684_topology_hotpath_alloc`, katgpt_core TrackingAllocator counters) initially FALSIFIED the "zero allocs/step" claim: 64.3 allocs/step remained (the per-call `edge_layers_and_offsets()` 2-Vec classification × forward+backward). T3 root-cause fix landed: `topology` field private + construction-time routing cache — measured 80.3 → 64.3 → **0.35 allocs/step** (1758 → 286 → 30 B/step), final_loss bit-identical; 648 edge_lora lib tests + 22 bench_298 gates. Deferred `[-]`: T9 adapter-skeleton extraction, H9 O(W·H) copy. Record: riir-clippy Research 035. |
| **685** edge_lora self-play centroid storm | RESOLVED — H1-H4/H6-H10/T8 fixed. Deferred `[-]`: T9 adapter-skeleton extraction (~200+ lines ×3, worth its own plan with the 684-T3 CSR follow-up); H9's O(W·H) copy until a borrowed-view ArenaGrid. Record: riir-clippy Research 036. |
| **686** spectralquant calibration hazards | RESOLVED — 12 of 15 hazards fixed (Batch 33 mining yield, bench-only cluster): H1 multi-tile covariance silent corruption deleted wholesale (`n_tiles` + `dispatch_covariance_reduce` + `covariance_reduce.wgsl` gone — the path can no longer be expressed), H2 `layer_offset` wired into the eigenbasis kernel base (was dead — multi-layer rotation read layer 0's data), H3 `total_layers` dead field, H5 `create_uniform_buffer` doc-lie, H7 covariance.wgsl smem staging + 2 barriers/position removed (**runtime-validated on the 4090 2026-08-16** — covariance + eigenbasis bench groups green), H9-H12/H14/H15 dead-code + tautological-test + doc-example fixes, H13 multi-dispatch shared-uniform hazard documented, H10 WONTFIX (bounds guard genuinely reachable for non-divisible d_h). Deferred `[-]`: H4 prebuilt bind groups + H6-full constructor field de-drift (same API ripple through 3 constructors; revisit when the `calibrate()` wiring lands), H8 triangle+mirror covariance (optimization opportunity, not a bug). Record: riir-clippy Research 037 + git history. |
| **696** CUDA Graphs LoRA decode capture | DONE/CLOSED — all gates PASS, commit `92065e0b6`, 77.32 tok/s bit-identical. Record: `Bench 674`. |
| **699** hypernet meta_lora OOB hazards | RESOLVED — H1+H2 (silent heap corruption) fixed + regression tests; H3-H5/T2 closed. Deferred `[-]`: T1 FD clone storm (`LossPrelude` hoist) — reopen if the CPU FD trainer gets a consumer. Record: riir-clippy Research 050. |
| **566** seal-edge-worker Alarm reliability | DEFERRED (owner call) — riir-side container fix shipped + e2e-validated (Bench 012); seal-specific recheck deferred until `seal-online-remaster` is checked out. Record: Issue 629 pass-2 table + `architecture.md` §22. |
| **676** CubeCLContext once-per-process | RESOLVED — OnceLock shared context + regression test (`e83918fe7`); verified GPU-exclusive 526/0/1 (was 401/119). Record: Issue 629 pass-2. |
| **678** dllm GPU test flakes | RESOLVED — de-flaked asserts (`269ce50e3`); verified 8/8 + 4/4. Record: Issue 629 pass-2. |
| **687** gemma4_cubecl_train hazards | DEFERRED-PARTIAL (gemma out of focus) — correctness guards + docs landed (`0f83071df`); restructures deferred. Record: Issue 629 pass-2 + git history. |
| **688** gemma2_cubecl_train hazards | DEFERRED-PARTIAL (gemma out of focus) — doc/assert/robustness fixed (`0f83071df`); perf refactors deferred. Record: Issue 629 pass-2. |
| **689** gemma2_resident RoPE recompute | DEFERRED-PARTIAL (gemma out of focus) — H1 headline fix landed both paths (`0f83071df`); readback refactors deferred. Record: Issue 629 pass-2. |
| **690** gemma4_q4k_train hazards | DEFERRED-PARTIAL (gemma out of focus) — docs + dead code + transpose fixed (`0f83071df`); per-step restructures deferred. Record: Issue 629 pass-2. |
| **691** gemma4_q4k unreached backward | DEFERRED-PARTIAL (gemma out of focus) — trivial items landed (`0f83071df`); the 941-LOC backward remains unwired. Record: Issue 629 pass-2. |
| **692** kimi_k3 forward hazards | DEFERRED-PARTIAL (kimi out of focus) — dead 640 MB embed deleted + doc/assert fixes (`03b2e4d5e`); clone storms deferred. Record: Issue 629 pass-2. |
| **693** kimi_k3 backward wrong-math + NaN-vacuous parity | DEFERRED-PARTIAL (kimi out of focus) — **CRITICAL H1/H2 wrong-math FIXED + falsified against the de-vacuous parity gate** (`03b2e4d5e` + katgpt-rs `380c3556`); CI Layer 1.8 pins it. Numerics remainder deferred. Record: Issue 629 pass-2. |
| **694** kimi_k3 sequence weight-cache hazards | DEFERRED-PARTIAL (kimi out of focus) — orphaned-write hard-panic + Ouro path fixed (`03b2e4d5e` + riir-train `30a73d12`); zombie lm-head deferred. Record: Issue 629 pass-2. |
| **695** gemma2_d2f decode hazards | DEFERRED-PARTIAL (gemma out of focus) — quick wins fixed (`a4512200c`); block-forward rework deferred. Record: Issue 629 pass-2. |
| **702** l2_normalize suite-order divergence | RESOLVED-BY-CLOSE-CONDITION (not root-caused) — 2026-08-16: serial full-suite PASS (refutes order pollution) + 3× default-parallel GPU-exclusive PASS at the failure-era population (786/811/817 s), plus the independent 526/0/1 in the 676 row → 4-5 green full-suite runs vs the single 2026-08-15 observation. Race hypothesis + stale-params-buffer signature preserved in the removed issue file (git history); reopen on any recurrence; never loosen the 1e-5 tolerance. Gate-coverage fix shipped: `ci_feature_guard.sh` Layer 1.10 pins the `cubecl_runtime` lib-test compile + `deltanet_cubecl` module (12 tests, ~1 s) — the class is no longer CI-invisible. Discharges the 676 row's outstanding-failure caveat. |
| **700** maxsim unchecked contracts + substrate-first CPU fallback | RESOLVED — all 14 findings fixed/noted (`d5ab08429`+`f943bd49d`): release-checked contracts (compressed_k/ld/bitstream_offset/d_eff/max_bits), +1 pad word host-side, substrate flip to `katgpt_core::simd::maxsim_score` (`maxsim = ["katgpt-core/maxsim"]`), `download_f32_reuse` staging, eigenvalue-ordering contract + tripwire, docs. **The re-pin headline finding (2026-08-16 GPU window): the RELEASE sweep puts the crossover at ~128k units (CPU SIMD ~0.12 µs/unit vs ~1.6-2.2 ms GPU per-call floor; GPU only 1.18-1.23× at scale) — the scalar-era 256 was ~3 orders off; `DEFAULT_MAXSIM_THRESHOLD` re-pinned 256 → 131_072** with profile/dim/dispatch-shape caveats on the constant. Debug sweeps are meaningless for it (crossover ~340). Remaining deferrals are production-caller-conditional (item 13 calibration-buffer caching, `score_chunked`, finder polish). Record: removed issue file (git history) + the threshold doc on the constant. |
| **701** hypernet second-half wrong out-proj + OOB + infeasible defaults | RESOLVED (fix set in `8bb716954`, docs `1c5a45dd6`) — P0 H-C wrong out-projection (`attn_val × rowsum(wo)`) fixed via the two-phase decomposition (per-head scratch + `m2p_out_projection` full-row kernel) in BOTH WGSL arms + `cpu_m2p_attention` (incl. the `wo_row` head-local bug); discriminating parity test vs `M2PTransformer::self_attention` under random weights (negative control fails at 0.1476). H-A heap scratch (no head_dim cap on CPU), H-B release-checked WGSL-array caps (128/256), H-H device-limit dispatch check, H-J single-source provision (default grid 251.3 GB → ~90 MB; pins 425,984→152, 1664→2428), H-K hard assert replaces silent zero-pad, H-L routing Options + fail-fast. GPU window closed same day: 63/63 hypernet tests + on-device cosine 1.000000. Deferred: H-D/E/F/G perf (per-column round-trip ~12 s/iter at 128³ — tracked by the bench), H-M per-module GeneratedLoRA, sigmoid dedup (blocked on meta_lora WIP). Record: removed issue file (git history). |
| **697** weaver_gpu corrector silent no-correction chain + carried G3 divergence | RESOLVED (2026-08-15 pass fixed H1/H2/H3/H5/H6/H8/H13/H14/H17; the carried G3 CPU↔GPU ranking divergence root-caused + fixed 2026-08-16 in `792ffa9b5`) — **the G3 root cause was the SHARED batched plane GEMV substrate, not weaver kernel numerics**: `gemv_batched_plane_f32` derived `out_dim = output.len()/batch` where `output.len()` is the FULL allocation size; the weaver per-depth path (batch=seq_len=2 into a 5-row scratch) computed out_dim=160 instead of 64 and wrote every batch row ≥ 1 at a stray offset (row 1's GEMV result found verbatim at [160..224) while the consumer read [64..128) — u_cond row 1 became pos_emb-only garbage; 0.28-max-diff corrected probs; CPU vid=36 vs GPU vid=41). Every exact-size caller (unit tests, gemma2 backward, batched path at depth==max_depth) computed the right out_dim — the exact trap that let all tests pass while production was broken. Fix: out_dim rides in params[2]; behavior-identical for exact-size callers. Post-fix all intermediates at fp noise (residual 9.5e-7); weaver 26/26 incl. restored G3; new regression test `test_gemv_batched_plane_oversized_output_weaver_scratch`; CI Layer 1.8 --skip removed, floor 25→26. Deferred `[-]`: H4 input-scratch residency, H9 entry-point dedup (pair with H4), H7/H10-H12/H15/H16 mechanical. Record: removed issue file (git history). |
| **698** weaver kernels dot-per-row cliff + KVarN VarN inversion | RESOLVED (2026-08-15 pass closed all 13 Part B kvarn correctness hazards + Part A A-H2/H3/H6/H7/H11 + A-H4 documented; A-H1 dot-per-row restructure landed 2026-08-16 in `959b6a0a9`) — one PLANE per output row (lane-strided + plane_sum, the gemv_plane pattern) replacing one-thread-per-row with h serial FMAs. GOAT: G1 26/26 weaver tests incl. G3 + dot_per_row parity (plane tree-reduction numerically better than serial); G2 per-depth prod 6.862→6.443 ms/call (−6.1%), batched 3.593→3.317 (−7.7%) — end-to-end bounded by dispatch overhead (1 of 19 dispatches), kernel itself 1 workgroup → ceil(D*K/8) coalesced ones; G3 full surfaces; G4 identical launch shape. The G3 cross-note is resolved by the 697 row (shared GEMV substrate `792ffa9b5`, not these kernels). Deferred `[-]`: A-H5 (softmax_k single-thread), A-H8 (download 12 blocking reads), A-H9 (naive transpose), A-H10 (naive sigmoid), B-H8/H9 (contract redoc — with the long-term B-H13 legacy-pair deletion pending a consumer audit). Record: removed issue file (git history). |
| **705** persistent grid-stride GEMV | RESOLVED (negative result, 2026-08-16) — premise REFUTED by measurement: the occupancy-derived residency cap (128 SMs × 6 = 768) measured a 1-2% LOSS vs the uncapped one-warp-per-row control at every tested size (512 under-resident lost 13%); the wave-quantization-tail theory is closed twice over (the z-class exact-1-wave counter-example at 706 GB/s vs lm_head's 854 + the locality trade — grid-stride scatters the tight DRAM wavefront the oversized grid gets for free). Shipped: uncapped default + `gemv_ternary_dp4a_multi_persistent` + `RIIR_GEMV_PERSISTENT_GRID` env cap retained as the tuning apparatus; **latent Issue 697 arg-layout bug FIXED** (the non-accum else-branches passed the 22-arg multi kernel handle into the 10-arg single-kernel launcher — UB reachable from `forward_token_profiled` + devpos variants, never hit by the 697 gates which only ran `accumulate_out = true`); dp4a row math deduped 3 byte-identical copies → one `gemv_ternary_row` device helper. GOAT: G1 ppl bit-exact 5.4681/3.7778/0.6909, G1a graph≡eager 0.000e0 + greedy token equality, G3 ≤ 0.45%, G4 alloc-free; new unit gate `test_persistent_gemv_matches_split_bit_identical` (4 grid sizes + accumulate mode, max_diff 0.0). Redirected levers preserved: per-row warp amortization untested; rmsnorm multi-block + attention_decode parallelization + row-splitting are tolerance-gated (owner decision). Record: `Bench 684`. |
| **706** rmsnorm_quantize single-pass | RESOLVED (2026-08-16, `e4e88c37d`) — the largest non-GEMV decode kernel cost (691 µs/token, 128 launches × 5.4 µs at 1 block × 512 threads) cut to a true single pass: `gamma` prefetched into registers alongside `input` in phase 1 (loads overlap the reduction barrier), phase 4 computes `v = vals[i] * inv_rms * gam[i]` from registers — load schedule changes, arithmetic instruction sequence identical → bit-identical. nsys: **5,460.9 → 4,455.7 ns avg (−18.4%, drift-controlled)**, −128.7 µs/token; ppl bit-exact 5.4681/3.7778/0.6909. **The v1→v2 iteration surfaced a pre-existing local-memory-spill pathology**: runtime-`ablock` loop bounds forced the exactly-sized `vals[16]`/`gam[16]` arrays into LOCAL memory (no unroll, no register form for dynamically-indexed arrays) — v1 without the const-bound+`#pragma unroll` fix measured 0.6-1% SLOWER; the pathology pre-existed in the Issue-623 kernel and hides until a neighbor adds same-address-space traffic. Distilled as riir-clippy Batch 49 rule `runtime-arg-loop-bound-demotes-arrays-to-local-memory` (corpus 344). Same shape applied to `rmsnorm_quantize_with_norm_x_f32`; 41/41 cudarc + 19/19 rmsnorm unit gates both versions. Record: `Bench 686` + riir-clippy Research 055. |
| **709** small-leaves doc-truth + loud-stub pass | RESOLVED (2026-08-16, truth-pass branch — every hazard addressed at the issue's own minimum fix-shape) — H1 game_mux demux relabeled as a superposition-biased heuristic estimate (module + demux docs, false W^(-1) comment replaced with the actual projection math, truth-pinning test `test_hash_pairwise_dots_positive_contamination`, riir-games `game_mux_bridge` consumer note: ranking uses survive the shared bias, absolute values do not); H2 cpu_reference f32-accumulation truth documented (f64 upgrade = optional); H3 `lora_b.wgsl` base-GEMV parity trap documented (kernel adds `base_sum`; parity needs `gemv() + lora_forward()`); H4 elf "rms_norm" L2-normalize mislabel fixed in docs + sqrt(d)/β-calibration note (variant rename deferred — breaking API); H5 embedding-SDAR silent no-op → one-time loud warning in riir-train `train_step_embedding_sdar`; H8 depth_tier stale "inference-forward consumer" claim corrected (sole consumer = the training batch sampler); H9 threshold-order `debug_assert` (cold < hot < plasma); H10 `compare_f32` panics-on-first-mismatch contract documented; H11 `mux_kl_loss.wgsl` 256 KB workgroup-storage warning (pipeline creation would fail if wired). Stronger remediations (real-model demux validation, f64 reference, variant rename, H7 params double-write) remain optional upgrades. Record: removed issue file (git history) + the doc notes themselves. |

| **707** `metal` dep not target-gated in riir-gpu | RESOLVED (2026-08-16, T1 `7c8b69045` + M3 T2 verification same-day) — the optional `metal = "0.31"` normal dep sat in riir-gpu's general `[dependencies]`, so `ternary_inference` / `--all-features` (which enable `metal_tensor_gemm` → `dep:metal`) compiled `core-foundation` on Windows and hard-failed (E0432/E0433). Fix: the dep moved under `[target.'cfg(target_os = "macos")'.dependencies]` beside `cubecl` — the feature stays defined on all platforms, simply inert on non-macOS (the same target-section + `dep:` pattern already proven in-crate for `cubecl`). Verified on BOTH hosts: 4090/Windows — `cargo tree -i metal` shows nothing in the host graph + full riir-clippy `ci_feature_guard.sh` ALL LAYERS PASS for the first time on Windows incl. `--all-features` with `ternary_inference` (T3); M3/macOS — `cargo check -p riir-gpu --features metal_tensor_gemm` (lib + `--tests`, compiling the 656 metal-tensor test family) resolves clean + `bench_663_zero_copy_interop` RUNS GREEN (1 passed — custom Metal kernel read+wrote the CubeCL buffer zero-copy, 1024 f32 doubled in-place, same GPU instance). Record: removed issue file (git history) + the rationale comment in `crates/riir-gpu/Cargo.toml`'s macOS target section. |

| **708** dflash forward non-functional + flashprefill doc-truthfulness | RESOLVED-PARTIAL (2026-08-17) — Part A (`forward_dflash`): the forward was an unfinished MVP presenting as functional (two no-op stub kernels + two stale-uniform wrong-math classes); made FUNCTIONAL + CPU-reference-pinned in `42b564759` (per-slot uniforms H1/H2, real gamma-less RMSNorm H3, real bidirectional concat attention H4 incl. the single-row q_cache structural bug, H5-H8/H11-H12 dead-work/stub deletions; G1 parity max_rel=5e-6 at Config::micro, 565/565 feature tests). Part B (`forward_flashprefill`): H13/H14/H15 doc-truth (two-pass softmax truth, metal() divergences, dead `tail_window`); H17 dead -1 tail-fill + H19 deterministic zero rows + H20 dims contract assert in `f1378a25d` (first-ever functional coverage: two-dispatch H19 regression on shared buffers + H20 should_panic); **H16** — fused `flashprefill_score_select` kernel replaces the block_score+block_select pair (both WGSL files + pipeline slots deleted, pipeline 4→3 dispatches, `block_scores` buffer + its num_blocks²×n_head ≈2 MB-at-8K round-trip GONE; two-sweep recompute with identical op order — no score array, no num_blocks cap); new content-level selection test (`flashprefill_score_select_threshold_path`: head_dim=1 exact dots, k={1,10,1}/alpha=0.5 → q_block 0 selects {0}, q_blocks 1-2 {1} ONLY, asserted through sparse_output) + kept `#[ignore]`d wall-clock probe. **Honest G2: NO measurable end-to-end timing gain** (seq=512 p50 11311→11308 µs, seq=8192 234952→235036 µs — the short-context floor is per-submit/poll not per-dispatch, and at 8K the two-pass sparse_forward (H13) dominates at ~99.9%); the win is the resource axis (−2 MB, −1 dispatch/bind-group/pass per call) + the now-tested selection path. Full riir-gpu default lib 540/0/2 + clippy 0 in touched files. Deferred `[-]`: H9/H10 (LOW — host wte/wpe clones need a gather kernel; encoder-submit merge). Record: removed issue file (git history — full 4-update resolution narrative there). |

| **710** kimi batched backward sync2 guard drift + 9-site CPU weight-grad backlog + q8kv untested multi-tile | RESOLVED (2026-08-16→17, 3-pass fix chain + GPU-window closeout: `c7c636bfb` → `decb45157` → `859005975` → `c390d729c`) — **H1** sync2 guard-drift entry check (`use_output_gate == w_g_t.is_some()`, both directions — the Batch-51 corpus rule applied to the live site); **H2 all 9 MLA weight-grad sites converted CPU outer-product → GPU GEMMs** (7 single-stage sites + 2-stage chains for W_UV/W_UK with stage-1 device-resident; sync count stays 4; the sequence gate now asserts ALL 9 sites + both norm gammas — was 2 of 9); **H3** k_c precompute (n_h·l(l+1)/2 → n_h·l matvecs); **H5** scratch hoists complete; **H6/H7/H8** dead `sync2_layout` + underscore-dead clones + `debug_assert!(pos < seq)`; **H10 (HIGH) q8kv multi-tile test PASSED in the GPU-exclusive window** (n_positions=512, 2 tiles, Phase-6 cross-tile rescale + tail-tile guard pinned, max_error 4.3e-5 vs tol 0.15; ignore removed); **H11** `n_positions` single-source-of-truth documented as caller contract; **H12** stale allow removed; **H13** quantize scratch hoist. **H2 G2 honest FAIL**: 0.998× prod / 0.936× reduced — the 2-stage chain overhead outweighs the removed CPU loops at small scale; kept as an architectural refactor (alloc-storm removal + GEMM unification + the all-9-sites gate), NOT a speedup — do not cite as one. Standing tok/s re-verification: cudarc LoRA graph 90.04 tok/s (highest recorded for the path — resolves the Bench 684 environment caveat). Parity was gated by Issue 711 (`Features::IMMEDIATES` gap — launches silently dropped), RESOLVED the same window (`59999cd58`): sequence gate PASSES end-to-end (lm_head 3.8e-6, embed 4.4e-5, all per-layer MLA/H2 asserts), kimi lib 7/7, weaver 26/26, gemv 80/80. Deferred `[-]`: H4 packed expert uploads + H14/H15 (WGSL numeric changes — reopen when a GPU-validated perf session wants them; H2's G2 warns chain-dispatch overhead can lose at small scale), H9 (deferred with cause: the CPU reference also recomputes postnorm — swapping only the GPU side = gratuitous divergence for a µs-scale win), the k_c stacked-GEMM candidate (reopen on merit only, with a production-scale G2 win case). Record: removed issue file (git history — 4-update resolution narrative) + the Issue 711 row + Bench 684. |

| **712** gemma2_cubecl forward tests ~45 GB device-alloc churn / commit cliff | RESOLVED (2026-08-17, landed the prior 4090 session's WIP + completed it) — the fix set: (1) `gemma2_cubecl`/`gemma2_d2f` tests restructured to ONE full-model instance live at a time (scoped drops; shared weights where seed-identical); (2) `test_gpu_support.rs` — `gpu_release_pages()` (explicit pool cleanup + FIFO-drain probe) called at heavy-test START + END, and `heavy_model_test_gate()` — a process-wide `Mutex` serializing every full-gemma2_2b-instance test (the 18 cubecl/d2f tests + `gemma2_forward`'s 10, whose own poison-prone `GPU_TEST_LOCK.lock().unwrap()` it replaces — one heavy test at a time, light tests stay parallel); (3) vendored `sliced_pool.rs` patch: `ALLOC_AFTER_FREE=5` hysteresis page dealloc on non-explicit cleanups (sliced pools previously NEVER released freed slices → monotonic committed growth), mirroring `ExclusiveMemoryPool`; (4) `gemv_autotune` cache made process-global (benchmark tensors once per process, not per instance). Measured (4090, GPU-exclusive, full `cubecl_runtime` lib suite): the host-alloc ABORT (`memory allocation of 84934656 bytes failed` → STATUS_STACK_BUFFER_OVERRUN) is GONE; 581 passed / 8 failed vs 575/14 at clean HEAD same-day — and the 8 remaining all trace to ONE device-lost event that is NOT this fix's (see Issue 714's bisect matrix: dealloc-off, cache-off, and no-WIP-at-all runs all show it; the 2026-08-16 493/93-era control did not — suspects: the `59999cd58..ce5651035` commit range or box driver-state drift; reboot-window bisect blocked by the sibling's resident bonsai process). |

reboot-window bisect blocked by the sibling's resident bonsai process). |

| **714** cubecl_runtime lib suite "Parent device is lost" late-tail cascade | RESOLVED-CASCADE (2026-08-17 later session; the slab-bind fix landed the prior window) — **506/84 → 589/1, 0 device-lost, 0 DSD panics, 0 host aborts**. Four fixes: (1) vendor `cubecl-wgpu` `initialize_memory` graceful-OOM + self-healing ALL-stream pool sweep + retry (the DSD-thread panic at `server.rs:299` that misreported as device-lost is GONE; `bind` made fallible + readback staging graceful); (2) `GpuContext::new()` `OnceLock`-cached — it previously created a NEW wgpu Instance/Adapter/Device/Queue + registered a NEW never-deregistered CubeCL server on EVERY call (**46 live Vulkan devices per suite run**, each pinning its test's pool high-water — the ~20 GB residue; diagnosed via 46× `device limits` eprintlns); (3) vendor `memory_cleanup` sweeps ALL streams (cubecl pools are PER-STREAM; a single-stream cleanup missed quiet streams' fully-free pages — every stream's pools now cleaned on any cleanup call); (4) the `gemma2_forward` release gap (the 712 fix gated its 10 tests but never added `gpu_release_pages` — `get_ctx()` sweeps+pumps on acquire; `new_cubecl` sweeps between its deliberately-double-allocating halves) + order-dependent `test_autotune_default` fixed (asserted the process-global cache empty at its runtime; clears first) + chunked `upload_f32` (64 MiB `write_buffer` chunks — kills `create_buffer_init`'s full-size mapped-at-creation staging transient, 2× the 2.36 GB wte). **Remaining, newly EXPOSED not caused (the cascade masked it):** `test_gemma2_gpu_forward_cubecl` OOMs DETERMINISTICALLY in isolation (module-only fresh process, device peaks 23,793 MiB) — `new_cubecl`'s double allocation (cubecl pool + full WGSL buffers — its own doc: "Total GPU memory usage doubles during the migration period") + wgpu-hal's allocator hoarding freed buffers (never returned to the driver mid-process; no trim API in wgpu 30). Ranked directions recorded in the issue (chunked upload landed; skip-WGSL-weights flag + upstream trim ask remain). **FULLY CLOSED (same day, follow-up session): the remaining forward_cubecl OOM fixed by SKIPPING the WGSL weight buffers on every new_cubecl* constructor** (`GpuGemmaWeightBuffers::skipped` — zero-size buffers + empty wte_cpu; `new_internal` gained a `skip_wgsl_weights` flag; Q4K cubecl constructors also skip the now-dead CPU-side quantize). Provably dead weight: no dispatch_* helper branches on cubecl_fwd — only public entry points do, and those delegate. Guards: forward_fused/forward_fused_sampled now delegate to CubeCL like forward/forward_sampled (also fixes the footgun where a cubecl pass silently ran WGSL), benchmark_per_layer/forward_layer_by_layer return a loud error, test_cubecl_layer_drift switched to new() (identical WGSL trace, half the memory). **Measured: module isolation 6/1 to 7/7; single-test peak 23,793 to 12,798 MiB; full serialized suite 589/1/2 to 591/0/2** (zero device-lost/DSD/aborts; +1 test from the sibling landed 715). Drive-by in the same commit: the pre-existing E0382 (client moved at gemma2_cubecl/tests.rs:941) that had left the cubecl_runtime,gpu_decode_fusion combo UNCOMPILABLE since the 712 sequencing edits — fixed with client.clone(); the combo now compiles but 3 new()-based GOAT tests OOM at the 24 GB ceiling (no green baseline ever existed on this box; same wgpu-hal-arena class — filed as Issue 718, NOT caused by this fix: the trio never touches the changed paths). Record: git history (issue file removed at close) + run logs target/714_fix_full{3..10}.log, target/714_single.log, peak trace target/714_memtrace2.txt (untracked evidence). |

| **716** sink-position-aware Q8KV scale policy (MA/outlier failure mode) | RESOLVED (2026-08-18, M3 Metal GPU-exclusive) — the arXiv:2608.12149 failure mode is **LIVE on the Gemma-2 decode path**: synthetic MA injection (1-2 channels ×100/×300 on pos 0 + delimiter) inflates q8kv attention output error **584×/18,844×** over uniform data (0.000268 → 0.157/5.050), concentrated at the poisoned 32-blocks (88×/299× vs clean dims); CPU row-level neighbor collapse 58× — the KV-side twin of the Research 085/086 weight-side collapse. Mitigation A/B: (b) lossless f32 sink sidecar WINS (guarded 4.781 → 0.000698, **6,850×**, bit-identical at S=0, 0.96× perf = neutral-to-faster) over (c) KVarN-style Hadamard sink rotation (CPU-simulated 19.6× reduction but nonzero + strictly larger kernel cost) — (b) ships behind opt-in `q8kv_sink_guard` (`quantize_kv_with_sink` + `launch_with_sink` + kernel 5th array deriving sink_rows from the sidecar ALLOCATION size). **Side finding (the Bench 642 lesson re-hit live):** cubecl's `KernelLauncher::process_buffer` derives the kernel-visible slice length from `handle.size_in_used()`, NOT the bound `BufferArg` length — a zero-length binding over a real handle silently read the QUERY buffer as position-0 K/V (uniform error 0.000268 → 0.12 across every test until root-caused; fix = 4-byte 1-f32 dummy allocation → 1/2/1024 = 0 rows). GOAT G1–G4 PASS (Bench 691); stays opt-in — S>0 is a caller policy decision (which rows are sinks belongs to the decode loop; `CpuKVCacheQ8` keeps `sink_kv` empty until a production consumer asks). T5 (Kimi K3 MLA design note) deferred `[-]`. Record: `Bench 691` + removed issue file (git history). |

| **717** Bonsai consumer gate for the modelless bigram Markov drafter (Metal wall-clock) | RESOLVED (2026-08-18, two-box split) — the 4090 half (Bench 693) discharged the zero-row kill-check (0.00% production L4 corpus / 1.37% diverse real Rust — Bench 664's 36% was a whitespace-tokenizer artifact, not a Bonsai property) + beat the factorized floor at every intended operating point, and found the dd_tree 16-bit path limit (see the katgpt-rs Issue 670 row); the M3 half (Bench 694) measured the rest: **G3a FAIL at the chain seam — 0.326x/0.252x speedup at K=1/2 with acceptance 0.000, statistically identical to the shipped NgramDrafter (0.328x)** — the loss is verify-side economics (K sequential `forward_speculative_verify` forwards + rollback/reapply at batch-1; only the logits READS batch), not draft cost, so Issue 659's "modelless wins where DSpark loses" motivation is refuted at the chain seam and `bigram_markov` stays opt-in in katgpt-rs; G1 chain losslessness PASS (bit-exact warm-up round). G2-full honestly OPEN: DSpark's trained rank-256 Markov head read standalone (`A[prev].B[next]`, both orientations probed) does not clear the unigram floor (z=-0.45) — WRONG CONTRACT, not a weak head (block-denoising diffusion drafter; the head is a low-rank logit bias composed under log-SNR conditioning; recovering the contract needs the DSpark forward). Reopen trigger: katgpt-rs Issue 670 lands -> re-gate with tree drafting (tree acceptance 0.89 lower-bound vs chain 0.30 — the only path to paying for verification). **Issue 670 RESOLVED 2026-08-18 same day** — `TreeNode.parent_path` widened u128 16-bit packing -> `TreePath` `[u32; 8]` (katgpt-rs commit `e6d526c5` + this repo `35439dfe1`): the 717 gates promoted clean16 -> all-positions (tree == chain at top_m=1 EXACT: 1.4736/1.4736 proxy, 0.4690/0.4690 diverse; the aliasing signature gone), bench_694 round-trip 258,560 nodes / 0 failures at ids >= 65,536 on 1.62% of deep nodes, tree acceptance now EXACT 0.8785 — the reopen trigger's blocker is cleared; the G3a tree-drafting re-gate (needs a batched tree-verify wall-clock harness) is the armed next step. GGUF loader hardening shipped en route: BF16 is GGML type 30 (29 is IQ1_M — previously NO real BF16 GGUF could open) + a Q4_1 dequant decoder (both verified against the llama.cpp layout, found via the dspark GGUF). Takeover note: this was stale sibling WIP allocated as "690" (a number collision with the forage IFD bench — renumbered 694). Record: `Bench 694` + `Bench 693` + removed issue file (git history). **G3b (4090) DISCHARGED same day (`Bench 695`): FAIL - 0.84x/0.32x at K=1/2 vs the CUDA-Graphs baseline (87 tok/s), corrected accounting (in-domain prompt fixing P3's acceptance-0.000 artifact, bonus banking, 8-byte argmax verify reads, sync-free dtod checkpoint) - the wall-clock case is closed-negative at the chain seam on BOTH GPU substrates.** Structural verdict (measured + arithmetic): with sequential verify every committed token still costs exactly one forward, so tok/forward <= 1.0 with equality only at acceptance 1.0 - the tree seam adds NOTHING under sequential verify (tree drafting degenerates to chain; branching cannot reduce forward count), so the "tree-drafting re-gate" armed above is VOID until batched/parallel verify (the Issue 652 chunkwise-parallel DeltaNet class) exists. Also measured: the online NgramDrafter BEATS the static bigram table at K=2 in-domain (acceptance 0.400 vs 0.087 - the trigram adapts to local text; flips at K=1) and K=1 bonus banking degenerates to greedy-with-overhead (0.81x = pure cycle overhead). The cudarc speculative seam shipped en route (`forward_speculative_verify{,_argmax}` + GPU-side `checkpoint/rollback_speculative_gpu` in `ternary_deltanet_gpu_forward_cudarc.rs`, G1-validated bit-exact, feature `speculative_decode`) is the ready consumer surface a batched verify will need; it caught two live bug classes - NgramDrafter's BLAKE3 fallback emits UNBOUNDED u32s (embedding OOB -> CUDA_ERROR_ILLEGAL_ADDRESS; the seam now asserts the in-vocab contract) and the mid-loop un-rotate position-shift hazard (fixed by keeping the cubecl field-holds-last-forward contract + `invalidate_graph` at run end). **G2-full DISCHARGED same day (`Bench 696`, 4090)**: the dspark GGUF is a PUBLIC HF artifact (1.9 GB = the 6-layer 3.6B drafter + trained Markov head - the only-on-M3 blocker was local-disk, not availability; downloaded to riir-train/data/) and the FULL llama.cpp contract was ported (encoder fusion of target layers [1,16,31,46,61] -> per-layer KV injection -> non-causal noise block -> markov-bias greedy chain + confidence head; new forward_token_with_layer_capture on the cudarc forward extracts the target features - 5 sync+20KB dtoh/token, diagnostic-only). Split verdict: on TRUE acceptance (vs the target own greedy - the deployment metric) the trained drafter WINS decisively (1.689 vs 1.439 tok/cyc, +17.4 percent; top1@0 0.288 vs 0.189); on the corpus-proxy protocol (the M3 P2 metric) the modelless table WINS (0.185 vs 0.065) - the corpus metric mis-measures target-conditioned drafters, resolving the P2 OPEN row with a metric correction. Ablations: bias-only 0.028 approx floor (replicates the wrong-contract finding at the same anchors); no-KV-injection 0.071 vs full 0.192 (features 2.7x load-bearing). conf@0 overconfident here (0.75-0.79 vs 0.288 measured - quantized-target/domain shift; flagged honestly). katgpt-rs Issue 659 RESOLVED + removed (records: Benches 663/664/693/694/695/696); bigram_markov stays opt-in; the only wall-clock path remains batched verify, now with the trained drafter +17 percent acceptance head start as the prize. |

| **718** fusion-combo OOM trio + 2 invisible combo failures (argmax oracle, delta_routing decode bug) | RESOLVED (2026-08-18, same-day follow-up) — **scope: the 3 gemma2_forward GOAT tests OOMing in cubecl_runtime,gpu_decode_fusion** (two live ~12 GB new() passes + wgpu-hal arena residue at the 24 GB ceiling). Fix = the issue's ranked direction 1: small_goat_config(n_layer) test helper — n_layer 26→2/1, vocab 256000→32000 (wte 2.36 GB→295 MB), block_size 8192→64 (KV 1.7 GB→13 MB); same kernels at production n_embd/head_dim/mlp geometry, parity properties config-independent; per-pass ~12 GB→~0.9 GB. Module 9/3→**12/0**. **Establishing the combo's first green baseline then surfaced TWO more failures in tests that are themselves gpu_decode_fusion-gated** (never ran while the combo was uncompilable — the E0382 window — and never run by plain cubecl_runtime): (1) test_goat_gpu_argmax_correctness — the test's cpu_argmax oracle used Rust max_by (LAST tied index) vs the ArgmaxCubeCL kernel's pinned FIRST-index spec (sampling_cubecl's own test_argmax_all_equal) + the engine decode convention (swir gemma2 argmax_u32, ict audit argmax_u8, test-pinned breaks-ties-by-lowest-index); failed all_equal 0 vs 99; oracle rewritten to strict-> first-index. (2) **test_goat_full_pipeline_decode — a REAL PRODUCTION BUG since eddac8c50 (2026-07-13): forward_gpu_logits_handle_max_layer had NO delta_routing fallback while forward_gpu/forward both delegate to the CPU-hybrid under that feature — and delta_routing is in riir-gpu's DEFAULTS (added for struct-layout sync). generate_gpu() (production greedy decode, the 4-byte-download path) decoded through a layer stack that never applied delta routing under DEFAULT features**: token divergence at pos 1 (255203 vs 67044), max logit diff 3.06, proven by a 4-phase probe (handle-vs-readback-vs-baseline; forward_gpu delegates → bit-exact 0.0 vs baseline; the handle path alone diverges, deterministically, independent of pool state). Fix: mirror forward_gpu's early-return — delegate to forward() + re-upload the logits to preserve the Handle return type (the lm_head_cpu pattern); the speculative draft loses its early-exit speedup under delta_routing but stays correct (verify loop re-runs the full model). Post-fix gemma2_cubecl test_goat_* **6/0** (was 4/2); full serialized combo suite green (first baseline ever on this box); clippy 0 in touched files. **Parallel-mode full-suite freeze found during validation filed as Issue 719** (serialized unaffected). Record: git history (issue file removed at close) + probes target/718_probe{,2,3,5}.log + module/final logs (untracked evidence). |

| **719** parallel-mode full-suite freeze — all 24 workers starved behind first-call `GpuContext::new()` | RESOLVED (2026-08-18, same-day; commit `97bcf4738`) — root cause = the "race-benign" first-call pattern in `GpuContext::new()` (Issue 714's fix): parallel first-callers that missed the OnceLock EACH ran the full init sequence — wgpu Instance/Adapter/Device creation + `init_device` CubeCL server registration — concurrently, which deadlocks non-deterministically on this box. Triage to root cause: (1) log-vs-list-set arithmetic on the frozen run pinned the stuck set to EXACTLY 24 tests (the libtest concurrency count) — every `GpuContext::new()` first-caller in the spawn window (8 backward-family, 4 buffer, 2 context, 10 deltanet); (2) bisection: backward-only PASSED, backward+buffer HUNG, backward+context HUNG; (3) pure-race repro (zero test bodies): 7 threads calling `GpuContext::new()` — 6/7 completed, 1 wedged inside `init_device`; instrumented re-run: 3/7 through request_device with 4 wedged INSIDE `adapter.request_device()` — the stuck point MOVES between runs (driver-level race, not a single deterministic lock site; probes also showed Instance::new + request_adapter always complete). The race additionally leaked every loser's CubeCL server + Vulkan device for process lifetime (no deregistration) — the partial reintroduction of Issue 714's 46-device leak (explains the frozen process's 3.8 GB residue). Fix: process-wide `GPU_INIT_LOCK` in context.rs, shared by `GpuContext::new` AND `CubeCLContext::new` (both run the same driver-level sequence; the OnceLock alone did not serialize cross-path racers) — one racer inits alone, the rest fast-path after the guard; fast path + uncached-failure semantics preserved. Regression gate: `crates/riir-gpu/tests/context_init_race.rs` (8-thread same-path race + 8-thread cross-path race; 3× green ~1s). **Validation: the exact command that froze at 101/604 now completes 605 passed / 0 failed / 2 ignored in 1664 s with DEFAULT parallelism — the first green parallel-mode full suite on this box** (serialized ≈29 min was the only safe mode before). Clippy 0 in touched files. Record: git history (issue file removed at close) + repro/evidence logs target/719_repro{,2,3}.log, target/719_full_parallel.log (untracked evidence). |

| **720** WGSL Q4_K path uploads the full dead f32 weight set | RESOLVED (2026-08-18, same-day) — `new_q4k`/`new_q4k_gguf` uploaded BOTH weight sets: the full f32 `GpuGemmaWeightBuffers` (~9.7 GiB device for Gemma 2 2B) alongside the ~1.4 GiB Q4_K set, while a full dispatch-site audit shows every Q4_K GEMV routes through `weights_q4k` and the f32 `wte` is only bound in `WeightFormat::F32` arms — the Issue 714 class, missed on the WGSL Q4_K constructors; `GpuGemmaWeightBuffersQ4K::wte_cpu` (2.2 GiB host) additionally had ZERO readers. Fix: `GpuGemmaWeightBuffers::norms_only(...)` (live fields only: 4 per-layer norm gammas + `final_norm` + host `wte_cpu`; zero-size placeholders for the rest, the `skipped()` pattern) wired automatically in `new_internal` when `weights_q4k.is_some()`; dead `wte_cpu` field removed (GGUF constructor drops the full F16→f32 embedding dequant); `forward_layer_by_layer` now refuses Q4_K passes (its unconditional f32 trace never reflected Q4_K numerics; only in-tree caller uses `new()`). `benchmark_per_layer` still works on Q4_K (format-branching fused dispatch). GOAT (Bench 692, probe `tests/probe_720_q4k_goat.rs` run identically on a pre-fix stash baseline): G1 bit-identity PASS (`LOGITS_HASH df741575cc8f33be` both sides); G2 **−10.3 GiB device per `new_q4k` construction** (14,089 → ~3,500 MiB) + −2.2 GiB host; G3 14/14 gemma2_forward + q4k_weights lib tests + example/bench compile + cubecl/fusion/q8kv combo clean + clippy 0; G4 constructor-only, strictly fewer allocs. Bonus: the WGSL Q4_K decode path had ZERO test coverage before — now gated by `test_gemma2_gpu_forward_q4k_goat` (e2e determinism + guard) + `test_q4k_norms_only_skips_f32_projections`. Record: `Bench 692` + git history (issue file removed at close). |

| **721** tree-masked batched verify for Ternary-Bonsai GPU (Issue 717 G3a re-gate) | RESOLVED (2026-08-19, negative result — structural) — all 8 tasks done; the correctness story is complete, the wall-clock story closed-negative. **T4a** (`28ac37fa2`): two new kernels in `qwen_attention_cubecl.rs` — `QwenRopePartialTreeCubeCL` (batched partial RoPE with a per-node positions upload, base_pos + depth; the prefill kernel's pos == row-index identity does not hold for topo-indexed tree rows) + `QwenAttentionTreeGatedCubeCL` (ancestor-masked online-softmax attention over [committed cache prefix ∪ ancestor-or-self tree rows], both Issue 715 barriers preserved in uniform control flow via −1e30 masked lanes, fused sigmoid gate, GQA); driver `tree_attention_layer_batched` (8 dispatches/layer for ALL T nodes); the per-branch bridge deleted — verify is now read-only on ALL state. Kernel unit tests vs CPU refs (GQA 4:2; 2-tile exercising the Issue 715 back-edge under the mask; parent-chain visibility independent of the bitmask). **Fast commit** (`1eb211ad2`): `commit_tree_verify_fast` — rank-1 state replay from the verify's own GPU-resident tree rows (conv advance + recurrent update + KV append via the decode path's own kernels, ZERO weight reads, ~2 ms/cycle vs ~82 ms sequential; measured bit-identical to the sequential replay, follow-up max_rel 0.00000) + the t < t_max logits-reuse fix varying trees exposed. **G1 PASS bit-identical through T4a** (branching T=9 max_rel 0.00435 / chain T=4 0.00000 vs TOL 0.03, measured twice); standing gates: `tree_verify_g1` (3 tests) + 5 kernel unit tests under `--features speculative_tree_verify`. **G2 FAIL — STRUCTURAL** (first end-to-end production-dim run on the real 27B — T=64 multi-root, hd=256, 64 layers, zero failures): B/S = **10.90× (T=16) / 44.87× (T=64)**, acceptance 0.045/0.211 — matches the structural prediction ≈T/(1+acc). Root causes: (1) `GemmTernaryBatchedCubeCL` runs large shapes at parity-per-unit-work on M3 Metal (its own Bench 641 record — occupancy-bound; the designed weight-read amortization does not convert into throughput) → verify cost measured LINEAR in T (~450 ms/cycle at T=16 → ~1860 ms at T=64, only ~1.4× effective amortization), refuting the issue's "~1 weight-read pass for T nodes" premise on this substrate; (2) live-from-BOS drafter acceptance 0.05–0.21 vs the replay-based 0.8785 (the table was built on repo markdown; live generation drifts) — compounding but not sufficient alone. **Issue 717 G3a re-gate stays CLOSED**; reopen triggers: (a) weight-stationary T-amortized ternary GEMM (must first beat 3× standalone on `ffn_gate/up` — a new kernel family; the Bench 641 tile sweep already showed the current one can't), (b) a real-context/trained-head drafter, (c) 4090 re-measure DISCHARGED 2026-08-20 (Issue 733 / Bench 703, GPU-exclusive, wgpu<spirv>/Vulkan): G1 bit-identical on Vulkan (same three max_rel numbers as Metal - correctness now cross-backend) + G2 re-measured: S=31.05 ms/tok, B/S=11.19x (T=16) / 32.52x (T=64) - FAIL CONFIRMED cross-substrate; per-node verify cost 0.67x a sequential token (M3: 0.79x) = only ~1.5x weight-read amortization, still ~linear in T; perfect-drafter bound ~21x - no drafter saves this kernel family on either substrate (the only remaining reopen path is (a)); the two G2 gates un-gated for Windows (any(macos, windows)). G3 PASS / G4 PASS; GPU-load disclosure recorded (sibling LoRA active throughout; interleaved median-of-ratios, 10–45× margins beyond load noise). Record: `Bench 697` + `Bench 703` + removed issue files (git history). |

| **722** cudarc forward-orchestration hazards (Batch 58 mining yield) | RESOLVED (2026-08-18, same-day, 4090 box) — 12 of 13 hazards fixed in `ternary_deltanet_gpu_forward_cudarc.rs` + `cudarc_kernels/mod.rs`; H10 deferred `[-]`. **H1 (HIGH, doc-lie to structural fix)**: the doc referenced a `spec_restore_buffers`/`spec_unrotate` pair that NEVER shipped; the real fix is stronger than doc-truth — `spec_verify_dispatch` now DROPS any captured graph itself (the rotation swaps field handles; a replay would write pre-rotation allocations while the fields point at pool buffers — silent wrong-position logits). Regression gate `test_speculative_run_invalidates_captured_graph` validated BOTH directions: fix disabled FAILS (max_diff=0.236, the stale position-1 read), fix enabled PASSES **bit-identical (0e0)** — and the first draft of that test was VACUOUSLY GREEN (unseeded `wte` made every token's logits identical; the embedding seeding is load-bearing, noted in-test). **H2**: `set_lora` returns the new `CudarcKernelError::InvalidArg` for non-DeltaNet / out-of-range targets in EVERY profile (was `debug_assert!`-only — silent never-firing adapter in release) + test. **H3+H12 together**: the triplicated capture block extracted to `ensure_graph_captured` with the error-path abort (end_capture on an invalidated region transitions the stream out of capture mode — previously one failed launch mid-capture wedged every later op on the stream); 3 sites to 1. **H4**: the argmax tail added to `forward_token_with_final_hidden` + `forward_token_training` — the `last_argmax()` family contract now holds per-variant, not per-family. **H5**: conv1d backward's full `[max_t_len x conv_dim]` dtoh + re-upload round-trip deleted (the host already owned the data — byte-identical direct upload) + the per-token `conv_state` clone_htod folded to scratch. **H6+H9**: the attention backward's ~10 fresh device allocs/uploads per token per layer folded into `BackwardScratchCudarc` (18 new attention fields + K/V caches sized `[max_t_len x attn_buf]`; the GQA guarantee kvd <= q_dim <= attn_buf keeps every per-token prefix in bounds); readbacks via `try_slice` prefix views (`memcpy_dtoh` copies the WHOLE device slice); the grad_attn_out borrow-staging is now a `memcpy_dtod` (was dtoh + re-upload). **H7**: all five host zero-vec + PCIe writes to `memset_zeros` (also clears stale tails the t_len-prefix htod missed). **H8**: `dequant_transpose(&lm_head)` cached in the scratch keyed by `(ptr, rows, cols)` (hot-swap-cache family; in-place mutation not detected — recreate the scratch, documented in-source). **H13**: stale `gemv_multi` doc rewritten (the occupancy cap was REFUTED by Bench 684 — opt-in only). **H10 deferred `[-]`**: the `forward_token_with_layer_capture` hand-duplicated layer loop is diagnostic-only (Issue 717 G2-full); restructure when Issue 721 T3 touches the layer loop or a second capture consumer appears. Validation (GPU-exclusive): 43/43 tests incl. `test_gpu_backward_matches_cpu_reference` (the H5/H6/H7/H8/H9 numerics gate, unchanged 2e-2 tol) under `ternary_gemv_cuda_raw,speculative_decode,cuda_graphs_forward`; clippy 0 in touched files (only the pre-existing dead-scratch-fields warning remains); minimal feature configs compile clean. Record: git history (issue file removed at close; the full 13-hazard map is there). |

| **731** Q4_K per-position attention read a WRONG KV LAYOUT — `get_window` interleaved `[k0,v0,…]` vs the kernel's concatenated `[keys||values]` (the "pre-existing 4090 G1 divergence") | RESOLVED (2026-08-19, same-day discovery+fix; no issue file — `Bench 701` is the record) — the symptom Bench 487 (riir-train) recorded as "8.72e0 divergence on 4090, Issue-429-FMA hypothesis, G1-PASS-is-M3-scoped" was neither FMA nor 4090-specific: `gemma4_q4k_train::CpuKVCache::get_window` built the combined KV buffer **interleaved per position** while every `attention_cubecl` decode kernel derives `n_positions = (kv.len()/2)/kv_stride` and reads values at offset `kv_half` — the **concatenated** `AttentionParams::combine_kv` contract (`gemma2_cubecl` + `gemma4_cubecl` builders both correct; the Q4K trainer was the sole wrong one). Same total length ⇒ no length check fires; `n_pos=1` layouts coincide ⇒ pos 0 correct; for `n_pos≥2` the kernel scored attention against **v-rows mixed into the keys** (on the G1 fixture, unit-scale v-rows make those q·v scores ~100× the real q·k scores → softmax concentrates on garbage → attn_out ≈ ±1 vs correct small-mean — the exact observed 0.84), and the fixture's RMSNorm row-collapse made everything downstream bit-identical — the fingerprint that killed the FMA hypothesis (fp contraction cannot produce a bit-identical downstream of a differing input). **The M3 "diff = 0.0" record (2026-08-10) could not have been measured against the committed state** — backend-independent garbage; evidently a pre-commit WIP verification (per-position CPU attention) that never re-ran after the GPU wiring landed (`eab36fcb9`, 08-09); the 4090 was simply the first box to run the committed test. Fix: `get_window` → two-pass concatenated layout (contract documented in-source). **Coverage hole closed**: `AttentionCubeCL::launch` fast-paths only `(8,4,256,50)` to the hardcoded Gemma-2 kernel — every other shape routes to the PARAMETRIC `attention_decode_llama_f32` kernel, which had ZERO parity tests; added `verify_attention_parametric` + 3 gates (Gemma-4 Q4K 4/2/256 scale=1.0 n_pos∈{1,4,8}; real 12B sliding 16/8/256 n_pos=300; no-softcap 16/2/128) — max_err ≤ 1.2e-7. Post-fix G1: **8.72e0 → 1.91e-6 PASS** (deterministic, run twice); GPU-vs-CPU reference test unchanged-green; 30 q4k lib tests + riir-train plan320 c1/c2 green. Exposure: per-position path is the non-default debug escape hatch (`--no-batched-forward`); production 4090 runs (`--quant-resident`) auto-enable batched-forward (CPU attention) — no production checkpoint affected; early Plan-320-C3 per-position experiments (08-09, the "grad norms 10 orders larger on WDDM" runs) plausibly included this garbage. |

| **728** katgpt-core feature-set explosion: 301 variants, 31.3 GB — stale swept, convergence REFUTED | RESOLVED (2026-08-19; PARTIALLY CLOSED by design — the actionable half done, the convergence hypothesis killed by the G4 stop rule) — parent `723`'s top follow-up; katgpt-core was the fleet's real hot spot (652 lib fingerprints, 301 distinct feature sets, 36.36 GB `debug/deps` vs riir-engine's 4.86 GB), NOT the "opposite shape" the original title claimed. **Stale sweep DONE: 5.57 GB reclaimed** — 565 stale hashes stranded by the 2026-06-28 `cb3cb35c` eggshell-IP migration of `interest_cochain`/`dec_terrain_ai`/`lattice_utility` to riir-neuron-db + 3,843 orphaned integration-test artifacts (method note: sweep by HASH, not crate-name prefix — integration-test artifacts are named `<test_target>-<hash>`, so a prefix sweep strands a second orphan generation); post-sweep `distinct=244 live=244 stale=0`, re-verified at removal (2026-08-20) as `247/247/stale=0` — the sweep held. **The convergence mechanism is REFUTED (G4)**, three independent measurements: (a) the audit's `diff=1` consumer pairs were artifacts of ignoring `default-features` (the `default` closure is 82 features — the "closest" pairs were `default-features = true` vs `false`, i.e. the two MOST distant consumers, inverted by the metric); (b) consumers ALREADY converge (44 consumers → 25 distinct resolved sets, 190 pairs at distance 0) yet 244 live variants exist; (c) **ZERO of the 244 built variants matches any consumer's resolved set** — consumer manifests are not the generator. 86% of built variants carry the `--no-default-features --features <X>` invocation signature: the repo's own gate/validation matrix generates them by construction (the same verdict 723 reached for riir-engine, from the opposite direction). Corrections: "≈121 MB/variant" → measured median 11.3 MB / mean 59.5 MB per live lib set; the real per-variant cost is the uncounted `test-lib` harnesses (149 live × 84 MB = 12.3 GB — a TTL gate-cache sweep is the residual lever, deliberately not pursued; refile if disk pressure returns). Record: git history (issue file removed at close; the full measured-results section is preserved there). |

| **730** Metal 65535-workgroup grid cap: batched prefill kernels panic on Apple GPUs | RESOLVED (2026-08-19, same-day) — the DEFAULT-ON batched prefill path (Issue 653) + the GDN batched expand kernel launch flat 1D grids scaling with prompt tokens; Metal caps the x-dimension at 65535 workgroups (wgpu validation panics above; CUDA allows 2^31-1 — why the 4090-validated path never failed). At Bonsai-27B dims the first p=2048 prefill died at 294912 wg (GDN `expand_and_l2_normalize_heads_batched`, layer 0); found live by Issue 726's armed T1/T4 chain — **zero tok/s rows had ever been produced on Metal before this fix**. Fix: token/element-boundary chunked launches, bit-identical by construction (kernels derive token/head/dim from LOCAL flat indexes against per-chunk params; sliced handles land local indexes at the chunk's absolute offset; `QwenAttentionPrefillGatedCubeCL` gained `params[5] = q_offset` for the causal range) — **11 launches total, one more than the issue file's table recorded**: `d1fc5a34f` fixed 4 (expand_and_l2 294912, split_qg 98304, qk_fused, attention+q_offset), `8ec341743` the next 6 (z_gating — the 13:40 re-run panic, SwiGLU gating 278528@2K, ResidualAdd 81920@4K, rope, kvfill combined+split, split_kv — the long-length class), and `440b5ba54` (16:06) the **11th**: `RmsNormBatched` self-chunks on row boundaries (the GDN per-head norm widens rows to p×n_v_heads = 98304@2K; call-site row counts evade launcher probes — found via registry `type_name+track_caller` instrumentation after the 14:10–14:22 re-runs STILL panicked `[98304,1,1]`). The issue file's cited hashes `d6fd3dff3`/`6dcf003db` are the pre-rebase twins of `d1fc5a34f`/`8ec341743` (same messages/timestamps). Validated by the re-run 726 chain: **2048 prefill RUNS on Metal for the first time — hybrid 18.44 warmup / 18.67 tok/s run 0** (recorded in the still-open 726 issue file: "T4 GRID SAGA RESOLVED", commit `8968466ec`); the remaining T4 failure is an intermittent ANE bridge eval (`forward.rs:3781` fail-open panic) — Issue 726's domain, NOT the grid cap. Kernel units 68/68 norms/deltanet/qwen + 22/22 ane_prefill green at fix time (not re-run at removal — GPU-gated; chunk guards verified in source); 4096+ covered by the same guards (static audit: legal at all lengths), 16K/32K best-effort per the issue. Lesson: a dispatch-grid limit is a PER-BACKEND contract — CUDA-validated launches are not Metal-safe. **Numbering note (preserved from the removed file): originally filed as 729 locally at 13:55; a sibling session concurrently allocated 729 (`729_armc_gpu_backward_forward_excludes_lora.md`, landed on origin first); renumbered → 730 per the push-wins tie-breaker (commit `33ffd72df`). The fix commits' messages + some code comments cite "729" (and the 726-armed ones cite "726") — they refer to THIS issue.** Record: git history + the 726 issue file's grid-saga note. |

| **735** art-vessel pipeline: the designed production path never landed + 097/724 boundary audit | RESOLVED (2026-08-21, Plan 543 complete — owner fork call "prod grade only": T2(a) real WASM packer, not the (b) relabel stopgap) — the audit found the designed art→vessel path (Prop 031 §4) had landed only as an ASBL **stub** whose `.vesselbundle` output lied about the loader contract (never `ensure_compiled`-valid), `asset_manifest_hash` was missing from `GenesisManifest`, and the Finding-2/3 FLAGS showed `UpdateItemCatalog`/`UnlockShopSlot` failing the 097 product test + module-resident ix sets evading the 096 sweep. Landed across 5 repos (boundary-checked first — all 5 contracts clean): **riir-neuron-db `f5a6c5c`** — `vessel/art_bundle.rs` behind `art_vessel = ["secure_vessel"]`: `ARTB` payload kind, `ArtEntry` (76B) + `ArtBundleManifest` (19472B) Pods, `encode_art_bundle` over a hand-assembled REAL WASM module (memory+export+data, 4-aligned manifest offset, zero new deps), wasmi test pinning `ensure_compiled` SUCCESS — the property the stub could never satisfy. **riir-game-sdk `31a2554`** — asset-converter `vessel_pack` rewritten onto the real encoder (stub + ASBL deleted) + `--signing-key` (SSH Ed25519 → `.sig` sidecar) + genesis-commitment integration tests. **riir-ai `811108165`+`094cddf74`+`3eb7f0c60`** — `GenesisManifest.asset_manifest_hash` appended last (SIZE 268→300, append-last offset pinned, pinned world hash `9bed5032…` untouched). **riir-mmorpg-examples `db20ebd`** — boot adopt-or-verify at all 3 native boot sites via `RIIR_MMORPG_EXAMPLES_ASSET_VESSEL` (mismatch = hard boot error). **riir-chain `80f20c3a`** — Issue **106** filed (asset_lifecycle catalog/shop 097 FLAGS + riir-dapps migration) + module-resident-ix sweep section. T5 (runtime delivery) stays `[-]` — Plan 319 territory; Bevy vessel-backed `AssetReader` deferred (view-layer, own plan); `asset_vessel` stays DEFAULT-OFF until the renderer consumes vessels. Record: `Plan 543` + git history (issue file removed at close). |

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.

## Cross-Repo Dependency Direction (contract)

**Rule: a sibling repo depends on riir-ai *leaf and facade* crates only — never
on a domain crate it does not own.** A developer working on chain code must not
compile the game tier; a developer working on the game tier must not compile the
chain daemon. Dependencies flow DOWN from consumers into leaves, never sideways
between domains.

Measured consumer map (2026-08-18, from every sibling's `Cargo.toml`). Built with
an **anchored** dep-line grep (`^\s*<crate>\s*=`), not a bare name search — a
loose grep also matches the crate name inside comments and silently inflates the
table. `riir-ffi` in particular was wrongly attributed to `riir-game-sdk` on the
first pass for exactly that reason; `riir-chaind` is its **only** consumer
anywhere (one dep line, `optional = true, default-features = false`):

| repo | domain | riir-ai crates it may depend on |
|---|---|---|
| `riir-chain` | consensus / wallet / daemon | `riir-engine`, `riir-wasm`, `riir-ffi`, `riir-mcp-client` |
| `riir-train` | GPU training | `riir-data`, `riir-engine`, `riir-games`, `riir-games-civ`, `riir-gpu`, `riir-gpu-async`, `riir-router` |
| `riir-neuron-db` | storage / KG | `riir-engine`, `riir-games`, `riir-games-civ`, `riir-gpu`, `riir-router` |
| `riir-game-sdk` | game vocabulary facade | `riir-agents`, `riir-engine`, `riir-games`, `riir-games-mmorpg`, `riir-games-shared`, `riir-mcp-client`, `riir-net`, `riir-wasm` |
| `riir-mmorpg-examples` | game product | `riir-agents`, `riir-engine`, `riir-games`, `riir-games-mmorpg`, `riir-games-quest`, `riir-net`, `riir-simloop` |
| `riir-viewbridge` | Unity view bridge | `riir-engine`, `riir-games-shared`, `riir-net` |
| `riir-clippy` | lint / mining tooling | `riir-engine`, `riir-gpu`, `riir-rag` |
| `katgpt-rs` | modelless primitives | **none** — katgpt-rs is UPSTREAM of riir-ai; a riir-ai dep here would be a cycle |
| `riir-dapps` | dApp layer (game outcome → chain settlement) | **none** — game crates call *into* `riir-dapps`, never the reverse. A riir-ai dep there would close the loop the layer exists to open. |

### Game crates must not call chain programs directly

**A game crate reaches the chain through `riir-dapps`, never through a
`riir-chain` program.** The dApp layer (created 2026-08-20, `riir-chain` Issue
096 T1) exists precisely because both ends were reaching for each other.

**RESOLVED 2026-08-21 (riir-dapps Plan 001 §3.1, riir-chain 096 T4 + T3).** The
former violation — `crates/riir-games-civ/src/civ/latcal_wire.rs` (438 LOC,
17 direct program calls, the largest boundary crossing in the workspace) —
now composes bounty/quest/crafting through `riir_dapps::dapps::{bounty,quest,crafting}`
(returning a `Composed` settlement; domain-separated BLAKE3 predicates, i64
money, no `MatrixAccount` mutation in the game layer). The chain programs
themselves retired: Quest (17) + Bounty (11) + Crafting (13) are gone from
the ledger — tags 45..=48 / 54..=57 / 68..=71 are dead holes. A contested
craft's roll is the dApps `FairRoll` service (`riir_dapps::service::fair_roll`
over the chain's split-key substrate), not a program. The currency/authority
programs (System/Swap/Stake/Lending/Multisig/Vote/Auction/Escrow/Permission)
stay direct by design — they move value or bind authority, which is what the
ledger is for.

Before adding any new game→chain call, apply the **three-test rule**
(`riir-chain/AGENTS.md` §"Domain Boundary", `riir-chain/.issues/097`): a
commerce customer must want it in their dependency, it must be BigInt currency
/ a token / an authority binding, and its write rate must fit a Glacial tier
(≤0.1 Hz). FAME, XP, items, reputation and quest progress fail test 2 — they
belong in `riir-neuron-db`, which is **1,627× cheaper per write** and secure
enough (BLAKE3-committed, keyed row MACs). Most quests are free and settle
nothing at all.

### Why this is a written rule

**Canonical failure — the riir-ffi game-tier leak (Issue 724 Phase 1, fixed
2026-08-18).** `crates/riir-ffi/Cargo.toml` declared
`riir-games = { path = "../riir-games", features = [...] }` **non-optional**.
`riir-chaind` consumes riir-ffi correctly — `optional = true`,
`default-features = false`, and it only ever touches `LatentSidecar` /
`latent_mirror` / `ReduceOutcome`, none of which reference riir-games. But
because the dep was unconditional, enabling riir-ffi at all dragged in
`riir-games` → `riir-games-civ` + `riir-games-quest` + `riir-games-shared` +
`katgpt-core`. Measured effect: `riir_games`, `riir_games_civ`,
`riir_games_quest`, `riir_games_shared`, `katgpt_core`, `metal` and
`objc2_metal` artifacts all present in `riir-chain/target/debug/deps` — for a
7-crate consensus/wallet workspace. `riir-games-civ` alone is 155k LOC.

The consumer's hygiene was already correct. **The leak was in the facade.** A
facade crate that unconditionally depends on a domain crate defeats every
consumer's `default-features = false`.

Fix pattern: the domain dep becomes `optional = true` behind a feature that is
listed in `default` (so existing consumers are unaffected), and any type the
non-domain surface genuinely needs is taken from the *light shared* crate
instead — here `riir-games-shared` (`default = []`) for `types::MapPos2D`,
rather than all of `riir-games`.

### How to check before adding a dep

```bash
# From the consumer repo: does the resolved graph contain a crate you don't own?
cargo tree --edges normal --prefix none | sed 's/ v.*//' | sort -u | grep -E '^riir-'

# From riir-ai: prove a facade stays clean without its domain feature
cd crates/riir-ffi
cargo tree --no-default-features --edges normal --prefix none | grep -E '^riir-games' \
  || echo "clean"
```

A facade crate MUST be verified in both directions: with default features (the
game path still resolves) and with `--no-default-features` (the domain tier is
absent). Checking only one direction is how this class of leak survives.

### Enforcement status

Currently **convention only, not enforced** — nothing fails when a new
sideways dep is added, which is why the riir-ffi leak survived. A CI
dep-allowlist check is Issue 724 Phase 2's remaining task. Until it lands,
`riir-ffi` deserves specific suspicion: it is in the workspace `exclude` list
(root `Cargo.toml`), so `scripts/ci_feature_guard.sh`'s
`cargo check --workspace --all-targets` **never compiles it**. It silently
rotted for exactly that reason — `use riir_games::types::MapPos` outlived the
`MapPos` → `MapPos2D` rename (commit `d691ecdc5`) and the crate did not build
at all on `develop` until Issue 724 Phase 1. **Any workspace-excluded crate
needs its own explicit CI step.**
