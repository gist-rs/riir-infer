# Bench 888 — Plan 622 T2.5: the layer-choice measurement (acc-vs-K at L ∈ {0, 1, mid}, one NIAH suite)

**Status:** RECORD — T2.5 DONE; the prune-layer constant for OUR hybrid is **L = mid full-attention layer** (not upstream's L=1/L=3).

**Date:** 2026-10-25 (zed-m3-continue)
**Plan:** [.plans/622_tokengate_keybias_pruned_prefill_lane.md](../.plans/622_tokengate_keybias_pruned_prefill_lane.md) T2.5
**Instrument:** `src/bin/tokengate_layer_choice.rs` (new; `required-features = ["tokengate", "deltanet_inference"]`)
**Model:** `../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf` — qwen35-hybrid, 24 layers (GDN×2 / full-attn×2 pattern), n_head 8 / n_kv 2, head_dim 256, vocab 248320. RESTORED this session from the 4090 box's `E:\git\riir-train\data` copy via scp (811,843,200 bytes both sides — byte-count verified; LOCAL-KEEP, not a BACKUP_MANIFEST row; the `.env` HF token is rolled/401 so the `data_backup.sh pull` lane could not fetch the retired Q4_0 row).
**Corpus:** `../riir-train/data/chat_probe` (21 pages, restored from the same box).
**Suite:** the `needle_eviction_gate` NIAH rig minus the hub/wall regimes — filler + 4 needle sentences at depths {377, 746, 1115, 1484} + question + 48-token teacher-forced answer; metric = the gate's pre-registered per-needle NLL delta (`delta_i = NLL_i(arm) − NLL_i(full)`, retained iff ≤ 1.0 nat/token, retrieval = retained/4). SELECTION-ALONE posture: decode gate `None` (the steering overlay's L-sensitivity is T3.4's axis).
**Box:** M3 Max (m3), loadavg 4.07, mem free 93%, AC; release build; CPU forward. Total wall 482.7 s, seed 622.

## Table (ctx 2048, prunable 2043, scaffold = 4 sinks + last)

| cell | arm | kept | meanΔ nats | retrieval | wall_s |
|---|---|---|---|---|---|
| full (no prune) | — | 2048 | 0 | (reference) | 70.1 |
| L=0/K=0.25 | hiddennorm | 516 | 2.149 | 0.250 | 21.6 |
| L=0/K=0.50 | hiddennorm | 1027 | 1.899 | 0.250 | 37.7 |
| L=0/K=0.75 | hiddennorm | 1537 | 0.820 | 0.750 | 53.9 |
| L=1/K=0.25 | hiddennorm | 516 | 1.740 | 0.250 | 23.9 |
| L=1/K=0.50 | hiddennorm | 1027 | 1.520 | 0.250 | 39.4 |
| L=1/K=0.75 | hiddennorm | 1537 | 0.936 | 0.500 | 54.7 |
| **L=15/K=0.25** | **mass** | **516** | **0.150** | **1.000** | 53.6 |
| **L=15/K=0.50** | **mass** | **1027** | **0.048** | **1.000** | 59.4 |
| **L=15/K=0.75** | **mass** | **1537** | **−0.003** | **1.000** | 65.1 |

Full-arm needle NLL: [0.790, 0.143, 0.009, 0.004] nats/token.

## Findings

1. **The pin: L = mid (15), an attention layer, arm (a) mass.** Retrieval 1.000 at EVERY K measured — including K=0.25 (4× row reduction above L) — with meanΔ ≤ 0.15 nats. At K=0.75 the pruned arm's meanΔ is −0.003: dropping 25% of the mid-prompt rows at L=15 moves the needle NLL BELOW the full run's (noise-class, but the direction is not harm).
2. **The early-L placements fail on OUR hybrid** — L=0 and L=1 are GDN layers here (the 2×2 pattern), so arm (a) is structurally unavailable (no mass at a GDN mixer) and arm (b) hiddennorm retrieves 0.25–0.75. Upstream's "gate at L0 too weak, L≥1 sufficient" (dense L=1 / hybrid L=3) does NOT transfer: OUR hybrid's L=1 is a GDN layer and even the mixer-agnostic arm cannot rank the needles from position-1 hidden norms.
3. **The L constant for the workspace lane: the middle full-attention layer.** For the league/0.8B pattern (GDN² A² repeated), `L = the attention layer nearest n_layer/2` generalizes; the exact-grid generalization (upstream's dense L=1 analog for OTHER mixes) needs a second pattern — T3.4's harness runs at this measured L.
4. **Scorer-arm confound, named:** L=15 ran mass while L=0/1 ran hiddennorm (structurally forced — a GDN mixer has no mass). The L gradient conflates placement depth with scorer arm at the early Ls. The clean read is still possible: hiddennorm at L=15 would isolate the arm axis (a T3.4 ablation row; not re-run here — the T2.5 question was WHERE the gate goes, and the mass arm at mid answers it).
5. **Prefill compute at the winner:** L=15/K=0.25 prunes 75% of prunable rows above layer 15 (16 of 24 layers run ~516-row sequences) — wall 53.6 s vs 70.1 s full (−24% CPU wall; the layers ≤ L still run the full 2048-row pass, and the CPU forward's cost is dominated by the GDN layers' recurrence). The TTFT break-even grid is T4.1's row.

## En-route defects

- The bin's first splice-depth formula (the gate bin's slot indirection) COLLIDES when the slot count equals the needle count — both smoke needles landed at depth 44. Fixed to a direct even spread (`depth_i = usable·(i+1)/(needles+1)`), strictly increasing. (The shipped `needle_eviction_gate` at `--hubs 0` carries the same collision shape — `slot_of[s] = Needle(i)` overwrites, silently dropping an earlier needle from the prompt; NOT fixed in that bin this session — flagged for the kv_evict lane's owner, since its Bench-008-fixture runs may have measured a 3-needle prompt wearing a 4-needle label.)
- The `.env` HF token in riir-train is rolled (401 on the API) — the retired-row restore lane (`data_backup.sh pull`) is down until the owner refreshes; the model came from the 4090 box instead.

## Provenance

- Run: `/tmp/p622t/t25_run.md` (stdout) + `t25_run.log` (stderr progress) — reproduced by `cargo run --release --features tokengate,deltanet_inference --bin tokengate_layer_choice -- --context 2048 --needles 4 --k-fracs 0.25,0.5,0.75` from the riir-infer root with the sibling layout.
- The bin lands in the same commit as this record; the plan tick carries the commit hash.
