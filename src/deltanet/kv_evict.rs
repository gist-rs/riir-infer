//! Differential KV eviction wired onto the hybrid attention KV path —
//! riir-infer Issue 012, the model-bound quality gate for katgpt-rs Issue
//! 882 P3 (the `DifferentialEvictTable` primitive, katgpt-rs Bench 894).
//!
//! # The wiring
//!
//! The forward seam is `Option<&mut EvictorState>` on the qwen35-hybrid
//! attention layer paths (decode `forward_attention_layer_evictable`,
//! prefill `prefill_qwen_deltanet_chunk_into`). Per attention layer and per
//! Q head, one side table observes the post-softmax row the forward already
//! computes (`head_scores` — no kernel change, the kv_eviction house
//! pattern). When the live slot count passes the per-layer budget, eviction
//! selects through the shipped sink-exempt rule and the cache compacts in
//! place: retained K/V rows gather to the front, and the side tables +
//! slot→logical-position map gather with them (the `gather_rows` twin, so
//! slot indexing never desynchronizes).
//!
//! # Design decisions this issue owns (all pre-registered in the gate bin)
//!
//! - **One budget per layer; selection aggregates across heads by MEAN.**
//!   The KV rows of one layer are shared by its Q heads (GQA), so eviction
//!   is a single per-layer decision; the per-head specificities (or usage
//!   scores) are averaged into the selection score. Per-head tables still
//!   observe per-head rows — the aggregation is at selection only.
//! - **RoPE keeps the logical position.** Compaction changes a token's
//!   physical slot, never its absolute position: K/V were already roped at
//!   write time from the logical position, so surviving relative phases are
//!   exact, and attention over the compacted prefix sees the same values
//!   the full cache would have (minus evicted rows).
//! - **Cadence.** Selection (the O(live log live) sort) runs at most every
//!   `cadence` positions; between selections the cache may overshoot the
//!   budget by up to `cadence − 1` rows. The caller's cache allocation must
//!   hold `budget + cadence` rows (the gate bin sizes it from the context).
//! - **T3, by construction.** With `budget ≥` the whole sequence, `evict‑
//!   ions_for_budget` returns 0 every step: admissions and observations are
//!   pure side state and the write path reduces to the existing one (slot
//!   == position, `t_n == pos + 1`), so the logits are `to_bits`-identical
//!   to the unarmed path. The gate bin asserts this on real weights.
//!
//! # Sync boundary
//!
//! None. Latent-side bookkeeping over the model's own attention masses;
//! nothing here is synced or committed.

use katgpt_core::kv_eviction::differential::{
    DiffEvictConfig, DifferentialEvictTable, evictions_for_budget,
};
use katgpt_core::kv_eviction::{UsageScoreTable, select_evict_into};
use katgpt_core::kv_sink_window::{SinkWindowPolicy, sink_pin_mask_into};
use katgpt_transformer::KVCache;

/// The eviction score family the gate compares (T2's arms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvictPolicy {
    /// The differential specificity score at `(λ, β, W)`. `λ = 0` is the
    /// max-recent baseline, bit-identically (Bench 894 reduction).
    Differential(DiffEvictConfig),
    /// The shipped usage-rate score `cum_mass / max(1, age)`; age in
    /// logical positions.
    UsageRate,
    /// The prompt-pinned random null: one uniform draw per live slot per
    /// selection, from a per-(layer, head)-free deterministic stream seeded
    /// by `seed` (layer-level draws — the same aggregation level as the
    /// scored policies' selection).
    Random { seed: u64 },
}

/// Per-layer eviction configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvictLayerConfig {
    pub policy: EvictPolicy,
    /// Live-slot ceiling per attention layer.
    pub budget: usize,
    /// Minimum positions between selection attempts.
    pub cadence: usize,
    /// Logical position eviction may START at. `0` = the pure streaming
    /// protocol (selection from the first overshoot); `prompt_len` = the
    /// deferred protocol — the whole prompt prefills into the FULL cache and
    /// the first compression fires at decode start, on evidence that
    /// includes the question's own queries. The deferral is the regime the
    /// primitive's synthetic win describes (Bench 894: a needle's recent
    /// evidence must be able to COMPETE at eviction time; during-haystack
    /// streaming evicts every needle long before the question arrives,
    /// because no query attends it while its evidence window is live).
    /// Headroom posture (`budget ≥` everything) is unaffected — the gate is
    /// `k == 0` either way, so T3 holds under both protocols.
    pub defer_until: u64,
}

impl EvictLayerConfig {
    /// The λ = 0 max-recent baseline arm (the same machinery, one const).
    pub fn max_recent(budget: usize, cadence: usize, window: u32) -> Self {
        Self {
            policy: EvictPolicy::Differential(DiffEvictConfig::max_recent(window)),
            budget,
            cadence,
            defer_until: 0,
        }
    }
}

/// One attention layer's eviction state: per-head tables, the slot→logical
/// map, and the compaction scratch. `live` is the number of assigned slots;
/// every assigned slot is causal (slots are assigned at write time, and the
/// staged-prefill path writes a position's K/V only when that position
/// attends).
pub struct LayerEvict {
    cfg: EvictLayerConfig,
    sinks: SinkWindowPolicy,
    n_heads: usize,
    diff: Vec<DifferentialEvictTable>,
    usage: Vec<UsageScoreTable>,
    random: Option<katgpt_types::Rng>,
    /// Logical (absolute) token position of each physical slot.
    slot_logical: Vec<u64>,
    /// Positions since the last selection attempt (cadence counter).
    since_select: usize,
    // ── scratch (allocated once, reused — G4) ──
    head_scores_scratch: Vec<f32>,
    scores: Vec<f32>,
    pin: Vec<bool>,
    evict_sel: Vec<usize>,
    is_evicted: Vec<bool>,
    keep: Vec<usize>,
    /// Per-selection event counter (the UsageRate tick source is the
    /// logical position; this counts selections for the random stream only).
    stats: EvictStats,
}

/// Counters the gate bin reports (how lossy the run actually was).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvictStats {
    /// K/V rows evicted (per layer, summed by the bin).
    pub evicted_rows: u64,
    /// Selection attempts that evicted something.
    pub evict_events: u64,
    /// Selection attempts that found nothing to do (live ≤ budget).
    pub idle_events: u64,
}

impl LayerEvict {
    /// Allocate for one attention layer: `n_heads` per-head tables over
    /// `capacity` slots. `capacity` must hold the whole sequence for the
    /// no-eviction posture (budget ≥ context ⇒ the tables never gather).
    pub fn new(
        cfg: EvictLayerConfig,
        sinks: SinkWindowPolicy,
        n_heads: usize,
        capacity: usize,
    ) -> Self {
        let diff = match cfg.policy {
            EvictPolicy::Differential(dcfg) => (0..n_heads)
                .map(|_| DifferentialEvictTable::with_capacity(capacity, dcfg))
                .collect(),
            _ => Vec::new(),
        };
        let usage = match cfg.policy {
            EvictPolicy::UsageRate => (0..n_heads)
                .map(|_| UsageScoreTable::with_capacity(capacity))
                .collect(),
            _ => Vec::new(),
        };
        let random = match cfg.policy {
            EvictPolicy::Random { seed } => Some(katgpt_types::Rng::new(seed)),
            _ => None,
        };
        Self {
            cfg,
            sinks,
            n_heads,
            diff,
            usage,
            random,
            slot_logical: Vec::with_capacity(capacity),
            since_select: 0,
            head_scores_scratch: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
            pin: Vec::with_capacity(capacity),
            evict_sel: Vec::with_capacity(capacity.min(64)),
            is_evicted: Vec::with_capacity(capacity),
            keep: Vec::with_capacity(capacity),
            stats: EvictStats::default(),
        }
    }

    /// Live slot count.
    pub fn live(&self) -> usize {
        self.slot_logical.len()
    }

    /// Admit one slot for the token at absolute position `pos`; returns the
    /// physical slot to write K/V into (always the packed tail).
    pub fn admit(&mut self, pos: u64) -> usize {
        let slot = self.slot_logical.len();
        self.slot_logical.push(pos);
        match self.cfg.policy {
            EvictPolicy::Differential(_) => {
                for t in &mut self.diff {
                    t.reset_row(slot);
                }
            }
            EvictPolicy::UsageRate => {
                for t in &mut self.usage {
                    t.reset_row(slot, pos);
                }
            }
            EvictPolicy::Random { .. } => {}
        }
        slot
    }

    /// Observe one query's per-head softmax rows. `head_scores` is laid out
    /// `[h * stride + t]` for `t` in `0..live` (the forward's `head_scores`
    /// scratch, `stride = block_size`). The logical position `pos` is the
    /// UsageRate tick. Each head's table observes its own row; nothing here
    /// aggregates.
    pub fn observe(&mut self, head_scores: &[f32], stride: usize, pos: u64) {
        let live = self.live();
        debug_assert!(
            live <= stride,
            "live {live} exceeds head_scores stride {stride}"
        );
        match self.cfg.policy {
            EvictPolicy::Differential(_) => {
                for (h, t) in self.diff.iter_mut().enumerate() {
                    t.observe_query(&head_scores[h * stride..h * stride + live]);
                }
            }
            EvictPolicy::UsageRate => {
                for (h, t) in self.usage.iter_mut().enumerate() {
                    let row = &head_scores[h * stride..h * stride + live];
                    for (slot, &m) in row.iter().enumerate() {
                        katgpt_core::kv_eviction::observe(t.row_mut(slot), m, pos);
                    }
                }
            }
            EvictPolicy::Random { .. } => {}
        }
    }

    /// Mean-aggregate the per-head scores into `self.scores` (selection
    /// level). Differential reads specificities; usage reads usage scores;
    /// random draws one uniform per slot.
    fn aggregate_scores(&mut self, pos: u64) {
        let live = self.live();
        self.scores.clear();
        match self.cfg.policy {
            EvictPolicy::Differential(_) => {
                self.scores.resize(live, 0.0);
                for t in &self.diff {
                    t.specificity_into(&mut self.head_scores_scratch);
                    for (s, &v) in self.scores.iter_mut().zip(self.head_scores_scratch.iter()) {
                        *s += v;
                    }
                }
                for s in &mut self.scores {
                    *s /= self.n_heads as f32;
                }
            }
            EvictPolicy::UsageRate => {
                self.scores.resize(live, 0.0);
                for t in &self.usage {
                    t.scores(pos, &mut self.head_scores_scratch);
                    for (s, &v) in self.scores.iter_mut().zip(self.head_scores_scratch.iter()) {
                        *s += v;
                    }
                }
                for s in &mut self.scores {
                    *s /= self.n_heads as f32;
                }
            }
            EvictPolicy::Random { .. } => {
                let rng = self.random.as_mut().expect("random policy without stream");
                self.scores.extend((0..live).map(|_| rng.uniform()));
            }
        }
    }

    /// Evict to budget when due. `cache` is this layer's KV cache
    /// (`key`/`value` row-major, `kvd` floats per row). Returns `true` when
    /// rows were evicted.
    pub fn maybe_evict(&mut self, cache: &mut KVCache, kvd: usize, pos: u64) -> bool {
        if pos < self.cfg.defer_until {
            return false;
        }
        if self.since_select < self.cfg.cadence {
            return false;
        }
        self.since_select = 0;
        let live = self.live();
        let k = evictions_for_budget(live, self.cfg.budget);
        if k == 0 {
            self.stats.idle_events += 1;
            return false;
        }
        self.aggregate_scores(pos);
        // Sink-exempt selection over the aggregated scores. `positions` is
        // the slot→logical map; `pos` the current logical position.
        sink_pin_mask_into(&self.sinks, &self.slot_logical, pos, &mut self.pin);
        select_evict_into(&self.scores, k, &self.pin, &mut self.evict_sel);
        if self.evict_sel.is_empty() {
            self.stats.idle_events += 1;
            return false;
        }
        // Mark + gather. `evict_sel` is in eviction-priority order; the
        // keep-list is the ascending complement.
        self.is_evicted.clear();
        self.is_evicted.resize(live, false);
        for &e in &self.evict_sel {
            self.is_evicted[e] = true;
        }
        self.keep.clear();
        for (slot, &ev) in self.is_evicted.iter().enumerate() {
            if !ev {
                self.keep.push(slot);
            }
        }
        // K/V rows gather to the front (stable, in place — the write index
        // never passes the read index).
        for (w, &r) in self.keep.iter().enumerate() {
            if r != w {
                let (dst, src) = (w * kvd, r * kvd);
                cache.key.copy_within(src..src + kvd, dst);
                cache.value.copy_within(src..src + kvd, dst);
            }
        }
        // The side state gathers the same way.
        match self.cfg.policy {
            EvictPolicy::Differential(_) => {
                for t in &mut self.diff {
                    t.gather_rows(&self.keep);
                }
            }
            EvictPolicy::UsageRate => {
                for t in &mut self.usage {
                    t.gather_rows(&self.keep);
                }
            }
            EvictPolicy::Random { .. } => {}
        }
        for j in 0..self.keep.len() {
            self.slot_logical[j] = self.slot_logical[self.keep[j]];
        }
        self.slot_logical.truncate(self.keep.len());
        let evicted = live - self.keep.len();
        self.stats.evicted_rows += evicted as u64;
        self.stats.evict_events += 1;
        true
    }

    /// Advance the cadence counter (call once per attended position).
    #[inline]
    pub fn tick(&mut self) {
        self.since_select += 1;
    }

    /// Counters for the run report.
    pub fn stats(&self) -> EvictStats {
        self.stats
    }

    /// Reverse map: the physical slot currently holding logical position
    /// `pos` (linear scan — the gate bin calls this a handful of times per
    /// needle, never per step).
    pub fn slot_of_logical(&self, pos: u64) -> Option<usize> {
        self.slot_logical.iter().position(|&p| p == pos)
    }
}

/// Whole-forward eviction state: one [`LayerEvict`] per attention layer.
/// Attention layers are identified by the same `DeltaNetLayerType::Attention`
/// convention the forward dispatches on; the state is indexed by LAYER
/// index (non-attention layers simply never touch their entry).
pub struct EvictorState {
    layers: Vec<LayerEvict>,
}

impl EvictorState {
    /// Build for a hybrid forward: `layer_is_attention[i]` selects which
    /// layers get state. Non-attention entries are zero-budget placeholders
    /// that are never consulted.
    pub fn new(
        layer_cfg: Option<&EvictLayerConfig>,
        sinks: SinkWindowPolicy,
        layer_is_attention: &[bool],
        n_heads: usize,
        capacity: usize,
    ) -> Self {
        let layers = layer_is_attention
            .iter()
            .map(|&is_attn| match (is_attn, layer_cfg) {
                (true, Some(cfg)) => LayerEvict::new(*cfg, sinks, n_heads, capacity),
                _ => LayerEvict::new(
                    EvictLayerConfig {
                        policy: EvictPolicy::Differential(DiffEvictConfig::max_recent(1)),
                        budget: 0,
                        cadence: usize::MAX,
                        defer_until: 0,
                    },
                    sinks,
                    n_heads,
                    1,
                ),
            })
            .collect();
        Self { layers }
    }

    /// Per-layer access for the forward seam.
    pub fn layer(&mut self, idx: usize) -> &mut LayerEvict {
        &mut self.layers[idx]
    }

    /// Summed counters across layers.
    pub fn total_stats(&self) -> EvictStats {
        self.layers
            .iter()
            .fold(EvictStats::default(), |acc, l| EvictStats {
                evicted_rows: acc.evicted_rows + l.stats().evicted_rows,
                evict_events: acc.evict_events + l.stats().evict_events,
                idle_events: acc.idle_events + l.stats().idle_events,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::forward::{
        HybridCache, HybridForwardScratch, PrefillContext, forward_qwen_deltanet,
        forward_qwen_deltanet_evictable, prefill_qwen_deltanet, prefill_qwen_deltanet_chunk_into,
    };
    use crate::deltanet::weights::QwenDeltaNetWeights;

    fn sinks() -> SinkWindowPolicy {
        SinkWindowPolicy::new(4, usize::MAX)
    }

    /// A hub (constant mass every query) collapses to low specificity at
    /// λ = 1; a one-off spike keeps nearly all of it; zero-mass rows sit at
    /// −∞ and go first. Budget forces eviction; the survivors must be the
    /// needle + sinks + whatever the score ranks above the hubs, the cache
    /// rows must compact with the gather, and the side table must keep the
    /// survivors' pre-eviction state at their NEW indices.
    #[test]
    fn evict_compacts_cache_and_tables_together() {
        let cap = 16usize;
        let kvd = 4usize;
        let n_heads = 1usize;
        let cfg = EvictLayerConfig {
            policy: EvictPolicy::Differential(DiffEvictConfig::new(1.0, 0.5, 4)),
            budget: 8,
            cadence: 1,
            defer_until: 0,
        };
        let mut ev = LayerEvict::new(cfg, sinks(), n_heads, cap);
        let mut cache = KVCache {
            key: vec![0.0; cap * kvd],
            value: vec![0.0; cap * kvd],
        };

        // Distinct K rows so compaction is observable: key[slot] = [slot].
        for slot in 0..cap {
            for d in 0..kvd {
                cache.key[slot * kvd + d] = slot as f32;
            }
        }

        // Admit 13 slots (4 sinks at 0..4, then hub×2, needle, filler…).
        // Slot→logical identity here (pos == slot).
        let live_target = 13usize;
        let mut head_scores = vec![0.0f32; n_heads * cap];
        for pos in 0..live_target {
            let slot = ev.admit(pos as u64);
            assert_eq!(slot, pos, "pre-eviction slots must pack in order");
            ev.tick();
        }
        assert_eq!(ev.live(), live_target);

        // Crafted observation history: hubs 4/5 at 0.3 every query; needle
        // 6 spikes 0.5 on the LAST query only; 7..13 zero.
        for q in 0..20 {
            let mut m = vec![0.0f32; live_target];
            m[4] = 0.3;
            m[5] = 0.3;
            if q == 19 {
                m[6] = 0.5;
            }
            head_scores[..live_target].copy_from_slice(&m);
            ev.observe(&head_scores, cap, live_target as u64);
            ev.tick();
        }

        // Snapshot survivor state pre-eviction (for the gather check).
        let pre_spec_4 = ev.diff[0].specificity(4);
        let pre_spec_6 = ev.diff[0].specificity(6);
        let pre_mu_5 = ev.diff[0].mass_ema(5);
        assert!(pre_spec_6 > pre_spec_4, "needle must out-score the hub");

        let did = ev.maybe_evict(&mut cache, kvd, live_target as u64);
        assert!(did, "13 live against budget 8 must evict 5");
        assert_eq!(ev.live(), 8);
        assert_eq!(ev.stats().evicted_rows, 5);

        // The needle and both hubs SURVIVE? No — at λ=1 the hubs are the
        // LOWEST-scoring non-−∞ rows: the zero rows (7..13) rank −∞ first.
        // live 13 − budget 8 = 5 evictions, all from the −∞ tie group, and
        // ties break by ascending index — so 7..12 go and slot 12 (also
        // −∞) survives on the tie-break.
        let surviving: Vec<u64> = ev.slot_logical.clone();
        assert!(surviving.contains(&6), "needle must survive");
        assert!(
            surviving.contains(&4) && surviving.contains(&5),
            "hubs survive (zero rows evicted first)"
        );
        assert_eq!(surviving.len(), 8);
        for gone in 7..12u64 {
            assert!(
                !surviving.contains(&gone),
                "zero-mass slot {gone} must be evicted"
            );
        }

        // Cache compacted: a survivor's key row moved WITH it. Needle was
        // slot 6 pre-eviction with key [6,...]; its new slot is wherever
        // slot_of_logical(6) says, and the row must still read [6,...].
        let needle_slot = ev.slot_of_logical(6).expect("needle retained");
        assert_eq!(cache.key[needle_slot * kvd], 6.0);
        // Descending order preserved: surviving logical positions ascend
        // with slots.
        for w in 1..surviving.len() {
            assert!(surviving[w - 1] < surviving[w], "gather must keep order");
        }

        // Tables gathered: hub 4's specificity at its NEW index equals its
        // pre-eviction value; same for the needle and hub 5's μ.
        let new4 = ev.slot_of_logical(4).unwrap();
        let new5 = ev.slot_of_logical(5).unwrap();
        let new6 = needle_slot;
        assert_eq!(ev.diff[0].specificity(new4), pre_spec_4);
        assert_eq!(ev.diff[0].specificity(new6), pre_spec_6);
        assert_eq!(ev.diff[0].mass_ema(new5), pre_mu_5);

        // The table's query counter was untouched (buckets are time).
        // Observe continues post-gather from the moved row's μ.
        let mu_before = ev.diff[0].mass_ema(new4);
        head_scores.fill(0.0);
        head_scores[new4] = 0.1;
        ev.observe(&head_scores, cap, 99);
        let want = mu_before + 0.5 * (0.1 - mu_before);
        assert!((ev.diff[0].mass_ema(new4) - want).abs() < 1e-6);
    }

    /// Cadence: no selection before `cadence` positions have passed.
    #[test]
    fn cadence_blocks_selection() {
        let cfg = EvictLayerConfig {
            policy: EvictPolicy::Differential(DiffEvictConfig::new(1.0, 0.5, 4)),
            budget: 2,
            cadence: 4,
            defer_until: 0,
        };
        let mut ev = LayerEvict::new(cfg, sinks(), 1, 64);
        let mut cache = KVCache {
            key: vec![0.0; 64 * 4],
            value: vec![0.0; 64 * 4],
        };
        for pos in 0..8u64 {
            ev.admit(pos);
            ev.tick();
            // budget 2 < live from pos 2 on, but the cadence gate blocks
            // selection until 4 positions have passed.
        }
        assert_eq!(ev.live(), 8, "nothing evicted while cadence blocked");
        assert!(ev.maybe_evict(&mut cache, 4, 8));
        // live 8, budget 2, k = 6 — but slots 0..4 are PINNED sinks, so
        // only the 4 unpinned rows can go. Everything unpinned is evicted
        // and the sinks remain.
        assert_eq!(ev.live(), 4);
        let surviving: Vec<u64> = ev.slot_logical.clone();
        assert_eq!(surviving, vec![0, 1, 2, 3], "only the sinks survive");
    }

    /// Budget ≥ live never evicts (the T3 precondition at the state level).
    #[test]
    fn headroom_budget_is_a_no_op() {
        let cfg = EvictLayerConfig {
            policy: EvictPolicy::Differential(DiffEvictConfig::new(1.5, 0.5, 4)),
            budget: usize::MAX,
            cadence: 1,
            defer_until: 0,
        };
        let mut ev = LayerEvict::new(cfg, sinks(), 1, 32);
        let mut cache = KVCache {
            key: vec![0.0; 32 * 4],
            value: vec![0.0; 32 * 4],
        };
        let mut hs = vec![0.0f32; 32];
        for pos in 0..16u64 {
            ev.admit(pos);
            hs[pos as usize] = 0.25;
            ev.observe(&hs, 32, pos);
            ev.tick();
            assert!(!ev.maybe_evict(&mut cache, 4, pos));
        }
        // No EVICTION ever fires (headroom is the T3 posture); the idle
        // counter counts the no-op selection attempts, by design.
        assert_eq!(ev.stats().evicted_rows, 0);
        assert_eq!(ev.stats().evict_events, 0);
        assert_eq!(ev.stats().idle_events, 16);
        assert_eq!(ev.live(), 16);
    }

    // ── T3: the wired forwards are bit-identical without eviction ──

    fn tiny_hybrid() -> crate::types::Config {
        use crate::types::DeltaNetLayerType::*;
        let mut config =
            crate::types::Config::qwen_deltanet(4, vec![DeltaNet, Attention, DeltaNet, Attention]);
        config.vocab_size = 97;
        config.block_size = 512;
        config
    }

    /// Non-degenerate weights: EVERY dense projection filled deterministically
    /// (zeros elsewhere hid a real wiring bug — the batched-deltanet residual
    /// add — because zero out_proj made the wrong base of the residual add
    /// invisible; the real-model bisect caught it, this test now must too).
    fn perturbed_weights(config: &crate::types::Config) -> QwenDeltaNetWeights {
        let mut w = QwenDeltaNetWeights::zeros(config);
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut fill = |dst: &mut [f32]| {
            for val in dst.iter_mut() {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *val = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
        };
        fill(&mut w.wte);
        fill(&mut w.final_norm);
        fill(w.lm_head.dense_data_mut());
        for layer in &mut w.layers {
            fill(&mut layer.input_norm);
            fill(&mut layer.post_attn_norm);
            if layer.attn_wq.is_empty() {
                fill(layer.in_proj_qkv.dense_data_mut());
                fill(layer.in_proj_z.dense_data_mut());
                fill(layer.in_proj_a.dense_data_mut());
                fill(layer.in_proj_b.dense_data_mut());
                fill(layer.out_proj.dense_data_mut());
                fill(&mut layer.conv1d_weight);
                fill(&mut layer.dt_bias);
                fill(&mut layer.a_log);
                fill(&mut layer.linear_norm);
            } else {
                fill(layer.attn_wq.dense_data_mut());
                fill(layer.attn_wk.dense_data_mut());
                fill(layer.attn_wv.dense_data_mut());
                fill(layer.attn_wo.dense_data_mut());
                fill(&mut layer.attn_q_norm);
                fill(&mut layer.attn_k_norm);
            }
        }
        w
    }

    fn hybrid_fixture(
        config: &crate::types::Config,
    ) -> (
        HybridCache,
        HybridForwardScratch,
        crate::rope::RopeFreqTable,
    ) {
        let layer_types = config.layer_types.clone();
        (
            HybridCache::with_layer_types(config, &layer_types),
            HybridForwardScratch::new(config),
            crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim),
        )
    }

    /// T3 (prefill): armed-with-headroom chunked prefill is `to_bits`
    /// identical to the unarmed chunked prefill, and chunking itself is
    /// bit-transparent against the legacy whole-prompt path.
    #[test]
    fn t3_armed_headroom_and_chunking_are_bit_identical() {
        let config = tiny_hybrid();
        let weights = perturbed_weights(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let tokens: Vec<usize> = (0..16).map(|i| 3 + (i * 5) % 90).collect();
        let v = config.vocab_size;

        // Legacy whole-prompt prefill.
        let (mut cache_l, mut scratch_l, rf_l) = hybrid_fixture(&config);
        let mut pctx_l = PrefillContext::new(&config, 16);
        let logits_legacy = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache_l,
            &tokens,
            &mut scratch_l,
            &rf_l,
            &mut pctx_l,
        );

        // Chunked, unarmed (2 × 8).
        let (mut cache_c, mut scratch_c, rf_c) = hybrid_fixture(&config);
        let mut pctx_c = PrefillContext::new(&config, 8);
        let mut logits_chunk = vec![0.0f32; v];
        for (c, chunk) in tokens.chunks(8).enumerate() {
            prefill_qwen_deltanet_chunk_into(
                &weights,
                &config,
                &mut cache_c,
                chunk,
                c * 8,
                &mut scratch_c,
                &rf_c,
                &mut pctx_c,
                &mut logits_chunk,
                None,
            );
        }

        // Chunked, ARMED with headroom (no eviction possible).
        let (mut cache_a, mut scratch_a, rf_a) = hybrid_fixture(&config);
        let mut pctx_a = PrefillContext::new(&config, 8);
        let mut logits_armed = vec![0.0f32; v];
        {
            let layer_attn: Vec<bool> = config
                .layer_types
                .iter()
                .map(|&t| t == crate::types::DeltaNetLayerType::Attention)
                .collect();
            let cfg = EvictLayerConfig {
                policy: EvictPolicy::Differential(DiffEvictConfig::new(1.0, 0.5, 4)),
                budget: usize::MAX,
                cadence: 1,
                defer_until: 0,
            };
            let mut evictor =
                EvictorState::new(Some(&cfg), sinks(), &layer_attn, config.n_head, 512);
            for (c, chunk) in tokens.chunks(8).enumerate() {
                prefill_qwen_deltanet_chunk_into(
                    &weights,
                    &config,
                    &mut cache_a,
                    chunk,
                    c * 8,
                    &mut scratch_a,
                    &rf_a,
                    &mut pctx_a,
                    &mut logits_armed,
                    Some(&mut evictor),
                );
            }
            assert_eq!(evictor.total_stats().evicted_rows, 0);
        }

        assert_eq!(logits_legacy.len(), v);
        for i in 0..v {
            assert_eq!(
                logits_legacy[i].to_bits(),
                logits_chunk[i].to_bits(),
                "chunking must be bit-transparent [{i}]"
            );
            assert_eq!(
                logits_legacy[i].to_bits(),
                logits_armed[i].to_bits(),
                "armed headroom must be bit-identical (T3) [{i}]"
            );
        }
    }

    /// T3 (decode): armed-with-headroom decode is `to_bits` identical to
    /// the unarmed decode over a full continuation, and the armed state
    /// never fired.
    #[test]
    fn t3_armed_decode_bit_identical() {
        let config = tiny_hybrid();
        let weights = perturbed_weights(&config);
        let prompt: Vec<usize> = (0..8).map(|i| 3 + (i * 11) % 90).collect();
        let v = config.vocab_size;

        let run = |armed: bool| -> (Vec<f32>, EvictStats) {
            let (mut cache, mut scratch, rf) = hybrid_fixture(&config);
            let mut pctx = PrefillContext::new(&config, 8);
            let mut logits = vec![0.0f32; v];
            let layer_attn: Vec<bool> = config
                .layer_types
                .iter()
                .map(|&t| t == crate::types::DeltaNetLayerType::Attention)
                .collect();
            let cfg = EvictLayerConfig {
                policy: EvictPolicy::Differential(DiffEvictConfig::new(1.0, 0.5, 4)),
                budget: usize::MAX,
                cadence: 1,
                defer_until: 0,
            };
            let mut evictor =
                EvictorState::new(Some(&cfg), sinks(), &layer_attn, config.n_head, 512);
            let mut x = vec![0.0f32; v.max(config.n_embd)];
            prefill_qwen_deltanet_chunk_into(
                &weights,
                &config,
                &mut cache,
                &prompt,
                0,
                &mut scratch,
                &rf,
                &mut pctx,
                &mut logits,
                armed.then_some(&mut evictor),
            );
            let mut seq = prompt.clone();
            for step in 0..4 {
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                seq.push(next);
                let _ = step;
                let out = if armed {
                    forward_qwen_deltanet_evictable(
                        &mut x,
                        &weights,
                        &mut cache,
                        next,
                        seq.len() - 1,
                        &config,
                        &mut scratch,
                        &rf,
                        &mut evictor,
                    )
                } else {
                    forward_qwen_deltanet(
                        &mut x,
                        &weights,
                        &mut cache,
                        next,
                        seq.len() - 1,
                        &config,
                        &mut scratch,
                        &rf,
                    )
                };
                logits.copy_from_slice(&out[..v]);
            }
            (logits.clone(), evictor.total_stats())
        };

        let (plain, _) = run(false);
        let (armed, stats) = run(true);
        assert_eq!(stats.evicted_rows, 0);
        for i in 0..v {
            assert_eq!(plain[i].to_bits(), armed[i].to_bits(), "[{i}]");
        }
    }
}
