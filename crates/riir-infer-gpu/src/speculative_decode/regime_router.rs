//! Issue 755 T3 — the regime router: per-request drafter-lane selection.
//!
//! The record's own mechanism split (Bench 742 vs Bench 759): the qwen38
//! stack has TWO measured winning lanes that dominate in DISJOINT text
//! regimes —
//!
//! - **Lookup lane** ([`NgramDrafter`] + p≤16 graph verify chunks, the
//!   `t9_verify_loop_online` construction): 15.06/16 acceptance on
//!   doc-repro (instruction-driven quoting), 203.1 tok/s @20K — the doc /
//!   repetitive / copying regime. Self-gating: on a needle miss the fill
//!   returns `filled == 0` → p=1 chunks ≈ plain greedy decode (the lane
//!   never LOSES to the baseline it replaced).
//! - **DFlash2 lane** (`DFlash2GpuDrafter` + tap chunks + GDN journal
//!   replay, the `bench_759` construction): 1.081–1.110× vs greedy on
//!   novel/chat text (the walk-parallel + replay levers, G1 bit-identical)
//!   — the novel regime.
//!
//! T3 routes each REQUEST to its lane. The design exploits one measured
//! asymmetry: **switching DFlash2 → lookup is FREE** (the lookup drafter is
//! a CPU table over the committed context; the model state is shared), while
//! switching lookup → DFlash2 needs a capture re-fill (prompt re-run with
//! tap capture — prohibitive at long context, ~prompt_len × decode_ms).
//!
//! ## The needle watch (the free regime signal)
//!
//! During generation, the lookup needle on the COMMITTED tail measures
//! exactly the condition under which lookup acceptance is high: "is the
//! output continuing from context?" On doc-repro the generated tail matches
//! the doc in the prompt → the needle fills (15/15 in the t9 signature).
//! On novel output the tail does not repeat → miss. This is computed per
//! cycle on the CPU (a hash lookup, ~µs — the same call the lookup lane
//! would make anyway) and is therefore a FREE first-cycle-acceptance probe:
//! it predicts acceptance WITHOUT spending a verify chunk. It is also more
//! correct than a "does the prompt contain a doc" heuristic: an RAG answer
//! over a long document has the doc in context but generates novel text —
//! the needle misses, the router correctly stays on DFlash2.
//!
//! ## Decision table
//!
//! | hint | prompt_len | start lane | watch |
//! |---|---|---|---|
//! | `Chat` | any | DFlash2 | none (caller-declared) |
//! | `Doc` | any | Lookup | none (caller-declared) |
//! | `Auto` | ≤ `auto_chat_max_prompt` (2048) | DFlash2 | [`NeedleWatch`] → free switch to lookup when the output proves repetitive |
//! | `Auto` | > `auto_chat_max_prompt` | Lookup | [`MissMonitor`] (observability: confirms the novel regime; the p=1 self-gating floor is plain-greedy — no re-fill, honestly out of scope) |
//!
//! The length prior in `Auto` is the capture-fill budget: starting a short
//! prompt on DFlash2 costs the same fill + tap capture (~+10% fill overhead,
//! measured at chat ctx in Bench 759's `capture_fill`), while at long ctx
//! the capture overhead and the (unused-when-doc) drafter residency argue
//! for the lookup start, whose warm-up p=1 cycles are already included in
//! the measured 203.1 column.
//!
//! Modelless (no GPU, no CUDA): unit-testable in every build that enables
//! `speculative_decode`.

use super::ngram_drafter::LookupOutcome;

/// Caller-declared regime (the prod-grade primary — zero router cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskHint {
    /// Novel-text generation (chat, essays, codegen) → the DFlash2 lane.
    Chat,
    /// Copying/repetitive generation (doc-repro, quoting, regurgitation)
    /// → the lookup lane.
    Doc,
    /// Router decides from the prompt length + the runtime needle signal.
    Auto,
}

/// The drafter lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// `DFlash2GpuDrafter` + tap chunks + journal-replay advance
    /// (the Bench-759 chat-loop construction).
    DFlash2,
    /// `NgramDrafter` lookup fill + p≤16 verify chunks (the t9 / Bench-742
    /// doc-loop construction).
    Lookup,
}

impl Lane {
    /// The lane the runtime needle watch may switch INTO (the free
    /// direction). The reverse direction would need a capture re-fill and
    /// is deliberately not expressible here.
    #[inline]
    pub fn free_switch_target(self) -> Option<Self> {
        match self {
            Self::DFlash2 => Some(Self::Lookup),
            Self::Lookup => None,
        }
    }
}

/// The runtime watch installed alongside the starting lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchMode {
    /// No runtime signal consumed (explicit hints).
    None,
    /// Free per-cycle lookup-needle check on the committed tail →
    /// [`SwitchVerdict::SwitchToLookup`] when the output proves repetitive.
    NeedleWatch,
    /// Consecutive-p=1 counter → [`NoveltyVerdict::NovelConfirmed`]
    /// (observability; the self-gating floor needs no action).
    MissMonitor,
}

/// The routing decision for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutePlan {
    pub lane: Lane,
    pub watch: WatchMode,
    /// Human-readable rationale (the bench's decision log).
    pub reason: &'static str,
}

/// Tuning knobs. Defaults encode the measured regime boundaries: doc-quote
/// needles fill 15/15 (t9 signature) and novel tails fill 0–a few, so a
/// fill ≥ 12 sustained over 2 consecutive cycles is unambiguously copy
/// territory (a single deep fill can be a one-line quotation inside an
/// otherwise novel answer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegimeRouterConfig {
    /// `Auto` prompts at or below this length start on the DFlash2 lane
    /// (the capture-fill budget). Default 2048 tokens.
    pub auto_chat_max_prompt: usize,
    /// Needle fill (of K−1 = 15 lookup slots) that counts as a
    /// repetitive-output observation. Default 12.
    pub switch_fill_threshold: usize,
    /// Consecutive observations at/above the fill threshold that fire the
    /// switch. Default 2.
    pub switch_streak: usize,
    /// Consecutive p=1 full-accept cycles that confirm the novel regime on
    /// the long-prompt arm. Default 16.
    pub novel_confirm_cycles: usize,
}

impl Default for RegimeRouterConfig {
    fn default() -> Self {
        Self {
            auto_chat_max_prompt: 2048,
            switch_fill_threshold: 12,
            switch_streak: 2,
            novel_confirm_cycles: 16,
        }
    }
}

/// Route one request. Pure function of (hint, prompt_len, config).
pub fn route_plan(cfg: &RegimeRouterConfig, hint: TaskHint, prompt_len: usize) -> RoutePlan {
    match hint {
        TaskHint::Chat => RoutePlan {
            lane: Lane::DFlash2,
            watch: WatchMode::None,
            reason: "caller-declared chat → DFlash2 (capture fill)",
        },
        TaskHint::Doc => RoutePlan {
            lane: Lane::Lookup,
            watch: WatchMode::None,
            reason: "caller-declared doc/copy → lookup (graph fill, t9 construction)",
        },
        TaskHint::Auto if prompt_len <= cfg.auto_chat_max_prompt => RoutePlan {
            lane: Lane::DFlash2,
            watch: WatchMode::NeedleWatch,
            reason: "auto, short prompt → DFlash2 + needle watch (free switch to lookup)",
        },
        TaskHint::Auto => RoutePlan {
            lane: Lane::Lookup,
            watch: WatchMode::MissMonitor,
            reason: "auto, long prompt → lookup + miss monitor (capture re-fill out of scope)",
        },
    }
}

/// The needle-watch verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchVerdict {
    /// Keep the current lane.
    NoSwitch,
    /// The output has proven repetitive — switch to the lookup lane (free:
    /// the table is already warm from the watch's own maintenance).
    SwitchToLookup,
}

/// The free per-cycle regime signal on the DFlash2 lane: observe the
/// lookup-needle outcome computed over the committed context tail (the
/// same call the lookup lane would make anyway — the watch's drafter IS
/// the post-switch drafter).
#[derive(Debug, Clone)]
pub struct NeedleWatch {
    fill_threshold: usize,
    streak_needed: usize,
    streak: usize,
    fired: bool,
    /// Last observed outcome (diagnostics; the bench prints it).
    pub last: LookupOutcome,
}

impl NeedleWatch {
    pub fn new(cfg: &RegimeRouterConfig) -> Self {
        Self {
            fill_threshold: cfg.switch_fill_threshold,
            streak_needed: cfg.switch_streak.max(1),
            streak: 0,
            fired: false,
            last: LookupOutcome::MISS,
        }
    }

    /// Observe one cycle's needle outcome. Idempotent after firing.
    pub fn observe(&mut self, outcome: &LookupOutcome) -> SwitchVerdict {
        self.last = *outcome;
        if self.fired {
            return SwitchVerdict::NoSwitch;
        }
        if outcome.filled >= self.fill_threshold {
            self.streak += 1;
            if self.streak >= self.streak_needed {
                self.fired = true;
                return SwitchVerdict::SwitchToLookup;
            }
        } else {
            self.streak = 0;
        }
        SwitchVerdict::NoSwitch
    }

    /// Whether the switch already fired.
    pub fn fired(&self) -> bool {
        self.fired
    }
}

/// The novelty verdict on the long-prompt lookup arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoveltyVerdict {
    /// Below the confirmation streak.
    Undetermined,
    /// Sustained p=1 self-gating confirmed: the output is novel at long
    /// context. The lookup lane's p=1 chunk IS plain greedy (one commit per
    /// chunk at decode cost) — no action required; a host may use this to
    /// drop the per-cycle snapshot overhead and run the plain graph decode
    /// path (an ~snapshot-ms/cycle saving).
    NovelConfirmed,
}

/// Consecutive-p=1 monitor for the long-prompt lookup arm. Observing
/// `(chunk_p, committed)` per cycle; a full p=1 accept commits exactly 1.
#[derive(Debug, Clone)]
pub struct MissMonitor {
    confirm_cycles: usize,
    consecutive: usize,
    confirmed: bool,
}

impl MissMonitor {
    pub fn new(cfg: &RegimeRouterConfig) -> Self {
        Self {
            confirm_cycles: cfg.novel_confirm_cycles.max(1),
            consecutive: 0,
            confirmed: false,
        }
    }

    /// Observe one cycle: `chunk_p` = the drafted chunk width, `committed`
    /// = tokens the cycle committed.
    pub fn observe(&mut self, chunk_p: usize, committed: usize) -> NoveltyVerdict {
        if self.confirmed {
            return NoveltyVerdict::NovelConfirmed;
        }
        if chunk_p == 1 && committed == 1 {
            self.consecutive += 1;
            if self.consecutive >= self.confirm_cycles {
                self.confirmed = true;
                return NoveltyVerdict::NovelConfirmed;
            }
        } else {
            self.consecutive = 0;
        }
        NoveltyVerdict::Undetermined
    }

    /// Whether novelty was confirmed.
    pub fn confirmed(&self) -> bool {
        self.confirmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(fill: usize) -> LookupOutcome {
        LookupOutcome {
            filled: fill,
            matched_order: 8,
        }
    }

    #[test]
    fn explicit_hints_route_directly() {
        let cfg = RegimeRouterConfig::default();
        let chat = route_plan(&cfg, TaskHint::Chat, 20_000);
        assert_eq!((chat.lane, chat.watch), (Lane::DFlash2, WatchMode::None));
        let doc = route_plan(&cfg, TaskHint::Doc, 10);
        assert_eq!((doc.lane, doc.watch), (Lane::Lookup, WatchMode::None));
    }

    #[test]
    fn auto_short_starts_dflash2_with_watch() {
        let cfg = RegimeRouterConfig::default();
        let plan = route_plan(&cfg, TaskHint::Auto, 35);
        assert_eq!(plan.lane, Lane::DFlash2);
        assert_eq!(plan.watch, WatchMode::NeedleWatch);
    }

    #[test]
    fn auto_long_starts_lookup_with_monitor() {
        let cfg = RegimeRouterConfig::default();
        let plan = route_plan(&cfg, TaskHint::Auto, 8_192);
        assert_eq!(plan.lane, Lane::Lookup);
        assert_eq!(plan.watch, WatchMode::MissMonitor);
    }

    #[test]
    fn auto_boundary_is_inclusive_on_the_short_side() {
        let cfg = RegimeRouterConfig::default();
        let at = route_plan(&cfg, TaskHint::Auto, cfg.auto_chat_max_prompt);
        assert_eq!(at.lane, Lane::DFlash2);
        let past = route_plan(&cfg, TaskHint::Auto, cfg.auto_chat_max_prompt + 1);
        assert_eq!(past.lane, Lane::Lookup);
    }

    #[test]
    fn watch_fires_on_the_doc_signature_after_streak() {
        let cfg = RegimeRouterConfig::default();
        let mut w = NeedleWatch::new(&cfg);
        // t9 doc signature: needle fills 15/15 every cycle.
        assert_eq!(w.observe(&hit(15)), SwitchVerdict::NoSwitch);
        assert_eq!(w.observe(&hit(15)), SwitchVerdict::SwitchToLookup);
        assert!(w.fired());
        // idempotent
        assert_eq!(w.observe(&hit(15)), SwitchVerdict::NoSwitch);
    }

    #[test]
    fn watch_never_fires_on_the_chat_signature() {
        let cfg = RegimeRouterConfig::default();
        let mut w = NeedleWatch::new(&cfg);
        // chat/novel signature: misses and shallow fills, 100 cycles.
        let novel = [LookupOutcome::MISS, hit(1), hit(2), hit(0)];
        for k in 0..100 {
            let v = w.observe(&novel[k % novel.len()]);
            assert_eq!(v, SwitchVerdict::NoSwitch);
        }
        assert!(!w.fired());
    }

    #[test]
    fn watch_threshold_boundary() {
        let cfg = RegimeRouterConfig::default();
        let mut w = NeedleWatch::new(&cfg);
        assert_eq!(w.observe(&hit(cfg.switch_fill_threshold)), SwitchVerdict::NoSwitch);
        assert_eq!(
            w.observe(&hit(cfg.switch_fill_threshold)),
            SwitchVerdict::SwitchToLookup
        );
        let mut lo = NeedleWatch::new(&cfg);
        for _ in 0..10 {
            assert_eq!(
                lo.observe(&hit(cfg.switch_fill_threshold - 1)),
                SwitchVerdict::NoSwitch
            );
        }
        assert!(!lo.fired());
    }

    #[test]
    fn watch_streak_resets_on_an_interleaved_miss() {
        let cfg = RegimeRouterConfig::default();
        let mut w = NeedleWatch::new(&cfg);
        // RAG-answer signature: one deep quotation, then novel again.
        assert_eq!(w.observe(&hit(15)), SwitchVerdict::NoSwitch);
        assert_eq!(w.observe(&LookupOutcome::MISS), SwitchVerdict::NoSwitch);
        assert_eq!(w.observe(&hit(15)), SwitchVerdict::NoSwitch); // streak restarted
        assert_eq!(w.observe(&hit(15)), SwitchVerdict::SwitchToLookup);
    }

    #[test]
    fn monitor_confirms_sustained_p1() {
        let cfg = RegimeRouterConfig::default();
        let mut m = MissMonitor::new(&cfg);
        let n = cfg.novel_confirm_cycles;
        for _ in 0..n - 1 {
            assert_eq!(m.observe(1, 1), NoveltyVerdict::Undetermined);
        }
        assert_eq!(m.observe(1, 1), NoveltyVerdict::NovelConfirmed);
        assert!(m.confirmed());
        // sticky
        assert_eq!(m.observe(16, 15), NoveltyVerdict::NovelConfirmed);
    }

    #[test]
    fn monitor_resets_on_a_productive_cycle() {
        let cfg = RegimeRouterConfig::default();
        let mut m = MissMonitor::new(&cfg);
        for _ in 0..cfg.novel_confirm_cycles - 1 {
            assert_eq!(m.observe(1, 1), NoveltyVerdict::Undetermined);
        }
        // a quoting cycle interrupts the streak (doc warm-up ended)
        assert_eq!(m.observe(16, 15), NoveltyVerdict::Undetermined);
        for _ in 0..cfg.novel_confirm_cycles - 1 {
            assert_eq!(m.observe(1, 1), NoveltyVerdict::Undetermined);
        }
        assert!(!m.confirmed());
    }

    #[test]
    fn free_switch_direction_is_dflash2_to_lookup_only() {
        assert_eq!(Lane::DFlash2.free_switch_target(), Some(Lane::Lookup));
        assert_eq!(Lane::Lookup.free_switch_target(), None);
    }
}
