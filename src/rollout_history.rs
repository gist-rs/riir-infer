//! Issue 1122 T2 — the cross-call history n-gram drafter's OFFLINE
//! acceptance-rate estimator (the Issue-665 honest-assessment pattern:
//! measure α on a real repeated-prompt workload BEFORE any wiring).
//!
//! This is the measurement instrument only — NOT the T1 drafter arm. T1
//! ("rollout_drafter" wired through the regime_router) stays gated on a
//! claimed serving-side consumer. Everything here is pure token-math over
//! captured decode logs; no GPU, no forward, no router involvement.
//!
//! ## Semantics mirrored from the shipped family
//!
//! The lookup mirrors `speculative_decode::ngram_drafter`'s prompt-lookup
//! fill (Issue 742): at each verify cycle the draft is the continuation of
//! the LONGEST suffix of the already-committed prefix found in the history
//! streams (max lookup order 8, the family's `DEFAULT_MAX_LOOKUP_ORDER`),
//! capped at K draft tokens (K = 2 is the family's `DEFAULT_MAX_DRAFT`, the
//! posture whose breakeven bar is α ≥ 0.58 — `ngram_drafter.rs` module doc).
//! Acceptance is TRUE acceptance (the Bench-696 lesson): per-cycle matched
//! length against the TARGET's own token stream, never a corpus proxy.
//!
//! ## Determinism (the Issue-940 law, T3)
//!
//! No HashMap iteration anywhere a result depends on it: prompt keys live
//! in a `BTreeMap`, history streams are scanned in pinned FIFO order (most
//! recent LAST — the freshest stream wins ties), suffix lengths scan from
//! long to short, and the first match at each length wins. A run is
//! reproducible from the committed call log alone.

/// Draft tokens issued per verify cycle (the family's `DEFAULT_MAX_DRAFT`).
pub const DEFAULT_MAX_DRAFT: usize = 2;
/// Longest suffix length probed in the history (the family's
/// `DEFAULT_MAX_LOOKUP_ORDER`).
pub const DEFAULT_MAX_LOOKUP_ORDER: usize = 8;
/// The K=2 breakeven bar from the family's own module doc
/// (`ngram_drafter.rs`): net-positive only at α ≥ 0.58 for K=2.
pub const BREAKEVEN_ALPHA_K2: f64 = 0.58;
/// History streams retained per prompt key (bounded-recent FIFO; the
/// oldest evicted first). Sized for the probe's multi-sample workload.
pub const DEFAULT_HISTORY_CAPACITY: usize = 8;

/// BLAKE3-keyed prompt identity (16-byte prefix of the 32-byte digest —
/// collision odds at workload scale are nil, and 16 bytes print compactly).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PromptKey([u8; 16]);

impl PromptKey {
    /// BLAKE3 of the prompt string, first 16 bytes.
    #[must_use]
    pub fn of(prompt: &str) -> Self {
        let full = blake3::hash(prompt.as_bytes());
        let mut k = [0u8; 16];
        k.copy_from_slice(&full.as_bytes()[..16]);
        Self(k)
    }

    /// The hex form (stable, prints in tables).
    #[must_use]
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// One prompt's cross-call history: the model's own prior output token
/// streams for that prompt, oldest first (bounded-recent FIFO — evicting
/// the oldest keeps the table bounded and the "recent rollouts" signal
/// the paper class names).
#[derive(Clone, Debug, Default)]
pub struct PromptHistory {
    streams: Vec<Vec<u32>>,
    capacity: usize,
}

impl PromptHistory {
    /// Empty history with the given bounded capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            streams: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    /// Record one completed output stream (oldest evicted at capacity).
    /// A byte-identical repeat of the newest stream is still recorded —
    /// the greedy self-consistency control DEPENDS on counting it.
    pub fn record(&mut self, tokens: &[u32]) {
        if self.streams.len() >= self.capacity {
            self.streams.remove(0);
        }
        self.streams.push(tokens.to_vec());
    }

    /// Number of retained streams.
    #[must_use]
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Whether no stream is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// The longest-suffix continuation: given the committed prefix, find
    /// the longest suffix (≤ `max_order`) present in any retained stream,
    /// and return that stream's continuation capped at `max_draft`.
    ///
    /// Scan order is pinned: suffix length LONG→SHORT; at equal length the
    /// stream recorded MORE RECENTLY wins (FIFO scan, last-first); within
    /// a stream the first occurrence wins. Empty history → empty draft
    /// (never fabricates across prompts — the table is keyed).
    #[must_use]
    pub fn longest_suffix_continuation(
        &self,
        prefix: &[u32],
        max_order: usize,
        max_draft: usize,
    ) -> Vec<u32> {
        if self.streams.is_empty() || prefix.is_empty() {
            return Vec::new();
        }
        let lo = 1usize;
        let hi = max_order.min(prefix.len());
        let mut order = hi;
        while order >= lo {
            let needle = &prefix[prefix.len() - order..];
            // Most recent stream first (FIFO append ⇒ reverse scan).
            for stream in self.streams.iter().rev() {
                if stream.len() > order {
                    let mut start = 0usize;
                    while start + order <= stream.len() {
                        if &stream[start..start + order] == needle {
                            let end = (start + order + max_draft).min(stream.len());
                            return stream[start + order..end].to_vec();
                        }
                        start += 1;
                    }
                }
            }
            if order == 1 {
                break;
            }
            order -= 1;
        }
        Vec::new()
    }
}

/// Per-position cycle record from one measured stream.
#[derive(Clone, Copy, Debug)]
pub struct CycleRecord {
    /// Draft tokens issued this cycle (0 = miss cycle).
    pub drafted: usize,
    /// Draft tokens the target actually matched (≤ drafted).
    pub accepted: usize,
}

/// Result of measuring one target stream against one history.
#[derive(Clone, Debug, Default)]
pub struct AlphaReport {
    /// Target stream length in tokens.
    pub target_len: usize,
    /// Verify cycles walked.
    pub cycles: usize,
    /// Draft tokens issued in total.
    pub drafted_total: usize,
    /// Draft tokens the target actually accepted, in total.
    pub accepted_total: usize,
    /// TRUE per-draft-token acceptance: accepted/drafted — the family's
    /// α (the breakeven bar is stated on this quantity; a perfect
    /// drafter reads 1.0). 0.0 when nothing was drafted (cold start).
    pub alpha: f64,
    /// Tokens per forward under chain-verify economics: each cycle costs
    /// one verify forward and commits (accepted + 1 forced) tokens.
    /// The draft-side ceiling; wall-clock discounts it by verify cost.
    pub tokens_per_forward: f64,
    /// The per-cycle records (for the distribution, not just the mean).
    pub cycles_detail: Vec<CycleRecord>,
}

impl AlphaReport {
    /// Whether this report clears the family's K=2 breakeven bar.
    #[must_use]
    pub fn clears_breakeven(&self) -> bool {
        self.alpha >= BREAKEVEN_ALPHA_K2
    }
}

/// Measure TRUE acceptance of the history drafter on one target stream.
///
/// `seed` is the SHARED prompt tail both the history streams and the
/// target continue from (the deployment caches store prompt+output
/// continuations — without it the first cycle has no context to match
/// and the cold-start α=0 becomes a construction artifact). The target
/// is measured output-only; matching always sees `seed + generated`.
///
/// Walks the target in verify cycles: draft = the history's
/// longest-suffix continuation of the committed prefix (capped at
/// `max_draft`); accepted = common-prefix length of draft vs the target's
/// remaining tokens; the cycle then commits (accepted + 1) tokens (the +1
/// is the forced correction/next token every chain-verify cycle pays).
/// An empty history drafts nothing — every cycle is a miss — which is the
/// cold-start honest baseline, reported separately by the caller.
pub fn measure_alpha_seeded(
    seed: &[u32],
    history: &PromptHistory,
    target: &[u32],
    max_draft: usize,
    max_order: usize,
) -> AlphaReport {
    if target.is_empty() {
        return AlphaReport {
            target_len: 0,
            ..AlphaReport::default()
        };
    }
    // The seed participates in matching on BOTH sides — the deployment
    // caches store prompt+output continuations, so the history streams
    // match as `seed + output` and the committed prefix is `seed +
    // generated`. The target is still measured output-only.
    let mut view = PromptHistory::new(history.capacity);
    for s in &history.streams {
        let mut full = seed.to_vec();
        full.extend_from_slice(s);
        view.record(&full);
    }
    let mut combined = seed.to_vec();
    combined.extend_from_slice(target);
    let seed_len = seed.len();
    let mut report = AlphaReport {
        target_len: target.len(),
        ..AlphaReport::default()
    };
    let mut pos = 0usize;
    while pos < target.len() {
        let committed = seed_len + pos;
        let draft = view.longest_suffix_continuation(&combined[..committed], max_order, max_draft);
        let accepted = draft
            .iter()
            .zip(target[pos..].iter())
            .take_while(|(d, t)| d == t)
            .count();
        report.cycles_detail.push(CycleRecord {
            drafted: draft.len(),
            accepted,
        });
        report.cycles += 1;
        pos += accepted + 1;
    }
    let drafted_total: usize = report.cycles_detail.iter().map(|c| c.drafted).sum();
    let accepted_total: usize = report.cycles_detail.iter().map(|c| c.accepted).sum();
    report.drafted_total = drafted_total;
    report.accepted_total = accepted_total;
    report.alpha = if drafted_total == 0 {
        0.0
    } else {
        f64::from(accepted_total as u32) / f64::from(drafted_total as u32)
    };
    report.tokens_per_forward =
        f64::from(target.len() as u32) / f64::from(report.cycles.max(1) as u32);
    report
}

/// [`measure_alpha_seeded`] with an empty seed — the no-prompt-context
/// posture (back-compat; the probe passes the prompt tail).
pub fn measure_alpha(
    history: &PromptHistory,
    target: &[u32],
    max_draft: usize,
    max_order: usize,
) -> AlphaReport {
    measure_alpha_seeded(&[], history, target, max_draft, max_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identical prior stream in history + the shared seed → every token
    /// drafts + accepts, α = 1.0 (the greedy self-consistency control's
    /// expected read). Without the seed the first cycle provably cannot
    /// match — that construction artifact is why the seeded form exists.
    #[test]
    fn identical_history_accepts_everything() {
        let mut h = PromptHistory::new(4);
        let t = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        h.record(&t);
        let seed = vec![90u32, 91, 92];
        let r = measure_alpha_seeded(&seed, &h, &t, 2, 8);
        assert!((r.alpha - 1.0).abs() < 1e-12, "alpha {}", r.alpha);
        assert!(r.clears_breakeven());
    }

    /// Unseeded identical history: cycle 0 drafts nothing (no prefix to
    /// match — the construction artifact the seed form exists to remove);
    /// every later cycle drafts 2 and accepts 2, so per-draft-token α is
    /// still 1.0, but tokens_per_forward pays for the wasted cycle.
    #[test]
    fn unseeded_identical_history_misses_only_cycle_zero() {
        let mut h = PromptHistory::new(4);
        let t = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        h.record(&t);
        let r = measure_alpha(&h, &t, 2, 8);
        assert_eq!(r.cycles_detail[0].drafted, 0);
        assert!(
            r.cycles_detail[1..]
                .iter()
                .all(|c| c.accepted == c.drafted && c.accepted > 0)
        );
        assert!((r.alpha - 1.0).abs() < 1e-12, "alpha {}", r.alpha);
        let seeded = measure_alpha_seeded(&[90u32, 91, 92], &h, &t, 2, 8);
        assert!(
            seeded.tokens_per_forward > r.tokens_per_forward,
            "seeded tpf {} must beat unseeded {} — the prompt context is draftable",
            seeded.tokens_per_forward,
            r.tokens_per_forward
        );
    }

    /// Empty history → all miss cycles, α = 0, never fabricates.
    #[test]
    fn empty_history_is_all_misses() {
        let h = PromptHistory::new(4);
        let r = measure_alpha(&h, &[9u32, 8, 7, 6], 2, 8);
        assert_eq!(r.alpha, 0.0);
        assert_eq!(r.cycles, 4);
        assert!(r.cycles_detail.iter().all(|c| c.drafted == 0));
    }

    /// Keyed isolation: a DIFFERENT prompt's history never drafts — the
    /// never-fabricates-across-prompts law is the table's keying itself,
    /// so the caller holding the wrong key simply sees an empty history.
    #[test]
    fn prompt_keys_are_distinct() {
        let a = PromptKey::of("fix the clippy warning in foo.rs");
        let b = PromptKey::of("fix the clippy warning in bar.rs");
        assert_ne!(a, b);
        assert_ne!(a.hex(), b.hex());
    }

    /// Longest-suffix preference: a 3-gram suffix match must beat a
    /// 1-gram match when both exist (order scan is LONG→SHORT).
    #[test]
    fn longest_suffix_wins_over_short() {
        let mut h = PromptHistory::new(4);
        // Contains 1-gram continuation after [7] AND a longer [5,6,7]→[8,9].
        h.record(&[5u32, 6, 7, 8, 9]);
        let prefix = vec![1u32, 2, 3, 5, 6, 7];
        let d = h.longest_suffix_continuation(&prefix, 8, 2);
        assert_eq!(d, vec![8u32, 9]);
    }

    /// The recency tie rule: two streams match the same suffix length with
    /// DIFFERENT continuations — the more recently recorded stream wins.
    #[test]
    fn most_recent_stream_wins_ties() {
        let mut h = PromptHistory::new(4);
        h.record(&[1u32, 2, 3, 40]); // older: [3] → 40
        h.record(&[1u32, 2, 3, 50]); // newer: [3] → 50
        let prefix = vec![9u32, 1, 2, 3];
        let d = h.longest_suffix_continuation(&prefix, 8, 1);
        assert_eq!(d, vec![50u32]);
    }

    /// FIFO eviction at capacity: recording past capacity drops the OLDEST
    /// (its continuation becomes unavailable to the matcher).
    #[test]
    fn fifo_eviction_drops_oldest() {
        let mut h = PromptHistory::new(2);
        h.record(&[1u32, 10]);
        h.record(&[2u32, 20]);
        h.record(&[3u32, 30]);
        assert_eq!(h.len(), 2);
        // [1,10] evicted: the order-2 suffix [9,1] has NO match anywhere —
        // the eviction is observable as a miss.
        let d = h.longest_suffix_continuation(&[9u32, 1], 8, 1);
        assert!(d.is_empty());
        // The retained middle stream still drafts: suffix [8,2] → [20].
        let d = h.longest_suffix_continuation(&[8u32, 2], 8, 1);
        assert_eq!(d, vec![20u32]);
    }

    /// tokens_per_forward: with α=1 the whole stream commits in ~1 cycle
    /// per K+1 tokens (the chain-verify economics the report promises).
    #[test]
    fn tokens_per_forward_reflects_chain_economics() {
        let mut h = PromptHistory::new(4);
        let t = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        h.record(&t);
        let r = measure_alpha(&h, &t, 2, 8);
        // K=2: each cycle commits ≤ 3 tokens → cycles ≥ ceil(10/3).
        assert!(r.tokens_per_forward > 2.0, "tpf {}", r.tokens_per_forward);
    }

    /// Determinism: two identical runs produce byte-identical reports.
    #[test]
    fn repeated_measurement_is_identical() {
        let mut h = PromptHistory::new(4);
        h.record(&[7u32, 8, 9, 1, 2, 3, 4]);
        h.record(&[7u32, 8, 9, 5, 6, 1, 2]);
        let t = vec![7u32, 8, 9, 5, 6, 1, 2, 3, 4];
        let a = measure_alpha(&h, &t, 2, 8);
        let b = measure_alpha(&h, &t, 2, 8);
        assert_eq!(a.alpha, b.alpha);
        assert_eq!(a.cycles, b.cycles);
        assert_eq!(a.cycles_detail.len(), b.cycles_detail.len());
    }
}
