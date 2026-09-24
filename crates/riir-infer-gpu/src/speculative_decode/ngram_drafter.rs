//! Issue 665 T2 — N-gram draft model for speculative decoding.
//!
//! A modelless CPU-only draft model that predicts the next K tokens from a
//! frequency table built from the generated context. Zero training dependency,
//! zero GPU cost — runs in microseconds on the CPU.
//!
//! ## How it works
//!
//! The drafter maintains an n-gram frequency table (default: trigram, n=3).
//! Given the last (n-1) tokens of the context, it looks up the most likely
//! continuation token. For K>1 lookahead, it chains predictions: after
//! predicting token i, it appends token i to the context suffix and predicts
//! token i+1.
//!
//! ## Acceptance rate expectations
//!
//! Per Issue 665 T1 analysis:
//! - **Repetitive text (code, JSON, structured output):** 70-90% per-token
//!   acceptance → 1.3-1.6× speedup at K=2
//! - **Creative/chatty text:** 30-50% per-token acceptance → net-neutral or
//!   net-negative (breakeven at α ≥ 0.58 for K=2)
//!
//! The drafter is cheap to build (~100 LOC) and the investigation cost is low,
//! so it's worth trying on real workloads to measure the actual acceptance rate.
//!
//! ## Design notes
//!
//! - **Greedy prediction** (most frequent continuation): the simplest approach.
//!   Stochastic sampling from the frequency distribution would improve diversity
//!   but at the cost of lower acceptance rate (greedy maximizes argmax-match
//!   probability). Since the main model uses argmax for verification, greedy
//!   draft is optimal for acceptance rate.
//!
//! - **Fallback to a uniform guess** when no n-gram match exists: this ensures
//!   the drafter always produces a prediction (even if likely wrong), so the
//!   speculative verify path always runs K forwards. The caller can check
//!   `confidence()` to decide whether to speculate at all.
//!
//! - **BLAKE3-deterministic fallback:** when no n-gram match exists, instead of
//!   a random guess, use a BLAKE3 hash of the context suffix to deterministically
//!   pick a token. This ensures reproducibility (same context → same draft) and
//!   avoids the cost of a thread-local RNG on the hot path.
//!
//! ## Lookup mode (Issue 742 T2 — prompt-lookup-decoding fill)
//!
//! [`NgramDrafter::fill_lookup_draft`] implements the verbatim-span drafter the
//! DFlash2-record recipe calls "lookup-augmented drafting": find the most
//! recent occurrence of the longest (≤ `max_lookup_order`, default 8) suffix of
//! the context INSIDE the context itself, then copy the tokens that FOLLOW that
//! occurrence — up to K=16 slots per cycle. On document-reproduction tasks the
//! next 16 tokens are frequently a verbatim copy of an earlier passage, so a
//! single span match fills the whole draft block (the external record's 15/16
//! acceptance class).
//!
//! Contract (deliberate differences from the chained predictor):
//! - **Never fabricates.** On no match at any order it fills 0 tokens — unlike
//!   `predict`, which always returns K guesses. A 0-fill tells the caller to
//!   fall back (chained predict, a trained head, or plain decode).
//! - **In-vocab by construction.** Copied tokens come from the context itself;
//!   a context produced by the tokenizer/model is in-vocab, so copies are too.
//!   The fabricated fallback inside `predict` is bounded by `vocab_size` when
//!   set (the drafter-side half of the cudarc seam's in-vocab assert — the
//!   Issue 717 `CUDA_ERROR_ILLEGAL_ADDRESS` class is impossible by
//!   construction once the caller sets the vocab).
//! - **Zero-allocation.** The hot-path form fills a caller-owned buffer.
//! - Highest matching order wins (a longer needle is a more specific copy
//!   anchor); among same-order matches, the most recent wins (locality: the
//!   active copy source is usually the latest occurrence).

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

/// Default n-gram order (trigram: 2 context tokens → 1 predicted token).
pub const DEFAULT_NGRAM_ORDER: usize = 3;

/// Default maximum draft lookahead (K=2 is the sweet spot per Issue 665 T1).
pub const DEFAULT_MAX_DRAFT: usize = 2;

/// Default maximum needle length for lookup fill (Issue 742 T2). The external
/// record's lookup drafting uses an 8-token context match to fill 16 draft
/// slots; longer needles are more specific copy anchors (fewer false matches)
/// but need longer repeated passages to fire.
pub const DEFAULT_MAX_LOOKUP_ORDER: usize = 8;

/// Result of a lookup fill: how many tokens were written and at which needle
/// order the span match was found. `filled == 0` means no match at any order
/// (a miss — the caller must not treat the output buffer as valid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupOutcome {
    /// Number of tokens written into the caller's buffer (0 on miss).
    pub filled: usize,
    /// Needle length (n-gram order) that produced the match; 0 on miss.
    pub matched_order: usize,
}

impl LookupOutcome {
    /// A miss: nothing filled, no order matched.
    pub const MISS: Self = Self {
        filled: 0,
        matched_order: 0,
    };

    /// Whether any token was filled (a hit).
    #[inline]
    pub fn is_hit(&self) -> bool {
        self.filled > 0
    }
}

/// An n-gram draft model for speculative decoding.
///
/// Builds a frequency table from observed token sequences and predicts the most
/// likely next token given a context suffix. Thread-safe via `&self` reads
/// (the table is built once per generation, then queried read-only).
pub struct NgramDrafter {
    /// N-gram order (n=3 means trigram: last 2 tokens → next token).
    order: usize,
    /// Frequency table: (n-1)-token suffix → (next_token → count).
    /// Keyed on a hashed representation of the suffix for O(1) lookup.
    table: HashMap<u64, HashMap<u32, u32>>,
    /// BLAKE3 key material for deterministic fallback (when no n-gram match).
    /// Using a fixed seed for reproducibility.
    fallback_seed: [u8; 32],
    /// Maximum needle length for [`Self::fill_lookup_draft`] (Issue 742 T2).
    max_lookup_order: usize,
    /// Vocabulary bound for FABRICATED fallback tokens (Issue 742 T2 — the
    /// drafter-side in-vocab guarantee). `None` keeps the legacy unbounded
    /// fallback (back-compat for callers that never hit fabrication, e.g.
    /// Bench 665); `Some(v)` reduces every fabricated token mod v so a
    /// no-match draft can never index past the embedding table (the Issue 717
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` class). Copied lookup tokens are NOT
    /// bounded — they come from the context and are in-vocab by construction.
    vocab_size: Option<u32>,
}

impl NgramDrafter {
    /// Create a new drafter with the given n-gram order.
    ///
    /// `order` must be ≥ 2 (bigram minimum). For natural language, trigram
    /// (order=3) is a good default. Higher orders capture longer-range patterns
    /// but need more data to fill the table.
    pub fn new(order: usize) -> Self {
        assert!(order >= 2, "n-gram order must be >= 2 (bigram minimum)");
        Self {
            order,
            table: HashMap::new(),
            max_lookup_order: DEFAULT_MAX_LOOKUP_ORDER,
            vocab_size: None,
            // 32-byte seed for BLAKE3-style deterministic fallback.
            fallback_seed: *b"ngram_drafter_v1_seed_0000000000",
        }
    }

    /// Create a drafter with the default trigram order.
    pub fn default_trigram() -> Self {
        Self::new(DEFAULT_NGRAM_ORDER)
    }

    /// Create a drafter whose FABRICATED fallback tokens are bounded by
    /// `vocab_size` (Issue 742 T2 — the drafter-side in-vocab guarantee).
    ///
    /// Once set, every token the drafter can emit — fabricated fallback OR
    /// verbatim lookup copy — is either a copy of a context token (in-vocab
    /// by construction) or reduced mod `vocab_size`. The cudarc verify seam's
    /// `tok < vocab_size` assert becomes unreachable rather than load-bearing.
    pub fn with_vocab_size(order: usize, vocab_size: u32) -> Self {
        assert!(vocab_size > 0, "vocab_size must be > 0");
        let mut d = Self::new(order);
        d.vocab_size = Some(vocab_size);
        d
    }

    /// Bound fabricated fallback tokens by `vocab_size` (see
    /// [`Self::with_vocab_size`]). Call this once the model config is known.
    pub fn set_vocab_size(&mut self, vocab_size: u32) {
        assert!(vocab_size > 0, "vocab_size must be > 0");
        self.vocab_size = Some(vocab_size);
    }

    /// Set the maximum needle length for [`Self::fill_lookup_draft`].
    /// The Issue 742 T0(c) order sweep (3/5/8) varies this to find the
    /// doc-repro sweet spot; 8 is the external record's lookup depth.
    pub fn set_max_lookup_order(&mut self, max_order: usize) {
        self.max_lookup_order = max_order.max(2);
    }

    /// The configured maximum lookup needle length.
    pub fn max_lookup_order(&self) -> usize {
        self.max_lookup_order
    }

    /// Build the frequency table from a token sequence (the prompt + generated
    /// text so far). This is called once at the start of generation, and
    /// incrementally updated as new tokens are committed.
    ///
    /// The `tokens` slice is the full context (prompt + generated). The drafter
    /// extracts all n-grams from it.
    pub fn build_from_context(&mut self, tokens: &[u32]) {
        self.table.clear();
        self.add_context(tokens);
    }

    /// Incrementally add new tokens to the frequency table. Called after each
    /// committed token (or batch of tokens) to keep the table up-to-date.
    ///
    /// `new_tokens` should include enough preceding context to form complete
    /// n-grams with the new tokens. Typically, pass the last (order-1) tokens
    /// of the previous context plus the new tokens.
    pub fn add_context(&mut self, tokens: &[u32]) {
        if tokens.len() < self.order {
            return;
        }
        let suffix_len = self.order - 1;
        for i in 0..tokens.len() - suffix_len {
            let suffix = &tokens[i..i + suffix_len];
            let next = tokens[i + suffix_len];
            let key = hash_suffix(suffix, &self.fallback_seed);
            *self
                .table
                .entry(key)
                .or_default()
                .entry(next)
                .or_insert(0) += 1;
        }
    }

    /// Predict the next K tokens given the current context suffix.
    ///
    /// Returns a vector of K predicted tokens (greedy: most frequent continuation
    /// at each step). If no n-gram match exists at any step, the remaining
    /// predictions use the deterministic BLAKE3 fallback.
    ///
    /// The caller should check `confidence()` to decide whether the predictions
    /// are reliable enough to warrant speculation.
    pub fn predict(&self, context: &[u32], k: usize) -> Vec<u32> {
        let suffix_len = self.order - 1;
        if context.len() < suffix_len {
            // Not enough context for even one prediction — return fallback guesses.
            return (0..k)
                .map(|step| {
                    self.bound_fabricated(deterministic_fallback(
                        &context
                            .iter()
                            .copied()
                            .chain(std::iter::once(step as u32))
                            .collect::<Vec<_>>(),
                        &self.fallback_seed,
                    ))
                })
                .collect();
        }

        let mut predictions = Vec::with_capacity(k);
        let mut suffix: Vec<u32> = context[context.len() - suffix_len..].to_vec();

        for _step in 0..k {
            let key = hash_suffix(&suffix, &self.fallback_seed);
            if let Some(contins) = self.table.get(&key) {
                // Greedy: pick the most frequent continuation.
                let best = contins
                    .iter()
                    .max_by_key(|&(_, &count)| count).map_or_else(|| {
                        self.bound_fabricated(deterministic_fallback(
                            &suffix,
                            &self.fallback_seed,
                        ))
                    }, |(&tok, _)| tok);
                predictions.push(best);
                // Advance the suffix: drop the first element, append the prediction.
                if suffix_len > 0 {
                    suffix.remove(0);
                    suffix.push(best);
                }
            } else {
                // No n-gram match — use deterministic fallback for this + remaining steps.
                let fb = self.bound_fabricated(deterministic_fallback(
                    &suffix,
                    &self.fallback_seed,
                ));
                predictions.push(fb);
                if suffix_len > 0 {
                    suffix.remove(0);
                    suffix.push(fb);
                }
            }
        }

        predictions
    }

    /// Confidence that the next prediction will be correct (empirical
    /// acceptance rate estimate). Returns the probability of the most frequent
    /// continuation given the current suffix, or 0.0 if no match exists.
    ///
    /// The caller can use this to decide whether to speculate (e.g., only
    /// speculate if confidence > 0.5).
    pub fn confidence(&self, context: &[u32]) -> f32 {
        let suffix_len = self.order - 1;
        if context.len() < suffix_len {
            return 0.0;
        }
        let suffix = &context[context.len() - suffix_len..];
        let key = hash_suffix(suffix, &self.fallback_seed);
        match self.table.get(&key) {
            Some(contins) => {
                let total: u32 = contins.values().sum();
                if total == 0 {
                    return 0.0;
                }
                let max_count = contins.values().copied().max().unwrap_or(0);
                max_count as f32 / total as f32
            }
            None => 0.0,
        }
    }

    /// Number of distinct n-grams in the table (for diagnostics).
    pub fn ngram_count(&self) -> usize {
        self.table.values().map(|m| m.len()).sum()
    }

    /// The n-gram order this drafter was configured with.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Reduce a FABRICATED fallback token into vocab range. No-op when no
    /// vocab bound is configured (legacy behavior). Greedy table predictions
    /// are not bounded — they are copies of observed context tokens.
    #[inline]
    fn bound_fabricated(&self, token: u32) -> u32 {
        match self.vocab_size {
            Some(v) => token % v,
            None => token,
        }
    }

    /// Verbatim lookup fill — the Issue 742 T2 "lookup-augmented drafting"
    /// primitive (prompt-lookup-decoding shape).
    ///
    /// Searches for the most recent occurrence of the longest available
    /// suffix of `context` (needle length 2..=`max_lookup_order`, highest
    /// first) INSIDE `context` itself — excluding the suffix's own trailing
    /// position — then copies the tokens that FOLLOW that occurrence into
    /// `out`, up to `out.len()` tokens.
    ///
    /// Contract:
    /// - **Never fabricates**: on no match at any order, returns
    ///   [`LookupOutcome::MISS`] and leaves `out` untouched. The caller falls
    ///   back (chained [`Self::predict`], a trained head, or plain decode).
    /// - **In-vocab by construction**: every filled token is a copy of a
    ///   context token.
    /// - **Zero-allocation**: fills the caller's buffer; no heap traffic.
    /// - A hit may fill FEWER than `out.len()` tokens when the matched
    ///   occurrence sits near the end of the context (the copy span is capped
    ///   by what follows it). Fewer-than-K fills are still useful — commit a
    ///   short block or pad via the chained predictor.
    ///
    /// Cost: worst case `O(max_lookup_order × context.len())` token compares
    /// (a full backward scan per order level with an early first-token
    /// rejection); at 25K context this is microseconds on CPU — negligible
    /// next to a GPU verify cycle.
    pub fn fill_lookup_draft(&self, context: &[u32], out: &mut [u32]) -> LookupOutcome {
        if out.is_empty() || context.len() < 3 {
            // Need at least a 2-token needle plus 1 token after a match.
            return LookupOutcome::MISS;
        }
        let max_order = self.max_lookup_order.min(context.len() - 1);
        for order in (2..=max_order).rev() {
            let needle = &context[context.len() - order..];
            // Backward scan over candidate start positions p in
            // 0..=context.len()-order-1: p + order < context.len() guarantees
            // at least one copyable token after the match, and p <
            // context.len() - order excludes the needle's own trailing
            // position (a self-match copies nothing new).
            let mut p = context.len() - order;
            while p > 0 {
                p -= 1;
                if context[p] == needle[0] && &context[p..p + order] == needle {
                    let src = p + order;
                    let n = out.len().min(context.len() - src);
                    out[..n].copy_from_slice(&context[src..src + n]);
                    return LookupOutcome {
                        filled: n,
                        matched_order: order,
                    };
                }
            }
        }
        LookupOutcome::MISS
    }

    /// Allocating convenience wrapper over [`Self::fill_lookup_draft`] for
    /// harnesses and tests — the verify-cycle hot path should prefer the
    /// buffer-filling form.
    pub fn predict_lookup(&self, context: &[u32], k: usize) -> (Vec<u32>, LookupOutcome) {
        let mut out = vec![0u32; k];
        let outcome = self.fill_lookup_draft(context, &mut out);
        out.truncate(outcome.filled);
        (out, outcome)
    }
}

/// Hash a token suffix into a u64 key for the frequency table.
///
/// Uses a simple polynomial rolling hash (FNV-1a variant) seeded with the
/// fallback seed. This is NOT cryptographically secure — it's just a hash map
/// key. Collisions are rare for natural token distributions and don't affect
/// correctness (they just merge two different suffixes' counts, which slightly
/// degrades prediction quality).
#[inline]
fn hash_suffix(suffix: &[u32], seed: &[u8; 32]) -> u64 {
    // FNV-1a offset basis (mixed with seed byte 0 for per-instance uniqueness).
    let mut hash: u64 = 0xcbf29ce484222325 ^ (seed[0] as u64);
    for &tok in suffix {
        // FNV-1a prime.
        hash ^= tok as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Deterministic fallback when no n-gram match exists.
///
/// Uses a simple hash of the suffix to pick a token. This ensures
/// reproducibility (same context → same draft) without a thread-local RNG.
/// The prediction is almost certainly wrong, but the speculative verify path
/// handles wrong predictions gracefully (it just rejects and resamples).
///
/// The raw return value is an UNBOUNDED u32 — callers reduce it into vocab
/// range via [`NgramDrafter::with_vocab_size`] (the drafter's internal
/// `bound_fabricated` applies it on every fabrication path). Without a bound
/// configured, a fabricated token can index past the embedding table — the
/// Issue 717 `CUDA_ERROR_ILLEGAL_ADDRESS` class. Prefer lookup fill
/// ([`NgramDrafter::fill_lookup_draft`]), which never fabricates.
#[inline]
fn deterministic_fallback(suffix: &[u32], seed: &[u8; 32]) -> u32 {
    let h = hash_suffix(suffix, seed);
    // Mix in a second seed byte for better distribution.
    (h ^ (seed[1] as u64).wrapping_mul(0x9e3779b97f4a7c15)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_predict_basic() {
        let mut drafter = NgramDrafter::default_trigram();
        // Context: [1, 2, 3, 1, 2, 4, 1, 2, 3]
        // Trigrams: (1,2)→3, (2,3)→1, (3,1)→2, (1,2)→4, (2,4)→1, (4,1)→2, (1,2)→3
        // (1,2) appears 3 times: →3 twice, →4 once → greedy predicts 3
        let context = vec![1u32, 2, 3, 1, 2, 4, 1, 2, 3];
        drafter.build_from_context(&context);

        // Predict after suffix [1, 2] → should predict 3 (most frequent)
        let preds = drafter.predict(&[1, 2], 1);
        assert_eq!(preds, vec![3]);
    }

    #[test]
    fn test_predict_k2_chains() {
        let mut drafter = NgramDrafter::default_trigram();
        // Context where (1,2)→3 and (2,3)→4 and (3,4)→5
        let context = vec![1u32, 2, 3, 4, 5, 1, 2, 3, 4, 5];
        drafter.build_from_context(&context);

        // Predict 2 tokens after suffix [1, 2] → [3, 4]
        let preds = drafter.predict(&[1, 2], 2);
        assert_eq!(preds, vec![3, 4]);
    }

    #[test]
    fn test_confidence_reflects_frequency() {
        let mut drafter = NgramDrafter::default_trigram();
        // (1,2) → 3 (twice), 4 (once). Confidence = 2/3.
        drafter.build_from_context(&[1u32, 2, 3, 1, 2, 3, 1, 2, 4]);
        let conf = drafter.confidence(&[1, 2]);
        assert!((conf - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_match_returns_fallback() {
        let drafter = NgramDrafter::default_trigram();
        // Empty table → confidence 0, predict returns deterministic fallback
        let conf = drafter.confidence(&[1, 2]);
        assert_eq!(conf, 0.0);

        let preds = drafter.predict(&[1, 2], 2);
        assert_eq!(preds.len(), 2);
        // Fallback is deterministic — same input → same output
        let preds2 = drafter.predict(&[1, 2], 2);
        assert_eq!(preds, preds2);
    }

    #[test]
    fn test_incremental_add() {
        let mut drafter = NgramDrafter::default_trigram();
        // Build initial table
        drafter.build_from_context(&[1u32, 2, 3, 1, 2, 3]);
        assert_eq!(drafter.predict(&[1, 2], 1), vec![3]);

        // Add new context that changes the prediction
        drafter.add_context(&[2, 3, 1, 2, 4, 1, 2, 4, 1, 2, 4]);
        // Now (1,2) → 3 (twice from before, via 1,2,3,1,2,3)
        //             → 4 (three times from the new context: 1,2,4 ×3)
        // Greedy should now predict 4
        assert_eq!(drafter.predict(&[1, 2], 1), vec![4]);
    }

    #[test]
    fn test_bigram_minimum_order() {
        let mut drafter = NgramDrafter::new(2);
        // Bigram: (1)→2, (2)→3
        drafter.build_from_context(&[1u32, 2, 3]);
        assert_eq!(drafter.predict(&[1], 1), vec![2]);
        assert_eq!(drafter.predict(&[2], 1), vec![3]);
    }

    #[test]
    fn test_order_too_low_panics() {
        let result = std::panic::catch_unwind(|| NgramDrafter::new(1));
        assert!(result.is_err(), "order < 2 should panic");
    }

    #[test]
    fn test_short_context_uses_fallback() {
        let mut drafter = NgramDrafter::default_trigram();
        drafter.build_from_context(&[1u32, 2, 3]);
        // Only 1 token in context — can't form a 2-token suffix
        let preds = drafter.predict(&[5], 2);
        assert_eq!(preds.len(), 2);
        // Deterministic
        let preds2 = drafter.predict(&[5], 2);
        assert_eq!(preds, preds2);
    }

    #[test]
    fn test_determinism_same_context_same_prediction() {
        let mut drafter = NgramDrafter::default_trigram();
        drafter.build_from_context(&[1u32, 2, 3, 4, 5, 1, 2, 3, 4, 5]);
        let p1 = drafter.predict(&[3, 4], 3);
        let p2 = drafter.predict(&[3, 4], 3);
        assert_eq!(p1, p2, "same context must produce same prediction");
    }

    #[test]
    fn test_ngram_count() {
        let mut drafter = NgramDrafter::default_trigram();
        drafter.build_from_context(&[1u32, 2, 3, 1, 2, 4]);
        // Trigrams: (1,2)→3, (2,3)→1, (3,1)→2, (1,2)→4
        // Distinct (suffix, next) pairs: 4 (note (1,2)→3 and (1,2)→4 are distinct)
        assert_eq!(drafter.ngram_count(), 4);
    }

    // ── Issue 742 T2: lookup fill (prompt-lookup-decoding shape) ──

    #[test]
    fn lookup_fill_copies_verbatim_span_k16() {
        // A 40-token passage repeated verbatim — the doc-repro shape. The
        // last-8 needle of the second copy matches inside the first copy;
        // the fill is the 16 tokens that follow that occurrence.
        let passage: Vec<u32> = (100..140).collect();
        let mut ctx = passage.clone();
        ctx.extend_from_slice(&passage);
        let drafter = NgramDrafter::default_trigram();

        let mut out = [0u32; 16];
        let outcome = drafter.fill_lookup_draft(&ctx, &mut out);
        assert!(outcome.is_hit(), "repeated passage must hit");
        assert_eq!(outcome.matched_order, 8, "highest available order (8) wins");
        assert_eq!(outcome.filled, 16, "long repeat fills all K slots");
        assert_eq!(&out[..16], &passage[0..16], "fill is the verbatim continuation");
    }

    #[test]
    fn lookup_prefers_highest_order_match() {
        // last-5 needle matches an early occurrence (continuation 100,101,...)
        // while the last-3 needle ALSO matches a more recent, DIFFERENT
        // occurrence (continuation 200,201,...). Highest order must win.
        let ctx: Vec<u32> = vec![
            10, 11, 12, 13, 14, // idx 0-4: the order-5 anchor
            100, 101, 102, // idx 5-7: its continuation
            12, 13, 14, // idx 8-10: a later order-3 anchor
            200, 201, // idx 11-12: ITS continuation
            10, 11, 12, 13, 14, // idx 13-17: the needle (context tail)
        ];
        let drafter = NgramDrafter::default_trigram();
        let mut out = [0u32; 16];
        let outcome = drafter.fill_lookup_draft(&ctx, &mut out);
        assert_eq!(outcome.matched_order, 5);
        assert_eq!(outcome.filled, 13); // everything after idx 5 up to ctx end
        let expected: Vec<u32> = ctx[5..].to_vec();
        assert_eq!(&out[..outcome.filled], expected.as_slice());
        // Sanity: the order-3 match would have produced a DIFFERENT fill.
        let mut d3 = NgramDrafter::default_trigram();
        d3.set_max_lookup_order(3);
        let mut out3 = [0u32; 16];
        let o3 = d3.fill_lookup_draft(&ctx, &mut out3);
        assert_eq!(o3.matched_order, 3);
        assert_ne!(&out[..o3.filled], &out3[..o3.filled]);
    }

    #[test]
    fn lookup_falls_back_to_lower_order_most_recent() {
        // No 5-gram or 4-gram match; the 3-gram [3,4,5] occurs twice — the
        // MOST RECENT occurrence's continuation is taken.
        let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 99, 1, 2, 3, 4, 5, 99, 3, 4, 5];
        let drafter = NgramDrafter::default_trigram();
        let mut out = [0u32; 16];
        let outcome = drafter.fill_lookup_draft(&ctx, &mut out);
        assert_eq!(outcome.matched_order, 3);
        assert_eq!(outcome.filled, 4); // [99, 3, 4, 5] — context ends after 4
        assert_eq!(&out[..4], &[99, 3, 4, 5]);
    }

    #[test]
    fn lookup_miss_never_fabricates() {
        // All-distinct context: no suffix occurs earlier. MISS must leave the
        // caller's buffer untouched (never fabricate) and report order 0.
        let ctx: Vec<u32> = (0..50u32).collect();
        let drafter = NgramDrafter::default_trigram();
        let mut out = [u32::MAX; 16];
        let outcome = drafter.fill_lookup_draft(&ctx, &mut out);
        assert_eq!(outcome, LookupOutcome::MISS);
        assert!(!outcome.is_hit());
        assert!(out.iter().all(|&t| t == u32::MAX), "miss must not touch the buffer");
    }

    #[test]
    fn lookup_fill_capped_by_context_end() {
        // The matched occurrence sits near the end of the context — the copy
        // span is capped by what follows it, not by out.len().
        let ctx: Vec<u32> = vec![1, 2, 50, 1, 2];
        let drafter = NgramDrafter::default_trigram();
        let mut out = [0u32; 16];
        let outcome = drafter.fill_lookup_draft(&ctx, &mut out);
        assert_eq!(outcome.matched_order, 2);
        assert_eq!(outcome.filled, 3);
        assert_eq!(&out[..3], &[50, 1, 2]);
    }

    #[test]
    fn lookup_short_context_and_empty_out_are_miss() {
        let drafter = NgramDrafter::default_trigram();
        // Context shorter than needle(2)+1 → miss.
        let mut out = [0u32; 16];
        assert_eq!(drafter.fill_lookup_draft(&[1, 2], &mut out), LookupOutcome::MISS);
        // Empty output buffer → miss.
        assert_eq!(drafter.fill_lookup_draft(&[1, 2, 3], &mut []), LookupOutcome::MISS);
    }

    #[test]
    fn lookup_is_deterministic() {
        let passage: Vec<u32> = (0..40u32).map(|i| i % 7).collect();
        let mut ctx = passage.clone();
        ctx.extend_from_slice(&passage);
        let drafter = NgramDrafter::default_trigram();
        let (a, oa) = drafter.predict_lookup(&ctx, 16);
        let (b, ob) = drafter.predict_lookup(&ctx, 16);
        assert_eq!((a, oa), (b, ob), "same context must produce same lookup draft");
    }

    #[test]
    fn predict_fallback_bounded_when_vocab_set() {
        // The Issue 717 hazard class: fabricated fallback tokens were
        // UNBOUNDED u32s (embedding OOB → CUDA_ERROR_ILLEGAL_ADDRESS).
        // With a vocab bound set, every fabricated token is < vocab.
        let mut drafter = NgramDrafter::with_vocab_size(3, 100);
        // Empty table → every prediction is a fabricated fallback.
        drafter.build_from_context(&[]);
        for k in [1usize, 2, 8, 16] {
            let preds = drafter.predict(&[7, 8], k);
            assert_eq!(preds.len(), k);
            assert!(
                preds.iter().all(|&t| t < 100),
                "all fabricated tokens must be < vocab, got {preds:?}"
            );
        }
        // Short-context path (context < suffix) is bounded too.
        let short = drafter.predict(&[7], 4);
        assert!(short.iter().all(|&t| t < 100));
        // Greedy table predictions are NOT reduced (they are context copies).
        // build_from_context(&[]) left the table empty; feed one now.
        drafter.build_from_context(&[1u32, 2, 3]);
        let preds = drafter.predict(&[1, 2], 1);
        assert_eq!(preds, vec![3]);
    }

    #[test]
    fn vocab_bound_is_opt_in_legacy_unbounded() {
        // Legacy behavior (no vocab set) is unchanged: deterministic but not
        // necessarily bounded. Bench 665 relies on the old constructor.
        let mut legacy = NgramDrafter::default_trigram();
        legacy.build_from_context(&[]);
        let a = legacy.predict(&[7, 8], 4);
        let b = legacy.predict(&[7, 8], 4);
        assert_eq!(a, b, "legacy fallback stays deterministic");
        let mut bounded = NgramDrafter::default_trigram();
        bounded.set_vocab_size(10);
        bounded.build_from_context(&[]);
        assert!(bounded.predict(&[7, 8], 4).iter().all(|&t| t < 10));
    }

    #[test]
    fn vocab_zero_panics() {
        let r = std::panic::catch_unwind(|| NgramDrafter::with_vocab_size(3, 0));
        assert!(r.is_err(), "vocab_size 0 must panic (modulo-by-zero guard)");
    }

    #[test]
    fn lookup_fill_is_alloc_free_shape() {
        // The hot-path form must not need heap allocation: a caller-owned
        // buffer + Copy result. This test pins the API shape (compile-time).
        let drafter = NgramDrafter::default_trigram();
        let ctx: Vec<u32> = vec![5, 6, 7, 8, 5, 6, 7, 8];
        let mut out = [0u32; 16];
        let o1: LookupOutcome = drafter.fill_lookup_draft(&ctx, &mut out);
        let o2: LookupOutcome = o1; // Copy — no Clone needed on the hot path
        assert_eq!(o1, o2);
    }

    // ── Issue 742 T0(c) CPU proxy: regurgitation-shape hit-rate sweep ──
    //
    // The external record's doc-repro task ("reproduce this document") puts
    // the FULL document in the prompt — the model regurgitates text that is
    // already in context verbatim. Under that premise the document itself is
    // the ground-truth continuation, so lookup acceptance is measurable
    // WITHOUT the model. Byte-level tokens (coarser than BPE — an order-n
    // byte needle covers less text than an order-n BPE needle, so this is
    // CONSERVATIVE for match specificity). The real gate (T2 G1, model+
    // tokenizer on GPU) still runs in the GPU-exclusive window; this proxy
    // validates the mechanism + the order-sweep shape early.
    #[test]
    fn lookup_regurgitation_order_sweep_proxy() {
        // Frozen real repo prose (headers, tables, code fences — natural
        // repetition): the T2-era AGENTS.md snapshot (commit 992a32751),
        // byte-exact so the calibration below stays valid. Issue 859: this
        // used to `include_str!` the LIVE AGENTS.md — a moving corpus. The
        // doc grew 103 KB -> 239 KB after T2 and the ambiguity profile
        // shifted with it (order-8 mean 12.30 -> 11.37, hit_rate still 1.000
        // — mechanism fine, instrument drifted), reding the assert on every
        // docs-only commit. A calibrated gate must not read a file that
        // changes under it; re-point at the live file only if you also
        // re-derive the expected numbers.
        let text: &str = include_str!("testdata/prose_proxy.md");
        let doc: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        assert!(doc.len() > 20_000, "proxy needs a substantial document");

        // Simulate the task: prompt = the whole doc; the model regurgitates
        // it from the start. At regurgitation position i the context is
        // [doc || doc[0..i]]; the draft should fill doc[i..i+16].
        let prompt_len = doc.len();
        let sample_stride = 97usize; // decorrelated positions
        let mut context: Vec<u32> = doc.clone();

        for &order in &[3usize, 5, 8] {
            let mut drafter = NgramDrafter::default_trigram();
            drafter.set_max_lookup_order(order);
            let mut hits = 0usize;
            let mut samples = 0usize;
            let mut accept_sum = 0usize;
            let mut accept_hist = [0usize; 17]; // leading-prefix length 0..=16
            for i in (0..prompt_len.saturating_sub(16)).step_by(sample_stride) {
                // Grow the context to include the regurgitated prefix.
                if context.len() < prompt_len + i {
                    context.extend_from_slice(&doc[context.len() - prompt_len..i]);
                } else if context.len() > prompt_len + i {
                    context.truncate(prompt_len + i);
                }
                let mut out = [0u32; 16];
                let outcome = drafter.fill_lookup_draft(&context, &mut out);
                samples += 1;
                if outcome.is_hit() {
                    hits += 1;
                }
                // Spec-decode chain acceptance: leading tokens before the
                // first mismatch against the ground-truth continuation.
                let mut accepted = 0usize;
                while accepted < outcome.filled
                    && accepted < 16
                    && out[accepted] == doc[i + accepted]
                {
                    accepted += 1;
                }
                accept_sum += accepted;
                accept_hist[accepted] += 1;
            }
            let mean = accept_sum as f64 / samples as f64;
            let ge14 = accept_hist[14..].iter().sum::<usize>();
            eprintln!(
                "order {order}: samples={samples} hit_rate={:.3} mean_accepted={mean:.2} frac>=14/16={:.3}",
                hits as f64 / samples as f64,
                ge14 as f64 / samples as f64,
            );
            if order == 8 {
                // Regurgitation with an 8-byte needle: near-total acceptance
                // is the premise of the external record. Allow markdown's
                // repetitive structure some ambiguity, but demand the class.
                assert!(
                    mean >= 12.0,
                    "order-8 regurgitation mean acceptance {mean:.2} below 12 — \
                     lookup mechanism is broken, not just ambiguous"
                );
            }
        }
    }

    /// Issue 742 T0(c): lookup acceptance at K=16 with the REAL Qwen3.8 BPE
    /// tokenizer on a real ~25K+-token document (the byte-level proxy above
    /// is the conservative bound; BPE needles are more specific, so token-
    /// level acceptance should recover toward the external record's 15.0/16).
    ///
    /// Same regurgitation protocol as the proxy: the doc is the prompt, the
    /// model echoes it from the start; the draft fills the next 16 tokens
    /// and acceptance = leading tokens before the first mismatch.
    /// Skips when the qwen38 GGUF is absent (`QWEN38_GGUF` overrides).
    #[test]
    fn lookup_acceptance_real_bpe_tokens_qwen38() {
        let gguf_path = std::path::PathBuf::from(
            std::env::var("QWEN38_GGUF").unwrap_or_else(|_| {
                "F:/models/qwen38-27b-dbirks-Q4_K_M.gguf".to_string()
            }),
        );
        if !gguf_path.exists() {
            eprintln!("SKIP: {} not found", gguf_path.display());
            return;
        }
        let doc_path = std::path::PathBuf::from(
            std::env::var("QWEN38_DOC").unwrap_or_else(|_| {
                // Frozen corpus (Issue 859): the fixture the byte-level proxy
                // calibrates against, so token-level numbers stay comparable
                // across sessions. Override with QWEN38_DOC for other docs.
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/speculative_decode/testdata/prose_proxy.md"
                )
                .to_string()
            }),
        );
        let text = match std::fs::read_to_string(&doc_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP: cannot read {}: {e}", doc_path.display());
                return;
            }
        };

        let gguf = riir_infer_core::gguf_loader::GgufFile::open(&gguf_path).expect("open gguf");
        let tok = riir_infer_core::tokenizer::BpeTokenizer::from_gguf(&gguf).expect("tokenizer");
        let doc: Vec<u32> = tok.encode(&text).into_iter().map(|t| t as u32).collect();
        assert!(
            doc.len() > 25_000,
            "doc-repro sim needs >=25K real tokens, got {}",
            doc.len()
        );

        let prompt_len = doc.len();
        let sample_stride = 97usize;
        let mut context: Vec<u32> = doc.clone();

        eprintln!(
            "[T0c] doc: {} chars -> {} real BPE tokens (Qwen3.8)",
            text.len(),
            doc.len()
        );
        for &order in &[3usize, 5, 8, 10, 12] {
            let mut drafter = NgramDrafter::default_trigram();
            drafter.set_max_lookup_order(order);
            let mut hits = 0usize;
            let mut samples = 0usize;
            let mut accept_sum = 0usize;
            let mut accept_hist = [0usize; 17];
            for i in (0..prompt_len.saturating_sub(16)).step_by(sample_stride) {
                if context.len() < prompt_len + i {
                    context.extend_from_slice(&doc[context.len() - prompt_len..i]);
                } else if context.len() > prompt_len + i {
                    context.truncate(prompt_len + i);
                }
                let mut out = [0u32; 16];
                let outcome = drafter.fill_lookup_draft(&context, &mut out);
                samples += 1;
                if outcome.is_hit() {
                    hits += 1;
                }
                let mut accepted = 0usize;
                while accepted < outcome.filled
                    && accepted < 16
                    && out[accepted] == doc[i + accepted]
                {
                    accepted += 1;
                }
                accept_sum += accepted;
                accept_hist[accepted] += 1;
            }
            let mean = accept_sum as f64 / samples as f64;
            let ge14 = accept_hist[14..].iter().sum::<usize>();
            eprintln!(
                "[T0c] order {order:>2}: samples={samples} hit_rate={:.3} mean_accepted={mean:.2} frac>=14/16={:.3}",
                hits as f64 / samples as f64,
                ge14 as f64 / samples as f64,
            );
            if order == 8 {
                // Real BPE needles are strictly more specific than the byte-
                // level proxy's (12.30 mean): expect a strict improvement.
                assert!(
                    mean >= 12.5,
                    "order-8 real-token acceptance {mean:.2} should beat the \
                     byte-level proxy (12.30)"
                );
            }
        }
    }
}
