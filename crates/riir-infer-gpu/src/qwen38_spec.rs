//! Plan 614 Phase 2 — the keyed-sampled spec-decode lane's PURE decision
//! layer (the target half of the exact-sampling posture Plan 614 Phase 1
//! landed the drafter half of).
//!
//! The production lane's target decodes SAMPLED (chat T ≈ 0.6 with the
//! deployment truncation), and the whole lane's losslessness rests on one
//! definition: **the committed stream is exactly the serial keyed decode**
//! — the token sequence a one-row-at-a-time decode produces when every
//! position's token is `keyed_gumbel_max_sample_truncated(logits/T, seed,
//! position, T, top_k, top_p)`. The verify loop never decides a token by
//! itself; it only CHECKS drafts against that stream and commits whatever
//! the stream says (the drafter's pick when it matches, the target's own
//! keyed sample — the correction — when it does not).
//!
//! Everything in this module is pure CPU over downloaded logits rows and
//! M3-testable; the cudarc composition (the loop that produces the rows)
//! rides the `qwen38_spec` feature in [`crate::qwen38_spec_cudarc`].
//!
//! # The cycle contract (positions)
//!
//! With the anchor token `pending` at position `pos` (the 759-loop
//! convention — the bonus token from the previous cycle, committed to the
//! stream but not yet forwarded), a cycle feeds `draft[0] = pending` plus
//! the drafter's chain `draft[1..p]` at positions `pos..pos+p-1`. Row `r`
//! of the verify chunk predicts position `pos+r+1`; the stream's token
//! there is `s_r = sample(row r, pos+r+1)`. `draft[1+r]` is accepted
//! exactly when it equals `s_r`; at the FIRST mismatch (`s_j−1 ≠
//! draft[j]`) the stream's token `s_j−1` is the correction. On a full
//! accept the bonus is `s_p−1` (row `p−1`, position `pos+p`).
//!
//! [`keyed_accept`] is that walk. It is order-free by construction — each
//! row's verdict is a pure function of that row's logits, its position,
//! and the shared key — and the tests pin both the walk and the
//! end-to-end lossless identity against a fake serial model.

use katgpt_core::{keyed_gumbel_max_sample_truncated, truncation_keep_mask};

/// The deployment sampling posture: the serial stream's definition. The
/// defaults mirror Qwen3's chat recommendation (top-p 0.95 / top-k 20 at
/// T = 0.6) — the truncation the keyed walk's doc pins as the Phase-2
/// contract.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyedVerifyPosture {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
}

impl Default for KeyedVerifyPosture {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: Some(20),
            top_p: Some(0.95),
            seed: 0,
        }
    }
}

impl KeyedVerifyPosture {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// The stream's token at `position` given the model's full-vocab
    /// `logits` there — the ONE sampling rule, shared by the serial
    /// reference and every verify row (mask → scale → keyed argmax; the
    /// masked sampler consults survivor keys only).
    pub fn sample_row(&self, logits: &[f32], position: usize) -> u32 {
        keyed_gumbel_max_sample_truncated(
            logits,
            self.seed,
            position as u64,
            self.temperature,
            self.top_k,
            self.top_p,
        )
    }

    /// The keep-mask (exposed for parity instrumentation and tests; the
    /// sampler recomputes it internally per call — the Phase-2 posture
    /// pays one full-vocab sort per row, the Phase-3 GPU sampler replaces
    /// both).
    pub fn keep_mask(&self, logits: &[f32]) -> Vec<bool> {
        truncation_keep_mask(logits, self.top_k, self.top_p)
    }
}

/// One verify cycle's acceptance verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyedAccept {
    /// Number of FED positions accepted: the leading `j` tokens of the fed
    /// window (`draft[0] = pending` plus the chain prefix) are on the
    /// stream. `j ∈ 1..=p` — `j == 1` with a mismatched chain head is the
    /// correction cycle (the stream's own token at `pos+1` was committed
    /// as the bonus).
    pub j: usize,
    /// The stream's token at position `pos + j`: the correction when the
    /// chain mismatched at `j−1`, the bonus on a full accept. It is the
    /// next cycle's `pending` either way.
    pub bonus: u32,
}

/// The keyed acceptance walk over a verify chunk's logits rows.
///
/// `rows_logits` is the flat `[p][vocab]` verify-chunk logits (row `r` fed
/// at position `pos + r`, predicting `pos + r + 1`); `chain` is the
/// drafter's `p − 1` chain tokens (their feed positions are `pos + 1 ...
/// pos + p − 1`). Every verdict consults ONLY the shared keyed stream —
/// per-row order-invariant by construction (pinned by
/// `keyed_accept_is_order_invariant`).
pub fn keyed_accept(
    posture: &KeyedVerifyPosture,
    rows_logits: &[f32],
    vocab: usize,
    chain: &[u32],
    pos: usize,
) -> KeyedAccept {
    assert!(vocab > 0, "vocab must be nonzero");
    assert!(
        rows_logits.len().is_multiple_of(vocab),
        "rows_logits {} not a multiple of vocab {vocab}",
        rows_logits.len()
    );
    let p = rows_logits.len() / vocab;
    assert!(
        chain.len() + 1 == p,
        "chain len {} must be p−1 for p rows (got p {p})",
        chain.len()
    );
    for r in 0..chain.len() {
        let row = &rows_logits[r * vocab..(r + 1) * vocab];
        let s = posture.sample_row(row, pos + r + 1);
        if s != chain[r] {
            // First mismatch at chain index r: positions pos..pos+r accepted
            // (pending + r−1 chain tokens... exactly j = r+1 fed positions),
            // and the stream's own token s is the correction.
            return KeyedAccept { j: r + 1, bonus: s };
        }
    }
    // Full accept: the bonus comes from the LAST row (position pos + p).
    let last = &rows_logits[(p - 1) * vocab..p * vocab];
    let bonus = posture.sample_row(last, pos + p);
    KeyedAccept { j: p, bonus }
}

/// The serial keyed reference, one step: the token the stream commits at
/// `position` from a one-row forward's logits. The lossless gate compares
/// the lane's committed stream against this walk of one-row logits —
/// the two agree exactly when the verify chunk's rows are bit-identical
/// to the one-row forwards at the accepted positions (the 556 T9.9
/// chunk-identity contract, re-proven end-to-end by the 4090 G1).
pub fn serial_keyed_step(posture: &KeyedVerifyPosture, logits: &[f32], position: usize) -> u32 {
    posture.sample_row(logits, position)
}

// ── riir-ai Issue 1142: relaxed acceptance on the keyed lane ─────────

/// The PLM relaxed-acceptance knob for the keyed lane (riir-ai Issue 1142,
/// katgpt-rs Issue 973, arXiv:2610.26693 Alg. 1): a drafted token `d` at a
/// verify row is accepted iff `d == s` (the keyed stream's own sample — the
/// verifier-side draw, the exact term) `∨ d ∈ ρ-nucleus(q) ∨ q(d) ≥ ε`, where
/// `q` is the posture's deployed law at that row (truncation keep-mask, then
/// `softmax(logits / T)` over survivors). Rejection commits `s` — no residual.
///
/// LOSSY BY DESIGN away from [`RelaxedAccept::EXACT`]: an admitted `d ≠ s`
/// leaves the serial keyed stream, so the committed stream stops being the
/// serial decode. The quality budget (MAUVE within 1% of exact) is fitted by
/// the riir-ai 1142 calibration lane, never assumed. `q(d) = 0` outside the
/// keep-mask, so a relaxed accept never emits outside the deployed truncation.
/// Feature `relaxed_accept`.
#[cfg(feature = "relaxed_accept")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelaxedAccept {
    pub rho: f32,
    pub eps: f32,
}

#[cfg(feature = "relaxed_accept")]
impl RelaxedAccept {
    /// Admits nothing (`ρ = 0`, `ε = ∞`): [`keyed_accept_relaxed`] is then
    /// bit-identical to [`keyed_accept`] — the lossless corner. (The paper's
    /// `ε = 1` corner differs only where f32 rounds a survivor's `q` to 1.0.)
    pub const EXACT: Self = Self {
        rho: 0.0,
        eps: f32::INFINITY,
    };

    pub fn is_exact(&self) -> bool {
        self.rho <= 0.0 && self.eps > 1.0
    }
}

#[cfg(feature = "relaxed_accept")]
impl KeyedVerifyPosture {
    /// The posture's deployed law at one row, written into `out` (len =
    /// vocab): the keep-mask survivors' `softmax(logits / T)`, zero
    /// elsewhere. The greedy posture (T ≤ 0) has no sampled law; it reads the
    /// untempered (T = 1) survivor softmax so the relaxed terms stay defined.
    /// Max-subtracted, sequential ascending folds (Issue 940 order pin).
    pub fn row_law_into(&self, logits: &[f32], out: &mut [f32]) {
        assert_eq!(logits.len(), out.len(), "row_law_into: len mismatch");
        let keep = self.keep_mask(logits);
        let t = if self.is_greedy() {
            1.0
        } else {
            self.temperature
        };
        let mut max = f32::NEG_INFINITY;
        for (&l, &k) in logits.iter().zip(&keep) {
            if k && l > max {
                max = l;
            }
        }
        let mut sum = 0.0f32;
        for ((o, &l), &k) in out.iter_mut().zip(logits).zip(&keep) {
            *o = if k { ((l - max) / t).exp() } else { 0.0 };
            sum += *o;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }
    }
}

/// [`keyed_accept`] under a [`RelaxedAccept`] rule. Returns the verdict plus
/// the number of RELAXED admits (chain tokens accepted while `≠ s`) — the
/// lossy-acceptance count the calibration lane reports beside accepted/call.
///
/// Row `r`'s verdict is a pure function of that row's logits, position, key
/// and the rule (the [`keyed_accept`] order-invariance carries over). The
/// law `q` is computed only on rows where the exact term fails. `law` is a
/// caller-owned scratch of length `vocab` (reused across cycles).
///
/// Monotone: loosening (`ρ↑` or `ε↓`) grows every row's admit set, so the
/// first rejected row can only move later — `j` never decreases.
#[cfg(feature = "relaxed_accept")]
pub fn keyed_accept_relaxed(
    posture: &KeyedVerifyPosture,
    rule: RelaxedAccept,
    rows_logits: &[f32],
    vocab: usize,
    chain: &[u32],
    pos: usize,
    law: &mut [f32],
) -> (KeyedAccept, usize) {
    assert!(vocab > 0, "vocab must be nonzero");
    assert_eq!(law.len(), vocab, "law scratch must be vocab-sized");
    assert!(
        rows_logits.len().is_multiple_of(vocab),
        "rows_logits {} not a multiple of vocab {vocab}",
        rows_logits.len()
    );
    let p = rows_logits.len() / vocab;
    assert!(
        chain.len() + 1 == p,
        "chain len {} must be p−1 for p rows (got p {p})",
        chain.len()
    );
    let mut relaxed = 0usize;
    for r in 0..chain.len() {
        let row = &rows_logits[r * vocab..(r + 1) * vocab];
        let s = posture.sample_row(row, pos + r + 1);
        if s == chain[r] {
            continue;
        }
        if !rule.is_exact() {
            posture.row_law_into(row, law);
            if katgpt_core::relaxed_accept::relaxed_admit(
                law,
                chain[r] as usize,
                rule.rho,
                rule.eps,
            ) {
                relaxed += 1;
                continue;
            }
        }
        return (KeyedAccept { j: r + 1, bonus: s }, relaxed);
    }
    let last = &rows_logits[(p - 1) * vocab..p * vocab];
    let bonus = posture.sample_row(last, pos + p);
    (KeyedAccept { j: p, bonus }, relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::{keyed_gumbel_max_sample, keyed_gumbel_noise};

    /// A deterministic fake model: full-vocab logits as a pure function of
    /// the context — `logits[i] = ((h(context) ⊕ i·GOLDEN) finalizer bits
    /// scaled)` — so both the serial reference and the verify rows derive
    /// from one arithmetic truth, and every "forward" is exact.
    struct FakeModel {
        vocab: usize,
    }

    impl FakeModel {
        fn fin(z: u64) -> u64 {
            let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn ctx_hash(&self, ctx: &[u32]) -> u64 {
            let mut h = 0x1234_5678_9ABC_DEF0u64 ^ (self.vocab as u64);
            for (i, &t) in ctx.iter().enumerate() {
                h = Self::fin(h ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64));
            }
            h
        }
        fn logits(&self, ctx: &[u32]) -> Vec<f32> {
            let h = self.ctx_hash(ctx);
            (0..self.vocab as u64)
                .map(|i| {
                    let z = Self::fin(h ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    (((z >> 40) as f32) / 16777216.0 - 128.0) * 0.1
                })
                .collect()
        }
    }

    /// A drafter whose chain proposes, at each chain step, from a PERTURBED
    /// view of the model's distribution (`quality ∈ [0,1]`: 1 = the true
    /// logits, 0 = a fresh distribution = guaranteed junk). The perturbed
    /// candidate set is the perturbed logits' top-`top_k` — the same
    /// top-k-then-score shape the real selector has, minus the lattice.
    struct FakeDrafter {
        model: FakeModel,
        quality: f32,
    }

    impl FakeDrafter {
        fn chain(
            &self,
            ctx: &[u32],
            pos: usize,
            posture: &KeyedVerifyPosture,
            n: usize,
        ) -> Vec<u32> {
            let mut chain = Vec::with_capacity(n);
            let mut local = ctx.to_vec();
            for i in 0..n {
                let truth = self.model.logits(&local);
                let junk_seed = 0xF00D_u64 ^ (i as u64) ^ (pos as u64);
                let junk: Vec<f32> = (0..self.model.vocab as u64)
                    .map(|t| {
                        let z = FakeModel::fin(junk_seed ^ t.wrapping_mul(0xBF58_476D_1CE4_E5B9));
                        (((z >> 40) as f32) / 16777216.0 - 128.0) * 0.1
                    })
                    .collect();
                let mixed: Vec<f32> = truth
                    .iter()
                    .zip(&junk)
                    .map(|(&a, &b)| self.quality * a + (1.0 - self.quality) * b)
                    .collect();
                // The proposal: keyed pick over the perturbed distribution
                // with the SHARED keys (the keyed drafter's convention).
                let pick = keyed_gumbel_max_sample(
                    &mixed
                        .iter()
                        .map(|&l| l / posture.temperature)
                        .collect::<Vec<_>>(),
                    posture.seed,
                    (pos + 1 + i) as u64,
                );
                chain.push(pick);
                local.push(pick);
            }
            chain
        }
    }

    fn spec_stream(
        model: &FakeModel,
        drafter: &FakeDrafter,
        posture: &KeyedVerifyPosture,
        prompt: &[u32],
        steps: usize,
        block: usize,
    ) -> (Vec<u32>, Vec<usize>) {
        spec_stream_with(
            model,
            drafter,
            posture,
            posture,
            prompt,
            steps,
            block,
            &mut |flat, chain, pos| keyed_accept(posture, flat, model.vocab, chain, pos),
        )
    }

    /// One verify cycle's acceptance walk: `(flat rows, chain, pos)`.
    type AcceptFn<'a> = dyn FnMut(&[f32], &[u32], usize) -> KeyedAccept + 'a;

    /// [`spec_stream`] with the acceptance walk and the drafter's key
    /// posture injected (the relaxed rule's tests drive the same loop).
    #[allow(clippy::too_many_arguments)]
    fn spec_stream_with(
        model: &FakeModel,
        drafter: &FakeDrafter,
        posture: &KeyedVerifyPosture,
        draft_posture: &KeyedVerifyPosture,
        prompt: &[u32],
        steps: usize,
        block: usize,
        accept: &mut AcceptFn<'_>,
    ) -> (Vec<u32>, Vec<usize>) {
        // Serial prefix: teacher-force the prompt, then seed the stream
        // with the FIRST pending (the stream's token at pos — the 759-loop
        // `generated = vec![first]` shape: the pending is a real stream
        // token, committed by definition before any cycle runs).
        let mut ctx = prompt.to_vec();
        let bonus_logits = model.logits(&ctx);
        let first = posture.sample_row(&bonus_logits, prompt.len());
        let mut stream: Vec<u32> = vec![first];
        let mut js: Vec<usize> = Vec::new();
        let mut pos = prompt.len();
        while stream.len() < steps {
            // Draft (pending = the stream's token at pos — always present:
            // the stream is seeded with the first pending before the loop).
            let pending = *stream.last().expect("seeded stream");
            let mut chain_ctx = ctx.clone();
            chain_ctx.push(pending);
            let chain = drafter.chain(&chain_ctx, pos, draft_posture, block - 1);
            // Verify rows: the IDEAL chunk — row r = the one-row logits at
            // pos+r (the chunk-identity contract holds by construction).
            let mut rows: Vec<Vec<f32>> = Vec::with_capacity(block);
            let mut feed = ctx.clone();
            feed.push(pending);
            rows.push(model.logits(&feed));
            for &c in &chain {
                feed.push(c);
                rows.push(model.logits(&feed));
            }
            let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * model.vocab);
            for r in &rows {
                flat.extend_from_slice(r);
            }
            let acc = accept(&flat, &chain, pos);
            // Commit. STREAM: the accepted chain tokens + the bonus (the
            // pending is already on the stream from the previous cycle's
            // bonus — or the seed above). MODEL CONTEXT: the chunk
            // FORWARDED positions pos..pos+j-1 — the pending and the
            // accepted chain — so exactly those join `ctx`; the bonus at
            // pos+j stays unforwarded (it is the next cycle's pending, fed
            // by the next chunk's row 0).
            ctx.push(pending);
            for &c in &chain[..acc.j - 1] {
                stream.push(c);
                ctx.push(c);
            }
            stream.push(acc.bonus);
            js.push(acc.j);
            pos += acc.j;
        }
        stream.truncate(steps);
        (stream, js)
    }

    fn serial_stream(
        model: &FakeModel,
        posture: &KeyedVerifyPosture,
        prompt: &[u32],
        steps: usize,
    ) -> Vec<u32> {
        let mut ctx = prompt.to_vec();
        let mut out = Vec::with_capacity(steps);
        for i in 0..steps {
            let lg = model.logits(&ctx);
            let t = serial_keyed_step(posture, &lg, prompt.len() + i);
            out.push(t);
            ctx.push(t);
        }
        out
    }

    #[test]
    fn keyed_accept_walk_matches_hand_computed_keyed_samples() {
        // vocab 8; two rows; chain of 1. Compute the stream's tokens with
        // the raw sampler and check both the mismatch and full-accept paths.
        let posture = KeyedVerifyPosture {
            temperature: 0.7,
            top_k: Some(4),
            top_p: Some(0.9),
            seed: 123,
        };
        let vocab = 8usize;
        let mk_row = |h: u64| -> Vec<f32> {
            (0..vocab as u64)
                .map(|i| {
                    let z = FakeModel::fin(h ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    (((z >> 40) as f32) / 16777216.0 - 128.0) * 0.1
                })
                .collect()
        };
        let row0 = mk_row(0xAA);
        let row1 = mk_row(0xBB);
        let mut flat = row0.clone();
        flat.extend_from_slice(&row1);
        let s0 = posture.sample_row(&row0, 101);
        let s1 = posture.sample_row(&row1, 102);
        // Mismatch path: chain head wrong → j=1, bonus = the stream's token.
        let acc = keyed_accept(&posture, &flat, vocab, &[if s0 == 3 { 4 } else { 3 }], 100);
        assert_eq!(acc.j, 1);
        assert_eq!(acc.bonus, s0);
        // Full accept: chain head right → j=2 (p), bonus = row 1's sample.
        let acc = keyed_accept(&posture, &flat, vocab, &[s0], 100);
        assert_eq!(acc.j, 2);
        assert_eq!(acc.bonus, s1);
        // The walk equals the serial rule at each position, row for row.
        assert_eq!(s0, serial_keyed_step(&posture, &row0, 101));
    }

    #[test]
    fn keyed_accept_is_order_invariant() {
        // The contract that makes the keyed stream batch-order-free: each
        // row's verdict is a pure function of that row's logits, its
        // position, and the key — no hidden state, no neighbor dependence —
        // so the walk's j is exactly the first ascending row whose stream
        // token disagrees with the chain, whatever order the rows were
        // computed in.
        let posture = KeyedVerifyPosture::default();
        let vocab = 16usize;
        let rows: Vec<Vec<f32>> = (0..5)
            .map(|r| {
                (0..vocab)
                    .map(|i| ((r * 31 + i * 7) as f32).sin() * 3.0)
                    .collect()
            })
            .collect();
        let mut flat = Vec::new();
        for r in &rows {
            flat.extend_from_slice(r);
        }
        let verdict = |r: usize| posture.sample_row(&rows[r], 50 + r + 1);
        // The full verdict set, computed in REVERSE order (purity: calling
        // the sampler in any order yields the same set).
        let mut rev_verdicts = [0u32; 5];
        for r in (0..5).rev() {
            rev_verdicts[r] = verdict(r);
        }
        // Three chains: mismatch at 0, in the middle, none — j must equal
        // the first ascending disagreement, computed from either verdict
        // ordering.
        for chain in [
            vec![
                rev_verdicts[0] ^ 1,
                rev_verdicts[1],
                rev_verdicts[2],
                rev_verdicts[3],
            ],
            vec![
                rev_verdicts[0],
                rev_verdicts[1],
                rev_verdicts[2] ^ 1,
                rev_verdicts[3],
            ],
            vec![
                rev_verdicts[0],
                rev_verdicts[1],
                rev_verdicts[2],
                rev_verdicts[3],
            ],
        ] {
            let expected = (0..4)
                .find(|&r| rev_verdicts[r] != chain[r])
                .map_or(5, |r| r + 1);
            let a = keyed_accept(&posture, &flat, vocab, &chain, 50);
            assert_eq!(a.j, expected, "chain {chain:?}");
            // Corrupting a LATER row never moves an earlier verdict.
            if expected <= 4 {
                let mut corrupted = flat.clone();
                let off = 4 * vocab;
                for v in &mut corrupted[off..] {
                    *v = -1.0e9;
                }
                let c = keyed_accept(&posture, &corrupted, vocab, &chain, 50);
                assert_eq!(c.j, expected, "later-row corruption moved the verdict");
            }
        }
    }

    #[test]
    fn spec_stream_is_lossless_against_the_serial_keyed_decode() {
        // THE Phase-2 contract, end to end on a fake model: the spec loop
        // (drafts + verify rows + keyed_accept + correction commits)
        // reproduces the serial keyed decode token-for-token — with a GOOD
        // drafter (many full accepts), a WEAK drafter (frequent corrections)
        // and a JUNK drafter (every cycle is a correction).
        let model = FakeModel { vocab: 64 };
        let prompt: Vec<u32> = (10..35).collect();
        for (label, quality) in [("good", 1.0f32), ("weak", 0.35), ("junk", 0.0)] {
            let posture = KeyedVerifyPosture {
                temperature: 0.6,
                top_k: Some(8),
                top_p: Some(0.95),
                seed: 0x5EED,
            };
            let drafter = FakeDrafter {
                model: FakeModel { vocab: 64 },
                quality,
            };
            let (stream, js) = spec_stream(&model, &drafter, &posture, &prompt, 60, 5);
            let serial = serial_stream(&model, &posture, &prompt, 60);
            assert_eq!(
                stream, serial,
                "{label}: spec stream diverged from the serial keyed decode"
            );
            assert_eq!(stream.len(), 60);
            if label == "junk" {
                // Correction-DOMINATED: a junk pick still coincides with
                // the stream's token ~1/vocab of the time (64 vocab × 40
                // cycles → a handful of lucky j=2 cycles is the correct
                // behavior, not a bug).
                let corrections = js.iter().filter(|&&j| j == 1).count();
                assert!(
                    corrections * 5 >= js.len() * 4,
                    "junk drafter must correct almost every cycle: {js:?}"
                );
            } else {
                assert!(
                    js.iter().any(|&j| j > 1),
                    "{label}: expected at least one non-correction cycle: {js:?}"
                );
            }
        }
    }

    #[test]
    fn greedy_posture_matches_the_plain_argmax_stream() {
        // temperature <= 0: the posture is the argmax (over survivors) with
        // no noise — the serial stream is the classic greedy decode, and the
        // masked sampler's survivor restriction is visible (the argmax token
        // masked out demotes to the best survivor).
        let posture = KeyedVerifyPosture {
            temperature: 0.0,
            top_k: Some(2),
            top_p: None,
            seed: 7,
        };
        let logits = [-5.0f32, 9.0, 8.5, -5.0];
        let mask = posture.keep_mask(&logits);
        assert_eq!(mask, vec![false, true, true, false]);
        assert_eq!(posture.sample_row(&logits, 3), 1);
        let mut masked = vec![f32::NEG_INFINITY; 4];
        for (i, &k) in mask.iter().enumerate() {
            if k {
                masked[i] = logits[i];
            }
        }
        assert_eq!(
            posture.sample_row(&logits, 3),
            keyed_gumbel_max_sample(&masked, 7, 3)
        );
    }

    #[test]
    fn keyed_noise_bound_documented_via_sampler_smoke() {
        // Smoke the shared noise the sampler rides (the bound the
        // greedy-in-the-limit claims leans on): finite everywhere, and a
        // logit gap far above the bound forces the argmax even with a mask.
        let posture = KeyedVerifyPosture {
            temperature: 0.6,
            top_k: Some(3),
            top_p: None,
            seed: 3,
        };
        let mut logits = vec![-1000.0f32; 32];
        logits[17] = 1000.0;
        for p in 0..50usize {
            assert_eq!(posture.sample_row(&logits, p), 17);
        }
        for p in 0..200u64 {
            let g = keyed_gumbel_noise(9, p, 11);
            assert!(g.is_finite() && g.abs() < 64.0);
        }
    }

    #[cfg(feature = "relaxed_accept")]
    mod relaxed {
        use super::*;

        fn posture() -> KeyedVerifyPosture {
            KeyedVerifyPosture {
                temperature: 0.8,
                top_k: Some(16),
                top_p: Some(0.95),
                seed: 0x1142,
            }
        }

        fn run(rule: RelaxedAccept, quality: f32) -> (Vec<u32>, Vec<usize>, usize) {
            run_seeded(Some(rule), quality, posture().seed)
        }

        /// `rule = None` drives the plain [`keyed_accept`] walk over the
        /// same drafter (the corner test's reference).
        fn run_seeded(
            rule: Option<RelaxedAccept>,
            quality: f32,
            seed: u64,
        ) -> (Vec<u32>, Vec<usize>, usize) {
            let model = FakeModel { vocab: 64 };
            let drafter = FakeDrafter {
                model: FakeModel { vocab: 64 },
                quality,
            };
            let posture = KeyedVerifyPosture { seed, ..posture() };
            // INDEPENDENT drafter keys: the fake logits are flat (span 0.1),
            // so a shared-key drafter's pick is the Gumbel argmax on both
            // sides — every in-mask draft equals the stream's sample and the
            // rule has nothing to admit. Independent keys give in-mask
            // mismatches (the real drafter's case: its own distribution).
            let draft_posture = KeyedVerifyPosture {
                seed: seed ^ 0xD4AF_7000,
                ..posture.clone()
            };
            let prompt: Vec<u32> = (3..28).collect();
            let mut law = vec![0.0f32; 64];
            let mut n_relaxed = 0usize;
            let (stream, js) = spec_stream_with(
                &model,
                &drafter,
                &posture,
                &draft_posture,
                &prompt,
                120,
                6,
                &mut |f, c, p| {
                    let Some(rule) = rule else {
                        return keyed_accept(&posture, f, 64, c, p);
                    };
                    let (acc, r) = keyed_accept_relaxed(&posture, rule, f, 64, c, p, &mut law);
                    n_relaxed += r;
                    acc
                },
            );
            (stream, js, n_relaxed)
        }

        #[test]
        fn exact_corner_is_bit_identical_to_keyed_accept_and_lossless() {
            for quality in [1.0f32, 0.6, 0.35, 0.0] {
                let (stream, js, n_relaxed) = run(RelaxedAccept::EXACT, quality);
                let (exact, exact_js, _) = run_seeded(None, quality, posture().seed);
                let model = FakeModel { vocab: 64 };
                let prompt: Vec<u32> = (3..28).collect();
                assert_eq!(
                    stream, exact,
                    "q={quality}: corner stream != keyed_accept stream"
                );
                assert_eq!(
                    js, exact_js,
                    "q={quality}: corner verdicts != keyed_accept verdicts"
                );
                assert_eq!(n_relaxed, 0);
                assert_eq!(stream, serial_stream(&model, &posture(), &prompt, 120));
            }
        }

        #[test]
        fn row_law_is_the_truncated_tempered_softmax() {
            let posture = KeyedVerifyPosture {
                temperature: 0.5,
                top_k: Some(2),
                top_p: None,
                seed: 1,
            };
            let logits = [1.0f32, 3.0, 2.0, -4.0];
            let mut q = [0.0f32; 4];
            posture.row_law_into(&logits, &mut q);
            let (a, b) = ((0.0f32).exp(), ((2.0f32 - 3.0) / 0.5).exp());
            assert_eq!(q[0], 0.0);
            assert_eq!(q[3], 0.0);
            assert!((q[1] - a / (a + b)).abs() < 1e-6);
            assert!((q[2] - b / (a + b)).abs() < 1e-6);
        }

        /// Per-cycle monotonicity on identical verify rows: loosening never
        /// lowers `j` (checked row-by-row on fixed fixtures, so the claim is
        /// the walk's, not an artefact of diverging streams).
        #[test]
        fn loosening_never_lowers_j_on_fixed_rows() {
            let posture = posture();
            let vocab = 32usize;
            let mut law = vec![0.0f32; vocab];
            let grid = [
                RelaxedAccept::EXACT,
                RelaxedAccept { rho: 0.5, eps: 0.5 },
                RelaxedAccept { rho: 0.8, eps: 0.1 },
                RelaxedAccept {
                    rho: 0.95,
                    eps: 0.02,
                },
                RelaxedAccept { rho: 1.0, eps: 0.0 },
            ];
            let mut loosest_gain = 0usize;
            for case in 0..400u64 {
                let rows: Vec<f32> = (0..6 * vocab as u64)
                    .map(|i| {
                        let z =
                            FakeModel::fin(case.wrapping_mul(0x9E37) ^ i.wrapping_mul(0xBF58_476D));
                        (((z >> 40) as f32) / 16777216.0 - 128.0) * 0.08
                    })
                    .collect();
                let chain: Vec<u32> = (0..5u64)
                    .map(|i| (FakeModel::fin(case ^ (i << 32)) % vocab as u64) as u32)
                    .collect();
                let js: Vec<usize> = grid
                    .iter()
                    .map(|&g| {
                        keyed_accept_relaxed(&posture, g, &rows, vocab, &chain, 40, &mut law)
                            .0
                            .j
                    })
                    .collect();
                for w in js.windows(2) {
                    assert!(w[1] >= w[0], "case {case}: loosening lowered j: {js:?}");
                }
                loosest_gain += js[4] - js[0];
            }
            assert!(
                loosest_gain >= 200,
                "non-vacuity: loosest rule gained only {loosest_gain}"
            );
        }

        /// The ρ = 1, ε = 0 extreme admits every survivor: a chain drawn
        /// inside the keep-mask is accepted in full.
        #[test]
        fn loosest_rule_accepts_any_survivor_and_never_a_masked_token() {
            let posture = posture();
            let vocab = 32usize;
            let rows: Vec<f32> = (0..3 * vocab)
                .map(|i| ((i * 37 % 101) as f32) * 0.05)
                .collect();
            let mut law = vec![0.0f32; vocab];
            let loose = RelaxedAccept { rho: 1.0, eps: 0.0 };
            let keep0 = posture.keep_mask(&rows[..vocab]);
            let in_mask = keep0.iter().position(|&k| k).unwrap() as u32;
            let out_mask = keep0.iter().position(|&k| !k).unwrap() as u32;
            let keep1 = posture.keep_mask(&rows[vocab..2 * vocab]);
            let in1 = keep1.iter().position(|&k| k).unwrap() as u32;
            let (acc, _) =
                keyed_accept_relaxed(&posture, loose, &rows, vocab, &[in_mask, in1], 7, &mut law);
            assert_eq!(acc.j, 3);
            let (acc, _) =
                keyed_accept_relaxed(&posture, loose, &rows, vocab, &[out_mask, in1], 7, &mut law);
            assert_eq!(acc.j, 1, "a masked token must never be admitted");
        }

        /// Lossy away from the corner, and that is the point: a weak drafter
        /// gains accepted positions per cycle, and the stream leaves the serial
        /// decode (the quality budget for that departure is the 1142 lane's).
        #[test]
        fn relaxed_rule_gains_acceptance_and_is_lossy() {
            let (_, exact_js, _) = run(RelaxedAccept::EXACT, 0.35);
            let (stream, js, n_relaxed) = run(
                RelaxedAccept {
                    rho: 0.9,
                    eps: 0.02,
                },
                0.35,
            );
            let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
            assert!(n_relaxed > 0);
            assert!(
                mean(&js) > mean(&exact_js) * 1.25,
                "relaxed mean_j {} vs exact {}",
                mean(&js),
                mean(&exact_js)
            );
            let model = FakeModel { vocab: 64 };
            let prompt: Vec<u32> = (3..28).collect();
            assert_ne!(stream, serial_stream(&model, &posture(), &prompt, 120));
        }
    }
}
