//! Plan 622 TokenGate lane switches (T1.5).
//!
//! Two-level gate over the prompt-skimming lane (per-key-bias attention +
//! exact pruned prefill + the deterministically-constructed gate overlay):
//!
//! - **Compile-time** — the `tokengate` cargo feature. OFF: the lane's
//!   serving code compiles away and [`runtime_enabled`] is a constant
//!   `false` that never reads the environment.
//! - **Runtime** — `RIIR_TOKENGATE` DEMOTES inside the feature (the
//!   kill-switch direction: an armed lane is disabled by env, never
//!   enabled by env — the `RIIR_MMORPG_EXAMPLES_GOAL_SALIENCE=0` /
//!   `LAYA_METAL_FLASH=0` posture). Demote literals: `0`, `false`, `off`
//!   (case-insensitive, trimmed); anything else — including unset —
//!   leaves the lane armed.
//!
//! The bias KERNEL capability (Phase 1's `*_bias` attention variants) is
//! deliberately NOT gated: it rides `Option` parameters and is a dispatch
//! identity at `None`. This feature gates the LANE — the serving quartet +
//! gate overlay (Phases 2–3) and the lane's harness targets (the
//! `required-features` rows). Phase 2's first consumer calls
//! [`runtime_enabled`] once at boot; callers must not read it per tick.

/// Compile-time-and-runtime gate read. Call ONCE at boot (the read is
/// cached; the feature-off arm folds to a constant `false` with no
/// environment access — the flag-off posture compiles the whole read away).
pub fn runtime_enabled() -> bool {
    #[cfg(feature = "tokengate")]
    {
        use std::sync::OnceLock;
        static ARMED: OnceLock<bool> = OnceLock::new();
        *ARMED.get_or_init(|| match std::env::var("RIIR_TOKENGATE") {
            Ok(v) => !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            ),
            Err(_) => true,
        })
    }
    #[cfg(not(feature = "tokengate"))]
    {
        false
    }
}

/// Plan 622 T2.1 — the scaffold-masked top-K selection (pure fn).
///
/// Scaffold = original positions `< n_sink` (the `kv_evict` sink convention,
/// `SinkWindowPolicy::default().n_sink`) PLUS the final position (the logits
/// source — always kept regardless of score). Among the remaining prunable
/// positions, the `k` highest scores survive; `k >=` prunable count keeps
/// everything. Ties break FIRST-INDEX (the determinism law); NaN scores sort
/// WORST — demoted to −inf in the sort key (NOT by `total_cmp` alone: IEEE
/// totalOrder puts NaN above +inf ascending, so a raw total_cmp descending
/// would rank NaN FIRST — pinned by the select_nan unit test). Writes
/// ASCENDING original indices into `out` (cleared first). One scratch
/// allocation for the prunable index sort (CPU reference — the G4 zero-alloc
/// law prices the serving path, not this lane's prune step).
#[cfg(feature = "tokengate")]
pub fn select_kept_into(scores: &[f32], n_sink: usize, k: usize, out: &mut Vec<usize>) {
    let t = scores.len();
    out.clear();
    // Scaffold: first `n_sink` positions + the final position (deduped below
    // — the two overlap when t <= n_sink).
    out.extend(0..n_sink.min(t));
    if t == 0 {
        return;
    }
    out.push(t - 1);

    // Prunable range: (n_sink .. t-1). Empty when the scaffold already covers
    // everything except (possibly) the last token.
    let prunable = n_sink..t.saturating_sub(1);
    let prunable_len = prunable.len();
    if k > 0 && prunable_len > 0 {
        let key = |s: f32| if s.is_nan() { f32::NEG_INFINITY } else { s };
        let mut ranked: Vec<usize> = prunable.collect();
        // Descending by score; equal scores keep ascending original index.
        // NaN demotes to −inf BEFORE the total_cmp (see the doc: a raw
        // total_cmp ranks NaN FIRST descending — the opposite of the law).
        ranked.sort_by(|&a, &b| {
            key(scores[b])
                .total_cmp(&key(scores[a]))
                .then_with(|| a.cmp(&b))
        });
        let take = k.min(prunable_len);
        out.extend_from_slice(&ranked[..take]);
    }

    out.sort_unstable();
    out.dedup();
}

/// Logit-space bias that makes a pad key dead under any finite real score:
/// `exp(-1e9 - max) == 0.0` exactly in f32, so the pad contributes `+0.0` to
/// the softmax denominator and `+0.0 * v_pad` to the numerator — the real
/// keys' sums are bitwise unchanged (a real key always supplies the max, and
/// `+0.0` additions cannot flip a non-negative partial sum).
pub const PAD_KEY_BIAS: f32 = -1.0e9;

/// Plan 622 T2.2 — the bucketed shape's pad-key bias rows (pure fn).
///
/// Writes head-major `[n_head][padded_len]` (the
/// `crate::transformer::KeyBiasLayout` contract with
/// `head_stride = padded_len`): slots `< real_len` get `0.0` (inert),
/// slots `>= real_len` get [`PAD_KEY_BIAS`] (dead key). This is the table the
/// fixed-shape arm (T2.4's graph-per-padded-length) relies on — its captured
/// kernel attends the FULL padded window per query, so the pads must be
/// masked in logit space, not by a per-query causal bound. The bucketed
/// CPU reference (`prefill_qwen_deltanet_pruned_into` with `bucket`) does NOT
/// consume this on its own scoring path — its real queries are causally
/// bounded below `real_len`, so pads are already unreachable and the bias
/// add would be dead weight; the law it embodies is pinned at kernel level
/// by `tests/plan622_bucketed_prefill_g1.rs` (full-window + pad bias ≡
/// causal window, bit-for-bit). When a caller bias table joins (Phase 3's
/// gate overlay, kept-order rows), the COMPOSED table must extend the same
/// way: gate rows for the kept slots, [`PAD_KEY_BIAS`] for the pads.
///
/// Zero-alloc (`*_into` form); `dst` is `n_head * padded_len`.
#[cfg(feature = "tokengate")]
pub fn pad_bias_rows_into(dst: &mut [f32], n_head: usize, real_len: usize, padded_len: usize) {
    assert!(
        dst.len() >= n_head * padded_len,
        "pad-bias buffer {} < {} rows",
        dst.len(),
        n_head * padded_len
    );
    assert!(real_len <= padded_len, "real rows exceed the padded width");
    for h in 0..n_head {
        let row = &mut dst[h * padded_len..][..padded_len];
        row[..real_len].fill(0.0);
        row[real_len..].fill(PAD_KEY_BIAS);
    }
}

/// Plan 622 T2.3 — the frozen prefill result a pruned DECODE consumes.
///
/// `full_rows` = the prefill's full prompt length T (the `<= L` band's row
/// count and the ORIGINAL-position clock); `kept_rows` = the `> L` band's
/// PACKED row count (`PrunedPrefillPlan::real_len`); `prune_at_layer` = L.
/// Invariant: the scaffold keeps the original final token, so the last row
/// of both bands is original position T-1 — a decode token's RoPE position
/// is `full_rows + step` on EVERY layer (one position clock), while the KV
/// write slot is band-local (`full_rows + step` at `<= L`, `kept_rows +
/// step` above). That slot/phase split is the decode half of the subset-
/// softmax exactness law (compaction moves storage, never phases — the
/// `kv_evict` law, same shape).
#[cfg(feature = "tokengate")]
pub struct PrunedDecodeLayout {
    pub prune_at_layer: usize,
    pub full_rows: usize,
    pub kept_rows: usize,
}

#[cfg(feature = "tokengate")]
impl PrunedDecodeLayout {
    /// Build from the prefill's own bookkeeping (`prompt_len` = the full T).
    pub fn from_plan(
        prompt_len: usize,
        plan: &crate::deltanet::forward::PrunedPrefillPlan,
    ) -> Self {
        Self {
            prune_at_layer: plan.prune_at_layer,
            full_rows: prompt_len,
            kept_rows: plan.real_len,
        }
    }
}

/// Plan 622 T2.3 — the gate FOLDED ONCE at prefill end into per-layer
/// decode-side surfaces: an additive per-key mask (the bias rows, PACKED
/// slot order) + the per-head logit scale `s_eff` (applied as a query
/// multiply — `(s·q)·k = s·(q·k)`, the bias rides AFTER the scale, so the
/// mask entries are unscaled logits). Dense layout, the T1.2 CUDA wire
/// shape: layer l's bias window starts at `l * n_head * head_stride`, rows
/// head-major; layer l's scales sit at `scale[l * n_head ..][..n_head]`.
/// Layers `<= prune_at_layer` never read their windows (the gate exists
/// above L only), and GDN layers consume neither surface.
///
/// CONTRACT (the constructor's — Phase 3's scorer builds this at prefill
/// end, once): `head_stride >= kept_rows + every decode step the caller
/// will run`, slots `< kept_rows` carry the prompt mask, slots `>=
/// kept_rows` are ZERO (decode-appended keys are always attended — bias
/// 0, they are never pruned) and scales are exactly `1.0` where the lane
/// has no opinion (the `≡ 1` whole-table check then skips the q-multiply
/// entirely — dispatch identity, and `q * 1.0` is a bitwise no-op besides).
#[cfg(feature = "tokengate")]
pub struct PrunedDecodeGate<'a> {
    pub bias: &'a [f32],
    pub head_stride: usize,
    pub scale: Option<&'a [f32]>,
}

/// Plan 622 T3.1 — the rank-ratio law `r = ln(max(2, m)) / ln(max(2, p))`:
/// the query's rank among KEPT tokens over its rank in the FULL prompt
/// (both 1-based; `m <= p` always — kept ⊆ full). The `max(2, ·)` guard is
/// SYMMETRIC on purpose: a denominator-only guard breaks the `r ≡ 1`
/// identity at rank 1 (ln 1 = 0 over ln 2 = 0 gives r = 0 for the very
/// token whose m == p) — the unit test pins the identity at p = 1 too.
/// `m == p` (the K = full posture) short-circuits to exactly 1.0 so the
/// overlay is an EXACT no-op there for any γ.
#[cfg(feature = "tokengate")]
#[inline]
pub fn rank_ratio(rank_kept: usize, rank_full: usize) -> f32 {
    debug_assert!(rank_kept >= 1 && rank_full >= 1, "ranks are 1-based");
    debug_assert!(
        rank_kept <= rank_full,
        "kept rank {} exceeds full rank {}",
        rank_kept,
        rank_full
    );
    if rank_kept == rank_full {
        return 1.0;
    }
    let m = rank_kept.max(2) as f64;
    let p = rank_full.max(2) as f64;
    (m.ln() / p.ln()) as f32
}

/// Plan 622 T3.1 — the rank-ratio cooling `s_eff = 1 + γ·(r − 1)` (the
/// paper's per-query temperature, the code form the research quotes). γ = 1
/// is the lane's fixed value (Plan 622 T3.1); any γ is accepted here and
/// the K = full posture returns exactly 1.0 FOR ANY γ (the identity law —
/// pinned by the unit test). Consumers: the `> L` decode gate's per-key
/// scale window (T2.3's `PrunedDecodeGate::scale`), keyed by the KEPT rank
/// of each packed slot.
#[cfg(feature = "tokengate")]
#[inline]
pub fn rank_cooling(rank_kept: usize, rank_full: usize, gamma: f32) -> f32 {
    1.0 + gamma * (rank_ratio(rank_kept, rank_full) - 1.0)
}

/// Plan 622 T3.1 — sequence-RMSNorm scorer normalization, fused with the
/// readout dot: `scores[i] = readout · (h_i / rms(h_i))`, one pure pass
/// over the layer-L residual rows `[rows, n]`. Per-TOKEN row norm (the
/// codebase's own `rmsnorm` reading — each token scale-invariant, so the
/// readout dot is a pure direction measure comparable across the sequence);
/// gamma-free (the modelless overlay carries no trained gains — a readout
/// row applies its own scale by construction). The alternative reading
/// (one global statistic over the whole [rows × n] buffer) is a named
/// choice for T3.4's ablation rows, not a second fn. Zero rows are safe
/// (inv_rms = 1/√eps, score 0 — pinned). f64 statistics, f32 readout;
/// the dot folds sequentially (the Issue-940 determinism law).
/// Zero-alloc (`*_into`); `dst` is `rows`, `readout` is `n`.
#[cfg(feature = "tokengate")]
pub fn seq_rmsnorm_readout_into(
    dst: &mut [f32],
    hidden: &[f32],
    readout: &[f32],
    n: usize,
    eps: f64,
) {
    let rows = dst.len();
    assert_eq!(hidden.len(), rows * n, "hidden rows mismatch");
    assert_eq!(readout.len(), n, "readout width mismatch");
    for i in 0..rows {
        let row = &hidden[i * n..(i + 1) * n];
        let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv_rms = 1.0 / (sum_sq / n as f64 + eps).sqrt();
        let mut acc = 0.0f32;
        for (v, w) in row.iter().zip(readout.iter()) {
            acc += ((*v as f64 * inv_rms) as f32) * w;
        }
        dst[i] = acc;
    }
}

/// Plan 622 T3.2 — the QK-norm-derived magnitude table, derived once from the
/// loaded weights: `s_l = ‖g_q ∘ g_k‖₂ / √d` per layer (elementwise gain
/// product, L2 norm, `/√head_dim` — the isotropic-model logit std; upstream
/// measured 2.05–3.60 across Qwen3-8B's layers, the same range a LEARNED
/// gate_scale converged to — TokenGate `e2e_gate_module.py::set_qk_norm_scales`
/// @ `6e51aac5`). The table feeds the gate overlay's per-key BIAS rows (the
/// upstream law `b_j[l,h] = magnitude[l,h] · RMSNorm_seq(score)_j` — the bias
/// starts at about one native logit-sigma), NOT the query scale (that is
/// T3.1's [`rank_cooling`] surface).
///
/// PER-LAYER DEGENERACY (pinned, both sides): the plan's formula is per-head
/// `s_{l,h}`, but the qk-norm gains are `[head_dim]` SHARED across heads in
/// this codebase (`DeltaNetLayerWeights::attn_q_norm`) — and upstream's are
/// per-layer too ("Broadcast over heads (qk_norm gains are per-layer)") — so
/// the table is one scalar per layer and the CONSUMER broadcasts it to
/// `n_head` when filling per-(layer, head) bias windows. There is no per-head
/// information to recover here.
///
/// LOAD-TIME discipline: derive ONCE at boot (mirrors [`runtime_enabled`] —
/// call once, never per tick). GDN layers carry EMPTY gains (the weights
/// contract: unused fields are empty `Vec`); their table slots are `0.0` —
/// inert by multiplication (a misread zeroes the bias rows, a no-op, never
/// garbage) and dead by the consumer contract anyway (the gate exists on
/// full-attention layers above L only; GDN layers read neither gate surface).
/// A half-populated gain pair is a loader bug — refused loud.
///
/// NO softplus inverse: upstream stores `softplus⁻¹(s_l)` in a learnable
/// parameter; a deterministic construction has no parameter, so the magnitude
/// IS `s_l`. Statistics in f64, output f32, sequential fold (Issue-940).
/// Zero-alloc (`*_into`); `dst` is `[n_layers]` aligned by layer index.
#[cfg(feature = "tokengate")]
pub fn derived_magnitudes_into(
    dst: &mut [f32],
    weights: &crate::deltanet::weights::QwenDeltaNetWeights,
) {
    assert_eq!(
        dst.len(),
        weights.layers.len(),
        "magnitude table must be [n_layers], aligned by layer index"
    );
    for (l, (dst_l, layer)) in dst.iter_mut().zip(weights.layers.iter()).enumerate() {
        let (gq, gk) = (&layer.attn_q_norm, &layer.attn_k_norm);
        match (gq.is_empty(), gk.is_empty()) {
            (true, true) => *dst_l = 0.0,
            (true, false) | (false, true) => panic!(
                "layer {l}: half-populated qk-norm gains (q {} / k {}) — loader bug",
                gq.len(),
                gk.len()
            ),
            (false, false) => {
                assert_eq!(
                    gq.len(),
                    gk.len(),
                    "layer {l}: qk-norm gain width mismatch (q {} / k {})",
                    gq.len(),
                    gk.len()
                );
                let d = gq.len() as f64;
                let sum_sq: f64 = gq
                    .iter()
                    .zip(gk.iter())
                    .map(|(q, k)| {
                        let p = (*q as f64) * (*k as f64);
                        p * p
                    })
                    .sum();
                *dst_l = (sum_sq / d).sqrt() as f32;
            }
        }
    }
}

/// Plan 622 T3.3 — the scorer family v0 (the deterministic gate overlay's
/// selection surface). One score per prompt token, consumed by
/// [`select_kept_into`] at layer L and by [`build_decode_gate_into`] at
/// prefill end. Default arm = (a); (b)/(c) are ablation rows.
#[cfg(feature = "tokengate")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScorerArm {
    /// (a) attention-mass received at layer L — the T3.3 default. The scores
    /// are the per-key causal attention mass the layer-L pass computes
    /// (upstream's gate substitutes a trained MLP for this statistic).
    #[default]
    AttentionMass,
    /// (b) hidden-state norm at L (ablation row): the residual row RMS —
    /// works at ANY mixer type (a GDN layer at L has no attention mass).
    HiddenNorm,
    /// (c) all-ones (ablation row): no token is distinguished — selection
    /// degenerates to first-index keeps, the no-information floor.
    AllOnes,
}

/// Plan 622 T3.3 arm (a): copy the layer-L mass tap into the scores. Refuses
/// loud when the tap was not armed (the caller passed no `mass_scratch`).
#[cfg(feature = "tokengate")]
pub fn scorer_attention_mass(_hidden: &[f32], mass: Option<&[f32]>, scores: &mut [f32]) {
    let mass = mass.expect(
        "arm (a) attention-mass requires the layer-L tap (pass mass_scratch to the pruned prefill)",
    );
    assert_eq!(mass.len(), scores.len(), "mass tap width mismatch");
    scores.copy_from_slice(mass);
}

/// Plan 622 T3.3 arm (b): per-token residual RMS at L — `scores[i] =
/// √(mean(row_i²))`, f64 statistics, f32 out (Issue-940). The mixer-agnostic
/// arm: a GDN layer at L scores through this (its hidden rows exist; its
/// attention mass does not).
#[cfg(feature = "tokengate")]
pub fn scorer_hidden_norm(hidden: &[f32], _mass: Option<&[f32]>, scores: &mut [f32]) {
    let rows = scores.len();
    assert!(rows > 0, "no tokens to score");
    let n = hidden.len() / rows;
    assert!(n > 0 && hidden.len() == rows * n, "hidden rows mismatch");
    for (i, dst) in scores.iter_mut().enumerate() {
        let row = &hidden[i * n..(i + 1) * n];
        let sum_sq: f64 = row
            .iter()
            .map(|v| {
                let v = *v as f64;
                v * v
            })
            .sum();
        *dst = (sum_sq / n as f64).sqrt() as f32;
    }
}

/// Plan 622 T3.3 arm (c): every token scores 1.0 — selection keeps the
/// first-index prunable tokens, the no-information ablation floor.
#[cfg(feature = "tokengate")]
pub fn scorer_all_ones(_hidden: &[f32], _mass: Option<&[f32]>, scores: &mut [f32]) {
    scores.fill(1.0);
}

/// Plan 622 T3.3 — the arm dispatch (the harness and boot pick an arm; the
/// pruned prefill consumes the plain closure seam).
#[cfg(feature = "tokengate")]
pub fn run_scorer(arm: ScorerArm, hidden: &[f32], mass: Option<&[f32]>, scores: &mut [f32]) {
    match arm {
        ScorerArm::AttentionMass => scorer_attention_mass(hidden, mass, scores),
        ScorerArm::HiddenNorm => scorer_hidden_norm(hidden, mass, scores),
        ScorerArm::AllOnes => scorer_all_ones(hidden, mass, scores),
    }
}

/// Plan 622 T3.4 — the random-K floor's score writer (pure fn): a SEEDED
/// uniform pick of `keep` prunable tokens, written as 1.0 / 0.0 scores so the
/// incumbent [`select_kept_into`] machinery (scaffold + top-K, first-index
/// ties) selects exactly that subset — the random arm rides the SAME seam as
/// the derived arms, no second selection path. Splitsmix64 (the
/// `fa_posterior` sampler, no external dep) drives a partial Fisher–Yates
/// over the prunable index range `(n_sink .. t-1)` — deterministic per seed,
/// distinct picks, scaffold positions and the final token NEVER selected
/// (the scorer cannot reach them; selection keeps them structurally).
/// `keep >=` prunable count → all-ones (≡ K = full). One allocation for the
/// index shuffle — the [`select_kept_into`] CPU-reference class.
#[cfg(feature = "tokengate")]
pub fn random_scores_into(scores: &mut [f32], n_sink: usize, keep: usize, seed: u64) {
    let t = scores.len();
    scores.fill(0.0);
    let prunable = n_sink..t.saturating_sub(1);
    let prunable_len = prunable.len();
    if keep == 0 || prunable_len == 0 {
        return;
    }
    let mut sm = crate::fa_posterior::SplitMix64::new(seed);
    let mut idx: Vec<usize> = prunable.collect();
    let take = keep.min(prunable_len);
    // Partial Fisher–Yates: position i draws uniformly from [i, len).
    for i in 0..take {
        let j = i + (sm.next_u64() % (idx.len() - i) as u64) as usize;
        idx.swap(i, j);
        scores[idx[i]] = 1.0;
    }
}

/// Plan 622 T3.3 — build the T2.3 decode gate's tables at prefill end, once:
/// per-(layer, head) bias rows `b[s] = magnitude[l] · seqRMSNorm(score)[kept[s]]`
/// over the PACKED kept slots (upstream law: the magnitude scales the per-key
/// BIAS — "the bias starts at about one native logit-sigma" — and the score
/// is sequence-RMS-normalized per slot, `Qwen3RMSNorm` mechanics), appended
/// slots ZERO (decode-appended keys are always attended), scaffold slots ZERO
/// (upstream forces the gate value 0 on non-prunable tokens — sinks + the
/// final token are kept regardless and are never steering targets), and the
/// scale table all `1.0` (the v0 decode posture carries the folded MASK only —
/// the rank cooling is a pruned-PREFILL surface; the ≡1 whole-table check then
/// skips the decode q-multiply entirely).
///
/// Layers that never read their windows are left untouched (caller-zeroed):
/// layers `<= prune_at_layer` (the gate exists above L only) and GDN layers
/// (they consume neither gate surface — the T2.3 contract). Only full-attn
/// layers `> L` get rows; each layer's own [`derived_magnitudes_into`] value
/// scales its rows (the per-layer degeneracy broadcast to `n_head` — head 0's
/// row is computed and copied).
///
/// The normalized score is order-preserving (positive scalar), so selection
/// by raw scores (the T2.1 seam) ≡ selection by any layer's bias row — the
/// upstream selection reads the first phase-2 layer's bias; both orders agree.
/// The pruned-PREFILL steering window (T3.4's bias arm) derives from the same
/// packed rows extended with [`PAD_KEY_BIAS`] at the pads (the T2.2 doc law);
/// this fn builds the DECODE gate. Zero-alloc; `eps` — upstream freezes
/// `_RMS_EPS = 1e-6` (Qwen3RMSNorm's variance placement).
#[cfg(feature = "tokengate")]
#[allow(clippy::too_many_arguments)] // the zero-alloc *_into window shape
pub fn build_decode_gate_into(
    bias_dst: &mut [f32],
    mut scale_dst: Option<&mut [f32]>,
    scores: &[f32],
    kept: &[usize],
    n_sink: usize,
    magnitudes: &[f32],
    layer_types: &[crate::types::DeltaNetLayerType],
    prune_at_layer: usize,
    n_head: usize,
    head_stride: usize,
    eps: f64,
) {
    use crate::types::DeltaNetLayerType;
    let n_layers = layer_types.len();
    assert!(!kept.is_empty(), "the kept list is never empty (scaffold)");
    assert!(n_head > 0, "no heads");
    assert!(
        head_stride >= kept.len(),
        "head_stride {} < kept rows {}",
        head_stride,
        kept.len()
    );
    assert_eq!(
        magnitudes.len(),
        n_layers,
        "magnitude table must be [n_layers]"
    );
    assert!(
        bias_dst.len() >= n_layers * n_head * head_stride,
        "bias table {} < {} windows",
        bias_dst.len(),
        n_layers * n_head * head_stride
    );
    if let Some(scale) = scale_dst.as_deref() {
        assert!(
            scale.len() >= n_layers * n_head,
            "scale table {} < {} entries",
            scale.len(),
            n_layers * n_head
        );
    }
    if let Some(&last) = kept.last() {
        assert!(last < scores.len(), "kept index {last} outside the scores");
    }

    // Sequence-RMS denominator over the FULL score vector (upstream
    // normalizes the gate output before forcing the scaffold zeros).
    let t = scores.len();
    let sum_sq: f64 = scores
        .iter()
        .map(|s| {
            let v = *s as f64;
            v * v
        })
        .sum();
    let inv_rms = 1.0 / (sum_sq / t.max(1) as f64 + eps).sqrt();

    let kept_len = kept.len();
    let last_slot = kept_len - 1;
    for (l, lt) in layer_types.iter().enumerate() {
        if let Some(scale) = scale_dst.as_deref_mut() {
            scale[l * n_head..(l + 1) * n_head].fill(1.0);
        }
        // Skip the never-read windows: at-or-below L, and GDN layers.
        if l <= prune_at_layer || *lt == DeltaNetLayerType::DeltaNet {
            continue;
        }
        let mag = magnitudes[l];
        let base = l * n_head * head_stride;
        {
            let row0 = &mut bias_dst[base..base + head_stride];
            for (s, &orig) in kept.iter().enumerate() {
                row0[s] = if orig < n_sink || s == last_slot {
                    0.0
                } else {
                    let normed = (scores[orig] as f64 * inv_rms) as f32;
                    mag * normed
                };
            }
            row0[kept_len..].fill(0.0);
        }
        for h in 1..n_head {
            bias_dst.copy_within(base..base + head_stride, base + h * head_stride);
        }
    }
}

#[cfg(test)]
mod tests {
    // Feature-off posture: the switch is the constant false, no env read.
    // (The feature-on arm's env behavior is a deployment switch, not a
    // unit-test surface — it is exercised by the walk harness boot at the
    // armed posture.)
    #[cfg(not(feature = "tokengate"))]
    #[test]
    fn feature_off_the_switch_is_constant_false() {
        assert!(!super::runtime_enabled());
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn feature_on_the_switch_is_a_bool() {
        // Value-agnostic at test time (the env may carry a demote literal on
        // a shared box): the only law unit-testable without env control is
        // that the read terminates and is stable across calls.
        let a = super::runtime_enabled();
        let b = super::runtime_enabled();
        assert_eq!(a, b);
    }

    // ---- select_kept_into (T2.1 selection laws) ------------------------

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_scaffold_survives_the_worst_scores() {
        // Scores poison the scaffold; it is kept anyway.
        let scores = vec![0.1, -1e9, -1e9, -1e9, -1e9, 0.5];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 4, 0, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 5]); // sinks + last
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_topk_over_prunable_first_index_on_ties() {
        // k=2 over prunable 4..5 (single token) → t=6, prunable = {4}.
        let scores = vec![0.0, 0.0, 0.0, 0.0, 9.0, 0.0];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 4, 1, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5]);

        // Ties: t=8, n_sink=2, k=2, all prunable equal → first-index wins
        // (2 and 3, never 6/7 despite identical scores).
        let scores = vec![0.0; 8];
        super::select_kept_into(&scores, 2, 2, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 7]);
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_k_at_least_prunable_keeps_everything() {
        let scores = vec![0.3, 0.1, 0.4, 0.1, 0.5, 0.9];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 4, usize::MAX, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5]);
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn select_nan_sorts_worst_and_tiny_prompts_dedup() {
        // NaN scores never beat a real score.
        let scores = vec![f32::NAN, f32::NAN, 0.2, f32::NAN, 0.1, f32::NAN];
        let mut out = Vec::new();
        super::select_kept_into(&scores, 0, 1, &mut out);
        assert_eq!(out, vec![2, 5]); // scaffold(last=5) + best real (2)

        // t <= n_sink: scaffold is everything, last dedups, no panic.
        let scores = vec![1.0, 2.0, 3.0];
        super::select_kept_into(&scores, 8, 5, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn rank_ratio_is_exactly_one_when_ranks_match_including_rank_one() {
        // The identity law, every posture the asymmetric-guard variant would
        // have broken: m == p → exactly 1.0, INCLUDING p = 1 (ln 1 = 0 —
        // the raw formula gives 0/ln2 = 0 there; the symmetric guard + the
        // short-circuit are what make the claim true).
        for p in [1usize, 2, 3, 10, 24, 1000] {
            assert_eq!(super::rank_ratio(p, p), 1.0, "m == p at p={p}");
        }
        // And the cooling is therefore an exact no-op at K = full for ANY γ.
        for gamma in [0.0f32, 1.0, 2.5, -1.0, 100.0] {
            assert_eq!(super::rank_cooling(7, 7, gamma), 1.0);
        }
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn rank_ratio_cools_monotonically_and_saturates_at_the_rank_two_floor() {
        // Deeper in the kept order → cooler (r strictly decreases in m).
        let r1 = super::rank_ratio(2, 10);
        let r2 = super::rank_ratio(5, 10);
        let r3 = super::rank_ratio(9, 10);
        assert!(r1 < r2 && r2 < r3 && r3 < 1.0);
        // The rank-2 floor: m = 1 floors to 2 (ln 2 / ln 10), never 0.
        let floor = super::rank_ratio(1, 10);
        assert_eq!(floor, r1, "m=1 and m=2 share the max(2, ·) floor");
        let expected = (2.0f64.ln() / 10.0f64.ln()) as f32;
        assert_eq!(floor, expected);
        // Cooling at γ = 1: s = 1 + (r − 1) by the definition.
        let s = super::rank_cooling(2, 10, 1.0);
        assert_eq!(s, 1.0 + 1.0 * (r1 - 1.0));
        assert!(s < 1.0, "a pruned rank cools below 1");
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn seq_rmsnorm_readout_known_answer_and_zero_row_safety() {
        let eps = 1e-6f64;
        // Hand-computable: row [3, 4], rms = √((9+16)/2) = √12.5, readout
        // [1, 0] → score = 3/√12.5.
        let hidden = [3.0f32, 4.0];
        let readout = [1.0f32, 0.0];
        let mut scores = [0.0f32; 1];
        super::seq_rmsnorm_readout_into(&mut scores, &hidden, &readout, 2, eps);
        let expected = (3.0f64 / 12.5f64.sqrt()) as f32;
        assert!(
            (scores[0] - expected).abs() < 1e-6,
            "{} vs {expected}",
            scores[0]
        );

        // Row independence: a second row does not move the first (and the
        // second row's own score is its direction measure: [100, -100] has
        // rms 100 → [1, -1], readout [1, 0] → 1.0).
        let two_rows = [3.0, 4.0, 100.0, -100.0];
        let mut two_scores = [0.0f32; 2];
        super::seq_rmsnorm_readout_into(&mut two_scores, &two_rows, &readout, 2, eps);
        assert_eq!(two_scores[0].to_bits(), scores[0].to_bits());
        assert!((two_scores[1] - 1.0).abs() < 1e-6);

        // Zero row: inv_rms = 1/√eps, score 0, finite — no NaN.
        let zeros = [0.0f32; 4];
        super::seq_rmsnorm_readout_into(&mut two_scores, &zeros, &readout, 2, eps);
        assert_eq!(two_scores[0], 0.0);
        assert!(two_scores[0].is_finite());
    }

    // ---- derived_magnitudes_into (T3.2 qk-norm magnitude table) ---------

    /// A minimal weights shell: only `layers` matters to the derivation;
    /// everything else is empty by the weights contract (no Default derive on
    /// the weights structs — explicit empties).
    #[cfg(feature = "tokengate")]
    fn shell_layers(
        layers: Vec<crate::deltanet::weights::DeltaNetLayerWeights>,
    ) -> crate::deltanet::weights::QwenDeltaNetWeights {
        crate::deltanet::weights::QwenDeltaNetWeights {
            wte: Vec::new(),
            final_norm: Vec::new(),
            lm_head: crate::deltanet::weights::Proj::empty(),
            layers,
            layer_types: Vec::new(),
        }
    }

    #[cfg(feature = "tokengate")]
    fn attn_layer(gq: Vec<f32>, gk: Vec<f32>) -> crate::deltanet::weights::DeltaNetLayerWeights {
        use crate::deltanet::weights::Proj;
        crate::deltanet::weights::DeltaNetLayerWeights {
            attn_wq: Proj::empty(),
            attn_wk: Proj::empty(),
            attn_wv: Proj::empty(),
            attn_wo: Proj::empty(),
            attn_q_norm: gq,
            attn_k_norm: gk,
            in_proj_qkv: Proj::empty(),
            in_proj_a: Proj::empty(),
            in_proj_b: Proj::empty(),
            in_proj_z: Proj::empty(),
            out_proj: Proj::empty(),
            conv1d_weight: Vec::new(),
            a_log: Vec::new(),
            dt_bias: Vec::new(),
            linear_norm: Vec::new(),
            gate_proj: Proj::empty(),
            up_proj: Proj::empty(),
            down_proj: Proj::empty(),
            input_norm: Vec::new(),
            post_attn_norm: Vec::new(),
        }
    }

    #[cfg(feature = "tokengate")]
    fn gdn_layer() -> crate::deltanet::weights::DeltaNetLayerWeights {
        attn_layer(Vec::new(), Vec::new())
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn derived_magnitudes_known_answer_all_ones_and_hand_case() {
        // All-ones gains, d = 4: products all 1, sum_sq 4, √(4/4) = exactly 1.
        let mut w = shell_layers(vec![attn_layer(vec![1.0; 4], vec![1.0; 4])]);
        let mut table = [7.5f32; 1];
        super::derived_magnitudes_into(&mut table, &w);
        assert_eq!(table[0], 1.0, "all-ones gains derive exactly one");

        // Hand case: gq [1,2,3,4], gk all 2 → products [2,4,6,8],
        // sum_sq = 4+16+36+64 = 120, √(120/4) = √30.
        w.layers[0] = attn_layer(vec![1.0, 2.0, 3.0, 4.0], vec![2.0; 4]);
        super::derived_magnitudes_into(&mut table, &w);
        let expected = (120.0f64 / 4.0).sqrt() as f32;
        assert_eq!(table[0], expected);
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn derived_magnitudes_gdn_layers_are_inert_zero_and_half_gains_refuse() {
        // Hybrid: GDN (empty gains → 0.0), attention (derived), GDN again.
        let w = shell_layers(vec![
            gdn_layer(),
            attn_layer(vec![1.0, 2.0], vec![3.0, 4.0]),
            gdn_layer(),
        ]);
        let mut table = [0.0f32; 3];
        super::derived_magnitudes_into(&mut table, &w);
        assert_eq!(table[0], 0.0, "GDN slot inert-zero");
        assert_eq!(table[2], 0.0, "GDN slot inert-zero");
        // Layer 1: products [3, 8], sum_sq 9+64=73, √(73/2).
        assert_eq!(table[1], (73.0f64 / 2.0).sqrt() as f32);

        // Half-populated gains are a loader bug — refused loud, never a
        // silent 0.0 (which would inert a real layer's bias rows).
        let bad = shell_layers(vec![attn_layer(vec![1.0, 2.0], Vec::new())]);
        let mut t = [0.0f32; 1];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::derived_magnitudes_into(&mut t, &bad);
        }));
        assert!(r.is_err(), "half-populated gains must refuse");

        // Table length is the layer count — mis-sized dst refuses loud.
        let mut short = [0.0f32; 2];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::derived_magnitudes_into(&mut short, &w);
        }));
        assert!(r.is_err(), "dst length must equal n_layers");
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn derived_magnitudes_sign_is_squared_away_and_l2_upweights_the_product() {
        // Negative gains: the product's sign squares away — gq [1,-1],
        // gk [1,1] → products [1,-1] → √((1+1)/2) = 1.
        let mut w = shell_layers(vec![attn_layer(vec![1.0, -1.0], vec![1.0, 1.0])]);
        let mut table = [0.0f32; 1];
        super::derived_magnitudes_into(&mut table, &w);
        assert_eq!(table[0], 1.0);

        // The L2 norm reads the SQUARE law: doubling a peaky gain coordinate
        // quadruples its contribution — [2,0] products weigh 4, [√2,√2]
        // products weigh 2+2 = 4 (equal here), but [4,0] beats [2,2]
        // (16 vs 8): peaky gain coordinates upweight, never average away.
        w.layers[0] = attn_layer(vec![4.0, 0.0], vec![1.0, 1.0]);
        super::derived_magnitudes_into(&mut table, &w);
        let peaky = table[0];
        w.layers[0] = attn_layer(vec![2.0, 2.0], vec![1.0, 1.0]);
        super::derived_magnitudes_into(&mut table, &w);
        let flat = table[0];
        assert!(peaky > flat, "peaky {peaky} must exceed flat {flat}");
        assert_eq!(
            peaky,
            (16.0f64 / 2.0).sqrt() as f32,
            "peaky derives √(sum_sq/d) = √(16/2)"
        );
    }

    // ---- build_decode_gate_into (T3.3 gate construction) ----------------

    #[cfg(feature = "tokengate")]
    #[test]
    fn decode_gate_rows_carry_magnitude_times_normed_score_with_scaffold_and_appended_zeros() {
        use crate::types::DeltaNetLayerType;
        // Hybrid [GDN, Attn, Attn], L = 0: layers 1 and 2 steer.
        let layer_types = vec![
            DeltaNetLayerType::DeltaNet,
            DeltaNetLayerType::Attention,
            DeltaNetLayerType::Attention,
        ];
        let scores = [2.0f32, -1.0, 4.0, 0.5, 1.0, 3.0]; // T = 6
        let kept = [0usize, 2, 4, 5]; // packed slots; T-1 = 5 kept (last)
        let n_sink = 1usize; // scaffold: original position 0 + the last token
        let mags = [0.0f32, 2.0, 3.0];
        let (n_head, stride) = (2usize, 6usize);

        let mut bias = vec![7.5f32; 3 * n_head * stride]; // poisoned sentinel
        let mut scale = vec![0.0f32; 3 * n_head];
        super::build_decode_gate_into(
            &mut bias,
            Some(&mut scale),
            &scores,
            &kept,
            n_sink,
            &mags,
            &layer_types,
            0,
            n_head,
            stride,
            1e-6,
        );

        // seq-RMS over ALL six scores: mean sq = (4+1+16+0.25+1+9)/6 = 31.25/6.
        let inv = 1.0 / (31.25f64 / 6.0 + 1e-6).sqrt();
        let normed = |j: usize| (scores[j] as f64 * inv) as f32;
        // Scaffold packed slots: 0 (original < n_sink) and 3 (the last token).
        let expected_non_scaffold = [2.0 * normed(2), 2.0 * normed(4)];

        // Layer 0 (at L) untouched: sentinel stands (never read).
        assert!(bias[..n_head * stride].iter().all(|&v| v == 7.5));
        // Layer 1: head 0 = head 1 (the degeneracy copy), slots [0]/[3] zero,
        // slots [1]/[2] = mag × normed, appended slots [4..6] zero.
        for h in 0..n_head {
            let row = &bias[(n_head + h) * stride..][..stride];
            assert_eq!(row[0], 0.0, "scaffold sink slot");
            assert_eq!(row[1], expected_non_scaffold[0]);
            assert_eq!(row[2], expected_non_scaffold[1]);
            assert_eq!(row[3], 0.0, "scaffold last-token slot");
            assert_eq!(row[4..], [0.0; 2], "appended keys zero");
        }
        // Layer 2: same shape × its own magnitude (the per-layer law).
        for h in 0..n_head {
            let row = &bias[(2 * n_head + h) * stride..][..stride];
            assert_eq!(row[1], 3.0 * normed(2));
            assert_eq!(row[2], 3.0 * normed(4));
        }
        // Scale table: 1.0 across EVERY layer (the ≡1 skip law) — including
        // layers that never steer.
        assert!(scale.iter().all(|&v| v == 1.0));
    }

    #[cfg(feature = "tokengate")]
    #[test]
    fn decode_gate_skips_gdn_and_below_l_windows_and_refuses_bad_shapes() {
        use crate::types::DeltaNetLayerType;
        let layer_types = vec![
            DeltaNetLayerType::Attention,
            DeltaNetLayerType::DeltaNet,
            DeltaNetLayerType::Attention,
        ];
        let scores = [1.0f32, 2.0, -1.0, 0.5, 3.0];
        let kept = [0usize, 2, 4]; // slot 1 is non-scaffold (original 2 ≥ n_sink... n_sink=1)
        let mags = [5.0f32; 3];
        let mut bias = vec![0.0f32; 3 * 2 * 5];
        let stride = 5usize;
        // L = 1: layer 0 is BELOW L (skip), layer 1 is GDN (skip), layer 2
        // steers — the only window written.
        super::build_decode_gate_into(
            &mut bias,
            None,
            &scores,
            &kept,
            1,
            &mags,
            &layer_types,
            1,
            2,
            stride,
            1e-6,
        );
        assert!(bias[..2 * stride].iter().all(|&v| v == 0.0), "below-L skip");
        assert!(
            bias[2 * stride..2 * 2 * stride].iter().all(|&v| v == 0.0),
            "GDN skip"
        );
        // Layer 2 written: scaffold slots 0 (original < n_sink) and 2 (last)
        // zero; the non-scaffold slot 1 carries mag × normed(2) ≠ 0.
        // sum_sq = 1+4+1+0.25+9 = 15.25 over T = 5.
        let inv = 1.0 / (15.25f64 / 5.0 + 1e-6).sqrt();
        let want = 5.0 * ((scores[2] as f64 * inv) as f32);
        let row = &bias[2 * 2 * stride..3 * 2 * stride];
        assert_eq!(row[1], want, "the steering window carries the bias");
        assert_eq!(row[0], 0.0);
        assert_eq!(row[2], 0.0);
        assert!(row[stride..2 * stride] == row[..stride], "head copy");

        // Refusals: short stride, short bias table.
        let mut small = vec![0.0f32; 3 * 2 * 5];
        for mut bad in [
            // (bias, stride): stride < kept_len
            (vec![0.0; 8], 2usize),
            // bias table too short
            (vec![0.0; 1], 5usize),
        ] {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::build_decode_gate_into(
                    &mut bad.0,
                    None,
                    &scores,
                    &kept,
                    4,
                    &mags,
                    &layer_types,
                    1,
                    2,
                    bad.1,
                    1e-6,
                );
            }));
            assert!(r.is_err(), "shape refusal must fire (stride={})", bad.1);
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let empty_kept: Vec<usize> = Vec::new();
            super::build_decode_gate_into(
                &mut small,
                None,
                &scores,
                &empty_kept,
                4,
                &mags,
                &layer_types,
                1,
                2,
                4,
                1e-6,
            );
        }));
        assert!(r.is_err(), "empty kept refuses");
    }

    // ---- random_scores_into (T3.4 random-K floor) -------------------------

    #[cfg(feature = "tokengate")]
    #[test]
    fn random_scores_are_deterministic_distinct_and_scaffold_free() {
        let t = 64usize;
        let n_sink = 4usize;
        let keep = 20usize;
        let mut a = vec![0.0f32; t];
        let mut b = vec![0.0f32; t];
        super::random_scores_into(&mut a, n_sink, keep, 0x6220_0044);
        super::random_scores_into(&mut b, n_sink, keep, 0x6220_0044);
        assert_eq!(a, b, "same seed → same bits");

        let picked: Vec<usize> = (0..t).filter(|&i| a[i] == 1.0).collect();
        assert_eq!(picked.len(), keep, "exactly `keep` picks");
        assert!(
            picked.iter().all(|&i| i >= n_sink && i < t - 1),
            "scaffold + last never picked"
        );
        assert!(a.iter().all(|&v| v == 0.0 || v == 1.0), "binary scores");

        // A different seed picks a different subset (overwhelmingly — 20 of
        // 59 identical picks by chance is ~1e-18; assert at least one differs).
        super::random_scores_into(&mut b, n_sink, keep, 0x6220_0045);
        let picked_b: Vec<usize> = (0..t).filter(|&i| b[i] == 1.0).collect();
        assert_ne!(picked, picked_b, "different seeds → different picks");

        // keep >= prunable → everything prunable is 1.0 (≡ K = full).
        super::random_scores_into(&mut a, n_sink, usize::MAX, 7);
        let prunable_ones = (n_sink..t - 1).filter(|&i| a[i] == 1.0).count();
        assert_eq!(prunable_ones, t - n_sink - 1, "K = full posture all-ones");
        assert_eq!(a[0], 0.0, "scaffold untouched by the writer");

        // keep = 0 → all zeros (scaffold-only posture).
        super::random_scores_into(&mut a, n_sink, 0, 7);
        assert!(a.iter().all(|&v| v == 0.0));
    }
}
