//! eDLM (Drex DLM) segment-structured scoring lane — riir-infer Issue 1005.
//!
//! Loads and serves `nace-ai/drex-dlm` (Efficient-DLM-8B backbone, GGUF arch
//! key `edlm`): a Qwen3-family transformer scored through a segment-based
//! block-causal mask instead of plain causality. One forward pass produces
//! hidden states; a pointer head reads per-option probabilities off the
//! `<decide>`/`</opt>` marker positions.
//!
//! Clean-room from the wire + published mechanics (their code is MIT —
//! reference reading only; weights CC BY-NC 4.0 — local bench/comparison
//! serving only, never a product lane, never a distill teacher).
//!
//! Phase 1 (this module):
//! - T1 `load_edlm_weights_gguf` — `edlm` GGUF arch support (Qwen3 tensor
//!   naming, `attn_{q,k}_norm` QK-RMSNorm carried, `pointer.*` head tensors,
//!   NEOX RoPE — NO Q/K row permutation, unlike the `llama` arch loader).
//! - T2 `branch_mask` — segment mask: `attend(i,j) iff j<=i and
//!   (seg[j]==0 or seg[j]==seg[i])`, plus `state_bidir` (state↔state both
//!   directions) and the option-isolation variant; pads (`seg < 0`) are keys
//!   to nobody and produce garbage queries (never read). Generalizes
//!   `attention::block_causal_t_n` from a prefix length to arbitrary
//!   per-(query, key) eligibility.
//! - T3 `rows_of` + `forward_edlm_rows` — row-form fallback: every question
//!   runs as a self-contained causal row `state + branch` (same positions as
//!   the packed form), the mask-parity escape for over-long packed sequences.
//!   Parity with the packed form holds by construction (the KV set of a
//!   branch query is identical in both forms — state never sees branches) and
//!   is pinned by a test on both `state_bidir` postures.
//!
//! Phase 2 (this module too):
//! - T4 marker encode/pack — `EDLM_MARKERS` (the five reused Qwen special
//!   tokens), `defuse_special_spellings` (the `<|name|>` → `<¦name¦>`
//!   anti-forgery rewrite), `encode_packed` (per-question branches,
//!   strict over-context refusal, the `option_isolation` position law).
//! - T5 pointer head — `EdlmPointerHead` (Phase 1) + the GGUF-tensor load
//!   path; the `head.pt` torch-pickle reader is deferred (the serving
//!   artifact is the GGUF; the conversion embeds the same head).
//! - T6 end-to-end parity — `EdlmGgufModel` (quant-resident mmap model,
//!   one-layer-at-a-time dequant scratch: O(file) RAM) + the env-gated
//!   sample-request test against their published outputs.
//!
//! Phase 3 (this module, T7): state-prefix KV reuse — the state segment's
//! per-layer K/V computed once ([`EdlmStateKv`]) and every question's branch
//! run as a continuation attending the cached prefix, exact by construction
//! (state never sees branches). T8 (GPU forward) is a separate unit.
//!
//! Reference distillation: the subagent note on `nace-ai/drex-dlm` @ `6c63df2`
//! (`code/kev/model.py::branch_mask_batch` + `rows_of` + `PointerHead`) and
//! the `nace-ai/llama.cpp` branch `edlm` (`src/models/edlm.cpp` — the C++
//! mask predicate is the same law; `pointer.temperature.weight` is the GGUF
//! tensor name, NE 1). Every name/key below is header-verified against the
//! real `drex-dlm-Q8_0.gguf` (2026-10-08).

use crate::llama_layer::LlamaLayerWeights;
use crate::rope::RopeFreqTable;
use crate::tokenizer::BpeTokenizer;
use anyhow::{Result, bail};

/// Segment id: option-isolation "no option" (state / instruction token).
pub const OPT_NONE: i32 = -1;
/// Segment id: option-isolation `<decide>` marker (sees every span).
pub const OPT_DECIDE: i32 = -2;

// ── T2 — the branch mask ────────────────────────────────────────────

/// Build the eDLM branch attention mask.
///
/// `seg[t]`: `0` = state, `k ≥ 1` = question-`k` branch, `< 0` = pad.
/// Writes row-major `out[i * l + j] = true` iff query `i` may attend key `j`:
///
/// - causal: `j <= i`
/// - keys are the state (`seg[j] == 0`) or the query's own segment
/// - `state_bidir`: state↔state attends both directions (the released
///   posture; the state prefix KV cache stays exact because state still never
///   sees branches)
/// - `opts` (option isolation, same length as `seg`): a key inside an option
///   span (`opt >= 0`) is visible only to the decide marker
///   (`opt == OPT_DECIDE`) or the same span — spans never see each other
/// - pads (`seg < 0`) are visible to nobody; every row keeps its diagonal
///   (softmax never runs on an all-blocked row; pad rows produce garbage that
///   is never read)
///
/// This is the additive-mask law their Python (`branch_mask_batch`,
/// blocked = `finfo(dtype).min`, allowed = 0) and their C++ server
/// (`-INFINITY`, allowed = 0) both realize; the bool form feeds
/// `attention_head_masked` which skips ineligible keys outright.
pub fn branch_mask(seg: &[i32], state_bidir: bool, opts: Option<&[i32]>, out: &mut [bool]) {
    let l = seg.len();
    assert_eq!(
        out.len(),
        l * l,
        "branch_mask out buffer must be l*l ({l}*{l}), got {}",
        out.len()
    );
    if let Some(o) = opts {
        assert_eq!(
            o.len(),
            l,
            "opts must have the same length as seg ({} vs {l})",
            o.len()
        );
    }

    for i in 0..l {
        for j in 0..l {
            let allow = if i == j {
                // Diagonal always — every row keeps a finite softmax.
                true
            } else {
                let (gi, gj) = (seg[i], seg[j]);
                let real = gi >= 0 && gj >= 0;
                let causal = j <= i;
                let key_is_state = gj == 0;
                let same_seg = gj == gi;
                // Causal base, then the state_bidir disjunct OR-ed in
                // independently of causality (state↔state is bidirectional),
                // then the option-isolation conjunct — the reference order
                // (`branch_mask_batch`: base → state_bidir OR → opts AND).
                let base = real && causal && (key_is_state || same_seg);
                let bidir = state_bidir && real && gi == 0 && gj == 0;
                let mut allow = base || bidir;
                if let Some(o) = opts
                    && allow
                {
                    let (oi, oj) = (o[i], o[j]);
                    let key_is_option = oj >= 0;
                    let query_is_decide = oi == OPT_DECIDE;
                    let same_option = oj == oi;
                    allow = !key_is_option || query_is_decide || same_option;
                }
                allow
            };
            out[i * l + j] = allow;
        }
    }
}

/// Build the row-form mask for one `state + branch` causal row
/// (`rows_of` output): causal triangle + the full state block bidirectional
/// when `state_bidir` (state keys are fully visible either way for branch
/// queries — they precede the branch; the flag only widens state↔state).
///
/// No pads: rows are exact slices, so the mask is exact too.
pub fn row_branch_mask(state_len: usize, branch_len: usize, state_bidir: bool) -> Vec<bool> {
    let l = state_len + branch_len;
    let mut out = vec![false; l * l];
    for i in 0..l {
        for j in 0..l {
            let causal = j <= i;
            let state_pair = state_bidir && i < state_len && j < state_len;
            out[i * l + j] = causal || state_pair;
        }
    }
    out
}

// ── T3 — the row form ───────────────────────────────────────────────

/// A packed eDLM encoding: token ids, per-token position ids (branch positions
/// RESTART at `state_len` — RoPE phases are branch-local, duplicated across
/// questions by design), segment ids, the marker readout indices, and the
/// option-isolation per-token option index (`opt`, empty = no isolation).
#[derive(Debug, Clone, Default)]
pub struct PackedEncoding {
    pub ids: Vec<usize>,
    pub pos: Vec<usize>,
    pub seg: Vec<i32>,
    /// Per-token option index within its question (`OPT_NONE` for
    /// state/instruction tokens, `0..K-1` for option spans, `OPT_DECIDE` for
    /// the decide marker). Empty when the record was encoded WITHOUT option
    /// isolation (the common posture) — `branch_mask` then takes `opts=None`.
    pub opt: Vec<i32>,
    /// Token count of the state segment (`seg[..state_len]` all `0`).
    pub state_len: usize,
    /// True when a non-strict encode CROPPED the state to `max_state` — the
    /// disclosure that a silent crop happened (strict encodes refuse instead).
    pub state_truncated: bool,
    /// Absolute index of each question's decide marker (its branch's last token).
    pub decide_idx: Vec<usize>,
    /// Absolute index of each question's `</opt>` key positions.
    pub opt_idx: Vec<Vec<usize>>,
}

impl PackedEncoding {
    /// Assert the structural invariants the mask/row machinery relies on.
    pub fn validate(&self) -> Result<()> {
        let l = self.ids.len();
        if self.pos.len() != l || self.seg.len() != l {
            bail!(
                "ids/pos/seg length mismatch: {l}/{}/{}",
                self.pos.len(),
                self.seg.len()
            );
        }
        if !self.opt.is_empty() && self.opt.len() != l {
            bail!(
                "opt length {} does not match ids {l} (empty = no isolation)",
                self.opt.len()
            );
        }
        if self.state_len == 0 || self.state_len > l {
            bail!("state_len {} out of range for l {l}", self.state_len);
        }
        if self.seg[..self.state_len].iter().any(|&g| g != 0) {
            bail!("seg[..state_len] is not all zeros");
        }
        Ok(())
    }
}

/// One question's branch as a self-contained row: `state ++ branch`.
///
/// `decide`/`opts` are offsets WITHIN the branch; add `state_len` for
/// row-relative readout positions. `start..end` index the packed encoding.
#[derive(Debug, Clone)]
pub struct BranchRow {
    pub start: usize,
    pub end: usize,
    pub decide: usize,
    pub opts: Vec<usize>,
}

impl BranchRow {
    /// Branch token count (excluding the state prefix).
    pub fn branch_len(&self) -> usize {
        self.end - self.start
    }
}

/// Split a packed encoding into per-question rows.
///
/// Row `k` = `state ++ branch_k`: the branch slice runs from the question's
/// first segment token to its decide marker (inclusive). Positions carry over
/// unchanged — the branch already restarts at `state_len` in the packed form,
/// so packed and row forms agree position-for-position and the KV set of a
/// branch query is identical in both (parity by construction; pinned by
/// `packed_rows_parity`).
pub fn rows_of(enc: &PackedEncoding) -> Result<Vec<BranchRow>> {
    enc.validate()?;
    let mut rows = Vec::with_capacity(enc.decide_idx.len());
    for (k, &decide_abs) in enc.decide_idx.iter().enumerate() {
        let start = if k == 0 {
            enc.state_len
        } else {
            enc.decide_idx[k - 1] + 1
        };
        let end = decide_abs + 1;
        if end > enc.seg.len() || start >= end {
            bail!("branch {k}: empty or out-of-range slice [{start}, {end})");
        }
        if enc.seg[start] != (k as i32) + 1 || enc.seg[end - 1] != (k as i32) + 1 {
            bail!(
                "branch {k}: layout mismatch — seg[start]={} seg[end-1]={}, expected {}",
                enc.seg[start],
                enc.seg[end - 1],
                (k as i32) + 1
            );
        }
        let opts = enc
            .opt_idx
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("branch {k}: missing opt_idx row"))?
            .iter()
            .map(|&o| {
                if o < start || o >= end {
                    bail!("branch {k}: opt index {o} outside [{start}, {end})");
                }
                Ok(o - start)
            })
            .collect::<Result<Vec<usize>>>()?;
        rows.push(BranchRow {
            start,
            end,
            decide: decide_abs - start,
            opts,
        });
    }
    Ok(rows)
}

// ── T4 — marker encode/pack ─────────────────────────────────────────

/// The five delimiter tokens, in packing order: state, question,
/// option-open, option-close, decide.
///
/// The release reuses existing rarely-used Qwen special tokens so no
/// embedding rows need to be added or trained (the adapter trained their
/// meaning). Resolved through the tokenizer vocab at encode time — never
/// hardcoded ids (a different conversion may renumber).
pub const EDLM_MARKERS: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];

/// Encoding limits (the training-context law): state tokens (including the
/// state marker), per-question branch tokens, and the whole packed record.
/// Frozen suites are admitted under the same rule, so eval sees the same
/// population training did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdlmLimits {
    pub max_state: usize,
    pub max_branch: usize,
    pub max_packed: usize,
}

impl Default for EdlmLimits {
    fn default() -> Self {
        Self {
            max_state: 384,
            max_branch: 1024,
            max_packed: 2048,
        }
    }
}

impl EdlmLimits {
    /// The serving posture: every cap at the context window (16,384
    /// recommended; the weights' position window is 32,768).
    pub fn serving(context: usize) -> Self {
        Self {
            max_state: context,
            max_branch: context,
            max_packed: context,
        }
    }
}

/// One question: instruction text + option texts, already rendered from the
/// caller's request shape (the JSON request mapping is the consumer's job —
/// this layer speaks text).
#[derive(Debug, Clone)]
pub struct EdlmQuestion {
    pub instr: String,
    pub options: Vec<String>,
}

/// One decision record: state text + ordered questions. Question order is
/// option order for the readout (`decide_idx`/`opt_idx` follow this vec).
#[derive(Debug, Clone, Default)]
pub struct EdlmRecord {
    pub state: String,
    pub questions: Vec<EdlmQuestion>,
}

/// A record does not encode within its limits. Serving maps this to a 422;
/// a benchmark counts it as a rejected record. Downcast through `anyhow` to
/// distinguish refusal from other failures.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextOverflow {
    pub detail: String,
}

impl std::fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "context overflow: {}", self.detail)
    }
}

impl std::error::Error for ContextOverflow {}

/// The encode-side tokenizer seam: plain-text tokenization for user content
/// plus exact vocab lookup for the delimiter tokens.
///
/// [`BpeTokenizer`] (the GGUF-embedded Qwen-family vocab) is the production
/// impl; tests plug a mock so the layout law is pinned without weights.
pub trait EdlmEncodeTokenizer {
    /// Tokenize caller text that contains no special-token spellings (the
    /// caller defused them first). No BOS/EOS, no special splitting.
    fn encode_text(&self, text: &str) -> Vec<usize>;
    /// Exact vocab lookup (`convert_tokens_to_ids` semantics).
    fn token_to_id(&self, token: &str) -> Option<usize>;
}

impl EdlmEncodeTokenizer for BpeTokenizer {
    fn encode_text(&self, text: &str) -> Vec<usize> {
        // Post-defuse text cannot contain a special spelling, so plain
        // BPE (no special matching) is both exact and faster.
        self.encode_no_special(text)
    }

    fn token_to_id(&self, token: &str) -> Option<usize> {
        BpeTokenizer::token_to_id(self, token)
    }
}

/// Rewrite `<|name|>` → `<¦name¦>` (U+00A6 broken bar) for every
/// `[A-Za-z0-9_]+` name, so user text can never produce delimiter or control
/// tokens — option boundaries are unforgeable.
///
/// The rewrite happens BEFORE tokenization (a fast tokenizer ignores
/// split-special flags, so the spellings must not survive into the BPE
/// input). A `<` that does not open a well-formed delimiter is copied
/// verbatim and the scan continues after it — the same positions the
/// reference regex would try next.
pub fn defuse_special_spellings(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            // Try `<|name|>` with name = [A-Za-z0-9_]+.
            let mut j = i + 2;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            if j > i + 2 && j + 1 < bytes.len() && bytes[j] == b'|' && bytes[j + 1] == b'>' {
                out.push_str("<\u{a6}");
                out.push_str(&text[i + 2..j]);
                out.push_str("\u{a6}>");
                i = j + 2;
                continue;
            }
        }
        // Not a delimiter start: copy one full char (a multi-byte char can
        // never contain an ASCII delimiter byte as a continuation byte).
        let len = text[i..].chars().next().map_or(1, char::len_utf8);
        out.push_str(&text[i..i + len]);
        i += len;
    }
    out
}

/// Tokenize caller-supplied text for packing: defuse delimiters, then
/// tokenize (no BOS/EOS, no special splitting — the defused text has no
/// special spellings left).
pub fn user_tokens<T: EdlmEncodeTokenizer>(tok: &T, text: &str) -> Vec<usize> {
    tok.encode_text(&defuse_special_spellings(text))
}

/// Pack one record: `[state-marker state…]` then per-question
/// `[q-marker instr opt-marker o /opt-marker… decide-marker]`.
///
/// Returns the [`PackedEncoding`]: ids, seg (`0` = state, `k ≥ 1` = question
/// k), pos (branch positions restart after the state — RoPE phases are
/// branch-local), decide/opt readout indices, and the per-token option index
/// when `option_isolation` is set.
///
/// Over-context law: `strict = true` refuses an over-long state
/// ([`ContextOverflow`]); `strict = false` crops it and sets
/// [`PackedEncoding::state_truncated`] — never a silent crop. An over-long
/// BRANCH refuses in both modes (a cropped branch loses options — a different
/// question). The packed-limit check is the caller's (`max_packed`; the row
/// form is the escape for over-long packed sequences).
///
/// `option_isolation = true`: every option span becomes its own sub-branch —
/// it sees state + instruction + itself only, all spans share the same
/// position ids, and `<decide>` sits at one fixed position after the longest
/// span — so per-option representations and decide's attention over them are
/// permutation-invariant by construction. The packed mask is required (the
/// row form cannot express it).
pub fn encode_packed<T: EdlmEncodeTokenizer>(
    tok: &T,
    rec: &EdlmRecord,
    limits: EdlmLimits,
    strict: bool,
    option_isolation: bool,
) -> Result<PackedEncoding> {
    let marker = |name: &str| -> Result<usize> {
        tok.token_to_id(name)
            .ok_or_else(|| anyhow::anyhow!("eDLM marker '{name}' not in tokenizer vocab"))
    };
    let [m_state, m_q, m_opt, m_opt_end, m_decide] = {
        let mut ids = [0usize; 5];
        for (slot, name) in ids.iter_mut().zip(EDLM_MARKERS.iter()) {
            *slot = marker(name)?;
        }
        ids
    };

    let state_tokens = user_tokens(tok, &rec.state);
    let state_truncated = state_tokens.len() + 1 > limits.max_state;
    if strict && state_truncated {
        bail!(ContextOverflow {
            detail: format!(
                "state exceeds {} tokens: {}",
                limits.max_state,
                state_tokens.len() + 1
            ),
        });
    }
    let mut ids = Vec::with_capacity(limits.max_packed.min(4096));
    let mut pos = Vec::with_capacity(ids.capacity());
    let mut seg = Vec::with_capacity(ids.capacity());
    let mut opt = Vec::with_capacity(ids.capacity());
    ids.push(m_state);
    pos.push(0);
    seg.push(0);
    opt.push(OPT_NONE);
    for &t in state_tokens.iter().take(limits.max_state - 1) {
        ids.push(t);
        pos.push(ids.len() - 1);
        seg.push(0);
        opt.push(OPT_NONE);
    }
    let s_len = ids.len();

    let mut decide_idx = Vec::with_capacity(rec.questions.len());
    let mut opt_idx = Vec::with_capacity(rec.questions.len());
    for (k, q) in rec.questions.iter().enumerate() {
        let k = (k + 1) as i32;
        if q.options.is_empty() {
            bail!("question {k}: at least one option required");
        }
        let instr_len = 1 + user_tokens(tok, &q.instr).len();
        let mut branch: Vec<usize> = Vec::new();
        let mut span_lens: Vec<usize> = Vec::with_capacity(q.options.len());
        branch.push(m_q);
        branch.extend(user_tokens(tok, &q.instr));
        for o in &q.options {
            let ot = user_tokens(tok, o);
            span_lens.push(ot.len() + 2);
            branch.push(m_opt);
            branch.extend(ot);
            branch.push(m_opt_end);
        }
        branch.push(m_decide);

        if branch.len() > limits.max_branch.saturating_sub(s_len) {
            bail!(ContextOverflow {
                detail: format!(
                    "branch too long: {} tokens with a {}-token state (row limit {})",
                    branch.len(),
                    s_len,
                    limits.max_branch
                ),
            });
        }

        let base = ids.len();
        // Option tokens' option indices: instruction OPT_NONE, then the span
        // index per span token, then OPT_DECIDE.
        opt.push(OPT_NONE); // the q marker
        opt.extend(std::iter::repeat_n(OPT_NONE, instr_len - 1));
        for (j, &sl) in span_lens.iter().enumerate() {
            opt.extend(std::iter::repeat_n(j as i32, sl));
        }
        opt.push(OPT_DECIDE);

        if option_isolation {
            let longest = span_lens.iter().copied().max().unwrap_or(0);
            for i in 0..instr_len {
                pos.push(s_len + i);
            }
            for &sl in &span_lens {
                for i in 0..sl {
                    pos.push(s_len + instr_len + i);
                }
            }
            pos.push(s_len + instr_len + longest);
        } else {
            for i in 0..branch.len() {
                pos.push(s_len + i);
            }
        }

        ids.extend_from_slice(&branch);
        seg.extend(std::iter::repeat_n(k, branch.len()));

        let mut cursor = instr_len;
        let ends: Vec<usize> = span_lens
            .iter()
            .map(|&sl| {
                cursor += sl;
                cursor - 1
            })
            .collect();
        decide_idx.push(base + branch.len() - 1);
        opt_idx.push(ends.iter().map(|&e| base + e).collect());
    }

    Ok(PackedEncoding {
        ids,
        pos,
        seg,
        opt: if option_isolation { opt } else { Vec::new() },
        state_len: s_len,
        state_truncated,
        decide_idx,
        opt_idx,
    })
}

// ── T1 — weights ─────────────────────────────────────────────────────────

/// The eDLM pointer head (their `PointerHead`, 256-dim default).
///
/// `z_j = (W_k·h_j + b_k) · (W_q·h_decide + b_q) / sqrt(dp)`, probabilities =
/// softmax per question over the option keys; `temperature` divides the
/// logits BEFORE the softmax (eval only; training always saw T=1 — argmax
/// unchanged by construction). The GGUF carries `pointer.q.weight` F16,
/// `pointer.{q,k}.bias` F32, `pointer.temperature.weight` F32 {1}.
#[derive(Debug, Clone, Default)]
pub struct EdlmPointerHead {
    /// `[dp, d]` row-major (our matmul convention).
    pub q: Vec<f32>,
    pub q_bias: Vec<f32>,
    /// `[dp, d]` row-major.
    pub k: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub dp: usize,
    /// Input hidden dim `d` (4096 on the 8B release).
    pub d: usize,
    pub temperature: f32,
}

impl EdlmPointerHead {
    /// Per-question option probabilities: `h_decide` `[d]`,
    /// `h_opts` `[n_opts * d]` concatenated in option order.
    ///
    /// `z_j = (W_k·h_j + b_k) · (W_q·h_decide + b_q) / sqrt(dp)`, then
    /// softmax over the option keys (plain `exp`, the reference numerics —
    /// not the decode exp-table path).
    pub fn question_probs(&self, h_decide: &[f32], h_opts: &[f32], n_opts: usize) -> Vec<f32> {
        assert_eq!(h_decide.len(), self.d, "h_decide must be [d]");
        assert_eq!(h_opts.len(), n_opts * self.d, "h_opts must be [n_opts * d]");
        assert!(n_opts > 0);
        let mut qv = vec![0.0f32; self.dp];
        let mut kv = vec![0.0f32; self.dp];
        crate::types::matmul(&mut qv, &self.q, h_decide, self.dp, self.d);
        for (qv_i, qb) in qv.iter_mut().zip(self.q_bias.iter()) {
            *qv_i += qb;
        }
        let scale = 1.0 / (self.dp as f32).sqrt();
        let t = if self.temperature > 0.0 {
            self.temperature
        } else {
            1.0
        };
        let mut z = vec![0.0f32; n_opts];
        let mut max_z = f32::NEG_INFINITY;
        for j in 0..n_opts {
            crate::types::matmul(
                &mut kv,
                &self.k,
                &h_opts[j * self.d..(j + 1) * self.d],
                self.dp,
                self.d,
            );
            let mut dot = 0.0f32;
            for i in 0..self.dp {
                dot += (kv[i] + self.k_bias[i]) * qv[i];
            }
            z[j] = dot * scale / t;
            max_z = max_z.max(z[j]);
        }
        // Softmax (stable).
        let mut sum = 0.0f32;
        for v in z.iter_mut() {
            *v = (*v - max_z).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in z.iter_mut() {
            *v *= inv;
        }
        z
    }
}

/// One eDLM transformer layer: the llama-family projections plus the Qwen3
/// QK-RMSNorm gammas (`attn_q_norm` / `attn_k_norm`, `[head_dim]`, applied
/// per head AFTER projection, BEFORE RoPE).
pub struct EdlmLayerWeights {
    pub base: LlamaLayerWeights,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// Full eDLM weight set (dequantized f32). No KV cache — the lane scores one
/// full sequence per forward. `lm_head` is optional: the scoring readout is
/// the pointer head over hidden states, never text logits (the GGUF carries
/// `output.weight`, but a tied-emissions conversion may omit it).
pub struct EdlmWeights {
    pub wte: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f32>>,
    pub layers: Vec<EdlmLayerWeights>,
    pub pointer: Option<EdlmPointerHead>,
}

/// `Config` from `edlm.*` GGUF metadata (prefix mirrors
/// `qwen2_config_from_gguf_metadata`; header-verified defaults).
fn edlm_config_from_gguf_metadata(
    gguf: &crate::gguf_loader::GgufFile,
) -> Result<crate::types::Config> {
    let prefix = "edlm.";
    let context_length = gguf
        .metadata_u64(&format!("{prefix}context_length"))
        .unwrap_or(32768) as usize;
    let n_embd = gguf
        .metadata_u64(&format!("{prefix}embedding_length"))
        .unwrap_or(4096) as usize;
    let n_layer = gguf
        .metadata_u64(&format!("{prefix}block_count"))
        .unwrap_or(36) as usize;
    let mlp_hidden = gguf
        .metadata_u64(&format!("{prefix}feed_forward_length"))
        .unwrap_or(12288) as usize;
    let n_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count"))
        .unwrap_or(32) as usize;
    let n_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count_kv"))
        .unwrap_or(8) as usize;
    let rms_norm_eps = gguf
        .metadata_f64(&format!("{prefix}attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length"))
        .unwrap_or((n_embd / n_head) as u64) as usize;
    let rope_theta = gguf
        .metadata_f64(&format!("{prefix}rope.freq_base"))
        .unwrap_or(1_000_000.0) as f32;

    let vocab_size = gguf
        .tensor_info("token_embd.weight")
        .and_then(|info| info.shape.last().copied())
        .unwrap_or(151_936);

    let mut config = crate::types::Config::micro();
    config.vocab_size = vocab_size;
    config.block_size = context_length;
    config.n_embd = n_embd;
    config.n_head = n_head;
    config.head_dim = head_dim;
    config.mlp_hidden = mlp_hidden;
    config.n_layer = n_layer;
    config.n_kv_head = n_kv_head;
    config.rms_norm_eps = rms_norm_eps;
    config.rope_theta = rope_theta;
    // The lane's forward is `forward_edlm_packed`, invoked explicitly — the
    // enum is a dispatch hint for OTHER code paths only. `Llama` is the
    // closest surface (RoPE-family, SwiGLU, no embedding scaling), the same
    // call the qwen2 loader makes. A dedicated variant would be an
    // upstream (katgpt-types) change — not this unit's blast radius.
    config.model_arch = crate::types::ModelArchitecture::Llama;
    config.use_rope = true;
    config.tied_embeddings = false;
    config.rms_norm_offset = false;
    config.post_norm = false;
    config.attn_logit_softcapping = 0.0;
    config.final_logit_softcapping = 0.0;
    Ok(config)
}

/// Load eDLM weights from a GGUF file (arch `edlm`).
///
/// Tensor map (header-verified against `drex-dlm-Q8_0.gguf`):
/// `token_embd.weight`, `output_norm.weight`, `output.weight` (optional),
/// per layer `blk.{N}.attn_norm.weight`, `blk.{N}.attn_{q,k,v}.weight`,
/// `blk.{N}.attn_output.weight`, `blk.{N}.attn_{q,k}_norm.weight`,
/// `blk.{N}.ffn_{norm,gate,up,down}.weight`; pointer
/// `pointer.{q,k}.weight` (F16) + `pointer.{q,k}.bias` (F32) +
/// `pointer.temperature.weight` (F32, NE 1; absent ⇒ 1.0).
///
/// RoPE is NEOX (rotate-half) — the fork maps `LLM_ARCH_EDLM` to
/// `LLAMA_ROPE_TYPE_NEOX` — so Q/K rows are stored in HF order and are NOT
/// unpermuted (the `qwen2` loader posture, not the `llama` one).
///
/// Supports every quant `GgufFile::dequant_f16_to_f32` handles (the release
/// GGUF is Q8_0 with F16 pointer weights / F32 norms).
pub fn load_edlm_weights_gguf(
    path: &std::path::Path,
) -> Result<(crate::types::Config, EdlmWeights)> {
    let gguf = crate::gguf_loader::GgufFile::open(path)?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "edlm" {
        bail!("expected edlm architecture, got '{arch}'");
    }
    let config = edlm_config_from_gguf_metadata(&gguf)?;
    let n_layer = config.n_layer;

    let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;
    let lm_head = gguf.try_dequant_f16_to_f32("output.weight")?;

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        layers.push(dequant_edlm_layer(&gguf, i, &config)?);
    }

    let pointer = edlm_pointer_head_from_gguf(&gguf, config.n_embd)?;

    Ok((
        config,
        EdlmWeights {
            wte,
            final_norm,
            lm_head,
            layers,
            pointer,
        },
    ))
}

/// Dequant the pointer-head tensors (`pointer.{q,k}.weight` + biases +
/// temperature) into an [`EdlmPointerHead`]; `None` when the file carries no
/// pointer (the head is optional — a backbone-only conversion).
fn edlm_pointer_head_from_gguf(
    gguf: &crate::gguf_loader::GgufFile,
    n_embd: usize,
) -> Result<Option<EdlmPointerHead>> {
    if gguf.tensor_info("pointer.q.weight").is_none() {
        return Ok(None);
    }
    let q = gguf.dequant_f16_to_f32("pointer.q.weight")?;
    let k = gguf.dequant_f16_to_f32("pointer.k.weight")?;
    let q_bias = gguf
        .try_dequant_f16_to_f32("pointer.q.bias")?
        .unwrap_or_default();
    let k_bias = gguf
        .try_dequant_f16_to_f32("pointer.k.bias")?
        .unwrap_or_default();
    let temperature = gguf
        .try_dequant_f16_to_f32("pointer.temperature.weight")?
        .and_then(|v| v.first().copied())
        .unwrap_or(1.0);
    if q.len() != k.len() || !q.len().is_multiple_of(n_embd) {
        bail!(
            "pointer weight shape mismatch: q {} k {} (n_embd {})",
            q.len(),
            k.len(),
            n_embd
        );
    }
    let dp = q.len() / n_embd;
    Ok(Some(EdlmPointerHead {
        q,
        q_bias,
        k,
        k_bias,
        dp,
        d: n_embd,
        temperature,
    }))
}

// ── the forward ─────────────────────────────────────────────────────

/// Per-forward reusable scratch: position-independent buffers sized once from
/// (config, sequence length), reused across every layer — nothing inside a
/// layer's loop allocates.
struct EdlmLayerScratch {
    x: Vec<f32>,
    xr: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    mlp_out: Vec<f32>,
    freq: RopeFreqTable,
    scale: f32,
}

impl EdlmLayerScratch {
    fn new(config: &crate::types::Config, l: usize) -> Self {
        let n = config.n_embd;
        let hd = config.head_dim;
        let q_dim = config.n_head * hd;
        let kvd = crate::types::kv_dim(config);
        Self {
            x: vec![0.0; n],
            xr: vec![0.0; n],
            q: vec![0.0; q_dim],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; q_dim],
            scores: vec![0.0; l],
            key_cache: vec![0.0; l * kvd],
            value_cache: vec![0.0; l * kvd],
            gate: vec![0.0; config.mlp_hidden],
            up: vec![0.0; config.mlp_hidden],
            mlp_out: vec![0.0; config.mlp_hidden],
            freq: RopeFreqTable::new(config.rope_theta, hd),
            scale: 1.0 / (hd as f32).sqrt(),
        }
    }
}

/// One transformer layer over the whole sequence, mutating `hidden` in place.
/// The op order Phase 1 pinned: pre-norm residual stream, per-head QK-RMSNorm
/// BEFORE RoPE (NEOX), masked GQA (no KV cache — the mask IS the segment
/// structure), SwiGLU MLP with the post-attn-norm stream.
fn edlm_layer_forward(
    layer: &EdlmLayerWeights,
    config: &crate::types::Config,
    hidden: &mut [f32],
    pos: &[usize],
    allow: &[bool],
    s: &mut EdlmLayerScratch,
) {
    let l = pos.len();
    edlm_layer_kv(layer, config, hidden, pos, 0, s);
    edlm_layer_attend(layer, config, hidden, pos, allow, l, s);
}

/// Phase A — K/V for every position (mask-independent), written to cache
/// slots `kv_base..kv_base + pos.len()`. Split from [`edlm_layer_forward`]
/// so the state-prefix prefill can capture K/V per layer before the
/// attention runs (T7); the plain path passes `kv_base = 0`.
fn edlm_layer_kv(
    layer: &EdlmLayerWeights,
    config: &crate::types::Config,
    hidden: &[f32],
    pos: &[usize],
    kv_base: usize,
    s: &mut EdlmLayerScratch,
) {
    let l = pos.len();
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let EdlmLayerScratch {
        x,
        k,
        v,
        key_cache,
        value_cache,
        freq,
        ..
    } = s;

    for p in 0..l {
        x.copy_from_slice(&hidden[p * n..(p + 1) * n]);
        crate::types::rmsnorm_with_gamma_eps(x, &layer.base.input_norm, config.rms_norm_eps);
        crate::types::matmul(k, &layer.base.attn_wk, x, kvd, n);
        crate::types::matmul(v, &layer.base.attn_wv, x, kvd, n);
        // K: per-KV-head QK-norm, then RoPE at this position (rotate the
        // K buffer only — an empty second slice applies zero K heads).
        qk_norm_inplace(k, &layer.k_norm, n_kv, hd, config.rms_norm_eps);
        crate::rope::apply_rope_with_freq(k, &mut [], pos[p], hd, freq.as_slice());
        let slot = (kv_base + p) * kvd;
        key_cache[slot..slot + kvd].copy_from_slice(k);
        value_cache[slot..slot + kvd].copy_from_slice(v);
    }
}

/// Phase B — Q per position + masked GQA attention over `key_count` cached
/// keys, then the SwiGLU MLP, mutating `hidden` in place. Query `p` reads
/// allow row `p` (`[key_count]` wide); the keys it may see span the whole
/// cache from slot 0 — including state slots written by a preceding Phase A
/// at `kv_base > 0` (the T7 continuation), which this half never needs
/// itself: the plain path passes `key_count = pos.len()`, the continuation
/// the combined `state_len + branch_len`.
#[allow(clippy::too_many_arguments)]
fn edlm_layer_attend(
    layer: &EdlmLayerWeights,
    config: &crate::types::Config,
    hidden: &mut [f32],
    pos: &[usize],
    allow: &[bool],
    key_count: usize,
    s: &mut EdlmLayerScratch,
) {
    let l = pos.len();
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * hd;
    let kvd = crate::types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let EdlmLayerScratch {
        x,
        xr,
        q,
        attn_out,
        scores,
        key_cache,
        value_cache,
        gate,
        up,
        mlp_out,
        freq,
        scale,
        ..
    } = s;

    for p in 0..l {
        x.copy_from_slice(&hidden[p * n..(p + 1) * n]);
        // Residual saved PRE-norm — the attention block adds back the
        // un-normed stream (the forward_llama residual law).
        xr.copy_from_slice(x);
        crate::types::rmsnorm_with_gamma_eps(x, &layer.base.input_norm, config.rms_norm_eps);
        crate::types::matmul(q, &layer.base.attn_wq, x, q_dim, n);
        // Q: per-head QK-norm, then RoPE (q-only — empty second slice).
        qk_norm_inplace(q, &layer.q_norm, config.n_head, hd, config.rms_norm_eps);
        crate::rope::apply_rope_with_freq(q, &mut [], pos[p], hd, freq.as_slice());

        attn_out[..n].fill(0.0);
        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                super::attention::attention_head_masked(
                    q,
                    key_cache,
                    value_cache,
                    attn_out,
                    scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    key_count,
                    *scale,
                    allow,
                    p,
                );
            }
        }

        crate::types::matmul(x, &layer.base.attn_wo, attn_out, n, q_dim);
        for i in 0..n {
            x[i] += xr[i];
        }

        // MLP: SwiGLU (out = silu(gate) ⊙ up, then down-proj).
        xr.copy_from_slice(x);
        crate::types::rmsnorm_with_gamma_eps(x, &layer.base.post_attn_norm, config.rms_norm_eps);
        crate::types::matmul(gate, &layer.base.gate_proj, x, config.mlp_hidden, n);
        crate::types::matmul(up, &layer.base.up_proj, x, config.mlp_hidden, n);
        crate::types::swiglu(mlp_out, gate, up);
        crate::types::matmul(x, &layer.base.down_proj, mlp_out, n, config.mlp_hidden);
        for i in 0..n {
            x[i] += xr[i];
        }

        hidden[p * n..(p + 1) * n].copy_from_slice(x);
    }
}

/// Embed token ids into a fresh hidden buffer (no scaling, no wpe — RoPE
/// carries position).
fn edlm_embed(wte: &[f32], ids: &[usize], n: usize) -> Vec<f32> {
    let mut hidden = vec![0.0f32; ids.len() * n];
    for (p, &id) in ids.iter().enumerate() {
        hidden[p * n..(p + 1) * n].copy_from_slice(&wte[id * n..(id + 1) * n]);
    }
    hidden
}

/// Final norm on every position (the pointer head reads marker positions from
/// the normed buffer; no lm_head — the lane never generates text).
fn edlm_final_norm(hidden: &mut [f32], final_norm: &[f32], config: &crate::types::Config) {
    let n = config.n_embd;
    for chunk in hidden.chunks_exact_mut(n) {
        crate::types::rmsnorm_with_gamma_eps(chunk, final_norm, config.rms_norm_eps);
    }
}

/// Full-sequence eDLM scoring forward (Qwen3 blocks, NO KV cache).
///
/// `ids`/`pos`/`allow` describe the sequence (`allow` row-major
/// `[l * l]` — `branch_mask` or `row_branch_mask` output). Returns the
/// post-final-norm hidden states `[l * n_embd]`; the pointer head reads
/// marker positions from this buffer (no lm_head — the lane never generates
/// text).
pub fn forward_edlm_packed(
    weights: &EdlmWeights,
    config: &crate::types::Config,
    ids: &[usize],
    pos: &[usize],
    allow: &[bool],
) -> Vec<f32> {
    let l = ids.len();
    assert_eq!(pos.len(), l, "pos must match ids");
    assert_eq!(allow.len(), l * l, "allow must be l*l");
    assert!(l > 0 && l <= config.block_size, "l {l} out of range");
    assert_eq!(weights.layers.len(), config.n_layer, "layer count mismatch");
    assert_eq!(
        config.n_embd % config.n_head,
        0,
        "n_embd must divide by n_head"
    );

    let mut scratch = EdlmLayerScratch::new(config, l);
    let mut hidden = edlm_embed(&weights.wte, ids, config.n_embd);
    for layer in &weights.layers {
        edlm_layer_forward(layer, config, &mut hidden, pos, allow, &mut scratch);
    }
    edlm_final_norm(&mut hidden, &weights.final_norm, config);
    hidden
}

/// Quant-resident eDLM model: the GGUF mmap stays open and the forward
/// dequants ONE layer at a time into a reused f32 scratch.
///
/// Resident RAM is O(file) — the mmap is file-backed and evictable under
/// pressure — plus O(one layer) of f32 (~0.8 GB on the 8B release) and the
/// whole embedding table (~2.5 GB, dequanted at open: every sequence touches
/// scattered rows, and the one-time whole-table dequant is cheap). Fully
/// dequantizing the 8B Q8_0 release would hold ~32 GB of f32 instead.
pub struct EdlmGgufModel {
    /// The open mmap — `pub` for the GPU lane (Issue 1005 T8): the f16
    /// upload path dequants the same tensors the streaming CPU forward
    /// reads, from the same handle (one open, two consumers).
    pub gguf: crate::gguf_loader::GgufFile,
    pub config: crate::types::Config,
    pub wte: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub pointer: Option<EdlmPointerHead>,
}

impl EdlmGgufModel {
    /// Open an `edlm`-arch GGUF (arch check + config + embedding/final-norm/
    /// pointer dequant). Tensor data itself is read lazily from the mmap by
    /// the forward.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let gguf = crate::gguf_loader::GgufFile::open(path)?;
        let arch = gguf.architecture().unwrap_or("unknown");
        if arch != "edlm" {
            bail!("expected edlm architecture, got '{arch}'");
        }
        let config = edlm_config_from_gguf_metadata(&gguf)?;
        let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
        if wte.len() != config.vocab_size * config.n_embd {
            bail!(
                "token_embd dequants to {} elements, expected vocab {} × n_embd {}",
                wte.len(),
                config.vocab_size,
                config.n_embd
            );
        }
        let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;
        let pointer = edlm_pointer_head_from_gguf(&gguf, config.n_embd)?;
        Ok(Self {
            gguf,
            config,
            wte,
            final_norm,
            pointer,
        })
    }

    /// The GGUF-embedded tokenizer (Qwen-family BPE).
    pub fn tokenizer(&self) -> Result<BpeTokenizer> {
        BpeTokenizer::from_gguf(&self.gguf)
    }
}

/// Full-sequence eDLM forward over a quant-resident model — the [`
/// forward_edlm_packed`] contract, one layer dequanted at a time.
///
/// Per layer the nine block tensors + the two QK-norm gammas are dequanted
/// into a reused [`EdlmLayerWeights`] (~0.8 GB churn per layer, allocator-
/// recycled) and run through the SAME `edlm_layer_forward` body the
/// fully-loaded path uses — byte-identical math, O(one layer) f32 residency.
pub fn forward_edlm_packed_streaming(
    model: &EdlmGgufModel,
    ids: &[usize],
    pos: &[usize],
    allow: &[bool],
) -> Result<Vec<f32>> {
    let config = &model.config;
    let l = ids.len();
    assert_eq!(pos.len(), l, "pos must match ids");
    assert_eq!(allow.len(), l * l, "allow must be l*l");
    assert!(l > 0 && l <= config.block_size, "l {l} out of range");
    assert_eq!(
        config.n_embd % config.n_head,
        0,
        "n_embd must divide by n_head"
    );

    let mut scratch = EdlmLayerScratch::new(config, l);
    let mut hidden = edlm_embed(&model.wte, ids, config.n_embd);
    for i in 0..config.n_layer {
        let layer = dequant_edlm_layer(&model.gguf, i, config)?;
        edlm_layer_forward(&layer, config, &mut hidden, pos, allow, &mut scratch);
    }
    edlm_final_norm(&mut hidden, &model.final_norm, config);
    Ok(hidden)
}

/// Row-form forward over a quant-resident model — the [`forward_edlm_rows`]
/// contract through [`forward_edlm_packed_streaming`] (`state_bidir` must
/// match the packed posture being mirrored, same law).
pub fn forward_edlm_rows_streaming(
    model: &EdlmGgufModel,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    state_bidir: bool,
) -> Result<Vec<Vec<f32>>> {
    let sl = enc.state_len;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let bl = row.branch_len();
        let mut ids = Vec::with_capacity(sl + bl);
        let mut pos = Vec::with_capacity(sl + bl);
        ids.extend_from_slice(&enc.ids[..sl]);
        pos.extend_from_slice(&enc.pos[..sl]);
        ids.extend_from_slice(&enc.ids[row.start..row.end]);
        pos.extend_from_slice(&enc.pos[row.start..row.end]);
        let allow = row_branch_mask(sl, bl, state_bidir);
        out.push(forward_edlm_packed_streaming(model, &ids, &pos, &allow)?);
    }
    Ok(out)
}

/// Dequant one eDLM block's nine tensors + QK-norm gammas from the mmap.
fn dequant_edlm_layer(
    gguf: &crate::gguf_loader::GgufFile,
    i: usize,
    config: &crate::types::Config,
) -> Result<EdlmLayerWeights> {
    let base = LlamaLayerWeights {
        attn_wq: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_q.weight"))?,
        attn_wk: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_k.weight"))?,
        attn_wv: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_v.weight"))?,
        attn_wo: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_output.weight"))?,
        gate_proj: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_gate.weight"))?,
        up_proj: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_up.weight"))?,
        down_proj: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_down.weight"))?,
        input_norm: gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_norm.weight"))?,
        post_attn_norm: gguf.dequant_f16_to_f32(&format!("blk.{i}.ffn_norm.weight"))?,
    };
    let q_norm = gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_q_norm.weight"))?;
    let k_norm = gguf.dequant_f16_to_f32(&format!("blk.{i}.attn_k_norm.weight"))?;
    if q_norm.len() != config.head_dim || k_norm.len() != config.head_dim {
        bail!(
            "layer {i}: QK-norm gamma length {}/{} != head_dim {}",
            q_norm.len(),
            k_norm.len(),
            config.head_dim
        );
    }
    Ok(EdlmLayerWeights {
        base,
        q_norm,
        k_norm,
    })
}

/// Per-head RMSNorm in place over a concatenated multi-head buffer.
///
/// `pub` for the GPU eDLM lane (riir-infer-gpu Issue 1005 T8): the hybrid
/// forward runs qk-norm host-side between the qkv readback and the attention
/// upload — the exact same helper, never a copy.
pub fn qk_norm_inplace(buf: &mut [f32], gamma: &[f32], n_heads: usize, head_dim: usize, eps: f64) {
    debug_assert_eq!(buf.len(), n_heads * head_dim);
    debug_assert_eq!(gamma.len(), head_dim);
    for h in 0..n_heads {
        let off = h * head_dim;
        let head = &mut buf[off..off + head_dim];
        let mut sum_sq = 0.0f32;
        for &v in head.iter() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps as f32).sqrt();
        for (v, &g) in head.iter_mut().zip(gamma.iter()) {
            *v *= inv_rms * g;
        }
    }
}

/// Row-form forward: every question runs as a self-contained
/// `state ++ branch` causal row (mask from [`row_branch_mask`]).
///
/// `state_bidir` must MATCH the packed posture being mirrored — parity holds
/// in both postures, never across them (with the flag off the packed state is
/// a causal prefix of itself, and the row mask must say the same).
///
/// Returns per-question hidden states `[row_len * n_embd]` — slice branch
/// tokens at `[(state_len + off) * n_embd ..]` with the row's offsets.
pub fn forward_edlm_rows(
    weights: &EdlmWeights,
    config: &crate::types::Config,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    state_bidir: bool,
) -> Vec<Vec<f32>> {
    let sl = enc.state_len;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let bl = row.branch_len();
        let mut ids = Vec::with_capacity(sl + bl);
        let mut pos = Vec::with_capacity(sl + bl);
        ids.extend_from_slice(&enc.ids[..sl]);
        pos.extend_from_slice(&enc.pos[..sl]);
        ids.extend_from_slice(&enc.ids[row.start..row.end]);
        pos.extend_from_slice(&enc.pos[row.start..row.end]);
        let allow = row_branch_mask(sl, bl, state_bidir);
        out.push(forward_edlm_packed(weights, config, &ids, &pos, &allow));
    }
    out
}

// ── T7 — state-prefix KV reuse ──────────────────────────────────────

/// The state-prefix reuse crossover (their `prefix_min_tokens` law): cache
/// the state K/V only from this many state tokens — below it the branch-only
/// pass is not faster than one whole-sequence pass (per-op overhead; their
/// MPS measurement, the same per-op class on our CPU lane). Their hybrid
/// (recurrent-layer) backbones always cache (`0`); this lane is dense
/// attention-only, so 384 — their training `MAX_STATE`.
pub const EDLM_PREFIX_MIN_TOKENS: usize = 384;

/// One layer's cached state K/V, each `[state_len * kvd]` row-major (the
/// slot order matches the state segment's token order).
#[derive(Debug, Clone)]
pub struct EdlmLayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// The cached state prefix (T7): per-layer K/V for the state segment,
/// computed once and reused across every question's branch row.
///
/// Exact by construction — the state never sees branches (the packed mask
/// law), so the state's hidden states and K/V are identical with or without
/// the branches present; the branch continuation attends the cached keys in
/// the same ascending order the whole-sequence pass would. Cache footprint:
/// `n_layer * 2 * state_len * kvd * 4` bytes (~113 MB at the 384-token
/// training posture on the 8B release; ~4.8 GB at a 16K serving state — the
/// caller's choice via the state length it prefills).
#[derive(Debug, Clone)]
pub struct EdlmStateKv {
    pub state_len: usize,
    pub kvd: usize,
    /// The state token ids the cache was built from — the stale-cache
    /// content check (a same-length different-state cache must REFUSE, never
    /// silently serve).
    pub state_ids: Vec<usize>,
    /// One entry per layer, in layer order.
    pub layers: Vec<EdlmLayerKv>,
    /// Post-final-norm state hidden states `[state_len * n_embd]` — the row
    /// form's state prefix for free (the pointer head reads markers in the
    /// branches, so this is a caller convenience and the drop-in row shape).
    pub hidden: Vec<f32>,
}

impl EdlmStateKv {
    /// Refuse a stale or mismatched cache: same state token content, same
    /// geometry, same layer count. A cache that does not match MUST error,
    /// never silently reuse (wrongness, not a fast path).
    pub fn check_matches(&self, config: &crate::types::Config, state_ids: &[usize]) -> Result<()> {
        if self.state_len != state_ids.len() {
            bail!(
                "state-prefix cache holds {} state tokens, encoding has {}",
                self.state_len,
                state_ids.len()
            );
        }
        if self.kvd != crate::types::kv_dim(config) {
            bail!(
                "state-prefix cache kvd {} != config kv_dim {}",
                self.kvd,
                crate::types::kv_dim(config)
            );
        }
        if self.state_ids != state_ids {
            bail!("stale state-prefix cache: state token content differs");
        }
        if self.layers.len() != config.n_layer {
            bail!(
                "state-prefix cache holds {} layers, config has {}",
                self.layers.len(),
                config.n_layer
            );
        }
        Ok(())
    }
}

/// Branch-continuation attention eligibility: branch query `p` attends ALL
/// state keys (they precede every branch query — posture-independent: the
/// `state_bidir` flag only widens state↔state, and branch rows never contain
/// state queries) plus its own branch prefix `j <= p`. Row-major
/// `[branch_len * (state_len + branch_len)]`, one row per branch query.
fn branch_continuation_allow(state_len: usize, branch_len: usize) -> Vec<bool> {
    let l = state_len + branch_len;
    let mut out = vec![false; branch_len * l];
    for p in 0..branch_len {
        let row = &mut out[p * l..(p + 1) * l];
        row[..state_len].fill(true);
        row[state_len..state_len + p + 1].fill(true);
    }
    out
}

/// Run the state tokens only and capture their per-layer K/V (their
/// `prefix`): the state-only mask is `branch_mask` over an all-state segment
/// — causal, widened to state↔state bidirectional when `state_bidir` —
/// exactly the state block the row form would compute. The returned
/// [`EdlmStateKv`] feeds [`forward_edlm_branches_cached`].
pub fn edlm_state_prefill(
    weights: &EdlmWeights,
    config: &crate::types::Config,
    state_ids: &[usize],
    state_pos: &[usize],
    state_bidir: bool,
) -> EdlmStateKv {
    let sl = state_ids.len();
    assert_eq!(state_pos.len(), sl, "state_pos must match state_ids");
    assert!(
        sl > 0 && sl <= config.block_size,
        "state len {sl} out of range"
    );
    let kvd = crate::types::kv_dim(config);
    let mut allow = vec![false; sl * sl];
    branch_mask(&vec![0i32; sl], state_bidir, None, &mut allow);
    let mut scratch = EdlmLayerScratch::new(config, sl);
    let mut hidden = edlm_embed(&weights.wte, state_ids, config.n_embd);
    let mut layers = Vec::with_capacity(weights.layers.len());
    for layer in &weights.layers {
        edlm_layer_kv(layer, config, &hidden, state_pos, 0, &mut scratch);
        layers.push(EdlmLayerKv {
            k: scratch.key_cache.clone(),
            v: scratch.value_cache.clone(),
        });
        edlm_layer_attend(
            layer,
            config,
            &mut hidden,
            state_pos,
            &allow,
            sl,
            &mut scratch,
        );
    }
    edlm_final_norm(&mut hidden, &weights.final_norm, config);
    EdlmStateKv {
        state_len: sl,
        kvd,
        state_ids: state_ids.to_vec(),
        layers,
        hidden,
    }
}

/// [`edlm_state_prefill`] over a quant-resident model — one layer dequanted
/// at a time, the same [`EdlmStateKv`].
pub fn edlm_state_prefill_streaming(
    model: &EdlmGgufModel,
    state_ids: &[usize],
    state_pos: &[usize],
    state_bidir: bool,
) -> Result<EdlmStateKv> {
    let config = &model.config;
    let sl = state_ids.len();
    assert_eq!(state_pos.len(), sl, "state_pos must match state_ids");
    assert!(
        sl > 0 && sl <= config.block_size,
        "state len {sl} out of range"
    );
    let kvd = crate::types::kv_dim(config);
    let mut allow = vec![false; sl * sl];
    branch_mask(&vec![0i32; sl], state_bidir, None, &mut allow);
    let mut scratch = EdlmLayerScratch::new(config, sl);
    let mut hidden = edlm_embed(&model.wte, state_ids, config.n_embd);
    let mut layers = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let layer = dequant_edlm_layer(&model.gguf, i, config)?;
        edlm_layer_kv(&layer, config, &hidden, state_pos, 0, &mut scratch);
        layers.push(EdlmLayerKv {
            k: scratch.key_cache.clone(),
            v: scratch.value_cache.clone(),
        });
        edlm_layer_attend(
            &layer,
            config,
            &mut hidden,
            state_pos,
            &allow,
            sl,
            &mut scratch,
        );
    }
    edlm_final_norm(&mut hidden, &model.final_norm, config);
    Ok(EdlmStateKv {
        state_len: sl,
        kvd,
        state_ids: state_ids.to_vec(),
        layers,
        hidden,
    })
}

/// Seed the scratch's combined key/value caches with a layer's cached state
/// K/V (the continuation writes its own branch K/V after the state slots).
fn seed_state_kv(scratch: &mut EdlmLayerScratch, kv: &EdlmLayerKv, sl: usize, kvd: usize) {
    debug_assert_eq!(kv.k.len(), sl * kvd);
    debug_assert_eq!(kv.v.len(), sl * kvd);
    scratch.key_cache[..sl * kvd].copy_from_slice(&kv.k);
    scratch.value_cache[..sl * kvd].copy_from_slice(&kv.v);
}

/// Branch rows continuing a cached state prefix (their
/// `_branch_rows_from_prefix`): the plain [`forward_edlm_rows`] layout minus
/// the per-row state recompute. `cache` must be this encoding's state prefix
/// ([`EdlmStateKv::check_matches`] refuses a stale one). Returns the same
/// per-row `[row_len * n_embd]` hidden states as the plain row form — the
/// state prefix comes from the cache, so the output is a drop-in.
///
/// The option-isolation caveat carries from the plain row form: isolation is
/// a packed-mask posture and cannot be expressed as rows.
pub fn forward_edlm_branches_cached(
    weights: &EdlmWeights,
    config: &crate::types::Config,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    cache: &EdlmStateKv,
) -> Result<Vec<Vec<f32>>> {
    let sl = enc.state_len;
    cache.check_matches(config, &enc.ids[..sl])?;
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let bl = row.branch_len();
        let key_count = sl + bl;
        let allow = branch_continuation_allow(sl, bl);
        let mut scratch = EdlmLayerScratch::new(config, key_count);
        let mut hidden = edlm_embed(&weights.wte, &enc.ids[row.start..row.end], n);
        for (i, layer) in weights.layers.iter().enumerate() {
            seed_state_kv(&mut scratch, &cache.layers[i], sl, kvd);
            edlm_layer_kv(
                layer,
                config,
                &hidden,
                &enc.pos[row.start..row.end],
                sl,
                &mut scratch,
            );
            edlm_layer_attend(
                layer,
                config,
                &mut hidden,
                &enc.pos[row.start..row.end],
                &allow,
                key_count,
                &mut scratch,
            );
        }
        edlm_final_norm(&mut hidden, &weights.final_norm, config);
        let mut full = Vec::with_capacity((sl + bl) * n);
        full.extend_from_slice(&cache.hidden);
        full.extend_from_slice(&hidden);
        out.push(full);
    }
    Ok(out)
}

/// [`forward_edlm_branches_cached`] over a quant-resident model.
pub fn forward_edlm_branches_cached_streaming(
    model: &EdlmGgufModel,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    cache: &EdlmStateKv,
) -> Result<Vec<Vec<f32>>> {
    let config = &model.config;
    let sl = enc.state_len;
    cache.check_matches(config, &enc.ids[..sl])?;
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let bl = row.branch_len();
        let key_count = sl + bl;
        let allow = branch_continuation_allow(sl, bl);
        let mut scratch = EdlmLayerScratch::new(config, key_count);
        let mut hidden = edlm_embed(&model.wte, &enc.ids[row.start..row.end], n);
        for i in 0..config.n_layer {
            let layer = dequant_edlm_layer(&model.gguf, i, config)?;
            seed_state_kv(&mut scratch, &cache.layers[i], sl, kvd);
            edlm_layer_kv(
                &layer,
                config,
                &hidden,
                &enc.pos[row.start..row.end],
                sl,
                &mut scratch,
            );
            edlm_layer_attend(
                &layer,
                config,
                &mut hidden,
                &enc.pos[row.start..row.end],
                &allow,
                key_count,
                &mut scratch,
            );
        }
        edlm_final_norm(&mut hidden, &model.final_norm, config);
        let mut full = Vec::with_capacity((sl + bl) * n);
        full.extend_from_slice(&cache.hidden);
        full.extend_from_slice(&hidden);
        out.push(full);
    }
    Ok(out)
}

/// The one-call row form with the state-prefix reuse crossover: at or above
/// `prefix_min_tokens` state tokens the state K/V is prefilled once and the
/// branches run as continuations; below it the plain [`forward_edlm_rows`]
/// pass is cheaper (the [`EDLM_PREFIX_MIN_TOKENS`] law). Output is the
/// plain row form's, bit-identical either way.
pub fn forward_edlm_rows_cached(
    weights: &EdlmWeights,
    config: &crate::types::Config,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    state_bidir: bool,
    prefix_min_tokens: usize,
) -> Vec<Vec<f32>> {
    if rows.is_empty() {
        return Vec::new();
    }
    if enc.state_len >= prefix_min_tokens {
        let cache = edlm_state_prefill(
            weights,
            config,
            &enc.ids[..enc.state_len],
            &enc.pos[..enc.state_len],
            state_bidir,
        );
        forward_edlm_branches_cached(weights, config, enc, rows, &cache)
            .expect("state cache built from this encoding")
    } else {
        forward_edlm_rows(weights, config, enc, rows, state_bidir)
    }
}

/// [`forward_edlm_rows_cached`] over a quant-resident model.
pub fn forward_edlm_rows_cached_streaming(
    model: &EdlmGgufModel,
    enc: &PackedEncoding,
    rows: &[BranchRow],
    state_bidir: bool,
    prefix_min_tokens: usize,
) -> Result<Vec<Vec<f32>>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    if enc.state_len >= prefix_min_tokens {
        let cache = edlm_state_prefill_streaming(
            model,
            &enc.ids[..enc.state_len],
            &enc.pos[..enc.state_len],
            state_bidir,
        )?;
        forward_edlm_branches_cached_streaming(model, enc, rows, &cache)
    } else {
        forward_edlm_rows_streaming(model, enc, rows, state_bidir)
    }
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const S: i32 = 0;

    fn seg_state_branches(state: usize, branches: &[usize]) -> Vec<i32> {
        let mut seg = vec![S; state];
        for (k, &bl) in branches.iter().enumerate() {
            seg.extend(std::iter::repeat_n(k as i32 + 1, bl));
        }
        seg
    }

    #[test]
    fn branch_mask_four_quadrant_law() {
        // state(2) | q1(3) | q2(3)
        let seg = seg_state_branches(2, &[3, 3]);
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, true, None, &mut m);

        let at = |i: usize, j: usize| m[i * l + j];

        // state↔state: bidirectional (state_bidir = true).
        assert!(at(0, 1) && at(1, 0), "state pair must attend both ways");
        // state ↛ any branch (either direction).
        for i in 0..2 {
            for j in 2..l {
                assert!(!at(i, j), "state query {i} must not see branch key {j}");
            }
        }
        for i in 2..l {
            for j in 0..2 {
                assert!(at(i, j), "branch query {i} must see state key {j}");
            }
        }
        // q1 (2..5) sees only itself; q2 (5..8) sees state + own prefix.
        assert!(!at(3, 4) && at(4, 3), "q1 causal within own branch");
        assert!(!at(6, 2) && !at(6, 4), "q2 must not see q1 tokens");
        assert!(at(6, 5) && at(7, 5), "q2 causal within own branch");
        // Diagonal always.
        for i in 0..l {
            assert!(at(i, i), "diagonal must survive");
        }
    }

    #[test]
    fn branch_mask_state_bidir_off_prefixes_the_state() {
        let seg = seg_state_branches(2, &[2]);
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, false, None, &mut m);
        // Without state_bidir the state is a causal prefix of itself.
        assert!(!m[1], "state query 0 must not see future state key 1");
        assert!(m[l], "state query 1 sees earlier state key 0");
        // Branch queries still see the whole state (it precedes them).
        assert!(m[2 * l] && m[2 * l + 1]);
    }

    #[test]
    fn branch_mask_pads_are_invisible_keys_and_dead_queries() {
        let seg = vec![S, 1, -1, -1];
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, true, None, &mut m);
        let at = |i: usize, j: usize| m[i * l + j];
        // Pads as keys: visible to nobody (a pad query's own diagonal excepted).
        for i in 0..l {
            for j in [2usize, 3] {
                if i != j {
                    assert!(!at(i, j), "pad key {j} must be blocked for query {i}");
                }
            }
        }
        // Pads as queries: only their own diagonal survives (never read).
        for j in 0..l {
            assert_eq!(at(2, j), j == 2, "pad query 2 only diagonal");
            assert_eq!(at(3, j), j == 3, "pad query 3 only diagonal");
        }
    }

    #[test]
    fn branch_mask_option_isolation() {
        // state | instruction | opt0(2) | opt1(1) | decide — one question.
        let seg = vec![S, 1, 1, 1, 1, 1];
        let opts = vec![OPT_NONE, OPT_NONE, 0, 0, 1, OPT_DECIDE];
        let l = seg.len();
        let mut m = vec![false; l * l];
        branch_mask(&seg, true, Some(&opts), &mut m);
        let at = |i: usize, j: usize| m[i * l + j];
        // opt0 tokens see state + instruction + their own EARLIER span token
        // (spans stay causal within themselves — the isolation conjunct only
        // narrows, never widens the causal base).
        assert!(at(2, 0) && at(2, 1) && !at(2, 3) && at(3, 2) && !at(2, 4));
        // opt1 sees state + instruction + itself, NOT opt0.
        assert!(at(4, 0) && at(4, 1) && !at(4, 2) && !at(4, 3));
        // decide sees everything in its question.
        for j in 0..l {
            assert!(at(5, j), "decide must see key {j}");
        }
        // instruction is not an option key: visible to everyone (causal permitting).
        assert!(at(4, 1));
        // diagonal survives isolation.
        for i in 0..l {
            assert!(at(i, i));
        }
    }

    #[test]
    fn branch_mask_rejects_bad_buffers() {
        let seg = vec![S, 1];
        let mut m = vec![false; 4];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            branch_mask(&seg, true, None, &mut m[..3])
        }));
        assert!(r.is_err(), "l*l buffer required");
        let opts = vec![OPT_NONE];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            branch_mask(&seg, true, Some(&opts), &mut m)
        }));
        assert!(r.is_err(), "opts must match seg length");
    }

    fn sample_encoding() -> PackedEncoding {
        // state(5) | q1: [instr(1), opt(2), decide] | q2: [instr(1), opt(3), opt(1), decide]
        let mut ids = Vec::with_capacity(5);
        let mut pos = Vec::with_capacity(5);
        let mut seg = Vec::with_capacity(5);
        // state
        for i in 0..5 {
            ids.push(i);
            pos.push(i);
            seg.push(S);
        }
        let mut decide_idx = Vec::new();
        let mut opt_idx = Vec::new();
        // q1 branch: len 4
        for (bi, _) in (0..3).enumerate() {
            ids.push(10 + bi);
            pos.push(5 + bi);
            seg.push(1);
        }
        ids.push(63); // the decide marker (any in-vocab id — markers are T4)
        pos.push(5 + 3);
        seg.push(1);
        decide_idx.push(ids.len() - 1);
        opt_idx.push(vec![ids.len() - 2]);
        // q2 branch: len 6
        for bi in 0..5 {
            ids.push(20 + bi);
            pos.push(5 + bi);
            seg.push(2);
        }
        ids.push(62); // q2's decide marker
        pos.push(5 + 5);
        seg.push(2);
        decide_idx.push(ids.len() - 1);
        opt_idx.push(vec![ids.len() - 4, ids.len() - 2]);
        PackedEncoding {
            ids,
            pos,
            seg,
            opt: Vec::new(),
            state_len: 5,
            state_truncated: false,
            decide_idx,
            opt_idx,
        }
    }

    #[test]
    fn rows_of_slices_and_offsets() {
        let enc = sample_encoding();
        let rows = rows_of(&enc).expect("rows");
        assert_eq!(rows.len(), 2);
        // q1: branch [5, 9), decide offset 3, opt offset 2.
        assert_eq!(rows[0].start, 5);
        assert_eq!(rows[0].end, 9);
        assert_eq!(rows[0].decide, 3);
        assert_eq!(rows[0].opts, vec![2]);
        // q2: branch [9, 15), decide offset 5, opts 2 and 4.
        assert_eq!(rows[1].start, 9);
        assert_eq!(rows[1].end, 15);
        assert_eq!(rows[1].decide, 5);
        assert_eq!(rows[1].opts, vec![2, 4]);
    }

    #[test]
    fn rows_of_refuses_layout_mismatch() {
        let mut enc = sample_encoding();
        enc.seg[9] = 1; // q2's first token claims q1's segment
        let r = rows_of(&enc);
        assert!(r.is_err(), "layout mismatch must refuse");
        let mut enc = sample_encoding();
        enc.opt_idx[1][0] = 3; // opt index outside q2's slice
        let r = rows_of(&enc);
        assert!(r.is_err(), "opt outside slice must refuse");
    }

    // Deterministic tiny random weights (LCG) for the forward parity test.
    fn tiny_config() -> crate::types::Config {
        let mut c = crate::types::Config::micro();
        c.vocab_size = 64;
        c.block_size = 256;
        // Qwen3 shape law: n_embd == n_head * head_dim (no q_dim != n_embd).
        c.n_embd = 32;
        c.n_head = 4;
        c.n_kv_head = 2;
        c.head_dim = 8;
        c.mlp_hidden = 64;
        c.n_layer = 2;
        c.rms_norm_eps = 1e-6;
        c.rope_theta = 1_000_000.0;
        c.use_rope = true;
        c.tied_embeddings = false;
        c.rms_norm_offset = false;
        c.post_norm = false;
        c.attn_logit_softcapping = 0.0;
        c.final_logit_softcapping = 0.0;
        c
    }

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
        fn mat(&mut self, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
            (0..rows * cols).map(|_| self.next_f32() * scale).collect()
        }
        fn gamma(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| 1.0 + 0.1 * self.next_f32()).collect()
        }
    }

    fn tiny_weights(config: &crate::types::Config) -> EdlmWeights {
        let mut r = Lcg(0x5EED_1005);
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(EdlmLayerWeights {
                base: LlamaLayerWeights {
                    attn_wq: r.mat(config.n_head * config.head_dim, config.n_embd, 0.2),
                    attn_wk: r.mat(config.n_kv_head * config.head_dim, config.n_embd, 0.2),
                    attn_wv: r.mat(config.n_kv_head * config.head_dim, config.n_embd, 0.2),
                    attn_wo: r.mat(config.n_embd, config.n_head * config.head_dim, 0.2),
                    gate_proj: r.mat(config.mlp_hidden, config.n_embd, 0.2),
                    up_proj: r.mat(config.mlp_hidden, config.n_embd, 0.2),
                    down_proj: r.mat(config.n_embd, config.mlp_hidden, 0.2),
                    input_norm: r.gamma(config.n_embd),
                    post_attn_norm: r.gamma(config.n_embd),
                },
                q_norm: r.gamma(config.head_dim),
                k_norm: r.gamma(config.head_dim),
            });
        }
        EdlmWeights {
            wte: r.mat(config.vocab_size, config.n_embd, 0.5),
            final_norm: r.gamma(config.n_embd),
            lm_head: None,
            layers,
            pointer: None,
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn parity_case(state_bidir: bool) -> f32 {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let l = enc.ids.len();
        let mut allow = vec![false; l * l];
        branch_mask(&enc.seg, state_bidir, None, &mut allow);
        let packed = forward_edlm_packed(&weights, &config, &enc.ids, &enc.pos, &allow);

        let rows = rows_of(&enc).expect("rows");
        let row_hiddens = forward_edlm_rows(&weights, &config, &enc, &rows, state_bidir);

        let n = config.n_embd;
        let sl = enc.state_len;
        let mut worst = 0.0f32;
        // State positions: identical (state never sees branches in either form).
        for p in 0..sl {
            let row0 = &row_hiddens[0];
            let d = max_abs_diff(&packed[p * n..(p + 1) * n], &row0[p * n..(p + 1) * n]);
            worst = worst.max(d);
        }
        // Branch positions: packed branch token (abs p) == row token (sl + off).
        for (k, row) in rows.iter().enumerate() {
            let rh = &row_hiddens[k];
            for off in 0..row.branch_len() {
                let d = max_abs_diff(
                    &packed[(row.start + off) * n..(row.start + off + 1) * n],
                    &rh[(sl + off) * n..(sl + off + 1) * n],
                );
                worst = worst.max(d);
            }
        }
        worst
    }

    #[test]
    fn packed_rows_parity_bidir() {
        let worst = parity_case(true);
        assert!(
            worst < 1e-5,
            "packed-vs-row parity exceeded tolerance: {worst}"
        );
    }

    #[test]
    fn packed_rows_parity_no_bidir() {
        let worst = parity_case(false);
        assert!(
            worst < 1e-5,
            "packed-vs-row parity (no state_bidir) exceeded tolerance: {worst}"
        );
    }

    /// Diagnostic: single-question packed-vs-row must be exact — pins whether
    /// a divergence is within-branch (forward bug) or cross-question (mask bug).
    #[test]
    fn parity_single_question_diagnostic() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        // state(3) | q1(3): ids 0..3 state, 3..6 branch; decide = 5.
        let enc = PackedEncoding {
            ids: vec![0, 1, 2, 10, 11, 63],
            pos: vec![0, 1, 2, 3, 4, 5],
            seg: vec![0, 0, 0, 1, 1, 1],
            opt: Vec::new(),
            state_len: 3,
            state_truncated: false,
            decide_idx: vec![5],
            opt_idx: vec![vec![4]],
        };
        let n = config.n_embd;
        let l = enc.ids.len();
        let mut allow = vec![false; l * l];
        branch_mask(&enc.seg, true, None, &mut allow);
        let packed = forward_edlm_packed(&weights, &config, &enc.ids, &enc.pos, &allow);
        let rows = rows_of(&enc).expect("rows");
        let row_hiddens = forward_edlm_rows(&weights, &config, &enc, &rows, true);
        let rh = &row_hiddens[0];
        let sl = enc.state_len;
        for p in 0..l {
            let d = max_abs_diff(&packed[p * n..(p + 1) * n], &rh[p * n..(p + 1) * n]);
            println!("pos {p}: diff {d}");
        }
        let _ = sl;
        let worst = max_abs_diff(&packed, rh);
        assert!(worst < 1e-5, "single-question parity: {worst}");
    }

    /// Diagnostic: two questions — prints per-position packed-vs-row diffs to
    /// pin which branch diverges (row k shares the state prefix with row 0,
    /// so any nonzero is a mask/cross-question defect, not fp noise).
    #[test]
    fn parity_two_question_diagnostic() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let n = config.n_embd;
        let l = enc.ids.len();
        let mut allow = vec![false; l * l];
        branch_mask(&enc.seg, true, None, &mut allow);
        let packed = forward_edlm_packed(&weights, &config, &enc.ids, &enc.pos, &allow);
        let rows = rows_of(&enc).expect("rows");
        let row_hiddens = forward_edlm_rows(&weights, &config, &enc, &rows, true);
        let sl = enc.state_len;
        for (k, row) in rows.iter().enumerate() {
            let rh = &row_hiddens[k];
            // state prefix of every row vs packed state
            for p in 0..sl {
                let d = max_abs_diff(&packed[p * n..(p + 1) * n], &rh[p * n..(p + 1) * n]);
                println!("row{k} state pos {p}: {d}");
            }
            for off in 0..row.branch_len() {
                let d = max_abs_diff(
                    &packed[(row.start + off) * n..(row.start + off + 1) * n],
                    &rh[(sl + off) * n..(sl + off + 1) * n],
                );
                println!("row{k} branch off {off}: {d}");
            }
        }
        // Eligible-key audit: state query 0 and q2's first branch token (abs 9).
        let elig = |q: usize| -> Vec<usize> { (0..l).filter(|&j| allow[q * l + j]).collect() };
        println!("packed state q0 eligible: {:?}", elig(0));
        println!("packed q2off0 (abs 9) eligible: {:?}", elig(9));
        let rall = row_branch_mask(sl, rows[1].branch_len(), true);
        let rl = sl + rows[1].branch_len();
        let elig_r = |q: usize| -> Vec<usize> { (0..rl).filter(|&j| rall[q * rl + j]).collect() };
        println!("row q2off0 eligible: {:?}", elig_r(sl));
        // Contamination audit: row0-state vs row1-state must be BIT-IDENTICAL
        // (state never sees either branch); and packed-state vs a standalone
        // state-only forward (l = sl, full bidir) must be too.
        let d_rows = max_abs_diff(&row_hiddens[0][..sl * n], &row_hiddens[1][..sl * n]);
        println!("row0-state vs row1-state: {d_rows}");
        let mut s_allow = vec![false; sl * sl];
        branch_mask(&enc.seg[..sl], true, None, &mut s_allow);
        let standalone =
            forward_edlm_packed(&weights, &config, &enc.ids[..sl], &enc.pos[..sl], &s_allow);
        let d_packed = max_abs_diff(&packed[..sl * n], &standalone);
        println!("packed-state vs standalone-state: {d_packed}");
        let d_row0 = max_abs_diff(&row_hiddens[0][..sl * n], &standalone);
        println!("row0-state vs standalone-state: {d_row0}");
    }

    #[test]
    fn qk_norm_matches_reference_formula() {
        let mut buf = vec![0.5, -1.25, 2.0, 0.125, 3.0, -0.5, 1.5, -2.5];
        let gamma = vec![1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.05, 0.95];
        let expected: Vec<f32> = {
            let mut e = buf.clone();
            let sum_sq: f32 = e.iter().map(|v| v * v).sum();
            let inv = 1.0 / (sum_sq / 8.0 + 1e-6).sqrt();
            for (v, &g) in e.iter_mut().zip(gamma.iter()) {
                *v *= inv * g;
            }
            e
        };
        qk_norm_inplace(&mut buf, &gamma, 1, 8, 1e-6);
        assert!(max_abs_diff(&buf, &expected) < 1e-7);
    }

    #[test]
    fn loader_env_gated_header_check() {
        let path = match std::env::var("EDLM_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "SKIP: EDLM_GGUF unset — set it to drex-dlm-Q8_0.gguf for the header check"
                );
                return;
            }
        };
        let gguf = crate::gguf_loader::GgufFile::open(std::path::Path::new(&path))
            .expect("open EDLM_GGUF");
        assert_eq!(gguf.architecture(), Some("edlm"));
        let config = edlm_config_from_gguf_metadata(&gguf).expect("config");
        assert_eq!(config.n_layer, 36);
        assert_eq!(config.n_embd, 4096);
        assert_eq!(config.n_head, 32);
        assert_eq!(config.n_kv_head, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.mlp_hidden, 12288);
        assert_eq!(config.vocab_size, 151_936);
        assert_eq!(config.block_size, 32768);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert_eq!(config.rope_theta, 1_000_000.0);
        // Pointer tensors present; temperature is F32 {1}.
        assert!(gguf.tensor_info("pointer.q.weight").is_some());
        assert!(gguf.tensor_info("pointer.k.weight").is_some());
        let temp = gguf
            .try_dequant_f16_to_f32("pointer.temperature.weight")
            .expect("temperature dequant")
            .expect("temperature present");
        assert_eq!(temp.len(), 1);
        // Norm tensors are F32 — cheap dequant smoke.
        let gamma = gguf
            .dequant_f16_to_f32("blk.0.attn_q_norm.weight")
            .expect("q norm");
        assert_eq!(gamma.len(), 128);
        assert!(gamma.iter().all(|v| v.is_finite()));
    }

    // ── T4 — marker encode/pack ────────────────────────────────────

    /// Deterministic mock: text bytes map to ids 1..=255 (0 is never a text
    /// id), the five markers to 900..905. Layout tests need no real vocab.
    struct MockTok;

    impl EdlmEncodeTokenizer for MockTok {
        fn encode_text(&self, text: &str) -> Vec<usize> {
            text.bytes().map(|b| b as usize + 1).collect()
        }

        fn token_to_id(&self, token: &str) -> Option<usize> {
            EDLM_MARKERS
                .iter()
                .position(|&m| m == token)
                .map(|i| 900 + i)
        }
    }

    fn mock_record() -> EdlmRecord {
        EdlmRecord {
            state: "abc".into(),
            questions: vec![
                EdlmQuestion {
                    instr: "I1".into(),
                    options: vec!["o1a".into(), "o1b".into()],
                },
                EdlmQuestion {
                    instr: "I2".into(),
                    options: vec!["o2".into()],
                },
            ],
        }
    }

    #[test]
    fn encode_packed_layout_law() {
        let enc = encode_packed(&MockTok, &mock_record(), EdlmLimits::default(), true, false)
            .expect("encode");
        enc.validate().expect("invariants");
        // [marker-state a b c] — state ids, positions 0-based, seg 0.
        assert_eq!(&enc.ids[..4], &[900, 98, 99, 100]);
        assert_eq!(enc.state_len, 4);
        assert_eq!(&enc.pos[..4], &[0, 1, 2, 3]);
        assert!(enc.seg[..4].iter().all(|&g| g == 0));
        // q1 branch: [marker-q I 1 marker-opt o 1 a marker-/opt marker-opt o 1 b marker-/opt marker-decide]
        let q1 = &enc.ids[4..];
        assert_eq!(q1[0], 901);
        assert_eq!(q1[3], 902); // span-0 open
        assert_eq!(q1[7], 903); // span-0 close
        assert_eq!(q1[8], 902); // span-1 open
        assert_eq!(q1[12], 903); // span-1 close
        assert_eq!(*q1.last().unwrap(), 904); // decide
        // q1's decide at 4+13; q2's branch (8 tokens) ends the encoding.
        assert_eq!(enc.decide_idx, vec![4 + 13, enc.ids.len() - 1]);
        // q1's two </opt> keys: after span 0 and after span 1 (spans count
        // their open/close markers: 3 text bytes → 5-token spans).
        assert_eq!(enc.opt_idx[0], vec![4 + 7, 4 + 12]);
        // Branch positions restart at state_len, contiguous in the plain form.
        assert_eq!(enc.pos[4], 4);
        assert_eq!(enc.pos[enc.decide_idx[0]], 17); // 4 + q1's 13 branch tokens before decide
        assert_eq!(enc.pos[*enc.decide_idx.last().unwrap()], 11); // q2: 4 + 7 prior branch tokens
        // seg: state 0, q1 = 1, q2 = 2, one contiguous block per question.
        assert_eq!(enc.seg[4], 1);
        assert_eq!(*enc.seg.last().unwrap(), 2);
        assert_eq!(enc.seg[enc.decide_idx[0]], 1);
        // No isolation: opt is empty (mask takes opts=None).
        assert!(enc.opt.is_empty());
        assert!(!enc.state_truncated);
    }

    #[test]
    fn encode_packed_two_questions_blocks_and_readouts() {
        let enc = encode_packed(&MockTok, &mock_record(), EdlmLimits::default(), true, false)
            .expect("encode");
        let rows = rows_of(&enc).expect("rows");
        assert_eq!(rows.len(), 2);
        // Row 0: state(4) + q1 branch; decide at 4 + (q1 branch len - 1).
        let q1_len = enc.decide_idx[0] + 1 - enc.state_len;
        assert_eq!(rows[0].decide, q1_len - 1);
        assert_eq!(rows[0].opts, vec![7, 12]); // within-branch </opt> offsets
        // Row 1 starts right after q1's decide.
        assert_eq!(rows[1].start, enc.decide_idx[0] + 1);
        assert_eq!(rows[1].decide, enc.ids.len() - 1 - rows[1].start);
    }

    #[test]
    fn encode_state_overflow_strict_refuses_and_non_strict_discloses() {
        let rec = EdlmRecord {
            state: "x".repeat(10),
            questions: vec![EdlmQuestion {
                instr: "i".into(),
                options: vec!["o".into()],
            }],
        };
        // max_state 5 = marker + 4 text tokens.
        let limits = EdlmLimits {
            max_state: 5,
            ..EdlmLimits::default()
        };
        let err = encode_packed(&MockTok, &rec, limits, true, false).unwrap_err();
        let overflow = err
            .downcast_ref::<ContextOverflow>()
            .expect("ContextOverflow");
        assert!(overflow.detail.contains("state exceeds 5 tokens: 11"));
        // Non-strict: crops to the limit and DISCLOSES.
        let enc = encode_packed(&MockTok, &rec, limits, false, false).expect("encode");
        assert_eq!(enc.state_len, 5);
        assert!(enc.state_truncated);
        enc.validate().expect("invariants");
    }

    #[test]
    fn encode_branch_overflow_refuses_in_both_modes() {
        let rec = EdlmRecord {
            state: "s".into(),
            questions: vec![EdlmQuestion {
                instr: "i".into(),
                options: vec!["o".into()],
            }],
        };
        // state = 2 tokens; branch (marker + i + 3 span tokens + decide = 6)
        // must exceed max_branch - 2.
        let limits = EdlmLimits {
            max_state: 16,
            max_branch: 7,
            max_packed: 64,
        };
        for strict in [true, false] {
            let err = encode_packed(&MockTok, &rec, limits, strict, false).unwrap_err();
            let overflow = err
                .downcast_ref::<ContextOverflow>()
                .expect("ContextOverflow");
            assert!(overflow.detail.contains("branch too long"));
        }
    }

    #[test]
    fn encode_option_isolation_positions_law() {
        // instr = 2 branch tokens (marker + 1 text); spans 5 and 3 (markers
        // count); longest span = 5.
        let rec = EdlmRecord {
            state: "ab".into(),
            questions: vec![EdlmQuestion {
                instr: "i".into(),
                options: vec!["ooo".into(), "o".into()],
            }],
        };
        let enc = encode_packed(&MockTok, &rec, EdlmLimits::default(), true, true).expect("encode");
        enc.validate().expect("invariants");
        let sl = enc.state_len; // 3 (marker + 2 text)
        let base = sl; // the branch starts here
        let instr_len = 2; // q marker + 1 text token
        // Instruction positions: sl..sl+instr_len.
        assert_eq!(&enc.pos[base..base + instr_len], &[sl, sl + 1]);
        // EVERY span token — open/close markers included — shares the same
        // position ids: span token i sits at sl+instr_len+i regardless of
        // which span it belongs to. span0 = 5 tokens, span1 = 3.
        assert_eq!(
            &enc.pos[base + 2..base + 7],
            &[sl + 2, sl + 3, sl + 4, sl + 5, sl + 6]
        );
        assert_eq!(&enc.pos[base + 7..base + 10], &[sl + 2, sl + 3, sl + 4]);
        // Decide sits at ONE fixed position after the longest span.
        let d = enc.decide_idx[0];
        assert_eq!(d, base + 10);
        assert_eq!(enc.pos[d], sl + instr_len + 5);
        // opt indices: instruction OPT_NONE, spans 0/1, decide OPT_DECIDE.
        assert_eq!(enc.opt[base], OPT_NONE);
        assert_eq!(enc.opt[base + 2], 0);
        assert_eq!(enc.opt[base + 6], 0);
        assert_eq!(enc.opt[base + 7], 1);
        assert_eq!(enc.opt[base + 9], 1);
        assert_eq!(enc.opt[d], OPT_DECIDE);
    }

    #[test]
    fn defuse_special_spellings_reference_semantics() {
        // A well-formed delimiter is rewritten to the broken-bar spelling.
        assert_eq!(
            defuse_special_spellings("<|fim_prefix|>"),
            "<\u{a6}fim_prefix\u{a6}>"
        );
        assert_eq!(
            defuse_special_spellings("a<|box_start|>b"),
            "a<\u{a6}box_start\u{a6}>b"
        );
        // Names allow [A-Za-z0-9_] only.
        assert_eq!(defuse_special_spellings("<|not-a-name|>"), "<|not-a-name|>");
        assert_eq!(defuse_special_spellings("<|has space|>"), "<|has space|>");
        // Unterminated / empty names stay verbatim.
        assert_eq!(defuse_special_spellings("<|open|>"), "<\u{a6}open\u{a6}>");
        assert_eq!(defuse_special_spellings("<||>"), "<||>");
        assert_eq!(defuse_special_spellings("<|tail"), "<|tail");
        // Plain text and multi-byte chars pass through untouched.
        assert_eq!(defuse_special_spellings("héllo → wörld"), "héllo → wörld");
        assert_eq!(defuse_special_spellings(""), "");
        // Both runs rewrite; a non-name char anywhere breaks the match.
        assert_eq!(
            defuse_special_spellings("x <|a_1|> <|no!|>"),
            "x <\u{a6}a_1\u{a6}> <|no!|>"
        );
    }

    #[test]
    fn user_text_cannot_forge_delimiters() {
        // An option whose text embeds a delimiter spelling must not produce
        // any marker id in the packed encoding.
        let rec = EdlmRecord {
            state: "<|fim_suffix|> evil <|box_end|>".into(),
            questions: vec![EdlmQuestion {
                instr: "<|fim_middle|>".into(),
                options: vec!["<|fim_prefix|>".into()],
            }],
        };
        let enc =
            encode_packed(&MockTok, &rec, EdlmLimits::default(), true, false).expect("encode");
        let markers: [usize; 5] = [900, 901, 902, 903, 904];
        // Marker ids appear EXACTLY at the structural positions: 1 (state),
        // then per branch [q, opt, /opt, decide] — nowhere else.
        let structural: Vec<usize> = std::iter::once(enc.ids[0])
            .chain([901, 902, 903, 904])
            .collect();
        let found: Vec<usize> = enc
            .ids
            .iter()
            .copied()
            .filter(|id| markers.contains(id))
            .collect();
        assert_eq!(found, structural, "marker ids only at structural positions");
        // The defused spellings tokenized as ordinary bytes (broken bar is
        // 0xC2 0xA6 in UTF-8 → mock ids 195/167).
        assert!(enc.ids.contains(&195));
    }

    #[test]
    fn encode_refuses_empty_options_and_missing_markers() {
        let rec = EdlmRecord {
            state: "s".into(),
            questions: vec![EdlmQuestion {
                instr: "i".into(),
                options: vec![],
            }],
        };
        assert!(encode_packed(&MockTok, &rec, EdlmLimits::default(), true, false).is_err());
        // A vocab without the markers refuses loudly (never silently packs).
        struct NoMarkers;
        impl EdlmEncodeTokenizer for NoMarkers {
            fn encode_text(&self, text: &str) -> Vec<usize> {
                text.bytes().map(|b| b as usize).collect()
            }
            fn token_to_id(&self, _token: &str) -> Option<usize> {
                None
            }
        }
        let ok_rec = mock_record();
        assert!(encode_packed(&NoMarkers, &ok_rec, EdlmLimits::default(), true, false).is_err());
    }

    /// The published sample (`examples/request.json`), rendered exactly as
    /// their `api.py::to_record` renders it — the T6 parity fixture.
    fn published_sample_record() -> EdlmRecord {
        EdlmRecord {
            state:
                "ticket: I was charged twice for the same order. Please refund the extra payment."
                    .into(),
            questions: vec![
                EdlmQuestion {
                    instr: "Which team should handle this ticket?".into(),
                    options: vec![
                        "billing: Payments, charges, and refunds".into(),
                        "technical: Bugs and outages".into(),
                        "other: Anything else".into(),
                    ],
                },
                EdlmQuestion {
                    instr: "Does the customer explicitly ask for a refund?".into(),
                    options: vec!["no".into(), "yes".into()],
                },
                EdlmQuestion {
                    instr: "How urgent is this ticket?".into(),
                    options: vec!["Routine".into(), "Soon".into(), "Urgent".into()],
                },
            ],
        }
    }

    /// T6 — end-to-end parity against their published sample outputs (README:
    /// billing 0.9641 / technical 0.001 / other 0.0349; noul 0.8548;
    /// 0.1876 / 0.2169 / 0.5955; 87 input tokens), on the REAL Q8_0 GGUF.
    ///
    /// Tolerance law: argmax EXACT (the answers a consumer publishes) +
    /// per-option drift < 0.015. Their own cross-runner spread against the
    /// BF16-published numbers is 0.0050 on this sample and 0.0094 on their
    /// 255-option case (validation/RESULTS.md); MEASURED here (2026-10-08,
    /// Q8_0, state_bidir=true): urgency max drift 0.0103, team/refund under
    /// 0.01, all three argmaxes exact — same tolerance class as their own
    /// GGUF runners, one compute path over (ours dequants Q8_0 to f32 and
    /// computes fp32 throughout; their native computes f16 accum over the
    /// same blocks). The 87-token encode is an EXACT pin — no tolerance.
    ///
    /// Posture: the RELEASED checkpoint posture (`state_bidir = true` —
    /// their checkpoint meta default), plain packed mask (no option
    /// isolation), strict serve limits.
    ///
    /// Skip-loud without EDLM_GGUF (CC BY-NC weights, local bench only).
    #[test]
    fn sample_request_end_to_end_parity_env_gated() {
        let path = match std::env::var("EDLM_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "SKIP: EDLM_GGUF unset — set it to drex-dlm-Q8_0.gguf for the T6 parity run"
                );
                return;
            }
        };
        let model = EdlmGgufModel::open(std::path::Path::new(&path)).expect("open edlm gguf");
        let tok = model.tokenizer().expect("gguf tokenizer");

        // Marker ids (the distilled reuse map — pinned, never assumed).
        let marker_ids: Vec<usize> = EDLM_MARKERS
            .iter()
            .map(|m| tok.token_to_id(m).expect("marker in vocab"))
            .collect();
        assert_eq!(
            marker_ids,
            vec![151659, 151660, 151648, 151649, 151661],
            "the five reused Qwen special-token ids"
        );

        let enc = encode_packed(
            &tok,
            &published_sample_record(),
            EdlmLimits::serving(16_384),
            true,
            false,
        )
        .expect("encode");
        enc.validate().expect("invariants");
        assert_eq!(enc.ids.len(), 87, "their published input-token count");
        assert_eq!(enc.decide_idx.len(), 3);

        let allow = {
            let mut m = vec![false; enc.ids.len() * enc.ids.len()];
            branch_mask(&enc.seg, true, None, &mut m);
            m
        };
        let hidden =
            forward_edlm_packed_streaming(&model, &enc.ids, &enc.pos, &allow).expect("forward");

        let pointer = model.pointer.as_ref().expect("pointer head in gguf");
        let n = model.config.n_embd;
        let read = |q: usize| -> Vec<f32> {
            let d = &hidden[enc.decide_idx[q] * n..(enc.decide_idx[q] + 1) * n];
            let mut opts = Vec::with_capacity(enc.opt_idx[q].len() * n);
            for &o in &enc.opt_idx[q] {
                opts.extend_from_slice(&hidden[o * n..(o + 1) * n]);
            }
            pointer.question_probs(d, &opts, enc.opt_idx[q].len())
        };
        let team = read(0);
        let refund = read(1);
        let urgency = read(2);

        let published: [&[f64]; 3] = [
            &[0.9641, 0.001, 0.0349],  // team: billing / technical / other
            &[1.0 - 0.8548, 0.8548],   // refund: no / yes
            &[0.1876, 0.2169, 0.5955], // urgency: Routine / Soon / Urgent
        ];
        let mut worst: (f64, &'static str) = (0.0, "");
        for (name, got, want) in [
            ("team", &team, published[0]),
            ("refund", &refund, published[1]),
            ("urgency", &urgency, published[2]),
        ] {
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                let d = (*g as f64 - w).abs();
                if d > worst.0 {
                    worst = (d, name);
                }
                assert!(
                    d < 0.015,
                    "parity drift on {name}: got {got:?} want {want:?} (max {d})"
                );
            }
        }
        println!(
            "parity: team {team:?} refund {refund:?} urgency {urgency:?} — worst drift {:.4} ({})",
            worst.0, worst.1
        );
        // Argmax pins (the answers a consumer would publish).
        assert_eq!(
            team.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            0,
            "billing"
        );
        assert_eq!(
            refund
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            1,
            "yes"
        );
        assert_eq!(
            urgency
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            2,
            "Urgent"
        );

        // Packed-vs-row on REAL weights: the Phase 1 structural law must hold
        // where it matters — the shipped checkpoint. (Row form mirrors the
        // packed posture: state_bidir = true here.)
        let rows = rows_of(&enc).expect("rows");
        let row_hiddens =
            forward_edlm_rows_streaming(&model, &enc, &rows, true).expect("row forward");
        for (k, row) in rows.iter().enumerate() {
            let h = &row_hiddens[k];
            let off = |idx_in_branch: usize| (enc.state_len + idx_in_branch) * n;
            let d_row = &h[off(row.decide)..off(row.decide) + n];
            let d_pack = &hidden[enc.decide_idx[k] * n..(enc.decide_idx[k] + 1) * n];
            let diff: f32 = d_row
                .iter()
                .zip(d_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                diff < 1e-4,
                "question {k}: packed-vs-row hidden diff {diff}"
            );
            let mut opts_row = Vec::new();
            for &o in &row.opts {
                opts_row.extend_from_slice(&h[off(o)..off(o) + n]);
            }
            let probs_row = pointer.question_probs(d_row, &opts_row, row.opts.len());
            let probs_pack = read(k);
            let pdiff: f32 = probs_row
                .iter()
                .zip(probs_pack.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                pdiff < 1e-3,
                "question {k}: packed-vs-row prob diff {pdiff}"
            );
        }
    }

    // ── T7 — state-prefix KV reuse ─────────────────────────────────

    #[test]
    fn state_prefill_matches_standalone_state_forward() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let sl = enc.state_len;
        for bidir in [true, false] {
            let cache =
                edlm_state_prefill(&weights, &config, &enc.ids[..sl], &enc.pos[..sl], bidir);
            let mut s_allow = vec![false; sl * sl];
            branch_mask(&enc.seg[..sl], bidir, None, &mut s_allow);
            let standalone =
                forward_edlm_packed(&weights, &config, &enc.ids[..sl], &enc.pos[..sl], &s_allow);
            assert_eq!(cache.state_len, sl);
            assert_eq!(cache.kvd, crate::types::kv_dim(&config));
            assert_eq!(cache.layers.len(), config.n_layer);
            assert_eq!(
                cache.hidden, standalone,
                "prefill state hiddens must be bit-identical to the standalone state forward \
                 (bidir={bidir})"
            );
        }
    }

    #[test]
    fn branch_continuation_allow_law() {
        let allow = branch_continuation_allow(2, 3);
        // l = 5; branch queries are rows 0..3 (branch-local), state keys 0..2.
        let at = |p: usize, j: usize| allow[p * 5 + j];
        // Every branch query sees every state key.
        for p in 0..3 {
            for j in 0..2 {
                assert!(
                    at(p, j),
                    "state key {j} must be visible to branch query {p}"
                );
            }
        }
        // Causal within the branch: query p sees branch keys ..=p, nothing later.
        assert!(at(0, 2) && !at(0, 3) && !at(0, 4));
        assert!(at(1, 2) && at(1, 3) && !at(1, 4));
        assert!(at(2, 2) && at(2, 3) && at(2, 4));
    }

    #[test]
    fn cached_rows_bit_identical_to_plain() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let rows = rows_of(&enc).expect("rows");
        for bidir in [true, false] {
            let plain = forward_edlm_rows(&weights, &config, &enc, &rows, bidir);
            // Forced cache: the tiny state (5 tokens) is far below the 384
            // crossover, so prefix_min_tokens = 0 selects the cache path.
            let cached = forward_edlm_rows_cached(&weights, &config, &enc, &rows, bidir, 0);
            assert_eq!(
                cached, plain,
                "cached rows must be BIT-identical to the plain row form (bidir={bidir})"
            );
            // The crossover wiring: below the threshold the one-call form is
            // the plain path.
            let below = forward_edlm_rows_cached(&weights, &config, &enc, &rows, bidir, usize::MAX);
            assert_eq!(
                below, plain,
                "below-threshold rows must equal plain (bidir={bidir})"
            );
        }
    }

    #[test]
    fn stale_state_cache_refuses() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let sl = enc.state_len;
        let rows = rows_of(&enc).expect("rows");
        let cache = edlm_state_prefill(&weights, &config, &enc.ids[..sl], &enc.pos[..sl], true);
        // Different state CONTENT, same length: must refuse, never serve.
        let mut swapped = sample_encoding();
        swapped.ids[0] = (swapped.ids[0] + 1) % config.vocab_size;
        let r = forward_edlm_branches_cached(&weights, &config, &swapped, &rows, &cache);
        assert!(
            r.is_err(),
            "a same-length different-state cache must refuse"
        );
        // Different state LENGTH: must refuse.
        let mut longer = sample_encoding();
        longer.ids.insert(0, 7);
        longer.pos.insert(0, 0);
        longer.seg.insert(0, 0);
        longer.state_len += 1;
        let cache2 = edlm_state_prefill(
            &weights,
            &config,
            &longer.ids[..longer.state_len],
            &longer.pos[..longer.state_len],
            true,
        );
        let r = forward_edlm_branches_cached(&weights, &config, &enc, &rows, &cache2);
        assert!(r.is_err(), "a different state_len cache must refuse");
        // The matching cache passes the check (control arm).
        let r = forward_edlm_branches_cached(&weights, &config, &enc, &rows, &cache);
        assert!(r.is_ok(), "the matching cache must be accepted");
    }

    #[test]
    fn rows_cached_empty_rows_short_circuit() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let enc = sample_encoding();
        let out = forward_edlm_rows_cached(&weights, &config, &enc, &[], true, 0);
        assert!(out.is_empty(), "no rows -> no output, no prefill");
    }

    /// T7 on the REAL Q8_0 weights: cached-vs-plain row parity (bit-identical
    /// — the exactness claim, proven not tolerated) plus the reuse timing.
    /// The published sample's state is ~35 tokens — BELOW the 384 crossover —
    /// so this exercises the small-state posture honestly; the numbers are
    /// disclosed, never asserted (box state is part of every latency claim).
    /// Skip-loud without EDLM_GGUF (CC BY-NC weights, local bench only).
    #[test]
    fn state_prefix_cache_real_weights_env_gated() {
        let path = match std::env::var("EDLM_GGUF") {
            Ok(p) if !p.is_empty() => p,
            _ => {
                eprintln!(
                    "SKIP: EDLM_GGUF unset — set it to drex-dlm-Q8_0.gguf for the T7 parity run"
                );
                return;
            }
        };
        let model = EdlmGgufModel::open(std::path::Path::new(&path)).expect("open edlm gguf");
        let tok = model.tokenizer().expect("gguf tokenizer");
        let enc = encode_packed(
            &tok,
            &published_sample_record(),
            EdlmLimits::serving(16_384),
            true,
            false,
        )
        .expect("encode");
        let rows = rows_of(&enc).expect("rows");

        let t0 = std::time::Instant::now();
        let plain = forward_edlm_rows_streaming(&model, &enc, &rows, true).expect("plain rows");
        let plain_t = t0.elapsed();

        let t1 = std::time::Instant::now();
        let cache = edlm_state_prefill_streaming(
            &model,
            &enc.ids[..enc.state_len],
            &enc.pos[..enc.state_len],
            true,
        )
        .expect("prefill");
        let prefill_t = t1.elapsed();
        let t2 = std::time::Instant::now();
        let cached =
            forward_edlm_branches_cached_streaming(&model, &enc, &rows, &cache).expect("branches");
        let branches_t = t2.elapsed();

        for (k, row) in rows.iter().enumerate() {
            let _ = row;
            assert_eq!(
                cached[k], plain[k],
                "question {k}: cached row hiddens must be BIT-identical to plain"
            );
        }
        let cached_t = prefill_t + branches_t;
        println!(
            "state-prefix reuse: state {} tokens x {} questions — plain {:?} vs cached {:?} \
             (prefill {:?} + branches {:?}, {:.2}x; crossover {} -> {} path)",
            enc.state_len,
            rows.len(),
            plain_t,
            cached_t,
            prefill_t,
            branches_t,
            plain_t.as_secs_f64() / cached_t.as_secs_f64().max(1e-9),
            EDLM_PREFIX_MIN_TOKENS,
            if enc.state_len >= EDLM_PREFIX_MIN_TOKENS {
                "cache"
            } else {
                "plain"
            },
        );
    }
}
