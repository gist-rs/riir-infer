//! Issue 1156 T2 / Plan 624 — the block-sparse prefill attention arm
//! (XAttention-class diagonal-band scoring + pooled-block probe, static τ
//! posture, sink guarantee).
//!
//! ## Why
//!
//! T1 (bench_734 stage decomposition, league posture, amo) measured the
//! attention share of hybrid prefill at **24.6% @16K / 39.3% @32K** — the
//! O(P²) term rising with P — and the serving arm at those lengths on macOS
//! is the m32 tiled flash kernel (DEFAULT-ON macOS, `TILED_FLASH_M32_MIN_P`
//! 8192, routed on the full prompt length; the tiled/m16/cmma counters read
//! zero because m32 precedes them in the ladder). The sparse arm slots ABOVE
//! the whole dense ladder and must beat m32, not a legacy path.
//!
//! ## Design (training-free, static per-model posture — no router)
//!
//! Four kernels per (attention layer, chunk):
//!
//! 1. **Pool** — per (KV group, 64-row KV block): mean K vector, written once
//!    per layer call; the score kernel consumes it lane-locally.
//! 2. **Score** — per (head, 128-query window, KV block ≤ the window's
//!    diagonal horizon): the block's attention-mass estimate
//!    `score = band + probe` (the probe is scorer-dependent):
//!    - `band`: the sum of the block's cells ON the global causal diagonal
//!      (the XAttention antidiagonal idea generalized to rectangular blocks;
//!      every band cell visited exactly once — no undersampling). Catches the
//!      local/streaming mass. Key-index shortcut: a band cell for window-local
//!      query `x` is the key row `q_start + x` (the main diagonal), so the
//!      kernel needs no per-term branch.
//!    - `Envelope` (v2 DEFAULT): the Quest-class bound `Σ_d q̄_d·(q̄_d ≥ 0 ?
//!      kmax_d : kmin_d)` ≥ `max_r(q̄·k_r)` — the block's best single row,
//!      upper-bounded per dim from the min/max pool. A single fact row
//!      survives at full magnitude on its dims instead of being averaged
//!      into a 64-row mean.
//!    - `Mean` (v1 demote): ⟨mean Q window, mean K block⟩ (the FlexPrefill
//!      pooled probe). T3's verdict measured its named dilution risk: a
//!      single needle row inside a 64-row block washes out in the mean and
//!      the τ-threshold dropped the retrieval-critical blocks first — 6/7
//!      families ΩMSR = None (the v1 manifest is the record; the envelope
//!      is the re-open lever, plan 625 §Protocol v2).
//! 3. **Select** — per (head, query window): pick the threshold θ by a fixed
//!    28-iteration f32 bisection on `mass(θ) = Σ_{score ≥ θ} score` so the
//!    selected set carries ≥ τ of the row's total score (the same set
//!    ranked-prefix selection gives, without an in-kernel sort; deterministic).
//!    `τ = 1.0` is the IDENTITY posture: every in-horizon block is selected
//!    regardless of score, the skip branch goes dead, and outputs are
//!    **bit-identical to m32** (the G3 anchor). Sink blocks {0, 1} (the first
//!    128 tokens) are ALWAYS selected.
//! 4. **Sparse attention** — the m32 body VERBATIM plus one addition: at
//!    KV-block boundaries test the mask bit and jump `pos += 64` when unset.
//!    The decision is whole-cube uniform (every 32-row Q tile lies inside one
//!    128-row mask window; the launcher passes the chunk's absolute mask-row
//!    base). Skipping causal-masked or unselected rows is a bit-exact no-op
//!    for the online-softmax state (w = 0, corr = 1 — the m32 doc's own
//!    widened-n_loop argument).
//!
//! All four are one CubeCL launch each per (layer, chunk) — at the league
//! posture (15 attention layers × 8 chunks at 32K) that is 60 extra launches
//! against ~71 s of prefill.
//!
//! ## Posture (the repo's env-arm vocabulary)
//!
//! - `RIIR_PREFILL_SPARSE` = `1|on|true` arms the lane (exact-literal;
//!   junk/unset = OFF — the Issue-750-T3 lossy law: opt-in, never default
//!   without the GOAT).
//! - `RIIR_PREFILL_SPARSE_TAU` f32 (default 0.9, clamped [0.5, 1.0]).
//! - `RIIR_PREFILL_SPARSE_SCORER` = `envelope` (protocol v2 DEFAULT) or the
//!   exact literal `mean` (v1 demote); junk/unset → envelope.
//! - `RIIR_PREFILL_SPARSE_DENSE_HEADS` (T5, Issue 1156) = a comma list of
//!   indices and inclusive ranges (`"0-3,7,12"`) naming heads that run
//!   DENSE inside the sparse launch — their cubes never take the skip
//!   branch (mixed-mode execution, one launch). Junk demotes the WHOLE knob
//!   (one loud line, no partial lists); heads ≥ 62 can't ride the bitmask
//!   and refuse the same way. Unset = the incumbent all-sparse posture.
//! - `RIIR_PREFILL_REGIME_GATE` + `RIIR_REGIME_CENTROIDS` +
//!   `RIIR_OMSR_MANIFEST` (T4, Issue 1156 — see [`crate::regime_gate`]):
//!   the modelless regime gate reads the chunk's boundary hidden states and
//!   overrides the chunk's τ / sparse-vs-dense routing within the T3
//!   manifest's floors. INERT until the freeze pass lands; a failed load
//!   falls back to the plain arm (one loud line + a refusal counter).
//! - Engagement (checked once per chunk at scratch-build):
//!   `enabled && head_dim == 256 && active_prefill_len >= PREFILL_SPARSE_MIN_P`
//!   (the prompt-level gate — the Issue-782 lesson; `reset_state` zeroes
//!   `active_prefill_len` so decode never misroutes).
//! - Launch counters (the vacuous-guard pattern): `sparse_pool_launch_count`
//!   / `sparse_score_launch_count` / `sparse_select_launch_count` /
//!   `sparse_attn_launch_count` — outputs are lossy, so ONLY the counters
//!   prove the toggle reached a dispatch.
//!
//! ## Gates (pre-registered, Plan 624)
//!
//! - G1: τ=1.0 bit-identity vs m32 (the kernel test here + the e2e anchor in
//!   the quality walk); τ<1.0 reads ride the T2.4 walk (per-family, never
//!   aggregate-only).
//! - G3: env unset → no branch of the dispatch chain changes, no scratch
//!   allocated (`Option::None`).
//! - G4: scratch rides the per-chunk `BatchedAttentionScratch` pool, no new
//!   deps.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Tokens per KV block (the FlashMemory/BSA block vocabulary; Research 436).
#[cfg(feature = "cubecl_runtime")]
pub const SPARSE_KV_BLOCK: usize = 64;

/// Queries per mask window. Every m32 Q tile (32 rows) lies inside one
/// window, so the attention kernel's skip decision is whole-cube uniform.
#[cfg(feature = "cubecl_runtime")]
pub const SPARSE_Q_BLOCK: usize = 128;

/// Always-selected blocks at the sequence start (sink 128 tokens = blocks
/// {0, 1} — the streaming/sink pattern of the BSA config).
#[cfg(feature = "cubecl_runtime")]
pub const SPARSE_SINK_BLOCKS: u32 = 2;

/// Minimum prompt length for the sparse arm to route (the
/// `PREFILL_Q8KV_MIN_P` slot precedent — the ≥16K regime where T1 measured
/// the attention share at/above the go signal).
#[cfg(feature = "cubecl_runtime")]
pub const PREFILL_SPARSE_MIN_P: usize = 16384;

// ---------------------------------------------------------------------------
// Posture: env arms + counters (the cmma enabled-fn + vacuous-guard patterns)
// ---------------------------------------------------------------------------

#[cfg(feature = "cubecl_runtime")]
static SPARSE_INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(feature = "cubecl_runtime")]
static SPARSE_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(feature = "cubecl_runtime")]
static SPARSE_TAU_BITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[cfg(feature = "cubecl_runtime")]
static SPARSE_POOL_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "cubecl_runtime")]
static SPARSE_SCORE_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "cubecl_runtime")]
static SPARSE_SELECT_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "cubecl_runtime")]
static SPARSE_ATTN_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Whether the block-sparse prefill arm is active (DEFAULT OFF — the lossy
/// law). Env is read exactly once — the first caller wins the store and the
/// env-derived value IS the live value on that call (the Bench 805 protocol:
/// no AtomicBool divergence between the env read and the live value).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_sparse_enabled() -> bool {
    let env_enabled = matches!(
        std::env::var("RIIR_PREFILL_SPARSE")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str(),
        "1" | "on" | "true",
    );
    if SPARSE_INITIALIZED
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_ok()
    {
        SPARSE_ENABLED.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    SPARSE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The selection mass threshold τ (default 0.9, clamped [0.5, 1.0]). τ = 1.0
/// is the identity posture: every in-horizon block selected, the skip branch
/// dead, outputs bit-identical to m32.
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_sparse_tau() -> f32 {
    let mut parsed = std::env::var("RIIR_PREFILL_SPARSE_TAU")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .map_or(0.9f32, |v| v.clamp(0.5, 1.0));
    let bits = parsed.to_bits();
    if SPARSE_TAU_BITS
        .compare_exchange(
            0,
            bits,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        parsed = f32::from_bits(SPARSE_TAU_BITS.load(std::sync::atomic::Ordering::Relaxed));
    }
    parsed
}

/// The block scorer family (Issue 1156 T3-v2, plan 625 §Protocol v2) —
/// [`OmegaScorer`] re-exported under the arm's historical name. `Mean` is
/// the v1 pooled-centroid scorer (T3 verdict: 6/7 families ΩMSR = None; the
/// demote posture, its floors live in the committed v1 manifest);
/// `Envelope` is the Quest-class min/max envelope: kernel 1 pools per-dim
/// max + min K per block, kernel 2 scores
/// `band + Σ_d q̄_d·(q̄_d ≥ 0 ? kmax_d : kmin_d)` — an upper bound on
/// `max_r(q̄·k_r)` over the block's rows, tight exactly on single-fact rows
/// (the facts-first fix for the T3 finding: a single needle row carries no
/// pooled-centroid mass and was dropped first at every τ ≤ 0.95).
pub use crate::sparse_omega_msr::OmegaScorer as SparseScorer;

#[cfg(feature = "cubecl_runtime")]
static SPARSE_SCORER_BITS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The block scorer (protocol v2 DEFAULT: `Envelope`; the exact literal
/// `RIIR_PREFILL_SPARSE_SCORER=mean` demotes to the v1 pooled-centroid
/// scorer — junk/unset → envelope). Env read exactly once — the first caller
/// wins the store (the τ protocol).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_sparse_scorer() -> SparseScorer {
    let mut scorer = match std::env::var("RIIR_PREFILL_SPARSE_SCORER")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        .as_str()
    {
        "mean" => SparseScorer::Mean,
        _ => SparseScorer::Envelope,
    };
    let bits = scorer.as_u32() + 1; // +1: 0 is the uninitialized sentinel
    if SPARSE_SCORER_BITS
        .compare_exchange(
            0,
            bits,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        scorer = SparseScorer::from_bits(
            SPARSE_SCORER_BITS.load(std::sync::atomic::Ordering::Relaxed) - 1,
        );
    }
    scorer
}

/// Test/demo setter for the scorer (the `set_prefill_sparse_tau` precedent).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_sparse_scorer(scorer: SparseScorer) {
    SPARSE_SCORER_BITS.store(scorer.as_u32() + 1, std::sync::atomic::Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// T5 — mixed-mode: the per-head DENSE bitmask (Issue 1156, plan 625 T5)
// ---------------------------------------------------------------------------

/// Bit 63 of the packed head mask = "initialized" (the first-caller-wins
/// sentinel); heads live in bits 0..=62 — a model with more heads refuses
/// the env (fail-closed), it is never silently truncated.
#[cfg(feature = "cubecl_runtime")]
const SPARSE_DENSE_HEADS_INIT: u64 = 1u64 << 63;
#[cfg(feature = "cubecl_runtime")]
const SPARSE_DENSE_HEADS_MAX_HEAD: usize = 62;
#[cfg(feature = "cubecl_runtime")]
static SPARSE_DENSE_HEADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Parse `RIIR_PREFILL_SPARSE_DENSE_HEADS` — a comma list of head indices
/// and inclusive ranges (`"0-3,7,12"`, whitespace tolerated). Junk ANYWHERE
/// is an Err: a typo silently ignoring half the list is exactly the failure
/// class the knob must not have, so the whole spec demotes. Pure (ungated
/// module consumers: tests + the env read).
pub fn parse_dense_head_mask(spec: &str) -> Result<u64, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(0);
    }
    let mut mask = 0u64;
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("empty element in the head list".into());
        }
        let (lo_s, hi_s) = match part.split_once('-') {
            Some((lo, hi)) => (lo.trim(), Some(hi.trim())),
            None => (part, None),
        };
        let lo: usize = lo_s
            .parse()
            .map_err(|_| format!("bad head index `{lo_s}`"))?;
        let hi = match hi_s {
            Some(h) => h
                .parse::<usize>()
                .map_err(|_| format!("bad head index `{h}`"))?,
            None => lo,
        };
        if hi < lo {
            return Err(format!("inverted range `{part}`"));
        }
        if hi > SPARSE_DENSE_HEADS_MAX_HEAD {
            return Err(format!(
                "head {hi} exceeds the bitmask cap {} — wider models need a reader bump",
                SPARSE_DENSE_HEADS_MAX_HEAD
            ));
        }
        for h in lo..=hi {
            mask |= 1u64 << h;
        }
    }
    Ok(mask)
}

/// The per-head DENSE bitmask (bit h set = head h never skips; 0 = the
/// incumbent all-sparse posture). Env `RIIR_PREFILL_SPARSE_DENSE_HEADS`,
/// read exactly once — the first caller wins the store (the τ protocol);
/// a junk env demotes the knob to 0 with ONE loud line.
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_sparse_dense_head_mask() -> u64 {
    let cur = SPARSE_DENSE_HEADS.load(std::sync::atomic::Ordering::Relaxed);
    if cur & SPARSE_DENSE_HEADS_INIT != 0 {
        return cur & !SPARSE_DENSE_HEADS_INIT;
    }
    let parsed = match std::env::var("RIIR_PREFILL_SPARSE_DENSE_HEADS") {
        Ok(spec) => match parse_dense_head_mask(&spec) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[sparse-prefill] RIIR_PREFILL_SPARSE_DENSE_HEADS rejected: {e} — \
                     the knob demotes to none"
                );
                0
            }
        },
        Err(_) => 0,
    };
    match SPARSE_DENSE_HEADS.compare_exchange(
        cur,
        parsed | SPARSE_DENSE_HEADS_INIT,
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
    ) {
        Ok(_) => parsed,
        Err(winner) => winner & !SPARSE_DENSE_HEADS_INIT,
    }
}

/// Test/demo setter (the `set_prefill_sparse_tau` precedent) — overrides the
/// env-derived mask; pass HEAD BITS only (bit 63 is managed here).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_sparse_dense_head_mask(mask: u64) {
    SPARSE_DENSE_HEADS.store(
        (mask & !SPARSE_DENSE_HEADS_INIT) | SPARSE_DENSE_HEADS_INIT,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The per-layer stride of the device mode buffer, in u32 words — padded to
/// the wgpu/mtl 32-byte binding-offset alignment (8 words). The logic words
/// ([layer][head]) upload into this padded layout; the kernel slices a
/// layer's words at `layer_idx · stride`.
#[cfg(feature = "cubecl_runtime")]
pub fn sparse_mode_layer_stride_words(n_head: usize) -> usize {
    n_head.div_ceil(8) * 8
}

/// Upload layout: the logic-shaped `[layer][head]` mirror re-laid as
/// `[layer][stride]` (first `n_head` words per stride carry the heads, the
/// rest zero). Pure — unit-tested against the alignment law.
#[cfg(feature = "cubecl_runtime")]
pub fn sparse_mode_padded_words(mirror: &[u32], n_layers: usize, n_head: usize) -> Vec<u32> {
    let stride = sparse_mode_layer_stride_words(n_head);
    let mut out = vec![0u32; n_layers * stride];
    for layer in 0..n_layers {
        let src = &mirror[layer * n_head..(layer + 1) * n_head];
        out[layer * stride..layer * stride + n_head].copy_from_slice(src);
    }
    out
}

/// Whether `layer_idx` has ANY dense head in the logic-shaped mirror — the
/// dispatch consults this per (layer, chunk): a fully dense layer skips the
/// whole sparse pipeline (pool/score/select too) and takes the dense ladder.
#[cfg(feature = "cubecl_runtime")]
pub fn sparse_layer_dense(mirror: &[u32], layer_idx: usize, n_head: usize) -> bool {
    let lo = layer_idx * n_head;
    mirror
        .get(lo..lo + n_head)
        .is_some_and(|row| row.iter().any(|&w| w != 0))
}

/// Test/demo setter — overrides the env-derived value (the
/// `set_concat_split_fused` precedent).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_sparse(enabled: bool) {
    SPARSE_INITIALIZED.store(true, std::sync::atomic::Ordering::Relaxed);
    SPARSE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Test/demo setter for τ (clamped like the env path).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_sparse_tau(tau: f32) {
    SPARSE_TAU_BITS.store(
        tau.clamp(0.5, 1.0).to_bits(),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The scratch-build engagement predicate: enabled + the prompt-level length
/// gate. `head_dim == 256` is checked by the caller alongside the rest of the
/// dispatch gates.
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_sparse_engaged(active_prefill_len: usize) -> bool {
    prefill_sparse_enabled() && active_prefill_len >= PREFILL_SPARSE_MIN_P
}

#[cfg(feature = "cubecl_runtime")]
pub fn sparse_pool_launch_count() -> usize {
    SPARSE_POOL_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "cubecl_runtime")]
pub fn sparse_score_launch_count() -> usize {
    SPARSE_SCORE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "cubecl_runtime")]
pub fn sparse_select_launch_count() -> usize {
    SPARSE_SELECT_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "cubecl_runtime")]
pub fn sparse_attn_launch_count() -> usize {
    SPARSE_ATTN_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "cubecl_runtime")]
fn note_sparse_pool_launch() {
    SPARSE_POOL_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cubecl_runtime")]
fn note_sparse_score_launch() {
    SPARSE_SCORE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cubecl_runtime")]
fn note_sparse_select_launch() {
    SPARSE_SELECT_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cubecl_runtime")]
fn note_sparse_attn_launch() {
    SPARSE_ATTN_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Per-launch scratch (rides the per-chunk BatchedAttentionScratch pool)
// ---------------------------------------------------------------------------

/// The block-sparse lane's transient buffers for one prefill chunk. All
/// three are pure scratch: `kpool`/`scores` are rewritten by their kernels
/// every (layer, chunk); `mask` is rewritten by the select kernel every
/// (layer, chunk) — no cross-chunk state, no stable-address requirement.
#[cfg(feature = "cubecl_runtime")]
pub struct SparsePrefillScratch {
    /// Pooled K vectors per (KV group, KV block), TWO HALVES:
    /// `[n_kv * n_kv_blocks * head_dim]` = the per-dim **max** (or the v1
    /// mean, in the first half only), and a second same-size half for the
    /// per-dim **min** (envelope mode; mean mode leaves it untouched).
    pub kpool: cubecl::server::Handle,
    /// Block scores per (window, head, KV block): `[q_windows * n_head * n_kv_blocks]`.
    pub scores: cubecl::server::Handle,
    /// Selection bitmask per (window, head): `[q_windows * n_head * wpr]` u32 words.
    pub mask: cubecl::server::Handle,
    /// The cache-row count everything was sized for (`base_pos + p`).
    pub cache_rows: usize,
    /// Issue 1156 T5 — the per-layer per-head DENSE mode words on device,
    /// uploaded in the padded `[layer][stride]` layout
    /// ([`sparse_mode_padded_words`]); word != 0 marks the head DENSE (its
    /// cubes never take the skip branch). All-zero = the incumbent
    /// all-sparse posture, byte-identical behavior.
    pub head_mode: cubecl::server::Handle,
    /// The CPU mirror of `head_mode`, LOGIC-shaped `[layer][head]` — the
    /// dispatch consults it per layer (a fully dense layer skips the whole
    /// sparse pipeline) without a device readback.
    pub head_mode_words: Vec<u32>,
    /// Per-layer word stride of the padded upload layout.
    pub head_mode_stride: usize,
    /// Issue 1156 T4 — the regime gate's chunk verdict (`true` = run the
    /// sparse branch; the gate routes dense by clearing this).
    pub sparse_engaged: bool,
    /// Issue 1156 T4 — the gate's τ override for this chunk (`None` = the
    /// env τ). Legal per the T3 manifest via `verify_posture_at_len`.
    pub tau_override: Option<f32>,
    /// Issue 1156 T4 — boundary-row staging for the gate's readback
    /// (`Some` iff the gate is armed at build; `None` = zero cost, G3).
    pub bnd: Option<cubecl::server::Handle>,
}

/// Derive the scratch handles for one chunk. Returns None on allocation
/// failure (the dispatch then falls through to the dense ladder — a loud
/// counter absence, never a wrong answer).
///
/// `n_layers` sizes the T5 mode buffer (the layer dimension); `bnd_elems`
/// sizes the T4 boundary staging (0 = the gate is disarmed — no alloc, G3).
#[cfg(feature = "cubecl_runtime")]
pub fn alloc_sparse_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    cache_rows: usize,
    n_layers: usize,
    bnd_elems: usize,
) -> Option<SparsePrefillScratch> {
    let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
    let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
    let wpr = n_kv_blocks.div_ceil(32);
    // kpool is posture-independent 2× (max half + min half) so the alloc
    // never consults the scorer static — the envelope mode simply uses both
    // halves, mean mode the first.
    let kpool_elems = n_kv * n_kv_blocks * head_dim * 2;
    // T5: build the mode mirror from the (env-derived, first-caller-wins)
    // dense-head bitmask — all layers share it at v1; the uploaded words are
    // padded to the 32-byte binding alignment.
    let head_bits = prefill_sparse_dense_head_mask();
    let one_row: Vec<u32> = (0..n_head)
        .map(|h| if head_bits >> h & 1 != 0 { 1 } else { 0 })
        .collect();
    let mut head_mode_words = Vec::with_capacity(n_layers * n_head);
    for _ in 0..n_layers {
        head_mode_words.extend_from_slice(&one_row);
    }
    let head_mode_stride = sparse_mode_layer_stride_words(n_head);
    let padded = sparse_mode_padded_words(&head_mode_words, n_layers, n_head);
    // T4: the gate's boundary staging — only when armed (zero cost, G3).
    let bnd = if bnd_elems > 0 {
        Some(client.empty(bnd_elems * 4))
    } else {
        None
    };
    Some(SparsePrefillScratch {
        kpool: client.empty(kpool_elems * 4),
        scores: client.empty(q_windows * n_head * n_kv_blocks * 4),
        mask: client.empty(q_windows * n_head * wpr * 4),
        cache_rows,
        head_mode: client.create_from_slice(u32::as_bytes(&padded)),
        head_mode_words,
        head_mode_stride,
        sparse_engaged: true,
        tau_override: None,
        bnd,
    })
}

/// Mask words per (window, head) row for a cache-row capacity — the shared
/// derivation (launcher-side and test-side).
#[cfg(feature = "cubecl_runtime")]
pub fn sparse_mask_wpr(cache_rows: usize) -> usize {
    cache_rows.div_ceil(SPARSE_KV_BLOCK).div_ceil(32)
}

// ---------------------------------------------------------------------------
// Kernel 1 — the K-block pool (per KV group × KV block)
// ---------------------------------------------------------------------------

/// Per (KV group, KV block) per dim, by scorer mode:
/// - `Mean` (v1): the block's mean K vector over valid rows.
/// - `Envelope` (v2): the per-dim **max** and **min** K over valid rows —
///   the Quest-class metadata. Max lives in the buffer's first half, min in
///   the second half at `+ kpool_half_elems`. Finite sentinels
///   (`f32::MIN`/`f32::MAX`) seed the reductions — no infinity literals in
///   kernel bodies; every block carries ≥ 1 valid row by construction.
///
/// Layout: `kpool[(kv_group * n_kv_blocks + jb) * head_dim + d]` (max half);
/// `kpool[kpool_half_elems + (kv_group * n_kv_blocks + jb) * head_dim + d]`
/// (min half, envelope mode only).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn sparse_prefill_pool_f32(key: &[f32], kpool: &mut [f32], params: &[f32]) {
    let head_dim = params[0usize] as u32;
    let n_kv = params[1usize] as u32;
    let cache_rows = params[2usize] as u32;
    let n_kv_blocks = params[3usize] as u32;
    let mode = params[4usize] as u32;
    let half_elems = params[5usize] as u32;

    let cube_id = CUBE_POS_X;
    let kv_group = cube_id / n_kv_blocks;
    let jb = cube_id % n_kv_blocks;

    let lane = UNIT_POS_PLANE;
    let dims_base = (lane * 8u32) as usize;
    let kv_stride = n_kv * head_dim;
    let kv_head_off = kv_group * head_dim;

    let row_lo = jb * 64u32;
    let row_hi = if row_lo + 64u32 < cache_rows {
        row_lo + 64u32
    } else {
        cache_rows
    };

    // Lane-strided rows: lane and lane+32 cover the ≤64-row block.
    let r0 = row_lo + lane;
    let r1 = r0 + 32u32;

    let mut s0 = f32::new(0.0f32);
    let mut s1 = f32::new(0.0f32);
    let mut s2 = f32::new(0.0f32);
    let mut s3 = f32::new(0.0f32);
    let mut s4 = f32::new(0.0f32);
    let mut s5 = f32::new(0.0f32);
    let mut s6 = f32::new(0.0f32);
    let mut s7 = f32::new(0.0f32);

    // Envelope accumulators: finite sentinels so an inactive lane cannot
    // poison a max/min reduction (f32::MIN seeds max, f32::MAX seeds min).
    let mut m0 = f32::new(f32::MIN);
    let mut m1 = f32::new(f32::MIN);
    let mut m2 = f32::new(f32::MIN);
    let mut m3 = f32::new(f32::MIN);
    let mut m4 = f32::new(f32::MIN);
    let mut m5 = f32::new(f32::MIN);
    let mut m6 = f32::new(f32::MIN);
    let mut m7 = f32::new(f32::MIN);
    let mut n0 = f32::new(f32::MAX);
    let mut n1 = f32::new(f32::MAX);
    let mut n2 = f32::new(f32::MAX);
    let mut n3 = f32::new(f32::MAX);
    let mut n4 = f32::new(f32::MAX);
    let mut n5 = f32::new(f32::MAX);
    let mut n6 = f32::new(f32::MAX);
    let mut n7 = f32::new(f32::MAX);

    if r0 < row_hi {
        let base = (r0 * kv_stride + kv_head_off) as usize + dims_base;
        s0 = key[base];
        s1 = key[base + 1usize];
        s2 = key[base + 2usize];
        s3 = key[base + 3usize];
        s4 = key[base + 4usize];
        s5 = key[base + 5usize];
        s6 = key[base + 6usize];
        s7 = key[base + 7usize];
        m0 = s0;
        m1 = s1;
        m2 = s2;
        m3 = s3;
        m4 = s4;
        m5 = s5;
        m6 = s6;
        m7 = s7;
        n0 = s0;
        n1 = s1;
        n2 = s2;
        n3 = s3;
        n4 = s4;
        n5 = s5;
        n6 = s6;
        n7 = s7;
    }
    if r1 < row_hi {
        let base = (r1 * kv_stride + kv_head_off) as usize + dims_base;
        let t0 = key[base];
        let t1 = key[base + 1usize];
        let t2 = key[base + 2usize];
        let t3 = key[base + 3usize];
        let t4 = key[base + 4usize];
        let t5 = key[base + 5usize];
        let t6 = key[base + 6usize];
        let t7 = key[base + 7usize];
        s0 += t0;
        s1 += t1;
        s2 += t2;
        s3 += t3;
        s4 += t4;
        s5 += t5;
        s6 += t6;
        s7 += t7;
        m0 = if t0 > m0 { t0 } else { m0 };
        m1 = if t1 > m1 { t1 } else { m1 };
        m2 = if t2 > m2 { t2 } else { m2 };
        m3 = if t3 > m3 { t3 } else { m3 };
        m4 = if t4 > m4 { t4 } else { m4 };
        m5 = if t5 > m5 { t5 } else { m5 };
        m6 = if t6 > m6 { t6 } else { m6 };
        m7 = if t7 > m7 { t7 } else { m7 };
        n0 = if t0 < n0 { t0 } else { n0 };
        n1 = if t1 < n1 { t1 } else { n1 };
        n2 = if t2 < n2 { t2 } else { n2 };
        n3 = if t3 < n3 { t3 } else { n3 };
        n4 = if t4 < n4 { t4 } else { n4 };
        n5 = if t5 < n5 { t5 } else { n5 };
        n6 = if t6 < n6 { t6 } else { n6 };
        n7 = if t7 < n7 { t7 } else { n7 };
    }

    let out_base = (cube_id * head_dim) as usize + dims_base;
    if mode == 1u32 {
        // Envelope: per-dim max -> the buffer's first half, per-dim min ->
        // the second half. One hardware plane reduce each.
        let min_base = out_base + half_elems as usize;
        kpool[out_base] = plane_max(m0);
        kpool[out_base + 1usize] = plane_max(m1);
        kpool[out_base + 2usize] = plane_max(m2);
        kpool[out_base + 3usize] = plane_max(m3);
        kpool[out_base + 4usize] = plane_max(m4);
        kpool[out_base + 5usize] = plane_max(m5);
        kpool[out_base + 6usize] = plane_max(m6);
        kpool[out_base + 7usize] = plane_max(m7);
        kpool[min_base] = plane_min(n0);
        kpool[min_base + 1usize] = plane_min(n1);
        kpool[min_base + 2usize] = plane_min(n2);
        kpool[min_base + 3usize] = plane_min(n3);
        kpool[min_base + 4usize] = plane_min(n4);
        kpool[min_base + 5usize] = plane_min(n5);
        kpool[min_base + 6usize] = plane_min(n6);
        kpool[min_base + 7usize] = plane_min(n7);
    } else {
        // Mean (the v1 scorer): tree reduce per dim (deterministic
        // butterfly), then mean.
        let t0 = plane_sum(s0);
        let t1 = plane_sum(s1);
        let t2 = plane_sum(s2);
        let t3 = plane_sum(s3);
        let t4 = plane_sum(s4);
        let t5 = plane_sum(s5);
        let t6 = plane_sum(s6);
        let t7 = plane_sum(s7);

        let count = row_hi - row_lo;
        let inv = f32::new(1.0f32) / f32::cast_from(count);
        kpool[out_base] = t0 * inv;
        kpool[out_base + 1usize] = t1 * inv;
        kpool[out_base + 2usize] = t2 * inv;
        kpool[out_base + 3usize] = t3 * inv;
        kpool[out_base + 4usize] = t4 * inv;
        kpool[out_base + 5usize] = t5 * inv;
        kpool[out_base + 6usize] = t6 * inv;
        kpool[out_base + 7usize] = t7 * inv;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 — the block scorer (diagonal band + pooled-centroid probe)
// ---------------------------------------------------------------------------

/// Per (head, 128-query window, KV block ≤ the window's diagonal horizon):
/// `scores[(qb * n_head + head) * n_kv_blocks + jb] = band + probe` where the
/// probe is scorer-dependent:
/// - `Mean` (v1): the pooled-centroid dot `q̄ · k̄_block` (the FlexPrefill
///   pooled probe).
/// - `Envelope` (v2): the Quest-class bound `Σ_d q̄_d·(q̄_d ≥ 0 ? kmax_d :
///   kmin_d)` ≥ `max_r(q̄·k_r)` — the block's best single row, upper-bounded
///   per dim. A single fact row survives at full magnitude on its dims
///   instead of being averaged into a 64-row mean.
///
/// The window's overlap `[lo_abs, hi_abs)` bounds both the Q rows consumed and
/// the KV blocks scored; windows with no chunk-local rows do nothing (their
/// mask rows are never consulted — the attention kernel only exists for
/// actual chunk rows).
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::manual_div_ceil,
    reason = "the CubeCL kernel frontend has no u32 div_ceil intrinsic"
)]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn sparse_prefill_score_f32(
    query: &[f32],
    key: &[f32],
    kpool: &[f32],
    scores: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv = params[2usize] as u32;
    let cache_rows = params[3usize] as u32;
    let qc = params[4usize] as u32;
    let q_blk_base = params[5usize] as u32;
    let n_kv_blocks = params[6usize] as u32;
    let base_pos = params[7usize] as u32;
    let mode = params[8usize] as u32;
    let half_elems = params[9usize] as u32;

    let cube_id = CUBE_POS_X;
    let head_idx = cube_id / qc;
    let qb = q_blk_base + cube_id % qc;

    let q_start = qb * 128u32;
    // The window's overlap with this chunk's Q rows [base_pos, base_pos + p):
    // cache_rows == base_pos + p by the launcher contract.
    let lo_abs = if q_start > base_pos {
        q_start
    } else {
        base_pos
    };
    let hi_abs = if q_start + 128u32 < cache_rows {
        q_start + 128u32
    } else {
        cache_rows
    };

    let lane = UNIT_POS_PLANE;
    let dims_base = (lane * 8u32) as usize;

    if lo_abs < hi_abs {
        let valid_rows = hi_abs - lo_abs;
        let kv_group = head_idx * n_kv / n_head;
        let kv_head_off = kv_group * head_dim;
        let kv_stride = n_kv * head_dim;
        let q_stride = n_head * head_dim;

        // This lane's 4 Q rows (post-norm post-RoPE, the same rows the flash
        // kernel consumes), chunk-local indexing.
        let q_row0 = q_start + lane * 4u32;
        let qa_active = q_row0 >= lo_abs && q_row0 < hi_abs;
        let qb_active = q_row0 + 1u32 >= lo_abs && q_row0 + 1u32 < hi_abs;
        let qc_active = q_row0 + 2u32 >= lo_abs && q_row0 + 2u32 < hi_abs;
        let qd_active = q_row0 + 3u32 >= lo_abs && q_row0 + 3u32 < hi_abs;
        let qa_off = ((q_row0 - base_pos) * q_stride + head_idx * head_dim) as usize + dims_base;
        let qb_off =
            ((q_row0 + 1u32 - base_pos) * q_stride + head_idx * head_dim) as usize + dims_base;
        let qc_off =
            ((q_row0 + 2u32 - base_pos) * q_stride + head_idx * head_dim) as usize + dims_base;
        let qd_off =
            ((q_row0 + 3u32 - base_pos) * q_stride + head_idx * head_dim) as usize + dims_base;

        let qa0 = if qa_active {
            query[qa_off]
        } else {
            f32::new(0.0f32)
        };
        let qa1 = if qa_active {
            query[qa_off + 1usize]
        } else {
            f32::new(0.0f32)
        };
        let qa2 = if qa_active {
            query[qa_off + 2usize]
        } else {
            f32::new(0.0f32)
        };
        let qa3 = if qa_active {
            query[qa_off + 3usize]
        } else {
            f32::new(0.0f32)
        };
        let qa4 = if qa_active {
            query[qa_off + 4usize]
        } else {
            f32::new(0.0f32)
        };
        let qa5 = if qa_active {
            query[qa_off + 5usize]
        } else {
            f32::new(0.0f32)
        };
        let qa6 = if qa_active {
            query[qa_off + 6usize]
        } else {
            f32::new(0.0f32)
        };
        let qa7 = if qa_active {
            query[qa_off + 7usize]
        } else {
            f32::new(0.0f32)
        };
        let qb0 = if qb_active {
            query[qb_off]
        } else {
            f32::new(0.0f32)
        };
        let qb1 = if qb_active {
            query[qb_off + 1usize]
        } else {
            f32::new(0.0f32)
        };
        let qb2 = if qb_active {
            query[qb_off + 2usize]
        } else {
            f32::new(0.0f32)
        };
        let qb3 = if qb_active {
            query[qb_off + 3usize]
        } else {
            f32::new(0.0f32)
        };
        let qb4 = if qb_active {
            query[qb_off + 4usize]
        } else {
            f32::new(0.0f32)
        };
        let qb5 = if qb_active {
            query[qb_off + 5usize]
        } else {
            f32::new(0.0f32)
        };
        let qb6 = if qb_active {
            query[qb_off + 6usize]
        } else {
            f32::new(0.0f32)
        };
        let qb7 = if qb_active {
            query[qb_off + 7usize]
        } else {
            f32::new(0.0f32)
        };
        let qc0 = if qc_active {
            query[qc_off]
        } else {
            f32::new(0.0f32)
        };
        let qc1 = if qc_active {
            query[qc_off + 1usize]
        } else {
            f32::new(0.0f32)
        };
        let qc2 = if qc_active {
            query[qc_off + 2usize]
        } else {
            f32::new(0.0f32)
        };
        let qc3 = if qc_active {
            query[qc_off + 3usize]
        } else {
            f32::new(0.0f32)
        };
        let qc4 = if qc_active {
            query[qc_off + 4usize]
        } else {
            f32::new(0.0f32)
        };
        let qc5 = if qc_active {
            query[qc_off + 5usize]
        } else {
            f32::new(0.0f32)
        };
        let qc6 = if qc_active {
            query[qc_off + 6usize]
        } else {
            f32::new(0.0f32)
        };
        let qc7 = if qc_active {
            query[qc_off + 7usize]
        } else {
            f32::new(0.0f32)
        };
        let qd0 = if qd_active {
            query[qd_off]
        } else {
            f32::new(0.0f32)
        };
        let qd1 = if qd_active {
            query[qd_off + 1usize]
        } else {
            f32::new(0.0f32)
        };
        let qd2 = if qd_active {
            query[qd_off + 2usize]
        } else {
            f32::new(0.0f32)
        };
        let qd3 = if qd_active {
            query[qd_off + 3usize]
        } else {
            f32::new(0.0f32)
        };
        let qd4 = if qd_active {
            query[qd_off + 4usize]
        } else {
            f32::new(0.0f32)
        };
        let qd5 = if qd_active {
            query[qd_off + 5usize]
        } else {
            f32::new(0.0f32)
        };
        let qd6 = if qd_active {
            query[qd_off + 6usize]
        } else {
            f32::new(0.0f32)
        };
        let qd7 = if qd_active {
            query[qd_off + 7usize]
        } else {
            f32::new(0.0f32)
        };

        // Pooled query mean for this lane's dims (one butterfly per dim;
        // the window's valid-row count normalizes).
        let inv_rows = f32::new(1.0f32) / f32::cast_from(valid_rows);
        let qm0 = plane_sum(qa0 + qb0 + qc0 + qd0) * inv_rows;
        let qm1 = plane_sum(qa1 + qb1 + qc1 + qd1) * inv_rows;
        let qm2 = plane_sum(qa2 + qb2 + qc2 + qd2) * inv_rows;
        let qm3 = plane_sum(qa3 + qb3 + qc3 + qd3) * inv_rows;
        let qm4 = plane_sum(qa4 + qb4 + qc4 + qd4) * inv_rows;
        let qm5 = plane_sum(qa5 + qb5 + qc5 + qd5) * inv_rows;
        let qm6 = plane_sum(qa6 + qb6 + qc6 + qd6) * inv_rows;
        let qm7 = plane_sum(qa7 + qb7 + qc7 + qd7) * inv_rows;

        // The window's diagonal horizon in KV blocks: keys ≤ hi_abs - 1.
        let blocks_valid = (hi_abs + 63u32) / 64u32;
        let row_base = (qb * n_head + head_idx) * n_kv_blocks;

        let mut jb = 0u32;
        while jb < blocks_valid {
            let block_start = jb * 64u32;
            // Band: the window-diagonal cells INSIDE this block. For
            // window-local query x the diagonal key row is q_start + x; it
            // lies in block jb iff block_start ≤ q_start + x < block_start + 64.
            let diag_lo = block_start.saturating_sub(q_start);
            let diag_hi_raw = (block_start + 64u32).saturating_sub(q_start);
            let x_hi = if diag_hi_raw < valid_rows {
                diag_hi_raw
            } else {
                valid_rows
            };
            let x_lo = if diag_lo < x_hi { diag_lo } else { x_hi };

            let mut bd = f32::new(0.0f32);
            let mut r = 0u32;
            while r < 4u32 {
                let x_r = lane * 4u32 + r;
                if x_r >= x_lo && x_r < x_hi {
                    let krow = q_start + x_r;
                    let kb = (krow * kv_stride + kv_head_off) as usize + dims_base;
                    let k0 = key[kb];
                    let k1 = key[kb + 1usize];
                    let k2 = key[kb + 2usize];
                    let k3 = key[kb + 3usize];
                    let k4 = key[kb + 4usize];
                    let k5 = key[kb + 5usize];
                    let k6 = key[kb + 6usize];
                    let k7 = key[kb + 7usize];
                    let dot = if r == 0u32 {
                        qa0 * k0
                            + qa1 * k1
                            + qa2 * k2
                            + qa3 * k3
                            + qa4 * k4
                            + qa5 * k5
                            + qa6 * k6
                            + qa7 * k7
                    } else {
                        if r == 1u32 {
                            qb0 * k0
                                + qb1 * k1
                                + qb2 * k2
                                + qb3 * k3
                                + qb4 * k4
                                + qb5 * k5
                                + qb6 * k6
                                + qb7 * k7
                        } else {
                            if r == 2u32 {
                                qc0 * k0
                                    + qc1 * k1
                                    + qc2 * k2
                                    + qc3 * k3
                                    + qc4 * k4
                                    + qc5 * k5
                                    + qc6 * k6
                                    + qc7 * k7
                            } else {
                                qd0 * k0
                                    + qd1 * k1
                                    + qd2 * k2
                                    + qd3 * k3
                                    + qd4 * k4
                                    + qd5 * k5
                                    + qd6 * k6
                                    + qd7 * k7
                            }
                        }
                    };
                    bd += dot;
                }
                r += 1u32;
            }
            let band = plane_sum(bd);

            // Pooled-centroid probe against this block's mean K (the v1
            // scorer) — or the Quest-class envelope against the block's
            // per-dim max/min K (v2). The min half sits at + half_elems.
            let kp = ((kv_group * n_kv_blocks + jb) * head_dim) as usize + dims_base;
            let km0 = kpool[kp];
            let km1 = kpool[kp + 1usize];
            let km2 = kpool[kp + 2usize];
            let km3 = kpool[kp + 3usize];
            let km4 = kpool[kp + 4usize];
            let km5 = kpool[kp + 5usize];
            let km6 = kpool[kp + 6usize];
            let km7 = kpool[kp + 7usize];
            let probe = if mode == 1u32 {
                let min_base = kp + half_elems as usize;
                let kn0 = kpool[min_base];
                let kn1 = kpool[min_base + 1usize];
                let kn2 = kpool[min_base + 2usize];
                let kn3 = kpool[min_base + 3usize];
                let kn4 = kpool[min_base + 4usize];
                let kn5 = kpool[min_base + 5usize];
                let kn6 = kpool[min_base + 6usize];
                let kn7 = kpool[min_base + 7usize];
                let zero = f32::new(0.0f32);
                // The envelope: sign-select the per-dim extreme that maximizes
                // the dot; the sum upper-bounds max_r(q̄·k_r) for the block.
                let e0 = if qm0 >= zero { qm0 * km0 } else { qm0 * kn0 };
                let e1 = if qm1 >= zero { qm1 * km1 } else { qm1 * kn1 };
                let e2 = if qm2 >= zero { qm2 * km2 } else { qm2 * kn2 };
                let e3 = if qm3 >= zero { qm3 * km3 } else { qm3 * kn3 };
                let e4 = if qm4 >= zero { qm4 * km4 } else { qm4 * kn4 };
                let e5 = if qm5 >= zero { qm5 * km5 } else { qm5 * kn5 };
                let e6 = if qm6 >= zero { qm6 * km6 } else { qm6 * kn6 };
                let e7 = if qm7 >= zero { qm7 * km7 } else { qm7 * kn7 };
                plane_sum(e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7)
            } else {
                let cp = qm0 * km0
                    + qm1 * km1
                    + qm2 * km2
                    + qm3 * km3
                    + qm4 * km4
                    + qm5 * km5
                    + qm6 * km6
                    + qm7 * km7;
                plane_sum(cp)
            };

            if lane == 0u32 {
                scores[(row_base + jb) as usize] = band + probe;
            }
            jb += 1u32;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3 — the τ-mass selection (bisect threshold + sink + identity path)
// ---------------------------------------------------------------------------

/// Per (head, query window): clear + select the KV blocks carrying ≥ τ of the
/// window's score mass. The mask word assembly is one lane per word (no
/// read-modify-write races); sink blocks {0, 1} are folded into word 0.
///
/// params: `[n_head, cache_rows, qc, q_blk_base, n_kv_blocks, wpr, tau, base_pos]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::manual_div_ceil,
    reason = "the CubeCL kernel frontend has no u32 div_ceil intrinsic"
)]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn sparse_prefill_select_f32(scores: &[f32], mask: &mut [u32], params: &[f32]) {
    let n_head = params[0usize] as u32;
    let cache_rows = params[1usize] as u32;
    let qc = params[2usize] as u32;
    let q_blk_base = params[3usize] as u32;
    let n_kv_blocks = params[4usize] as u32;
    let wpr = params[5usize] as u32;
    let tau = params[6usize];
    let base_pos = params[7usize] as u32;

    let cube_id = CUBE_POS_X;
    let head_idx = cube_id / qc;
    let qb = q_blk_base + cube_id % qc;

    let q_start = qb * 128u32;
    let lo_abs = if q_start > base_pos {
        q_start
    } else {
        base_pos
    };
    let hi_abs = if q_start + 128u32 < cache_rows {
        q_start + 128u32
    } else {
        cache_rows
    };

    let lane = UNIT_POS_PLANE;

    if lo_abs < hi_abs {
        // The window's causal horizon in KV blocks.
        let blocks_valid = (hi_abs + 63u32) / 64u32;
        let srow = ((qb * n_head + head_idx) * n_kv_blocks) as usize;
        let mrow = ((qb * n_head + head_idx) * wpr) as usize;

        // 1. Clear the row; word 0 carries the sink bits (blocks {0, 1} —
        //    the first 128 tokens; a bit past `blocks_valid` is never
        //    consulted by the attention kernel, so setting it is harmless).
        let mut w = lane;
        while w < wpr {
            mask[mrow + w as usize] = if w == 0u32 { 3u32 } else { 0u32 };
            w += 32u32;
        }

        // 2. Total score mass over the valid blocks.
        let mut part = f32::new(0.0f32);
        let mut b = lane;
        while b < blocks_valid {
            part += scores[srow + b as usize];
            b += 32u32;
        }
        let total = plane_sum(part);

        if tau >= f32::new(1.0f32) {
            // The identity posture: every in-horizon block selected —
            // REGARDLESS of score (a zero-mass window still selects all its
            // blocks; skipping any would break the τ=1.0 bit-identity with
            // m32). The attention kernel's skip branch goes dead and the
            // output is bit-identical to m32.
            let mut w = lane;
            while w < wpr {
                let b_lo = w * 32u32;
                let b_hi = if b_lo + 32u32 < blocks_valid {
                    b_lo + 32u32
                } else {
                    blocks_valid
                };
                let mut word = 0u32;
                let mut bb = b_lo;
                while bb < b_hi {
                    word |= 1u32 << (bb - b_lo);
                    bb += 1u32;
                }
                if w == 0u32 {
                    word |= 3u32;
                }
                mask[mrow + w as usize] = word;
                w += 32u32;
            }
        } else if total > f32::new(0.0f32) {
            // Bisect: find the largest θ with mass(θ) ≥ τ·total.
            //    mass(lo) ≥ budget is the invariant (lo starts at 0,
            //    where mass = total); 28 fixed iterations bound the
            //    interval by 2⁻²⁸·total.
            let budget = tau * total;
            let mut lo = f32::new(0.0f32);
            let mut hi = total;
            let mut it = 0u32;
            while it < 28u32 {
                let mid = (lo + hi) * f32::new(0.5f32);
                let mut mp = f32::new(0.0f32);
                let mut bb = lane;
                while bb < blocks_valid {
                    let s = scores[srow + bb as usize];
                    if s >= mid {
                        mp += s;
                    }
                    bb += 32u32;
                }
                let mass = plane_sum(mp);
                if mass >= budget {
                    lo = mid;
                } else {
                    hi = mid;
                }
                it += 1u32;
            }

            // Write the selected bits (one lane per word — no RMW race).
            let mut w = lane;
            while w < wpr {
                let b_lo = w * 32u32;
                if b_lo < blocks_valid {
                    let b_hi = if b_lo + 32u32 < blocks_valid {
                        b_lo + 32u32
                    } else {
                        blocks_valid
                    };
                    let mut word = 0u32;
                    let mut bb = b_lo;
                    while bb < b_hi {
                        if scores[srow + bb as usize] >= lo {
                            word |= 1u32 << (bb - b_lo);
                        }
                        bb += 1u32;
                    }
                    if w == 0u32 {
                        word |= 3u32;
                    }
                    mask[mrow + w as usize] = word;
                }
                w += 32u32;
            }
        }
        // τ<1.0 with total ≤ 0: the cleared row + sink bits stand (a
        // sink-only mask — every query row has ≥ 128 causal sink keys, so
        // the softmax renormalization stays well-formed).
    }
}

// ---------------------------------------------------------------------------
// Kernel 4 — the sparse attention body (m32 VERBATIM + the block skip)
// ---------------------------------------------------------------------------

/// Queries per cube (= 4 per plane at `CubeDim::new_1d(256)`) — the m32
/// geometry this kernel clones.
#[cfg(feature = "cubecl_runtime")]
const SPARSE_ATTN_Q_PER_CUBE: u32 = 32;

#[cfg(feature = "cubecl_runtime")]
const SPARSE_KV_BLOCK_U32: u32 = 64;

/// The M=32 four-rows-per-plane tiled causal flash-attention prefill with the
/// block-sparse KV skip (Issue 1156 T2) and the per-head mixed-mode bitmask
/// (T5). The per-row arithmetic is the m32 shape VERBATIM (the bit-identity
/// anchor at τ=1.0); the ONLY changes are: the mask test at KV-block
/// boundaries and the block jump when the bit is clear (T2), and the
/// per-head DENSE mode word that kills the skip branch outright (T5 —
/// `head_mode[head_idx] != 0`; cube-uniform since head_idx is fixed per
/// cube). Buffer contract = the m32 kernel's + `mask`, `head_mode` and three
/// params:
/// `[head_dim, n_head, n_kv_head, tc, scale, q_offset, tiles, wpr,
///   q_blk_base, q_offset_mod]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::manual_is_multiple_of,
    reason = "the CubeCL kernel frontend has no u32 is_multiple_of intrinsic"
)]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_sparse_m32_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    attn_out: &mut [f32],
    mask: &[u32],
    head_mode: &[u32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let p = params[3usize] as u32;
    let scale = params[4usize];
    let q_offset = params[5usize] as u32;
    let q_tiles = params[6usize] as u32;
    let wpr = params[7usize] as u32;
    let q_blk_base = params[8usize] as u32;
    let q_offset_mod = params[9usize] as u32;

    let cube_id = CUBE_POS_X;
    let head_idx = cube_id / q_tiles;
    let q_tile = cube_id % q_tiles;

    // Plane/lane decomposition (simdgroup i = threads [32i, 32i+32)).
    let pl = UNIT_POS / 32u32;
    let lane = UNIT_POS_PLANE;

    // This cube's 32 query rows share ONE 128-row mask window: q_offset is
    // 32-aligned (tile-aligned grid chunks) so the tile never crosses a
    // window boundary.
    let q_blk = q_blk_base + (q_offset_mod + q_tile * SPARSE_ATTN_Q_PER_CUBE) / 128u32;
    let mask_row = ((q_blk * n_head + head_idx) * wpr) as usize;

    // Issue 1156 T5 — per-head mixed mode: a nonzero word marks the head
    // DENSE — the skip branch goes dead for this cube's whole run (the mask
    // is never consulted; the executed sequence is m32's exactly). The mode
    // buffer is sliced per layer by the launcher at
    // `layer_idx · sparse_mode_layer_stride_words(n_head)` words, so index 0
    // here IS this layer's word for `head_idx`. Cube-uniform: head_idx is
    // fixed per cube, so mixed launches branch per cube, never per lane
    // (no divergent shuffle hazard).
    let head_dense = head_mode[head_idx as usize] != 0u32;

    // Each plane owns FOUR ADJACENT query rows — the quartet's causal bounds
    // diverge across at most three positions, so the shared-loop waste is ≤3
    // bit-exact no-op iterations per plane.
    let q_base = q_tile * SPARSE_ATTN_Q_PER_CUBE + pl * 4u32;
    let q_pos_a = q_base;
    let q_pos_b = q_base + 1u32;
    let q_pos_c = q_base + 2u32;
    let q_pos_d = q_base + 3u32;
    let active_a = q_pos_a < p;
    let active_b = q_pos_b < p;
    let active_c = q_pos_c < p;
    let active_d = q_pos_d < p;

    // GQA: map query head to key/value head group.
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;
    let q_stride = n_head * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let q_abs_a = q_pos_a + q_offset;
    let q_abs_b = q_pos_b + q_offset;
    let q_abs_c = q_pos_c + q_offset;
    let q_abs_d = q_pos_d + q_offset;
    // Chunk-local bases of the four query rows (q/gate/attn_out are
    // chunk-sliced).
    let q_off_a = (q_pos_a * q_stride + head_idx * head_dim) as usize;
    let q_off_b = (q_pos_b * q_stride + head_idx * head_dim) as usize;
    let q_off_c = (q_pos_c * q_stride + head_idx * head_dim) as usize;
    let q_off_d = (q_pos_d * q_stride + head_idx * head_dim) as usize;
    let dims_base = (lane * 8u32) as usize;

    // Lane dims, row A (hand-unrolled: statically-indexed scalars stay in
    // registers — the Batch-49 local-memory-demotion class).
    let qa0 = if active_a {
        query[q_off_a + dims_base]
    } else {
        f32::new(0.0f32)
    };
    let qa1 = if active_a {
        query[q_off_a + dims_base + 1usize]
    } else {
        f32::new(0.0f32)
    };
    let qa2 = if active_a {
        query[q_off_a + dims_base + 2usize]
    } else {
        f32::new(0.0f32)
    };
    let qa3 = if active_a {
        query[q_off_a + dims_base + 3usize]
    } else {
        f32::new(0.0f32)
    };
    let qa4 = if active_a {
        query[q_off_a + dims_base + 4usize]
    } else {
        f32::new(0.0f32)
    };
    let qa5 = if active_a {
        query[q_off_a + dims_base + 5usize]
    } else {
        f32::new(0.0f32)
    };
    let qa6 = if active_a {
        query[q_off_a + dims_base + 6usize]
    } else {
        f32::new(0.0f32)
    };
    let qa7 = if active_a {
        query[q_off_a + dims_base + 7usize]
    } else {
        f32::new(0.0f32)
    };
    // Row B.
    let qb0 = if active_b {
        query[q_off_b + dims_base]
    } else {
        f32::new(0.0f32)
    };
    let qb1 = if active_b {
        query[q_off_b + dims_base + 1usize]
    } else {
        f32::new(0.0f32)
    };
    let qb2 = if active_b {
        query[q_off_b + dims_base + 2usize]
    } else {
        f32::new(0.0f32)
    };
    let qb3 = if active_b {
        query[q_off_b + dims_base + 3usize]
    } else {
        f32::new(0.0f32)
    };
    let qb4 = if active_b {
        query[q_off_b + dims_base + 4usize]
    } else {
        f32::new(0.0f32)
    };
    let qb5 = if active_b {
        query[q_off_b + dims_base + 5usize]
    } else {
        f32::new(0.0f32)
    };
    let qb6 = if active_b {
        query[q_off_b + dims_base + 6usize]
    } else {
        f32::new(0.0f32)
    };
    let qb7 = if active_b {
        query[q_off_b + dims_base + 7usize]
    } else {
        f32::new(0.0f32)
    };
    // Row C.
    let qc0 = if active_c {
        query[q_off_c + dims_base]
    } else {
        f32::new(0.0f32)
    };
    let qc1 = if active_c {
        query[q_off_c + dims_base + 1usize]
    } else {
        f32::new(0.0f32)
    };
    let qc2 = if active_c {
        query[q_off_c + dims_base + 2usize]
    } else {
        f32::new(0.0f32)
    };
    let qc3 = if active_c {
        query[q_off_c + dims_base + 3usize]
    } else {
        f32::new(0.0f32)
    };
    let qc4 = if active_c {
        query[q_off_c + dims_base + 4usize]
    } else {
        f32::new(0.0f32)
    };
    let qc5 = if active_c {
        query[q_off_c + dims_base + 5usize]
    } else {
        f32::new(0.0f32)
    };
    let qc6 = if active_c {
        query[q_off_c + dims_base + 6usize]
    } else {
        f32::new(0.0f32)
    };
    let qc7 = if active_c {
        query[q_off_c + dims_base + 7usize]
    } else {
        f32::new(0.0f32)
    };
    // Row D.
    let qd0 = if active_d {
        query[q_off_d + dims_base]
    } else {
        f32::new(0.0f32)
    };
    let qd1 = if active_d {
        query[q_off_d + dims_base + 1usize]
    } else {
        f32::new(0.0f32)
    };
    let qd2 = if active_d {
        query[q_off_d + dims_base + 2usize]
    } else {
        f32::new(0.0f32)
    };
    let qd3 = if active_d {
        query[q_off_d + dims_base + 3usize]
    } else {
        f32::new(0.0f32)
    };
    let qd4 = if active_d {
        query[q_off_d + dims_base + 4usize]
    } else {
        f32::new(0.0f32)
    };
    let qd5 = if active_d {
        query[q_off_d + dims_base + 5usize]
    } else {
        f32::new(0.0f32)
    };
    let qd6 = if active_d {
        query[q_off_d + dims_base + 6usize]
    } else {
        f32::new(0.0f32)
    };
    let qd7 = if active_d {
        query[q_off_d + dims_base + 7usize]
    } else {
        f32::new(0.0f32)
    };

    // Online softmax state per row, per lane (uniform across the plane after
    // every broadcast).
    let mut run_max_a = f32::new(-1e30f32);
    let mut run_sum_a = f32::new(0.0f32);
    let mut run_max_b = f32::new(-1e30f32);
    let mut run_sum_b = f32::new(0.0f32);
    let mut run_max_c = f32::new(-1e30f32);
    let mut run_sum_c = f32::new(0.0f32);
    let mut run_max_d = f32::new(-1e30f32);
    let mut run_sum_d = f32::new(0.0f32);
    let mut oa0 = f32::new(0.0f32);
    let mut oa1 = f32::new(0.0f32);
    let mut oa2 = f32::new(0.0f32);
    let mut oa3 = f32::new(0.0f32);
    let mut oa4 = f32::new(0.0f32);
    let mut oa5 = f32::new(0.0f32);
    let mut oa6 = f32::new(0.0f32);
    let mut oa7 = f32::new(0.0f32);
    let mut ob0 = f32::new(0.0f32);
    let mut ob1 = f32::new(0.0f32);
    let mut ob2 = f32::new(0.0f32);
    let mut ob3 = f32::new(0.0f32);
    let mut ob4 = f32::new(0.0f32);
    let mut ob5 = f32::new(0.0f32);
    let mut ob6 = f32::new(0.0f32);
    let mut ob7 = f32::new(0.0f32);
    let mut oc0 = f32::new(0.0f32);
    let mut oc1 = f32::new(0.0f32);
    let mut oc2 = f32::new(0.0f32);
    let mut oc3 = f32::new(0.0f32);
    let mut oc4 = f32::new(0.0f32);
    let mut oc5 = f32::new(0.0f32);
    let mut oc6 = f32::new(0.0f32);
    let mut oc7 = f32::new(0.0f32);
    let mut od0 = f32::new(0.0f32);
    let mut od1 = f32::new(0.0f32);
    let mut od2 = f32::new(0.0f32);
    let mut od3 = f32::new(0.0f32);
    let mut od4 = f32::new(0.0f32);
    let mut od5 = f32::new(0.0f32);
    let mut od6 = f32::new(0.0f32);
    let mut od7 = f32::new(0.0f32);

    // Uniform loop bound: the max causal position across the cube's 32
    // queries (+1). Out-of-causal positions mask to a bit-exact no-op.
    let q_last = q_tile * SPARSE_ATTN_Q_PER_CUBE + (SPARSE_ATTN_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    let mut pos = 0u32;
    while pos < n_loop {
        // ── Issue 1156 T2 — the ONLY change vs the m32 body: the block-sparse
        //    skip. At KV-block boundaries the mask bit decides; a cleared bit
        //    jumps the loop forward by the whole block (K load, QK dots, PV
        //    chain and all). The decision is whole-cube uniform. Skipping
        //    causal-masked or unselected rows is a bit-exact no-op for the
        //    online-softmax state (w = 0, corr = 1), so at τ=1.0 (every bit
        //    set) the executed sequence is m32's exactly. T5: a DENSE head's
        //    cubes never skip, whatever the mask says.
        let blk = pos / SPARSE_KV_BLOCK_U32;
        let word = mask[mask_row + (blk / 32u32) as usize];
        let is_boundary = pos % SPARSE_KV_BLOCK_U32 == 0u32;
        let bit_clear = ((word >> (blk % 32u32)) & 1u32) == 0u32;
        let skip = is_boundary && bit_clear && !head_dense;
        if skip {
            pos += SPARSE_KV_BLOCK_U32;
        } else {
            // ONE K row load shared by all four rows — the traffic-halving win.
            let k_base = (pos * kv_stride + kv_head_off) as usize;
            let k0 = key[k_base + dims_base];
            let k1 = key[k_base + dims_base + 1usize];
            let k2 = key[k_base + dims_base + 2usize];
            let k3 = key[k_base + dims_base + 3usize];
            let k4 = key[k_base + dims_base + 4usize];
            let k5 = key[k_base + dims_base + 5usize];
            let k6 = key[k_base + dims_base + 6usize];
            let k7 = key[k_base + dims_base + 7usize];

            let partial_a = qa0 * k0
                + qa1 * k1
                + qa2 * k2
                + qa3 * k3
                + qa4 * k4
                + qa5 * k5
                + qa6 * k6
                + qa7 * k7;
            let partial_b = qb0 * k0
                + qb1 * k1
                + qb2 * k2
                + qb3 * k3
                + qb4 * k4
                + qb5 * k5
                + qb6 * k6
                + qb7 * k7;
            let partial_c = qc0 * k0
                + qc1 * k1
                + qc2 * k2
                + qc3 * k3
                + qc4 * k4
                + qc5 * k5
                + qc6 * k6
                + qc7 * k7;
            let partial_d = qd0 * k0
                + qd1 * k1
                + qd2 * k2
                + qd3 * k3
                + qd4 * k4
                + qd5 * k5
                + qd6 * k6
                + qd7 * k7;
            // Reduce + broadcast within this plane's 32 lanes (tree order — the
            // same FP-equivalent class as the m16 kernel, unchanged per row).
            let score_a = plane_sum(partial_a) * scale;
            let score_b = plane_sum(partial_b) * scale;
            let score_c = plane_sum(partial_c) * scale;
            let score_d = plane_sum(partial_d) * scale;

            let in_causal_a = active_a && pos <= q_abs_a;
            let in_causal_b = active_b && pos <= q_abs_b;
            let in_causal_c = active_c && pos <= q_abs_c;
            let in_causal_d = active_d && pos <= q_abs_d;
            let masked_a = if in_causal_a {
                score_a
            } else {
                f32::new(-1e30f32)
            };
            let masked_b = if in_causal_b {
                score_b
            } else {
                f32::new(-1e30f32)
            };
            let masked_c = if in_causal_c {
                score_c
            } else {
                f32::new(-1e30f32)
            };
            let masked_d = if in_causal_d {
                score_d
            } else {
                f32::new(-1e30f32)
            };

            let new_max_a = if masked_a > run_max_a {
                masked_a
            } else {
                run_max_a
            };
            let new_max_b = if masked_b > run_max_b {
                masked_b
            } else {
                run_max_b
            };
            let new_max_c = if masked_c > run_max_c {
                masked_c
            } else {
                run_max_c
            };
            let new_max_d = if masked_d > run_max_d {
                masked_d
            } else {
                run_max_d
            };

            // Exp hoist per row: each plane-uniform exp runs on lane 0 ONLY and
            // is broadcast with one shuffle (the Bench 786 reduce-on-all-lanes
            // lesson; no shuffle inside a divergent branch).
            let w_a0 = if lane == 0u32 {
                (masked_a - new_max_a).exp()
            } else {
                f32::new(0.0f32)
            };
            let corr_a0 = if lane == 0u32 {
                (run_max_a - new_max_a).exp()
            } else {
                f32::new(0.0f32)
            };
            let w_b0 = if lane == 0u32 {
                (masked_b - new_max_b).exp()
            } else {
                f32::new(0.0f32)
            };
            let corr_b0 = if lane == 0u32 {
                (run_max_b - new_max_b).exp()
            } else {
                f32::new(0.0f32)
            };
            let w_c0 = if lane == 0u32 {
                (masked_c - new_max_c).exp()
            } else {
                f32::new(0.0f32)
            };
            let corr_c0 = if lane == 0u32 {
                (run_max_c - new_max_c).exp()
            } else {
                f32::new(0.0f32)
            };
            let w_d0 = if lane == 0u32 {
                (masked_d - new_max_d).exp()
            } else {
                f32::new(0.0f32)
            };
            let corr_d0 = if lane == 0u32 {
                (run_max_d - new_max_d).exp()
            } else {
                f32::new(0.0f32)
            };
            let w_a = plane_broadcast(w_a0, 0u32);
            let corr_a = plane_broadcast(corr_a0, 0u32);
            let w_b = plane_broadcast(w_b0, 0u32);
            let corr_b = plane_broadcast(corr_b0, 0u32);
            let w_c = plane_broadcast(w_c0, 0u32);
            let corr_c = plane_broadcast(corr_c0, 0u32);
            let w_d = plane_broadcast(w_d0, 0u32);
            let corr_d = plane_broadcast(corr_d0, 0u32);

            run_sum_a = run_sum_a * corr_a + w_a;
            run_sum_b = run_sum_b * corr_b + w_b;
            run_sum_c = run_sum_c * corr_c + w_c;
            run_sum_d = run_sum_d * corr_d + w_d;

            // ONE V row load shared by all four rows.
            let v_base = k_base;
            let v0 = value[v_base + dims_base];
            let v1 = value[v_base + dims_base + 1usize];
            let v2 = value[v_base + dims_base + 2usize];
            let v3 = value[v_base + dims_base + 3usize];
            let v4 = value[v_base + dims_base + 4usize];
            let v5 = value[v_base + dims_base + 5usize];
            let v6 = value[v_base + dims_base + 6usize];
            let v7 = value[v_base + dims_base + 7usize];

            oa0 = oa0 * corr_a + w_a * v0;
            oa1 = oa1 * corr_a + w_a * v1;
            oa2 = oa2 * corr_a + w_a * v2;
            oa3 = oa3 * corr_a + w_a * v3;
            oa4 = oa4 * corr_a + w_a * v4;
            oa5 = oa5 * corr_a + w_a * v5;
            oa6 = oa6 * corr_a + w_a * v6;
            oa7 = oa7 * corr_a + w_a * v7;
            ob0 = ob0 * corr_b + w_b * v0;
            ob1 = ob1 * corr_b + w_b * v1;
            ob2 = ob2 * corr_b + w_b * v2;
            ob3 = ob3 * corr_b + w_b * v3;
            ob4 = ob4 * corr_b + w_b * v4;
            ob5 = ob5 * corr_b + w_b * v5;
            ob6 = ob6 * corr_b + w_b * v6;
            ob7 = ob7 * corr_b + w_b * v7;
            oc0 = oc0 * corr_c + w_c * v0;
            oc1 = oc1 * corr_c + w_c * v1;
            oc2 = oc2 * corr_c + w_c * v2;
            oc3 = oc3 * corr_c + w_c * v3;
            oc4 = oc4 * corr_c + w_c * v4;
            oc5 = oc5 * corr_c + w_c * v5;
            oc6 = oc6 * corr_c + w_c * v6;
            oc7 = oc7 * corr_c + w_c * v7;
            od0 = od0 * corr_d + w_d * v0;
            od1 = od1 * corr_d + w_d * v1;
            od2 = od2 * corr_d + w_d * v2;
            od3 = od3 * corr_d + w_d * v3;
            od4 = od4 * corr_d + w_d * v4;
            od5 = od5 * corr_d + w_d * v5;
            od6 = od6 * corr_d + w_d * v6;
            od7 = od7 * corr_d + w_d * v7;
            run_max_a = new_max_a;
            run_max_b = new_max_b;
            run_max_c = new_max_c;
            run_max_d = new_max_d;

            pos += 1u32;
        }
    }

    // Epilogue, rows A-D — the m16 gated-attention shape per row (sigmoid
    // gate, one exp per output element).
    if active_a {
        let inv_sum = f32::new(1.0f32) / run_sum_a;
        let g_off = q_off_a + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = oa0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = oa1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = oa2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = oa3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = oa4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = oa5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = oa6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = oa7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_b {
        let inv_sum = f32::new(1.0f32) / run_sum_b;
        let g_off = q_off_b + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = ob0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = ob1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = ob2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = ob3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = ob4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = ob5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = ob6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = ob7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_c {
        let inv_sum = f32::new(1.0f32) / run_sum_c;
        let g_off = q_off_c + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = oc0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = oc1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = oc2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = oc3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = oc4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = oc5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = oc6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = oc7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_d {
        let inv_sum = f32::new(1.0f32) / run_sum_d;
        let g_off = q_off_d + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = od0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = od1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = od2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = od3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = od4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = od5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = od6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = od7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
}

// ---------------------------------------------------------------------------
// Launchers (the m32 launcher shapes: params_handle + the 65535 grid guard)
// ---------------------------------------------------------------------------

/// Launch the K-block pool (one cube per KV group × KV block) in the given
/// scorer mode.
#[cfg(feature = "cubecl_runtime")]
pub struct SparsePrefillPoolCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SparsePrefillPoolCubeCL {
    /// # Safety
    /// `key` holds `[cache_rows, n_kv, head_dim]` f32; `kpool` holds the
    /// TWO-HALF layout `[2 * n_kv * ceil(cache_rows/64) * head_dim]` f32.
    /// `head_dim` must be 256.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        key_handle: Handle,
        kpool_handle: Handle,
        head_dim: usize,
        n_kv: usize,
        cache_rows: usize,
        scorer: SparseScorer,
    ) {
        debug_assert_eq!(head_dim, 256, "sparse pool is head_dim-256 specialized");
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let half_elems = n_kv * n_kv_blocks * head_dim;
        let params: [f32; 6] = [
            head_dim as f32,
            n_kv as f32,
            cache_rows as f32,
            n_kv_blocks as f32,
            scorer.as_u32() as f32,
            half_elems as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        note_sparse_pool_launch();
        unsafe {
            sparse_prefill_pool_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static((n_kv * n_kv_blocks).max(1) as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(key_handle, cache_rows * n_kv * head_dim),
                BufferArg::from_raw_parts(kpool_handle, half_elems * 2),
                BufferArg::from_raw_parts(params_handle, 6),
            );
        }
    }
}

/// Launch the block scorer (grid = n_head × query windows, chunked on the
/// Metal 65535-workgroup guard).
#[cfg(feature = "cubecl_runtime")]
pub struct SparsePrefillScoreCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SparsePrefillScoreCubeCL {
    /// # Safety
    /// `query` holds the chunk-local `[p, n_head, head_dim]` f32
    /// (`p = cache_rows - base_pos`); `key` the full-cache
    /// `[cache_rows, n_kv, head_dim]`; `kpool` the TWO-HALF alloc_sparse_scratch
    /// shape, `scores` the alloc_sparse_scratch shape. `head_dim` must be 256.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_handle: Handle,
        kpool_handle: Handle,
        scores_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv: usize,
        cache_rows: usize,
        base_pos: usize,
        scorer: SparseScorer,
    ) {
        debug_assert_eq!(head_dim, 256, "sparse score is head_dim-256 specialized");
        const MAX_WG_X: u32 = 65535;
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK).max(1);
        let half_elems = n_kv * n_kv_blocks * head_dim;
        let per_chunk = (MAX_WG_X as usize / n_head.max(1)).max(1);
        let mut qb0 = 0usize;
        while qb0 < q_windows {
            let qc = per_chunk.min(q_windows - qb0);
            let params: [f32; 10] = [
                head_dim as f32,
                n_head as f32,
                n_kv as f32,
                cache_rows as f32,
                qc as f32,
                qb0 as f32,
                n_kv_blocks as f32,
                base_pos as f32,
                scorer.as_u32() as f32,
                half_elems as f32,
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            note_sparse_score_launch();
            unsafe {
                sparse_prefill_score_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static((n_head * qc) as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(
                        query_handle.clone(),
                        (cache_rows - base_pos) * n_head * head_dim,
                    ),
                    BufferArg::from_raw_parts(key_handle.clone(), cache_rows * n_kv * head_dim),
                    BufferArg::from_raw_parts(kpool_handle.clone(), half_elems * 2),
                    BufferArg::from_raw_parts(
                        scores_handle.clone(),
                        q_windows * n_head * n_kv_blocks,
                    ),
                    BufferArg::from_raw_parts(params_handle, 10),
                );
            }
            qb0 += qc;
        }
    }
}

/// Launch the τ-mass selection (grid = n_head × query windows, chunked).
#[cfg(feature = "cubecl_runtime")]
pub struct SparsePrefillSelectCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SparsePrefillSelectCubeCL {
    /// # Safety
    /// `scores`/`mask` the alloc_sparse_scratch shapes for `cache_rows`;
    /// `tau` in [0.5, 1.0].
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        scores_handle: Handle,
        mask_handle: Handle,
        n_head: usize,
        cache_rows: usize,
        tau: f32,
        base_pos: usize,
    ) {
        const MAX_WG_X: u32 = 65535;
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let wpr = n_kv_blocks.div_ceil(32);
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK).max(1);
        let per_chunk = (MAX_WG_X as usize / n_head.max(1)).max(1);
        let mut qb0 = 0usize;
        while qb0 < q_windows {
            let qc = per_chunk.min(q_windows - qb0);
            let params: [f32; 8] = [
                n_head as f32,
                cache_rows as f32,
                qc as f32,
                qb0 as f32,
                n_kv_blocks as f32,
                wpr as f32,
                tau,
                base_pos as f32,
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            note_sparse_select_launch();
            unsafe {
                sparse_prefill_select_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static((n_head * qc) as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(
                        scores_handle.clone(),
                        q_windows * n_head * n_kv_blocks,
                    ),
                    BufferArg::from_raw_parts(mask_handle.clone(), q_windows * n_head * wpr),
                    BufferArg::from_raw_parts(params_handle, 8),
                );
            }
            qb0 += qc;
        }
    }
}

/// Launch the sparse M=32 attention body. Identical to the m32 launcher
/// except the `mask` binding and the three extra params (wpr, the chunk's
/// absolute mask-window base, its 128-remainder).
#[cfg(feature = "cubecl_runtime")]
pub struct SparsePrefillSparseM32CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SparsePrefillSparseM32CubeCL {
    /// # Safety
    /// Same contract as the m32 kernel's launch — identical handle shapes and
    /// chunked `base_pos` cache semantics — plus `mask` holding
    /// `[q_windows * n_head * wpr]` u32 words and `head_mode` holding the
    /// padded per-layer mode words (`layer_idx · [`sparse_mode_layer_stride_words`]`
    /// sliced here; first `n_head` words of the slice carry the heads).
    /// `head_dim` must be 256.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_handle: Handle,
        value_handle: Handle,
        gate_handle: Handle,
        attn_out_handle: Handle,
        mask_handle: Handle,
        head_mode_handle: Handle,
        layer_idx: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) {
        const MAX_WG_X: u32 = 65535;

        debug_assert_eq!(
            head_dim, 256,
            "sparse m32 flash kernel is head_dim-256 specialized"
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let n_kv_blocks = (base_pos + p).div_ceil(SPARSE_KV_BLOCK);
        let wpr = n_kv_blocks.div_ceil(32);
        let q_windows = (base_pos + p).div_ceil(SPARSE_Q_BLOCK);
        // T5: this layer's slice of the padded mode buffer (the stride keeps
        // every layer offset 32-byte binding aligned).
        let mode_stride = sparse_mode_layer_stride_words(n_head);
        let mode_slice = head_mode_handle
            .clone()
            .offset_start((layer_idx * mode_stride * 4) as u64);
        // Grid = n_head × q_tiles (32 queries per cube) — chunk on query-TILE
        // boundaries with the same 65535 guard as the m16/m32 launchers.
        let q_tiles_total = p.div_ceil(SPARSE_ATTN_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 32).max(1);
        let mut t0 = 0usize; // query-token base of the chunk
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(SPARSE_ATTN_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * SPARSE_ATTN_Q_PER_CUBE as usize).min(p - t0);
            let q_abs = base_pos + t0;
            let params: [f32; 10] = [
                head_dim as f32,
                n_head as f32,
                n_kv_head as f32,
                tc as f32,
                scale,
                q_abs as f32,
                tiles as f32,
                wpr as f32,
                (q_abs / SPARSE_Q_BLOCK) as f32,
                (q_abs % SPARSE_Q_BLOCK) as f32,
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let q_len = tc * n_head * head_dim;
            let kv_len = (base_pos + p) * n_kv_head * head_dim;
            let q_slice = query_handle
                .clone()
                .offset_start((t0 * n_head * head_dim * 4) as u64);
            let g_slice = gate_handle
                .clone()
                .offset_start((t0 * n_head * head_dim * 4) as u64);
            let o_slice = attn_out_handle
                .clone()
                .offset_start((t0 * n_head * head_dim * 4) as u64);
            let n_cubes = n_head * tiles;
            note_sparse_attn_launch();
            unsafe {
                qwen_attention_prefill_sparse_m32_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(n_cubes as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(q_slice, q_len),
                    BufferArg::from_raw_parts(key_handle.clone(), kv_len),
                    BufferArg::from_raw_parts(value_handle.clone(), kv_len),
                    BufferArg::from_raw_parts(g_slice, q_len),
                    BufferArg::from_raw_parts(o_slice, q_len),
                    BufferArg::from_raw_parts(mask_handle.clone(), q_windows * n_head * wpr),
                    BufferArg::from_raw_parts(mode_slice.clone(), n_head),
                    BufferArg::from_raw_parts(params_handle, 10),
                );
            }
            t0 += tc;
        }
    }
}

// ---------------------------------------------------------------------------
// T4 — the boundary-row gather + the gate's chunk runner (Issue 1156)
// ---------------------------------------------------------------------------

/// Gather a chunk's boundary rows for the regime gate's readback: output row
/// `r < head_rows` copies input row `r`; output row `r >= head_rows` copies
/// input row `tail_start + (r - head_rows)`. The EXPLICIT `params` bound the
/// work (never `output.len()` — the len-derived-binding class); every param
/// stays exact in f32 at the lane's shapes (n ≤ 2^24, prompt lengths far
/// below it).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn boundary_gather_rows_f32(input: &[f32], output: &mut [f32], params: &[f32]) {
    let n = params[0usize] as u32;
    let head_rows = params[1usize] as u32;
    let tail_start = params[2usize] as u32;
    let total_rows = params[3usize] as u32;
    let tid = ABSOLUTE_POS;
    if tid < (total_rows * n) as usize {
        let r = tid as u32 / n;
        let c = tid as u32 % n;
        let src_row = if r < head_rows {
            r
        } else {
            tail_start + (r - head_rows)
        };
        output[tid] = input[(src_row * n + c) as usize];
    }
}

/// CubeCL launcher for the boundary gather: `output` sized
/// `(head_rows + tail_rows) * n` f32, `input` the chunk's residual stream
/// `[p, n]` with `tail_start + tail_rows <= p`.
#[cfg(feature = "cubecl_runtime")]
pub struct BoundaryGatherRowsCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl BoundaryGatherRowsCubeCL {
    /// # Safety
    /// `input_handle` holds ≥ `(tail_start + tail_rows) · n` f32;
    /// `output_handle` holds exactly `(head_rows + tail_rows) · n` f32.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        output_handle: Handle,
        n: usize,
        head_rows: usize,
        tail_start: usize,
        tail_rows: usize,
    ) {
        let total_rows = head_rows + tail_rows;
        let total = total_rows * n;
        let params: [f32; 4] = [
            n as f32,
            head_rows as f32,
            tail_start as f32,
            total_rows as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let n_wg = total.div_ceil(256).max(1) as u32;
        unsafe {
            boundary_gather_rows_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, (tail_start + tail_rows) * n),
                BufferArg::from_raw_parts(output_handle, total),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

/// The boundary-staging element count the scratch builder passes for a
/// chunk of `p` tokens at hidden width `n` — 0 when the gate is disarmed
/// (no alloc, the G3 posture). Kept HERE so the arming read and the
/// staging size can never drift apart (the caller never guesses).
#[cfg(feature = "cubecl_runtime")]
pub fn regime_gate_bnd_elems(p: usize, n: usize) -> usize {
    if crate::regime_gate::regime_gate_armed() {
        2 * crate::regime_gate::REGIME_POOL_TOKENS.min(p).max(1) * n
    } else {
        0
    }
}

/// The regime gate's chunk runner (T4): gather the chunk's boundary rows
/// from `x_b` (the residual stream, `[p, n]`, already staged), pool them
/// into the gate's features, classify, and — when the decision and the T3
/// belt both pass — override this chunk's τ. Returns the final engagement
/// (false = the chunk routes DENSE); on a materials failure it keeps the
/// plain arm (`true`, no override) — the operator's env posture stands,
/// the refusal counter tells the story.
///
/// Disarmed → immediate `true` (zero work, G3).
#[cfg(feature = "cubecl_runtime")]
pub fn run_regime_gate<R: Runtime>(
    client: &ComputeClient<R>,
    x_b: &Handle,
    p: usize,
    n: usize,
    prompt_len: usize,
    sp: &mut SparsePrefillScratch,
) -> bool {
    if !crate::regime_gate::regime_gate_armed() {
        return true;
    }
    let materials = match crate::regime_gate::gate_materials() {
        Ok(m) => m,
        Err(_) => {
            crate::regime_gate::note_regime_refusal();
            return true;
        }
    };
    let pool = crate::regime_gate::REGIME_POOL_TOKENS.min(p).max(1);
    let bnd = match sp.bnd.as_ref() {
        Some(h) => h.clone(),
        None => {
            // Armed but the staging buffer is missing — a scratch/gate
            // arming race. Fail closed to the plain arm (the buffer's
            // absence is the loud counter absence pattern).
            crate::regime_gate::note_regime_refusal();
            return true;
        }
    };
    let tail_start = p - pool;
    unsafe {
        BoundaryGatherRowsCubeCL::launch::<R>(
            client,
            x_b.clone(),
            bnd.clone(),
            n,
            pool,
            tail_start,
            pool,
        );
    }
    let bytes = match client.read_one(bnd.clone()) {
        Ok(b) => b,
        Err(_) => {
            crate::regime_gate::note_regime_refusal();
            return true;
        }
    };
    let gathered = f32::from_bytes(&bytes);
    if gathered.len() < 2 * pool * n {
        crate::regime_gate::note_regime_refusal();
        return true;
    }
    // The gather laid out [head rows; tail rows] — pooling that buffer at
    // n_tokens = 2·pool reads bit-identical features to pooling `x_b`
    // directly (the CPU-side equivalence is pinned in regime_gate's tests).
    let feats = crate::regime_gate::boundary_pool(gathered, 2 * pool, n, pool);
    let scorer = prefill_sparse_scorer();
    match crate::regime_gate::decide_chunk(&materials.0, &materials.1, &feats, prompt_len, scorer) {
        Ok(decision) => {
            if decision.engage_sparse {
                sp.tau_override = Some(decision.tau);
                true
            } else {
                crate::regime_gate::note_regime_dense_chunk();
                sp.sparse_engaged = false;
                false
            }
        }
        Err(_) => {
            crate::regime_gate::note_regime_refusal();
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use crate::qwen_attention_prefill_m32_cubecl::QwenAttentionPrefillTiledM32CubeCL;

    /// Deterministic pseudo-random data in a realistic range (LCG, no RNG dep
    /// — the sibling kernel-test helper shape).
    fn lcg_fill(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (x >> 40) as f32 / (1u64 << 24) as f32;
                lo + (hi - lo) * u
            })
            .collect()
    }

    /// G1's kernel-level anchor: with every in-horizon block selected
    /// (all-ones mask) the skip branch is dead, the executed sequence is
    /// m32's exactly, and the outputs must match BIT-FOR-BIT — ragged tiles,
    /// sub-cube tail and chunked-cache `base_pos` shapes all included.
    #[test]
    fn sparse_tau1_is_bit_identical_to_m32() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 6usize;
        let n_kv = 1usize;
        let hd = 256usize;
        for (p, base_pos) in [(300usize, 0usize), (300, 256), (256, 0)] {
            let cache_rows = base_pos + p;
            let q = lcg_fill(1, p * n_head * hd, -1.0, 1.0);
            let k = lcg_fill(2, cache_rows * n_kv * hd, -1.0, 1.0);
            let v = lcg_fill(3, cache_rows * n_kv * hd, -1.0, 1.0);
            let g = lcg_fill(4, p * n_head * hd, -1.0, 1.0);
            let q_h = client.create_from_slice(f32::as_bytes(&q));
            let k_h = client.create_from_slice(f32::as_bytes(&k));
            let v_h = client.create_from_slice(f32::as_bytes(&v));
            let g_h = client.create_from_slice(f32::as_bytes(&g));
            let out_a = client.empty(p * n_head * hd * 4);
            let out_b = client.empty(p * n_head * hd * 4);
            let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
            let wpr = sparse_mask_wpr(cache_rows);
            let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
            let words = vec![u32::MAX; q_windows * n_head * wpr];
            let mask_h = client.create_from_slice(u32::as_bytes(&words));
            // T5: an all-zero mode buffer (the incumbent all-sparse posture;
            // n_head words, one padded layer stride).
            let mode_words = vec![0u32; sparse_mode_layer_stride_words(n_head)];
            let mode_h = client.create_from_slice(u32::as_bytes(&mode_words));
            unsafe {
                QwenAttentionPrefillTiledM32CubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_h.clone(),
                    k_h.clone(),
                    v_h.clone(),
                    g_h.clone(),
                    out_a.clone(),
                    hd,
                    n_head,
                    n_kv,
                    p,
                    base_pos,
                );
                SparsePrefillSparseM32CubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_h.clone(),
                    k_h.clone(),
                    v_h.clone(),
                    g_h.clone(),
                    out_b.clone(),
                    mask_h.clone(),
                    mode_h.clone(),
                    0,
                    hd,
                    n_head,
                    n_kv,
                    p,
                    base_pos,
                );
            }
            let bytes_a = client.read_one(out_a.clone()).expect("readback");
            let bytes_b = client.read_one(out_b).expect("readback");
            let a = f32::from_bytes(&bytes_a);
            let b = f32::from_bytes(&bytes_b);
            let diffs = a
                .iter()
                .zip(b.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diffs, 0,
                "tau=1.0 sparse must be bit-identical to m32 at p={p} base_pos={base_pos} \
                 (n_kv_blocks={n_kv_blocks})"
            );
        }
    }

    /// Selection semantics over CRAFTED scores (no GPU scoring in the loop):
    /// the bisect set is the ranked-prefix set, sink blocks {0,1} always
    /// selected, τ=1.0 selects every in-horizon block, blocks beyond the
    /// window's causal horizon are never set.
    #[test]
    fn select_semantics_threshold_sink_identity_horizon() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 1usize;

        // Case 1: threshold + sink. Window 1 (queries 128..256, causal
        // horizon blocks_valid = 4). Crafted row [100, 1, 50, 25] over the
        // in-horizon blocks; total = 176, budget(τ=0.9) = 158.4 → ranked
        // prefix {0, 2, 3} (cum 100 → 150 → 175 ≥ 158.4) + sink {0, 1}
        // → bits {0,1,2,3}. Window 0's row stays zero → sink-only.
        let cache_rows = 512usize; // n_kv_blocks 8, q_windows 4, wpr 1
        let scores = {
            let mut s = vec![0.0f32; 4 * n_head * 8];
            let w1 = n_head * 8;
            s[w1] = 100.0;
            s[w1 + 1] = 1.0;
            s[w1 + 2] = 50.0;
            s[w1 + 3] = 25.0;
            s
        };
        let s_h = client.create_from_slice(f32::as_bytes(&scores));
        let m_h = client.empty(4 * n_head * sparse_mask_wpr(cache_rows) * 4);
        unsafe {
            SparsePrefillSelectCubeCL::launch::<ActiveRuntime>(
                &client,
                s_h.clone(),
                m_h.clone(),
                n_head,
                cache_rows,
                0.9,
                0,
            );
        }
        let word_bytes = client.read_one(m_h.clone()).expect("readback");
        let words = u32::from_bytes(&word_bytes);
        assert_eq!(words[0] & 0b11, 0b11, "zero-mass window keeps sink only");
        assert_eq!(words[0] & !0b11, 0, "zero-mass window selects nothing else");
        let w1 = words[1];
        assert_eq!(w1 & 0b1111, 0b1111, "threshold set blocks 0+2+3 + sink 0+1");
        assert_eq!(w1 & !0b1111, 0, "no extra blocks selected");

        // Case 2: the identity posture — τ=1.0 selects every in-horizon
        // block (bits 0..8 of window 0's word).
        unsafe {
            SparsePrefillSelectCubeCL::launch::<ActiveRuntime>(
                &client,
                s_h,
                m_h.clone(),
                n_head,
                cache_rows,
                1.0,
                0,
            );
        }
        let word_bytes = client.read_one(m_h.clone()).expect("readback");
        let words = u32::from_bytes(&word_bytes);
        assert_eq!(
            words[0] & 0b11,
            0b11,
            "tau=1.0 must select all in-horizon blocks (window 0: blocks 0..2)"
        );
        assert_eq!(
            words[1] & 0b1111,
            0b1111,
            "tau=1.0 must select all in-horizon blocks (window 1: blocks 0..4)"
        );

        // Case 3: non-positive total mass → sink-only (the well-formed
        // degenerate mask) at τ < 1.0.
        let mut neg = vec![0.0f32; 4 * n_head * 8];
        for cell in neg.iter_mut() {
            *cell = -3.5f32;
        }
        let n_h = client.create_from_slice(f32::as_bytes(&neg));
        unsafe {
            SparsePrefillSelectCubeCL::launch::<ActiveRuntime>(
                &client,
                n_h,
                m_h.clone(),
                n_head,
                cache_rows,
                0.9,
                0,
            );
        }
        let word_bytes = client.read_one(m_h.clone()).expect("readback");
        let words = u32::from_bytes(&word_bytes);
        assert_eq!(words[1] & 0b11, 0b11, "sink always set");
        assert_eq!(
            words[1] & !0b11,
            0,
            "non-positive mass selects nothing else"
        );

        // Case 4: the causal horizon. cache_rows = 320 → n_kv_blocks = 5,
        // q_windows = 3; window 1 covers absolute queries [128, 256),
        // blocks_valid = 4 — so block 4 (rows 256..320) exists in the buffer
        // but lies beyond window 1's horizon. Its hot score (9, ranked
        // first) must never be selected for window 1 even though the
        // ranking would admit it.
        let cache_rows = 320usize;
        let n_kv_blocks = 5usize;
        let q_windows = 3usize;
        let mut scores = vec![0.0f32; q_windows * n_head * n_kv_blocks];
        // window 1 row: [5, 0, 5, 0, 9] — block 4 beyond the horizon.
        let w1 = n_head * n_kv_blocks;
        scores[w1] = 5.0;
        scores[w1 + 2] = 5.0;
        scores[w1 + 4] = 9.0;
        let s_h = client.create_from_slice(f32::as_bytes(&scores));
        let m_h = client.empty(q_windows * n_head * sparse_mask_wpr(cache_rows) * 4);
        unsafe {
            SparsePrefillSelectCubeCL::launch::<ActiveRuntime>(
                &client,
                s_h,
                m_h.clone(),
                n_head,
                cache_rows,
                0.9,
                0,
            );
        }
        let word_bytes = client.read_one(m_h).expect("readback");
        let words = u32::from_bytes(&word_bytes);
        assert_eq!(
            words[1] & (1u32 << 4),
            0,
            "a block beyond the causal horizon must never be selected"
        );
        assert_eq!(
            words[1] & 0b111,
            0b111,
            "in-horizon ranked set blocks 0+2 + sink blocks 0+1"
        );
        assert_eq!(words[1] & !0b111, 0, "no other blocks selected");
    }

    /// Structural scorer check: a single hot (Q,K) pair ON the diagonal of
    /// window 1 (query 128 vs key 192, both inside block 3's reach and the
    /// window's causal horizon) must make block 3's score dominate its
    /// zero neighbours — in BOTH scorer modes (the centroid probe fires
    /// through the mean pool; the envelope fires through the block max).
    /// The band term stays empty for this pairing (the hot query is the
    /// window's FIRST row, far above the diagonal cells of blocks 0..2).
    #[test]
    fn score_ranks_the_hot_block() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 1usize;
        let n_kv = 1usize;
        let hd = 256usize;
        let cache_rows = 512usize;
        let base_pos = 0usize;
        let p = 512usize;
        let mut q = vec![0.0f32; p * n_head * hd];
        let mut k = vec![0.0f32; cache_rows * n_kv * hd];
        for d in 0..hd {
            q[128 * hd + d] = 3.0; // query row 128 = window 1's first row
            k[192 * hd + d] = 3.0; // key row 192 = block 3 (inside win-1 horizon)
        }
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
        for scorer in [SparseScorer::Mean, SparseScorer::Envelope] {
            let q_h = client.create_from_slice(f32::as_bytes(&q));
            let k_h = client.create_from_slice(f32::as_bytes(&k));
            let kpool_h = client.empty(n_kv * n_kv_blocks * hd * 2 * 4);
            let scores_h = client.empty(q_windows * n_head * n_kv_blocks * 4);
            unsafe {
                SparsePrefillPoolCubeCL::launch::<ActiveRuntime>(
                    &client,
                    k_h.clone(),
                    kpool_h.clone(),
                    hd,
                    n_kv,
                    cache_rows,
                    scorer,
                );
                SparsePrefillScoreCubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_h,
                    k_h,
                    kpool_h,
                    scores_h.clone(),
                    hd,
                    n_head,
                    n_kv,
                    cache_rows,
                    base_pos,
                    scorer,
                );
            }
            let score_bytes = client.read_one(scores_h).expect("readback");
            let scores = f32::from_bytes(&score_bytes);
            // Window 1 row = [n_kv_blocks .. 2*n_kv_blocks).
            let w1 = n_kv_blocks;
            let s3 = scores[w1 + 3];
            let s0 = scores[w1];
            let s1 = scores[w1 + 1];
            let s2 = scores[w1 + 2];
            assert!(
                s3 > s0 + 0.1 && s3 > s1 + 0.1 && s3 > s2 + 0.1,
                "[{:?}] hot block 3 ({s3}) must dominate its zero neighbours ({s0}, {s1}, {s2})",
                scorer
            );
            assert!(
                s3 > 0.1,
                "[{scorer:?}] block probe must be positive on the hot pair"
            );
        }
    }

    /// The T3-v2 mechanism proof: a needle row that is 1-of-64 in its block,
    /// aligned with the query on only a QUARTER of the dims (the realistic
    /// sparse-alignment shape), far off the diagonal — the mean scorer
    /// washes it into the block average (a ~64× dilution on the aligned
    /// dims), the envelope keeps the extremes at full magnitude. Window 1's
    /// diagonal-range K rows are zeroed (except the needle) so the band term
    /// is exactly 0 on every compared block and the read is pure probe.
    #[test]
    fn envelope_survives_a_single_needle_row_the_mean_washes() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 1usize;
        let n_kv = 1usize;
        let hd = 256usize;
        let cache_rows = 512usize;
        let base_pos = 0usize;
        let p = 512usize;
        let needle_block = 2usize; // far of the diagonal for window 1
        let needle_row = needle_block * SPARSE_KV_BLOCK + 17; // 1-of-64, mid-block
        let needle_q_row = 128usize; // window 1's first row
        let mut q = vec![0.0f32; p * n_head * hd];
        let mut k = lcg_fill(7, cache_rows * n_kv * hd, -0.05, 0.05);
        // Zero window 1's diagonal range (rows 128..256) except the needle:
        // the band cells there pair q_row(128+x)·k_row(128+x), and every
        // such q row except 128 is zero — this makes the band exactly 0 for
        // every block window 1 scores, so the test reads the probe alone.
        for row in 128..256 {
            if row != needle_row {
                k[row * hd..(row + 1) * hd].fill(0.0);
            }
        }
        for d in 0..hd {
            // The needle and its query align on every 4th dim only.
            if d % 4 == 0 {
                q[needle_q_row * hd + d] = 3.0;
                k[needle_row * hd + d] = 3.0;
            }
        }
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
        let w1 = n_kv_blocks; // window 1's score row
        let mut sep_mean = 0.0f32;
        let mut sep_env = 0.0f32;
        for scorer in [SparseScorer::Mean, SparseScorer::Envelope] {
            let q_h = client.create_from_slice(f32::as_bytes(&q));
            let k_h = client.create_from_slice(f32::as_bytes(&k));
            let kpool_h = client.empty(n_kv * n_kv_blocks * hd * 2 * 4);
            let scores_h = client.empty(q_windows * n_head * n_kv_blocks * 4);
            unsafe {
                SparsePrefillPoolCubeCL::launch::<ActiveRuntime>(
                    &client,
                    k_h.clone(),
                    kpool_h.clone(),
                    hd,
                    n_kv,
                    cache_rows,
                    scorer,
                );
                SparsePrefillScoreCubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_h,
                    k_h,
                    kpool_h,
                    scores_h.clone(),
                    hd,
                    n_head,
                    n_kv,
                    cache_rows,
                    base_pos,
                    scorer,
                );
            }
            let score_bytes = client.read_one(scores_h).expect("readback");
            let scores = f32::from_bytes(&score_bytes);
            let needle_score = scores[w1 + needle_block];
            let best_other = (0..4usize)
                .filter(|&b| b != needle_block)
                .map(|b| scores[w1 + b])
                .fold(f32::NEG_INFINITY, f32::max);
            if scorer == SparseScorer::Mean {
                sep_mean = needle_score - best_other;
            } else {
                sep_env = needle_score - best_other;
            }
        }
        assert!(
            sep_mean > 0.0,
            "mean separation {sep_mean} must stay positive for the ratio read"
        );
        assert!(
            sep_env > sep_mean * 16.0,
            "envelope separation {sep_env} must exceed 16x the mean's {sep_mean} \
             (the mean washes the 1-of-64 needle into the block average)"
        );
    }

    /// The env-arm grammar: exact-literal opt-in, the τ clamp, and the
    /// prompt-level engagement gate (the Issue-782 lesson — a stale long
    /// `active_prefill_len` must not be the only thing consulted; the gate
    /// takes the live value the caller passes).
    #[test]
    fn sparse_env_grammar() {
        set_prefill_sparse(true);
        assert!(prefill_sparse_enabled());
        assert!(prefill_sparse_engaged(16384));
        assert!(!prefill_sparse_engaged(16383));
        assert!(!prefill_sparse_engaged(0));
        set_prefill_sparse(false);
        assert!(!prefill_sparse_enabled());
        assert!(!prefill_sparse_engaged(usize::MAX));

        set_prefill_sparse_tau(2.0);
        assert_eq!(prefill_sparse_tau(), 1.0, "τ clamps at the identity");
        set_prefill_sparse_tau(0.1);
        assert_eq!(prefill_sparse_tau(), 0.5, "τ clamps at the floor");
        set_prefill_sparse_tau(0.9);
        assert_eq!(prefill_sparse_tau(), 0.9);
        set_prefill_sparse(false);

        // The scorer posture (protocol v2): setter overrides, both spellings
        // parse, the default is the envelope.
        set_prefill_sparse_scorer(SparseScorer::Mean);
        assert_eq!(prefill_sparse_scorer(), SparseScorer::Mean);
        set_prefill_sparse_scorer(SparseScorer::Envelope);
        assert_eq!(prefill_sparse_scorer(), SparseScorer::Envelope);
        assert_eq!(SparseScorer::parse("mean"), Some(SparseScorer::Mean));
        assert_eq!(
            SparseScorer::parse("envelope"),
            Some(SparseScorer::Envelope)
        );
        assert_eq!(SparseScorer::parse("junk"), None);
        assert_eq!(SparseScorer::Envelope.as_str(), "envelope");
        assert_eq!(SparseScorer::Mean.as_u32(), 0);
    }

    // -- T5 pure helpers ------------------------------------------------------

    #[test]
    fn dense_head_mask_parses_ranges_and_refuses_junk() {
        assert_eq!(parse_dense_head_mask(""), Ok(0));
        assert_eq!(parse_dense_head_mask("   "), Ok(0));
        assert_eq!(parse_dense_head_mask("3"), Ok(1 << 3));
        assert_eq!(parse_dense_head_mask("0-3"), Ok(0b1111));
        assert_eq!(
            parse_dense_head_mask("0-3,7,12"),
            Ok(0b1111 | 1 << 7 | 1 << 12)
        );
        assert_eq!(parse_dense_head_mask(" 5 - 9 , 1 "), Ok((0b11111 << 5) | 2));
        // Overlaps union (idempotent).
        assert_eq!(parse_dense_head_mask("2-4,3-5"), Ok(0b1111 << 2));
        // Junk ANYWHERE demotes the whole spec.
        assert!(parse_dense_head_mask("0-3,x").is_err());
        assert!(parse_dense_head_mask("0,,2").is_err());
        assert!(parse_dense_head_mask("5-2").is_err(), "inverted range");
        assert!(parse_dense_head_mask("-3").is_err(), "a bare range start");
        assert!(
            parse_dense_head_mask("999").is_err(),
            "past the bitmask cap"
        );
    }

    #[test]
    fn mode_layout_helpers_pin_the_alignment_law() {
        // The stride keeps every layer offset 32-byte aligned (8-word
        // multiples) at every head count.
        for n_head in [1usize, 6, 8, 20, 32, 40] {
            let stride = sparse_mode_layer_stride_words(n_head);
            assert_eq!(stride % 8, 0, "stride must be 8-word aligned");
            assert!(stride >= n_head);
        }
        // 20 heads → ceil(20/8)=3 → 24 words (96 bytes).
        assert_eq!(sparse_mode_layer_stride_words(20), 24);
        // The padded layout copies each layer's heads into its stride slot,
        // zero-fills the padding, and preserves word values.
        let mirror: Vec<u32> = (0..3 * 20).map(|i| (i % 7 + 1) as u32).collect();
        let padded = sparse_mode_padded_words(&mirror, 3, 20);
        assert_eq!(padded.len(), 3 * 24);
        for layer in 0..3usize {
            for h in 0..20usize {
                assert_eq!(padded[layer * 24 + h], mirror[layer * 20 + h]);
            }
            for pad in 20..24 {
                assert_eq!(padded[layer * 24 + pad], 0, "padding stays zero");
            }
        }
        // layer_dense reads the logic-shaped mirror.
        let mut m = vec![0u32; 4 * 6];
        assert!(!sparse_layer_dense(&m, 1, 6));
        m[2 * 6 + 3] = 1;
        assert!(!sparse_layer_dense(&m, 0, 6));
        assert!(sparse_layer_dense(&m, 2, 6));
        assert!(!sparse_layer_dense(&m, 3, 6));
    }

    // -- T5 GPU battery (the G1 anchors) --------------------------------------
    // Mixed-mode contract: a DENSE head's cubes never consult the mask, an
    // all-zero mode is the incumbent sparse posture, and a mixed launch reads
    // as the per-head union of the single-mode runs.

    /// Build the shared fixtures + run the select pipeline at τ, returning
    /// (handles, mask words host-side). The three attention launches reuse
    /// the SAME q/k/v/g/mask so only the mode buffer differs.
    #[allow(clippy::type_complexity)]
    fn mixed_fixtures(
        client: &ComputeClient<ActiveRuntime>,
        p: usize,
        base_pos: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        tau: f32,
    ) -> (Handle, Handle, Handle, Handle, Handle, Vec<u32>) {
        let cache_rows = base_pos + p;
        let q = lcg_fill(11, p * n_head * hd, -1.0, 1.0);
        let k = lcg_fill(12, cache_rows * n_kv * hd, -1.0, 1.0);
        let v = lcg_fill(13, cache_rows * n_kv * hd, -1.0, 1.0);
        let g = lcg_fill(14, p * n_head * hd, -1.0, 1.0);
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let k_h = client.create_from_slice(f32::as_bytes(&k));
        let v_h = client.create_from_slice(f32::as_bytes(&v));
        let g_h = client.create_from_slice(f32::as_bytes(&g));
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
        let wpr = sparse_mask_wpr(cache_rows);
        // Pool + score + select at τ < 1.0 — a REAL lossy mask (some blocks
        // unset), the shape mixed-mode exists for.
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let kpool_elems = n_kv * n_kv_blocks * hd * 2;
        let kpool = client.empty(kpool_elems * 4);
        let scores = client.empty(q_windows * n_head * n_kv_blocks * 4);
        let mask_h = client.empty(q_windows * n_head * wpr * 4);
        unsafe {
            SparsePrefillPoolCubeCL::launch::<ActiveRuntime>(
                client,
                k_h.clone(),
                kpool.clone(),
                hd,
                n_kv,
                cache_rows,
                SparseScorer::Envelope,
            );
            SparsePrefillScoreCubeCL::launch::<ActiveRuntime>(
                client,
                q_h.clone(),
                k_h.clone(),
                kpool.clone(),
                scores.clone(),
                hd,
                n_head,
                n_kv,
                cache_rows,
                base_pos,
                SparseScorer::Envelope,
            );
            SparsePrefillSelectCubeCL::launch::<ActiveRuntime>(
                client,
                scores.clone(),
                mask_h.clone(),
                n_head,
                cache_rows,
                tau,
                base_pos,
            );
        }
        let mask_bytes = client.read_one(mask_h.clone()).expect("mask readback");
        (
            q_h,
            k_h,
            v_h,
            g_h,
            mask_h,
            u32::from_bytes(&mask_bytes).to_vec(),
        )
    }

    fn launch_sparse_with_mode(
        client: &ComputeClient<ActiveRuntime>,
        q: &Handle,
        k: &Handle,
        v: &Handle,
        g: &Handle,
        out: &Handle,
        mask: &Handle,
        mode_words: &[u32],
        n_head: usize,
        p: usize,
        base_pos: usize,
        hd: usize,
        n_kv: usize,
        layer_idx: usize,
    ) {
        let mode_h = client.create_from_slice(u32::as_bytes(mode_words));
        unsafe {
            SparsePrefillSparseM32CubeCL::launch::<ActiveRuntime>(
                client,
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                out.clone(),
                mask.clone(),
                mode_h,
                layer_idx,
                hd,
                n_head,
                n_kv,
                p,
                base_pos,
            );
        }
    }

    fn head_rows_differ(
        a: &[f32],
        b: &[f32],
        head: usize,
        n_head: usize,
        hd: usize,
        p: usize,
    ) -> usize {
        let lo = head * hd;
        let mut diffs = 0usize;
        for t in 0..p {
            let base = t * n_head * hd + lo;
            for i in 0..hd {
                if a[base + i].to_bits() != b[base + i].to_bits() {
                    diffs += 1;
                }
            }
        }
        diffs
    }

    #[test]
    fn all_dense_mode_ignores_the_mask_and_is_bit_identical_to_m32() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 6usize;
        let n_kv = 1usize;
        let hd = 256usize;
        let p = 300usize;
        let base_pos = 0usize;
        let (q, k, v, g, _mask, mask_words) =
            mixed_fixtures(&client, p, base_pos, n_head, n_kv, hd, 0.7);
        // The mask must actually be lossy (some block bit unset) — else this
        // test proves nothing about the mode bit.
        let lossy = mask_words.iter().any(|&w| w != u32::MAX && w != 0);
        assert!(lossy, "the τ=0.7 fixture mask must leave blocks unset");
        let out_dense = client.empty(p * n_head * hd * 4);
        let out_m32 = client.empty(p * n_head * hd * 4);
        // All-ones mode: one word per head (layer 0's padded stride slot).
        let mode_all_dense = vec![1u32; sparse_mode_layer_stride_words(n_head)];
        launch_sparse_with_mode(
            &client,
            &q,
            &k,
            &v,
            &g,
            &out_dense,
            &_mask,
            &mode_all_dense,
            n_head,
            p,
            base_pos,
            hd,
            n_kv,
            0,
        );
        let cache_rows = base_pos + p;
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
        let wpr = sparse_mask_wpr(cache_rows);
        unsafe {
            QwenAttentionPrefillTiledM32CubeCL::launch::<ActiveRuntime>(
                &client,
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                out_m32.clone(),
                hd,
                n_head,
                n_kv,
                p,
                base_pos,
            );
        }
        let bytes_d = client.read_one(out_dense).expect("readback");
        let bytes_m = client.read_one(out_m32).expect("readback");
        let d = f32::from_bytes(&bytes_d);
        let m = f32::from_bytes(&bytes_m);
        let diffs = d
            .iter()
            .zip(m.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "all-dense mode must bypass the mask entirely and equal m32 bit-for-bit"
        );
    }

    #[test]
    fn all_zero_mode_tau1_stays_bit_identical_to_m32() {
        // The incumbent anchor re-run THROUGH the mode arg: all-zero words +
        // an all-ones mask at τ semantics (the mask built by the fixture's
        // select at τ=1.0 — the identity posture) == m32.
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 6usize;
        let n_kv = 1usize;
        let hd = 256usize;
        let p = 256usize;
        let base_pos = 256usize;
        let (q, k, v, g, mask, _) = mixed_fixtures(&client, p, base_pos, n_head, n_kv, hd, 1.0);
        let out = client.empty(p * n_head * hd * 4);
        let mode_all_zero = vec![0u32; sparse_mode_layer_stride_words(n_head)];
        launch_sparse_with_mode(
            &client,
            &q,
            &k,
            &v,
            &g,
            &out,
            &mask,
            &mode_all_zero,
            n_head,
            p,
            base_pos,
            hd,
            n_kv,
            0,
        );
        let out_m32 = client.empty(p * n_head * hd * 4);
        unsafe {
            QwenAttentionPrefillTiledM32CubeCL::launch::<ActiveRuntime>(
                &client,
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                out_m32.clone(),
                hd,
                n_head,
                n_kv,
                p,
                base_pos,
            );
        }
        let bytes_a = client.read_one(out).expect("readback");
        let bytes_b = client.read_one(out_m32).expect("readback");
        let a = f32::from_bytes(&bytes_a);
        let b = f32::from_bytes(&bytes_b);
        let diffs = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "all-zero mode at τ=1.0 == m32 (the T2 anchor via the T5 wire)"
        );
    }

    #[test]
    fn mixed_mode_equals_the_union_of_single_mode_runs() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n_head = 6usize;
        let n_kv = 1usize;
        let hd = 256usize;
        let p = 300usize;
        let base_pos = 0usize;
        let (q, k, v, g, mask, mask_words) =
            mixed_fixtures(&client, p, base_pos, n_head, n_kv, hd, 0.7);
        assert!(
            mask_words.iter().any(|&w| w != u32::MAX && w != 0),
            "the fixture mask must be lossy for the union claim to bind"
        );
        // Run A: all-sparse (mode zero). Run B: all-dense (mode ones).
        let out_sparse = client.empty(p * n_head * hd * 4);
        let out_dense = client.empty(p * n_head * hd * 4);
        let mode_zero = vec![0u32; sparse_mode_layer_stride_words(n_head)];
        let mode_ones = vec![1u32; sparse_mode_layer_stride_words(n_head)];
        launch_sparse_with_mode(
            &client,
            &q,
            &k,
            &v,
            &g,
            &out_sparse,
            &mask,
            &mode_zero,
            n_head,
            p,
            base_pos,
            hd,
            n_kv,
            0,
        );
        launch_sparse_with_mode(
            &client, &q, &k, &v, &g, &out_dense, &mask, &mode_ones, n_head, p, base_pos, hd, n_kv,
            0,
        );
        // Run C: MIXED — heads {0, 2, 4} dense, {1, 3, 5} sparse — in ONE
        // launch through the same launcher (the no-serial-split law).
        let out_mixed = client.empty(p * n_head * hd * 4);
        let mut mode_mixed = vec![0u32; sparse_mode_layer_stride_words(n_head)];
        for h in [0usize, 2, 4] {
            mode_mixed[h] = 1;
        }
        launch_sparse_with_mode(
            &client,
            &q,
            &k,
            &v,
            &g,
            &out_mixed,
            &mask,
            &mode_mixed,
            n_head,
            p,
            base_pos,
            hd,
            n_kv,
            0,
        );
        let bytes_s = client.read_one(out_sparse).expect("readback");
        let bytes_d = client.read_one(out_dense).expect("readback");
        let bytes_m = client.read_one(out_mixed).expect("readback");
        let s = f32::from_bytes(&bytes_s);
        let d = f32::from_bytes(&bytes_d);
        let m = f32::from_bytes(&bytes_m);
        for h in 0..n_head {
            let (want, name) = if h % 2 == 0 {
                (&d, "dense")
            } else {
                (&s, "sparse")
            };
            let diffs = head_rows_differ(&m, want, h, n_head, hd, p);
            assert_eq!(
                diffs, 0,
                "mixed head {h} (mode {name}) must match the single-mode run bit-for-bit"
            );
        }
        // And the two single-mode runs must genuinely differ somewhere (the
        // fixture must make the union claim non-vacuous).
        let any = (0..n_head)
            .map(|h| head_rows_differ(&s, &d, h, n_head, hd, p))
            .sum::<usize>();
        assert!(
            any > 0,
            "the τ=0.7 fixture must make dense ≠ sparse somewhere"
        );
    }

    #[test]
    fn boundary_gather_matches_the_cpu_rows() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let n = 64usize; // a small row width — the gather is width-agnostic
        let p = 40usize;
        let pool = 8usize;
        let hid: Vec<f32> = (0..p * n).map(|i| (i as f32) * 0.25 - 7.0).collect();
        let x_h = client.create_from_slice(f32::as_bytes(&hid));
        let bnd = client.empty(2 * pool * n * 4);
        let tail_start = p - pool;
        unsafe {
            BoundaryGatherRowsCubeCL::launch::<ActiveRuntime>(
                &client,
                x_h.clone(),
                bnd.clone(),
                n,
                pool,
                tail_start,
                pool,
            );
        }
        let bytes_got = client.read_one(bnd).expect("readback");
        let got = f32::from_bytes(&bytes_got);
        let mut want = Vec::with_capacity(2 * pool * n);
        want.extend_from_slice(&hid[0..pool * n]);
        want.extend_from_slice(&hid[tail_start * n..p * n]);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "the gather must copy exact bits");
        }
        // And the CPU pooling equivalence the gate relies on: pooling the
        // gathered buffer == pooling the original, bit-for-bit.
        let direct = crate::regime_gate::boundary_pool(&hid, p, n, pool);
        let from_gather = crate::regime_gate::boundary_pool(&got, 2 * pool, n, pool);
        for (a, b) in direct.iter().zip(from_gather.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
