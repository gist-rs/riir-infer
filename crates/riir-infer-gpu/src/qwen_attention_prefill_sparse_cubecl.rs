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
//!    `score = band + centroid`:
//!    - `band`: the sum of the block's cells ON the global causal diagonal
//!      (the XAttention antidiagonal idea generalized to rectangular blocks;
//!      every band cell visited exactly once — no undersampling). Catches the
//!      local/streaming mass. Key-index shortcut: a band cell for window-local
//!      query `x` is the key row `q_start + x` (the main diagonal), so the
//!      kernel needs no per-term branch.
//!    - `centroid`: ⟨mean Q window, mean K block⟩ (the FlexPrefill pooled
//!      probe). Catches retrieval-aligned blocks the band misses. Known
//!      dilution risk: a single needle row inside a 64-row block washes out
//!      in the mean — the quality walk (Plan 624 T2.4) is the instrument that
//!      judges this; iterate the scorer there if needle families tank.
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
    /// Mean K vector per (KV group, KV block): `[n_kv * n_kv_blocks * head_dim]`.
    pub kpool: cubecl::server::Handle,
    /// Block scores per (window, head, KV block): `[q_windows * n_head * n_kv_blocks]`.
    pub scores: cubecl::server::Handle,
    /// Selection bitmask per (window, head): `[q_windows * n_head * wpr]` u32 words.
    pub mask: cubecl::server::Handle,
    /// The cache-row count everything was sized for (`base_pos + p`).
    pub cache_rows: usize,
}

/// Derive the scratch handles for one chunk. Returns None on allocation
/// failure (the dispatch then falls through to the dense ladder — a loud
/// counter absence, never a wrong answer).
#[cfg(feature = "cubecl_runtime")]
pub fn alloc_sparse_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    cache_rows: usize,
) -> Option<SparsePrefillScratch> {
    let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
    let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
    let wpr = n_kv_blocks.div_ceil(32);
    Some(SparsePrefillScratch {
        kpool: client.empty(n_kv * n_kv_blocks * head_dim * 4),
        scores: client.empty(q_windows * n_head * n_kv_blocks * 4),
        mask: client.empty(q_windows * n_head * wpr * 4),
        cache_rows,
    })
}

/// Mask words per (window, head) row for a cache-row capacity — the shared
/// derivation (launcher-side and test-side).
#[cfg(feature = "cubecl_runtime")]
pub fn sparse_mask_wpr(cache_rows: usize) -> usize {
    cache_rows.div_ceil(SPARSE_KV_BLOCK).div_ceil(32)
}

// ---------------------------------------------------------------------------
// Kernel 1 — the K-block mean pool (per KV group × KV block)
// ---------------------------------------------------------------------------

/// Per (KV group, KV block): mean K vector over the block's valid rows.
/// Layout: `kpool[(kv_group * n_kv_blocks + jb) * head_dim + d]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn sparse_prefill_pool_f32(key: &[f32], kpool: &mut [f32], params: &[f32]) {
    let head_dim = params[0usize] as u32;
    let n_kv = params[1usize] as u32;
    let cache_rows = params[2usize] as u32;
    let n_kv_blocks = params[3usize] as u32;

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
    }
    if r1 < row_hi {
        let base = (r1 * kv_stride + kv_head_off) as usize + dims_base;
        s0 += key[base];
        s1 += key[base + 1usize];
        s2 += key[base + 2usize];
        s3 += key[base + 3usize];
        s4 += key[base + 4usize];
        s5 += key[base + 5usize];
        s6 += key[base + 6usize];
        s7 += key[base + 7usize];
    }

    // Tree reduce per dim (deterministic butterfly), then mean.
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
    let out_base = (cube_id * head_dim) as usize + dims_base;
    kpool[out_base] = t0 * inv;
    kpool[out_base + 1usize] = t1 * inv;
    kpool[out_base + 2usize] = t2 * inv;
    kpool[out_base + 3usize] = t3 * inv;
    kpool[out_base + 4usize] = t4 * inv;
    kpool[out_base + 5usize] = t5 * inv;
    kpool[out_base + 6usize] = t6 * inv;
    kpool[out_base + 7usize] = t7 * inv;
}

// ---------------------------------------------------------------------------
// Kernel 2 — the block scorer (diagonal band + pooled-centroid probe)
// ---------------------------------------------------------------------------

/// Per (head, 128-query window, KV block ≤ the window's diagonal horizon):
/// `scores[(qb * n_head + head) * n_kv_blocks + jb] = band + centroid`.
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

            // Pooled-centroid probe against this block's mean K.
            let kp = ((kv_group * n_kv_blocks + jb) * head_dim) as usize + dims_base;
            let km0 = kpool[kp];
            let km1 = kpool[kp + 1usize];
            let km2 = kpool[kp + 2usize];
            let km3 = kpool[kp + 3usize];
            let km4 = kpool[kp + 4usize];
            let km5 = kpool[kp + 5usize];
            let km6 = kpool[kp + 6usize];
            let km7 = kpool[kp + 7usize];
            let cp = qm0 * km0
                + qm1 * km1
                + qm2 * km2
                + qm3 * km3
                + qm4 * km4
                + qm5 * km5
                + qm6 * km6
                + qm7 * km7;
            let centroid = plane_sum(cp);

            if lane == 0u32 {
                scores[(row_base + jb) as usize] = band + centroid;
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
/// block-sparse KV skip (Issue 1156 T2). The per-row arithmetic is the m32
/// shape VERBATIM (the bit-identity anchor at τ=1.0); the ONLY change is the
/// mask test at KV-block boundaries and the block jump when the bit is
/// clear. Buffer contract = the m32 kernel's + `mask` and three params:
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
        //    set) the executed sequence is m32's exactly.
        let blk = pos / SPARSE_KV_BLOCK_U32;
        let word = mask[mask_row + (blk / 32u32) as usize];
        let is_boundary = pos % SPARSE_KV_BLOCK_U32 == 0u32;
        let bit_clear = ((word >> (blk % 32u32)) & 1u32) == 0u32;
        let skip = is_boundary && bit_clear;
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

/// Launch the K-block mean pool (one cube per KV group × KV block).
#[cfg(feature = "cubecl_runtime")]
pub struct SparsePrefillPoolCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SparsePrefillPoolCubeCL {
    /// # Safety
    /// `key` holds `[cache_rows, n_kv, head_dim]` f32; `kpool` holds
    /// `[n_kv * ceil(cache_rows/64) * head_dim]` f32. `head_dim` must be 256.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        key_handle: Handle,
        kpool_handle: Handle,
        head_dim: usize,
        n_kv: usize,
        cache_rows: usize,
    ) {
        debug_assert_eq!(head_dim, 256, "sparse pool is head_dim-256 specialized");
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let params: [f32; 4] = [
            head_dim as f32,
            n_kv as f32,
            cache_rows as f32,
            n_kv_blocks as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        note_sparse_pool_launch();
        unsafe {
            sparse_prefill_pool_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static((n_kv * n_kv_blocks).max(1) as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(key_handle, cache_rows * n_kv * head_dim),
                BufferArg::from_raw_parts(kpool_handle, n_kv * n_kv_blocks * head_dim),
                BufferArg::from_raw_parts(params_handle, 4),
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
    /// `[cache_rows, n_kv, head_dim]`; `kpool`/`scores` the alloc_sparse_scratch
    /// shapes. `head_dim` must be 256.
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
    ) {
        debug_assert_eq!(head_dim, 256, "sparse score is head_dim-256 specialized");
        const MAX_WG_X: u32 = 65535;
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK).max(1);
        let per_chunk = (MAX_WG_X as usize / n_head.max(1)).max(1);
        let mut qb0 = 0usize;
        while qb0 < q_windows {
            let qc = per_chunk.min(q_windows - qb0);
            let params: [f32; 8] = [
                head_dim as f32,
                n_head as f32,
                n_kv as f32,
                cache_rows as f32,
                qc as f32,
                qb0 as f32,
                n_kv_blocks as f32,
                base_pos as f32,
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
                    BufferArg::from_raw_parts(kpool_handle.clone(), n_kv * n_kv_blocks * head_dim),
                    BufferArg::from_raw_parts(
                        scores_handle.clone(),
                        q_windows * n_head * n_kv_blocks,
                    ),
                    BufferArg::from_raw_parts(params_handle, 8),
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
    /// `[q_windows * n_head * wpr]` u32 words. `head_dim` must be 256.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_handle: Handle,
        value_handle: Handle,
        gate_handle: Handle,
        attn_out_handle: Handle,
        mask_handle: Handle,
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
                    BufferArg::from_raw_parts(params_handle, 10),
                );
            }
            t0 += tc;
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
            let w1 = 1 * n_head * 8;
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
        let w1 = 1 * n_head * n_kv_blocks;
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
    /// zero neighbours — the centroid probe fires through the pool kernel;
    /// the band term stays empty for this pairing (the hot query is the
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
        let q_h = client.create_from_slice(f32::as_bytes(&q));
        let k_h = client.create_from_slice(f32::as_bytes(&k));
        let n_kv_blocks = cache_rows.div_ceil(SPARSE_KV_BLOCK);
        let kpool_h = client.empty(n_kv * n_kv_blocks * hd * 4);
        let q_windows = cache_rows.div_ceil(SPARSE_Q_BLOCK);
        let scores_h = client.empty(q_windows * n_head * n_kv_blocks * 4);
        unsafe {
            SparsePrefillPoolCubeCL::launch::<ActiveRuntime>(
                &client,
                k_h.clone(),
                kpool_h.clone(),
                hd,
                n_kv,
                cache_rows,
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
            "hot block 3 ({s3}) must dominate its zero neighbours ({s0}, {s1}, {s2})"
        );
        assert!(s3 > 0.1, "centroid probe must be positive on the hot pair");
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
    }
}
