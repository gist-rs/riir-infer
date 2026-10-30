//! Issue 1156 T4 — the modelless regime gate (the elastic extraction).
//!
//! ## Why
//!
//! The envelope scorer (T3-v2) made a FIXED τ legal on every family, but the
//! measured floors are still family-shaped: `needle`/`qa`/`summarization`/
//! `induction` bind at 0.85, `code` at 0.8, `math` at 0.5, `continuation` at
//! 0.6@16k / 0.85@32k. A fixed τ must pay the WORST input class everywhere.
//! The regime gate reads the input's own regime off its boundary hidden
//! states and routes the chunk to the tightest legal posture for THAT class
//! — modellessly (no training): pooled features → frozen centroids →
//! sigmoid margin → per-chunk sparse-vs-dense decision that consumes the
//! T3 ΩMSR manifest's budgets through
//! [`crate::sparse_omega_msr::verify_posture_at_len`] (one mechanism).
//! This is the baseline the trained router (riir-train Plan 476) must beat.
//!
//! ## Policy (pre-registered in plan 625 T4)
//!
//! - **Features**: mean-pool the first [`REGIME_POOL_TOKENS`] and the last
//!   [`REGIME_POOL_TOKENS`] rows of the chunk's residual stream
//!   ([`boundary_pool`]; concat(head, tail), `2·d` dims). Sigmoid margin —
//!   never softmax (the latent-space rule).
//! - **Classification**: cosine to two frozen centroids →
//!   `g = sigmoid(gain · (cos_retr − cos_flow))`; `g ≥ 0.5 + margin` =
//!   retrieval-like, `g ≤ 0.5 − margin` = flow-like, else ambiguous.
//! - **τ proposal**: the decided class's floor (max over its families at
//!   the CURRENT prompt length — the measured floors grow with length, so
//!   the read is [`crate::sparse_omega_msr::conservative_floor_for_len`]);
//!   ambiguous → the max of both class floors. A class with any family that
//!   is missing / unmeasured / floor-less at this length is ILLEGAL (floor
//!   `None`) — the gate routes that decision DENSE (fail-closed), never
//!   guesses.
//! - **Belt**: before overriding anything, every family of the decided
//!   class re-verifies through [`crate::sparse_omega_msr::verify_posture_at_len`]
//!   at the runtime scorer. The floors propose; `verify_posture` disposes.
//!
//! ## Honest scope (v1)
//!
//! - The dynamic range with TWO centroids on the committed v2 manifest is
//!   τ ∈ [0.8, 0.85] at 16K and degenerates to 0.85 at 32K (the measured
//!   32K continuation floor binds). The gate's value is the machinery +
//!   the modelless baseline, not a perf breakthrough.
//! - The decision is per-CHUNK and uniform across heads/layers; the
//!   per-head/per-layer FA-vs-sparse WIRE ships in T5
//!   (`head_mode` words + `RIIR_PREFILL_SPARSE_DENSE_HEADS`) — per-layer
//!   differentiation is the trained router's upgrade path.
//! - The frozen centroids are NOT YET CALIBRATED: the gate is INERT until
//!   `RIIR_PREFILL_REGIME_GATE=1` is set AND both sidecar paths exist
//!   (`RIIR_REGIME_CENTROIDS`, `RIIR_OMSR_MANIFEST`). A missing/failed load
//!   falls back to the plain sparse arm with ONE loud line + a refusal
//!   counter — the operator's env τ posture stands, never a silent change.
//!   [`REGIME_GAIN`] is a named placeholder pending the freeze pass (the
//!   sweep anchors are the calibration corpus; the instrument extension is
//!   the follow-up lane).

// ---------------------------------------------------------------------------
// Vocabulary + consts (ungated — the whole module is CPU-pure except the
// sidecar loader, which rides `cubecl_runtime` for blake3)
// ---------------------------------------------------------------------------

/// Boundary pool width (rows per side). The T4 spec's "first/last 100
/// tokens".
pub const REGIME_POOL_TOKENS: usize = 100;

/// Sigmoid gain over the centroid-margin difference. PLACEHOLDER pending
/// the centroid freeze (the sweep anchors are the calibration corpus) —
/// the machinery tests use synthetic centroids where the margin is extreme,
/// so the gain's exact value gates nothing today.
pub const REGIME_GAIN: f32 = 10.0;

/// The classification margin around 0.5: inside the band the gate reads
/// AMBIGUOUS and pays the conservative (max) floor.
pub const REGIME_MARGIN: f32 = 0.1;

/// Families whose ΩMSR floors bind at retrieval tightness (single-fact /
/// structure-sensitive reads — the T3 failure class).
pub const RETRIEVAL_FAMILIES: [&str; 4] = ["needle", "qa", "summarization", "induction"];

/// Families whose floors are loosest (flow/derivation-shaped reads).
pub const FLOW_FAMILIES: [&str; 3] = ["code", "math", "continuation"];

/// Every family the committed manifest measures — the ambiguous belt runs
/// all of them.
pub const ALL_FAMILIES: [&str; 7] = [
    "needle",
    "qa",
    "summarization",
    "induction",
    "code",
    "math",
    "continuation",
];

/// The gate's regime verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegimeClass {
    /// Retrieval-like: the tight class floor binds.
    Retrieval,
    /// Flow-like: the loose class floor applies.
    Flow,
    /// Inside the margin band: the conservative max floor binds, and the
    /// belt verifies EVERY family.
    Ambiguous,
}

/// The families a decided class must verify through `verify_posture_at_len`.
pub fn class_families(class: RegimeClass) -> &'static [&'static str] {
    match class {
        RegimeClass::Retrieval => &RETRIEVAL_FAMILIES,
        RegimeClass::Flow => &FLOW_FAMILIES,
        RegimeClass::Ambiguous => &ALL_FAMILIES,
    }
}

/// Boundary-pool a `[n_tokens, d]` hidden matrix into the gate's feature
/// vector: concat(mean of the first `pool` rows, mean of the last `pool`
/// rows), `2·d` dims. A short input pools its whole extent twice (head ==
/// tail) — deterministic, and the same math the GPU gather path feeds back
/// (the gather copies exactly these rows, so both paths accumulate in the
/// same order and read bit-identical features). Local decision input —
/// never a synced value.
///
/// Returns zeros for an empty input (the caller refuses before that).
pub fn boundary_pool(hid: &[f32], n_tokens: usize, d: usize, pool: usize) -> Vec<f32> {
    assert!(hid.len() >= n_tokens * d, "boundary_pool: hid too short");
    let mut feats = vec![0.0f32; 2 * d];
    if n_tokens == 0 || d == 0 {
        return feats;
    }
    let head_cnt = pool.min(n_tokens);
    let tail_start = n_tokens.saturating_sub(pool);
    let tail_cnt = n_tokens - tail_start;
    for t in 0..head_cnt {
        let row = &hid[t * d..(t + 1) * d];
        for (i, v) in row.iter().enumerate() {
            feats[i] += v;
        }
    }
    for t in tail_start..n_tokens {
        let row = &hid[t * d..(t + 1) * d];
        for (i, v) in row.iter().enumerate() {
            feats[d + i] += v;
        }
    }
    for v in feats.iter_mut().take(d) {
        *v /= head_cnt as f32;
    }
    for v in feats.iter_mut().skip(d) {
        *v /= tail_cnt as f32;
    }
    feats
}

/// The two class floors a decision consumes, read off a verified manifest at
/// the CURRENT prompt length. `None` = that class is ILLEGAL at this length
/// (a family missing, unmeasured here, or floor-less) — the gate routes
/// dense, never guesses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeFloors {
    pub retrieval: Option<f32>,
    pub flow: Option<f32>,
}

/// Max over a class's families of the conservative per-length floor. Any
/// family missing from the manifest, refusing this length, or carrying no
/// safe floor poisons the WHOLE class (fail-closed).
pub fn class_floor(
    manifest: &crate::sparse_omega_msr::OmegaManifest,
    families: &[&str],
    prompt_len: usize,
) -> Option<f32> {
    let mut acc: Option<f32> = None;
    for fam in families {
        let entry = manifest
            .families
            .iter()
            .find(|f| &f.family == fam)
            .ok_or(()) // missing family → illegal class
            .ok()?;
        let floor = crate::sparse_omega_msr::conservative_floor_for_len(&entry.lengths, prompt_len)
            .ok() // unknown length tag → illegal class
            .flatten()?;
        acc = Some(acc.map_or(floor, |a: f32| a.max(floor)));
    }
    acc
}

/// Both class floors for `prompt_len` (see [`class_floor`]).
pub fn regime_floors(
    manifest: &crate::sparse_omega_msr::OmegaManifest,
    prompt_len: usize,
) -> RegimeFloors {
    RegimeFloors {
        retrieval: class_floor(manifest, &RETRIEVAL_FAMILIES, prompt_len),
        flow: class_floor(manifest, &FLOW_FAMILIES, prompt_len),
    }
}

/// The gate's routing decision for one chunk.
#[derive(Clone, Debug)]
pub struct RegimeDecision {
    pub class: RegimeClass,
    /// false = route this chunk DENSE (the class has no legal τ < 1.0).
    pub engage_sparse: bool,
    /// The proposed selection threshold (legal per the manifest floors;
    /// `1.0` when dense).
    pub tau: f32,
    /// |g − 0.5| — how far the sigmoid read sits from the ambiguity band.
    pub confidence: f32,
    /// v1: always empty (uniform decision). The trained router fills this
    /// through the T5 `head_mode` wire.
    pub dense_heads: Vec<usize>,
}

/// The two frozen regime centroids + the gate's calibration knobs, loaded
/// from a digest-pinned sidecar pair. NEVER derived at runtime — a missing
/// artifact refuses the gate (the A8/A10 freeze law: nothing mints here).
#[derive(Clone, Debug)]
pub struct RegimeCentroids {
    pub pool_tokens: usize,
    /// Feature dim (2 · the model's hidden dim at freeze time).
    pub dim: usize,
    pub gain: f32,
    pub margin: f32,
    /// `dim` floats — the retrieval-class centroid.
    pub retrieval: Vec<f32>,
    /// `dim` floats — the flow-class centroid.
    pub flow: Vec<f32>,
}

impl RegimeCentroids {
    /// The policy: cosine margin → sigmoid → class → floor → decision.
    pub fn decide(&self, feats: &[f32], floors: &RegimeFloors) -> RegimeDecision {
        debug_assert_eq!(feats.len(), self.dim, "regime gate: feature dim drift");
        let cr = cosine(feats, &self.retrieval);
        let cf = cosine(feats, &self.flow);
        let g = sigmoid(self.gain * (cr - cf));
        let class = if g >= 0.5 + self.margin {
            RegimeClass::Retrieval
        } else if g <= 0.5 - self.margin {
            RegimeClass::Flow
        } else {
            RegimeClass::Ambiguous
        };
        let floor = match class {
            RegimeClass::Retrieval => floors.retrieval,
            RegimeClass::Flow => floors.flow,
            RegimeClass::Ambiguous => match (floors.retrieval, floors.flow) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            },
        };
        RegimeDecision {
            class,
            engage_sparse: floor.is_some_and(|f| f < 1.0),
            tau: floor.unwrap_or(1.0),
            confidence: (g - 0.5).abs(),
            dense_heads: Vec::new(),
        }
    }
}

/// Stable sigmoid — never softmax (the latent-space rule). Local decision
/// arithmetic, not a synced value.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Cosine over f64 accumulation (a 2·5120-dim mean; f64 keeps the pooled
/// dot products stable). Zero-norm inputs read 0.0 — the neutral margin,
/// which lands in the ambiguous band by construction.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..a.len().min(b.len()) {
        let x = f64::from(a[i]);
        let y = f64::from(b[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

// ---------------------------------------------------------------------------
// Sidecar pair (gated `cubecl_runtime` — blake3 rides that feature union)
// ---------------------------------------------------------------------------

/// Parse a centroids sidecar body (ungated — testable without blake3).
/// Shape:
/// `{kind: "sparse_prefill_regime_centroids", version: 1, pool_tokens, dim,
///   gain, margin, retrieval: [dim], flow: [dim]}`.
pub fn parse_centroids_body(body: &[u8]) -> Result<RegimeCentroids, String> {
    let body: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("centroids body not JSON: {e}"))?;
    let kind = body
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or("centroids field `kind` missing")?;
    if kind != "sparse_prefill_regime_centroids" {
        return Err(format!(
            "centroids kind `{kind}` is not `sparse_prefill_regime_centroids`"
        ));
    }
    let version = body
        .get("version")
        .and_then(|v| v.as_u64())
        .ok_or("centroids field `version` missing")?;
    if version != 1 {
        return Err(format!(
            "centroids version {version} is not 1 — newer sidecars need a reader bump, \
             never a silent trust"
        ));
    }
    let pool_tokens = body
        .get("pool_tokens")
        .and_then(|v| v.as_u64())
        .ok_or("centroids field `pool_tokens` missing")? as usize;
    if pool_tokens == 0 {
        return Err("centroids pool_tokens must be > 0".into());
    }
    let dim = body
        .get("dim")
        .and_then(|v| v.as_u64())
        .ok_or("centroids field `dim` missing")? as usize;
    if dim == 0 {
        return Err("centroids dim must be > 0".into());
    }
    let gain = body
        .get("gain")
        .and_then(|v| v.as_f64())
        .ok_or("centroids field `gain` missing")? as f32;
    // Strictly positive, NaN included in the refusal (fail-closed).
    if gain.is_nan() || gain <= 0.0 {
        return Err("centroids gain must be > 0".into());
    }
    let margin = body
        .get("margin")
        .and_then(|v| v.as_f64())
        .ok_or("centroids field `margin` missing")? as f32;
    if !(0.0..=0.5).contains(&margin) {
        return Err(format!(
            "centroids margin {margin} outside [0, 0.5] — the ambiguity band must be sane"
        ));
    }
    let read_vec = |name: &str| -> Result<Vec<f32>, String> {
        let arr = body
            .get(name)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("centroids field `{name}` missing"))?;
        if arr.len() != dim {
            return Err(format!(
                "centroids `{name}` has {} entries, dim says {dim}",
                arr.len()
            ));
        }
        arr.iter()
            .map(|v| {
                v.as_f64()
                    .map(|x| x as f32)
                    .ok_or_else(|| format!("centroids `{name}` entry not a number"))
            })
            .collect()
    };
    Ok(RegimeCentroids {
        pool_tokens,
        dim,
        gain,
        margin,
        retrieval: read_vec("retrieval")?,
        flow: read_vec("flow")?,
    })
}

/// Read `<path>.json` + the `<path>.blake3` sidecar, verify the digest, and
/// parse (the [`crate::sparse_omega_msr::load_and_verify_manifest`] shape —
/// one reader mechanism per artifact family, never a divergent copy).
#[cfg(feature = "cubecl_runtime")]
pub fn load_and_verify_centroids(json_path: &std::path::Path) -> Result<RegimeCentroids, String> {
    let body =
        std::fs::read(json_path).map_err(|e| format!("read {}: {e}", json_path.display()))?;
    let sidecar_path = crate::sparse_omega_msr::digest_sidecar_path(json_path);
    let sidecar = std::fs::read_to_string(&sidecar_path)
        .map_err(|e| format!("read {}: {e}", sidecar_path.display()))?;
    let expected = sidecar.trim();
    let actual = crate::sparse_omega_msr::blake3_hex(&body);
    if expected != actual {
        return Err(format!(
            "centroids digest mismatch: sidecar says `{expected}`, body hashes to `{actual}` — \
             the sidecar or its digest was edited; re-run the freeze pass"
        ));
    }
    parse_centroids_body(&body)
}

// ---------------------------------------------------------------------------
// Env arms + counters + the cached gate materials
// ---------------------------------------------------------------------------

#[cfg(feature = "cubecl_runtime")]
static GATE_INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(feature = "cubecl_runtime")]
static GATE_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

static REGIME_DECISIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REGIME_DENSE_CHUNKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REGIME_REFUSALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether the regime gate is active (DEFAULT OFF — inert until the freeze
/// pass lands AND the operator arms it). Env `RIIR_PREFILL_REGIME_GATE` = the
/// exact literals `1|on|true`; junk/unset = OFF. Env read exactly once — the
/// first caller wins the store (the τ protocol).
#[cfg(feature = "cubecl_runtime")]
pub fn regime_gate_armed() -> bool {
    let env_enabled = matches!(
        std::env::var("RIIR_PREFILL_REGIME_GATE")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str(),
        "1" | "on" | "true",
    );
    if GATE_INITIALIZED
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_ok()
    {
        GATE_ARMED.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    GATE_ARMED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Test/demo setter (the `set_prefill_sparse` precedent).
#[cfg(feature = "cubecl_runtime")]
pub fn set_regime_gate_armed(enabled: bool) {
    GATE_INITIALIZED.store(true, std::sync::atomic::Ordering::Relaxed);
    GATE_ARMED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// `(decisions, dense_chunks, refusals)` — the vacuous-guard triple for T6:
/// decisions count the gate's verdicts, dense_chunks the dense routes, and
/// refusals the load/belt failures (a decision whose refusals == decisions
/// routed nothing — the loud-zero read).
pub fn regime_gate_counters() -> (usize, usize, usize) {
    (
        REGIME_DECISIONS.load(std::sync::atomic::Ordering::Relaxed),
        REGIME_DENSE_CHUNKS.load(std::sync::atomic::Ordering::Relaxed),
        REGIME_REFUSALS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Counted by the forward hook on every gate verdict (dense or sparse).
pub fn note_regime_decision() {
    REGIME_DECISIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Counted when the gate routes a chunk dense (class illegal or belt
/// refusal).
pub fn note_regime_dense_chunk() {
    REGIME_DENSE_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Counted on a materials-load failure or a belt refusal.
pub fn note_regime_refusal() {
    REGIME_REFUSALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cubecl_runtime")]
static CENTROIDS_CACHE: std::sync::OnceLock<Result<std::sync::Arc<RegimeCentroids>, String>> =
    std::sync::OnceLock::new();
#[cfg(feature = "cubecl_runtime")]
static MANIFEST_CACHE: std::sync::OnceLock<
    Result<std::sync::Arc<crate::sparse_omega_msr::OmegaManifest>, String>,
> = std::sync::OnceLock::new();

/// Load (once per process) the centroids + ΩMSR manifest from their env
/// paths — `RIIR_REGIME_CENTROIDS` / `RIIR_OMSR_MANIFEST`. A failure is
/// cached, printed ONCE (loud), and re-reported to every caller as `Err`
/// (each caller counts its own refusal — the counters tell the story).
/// Public: the sparse arm's chunk runner is the one caller (the lane seam).
#[cfg(feature = "cubecl_runtime")]
pub fn gate_materials() -> Result<
    (
        std::sync::Arc<RegimeCentroids>,
        std::sync::Arc<crate::sparse_omega_msr::OmegaManifest>,
    ),
    String,
> {
    let centroids = CENTROIDS_CACHE.get_or_init(|| {
        let r = match std::env::var("RIIR_REGIME_CENTROIDS") {
            Ok(path) => load_and_verify_centroids(std::path::Path::new(path.trim())),
            Err(_) => Err("RIIR_REGIME_CENTROIDS is not set — the gate cannot arm \
                           without its frozen centroids"
                .into()),
        };
        if let Err(e) = &r {
            eprintln!(
                "[regime-gate] centroids unavailable: {e} — falling back to the \
                       plain sparse arm (the operator's env τ posture stands)"
            );
        }
        r.map(std::sync::Arc::new)
    });
    let manifest = MANIFEST_CACHE.get_or_init(|| {
        let r = match std::env::var("RIIR_OMSR_MANIFEST") {
            Ok(path) => {
                crate::sparse_omega_msr::load_and_verify_manifest(std::path::Path::new(path.trim()))
            }
            Err(_) => Err("RIIR_OMSR_MANIFEST is not set — the gate cannot arm \
                           without the T3 floor manifest"
                .into()),
        };
        if let Err(e) = &r {
            eprintln!(
                "[regime-gate] OMSR manifest unavailable: {e} — falling back to the \
                       plain sparse arm"
            );
        }
        r.map(std::sync::Arc::new)
    });
    Ok((centroids.clone()?, manifest.clone()?))
}

/// The gate's full chunk decision: load materials → pool the boundary rows →
/// classify → floor the τ → belt-verify. Returns the decision with
/// `engage_sparse` final (the belt may demote a sparse proposal to dense);
/// `Err` = materials unavailable (the caller keeps the plain arm).
///
/// Pure CPU — the caller owns the gather + readback and hands the features
/// in (the GPU side of that lives in the sparse arm module).
pub fn decide_chunk(
    centroids: &RegimeCentroids,
    manifest: &crate::sparse_omega_msr::OmegaManifest,
    feats: &[f32],
    prompt_len: usize,
    runtime_scorer: crate::sparse_omega_msr::OmegaScorer,
) -> Result<RegimeDecision, String> {
    if feats.len() != centroids.dim {
        note_regime_refusal();
        return Err(format!(
            "regime gate: feature dim {} != centroids dim {} — model/centroids drift",
            feats.len(),
            centroids.dim
        ));
    }
    let floors = regime_floors(manifest, prompt_len);
    let mut decision = centroids.decide(feats, &floors);
    note_regime_decision();
    if decision.engage_sparse {
        // Belt: every family of the decided class must verify at this
        // length under the runtime scorer. The floors propose;
        // `verify_posture` disposes (one mechanism). The first refusal
        // demotes the whole chunk to dense (fail-closed) — the counter is
        // the observable verdict, the message names the family.
        for fam in class_families(decision.class) {
            if let Err(e) = crate::sparse_omega_msr::verify_posture_at_len(
                manifest,
                fam,
                prompt_len,
                decision.tau,
                runtime_scorer,
            ) {
                // One loud line for the FIRST belt refusal; every refusal
                // after it rides the counter (per-chunk printing would spam).
                static BELT_REFUSAL_ANNOUNCED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !BELT_REFUSAL_ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[regime-gate] belt refusal on `{fam}` at {prompt_len} tokens: {e} — \
                         routing the chunk dense (further refusals ride the counter)"
                    );
                }
                note_regime_refusal();
                decision.engage_sparse = false;
                decision.tau = 1.0;
                break;
            }
        }
    }
    Ok(decision)
}

// ---------------------------------------------------------------------------
// Self-tests — pure CPU, ungated (the file-pair loader test rides
// `cubecl_runtime` like the manifest pair's)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- boundary_pool ------------------------------------------------------

    #[test]
    fn boundary_pool_exact_means() {
        // rows 0..2 and 3..5 of a 5-row, 2-dim matrix, pool 2.
        // hid rows: [0,0], [1,1], [2,2], [3,3], [4,4]
        let hid: Vec<f32> = (0..5).flat_map(|i| [i as f32, i as f32]).collect();
        let feats = boundary_pool(&hid, 5, 2, 2);
        assert_eq!(feats.len(), 4);
        // head = mean(rows 0,1) = 0.5; tail = mean(rows 3,4) = 3.5.
        assert_eq!(feats[0], 0.5);
        assert_eq!(feats[1], 0.5);
        assert_eq!(feats[2], 3.5);
        assert_eq!(feats[3], 3.5);
    }

    #[test]
    fn boundary_pool_short_input_pools_twice() {
        // 3 rows, pool 100 → head == tail == the whole extent, mean 1.0.
        let hid: Vec<f32> = (0..3).flat_map(|i| [i as f32; 2]).collect();
        let feats = boundary_pool(&hid, 3, 2, 100);
        for v in &feats {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn boundary_pool_gathered_rows_are_bit_identical_to_direct() {
        // The GPU path hands back [head rows; tail rows] — pooling that
        // buffer with n_tokens = 2·pool must read the SAME bits as pooling
        // the original (this is what makes the gather kernel's output
        // contract checkable).
        let hid: Vec<f32> = (0..97)
            .flat_map(|i| [(i as f32) * 0.5, -(i as f32)])
            .collect();
        let pool = 100.min(97);
        let direct = boundary_pool(&hid, 97, 2, pool);
        let mut gathered = Vec::with_capacity(2 * pool * 2);
        gathered.extend_from_slice(&hid[0..pool * 2]);
        let tail_start = 97 - pool;
        gathered.extend_from_slice(&hid[tail_start * 2..97 * 2]);
        let from_gather = boundary_pool(&gathered, 2 * pool, 2, pool);
        assert_eq!(direct.len(), from_gather.len());
        for (a, b) in direct.iter().zip(from_gather.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // -- sigmoid / cosine ----------------------------------------------------

    #[test]
    fn sigmoid_is_monotone_and_bounded() {
        assert!(sigmoid(-4.0) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(4.0));
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
        // The stable halves agree at the seam.
        assert!((sigmoid(0.5) - 1.0 / (1.0 + (-0.5f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm_is_neutral() {
        assert_eq!(cosine(&[0.0; 4], &[1.0, 2.0, 3.0, 4.0]), 0.0);
        assert_eq!(cosine(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    // -- decide (synthetic centroids, the machinery policy both directions) --

    fn synthetic_centroids() -> RegimeCentroids {
        RegimeCentroids {
            pool_tokens: REGIME_POOL_TOKENS,
            dim: 4,
            gain: REGIME_GAIN,
            margin: REGIME_MARGIN,
            retrieval: vec![1.0, 0.0, 0.0, 0.0],
            flow: vec![0.0, 1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn decide_retrieval_aligned_routes_sparse_at_the_class_floor() {
        let c = synthetic_centroids();
        let feats = vec![2.0, 0.0, 0.5, 0.0]; // cosine 1.0 with retrieval
        let floors = RegimeFloors {
            retrieval: Some(0.85),
            flow: Some(0.8),
        };
        let d = c.decide(&feats, &floors);
        assert_eq!(d.class, RegimeClass::Retrieval);
        assert!(d.engage_sparse);
        assert_eq!(d.tau, 0.85);
        assert!(d.confidence > 0.4);
        assert!(d.dense_heads.is_empty(), "v1 is uniform across heads");
    }

    #[test]
    fn decide_flow_aligned_takes_the_loose_floor() {
        let c = synthetic_centroids();
        let feats = vec![0.0, 2.0, 0.5, 0.0]; // cosine 1.0 with flow
        let floors = RegimeFloors {
            retrieval: Some(0.85),
            flow: Some(0.8),
        };
        let d = c.decide(&feats, &floors);
        assert_eq!(d.class, RegimeClass::Flow);
        assert!(d.engage_sparse);
        assert_eq!(d.tau, 0.8);
    }

    #[test]
    fn decide_orthogonal_is_ambiguous_and_conservative() {
        let c = synthetic_centroids();
        let feats = vec![0.0, 0.0, 3.0, 1.0]; // cosine 0 with both
        let floors = RegimeFloors {
            retrieval: Some(0.85),
            flow: Some(0.8),
        };
        let d = c.decide(&feats, &floors);
        assert_eq!(d.class, RegimeClass::Ambiguous);
        assert_eq!(d.tau, 0.85, "ambiguous pays the max of both class floors");
    }

    #[test]
    fn decide_illegal_class_routes_dense() {
        let c = synthetic_centroids();
        let feats = vec![2.0, 0.0, 0.5, 0.0];
        // The retrieval class is illegal here (an unmeasured family in it).
        let floors = RegimeFloors {
            retrieval: None,
            flow: Some(0.8),
        };
        let d = c.decide(&feats, &floors);
        assert_eq!(d.class, RegimeClass::Retrieval);
        assert!(!d.engage_sparse, "an illegal class must route DENSE");
        assert_eq!(d.tau, 1.0);
        // Ambiguous with one illegal class also goes dense (never the
        // surviving class's floor — the ambiguity is unresolved).
        let feats_amb = vec![0.0, 0.0, 3.0, 1.0];
        let d = c.decide(&feats_amb, &floors);
        assert_eq!(d.class, RegimeClass::Ambiguous);
        assert!(!d.engage_sparse);
    }

    // -- regime_floors over the committed v2 fixture --------------------------

    fn v2_manifest() -> crate::sparse_omega_msr::OmegaManifest {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sparse_omega_msr_manifest_v2.json"
        );
        let body = std::fs::read(path)
            .unwrap_or_else(|e| panic!("committed v2 fixture unreadable at {path}: {e}"));
        crate::sparse_omega_msr::parse_and_verify_body(&body)
            .expect("the committed v2 manifest must parse at every posture")
    }

    #[test]
    fn floors_from_the_v2_manifest_at_16k() {
        let m = v2_manifest();
        let f = regime_floors(&m, 16_384);
        assert_eq!(f.retrieval, Some(0.85));
        assert_eq!(f.flow, Some(0.8));
    }

    #[test]
    fn floors_from_the_v2_manifest_at_32k() {
        let m = v2_manifest();
        let f = regime_floors(&m, 32_768);
        // Retrieval families measured only at 16k → past-every-bucket →
        // tightest = 0.85. Flow: continuation's own 32k row (0.85) binds.
        assert_eq!(f.retrieval, Some(0.85));
        assert_eq!(f.flow, Some(0.85));
    }

    #[test]
    fn ambiguous_belt_verifies_every_family_and_passes_at_the_max() {
        let m = v2_manifest();
        let c = synthetic_centroids();
        let feats = vec![0.0, 0.0, 3.0, 1.0];
        let d = decide_chunk(
            &c,
            &m,
            &feats,
            16_384,
            crate::sparse_omega_msr::OmegaScorer::Envelope,
        )
        .expect("belt");
        assert_eq!(d.class, RegimeClass::Ambiguous);
        assert!(d.engage_sparse);
        assert_eq!(
            d.tau, 0.85,
            "the belt must pass at max(retrieval, flow) = 0.85"
        );
        // The counter moved (the vacuous-guard read; decide_chunk counts
        // its own verdicts).
        assert!(regime_gate_counters().0 >= 1);
    }

    #[test]
    fn decide_chunk_dim_drift_refuses() {
        let m = v2_manifest();
        let c = synthetic_centroids();
        let err = decide_chunk(
            &c,
            &m,
            &[0.0; 3],
            16_384,
            crate::sparse_omega_msr::OmegaScorer::Envelope,
        )
        .expect_err("dim drift must refuse");
        assert!(err.contains("drift"), "got: {err}");
        // The refusal is counted (the loud-zero read).
        assert!(regime_gate_counters().2 >= 1);
    }

    // -- sidecar body parser ---------------------------------------------------

    fn centroids_body(dim: usize) -> String {
        let retr: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.25).collect();
        let flow: Vec<f64> = (0..dim).map(|i| -(i as f64) * 0.25).collect();
        format!(
            r#"{{"kind": "sparse_prefill_regime_centroids", "version": 1,
                "pool_tokens": 100, "dim": {dim}, "gain": 10.0, "margin": 0.1,
                "retrieval": {retr:?}, "flow": {flow:?}}}"#,
            retr = retr,
            flow = flow
        )
    }

    #[test]
    fn centroids_body_roundtrips() {
        let c = parse_centroids_body(centroids_body(8).as_bytes()).expect("parse");
        assert_eq!(c.dim, 8);
        assert_eq!(c.pool_tokens, 100);
        assert_eq!(c.gain, 10.0);
        assert_eq!(c.margin, 0.1);
        assert_eq!(c.retrieval.len(), 8);
        assert_eq!(c.flow[3], -0.75);
    }

    #[test]
    fn centroids_body_refuses_drift() {
        // Wrong kind.
        let bad = centroids_body(4).replace("sparse_prefill_regime_centroids", "other");
        assert!(parse_centroids_body(bad.as_bytes()).is_err());
        // Newer version refuses (the manifest-reader rule).
        let bad = centroids_body(4).replace("\"version\": 1", "\"version\": 2");
        assert!(parse_centroids_body(bad.as_bytes()).is_err());
        // Dim mismatch between the arrays and the declared dim.
        let bad = centroids_body(4).replace("\"dim\": 4", "\"dim\": 5");
        assert!(parse_centroids_body(bad.as_bytes()).is_err());
        // A non-numeric entry.
        let bad = centroids_body(4).replace("0.25", "\"x\"");
        assert!(parse_centroids_body(bad.as_bytes()).is_err());
        // Margin outside the sane band.
        let bad = centroids_body(4).replace("\"margin\": 0.1", "\"margin\": 0.9");
        assert!(parse_centroids_body(bad.as_bytes()).is_err());
    }
}
