//! Issue 1156 T3 / Plan 625 — the per-family ΩMSR manifest + knee calculator.
//!
//! The block-sparse prefill arm (Issue 1156 T2, Plan 624) is a lossy
//! surface: τ < 1.0 drops KV blocks, so promotion is bound by **per-family
//! conditional retention** floors, never an aggregate read (the
//! Issue-750-T3 lossy law; the bench-889 lesson — static selection lost the
//! filler-continuation axis while looking flat in aggregate). This module
//! is the ONE mechanism every T4/T6 gate run consumes:
//!
//! - [`omega_msr`] — the knee calculator (pure, CPU, ungated; self-tested
//!   in both directions).
//! - [`parse_and_verify_body`] — manifest parse + re-derivation (ungated;
//!   the reader recomputes retention and ΩMSR from the evidence rows, so a
//!   manifest cannot assert floors its own rows contradict).
//! - [`blake3_hex`] / [`blake3_file_hex`] / [`manifest_body_and_digest`] /
//!   [`load_and_verify_manifest`] — the digest-pinned file pair
//!   (`<name>.json` + `<name>.blake3` sidecar; the laya/Q8 sidecar
//!   convention), gated `cubecl_runtime` — the arm's own posture.
//! - [`verify_posture`] — the call every subsequent gate run MUST route its
//!   posture check through (one mechanism, no re-derivation).
//!
//! ## ΩMSR (pre-registered, Plan 625)
//!
//! **ΩMSR(family) = the bottom edge of the top-contiguous passing run of
//! the ascending τ grid** — the LOWEST τ whose row passes AND every higher
//! grid row also passes. The safe operating region is `[ΩMSR, 1.0]`; the
//! sweep runs top-down and ΩMSR is the last (lowest) t that still passed —
//! the issue's literal wording under the sweep's direction. Higher τ keeps
//! more mass, so passes concentrate at the top; a lone pass BELOW a failure
//! never lifts the floor (non-monotone noise — a near-tie lottery, not a
//! floor). The top row failing → `None` — no proven-safe posture, loud,
//! never a silent fallback.
//!
//! A row passes iff BOTH:
//! - `retention ≥ 1 − ε` — pooled argmax agreement over the final-prompt
//!   probe + the teacher-forced continuation probes (ε =
//!   [`OMEGA_EPS`]); AND
//! - `ppl_excess ≤ [`OMEGA_PPL_BAND`]` where measured (Δppl =
//!   ppl_sparse − ppl_dense over the continuation decode — the bench-889
//!   axis; `None` = the family carries no ppl axis, which never happens in
//!   the T3 sweep: every family carries a continuation).
//!
//! τ = 1.0 is the identity anchor (bit-identical to m32 — the T2 G3
//! contract), not a floor candidate: [`verify_posture`] accepts it
//! unconditionally.

// ---------------------------------------------------------------------------
// Constants (code-owned; the manifest records them and the reader refuses
// drift — a sweep under a different protocol must land a new manifest
// version, not silently reuse the floors)
// ---------------------------------------------------------------------------

/// The ascending τ grid every sweep and manifest is pinned to.
pub const OMEGA_GRID: [f32; 7] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95];

/// Retention floor slack: a row passes at `retention ≥ 1 − ε`.
pub const OMEGA_EPS: f32 = 0.05;

/// Continuation-ppl ceiling: `ppl_sparse − ppl_dense ≤ +0.25 ppl` (the
/// bench-889 continuation axis; absolute-ppl units).
pub const OMEGA_PPL_BAND: f32 = 0.25;

/// Manifest protocol version (grid/ε/band/row-shape/SCORER changes bump
/// this and invalidate older manifests — the reader refuses other versions).
/// Version 2 (T3-v2): the scorer became part of the protocol — a v2 manifest
/// REQUIRES the `scorer` field; the committed v1 manifests are the MEAN
/// scorer's floors and stay verifiable (scorer implicit `mean`).
pub const OMEGA_MANIFEST_VERSION: u32 = 2;

/// The block scorer a manifest's floors were measured under (protocol v2).
/// Version-1 manifests are implicitly [`OmegaScorer::Mean`]. This is THE
/// canonical enum (one source of truth): the sparse arm re-exports it as
/// `SparseScorer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OmegaScorer {
    Mean,
    Envelope,
}

impl OmegaScorer {
    /// The kernel-mode encoding packed into the params buffer.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::Mean => 0,
            Self::Envelope => 1,
        }
    }

    /// The manifest / env spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Envelope => "envelope",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mean" => Some(Self::Mean),
            "envelope" => Some(Self::Envelope),
            _ => None,
        }
    }

    /// The inverse of [`OmegaScorer::as_u32`] (any non-0 reads as
    /// `Envelope` — the caller's bits come from `as_u32`, never the wire).
    pub const fn from_bits(bits: u32) -> Self {
        if bits == Self::Mean.as_u32() {
            Self::Mean
        } else {
            Self::Envelope
        }
    }
}

/// Float compare slack for grid/ε/band equality (all values are
/// shortest-roundtrip exact in JSON, but the slack keeps the read honest
/// against hand-edits that reformat).
const FLOAT_SLACK: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Knee calculator (pure, ungated, self-tested both directions)
// ---------------------------------------------------------------------------

/// One grid row's gate inputs, as the knee calc consumes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KneeRow {
    pub tau: f32,
    pub retention: f32,
    /// `Some(Δppl)` where the family carries the ppl axis; `None` = no ppl
    /// gate for this row.
    pub ppl_excess: Option<f32>,
}

fn row_passes(row: &KneeRow) -> bool {
    row.retention >= 1.0 - OMEGA_EPS && row.ppl_excess.is_none_or(|excess| excess <= OMEGA_PPL_BAND)
}

/// The ΩMSR knee over an ascending grid (see the module doc for the rule).
///
/// Refuses loud on a non-strictly-ascending grid or rows misaligned with
/// it — a silent misalignment would floor the wrong posture.
pub fn omega_msr(grid: &[f32], rows: &[KneeRow]) -> Option<f32> {
    for pair in grid.windows(2) {
        assert!(
            pair[0] < pair[1],
            "OMSR grid must be strictly ascending, got {} then {}",
            pair[0],
            pair[1]
        );
    }
    assert_eq!(
        grid.len(),
        rows.len(),
        "OMSR rows must align 1:1 with the grid"
    );
    for (tau, row) in grid.iter().zip(rows.iter()) {
        assert!(
            (f64::from(*tau) - f64::from(row.tau)).abs() <= FLOAT_SLACK,
            "OMSR row tau {} misaligned with grid tau {}",
            row.tau,
            tau
        );
    }
    // The top-contiguous passing run: scan from the top, keep re-flooring
    // while rows pass, stop at the first failure. All rows pass → grid[0]
    // (the whole grid is safe); the top row fails → None.
    let mut floor = None;
    for (tau, row) in grid.iter().zip(rows.iter()).rev() {
        if row_passes(row) {
            floor = Some(*tau);
        } else {
            break;
        }
    }
    floor
}

/// The posture verdict one gate run needs: is `tau` safe for `family` when
/// the runtime runs `scorer`?
///
/// τ ≥ 1.0 is the identity posture — always safe (the T2 G3 bit-identity
/// contract; scores are never consulted). Below that, the manifest's scorer
/// must MATCH the runtime's (a floor never blesses the other scorer), the
/// family must be present, and its binding ΩMSR (the min across measured
/// lengths) must be ≤ τ; a family with no floor refuses every τ < 1.0 loud.
pub fn verify_posture(
    manifest: &OmegaManifest,
    family: &str,
    tau: f32,
    scorer: OmegaScorer,
) -> Result<(), String> {
    if manifest.scorer != scorer {
        return Err(format!(
            "OMSR posture check: manifest floors are for scorer `{}`, \
             runtime runs `{}` — a floor never blesses the other scorer",
            manifest.scorer.as_str(),
            scorer.as_str()
        ));
    }
    if tau >= 1.0 {
        return Ok(());
    }
    let entry = manifest
        .families
        .iter()
        .find(|f| f.family == family)
        .ok_or_else(|| format!("OMSR posture check: family `{family}` is not in the manifest"))?;
    let floor = entry.omega_msr.ok_or_else(|| {
        format!(
            "OMSR posture check: family `{family}` has NO safe τ floor — \
             the sparse arm must not serve τ < 1.0 on it"
        )
    })?;
    if f64::from(tau) + FLOAT_SLACK < f64::from(floor) {
        Err(format!(
            "OMSR posture check: τ = {tau} is below the `{family}` floor {floor} — refused"
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Manifest types (parsed + verified)
// ---------------------------------------------------------------------------

/// One evidence row as parsed from the manifest.
#[derive(Clone, Debug)]
pub struct OmegaRowRecord {
    pub tau: f32,
    pub len: String,
    pub probes: usize,
    pub agreements: usize,
    pub retention: f64,
    pub final_flips: usize,
    pub mean_abs_dnll: f64,
    pub ppl_sparse: Option<f64>,
    pub ppl_dense: Option<f64>,
    pub ppl_excess: Option<f64>,
    pub sparse_routed: bool,
}

/// One family: its per-length floors + the binding (min-across-lengths)
/// floor [`verify_posture`] consumes.
#[derive(Clone, Debug)]
pub struct OmegaFamilyVerified {
    pub family: String,
    /// The binding floor: min across measured lengths. `None` = no length
    /// yielded a passing posture — every τ < 1.0 is refused on this family.
    pub omega_msr: Option<f32>,
    /// `(len_tag, per-length ΩMSR)` in manifest order.
    pub lengths: Vec<(String, Option<f32>)>,
}

/// A parsed, fully re-derived manifest (what [`verify_posture`] consumes).
#[derive(Clone, Debug)]
pub struct OmegaManifest {
    pub model_blake3: String,
    pub corpus_blake3: String,
    /// The scorer these floors were measured under (v1 manifests: `Mean`).
    pub scorer: OmegaScorer,
    pub families: Vec<OmegaFamilyVerified>,
}

// ---------------------------------------------------------------------------
// JSON helpers (ungated — serde_json is an unconditional dep)
// ---------------------------------------------------------------------------

fn body_f64(body: &serde_json::Value, key: &str) -> Result<f64, String> {
    body.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("manifest field `{key}` missing or not a number"))
}

fn body_u32(body: &serde_json::Value, key: &str) -> Result<u32, String> {
    body.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| u32::try_from(v).map_err(|_| format!("manifest field `{key}` out of u32 range")))
        .transpose()?
        .ok_or_else(|| format!("manifest field `{key}` missing or not a number"))
}

fn body_string(body: &serde_json::Value, key: &str) -> Result<String, String> {
    body.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| format!("manifest field `{key}` missing or not a string"))
}

fn body_opt_f64(value: &serde_json::Value, key: &str) -> Result<Option<f64>, String> {
    match value.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("manifest row field `{key}` not a number")),
    }
}

fn row_from_json(row: &serde_json::Value, grid_tau: f32) -> Result<OmegaRowRecord, String> {
    let get_u = |key: &str| -> Result<usize, String> {
        row.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("manifest row field `{key}` missing or not a number"))
    };
    let get_f = |key: &str| -> Result<f64, String> {
        row.get(key)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("manifest row field `{key}` missing or not a number"))
    };
    let tau = f64::from(grid_tau);
    let stored_tau = get_f("tau")?;
    if (stored_tau - tau).abs() > FLOAT_SLACK {
        return Err(format!(
            "manifest row tau {stored_tau} misaligned with grid tau {grid_tau}"
        ));
    }
    let probes = get_u("probes")?;
    let agreements = get_u("agreements")?;
    if agreements > probes {
        return Err(format!(
            "manifest row at τ {grid_tau}: agreements {agreements} > probes {probes}"
        ));
    }
    let retention = get_f("retention")?;
    let derived = if probes == 0 {
        0.0
    } else {
        agreements as f64 / probes as f64
    };
    if (retention - derived).abs() > 1e-9 {
        return Err(format!(
            "manifest row at τ {grid_tau}: stored retention {retention} != \
             agreements/probes = {derived} — the manifest may not assert floors \
             its own rows contradict"
        ));
    }
    Ok(OmegaRowRecord {
        tau: grid_tau,
        len: body_string(row, "len")?,
        probes,
        agreements,
        retention,
        final_flips: get_u("final_flips")?,
        mean_abs_dnll: get_f("mean_abs_dnll")?,
        ppl_sparse: body_opt_f64(row, "ppl_sparse")?,
        ppl_dense: body_opt_f64(row, "ppl_dense")?,
        ppl_excess: body_opt_f64(row, "ppl_excess")?,
        sparse_routed: row
            .get("sparse_routed")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| "manifest row field `sparse_routed` missing".to_string())?,
    })
}

fn knee_rows(records: &[OmegaRowRecord]) -> Vec<KneeRow> {
    records
        .iter()
        .map(|r| KneeRow {
            tau: r.tau,
            retention: r.retention as f32,
            ppl_excess: r.ppl_excess.map(|x| x as f32),
        })
        .collect()
}

fn verify_length_rows(
    family: &str,
    len: &str,
    rows_json: &serde_json::Value,
    grid: &[f32],
) -> Result<(Vec<OmegaRowRecord>, Option<f32>), String> {
    let rows_arr = rows_json
        .get("rows")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("family `{family}` length `{len}`: `rows` missing"))?;
    if rows_arr.len() != grid.len() {
        return Err(format!(
            "family `{family}` length `{len}`: {} rows for a {}-point grid",
            rows_arr.len(),
            grid.len()
        ));
    }
    let mut records = Vec::with_capacity(rows_arr.len());
    for (row_json, tau) in rows_arr.iter().zip(grid.iter()) {
        records.push(row_from_json(row_json, *tau)?);
    }
    let recomputed = omega_msr(grid, &knee_rows(&records));
    let stored = rows_json.get("omega_msr").map_or(Ok(None), |v| match v {
        serde_json::Value::Null => Ok(None),
        other => other
            .as_f64()
            .map(|x| Some(x as f32))
            .ok_or_else(|| format!("family `{family}` length `{len}`: `omega_msr` not a number")),
    })?;
    match (stored, recomputed) {
        (Some(a), Some(b)) => {
            if (f64::from(a) - f64::from(b)).abs() > FLOAT_SLACK {
                return Err(format!(
                    "family `{family}` length `{len}`: stored ΩMSR {a} != \
                     re-derived {b} — floors are re-derived from rows, never read"
                ));
            }
        }
        (None, None) => {}
        (stored_val, re_val) => {
            return Err(format!(
                "family `{family}` length `{len}`: stored ΩMSR {stored_val:?} vs \
                 re-derived {re_val:?} — the manifest may not assert floors its \
                 own rows contradict"
            ));
        }
    }
    Ok((records, recomputed))
}

/// Parse + verify a manifest body against the code-owned protocol.
///
/// The body is re-derived, not trusted: retentions are recomputed from
/// probes/agreements, per-length ΩMSRs from the knee calc, family floors as
/// the min across lengths, and the protocol constants must match this
/// build's ([`OMEGA_GRID`], [`OMEGA_EPS`], [`OMEGA_PPL_BAND`]). Versions ≤
/// [`OMEGA_MANIFEST_VERSION`] parse (v1 = the MEAN scorer implicitly); a
/// NEWER version refuses — floors this build cannot re-derive are never
/// trusted. The DIGEST check lives wholly in the gated file reader
/// ([`load_and_verify_manifest`]) — this function is pure content
/// verification and never sees a filesystem.
pub fn parse_and_verify_body(body: &[u8]) -> Result<OmegaManifest, String> {
    let body: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("manifest body not JSON: {e}"))?;
    let version = body_u32(&body, "manifest_version")?;
    if version > OMEGA_MANIFEST_VERSION {
        return Err(format!(
            "manifest version {version} is NEWER than this build's protocol \
             {OMEGA_MANIFEST_VERSION} — upgrade the reader, never trust \
             floors this build cannot re-derive"
        ));
    }
    // Protocol v2 made the scorer part of the manifest (T3-v2); v1 bodies
    // are implicitly the MEAN scorer's floors. A v1 body that CARRIES a
    // scorer field must agree (mean) — anything else is drift.
    let scorer = if version == 1 {
        match body.get("scorer") {
            None | Some(serde_json::Value::Null) => OmegaScorer::Mean,
            Some(serde_json::Value::String(s)) => {
                let parsed = OmegaScorer::parse(s)
                    .ok_or_else(|| format!("manifest scorer `{s}` is not mean|envelope"))?;
                if parsed != OmegaScorer::Mean {
                    return Err(format!(
                        "version-1 manifest carries scorer `{s}` — v1 floors are \
                         implicitly mean; a newer scorer needs manifest_version 2"
                    ));
                }
                parsed
            }
            Some(_) => return Err("manifest field `scorer` not a string".to_string()),
        }
    } else {
        let s = body_string(&body, "scorer")?;
        OmegaScorer::parse(&s)
            .ok_or_else(|| format!("manifest scorer `{s}` is not mean|envelope"))?
    };
    let kind = body_string(&body, "kind")?;
    if kind != "sparse_prefill_omega_msr" {
        return Err(format!(
            "manifest kind `{kind}` is not `sparse_prefill_omega_msr`"
        ));
    }
    let grid_json = body
        .get("grid")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "manifest field `grid` missing".to_string())?;
    let mut grid = Vec::with_capacity(grid_json.len());
    for v in grid_json {
        grid.push(
            v.as_f64()
                .ok_or_else(|| "manifest grid entry not a number".to_string())? as f32,
        );
    }
    if grid.len() != OMEGA_GRID.len() {
        return Err(format!(
            "manifest grid has {} points, this build's protocol pins {}",
            grid.len(),
            OMEGA_GRID.len()
        ));
    }
    for (got, want) in grid.iter().zip(OMEGA_GRID.iter()) {
        if (f64::from(*got) - f64::from(*want)).abs() > FLOAT_SLACK {
            return Err(format!(
                "manifest grid entry {got} != protocol grid {want} — protocol drift refused"
            ));
        }
    }
    let epsilon = body_f64(&body, "epsilon")?;
    if (epsilon - f64::from(OMEGA_EPS)).abs() > FLOAT_SLACK {
        return Err(format!(
            "manifest epsilon {epsilon} != protocol {} — drift refused",
            OMEGA_EPS
        ));
    }
    let band = body_f64(&body, "ppl_band")?;
    if (band - f64::from(OMEGA_PPL_BAND)).abs() > FLOAT_SLACK {
        return Err(format!(
            "manifest ppl_band {band} != protocol {} — drift refused",
            OMEGA_PPL_BAND
        ));
    }
    let families_json = body
        .get("families")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "manifest field `families` missing".to_string())?;
    if families_json.is_empty() {
        return Err("manifest carries no families — an empty floor set is not a manifest".into());
    }
    let mut families = Vec::with_capacity(families_json.len());
    for fam_json in families_json {
        let name = body_string(fam_json, "family")?;
        let lengths_json = fam_json
            .get("lengths")
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("family `{name}`: `lengths` missing"))?;
        if lengths_json.is_empty() {
            return Err(format!("family `{name}`: no lengths measured"));
        }
        let mut lengths = Vec::with_capacity(lengths_json.len());
        let mut binding: Option<f32> = None;
        for (len, rows_json) in lengths_json {
            let (records, floor) = verify_length_rows(&name, len, rows_json, &grid)?;
            lengths.push((len.clone(), floor));
            binding = match (binding, floor) {
                (None, f) => f,
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) => Some(a),
            };
            let _ = records;
        }
        let stored_binding = fam_json.get("omega_msr").map_or(Ok(None), |v| match v {
            serde_json::Value::Null => Ok(None),
            other => other
                .as_f64()
                .map(|x| Some(x as f32))
                .ok_or_else(|| format!("family `{name}`: `omega_msr` not a number")),
        })?;
        if stored_binding != binding {
            return Err(format!(
                "family `{name}`: stored binding ΩMSR {stored_binding:?} != re-derived \
                 min-across-lengths {binding:?}"
            ));
        }
        families.push(OmegaFamilyVerified {
            family: name,
            omega_msr: binding,
            lengths,
        });
    }
    Ok(OmegaManifest {
        model_blake3: body_string(&body, "model_blake3")?,
        corpus_blake3: body_string(&body, "corpus_blake3")?,
        scorer,
        families,
    })
}

// ---------------------------------------------------------------------------
// Digest-pinned file pair (gated `cubecl_runtime` — blake3 rides that
// feature union; the arm's own posture)
// ---------------------------------------------------------------------------

/// The sidecar path convention: `<name>.json` pairs with `<name>.blake3`
/// beside it (the laya/Q8 sidecar convention).
#[cfg(feature = "cubecl_runtime")]
pub fn digest_sidecar_path(json_path: &std::path::Path) -> std::path::PathBuf {
    let mut sidecar = json_path.as_os_str().to_owned();
    sidecar.push(".blake3");
    std::path::PathBuf::from(sidecar)
}

#[cfg(feature = "cubecl_runtime")]
pub fn blake3_hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

/// Streaming file digest (the 5 GB league model digests in seconds).
#[cfg(feature = "cubecl_runtime")]
pub fn blake3_file_hex(path: &std::path::Path) -> Result<String, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Pretty body + its BLAKE3 hex — the writer half of the pair.
#[cfg(feature = "cubecl_runtime")]
pub fn manifest_body_and_digest(manifest: &serde_json::Value) -> Result<(String, String), String> {
    let body = serde_json::to_string_pretty(manifest).map_err(|e| format!("serialize: {e}"))?;
    let digest = blake3_hex(body.as_bytes());
    Ok((body, digest))
}

/// Read `<path>.json` + the `<path>.blake3` sidecar, verify the digest, and
/// re-derive every floor ([`parse_and_verify_body`]).
#[cfg(feature = "cubecl_runtime")]
pub fn load_and_verify_manifest(json_path: &std::path::Path) -> Result<OmegaManifest, String> {
    let body =
        std::fs::read(json_path).map_err(|e| format!("read {}: {e}", json_path.display()))?;
    let sidecar_path = digest_sidecar_path(json_path);
    let sidecar = std::fs::read_to_string(&sidecar_path)
        .map_err(|e| format!("read {}: {e}", sidecar_path.display()))?;
    let expected = sidecar.trim();
    let actual = blake3_hex(&body);
    if expected != actual {
        return Err(format!(
            "manifest digest mismatch: sidecar says `{expected}`, body hashes to `{actual}` — \
             the manifest or its sidecar was edited; re-run the sweep"
        ));
    }
    parse_and_verify_body(&body)
}

// ---------------------------------------------------------------------------
// Self-tests — the planted-knee instrument, both directions + boundaries.
// The knee/parse tests are ungated (run at every lib-test posture); the
// file-pair tests need `cubecl_runtime` (blake3).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const GRID: [f32; 5] = [0.5, 0.7, 0.85, 0.9, 0.95];

    fn knee(retentions: [f32; 5]) -> Vec<KneeRow> {
        GRID.iter()
            .zip(retentions)
            .map(|(tau, retention)| KneeRow {
                tau: *tau,
                retention,
                ppl_excess: None,
            })
            .collect()
    }

    #[test]
    fn planted_knee_is_found() {
        // Retention degrades as τ drops (less mass kept); the run holds
        // until the bottom falls out at 0.5 → the floor is 0.7.
        let rows = knee([0.94, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(omega_msr(&GRID, &rows), Some(0.7));
    }

    #[test]
    fn all_pass_floors_at_grid_bottom() {
        // Every grid point proven safe → the smallest safe τ is grid[0].
        let rows = knee([1.0; 5]);
        assert_eq!(omega_msr(&GRID, &rows), Some(0.5));
    }

    #[test]
    fn never_pass_returns_none() {
        let rows = knee([0.5; 5]);
        assert_eq!(omega_msr(&GRID, &rows), None);
    }

    #[test]
    fn top_row_failure_returns_none() {
        // The literal "largest passing τ" would read 0.9 here; the strict
        // top-contiguous rule refuses — a floor whose neighbor above failed
        // is a near-tie lottery, not a floor.
        let rows = knee([1.0, 1.0, 0.99, 0.96, 0.93]);
        assert_eq!(omega_msr(&GRID, &rows), None);
    }

    #[test]
    fn epsilon_boundary_exactly_passes() {
        let floor = 1.0 - OMEGA_EPS;
        let rows = knee([floor; 5]);
        assert_eq!(omega_msr(&GRID, &rows), Some(0.5));
    }

    #[test]
    fn epsilon_just_below_fails() {
        let rows = knee([1.0 - OMEGA_EPS - 0.001; 5]);
        assert_eq!(omega_msr(&GRID, &rows), None);
    }

    #[test]
    fn non_monotone_curve_floors_the_contiguous_tail() {
        // Pass run [0.9, 0.95], a dip at 0.85, a lone pass at 0.8: the floor
        // is the tail's bottom edge 0.9 — the lone lower pass never lifts it.
        let rows = knee([1.0, 1.0, 0.94, 1.0, 1.0]);
        assert_eq!(omega_msr(&GRID, &rows), Some(0.9));
    }

    #[test]
    fn ppl_excess_blocks_a_row_retention_cannot_rescue() {
        let mut rows = knee([1.0; 5]);
        rows[4].ppl_excess = Some(OMEGA_PPL_BAND + 0.01);
        assert_eq!(omega_msr(&GRID, &rows), None);
        rows[4].ppl_excess = Some(OMEGA_PPL_BAND);
        assert_eq!(omega_msr(&GRID, &rows), Some(0.5));
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn unsorted_grid_refuses() {
        let grid = [0.9, 0.5, 0.7, 0.8, 0.95];
        let rows = knee([1.0; 5]);
        let _ = omega_msr(&grid, &rows);
    }

    #[test]
    #[should_panic(expected = "align 1:1")]
    fn misaligned_row_count_refuses() {
        let rows = knee([1.0; 5]);
        let _ = omega_msr(&GRID[..4], &rows);
    }

    #[test]
    #[should_panic(expected = "misaligned")]
    fn misaligned_row_tau_refuses() {
        let mut rows = knee([1.0; 5]);
        rows[2].tau = 0.8;
        let _ = omega_msr(&GRID, &rows);
    }

    fn manifest_fixture_value() -> serde_json::Value {
        fixture_body(OMEGA_MANIFEST_VERSION, Some(OmegaScorer::Envelope))
    }

    /// A v1 body (implicitly the MEAN scorer's floors — no scorer field).
    fn manifest_v1_fixture_value() -> serde_json::Value {
        fixture_body(1, None)
    }

    fn fixture_body(version: u32, scorer: Option<OmegaScorer>) -> serde_json::Value {
        let grid = OMEGA_GRID.iter().map(|t| serde_json::json!(t));
        let row = |tau: f32, agreements: u64, ppl_excess: Option<f64>| {
            serde_json::json!({
                "tau": tau,
                "len": "16k",
                "probes": 130,
                "agreements": agreements,
                "retention": agreements as f64 / 130.0,
                "final_flips": 0,
                "mean_abs_dnll": 0.01,
                "ppl_sparse": 3.2,
                "ppl_dense": 3.1,
                "ppl_excess": ppl_excess,
                "sparse_routed": tau < 1.0,
            })
        };
        let rows: Vec<serde_json::Value> = OMEGA_GRID
            .iter()
            .map(|tau| match *tau {
                t if t > 0.8 => row(t, 130, Some(0.01)),
                0.8 => row(0.8, 122, Some(0.01)),
                _ => row(*tau, 130, Some(0.01)),
            })
            .collect();
        let recomputed = omega_msr(
            &OMEGA_GRID,
            &rows
                .iter()
                .zip(OMEGA_GRID.iter())
                .map(|(r, tau)| KneeRow {
                    tau: *tau,
                    retention: r["retention"].as_f64().unwrap() as f32,
                    ppl_excess: r["ppl_excess"].as_f64().map(|x| x as f32),
                })
                .collect::<Vec<_>>(),
        );
        let mut body = serde_json::json!({
            "manifest_version": version,
            "kind": "sparse_prefill_omega_msr",
            "issue": "riir-infer 1156 T3",
            "plan": "riir-infer 625",
            "model_blake3": "deadbeef",
            "model_path_hint": "riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf",
            "corpus_blake3": "feedface",
            "grid": grid.collect::<Vec<_>>(),
            "epsilon": OMEGA_EPS,
            "ppl_band": OMEGA_PPL_BAND,
            "cont_tokens": 64,
            "probe_protocol": "fixture",
            "families": [{
                "family": "needle",
                "omega_msr": recomputed,
                "lengths": {"16k": {"omega_msr": recomputed, "rows": rows}},
            }],
            "box_state": "fixture",
            "notes": [],
        });
        if let Some(s) = scorer {
            body["scorer"] = serde_json::json!(s.as_str());
        }
        body
    }

    fn temp_manifest_paths(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        // The shared-temp-path law: pid-scoped, never a fixed name.
        let dir = std::env::temp_dir().join(format!(
            "sparse_omega_msr_test_{}_{}",
            std::process::id(),
            tag
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let json_path = dir.join("manifest.json");
        (json_path, dir)
    }

    #[test]
    fn parse_and_verify_roundtrips_a_honest_body() {
        let value = manifest_fixture_value();
        let body = serde_json::to_string_pretty(&value).expect("serialize");
        let manifest = parse_and_verify_body(body.as_bytes()).expect("honest fixture verifies");
        let needle = manifest
            .families
            .iter()
            .find(|f| f.family == "needle")
            .expect("needle present");
        // Fixture curve: passes run [0.85..0.95]; 0.8 fails retention
        // (0.94 < 0.95); the lone lower passes (0.5/0.7 at 1.0) never lift
        // the floor → ΩMSR = 0.85, the run's bottom edge.
        assert_eq!(needle.omega_msr, Some(0.85));
    }

    #[cfg(feature = "cubecl_runtime")]
    #[test]
    fn file_pair_roundtrip_and_tamper() {
        let value = manifest_fixture_value();
        let (body, digest) = manifest_body_and_digest(&value).expect("body+digest");
        let (json_path, dir) = temp_manifest_paths("roundtrip");
        std::fs::write(&json_path, &body).expect("write body");
        std::fs::write(digest_sidecar_path(&json_path), format!("{digest}\n"))
            .expect("write sidecar");
        let manifest = load_and_verify_manifest(&json_path).expect("round-trip verifies");
        assert_eq!(manifest.model_blake3, "deadbeef");
        assert_eq!(manifest.families.len(), 1);

        // Tamper the BODY (one byte) — the sidecar no longer matches.
        let tampered = body.replace("needle", "needlx");
        assert_ne!(tampered, body);
        std::fs::write(&json_path, &tampered).expect("write tampered body");
        let err = load_and_verify_manifest(&json_path).expect_err("tampered body refuses");
        assert!(err.contains("digest mismatch"), "got: {err}");

        // Tamper the SIDECAR — same refusal, named direction.
        std::fs::write(&json_path, &body).expect("restore body");
        std::fs::write(
            digest_sidecar_path(&json_path),
            format!("{}\n", "f".repeat(64)),
        )
        .expect("write tampered sidecar");
        let err = load_and_verify_manifest(&json_path).expect_err("tampered sidecar refuses");
        assert!(err.contains("digest mismatch"), "got: {err}");

        // Consts drift: an honest digest over a drifted protocol refuses.
        let mut drifted = manifest_fixture_value();
        drifted["epsilon"] = serde_json::json!(0.5);
        let (body, digest) = manifest_body_and_digest(&drifted).expect("body+digest");
        std::fs::write(&json_path, &body).expect("write drifted body");
        std::fs::write(digest_sidecar_path(&json_path), format!("{digest}\n")).expect("sidecar");
        let err = load_and_verify_manifest(&json_path).expect_err("epsilon drift refuses");
        assert!(err.contains("drift refused"), "got: {err}");

        // Retention lie: stored retention contradicting agreements/probes.
        let mut lying = manifest_fixture_value();
        lying["families"][0]["lengths"]["16k"]["rows"][0]["retention"] =
            serde_json::json!(0.123456);
        let (body, digest) = manifest_body_and_digest(&lying).expect("body+digest");
        std::fs::write(&json_path, &body).expect("write lying body");
        std::fs::write(digest_sidecar_path(&json_path), format!("{digest}\n")).expect("sidecar");
        let err = load_and_verify_manifest(&json_path).expect_err("retention lie refuses");
        assert!(err.contains("may not assert floors"), "got: {err}");

        // ΩMSR lie: stored floor contradicting the re-derived knee.
        let mut lying = manifest_fixture_value();
        lying["families"][0]["omega_msr"] = serde_json::json!(0.95);
        lying["families"][0]["lengths"]["16k"]["omega_msr"] = serde_json::json!(0.95);
        let (body, digest) = manifest_body_and_digest(&lying).expect("body+digest");
        std::fs::write(&json_path, &body).expect("write lying body");
        std::fs::write(digest_sidecar_path(&json_path), format!("{digest}\n")).expect("sidecar");
        let err = load_and_verify_manifest(&json_path).expect_err("ΩMSR lie refuses");
        assert!(err.contains("re-derived"), "got: {err}");

        std::fs::remove_dir_all(dir).expect("cleanup temp dir");
    }

    #[cfg(feature = "cubecl_runtime")]
    #[test]
    fn verify_posture_semantics() {
        let value = manifest_fixture_value();
        let (body, _digest) = manifest_body_and_digest(&value).expect("body+digest");
        let manifest = parse_and_verify_body(body.as_bytes()).expect("verify");
        assert_eq!(manifest.scorer, OmegaScorer::Envelope);
        // Identity always safe — even on a family the manifest omits.
        verify_posture(&manifest, "unheard_of_family", 1.0, OmegaScorer::Envelope)
            .expect("identity is unconditionally safe");
        // At the floor: 0.85 passes.
        verify_posture(&manifest, "needle", 0.85, OmegaScorer::Envelope).expect("floor passes");
        verify_posture(&manifest, "needle", 0.95, OmegaScorer::Envelope)
            .expect("above floor passes");
        // Below the floor refuses, naming both numbers.
        let err = verify_posture(&manifest, "needle", 0.8, OmegaScorer::Envelope)
            .expect_err("below floor refuses");
        assert!(err.contains("0.8") && err.contains("0.85"), "got: {err}");
        // Unknown family refuses below identity.
        let err = verify_posture(&manifest, "unheard_of_family", 0.9, OmegaScorer::Envelope)
            .expect_err("unknown family refuses");
        assert!(err.contains("not in the manifest"), "got: {err}");
        // A floor never blesses the other scorer.
        let err = verify_posture(&manifest, "needle", 0.85, OmegaScorer::Mean)
            .expect_err("scorer mismatch refuses");
        assert!(err.contains("never blesses"), "got: {err}");
    }

    #[test]
    fn v1_manifest_is_implicitly_the_mean_scorer() {
        let value = manifest_v1_fixture_value();
        let body = serde_json::to_string_pretty(&value).expect("serialize");
        let manifest = parse_and_verify_body(body.as_bytes()).expect("v1 body verifies");
        assert_eq!(manifest.scorer, OmegaScorer::Mean);
        verify_posture(&manifest, "needle", 0.85, OmegaScorer::Mean).expect("mean floor passes");
        let err = verify_posture(&manifest, "needle", 0.85, OmegaScorer::Envelope)
            .expect_err("envelope cannot use mean floors");
        assert!(err.contains("never blesses"), "got: {err}");
    }

    #[test]
    fn v1_body_carrying_envelope_scorer_refuses() {
        let mut value = manifest_v1_fixture_value();
        value["scorer"] = serde_json::json!("envelope");
        let body = serde_json::to_string_pretty(&value).expect("serialize");
        let err = parse_and_verify_body(body.as_bytes()).expect_err("v1+envelope drift refuses");
        assert!(err.contains("implicitly mean"), "got: {err}");
    }

    #[test]
    fn v2_body_without_scorer_refuses() {
        let mut value = manifest_fixture_value();
        value["scorer"] = serde_json::Value::Null;
        let body = serde_json::to_string_pretty(&value).expect("serialize");
        let err = parse_and_verify_body(body.as_bytes()).expect_err("v2 needs a scorer");
        assert!(
            err.contains("scorer") && err.contains("missing"),
            "got: {err}"
        );
    }

    #[test]
    fn future_manifest_version_refuses() {
        let mut value = manifest_fixture_value();
        value["manifest_version"] = serde_json::json!(OMEGA_MANIFEST_VERSION + 1);
        let body = serde_json::to_string_pretty(&value).expect("serialize");
        let err = parse_and_verify_body(body.as_bytes()).expect_err("newer protocol refuses");
        assert!(err.contains("NEWER"), "got: {err}");
    }
}
