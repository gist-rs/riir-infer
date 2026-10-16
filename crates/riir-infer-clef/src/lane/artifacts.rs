//! The clef-text artifact store — plan 619 T1.2. Download-on-demand ON
//! the shared [`riir_infer_ane`] host: per-asset sha256 verify BEFORE
//! use, a `.verified-<rev>` install marker, loud refusal on any mismatch
//! (never a silent re-download). Weights never committed (repo law).
//!
//! The pin (revision + the 9-asset sha256/size table) is the T0.2
//! manifest, re-derived 2026-10-16 from the HF tree API at the pinned
//! revision (LFS oids ARE content sha256; the three small non-LFS files
//! hashed from their raw content) and cross-checked against the
//! FluidUse `ClefTextModelStore` prefixes recorded in Research 012 §7 —
//! `config af356b87…` and `tokenizer be756060…` both match. Trust the
//! checksum, re-derive the manifest (Research 012 §7 law).
//!
//! Transport is the laya posture (weights.rs): a `curl` subprocess per
//! asset (follows the HF resolve redirect chain), `.part` suffix while
//! in flight, atomic rename on success — rustls/reqwest never enter the
//! dep tree. The `ensure_artifacts` fetch closure is injectable and the
//! choreography is table-generic, so the gates run the full
//! stage/verify/install/marker law against a synthetic manifest — the
//! production table and the gates share every line of the law.

use std::path::{Path, PathBuf};

use riir_infer_ane::stage_files;

/// The pinned HF revision — Research 012 §7 (T0.2).
pub const ARTIFACT_REVISION: &str = "3c61a60954478ef385aa42d67d5764642dcc9135";

/// The HF repo (owner/name) the revision resolves in.
pub const ARTIFACT_REPO: &str = "FluidInference/clef-text-0.6b-coreml";

/// One pinned asset: repo-relative path, content sha256 (hex), size in
/// bytes. The 9 assets are exactly FluidUse's `ClefTextModelStore`
/// table (config + tokenizer + embeddings + 3 + 3 for the two
/// packages); `.gitattributes`/`LICENSE`/`README.md` are not lane
/// inputs and are not staged.
pub const ARTIFACTS: &[(&str, &str, u64)] = &[
    (
        "Decoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "3e665021d57e3e7ff05396574a592ab95aea98a72a7744401336b641e796cee7",
        8_076_276,
    ),
    (
        "Decoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "cbe7a6f22befa4edee09e7c806b2b0b9db8533d1ede737540fa82a0261435742",
        880_967_232,
    ),
    (
        "Decoder.mlpackage/Manifest.json",
        "3f54235a774d5891b2199e6d55e6a1de4a2f7aef3ce8b48a22d1904c66a3dffd",
        617,
    ),
    (
        "Head.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "02e92c4f2cd13b5217e6402934c1fbe20a7c8afac19a79d3d02ed0c6279cd7c3",
        457_451,
    ),
    (
        "Head.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "d028526027d6ce12313fbc80ad35f148ecb815f09f83695c5bb2b5872dbd4657",
        205_732_544,
    ),
    (
        "Head.mlpackage/Manifest.json",
        "9eac1a73bb7772ea2ff3cb5167339bd0090ad6a819e3a3671fa8a80ef90f2ad2",
        617,
    ),
    (
        "config.json",
        "af356b87b840722b77068df83612d6ad6efdd19804c6c955e4689a0e10c6bf58",
        406,
    ),
    (
        "embeddings.f16",
        "616f9374ddbaae3a033a0174557d502719154882e21e5d975ab5b58e0244459d",
        311_164_928,
    ),
    (
        "tokenizer.json",
        "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
        11_422_650,
    ),
];

/// The lane artifact root: env `CLEF_COREML_DIR` overrides, else
/// `~/.cache/riir-infer/clef-text-0.6b-coreml` (the plan's default).
pub fn artifacts_dir() -> std::io::Result<PathBuf> {
    if let Ok(dir) = std::env::var("CLEF_COREML_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var("HOME")
        .map_err(|_| std::io::Error::other("CLEF_COREML_DIR unset and no HOME in env"))?;
    Ok(Path::new(&home)
        .join(".cache")
        .join("riir-infer")
        .join("clef-text-0.6b-coreml"))
}

/// sha256 of a file, streamed (the weights are ~GB — never whole-file
/// buffered).
fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path)?;
    let mut h = sha2::Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// The `.verified-<rev>` install marker path.
fn marker_path(dir: &Path) -> PathBuf {
    dir.join(format!(".verified-{ARTIFACT_REVISION}"))
}

/// Cheap per-asset sanity: present at its pinned SIZE (the sha256 walls
/// run at install time; the marker vouches for them).
fn size_ok(path: &Path, size: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.len() == size)
}

/// The real wire: one `curl` per asset from the HF resolve URL (the
/// laya transport posture — see the module doc). `rel` is the
/// repo-relative path; `dest` is where the staged file goes.
pub fn hf_fetch(rel: &str, dest: &Path) -> Result<(), String> {
    let url = format!("https://huggingface.co/{ARTIFACT_REPO}/resolve/{ARTIFACT_REVISION}/{rel}");
    let tmp = dest.with_extension("part");
    let status = std::process::Command::new("curl")
        .args(["--location", "--silent", "--show-error", "--fail", "--output"])
        .arg(&tmp)
        .arg(&url)
        .status()
        .map_err(|e| format!("spawn curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("download failed ({url}): {status}"));
    }
    std::fs::rename(&tmp, dest).map_err(|e| format!("rename {}: {e}", dest.display()))
}

/// Ensure the pinned assets are present, sha256-verified, and marked —
/// the T1.2 `ensure_artifacts` law against the production table,
/// returning the artifact root.
pub fn ensure_artifacts<E: std::fmt::Display>(
    dir: &Path,
    fetch: impl FnMut(&str, &Path) -> std::result::Result<(), E>,
) -> std::io::Result<PathBuf> {
    ensure_artifacts_with(dir, ARTIFACTS, fetch)
}

/// The choreography, table-generic (the gates run it against a
/// synthetic manifest; production passes [`ARTIFACTS`]).
///
/// Fast path: the `.verified-<rev>` marker exists AND every asset is at
/// its pinned size → return (no hashing, no wire). Install path: each
/// MISSING or mis-sized asset stages through `fetch`; EVERY asset —
/// fetched or already-local — then verifies against its pinned sha256
/// BEFORE anything fetched installs, so a bad download never lands and
/// a partial local tree cannot mask a drifted sibling. The marker is
/// written last (tmp + rename), so an interrupted install leaves no
/// marker and the next run re-installs. Any mismatch refuses loud
/// naming the asset and both digests — never a silent re-download,
/// never a partially-verified tree marked good.
pub fn ensure_artifacts_with<E: std::fmt::Display>(
    dir: &Path,
    table: &[(&str, &str, u64)],
    mut fetch: impl FnMut(&str, &Path) -> std::result::Result<(), E>,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let marker = marker_path(dir);
    let all_sized = table.iter().all(|(rel, _, size)| size_ok(&dir.join(rel), *size));
    if marker.is_file() && all_sized {
        return Ok(dir.to_path_buf());
    }

    // Staging tree — one per process, cleaned on every exit path.
    let staging = dir.join(format!(".staging-{}", std::process::id()));
    let result = install(dir, &staging, &marker, table, &mut fetch);
    let _ = std::fs::remove_dir_all(&staging);
    result?;
    Ok(dir.to_path_buf())
}

fn install<E: std::fmt::Display>(
    dir: &Path,
    staging: &Path,
    marker: &Path,
    table: &[(&str, &str, u64)],
    fetch: &mut impl FnMut(&str, &Path) -> std::result::Result<(), E>,
) -> std::io::Result<()> {
    // Fetch ONLY the missing/mis-sized assets — an 880 MB decoder is
    // not re-pulled because a 406-byte config drifted.
    let mut fetched: Vec<usize> = Vec::new();
    for (i, (rel, _, size)) in table.iter().enumerate() {
        if size_ok(&dir.join(rel), *size) {
            continue;
        }
        stage_files(staging, &[rel], |r, d| fetch(r, d)).map_err(|e| {
            std::io::Error::other(format!("clef artifacts: staging {rel} failed: {e}"))
        })?;
        fetched.push(i);
    }
    // Verify EVERY asset — fetched (from staging) or already-local —
    // against its pinned sha256 BEFORE any fetched bytes install.
    for (i, (rel, pin, size)) in table.iter().enumerate() {
        let from_staging = fetched.contains(&i);
        let src = if from_staging {
            staging.join(rel)
        } else {
            dir.join(rel)
        };
        let meta = std::fs::metadata(&src).map_err(|e| {
            std::io::Error::other(format!("clef artifacts: {rel} missing: {e}"))
        })?;
        if meta.len() != *size {
            return Err(std::io::Error::other(format!(
                "clef artifacts: {rel} size {} vs pinned {size} — refusing (never a silent \
                 re-download; delete the artifact dir to force a fresh fetch)",
                meta.len()
            )));
        }
        let got = sha256_file(&src)
            .map_err(|e| std::io::Error::other(format!("clef artifacts: {rel} unreadable: {e}")))?;
        if got != *pin {
            return Err(std::io::Error::other(format!(
                "clef artifacts: {rel} sha256 mismatch — got {}…, pinned {}… (the bytes are \
                 not the manifest's; refusing loud)",
                &got[..16],
                &(*pin)[..16]
            )));
        }
        if from_staging {
            let dest = dir.join(rel);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::rename(&src, &dest)
                .map_err(|e| std::io::Error::other(format!("clef artifacts: install {rel}: {e}")))?;
        }
    }
    // Everything present and verified — write the marker LAST (tmp +
    // rename, atomic; an interrupted run leaves no marker).
    let tmp = marker.with_extension("tmp");
    std::fs::write(&tmp, ARTIFACT_REVISION)?;
    std::fs::rename(&tmp, marker)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gates' synthetic manifest: same choreography, test-owned
    /// pins (sha256 of the mock payloads, computed here).
    fn synth_table() -> Vec<(String, String, u64)> {
        ["a/one.bin", "b/two.bin", "three.json"]
            .iter()
            .map(|rel| {
                let content = format!("payload-for-{rel}");
                let digest = {
                    use sha2::Digest;
                    let mut h = sha2::Sha256::new();
                    h.update(content.as_bytes());
                    format!("{:x}", h.finalize())
                };
                (rel.to_string(), digest, content.len() as u64)
            })
            .collect()
    }

    fn borrow(t: &[(String, String, u64)]) -> Vec<(&str, &str, u64)> {
        t.iter().map(|(r, p, s)| (r.as_str(), p.as_str(), *s)).collect()
    }

    /// A truthful mock wire: writes the synthetic manifest's exact
    /// content; counts calls.
    fn truthful_mock(count: &mut usize) -> impl FnMut(&str, &Path) -> Result<(), String> + '_ {
        move |rel, dest| {
            *count += 1;
            std::fs::write(dest, format!("payload-for-{rel}")).map_err(|e| e.to_string())
        }
    }

    #[test]
    fn install_verifies_marks_and_fast_paths() {
        let tmp = std::env::temp_dir().join(format!("clef_art_1_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let table = synth_table();
        let rows = borrow(&table);
        let dir = tmp.join("artifacts");

        // 1. Cold install: every asset fetched, verified, installed,
        //    marker written.
        let mut calls = 0;
        ensure_artifacts_with(&dir, &rows, truthful_mock(&mut calls)).expect("cold install");
        assert_eq!(calls, 3, "all three assets fetched once");
        assert!(marker_path(&dir).is_file(), "marker written last");
        for (rel, _, size) in rows.clone() {
            assert!(size_ok(&dir.join(rel), size), "{rel} installed");
        }

        // 2. Fast path: marker + sizes → the wire is NEVER touched.
        let mut calls = 0;
        ensure_artifacts_with(&dir, &rows, truthful_mock(&mut calls)).expect("fast path");
        assert_eq!(calls, 0, "marker + sizes must skip the wire");

        // 3. Truncation (marker still present): the size check notices,
        //    the install path re-fetches exactly the truncated asset,
        //    and the tree is whole again.
        std::fs::write(dir.join("three.json"), b"short").unwrap();
        let mut calls = 0;
        ensure_artifacts_with(&dir, &rows, truthful_mock(&mut calls)).expect("repair");
        assert_eq!(calls, 1, "only the truncated asset re-fetched");
        assert!(size_ok(&dir.join("three.json"), table[2].2));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn corrupt_bytes_refuse_loud_and_install_nothing() {
        let tmp = std::env::temp_dir().join(format!("clef_art_2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let table = synth_table();
        let rows = borrow(&table);
        let dir = tmp.join("artifacts");

        let err = ensure_artifacts_with(&dir, &rows, |rel, dest| {
            // Right SIZE, wrong bytes — the corruption must reach the
            // sha256 wall, not die at the size wall.
            let mut bytes = format!("payload-for-{rel}").into_bytes();
            bytes[0] = bytes[0].wrapping_add(1);
            std::fs::write(dest, bytes).map_err(|e| e.to_string())
        })
        .expect_err("wrong bytes must refuse");
        let text = err.to_string();
        assert!(text.contains("sha256 mismatch"), "got: {text}");
        assert!(!marker_path(&dir).is_file(), "no marker on refusal");
        // The two honest assets may have installed (they verified); the
        // corrupt one must NOT exist at its destination.
        assert!(
            !dir.join("three.json").exists(),
            "a refused asset must never install"
        );
        // Staging cleaned.
        let staging: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with(".staging-"))
            .collect();
        assert!(staging.is_empty(), "staging tree must be cleaned: {staging:?}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn size_drift_refuses_and_names_the_asset() {
        let tmp = std::env::temp_dir().join(format!("clef_art_3_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let table = synth_table();
        let rows = borrow(&table);
        let dir = tmp.join("artifacts");

        // A wire whose bytes are the right CONTENT but a wrong recorded
        // size in the table → the size wall fires before the hash.
        let drifted = {
            let mut t = table.clone();
            t[0].2 += 1;
            t
        };
        let drifted_rows = borrow(&drifted);
        let err = ensure_artifacts_with(&dir, &drifted_rows, |rel, dest| {
            std::fs::write(dest, format!("payload-for-{rel}")).map_err(|e| e.to_string())
        })
        .expect_err("size wall must fire");
        assert!(
            err.to_string().contains("size") && err.to_string().contains("a/one.bin"),
            "got: {err}"
        );
        // Production table untouched by the drift arm.
        assert_eq!(rows.len(), table.len());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
