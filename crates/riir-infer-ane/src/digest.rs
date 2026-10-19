//! Artifact digest + verify + staging — the wire-and-bytes half of the
//! ANE host (plan 619 T1.0 group 1), lifted verbatim from
//! `riir-infer-laya`'s `ane.rs` (the blake3-dir-v1 law, the count / byte
//! / digest walls, the staging choreography). Error strings are
//! byte-identical to the lane's historical text; the one lane-specific
//! fragment (the "artifact missing" remedy) is a parameter.

use std::path::{Path, PathBuf};

use super::{Error, Result};

/// Walk `dir` recursively collecting `(relative path, absolute path)`
/// pairs — file order is caller-determined (the digest sorts).
pub fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| Error::msg(format!("ane: readdir {dir:?}: {e}")))?;
    for e in entries {
        let e = e.map_err(|err| Error::msg(format!("ane: readdir entry {dir:?}: {err}")))?;
        let p = e.path();
        if p.is_dir() {
            collect_files(root, &p, out)?;
        } else if p.is_file() {
            let rel = p
                .strip_prefix(root)
                .expect("walk rooted at `root`")
                .to_string_lossy()
                .to_string();
            out.push((rel, p));
        }
    }
    Ok(())
}

/// blake3-dir-v1 over a bundle directory — the conversion tool's digest,
/// byte-for-byte (sorted relative path, `rel \0 data \0` per file).
/// Returns `(hex digest, file count, byte total)` so a caller can pin
/// all three the way the manifest does.
pub fn blake3_dir_v1(dir: &Path) -> Result<(String, usize, u64)> {
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect_files(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut total: u64 = 0;
    let mut h = blake3::Hasher::new();
    for (rel, path) in &files {
        let data = std::fs::read(path)
            .map_err(|e| Error::msg(format!("ane artifact file unreadable {path:?}: {e}")))?;
        total += data.len() as u64;
        h.update(rel.as_bytes());
        h.update(b"\0");
        h.update(&data);
        h.update(b"\0");
    }
    Ok((h.finalize().to_hex().to_string(), files.len(), total))
}

/// The load-time verify: the bundle on disk must match the pinned
/// digest / file count / byte total — a swapped or stale artifact is a
/// hard error (issue 017: download-on-demand, digest-pinned).
/// `files == 0 && bytes == 0` disables the shape walls (a digest-only
/// pin); any other pair enforces both walls exactly.
///
/// `missing_remedy` completes the missing-artifact refusal (each lane
/// names ITS converter: reflex's `ane_convert.py`, plan 619's clef
/// fetcher, …) — every other message is fixed text.
pub fn verify_dir_digest(
    dir: &Path,
    digest: &str,
    digest_files: usize,
    digest_bytes: u64,
    missing_remedy: &str,
) -> Result<()> {
    if !dir.is_dir() {
        return Err(Error::msg(format!(
            "ane artifact missing: {} — {missing_remedy}",
            dir.display()
        )));
    }
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect_files(dir, dir, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let walls_armed = !(digest_files == 0 && digest_bytes == 0);
    let total: u64 = files
        .iter()
        .filter_map(|(_, path)| std::fs::metadata(path).ok())
        .map(|m| m.len())
        .sum();
    if walls_armed && (files.len() != digest_files || total != digest_bytes) {
        return Err(Error::msg(format!(
            "ane artifact shape drift at {}: {} files / {} bytes vs pinned {} / {}",
            dir.display(),
            files.len(),
            total,
            digest_files,
            digest_bytes
        )));
    }
    let mut h = blake3::Hasher::new();
    for (rel, path) in &files {
        let data = std::fs::read(path)
            .map_err(|e| Error::msg(format!("ane artifact file unreadable {path:?}: {e}")))?;
        h.update(rel.as_bytes());
        h.update(b"\0");
        h.update(&data);
        h.update(b"\0");
    }
    let got = h.finalize().to_hex().to_string();
    if got != digest {
        return Err(Error::msg(format!(
            "ane artifact digest mismatch at {} — got {}, pinned {} \
             (the artifact is not the one the manifest verified)",
            dir.display(),
            &got[..16],
            &digest[..16]
        )));
    }
    Ok(())
}

/// Assemble one artifact bundle from the wire into `staging` — the
/// staging half of download-on-demand (the `.mlpackage` layout law —
/// the fixed per-file fetch paths — stays with the CALLER, which also
/// supplies the fetch primitive and then runs [`verify_dir_digest`] on
/// the assembled tree before installing; a bad download never installs).
///
/// Per file: `curl`-free by contract — the fetch closure does the wire
/// (each lane's weights substrate owns its own; the host never links a
/// downloader). Directories are created as needed; a failed file leaves
/// the staging tree for the CALLER to clean (the historical caller
/// removes it on its failure path).
pub fn stage_files<E: std::fmt::Display>(
    staging: &Path,
    rels: &[&str],
    mut fetch: impl FnMut(&str, &Path) -> std::result::Result<(), E>,
) -> Result<()> {
    for rel in rels {
        let dest = staging.join(rel);
        let parent = dest.parent().expect("rel carries a dir");
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::msg(format!("create {parent:?}: {e}")))?;
        fetch(rel, &dest).map_err(|e| Error::msg(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 3-file fixture bundle at the historical `.mlpackage` layout.
    fn write_bundle(dir: &Path, payload: &[u8]) {
        for rel in [
            "Manifest.json",
            "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin",
        ] {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, payload).unwrap();
        }
    }

    #[test]
    fn digest_is_sorted_relative_path_rel_nul_data_nul() {
        let tmp = std::env::temp_dir().join(format!("ane_host_d1_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let bundle = tmp.join("en/L8.mlpackage");
        write_bundle(&bundle, b"0123456789");
        let (digest, files, bytes) = blake3_dir_v1(&bundle).expect("digests");
        assert_eq!(files, 3);
        assert_eq!(bytes, 30);
        // Recompute by hand over the sorted rels — the order IS the law.
        let mut rels: Vec<String> = [
            "Manifest.json",
            "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        rels.sort();
        let mut h = blake3::Hasher::new();
        for rel in &rels {
            h.update(rel.as_bytes());
            h.update(b"\0");
            h.update(b"0123456789");
            h.update(b"\0");
        }
        assert_eq!(digest, h.finalize().to_hex().to_string());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn verify_passes_on_truth_and_refuses_on_swapped_bytes() {
        let tmp = std::env::temp_dir().join(format!("ane_host_d2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let bundle = tmp.join("en/L8.mlpackage");
        write_bundle(&bundle, b"0123456789");
        let (digest, files, bytes) = blake3_dir_v1(&bundle).expect("digests");
        verify_dir_digest(&bundle, &digest, files, bytes, "remedy text").expect("truth verifies");
        // A swapped artifact (same shape, different bytes) refuses at the
        // digest wall, never loads.
        std::fs::write(bundle.join("Manifest.json"), b"XXXXXXXXXX").unwrap();
        let err = verify_dir_digest(&bundle, &digest, files, bytes, "remedy text")
            .expect_err("swapped bytes refuse");
        assert!(err.to_string().contains("digest mismatch"), "got: {err}");
        // A missing directory names the caller's remedy verbatim.
        let err = verify_dir_digest(&tmp.join("gone"), &digest, files, bytes, "run the remedy")
            .expect_err("missing refuses");
        let text = err.to_string();
        assert!(text.contains("ane artifact missing"), "got: {text}");
        assert!(text.contains("run the remedy"), "got: {text}");
        // A count/byte drift refuses at the shape wall.
        std::fs::write(bundle.join("extra.bin"), b"z").unwrap();
        let err =
            verify_dir_digest(&bundle, &digest, files, bytes, "remedy").expect_err("drift refuses");
        assert!(err.to_string().contains("shape drift"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn stage_files_creates_dirs_and_surfaces_fetch_errors() {
        let tmp = std::env::temp_dir().join(format!("ane_host_d3_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let staging = tmp.join(".staging/en/L8.mlpackage");
        let rels = [
            "Manifest.json",
            "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin",
        ];
        let mut calls: Vec<String> = Vec::new();
        stage_files(&staging, &rels, |rel, dest| {
            calls.push(rel.to_string());
            std::fs::write(dest, format!("bytes:{rel}")).map_err(|e| e.to_string())
        })
        .expect("stages");
        assert_eq!(calls.len(), 3, "every rel fetched once");
        for rel in rels {
            assert!(staging.join(rel).is_file(), "{rel} staged");
        }
        // A failing fetch surfaces the error text through the host type.
        let err = stage_files(&staging, &["X"], |_, _| {
            Err::<(), String>("wire down".into())
        })
        .expect_err("fetch error propagates");
        assert!(err.to_string().contains("wire down"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
