//! The content-addressed `.mlmodelc` compile cache — plan 619 T1.0
//! group 2, lifted verbatim from `riir-infer-laya`'s `ane.rs`
//! (`compile_cached` + `move_dir`). The cache key IS the digest
//! (recompiles never load a stale artifact); staging is process-unique
//! and the final name is claimed atomically (two processes racing one
//! digest both compile, one wins the rename, the loser discards — the
//! shared-fixed-path lesson, mechanized).
//!
//! The cache root is parameterized per lane so each lane's compiled
//! bundles keep their historical home: env var first (`LAYA_ANE_CACHE`
//! for the laya lane, `CLEF_ANE_CACHE` for plan 619's clef lane), else
//! `~/Library/Caches/riir-infer/<default_subdir>` (the macOS cache
//! home — a persistent root survives /tmp cleanup and keeps the
//! compile-once-per-process-tree contract across processes), else a
//! pid-suffixed temp dir when HOME is unset. Trade-off (unchanged):
//! the persistent root grows one bundle per digest, unbounded —
//! acceptable for compiled model bundles; clear it (or set the env) to
//! reclaim.

use std::path::{Path, PathBuf};

use objc2_core_ml::MLModel;
use objc2_foundation::{NSError, NSString, NSURL};

use super::{Error, Result};

/// Compile the `.mlpackage` at `artifact_dir` into a content-addressed
/// `.mlmodelc` cache entry (key = `digest`) and return the compiled
/// bundle path. `cache_env` names the lane's override env;
/// `default_subdir` names the lane's subtree under
/// `~/Library/Caches/riir-infer/`.
pub(crate) fn compile_cached(
    artifact_dir: &Path,
    digest: &str,
    cache_env: &str,
    default_subdir: &str,
) -> Result<PathBuf> {
    let key = &digest[..16.min(digest.len())];
    let cache_root = match std::env::var_os(cache_env) {
        Some(p) => PathBuf::from(p),
        None => match std::env::var_os("HOME") {
            Some(h) => PathBuf::from(h)
                .join("Library/Caches/riir-infer")
                .join(default_subdir),
            None => std::env::temp_dir().join(format!("riir-ane-cache_{}", std::process::id())),
        },
    };
    let final_dir = cache_root.join(format!("mlmodelc-{key}"));
    if final_dir.is_dir() {
        return Ok(final_dir);
    }
    std::fs::create_dir_all(&cache_root)
        .map_err(|e| Error::msg(format!("ane: cache dir {}: {e}", cache_root.display())))?;
    // Compile into a process-unique staging dir, then atomically claim
    // the final name — two processes racing one digest both compile, one
    // wins the rename, the loser discards (never reads a half-written
    // bundle — the shared-fixed-path lesson, mechanized).
    let staging = cache_root.join(format!("staging-{}-{}", key, std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    let src = NSURL::fileURLWithPath_isDirectory(
        &NSString::from_str(&artifact_dir.to_string_lossy()),
        true,
    );
    // SAFETY: plain ObjC class call; the deprecated sync twin of the async
    // compile API — a one-shot load-time call, not worth a semaphore dance.
    #[allow(deprecated)]
    let compiled = unsafe { MLModel::compileModelAtURL_error(&src) }.map_err(|e| {
        Error::msg(format!(
            "ane: Core ML compile failed for {}: {}",
            artifact_dir.display(),
            ns_error_string(&e)
        ))
    })?;
    let compiled_path = compiled
        .path()
        .ok_or_else(|| Error::msg("ane: compiled model has no file path"))?
        .to_string();
    move_dir(Path::new(&compiled_path), &final_dir)?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(final_dir)
}

/// Move `src` into `dst` (rename; same-volume fallback copy), tolerating
/// `dst` having appeared meanwhile (another process won the claim).
fn move_dir(src: &Path, dst: &Path) -> Result<()> {
    if dst.is_dir() {
        let _ = std::fs::remove_dir_all(src);
        return Ok(());
    }
    if std::fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst)
        .map_err(|e| Error::msg(format!("ane: cache move {src:?} → {dst:?}: {e}")))?;
    let _ = std::fs::remove_dir_all(src);
    Ok(())
}

pub(crate) fn ns_error_string(e: &NSError) -> String {
    e.localizedDescription().to_string()
}
