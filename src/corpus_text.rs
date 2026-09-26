//! Natural-text corpus loading for the offline calibration bins.
//!
//! One loader, three consumers (`vk_calibration`, `act_diagonal_calibration`,
//! `act_retention_walk` — the Issue-883/886-family offline instruments): a
//! `.txt`/`.md` file directly, or a directory of either HF datasets-server
//! `page_*.json` files (the sibling riir-train `chat_probe` shape:
//! `rows[].row.messages[].content` + `rows[].row.prompt`) or plain
//! `.txt`/`.md` files (one file = one page; the walk's family buckets).

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// Load the corpus as `(name, text)` per page file (sorted by file name), or
/// one `(stem, text)` entry for a single `.txt`/`.md` file. A directory may
/// hold either HF datasets-server `page_*.json` files (the chat_probe shape)
/// or plain `.txt`/`.md` files — one file = one page. The page granularity of
/// the Issue-014 T3 retention walk: a page is one deterministic content
/// bucket.
pub fn load_corpus_pages(path: &Path) -> Result<Vec<(String, String)>> {
    if path.is_file() {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("corpus")
            .to_string();
        return Ok(vec![(stem, std::fs::read_to_string(path)?)]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| {
                    (n.starts_with("page_") && n.ends_with(".json"))
                        || n.ends_with(".txt")
                        || n.ends_with(".md")
                })
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no page_*.json (or *.txt/*.md) under {}", path.display());
    }
    let mut pages = Vec::with_capacity(files.len());
    for f in &files {
        let name = f
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("page")
            .trim_start_matches("page_")
            .to_string();
        let text = if f.extension().is_some_and(|e| e == "json") {
            extract_page_text(f)?
        } else {
            std::fs::read_to_string(f)?
        };
        pages.push((name, text));
    }
    if pages.iter().all(|(_, t)| t.is_empty()) {
        bail!("corpus text empty from {}", path.display());
    }
    Ok(pages)
}

/// Load the corpus text: a `.txt`/`.md` file directly, or a directory of
/// HF datasets-server `page_*.json` files (the sibling riir-train
/// `chat_probe` shape: rows[].row.messages[].content + prompt).
pub fn load_corpus_text(path: &Path) -> Result<String> {
    let pages = load_corpus_pages(path)?;
    let mut text = String::new();
    for (_, t) in &pages {
        text.push_str(t);
    }
    if text.is_empty() {
        bail!("corpus text empty from {}", path.display());
    }
    Ok(text)
}

/// One page file → its natural text (the loader's extraction, verbatim).
fn extract_page_text(f: &Path) -> Result<String> {
    let raw = std::fs::read_to_string(f)?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let mut text = String::new();
    if let Some(rows) = v.get("rows").and_then(|r| r.as_array()) {
        for row in rows {
            if let Some(msgs) = row.pointer("/row/messages").and_then(|m| m.as_array()) {
                for m in msgs {
                    if let Some(c) = m.get("content").and_then(|c| c.as_str()) {
                        text.push_str(c);
                        text.push('\n');
                    }
                }
            }
            if let Some(p) = row.pointer("/row/prompt").and_then(|p| p.as_str()) {
                text.push_str(p);
                text.push('\n');
            }
        }
    }
    Ok(text)
}
