//! Natural-text corpus loading for the offline calibration bins.
//!
//! One loader, two consumers (`vk_calibration`, `act_diagonal_calibration` —
//! both Issue-883/886-family offline instruments): a `.txt`/`.md` file
//! directly, or a directory of HF datasets-server `page_*.json` files (the
//! sibling riir-train `chat_probe` shape:
//! `rows[].row.messages[].content` + `rows[].row.prompt`).

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// Load the corpus text: a `.txt`/`.md` file directly, or a directory of
/// HF datasets-server `page_*.json` files (the sibling riir-train
/// `chat_probe` shape: rows[].row.messages[].content + prompt).
pub fn load_corpus_text(path: &Path) -> Result<String> {
    if path.is_file() {
        return Ok(std::fs::read_to_string(path)?);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("page_") && n.ends_with(".json"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no page_*.json under {}", path.display());
    }
    let mut text = String::new();
    for f in &files {
        let raw = std::fs::read_to_string(f)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
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
    }
    if text.is_empty() {
        bail!("corpus text empty from {}", path.display());
    }
    Ok(text)
}
