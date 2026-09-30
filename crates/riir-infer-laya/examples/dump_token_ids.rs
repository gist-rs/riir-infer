//! riir-train 602 (instinct 014 C2) — the token-id sidecar dump: walk a
//! cases.jsonl (`{"id","state","question"}` per line, the same wire
//! `dump_encoder_states` encodes) and write the TOKN sidecar carrying each
//! row's EXACT token stream (`RiirAgent::tokenize_question` — the
//! `build_sequence` twin, so `ids.len()` joins the LENC cache's `seq_len`
//! row-for-row). The static-vector surrogate's corpus averaging keys
//! per-token hidden states by these ids; the join law is asserted by the
//! consumer, never assumed.
//!
//! CPU posture by construction (tokenization only — no encoder forward),
//! but the agent load honors `LAYA_DEVICE` like every lane; `cpu` is the
//! documented invocation.
//!
//! ```text
//! LAYA_DEVICE=cpu cargo run --release -p riir-infer-laya \
//!     --features laya-riir --example dump_token_ids -- \
//!     --checkpoint english --cases /tmp/t602/train_cases.jsonl \
//!     --out /tmp/t602/train_tokens.tokn
//! ```
//!
//! Sidecar format (all LE): magic `TOKN`, u32 version 1, u32 n_rows, then
//! per row: u32 id_len + id bytes, u32 seq_len, seq_len × u32 token ids.

use std::io::Write as _;
use std::path::PathBuf;

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::RiirAgent;
use riir_infer_laya::laya::Result as LayaResult;

const MAGIC: &[u8; 4] = b"TOKN";
const VERSION: u32 = 1;

struct Args {
    checkpoint: String,
    cases: PathBuf,
    out: PathBuf,
}

fn parse_args() -> Args {
    let mut a = Args {
        checkpoint: "english".into(),
        cases: PathBuf::from("cases.jsonl"),
        out: PathBuf::from("tokens.tokn"),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let val = |i: &mut usize, name: &str| -> String {
            *i += 1;
            args.get(*i)
                .unwrap_or_else(|| panic!("{name} needs a value"))
                .clone()
        };
        match args[i].as_str() {
            "--checkpoint" => a.checkpoint = val(&mut i, "--checkpoint"),
            "--cases" => a.cases = val(&mut i, "--cases").into(),
            "--out" => a.out = val(&mut i, "--out").into(),
            other => panic!("unknown flag {other}"),
        }
        i += 1;
    }
    a
}

struct CaseRow {
    id: String,
    state: serde_json::Value,
    question: serde_json::Value,
}

fn load_cases(path: &std::path::Path) -> Result<Vec<CaseRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("{}:{ln}: {e}", path.display()))?;
        rows.push(CaseRow {
            id: v
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            state: v
                .get("state")
                .cloned()
                .ok_or_else(|| format!("{}:{ln}: no state", path.display()))?,
            question: v
                .get("question")
                .cloned()
                .ok_or_else(|| format!("{}:{ln}: no question", path.display()))?,
        });
    }
    Ok(rows)
}

fn main() -> LayaResult<()> {
    let args = parse_args();
    let ck = match args.checkpoint.as_str() {
        "english" => Checkpoint::English,
        "multilingual" => Checkpoint::Multilingual,
        "typed" => Checkpoint::TypedDecisions,
        other => panic!("unknown checkpoint {other}"),
    };
    let root = riir_infer_laya::laya::weights::weights_root();
    let agent = RiirAgent::load(&root, ck)?;
    eprintln!(
        "[dump_token_ids] ckpt {} device {} (tokenization only — no forward)",
        args.checkpoint,
        agent.device()
    );
    let cases = load_cases(&args.cases).expect("cases");
    let t0 = std::time::Instant::now();
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&args.out).map_err(|e| {
            riir_infer_laya::laya::LayaError::Runtime(format!(
                "create {}: {e}",
                args.out.display()
            ))
        })?,
    );
    out.write_all(MAGIC).map_err(io_err)?;
    out.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    out.write_all(&(cases.len() as u32).to_le_bytes()).map_err(io_err)?;
    let mut max_id: u32 = 0;
    for (n, c) in cases.iter().enumerate() {
        let (ids, _markers) = agent
            .tokenize_question(&c.state, &c.question)
            .map_err(|e| {
                riir_infer_laya::laya::LayaError::Runtime(format!(
                    "row {} (id {}): {e:?}",
                    n, c.id
                ))
            })?;
        max_id = max_id.max(ids.iter().copied().max().unwrap_or(0));
        out.write_all(&(c.id.len() as u32).to_le_bytes()).map_err(io_err)?;
        out.write_all(c.id.as_bytes()).map_err(io_err)?;
        out.write_all(&(ids.len() as u32).to_le_bytes()).map_err(io_err)?;
        for t in &ids {
            out.write_all(&t.to_le_bytes()).map_err(io_err)?;
        }
        if (n + 1).is_multiple_of(2000) {
            eprintln!(
                "[dump_token_ids] {}/{} rows · {:.1}s",
                n + 1,
                cases.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    out.flush().map_err(io_err)?;
    eprintln!(
        "[dump_token_ids] done: {} rows → {} · max token id {max_id} · {:.1}s",
        cases.len(),
        args.out.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn io_err(e: std::io::Error) -> riir_infer_laya::laya::LayaError {
    riir_infer_laya::laya::LayaError::Runtime(format!("sidecar write: {e}"))
}
