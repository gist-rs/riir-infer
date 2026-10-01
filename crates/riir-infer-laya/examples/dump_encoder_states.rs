//! Plan 425 T1 (riir-train Issue 579) — the teacher fine-tune's extraction
//! bin: walk a cases.jsonl (`{"id","state","question"}` per line, the
//! question in the agent wire form) and write the ENCODER-STATE cache the
//! riir-train head fine-tune trains over, plus the unchanged agent's own
//! logits per row as the parity reference.
//!
//! ```text
//! LAYA_DEVICE=metal cargo run --release -p riir-infer-laya \
//!     --features laya-riir-metal --example dump_encoder_states -- \
//!     --checkpoint english --cases /tmp/p425/cases.jsonl \
//!     --out /tmp/p425/hidden_cache.bin [--limit 200]
//! ```
//!
//! Cache format (all LE): magic `LENC`, u32 version 1, u32 n_rows, then per
//! row: u32 id_len + id bytes, u32 seq_len, u32 d, u32 qtype, u32 n_markers,
//! n_markers × u32, n_markers*d × f32 marker_rows (the head's FROZEN
//! representation half: post-layer gathered rows — the scorer's LN input),
//! seq_len*d × f32 hidden (row-major, PRE-type-emb), n_markers × f32
//! reference logits. The reference logits come from the UNCHANGED forward
//! path (head + temperatures resolved by the caller) — on the CPU device
//! they are the transcription-fidelity target for the riir-train scorer
//! reimplementation (tolerance-gated: same math, not the same reduction
//! order).

use std::io::Write as _;
use std::path::PathBuf;

use riir_infer_laya::laya::Result as LayaResult;
use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::RiirAgent;

const MAGIC: &[u8; 4] = b"LENC";
const VERSION: u32 = 1;

struct Args {
    checkpoint: String,
    cases: PathBuf,
    out: PathBuf,
    limit: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        checkpoint: "english".into(),
        cases: PathBuf::from("cases.jsonl"),
        out: PathBuf::from("hidden_cache.bin"),
        limit: 0,
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
            "--limit" => a.limit = val(&mut i, "--limit").parse().expect("--limit number"),
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

fn load_cases(path: &std::path::Path, limit: usize) -> Result<Vec<CaseRow>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("{}:{ln}: {e}", path.display()))?;
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
    if limit > 0 {
        rows.truncate(limit);
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
        "[dump_encoder_states] ckpt {} device {} d {}",
        args.checkpoint,
        agent.device(),
        {
            // d is carried per row by the seam; the log line reads it from
            // the first encoded row.
            let cases = load_cases(&args.cases, 1).expect("cases");
            match cases.first() {
                Some(c) => {
                    let e = agent
                        .encode_question(&c.state, &c.question)
                        .expect("probe encode");
                    e.d
                }
                None => 0,
            }
        }
    );
    let cases = load_cases(&args.cases, args.limit).expect("cases");
    let t0 = std::time::Instant::now();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out).map_err(|e| {
        riir_infer_laya::laya::LayaError::Runtime(format!("create {}: {e}", args.out.display()))
    })?);
    out.write_all(MAGIC).map_err(io_err)?;
    out.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    out.write_all(&(cases.len() as u32).to_le_bytes())
        .map_err(io_err)?;
    for (n, c) in cases.iter().enumerate() {
        let enc = agent.encode_question(&c.state, &c.question)?;
        let fwd = agent.forward_question(&c.state, &c.question)?;
        assert_eq!(enc.markers.len(), fwd.logits.len(), "marker/logit join");
        out.write_all(&(c.id.len() as u32).to_le_bytes())
            .map_err(io_err)?;
        out.write_all(c.id.as_bytes()).map_err(io_err)?;
        out.write_all(&(enc.seq_len as u32).to_le_bytes())
            .map_err(io_err)?;
        out.write_all(&(enc.d as u32).to_le_bytes())
            .map_err(io_err)?;
        out.write_all(&(enc.qtype as u32).to_le_bytes())
            .map_err(io_err)?;
        out.write_all(&(enc.markers.len() as u32).to_le_bytes())
            .map_err(io_err)?;
        for m in &enc.markers {
            out.write_all(&(*m as u32).to_le_bytes()).map_err(io_err)?;
        }
        for v in &enc.marker_rows {
            out.write_all(&v.to_le_bytes()).map_err(io_err)?;
        }
        for v in &enc.hidden {
            out.write_all(&v.to_le_bytes()).map_err(io_err)?;
        }
        for v in &fwd.logits {
            out.write_all(&v.to_le_bytes()).map_err(io_err)?;
        }
        if (n + 1).is_multiple_of(500) {
            eprintln!(
                "[dump_encoder_states] {}/{} rows · {:.1}s",
                n + 1,
                cases.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    out.flush().map_err(io_err)?;
    eprintln!(
        "[dump_encoder_states] done: {} rows → {} · {:.1}s",
        cases.len(),
        args.out.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn io_err(e: std::io::Error) -> riir_infer_laya::laya::LayaError {
    riir_infer_laya::laya::LayaError::Runtime(format!("cache write: {e}"))
}
