//! Plan 425 — the LSTN sidecar writer: the checkpoint scorer's tensors
//! cross to the riir-train fine-tune as DATA (riir-train may not dep
//! riir-infer; BOUNDARY.md). Format: magic `LSTN`, u32 version 1, u32 d,
//! f32 eps, then per tensor (u32 len + f32 LE values) in s0w, s0b, s1w,
//! s1b, s3w, s3b order.
//!
//! ```text
//! cargo run --release -p riir-infer-laya --features laya-riir \
//!     --example dump_scorer_tensors -- \
//!     --checkpoint english --out /tmp/p425/scorer_english.lstn
//! ```

use std::io::Write as _;
use std::path::PathBuf;

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::RiirAgent;
use riir_infer_laya::laya::Result as LayaResult;

struct Args {
    checkpoint: String,
    out: PathBuf,
}

fn parse_args() -> Args {
    let mut a = Args {
        checkpoint: "english".into(),
        out: PathBuf::from("scorer.lstn"),
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
            "--out" => a.out = val(&mut i, "--out").into(),
            other => panic!("unknown flag {other}"),
        }
        i += 1;
    }
    a
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
    let st = agent.scorer_tensors();
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&args.out).map_err(|e| {
            riir_infer_laya::laya::LayaError::Runtime(format!("create {}: {e}", args.out.display()))
        })?,
    );
    out.write_all(b"LSTN").map_err(io_err)?;
    out.write_all(&1u32.to_le_bytes()).map_err(io_err)?;
    out.write_all(&(st.d as u32).to_le_bytes()).map_err(io_err)?;
    out.write_all(&st.eps.to_le_bytes()).map_err(io_err)?;
    for t in [&st.s0w, &st.s0b, &st.s1w, &st.s1b, &st.s3w, &st.s3b] {
        out.write_all(&(t.len() as u32).to_le_bytes()).map_err(io_err)?;
        for v in t.iter() {
            out.write_all(&v.to_le_bytes()).map_err(io_err)?;
        }
    }
    out.flush().map_err(io_err)?;
    let s1_params = st.s1w.len();
    eprintln!(
        "[dump_scorer_tensors] {} checkpoint {} d {} eps {} (s1w {s1_params} params) → {}",
        args.checkpoint,
        agent.checkpoint(),
        st.d,
        st.eps,
        args.out.display()
    );
    Ok(())
}

fn io_err(e: std::io::Error) -> riir_infer_laya::laya::LayaError {
    riir_infer_laya::laya::LayaError::Runtime(format!("sidecar write: {e}"))
}
