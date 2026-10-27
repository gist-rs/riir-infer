//! `dist_stage` — run one stage of a 2-host pipeline (Plan 623 T2.2 PoC).
//!
//! ```text
//! # tail (listens; owns layers [k, n) + lm_head + sampler)
//! RIIR_DIST_KEY=<64 hex> dist_stage --role tail --model M.gguf --split 32 --listen 192.168.10.2:7623
//! # head (connects; owns embed + layers [0, k); drives the token stream)
//! RIIR_DIST_KEY=<64 hex> dist_stage --role head --model M.gguf --split 32 --connect 192.168.10.2:7623 \
//!     --prompt 1,2,3 --max-new 16
//! ```
//!
//! Token ids in, token ids out (no tokenizer in the PoC). `--full-check`
//! on the head additionally runs the 1-box forward locally and asserts
//! the 2-host token stream is byte-identical (the G1 gate, live).

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_dist::runner::{run_head, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

struct Args {
    role: String,
    model: PathBuf,
    split: u32,
    listen: Option<String>,
    connect: Option<String>,
    prompt: Vec<u32>,
    max_new: usize,
    full_check: bool,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        role: String::new(),
        model: PathBuf::new(),
        split: 0,
        listen: None,
        connect: None,
        prompt: vec![1],
        max_new: 16,
        full_check: false,
    };
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--role" => a.role = val()?,
            "--model" => a.model = val()?.into(),
            "--split" => a.split = val()?.parse().map_err(|e| format!("--split: {e}"))?,
            "--listen" => a.listen = Some(val()?),
            "--connect" => a.connect = Some(val()?),
            "--prompt" => {
                a.prompt = val()?
                    .split(',')
                    .map(|t| {
                        t.trim()
                            .parse::<u32>()
                            .map_err(|e| format!("--prompt: {e}"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "--max-new" => a.max_new = val()?.parse().map_err(|e| format!("--max-new: {e}"))?,
            "--full-check" => a.full_check = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    match a.role.as_str() {
        "head" | "tail" => Ok(a),
        _ => Err("--role head|tail required".into()),
    }
}

fn run() -> Result<(), String> {
    let a = parse()?;
    let key = PeerKey::from_hex(
        &std::env::var("RIIR_DIST_KEY").map_err(|_| "RIIR_DIST_KEY (64 hex) not set")?,
    )?;

    let t = Instant::now();
    let identity = ArtifactIdentity::of_gguf(&a.model)?;
    let (config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&a.model).map_err(|e| format!("load: {e}"))?;
    let n_layer = config.n_layer as u32;
    eprintln!(
        "[{}] loaded {} layers, n_embd {}, vocab {} in {:.1}s",
        a.role,
        n_layer,
        config.n_embd,
        config.vocab_size,
        t.elapsed().as_secs_f64()
    );
    let weights = Arc::new(weights);
    let range = match a.role.as_str() {
        "head" => LayerRange::new(0, a.split),
        _ => LayerRange::new(a.split, n_layer),
    };
    let mut stage = TernaryCpuStage::new(config.clone(), weights.clone(), range, identity)?;

    match a.role.as_str() {
        "tail" => {
            let addr = a.listen.ok_or("tail needs --listen ADDR")?;
            let l = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
            eprintln!("[tail] layers [{}, {n_layer}) listening on {addr}", a.split);
            let (sock, peer) = l.accept().map_err(|e| format!("accept: {e}"))?;
            eprintln!("[tail] head connected from {peer}");
            let mut link = TcpTransport::new(sock, key, Direction::Upstream)
                .map_err(|e| format!("socket: {e}"))?;
            let steps = run_tail(&mut stage, &mut link, |_, _| {}).map_err(|e| e.to_string())?;
            eprintln!("[tail] served {steps} positions, clean shutdown");
        }
        _ => {
            let addr = a.connect.ok_or("head needs --connect ADDR")?;
            let sock = TcpStream::connect(&addr).map_err(|e| format!("connect {addr}: {e}"))?;
            let mut link = TcpTransport::new(sock, key, Direction::Downstream)
                .map_err(|e| format!("socket: {e}"))?;
            let r =
                run_head(&mut stage, &mut link, &a.prompt, a.max_new).map_err(|e| e.to_string())?;
            println!("generated: {:?}", r.generated);
            println!(
                "prefill {} tok in {:.3}s · decode {:.2} tok/s ({} steps)",
                a.prompt.len(),
                r.prefill.as_secs_f64(),
                r.decode_tok_s(),
                r.generated.len().saturating_sub(1)
            );
            if a.full_check {
                let mut full =
                    TernaryCpuStage::new(config, weights, LayerRange::full(n_layer), identity)?;
                let want = riir_infer_dist::runner::run_local(&mut full, &a.prompt, a.max_new);
                match want == r.generated {
                    true => println!("G1 PASS: 2-host stream == 1-box stream"),
                    false => return Err(format!("G1 FAIL: 1-box {want:?}")),
                }
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dist_stage: {e}");
            ExitCode::FAILURE
        }
    }
}
