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
//! `--backend gpu` (build with `--features gpu`) runs the stage on the
//! Metal/CubeCL forwarder; `--ctx N` caps the position capacity (default
//! 4096 — prompt + max-new must fit).

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use riir_infer_core::deltanet::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::runner::{run_head, run_local, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
#[cfg(feature = "gpu")]
use riir_infer_dist::stage_gpu::TernaryGpuStage;
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
    backend: String,
    ctx: usize,
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
        backend: "cpu".into(),
        ctx: 4096,
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
            "--backend" => a.backend = val()?,
            "--ctx" => a.ctx = val()?.parse().map_err(|e| format!("--ctx: {e}"))?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    match a.role.as_str() {
        "head" | "tail" => Ok(a),
        _ => Err("--role head|tail required".into()),
    }
}

/// Backend-specific stage construction, behind one signature so the role
/// loops stay generic.
fn build_stage(
    backend: &str,
    config: &Config,
    weights: &Arc<QwenDeltaNetTernaryWeights>,
    range: LayerRange,
    identity: ArtifactIdentity,
) -> Result<Box<dyn StageCompute>, String> {
    match backend {
        "cpu" => Ok(Box::new(TernaryCpuStage::new(
            config.clone(),
            weights.clone(),
            range,
            identity,
        )?)),
        #[cfg(feature = "gpu")]
        "gpu" => Ok(Box::new(TernaryGpuStage::new(
            config,
            weights.clone(),
            range,
            identity,
        )?)),
        other => Err(format!(
            "--backend {other}: unknown or not compiled in (gpu needs --features gpu)"
        )),
    }
}

fn run() -> Result<(), String> {
    let a = parse()?;
    let key = PeerKey::from_hex(
        &std::env::var("RIIR_DIST_KEY").map_err(|_| "RIIR_DIST_KEY (64 hex) not set")?,
    )?;

    let t = Instant::now();
    let identity = ArtifactIdentity::of_gguf(&a.model)?;
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&a.model).map_err(|e| format!("load: {e}"))?;
    // Position capacity: the GPU forwarder sizes its KV to block_size (the
    // Issue-994 VRAM gate refuses the context-sized default); the CPU
    // cache follows the same bound so both backends refuse identically.
    config.block_size = a.ctx;
    let n_layer = config.n_layer as u32;
    eprintln!(
        "[{}] loaded {} layers, n_embd {}, vocab {} in {:.1}s (backend {}, ctx {})",
        a.role,
        n_layer,
        config.n_embd,
        config.vocab_size,
        t.elapsed().as_secs_f64(),
        a.backend,
        a.ctx
    );
    let weights = Arc::new(weights);
    let range = match a.role.as_str() {
        "head" => LayerRange::new(0, a.split),
        _ => LayerRange::new(a.split, n_layer),
    };
    let mut stage = build_stage(&a.backend, &config, &weights, range, identity)?;

    match a.role.as_str() {
        "tail" => {
            let addr = a.listen.ok_or("tail needs --listen ADDR")?;
            let l = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
            eprintln!("[tail] layers [{}, {n_layer}) listening on {addr}", a.split);
            let (sock, peer) = l.accept().map_err(|e| format!("accept: {e}"))?;
            eprintln!("[tail] head connected from {peer}");
            let mut link = TcpTransport::new(sock, key, Direction::Upstream)
                .map_err(|e| format!("socket: {e}"))?;
            let steps =
                run_tail(stage.as_mut(), &mut link, |_, _| {}).map_err(|e| e.to_string())?;
            eprintln!("[tail] served {steps} positions, clean shutdown");
        }
        _ => {
            let addr = a.connect.ok_or("head needs --connect ADDR")?;
            let sock = TcpStream::connect(&addr).map_err(|e| format!("connect {addr}: {e}"))?;
            let mut link = TcpTransport::new(sock, key, Direction::Downstream)
                .map_err(|e| format!("socket: {e}"))?;
            let r = run_head(stage.as_mut(), &mut link, &a.prompt, a.max_new)
                .map_err(|e| e.to_string())?;
            println!("generated: {:?}", r.generated);
            println!(
                "prefill {} tok in {:.3}s · decode {:.2} tok/s ({} steps)",
                a.prompt.len(),
                r.prefill.as_secs_f64(),
                r.decode_tok_s(),
                r.generated.len().saturating_sub(1)
            );
            if a.full_check {
                // Free the head stage first: a GPU stage holds a full model
                // upload, the oracle needs its own.
                drop(stage);
                let mut full = build_stage(
                    &a.backend,
                    &config,
                    &weights,
                    LayerRange::full(n_layer),
                    identity,
                )?;
                let t = Instant::now();
                let want = run_local(full.as_mut(), &a.prompt, a.max_new);
                let tok_s = a.max_new.saturating_sub(1) as f64 / t.elapsed().as_secs_f64();
                println!(
                    "1-box {} oracle: {tok_s:.2} tok/s end-to-end (incl. prefill)",
                    a.backend
                );
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
