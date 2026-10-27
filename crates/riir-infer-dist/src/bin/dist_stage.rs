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
//! Token ids in (`--prompt 1,2,3`) or text in (`--text "..."`, encoded with
//! the GGUF's own BPE vocab; `--chat` wraps it in the Qwen chat template);
//! the head prints the generated ids and their decoded text, and stops at
//! the vocab's EOS (`--no-eos` generates the full `--max-new`). `--full-check`
//! on the head additionally runs the 1-box forward locally and asserts
//! the 2-host token stream is byte-identical (the G1 gate, live).
//! `--backend gpu` (build with `--features gpu`) runs the stage on the
//! Metal/CubeCL forwarder; `--ctx N` caps the position capacity (default
//! 4096 — prompt + max-new must fit). Each stage loads ONLY its own slice
//! of the model (Plan 623 T2.2c); `--full-check` reloads the whole model
//! for the oracle after the pipeline run. `--prefill-chunk N` sends the
//! prompt in ~N-position frames through the stages' batched prefill
//! (T2.3b; default 128 on `--backend gpu` — the measured 2-box optimum —
//! and 1 on cpu, where chunking only forfeits the per-position overlap);
//! the `--full-check` oracle prefills at the same width. T2.1b socket
//! knobs: `--sock-buf BYTES` sets `SO_SNDBUF`/`RCVBUF` on the stage link
//! (read-back printed); `--iface NAME` pins the link to that interface
//! (macOS `IP_BOUND_IF`, Linux `SO_BINDTODEVICE`) — applied before
//! bind/connect so the route is actually pinned. Both default to std
//! behavior (byte-identical). T2.5 boot probe: `--plan-report 1,16,32`
//! probes those candidate splits on THIS host (scoped loads, timed
//! decode) and prints one JSON record line — two records + a link cost
//! feed `planner::plan_split` (the manual `--split` always overrides).
//! T2.4 multi-stream: `--streams N` runs N round-robined copies of the
//! SAME request over one link (one stage instance per stream — per-stream
//! caches, weights shared), head and tail alike; the head prints per-
//! stream decode rates and the AGGREGATE tok/s (the pipeline-bubble read).
//! `--streams 1` (default) is the single-stream runner byte-identically.

use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use riir_infer_core::deltanet::QwenDeltaNetTernaryWeights;
use riir_infer_core::deltanet::WeightScope;
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;
use riir_infer_dist::multi::{MultiRequest, run_head_multi, run_tail_multi};
use riir_infer_dist::runner::{HeadOptions, run_head_with, run_local_with, run_tail};
use riir_infer_dist::sockopts::{self, Applied, LinkOptions};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage, stage_scope};
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
    prefill_chunk: Option<usize>,
    text: Option<String>,
    chat: bool,
    no_eos: bool,
    iface: Option<String>,
    sock_buf: Option<usize>,
    /// T2.4: stream count for the multi-stream round-robin runner (1 =
    /// the single-stream path, byte-identical). The request is duplicated
    /// across streams — the demo of the aggregate bubble fill.
    streams: usize,
    /// T2.5 boot probe: probe these candidate splits on THIS host's
    /// stage and print the probe record (one JSON line), then exit —
    /// no serving. The record crosses hosts by the operator; the
    /// planner (`riir_infer_dist::planner::plan_split`) merges two
    /// records + the link cost into the split choice.
    plan_report: Option<Vec<u32>>,
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
        prefill_chunk: None,
        text: None,
        chat: false,
        no_eos: false,
        iface: None,
        sock_buf: None,
        streams: 1,
        plan_report: None,
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
            "--text" => a.text = Some(val()?),
            "--chat" => a.chat = true,
            "--no-eos" => a.no_eos = true,
            "--backend" => a.backend = val()?,
            "--ctx" => a.ctx = val()?.parse().map_err(|e| format!("--ctx: {e}"))?,
            "--prefill-chunk" => {
                a.prefill_chunk = Some(
                    val()?
                        .parse()
                        .map_err(|e| format!("--prefill-chunk: {e}"))?,
                )
            }
            "--iface" => a.iface = Some(val()?),
            "--sock-buf" => {
                a.sock_buf = Some(val()?.parse().map_err(|e| format!("--sock-buf: {e}"))?)
            }
            "--streams" => a.streams = val()?.parse().map_err(|e| format!("--streams: {e}"))?,
            "--plan-report" => {
                a.plan_report = Some(
                    val()?
                        .split(',')
                        .map(|t| {
                            t.trim()
                                .parse::<u32>()
                                .map_err(|e| format!("--plan-report: {e}"))
                        })
                        .collect::<Result<_, _>>()?,
                )
            }
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

/// Scoped load + the position-capacity clamp. The GPU forwarder sizes its
/// KV to `block_size` (the Issue-994 VRAM gate refuses the loader's
/// context-sized default); the CPU cache follows the same bound so both
/// backends refuse identically.
fn load(
    model: &std::path::Path,
    ctx: usize,
    scope: impl FnOnce(&Config) -> WeightScope,
) -> Result<(Config, Arc<QwenDeltaNetTernaryWeights>), String> {
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf_scoped(model, scope)
        .map_err(|e| format!("load: {e}"))?;
    config.block_size = ctx;
    Ok((config, Arc::new(weights)))
}

/// The head's text front: the GGUF's own BPE vocab (T2.2d).
fn tokenizer(model: &std::path::Path) -> Result<BpeTokenizer, String> {
    let gguf = GgufFile::open(model).map_err(|e| format!("open {}: {e}", model.display()))?;
    BpeTokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))
}

/// Qwen chat template around one user turn (the tokenizer's special-token
/// matcher keeps `<|im_start|>` / `<|im_end|>` whole).
fn chat_turn(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn run() -> Result<(), String> {
    let a = parse()?;
    let key = PeerKey::from_hex(
        &std::env::var("RIIR_DIST_KEY").map_err(|_| "RIIR_DIST_KEY (64 hex) not set")?,
    )?;
    let link_opts = LinkOptions {
        sock_buf: a.sock_buf,
        iface: a.iface.clone(),
    };
    let applied_line = |what: &str, ap: Applied| {
        format!(
            "[{what}] sock-buf snd {} rcv {}{}",
            ap.snd,
            ap.rcv,
            match ap.iface {
                Some(idx) => format!(" iface#{idx}"),
                None => String::new(),
            }
        )
    };

    let t = Instant::now();
    let identity = ArtifactIdentity::of_gguf(&a.model)?;
    let (split, head) = (a.split, a.role == "head");
    let range_of = |n_layer: u32| match head {
        true => LayerRange::new(0, split),
        false => LayerRange::new(split, n_layer),
    };
    let (config, weights) = load(&a.model, a.ctx, |c| {
        stage_scope(range_of(c.n_layer as u32), c.n_layer as u32)
    })?;
    let n_layer = config.n_layer as u32;
    let range = range_of(n_layer);
    eprintln!(
        "[{}] loaded layers [{}, {}) of {n_layer}, n_embd {}, vocab {} in {:.1}s (backend {}, ctx {})",
        a.role,
        range.lo,
        range.hi,
        config.n_embd,
        config.vocab_size,
        t.elapsed().as_secs_f64(),
        a.backend,
        a.ctx
    );

    // T2.5 boot probe: per-candidate scoped loads, timed decode, one
    // JSON record line, exit. `k == n_layer` on the head probes the
    // FULL range — the 1-box reference row the planner needs.
    if let Some(splits) = &a.plan_report {
        let steps = 8usize;
        let rounds = 3usize;
        let mut rows = String::new();
        for (i, &k) in splits.iter().enumerate() {
            if k == 0 || k > n_layer {
                return Err(format!("--plan-report: split {k} outside 1..={n_layer}"));
            }
            let cand = match head {
                true => LayerRange::new(0, k),
                false => LayerRange::new(k, n_layer),
            };
            let (c2, w2) = load(&a.model, a.ctx, |c| stage_scope(cand, c.n_layer as u32))?;
            let mut st = build_stage(&a.backend, &c2, &w2, cand, identity)?;
            let spt = riir_infer_dist::planner::probe_decode_spt(st.as_mut(), 0, steps, rounds)
                .map_err(|e| format!("probe split {k}: {e}"))?;
            if i > 0 {
                rows.push(',');
            }
            rows.push_str(&format!("[{k},{spt:.6}]"));
            eprintln!(
                "[plan] {} split {k}: {spt:.6} s/tok (decode, {steps}x{rounds})",
                a.role
            );
        }
        println!(
            "{{\"host\":\"{}\",\"n_layer\":{},\"hidden\":{},\"rows\":[{rows}]}}",
            a.role, n_layer, config.n_embd
        );
        return Ok(());
    }

    // T2.4: one stage instance per stream (per-stream caches; the weights
    // Arc-share inside the constructor). 1 = the single-stream runner.
    let streams = a.streams.max(1);
    let mut stages: Vec<Box<dyn StageCompute>> = (0..streams)
        .map(|_| build_stage(&a.backend, &config, &weights, range, identity))
        .collect::<Result<_, _>>()?;
    let mut stage_refs: Vec<&mut dyn StageCompute> = stages
        .iter_mut()
        .map(|b| b.as_mut() as &mut dyn StageCompute)
        .collect();

    match a.role.as_str() {
        "tail" => {
            let addr = a.listen.ok_or("tail needs --listen ADDR")?;
            // std-equivalent resolution: TcpListener::bind accepted both
            // IP literals and hostnames — keep both.
            let addr: std::net::SocketAddr = addr
                .to_socket_addrs()
                .map_err(|e| format!("--listen {addr}: {e}"))?
                .next()
                .ok_or(format!("--listen {addr}: no address"))?;
            let (l, applied) =
                sockopts::listen(addr, &link_opts).map_err(|e| format!("listen {addr}: {e}"))?;
            eprintln!("{}", applied_line("tail", applied));
            eprintln!(
                "[tail] layers [{}, {n_layer}) × {streams} stream(s) listening on {addr}",
                a.split
            );
            let (sock, peer) = l.accept().map_err(|e| format!("accept: {e}"))?;
            eprintln!("[tail] head connected from {peer}");
            let mut link = TcpTransport::new(sock, key, Direction::Upstream)
                .map_err(|e| format!("socket: {e}"))?;
            let steps = match streams {
                1 => run_tail(stage_refs[0], &mut link, |_, _| {}),
                _ => run_tail_multi(&mut stage_refs, &mut link, |_, _, _| {}),
            }
            .map_err(|e| e.to_string())?;
            eprintln!("[tail] served {steps} positions, clean shutdown");
        }
        _ => {
            let tok = tokenizer(&a.model)?;
            let prompt: Vec<u32> = match &a.text {
                Some(t) => {
                    let text = match a.chat {
                        true => chat_turn(t),
                        false => t.clone(),
                    };
                    tok.encode(&text).into_iter().map(|id| id as u32).collect()
                }
                None => a.prompt.clone(),
            };
            if prompt.is_empty() {
                return Err("empty prompt after encoding".into());
            }
            let addr = a.connect.ok_or("head needs --connect ADDR")?;
            let addr: std::net::SocketAddr = addr
                .to_socket_addrs()
                .map_err(|e| format!("--connect {addr}: {e}"))?
                .next()
                .ok_or(format!("--connect {addr}: no address"))?;
            let (sock, applied) =
                sockopts::dial(addr, &link_opts).map_err(|e| format!("connect {addr}: {e}"))?;
            eprintln!("{}", applied_line("head", applied));
            let mut link = TcpTransport::new(sock, key, Direction::Downstream)
                .map_err(|e| format!("socket: {e}"))?;
            let opts = HeadOptions {
                prefill_chunk: a.prefill_chunk.unwrap_or(match a.backend.as_str() {
                    "gpu" => 128,
                    _ => 1,
                }),
                eos: match a.no_eos {
                    true => None,
                    false => Some(tok.eos_id() as u32),
                },
                ..HeadOptions::default()
            };
            let (reports, wall) = match streams {
                1 => {
                    let r = run_head_with(stage_refs[0], &mut link, &prompt, a.max_new, opts)
                        .map_err(|e| e.to_string())?;
                    (vec![r], None)
                }
                _ => {
                    let reqs: Vec<MultiRequest> = (0..streams)
                        .map(|_| MultiRequest::new(prompt.clone(), a.max_new))
                        .collect();
                    let t = Instant::now();
                    let rs = run_head_multi(&mut stage_refs, &mut link, &reqs, opts)
                        .map_err(|e| e.to_string())?;
                    (rs, Some(t.elapsed()))
                }
            };
            println!("prompt: {} tokens × {streams} stream(s)", prompt.len());
            for (i, r) in reports.iter().enumerate() {
                println!(
                    "stream {i}: decode {:.2} tok/s ({} steps) prefill {:.3}s",
                    r.decode_tok_s(),
                    r.generated.len().saturating_sub(1),
                    r.prefill.as_secs_f64(),
                );
            }
            let first = reports.first().ok_or("no reports")?;
            println!("generated: {:?}", first.generated);
            let ids: Vec<usize> = first.generated.iter().map(|&t| t as usize).collect();
            println!("text: {:?}", tok.decode(&ids));
            match wall {
                Some(w) => {
                    let total: usize = reports
                        .iter()
                        .map(|r| r.generated.len().saturating_sub(1))
                        .sum();
                    println!(
                        "aggregate: {} tokens in {:.3}s = {:.2} tok/s ({} streams, wall-clock)",
                        total,
                        w.as_secs_f64(),
                        total as f64 / w.as_secs_f64().max(1e-9),
                        streams
                    );
                }
                None => println!(
                    "prefill {} tok in {:.3}s · decode {:.2} tok/s ({} steps)",
                    prompt.len(),
                    first.prefill.as_secs_f64(),
                    first.decode_tok_s(),
                    first.generated.len().saturating_sub(1)
                ),
            }
            if a.full_check {
                // Free the head stage + its scoped weights first, then
                // load the whole model for the oracle.
                drop(stage_refs);
                drop(stages);
                drop(weights);
                let (_, full_weights) = load(&a.model, a.ctx, |c| WeightScope::full(c.n_layer))?;
                let mut full = build_stage(
                    &a.backend,
                    &config,
                    &full_weights,
                    LayerRange::full(n_layer),
                    identity,
                )?;
                let t = Instant::now();
                let want = run_local_with(full.as_mut(), &prompt, a.max_new, opts);
                let tok_s = a.max_new.saturating_sub(1) as f64 / t.elapsed().as_secs_f64();
                println!(
                    "1-box {} oracle: {tok_s:.2} tok/s end-to-end (incl. prefill)",
                    a.backend
                );
                for (i, r) in reports.iter().enumerate() {
                    match want == r.generated {
                        true => println!("G1 PASS: stream {i} == 1-box stream"),
                        false => return Err(format!("G1 FAIL: stream {i} 1-box {want:?}")),
                    }
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
