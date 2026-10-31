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
//! T2.5 probe EXCHANGE (the wire step — no hand-carried JSON): the
//! tail runs `--answer-probe` (answers ping/probe frames, never
//! serves), the head runs `--plan-wire 1,16,32` (probes itself, asks
//! the tail over the wire, MEASURES the RTT on the same link, composes
//! `plan_split` live, prints the plan, exits). The link's payload bw
//! is carried (`--link-bw BPS`, default the T0.2 TB5 row 8.6 GB/s — at
//! decode-hop sizes it is ~0.5% of the hop); the RTT is measured.
//! T2.4 multi-stream: `--streams N` runs N round-robined copies of the
//! SAME request over one link (one stage instance per stream — per-stream
//! caches, weights shared), head and tail alike; the head prints per-
//! stream decode rates and the AGGREGATE tok/s (the pipeline-bubble read).
//! `--streams 1` (default) is the single-stream runner byte-identically.
//! The fused decode batch is OPT-IN (`RIIR_DIST_PAIR=1`, loopback rigs
//! only — Bench 913: it forfeits the 2-box head⇄tail GPU overlap).

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
use riir_infer_dist::runner::{HeadOptions, run_head_with, run_local_with_report, run_tail};
use riir_infer_dist::sockopts::{self, Applied, LinkOptions};
use riir_infer_dist::spec_fill::{RepeatLastDrafter, run_head_spec, run_tail_spec};
use riir_infer_dist::spec_mtp::{BmthHead, MtpHeadDrafter};

/// The bin's drafter seat (T5.1d): a concrete enum — `run_head_spec` is
/// generic over a SIZED drafter, and both arms delegate the roll seam
/// (repeat-last ignores it; the MTP head drafts through it).
enum BinDrafter {
    Repeat(RepeatLastDrafter),
    Mtp(MtpHeadDrafter),
}

impl riir_infer_dist::spec_fill::DraftSource for BinDrafter {
    fn propose(&mut self, ctx: &riir_infer_dist::spec_fill::DraftCtx<'_>) -> Vec<u32> {
        match self {
            BinDrafter::Repeat(d) => d.propose(ctx),
            BinDrafter::Mtp(d) => d.propose(ctx),
        }
    }

    fn propose_with_roll(
        &mut self,
        ctx: &riir_infer_dist::spec_fill::DraftCtx<'_>,
        roll: &mut dyn FnMut(u32, usize, &mut [f32]),
    ) -> Vec<u32> {
        match self {
            BinDrafter::Repeat(d) => d.propose_with_roll(ctx, roll),
            BinDrafter::Mtp(d) => d.propose_with_roll(ctx, roll),
        }
    }
}
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage, stage_scope};
#[cfg(feature = "gpu")]
use riir_infer_dist::stage_gpu::TernaryGpuStage;
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport, Transport};

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
    /// T2.5 probe EXCHANGE (the wire step — the landing residue): the
    /// head probes its own candidates, asks the TAIL for its record
    /// over the wire (an `--answer-probe` tail), measures the RTT on
    /// the same link, composes `plan_split` live and prints the plan;
    /// then exits. Candidates are strictly 2-stage (`1..n_layer`).
    plan_wire: Option<Vec<u32>>,
    /// T2.5 probe EXCHANGE tail posture: listen, answer ping/probe
    /// frames until Shutdown, exit — never serves. `--split` still
    /// names the slice the boot load takes (the serving posture's
    /// memory shape); the probes load their own scoped candidates.
    answer_probe: bool,
    /// The link's payload bandwidth for the plan's hop term (bytes/s).
    /// `None` MEASURES it live (the echo-ack bulk — the tail does pure
    /// I/O, so the timing is wire time); `--link-bw` carries the
    /// operator's override for a link the measurement cannot cross.
    /// At decode-hop sizes (~20 KB on a 27B hidden) the bw term is
    /// ~0.5% of the TB5 hop — the measurement exists so a WRONG link
    /// (WiFi, a different fabric) can never silently price a plan with
    /// the TB5 default.
    link_bw: Option<f64>,
    /// T5.1 v0: speculative fill with K drafts per verify round (head
    /// and tail — the tail's loop must speak the verify arm). 1-stream
    /// only; the CPU stage is the v0 posture (SpecStage = snapshot /
    /// restore). The drafter is repeat-last by default; `--drafter mtp`
    /// seats the trained BMTH head (T5.1d, Bench 677/917) from
    /// `--head PATH` — split-63 seat ONLY (Bench 917's basis curve).
    spec_k: Option<usize>,
    /// The spec drafter: `repeat` (default) or `mtp` (the trained head).
    drafter: String,
    /// The BMTH v1 artifact path for `--drafter mtp` (verified against
    /// its `.blake3` sidecar at boot; refuses loud on drift).
    head: Option<PathBuf>,
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
        plan_wire: None,
        answer_probe: false,
        link_bw: None,
        spec_k: None,
        drafter: "repeat".to_string(),
        head: None,
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
            "--spec-k" => a.spec_k = Some(val()?.parse().map_err(|e| format!("--spec-k: {e}"))?),
            "--drafter" => a.drafter = val()?,
            "--head" => a.head = Some(PathBuf::from(val()?)),
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
            "--plan-wire" => {
                a.plan_wire = Some(
                    val()?
                        .split(',')
                        .map(|t| {
                            t.trim()
                                .parse::<u32>()
                                .map_err(|e| format!("--plan-wire: {e}"))
                        })
                        .collect::<Result<_, _>>()?,
                )
            }
            "--answer-probe" => a.answer_probe = true,
            "--link-bw" => a.link_bw = Some(val()?.parse().map_err(|e| format!("--link-bw: {e}"))?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    match a.role.as_str() {
        "head" | "tail" => {}
        _ => return Err("--role head|tail required".into()),
    }
    if a.answer_probe {
        if a.role != "tail" {
            return Err("--answer-probe is a tail posture".into());
        }
        if a.plan_wire.is_some() || a.plan_report.is_some() {
            return Err(
                "--answer-probe composes with neither --plan-wire nor --plan-report".into(),
            );
        }
    }
    if let Some(cands) = &a.plan_wire {
        if a.role != "head" {
            return Err("--plan-wire is a head posture".into());
        }
        if a.plan_report.is_some() {
            return Err(
                "--plan-wire composes with neither --plan-report nor --answer-probe".into(),
            );
        }
        if cands.is_empty() {
            return Err("--plan-wire needs at least one candidate".into());
        }
        if cands.contains(&0) {
            return Err("--plan-wire candidates are strictly 1..n_layer (0 is not a split)".into());
        }
        if a.split == 0 {
            return Err(
                "--plan-wire needs --split (the provisional boot slice; the plan overrides)".into(),
            );
        }
    }
    if a.plan_report.is_some() && a.plan_wire.is_some() {
        return Err("--plan-report and --plan-wire are exclusive".into());
    }
    Ok(a)
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

/// std-equivalent address resolution (both IP literals and hostnames).
fn resolve_addr(addr: &str, what: &str) -> Result<std::net::SocketAddr, String> {
    addr.to_socket_addrs()
        .map_err(|e| format!("{what} {addr}: {e}"))?
        .next()
        .ok_or(format!("{what} {addr}: no address"))
}

/// The per-candidate scoped-load + timed-decode probe loop — ONE probe
/// implementation shared by `--plan-report`, the `--answer-probe` tail
/// and the `--plan-wire` head (the T2.5 landing's probe, never a
/// forked copy). `k` is the tail-side split index; the head's own
/// range is `[0, k)`, the tail's `[k, n_layer)`.
fn probe_candidates(
    a: &Args,
    identity: ArtifactIdentity,
    n_layer: u32,
    splits: &[u32],
    quiet: bool,
) -> Result<Vec<(u32, f64)>, String> {
    const PROBE_STEPS: usize = 8;
    const PROBE_ROUNDS: usize = 3;
    let head = a.role == "head";
    let mut rows = Vec::with_capacity(splits.len());
    for &k in splits {
        if k == 0 || k > n_layer {
            return Err(format!("probe candidate {k} outside 1..={n_layer}"));
        }
        let cand = match head {
            true => LayerRange::new(0, k),
            false => LayerRange::new(k, n_layer),
        };
        let (c2, w2) = load(&a.model, a.ctx, |c| stage_scope(cand, c.n_layer as u32))?;
        let mut st = build_stage(&a.backend, &c2, &w2, cand, identity)?;
        let spt =
            riir_infer_dist::planner::probe_decode_spt(st.as_mut(), 0, PROBE_STEPS, PROBE_ROUNDS)
                .map_err(|e| format!("probe split {k}: {e}"))?;
        if !quiet {
            eprintln!(
                "[plan] {} split {k}: {spt:.6} s/tok (decode, {PROBE_STEPS}x{PROBE_ROUNDS})",
                a.role
            );
        }
        rows.push((k, spt));
    }
    Ok(rows)
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
        let rows = probe_candidates(&a, identity, n_layer, splits, false)?;
        let cells: Vec<String> = rows.iter().map(|(k, s)| format!("[{k},{s:.6}]")).collect();
        println!(
            "{{\"host\":\"{}\",\"n_layer\":{},\"hidden\":{},\"rows\":[{}]}}",
            a.role,
            n_layer,
            config.n_embd,
            cells.join(",")
        );
        return Ok(());
    }

    // T2.5 probe EXCHANGE head posture: probe self, ask the tail over
    // the wire, MEASURE the RTT on the same link, compose the plan,
    // print it, exit clean. Candidates are strictly 2-stage
    // (1..n_layer — plan_split's domain; the 1-box reference row is a
    // --plan-report file convention, never a 2-stage candidate).
    if let Some(cands) = &a.plan_wire {
        for &k in cands {
            if k >= n_layer {
                return Err(format!("--plan-wire candidate {k} outside 1..{n_layer}"));
            }
        }
        let rows = probe_candidates(&a, identity, n_layer, cands, false)?;
        let own = riir_infer_dist::planner::HostProbe {
            host: "head".into(),
            rows,
            state_budget: None,
        };
        let addr = resolve_addr(
            a.connect
                .as_deref()
                .ok_or("--plan-wire needs --connect ADDR")?,
            "--connect",
        )?;
        let (sock, applied) =
            sockopts::dial(addr, &link_opts).map_err(|e| format!("connect {addr}: {e}"))?;
        eprintln!("{}", applied_line("head", applied));
        let mut link = TcpTransport::new(sock, key, Direction::Downstream)
            .map_err(|e| format!("socket: {e}"))?;
        let ctx = riir_infer_dist::probe::ExchangeCtx {
            own,
            n_layer,
            hidden: config.n_embd,
            bw_bps: a.link_bw,
            n_positions: a.ctx,
            // Conservative per-position figure (the attention layers'
            // KV row; the hybrid's GDN layers hold none). Inert at the
            // v0 posture — neither record carries a budget — and the
            // safe direction if one ever does.
            state_bytes_per_position_per_layer: riir_infer_dist::planner::state_bytes_per_position(
                riir_infer_core::types::kv_dim(&config),
                0,
            ),
            rtt_rounds: riir_infer_dist::probe::RTT_ROUNDS,
        };
        let out =
            riir_infer_dist::probe::run_probe_exchange(&mut link, cands, &ctx, &addr.to_string())
                .map_err(|e| format!("probe exchange: {e}"))?;
        link.send_frame(riir_infer_dist::transport::FrameKind::Shutdown, &[])
            .map_err(|e| format!("shutdown: {e}"))?;
        println!(
            "link: rtt {:.3} ms (median of {}), bw {} B/s ({})",
            out.rtt_s * 1e3,
            riir_infer_dist::probe::RTT_ROUNDS,
            out.bw_bps,
            match a.link_bw {
                Some(_) => "carried (--link-bw)",
                None => "measured, echo-ack bulk",
            }
        );
        println!(
            "tail record: {} rows · n_layer {n_layer} · hidden {} (cross-checked)",
            out.tail.rows.len(),
            config.n_embd
        );
        for (k, s) in &out.plan.table {
            println!("  split {k}: {s:.6} s/tok predicted");
        }
        println!("PLAN: --split {}", out.plan.k);
        return Ok(());
    }

    // T2.5 probe EXCHANGE tail posture: answer pings and probe
    // requests until Shutdown, exit clean — never serves. The boot
    // slice (the operator's --split) stays resident; the probes load
    // their own scoped candidates.
    if a.answer_probe {
        let addr = resolve_addr(
            a.listen
                .as_deref()
                .ok_or("--answer-probe needs --listen ADDR")?,
            "--listen",
        )?;
        let (l, applied) =
            sockopts::listen(addr, &link_opts).map_err(|e| format!("listen {addr}: {e}"))?;
        eprintln!("{}", applied_line("tail", applied));
        eprintln!(
            "[tail] probe-answer posture on {addr} (candidates 1..{n_layer}); serving NOT armed"
        );
        let (sock, peer) = l.accept().map_err(|e| format!("accept: {e}"))?;
        eprintln!("[tail] probe head connected from {peer}");
        let mut link = TcpTransport::new(sock, key, Direction::Upstream)
            .map_err(|e| format!("socket: {e}"))?;
        let t0 = Instant::now();
        let mut probe_one = |k: u32| -> std::io::Result<f64> {
            probe_candidates(&a, identity, n_layer, &[k], true)
                .map(|mut r| {
                    let (k, spt) = r.remove(0);
                    // The tail's own row on ITS stderr (the head prints
                    // the combined predictions; the Bench-915 residue:
                    // an operator watching the tail wants the raw lines
                    // too). The answer posture is a boot tool — stderr
                    // is its log, two lines per request cost nothing.
                    eprintln!("[tail] probe split {k}: {spt:.6} s/tok");
                    spt
                })
                .map_err(std::io::Error::other)
        };
        riir_infer_dist::probe::serve_probe_session(
            &mut link,
            n_layer,
            config.n_embd as u32,
            &mut probe_one,
        )
        .map_err(|e| format!("probe session: {e}"))?;
        eprintln!(
            "[tail] probe session answered in {:.1}s, clean shutdown",
            t0.elapsed().as_secs_f64()
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
            let addr = resolve_addr(
                a.listen.as_deref().ok_or("tail needs --listen ADDR")?,
                "--listen",
            )?;
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
            let steps = match (streams, a.spec_k) {
                (1, None) => run_tail(stage_refs[0], &mut link, |_, _| {}),
                (1, Some(k)) => {
                    // The verify arm needs SpecStage (snapshot/restore) —
                    // the dyn stage_refs cannot carry the bound, so spec
                    // mode seats its own CONCRETE CPU stage (the v0
                    // posture; the GPU rewind surface is a later rung).
                    if a.backend != "cpu" {
                        return Err(format!(
                            "--spec-k v0 is CPU-stage only (backend {} has no SpecStage impl)",
                            a.backend
                        ));
                    }
                    let mut stage =
                        TernaryCpuStage::new(config.clone(), weights.clone(), range, identity)?;
                    eprintln!("[tail] spec mode: verify arm armed (k={k})");
                    run_tail_spec(&mut stage, &mut link, |_, _| {})
                }
                (_, Some(_)) => {
                    return Err("--spec-k rides the single-stream loop only".into());
                }
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
            let addr = resolve_addr(
                a.connect.as_deref().ok_or("head needs --connect ADDR")?,
                "--connect",
            )?;
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
                // T2.4 fused batch step — OPT-IN since the 2-box wire
                // refutation (Bench 913): `RIIR_DIST_PAIR=1` arms it for
                // LOOPBACK rigs (one GPU — the shared queue's inter-chain
                // idle removal measured +39%, Bench 912). On the 2-box wire
                // the BatchN wave forfeits the head⇄tail GPU overlap that is
                // the pipeline's whole s=2 win (ABBA ×2: 46.9–52.6 armed vs
                // 99.2–108.5 incumbent aggregate decode window) — the
                // incumbent single-drain cadence is the serving default.
                pair_decode: matches!(
                    std::env::var("RIIR_DIST_PAIR").as_deref(),
                    Ok("1") | Ok("true") | Ok("on")
                ),
                // Issue 1147 B8: block-exact frames when the GPU stage's
                // ANE split is armed (its full frames are then whole blocks).
                prefill_block_exact: a.backend == "gpu" && gpu_ane_split_armed(),
                ..HeadOptions::default()
            };
            let (reports, wall) = match (streams, a.spec_k) {
                (1, None) => {
                    let r = run_head_with(stage_refs[0], &mut link, &prompt, a.max_new, opts)
                        .map_err(|e| e.to_string())?;
                    (vec![r], None)
                }
                (1, Some(k)) => {
                    // Same SpecStage seating as the tail arm: a concrete
                    // CPU stage (v0 posture), never the dyn box.
                    if a.backend != "cpu" {
                        return Err(format!(
                            "--spec-k v0 is CPU-stage only (backend {} has no SpecStage impl)",
                            a.backend
                        ));
                    }
                    let mut stage =
                        TernaryCpuStage::new(config.clone(), weights.clone(), range, identity)?;
                    // The drafter (T5.1d): repeat-last is the machinery
                    // baseline; `--drafter mtp` seats the trained BMTH
                    // head (Bench 677 artifact, sidecar-verified). Bench
                    // 917's basis curve: the head reads 0.68 per-slot at
                    // split 63 and 0.03 at the production split 32 — a
                    // split-63 boot prints the seat note, anything lower
                    // refuses (the seat is part of the contract).
                    let mut drafter: BinDrafter = match a.drafter.as_str() {
                        "repeat" => BinDrafter::Repeat(RepeatLastDrafter),
                        "mtp" => {
                            let head_path = a
                                .head
                                .as_ref()
                                .ok_or("--drafter mtp needs --head PATH (the BMTH v1 artifact)")?;
                            let split = a.split;
                            if split + 8 < config.n_layer as u32 {
                                return Err(format!(
                                    "--drafter mtp at split {split}: Bench 917's basis curve reads \
                                         0.03 per-slot at split 32 — the viable seat is split {} (head owns \
                                         all but the last layer); pass --split {}",
                                    config.n_layer - 1,
                                    config.n_layer - 1
                                ));
                            }
                            let head = BmthHead::load_verified(head_path)
                                .map_err(|e| format!("--head: {e}"))?;
                            let d = MtpHeadDrafter::seat(head, config.n_embd)
                                .map_err(|e| format!("--head: {e}"))?;
                            eprintln!(
                                "[head] mtp drafter seated (support {}, d {}) — Bench 917 seat: split {split}",
                                d.head().support(),
                                d.head().d()
                            );
                            BinDrafter::Mtp(d)
                        }
                        other => {
                            return Err(format!(
                                "--drafter {other}: known drafters are repeat | mtp"
                            ));
                        }
                    };
                    let r = run_head_spec(
                        &mut stage,
                        &mut link,
                        &prompt,
                        a.max_new,
                        opts,
                        &mut drafter,
                        k,
                    )
                    .map_err(|e| e.to_string())?;
                    (vec![r], None)
                }
                (_, Some(_)) => {
                    return Err("--spec-k rides the single-stream loop only".into());
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
                    // The decode-window aggregate — tokens past each
                    // stream's first over the slowest stream's first-to-
                    // last window — is the pipeline-rate metric the
                    // single-stream decode column compares against; the
                    // wall figure includes prefill + begins and dilutes it.
                    let dec_wall = reports
                        .iter()
                        .map(|r| r.decode)
                        .max()
                        .unwrap_or_default()
                        .as_secs_f64();
                    println!(
                        "aggregate: {} tokens in {:.3}s = {:.2} tok/s wall ({} streams) · decode window {:.2} tok/s",
                        total,
                        w.as_secs_f64(),
                        total as f64 / w.as_secs_f64().max(1e-9),
                        streams,
                        total as f64 / dec_wall.max(1e-9),
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
                let want = run_local_with_report(full.as_mut(), &prompt, a.max_new, opts);
                let wall = t.elapsed().as_secs_f64();
                let tok_s = a.max_new.saturating_sub(1) as f64 / wall;
                // The decode-only window + prefill beside the end-to-end
                // figure: the fits-one-box G2 row's comparator is the
                // DECODE rate (the same window the pipeline arm prints),
                // never the prefill-diluted end-to-end number.
                println!(
                    "1-box {} oracle: {tok_s:.2} tok/s end-to-end (incl. prefill) · decode {:.2} tok/s ({} steps) · prefill {:.3}s",
                    a.backend,
                    want.decode_tok_s(),
                    want.generated.len().saturating_sub(1),
                    want.prefill.as_secs_f64(),
                );
                for (i, r) in reports.iter().enumerate() {
                    match want.generated == r.generated {
                        true => println!("G1 PASS: stream {i} == 1-box stream"),
                        false => {
                            return Err(format!("G1 FAIL: stream {i} 1-box {:?}", want.generated));
                        }
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

/// Issue 1147 B8: is the GPU stage's ANE channel split armed (so the head
/// sends block-exact frames)?
#[cfg(feature = "gpu")]
fn gpu_ane_split_armed() -> bool {
    riir_infer_dist::stage_gpu::ane_split_block().is_some()
}

#[cfg(not(feature = "gpu"))]
fn gpu_ane_split_armed() -> bool {
    false
}
