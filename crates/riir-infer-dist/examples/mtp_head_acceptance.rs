//! Plan 623 T5.1d — the in-posture acceptance measurement for the trained
//! MTP-head drafter (`spec_mtp::MtpHeadDrafter`) over the REAL Bonsai-2
//! artifact, on one box, in-process 2-stage pair over loopback TCP.
//!
//! WHY THIS EXISTS (the honest-basis law): the head was fit on the folded
//! runtime's POST-FINAL residual (riir-train Bench 677: 0.7929 per-slot on
//! the frozen read split). The serving wire carries the HEAD STAGE's own
//! layer-window output — at split `s < n_layer` that is a mid-pipeline
//! activation the head never saw. This harness MEASURES what the head
//! reads in-posture instead of inheriting the fit-split number:
//!
//! - E[m] per verify round (the echo counts: d_0 == pending always
//!   accepts, so E[m] ≥ 1 whenever the machinery holds);
//! - per-slot rate beyond the echo — the number comparable to the fit
//!   split's 0.7929;
//! - decode tok/s for the MTP drafter vs repeat-last (the machinery
//!   baseline) vs the ordinary non-spec pair (the incumbent row).
//!
//! ```text
//! cargo run --release -p riir-infer-dist --features bonsai2_hadamard \
//!   --example mtp_head_acceptance -- \
//!   --model ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --head ../riir-train/data/bonsai_mtp/head/bonsai_mtp_head_v1.bin \
//!   --tokens ../riir-train/data/bonsai_mtp/tokens.read.tokens \
//!   --split 63 --prompt-tokens 768 --gen-tokens 192 --k 2
//! ```
//!
//! Reports a `PROVENANCE` line next to every number (box, load class) —
//! a latency number without its box state is not a measurement.

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_dist::runner::{HeadOptions, run_head_with};
use riir_infer_dist::spec_fill::{DraftSource, RepeatLastDrafter, run_head_spec};
use riir_infer_dist::spec_mtp::{BmthHead, MtpHeadDrafter};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

struct Args {
    model: PathBuf,
    head: PathBuf,
    tokens: PathBuf,
    split: u32,
    prompt_tokens: usize,
    gen_tokens: usize,
    k: usize,
    /// Dump the first run's generated tokens (u32-LE) — the free-stream
    /// diagnostic (repeat attractors, drift classes).
    dump_gen: Option<PathBuf>,
    /// Teacher-forced basis-isolation mode: feed TRUE tokens, score the
    /// head against the model's own argmax on the SAME head-stage row —
    /// the fit eval's exact pairing, the only delta being the row's basis
    /// (mid-pipeline at --split vs the post-final fit stream). No spec
    /// machinery, no snapshots — the pure basis effect.
    teacher_forced: Option<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"),
        head: PathBuf::from("../riir-train/data/bonsai_mtp/head/bonsai_mtp_head_v1.bin"),
        tokens: PathBuf::from("../riir-train/data/bonsai_mtp/tokens.read.tokens"),
        split: 63,
        prompt_tokens: 768,
        gen_tokens: 192,
        k: 2,
        dump_gen: None,
        teacher_forced: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => a.model = PathBuf::from(val()),
            "--head" => a.head = PathBuf::from(val()),
            "--tokens" => a.tokens = PathBuf::from(val()),
            "--split" => a.split = val().parse().expect("split u32"),
            "--prompt-tokens" => a.prompt_tokens = val().parse().expect("usize"),
            "--gen-tokens" => a.gen_tokens = val().parse().expect("usize"),
            "--k" => a.k = val().parse().expect("usize"),
            "--teacher-forced" => a.teacher_forced = Some(val().parse().expect("usize")),
            "--dump-gen" => a.dump_gen = Some(PathBuf::from(val())),
            "--help" | "-h" => {
                println!(
                    "--model P --head P --tokens P --split N --prompt-tokens N --gen-tokens N --k N [--teacher-forced N]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(a.k >= 1 && a.k <= 64, "--k sits in 1..=64");
    a
}

fn read_tokens(path: &std::path::Path, n: usize) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < n * 4 {
        return Err(format!(
            "{}: need {n} tokens, has {}",
            path.display(),
            bytes.len() / 4
        ));
    }
    Ok(bytes[..n * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

struct Pair {
    config: riir_infer_core::types::Config,
    weights: Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
}

fn load_pair(model: &std::path::Path) -> Result<Pair, String> {
    let (mut config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(
            model,
            |cfg| WeightScope::full(cfg.n_layer),
        )
        .map_err(|e| format!("load model: {e}"))?;
    // The Bench 917 posture (ctx 4096): the artifact now declares a 256k
    // context and the unbounded stage cache would allocate 256k KV rows
    // per layer (the Issue-929 class — >100 GiB of commit). The ctx
    // clamp mirrors dist_stage's `load()`.
    config.block_size = 4096;
    let id = ArtifactIdentity::of_gguf(model).map_err(|e| format!("identity: {e}"))?;
    Ok(Pair {
        config,
        weights: Arc::new(weights),
        id,
    })
}

/// Outcome of one pair run: the head report + the tail's step count.
struct PairRun {
    generated: Vec<u32>,
    decode: std::time::Duration,
    spec_rounds: u64,
    spec_accepted: u64,
    tail_steps: u64,
}

fn run_spec_pair<D>(
    pair: &Pair,
    split: u32,
    prompt: &[u32],
    max_new: usize,
    drafter: D,
    k: usize,
) -> Result<PairRun, String>
where
    D: DraftSource + Send + 'static,
{
    let n = pair.config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let (tc, tw) = (pair.config.clone(), pair.weights.clone());
    let tail_id = pair.id;
    let tail = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(split, n), tail_id).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey([1u8; 32]), Direction::Upstream).unwrap();
        riir_infer_dist::spec_fill::run_tail_spec(&mut stage, &mut link, |_, _| {})
    });
    let (hc, hw) = (pair.config.clone(), pair.weights.clone());
    let head_id = pair.id;
    let prompt_vec = prompt.to_vec();
    let head = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(hc, hw, LayerRange::new(0, split), head_id).unwrap();
        let sock = TcpStream::connect(addr).unwrap();
        let mut link = TcpTransport::new(sock, PeerKey([1u8; 32]), Direction::Downstream).unwrap();
        let mut d = drafter;
        run_head_spec(
            &mut stage,
            &mut link,
            &prompt_vec,
            max_new,
            HeadOptions::default(),
            &mut d,
            k,
        )
    });
    let head_report = head
        .join()
        .map_err(|_| "head thread panicked".to_string())?
        .map_err(|e| format!("head: {e}"))?;
    let tail_steps = tail
        .join()
        .map_err(|_| "tail thread panicked".to_string())?
        .map_err(|e| format!("tail: {e}"))?;
    Ok(PairRun {
        generated: head_report.generated,
        decode: head_report.decode,
        spec_rounds: head_report.spec_rounds,
        spec_accepted: head_report.spec_accepted,
        tail_steps: tail_steps as u64,
    })
}

fn run_ordinary_pair(
    pair: &Pair,
    split: u32,
    prompt: &[u32],
    max_new: usize,
) -> Result<PairRun, String> {
    let n = pair.config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let (tc, tw) = (pair.config.clone(), pair.weights.clone());
    let tail_id = pair.id;
    let tail = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(tc, tw, LayerRange::new(split, n), tail_id).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey([2u8; 32]), Direction::Upstream).unwrap();
        // The ordinary tail arm (the plain receive loop).
        riir_infer_dist::runner::run_tail(&mut stage, &mut link, |_, _| {})
    });
    let (hc, hw) = (pair.config.clone(), pair.weights.clone());
    let head_id = pair.id;
    let prompt_vec = prompt.to_vec();
    let head = thread::spawn(move || {
        let mut stage = TernaryCpuStage::new(hc, hw, LayerRange::new(0, split), head_id).unwrap();
        let sock = TcpStream::connect(addr).unwrap();
        let mut link = TcpTransport::new(sock, PeerKey([2u8; 32]), Direction::Downstream).unwrap();
        run_head_with(
            &mut stage,
            &mut link,
            &prompt_vec,
            max_new,
            HeadOptions::default(),
        )
    });
    let report = head
        .join()
        .map_err(|_| "head thread panicked".to_string())?
        .map_err(|e| format!("head: {e}"))?;
    let tail_steps = tail
        .join()
        .map_err(|_| "tail thread panicked".to_string())?
        .map_err(|e| format!("tail: {e}"))?;
    Ok(PairRun {
        generated: report.generated,
        decode: report.decode,
        spec_rounds: report.spec_rounds,
        spec_accepted: report.spec_accepted,
        tail_steps: tail_steps as u64,
    })
}

fn print_row(label: &str, run: &PairRun, k: usize) {
    let tok = run.generated.len().saturating_sub(1).max(1) as f64;
    let tok_s = tok / run.decode.as_secs_f64();
    let rounds = run.spec_rounds;
    let em = match rounds {
        0 => 0.0,
        _ => run.spec_accepted as f64 / rounds as f64,
    };
    // Per-slot beyond the echo: the accepted drafts past d_0 over the slots
    // that carried a real prediction — the fit-split-comparable number.
    let beyond = run.spec_accepted.saturating_sub(rounds);
    let slots = rounds * (k as u64).saturating_sub(1);
    let per_slot = match slots {
        0 => 0.0,
        _ => beyond as f64 / slots as f64,
    };
    println!(
        "{label:<28} tok/s {tok_s:7.3}  E[m] {em:.4}  per-slot>echo {per_slot:.4}  rounds {rounds}  committed {}  tail_steps {}",
        run.generated.len(),
        run.tail_steps
    );
}

fn main() {
    let args = parse_args();
    println!(
        "PROVENANCE: host {} cores {} — in-process pair, CPU stages, loopback TCP; release build",
        std::env::var("HOST_LABEL").unwrap_or_else(|_| whoami_fallback()),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    let t0 = Instant::now();
    let pair = load_pair(&args.model).unwrap_or_else(|e| {
        eprintln!("mtp_head_acceptance: {e}");
        std::process::exit(1);
    });
    let n = pair.config.n_layer as u32;
    println!(
        "# model {} loaded in {:.1}s — n_layer {n}, n_embd {}, split {}",
        args.model.display(),
        t0.elapsed().as_secs_f32(),
        pair.config.n_embd,
        args.split
    );
    if args.split == 0 || args.split >= n {
        eprintln!("mtp_head_acceptance: --split must sit in 1..{n}");
        std::process::exit(1);
    }
    let need = args.prompt_tokens;
    let tokens = read_tokens(&args.tokens, need).unwrap_or_else(|e| {
        eprintln!("mtp_head_acceptance: {e}");
        std::process::exit(1);
    });
    let prompt = &tokens[..args.prompt_tokens];

    // ── the trained head ──
    let t1 = Instant::now();
    let head = BmthHead::load_verified(&args.head).unwrap_or_else(|e| {
        eprintln!("mtp_head_acceptance: {e}");
        std::process::exit(1);
    });
    let hidden_dim = pair.config.n_embd;
    let drafter = MtpHeadDrafter::seat(head, hidden_dim).unwrap_or_else(|e| {
        eprintln!("mtp_head_acceptance: {e}");
        std::process::exit(1);
    });
    println!(
        "# head {} seated (d {}, support {}) in {:.1}s",
        args.head.display(),
        hidden_dim,
        drafter.head().support(),
        t1.elapsed().as_secs_f32()
    );

    println!(
        "# prompt {} tokens, gen {}, k {}",
        args.prompt_tokens, args.gen_tokens, args.k
    );

    // ── teacher-forced basis isolation (the fit-comparable number) ──
    if let Some(n_tok) = args.teacher_forced {
        let tokens = read_tokens(&args.tokens, n_tok).unwrap_or_else(|e| {
            eprintln!("mtp_head_acceptance: {e}");
            std::process::exit(1);
        });
        let n = pair.config.n_layer as u32;
        let mut stage_h = TernaryCpuStage::new(
            pair.config.clone(),
            pair.weights.clone(),
            LayerRange::new(0, args.split),
            pair.id,
        )
        .unwrap_or_else(|e| {
            eprintln!("mtp_head_acceptance: head stage: {e}");
            std::process::exit(1);
        });
        let mut stage_t = TernaryCpuStage::new(
            pair.config.clone(),
            pair.weights.clone(),
            LayerRange::new(args.split, n),
            pair.id,
        )
        .unwrap_or_else(|e| {
            eprintln!("mtp_head_acceptance: tail stage: {e}");
            std::process::exit(1);
        });
        let empty: [f32; 0] = [];
        let mut head_tf = BmthHead::load_verified(&args.head).unwrap_or_else(|e| {
            eprintln!("mtp_head_acceptance: {e}");
            std::process::exit(1);
        });
        if head_tf.d() != hidden_dim {
            eprintln!(
                "mtp_head_acceptance: head d {} != n_embd {hidden_dim}",
                head_tf.d()
            );
            std::process::exit(1);
        }
        let mut row = vec![0f32; hidden_dim];
        let mut hits = 0usize;
        let mut am_repeat = 0usize;
        let mut am_in_support = 0usize;
        let mut argmax_log: Vec<u8> = Vec::new();
        let t2 = Instant::now();
        let mut prev_am = 0u32;
        for (t, &tok) in tokens.iter().enumerate() {
            let out = stage_h.step(tok, t, &empty);
            row.copy_from_slice(out);
            let logits = stage_t.step(0, t, &row);
            let am = riir_infer_dist::argmax_first(logits);
            // The head predicts the token AFTER position t from row_t; the
            // tail's argmax on the SAME row is the model's own prediction —
            // one pairing, one comparison.
            let pred = head_tf.score(&row).unwrap_or_else(|e| {
                eprintln!("mtp_head_acceptance: score: {e}");
                std::process::exit(1);
            });
            hits += usize::from(pred == am);
            if t > 0 {
                am_repeat += usize::from(am == prev_am);
            }
            prev_am = am;
            am_in_support += usize::from(head_tf.contains(am));
            argmax_log.extend_from_slice(&am.to_le_bytes());
        }
        let rate = hits as f64 / tokens.len() as f64;
        let rep = if tokens.len() > 1 {
            am_repeat as f64 / (tokens.len() - 1) as f64
        } else {
            0.0
        };
        println!(
            "teacher-forced(split {}) per-slot {:.4} ({hits}/{} rows, {:.1}s) — fit-basis post-final was 0.7929; the delta is the BASIS effect (regime-clean)",
            args.split,
            rate,
            tokens.len(),
            t2.elapsed().as_secs_f32()
        );
        println!(
            "  calibration: model-argmax consecutive-repeat {:.4} · argmax-in-support {:.4} — the fit split's repeat read 0.4150",
            rep,
            am_in_support as f64 / tokens.len() as f64
        );
        if let Some(p) = &args.dump_gen {
            // The last 16 argmaxes (u32-LE) — cross-checks the free-run's
            // first generated tokens: same positions, same state path.
            let tail_n = 16.min(tokens.len());
            let lo = argmax_log.len() - tail_n * 4;
            std::fs::write(p, &argmax_log[lo..]).unwrap_or_else(|e| {
                eprintln!("mtp_head_acceptance: dump-gen: {e}");
                std::process::exit(1);
            });
            println!("# last {tail_n} argmaxes dumped to {}", p.display());
        }
        if args.gen_tokens == 0 {
            return;
        }
    }

    let mtp = run_spec_pair(&pair, args.split, prompt, args.gen_tokens, drafter, args.k)
        .unwrap_or_else(|e| {
            eprintln!("mtp_head_acceptance: {e}");
            std::process::exit(1);
        });
    if let Some(p) = &args.dump_gen {
        let bytes: Vec<u8> = mtp.generated.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(p, bytes).unwrap_or_else(|e| {
            eprintln!("mtp_head_acceptance: dump-gen: {e}");
            std::process::exit(1);
        });
        println!("# generated tokens dumped to {}", p.display());
    }
    print_row("mtp_head", &mtp, args.k);

    let rep = run_spec_pair(
        &pair,
        args.split,
        prompt,
        args.gen_tokens,
        RepeatLastDrafter,
        args.k,
    )
    .unwrap_or_else(|e| {
        eprintln!("mtp_head_acceptance: {e}");
        std::process::exit(1);
    });
    print_row("repeat_last (baseline)", &rep, args.k);

    let ord = run_ordinary_pair(&pair, args.split, prompt, args.gen_tokens).unwrap_or_else(|e| {
        eprintln!("mtp_head_acceptance: {e}");
        std::process::exit(1);
    });
    print_row("ordinary pair (incumbent)", &ord, args.k);
}

fn whoami_fallback() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
