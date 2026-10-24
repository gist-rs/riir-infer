//! Issue 1110 T2 — the β-reopen probe on the gemma2 d2f decode distribution.
//!
//! Measurement-only instrument (no production change): the decode loop's
//! confidence commits are an ABSORBING posture — once a position commits it
//! never reopens. This probe prices what a reopen capability would act on,
//! mirroring katgpt-rs Issue 956 T4's flip-move pricing on the lattice shape:
//!
//! 1. **Drift census** — an incumbent walk (byte-mirrors the disarmed
//!    `d2f_decode_gemma2` seams over public API only); at every denoise step,
//!    for every already-committed position, the mask-suppressed row is read:
//!    `q_u` (current prob of the committed token), top-1 (id, prob), and
//!    `rank_u` (strictly-greater count, logit space — monotone in prob).
//! 2. **Drift trigger** — the T2b η·headroom shape: the committed token lost
//!    top-1 AND `q_u < η · p_top1`. η is a knob; the full η-curve is computed
//!    post-hoc from the recorded rows. Counterfactuals attach at FIRST fire.
//! 3. **Flip-vs-resample win rate** — at each fire: flip = current top-1;
//!    K resamples = seeded draws from the fire-step distribution; incumbent =
//!    the committed token. Each variant is scored by a teacher-forced
//!    re-forward over the completed block (mask-suppressed log-prob of the
//!    variant token at the event position — any-order PLL). This measures the
//!    END-OF-DECODE endorsement, not the mid-walk context; disclosed.
//! 4. **Bounded JSD** at fire (commit-time topk-32 vs fire topk-32 over the
//!    union, renormalized) — the StabilityTracker-class signal read.
//!
//! Row-death condition (the issue): if flip-moves never beat resample-moves
//! on our shape, the row dies honestly at probe cost.
//!
//! Posture: CUDA-only (4090). Run DETACHED with a GPU-exclusive box:
//! `cargo run --release -p riir-infer-gpu --features gemma2_d2f,cuda_backend \
//!    --bin issue1110_reopen_probe -- --out <dir>`
//!
//! Everything is deterministic per seed: seeded fastrand per prompt and per
//! event draw-set; no HashMap iteration reaches the output (BTreeMap/Vec
//! only); sequential f32 folds (Issue-940 local-only posture).

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{GgufFile, load_gemma2_weights_gguf};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_gpu::GpuContext;
use riir_infer_gpu::gemma2_cubecl::GpuGemmaCubeCL;
use riir_infer_gpu::gemma2_d2f::{Gemma2D2fConfig, GpuGemmaCubeCLD2F};

/// Default model (the riir-train pin; restore via `data_backup.sh pull`).
const DEFAULT_GGUF: &str = r"E:\git\riir-train\data\gemma-2-2b-it-f16.gguf";

/// Top-k kept for the commit-time / fire-time snapshots (bounded-JSD basis).
const SNAPSHOT_K: usize = 32;

// ── Prompts (fixed census; arithmetic / factual / reasoning) ────────

const PROMPTS: &[&str] = &[
    "What is 17 * 24? Give just the number.",
    "What is 123 + 456? Give just the number.",
    "What is 15% of 80? Give just the number.",
    "A train travels 60 km per hour for 2.5 hours. How many km does it travel? Give just the number.",
    "What is the capital of France? Answer in one word.",
    "Who wrote the play Hamlet? Answer with the author's name only.",
    "What chemical element has the symbol Fe? Answer in one word.",
    "What year did the Berlin Wall fall? Give just the year.",
    "Name three primary colors, comma separated.",
    "Complete: The opposite of hot is ___ and the opposite of big is ___.",
    "Rearrange the letters of the word 'listen' to form another common English word. One word only.",
    "What is the next number in the sequence 2, 4, 8, 16, ___? Give just the number.",
];

// ── Records ─────────────────────────────────────────────────────────

/// One census observation: a committed position's row at a later step.
#[derive(Clone, Copy)]
struct ObsRow {
    /// Observation step (the forward that produced the row).
    step: usize,
    /// Block slot (`pos - block_start`).
    slot: usize,
    /// Current prob of the committed token.
    q_u: f32,
    /// Current top-1 token id.
    top1: u32,
    /// Current top-1 prob.
    p_top1: f32,
    /// Rank of the committed token (strictly-greater count).
    rank_u: usize,
}

impl ObsRow {
    fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "step": self.step, "slot": self.slot, "q_u": self.q_u,
            "top1": self.top1, "p_top1": self.p_top1, "rank_u": self.rank_u,
        })
    }
}

/// A commit-time snapshot (the census anchor).
#[derive(Clone)]
struct CommitRec {
    slot: usize,
    step: usize,
    token: u32,
    p_at_commit: f32,
    /// Top-k (id, prob) at the commit step — the bounded-JSD baseline.
    topk: Vec<(u32, f32)>,
}

impl CommitRec {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "slot": self.slot, "step": self.step, "token": self.token,
            "p_at_commit": self.p_at_commit,
            "topk": self.topk.iter().map(|(t, p)| serde_json::json!([t, p])).collect::<Vec<_>>(),
        })
    }
}

/// A first-fire event with its counterfactual scoring.
#[derive(Clone)]
struct FireEvent {
    slot: usize,
    token: u32,
    commit_step: usize,
    p_at_commit: f32,
    fire_step: usize,
    /// Prob of the committed token at fire.
    q_u: f32,
    /// Flip candidate (top-1 at fire).
    flip: u32,
    p_flip: f32,
    rank_u: usize,
    /// Bounded JSD (nats, ≤ ln 2) between commit and fire topk snapshots.
    jsd: f32,
    /// PLL (mask-suppressed log-prob) of the incumbent token, shared scoring.
    pll_u: f32,
    pll_flip: f32,
    /// PLL of each resample draw.
    pll_resamples: Vec<f32>,
    flip_beats_incumbent: bool,
    /// Fraction of draws the flip beat.
    flip_beats_resample_frac: f32,
}

impl FireEvent {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "slot": self.slot, "token": self.token,
            "commit_step": self.commit_step, "p_at_commit": self.p_at_commit,
            "fire_step": self.fire_step, "q_u": self.q_u,
            "flip": self.flip, "p_flip": self.p_flip, "rank_u": self.rank_u,
            "jsd": self.jsd, "pll_u": self.pll_u, "pll_flip": self.pll_flip,
            "pll_resamples": self.pll_resamples,
            "flip_beats_incumbent": self.flip_beats_incumbent,
            "flip_beats_resample_frac": self.flip_beats_resample_frac,
        })
    }
}

#[derive(Clone)]
struct PromptReport {
    idx: usize,
    prompt: String,
    n_prompt_tokens: usize,
    steps_used: usize,
    converged: bool,
    n_committed: usize,
    commits: Vec<CommitRec>,
    obs: Vec<ObsRow>,
    /// First-fire events at the run's η (with counterfactuals scored).
    events: Vec<FireEvent>,
    /// First-fire count per η (post-hoc curve from the obs rows).
    eta_fire_curve: BTreeF32,
    walk_forwards: usize,
    scoring_forwards: usize,
    /// Decoded block text (the incumbent's committed tokens).
    decoded_block: String,
}

impl PromptReport {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "idx": self.idx,
            "prompt": self.prompt,
            "n_prompt_tokens": self.n_prompt_tokens,
            "steps_used": self.steps_used,
            "converged": self.converged,
            "n_committed": self.n_committed,
            "commits": self.commits.iter().map(CommitRec::to_json).collect::<Vec<_>>(),
            "obs": self.obs.iter().map(|&r| r.to_json()).collect::<Vec<_>>(),
            "events": self.events.iter().map(FireEvent::to_json).collect::<Vec<_>>(),
            "eta_fire_curve": self.eta_fire_curve.0,
            "walk_forwards": self.walk_forwards,
            "scoring_forwards": self.scoring_forwards,
            "decoded_block": self.decoded_block,
        })
    }
}

/// Ordered f32 → usize pairs that serialize stably (never a HashMap).
#[derive(Clone)]
struct BTreeF32(Vec<(f32, usize)>);

impl BTreeF32 {
    fn from_pairs(mut pairs: Vec<(f32, usize)>) -> Self {
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self(pairs)
    }
}

// ── Pure helpers (unit-tested below, no GPU) ────────────────────────

/// In-place softmax over an already-mask-suppressed row. Sequential f32 fold
/// (Issue-940 local-only posture); max-subtract for stability.
fn softmax_in_place(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    let mut z = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        z += *v;
    }
    if z > 0.0 {
        for v in row.iter_mut() {
            *v /= z;
        }
    }
}

/// Mask-suppressed softmax of a logits row into `out` (reused scratch).
fn probs_of(logits: &[f32], mask_token_id: usize, out: &mut Vec<f32>) {
    out.clear();
    out.extend_from_slice(logits);
    if mask_token_id < out.len() {
        out[mask_token_id] = f32::NEG_INFINITY;
    }
    softmax_in_place(out);
}

/// Top-k (id, prob) of a PROBABILITY row, descending; ties by first index.
/// Sorted-insert into a bounded buffer — O(n·k) worst, calls are rare.
fn topk_of(probs: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut buf: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, &p) in probs.iter().enumerate() {
        if buf.len() == k && p <= buf[k - 1].1 {
            continue;
        }
        let pos = buf.partition_point(|&(_, bp)| bp >= p);
        buf.insert(pos, (i as u32, p));
        if buf.len() > k {
            buf.pop();
        }
    }
    buf
}

/// Bounded JSD (nats, ≤ ln 2) between two top-k (id, prob) snapshots over
/// the union, each side renormalized on the union. Deterministic: BTreeMap.
fn bounded_jsd(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    use std::collections::BTreeMap;
    let mut m: BTreeMap<u32, (f64, f64)> = BTreeMap::new();
    let mut za = 0.0f64;
    let mut zb = 0.0f64;
    for &(t, p) in a {
        let e = m.entry(t).or_insert((0.0, 0.0));
        e.0 += p as f64;
        za += p as f64;
    }
    for &(t, p) in b {
        let e = m.entry(t).or_insert((0.0, 0.0));
        e.1 += p as f64;
        zb += p as f64;
    }
    if za <= 0.0 || zb <= 0.0 {
        return 0.0;
    }
    let mut jsd = 0.0f64;
    for (_, (pa, pb)) in m {
        let pa = pa / za;
        let pb = pb / zb;
        let mm = 0.5 * (pa + pb);
        if pa > 0.0 {
            jsd += 0.5 * pa * (pa / mm).ln();
        }
        if pb > 0.0 {
            jsd += 0.5 * pb * (pb / mm).ln();
        }
    }
    // f64→f32 clamp: tiny negatives from fp error are noise.
    jsd.clamp(0.0, std::f64::consts::LN_2) as f32
}

/// The drift trigger: committed token lost top-1 AND its prob collapsed
/// below an η fraction of the current leader's (η < 1 — for η ≥ 1 the
/// condition is vacuous: q_u < p_top1 ≤ η·p_top1 always holds once the top
/// is lost, and the sweep would be flat). The η sweep runs the
/// collapse-depth axis; deeper collapse = rarer fires.
fn fires(eta: f32, q_u: f32, p_top1: f32, top1: u32, u: u32) -> bool {
    top1 != u && q_u < eta * p_top1
}

/// One seeded draw from a probability row (cumulative walk, first-index ties).
fn draw_from(probs: &[f32], rng: &mut fastrand::Rng) -> usize {
    let x = rng.f32() * probs_last(probs);
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if x < acc {
            return i;
        }
    }
    probs.len() - 1
}

fn probs_last(probs: &[f32]) -> f32 {
    let mut z = 0.0f32;
    for &p in probs {
        z += p;
    }
    z
}

/// Mask-suppressed log-prob of `tok` under a logits row.
fn pll_of(logits: &[f32], tok: usize, mask_token_id: usize) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if i != mask_token_id && v > m {
            m = v;
        }
    }
    let mut z = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if i != mask_token_id {
            z += (v - m).exp();
        }
    }
    let lz = z.ln();
    (logits[tok] - m - lz).clamp(-32.0, 0.0)
}

// ── Box state (the PROVENANCE line) ─────────────────────────────────

fn nvidia_smi(args: &[&str]) -> Option<String> {
    std::process::Command::new("nvidia-smi")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn capture_provenance() -> serde_json::Value {
    let query = [
        "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader",
    ];
    let gpu_line = nvidia_smi(&query).unwrap_or_default();
    let compute = nvidia_smi(&[
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ])
    .unwrap_or_else(|| "n/a".into());
    serde_json::json!({
        "gpu_csv": gpu_line,
        "compute_apps": compute,
        "gpu_exclusive": compute.trim().is_empty() || compute.trim() == "n/a",
        "captured_at_unix": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    })
}

// ── The instrumented incumbent walk ─────────────────────────────────

struct WalkOut {
    tokens: Vec<usize>,
    block_start: usize,
    seq_len: usize,
    steps_used: usize,
    converged: bool,
    commits: Vec<CommitRec>,
    obs: Vec<ObsRow>,
    /// First-fire event per slot at `eta`, with the fire-step probs row.
    fires: Vec<(usize, usize, Vec<f32>)>, // (slot, fire_step, fire_probs)
    fire_rows: Vec<ObsRow>,
}

#[allow(clippy::too_many_arguments)]
fn instrumented_walk(
    gpu: &mut GpuGemmaCubeCLD2F,
    prompt: &[usize],
    mask_token_id: usize,
    decode_config: &Gemma2D2fConfig,
    eta: f32,
    rng: &mut fastrand::Rng,
    probs_scratch: &mut Vec<f32>,
) -> WalkOut {
    let block_size = decode_config.block_size;
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let temperature = decode_config.temperature;
    let prompt_len_raw = prompt.len();

    // Issue 695 H7 prompt-tail clamp, mirrored from d2f_decode_gemma2.
    let max_seq = gpu.config().block_size;
    let prompt_tail_start = prompt_len_raw.saturating_sub(max_seq.saturating_sub(block_size));
    let mut tokens: Vec<usize> = prompt[prompt_tail_start..].to_vec();
    tokens.extend(std::iter::repeat_n(mask_token_id, block_size));
    let seq_len = tokens.len();
    let block_start = prompt_len_raw - prompt_tail_start;
    assert!(
        seq_len <= max_seq,
        "seq_len {seq_len} exceeds max {max_seq}"
    );

    let mut masked: Vec<bool> = vec![false; seq_len];
    masked[block_start..seq_len].fill(true);

    // slot → CommitRec (dense by slot).
    let mut commits: Vec<Option<CommitRec>> = vec![None; block_size];
    let mut obs: Vec<ObsRow> = Vec::new();
    let mut fire_events: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    let mut fire_rows: Vec<ObsRow> = Vec::new();
    let mut fired_slot: Vec<bool> = vec![false; block_size];

    let mut step_confidence_history: Vec<f32> = Vec::with_capacity(max_steps);
    let mut converged = false;

    for step in 0..max_steps {
        let all_logits = gpu.block_causal_forward(&tokens, block_start);

        // Census: every ALREADY-committed position's current row.
        for pos in block_start..seq_len {
            if masked[pos] {
                continue;
            }
            let slot = pos - block_start;
            let Some(rec) = &commits[slot] else { continue };
            probs_of(&all_logits[pos], mask_token_id, probs_scratch);
            let q_u = probs_scratch[rec.token as usize];
            // Top-1 + rank in one extra pair of scans over the probs row.
            let mut top1 = 0usize;
            let mut p_top1 = -1.0f32;
            for (i, &p) in probs_scratch.iter().enumerate() {
                if p > p_top1 {
                    p_top1 = p;
                    top1 = i;
                }
            }
            let u_logit = all_logits[pos][rec.token as usize];
            let mut rank_u = 0usize;
            for (i, &l) in all_logits[pos].iter().enumerate() {
                if i != mask_token_id && l > u_logit {
                    rank_u += 1;
                }
            }
            let row = ObsRow {
                step,
                slot,
                q_u,
                top1: top1 as u32,
                p_top1,
                rank_u,
            };
            let is_fire = !fired_slot[slot] && fires(eta, q_u, p_top1, top1 as u32, rec.token);
            if is_fire {
                fired_slot[slot] = true;
                fire_events.push((slot, step, probs_scratch.clone()));
                fire_rows.push(row);
            }
            obs.push(row);
        }

        // Accept walk — the incumbent seams, mirrored byte-for-byte.
        let mut n_unmasked = 0usize;
        for pos in block_start..seq_len {
            if !masked[pos] {
                n_unmasked += 1;
                continue;
            }
            let logits = &all_logits[pos];
            // sample_with_confidence, mirrored exactly: mask-suppress, scale
            // by temperature, softmax ONCE, confidence = max, cumulative draw.
            probs_scratch.clear();
            probs_scratch.extend_from_slice(logits);
            if mask_token_id < probs_scratch.len() {
                probs_scratch[mask_token_id] = f32::NEG_INFINITY;
            }
            if temperature > 0.0 && temperature != 1.0 {
                let inv = 1.0 / temperature;
                for p in probs_scratch.iter_mut() {
                    *p *= inv;
                }
            }
            softmax_in_place(probs_scratch);
            let confidence = probs_scratch.iter().copied().fold(0.0f32, f32::max);
            let sampled = draw_from(probs_scratch, rng);
            if confidence >= tau_conf && sampled != mask_token_id {
                tokens[pos] = sampled;
                masked[pos] = false;
                n_unmasked += 1;
                let slot = pos - block_start;
                commits[slot] = Some(CommitRec {
                    slot,
                    step,
                    token: sampled as u32,
                    p_at_commit: confidence,
                    topk: topk_of(probs_scratch, SNAPSHOT_K),
                });
            }
        }

        step_confidence_history.push(n_unmasked as f32 / block_size as f32);
        if (block_start..seq_len).all(|pos| !masked[pos]) {
            converged = true;
            break;
        }
    }

    WalkOut {
        tokens,
        block_start,
        seq_len,
        steps_used: step_confidence_history.len(),
        converged,
        commits: commits.into_iter().flatten().collect(),
        obs,
        fires: fire_events,
        fire_rows,
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gguf_path =
        PathBuf::from(std::env::var("GEMMA_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string()));
    let mut out_dir = PathBuf::from(".benchmarks/1110_reopen_probe");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_dir = PathBuf::from(&args[i]);
            }
            "--gguf" => {
                i += 1;
                gguf_path = PathBuf::from(&args[i]);
            }
            other => {
                eprintln!("unknown arg {other}; usage: [--out DIR] [--gguf PATH]");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let eta: f32 = std::env::var("D2F_PROBE_ETA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    let k_draws: usize = std::env::var("D2F_PROBE_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let max_prompts: usize = std::env::var("D2F_PROBE_MAX_PROMPTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PROMPTS.len());
    let seed: u64 = std::env::var("D2F_PROBE_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1110);

    std::fs::create_dir_all(&out_dir).expect("create out dir");

    println!("== Issue 1110 T2 reopen probe ==");
    println!("gguf: {}", gguf_path.display());
    let provenance = capture_provenance();
    println!(
        "PROVENANCE: gpu_exclusive={} gpu={} captured_at={}",
        provenance["gpu_exclusive"], provenance["gpu_csv"], provenance["captured_at_unix"]
    );

    // ── Load (tokenizer first: the 5.2 GB GgufFile drops before the f32
    // weights + GPU buffers arrive — keeps the RAM peak flat) ──
    let t0 = std::time::Instant::now();
    let tok_gguf = GgufFile::open(&gguf_path).expect("open gguf for tokenizer");
    let tok = SentencePieceGgufTokenizer::from_gguf(&tok_gguf).expect("sentencepiece tokenizer");
    drop(tok_gguf);
    let (config, weights) = load_gemma2_weights_gguf(&gguf_path).expect("load gemma2 f32 weights");
    println!(
        "weights loaded: {} layers, vocab {}, head_dim {} ({} s)",
        config.n_layer,
        config.vocab_size,
        config.head_dim,
        t0.elapsed().as_secs()
    );
    // Gemma 2 IT user-turn template (the SentencePieceTokenizer lane's shape;
    // SP recognizes <start_of_turn>/<end_of_turn> as single tokens).
    let encode_chat = |text: &str| -> Vec<usize> {
        let mut t = String::with_capacity(text.len() + 64);
        t.push_str("<start_of_turn>user\n");
        t.push_str(text);
        t.push_str("<end_of_turn>\n<start_of_turn>model\n");
        tok.encode_with_bos(&t)
    };

    // ── GPU + D2F (f16 GPU buffers — the lane's calibrated posture) ──
    let ctx = GpuContext::new().expect("GpuContext");
    let client = ctx.cubecl_client();
    let decode_config = Gemma2D2fConfig::quality(); // block 16, steps 12, tau 0.8
    let gpu_gemma = GpuGemmaCubeCL::new_f16(client, &weights, &config);
    let mut d2f = GpuGemmaCubeCLD2F::from_gpu_gemma(gpu_gemma, decode_config);
    println!(
        "d2f ready: block {} steps {} tau {}",
        decode_config.block_size, decode_config.denoise_steps, decode_config.confidence_threshold
    );

    let mask_token_id: usize = std::env::var("D2F_PROBE_MASK_ID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0); // <pad> — the lane's mask convention
    let vocab = config.vocab_size;
    let mut probs_scratch: Vec<f32> = Vec::with_capacity(vocab);

    let mut reports: Vec<PromptReport> = Vec::new();
    let mut total_forwards = 0usize;

    for (idx, text) in PROMPTS.iter().enumerate().take(max_prompts) {
        let prompt_tokens = encode_chat(text);
        let mut rng =
            fastrand::Rng::with_seed(seed ^ (idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
        println!(
            "\n-- prompt {idx}: {:?} ({} tokens)",
            text,
            prompt_tokens.len()
        );

        let walk = instrumented_walk(
            &mut d2f,
            &prompt_tokens,
            mask_token_id,
            &decode_config,
            eta,
            &mut rng,
            &mut probs_scratch,
        );
        let walk_forwards = walk.steps_used;
        println!(
            "walk: {} steps, converged {}, committed {}/{}",
            walk.steps_used,
            walk.converged,
            walk.commits.len(),
            decode_config.block_size
        );

        // ── Counterfactual scoring at the first-fire events ──
        let mut events: Vec<FireEvent> = Vec::new();
        let mut scoring_forwards = 0usize;
        if !walk.fires.is_empty() {
            // Shared incumbent scoring forward over the completed block.
            let score_logits = d2f.block_causal_forward(&walk.tokens, walk.block_start);
            scoring_forwards += 1;
            for fi in 0..walk.fires.len() {
                let (slot, fire_step) = (walk.fires[fi].0, walk.fires[fi].1);
                let fire_probs = &walk.fires[fi].2;
                let pos = walk.block_start + slot;
                let rec = match walk.commits.iter().find(|c| c.slot == slot) {
                    Some(c) => c.clone(),
                    None => continue,
                };
                let fire_row = walk.fire_rows[fi];
                let flip = fire_row.top1 as usize;
                let pll_u = pll_of(&score_logits[pos], rec.token as usize, mask_token_id);

                // The flip variant needs its OWN forward: the block differs
                // from the incumbent at `pos`.
                let mut flip_variant = walk.tokens.clone();
                flip_variant[pos] = flip;
                let fl_logits = d2f.block_causal_forward(&flip_variant, walk.block_start);
                let pll_flip = pll_of(&fl_logits[pos], flip, mask_token_id);
                scoring_forwards += 1;

                // K seeded resample draws from the fire-step distribution.
                let mut draw_rng = fastrand::Rng::with_seed(
                    seed ^ (0x5DEECE66D_u64).wrapping_mul((idx * 1000 + fi + 1) as u64),
                );
                let mut pll_resamples = Vec::with_capacity(k_draws);
                for _ in 0..k_draws {
                    let r = draw_from(fire_probs, &mut draw_rng);
                    if r == rec.token as usize {
                        pll_resamples.push(pll_u);
                        continue; // the draw IS the incumbent — no variant, no forward
                    }
                    let mut variant = walk.tokens.clone();
                    variant[pos] = r;
                    let vl = d2f.block_causal_forward(&variant, walk.block_start);
                    pll_resamples.push(pll_of(&vl[pos], r, mask_token_id));
                    scoring_forwards += 1;
                }
                let beats = pll_resamples.iter().filter(|&&p| pll_flip > p).count();
                events.push(FireEvent {
                    slot,
                    token: rec.token,
                    commit_step: rec.step,
                    p_at_commit: rec.p_at_commit,
                    fire_step,
                    q_u: fire_row.q_u,
                    flip: flip as u32,
                    p_flip: fire_row.p_top1,
                    rank_u: fire_row.rank_u,
                    jsd: bounded_jsd(&rec.topk, &topk_of(fire_probs, SNAPSHOT_K)),
                    pll_u,
                    pll_flip,
                    flip_beats_incumbent: pll_flip > pll_u,
                    pll_resamples,
                    flip_beats_resample_frac: if k_draws > 0 {
                        beats as f32 / k_draws as f32
                    } else {
                        0.0
                    },
                });
            }
            println!(
                "events(η={eta}): {} fired; scoring forwards {}",
                events.len(),
                scoring_forwards
            );
        }

        // Post-hoc η-curve from the recorded rows (first fire per slot per η;
        // `rank_u >= 1` encodes `top1 != u` — the top-1 is always strictly
        // greater when the incumbent lost it).
        let eta_fire_curve = eta_curve(&walk.obs);

        let decoded_block = tok.decode(&walk.tokens[walk.block_start..walk.seq_len]);
        println!("block: {:?}", decoded_block.trim());

        total_forwards += walk_forwards + scoring_forwards;
        reports.push(PromptReport {
            idx,
            prompt: (*text).to_string(),
            n_prompt_tokens: prompt_tokens.len(),
            steps_used: walk.steps_used,
            converged: walk.converged,
            n_committed: walk.commits.len(),
            commits: walk.commits,
            obs: walk.obs,
            events,
            eta_fire_curve,
            walk_forwards,
            scoring_forwards,
            decoded_block: decoded_block.trim().to_string(),
        });
    }

    // ── Pooled read ──
    let pooled = pooled_report(&reports, eta, k_draws);
    println!("\n== pooled ==");
    println!("{}", serde_json::to_string_pretty(&pooled).unwrap());

    let doc = serde_json::json!({
        "issue": "riir-infer 1110 T2",
        "provenance": provenance,
        "config": {
            "gguf": gguf_path.display().to_string(),
            "block_size": decode_config.block_size,
            "denoise_steps": decode_config.denoise_steps,
            "tau": decode_config.confidence_threshold,
            "temperature": decode_config.temperature,
            "eta": eta,
            "k_draws": k_draws,
            "seed": seed,
            "mask_token_id": mask_token_id,
            "snapshot_k": SNAPSHOT_K,
            "posture": "gemma2_d2f,cuda_backend (f16 GPU buffers, CubeCL CUDA)",
        },
        "total_forwards": total_forwards,
        "prompts": reports.iter().map(PromptReport::to_json).collect::<Vec<_>>(),
        "pooled": pooled,
    });
    let out_path = out_dir.join("reopen_probe.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap()).expect("write json");
    println!("\nwrote {}", out_path.display());
}

/// First-fire count per η from recorded obs rows (post-hoc η-curve).
/// A row fires for η iff `rank_u >= 1 && q_u < η·p_top1`; per-slot per-η
/// first-fire only.
fn eta_curve(obs: &[ObsRow]) -> BTreeF32 {
    const ETAS: [f32; 5] = [0.05, 0.1, 0.25, 0.5, 0.75];
    let mut counts = vec![0usize; ETAS.len()];
    let mut fired: std::collections::BTreeMap<usize, [bool; ETAS.len()]> =
        std::collections::BTreeMap::new();
    for r in obs {
        let flags = fired.entry(r.slot).or_insert([false; ETAS.len()]);
        for (ei, &eta) in ETAS.iter().enumerate() {
            if !flags[ei] && r.rank_u >= 1 && r.q_u < eta * r.p_top1 {
                counts[ei] += 1;
                flags[ei] = true;
            }
        }
    }
    BTreeF32::from_pairs(ETAS.iter().copied().zip(counts).collect())
}

/// Pooled stats over prompts.
fn pooled_report(reports: &[PromptReport], eta: f32, k_draws: usize) -> serde_json::Value {
    let n_prompts = reports.len();
    let n_converged = reports.iter().filter(|r| r.converged).count();
    let n_committed: usize = reports.iter().map(|r| r.n_committed).sum();
    let n_obs: usize = reports.iter().map(|r| r.obs.len()).sum();
    let n_events: usize = reports.iter().map(|r| r.events.len()).sum();
    let alarm_prompts = reports.iter().filter(|r| !r.events.is_empty()).count();

    let mut jsds: Vec<f32> = reports
        .iter()
        .flat_map(|r| r.events.iter().map(|e| e.jsd))
        .collect();
    let mut ranks: Vec<usize> = reports
        .iter()
        .flat_map(|r| r.events.iter().map(|e| e.rank_u))
        .collect();
    ranks.sort_unstable();

    let flip_gt_inc: usize = reports
        .iter()
        .flat_map(|r| r.events.iter())
        .filter(|e| e.flip_beats_incumbent)
        .count();
    let mean_pll_flip_delta: f64 = if n_events > 0 {
        reports
            .iter()
            .flat_map(|r| r.events.iter())
            .map(|e| (e.pll_flip - e.pll_u) as f64)
            .sum::<f64>()
            / n_events as f64
    } else {
        0.0
    };
    let draws: usize = reports
        .iter()
        .flat_map(|r| r.events.iter())
        .map(|e| e.pll_resamples.len())
        .sum();
    let flip_gt_res: usize = reports
        .iter()
        .flat_map(|r| r.events.iter())
        .map(|e| (e.flip_beats_resample_frac * e.pll_resamples.len() as f32).round() as usize)
        .sum();
    let mean_pll_res_delta: f64 = if n_events > 0 {
        let mut acc = 0.0f64;
        for r in reports {
            for e in &r.events {
                let n = e.pll_resamples.len();
                if n > 0 {
                    acc += e
                        .pll_resamples
                        .iter()
                        .map(|&p| (e.pll_flip - p) as f64)
                        .sum::<f64>()
                        / n as f64;
                }
            }
        }
        acc / n_events as f64
    } else {
        0.0
    };

    serde_json::json!({
        "eta": eta,
        "k_draws": k_draws,
        "n_prompts": n_prompts,
        "n_converged": n_converged,
        "n_committed_positions": n_committed,
        "n_obs_rows": n_obs,
        "fire_events_at_eta": n_events,
        "fire_rate_at_eta": if n_obs > 0 { n_events as f64 / n_obs as f64 } else { 0.0 },
        "prompt_alarm_rate_at_eta": if n_prompts > 0 { alarm_prompts as f64 / n_prompts as f64 } else { 0.0 },
        "jsd_at_fire": {
            "median": median_of(&mut jsds),
            "max": jsds.iter().copied().fold(0.0f32, f32::max),
        },
        "rank_u_at_fire": {
            "median": ranks.get(ranks.len() / 2).copied().unwrap_or(0),
            "max": ranks.last().copied().unwrap_or(0),
        },
        "win": {
            "n_events": n_events,
            "p_flip_gt_incumbent": if n_events > 0 { flip_gt_inc as f64 / n_events as f64 } else { 0.0 },
            "mean_pll_delta_flip_minus_u": mean_pll_flip_delta,
            "n_resample_draws": draws,
            "p_flip_gt_resample": if draws > 0 { flip_gt_res as f64 / draws as f64 } else { 0.0 },
            "mean_pll_delta_flip_minus_resample": mean_pll_res_delta,
        },
    })
}

/// Median as a named order statistic (event counts are small — no percentile
/// index claims; the max is quoted beside it and `n_events` sits in the same
/// JSON object).
fn median_of(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

// ── Tests (pure helpers; feature posture gemma2_d2f, no GPU) ────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_is_monotone() {
        let mut r = vec![0.5f32, -1.0, 2.0, 0.0];
        softmax_in_place(&mut r);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(r[2] > r[0] && r[0] > r[3] && r[3] > r[1]);
    }

    #[test]
    fn topk_is_descending_and_first_index_on_ties() {
        let p = vec![0.1f32, 0.4, 0.4, 0.05, 0.05];
        let k = topk_of(&p, 3);
        assert_eq!(k.len(), 3);
        assert_eq!(k[0], (1, 0.4));
        assert_eq!(k[1], (2, 0.4));
        assert!(k[2].1 < 0.4);
        for w in k.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn jsd_zero_on_identical_and_bounded_by_ln2() {
        let a = vec![(1u32, 0.6f32), (2, 0.4)];
        assert_eq!(bounded_jsd(&a, &a), 0.0);
        let b = vec![(3u32, 1.0f32)];
        let d = bounded_jsd(&a, &b);
        assert!(d > 0.0 && d <= std::f32::consts::LN_2 + 1e-6);
    }

    #[test]
    fn trigger_requires_both_top1_loss_and_headroom() {
        // Same top-1 → never fires, however small q_u.
        assert!(!fires(0.5, 0.001, 0.9, 5, 5));
        // Top-1 lost + deep collapse → fires.
        assert!(fires(0.5, 0.01, 0.9, 7, 5));
        // Top-1 lost but prob still near the leader's → no fire.
        assert!(!fires(0.5, 0.45, 0.5, 7, 5));
        // The vacuous-η law at η = 1 (valid distribution: u lost the top so
        // q_u < p_top1, hence q_u < 1·p_top1 always fires — the shipped
        // sweep stays below 1 so the axis has teeth).
        assert!(fires(1.0, 0.45, 0.5, 7, 5));
    }

    #[test]
    fn pll_prefers_the_modal_token() {
        let logits = vec![0.0f32; 8];
        let mut sharp = logits.clone();
        sharp[3] = 10.0;
        assert!(pll_of(&sharp, 3, 0) > pll_of(&sharp, 1, 0));
        assert!(pll_of(&sharp, 1, 0) < 0.0);
    }

    #[test]
    fn draw_respects_mass_concentration() {
        let mut p = vec![0.0f32; 16];
        p[9] = 1.0;
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..32 {
            assert_eq!(draw_from(&p, &mut rng), 9);
        }
    }
}
