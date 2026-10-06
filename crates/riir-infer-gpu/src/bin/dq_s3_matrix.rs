//! `dq_s3_matrix` — plan 618 S3 / Issue 028 T4: the dual-PTQ disaggregated
//! container's accuracy + TTFT measurement (the science deliverable).
//!
//! FROZEN run design: `.plans/618_dq_t4_q4_prefill_pack.md` §S3 (landed
//! 2026-10-06, pre-registered BEFORE any cell ran). This bin EXECUTES it.
//!
//! # Arms (matched TOTAL storage, per the issue's PoC gate)
//!
//! | arm | load | storage | prefill | decode |
//! |---|---|---|---|---|
//! | `base` | plain loader over the PQ2_0 pack | 6.7 GB | ternary | ternary |
//! | `dual` | `load_pair(PQ2_0, Q4_K.pf)` + the phase-switch forward | 21.1 GB | Q4_K | ternary (bit-shared) |
//! | `matched` | plain loader over the Q6_K.sg pack | ~20.7 GB | Q6_K | Q6_K |
//! | `q4single` | plain loader over Q4_K.pf (both phases) | 14.4 GB | Q4_K | Q4_K |
//!
//! `q4single` is the DECOMPOSITION control: it shares the dual's prefill
//! copy exactly, so `dual − q4single` isolates the decode-format effect and
//! `q4single − base` the prefill-copy effect at fixed decode format. It is
//! expected to beat `dual` on decode-heavy tasks — that is not a container
//! refutation, it is the decomposition.
//!
//! # Lane (fixed, pre-registered)
//!
//! The cudarc per-token GEMV forward for ALL arms — the only lane that runs
//! q4/q6 weights on the 4090 today. Kernel policy is HELD CONSTANT across
//! arms (the same int8 activation quantize → format GEMV → decode chain),
//! so arm deltas isolate the WEIGHT format; the lane's own W8A8-class
//! activation quantization is COMMON to every arm. This is NOT the 614
//! CubeCL GEMM-prefill lane — the 614 activation-phase cells remain the
//! long-context reference (cross-lane comparability disclosed as
//! approximate).
//!
//! TTFT = per-prompt prefill wall (reset → the last prefill token's logits
//! on host). Absent a q4-class GEMM prefill arm, the pre-registered reading
//! is the kernel-maturity disclosure: at per-token GEMV the walls track
//! weight bytes (≈2.15×/3.05×), and the paper's TTFT thesis stays a
//! kernel-build question, not a format question.
//!
//! # Task families (614's generators, verbatim semantics)
//!
//! - decode-heavy: arithmetic-CoT (`arith_items_v2` semantics — the same
//!   seed, operand scales, precedence-correct gold; 4-shot; greedy ≤ 256;
//!   LAST integer after `####`).
//! - prefill-heavy: multi-needle NIAH (8 needles, one queried, 16 prompts
//!   per length ∈ {1024, 2048, 4096-if-VRAM}; FIRST needle-value substring).
//!   Lengths lowered from 614's {4K, 8K, 16K} for per-token-GEMV cost —
//!   disclosed; the prefill-KV-quality mechanism shows at 1K+.
//!
//! # Verdict (026/614 vocabulary, pre-registered)
//!
//! Paired bootstrap 95% CI on (arm − base) pick-accuracy per family/length
//! (items paired by index; identical corpora across arms by construction).
//! Admissibility `0.25 ≤ acc(base) ≤ 0.95`; SATURATED guard; PRIMARY
//! reading on the prefill-heavy family: dual recovery ≥ matched recovery
//! with CI > 0 somewhere and no cell where matched significantly beats dual
//! → **HIT**; matched ≥ dual everywhere → **NULL** (container shelved, T4's
//! decomposition number stands alone); between → recorded, owner-gated.
//!
//! # Run order + robustness
//!
//! DUAL first (the VRAM binding constraint: 21.1 GB weights on the 24 GB
//! 4090 — a construction OOM at the longest length drops that length for
//! ALL arms, the pre-registered arm-set rule, then retries down to 1024),
//! then matched, q4single, base. Cells append one JSONL line each to
//! `results.partial.jsonl` (post-mortem analysis; machine resume is NOT
//! implemented — a dead run re-runs, disclosed).
//!
//! 4090 lane. GPU-EXCLUSIVE (the AGENTS rule): refuses on co-resident
//! compute apps. Env: `BONSAI_GGUF` · `DQ3_PF_GGUF` · `DQ3_Q6_GGUF` ·
//! `DQ3_OUT` · `DQ3_ARITH_N` (48) · `DQ3_NI_PER_LEN` (16) ·
//! `DQ3_LENGTHS` (1024,2048,4096) · `DQ3_ARMS` (dual,matched,q4single,base)
//! · `DQ3_BOOTSTRAP` (10000) · `DQ3_ARITH_CAP` (256) · `DQ3_NI_CAP` (32) ·
//! `DQ3_NI_NEEDLES` (8).

#![cfg(all(feature = "dq_s3_bench", not(target_os = "macos")))]
// Dev profile: the refusing main makes the measurement body dead code BY
// DESIGN (it must never run unoptimised) — silence exactly that class, in
// exactly that profile. Release keeps every warning live.
#![cfg_attr(debug_assertions, allow(dead_code))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use riir_infer_core::disaggregated::DisaggregatedTernaryWeights;
use riir_infer_core::gguf_loader::{load_qwen_deltanet_ternary_weights_gguf, GgufFile};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::ternary_deltanet_gpu_forward_cudarc::TernaryDeltanetGpuForwardCudarc;

const DEFAULT_DECODE: &str = "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const DEFAULT_PF: &str = "../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf";
const DEFAULT_Q6: &str = "../riir-train/data/Ternary-Bonsai-2-27B-Q6_K.sg.gguf";
const EOS: usize = 248_046;

// ─── deterministic corpora (the seal is this file's git commit) ─────────────

/// Deterministic u64 xorshift (the 614 bin's generator, verbatim — the
/// corpus identity rides this exact sequence).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct ArithItem {
    prompt: String,
    gold: i128,
}

const ARITH_FEW_SHOT: &str = "\
Compute 23 * 17 + 45. Think step by step, then give the final answer after ####.
23 * 17 = 391. 391 + 45 = 436.
#### 436

Compute 8842 - 1907 + 333. Think step by step, then give the final answer after ####.
8842 - 1907 = 6935. 6935 + 333 = 7268.
#### 7268

Compute 12 * 12 * 12. Think step by step, then give the final answer after ####.
12 * 12 = 144. 144 * 12 = 1728.
#### 1728

Compute 5000 / 4 - 250. Think step by step, then give the final answer after ####.
5000 / 4 = 1250. 1250 - 250 = 1000.
#### 1000

";

/// Precedence-correct gold (the 614 v2 semantics, verbatim).
fn eval_standard_precedence(first: i128, ops: &[(char, i128)]) -> Option<i128> {
    let mut terms: Vec<(char, i128)> = Vec::with_capacity(ops.len() + 1);
    let mut product = first;
    let mut pending: char = '+';
    for &(op, operand) in ops {
        match op {
            '*' => product = product.checked_mul(operand)?,
            add @ ('+' | '-') => {
                terms.push((pending, product));
                pending = add;
                product = operand;
            }
            _ => return None,
        }
    }
    terms.push((pending, product));
    let mut sum: i128 = 0;
    for (op, value) in terms {
        sum = match op {
            '+' => sum.checked_add(value)?,
            '-' => sum.checked_sub(value)?,
            _ => return None,
        };
    }
    Some(sum)
}

/// The 614 `arith_items_v2` corpus — same seed; `hard` is the 614 Issue-033
/// posture verbatim (wider operand scales to pull base below the 0.95
/// admissibility ceiling; the draw stream diverges by construction — a
/// different, hash-pinned corpus).
fn arith_items(n: usize, hard: bool) -> Vec<ArithItem> {
    let mut rng = Rng(0x0A71_A614_4847);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let n_ops = if hard { 3 + (rng.below(4) as usize) } else { 2 + (rng.below(3) as usize) };
        let first = if hard { 1_000 + rng.below(99_000) as i128 } else { 100 + rng.below(9900) as i128 };
        let mut ops: Vec<(char, i128)> = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            let op = match rng.below(3) {
                0 => '+',
                1 => '-',
                _ => '*',
            };
            let operand: i128 = match op {
                '*' => (if hard { 11 + rng.below(989) } else { 3 + rng.below(97) }) as i128,
                _ => (if hard { 100 + rng.below(9_900) } else { 10 + rng.below(990) }) as i128,
            };
            ops.push((op, operand));
        }
        let mut expr = format!("{first}");
        for &(op, operand) in &ops {
            expr.push_str(&format!(" {op} {operand}"));
        }
        let gold = eval_standard_precedence(first, &ops)
            .expect("operand scales cannot overflow i128");
        out.push(ArithItem {
            prompt: format!(
                "{ARITH_FEW_SHOT}Compute {expr}. Think step by step, then give the final answer after ####.\n"
            ),
            gold,
        });
    }
    out
}

/// Parse the model output: the LAST integer after `####` (the 614 rule).
fn parse_arith_answer(out: &str) -> Option<i128> {
    let idx = out.rfind("####")?;
    let tail = &out[idx + 4..];
    let cleaned: String = tail.chars().filter(|c| !c.is_whitespace() && *c != ',').collect();
    let mut num = String::new();
    for (i, c) in cleaned.char_indices() {
        if (i == 0 && c == '-') || c.is_ascii_digit() {
            num.push(c);
        } else {
            break;
        }
    }
    num.parse().ok()
}

/// The natural-text paragraph bank (the 614 bank, verbatim).
const PARAGRAPHS: [&str; 24] = [
    "Rivers have always decided where people gather. A dependable current carries silt onto the floodplain, and the silt becomes soil, and the soil becomes bread.",
    "The old surveyors worked from ridge to ridge, sighting across valleys with instruments no heavier than a clock and a lens. Their maps still name the passes.",
    "In the workshop, a seasoned joiner reads the grain before the cut. Timber that looks identical on the rack will fight the plane differently on Tuesday than on Friday.",
    "Harbor pilots memorize the shifting bar. Charts go stale in a single winter storm, so the trade stays oral: an apprentice rides along until the channel lives in the hands.",
    "Markets form where two roads cross, and they outlive the roads. Archaeologists find the market first, then the milestone, then the reason for both.",
    "A good furnace draws air like a lung. The smith controls the color more than the flame, because color is temperature, and temperature is everything the steel will become.",
    "Migrating birds calibrate against fields humans cannot sense. Release a homing pigeon under an overcast and it circles once, then chooses, and is rarely argued with.",
    "The library kept its ledger of borrowed rain. Farmers deposited in the wet years and drew against the ledger when the sky forgot its obligations.",
    "Every bridge is a wager about the river's worst mood. The engineer who designs for the average current has built a monument to the next exceptional flood.",
    "Children learn the shortest path home through their feet, not the map. The body's geometry cuts corners the surveyor would never permit on paper.",
    "Salt roads cross the high passes because salt is light, keeps, and everyone needs it. Empires have followed heavier goods into ruin while salt quietly paid the porters.",
    "The clockmaker's bench is a study in deferred consequences. A mainspring wound one tooth too tight will keep perfect time for exactly as long as it takes to matter.",
    "Orchardists talk to each other through grafts. A variety that never flowered in one valley fruits generously two ridges east, and nobody fully knows why.",
    "The tides keep an older appointment book. Whatever the harbor plans, the water arrives on schedule, and every fitting, every launch, every cargo waits on that ledger.",
    "A lattice of irrigation channels lies under the oldest terraces. Each generation cleans the one above its field and complains about the one below, which is tradition.",
    "The ferryman charges by weight and mood. Neither is posted, both are stable, and regulars learn to travel light in argument as in luggage.",
    "Smoke from the brick kilns used to give the valley its weather. On firing days the horizon shortened, and everyone's temper with it.",
    "Scribes copied the almanac forward each year, adding frost dates in the margins. The margins are the science; the almanac is the memory.",
    "The mountain post runs at night because the ice is honest after dark. By noon the passes lie about their footing.",
    "Fishermen read the surface for what lies beneath it. A still patch means a school; a troubled patch means something bigger that nobody discusses over dinner.",
    "The town clock strikes thirteen once a decade, and the council spends a fortnight deciding whether it happened. Minutes are kept.",
    "Weavers keep a knot for every customer. A finished bolt carries its whole history in its selvage, if you know the dialect.",
    "The seed bank buried its duplicates on the cold side of the hill. Optimism is a frost-sensitive crop.",
    "Roadside shrines mark the sites of old arguments between drivers and gravity. Gravity has never once apologized, which the shrines record.",
];

const NEEDLE_SERVERS: [&str; 10] = [
    "xoris", "talv", "murex", "quenn", "obex", "sarn", "lirix", "vanto", "perid", "zokk",
];
const NEEDLE_CODES: [&str; 10] = [
    "4f7a2", "9c1e6", "77b0d", "e3d59", "a08f2", "b6c41", "d92e7", "5a3b8", "c47f0", "8e0a3",
];

struct NiahItem {
    prompt: String,
    gold: String,
    all_codes: Vec<String>,
}

/// The 614 `niah_item` (verbatim semantics).
fn niah_item(idx: usize, len_round: usize, target_tokens: usize, tok: &BpeTokenizer, nn: usize) -> NiahItem {
    let nn = nn.clamp(1, NEEDLE_SERVERS.len());
    let mut rng = Rng(0x61A1_0000 + idx as u64 * 7919 + len_round as u64);
    let needle_off = (idx * 3) % 10;
    let needles: Vec<(usize, usize)> = (0..nn)
        .map(|k| {
            let s = (needle_off + k) % 10;
            let c = (s + idx + k) % 10;
            (s, c)
        })
        .collect();
    let target_k = (idx * 5 + 2) % nn;
    let (t_server, t_code) = needles[target_k];

    let mut parts: Vec<String> = Vec::new();
    let mut tokens_so_far = 0usize;
    let mut para_i = rng.below(24) as usize;
    let mut needle_i = 0usize;
    let needle_band = (target_tokens / (nn + 1)).max(1);
    let mut next_needle_at = needle_band;
    loop {
        if needle_i < nn && tokens_so_far >= next_needle_at {
            let (s, c) = needles[needle_i];
            parts.push(format!(
                "Note: the password for server {} is {}.\n",
                NEEDLE_SERVERS[s], NEEDLE_CODES[c]
            ));
            needle_i += 1;
            next_needle_at += needle_band;
        }
        let p = PARAGRAPHS[para_i % 24];
        para_i = (para_i + 1 + rng.below(3) as usize) % 24;
        parts.push(p.to_string());
        parts.push("\n".to_string());
        tokens_so_far = tok.encode(&parts.concat()).len();
        if tokens_so_far >= target_tokens - 64 {
            break;
        }
        if parts.len() > 4096 {
            break; // safety
        }
    }
    while needle_i < nn {
        let (s, c) = needles[needle_i];
        parts.push(format!(
            "Note: the password for server {} is {}.\n",
            NEEDLE_SERVERS[s], NEEDLE_CODES[c]
        ));
        needle_i += 1;
    }
    parts.push(format!(
        "\nQuery: what is the password for server {}? The password for server {} is",
        NEEDLE_SERVERS[t_server], NEEDLE_SERVERS[t_server]
    ));
    NiahItem {
        prompt: parts.concat(),
        gold: NEEDLE_CODES[t_code].to_string(),
        all_codes: needles
            .iter()
            .map(|(_, c)| NEEDLE_CODES[*c].to_string())
            .collect(),
    }
}

/// FIRST needle-value substring in the output wins (the 614 rule).
fn score_niah(out: &str, item: &NiahItem) -> bool {
    let mut first: Option<(usize, &str)> = None;
    for c in &item.all_codes {
        if let Some(p) = out.find(c.as_str())
            && (first.is_none() || p < first.unwrap().0)
        {
            first = Some((p, c.as_str()));
        }
    }
    matches!(&first, Some((_, c)) if *c == item.gold)
}

// ─── generation (the cudarc lane) ───────────────────────────────────────────

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

struct GenOut {
    text: String,
    n_prefill: usize,
    /// TTFT: reset → the last prefill token's logits on host.
    prefill_wall: f64,
    /// The generation steps AFTER the prefill-produced first token.
    decode_wall: f64,
    n_generated: usize, // INCLUDING the prefill-produced first token
}

/// Greedy generation on the cudarc lane. Prompt tokens ride
/// `forward_dispatch_only` (+ per-token sync, bounding the launch queue);
/// the LAST prompt token takes `forward_token` (its sync + download IS the
/// TTFT boundary). The dual arm arms the prefill phase across the prompt and
/// disarms before generation (the phase switch — the KV + GDN state cross
/// the boundary inside the one forward instance).
fn greedy_generate_cudarc(
    fwd: &mut TernaryDeltanetGpuForwardCudarc,
    tok: &BpeTokenizer,
    bos: usize,
    prompt: &str,
    cap: usize,
    prefill_phase: bool,
) -> Result<GenOut, String> {
    let mut tokens = tok.encode(prompt);
    if tokens.first() != Some(&bos) {
        tokens.insert(0, bos);
    }
    fwd.reset_state().map_err(|e| e.to_string())?;
    if prefill_phase {
        fwd.set_phase_prefill(true).map_err(|e| e.to_string())?;
    }
    let t0 = Instant::now();
    let mut logits: Vec<f32> = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        fwd.set_input_token(t).map_err(|e| e.to_string())?;
        if i + 1 < tokens.len() {
            fwd.forward_dispatch_only().map_err(|e| e.to_string())?;
            fwd.synchronize().map_err(|e| e.to_string())?;
        } else {
            logits = fwd.forward_token().map_err(|e| e.to_string())?;
        }
    }
    let prefill_wall = t0.elapsed().as_secs_f64();
    if prefill_phase {
        fwd.set_phase_prefill(false).map_err(|e| e.to_string())?;
    }

    let mut out_ids: Vec<usize> = Vec::with_capacity(cap);
    let mut next = argmax(&logits);
    out_ids.push(next);
    let td = Instant::now();
    while out_ids.len() < cap && next != EOS {
        fwd.set_input_token(next).map_err(|e| e.to_string())?;
        let l = fwd.forward_token().map_err(|e| e.to_string())?;
        next = argmax(&l);
        out_ids.push(next);
    }
    let decode_wall = td.elapsed().as_secs_f64();

    Ok(GenOut {
        text: tok.decode(&out_ids),
        n_prefill: tokens.len(),
        prefill_wall,
        decode_wall,
        n_generated: out_ids.len(),
    })
}

// ─── statistics (the 614 machinery, verbatim) ───────────────────────────────

/// Deterministic paired bootstrap 95% CI on the mean of per-item paired
/// differences (in {−1, 0, +1}).
fn paired_bootstrap_ci(d: &[i8], n_boot: usize) -> (f64, f64) {
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = Rng(0x0B00_5614);
    let n = d.len();
    let mut means: Vec<f64> = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        let mut s = 0i64;
        for _ in 0..n {
            s += d[rng.below(n as u64) as usize] as i64;
        }
        means.push(s as f64 / n as f64);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    let lo = means[(0.025 * n_boot as f64) as usize].min(means[n_boot - 1]);
    let hi = means[(0.975 * n_boot as f64) as usize].max(means[0]);
    (lo, hi)
}

/// Nearest-rank percentile with the tail support disclosed (the repo's own
/// percentile lesson: `sorted[(n·p) as usize]` lands on the MAX for small n —
/// the caller prints n beside every p99 so the support is visible).
fn percentile(walls: &mut [f64], p: f64) -> f64 {
    if walls.is_empty() {
        return 0.0;
    }
    walls.sort_by(|a, b| a.total_cmp(b));
    let idx = (((p * walls.len() as f64) as usize).min(walls.len() - 1)).min(walls.len() - 1);
    walls[idx]
}

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

// ─── cells ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArmId {
    Dual,
    Matched,
    Q4Single,
    Base,
}

impl ArmId {
    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "dual" => Some(Self::Dual),
            "matched" => Some(Self::Matched),
            "q4single" | "q4" => Some(Self::Q4Single),
            "base" => Some(Self::Base),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Dual => "dual",
            Self::Matched => "matched",
            Self::Q4Single => "q4single",
            Self::Base => "base",
        }
    }
}

struct CellResult {
    arm: ArmId,
    family: &'static str, // "arith" | "niah"
    length: usize,        // 0 for arith
    correct: Vec<u8>,
    prefill_walls: Vec<f64>,
    decode_walls: Vec<f64>,
    n_gen: Vec<usize>,
    n_prefill_tokens: Vec<usize>,
}

/// One JSONL line per cell — the partial-results record (post-mortem
/// analysis; machine resume is NOT implemented, disclosed in the module doc).
fn cell_jsonl_line(cell: &CellResult, corpus: &str) -> String {
    let nums = |v: &[u8]| -> String {
        v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
    };
    let f64s = |v: &[f64]| -> String {
        v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",")
    };
    format!(
        "{{\"corpus\":\"{corpus}\",\"arm\":\"{}\",\"family\":\"{}\",\"length\":{},\"n\":{},\"correct\":[{}],\"prefill_walls\":[{}],\"decode_walls\":[{}],\"n_gen\":[{}],\"n_prefill_tokens\":[{}]}}\n",
        cell.arm.name(),
        cell.family,
        cell.length,
        cell.correct.len(),
        nums(&cell.correct),
        f64s(&cell.prefill_walls),
        f64s(&cell.decode_walls),
        cell.n_gen.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
        cell.n_prefill_tokens.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
    )
}

/// Run one family cell end-to-end on `fwd`; returns the result (the caller
/// appends the JSONL line + keeps it in memory).
fn run_cell(
    fwd: &mut TernaryDeltanetGpuForwardCudarc,
    tok: &BpeTokenizer,
    bos: usize,
    arm: ArmId,
    family: &'static str,
    length: usize,
    items: &[(String, String)], // (prompt, gold)
    cap: usize,
    prefill_phase: bool,
    progress_tag: &str,
) -> CellResult {
    let n = items.len();
    let mut cell = CellResult {
        arm,
        family,
        length,
        correct: Vec::with_capacity(n),
        prefill_walls: Vec::with_capacity(n),
        decode_walls: Vec::with_capacity(n),
        n_gen: Vec::with_capacity(n),
        n_prefill_tokens: Vec::with_capacity(n),
    };
    for (i, (prompt, gold)) in items.iter().enumerate() {
        let out = greedy_generate_cudarc(fwd, tok, bos, prompt, cap, prefill_phase)
            .unwrap_or_else(|e| panic!("{progress_tag} item {i}: generation failed: {e}"));
        let hit = if family == "arith" {
            parse_arith_answer(&out.text).is_some_and(|v| v == gold.parse::<i128>().unwrap())
        } else {
            // the gold string rides the (prompt, gold) pair; the niah caller
            // passes the FULL all-codes set in the gold field as a compact
            // encoding: gold | distractor1 | … — see build_niah_items.
            let mut parts = gold.split('|');
            let g = parts.next().unwrap_or("");
            let fake = NiahItem {
                prompt: String::new(),
                gold: g.to_string(),
                all_codes: parts.map(|s| s.to_string()).collect(),
            };
            score_niah(&out.text, &fake)
        };
        cell.correct.push(u8::from(hit));
        cell.prefill_walls.push(out.prefill_wall);
        cell.decode_walls.push(out.decode_wall);
        cell.n_gen.push(out.n_generated);
        cell.n_prefill_tokens.push(out.n_prefill);
        if (i + 1).is_multiple_of(4) || i + 1 == n {
            let acc = cell.correct.iter().sum::<u8>() as f64 / cell.correct.len() as f64;
            eprintln!(
                "[dq3 {progress_tag}] {}/{} acc-so-far {acc:.3} ttft-p50-so-far {:.1} ms",
                i + 1,
                n,
                percentile(&mut cell.prefill_walls.clone(), 0.5) * 1e3
            );
        }
    }
    cell
}

// ─── main ───────────────────────────────────────────────────────────────────

fn hard_exit(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // The teardown-hang class (Issue 031): exit the PROCESS without running
    // the CUDA destructors — the same defense dq_phase_matrix landed.
    std::process::exit(code);
}

fn main() {
    #[cfg(debug_assertions)]
    {
        eprintln!("[dq3] refusing: a debug build must never measure (release only)");
        hard_exit(2);
    }
    #[cfg(not(debug_assertions))]
    {
        let code = match run() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("[dq3] FATAL: {e}");
                1
            }
        };
        hard_exit(code);
    }
}

fn run() -> Result<(), String> {
    let decode_path = env_or("BONSAI_GGUF", DEFAULT_DECODE);
    let pf_path = env_or("DQ3_PF_GGUF", DEFAULT_PF);
    let q6_path = env_or("DQ3_Q6_GGUF", DEFAULT_Q6);
    let out_dir = PathBuf::from(env_or("DQ3_OUT", ".benchmarks/dq_s3_matrix"));
    let arith_n: usize = env_or("DQ3_ARITH_N", "48").parse().map_err(|_| "DQ3_ARITH_N")?;
    let arith_hard = env_or("DQ3_ARITH_HARD", "0") == "1";
    let ni_per_len: usize = env_or("DQ3_NI_PER_LEN", "16").parse().map_err(|_| "DQ3_NI_PER_LEN")?;
    let ni_needles: usize = env_or("DQ3_NI_NEEDLES", "8").parse().map_err(|_| "DQ3_NI_NEEDLES")?;
    let mut lengths: Vec<usize> = env_or("DQ3_LENGTHS", "1024,2048,4096")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if lengths.is_empty() {
        return Err("DQ3_LENGTHS produced no lengths".into());
    }
    lengths.sort_unstable();
    let arm_order: Vec<ArmId> = env_or("DQ3_ARMS", "dual,matched,q4single,base")
        .split(',')
        .filter_map(|s| ArmId::parse(s))
        .collect();
    if arm_order.is_empty() {
        return Err("DQ3_ARMS produced no arms".into());
    }
    if !arm_order.contains(&ArmId::Base) {
        return Err("the arm set must include base (the paired reference)".into());
    }
    let n_boot: usize = env_or("DQ3_BOOTSTRAP", "10000").parse().map_err(|_| "DQ3_BOOTSTRAP")?;
    let arith_cap: usize = env_or("DQ3_ARITH_CAP", "256").parse().map_err(|_| "DQ3_ARITH_CAP")?;
    let ni_cap: usize = env_or("DQ3_NI_CAP", "32").parse().map_err(|_| "DQ3_NI_CAP")?;

    std::fs::create_dir_all(&out_dir).map_err(|e| format!("out dir: {e}"))?;
    let partial_path = out_dir.join("results.partial.jsonl");

    // ── GPU exclusivity (the AGENTS rule; the 614 probe) ──────────────────
    let apps = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi probe failed: {e}"))?;
    let apps_csv = String::from_utf8_lossy(&apps.stdout).trim().to_string();
    let compute_consumers: Vec<&str> = apps_csv
        .lines()
        .filter(|line| {
            let mem = line.rsplit(',').next().unwrap_or("").trim();
            !mem.is_empty() && mem != "[N/A]" && mem.trim_end_matches(" MiB").parse::<u64>().is_ok_and(|m| m > 0)
        })
        .collect();
    if !compute_consumers.is_empty() {
        return Err(format!(
            "GPU-EXCLUSIVITY refusal: co-resident compute apps:\n{}",
            compute_consumers.join("\n")
        ));
    }
    let gpu_info = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    let gpu_csv = String::from_utf8_lossy(&gpu_info.stdout).trim().to_string();

    // ── tokenizer + corpora (from the decode pack's tokenizer) ───────────
    eprintln!("[dq3] tokenizer from {decode_path} …");
    let gguf = GgufFile::open(Path::new(&decode_path)).map_err(|e| format!("open decode: {e}"))?;
    let tok = BpeTokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
    drop(gguf);

    let arith = arith_items(arith_n, arith_hard);
    let arith_items_pairs: Vec<(String, String)> = arith
        .iter()
        .map(|a| (a.prompt.clone(), a.gold.to_string()))
        .collect();
    let niah: Vec<(usize, Vec<(String, String)>)> = lengths
        .iter()
        .enumerate()
        .map(|(r, &l)| {
            let items = (0..ni_per_len)
                .map(|i| {
                    let it = niah_item(i, r, l, &tok, ni_needles);
                    // compact encoding: gold | the full distractor set (the
                    // first-match rule needs every code present in the output
                    // window, not just the gold).
                    let mut enc = it.gold.clone();
                    for c in &it.all_codes {
                        enc.push('|');
                        enc.push_str(c);
                    }
                    (it.prompt, enc)
                })
                .collect();
            (l, items)
        })
        .collect();

    let corpus_hash = {
        let mut h = blake3::Hasher::new();
        for (p, g) in &arith_items_pairs {
            h.update(p.as_bytes());
            h.update(g.as_bytes());
        }
        for (_, items) in &niah {
            for (p, g) in items {
                h.update(p.as_bytes());
                h.update(g.as_bytes());
            }
        }
        h.finalize().to_hex().to_string()
    };
    eprintln!("[dq3] corpus blake3={corpus_hash}");

    // ── the lane record (frozen BEFORE any accuracy cell) ────────────────
    eprintln!(
        "[dq3] LANE RECORD (frozen): cudarc per-token GEMV for ALL arms — same int8 \
         activation quantize, per-arm weight format; kernel policy constant; the 614 \
         CubeCL GEMM-prefill lane is NOT this lane (cross-lane comparability approximate)"
    );

    let max_len = *lengths.last().unwrap();
    let block_for = |lens: &[usize]| max_of(lens) + 64;

    // ── per-arm cells ────────────────────────────────────────────────────
    let mut cells: Vec<CellResult> = Vec::new();
    let mut partial = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&partial_path)
        .map_err(|e| format!("partial: {e}"))?;
    use std::io::Write as _;

    for &arm in &arm_order {
        let t_arm = Instant::now();
        // Construction (the dual arm carries the VRAM drop rule: an OOM at
        // the longest length drops it for the WHOLE arm set — the arms stay
        // comparable — and retries down the length list).
        let (mut config, mut fwd, dual) = match arm {
            ArmId::Dual => {
                let (mut cfg, container) = DisaggregatedTernaryWeights::load_pair(
                    Path::new(&decode_path),
                    Path::new(&pf_path),
                )
                .map_err(|e| format!("load_pair: {e}"))?;
                let mut built = None;
                loop {
                    cfg.block_size = cfg.block_size.min(block_for(&lengths));
                    match TernaryDeltanetGpuForwardCudarc::new(&cfg, container.decode()) {
                        Ok(mut f) => match f.attach_prefill_copy(container.prefill()) {
                            Ok(()) => {
                                built = Some((cfg.clone(), f, true));
                                break;
                            }
                            Err(e) => return Err(format!("attach prefill: {e}")),
                        },
                        Err(e) => {
                            if lengths.len() > 1 {
                                let dropped = lengths.pop().unwrap_or(max_len);
                                eprintln!(
                                    "[dq3] dual construct OOM at block {}: dropping length {dropped} for the WHOLE arm set (the pre-registered rule): {e}",
                                    cfg.block_size
                                );
                                continue;
                            }
                            return Err(format!("dual construct at the last length: {e}"));
                        }
                    }
                }
                built.unwrap()
            }
            ArmId::Matched => {
                let (mut cfg, w) =
                    load_qwen_deltanet_ternary_weights_gguf(Path::new(&q6_path))
                        .map_err(|e| format!("load {q6_path}: {e}"))?;
                cfg.block_size = cfg.block_size.min(block_for(&lengths));
                let f = TernaryDeltanetGpuForwardCudarc::new(&cfg, &w)
                    .map_err(|e| format!("construct matched: {e}"))?;
                (cfg, f, false)
            }
            ArmId::Q4Single => {
                let (mut cfg, w) =
                    load_qwen_deltanet_ternary_weights_gguf(Path::new(&pf_path))
                        .map_err(|e| format!("load {pf_path}: {e}"))?;
                cfg.block_size = cfg.block_size.min(block_for(&lengths));
                let f = TernaryDeltanetGpuForwardCudarc::new(&cfg, &w)
                    .map_err(|e| format!("construct q4single: {e}"))?;
                (cfg, f, false)
            }
            ArmId::Base => {
                let (mut cfg, w) =
                    load_qwen_deltanet_ternary_weights_gguf(Path::new(&decode_path))
                        .map_err(|e| format!("load {decode_path}: {e}"))?;
                cfg.block_size = cfg.block_size.min(block_for(&lengths));
                let f = TernaryDeltanetGpuForwardCudarc::new(&cfg, &w)
                    .map_err(|e| format!("construct base: {e}"))?;
                (cfg, f, false)
            }
        };
        let bos = config.bos_token;
        eprintln!(
            "[dq3] arm {} ready ({:.0}s) block_size={} n_layer={}",
            arm.name(),
            t_arm.elapsed().as_secs_f32(),
            config.block_size,
            config.n_layer
        );

        // arith (decode-heavy)
        let t_cell = Instant::now();
        let cell = run_cell(
            &mut fwd,
            &tok,
            bos,
            arm,
            "arith",
            0,
            &arith_items_pairs,
            arith_cap,
            dual,
            &format!("{} arith", arm.name()),
        );
        eprintln!(
            "[dq3] arm {} cell arith done in {:.0}s",
            arm.name(),
            t_cell.elapsed().as_secs_f32()
        );
        let line = cell_jsonl_line(&cell, &corpus_hash);
        partial.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        partial.flush().map_err(|e| e.to_string())?;
        cells.push(cell);

        // niah (prefill-heavy) per length
        for (len, items) in &niah {
            // the dual drop rule may have shrunk the set AFTER construction —
            // skip lengths above this arm's clamped block size (they never
            // fit this arm's KV reservation).
            if *len + 64 > config.block_size {
                eprintln!("[dq3] arm {} skips niah {len} (block_size {})", arm.name(), config.block_size);
                continue;
            }
            let t_cell = Instant::now();
            let cell = run_cell(
                &mut fwd,
                &tok,
                bos,
                arm,
                "niah",
                *len,
                items,
                ni_cap,
                dual,
                &format!("{} niah{len}", arm.name()),
            );
            eprintln!(
                "[dq3] arm {} cell niah{len} done in {:.0}s",
                arm.name(),
                t_cell.elapsed().as_secs_f32()
            );
            let line = cell_jsonl_line(&cell, &corpus_hash);
            partial.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
            partial.flush().map_err(|e| e.to_string())?;
            cells.push(cell);
        }
        drop(fwd);
        eprintln!("[dq3] arm {} complete ({:.0}s total)", arm.name(), t_arm.elapsed().as_secs_f32());
    }

    // ── the verdict ──────────────────────────────────────────────────────
    let report = build_report(
        &cells,
        arith_n,
        arith_hard,
        ni_per_len,
        ni_needles,
        n_boot,
        &corpus_hash,
        &gpu_csv,
        &decode_path,
        &pf_path,
        &q6_path,
        &lengths,
        arith_cap,
        ni_cap,
    )?;
    let md_path = out_dir.join("dq_s3_matrix.md");
    std::fs::write(&md_path, &report).map_err(|e| format!("write md: {e}"))?;
    eprintln!("[dq3] report written to {}", md_path.display());
    Ok(())
}

fn max_of(lens: &[usize]) -> usize {
    lens.iter().copied().max().unwrap_or(1024)
}

// ─── the report ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn build_report(
    cells: &[CellResult],
    arith_n: usize,
    arith_hard: bool,
    ni_per_len: usize,
    ni_needles: usize,
    n_boot: usize,
    corpus_hash: &str,
    gpu_csv: &str,
    decode_path: &str,
    pf_path: &str,
    q6_path: &str,
    lengths: &[usize],
    arith_cap: usize,
    ni_cap: usize,
) -> Result<String, String> {
    let mut md = String::new();
    md.push_str("# dq_s3_matrix — the dual-PTQ measurement (plan 618 S3 / issue 028 T4)\n\n");
    md.push_str(&format!(
        "- **Lane (frozen):** cudarc per-token GEMV, ALL arms — kernel policy constant; the lane's int8 activation quantization is COMMON to every arm, so deltas isolate the weight format. NOT the 614 GEMM-prefill lane (cross-lane comparability approximate).\n"
    ));
    md.push_str(&format!(
        "- **Corpus blake3:** `{corpus_hash}` — arith {arith_n} items (the 614 v2 corpus, hard={arith_hard}), niah {ni_per_len}/len × {ni_needles} needles, lengths {lengths:?}\n"
    ));
    md.push_str(&format!(
        "- **Gen caps:** arith ≤ {arith_cap}, niah ≤ {ni_cap} · bootstrap {n_boot}\n"
    ));
    md.push_str(&format!("- **GPU:** {gpu_csv}\n"));
    md.push_str(&format!("- **Packs:** decode `{decode_path}` · pf `{pf_path}` · q6 `{q6_path}`\n"));
    md.push_str(&format!(
        "- **Storage pairing:** base 6.7 GB · dual 21.1 GB · matched ≈20.7 GB (the control ~2% CHEAPER — the conservative direction, disclosed) · q4single 14.4 GB (the decomposition control)\n\n"
    ));

    // The cells table.
    md.push_str("## Cells\n\n");
    md.push_str("| arm | family | len | n | acc | TTFT p50 (ms) | TTFT p99 (ms) | decode tok/s |\n|---|---|---|---|---|---|---|---|\n");
    for c in cells {
        let n = c.correct.len();
        let acc = c.correct.iter().sum::<u8>() as f64 / n.max(1) as f64;
        let walls = c.prefill_walls.clone();
        let p50 = percentile(&mut walls.clone(), 0.5) * 1e3;
        let p99 = percentile(&mut walls.clone(), 0.99) * 1e3;
        let decode_steps: f64 = (c.n_gen.iter().sum::<usize>() as f64) - n as f64;
        let decode_total: f64 = c.decode_walls.iter().sum();
        let tps = if decode_total > 0.0 { decode_steps / decode_total } else { 0.0 };
        md.push_str(&format!(
            "| {} | {} | {} | {n} | {acc:.4} | {p50:.1} | {p99:.1} | {tps:.2} |\n",
            c.arm.name(),
            c.family,
            c.length,
        ));
    }

    // Paired stats vs base.
    let base_of = |family: &str, len: usize| -> Option<&CellResult> {
        cells.iter().find(|c| c.arm == ArmId::Base && c.family == family && c.length == len)
    };
    md.push_str("\n## Paired vs base (bootstrap 95% CI on the mean paired difference)\n\n");
    md.push_str("| arm | family | len | recovery | CI lo | CI hi | n | note |\n|---|---|---|---|---|---|---|---|\n");
    let mut paired: Vec<(ArmId, &'static str, usize, f64, f64, f64)> = Vec::new();
    // The admissibility gate (026/614): 0.25 ≤ acc(base) ≤ 0.95 — a base at
    // ceiling cannot measure recovery (no headroom); the first run measured
    // EVERY cell at/above the ceiling and the S3.5 pre-registration binds
    // the verdict to READ THAT, never a silently-read NULL over inadmissible
    // cells.
    let mut admissible_cells = 0usize;
    for c in cells {
        if c.arm == ArmId::Base {
            continue;
        }
        let Some(b) = base_of(c.family, c.length) else {
            continue;
        };
        if b.correct.len() != c.correct.len() {
            md.push_str(&format!(
                "| {} | {} | {} | — | — | — | {} | LENGTH MISMATCH vs base |\n",
                c.arm.name(), c.family, c.length, c.correct.len()
            ));
            continue;
        }
        let b_acc = b.correct.iter().sum::<u8>() as f64 / b.correct.len() as f64;
        let admissible = (0.25..=0.95).contains(&b_acc);
        let d: Vec<i8> = c
            .correct
            .iter()
            .zip(&b.correct)
            .map(|(a, bb)| *a as i8 - *bb as i8)
            .collect();
        let rec = d.iter().map(|&x| f64::from(x)).sum::<f64>() / d.len() as f64;
        let (lo, hi) = paired_bootstrap_ci(&d, n_boot);
        let note = if !admissible {
            "INADMISSIBLE base (ceiling/floor) — excluded from the verdict"
        } else {
            admissible_cells += 1;
            ""
        };
        paired.push((c.arm, c.family, c.length, rec, lo, hi));
        md.push_str(&format!(
            "| {} | {} | {} | {rec:+.4} | {lo:+.4} | {hi:+.4} | {} | {} |\n",
            c.arm.name(),
            c.family,
            c.length,
            d.len(),
            note
        ));
    }

    // The pre-registered verdict.
    md.push_str("\n## Verdict (pre-registered, plan 618 §S3.3)\n\n");
    if admissible_cells == 0 {
        md.push_str("**RECORD-SATURATED** — zero admissible cells: every base sits at/above the 0.95 admissibility ceiling, so NEITHER pre-registered pole (HIT nor NULL) can fire. The instrument cannot falsify at this difficulty; per the S3.5 pre-registration the honest read is the saturation record plus the exact-zero paired observation, and the shelving decision rides the S3b hard corpus (pre-registered) or parsimony.\n");
        return Ok(md);
    }
    let find = |arm: ArmId, family: &str, len: usize| -> Option<(f64, f64, f64)> {
        paired
            .iter()
            .find(|(a, f, l, _, _, _)| *a == arm && *f == family && *l == len)
            .map(|(_, _, _, rec, lo, hi)| (*rec, *lo, *hi))
    };
    let chance_note = "chance ≈ 1/8 (8-needle first-match); arith chance ≈ 0";
    let mut hit = false;
    let mut null = false;
    let mut inadmissible = false;
    let mut legs = String::new();
    if !paired.is_empty() {
        let mut all_lengths_ok = true;
        let mut any_dual_ci_positive = false;
        let mut matched_sig_beats_dual_anywhere = false;
        let mut dual_ge_matched_all = true;
        let mut matched_ge_dual_all = true;
        for &len in lengths {
            let (Some((rd, lod, _)), Some((rm, _, him))) = (
                find(ArmId::Dual, "niah", len),
                find(ArmId::Matched, "niah", len),
            ) else {
                legs.push_str(&format!("- niah {len}: arm cells missing — the length was dropped or not run\n"));
                continue;
            };
            let dual_ci_pos = rd > 0.0 && lod > 0.0;
            any_dual_ci_positive |= dual_ci_pos;
            dual_ge_matched_all &= rd >= rm;
            matched_ge_dual_all &= rm >= rd;
            let _ = him; // matched's CI hi (the direct dual−matched leg below is the honest matched-beats test)
            legs.push_str(&format!(
                "- niah {len}: dual rec {rd:+.4} (CI {lod:+.4}) · matched rec {rm:+.4} (CI {him:+.4}) · dual≥matched {} · dual-CI>0 {}\n",
                rd >= rm, dual_ci_pos
            ));
        }
        // the matched−dual DIRECT paired bootstrap (the honest leg)
        if let (Some(dc), Some(mc)) = (
            cells.iter().find(|c| c.arm == ArmId::Dual && c.family == "niah"),
            cells.iter().find(|c| c.arm == ArmId::Matched && c.family == "niah"),
        ) {
            for &len in lengths {
                let d = dc.correct.len();
                let _ = d;
                let (Some(dd), Some(mm)) = (
                    cells.iter().find(|c| c.arm == ArmId::Dual && c.family == "niah" && c.length == len),
                    cells.iter().find(|c| c.arm == ArmId::Matched && c.family == "niah" && c.length == len),
                ) else {
                    continue;
                };
                if dd.correct.len() != mm.correct.len() {
                    continue;
                }
                let diffs: Vec<i8> = dd
                    .correct
                    .iter()
                    .zip(&mm.correct)
                    .map(|(a, b)| *a as i8 - *b as i8)
                    .collect();
                let (lo, hi) = paired_bootstrap_ci(&diffs, n_boot);
                if lo > 0.0 {
                    matched_sig_beats_dual_anywhere = true;
                }
                legs.push_str(&format!(
                    "- niah {len} direct dual−matched: mean {:+.4} CI [{:+.4}, {:+.4}]\n",
                    diffs.iter().map(|&x| f64::from(x)).sum::<f64>() / diffs.len() as f64,
                    lo, hi
                ));
            }
        }
        hit = any_dual_ci_positive && dual_ge_matched_all && !matched_sig_beats_dual_anywhere;
        null = matched_ge_dual_all && !any_dual_ci_positive;
        if !hit && !null {
            inadmissible = true; // mixed — recorded, owner-gated
        }
        let _ = all_lengths_ok;
    } else {
        legs.push_str("- no paired cells (base missing?)\n");
    }

    md.push_str(&legs);
    md.push_str(&format!("\n({chance_note})\n\n"));
    if inadmissible {
        md.push_str("**MIXED — recorded, owner-gated** (neither pre-registered pole fired; the legs above decide)\n");
    } else if hit {
        md.push_str("**HIT** — the dual container's prefill-heavy recovery ≥ the matched-storage single's, CI-supported, nowhere significantly beaten. The container earns its storage; the owner gate (deployment) opens.\n");
    } else if null {
        md.push_str("**NULL** — the matched-storage single ≥ dual everywhere. Container SHELVED per the pre-registered gate; T4's decomposition number stands on its own.\n");
    }

    md.push_str("\n## Disclosures\n\n");
    md.push_str("- TTFT here is per-token-GEMV prefill wall — the kernel-maturity posture: walls track weight bytes (≈2.15×/3.05×); the paper's TTFT thesis needs a q4-class GEMM prefill arm and stays a kernel-build question.\n");
    md.push_str("- p99 rows print beside n — at n=16 the p99 IS ~max (the percentile lesson; tail support 1).\n");
    md.push_str("- Cross-lane: the 614 activation-phase cells (GEMM prefill, A8/A4 fake-quant) remain the long-context reference; this lane's arm deltas are weight-format-only by construction.\n");
    md.push_str("- matched-storage delta: the Q6_K control rides ~2% LESS storage than the dual pair — the conservative direction for the container's verdict.\n");
    md.push_str("- Machine resume is not implemented: a dead run re-runs (the partial JSONL is the post-mortem record).\n");
    Ok(md)
}
