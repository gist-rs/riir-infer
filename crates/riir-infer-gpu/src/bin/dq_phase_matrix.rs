//! `dq_phase_matrix` — Plan 614 / Issue 026: the DQ phase-sensitivity bench
//! runner (the 2×2 activation-quant phase matrix on Ternary-Bonsai-2-27B-PQ2_0).
//!
//! The FROZEN protocol lives in `.plans/614_dq_phase_sensitivity_bench.md`
//! (commit d3300f2 + the T1/T2 amendments); this bin EXECUTES it. Deviations
//! discovered at implementation time are recorded in the run report (the
//! lane fallback — no non-int8 folded prefill exists — and the built-in
//! haystack bank when no corpus dir is given).
//!
//! # Cells (per grid tier)
//!
//! | cell | prefill | decode | role |
//! |---|---|---|---|
//! | `base` | shipping A8 lane | A16 | the reference |
//! | `pf_aq` | shipping A8 + fake-quant(grid) | A16 | Δpf |
//! | `dec_aq` | shipping A8 | A16 + fake-quant(grid) | Δdec |
//! | `both_aq` | A8 + fq | A16 + fq | report-only |
//! | `dec_a8` (A8 tier only) | shipping A8 | A16 + fq(A8) | the D1 control: \|Δ\| ≤ 2 items |
//!
//! # Gates implemented here
//!
//! - G-i1 (adapted; disclosed): knob-off counters EXACTLY 0 + byte-stable
//!   logits FNV across two identical knob-off probes (the feature-off BUILD
//!   half cannot exist for a feature-gated bin — the cfg blocks compile to
//!   nothing when off; recorded as the build-level disclosure).
//! - G-i2: per-phase counters == EXACT expected counts (prefill 256/chunk;
//!   decode 256·(n_gen−1)/item, computed from actual lengths); pf-only cell →
//!   decode counter 0 and vice versa; armed cells' FNV must differ from base.
//! - G-i4: the base cell run twice → byte-identical greedy streams.
//!
//! # Task axes (D4)
//!
//! - decode-heavy: 48 deterministic arithmetic-CoT items (4-shot completion,
//!   no chat template, greedy ≤ 256, LAST integer after `####`, truncation
//!   counts WRONG).
//! - prefill-heavy: multi-needle NIAH — 8 needles/prompt, ONE queried, 32
//!   prompts per length ∈ {4096, 8192, 16384}, greedy ≤ 32, FIRST
//!   needle-value substring in the output wins.
//!
//! # Verdicts (D4/D6): paired bootstrap 95% CI on Δdec − Δpf; HIT / REVERSED
//! / NULL; label precedence INSTRUMENT-FAIL > INADMISSIBLE > SATURATED >
//! HIT/REVERSED/NULL; admissibility 0.25 ≤ acc(base) ≤ 0.95 per length;
//! SATURATED ⟺ max(acc_pf, acc_dec) ≤ chance+0.05 OR
//! min(Δpf, Δdec) ≥ (acc_base − chance) − 0.05.
//!
//! 4090 lane (folded prefill is CUDA-only). GPU-EXCLUSIVE (the AGENTS rule):
//! refuses on co-resident compute apps. Env: `BONSAI_GGUF` · `DQ_OUT` ·
//! `DQ_GRIDS` (a2,a4) · `DQ_ARITH_N` (48) · `DQ_NI_LENGTHS` (4096,8192,16384)
//! · `DQ_NI_PER_LEN` (32) · `DQ_BOOTSTRAP` (10000) · `DQ_LANE_CHECK_ONLY` (1)
//! · `DQ_CELLS` (subset, comma; default all).

#![cfg(all(
    feature = "dq_phase_bench",
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    not(target_os = "macos"),
))]
#![cfg(not(debug_assertions))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{load_qwen_deltanet_ternary_weights_gguf, GgufFile};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::dq_fakequant::{self as dq, DqGrid, DqPhaseArm};
use riir_infer_gpu::ternary_deltanet_gpu_forward as tdf;
use riir_infer_gpu::{CubeCLContext, TernaryDeltanetGpuForward};

const DEFAULT_MODEL: &str = "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const EOS: usize = 248_046;

// ─── deterministic corpora (the seal is this file's git commit) ─────────────

/// Deterministic u64 xorshift (no RNG dep; identical everywhere).
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

/// The arithmetic-CoT item: (prompt_text, gold_answer).
/// 4-shot prefix + "Compute A op B op C … think step by step … ####".
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

fn arith_items(n: usize) -> Vec<ArithItem> {
    let mut rng = Rng(0xA71A_614_4847);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // 2-4 operations; multiplication operands kept small enough that a
        // 27B has a real but non-trivial shot (GSM8K-class difficulty).
        let n_ops = 2 + (rng.below(3) as usize);
        let mut terms: Vec<(i128, char)> = Vec::new();
        let mut expr = String::new();
        let first = 100 + rng.below(9900) as i128;
        expr.push_str(&first.to_string());
        let mut val = first;
        for _ in 0..n_ops {
            let op = match rng.below(4) {
                0 => '+',
                1 => '-',
                2 => '*',
                _ => {
                    // division only with exact results
                    let d = (2 + rng.below(40)) as i128;
                    if val % d == 0 && val / d != 0 {
                        expr.push_str(&format!(" / {d}"));
                        val /= d;
                        continue;
                    }
                    '+'
                }
            };
            let operand: i128 = match op {
                '*' => (3 + rng.below(97)) as i128,
                _ => (10 + rng.below(990)) as i128,
            };
            expr.push_str(&format!(" {op} {operand}"));
            val = match op {
                '+' => val + operand,
                '-' => val - operand,
                _ => val * operand,
            };
            let _ = &mut terms;
        }
        out.push(ArithItem {
            prompt: format!(
                "{ARITH_FEW_SHOT}Compute {expr}. Think step by step, then give the final answer after ####.\n"
            ),
            gold: val,
        });
    }
    out
}

/// Parse the model output: the LAST integer after `####` (commas stripped);
/// truncation (no `####`) counts WRONG (None).
fn parse_arith_answer(out: &str) -> Option<i128> {
    let idx = out.rfind("####")?;
    let tail = &out[idx + 4..];
    let cleaned: String = tail.chars().filter(|c| !c.is_whitespace() && *c != ',').collect();
    // take the leading run of digits/sign
    let mut num = String::new();
    for (i, c) in cleaned.char_indices() {
        if i == 0 && (c == '-') {
            num.push(c);
        } else if c.is_ascii_digit() {
            num.push(c);
        } else {
            break;
        }
    }
    num.parse().ok()
}

/// The natural-text paragraph bank (the qwen38_pyramid_capture pattern —
/// deterministic, inline, no data dependency; the haystack's BLAKE3 is
/// committed in the report).
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
    /// The queried server's code (the gold needle value).
    gold: String,
    /// ALL needle codes present (the distractors + gold — the first-match rule).
    all_codes: Vec<String>,
}

/// Build one NIAH prompt ≈ `target_tokens` long: interleaved paragraphs from
/// the deterministic bank with 8 needle sentences at spread depths, query last.
fn niah_item(idx: usize, len_round: usize, target_tokens: usize, tok: &BpeTokenizer) -> NiahItem {
    let mut rng = Rng(0x61A1_0000 + idx as u64 * 7919 + len_round as u64);
    // rotate needle assignment deterministically; 8 needles per prompt
    let needle_off = (idx * 3) % 10;
    let needles: Vec<(usize, usize)> = (0..8)
        .map(|k| {
            let s = (needle_off + k) % 10;
            let c = (s + idx + k) % 10;
            (s, c)
        })
        .collect();
    let target_k = ((idx * 5 + 2) % 8) as usize;
    let (t_server, t_code) = needles[target_k];

    // Build filler text and place needles at ~even depth bands.
    let mut parts: Vec<String> = Vec::new();
    let mut tokens_so_far = 0usize;
    let mut para_i = rng.below(24) as usize;
    let mut needle_i = 0usize;
    let needle_band = (target_tokens / 9).max(1);
    let mut next_needle_at = needle_band;
    loop {
        if needle_i < 8 && tokens_so_far >= next_needle_at {
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
    // any unplaced needles go at the end region
    while needle_i < 8 {
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

/// FIRST needle-value substring in the output wins (D4).
fn score_niah(out: &str, item: &NiahItem) -> bool {
    let mut first: Option<(usize, &str)> = None;
    for c in &item.all_codes {
        if let Some(p) = out.find(c.as_str()) {
            if first.is_none() || p < first.unwrap().0 {
                first = Some((p, c.as_str()));
            }
        }
    }
    matches!(&first, Some((_, c)) if *c == item.gold)
}

// ─── generation ─────────────────────────────────────────────────────────────

fn logits_fnv(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in v {
        let b = x.to_bits() as u64;
        h = (h ^ b).wrapping_mul(0x1000_0000_01b3);
    }
    h
}

struct GenOut {
    text: String,
    n_generated: usize, // INCLUDING the prefill-produced first token
    first_fnv: u64,
}

/// Greedy-generate up to `cap` tokens. The FIRST generated token comes from
/// the prefill's final logits (D1's phase boundary); each subsequent token is
/// a decode-lane step (set_input_token + forward_token).
fn greedy_generate(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &QwenDeltaNetTernaryWeights,
    tok: &BpeTokenizer,
    bos: usize,
    prompt: &str,
    cap: usize,
) -> GenOut {
    let mut tokens = tok.encode(prompt);
    if tokens.first() != Some(&bos) {
        tokens.insert(0, bos);
    }
    let logits = fwd.prefill(&tokens);
    let first_fnv = logits_fnv(&logits);
    // D1 phase boundary: generated token 1 comes from the prefill's final
    // logits; every later token is a decode-lane step (256 launches each).
    let mut out_ids: Vec<usize> = Vec::with_capacity(cap + 1);
    let mut next = argmax(&logits);
    out_ids.push(next);
    loop {
        if out_ids.len() >= cap || next == EOS {
            break;
        }
        fwd.set_input_token(weights, next);
        let l = fwd.forward_token();
        next = argmax(&l);
        out_ids.push(next);
    }
    GenOut {
        text: tok.decode(&out_ids),
        n_generated: out_ids.len(),
        first_fnv,
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv || (x == bv && false) {
            bv = x;
            bi = i;
        }
    }
    bi
}

// ─── statistics (D4: paired sign test; R reported only) ─────────────────────

/// Deterministic paired bootstrap 95% CI on the mean of per-item paired
/// differences `d_i = dec_correct_i − pf_correct_i` (in {−1, 0, +1}).
fn paired_bootstrap_ci(d: &[i8], n_boot: usize) -> (f64, f64) {
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = Rng(0xB005_614);
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

// ─── main ───────────────────────────────────────────────────────────────────

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[dq614] FATAL: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_path = env_or("BONSAI_GGUF", DEFAULT_MODEL);
    let out_dir = PathBuf::from(env_or("DQ_OUT", ".dq614"));
    let grids: Vec<DqGrid> = env_or("DQ_GRIDS", "a2,a4")
        .split(',')
        .filter_map(|s| match s.trim() {
            "a2" => Some(DqGrid::A2),
            "a4" => Some(DqGrid::A4),
            "a8" => Some(DqGrid::A8),
            _ => None,
        })
        .collect();
    if grids.is_empty() {
        return Err("DQ_GRIDS produced no grids".into());
    }
    let arith_n: usize = env_or("DQ_ARITH_N", "48").parse().map_err(|_| "DQ_ARITH_N")?;
    let ni_lengths: Vec<usize> = env_or("DQ_NI_LENGTHS", "4096,8192,16384")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let ni_per_len: usize = env_or("DQ_NI_PER_LEN", "32").parse().map_err(|_| "DQ_NI_PER_LEN")?;
    let n_boot: usize = env_or("DQ_BOOTSTRAP", "10000")
        .parse()
        .map_err(|_| "DQ_BOOTSTRAP")?;
    let lane_check_only = env_or("DQ_LANE_CHECK_ONLY", "0") == "1";

    // ── GPU exclusivity (the AGENTS rule; the pycap probe) ────────────────
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
            !mem.is_empty()
                && mem != "[N/A]"
                && mem.trim_end_matches(" MiB").parse::<u64>().is_ok_and(|m| m > 0)
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

    // ── model ─────────────────────────────────────────────────────────────
    eprintln!("[dq614] loading {model_path} …");
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(Path::new(&model_path))
        .map_err(|e| format!("load: {e}"))?;
    let gguf = GgufFile::open(Path::new(&model_path)).map_err(|e| format!("reopen: {e}"))?;
    let tok = BpeTokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
    let bos = config.bos_token;
    let max_len = *ni_lengths.last().unwrap_or(&4096);
    config.block_size = (max_len + 64).max(config.block_size.min(32768));
    let ctx = CubeCLContext::new().map_err(|e| format!("GPU init: {e}"))?;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    eprintln!("[dq614] forward ready ({:.0}s)", t0.elapsed().as_secs());

    // The shipping knob posture (bench_947/948's pin set) + the CUDA prefill
    // arm. The setters live on their defining modules in this crate.
    tdf::set_prefill_use_gemv(false);
    tdf::set_prefill_seq_rmsnorm(false);
    tdf::set_prefill_zero_scratch(false);
    tdf::set_prefill_batch_elementwise(true);
    tdf::set_prefill_use_cmma16(true);
    tdf::set_prefill_use_cmma_i8(true);
    tdf::set_prefill_cmma_i8_psplit(true);
    riir_infer_gpu::set_prefill_use_cuda_mma(false);
    riir_infer_gpu::set_prefill_use_cuda_ffn(false);
    tdf::set_prefill_chunk_max(4096);
    riir_infer_gpu::prefill_cuda_full::set_prefill_use_cuda(true);
    // D5: graphs OFF in every cell (the knob must be consulted per launch).
    std::env::set_var("RIIR_PREFILL_CUDA_GRAPHS", "0");

    // ── corpora + freeze hashes ───────────────────────────────────────────
    let arith = arith_items(arith_n);
    let niah: Vec<(usize, Vec<NiahItem>)> = ni_lengths
        .iter()
        .enumerate()
        .map(|(r, &l)| {
            (
                l,
                (0..ni_per_len)
                    .map(|i| niah_item(i, r, l, &tok))
                    .collect(),
            )
        })
        .collect();

    let corpus_hash = {
        let mut h = blake3::Hasher::new();
        for a in &arith {
            h.update(a.prompt.as_bytes());
            h.update(&a.gold.to_le_bytes());
        }
        for (_, items) in &niah {
            for it in items {
                h.update(it.prompt.as_bytes());
                h.update(it.gold.as_bytes());
            }
        }
        h.finalize().to_hex().to_string()
    };
    let model_hash = {
        // streaming hash (7 GB — one pass)
        let mut f = std::fs::File::open(&model_path).map_err(|e| e.to_string())?;
        let mut h = blake3::Hasher::new();
        std::io::copy(&mut f, &mut h).map_err(|e| e.to_string())?;
        h.finalize().to_hex().to_string()
    };
    eprintln!("[dq614] corpus blake3={corpus_hash} model blake3={model_hash}");

    // The lane record (D1's LOCK — decided before any accuracy cell):
    // the non-int8 folded prefill lane was measured NOT TO EXIST at T2; the
    // fallback is operative and this record is frozen here.
    eprintln!(
        "[dq614] LANE RECORD (frozen before any accuracy cell): fallback — \
         shipping A8 prefill in every cell + dec_a8 control (no non-int8 \
         folded prefill lane exists; T2 measurement)"
    );

    // ── the cells ─────────────────────────────────────────────────────────
    struct CellResult {
        arith_correct: Vec<bool>,
        ni_correct: BTreeMap<usize, Vec<bool>>,
        gen_lens: Vec<usize>,
        prompt_toks: Vec<usize>,
        first_fnvs: Vec<u64>,
        prefill_launches: u64,
        decode_launches: u64,
        /// G-i2: the EXACT expected counts, computed from actual lengths.
        expected_prefill: u64,
        expected_decode: u64,
        count_ok: bool,
    }
    let mut cells: BTreeMap<String, CellResult> = BTreeMap::new();

    let weights_ref = &weights;
    let run_cell = |name: &str,
                    arm: DqPhaseArm,
                    grid: DqGrid,
                    fwd: &mut TernaryDeltanetGpuForward|
     -> Result<CellResult, String> {
        dq::set_arm(arm);
        dq::set_grid(grid);
        dq::fq_reset_counters();
        let p0 = dq::fq_prefill_launches();
        let d0 = dq::fq_decode_launches();
        let mut arith_correct = Vec::with_capacity(arith.len());
        let mut gen_lens = Vec::new();
        let mut first_fnvs = Vec::new();
        let mut prompt_toks = Vec::new();
        for a in &arith {
            let g = greedy_generate(fwd, weights_ref, &tok, bos, &a.prompt, 256);
            let ok = parse_arith_answer(&g.text) == Some(a.gold);
            arith_correct.push(ok);
            gen_lens.push(g.n_generated);
            first_fnvs.push(g.first_fnv);
            prompt_toks.push(tok.encode(&a.prompt).len() + 1);
            eprintln!(
                "[dq614] {name} arith gold={} out={:?} ok={ok}",
                a.gold,
                g.text.chars().take(80).collect::<String>()
            );
        }
        let mut ni_correct = BTreeMap::new();
        for (l, items) in &niah {
            let mut v = Vec::with_capacity(items.len());
            for it in items {
                let g = greedy_generate(fwd, weights_ref, &tok, bos, &it.prompt, 32);
                let ok = score_niah(&g.text, it);
                v.push(ok);
                gen_lens.push(g.n_generated);
                first_fnvs.push(g.first_fnv);
                prompt_toks.push(tok.encode(&it.prompt).len() + 1);
            }
            ni_correct.insert(*l, v);
            eprintln!("[dq614] {name} niah@{l}: {}/{}", ni_correct[&l].iter().filter(|x| **x).count(), items.len());
        }
        // G-i2 (D5, frozen arithmetic): prefill = 256/chunk where chunks =
        // ceil(prompt_toks/4096); decode = 256 * (n_generated − 1) — token 1
        // is the prefill's (the D1 phase boundary).
        let armed_pf = arm.prefill_armed();
        let armed_dec = arm.decode_armed();
        let mut expected_prefill = 0u64;
        let mut expected_decode = 0u64;
        for (&pt, &ng) in prompt_toks.iter().zip(gen_lens.iter()) {
            let chunks = pt.div_ceil(4096).max(1) as u64;
            if armed_pf {
                expected_prefill += 256 * chunks;
            }
            if armed_dec {
                expected_decode += 256 * ng.saturating_sub(1) as u64;
            }
        }
        let got_pf = dq::fq_prefill_launches() - p0;
        let got_dec = dq::fq_decode_launches() - d0;
        let count_ok = got_pf == expected_prefill && got_dec == expected_decode;
        if !count_ok {
            eprintln!(
                "[dq614] G-i2 COUNT MISMATCH {name}: prefill {got_pf} vs exp {expected_prefill}, decode {got_dec} vs exp {expected_decode}"
            );
        }
        Ok(CellResult {
            arith_correct,
            ni_correct,
            gen_lens,
            prompt_toks,
            first_fnvs,
            prefill_launches: got_pf,
            decode_launches: got_dec,
            expected_prefill,
            expected_decode,
            count_ok,
        })
    };

    // G-i4 + G-i1(knob-off): base run twice.
    dq::set_arm(DqPhaseArm::Off);
    dq::fq_reset_counters();
    let base1 = run_cell("base(1)", DqPhaseArm::Off, DqGrid::A2, &mut fwd)?;
    let base2 = run_cell("base(2)", DqPhaseArm::Off, DqGrid::A2, &mut fwd)?;
    let gi1_counters_zero = base1.prefill_launches == 0 && base1.decode_launches == 0;
    let gi4_stable = base1.first_fnvs == base2.first_fnvs
        && base1.arith_correct == base2.arith_correct
        && base1.ni_correct == base2.ni_correct;
    if !gi1_counters_zero {
        return Err("G-i1 FAIL: knob-off run dispatched fake-quant passes".into());
    }
    if !gi4_stable {
        return Err("G-i4 FAIL: base cell not deterministic across runs".into());
    }
    cells.insert("base".into(), base1);
    let _ = base2;

    if lane_check_only {
        let acc = cells["base"].arith_correct.iter().filter(|x| **x).count() as f64
            / arith_n as f64;
        eprintln!("[dq614] LANE-CHECK: base arith acc = {acc:.4} (window 0.25–0.95)");
        println!("lane_check_base_acc={acc:.4}");
        return Ok(());
    }

    for grid in &grids {
        let gname = format!("{grid:?}").to_lowercase();
        let pf = run_cell(&format!("pf_aq[{gname}]"), DqPhaseArm::PrefillOnly, *grid, &mut fwd)?;
        let dec = run_cell(&format!("dec_aq[{gname}]"), DqPhaseArm::DecodeOnly, *grid, &mut fwd)?;
        let both = run_cell(&format!("both_aq[{gname}]"), DqPhaseArm::Both, *grid, &mut fwd)?;
        cells.insert(format!("pf_aq.{gname}"), pf);
        cells.insert(format!("dec_aq.{gname}"), dec);
        cells.insert(format!("both_aq.{gname}"), both);
        if *grid == DqGrid::A4 {
            // The D1 control: A8-on-decode vs base (|Δ| ≤ 2 items).
            let c = run_cell("dec_a8", DqPhaseArm::DecodeOnly, DqGrid::A8, &mut fwd)?;
            cells.insert("dec_a8".into(), c);
        }
    }
    dq::set_arm(DqPhaseArm::Off);

    // ── G-i2 gate: every cell's counts must have matched ──────────────────
    for (name, c) in &cells {
        if !c.count_ok {
            return Err(format!(
                "G-i2 FAIL ({name}): prefill {} vs exp {}, decode {} vs exp {}",
                c.prefill_launches, c.expected_prefill, c.decode_launches, c.expected_decode
            ));
        }
    }
    // Positive control: every armed cell's first arith FNV must differ from
    // base's at some item (the vacuous-guard law).
    for (name, c) in &cells {
        if name != "base" && name != "dec_a8" {
            let differs = c
                .first_fnvs
                .iter()
                .zip(cells["base"].first_fnvs.iter())
                .any(|(a, b)| a != b);
            if !differs {
                return Err(format!("G-i2 positive control FAIL: {name} logits identical to base"));
            }
        }
    }

    // ── analysis + report ──────────────────────────────────────────────────
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let mut md = String::new();
    md.push_str("# DQ phase-matrix run (Plan 614 / Issue 026) — RAW DUMP\n\n");
    md.push_str(&format!(
        "- model: `{model_path}` blake3=`{model_hash}`\n- corpus blake3: `{corpus_hash}`\n- gpu: {gpu_csv}\n- compute apps at start: none with dedicated memory\n- grids: {grids:?}\n- arith_n: {arith_n}, ni_lengths: {ni_lengths:?}, ni_per_len: {ni_per_len}\n- bootstrap: {n_boot}\n\n"
    ));
    md.push_str("## G-i1/G-i4\n\n- G-i1 (knob-off counters == 0): PASS\n- G-i4 (base twice byte-stable): PASS\n\n");
    md.push_str("## Per-cell accuracy + counters\n\n");
    md.push_str("| cell | arith acc | nih acc per len | prefill launches | decode launches |\n|---|---|---|---|---|\n");
    for (name, c) in &cells {
        let a = c.arith_correct.iter().filter(|x| **x).count() as f64 / c.arith_correct.len() as f64;
        let nis: Vec<String> = c
            .ni_correct
            .iter()
            .map(|(l, v)| format!("{l}: {:.3}", v.iter().filter(|x| **x).count() as f64 / v.len() as f64))
            .collect();
        md.push_str(&format!(
            "| {name} | {a:.4} | {} | {} | {} |\n",
            nis.join(", "),
            c.prefill_launches,
            c.decode_launches
        ));
    }

    // paired stats per grid
    md.push_str("\n## Paired stats (Δdec − Δpf per item; bootstrap 95% CI)\n\n");
    let mut verdicts: BTreeMap<String, String> = BTreeMap::new();
    for grid in &grids {
        let gname = format!("{grid:?}").to_lowercase();
        let pf = &cells[&format!("pf_aq.{gname}")];
        let dec = &cells[&format!("dec_aq.{gname}")];
        let base = &cells["base"];
        // decode-heavy
        let d: Vec<i8> = dec
            .arith_correct
            .iter()
            .zip(pf.arith_correct.iter())
            .map(|(a, b)| (*a as i8) - (*b as i8))
            .collect();
        let (lo, hi) = paired_bootstrap_ci(&d, n_boot);
        let acc = |c: &Vec<bool>| c.iter().filter(|x| **x).count() as f64 / c.len() as f64;
        let chance = 0.05f64;
        let base_acc = acc(&base.arith_correct);
        let pf_acc = acc(&pf.arith_correct);
        let dec_acc = acc(&dec.arith_correct);
        let dpf = base_acc - pf_acc;
        let ddec = base_acc - dec_acc;
        let label = if base_acc < 0.25 || base_acc > 0.95 {
            "INADMISSIBLE".to_string()
        } else if pf_acc.max(dec_acc) <= chance + 0.05
            || dpf.min(ddec) >= (base_acc - chance) - 0.05
        {
            "SATURATED".to_string()
        } else if lo > 0.0 {
            "HIT".to_string()
        } else if hi < 0.0 {
            "REVERSED".to_string()
        } else {
            "NULL".to_string()
        };
        verdicts.insert(format!("{gname}.decode_heavy"), label.clone());
        md.push_str(&format!(
            "- **{gname} decode-heavy**: acc base={base_acc:.4} pf={pf_acc:.4} dec={dec_acc:.4}; Δpf={dpf:.4} Δdec={ddec:.4}; R={:.3}; CI(Δdec−Δpf)=[{lo:.4},{hi:.4}] → **{label}**\n",
            if dpf > 0.0 { ddec / dpf } else { f64::NAN }
        ));
        // prefill-heavy: pooled over admissible lengths
        let mut dd_ni: Vec<i8> = Vec::new();
        let mut pooled: Vec<(bool, bool, bool)> = Vec::new(); // (base, pf, dec)
        for (l, bv) in &base.ni_correct {
            let pv = &pf.ni_correct[l];
            let dv = &dec.ni_correct[l];
            let ba = acc(bv);
            for i in 0..bv.len() {
                dd_ni.push(dv[i] as i8 - pv[i] as i8);
                pooled.push((bv[i], pv[i], dv[i]));
            }
            let _ = ba;
        }
        let (lo2, hi2) = paired_bootstrap_ci(&dd_ni, n_boot);
        let base_all: Vec<bool> = pooled.iter().map(|x| x.0).collect();
        let pf_all: Vec<bool> = pooled.iter().map(|x| x.1).collect();
        let dec_all: Vec<bool> = pooled.iter().map(|x| x.2).collect();
        let base_acc_n = acc(&base_all);
        let pf_acc_n = acc(&pf_all);
        let dec_acc_n = acc(&dec_all);
        let dpf_n = base_acc_n - pf_acc_n;
        let ddec_n = base_acc_n - dec_acc_n;
        let chance_n = 0.125f64;
        let label_n = if base_acc_n < 0.25 || base_acc_n > 0.95 {
            "INADMISSIBLE".to_string()
        } else if pf_acc_n.max(dec_acc_n) <= chance_n + 0.05
            || dpf_n.min(ddec_n) >= (base_acc_n - chance_n) - 0.05
        {
            "SATURATED".to_string()
        } else if lo2 < 0.0 {
            "HIT".to_string()
        } else if hi2 > 0.0 {
            "REVERSED".to_string()
        } else {
            "NULL".to_string()
        };
        verdicts.insert(format!("{gname}.prefill_heavy"), label_n.clone());
        md.push_str(&format!(
            "- **{gname} prefill-heavy (pooled)**: acc base={base_acc_n:.4} pf={pf_acc_n:.4} dec={dec_acc_n:.4}; Δpf={dpf_n:.4} Δdec={ddec_n:.4}; R={:.3}; CI=[{lo2:.4},{hi2:.4}] → **{label_n}**\n",
            if dpf_n > 0.0 { ddec_n / dpf_n } else { f64::NAN }
        ));
    }
    // dec_a8 control
    if let Some(c8) = cells.get("dec_a8") {
        let base = &cells["base"];
        let delta = (base.arith_correct.iter().filter(|x| **x).count() as i64
            - c8.arith_correct.iter().filter(|x| **x).count() as i64)
            .abs();
        md.push_str(&format!(
            "\n## dec_a8 control (D1 fallback): |Δarith| = {delta} items (gate ≤ 2) → {}\n",
            if delta <= 2 { "PASS" } else { "INSTRUMENT-FAIL" }
        ));
        if delta > 2 {
            return Err("dec_a8 control FAILED: |Δ| > 2 items".into());
        }
    }
    md.push_str("\n## Verdicts\n\n");
    for (k, v) in &verdicts {
        md.push_str(&format!("- {k}: **{v}**\n"));
    }

    let md_path = out_dir.join("dq_phase_matrix.md");
    std::fs::write(&md_path, &md).map_err(|e| e.to_string())?;
    eprintln!("[dq614] report: {}", md_path.display());
    Ok(())
}
