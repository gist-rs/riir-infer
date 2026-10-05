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
//! · `DQ_NI_PER_LEN` (32) · `DQ_BOOTSTRAP` (10000) · `DQ_LANE_CHECK_ONLY` (1). Issue 031 repro axes
//! (no-op by default, never set by the runner): `DQ614_FORCE_FATAL=1` FATALs
//! after the G-i1/G-i4 base cells (both GPU stacks warm) · `DQ614_EXIT_PLAIN=1`
//! takes the v1 plain-exit error path instead of hard_exit.
//! (The header's `DQ_CELLS` subset knob was documented but never implemented;
//! smokes narrow via `DQ_GRIDS` + the N knobs.)

#![cfg(all(
    feature = "dq_phase_bench",
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    not(target_os = "macos"),
))]
// Dev profile: the refusing main makes the measurement body dead code BY
// DESIGN (it must never run unoptimised) — silence exactly that class, in
// exactly that profile. Release keeps every warning live.
#![cfg_attr(debug_assertions, allow(dead_code))]

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

fn arith_items(n: usize, hard: bool) -> Vec<ArithItem> {
    arith_items_v2(n, hard)
}

/// Precedence-correct gold: × binds before +/-; each class evaluates
/// left-to-right. Returns `None` on overflow (checked) — the caller skips
/// such items (none occur at these operand scales; the check is a guard).
fn eval_standard_precedence(first: i128, ops: &[(char, i128)]) -> Option<i128> {
    // Pass 1: fold the multiplicative runs into term values.
    let mut terms: Vec<(char, i128)> = Vec::with_capacity(ops.len() + 1); // (pending_addop, value)
    let mut product = first;
    let mut pending: char = '+'; // the additive op that will join `product` into the sum
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
    // Pass 2: additive, left-to-right.
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

/// v2 (the Issue-030 lane-check fix): v1 accumulated the gold LEFT-TO-RIGHT
/// while the model — and the few-shot examples themselves — apply standard
/// ×-before-+/- precedence, so every mixed-precedence item was mislabeled
/// (measured: the model answering its own prompt correctly while gold
/// disagreed; base acc ≈ the precedence-neutral fraction). v2 generates the
/// same operator/operand stream but takes the gold from
/// `eval_standard_precedence`, so gold == what the convention the shots teach
/// actually computes. Division is dropped: keeping generated divisions exact
/// under precedence adds a retry lane for a class the shots already cover;
/// +/-/* suffice for the phase-sensitivity axis. The corpus hash changes —
/// disclosed in the run record (the stop rule's instrument-defect clause; no
/// accuracy cell was admissible before this fix).
///
/// `hard` (DQ_ARITH_HARD=1, Issue 033): the v2 operand scales read base
/// arith 0.9583 at n=48 — ABOVE the 0.95 admissibility ceiling, so no Δ can
/// gate (bench 023). The hard posture widens every scale (ops 3-6, first
/// 1_000-99_999, +/- operands 100-9_999, × operands 11-999) to pull base
/// into the window; the frozen default (hard=false) is BYTE-IDENTICAL to
/// the v2 corpus the earlier runs measured.
fn arith_items_v2(n: usize, hard: bool) -> Vec<ArithItem> {
    let mut rng = Rng(0x0A71_A614_4847);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // 2-4 operations; multiplication operands kept small enough that a
        // 27B has a real but non-trivial shot (GSM8K-class difficulty).
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

/// Parse the model output: the LAST integer after `####` (commas stripped);
/// truncation (no `####`) counts WRONG (None).
fn parse_arith_answer(out: &str) -> Option<i128> {
    let idx = out.rfind("####")?;
    let tail = &out[idx + 4..];
    let cleaned: String = tail.chars().filter(|c| !c.is_whitespace() && *c != ',').collect();
    // take the leading run of digits/sign
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
/// the deterministic bank with `nn` needle sentences at spread depths, query last.
/// `nn` (DQ_NI_NEEDLES, Issue 033): the default 8 read base pooled 0.9896 —
/// ABOVE the 0.95 admissibility ceiling; more distractors pull base down.
/// Capped at the 10-entry needle banks.
fn niah_item(idx: usize, len_round: usize, target_tokens: usize, tok: &BpeTokenizer, nn: usize) -> NiahItem {
    let nn = nn.clamp(1, NEEDLE_SERVERS.len());
    let mut rng = Rng(0x61A1_0000 + idx as u64 * 7919 + len_round as u64);
    // rotate needle assignment deterministically; nn needles per prompt
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

    // Build filler text and place needles at ~even depth bands.
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
    // any unplaced needles go at the end region
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

/// FIRST needle-value substring in the output wins (D4).
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
    /// FNV of the FIRST DECODE-LANE logits (token 2's) — 0 when generation
    /// stopped at the prefill token. The decode-side positive-control
    /// signal: a decode-only fake-quant arm cannot change `first_fnv` (the
    /// prefill computes it BEFORE any decode step — the G-i2 control as
    /// first written compared prefill logits for decode-armed cells and was
    /// structurally unsatisfiable, FATALing the completed matrix at the
    /// final gate).
    second_fnv: u64,
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
    // Every item is an INDEPENDENT sequence: the GDN recurrent state carries
    // IN PLACE across prefill calls (never self-zeroing), so without this
    // reset every item after the first starts from the previous item's final
    // state — degradation ACCUMULATES across the run (measured: arith fell
    // from 4/5 early to 2/43 late within base(1); base(2)'s niah@4096 —
    // items 97+, after 96 accumulated items — collapsed to 12/32 vs
    // base(1)'s 27/32; the shot-regeneration outputs were state pollution,
    // not model behavior). The attention KV needs no zeroing — decode bounds
    // reads to pos+1 and every touched position is overwritten before read
    // (reset_state's own doc).
    fwd.reset_state();
    let logits = fwd.prefill(&tokens);
    let first_fnv = logits_fnv(&logits);
    let mut second_fnv: u64 = 0;
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
        if second_fnv == 0 {
            second_fnv = logits_fnv(&l);
        }
        next = argmax(&l);
        out_ids.push(next);
    }
    GenOut {
        text: tok.decode(&out_ids),
        n_generated: out_ids.len(),
        first_fnv,
        second_fnv,
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

// ─── main ───────────────────────────────────────────────────────────────────

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// Release-only measurement: a debug build would time an unoptimised
/// binary, so the dev-profile arm is a LOUD refusal (compile_error is
/// E0601-adjacent noise; a refusing main keeps `--all-features` dev
/// builds valid while refusing to measure).
#[cfg(debug_assertions)]
fn main() {
    eprintln!(
        "[dq614] REFUSED: release-only measurement — rebuild with \
         `cargo build --release --features dq_phase_bench,ternary_gemv_cuda_raw,ternary_gemm_batched`"
    );
    std::process::exit(2);
}

#[cfg(not(debug_assertions))]
fn main() {
    if let Err(e) = run() {
        eprintln!("[dq614] FATAL: {e}");
        // Issue 031 repro axis: DQ614_EXIT_PLAIN=1 restores the v1 error
        // path (std::process::exit — the hang site, by elimination) for the
        // teardown A/B; the default stays hard_exit (the landed defense).
        if std::env::var("DQ614_EXIT_PLAIN").as_deref() == Ok("1") {
            eprintln!("[dq614] DQ614_EXIT_PLAIN=1 — taking the v1 plain-exit path (repro axis)");
            std::process::exit(1);
        }
        hard_exit(1);
    }
}

/// The error path must not trust the CRT exit: the v1 matrix FATALed and
/// then HUNG >10 min inside `std::process::exit` (Issue 031 — the CUDA
/// driver's detach-time context cleanup blocks under the loader lock; RAM
/// climbed 4.7→8.8 GB and the process had to be killed by hand, which is
/// the trap that made live runs look like zombies). Flush what we have,
/// then terminate hard: TerminateProcess on SELF skips atexit AND
/// DLL_PROCESS_DETACH, so no handler can block the exit.
fn hard_exit(code: i32) -> ! {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            code as u32,
        );
    }
    std::process::exit(code)
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
    let arith_hard = env_or("DQ_ARITH_HARD", "0") == "1";
    let ni_lengths: Vec<usize> = env_or("DQ_NI_LENGTHS", "4096,8192,16384")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let ni_per_len: usize = env_or("DQ_NI_PER_LEN", "32").parse().map_err(|_| "DQ_NI_PER_LEN")?;
    let ni_needles: usize = env_or("DQ_NI_NEEDLES", "8").parse().map_err(|_| "DQ_NI_NEEDLES")?;
    // Issue 033 — the KV-STORE axis cells. Empty string = no KV cells (the
    // axis-1-only replication posture); "off" entries are skipped like
    // unknown spellings. Default a8,a4: the q8kv-class cell + one sensitivity
    // rung below it.
    let kv_grids: Vec<DqGrid> = env_or("DQ_KV_GRIDS", "a8,a4")
        .split(',')
        .filter_map(|s| match s.trim() {
            "a2" => Some(DqGrid::A2),
            "a4" => Some(DqGrid::A4),
            "a8" => Some(DqGrid::A8),
            _ => None,
        })
        .collect();
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
    // Issue 864's own advice, CLAMP-DOWN (the first lane check OOM'd at
    // NIAH entry with the v1 form — `(max_len+64).max(block.min(32768))`,
    // which keeps the model's full 32768 block (4 GiB KV reservation) and
    // never clamps: `.max` selects the LARGER. The bench's real sequence
    // bound is the longest NIAH prompt + generation headroom; clamping to it
    // halves the attention KV working set (32768 → 16448 slots).
    let max_len = *ni_lengths.last().unwrap_or(&4096);
    config.block_size = config.block_size.min(max_len + 64);
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
    // The env is read once via OnceLock — set it in the INVOCATION, not here
    // (an in-main set_var can race the first read); the runner refuses to
    // run without it.
    if std::env::var("RIIR_PREFILL_CUDA_GRAPHS").ok().as_deref() != Some("0") {
        return Err(
            "D5: run with RIIR_PREFILL_CUDA_GRAPHS=0 (graphs would freeze the \
             fake-quant knob into a capture)"
                .into(),
        );
    }

    // ── corpora + freeze hashes ───────────────────────────────────────
    let arith = arith_items(arith_n, arith_hard);
    let niah: Vec<(usize, Vec<NiahItem>)> = ni_lengths
        .iter()
        .enumerate()
        .map(|(r, &l)| {
            (
                l,
                (0..ni_per_len)
                    .map(|i| niah_item(i, r, l, &tok, ni_needles))
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

    // ── the cells ────────────────────────────────────────────────
    // Issue 033 — the attention-layer count (the KV axis fires per
    // ATTENTION layer; the GDN layers carry recurrent state, no KV cache).
    let attn_layers = weights
        .layer_types
        .iter()
        .filter(|t| **t != riir_infer_core::types::DeltaNetLayerType::DeltaNet)
        .count();
    eprintln!("[dq614] attn_layers={attn_layers} (KV axis fires per attention layer)");
    struct CellResult {
        arith_correct: Vec<bool>,
        ni_correct: BTreeMap<usize, Vec<bool>>,
        first_fnvs: Vec<u64>,
        second_fnvs: Vec<u64>,
        prefill_launches: u64,
        decode_launches: u64,
        kv_launches: u64,
        /// G-i2: the EXACT expected counts, computed from actual lengths.
        expected_prefill: u64,
        expected_decode: u64,
        expected_kv: u64,
        count_ok: bool,
    }
    let mut cells: BTreeMap<String, CellResult> = BTreeMap::new();

    let weights_ref = &weights;
    let run_cell = |name: &str,
                    arm: DqPhaseArm,
                    grid: DqGrid,
                    kv_grid: Option<DqGrid>,
                    fwd: &mut TernaryDeltanetGpuForward|
     -> Result<CellResult, String> {
        dq::set_arm(arm);
        dq::set_grid(grid);
        dq::set_kv_grid(kv_grid);
        dq::fq_reset_counters();
        let p0 = dq::fq_prefill_launches();
        let d0 = dq::fq_decode_launches();
        let k0 = dq::fq_kv_launches();
        let mut arith_correct = Vec::with_capacity(arith.len());
        let mut gen_lens = Vec::new();
        let mut first_fnvs = Vec::new();
        let mut second_fnvs = Vec::new();
        let mut prompt_toks = Vec::new();
        for a in &arith {
            let g = greedy_generate(fwd, weights_ref, &tok, bos, &a.prompt, 256);
            let ok = parse_arith_answer(&g.text) == Some(a.gold);
            arith_correct.push(ok);
            gen_lens.push(g.n_generated);
            first_fnvs.push(g.first_fnv);
            second_fnvs.push(g.second_fnv);
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
                second_fnvs.push(g.second_fnv);
                prompt_toks.push(tok.encode(&it.prompt).len() + 1);
            }
            ni_correct.insert(*l, v);
            eprintln!("[dq614] {name} niah@{l}: {}/{}", ni_correct[l].iter().filter(|x| **x).count(), items.len());
        }
        // G-i2 (D5, frozen arithmetic): prefill = 256/chunk where chunks =
        // ceil(prompt_toks/4096); decode = 256 × (n_generated − 1) — token 1
        // is the prefill's (the D1 phase boundary). Issue 033: KV = 2 (k+v)
        // × attn_layers per prefill chunk and per decode step, whenever the
        // KV arm is set.
        let armed_pf = arm.prefill_armed();
        let armed_dec = arm.decode_armed();
        let per_item_kv = 2u64 * attn_layers as u64;
        let mut expected_prefill = 0u64;
        let mut expected_decode = 0u64;
        let mut expected_kv = 0u64;
        for (&pt, &ng) in prompt_toks.iter().zip(gen_lens.iter()) {
            let chunks = pt.div_ceil(4096).max(1) as u64;
            if armed_pf {
                expected_prefill += 256 * chunks;
            }
            if armed_dec {
                expected_decode += 256 * ng.saturating_sub(1) as u64;
            }
            if kv_grid.is_some() {
                expected_kv += per_item_kv * chunks + per_item_kv * ng.saturating_sub(1) as u64;
            }
        }
        let got_pf = dq::fq_prefill_launches() - p0;
        let got_dec = dq::fq_decode_launches() - d0;
        let got_kv = dq::fq_kv_launches() - k0;
        let count_ok =
            got_pf == expected_prefill && got_dec == expected_decode && got_kv == expected_kv;
        if !count_ok {
            eprintln!(
                "[dq614] G-i2 COUNT MISMATCH {name}: prefill {got_pf} vs exp {expected_prefill}, decode {got_dec} vs exp {expected_decode}, kv {got_kv} vs exp {expected_kv}"
            );
        }
        Ok(CellResult {
            arith_correct,
            ni_correct,
            first_fnvs,
            second_fnvs,
            prefill_launches: got_pf,
            decode_launches: got_dec,
            kv_launches: got_kv,
            expected_prefill,
            expected_decode,
            expected_kv,
            count_ok,
        })
    };

    // G-i4 + G-i1(knob-off): base run twice.
    dq::set_arm(DqPhaseArm::Off);
    dq::set_kv_grid(None);
    dq::fq_reset_counters();
    let base1 = run_cell("base(1)", DqPhaseArm::Off, DqGrid::A2, None, &mut fwd)?;
    let base2 = run_cell("base(2)", DqPhaseArm::Off, DqGrid::A2, None, &mut fwd)?;
    let gi1_counters_zero = base1.prefill_launches == 0
        && base1.decode_launches == 0
        && base1.kv_launches == 0;
    let gi4_stable = base1.first_fnvs == base2.first_fnvs
        && base1.second_fnvs == base2.second_fnvs
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

    // Issue 031 T-root-cause: the error-path repro knob (no-op by default).
    // With DQ614_FORCE_FATAL=1 the run FATALs here — both GPU stacks (cudarc
    // + CubeCL) warm, real launches done, the exact v1 error-path shape —
    // and main routes the FATAL per DQ614_EXIT_PLAIN (old path) or the
    // default hard_exit (the landed defense). A/B instrument, never set by
    // the runner.
    if env_or("DQ614_FORCE_FATAL", "0") == "1" {
        return Err("forced FATAL (Issue 031 repro: DQ614_FORCE_FATAL=1)".into());
    }

    if lane_check_only {
        let acc = cells["base"].arith_correct.iter().filter(|x| **x).count() as f64
            / arith_n as f64;
        let ni_pooled: Vec<bool> = cells["base"]
            .ni_correct
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        let ni_acc = ni_pooled.iter().filter(|x| **x).count() as f64 / ni_pooled.len() as f64;
        eprintln!(
            "[dq614] LANE-CHECK: base arith acc = {acc:.4}, niah pooled = {ni_acc:.4} (window 0.25–0.95)"
        );
        println!("lane_check_base_acc={acc:.4}");
        println!("lane_check_ni_pooled_acc={ni_acc:.4}");
        return Ok(());
    }

    for grid in &grids {
        let gname = format!("{grid:?}").to_lowercase();
        let pf = run_cell(
            &format!("pf_aq[{gname}]"),
            DqPhaseArm::PrefillOnly,
            *grid,
            None,
            &mut fwd,
        )?;
        let dec = run_cell(
            &format!("dec_aq[{gname}]"),
            DqPhaseArm::DecodeOnly,
            *grid,
            None,
            &mut fwd,
        )?;
        let both = run_cell(
            &format!("both_aq[{gname}]"),
            DqPhaseArm::Both,
            *grid,
            None,
            &mut fwd,
        )?;
        cells.insert(format!("pf_aq.{gname}"), pf);
        cells.insert(format!("dec_aq.{gname}"), dec);
        cells.insert(format!("both_aq.{gname}"), both);
        if *grid == DqGrid::A4 {
            // The D1 control: A8-on-decode vs base (|Δ| ≤ 2 items).
            let c = run_cell("dec_a8", DqPhaseArm::DecodeOnly, DqGrid::A8, None, &mut fwd)?;
            cells.insert("dec_a8".into(), c);
        }
    }
    dq::set_arm(DqPhaseArm::Off);

    // Issue 033 — the KV-STORE axis cells: kv-only (activation arm Off) and
    // the pf×kv interaction (prefill activation quant + KV store quant,
    // decode clean) per KV grid. The interaction cell separates "damage
    // composes additively" from "the axes overlap in one mechanism".
    for grid in &kv_grids {
        let gname = format!("{grid:?}").to_lowercase();
        let kv = run_cell(
            &format!("kv_aq[{gname}]"),
            DqPhaseArm::Off,
            DqGrid::A2,
            Some(*grid),
            &mut fwd,
        )?;
        cells.insert(format!("kv_aq.{gname}"), kv);
        let pfkv = run_cell(
            &format!("pfkv_aq[{gname}]"),
            DqPhaseArm::PrefillOnly,
            *grid,
            Some(*grid),
            &mut fwd,
        )?;
        cells.insert(format!("pfkv_aq.{gname}"), pfkv);
    }
    dq::set_kv_grid(None);

    // ── G-i2 gate: every cell's counts must have matched ──────────────
    for (name, c) in &cells {
        if !c.count_ok {
            return Err(format!(
                "G-i2 FAIL ({name}): prefill {} vs exp {}, decode {} vs exp {}, kv {} vs exp {}",
                c.prefill_launches,
                c.expected_prefill,
                c.decode_launches,
                c.expected_decode,
                c.kv_launches,
                c.expected_kv
            ));
        }
    }
    // Positive control (the vacuous-guard law), PHASE-AWARE: the signal a
    // fake-quant arm can move depends on WHICH phase it arms — a decode-only
    // arm cannot change the prefill-produced `first_fnv` (token 1 comes from
    // the prefill BEFORE any decode step), and a prefill-only arm's decode
    // continuation may legitimately coincide on some items. The FIRST
    // matrix run FATALed here at the final gate with every cell measured:
    // the v1 control compared first_fnvs for ALL armed cells, structurally
    // unsatisfiable for dec_aq. (Cells: pf/both → first_fnv; dec →
    // second_fnv — the first decode-lane logits.)
    for (name, c) in &cells {
        if name == "base" || name == "dec_a8" {
            continue;
        }
        let decode_armed = name.starts_with("dec_aq");
        let (theirs, bases) = if decode_armed {
            (&c.second_fnvs, &cells["base"].second_fnvs)
        } else {
            (&c.first_fnvs, &cells["base"].first_fnvs)
        };
        let differs = theirs.iter().zip(bases.iter()).any(|(a, b)| a != b);
        if !differs {
            return Err(format!(
                "G-i2 positive control FAIL: {name} logits identical to base"
            ));
        }
    }

    // ── analysis + report ──────────────────────────────────────────────────
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let mut md = String::new();
    md.push_str("# DQ phase-matrix run (Plan 614 / Issue 026) — RAW DUMP\n\n");
    md.push_str(&format!(
        "- model: `{model_path}` blake3=`{model_hash}`\n- corpus blake3: `{corpus_hash}`\n- gpu: {gpu_csv}\n- compute apps at start: none with dedicated memory\n- grids: {grids:?}\n- kv grids (Issue 033): {kv_grids:?}\n- arith_n: {arith_n}, arith_hard: {arith_hard}, ni_lengths: {ni_lengths:?}, ni_per_len: {ni_per_len}, ni_needles: {ni_needles}\n- bootstrap: {n_boot}\n- attn_layers: {attn_layers}\n\n"
    ));
    md.push_str("## G-i1/G-i4\n\n- G-i1 (knob-off counters == 0): PASS\n- G-i4 (base twice byte-stable): PASS\n\n");
    md.push_str("## Per-cell accuracy + counters\n\n");
    md.push_str("| cell | arith acc | nih acc per len | prefill launches | decode launches | kv launches |\n|---|---|---|---|---|---|\n");
    for (name, c) in &cells {
        let a = c.arith_correct.iter().filter(|x| **x).count() as f64 / c.arith_correct.len() as f64;
        let nis: Vec<String> = c
            .ni_correct
            .iter()
            .map(|(l, v)| format!("{l}: {:.3}", v.iter().filter(|x| **x).count() as f64 / v.len() as f64))
            .collect();
        md.push_str(&format!(
            "| {name} | {a:.4} | {} | {} | {} | {} |\n",
            nis.join(", "),
            c.prefill_launches,
            c.decode_launches,
            c.kv_launches
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

    // ── Issue 033: the KV-STORE axis + the pf×kv interaction ──────────
    // One-arm ladder: the two-arm SATURATED test degenerates with a single
    // arm, so the kv rows use the one-arm form (arm dead OR damage
    // ≥ near-total); the axis-1 pf/dec rows above keep their own two-arm
    // ladder. The store spelling IS the read spelling for accuracy
    // (quant-dequant is idempotent: values entering the attention dot are
    // round(x) either way) — the record states this once, here.
    let one_arm_label = |base_acc: f64, arm_acc: f64, lo: f64, hi: f64, chance: f64| -> String {
        if base_acc < 0.25 || base_acc > 0.95 {
            "INADMISSIBLE".to_string()
        } else if arm_acc <= chance + 0.05 || (base_acc - arm_acc) >= (base_acc - chance) - 0.05 {
            "SATURATED".to_string()
        } else if lo > 0.0 {
            "HIT".to_string()
        } else if hi < 0.0 {
            "REVERSED".to_string()
        } else {
            "NULL".to_string()
        }
    };
    let acc_of = |c: &Vec<bool>| c.iter().filter(|x| **x).count() as f64 / c.len() as f64;
    let suite_acc = |c: &CellResult| -> f64 {
        let all: Vec<bool> = c.ni_correct.values().flat_map(|v| v.iter().copied()).collect();
        acc_of(&all)
    };
    md.push_str("\n## Issue 033 — KV-store axis + interaction\n\n");
    md.push_str(
        "- Spelling: KV rows rounded in place to the grid at WRITE (the q8kv store posture). For accuracy this equals a q8 read rounding (idempotence) — the axis is ONE accuracy experiment.\n",
    );
    for grid in &kv_grids {
        let gname = format!("{grid:?}").to_lowercase();
        let kv = cells
            .get(&format!("kv_aq.{gname}"))
            .ok_or_else(|| format!("missing kv_aq.{gname} cell"))?;
        let base = &cells["base"];
        // decode-heavy (arith — the issue's primary read)
        let d: Vec<i8> = kv
            .arith_correct
            .iter()
            .zip(base.arith_correct.iter())
            .map(|(a, b)| (*a as i8) - (*b as i8))
            .collect();
        let (lo, hi) = paired_bootstrap_ci(&d, n_boot);
        let (base_acc, kv_acc) = (acc_of(&base.arith_correct), acc_of(&kv.arith_correct));
        let dkv = base_acc - kv_acc;
        let label = one_arm_label(base_acc, kv_acc, lo, hi, 0.05);
        verdicts.insert(format!("kv[{gname}].decode_heavy"), label.clone());
        md.push_str(&format!(
            "- **kv[{gname}] decode-heavy (arith)**: acc base={base_acc:.4} kv={kv_acc:.4}; Δkv={dkv:.4}; CI(Δkv)=[{lo:.4},{hi:.4}] → **{label}**\n"
        ));
        // prefill-heavy (pooled NIAH)
        let d_ni: Vec<i8> = kv
            .ni_correct
            .iter()
            .zip(base.ni_correct.iter())
            .flat_map(|((_, kv_v), (_, base_v))| {
                kv_v.iter().zip(base_v.iter()).map(|(a, b)| (*a as i8) - (*b as i8))
            })
            .collect();
        let (lo2, hi2) = paired_bootstrap_ci(&d_ni, n_boot);
        let (base_n, kv_n) = (suite_acc(base), suite_acc(kv));
        let dkv_n = base_n - kv_n;
        let label_n = one_arm_label(base_n, kv_n, lo2, hi2, 0.125);
        verdicts.insert(format!("kv[{gname}].prefill_heavy"), label_n.clone());
        md.push_str(&format!(
            "- **kv[{gname}] prefill-heavy (pooled)**: acc base={base_n:.4} kv={kv_n:.4}; Δkv={dkv_n:.4}; CI=[{lo2:.4},{hi2:.4}] → **{label_n}**\n"
        ));
        // the pf×kv interaction (decode-heavy) — needs the SAME-grid pf cell
        // (an interaction against a different activation grid would mix
        // grids); skipped loud when the activation sweep didn't include it.
        match (
            cells.get(&format!("pf_aq.{gname}")),
            cells.get(&format!("pfkv_aq.{gname}")),
        ) {
            (Some(pf), Some(pfkv)) => {
                let d_i: Vec<i8> = pfkv
                    .arith_correct
                    .iter()
                    .zip(pf.arith_correct.iter())
                    .map(|(a, b)| (*a as i8) - (*b as i8))
                    .collect();
                let (loi, hii) = paired_bootstrap_ci(&d_i, n_boot);
                let pfkv_acc = acc_of(&pfkv.arith_correct);
                let pf_acc = acc_of(&pf.arith_correct);
                let dpi = pf_acc - pfkv_acc;
                md.push_str(&format!(
                    "- **interaction pf→pfkv [{gname}] decode-heavy**: pf={pf_acc:.4} pfkv={pfkv_acc:.4}; Δ(kv|pf armed)={dpi:.4}; CI=[{loi:.4},{hii:.4}] (marginal KV damage ON TOP of prefill quant)\n"
                ));
            }
            (None, Some(_)) => {
                md.push_str(&format!(
                    "- interaction [{gname}]: SKIPPED — no same-grid pf_aq cell (DQ_GRIDS did not include {gname})\n"
                ));
            }
            _ => {}
        }
    }
    // Axis dominance summary (the issue's deliverable): rank |Δ| at the
    // q8kv-class grid on the decode-heavy suite.
    md.push_str("\n### Axis dominance (decode-heavy, |Δacc| vs base)\n\n");
    {
        let base = &cells["base"];
        let base_acc = acc_of(&base.arith_correct);
        let mut rows: Vec<(String, f64)> = Vec::new();
        for g in &grids {
            let gname = format!("{g:?}").to_lowercase();
            if let Some(pf) = cells.get(&format!("pf_aq.{gname}")) {
                rows.push((format!("prefill-act[{gname}]"), base_acc - acc_of(&pf.arith_correct)));
            }
            if let Some(dec) = cells.get(&format!("dec_aq.{gname}")) {
                rows.push((format!("decode-act[{gname}]"), base_acc - acc_of(&dec.arith_correct)));
            }
        }
        for g in &kv_grids {
            let gname = format!("{g:?}").to_lowercase();
            if let Some(kv) = cells.get(&format!("kv_aq.{gname}")) {
                rows.push((format!("kv-store[{gname}]"), base_acc - acc_of(&kv.arith_correct)));
            }
        }
        rows.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        for (k, v) in &rows {
            md.push_str(&format!("- {k}: {v:+.4}\n"));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The Issue-030 lane-check defect, pinned: v1's left-to-right gold for a
    /// mixed-precedence expr disagreed with what the few-shots teach. The
    /// failing lane-check specimen was `4359 + 592 - 198 * 96 * 8` — v1 gold
    /// (((4359+592)-198)*96*8) = 3650304; the model (standard precedence, per
    /// its shots) computed 4359+592-(198*96*8) = -147113. The v2 gold IS the
    /// standard-precedence value.
    #[test]
    fn eval_matches_the_few_shot_convention() {
        // The shots' own two examples (both standard precedence).
        assert_eq!(eval_standard_precedence(23, &[('*', 17), ('+', 45)]), Some(436));
        assert_eq!(eval_standard_precedence(8842, &[('-', 1907), ('+', 333)]), Some(7268));
        assert_eq!(eval_standard_precedence(12, &[('*', 12), ('*', 12)]), Some(1728));
        // The lane-check specimen: standard precedence, NOT left-to-right.
        assert_eq!(
            eval_standard_precedence(4359, &[('+', 592), ('-', 198), ('*', 96), ('*', 8)]),
            Some(-147_113)
        );
        // Precedence-neutral chains agree with left-to-right by construction.
        assert_eq!(eval_standard_precedence(100, &[('+', 200), ('-', 50)]), Some(250));
        // Multiplicative run binds as one term.
        assert_eq!(
            eval_standard_precedence(1000, &[('*', 3), ('+', 2), ('*', 4), ('-', 1)]),
            Some(1000 * 3 + 2 * 4 - 1)
        );
    }

    /// Every generated item's gold is exactly the standard-precedence value of
    /// its own rendered expression, and no v2 item overflows.
    #[test]
    fn arith_items_gold_is_precedence_correct() {
        let items = arith_items(48, false);
        assert_eq!(items.len(), 48);
        // The hard posture (Issue 033): same count, same evaluator — the gold
        // property must hold at BOTH operand scales.
        let hard_items = arith_items(48, true);
        assert_eq!(hard_items.len(), 48);
        assert!(hard_items.iter().zip(items.iter()).any(|(h, s)| h.gold != s.gold));
        assert_eq!(items.len(), 48);
        for it in &items {
            // Re-parse the rendered expression and re-evaluate independently
            // (a second evaluator, written differently: shunting to RPN).
            // The item's question is the LAST "Compute " line — the 4-shot
            // prefix carries four earlier ones (v1 of this test parsed the
            // FIRST and compared item gold vs shot-1's body — a test bug the
            // first 4090 run caught).
            let expr_line = it
                .prompt
                .lines()
                .filter(|l| l.starts_with("Compute "))
                .last()
                .expect("Compute line");
            let body = expr_line
                .strip_prefix("Compute ")
                .and_then(|s| s.strip_suffix(". Think step by step, then give the final answer after ####."))
                .expect("body");
            let mut tokens = body.split_whitespace();
            let first: i128 = tokens.next().unwrap().parse().unwrap();
            let mut ops = Vec::new();
            while let Some(op) = tokens.next() {
                let operand: i128 = tokens.next().unwrap().parse().unwrap();
                let op = op.chars().next().unwrap();
                assert!(matches!(op, '+' | '-' | '*'), "v2 generated a division: {body}");
                ops.push((op, operand));
            }
            // Independent RPN evaluation: fold multiplicative runs into
            // (pending_addop, product) pairs FIRST — including the FINAL run,
            // which joins with its own pending op (v1 of this re-derivation
            // always ADDED the last term — wrong when the expr ends in '-').
            let mut terms: Vec<(char, i128)> = Vec::new();
            let mut mul_run: Vec<i128> = vec![first];
            let mut pending: char = '+';
            for &(op, operand) in &ops {
                match op {
                    '*' => mul_run.push(operand),
                    add => {
                        let product: i128 = mul_run.iter().product();
                        terms.push((pending, product));
                        pending = add;
                        mul_run = vec![operand];
                    }
                }
            }
            let last: i128 = mul_run.iter().product();
            terms.push((pending, last));
            let mut acc = 0i128;
            for &(op, value) in &terms {
                acc = if op == '+' { acc + value } else { acc - value };
            }
            assert_eq!(it.gold, acc, "gold mismatch for {body}");
        }
    }
}
