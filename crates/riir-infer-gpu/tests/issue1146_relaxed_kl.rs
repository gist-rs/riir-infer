//! Issue 1146 rung 2b — the per-family QUALITY instrument for the lossy
//! `matmul2d relaxed_precision` lever (`RIIR_METAL_TENSOR_RELAXED=1`,
//! `6125f9e`).
//!
//! The lossy-surface promotion rule gates on per-family, deployed-path
//! behaviour, never on token-id agreement alone: aggregate agreement can stay
//! flat while one family's distribution moves. So this measures, per family
//! (code · math · prose · zh), teacher-forced over FIXED text:
//!
//! - `KL(ref ‖ cand)` of the next-token distribution (mean + max over the
//!   scored positions),
//! - top-1 agreement rate,
//! - `ΔNLL` = NLL(cand) − NLL(ref) of the TRUE next token (mean), plus the
//!   reference NLL itself (a sanity read: a broken reference reads as a huge
//!   NLL, not as a small KL).
//!
//! `ref` = the simdgroup incumbent (tensor zero-copy OFF); `cand` = the Metal
//! tensor GEMM (ON), compiled with whatever posture the env selects. Run it
//! twice — once plain (the strict tensor control: G1-exact, so KL must read
//! exactly 0) and once under `RIIR_METAL_TENSOR_RELAXED=1` (the measurement).
//! The control run is what makes a non-zero relaxed number attributable to
//! the lever and not to the instrument.
//!
//! Scored positions: `I1146K_K` prefix lengths spread evenly over the second
//! half of each family's first `I1146K_P` tokens; each prefix is one whole
//! `prefill()` whose last-position logits predict the next text token. The
//! tensor arm runs FIRST in every pair (the first-use pool hazard, as in
//! `issue1146_tensor_range_g1`).
//!
//! The distributions are the model's own softmax readout — this is a
//! measurement of a softmax LM's predictive distribution, not a mechanism
//! choice, so log-softmax (f64, max-shifted) is the definition here.
//!
//! The STRICT run is a hard gate (max KL == 0 and top-1 n/n, every family).
//! The RELAXED run is a REPORT by default (no threshold is fixed before the
//! first measurement); `I1146K_MAX_KL` (nats) arms an assertion on every family's max KL once a
//! bar has been chosen from data.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1146k cargo test -p riir-infer-gpu --release \
//!   --features metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test issue1146_relaxed_kl -- --ignored --nocapture --test-threads=1
//! # then the same with RIIR_METAL_TENSOR_RELAXED=1 prefixed
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` (model path), `I1146K_P` (tokens per family,
//! default 256), `I1146K_K` (scored positions per family, default 16),
//! `I1146K_MAX_KL` (optional bar).

#![cfg(all(
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "multi_host",
    feature = "ternary_gemm_batched",
    target_os = "macos"
))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::set_prefill_use_metal_tensor_zerocopy;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

/// Fixed per-family text. Frozen: a changed text changes every number, so
/// edit only together with a re-measured control row.
const FAMILIES: [(&str, &str); 4] = [
    (
        "code",
        r#"use std::collections::HashMap;

/// Count how often each word appears in the input, ignoring case.
pub fn word_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let key = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if key.is_empty() {
            continue;
        }
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

/// Return the `n` most frequent words, ties broken alphabetically.
pub fn top_words(counts: &HashMap<String, usize>, n: usize) -> Vec<(String, usize)> {
    let mut pairs: Vec<(String, usize)> = counts
        .iter()
        .map(|(word, &count)| (word.clone(), count))
        .collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(n);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_case_insensitive() {
        let counts = word_counts("The cat saw the other cat.");
        assert_eq!(counts["the"], 2);
        assert_eq!(counts["cat"], 2);
        assert_eq!(counts["saw"], 1);
    }

    #[test]
    fn top_words_breaks_ties_alphabetically() {
        let counts = word_counts("b a c b a");
        let top = top_words(&counts, 2);
        assert_eq!(top, vec![("a".to_string(), 2), ("b".to_string(), 2)]);
    }
}
"#,
    ),
    (
        "math",
        "Problem. A rectangle has a perimeter of 54 centimetres, and its length is \
         three centimetres more than twice its width. Find the area of the rectangle.\n\n\
         Solution. Let the width be w centimetres. Then the length is 2w + 3. The \
         perimeter of a rectangle is twice the sum of its length and width, so \
         2(w + 2w + 3) = 54. Dividing both sides by 2 gives 3w + 3 = 27, hence \
         3w = 24 and w = 8. The length is therefore 2(8) + 3 = 19 centimetres. \
         Check: 2(8 + 19) = 2(27) = 54, which matches the given perimeter. The area \
         is length times width, 19 × 8 = 152 square centimetres.\n\n\
         Problem. Solve the equation x^2 - 5x + 6 = 0.\n\n\
         Solution. We look for two numbers whose product is 6 and whose sum is 5. \
         These are 2 and 3, so the quadratic factors as (x - 2)(x - 3) = 0. A product \
         is zero exactly when one of its factors is zero, so x = 2 or x = 3. \
         Substituting x = 2 gives 4 - 10 + 6 = 0, and x = 3 gives 9 - 15 + 6 = 0, \
         so both roots are correct.\n\n\
         Problem. What is the sum of the first 100 positive integers?\n\n\
         Solution. Write the sum forwards and backwards: S = 1 + 2 + ... + 100 and \
         S = 100 + 99 + ... + 1. Adding the two lines term by term gives 100 pairs, \
         each summing to 101, so 2S = 100 × 101 = 10100 and S = 5050.",
    ),
    (
        "prose",
        "The lighthouse keeper had kept the same routine for eleven years. Each \
         evening, an hour before sunset, he climbed the one hundred and twelve steps \
         to the lamp room, wiped the salt from the great lens, and checked the oil. \
         The island was small enough to walk around in twenty minutes, and in winter \
         the supply boat came only twice a month, so he had learned to read the \
         weather the way other people read the newspaper. A grey band low on the \
         horizon meant rain by morning. A sudden stillness in the gulls meant wind.\n\n\
         That autumn a letter arrived with the boat, addressed in handwriting he did \
         not recognise. It was from a schoolteacher on the mainland who wanted to \
         bring her class to see the light. He read it three times at the kitchen \
         table before he understood that he was nervous. He had not spoken to more \
         than one person at a time in years. Still, he wrote back the same night and \
         said yes, and then he spent the following week repainting the railings, \
         sweeping the stairs, and practising, out loud and to nobody, an explanation \
         of how a lens can bend a small flame into a beam that sailors see twenty \
         miles away.\n\n\
         The children arrived on a calm morning in October. There were nineteen of \
         them, wrapped in bright coats, and they asked more questions in the first \
         ten minutes than anyone had asked him in a decade. Why was the lamp so \
         large? What happened if the oil ran out? Had he ever seen a shipwreck? He \
         answered every question slowly and carefully, and by the time they reached \
         the top of the tower he had forgotten to be nervous at all. When the boat \
         carried them away that afternoon, they waved until the island was only a \
         dark line on the water, and he stood on the rocks and waved back.",
    ),
    (
        "zh",
        "春天的早晨，小镇上的市场总是最热闹的。天刚亮，卖菜的农民就推着车子来了，车上装满了新鲜的青菜、萝卜和西红柿。\
         卖早点的小店门口排起了长队，包子和豆浆的香味飘满了整条街。老人们喜欢在这个时候出来散步，\
         一边走一边和熟人打招呼，问问家里的情况，聊聊最近的天气。\n\n\
         李阿姨在市场里开了一家小小的面馆，已经二十多年了。她每天四点钟起床和面，六点钟准时开门。\
         她说，做面最重要的是耐心，面要揉得够久，汤要熬得够长，客人吃得出来。很多年轻人离开小镇去城里工作，\
         可是每次回家，第一件事就是来她的店里吃一碗面。他们说，城里的面再好吃，也没有家乡的味道。\n\n\
         到了中午，市场慢慢安静下来。农民们收拾好剩下的菜，准备回家吃饭，下午还要去地里干活。\
         李阿姨也终于可以坐下来休息一会儿，喝一杯热茶，看着街上来来往往的人，觉得日子虽然平淡，却很踏实。\n\n\
         傍晚的时候，镇上的孩子们放学回家，路过面馆总要停下来看一看。有的孩子帮李阿姨搬椅子，\
         有的孩子坐在门口写作业。李阿姨常常给他们煮一小碗清汤面，看着他们吃得干干净净，心里很高兴。\
         她说，等这些孩子长大了，走得再远，也会记得小镇上有一碗热乎乎的面在等着他们。",
    ),
];

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

/// Log-softmax in f64, max-shifted (sequential fold — a measurement, so the
/// reduction order is pinned and repeatable).
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits
        .iter()
        .fold(f64::NEG_INFINITY, |a, &x| a.max(x as f64));
    let lse = m + logits
        .iter()
        .map(|&x| ((x as f64) - m).exp())
        .sum::<f64>()
        .ln();
    logits.iter().map(|&x| x as f64 - lse).collect()
}

/// `KL(p ‖ q)` from log-probabilities, nats.
fn kl(lp: &[f64], lq: &[f64]) -> f64 {
    lp.iter()
        .zip(lq)
        .map(|(&a, &b)| a.exp() * (a - b))
        .sum::<f64>()
        .max(0.0)
}

/// Per-family aggregate over the scored positions.
#[derive(Default)]
struct FamilyRow {
    n: usize,
    kl_sum: f64,
    kl_max: f64,
    top1_agree: usize,
    dnll_sum: f64,
    nll_ref_sum: f64,
}

impl FamilyRow {
    fn push(&mut self, ref_logits: &[f32], cand_logits: &[f32], target: usize) {
        let (lr, lc) = (log_softmax(ref_logits), log_softmax(cand_logits));
        let d = kl(&lr, &lc);
        self.n += 1;
        self.kl_sum += d;
        self.kl_max = self.kl_max.max(d);
        self.top1_agree += usize::from(argmax(ref_logits) == argmax(cand_logits));
        self.dnll_sum += lr[target] - lc[target];
        self.nll_ref_sum += -lr[target];
    }
}

/// Prefix lengths to score: `k` points spread evenly over `[p/2, p-1]`.
fn scored_lengths(p: usize, k: usize) -> Vec<usize> {
    let lo = (p / 2).max(1);
    let span = p - 1 - lo;
    let mut v: Vec<usize> = (0..k)
        .map(|i| {
            lo + match k {
                1 => span,
                _ => i * span / (k - 1),
            }
        })
        .collect();
    v.dedup();
    v
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn relaxed_precision_per_family_kl() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let p = env_usize("I1146K_P", 256);
    let k = env_usize("I1146K_K", 16);
    let max_kl: Option<f64> = std::env::var("I1146K_MAX_KL")
        .ok()
        .and_then(|v| v.parse().ok());
    let relaxed = std::env::var("RIIR_METAL_TENSOR_RELAXED").as_deref() == Ok("1");

    let tok = {
        let gguf = GgufFile::open(&model).expect("open gguf for tokenizer");
        BpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf")
    };
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    config.block_size = p + 8;
    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    println!(
        "# issue1146_relaxed_kl: posture {} · P={p} · K={k} · ref = simdgroup, cand = tensor",
        match relaxed {
            true => "RELAXED",
            false => "STRICT (control — KL must be 0)",
        }
    );
    let mut over: Vec<String> = Vec::new();
    for (family, text) in FAMILIES {
        let toks = tok.encode(text);
        assert!(
            toks.len() > p,
            "{family}: text encodes to {} tokens, need > P={p}",
            toks.len()
        );
        let mut row = FamilyRow::default();
        for len in scored_lengths(p, k) {
            let prefix = &toks[..len];
            set_prefill_use_metal_tensor_zerocopy(true);
            fwd.reset_state();
            let cand = fwd.prefill(prefix);
            set_prefill_use_metal_tensor_zerocopy(false);
            fwd.reset_state();
            let reference = fwd.prefill(prefix);
            assert!(
                cand.iter().chain(&reference).all(|x| x.is_finite()),
                "{family}@{len}: non-finite logits"
            );
            row.push(&reference, &cand, toks[len]);
        }
        let n = row.n as f64;
        println!(
            "{family:<6} n={:>2} · KL mean {:.3e} max {:.3e} nats · top1 {}/{} · ΔNLL mean {:+.3e} · NLL(ref) mean {:.3}",
            row.n,
            row.kl_sum / n,
            row.kl_max,
            row.top1_agree,
            row.n,
            row.dnll_sum / n,
            row.nll_ref_sum / n
        );
        // The strict control is a HARD gate: the tensor path is G1-exact, so
        // any non-zero KL here is the instrument (or a pool hazard), never the
        // lever — a relaxed number is only attributable once this reads 0.
        if !relaxed && (row.kl_max != 0.0 || row.top1_agree != row.n) {
            over.push(format!(
                "{family} STRICT control not exact: max KL {:.3e}, top1 {}/{}",
                row.kl_max, row.top1_agree, row.n
            ));
        }
        if let Some(bar) = max_kl
            && row.kl_max > bar
        {
            over.push(format!("{family} max KL {:.3e} > {bar:.3e}", row.kl_max));
        }
    }
    assert!(over.is_empty(), "KL gate failed: {over:?}");
}

#[test]
fn scored_lengths_span_second_half() {
    assert_eq!(scored_lengths(256, 1), vec![255]);
    let v = scored_lengths(256, 16);
    assert_eq!(v.first(), Some(&128));
    assert_eq!(v.last(), Some(&255));
    assert_eq!(v.len(), 16);
}

#[test]
fn kl_zero_on_identical_and_positive_on_shift() {
    let a = [1.0f32, 2.0, 3.0, 0.5];
    let la = log_softmax(&a);
    assert_eq!(kl(&la, &la), 0.0);
    let b = [1.0f32, 2.5, 3.0, 0.5];
    assert!(kl(&la, &log_softmax(&b)) > 0.0);
}
