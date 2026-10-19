//! Plan 619 T2.3 — the clef-text latency bench (the G2 gate).
//!
//! Per-bucket decoder/head/total milliseconds for one compute-unit
//! posture per invocation (`ane` default, `gpu`, `cpu`), over
//! representative records landed in each bucket's upper band, on the
//! REAL pinned artifact tree. On `cpu` the runtime is a disclosed
//! SPLIT (issue 048): decoder CpuOnly + head CpuAndNe — the decode
//! column is the honest cpu leg; head/total are MIXED numbers. The bench record quotes the preflight
//! `PROVENANCE:` line beside every number (the box-state law) — run:
//!
//! ```text
//! cargo run --release -p riir-infer-clef --features clef-ane \
//!     --example clef_text_timing -- ane --reps 20
//! ```
//!
//! One posture per process: compute units are baked into each
//! `MLModelConfiguration` at load (the T1.6 posture), so the three
//! postures are three loads, never a mid-process switch. Decode and
//! head are timed as separate legs over the SAME forward — the split
//! the plan's table wants — plus one whole-`answer()` readout.

use riir_infer_ane::ComputeUnits;
use riir_infer_clef::lane::{
    ARTIFACT_REVISION, ClefQuestion, ClefText, ClefTokenizer, EmbeddingTable, artifacts_dir,
    encode, ensure_and_load,
};
use std::collections::BTreeMap;
use std::time::Instant;

const SENTENCE: &str = "The platform team reviewed the weekly reliability report and noted that error budgets remain comfortable across every customer-facing service.";

/// One sentence ≈ 24 tokens (the measured ratio; the loop below only
/// needs a slack estimate to stop growing the state).
const SENTENCE_TOKEN_SLACK: usize = 26;

#[derive(Clone, Copy)]
struct Stats {
    p50: f64,
    p90: f64,
}

/// Median + p90 over samples already in ascending order (milliseconds).
fn stats(sorted: &[f64]) -> Stats {
    let n = sorted.len();
    Stats {
        p50: sorted[n / 2],
        p90: sorted[(n * 9 / 10).min(n - 1)],
    }
}

fn healthy_question() -> (String, ClefQuestion) {
    (
        "healthy".to_string(),
        ClefQuestion::Noul {
            instructions: Some("Does the report describe a healthy system?".into()),
            criteria: BTreeMap::new(),
        },
    )
}

/// Build a record landing in `bucket`: repeat the sentence until the
/// encoded length enters `[bucket - 96, bucket - 8]` (the decode cost is
/// the bucket's padded length L, so any n in the band costs the same —
/// the band just keeps the record inside its bucket).
fn record_for_bucket(
    tokenizer: &ClefTokenizer,
    bucket: usize,
    max_length: usize,
) -> riir_infer_clef::lane::ClefEncodedRecord {
    let questions = [healthy_question()];
    let mut state = String::new();
    loop {
        let record = encode(
            tokenizer,
            &serde_json::Value::String(state.clone()),
            &questions,
            max_length,
        )
        .expect("bench record encodes");
        let n = record.input_ids.len();
        if n + SENTENCE_TOKEN_SLACK >= bucket - 96 {
            if n <= bucket - 8 {
                return record;
            }
            panic!("sentence overshoots bucket {bucket} (n={n})");
        }
        if !state.is_empty() {
            state.push(' ');
        }
        state.push_str(SENTENCE);
        assert!(
            state.len() <= bucket * 24,
            "cannot land a record in bucket {bucket}"
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let posture = args.get(1).map(String::as_str).unwrap_or("ane");
    if !matches!(posture, "ane" | "gpu" | "cpu") {
        panic!("posture must be ane|gpu|cpu (got {posture:?})");
    }
    let units = match posture {
        "gpu" => ComputeUnits::CpuAndGpu,
        "cpu" => ComputeUnits::CpuOnly,
        _ => ComputeUnits::CpuAndNe,
    };
    let reps: usize = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let warm: usize = args
        .iter()
        .position(|a| a == "--warm")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    // Optional subset (`--buckets 1024,2048`) — one process per subset
    // keeps the footprint down on a memory-pressured box (the cpu
    // posture's large-bucket forwards were jetsam-killed when all eight
    // models were resident); each invocation still loads exactly the
    // buckets it times.
    let dir = artifacts_dir().expect("artifact dir resolves");
    let model = ensure_and_load(&dir).expect("the artifact law + materials");
    let buckets: Vec<u32> = args
        .iter()
        .position(|a| a == "--buckets")
        .and_then(|i| args.get(i + 1))
        .map(|csv| {
            csv.split(',')
                .map(|v| v.trim().parse().expect("bucket u32"))
                .collect()
        })
        .unwrap_or_else(|| model.config.buckets.clone());
    for bucket in &buckets {
        assert!(
            model.config.buckets.contains(bucket),
            "bucket {bucket} is not a config bucket"
        );
    }

    let started = Instant::now();
    let runtime =
        ClefText::load_from(&dir, model.config.clone(), &buckets, units).expect("all buckets load");
    let embeddings = EmbeddingTable::open(&dir, &model.config).expect("embeddings reopen");
    let max_length = *model.config.buckets.last().expect("buckets") as usize;

    // Issue 048: on cpu the runtime is a disclosed split (decoder
    // CpuOnly, head CpuAndNe) — say so on the banner and in the NOTE
    // below; the row formats are untouched.
    let banner_head = if posture == "cpu" {
        " head_units=CpuAndNe"
    } else {
        ""
    };
    println!(
        "CLEF-TIMING posture={posture} model=clef-text-0.6b@{} buckets={buckets:?} reps={reps} warm={warm} load={:?}{}",
        ARTIFACT_REVISION,
        started.elapsed(),
        banner_head
    );
    for line in &runtime.placements {
        println!("PLACEMENT {line}");
    }
    if posture == "cpu" {
        println!(
            "NOTE cpu posture is a SPLIT (issue 048): decoder CpuOnly — the decode \
             column is the honest cpu leg; head loaded CpuAndNe — head/total are \
             MIXED numbers; head-cpu NOT-MEASURABLE-HERE (Core ML CpuOnly planner \
             balloons the head load ~202GB, jetsam-killed)."
        );
    }
    println!(
        "table: bucket tokens decode_p50 decode_p90 head_p50 head_p90 total_p50 total_p90 answer_p50 (ms)"
    );

    for bucket in &buckets {
        let bucket = *bucket as usize;
        let record = record_for_bucket(&model.tokenizer, bucket, max_length);
        let n = record.input_ids.len();

        let mut decode = Vec::with_capacity(reps);
        let mut head = Vec::with_capacity(reps);
        let mut total = Vec::with_capacity(reps);
        for i in 0..(warm + reps) {
            let t0 = Instant::now();
            let (b, states) = runtime
                .decode_forward(&record, &embeddings)
                .expect("decode");
            let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let _ = runtime
                .head_forward(b, &record, &states, &embeddings)
                .expect("head");
            let head_ms = t1.elapsed().as_secs_f64() * 1e3;
            if i >= warm {
                decode.push(decode_ms);
                head.push(head_ms);
                total.push(decode_ms + head_ms);
            }
        }
        decode.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        head.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        total.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let (d, h, t) = (stats(&decode), stats(&head), stats(&total));

        // one whole-pipeline readout (encode → bucket select → decode →
        // head → softmax) on a SHORT record — the user-facing cost the
        // serving posture pays per decision
        let mut whole = Vec::with_capacity(reps);
        let state = serde_json::Value::String(String::new());
        let questions = [healthy_question()];
        for _ in 0..reps {
            let t0 = Instant::now();
            let _ = runtime.answer(&state, &questions).expect("answer");
            whole.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        whole.sort_by(|a, b| a.partial_cmp(b).expect("finite"));

        println!(
            "ROW {bucket} {n} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
            d.p50,
            d.p90,
            h.p50,
            h.p90,
            t.p50,
            t.p90,
            stats(&whole).p50
        );
    }
}
