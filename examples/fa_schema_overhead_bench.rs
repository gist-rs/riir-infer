//! Issue 035 P2 bench — per-step constrained-decode overhead vs the
//! unconstrained per-position sampler, CPU-only with synthetic logits.
//!
//! The constrained lane does NOT add forward passes (both arms run the same
//! denoising forwards), so the honest delta is the SAMPLING step. This bench
//! isolates exactly that: one step's sampling cost over a block of L
//! positions, unconstrained (L × `sample_with_confidence`-shaped draws over
//! a synthetic vocab) vs constrained (one exact joint draw over a compiled
//! schema automaton), at L ∈ {64, 128, 256} × three grammar scales (toy
//! enum → 2-prop object → function-call-shaped object).
//!
//! Since P2.5 the compiled automaton is MINIMIZED (`fa_schema::compile` →
//! `fa_minimize`); the `raw n/e` vs `min n/e` columns expose the subset
//! construction's 2^k·k bloat and what the minimizer collapsed it to.
//!
//! CPU-only: no GPU, no model. Box state printed per the G2 provenance law
//! (the numbers are CPU µs/step on THIS box at THIS moment — record the
//! line beside any published figure).

use riir_infer_core::fa_posterior::{FREE, FaScratch, SplitMix64};
use riir_infer_core::fa_schema::{JsonLit, Schema, compile_stats};
use std::time::Instant;

fn box_state_line() -> String {
    let n_cpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    format!(
        "PROVENANCE: cpu-only bench, {} logical CPUs, ts {}",
        n_cpu,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    )
}

/// Best-of-5 timing of a closure (min filters scheduler noise; the G2
/// convention).
fn best_of_5_us(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        f();
        let us = t.elapsed().as_secs_f64() * 1e6;
        if us < best {
            best = us;
        }
    }
    best
}

fn enum_schema() -> Schema {
    Schema::Enum(vec![
        JsonLit::Str("alice".into()),
        JsonLit::Str("bob".into()),
        JsonLit::Int(42),
        JsonLit::Bool(true),
        JsonLit::Bool(false),
        JsonLit::Null,
    ])
}

fn object2_schema() -> Schema {
    Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("age".into(), Schema::Integer),
        ],
        required: vec!["name".into(), "age".into()],
    }
}

/// Function-call-shaped: 5 properties (2 required), one nested array of
/// integers, an anyOf value.
fn fn_call_schema() -> Schema {
    Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("id".into(), Schema::Integer),
            (
                "tags".into(),
                Schema::Array {
                    items: Box::new(Schema::String),
                },
            ),
            (
                "mode".into(),
                Schema::AnyOf(vec![Schema::Null, Schema::Boolean, Schema::String]),
            ),
            ("note".into(), Schema::String),
        ],
        required: vec!["name".into(), "id".into()],
    }
}

/// A JSON-ish vocabulary at three granularities: single chars only, plus
/// word tokens, plus multi-char JSON fragments. Token ids are the index.
fn vocab(scale: usize) -> Vec<(u32, String)> {
    let mut v: Vec<(u32, String)> = Vec::new();
    let push = |s: &str, v: &mut Vec<(u32, String)>| {
        v.push((v.len() as u32, s.to_string()));
    };
    for c in "{}[]:,\" \n\t0123456789.-+eE".chars() {
        let mut s = String::new();
        s.push(c);
        push(&s, &mut v);
    }
    for w in [
        "true", "false", "null", "name", "age", "id", "tags", "mode", "note", "alice", "bob", "abc",
    ] {
        push(w, &mut v);
    }
    if scale >= 2 {
        for f in [
            "{\"name\":",
            ",\"age\":",
            "\":",
            "\",\"",
            "\":null}",
            "[\"",
            "\"]",
        ] {
            push(f, &mut v);
        }
    }
    v
}

fn main() {
    println!("{}", box_state_line());
    println!(
        "{:<12} {:>5} {:>10} {:>13} {:>13} {:>14} {:>12} {:>10}",
        "grammar", "L", "vocab", "raw n/e", "min n/e", "unconstr µs/st", "constr µs/st", "ratio"
    );
    let scales = [
        ("toy-enum", enum_schema()),
        ("object-2p", object2_schema()),
        ("fn-call-5p", fn_call_schema()),
    ];
    for (gname, schema) in &scales {
        for vscale in 1..=2 {
            let v = vocab(vscale);
            let v_refs: Vec<(u32, &str)> = v.iter().map(|&(id, ref s)| (id, s.as_str())).collect();
            let (fa, stats) =
                compile_stats(schema, &v_refs).unwrap_or_else(|e| panic!("{gname} v{vscale}: {e}"));
            let vcount = v.len();
            for &len in &[64usize, 128, 256] {
                let logits: Vec<f32> = (0..len * vcount)
                    .map(|i| (((i * 7919) % 23) as f32 - 8.0) / 4.0)
                    .collect();
                let forced = vec![FREE; len];
                let mut out = vec![0u32; len];

                // Unconstrained arm: L independent mask-suppressed draws
                // (the base lane's per-position sample_with_confidence
                // shape, over the same synthetic vocab).
                let mut rng_u = SplitMix64::new(42);
                let unconstr = best_of_5_us(|| {
                    for t in 0..len {
                        // One softmax draw over the vocab (mask-free here —
                        // the cost class is the draw, not the mask).
                        let row = &logits[t * vcount..(t + 1) * vcount];
                        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut acc = 0.0f64;
                        let weights: Vec<f64> = row
                            .iter()
                            .map(|&l| ((l - max) as f64).exp())
                            .inspect(|w| acc += *w)
                            .collect();
                        let r = rng_u.next_f64() * acc;
                        let mut cum = 0.0f64;
                        let mut pick = vcount - 1;
                        for (i, &w) in weights.iter().enumerate() {
                            cum += w;
                            if r < cum {
                                pick = i;
                                break;
                            }
                        }
                        out[t] = pick as u32;
                    }
                });

                // Constrained arm: ONE exact joint draw over the block.
                let mut rng_c = SplitMix64::new(42);
                let mut scratch = FaScratch::new();
                let constrained = best_of_5_us(|| {
                    fa.sample_joint(
                        &mut scratch,
                        &logits,
                        len,
                        &forced,
                        1.0,
                        false,
                        &mut rng_c,
                        &mut out,
                    )
                    .expect("constrained draw");
                });

                // Sanity: the constrained draw is grammar-valid.
                let node = fa
                    .walk(&out)
                    .unwrap_or_else(|| panic!("{gname}: draw not grammar-valid"));
                assert!(fa.is_accept(node));

                println!(
                    "{:<12} {:>5} {:>10} {:>7}/{:<5} {:>7}/{:<5} {:>14.1} {:>12.1} {:>9.2}x",
                    format!("{gname}/v{vscale}"),
                    len,
                    vcount,
                    stats.raw_nodes,
                    stats.raw_edges,
                    stats.nodes,
                    stats.edges,
                    unconstr,
                    constrained,
                    constrained / unconstr
                );
            }
        }
    }
}
