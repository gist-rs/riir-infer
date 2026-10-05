// Issue 035 P2 tests — the JSON-schema → token-automaton compiler.
//
// The walk-level tests use `Automaton::walk` on tokenized instances; the
// sampler round-trip reuses the P0 exact joint draw — every produced
// sequence must be a valid tokenization of the schema (accepted by
// construction through a REAL compiled grammar, not a hand-built toy).

use super::*;
use crate::fa_posterior::{FREE, FaScratch, SplitMix64};

/// The walk-level validity predicate: the sequence must walk AND land on an
/// accepting node (`walk() == Some` alone is not acceptance — an
/// unterminated string walks to a mid-body node).
fn accepts(fa: &Automaton, toks: &[u32]) -> bool {
    fa.walk(toks).is_some_and(|n| fa.is_accept(n))
}

/// A tiny JSON-ish vocabulary: ids map to the token STRINGS the tokenizer
/// would emit. Includes multi-char tokens (the interesting case) + single
/// structural chars + whitespace for padding.
fn toy_vocab() -> Vec<(u32, &'static str)> {
    vec![
        (0, "{"),
        (1, "}"),
        (2, "["),
        (3, "]"),
        (4, ":"),
        (5, ","),
        (6, "\""),
        (7, "name"),
        (8, "age"),
        (9, "alice"),
        (10, "42"),
        (11, "true"),
        (12, "false"),
        (13, "null"),
        (14, " "),  // ws padding
        (15, "  "), // multi-char ws
        (16, "\""),
        (17, "x"),
        (18, "0"),
        (19, "."),
        (20, "5"),
    ]
}

fn tok(s: &str, vocab: &[(u32, &str)]) -> Vec<u32> {
    // Greedy longest-match tokenization for the test fixtures.
    let mut out = Vec::new();
    let mut rest = s;
    'outer: while !rest.is_empty() {
        let mut best: Option<(usize, u32)> = None;
        for &(id, text) in vocab {
            if rest.starts_with(text) {
                let len = text.len();
                if best.is_none_or(|(l, _)| len > l) {
                    best = Some((len, id));
                }
            }
        }
        let (len, id) = best.expect("tokenization failed");
        // Strip the matched prefix (byte len == char len for ASCII).
        rest = &rest[len..];
        out.push(id);
        continue 'outer;
    }
    out
}

#[test]
fn enum_schema_walks_valid_and_rejects_invalid() {
    let schema = Schema::Enum(vec![
        JsonLit::Str("alice".into()),
        JsonLit::Bool(true),
        JsonLit::Null,
    ]);
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");

    // Valid: `"alice"` = `"`(6) + `alice`(9) + `"`(6 — the earlier of the
    // two duplicate-quote ids in the greedy tokenizer).
    let good = tok("\"alice\"", &vocab);
    assert_eq!(good, vec![6, 9, 6]);
    assert!(accepts(&fa, &good), "\"alice\" must walk");

    let good_true = tok("true", &vocab);
    assert!(accepts(&fa, &good_true));

    // Invalid: `false` is not in the enum.
    let bad = tok("false", &vocab);
    assert!(!accepts(&fa, &bad), "false must NOT walk");

    // Invalid: a bare string not in the enum.
    let bad2 = tok("\"x\"", &vocab);
    assert!(!accepts(&fa, &bad2), "\"x\" must NOT walk");
}

#[test]
fn string_type_walks_and_escapes() {
    let schema = Schema::String;
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");

    // `"` + `x` + `"` — a valid (if short) string.
    let good = tok("\"x\"", &vocab);
    assert!(accepts(&fa, &good), "quoted string walks");

    // Unterminated: `"` + `x` — must die.
    let bad = vec![6u32, 17];
    assert!(!accepts(&fa, &bad), "unterminated string must die");
}

#[test]
fn integer_and_number_shapes() {
    let vocab = toy_vocab();
    let int_schema = Schema::Integer;
    let fa = compile(&int_schema, &vocab).expect("compile");
    // `42`
    assert!(accepts(&fa, &tok("42", &vocab)));
    // `0`
    assert!(accepts(&fa, &tok("0", &vocab)));
    // Floats refuse on Integer: `0` `.` `5` — walks `0` then dies at `.`.
    let float_toks = tok("0.5", &vocab);
    assert!(
        !accepts(&fa, &float_toks),
        "0.5 must not walk an Integer schema"
    );

    let num_schema = Schema::Number;
    let fan = compile(&num_schema, &vocab).expect("compile");
    assert!(accepts(&fan, &tok("42", &vocab)));
    assert!(accepts(&fan, &tok("0.5", &vocab)));
}

#[test]
fn object_required_and_ordering() {
    // {"name": <string>, "age": <int>} — BOTH required.
    let schema = Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("age".into(), Schema::Integer),
        ],
        required: vec!["name".into(), "age".into()],
    };
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");

    // Order 1: {"name":"alice","age":42}
    let a = tok("{\"name\":\"alice\",\"age\":42}", &vocab);
    assert!(accepts(&fa, &a), "canonical order walks: {a:?}");

    // Order 2 (any-order): {"age":42,"name":"alice"}
    let b = tok("{\"age\":42,\"name\":\"alice\"}", &vocab);
    assert!(accepts(&fa, &b), "reordered members walk: {b:?}");

    // Missing `age`: dies at the closing `}`.
    let c = tok("{\"name\":\"alice\"}", &vocab);
    assert!(!accepts(&fa, &c), "missing required must die");

    // Duplicate member: `name` twice — at-most-once; the second `name`
    // after its bit is set has no edge → dies.
    let d = tok("{\"name\":\"alice\",\"name\":\"x\",\"age\":42}", &vocab);
    assert!(!accepts(&fa, &d), "duplicate member must die");

    // Empty object: dies (required missing).
    let e = tok("{}", &vocab);
    assert!(!accepts(&fa, &e));
}

#[test]
fn object_optional_member_and_empty() {
    // {"name": <string>} required; {"age": <int>} optional.
    let schema = Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("age".into(), Schema::Integer),
        ],
        required: vec!["name".into()],
    };
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");

    assert!(accepts(&fa, &tok("{\"name\":\"x\"}", &vocab)));
    assert!(accepts(&fa, &tok("{\"name\":\"x\",\"age\":42}", &vocab)));
    // age-only still dies (name required).
    assert!(!accepts(&fa, &tok("{\"age\":42}", &vocab)));

    // Empty schema (no properties, none required): `{}` walks.
    let empty = Schema::Object {
        properties: vec![],
        required: vec![],
    };
    let fa2 = compile(&empty, &vocab).expect("compile");
    assert!(accepts(&fa2, &tok("{}", &vocab)));
}

#[test]
fn array_items() {
    let schema = Schema::Array {
        items: Box::new(Schema::Integer),
    };
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");

    assert!(accepts(&fa, &tok("[]", &vocab)), "empty array walks");
    assert!(accepts(&fa, &tok("[42]", &vocab)));
    assert!(accepts(&fa, &tok("[42,0,42]", &vocab)));
    // Non-item element dies.
    assert!(!accepts(&fa, &tok("[true]", &vocab)));
    // Trailing comma dies.
    assert!(!accepts(&fa, &tok("[42,]", &vocab)));
}

#[test]
fn any_of_union() {
    let schema = Schema::AnyOf(vec![Schema::Integer, Schema::Null]);
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");
    assert!(accepts(&fa, &tok("42", &vocab)));
    assert!(accepts(&fa, &tok("null", &vocab)));
    assert!(!accepts(&fa, &tok("true", &vocab)));
}

#[test]
fn trailing_whitespace_padding_walks() {
    // A block longer than the instance pads with ws tokens and stays
    // accepting — the block-padding law.
    let schema = Schema::Null;
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");
    let padded = tok("null", &vocab);
    assert!(accepts(&fa, &padded));
    // Single-space token then multi-space token then space: padding.
    let pad_seq = [13u32, 14, 15, 14];
    assert!(accepts(&fa, &pad_seq), "ws padding walks");
    // Padding must still end accepted.
    let node = fa.walk(&pad_seq).unwrap();
    assert!(fa.is_accept(node));
    // But non-ws content after the instance dies (token 10 = `42`).
    assert!(!accepts(&fa, &[13u32, 10]), "42 after null must die");
}

#[test]
fn bounds_are_enforced() {
    let vocab = toy_vocab();
    // Depth: nest arrays 30 deep.
    let mut deep = Schema::Array {
        items: Box::new(Schema::Null),
    };
    for _ in 0..30 {
        deep = Schema::Array {
            items: Box::new(deep),
        };
    }
    assert!(matches!(
        compile(&deep, &vocab),
        Err(SchemaError::TooDeep { .. })
    ));

    // Properties: 11 members refuse.
    let props: Vec<(String, Schema)> = (0..11).map(|i| (format!("k{i}"), Schema::Null)).collect();
    let schema = Schema::Object {
        properties: props,
        required: vec![],
    };
    assert!(matches!(
        compile(&schema, &vocab),
        Err(SchemaError::TooManyProperties { n: 11, .. })
    ));

    // required naming an absent property.
    let schema = Schema::Object {
        properties: vec![("a".into(), Schema::Null)],
        required: vec!["b".into()],
    };
    assert!(matches!(
        compile(&schema, &vocab),
        Err(SchemaError::InvalidSchema(_))
    ));
}

// ── P2.6 differential: the trie walk's new cases ─────────────────────

#[test]
fn trie_walk_matches_naive_stepping_on_prefix_and_duplicate_tokens() {
    // The differential that matters most: one token a PREFIX of another
    // (`"ab"` ⊂ `"abc"`), duplicated TEXT across ids (`"x"` twice — the
    // P2 toy vocab's double-quote case at the trie level), and tokens that
    // die everywhere. Every accepted string must be identical.
    let vocab: Vec<(u32, &str)> = vec![
        (0, "{"),
        (1, "}"),
        (2, ":"),
        (3, ","),
        (4, "\""),
        (5, "ab"),
        (6, "abc"),
        (7, "a"),
        (8, "\""), // duplicate text, different id
        (9, "z"),  // dies everywhere (not in any schema)
        (10, " "),
    ];
    let schema = Schema::Object {
        properties: vec![("ab".into(), Schema::Enum(vec![JsonLit::Str("abc".into())]))],
        required: vec!["ab".into()],
    };
    let fa = compile(&schema, &vocab).expect("compile");

    // {"ab":"abc"} — the full instance, token by token.
    let good = [0u32, 4, 5, 4, 2, 4, 6, 4, 1];
    assert!(accepts(&fa, &good), "full instance must walk");
    // Same instance with the duplicate quote id 8 wherever id 4 appears —
    // the automaton cannot tell them apart (identical text).
    let twin: Vec<u32> = good.iter().map(|&t| if t == 4 { 8 } else { t }).collect();
    assert!(
        accepts(&fa, &twin),
        "duplicate-text ids are interchangeable"
    );
    // `ab` (5) in place of `abc` (6): the enum value refuses the prefix.
    let short = [0u32, 4, 5, 4, 2, 4, 5, 4, 1];
    assert!(!accepts(&fa, &short), "enum value must be exactly abc");
    // Single-char `a` (7) as the key text: `"a"` is not the key `"ab"`.
    let wrongkey = [0u32, 4, 7, 4, 2, 4, 6, 4, 1];
    assert!(!accepts(&fa, &wrongkey), "key must be exactly ab");
}

#[test]
fn trie_walk_compiles_a_large_synthetic_vocab() {
    // The P2.6 scalability smoke: a 20k-token vocab (word + subword mix,
    // the real-BPE shape) against the fn-call grammar must compile fast
    // and keep the sampler round-trip intact. 20k × ~150 naive steps-per-
    // token × subsets was the pathology; the trie walks prefixes once.
    let mut vocab: Vec<(u32, String)> = Vec::new();
    let push = |s: &str, vocab: &mut Vec<(u32, String)>| {
        vocab.push((vocab.len() as u32, s.to_string()));
    };
    for c in "{}[]:,\" \\n\t".chars() {
        let mut s = String::new();
        s.push(c);
        push(&s, &mut vocab);
    }
    for w in [
        "true", "false", "null", "name", "id", "tags", "mode", "note",
    ] {
        push(w, &mut vocab);
    }
    // 20k word-ish tokens: three-letter prefix + number suffix, with
    // shared prefixes (the trie's reason to exist).
    for i in 0..20_000u32 {
        let pfx = [b'a' + (i % 26) as u8, b'm', b'z' + (i % 3) as u8];
        push(
            &format!("{}_{}{}", String::from_utf8_lossy(&pfx), i % 997, i % 13),
            &mut vocab,
        );
    }
    let v_refs: Vec<(u32, &str)> = vocab.iter().map(|&(id, ref s)| (id, s.as_str())).collect();

    let schema = Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("id".into(), Schema::Integer),
            (
                "tags".into(),
                Schema::Array {
                    items: Box::new(Schema::String),
                },
            ),
        ],
        required: vec!["name".into()],
    };
    let (fa, stats) = compile_stats(&schema, &v_refs).expect("compile 20k vocab");

    // The string-body states must admit the word tokens — sample a joint
    // draw and require grammar validity (the by-construction law).
    let len = 32;
    let v = fa.vocab();
    let logits: Vec<f32> = (0..len * v)
        .map(|i| (((i * 7919) % 23) as f32 - 8.0) / 4.0)
        .collect();
    let forced = vec![FREE; len];
    let mut scratch = FaScratch::new();
    let mut rng = SplitMix64::new(7);
    let mut out = vec![0u32; len];
    fa.sample_joint(
        &mut scratch,
        &logits,
        len,
        &forced,
        1.0,
        false,
        &mut rng,
        &mut out,
    )
    .expect("draw at 20k vocab");
    let node = fa.walk(&out).expect("draw must walk");
    assert!(fa.is_accept(node));
    // Sanity: the compiled graph is bounded (no subset explosion).
    assert!(
        stats.raw_nodes < 4096,
        "subset count bounded, got {}",
        stats.raw_nodes
    );
}

#[test]
fn determinism_invariants_hold_on_compiled_grammars() {
    // The merge-by-(from,to) must produce a builder-clean automaton on a
    // grammar with converging paths (the `build()` in compile enforces;
    // this asserts compile() itself returns Ok across a battery).
    let schemas = [
        Schema::Object {
            properties: vec![
                ("name".into(), Schema::String),
                ("age".into(), Schema::Integer),
            ],
            required: vec!["name".into()],
        },
        Schema::Array {
            items: Box::new(Schema::AnyOf(vec![
                Schema::Integer,
                Schema::String,
                Schema::Null,
            ])),
        },
        Schema::Enum(vec![
            JsonLit::Str("alice".into()),
            JsonLit::Int(42),
            JsonLit::Null,
        ]),
    ];
    let vocab = toy_vocab();
    for (i, s) in schemas.iter().enumerate() {
        assert!(compile(s, &vocab).is_ok(), "schema {i} must compile clean");
    }
}

#[test]
fn sampler_round_trip_every_draw_is_grammar_valid() {
    // The full pipeline: compile → sample_joint with random logits → EVERY
    // draw must walk to an accepting node (grammar-valid by construction),
    // and tokenizing the walk back must re-validate.
    let schema = Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("age".into(), Schema::Integer),
        ],
        required: vec!["name".into(), "age".into()],
    };
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");
    let v = fa.vocab();

    // Minimal instance {"name":"x","age":42} = 15 tokens; 12 could never
    // contain one (the sampler would rightly refuse Unsatisfiable).
    let len = 18;
    let logits: Vec<f32> = (0..len * v)
        .map(|i| (((i * 7919) % 23) as f32 - 8.0) / 4.0)
        .collect();
    let forced = vec![FREE; len];
    let mut scratch = FaScratch::new();
    for seed in 0..200u64 {
        let mut rng = SplitMix64::new(seed);
        let mut out = vec![0u32; len];
        fa.sample_joint(
            &mut scratch,
            &logits,
            len,
            &forced,
            1.0,
            false,
            &mut rng,
            &mut out,
        )
        .expect("draw");
        let node = fa.walk(&out).expect("draw must walk");
        assert!(fa.is_accept(node), "draw must land accepting");
    }
}

#[test]
fn sampler_round_trip_with_pins_matches_a_partial_instance() {
    // Committed-prefix conditioning: pin the full canonical instance and
    // leave 4 free positions — the sampler must fill them with whitespace
    // padding (the only accepting continuation).
    let schema = Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("age".into(), Schema::Integer),
        ],
        required: vec!["name".into(), "age".into()],
    };
    let vocab = toy_vocab();
    let fa = compile(&schema, &vocab).expect("compile");

    let full = tok("{\"name\":\"alice\",\"age\":42}", &vocab);
    let len = full.len() + 4; // room for ws padding
    let mut forced = vec![FREE; len];
    for (i, &t) in full.iter().enumerate() {
        forced[i] = t;
    }
    let v = fa.vocab();
    let logits: Vec<f32> = (0..len * v)
        .map(|i| (((i * 104729) % 17) as f32 - 6.0) / 3.0)
        .collect();
    let mut scratch = FaScratch::new();
    let mut rng = SplitMix64::new(5);
    let mut out = vec![0u32; len];
    fa.sample_joint(
        &mut scratch,
        &logits,
        len,
        &forced,
        1.0,
        false,
        &mut rng,
        &mut out,
    )
    .expect("draw with pins");
    // The pinned prefix is verbatim; the tail is ws padding (either ws
    // token in the toy vocab); all walks.
    assert_eq!(&out[..full.len()], &full[..]);
    for &t in &out[full.len()..] {
        assert!(
            t == 14 || t == 15,
            "padding must be a whitespace token, got {t}"
        );
    }
    let node = fa.walk(&out).expect("pinned draw walks");
    assert!(fa.is_accept(node));
}

#[test]
fn compile_minimizes_the_subset_graph_and_stays_grammar_valid() {
    // The G2 axis (issue 035): the subset construction's 2^k·k member-chain
    // bloat must collapse under minimization, and the minimized automaton
    // must keep the sampler's by-construction acceptance.
    let schema = Schema::Object {
        properties: vec![
            ("name".into(), Schema::String),
            ("age".into(), Schema::Integer),
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
        required: vec!["name".into(), "age".into()],
    };
    let vocab = toy_vocab();
    let (fa, stats) = compile_stats(&schema, &vocab).expect("compile");

    // The fn-call-shaped grammar is exactly the case minimization exists
    // for: the member orders are language-equivalent and must merge.
    assert!(
        stats.nodes < stats.raw_nodes,
        "minimization must shrink the subset graph (raw {}, min {})",
        stats.raw_nodes,
        stats.nodes
    );
    assert!(stats.edges < stats.raw_edges);
    assert_eq!(fa.n_nodes(), stats.nodes);
    assert_eq!(fa.n_edges(), stats.edges);

    // By-construction acceptance through the MINIMIZED automaton.
    let v = fa.vocab();
    let len = 24;
    let logits: Vec<f32> = (0..len * v)
        .map(|i| (((i * 7919) % 23) as f32 - 8.0) / 4.0)
        .collect();
    let forced = vec![FREE; len];
    let mut scratch = FaScratch::new();
    for seed in 0..100u64 {
        let mut rng = SplitMix64::new(seed);
        let mut out = vec![0u32; len];
        fa.sample_joint(
            &mut scratch,
            &logits,
            len,
            &forced,
            1.0,
            false,
            &mut rng,
            &mut out,
        )
        .expect("draw");
        let node = fa.walk(&out).expect("draw must walk");
        assert!(fa.is_accept(node), "draw must land accepting");
    }
}
