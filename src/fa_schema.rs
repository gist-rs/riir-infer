//! JSON-schema subset → token automaton (issue 035 P2) — the schema front
//! end for the FA-constrained decode lane.
//!
//! Compiles a [`Schema`] into an [`Automaton`] over MODEL TOKEN ids whose
//! accepted token sequences are exactly the tokenizations of valid JSON
//! instances of the schema. Pipeline (mosaic `grammar/json_schema.py`'s
//! reference shape, re-derived):
//!
//! 1. [`Schema`] → Thompson NFA over chars (shared subgraphs, single final
//!    state per fragment). Supported: the six JSON types, `enum` literals,
//!    `anyOf`, object `properties`/`required` (any-order, at-most-once
//!    members via bitmask NFA states), array `items`, and trailing
//!    whitespace — which doubles as the block-padding language (JSON's own
//!    grammar allows it, so a block longer than the instance pads with
//!    whitespace tokens and the walk stays accepting).
//! 2. NFA → subset graph, stepped ON DEMAND by the CONCRETE CHARS of token
//!    strings — no alphabet enumeration, no char-DFA table (the token
//!    re-alphabetization is the only consumer of the char-level machine).
//!    A token whose chars step the subset to the empty set is dead at that
//!    state. Accept states: subsets containing the NFA final.
//! 3. Subset graph → [`Automaton`]: one edge per (subset, subset) pair with
//!    the UNION of the tokens reaching it (merge is semantics-preserving —
//!    the merged tokens lead to the same state, so the path distribution is
//!    unchanged and the token draw is by LM probability inside the edge).
//!    The merge is what satisfies the automaton's ≤1-edge-per-(node,dst)
//!    determinism invariant, which the segment-tree lane requires.
//!
//! Bounds (each a named [`SchemaError`] refusal, never a silent
//! truncation): nesting depth, NFA size, subset-graph size, and object
//! property count — the bitmask construction is exponential in the member
//! count, so it is capped hard.

use crate::fa_posterior::{Automaton, AutomatonBuilder, FaError};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, VecDeque};

/// Maximum schema nesting depth (objects/arrays/anyOf).
pub const MAX_DEPTH: usize = 24;

/// Maximum NFA states before refusing (a mis-specified schema must fail at
/// compile time, never OOM). The object bitmask construction costs
/// O(2^k · k) member chains, so this sits above that curve at k = 10.
pub const MAX_NFA_STATES: usize = 400_000;

/// Maximum subset-graph states before refusing.
pub const MAX_SUBSETS: usize = 8_192;

/// Maximum object members — the bitmask construction is O(2^k) NFA states.
pub const MAX_PROPERTIES: usize = 10;

/// A JSON-literal value (`enum` payloads).
#[derive(Clone, Debug, PartialEq)]
pub enum JsonLit {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

/// The supported JSON-Schema subset.
#[derive(Clone, Debug)]
pub enum Schema {
    Null,
    Boolean,
    /// Any JSON number (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`).
    Number,
    /// Integers only (`-?[0-9]+`).
    Integer,
    /// A JSON string: `"` (escape ∪ non-quote-non-backslash-non-control)* `"`.
    String,
    /// Object with named properties; `required` keys must appear, every
    /// member at most once, any order.
    Object {
        properties: Vec<(String, Schema)>,
        required: Vec<String>,
    },
    /// Array whose elements all match `items` (unbounded length; the
    /// sampler's exact-length backward pass enforces block completability).
    Array {
        items: Box<Schema>,
    },
    /// Literal values (the `enum` keyword).
    Enum(Vec<JsonLit>),
    /// Union of schemas.
    AnyOf(Vec<Schema>),
}

#[derive(Debug, PartialEq)]
pub enum SchemaError {
    TooDeep {
        depth: usize,
        max: usize,
    },
    NfaTooLarge {
        n: usize,
        max: usize,
    },
    TooManyStates {
        n: usize,
        max: usize,
    },
    TooManyProperties {
        n: usize,
        max: usize,
    },
    /// A malformed schema the subset cannot express, or a compiler-bug
    /// invariant trip.
    InvalidSchema(String),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaError::TooDeep { depth, max } => {
                write!(f, "schema nesting depth {depth} exceeds max {max}")
            }
            SchemaError::NfaTooLarge { n, max } => {
                write!(f, "compiled NFA has {n} states, exceeds max {max}")
            }
            SchemaError::TooManyStates { n, max } => {
                write!(f, "token-subset graph has {n} states, exceeds max {max}")
            }
            SchemaError::TooManyProperties { n, max } => write!(
                f,
                "object has {n} properties, exceeds max {max} (bitmask construction is exponential)"
            ),
            SchemaError::InvalidSchema(what) => write!(f, "invalid schema: {what}"),
        }
    }
}

impl std::error::Error for SchemaError {}

// ── the NFA ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum Label {
    Eps,
    Ch(char),
    /// Inclusive char ranges.
    Class(Vec<(char, char)>),
}

impl Label {
    fn allows(&self, c: char) -> bool {
        match self {
            Label::Eps => false,
            Label::Ch(d) => *d == c,
            Label::Class(ranges) => ranges.iter().any(|&(a, b)| a <= c && c <= b),
        }
    }
}

/// Thompson-style NFA fragments: every `frag` is `(start, final)` with a
/// single final state; subgraphs may be shared (cycles included).
#[derive(Default)]
struct NfaBuilder {
    edges: Vec<(usize, Label, usize)>,
    states: usize,
}

impl NfaBuilder {
    fn state(&mut self) -> usize {
        let s = self.states;
        self.states += 1;
        s
    }

    fn eps(&mut self, a: usize, b: usize) {
        self.edges.push((a, Label::Eps, b));
    }

    fn ch(&mut self, c: char) -> (usize, usize) {
        let (a, b) = (self.state(), self.state());
        self.edges.push((a, Label::Ch(c), b));
        (a, b)
    }

    fn class(&mut self, ranges: &[(char, char)]) -> (usize, usize) {
        let (a, b) = (self.state(), self.state());
        self.edges.push((a, Label::Class(ranges.to_vec()), b));
        (a, b)
    }

    /// A fragment for a literal char sequence ("" = an ε self-state).
    fn lit(&mut self, s: &str) -> (usize, usize) {
        let mut frags: Vec<(usize, usize)> = Vec::new();
        for c in s.chars() {
            frags.push(self.ch(c));
        }
        match frags.is_empty() {
            true => {
                let a = self.state();
                (a, a)
            }
            false => self.seq(&frags),
        }
    }

    /// Chain fragments in sequence.
    fn seq(&mut self, frags: &[(usize, usize)]) -> (usize, usize) {
        match frags.split_first() {
            None => {
                let a = self.state();
                (a, a)
            }
            Some((&(s0, f0), rest)) => {
                let mut prev = f0;
                for &(s, f) in rest {
                    self.eps(prev, s);
                    prev = f;
                }
                (s0, prev)
            }
        }
    }

    /// Alternation (empty list compiles to a dead fragment).
    fn alt(&mut self, frags: &[(usize, usize)]) -> (usize, usize) {
        let (a, b) = (self.state(), self.state());
        for &(s, f) in frags {
            self.eps(a, s);
            self.eps(f, b);
        }
        (a, b)
    }

    /// Kleene star (zero or more; the body may be re-entered — shared).
    fn star(&mut self, (s, f): (usize, usize)) -> (usize, usize) {
        let (a, b) = (self.state(), self.state());
        self.eps(a, s);
        self.eps(f, b);
        self.eps(f, s);
        self.eps(a, b);
        (a, b)
    }

    /// Optional (zero or one).
    fn opt(&mut self, (s, f): (usize, usize)) -> (usize, usize) {
        let (a, b) = (self.state(), self.state());
        self.eps(a, s);
        self.eps(f, b);
        self.eps(a, b);
        (a, b)
    }

    /// One-or-more of `frag` (the fragment is shared between the first
    /// occurrence and the loop).
    fn one_or_more(&mut self, frag: (usize, usize)) -> (usize, usize) {
        let (a, b) = (self.state(), self.state());
        let looped = self.star(frag);
        self.eps(a, frag.0);
        self.eps(frag.1, looped.0);
        self.eps(looped.1, b);
        (a, b)
    }
}

// ── the grammar (a compact JSON reader) ─────────────────────────────

/// A FRESH whitespace star, built per call site. ⚠ Deliberately NOT shared:
/// a star fragment's final state carries the ε-exits of every consumer
/// sequence that links from it — sharing one wrapper across syntactic
/// positions would let any ws position ε-reach every other position's
/// continuation (measured: `[42,]` walked an Integer array). Char-class
/// fragments without ε-exits (digits, hex) remain shared — safe.
fn ws(b: &mut NfaBuilder) -> (usize, usize) {
    let arms = [b.ch(' '), b.ch('\t'), b.ch('\n'), b.ch('\r')];
    let alt = b.alt(&arms);
    b.star(alt)
}

/// A JSON string body: plain chars ∪ escapes (incl. `\uXXXX`).
fn string_frag(b: &mut NfaBuilder) -> (usize, usize) {
    let quote = b.ch('"');
    // Plain char: anything but ", \, and control (< 0x20).
    let plain = b.class(&[
        ('\u{20}', '\u{21}'),     // space, !
        ('\u{23}', '\u{5B}'),     // # .. [
        ('\u{5D}', '\u{10FFFF}'), // ] .. max (excludes \)
    ]);
    let hex = b.class(&[('0', '9'), ('a', 'f'), ('A', 'F')]);

    // Escape: \ ( " \ / b f n r t | u hex{4} ).
    let backslash = b.ch('\\');
    let simple_arms = [
        b.ch('"'),
        b.ch('\\'),
        b.ch('/'),
        b.ch('b'),
        b.ch('f'),
        b.ch('n'),
        b.ch('r'),
        b.ch('t'),
    ];
    let simple = b.alt(&simple_arms);
    let esc_fin = b.state();
    b.eps(backslash.1, simple.0);
    b.eps(simple.1, esc_fin);
    // \u + 4 hex digits (the hex fragment is shared across the 4 steps —
    // NFA cycles make that exact).
    let u = b.ch('u');
    b.eps(backslash.1, u.0);
    let mut cur = u.1;
    for _ in 0..4 {
        b.eps(cur, hex.0);
        let next = b.state();
        b.eps(hex.1, next);
        cur = next;
    }
    b.eps(cur, esc_fin);
    let esc = (backslash.0, esc_fin);

    let body_arms = [plain, esc];
    let body_alt = b.alt(&body_arms);
    let body = b.star(body_alt);
    let close = b.ch('"');
    let parts = [quote, body, close];
    b.seq(&parts)
}

/// A JSON number; `integer_only` = `-?[0-9]+` (no frac/exp).
fn number_frag(b: &mut NfaBuilder, integer_only: bool) -> (usize, usize) {
    let minus = b.ch('-');
    let digit = b.class(&[('0', '9')]);
    let nonzero = b.class(&[('1', '9')]);
    let zero = b.ch('0');
    let digits = b.star(digit);
    let lead = b.seq(&[nonzero, digits]);
    let lead_alt = [zero, lead];
    let int_part = b.alt(&lead_alt);
    let minus_opt = b.opt(minus);
    let mut parts = vec![minus_opt, int_part];
    if !integer_only {
        let dot = b.ch('.');
        let frac_digits = b.one_or_more(digit);
        let frac = b.seq(&[dot, frac_digits]);
        let e_arms = [b.ch('e'), b.ch('E')];
        let e = b.alt(&e_arms);
        let sign_arms = [b.ch('+'), b.ch('-')];
        let sign_alt = b.alt(&sign_arms);
        let sign = b.opt(sign_alt);
        let exp_digits = b.one_or_more(digit);
        let exp = b.seq(&[e, sign, exp_digits]);
        parts.push(b.opt(frac));
        parts.push(b.opt(exp));
    }
    b.seq(&parts)
}

// ── schema → NFA ─────────────────────────────────────────────────────

fn lit_frag(b: &mut NfaBuilder, lit: &JsonLit) -> Result<(usize, usize), SchemaError> {
    Ok(match lit {
        JsonLit::Str(s) => {
            let mut esc = String::with_capacity(s.len() + 2);
            esc.push('"');
            for c in s.chars() {
                match c {
                    '"' => esc.push_str("\\\""),
                    '\\' => esc.push_str("\\\\"),
                    '\n' => esc.push_str("\\n"),
                    '\r' => esc.push_str("\\r"),
                    '\t' => esc.push_str("\\t"),
                    c if (c as u32) < 0x20 => {
                        return Err(SchemaError::InvalidSchema(
                            "control char in enum string (use \\u escapes in the schema source)"
                                .to_string(),
                        ));
                    }
                    c => esc.push(c),
                }
            }
            esc.push('"');
            b.lit(&esc)
        }
        JsonLit::Int(i) => b.lit(&i.to_string()),
        JsonLit::Float(f) => b.lit(&f.to_string()),
        JsonLit::Bool(v) => b.lit(if *v { "true" } else { "false" }),
        JsonLit::Null => b.lit("null"),
    })
}

fn compile_value(
    b: &mut NfaBuilder,
    schema: &Schema,
    depth: usize,
) -> Result<(usize, usize), SchemaError> {
    if depth > MAX_DEPTH {
        return Err(SchemaError::TooDeep {
            depth,
            max: MAX_DEPTH,
        });
    }
    let frag = match schema {
        Schema::Null => b.lit("null"),
        Schema::Boolean => {
            let arms = [b.lit("true"), b.lit("false")];
            b.alt(&arms)
        }
        Schema::Integer => number_frag(b, true),
        Schema::Number => number_frag(b, false),
        Schema::String => string_frag(b),
        Schema::Enum(lits) => {
            let frags: Vec<(usize, usize)> = lits
                .iter()
                .map(|l| lit_frag(b, l))
                .collect::<Result<_, _>>()?;
            b.alt(&frags)
        }
        Schema::AnyOf(schemas) => {
            let frags: Vec<(usize, usize)> = schemas
                .iter()
                .map(|s| compile_value(b, s, depth + 1))
                .collect::<Result<_, _>>()?;
            b.alt(&frags)
        }
        Schema::Array { items } => {
            let item = compile_value(b, items, depth + 1)?;
            // '[' ws ( item (ws , ws item)* )? ws ']'
            let open = b.ch('[');
            let ws0 = ws(b);
            let comma = b.ch(',');
            let ws1 = ws(b);
            let comma_item = b.seq(&[ws1, comma, ws1, item]);
            let comma_star = b.star(comma_item);
            let list = b.seq(&[item, comma_star]);
            let list_opt = b.opt(list);
            let ws2 = ws(b);
            let close = b.ch(']');
            let parts = [open, ws0, list_opt, ws2, close];
            b.seq(&parts)
        }
        Schema::Object {
            properties,
            required,
        } => {
            if properties.len() > MAX_PROPERTIES {
                return Err(SchemaError::TooManyProperties {
                    n: properties.len(),
                    max: MAX_PROPERTIES,
                });
            }
            compile_object(b, properties, required, depth)?
        }
    };
    if b.states > MAX_NFA_STATES {
        return Err(SchemaError::NfaTooLarge {
            n: b.states,
            max: MAX_NFA_STATES,
        });
    }
    Ok(frag)
}

/// Any-order, at-most-once members over bitmask NFA states.
///
/// ⚠ THE MASK INVARIANT: nothing between mask states may be SHARED. The
/// subset construction merges shared subgraphs, so a value fragment linked
/// from two origin masks funnels both into ONE successor subset — the post-
/// value state could not tell which members were already emitted, and every
/// required-complete close would leak into every path (measured:
/// `{"age":42}` accepted with `name` required). Each (origin mask, member)
/// pair therefore compiles a FRESH `"key" ws : ws value` chain exiting into
/// exactly one successor mask. Cost is O(2^k · k) member chains — the
/// reason [`MAX_PROPERTIES`] is small.
///
/// Per the [`ws`] leak rule every whitespace wrapper here is fresh;
/// fragment FINALS may multi-consume (reached only on completion), wrapper
/// finals may not. The close `ws }` is shared with a single consumer (the
/// `}`), entered only from required-complete masks.
fn compile_object(
    b: &mut NfaBuilder,
    properties: &[(String, Schema)],
    required: &[String],
    depth: usize,
) -> Result<(usize, usize), SchemaError> {
    let k = properties.len();
    let mut required_bits = 0usize;
    for req in required {
        let Some(idx) = properties.iter().position(|(name, _)| name == req) else {
            return Err(SchemaError::InvalidSchema(
                "required names a property absent from properties".to_string(),
            ));
        };
        required_bits |= 1usize << idx;
    }

    /// A FRESH `"key" ws : ws value` chain for one (origin mask, member).
    fn member_chain(
        b: &mut NfaBuilder,
        name: &str,
        schema: &Schema,
        depth: usize,
    ) -> Result<(usize, usize), SchemaError> {
        let mut key = String::with_capacity(name.len() + 2);
        key.push('"');
        for c in name.chars() {
            match c {
                '"' => key.push_str("\\\""),
                '\\' => key.push_str("\\\\"),
                c => key.push(c),
            }
        }
        key.push('"');
        let key_frag = b.lit(&key);
        let colon_ws0 = ws(b);
        let colon_ch = b.ch(':');
        let colon_ws1 = ws(b);
        let colon = b.seq(&[colon_ws0, colon_ch, colon_ws1]);
        let head = b.seq(&[key_frag, colon]);
        let value = compile_value(b, schema, depth + 1)?;
        b.eps(head.1, value.0);
        Ok((head.0, value.1))
    }

    let n_masks = 1usize << k;
    let mask_states: Vec<usize> = (0..n_masks).map(|_| b.state()).collect();
    let obj_fin = b.state();

    let brace_open = b.ch('{');
    let open_ws = ws(b);
    let open = b.seq(&[brace_open, open_ws]);
    b.eps(open.1, mask_states[0]);

    for mask in 0..n_masks {
        // First member (mask 0): direct entry, no preceding comma. Every
        // other mask: a fresh `ws , ws` separator, then fresh chains for the
        // still-unemitted members (the bitmask gates which).
        let entry_source = if mask == 0 {
            mask_states[0]
        } else {
            let sep_ws0 = ws(b);
            let comma = b.ch(',');
            let sep_ws1 = ws(b);
            let sep = b.seq(&[sep_ws0, comma, sep_ws1]);
            b.eps(mask_states[mask], sep.0);
            sep.1
        };
        for j in 0..k {
            if mask & (1usize << j) != 0 {
                continue; // at most once per member
            }
            let (name, schema) = &properties[j];
            let chain = member_chain(b, name, schema, depth)?;
            b.eps(entry_source, chain.0);
            b.eps(chain.1, mask_states[mask | (1usize << j)]);
        }
    }

    // Close: one shared `ws }` (single consumer), required-complete only.
    let brace_close = b.ch('}');
    let close_ws = ws(b);
    let close = b.seq(&[close_ws, brace_close]);
    b.eps(close.1, obj_fin);
    // (indexed: mask_states is addressed BY the mask bit-pattern — the
    // index IS the semantic value here.)
    #[allow(clippy::needless_range_loop)]
    for mask in 0..n_masks {
        if mask & required_bits == required_bits {
            b.eps(mask_states[mask], close.0);
        }
    }

    if b.states > MAX_NFA_STATES {
        return Err(SchemaError::NfaTooLarge {
            n: b.states,
            max: MAX_NFA_STATES,
        });
    }
    Ok((open.0, obj_fin))
}

/// The top-level language: `<instance> <ws>*` — the trailing-whitespace
/// star is the block-padding language (a block longer than the instance
/// pads with whitespace tokens and stays accepting).
fn top_level(
    b: &mut NfaBuilder,
    schema: &Schema,
    depth: usize,
) -> Result<(usize, usize), SchemaError> {
    let value = compile_value(b, schema, depth)?;
    let pad = ws(b);
    b.eps(value.1, pad.0);
    Ok((value.0, pad.1))
}

// ── subset graph over token strings ──────────────────────────────────

/// Per-state out-edge lists (grouped at build time for move/closure).
struct NfaIndex {
    out: Vec<Vec<(Label, usize)>>,
}

impl NfaIndex {
    fn build(edges: Vec<(usize, Label, usize)>, n_states: usize) -> Self {
        let mut out = vec![Vec::new(); n_states];
        for (a, label, b) in edges {
            out[a].push((label, b));
        }
        Self { out }
    }

    fn move_state(&self, q: usize, c: char, dst: &mut Vec<usize>) {
        for (label, d) in &self.out[q] {
            if label.allows(c) {
                dst.push(*d);
            }
        }
    }

    /// ε-closure of a state set (result: sorted, deduped). `seen` is a
    /// generation-stamped visit buffer owned by the caller's [`Stepper`] —
    /// one allocation for the whole compile, `stamp` must differ from every
    /// stamp still present in `seen` (the caller bumps per call).
    fn closure_into(&self, set: &mut Vec<usize>, seen: &mut [u32], stamp: u32) {
        let mut stack: Vec<usize> = std::mem::take(set);
        stack.sort_unstable();
        stack.dedup();
        let mut result = Vec::new();
        while let Some(q) = stack.pop() {
            if seen[q] == stamp {
                continue;
            }
            seen[q] = stamp;
            result.push(q);
            for (label, d) in &self.out[q] {
                if matches!(label, Label::Eps) && seen[*d] != stamp {
                    stack.push(*d);
                }
            }
        }
        result.sort_unstable();
        *set = result;
    }
}

/// The subset-machine driver: one [`NfaIndex`] plus reusable scratch, so a
/// real-vocabulary compile (10⁵ tokens × 10³ subsets) does not allocate a
/// fresh `seen` bitmap per char step (the P2.6 law — the naive per-call
/// allocation was the compile-time pathology).
struct Stepper<'a> {
    index: &'a NfaIndex,
    /// Generation-stamped visit marks for [`NfaIndex::closure_into`] — one
    /// buffer for the whole compile, stamped per call.
    seen: Vec<u32>,
    stamp: u32,
    /// `move_state` accumulator, reused per char step.
    moved: Vec<usize>,
}

impl<'a> Stepper<'a> {
    fn new(index: &'a NfaIndex) -> Self {
        Self {
            index,
            seen: vec![0; index.out.len()],
            stamp: 0,
            moved: Vec::new(),
        }
    }

    /// Step one subset over one char: `move` then `ε-closure`.
    /// `None` = the char kills every state (dead).
    fn step(&mut self, cur: &[usize], c: char) -> Option<Vec<usize>> {
        self.moved.clear();
        for &q in cur {
            self.index.move_state(q, c, &mut self.moved);
        }
        if self.moved.is_empty() {
            return None;
        }
        self.begin_stamp();
        let mut out = std::mem::take(&mut self.moved);
        let stamp = self.stamp;
        self.index.closure_into(&mut out, &mut self.seen, stamp);
        Some(out)
    }

    /// ε-closure only (the start subset's preparation).
    fn closure_only(&mut self, set: &mut Vec<usize>) {
        self.begin_stamp();
        let stamp = self.stamp;
        self.index.closure_into(set, &mut self.seen, stamp);
    }

    fn begin_stamp(&mut self) {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            // u32 stamps exhausted — reset the buffer cleanly.
            self.seen.iter_mut().for_each(|v| *v = 0);
            self.stamp = 1;
        }
    }
}

/// Intern a step-result set by content; returns the pool id (stable for
/// the whole compile, so the char-step memo can key on it).
fn pool_intern(
    pool_sets: &mut Vec<Vec<usize>>,
    pool: &mut HashMap<Vec<usize>, u32>,
    s: Vec<usize>,
) -> u32 {
    if let Some(&idx) = pool.get(&s) {
        return idx;
    }
    let idx = pool_sets.len() as u32;
    pool.insert(s.clone(), idx);
    pool_sets.push(s);
    idx
}

/// Trie over the token strings: shared prefixes are single nodes, so a
/// subset's token walk visits (trie node × subsets-that-reach-it) instead
/// of (token × subset × token length). Children are kept char-sorted at
/// insert; `terminals[node]` holds the token ids ENDING there (a node can
/// be both terminal and internal — one token may be another's prefix).
#[derive(Default)]
struct TokenTrie {
    children: Vec<Vec<(char, u32)>>,
    terminals: Vec<Vec<u32>>,
}

impl TokenTrie {
    fn new() -> Self {
        Self {
            children: vec![Vec::new()],
            terminals: vec![Vec::new()],
        }
    }

    fn insert(&mut self, tid: u32, text: &str) {
        let mut node = 0usize;
        for c in text.chars() {
            match self.children[node].binary_search_by_key(&c, |&(ch, _)| ch) {
                Ok(i) => node = self.children[node][i].1 as usize,
                Err(i) => {
                    let kid = self.children.len() as u32;
                    self.children.push(Vec::new());
                    self.terminals.push(Vec::new());
                    self.children[node].insert(i, (c, kid));
                    node = kid as usize;
                }
            }
        }
        self.terminals[node].push(tid);
    }
}

/// Compile shape counts — the raw subset-graph automaton vs the minimized
/// one `compile` returns (issue 035, the G2 axis: the sampler's per-step
/// cost is O(L·E) over the MINIMIZED automaton).
#[derive(Clone, Copy, Debug)]
pub struct CompileStats {
    /// NFA states the schema compiled to (pre-subset-construction).
    pub nfa_states: usize,
    /// Subset-graph nodes/edges before minimization.
    pub raw_nodes: usize,
    pub raw_edges: usize,
    /// Minimized nodes/edges — what the returned [`Automaton`] carries.
    pub nodes: usize,
    pub edges: usize,
}

/// Compile `schema` into a token automaton over `tokens` — `(token id,
/// token string)` pairs — reporting raw vs minimized shape. The returned
/// [`Automaton`] plugs straight into the `fa_constraint` decode lane
/// (`FaConstraintConfig::new(&automaton)`).
///
/// The result is minimized ([`crate::fa_minimize`]): language-preserving,
/// distribution-preserving for the exact joint sampler, and it collapses
/// the subset construction's 2^k·k member-chain bloat. A schema whose
/// automaton has no accepting path from the start (accepts no instance)
/// errors here instead of at the first decode draw.
pub fn compile_stats(
    schema: &Schema,
    tokens: &[(u32, &str)],
) -> Result<(Automaton, CompileStats), SchemaError> {
    let mut b = NfaBuilder::default();
    let (start, fin) = top_level(&mut b, schema, 0)?;
    let index = NfaIndex::build(b.edges, b.states);

    // Subset BFS. State sets are sorted deduped Vec<usize>; ids dense.
    let mut subsets: Vec<Vec<usize>> = Vec::new();
    let mut ids: HashMap<Vec<usize>, u32> = HashMap::new();
    let mut accept: Vec<bool> = Vec::new();
    // (from subset id, to subset id) → token ids (sorted+deduped on emit).
    let mut edges: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();

    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut queued: Vec<bool> = Vec::new();

    let id_of = |subsets: &mut Vec<Vec<usize>>,
                 ids: &mut HashMap<Vec<usize>, u32>,
                 accept: &mut Vec<bool>,
                 queued: &mut Vec<bool>,
                 s: Vec<usize>|
     -> Result<u32, SchemaError> {
        if let Some(&id) = ids.get(&s) {
            return Ok(id);
        }
        if subsets.len() >= MAX_SUBSETS {
            return Err(SchemaError::TooManyStates {
                n: subsets.len() + 1,
                max: MAX_SUBSETS,
            });
        }
        let id = subsets.len() as u32;
        let acc = s.contains(&fin);
        subsets.push(s.clone());
        accept.push(acc);
        queued.push(false);
        ids.insert(s, id);
        Ok(id)
    };

    let mut stepper = Stepper::new(&index);
    let start_subset = {
        let mut s = vec![start];
        stepper.closure_only(&mut s);
        s
    };

    // Token trie + step memo (P2.6 — the real-vocabulary machinery). The
    // trie dedups shared token prefixes; `steps` memoizes every char step
    // by (source subset, char); `step_sets`/`step_pool` intern step RESULTS
    // by content so memo values stay identity-stable across walks. All
    // compile-time only; none of it iterates, so ordering stays
    // deterministic.
    let mut trie = TokenTrie::new();
    for &(tid, text) in tokens {
        if !text.is_empty() {
            trie.insert(tid, text);
        }
    }
    let mut steps: HashMap<(u32, char), Option<u32>> = HashMap::new();
    let mut step_pool: HashMap<Vec<usize>, u32> = HashMap::new();
    let mut step_sets: Vec<Vec<usize>> = Vec::new();
    let start_id = id_of(
        &mut subsets,
        &mut ids,
        &mut accept,
        &mut queued,
        start_subset,
    )?;
    queue.push_back(start_id);
    queued[start_id as usize] = true;

    while let Some(from) = queue.pop_front() {
        // `queued` means DISCOVERED (never reset on pop — a self-loop subset
        // would otherwise re-queue itself forever). Each subset is processed
        // exactly once; self-loops only add their edge.
        //
        // The token → target mapping is walked over a TRIE of the token
        // strings, DFS from (root, this subset), with every CHAR STEP
        // memoized on `(current subset, char)` (P2.6 — the real-vocabulary
        // law). A string-body self-loop class collapses its whole trie
        // subtree to one memo hit per distinct char, so the walk cost is
        // trie nodes × subsets-that-reach-them, never tokens × subsets ×
        // token length. Step results are interned in a content pool, so
        // the memo is identity-stable across source subsets. Targets are
        // gathered per pool id, then ordered by their SET (lexicographic
        // Vec order — exactly the BTreeMap order this loop used to intern
        // in), keeping subset numbering byte-identical with the P2 naive
        // walk.
        let mut by_target: HashMap<u32, Vec<u32>> = HashMap::new();
        let src = subsets[from as usize].clone();
        let src_idx = pool_intern(&mut step_sets, &mut step_pool, src);
        let mut stack: Vec<(u32, u32)> = vec![(0, src_idx)];
        while let Some((node, pidx)) = stack.pop() {
            let tids = &trie.terminals[node as usize];
            if !tids.is_empty() {
                by_target.entry(pidx).or_default().extend_from_slice(tids);
            }
            for &(c, kid) in &trie.children[node as usize] {
                // Memo key = (CURRENT pool subset, char) — the pool id is
                // content-interned and stable for the whole compile, so a
                // hit is the same step result the walk took before.
                let target = match steps.entry((pidx, c)) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(v) => {
                        let t = stepper
                            .step(&step_sets[pidx as usize], c)
                            .map(|s| pool_intern(&mut step_sets, &mut step_pool, s));
                        v.insert(t);
                        t
                    }
                };
                if let Some(t) = target {
                    stack.push((kid, t));
                }
            }
        }
        // Emit in set order (the P2 interning order).
        let mut targets: Vec<(Vec<usize>, Vec<u32>)> = by_target
            .into_iter()
            .map(|(pidx, mut tids)| {
                tids.sort_unstable();
                tids.dedup();
                (step_sets[pidx as usize].clone(), tids)
            })
            .collect();
        targets.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (target, tids) in targets {
            let to = id_of(&mut subsets, &mut ids, &mut accept, &mut queued, target)?;
            edges.entry((from, to)).or_default().extend(tids);
            if !queued[to as usize] {
                queued[to as usize] = true;
                queue.push_back(to);
            }
        }
    }

    // Assemble the Automaton. Vocab bound = max token id + 1 (the builder
    // only range-checks edge tokens against it).
    let vocab = tokens
        .iter()
        .map(|&(t, _)| t as usize)
        .max()
        .map_or(1, |m| m + 1);
    let mut ab = AutomatonBuilder::new(subsets.len(), vocab, start_id as usize);
    for (i, &a) in accept.iter().enumerate() {
        if a {
            ab.accept(i);
        }
    }
    for ((from, to), mut tids) in edges {
        tids.sort_unstable();
        tids.dedup();
        ab.edge(from as usize, to as usize, &tids);
    }
    let raw = ab.build().map_err(|_| {
        SchemaError::InvalidSchema("automaton invariants violated (compiler bug)".to_string())
    })?;
    let (fa, m) = crate::fa_minimize::minimize_with_stats(&raw).map_err(|e| match e {
        FaError::DeadStart(_) => SchemaError::InvalidSchema(
            "schema accepts no instance (no accepting path from the start)".to_string(),
        ),
        other => SchemaError::InvalidSchema(format!(
            "minimization rejected the compiled automaton: {other}"
        )),
    })?;
    Ok((
        fa,
        CompileStats {
            nfa_states: b.states,
            raw_nodes: m.nodes_in,
            raw_edges: m.edges_in,
            nodes: m.nodes_out,
            edges: m.edges_out,
        },
    ))
}

/// Compile `schema` (minimized; stats discarded).
pub fn compile(schema: &Schema, tokens: &[(u32, &str)]) -> Result<Automaton, SchemaError> {
    compile_stats(schema, tokens).map(|(fa, _)| fa)
}

#[cfg(test)]
mod tests;
