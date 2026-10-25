//! Issue 1118 Finding 1 — G4: the BPE merge loop is allocation-free in its
//! probe phase. Its OWN test target, deliberately (the fa_g4/twt precedent):
//! the counting allocator is process-global and parallel sibling tests
//! would leak allocations into any count taken beside them.
//!
//! Two properties, both on `encode` (the public surface):
//! 1. **Constant, not proportional** — the old shape allocated TWO heap
//!    Strings per adjacent pair PER merge iteration, so allocs scaled with
//!    chunk complexity (an n-byte run of mergeable pairs merges ~n/2
//!    times). The id-pair key probes a packed 8-byte key in registers: the
//!    merge-loop body allocates NOTHING — encode allocs stay flat as the
//!    merge depth grows.
//! 2. **Bounded absolute** — the whole encode of a mid-size text allocates
//!    a bounded handful of times (result-Vec growth), never per probe.
//!
//! Runs: `cargo test -p riir-infer-core --test tokenizer_bpe_g4_alloc`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

use riir_infer_core::tokenizer::BpeTokenizer;

/// The GPT-2 byte→unicode spelling — needed to name byte tokens in the
/// synthetic vocab (the standard pure function, duplicated test-side).
fn byte_to_unicode() -> Vec<String> {
    let mut out = Vec::with_capacity(256);
    let mut n = 0u32;
    for b in 0u8..=255 {
        let printable = b.is_ascii_alphanumeric() || (0x21..=0x7e).contains(&b);
        let c = if printable {
            b as char
        } else {
            char::from_u32(256 + n).expect("GPT-2 table codepoints are valid")
        };
        out.push(c.to_string());
        if !printable {
            n += 1;
        }
    }
    out
}

/// Vocab: the 256 byte tokens (spelled), then `th`, `he`, `the`, `e`.
/// Merge chain: t+h → th (rank 0), th+e → the (rank 1), h+e → he (rank 2)
/// — so a run of "the"s merges many times per chunk.
fn tiny_tokenizer() -> BpeTokenizer {
    let mut vocab = byte_to_unicode();
    let spell = |s: &str| -> String {
        // Spell an ASCII fragment through the same table.
        let table = byte_to_unicode();
        s.bytes().map(|b| table[b as usize].clone()).collect()
    };
    // Every merge's RESULT must be in the vocab (true of every real GPT-2
    // vocab — a rule producing a non-token is a degenerate fixture: both
    // keyings merge it then fall back to unk, but the token count asserts
    // below assume the sane shape).
    vocab.push(spell("th"));
    vocab.push(spell("he"));
    vocab.push(spell("the"));
    vocab.push(spell(" the"));
    let (ut, uh) = (spell("t"), spell("h"));
    let (ue, uth) = (spell("e"), spell("th"));
    let (uthe, usp) = (spell("the"), spell(" "));
    BpeTokenizer::from_parts_for_test(
        vocab,
        &[
            (ut.as_str(), uh.as_str()),    // t h -> th
            (uth.as_str(), ue.as_str()),   // th e -> the
            (uh.as_str(), ue.as_str()),    // h e -> he
            (uthe.as_str(), usp.as_str()), // the␣ -> " the" (crosses pre-tokens, never fires)
        ],
    )
}

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

#[test]
fn bpe_merge_loop_allocates_constant_not_proportional() {
    let tok = tiny_tokenizer();

    // Sanity: the chain merges — "the" is one token.
    assert_eq!(
        tok.encode("the"),
        vec![tok.token_to_id("the").expect("the in vocab")]
    );
    let the_id = tok.token_to_id("the").expect("the in vocab");

    // Warm every lazy path (result Vec growth, pretokenizer buffers).
    let _ = tok.encode("the the the");

    // Property 1: allocs stay FLAT as the merge-iteration count grows —
    // measured on ONE pre-token ("thethe…" — no spaces), so the per-chunk
    // setup is constant and the ONLY thing growing is the merge-loop
    // iteration count (~2n iterations at n repeats). The OLD code allocated
    // 2 Strings per adjacent pair per iteration here (thousands at n=512);
    // the id-pair probe allocates nothing per iteration.
    let mut before = allocs();
    let _ = tok.encode("the");
    let small = allocs() - before;

    before = allocs();
    let _ = tok.encode(&"the".repeat(64));
    let mid = allocs() - before;

    before = allocs();
    let big_ids = tok.encode(&"the".repeat(512));
    let big = allocs() - before;

    println!(
        "G4   single-chunk encode allocs: 1x {small}, 64x {mid}, 512x {big} ({} tokens out)",
        big_ids.len()
    );
    assert_eq!(
        big_ids.len(),
        512,
        "512 merged 'the' tokens from one chunk ({} for 1x)",
        {
            let _ = &big_ids;
            512
        }
    );
    assert!(
        big <= small + 8,
        "G4: 512× the merge iterations allocated {big}× vs {small}× for one — the merge loop still allocates per probe"
    );

    // Property 2 (the same axis at the extreme): one 3 KB single-word
    // chunk — thousands of merge iterations, everything outside the merge
    // loop allocated ONCE up front.
    let long_chunk = "the".repeat(1024);
    before = allocs();
    let ids = tok.encode(&long_chunk);
    let long_run = allocs() - before;
    println!(
        "G4   3 KB single-chunk encode: {} allocs total ({} tokens)",
        long_run,
        ids.len()
    );
    assert_eq!(
        *ids.last().expect("ids"),
        the_id,
        "the long run resolves to 'the' tokens"
    );
    assert!(
        long_run <= 16,
        "G4: a 3 KB chunk encode allocated {long_run}× — per-probe growth survived somewhere"
    );

    // Non-vacuity: the counter counts (an explicit allocation moves it).
    before = allocs();
    let _ = vec![0u8; 64];
    assert_eq!(
        allocs() - before,
        1,
        "the counting allocator stopped counting"
    );
}
