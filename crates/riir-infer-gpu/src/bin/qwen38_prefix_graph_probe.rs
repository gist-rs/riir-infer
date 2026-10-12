//! `qwen38_prefix_graph_probe` — Issue 041 T5: the graph-capture
//! co-existence re-proof on the real 27B. The issue-742 T4 harness is
//! gone (git history), so this is the rebuilt gate: cache hits must
//! survive under BOTH graph arms (the p=8 verify chunk graph + the p=1
//! decode graph), and the pointer-table copy path (issue 041 T1) must
//! behave exactly like the dtod loop beside them.
//!
//! The hazard this closes: a captured CUDA graph bakes the live GDN
//! state's buffer addresses at capture time. A prefix-cache restore that
//! ALLOCATED or MOVED live buffers would corrupt every subsequent replay.
//! The proof is the run, not the argument: fill a prefix, capture both
//! graphs, checkpoint the state, dirty it, restore the checkpoint, and
//! require the graphed verify chunk to replay BIT-IDENTICALLY — under the
//! dtod loop AND under `QWEN38_PREFIX_TABLE_COPY=1`.
//!
//! Sequence (one model boot, two postures). NOTE: the verify chunk COMMITS
//! its GDN state advance (the conv window + recurrence kernels run
//! in-place over the chunk), so the checkpoint is taken BEFORE the
//! reference verify — the only true prefix-end snapshot:
//!
//! * **A (dtod, env unset)** — eager prefix fill → checkpoint insert →
//!   capture the verify graph (`ref`, advances state to pos 16) → dirty
//!   via the decode graph (pos 16..24) → restore (dtod loop, back to
//!   pos 8) → graphed verify replay `== ref`.
//! * **B (table, env set)** — dirty (pos 24..32) → restore (the entry's
//!   tables build LAZILY here — T5's module change: the old loud refusal
//!   stranded pre-arm checkpoints) → replay `== ref` → then the A-pattern
//!   at base 16: refresh-insert (gather) → reference verify `ref2` →
//!   dirty (pos 32..40) → restore (scatter) → replay `== ref2`.
//!
//! The KV-lineage contract makes this sound: `note_write(pos)` prunes
//! checkpoints with `len > pos`, and the entry here has `len == 8` — the
//! dirty fills at pos ≥ 8 keep it alive, and the re-run verify chunk
//! rewrites its own KV rows before reading them.
//!
//! Skip-loud without a GPU or the model (the pyramid-capture default):
//!
//! ```text
//! # 4090 (PowerShell):
//! cargo run --release -p riir-infer-gpu --features ternary_gemv_cuda_raw `
//!   --bin qwen38_prefix_graph_probe
//! ```
//!
//! Env: `QWEN38_GGUF` (default `F:/models/qwen38-27b-dbirks-Q4_K_M.gguf`).

use riir_infer_gpu::qwen38_dense_cudarc::Qwen38DenseForward;
use riir_infer_gpu::qwen38_prefix_cache::{submit_probe_counts, submit_probe_reset};

const PREFIX_LEN: usize = 8;
const DIRTY_LEN: usize = 8;
const CTX_LEN: usize = 64;

/// LCG token stream — deterministic, inside the vocab.
struct Lcg(u64);

impl Lcg {
    fn next_token(&mut self, vocab: usize) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % vocab as u64) as u32
    }
}

fn main() {
    let gguf = std::env::var("QWEN38_GGUF")
        .unwrap_or_else(|_| "F:/models/qwen38-27b-dbirks-Q4_K_M.gguf".to_owned());
    if !std::path::Path::new(&gguf).exists() {
        eprintln!("[041-t5] SKIP (loud): no GPU model at {gguf} — set QWEN38_GGUF");
        return;
    }

    println!("[041-t5] booting {gguf} (ctx {CTX_LEN}) — ~15 GiB upload, idle GPU assumed");
    let boot = std::time::Instant::now();
    let mut fwd = Qwen38DenseForward::new(std::path::Path::new(&gguf), CTX_LEN)
        .unwrap_or_else(|e| panic!("[041-t5] boot failed: {e}"));
    let vocab = fwd.cfg.vocab_size;
    println!(
        "[041-t5] booted in {:.1}s (vocab {vocab}, n_embd {}, layers {})",
        boot.elapsed().as_secs_f32(),
        fwd.cfg.n_embd,
        fwd.cfg.layer_types.len()
    );

    let mut rng = Lcg(0x041_5EED);
    let prefix: Vec<u32> = (0..PREFIX_LEN).map(|_| rng.next_token(vocab)).collect();
    let probe: Vec<u32> = (0..PREFIX_LEN).map(|_| rng.next_token(vocab)).collect();
    let junk: Vec<u32> = (0..DIRTY_LEN).map(|_| rng.next_token(vocab)).collect();

    // ── Phase A — the dtod loop (env unset) ──
    unsafe { std::env::remove_var("QWEN38_PREFIX_TABLE_COPY") };
    println!("\n[041-t5] phase A: dtod posture (env unset)");

    for (i, &t) in prefix.iter().enumerate() {
        fwd.forward_token(t, i)
            .unwrap_or_else(|e| panic!("prefix fill {i}: {e}"));
    }
    println!("[041-t5] prefix filled (pos 0..{PREFIX_LEN})");

    // Checkpoint FIRST — the verify chunk COMMITS its GDN state advance
    // (conv window + recurrence run in-place over the chunk), so the only
    // true prefix-end snapshot is the one taken before any verify runs.
    fwd.prefix_cache_insert(&prefix)
        .unwrap_or_else(|e| panic!("insert: {e}"));
    println!(
        "[041-t5] checkpoint inserted ({} entr{}) — probe {:?}",
        fwd.prefix_cache_len(),
        if fwd.prefix_cache_len() == 1 {
            "y"
        } else {
            "ies"
        },
        submit_probe_counts()
    );

    // Capture the p=8 verify graph — the reference output. This ADVANCES
    // the GDN state through the probe tokens (pos 8..16) — by design; the
    // checkpoint above holds the pos-8 state.
    let t0 = std::time::Instant::now();
    let ref_out = fwd
        .forward_verify_chunk_graph(&probe, PREFIX_LEN)
        .unwrap_or_else(|e| panic!("ref verify: {e}"));
    println!(
        "[041-t5] verify graph captured + ref run in {:.1} ms → {:?}",
        t0.elapsed().as_secs_f32() * 1e3,
        ref_out
    );

    // Dirty the state through the decode graph, continuing AFTER the
    // verify's own advance (pos 16..24) — a consistent decode walk whose
    // only job is to scramble everything the restore must fix.
    for (i, &t) in junk.iter().enumerate() {
        fwd.forward_token_graph(t, 2 * PREFIX_LEN + i)
            .unwrap_or_else(|e| panic!("dirty fill {i}: {e}"));
    }
    println!(
        "[041-t5] state dirtied via the decode graph (pos {}..{})",
        2 * PREFIX_LEN,
        2 * PREFIX_LEN + DIRTY_LEN
    );

    let matched = fwd
        .prefix_cache_restore(&prefix)
        .unwrap_or_else(|e| panic!("restore: {e}"));
    assert_eq!(matched, Some(PREFIX_LEN), "prefix hit expected");
    println!(
        "[041-t5] restore (dtod) matched {matched:?} — probe {:?}",
        submit_probe_counts()
    );

    let out = fwd
        .forward_verify_chunk_graph(&probe, PREFIX_LEN)
        .unwrap_or_else(|e| panic!("replay verify: {e}"));
    assert_eq!(
        out, ref_out,
        "dtod restore must replay the graphed verify bit-identically"
    );
    println!("[041-t5] A PASS: graphed verify replay bit-identical after the dtod restore");

    // ── Phase B — the pointer-table copy path (env set) ──
    // The entry was born on the dtod path; the restore's lazy table build
    // (T5) arms it here — dirty, restore (table scatter), replay.
    unsafe { std::env::set_var("QWEN38_PREFIX_TABLE_COPY", "1") };
    println!("\n[041-t5] phase B: table posture (QWEN38_PREFIX_TABLE_COPY=1)");

    for (i, &t) in junk.iter().enumerate() {
        fwd.forward_token_graph(t, 3 * PREFIX_LEN + i)
            .unwrap_or_else(|e| panic!("dirty fill B{i}: {e}"));
    }
    submit_probe_reset();
    let matched = fwd
        .prefix_cache_restore(&prefix)
        .unwrap_or_else(|e| panic!("restore B: {e}"));
    assert_eq!(
        matched,
        Some(PREFIX_LEN),
        "prefix hit expected (table posture)"
    );
    println!(
        "[041-t5] restore (table scatter; lazy entry-table build) matched {matched:?} — probe {:?} (expected [0, 0, 2, 4]: one scatter + one-time engine/entry-table htods)",
        submit_probe_counts()
    );
    let out2 = fwd
        .forward_verify_chunk_graph(&probe, PREFIX_LEN)
        .unwrap_or_else(|e| panic!("replay verify B: {e}"));
    assert_eq!(
        out2, ref_out,
        "table restore must replay the graphed verify bit-identically"
    );
    println!("[041-t5] B1 PASS: graphed verify replay bit-identical after the table restore");

    // One more full gather+scatter cycle under the graphs — the A-pattern
    // at base 16 (the state B1 left behind): refresh-insert FIRST (the
    // only true snapshot of the current state), then the reference verify
    // (which advances the state), then dirty, restore, replay.
    let probe2: Vec<u32> = (0..PREFIX_LEN).map(|_| rng.next_token(vocab)).collect();
    submit_probe_reset();
    fwd.prefix_cache_insert(&prefix)
        .unwrap_or_else(|e| panic!("insert B2: {e}"));
    println!(
        "[041-t5] checkpoint refreshed (table gather) — probe {:?} (expected [0, 0, 2, 0]: one gather, all tables warm)",
        submit_probe_counts()
    );
    let ref2 = fwd
        .forward_verify_chunk_graph(&probe2, 2 * PREFIX_LEN)
        .unwrap_or_else(|e| panic!("ref2 verify: {e}"));
    for (i, &t) in junk.iter().enumerate() {
        fwd.forward_token_graph(t, 4 * PREFIX_LEN + i)
            .unwrap_or_else(|e| panic!("dirty fill B2-{i}: {e}"));
    }
    let matched = fwd
        .prefix_cache_restore(&prefix)
        .unwrap_or_else(|e| panic!("restore B2: {e}"));
    assert_eq!(
        matched,
        Some(PREFIX_LEN),
        "prefix hit expected (table posture)"
    );
    println!(
        "[041-t5] restore (table scatter) matched {matched:?} — probe {:?} (expected [0, 0, 2, 0])",
        submit_probe_counts()
    );
    let out3 = fwd
        .forward_verify_chunk_graph(&probe2, 2 * PREFIX_LEN)
        .unwrap_or_else(|e| panic!("replay verify B2: {e}"));
    assert_eq!(
        out3, ref2,
        "second table restore must replay bit-identically"
    );
    println!(
        "[041-t5] B2 PASS: insert (gather) + restore (scatter) cycle under the graphs is bit-identical"
    );

    unsafe { std::env::remove_var("QWEN38_PREFIX_TABLE_COPY") };
    println!(
        "\n[041-t5] ALL PASS — issue 041 T5: cache hits co-exist with both graph arms, both copy paths"
    );
}
