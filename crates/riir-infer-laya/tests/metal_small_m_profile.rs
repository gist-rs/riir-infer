//! Issue 020 T11 sizing probe — where the GPU time goes at the SMALL-m
//! shape (the arena's tetris spot sentence, 1 noul question, the widest
//! losing cell: rust Metal 22.2 vs torch MPS 16.2 ms p50, reflex-site
//! `68f056d`). MEASUREMENT ONLY (`#[ignore]`d, never a gate):
//!
//! ```sh
//! LAYA_METAL_PROFILE=1 cargo test --release --features laya-riir-metal \
//!     --test metal_small_m_profile -- --ignored --nocapture
//! ```
//!
//! Two readings, never pooled:
//! - `wall` — the shipped per-question wall (profile OFF in-process is not
//!   possible once the env is read, so run once WITHOUT the env for it);
//! - `profile` — per-(kernel, grid) GPU time with every dispatch in its own
//!   command buffer. That serializes the pass: read SHARES, not the total.
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use std::collections::BTreeMap;

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::agent::RiirAgent;
use riir_infer_laya::laya::riir::metal::profile_take;
use riir_infer_laya::laya::weights::weights_root;
use serde_json::json;

const SENTENCES: &[&str] = &[
    "The piece leaves no holes under it on the left edge, makes a small bump on top, and the stack stays low.",
    "The piece leaves one hole under it in the middle, makes a big bump on top, and the stack stands medium.",
    "The piece fills one row, leaves no holes under it on the right side, and the stack stays low.",
];
const ROUNDS: usize = 30;

/// (kernel, dispatch grid) → (dispatches, GPU seconds).
type ShapeTally<'a> = BTreeMap<(&'a str, (u64, u64, u64)), (usize, f64)>;

#[test]
#[ignore = "measurement-only sizing probe (issue 020 T11) — run with --ignored --nocapture"]
fn t11_small_m_profile() {
    let agent = RiirAgent::load(&weights_root(), Checkpoint::English).expect("agent");
    assert_eq!(agent.device(), "metal", "run on the Metal posture");
    let q = json!({"type": "noul", "instructions": "Does the stack look clean?"});
    let states: Vec<_> = SENTENCES.iter().map(|s| json!(s)).collect();
    for s in &states {
        for _ in 0..3 {
            agent.forward_question(s, &q).expect("warm");
        }
    }
    let _ = profile_take();
    let mut wall = Vec::with_capacity(ROUNDS * states.len());
    let mut seq = 0;
    for _ in 0..ROUNDS {
        for s in &states {
            let t = std::time::Instant::now();
            let f = agent.forward_question(s, &q).expect("forward");
            wall.push(t.elapsed().as_secs_f64() * 1e3);
            seq = f.seq_len;
        }
    }
    wall.sort_by(f64::total_cmp);
    let n_fwd = wall.len();
    println!(
        "t11: seq_len {seq} · {n_fwd} forwards · wall p50 {:.2} ms (min {:.2}) · profile {}",
        wall[n_fwd / 2],
        wall[0],
        if std::env::var("LAYA_METAL_PROFILE").as_deref() == Ok("1") {
            "ON (serialized — read shares)"
        } else {
            "off"
        }
    );
    let rows = profile_take();
    if rows.is_empty() {
        return;
    }
    let mut by_kernel: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    let mut by_shape: ShapeTally = BTreeMap::new();
    let total: f64 = rows.iter().map(|r| r.gpu_s).sum();
    for r in &rows {
        let e = by_kernel.entry(r.kernel).or_default();
        e.0 += 1;
        e.1 += r.gpu_s;
        let e = by_shape.entry((r.kernel, r.grid)).or_default();
        e.0 += 1;
        e.1 += r.gpu_s;
    }
    let per = |t: f64| t * 1e3 / n_fwd as f64;
    println!(
        "GPU sum {:.2} ms/forward over {} dispatches/forward",
        per(total),
        rows.len() / n_fwd
    );
    let mut k: Vec<_> = by_kernel.into_iter().collect();
    k.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    println!(
        "{:<22} {:>8} {:>10} {:>7} {:>9}",
        "kernel", "n/fwd", "ms/fwd", "share", "µs/disp"
    );
    for (name, (n, t)) in &k {
        println!(
            "{name:<22} {:>8} {:>10.3} {:>6.1}% {:>9.1}",
            n / n_fwd,
            per(*t),
            t / total * 100.0,
            t / *n as f64 * 1e6
        );
    }
    let mut sh: Vec<_> = by_shape.into_iter().collect();
    sh.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    println!("top (kernel, grid) instances:");
    for ((name, g), (n, t)) in sh.iter().take(14) {
        println!(
            "  {name:<18} grid {:>5}x{:<4}x{:<3} n/fwd {:>4} ms/fwd {:>7.3} µs/disp {:>7.1}",
            g.0,
            g.1,
            g.2,
            n / n_fwd,
            per(*t),
            t / *n as f64 * 1e6
        );
    }
}
