//! Issue 020 T6 sizing probe — the host/GPU split of one encoder forward.
//! MEASUREMENT ONLY (`#[ignore]`d, never a gate): run explicitly with
//! `cargo test --release --features laya-riir-metal --test metal_host_gpu_split
//! -- --ignored --nocapture`.
//!
//! T6's premise is that per-forward allocation churn (~60 MB of host `Vec`s
//! rebuilt per forward + every activation device buffer freed by the
//! per-pass chain clear) costs case wall. Both halves are HOST-side work,
//! and the Metal pipeline runs the host ahead of the GPU — so churn can
//! only matter if the host-enqueue phase is a material fraction of the
//! forward's wall. This probe measures the split directly at the real
//! english geometry (d 1024, 28 layers, intermediate 2624):
//!
//! - `enq`  — the wall of `Encoder::forward_packed` alone. The encoder body
//!   contains NO sync (ops enqueue and return device handles), so this is
//!   the whole host side of the encoder: MSL dispatch encoding, the chain
//!   uploads + destination slots (the T6 device-alloc surface), and the
//!   host `Vec` churn (the T6 host-alloc surface).
//! - `sync` — the wall of `download_into` on the forward's output: the
//!   commit + wait drains every enqueued dispatch, i.e. the GPU side.
//!
//! Decision rule (recorded before measuring): allocation pooling can shrink
//! only part of `enq`. If `enq` is a small fraction of `enq + sync` — even
//! under the CPU load this run records — the host is not the bottleneck and
//! T6 closes NEGATIVE (the 09-25 head scratch-pool rung's verdict, at the
//! encoder's scale). If `enq` is comparable to `sync`, the verdict is
//! DEFERRED to a quiet box (load inflates the host side; a host-bound
//! reading under load is not evidence).
//!
//! Box state is recorded beside every published figure (AGENTS.md G2 rule):
//! quote `uptime` + free RAM from the run's shell, not from memory.
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::config::{load_checkpoint_configs, Checkpoint};
use riir_infer_laya::laya::riir::backend::Backend;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::metal::Metal;
use riir_infer_laya::laya::riir::weights as ckpt_weights;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

const ROUNDS: usize = 9;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
#[ignore = "measurement-only sizing probe (issue 020 T6) — run with --ignored --nocapture"]
fn t6_host_gpu_split() {
    let ckpt = Checkpoint::English;
    let dir = ensure_checkpoint(&weights_root(), ckpt).expect("checkpoint present");
    let name = ckpt.subfolder();
    let (_agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name).expect("configs");
    let mut raw = ckpt_weights::load(&dir.join("model.safetensors"), name).expect("weights");
    let enc = Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder");
    let backend = Metal::new().expect("metal backend");
    // Weight residency is a load cost (T1) — keep it off the probe.
    enc.warm(&backend);

    let vocab = 50368usize;
    let shapes: &[(&str, &[usize])] = &[
        ("seq46", &[46]),
        ("seq106", &[106]),
        ("seq188", &[188]),
        ("seq512", &[512]),
        ("packed2x256", &[256, 256]),
    ];

    println!(
        "t6 probe: english geometry, {} rounds/shape, one forward per round \
         (begin_pass → enq → download-drain)",
        ROUNDS
    );
    for (label, seqs) in shapes {
        let total: usize = seqs.iter().sum();
        let ids: Vec<u32> = (0..total).map(|i| ((i * 7919) % vocab) as u32).collect();
        // Two warmups: first-miss chain slots + pipeline warm-in are not
        // the quantity being split.
        for _ in 0..2 {
            backend.begin_pass();
            let h = enc.forward_packed(&backend, &ids, seqs).expect("warm forward");
            let mut out = vec![0f32; h.len()];
            backend.download_into(&h, &mut out);
        }
        let mut enq = Vec::with_capacity(ROUNDS);
        let mut sync = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            backend.begin_pass();
            let t0 = std::time::Instant::now();
            let h = enc.forward_packed(&backend, &ids, seqs).expect("forward");
            let e = t0.elapsed().as_secs_f64() * 1e3;
            let mut out = vec![0f32; h.len()];
            let t1 = std::time::Instant::now();
            backend.download_into(&h, &mut out);
            let s = t1.elapsed().as_secs_f64() * 1e3;
            enq.push(e);
            sync.push(s);
        }
        let (e_med, s_med) = (median(&mut enq.clone()), median(&mut sync.clone()));
        let e_min = enq.iter().cloned().fold(f64::INFINITY, f64::min);
        let s_min = sync.iter().cloned().fold(f64::INFINITY, f64::min);
        let host_share = e_med / (e_med + s_med) * 100.0;
        println!(
            "[{label:>12}] enq p50 {e_med:7.2}ms (min {e_min:7.2}) · \
             sync p50 {s_med:7.2}ms (min {s_min:7.2}) · host share {host_share:4.1}%"
        );
        println!(
            "[{label:>12}] enq rounds: {}",
            enq.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(" ")
        );
        println!(
            "[{label:>12}] sync rounds: {}",
            sync.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(" ")
        );
    }
    println!("done — record load average + free RAM beside these numbers");
}
