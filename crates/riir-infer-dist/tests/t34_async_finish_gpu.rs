//! Plan 623 T3.4 — the GPU↔comm sync unit (issue 1151): the decode-feed
//! async-finish arm. Two gates on the REAL Bonsai-2 27B:
//!
//! 1. **Bit-identity** — `submit + schedule_async_finish + finish` must
//!    produce the same hidden bits as the synchronous `step`, per position:
//!    the async arm is scheduling only (the staged-copy future resolves to
//!    the same bytes the sync read copies).
//! 2. **The all-feed s=2 A/B** — `run_head_multi` + `run_tail_multi` over
//!    loopback TCP, two GPU streams, both postures (demote env vs armed):
//!    the armed posture submits every stream's chain before any readback,
//!    so stream i's readback+send overlaps stream i+1's chain on the shared
//!    queue. The read is a LATENCY measurement — quote the box state beside
//!    it (the G2 law); the bit-identity gate above is the correctness bar.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1151 cargo test -p riir-infer-dist --release \
//!   --features gpu,bonsai2_hadamard --test t34_async_finish_gpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` overrides the sibling-layout default;
//! `I623_T34_NEW` the decode step count (default 6).

#![cfg(feature = "gpu")]

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use riir_infer_core::deltanet::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::multi::{MultiRequest, run_head_multi, run_tail_multi};
use riir_infer_dist::runner::HeadOptions;
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, stage_scope};
use riir_infer_dist::stage_gpu::TernaryGpuStage;
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const SPLIT: u32 = 32;
/// Decode-step walk for the bit-identity arm (one warm + the measured).
const STEPS: usize = 6;
const PROMPT_A: [u32; 4] = [9707, 11, 1879, 0];
const PROMPT_B: [u32; 4] = [42, 100, 5, 7];

fn model_path() -> PathBuf {
    std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL))
}

fn env_or<T: std::str::FromStr>(var: &str, default: T) -> T {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn load_head(
    model: &std::path::Path,
) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let id = ArtifactIdentity::of_gguf(model).expect("artifact identity");
    // A throwaway config read for n_layer (the loader needs the scope's
    // n_layer; the GGUF carries it — load once unscoped-shaped by reading
    // the header through a scoped probe at the LAST layer first).
    let (mut config, w) = load_qwen_deltanet_ternary_weights_gguf_scoped(model, |c| {
        stage_scope(LayerRange::new(0, SPLIT), c.n_layer as u32)
    })
    .expect("load head slice");
    config.block_size = 256;
    (config, Arc::new(w), id)
}

/// Arm 1: the async arm is scheduling only — same hidden bits per position.
#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn t34_async_finish_hidden_bits_are_identical() {
    let (config, w, id) = load_head(&model_path());
    let n_layer = config.n_layer as u32;
    let mut sync_stage =
        TernaryGpuStage::new(&config, w.clone(), LayerRange::new(0, SPLIT), id).unwrap();
    let mut async_stage =
        TernaryGpuStage::new(&config, w.clone(), LayerRange::new(0, SPLIT), id).unwrap();
    assert!(!LayerRange::new(0, SPLIT).is_last(n_layer));

    let mut tok = PROMPT_A[0];
    for step in 0..STEPS {
        let pos = step;
        let sync_hidden: Vec<u32> = sync_stage
            .step(tok, pos, &[])
            .iter()
            .map(|f| f.to_bits())
            .collect();
        async_stage.step_submit(tok, pos, &[]);
        async_stage.schedule_async_finish();
        let async_hidden: Vec<u32> = async_stage
            .step_finish()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        assert_eq!(
            sync_hidden, async_hidden,
            "step {step}: the async finish changed the hidden bits"
        );
        tok = (tok * 31 + step as u32) % 100_000;
    }
    println!("t34 bit-identity: {STEPS} decode positions, hidden bits exact (split 0..{SPLIT})");
}

/// Arm 2: the all-feed s=2 A/B — demote env vs armed, same rig, same
/// prompts. The env flip happens between phases with BOTH runner threads
/// joined (the reader is this thread only at flip time — the unsafe window
/// is serial by construction).
#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn t34_all_feed_s2_ab() {
    let (config, w, id) = load_head(&model_path());
    let n_layer = config.n_layer as u32;
    let max_new = env_or("I623_T34_NEW", 8usize);

    let run_once = |new_max: usize| -> (Duration, Vec<Vec<u32>>, Vec<Duration>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let model = model_path();
        let tail_thread_config = config.clone();
        let tail = thread::spawn(move || {
            let (_, tw, tid) = {
                let id_t = ArtifactIdentity::of_gguf(&model).expect("artifact identity");
                let (mut cfg_t, w_t) =
                    load_qwen_deltanet_ternary_weights_gguf_scoped(&model, |c| {
                        stage_scope(LayerRange::new(SPLIT, c.n_layer as u32), c.n_layer as u32)
                    })
                    .expect("load tail slice");
                cfg_t.block_size = 256;
                (cfg_t, Arc::new(w_t), id_t)
            };
            let mut stages: Vec<Box<dyn StageCompute>> = (0..2)
                .map(|_| {
                    Box::new(
                        TernaryGpuStage::new(
                            &tail_thread_config,
                            tw.clone(),
                            LayerRange::new(SPLIT, n_layer),
                            tid,
                        )
                        .unwrap(),
                    ) as Box<dyn StageCompute>
                })
                .collect();
            let (sock, _) = listener.accept().unwrap();
            let mut link =
                TcpTransport::new(sock, PeerKey([22u8; 32]), Direction::Upstream).unwrap();
            let mut refs: Vec<&mut dyn StageCompute> = stages
                .iter_mut()
                .map(|b| b.as_mut() as &mut dyn StageCompute)
                .collect();
            run_tail_multi(&mut refs, &mut link, |_, _, _| {}).map(|_| ())
        });

        let mut stages: Vec<TernaryGpuStage> = (0..2)
            .map(|_| {
                TernaryGpuStage::new(&config, w.clone(), LayerRange::new(0, SPLIT), id).unwrap()
            })
            .collect();
        let sock = TcpStream::connect(addr).unwrap();
        let mut link = TcpTransport::new(sock, PeerKey([22u8; 32]), Direction::Downstream).unwrap();
        let reqs = vec![
            MultiRequest::new(PROMPT_A.to_vec(), new_max),
            MultiRequest::new(PROMPT_B.to_vec(), new_max),
        ];
        let t0 = Instant::now();
        let mut stage_refs: Vec<&mut TernaryGpuStage> = stages.iter_mut().collect();
        let reports = run_head_multi(
            stage_refs.as_mut_slice(),
            &mut link,
            &reqs,
            HeadOptions::default(),
        )
        .expect("head");
        let wall = t0.elapsed();
        tail.join().unwrap().expect("tail");
        let decode: Vec<Duration> = reports.iter().map(|r| r.decode).collect();
        (
            wall,
            reports.iter().map(|r| r.generated.clone()).collect(),
            decode,
        )
    };

    // Demote first (the incumbent cadence), then armed — each phase pays
    // its own two stage loads; the comparison is the DECODE WINDOW (the
    // reports carry it — the wall mixes the per-phase JIT/load noise).
    unsafe { std::env::set_var("RIIR_DIST_ASYNC_FINISH", "0") };
    let (wall_demote, toks_demote, dec_demote) = run_once(max_new);
    unsafe { std::env::remove_var("RIIR_DIST_ASYNC_FINISH") };
    let (wall_armed, toks_armed, dec_armed) = run_once(max_new);

    assert_eq!(
        toks_demote, toks_armed,
        "the async arm changed the generated tokens"
    );
    println!(
        "t34 s=2 all-feed A/B: wall demote {wall_demote:?} vs armed {wall_armed:?} · \
         decode demote {dec_demote:?} vs armed {dec_armed:?} — \
         tokens identical ({max_new} new × 2 streams)"
    );
}

use std::time::Duration;
