//! Plan 623 T2.5 — the split planner end to end on the CPU stage: the
//! boot probe produces per-split decode s/token records for candidate
//! splits, and the pure planner merges them (with a synthetic link) into
//! the chosen split. Times are real but tiny — the assertion is the
//! PLUMBING (records feed the decision; the memory axis refuses), not a
//! latency claim.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::sync::Arc;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::planner::{HostProbe, LinkCost, plan_split, probe_decode_spt};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};

fn load_scoped(
    name: &str,
    scope: impl FnOnce(&Config) -> WeightScope,
) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp(name, &bytes);
    let (c, w) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(&path, scope)
            .expect("load synth");
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let _ = std::fs::remove_file(&path);
    (c, Arc::new(w), id)
}

fn scoped_stage(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    lo: u32,
    hi: u32,
) -> TernaryCpuStage {
    TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(lo, hi), id).unwrap()
}

#[test]
fn probe_then_plan_picks_a_common_split() {
    let n_layer = 4u32;
    let (config, w, id) = load_scoped("i623_planner_probe", |c| WeightScope::full(c.n_layer));

    // "host A" probes split 1; "host B" probes splits 1 and 3. The probe
    // is REAL (timed decode steps on the CPU stage); the planner must
    // consume the records and pick a k both hosts measured.
    let t_a1 = {
        let mut s = scoped_stage(&config, &w, id, 0, 1);
        let t = probe_decode_spt(&mut s, 0, 4, 2).expect("probe split 1 (head side)");
        drop(s);
        t
    };
    let (t_b1, t_b3) = {
        let mut s1 = scoped_stage(&config, &w, id, 1, n_layer);
        let t1 = probe_decode_spt(&mut s1, 0, 4, 2).expect("probe split 1 (tail side)");
        drop(s1);
        let mut s3 = scoped_stage(&config, &w, id, 3, n_layer);
        let t3 = probe_decode_spt(&mut s3, 0, 4, 2).expect("probe split 3 (tail side)");
        (t1, t3)
    };
    let a = HostProbe {
        host: "a".into(),
        rows: vec![(1, t_a1)],
        state_budget: None,
    };
    let b = HostProbe {
        host: "b".into(),
        rows: vec![(1, t_b1), (3, t_b3)],
        state_budget: None,
    };

    let link = LinkCost {
        rtt_s: 0.000_1,
        bw_bps: 1e9,
    };
    let plan = plan_split(&a, &b, link, config.n_embd, n_layer, 64, 1).expect("plan");
    assert_eq!(plan.k, 1, "only split 1 is common: {:?}", plan.table);
    assert_eq!(plan.table.len(), 1);
    // The prediction is max(head, tail) + hop — strictly above the
    // faster side's raw time, and equal to the formula.
    let want = t_a1.max(t_b1) + link.hop(4 * config.n_embd + 4);
    assert!((plan.s_per_tok - want).abs() < 1e-15);
    assert!(plan.s_per_tok >= t_a1 && plan.s_per_tok >= t_b1);
}

#[test]
fn probe_refuses_capacity_overrun_and_zero_steps() {
    let (config, w, id) = load_scoped("i623_planner_probe_cap", |c| WeightScope::full(c.n_layer));
    let mut s = scoped_stage(&config, &w, id, 0, config.n_layer as u32);
    let over = s.max_positions() + 10;
    let e = probe_decode_spt(&mut s, over, 4, 2).expect_err("span past capacity refuses");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    let e = probe_decode_spt(&mut s, 0, 0, 2).expect_err("zero steps refuses");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    // The happy path returns a positive finite time.
    let t = probe_decode_spt(&mut s, 0, 4, 2).expect("probe");
    assert!(t > 0.0 && t.is_finite());
}
