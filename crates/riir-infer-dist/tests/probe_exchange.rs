//! Plan 623 T2.5 — the probe EXCHANGE end to end over loopback TCP on
//! the CPU stage: the tail answers pings and one probe request with
//! REAL per-candidate probe times, the head measures the RTT on the
//! same link, composes `plan_split` live from the two records and the
//! formula is exact. The assertion is the PLUMBING (the wire step
//! replaces the hand-carried JSON), never a latency claim. The
//! refusal arms (out-of-range candidate, malformed records, model
//! mismatch) live in the module tests + here for the wire path.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::planner::{HostProbe, probe_decode_spt};
use riir_infer_dist::probe::{ExchangeCtx, run_probe_exchange, serve_probe_session};
use riir_infer_dist::stage::{ArtifactIdentity, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport, Transport};

fn load(name: &str) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp(name, &bytes);
    let (c, w) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(&path, |c| {
            WeightScope::full(c.n_layer)
        })
        .expect("load synth");
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let _ = std::fs::remove_file(&path);
    (c, Arc::new(w), id)
}

/// The tail's `probe_one`: scoped stage at `[k, n)`, timed decode —
/// the same loop the bin runs, at test size.
fn make_probe_one<'a>(
    config: &'a Config,
    w: &'a Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
) -> impl FnMut(u32) -> std::io::Result<f64> + 'a {
    let n = config.n_layer as u32;
    move |k: u32| {
        let mut st = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(k, n), id)
            .map_err(std::io::Error::other)?;
        probe_decode_spt(&mut st, 0, 4, 2)
    }
}

#[test]
fn exchange_over_loopback_plans_the_common_split() {
    let (config, w, id) = load("riir_dist_probe_exchange");
    let n_layer = config.n_layer as u32;
    let hidden = config.n_embd;
    let key = [21u8; 32];

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let tail = thread::spawn(move || {
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut probe_one = make_probe_one(&tc, &tw, id);
        serve_probe_session(&mut link, n_layer, tc.n_embd as u32, &mut probe_one)
    });

    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();

    // The head probes the SAME candidates it will request: splits 1
    // and 3 of a 4-layer model (its own ranges [0,1) and [0,3)).
    let mut own_rows = Vec::new();
    let head_probe = |k: u32| -> std::io::Result<f64> {
        let mut st = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, k), id)
            .map_err(std::io::Error::other)?;
        probe_decode_spt(&mut st, 0, 4, 2)
    };
    for &k in &[1u32, 3] {
        own_rows.push((k, head_probe(k).expect("head probe")));
    }
    let own = HostProbe {
        host: "head".into(),
        rows: own_rows,
        state_budget: None,
    };

    let ctx = ExchangeCtx {
        own,
        n_layer,
        hidden,
        bw_bps: 1e9,
        n_positions: 64,
        state_bytes_per_position_per_layer: 1,
        rtt_rounds: 4,
    };
    let out = run_probe_exchange(&mut link, &[1, 3], &ctx, &addr.to_string()).expect("exchange");
    link.send_frame(riir_infer_dist::transport::FrameKind::Shutdown, &[])
        .unwrap();
    tail.join().unwrap().expect("tail session clean");

    // The RTT was measured on the loopback link: finite, non-negative.
    assert!(out.rtt_s.is_finite() && out.rtt_s >= 0.0);
    // The plan's formula, exactly, over the two live records.
    let hop = out.rtt_s + (4 * hidden + 4) as f64 / 1e9;
    for (k, s) in &out.plan.table {
        let h = ctx.own.get(*k).unwrap();
        let t = out.tail.get(*k).unwrap();
        assert!((s - (h.max(t) + hop)).abs() < 1e-12, "split {k}");
    }
    assert_eq!(out.plan.table.len(), 2, "both candidates are common");
    // The chosen k is the table's argmin.
    let (want_k, _) = out
        .plan
        .table
        .iter()
        .copied()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert_eq!(out.plan.k, want_k);
}

#[test]
fn tail_session_refuses_out_of_range_candidate_over_the_wire() {
    let (config, w, id) = load("riir_dist_probe_exchange_oob");
    let n_layer = config.n_layer as u32;
    let key = [22u8; 32];

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let tail = thread::spawn(move || {
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut probe_one = make_probe_one(&tc, &tw, id);
        serve_probe_session(&mut link, n_layer, tc.n_embd as u32, &mut probe_one)
    });

    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();

    // Candidate k == n_layer is the 1-box reference row, never a
    // 2-stage candidate — the session refuses it loud (the head sees
    // the error surface, the connection dies).
    let mut buf = Vec::new();
    riir_infer_dist::probe::encode_request(&[n_layer], &mut buf);
    link.send_frame(riir_infer_dist::transport::FrameKind::ProbeRequest, &buf)
        .unwrap();
    let r = tail.join().unwrap();
    let msg = r.expect_err("out-of-range candidate refuses").to_string();
    assert!(
        msg.contains("outside 1.."),
        "refusal names the range: {msg}"
    );
}

#[test]
fn head_refuses_a_wrong_model_reply_over_the_wire() {
    // A rigged tail that answers a VALID request with a reply whose
    // n_layer field lies: the head's cross-check refuses the record —
    // a plan over two different models is a lie.
    let (config, _w, _id) = load("riir_dist_probe_exchange_lie");
    let n_layer = config.n_layer as u32;
    let hidden = config.n_embd;
    let key = [23u8; 32];

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let tail = thread::spawn(move || {
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        // Answer the (single) ping, then the request with a LYING reply.
        let mut buf = Vec::new();
        let kind = link.recv_frame_into(&mut buf).unwrap();
        assert_eq!(kind, riir_infer_dist::transport::FrameKind::ProbePing);
        link.send_frame(riir_infer_dist::transport::FrameKind::ProbePong, &[])
            .unwrap();
        let kind = link.recv_frame_into(&mut buf).unwrap();
        assert_eq!(kind, riir_infer_dist::transport::FrameKind::ProbeRequest);
        let reply = riir_infer_dist::probe::ProbeReplyWire {
            rows: vec![(1, 0.01)],
            n_layer: n_layer + 8,
            hidden: hidden as u32,
        };
        riir_infer_dist::probe::encode_reply(&reply, &mut buf);
        link.send_frame(riir_infer_dist::transport::FrameKind::ProbeReply, &buf)
            .unwrap();
    });

    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    let own = HostProbe {
        host: "head".into(),
        rows: vec![(1, 0.02)],
        state_budget: None,
    };
    let ctx = ExchangeCtx {
        own,
        n_layer,
        hidden,
        bw_bps: 1e9,
        n_positions: 64,
        state_bytes_per_position_per_layer: 1,
        rtt_rounds: 1,
    };
    let e = run_probe_exchange(&mut link, &[1], &ctx, &addr.to_string())
        .expect_err("model mismatch refuses");
    assert!(e.contains("model mismatch"), "names the lie: {e}");
    tail.join().unwrap();
}
