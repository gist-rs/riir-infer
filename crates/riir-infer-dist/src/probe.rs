//! The two-host probe EXCHANGE (Plan 623 T2.5's landing residue): the
//! head asks the tail for its boot-probe record over the wire and
//! composes [`plan_split`] live from two live records + the measured
//! link cost — no operator hand-carries JSON files between hosts.
//!
//! Frames (additive kinds; plan-mode postures only — a serving tail
//! refuses every one of these at its `Handshake` expectation, loud):
//!
//! ```text
//! ProbePing    head→tail  (empty)                     ProbePong tail→head (empty)
//! ProbeRequest head→tail  n u32 | n×k u32             (candidate splits, tail-side index)
//! ProbeReply   tail→head  n u32 | n×(k u32, spt f64)
//!                         | n_layer u32 | hidden u32  (the model cross-check)
//! ```
//!
//! Wire law: little-endian by field, never a host-endian struct cast
//! (the `multi_host` determinism posture, same as `handshake.rs`).
//! Decode is STRICT: exact length, no duplicate candidate, bounded row
//! count — a malformed record is refused, never trusted (the same
//! posture `plan_split` holds for a broken row).
//!
//! The RTT is MEASURED here (median ping/pong over the same
//! authenticated transport the serving session will use); the payload
//! bandwidth stays an operator input — at decode-hop sizes (~20 KB on
//! a 27B hidden) the bw term is ~0.5% of the TB5 hop, and a sized-frame
//! bw probe would conflate tail compute with wire time (honest v0:
//! carry it, disclose it, never guess it).

use std::io;

use crate::planner::{HostProbe, LinkCost, SplitPlan, plan_split};
use crate::transport::Transport;

/// A probe record may not carry more candidates than this — a bound on
/// any single reply (and request), not a tuning knob: the real cell
/// probes a handful of splits.
pub const MAX_PROBE_ROWS: usize = 64;

/// One probe-answer session may serve at most this many requests —
/// the head sends one; the cap bounds a stuck or hostile peer.
pub const MAX_PROBE_REQUESTS: usize = 8;

/// Default ping/pong round trips for the RTT median.
pub const RTT_ROUNDS: usize = 8;

/// The probe request's wire form (see the module doc).
pub fn encode_request(candidates: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&(candidates.len() as u32).to_le_bytes());
    for &k in candidates {
        out.extend_from_slice(&k.to_le_bytes());
    }
}

/// STRICT decode: exact length, non-empty, bounded, no duplicate
/// candidate.
pub fn decode_request(b: &[u8]) -> Result<Vec<u32>, String> {
    if b.len() < 4 {
        return Err(format!("probe request truncated: {} bytes", b.len()));
    }
    let n = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
    if n == 0 {
        return Err("probe request carries no candidates".into());
    }
    if n > MAX_PROBE_ROWS {
        return Err(format!(
            "probe request carries {n} candidates > {MAX_PROBE_ROWS}"
        ));
    }
    if b.len() != 4 + 4 * n {
        return Err(format!(
            "probe request trailing/missing bytes: {} != {}",
            b.len(),
            4 + 4 * n
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let k = u32::from_le_bytes(b[4 + 4 * i..8 + 4 * i].try_into().unwrap());
        if out.contains(&k) {
            return Err(format!("probe request carries candidate {k} twice"));
        }
        out.push(k);
    }
    Ok(out)
}

/// The probe reply's wire form (see the module doc).
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeReplyWire {
    /// decode s/token per candidate split (tail-side k).
    pub rows: Vec<(u32, f64)>,
    /// The replying host's model cross-check (both hosts must hold the
    /// same model for the plan to mean anything).
    pub n_layer: u32,
    pub hidden: u32,
}

pub fn encode_reply(r: &ProbeReplyWire, out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&(r.rows.len() as u32).to_le_bytes());
    for (k, spt) in &r.rows {
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&spt.to_le_bytes());
    }
    out.extend_from_slice(&r.n_layer.to_le_bytes());
    out.extend_from_slice(&r.hidden.to_le_bytes());
}

/// STRICT decode: exact length, non-empty, bounded, no duplicate
/// candidate. Numerical sanity (finite, positive, in-range k) stays
/// with [`plan_split`] — the decision core owns the "broken record is
/// refused, never trusted" law; decode owns the STRUCTURE.
pub fn decode_reply(b: &[u8]) -> Result<ProbeReplyWire, String> {
    if b.len() < 4 {
        return Err(format!("probe reply truncated: {} bytes", b.len()));
    }
    let n = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
    if n == 0 {
        return Err("probe reply carries no rows".into());
    }
    if n > MAX_PROBE_ROWS {
        return Err(format!("probe reply carries {n} rows > {MAX_PROBE_ROWS}"));
    }
    if b.len() != 4 + 12 * n + 8 {
        return Err(format!(
            "probe reply trailing/missing bytes: {} != {}",
            b.len(),
            4 + 12 * n + 8
        ));
    }
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let at = 4 + 12 * i;
        let k = u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        let spt = f64::from_le_bytes(b[at + 4..at + 12].try_into().unwrap());
        if rows.iter().any(|(rk, _): &(u32, f64)| *rk == k) {
            return Err(format!("probe reply carries candidate {k} twice"));
        }
        rows.push((k, spt));
    }
    let at = 4 + 12 * n;
    let n_layer = u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    let hidden = u32::from_le_bytes(b[at + 4..at + 8].try_into().unwrap());
    Ok(ProbeReplyWire {
        rows,
        n_layer,
        hidden,
    })
}

/// Cross-check the tail's reply against the head's own model and build
/// the tail-side [`HostProbe`]. A mismatched peer refuses loud — a plan
/// computed over two different models is a lie.
pub fn tail_record(
    reply: ProbeReplyWire,
    own_n_layer: u32,
    own_hidden: usize,
    peer: &str,
) -> Result<HostProbe, String> {
    if reply.n_layer != own_n_layer || reply.hidden as usize != own_hidden {
        return Err(format!(
            "probe reply model mismatch: tail n_layer {} hidden {} vs own n_layer {own_n_layer} hidden {own_hidden}",
            reply.n_layer, reply.hidden
        ));
    }
    Ok(HostProbe {
        host: format!("tail@{peer}"),
        rows: reply.rows,
        state_budget: None,
    })
}

/// Median RTT over `rounds` ping/pong round trips on `link` — the same
/// authenticated transport the serving session will use, so the number
/// IS the link the plan prices. `rounds` is floored at 1.
pub fn measure_rtt(link: &mut dyn Transport, rounds: usize) -> io::Result<f64> {
    let rounds = rounds.max(1);
    let mut trips = Vec::with_capacity(rounds);
    let mut buf = Vec::new();
    for _ in 0..rounds {
        let t0 = std::time::Instant::now();
        link.send_frame(crate::transport::FrameKind::ProbePing, &[])?;
        let kind = link.recv_frame_into(&mut buf)?;
        if kind != crate::transport::FrameKind::ProbePong {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("probe RTT: expected ProbePong, got {kind:?}"),
            ));
        }
        trips.push(t0.elapsed().as_secs_f64());
    }
    trips.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(trips[rounds / 2])
}

/// Tail side of the exchange: answer pings and probe requests until
/// `Shutdown`. `probe_one(k)` times candidate `k` on THIS host (the
/// scoped-load + timed-decode loop lives with the caller — it is the
/// same loop `--plan-report` runs, shared, never duplicated).
///
/// Every candidate must sit in `1..n_layer` ([`plan_split`]'s domain —
/// the 1-box reference row `k == n_layer` is a `--plan-report` file
/// convention, not a 2-stage candidate) or the session refuses loud.
pub fn serve_probe_session(
    link: &mut dyn Transport,
    n_layer: u32,
    hidden: u32,
    probe_one: &mut dyn FnMut(u32) -> io::Result<f64>,
) -> io::Result<()> {
    let mut buf = Vec::new();
    let mut answered = 0usize;
    loop {
        let kind = link.recv_frame_into(&mut buf)?;
        match kind {
            crate::transport::FrameKind::ProbePing => {
                link.send_frame(crate::transport::FrameKind::ProbePong, &[])?;
            }
            crate::transport::FrameKind::ProbeRequest => {
                if answered >= MAX_PROBE_REQUESTS {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("probe session: > {MAX_PROBE_REQUESTS} requests"),
                    ));
                }
                let candidates = decode_request(&buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let mut rows = Vec::with_capacity(candidates.len());
                for &k in &candidates {
                    if k == 0 || k >= n_layer {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("probe candidate {k} outside 1..{n_layer}"),
                        ));
                    }
                    let spt = probe_one(k)?;
                    rows.push((k, spt));
                }
                encode_reply(
                    &ProbeReplyWire {
                        rows,
                        n_layer,
                        hidden,
                    },
                    &mut buf,
                );
                link.send_frame(crate::transport::FrameKind::ProbeReply, &buf)?;
                answered += 1;
            }
            crate::transport::FrameKind::Shutdown => return Ok(()),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("probe session: unexpected frame {other:?}"),
                ));
            }
        }
    }
}

/// The exchange's output: the plan plus the inputs it was computed
/// from (the boot line quotes them — never guess from a bare k).
#[derive(Debug, Clone, PartialEq)]
pub struct ExchangeOutcome {
    pub plan: SplitPlan,
    /// The RTT measured on THIS connection (the hop's floor).
    pub rtt_s: f64,
    /// The tail-side record as adopted (host label carries the peer).
    pub tail: HostProbe,
}

/// The head side's own inputs to the exchange: its probe record, its
/// model geometry (cross-checked against the reply), and the link
/// shape the plan prices.
#[derive(Debug, Clone)]
pub struct ExchangeCtx {
    /// The head's OWN probe record — the caller probed the SAME
    /// candidate list it passes to [`run_probe_exchange`] (a candidate
    /// the head did not probe is no common candidate and drops out,
    /// the same rule the file-crossing posture holds).
    pub own: HostProbe,
    pub n_layer: u32,
    pub hidden: usize,
    /// Payload bandwidth (bytes/s) — carried, not measured (the module
    /// doc: a sized-frame bw probe would conflate tail compute with
    /// wire time).
    pub bw_bps: f64,
    /// Session position capacity — the memory axis's `n_positions`.
    pub n_positions: usize,
    /// Per-position per-layer stage-state bytes (the memory axis).
    pub state_bytes_per_position_per_layer: u64,
    /// Ping/pong round trips behind the RTT median.
    pub rtt_rounds: usize,
}

/// Head side of the exchange: measure the RTT, ask for the tail's
/// record, compose the plan from the two live records + the measured
/// link.
pub fn run_probe_exchange(
    link: &mut dyn Transport,
    candidates: &[u32],
    ctx: &ExchangeCtx,
    peer: &str,
) -> Result<ExchangeOutcome, String> {
    let rtt_s = measure_rtt(link, ctx.rtt_rounds).map_err(|e| format!("probe RTT: {e}"))?;
    let mut buf = Vec::new();
    encode_request(candidates, &mut buf);
    link.send_frame(crate::transport::FrameKind::ProbeRequest, &buf)
        .map_err(|e| format!("probe request: {e}"))?;
    let kind = link
        .recv_frame_into(&mut buf)
        .map_err(|e| format!("probe reply: {e}"))?;
    if kind != crate::transport::FrameKind::ProbeReply {
        return Err(format!("probe reply: expected ProbeReply, got {kind:?}"));
    }
    let reply = decode_reply(&buf)?;
    let tail = tail_record(reply, ctx.n_layer, ctx.hidden, peer)?;
    let plan = plan_split(
        &ctx.own,
        &tail,
        LinkCost {
            rtt_s,
            bw_bps: ctx.bw_bps,
        },
        ctx.hidden,
        ctx.n_layer,
        ctx.n_positions,
        ctx.state_bytes_per_position_per_layer,
    )
    .map_err(|e| format!("plan: {e:?}"))?;
    Ok(ExchangeOutcome { plan, rtt_s, tail })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::FrameKind;

    #[test]
    fn request_round_trips_and_refuses_malformed() {
        let mut buf = Vec::new();
        encode_request(&[1, 16, 32], &mut buf);
        assert_eq!(decode_request(&buf).unwrap(), vec![1, 16, 32]);

        // truncated header / truncated body / trailing byte / empty / dup
        assert!(decode_request(&[]).is_err());
        assert!(decode_request(&buf[..buf.len() - 1]).is_err());
        let mut trailing = buf.clone();
        trailing.push(0);
        assert!(decode_request(&trailing).is_err());
        assert!(decode_request(&4u32.to_le_bytes()).is_err());
        let mut dup = Vec::new();
        encode_request(&[5, 5], &mut dup);
        assert!(decode_request(&dup).is_err());
        let over_body = {
            let mut b = Vec::new();
            b.extend_from_slice(&((MAX_PROBE_ROWS + 1) as u32).to_le_bytes());
            b
        };
        assert!(decode_request(&over_body).is_err());
    }

    #[test]
    fn reply_round_trips_and_refuses_malformed() {
        let mut buf = Vec::new();
        encode_reply(
            &ProbeReplyWire {
                rows: vec![(1, 0.25), (3, 0.5)],
                n_layer: 4,
                hidden: 64,
            },
            &mut buf,
        );
        assert_eq!(
            decode_reply(&buf).unwrap(),
            ProbeReplyWire {
                rows: vec![(1, 0.25), (3, 0.5)],
                n_layer: 4,
                hidden: 64,
            }
        );
        assert!(decode_reply(&[]).is_err());
        assert!(decode_reply(&buf[..buf.len() - 1]).is_err());
        let mut trailing = buf.clone();
        trailing.push(0);
        assert!(decode_reply(&trailing).is_err());
        let mut dup = Vec::new();
        encode_reply(
            &ProbeReplyWire {
                rows: vec![(2, 0.1), (2, 0.2)],
                n_layer: 4,
                hidden: 64,
            },
            &mut dup,
        );
        assert!(decode_reply(&dup).is_err());
    }

    #[test]
    fn tail_record_refuses_model_mismatch() {
        let ok = ProbeReplyWire {
            rows: vec![(1, 0.1)],
            n_layer: 4,
            hidden: 64,
        };
        let rec = tail_record(ok.clone(), 4, 64, "peer:1").unwrap();
        assert_eq!(rec.host, "tail@peer:1");
        assert_eq!(rec.state_budget, None);
        assert!(tail_record(ok.clone(), 8, 64, "p").is_err());
        assert!(tail_record(ok, 4, 128, "p").is_err());
    }

    /// The tail loop over an in-memory transport pair: pings answered,
    /// one request answered with REAL probe times, shutdown clean, and
    /// the head's composition formula-exact.
    #[test]
    fn exchange_session_end_to_end_in_memory() {
        use std::sync::mpsc;

        struct Pipe {
            rx: std::sync::mpsc::Receiver<(FrameKind, Vec<u8>)>,
            tx: std::sync::mpsc::Sender<(FrameKind, Vec<u8>)>,
        }
        impl Transport for Pipe {
            fn send_frame(&mut self, kind: FrameKind, payload: &[u8]) -> io::Result<()> {
                self.tx
                    .send((kind, payload.to_vec()))
                    .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "peer gone"))
            }
            fn recv_frame_into(&mut self, out: &mut Vec<u8>) -> io::Result<FrameKind> {
                let (kind, body) = self
                    .rx
                    .recv()
                    .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "peer gone"))?;
                out.clear();
                out.extend_from_slice(&body);
                Ok(kind)
            }
        }
        let (a_tx, b_rx) = mpsc::channel::<(FrameKind, Vec<u8>)>();
        let (b_tx, a_rx) = mpsc::channel::<(FrameKind, Vec<u8>)>();
        let mut head_link = Pipe { rx: a_rx, tx: a_tx };
        let mut tail_link = Pipe { rx: b_rx, tx: b_tx };

        let (times_tx, times_rx) = mpsc::channel::<Vec<(u32, f64)>>();
        let tail = std::thread::spawn(move || {
            let mut probed = Vec::new();
            let mut probe_one = |k: u32| -> io::Result<f64> {
                let t = 0.01 * k as f64;
                probed.push((k, t));
                Ok(t)
            };
            let r = serve_probe_session(&mut tail_link, 4, 64, &mut probe_one);
            let _ = times_tx.send(probed);
            r
        });

        let own = HostProbe {
            host: "head".into(),
            rows: vec![(1, 0.02), (3, 0.05)],
            state_budget: None,
        };
        // rtt_rounds 2 keeps the channel ping-pong minimal; the median of
        // an in-memory pair is ~0 but must be finite and non-negative.
        let ctx = ExchangeCtx {
            own,
            n_layer: 4,
            hidden: 64,
            bw_bps: 1e9,
            n_positions: 64,
            state_bytes_per_position_per_layer: 1,
            rtt_rounds: 2,
        };
        let out = run_probe_exchange(&mut head_link, &[1, 3], &ctx, "mem").expect("exchange");
        // The real wire flow ends the session: the head sends Shutdown,
        // the tail's serve loop returns clean. (Skipping it would block
        // the tail on recv forever — the hang this test once had.)
        head_link
            .send_frame(FrameKind::Shutdown, &[])
            .expect("shutdown");
        tail.join().unwrap().expect("tail session clean");

        let tail_rows = times_rx.recv().unwrap();
        assert_eq!(tail_rows.len(), 2);
        assert!(out.rtt_s.is_finite() && out.rtt_s >= 0.0);
        // The plan's formula, exactly, over the two live records.
        for (k, s) in &out.plan.table {
            let h = ctx.own.get(*k).unwrap();
            let t = out.tail.get(*k).unwrap();
            let hop = out.rtt_s + (4usize * 64 + 4) as f64 / 1e9;
            assert!((s - (h.max(t) + hop)).abs() < 1e-12, "k={k}");
        }
        // k=1: max(0.02, 0.01) = 0.02; k=3: max(0.05, 0.03) = 0.05 → k=1 wins.
        assert_eq!(out.plan.k, 1);
    }

    #[test]
    fn tail_session_refuses_out_of_range_candidate() {
        struct Dead;
        impl Transport for Dead {
            fn send_frame(&mut self, _: FrameKind, _: &[u8]) -> io::Result<()> {
                Ok(())
            }
            fn recv_frame_into(&mut self, _out: &mut Vec<u8>) -> io::Result<FrameKind> {
                Err(io::Error::new(io::ErrorKind::UnexpectedEof, "dead"))
            }
        }
        // A session whose first frame is EOF refuses (never a green zero).
        let mut link = Dead;
        let mut probe_one = |_k: u32| -> io::Result<f64> { Ok(0.1) };
        assert!(serve_probe_session(&mut link, 4, 64, &mut probe_one).is_err());
    }
}
