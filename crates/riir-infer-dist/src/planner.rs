//! The split planner (Plan 623 T2.5) — choose the layer split `k` that
//! equalises the two stages' time.
//!
//! Shape: each host BOOT-PROBES its own per-candidate stage times with
//! [`probe_stage_times`] (scoped loads, timed decode steps — works for any
//! [`StageCompute`]; the CPU stage in tests, the GPU stage in real cells),
//! the two probe records cross by file, and [`plan_split`] — the pure
//! decision core — merges them with the link cost into the chosen split:
//!
//! ```text
//! per-token time at split k = max(head[k], tail[k]) + hop(k)
//! hop(k) = rtt + activation_bytes(k, chunk, hidden) / bw   (decode hop;
//!          the prefill hop amortises over the chunk — same form, n_tok)
//! ```
//!
//! `max` (not the sum) is the PP steady state: the stages run in
//! parallel, the pipeline is as slow as its slower stage plus the one
//! boundary hop. Manual override wins (`--split N` never consults this).
//!
//! Memory law ("equalise stage time UNDER memory"): a candidate whose
//! stage state exceeds its host's budget is refused BEFORE the timing
//! choice — a faster split that cannot hold its KV/GDN state is not a
//! candidate.

use std::io;
use std::time::Instant;

use crate::stage::StageCompute;

/// Measured link cost between the two hosts (T0.2's probe rows: TB5
/// ~0.45 ms RTT / 8.6 GB/s; tailnet ~5.9 ms — the tailnet decode row is
/// exactly its RTT, which is the hop model below).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkCost {
    /// Round-trip seconds (the per-token boundary is one hop: activation
    /// down, token id up — the 4-byte id's serialisation is noise, the
    /// RTT is not).
    pub rtt_s: f64,
    /// Payload bandwidth, bytes/second.
    pub bw_bps: f64,
}

impl LinkCost {
    /// Seconds to move `bytes` across the boundary once (bw + rtt —
    /// a hop cannot beat its RTT even at zero bytes).
    pub fn hop(&self, bytes: usize) -> f64 {
        let wire = match self.bw_bps > 0.0 {
            true => bytes as f64 / self.bw_bps,
            false => 0.0,
        };
        self.rtt_s.max(0.0) + wire
    }
}

/// One host's boot-probe record: measured decode seconds/token per
/// candidate split. `k` indexes the TAIL-side layer (`split` in the bin);
/// candidate `n_layer` (no split) means 1-box and is not a 2-stage row.
#[derive(Debug, Clone, PartialEq)]
pub struct HostProbe {
    /// Which host produced this (free-form: hostname or role).
    pub host: String,
    /// decode s/token per candidate split k.
    pub rows: Vec<(u32, f64)>,
    /// Optional memory budget the probe ran under (bytes of stage state
    /// the host can hold).
    pub state_budget: Option<u64>,
}

impl HostProbe {
    pub fn get(&self, k: u32) -> Option<f64> {
        self.rows.iter().find(|(c, _)| *c == k).map(|(_, t)| *t)
    }
}

/// Stage state per position (bytes) per layer — the planner's memory
/// axis. The Bonsai-2 hybrid at the f32 cache posture: attention
/// K+V rows (`2 × kv_dim × 4`) + the GDN recurrent matrix + conv window
/// (`state_bytes` from the weights). The probe carries the REAL number
/// from the stage; this estimate is the planner-side fallback.
pub fn state_bytes_per_position(kv_dim: usize, gdn_state_bytes: u64) -> u64 {
    2 * kv_dim as u64 * 4 + gdn_state_bytes
}

/// A candidate split refused by the memory axis.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRefusal {
    pub k: u32,
    pub need_bytes: u64,
    pub budget_bytes: u64,
}

/// The planner's decision.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPlan {
    /// Chosen split (tail-side layer index, the `--split` value).
    pub k: u32,
    /// Predicted steady-state per-token time at that split.
    pub s_per_tok: f64,
    /// The per-candidate predictions (every feasible candidate, for the
    /// boot line — never guess from a bare k).
    pub table: Vec<(u32, f64)>,
}

/// Why no split was chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanRefusal {
    /// A candidate needs more stage state than its host holds.
    Memory(MemoryRefusal),
    /// The two probes share NO candidate (a probe must carry at least
    /// one k the other also measured).
    NoCommonCandidate,
    /// A probe row is non-finite or non-positive (a broken probe record
    /// is refused, never trusted).
    BadProbe(String),
}

/// bytes crossing the boundary per decode step at split `k`: the hidden
/// activation down (`hidden × f32`) + the 4-byte token id up.
fn decode_hop_bytes(hidden_dim: usize) -> usize {
    4 * hidden_dim + 4
}

/// Stage state a split holds for a session of `n_positions`: the head
/// owns layers `0..k`, the tail owns `k..n_layer` — each host's need is
/// its own share, and a candidate refuses when EITHER host cannot hold
/// its share.
fn host_state_bytes(
    layers: u64,
    n_positions: usize,
    state_bytes_per_position_per_layer: u64,
) -> u64 {
    layers
        .saturating_mul(state_bytes_per_position_per_layer)
        .saturating_mul(n_positions as u64)
}

/// Choose the split. `head`/`tail` are the two hosts' probe records; the
/// head's `k` rows and the tail's `k` rows are matched per candidate.
/// Memory runs BEFORE timing: head needs `k` layers of state, tail needs
/// `n_layer − k` — a candidate whose either host cannot hold its share
/// for `n_positions` is refused, never timed.
pub fn plan_split(
    head: &HostProbe,
    tail: &HostProbe,
    link: LinkCost,
    hidden_dim: usize,
    n_layer: u32,
    n_positions: usize,
    state_bytes_per_position_per_layer: u64,
) -> Result<SplitPlan, PlanRefusal> {
    for (who, probe) in [(&head.host, head), (&tail.host, tail)] {
        for (k, t) in &probe.rows {
            if !t.is_finite() || *t <= 0.0 {
                return Err(PlanRefusal::BadProbe(format!(
                    "host {who}: split {k} decode {t} s/tok — non-finite or non-positive"
                )));
            }
            if *k == 0 || *k >= n_layer {
                return Err(PlanRefusal::BadProbe(format!(
                    "host {who}: split {k} outside 1..{n_layer}"
                )));
            }
        }
    }
    let hop = link.hop(decode_hop_bytes(hidden_dim));
    let mut table = Vec::new();
    for (k, _) in &head.rows {
        let (Some(h), Some(t)) = (head.get(*k), tail.get(*k)) else {
            continue;
        };
        if let Some(budget) = head.state_budget {
            let need = host_state_bytes(*k as u64, n_positions, state_bytes_per_position_per_layer);
            if need > budget {
                return Err(PlanRefusal::Memory(MemoryRefusal {
                    k: *k,
                    need_bytes: need,
                    budget_bytes: budget,
                }));
            }
        }
        if let Some(budget) = tail.state_budget {
            let need = host_state_bytes(
                n_layer as u64 - *k as u64,
                n_positions,
                state_bytes_per_position_per_layer,
            );
            if need > budget {
                return Err(PlanRefusal::Memory(MemoryRefusal {
                    k: *k,
                    need_bytes: need,
                    budget_bytes: budget,
                }));
            }
        }
        table.push((*k, h.max(t) + hop));
    }
    match table.is_empty() {
        true => Err(PlanRefusal::NoCommonCandidate),
        false => {
            let (k, s) = table
                .iter()
                .copied()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("non-empty table");
            Ok(SplitPlan {
                k,
                s_per_tok: s,
                table,
            })
        }
    }
}

/// Boot probe: time one candidate split's decode on THIS host's stage —
/// `steps` decode steps (warm cache, median of `rounds` batches), the
/// synthetic-token stepping the stage itself validates. The stage is
/// borrowed so the caller controls its lifetime (scoped loads per
/// candidate live in the bin's `--split auto` lane).
pub fn probe_decode_spt(
    stage: &mut dyn StageCompute,
    start_pos: usize,
    steps: usize,
    rounds: usize,
) -> io::Result<f64> {
    if steps == 0 || rounds == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "probe needs steps > 0 and rounds > 0",
        ));
    }
    if start_pos + steps > stage.max_positions() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "probe span {start_pos}+{steps} > stage capacity {}",
                stage.max_positions()
            ),
        ));
    }
    let hidden = stage.hidden_dim();
    let mut input = vec![0.0f32; hidden];
    let empty: [f32; 0] = [];
    let mut token = 1u32;
    // Warm-up (cache + scratch) untimed, one step.
    if stage.handshake().range.is_first() {
        stage.step(token, start_pos, &empty);
    } else {
        stage.step(token, start_pos, &input);
    }
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t0 = Instant::now();
        for i in 0..steps {
            let pos = start_pos + i;
            token = (token % 9973) + 1;
            let r = match stage.handshake().range.is_first() {
                true => stage.step(token, pos, &empty),
                false => stage.step(token, pos, &input),
            };
            // A last stage returns LOGITS (vocab-wide) — carry at most
            // `hidden` back; the probe times the work, it does not
            // interpret the output.
            let n = r.len().min(hidden);
            input[..n].copy_from_slice(&r[..n]);
        }
        let spt = t0.elapsed().as_secs_f64() / steps as f64;
        best = best.min(spt);
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> LinkCost {
        LinkCost {
            rtt_s: 0.00045,
            bw_bps: 8.6e9,
        }
    }

    fn probe(rows: &[(u32, f64)]) -> HostProbe {
        HostProbe {
            host: "h".into(),
            rows: rows.to_vec(),
            state_budget: None,
        }
    }

    #[test]
    fn chooses_the_equalising_split() {
        // amo: fast on few layers (each decode step cheap per layer),
        // bmo: slower per layer — the equaliser should hand bmo FEWER
        // layers, i.e. pick the k where the max() balances.
        let head = probe(&[(16, 0.008), (32, 0.014), (48, 0.021)]);
        let tail = probe(&[(16, 0.018), (32, 0.012), (48, 0.007)]);
        let plan = plan_split(&head, &tail, link(), 5120, 64, 4096, 1).expect("plan");
        assert_eq!(plan.k, 32, "table {:?}", plan.table);
        let want = 0.014_f64.max(0.012) + link().hop(4 * 5120 + 4);
        assert!((plan.s_per_tok - want).abs() < 1e-12);
    }

    #[test]
    fn hop_is_bounded_by_rtt() {
        let l = LinkCost {
            rtt_s: 0.5,
            bw_bps: 1e12,
        };
        // The hop cannot beat its RTT, and the wire term adds on top.
        assert!((l.hop(1 << 20) - (0.5 + (1 << 20) as f64 / 1e12)).abs() < 1e-15);
        let zero_wire = LinkCost {
            rtt_s: 0.5,
            bw_bps: 0.0,
        };
        assert!(
            (zero_wire.hop(1 << 30) - 0.5).abs() < 1e-15,
            "no bw = RTT-only hop"
        );
    }

    #[test]
    fn memory_refuses_the_over_budget_host_before_timing() {
        // Head owns k layers: split 48 over-refuses the HEAD (budget
        // sized so 48 layers × 10 B × 4096 pos > budget but 16 do not).
        let head = HostProbe {
            host: "head".into(),
            rows: vec![(16, 0.01), (48, 0.02)],
            state_budget: Some(16 * 10 * 4096 + 1),
        };
        let tail = probe(&[(16, 0.01), (48, 0.01)]);
        let e = plan_split(&head, &tail, link(), 5120, 64, 4096, 10)
            .expect_err("split 48 over the head's budget");
        assert_eq!(
            e,
            PlanRefusal::Memory(MemoryRefusal {
                k: 48,
                need_bytes: 48 * 10 * 4096,
                budget_bytes: 16 * 10 * 4096 + 1,
            })
        );

        // And the tail's budget bounds ITS share (n_layer − k).
        let head = probe(&[(48, 0.02)]);
        let tail = HostProbe {
            host: "tail".into(),
            rows: vec![(48, 0.01)],
            state_budget: Some((64 - 48) * 10 * 4096 - 1),
        };
        let e = plan_split(&head, &tail, link(), 5120, 64, 4096, 10)
            .expect_err("split 48 leaves the tail 16 layers over ITS budget");
        assert_eq!(
            e,
            PlanRefusal::Memory(MemoryRefusal {
                k: 48,
                need_bytes: 16 * 10 * 4096,
                budget_bytes: 16 * 10 * 4096 - 1,
            })
        );
    }

    #[test]
    fn no_common_candidate_refuses() {
        let head = probe(&[(16, 0.01)]);
        let tail = probe(&[(48, 0.01)]);
        assert_eq!(
            plan_split(&head, &tail, link(), 5120, 64, 4096, 1),
            Err(PlanRefusal::NoCommonCandidate)
        );
    }

    #[test]
    fn broken_probe_row_refuses() {
        let tail = probe(&[(16, 0.01)]);
        let head = probe(&[(16, 0.0)]);
        assert!(matches!(
            plan_split(&head, &tail, link(), 5120, 64, 4096, 1),
            Err(PlanRefusal::BadProbe(_))
        ));
        let head = probe(&[(16, f64::NAN)]);
        assert!(matches!(
            plan_split(&head, &tail, link(), 5120, 64, 4096, 1),
            Err(PlanRefusal::BadProbe(_))
        ));
        // A split outside 1..n_layer is a broken record too (1-box is
        // not a 2-stage candidate).
        let head = probe(&[(64, 0.01)]);
        assert!(matches!(
            plan_split(&head, &tail, link(), 5120, 64, 4096, 1),
            Err(PlanRefusal::BadProbe(_))
        ));
    }
}
