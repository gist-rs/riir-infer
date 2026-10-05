//! FA-constrained exact posterior sampling for the dLLM decode lane
//! (Issue 035 P0) — the pure sampler module, no model.
//!
//! Distilled from Mosaic (arXiv:2607.07026 / `MhDang/mosaic` @ `36945e4b`,
//! MIT): a finite automaton viewed as an HMM whose edges emit their allowed
//! tokens. Given per-position LM logits over a block of `len` positions, the
//! exact posterior `p(x | constraint) ∝ Π_t p_lm(x_t) · 1[A accepts x]` is
//! sampled by drawing a joint state path (forward pass over edge-space flow
//! weights conditioned by a backward pass) and then tokens given the edges.
//! Per-position marginals do NOT factorize this joint — independent
//! per-position draws almost never produce an accepted sequence; the joint
//! draw accepts by construction.
//!
//! Layout follows mosaic's design in CSR form (fusion note, Research 008):
//! edges grouped by source node, per-edge token sets as sorted CSR lists —
//! never the dense `n_states × vocab` δ table.
//!
//! Determinism invariants (the two constraints that make the edge-flow
//! decomposition exact; enforced at [`AutomatonBuilder::build`]):
//! ≤1 edge per (node, token) and ≤1 edge per (node, dst) pair.
//!
//! Steady state is allocation-free: [`Automaton::sample_joint`] writes into
//! caller-provided buffers and bounded scratch ([`FaScratch`]); the G4
//! counting-allocator gate lives in `tests/fa_g4_alloc.rs` (its own test
//! target — the counting allocator is process-global). P0.5 adds the
//! O(log L) segment-tree parallel sampler + fp64 escape hatch; P1 wires
//! `propose_x0` into the `gemma2_d2f` denoising loop behind `fa_constraint`.

use thiserror::Error;

/// The `forced` sentinel: a position carrying this value is FREE (drawn from
/// the posterior); any other value pins the position to that token.
pub const FREE: u32 = u32::MAX;

/// `steps_to_accept` sentinel: the node cannot reach an accepting node.
pub const UNREACHABLE: u32 = u32::MAX;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum FaError {
    #[error("node {node} out of range (n_nodes {n_nodes})")]
    NodeOutOfRange { node: usize, n_nodes: usize },
    #[error("token {token} out of range (vocab {vocab}) on edge {src}->{dst}")]
    TokenOutOfRange {
        token: usize,
        vocab: usize,
        src: usize,
        dst: usize,
    },
    #[error("edge {src}->{dst} allows token {token} twice")]
    DuplicateTokenInEdge { src: usize, dst: usize, token: usize },
    #[error("determinism violation: node {node} has two outgoing edges allowing token {token}")]
    DuplicateEdgeToken { node: usize, token: usize },
    #[error("determinism violation: node {node} has two edges to node {dst}")]
    DuplicatePair { node: usize, dst: usize },
    #[error("edge {src}->{dst} has an empty token set")]
    EmptyEdge { src: usize, dst: usize },
    #[error("start node {0} out of range")]
    BadStart(usize),
    #[error("logits shape mismatch: expected {expected} (len {len} × vocab {vocab}), got {got}")]
    LogitsShape {
        expected: usize,
        got: usize,
        len: usize,
        vocab: usize,
    },
    #[error("forced slice length {got} != block length {len}")]
    ForcedShape { got: usize, len: usize },
    #[error("out_tokens length {got} != block length {len}")]
    OutShape { got: usize, len: usize },
    #[error("no outgoing edge from start node {0} — the block cannot begin")]
    DeadStart(usize),
    #[error("constraint unsatisfiable: no accepting path of length {len} from the start node")]
    Unsatisfiable { len: usize },
    #[error("internal: sampled edge to a dead state at position {pos} (backward-pass accounting bug)")]
    InternalDeadState { pos: usize },
}

// ── log-space helpers ────────────────────────────────────────────────

/// `log Σ exp(vᵢ)` over a slice that may contain `-INF` entries (max-shift;
/// an all-`-INF` slice returns `-INF`).
#[inline]
fn logsumexp_shifted(values: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v > max {
            max = v;
        }
    }
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let mut acc = 0.0f32;
    for &v in values {
        acc += (v - max).exp();
    }
    max + acc.ln()
}

// ── deterministic RNG (no external dep; seeded ⇒ reproducible tests) ─

/// SplitMix64 — small, fast, fully deterministic. The sampler takes
/// `&mut Self` so a caller threads one seeded stream through a decode loop
/// (same seed + same logits ⇒ same draws).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) — 53-bit resolution.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ── automaton ────────────────────────────────────────────────────────

/// Builder phase — edges accumulate unsorted; [`AutomatonBuilder::build`]
/// sorts them into CSR form and enforces the determinism invariants.
pub struct AutomatonBuilder {
    n_nodes: usize,
    vocab: usize,
    start: usize,
    accept: Vec<bool>,
    edges: Vec<(usize, usize, Vec<u32>)>,
}

impl AutomatonBuilder {
    pub fn new(n_nodes: usize, vocab: usize, start: usize) -> Self {
        Self {
            n_nodes,
            vocab,
            start,
            accept: vec![false; n_nodes],
            edges: Vec::new(),
        }
    }

    /// Mark one node accepting (repeat calls idempotent).
    pub fn accept(&mut self, node: usize) -> &mut Self {
        self.accept[node] = true;
        self
    }

    /// Add an edge `src → dst` allowing exactly `tokens`.
    pub fn edge(&mut self, src: usize, dst: usize, tokens: &[u32]) -> &mut Self {
        self.edges.push((src, dst, tokens.to_vec()));
        self
    }

    /// Finalize: stable-sort edges by source (insertion order preserved
    /// within a node), sort each edge's token set, enforce the invariants.
    /// Takes `&self` so a chain of `.edge()` calls can end in `.build()`;
    /// the builder stays reusable (setup-time only — the clone is not a
    /// hot-path concern).
    pub fn build(&self) -> Result<Automaton, FaError> {
        let n_nodes = self.n_nodes;
        if self.start >= n_nodes {
            return Err(FaError::BadStart(self.start));
        }

        let mut edges = self.edges.clone();
        edges.sort_by_key(|(src, _, _)| *src);

        let n_edges = self.edges.len();
        let mut out_counts = vec![0u32; n_nodes];
        let mut edge_dst = vec![0u32; n_edges];
        let mut emit_start = Vec::with_capacity(n_edges + 1);
        let mut emit_tokens = Vec::new();
        emit_start.push(0u32);

        // Invariant-1 scratch: token → edge index within the CURRENT source
        // node; cleared when the source changes.
        let mut token_owner: Vec<i32> = vec![-1; self.vocab];
        let mut node_first_edge = 0usize;
        let mut prev_src: Option<usize> = None;

        for (i, (src, dst, tokens)) in edges.iter().enumerate() {
            if *src >= n_nodes || *dst >= n_nodes {
                return Err(FaError::NodeOutOfRange {
                    node: if *src >= n_nodes { *src } else { *dst },
                    n_nodes,
                });
            }
            if prev_src != Some(*src) {
                if prev_src.is_some() {
                    for (_, _, owner_toks) in &edges[node_first_edge..i] {
                        for &t in owner_toks {
                            token_owner[t as usize] = -1;
                        }
                    }
                }
                prev_src = Some(*src);
                node_first_edge = i;
            }
            if tokens.is_empty() {
                return Err(FaError::EmptyEdge {
                    src: *src,
                    dst: *dst,
                });
            }

            let mut toks = tokens.clone();
            toks.sort_unstable();
            let mut prev: Option<u32> = None;
            for &t in &toks {
                if (t as usize) >= self.vocab {
                    return Err(FaError::TokenOutOfRange {
                        token: t as usize,
                        vocab: self.vocab,
                        src: *src,
                        dst: *dst,
                    });
                }

                if Some(t) == prev {
                    return Err(FaError::DuplicateTokenInEdge {
                        src: *src,
                        dst: *dst,
                        token: t as usize,
                    });
                }
                prev = Some(t);
                if token_owner[t as usize] >= 0 {
                    return Err(FaError::DuplicateEdgeToken {
                        node: *src,
                        token: t as usize,
                    });
                }
                token_owner[t as usize] = i as i32;
            }

            edge_dst[i] = *dst as u32;
            emit_tokens.extend_from_slice(&toks);
            emit_start.push(emit_tokens.len() as u32);
            out_counts[*src] += 1;
        }

        // Counts → CSR offsets.
        let mut out_start = Vec::with_capacity(n_nodes + 1);
        out_start.push(0u32);
        for c in &out_counts {
            out_start.push(out_start.last().unwrap() + c);
        }

        // Invariant 2: ≤1 edge per (node, dst) pair.
        for s in 0..n_nodes {
            let (a, b) = (out_start[s] as usize, out_start[s + 1] as usize);
            for i in a..b {
                for j in (i + 1)..b {
                    if edge_dst[i] == edge_dst[j] {
                        return Err(FaError::DuplicatePair {
                            node: s,
                            dst: edge_dst[i] as usize,
                        });
                    }
                }
            }
        }

        Ok(Automaton {
            n_nodes,
            vocab: self.vocab,
            start: self.start,
            accept: self.accept.clone(),
            out_start,
            edge_dst,
            emit_start,
            emit_tokens,
        })
    }
}

/// A compiled finite automaton: CSR edges by source node, per-edge sorted
/// token sets, accepting-node mask, designated start node.
#[derive(Clone, Debug)]
pub struct Automaton {
    n_nodes: usize,
    vocab: usize,
    start: usize,
    accept: Vec<bool>,
    /// (N+1,) offsets into `edge_dst`/`emit_start` per source node.
    out_start: Vec<u32>,
    /// (E,) destination per edge.
    edge_dst: Vec<u32>,
    /// (E+1,) offsets into `emit_tokens` per edge.
    emit_start: Vec<u32>,
    /// Sorted token sets, concatenated per edge.
    emit_tokens: Vec<u32>,
}

impl Automaton {
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    pub fn n_edges(&self) -> usize {
        self.edge_dst.len()
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn is_accept(&self, node: usize) -> bool {
        self.accept[node]
    }

    /// Outgoing edge index range `[a, b)` of `node`.
    #[inline]
    pub fn out_range(&self, node: usize) -> std::ops::Range<usize> {
        self.out_start[node] as usize..self.out_start[node + 1] as usize
    }

    pub fn edge_dst(&self, edge: usize) -> usize {
        self.edge_dst[edge] as usize
    }

    /// The edge leaving `node` that allows `token`, if any (the determinism
    /// invariant makes it unique). Linear over the node's fanout with a
    /// binary search inside each edge's sorted token set.
    pub fn edge_for_token(&self, node: usize, token: u32) -> Option<usize> {
        for e in self.out_range(node) {
            let (a, b) = (self.emit_start[e] as usize, self.emit_start[e + 1] as usize);
            if self.emit_tokens[a..b].binary_search(&token).is_ok() {
                return Some(e);
            }
        }
        None
    }

    /// Walk a token sequence from the start node, returning the final node —
    /// the determinism invariant makes the walk unique. `None` if some
    /// position has no allowed edge (the walk dies).
    pub fn walk(&self, tokens: &[u32]) -> Option<usize> {
        let mut node = self.start;
        for &t in tokens {
            let e = self.edge_for_token(node, t)?;
            node = self.edge_dst[e] as usize;
        }
        Some(node)
    }

    /// Source node of edge `e` (binary search over the CSR offsets).
    fn source_of(&self, edge: usize) -> usize {
        let s = self
            .out_start
            .partition_point(|&off| (off as usize) <= edge);
        s - 1
    }

    /// Reverse-BFS: minimum steps from every node to ANY accepting node
    /// (`UNREACHABLE` where none exists). The length-budget probe behind
    /// `finish_within`: a node with `steps > remaining` cannot complete the
    /// block in the positions that remain.
    pub fn steps_to_accept(&self) -> Vec<u32> {
        let n = self.n_nodes;
        let mut rev_in: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in 0..self.n_edges() {
            rev_in[self.edge_dst[e] as usize].push(self.source_of(e));
        }
        let mut dist = vec![UNREACHABLE; n];
        let mut queue = std::collections::VecDeque::new();
        for (s, &a) in self.accept.iter().enumerate() {
            if a {
                dist[s] = 0;
                queue.push_back(s);
            }
        }
        while let Some(d) = queue.pop_front() {
            for &src in &rev_in[d] {
                if dist[src] == UNREACHABLE {
                    dist[src] = dist[d] + 1;
                    queue.push_back(src);
                }
            }
        }
        dist
    }
}

// ── scratch + the sampler ────────────────────────────────────────────

/// Bounded scratch for the exact joint sampler. Allocate once per decode
/// loop, [`FaScratch::ensure`] grows lazily, and the hot path never
/// allocates (the G4 gate pins it).
#[derive(Default)]
pub struct FaScratch {
    /// Per-position per-edge emission fold `log Σ_{v∈e} exp(logits/T)`,
    /// with committed positions folded to a 0/1 indicator.
    e_log: Vec<f32>,
    /// Node-level backward table `(len+1) × N`:
    /// `back[t][s] = log P(accept at len | at s before position t)`.
    back: Vec<f32>,
    /// Categorical weight buffer — sized for max(node fanout, edge token
    /// count) since the edge draw and the token draw share it.
    weights: Vec<f32>,
}

impl FaScratch {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure(&mut self, len: usize, n_edges: usize, n_nodes: usize, fanout: usize) {
        if self.e_log.len() < len * n_edges {
            self.e_log.resize(len * n_edges, 0.0);
        }
        let back_need = (len + 1) * n_nodes;
        if self.back.len() < back_need {
            self.back.resize(back_need, 0.0);
        }
        if self.weights.len() < fanout {
            self.weights.resize(fanout, 0.0);
        }
    }
}

impl Automaton {
    /// The largest single edge token set (the token draw's buffer need).
    fn max_edge_tokens(&self) -> usize {
        (0..self.n_edges())
            .map(|e| self.emit_start[e + 1] - self.emit_start[e])
            .max()
            .unwrap_or(0) as usize
    }

    fn max_fanout(&self) -> usize {
        self.out_start
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
    }

    /// The exact joint draw over one block.
    ///
    /// * `logits` — `len × vocab` row-major per-position LM logits.
    /// * `forced` — per-position pin (`FREE` = drawn); a pinned position
    ///   contributes a 0/1 indicator through the fold (branchless — the only
    ///   visible effect is which edges carry finite flow).
    /// * `temperature` — applied to the LM logits (p ∝ exp(lm/T)); a
    ///   non-finite or non-positive value falls back to 1.0. The argmax
    ///   token draw ignores T (argmax is T-invariant for T > 0) and reads
    ///   the raw logits.
    /// * `argmax_tokens` — greedy token given the sampled edge (the mosaic
    ///   "greedy" posture: the PATH is still drawn from the exact joint);
    ///   `false` = multinomial token draw.
    ///
    /// Returns the final node (accepting by construction — every produced
    /// sequence is accepted; that is the sampler's contract, not luck).
    ///
    /// Allocation-free in steady state: all buffers come from `scratch`,
    /// tokens land in `out_tokens`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_joint(
        &self,
        scratch: &mut FaScratch,
        logits: &[f32],
        len: usize,
        forced: &[u32],
        temperature: f32,
        argmax_tokens: bool,
        rng: &mut SplitMix64,
        out_tokens: &mut [u32],
    ) -> Result<usize, FaError> {
        if forced.len() != len {
            return Err(FaError::ForcedShape {
                got: forced.len(),
                len,
            });
        }
        if out_tokens.len() != len {
            return Err(FaError::OutShape {
                got: out_tokens.len(),
                len,
            });
        }
        let expected = len * self.vocab;
        if logits.len() != expected {
            return Err(FaError::LogitsShape {
                expected,
                got: logits.len(),
                len,
                vocab: self.vocab,
            });
        }
        let n_edges = self.n_edges();
        if len == 0 {
            return if self.accept[self.start] {
                Ok(self.start)
            } else {
                Err(FaError::Unsatisfiable { len: 0 })
            };
        }
        if self.out_start[self.start] == self.out_start[self.start + 1] {
            return Err(FaError::DeadStart(self.start));
        }
        scratch.ensure(
            len,
            n_edges,
            self.n_nodes,
            self.max_fanout().max(self.max_edge_tokens()),
        );
        let temp = if temperature.is_finite() && temperature > 0.0 {
            temperature
        } else {
            1.0
        };

        // ── 1. the emission fold ─────────────────────────────────────
        // Free position:  e_log[t][e] = log Σ_{v∈e} exp(logits[t,v]/T)
        // Pinned position: e_log[t][e] = 0 if forced[t] ∈ e else -INF
        for t in 0..len {
            let row = &logits[t * self.vocab..(t + 1) * self.vocab];
            let pin = forced[t];
            let base = t * n_edges;
            for e in 0..n_edges {
                let (a, b) = (self.emit_start[e] as usize, self.emit_start[e + 1] as usize);
                let toks = &self.emit_tokens[a..b];
                scratch.e_log[base + e] = if pin != FREE {
                    if toks.binary_search(&pin).is_ok() {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                } else {
                    let mut max = f32::NEG_INFINITY;
                    for &v in toks {
                        let l = row[v as usize] / temp;
                        if l > max {
                            max = l;
                        }
                    }
                    let mut acc = 0.0f32;
                    for &v in toks {
                        acc += (row[v as usize] / temp - max).exp();
                    }
                    max + acc.ln()
                };
            }
        }

        // ── 2. the backward pass ─────────────────────────────────────
        // back[len][s] = accept(s) ? 0 : -INF;
        // back[t][s] = LSE over out(s) of (e_log[t][e] + back[t+1][dst(e)]).
        let n = self.n_nodes;
        for (s, &a) in self.accept.iter().enumerate() {
            scratch.back[len * n + s] = if a { 0.0 } else { f32::NEG_INFINITY };
        }
        for t in (0..len).rev() {
            let base = t * n_edges;
            let next = (t + 1) * n;
            for s in 0..n {
                let range = self.out_range(s);
                if range.is_empty() {
                    scratch.back[t * n + s] = f32::NEG_INFINITY;
                    continue;
                }
                let mut k = 0usize;
                for e in range.clone() {
                    scratch.weights[k] =
                        scratch.e_log[base + e] + scratch.back[next + self.edge_dst[e] as usize];
                    k += 1;
                }
                scratch.back[t * n + s] = logsumexp_shifted(&scratch.weights[..k]);
            }
        }
        if scratch.back[self.start] == f32::NEG_INFINITY {
            return Err(FaError::Unsatisfiable { len });
        }

        // ── 3. the forward joint draw ────────────────────────────────
        let mut state = self.start;
        for t in 0..len {
            let range = self.out_range(state);
            let base = t * n_edges;
            let next = (t + 1) * n;
            let mut k = 0usize;
            let mut max = f32::NEG_INFINITY;
            for e in range.clone() {
                let w = scratch.e_log[base + e] + scratch.back[next + self.edge_dst[e] as usize];
                scratch.weights[k] = w;
                if w > max {
                    max = w;
                }
                k += 1;
            }
            // `max` is finite: position t's node has a finite continuation by
            // the backward pass's own accounting (guarded at t=0 via the
            // Unsatisfiable check; each step lands on such a node).
            let mut total = 0.0f32;
            for w in &mut scratch.weights[..k] {
                *w = (*w - max).exp();
                total += *w;
            }
            // NaN or zero total ⇒ the position has no live edge — the
            // backward pass's accounting should have made this unreachable;
            // refuse rather than emit a rejected sequence.
            if total.is_nan() || total <= 0.0 {
                return Err(FaError::InternalDeadState { pos: t });
            }
            let r = rng.next_f64() as f32 * total;
            let mut acc = 0.0f32;
            let mut chosen = range.end - 1;
            for (i, e) in range.clone().enumerate() {
                acc += scratch.weights[i];
                if r < acc {
                    chosen = e;
                    break;
                }
            }
            state = self.edge_dst[chosen] as usize;

            // Token given the edge. Pinned positions take the pin (the
            // chosen edge allows it — its flow weight was finite).
            let pin = forced[t];
            if pin != FREE {
                out_tokens[t] = pin;
                continue;
            }
            let row = &logits[t * self.vocab..(t + 1) * self.vocab];
            let (a, b) = (self.emit_start[chosen] as usize, self.emit_start[chosen + 1] as usize);
            let toks = &self.emit_tokens[a..b];
            out_tokens[t] = if argmax_tokens {
                let mut best = toks[0];
                let mut best_l = row[toks[0] as usize];
                for &v in &toks[1..] {
                    let l = row[v as usize];
                    if l > best_l {
                        best_l = l;
                        best = v;
                    }
                }
                best
            } else {
                let mut m = f32::NEG_INFINITY;
                for &v in toks {
                    let l = row[v as usize] / temp;
                    if l > m {
                        m = l;
                    }
                }
                let mut acc2 = 0.0f32;
                for (i, &v) in toks.iter().enumerate() {
                    scratch.weights[i] = (row[v as usize] / temp - m).exp();
                    acc2 += scratch.weights[i];
                }
                let r2 = rng.next_f64() as f32 * acc2;
                let mut acc3 = 0.0f32;
                let mut chosen_v = toks[toks.len() - 1];
                for (i, &v) in toks.iter().enumerate() {
                    acc3 += scratch.weights[i];
                    if r2 < acc3 {
                        chosen_v = v;
                        break;
                    }
                }
                chosen_v
            };
        }
        debug_assert!(self.accept[state], "joint draw must land accepting");
        Ok(state)
    }
}

// ── P0.5 — the parallel sampler (the paper's headline) ──────────────

/// The fp64 escape hatch is STRUCTURAL here: every tree quantity is f64
/// log-space, so the upstream fp32-underflow-at-L=64 trap (mosaic
/// `tests/test_marginal.py:86-103`) cannot occur by construction — there is
/// no f32 tree-product posture to escape from. The f32 sequential sampler
/// above stays the decode-loop hot path; this lane is the O(log L)-depth
/// sampler, rebuilt per denoising step (the logits change each step).
///
/// Scratch for the segment-tree build. Reuse across denoising steps: the
/// logits change per step, the shape usually does not.
#[derive(Default)]
pub struct TreeScratch {
    /// `levels[0]` = the `len_pad` leaves (each an N×N f64 log-space
    /// matrix, row-major), `levels[k+1]` = pairwise products (half the
    /// count) — laid out contiguously per level, leaves first, root last.
    /// Total O(2·len_pad·N²) f64.
    pub levels: Vec<Vec<f64>>,
    /// (N²,) edge lookup: `pair_edge[i*n+j]` = the unique edge i→j or −1
    /// (the determinism invariant makes it unique). Rebuilt only when the
    /// automaton changes.
    pair_edge: Vec<i32>,
    pair_edge_nodes: usize,
}

impl TreeScratch {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_pair_edge(&mut self, n: usize) {
        if self.pair_edge_nodes != n {
            self.pair_edge = vec![-1; n * n];
            self.pair_edge_nodes = n;
        } else {
            self.pair_edge.iter_mut().for_each(|v| *v = -1);
        }
    }
}

/// A built segment tree over one block, borrowing the scratch's levels.
/// Root sample + O(log L) top-down midpoint walk; tokens drawn per crossed
/// edge exactly as the sequential sampler does (the tree holds the logits
/// reference for that half).
pub struct ParallelTree<'a> {
    fa: &'a Automaton,
    scratch: &'a TreeScratch,
    logits: &'a [f32],
    forced: &'a [u32],
    len: usize,
    len_pad: usize,
    n: usize,
    temperature: f32,
    /// First level-array index per level (leaves at 0). Root is last.
    level_offsets: Vec<usize>,
}

/// log-space N×N matrix product, per-entry max-shift:
/// `out[i,j] = log Σ_k exp(a[i,k] + b[k,j])`.
fn matmul_log(a: &[f64], b: &[f64], n: usize, out: &mut [f64]) {
    for i in 0..n {
        let a_row = &a[i * n..(i + 1) * n];
        for j in 0..n {
            let mut m = f64::NEG_INFINITY;
            for k in 0..n {
                let v = a_row[k] + b[k * n + j];
                if v > m {
                    m = v;
                }
            }
            if m == f64::NEG_INFINITY {
                out[i * n + j] = f64::NEG_INFINITY;
                continue;
            }
            let mut s = 0.0f64;
            for k in 0..n {
                s += (a_row[k] + b[k * n + j] - m).exp();
            }
            out[i * n + j] = m + s.ln();
        }
    }
}

/// Categorical draw over non-normalized positive weights; falls through to
/// the last index on rounding (same convention as the sequential sampler).
fn draw_categorical(w: &[f64], rng: &mut SplitMix64) -> usize {
    let total: f64 = w.iter().sum();
    let r = rng.next_f64() * total;
    let mut acc = 0.0f64;
    for (i, &wi) in w.iter().enumerate() {
        acc += wi;
        if r < acc {
            return i;
        }
    }
    w.len() - 1
}

impl Automaton {
    /// Build the segment tree for one block. `logits`/`forced`/
    /// `temperature` follow [`Automaton::sample_joint`]'s contract.
    /// Unsatisfiable blocks error before any sampling.
    pub fn build_tree<'a>(
        &'a self,
        scratch: &'a mut TreeScratch,
        logits: &'a [f32],
        len: usize,
        forced: &'a [u32],
        temperature: f32,
    ) -> Result<ParallelTree<'a>, FaError> {
        if forced.len() != len {
            return Err(FaError::ForcedShape {
                got: forced.len(),
                len,
            });
        }
        let expected = len * self.vocab;
        if logits.len() != expected {
            return Err(FaError::LogitsShape {
                expected,
                got: logits.len(),
                len,
                vocab: self.vocab,
            });
        }
        if len == 0 {
            return Err(FaError::Unsatisfiable { len: 0 });
        }
        let n = self.n_nodes;
        let temp = if temperature.is_finite() && temperature > 0.0 {
            temperature
        } else {
            1.0
        };

        // The (i,j) → edge map (the determinism invariant makes it unique).
        scratch.ensure_pair_edge(n);
        for s in 0..n {
            for e in self.out_range(s) {
                scratch.pair_edge[s * n + self.edge_dst(e)] = e as i32;
            }
        }

        // Leaf matrices. Position t: leaf[i][j] = e_log[t][edge(i,j)] —
        // the fold with temperature and pins; the LAST REAL leaf folds the
        // acceptance weights on its diagonal (only accepting dsts survive).
        // Each leaf is shifted by its own max before storing.
        let len_pad = len.next_power_of_two();
        let mut levels: Vec<Vec<f64>> = Vec::with_capacity(2 * len_pad);
        let mut leaf = vec![0.0f64; n * n];
        for t in 0..len {
            let row = &logits[t * self.vocab..(t + 1) * self.vocab];
            let pin = forced[t];
            let last_real = t + 1 == len;
            let mut m = f64::NEG_INFINITY;
            for i in 0..n {
                for j in 0..n {
                    let e = scratch.pair_edge[i * n + j];
                    let v = match usize::try_from(e) {
                        Err(_) => f64::NEG_INFINITY,
                        Ok(e) => {
                            let (a, b) = (
                                self.emit_start[e] as usize,
                                self.emit_start[e + 1] as usize,
                            );
                            let toks = &self.emit_tokens[a..b];
                            let base = if pin != FREE {
                                // 0/1 indicator: the pin is IN or the edge is dead.
                                if toks.binary_search(&pin).is_ok() {
                                    0.0
                                } else {
                                    f64::NEG_INFINITY
                                }
                            } else {
                                let mut mx = f64::NEG_INFINITY;
                                for &v in toks {
                                    let l = (row[v as usize] as f64) / (temp as f64);
                                    if l > mx {
                                        mx = l;
                                    }
                                }
                                if mx == f64::NEG_INFINITY {
                                    f64::NEG_INFINITY
                                } else {
                                    let mut acc = 0.0f64;
                                    for &v in toks {
                                        acc +=
                                            ((row[v as usize] as f64) / (temp as f64) - mx).exp();
                                    }
                                    mx + acc.ln()
                                }
                            };
                            if last_real && !(base.is_finite() && self.accept[j]) {
                                f64::NEG_INFINITY
                            } else {
                                base
                            }
                        }
                    };
                    leaf[i * n + j] = v;
                    if v > m {
                        m = v;
                    }
                }
            }
            if m == f64::NEG_INFINITY {
                return Err(FaError::Unsatisfiable { len });
            }
            for v in leaf.iter_mut() {
                *v -= m;
            }
            levels.push(leaf.clone());
        }
        // Identity padding to the power of two (log I: 0 diagonal, -INF off).
        for _ in len..len_pad {
            let mut id = vec![f64::NEG_INFINITY; n * n];
            for (i, v) in id.iter_mut().enumerate() {
                *v = if i % (n + 1) == 0 { 0.0 } else { f64::NEG_INFINITY };
            }
            levels.push(id);
        }

        // Bottom-up pairwise products (leaves at [0, len_pad), then halves;
        // every level is per-matrix so the offsets indexing stays uniform).
        let mut width = len_pad;
        while width > 1 {
            let prev_start = levels.len() - width;
            let mut prod = vec![0.0f64; n * n];
            for pair in 0..width / 2 {
                let a = &levels[prev_start + pair * 2];
                let b = &levels[prev_start + pair * 2 + 1];
                matmul_log(a, b, n, &mut prod);
                levels.push(prod.clone());
            }
            width /= 2;
        }

        // Root row over the start node must carry mass.
        let root = levels.last().unwrap();
        if root[self.start * n..(self.start + 1) * n]
            .iter()
            .all(|&v| v == f64::NEG_INFINITY)
        {
            return Err(FaError::Unsatisfiable { len });
        }

        let mut level_offsets = Vec::new();
        {
            let mut w = len_pad;
            let mut off = 0usize;
            while w >= 1 {
                level_offsets.push(off);
                off += w;
                if w == 1 {
                    break;
                }
                w /= 2;
            }
        }

        scratch.levels = levels;
        Ok(ParallelTree {
            fa: self,
            scratch,
            logits,
            forced,
            len,
            len_pad,
            n,
            temperature: temp,
            level_offsets,
        })
    }
}

impl ParallelTree<'_> {
    /// Level array index of `level`'s first matrix.
    #[inline]
    fn level_off(&self, level: usize) -> usize {
        self.level_offsets[level]
    }

    /// The O(log L)-depth joint draw: sample the root's final state, then
    /// walk the tree top-down, sampling each segment's midpoint conditional
    /// `p(s_mid = m) ∝ Ql[s_begin][m] + Qr[m][s_end]`. A NaN/degenerate
    /// conditional falls back to uniform over finite entries and bumps the
    /// caller's `nan_count` (the deferred-warning law — no mid-step panic).
    /// Returns the final node (accepting by construction, as the sequential
    /// sampler's contract).
    pub fn sample(
        &self,
        argmax_tokens: bool,
        rng: &mut SplitMix64,
        nan_count: &mut u64,
        out_tokens: &mut [u32],
    ) -> Result<usize, FaError> {
        if out_tokens.len() != self.len {
            return Err(FaError::OutShape {
                got: out_tokens.len(),
                len: self.len,
            });
        }
        *nan_count = 0;
        let n = self.n;
        let levels = &self.scratch.levels;
        let root = levels.last().unwrap();

        // 1. The final state from the root row (already accept-folded).
        let final_state = {
            let row = &root[self.fa.start * n..(self.fa.start + 1) * n];
            let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                return Err(FaError::Unsatisfiable { len: self.len });
            }
            let w: Vec<f64> = row.iter().map(|&v| (v - m).exp()).collect();
            draw_categorical(&w, rng)
        };

        // 2. Top-down midpoint walk. Frames: (level, idx, s_begin, s_end);
        //    segment positions are [idx·width, (idx+1)·width) at `level`.
        let mut states = vec![0usize; self.len_pad + 1];
        states[0] = self.fa.start;
        states[self.len_pad] = final_state;
        let mut stack: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(2 * self.len_pad);
        stack.push((0usize, 0usize, self.fa.start, final_state));
        while let Some((level, idx, sb, se)) = stack.pop() {
            let width = self.len_pad >> level;
            if width == 1 {
                continue; // leaf: boundaries already fixed
            }
            let mid_pos = idx * width + width / 2;
            // Children of walk-level L live in tree-array level (total-2-L):
            // the root is the LAST array, and each step down is one level
            // closer to the leaves.
            let child_off = self.level_offsets[self.level_offsets.len() - 2 - level];
            let ql = &levels[child_off + idx * 2];
            let qr = &levels[child_off + idx * 2 + 1];
            let mut cond = vec![0.0f64; n];
            let mut cm = f64::NEG_INFINITY;
            let mut any_nan = false;
            for (m, c) in cond.iter_mut().enumerate() {
                let v = ql[sb * n + m] + qr[m * n + se];
                if v.is_nan() {
                    any_nan = true;
                }
                *c = v;
                if v > cm {
                    cm = v;
                }
            }
            let s_mid = if any_nan {
                // Deferred-warning fallback: uniform over finite entries.
                *nan_count += 1;
                let finite: Vec<usize> = cond
                    .iter()
                    .enumerate()
                    .filter(|&(_, v)| v.is_finite())
                    .map(|(i, _)| i)
                    .collect();
                match finite.len() {
                    0 => 0,
                    k => finite[(rng.next_u64() as usize) % k],
                }
            } else if cm == f64::NEG_INFINITY {
                // Degenerate segment (unreachable for a satisfiable tree).
                *nan_count += 1;
                (rng.next_u64() as usize) % n
            } else {
                let w: Vec<f64> = cond.iter().map(|&v| (v - cm).exp()).collect();
                draw_categorical(&w, rng)
            };
            states[mid_pos] = s_mid;
            stack.push((level + 1, idx * 2 + 1, s_mid, se));
            stack.push((level + 1, idx * 2, sb, s_mid));
        }

        // 3. Tokens per crossed edge — the sequential sampler's own rule.
        //    Pinned positions take the pin (the leaf already zeroed every
        //    edge not allowing it, so the crossed edge allows the pin).
        for t in 0..self.len {
            let (s, d) = (states[t], states[t + 1]);
            let e = self
                .fa
                .out_range(s)
                .find(|&e| self.fa.edge_dst(e) == d)
                .ok_or(FaError::InternalDeadState { pos: t })?;
            let pin = self.forced[t];
            if pin != FREE {
                out_tokens[t] = pin;
                continue;
            }
            let row = &self.logits[t * self.fa.vocab..(t + 1) * self.fa.vocab];
            let (a, b) = (self.fa.emit_start[e] as usize, self.fa.emit_start[e + 1] as usize);
            let toks = &self.fa.emit_tokens[a..b];
            if std::env::var("FA_DEBUG").is_ok() {
                println!("t={t} s={s} d={d} edge={e:?}");
            }
            out_tokens[t] = if argmax_tokens {
                let mut best = toks[0];
                let mut best_l = row[toks[0] as usize];
                for &v in &toks[1..] {
                    let l = row[v as usize];
                    if l > best_l {
                        best_l = l;
                        best = v;
                    }
                }
                best
            } else {
                draw_token_from_edge(toks, row, self.temperature, rng)
            };
        }
        Ok(final_state)
    }

    /// Per-position token marginals under the same folded leaves — f64
    /// forward/backward prefix products, `p(x_t = v) ∝ p_lm(v) · Σ_e
    /// F_t[s(e)] · exp(leaf_t[e]) · B_{t+1}[d(e)]`. The GOAT-gate quantity
    /// (1e-9 vs brute force); O(L·N³) — the tree's own cost class.
    pub fn marginals(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let levels = &self.scratch.levels;
        let leaf_off = self.level_off(0);
        let one = |m: f64| -> f64 {
            if m == f64::NEG_INFINITY {
                0.0
            } else {
                m.exp()
            }
        };
        // Forward prefixes F[0..=len]: F[0] = I; F[t+1] = F[t]·M_t.
        let mut fwd: Vec<Vec<f64>> = Vec::with_capacity(self.len + 1);
        let mut id = vec![0.0f64; n * n];
        for (i, v) in id.iter_mut().enumerate() {
            *v = if i % (n + 1) == 0 { 1.0 } else { 0.0 };
        }
        fwd.push(id);
        for t in 0..self.len {
            let prev = &fwd[t];
            let m = &levels[leaf_off + t];
            // exp-space product of an exp-space prefix and a log-space leaf.
            let mut next = vec![0.0f64; n * n];
            for (i, r) in next.chunks_mut(n).enumerate() {
                for (j, c) in r.iter_mut().enumerate() {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        s += prev[i * n + k] * one(m[k * n + j]);
                    }
                    *c = s;
                }
            }
            fwd.push(next);
        }
        // Backward suffixes B[len] = accept diag; B[t] = M_t·B[t+1].
        let mut bwd: Vec<Vec<f64>> = Vec::with_capacity(self.len + 1);
        let mut acc_diag = vec![0.0f64; n * n];
        for i in 0..n {
            acc_diag[i * n + i] = if self.fa.accept[i] { 1.0 } else { 0.0 };
        }
        bwd.push(acc_diag);
        for t in (0..self.len).rev() {
            let next = &bwd[bwd.len() - 1];
            let m = &levels[leaf_off + t];
            let mut prev = vec![0.0f64; n * n];
            for (i, r) in prev.chunks_mut(n).enumerate() {
                for (j, c) in r.iter_mut().enumerate() {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        s += one(m[i * n + k]) * next[k * n + j];
                    }
                    *c = s;
                }
            }
            bwd.push(prev);
        }
        bwd.reverse(); // bwd[t] = weight of positions t..len from state s

        // Token marginals: p(x_t = v) ∝ Σ_e flow_e · exp(lm(v)/T) for v in e
        // — ONE global shift for the position (a per-edge shift would break
        // the relative weighting between edges), divided out by the final
        // normalization. B_{t+1}[d] is the ROW SUM (total accepting weight
        // from state d over positions t+1..len).
        let temp = self.temperature as f64;
        let mut out = Vec::with_capacity(self.len);
        for t in 0..self.len {
            let f = &fwd[t][self.fa.start * n..(self.fa.start + 1) * n];
            let bnext = &bwd[t + 1];
            let row = &self.logits[t * self.fa.vocab..(t + 1) * self.fa.vocab];
            let gmax = row
                .iter()
                .fold(f64::NEG_INFINITY, |a, &l| a.max(l as f64 / temp));
            let mut marginal = vec![0.0f64; self.fa.vocab];
            for (s, &fs) in f.iter().enumerate() {
                if fs == 0.0 {
                    continue;
                }
                for e in self.fa.out_range(s) {
                    let d = self.fa.edge_dst(e);
                    let b_total: f64 = bnext[d * n..(d + 1) * n].iter().sum();
                    // The leaf mass e_log(e) does NOT appear here: it is the
                    // per-edge token normalizer, and conditioning on v
                    // cancels it exactly (p(v|e) ∝ exp(l(v)/T), and the
                    // path weight carries e_log(e) — product = exp(l(v)/T)).
                    let flow = fs * b_total;
                    if flow == 0.0 {
                        continue;
                    }
                    let (a, b) =
                        (self.fa.emit_start[e] as usize, self.fa.emit_start[e + 1] as usize);
                    let pin = self.forced[t];
                    for &v in &self.fa.emit_tokens[a..b] {
                        let w = if pin != FREE {
                            if v == pin {
                                1.0
                            } else {
                                0.0
                            }
                        } else {
                            ((row[v as usize] as f64) / temp - gmax).exp()
                        };
                        marginal[v as usize] += flow * w;
                    }
                }
            }
            // Normalize over the vocab (the constrained posterior).
            let z: f64 = marginal.iter().sum();
            if z > 0.0 {
                for v in marginal.iter_mut() {
                    *v /= z;
                }
            }
            out.push(marginal);
        }
        out
    }
}

/// Token draw given the crossed edge's token set — temperature multinomial
/// (greedy is handled at the call site; the raw argmax needs no draw).
fn draw_token_from_edge(
    toks: &[u32],
    row: &[f32],
    temp: f32,
    rng: &mut SplitMix64,
) -> u32 {
    let t = temp as f64;
    let weights: Vec<f64> = toks.iter().map(|&v| (row[v as usize] as f64 / t).exp()).collect();
    toks[draw_categorical(&weights, rng)]
    // (exp(l/T) without a shift: test-scale logits stay in f64 range; the
    // sequential sampler shifts because it is the f32 hot path.)
}


#[cfg(test)]
mod tests {
    use super::*;

    const N_DRAWS: usize = 200_000;

    /// Log-softmax of one row — the brute-force reference's p_lm (f64
    /// throughout so the reference is sharper than the sampler's f32).
    fn lm_probs(row: &[f32], temp: f32) -> Vec<f64> {
        let vals: Vec<f64> = row.iter().map(|&l| (l as f64) / (temp as f64)).collect();
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let acc: f64 = vals.iter().map(|&v| (v - max).exp()).sum();
        vals.iter().map(|&v| (v - max).exp() / acc).collect()
    }

    /// Exact constrained posterior by brute-force enumeration over all
    /// `vocab^len` sequences; `forced` pins positions (FREE = free).
    fn brute_force(
        fa: &Automaton,
        logits: &[f32],
        len: usize,
        forced: &[u32],
        temp: f32,
    ) -> Vec<f64> {
        let vocab = fa.vocab();
        let mut dist = vec![0.0f64; vocab.pow(len as u32)];
        let mut seq = vec![0u32; len];
        for (code, d) in dist.iter_mut().enumerate() {
            let mut c = code;
            for t in (0..len).rev() {
                seq[t] = (c % vocab) as u32;
                c /= vocab;
            }
            // Each code IS one full sequence; a code whose sequence violates
            // a pin is simply zero-probability under the conditioned
            // posterior (the pin itself contributes a constant factor that
            // cancels in the normalization).
            let mut skip = false;
            for t in 0..len {
                if forced[t] != FREE && seq[t] != forced[t] {
                    skip = true;
                    break;
                }
            }
            if skip {
                continue;
            }
            match fa.walk(&seq) {
                Some(end) if fa.is_accept(end) => {
                    let mut p = 1.0f64;
                    for t in 0..len {
                        let row = &logits[t * vocab..(t + 1) * vocab];
                        p *= lm_probs(row, temp)[seq[t] as usize];
                    }
                    *d = p;
                }
                _ => *d = 0.0,
            }
        }
        let z: f64 = dist.iter().sum();
        for d in &mut dist {
            *d /= z;
        }
        dist
    }

    /// TV distance between N sampler draws and an exact distribution.
    fn tv_distance(exact: &[f64], draws: impl Iterator<Item = Vec<u32>>, vocab: usize) -> f64 {
        let mut emp = vec![0.0f64; exact.len()];
        let mut n = 0usize;
        for seq in draws {
            let mut code = 0usize;
            for &t in &seq {
                code = code * vocab + t as usize;
            }
            emp[code] += 1.0;
            n += 1;
        }
        assert_eq!(n, N_DRAWS);
        for e in &mut emp {
            *e /= n as f64;
        }
        // The support must agree exactly: the sampler never emits a
        // zero-probability sequence (accepted-by-construction).
        for (i, &e) in emp.iter().enumerate() {
            if exact[i] == 0.0 {
                assert_eq!(e, 0.0, "sampler emitted rejected sequence {i}");
            }
        }
        0.5 * emp
            .iter()
            .zip(exact)
            .map(|(e, p)| (e - p).abs())
            .sum::<f64>()
    }

    fn sample_iter<'a>(
        fa: &'a Automaton,
        logits: &'a [f32],
        len: usize,
        forced: &'a [u32],
        temp: f32,
        seed: u64,
    ) -> impl Iterator<Item = Vec<u32>> + 'a {
        (0..N_DRAWS).map(move |i| {
            let mut scratch = FaScratch::new();
            let mut rng = SplitMix64::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut out = vec![0u32; len];
            fa.sample_joint(&mut scratch, logits, len, forced, temp, false, &mut rng, &mut out)
                .expect("draw");
            out
        })
    }

    #[test]
    fn determinism_invariants_are_enforced() {
        // (node, dst) duplicate with disjoint token sets: tokens stay unique
        // per node, the dsts collide.
        let err = AutomatonBuilder::new(2, 4, 0)
            .accept(1)
            .edge(0, 1, &[0])
            .edge(0, 1, &[1])
            .build()
            .unwrap_err();
        assert!(matches!(err, FaError::DuplicatePair { node: 0, dst: 1 }));

        // Distinct dsts but overlapping tokens.
        let err = AutomatonBuilder::new(3, 4, 0)
            .accept(2)
            .edge(0, 1, &[0, 1])
            .edge(0, 2, &[1, 2])
            .edge(1, 2, &[0, 1, 2, 3])
            .build()
            .unwrap_err();
        assert!(matches!(err, FaError::DuplicateEdgeToken { node: 0, token: 1 }));

        // Duplicate token WITHIN one edge.
        let err = AutomatonBuilder::new(2, 4, 0)
            .accept(1)
            .edge(0, 1, &[2, 2])
            .build()
            .unwrap_err();
        assert!(matches!(err, FaError::DuplicateTokenInEdge { token: 2, .. }));

        // Empty edge + out-of-range token + bad start.
        assert!(AutomatonBuilder::new(2, 4, 0)
            .accept(1)
            .edge(0, 1, &[])
            .build()
            .is_err());
        assert!(AutomatonBuilder::new(2, 4, 0)
            .accept(1)
            .edge(0, 1, &[9])
            .build()
            .is_err());
        assert!(AutomatonBuilder::new(2, 4, 5).accept(1).build().is_err());
    }

    #[test]
    fn chain_automaton_matches_brute_force() {
        // 0→1→2→3(accept), every edge allows all 4 tokens ⇒ the constraint
        // never binds and the posterior is the product of per-position marginals.
        let fa = AutomatonBuilder::new(4, 4, 0)
            .accept(3)
            .edge(0, 1, &[0, 1, 2, 3])
            .edge(1, 2, &[0, 1, 2, 3])
            .edge(2, 3, &[0, 1, 2, 3])
            .build()
            .unwrap();
        let len = 3;
        let logits: Vec<f32> = (0..(len * 4)).map(|i| ((i * 7919) % 23) as f32 - 8.0).collect();
        let forced = [FREE; 3];
        let exact = brute_force(&fa, &logits, len, &forced, 1.0);
        let tv = tv_distance(&exact, sample_iter(&fa, &logits, len, &forced, 1.0, 42), 4);
        assert!(tv < 0.03, "tv {tv}");
    }

    #[test]
    fn parity_automaton_matches_brute_force() {
        // Even number of 'a' (token 0): state 0 = even, 1 = odd; accept = 0.
        let fa = AutomatonBuilder::new(2, 2, 0)
            .accept(0)
            .edge(0, 1, &[0])
            .edge(0, 0, &[1])
            .edge(1, 0, &[0])
            .edge(1, 1, &[1])
            .build()
            .unwrap();
        let len = 4;
        let logits: Vec<f32> = (0..(len * 2)).map(|i| ((i * 104729) % 17) as f32 - 6.0).collect();
        let forced = [FREE; 4];
        let exact = brute_force(&fa, &logits, len, &forced, 1.0);
        let tv = tv_distance(&exact, sample_iter(&fa, &logits, len, &forced, 1.0, 7), 2);
        assert!(tv < 0.03, "tv {tv}");

        // Forced subset: pin x0='a', x3='b' — the exact posterior over the
        // two free positions conditioned on the pins.
        let forced = [0, FREE, FREE, 1];
        let exact = brute_force(&fa, &logits, len, &forced, 1.0);
        let tv = tv_distance(&exact, sample_iter(&fa, &logits, len, &forced, 1.0, 11), 2);
        assert!(tv < 0.03, "tv {tv}");
    }

    #[test]
    fn seeded_random_automata_match_brute_force() {
        // Four seeded random automata, all invariants-satisfying by
        // construction: a guaranteed chain spine 0→1→…→accept (so the block
        // is always satisfiable at len = spine length) plus random extra
        // edges carrying disjoint token subsets.
        for case in 0u64..4 {
            let (n, vocab, len) = (4, 4, 3);
            let mut rng = SplitMix64::new(0xB00B_0000 + case);
            let mut b = AutomatonBuilder::new(n, vocab, 0);
            b.accept(3);
            for s in 0..n - 1 {
                // Deterministic shuffle of the whole vocab, then a random
                // split: the first slice rides the spine edge s→s+1, the
                // rest splits across 0–2 extra edges to random distinct
                // dsts. A node's out-edges need not cover the vocab —
                // uncovered tokens are simply disallowed there.
                let mut toks: Vec<u32> = (0..vocab as u32).collect();
                for i in (1..toks.len()).rev() {
                    let j = (rng.next_u64() as usize) % (i + 1);
                    toks.swap(i, j);
                }
                let spine_take = 1 + (rng.next_u64() as usize) % (vocab - 1);
                b.edge(s, s + 1, &toks[..spine_take]);
                let rest = &toks[spine_take..];
                if rest.is_empty() {
                    continue;
                }
                let extra = (rng.next_u64() as usize) % 3; // 0..2 extra edges
                let mut used = std::collections::HashSet::new();
                used.insert(s + 1); // the spine edge owns this dst
                let mut dst = (s + 2 + (rng.next_u64() as usize)) % n;
                let mut off = 0usize;
                for f in 0..extra {
                    let take = if f == extra - 1 {
                        rest.len() - off
                    } else {
                        1 + (rng.next_u64() as usize) % (rest.len() - off - (extra - 1 - f))
                    };
                    if take == 0 {
                        break;
                    }
                    while !used.insert(dst) {
                        dst = (dst + 1) % n;
                    }
                    b.edge(s, dst, &rest[off..off + take]);
                    off += take;
                }
            }
            let fa = b.build().expect("generator must satisfy the invariants");
            let logits: Vec<f32> =
                (0..(len * vocab)).map(|i| ((i * 65537 + 11) % 29) as f32 - 11.0).collect();
            let forced = [FREE; 3];
            let exact = brute_force(&fa, &logits, len, &forced, 1.0);
            // The spine guarantees a satisfiable block, so the posterior has
            // mass — assert it rather than skipping (a vacuous pass here
            // would hide a sampler bug).
            assert!(exact.iter().sum::<f64>() > 0.0, "case {case}: unsatisfiable");
            let tv = tv_distance(
                &exact,
                sample_iter(&fa, &logits, len, &forced, 1.0, 100 + case),
                vocab,
            );
            assert!(tv < 0.03, "case {case}: tv {tv}");
        }
    }

    #[test]
    fn every_draw_is_accepted_by_construction() {
        let fa = AutomatonBuilder::new(3, 3, 0)
            .accept(2)
            .edge(0, 1, &[0, 1])
            .edge(1, 2, &[1, 2])
            .edge(2, 0, &[0])
            .build()
            .unwrap();
        let len = 5;
        let logits: Vec<f32> = (0..(len * 3)).map(|i| ((i * 7919) % 13) as f32 - 4.0).collect();
        let forced = [FREE; 5];
        for seq in sample_iter(&fa, &logits, len, &forced, 1.0, 99) {
            let end = fa.walk(&seq).expect("walk must survive (determinism)");
            assert!(fa.is_accept(end), "sequence {seq:?} not accepted");
        }
    }

    #[test]
    fn unsatisfiable_and_degenerate_blocks_error() {
        // Accepting node 2 needs 2 steps from start; a len-1 block cannot
        // land there.
        let fa = AutomatonBuilder::new(3, 2, 0)
            .accept(2)
            .edge(0, 1, &[0, 1])
            .edge(1, 2, &[0, 1])
            .build()
            .unwrap();
        let mut scratch = FaScratch::new();
        let mut rng = SplitMix64::new(1);
        let mut out = [0u32; 1];
        let logits1 = [0.0f32; 2];
        let err = fa
            .sample_joint(&mut scratch, &logits1, 1, &[FREE], 1.0, false, &mut rng, &mut out)
            .unwrap_err();
        assert!(matches!(err, FaError::Unsatisfiable { len: 1 }));

        // Zero-length block: accept iff the start node accepts.
        let mut out0: [u32; 0] = [];
        assert_eq!(
            fa.sample_joint(&mut scratch, &[], 0, &[], 1.0, false, &mut rng, &mut out0)
                .unwrap_err(),
            FaError::Unsatisfiable { len: 0 }
        );
        let fa2 = AutomatonBuilder::new(1, 2, 0).accept(0).build().unwrap();
        assert_eq!(
            fa2.sample_joint(&mut scratch, &[], 0, &[], 1.0, false, &mut rng, &mut out0)
                .unwrap(),
            0
        );

        // Dead start (logits shape-check first — supply a well-shaped row).
        let fa3 = AutomatonBuilder::new(2, 2, 1).accept(0).build().unwrap();
        let err = fa3
            .sample_joint(&mut scratch, &[0.0f32; 4], 2, &[FREE; 2], 1.0, false, &mut rng, &mut [0u32; 2])
            .unwrap_err();
        assert!(matches!(err, FaError::DeadStart(1)));

        // Shape guards (they fire before any satisfiability work).
        let fa4 = AutomatonBuilder::new(2, 2, 0).accept(1).edge(0, 1, &[0, 1]).build().unwrap();
        let logits2 = [0.0f32; 4];
        let mut out2 = [0u32; 2];
        assert!(fa4
            .sample_joint(&mut scratch, &logits2, 2, &[FREE], 1.0, false, &mut rng, &mut out2)
            .is_err());
        assert!(fa4
            .sample_joint(&mut scratch, &logits2, 2, &[FREE; 2], 1.0, false, &mut rng, &mut [0u32; 3])
            .is_err());
        assert!(fa4
            .sample_joint(&mut scratch, &[0.0; 5], 2, &[FREE; 2], 1.0, false, &mut rng, &mut out2)
            .is_err());
    }

    #[test]
    fn steps_to_accept_known_answers() {
        // 0→1→2(accept), plus a dead branch 0→3 (3 has no outgoing edges).
        let fa = AutomatonBuilder::new(4, 2, 0)
            .accept(2)
            .edge(0, 1, &[0])
            .edge(0, 3, &[1])
            .edge(1, 2, &[0, 1])
            .edge(2, 2, &[0, 1])
            .build()
            .unwrap();
        let d = fa.steps_to_accept();
        assert_eq!(d[2], 0);
        assert_eq!(d[1], 1);
        assert_eq!(d[0], 2);
        assert_eq!(d[3], UNREACHABLE);
    }

    #[test]
    fn greedy_tokens_take_the_raw_argmax_per_edge() {
        // Single path, last edge allows {1, 2}; logits favor 2 ⇒ greedy
        // picks 2 regardless of the sampled path's randomness.
        let fa = AutomatonBuilder::new(3, 3, 0)
            .accept(2)
            .edge(0, 1, &[0])
            .edge(1, 2, &[1, 2])
            .build()
            .unwrap();
        let len = 2;
        let logits = [5.0f32, -9.0, -9.0, -9.0, -9.0, 4.0]; // pos1 favors token 2
        let mut scratch = FaScratch::new();
        let mut out = [0u32; 2];
        let forced = [FREE; 2];
        let mut rng = SplitMix64::new(3);
        fa.sample_joint(&mut scratch, &logits, len, &forced, 1.0, true, &mut rng, &mut out)
            .unwrap();
        assert_eq!(out, [0, 2]);
    }

    #[test]
    fn same_seed_same_logits_same_draw() {
        let fa = AutomatonBuilder::new(2, 2, 0)
            .accept(1)
            .edge(0, 1, &[0, 1])
            .build()
            .unwrap();
        let logits = [0.3f32, -0.2];
        let mut a = [0u32; 1];
        let mut b = [0u32; 1];
        let (mut r1, mut r2) = (SplitMix64::new(2024), SplitMix64::new(2024));
        let (mut s1, mut s2) = (FaScratch::new(), FaScratch::new());
        let f = [FREE];
        fa.sample_joint(&mut s1, &logits, 1, &f, 1.0, false, &mut r1, &mut a)
            .unwrap();
        fa.sample_joint(&mut s2, &logits, 1, &f, 1.0, false, &mut r2, &mut b)
            .unwrap();
        assert_eq!(a, b);
    }

    // ── P0.5 — the parallel lane ────────────────────────────────────

    fn sample_parallel_iter<'a>(
        fa: &'a Automaton,
        logits: &'a [f32],
        len: usize,
        forced: &'a [u32],
        temp: f32,
        argmax: bool,
        seed: u64,
    ) -> impl Iterator<Item = Vec<u32>> + 'a {
        (0..N_DRAWS).map(move |i| {
            let mut scratch = TreeScratch::new();
            let tree = fa
                .build_tree(&mut scratch, logits, len, forced, temp)
                .expect("tree");
            let mut rng = SplitMix64::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut out = vec![0u32; len];
            let mut nan = 0u64;
            tree.sample(argmax, &mut rng, &mut nan, &mut out).expect("draw");
            assert_eq!(nan, 0, "unexpected NaN fallback");
            out
        })
    }

    #[test]
    fn parallel_matches_brute_force_and_sequential() {
        // Parity automaton, NON-power-of-two len (5 → 4 leaves + 1 identity
        // pad — the padding path), forced subset, both token postures.
        let fa = AutomatonBuilder::new(2, 2, 0)
            .accept(0)
            .edge(0, 1, &[0])
            .edge(0, 0, &[1])
            .edge(1, 0, &[0])
            .edge(1, 1, &[1])
            .build()
            .unwrap();
        let len = 5;
        let logits: Vec<f32> =
            (0..(len * 2)).map(|i| ((i * 104729) % 17) as f32 - 6.0).collect();
        let forced = [FREE; 5];
        let exact = brute_force(&fa, &logits, len, &forced, 1.0);
        let tv_par = tv_distance(
            &exact,
            sample_parallel_iter(&fa, &logits, len, &forced, 1.0, false, 77),
            2,
        );
        assert!(tv_par < 0.03, "parallel tv {tv_par}");
        let tv_seq = tv_distance(
            &exact,
            sample_iter(&fa, &logits, len, &forced, 1.0, 78),
            2,
        );
        assert!(tv_seq < 0.03, "sequential tv {tv_seq}");
        // Mutual agreement is bounded by the sum of the two bars above.

        // Forced subset through the parallel path.
        let forced = [0, FREE, FREE, FREE, 1];
        let exact = brute_force(&fa, &logits, len, &forced, 1.0);
        let tv = tv_distance(
            &exact,
            sample_parallel_iter(&fa, &logits, len, &forced, 1.0, false, 79),
            2,
        );
        assert!(tv < 0.03, "parallel forced tv {tv}");
    }

    #[test]
    fn parallel_power_of_two_len_matches_brute_force() {
        // 3-step cycle plus a self-loop on the accept node so len 4 (a true
        // power of two — no padding) is satisfiable.
        let fa = AutomatonBuilder::new(3, 3, 0)
            .accept(2)
            .edge(0, 1, &[0, 1])
            .edge(1, 2, &[1, 2])
            .edge(2, 0, &[0])
            .edge(2, 2, &[1, 2])
            .build()
            .unwrap();
        let len = 4; // exactly 2² — no padding
        let logits: Vec<f32> =
            (0..(len * 3)).map(|i| ((i * 65537) % 19) as f32 - 5.0).collect();
        let forced = [FREE; 4];
        let exact = brute_force(&fa, &logits, len, &forced, 1.0);
        let tv = tv_distance(
            &exact,
            sample_parallel_iter(&fa, &logits, len, &forced, 1.0, false, 81),
            3,
        );
        assert!(tv < 0.03, "tv {tv}");
    }

    #[test]
    fn parallel_greedy_tokens_match_the_sequential_rule() {
        let fa = AutomatonBuilder::new(3, 3, 0)
            .accept(2)
            .edge(0, 1, &[0])
            .edge(1, 2, &[1, 2])
            .build()
            .unwrap();
        let len = 2;
        let logits = [5.0f32, -9.0, -9.0, -9.0, -9.0, 4.0]; // pos1 favors 2
        let mut scratch = TreeScratch::new();
        let tree = fa.build_tree(&mut scratch, &logits, len, &[FREE; 2], 1.0).unwrap();
        let mut out = [0u32; 2];
        let mut rng = SplitMix64::new(3);
        let mut nan = 0u64;
        tree.sample(true, &mut rng, &mut nan, &mut out).unwrap();
        assert_eq!(out, [0, 2]);
    }

    #[test]
    fn marginals_match_brute_force_to_1e9() {
        let fa = AutomatonBuilder::new(3, 4, 0)
            .accept(2)
            .edge(0, 1, &[0, 1])
            .edge(0, 2, &[2])
            .edge(1, 2, &[1, 2, 3])
            .edge(2, 2, &[0, 1, 2, 3])
            .build()
            .unwrap();
        let len = 3;
        let logits: Vec<f32> =
            (0..(len * 4)).map(|i| ((i * 7919) % 23) as f32 - 8.0).collect();
        let forced = [FREE; 3];
        let mut scratch = TreeScratch::new();
        let tree = fa
            .build_tree(&mut scratch, &logits, len, &forced, 1.0)
            .unwrap();
        let marg = tree.marginals();
        assert_eq!(marg.len(), len);
        for (t, mt) in marg.iter().enumerate() {
            // Brute force: Σ over full sequences with x_t = v, normalized.
            let exact = brute_force(&fa, &logits, len, &forced, 1.0);
            let vocab = fa.vocab();
            for (v, mv) in mt.iter().enumerate() {
                let mut p = 0.0f64;
                for (code, &q) in exact.iter().enumerate() {
                    // Decode code → tokens.
                    let mut c = code;
                    let mut tok_at_t = 0u32;
                    for tt in (0..len).rev() {
                        let tok = (c % vocab) as u32;
                        c /= vocab;
                        if tt == t {
                            tok_at_t = tok;
                        }
                    }
                    if tok_at_t == v as u32 {
                        p += q;
                    }
                }
                assert!(
                    (mv - p).abs() < 1e-9,
                    "t={t} v={v}: {mv} vs {p}"
                );
            }
        }
    }

    #[test]
    fn parallel_unsatisfiable_errors_like_sequential() {
        // Accept node 2 needs 2 steps from start; a len-1 block cannot land
        // there (same shape as the sequential lane's test).
        let fa = AutomatonBuilder::new(3, 2, 0)
            .accept(2)
            .edge(0, 1, &[0, 1])
            .edge(1, 2, &[0, 1])
            .build()
            .unwrap();
        let mut scratch = TreeScratch::new();
        let built = fa.build_tree(&mut scratch, &[0.0f32; 2], 1, &[FREE], 1.0);
        assert!(matches!(built, Err(FaError::Unsatisfiable { len: 1 })));
    }
}
