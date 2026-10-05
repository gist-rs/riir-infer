// Issue 035 — language-preserving minimization for compiled token
// automata (the G2 axis; ungated compiler engineering — the promote /
// demote GOAT verdict itself stays owner-gated).
//
// The schema compiler's subset construction bloats on object grammars:
// the 2^k member masks × k member chains land in k!·2^k subset states
// whose FUTURE language coincides (every order of emitting the same
// member set has the same continuations). Minimization collapses those
// to the 2^k·k (or better) distinct futures, shrinking both N and E —
// and the sampler's per-step cost is O(L·E).
//
// Two passes, both language-preserving by construction:
//
//   1. TRIM — drop states that are not reachable from the start or not
//      co-reachable to an accepting state. Neither class contributes a
//      single accepted string. (Dead branches survive the on-demand
//      subset construction: only forward reachability is enforced there.)
//
//   2. REFINEMENT — the standard partition fixed point, evaluated at
//      BLOCK granularity: two states merge when their accepting flags
//      agree and, for every token, the destination BLOCKS agree. Edges
//      into the same block are re-merged with unioned token sets before
//      the signature is taken, so the merged automaton keeps the
//      ≤1-edge-per-(from,to) determinism invariant.
//
// Why the exact-joint guarantee carries over (the load-bearing argument,
// not folklore): the sampler's next-token distribution at (state s,
// position t) mixes its out-edges by weight `e_log[t,e] · β(dst(e))`,
// where `e_log` depends only on the edge's TOKEN SET and `β` is the
// completion probability — equal for equivalent states at the fixed
// point. For a token v the edge containing it is UNIQUE per node (the
// builder's ≤1-edge-per-(node,token) law keeps every node's out-edge
// token sets disjoint), so its raw flow is `e_v · β(dst(e))` and the
// merged automaton's is `e_v · β(block(dst(e)))` — identical. Every
// continuation lands in an equivalent state, so the marginal over token
// sequences is preserved. The test battery pins this distributionally
// (same logits → matching per-sequence frequencies), not just
// language-level.

use crate::fa_posterior::{Automaton, AutomatonBuilder, FaError};
use std::collections::HashMap;

/// Before/after shape counts for one [`minimize_with_stats`] call.
#[derive(Clone, Copy, Debug)]
pub struct MinimizeStats {
    pub nodes_in: usize,
    pub edges_in: usize,
    pub nodes_out: usize,
    pub edges_out: usize,
}

/// Minimize `a`, reporting the shape delta. See the module doc for the two
/// passes and the exactness argument.
pub fn minimize_with_stats(a: &Automaton) -> Result<(Automaton, MinimizeStats), FaError> {
    let nodes_in = a.n_nodes();
    let edges_in = a.n_edges();

    // ── trim + dense relabel (ascending original order) ──────────────────
    let keep = trim_mask(a);
    if !keep[a.start()] {
        // No accepting string exists at all — the grammar accepts nothing.
        return Err(FaError::DeadStart(a.start()));
    }
    let mut relabel = vec![usize::MAX; nodes_in];
    let mut accept = Vec::with_capacity(nodes_in);
    for old in 0..nodes_in {
        if keep[old] {
            relabel[old] = accept.len();
            accept.push(a.is_accept(old));
        }
    }
    let nn = accept.len();

    // Edges whose destination was trimmed are dropped: the destination has
    // no path to an accepting state, so no accepted string traverses it.
    let mut out: Vec<Vec<(usize, Vec<u32>)>> = vec![Vec::new(); nn];
    for src in 0..nodes_in {
        if !keep[src] {
            continue;
        }
        for e in a.out_range(src) {
            let dst = a.edge_dst(e);
            if keep[dst] {
                out[relabel[src]].push((relabel[dst], a.edge_tokens(e).to_vec()));
            }
        }
    }

    // ── partition refinement at block granularity ────────────────────────
    let mut block: Vec<u32> = accept.iter().map(|&a| a as u32).collect();
    let mut n_blocks = block.iter().copied().max().map_or(0, |m| m as usize + 1);

    let mut sig_ids = vec![0u32; nn];
    loop {
        // Signature of one node: accepting flag + the node's out-edges
        // re-merged per destination BLOCK (token sets unioned, sorted).
        // Hash-consed to a dense id; ids never leak into the new numbering
        // (blocks are numbered by first member, nodes iterated ascending —
        // fully deterministic).
        let mut sig_map: HashMap<Vec<u32>, u32> = HashMap::new();
        for node in 0..nn {
            let mut merged: Vec<(u32, Vec<u32>)> = Vec::new();
            for &(dst, ref toks) in &out[node] {
                let db = block[dst];
                match merged.iter_mut().find(|(b, _)| *b == db) {
                    Some((_, acc)) => acc.extend_from_slice(toks),
                    None => merged.push((db, toks.clone())),
                }
            }
            merged.sort_unstable_by_key(|&(b, _)| b);
            let mut sig = Vec::with_capacity(1 + merged.len() * 2);
            sig.push(accept[node] as u32);
            for (db, mut toks) in merged {
                toks.sort_unstable();
                toks.dedup();
                sig.push(db);
                sig.push(toks.len() as u32);
                sig.extend_from_slice(&toks);
            }
            let next_id = sig_map.len() as u32;
            sig_ids[node] = *sig_map.entry(sig).or_insert(next_id);
        }

        // Split every block by signature id; new ids follow first-member
        // order (nodes ascending), so relabeling is deterministic.
        let mut group_new: HashMap<(u32, u32), u32> = HashMap::new();
        let mut new_block = vec![0u32; nn];
        let mut next = 0u32;
        for node in 0..nn {
            let key = (block[node], sig_ids[node]);
            let nb = match group_new.get(&key) {
                Some(&existing) => existing,
                None => {
                    group_new.insert(key, next);
                    next += 1;
                    next - 1
                }
            };
            new_block[node] = nb;
        }
        let refined = next as usize;
        block = new_block;
        if refined == n_blocks {
            break;
        }
        n_blocks = refined;
    }

    // ── rebuild ──────────────────────────────────────────────────────────
    // All members of a refined block share their accepting flag (the
    // initial partition splits on it; refinement never merges across
    // blocks). `start` keeps its relative position: blocks are numbered by
    // first member and the start node is live, so block[start] ≤ start.
    let mut block_accept = vec![false; n_blocks];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_blocks];
    for (node, &b) in block.iter().enumerate() {
        block_accept[b as usize] = accept[node];
        members[b as usize].push(node);
    }
    let start_new = block[relabel[a.start()]] as usize;
    let mut ab = AutomatonBuilder::new(n_blocks, a.vocab(), start_new);
    for (blk, &acc) in block_accept.iter().enumerate() {
        if acc {
            ab.accept(blk);
        }
    }
    let mut edges_out = 0usize;
    for (blk, mems) in members.iter().enumerate() {
        let mut merged: Vec<(usize, Vec<u32>)> = Vec::new();
        for &node in mems {
            for &(dst, ref toks) in &out[node] {
                let db = block[dst] as usize;
                match merged.iter_mut().find(|(d, _)| *d == db) {
                    Some((_, acc)) => acc.extend_from_slice(toks),
                    None => merged.push((db, toks.clone())),
                }
            }
        }
        merged.sort_unstable_by_key(|&(d, _)| d);
        for (db, mut toks) in merged {
            toks.sort_unstable();
            toks.dedup();
            ab.edge(blk, db, &toks);
            edges_out += 1;
        }
    }

    let minimized = ab.build()?;
    debug_assert_eq!(minimized.n_nodes(), n_blocks);
    debug_assert_eq!(minimized.n_edges(), edges_out);
    Ok((
        minimized,
        MinimizeStats {
            nodes_in,
            edges_in,
            nodes_out: n_blocks,
            edges_out,
        },
    ))
}

/// Minimize `a` (stats discarded).
pub fn minimize(a: &Automaton) -> Result<Automaton, FaError> {
    minimize_with_stats(a).map(|(m, _)| m)
}

/// States reachable from `start` AND co-reachable to some accepting state.
fn trim_mask(a: &Automaton) -> Vec<bool> {
    let n = a.n_nodes();
    // Forward reachability (iterative DFS over out-edges).
    let mut fwd = vec![false; n];
    let mut stack = vec![a.start()];
    fwd[a.start()] = true;
    while let Some(s) = stack.pop() {
        for e in a.out_range(s) {
            let d = a.edge_dst(e);
            if !fwd[d] {
                fwd[d] = true;
                stack.push(d);
            }
        }
    }

    // Reverse reachability from every accepting state.
    let mut rev_in: Vec<Vec<usize>> = vec![Vec::new(); n];
    for s in 0..n {
        for e in a.out_range(s) {
            rev_in[a.edge_dst(e)].push(s);
        }
    }
    let mut live = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    for (s, live_s) in live.iter_mut().enumerate() {
        if a.is_accept(s) {
            *live_s = true;
            queue.push_back(s);
        }
    }
    while let Some(d) = queue.pop_front() {
        for &s in &rev_in[d] {
            if !live[s] {
                live[s] = true;
                queue.push_back(s);
            }
        }
    }

    fwd.iter().zip(live.iter()).map(|(&f, &l)| f && l).collect()
}

#[cfg(test)]
mod tests;
