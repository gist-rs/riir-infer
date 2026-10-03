//! Plan 614 Phase 2 — the keyed-sampled spec-decode lane's cudarc
//! composition: the keyed verify loop over the dense forward + the DFlash2
//! GPU drafter, the serial keyed reference (the lossless oracle), and the
//! boot parity check. The pure decision layer is
//! [`crate::qwen38_spec`] (always-on, M3-tested); this module produces the
//! rows it decides on. CUDA-only — the whole file rides
//! `#![cfg(all(feature = "qwen38_spec", not(target_os = "macos")))]`
//! (the feature implies `ternary_gemv_cuda_raw`, the drafter's own gate).
//!
//! # The loop (the keyed variant of the Bench-759 chat loop)
//!
//! Per cycle, with the anchor `pending` at position `pos` (the unforwarded
//! bonus from the previous cycle): draft a chain from the GPU drafter's
//! block hidden + the target lm_head (the CPU selector walks it keyed —
//! the shared `(seed, position, token)` stream), optionally gated by the
//! per-position p-min confidence cut, feed `[pending, chain…]` as ONE
//! verify chunk WITH taps, decide the acceptance with
//! [`qwen38_spec::keyed_accept`] (per-row truncated keyed samples), commit
//! the accepted prefix via the 556 GDN rollback + journal replay, inject
//! the accepted rows' taps into the drafter ring, and emit `draft[1..j]`
//! plus the bonus. The committed stream is the SERIAL KEYED DECODE by
//! construction — the 4090 G1 gate (`plan614_phase2_spec_gates`) pins it
//! byte-exactly against [`run_serial_keyed`].
//!
//! Every piece is production-callable: no test-only state, no `dbg!`, the
//! stats struct is the consumer's telemetry.
#![cfg(all(feature = "qwen38_spec", not(target_os = "macos")))]

use crate::qwen38_dflash2::DFlash2Drafter;
use crate::qwen38_dflash2_gpu::{DFlash2GpuDrafter, DFlash2RingState};
use crate::qwen38_dense_cudarc::{Qwen38DenseForward, QWEN38_DFLASH2_TAP_LAYERS};
use crate::qwen38_spec::{keyed_accept, KeyedVerifyPosture};
use std::time::Instant;

/// The keyed lane's loop configuration: the sampling posture (the serial
/// stream's definition) plus the drafter-side p-min gate. The gate is a
/// DRAFTER property (which positions get a proposal), never a sampler
/// property — it changes the cost, never the stream.
#[derive(Debug, Clone)]
pub struct KeyedSpecConfig {
    pub posture: KeyedVerifyPosture,
    /// Per-position confidence floor: the chain is truncated BEFORE the
    /// first position whose keyed-walk confidence falls below it (the PR's
    /// gate semantics — an unconfident position stops the draft, it does
    /// not skip). `None` = ungated.
    pub p_min: Option<f32>,
    /// The DRAFTER's truncation (Plan 614's Phase-2 contract drafter half:
    /// the drafter never proposes outside its own nucleus — a pick outside
    /// the target's is a guaranteed miss). `(top_k, top_p)` over the
    /// drafter's own lm_head rows, the SAME semantics as the posture's
    /// sampler; `None`/`None` (the default) is the unmasked keyed walk —
    /// the A/B's 4th arm (`keyed_masked`) passes the posture's own
    /// truncation here, and promoting masked-to-default is a GOAT call on
    /// the A/B numbers, not a default.
    pub draft_top_k: Option<usize>,
    pub draft_top_p: Option<f32>,
}

impl Default for KeyedSpecConfig {
    fn default() -> Self {
        Self {
            posture: KeyedVerifyPosture::default(),
            p_min: None,
            draft_top_k: None,
            draft_top_p: None,
        }
    }
}

/// The keyed loop's telemetry (the Bench-759 `LoopStats` shape, keyed).
#[derive(Debug, Default)]
pub struct KeyedLoopStats {
    /// Acceptance histogram over fed positions `j ∈ 1..=p` (index 0 unused).
    pub hist: Vec<usize>,
    pub n_chunks: usize,
    /// Cycles accepting the whole fed window.
    pub n_full: usize,
    /// Cycles rewinding (j < p).
    pub n_rewind: usize,
    /// Cycles where a DRAFTED chain head was rejected (j == 1 with p > 1
    /// — the stream's own token committed as the correction).
    pub n_correction: usize,
    /// Cycles with NO draft at all (p == 1 — the p-min gate cut the chain
    /// to zero, or the caller ran the 1-token posture). Nothing was
    /// rejected; the cycle is a serial step through the chunk path.
    pub n_nodraft: usize,
    pub wall_s: f64,
    pub draft_ms: Vec<f64>,
    pub verify_ms: Vec<f64>,
    pub rewind_ms: Vec<f64>,
    pub inject_ms: Vec<f64>,
    pub committed: usize,
    /// Mean accepted fed positions per chunk (the L_eff proxy per cycle).
    pub mean_j: f64,
}

/// The keyed spec loop. `mask_emb` is the mask token's embedding row
/// (cached by the caller — 20 KB); `first` is the unforwarded bonus token
/// after the prompt (the caller's fill provides it); `prompt_len` is the
/// anchor's start position. Returns (generated stream, stats).
///
/// # Errors
/// Propagates GPU errors verbatim; a drafter/lm_head shape mismatch is a
/// bug and errors loudly (never a silent greedy fallback — the graceful
/// degrade posture is the CALLER's parity check, see
/// [`verify_chunk_one_row_parity`]).
#[allow(clippy::too_many_lines)]
pub fn run_keyed_spec_loop(
    gpu: &mut Qwen38DenseForward,
    drafter_gpu: &mut DFlash2GpuDrafter,
    cpu: &DFlash2Drafter,
    mask_emb: &[f32],
    cfg: &KeyedSpecConfig,
    n_gen: usize,
    first: u32,
    prompt_len: usize,
) -> Result<(Vec<u32>, KeyedLoopStats), String> {
    let e = cpu.cfg.n_embd;
    let bs = cpu.cfg.block_size;
    let vocab = gpu.cfg.vocab_size;
    let posture = &cfg.posture;
    let mut pending = first;
    let mut pos = prompt_len;
    let mut generated: Vec<u32> = vec![first];
    let mut st = KeyedLoopStats {
        hist: vec![0usize; bs + 2],
        ..Default::default()
    };
    let t_loop = Instant::now();
    while generated.len() < n_gen {
        // ── draft: anchor embedding (device dequant) + the noise block
        //       + lm_head + the KEYED walk (the shared stream). ──
        let t = Instant::now();
        let emb = gpu.embed_rows_host(&[pending])?;
        let hidden = drafter_gpu.draft_block_hidden_gpu(&emb[..e], mask_emb, pos)?;
        let rows_n: Vec<f32> = hidden[e..bs * e].to_vec();
        let logits_n = gpu.lm_head_rows_batched(&rows_n, bs - 1)?;
        // The keyed walk (the shared stream); the greedy posture delegates
        // inside it bit-identically — the explicit arm only skips the
        // noise arithmetic for the incumbent-exact A/B posture. The
        // masked arm (`draft_top_k`/`draft_top_p` set) filters the
        // drafter's own nucleus before the pick — the Phase-2 contract's
        // drafter half.
        let walk = if posture.is_greedy() {
            cpu.lattice_walk(pending, &hidden, &logits_n)
        } else if cfg.draft_top_k.is_some() || cfg.draft_top_p.is_some() {
            cpu.lattice_walk_keyed_masked(
                pending,
                &hidden,
                &logits_n,
                posture.seed,
                pos,
                posture.temperature,
                cfg.draft_top_k,
                cfg.draft_top_p,
            )
        } else {
            cpu.lattice_walk_keyed(
                pending,
                &hidden,
                &logits_n,
                posture.seed,
                pos,
                posture.temperature,
            )
        };
        // The p-min gate: the chain is truncated BEFORE the first
        // unconfident position (the PR's semantics — a cut, not a skip).
        let chain: Vec<(u32, f32)> = match cfg.p_min {
            Some(pm) => walk
                .chain
                .iter()
                .copied()
                .take_while(|&(_, c)| c >= pm)
                .collect(),
            None => walk.chain.clone(),
        };
        st.draft_ms.push(t.elapsed().as_secs_f64() * 1e3);

        let mut draft: Vec<u32> = Vec::with_capacity(chain.len() + 1);
        draft.push(pending);
        draft.extend(chain.iter().map(|&(t, _)| t));
        let p = draft.len();

        // ── verify + keyed acceptance ──
        let t = Instant::now();
        gpu.verify_snapshot_gdn()?;
        let (am, logits_flat) = gpu.forward_verify_chunk_taps_logits(&draft, pos)?;
        st.verify_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let acc = keyed_accept(posture, &logits_flat, vocab, &draft[1..], pos);
        let j = acc.j;
        st.hist[j] += 1;
        st.n_chunks += 1;
        if p == 1 {
            // No draft existed to accept or reject — a serial step through
            // the chunk path (the p-min gate cut the chain to zero).
            st.n_nodraft += 1;
        } else if j == p {
            st.n_full += 1;
        } else {
            st.n_rewind += 1;
            if j == 1 {
                st.n_correction += 1;
            }
            let t = Instant::now();
            gpu.verify_rollback_gdn()?;
            // The journal replay: same state trajectory as the advance
            // chunk, no weight reads (the 556 Stage-2 machinery; its own
            // gates pin the replay-vs-chunk bit-identity).
            gpu.verify_advance_gdn_replay(j)?;
            st.rewind_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }

        // ── inject the accepted prefix's taps (rows 0..j-1 = positions
        //       pos..pos+j-1; the bonus at pos+j rides the NEXT cycle's
        //       row 0) and commit. ──
        let t = Instant::now();
        let taps = gpu.verify_taps_download(j)?;
        drafter_gpu.inject_positions(&taps, pos)?;
        st.inject_ms.push(t.elapsed().as_secs_f64() * 1e3);

        // The greedy-posture cross-check: at T<=0 the keyed sampler IS the
        // argmax over survivors, which equals the chunk argmax whenever the
        // raw argmax is inside the (trivial, unfiltered) nucleus. Premise,
        // stated: this holds when the GPU argmax's tie-break matches the
        // CPU sampler's first-index rule (the Issue-697 packed reduction is
        // first-index by contract). DEBUG-ONLY — the release gate never
        // runs it; the real greedy identity gate is G1-greedy in the
        // plan614_phase2_spec_gates harness.
        if posture.is_greedy() {
            debug_assert_eq!(
                acc.bonus, am[j - 1],
                "greedy posture: keyed bonus {} != chunk argmax {} at pos {}",
                acc.bonus, am[j - 1], pos
            );
        }

        generated.extend_from_slice(&draft[1..j]);
        generated.push(acc.bonus);
        pos += j;
        pending = acc.bonus;
    }
    st.wall_s = t_loop.elapsed().as_secs_f64();
    st.committed = generated.len();
    st.mean_j = if st.n_chunks > 0 {
        st.hist.iter().enumerate().map(|(j, &n)| j * n).sum::<usize>() as f64
            / st.n_chunks as f64
    } else {
        0.0
    };
    // The final chunk can commit past `n_gen` (a full-block accept spanning
    // the boundary — G1-keyed seed 0xa11ce measured 98 for n_gen 96, 4090
    // 2026-10-03). The returned stream is exactly `n_gen`: the committed
    // stream IS the serial keyed decode, so its first `n_gen` tokens are
    // the serial decode's first `n_gen` — truncation is semantically exact,
    // never a repair. `st.committed` keeps the raw count.
    generated.truncate(n_gen);
    Ok((generated, st))
}

/// One decode lane's keyed session (the per-lane slice of
/// [`run_keyed_spec_loop_lanes`]): the anchor's start position, the
/// unforwarded bonus token after the lane's prompt, the lane's own
/// generation target, and the lane's OWN stream seed (the keyed walk and
/// the acceptance verdict both sample `(seed, position, token)` — lanes
/// sharing a seed and position would draft identical chains; per-lane seeds
/// keep the proposals independent while the acceptance contract keeps the
/// committed stream lossless either way). The lane's prompt must ALREADY be
/// in the target's lane state (KV + GDN) — the caller fills it; the drafter
/// ring the lane will use starts from whatever the caller injected (an
/// empty ring is a valid, tested posture — the ring shapes proposals, never
/// the stream).
#[derive(Debug, Clone, Copy)]
pub struct LaneKeyedSession {
    pub prompt_len: usize,
    pub first: u32,
    pub n_gen: usize,
    pub seed: u64,
}

/// The keyed spec loop over N decode lanes — the multi-lane composition of
/// [`run_keyed_spec_loop`] (Plan 614 Phase 2's open increment).
///
/// Per cycle, each lane drafts from its OWN pending/position/ring/seed (the
/// rings exchange through [`DFlash2GpuDrafter::exchange_rings`] — the
/// weights, kernels and scratch stay shared), the uniform window (pending +
/// the full block chain, `k = block_size` per lane) feeds ONE
/// `forward_lanes_verify_taps_logits` chunk (n·k packed rows,
/// row-invariant), every lane's acceptance runs on its OWN real rows only,
/// [`Qwen38DenseForward::lanes_commit`] applies the per-lane partial
/// accepts (rollback + journal replay inside), and each lane's accepted
/// tap rows inject into that lane's ring.
///
/// THE CONTRACT (the reason this composition is safe): the committed
/// stream is the serial keyed decode PER LANE by the same construction the
/// solo loop is — keyed acceptance filters whatever the drafter proposes,
/// so ring contents and cross-lane independence shape the COST, never the
/// stream. The 4090 G-lanes gate pins lane-vs-solo byte-identity.
///
/// `p_min` must be `None`: the chain is the full block on every lane (the
/// packing rule wants uniform `k`; a per-lane p-min cut would give uneven
/// chain lengths and right-padding waste — a future increment if the
/// gate ever earns it). Lanes overshoot their `n_gen` while slower lanes
/// finish — the overshoot is truncated from the returned streams.
///
/// `rings` is one [`DFlash2RingState`] per lane, caller-allocated (the
/// first can be the constructor's own set; extras via
/// [`DFlash2GpuDrafter::new_rings`]); after the call each slot holds the
/// same set it held before (the exchange is bracketed). Returns one
/// (stream, stats) pair per lane, in lane order; the per-cycle timings are
/// the SHARED phase costs (the draft phase serializes the lanes' walks, so
/// every lane's `draft_ms` sees the whole phase — the honest number for a
/// shared-GPU composition).
///
/// # Errors
/// Propagates GPU errors verbatim; per-lane wiring (rings count, `p_min`)
/// errors loudly before the first cycle.
#[allow(clippy::too_many_lines)]
pub fn run_keyed_spec_loop_lanes(
    gpu: &mut Qwen38DenseForward,
    drafter_gpu: &mut DFlash2GpuDrafter,
    cpu: &DFlash2Drafter,
    mask_emb: &[f32],
    cfg: &KeyedSpecConfig,
    sessions: &[LaneKeyedSession],
    rings: &mut [DFlash2RingState],
) -> Result<Vec<(Vec<u32>, KeyedLoopStats)>, String> {
    let e = cpu.cfg.n_embd;
    let bs = cpu.cfg.block_size;
    let vocab = gpu.cfg.vocab_size;
    let posture = &cfg.posture;
    let n = sessions.len();
    if rings.len() != n {
        return Err(format!(
            "lanes keyed loop: rings.len() {} != sessions.len() {n}",
            rings.len()
        ));
    }
    if cfg.p_min.is_some() {
        return Err(
            "lanes keyed loop: p_min must be None (uniform full-block chains; \
             a per-lane cut would right-pad the packed verify window)"
                .to_string(),
        );
    }
    if n == 0 {
        return Err("lanes keyed loop: at least one lane".to_string());
    }
    // The per-lane sampler: the shared posture's truncation + the lane's
    // own stream seed (the walk and the acceptance must sample the SAME
    // stream — the solo loop's single-posture property, per lane).
    let lane_posture: Vec<KeyedVerifyPosture> = sessions
        .iter()
        .map(|s| KeyedVerifyPosture {
            temperature: posture.temperature,
            top_k: posture.top_k,
            top_p: posture.top_p,
            seed: s.seed,
        })
        .collect();
    let mut pending: Vec<u32> = sessions.iter().map(|s| s.first).collect();
    let mut pos: Vec<usize> = sessions.iter().map(|s| s.prompt_len).collect();
    let mut generated: Vec<Vec<u32>> = sessions.iter().map(|s| vec![s.first]).collect();
    let mut st: Vec<KeyedLoopStats> = (0..n)
        .map(|_| KeyedLoopStats {
            hist: vec![0usize; bs + 2],
            ..Default::default()
        })
        .collect();
    let t_loop = Instant::now();
    loop {
        if (0..n).all(|l| generated[l].len() >= sessions[l].n_gen) {
            break;
        }
        // ── draft per lane (rings exchange in/out; weights stay shared) ──
        let t = Instant::now();
        let mut drafts: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut k_max = 1usize;
        for l in 0..n {
            drafter_gpu.exchange_rings(&mut rings[l]);
            let emb = gpu.embed_rows_host(&[pending[l]])?;
            let hidden = drafter_gpu.draft_block_hidden_gpu(&emb[..e], mask_emb, pos[l])?;
            let rows_n: Vec<f32> = hidden[e..bs * e].to_vec();
            let logits_n = gpu.lm_head_rows_batched(&rows_n, bs - 1)?;
            let walk = if posture.is_greedy() {
                cpu.lattice_walk(pending[l], &hidden, &logits_n)
            } else if cfg.draft_top_k.is_some() || cfg.draft_top_p.is_some() {
                cpu.lattice_walk_keyed_masked(
                    pending[l],
                    &hidden,
                    &logits_n,
                    lane_posture[l].seed,
                    pos[l],
                    lane_posture[l].temperature,
                    cfg.draft_top_k,
                    cfg.draft_top_p,
                )
            } else {
                cpu.lattice_walk_keyed(
                    pending[l],
                    &hidden,
                    &logits_n,
                    lane_posture[l].seed,
                    pos[l],
                    lane_posture[l].temperature,
                )
            };
            let mut draft: Vec<u32> = Vec::with_capacity(1 + walk.chain.len());
            draft.push(pending[l]);
            draft.extend(walk.chain.iter().map(|&(t, _)| t));
            k_max = k_max.max(draft.len());
            drafts.push(draft);
            drafter_gpu.exchange_rings(&mut rings[l]);
        }
        let draft_ms = t.elapsed().as_secs_f64() * 1e3;
        // ── pad to the uniform window (the packing rule; a no-op at the
        //       full-block chain every arm produces) ──
        let real_p: Vec<usize> = drafts.iter().map(|d| d.len()).collect();
        for draft in drafts.iter_mut() {
            // The lane's own pending again — padded rows are computed and
            // discarded, never sampled (the acceptance reads only the
            // lane's real rows below).
            let last = *draft.last().expect("draft carries its pending");
            draft.resize(k_max, last);
        }

        // ── verify (ONE packed chunk) + per-lane keyed acceptance ──
        let t = Instant::now();
        let (am, logits_flat) = gpu.forward_lanes_verify_taps_logits(&drafts, &pos)?;
        let mut accepted = vec![0usize; n];
        let mut bonuses = vec![0u32; n];
        for l in 0..n {
            let p_l = real_p[l];
            let slice = &logits_flat[(l * k_max) * vocab..(l * k_max + p_l) * vocab];
            let acc = keyed_accept(
                &lane_posture[l],
                slice,
                vocab,
                &drafts[l][1..p_l],
                pos[l],
            );
            accepted[l] = acc.j;
            bonuses[l] = acc.bonus;
            let s = &mut st[l];
            s.hist[acc.j] += 1;
            s.n_chunks += 1;
            if p_l == 1 {
                s.n_nodraft += 1;
            } else if acc.j == p_l {
                s.n_full += 1;
            } else {
                s.n_rewind += 1;
                if acc.j == 1 {
                    s.n_correction += 1;
                }
            }
            // The greedy-posture cross-check (the solo loop's, per lane).
            if lane_posture[l].is_greedy() {
                debug_assert_eq!(
                    acc.bonus, am[l][acc.j - 1],
                    "greedy posture: lane {l} keyed bonus {} != chunk argmax {} at pos {}",
                    acc.bonus, am[l][acc.j - 1], pos[l]
                );
            }
        }
        let verify_ms = t.elapsed().as_secs_f64() * 1e3;

        // ── commit (per-lane partial accepts: rollback + journal replay
        //       inside for every lane below the full window) ──
        gpu.lanes_commit(&accepted)?;

        // ── inject the accepted prefix's taps into each lane's ring ──
        // (VERBATIM the solo ordering: after the commit, whose
        // rollback/replay never touches the tap buffer; rows l·k .. l·k+j
        // are positions pos_l .. pos_l+j−1, the committed prefix.)
        let t = Instant::now();
        let taps_all = gpu.verify_taps_download(n * k_max)?;
        let inp = taps_all.len() / (n * k_max);
        for l in 0..n {
            let j = accepted[l];
            let taps_l = &taps_all[(l * k_max) * inp..(l * k_max + j) * inp];
            drafter_gpu.exchange_rings(&mut rings[l]);
            drafter_gpu.inject_positions(taps_l, pos[l])?;
            drafter_gpu.exchange_rings(&mut rings[l]);
        }
        let inject_ms = t.elapsed().as_secs_f64() * 1e3;
        for s in st.iter_mut() {
            s.draft_ms.push(draft_ms);
            s.verify_ms.push(verify_ms);
            s.inject_ms.push(inject_ms);
        }

        // ── advance every lane ──
        for l in 0..n {
            let j = accepted[l];
            generated[l].extend_from_slice(&drafts[l][1..j]);
            generated[l].push(bonuses[l]);
            pos[l] += j;
            pending[l] = bonuses[l];
        }
    }
    for (l, s) in st.iter_mut().enumerate() {
        s.wall_s = t_loop.elapsed().as_secs_f64();
        s.committed = generated[l].len();
        s.mean_j = if s.n_chunks > 0 {
            s.hist.iter().enumerate().map(|(j, &c)| j * c).sum::<usize>() as f64
                / s.n_chunks as f64
        } else {
            0.0
        };
    }
    // The doc'd overshoot: lanes finish on chunk boundaries while slower
    // lanes keep cycling, and any lane's final chunk can commit past its
    // own `n_gen`. Truncate per lane to the session's target (the same
    // semantics as the solo loop — the committed stream IS the serial
    // keyed decode, so the first `n_gen` tokens are exactly it;
    // `committed` keeps the raw count).
    for (l, g) in generated.iter_mut().enumerate() {
        g.truncate(sessions[l].n_gen);
    }
    Ok(generated.into_iter().zip(st).collect())
}
/// time, sampling EVERY position with the posture's truncated keyed
/// sampler (the argmax-only forward throws away the row the sampler
/// needs, so this rides the logits twin). Also collects the per-position
/// argmaxes (diagnostics: how far the keyed stream sits from greedy) and
/// the 5-tap features for every forwarded position (the A/B harness's
/// teacher-forced drafter input). Returns (stream, argmaxes, feats).
pub fn run_serial_keyed(
    gpu: &mut Qwen38DenseForward,
    prompt: &[u32],
    n_gen: usize,
    posture: &KeyedVerifyPosture,
) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>), String> {
    gpu.verify_reset_gdn()?;
    let n_tap = QWEN38_DFLASH2_TAP_LAYERS.len();
    let inp = n_tap * gpu.cfg.n_embd;
    let mut cap: Vec<Vec<f32>> = vec![vec![0.0f32; gpu.cfg.n_embd]; n_tap];
    let total = prompt.len() + n_gen;
    let mut feats: Vec<f32> = Vec::with_capacity(total * inp);
    let mut argmaxes: Vec<u32> = Vec::with_capacity(n_gen);
    let mut stream: Vec<u32> = Vec::with_capacity(n_gen);
    // The prompt: teacher-forced; only the LAST step's logits seed the
    // first keyed sample (the token at position prompt.len()).
    let mut next = 0u32;
    for (i, &t) in prompt.iter().enumerate() {
        let (_am, logits) =
            gpu.forward_token_capture_logits(t, i, &QWEN38_DFLASH2_TAP_LAYERS, &mut cap)?;
        for buf in &cap {
            feats.extend_from_slice(&buf[..gpu.cfg.n_embd]);
        }
        if i + 1 == prompt.len() {
            next = posture.sample_row(&logits, i + 1);
        }
    }
    // The generated stream: each step forwards the sampled token and
    // samples the next position from the row it produces.
    for pos in prompt.len()..total {
        stream.push(next);
        let (am, logits) =
            gpu.forward_token_capture_logits(next, pos, &QWEN38_DFLASH2_TAP_LAYERS, &mut cap)?;
        for buf in &cap {
            feats.extend_from_slice(&buf[..gpu.cfg.n_embd]);
        }
        argmaxes.push(am);
        next = posture.sample_row(&logits, pos + 1);
    }
    Ok((stream, argmaxes, feats))
}

/// The boot parity check (the graceful-degrade gate): a drafted verify
/// window must reproduce one-row decoding on THIS GPU/driver — the chunk's
/// per-row logits compared BIT-EXACTLY against one-row forwards of the
/// same tokens at the same positions, from the same pre-chunk state.
///
/// State discipline: snapshot at entry → the chunk → rollback → the
/// one-row feed (which rewrites the same KV rows; overwrite-before-read)
/// → compare → rollback → journal replay of the chunk, so the caller's
/// post-boot state is the CHUNK's end state either way. `probe` is the
/// drafted window (`probe[0]` = the pending token at `pos`); 2..=8 tokens.
/// Requires `enable_verify_taps()` + `enable_gdn_journal()` (the replay
/// leg) like the loop itself.
///
/// On ANY mismatch the caller must disable drafting (the TensorFold
/// posture: decode serial, log loud) — the check never repairs, it gates.
pub fn verify_chunk_one_row_parity(
    gpu: &mut Qwen38DenseForward,
    probe: &[u32],
    pos: usize,
) -> Result<(), String> {
    if probe.len() < 2 || probe.len() > 8 {
        return Err(format!(
            "parity probe: 2..=8 tokens (got {})",
            probe.len()
        ));
    }
    let vocab = gpu.cfg.vocab_size;
    gpu.verify_snapshot_gdn()?;
    let (chunk_am, chunk_logits) = gpu.forward_verify_chunk_taps_logits(probe, pos)?;
    gpu.verify_rollback_gdn()?;
    // The one-row feed: same tokens, same positions, one forward each.
    let mut one_logits: Vec<f32> = Vec::with_capacity(probe.len() * vocab);
    let mut one_am: Vec<u32> = Vec::with_capacity(probe.len());
    for (i, &t) in probe.iter().enumerate() {
        // No tap capture needed for the parity leg; the capture twin keeps
        // the call shape uniform (empty capture set = no taps).
        let (am, logits) = gpu.forward_token_capture_logits(t, pos + i, &[], &mut [])?;
        one_logits.extend_from_slice(&logits);
        one_am.push(am);
    }
    gpu.verify_rollback_gdn()?;
    gpu.verify_advance_gdn_replay(probe.len())?;
    // Bit-exact comparison — the contract is the T9.9 chunk-identity (the
    // same fold order both paths), never a tolerance.
    for r in 0..probe.len() {
        let a = &chunk_logits[r * vocab..(r + 1) * vocab];
        let b = &one_logits[r * vocab..(r + 1) * vocab];
        if a != b {
            let first = a.iter().zip(b).position(|(x, y)| x != y).expect("rows differ");
            return Err(format!(
                "parity FAIL row {r}: logits differ at token {first} (chunk {} vs one-row {}) — \
                 drafting must stay OFF on this GPU/driver",
                a[first], b[first]
            ));
        }
        if chunk_am[r] != one_am[r] {
            return Err(format!(
                "parity FAIL row {r}: argmax {} vs {} — drafting must stay OFF",
                chunk_am[r], one_am[r]
            ));
        }
    }
    Ok(())
}
