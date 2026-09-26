# Bench 005 — act-diagonal first slice: Ternary-Bonsai-2-27B-PQ2 (Issue 014 T1)

**Status:** RECORD — T1 of [Issue 014](../.issues/014_act_aware_ternary_fit_retention_walk.md)
landed 2026-09-26 (the collector bin + this first-slice read); T2/T3 not started.

## What ran

`act_diagonal_calibration` (new, opt-in feature `act_diagonal_calibration`):
one pass of the Bonsai-2 checkpoint over the `chat_probe` natural-text corpus,
observing the INPUT of every ternary projection beside each matvec through the
forward's own `TernaryMatvecHook` seam (no forward edits). The hook runs the
same `simd_ternary_group_matvec_parallel` the unhooked path runs, so the
forward is bit-identical and the observations are side-band. On the folded
model the observed inputs are the **post-rotation** tensors (the issue's trap 1
— what the ternary matvec actually sees).

- Model: `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (qwen35 hybrid,
  64 layers = 48 DeltaNet + 16 attention at l ≡ 3 (mod 4), n_embd 5120,
  mlp_hidden 17408, vocab 248320, `prism.hadamard` rotation ACTIVE, 7,206,168,928 bytes).
- Corpus: `riir-train/data/chat_probe` (21 pages → 2,713,703 tokens; first
  8192 taken).
- Slice: 16 sequences × 512 tokens = 8192 tokens (the issue's 16–128-sequence
  AWQ sampling spec, at its floor).
- Taps: 257 distinct inputs (4/layer: attn_in, layer_out, ffn_in, swiglu; +
  final_in), 401 hook calls/token, 2,167,808 channels, 33.08 MiB accumulator.
- Artifact: `.raw/act_diag_005/diagonal.bin` (17.3 MB, gitignored — the
  pipeline is deterministic; the digest below is the record).
  **BLAKE3 `5b6aead0901e3133cd7ee28512e75122c788ae9d720dec78d099e6b390d9a9d2`**

## Box state (the G2 law)

4090 workstation (shikuwa), i7-13700K 16 cores, **CPU lane** (the collector is
CPU-only by design), AC power, wall ≈ 87 min incl. the 231 s model load
(≈ 1.6 tok/s end-to-end; mtimes: run dir 14:39 → artifact 16:06), free RAM
18 GiB / 33.3 GiB at launch, GPU idle (6%, 473 MiB — the run does not touch
it), no competing compute jobs.

## The readout (aggregate per tap kind, 64 layers each)

| kind | median-of-med E[x²] | max/med (median · max) | top-1% share (median · max) |
|---|---|---|---|
| attn_in | 5.5e-1 | 6.7 · 56.9 | 3.7% · 6.0% |
| layer_out | 1.8e-2 | 3.2 · 90.9 | 2.2% · 11.6% |
| ffn_in | 4.1e-1 | 3.1 · 10.7 | 2.2% · 4.6% |
| swiglu | 1.8e-2 | 1.6 · 10.4 | 1.4% · 4.0% |
| final_in | 1.9 | 9.3 · 9.3 | 4.2% |

Uniform reference: max/med = 1, top-1% share = 1%.
Bench 896's synthetic anchors: the −54% diagonal gain came from planted
1%-channels×20 (a ~20% top-1% share); the log-normal spread gave −14%.

## Verdict (T1's question: did the rotation flatten the diagonal?)

**No — not globally, and the per-tap split is the finding:**

1. **`layer_out`** (the GDN recurrent output / attention output feeding
   out_proj and attn_wo) is the HEAVIEST-tapped input: median max/med 3.2,
   worst 90.9 (l00), top-1% share up to 11.6%. If any tensor benefits from the
   act-aware scale fit, it is this one.
2. **`attn_in`** carries real structure: median max/med 6.7, worst 56.9
   (l03 — the first attention layer), shares 3.7–6%. The in_proj_qkv/z and
   attn_wq/wk/wv groups all read this tap.
3. **`final_in` / lm_head** reads 9.3× with a 4.2% share.
4. **`swiglu` is the near-uniform tap** (1.6× median ratio, 1.4% median
   share) — the null prediction is LIVE for `down_proj` specifically: the
   SwiGLU product concentrates channels by construction, and the rotation
   plus that product have flattened what down_proj sees.
5. `ffn_in` is between (3.1× / 2.2%).

**T2 is not foreclosed.** Proceed to the scale-refit arms with per-tap
priors: act-aware fit has structure to exploit on layer_out / attn_in /
final_in, marginal on ffn_in, and down_proj is the predicted-null control the
issue's trap-3 comparison wants for free.

## Honest caveats

- **Reduced scale, honestly:** 8192 tokens (16 × 512) is the AWQ sampling
  floor, not the 10⁸–10⁹-token calibration the AWQ paper targets; the
  vk_calibration precedent applies — a reduced-scale read with measured
  coverage, never presented as the full pass. The digest pins this exact
  slice.
- **One box, one corpus.** chat_probe is chat text; a different domain
  shifts the diagonal. The artifact + digest make the slice reproducible.
- **`in_proj_a`/`in_proj_b` are out of scope by construction** — on Bonsai-2
  they are the Issue-980 DENSE (BF16) escape set, never quantized, so no
  ternary scale exists for T2 to refit on them.
- **The observation is side-band, not free of ordering effects:** the hook
  observes AFTER running the real kernel, per call, in call order — asserted
  every token (401 calls; any forward change trips the count/shape assert).
- **max/med and top-1% share are summaries, not the fit input** — T2 consumes
  the per-channel `E[x²]` diagonal from the artifact, not these tables.

## Fixture notes

- KV cache capped at seq_len (the `row_logit_floor_ppl` law): a 262K-context
  GGUF would otherwise allocate its whole advertised window per attention
  layer.
- The first-slice smoke (128 tokens) agreed directionally with the full run
  on every aggregate sign; only the digest differs (more tokens).
- Layer-type map confirmed in-run: attention layers sit at l ≡ 3 (mod 4)
  (n_obs 24576 = 3 obs/token on attn_in vs 16384 = 2 on GDN layers).
