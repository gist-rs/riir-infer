# Issue 044 — `ternary_q2_resident` CUDA lane bit-identity on the 4090

**Status:** OPEN 2026-10-13 — 4090 lane (needs a CUDA box; the M3 only type-checked it).

## Why

`ternary_q2_resident` was promoted into `deltanet_ternary_inference` in
`5ad32ba` (Bench 879 §Promotion). The loader now keeps Bonsai-2 projections
as raw Q2_0 blocks, and the GPU uploaders read a bit-plane view: cudarc
`proj_of` (`ternary_deltanet_gpu_forward_cudarc.rs`) takes the TRANSIENT view,
and `gemv_ternary_cuda_raw` keys its upload cache on the RESIDENT view's
buffer pointers. On the M3 this was proven only by compiling: the cudarc lane
was type-checked for `aarch64-unknown-linux-gnu` (11 gated modules compiled).
Metal was run (the pre-rotation prefill pin was bit-exact). The CUDA path has
not been run.

## Instrument

`si_flag0_probe` is a byte-identity recorder. It loads the Bonsai-2 PQ2_0
checkpoint, runs 128 prefill + 64 greedy tokens on the eager AND the graph
lane, and prints blake3 logit digests plus the full argmax sequence. Two
builds that differ anywhere fail it.

## Tasks

- [ ] T1: on the 4090, build and run at `995b577` (the parent of `5ad32ba`,
      planes-resident) and save stdout:
      `cargo build --release -p riir-infer-gpu --features ternary_gemv_cuda_raw,cuda_graphs_forward --bin si_flag0_probe`
      then `./target/release/si_flag0_probe --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`.
- [ ] T2: same at HEAD (Q2-resident + the N-way fused CPU calls from
      `7676556`). Diff the two stdout streams. They must be byte-identical.
- [ ] T3: record the diff verdict and box state in Bench 879. If it differs,
      bisect between `995b577` and HEAD before touching the uploaders.

Sync the box first via `schtasks /run /tn gitsync` (git over ssh hangs inside
sshd sessions; see the global 4090 notes).
