# Issue 1107: Batched recurrent (GDN) prefill right-padding audit — pads-to-front state no-op

**Status:** Open — unclaimed

**Date:** 2026-10-22
**Source:** riir-infer Research 013 (arXiv:2610.26754 TokenGate distillation), upstream repo `token-gate-llms/TokenGate` @ `6e51aac5` (Apache-2.0), `tokengate/e2e_gate/gate_module/gate_forward.py::call_layer_padsafe`

## Claim (from the upstream source, measured there)

A batched linear-attention/recurrent layer fed RIGHT-PADDED rows silently poisons per-row state: pad tokens enter the recurrence after the real tail (and neighbor rows' padding shapes differ), so the state at the real sequence end is wrong. Upstream measured the keep-all boundary heldout collapsing to 0.679 vs 1.0 before the fix. The fix is closed-form and EXACT, no approximation: roll each row's padding to the FRONT of the row before the recurrence — a zero state is a provable no-op (k = v = 0 writes nothing to the state, decay multiplies 0, the conv window flushes with the real tail) — then roll the output back so the surrounding right-padded layout is untouched. One gather into scratch, alloc-free.

## Task

1. **Audit:** determine whether any batched GDN/qwen35-hybrid prefill path in this repo right-pads rows into a shared recurrence — candidates: `src/deltanet/forward.rs` (`prefill_qwen_deltanet_chunk_into` and any batched variants), `crates/riir-infer-gpu/src/deltanet_chunked_cubecl.rs`, `deltanet_pre_rec_fused_cubecl.rs`, `cudarc_kernels/deltanet.rs`, the flashmemory_gqa batch selector. Also check `dflash.rs` batched paths.
2. **If right-padding exists:** reproduce the defect with a state-hash test (padded batch vs per-row unpadded — expect mismatch), then implement the pads-to-front roll and pin G1 = state-hash identity across league GDN shapes, G4 = the roll is one gather into preallocated scratch.
3. **If no path right-pads:** close this issue with the audit record (paths checked + why padding cannot occur — e.g. strict batch-1 serving). A negative result is a valid close; do not leave it open as a maybe.

## Notes

- Applies independently of TokenGate gating — any batched recurrent prefill is exposed today. riir-infer Plan 622 (if it batch-prunes rows) inherits this law.
- Same law plausibly applies to riir-ai's HLA engine batching — file THERE only after verifying it actually batch-pads (boundary check first, do not assume).
- Upstream evidence quote (from `call_layer_padsafe` docstring): the pad-prefix roll makes batched state "exact, not an approximation" — keep-all boundary heldout 0.679 → 1.0 when introduced.
