//! CubeCL-backend op parity + residency smoke (plan 611 S1b — T7 op-layer
//! unification). Runs only under `laya-riir-cubecl` — the [[test]] row
//! keeps the green-zero rule honest. Pattern: `metal_ops_smoke.rs` —
//! OP-level equivalence against the `Cpu` lane so a bad kernel or a bad
//! residency mapping is diagnosable without re-deriving it from a red G5
//! (the forward G5 at the cubecl posture is S4's gate; this file proves
//! the S1b slice it will stand on).
//!
//! The residency-specific arms (this file's reason to exist beyond the
//! Metal smoke's shape):
//!
//! - **chain reuse within an epoch** — running `add` twice on the same
//!   slice in ONE pass must produce x+2y, not x+y: a re-upload would read
//!   the ORIGINAL host bytes the second time (the device-currency
//!   contract, proven behaviorally);
//! - **begin_pass invalidation** — after `begin_pass`, a slot from the
//!   previous pass must be GONE: `download_into` on its slice panics
//!   (the stale-epoch-must-never-hit invariant, deterministic where the
//!   Metal smoke's recycled-address arms are a heap lottery);
//! - **offset views** — `copy_at`/`add` at non-zero offsets into a parent
//!   whose other regions were written by earlier ops in the SAME pass
//!   (the packed-slab seam shape).
//!
//! The Metal smoke's two anti-lottery disciplines are carried verbatim:
//! per-test `CubeclBackend::new()` (the weights cache keys device buffers
//! by slice `(ptr, len)` under the agent-lifetime stable-address contract
//! — a shared instance would violate it across scratch-Vec recycling),
//! and `begin_pass()` at the START of every independent arm (a recycled
//! chain address from an earlier arm cannot hit — the epoch moved). The
//! weight-slice Vecs (the biases) are additionally held to the end of
//! their test body: the weights cache has NO epoch, so a dropped weight
//! Vec whose address recycles into a same-(ptr,len) later Vec is the one
//! hazard `begin_pass` cannot clear.

#![cfg(feature = "laya-riir-cubecl")]

use std::sync::{Mutex, MutexGuard, OnceLock};

use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::cubecl::CubeclBackend;

/// Serialize test BODIES (the metal smoke's discipline — one live GPU
/// posture at a time keeps any residency failure attributable; poison-
/// recovering so one panic never cascades).
fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lcg() -> impl FnMut() -> f32 {
    let mut s = 0x9e37_79b9u32;
    move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) as f32) / 8_388_608.0 - 1.0
    }
}

fn vec_of(n: usize) -> Vec<f32> {
    let mut r = lcg();
    (0..n).map(|_| r()).collect()
}

fn assert_exact(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{name}: extent");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{name}: mismatch at {i}");
    }
}

fn assert_close(name: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: extent");
    let max = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("{name}: max abs {max:.4e} · tol {tol:.1e}");
    assert!(max <= tol, "{name}: DIVERGED max {max:.4e}");
}

/// Download a written slice's device result (the forward body's shape:
/// the host slice still holds stale bytes — the device is the truth).
fn sync_out(b: &CubeclBackend, buf: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; buf.len()];
    b.download_into(buf, &mut out);
    out
}

#[test]
fn cubecl_backend_posture() {
    let _gpu = gpu_lock();
    let b = CubeclBackend::new().expect("cubecl backend");
    assert_eq!(b.name(), "cubecl");
    let label = b.runtime_label();
    println!("cubecl posture: runtime {label}");
    assert!(!label.is_empty(), "runtime label must be resolved");
    // plan 611: the backend selection must be a REAL backend — on macOS
    // wgpu must resolve Metal (wgpu<msl>), never a silent software
    // fallback. Print, don't gate: the label is contract-checked by the
    // A/B record, not by a test that would break portability.
}

#[test]
fn cubecl_ops_match_cpu_op_by_op() {
    let _gpu = gpu_lock();
    let b = CubeclBackend::new().expect("cubecl backend");
    let c = Cpu;

    // The weight-slice Vecs live to the end of the body — the weights
    // cache is keyed (ptr, len) with NO epoch (see the module doc).
    let bias_a: Vec<f32> = vec_of(16).into_iter().map(|v| v * 0.5).collect();
    let bias_b: Vec<f32> = vec_of(24).into_iter().map(|v| v * 0.25 + 0.1).collect();

    // add — offset views into non-trivial parents (x_off 3, y_off 5).
    {
        b.begin_pass();
        let mut x = vec_of(20);
        let y = vec_of(16);
        let mut x_cpu = x.clone();
        c.add(&mut x_cpu, 3, &y, 5, 8);
        b.add(&mut x, 3, &y, 5, 8);
        assert_exact("add", &sync_out(&b, &x), &x_cpu);
    }

    // add_bias_row — rows × d with a weight slice.
    {
        b.begin_pass();
        let (rows, d) = (4usize, 16usize);
        let mut x = vec_of(rows * d);
        let mut x_cpu = x.clone();
        c.add_bias_row(&mut x_cpu, d, &bias_a);
        b.add_bias_row(&mut x, d, &bias_a);
        assert_exact("add_bias_row", &sync_out(&b, &x), &x_cpu);
    }

    // add_bias_row again with a DIFFERENT weight — proves the weights
    // cache distinguishes slices within the same backend instance.
    {
        b.begin_pass();
        let (rows, d) = (2usize, 24usize);
        let mut x = vec_of(rows * d);
        let mut x_cpu = x.clone();
        c.add_bias_row(&mut x_cpu, d, &bias_b);
        b.add_bias_row(&mut x, d, &bias_b);
        assert_exact("add_bias_row 2", &sync_out(&b, &x), &x_cpu);
    }

    // scale.
    {
        b.begin_pass();
        let mut x = vec_of(32);
        let mut x_cpu = x.clone();
        c.scale(&mut x_cpu, 0.7071);
        b.scale(&mut x, 0.7071);
        assert_exact("scale", &sync_out(&b, &x), &x_cpu);
    }

    // relu.
    {
        b.begin_pass();
        let mut x = vec_of(33); // ragged length: the launch grid rounds up,
        // the kernel's bounds check covers the tail.
        let mut x_cpu = x.clone();
        c.relu(&mut x_cpu);
        b.relu(&mut x);
        assert_exact("relu", &sync_out(&b, &x), &x_cpu);
    }

    // gelu_erf — the ONE tolerance-bearing op (cubecl's f32::erf vs the
    // CPU lane's libm::erff: a few ulp; 2e-5 catches a wrong-form gelu).
    {
        b.begin_pass();
        let mut x: Vec<f32> = vec_of(64).into_iter().map(|v| v * 3.0).collect();
        let mut x_cpu = x.clone();
        c.gelu_erf(&mut x_cpu);
        b.gelu_erf(&mut x);
        assert_close("gelu_erf", &sync_out(&b, &x), &x_cpu, 2e-5);
    }

    // glu_gelu_gate — write-first dst slot.
    {
        b.begin_pass();
        let (rows, i_sz) = (5usize, 12usize);
        let fused: Vec<f32> = vec_of(rows * 2 * i_sz).into_iter().map(|v| v * 2.0).collect();
        let mut out = vec![0f32; rows * i_sz];
        let mut out_cpu = vec![0f32; rows * i_sz];
        c.glu_gelu_gate(&fused, rows, i_sz, &mut out_cpu);
        b.glu_gelu_gate(&fused, rows, i_sz, &mut out);
        assert_close("glu_gelu_gate", &sync_out(&b, &out), &out_cpu, 2e-5);
    }
}

#[test]
fn cubecl_chain_reuse_within_epoch_is_device_current() {
    let _gpu = gpu_lock();
    let b = CubeclBackend::new().expect("cubecl backend");
    let c = Cpu;
    b.begin_pass();

    let mut x = vec_of(16);
    let y = vec_of(16);

    let mut x_cpu = x.clone();
    c.add(&mut x_cpu, 0, &y, 0, 16);
    c.add(&mut x_cpu, 0, &y, 0, 16);

    b.add(&mut x, 0, &y, 0, 16);
    b.add(&mut x, 0, &y, 0, 16);

    // A second re-upload would read the ORIGINAL host bytes (x still
    // holds stale content) and produce x+y — only a REUSED,
    // device-current slot produces x+2y.
    assert_exact("chain reuse", &sync_out(&b, &x), &x_cpu);
}

#[test]
fn cubecl_copy_paths_are_device_side() {
    let _gpu = gpu_lock();
    let b = CubeclBackend::new().expect("cubecl backend");
    let c = Cpu;

    // copy_into: the layer-0 identity path — src is DEVICE-current (an
    // op wrote it), so the copy must run device-side.
    {
        b.begin_pass();
        let mut src = vec_of(24);
        b.scale(&mut src, 2.0); // make src device-current with known content
        let mut src_cpu = src.clone();
        c.scale(&mut src_cpu, 2.0);

        let mut dst = vec![0f32; 24];
        let mut dst_cpu = dst.clone();
        c.copy_into(&src_cpu, &mut dst_cpu);
        b.copy_into(&src, &mut dst);
        assert_exact("copy_into", &sync_out(&b, &dst), &dst_cpu);
    }

    // copy_at: the packed-slab seam — two offset copies into ONE parent,
    // then both slabs read back correct (device-current parent, offsets
    // at bind time). Only the WRITTEN slabs are compared: untouched
    // regions of a fresh dst slot are garbage by contract (write-first
    // audit — the Metal lane's scratch() has the same property).
    {
        b.begin_pass();
        let s1 = vec_of(8);
        let s2 = vec_of(8);
        let mut parent = vec![0f32; 24];
        let mut parent_cpu = parent.clone();
        c.copy_at(&s1, 0, &mut parent_cpu, 4, 8);
        c.copy_at(&s2, 0, &mut parent_cpu, 16, 8);
        b.copy_at(&s1, 0, &mut parent, 4, 8);
        b.copy_at(&s2, 0, &mut parent, 16, 8);
        let got = sync_out(&b, &parent);
        assert_exact("copy_at slab1", &got[4..12], &parent_cpu[4..12]);
        assert_exact("copy_at slab2", &got[16..24], &parent_cpu[16..24]);
    }
}

#[test]
fn cubecl_begin_pass_invalidates_previous_pass_slots() {
    let _gpu = gpu_lock();
    let b = CubeclBackend::new().expect("cubecl backend");
    let c = Cpu;
    b.begin_pass();

    let mut x = vec_of(16);
    b.scale(&mut x, 2.0);
    let v = sync_out(&b, &x); // pass 0: fine
    let mut x_cpu = x.clone();
    c.scale(&mut x_cpu, 2.0);
    assert_exact("pass-0 scale", &v, &x_cpu);

    // New pass: the chain cleared. The old slot is gone — a download of
    // the stale slice must PANIC (the lazy-sync contract), never serve
    // stale bytes. Deterministic, unlike the Metal smoke's
    // recycled-address arms: the map was cleared, the key cannot exist.
    b.begin_pass();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sync_out(&b, &x);
    }));
    assert!(
        result.is_err(),
        "download_into after begin_pass must panic on a cleared slot"
    );
}
