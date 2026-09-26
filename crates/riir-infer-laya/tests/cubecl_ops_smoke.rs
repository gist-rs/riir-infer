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
        // 1/√2 — the named constant (an approx_constant-clean spelling of
        // the literal this arm was landed with).
        let s = core::f32::consts::FRAC_1_SQRT_2;
        c.scale(&mut x_cpu, s);
        b.scale(&mut x, s);
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

/// The S2 matmul family, op by op vs the CPU lane (plan 611 S2). Shape
/// table: m over {1, 4, 54, 128, 512} × the encoder geometry classes for
/// `matmul_w`; the attention batch at heads ∈ {1, 4}; the offset ops over
/// genuinely-padded parents (the whole-parent-bind shape). Tolerance 1e-3:
/// tiled-16 accumulation vs the CPU lane's sequential row dot over k ≤
/// 3072 of [-1,1) values — the same reduction-order class the G5 1e-3
/// budget prices, far above the ~1e-6 the gpu-side kernel tests measure.
#[test]
fn cubecl_matmul_family_matches_cpu() {
    let _gpu = gpu_lock();
    let b = CubeclBackend::new().expect("cubecl backend");
    let c = Cpu;
    const TOL: f32 = 1e-3;

    // The weights cache has NO epoch — every weight Vec below is held to
    // the end of the body (the module-doc hazard: a dropped weight Vec
    // whose address recycles into a same-(ptr,len) later Vec would hit a
    // stale permanent entry; the shape table's colliding n·k products make
    // that a real possibility, not a hypothetical).
    let mut weights: Vec<Vec<f32>> = Vec::new();

    // matmul_w — the weight projections, the encoder's exact shape.
    for &(m, k, n) in &[
        (1usize, 1024usize, 3072usize),
        (4, 1024, 1024),
        (54, 512, 2048),
        (128, 2048, 512),
        (512, 256, 256),
        (1, 260, 256), // the act_of class (k = d + 4 features)
    ] {
        b.begin_pass();
        let a = vec_of(m * k);
        // Move the ORIGINAL allocation into the keeper and borrow it back —
        // a clone would leave the cached (ptr, len) pointing at a Vec this
        // loop then drops.
        weights.push(vec_of(n * k).into_iter().map(|v| v * 0.5).collect());
        let w = weights.last().expect("keeper");
        let mut dst = vec![0f32; m * n];
        let mut dst_cpu = vec![0f32; m * n];
        c.matmul_w(&a, m, k, w, n, &mut dst_cpu);
        b.matmul_w(&a, m, k, w, n, &mut dst);
        assert_close(
            &format!("matmul_w {m}x{k}x{n}"),
            &sync_out(&b, &dst),
            &dst_cpu,
            TOL,
        );
    }

    // matmul — row-major × row-major at NON-ZERO offsets over padded
    // parents (the slab-in-parent shape the derivation-from-lengths kernel
    // cannot express; the offsets are the point).
    for &(m, k, n) in &[(3usize, 8usize, 5usize), (54, 64, 64), (128, 128, 64)] {
        b.begin_pass();
        let (ao, bo, doo) = (7usize, 11usize, 13usize);
        let a = vec_of(ao + m * k + 5);
        let bb = vec_of(bo + k * n + 9);
        let mut dst = vec![0f32; doo + m * n + 3];
        let mut dst_cpu = dst.clone();
        c.matmul(&a, ao, m, k, &bb, bo, n, &mut dst_cpu, doo);
        b.matmul(&a, ao, m, k, &bb, bo, n, &mut dst, doo);
        let got = sync_out(&b, &dst);
        assert_close(
            &format!("matmul {m}x{k}x{n} (offsets)"),
            &got[doo..doo + m * n],
            &dst_cpu[doo..doo + m * n],
            TOL,
        );
    }

    // matmul_kt — the score shape at offsets.
    for &(m, hd) in &[(5usize, 16usize), (54, 64), (128, 64)] {
        b.begin_pass();
        let (qo, ko, oo) = (3usize, 5usize, 2usize);
        let q = vec_of(qo + m * hd + 4);
        let k = vec_of(ko + m * hd + 6);
        let mut dst = vec![0f32; oo + m * m + 1];
        let mut dst_cpu = dst.clone();
        c.matmul_kt(&q, qo, m, hd, &k, ko, &mut dst_cpu, oo);
        b.matmul_kt(&q, qo, m, hd, &k, ko, &mut dst, oo);
        let got = sync_out(&b, &dst);
        assert_close(
            &format!("matmul_kt {m}x{hd} (offsets)"),
            &got[oo..oo + m * m],
            &dst_cpu[oo..oo + m * m],
            TOL,
        );
    }

    // matmul_kt_heads — the score batch (heads=1 exercises the same kernel
    // at z=1; heads=4 the real attention shape).
    for &(heads, m, hd) in &[
        (1usize, 4usize, 64usize),
        (4, 1, 64),
        (4, 4, 64),
        (4, 54, 64),
        (4, 128, 32),
    ] {
        b.begin_pass();
        let q = vec_of(heads * m * hd);
        let k = vec_of(heads * m * hd);
        let mut dst = vec![0f32; heads * m * m];
        let mut dst_cpu = vec![0f32; heads * m * m];
        c.matmul_kt_heads(&q, &k, heads, m, hd, &mut dst_cpu);
        b.matmul_kt_heads(&q, &k, heads, m, hd, &mut dst);
        assert_close(
            &format!("matmul_kt_heads {heads}x{m}x{hd}"),
            &sync_out(&b, &dst),
            &dst_cpu,
            TOL,
        );
    }

    // matmul_heads — the context batch (scores @ v per head; a is the
    // scores-shaped operand).
    for &(heads, m, hd) in &[(4usize, 4usize, 64usize), (4, 128, 64)] {
        b.begin_pass();
        let a = vec_of(heads * m * m);
        let v = vec_of(heads * m * hd);
        let mut dst = vec![0f32; heads * m * hd];
        let mut dst_cpu = vec![0f32; heads * m * hd];
        c.matmul_heads(&a, &v, heads, m, m, hd, &mut dst_cpu);
        b.matmul_heads(&a, &v, heads, m, m, hd, &mut dst);
        assert_close(
            &format!("matmul_heads {heads}x{m}x{hd}"),
            &sync_out(&b, &dst),
            &dst_cpu,
            TOL,
        );
    }
}

/// The residual-stream fold (`matmul_w_accum`) and the MLP fold
/// (`matmul_w_glu`) compose through the TRAIT DEFAULTS on this backend —
/// the plan's bit-identical-by-construction claim, proven behaviorally:
/// accum reads `x` device-current (the fold's add must see the matmul's
/// slot, not a stale host byte), and glu reads the fused slot `matmul_w`
/// just wrote in the same pass.
#[test]
fn cubecl_matmul_folds_match_cpu() {
    let _gpu = gpu_lock();
    let b = CubeclBackend::new().expect("cubecl backend");
    let c = Cpu;
    const TOL: f32 = 1e-3;

    let w_accum: Vec<f32> = vec_of(24 * 12).into_iter().map(|v| v * 0.25).collect();
    let w_glu: Vec<f32> = vec_of(48 * 12).into_iter().map(|v| v * 0.25).collect();

    // matmul_w_accum — x starts host-authored (uploaded on the first miss),
    // then accumulates the projection.
    {
        b.begin_pass();
        let (m, k, n) = (6usize, 12usize, 24usize);
        let mut x = vec_of(m * n);
        let a = vec_of(m * k);
        let mut x_cpu = x.clone();
        c.matmul_w_accum(&a, m, k, &w_accum, n, &mut x_cpu);
        b.matmul_w_accum(&a, m, k, &w_accum, n, &mut x);
        assert_close("matmul_w_accum", &sync_out(&b, &x), &x_cpu, TOL);
    }

    // matmul_w_glu — the fused [m × 2i] intermediate never exists on the
    // host; the gate must read the device slot.
    {
        b.begin_pass();
        let (m, k, i_sz) = (5usize, 12usize, 24usize);
        let a = vec_of(m * k);
        let mut act = vec![0f32; m * i_sz];
        let mut act_cpu = vec![0f32; m * i_sz];
        c.matmul_w_glu(&a, m, k, &w_glu, i_sz, &mut act_cpu);
        b.matmul_w_glu(&a, m, k, &w_glu, i_sz, &mut act);
        assert_close("matmul_w_glu", &sync_out(&b, &act), &act_cpu, TOL);
    }
}
