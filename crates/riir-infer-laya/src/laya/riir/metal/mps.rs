//! The MPS GEMM arm of the Metal backend (reflex issue 020 T13).
//!
//! The typed_decisions trio lost on the encoder GEMM at big m, where the
//! torch MPS oracle "runs the same FLOPs and still wins ~1.4×". T7 and the
//! roofline probe closed that axis among OUR instances (five refuted
//! kernel-level axes; the narrow tile is the local optimum of its family).
//! The reopen the issue named was a Metal-stack change, and the cheapest
//! one is to dispatch Apple's own `MPSMatrixMultiplication` — the kernel
//! family the oracle itself runs — for the unsplit batch-1 GEMMs.
//!
//! Priced before building (`examples/sgemm_mps_probe.rs`, 24 encoder
//! projection cells, paired): MPS/narrow geo-mean **0.88 at m 106 → 0.59
//! at m 1700**, and the output is **bit-identical to the narrow instance
//! on every cell** (CPU-anchored, not two zeros agreeing). Bit-identity is
//! what makes this an arm rather than a numerics change: every downstream
//! bit gate (fold bits, packed same-shape, G5) sees the same values.
//!
//! Resources: MPS encodes into the command buffer itself, so the backend's
//! one compute encoder is ended before and re-opened after each MPS call
//! (see `Metal::run_sgemm_mps`). All buffers are hazard-TRACKED, so the
//! encoder boundary orders the dependency exactly as the serial compute
//! encoder did.

// The legacy `objc` macros probe `feature = "cargo-clippy"`.
#![allow(unexpected_cfgs)]

use std::collections::HashMap;
use std::sync::Mutex;

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::runtime::{Class, NO, Object};
use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, CommandBufferRef, Device};

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

/// `MPSDataTypeFloat32` = `MPSDataTypeFloatBit | 32`.
const MPS_F32: u32 = 0x1000_0000 | 32;

/// Kernel-cache bound. The packed forward's m is Σseq per case, so the
/// `(m, n, k)` key population is unbounded across a suite; past the cap
/// the cache is flushed (one re-init per shape afterwards — cheap next to
/// a GEMM, and bounded memory beats an LRU's bookkeeping here).
const KERNEL_CACHE_CAP: usize = 512;

/// A +1-retained ObjC object, released on drop.
struct Owned(*mut Object);

// SAFETY: the wrapped objects are MPS kernels / matrices, immutable after
// init. Every use goes through `MpsGemm::kernels`' mutex (one encoding
// thread at a time — MPSKernel's documented requirement).
unsafe impl Send for Owned {}

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from an `alloc`/`init` pair (+1) and is
        // released exactly once.
        unsafe {
            let () = msg_send![self.0, release];
        }
    }
}

/// One GEMM operand: `(buffer, byte offset, rows, cols, row stride in f32)`.
pub(super) type Operand<'a> = (&'a Buffer, u64, u32, u32, u32);

/// The MPS GEMM dispatcher: class handles resolved once, plus the
/// `(m, n, k)`-keyed `MPSMatrixMultiplication` cache.
pub(super) struct MpsGemm {
    desc_cls: &'static Class,
    mat_cls: &'static Class,
    gemm_cls: &'static Class,
    kernels: Mutex<HashMap<(u32, u32, u32), Owned>>,
}

impl MpsGemm {
    /// `None` when the MPS classes do not resolve (the framework is absent
    /// or unlinkable) — the caller then stays on the narrow instance.
    pub(super) fn new() -> Option<Self> {
        Some(Self {
            desc_cls: Class::get("MPSMatrixDescriptor")?,
            mat_cls: Class::get("MPSMatrix")?,
            gemm_cls: Class::get("MPSMatrixMultiplication")?,
            kernels: Mutex::new(HashMap::new()),
        })
    }

    /// `rows × cols` f32 matrix view at `off` bytes into `buf`, rows `rs`
    /// floats apart. The descriptor is autoreleased (the caller's pool).
    fn matrix(&self, (buf, off, rows, cols, rs): Operand<'_>) -> Owned {
        // SAFETY: plain MPS object construction over a live MTLBuffer.
        unsafe {
            let desc: *mut Object = msg_send![self.desc_cls,
                matrixDescriptorWithRows: u64::from(rows)
                columns: u64::from(cols)
                rowBytes: u64::from(rs) * 4
                dataType: MPS_F32];
            let mat: *mut Object = msg_send![self.mat_cls, alloc];
            let buf_ptr = buf.as_ptr().cast::<Object>();
            Owned(msg_send![mat, initWithBuffer: buf_ptr offset: off descriptor: desc])
        }
    }

    /// Encode `c = a · b` (a `m × k`, b `k × n`, c `m × n`) into `cb`.
    /// Must run inside an autoreleasepool with no compute encoder open on
    /// `cb`.
    pub(super) fn encode(
        &self,
        device: &Device,
        cb: &CommandBufferRef,
        a: Operand<'_>,
        b: Operand<'_>,
        c: Operand<'_>,
    ) {
        let (m, k, n) = (a.2, a.3, c.3);
        debug_assert_eq!((b.2, b.3, c.2), (k, n, m), "mps operand shapes");
        let (ma, mb, mc) = (self.matrix(a), self.matrix(b), self.matrix(c));
        let mut cache = self.kernels.lock().expect("mps kernel cache poison");
        if cache.len() >= KERNEL_CACHE_CAP && !cache.contains_key(&(m, n, k)) {
            cache.clear();
        }
        let gemm = cache.entry((m, n, k)).or_insert_with(|| {
            // SAFETY: plain MPS kernel construction on the backend device.
            unsafe {
                let g: *mut Object = msg_send![self.gemm_cls, alloc];
                let dev = device.as_ptr().cast::<Object>();
                Owned(msg_send![g, initWithDevice: dev
                    transposeLeft: NO
                    transposeRight: NO
                    resultRows: u64::from(m)
                    resultColumns: u64::from(n)
                    interiorColumns: u64::from(k)
                    alpha: 1.0f64
                    beta: 0.0f64])
            }
        });
        // SAFETY: all four objects are live; `cb` has no open encoder.
        unsafe {
            let cbp = cb.as_ptr().cast::<Object>();
            let () = msg_send![gemm.0, encodeToCommandBuffer: cbp
                leftMatrix: ma.0
                rightMatrix: mb.0
                resultMatrix: mc.0];
        }
    }
}
