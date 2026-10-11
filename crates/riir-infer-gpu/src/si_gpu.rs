//! Plan 620 (Plan 622 Phase 7) — the GPU-side scale-invariant session.
//!
//! [`SiGpuSession`] is the device half of `SiAttentionPlan` (the CPU
//! reference is `riir_infer_core::transformer::attention_si`): the schedule
//! LUT + the per-(layer, head) inv-sigma table + the armed flag, resident as
//! three device buffers whose ADDRESSES are stable for the session's
//! lifetime. Arming/disarming is a CONTENT flip on the flag — a captured
//! CUDA graph replays against the same pointers, so the arm/disarm between
//! graph replays never recaptures (the Issue-618 devpos law applied to si
//! state). Attaching/detaching the session itself changes which pointers the
//! decode launchers push, so the consumer invalidates its graph on those —
//! the `set_lora` precedent, `TernaryDeltanetGpuForwardCudarc`.
//!
//! The σ probe stays a CPU-forward instrument (never per-head score
//! downloads): probe once on any host, freeze the
//! [`HeadCalibration`] sidecar (BLAKE3-checked), and arm the GPU from the
//! sidecar. Cross-host digests differ in the f32 LSBs (AVX2 vs NEON dot
//! products) — quote digest + host beside every row (Plan 622 G3 law).

use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream};

use crate::cudarc_kernels::attention::SiDecodeArgs;
use katgpt_core::scale_invariant::{HeadCalibration, ScaleInvariantLut};

/// Device-resident scale-invariant session — see the module docs.
pub struct SiGpuSession {
    /// Interleaved schedule pairs `[a_0, m_0, …]` — the `ScaleInvariantLut`
    /// `am` layout verbatim (one upload, broadcast reads, LUT-local).
    lut: CudaSlice<f32>,
    /// `1/σ` per (layer, head); slot = `layer_idx * n_head + head` (the CPU
    /// harness contract). Neutral slots carry exactly `1.0` — the GDN
    /// layers' slots never launch attention, and an uncalibrated arm is all
    /// ones (near-unit-variance logits under QK-norm).
    inv_sigma: CudaSlice<f32>,
    /// The armed flag — the ONE buffer whose content flips between graph
    /// replays. 0 = the incumbent (bit-identical) posture.
    enabled: CudaSlice<i32>,
    /// Host-side mirror of `enabled` (API sanity only; the device buffer is
    /// the truth the kernels read).
    armed: bool,
    n_layers: usize,
    n_head: usize,
}

impl SiGpuSession {
    /// Upload the session for `n_layers · n_head` heads. `cal = None` is the
    /// uncalibrated arm (all-ones inv-sigma); `Some(cal)` must cover exactly
    /// `n_layers · n_head` slots (the wrong sidecar is loud here, at attach —
    /// never deep in a decode).
    ///
    /// # Errors
    /// Device allocation/copy failures, or a calibration table whose slot
    /// count does not match `n_layers * n_head`.
    pub fn attach(
        stream: &Arc<CudaStream>,
        lut: &ScaleInvariantLut,
        cal: Option<&HeadCalibration>,
        n_layers: usize,
        n_head: usize,
    ) -> Result<Self, crate::CudarcKernelError> {
        if let Some(cal) = cal {
            cal.validate_heads(n_layers * n_head)
                .map_err(crate::CudarcKernelError::InvalidArg)?;
        }
        // The LUT's `am` pairs, via the public per-distance accessor (the
        // layout contract lives in katgpt-core; this builds it without a
        // private-field reach-through).
        let max_distance = lut.max_distance();
        let mut am = Vec::with_capacity(2 * (max_distance + 1));
        for dist in 0..=max_distance {
            let (a, m) = lut.pair(dist);
            am.push(a);
            am.push(m);
        }
        let lut_dev = stream
            .clone_htod(&am)
            .map_err(|e| crate::CudarcKernelError::Launch(e.to_string()))?;

        let inv: Vec<f32> = match cal {
            None => vec![1.0; n_layers * n_head],
            Some(cal) => (0..n_layers * n_head)
                .map(|slot| 1.0 / cal.scale_for(slot))
                .collect(),
        };
        let inv_sigma = stream
            .clone_htod(&inv)
            .map_err(|e| crate::CudarcKernelError::Launch(e.to_string()))?;
        let enabled = stream
            .clone_htod(&[0i32])
            .map_err(|e| crate::CudarcKernelError::Launch(e.to_string()))?;

        Ok(Self {
            lut: lut_dev,
            inv_sigma,
            enabled,
            armed: false,
            n_layers,
            n_head,
        })
    }

    /// The per-forward launch args (full table; the per-layer window is
    /// narrowed at the launch site by [`SiDecodeArgs::window_for`]).
    pub fn decode_args(&self) -> SiDecodeArgs<'_> {
        SiDecodeArgs {
            lut: &self.lut,
            inv_sigma: &self.inv_sigma,
            enabled: &self.enabled,
        }
    }

    /// Arm or disarm by CONTENT flip — safe between captured-graph replays,
    /// never recaptures. The host mirror follows the write.
    ///
    /// # Errors
    /// Device copy failure.
    pub fn set_armed(
        &mut self,
        stream: &Arc<CudaStream>,
        on: bool,
    ) -> Result<(), crate::CudarcKernelError> {
        stream
            .memcpy_htod(std::slice::from_ref(&(on as i32)), &mut self.enabled)
            .map_err(|e| crate::CudarcKernelError::Launch(e.to_string()))?;
        self.armed = on;
        Ok(())
    }

    /// The host-side armed mirror (the device buffer is the truth).
    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// Slot geometry (diagnostics + the harness's row disclosures).
    pub fn slots(&self) -> usize {
        self.n_layers * self.n_head
    }

    pub fn n_head(&self) -> usize {
        self.n_head
    }
}
