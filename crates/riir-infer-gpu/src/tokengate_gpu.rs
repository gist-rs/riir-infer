//! Plan 622 T1.3 — the GPU-side per-key-bias session (the TokenGate lane's
//! device half, the [`crate::si_gpu::SiGpuSession`] twin for the Plan-622
//! attention-logit bias).
//!
//! [`KeyBiasGpuSession`] owns the full-table bias buffer
//! (`[n_layers][n_head][stride]` head-major per layer, slot-indexed — cache
//! slot IS the key identity), the arming flag, and the host mirrors. The
//! ADDRESSES are stable for the session's lifetime: arming/disarming AND
//! table rewrites are CONTENT flips (`set_armed` / `write_table`), so a
//! captured whole-prefill CUDA graph replays against the same pointers and
//! the posture flips never recapture (the Issue-618 devpos law — the si
//! session's precedent). Attaching/detaching the session itself changes
//! which pointers the prefill launchers push, so the consumer invalidates
//! its graphs on those (`attach_kb`/`detach_kb` do it).
//!
//! `stride` is fixed at attach and must cover EVERY live-key count the
//! session will serve (the largest `base_pos + chunk_len` the prefill can
//! reach — for a serving lane that is the model's block size). The kernels
//! index `bias[head * stride + pos]` only for in-range positions, so a
//! larger-than-needed stride is harmless (the pad rows are never read — the
//! walk gate proves it with a NaN pad); a SMALLER stride is a caller bug the
//! attach-time validation cannot see (the live key count is a runtime fact)
//! and reads the next head's row — size it at the context bound.

use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream};

use crate::cudarc_kernels::attention::KeyBiasDecodeArgs;

/// Device-resident per-key-bias session — see the module docs.
pub struct KeyBiasGpuSession {
    /// The full bias table — `[n_layers][n_head][stride]` f32, head-major
    /// per layer (the `KeyBiasDecodeArgs::window_for` layout verbatim).
    table: CudaSlice<f32>,
    /// Keys per head row (≥ every live-key count this session serves).
    stride: usize,
    /// The armed flag — the ONE buffer whose content flips between graph
    /// replays without recapture. 0 = the incumbent (bit-identical) posture.
    enabled: CudaSlice<i32>,
    /// Host-side mirror of `enabled` (API sanity only; the device buffer is
    /// the truth the kernels read).
    armed: bool,
    n_layers: usize,
    n_head: usize,
}

impl KeyBiasGpuSession {
    /// Upload the session for `n_layers · n_head` head rows of `stride`
    /// keys each. `host_table` is the initial CONTENT (an all-zero table +
    /// whatever pad the caller wants to prove unread) and must be exactly
    /// `n_layers * n_head * stride` f32 — the wrong-size table is loud here,
    /// at attach, never deep in a prefill. The session arrives DISARMED.
    ///
    /// # Errors
    /// Device allocation/copy failures, or a table whose length does not
    /// match `n_layers * n_head * stride`.
    pub fn attach(
        stream: &Arc<CudaStream>,
        host_table: &[f32],
        stride: usize,
        n_layers: usize,
        n_head: usize,
    ) -> Result<Self, crate::CudarcKernelError> {
        let expected = n_layers * n_head * stride;
        if host_table.len() != expected {
            return Err(crate::CudarcKernelError::InvalidArg(format!(
                "key-bias table len {} != n_layers({n_layers}) * n_head({n_head}) * stride({stride}) = {expected}",
                host_table.len()
            )));
        }
        let table = stream
            .clone_htod(host_table)
            .map_err(|e| crate::CudarcKernelError::Launch(e.to_string()))?;
        let enabled = stream
            .clone_htod(&[0i32])
            .map_err(|e| crate::CudarcKernelError::Launch(e.to_string()))?;
        Ok(Self {
            table,
            stride,
            enabled,
            armed: false,
            n_layers,
            n_head,
        })
    }

    /// The per-forward launch args (full table; the per-layer window is
    /// narrowed at the launch site by [`KeyBiasDecodeArgs::window_for`]).
    pub fn decode_args(&self) -> KeyBiasDecodeArgs<'_> {
        KeyBiasDecodeArgs {
            bias: &self.table,
            stride: self.stride,
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

    /// Rewrite the table CONTENT in place — pointer-stable, so a captured
    /// graph's next replay reads the new values (the same law as
    /// [`Self::set_armed`]; the caller synchronizes the stream when it needs
    /// the write ordered before other work).
    ///
    /// # Errors
    /// Device copy failure, or a table whose length does not match the
    /// attached geometry.
    pub fn write_table(
        &mut self,
        stream: &Arc<CudaStream>,
        host_table: &[f32],
    ) -> Result<(), crate::CudarcKernelError> {
        if host_table.len() != self.table.len() {
            return Err(crate::CudarcKernelError::InvalidArg(format!(
                "key-bias table len {} != attached {}",
                host_table.len(),
                self.table.len()
            )));
        }
        stream
            .memcpy_htod(host_table, &mut self.table)
            .map_err(|e| crate::CudarcKernelError::Launch(e.to_string()))?;
        Ok(())
    }

    /// The host-side armed mirror (the device buffer is the truth).
    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// Keys per head row (the attach-time stride).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Head rows total (`n_layers * n_head` — diagnostics + harness
    /// disclosures).
    pub fn slots(&self) -> usize {
        self.n_layers * self.n_head
    }

    pub fn n_head(&self) -> usize {
        self.n_head
    }
}
