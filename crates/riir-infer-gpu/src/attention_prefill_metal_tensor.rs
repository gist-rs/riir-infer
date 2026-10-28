//! Issue 1146 ladder 4(c) — the matmul2d flash-attention prefill.
//!
//! The Bench-925 census put attention at 37% of the promoted dist prefill
//! frame, the largest class, while the production macOS arm
//! (`QwenAttentionPrefillTiledM32CubeCL`) still runs its QK and PV as
//! per-lane dot chains with no matrix units. The incumbent we are measured
//! against does both on matrix hardware, and upstream llama.cpp on M5 runs
//! them on the Metal 4 tensor ops (`fa_tensor.metal`). This module is that
//! kernel for our layout: causal with `base_pos`, GQA, `1/sqrt(256)` scale,
//! the `σ(gate)` epilogue — the M=32 arm's semantics, same handles.
//!
//! **A tolerance arm.** Q and P are half, scores and O accumulate in f32,
//! K/V are read as f32 straight from the cache — never bit-identical to the
//! f32 CubeCL arms. G1 is a tolerance gate against the M=32 arm plus the
//! per-family KL read; see `tests/issue1146_attn_tensor_g1.rs`.
//!
//! **Dispatch.** Zero-copy on the shared CubeCL Metal queue through the
//! forward's existing [`MetalTensorZeroCopyGemm`]: its raw-buffer extraction
//! and its first-use wgpu zero-init tracker are SHARED (one tracker per pool
//! buffer — a second tracker would re-fill ranges the GEMM already marked),
//! and the output takes the same dedicated-buffer + blit-back route the GEMM
//! needs for cooperative-tensor stores into a pool buffer.
//!
//! **Ordering hazard (measured, `issue1146_attn_tensor_g1` `I1146A_NO_SYNC=1`):**
//! the leading `client.flush()` orders CubeCL KERNEL producers ahead of the
//! raw commit, but NOT a still-deferred host `client.write` — with one
//! pending, the kernel read the previous fixture's bytes from reused pool
//! memory (chance-level argmax, 8 of 20 cases). Production inputs (q, gate,
//! the KV cache) are kernel-produced; a caller that host-writes an input
//! must `sync()` first. The zero-copy GEMM shares the same contract.
//!
//! **Selector.** Opt-in `RIIR_PREFILL_ATTN_METAL_TENSOR=1` (or
//! [`set_prefill_attn_metal_tensor`]); when on it takes precedence over every
//! other arm in the `attn_decode` chain for routable shapes
//! ([`MetalTensorAttentionPrefill::routable`]).

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use metal::foreign_types::ForeignTypeRef;
use metal::{
    Buffer, BufferRef, CommandBuffer, CommandQueueRef, CompileOptions, ComputePipelineState,
    DeviceRef, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use crate::cubecl_runtime::ActiveRuntime;
use crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm;

const MSL_SOURCE: &str = include_str!("attention_prefill_metal_tensor.metal");
const KERNEL_NAME: &str = "attn_prefill_tensor_f32kv";

/// Head dim the kernel is specialised for.
pub const HEAD_DIM: usize = 256;
/// Queries per threadgroup (`Q` in the MSL).
const Q_PER_TG: usize = 32;
/// Cache rows per block (`C` in the MSL) — also the minimum cache length the
/// kernel can read in bounds.
pub const CACHE_BLOCK: usize = 64;
/// Threads per threadgroup (`NSG * 32`).
const THREADS: u64 = 8 * 32;
/// `sq half[Q*DK] | ss float[Q*C] | sp half[Q*C] | sr float[Q] | sf int[1]`,
/// padded to 16 bytes — Metal requires a threadgroup-memory length that is a
/// multiple of 16, and an unpadded 28_804 left `sf` (the last 4 bytes)
/// outside the allocation.
const THREADGROUP_BYTES: u64 = ((Q_PER_TG * HEAD_DIM * 2
    + Q_PER_TG * CACHE_BLOCK * 4
    + Q_PER_TG * CACHE_BLOCK * 2
    + Q_PER_TG * 4
    + 4)
.div_ceil(16)
    * 16) as u64;

// The MSL layout must fit Apple's 32 KiB threadgroup memory.
const _: () = assert!(THREADGROUP_BYTES == 28_816 && THREADGROUP_BYTES <= 32 * 1024);

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Process-wide launches through this arm (the vacuous-guard counter for
/// callers that cannot see the forward's private instance).
static LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Total tensor-attention dispatches in this process.
pub fn prefill_attn_metal_tensor_launch_count() -> u64 {
    LAUNCHES.load(Ordering::Relaxed)
}
static INITIALIZED: OnceLock<()> = OnceLock::new();

/// Whether the matmul2d attention arm is selected. Env
/// `RIIR_PREFILL_ATTN_METAL_TENSOR=1`/`on`/`true` opts in; read once, stored
/// into the live flag on the first call; the setter is authoritative after.
pub fn prefill_attn_metal_tensor_enabled() -> bool {
    INITIALIZED.get_or_init(|| {
        let on = matches!(
            std::env::var("RIIR_PREFILL_ATTN_METAL_TENSOR")
                .unwrap_or_default()
                .as_str(),
            "1" | "on" | "true"
        );
        if on {
            eprintln!("[Issue 1146] prefill attention = matmul2d tensor arm");
        }
        ENABLED.store(on, Ordering::Relaxed);
    });
    ENABLED.load(Ordering::Relaxed)
}

/// Select (or deselect) the matmul2d attention arm, overriding the env.
pub fn set_prefill_attn_metal_tensor(on: bool) {
    let _ = prefill_attn_metal_tensor_enabled();
    ENABLED.store(on, Ordering::Relaxed);
}

/// The compiled matmul2d flash-attention pipeline.
pub struct MetalTensorAttentionPrefill {
    pipeline: ComputePipelineState,
    /// Dedicated output buffers keyed by byte length (see the module doc).
    output_cache: Mutex<HashMap<u64, Buffer>>,
    dispatch_count: AtomicU64,
    /// The last committed command buffer, for [`Self::wait_last`].
    last_cmd: Mutex<Option<CommandBuffer>>,
}

// SAFETY: the pipeline state and buffers are Metal objects whose methods are
// thread-safe; mutation of the cache is behind a Mutex.
unsafe impl Send for MetalTensorAttentionPrefill {}
unsafe impl Sync for MetalTensorAttentionPrefill {}

impl MetalTensorAttentionPrefill {
    /// Compile the kernel on the zero-copy GEMM's device.
    pub fn new(gemm: &MetalTensorZeroCopyGemm) -> Result<Self, String> {
        let device_ref: &DeviceRef =
            unsafe { DeviceRef::from_ptr(gemm.device_ptr() as *mut metal::MTLDevice) };
        // Diagnostic knob: extra `#define`s prepended to the source
        // (`RIIR_ATTN_TENSOR_DEFINES="ATTN_RESCALE_THRESH=0.0f;..."`), the
        // `SWEEP_*` pattern of the GEMM — unset compiles the source as is.
        let defines: String = std::env::var("RIIR_ATTN_TENSOR_DEFINES")
            .unwrap_or_default()
            .split(';')
            .filter(|d| !d.trim().is_empty())
            .map(|d| format!("#define {}\n", d.trim().replacen('=', " ", 1)))
            .collect();
        let source = format!("{defines}{MSL_SOURCE}");
        let library = device_ref
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(|e| format!("attention MSL compilation failed: {e}"))?;
        let function = library
            .get_function(KERNEL_NAME, None)
            .map_err(|e| format!("function '{KERNEL_NAME}' not found: {e}"))?;
        let pipeline = device_ref
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("attention pipeline creation failed: {e}"))?;
        if pipeline.max_total_threads_per_threadgroup() < THREADS {
            return Err(format!(
                "attention pipeline admits {} threads/threadgroup, kernel needs {THREADS}",
                pipeline.max_total_threads_per_threadgroup()
            ));
        }
        Ok(Self {
            pipeline,
            output_cache: Mutex::new(HashMap::new()),
            dispatch_count: AtomicU64::new(0),
            last_cmd: Mutex::new(None),
        })
    }

    /// Whether a shape can route here: head_dim 256 and a cache of at least
    /// one full block (every block is then read in bounds).
    pub fn routable(head_dim: usize, base_pos: usize, p: usize) -> bool {
        head_dim == HEAD_DIM && p > 0 && base_pos + p >= CACHE_BLOCK
    }

    /// Block until the last dispatch completed — only for readers outside the
    /// shared queue's ordering (tests reading via `client.read_one`).
    pub fn wait_last(&self) {
        if let Some(cb) = self.last_cmd.lock().expect("last_cmd lock").as_ref() {
            cb.wait_until_completed();
        }
    }

    /// Dispatches issued so far (the vacuous-guard counter).
    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count.load(Ordering::Relaxed)
    }

    /// Causal gated attention for `p` queries at absolute positions
    /// `base_pos..base_pos + p` — the handle contract of
    /// `QwenAttentionPrefillTiledM32CubeCL::launch` (`query`/`gate`/`out`
    /// `[p][n_head][256]`, `key`/`value` `[base_pos + p][n_kv][256]`, f32).
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub fn dispatch(
        &self,
        gemm: &MetalTensorZeroCopyGemm,
        client: &ComputeClient<ActiveRuntime>,
        query: &Handle,
        key: &Handle,
        value: &Handle,
        gate: &Handle,
        out: &Handle,
        n_head: usize,
        n_kv: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        if !Self::routable(HEAD_DIM, base_pos, p) {
            return Err(format!(
                "shape not routable (base_pos {base_pos} + p {p} < {CACHE_BLOCK})"
            ));
        }
        // Producers of q/k/v/gate must be SUBMITTED before our raw commit
        // (same-queue ordering then makes their writes visible), and the
        // output range must be wgpu-initialized before the blit lands
        // behind wgpu's back — both via the GEMM's shared machinery.
        let _ = client.flush();
        let out_len = p * n_head * HEAD_DIM;
        gemm.ensure_wgpu_initialized(client, out, out_len)?;

        let (q_ptr, q_off) = MetalTensorZeroCopyGemm::extract_raw_buffer(client, query)?;
        let (k_ptr, k_off) = MetalTensorZeroCopyGemm::extract_raw_buffer(client, key)?;
        let (v_ptr, v_off) = MetalTensorZeroCopyGemm::extract_raw_buffer(client, value)?;
        let (g_ptr, g_off) = MetalTensorZeroCopyGemm::extract_raw_buffer(client, gate)?;
        let (o_ptr, o_off) = MetalTensorZeroCopyGemm::extract_raw_buffer(client, out)?;
        let q_ref: &BufferRef = unsafe { BufferRef::from_ptr(q_ptr) };
        let k_ref: &BufferRef = unsafe { BufferRef::from_ptr(k_ptr) };
        let v_ref: &BufferRef = unsafe { BufferRef::from_ptr(v_ptr) };
        let g_ref: &BufferRef = unsafe { BufferRef::from_ptr(g_ptr) };
        let o_ref: &BufferRef = unsafe { BufferRef::from_ptr(o_ptr) };

        let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;
        let dedicated = {
            let mut cache = self.output_cache.lock().expect("output_cache lock");
            cache
                .entry(out_bytes)
                .or_insert_with(|| {
                    let device_ref: &DeviceRef =
                        unsafe { DeviceRef::from_ptr(gemm.device_ptr() as *mut metal::MTLDevice) };
                    device_ref.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared)
                })
                .clone()
        };

        let (n_head_u, n_kv_u, p_u, base_u) =
            (n_head as u32, n_kv as u32, p as u32, base_pos as u32);
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

        let queue_ref: &CommandQueueRef =
            unsafe { CommandQueueRef::from_ptr(gemm.queue_ptr() as *mut metal::MTLCommandQueue) };
        let cmd = queue_ref.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(q_ref), q_off);
        enc.set_buffer(1, Some(k_ref), k_off);
        enc.set_buffer(2, Some(v_ref), v_off);
        enc.set_buffer(3, Some(g_ref), g_off);
        enc.set_buffer(4, Some(&dedicated), 0);
        enc.set_bytes(5, 4, &n_head_u as *const u32 as *const _);
        enc.set_bytes(6, 4, &n_kv_u as *const u32 as *const _);
        enc.set_bytes(7, 4, &p_u as *const u32 as *const _);
        enc.set_bytes(8, 4, &base_u as *const u32 as *const _);
        enc.set_bytes(9, 4, &scale as *const f32 as *const _);
        enc.set_threadgroup_memory_length(0, THREADGROUP_BYTES);
        for r in [q_ref, k_ref, v_ref, g_ref] {
            enc.use_resource(r, MTLResourceUsage::Read);
        }
        enc.use_resource(&dedicated, MTLResourceUsage::Write);
        enc.dispatch_thread_groups(
            MTLSize {
                width: p.div_ceil(Q_PER_TG) as u64,
                height: n_head as u64,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();

        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&dedicated, 0, o_ref, o_off, out_bytes);
        blit.end_encoding();
        cmd.commit();
        *self.last_cmd.lock().expect("last_cmd lock") = Some(cmd.to_owned());
        // No wait: the next CubeCL consumer is ordered after this command
        // buffer on the shared queue (the GEMM dispatch's contract).
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routable_needs_one_full_block() {
        assert!(MetalTensorAttentionPrefill::routable(256, 0, 64));
        assert!(MetalTensorAttentionPrefill::routable(256, 60, 4));
        assert!(!MetalTensorAttentionPrefill::routable(256, 0, 63));
        assert!(!MetalTensorAttentionPrefill::routable(128, 0, 512));
        assert!(!MetalTensorAttentionPrefill::routable(256, 512, 0));
    }
}
