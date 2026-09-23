//! Plan 550 / Bench 775 fix ladder #1 — the ANE IOSurface **zero-copy** split
//! executor.
//!
//! Bench 776 isolated the host-IO wall: `read(map+sync)` at 231-749 ms/op is
//! 60-75% of the hybrid total. This module eliminates the host roundtrip
//! entirely — the ANE request's OWN io surfaces are wrapped as R32Uint
//! `MTLTexture`s on the SHARED CubeCL Metal device (Issue 663 extraction),
//! and two raw-Metal kernels do the layout staging:
//!
//! ```text
//! per block:
//!   [zc_pack]   CubeCL input buffer (f32 token-major) → io_in texture
//!               (2×fp16/u32, channel-major)           — shared queue
//!   wait(pack)  ms-class (producer + pack only) — but see the P1 note below
//!   [eval_nc]   io_in → io_out                        — ANE, no memcpy
//!   [zc_unpack] io_out texture → target CubeCL buffers (channel offset +
//!               segment width, block token offset)    — shared queue
//! ```
//!
//! # Thread structure (Phase C contract preserved)
//!
//! The MAIN thread extracts every raw `id<MTLBuffer>` (input + per-segment
//! targets) BEFORE spawning; the worker touches ZERO CubeCL objects — only
//! retained `metal::Buffer`s, the raw queue (`MTLCommandQueue` is documented
//! thread-safe; command buffers may be created from any thread), the two
//! pipelines, and the ANE bridge.
//!
//! # Ordering
//!
//! Shared-queue FIFO: producer ⇒ pack (committed after the producer) ⇒
//! [worker waits pack completion, evals, enqueues unpacks] ⇒ the consumer
//! kernels the caller submits after `finish()` see complement + unpacks in
//! queue order. `finish()` is a JOIN ONLY — deliberately NO queue-drain
//! fence, which would serialize the next layer's submission behind this
//! layer's complement and destroy cross-layer pipelining.
//!
//! # Issue 769 T10 probe P1 — the wait is NOT scoped to this layer
//!
//! The paragraph above keeps a queue-drain fence out of `finish()` for a
//! stated reason, and the worker's `cmd.wait_until_completed()` is one such
//! fence, one level down: the pack rides the SHARED CubeCL queue (the raw
//! `MTLCommandQueue` extracted from `wgpu_queue.as_hal::<Metal>()` in
//! [`zc_context`] — the same object every CubeCL submission uses), so
//! waiting on it drains everything already enqueued ahead of it, not just
//! this layer's producer. Bench 777 measures that wait at 361 ms/op = 87%
//! of the hybrid's per-op budget.
//!
//! Two instruments split it, deliberately by different mechanisms:
//!
//! - **non-perturbing** (always live here): the pack buffer's own
//!   `GPUStartTime`/`GPUEndTime`, accumulated into [`T_ZC_PACKEXEC_NS`].
//!   `backlog = T_ZC_WAIT_NS - packexec`.
//! - **perturbing** (`RIIR_ANE_ZC_DRAIN_PROBE=1`, default off): an EMPTY
//!   command buffer committed and waited first — [`T_ZC_DRAIN_NS`].
//!
//! They answer the same question from opposite directions, which is the
//! point: agreeing is the cross-check, and the perturbing one adds a
//! submission so it must never be armed in an arm whose absolute is cited.
//!
//! # Fail-open (amended vs the host-IO job)
//!
//! The host-IO job guarantees "no writes before a successful join of ALL
//! blocks". The ZC path may land PARTIAL unpacks (block 0's unpack committed
//! before block 2's eval fails). This is safe: the fail-open path re-runs
//! the FULL GPU projection for the same outputs — queue-ordered AFTER our
//! unpacks, overwriting them. The contract is "eventually-consistent under
//! fail-open", not "zero partial writes".
//!
//! # Lifetime
//!
//! Target handles must outlive queue execution of the unpacks — they do:
//! the caller holds them until the consumer runs, and the consumer is
//! queue-ordered after the unpacks. A live CubeCL handle pins its pool
//! slice, so the extracted raw buffer cannot move while enqueued work
//! references it.

#![cfg(all(
    feature = "ane_prefill",
    feature = "metal_tensor_gemm",
    target_os = "macos",
    target_arch = "aarch64"
))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use metal::foreign_types::ForeignTypeRef;
use metal::{
    BufferRef, CommandBufferRef, CommandQueueRef, CompileOptions, ComputePipelineState, DeviceRef,
    MTLOrigin, MTLResourceOptions, MTLSize,
};
use wgpu::{Device as WgpuDevice, Queue as WgpuQueue};

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::bridge::{BridgeKernel, ZcTextures};

// ── stage counters (the ZC analog of exec.rs's; dumped by the harness) ─────
pub static T_ZC_PACK_NS: AtomicU64 = AtomicU64::new(0); // pack kernel ENQUEUE (encode+commit)
pub static T_ZC_WAIT_NS: AtomicU64 = AtomicU64::new(0); // wait_until_completed(pack)
pub static T_ZC_EVAL_NS: AtomicU64 = AtomicU64::new(0); // eval_nc (ANE)
pub static T_ZC_FLUSH_NS: AtomicU64 = AtomicU64::new(0); // the coherency flush (1-texel read + sync)
pub static T_ZC_UNPACK_NS: AtomicU64 = AtomicU64::new(0); // unpack kernel ENQUEUE
// Issue 769 T10 probe P1 (Bench 846 §9): splits `T_ZC_WAIT_NS` into its two
// parts. `T_ZC_WAIT_NS` times `wait_until_completed()` on the PACK buffer,
// which is committed to the SHARED CubeCL queue — so it drains everything
// already enqueued ahead of the pack, not just this layer's producer. With
// the probe armed, an EMPTY command buffer is committed and waited FIRST:
// that wait is the pure backlog drain, and the pack's own wait then runs
// against an empty queue, i.e. it measures pack execution alone.
//
// Validity self-check: `drain + wait` under the probe must ≈ `wait` without
// it. If it does not, the probe is lying and its split must not be cited.
pub static T_ZC_DRAIN_NS: AtomicU64 = AtomicU64::new(0); // backlog drain (probe only)
pub static N_ZC_DRAINS: AtomicU64 = AtomicU64::new(0);
// Issue 769 T10 probe P1, the NON-PERTURBING half (Bench 846 §9). The drain
// probe above answers the same question by adding a submission, so it must
// not be armed in an arm whose absolute is cited. These two read the pack
// command buffer's OWN `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` AFTER
// `wait_until_completed()` returns — two ObjC property reads, zero extra
// submissions, zero ordering change — so they are always live inside this
// (default-off) module and MAY be cited from a perf arm.
//
//   packexec = GPUEndTime - GPUStartTime  ... the pack's own GPU execution
//   backlog  = T_ZC_WAIT_NS - packexec    ... derived at dump time: every
//              part of the wait that is NOT our pack running, i.e. the time
//              the pack sat behind work already on the SHARED CubeCL queue
//              (plus driver scheduling + completion notification).
//
// The verdict this exists to decide: `backlog >> packexec` means the 361
// ms/op wait (Bench 777) is shared-queue coupling and a dedicated ANE-staging
// MTLCommandQueue fixes it cheaply; `backlog ~ 0` means the dependency is
// real and T10 has no cheap implementation.
pub static T_ZC_PACKEXEC_NS: AtomicU64 = AtomicU64::new(0);
// Timestamp validity split — an invalid (zero / non-monotone) pair is NOT the
// same finding as a measured zero, so it is counted separately rather than
// contributing 0 ns. A dump with `ts_bad > 0` must not be read as a share.
pub static N_ZC_TS_OK: AtomicU64 = AtomicU64::new(0);
pub static N_ZC_TS_BAD: AtomicU64 = AtomicU64::new(0);
// Issue 769 T10 (the second-queue variant). `T_ZC_FENCE_NS` is the MAIN
// thread's cost of arming the producer fence (flush + one empty
// signal-only command buffer per job); `N_ZC_2Q_PACKS` is the LIVENESS
// SENTINEL for the whole mechanism - a zero here with the flag armed means
// the pack never took the dedicated queue, which must not be read as "the
// second queue did not help".
pub static T_ZC_FENCE_NS: AtomicU64 = AtomicU64::new(0);
pub static N_ZC_FENCES: AtomicU64 = AtomicU64::new(0);
pub static N_ZC_2Q_PACKS: AtomicU64 = AtomicU64::new(0);
pub static N_ZC_EVALS: AtomicU64 = AtomicU64::new(0);
pub static N_ZC_FAILS: AtomicU64 = AtomicU64::new(0);

fn bump(c: &AtomicU64, t: Instant) {
    c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Issue 769 T10 probe P1 — `RIIR_ANE_ZC_DRAIN_PROBE=1` arms the backlog/exec
/// split described on [`T_ZC_DRAIN_NS`]. DEFAULT OFF: it commits one extra
/// (empty) command buffer per block, so it must never be on in a perf arm
/// whose absolute is being cited.
fn drain_probe_armed() -> bool {
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        std::env::var("RIIR_ANE_ZC_DRAIN_PROBE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Issue 769 T10 probe P1 (non-perturbing half) — accumulate the pack command
/// buffer's own GPU execution window from its `GPUStartTime`/`GPUEndTime`.
///
/// Call ONLY after `wait_until_completed()` has returned for `cmd`: before
/// completion both properties are 0 and the pair would be classified bad.
///
/// The two deltas are read as a DIFFERENCE, so no host/GPU timebase
/// conversion is involved — `GPUEndTime - GPUStartTime` is in seconds in
/// whatever base the driver reports, and only its magnitude is used.
// objc 0.2's `sel_impl!` expands a `cfg(feature = "cargo-clippy")` test, which
// resolves against THIS crate's feature set (the documented macro-cfg hazard)
// and warns under clippy only. Third-party expansion, nothing to fix here.
#[allow(unexpected_cfgs)]
fn record_pack_exec(cmd: &CommandBufferRef) {
    // SAFETY: `cmd` is a live `MTLCommandBuffer` (owned by the caller's
    // frame); `GPUStartTime`/`GPUEndTime` are read-only `CFTimeInterval`
    // (f64) properties on that protocol, present since macOS 10.15.
    // `msg_send!` expands to an unqualified `sel!`, which itself expands to
    // `sel_impl!` — all three must be in scope (the objc 0.2 incantation).
    use metal::objc::{msg_send, sel, sel_impl};
    let (t0, t1): (f64, f64) = unsafe {
        let obj = cmd.as_ptr() as *mut metal::objc::runtime::Object;
        (msg_send![obj, GPUStartTime], msg_send![obj, GPUEndTime])
    };
    if t0 > 0.0 && t1 >= t0 {
            T_ZC_PACKEXEC_NS.fetch_add(((t1 - t0) * 1e9) as u64, Ordering::Relaxed);
            N_ZC_TS_OK.fetch_add(1, Ordering::Relaxed);
        } else {
            N_ZC_TS_BAD.fetch_add(1, Ordering::Relaxed);
        }
}

/// Reset the ZC stage counters (the harness brackets each arm).
pub fn zc_stage_reset() {
    for c in [
        &T_ZC_PACK_NS,
        &T_ZC_WAIT_NS,
        &T_ZC_EVAL_NS,
        &T_ZC_FLUSH_NS,
        &T_ZC_UNPACK_NS,
        &T_ZC_DRAIN_NS,
        &T_ZC_PACKEXEC_NS,
        &T_ZC_FENCE_NS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    N_ZC_DRAINS.store(0, Ordering::Relaxed);
    N_ZC_FENCES.store(0, Ordering::Relaxed);
    N_ZC_2Q_PACKS.store(0, Ordering::Relaxed);
    N_ZC_TS_OK.store(0, Ordering::Relaxed);
    N_ZC_TS_BAD.store(0, Ordering::Relaxed);
    N_ZC_EVALS.store(0, Ordering::Relaxed);
    N_ZC_FAILS.store(0, Ordering::Relaxed);
    // Issue 769 T10 P5: the ZC path's `eval` share is the quantity T10 is
    // decided on, and a retried eval doubles it while leaving `fails` at 0 —
    // so the retry epoch must be this arm's, not the process's.
    super::bridge::eval_retry_reset();
}

/// One-line decomposition (ns each + counts).
pub fn zc_stage_dump() -> String {
    let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
    format!(
        "zc: pack {:.1}ms wait {:.1}ms eval {:.1}ms flush {:.1}ms unpack {:.1}ms · evals {} fails {} retries {}/{} ({:.1}%){}{}{}",
        g(&T_ZC_PACK_NS) as f64 / 1e6,
        g(&T_ZC_WAIT_NS) as f64 / 1e6,
        g(&T_ZC_EVAL_NS) as f64 / 1e6,
        g(&T_ZC_FLUSH_NS) as f64 / 1e6,
        g(&T_ZC_UNPACK_NS) as f64 / 1e6,
        g(&N_ZC_EVALS),
        g(&N_ZC_FAILS),
        // Issue 769 T10 P5: `fails` counts only BOTH-attempts-failed. A
        // first-attempt failure retries once (Issue 726 T4), usually succeeds,
        // and doubles that op's 100-245 ms ANE eval — invisible in `fails` and
        // indistinguishable from a slow ANE in the `eval` term. Printed with
        // its denominator and unconditionally, zeros included: a health field
        // that appears only when unhealthy cannot be read as green.
        super::bridge::N_EVAL_RETRIES.load(Ordering::Relaxed),
        super::bridge::N_EVAL_ATTEMPTS.load(Ordering::Relaxed),
        match super::bridge::N_EVAL_ATTEMPTS.load(Ordering::Relaxed) {
            0 => 0.0,
            n => super::bridge::N_EVAL_RETRIES.load(Ordering::Relaxed) as f64 / n as f64 * 100.0,
        },
        // Issue 769 T10: present when the second-queue variant OR its P4
        // flush-only control ran, and it reports the LIVENESS SENTINEL first.
        // `packs > 0` = the dedicated queue ran; `packs 0 fences > 0` = the
        // flush-only control arm; `packs 0 fences 0` prints nothing at all,
        // so a mechanism that never armed cannot be read as one that armed
        // and did not help.
        match g(&N_ZC_FENCES) + g(&N_ZC_2Q_PACKS) {
            0 => String::new(),
            _ => format!(
                " · 2q[packs {} fences {} fence-arm {:.2}ms]",
                g(&N_ZC_2Q_PACKS),
                g(&N_ZC_FENCES),
                g(&T_ZC_FENCE_NS) as f64 / 1e6,
            ),
        },
        // Probe P1 (non-perturbing half): always present when the module ran
        // at all. `ts_bad` is reported alongside so a dump whose timestamps
        // were unavailable cannot be read as a measured share.
        match g(&N_ZC_TS_OK) + g(&N_ZC_TS_BAD) {
            0 => String::new(),
            _ => {
                let wait = g(&T_ZC_WAIT_NS) as f64 / 1e6;
                let exec = g(&T_ZC_PACKEXEC_NS) as f64 / 1e6;
                let backlog = (wait - exec).max(0.0);
                format!(
                    " · P1ts[wait {wait:.1}ms = backlog {backlog:.1}ms ({:.0}%) + packexec {exec:.1}ms ({:.0}%) · ts_ok {} ts_bad {}]",
                    100.0 * backlog / wait.max(1e-9),
                    100.0 * exec / wait.max(1e-9),
                    g(&N_ZC_TS_OK),
                    g(&N_ZC_TS_BAD),
                )
            }
        },
        // Probe P1 (drain half): only printed when armed, so an unarmed dump
        // cannot be misread as "backlog measured at 0".
        match g(&N_ZC_DRAINS) {
            0 => String::new(),
            n => {
                let drain = g(&T_ZC_DRAIN_NS) as f64 / 1e6;
                let wait = g(&T_ZC_WAIT_NS) as f64 / 1e6;
                format!(
                    " · P1[drain {drain:.1}ms ({:.0}% of drain+wait) packexec {wait:.1}ms n {n}]",
                    100.0 * drain / (drain + wait).max(1e-9),
                )
            }
        },
    )
}

// ── MSL ─────────────────────────────────────────────────────────────────────

const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Pack: token-major f32 `[p × n]` → channel-major fp16 `[n × W]` (2 fp16 per
// R32Uint texel). One thread per u32 = one (channel, token-pair).
struct ZcPackParams {
    uint n;         // input channels (normx row stride)
    uint w;         // block tokens
    uint block;     // block index (token base = block * w)
    uint tex_w;     // io_in width in u32 texels
    uint total_u32; // u32 count = n * w / 2
};

kernel void zc_pack(device const float* normx [[buffer(0)]],
                    texture2d<uint, access::write> io_in [[texture(0)]],
                    constant ZcPackParams& p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.total_u32) return;
    uint l0 = 2u * gid;             // linear fp16 index of the low half
    uint c  = l0 / p.w;
    uint t  = l0 - c * p.w;
    ulong base = (ulong)(p.block * p.w + t) * p.n + c;
    half h0 = (half)normx[base];
    half h1 = (half)normx[base + p.n]; // next token, same channel
    uint s0 = (uint)as_type<ushort>(h0);
    uint s1 = (uint)as_type<ushort>(h1);
    uint packed = s1 << 16 | s0;
    io_in.write(uint4(packed, 0u, 0u, 0u), uint2(gid % p.tex_w, gid / p.tex_w));
}

// Unpack: channel-major fp16 `[oc_total × W]` (io_out) → token-major f32
// `[W × seg_dim]` slice of the fused output, at the block's token offset.
// One thread per OUTPUT element.
struct ZcUnpackParams {
    uint seg_dim;  // segment channel count
    uint w;        // block tokens
    uint chan_off; // segment start channel in the FUSED output
    uint tex_w;    // io_out width in u32 texels
    uint total;    // w * seg_dim output elements
};

kernel void zc_unpack(texture2d<uint, access::read> io_out [[texture(0)]],
                      device float* out [[buffer(0)]],
                      constant ZcUnpackParams& p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= p.total) return;
    uint t  = gid / p.seg_dim;
    uint cl = gid - t * p.seg_dim;
    uint c  = p.chan_off + cl;
    uint l  = c * p.w + t;
    uint u  = l / 2u;
    uint v  = io_out.read(uint2(u % p.tex_w, u / p.tex_w)).x;
    ushort bits = (l & 1u) ? ushort(v >> 16) : ushort(v & 0xffffu);
    half h = as_type<half>(bits);
    out[gid] = (float)h;
}

// Probe 778: the BUFFER-path twin of zc_unpack — identical indexing, but the
// source is a device buffer over the surface's OWN pages
// (newBufferWithBytesNoCopy on IOSurfaceGetBaseAddress) instead of a
// texture. The isolation ladder only ever exercised the texture-fetch path;
// the buffer path is a different cacheability domain and is the practical
// ZC-output fallback if it turns out coherent.
kernel void zc_unpack_buf(device const uint* io_out [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant ZcUnpackParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]])
{
    if (gid >= p.total) return;
    uint t  = gid / p.seg_dim;
    uint cl = gid - t * p.seg_dim;
    uint c  = p.chan_off + cl;
    uint l  = c * p.w + t;
    uint u  = l / 2u;
    uint v  = io_out[u];
    ushort bits = (l & 1u) ? ushort(v >> 16) : ushort(v & 0xffffu);
    half h = as_type<half>(bits);
    out[gid] = (float)h;
}

// DEBUG ONLY (the isolation ladder): fill a texture with a constant u32 —
// the GPU-write→GPU-read round-trip probe.
kernel void zc_debug_fill(texture2d<uint, access::write> tex [[texture(0)]],
                          constant uint2& geom [[buffer(0)]],
                          constant uint& val [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= geom.x || gid.y >= geom.y) return;
    tex.write(uint4(val, 0u, 0u, 0u), gid);
}
"#;

#[repr(C)]
struct ZcPackParams {
    n: u32,
    w: u32,
    block: u32,
    tex_w: u32,
    total_u32: u32,
}

#[repr(C)]
struct ZcUnpackParams {
    seg_dim: u32,
    w: u32,
    chan_off: u32,
    tex_w: u32,
    total: u32,
}

// ── the shared raw-Metal context ────────────────────────────────────────────

/// Raw device + queue (extracted from the shared CubeCL wgpu objects, the
/// Issue 663 T4 pattern) + the two staging pipelines. ONE per process —
/// initialized lazily on the first ZC dispatch from the forward's
/// `CubeCLContext` (single-GPU-context precedent: `GPU_INIT_LOCK`).
pub struct ZcContext {
    device: *mut metal::MTLDevice,
    queue: *mut metal::MTLCommandQueue,
    /// Issue 769 T10: the DEDICATED ANE-staging queue, created on the same
    /// shared `MTLDevice`. Only the pack rides it (see [`zc_worker_loop`]);
    /// the unpack stays on [`Self::queue`] because its ordering guarantee
    /// against the caller's consumer IS the shared-queue FIFO.
    ane_queue: metal::CommandQueue,
    /// The producer fence for the dedicated queue. Signalled on the SHARED
    /// queue (so it completes after everything enqueued at fence time,
    /// which includes the pack's input producer) and waited on the
    /// dedicated queue. Values are handed out by [`Self::next_fence`].
    fence: metal::Event,
    fence_ctr: AtomicU64,
    pack: ComputePipelineState,
    unpack: ComputePipelineState,
    unpack_buf: ComputePipelineState,
    fill: ComputePipelineState,
    /// Borrow justification for the raw pointers (Issue 663 pattern).
    _wgpu_device: Arc<WgpuDevice>,
    _wgpu_queue: Arc<WgpuQueue>,
}

impl ZcContext {
    /// The next fence value. `MTLEvent` requires strictly increasing
    /// signalled values per event, so this is a process-monotone counter,
    /// never reset (a reset would make a later job wait on a value the
    /// event has already passed - which succeeds immediately and silently
    /// drops the fence).
    fn next_fence(&self) -> u64 {
        self.fence_ctr.fetch_add(1, Ordering::Relaxed) + 1
    }
}

// SAFETY: the raw `id` pointers are ObjC objects usable from any thread
// (MTLDevice/MTLCommandQueue are documented thread-safe); the pipelines are
// Send+Sync via foreign-types; the wgpu Arcs are Send+Sync.
unsafe impl Send for ZcContext {}
unsafe impl Sync for ZcContext {}

static ZC_CTX: OnceLock<Option<Arc<ZcContext>>> = OnceLock::new();

/// The process-global ZC context, or `None` (non-Metal backend / MSL compile
/// failure) — callers fall back to the host-IO split.
pub fn zc_context(
    wgpu_device: Arc<WgpuDevice>,
    wgpu_queue: Arc<WgpuQueue>,
) -> Option<Arc<ZcContext>> {
    ZC_CTX
        .get_or_init(|| {
            let device_ptr: *mut metal::MTLDevice = unsafe {
                let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
                let dev = guard.as_ref()?;
                let proto = dev.raw_device();
                (&**proto) as *const _ as *mut metal::MTLDevice
            };
            let queue_ptr: *mut metal::MTLCommandQueue = unsafe {
                let guard = wgpu_queue.as_hal::<wgpu::hal::api::Metal>();
                let q = guard.as_ref()?;
                let proto = q.as_raw();
                proto as *const _ as *mut metal::MTLCommandQueue
            };
            let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
            let library = device_ref
                .new_library_with_source(MSL_SOURCE, &CompileOptions::new())
                .map_err(|e| {
                    eprintln!("[ane-zc] MSL compilation failed ({e}); zero-copy IO disabled");
                    e
                })
                .ok()?;
            let mk = |name: &str| -> Option<ComputePipelineState> {
                let function = library.get_function(name, None).ok()?;
                device_ref
                    .new_compute_pipeline_state_with_function(&function)
                    .ok()
            };
            // Issue 769 T10: the dedicated staging queue + its producer
            // fence. Both are created unconditionally (an unused
            // MTLCommandQueue costs no submission and no memory worth
            // measuring), so arming the flag needs no re-init and the
            // default-off arm is byte-identical in behaviour.
            let ane_queue = device_ref.new_command_queue();
            let fence = device_ref.new_event();
            let pack = mk("zc_pack")?;
            let unpack = mk("zc_unpack")?;
            let unpack_buf = mk("zc_unpack_buf")?;
            let fill = mk("zc_debug_fill")?;
            Some(Arc::new(ZcContext {
                device: device_ptr,
                queue: queue_ptr,
                ane_queue,
                fence,
                fence_ctr: AtomicU64::new(0),
                pack,
                unpack,
                unpack_buf,
                fill,
                _wgpu_device: wgpu_device,
                _wgpu_queue: wgpu_queue,
            }))
        })
        .clone()
}

/// The cached global — initialized by the forward's `new()` (which holds the
/// `CubeCLContext`); the split dispatch sites read this.
pub fn zc_context_cached() -> Option<Arc<ZcContext>> {
    ZC_CTX.get().and_then(|o| o.clone())
}

/// Extract `(raw id<MTLBuffer>, byte offset)` from a CubeCL handle — the
/// Issue 663 `extract_raw_buffer` pattern.
///
/// NO retain is taken: the raw pointer is valid for as long as the HANDLE
/// lives. Every ZC job's handles are the forward's persistent buffers (or
/// the test's outliving buffers) — they outlive every dispatch by
/// construction, so the borrowed pointer is sound without a retain dance.
fn extract_raw_buffer(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    handle: &Handle,
) -> Result<(*mut metal::MTLBuffer, u64), String> {
    // cuda_backend twin: the CubeCL resource is CUDA-shaped (no wgpu buffer
    // to extract a Metal handle from) — report unavailable instead of
    // failing to compile. `zc_context` propagates the Err and the ZC seam
    // falls back to the host-IO path (`zc_context_cached()` stays None).
    #[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
    {
        let _ = (client, handle);
        Err(
            "zero-copy Metal extraction unavailable: cuda_backend replaces the wgpu CubeCL runtime"
                .to_string(),
        )
    }
    #[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
    {
        let managed = client
            .get_resource(handle.clone())
            .map_err(|e| format!("get_resource failed: {e:?}"))?;
        let resource = managed.resource();
        let offset = resource.offset;
        let wgpu_buffer = &resource.buffer;
        let id_ptr = unsafe {
            let guard = wgpu_buffer.as_hal::<wgpu::hal::api::Metal>();
            let hal_buf = guard
                .as_ref()
                .ok_or("CubeCL buffer is not Metal — ZC requires the Metal backend")?;
            let proto = hal_buf.raw_handle();
            proto as *const _ as *mut metal::MTLBuffer
        };
        Ok((id_ptr, offset))
    }
}

// ── the worker payload + job ────────────────────────────────────────────────

/// Everything the worker needs — NO CubeCL objects cross the thread boundary
/// (raw borrowed pointers + the retained texture Arcs; the Phase C contract).
///
/// Pointer validity: the input/target pointers borrow the forward's
/// persistent CubeCL buffers, which outlive every dispatch by construction;
/// the texture Arc keeps the kernel's io-surface views alive.
struct ZcWorker {
    kernel: Arc<BridgeKernel>,
    ctx: Arc<ZcContext>,
    input_ptr: *mut metal::MTLBuffer,
    input_off: u64,
    /// `(target raw buffer, byte offset, fused-output channel offset, seg_dim)`
    targets: Vec<(*mut metal::MTLBuffer, u64, usize, usize)>,
    /// Probe 778: the io_out surface's own pages as a no-copy MTLBuffer —
    /// the BUFFER cacheability domain, created ONCE at dispatch begin on the
    /// main thread. The unpack reads through THIS (buffer path) instead of a
    /// texture view: buffer visibility proved robust where texture views
    /// (fresh or cached, any thread) read the stale post-eval backing in
    /// some dispatch sequences.
    out_nc_buffer: metal::Buffer,
    n: usize,
    w: usize,
    blocks: usize,
    /// Issue 769 T10: `Some(v)` = the second-queue variant is armed and the
    /// producer fence has ALREADY been signalled (value `v`) on the shared
    /// queue by `begin_*` on the main thread; the pack goes to
    /// `ctx.ane_queue` behind `wait_for_event(fence, v)`. `None` = the
    /// shared-queue path (default).
    fence_value: Option<u64>,
}

// SAFETY: metal ObjC objects (MTLBuffer/MTLTexture/pipelines/queue) are
// thread-safe per Metal's API contract; BridgeKernel is Send+Sync (P4).
unsafe impl Send for ZcWorker {}

fn encode_pack(
    encoder: &metal::ComputeCommandEncoderRef,
    ctx: &ZcContext,
    textures: &ZcTextures,
    input: &BufferRef,
    input_off: u64,
    n: usize,
    w: usize,
    block: usize,
) {
    let total_u32 = (n * w / 2) as u32;
    let params = ZcPackParams {
        n: n as u32,
        w: w as u32,
        block: block as u32,
        tex_w: textures.w_in as u32,
        total_u32,
    };
    encoder.set_compute_pipeline_state(&ctx.pack);
    encoder.set_buffer(0, Some(input), input_off);
    encoder.set_texture(0, Some(&textures.tex_in));
    encoder.set_bytes(1, std::mem::size_of::<ZcPackParams>() as u64, &params as *const _ as *const _);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(&textures.tex_in, metal::MTLResourceUsage::Write);
    encoder.dispatch_threads(
        MTLSize::new(total_u32 as u64, 1, 1),
        MTLSize::new(64, 1, 1),
    );
}

fn encode_unpack(
    encoder: &metal::ComputeCommandEncoderRef,
    ctx: &ZcContext,
    textures: &ZcTextures,
    target: &BufferRef,
    target_off: u64,
    w: usize,
    chan_off: usize,
    seg_dim: usize,
) {
    let total = (w * seg_dim) as u32;
    let params = ZcUnpackParams {
        seg_dim: seg_dim as u32,
        w: w as u32,
        chan_off: chan_off as u32,
        tex_w: textures.w_out as u32,
        total,
    };
    encoder.set_compute_pipeline_state(&ctx.unpack);
    encoder.set_texture(0, Some(&textures.tex_out));
    encoder.set_buffer(0, Some(target), target_off);
    encoder.set_bytes(1, std::mem::size_of::<ZcUnpackParams>() as u64, &params as *const _ as *const _);
    encoder.use_resource(&textures.tex_out, metal::MTLResourceUsage::Read);
    encoder.use_resource(target, metal::MTLResourceUsage::Write);
    encoder.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(64, 1, 1));
}

fn zc_worker_loop(worker: ZcWorker) -> Result<(), String> {
    let ZcWorker {
        kernel,
        ctx,
        input_ptr,
        input_off,
        targets,
        out_nc_buffer,
        n,
        w,
        blocks,
        fence_value,
    } = worker;
    let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
    let target_refs: Vec<(&BufferRef, u64, usize, usize)> = targets
        .iter()
        .map(|&(p, off, c, d)| unsafe { (BufferRef::from_ptr(p), off, c, d) })
        .collect();
    let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(ctx.queue) };
    // Probe 778: the unpack reads through the NO-COPY BUFFER (the buffer
    // cacheability domain) — see the ZcWorker field doc. The PACK keeps the
    // fresh per-block texture views (that direction is proven through any
    // view).
    let fresh = |kernel: &BridgeKernel| -> Result<std::sync::Arc<ZcTextures>, String> {
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(ctx.device) };
        kernel.metal_textures_fresh(device_ref)
    };
    let nc_ref: &BufferRef =
        unsafe { BufferRef::from_ptr(out_nc_buffer.as_ptr() as *mut _) };
    let probe = drain_probe_armed();
    // Issue 769 T10: the pack's queue. The dedicated one when armed - its
    // `wait_until_completed()` then drains only work on THAT queue (our own
    // packs) plus the fenced producer, not the caller's complement, which
    // lands on the shared queue after `begin_*` returned.
    let pack_queue: &CommandQueueRef = match fence_value {
        Some(_) => &ctx.ane_queue,
        None => queue_ref,
    };
    for block in 0..blocks {
        let tex = fresh(&kernel)?;
        // 0. Probe P1 (Issue 769 T10, Bench 846 §9; armed only): drain the
        //    shared queue with an EMPTY command buffer FIRST. Nothing is
        //    encoded into it, so its wait is pure backlog — and the pack's
        //    own wait below then runs against an empty queue, isolating pack
        //    execution. Semantically inert: an empty buffer has no encoder
        //    and touches no resource, so ordering and results are unchanged.
        if probe {
            let t = Instant::now();
            let drain = queue_ref.new_command_buffer();
            drain.commit();
            drain.wait_until_completed();
            bump(&T_ZC_DRAIN_NS, t);
            N_ZC_DRAINS.fetch_add(1, Ordering::Relaxed);
        }
        // 1. Pack (enqueue + wait — the eval needs the surface bytes).
        let t = Instant::now();
        let cmd = pack_queue.new_command_buffer();
        // The producer fence, cross-queue. Encoded on EVERY block: the wait
        // is satisfied for good once the event passes `v`, so blocks 1..n
        // pay only the encode (the producer is the same normx for all
        // blocks). Skipped entirely on the shared-queue path, where FIFO
        // already orders us after the producer.
        if let Some(v) = fence_value {
            cmd.encode_wait_for_event(&ctx.fence, v);
            N_ZC_2Q_PACKS.fetch_add(1, Ordering::Relaxed);
        }
        {
            let enc = cmd.new_compute_command_encoder();
            encode_pack(enc, &ctx, &tex, input_ref, input_off, n, w, block);
            enc.end_encoding();
        }
        cmd.commit();
        bump(&T_ZC_PACK_NS, t);
        let t = Instant::now();
        cmd.wait_until_completed();
        bump(&T_ZC_WAIT_NS, t);
        record_pack_exec(cmd);
        // 2. ANE eval (no host memcpy — IO lives in the surfaces).
        let t = Instant::now();
        if let Err(e) = kernel.eval_nc() {
            bump(&T_ZC_EVAL_NS, t);
            N_ZC_FAILS.fetch_add(1, Ordering::Relaxed);
            return Err(format!("ANE zc eval failed (block {block}): {e}"));
        }
        bump(&T_ZC_EVAL_NS, t);
        N_ZC_EVALS.fetch_add(1, Ordering::Relaxed);
        // 2.5. THE SETTLE (Probe 778): `evaluateWithQoS` returns before the
        //      daemon's io_out writeback is guaranteed visible to GPU reads
        //      (measured: the surface holds correct bytes via the host lock
        //      path while an immediately-enqueued GPU unpack reads stale
        //      zeros when the queue is idle). One 1-element host lock+read
        //      arbitrates the surface with the writer (the gpin path has
        //      always relied on exactly this through its full stage(1)).
        let mut settle = [0u16; 1];
        kernel.stage(1, &mut settle);
        // 3. Unpack every segment through the NO-COPY BUFFER (enqueue only —
        //    the consumer is ordered by queue FIFO; targets must simply
        //    stay alive, which they do).
        let t = Instant::now();
        let cmd2 = queue_ref.new_command_buffer();
        {
            let enc = cmd2.new_compute_command_encoder();
            for (buf, off, chan_off, seg_dim) in &target_refs {
                // The block's slice starts at block * w * seg_dim elements.
                let slice_off = off + (block * w * seg_dim * 4) as u64;
                let total = (w * seg_dim) as u32;
                let params = ZcUnpackParams {
                    seg_dim: *seg_dim as u32,
                    w: w as u32,
                    chan_off: *chan_off as u32,
                    tex_w: 0, // unused by the buffer-path kernel
                    total,
                };
                enc.set_compute_pipeline_state(&ctx.unpack_buf);
                enc.set_buffer(0, Some(nc_ref), 0);
                enc.set_buffer(1, Some(buf), slice_off);
                enc.set_bytes(
                    2,
                    std::mem::size_of::<ZcUnpackParams>() as u64,
                    &params as *const _ as *const _,
                );
                enc.use_resource(nc_ref, metal::MTLResourceUsage::Read);
                enc.use_resource(buf, metal::MTLResourceUsage::Write);
                enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(64, 1, 1));
            }
            enc.end_encoding();
        }
        cmd2.commit();
        bump(&T_ZC_UNPACK_NS, t);
    }
    Ok(())
}

/// An in-flight zero-copy split dispatch. Created by
/// [`begin_split_overlapped_zc`]; complete with [`Self::finish`].
pub struct AneSplitJobZc {
    handle: Option<std::thread::JoinHandle<Result<(), String>>>,
}

/// Begin a zero-copy split-overlap dispatch: extract the raw buffers on the
/// MAIN thread, create the kernel's texture views (once per kernel), spawn
/// the worker. The caller submits the GPU complement NOW and then calls
/// [`AneSplitJobZc::finish`].
#[allow(clippy::too_many_arguments)]
pub fn begin_split_overlapped_zc(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    ctx: Arc<ZcContext>,
    kernel: Arc<BridgeKernel>,
    input: &Handle,
    p: usize,
    n: usize,
    _oc_total: usize,
    w: usize,
    segments: &[(Handle, usize, usize)],
) -> Result<AneSplitJobZc, String> {
    debug_assert!(p.is_multiple_of(w), "zc split requires exact blocks");
    debug_assert_eq!((n * w) % 2, 0, "zc pack requires an even fp16 count");
    // Probe 778: the unpack reads io_out through a NO-COPY BUFFER over the
    // surface's own pages (the buffer cacheability domain), created ONCE on
    // the main thread. Texture-view reads of the post-eval backing proved
    // sequence-fragile (fresh or cached, either thread); the buffer path
    // reads the live surface bytes unconditionally (Probe 778 C).
    let out_nc_buffer = kernel.out_no_copy_buffer(unsafe {
        DeviceRef::from_ptr(ctx.device)
    })?;
    // Issue 769 T10 - arm the producer fence BEFORE the caller's complement
    // exists, which is what makes the dedicated queue mean anything:
    //
    //   1. `client.flush()` submits the input producer to the SHARED queue.
    //      Without it the producer may still sit in CubeCL's server channel
    //      and the fence would signal ahead of it (the documented
    //      `gemm_ternary_metal_zero_copy` hazard: a raw commit enqueued
    //      ahead of its producer reads stale input). It submits, it does
    //      not wait.
    //   2. One signal-only command buffer on the shared queue. Shared-queue
    //      FIFO makes it complete after the producer; nothing the caller
    //      submits after we return can precede it.
    //   3. Every pack then waits on that value from the dedicated queue -
    //      so the pack's dependency set is exactly "the producer and what
    //      was already ahead of it", never the complement.
    let two_q = crate::ane_prefill::prefill_ane_zc_second_queue();
    // P4: the flush is armable ALONE so the second queue's delta can be
    // attributed. The fence implies it (correctness); the reverse does not.
    let want_flush = two_q || crate::ane_prefill::prefill_ane_zc_producer_flush();
    let fence_value = if !want_flush { None } else {
            let t = Instant::now();
            let _ = client.flush();
            let v = if !two_q { None } else {
                    let v = ctx.next_fence();
                    let shared: &CommandQueueRef =
                        unsafe { CommandQueueRef::from_ptr(ctx.queue) };
                    let sig = shared.new_command_buffer();
                    sig.encode_signal_event(&ctx.fence, v);
                    sig.commit();
                    Some(v)
                };
            bump(&T_ZC_FENCE_NS, t);
            N_ZC_FENCES.fetch_add(1, Ordering::Relaxed);
            v
        };
    // Main-thread extraction — the worker never touches CubeCL. Texture
    // views are fetched FRESH per block inside the loop (Probe 778 A —
    // pre-eval views read the stale io_out backing).
    let (input_ptr, input_off) = extract_raw_buffer(client, input)?;
    let mut targets = Vec::with_capacity(segments.len());
    for (handle, chan_off, seg_dim) in segments {
        let (buf, off) = extract_raw_buffer(client, handle)?;
        targets.push((buf, off, *chan_off, *seg_dim));
    }
    let worker = ZcWorker {
        kernel,
        ctx,
        input_ptr,
        input_off,
        targets,
        out_nc_buffer,
        n,
        w,
        blocks: p / w,
        fence_value,
    };
    let join = std::thread::Builder::new()
        .name("ane-zc".into())
        .spawn(move || zc_worker_loop(worker))
        .map_err(|e| format!("ANE zc worker spawn failed: {e}"))?;
    Ok(AneSplitJobZc {
        handle: Some(join),
    })
}

impl AneSplitJobZc {
    /// Join the worker. NO queue fence (see the module docs) — the consumer
    /// kernels the caller submits afterwards are queue-ordered after the
    /// unpacks. On error, earlier blocks' unpacks may have landed; the
    /// fail-open GPU re-projection overwrites them (queue-ordered).
    pub fn finish(mut self) -> Result<(), String> {
        let join = self
            .handle
            .take()
            .expect("finish called twice on AneSplitJobZc");
        join.join()
            .map_err(|_| "ANE zc worker panicked".to_string())??;
        super::note_ane_dispatch();
        Ok(())
    }
}

impl Drop for AneSplitJobZc {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
            debug_assert!(false, "AneSplitJobZc dropped without finish()");
        }
    }
}

// ── Plan 550 v1 (measured reality): the GPU-IN hybrid ──────────────────────
//
// The isolation ladder (Bench 777 §debug) established the coherency map:
//
//   GPU-write →io_in→ ANE-read   ✓ WORKS  (bit-identical pack verified)
//   ANE-write →io_out→ GPU-read  ✗ BROKEN (zeros — the ANE's writes are
//                                 invisible to the GPU texture-fetch path,
//                                 visible to the CPU; not timing (sleep
//                                 ruled out), not mapping (CPU-write→GPU-read
//                                 on the SAME view works bit-correctly),
//                                 not lock-fence-able (a CPU lock+read cycle
//                                 between the ANE write and the GPU read
//                                 does not help). A driver-level cache-domain
//                                 asymmetry — the documented open lever.)
//
// v1 therefore ships the HALF that works: the GPU pack kernel replaces the
// 231-749 ms/op `read_one` readback + the 15.5 ms host pack (Bench 776's
// isolated 60-75% wall), while the output keeps the measured host path
// (lock+memcpy ≈10-20 ms + tiled unpack 31.5 ms + async write ≈0). The
// ANE→GPU output direction stays armed as the follow-up (its removal is a
// further ~40-50 ms/op — second-order vs the read wall).

/// The GPU-IN hybrid split job — the SAME shape + finish contract as the
/// host-IO [`super::exec::AneSplitJob`] (worker fills host staging;
/// `finish` writes via `client.write`), with the input readback + pack
/// replaced by the zero-copy GPU pack kernel.
pub struct AneSplitJobGpuIn {
    job: super::exec::AneSplitJob,
}

/// Module-scope worker payload (the ZcWorker precedent — local
/// `unsafe impl Send` inside a fn body was not picked up for the closure
/// capture inference).
struct GpuInPayload {
    kernel: Arc<BridgeKernel>,
    ctx: Arc<ZcContext>,
    textures: Arc<ZcTextures>,
    input_ptr: *mut metal::MTLBuffer,
    input_off: u64,
    n: usize,
    w: usize,
    blocks: usize,
    oc_total: usize,
    seg_meta: Vec<(usize, usize)>,
    seg_dims: Vec<usize>,
    queue: *mut metal::MTLCommandQueue,
}

// SAFETY: raw ObjC pointers (MTLBuffer/MTLCommandQueue — thread-safe per
// Metal's contract) + Send+Sync Arcs (the ZcWorker precedent).
unsafe impl Send for GpuInPayload {}

/// Begin a GPU-IN hybrid split dispatch. Requires the split flags + the ZC
/// context; the caller submits the GPU complement between this and
/// [`AneSplitJobGpuIn::finish`] exactly as for the host-IO job.
#[allow(clippy::too_many_arguments)]
pub fn begin_split_gpu_in(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    ctx: Arc<ZcContext>,
    kernel: Arc<BridgeKernel>,
    input: &Handle,
    p: usize,
    n: usize,
    oc_total: usize,
    w: usize,
    segments: &[(Handle, usize, usize)],
) -> Result<AneSplitJobGpuIn, String> {
    debug_assert!(p.is_multiple_of(w), "gpu-in split requires exact blocks");
    let queue_ptr = ctx.queue;
    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(ctx.device) };
    let textures = kernel.metal_textures(device_ref)?;
    let (input_ptr, input_off) = extract_raw_buffer(client, input)?;
    let blocks = p / w;
    let seg_meta: Vec<(usize, usize)> = segments.iter().map(|&(_, o, d)| (o, d)).collect();
    let seg_dims: Vec<usize> = seg_meta.iter().map(|&(_, d)| d).collect();
    // SAFETY: raw ObjC pointers (MTLBuffer/queue/textures) + the Send+Sync
    // kernel + context — all thread-safe per Metal's contract (the ZcWorker
    // precedent).
    let payload = GpuInPayload {
        kernel,
        ctx,
        textures,
        input_ptr,
        input_off,
        n,
        w,
        blocks,
        oc_total,
        seg_meta,
        seg_dims,
        queue: queue_ptr,
    };
    // The pack-0 handshake: begin must not return until the worker has
    // COMMITTED block-0's pack — otherwise the caller's complement can land
    // on the queue FIRST and the pack (whose only dependency is the input
    // producer) needlessly waits behind the complement (+a full GEMM of
    // serial time per op at blocks==1).
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let join = std::thread::Builder::new()
        .name("ane-gpu-in".into())
        .spawn(move || {
            // Rebind to force whole-struct capture (RFC 2229 disjoint
            // capture would otherwise capture the raw-pointer FIELDS
            // individually, bypassing the Send impl).
            let payload = payload;
            let GpuInPayload {
                kernel,
                ctx,
                textures,
                input_ptr,
                input_off,
                n,
                w,
                blocks,
                oc_total,
                seg_meta,
                seg_dims,
                queue,
            } = payload;
            let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
            let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(queue) };
            let mut y16 = vec![0u16; oc_total * w];
            let mut staging: Vec<Vec<f32>> = seg_dims
                .iter()
                .map(|&d| vec![0f32; blocks * w * d])
                .collect();
            for block in 0..blocks {
                // 1. GPU pack (zero-copy: reads the CubeCL input directly,
                //    writes the ANE's io surface — NO read_one, NO host pack).
                let t = Instant::now();
                let cmd = queue_ref.new_command_buffer();
                {
                    let enc = cmd.new_compute_command_encoder();
                    encode_pack(enc, &ctx, &textures, input_ref, input_off, n, w, block);
                    enc.end_encoding();
                }
                cmd.commit();
                bump(&T_ZC_PACK_NS, t);
                if block == 0 {
                    // Pack-0 is on the queue — the caller may now submit the
                    // complement (guaranteed to land AFTER the pack).
                    let _ = tx.send(());
                }
                let t = Instant::now();
                cmd.wait_until_completed();
                bump(&T_ZC_WAIT_NS, t);
                record_pack_exec(cmd);
                // 2. ANE eval (no input memcpy — the surface holds the bytes).
                let t = Instant::now();
                if let Err(e) = kernel.eval_nc() {
                    bump(&T_ZC_EVAL_NS, t);
                    super::exec::N_EVAL_FAILS.fetch_add(1, Ordering::Relaxed);
                    return Err(format!("ANE gpu-in eval failed (block {block}): {e}"));
                }
                bump(&T_ZC_EVAL_NS, t);
                super::exec::N_EVALS.fetch_add(1, Ordering::Relaxed);
                N_ZC_EVALS.fetch_add(1, Ordering::Relaxed);
                // 3. Host read-out (lock+memcpy — the ANE→GPU direction is the
                //    documented broken link) + tiled unpack into staging.
                let t = Instant::now();
                kernel.stage(1, &mut y16);
                bump(&T_ZC_UNPACK_NS, t);
                let t = Instant::now();
                for (si, &(chan_off, seg_dim)) in seg_meta.iter().enumerate() {
                    let out = &mut staging[si][block * w * seg_dim..(block + 1) * w * seg_dim];
                    super::exec::unpack_segment_block_tiled(&y16, chan_off, seg_dim, w, out);
                }
                bump(&super::exec::T_UNPACK_NS, t);
            }
            Ok(staging)
        })
        .map_err(|e| format!("ANE gpu-in worker spawn failed: {e}"))?;
    // Block until pack-0 is committed (see the handshake comment above). The
    // channel closes when the worker exits early on error — a disconnected
    // receive here means the worker already failed; surface it via finish().
    if rx.recv().is_err() {
        // The worker died before committing pack-0 (texture/spawn-path
        // failure inside the loop's first iteration) — join it for the error.
        let e = join
            .join()
            .ok()
            .and_then(|r| r.err())
            .unwrap_or_else(|| "ANE gpu-in worker exited before pack-0".into());
        return Err(e);
    }
    // Reuse the host-IO job's finish machinery (join + per-block
    // client.write) — the staging contract is identical.
    let job = super::exec::AneSplitJob::from_parts(join, segments.to_vec(), w, blocks);
    Ok(AneSplitJobGpuIn { job })
}

impl AneSplitJobGpuIn {
    /// See [`super::exec::AneSplitJob::finish`].
    pub fn finish<R: cubecl::Runtime>(
        self,
        client: &ComputeClient<R>,
    ) -> Result<(), String> {
        self.job.finish(client)
    }
}

// ── debug probes (test-only; the pack-kernel isolation ladder) ─────────────

#[doc(hidden)]
pub fn debug_unpack_probe(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    ctx: &ZcContext,
    kernel: &BridgeKernel,
    target: &Handle,
    w: usize,
    chan_off: usize,
    seg_dim: usize,
    block: usize,
    gpu_prefill: bool,
) -> Result<(), String> {
    use metal::foreign_types::ForeignTypeRef;
    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(ctx.device) };
    let textures = kernel.metal_textures(device_ref)?;
    let (target_ptr, target_off) = extract_raw_buffer(client, target)?;
    let target_ref: &BufferRef = unsafe { BufferRef::from_ptr(target_ptr) };
    let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(ctx.queue) };
    let cmd = queue_ref.new_command_buffer();
    {
        let enc = cmd.new_compute_command_encoder();
        if gpu_prefill {
            // The round-trip probe: GPU-fill tex_out FIRST, then unpack —
            // separates read-path breakage from ANE-write visibility.
            let geom = [
                textures.w_out as u32,
                textures.h_out as u32,
            ];
            let val: u32 = 0x3C00_3C00; // two fp16 1.0s
            enc.set_compute_pipeline_state(&ctx.fill);
            enc.set_texture(0, Some(&textures.tex_out));
            enc.set_bytes(
                0,
                std::mem::size_of::<[u32; 2]>() as u64,
                geom.as_ptr() as *const _,
            );
            enc.set_bytes(1, 4, &val as *const u32 as *const _);
            enc.use_resource(&textures.tex_out, metal::MTLResourceUsage::Write);
            enc.dispatch_threads(
                MTLSize::new(textures.w_out, textures.h_out, 1),
                MTLSize::new(64, 1, 1),
            );
        }
        let slice_off = target_off + (block * w * seg_dim * 4) as u64;
        encode_unpack(enc, ctx, &textures, target_ref, slice_off, w, chan_off, seg_dim);
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

#[doc(hidden)]
pub fn debug_pack_probe(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    ctx: &ZcContext,
    kernel: &BridgeKernel,
    input: &Handle,
    n: usize,
    w: usize,
    block: usize,
) -> Result<Vec<u16>, String> {
    use metal::foreign_types::ForeignTypeRef;
    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(ctx.device) };
    let textures = kernel.metal_textures(device_ref)?;
    let (input_ptr, input_off) = extract_raw_buffer(client, input)?;
    let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
    let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(ctx.queue) };
    let cmd = queue_ref.new_command_buffer();
    {
        let enc = cmd.new_compute_command_encoder();
        encode_pack(enc, ctx, &textures, input_ref, input_off, n, w, block);
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    let mut out = vec![0u16; n * w];
    kernel.stage(2, &mut out);
    Ok(out)
}

// ── Probe 778: the ANE→GPU output-coherency probes ────────────────────────
//
// Bench 777's isolation ladder proved the ANE-write→GPU-texture-read
// direction reads ZEROS (visible to the CPU, invisible to the GPU texture
// path; not timing, not mapping, not lock-fence-able). These probes test
// the three REMAINING mechanisms, each a different GPU cache/coherency
// domain:
//
// - FRESH TEXTURES — recreate the MTLTexture views after the eval. If the
//   driver swaps the surface's backing store at eval completion, a texture
//   created before the eval is pinned to the stale backing.
// - BLIT COPY — the copy engine (DMA) reading the texture, landing the
//   bytes in a plain MTLBuffer. A different fetch path than shader reads.
// - NO-COPY BUFFER — `newBufferWithBytesNoCopy` over the surface's own
//   pages: the BUFFER cacheability domain, never exercised by the ladder.

/// Fresh (uncached) texture views for one kernel — probe-only.
#[doc(hidden)]
pub fn debug_fresh_textures(
    ctx: &ZcContext,
    kernel: &BridgeKernel,
) -> Result<std::sync::Arc<ZcTextures>, String> {
    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(ctx.device) };
    kernel.metal_textures_fresh(device_ref)
}

/// The unpack kernel over CALLER-SUPPLIED textures (probe variant of
/// [`debug_unpack_probe`] — same dispatch, no texture cache lookup).
#[doc(hidden)]
pub fn debug_unpack_probe_tex(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    ctx: &ZcContext,
    textures: &ZcTextures,
    target: &Handle,
    w: usize,
    chan_off: usize,
    seg_dim: usize,
    block: usize,
) -> Result<(), String> {
    use metal::foreign_types::ForeignTypeRef;
    let (target_ptr, target_off) = extract_raw_buffer(client, target)?;
    let target_ref: &BufferRef = unsafe { BufferRef::from_ptr(target_ptr) };
    let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(ctx.queue) };
    let cmd = queue_ref.new_command_buffer();
    {
        let enc = cmd.new_compute_command_encoder();
        let slice_off = target_off + (block * w * seg_dim * 4) as u64;
        encode_unpack(enc, ctx, textures, target_ref, slice_off, w, chan_off, seg_dim);
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

/// BLIT-path read: copy the whole io_out texture into a fresh shared
/// MTLBuffer with the copy engine and return the raw bytes. The blit is a
/// different hardware path than the shader texture-fetch that Bench 777
/// measured as broken.
#[doc(hidden)]
pub fn debug_blit_probe(ctx: &ZcContext, textures: &ZcTextures) -> Result<Vec<u8>, String> {
    use metal::foreign_types::ForeignTypeRef;
    let bytes = textures.w_out * textures.h_out * 4;
    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(ctx.device) };
    let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(ctx.queue) };
    let buf = device_ref.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let cmd = queue_ref.new_command_buffer();
    {
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_texture_to_buffer(
            &textures.tex_out,
            0, // source slice
            0, // source mip level
            MTLOrigin {
                x: 0,
                y: 0,
                z: 0,
            },
            MTLSize::new(textures.w_out, textures.h_out, 1),
            &buf,
            0,                  // destination offset
            textures.w_out * 4, // destination bytes per row
            0,                  // destination bytes per image
            metal::MTLBlitOption::empty(),
        );
        blit.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    let base = buf.contents();
    let out = unsafe { std::slice::from_raw_parts(base as *const u8, bytes as usize) }.to_vec();
    Ok(out)
}

/// BUFFER-path read: wrap the io_out surface's own pages in a no-copy
/// MTLBuffer and run the [`zc_unpack_buf`] kernel (identical indexing to
/// [`encode_unpack`]) into a CubeCL target. The buffer cacheability domain
/// was never exercised by the Bench 777 ladder.
#[doc(hidden)]
pub fn debug_unpack_buf_probe(
    client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    ctx: &ZcContext,
    kernel: &BridgeKernel,
    target: &Handle,
    w: usize,
    chan_off: usize,
    seg_dim: usize,
    block: usize,
) -> Result<(), String> {
    use metal::foreign_types::{ForeignType, ForeignTypeRef};
    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(ctx.device) };
    let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(ctx.queue) };
    let nc = kernel.out_no_copy_buffer(device_ref)?;
    let nc_ref: &BufferRef = unsafe { BufferRef::from_ptr(nc.as_ptr()) };
    let (target_ptr, target_off) = extract_raw_buffer(client, target)?;
    let target_ref: &BufferRef = unsafe { BufferRef::from_ptr(target_ptr) };
    let total = (w * seg_dim) as u32;
    // tex_w is unused by the buffer-path kernel; the params struct carries
    // it for shape parity with the texture path.
    let params = ZcUnpackParams {
        seg_dim: seg_dim as u32,
        w: w as u32,
        chan_off: chan_off as u32,
        tex_w: 0,
        total,
    };
    let cmd = queue_ref.new_command_buffer();
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.unpack_buf);
        enc.set_buffer(0, Some(nc_ref), 0);
        enc.set_buffer(1, Some(target_ref), target_off + (block * w * seg_dim * 4) as u64);
        enc.set_bytes(2, std::mem::size_of::<ZcUnpackParams>() as u64, &params as *const _ as *const _);
        enc.use_resource(nc_ref, metal::MTLResourceUsage::Read);
        enc.use_resource(target_ref, metal::MTLResourceUsage::Write);
        enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}
