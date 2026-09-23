use std::future::Future;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll as TaskPoll, Waker};

use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use super::context::GpuError;

/// Result slot for a `map_async` callback, readable after the callback runs.
///
/// **This was an `UnsafeCell` with a hand-written `unsafe impl Sync` until
/// 2026-09-04, and both the safety argument and the "impossible" case below
/// were refuted by measurement** (riir-train `.issues/511`, the first
/// full-workspace `--all-features` execution). The old doc comment read:
/// "Since `device.poll(Wait)` blocks until the callback fires, the callback
/// writes exactly once before `poll` returns", justifying `unsafe impl Sync`
/// as "a single-producer-single-consumer pattern where `device.poll(Wait)`
/// provides the synchronization barrier".
///
/// Neither holds on a **process-shared** device, which is what
/// `GpuContext::new()` has handed every caller since Issue 714 made it a
/// `OnceLock`:
///
/// 1. `poll` drains the callback queue, so a CONCURRENT thread's poll can be
///    the one that invokes *our* callback — the producer is then a thread
///    that our own `poll(Wait)` never synchronized with. That is a data race
///    on the cell, not a barrier.
/// 2. Consequently our `poll` can return before our callback has run. The
///    old code called that case "impossible with `Wait` and no timeout" and
///    reached for `(*arc.inner.get()).take().unwrap()`, which **panics**.
///    Measured: `test_set_causal_backward_loss_decreases` and
///    `_training_converges` both panicked at exactly that `unwrap` under
///    `--all-features`, and 8 more tests in one dllm target died on the
///    downstream "staging download buffer is invalid" that follows.
///
/// A `Mutex` fixes both at a cost that does not matter here: an uncontended
/// lock is tens of nanoseconds against a staging round-trip that blocks on
/// the GPU for orders of magnitude longer. `unsafe` is gone entirely.
///
/// `pub`, and the API is deliberately unchanged, because riir-train-gpu
/// constructs this type directly in three places (Issue 741 T3:
/// `optimizer.rs`, `optimizer_amuse`, `optimizer_cm_lora`) — those callers
/// keep compiling and become race-free for free.
pub struct SyncMapResult {
    inner: Mutex<Option<Result<(), wgpu::BufferAsyncError>>>,
}

impl SyncMapResult {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    pub fn callback(
        self: &Arc<Self>,
    ) -> impl FnOnce(Result<(), wgpu::BufferAsyncError>) + 'static {
        let this = self.clone();
        move |res| {
            // A poisoned lock here would mean a previous callback panicked
            // while holding it, which cannot happen: the critical section is
            // a single assignment. Recover rather than propagate a panic out
            // of a wgpu callback, where it would unwind through FFI.
            let mut slot = this.inner.lock().unwrap_or_else(|e| e.into_inner());
            *slot = Some(res);
        }
    }

    /// Take the result if the callback has already run.
    ///
    /// `None` means "not yet", which is a retryable state and NOT an error —
    /// see [`await_map_result`].
    fn try_take(&self) -> Option<Result<(), wgpu::BufferAsyncError>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Kept for the three riir-train-gpu callers that own the `Arc` and
    /// unwrap it themselves. Prefer [`await_map_result`] in new code: this
    /// returns `None` for "the callback has not run yet" and so cannot tell
    /// that apart from a real failure, which is the bug this whole comment
    /// is about.
    pub fn into_result(self) -> Option<Result<(), wgpu::BufferAsyncError>> {
        self.inner.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for SyncMapResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Blocking poll-for-map-completion, with the poll result PROPAGATED
/// (Issue 679 B10).
///
/// The 5 download helpers previously discarded `device.poll`'s result with
/// `let _ =`; if poll errors without firing the map callback (e.g. device
/// lost), the fallback `take().unwrap()` panicked far from the cause. This
/// helper returns the poll error as a `GpuError` instead, keeping the
/// "callback-not-called" path for the (impossible-with-`Wait`-and-no-timeout)
/// case where poll succeeds but the callback never ran.
fn poll_wait(device: &wgpu::Device) -> Result<(), GpuError> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| GpuError::BufferError(format!("device poll failed: {e}")))?;
    Ok(())
}

/// How many times to re-poll while waiting for OUR OWN map callback.
///
/// `device.poll(Wait { submission_index: None })` waits for the device's
/// LAST submission and drains the callback queue. On a process-shared device
/// (Issue 714) neither of those is a guarantee about *our* callback: a
/// concurrent thread can submit after us so `None` names its work, and a
/// concurrent poll can be the one that drains ours. So a single poll is a
/// likely-but-not-certain completion, which is precisely what the previous
/// code asserted when it wrote `.take().unwrap()` under a comment calling
/// the alternative impossible.
///
/// 8 is chosen to be generous rather than tuned: every iteration after the
/// first is a `poll(Wait)` on an already-drained queue, so the loop costs
/// nothing in the overwhelmingly common case where the first poll suffices.
/// A bound rather than a spin because an unbounded retry would turn a real
/// device loss into a hang.
///
/// **"Costs nothing" is exactly why a COUNT is the wrong budget** (riir-train
/// `.issues/511`: `goat_dflare_full_system` hit the error at train_batch step
/// 130). A `poll(Wait)` on a drained queue returns immediately, so eight of
/// them are eight instants — the loop is not a wait at all, it is eight
/// consecutive glances. When our callback is genuinely still in flight behind
/// another thread's submission, waiting is the only thing that can help, and
/// this loop did none of it. The count is kept as a FLOOR (cheap glances
/// first, for the common case) and a wall-clock deadline is what actually
/// bounds the retry.
const MAP_POLL_ATTEMPTS: usize = 8;

/// How long to keep re-polling after the cheap attempts are spent.
///
/// Bounds the retry in the unit that matters. Chosen to be far longer than
/// any plausible submission on this hardware and far shorter than a hang; a
/// real device loss does not reach the deadline at all, because `poll_wait`
/// returns `Err` on the first poll.
const MAP_POLL_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Pause between polls once the cheap attempts are spent, so a five-second
/// deadline is not five seconds of a spinning core.
const MAP_POLL_BACKOFF: std::time::Duration = std::time::Duration::from_millis(1);

/// Block until `result`'s `map_async` callback has run, then return its
/// outcome.
///
/// Replaces five byte-similar copies of a poll-once-then-unwrap block, each
/// of which could panic. The three states are now distinct: mapped OK, the
/// map failed (a `wgpu::BufferAsyncError`), or the callback never ran within
/// the poll budget — and the last one is an `Err` a caller can act on rather
/// than an unwind out of a library.
///
/// **`pub` because the copies are not all in this crate.** riir-train-gpu
/// carried four more — three optimizers plus a hand-rolled duplicate of
/// `SyncMapResult` in `compress.rs` — and every one of them still spelled
/// the sync as `let _ = device.poll(...)`, i.e. the discarded poll result
/// that Issue 679 B10 fixed *here* and never propagated *there*. A fix that
/// cannot be imported gets re-implemented with the old bug, so this is the
/// import.
pub fn await_map_result(
    device: &wgpu::Device,
    result: &Arc<SyncMapResult>,
) -> Result<(), GpuError> {
    let started = std::time::Instant::now();
    let mut attempts = 0usize;
    loop {
        poll_wait(device)?;
        attempts += 1;
        if let Some(outcome) = result.try_take() {
            return outcome.map_err(|e| GpuError::BufferError(e.to_string()));
        }
        let waited = started.elapsed();
        if attempts >= MAP_POLL_ATTEMPTS && waited >= MAP_POLL_DEADLINE {
            return Err(GpuError::BufferError(format!(
                "map callback not called after {attempts} device polls over \
                 {waited:.1?} (deadline {MAP_POLL_DEADLINE:.1?})"
            )));
        }
        if attempts >= MAP_POLL_ATTEMPTS {
            std::thread::sleep(MAP_POLL_BACKOFF);
        }
    }
}

/// Errors captured by the three error scopes guarding one download phase
/// (riir-train `.issues/511`, the sixth probe / error-semantics
/// discriminator).
///
/// Empty on every healthy phase. Non-empty entries are `(filter, error)`
/// pairs — the ORIGINAL wgpu error, not the downstream symptom.
struct CapturedErrors(Vec<(&'static str, wgpu::Error)>);

impl CapturedErrors {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn text(&self) -> String {
        self.0
            .iter()
            .map(|(filter, err)| format!("[{filter}] {err}"))
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

/// Resolve a wgpu error-scope pop future without an executor.
///
/// wgpu stores scope errors SYNCHRONOUSLY at error-delivery time and its
/// `pop_error_scope` future is literally `ready(scope.error)`, so this
/// always resolves on the first poll. `Pending` is treated as "nothing
/// captured" rather than blocking — resolving is a diagnosis, never a wait.
fn resolve_scope_now<F: Future<Output = Option<wgpu::Error>>>(fut: F) -> Option<wgpu::Error> {
    let mut fut = std::pin::pin!(fut);
    match fut.as_mut().poll(&mut TaskContext::from_waker(Waker::noop())) {
        TaskPoll::Ready(captured) => captured,
        TaskPoll::Pending => None,
    }
}

/// Run one download phase (staging-buffer creation, or
/// encode+copy+submit+`map_async`) under all three wgpu error-scope
/// filters, returning the phase's value plus whatever the scopes captured
/// (riir-train `.issues/511`, the sixth probe).
///
/// # Why the download paths run under error scopes
///
/// wgpu delivers resource errors to the innermost matching error scope on
/// the CURRENT thread (scopes are thread-local with `std`), and an error
/// that matches NO scope reaches the default uncaptured handler, which
/// PANICS — that panic is the `wgpu_core.rs:2255` signature the issue-511
/// census recorded on the dllm/gdsd targets. Running the phases under
/// scopes converts that process-killing panic into an ordinary `Err` that
/// can CARRY the diagnosis, and captures the original creation/operation
/// error instead of its downstream symptom.
///
/// One error class is invisible to scopes BY DESIGN: `handle_error_inner`
/// returns early for `ErrorType::DeviceLost` ("will be surfaced via
/// callback"), so a buffer whose creation failed because the device was
/// lost comes back as a silent invalid error object — creation never
/// panics, and only later use dies. [`diagnose_download_failure`] turns
/// exactly that signature (invalid-at-use + clean scopes) into the named
/// device-lost verdict.
///
/// Cost on the healthy path is six thread-local scope-stack operations and
/// no allocation (the captured-`Vec` stays empty) against a staging
/// round-trip that blocks on the GPU — noise. Scopes are pushed and popped
/// in LIFO order around `phase` alone; if `phase` itself panics, the guards'
/// `Drop` pops them (wgpu's guards are unwind-safe).
fn under_error_scopes<T>(
    device: &wgpu::Device,
    phase: impl FnOnce() -> T,
) -> (T, CapturedErrors) {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let value = phase();
    // LIFO: `internal` was pushed last, so it pops first.
    let captured = [
        ("internal", resolve_scope_now(internal.pop())),
        ("out-of-memory", resolve_scope_now(out_of_memory.pop())),
        ("validation", resolve_scope_now(validation.pop())),
    ]
    .into_iter()
    .filter_map(|(filter, captured)| captured.map(|err| (filter, err)))
    .collect();
    (value, CapturedErrors(captured))
}

/// Attach the discriminated cause to a failed download (riir-train
/// `.issues/511` — the "three causes" discriminator).
///
/// `map_async` on a buffer reporting it `is invalid` has exactly three
/// causes in wgpu, and the evidence separates them:
///
/// 1. **Creation failed** (allocation failure, usage/size/limits
///    violation) — the buffer object is an error object. With the creation
///    scopes in hand (`creation` = `Some` with entries) this names the
///    ORIGINAL error; uncaptured, the same error would have panicked at
///    creation time with `Device::create_buffer` context.
/// 2. **Destroyed** between creation and use — surfaces as
///    `has been destroyed` in the operation error.
/// 3. **Device lost** — the only error class wgpu's sink DROPS by design,
///    so it is never in any scope. Invalid-at-use + clean scopes IS this
///    signature.
///
/// `creation` is `None` on the reuse paths, where the staging buffer was
/// allocated by an earlier `ensure_capacity` call whose scopes no longer
/// exist — the inference then applies to whichever buffer the operation
/// error names (its label is quoted verbatim in the message).
fn diagnose_download_failure(
    underlying: GpuError,
    creation: Option<&CapturedErrors>,
    op: &CapturedErrors,
) -> GpuError {
    let op_text = op.text();
    let creation_text = creation.map_or_else(
        || "n/a (reused staging — created by an earlier ensure_capacity)".to_string(),
        CapturedErrors::text,
    );

    let cause = match (creation, creation.is_some_and(|c| !c.is_empty())) {
        (Some(captured), true) => format!(
            "CAUSE (captured): staging-buffer CREATION failed — {}",
            captured.text()
        ),
        _ if op_text.contains("is invalid") => {
            let caveat = if creation.is_some() {
                ""
            } else {
                " (creation scopes unavailable on the reuse path — applies to \
                 whichever buffer the operation error names)"
            };
            format!(
                "CAUSE (inferred): DeviceLost-class creation error{caveat} — a buffer \
                 is an invalid error object at use while every scope is clean, and \
                 DeviceLost is the one error class wgpu's sink drops by design"
            )
        }
        _ if op_text.contains("has been destroyed") => {
            "CAUSE (captured): buffer DESTROYED before use".to_string()
        }
        _ if op_text.is_empty() => {
            "CAUSE (inferred): device lost mid-operation — no capturable error in \
             any scope and the map callback still failed"
                .to_string()
        }
        _ => format!("operation error (cause outside the modelled three): {op_text}"),
    };

    let op_line = if op_text.is_empty() {
        "clean".to_string()
    } else {
        op_text
    };
    GpuError::BufferError(format!(
        "{underlying}; diagnosis[{cause}; creation scopes: {creation_text}; \
         operation scopes: {op_line}]"
    ))
}

/// Reusable staging buffer for GPU→CPU downloads.
///
/// Avoids creating a new staging buffer allocation on every [`download_f32`] call.
/// Callers that perform repeated downloads (e.g., training loops) should create one
/// of these and pass it to [`download_f32_reuse`].
pub struct DownloadStaging {
    buffer: Option<wgpu::Buffer>,
    capacity: usize, // in f32 elements
}

impl DownloadStaging {
    /// Create an empty staging context (no pre-allocated buffer).
    pub fn new() -> Self {
        Self {
            buffer: None,
            capacity: 0,
        }
    }

    /// Ensure the staging buffer can hold at least `count` f32 elements.
    /// Reuses existing buffer if large enough; otherwise reallocates.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if self.capacity >= count && self.buffer.is_some() {
            return;
        }
        let bytes_needed = count * std::mem::size_of::<f32>();
        self.buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("reusable staging download"),
            size: bytes_needed as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.capacity = count;
    }

    /// Access the underlying staging buffer (after `ensure_capacity` has been called).
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }
}

impl Default for DownloadStaging {
    fn default() -> Self {
        Self::new()
    }
}

/// Upload chunk size for large f32 weight buffers (Issue 714).
///
/// `create_buffer_init`'s mapped-at-creation path stages the FULL buffer size
/// for device-local memory (a second same-size allocation — for a 2.36 GB wte
/// that is a 2.36 GB transient that pushed three stacked full-model tests to
/// 23.8 GB on the 24 GB 4090 and OOM'd `forward_cubecl`). `queue.write_buffer`
/// chunks through wgpu's small internal staging path instead. 64 MiB is a
/// multiple of COPY_BUFFER_ALIGNMENT (4) and large enough to amortize the
/// per-write overhead.
const UPLOAD_CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// Upload f32 data to a GPU buffer.
///
/// Chunked via `queue.write_buffer` (see [`UPLOAD_CHUNK_BYTES`]) — avoids the
/// full-size mapped-at-creation staging transient for multi-GB weight uploads.
/// Every chunk boundary is 4-byte aligned by construction (f32 slice + a
/// 4-multiple chunk size), satisfying COPY_BUFFER_ALIGNMENT.
pub fn upload_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    label: &str,
) -> Buffer {
    let bytes = bytemuck::cast_slice(data);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for (offset, chunk) in bytes.chunks(UPLOAD_CHUNK_BYTES).enumerate() {
        queue.write_buffer(
            &buffer,
            (offset * UPLOAD_CHUNK_BYTES) as u64,
            chunk,
        );
    }
    buffer
}

/// Upload raw bytes to a GPU buffer.
///
/// Used for uploading packed quantized weights (e.g., Q4_K blocks as `array<u32>` in WGSL).
pub fn upload_bytes(
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    data: &[u8],
    label: &str,
) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    })
}

/// Download f32 data from a GPU buffer (blocking).
pub fn download_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &Buffer,
    count: usize,
) -> Result<Vec<f32>, GpuError> {
    let bytes_needed = count * std::mem::size_of::<f32>();

    let (staging_buffer, creation_errs) = under_error_scopes(device, || {
        device.create_buffer(&BufferDescriptor {
            label: Some("staging download buffer"),
            size: bytes_needed as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });

    let result = Arc::new(SyncMapResult::new());

    let (buffer_slice, op_errs) = under_error_scopes(device, || {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download encoder"),
        });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, bytes_needed as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, result.callback());
        buffer_slice
    });

    if let Err(e) = await_map_result(device, &result) {
        return Err(diagnose_download_failure(e, Some(&creation_errs), &op_errs));
    }

    let data = buffer_slice.get_mapped_range()
        .map_err(|e| GpuError::BufferError(e.to_string()))?;
    let output: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Ok(output)
}

/// Download f32 data from a GPU buffer, reusing a pre-allocated staging buffer.
///
/// Use this instead of [`download_f32`] in hot loops to avoid repeated
/// staging buffer allocation. The staging buffer grows as needed but never shrinks.
pub fn download_f32_reuse(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &Buffer,
    count: usize,
    staging: &mut DownloadStaging,
) -> Result<Vec<f32>, GpuError> {
    let mut out = Vec::with_capacity(count);
    download_f32_reuse_into(device, queue, src, count, staging, &mut out)?;
    Ok(out)
}

/// Download f32 data into a pre-allocated output buffer, avoiding both staging
/// and output Vec allocation. The output buffer is cleared and resized as needed.
pub fn download_f32_reuse_into(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &Buffer,
    count: usize,
    staging: &mut DownloadStaging,
    output: &mut Vec<f32>,
) -> Result<(), GpuError> {
    let bytes_needed = count * std::mem::size_of::<f32>();

    // Reuse or create staging buffer
    staging.ensure_capacity(device, count);
    let staging_buf = staging.buffer.as_ref().unwrap();

    let result = Arc::new(SyncMapResult::new());

    let (buffer_slice, op_errs) = under_error_scopes(device, || {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download encoder"),
        });

        encoder.copy_buffer_to_buffer(src, 0, staging_buf, 0, bytes_needed as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // Only map the portion we actually copied
        let buffer_slice = staging_buf.slice(..bytes_needed as u64);
        buffer_slice.map_async(wgpu::MapMode::Read, result.callback());
        buffer_slice
    });

    if let Err(e) = await_map_result(device, &result) {
        return Err(diagnose_download_failure(e, None, &op_errs));
    }

    let data = buffer_slice.get_mapped_range()
        .map_err(|e| GpuError::BufferError(e.to_string()))?;
    let src_slice: &[f32] = bytemuck::cast_slice(&data);
    output.clear();
    output.extend_from_slice(src_slice);
    drop(data);
    staging_buf.unmap();

    Ok(())
}

/// A single request in a batched GPU→CPU download.
///
/// See [`download_f32_batched_reuse_into`] — all requests in one call share a
/// single command encoder, submit, map, and poll, eliminating per-download
/// GPU sync overhead.
pub struct BatchedDownloadRequest<'a> {
    /// Source GPU buffer to copy from.
    pub src: &'a Buffer,
    /// Number of f32 elements to read.
    pub count: usize,
    /// Pre-allocated output buffer. Cleared and filled with `count` elements.
    pub output: &'a mut Vec<f32>,
}

/// Download multiple f32 buffers in a single GPU sync cycle (Issue 421 P0.5).
///
/// All copies share one command encoder, one `queue.submit`, one `map_async`,
/// and one `device.poll(Wait)`. This reduces GPU sync overhead from N (one
/// per download) to 1 — the dominant cost in launch-overhead-bound paths like
/// the LoRA-Muon optimizer step, where P0 profiling measured each sync at
/// ~1.28 ms regardless of data size.
///
/// The staging buffer is grown (if needed) to hold the sum of all request
/// counts. Each source is copied into a contiguous, non-overlapping region
/// of the staging buffer at 4-byte (f32) aligned offsets (wgpu 23 requires
/// `copy_buffer_to_buffer` size/offset to be multiples of 4, not 256).
///
/// Each `request.output` is cleared and filled with exactly `request.count`
/// elements. The result is bit-identical to calling [`download_f32_reuse_into`]
/// N times — only the sync overhead differs.
///
/// # Example (the LoRA-Muon 4-buffer download pattern)
///
/// ```no_run
/// # use riir_infer_gpu::buffer::{BatchedDownloadRequest, DownloadStaging, download_f32_batched_reuse_into};
/// # use riir_infer_gpu::context::GpuContext;
/// # let ctx = GpuContext::new().unwrap();
/// # let (buf_a, buf_b) = (ctx.device.create_buffer(&wgpu::BufferDescriptor {
/// #     label: None, size: 64, usage: wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
/// #   ctx.device.create_buffer(&wgpu::BufferDescriptor {
/// #     label: None, size: 64, usage: wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }));
/// # let count = 16usize;
/// let mut out_a = vec![0.0f32; count];
/// let mut out_b = vec![0.0f32; count];
/// let mut staging = DownloadStaging::new();
/// let mut reqs = [
///     BatchedDownloadRequest { src: &buf_a, count, output: &mut out_a },
///     BatchedDownloadRequest { src: &buf_b, count, output: &mut out_b },
/// ];
/// download_f32_batched_reuse_into(&ctx.device, &ctx.queue, &mut reqs, &mut staging).unwrap();
/// ```
pub fn download_f32_batched_reuse_into(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    requests: &mut [BatchedDownloadRequest<'_>],
    staging: &mut DownloadStaging,
) -> Result<(), GpuError> {
    if requests.is_empty() {
        return Ok(());
    }

    let total_count: usize = requests.iter().map(|r| r.count).sum();
    if total_count == 0 {
        for r in requests.iter_mut() {
            r.output.clear();
        }
        return Ok(());
    }

    // Ensure the staging buffer can hold all requests concatenated.
    staging.ensure_capacity(device, total_count);
    let staging_buf = staging.buffer.as_ref().unwrap();

    let result = Arc::new(SyncMapResult::new());
    let total_bytes = (total_count * std::mem::size_of::<f32>()) as u64;

    let (buffer_slice, op_errs) = under_error_scopes(device, || {
        // Issue all buffer-to-buffer copies in a single command encoder.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched download encoder"),
        });

        let mut offset_bytes: u64 = 0;
        for req in requests.iter() {
            let bytes_needed = (req.count * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(req.src, 0, staging_buf, offset_bytes, bytes_needed);
            offset_bytes += bytes_needed;
        }

        queue.submit(std::iter::once(encoder.finish()));

        // One map + one poll for the entire concatenated region.
        let buffer_slice = staging_buf.slice(..total_bytes);
        buffer_slice.map_async(wgpu::MapMode::Read, result.callback());
        buffer_slice
    });

    if let Err(e) = await_map_result(device, &result) {
        return Err(diagnose_download_failure(e, None, &op_errs));
    }

    // Read each region out of the mapped range into its output buffer.
    let data = buffer_slice.get_mapped_range()
        .map_err(|e| GpuError::BufferError(e.to_string()))?;
    let all_bytes: &[u8] = &data;

    let mut offset_bytes = 0usize;
    for req in requests.iter_mut() {
        let bytes_needed = req.count * std::mem::size_of::<f32>();
        let region: &[f32] = bytemuck::cast_slice(&all_bytes[offset_bytes..offset_bytes + bytes_needed]);
        req.output.clear();
        req.output.extend_from_slice(region);
        offset_bytes += bytes_needed;
    }

    drop(data);
    staging_buf.unmap();

    Ok(())
}

/// Create an empty GPU buffer with the specified size (in f32 elements).
pub fn create_buffer(device: &wgpu::Device, count: usize, label: &str) -> Buffer {
    let size = (count * std::mem::size_of::<f32>()) as u64;
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Upload BF16 data as f32 (dequantize during upload).
///
/// BF16 = upper 16 bits of IEEE 754 f32. Conversion: `f32::from_bits((bits as u32) << 16)`.
/// Useful for loading Gemma 2 weights stored as BF16 in safetensors.
#[allow(dead_code)]
pub fn upload_bf16_as_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u16],
    label: &str,
) -> Buffer {
    let mut f32_data = Vec::with_capacity(data.len());
    for &bits in data {
        f32_data.push(f32::from_bits((bits as u32) << 16));
    }
    upload_f32(device, queue, &f32_data, label)
}

/// Download a single u32 from a GPU buffer (blocking).
///
/// Used for GPU-resident argmax: download only 4 bytes instead of full logits.
/// Returns the u32 value at offset 0 in the buffer.
pub fn download_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &Buffer,
) -> Result<u32, GpuError> {
    let (staging_buffer, creation_errs) = under_error_scopes(device, || {
        device.create_buffer(&BufferDescriptor {
            label: Some("staging u32 download"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });

    let result = Arc::new(SyncMapResult::new());

    let (buffer_slice, op_errs) = under_error_scopes(device, || {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_u32 encoder"),
        });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..4);
        buffer_slice.map_async(wgpu::MapMode::Read, result.callback());
        buffer_slice
    });

    if let Err(e) = await_map_result(device, &result) {
        return Err(diagnose_download_failure(e, Some(&creation_errs), &op_errs));
    }

    let data = buffer_slice.get_mapped_range()
        .map_err(|e| GpuError::BufferError(e.to_string()))?;
    let value = u32::from_ne_bytes(data[..4].try_into().unwrap());
    drop(data);
    staging_buffer.unmap();

    Ok(value)
}

/// Download a single u32 from a GPU buffer, reusing a pre-allocated staging buffer.
///
/// Use this instead of [`download_u32`] in hot loops (e.g., per-token decode)
/// to avoid repeated staging buffer allocation (~50-500μs driver overhead per call).
/// The staging buffer is lazily allocated on first call and reused thereafter.
pub fn download_u32_reuse(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &Buffer,
    staging: &mut DownloadStaging,
) -> Result<u32, GpuError> {
    // 1 element = 4 bytes (sizeof<u32> == sizeof<f32>).
    staging.ensure_capacity(device, 1);
    let staging_buf = staging.buffer.as_ref().unwrap();

    let result = Arc::new(SyncMapResult::new());

    let (buffer_slice, op_errs) = under_error_scopes(device, || {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_u32_reuse encoder"),
        });

        encoder.copy_buffer_to_buffer(src, 0, staging_buf, 0, 4);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buf.slice(..4);
        buffer_slice.map_async(wgpu::MapMode::Read, result.callback());
        buffer_slice
    });

    if let Err(e) = await_map_result(device, &result) {
        return Err(diagnose_download_failure(e, None, &op_errs));
    }

    let data = buffer_slice.get_mapped_range()
        .map_err(|e| GpuError::BufferError(e.to_string()))?;
    let value = u32::from_ne_bytes(data[..4].try_into().unwrap());
    drop(data);
    staging_buf.unmap();

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuContext;

    #[test]
    fn test_buffer_upload_download_roundtrip() {
        let ctx = if let Ok(ctx) = GpuContext::new() { ctx } else {
                println!("No GPU — skipping buffer test");
                return;
            };

        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let buffer = upload_f32(&ctx.device, &ctx.queue, &original, "test buffer");

        let downloaded =
            download_f32(&ctx.device, &ctx.queue, &buffer, 16).expect("download should succeed");

        assert_eq!(original.len(), downloaded.len());
        for (a, b) in original.iter().zip(downloaded.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_create_empty_buffer() {
        let ctx = if let Ok(ctx) = GpuContext::new() { ctx } else {
                println!("No GPU — skipping empty buffer test");
                return;
            };

        let buffer = create_buffer(&ctx.device, 1024, "empty test buffer");
        assert_eq!(buffer.size(), 1024 * std::mem::size_of::<f32>() as u64);
    }

    #[test]
    fn test_buffer_download_reuse_roundtrip() {
        let ctx = if let Ok(ctx) = GpuContext::new() { ctx } else {
                println!("No GPU — skipping buffer reuse test");
                return;
            };

        let mut staging = DownloadStaging::new();

        // First download — should allocate staging buffer
        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let buffer = upload_f32(&ctx.device, &ctx.queue, &original, "test buffer reuse");
        let downloaded = download_f32_reuse(&ctx.device, &ctx.queue, &buffer, 16, &mut staging)
            .expect("reuse download should succeed");
        assert_eq!(original, downloaded);
        assert!(staging.buffer.is_some());
        assert_eq!(staging.capacity, 16);

        // Second download with smaller count — should reuse staging buffer
        let small: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let small_buf = upload_f32(&ctx.device, &ctx.queue, &small, "small buffer");
        let downloaded_small =
            download_f32_reuse(&ctx.device, &ctx.queue, &small_buf, 4, &mut staging)
                .expect("reuse download small should succeed");
        assert_eq!(small, downloaded_small);
        // Capacity should still be 16 (not shrunk)
        assert_eq!(staging.capacity, 16);

        // Third download with larger count — should reallocate
        let large: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let large_buf = upload_f32(&ctx.device, &ctx.queue, &large, "large buffer");
        let downloaded_large =
            download_f32_reuse(&ctx.device, &ctx.queue, &large_buf, 64, &mut staging)
                .expect("reuse download large should succeed");
        assert_eq!(large, downloaded_large);
        assert_eq!(staging.capacity, 64);
    }

    #[test]
    fn test_batched_download_roundtrip() {
        let ctx = if let Ok(ctx) = GpuContext::new() { ctx } else {
                println!("No GPU — skipping batched download test");
                return;
            };

        // Three buffers of different sizes (mirrors the LoRA-Muon 4-buffer
        // pattern, but with 3 to test non-power-of-two request counts).
        let data_a: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let data_b: Vec<f32> = (0..32).map(|i| i as f32 * -0.2).collect();
        let data_c: Vec<f32> = (0..8).map(|i| i as f32 * 0.3).collect();

        let buf_a = upload_f32(&ctx.device, &ctx.queue, &data_a, "batched_a");
        let buf_b = upload_f32(&ctx.device, &ctx.queue, &data_b, "batched_b");
        let buf_c = upload_f32(&ctx.device, &ctx.queue, &data_c, "batched_c");

        let mut out_a = vec![0.0f32; 16];
        let mut out_b = vec![0.0f32; 32];
        let mut out_c = vec![0.0f32; 8];
        let mut staging = DownloadStaging::new();

        let mut reqs = [
            BatchedDownloadRequest { src: &buf_a, count: 16, output: &mut out_a },
            BatchedDownloadRequest { src: &buf_b, count: 32, output: &mut out_b },
            BatchedDownloadRequest { src: &buf_c, count: 8,  output: &mut out_c },
        ];
        download_f32_batched_reuse_into(&ctx.device, &ctx.queue, &mut reqs, &mut staging)
            .expect("batched download should succeed");

        assert_eq!(out_a, data_a, "buffer A mismatch");
        assert_eq!(out_b, data_b, "buffer B mismatch");
        assert_eq!(out_c, data_c, "buffer C mismatch");

        // Staging should have grown to the sum: 16 + 32 + 8 = 56 elements.
        assert_eq!(staging.capacity, 56);
    }

    #[test]
    fn test_batched_download_matches_sequential() {
        // Parity check: batched download produces bit-identical results to
        // sequential `download_f32_reuse_into` calls. This is the G1 gate
        // for the Issue 421 P0.5 optimization.
        let ctx = if let Ok(ctx) = GpuContext::new() { ctx } else {
                println!("No GPU — skipping batched parity test");
                return;
            };

        // Use sizes that mirror the LoRA-Muon pattern: two different counts.
        let data_a: Vec<f32> = (0..48).map(|i| (i as f32).sin()).collect();
        let data_b: Vec<f32> = (0..24).map(|i| (i as f32).cos()).collect();
        let buf_a = upload_f32(&ctx.device, &ctx.queue, &data_a, "parity_a");
        let buf_b = upload_f32(&ctx.device, &ctx.queue, &data_b, "parity_b");

        // Sequential downloads.
        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();
        let mut seq_staging = DownloadStaging::new();
        download_f32_reuse_into(&ctx.device, &ctx.queue, &buf_a, 48, &mut seq_staging, &mut seq_a)
            .unwrap();
        download_f32_reuse_into(&ctx.device, &ctx.queue, &buf_b, 24, &mut seq_staging, &mut seq_b)
            .unwrap();

        // Batched downloads.
        let mut bat_a = Vec::new();
        let mut bat_b = Vec::new();
        let mut bat_staging = DownloadStaging::new();
        let mut reqs = [
            BatchedDownloadRequest { src: &buf_a, count: 48, output: &mut bat_a },
            BatchedDownloadRequest { src: &buf_b, count: 24, output: &mut bat_b },
        ];
        download_f32_batched_reuse_into(&ctx.device, &ctx.queue, &mut reqs, &mut bat_staging)
            .unwrap();

        // Bit-identical comparison.
        assert_eq!(seq_a, bat_a, "A: batched must match sequential bit-for-bit");
        assert_eq!(seq_b, bat_b, "B: batched must match sequential bit-for-bit");
    }

    #[test]
    fn test_batched_download_empty() {
        // Edge case: empty request slice should be a no-op.
        let ctx = if let Ok(ctx) = GpuContext::new() { ctx } else {
                println!("No GPU — skipping batched empty test");
                return;
            };

        let mut staging = DownloadStaging::new();
        let reqs: &mut [BatchedDownloadRequest<'_>] = &mut [];
        let result = download_f32_batched_reuse_into(&ctx.device, &ctx.queue, reqs, &mut staging);
        assert!(result.is_ok());
        assert_eq!(staging.capacity, 0);
    }
}
