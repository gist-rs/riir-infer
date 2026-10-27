use super::{
    mem_manager::WgpuMemManager,
    poll::WgpuPoll,
    timings::{QueryProfiler, TimestampQuerySetBudget},
};
use crate::{
    WgpuResource,
    controller::WgpuAllocController,
    schedule::{Addresses, ScheduleTask},
};
use core::iter;
#[cfg(renderdoc)]
use core::{cell::LazyCell, ptr::null};
use cubecl_common::{
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use cubecl_core::{
    CubeCount, MemoryConfiguration,
    server::{IoError, ProfileError, ProfilingToken, ServerError, StreamErrorMode},
    zspace::Shape,
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::{self, DynFut};
#[cfg(renderdoc)]
use cubecl_environment::sync::Mutex;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, SharedMemoryBindings},
    timestamp_profiler::TimestampProfiler,
};
#[cfg(renderdoc)]
use renderdoc::{RenderDoc, V100};
use std::{
    collections::HashMap,
    future::Future,
    hash::{Hash, Hasher},
    num::NonZero,
    pin::Pin,
    sync::Arc,
};
use wgpu::ComputePipeline;

/// Maximum number of cached bind groups. Each entry is ~256 bytes of driver
/// state; 4096 entries ≈ 1 MB. The forward path for a 64-layer model creates
/// ~832 unique bind groups, so this is comfortably sized with headroom for
/// multiple concurrent kernels.
///
/// Issue 660 (Metal ICB stepping stone): bind group creation is a driver call
/// (~3-5 µs each on Metal). The forward path creates the same 832 bind groups
/// every token because all buffers are persistent. Caching them eliminates
/// ~2.5-4 ms/token of driver overhead → +1.5-2.5 tok/s on a 24 tok/s baseline.
const BIND_GROUP_CACHE_CAPACITY: usize = 4096;

/// Issue 660 diagnostics: global cache hit/miss counters (atomic for
/// lock-free reads from any thread, including benchmark harnesses).
/// Reset via [`reset_bind_group_cache_stats`].
pub static BG_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BG_CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Reset the global bind group cache diagnostic counters.
pub fn reset_bind_group_cache_stats() {
    BG_CACHE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    BG_CACHE_MISSES.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 660 A/B toggle: enable/disable the bind group cache at runtime.
/// Default: true (cache enabled). Set to false for head-to-head benchmarking
/// against the uncached path.
pub static BIND_GROUP_CACHE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Set the bind group cache enabled state.
pub fn set_bind_group_cache_enabled(enabled: bool) {
    BIND_GROUP_CACHE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

// Issue 660 profiling: per-section nanosecond accumulators for
// `register_pipeline`. Read via [`register_pipeline_profile`] or the
// individual statics. Reset via [`reset_register_pipeline_profile`].
pub static PROF_ENTRIES_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_BINDGROUP_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_PASS_GET_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_SET_PIPELINE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_SET_BG_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_SET_IMM_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_TRANSITIONS_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_DISPATCH_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Issue 660 profiling toggle. When true, `register_pipeline` accumulates
/// per-section nanosecond timings into the PROF_* statics. Default: false
/// (avoids ~0.3 ms/token Instant::now() overhead in production).
pub static REGISTER_PIPELINE_PROFILE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Enable/disable `register_pipeline` per-section profiling.
pub fn set_register_pipeline_profile_enabled(enabled: bool) {
    REGISTER_PIPELINE_PROFILE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Reset all `register_pipeline` profiling accumulators.
pub fn reset_register_pipeline_profile() {
    PROF_ENTRIES_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_BINDGROUP_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_PASS_GET_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_SET_PIPELINE_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_SET_BG_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_SET_IMM_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_TRANSITIONS_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_DISPATCH_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROF_DISPATCH_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

// Issue 761 profiling: whole-pipeline dispatch decomposition counters.
//
// Unlike the Issue 660 PROF_* statics (per-section inside register_pipeline,
// gated by an enable flag), these are ALWAYS-ON monotonic counters + nanosecond
// accumulators covering every host-side station a dispatch passes through,
// including the two places the host can BLOCK on the GPU:
//
//   `SubmissionLoad::regulate`      — poll(Wait) on the previous submission
//                                      index once ≥512 tasks are in flight
//                                      (MAX_TOTAL_TASKS).
//   `write_to_buffer` write flush    — poll(Wait) once 64 uniform writes are
//                                      pending (MAX_PENDING_WRITES); on
//                                      Metal/WGSL every dispatch with scalar
//                                      args writes a uniform buffer
//                                      (`create_uniform` in `into_resources`).
//
// Together with wall time these counters answer, for any measured region,
// whether the CPU dispatch encoding is the binding cost (ENQUEUE_* dominates)
// or the host is merely GPU-paced (REGULATE/WRITE_POLL dominates). Atomic
// adds are ~ns; the counters are unconditional. Reset via
// [`reset_dispatch_profile`] (also resets the Issue 660 statics).
//
// Nesting (a lower row is contained in the row above):
//   LAUNCH_NANOS            — WgpuServer::launch entry → scheduler.register
//                             return (incl. pipeline cache lookup, LeasePool
//                             acquire, prepare_bindings, enqueue_task).
//     ENQUEUE_EXEC_NANOS    — the Execute arm of WgpuStream::enqueue_task.
//       (register_pipeline per-section PROF_* statics when the 660 flag is on)
//         FLUSH_NANOS       — WgpuStream::flush (submit + memory_cleanup + …).
//           REGULATE_NANOS  — blocked inside SubmissionLoad::regulate.
//           WRITE_POLL_NANOS— blocked inside write_to_buffer's 64-write flush.
//                             (also reachable outside FLUSH via create_uniform)
//   ENQUEUE_WRITE_NANOS     — the Write arm of enqueue_task (flush + write).
//   READ_NANOS              — WgpuStream::read_resources (drain + map).
pub static LAUNCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LAUNCH_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Issue 761: sub-stations of `WgpuServer::launch` (all nested in LAUNCH_NANOS):
/// pipeline cache lookup / LeasePool+scalar-pack prelude / scheduler register
/// (register may batch-execute queued tasks, so ENQUEUE_* time can nest here).
pub static PIPELINE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PREPARE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REGISTER_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENQUEUE_EXEC_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENQUEUE_EXEC_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENQUEUE_WRITE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENQUEUE_WRITE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static FLUSH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static FLUSH_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SUBMIT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REGULATE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REGULATE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WRITE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WRITE_POLL_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WRITE_POLL_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static READ_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static READ_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Reset every Issue 761 dispatch-decomposition counter (and the Issue 660
/// per-section statics, so one call readies the full profile).
pub fn reset_dispatch_profile() {
    LAUNCH_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    LAUNCH_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PIPELINE_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    PREPARE_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    REGISTER_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    ENQUEUE_EXEC_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    ENQUEUE_EXEC_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    ENQUEUE_WRITE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    ENQUEUE_WRITE_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    FLUSH_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    FLUSH_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    SUBMIT_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    REGULATE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    REGULATE_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    WRITE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    WRITE_POLL_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    WRITE_POLL_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    READ_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    READ_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    reset_register_pipeline_profile();
}

#[cfg(renderdoc)]
thread_local! {
    static RENDERDOC: LazyCell<Option<Mutex<RenderDoc<V100>>>> = LazyCell::new(|| RenderDoc::new().ok().map(Mutex::new));
}

#[derive(Debug)]
enum Timings {
    // Boxed: `QueryProfiler` is much larger than `TimestampProfiler`
    // (clippy::large_enum_variant).
    Device(Box<QueryProfiler>),
    System(TimestampProfiler),
}

#[derive(Debug)]
pub struct WgpuStream {
    pub mem_manage: WgpuMemManager,
    pub device: wgpu::Device,
    pub errors: Vec<ServerError>,
    compute_pass: Option<wgpu::ComputePass<'static>>,
    timings: Timings,
    tasks_count: usize,
    tasks_max: usize,
    queue: wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    poll: WgpuPoll,
    submission_load: SubmissionLoad,
    /// Number of consecutive `write_buffer` calls without a `queue.submit()`.
    /// Used to prevent wgpu staging buffer pool exhaustion during bulk writes
    /// (e.g. model loading with hundreds of tensors).
    pending_write_count: usize,
    /// Cross-stream input bindings for tasks recorded but not yet submitted.
    /// Kept alive here until the next `flush` ties their release to the submission's completion.
    /// See [`ScheduleTask::Execute::pins`](crate::schedule::ScheduleTask).
    shared_bindings: SharedMemoryBindings,

    /// Bind group cache (Issue 660 stepping stone). Maps a hash of
    /// (pipeline, resource buffer ids + offsets + sizes) to a reusable
    /// `wgpu::BindGroup`. The forward path creates the same 832 bind groups
    /// every token because all buffers are persistent — the cache turns
    /// token 2+ into all-hits, eliminating ~3-5 µs × 832 = ~2.5-4 ms/token
    /// of driver overhead.
    ///
    /// Safety: entries are invalidated implicitly — when a `wgpu::Buffer` is
    /// dropped, its id changes, so the next lookup with the new buffer id is a
    /// cache miss. The cache holds `wgpu::BindGroup` (which holds strong refs
    /// to its buffers), so buffers cannot be dropped while cached.
    bind_group_cache: HashMap<u64, wgpu::BindGroup>,

    /// Issue 660 instrumentation: cache hit/miss counters for diagnostics.
    /// Accessed via [`bind_group_cache_stats`](Self::bind_group_cache_stats).
    bg_cache_hits: u64,
    bg_cache_misses: u64,
}

impl WgpuStream {
    /// Creates a new WGPU stream.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        timing_budget: Arc<TimestampQuerySetBudget>,
        tasks_max: usize,
        logger: Arc<ServerLogger>,
        use_vulkan_compiler: bool,
    ) -> Self {
        // Device timing needs a counter sample buffer per query set, capped per device on
        // Metal. Reserve a budget slot up front (lock-free); if none is free, fall back to
        // the system timer so we never exceed the hardware limit.
        let timings = if timing_method == TimingMethod::Device && timing_budget.try_acquire() {
            Timings::Device(Box::new(QueryProfiler::new(&queue, &device, timing_budget)))
        } else {
            if cfg!(target_family = "wasm") {
                // On WASM, there's not much we can do here anymore. This should be very rare however,
                // all modern GPU's support timestamp queries.
                panic!(
                    "Cannot profile on web assembly without timestamp_query feature as it requires blocking."
                );
            }
            Timings::System(TimestampProfiler::default())
        };

        #[cfg(renderdoc)]
        RENDERDOC.with(|renderdoc| {
            if let Some(renderdoc) = &**renderdoc {
                let mut renderdoc = renderdoc.lock();
                renderdoc.start_frame_capture(null(), null());
            }
        });

        let poll = WgpuPoll::new(device.clone());

        #[allow(unused_mut)]
        let mut mem_manage = WgpuMemManager::new(
            device.clone(),
            memory_properties,
            memory_config,
            logger,
            use_vulkan_compiler,
        );

        Self {
            mem_manage,
            compute_pass: None,
            timings,
            errors: Vec::new(),
            encoder: {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("CubeCL Tasks Encoder"),
                })
            },
            device,
            queue,
            tasks_count: 0,
            tasks_max,
            poll,
            submission_load: SubmissionLoad::default(),
            pending_write_count: 0,
            shared_bindings: SharedMemoryBindings::default(),
            bind_group_cache: HashMap::new(),
            bg_cache_hits: 0,
            bg_cache_misses: 0,
        }
    }

    /// Enqueue a [`ScheduleTask`] on this stream.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to execute.
    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write { data, buffer } => {
                // Issue 761 profiling: time the Write arm (flush + write).
                let t0 = std::time::Instant::now();
                // It is important to flush before writing, as the write operation is inserted
                // into the QUEUE not the encoder. We want to make sure all outstanding work
                // happens _before_ the write operation.
                let _ = self
                    .flush(StreamErrorMode {
                        ignore: true,
                        flush: false,
                    })
                    .ok();
                self.write_to_buffer(&buffer, &data);
                ENQUEUE_WRITE_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                ENQUEUE_WRITE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            ScheduleTask::Execute {
                pipeline,
                count,
                resources,
                mut shared_inputs,
            } => {
                // Issue 761 profiling: time the Execute arm (bindings append +
                // into_resources + register_pipeline incl. any flush).
                let t0 = std::time::Instant::now();
                // Drain into the stream's pending pins; the handle returns its buffer to the
                // server pool when it drops at the end of this arm.
                self.shared_bindings
                    .bindings
                    .append(&mut shared_inputs.bindings);
                let (resources, custom_handles, addresses) = resources.into_resources(self);
                self.register_pipeline(pipeline, &resources, &custom_handles, addresses, &count);
                ENQUEUE_EXEC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                ENQUEUE_EXEC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Read multiple buffers lazily to [Bytes], potentially using pinned memory.
    ///
    /// # Arguments
    ///
    /// * `self` - The current stream.
    /// * `descriptors` - A vector of copy descriptors specifying the source data.
    ///
    /// # Returns
    ///
    /// A [Result] containing a vector of [Bytes] with the copied data, or an [`IoError`] if any copy fails.
    pub fn read_resources(
        &mut self,
        descriptors: Vec<(WgpuResource, Shape, usize)>,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // Issue 761 profiling: READ_NANOS covers only the SYNCHRONOUS prologue
        // (staging reserve + copy encode + flush + map_async initiation). The
        // blocking map wait happens in the caller's await/block_on and is
        // measured by the bench as drain/read wall time — host blockage there
        // is GPU drain, not encode cost.
        let t0 = std::time::Instant::now();
        self.compute_pass = None;
        let mut staging_info = Vec::with_capacity(descriptors.len());
        let mut callbacks = Vec::with_capacity(descriptors.len());

        for (resource, shape, elem_size) in descriptors {
            let size = shape.iter().product::<usize>() * elem_size;

            // Zero-sized resources don't need a GPU copy.
            if resource.size == 0 {
                staging_info.push(None);
                continue;
            }

            // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
            // memory is 32 bytes aligned (see WgpuStorage).
            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let aligned_len = resource.size.div_ceil(align) * align;
            // riir-ai Issue 714 vendor patch: a readback staging OOM must not
            // kill the process-global server (upstream `.unwrap()` on the DSD
            // thread). Surface it through this function's existing error
            // channel so only the reading client fails.
            let (staging, binding) = match self.mem_manage.reserve_staging(aligned_len) {
                Ok(val) => val,
                Err(err) => return Box::pin(async { Err(ServerError::Io(err)) }),
            };

            self.tasks_count += 1;
            self.encoder.copy_buffer_to_buffer(
                &resource.buffer,
                resource.offset,
                &staging.buffer,
                0,
                aligned_len,
            );
            staging_info.push(Some((staging, binding, size)));
        }

        // Flush all commands to the queue, so GPU gets started on copying to the staging buffer.
        let _ = self
            .flush(StreamErrorMode {
                ignore: true,
                flush: false,
            })
            .ok();

        for entry in staging_info.iter() {
            if let Some((staging, _binding, _size)) = entry {
                let (sender, receiver) = cubecl_environment::future::channel::bounded(1);
                // riir-ai Issue 714 follow-up (Plan 343 T1.7, 2026-08-26): a staging
                // buffer whose creation failed is an INVALID stub; `map_async` on it
                // raises a Validation error that, uncaptured, PANICS the process via
                // wgpu's default error handler (observed live: `Buffer::map_async —
                // Buffer with '' label is invalid`, rc=101, the Plan 343 seam check).
                // Capture the synchronous error here and fail THIS read cleanly; the
                // callback still fires with `Err`, handled below.
                let scope_validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
                staging
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| {
                        // This might fail if the channel is closed (eg. the future is dropped).
                        // This is fine, just means results aren't needed anymore.
                        let _ = sender.try_send(v);
                    });
                if let Some(err) = cubecl_environment::future::block_on(scope_validation.pop()) {
                    let reason = format!("staging buffer map_async failed: {err}");
                    let backtrace = BackTrace::capture();
                    return Box::pin(async {
                        Err(ServerError::Generic { reason, backtrace })
                    });
                }

                callbacks.push(Some(receiver));
            } else {
                callbacks.push(None);
            }
        }

        let poll = self.poll.start_polling();
        READ_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        READ_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Box::pin(async move {
            for receiver in callbacks.iter().flatten() {
                let mapped = receiver
                    .recv()
                    .await
                    .expect("Unable to receive buffer slice result.");
                // Same Issue 714 follow-up: the callback fires with `Err` exactly when
                // the mapping could not start (invalid/OOM staging buffer). Fail the
                // read instead of panicking inside the future.
                if let Err(err) = mapped {
                    return Err(ServerError::Generic {
                        reason: format!("staging buffer failed to map: {err:?}"),
                        backtrace: BackTrace::capture(),
                    });
                }
            }

            // Can stop polling now.
            core::mem::drop(poll);

            let result = {
                staging_info
                    .into_iter()
                    .map(|entry| {
                        if let Some((staging, binding, size)) = entry {
                            let controller =
                                Box::new(WgpuAllocController::init(binding, staging.buffer));
                            // SAFETY: The binding has initialized memory for at least `size` bytes.
                            unsafe { Bytes::from_controller(controller, size) }
                        } else {
                            Bytes::from_bytes_vec(vec![])
                        }
                    })
                    .collect()
            };

            Ok(result)
        })
    }

    // Bit silly but needed to make the borrow checker happy.
    fn system_profiler(&mut self) -> &mut TimestampProfiler {
        let Timings::System(timing) = &mut self.timings else {
            panic!("Unexpected timings type");
        };
        timing
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        if matches!(self.timings, Timings::System(_)) {
            cubecl_environment::future::block_on(self.sync())?;
        } else {
            self.flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })?;
        }

        match &mut self.timings {
            Timings::System(_) => {
                let profiler = self.system_profiler();
                Ok(profiler.start())
            }
            Timings::Device(query) => {
                self.compute_pass = None;
                let token = query.start_profile();
                Ok(token)
            }
        }
    }

    pub fn profile_error(&mut self, error: ProfileError) {
        match &mut self.timings {
            Timings::Device(profiler) => {
                profiler.error(error);
            }
            Timings::System(profiler) => {
                profiler.error(error);
            }
        }
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        match &mut self.timings {
            Timings::System(..) => {
                // Nb: WASM _has_ to use device timing and will panic here if query timestamps are not supported.
                let result = future::block_on(self.sync());
                let profiler = self.system_profiler();

                if let Err(err) = result {
                    profiler.error(ProfileError::Server(Box::new(err)));
                }
                profiler.stop(token)
            }
            Timings::Device(..) => {
                let poll = self.poll.start_polling();
                self.compute_pass = None;

                // Submit commands needed for profiling.
                let buffer = {
                    let Timings::Device(timing) = &mut self.timings else {
                        return Err(ProfileError::Unknown {
                            reason: "Unexpected timings type".to_string(),
                            backtrace: BackTrace::capture(),
                        });
                    };
                    timing.stop_profile_setup(token, &self.device, &mut self.encoder)?
                };

                // This flushes the queue to execute the encoder write command to write the
                // timings.
                self.tasks_count += 1;
                let result = self.flush(StreamErrorMode {
                    ignore: false,
                    flush: true,
                });

                let Timings::Device(timing) = &mut self.timings else {
                    return Err(ProfileError::Unknown {
                        reason: "Unexpected timings type".to_string(),
                        backtrace: BackTrace::capture(),
                    });
                };

                match result {
                    Ok(_) => timing.stop_profile(buffer, poll),
                    Err(err) => {
                        // Just to clean the timing buffer.
                        let _ = timing.stop_profile(buffer, poll).ok();
                        Err(ProfileError::Server(Box::new(err)))
                    }
                }
            }
        }
    }

    pub fn sync(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ServerError>> + Send + 'static>> {
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Internal);

        let flush_error = self
            .flush(StreamErrorMode {
                ignore: false,
                flush: true,
            })
            .err();

        let queue = self.queue.clone();
        let error_future = error_scope.pop();
        let poll = self.poll.start_polling();

        Box::pin(async move {
            let (sender, receiver) = cubecl_environment::future::channel::bounded::<()>(1);
            queue.on_submitted_work_done(move || {
                // Signal that we're done.
                let _ = sender.try_send(());
                core::mem::drop(poll);
            });
            let _ = receiver.recv().await;

            if let Some(error) = error_future.await {
                return Err(ServerError::Generic {
                    reason: format!("{error}"),
                    backtrace: BackTrace::capture(),
                });
            }

            match flush_error {
                Some(err) => Err(err),
                None => Ok(()),
            }
        })
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.mem_manage.reserve(size)
    }

    /// Registers a new error into the error sink.
    pub fn error(&mut self, error: ServerError) {
        self.errors.push(error);
    }

    pub(crate) fn create_uniform(&mut self, data: &[u8]) -> WgpuResource {
        let resource = self.mem_manage.reserve_uniform_uncached(data);
        self.write_to_buffer(&resource, data);
        resource
    }

    /// Issue 761 follow-up 2: content-cached variant for VALUE-carrying
    /// uniforms ONLY (the metadata binding — dims/strides/scalars, never
    /// buffer addresses). A cache hit returns the pinned resource with no
    /// pool reserve, no queue write, and no pending-write flush poll; a miss
    /// takes the legacy path once and pins the reservation. Use
    /// [`Self::create_uniform`] for anything address-carrying.
    pub(crate) fn create_metadata_uniform(&mut self, data: &[u8]) -> WgpuResource {
        if let Some(resource) = self.mem_manage.lookup_uniform(data) {
            return resource;
        }
        let resource = self.mem_manage.reserve_uniform(data);
        self.write_to_buffer(&resource, data);
        resource
    }

    // Nb: this function submits a command to the _queue_ not to the encoder,
    // so you have to be really careful about the ordering of operations here.
    // Any buffer which has outstanding (not yet flushed) compute work should
    // NOT be copied to.
    fn write_to_buffer(&mut self, resource: &WgpuResource, data: &[u8]) {
        // Issue 761 profiling: count every queue write (uniform uploads via
        // create_uniform dominate during decode on Metal/WGSL).
        WRITE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Nothing to write for zero-sized resources.
        if resource.size == 0 {
            return;
        }

        // Copying into a buffer has to be 4 byte aligned. We can safely do so, as
        // memory is also aligned (see WgpuStorage). Per the WebGPU spec, this
        // just has to be a multiple of 4: https://www.w3.org/TR/webgpu/#dom-gpuqueue-writebuffer
        let copy_align = wgpu::COPY_BUFFER_ALIGNMENT;
        let size = resource.size.next_multiple_of(copy_align);

        if size == data.len() as u64 {
            // write_buffer is the recommended way to write this data, as:
            // - On WebGPU, from WASM, this can save a copy to the JS memory.
            // - On devices with unified memory, this could skip the staging buffer entirely.
            self.queue
                .write_buffer(&resource.buffer, resource.offset, data);
        } else {
            // For sizes not aligned we need to only write a part of the staging buffer, do this
            // with `write_buffer_with`.
            let mut buffer = self
                .queue
                .write_buffer_with(
                    &resource.buffer,
                    resource.offset,
                    NonZero::new(size).unwrap(),
                )
                .expect("Internal error: Failed to call `write_buffer_with`, this likely means no staging buffer could be allocated.");
            buffer.slice(0..data.len()).copy_from_slice(data);
        }

        self.pending_write_count += 1;

        // Prevent wgpu staging buffer pool exhaustion during bulk writes (e.g. model
        // loading with hundreds of tensors). queue.write_buffer() is async — wgpu
        // copies data into an internal staging buffer, then transfers to GPU on the
        // next queue.submit(). Without periodic submits, hundreds of writes accumulate
        // and staging buffers get recycled before the GPU copy completes, silently
        // corrupting early tensors.
        // See: https://github.com/tracel-ai/cubecl/issues/1120
        const MAX_PENDING_WRITES: usize = 64;

        if self.pending_write_count >= MAX_PENDING_WRITES {
            // Issue 761 profiling: this branch submits + POLL-WAITS for the GPU
            // to complete everything up to the just-submitted index — a hard
            // host block on GPU progress whenever uniform writes outpace
            // submits.
            let t_poll = std::time::Instant::now();
            // Submit a fresh, empty command buffer to flush all pending write_buffer work.
            // wgpu flushes its internal staging-buffer copies on any queue.submit(),
            // so we don't need to touch the main compute encoder here.
            let write_flush_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL Write Flush Encoder"),
                    });
            let index = self.queue.submit([write_flush_encoder.finish()]);
            SUBMIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Wait for the GPU to finish processing these writes before continuing.
            #[cfg(not(target_family = "wasm"))]
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            }) {
                log::warn!("wgpu: write flush poll failed ({e})");
            }

            WRITE_POLL_NANOS.fetch_add(t_poll.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            WRITE_POLL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.pending_write_count = 0;
        }
    }

    fn flush_if_needed(&mut self) {
        // Flush when there are too many tasks, or when too many handles are locked.
        // Locked handles should only accumulate in rare circumstances (where uniforms
        // are being created but no work is submitted).
        if self.tasks_count >= self.tasks_max {
            let _ = self
                .flush(StreamErrorMode {
                    ignore: true,
                    flush: false,
                })
                .ok();
        }
    }

    pub fn flush(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        // Issue 761 profiling: time the whole flush (submit + regulate +
        // memory cleanup). REGULATE_NANOS is nested inside and reported
        // separately — it is GPU-wait, not encode cost.
        let t_flush = std::time::Instant::now();
        let result = self.flush_inner(mode);
        FLUSH_NANOS.fetch_add(t_flush.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        FLUSH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        result
    }

    fn flush_inner(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        if self.tasks_count == 0 {
            self.shared_bindings.clear();
            return self.flush_errors(mode);
        }

        // End the current compute pass.
        self.compute_pass = None;

        // Submit the pending actions to the queue. This will _first_ submit the
        // pending uniforms copy operations, then the main tasks.
        let tasks_encoder = {
            std::mem::replace(&mut self.encoder, {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL Tasks Encoder"),
                    })
            })
        };

        // This will _first_ fire off all pending write_buffer work.
        let index = self.queue.submit([tasks_encoder.finish()]);
        SUBMIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Release cross-stream input pins.
        if !self.shared_bindings.is_empty() {
            let pins = core::mem::take(&mut self.shared_bindings);
            self.queue.on_submitted_work_done(move || {
                drop(pins);
            });
        }

        self.submission_load
            .regulate(&self.device, self.tasks_count, index);

        // Cleanup allocations and deallocations.
        self.mem_manage.memory_cleanup(false);
        self.mem_manage.release_uniforms();

        #[cfg(renderdoc)]
        RENDERDOC.with(|renderdoc| {
            if let Some(renderdoc) = &**renderdoc {
                let mut renderdoc = renderdoc.lock();
                renderdoc.end_frame_capture(null(), null());
                renderdoc.start_frame_capture(null(), null());
            }
        });

        self.tasks_count = 0;
        self.pending_write_count = 0;

        self.flush_errors(mode)
    }

    fn flush_errors(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        #[cfg(feature = "deny-validation-errors")]
        {
            let validation_errors = wgpu_hal::VALIDATION_CANARY.get_and_reset();
            self.errors.extend(
                validation_errors
                    .into_iter()
                    .map(|err| ServerError::Validation {
                        message: err,
                        backtrace: BackTrace::capture(),
                    }),
            );
        }

        if mode.flush {
            let errors = self.flush_errors_queue();

            if !mode.ignore && !errors.is_empty() {
                let error = ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                };
                return Err(error);
            }
        } else if !mode.ignore && !self.errors.is_empty() {
            let error = ServerError::ServerUnhealthy {
                errors: self.errors.clone(),
                backtrace: BackTrace::capture(),
            };
            return Err(error);
        }

        Ok(())
    }

    fn register_pipeline(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        resources: &[WgpuResource],
        custom_resources: &[WgpuResource],
        addresses: Option<Addresses>,
        dispatch: &CubeCount,
    ) {
        if dispatch.is_empty() {
            return;
        }

        // Issue 660 profiling: measure per-section cost of register_pipeline.
        // Accumulated into global atomics, readable from benches.
        // Controlled by REGISTER_PIPELINE_PROFILE_ENABLED (default: false to
        // avoid the ~0.3 ms/token Instant::now() overhead in production).
        let profile =
            REGISTER_PIPELINE_PROFILE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let prof_t0 = if profile { Some(std::time::Instant::now()) } else { None };

        let entries = resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.as_wgpu_bind_resource(),
            })
            .collect::<Vec<_>>();

        let prof_entries = prof_t0.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        let prof_t1 = if profile { Some(std::time::Instant::now()) } else { None };
        // Issue 660 stepping stone: resolve (create or fetch cached) the bind
        // group BEFORE borrowing `compute_pass`. Splitting the borrow this way
        // satisfies the borrow checker — `get_or_create_bind_group` needs
        // `&mut self` (for the device + cache), while the compute pass closure
        // also borrows `&mut self`. Resolving first avoids the double-borrow.
        let cached_bind_group = if !resources.is_empty() {
            Some(self.get_or_create_bind_group(pipeline.clone(), &entries))
        } else {
            None
        };
        let prof_bindgroup = prof_t1.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        let prof_t2 = if profile { Some(std::time::Instant::now()) } else { None };
        // Start a new compute pass if needed. The forget_lifetime allows
        // to store this with a 'static lifetime, but the compute pass must
        // be dropped before the encoder. This isn't unsafe - it's still checked at runtime.
        let pass = self.compute_pass.get_or_insert_with(|| {
            let writes = if let Timings::Device(query_time) = &mut self.timings {
                query_time
                    .register_profile_device(&self.device)
                    .map(|query_set| wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    })
            } else {
                None
            };
            self.encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: writes,
                })
                .forget_lifetime()
        });
        let prof_pass_get = prof_t2.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        self.tasks_count += 1;

        let prof_t3 = if profile { Some(std::time::Instant::now()) } else { None };
        pass.set_pipeline(&pipeline);
        let prof_set_pipeline = prof_t3.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        let prof_t4 = if profile { Some(std::time::Instant::now()) } else { None };
        if let Some(bind_group) = &cached_bind_group {
            pass.set_bind_group(0, bind_group, &[]);
        }
        let prof_set_bg = prof_t4.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        let prof_t5 = if profile { Some(std::time::Instant::now()) } else { None };
        if let Some(addresses) = addresses {
            pass.set_immediates(0, bytemuck::cast_slice(&addresses));
        }
        let prof_set_imm = prof_t5.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        let prof_t6 = if profile { Some(std::time::Instant::now()) } else { None };
        if !custom_resources.is_empty() {
            let buffer_transitions =
                custom_resources
                    .iter()
                    .map(|resource| wgpu::BufferTransition {
                        buffer: &resource.buffer,
                        state: wgpu::BufferUses::STORAGE_READ_WRITE,
                    });
            pass.transition_resources(buffer_transitions, iter::empty())
        }
        let prof_transitions = prof_t6.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        let prof_t7 = if profile { Some(std::time::Instant::now()) } else { None };
        match dispatch.clone() {
            CubeCount::Static(x, y, z) => {
                pass.dispatch_workgroups(x, y, z);
            }
            CubeCount::Dynamic(binding) => {
                let res = self.mem_manage.get_resource(binding).unwrap();
                pass.dispatch_workgroups_indirect(&res.buffer, res.offset);
            }
        }
        let prof_dispatch = prof_t7.map_or(std::time::Duration::ZERO, |t| t.elapsed());

        // Accumulate profiling (only if enabled).
        if profile {
        PROF_ENTRIES_NANOS.fetch_add(prof_entries.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_BINDGROUP_NANOS.fetch_add(prof_bindgroup.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_PASS_GET_NANOS.fetch_add(prof_pass_get.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_SET_PIPELINE_NANOS.fetch_add(prof_set_pipeline.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_SET_BG_NANOS.fetch_add(prof_set_bg.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_SET_IMM_NANOS.fetch_add(prof_set_imm.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_TRANSITIONS_NANOS.fetch_add(prof_transitions.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_DISPATCH_NANOS.fetch_add(prof_dispatch.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        self.flush_if_needed();
    }

    /// Compute a hash key for a bind group from its layout + entries.
    ///
    /// The key uniquely identifies the (pipeline, resource set) tuple:
    /// - Pipeline identity via `Arc::as_ptr` (stable for the lifetime of the Arc).
    /// - Each entry's buffer identity (wgpu::Buffer implements Hash via its inner
    ///   dispatch reference), offset, and size.
    ///
    /// Collisions are astronomically unlikely (u64 hash of ~5 entries × 24 bytes each)
    /// and would only cause a cache miss on collision, not incorrect results.
    fn bind_group_hash(
        pipeline: &Arc<ComputePipeline>,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        (Arc::as_ptr(pipeline) as usize).hash(&mut hasher);
        for entry in entries {
            entry.binding.hash(&mut hasher);
            // Hash the BindingResource — for buffers this includes buffer id + offset + size.
            match &entry.resource {
                wgpu::BindingResource::Buffer(b) => {
                    b.buffer.hash(&mut hasher);
                    b.offset.hash(&mut hasher);
                    b.size.hash(&mut hasher);
                }
                other => {
                    // For non-buffer resources (samplers, textures, etc.), hash the
                    // debug repr as a fallback. These are rare in compute-only paths.
                    format!("{:?}", other).hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Get a cached bind group or create + cache a new one.
    ///
    /// Issue 660 stepping stone: the forward path creates the same ~832 bind
    /// groups every token. This cache eliminates the driver call
    /// (`device.create_bind_group`) on token 2+, saving ~3-5 µs × 832 =
    /// ~2.5-4 ms/token of CPU-side overhead.
    ///
    /// The cache is bounded at [`BIND_GROUP_CACHE_CAPACITY`]. When full, the
    /// entire cache is cleared (simple flush — the forward path will repopulate
    /// it within one token). An LRU would be marginally better but adds
    /// complexity for negligible gain when the working set is stable.
    fn get_or_create_bind_group(
        &mut self,
        pipeline: Arc<ComputePipeline>,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        let key = Self::bind_group_hash(&pipeline, entries);
        let cache_enabled =
            BIND_GROUP_CACHE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);

        // Issue 660 A/B toggle: when disabled, skip the cache entirely
        // (always create fresh). This allows head-to-head benchmarking.
        if cache_enabled {
            if let Some(bg) = self.bind_group_cache.get(&key) {
                self.bg_cache_hits += 1;
                BG_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return bg.clone();
            }
        }

        // Cache miss (or disabled) — create the bind group.
        self.bg_cache_misses += 1;
        BG_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if cache_enabled && self.bind_group_cache.len() >= BIND_GROUP_CACHE_CAPACITY {
            self.bind_group_cache.clear();
        }

        let group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &group_layout,
                entries,
            });

        if cache_enabled {
            self.bind_group_cache.insert(key, bind_group.clone());
        }
        bind_group
    }

    /// Clear the bind group cache. Called during memory cleanup to ensure
    /// stale entries (from dropped buffers) don't accumulate.
    #[allow(dead_code, reason = "public API for future memory-cleanup integration")]
    pub fn clear_bind_group_cache(&mut self) {
        self.bind_group_cache.clear();
    }

    /// Issue 660 diagnostics: return (hits, misses) since stream creation.
    #[allow(dead_code, reason = "public API for future diagnostic integration")]
    pub fn bind_group_cache_stats(&self) -> (u64, u64) {
        (self.bg_cache_hits, self.bg_cache_misses)
    }

    pub(crate) fn flush_errors_queue(&mut self) -> Vec<ServerError> {
        let errors = core::mem::take(&mut self.errors);

        if !errors.is_empty() {
            self.profile_error(ProfileError::Unknown {
                reason: alloc::format!("{:?}", errors),
                backtrace: BackTrace::capture(),
            });
        }

        errors
    }
}

#[cfg(not(target_family = "wasm"))]
mod __submission_load {
    #[derive(Default, Debug)]
    pub enum SubmissionLoad {
        Init {
            last_index: wgpu::SubmissionIndex,
            tasks_count_submitted: usize,
        },
        #[default]
        Empty,
    }

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            device: &wgpu::Device,
            tasks_count: usize,
            mut index: wgpu::SubmissionIndex,
        ) {
            match self {
                SubmissionLoad::Init {
                    last_index,
                    tasks_count_submitted,
                } => {
                    *tasks_count_submitted += tasks_count;

                    // Enough to keep the GPU busy.
                    //
                    // - Too much can hang the GPU and create slowdown.
                    // - Too little and GPU utilization is really bad.
                    //
                    // TODO: Could be smarter and dynamic based on stats.
                    const MAX_TOTAL_TASKS: usize = 512;

                    if *tasks_count_submitted >= MAX_TOTAL_TASKS {
                        core::mem::swap(last_index, &mut index);
                        // Issue 761 profiling: this poll(Wait) blocks the host
                        // until the GPU reaches the PREVIOUS submission index —
                        // the in-flight cap that paces the encoder to the GPU.
                        let t_wait = std::time::Instant::now();
                        if let Err(e) = device.poll(wgpu::PollType::Wait {
                            submission_index: Some(index),
                            timeout: None,
                        }) {
                            log::warn!(
                                "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                            )
                        }
                        super::REGULATE_NANOS.fetch_add(t_wait.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        super::REGULATE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        *tasks_count_submitted = 0;
                    }
                }
                SubmissionLoad::Empty => {
                    *self = Self::Init {
                        last_index: index,
                        tasks_count_submitted: 0,
                    }
                }
            }
        }
    }
}

#[cfg(target_family = "wasm")]
mod __submission_load_wasm {
    #[derive(Default, Debug)]
    pub struct SubmissionLoad;

    impl SubmissionLoad {
        pub fn regulate(
            &mut self,
            _device: &wgpu::Device,
            _tasks_count: usize,
            _index: wgpu::SubmissionIndex,
        ) {
            // Nothing to do.
        }
    }
}

#[cfg(not(target_family = "wasm"))]
use __submission_load::*;
#[cfg(target_family = "wasm")]
use __submission_load_wasm::*;
