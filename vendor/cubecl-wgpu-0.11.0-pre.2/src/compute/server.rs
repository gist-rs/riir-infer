use std::marker::PhantomData;

use super::storage::{WgpuResource, WgpuStorage};
use crate::WgpuCompiler;
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use alloc::sync::Arc;
use cubecl_common::pool::LeasePool;
use cubecl_common::{
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use cubecl_core::server::{Binding, StreamErrorMode};
use cubecl_core::zspace::Shape;
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
#[cfg(feature = "spirv")]
use cubecl_environment::persistence::Store;
use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
#[cfg(feature = "spirv")]
use cubecl_runtime::compiler::{KernelCacheKey, compilation_store, store_compiled};
use cubecl_runtime::memory_management::{ManagedMemoryHandle, MemoryUsage, SharedMemoryBindings};
use cubecl_runtime::{
    compiler::{CompilationCache, CubeTask},
    config::{CubeClRuntimeConfig, RuntimeConfig},
    dry_run::LaunchMode,
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::ComputeServer,
    storage::ManagedResource,
    stream::scheduler::{
        SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy,
        SchedulerStreamBackend,
    },
    validation::{validate_cube_dim, validate_units},
};
use wgpu::ComputePipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsTransfer {
    Immediate,
    Uniform,
}

/// Compiler kind and info used when compiling a specific kernel. Used to determine parameter passing strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerInfo {
    Vulkan { params_transfer: ParamsTransfer },
    Metal,
    WGSL,
    None,
}

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<C: WgpuCompiler> {
    pub(crate) device: wgpu::Device,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
    /// The pipelines built so far, in front of the SPIR-V store when there is
    /// one.
    pipelines: CompilationCache<KernelId, (Arc<ComputePipeline>, CompilerInfo)>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache: Option<Store<(u64, KernelCacheKey), cubecl_spirv::SpirvCacheEntry>>,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
    /// Reusable buffers for the cross-stream input bindings of each launch.
    shared_bindings_pool: LeasePool<SharedMemoryBindings>,
    _compiler: PhantomData<C>,
}

impl<C: WgpuCompiler> ServerCommunication for WgpuServer<C> {
    const SERVER_COMM_ENABLED: bool = false;
}

impl<C: WgpuCompiler> WgpuServer<C> {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        #[cfg(feature = "spirv")]
        let adapter_info = device.adapter_info();
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            backend,
            tasks_max,
            utilities.logger.clone(),
            compilation_options.supports_vulkan_compiler,
        );

        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        #[cfg(feature = "spirv")]
        let spirv_cache = compilation_store(
            "vulkan",
            format!("spirv_{}_{}", adapter_info.vendor, adapter_info.device),
        );

        // WGSL is compiled by the driver on every run, so without the SPIR-V
        // store there is nothing persisted for a switch to invalidate.
        #[cfg(feature = "spirv")]
        let pipelines = CompilationCache::mirroring(&spirv_cache);
        #[cfg(not(feature = "spirv"))]
        let pipelines = CompilationCache::unbound();

        Self {
            compilation_options,
            streams_pool: Vec::new(),
            device,
            pipelines,
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            #[cfg(feature = "spirv")]
            spirv_cache,
            backend,
            utilities: Arc::new(utilities),
            shared_bindings_pool: LeasePool::with_capacity(tasks_max * max_streams as usize),
            _compiler: PhantomData,
        }
    }

    fn prepare_bindings(
        &mut self,
        bindings: KernelArguments,
        compiler_info: CompilerInfo,
    ) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.buffers.len());

        for b in bindings.buffers.into_iter() {
            let stream = self.scheduler.stream(&b.stream);
            let resource = stream.mem_manage.get_resource(b)?;
            resources.push(resource);
        }

        Ok(BindingsResource {
            resources,
            info: bindings.info,
            compiler_info,
        })
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        mode: ExecutionMode,
    ) -> Result<(Arc<ComputePipeline>, CompilerInfo), LaunchError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        let definition = kernel.define();
        let cached = self.load_cached_pipeline(&kernel_id, &definition, bindings, mode)?;

        if let Some(Ok(pipeline)) = cached {
            self.pipelines.insert(kernel_id, pipeline.clone());
            return Ok(pipeline);
        }

        validate_cube_dim(&self.utilities.properties, &kernel_id)?;
        validate_units(&self.utilities.properties, &kernel_id)?;

        let mut compiler = C::init(self.backend, &self.compilation_options);
        let mut compiled = compiler.compile_kernel(self, kernel, definition, mode)?;

        if self.scheduler.logger.compilation_source_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);

        compiler.validate_ir(&compiled.repr, &self.utilities.properties)?;
        let (compiler_info, auto_repr) = compiler.normalize_repr(compiled.repr);
        let repr = auto_repr.as_ref().map(|r| r.as_ref());

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compiled.source).expect("should write to file");
        //     let status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //             "-w",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     if !status.success() {
        //         println!("SOURCE:\n{}", compiled.source);
        //         std::process::exit(status.code().unwrap());
        //     }
        // }

        let module = self.create_module(
            &compiled.entrypoint_name,
            kernel_id.cube_dim,
            repr,
            &compiled.source,
            mode,
        )?;
        let pipeline = self.create_pipeline(&compiled.entrypoint_name, repr, module, bindings);
        self.pipelines
            .insert(kernel_id.clone(), (pipeline.clone(), compiler_info));

        #[cfg(feature = "spirv")]
        if let Some(Err(key)) = cached
            && let Some(crate::AutoRepresentation::SpirV(kernel)) = auto_repr
        {
            let cache = self.spirv_cache.as_mut().unwrap();
            store_compiled(
                cache,
                key,
                cubecl_spirv::SpirvCacheEntry::new(compiled.entrypoint_name, kernel),
            );
        }

        Ok((pipeline, compiler_info))
    }
}

impl<C: WgpuCompiler> ComputeServer for WgpuServer<C> {
    type Kernel = Box<dyn CubeTask<C>>;
    type Storage = WgpuStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        // riir-ai Issue 714 vendor patch: a device-memory OOM must fail only the
        // allocation's owner, not the process-global server. The upstream
        // `unwrap_or_else(panic!)` here ran on the DSD server thread; one failed
        // 85 MB weight upload killed the server and every later cubecl test
        // failed with `DeviceError("Parent device is lost")` (84-test cascade).
        //
        // Route the error into the stream error sink — the same pattern `write`
        // and `launch` below already use. The un-bound handle then fails
        // visibly (`NotFound`) on any later use, and the OOM reason surfaces at
        // this client's next flush/sync. Other clients/tests keep working.
        let reserved = {
            let first = self.scheduler.stream(&stream_id).empty(size);
            match first {
                Ok(val) => val,
                Err(first_err) => {
                    // Self-healing retry (same patch): every stream owns PRIVATE
                    // memory pools, and fully-free pages on a quiet stream (a
                    // client whose work ended without a final cleanup) are never
                    // returned to the driver — multi-client processes (test
                    // binaries) accumulate that residue until allocations OOM
                    // against a device that still holds reclaimable pages. On the
                    // FIRST failure, sweep every stream's pools with an explicit
                    // cleanup (frees only FULLY-free pages — live allocations are
                    // never touched, by construction) and retry once.
                    let stream_ids: alloc::vec::Vec<StreamId> =
                        self.scheduler.stream_ids().collect();
                    for id in stream_ids {
                        self.scheduler
                            .stream(&id)
                            .mem_manage
                            .memory_cleanup(true);
                    }
                    match self.scheduler.stream(&stream_id).empty(size) {
                        Ok(val) => {
                            eprintln!(
                                "[cubecl-wgpu/riir-ai 714] allocation recovered after \
                                 all-stream pool sweep: {size} bytes (first error: \
                                 {first_err})"
                            );
                            val
                        }
                        Err(err) => {
                            // eprintln (not just the sink): the stream error sink
                            // only drains at a non-ignored flush/sync, but reads
                            // flush with `ignore: true` and a panicking test
                            // never reaches sync — without this line the OOM
                            // reason is invisible in test logs.
                            eprintln!(
                                "[cubecl-wgpu/riir-ai 714] device-memory allocation \
                                 failed (server stays alive; handle left unbound): \
                                 {size} bytes: {err}"
                            );
                            let stream = self.scheduler.stream(&stream_id);
                            stream.error(ServerError::Generic {
                                reason: format!(
                                    "failed to reserve {size} bytes of device memory: {err}"
                                ),
                                backtrace: BackTrace::capture(),
                            });
                            return;
                        }
                    }
                }
            }
        };
        // Same blast-radius class: `bind`'s internal failure previously
        // unwrapped and killed the server thread. Surface it through the sink.
        let bind_result = self.scheduler.stream(&stream_id).mem_manage.bind(reserved, memory);
        if let Err(err) = bind_result {
            self.scheduler.stream(&stream_id).error(ServerError::Io(err));
        }
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(&desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }
                    .into())
                });
            }
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams);

        let stream = self.scheduler.stream(&stream_id);
        stream.read_resources(resources)
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (desc, data) in descriptors {
            let stream = self.scheduler.stream(&desc.handle.stream);

            if contiguous_strides(&desc.shape) != desc.strides {
                stream.error(ServerError::Io(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                }));
                return;
            }

            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(r) => r,
                Err(err) => {
                    stream.error(ServerError::Io(err));
                    return;
                }
            };
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
            };

            self.scheduler.register(stream_id, task, &[]);
        }
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.mem_manage.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        args: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        // Issue 761 profiling: time the whole server launch (pipeline cache
        // lookup, LeasePool acquire, prepare_bindings, scheduler register,
        // and the stream enqueue incl. any flush/regulate it triggers). The
        // delta LAUNCH_NANOS − ENQUEUE_EXEC_NANOS is the per-dispatch cost of
        // the launch machinery ABOVE the stream.
        let t_launch = std::time::Instant::now();
        let (pipeline, compiler_info) = match {
            let t_pipe = std::time::Instant::now();
            let r = self.pipeline(kernel, &args, mode);
            super::stream::PIPELINE_NANOS.fetch_add(t_pipe.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            r
        } {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Launch(err));
                return;
            }
        };

        if launch_mode.is_skipped() {
            return;
        }

        self.streams_pool.clear();
        // Reuse a pooled buffer to avoid allocating on every launch; it returns to the pool
        // automatically when the guard drops.
        let mut shared_inputs = self.shared_bindings_pool.acquire();
        // Pin the memory of every input that lives on another stream (released in `WgpuStream::flush`).
        args.buffers.iter().for_each(|b| {
            self.streams_pool.push(b.stream);
            if b.stream != stream_id {
                shared_inputs.push(b.memory.clone());
            }
        });

        let resources = match {
            let t_prep = std::time::Instant::now();
            let r = self.prepare_bindings(args, compiler_info);
            super::stream::PREPARE_NANOS.fetch_add(t_prep.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            r
        } {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Io(err));
                return;
            }
        };
        let task = ScheduleTask::Execute {
            pipeline,
            count,
            resources,
            shared_inputs,
        };

        {
            let t_reg = std::time::Instant::now();
            self.scheduler.register(stream_id, task, &self.streams_pool);
            super::stream::REGISTER_NANOS.fetch_add(t_reg.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        super::stream::LAUNCH_NANOS.fetch_add(
            t_launch.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        super::stream::LAUNCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);

        let stream = self.scheduler.stream(&stream_id);

        stream.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.end_profile(token)
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_usage())
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.scheduler.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        // riir-ai Issue 714 vendor patch: clean ALL streams, not just the
        // caller's. Memory pools are PER-STREAM, so a single-stream cleanup
        // never touches other (possibly quiet) streams' fully-free pages —
        // multi-client processes accumulated that residue until the device
        // OOM'd (the full-suite cascade). Explicit cleanup only deallocates
        // pages observed FULLY free, so live allocations are never touched.
        let stream_ids: alloc::vec::Vec<StreamId> = self.scheduler.stream_ids().collect();
        for id in stream_ids {
            self.scheduler
                .stream(&id)
                .mem_manage
                .memory_cleanup(true);
        }
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }

    fn configure_memory_pools(&mut self, config: MemoryConfiguration, stream_id: StreamId) -> bool {
        // Streams created from now on build their main pool with the new
        // layout; memory is per stream, so already-created streams keep theirs.
        self.scheduler
            .backend_mut()
            .factory()
            .set_gpu_pools(config.clone());
        let (_, props) = self.scheduler.backend_mut().factory().gpu_pools();

        // The calling stream's pools are rebuilt in place (kept, with a log,
        // when something is still live in them).
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.configure_memory_pools(config, &props)
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
