//! GPU-side matrix transpose kernel (Issue 402 Phase 9h).
//!
//! Transposes a matrix on the GPU using a tiled WGSL compute kernel.
//! Used to eliminate the CPU transpose bottleneck for the LM head weight
//! matrix (640 MB at 0.40B scale), which was ~674 ms/step on M3 Max due
//! to the extreme aspect ratio (163840 × 1024) causing scattered cache
//! writes on CPU.
//!
//! ## How it works
//!
//! 1. Raw (non-transposed) weight data is uploaded via `queue.write_buffer`
//!    into a staging GPU buffer (sequential write — fast).
//! 2. A tiled transpose compute kernel reads from the staging buffer and
//!    writes the transposed data into the target GPU buffer at the correct
//!    offset.
//! 3. The target buffer is the same CubeCL-managed pooled memory that
//!    `WeightBufferSlot` caches — the kernel writes directly into it.

use std::sync::Arc;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, Buffer, BufferBinding, BufferBindingType, CommandEncoder, ComputePipeline,
    ComputePassDescriptor, Device, PipelineLayoutDescriptor, Queue, ShaderStages,
};

/// Parameters for the transpose kernel (16-byte aligned for WGSL uniform).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeUniform {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU transpose kernel manager — holds the compute pipeline + bind group
/// layout + device reference. Created once at boot.
pub struct GpuTranspose {
    device: Arc<Device>,
    pipeline: ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Reusable uniform buffer (updated via write_buffer before each dispatch).
    uniform_buffer: wgpu::Buffer,
}

impl GpuTranspose {
    /// Create the transpose kernel pipeline from the device.
    pub fn new(device: Arc<Device>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transpose.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels/transpose.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("transpose_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("transpose_pll"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transpose_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("transpose_matrix"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transpose_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            pipeline,
            bind_group_layout,
            uniform_buffer,
        }
    }

    /// Dispatch the transpose kernel: reads `input` [rows × cols] and writes
    /// the transposed data into `output` [cols × rows] at the given byte offset.
    ///
    /// Records a compute pass on the encoder. The caller submits the encoder.
    ///
    /// This is the per-call path — creates a bind group + writes the uniform
    /// every call. For hot loops with fixed buffers/dims, use
    /// [`create_dispatch`] + [`dispatch_cached`] instead.
    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        input: &Buffer,
        output: &Buffer,
        output_offset: u64,
        rows: usize,
        cols: usize,
    ) {
        let uniform = TransposeUniform {
            rows: rows as u32,
            cols: cols as u32,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("transpose_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: input,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: output,
                        offset: output_offset,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("transpose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = rows.div_ceil(16);
        let wg_y = cols.div_ceil(16);
        pass.dispatch_workgroups(wg_x as u32, wg_y as u32, 1);
    }

    /// Create a cached dispatch state for a fixed (input, output, offset, dims)
    /// tuple (Phase 9i). The bind group + uniform are created once here;
    /// [`dispatch_cached`] reuses them without per-call overhead.
    ///
    /// Use this when the input/output buffers + dimensions don't change
    /// between calls (only the data in the input buffer changes). This is
    /// the case for the LM head transpose in the training loop.
    pub fn create_dispatch(
        &self,
        queue: &Queue,
        input: &Buffer,
        output: &Buffer,
        output_offset: u64,
        rows: usize,
        cols: usize,
    ) -> GpuTransposeDispatch {
        // Write the uniform ONCE — rows/cols are constant across steps.
        let uniform = TransposeUniform {
            rows: rows as u32,
            cols: cols as u32,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("transpose_bg_cached"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: input,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: output,
                        offset: output_offset,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        GpuTransposeDispatch {
            bind_group,
            wg_x: rows.div_ceil(16) as u32,
            wg_y: cols.div_ceil(16) as u32,
        }
    }

    /// Dispatch the transpose kernel using a pre-created bind group (Phase 9i).
    ///
    /// No per-call bind group creation or uniform write — just begin the
    /// compute pass + dispatch. Use [`create_dispatch`] to create the
    /// cached dispatch state.
    pub fn dispatch_cached(
        &self,
        encoder: &mut CommandEncoder,
        cached: &GpuTransposeDispatch,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("transpose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &cached.bind_group, &[]);
        pass.dispatch_workgroups(cached.wg_x, cached.wg_y, 1);
    }

    /// Get the device reference (for creating command encoders).
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// Cached dispatch state for the transpose kernel (Phase 9i).
///
/// Holds a pre-created bind group + workgroup dimensions for a fixed
/// (input, output, offset, dims) tuple. Created once via
/// [`GpuTranspose::create_dispatch`]; reused across all subsequent steps.
///
/// The bind group references specific `wgpu::Buffer` objects. The caller
/// MUST ensure those buffers remain alive for the lifetime of this struct
/// (they do in the training loop — the staging buffer is owned by the
/// handles struct, the target buffer by the CubeCL pool).
pub struct GpuTransposeDispatch {
    bind_group: wgpu::BindGroup,
    wg_x: u32,
    wg_y: u32,
}
