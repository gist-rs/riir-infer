//! Zero-copy metal::tensor GEMM via wgpu MSL passthrough (Issue 657).
//!
//! **REFUTED end-to-end** (0.95× regression, Bench 658 under strict GPU
//! exclusivity with 3-run medians). The matmul2d kernel is 3.72× faster in
//! isolation (Bench 645), but per-dispatch staging copies + command buffer
//! overhead negate the kernel gain. The per-stage breakdown (Bench 658)
//! shows wgpu is slower on EVERY stage — DeltaNet, Attention, AND FFN —
//! confirming genuine overhead, not thermal bias. **Simdgroup cmma is the
//! optimal GEMM kernel on M3 Max Metal.**
//!
//! This module dispatches the metal::tensor matmul2d kernel through wgpu's
//! MSL passthrough API. Weights are pre-copied to dedicated wgpu buffers at
//! boot; input/output are staged via GPU→GPU copies per dispatch.
//!
//! # Why staging copies (not direct binding)
//!
//! CubeCL's memory management sub-allocates handles within pool buffers.
//! wgpu-core enforces that a single `wgpu::Buffer` can't be used as both
//! read-only and read-write storage within one compute dispatch. When the
//! output handle shares the same pool buffer as an input or weight handle,
//! direct binding triggers a validation error.
//!
//! The fix: copy data to/from dedicated staging buffers within the same
//! command buffer. The copies are GPU→GPU (no host round-trip) and chained
//! in the same command stream — the only host-visible cost is the one
//! `queue.submit` per dispatch, which is amortized across the command buffer.
//!
//! # Architecture
//!
//! ```text
//! Boot:
//!   CubeCL weight handles → copy → dedicated wgpu staging buffers (WgpuWeightCache)
//!
//! Per dispatch:
//!   1. encoder.copy_buffer_to_buffer(cubecl_input → input_staging)
//!   2. compute_pass: dispatch matmul2d(weights_staging + input_staging → output_staging)
//!   3. encoder.copy_buffer_to_buffer(output_staging → cubecl_output)
//!   4. queue.submit (shared with CubeCL — ordering is automatic)
//! ```
//!
//! # Feature gate
//!
//! `metal_tensor_gemm` + macOS-only. The MSL source uses `<metal_tensor>` +
//! `<MetalPerformancePrimitives>` headers (Metal 3.1+, macOS 14+).

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, Features,
    PipelineLayoutDescriptor, Queue, ShaderModuleDescriptorPassthrough, ShaderStages,
};

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::gemv_ternary_cubecl::TernaryHandle;

/// Monotonic ID counter for `WgpuWeightCache` instances (bind group cache key).
static WEIGHT_CACHE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The raw MSL source for the wgpu-passthrough ternary GEMM kernel.
const MSL_SOURCE: &str = include_str!("gemm_ternary_metal_wgpu.metal");

/// Packed scalar parameters for the ternary GEMM kernel (binding 5).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GemmParams {
    blocks64: u32,
    groups_per_row: u32,
    k_input_dim: u32,
    m_features: u32,
    p_tokens: u32,
}

/// Workgroup config for the matmul2d kernel (matches MSL constants).
mod kernel_config {
    pub const NRA: u32 = 64;
    pub const NRB: u32 = 128;
}

/// Dedicated wgpu staging buffers for one weight matrix.
///
/// Created once at boot by copying from CubeCL weight handles. These buffers
/// are separate from CubeCL's pool, avoiding wgpu usage-scope conflicts.
/// The `id` field is a monotonic counter used as a bind-group cache key.
pub struct WgpuWeightCache {
    id: u64,
    pos: Buffer,
    neg: Buffer,
    scale: Buffer,
}

impl Clone for WgpuWeightCache {
    fn clone(&self) -> Self {
        // Each clone gets its own id — it holds its own Buffer clones, so a
        // distinct bind group is needed. (In practice, clones are rare; the
        // hot path borrows &WgpuWeightCache from the TernaryHandle.)
        Self {
            id: WEIGHT_CACHE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            pos: self.pos.clone(),
            neg: self.neg.clone(),
            scale: self.scale.clone(),
        }
    }
}

/// Zero-copy metal::tensor GEMM dispatch context (Issue 657).
///
/// Holds the compiled wgpu compute pipeline + bind group layout. Created once
/// at model load time. Weights are cached in `WgpuWeightCache` (one per weight
/// matrix). The `dispatch` method stage input/output via GPU→GPU copies.
///
/// # Steady-state alloc-free dispatch (Issue 657 optimization)
///
/// The staging input/output buffers, the params uniform buffer, and the
/// bind groups are all cached and reused across dispatches with matching
/// dimensions. After warm-up (one `create_buffer` per distinct size + one
/// `create_bind_group` per distinct weight matrix), each dispatch performs
/// zero GPU-memory allocations — only the command encoder + queue.submit.
pub struct MetalTensorWgpuGemm {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    /// Cached staging input buffers, keyed by byte size.
    input_staging_cache: Mutex<HashMap<u64, Buffer>>,
    /// Cached staging output buffers, keyed by byte size.
    output_staging_cache: Mutex<HashMap<u64, Buffer>>,
    /// Cached params uniform buffers, keyed by weight_cache_id.
    ///
    /// Each weight matrix gets its own params buffer (the params are fully
    /// determined by the weight dimensions + p, which is constant per
    /// prefill). The buffer is written once on cache miss; subsequent
    /// dispatches for the same weight reuse it without a write_buffer call.
    /// This avoids the read-write race that sharing a single params buffer
    /// across different weight matrices would create.
    params_cache: Mutex<HashMap<u64, Buffer>>,
    /// Cached bind groups, keyed by (weight_cache_id, input_bytes, output_bytes).
    bind_group_cache: Mutex<HashMap<(u64, u64, u64), BindGroup>>,
    /// Issue 657 profiling: total dispatch count (for overhead analysis).
    dispatch_count: std::sync::atomic::AtomicU64,
}

impl MetalTensorWgpuGemm {
    /// Create a new wgpu-passthrough metal::tensor GEMM context.
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self, String> {
        if !device.features().contains(Features::PASSTHROUGH_SHADERS) {
            return Err(
                "PASSTHROUGH_SHADERS feature not enabled — required for MSL passthrough".to_string(),
            );
        }

        let shader_module = unsafe {
            device.create_shader_module_passthrough(ShaderModuleDescriptorPassthrough {
                label: Some("gemm_ternary_metal_wgpu"),
                msl: Some(std::borrow::Cow::Borrowed(MSL_SOURCE)),
                entry_points: std::borrow::Cow::Borrowed(&[
                    wgpu::PassthroughShaderEntryPoint {
                        name: std::borrow::Cow::Borrowed("gemm_ternary_tensor_wgpu"),
                        workgroup_size: (128, 1, 1),
                    },
                ]),
                ..Default::default()
            })
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_ternary_metal_wgpu_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(20).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("gemm_ternary_metal_wgpu_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("gemm_ternary_metal_wgpu_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("gemm_ternary_tensor_wgpu"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            input_staging_cache: Mutex::new(HashMap::new()),
            output_staging_cache: Mutex::new(HashMap::new()),
            params_cache: Mutex::new(HashMap::new()),
            bind_group_cache: Mutex::new(HashMap::new()),
            dispatch_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Total number of dispatches since creation (Issue 657 profiling).
    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Copy CubeCL weight handles to dedicated wgpu staging buffers.
    ///
    /// Called once per weight matrix at model load time. The dedicated buffers
    /// avoid wgpu usage-scope conflicts (CubeCL pool sub-allocation can
    /// co-locate multiple handles in the same `wgpu::Buffer`).
    pub fn cache_weights(
        &self,
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        w: &TernaryHandle,
    ) -> Result<WgpuWeightCache, String> {
        let (pos_src, pos_off, pos_size) = Self::extract_buffer_info(client, &w.pos_bits_u32)?;
        let (neg_src, neg_off, neg_size) = Self::extract_buffer_info(client, &w.neg_bits_u32)?;
        let (scale_src, scale_off, scale_size) =
            Self::extract_buffer_info(client, &w.group_scale_f32)?;

        // Create dedicated staging buffers.
        let pos_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_weight_pos"),
            size: pos_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let neg_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_weight_neg"),
            size: neg_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scale_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu_weight_scale"),
            size: scale_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy CubeCL → staging in one command buffer.
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("wgpu_weight_cache"),
        });
        encoder.copy_buffer_to_buffer(&pos_src, pos_off, &pos_staging, 0, pos_size);
        encoder.copy_buffer_to_buffer(&neg_src, neg_off, &neg_staging, 0, neg_size);
        encoder.copy_buffer_to_buffer(&scale_src, scale_off, &scale_staging, 0, scale_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Wait for the copy to complete before returning (the caller may
        // dispatch immediately after). On Metal, queue.submit is ordered, so
        // subsequent submits wait for this one automatically — but we poll
        // to be safe (the weight cache is built once at boot).
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        Ok(WgpuWeightCache {
            id: WEIGHT_CACHE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            pos: pos_staging,
            neg: neg_staging,
            scale: scale_staging,
        })
    }

    /// Extract `(wgpu::Buffer, offset, size)` from a CubeCL handle.
    fn extract_buffer_info(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        handle: &Handle,
    ) -> Result<(Buffer, u64, u64), String> {
        // cuda_backend twin: the CubeCL resource is CUDA-shaped (no wgpu
        // buffer to stage from) — report unavailable instead of failing to
        // compile. Callers treat Err as "path disabled" and fall through.
        #[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
        {
            let _ = (client, handle);
            Err(
                "wgpu passthrough extraction unavailable: cuda_backend replaces the wgpu CubeCL runtime"
                    .to_string(),
            )
        }
        #[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
        {
            let managed = client
                .get_resource(handle.clone())
                .map_err(|e| format!("get_resource failed: {e:?}"))?;
            let resource = managed.resource();
            Ok((
                resource.buffer.clone(),
                resource.offset,
                resource.size,
            ))
        }
    }

    /// Dispatch the ternary GEMM via wgpu compute pass with staging copies.
    ///
    /// Flow (all in one command buffer, no host round-trip):
    /// 1. Copy CubeCL input → input staging
    /// 2. Compute pass: matmul2d(weights_staging + input_staging → output_staging)
    /// 3. Copy output staging → CubeCL output
    /// 4. Submit on shared queue
    ///
    /// # Steady-state alloc-free (Issue 657 optimization)
    ///
    /// After warm-up (one `create_buffer` per distinct input/output size + one
    /// `create_bind_group` per distinct weight matrix), each dispatch reuses
    /// cached staging buffers + bind group. The only remaining per-dispatch
    /// allocation is the command encoder (`create_command_encoder`).
    pub fn dispatch(
        &self,
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        weights: &WgpuWeightCache,
        w: &TernaryHandle,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> Result<(), String> {
        // Extract CubeCL input/output buffers.
        let (input_src, input_off, _input_size) = Self::extract_buffer_info(client, input)?;
        let (output_dst, output_off, _output_size) = Self::extract_buffer_info(client, output)?;

        let input_bytes = (p * w.n * std::mem::size_of::<f32>()) as u64;
        let output_bytes = (p * w.m * std::mem::size_of::<f32>()) as u64;

        // ── Cache lookup: staging input buffer (keyed by byte size) ──
        let input_staging = {
            let mut cache = self.input_staging_cache.lock().unwrap();
            cache
                .entry(input_bytes)
                .or_insert_with(|| {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("wgpu_input_staging"),
                        size: input_bytes,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .clone()
        };

        // ── Cache lookup: staging output buffer (keyed by byte size) ──
        let output_staging = {
            let mut cache = self.output_staging_cache.lock().unwrap();
            cache
                .entry(output_bytes)
                .or_insert_with(|| {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("wgpu_output_staging"),
                        size: output_bytes,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .clone()
        };

        // ── Cache lookup: params uniform buffer (keyed by weight_id) ──
        // Params are fully determined by the weight dimensions + p. Within a
        // single prefill, p is constant, so params are fixed per weight matrix.
        // The buffer is written once on cache miss; subsequent dispatches for
        // the same weight reuse it. This avoids the read-write race that a
        // single shared params buffer would create across different weights.
        let params = GemmParams {
            blocks64: w.blocks64 as u32,
            groups_per_row: w.groups_per_row as u32,
            k_input_dim: w.n as u32,
            m_features: w.m as u32,
            p_tokens: p as u32,
        };
        let params_buf = {
            let mut pcache = self.params_cache.lock().unwrap();
            pcache
                .entry(weights.id)
                .or_insert_with(|| {
                    let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("wgpu_params"),
                        size: std::mem::size_of::<GemmParams>() as u64,
                        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    // Write the params once — they don't change across
                    // dispatches for the same weight within one prefill.
                    self.queue
                        .write_buffer(&buf, 0, bytemuck::bytes_of(&params));
                    buf
                })
                .clone()
        };

        // ── Cache lookup: bind group (keyed by weight_id + sizes) ──
        // The bind group references the cached staging buffers + the weight
        // buffers + the per-weight params buffer. Since all are long-lived
        // (held by the caches / WgpuWeightCache), the bind group is safe to
        // reuse across dispatches with matching dimensions.
        let cache_key = (weights.id, input_bytes, output_bytes);
        let bind_group = {
            let mut bg_cache = self.bind_group_cache.lock().unwrap();
            bg_cache
                .entry(cache_key)
                .or_insert_with(|| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("wgpu_gemm_bindgroup"),
                        layout: &self.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &weights.pos,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &weights.neg,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &weights.scale,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &input_staging,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &output_staging,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params_buf,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                        ],
                    })
                })
                .clone()
        };

        // ── Build the command buffer: copy input → dispatch → copy output ──
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gemm_ternary_metal_wgpu"),
        });

        // 1. Copy CubeCL input → staging.
        encoder.copy_buffer_to_buffer(&input_src, input_off, &input_staging, 0, input_bytes);

        // 2. Compute pass: matmul2d.
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("wgpu_gemm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let grid_x = (p as u32).div_ceil(kernel_config::NRB);
            let grid_y = (w.m as u32).div_ceil(kernel_config::NRA);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }

        // 3. Copy output staging → CubeCL output.
        encoder.copy_buffer_to_buffer(&output_staging, 0, &output_dst, output_off, output_bytes);

        // 4. Submit on the shared queue.
        self.queue.submit(std::iter::once(encoder.finish()));

        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }
}
