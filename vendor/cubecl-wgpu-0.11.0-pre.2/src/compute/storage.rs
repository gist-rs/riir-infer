use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::num::NonZeroU64;
use wgpu::BufferUsages;

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, WgpuMemory>,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
    #[allow(unused, reason = "keep it simple")]
    vk_storage: bool,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
///
/// riir-ai vendor patch (Issue 761 follow-up 2): `Clone` — cloning is a
/// `wgpu::Buffer` refcount bump (an Arc-like handle), which the uniform
/// content cache uses to hand out pinned cached resources.
#[derive(new, Debug, Clone)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
}

/// The memory that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuMemory {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
}

impl WgpuResource {
    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource<'_> {
        // wgpu enforces 4-byte alignment for buffer binding sizes per the WebGPU spec.
        // - https://github.com/gfx-rs/wgpu/pull/8041
        //
        // This padding is safe because:
        // 1. In checked mode, bounds checks prevent reading beyond the logical size.
        // 2. In unchecked mode, OOB access is already undefined behavior.
        //
        // For zero-sized resources, pass None (use rest of buffer from offset).
        // The allocator guarantees the buffer is at least MIN_BUFFER_SIZE bytes.
        let size = NonZeroU64::new(self.size.next_multiple_of(4));

        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size,
        };
        wgpu::BindingResource::Buffer(binding)
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(
        mem_alignment: usize,
        device: wgpu::Device,
        usages: BufferUsages,
        vk_storage: bool,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            device,
            buffer_usages: usages,
            mem_alignment,
            vk_storage,
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let memory = self.memory.get(&handle.id).unwrap();
        WgpuResource::new(
            memory.buffer.clone(),
            memory.address,
            handle.offset(),
            handle.size(),
        )
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let alloc_size = size.max(MIN_BUFFER_SIZE);

        let memory = self.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_size,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        })?;

        self.memory.insert(id, memory);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.memory.remove(&id);
    }

    fn flush(&mut self) {
        // We don't wait for dealloc
    }
}

impl WgpuStorage {
    #[cfg(feature = "spirv")]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        if self.vk_storage {
            // wgpu currently doesn't expose this, even though it's used internally for acceleration
            // structures. While we could use the acceleration structure input flag, that would
            // then require ray tracing to be supported. So we need to allocate manually, then import
            // the native buffer into wgpu using `from_raw_managed`.
            // This actually skips some of the buffer batching stuff we don't really want in `gpu_allocator`.
            let (buffer, addr) = crate::backend::vulkan::create_storage_buffer(&self.device, desc)?;
            Ok(WgpuMemory::new(buffer, NonZeroU64::new(addr)))
        } else {
            self.create_buffer_checked(desc)
        }
    }

    #[cfg(not(feature = "spirv"))]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        self.create_buffer_checked(desc)
    }

    /// riir-ai Issue 714 follow-up (Plan 343 T1.7, 2026-08-26): `Device::create_buffer` is
    /// NOT fallible — on any failure wgpu routes the error through the device's error sink,
    /// where, with no matching scope on this thread, the DEFAULT handler PANICS the process
    /// (`wgpu error: ...` at the create call), and hands back an INVALID-STUB buffer whose
    /// first consumer then dies inside `map_async` (observed live as `Buffer::map_async —
    /// Buffer with '' label is invalid`, rc=101, the Plan 343 seam check under device-memory
    /// pressure). Which error class gets swallowed en route depends on which scopes happen
    /// to be open on the allocating thread (e.g. `sync()`'s Internal scope), so the stub can
    /// enter a memory pool silently. Wrap the call in ALL THREE error-scope filters so a
    /// failed creation surfaces HERE as an `Err` the pool already knows how to propagate —
    /// the same contract the `vk_storage` arm above provides natively. The scope guards
    /// auto-pop on drop and are `!Send`, so this is thread-correct by construction; wgpu's
    /// pop "takes effect immediately", so the immediately-polled future is already resolved
    /// for creation-time errors.
    fn create_buffer_checked(
        &self,
        desc: &wgpu::BufferDescriptor<'_>,
    ) -> Result<WgpuMemory, IoError> {
        let scope_oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let scope_validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let scope_internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let buffer = self.device.create_buffer(desc);
        let captured = [
            cubecl_environment::future::block_on(scope_internal.pop()),
            cubecl_environment::future::block_on(scope_validation.pop()),
            cubecl_environment::future::block_on(scope_oom.pop()),
        ]
        .into_iter()
        .find_map(|scope| scope);
        if let Some(err) = captured {
            eprintln!(
                "[cubecl-wgpu/riir-ai 714] buffer creation failed ({} bytes): {err}",
                desc.size
            );
            return Err(match err {
                wgpu::Error::OutOfMemory { .. } => IoError::OutOfMemory {
                    size: desc.size,
                    backtrace: BackTrace::capture(),
                },
                other => IoError::Unknown {
                    description: other.to_string(),
                    backtrace: BackTrace::capture(),
                },
            });
        }
        Ok(WgpuMemory::new(buffer, None))
    }
}
