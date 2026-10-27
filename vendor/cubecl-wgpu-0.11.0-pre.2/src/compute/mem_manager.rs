use crate::{WgpuResource, WgpuStorage};
use cubecl_core::{
    MemoryConfiguration,
    server::{Binding, IoError},
};
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryBinding, ManagedMemoryHandle, MemoryAllocationMode, MemoryHandle,
        MemoryManagement, MemoryManagementOptions,
    },
    storage::ComputeStorage,
};
use std::collections::HashMap;
use wgpu::BufferUsages;

// ─────────────────────────────────────────────────────────────────────────
// Issue 761 follow-up 2 — uniform-content cache.
//
// After the params-handle cache (riir-gpu `params_handle_cache`) removed the
// per-launch params Write-tasks, the remaining ~852 `queue.write_buffer`s
// per decode token are the METADATA uniforms that `into_resources` creates
// for every dispatch carrying scalar args (on Metal/WGSL there are no
// immediate push constants, so dims/strides/eps ride a uniform buffer).
// Each is a pool reserve + a queue write + a pending-write counter, and
// every 64 pending writes force a submit + `device.poll(Wait)` — a hard
// host block on GPU progress (~13 blocks/token at 852 writes/token;
// ~4-8 ms/token of host cost).
//
// ~all of these metadata payloads are byte-identical every token (they
// depend only on model config). Measured (Bench 762): ~844 hits / ~1 miss
// per forward call — the position-carrying values (rope pos, kv-append
// pos, attention extent, embedding row) ride the PARAMS buffer, not the
// metadata uniform, so even those stay static here. This cache keys the uniform's CONTENT (`blake3` → u64) and pins the
// reservation — the entry holds the `ManagedMemoryHandle`, so the
// flush-time `release_uniforms` (which only clears the per-flush
// keep-alive `Vec`) can never free a cached slice. A hit returns the
// pinned resource with zero queue writes.
//
// Correctness contract:
// - Only VALUE-carrying uniforms are cached (the metadata binding —
//   `MetadataBindingInfo::data` is scalars + metadata values, never buffer
//   addresses). The Vulkan `address_buffer` (live GPU virtual addresses of
//   this dispatch's buffers) is deliberately NOT cached: a hit after those
//   buffers were freed would dispatch on stale addresses.
// - wgpu's queue timeline orders the original `write_buffer` before every
//   later consumer's encoder submit, and the pinned handle prevents the
//   ExclusivePages pool from re-using the slice, so the bytes a hit serves
//   are exactly the bytes the caller hashed.
// - Kernels bind the metadata uniform read-only; nothing writes it after
//   creation.
//
// DEFAULT ON since Bench 771 (promoting the Bench 762 GOAT): env
// `RIIR_UNIFORM_CACHE=0|off|false|no` is the KILL-SWITCH (disabled builds
// take the legacy path, byte-identical).
// `set_uniform_cache_override(Some(bool))` overrides for A/B benches.
// ─────────────────────────────────────────────────────────────────────────

/// Uniform-content cache capacity before a wholesale clear (the bind-group
/// cache / params-cache pattern — repopulation costs one legacy token). The
/// decode path holds ~840 static keys + ~1 volatile key/token (measured,
/// Bench 762), so 4096 covers a very long generation between evictions
/// (0 evicts across the whole bench).
const UNIFORM_CACHE_CAP: usize = 4096;

/// Issue 761 FU2 diagnostics: uniform-content cache hit/miss/eviction
/// counters (lock-free reads from benchmark harnesses). Reset via
/// [`reset_uniform_cache_stats`].
pub static UNIFORM_CACHE_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static UNIFORM_CACHE_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static UNIFORM_CACHE_EVICTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Reset the uniform-content cache diagnostic counters.
pub fn reset_uniform_cache_stats() {
    UNIFORM_CACHE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    UNIFORM_CACHE_MISSES.store(0, std::sync::atomic::Ordering::Relaxed);
    UNIFORM_CACHE_EVICTS.store(0, std::sync::atomic::Ordering::Relaxed);
}

mod uniform_cache_state {
    use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

    // Tri-state: 0 = follow env (cached on first read), 1 = force on, 2 = force off.
    // The override exists so one binary can A/B both arms for the GOAT gate
    // (Bench 762) without re-reading the env on the hot path.
    static OVERRIDE: AtomicU8 = AtomicU8::new(0);
    static ENV_ENABLED: AtomicBool = AtomicBool::new(false);
    static ENV_READ: AtomicBool = AtomicBool::new(false);

    fn env_enabled() -> bool {
        if !ENV_READ.swap(true, Ordering::Relaxed) {
            // DEFAULT ON (Bench 771, promoting the Bench 762 GOAT): an
            // explicit RIIR_UNIFORM_CACHE=0|off|false|no disables.
            let enabled = !matches!(
                std::env::var("RIIR_UNIFORM_CACHE")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "off" | "false" | "no"
            );
            ENV_ENABLED.store(enabled, Ordering::Relaxed);
        }
        ENV_ENABLED.load(Ordering::Relaxed)
    }

    pub fn enabled() -> bool {
        match OVERRIDE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => env_enabled(),
        }
    }

    pub fn set_override(enabled: Option<bool>) {
        OVERRIDE.store(
            match enabled {
                Some(true) => 1,
                Some(false) => 2,
                None => 0,
            },
            Ordering::Relaxed,
        );
    }
}

/// Force the uniform-content cache on/off for benches + tests (`None` =
/// follow the `RIIR_UNIFORM_CACHE` env). One binary, two arms — the GOAT
/// A/B pattern.
pub fn set_uniform_cache_override(enabled: Option<bool>) {
    uniform_cache_state::set_override(enabled);
}

/// blake3 content key, truncated to u64. Collision odds across ~4k live keys
/// are ~1e-10, and a collision would surface immediately as wrong-metadata
/// reads in the G1 bit-identity gate (the params-cache keying pattern).
fn uniform_cache_key(data: &[u8]) -> u64 {
    u64::from_le_bytes(blake3::hash(data).as_bytes()[..8].try_into().unwrap())
}

/// A pinned uniform. `handle` keeps the ExclusivePages slice reserved for
/// the entry's lifetime (the flush-time `release_uniforms` never sees it);
/// `resource` is what a hit hands out (a `wgpu::Buffer` refcount bump).
#[derive(Debug)]
struct CachedUniform {
    /// Never read — held for its keep-alive (dropping the last handle
    /// returns the slice to the ExclusivePages uniform pool).
    #[allow(dead_code, reason = "pin: the handle's existence reserves the slice")]
    handle: ManagedMemoryHandle,
    resource: WgpuResource,
}

#[derive(Debug)]
pub(crate) struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    memory_uniforms: MemoryManagement<WgpuStorage>,
    memory_pool_staging: MemoryManagement<WgpuStorage>,
    uniforms: Vec<ManagedMemoryHandle>,
    /// Issue 761 FU2: content-keyed pinned uniforms (see the module-level
    /// comment). Distinct from `uniforms`: these survive
    /// [`Self::release_uniforms`] — the entry owns the keep-alive handle.
    uniform_cache: HashMap<u64, CachedUniform>,
}

impl WgpuMemManager {
    pub(crate) fn new(
        device: wgpu::Device,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        use_vulkan_compiler: bool,
    ) -> Self {
        // Allocate storage & memory management for the main memory buffers. Any calls
        // to empty() or create() with a small enough size will be allocated from this
        // main memory pool.
        //
        // `memory_config` (which honors any programmatic pool override) shapes
        // the main pool only; the staging and uniforms pools below have
        // deliberate configurations that must not be overridden.
        let memory_main = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT,
                use_vulkan_compiler,
            ),
            &memory_properties,
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );

        let memory_staging = MemoryManagement::from_configuration(
            WgpuStorage::new(
                wgpu::COPY_BUFFER_ALIGNMENT as usize,
                device.clone(),
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                false,
            ),
            &memory_properties,
            // Unfortunately, we can't reuse a different part of a buffer for different reads, so we
            // can't have a single binding with multiple slices allocated.
            MemoryConfiguration::ExclusivePages,
            logger.clone(),
            MemoryManagementOptions::new("Staging CPU Memory").mode(MemoryAllocationMode::Auto),
        );

        // TODO: In the future this should not need STORAGE, if cube writes out all
        // uniforms as having <uniform> usage.
        let memory_uniforms = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
                use_vulkan_compiler,
            ),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
            logger,
            MemoryManagementOptions::new("Uniform GPU Memory").mode(MemoryAllocationMode::Auto),
        );

        Self {
            memory_pool: memory_main,
            memory_pool_staging: memory_staging,
            memory_uniforms,
            uniforms: vec![],
            uniform_cache: HashMap::new(),
        }
    }

    pub(crate) fn bind(
        &mut self,
        old: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
    ) -> Result<(), IoError> {
        // riir-ai Issue 714 vendor patch: fallible bind — the internal failure
        // previously `.unwrap()`ed on the DSD server thread and killed the
        // process-global server. `initialize_memory` routes the error into the
        // stream error sink instead.
        self.memory_pool.bind(old, new, 0)
    }

    pub(crate) fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        match self.memory_pool.reserve(size) {
            Ok(handle) => Ok(handle),
            Err(err) => Err(err),
        }
    }

    pub(crate) fn reserve_staging(
        &mut self,
        size: u64,
    ) -> Result<(WgpuResource, ManagedMemoryBinding), IoError> {
        let handle = self.memory_pool_staging.reserve(size)?;
        let binding = MemoryHandle::binding(handle);
        let resource = self
            .memory_pool_staging
            .get_resource(binding.clone(), None, None)
            .unwrap();

        Ok((resource, binding))
    }

    pub(crate) fn get_resource(&mut self, binding: Binding) -> Result<WgpuResource, IoError> {
        self.memory_pool
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// Reserve a uniform slice; returns the resource + the reservation
    /// handle. Private core — callers decide the keep-alive strategy.
    fn reserve_uniform_slice(&mut self, size: u64) -> (WgpuResource, ManagedMemoryHandle) {
        let slice = self
            .memory_uniforms
            .reserve(size)
            .expect("Must have enough memory for a uniform");
        let storage_handle = self
            .memory_uniforms
            .get_storage(slice.clone().binding())
            .expect("Failed to find storage!");
        let resource = self.memory_uniforms.storage().get(&storage_handle);
        (resource, slice)
    }

    /// Reserve a uniform whose content is NOT cacheable — the Vulkan
    /// `address_buffer` (live GPU addresses of this dispatch's buffers).
    /// Keep-alive rides the per-flush `uniforms` `Vec`, exactly as before
    /// the uniform-content cache existed.
    pub(crate) fn reserve_uniform_uncached(&mut self, data: &[u8]) -> WgpuResource {
        let (resource, slice) = self.reserve_uniform_slice(data.len() as u64);
        // Keep track of this uniform until it is released.
        self.uniforms.push(slice);
        resource
    }

    /// Issue 761 FU2: content-cache lookup for a value-carrying uniform.
    /// `Some(resource)` = hit — the caller skips the pool reserve AND the
    /// queue write entirely. Returns `None` when the cache is disabled.
    pub(crate) fn lookup_uniform(&mut self, data: &[u8]) -> Option<WgpuResource> {
        if data.is_empty() || !uniform_cache_state::enabled() {
            return None;
        }
        let hit = self
            .uniform_cache
            .get(&uniform_cache_key(data))
            .map(|entry| entry.resource.clone());
        if hit.is_some() {
            UNIFORM_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            UNIFORM_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        hit
    }

    /// Reserve a uniform for a value-carrying payload. When the content
    /// cache is active the reservation is pinned INSIDE the cache entry
    /// (not the per-flush `uniforms` `Vec`, which [`Self::release_uniforms`]
    /// clears at every flush) so the cached bytes stay valid for the
    /// stream's lifetime. When inactive, this is the legacy path.
    pub(crate) fn reserve_uniform(&mut self, data: &[u8]) -> WgpuResource {
        let (resource, slice) = self.reserve_uniform_slice(data.len() as u64);
        if data.is_empty() || !uniform_cache_state::enabled() {
            // Legacy keep-alive: released at the next flush.
            self.uniforms.push(slice);
            return resource;
        }
        if self.uniform_cache.len() >= UNIFORM_CACHE_CAP {
            // Wholesale clear (the bind-group-cache pattern). Dropping the
            // entries' handles frees their pages back to the uniform pool.
            self.uniform_cache.clear();
            UNIFORM_CACHE_EVICTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.uniform_cache.insert(
            uniform_cache_key(data),
            CachedUniform {
                handle: slice,
                resource: resource.clone(),
            },
        );
        resource
    }

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
    }

    pub(crate) fn mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }

    /// Rebuild the main pool with a new layout. Returns `false` (keeping the
    /// old layout, with a log) when something is still live in it. The staging
    /// and uniforms pools keep their deliberate configurations.
    pub(crate) fn configure_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        props: &MemoryDeviceProperties,
    ) -> bool {
        self.memory_pool.configure(config, props)
    }

    pub(crate) fn release_uniforms(&mut self) {
        self.uniforms.clear();
    }
}
