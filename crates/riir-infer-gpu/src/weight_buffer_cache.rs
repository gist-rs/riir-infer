//! Persistent GPU weight buffer cache (Issue 402 Phase 9).
//!
//! Eliminates the per-step `client.create_from_slice` allocation overhead by
//! pre-allocating GPU buffers once at boot + writing updated weight data
//! in-place via `queue.write_buffer` on subsequent steps.
//!
//! ## Why this exists
//!
//! `ComputeClient::create_from_slice` (CubeCL 0.10.0) does three things per call:
//! 1. `slice.to_vec()` — clones the entire data slice on the CPU side.
//! 2. `server.initialize_memory` — allocates a new handle from the memory pool.
//! 3. `server.write` — copies the data into the handle via `queue.write_buffer`.
//!
//! At ~474 calls/step (237 forward + 237 backward weight matrices), the
//! per-call overhead (clone + allocation + two server-thread channel
//! submissions) dominates the ~4.4s "update" phase of the 10.2s training step.
//!
//! ## How it works
//!
//! The cache holds `(Handle, wgpu::Buffer, offset)` triples. The `Handle` is
//! created once via `client.create_from_slice(data)` (initial allocation +
//! fill). The underlying `wgpu::Buffer` + offset is extracted via
//! `client.get_resource(handle)` — also a one-time cost at boot.
//!
//! On subsequent steps, `write_in_place` calls `queue.write_buffer` directly
//! on the cached `wgpu::Buffer` at the correct offset. This bypasses CubeCL
//! entirely for writes:
//! - No `slice.to_vec()` clone (we write directly from `&[f32]` bytes)
//! - No `initialize_memory` allocation (the buffer is reused)
//! - No server-thread channel submissions (we call wgpu directly)
//!
//! ## Safety
//!
//! The `wgpu::Queue` used here MUST be the same queue CubeCL uses (confirmed
//! by `GpuContext::new_async` passing `queue.clone()` to `init_device`). All
//! operations on the same `wgpu::Queue` are automatically ordered by wgpu's
//! submission ordering — CubeCL kernel launches after our writes will see
//! the updated data.
//!
//! The cached `wgpu::Buffer` references CubeCL-managed pooled memory. We only
//! WRITE to these buffers (never resize or free them), so CubeCL's pool
//! bookkeeping remains consistent. The Handle must stay alive (held by the
//! cache) to prevent CubeCL from recycling the buffer.

use cubecl::prelude::*;
use cubecl::server::Handle;
use crate::cubecl_runtime::ActiveRuntime;
use wgpu::Queue;

/// A persistent GPU weight buffer slot.
///
/// Holds a CubeCL `Handle` (for kernel binding) + the underlying `wgpu::Buffer`
/// + offset (for in-place writes via `queue.write_buffer`). Created once at
///   boot; reused across all subsequent training steps.
///
/// The `wgpu::Buffer` is an `Arc`-backed handle internally, so cloning it is
/// cheap (just increments a ref count). The `Handle` owns the CubeCL pool
/// reservation that backs the buffer.
#[derive(Clone)]
pub struct WeightBufferSlot {
    /// CubeCL handle — bind this to kernel launches.
    pub handle: Handle,
    /// Raw wgpu buffer — write updated data here via `queue.write_buffer`.
    /// `None` until `extract_buffer` is called once (lazy extraction via
    /// `client.get_resource`, which is `submit_blocking`).
    buffer: Option<wgpu::Buffer>,
    /// Byte offset within the wgpu buffer (CubeCL pools may sub-allocate).
    /// `write_in_place` passes this to `queue.write_buffer`.
    offset: u64,
    /// Size in bytes. Must match the data length passed to `write_in_place`.
    size_bytes: usize,
    /// True when constructed via [`from_handle`](Self::from_handle) — the
    /// handle is ALIASED (owned by another struct; the consumer dispatches
    /// through that owner, not through this slot). For such slots the
    /// slow-path fallback (replace the handle via `create_from_slice`)
    /// SEVERS the binding: the consumer keeps reading the original handle's
    /// boot weights forever + one orphaned buffer is allocated per step
    /// (Issue 694 H1). The slow path hard-errors instead.
    aliased: bool,
}

impl WeightBufferSlot {
    /// Allocate a new slot + write initial data via CubeCL.
    ///
    /// Equivalent to `client.create_from_slice(data)` but returns a slot
    /// that can be reused for in-place writes on subsequent steps.
    /// Call `extract_buffer` once after all slots are created to enable
    /// the fast path.
    pub fn from_data(client: &ComputeClient<ActiveRuntime>, data: &[f32]) -> Self {
        let size_bytes = std::mem::size_of_val(data);
        let handle = client.create_from_slice(f32::as_bytes(data));
        Self {
            handle,
            buffer: None,
            offset: 0,
            size_bytes,
            aliased: false,
        }
    }

    /// Create a slot from an existing CubeCL handle (no initial data write).
    ///
    /// Use this when the handle was already populated (e.g., by `update_weights`
    /// during boot). Call `extract_buffer` once to enable the fast path.
    pub fn from_handle(handle: Handle, n_elements: usize) -> Self {
        Self {
            handle,
            buffer: None,
            offset: 0,
            size_bytes: n_elements * std::mem::size_of::<f32>(),
            aliased: true,
        }
    }

    /// Lazily extract the underlying `wgpu::Buffer` from the CubeCL handle.
    ///
    /// This calls `client.get_resource` (a `submit_blocking` server-thread
    /// round-trip). Called once per slot at boot; after this, `write_in_place`
    /// can bypass CubeCL entirely.
    ///
    /// On the `cuda_backend` path this is a no-op — the CUDA `GpuResource` has
    /// no `wgpu::Buffer` to extract, so callers always use the
    /// `create_from_slice` slow path in `write_in_place`.
    #[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
    pub fn extract_buffer(&mut self, client: &ComputeClient<ActiveRuntime>) -> Result<(), String> {
        if self.buffer.is_some() {
            return Ok(());
        }
        let managed = client
            .get_resource(self.handle.clone())
            .map_err(|e| format!("get_resource failed: {e:?}"))?;
        let resource = managed.resource();
        self.buffer = Some(resource.buffer.clone());
        self.offset = resource.offset;
        Ok(())
    }

    /// CUDA backend variant (non-macOS only, Issue 949) — no-op (no
    /// `wgpu::Buffer` to extract).
    #[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
    pub fn extract_buffer(&mut self, _client: &ComputeClient<ActiveRuntime>) -> Result<(), String> {
        Ok(())
    }

    /// Write data into the buffer in-place via `queue.write_buffer`.
    ///
    /// Bypasses CubeCL entirely — no allocation, no server-thread round-trip.
    /// The data length MUST match the slot's size (checked in debug builds).
    ///
    /// If `extract_buffer` hasn't been called yet, falls back to replacing
    /// the handle via `create_from_slice` (the old slow path).
    pub fn write_in_place(
        &mut self,
        client: &ComputeClient<ActiveRuntime>,
        queue: &Queue,
        data: &[f32],
    ) {
        // Release-mode too (Issue 694 H5): a size mismatch writes wrong-length
        // bytes at the slot offset — inside a SHARED pool buffer an oversize
        // write lands in a NEIGHBORING slot's weights silently.
        assert_eq!(
            std::mem::size_of_val(data),
            self.size_bytes,
            "data size mismatch in write_in_place"
        );

        if let Some(ref buffer) = self.buffer {
            // Fast path: write directly into the cached wgpu buffer at the
            // correct offset (CubeCL pools may sub-allocate within a larger
            // buffer; our slot lives at `self.offset`).
            queue.write_buffer(buffer, self.offset, f32::as_bytes(data));
        } else if self.aliased {
            // The handle is owned by another struct (Issue 694 H1): replacing
            // it would sever the binding — the consumer keeps dispatching
            // through the ORIGINAL handle and reads boot weights forever,
            // while a fresh orphaned buffer is allocated here per step.
            // Hard-error instead (this fires on cuda_backend where
            // `extract_buffer` is a no-op — the in-place path is unusable
            // there for aliased slots, by construction).
            panic!(
                "write_in_place: aliased slot's raw buffer unavailable — the \
                 create_from_slice fallback would sever the consumer binding \
                 (Issue 694 H1). On cuda_backend this path is disabled by \
                 construction; elsewhere extraction failed at boot."
            );
        } else {
            // Slow path (own handle — safe to replace): buffer not yet
            // extracted. Allocate + write via CubeCL.
            self.handle = client.create_from_slice(f32::as_bytes(data));
        }
    }

    /// Get the CubeCL handle (for kernel binding).
    #[inline]
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// Slot capacity in f32 elements (`size_bytes / 4`).
    ///
    /// The write plan (per-slot lengths) should be DERIVED from this — not
    /// hand-maintained in a parallel list. A hand-maintained list that drifts
    /// from the boot cache's slot order (e.g. one side conditionally includes
    /// an optional slot and the other doesn't) shifts every subsequent write
    /// into the WRONG buffer (Issue 693 H6).
    #[inline]
    pub fn n_elements(&self) -> usize {
        self.size_bytes / std::mem::size_of::<f32>()
    }

    /// Get the raw wgpu buffer + byte offset (for compute kernel dispatch).
    /// Returns None if `extract_buffer` hasn't been called yet.
    #[inline]
    pub fn buffer_ref(&self) -> Option<(&wgpu::Buffer, u64)> {
        self.buffer.as_ref().map(|b| (b, self.offset))
    }

    /// Ensure the buffer has been extracted (idempotent).
    ///
    /// Returns the extraction error instead of swallowing it (Issue 694 H1:
    /// a swallowed error on an ALIASED slot later routes `write_in_place`
    /// into the severing fallback silently on the CPU-fallback path).
    pub fn extract_buffer_if_needed(
        &mut self,
        client: &ComputeClient<ActiveRuntime>,
    ) -> Result<(), String> {
        if self.buffer.is_none() {
            self.extract_buffer(client)?;
        }
        Ok(())
    }

    /// Write data into the cached buffer using parallel chunked `write_buffer`
    /// calls (Issue 402 Phase 9l pattern).
    ///
    /// Splits the data into N chunks (one per rayon thread, capped at 16) +
    /// writes them concurrently to different offsets of the same buffer.
    /// Metal/Vulkan/DX12 `write_buffer` does NOT serialize concurrent calls
    /// to different offsets, achieving near-linear bandwidth scaling.
    ///
    /// Falls back to a single `write_in_place` when the buffer hasn't been
    /// extracted, the data is too small to benefit, or n_chunks <= 1.
    ///
    /// wgpu requires write_buffer offsets to be aligned to COPY_BUFFER_ALIGNMENT
    /// (256 on most native backends, 4 on WebGPU). We align chunk_bytes to 256.
    pub fn write_in_place_parallel(
        &mut self,
        client: &ComputeClient<ActiveRuntime>,
        queue: &Queue,
        data: &[f32],
    ) {
        // Release-mode too (Issue 694 H5) — see write_in_place.
        assert_eq!(
            std::mem::size_of_val(data),
            self.size_bytes,
            "data size mismatch in write_in_place_parallel"
        );

        let Some(ref buffer) = self.buffer else {
            if self.aliased {
                // Issue 694 H1 — see write_in_place.
                panic!(
                    "write_in_place_parallel: aliased slot's raw buffer \
                     unavailable — the create_from_slice fallback would sever \
                     the consumer binding (Issue 694 H1)."
                );
            }
            // Slow path (own handle — safe to replace): buffer not extracted.
            self.handle = client.create_from_slice(f32::as_bytes(data));
            return;
        };

        parallel_write_buffer(queue, buffer, self.offset, f32::as_bytes(data));
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Parallel chunked write_buffer to a raw wgpu::Buffer (Issue 402 Phase 9l)
// ─────────────────────────────────────────────────────────────────────────

/// Write `data` to `buffer` at `base_offset` using parallel chunked
/// `queue.write_buffer` calls.
///
/// Splits the data into N chunks (one per rayon thread, capped at 16) +
/// writes them concurrently to different offsets of the same buffer.
/// Metal/Vulkan/DX12 `write_buffer` does NOT serialize concurrent calls
/// to different offsets, achieving near-linear bandwidth scaling.
///
/// wgpu requires write_buffer offsets to be aligned to COPY_BUFFER_ALIGNMENT
/// (256 on most native backends, 4 on WebGPU). We align chunk_bytes to 256.
///
/// This is the shared version used by both the forward LM head upload
/// (via `WeightBufferSlot::write_in_place_parallel`) + the backward LM head
/// staging buffer upload.
pub fn parallel_write_buffer(queue: &Queue, buffer: &wgpu::Buffer, base_offset: u64, data: &[u8]) {
    // The chunk SIZE is aligned below, but every chunk OFFSET (base + k·chunk)
    // is only 256-aligned if `base_offset` itself is (Issue 694 H5) — the code
    // silently depends on the pool handing out aligned offsets; a pool
    // granularity change would turn every chunk write into a wgpu validation
    // panic. Pin the contract.
    const WRITE_ALIGN: usize = 256;

debug_assert!(
        base_offset.is_multiple_of(256),
        "parallel_write_buffer base_offset {base_offset} not 256-aligned — chunk offsets would violate COPY_BUFFER_ALIGNMENT"
    );
    let total_bytes = data.len();
    let n_chunks = rayon::current_num_threads().clamp(1, 16);
    let chunk_bytes = total_bytes.div_ceil(n_chunks);
    let chunk_bytes = (chunk_bytes + WRITE_ALIGN - 1) & !(WRITE_ALIGN - 1);

    if n_chunks > 1 && chunk_bytes < total_bytes {
        rayon::scope(|s| {
            for offset in (0..total_bytes).step_by(chunk_bytes) {
                let end = (offset + chunk_bytes).min(total_bytes);
                let chunk = &data[offset..end];
                s.spawn(move |_| {
                    queue.write_buffer(buffer, base_offset + offset as u64, chunk);
                });
            }
        });
    } else {
        queue.write_buffer(buffer, base_offset, data);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Batched write_buffer for groups of WeightBufferSlots (Issue 402 Phase 9f)
// ─────────────────────────────────────────────────────────────────────────

/// Write data into a group of `WeightBufferSlot`s using the MINIMUM number of
/// `queue.write_buffer` calls.
///
/// Slots whose underlying `wgpu::Buffer` is the same AND whose byte ranges
/// are contiguous are coalesced into a single `write_buffer` call. This can
/// reduce ~246 calls/step to a handful when CubeCL's pool places consecutive
/// allocations in the same page.
///
/// ## How it works
///
/// 1. Group slots by `wgpu::Buffer::id()`.
/// 2. Within each group, sort by offset.
/// 3. Coalesce contiguous runs (where `offset[i] + size[i] == offset[i+1]`).
/// 4. For each coalesced run, build ONE staging buffer + ONE `write_buffer`.
/// 5. For non-contiguous slots in the same buffer, fall back to per-slot writes
///    (we don't write padding bytes that might belong to other allocations).
///
/// ## Safety
///
/// Same as `WeightBufferSlot::write_in_place`. We never write bytes outside
/// a slot's declared range — gaps between non-contiguous slots are skipped.
pub fn write_slots_batched(
    slots: &mut [WeightBufferSlot],
    client: &ComputeClient<ActiveRuntime>,
    queue: &Queue,
    data_slices: &[&[f32]],
) {
    debug_assert_eq!(slots.len(), data_slices.len(), "slots/data length mismatch");

    // Diagnostic: count write_buffer calls when RIIR_GPU_WRITE_BATCH_DEBUG is set.
    let debug = std::env::var("RIIR_GPU_WRITE_BATCH_DEBUG").is_ok();
    let mut write_calls = 0usize;
    let mut coalesced_runs = 0usize;

    // ── Pass 1: extract buffers if not yet done + collect buffer refs ──
    // Group slot indices by their parent buffer (linear scan — wgpu::Buffer
    // isn't Hash/Ord, but the number of distinct buffers per struct is small).
    let mut groups: Vec<(wgpu::Buffer, Vec<usize>)> = Vec::new();
    let mut fallback_indices: Vec<usize> = Vec::new();
    for (i, slot) in slots.iter().enumerate() {
        if slot.buffer.is_none() {
            // Slow path: buffer not extracted — defer to per-slot write below.
            fallback_indices.push(i);
            continue;
        }
        let buf = slot.buffer.as_ref().unwrap().clone();
        if let Some((_, indices)) = groups.iter_mut().find(|(b, _)| *b == buf) {
            indices.push(i);
        } else {
            groups.push((buf, vec![i]));
        }
    }

    // Handle slots whose buffers haven't been extracted (slow path).
    for &i in &fallback_indices {
        slots[i].write_in_place(client, queue, data_slices[i]);
        write_calls += 1;
    }

    // ── Pass 2: for each buffer group, coalesce contiguous slots ──
    for (_, indices) in &mut groups {
        // Sort by offset within the buffer.
        indices.sort_by_key(|&i| slots[i].offset);

        // Coalesce contiguous runs.
        let mut run_start = 0;
        while run_start < indices.len() {
            let mut run_end = run_start + 1;
            while run_end < indices.len() {
                let prev = indices[run_end - 1];
                let curr = indices[run_end];
                let prev_end = slots[prev].offset + slots[prev].size_bytes as u64;
                if prev_end == slots[curr].offset {
                    run_end += 1; // contiguous — extend the run
                } else {
                    break; // gap — stop the run
                }
            }

            if run_end - run_start > 1 {
                // Contiguous run of ≥2 slots → coalesce into ONE write.
                let run_indices: Vec<usize> = indices[run_start..run_end].to_vec();
                coalesce_and_write_indexed(slots, queue, data_slices, &run_indices);
                write_calls += 1;
                coalesced_runs += 1;
            } else {
                // Single slot — write directly.
                let idx = indices[run_start];
                slots[idx].write_in_place(client, queue, data_slices[idx]);
                write_calls += 1;
            }
            run_start = run_end;
        }
    }

    if debug {
        eprintln!(
            "write_slots_batched: {} slots → {} write_buffer calls ({} coalesced runs)",
            slots.len(),
            write_calls,
            coalesced_runs
        );
    }
}

/// Coalesce a contiguous run of slots into ONE staging buffer + ONE write.
///
/// All slots MUST share the same parent buffer + be contiguous in GPU memory
/// (each slot's offset + size == next slot's offset). The slot INDICES in the
/// `slots` array need NOT be contiguous — `run_indices` specifies which slots
/// participate in this coalesced write.
fn coalesce_and_write_indexed(
    slots: &[WeightBufferSlot],
    queue: &Queue,
    data_slices: &[&[f32]],
    run_indices: &[usize],
) {
    debug_assert!(run_indices.len() > 1, "coalesce requires ≥2 slots");

    let first = run_indices[0];
    // Clone the buffer Arc so we don't hold an immutable borrow of `slots`
    // while we read slot data for the staging buffer.
    let buffer = slots[first].buffer.clone().expect("buffer must be extracted");
    let base_offset = slots[first].offset;
    let total_bytes: usize = run_indices.iter().map(|&i| slots[i].size_bytes).sum();

    // Build ONE staging buffer with all data concatenated.
    // Reuse a thread-local staging buffer to avoid per-call allocation.
    thread_local! {
        static STAGING: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    STAGING.with(|s| {
        let mut staging = s.borrow_mut();
        if staging.len() < total_bytes {
            staging.resize(total_bytes, 0);
        }
        let staging = &mut staging[..total_bytes];

        let mut pos = 0;
        for &i in run_indices {
            let slot = &slots[i];
            let data = data_slices[i];
            // Release-mode too (Issue 694 H5): a mismatch inside a coalesced
            // run writes a SHARED staging buffer with shifted boundaries —
            // every subsequent slot in the run gets wrong bytes.
            assert_eq!(
                std::mem::size_of_val(data),
                slot.size_bytes,
                "data size mismatch in coalesce_and_write"
            );
            let bytes = f32::as_bytes(data);
            staging[pos..pos + bytes.len()].copy_from_slice(bytes);
            pos += bytes.len();
        }

        // ONE write_buffer call for the entire contiguous run.
        queue.write_buffer(&buffer, base_offset, &staging[..total_bytes]);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_from_data_and_write_in_place() {
        // This test requires a GPU + CubeCL runtime — skip on CI without one.
        // Run locally with: cargo test -p riir-gpu --features kimi_k3_gpu_backward weight_buffer_cache -- --nocapture
        let ctx = if let Ok(ctx) = crate::context::GpuContext::new() { ctx } else {
                eprintln!("Skipping test — no GPU available");
                return;
            };
        let client = ctx.cubecl_client();

        // Create a slot with initial data.
        let data1: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut slot = WeightBufferSlot::from_data(&client, &data1);

        // Read back to verify initial data.
        let bytes = client.read_one(slot.handle.clone()).unwrap();
        let result1 = f32::from_bytes(&bytes);
        assert_eq!(result1, &data1);

        // Extract the buffer (enables fast path).
        slot.extract_buffer(&client).unwrap();
        assert!(slot.buffer.is_some());

        // Write new data in-place.
        let data2: Vec<f32> = (0..16).map(|i| (i as f32) * 10.0).collect();
        slot.write_in_place(&client, &ctx.queue, &data2);

        // Read back to verify the new data landed in the same buffer.
        let bytes = client.read_one(slot.handle.clone()).unwrap();
        let result2 = f32::from_bytes(&bytes);
        assert_eq!(result2, &data2, "write_in_place must update the buffer");
    }
}
