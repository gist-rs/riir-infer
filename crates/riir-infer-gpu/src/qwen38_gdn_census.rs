//! Issue 1172 — the GDN-bucket **GPU-timestamp census** (the bench-942
//! record's named next instrument). Armed by `RIIR_QWEN38_GDN_CENSUS=1`
//! (exact literal), macOS + `metal_tensor_gemm` only.
//!
//! # Why this instrument
//!
//! Bench 942 left the pass gdn-dominated (74%) with ~30% of the bucket in
//! non-GEMM kernels the CPU-side `StageTimers` cannot split honestly:
//! per-section `client.sync()` serializes the queue (the bench-940
//! idle-queue stall inflates the following section), and unsynced CPU
//! deltas measure enqueue, never GPU wall. This module measures GPU wall
//! directly with `MTLCounterSampleBuffer` pass attachments.
//!
//! # The mechanism
//!
//! The whole forward runs on ONE shared in-order `MTLCommandQueue`: cubecl
//! submits onto the wgpu queue's own `MTLCommandQueue` (the
//! `SHARED_CUBECL_CONTEXT` singleton), and the raw zero-copy GEMMs extract
//! that same queue via `queue.as_hal` (the dense module's `self.queue`).
//! A tiny **marker** compute pass committed between sections samples the
//! GPU timestamp at its start-of-encoder and end-of-encoder boundaries
//! into a counter sample buffer, and the section wall is the
//! **end-to-end** span between consecutive markers' completions (the
//! start-of-encoder sample is a GPU SCHEDULE-time reading that can precede
//! the previous pass's completion — measured in the G1 gate — so it never
//! bounds a section; the end samples are in-order completions). Zero
//! in-pass syncs: every read happens once, at ingest end, after a single
//! `wait_until_completed` on the last marker.
//!
//! Order safety: each `mark` first calls `client.flush()` (the Issue-670
//! ordering law the raw GEMM dispatch also follows), so cubecl's pending
//! encoder tasks are committed onto the queue BEFORE the marker commits.
//! The markers add no memory traffic and no syncs — the free-running pass
//! timing is preserved to within the noop-kernel microseconds — and the
//! env-off build never constructs the census (byte-identical, G3).
//!
//! # Reading the samples
//!
//! The buffer is created with the device's **"timestamp" counter set**
//! (Apple Silicon validates a set-less buffer with "Invalid counter set
//! size. Expecting 1 counter"). The set carries exactly one counter — the
//! GPU timestamp — so each pass-attachment sample resolves to ONE u64 of
//! gpu ticks (start-of-encoder index and end-of-encoder index per marker,
//! committed in mark order). On Apple Silicon the GPU timestamp domain is
//! 1 ns/tick (wgpu-hal's adapter law: `timestamp_period` 1.0 for every
//! non-Intel device), so ms = ticks/1e6 — the G1 test validates the period
//! empirically (CPU sleep ↔ census delta) and every census record
//! reconciles the section sum against the free-running pass wall.

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use std::sync::Arc;

use metal::foreign_types::ForeignTypeRef;
use metal::{
    BufferRef, CommandBuffer, CommandQueueRef, CompileOptions, ComputePassDescriptor,
    ComputePassSampleBufferAttachmentDescriptor, ComputePipelineState, CounterSampleBuffer,
    CounterSampleBufferDescriptor, DeviceRef, MTLCommandQueue, MTLCounterSamplingPoint, MTLDevice,
    MTLResourceUsage, MTLStorageMode, MTLSize, NSRange, SharedEvent,
};
use wgpu::{Device as WgpuDevice, Queue as WgpuQueue};

use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use crate::cubecl_runtime::ActiveRuntime;

/// The marker kernel — an empty body; the census reads the pass-boundary
/// timestamps the pass attachment records, never any kernel output.
const MARKER_MSL: &str = "#include <metal_stdlib>\n\nkernel void riir_census_marker() {}\n";

/// The instrumented sections of one `prefill_gdn_layer` walk, in launch
/// order (the names in `qwen38_dense_cubecl::prefill_gdn_layer`'s marks
/// MUST stay positionally identical to this array — a rename is a
/// cross-file drift the report's share math would silently misattribute).
/// 14 sections closed by 15 marks per chunk; the 15th mark, "end", closes
/// the last section and opens nothing.
const SECTIONS: [&str; 14] = [
    "input_norm",
    "gemm_qkv_z_ab",
    "beta_decay",
    "conv1d",
    "expand_l2",
    "scan",
    "norm_ssm",
    "z_gate",
    "ssm_out_gemm",
    "residual_gdn",
    "mlp_norm_gateup",
    "mlp_act",
    "mlp_down_gemm",
    "mlp_residual",
];

/// Marks per instrumented chunk (each chunk emits exactly this many marks
/// for the census layer; the last section of a chunk is closed by the
/// first mark of the NEXT chunk — i.e. the `end` mark of chunk c and the
/// `input_norm` mark of chunk c+1 bound chunk c's last section).
pub const MARKS_PER_CHUNK: usize = SECTIONS.len() + 1;

/// 4096 samples = 2048 marks = 136 instrumented chunks — far above any
/// census run (pp5120 = 10 chunks = 150 marks); `mark` refuses loud past
/// it rather than wrapping into someone else's samples.
const SAMPLE_CAP: usize = 4096;

/// GPU-timestamp period: 1.0 ns/tick — the wgpu-hal adapter law for Apple
/// Silicon (see the module doc). Both mach and GPU timestamp domains are
/// 1 ns/tick, so slot identity cannot corrupt the ms math.
const NS_PER_TICK: f64 = 1.0;

// The capacity must cover a real census run's worth of chunks.
const _: () = assert!(SAMPLE_CAP / (2 * MARKS_PER_CHUNK) >= 128);

/// One finished census read: per-chunk-group section walls + totals.
pub struct GdnCensusReport {
    /// The instrumented GDN layer index.
    pub layer: usize,
    /// Full chunk groups emitted (marks / MARKS_PER_CHUNK).
    pub chunks: usize,
    /// One row per chunk group — the 14 section walls, ms.
    pub rows: Vec<[f64; SECTIONS.len()]>,
    /// Section wall totals across the ingest, ms.
    pub totals: [f64; SECTIONS.len()],
    /// The wall between chunks (an "end" mark → next chunk's
    /// "input_norm" mark) — the rest of the pass (layers 1..n, tail, wte),
    /// never a census section; carried for the reconciliation line only.
    pub inter_chunk_ms: f64,
    /// The whole gpu-domain sample span, ms (the section sum + marker
    /// executions + inter-group gaps — should reconcile with `totals`).
    pub gpu_span_ms: f64,
}

impl GdnCensusReport {
    fn empty(layer: usize) -> Self {
        Self {
            layer,
            chunks: 0,
            rows: Vec::new(),
            totals: [0.0; SECTIONS.len()],
            inter_chunk_ms: 0.0,
            gpu_span_ms: 0.0,
        }
    }

    /// The census printout — one line per section, then the per-chunk rows
    /// and the reconciliation line.
    pub fn print(&self) {
        if self.rows.is_empty() {
            return;
        }
        let groups = self.rows.len();
        let sum_per_group: f64 = self.totals.iter().sum::<f64>() / groups as f64;
        println!(
            "[gdn-census] gdn layer {} — {} chunk group(s), GPU wall per group \
             (ms, share of the {:.1} ms/group section sum):",
            self.layer, groups, sum_per_group
        );
        for (i, name) in SECTIONS.iter().enumerate() {
            let per = self.totals[i] / groups as f64;
            let share = if sum_per_group > 0.0 {
                100.0 * per / sum_per_group
            } else {
                0.0
            };
            println!("[gdn-census]   {name:<16} {per:8.2} ms ({share:5.1}%)");
        }
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                let cells: Vec<String> = r.iter().map(|v| format!("{v:.1}")).collect();
                format!("[{}]", cells.join(", "))
            })
            .collect();
        println!("[gdn-census] per-group ms (14 sections): {}", rows.join(" "));
        println!(
            "[gdn-census] reconciliation: section sum {:.1} ms/group | inter-chunk \
             wall total {:.1} ms (rest of the pass) | gpu sample span {:.1} ms \
             (timestamp counter set, 1 ns/tick)",
            sum_per_group, self.inter_chunk_ms, self.gpu_span_ms
        );
    }
}

/// The census instrument — one per `Qwen38WgpuForward` when armed. Holds
/// raw Metal handles (the dense-module pattern; the wgpu Arcs stay alive
/// in the context singleton, which outlives every forward).
pub struct GdnCensus {
    device: *mut MTLDevice,
    queue: *mut MTLCommandQueue,
    marker_pipeline: ComputePipelineState,
    csb: CounterSampleBuffer,
    /// The marker chain's ordering event: marker k waits value k and
    /// signals k+1 at its completion — the deep-queue fix (see mark()).
    chain: SharedEvent,
    chain_value: u64,
    /// (start_sample_index, section-opening name) per committed mark, in
    /// commit order.
    marks: Vec<(usize, &'static str)>,
    /// The last committed marker — `finish` waits on it (the in-order
    /// queue drains everything before it with it).
    last_cmd: Option<CommandBuffer>,
    next_sample: usize,
    layer: usize,
}

// SAFETY: raw Objective-C object pointers shareable across threads (Metal
// command-queue submission is thread-safe; `ComputePipelineState` and
// `CounterSampleBuffer` (metal-rs) are Send + Sync); the census is used
// single-threaded from the ingest path in practice.
unsafe impl Send for GdnCensus {}
unsafe impl Sync for GdnCensus {}

impl GdnCensus {
    /// Whether the env armed the census (`RIIR_QWEN38_GDN_CENSUS` exact
    /// `"1"`).
    pub fn armed() -> bool {
        std::env::var("RIIR_QWEN38_GDN_CENSUS").as_deref() == Ok("1")
    }

    /// Build the instrument on the shared CubeCL Metal device: check the
    /// stage-boundary sampling support, compile the marker pipeline, and
    /// allocate the timestamp counter sample buffer (NO counter set —
    /// timestamp-only samples, 16 bytes each).
    pub fn new(
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
        layer: usize,
    ) -> Result<Self, String> {
        let device_ptr: *mut MTLDevice = unsafe {
            let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
            let dev = guard
                .as_ref()
                .ok_or("gdn census: wgpu device is not Metal — the census requires Metal")?;
            let proto = dev.raw_device();
            (&**proto) as *const _ as *mut MTLDevice
        };
        let queue_ptr: *mut MTLCommandQueue = unsafe {
            let guard = wgpu_queue.as_hal::<wgpu::hal::api::Metal>();
            let q = guard
                .as_ref()
                .ok_or("gdn census: wgpu queue is not Metal — the census requires Metal")?;
            q.as_raw() as *const _ as *mut MTLCommandQueue
        };

        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
        if !device_ref.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err(
                "gdn census: the device does not support stage-boundary counter sampling \
                 (MTLCounterSamplingPoint::AtStageBoundary) — the pass-attachment \
                 timestamps would be unbacked; refuse loud"
                    .into(),
            );
        }

        let options = CompileOptions::new();
        let library = device_ref
            .new_library_with_source(MARKER_MSL, &options)
            .map_err(|e| format!("gdn census marker MSL compile failed: {e}"))?;
        let function = library
            .get_function("riir_census_marker", None)
            .map_err(|e| format!("gdn census: 'riir_census_marker' not found: {e}"))?;
        let marker_pipeline = device_ref
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("gdn census marker pipeline creation failed: {e}"))?;

        let desc = CounterSampleBufferDescriptor::new();
        desc.set_sample_count(SAMPLE_CAP as u64);
        desc.set_storage_mode(MTLStorageMode::Shared);
        // The device's "timestamp" counter set — Apple Silicon validates a
        // counter sample buffer against a counter set (a set-less buffer
        // refuses with "Invalid counter set size. Expecting 1 counter"):
        // the set carries exactly one counter (the GPU timestamp), so each
        // resolved sample is ONE u64 (gpu ticks), not the 16-byte
        // MTLCounterSampleTimestamp pair the set-less docs layout suggests.
        let ts_set = device_ref
            .counter_sets()
            .into_iter()
            .find(|s| s.name() == "timestamp")
            .ok_or("gdn census: the device exposes no 'timestamp' counter set — \
                    the pass-attachment samples would be unbacked; refuse loud")?;
        desc.set_counter_set(&ts_set);
        let csb = device_ref
            .new_counter_sample_buffer_with_descriptor(&desc)
            .map_err(|e| format!("gdn census counter sample buffer creation failed: {e}"))?;

        // Pipeline warmup — the FIRST dispatch of a fresh pipeline state
        // pays a prepare cost (measured ~1.5–3 ms) that would otherwise
        // land inside the first real marker's pass start and shorten that
        // ingest's first section wall. One throwaway marker at the RESERVED
        // tail sample indices (real marks never reach them — see mark()'s
        // capacity assert) plus a blocking wait: construction is once per
        // forward.
        {
            let warm_queue: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(queue_ptr) };
            let warm_cmd = warm_queue.new_command_buffer();
            let warm_desc = ComputePassDescriptor::new();
            let warm_att = ComputePassSampleBufferAttachmentDescriptor::new();
            warm_att.set_sample_buffer(&csb);
            warm_att.set_start_of_encoder_sample_index((SAMPLE_CAP - 4) as u64);
            warm_att.set_end_of_encoder_sample_index((SAMPLE_CAP - 3) as u64);
            warm_desc
                .sample_buffer_attachments()
                .set_object_at(0, Some(&warm_att));
            let warm_enc = warm_cmd.compute_command_encoder_with_descriptor(warm_desc);
            warm_enc.set_compute_pipeline_state(&marker_pipeline);
            warm_enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            warm_enc.end_encoding();
            warm_cmd.commit();
            warm_cmd.wait_until_completed();
        }

        Ok(Self {
            device: device_ptr,
            queue: queue_ptr,
            marker_pipeline,
            csb,
            chain: device_ref.new_shared_event(),
            chain_value: 0,
            marks: Vec::new(),
            last_cmd: None,
            next_sample: 0,
            layer,
        })
    }

    /// The instrumented GDN layer index.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Commit one marker pass on the shared queue: samples the GPU
    /// timestamp at the pass start (index `s`) and end (index `s + 1`).
    /// `name` is the section this marker OPENS (the first mark of a chunk
    /// opens `input_norm`; the last opens nothing — it closes the chunk's
    /// final section and the next chunk's first mark continues).
    ///
    /// `hazards` = the boundary buffers — the measured section's outputs
    /// plus the next section's inputs — declared `ReadWrite` so Metal's
    /// hazard tracker SANDWICHES the marker between the adjacent sections:
    /// without hazards the GPU overlaps hazard-free micro-passes freely
    /// (measured: a hazard-free marker slid ~47 ms ahead of queued rmsnorm
    /// work and read half the wall). Empty hazards are sound ONLY where
    /// real gaps separate the marks (an idle queue — the G1 sleep arms).
    ///
    /// Must be called on the ingest thread between the section launches —
    /// the flush inside pushes cubecl's pending tasks onto the queue first
    /// (the Issue-670 ordering law), so the marker commits AFTER everything
    /// logically before it.
    pub fn mark(
        &mut self,
        client: &ComputeClient<ActiveRuntime>,
        name: &'static str,
        hazards: &[Handle],
    ) -> Result<(), String> {
        assert!(
            self.next_sample + 2 <= SAMPLE_CAP - 4,
            "gdn census: sample buffer exhausted ({} usable samples cover \
             {} instrumented chunks — finish() resets between ingests; the \
             last pair is reserved for the construction warmup)",
            SAMPLE_CAP - 4,
            (SAMPLE_CAP - 4) / 2 / MARKS_PER_CHUNK,
        );
        let _ = client.flush();
        let start_idx = self.next_sample;
        self.next_sample += 2;
        let chain_value = self.chain_value;
        self.chain_value += 1;

        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd = queue_ref.new_command_buffer();
        cmd.set_label("gdn-census-mark");
        // Deep-queue ordering (the pp5120 rotation fix): wait for the
        // PREVIOUS marker's signal, then signal the NEXT value at this
        // buffer's completion. The wait pins this marker's pass behind the
        // previous marker INSIDE the GPU pipeline — without it, under a
        // 10-chunk queue the pass boundaries smeared one window per chunk
        // (the pp2560/pp5120 rotation). Command (CPU) commit order already
        // guarantees the chain is satisfiable (no deadlock).
        cmd.encode_wait_for_event(&self.chain, chain_value);
        let pass_desc = ComputePassDescriptor::new();
        let att = ComputePassSampleBufferAttachmentDescriptor::new();
        att.set_sample_buffer(&self.csb);
        att.set_start_of_encoder_sample_index(start_idx as u64);
        att.set_end_of_encoder_sample_index((start_idx + 1) as u64);
        pass_desc.sample_buffer_attachments().set_object_at(0, Some(&att));
        let enc = cmd.compute_command_encoder_with_descriptor(pass_desc);
        enc.set_compute_pipeline_state(&self.marker_pipeline);
        // Hazard pinning — RAW edges after the measured section's writes,
        // WAR/RAW edges against the next section's reads and writes (the
        // declared ReadWrite conflicts with both).
        for h in hazards {
            let (ptr, _off) = crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm
                ::extract_raw_buffer(client, h)?;
            let buf_ref: &BufferRef = unsafe { BufferRef::from_ptr(ptr) };
            enc.use_resource(buf_ref, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        enc.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cmd.encode_signal_event(&self.chain, chain_value + 1);
        cmd.commit();

        self.marks.push((start_idx, name));
        self.last_cmd = Some(cmd.to_owned());
        Ok(())
    }

    /// Drain the queue up to the last marker (one
    /// `wait_until_completed` — no per-section syncs anywhere), resolve
    /// every sample, and build the report. Resets the mark bookkeeping so
    /// the next ingest starts from sample 0.
    pub fn finish(&mut self) -> Result<GdnCensusReport, String> {
        let layer = self.layer;
        if let Some(cb) = self.last_cmd.take() {
            cb.wait_until_completed();
        }
        let marks = std::mem::take(&mut self.marks);
        self.next_sample = 0;
        self.chain_value = 0;
        if marks.is_empty() {
            return Ok(GdnCensusReport::empty(layer));
        }
        let n = marks.len();
        // Each resolved sample is ONE u64 — the "timestamp" counter set's
        // single counter value (gpu ticks) at that sample index. Every
        // marker owns TWO sample indices (start, end): 2n samples total.
        let raw = self.csb.resolve_counter_range(NSRange {
            location: 0,
            length: (2 * n) as u64,
        });
        if raw.len() != 2 * n {
            return Err(format!(
                "gdn census: resolve returned {} u64 for {n} markers (expected {})",
                raw.len(),
                2 * n
            ));
        }
        // Boundary choice (measured, not assumed — the G1 gate pins it): a
        // pass's START-of-encoder sample is recorded at GPU schedule time
        // and can precede the previous pass's completion, so consecutive
        // samples are NOT reliably ordered and start samples never bound a
        // section. The END-of-encoder samples are completions on an
        // in-order queue — strictly ordered — so every section wall is the
        // END-to-END span between consecutive markers: the section kernels
        // + any real queue gap + one noop marker execution (~µs).
        let ends: Vec<u64> = raw.iter().skip(1).step_by(2).copied().collect();
        if ends.len() != n {
            return Err(format!(
                "gdn census: odd sample count {} for {n} marks — the \
                 two-samples-per-marker layout assumption is broken; refuse loud",
                raw.len()
            ));
        }
        let mut prev = 0u64;
        for (k, s) in ends.iter().enumerate() {
            if *s < prev {
                return Err(format!(
                    "gdn census: marker completions went backwards at marker {} \
                     ({} < {prev}) — the queue-order assumption is broken; \
                     refuse loud",
                    k, *s
                ));
            }
            prev = *s;
        }
        let gpu_span_ms = (ends[n - 1] - ends[0]) as f64 * NS_PER_TICK / 1e6;

        // Section walls: completion of marker i − completion of marker i−1.
        // Group per chunk by stride MARKS_PER_CHUNK−1 sections; the
        // inter-chunk boundary (flat % MARKS_PER_CHUNK == SECTIONS.len())
        // spans the rest of the pass and is never a census section.
        let mut rows: Vec<[f64; SECTIONS.len()]> = Vec::new();
        let mut totals = [0.0f64; SECTIONS.len()];
        let mut inter_chunk_ms = 0.0f64;
        for i in 1..n {
            let wall_ms = (ends[i] - ends[i - 1]) as f64 * NS_PER_TICK / 1e6;
            let flat = i - 1;
            if flat % MARKS_PER_CHUNK == SECTIONS.len() {
                inter_chunk_ms += wall_ms;
                continue;
            }
            let sec = flat % SECTIONS.len();
            let group = flat / MARKS_PER_CHUNK;
            if group >= rows.len() {
                rows.push([0.0; SECTIONS.len()]);
            }
            rows[group][sec] = wall_ms;
            totals[sec] += wall_ms;
        }
        Ok(GdnCensusReport {
            layer,
            chunks: n / MARKS_PER_CHUNK,
            rows,
            totals,
            inter_chunk_ms,
            gpu_span_ms,
        })
    }
}
