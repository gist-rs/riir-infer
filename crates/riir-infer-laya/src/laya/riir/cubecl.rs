//! The CubeCL backend of the riir forward (feature `laya-riir-cubecl`,
//! plan 611 S1b — T7 op-layer unification).
//!
//! The PORTABLE arm: wgpu (Metal on macOS, Vulkan/DX12 elsewhere) over
//! `riir-infer-gpu`'s op layer — ONE op vocabulary, the laya `Backend`
//! trait speaking the same ops the engine's CubeCL kernels speak. This is
//! NOT the performance arm: the hand-tuned Metal MSL lane
//! ([`super::metal`]) is the product of a full optimization campaign
//! (reflex issue 020), and a first clean portable implementation is not
//! expected to beat it — plan 611's A/B (S5) measures instead of assuming,
//! and the arm's fate (keep as the portability/CI lane, or delete) is
//! decided from that verdict. The GAP kernels the plan lands regardless
//! (the mean-centered LayerNorm, S1a) live engine-side in
//! `riir-infer-gpu`, callable both ways.
//!
//! # Residency model (the Metal lane's model, on CubeCL's runtime)
//!
//! The trait is host-slice-shaped; the backend maps slices to device
//! memory exactly the way [`super::metal::Metal`] does, with CubeCL's
//! server doing what Metal's manual slot table does:
//!
//! - **`weights`** — `(ptr, len)` → handle, agent-OWNED weight slices,
//!   permanent, first-miss upload, never invalidated (stable-address
//!   contract). `warm_weight`/`warm_weight_2d` pre-take the copies at
//!   load (riir-reflex issue 020 T1).
//! - **`chain`** — `(ptr, len, epoch)` → (handle, touch-stamp) for
//!   activations and per-forward host-authored inputs. The epoch bumps
//!   and the map clears in [`Backend::begin_pass`], so a recycled heap
//!   address can never hit a stale epoch's bytes. Within an epoch a hit's
//!   device copy is current because the forward bodies write every
//!   activation device-side before reading it (the write-first audit).
//!   Read-modify-write ops go through `chain_buf` (upload-on-miss — the
//!   host bytes ARE the current value); write-first destinations go
//!   through `chain_slot_for` (`client.empty`, no upload).
//! - **`begin_pass` does NOT sync** — unlike the Metal lane, whose raw
//!   Buffer handles must not drop while command buffers reference them.
//!   CubeCL tasks hold their own handle references (the server's managed
//!   memory refcounts usage), so clearing the chain map releases only OUR
//!   refs and the engine's decode forward runs this exact no-sync
//!   residency pattern ~835 dispatches per token. The next forward's
//!   dispatches are stream-ordered behind whatever is in flight.
//! - **`download_into`** — the ONE host-read barrier: resolve the written
//!   slot by base pointer + most-recent touch (the Metal lane's rule — a
//!   `src` may be a PREFIX of the written parent), then
//!   `read_f32` (synchronous: it drains the stream). Reading a
//!   host-authored slice panics, exactly like Metal.
//! - **Offsets** — byte-offset sub-views (`Handle::offset_start` /
//!   `offset_end`, the qwen-prefill/deltanet precedent), the analog of the
//!   Metal lane's bind-time offsets. One slot per WHOLE parent; the views
//!   are computed at bind time so per-head loops never fragment the
//!   cache.
//!
//! # Slice status (plan 611)
//!
//! S1b (this file): the trivial op family — `add`, `add_bias_row`,
//! `scale`, `relu`, `gelu_erf`, `glu_gelu_gate`, `copy_into`, `copy_at` —
//! over `riir-infer-gpu`'s elementwise launchers, plus the residency
//! machinery above. Every math op that needs a kernel the S2/S3 slices
//! have not landed yet panics LOUD naming its slice — never a silent
//! wrong answer. `supports_packed_attention` answers `false` until the
//! matmul family lands (an honest "this backend cannot run a forward
//! yet").

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use riir_infer_gpu::{
    ActiveComputeClient, ActiveRuntime, CubeCLContext, Handle, create_f32, read_f32,
    elementwise_cubecl::{
        AddBiasRowCubeCL, AddCubeCL, CopyAtCubeCL, CopyCubeCL, GeluErfCubeCL, GluGeluGateCubeCL,
        ReluCubeCL, ScaleCubeCL,
    },
};

use super::super::{LayaError, Result};
use super::backend::Backend;

/// A chain-cache key: `(host ptr, len, epoch)` — the Metal lane's shape.
type ChainKey = (usize, usize, u64);

/// The CubeCL backend: the shared per-process context (one CubeCL server),
/// the permanent weight cache, and the generation-keyed activation cache.
pub struct CubeclBackend {
    client: ActiveComputeClient,
    /// The context is held for lifetime clarity (the client is backed by
    /// its server); clones share it (Issue-676 sharing semantics).
    _ctx: CubeCLContext,
    /// `(ptr, len)` → handle for the agent-OWNED weight slices — permanent,
    /// first-miss upload, never invalidated.
    weights: Mutex<HashMap<(usize, usize), Handle>>,
    /// `(ptr, len, epoch)` → (handle, touch-stamp) for activations and
    /// per-forward host-authored inputs. Cleared at every pass.
    chain: Mutex<HashMap<ChainKey, (Handle, u64)>>,
    /// Monotonic touch counter for the chain entries (the
    /// `download_into` tie-break).
    touch_seq: AtomicU64,
    /// The pass generation (how many `begin_pass` calls have run).
    epoch: AtomicU64,
    /// The resolved CubeCL runtime label (e.g. `"wgpu<msl>"`) — printed at
    /// construction and quoted in every A/B line (plan 611: the backend
    /// selection must be pinned and visible).
    runtime_label: String,
}

impl CubeclBackend {
    /// Build the backend — the shared per-process CubeCL context (wgpu →
    /// Metal on this host; the resolved backend name is printed so a
    /// Vulkan/MoltenVK fallback can never hide).
    pub fn new() -> Result<Self> {
        let ctx = CubeCLContext::new()
            .map_err(|e| LayaError::Runtime(format!("riir cubecl backend: init failed: {e}")))?;
        let label = ctx.runtime_name().to_string();
        println!("CubeclBackend: runtime {label}");
        let client = ctx.client();
        Ok(Self {
            client,
            _ctx: ctx,
            weights: Mutex::new(HashMap::new()),
            chain: Mutex::new(HashMap::new()),
            touch_seq: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            runtime_label: label,
        })
    }

    /// The resolved CubeCL runtime label — quote it in every A/B line
    /// (plan 611: a fair comparison is only against a real backend, never
    /// a silent software fallback).
    pub fn runtime_label(&self) -> &str {
        &self.runtime_label
    }

    /// Permanent device copy of an agent-owned weight slice.
    fn weight_buf(&self, data: &[f32]) -> Handle {
        let key = (data.as_ptr() as usize, data.len());
        let mut map = self.weights.lock().expect("weight cache poison");
        if let Some(h) = map.get(&key) {
            return h.clone();
        }
        let h = create_f32(&self.client, data);
        map.insert(key, h.clone());
        h
    }

    /// Activation / per-forward input: hit within the current epoch → the
    /// device copy is current (no copy); miss → create + upload.
    fn chain_buf(&self, data: &[f32]) -> Handle {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let key = (data.as_ptr() as usize, data.len(), epoch);
        let mut map = self.chain.lock().expect("chain cache poison");
        let stamp = self.touch_seq.fetch_add(1, Ordering::Relaxed);
        if let Some((h, t)) = map.get_mut(&key) {
            *t = stamp;
            return h.clone();
        }
        let h = create_f32(&self.client, data);
        map.insert(key, (h.clone(), stamp));
        h
    }

    /// A write-first device destination slot for this epoch's `(ptr, len)`
    /// — `client.empty`, no upload (the forward bodies write the whole
    /// range before any read; untouched regions of a fresh parent are
    /// garbage by contract, exactly the Metal lane's `scratch()`).
    fn chain_slot_for(&self, dst: &[f32]) -> Handle {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let key = (dst.as_ptr() as usize, dst.len(), epoch);
        let mut map = self.chain.lock().expect("chain cache poison");
        let stamp = self.touch_seq.fetch_add(1, Ordering::Relaxed);
        if let Some((h, t)) = map.get_mut(&key) {
            *t = stamp;
            return h.clone();
        }
        let h = self
            .client
            .empty(std::mem::size_of_val(dst));
        map.insert(key, (h.clone(), stamp));
        h
    }

    /// Resolve a WRITTEN slice's slot for download: match by base pointer
    /// with sufficient extent, most-recent touch wins (the Metal lane's
    /// rule — the src may be a PREFIX of the written parent, and within an
    /// epoch a recycled address can leave a dead key at the same base).
    fn resolve_written(&self, src: &[f32]) -> Handle {
        let ptr = src.as_ptr() as usize;
        let map = self.chain.lock().expect("chain cache poison");
        let slot = map
            .iter()
            .filter(|(k, _)| k.0 == ptr && k.1 >= src.len())
            .max_by(|a, b| a.1 .1.cmp(&b.1 .1))
            .map(|(_, (h, _))| h.clone());
        let Some(h) = slot else {
            panic!(
                "download_into: no device buffer for this slice — host reads \
                 require a backend-produced buffer (the lazy-sync contract)"
            );
        };
        h
    }
}

/// The math ops the S2/S3 slices have not landed yet — LOUD, never a
/// silent wrong answer.
fn not_yet(op: &str, slice: &str) -> ! {
    panic!(
        "CubeclBackend::{op}: not landed — plan 611 {slice} (this skeleton \
         is S1b: the trivial op family + residency)"
    )
}

impl Backend for CubeclBackend {
    fn name(&self) -> &'static str {
        "cubecl"
    }

    fn matmul(
        &self,
        _a: &[f32],
        _a_off: usize,
        _m: usize,
        _k: usize,
        _b: &[f32],
        _b_off: usize,
        _n: usize,
        _dst: &mut [f32],
        _dst_off: usize,
    ) {
        not_yet("matmul", "S2");
    }

    fn matmul_kt(
        &self,
        _q: &[f32],
        _q_off: usize,
        _m: usize,
        _hd: usize,
        _k: &[f32],
        _k_off: usize,
        _dst: &mut [f32],
        _dst_off: usize,
    ) {
        not_yet("matmul_kt", "S2");
    }

    fn matmul_w(&self, _a: &[f32], _m: usize, _k: usize, _w: &[f32], _n: usize, _dst: &mut [f32]) {
        not_yet("matmul_w", "S2");
    }

    fn matmul_kt_heads(
        &self,
        _q: &[f32],
        _k: &[f32],
        _heads: usize,
        _m: usize,
        _hd: usize,
        _dst: &mut [f32],
    ) {
        not_yet("matmul_kt_heads", "S2");
    }

    fn matmul_heads(
        &self,
        _a: &[f32],
        _b: &[f32],
        _heads: usize,
        _m: usize,
        _k: usize,
        _n: usize,
        _dst: &mut [f32],
    ) {
        not_yet("matmul_heads", "S2");
    }

    fn add_mask_broadcast(&self, _x: &mut [f32], _mask: &[f32], _heads: usize) {
        not_yet("add_mask_broadcast", "S3");
    }

    fn add(&self, x: &mut [f32], x_off: usize, y: &[f32], y_off: usize, len: usize) {
        assert!(x.len() >= len + x_off, "add x extent");
        assert!(y.len() >= len + y_off, "add y extent");
        // Read-modify-write on BOTH parents: the slot must start from the
        // host bytes (chain_buf), never an empty scratch. Offsets ride the
        // params array — see elementwise_cubecl's module doc for why not
        // handle views (wgpu's 32-byte storage-bind alignment).
        let xb = self.chain_buf(x);
        let yb = self.chain_buf(y);
        // SAFETY: extents asserted above; parents bound whole.
        unsafe {
            AddCubeCL::launch::<ActiveRuntime>(
                &self.client,
                xb,
                x.len(),
                yb,
                y.len(),
                x_off,
                y_off,
                len,
            )
        };
    }

    fn add_bias_row(&self, x: &mut [f32], d: usize, bias: &[f32]) {
        assert_eq!(bias.len(), d, "bias extent");
        assert_eq!(x.len() % d, 0, "row extent");
        let xb = self.chain_buf(x);
        let bb = self.weight_buf(bias);
        // SAFETY: bindings are the whole parents (asserted by the launcher).
        unsafe {
            AddBiasRowCubeCL::launch::<ActiveRuntime>(&self.client, xb, x.len(), bb, d)
        };
    }

    fn scale(&self, x: &mut [f32], s: f32) {
        let xb = self.chain_buf(x);
        // SAFETY: binding is the whole parent.
        unsafe { ScaleCubeCL::launch::<ActiveRuntime>(&self.client, xb, x.len(), s) };
    }

    fn layer_norm_nobias_into(
        &self,
        _x: &[f32],
        _w: &[f32],
        _eps: f32,
        _d: usize,
        _sq: &mut Vec<f32>,
        _out: &mut [f32],
    ) {
        not_yet("layer_norm_nobias_into", "S3 — the S1a GAP kernel's wiring");
    }

    fn softmax_rows(&self, _x: &mut [f32], _n: usize) {
        not_yet("softmax_rows", "S3");
    }

    fn relu(&self, x: &mut [f32]) {
        let xb = self.chain_buf(x);
        // SAFETY: binding is the whole parent.
        unsafe { ReluCubeCL::launch::<ActiveRuntime>(&self.client, xb, x.len()) };
    }

    fn gelu_erf(&self, x: &mut [f32]) {
        let xb = self.chain_buf(x);
        // SAFETY: binding is the whole parent.
        unsafe { GeluErfCubeCL::launch::<ActiveRuntime>(&self.client, xb, x.len()) };
    }

    fn glu_gelu_gate(&self, fused: &[f32], rows: usize, i_sz: usize, out: &mut [f32]) {
        assert_eq!(fused.len(), rows * 2 * i_sz, "fused extent");
        assert_eq!(out.len(), rows * i_sz, "glu out extent");
        let fb = self.chain_buf(fused);
        let ob = self.chain_slot_for(out);
        // SAFETY: bindings are the whole parents (asserted by the launcher).
        unsafe {
            GluGeluGateCubeCL::launch::<ActiveRuntime>(&self.client, fb, ob, rows, i_sz)
        };
    }

    fn apply_rope(
        &self,
        _q: &mut [f32],
        _seq: usize,
        _heads: usize,
        _hd: usize,
        _cos: &[f32],
        _sin: &[f32],
    ) {
        not_yet("apply_rope", "S3");
    }

    fn split_heads(
        &self,
        _src: &[f32],
        _row_stride: usize,
        _off: usize,
        _seq: usize,
        _heads: usize,
        _hd: usize,
        _out: &mut [f32],
    ) {
        not_yet("split_heads", "S3");
    }

    fn merge_heads(&self, _src: &[f32], _seq: usize, _heads: usize, _hd: usize, _out: &mut [f32]) {
        not_yet("merge_heads", "S3");
    }

    fn gather_rows(&self, _x: &[f32], _d: usize, _rows: &[usize], _out: &mut [f32]) {
        not_yet("gather_rows", "S3");
    }

    /// Device-side whole-buffer copy (the residual stream is
    /// device-current — a host `copy_from_slice` would read stale bytes,
    /// the exact bug the Metal lane's G5 gate caught at layer 0).
    fn copy_into(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len(), "copy extent");
        let sb = self.chain_buf(src);
        let db = self.chain_slot_for(dst);
        // SAFETY: bindings are whole equal-size parents.
        unsafe { CopyCubeCL::launch::<ActiveRuntime>(&self.client, sb, db, src.len()) };
    }

    fn copy_at(&self, src: &[f32], src_off: usize, dst: &mut [f32], dst_off: usize, len: usize) {
        assert!(src.len() >= len + src_off, "copy_at src extent");
        assert!(dst.len() >= len + dst_off, "copy_at dst extent");
        let sb = self.chain_buf(src);
        let db = self.chain_slot_for(dst);
        // SAFETY: extents asserted above; parents bound whole.
        unsafe {
            CopyAtCubeCL::launch::<ActiveRuntime>(
                &self.client,
                sb,
                src.len(),
                db,
                dst.len(),
                src_off,
                dst_off,
                len,
            )
        };
    }

    /// `false` until the matmul family lands (S2): this backend cannot run
    /// ANY forward yet, and the agent must never silently take a path it
    /// cannot execute.
    fn supports_packed_attention(&self, _hd: usize) -> bool {
        false
    }

    fn download_into(&self, src: &[f32], out: &mut [f32]) {
        assert!(src.len() <= out.len(), "download extent");
        let h = self.resolve_written(src);
        let got =
            read_f32(&self.client, h)
                .unwrap_or_else(|e| panic!("download_into: cubecl read failed: {e}"));
        debug_assert!(got.len() >= src.len(), "read back fewer f32 than bound");
        out[..src.len()].copy_from_slice(&got[..src.len()]);
    }

    /// One forward is beginning: bump the pass epoch and drop the previous
    /// pass's slots. NO sync — CubeCL tasks hold their own handle
    /// references, so clearing the chain releases only our refs (the
    /// engine's decode forward is the standing proof of the no-sync
    /// residency pattern); the next forward's dispatches are
    /// stream-ordered behind whatever is in flight. See the module doc.
    fn begin_pass(&self) {
        self.epoch.fetch_add(1, Ordering::Relaxed);
        self.chain.lock().expect("chain cache poison").clear();
    }

    fn needs_window_mask(&self, _hd: usize) -> bool {
        // The default (the trait's): consume the mask tensor. The composed
        // reference attention (inherited via `attention_forward_default`
        // once S2/S3 land) consumes it — nothing here predicates.
        true
    }

    fn warm_weight(&self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        let _ = self.weight_buf(data);
    }

    /// v1: warm the PLAIN row-major copy — the S2 matmul decides the
    /// device layout (the Metal lane holds Wᵀ; the CubeCL tiled kernel's
    /// B indexing is read before choosing, never assumed).
    fn warm_weight_2d(&self, data: &[f32], _n: usize, _k: usize) {
        if data.is_empty() {
            return;
        }
        let _ = self.weight_buf(data);
    }
}
