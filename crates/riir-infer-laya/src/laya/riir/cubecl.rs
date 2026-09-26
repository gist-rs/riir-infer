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
//! S1b: the trivial op family — `add`, `add_bias_row`, `scale`, `relu`,
//! `gelu_erf`, `glu_gelu_gate`, `copy_into`, `copy_at` — over
//! `riir-infer-gpu`'s elementwise launchers, plus the residency machinery
//! above. S2: the FULL matmul family (`matmul_w` over the shipped
//! derived-dims transB kernel; the four offset/batched ops over z-dispatched
//! tiled kernels). S3 (this slice): the remaining forward surface —
//! `add_mask_broadcast` + `softmax_rows` (the batched in-place row-softmax,
//! the single-row kernel's five phases with the row on `CUBE_POS_X`),
//! `layer_norm_nobias_into` (the S1a GAP kernel wired), `apply_rope`
//! (rotate-half, one thread per (head, position, lane) PAIR — race-free,
//! CPU expressions verbatim), `split_heads`/`merge_heads` (permutation
//! kernels, exact), `gather_rows` (the embedding gather; the table rides
//! the permanent weight cache, ids upload fresh as u32). `attention_forward`
//! stays INHERITED (`attention_forward_default`): with
//! `supports_packed_attention == false` the agent keeps the per-question
//! loop, so attention is only ever reached at qkv_off == 0 where the
//! default's host slices are whole-parent exact-extent binds (safe); the
//! rope/mask tables it slices are host-authored and therefore
//! host-authoritative. `set_row_segments` stays the trait's no-op.
//! `matmul_w_accum`/`matmul_w_glu` compose through the trait defaults.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use riir_infer_gpu::{
    ActiveComputeClient, ActiveRuntime, CubeCLContext, Handle, create_f32, create_u32, read_f32,
    elementwise_cubecl::{
        AddBiasRowCubeCL, AddCubeCL, AddMaskBroadcastCubeCL, CopyAtCubeCL, CopyCubeCL,
        GeluErfCubeCL, GluGeluGateCubeCL, ReluCubeCL, ScaleCubeCL, SoftmaxRowsInplaceCubeCL,
    },
    encoder_lane_cubecl::{GatherRowsCubeCL, MergeHeadsCubeCL, RopeRotateHalfCubeCL, SplitHeadsCubeCL},
    matmul_cubecl::{MatmulCubeCL, MatmulRrOffCubeCL, MatmulTransbOffCubeCL},
    norms_cubecl::LayerNormMeanBatchedCubeCL,
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

impl Backend for CubeclBackend {
    fn name(&self) -> &'static str {
        "cubecl"
    }

    /// `dst[m×n] = a[m×k] @ w[n×k]ᵀ` — the shipped derived-dims transB
    /// kernel EXACTLY (the kernel doc's "already the encoder's matmul_w
    /// shape" prior-art note). The call sites guarantee whole exact-extent
    /// parents (`HeadScratch::fit` then exact dims), so the in-kernel
    /// derivation from the declared lengths is exact. The weight rides the
    /// permanent cache (`warm_weight_2d` pre-takes it at load — the
    /// first-miss upload here is the un-warmed path only).
    fn matmul_w(&self, a: &[f32], m: usize, k: usize, w: &[f32], n: usize, dst: &mut [f32]) {
        assert_eq!(a.len(), m * k, "matmul_w a extent");
        assert_eq!(w.len(), n * k, "matmul_w w extent");
        assert_eq!(dst.len(), m * n, "matmul_w dst extent");
        let ab = self.chain_buf(a);
        let wb = self.weight_buf(w);
        let db = self.chain_slot_for(dst);
        // Exact extents asserted above; whole-parent binds. (`launch` is the
        // safe wrapper — the unsafe `launch_tiled` is internal.)
        MatmulCubeCL::launch::<ActiveRuntime>(&self.client, ab, wb, db, m, k, n);
    }

    /// `dst[dst_off..][m×n] = a[a_off..][m×k] @ b[b_off..][k×n]` — the
    /// offset-capable RR kernel at heads=1 (offsets ride the params
    /// buffer over whole-parent binds — the S1b alignment finding).
    fn matmul(
        &self,
        a: &[f32],
        a_off: usize,
        m: usize,
        k: usize,
        b: &[f32],
        b_off: usize,
        n: usize,
        dst: &mut [f32],
        dst_off: usize,
    ) {
        assert!(a.len() >= a_off + m * k, "matmul a extent");
        assert!(b.len() >= b_off + k * n, "matmul b extent");
        assert!(dst.len() >= dst_off + m * n, "matmul dst extent");
        let ab = self.chain_buf(a);
        let bb = self.chain_buf(b);
        let db = self.chain_slot_for(dst);
        // SAFETY: extents asserted above; parents bound whole.
        unsafe {
            MatmulRrOffCubeCL::launch::<ActiveRuntime>(
                &self.client,
                ab,
                a.len(),
                bb,
                b.len(),
                db,
                dst.len(),
                1,
                m,
                k,
                n,
                a_off,
                b_off,
                dst_off,
            )
        };
    }

    /// `dst[dst_off..][m×m] = q[q_off..][m×hd] @ k[k_off..][m×hd]ᵀ` — the
    /// offset-capable transB kernel at heads=1.
    fn matmul_kt(
        &self,
        q: &[f32],
        q_off: usize,
        m: usize,
        hd: usize,
        k: &[f32],
        k_off: usize,
        dst: &mut [f32],
        dst_off: usize,
    ) {
        assert!(q.len() >= q_off + m * hd, "matmul_kt q extent");
        assert!(k.len() >= k_off + m * hd, "matmul_kt k extent");
        assert!(dst.len() >= dst_off + m * m, "matmul_kt dst extent");
        let qb = self.chain_buf(q);
        let kb = self.chain_buf(k);
        let db = self.chain_slot_for(dst);
        // SAFETY: extents asserted above; parents bound whole.
        unsafe {
            MatmulTransbOffCubeCL::launch::<ActiveRuntime>(
                &self.client,
                qb,
                q.len(),
                kb,
                k.len(),
                db,
                dst.len(),
                1,
                m,
                hd,
                q_off,
                k_off,
                dst_off,
            )
        };
    }

    /// The score batch as ONE z-dispatched launch (heads ride `CUBE_POS_Z`;
    /// q/k/dst bind whole once, the head slabs are in-kernel offsets) — no
    /// host loop, no per-head cache lookups, no destination fragmentation.
    fn matmul_kt_heads(
        &self,
        q: &[f32],
        k: &[f32],
        heads: usize,
        m: usize,
        hd: usize,
        dst: &mut [f32],
    ) {
        assert_eq!(q.len(), heads * m * hd, "matmul_kt_heads q extent");
        assert_eq!(k.len(), heads * m * hd, "matmul_kt_heads k extent");
        assert_eq!(dst.len(), heads * m * m, "matmul_kt_heads dst extent");
        let qb = self.chain_buf(q);
        let kb = self.chain_buf(k);
        let db = self.chain_slot_for(dst);
        // SAFETY: exact extents asserted above; parents bound whole.
        unsafe {
            MatmulTransbOffCubeCL::launch::<ActiveRuntime>(
                &self.client,
                qb,
                q.len(),
                kb,
                k.len(),
                db,
                dst.len(),
                heads,
                m,
                hd,
                0,
                0,
                0,
            )
        };
    }

    /// The context batch as ONE z-dispatched launch (scores @ v per head).
    fn matmul_heads(
        &self,
        a: &[f32],
        b: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
        dst: &mut [f32],
    ) {
        assert_eq!(a.len(), heads * m * k, "matmul_heads a extent");
        assert_eq!(b.len(), heads * k * n, "matmul_heads b extent");
        assert_eq!(dst.len(), heads * m * n, "matmul_heads dst extent");
        let ab = self.chain_buf(a);
        let bb = self.chain_buf(b);
        let db = self.chain_slot_for(dst);
        // SAFETY: exact extents asserted above; parents bound whole.
        unsafe {
            MatmulRrOffCubeCL::launch::<ActiveRuntime>(
                &self.client,
                ab,
                a.len(),
                bb,
                b.len(),
                db,
                dst.len(),
                heads,
                m,
                k,
                n,
                0,
                0,
                0,
            )
        };
    }

    fn add_mask_broadcast(&self, x: &mut [f32], mask: &[f32], heads: usize) {
        assert!(heads > 0, "heads");
        let mlen = mask.len();
        assert_eq!(x.len(), heads * mlen, "mask broadcast extent");
        let xb = self.chain_buf(x);
        let mb = self.chain_buf(mask);
        // SAFETY: extents asserted above; parents bound whole.
        unsafe {
            AddMaskBroadcastCubeCL::launch::<ActiveRuntime>(&self.client, xb, x.len(), mb, mlen)
        };
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

    /// The S1a GAP kernel wired: batched mean-centered LayerNorm, one
    /// workgroup per row, the one-pass `E[x²] − μ²` variance (the kernel
    /// header documents the order divergence from the CPU lane's two-pass
    /// form — the G5 budget prices it).
    fn layer_norm_nobias_into(
        &self,
        x: &[f32],
        w: &[f32],
        eps: f32,
        d: usize,
        _sq: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        assert_eq!(x.len() % d, 0, "layer_norm row extent");
        assert_eq!(w.len(), d, "layer_norm gamma extent");
        assert_eq!(out.len(), x.len(), "layer_norm out extent");
        let rows = x.len() / d;
        let xb = self.chain_buf(x);
        let wb = self.weight_buf(w);
        let ob = self.chain_slot_for(out);
        // SAFETY: extents asserted above; parents bound whole. `sq` is the
        // CPU lane's scratch — ignored here (the trait doc's contract).
        unsafe {
            LayerNormMeanBatchedCubeCL::launch::<ActiveRuntime>(&self.client, xb, wb, ob, rows, d, eps)
        };
    }

    fn softmax_rows(&self, x: &mut [f32], n: usize) {
        assert_eq!(x.len() % n, 0, "softmax row extent");
        let xb = self.chain_buf(x);
        // SAFETY: extent asserted above; the whole parent is one binding
        // read before any write (the kernel's phase barriers).
        unsafe {
            SoftmaxRowsInplaceCubeCL::launch::<ActiveRuntime>(&self.client, xb, x.len() / n, n)
        };
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

    /// Rotate-half rope, one thread per (head, position, lane) pair —
    /// race-free by construction, the CPU lane's expressions verbatim.
    /// cos/sin are host-authored per-forward tables: chain_buf uploads
    /// their CURRENT host bytes (the one place a host-authored slice is
    /// legitimately the truth on a device backend).
    fn apply_rope(
        &self,
        q: &mut [f32],
        seq: usize,
        heads: usize,
        hd: usize,
        cos: &[f32],
        sin: &[f32],
    ) {
        assert_eq!(q.len(), heads * seq * hd, "rope q extent");
        assert_eq!(cos.len(), seq * hd, "rope cos extent");
        assert_eq!(sin.len(), seq * hd, "rope sin extent");
        let qb = self.chain_buf(q);
        let cb = self.chain_buf(cos);
        let sb = self.chain_buf(sin);
        // SAFETY: extents asserted above; the pair kernel is race-free.
        unsafe {
            RopeRotateHalfCubeCL::launch::<ActiveRuntime>(&self.client, qb, heads, seq, hd, cb, sb)
        };
    }

    fn split_heads(
        &self,
        src: &[f32],
        row_stride: usize,
        off: usize,
        seq: usize,
        heads: usize,
        hd: usize,
        out: &mut [f32],
    ) {
        assert_eq!(out.len(), heads * seq * hd, "split extent");
        let sb = self.chain_buf(src);
        let ob = self.chain_slot_for(out);
        // SAFETY: the launcher asserts the src read extent
        // ((seq−1)·row_stride + off + heads·hd) against src.len().
        unsafe {
            SplitHeadsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                sb,
                src.len(),
                ob,
                out.len(),
                row_stride,
                off,
                seq,
                heads,
                hd,
            )
        };
    }

    fn merge_heads(&self, src: &[f32], seq: usize, heads: usize, hd: usize, out: &mut [f32]) {
        assert_eq!(out.len(), seq * heads * hd, "merge extent");
        let sb = self.chain_buf(src);
        let ob = self.chain_slot_for(out);
        // SAFETY: extents asserted above (src heads·seq·hd, out seq·heads·hd).
        unsafe {
            MergeHeadsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                sb,
                ob,
                out.len(),
                seq,
                heads,
                hd,
            )
        };
    }

    /// The embedding row gather: the table is a model weight (the
    /// permanent cache, `warm_weight`'s slot); the row indices are the
    /// per-forward host ids, uploaded fresh as a u32 buffer (≤ a few KB
    /// per forward, once per encoder pass).
    fn gather_rows(&self, x: &[f32], d: usize, rows: &[usize], out: &mut [f32]) {
        assert_eq!(out.len(), rows.len() * d, "gather extent");
        let xb = self.weight_buf(x);
        let rows_u32: Vec<u32> = rows
            .iter()
            .map(|&r| {
                assert!(r <= u32::MAX as usize, "gather row index overflow");
                r as u32
            })
            .collect();
        let rb = create_u32(&self.client, &rows_u32);
        let ob = self.chain_slot_for(out);
        // SAFETY: extents asserted above + the launcher's per-row x extent
        // check against the usize ids.
        unsafe {
            GatherRowsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                xb,
                x.len(),
                rb,
                rows,
                ob,
                out.len(),
                d,
            )
        };
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

    /// `false` until the forward surface COMPLETES (S3): the matmul family
    /// is landed (S2) but a forward also needs the norms/softmax/rope/
    /// split/merge family — this backend cannot run ANY forward yet, and
    /// the agent must never silently take a path it cannot execute.
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
