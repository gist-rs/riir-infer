//! The riir forward's backend seam — ONE forward body, two devices.
//!
//! The forward bodies in [`super::encoder`] / [`super::head`] call these
//! methods and nothing else; the op order lives in exactly one place (the
//! `.issues/005` architecture: no third copy of the op order). [`Cpu`]
//! wraps the free fns in [`super::ops`] 1:1 — the CPU lane's semantics,
//! reduction orders and SIMD work all stay there; [`super::metal::Metal`]
//! re-plays the same op semantics as MSL kernels (feature
//! `laya-riir-metal`).
//!
//! Dispatch is `&dyn`: one vcall per op against op work measured in
//! microseconds — unmeasurable at this granularity, and it keeps
//! `Encoder` / `Head` / `RiirAgent` single concrete types (the device is
//! chosen at load, [`super::agent`]).

/// The ~15 ops the model forward needs. Signatures mirror the [`super::ops`]
/// free fns 1:1 so the CPU impl is a straight delegation and the forward
/// rewrite is mechanical (`ops::foo(..)` → `b.foo(..)`).
pub trait Backend {
    /// Posture name for logs and gate lines (`"cpu"` / `"metal"`).
    fn name(&self) -> &'static str;

    /// dst[dst_off..dst_off + m·n] ← a[a_off..][m×k] @ b[b_off..][k×n] (both
    /// row-major). Operands are WHOLE parent buffers + element offsets —
    /// the Metal backend keeps one device slot per parent and applies the
    /// offset at bind time, so per-head loops never fragment the cache.
    #[allow(clippy::too_many_arguments)]
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
    );

    /// dst[dst_off..] ← q[q_off..][m×hd] @ k[k_off..]ᵀ (k row-major
    /// `[m×hd]`).
    #[allow(clippy::too_many_arguments)]
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
    );

    /// dst[m×n] ← a[m×k] @ w[n×k]ᵀ (w row-major `[out, in]`) — the weight
    /// projections; whole buffers, no offsets.
    fn matmul_w(&self, a: &[f32], m: usize, k: usize, w: &[f32], n: usize, dst: &mut [f32]);

    /// Batched over heads in ONE backend op:
    /// `dst[h·m·m ..] ← q[h·m·hd ..] @ k[h·m·hd ..]ᵀ` for every `h < heads`
    /// (q/k row-major `[heads, m, hd]`, dst `[heads, m, m]`) — the
    /// attention score loop as one dispatch on the device backends. The
    /// CPU lane keeps the identical per-head op order (bit-unchanged
    /// numerics).
    #[allow(clippy::too_many_arguments)]
    fn matmul_kt_heads(
        &self,
        q: &[f32],
        k: &[f32],
        heads: usize,
        m: usize,
        hd: usize,
        dst: &mut [f32],
    );

    /// Batched over heads in ONE backend op:
    /// `dst[h·m·n ..] ← a[h·m·k ..] @ b[h·k·n ..]` (dst `[heads, m, n]`) —
    /// the attention context loop.
    #[allow(clippy::too_many_arguments)]
    fn matmul_heads(
        &self,
        a: &[f32],
        b: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
        dst: &mut [f32],
    );

    /// The whole attention block, projected QKV in → merged heads out.
    ///
    /// Input `qkv` is the `[seq, 3d]` Wqkv projection (contiguous thirds
    /// Q/K/V at column offsets 0/d/2d), taken at element offset `qkv_off`
    /// — ZERO for the single-sequence forward, the sequence's packed row
    /// offset under the multi-question forward (`Encoder::forward_packed`,
    /// reflex issue 020 T5). `rope_row` is the ROW offset into the
    /// `[rows, hd]` cos/sin tables (the sequence's first packed row); the
    /// tables are passed WHOLE so a device backend binds one slot per
    /// forward and offsets it per dispatch. `out_off` is the destination
    /// row offset into `out` (`[rows, d]`). With one sequence every offset
    /// is zero and this is the unbatched op exactly.
    ///
    /// The op contract — rope BOTH the q
    /// and k thirds in place (q additionally takes the `1/√hd` scale AFTER
    /// the rotate, the encoder's rope-then-scale order), score every head,
    /// mask, softmax, mix the values, merge heads into `out` `[seq, d]` —
    /// is implemented HERE for the CPU lane as the exact op sequence in
    /// [`Backend::attention_forward_default`]; device backends override
    /// `attention_forward` with their fused form (the Metal lane's
    /// `flash_attn`), whose kill-switch falls back to that same reference
    /// sequence — the G5 + smoke equivalence gates hold the two together.
    /// The allowed-set contract: `mask` (the `[seq, seq]` additive tensor,
    /// `Some` only when `window < seq − 1` and `seq > 1`) and `window`
    /// (the sliding radius, `usize::MAX` = full attention) describe the
    /// SAME set — 0/allowed inside `|q − k| ≤ window`, `f32::MIN`/masked
    /// outside — so a device lane may use either representation.
    /// `scratch` is the caller's per-forward buffer set, resized in place
    /// (device lanes ignore it).
    #[allow(clippy::too_many_arguments)]
    fn attention_forward(
        &self,
        qkv: &[f32],
        qkv_off: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        rope_row: usize,
        scale: f32,
        seq: usize,
        heads: usize,
        hd: usize,
        window: usize,
        mask: Option<&[f32]>,
        scratch: &mut AttnScratch,
        out: &mut [f32],
        out_off: usize,
    ) {
        self.attention_forward_default(
            qkv, qkv_off, rope_cos, rope_sin, rope_row, scale, seq, heads, hd, window, mask,
            scratch, out, out_off,
        )
    }

    /// The reference attention op sequence (split → rope → scale → scores →
    /// mask → softmax → value mix → merge). Concrete on the trait so a
    /// device override can reach it without re-stating the ops. Host
    /// memory only: it slices the parents at the offsets, so a device
    /// backend whose ops would mis-bind a sliced DEVICE buffer must not
    /// reach it with non-zero offsets (the Metal fused path binds offsets
    /// itself; its kill-switch path guards that contract loud).
    #[allow(clippy::too_many_arguments)]
    fn attention_forward_default(
        &self,
        qkv: &[f32],
        qkv_off: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        rope_row: usize,
        scale: f32,
        seq: usize,
        heads: usize,
        hd: usize,
        window: usize,
        mask: Option<&[f32]>,
        scratch: &mut AttnScratch,
        out: &mut [f32],
        out_off: usize,
    ) {
        let d = heads * hd;
        // The mask tensor is authoritative on this lane; `window` only
        // matters to lanes that predicate instead.
        let _ = window;
        // The slab this call processes: exact per-sequence extents so the
        // ops' extent asserts stay meaningful under packing.
        let qkv = &qkv[qkv_off..qkv_off + seq * 3 * d];
        let out = &mut out[out_off..out_off + seq * d];
        let cos = &rope_cos[rope_row * hd..(rope_row + seq) * hd];
        let sin = &rope_sin[rope_row * hd..(rope_row + seq) * hd];
        let AttnScratch {
            q,
            k,
            v,
            scores,
            ctx,
        } = scratch;
        q.resize(heads * seq * hd, 0.0);
        k.resize(heads * seq * hd, 0.0);
        v.resize(heads * seq * hd, 0.0);
        self.split_heads(qkv, 3 * d, 0, seq, heads, hd, q);
        self.split_heads(qkv, 3 * d, d, seq, heads, hd, k);
        self.split_heads(qkv, 3 * d, 2 * d, seq, heads, hd, v);
        self.apply_rope(q, seq, heads, hd, cos, sin);
        self.apply_rope(k, seq, heads, hd, cos, sin);
        self.scale(q, scale);
        scores.resize(heads * seq * seq, 0.0);
        self.matmul_kt_heads(q, k, heads, seq, hd, scores);
        if let Some(m) = mask {
            self.add_mask_broadcast(scores, m, heads);
        }
        self.softmax_rows(scores, seq);
        ctx.resize(heads * seq * hd, 0.0);
        self.matmul_heads(scores, v, heads, seq, seq, hd, ctx);
        self.merge_heads(ctx, seq, heads, hd, out);
    }

    /// `x[r] += mask[r % mask.len()]` over the whole `heads·mask.len()`
    /// scores parent — the attention mask broadcast for every head in one
    /// op (candle's broadcast_add: adding 0.0 to allowed entries is exact).
    fn add_mask_broadcast(&self, x: &mut [f32], mask: &[f32], heads: usize);

    /// x[x_off..x_off + len] += y[y_off..y_off + len]. Operands are WHOLE
    /// parent buffers + offsets (the mask add targets one head's slab of
    /// the scores parent).
    fn add(&self, x: &mut [f32], x_off: usize, y: &[f32], y_off: usize, len: usize);

    /// x rows += bias (broadcast over rows of `d`).
    fn add_bias_row(&self, x: &mut [f32], d: usize, bias: &[f32]);

    /// x *= s.
    fn scale(&self, x: &mut [f32], s: f32);

    /// Bias-free LayerNorm into `out` (`sq` is the CPU scratch, ignored by
    /// the device backend).
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_nobias_into(
        &self,
        x: &[f32],
        w: &[f32],
        eps: f32,
        d: usize,
        sq: &mut Vec<f32>,
        out: &mut [f32],
    );

    /// Softmax over each row of `n`, in place.
    fn softmax_rows(&self, x: &mut [f32], n: usize);

    /// ReLU in place.
    fn relu(&self, x: &mut [f32]);

    /// gelu_erf in place.
    fn gelu_erf(&self, x: &mut [f32]);

    /// `out[r, j] = gelu_erf(fused[r, j]) · fused[r, I + j]`.
    fn glu_gelu_gate(&self, fused: &[f32], rows: usize, i_sz: usize, out: &mut [f32]);

    /// Rotate-half RoPE in place on `[heads, seq, hd]`.
    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        &self,
        q: &mut [f32],
        seq: usize,
        heads: usize,
        hd: usize,
        cos: &[f32],
        sin: &[f32],
    );

    /// `[seq, in_dim]` → `[heads, seq, hd]` at column offset `off`.
    #[allow(clippy::too_many_arguments)]
    fn split_heads(
        &self,
        src: &[f32],
        row_stride: usize,
        off: usize,
        seq: usize,
        heads: usize,
        hd: usize,
        out: &mut [f32],
    );

    /// `[heads, seq, hd]` → `[seq, d]`.
    fn merge_heads(&self, src: &[f32], seq: usize, heads: usize, hd: usize, out: &mut [f32]);

    /// Gather whole rows: `out[r, :] = x[rows[r], :]`.
    fn gather_rows(&self, x: &[f32], d: usize, rows: &[usize], out: &mut [f32]);

    /// dst[i] = src[i] — a whole-buffer copy. The encoder's layer-0
    /// identity path: the residual stream is DEVICE-current under Metal,
    /// so the copy must run device-side (a host `copy_from_slice` would
    /// read stale bytes — the bug the G5 gate caught at exactly layer 0).
    fn copy_into(&self, src: &[f32], dst: &mut [f32]);

    /// Range copy at element offsets: `dst[dst_off..dst_off+len] =
    /// src[src_off..src_off+len]`, device-side. The packed forward's slab
    /// seam (reflex issue 020 T5): each question's head consumes its own
    /// exact-size hidden, copied out of the packed encoder residual — a
    /// host slice of a device-current parent would read stale bytes, so
    /// the copy runs as a dispatch like [`Backend::copy_into`].
    fn copy_at(&self, src: &[f32], src_off: usize, dst: &mut [f32], dst_off: usize, len: usize);

    /// Can this backend execute the packed multi-sequence attention (the
    /// `attention_forward` offsets) at head dim `hd`? The agent gates the
    /// batched `system_one` on this — when false it keeps the
    /// per-question loop, never a silent mis-read. The CPU lane slices
    /// host memory (always true); the Metal lane answers for its fused
    /// kernel only, because its reference-sequence fallback cannot bind
    /// offsets into a device buffer.
    fn supports_packed_attention(&self, _hd: usize) -> bool {
        true
    }

    /// Host-read barrier — the ONE way a forward body reads a device
    /// result. CPU: `out` is `src`'s content (a copy). Metal: sync the
    /// pending encodes, then copy the device-current bytes of `src`'s
    /// slot into `out`. `src` must be a slice an op WROTE this epoch —
    /// reading a host-authored slice here is a programming error (the
    /// backend panics).
    fn download_into(&self, src: &[f32], out: &mut [f32]);

    /// One forward is about to run — invalidate the previous forward's
    /// device slots (host-authored buffers — rope tables, mask, `act_in` —
    /// are rebuilt per forward, often at recycled heap addresses). CPU:
    /// no-op. Called once per forward by the agent, before the encoder.
    fn begin_pass(&self);

    /// Does this backend CONSUME the `[seq, seq]` additive sliding-window
    /// mask at head dim `hd`, or does it predicate the window in-kernel?
    ///
    /// riir-reflex Issue 020 T2: the Metal lane's fused attention takes the
    /// `window` argument and throws the tensor away, yet the encoder built
    /// it every forward — a `vec![f32::MIN; seq * seq]` plus an
    /// `O(seq · 2·window)` fill for nothing (402 KB at `seq = 317`). The
    /// DEFAULT is `true` (consume it) so a new backend is never silently
    /// starved of a mask it needs; only a backend that predicates says
    /// otherwise, and it must answer for the SAME `hd` its attention path
    /// will dispatch at.
    fn needs_window_mask(&self, _hd: usize) -> bool {
        true
    }

    /// Pre-place one agent-owned weight slice on the device.
    ///
    /// riir-reflex Issue 020 T1: the device copy is otherwise taken lazily on
    /// the FIRST forward, which is the one the bench times — ≈ 0.5 GB of
    /// f32 projections on request #1's critical path, while the torch
    /// reference moves its weights during `load`, before its own timer
    /// starts. CPU: no-op (its weights are already where they are used).
    fn warm_weight(&self, _data: &[f32]) {}

    /// Pre-place one agent-owned PROJECTION weight — the `matmul_w`
    /// operand, row-major `[n, k]`. Separate from [`Backend::warm_weight`]
    /// because a device backend may hold it in a DIFFERENT layout than the
    /// host slice (the Metal lane holds `Wᵀ`, riir-reflex Issue 020 T4), and the shape
    /// is the thing that cannot be recovered from the slice alone.
    fn warm_weight_2d(&self, _data: &[f32], _n: usize, _k: usize) {}
}

/// The attention block's host scratch — the split q/k/v thirds, the score
/// matrix and the per-head context. Owned by the encoder's per-forward
/// scratch (resized in place across layers, never reallocated in the layer
/// loop); device lanes run the fused form and ignore it entirely.
#[derive(Debug, Default)]
pub struct AttnScratch {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub scores: Vec<f32>,
    pub ctx: Vec<f32>,
}

/// The CPU backend — a 1:1 delegation onto the [`super::ops`] free fns, so
/// the CPU lane's kernel work (and its SIMD optimization pass) stays in
/// exactly one place.
#[derive(Debug, Default, Clone, Copy)]
pub struct Cpu;

impl Backend for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }

    #[allow(clippy::too_many_arguments)]
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
        let a = &a[a_off..a_off + m * k];
        let b = &b[b_off..b_off + k * n];
        super::ops::matmul_into(a, m, k, b, n, &mut dst[dst_off..dst_off + m * n]);
    }

    fn matmul_w(&self, a: &[f32], m: usize, k: usize, w: &[f32], n: usize, dst: &mut [f32]) {
        super::ops::matmul_w_into(a, m, k, w, n, dst);
    }

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
        let q = &q[q_off..q_off + m * hd];
        let k = &k[k_off..k_off + m * hd];
        super::ops::matmul_kt_into(q, m, hd, k, &mut dst[dst_off..dst_off + m * m]);
    }

    fn matmul_kt_heads(
        &self,
        q: &[f32],
        k: &[f32],
        heads: usize,
        m: usize,
        hd: usize,
        dst: &mut [f32],
    ) {
        super::ops::matmul_kt_heads(q, k, heads, m, hd, dst);
    }

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
        super::ops::matmul_heads(a, b, heads, m, k, n, dst);
    }

    fn add_mask_broadcast(&self, x: &mut [f32], mask: &[f32], heads: usize) {
        super::ops::add_mask_broadcast(x, mask, heads);
    }

    fn add(&self, x: &mut [f32], x_off: usize, y: &[f32], y_off: usize, len: usize) {
        assert!(x.len() >= len + x_off, "add x extent");
        assert!(y.len() >= len + y_off, "add y extent");
        super::ops::add_inplace(&mut x[x_off..x_off + len], &y[y_off..y_off + len]);
    }

    fn add_bias_row(&self, x: &mut [f32], d: usize, bias: &[f32]) {
        super::ops::add_bias_row(x, d, bias);
    }

    fn scale(&self, x: &mut [f32], s: f32) {
        super::ops::scale_inplace(x, s);
    }

    fn layer_norm_nobias_into(
        &self,
        x: &[f32],
        w: &[f32],
        eps: f32,
        d: usize,
        sq: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        super::ops::layer_norm_nobias_into(x, w, eps, d, sq, out);
    }

    fn softmax_rows(&self, x: &mut [f32], n: usize) {
        super::ops::softmax_rows(x, n);
    }

    fn relu(&self, x: &mut [f32]) {
        super::ops::relu_inplace(x);
    }

    fn gelu_erf(&self, x: &mut [f32]) {
        super::ops::gelu_erf_inplace(x);
    }

    fn glu_gelu_gate(&self, fused: &[f32], rows: usize, i_sz: usize, out: &mut [f32]) {
        super::ops::glu_gelu_gate(fused, rows, i_sz, out);
    }

    fn apply_rope(
        &self,
        q: &mut [f32],
        seq: usize,
        heads: usize,
        hd: usize,
        cos: &[f32],
        sin: &[f32],
    ) {
        super::ops::apply_rope_inplace(q, seq, heads, hd, cos, sin);
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
        super::ops::split_heads(src, row_stride, off, seq, heads, hd, out);
    }

    fn merge_heads(&self, src: &[f32], seq: usize, heads: usize, hd: usize, out: &mut [f32]) {
        super::ops::merge_heads(src, seq, heads, hd, out);
    }

    fn gather_rows(&self, x: &[f32], d: usize, rows: &[usize], out: &mut [f32]) {
        super::ops::gather_rows(x, d, rows, out);
    }

    fn copy_into(&self, src: &[f32], dst: &mut [f32]) {
        dst.copy_from_slice(src);
    }

    fn copy_at(&self, src: &[f32], src_off: usize, dst: &mut [f32], dst_off: usize, len: usize) {
        assert!(src.len() >= len + src_off, "copy_at src extent");
        assert!(dst.len() >= len + dst_off, "copy_at dst extent");
        dst[dst_off..dst_off + len].copy_from_slice(&src[src_off..src_off + len]);
    }

    fn download_into(&self, src: &[f32], out: &mut [f32]) {
        out.copy_from_slice(src);
    }

    fn begin_pass(&self) {}
}
