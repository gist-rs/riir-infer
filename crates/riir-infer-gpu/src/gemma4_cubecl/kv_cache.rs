//! Per-layer-dim CPU KV cache for Gemma-4 GPU forward.
//!
//! Unlike Gemma-2 (uniform `kv_dim` across all layers), Gemma-4 alternates
//! between Sliding layers (`n_kv_head * head_dim`, e.g. 8×256=2048) and Full
//! layers (`n_global_kv_head * global_head_dim`, e.g. 1×512=512). Each layer's
//! cache row therefore has a different stride, so the cache is keyed on
//! `(layer, kv_stride)` pairs derived from the per-layer type at construction.
//!
//! Sliding-window layers write at `pos % sliding_window` (ring-buffer) so the
//! cache never grows beyond the window. Full-attention layers write at `pos`
//! directly (unbounded, up to `block_size`).

/// Per-layer CPU KV cache for Gemma-4. Each layer has its own `kv_stride`
/// derived from its layer type (Sliding vs Full).
pub struct Gemma4CpuKVCache {
    /// Key cache per layer. Length = `(positions_stored) * kv_stride[layer]`.
    pub keys: Vec<Vec<f32>>,
    /// Value cache per layer. Length = `(positions_stored) * kv_stride[layer]`.
    pub values: Vec<Vec<f32>>,
    /// Per-layer KV stride = `n_kv_head_for(layer_type) * head_dim_for(layer_type)`.
    pub kv_stride: Vec<usize>,
    /// Per-layer sliding-window capacity (0 = unbounded / full-attention layer).
    pub sliding_capacity: Vec<usize>,
    /// Per-layer count of positions written (monotonic, for full layers).
    pub n_positions: Vec<usize>,
}

impl Gemma4CpuKVCache {
    /// Construct a per-layer-dim cache. `kv_stride[layer]` and
    /// `sliding_capacity[layer]` must be pre-derived from the layer type
    /// (Sliding → `n_kv_head * head_dim` + `sliding_window`; Full →
    /// `n_global_kv_head * global_head_dim` + 0).
    pub fn new(
        n_layer: usize,
        kv_stride: Vec<usize>,
        sliding_capacity: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(kv_stride.len(), n_layer, "kv_stride length mismatch");
        debug_assert_eq!(
            sliding_capacity.len(),
            n_layer,
            "sliding_capacity length mismatch"
        );
        Self {
            keys: vec![Vec::new(); n_layer],
            values: vec![Vec::new(); n_layer],
            kv_stride,
            sliding_capacity,
            n_positions: vec![0; n_layer],
        }
    }

    /// Store K and V for the given layer at the given position.
    ///
    /// For sliding-window layers (`sliding_capacity[layer] > 0`), writes at
    /// `pos % sliding_capacity` (ring-buffer). For full-attention layers,
    /// grows the buffer to `pos * kv_stride + kv_stride`.
    ///
    /// `k` and `v` must have exactly `kv_stride[layer]` elements.
    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let stride = self.kv_stride[layer];
        debug_assert_eq!(k.len(), stride, "K length mismatch for layer {layer}");
        debug_assert_eq!(v.len(), stride, "V length mismatch for layer {layer}");
        let sw = self.sliding_capacity[layer];

        let (offset, total_capacity) = if sw > 0 {
            // Ring-buffer: capacity = sw * stride.
            let ring_pos = pos % sw;
            let cap = sw * stride;
            if self.keys[layer].len() < cap {
                self.keys[layer].resize(cap, 0.0);
                self.values[layer].resize(cap, 0.0);
            }
            (ring_pos * stride, cap)
        } else {
            // Unbounded: grow to fit pos.
            let off = pos * stride;
            let end = off + stride;
            if self.keys[layer].len() < end {
                self.keys[layer].resize(end, 0.0);
                self.values[layer].resize(end, 0.0);
            }
            self.n_positions[layer] = self.n_positions[layer].max(pos + 1);
            (off, end)
        };

        self.keys[layer][offset..offset + stride].copy_from_slice(k);
        self.values[layer][offset..offset + stride].copy_from_slice(v);
        let _ = total_capacity; // silence unused warning on full-attn path
    }

    /// Build the combined `[keys | values]` buffer for a single attention
    /// dispatch, covering only the attended window `[t_start, t_start + n_pos)`.
    ///
    /// Layout: `keys(n_pos × kv_stride) || values(n_pos × kv_stride)`.
    ///
    /// For sliding-window layers, the window is `[pos.saturating_sub(sw-1), pos]`,
    /// remapped into the ring buffer. For full-attention layers, the window is
    /// `[0, pos]`.
    pub fn get_combined_kv_window(
        &self,
        layer: usize,
        t_start: usize,
        n_pos: usize,
    ) -> Vec<f32> {
        let stride = self.kv_stride[layer];
        let sw = self.sliding_capacity[layer];
        let keys = &self.keys[layer];
        let values = &self.values[layer];
        let kv_len = n_pos * stride;
        let mut combined = Vec::with_capacity(kv_len * 2);

        if sw > 0 {
            // Sliding: remap t_start into the ring buffer.
            let ring_start = (t_start % sw) * stride;
            // The window [t_start, t_start+n_pos) is contiguous in the ring
            // buffer as long as n_pos <= sw (guaranteed by the sliding window).
            let end = ring_start + kv_len;
            if end <= keys.len() {
                combined.extend_from_slice(&keys[ring_start..end]);
            } else {
                // Window wraps — copy primary + wrapped portions.
                let primary = keys.len() - ring_start;
                combined.extend_from_slice(&keys[ring_start..]);
                combined.extend_from_slice(&keys[..kv_len - primary]);
            }
            if end <= values.len() {
                combined.extend_from_slice(&values[ring_start..end]);
            } else {
                let primary = values.len() - ring_start;
                combined.extend_from_slice(&values[ring_start..]);
                combined.extend_from_slice(&values[..kv_len - primary]);
            }
        } else {
            // Full-attention: contiguous from t_start.
            let off = t_start * stride;
            let end = off + kv_len;
            combined.extend_from_slice(&keys[off..end.min(keys.len())]);
            if end > keys.len() {
                combined.resize(kv_len, 0.0);
            }
            combined.extend_from_slice(&values[off..end.min(values.len())]);
            if combined.len() < kv_len * 2 {
                combined.resize(kv_len * 2, 0.0);
            }
        }
        combined
    }

    /// Number of positions stored in the given layer's cache (for diagnostics).
    #[allow(dead_code)] // used by tests + future debugging
    pub fn n_positions(&self, layer: usize) -> usize {
        let sw = self.sliding_capacity[layer];
        if sw > 0 {
            // For sliding layers, report the effective window size.
            self.n_positions[layer].min(sw)
        } else {
            self.n_positions[layer]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_attn_cache_grows() {
        // Full-attention layer: kv_stride=4, no sliding window.
        let mut cache = Gemma4CpuKVCache::new(1, vec![4], vec![0]);
        cache.store(0, 0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.n_positions(0), 1);
        cache.store(0, 1, &[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0]);
        assert_eq!(cache.n_positions(0), 2);

        let combined = cache.get_combined_kv_window(0, 0, 2);
        assert_eq!(&combined[..8], &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_sliding_cache_ring_buffer() {
        // Sliding layer: kv_stride=2, sliding_window=2.
        let mut cache = Gemma4CpuKVCache::new(1, vec![2], vec![2]);
        cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.store(0, 1, &[5.0, 6.0], &[7.0, 8.0]);
        // pos=2 wraps to ring slot 0.
        cache.store(0, 2, &[9.0, 10.0], &[11.0, 12.0]);

        // Window for pos=2 is [1, 2] (sw=2), n_pos=2.
        let combined = cache.get_combined_kv_window(0, 1, 2);
        // keys: pos1=[5,6], pos2(ring0)=[9,10]; values: pos1=[7,8], pos2=[11,12]
        assert_eq!(&combined[..4], &[5.0, 6.0, 9.0, 10.0]);
        assert_eq!(&combined[4..], &[7.0, 8.0, 11.0, 12.0]);
    }
}
