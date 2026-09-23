use cubecl_common::bytes::Bytes;

/// Defines the thresholds that determine when a [`PendingDropQueue`] should be
/// flushed.
///
/// A flush is triggered when **either** limit is exceeded — whichever comes
/// first. Set a field to `u32::MAX` / `u64::MAX` to effectively disable it.
///
/// The `max_bytes_size` field is `u64` (not `u32`) so that single allocations
/// larger than 4 GiB — common when uploading multi-GB weight matrices — do not
/// silently truncate and bypass the flush threshold. See upstream issue
/// <https://github.com/tracel-ai/cubecl/issues/1359>.
#[derive(Debug)]
pub struct FlushingPolicy {
    /// Flush when this many allocations have been staged.
    pub max_bytes_count: u32,
    /// Flush when the total staged size reaches this many bytes.
    pub max_bytes_size: u64,
}

impl Default for FlushingPolicy {
    fn default() -> Self {
        Self {
            max_bytes_count: 64,
            max_bytes_size: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

/// Tracks staged allocations and evaluates them against a [`FlushingPolicy`].
#[derive(Default, Debug)]
pub(crate) struct FlushingPolicyState {
    bytes_count: u32,
    bytes_size: u64,
}

impl FlushingPolicyState {
    /// Record a newly staged [`Bytes`] allocation.
    ///
    /// Uses `saturating_add` on both counters so a pathological caller cannot
    /// trigger an arithmetic overflow panic (which would corrupt the drop
    /// queue and ultimately lose the GPU device). The size is accumulated as
    /// `u64` to avoid the `as u32` truncation that breaks flush thresholds for
    /// buffers ≥ 4 GiB.
    pub(crate) fn register(&mut self, bytes: &Bytes) {
        self.bytes_count = self.bytes_count.saturating_add(1);
        self.bytes_size = self.bytes_size.saturating_add(bytes.len() as u64);
    }

    /// Reset all counters, typically called after a flush.
    pub(crate) fn reset(&mut self) {
        self.bytes_count = 0;
        self.bytes_size = 0;
    }

    /// Returns `true` if either threshold in `policy` has been reached.
    pub(crate) fn should_flush(&self, policy: &FlushingPolicy) -> bool {
        self.bytes_count >= policy.max_bytes_count || self.bytes_size >= policy.max_bytes_size
    }
}

#[cfg(test)]
mod policy_tests {
    use std::vec;

    use super::*;

    fn policy() -> FlushingPolicy {
        FlushingPolicy {
            max_bytes_count: 4,
            max_bytes_size: 100,
        }
    }

    fn state() -> FlushingPolicyState {
        FlushingPolicyState {
            bytes_count: 0,
            bytes_size: 0,
        }
    }

    #[test]
    fn no_flush_when_below_both_thresholds() {
        let s = state();
        assert!(!s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_count_threshold_reached() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_size_threshold_reached() {
        let mut s = state();
        s.register(&Bytes::from_elems(vec![0u8; 101]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_triggered_by_whichever_limit_comes_first() {
        let mut s = state();
        // Only 2 allocations but already over the size limit.
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn reset_clears_state() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
        s.reset();
        assert!(!s.should_flush(&policy()));
    }

    /// Regression test for tracel-ai/cubecl#1359 — `FlushingPolicy.max_bytes_size`
    /// must be `u64` (not `u32`) so that thresholds above 4 GiB can be
    /// represented. Under the old API a caller wanting to effectively disable
    /// size-based flushing by setting a very high threshold could not express
    /// any value above `u32::MAX` (4 GiB), which is a realistic drop size when
    /// uploading multi-GB weight matrices. This test does not need to allocate
    /// anything close to 4 GiB — it just asserts the field accepts a `u64`
    /// value above `u32::MAX`, which would not compile under the old `u32` type.
    /// Combined with `register` using `saturating_add` (also compiled in), this
    /// is sufficient proof: the arithmetic cannot overflow because (a) the
    /// accumulator is `u64` and (b) addition is saturating.
    #[test]
    fn flush_threshold_accepts_u64_above_u32_max() {
        let policy = FlushingPolicy {
            max_bytes_count: u32::MAX,
            // 8 GiB — does not fit in u32. If the field were still `u32` this
            // literal would not compile.
            max_bytes_size: 8 * 1024 * 1024 * 1024_u64,
        };
        // Sanity: a tiny drop should NOT trigger flush against an 8 GiB threshold.
        let mut s = state();
        s.register(&Bytes::from_elems(vec![0u8; 1024]));
        assert!(!s.should_flush(&policy));
    }
}
