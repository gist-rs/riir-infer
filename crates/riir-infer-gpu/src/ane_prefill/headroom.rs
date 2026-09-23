//! Issue 886 T1/T2 — the machine side of the two-sided ANE bank budget.
//!
//! The model-side ceiling (`DEFAULT_MAX_ANE_BYTES`, 13 GB) bounds the bank
//! against the MODEL's worst case; nothing in it knows the box. This module
//! supplies the missing term: system-wide available memory, read through
//! the same kernel interfaces oMLX's `_ane_bank_memory_headroom_ok` uses.
//!
//! Quantities (macOS, aarch64):
//! - `total_bytes` — `sysctl hw.memsize`.
//! - `available_bytes` — `host_statistics64(HOST_VM_INFO64)`:
//!   `(free_count + inactive_count) * hw.pagesize`. Inactive memory is
//!   reclaimable (file-backed/purgeable), so free+inactive is the standard
//!   macOS "available" approximation — free-only would understate an
//!   idle box by ~2x and make the machine term bind spuriously.
//! - SYSTEM-WIDE, deliberately: the filing's motivating datum was sibling
//!   pressure (31.9 GiB in use with three agent sessions building), which
//!   a self-process `phys_footprint` read cannot see. A budget that cannot
//!   see the box cannot protect it; a budget that only sees ITSELF still
//!   cannot.
//!
//! T2 — the settle rule (measurement honesty): the driver releases program
//! memory ASYNCHRONOUSLY, so any headroom re-measure taken immediately
//! after a failed/abandoned registration reads memory that is logically
//! free but not yet reclaimed. Every re-measure must follow
//! [`settle_before_remeasure`] (oMLX: `gc.collect() + sleep(0.5)`; the Rust
//! analog of the gc half is the synchronous `Arc<BridgeKernel>` drop — the
//! sleep covers the driver's half). Call it before EVERY headroom
//! (re-)measure the ladder makes, including the first.
//!
//! Failure posture (the recorded deliberate divergence from oMLX): oMLX's
//! snapshot fails OPEN (`(0,0)` never blocks the fast path); for us that
//! would defeat T1's whole point, so a failed measurement returns `None`
//! and the budget falls back to the MODEL-side ceiling — the conservative
//! term — loudly logged at the resolution site.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::ffi::c_void;
use std::time::Duration;

/// `host_info_flavor_t` — verified against the MacOSX SDK headers
/// (`mach/host_info.h:182`): `HOST_VM_INFO64 = 4`.
const HOST_VM_INFO64: i32 = 4;
/// `HOST_VM_INFO64_COUNT` — verified by clang probe on this SDK
/// (2026-09-13): 62 (integer_t words) = `sizeof(vm_statistics64_data_t)`
/// = 248 bytes. The struct's first four `natural_t` fields are
/// `free_count`(0), `active_count`(4), `inactive_count`(8), `wire_count`(12)
/// — offsets stable since the type's introduction.
const HOST_VM_INFO64_COUNT: u32 = 62;
/// Byte offsets of the fields this module reads inside
/// `vm_statistics64_data_t`.
const OFF_FREE_COUNT: usize = 0;
const OFF_INACTIVE_COUNT: usize = 8;

unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> i32;
    fn mach_host_self() -> u32;
    fn host_statistics64(
        host: u32,
        flavor: i32,
        host_info: *mut c_void,
        host_info_count: *mut u32,
    ) -> i32;
}

/// The settle half of the T2 rule — sleep past the driver's asynchronous
/// program release so the next headroom read reports reclaimed memory.
/// 0.5 s, oMLX's measured constant (`_ane_bank_memory_headroom_ok` calls
/// `gc.collect() + sleep(0.5)` before EVERY re-measure, including the
/// first).
pub fn settle_before_remeasure() {
    std::thread::sleep(Duration::from_millis(500));
}

fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<u64>()).then_some(value)
}

/// Physical memory (`hw.memsize`), or `None` when the read fails.
pub fn total_bytes() -> Option<u64> {
    sysctl_u64(c"hw.memsize")
}

/// Page size (`hw.pagesize`), or `None` when the read fails.
fn page_size() -> Option<u64> {
    sysctl_u64(c"hw.pagesize")
}

/// System-wide available memory: `(free + inactive) pages * page size` via
/// `host_statistics64(HOST_VM_INFO64)` — or `None` on any read failure
/// (the caller's conservative fallback is the model-side ceiling).
pub fn available_bytes() -> Option<u64> {
    // 248 bytes, u32-aligned; only the two word-offsets we consume are read.
    let mut stats = [0u32; HOST_VM_INFO64_COUNT as usize];
    let mut count = HOST_VM_INFO64_COUNT;
    let rc = unsafe {
        host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            stats.as_mut_ptr() as *mut c_void,
            &mut count,
        )
    };
    if rc != 0 {
        return None;
    }
    let free = stats[OFF_FREE_COUNT / 4] as u64;
    let inactive = stats[OFF_INACTIVE_COUNT / 4] as u64;
    let pages = free.saturating_add(inactive);
    let bytes = pages.saturating_mul(page_size()?);
    (bytes > 0).then_some(bytes)
}

/// The machine-side ceiling term: `fraction * available_bytes`. `None`
/// when the measurement is unavailable (fail toward the model-side term).
pub fn machine_side_ceiling(fraction: f64) -> Option<u64> {
    // One snapshot per call — a re-read per term would compare TWO different
    // instants of a quantity that moves while you measure it (the same
    // measurement-honesty rule the T2 settle encodes, on the read side).
    Some(machine_side_ceiling_from(available_bytes()?, fraction))
}

/// Pure half of [`machine_side_ceiling`] (assertable against an injected
/// snapshot): `fraction.clamp(0.0, 1.0) * available`.
pub fn machine_side_ceiling_from(available: u64, fraction: f64) -> u64 {
    (fraction.clamp(0.0, 1.0) * available as f64) as u64
}

/// The T1 composition (pure half — assertable headless against injected
/// measurements): `min(model, machine)`, with the flag naming the binding
/// side so the resolution site can log it once.
pub fn two_sided_ceiling(model_side: u64, machine_side: Option<u64>) -> (u64, bool) {
    match machine_side {
        Some(machine) if machine < model_side => (machine, true),
        _ => (model_side, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure half: the composition + binding-flag table.
    #[test]
    fn two_sided_ceiling_picks_the_min_and_flags_the_binding_side() {
        assert_eq!(two_sided_ceiling(13, None), (13, false));
        assert_eq!(two_sided_ceiling(13, Some(20)), (13, false));
        assert_eq!(two_sided_ceiling(13, Some(5)), (5, true));
        assert_eq!(two_sided_ceiling(13, Some(13)), (13, false));
        // A zero model side binds ITSELF (nothing registers — the
        // conservative posture); the machine term is not the binding side.
        assert_eq!(two_sided_ceiling(0, Some(5)), (0, false));
    }

    /// Machine half, live on this box (cheap syscalls, no ANE device):
    /// sane magnitudes only — no box-specific golden, and no assertion that
    /// compares TWO live reads (memory moves between them).
    #[test]
    fn live_measurement_magnitudes() {
        let total = total_bytes().expect("hw.memsize is always readable");
        assert!(total >= 4 * 1024 * 1024 * 1024, "total {total}");
        let avail = available_bytes().expect("host_statistics64 readable on macOS");
        assert!(avail >= 64 * 1024 * 1024, "avail {avail}");
        assert!(avail <= total, "avail {avail} > total {total}");
        let ceiling = machine_side_ceiling(0.70).expect("fraction gate over live read");
        // One-instant magnitude bounds only (the live ceiling is taken from
        // its OWN snapshot; comparing it against a second read of `avail`
        // would race a moving quantity).
        assert!(ceiling > 0 && ceiling < total);
    }

    /// The fraction clamp is part of the contract (env input is untrusted) —
    /// asserted against the PURE half (a second live read would compare two
    /// different instants of a moving quantity; the clamp belongs to the
    /// arithmetic, not the snapshot).
    #[test]
    fn fraction_clamp_bounds_the_machine_term() {
        let avail = 10_000u64;
        assert_eq!(machine_side_ceiling_from(avail, 1.5), avail);
        assert_eq!(machine_side_ceiling_from(avail, -0.2), 0);
        assert_eq!(machine_side_ceiling_from(avail, 0.70), 7_000);
        assert_eq!(machine_side_ceiling_from(avail, 1.0), avail);
    }
}
