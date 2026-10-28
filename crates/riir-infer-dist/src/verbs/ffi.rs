//! Raw verbs FFI for Apple's RDMA library (Plan 623 Phase 3 T3.1).
//!
//! `dlopen("librdma.dylib")` + `dlsym` for every exported `ibv_*` symbol
//! we use — no link-time dependency (the crate stays std-only, and a box
//! without the library fails LOUD at open, never at link). The four verbs
//! the C header ships as inline functions (`ibv_post_send`, `ibv_post_recv`,
//! `ibv_poll_cq`, `ibv_req_notify_cq`) are called through the
//! `ibv_context.ops` vtable — mirrored below in the header's exact field
//! order.
//!
//! Every struct here mirrors
//! `<SDK>/usr/include/infiniband/verbs.h` (macOS 26.2+, Apple copyright
//! 2025 — the Thunderbolt RDMA variant of libibverbs). The layout pins in
//! `tests/verbs_layout.rs` hold this mirror to the C compiler's measured
//! offsets (arm64-apple-macos: enum = i32, size_t = u64, pointers 8-byte
//! aligned). macOS-ONLY: the ops vtable layout is this header's — rdma-core
//! on Linux has evolved it and is deliberately not claimed.

#![allow(non_camel_case_types)]
#![allow(clippy::upper_case_acronyms)]

use core::ffi::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// constants (the header's enums; C enums compile to i32 on this ABI)
// ---------------------------------------------------------------------------

/// `enum ibv_qp_state`
pub mod qp_state {
    pub const RESET: i32 = 0;
    pub const INIT: i32 = 1;
    pub const RTR: i32 = 2;
    pub const RTS: i32 = 3;
    pub const SQD: i32 = 4;
    pub const SQE: i32 = 5;
    pub const ERR: i32 = 6;
    pub const UNKNOWN: i32 = 7;
}

/// `enum ibv_qp_type`
pub mod qp_type {
    pub const RC: i32 = 2;
    pub const UC: i32 = 3;
    pub const UD: i32 = 4;
}

/// `enum ibv_mtu`
pub mod mtu {
    pub const MTU_256: u32 = 1;
    pub const MTU_512: u32 = 2;
    pub const MTU_1024: u32 = 3;
    pub const MTU_2048: u32 = 4;
    pub const MTU_4096: u32 = 5;
}

/// `enum ibv_port_state`
pub mod port_state {
    pub const DOWN: i32 = 1;
    pub const INIT: i32 = 2;
    pub const ARMED: i32 = 3;
    pub const ACTIVE: i32 = 4;
}

/// anonymous link-layer enum in the header
pub mod link_layer {
    pub const INFINIBAND: u8 = 1;
    pub const ETHERNET: u8 = 2;
    pub const THUNDERBOLT: u8 = 100;
}

/// `enum ibv_access_flags`
pub mod access {
    pub const LOCAL_WRITE: u32 = 1;
    pub const REMOTE_WRITE: u32 = 1 << 1;
    pub const REMOTE_READ: u32 = 1 << 2;
    pub const REMOTE_ATOMIC: u32 = 1 << 3;
}

/// `enum ibv_qp_attr_mask`
pub mod attr_mask {
    use core::ffi::c_int;

    pub const STATE: c_int = 1 << 0;
    pub const ACCESS_FLAGS: c_int = 1 << 3;
    pub const PKEY_INDEX: c_int = 1 << 4;
    pub const PORT: c_int = 1 << 5;
    pub const AV: c_int = 1 << 7;
    pub const PATH_MTU: c_int = 1 << 8;
    pub const TIMEOUT: c_int = 1 << 9;
    pub const RQ_PSN: c_int = 1 << 12;
    pub const SQ_PSN: c_int = 1 << 16;
    pub const DEST_QPN: c_int = 1 << 20;
}

/// `enum ibv_wr_opcode`
pub mod wr_opcode {
    pub const RDMA_WRITE: u32 = 0;
    pub const SEND: u32 = 2;
    pub const RDMA_READ: u32 = 4;
}

/// `enum ibv_send_flags`
pub mod send_flags {
    pub const FENCE: u32 = 1 << 0;
    pub const SIGNALED: u32 = 1 << 1;
    pub const SOLICITED: u32 = 1 << 2;
    pub const INLINE: u32 = 1 << 3;
}

/// `enum ibv_wc_opcode`
pub mod wc_opcode {
    pub const SEND: u32 = 0;
    pub const RDMA_WRITE: u32 = 1;
    pub const RDMA_READ: u32 = 2;
    /// `(opcode & IBV_WC_RECV) != 0` ⇒ a receive.
    pub const RECV: u32 = 1 << 7;
}

/// `enum ibv_wc_status` — success only; unknown statuses render
/// through the exported `ibv_wc_status_str`.
pub mod wc_status {
    pub const SUCCESS: i32 = 0;
}

// ---------------------------------------------------------------------------
// structs — exact mirrors of the header (layout pins in tests/verbs_layout.rs)
// ---------------------------------------------------------------------------

/// `union ibv_gid` — 16-byte GID; the header's `raw` view.
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvGid {
    pub raw: [u8; 16],
    pub global: IbvGidGlobal,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvGidGlobal {
    pub subnet_prefix: u64,
    pub interface_id: u64,
}

/// `struct ibv_gid_entry` (the `_ibv_query_gid_table` row).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvGidEntry {
    pub gid: IbvGid,
    pub gid_index: u32,
    pub port_num: u32,
    pub gid_type: u32,
    pub ndev_ifindex: u32,
}

/// `struct ibv_wc` — work completion (48 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvWc {
    pub wr_id: u64,
    pub status: i32,
    pub opcode: u32,
    pub vendor_err: u32,
    pub byte_len: u32,
    pub imm_data: u32,
    pub qp_num: u32,
    pub src_qp: u32,
    pub wc_flags: u32,
    pub pkey_index: u16,
    pub slid: u16,
    pub sl: u8,
    pub dlid_path_bits: u8,
}

/// `struct ibv_pd` — we only ever point at library-allocated PDs.
#[repr(C)]
pub struct IbvPd {
    pub context: *mut IbvContext,
    pub handle: u32,
}

/// `struct ibv_mr` — lkey/rkey live here.
#[repr(C)]
pub struct IbvMr {
    pub context: *mut IbvContext,
    pub pd: *mut IbvPd,
    pub addr: *mut c_void,
    pub length: usize,
    pub handle: u32,
    pub lkey: u32,
    pub rkey: u32,
}

/// `struct ibv_sge` — scatter/gather entry.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

/// The `wr` payload union of `ibv_send_wr` (rdma | atomic | ud). Only
/// `rdma` is ever read here, but the whole union is mirrored so the
/// trailing offsets stay exact.
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvSendWrId {
    pub rdma: IbvRdmaPart,
    /// atomic { remote_addr, compare_add, swap, rkey } — size-only.
    pub atomic_: [u64; 4],
    /// ud { ah, remote_qpn, remote_qkey } — size-only.
    pub ud_: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvRdmaPart {
    pub remote_addr: u64,
    pub rkey: u32,
}

/// The trailing `union { bind_mw | tso }` — size-only (we never bind MWs
/// or use TSO), kept so `IbvSendWr`'s total size matches.
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvSendWrTail {
    /// bind_mw is 48 bytes.
    pub bind_mw_: [u64; 6],
    /// tso is 16 bytes.
    pub tso_: [u64; 2],
}

/// `struct ibv_send_wr` — 128 bytes on this ABI.
#[repr(C)]
pub struct IbvSendWr {
    pub wr_id: u64,
    pub next: *mut IbvSendWr,
    pub sg_list: *mut IbvSge,
    pub num_sge: c_int,
    pub opcode: u32,
    pub send_flags: u32,
    /// `union { imm_data; invalidate_rkey; }`
    pub imm_data: u32,
    pub wr: IbvSendWrId,
    /// `union { struct { uint32_t remote_srqn; } xrc; }` — 4 bytes.
    pub qp_type_xrc: u32,
    /// `union { bind_mw | tso }` — 48 bytes.
    pub bind_mw: IbvSendWrTail,
}

/// `struct ibv_recv_wr` — 32 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvRecvWr {
    pub wr_id: u64,
    pub next: *mut IbvRecvWr,
    pub sg_list: *mut IbvSge,
    pub num_sge: c_int,
}

/// `struct ibv_qp_cap`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IbvQpCap {
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub max_inline_data: u32,
}

/// `struct ibv_qp_init_attr` — caller-allocated at `ibv_create_qp`; the
/// library writes the ACTUAL caps back into `cap`.
#[repr(C)]
pub struct IbvQpInitAttr {
    pub qp_context: *mut c_void,
    pub send_cq: *mut IbvCq,
    pub recv_cq: *mut IbvCq,
    pub srq: *mut c_void,
    pub cap: IbvQpCap,
    pub qp_type: i32,
    pub sq_sig_all: c_int,
}

/// `struct ibv_global_route`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvGlobalRoute {
    pub dgid: IbvGid,
    pub flow_label: u32,
    pub sgid_index: u8,
    pub hop_limit: u8,
    pub traffic_class: u8,
}

/// `struct ibv_ah_attr` — 32 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvAhAttr {
    pub grh: IbvGlobalRoute,
    pub dlid: u16,
    pub sl: u8,
    pub src_path_bits: u8,
    pub static_rate: u8,
    pub is_global: u8,
    pub port_num: u8,
}

/// `struct ibv_qp_attr` — 144 bytes; caller-allocated at modify/query.
#[repr(C)]
pub struct IbvQpAttr {
    pub qp_state: i32,
    pub cur_qp_state: i32,
    pub path_mtu: u32,
    pub path_mig_state: i32,
    pub qkey: u32,
    pub rq_psn: u32,
    pub sq_psn: u32,
    pub dest_qp_num: u32,
    pub qp_access_flags: u32,
    pub cap: IbvQpCap,
    pub ah_attr: IbvAhAttr,
    pub alt_ah_attr: IbvAhAttr,
    pub pkey_index: u16,
    pub alt_pkey_index: u16,
    pub en_sqd_async_notify: u8,
    pub sq_draining: u8,
    pub max_rd_atomic: u8,
    pub max_dest_rd_atomic: u8,
    pub min_rnr_timer: u8,
    pub port_num: u8,
    pub timeout: u8,
    pub retry_cnt: u8,
    pub rnr_retry: u8,
    pub alt_port_num: u8,
    pub alt_timeout: u8,
    pub rate_limit: u32,
}

/// `struct ibv_qp` — library-allocated; we read `qp_num`/`state` and hand
/// the pointer back. The opaque tail (mutex 64 + cond 48 + counter) is
/// size-only.
#[repr(C)]
pub struct IbvQp {
    pub context: *mut IbvContext,
    pub qp_context: *mut c_void,
    pub pd: *mut IbvPd,
    pub send_cq: *mut IbvCq,
    pub recv_cq: *mut IbvCq,
    pub srq: *mut c_void,
    pub handle: u32,
    pub qp_num: u32,
    pub state: i32,
    pub qp_type: i32,
    /// 120 bytes: mutex(64)+cond(48)+events_completed(4)+pad.
    pub _opaque_tail: [u64; 15],
}

/// `struct ibv_comp_channel`.
#[repr(C)]
pub struct IbvCompChannel {
    pub context: *mut IbvContext,
    pub fd: c_int,
    pub refcnt: c_int,
}

/// `struct ibv_cq` — library-allocated; the opaque tail is size-only.
#[repr(C)]
pub struct IbvCq {
    pub context: *mut IbvContext,
    pub channel: *mut IbvCompChannel,
    pub cq_context: *mut c_void,
    pub handle: u32,
    pub cqe: c_int,
    /// 120 bytes: mutex(64)+cond(48)+2 counters(8)+pad.
    pub _opaque_tail: [u64; 15],
}

/// `struct ibv_device` — name strings are fixed arrays.
#[repr(C)]
pub struct IbvDevice {
    pub _ops: [*mut c_void; 2],
    pub node_type: i32,
    pub transport_type: i32,
    pub name: [c_char; 64],
    pub dev_name: [c_char; 64],
    pub dev_path: [c_char; 256],
    pub ibdev_path: [c_char; 256],
}

// vtable slot types (the four inline verbs we call through ctx->ops)
pub type PollCqFn = unsafe extern "C" fn(*mut IbvCq, c_int, *mut IbvWc) -> c_int;
pub type ReqNotifyCqFn = unsafe extern "C" fn(*mut IbvCq, c_int) -> c_int;
pub type PostSendFn =
    unsafe extern "C" fn(*mut IbvQp, *mut IbvSendWr, *mut *mut IbvSendWr) -> c_int;
pub type PostRecvFn =
    unsafe extern "C" fn(*mut IbvQp, *mut IbvRecvWr, *mut *mut IbvRecvWr) -> c_int;

/// `struct ibv_context_ops` — the header's EXACT 32-slot order; the four
/// slots we call are typed, the rest stay opaque (same 8-byte size, so
/// every offset matches). Compiler-measured: poll_cq@88, req_notify@96,
/// post_srq_recv@160, post_send@200, post_recv@208.
#[repr(C)]
pub struct IbvContextOps {
    _compat_query_device: *const c_void,
    _compat_query_port: *const c_void,
    _compat_alloc_pd: *const c_void,
    _compat_dealloc_pd: *const c_void,
    _compat_reg_mr: *const c_void,
    _compat_rereg_mr: *const c_void,
    _compat_dereg_mr: *const c_void,
    _alloc_mw: *const c_void,
    _bind_mw: *const c_void,
    _dealloc_mw: *const c_void,
    _compat_create_cq: *const c_void,
    pub poll_cq: PollCqFn,
    pub req_notify_cq: ReqNotifyCqFn,
    _compat_cq_event: *const c_void,
    _compat_resize_cq: *const c_void,
    _compat_destroy_cq: *const c_void,
    _compat_create_srq: *const c_void,
    _compat_modify_srq: *const c_void,
    _compat_query_srq: *const c_void,
    _compat_destroy_srq: *const c_void,
    _post_srq_recv: *const c_void,
    _compat_create_qp: *const c_void,
    _compat_query_qp: *const c_void,
    _compat_modify_qp: *const c_void,
    _compat_destroy_qp: *const c_void,
    pub post_send: PostSendFn,
    pub post_recv: PostRecvFn,
    _compat_create_ah: *const c_void,
    _compat_destroy_ah: *const c_void,
    _compat_attach_mcast: *const c_void,
    _compat_detach_mcast: *const c_void,
    _compat_async_event: *const c_void,
}

/// `struct ibv_context` — library-allocated. `ops` sits at offset 8; the
/// opaque tail (mutex 64 + abi_compat) is size-only.
#[repr(C)]
pub struct IbvContext {
    pub device: *mut IbvDevice,
    pub ops: IbvContextOps,
    pub cmd_fd: c_int,
    pub async_fd: c_int,
    pub num_comp_vectors: c_int,
    /// 72 bytes: mutex(64)+abi_compat(8).
    pub _opaque_tail: [u64; 9],
}

/// `struct ibv_port_attr` — the EXTENDED (current) layout, 52 bytes. The
/// exported `ibv_query_port` writes the legacy prefix (everything through
/// `link_layer`, 48 bytes); the trailing two fields stay zero when only
/// the compat entry ran.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvPortAttr {
    pub state: i32,
    pub max_mtu: u32,
    pub active_mtu: u32,
    pub gid_tbl_len: c_int,
    pub port_cap_flags: u32,
    pub max_msg_sz: u32,
    pub bad_pkey_cntr: u32,
    pub qkey_viol_cntr: u32,
    pub pkey_tbl_len: u16,
    pub lid: u16,
    pub sm_lid: u16,
    pub lmc: u8,
    pub max_vl_num: u8,
    pub sm_sl: u8,
    pub subnet_timeout: u8,
    pub init_type_reply: u8,
    pub active_width: u8,
    pub active_speed: u8,
    pub phys_state: u8,
    pub link_layer: u8,
    pub flags: u8,
    pub port_cap_flags2: u16,
}

/// `struct ibv_device_attr` — 232 bytes; we read a few caps.
#[repr(C)]
pub struct IbvDeviceAttr {
    pub fw_ver: [c_char; 64],
    pub node_guid: u64,
    pub sys_image_guid: u64,
    pub max_mr_size: u64,
    pub page_size_cap: u64,
    pub vendor_id: u32,
    pub vendor_part_id: u32,
    pub hw_ver: u32,
    pub max_qp: c_int,
    pub max_qp_wr: c_int,
    pub device_cap_flags: u32,
    pub max_sge: c_int,
    pub max_sge_rd: c_int,
    pub max_cq: c_int,
    pub max_cqe: c_int,
    pub max_mr: c_int,
    pub max_pd: c_int,
    pub max_qp_rd_atom: c_int,
    pub max_ee_rd_atom: c_int,
    pub max_res_rd_atom: c_int,
    pub max_qp_init_rd_atom: c_int,
    pub max_ee_init_rd_atom: c_int,
    pub atomic_cap: c_int,
    pub max_ee: c_int,
    pub max_rdd: c_int,
    pub max_mw: c_int,
    pub max_raw_ipv6_qp: c_int,
    pub max_raw_ethy_qp: c_int,
    pub max_mcast_grp: c_int,
    pub max_mcast_qp_attach: c_int,
    pub max_total_mcast_qp_attach: c_int,
    pub max_ah: c_int,
    pub max_fmr: c_int,
    pub max_map_per_fmr: c_int,
    pub max_srq: c_int,
    pub max_srq_wr: c_int,
    pub max_srq_sge: c_int,
    pub max_pkeys: u16,
    pub local_ca_ack_delay: u8,
    pub phys_port_cnt: u8,
}

// ---------------------------------------------------------------------------
// dlopen / dlsym — declared extern "C" (libSystem on macOS; every Rust
// binary links it) — the sockopts.rs precedent, no libc crate.
// ---------------------------------------------------------------------------

const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

fn last_dlerror() -> String {
    unsafe {
        let p = dlerror();
        if p.is_null() {
            "unknown dlopen/dlsym failure".into()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// The typed symbol table. Every entry is dlsym'd at open; a missing
/// symbol is a loud named error (never a null deref later).
pub struct Syms {
    pub get_device_list: unsafe extern "C" fn(num_devices: *mut c_int) -> *mut *mut IbvDevice,
    pub free_device_list: unsafe extern "C" fn(list: *mut *mut IbvDevice),
    pub get_device_name: unsafe extern "C" fn(device: *mut IbvDevice) -> *const c_char,
    pub get_device_guid: unsafe extern "C" fn(device: *mut IbvDevice) -> u64,
    pub open_device: unsafe extern "C" fn(device: *mut IbvDevice) -> *mut IbvContext,
    pub close_device: unsafe extern "C" fn(context: *mut IbvContext) -> c_int,
    pub query_device: unsafe extern "C" fn(*mut IbvContext, *mut IbvDeviceAttr) -> c_int,
    /// The EXPORTED compat entry — takes the legacy-layout port attr (the
    /// first 48 bytes of [`IbvPortAttr`]); we hand it a full-size zeroed
    /// buffer, which strictly contains the legacy layout.
    pub query_port: unsafe extern "C" fn(*mut IbvContext, u8, *mut IbvPortAttr) -> c_int,
    pub query_gid: unsafe extern "C" fn(*mut IbvContext, u8, c_int, *mut IbvGid) -> c_int,
    pub alloc_pd: unsafe extern "C" fn(*mut IbvContext) -> *mut IbvPd,
    pub dealloc_pd: unsafe extern "C" fn(*mut IbvPd) -> c_int,
    pub reg_mr: unsafe extern "C" fn(*mut IbvPd, *mut c_void, usize, c_int) -> *mut IbvMr,
    pub dereg_mr: unsafe extern "C" fn(*mut IbvMr) -> c_int,
    pub create_cq: unsafe extern "C" fn(
        *mut IbvContext,
        c_int,
        *mut c_void,
        *mut IbvCompChannel,
        c_int,
    ) -> *mut IbvCq,
    pub destroy_cq: unsafe extern "C" fn(*mut IbvCq) -> c_int,
    pub create_qp: unsafe extern "C" fn(*mut IbvPd, *mut IbvQpInitAttr) -> *mut IbvQp,
    pub modify_qp: unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int) -> c_int,
    pub query_qp:
        unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int, *mut IbvQpInitAttr) -> c_int,
    pub destroy_qp: unsafe extern "C" fn(*mut IbvQp) -> c_int,
    pub wc_status_str: unsafe extern "C" fn(i32) -> *const c_char,
    /// Keep the dlopen handle alive for the process lifetime. Never
    /// `dlclose`d: live verbs objects would dangle across a close.
    _handle: *mut c_void,
}

// The handle is process-global shared library state (like the symbols
// themselves); the FFI entry points are thread-safe per Apple's provider.
unsafe impl Send for Syms {}
unsafe impl Sync for Syms {}

impl Syms {
    /// `dlopen("librdma.dylib")` + resolve every symbol. Fails loud
    /// naming the library and the first missing symbol.
    pub fn load() -> Result<Self, String> {
        let path = b"librdma.dylib\0";
        unsafe {
            // Clear any stale error state first (dlerror is sticky).
            dlerror();
            let handle = dlopen(path.as_ptr().cast::<c_char>(), RTLD_NOW);
            if handle.is_null() {
                return Err(format!(
                    "dlopen(\"librdma.dylib\") failed: {} — the RDMA lane needs macOS 26.2+ with RDMA over Thunderbolt (rdma_ctl)",
                    last_dlerror()
                ));
            }
            let sym = Self {
                get_device_list: Self::sym(handle, "ibv_get_device_list")?,
                free_device_list: Self::sym(handle, "ibv_free_device_list")?,
                get_device_name: Self::sym(handle, "ibv_get_device_name")?,
                get_device_guid: Self::sym(handle, "ibv_get_device_guid")?,
                open_device: Self::sym(handle, "ibv_open_device")?,
                close_device: Self::sym(handle, "ibv_close_device")?,
                query_device: Self::sym(handle, "ibv_query_device")?,
                query_port: Self::sym(handle, "ibv_query_port")?,
                query_gid: Self::sym(handle, "ibv_query_gid")?,
                alloc_pd: Self::sym(handle, "ibv_alloc_pd")?,
                dealloc_pd: Self::sym(handle, "ibv_dealloc_pd")?,
                reg_mr: Self::sym(handle, "ibv_reg_mr")?,
                dereg_mr: Self::sym(handle, "ibv_dereg_mr")?,
                create_cq: Self::sym(handle, "ibv_create_cq")?,
                destroy_cq: Self::sym(handle, "ibv_destroy_cq")?,
                create_qp: Self::sym(handle, "ibv_create_qp")?,
                modify_qp: Self::sym(handle, "ibv_modify_qp")?,
                query_qp: Self::sym(handle, "ibv_query_qp")?,
                destroy_qp: Self::sym(handle, "ibv_destroy_qp")?,
                wc_status_str: Self::sym(handle, "ibv_wc_status_str")?,
                _handle: handle,
            };
            Ok(sym)
        }
    }

    unsafe fn sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
        let mut buf = Vec::with_capacity(name.len() + 1);
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        unsafe {
            dlerror();
            let p = dlsym(handle, buf.as_ptr().cast::<c_char>());
            if p.is_null() {
                let err = last_dlerror();
                return Err(if err.is_empty() {
                    format!("dlsym(\"{name}\") resolved to NULL")
                } else {
                    format!("dlsym(\"{name}\") failed: {err}")
                });
            }
            // The symbol exists; give it its real type.
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&p))
        }
    }
}

impl Syms {
    /// Render a WC status through the library's own strings.
    pub fn wc_status(&self, status: i32) -> String {
        unsafe {
            let p = (self.wc_status_str)(status);
            if p.is_null() {
                format!("ibv status {status}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}
