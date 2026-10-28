//! Safe verbs wrappers + UC QP bring-up (Plan 623 Phase 3 T3.1).
//!
//! Layering: [`Rdma`] owns the dlopen'd symbol table; [`DevCtx`] owns one
//! opened device (Drop ⇒ `ibv_close_device`); [`ProtDom`] a protection
//! domain; [`CompQ`] a completion queue; [`MemReg`] a registered memory
//! region (lkey/rkey); [`Qp`] a queue pair with the INIT→RTR→RTS state
//! walk. Ownership is an Arc chain (Qp → DevCtx → Rdma), so drop order is
//! always child-before-parent and a wrapper can never outlive its context.
//!
//! Bring-up targets the plan's UC posture (no retry machinery, lowest
//! latency) with IPv4-mapped GIDs over the Thunderbolt link layer:
//! INIT (pkey 0, port, access) → RTR (peer QPN/PSN/MTU, AH with the peer
//! GID global) → RTS (SQ PSN, timeout). Both sides run the SAME walk with
//! each other's [`Endpoint`], exchanged over the TCP side channel
//! ([`exchange_endpoint`], BLAKE3-keyed MAC — no unauthenticated
//! endpoint is ever trusted).

use std::net::TcpStream;
use std::sync::Arc;
use std::time::Instant;

pub mod ffi;

use core::ffi::c_int;
use ffi::{
    IbvContext, IbvCq, IbvDevice, IbvGid, IbvMr, IbvPd, IbvPortAttr, IbvQp, IbvQpAttr, IbvQpCap,
    IbvQpInitAttr, IbvRecvWr, IbvSendWr, IbvSge, IbvWc, Syms, attr_mask, link_layer, mtu,
    port_state, qp_state, qp_type, send_flags, wc_status,
};

// ---------------------------------------------------------------------------
// Rdma — the process-wide symbol table
// ---------------------------------------------------------------------------

/// The dlopen'd library. Cheap to clone (Arc); load once.
#[derive(Clone)]
pub struct Rdma {
    syms: Arc<Syms>,
}

impl Rdma {
    /// `dlopen("librdma.dylib")` + resolve every symbol. Fails loud.
    pub fn open() -> Result<Self, String> {
        Ok(Self {
            syms: Arc::new(Syms::load()?),
        })
    }

    /// Names of the RDMA devices present (usually one per TB bridge).
    pub fn device_names(&self) -> Vec<String> {
        unsafe {
            let mut n: c_int = 0;
            let list = (self.syms.get_device_list)(&mut n);
            if list.is_null() {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(n.max(0) as usize);
            for i in 0..n {
                let d = *list.add(i as usize);
                if d.is_null() {
                    break; // the array is NULL-terminated
                }
                let name = (self.syms.get_device_name)(d);
                let s = if name.is_null() {
                    format!("dev{i}")
                } else {
                    std::ffi::CStr::from_ptr(name)
                        .to_string_lossy()
                        .into_owned()
                };
                out.push(s);
            }
            (self.syms.free_device_list)(list);
            out
        }
    }

    /// Open the named device, or the first present one when `name` is
    /// `None`. Fails loud naming the device when there is none — a box
    /// without an RDMA endpoint is a config fact, never a silent lane.
    pub fn open_device(&self, name: Option<&str>) -> Result<DevCtx, String> {
        let names = self.device_names();
        if names.is_empty() {
            return Err(
                "no RDMA devices present — enable the endpoint with `rdma_ctl` (macOS 26.2+, \
                 Thunderbolt link), or pass the right device once it enumerates"
                    .into(),
            );
        }
        let chosen = match name {
            Some(want) => {
                if !names.iter().any(|n| n == want) {
                    return Err(format!(
                        "RDMA device {want:?} not present; devices = {names:?}"
                    ));
                }
                want.to_string()
            }
            None => names[0].clone(),
        };
        let mut n: c_int = 0;
        let list = unsafe { (self.syms.get_device_list)(&mut n) };
        if list.is_null() {
            return Err("ibv_get_device_list returned NULL on re-list".into());
        }
        let mut found: *mut IbvDevice = std::ptr::null_mut();
        for i in 0..n {
            let d = unsafe { *list.add(i as usize) };
            if d.is_null() {
                break;
            }
            let dev_name = unsafe { (self.syms.get_device_name)(d) };
            if !dev_name.is_null() {
                let s = unsafe { std::ffi::CStr::from_ptr(dev_name) }
                    .to_string_lossy()
                    .into_owned();
                if s == chosen {
                    found = d;
                    break;
                }
            }
        }
        if found.is_null() {
            unsafe { (self.syms.free_device_list)(list) };
            return Err(format!("RDMA device {chosen:?} vanished between listings"));
        }
        let ctx = unsafe { (self.syms.open_device)(found) };
        // NOTE: the device array must stay allocated until open_device
        // returns; free it only now.
        unsafe { (self.syms.free_device_list)(list) };
        if ctx.is_null() {
            return Err(format!("ibv_open_device({chosen}) failed"));
        }
        Ok(DevCtx {
            rdma: self.clone(),
            ptr: ctx,
            dev_name: chosen,
        })
    }
}

// ---------------------------------------------------------------------------
// DevCtx
// ---------------------------------------------------------------------------

/// An opened device context. `Clone` is an Arc clone — all handles share
/// one `ibv_context`.
pub struct DevCtx {
    rdma: Rdma,
    ptr: *mut IbvContext,
    pub dev_name: String,
}

unsafe impl Send for DevCtx {}
unsafe impl Sync for DevCtx {}

impl DevCtx {
    /// Device caps (the few the lane reads).
    pub fn caps(&self) -> Result<DeviceCaps, String> {
        let mut a: ffi::IbvDeviceAttr = unsafe { std::mem::zeroed() };
        let rc = unsafe { (self.rdma.syms.query_device)(self.ptr, &mut a) };
        if rc != 0 {
            return Err(format!("ibv_query_device rc={rc}"));
        }
        Ok(DeviceCaps {
            max_qp: a.max_qp,
            max_qp_wr: a.max_qp_wr,
            max_sge: a.max_sge,
            max_cq: a.max_cq,
            max_cqe: a.max_cqe,
            max_mr_size: a.max_mr_size,
        })
    }

    /// Port attributes (through the exported compat entry — the legacy
    /// prefix of the struct, which carries everything the lane reads).
    pub fn query_port(&self, port: u8) -> Result<PortAttr, String> {
        let mut a: IbvPortAttr = unsafe { std::mem::zeroed() };
        let rc = unsafe { (self.rdma.syms.query_port)(self.ptr, port, &mut a) };
        if rc != 0 {
            return Err(format!("ibv_query_port(port {port}) rc={rc}"));
        }
        Ok(PortAttr {
            state: a.state,
            active_mtu: a.active_mtu,
            gid_tbl_len: a.gid_tbl_len,
            max_msg_sz: a.max_msg_sz,
            link_layer: a.link_layer,
        })
    }

    /// The first port whose state is ACTIVE; `(port_num, attrs)`.
    pub fn first_active_port(&self) -> Result<(u8, PortAttr), String> {
        for port in 1..=2u8 {
            let p = self.query_port(port)?;
            if p.state == port_state::ACTIVE {
                return Ok((port, p));
            }
        }
        Err(format!(
            "device {} has no ACTIVE port (rdma_ctl enable needed?)",
            self.dev_name
        ))
    }

    /// A GID table entry as raw bytes.
    pub fn gid(&self, port: u8, index: u32) -> Result<[u8; 16], String> {
        let mut g: IbvGid = unsafe { std::mem::zeroed() };
        let rc = unsafe { (self.rdma.syms.query_gid)(self.ptr, port, index as c_int, &mut g) };
        if rc != 0 {
            return Err(format!("ibv_query_gid(port {port}, idx {index}) rc={rc}"));
        }
        Ok(unsafe { g.raw })
    }

    /// Pick the local source GID index for a path toward `peer_v4`:
    /// prefer the IPv4-mapped GID whose v4 byte tail equals the peer's
    /// TCP address (same link), then any IPv4-mapped GID, then index 0.
    /// Returns `(index, raw)`.
    pub fn pick_gid(
        &self,
        port: u8,
        peer_v4: Option<std::net::Ipv4Addr>,
    ) -> Result<(u32, [u8; 16]), String> {
        let len = self.query_port(port)?.gid_tbl_len;
        if len <= 0 {
            return Err(format!(
                "device {} port {port} has an empty GID table",
                self.dev_name
            ));
        }
        let mut any_mapped: Option<(u32, [u8; 16])> = None;
        for idx in 0..len as u32 {
            let raw = self.gid(port, idx)?;
            let is_v4_mapped = raw[..12] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff];
            if !is_v4_mapped {
                continue;
            }
            let v4 = std::net::Ipv4Addr::new(raw[12], raw[13], raw[14], raw[15]);
            if let Some(peer) = peer_v4
                && v4 == peer
            {
                return Ok((idx, raw));
            }
            if any_mapped.is_none() {
                any_mapped = Some((idx, raw));
            }
        }
        if let Some(hit) = any_mapped {
            return Ok(hit);
        }
        Ok((0, self.gid(port, 0)?))
    }

    /// Allocate a protection domain.
    pub fn alloc_pd(self: &Arc<Self>) -> Result<ProtDom, String> {
        let pd = unsafe { (self.rdma.syms.alloc_pd)(self.ptr) };
        if pd.is_null() {
            return Err(format!("ibv_alloc_pd failed on {}", self.dev_name));
        }
        Ok(ProtDom {
            ctx: self.clone(),
            ptr: pd,
        })
    }
}

impl Drop for DevCtx {
    fn drop(&mut self) {
        unsafe {
            (self.rdma.syms.close_device)(self.ptr);
        }
    }
}

/// The device caps the lane reads.
#[derive(Debug, Clone, Copy)]
pub struct DeviceCaps {
    pub max_qp: i32,
    pub max_qp_wr: i32,
    pub max_sge: i32,
    pub max_cq: i32,
    pub max_cqe: i32,
    pub max_mr_size: u64,
}

/// The port facts the lane reads.
#[derive(Debug, Clone, Copy)]
pub struct PortAttr {
    pub state: i32,
    pub active_mtu: u32,
    pub gid_tbl_len: i32,
    pub max_msg_sz: u32,
    pub link_layer: u8,
}

impl PortAttr {
    pub fn link_layer_name(&self) -> &'static str {
        match self.link_layer {
            link_layer::THUNDERBOLT => "thunderbolt",
            link_layer::ETHERNET => "ethernet",
            link_layer::INFINIBAND => "infiniband",
            _ => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// ProtDom / MemReg
// ---------------------------------------------------------------------------

/// A protection domain.
pub struct ProtDom {
    ctx: Arc<DevCtx>,
    ptr: *mut IbvPd,
}

unsafe impl Send for ProtDom {}
unsafe impl Sync for ProtDom {}

impl ProtDom {
    /// Register a memory range for RDMA. The buffer must outlive the
    /// returned [`MemReg`]; caller keeps ownership of the bytes.
    pub fn reg_mr(&self, addr: *mut u8, len: usize, acc: u32) -> Result<MemReg<'_>, String> {
        if len == 0 {
            return Err("reg_mr: zero-length registration refused".into());
        }
        let mr = unsafe { (self.ctx.rdma.syms.reg_mr)(self.ptr, addr.cast(), len, acc as c_int) };
        if mr.is_null() {
            return Err(format!("ibv_reg_mr(len {len}) failed"));
        }
        let (lkey, rkey, handle) = unsafe { ((*mr).lkey, (*mr).rkey, (*mr).handle) };
        Ok(MemReg {
            pd: self,
            ptr: mr,
            lkey,
            rkey,
            handle,
            addr: addr as u64,
            len,
        })
    }

    /// Create a completion queue with at least `cqe` entries.
    pub fn create_cq(self: &Arc<Self>, cqe: u32) -> Result<CompQ, String> {
        let cq = unsafe {
            (self.ctx.rdma.syms.create_cq)(
                self.ctx.ptr,
                cqe as c_int,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if cq.is_null() {
            return Err(format!("ibv_create_cq({cqe}) failed"));
        }
        Ok(CompQ {
            pd: self.clone(),
            ptr: cq,
            cqe,
        })
    }

    /// Create a UC queue pair on this PD. Returns the QP plus the caps
    /// the device actually granted (the library writes them back into
    /// the init attr — requested ≠ granted, the granted wins).
    pub fn create_uc_qp(
        self: &Arc<Self>,
        send_cq: &Arc<CompQ>,
        recv_cq: &Arc<CompQ>,
        cap: IbvQpCap,
    ) -> Result<(Qp, IbvQpCap), String> {
        Self::create_qp_typed(qp_type::UC, self, send_cq, recv_cq, cap)
    }

    /// Create an RC queue pair on this PD. Same contract as
    /// [`ProtDom::create_uc_qp`]; the T3.2 write probe is the first
    /// consumer (RDMA READ is RC-only, so the probe needs the type).
    pub fn create_rc_qp(
        self: &Arc<Self>,
        send_cq: &Arc<CompQ>,
        recv_cq: &Arc<CompQ>,
        cap: IbvQpCap,
    ) -> Result<(Qp, IbvQpCap), String> {
        Self::create_qp_typed(qp_type::RC, self, send_cq, recv_cq, cap)
    }

    fn create_qp_typed(
        ty: i32,
        pd: &Arc<ProtDom>,
        send_cq: &Arc<CompQ>,
        recv_cq: &Arc<CompQ>,
        cap: IbvQpCap,
    ) -> Result<(Qp, IbvQpCap), String> {
        let mut init: IbvQpInitAttr = unsafe { std::mem::zeroed() };
        init.send_cq = send_cq.ptr;
        init.recv_cq = recv_cq.ptr;
        init.cap = cap;
        init.qp_type = ty;
        init.sq_sig_all = 0;
        let qp = unsafe { (pd.ctx.rdma.syms.create_qp)(pd.ptr, &mut init) };
        if qp.is_null() {
            return Err(format!("ibv_create_qp(type {ty}) failed"));
        }
        Ok((
            Qp {
                ctx: pd.ctx.clone(),
                ptr: qp,
                qp_ty: ty,
                // Owned CQ handles: the QP cannot outlive its CQs. Drop
                // order is enforced by the Arc chain — destroy_qp (the
                // struct's own Drop) runs first; the CQs, PD and context
                // then fall as their last Arc references release.
                _send_cq: send_cq.clone(),
                _recv_cq: recv_cq.clone(),
            },
            init.cap,
        ))
    }
}

impl Drop for ProtDom {
    fn drop(&mut self) {
        unsafe {
            (self.ctx.rdma.syms.dealloc_pd)(self.ptr);
        }
    }
}

/// A registered memory region.
pub struct MemReg<'a> {
    pd: &'a ProtDom,
    ptr: *mut IbvMr,
    pub lkey: u32,
    pub rkey: u32,
    pub handle: u32,
    pub addr: u64,
    pub len: usize,
}

impl MemReg<'_> {
    pub fn slice_at(&self, off: usize, len: u32) -> IbvSge {
        IbvSge {
            addr: self.addr + off as u64,
            length: len,
            lkey: self.lkey,
        }
    }
}

impl Drop for MemReg<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.pd.ctx.rdma.syms.dereg_mr)(self.ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// CompQ
// ---------------------------------------------------------------------------

/// A completion queue.
pub struct CompQ {
    pd: Arc<ProtDom>,
    ptr: *mut IbvCq,
    pub cqe: u32,
}

unsafe impl Send for CompQ {}
unsafe impl Sync for CompQ {}

impl CompQ {
    /// Poll up to `out.len()` completions; returns how many were reaped.
    /// Negative library return = CQ error, surfaced loud with the status
    /// string of the first slot when one is present.
    pub fn poll(&self, out: &mut [IbvWc]) -> Result<usize, String> {
        if out.is_empty() {
            return Ok(0);
        }
        let n = unsafe {
            (self.pd.ctx.raw_ctx_ops().poll_cq)(self.ptr, out.len() as c_int, out.as_mut_ptr())
        };
        if n < 0 {
            let why = if out[0].status != wc_status::SUCCESS {
                self.pd.ctx.rdma.syms.wc_status(out[0].status)
            } else {
                format!("rc={n}")
            };
            return Err(format!("ibv_poll_cq error: {why}"));
        }
        Ok(n as usize)
    }

    /// Busy-poll until at least one completion lands or the deadline
    /// passes. Yields every 256 empty rounds so a co-tenant core is not
    /// starved; the first rounds stay pure spin (RTT accuracy).
    pub fn poll_one(&self, deadline: Instant) -> Result<IbvWc, String> {
        let mut wc = IbvWc {
            wr_id: 0,
            status: 0,
            opcode: 0,
            vendor_err: 0,
            byte_len: 0,
            imm_data: 0,
            qp_num: 0,
            src_qp: 0,
            wc_flags: 0,
            pkey_index: 0,
            slid: 0,
            sl: 0,
            dlid_path_bits: 0,
        };
        let mut rounds: u32 = 0;
        loop {
            let n = self.poll(std::slice::from_mut(&mut wc))?;
            if n > 0 {
                return Ok(wc);
            }
            if Instant::now() >= deadline {
                return Err("poll_cq: completion timeout".into());
            }
            rounds = rounds.wrapping_add(1);
            if rounds.is_multiple_of(256) {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for CompQ {
    fn drop(&mut self) {
        unsafe {
            (self.pd.ctx.rdma.syms.destroy_cq)(self.ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Qp — UC bring-up + post/poll
// ---------------------------------------------------------------------------

pub struct Qp {
    ctx: Arc<DevCtx>,
    ptr: *mut IbvQp,
    /// The QP's transport type — RTS's attr mask is type-aware (the
    /// provider rejects the RC-only TIMEOUT attr on a UC QP: EINVAL,
    /// measured on the amo⇄bmo bring-up).
    qp_ty: i32,
    // Owned CQ handles — lifetime via the Arc chain, not field order:
    // the struct's Drop (destroy_qp) runs before any field drops, so the
    // CQs are still alive when the QP is destroyed.
    _send_cq: Arc<CompQ>,
    _recv_cq: Arc<CompQ>,
}

unsafe impl Send for Qp {}
unsafe impl Sync for Qp {}

impl Qp {
    pub fn qp_num(&self) -> u32 {
        unsafe { (*self.ptr).qp_num }
    }

    pub fn state(&self) -> i32 {
        unsafe { (*self.ptr).state }
    }

    fn modify(&self, attr: &IbvQpAttr, mask: c_int) -> Result<(), String> {
        let rc =
            unsafe { (self.ctx.rdma.syms.modify_qp)(self.ptr, attr as *const _ as *mut _, mask) };
        if rc != 0 {
            return Err(format!("ibv_modify_qp(mask {mask:#x}) rc={rc}"));
        }
        Ok(())
    }

    /// RESET → INIT (pkey index 0, this port, `acc` access flags).
    pub fn to_init(&self, port: u8, acc: u32) -> Result<(), String> {
        let mut a: IbvQpAttr = unsafe { std::mem::zeroed() };
        a.qp_state = qp_state::INIT;
        a.pkey_index = 0;
        a.port_num = port;
        a.qp_access_flags = acc;
        self.modify(
            &a,
            attr_mask::STATE | attr_mask::PKEY_INDEX | attr_mask::PORT | attr_mask::ACCESS_FLAGS,
        )
    }

    /// INIT → RTR toward `peer` (its QPN/PSN/MTU + GID), with `sgid_index`
    /// naming the LOCAL source GID for the path. Path MTU is clamped to
    /// the peer's advertised value by the CALLER (compose the endpoint
    /// with the clamped MTU before exchange).
    pub fn to_rtr(&self, peer: &Endpoint, sgid_index: u8, port: u8) -> Result<(), String> {
        let mut a: IbvQpAttr = unsafe { std::mem::zeroed() };
        a.qp_state = qp_state::RTR;
        a.dest_qp_num = peer.qpn;
        a.rq_psn = peer.psn & 0x00ff_ffff;
        a.path_mtu = peer.mtu;
        a.ah_attr.grh.dgid = IbvGid { raw: peer.gid };
        a.ah_attr.grh.flow_label = 0;
        a.ah_attr.grh.sgid_index = sgid_index;
        a.ah_attr.grh.hop_limit = 64;
        a.ah_attr.grh.traffic_class = 0;
        a.ah_attr.dlid = 0;
        a.ah_attr.sl = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.static_rate = 0;
        a.ah_attr.is_global = 1;
        a.ah_attr.port_num = port;
        self.modify(
            &a,
            attr_mask::STATE
                | attr_mask::DEST_QPN
                | attr_mask::RQ_PSN
                | attr_mask::PATH_MTU
                | attr_mask::AV,
        )
    }

    /// RTR → RTS (own SQ PSN). The response TIMEOUT attr is RC-only —
    /// the TB provider rejects it on UC with EINVAL (measured on the
    /// amo⇄bmo bring-up, 2026-10-29) — so a UC QP moves with
    /// STATE|SQ_PSN alone. An RC QP tries WITH TIMEOUT first and falls
    /// back to the minimal mask when the provider refuses it (the T3.2
    /// write probe measured the provider's RC posture; the first error
    /// is preserved when the fallback also fails).
    pub fn to_rts(&self, sq_psn: u32) -> Result<(), String> {
        let mut a: IbvQpAttr = unsafe { std::mem::zeroed() };
        a.qp_state = qp_state::RTS;
        a.sq_psn = sq_psn & 0x00ff_ffff;
        a.timeout = 14; // ~67 ms responder timeout (log scale, 4.096 µs · 2^n)
        if self.qp_ty == qp_type::UC {
            return self.modify(&a, attr_mask::STATE | attr_mask::SQ_PSN);
        }
        match self.modify(
            &a,
            attr_mask::STATE | attr_mask::SQ_PSN | attr_mask::TIMEOUT,
        ) {
            Ok(()) => Ok(()),
            Err(first) => self
                .modify(&a, attr_mask::STATE | attr_mask::SQ_PSN)
                .map_err(|_| first),
        }
    }

    /// Post one SEND from a registered slice. `inline` requires
    /// `len <= granted max_inline_data` (caller checks); with inline the
    /// buffer is consumed by the call and reusable immediately.
    pub fn post_send(&self, sge: &IbvSge, wr_id: u64, inline: bool) -> Result<(), String> {
        let mut wr: IbvSendWr = unsafe { std::mem::zeroed() };
        let mut sge = *sge;
        wr.wr_id = wr_id;
        wr.next = std::ptr::null_mut();
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        wr.opcode = ffi::wr_opcode::SEND;
        wr.send_flags = send_flags::SIGNALED | if inline { send_flags::INLINE } else { 0 };
        let mut bad: *mut IbvSendWr = std::ptr::null_mut();
        let rc = unsafe { (self.ctx.raw_ctx_ops().post_send)(self.ptr, &mut wr, &mut bad) };
        if rc != 0 {
            return Err(format!("ibv_post_send rc={rc}"));
        }
        Ok(())
    }

    /// Post one RDMA WRITE from a registered local slice to the remote
    /// range (`remote_addr`, `rkey`). Legal on RC and UC; the completion
    /// carries `wc_opcode::RDMA_WRITE`. `inline` requires
    /// `len <= granted max_inline_data`.
    pub fn post_rdma_write(
        &self,
        sge: &IbvSge,
        remote_addr: u64,
        rkey: u32,
        wr_id: u64,
        inline: bool,
    ) -> Result<(), String> {
        let mut wr: IbvSendWr = unsafe { std::mem::zeroed() };
        let mut sge = *sge;
        wr.wr_id = wr_id;
        wr.next = std::ptr::null_mut();
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        wr.opcode = ffi::wr_opcode::RDMA_WRITE;
        wr.send_flags = send_flags::SIGNALED | if inline { send_flags::INLINE } else { 0 };
        wr.wr.rdma = ffi::IbvRdmaPart { remote_addr, rkey };
        let mut bad: *mut IbvSendWr = std::ptr::null_mut();
        let rc = unsafe { (self.ctx.raw_ctx_ops().post_send)(self.ptr, &mut wr, &mut bad) };
        if rc != 0 {
            return Err(format!("ibv_post_send(RDMA_WRITE) rc={rc}"));
        }
        Ok(())
    }

    /// Post one RDMA READ pulling the remote range (`remote_addr`,
    /// `rkey`) into a LOCAL_WRITE-registered slice. RC-only (UC has no
    /// READ responder); the completion carries `wc_opcode::RDMA_READ`.
    pub fn post_rdma_read(
        &self,
        sge: &IbvSge,
        remote_addr: u64,
        rkey: u32,
        wr_id: u64,
    ) -> Result<(), String> {
        let mut wr: IbvSendWr = unsafe { std::mem::zeroed() };
        let mut sge = *sge;
        wr.wr_id = wr_id;
        wr.next = std::ptr::null_mut();
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        wr.opcode = ffi::wr_opcode::RDMA_READ;
        wr.send_flags = send_flags::SIGNALED;
        wr.wr.rdma = ffi::IbvRdmaPart { remote_addr, rkey };
        let mut bad: *mut IbvSendWr = std::ptr::null_mut();
        let rc = unsafe { (self.ctx.raw_ctx_ops().post_send)(self.ptr, &mut wr, &mut bad) };
        if rc != 0 {
            return Err(format!("ibv_post_send(RDMA_READ) rc={rc}"));
        }
        Ok(())
    }

    /// Post one RECEIVE into a registered slice.
    pub fn post_recv(&self, sge: &IbvSge, wr_id: u64) -> Result<(), String> {
        let mut wr: IbvRecvWr = unsafe { std::mem::zeroed() };
        let mut sge = *sge;
        wr.wr_id = wr_id;
        wr.next = std::ptr::null_mut();
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        let mut bad: *mut IbvRecvWr = std::ptr::null_mut();
        let rc = unsafe { (self.ctx.raw_ctx_ops().post_recv)(self.ptr, &mut wr, &mut bad) };
        if rc != 0 {
            return Err(format!("ibv_post_recv rc={rc}"));
        }
        Ok(())
    }
}

impl Drop for Qp {
    fn drop(&mut self) {
        unsafe {
            (self.ctx.rdma.syms.destroy_qp)(self.ptr);
        }
    }
}

impl DevCtx {
    pub(crate) fn raw_ctx_ops(&self) -> &ffi::IbvContextOps {
        unsafe { &(*self.ptr).ops }
    }
}

// ---------------------------------------------------------------------------
// Endpoint — the bootstrap exchange payload (28 bytes on the wire)
// ---------------------------------------------------------------------------

/// One side's QP coordinates: GID (the sender's), QPN, PSN, and the
/// clamped path-MTU enum. 28 bytes little-endian on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Endpoint {
    pub gid: [u8; 16],
    pub qpn: u32,
    pub psn: u32,
    /// `enum ibv_mtu` value (1..=5).
    pub mtu: u32,
}

pub const ENDPOINT_WIRE: usize = 16 + 4 + 4 + 4;

impl Endpoint {
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.gid);
        out.extend_from_slice(&self.qpn.to_le_bytes());
        out.extend_from_slice(&self.psn.to_le_bytes());
        out.extend_from_slice(&self.mtu.to_le_bytes());
    }

    pub fn decode(b: &[u8]) -> Result<Self, String> {
        if b.len() != ENDPOINT_WIRE {
            return Err(format!(
                "endpoint payload {} bytes, expected {ENDPOINT_WIRE}",
                b.len()
            ));
        }
        Ok(Self {
            gid: b[..16].try_into().expect("16 bytes"),
            qpn: u32::from_le_bytes(b[16..20].try_into().expect("4 bytes")),
            psn: u32::from_le_bytes(b[20..24].try_into().expect("4 bytes")),
            mtu: u32::from_le_bytes(b[24..28].try_into().expect("4 bytes")),
        })
    }

    pub fn mtu_name(&self) -> &'static str {
        match self.mtu {
            mtu::MTU_256 => "256",
            mtu::MTU_512 => "512",
            mtu::MTU_1024 => "1024",
            mtu::MTU_2048 => "2048",
            mtu::MTU_4096 => "4096",
            _ => "?",
        }
    }
}

/// A 24-bit random PSN from the OS entropy pool.
pub fn random_psn() -> Result<u32, String> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom").map_err(|e| format!("open urandom: {e}"))?;
    let mut b = [0u8; 4];
    f.read_exact(&mut b)
        .map_err(|e| format!("read urandom: {e}"))?;
    Ok((u32::from_le_bytes(b)) & 0x00ff_ffff)
}

/// The MAC domain of the endpoint bootstrap (distinct from every frame
/// kind in `transport` — this is the side-channel vocabulary).
const ENDPOINT_MAC_DOMAIN: &[u8] = b"riir-infer-dist/rdma-endpoint-v1";

/// Exchange endpoints over the TCP side channel, BLAKE3-keyed MAC'd both
/// ways with the shared pipeline key: `nonce[32] | endpoint[28] | mac[32]`
/// where mac = keyed_hash(key, domain | own_nonce | endpoint). A bad MAC
/// or a wrong length refuses loud — no unauthenticated endpoint is ever
/// handed to `to_rtr`.
pub fn exchange_endpoint(
    stream: &mut TcpStream,
    key: &crate::transport::PeerKey,
    mine: &Endpoint,
) -> std::io::Result<Endpoint> {
    use std::io::{Read, Write};

    let mut nonce = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut nonce)?;

    let mut payload = Vec::with_capacity(ENDPOINT_WIRE + 64 + 32);
    payload.extend_from_slice(&nonce);
    mine.encode(&mut payload);
    let mac = {
        let mut h = blake3::Hasher::new_keyed(&key.0);
        h.update(ENDPOINT_MAC_DOMAIN);
        h.update(&payload);
        h.finalize()
    };
    payload.extend_from_slice(mac.as_bytes());
    stream.write_all(&payload)?;
    stream.flush()?;

    let mut peer = vec![0u8; 32 + ENDPOINT_WIRE + 32];
    stream.read_exact(&mut peer)?;
    let (p_nonce_ep, p_mac) = peer.split_at(32 + ENDPOINT_WIRE);
    let want = {
        let mut h = blake3::Hasher::new_keyed(&key.0);
        h.update(ENDPOINT_MAC_DOMAIN);
        h.update(p_nonce_ep);
        h.finalize()
    };
    if want.as_bytes() != p_mac {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "endpoint MAC mismatch — refusing unauthenticated RDMA endpoint",
        ));
    }
    Endpoint::decode(&p_nonce_ep[32..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Clamp the local path MTU choice: `min(active, override)` with the
/// override expressed in the same enum, `None` = active.
pub fn clamp_mtu(active: u32, override_mtu: Option<u32>) -> Result<u32, String> {
    let m = match override_mtu {
        Some(o) if o < active => o,
        _ => active,
    };
    if !(mtu::MTU_256..=mtu::MTU_4096).contains(&m) {
        return Err(format!("port active_mtu enum {m} out of range"));
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_round_trip() {
        let ep = Endpoint {
            gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 10, 2],
            qpn: 0x1234_5678,
            psn: 0x00ab_cdef,
            mtu: mtu::MTU_4096,
        };
        let mut wire = Vec::new();
        ep.encode(&mut wire);
        assert_eq!(wire.len(), ENDPOINT_WIRE);
        let back = Endpoint::decode(&wire).expect("decode");
        assert_eq!(back, ep);
    }

    #[test]
    fn endpoint_decode_refuses_wrong_length() {
        assert!(Endpoint::decode(&[0u8; 27]).is_err());
        assert!(Endpoint::decode(&[0u8; 29]).is_err());
        assert!(Endpoint::decode(&[]).is_err());
    }

    #[test]
    fn psn_is_24bit() {
        for _ in 0..64 {
            assert!(random_psn().expect("psn") <= 0x00ff_ffff);
        }
    }

    #[test]
    fn clamp_mtu_law() {
        assert_eq!(clamp_mtu(mtu::MTU_4096, None).unwrap(), mtu::MTU_4096);
        assert_eq!(
            clamp_mtu(mtu::MTU_4096, Some(mtu::MTU_1024)).unwrap(),
            mtu::MTU_1024
        );
        // an override ABOVE active clamps DOWN to active
        assert_eq!(
            clamp_mtu(mtu::MTU_1024, Some(mtu::MTU_4096)).unwrap(),
            mtu::MTU_1024
        );
        assert!(clamp_mtu(9, None).is_err());
    }

    #[test]
    fn qp_state_walk_constants_match_header() {
        assert_eq!(qp_state::INIT, 1);
        assert_eq!(qp_state::RTR, 2);
        assert_eq!(qp_state::RTS, 3);
        assert_eq!(qp_type::UC, 3);
        assert_eq!(qp_type::RC, 2);
        assert_eq!(ffi::wc_opcode::RECV, 128);
        assert_eq!(ffi::wc_opcode::SEND, 0);
        assert_eq!(ffi::wc_opcode::RDMA_WRITE, 1);
        assert_eq!(ffi::wc_opcode::RDMA_READ, 2);
        assert_eq!(ffi::wr_opcode::RDMA_WRITE, 0);
        assert_eq!(ffi::wr_opcode::SEND, 2);
        assert_eq!(ffi::wr_opcode::RDMA_READ, 4);
        assert_eq!(send_flags::INLINE, 8);
        assert_eq!(ffi::access::LOCAL_WRITE, 1);
        assert_eq!(link_layer::THUNDERBOLT, 100);
        assert_eq!(port_state::ACTIVE, 4);
    }
}
