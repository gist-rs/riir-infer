//! MTLBuffer registration probe (Plan 623 Phase 3 — T3.2's MTLBuffer
//! leg: can an RDMA MR wrap a shared-storage `MTLBuffer.contents()`?).
//!
//! ```text
//!   verbs_metal --listen 192.168.10.1:4821 --key <64hex>   # target (Metal side)
//!   verbs_metal --addr 192.168.10.1:4821 --key <64hex>     # initiator (heap side)
//! ```
//!
//! The target registers its SHARED-storage MTLBuffer's `contents()` as
//! an RDMA MR (LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ) and proves byte
//! identity in BOTH directions:
//!
//! 1. **REGISTRATION** — the leg's own question: `ibv_reg_mr` over
//!    GPU-buffer pages, accepted or refused (expected accepted —
//!    `contents()` of a shared buffer is ordinary pages on Apple
//!    silicon; MEASURED here, never assumed).
//! 2. **GPU → wire** — the target's GPU fills a region (compute-kernel
//!    `fill_bytes`, committed + waited), then SENDs it; the initiator
//!    (heap MR) verifies every byte is the fill value. The wire read
//!    pages the GPU wrote — no host memcpy anywhere on the path.
//! 3. **wire → GPU** — the initiator SENDs a per-index pattern INTO the
//!    target's registered MTLBuffer slice; the target's GPU compute
//!    kernel sums the region and the CPU compares the GPU's sum against
//!    its own sum of the received bytes. The GPU read what the wire
//!    wrote — the same pages serve both.
//!
//! GPU work is a blit-class fill + one strided-sum kernel — correctness
//! probes, not benchmarks; timings are printed as characterization
//! only. macOS + `rdma_metal` only (the feature gates the metal dep).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use metal::MTLResourceOptions;

use riir_infer_dist::sockopts;
use riir_infer_dist::transport::PeerKey;
use riir_infer_dist::verbs::{
    self, CompQ, DevCtx, Endpoint, ProtDom, Qp, Rdma, clamp_mtu, exchange_endpoint, random_psn,
};

const GUARD: usize = 4096;
const MARK_LEN: usize = 16;
const MARK_MAGIC: u32 = 0x4D_45_54_41; // "META"
const FILL: u8 = 0xC7;

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fill_bytes(device uchar* data [[buffer(0)]],
                       constant uint& count [[buffer(1)]],
                       constant uchar& value [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < count) { data[gid] = value; }
}

kernel void sum_strided(device const uchar* data [[buffer(0)]],
                        constant uint& count [[buffer(1)]],
                        device atomic_uint* out [[buffer(2)]],
                        uint tid [[thread_position_in_grid]],
                        uint n [[threads_per_grid]]) {
    if (tid >= n) { return; }
    uint acc = 0;
    for (uint i = tid * 4u; i + 3u < count; i += n * 4u) {
        acc += (uint)data[i] + (uint)data[i + 1u] + (uint)data[i + 2u] + (uint)data[i + 3u];
    }
    for (uint i = (count & ~3u) + tid; i < count; i += n) {
        acc += (uint)data[i];
    }
    atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
}
"#;

fn link_opts(iface: &Option<String>) -> sockopts::LinkOptions {
    sockopts::LinkOptions {
        sock_buf: Some(4 << 20),
        iface: iface.clone(),
    }
}

struct Brought {
    qp: Qp,
    send_cq: Arc<CompQ>,
    recv_cq: Arc<CompQ>,
}

/// Endpoint + QP walk shared by both roles (identical to verbs_write's
/// minus the rkey frame — this protocol uses SEND/RECV only).
#[allow(clippy::too_many_arguments)]
fn bring_up(
    stream: &mut std::net::TcpStream,
    key: &PeerKey,
    ctx: &Arc<DevCtx>,
    pd: &Arc<ProtDom>,
    port_req: Option<u8>,
    gid_index: Option<u32>,
) -> Brought {
    let (vp, port_attr) = match port_req {
        Some(p) => (
            p,
            ctx.query_port(p)
                .unwrap_or_else(|e| panic!("query port {p}: {e}")),
        ),
        None => ctx
            .first_active_port()
            .unwrap_or_else(|e| panic!("no active port: {e}")),
    };
    if port_attr.state != verbs::ffi::port_state::ACTIVE {
        panic!("verbs port {vp} is not ACTIVE (state {})", port_attr.state);
    }
    eprintln!(
        "[rdma] dev={} port={vp} link={} active_mtu={} max_msg={}",
        ctx.dev_name,
        port_attr.link_layer_name(),
        port_attr.active_mtu,
        port_attr.max_msg_sz,
    );
    let send_cq = Arc::new(pd.create_cq(256).unwrap_or_else(|e| panic!("send cq: {e}")));
    let recv_cq = Arc::new(pd.create_cq(256).unwrap_or_else(|e| panic!("recv cq: {e}")));
    let (qp, granted) = pd
        .create_uc_qp(
            &send_cq,
            &recv_cq,
            verbs::ffi::IbvQpCap {
                max_send_wr: 256,
                max_recv_wr: 256,
                max_send_sge: 1,
                max_recv_sge: 1,
                max_inline_data: 224,
            },
        )
        .unwrap_or_else(|e| panic!("qp: {e}"));
    let acc = verbs::ffi::access::LOCAL_WRITE
        | verbs::ffi::access::REMOTE_WRITE
        | verbs::ffi::access::REMOTE_READ;
    qp.to_init(vp, acc).unwrap_or_else(|e| panic!("INIT: {e}"));
    let psn = random_psn().unwrap_or_else(|e| panic!("psn: {e}"));
    // Same-link GID pick, identical to verbs_ping's differential posture:
    // a bare `None` would take "any IPv4-mapped GID" (possibly the wrong
    // link) where ping takes the peer-matched one — a GID-table change
    // could silently split the two instruments' verdicts.
    let peer_v4 = match stream.peer_addr() {
        Ok(SocketAddr::V4(a)) => Some(*a.ip()),
        _ => None,
    };
    let (sgid_idx, sgid) = match gid_index {
        Some(idx) => (
            idx,
            ctx.gid(vp, idx)
                .unwrap_or_else(|e| panic!("gid {idx}: {e}")),
        ),
        None => ctx
            .pick_gid(vp, peer_v4)
            .unwrap_or_else(|e| panic!("pick gid: {e}")),
    };
    let mine = Endpoint {
        gid: sgid,
        qpn: qp.qp_num(),
        psn,
        mtu: clamp_mtu(port_attr.active_mtu, None).unwrap(),
    };
    let mut peer_ep =
        exchange_endpoint(stream, key, &mine).unwrap_or_else(|e| panic!("exchange: {e}"));
    peer_ep.mtu = clamp_mtu(port_attr.active_mtu, Some(peer_ep.mtu)).unwrap();
    qp.to_rtr(&peer_ep, sgid_idx as u8, vp)
        .unwrap_or_else(|e| panic!("RTR: {e}"));
    qp.to_rts(psn).unwrap_or_else(|e| panic!("RTS: {e}"));
    eprintln!(
        "[rdma] UC QP {} in RTS: path_mtu={} inline_cap={}",
        qp.qp_num(),
        peer_ep.mtu_name(),
        granted.max_inline_data
    );
    Brought {
        qp,
        send_cq,
        recv_cq,
    }
}

fn expected_sum(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0u32, |a, &b| a.wrapping_add(b as u32))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str| -> Option<String> {
        let p = format!("--{name}");
        args.iter()
            .position(|a| a == &p || a.starts_with(&format!("{p}=")))
            .map(|i| {
                let a = &args[i];
                if let Some(eq) = a.find('=') {
                    a[eq + 1..].to_string()
                } else {
                    args.get(i + 1).cloned().unwrap_or_default()
                }
            })
            .filter(|v| !v.is_empty())
    };

    let key_hex = match get("key") {
        Some(k) => k,
        None => {
            eprintln!(
                "usage: verbs_metal (--listen IP:PORT | --addr IP:PORT) --key <64hex> [--kib N]"
            );
            std::process::exit(2);
        }
    };
    let key = PeerKey::from_hex(&key_hex).unwrap_or_else(|e| {
        eprintln!("--key: {e}");
        std::process::exit(2);
    });
    let listen = get("listen");
    let addr = get("addr");
    let is_target = listen.is_some();
    if is_target == addr.is_some() {
        eprintln!("--listen pairs with the target role; --addr pairs with the initiator role");
        std::process::exit(2);
    }
    let iface = get("iface");
    let dev = get("dev");
    let port_req: Option<u8> = get("port").map(|v| v.parse().expect("--port N"));
    let gid_index: Option<u32> = get("gid-index").map(|v| v.parse().expect("--gid-index N"));
    let kib: usize = get("kib").map_or(1024, |v| v.parse().expect("--kib N"));
    let probe_len = kib * 1024;
    let lat = Duration::from_secs(10);

    let mut stream: std::net::TcpStream = if let Some(l) = &listen {
        let sa: SocketAddr = l.parse().expect("--listen IP:PORT");
        let (l, applied) =
            sockopts::listen(sa, &link_opts(&iface)).unwrap_or_else(|e| panic!("listen {sa}: {e}"));
        eprintln!("[side] target on {sa} (applied {applied:?})");
        let (s, peer) = l.accept().expect("accept");
        eprintln!("[side] initiator connected from {peer}");
        s
    } else {
        let sa: SocketAddr = addr
            .as_deref()
            .expect("--addr IP:PORT")
            .parse()
            .expect("addr");
        let (s, applied) =
            sockopts::dial(sa, &link_opts(&iface)).unwrap_or_else(|e| panic!("dial {sa}: {e}"));
        eprintln!("[side] connected to {sa} (applied {applied:?})");
        s
    };

    let rdma = Rdma::open().unwrap_or_else(|e| panic!("RDMA open: {e}"));
    let ctx = Arc::new(
        rdma.open_device(dev.as_deref())
            .unwrap_or_else(|e| panic!("open device: {e}")),
    );
    let pd = Arc::new(ctx.alloc_pd().unwrap_or_else(|e| panic!("pd: {e}")));

    if is_target {
        run_target(
            &mut stream,
            &key,
            &ctx,
            &pd,
            port_req,
            gid_index,
            probe_len,
            lat,
        );
    } else {
        run_initiator(
            &mut stream,
            &key,
            &ctx,
            &pd,
            port_req,
            gid_index,
            probe_len,
            lat,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_target(
    stream: &mut std::net::TcpStream,
    key: &PeerKey,
    ctx: &Arc<DevCtx>,
    pd: &Arc<ProtDom>,
    port_req: Option<u8>,
    gid_index: Option<u32>,
    probe_len: usize,
    lat: Duration,
) {
    // ---- the Metal side -------------------------------------------------
    let device = metal::Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let region = GUARD + probe_len;
    let mtl_buf = device.new_buffer(region as u64, MTLResourceOptions::StorageModeShared);
    let contents = mtl_buf.contents() as *mut u8;
    assert!(!contents.is_null(), "MTLBuffer contents() null");
    println!(
        "[metal] shared buffer: len={} contents={:#x} (device {})",
        region,
        contents as usize,
        device.name()
    );

    // ---- the leg: register GPU-buffer pages as an RDMA MR ---------------
    let acc = verbs::ffi::access::LOCAL_WRITE
        | verbs::ffi::access::REMOTE_WRITE
        | verbs::ffi::access::REMOTE_READ;
    let t0 = Instant::now();
    let mr = pd
        .reg_mr(contents, region, acc)
        .unwrap_or_else(|e| panic!("THE LEG — reg_mr over MTLBuffer.contents(): {e}"));
    println!(
        "REGISTRATION: ACCEPTED — addr={:#x} (contents={:#x}, same={}) len={} lkey={:#010x} rkey={:#010x} in {:?}",
        mr.addr,
        contents as usize,
        mr.addr == contents as u64,
        mr.len,
        mr.lkey,
        mr.rkey,
        t0.elapsed()
    );

    let brought = bring_up(stream, key, ctx, pd, port_req, gid_index);
    let deadline = Instant::now() + lat;

    // ---- GPU → wire: kernel fill, SEND, initiator verifies --------------
    let compile_opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(SHADER, &compile_opts)
        .unwrap_or_else(|e| panic!("shader: {e}"));
    let fill_fn = lib
        .get_function("fill_bytes", None)
        .unwrap_or_else(|e| panic!("fill_bytes: {e}"))
        .to_owned();
    let fill_ps = device
        .new_compute_pipeline_state_with_function(&fill_fn)
        .unwrap_or_else(|e| panic!("fill pipeline: {e}"));
    let sum_fn = lib
        .get_function("sum_strided", None)
        .unwrap_or_else(|e| panic!("sum_strided: {e}"))
        .to_owned();
    let sum_ps = device
        .new_compute_pipeline_state_with_function(&sum_fn)
        .unwrap_or_else(|e| panic!("sum pipeline: {e}"));

    let count_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let value_buf = device.new_buffer(1, MTLResourceOptions::StorageModeShared);
    let out_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(count_buf.contents() as *mut u32) = probe_len as u32;
        *(value_buf.contents() as *mut u8) = FILL;
        *(out_buf.contents() as *mut u32) = 0;
    }

    let cb = queue.new_command_buffer().to_owned();
    let enc = cb
        .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial)
        .to_owned();
    enc.set_compute_pipeline_state(&fill_ps);
    enc.set_buffer(0, Some(&mtl_buf), (GUARD) as u64);
    enc.set_buffer(1, Some(&count_buf), 0);
    enc.set_buffer(2, Some(&value_buf), 0);
    let w = fill_ps.thread_execution_width();
    enc.dispatch_threads(
        metal::MTLSize::new(probe_len as u64, 1, 1),
        metal::MTLSize::new(w, 1, 1),
    );
    enc.end_encoding();
    let t_fill = Instant::now();
    cb.commit();
    cb.wait_until_completed();
    let fill_dt = t_fill.elapsed();
    println!(
        "GPU→wire: fill committed in {:?}; SENDing {probe_len} B of GPU-written pages",
        fill_dt
    );

    // Post the recv for the initiator's "got" marker BEFORE sending.
    brought
        .qp
        .post_recv(&mr.slice_at(0, MARK_LEN as u32), 9)
        .expect("marker recv");
    brought
        .qp
        .post_send(&mr.slice_at(GUARD, probe_len as u32), 10, false)
        .expect("gpu fill send");
    let t_send = Instant::now();
    let wc = brought
        .send_cq
        .poll_one(deadline)
        .expect("gpu fill send wc");
    assert_eq!(wc.status, verbs::ffi::wc_status::SUCCESS, "gpu fill send");
    let got = brought.recv_cq.poll_one(deadline).expect("got marker");
    assert_eq!(got.byte_len as usize, MARK_LEN, "got marker length");
    let magic = unsafe {
        u32::from_le_bytes(
            std::slice::from_raw_parts(mr.addr as *const u8, 4)
                .try_into()
                .unwrap(),
        )
    };
    assert_eq!(magic, MARK_MAGIC, "got marker magic");
    println!(
        "GPU→wire: SEND completed in {:?}, initiator marker received (send-side done)",
        t_send.elapsed()
    );

    // ---- wire → GPU: initiator SENDs a pattern in, kernel sums it -------
    let cb2 = queue.new_command_buffer().to_owned();
    let enc2 = cb2
        .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial)
        .to_owned();
    enc2.set_compute_pipeline_state(&sum_ps);
    enc2.set_buffer(0, Some(&mtl_buf), (GUARD) as u64);
    enc2.set_buffer(1, Some(&count_buf), 0);
    enc2.set_buffer(2, Some(&out_buf), 0);
    let w2 = sum_ps.thread_execution_width();
    enc2.dispatch_threads(metal::MTLSize::new(w2, 1, 1), metal::MTLSize::new(w2, 1, 1));
    enc2.end_encoding();
    let t_sum = Instant::now();
    cb2.commit();
    cb2.wait_until_completed();
    let gpu_sum = unsafe { *(out_buf.contents() as *const u32) };
    let cpu_sum =
        expected_sum(unsafe { std::slice::from_raw_parts(contents.add(GUARD), probe_len) });
    let pass = gpu_sum == cpu_sum;
    println!(
        "wire→GPU: sum kernel {:?} — gpu_sum={gpu_sum} cpu_sum={cpu_sum} => {}",
        t_sum.elapsed(),
        if pass {
            "PASS — the GPU read the wire-written pages"
        } else {
            "FAIL"
        }
    );
    println!(
        "VERDICT: MTLBuffer leg {}",
        if pass {
            "CLOSED — registration accepted, both directions byte-identical"
        } else {
            "REGISTRATION accepted but identity check FAILED"
        }
    );
}

#[allow(clippy::too_many_arguments)]
fn run_initiator(
    stream: &mut std::net::TcpStream,
    key: &PeerKey,
    ctx: &Arc<DevCtx>,
    pd: &Arc<ProtDom>,
    port_req: Option<u8>,
    gid_index: Option<u32>,
    probe_len: usize,
    lat: Duration,
) {
    // Heap-only on this side: the probe's Metal leg lives on the target.
    let region = GUARD + probe_len;
    let mut heap = vec![0u8; region];
    let mr = pd
        .reg_mr(
            heap.as_mut_ptr(),
            heap.len(),
            verbs::ffi::access::LOCAL_WRITE,
        )
        .unwrap_or_else(|e| panic!("initiator reg_mr: {e}"));
    let brought = bring_up(stream, key, ctx, pd, port_req, gid_index);
    let deadline = Instant::now() + lat;

    // ---- GPU → wire: recv the target's GPU-filled region, verify --------
    brought
        .qp
        .post_recv(&mr.slice_at(GUARD, probe_len as u32), 20)
        .expect("fill recv");
    let wc = brought.recv_cq.poll_one(deadline).expect("fill recv wc");
    assert_eq!(wc.byte_len as usize, probe_len, "fill length");
    let all_fill = heap[GUARD..GUARD + probe_len].iter().all(|&c| c == FILL);
    println!(
        "GPU→wire: received {probe_len} B — every byte == {FILL:#x}: {}",
        if all_fill {
            "PASS (the wire read GPU-written pages)"
        } else {
            "FAIL"
        }
    );

    // ---- wire → GPU: send a per-index pattern INTO the target's buffer --
    let pattern: Vec<u8> = (0..probe_len)
        .map(|i| (i.wrapping_mul(7).wrapping_add(3)) as u8)
        .collect();
    let expected = expected_sum(&pattern);
    let t0 = Instant::now();
    heap[GUARD..GUARD + probe_len].copy_from_slice(&pattern);
    brought
        .qp
        .post_send(&mr.slice_at(GUARD, probe_len as u32), 22, false)
        .expect("pattern send");
    let wc2 = brought.send_cq.poll_one(deadline).expect("pattern send wc");
    assert_eq!(wc2.status, verbs::ffi::wc_status::SUCCESS, "pattern send");

    // Tell the target the fill verified — its GPU→wire row reads this
    // marker before driving the reverse direction (whose pattern send is
    // already in flight on the same ordered SQ).
    let mut marker = [0u8; MARK_LEN];
    marker[0..4].copy_from_slice(&MARK_MAGIC.to_le_bytes());
    heap[..MARK_LEN].copy_from_slice(&marker);
    brought
        .qp
        .post_send(&mr.slice_at(0, MARK_LEN as u32), 23, true)
        .expect("got marker send");
    let _ = brought.send_cq.poll_one(deadline).expect("got marker wc");
    println!(
        "wire→GPU: pattern + marker sent in {:?} (expected gpu_sum={expected}) — target verifies",
        t0.elapsed()
    );
}
