//! RDMA verbs probe (Plan 623 Phase 3 — T3.1 bring-up, T3.2/T3.3 rows).
//!
//! Two roles over the TB bridge, one TCP side channel for the MAC'd
//! endpoint bootstrap, then a UC QP pair:
//!
//! ```text
//!   verbs_ping --listen 192.168.10.2:4810 --key <64hex>            # host (echoes)
//!   verbs_ping --addr 192.168.10.2:4810 --key <64hex>              # peer (drives)
//! ```
//!
//! The peer drives the measurement ladder; the host is a pure echo
//! engine. Protocol (all payload lengths on ONE recv-at-a-time cadence
//! unless stated — UC has no RNR retry, so every send must meet a
//! posted receive):
//!
//! 1. **rtt** — peer sends `rtt_msg` bytes, host echoes back into the
//!    same registered buffer; `--rtt-iters` round trips, lowest 10%
//!    discarded as warmup, p50/p99/mean printed (T3.3 small-message).
//!    Content verified every 256 iters (the echo must return the bytes).
//! 2. **bulk** — peer sends a 16-byte header (`stripe u32 | stripes u32
//!    | total u32`); the host pre-posts its receive window and answers
//!    `ready` (8 B); the peer then pushes `stripes × stripe` bytes from
//!    its registered bulk area through a bounded in-flight window; the
//!    host drains and answers `done` (8 B). GB/s printed (T3.3 striped).
//!
//! The boot line carries the T3.2 registration facts (heap registration
//! accepted, lkey/rkey granted). `--mtu` / `--gid-index` pin the path
//! for the MTU/GID probe legs. CPU-side only: no Metal, no GPU
//! contention — safe next to a running capture on the same box.

use std::net::{IpAddr, SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use riir_infer_dist::sockopts;
use riir_infer_dist::transport::PeerKey;
use riir_infer_dist::verbs::{
    self, clamp_mtu, exchange_endpoint, random_psn, CompQ, Endpoint, MemReg, Qp, Rdma,
};

/// The bulk header magic length — distinguishes it from an RTT echo on
/// the length-only host dispatch.
const HDR_LEN: usize = 16;
/// The `ready`/`done` marker length.
const MARK_LEN: usize = 8;
/// Guard band before the bulk stripes inside the bulk area (header room).
const BULK_HDR_ROOM: usize = 64;

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

    let _usage = || {
        eprintln!("usage: verbs_ping (--listen IP:PORT | --addr IP:PORT) --key <64hex> [options]");
        eprintln!("  --iface NAME        bind the side channel to this interface");
        eprintln!("  --dev NAME          open this RDMA device (default: first)");
        eprintln!("  --port N            verbs port number (default: first ACTIVE)");
        eprintln!("  --gid-index N       pin the source GID (default: auto by peer v4)");
        eprintln!("  --mtu N             path MTU 256..4096 (default: active)");
        eprintln!("  --inline N          request N inline bytes (default 224)");
        eprintln!("  --rtt-msg BYTES     RTT probe message size (default 64; must not be {HDR_LEN})");
        eprintln!("  --rtt-iters N       RTT measured iterations (default 2000)");
        eprintln!("  --bw-size MIB       one-way bulk size in MiB (default 64)");
        eprintln!("  --stripe BYTES      bulk stripe size (default 262144)");
        eprintln!("  --window N          in-flight stripes (default 32)");
        eprintln!("  --lat-ms N          completion timeout ms (default 5000)");
    };

    let key_hex = match get("key") {
        Some(k) => k,
        None => {
            _usage();
            std::process::exit(2);
        }
    };
    let key = PeerKey::from_hex(&key_hex).unwrap_or_else(|e| {
        eprintln!("--key: {e}");
        std::process::exit(2);
    });

    let listen = get("listen");
    let addr = get("addr");
    let role_binding = get("role");
    let role: &str = role_binding
        .as_deref()
        .unwrap_or(if listen.is_some() { "host" } else { "peer" });
    if (listen.is_some()) != (role == "host") || (addr.is_some()) != (role == "peer") {
        eprintln!("--listen pairs with role host; --addr pairs with role peer");
        _usage();
        std::process::exit(2);
    }
    let iface = get("iface");
    let dev_name = get("dev");
    let verbs_port: Option<u8> = get("port").map(|v| v.parse().expect("--port 1|2"));
    let gid_index: Option<u32> = get("gid-index").map(|v| v.parse().expect("--gid-index N"));
    let mtu_override: Option<u32> = get("mtu").map(|v| match v.as_str() {
        "256" => verbs::ffi::mtu::MTU_256,
        "512" => verbs::ffi::mtu::MTU_512,
        "1024" => verbs::ffi::mtu::MTU_1024,
        "2048" => verbs::ffi::mtu::MTU_2048,
        "4096" => verbs::ffi::mtu::MTU_4096,
        other => {
            eprintln!("--mtu {other} invalid");
            std::process::exit(2);
        }
    });
    let inline_req: u32 = get("inline").map_or(224, |v| v.parse().expect("--inline N"));
    let rtt_msg: usize = get("rtt-msg").map_or(64, |v| v.parse().expect("--rtt-msg BYTES"));
    let rtt_iters: usize = get("rtt-iters").map_or(2000, |v| v.parse().expect("--rtt-iters N"));
    let bw_mib: usize = get("bw-size").map_or(64, |v| v.parse().expect("--bw-size MIB"));
    let stripe: usize = get("stripe").map_or(256 * 1024, |v| v.parse().expect("--stripe BYTES"));
    let window: usize = get("window").map_or(32, |v| v.parse().expect("--window N"));
    let lat = Duration::from_millis(get("lat-ms").map_or(5_000, |v| v.parse::<u64>().expect("--lat-ms N")));

    // ---- side channel --------------------------------------------------
    let mut stream: TcpStream = if let Some(l) = &listen {
        let sa: SocketAddr = l.parse().expect("--listen IP:PORT");
        let (listener, applied) =
            sockopts::listen(sa, &link_opts(&iface)).unwrap_or_else(|e| panic!("listen {sa}: {e}"));
        eprintln!("[side] listening on {sa} (applied {applied:?})");
        let (s, peer) = listener.accept().expect("accept");
        eprintln!("[side] peer connected from {peer}");
        s
    } else {
        let sa: SocketAddr = addr.as_deref().expect("--addr IP:PORT").parse().expect("addr");
        let (s, applied) =
            sockopts::dial(sa, &link_opts(&iface)).unwrap_or_else(|e| panic!("dial {sa}: {e}"));
        eprintln!("[side] connected to {sa} (applied {applied:?})");
        s
    };
    let peer_sock = stream.peer_addr().expect("peer addr");

    // ---- verbs bring-up ------------------------------------------------
    let rdma = Rdma::open().unwrap_or_else(|e| panic!("RDMA open: {e}"));
    eprintln!("[rdma] devices = {:?}", rdma.device_names());
    let ctx = Arc::new(
        rdma.open_device(dev_name.as_deref())
            .unwrap_or_else(|e| panic!("open device: {e}")),
    );
    let (vp, port_attr) = match verbs_port {
        Some(p) => (p, ctx.query_port(p).unwrap_or_else(|e| panic!("query port {p}: {e}"))),
        None => ctx
            .first_active_port()
            .unwrap_or_else(|e| panic!("no active port: {e}")),
    };
    if port_attr.state != verbs::ffi::port_state::ACTIVE {
        panic!("verbs port {vp} is not ACTIVE (state {})", port_attr.state);
    }
    let caps = ctx.caps().unwrap_or_else(|e| panic!("caps: {e}"));
    eprintln!(
        "[rdma] dev={} port={vp} link={} active_mtu={} max_msg={} gid_tbl={} max_qp_wr={} max_sge={} max_cqe={}",
        ctx.dev_name,
        port_attr.link_layer_name(),
        port_attr.active_mtu,
        port_attr.max_msg_sz,
        port_attr.gid_tbl_len,
        caps.max_qp_wr,
        caps.max_sge,
        caps.max_cqe,
    );

    let path_mtu = clamp_mtu(port_attr.active_mtu, mtu_override).unwrap_or_else(|e| panic!("mtu: {e}"));

    // One registration covering the RTT area + the bulk area (T3.2 row).
    let bulk_len = bw_mib * (1 << 20);
    let rtt_area = rtt_msg.max(4096);
    let mut bufs = vec![0u8; rtt_area + bulk_len];

    let pd = Arc::new(ctx.alloc_pd().unwrap_or_else(|e| panic!("pd: {e}")));
    let acc = verbs::ffi::access::LOCAL_WRITE
        | verbs::ffi::access::REMOTE_WRITE
        | verbs::ffi::access::REMOTE_READ;
    let mr = pd
        .reg_mr(bufs.as_mut_ptr(), bufs.len(), acc)
        .unwrap_or_else(|e| panic!("reg_mr: {e}"));
    eprintln!(
        "[rdma] T3.2 heap registration OK: addr={:#x} len={} lkey={:#x} rkey={:#x}",
        mr.addr,
        mr.len,
        mr.lkey,
        mr.rkey
    );

    let send_cq = Arc::new(pd.create_cq(1024).unwrap_or_else(|e| panic!("send cq: {e}")));
    let recv_cq = Arc::new(pd.create_cq(1024).unwrap_or_else(|e| panic!("recv cq: {e}")));
    let (qp, granted) = pd
        .create_uc_qp(
            &send_cq,
            &recv_cq,
            verbs::ffi::IbvQpCap {
                max_send_wr: 256,
                max_recv_wr: 256,
                max_send_sge: 1,
                max_recv_sge: 1,
                max_inline_data: inline_req,
            },
        )
        .unwrap_or_else(|e| panic!("qp: {e}"));
    eprintln!(
        "[rdma] UC QP {} granted caps: send_wr={} recv_wr={} inline={}",
        qp.qp_num(),
        granted.max_send_wr,
        granted.max_recv_wr,
        granted.max_inline_data
    );

    // Endpoint + bootstrap (both sides run the identical walk).
    let psn = random_psn().unwrap_or_else(|e| panic!("psn: {e}"));
    let peer_v4 = match peer_sock.ip() {
        IpAddr::V4(v4) => Some(v4),
        _ => None,
    };
    let (sgid_idx, sgid) = match gid_index {
        Some(idx) => (
            idx,
            ctx.gid(vp, idx).unwrap_or_else(|e| panic!("gid {idx}: {e}")),
        ),
        None => ctx
            .pick_gid(vp, peer_v4)
            .unwrap_or_else(|e| panic!("pick gid: {e}")),
    };
    eprintln!("[rdma] source gid[{sgid_idx}] = {}", Gid(&sgid));
    let mine = Endpoint {
        gid: sgid,
        qpn: qp.qp_num(),
        psn,
        mtu: path_mtu,
    };

    let mut peer_ep = exchange_endpoint(&mut stream, &key, &mine)
        .unwrap_or_else(|e| panic!("endpoint exchange: {e}"));
    eprintln!(
        "[rdma] peer endpoint qpn={} psn={:#x} mtu={} gid={}",
        peer_ep.qpn,
        peer_ep.psn,
        peer_ep.mtu_name(),
        Gid(&peer_ep.gid)
    );

    qp.to_init(vp, acc).unwrap_or_else(|e| panic!("to INIT: {e}"));
    // RTR at the peer's advertised MTU clamped DOWN by our active MTU.
    peer_ep.mtu = clamp_mtu(port_attr.active_mtu, Some(peer_ep.mtu)).unwrap();
    qp.to_rtr(&peer_ep, sgid_idx as u8, vp)
        .unwrap_or_else(|e| panic!("to RTR: {e}"));
    qp.to_rts(mine.psn).unwrap_or_else(|e| panic!("to RTS: {e}"));
    eprintln!(
        "[rdma] QP in RTS: path_mtu={} inline_cap={}",
        peer_ep.mtu_name(),
        granted.max_inline_data
    );

    // ---- the ladder ----------------------------------------------------
    if role == "host" {
        host_loop(&qp, &send_cq, &recv_cq, &mr, rtt_msg, rtt_area, bulk_len, stripe, window, lat);
        println!("[host] done");
    } else {
        if rtt_msg == HDR_LEN {
            eprintln!("--rtt-msg must not equal {HDR_LEN} (bulk-header dispatch is length-based)");
            std::process::exit(2);
        }
        peer_ladder(
            &qp, &send_cq, &recv_cq, &mr, rtt_msg, rtt_iters, rtt_area, bulk_len, stripe, window, lat,
        );
    }
    eprintln!("[done] QP drained, exiting clean");
}

fn link_opts(iface: &Option<String>) -> sockopts::LinkOptions {
    sockopts::LinkOptions {
        sock_buf: Some(4 << 20),
        iface: iface.clone(),
    }
}

struct Gid<'a>(&'a [u8; 16]);
impl std::fmt::Display for Gid<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = self.0;
        write!(
            f,
            "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13],
            r[14], r[15]
        )
    }
}

/// HOST: RTT echo (recv at 0, echo from 0), then the bulk phase:
/// header (16 B at the bulk-area front) → ready (8 B from 0) → windowed
/// stripe drain → done (8 B from 0).
#[allow(clippy::too_many_arguments)]
fn host_loop(
    qp: &Qp,
    send_cq: &Arc<CompQ>,
    recv_cq: &Arc<CompQ>,
    mr: &MemReg<'_>,
    rtt_msg: usize,
    rtt_area: usize,
    _bulk_len: usize,
    _stripe_hint: usize,
    _window_hint: usize,
    lat: Duration,
) {
    let deadline = || Instant::now() + lat;

    // Phase 1: echo until the bulk header (length dispatch).
    qp.post_recv(&mr.slice_at(0, rtt_msg as u32), 1).expect("host recv");
    let (stripe, stripes, total) = loop {
        let wc = recv_cq.poll_one(deadline()).expect("host recv");
        let n = wc.byte_len as usize;
        if n == HDR_LEN {
            unsafe {
                let p = mr.addr as *const u8;
                let b = std::slice::from_raw_parts(p.add(rtt_area), HDR_LEN);
                let stripe = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
                let stripes = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
                let total = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
                break (stripe, stripes, total);
            }
        }
        // echo straight back from the same registered bytes
        qp.post_send(&mr.slice_at(0, n as u32), 0, n <= 224)
            .expect("host echo send");
        send_cq.poll_one(deadline()).expect("host echo completion");
        qp.post_recv(&mr.slice_at(0, rtt_msg as u32), 1).expect("host recv re-post");
    };
    eprintln!("[host] bulk header: stripe={stripe} stripes={stripes} total={total}");

    // Phase 2: pre-post the receive window over the bulk area, then
    // answer ready.
    let base = rtt_area + BULK_HDR_ROOM;
    assert!(base + stripes * stripe <= mr.len, "host bulk area undersized");
    for i in 0..stripes {
        qp.post_recv(&mr.slice_at(base + i * stripe, stripe as u32), 10)
            .expect("host window post");
    }
    qp.post_send(&mr.slice_at(0, MARK_LEN as u32), 0, true)
        .expect("host ready send");
    send_cq.poll_one(deadline()).expect("host ready completion");

    // Phase 3: drain exactly `stripes` completions, replenishing past
    // the window.
    let mut reaped = 0usize;
    while reaped < stripes {
        let wc = recv_cq.poll_one(deadline()).expect("host bulk recv");
        if wc.status != verbs::ffi::wc_status::SUCCESS {
            panic!("host bulk recv wc status {}", wc.status);
        }
        reaped += 1;
    }
    let got = reaped * stripe;
    if got != total {
        panic!("host bulk total {got} != announced {total}");
    }
    qp.post_send(&mr.slice_at(0, MARK_LEN as u32), 0, true)
        .expect("host done send");
    send_cq.poll_one(deadline()).expect("host done completion");
    eprintln!("[host] bulk drained {got} bytes, done sent");
}

/// PEER: the RTT ladder + the bulk push, printing one row per probe.
#[allow(clippy::too_many_arguments)]
fn peer_ladder(
    qp: &Qp,
    send_cq: &Arc<CompQ>,
    recv_cq: &Arc<CompQ>,
    mr: &MemReg<'_>,
    rtt_msg: usize,
    rtt_iters: usize,
    rtt_area: usize,
    bulk_len: usize,
    stripe: usize,
    window: usize,
    lat: Duration,
) {
    let deadline = || Instant::now() + lat;

    // ---- RTT ----
    let warmup = rtt_iters / 10;
    let mut samples: Vec<Duration> = Vec::with_capacity(rtt_iters);
    qp.post_recv(&mr.slice_at(0, rtt_msg as u32), 1).expect("rtt recv");
    let mut i = 0usize;
    loop {
        if i.is_multiple_of(256) {
            unsafe {
                let p = mr.addr as *mut u8;
                for b in 0..rtt_msg {
                    p.add(b).write((i.wrapping_add(b)) as u8);
                }
            }
        }
        let t0 = Instant::now();
        qp.post_send(&mr.slice_at(0, rtt_msg as u32), 0, rtt_msg <= 224)
            .expect("rtt send");
        let _ = send_cq.poll_one(deadline()).expect("rtt send completion");
        let wc = recv_cq.poll_one(deadline()).expect("rtt recv");
        let dt = t0.elapsed();
        assert_eq!(wc.byte_len as usize, rtt_msg, "echo length");
        if i.is_multiple_of(256) {
            unsafe {
                let p = mr.addr as *const u8;
                for b in 0..rtt_msg {
                    assert_eq!(*p.add(b), (i.wrapping_add(b)) as u8, "echo content at iter {i}");
                }
            }
        }
        if i >= warmup {
            samples.push(dt);
        }
        i += 1;
        if i >= rtt_iters {
            break;
        }
        // re-post the receive for the next echo
        qp.post_recv(&mr.slice_at(0, rtt_msg as u32), 1).expect("rtt recv re-post");
    }
    samples.sort();
    let n = samples.len();
    let p50 = samples[n * 50 / 100];
    let p99 = samples[(n * 99 / 100).min(n - 1)];
    let mean = samples.iter().sum::<Duration>() / n.max(1) as u32;
    println!(
        "rtt msg={rtt_msg}B inline={} n={n} p50={:?} p99={:?} mean={:?}",
        rtt_msg <= 224,
        p50,
        p99,
        mean
    );

    // ---- bulk ----
    let stripes = (bulk_len - BULK_HDR_ROOM) / stripe;
    let total = stripes * stripe;
    // header into the bulk-area front
    unsafe {
        let p = mr.addr as *mut u8;
        let hdr = std::slice::from_raw_parts_mut(p.add(rtt_area), HDR_LEN);
        hdr[0..4].copy_from_slice(&(stripe as u32).to_le_bytes());
        hdr[4..8].copy_from_slice(&(stripes as u32).to_le_bytes());
        hdr[8..12].copy_from_slice(&(total as u32).to_le_bytes());
    }
    // receive for `ready` BEFORE sending the header (UC: every send must
    // meet a posted receive).
    qp.post_recv(&mr.slice_at(0, MARK_LEN as u32), 2).expect("ready recv");
    qp.post_send(&mr.slice_at(rtt_area, HDR_LEN as u32), 3, true)
        .expect("header send");
    send_cq.poll_one(deadline()).expect("header completion");
    let wc = recv_cq.poll_one(deadline()).expect("ready recv");
    assert_eq!(wc.byte_len as usize, MARK_LEN, "ready marker length");

    // receive for `done`, then push.
    qp.post_recv(&mr.slice_at(0, MARK_LEN as u32), 4).expect("done recv");
    let base = rtt_area + BULK_HDR_ROOM;
    let mut in_flight = 0usize;
    let mut sent = 0usize;
    let mut done = 0usize;
    let t0 = Instant::now();
    while done < stripes {
        while in_flight < window && sent < stripes {
            qp.post_send(&mr.slice_at(base + sent * stripe, stripe as u32), 5, false)
                .expect("bw send");
            sent += 1;
            in_flight += 1;
        }
        let wc = send_cq.poll_one(deadline()).expect("bw send completion");
        assert_eq!(wc.status, verbs::ffi::wc_status::SUCCESS, "bw send wc");
        done += 1;
        in_flight -= 1;
    }
    let dt = t0.elapsed();
    let wc = recv_cq.poll_one(deadline()).expect("done recv");
    assert_eq!(wc.byte_len as usize, MARK_LEN, "done marker length");
    let gbps = total as f64 / dt.as_secs_f64() / 1e9;
    println!(
        "bw total={total}B ({} MiB) stripe={stripe} window={window} time={dt:?} => {gbps:.3} GiB/s one-way",
        total / (1 << 20)
    );
}
