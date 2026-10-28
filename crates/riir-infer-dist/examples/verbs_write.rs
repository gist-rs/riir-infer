//! RDMA WRITE / READ probe (Plan 623 Phase 3 — T3.2's open half: the
//! rkey semantics the verbs_ping registrations could not answer).
//!
//! ```text
//!   verbs_write --listen 192.168.10.2:4820 --key <64hex>   # target (register + verify)
//!   verbs_write --addr 192.168.10.2:4820 --key <64hex>     # initiator (drive the ops)
//! ```
//!
//! One QP pair (RC first — RDMA READ is RC-only — falling back to UC,
//! where only WRITE is legal per the IB spec; both sides run the same
//! deterministic `auto` walk, so the fallback stays in bootstrap sync),
//! bootstrapped over the MAC'd TCP side channel: the endpoint exchange
//! as in `verbs_ping`, then a second MAC'd frame carrying the target's
//! `{raddr, rkey, region_len, qp_type}` to the initiator. The rows:
//!
//! 1. **grant matrix** (target boot) — MRs registered under each access
//!    flag combination, granted `lkey`/`rkey` printed. The verbs_ping
//!    rounds always read `rkey=0x0` under the full flag set; this
//!    settles whether ANY combination is granted a nonzero rkey.
//! 2. **RDMA WRITE** — N ops of `--len` bytes, per-op pattern and
//!    per-op source slice (no in-flight overwrite), posted straight
//!    into the target's registered range. Every completion's status +
//!    opcode recorded; the target verifies every byte against the
//!    per-op pattern after the `done` marker.
//! 3. **RDMA READ** (RC only) — the initiator pulls op 0's region back
//!    into its own LOCAL_WRITE slice and verifies the pattern.
//!
//! Every step prints its outcome — the probe's PURPOSE is to record the
//! provider's remote-access posture either way: a completing WRITE with
//! `rkey=0x0` says the zero-copy one-way shape IS available (and that
//! the provider's rkey is a handle, not an access token); a refused
//! WRITE names the WC status and closes the zero-copy claim for this
//! provider posture.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use riir_infer_dist::sockopts;
use riir_infer_dist::transport::PeerKey;
use riir_infer_dist::verbs::{
    self, CompQ, DevCtx, Endpoint, ProtDom, Qp, Rdma, clamp_mtu, exchange_endpoint, random_psn,
};

/// The `done` marker the initiator SENDs after the ops (magic u32 | ops
/// u32 | write_ok u32 | read_ok u32).
const MARK_LEN: usize = 16;
const MARK_MAGIC: u32 = 0x57_52_49_54; // "WRIT"
/// The verify guard band before the write-target area.
const GUARD: usize = 64;

/// The target's remote-access coordinates, MAC'd to the initiator.
struct RkeyFrame {
    raddr: u64,
    rkey: u32,
    region_len: u32,
    qp_ty: i32,
}

const RKEY_WIRE: usize = 8 + 4 + 4 + 4;
const RKEY_MAC_DOMAIN: &[u8] = b"riir-infer-dist/rdma-rkey-v1";

impl RkeyFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.raddr.to_le_bytes());
        out.extend_from_slice(&self.rkey.to_le_bytes());
        out.extend_from_slice(&self.region_len.to_le_bytes());
        out.extend_from_slice(&self.qp_ty.to_le_bytes());
    }

    fn decode(b: &[u8]) -> Result<Self, String> {
        if b.len() != RKEY_WIRE {
            return Err(format!(
                "rkey frame {} bytes, expected {RKEY_WIRE}",
                b.len()
            ));
        }
        Ok(Self {
            raddr: u64::from_le_bytes(b[0..8].try_into().expect("8")),
            rkey: u32::from_le_bytes(b[8..12].try_into().expect("4")),
            region_len: u32::from_le_bytes(b[12..16].try_into().expect("4")),
            qp_ty: i32::from_le_bytes(b[16..20].try_into().expect("4")),
        })
    }
}

/// Same shape as `exchange_endpoint`: `nonce[32] | frame | mac[32]`,
/// BLAKE3-keyed MAC'd both ways — no unauthenticated remote-access
/// coordinates are ever handed to a WR.
fn exchange_rkey(
    stream: &mut std::net::TcpStream,
    key: &PeerKey,
    mine: &RkeyFrame,
) -> std::io::Result<RkeyFrame> {
    use std::io::{Read, Write};

    let mut nonce = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut nonce)?;

    let mut payload = Vec::with_capacity(RKEY_WIRE + 64 + 32);
    payload.extend_from_slice(&nonce);
    mine.encode(&mut payload);
    let mac = {
        let mut h = blake3::Hasher::new_keyed(&key.0);
        h.update(RKEY_MAC_DOMAIN);
        h.update(&payload);
        h.finalize()
    };
    payload.extend_from_slice(mac.as_bytes());
    stream.write_all(&payload)?;
    stream.flush()?;

    let mut peer = vec![0u8; 32 + RKEY_WIRE + 32];
    stream.read_exact(&mut peer)?;
    let (p_nonce_frame, p_mac) = peer.split_at(32 + RKEY_WIRE);
    let want = {
        let mut h = blake3::Hasher::new_keyed(&key.0);
        h.update(RKEY_MAC_DOMAIN);
        h.update(p_nonce_frame);
        h.finalize()
    };
    if want.as_bytes() != p_mac {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "rkey frame MAC mismatch — refusing unauthenticated remote-access coordinates",
        ));
    }
    RkeyFrame::decode(&p_nonce_frame[32..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn pattern_byte(op: usize) -> u8 {
    (op as u8).wrapping_mul(0x11).wrapping_add(0x10) | 0x80
}

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
    peer: RkeyFrame,
    qp_ty_name: &'static str,
}

/// The shared bring-up: caps print → CQs → QP walk (RC first under
/// `auto`, UC fallback — both sides run the identical deterministic
/// walk, so the bootstrap stays in sync even mid-fallback) → endpoint
/// exchange → MAC'd rkey-frame exchange. `my_frame` is the target's real
/// remote-access coordinates; the initiator passes zeros.
#[allow(clippy::too_many_arguments)]
fn bring_up(
    stream: &mut std::net::TcpStream,
    key: &PeerKey,
    ctx: &Arc<DevCtx>,
    pd: &Arc<ProtDom>,
    port_req: Option<u8>,
    gid_index: Option<u32>,
    mtu_override: Option<u32>,
    qp_mode: &str,
    my_frame: &RkeyFrame,
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
    let path_mtu =
        clamp_mtu(port_attr.active_mtu, mtu_override).unwrap_or_else(|e| panic!("mtu: {e}"));

    let send_cq = Arc::new(pd.create_cq(256).unwrap_or_else(|e| panic!("send cq: {e}")));
    let recv_cq = Arc::new(pd.create_cq(256).unwrap_or_else(|e| panic!("recv cq: {e}")));
    let cap = verbs::ffi::IbvQpCap {
        max_send_wr: 256,
        max_recv_wr: 256,
        max_send_sge: 1,
        max_recv_sge: 1,
        max_inline_data: 224,
    };

    let mut try_walk = |ty: i32, ty_name: &'static str| -> Result<(Qp, &'static str), String> {
        let qp = match ty {
            verbs::ffi::qp_type::RC => pd.create_rc_qp(&send_cq, &recv_cq, cap),
            _ => pd.create_uc_qp(&send_cq, &recv_cq, cap),
        }
        .map_err(|e| format!("create: {e}"))?
        .0;
        let acc = verbs::ffi::access::LOCAL_WRITE
            | verbs::ffi::access::REMOTE_WRITE
            | verbs::ffi::access::REMOTE_READ;
        qp.to_init(vp, acc).map_err(|e| format!("INIT: {e}"))?;
        let psn = random_psn().map_err(|e| format!("psn: {e}"))?;
        // The GID index-0 fallback is the working path on the TB bridge
        // (GIDs are NOT IPv4-mapped there — Bench 927).
        let (sgid_idx, sgid) = match gid_index {
            Some(idx) => (
                idx,
                ctx.gid(vp, idx).map_err(|e| format!("gid {idx}: {e}"))?,
            ),
            None => ctx
                .pick_gid(vp, None)
                .map_err(|e| format!("pick gid: {e}"))?,
        };
        let mine = Endpoint {
            gid: sgid,
            qpn: qp.qp_num(),
            psn,
            mtu: path_mtu,
        };
        let mut peer_ep =
            exchange_endpoint(stream, key, &mine).map_err(|e| format!("endpoint exchange: {e}"))?;
        peer_ep.mtu = clamp_mtu(port_attr.active_mtu, Some(peer_ep.mtu)).unwrap();
        qp.to_rtr(&peer_ep, sgid_idx as u8, vp)
            .map_err(|e| format!("RTR: {e}"))?;
        qp.to_rts(psn).map_err(|e| format!("RTS: {e}"))?;
        eprintln!(
            "[rdma] {ty_name} QP {} in RTS: path_mtu={}",
            qp.qp_num(),
            peer_ep.mtu_name(),
        );
        Ok((qp, ty_name))
    };

    let (qp, qp_ty_name) = match qp_mode {
        "rc" => try_walk(verbs::ffi::qp_type::RC, "RC")
            .unwrap_or_else(|e| panic!("--qp rc: RC walk failed ({e})")),
        "uc" => try_walk(verbs::ffi::qp_type::UC, "UC")
            .unwrap_or_else(|e| panic!("--qp uc: UC walk failed ({e})")),
        _ => match try_walk(verbs::ffi::qp_type::RC, "RC") {
            Ok(found) => found,
            Err(rc_err) => {
                eprintln!("[qp] RC walk unavailable: {rc_err} — falling back to UC");
                try_walk(verbs::ffi::qp_type::UC, "UC").unwrap_or_else(|e| {
                    panic!("neither RC nor UC walk succeeded (RC: {rc_err}; UC: {e})")
                })
            }
        },
    };

    let peer = exchange_rkey(stream, key, my_frame).unwrap_or_else(|e| panic!("rkey frame: {e}"));
    Brought {
        qp,
        send_cq,
        recv_cq,
        peer,
        qp_ty_name,
    }
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
                "usage: verbs_write (--listen IP:PORT | --addr IP:PORT) --key <64hex> [--len N] [--ops N] [--qp rc|uc|auto] [--dev NAME] [--port N] [--gid-index N] [--mtu N]"
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
    let mtu_override: Option<u32> = get("mtu").map(|v| match v.as_str() {
        "1024" => verbs::ffi::mtu::MTU_1024,
        "2048" => verbs::ffi::mtu::MTU_2048,
        "4096" => verbs::ffi::mtu::MTU_4096,
        other => {
            eprintln!("--mtu {other} invalid");
            std::process::exit(2);
        }
    });
    let len: usize = get("len").map_or(4096, |v| v.parse().expect("--len BYTES"));
    let ops: usize = get("ops").map_or(8, |v| v.parse().expect("--ops N"));
    let qp_mode: String = get("qp").unwrap_or_else(|| "auto".into());
    let lat = Duration::from_secs(10);

    // ---- side channel --------------------------------------------------
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

    // ---- verbs bring-up ------------------------------------------------
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
            mtu_override,
            &qp_mode,
            len,
            ops,
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
            mtu_override,
            &qp_mode,
            len,
            ops,
            lat,
        );
    }
}

/// The access-flag grant matrix: one tiny MR per interesting combination,
/// granted lkey/rkey printed. A refused registration is a row, not a crash.
fn grant_matrix(pd: &ProtDom) {
    let combos: &[(&str, u32)] = &[
        ("LOCAL_WRITE", verbs::ffi::access::LOCAL_WRITE),
        (
            "LW|REMOTE_READ",
            verbs::ffi::access::LOCAL_WRITE | verbs::ffi::access::REMOTE_READ,
        ),
        (
            "LW|REMOTE_WRITE",
            verbs::ffi::access::LOCAL_WRITE | verbs::ffi::access::REMOTE_WRITE,
        ),
        (
            "LW|RW|RR (full)",
            verbs::ffi::access::LOCAL_WRITE
                | verbs::ffi::access::REMOTE_WRITE
                | verbs::ffi::access::REMOTE_READ,
        ),
        (
            "REMOTE_WRITE only (spec-invalid)",
            verbs::ffi::access::REMOTE_WRITE,
        ),
        (
            "REMOTE_READ only (spec-invalid)",
            verbs::ffi::access::REMOTE_READ,
        ),
    ];
    let mut buf = [0u8; 4096];
    println!("grant matrix (lkey/rkey per access-flag combination):");
    for (name, acc) in combos {
        match pd.reg_mr(buf.as_mut_ptr(), buf.len(), *acc) {
            Ok(mr) => println!("  {name:<34} lkey={:#010x} rkey={:#010x}", mr.lkey, mr.rkey),
            Err(e) => println!("  {name:<34} REFUSED: {e}"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_target(
    stream: &mut std::net::TcpStream,
    key: &PeerKey,
    ctx: &Arc<verbs::DevCtx>,
    pd: &Arc<verbs::ProtDom>,
    port_req: Option<u8>,
    gid_index: Option<u32>,
    mtu_override: Option<u32>,
    qp_mode: &str,
    len: usize,
    ops: usize,
    lat: Duration,
) {
    grant_matrix(pd);

    // Region: [0, GUARD) guard band | data area [GUARD, GUARD + ops*len).
    let region = GUARD + ops * len;
    let mut buf = vec![0u8; region];
    for b in buf.iter_mut() {
        *b = 0x5A; // sentinel — a WRITE that lands short shows here
    }
    let acc = verbs::ffi::access::LOCAL_WRITE
        | verbs::ffi::access::REMOTE_WRITE
        | verbs::ffi::access::REMOTE_READ;
    let mr = pd
        .reg_mr(buf.as_mut_ptr(), buf.len(), acc)
        .unwrap_or_else(|e| panic!("target reg_mr: {e}"));
    println!(
        "target MR: addr={:#x} len={} lkey={:#010x} rkey={:#010x}",
        mr.addr, mr.len, mr.lkey, mr.rkey
    );

    // The real remote-access coordinates cross in bring_up's rkey-frame
    // exchange (walk first, exchange last). The qp_ty byte stays 0 — both
    // sides run the same deterministic auto walk, so the type is already
    // agreed; the initiator's echoed frame is a cross-check only.
    let brought = bring_up(
        stream,
        key,
        ctx,
        pd,
        port_req,
        gid_index,
        mtu_override,
        qp_mode,
        &RkeyFrame {
            raddr: mr.addr + GUARD as u64,
            rkey: mr.rkey,
            region_len: (ops * len) as u32,
            qp_ty: 0,
        },
    );
    let peer = brought.peer;
    if peer.qp_ty != 0 && peer.qp_ty != i32::from(brought.qp_ty_name == "RC") {
        println!(
            "NOTE: initiator reports qp_ty={} vs our walk={}",
            peer.qp_ty, brought.qp_ty_name
        );
    }

    // Wait for the done marker, then verify every byte.
    let deadline = Instant::now() + lat;
    brought
        .qp
        .post_recv(&mr.slice_at(0, MARK_LEN as u32), 7)
        .expect("target marker recv");
    let wc = brought.recv_cq.poll_one(deadline).expect("target marker");
    if wc.byte_len as usize != MARK_LEN {
        panic!("target marker length {}", wc.byte_len);
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let iops = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let w_ok = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let r_ok = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    println!("initiator marker: magic={magic:#x} ops={iops} write_ok={w_ok} read_ok={r_ok}");

    let mut bad: Option<(usize, u8, u8)> = None;
    'verify: for op in 0..ops {
        let p = pattern_byte(op);
        let base = GUARD + op * len;
        for (i, &cell) in buf[base..base + len].iter().enumerate() {
            if cell != p {
                bad = Some((base + i, p, cell));
                break 'verify;
            }
        }
    }
    let guard_clean = buf[..GUARD].iter().all(|&c| c == 0x5A);
    match bad {
        None => println!(
            "target VERIFY: PASS — all {ops}×{len} B match their per-op patterns; guard band clean={guard_clean}"
        ),
        Some((off, want, got)) => println!(
            "target VERIFY: FAIL — offset {off}: expected {want:#x} got {got:#x} (guard_clean={guard_clean})"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_initiator(
    stream: &mut std::net::TcpStream,
    key: &PeerKey,
    ctx: &Arc<verbs::DevCtx>,
    pd: &Arc<verbs::ProtDom>,
    port_req: Option<u8>,
    gid_index: Option<u32>,
    mtu_override: Option<u32>,
    qp_mode: &str,
    len: usize,
    ops: usize,
    lat: Duration,
) {
    let placeholder = RkeyFrame {
        raddr: 0,
        rkey: 0,
        region_len: 0,
        qp_ty: 0,
    };
    let brought = bring_up(
        stream,
        key,
        ctx,
        pd,
        port_req,
        gid_index,
        mtu_override,
        qp_mode,
        &placeholder,
    );
    let Brought {
        qp,
        send_cq,
        recv_cq: _recv_cq,
        peer,
        qp_ty_name,
    } = brought;
    println!(
        "target frame: raddr={:#x} rkey={:#010x} region={} qp_ty={} (walk={qp_ty_name})",
        peer.raddr, peer.rkey, peer.region_len, peer.qp_ty
    );
    if (peer.region_len as usize) < ops * len {
        panic!("target region {} < ops*len {}", peer.region_len, ops * len);
    }
    let rc = qp_ty_name == "RC";
    let deadline = Instant::now() + lat;

    // Local MR: [0, MARK_LEN) marker | per-op source slices at 64.
    let mut local = vec![0u8; 64 + ops * len];
    let lmr = pd
        .reg_mr(
            local.as_mut_ptr(),
            local.len(),
            verbs::ffi::access::LOCAL_WRITE,
        )
        .unwrap_or_else(|e| panic!("initiator reg_mr: {e}"));

    // ---- RDMA WRITE ladder (per-op source slice: no in-flight overwrite)
    let mut completed = 0usize;
    for op in 0..ops {
        let p = pattern_byte(op);
        let src_off = 64 + op * len;
        local[src_off..src_off + len].fill(p);
        let sge = lmr.slice_at(src_off, len as u32);
        if let Err(e) = qp.post_rdma_write(
            &sge,
            peer.raddr + (op * len) as u64,
            peer.rkey,
            100 + op as u64,
            len <= 224,
        ) {
            println!("write op {op}: POST refused: {e}");
            break;
        }
        let wc = send_cq.poll_one(deadline).expect("write completion");
        if wc.status != verbs::ffi::wc_status::SUCCESS {
            println!(
                "write op {op}: WC status={} (vendor_err={:#x}) opcode={} — ladder stopped",
                wc.status, wc.vendor_err, wc.opcode
            );
            break;
        }
        completed += 1;
    }
    let inline = len <= 224;
    println!(
        "RDMA WRITE: {completed}/{ops} completed SUCCESS (qp={qp_ty_name}, len={len}, inline={inline})"
    );

    // ---- RDMA READ (RC only) -------------------------------------------
    let mut read_ok = 0u32;
    if !rc {
        println!("RDMA READ: skipped (UC has no READ responder)");
    } else if completed == ops {
        let sge = lmr.slice_at(64, len as u32);
        match qp.post_rdma_read(&sge, peer.raddr, peer.rkey, 900) {
            Ok(()) => {
                let wc = send_cq.poll_one(deadline).expect("read completion");
                if wc.status == verbs::ffi::wc_status::SUCCESS {
                    let p0 = pattern_byte(0);
                    let good = local[64..64 + len].iter().all(|&c| c == p0);
                    println!(
                        "RDMA READ: WC SUCCESS, content verify = {}",
                        if good { "PASS" } else { "FAIL" }
                    );
                    if good {
                        read_ok = 1;
                    }
                } else {
                    println!("RDMA READ: WC status={} — refused", wc.status);
                }
            }
            Err(e) => println!("RDMA READ: POST refused: {e}"),
        }
    } else {
        println!("RDMA READ: skipped (write ladder incomplete)");
    }

    // ---- done marker ----------------------------------------------------
    let marker: [u8; MARK_LEN] = {
        let mut m = [0u8; MARK_LEN];
        m[0..4].copy_from_slice(&MARK_MAGIC.to_le_bytes());
        m[4..8].copy_from_slice(&(ops as u32).to_le_bytes());
        m[8..12].copy_from_slice(&u32::from(completed == ops).to_le_bytes());
        m[12..16].copy_from_slice(&read_ok.to_le_bytes());
        m
    };
    local[..MARK_LEN].copy_from_slice(&marker);
    qp.post_send(&lmr.slice_at(0, MARK_LEN as u32), 2, true)
        .expect("marker send");
    let _ = send_cq.poll_one(deadline).expect("marker completion");
}
