//! Verbs FFI layout pins + endpoint codec + dlopen resolution
//! (Plan 623 Phase 3 T3.1).
//!
//! The layout expectations are the C COMPILER's measured offsets on
//! arm64-apple-macos (clang against the macOS 26.2+ SDK header — the
//! measurement program and its full output are on record in the T3.1
//! landing note). A struct drift here is a silent wire/ABI corruption
//! against the provider, so every mirrored struct is pinned to the
//! measured number, not to itself.
#![cfg(all(feature = "rdma", target_os = "macos"))]

use riir_infer_dist::verbs::ffi;
use riir_infer_dist::verbs::{clamp_mtu, Endpoint};

/// `(assert)` that a type's size and every named field's offset equal
/// the compiler-measured constants.
macro_rules! pin_layout {
    ($ty:ty, size = $size:expr, $( $field:ident @ $off:expr ),+ $(,)?) => {
        assert_eq!(
            std::mem::size_of::<$ty>(),
            $size,
            concat!(stringify!($ty), " size drift — re-measure against the SDK header")
        );
        $( assert_eq!(
            memoffset!($ty, $field),
            $off,
            concat!(stringify!($ty), ".", stringify!($field), " offset drift")
        ); )+
    };
}

macro_rules! memoffset {
    ($ty:ty, $field:ident) => {
        // offset_of! is stable since 1.77 — the pinned toolchain (1.99) has it.
        std::mem::offset_of!($ty, $field)
    };
}

#[test]
fn wire_structs_match_the_sdk_header_measured_layouts() {
    // union ibv_gid = 16
    assert_eq!(16, std::mem::size_of::<ffi::IbvGid>());
    pin_layout!(ffi::IbvGidEntry, size = 32,
        gid @ 0, gid_index @ 16, port_num @ 20, gid_type @ 24, ndev_ifindex @ 28);
    pin_layout!(ffi::IbvWc, size = 48,
        wr_id @ 0, status @ 8, opcode @ 12, vendor_err @ 16, byte_len @ 20,
        imm_data @ 24, qp_num @ 28, src_qp @ 32, wc_flags @ 36,
        pkey_index @ 40, slid @ 42, sl @ 44, dlid_path_bits @ 45);
    pin_layout!(ffi::IbvPd, size = 16, context @ 0, handle @ 8);
    pin_layout!(ffi::IbvMr, size = 48,
        context @ 0, pd @ 8, addr @ 16, length @ 24, handle @ 32, lkey @ 36, rkey @ 40);
    pin_layout!(ffi::IbvSge, size = 16, addr @ 0, length @ 8, lkey @ 12);
    pin_layout!(ffi::IbvSendWr, size = 128,
        wr_id @ 0, next @ 8, sg_list @ 16, num_sge @ 24, opcode @ 28,
        send_flags @ 32, imm_data @ 36, wr @ 40, qp_type_xrc @ 72, bind_mw @ 80);
    pin_layout!(ffi::IbvRecvWr, size = 32,
        wr_id @ 0, next @ 8, sg_list @ 16, num_sge @ 24);
    pin_layout!(ffi::IbvQpCap, size = 20,
        max_send_wr @ 0, max_recv_wr @ 4, max_send_sge @ 8, max_recv_sge @ 12,
        max_inline_data @ 16);
    pin_layout!(ffi::IbvQpInitAttr, size = 64,
        qp_context @ 0, send_cq @ 8, recv_cq @ 16, srq @ 24, cap @ 32,
        qp_type @ 52, sq_sig_all @ 56);
    pin_layout!(ffi::IbvGlobalRoute, size = 24,
        dgid @ 0, flow_label @ 16, sgid_index @ 20);
    pin_layout!(ffi::IbvAhAttr, size = 32,
        grh @ 0, dlid @ 24, sl @ 26, port_num @ 30);
    pin_layout!(ffi::IbvQpAttr, size = 144,
        qp_state @ 0, path_mtu @ 8, qkey @ 16, rq_psn @ 20, sq_psn @ 24,
        dest_qp_num @ 28, qp_access_flags @ 32, cap @ 36, ah_attr @ 56,
        alt_ah_attr @ 88, pkey_index @ 120, max_rd_atomic @ 126, port_num @ 129,
        timeout @ 130, retry_cnt @ 131, rnr_retry @ 132, rate_limit @ 136);
    pin_layout!(ffi::IbvQp, size = 184,
        context @ 0, qp_context @ 8, pd @ 16, send_cq @ 24, recv_cq @ 32,
        srq @ 40, handle @ 48, qp_num @ 52, state @ 56, qp_type @ 60);
    pin_layout!(ffi::IbvCompChannel, size = 16, context @ 0, fd @ 8, refcnt @ 12);
    pin_layout!(ffi::IbvCq, size = 152,
        context @ 0, channel @ 8, cq_context @ 16, handle @ 24, cqe @ 28);
    pin_layout!(ffi::IbvContextOps, size = 256,
        poll_cq @ 88, req_notify_cq @ 96, post_send @ 200, post_recv @ 208);
    pin_layout!(ffi::IbvContext, size = 352,
        device @ 0, ops @ 8, cmd_fd @ 264, async_fd @ 268, num_comp_vectors @ 272);
    pin_layout!(ffi::IbvPortAttr, size = 52,
        state @ 0, max_mtu @ 4, active_mtu @ 8, gid_tbl_len @ 12,
        port_cap_flags @ 16, max_msg_sz @ 20, pkey_tbl_len @ 32, lid @ 34,
        link_layer @ 46, flags @ 47, port_cap_flags2 @ 48);
    pin_layout!(ffi::IbvDeviceAttr, size = 232,
        node_guid @ 64, max_mr_size @ 80, max_qp @ 108, max_qp_wr @ 112,
        max_sge @ 120, max_cq @ 128, max_cqe @ 132, max_pkeys @ 224);
    pin_layout!(ffi::IbvDevice, size = 664,
        node_type @ 16, transport_type @ 20, name @ 24, dev_name @ 88,
        dev_path @ 152, ibdev_path @ 408);
}

#[test]
fn endpoint_wire_round_trip_and_refusals() {
    let ep = Endpoint {
        gid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 10, 2],
        qpn: 0xdead_beef,
        psn: 0x0012_34ef,
        mtu: ffi::mtu::MTU_4096,
    };
    let mut wire = Vec::new();
    ep.encode(&mut wire);
    assert_eq!(wire.len(), 28);
    assert_eq!(Endpoint::decode(&wire).unwrap(), ep);
    for bad in [27usize, 29, 0, 64] {
        assert!(Endpoint::decode(&vec![0u8; bad]).is_err(), "len {bad} must refuse");
    }
}

#[test]
fn dlopen_resolves_the_symbol_table() {
    // This box may have no RDMA DEVICE; the LIBRARY must still load.
    let syms = ffi::Syms::load().expect("dlopen librdma.dylib + resolve all symbols");
    // The status renderer round-trips through the library's own strings.
    assert_eq!(syms.wc_status(ffi::wc_status::SUCCESS), "success");
}

#[test]
fn device_enumeration_is_honest() {
    let rdma = riir_infer_dist::verbs::Rdma::open().expect("librdma loads");
    let names = rdma.device_names();
    // No assertion on the COUNT — a box without an RDMA endpoint is a
    // config fact. The contract is that enumeration either lists the
    // devices or returns empty WITHOUT panicking, and open_device on an
    // empty box fails LOUD naming the remedy.
    if names.is_empty() {
        let err = rdma
            .open_device(None)
            .err()
            .expect("open_device on an empty box must fail");
        assert!(err.contains("no RDMA devices"), "loud no-device error, got: {err}");
        eprintln!("no RDMA devices on this box (expected off the amo⇄bmo pair): {err}");
    } else {
        eprintln!("RDMA devices present: {names:?}");
        let ctx = rdma.open_device(None).expect("open first device");
        eprintln!("opened {} — ports probed next", ctx.dev_name);
        let caps = ctx.caps().expect("query_device");
        assert!(caps.max_qp > 0 && caps.max_qp_wr > 0 && caps.max_cqe > 0);
        for port in 1..=2u8 {
            if let Ok(p) = ctx.query_port(port) {
                eprintln!(
                    "port {port}: state={} mtu={} gid_tbl={} link={}",
                    p.state, p.active_mtu, p.gid_tbl_len, p.link_layer_name()
                );
            }
        }
    }
}

#[test]
fn mtu_clamp_law() {
    let (m1024, m2048, m4096) = (ffi::mtu::MTU_1024, ffi::mtu::MTU_2048, ffi::mtu::MTU_4096);
    assert_eq!(clamp_mtu(m4096, None).unwrap(), m4096);
    assert_eq!(clamp_mtu(m4096, Some(m1024)).unwrap(), m1024);
    assert_eq!(clamp_mtu(m1024, Some(m4096)).unwrap(), m1024);
    assert_eq!(clamp_mtu(m2048, Some(m2048)).unwrap(), m2048);
    assert!(clamp_mtu(9, None).is_err());
    assert!(clamp_mtu(0, None).is_err());
}

#[test]
fn qp_state_machine_constants_match_the_header_enums() {
    assert_eq!(ffi::qp_state::INIT, 1);
    assert_eq!(ffi::qp_state::RTR, 2);
    assert_eq!(ffi::qp_state::RTS, 3);
    assert_eq!(ffi::qp_type::RC, 2);
    assert_eq!(ffi::qp_type::UC, 3);
    assert_eq!(ffi::attr_mask::STATE, 1);
    assert_eq!(ffi::attr_mask::DEST_QPN, 1 << 20);
    assert_eq!(ffi::attr_mask::AV, 1 << 7);
    assert_eq!(ffi::send_flags::INLINE, 8);
    assert_eq!(ffi::wc_opcode::RECV, 1 << 7);
    assert_eq!(ffi::link_layer::THUNDERBOLT, 100);
    assert_eq!(ffi::port_state::ACTIVE, 4);
    assert_eq!(ffi::access::LOCAL_WRITE, 1);
    assert_eq!(ffi::access::REMOTE_WRITE, 2);
    assert_eq!(ffi::access::REMOTE_READ, 4);
}
