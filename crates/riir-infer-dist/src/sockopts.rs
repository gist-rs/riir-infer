//! Socket knobs for the stage link (Plan 623 T2.1b).
//!
//! [`dial`] / [`listen`] are the std-only replacements for
//! `TcpStream::connect` / `TcpListener::bind` that can apply the two
//! knobs std does not expose:
//!
//! - **`SO_SNDBUF`/`SO_RCVBUF` sizing** — a chunked prefill frame is
//!   ~`chunk × hidden × 4` bytes (2.6 MB at the T2.3b defaults: 128 ×
//!   5120 × f32); a larger socket buffer smooths the TB5 push.
//! - **bind to a named interface** — the rig has two paths (TB5 bridge
//!   192.168.10.x, tailnet 100.x); the stage link must be able to PIN
//!   the fast one (macOS `IP_BOUND_IF`/`IPV6_BOUND_IF`, Linux
//!   `SO_BINDTODEVICE`). On the listener this is also the posture knob:
//!   only that interface can reach the stage. Both entry points apply
//!   the interface bind BEFORE bind/connect — the route is chosen at
//!   connect time, so a post-connect bind would pin nothing.
//!
//! No new crates.io dep: the handful of libc symbols are declared
//! `extern "C"` here (every Rust binary links them on unix targets).
//! Values are READ BACK with `getsockopt` and returned in
//! [`Applied`] — the printed/recorded numbers are what the kernel
//! holds (Linux doubles a requested buffer; the truth wins).
//!
//! Byte-identical default: `LinkOptions::default()` applies nothing and
//! both functions behave like their std counterparts (plus
//! `SO_REUSEADDR` on the listener, which std's unix bind also sets).

use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream};

#[cfg(any(target_os = "macos", target_os = "linux"))]
use std::os::fd::FromRawFd;

#[cfg(any(target_os = "macos", target_os = "linux"))]
use core::ffi::c_int;

/// What the kernel actually holds after the knobs applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    /// `SO_SNDBUF` read back (0 = untouched).
    pub snd: usize,
    /// `SO_RCVBUF` read back (0 = untouched).
    pub rcv: usize,
    /// Interface index the socket is bound to (`None` = unbound).
    pub iface: Option<u32>,
}

impl Applied {
    fn untouched() -> Self {
        Self {
            snd: 0,
            rcv: 0,
            iface: None,
        }
    }
}

/// The T2.1b knobs. Absent = std behavior, byte-identical.
#[derive(Debug, Clone, Default)]
pub struct LinkOptions {
    /// Set BOTH `SO_SNDBUF` and `SO_RCVBUF` to this many bytes.
    pub sock_buf: Option<usize>,
    /// Bind the socket to this named interface (macOS: `en0`, `bridge0`,
    /// `lo0`; Linux: the device name, e.g. `eth0`).
    pub iface: Option<String>,
}

impl LinkOptions {
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn is_noop(&self) -> bool {
        self.sock_buf.is_none() && self.iface.is_none()
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use core::ffi::{c_int, c_void};

    pub const AF_INET: c_int = 2;
    pub const AF_INET6: c_int = 30;
    pub const SOCK_STREAM: c_int = 1;
    pub const SOL_SOCKET: c_int = 0xffff;
    pub const SO_REUSEADDR: c_int = 0x0004;
    pub const SO_SNDBUF: c_int = 0x1001;
    pub const SO_RCVBUF: c_int = 0x1002;
    pub const IPPROTO_IP: c_int = 0;
    pub const IPPROTO_IPV6: c_int = 41;
    pub const IP_BOUND_IF: c_int = 25;
    pub const IPV6_BOUND_IF: c_int = 125;
    pub const F_SETFD: c_int = 2;
    pub const FD_CLOEXEC: c_int = 1;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct SockaddrIn {
        pub sin_len: u8,
        pub sin_family: u8,
        pub sin_port: u16,
        pub sin_addr: [u8; 4],
        pub sin_zero: [u8; 8],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct SockaddrIn6 {
        pub sin6_len: u8,
        pub sin6_family: u8,
        pub sin6_port: u16,
        pub sin6_flowinfo: u32,
        pub sin6_addr: [u16; 8],
        pub sin6_scope_id: u32,
    }

    unsafe extern "C" {
        pub fn socket(domain: c_int, ty: c_int, protocol: c_int) -> c_int;
        pub fn close(fd: c_int) -> c_int;
        pub fn setsockopt(
            fd: c_int,
            level: c_int,
            name: c_int,
            value: *const c_void,
            len: u32,
        ) -> c_int;
        pub fn getsockopt(
            fd: c_int,
            level: c_int,
            name: c_int,
            value: *mut c_void,
            len: *mut u32,
        ) -> c_int;
        pub fn bind(fd: c_int, addr: *const c_void, len: u32) -> c_int;
        pub fn listen(fd: c_int, backlog: c_int) -> c_int;
        pub fn connect(fd: c_int, addr: *const c_void, len: u32) -> c_int;
        pub fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
        pub fn if_nametoindex(name: *const std::ffi::c_char) -> u32;
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use core::ffi::{c_int, c_void};

    pub const AF_INET: c_int = 2;
    pub const AF_INET6: c_int = 10;
    pub const SOCK_STREAM: c_int = 1;
    pub const SOCK_CLOEXEC: c_int = 0o2000000;
    pub const SOL_SOCKET: c_int = 1;
    pub const SO_REUSEADDR: c_int = 2;
    pub const SO_SNDBUF: c_int = 7;
    pub const SO_RCVBUF: c_int = 8;
    pub const SO_BINDTODEVICE: c_int = 25;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct SockaddrIn {
        pub sin_family: u16,
        pub sin_port: u16,
        pub sin_addr: [u8; 4],
        pub sin_zero: [u8; 8],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct SockaddrIn6 {
        pub sin6_family: u16,
        pub sin6_port: u16,
        pub sin6_flowinfo: u32,
        pub sin6_addr: [u16; 8],
        pub sin6_scope_id: u32,
    }

    unsafe extern "C" {
        pub fn socket(domain: c_int, ty: c_int, protocol: c_int) -> c_int;
        pub fn close(fd: c_int) -> c_int;
        pub fn setsockopt(
            fd: c_int,
            level: c_int,
            name: c_int,
            value: *const c_void,
            len: u32,
        ) -> c_int;
        pub fn getsockopt(
            fd: c_int,
            level: c_int,
            name: c_int,
            value: *mut c_void,
            len: *mut u32,
        ) -> c_int;
        pub fn bind(fd: c_int, addr: *const c_void, len: u32) -> c_int;
        pub fn listen(fd: c_int, backlog: c_int) -> c_int;
        pub fn connect(fd: c_int, addr: *const c_void, len: u32) -> c_int;
        pub fn if_nametoindex(name: *const std::ffi::c_char) -> u32;
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    // Compile-anywhere stub: every knob REFUSES (fail-closed) and both
    // entry points fall back to std.
}

// Unix-impl helper: every use sits inside the macos/linux regions, so the
// item is cfg'd to match (the platform-dead-code rule — ungated here reads
// dead on every other target).
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn last_err(what: &str) -> io::Error {
    let e = io::Error::last_os_error();
    io::Error::new(e.kind(), format!("{what}: {e}"))
}

/// `sock_buf`/`iface` on a platform with no knob impl = a loud refusal
/// (never a silent std fallback that lies about the posture).
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn knob_unsupported(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!("{what}: socket knobs are not implemented on this platform"),
    )
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod unix {
    use super::{Applied, LinkOptions, imp};
    use core::ffi::{c_int, c_void};
    use std::io;
    use std::net::SocketAddr;

    fn last_err(what: &str) -> io::Error {
        super::last_err(what)
    }

    fn cstring(name: &str) -> std::ffi::CString {
        std::ffi::CString::new(name).expect("interface name contains no NUL")
    }

    enum Sockaddr {
        V4(imp::SockaddrIn),
        V6(imp::SockaddrIn6),
    }

    fn sockaddr_of(addr: SocketAddr) -> (Sockaddr, u32) {
        const LEN_V4: u32 = std::mem::size_of::<imp::SockaddrIn>() as u32;
        const LEN_V6: u32 = std::mem::size_of::<imp::SockaddrIn6>() as u32;
        match addr {
            SocketAddr::V4(a) => {
                #[cfg(target_os = "macos")]
                let sin = imp::SockaddrIn {
                    sin_len: LEN_V4 as u8,
                    sin_family: imp::AF_INET as u8,
                    sin_port: a.port().to_be(),
                    sin_addr: a.ip().octets(),
                    sin_zero: [0; 8],
                };
                #[cfg(target_os = "linux")]
                let sin = imp::SockaddrIn {
                    sin_family: imp::AF_INET as u16,
                    sin_port: a.port().to_be(),
                    sin_addr: a.ip().octets(),
                    sin_zero: [0; 8],
                };
                (Sockaddr::V4(sin), LEN_V4)
            }
            SocketAddr::V6(a) => {
                #[cfg(target_os = "macos")]
                let sin6 = imp::SockaddrIn6 {
                    sin6_len: LEN_V6 as u8,
                    sin6_family: imp::AF_INET6 as u8,
                    sin6_port: a.port().to_be(),
                    sin6_flowinfo: a.flowinfo(),
                    sin6_addr: a.ip().segments(),
                    sin6_scope_id: a.scope_id(),
                };
                #[cfg(target_os = "linux")]
                let sin6 = imp::SockaddrIn6 {
                    sin6_family: imp::AF_INET6 as u16,
                    sin6_port: a.port().to_be(),
                    sin6_flowinfo: a.flowinfo(),
                    sin6_addr: a.ip().segments(),
                    sin6_scope_id: a.scope_id(),
                };
                (Sockaddr::V6(sin6), LEN_V6)
            }
        }
    }

    fn as_ptr(a: &Sockaddr) -> (*const c_void, u32) {
        const LEN_V4: u32 = std::mem::size_of::<imp::SockaddrIn>() as u32;
        const LEN_V6: u32 = std::mem::size_of::<imp::SockaddrIn6>() as u32;
        match a {
            Sockaddr::V4(s) => (s as *const imp::SockaddrIn as *const c_void, LEN_V4),
            Sockaddr::V6(s) => (s as *const imp::SockaddrIn6 as *const c_void, LEN_V6),
        }
    }

    pub(super) fn domain_of(addr: SocketAddr) -> c_int {
        match addr {
            SocketAddr::V4(_) => imp::AF_INET,
            SocketAddr::V6(_) => imp::AF_INET6,
        }
    }

    /// Fresh blocking TCP socket, close-on-exec.
    pub(super) fn new_socket(domain: c_int) -> io::Result<c_int> {
        #[cfg(target_os = "linux")]
        let ty = imp::SOCK_STREAM | imp::SOCK_CLOEXEC;
        #[cfg(target_os = "macos")]
        let ty = imp::SOCK_STREAM;
        let fd = unsafe { imp::socket(domain, ty, 0) };
        if fd < 0 {
            return Err(last_err("socket()"));
        }
        #[cfg(target_os = "macos")]
        if unsafe { imp::fcntl(fd, imp::F_SETFD, imp::FD_CLOEXEC) } != 0 {
            let e = last_err("fcntl(FD_CLOEXEC)");
            unsafe {
                imp::close(fd);
            }
            return Err(e);
        }
        Ok(fd)
    }

    fn set_buf(fd: c_int, bytes: usize, snd: bool) -> io::Result<usize> {
        let name = match snd {
            true => imp::SO_SNDBUF,
            false => imp::SO_RCVBUF,
        };
        let v: c_int = bytes.min(c_int::MAX as usize) as c_int;
        let what = match snd {
            true => "setsockopt(SO_SNDBUF)",
            false => "setsockopt(SO_RCVBUF)",
        };
        if unsafe {
            imp::setsockopt(
                fd,
                imp::SOL_SOCKET,
                name,
                (&v) as *const c_int as *const c_void,
                4,
            )
        } != 0
        {
            return Err(last_err(what));
        }
        let mut got: c_int = 0;
        let mut got_len = 4u32;
        if unsafe {
            imp::getsockopt(
                fd,
                imp::SOL_SOCKET,
                name,
                (&mut got) as *mut c_int as *mut c_void,
                &mut got_len,
            )
        } != 0
        {
            return Err(last_err("getsockopt(buffer)"));
        }
        Ok(got.max(0) as usize)
    }

    fn bind_iface(fd: c_int, v6: bool, name: &str) -> io::Result<u32> {
        let idx = unsafe { imp::if_nametoindex(cstring(name).as_ptr()) };
        if idx == 0 {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("interface {name:?}: no such interface"),
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let (level, opt) = match v6 {
                true => (imp::IPPROTO_IPV6, imp::IPV6_BOUND_IF),
                false => (imp::IPPROTO_IP, imp::IP_BOUND_IF),
            };
            if unsafe { imp::setsockopt(fd, level, opt, (&idx) as *const u32 as *const c_void, 4) }
                != 0
            {
                return Err(last_err("setsockopt(bound_if)"));
            }
        }
        #[cfg(target_os = "linux")]
        {
            let dev = cstring(name);
            if unsafe {
                imp::setsockopt(
                    fd,
                    imp::SOL_SOCKET,
                    imp::SO_BINDTODEVICE,
                    dev.as_ptr() as *const c_void,
                    dev.as_bytes_with_nul().len() as u32,
                )
            } != 0
            {
                return Err(last_err("setsockopt(SO_BINDTODEVICE)"));
            }
        }
        Ok(idx)
    }

    /// Apply the requested knobs, reading each back. Called BEFORE
    /// bind/connect so the interface bind actually pins the path.
    pub(super) fn apply_knobs(fd: c_int, v6: bool, o: &LinkOptions) -> io::Result<Applied> {
        let mut applied = Applied::untouched();
        if let Some(bytes) = o.sock_buf {
            applied.snd = set_buf(fd, bytes, true)?;
            applied.rcv = set_buf(fd, bytes, false)?;
        }
        if let Some(name) = &o.iface {
            applied.iface = Some(bind_iface(fd, v6, name)?);
        }
        Ok(applied)
    }

    pub(super) fn bind_listen(
        fd: c_int,
        addr: SocketAddr,
        v6: bool,
        o: &LinkOptions,
    ) -> io::Result<Applied> {
        // std's unix bind sets SO_REUSEADDR; keep the same relaunch
        // posture.
        let one: c_int = 1;
        if unsafe {
            imp::setsockopt(
                fd,
                imp::SOL_SOCKET,
                imp::SO_REUSEADDR,
                (&one) as *const c_int as *const c_void,
                4,
            )
        } != 0
        {
            return Err(last_err("setsockopt(SO_REUSEADDR)"));
        }
        // Knobs BEFORE bind: IP_BOUND_IF on a listener gates which
        // interface's traffic it accepts.
        let applied = apply_knobs(fd, v6, o)?;
        let (sa, _) = sockaddr_of(addr);
        let (p, l) = as_ptr(&sa);
        if unsafe { imp::bind(fd, p, l) } != 0 {
            return Err(last_err("bind()"));
        }
        if unsafe { imp::listen(fd, 4) } != 0 {
            return Err(last_err("listen()"));
        }
        Ok(applied)
    }

    pub(super) fn connect(
        fd: c_int,
        addr: SocketAddr,
        v6: bool,
        o: &LinkOptions,
    ) -> io::Result<Applied> {
        // Knobs BEFORE connect: the route is chosen at connect time, so
        // the interface bind must already be on the socket.
        let applied = apply_knobs(fd, v6, o)?;
        let (sa, _) = sockaddr_of(addr);
        let (p, l) = as_ptr(&sa);
        if unsafe { imp::connect(fd, p, l) } != 0 {
            return Err(last_err("connect()"));
        }
        Ok(applied)
    }
}

/// A tiny guard so every error path closes the raw fd exactly once.
#[cfg(any(target_os = "macos", target_os = "linux"))]
trait CloseFd {
    fn close_fd(self, fd: c_int) -> io::Error;
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
impl CloseFd for io::Error {
    fn close_fd(self, fd: c_int) -> io::Error {
        unsafe {
            imp::close(fd);
        }
        self
    }
}

/// Outbound stage link: std `TcpStream::connect` + the T2.1b knobs.
pub fn dial(addr: SocketAddr, o: &LinkOptions) -> io::Result<(TcpStream, Applied)> {
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        if !o.is_noop() {
            return Err(knob_unsupported("dial"));
        }
        Ok((TcpStream::connect(addr)?, Applied::untouched()))
    }
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let fd = unix::new_socket(unix::domain_of(addr))?;
        let v6 = matches!(addr, SocketAddr::V6(_));
        let applied = match unix::connect(fd, addr, v6, o) {
            Ok(a) => a,
            Err(e) => return Err(e.close_fd(fd)),
        };
        // SAFETY: we own `fd` (fresh socket, connected exactly once, and
        // every prior error path closed it before returning).
        Ok((unsafe { TcpStream::from_raw_fd(fd) }, applied))
    }
}

/// Listening stage socket: std `TcpListener::bind` + the T2.1b knobs
/// (`SO_REUSEADDR` set, matching std's unix bind posture).
pub fn listen(addr: SocketAddr, o: &LinkOptions) -> io::Result<(TcpListener, Applied)> {
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        if !o.is_noop() {
            return Err(knob_unsupported("listen"));
        }
        Ok((TcpListener::bind(addr)?, Applied::untouched()))
    }
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let fd = unix::new_socket(unix::domain_of(addr))?;
        let v6 = matches!(addr, SocketAddr::V6(_));
        match unix::bind_listen(fd, addr, v6, o) {
            Ok(applied) => {
                // SAFETY: we own `fd` (fresh listening socket; every
                // error path closed it before returning).
                Ok((unsafe { TcpListener::from_raw_fd(fd) }, applied))
            }
            Err(e) => Err(e.close_fd(fd)),
        }
    }
}

#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod tests {
    use super::*;
    use crate::transport::{Direction, FrameKind, PeerKey, TcpTransport, Transport};

    fn key() -> PeerKey {
        PeerKey([0x5a; 32])
    }

    fn buf_only(n: usize) -> LinkOptions {
        LinkOptions {
            sock_buf: Some(n),
            iface: None,
        }
    }

    #[test]
    fn dial_listen_applies_buffers_and_carries_frames() {
        let o = buf_only(1 << 19);
        let (l, la) = listen("127.0.0.1:0".parse().unwrap(), &o).expect("listen");
        assert!(la.snd >= 1 << 19, "listener snd {} < 512KiB", la.snd);
        assert!(la.rcv >= 1 << 19, "listener rcv {} < 512KiB", la.rcv);
        let addr = l.local_addr().unwrap();

        let server = std::thread::spawn(move || {
            let (sock, _peer) = l.accept().unwrap();
            let mut t = TcpTransport::new(sock, key(), Direction::Upstream).unwrap();
            let mut buf = Vec::new();
            let kind = t.recv_frame_into(&mut buf).unwrap();
            assert_eq!(kind, FrameKind::Begin);
            t.send_frame(FrameKind::Token, &7u32.to_le_bytes()).unwrap();
        });

        let (stream, da) = dial(addr, &o).expect("dial");
        assert!(da.snd >= 1 << 19, "dial snd {} < 512KiB", da.snd);
        assert!(da.rcv >= 1 << 19, "dial rcv {} < 512KiB", da.rcv);
        let mut t = TcpTransport::new(stream, key(), Direction::Downstream).unwrap();
        t.send_frame(FrameKind::Begin, &[1, 2, 3, 4, 1, 0, 0, 0])
            .unwrap();
        let mut buf = Vec::new();
        let kind = t.recv_frame_into(&mut buf).unwrap();
        assert_eq!(kind, FrameKind::Token);
        assert_eq!(buf, 7u32.to_le_bytes());
        server.join().unwrap();
    }

    #[test]
    fn no_options_is_std_equivalent() {
        let (l, la) =
            listen("127.0.0.1:0".parse().unwrap(), &LinkOptions::default()).expect("listen");
        assert_eq!(la, Applied::untouched());
        let addr = l.local_addr().unwrap();
        let (s, da) = dial(addr, &LinkOptions::default()).expect("dial");
        assert_eq!(da, Applied::untouched());
        drop(s);
    }

    #[test]
    fn unknown_interface_refuses() {
        let o = LinkOptions {
            sock_buf: None,
            iface: Some("no_such_if0".into()),
        };
        let e = listen("127.0.0.1:0".parse().unwrap(), &o).expect_err("listen must refuse");
        assert_eq!(e.kind(), io::ErrorKind::NotFound, "got {e}");
        let e = dial("127.0.0.1:9".parse().unwrap(), &o).expect_err("dial must refuse");
        assert_eq!(e.kind(), io::ErrorKind::NotFound, "got {e}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn iface_bound_listener_accepts_loopback() {
        let o = LinkOptions {
            sock_buf: None,
            iface: Some("lo0".into()),
        };
        let (l, la) = listen("127.0.0.1:0".parse().unwrap(), &o).expect("listen on lo0");
        assert_eq!(la.iface, Some(1), "lo0 is ifindex 1 on macOS");
        let addr = l.local_addr().unwrap();
        let server = std::thread::spawn(move || l.accept().unwrap().0);
        let (s, da) = dial(addr, &o).expect("dial over lo0");
        assert_eq!(da.iface, Some(1));
        let _peer = server.join().unwrap();
        drop(s);
    }
}
