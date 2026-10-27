//! Authenticated frame transport.
//!
//! Envelope (little-endian):
//!
//! ```text
//! body_len u32 | kind u8 | counter u64 | payload[body_len - 9] | mac[32]
//! mac = blake3::keyed_hash(key, direction u8 | kind | counter | payload)
//! ```
//!
//! `direction` is not on the wire — each side knows which way a frame
//! flows, so a frame reflected back at its sender fails the MAC. The
//! receiver requires `counter == expected` (strict, per direction): a
//! replayed, dropped or reordered frame refuses. The MAC comparison is
//! blake3's constant-time `Hash` equality.

use std::io::{self, IoSlice, Read, Write};
use std::net::TcpStream;

/// Shared 32-byte pipeline key (pinned peer key — both stages hold it).
#[derive(Clone)]
pub struct PeerKey(pub [u8; 32]);

impl PeerKey {
    /// Parse 64 hex chars.
    pub fn from_hex(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.len() != 64 {
            return Err(format!("peer key must be 64 hex chars, got {}", s.len()));
        }
        let mut k = [0u8; 32];
        for (i, byte) in k.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)
                .map_err(|e| format!("peer key hex at {}: {e}", 2 * i))?;
        }
        Ok(Self(k))
    }
}

/// Which way a frame flows; mixed into the MAC (anti-reflection).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    Downstream = 0,
    Upstream = 1,
}

impl Direction {
    fn flip(self) -> Self {
        match self {
            Self::Downstream => Self::Upstream,
            Self::Upstream => Self::Downstream,
        }
    }
}

/// Frame kinds of the stage protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    Handshake = 0,
    Activation = 1,
    Token = 2,
    Shutdown = 3,
    /// An activation the tail consumes WITHOUT a token reply — every
    /// prompt position but the last (T2.3 pipelined prefill: the head
    /// computes pos t+1 while the tail computes pos t, and no reply
    /// backlog can fill the head's receive buffer on a long prompt).
    Prefill = 4,
    /// Opens a sequence: `prompt_len` u32 LE · `prefill_chunk` u32 LE
    /// (T2.3b). The tail needs the WHOLE prompt length before the first
    /// chunk (the attention-arm gates key on it, Issue 782) and the chunk
    /// width to know which positions ride the batched prefill.
    Begin = 5,
    /// Opens ONE stream of a multi-stream run (T2.4): `seq_id u32 LE ·
    /// prompt_len u32 LE · prefill_chunk u32 LE`. The stream-addressed
    /// twin of [`FrameKind::Begin`] — the single-stream frame is left
    /// untouched so both protocols coexist on one transport vocabulary.
    BeginStream = 6,
    /// The token reply of ONE stream (T2.4): `seq_id u32 LE · token u32
    /// LE`. [`FrameKind::Token`] carries no id to route by; a multi-stream
    /// head has up to one reply outstanding per stream and needs the id.
    TokenStream = 7,
    /// One speculative VERIFY round (T5.1a): `seq_id u32 LE · n_tok u32 LE
    /// · n_tok×u32 draft tokens · n_tok×hidden f32 rows` — the head's half
    /// for the K drafted positions, so the tail verifies all K in ONE round
    /// trip (the per-token round trip is the decode lever, T2.6). The tail
    /// computes each row's argmax on its own logits — K u32 cross the wire,
    /// never K×vocab — and adjudicates the accepted prefix itself (both
    /// sides derive the same `m` from the same inputs; the reply carries it
    /// as a cross-check).
    VerifyStream = 8,
    /// The verify reply (T5.1a): `seq_id u32 LE · n_tok u32 LE · m u32 LE
    /// · n_tok×u32 per-position argmax`. Sent AFTER the tail's own rewind
    /// bookkeeping: on a partial accept the tail has already restored its
    /// pre-verify state, so the ordinary frame sequence resumes at the
    /// verify's base position with no extra round trip.
    VerifyReply = 9,
    /// Decode-batch hint (T2.4 micro-batching, OPT-IN since Bench 913):
    /// `count u32 LE`. The head says the NEXT `count` frames are one-token
    /// decode steps of DISTINCT streams, so the tail submits every chain
    /// BEFORE any readback — the shared queue runs the chains back to
    /// back with no inter-chain idle. A co-located (loopback) win only:
    /// on a 2-box pipeline the wave forfeits the head⇄tail GPU overlap
    /// that is the s=2 bubble fill (−52% measured), so the serving
    /// default is the incumbent cadence and this frame is never sent.
    /// Solo and prefill passes never send it: those frames process
    /// exactly as before, byte-identical (G3 at s=1 by construction).
    BatchN = 10,
}

impl FrameKind {
    fn from_u8(v: u8) -> io::Result<Self> {
        match v {
            0 => Ok(Self::Handshake),
            1 => Ok(Self::Activation),
            2 => Ok(Self::Token),
            3 => Ok(Self::Shutdown),
            4 => Ok(Self::Prefill),
            5 => Ok(Self::Begin),
            6 => Ok(Self::BeginStream),
            7 => Ok(Self::TokenStream),
            8 => Ok(Self::VerifyStream),
            9 => Ok(Self::VerifyReply),
            10 => Ok(Self::BatchN),
            _ => Err(invalid(format!("unknown frame kind {v}"))),
        }
    }
}

/// Upper bound on one frame body — a prefill chunk of a 27B hidden is
/// ~MBs; anything near this is a malformed or hostile peer.
pub const MAX_BODY: usize = 256 << 20;

/// Frame header on the wire: body_len u32 + kind u8 + counter u64.
pub const HDR_LEN: usize = 4 + 1 + 8;
/// Trailing BLAKE3 MAC bytes.
pub const MAC_LEN: usize = 32;

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// A bidirectional, ordered, authenticated frame pipe between two stages.
pub trait Transport {
    fn send_frame(&mut self, kind: FrameKind, payload: &[u8]) -> io::Result<()>;
    /// Receive the next frame into `out` (cleared, capacity reused).
    fn recv_frame_into(&mut self, out: &mut Vec<u8>) -> io::Result<FrameKind>;
}

/// std TCP transport (`TCP_NODELAY`, vectored writes, no allocation in
/// steady state beyond the caller's reused buffer).
pub struct TcpTransport {
    stream: TcpStream,
    key: PeerKey,
    send_dir: Direction,
    send_ctr: u64,
    recv_ctr: u64,
}

impl TcpTransport {
    /// `send_dir` = the direction THIS side pushes (head: Downstream).
    pub fn new(stream: TcpStream, key: PeerKey, send_dir: Direction) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        Ok(Self {
            stream,
            key,
            send_dir,
            send_ctr: 0,
            recv_ctr: 0,
        })
    }

    fn mac(&self, dir: Direction, kind: u8, ctr: u64, payload: &[u8]) -> blake3::Hash {
        let mut h = blake3::Hasher::new_keyed(&self.key.0);
        h.update(&[dir as u8, kind]);
        h.update(&ctr.to_le_bytes());
        h.update(payload);
        h.finalize()
    }
}

impl Transport for TcpTransport {
    fn send_frame(&mut self, kind: FrameKind, payload: &[u8]) -> io::Result<()> {
        let body = 1 + 8 + payload.len();
        if body > MAX_BODY {
            return Err(invalid(format!("frame body {body} > MAX_BODY")));
        }
        let ctr = self.send_ctr;
        let mac = self.mac(self.send_dir, kind as u8, ctr, payload);
        let mut hdr = [0u8; HDR_LEN];
        hdr[..4].copy_from_slice(&(body as u32).to_le_bytes());
        hdr[4] = kind as u8;
        hdr[5..].copy_from_slice(&ctr.to_le_bytes());
        let mut bufs = [
            IoSlice::new(&hdr),
            IoSlice::new(payload),
            IoSlice::new(mac.as_bytes()),
        ];
        write_all_vectored(&mut self.stream, &mut bufs)?;
        self.send_ctr += 1;
        Ok(())
    }

    fn recv_frame_into(&mut self, out: &mut Vec<u8>) -> io::Result<FrameKind> {
        let mut hdr = [0u8; HDR_LEN];
        self.stream.read_exact(&mut hdr)?;
        let body = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize;
        if !(9..=MAX_BODY).contains(&body) {
            return Err(invalid(format!("frame body length {body} out of range")));
        }
        let kind_byte = hdr[4];
        let ctr = u64::from_le_bytes(hdr[5..].try_into().unwrap());
        out.clear();
        out.resize(body - 9, 0);
        self.stream.read_exact(out)?;
        let mut mac = [0u8; MAC_LEN];
        self.stream.read_exact(&mut mac)?;
        // MAC first: never interpret an unauthenticated byte (kind, counter).
        let want = self.mac(self.send_dir.flip(), kind_byte, ctr, out);
        if want != blake3::Hash::from(mac) {
            return Err(invalid(
                "frame MAC mismatch — refusing unauthenticated frame".into(),
            ));
        }
        if ctr != self.recv_ctr {
            return Err(invalid(format!(
                "frame counter {ctr} != expected {} — replay/reorder refused",
                self.recv_ctr
            )));
        }
        self.recv_ctr += 1;
        FrameKind::from_u8(kind_byte)
    }
}

fn write_all_vectored(w: &mut impl Write, mut bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
    IoSlice::advance_slices(&mut bufs, 0);
    while !bufs.is_empty() {
        match w.write_vectored(bufs) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => IoSlice::advance_slices(&mut bufs, n),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
