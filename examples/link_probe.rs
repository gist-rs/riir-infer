//! Plan 623 T0.2 — link microbench for the multi-host lane (Proposal 042).
//!
//! Std-only TCP echo probe over the amo<->bmo TB bridge (192.168.10.x) or the
//! tailnet. Measures USER-SPACE round-trip latency per frame size (the transport
//! lane's real cost incl. syscalls + copies, not raw ICMP) and one-way push
//! bandwidth (the pipeline's activation-frame direction).
//!
//! Server: `link_probe --server 0.0.0.0:46230`
//! Client: `link_probe --addr 192.168.10.2:46230`
//!
//! Frame protocol (matches the probe only, NOT ActivationFrame):
//! u32 BE length prefix + payload; server echoes. Bandwidth: client pushes
//! 64 MiB, server acks 8 bytes after the full read.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

const PORT: u16 = 46230;
const BW_BYTES: usize = 64 * 1024 * 1024;
const BW_ROUNDS: usize = 5;
/// High bit of the length word = ONE-WAY push (server acks 8 bytes, never
/// echoes). Without it the server echoes the 64 MiB back into a client that is
/// not reading, and the byte stream desyncs (the first ack read consumes the
/// echo's first bytes as garbage).

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--server") => {
            let bind: String = args.get(2).cloned().unwrap_or_else(|| "0.0.0.0".to_string());
            let bind = if bind.contains(':') {
                bind
            } else {
                format!("{bind}:{PORT}")
            };
            run_server(&bind);
        }
        Some("--addr") => {
            let addr: String = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "192.168.10.2".to_string());
            run_client(&format!("{addr}:{PORT}"));
        }
        _ => {
            eprintln!(
                "usage: link_probe --server [bind] | link_probe --addr <ip>  (port {PORT})"
            );
            std::process::exit(2);
        }
    }
}

fn read_exact_n(stream: &mut TcpStream, buf: &mut [u8]) -> std::io::Result<()> {
    stream.read_exact(buf)
}

fn recv_frame(stream: &mut TcpStream, scratch: &mut Vec<u8>) -> std::io::Result<usize> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    scratch.clear();
    scratch.resize(len, 0);
    read_exact_n(stream, scratch)?;
    Ok(len)
}

const PUSH_FLAG: u32 = 0x8000_0000;
const LEN_MASK: u32 = 0x7FFF_FFFF;

fn run_server(bind: &str) {
    let listener = TcpListener::bind(bind).unwrap_or_else(|e| {
        eprintln!("bind {bind} failed: {e}");
        std::process::exit(1);
    });
    println!("[link_probe] server listening on {bind}");
    for conn in listener.incoming() {
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept failed: {e}");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        println!("[link_probe] connection from {peer}");
        let mut scratch: Vec<u8> = Vec::with_capacity(BW_BYTES);
        loop {
            let mut len_buf = [0u8; 4];
            if stream.read_exact(&mut len_buf).is_err() {
                println!("[link_probe] peer closed (len word read)");
                break;
            }
            let word = u32::from_be_bytes(len_buf);
            let len = (word & LEN_MASK) as usize;
            scratch.clear();
            scratch.resize(len, 0);
            if read_exact_n(&mut stream, &mut scratch).is_err() {
                println!("[link_probe] payload read FAILED at len={len} (push={})", word & PUSH_FLAG != 0);
                break;
            }
            if word & PUSH_FLAG != 0 {
                if stream.write_all(&[0u8; 8]).is_err() {
                    println!("[link_probe] ack write failed");
                    break;
                }
            } else {
                if stream.write_all(&len_buf).is_err() || stream.write_all(&scratch).is_err() {
                    println!("[link_probe] echo write failed");
                    break;
                }
            }
        }
        println!("[link_probe] connection closed ({peer})");
    }
}

fn echo_rtt(stream: &mut TcpStream, size: usize, iters: usize) -> Vec<f64> {
    let payload = vec![0u8; size];
    let mut scratch: Vec<u8> = Vec::with_capacity(size);
    let mut rtts_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let len_buf = (size as u32).to_be_bytes();
        let t0 = Instant::now();
        if stream.write_all(&len_buf).is_err() || stream.write_all(&payload).is_err() {
            eprintln!("write failed");
            std::process::exit(1);
        }
        let n = match recv_frame(stream, &mut scratch) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("echo read failed: {e}");
                std::process::exit(1);
            }
        };
        assert_eq!(n, size, "echo size mismatch");
        rtts_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    rtts_us
}

fn push_bw(stream: &mut TcpStream) -> f64 {
    let payload = vec![0xABu8; BW_BYTES];
    let mut ack = [0u8; 8];
    let mut best = 0.0f64;
    for round in 0..BW_ROUNDS {
        let t0 = Instant::now();
        let len_buf = (BW_BYTES as u32 | PUSH_FLAG).to_be_bytes();
        if let Err(e) = stream.write_all(&len_buf) {
            eprintln!("bw len write failed (round {round}): {e}");
            std::process::exit(1);
        }
        if let Err(e) = stream.write_all(&payload) {
            eprintln!("bw payload write failed (round {round}): {e}");
            std::process::exit(1);
        }
        if let Err(e) = stream.read_exact(&mut ack) {
            eprintln!("bw ack read failed (round {round}): {e}");
            std::process::exit(1);
        }
        let secs = t0.elapsed().as_secs_f64();
        let gbs = BW_BYTES as f64 / secs / 1e9;
        if gbs > best {
            best = gbs;
        }
        println!("  push round: {gbs:.3} GB/s ({secs:.4} s for 64 MiB)");
    }
    best
}

fn report(name: &str, mut rtts: Vec<f64>) {
    rtts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = rtts.len();
    let median = rtts[n / 2];
    let p90 = rtts[(n * 9) / 10];
    // Tail support disclosed beside the percentile (the n-1-max trap: p90 here
    // has n/10 samples at or above it, never misread as a max).
    println!(
        "{name:>8}: n={n}  min={:.1}us  median={:.1}us  p90={:.1}us (support {})  max={:.1}us",
        rtts[0],
        median,
        p90,
        n - (n * 9) / 10,
        rtts[n - 1]
    );
}

fn run_client(addr: &str) {
    let mut stream = TcpStream::connect(addr).unwrap_or_else(|e| {
        eprintln!("connect {addr} failed: {e}");
        std::process::exit(1);
    });
    stream.set_nodelay(true).expect("nodelay");
    println!("[link_probe] connected to {addr} (TCP_NODELAY on)");

    // Warmup.
    let _ = echo_rtt(&mut stream, 4, 50);

    let sizes: [(usize, usize); 4] = [(4, 2000), (16 * 1024, 1000), (1024 * 1024, 200), (64 * 1024 * 1024, 10)];
    for (size, iters) in sizes {
        let label = if size >= 1024 * 1024 {
            format!("{}MiB", size / (1024 * 1024))
        } else if size >= 1024 {
            format!("{}KiB", size / 1024)
        } else {
            format!("{size}B")
        };
        report(&label, echo_rtt(&mut stream, size, iters));
    }

    println!("bandwidth (64 MiB push, best of {BW_ROUNDS}):");
    let gbs = push_bw(&mut stream);
    println!("BEST_PUSH_GBPS {gbs:.3}");
}
