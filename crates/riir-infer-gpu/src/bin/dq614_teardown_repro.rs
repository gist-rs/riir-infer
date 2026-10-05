//! `dq614_teardown_repro` — Issue 031 T-root-cause: the Windows CUDA
//! teardown-hang repro harness.
//!
//! The v1 matrix (2026-10-01) FATALed and then hung >10 min inside
//! `std::process::exit` with a live cudarc context (RAM climbing ~7 MB/s,
//! killed by hand). `hard_exit` (TerminateProcess-on-self) landed the same
//! day as the defensive fix; this bin A/Bs the exit paths so a recurrence
//! can be isolated fast:
//!
//! | arm | context state | exit path | expected under H1 (driver detach-time
//!   cleanup of a live context) |
//! |---|---|---|---|
//! | `plain`  | live (healthy) | `std::process::exit(1)` | HANG |
//! | `graceful` | dropped + primary-ctx released + reset | `std::process::exit(0)` | clean exit |
//! | `hard`   | live | `TerminateProcess(self)` | clean exit (the landed defense) |
//!
//! **MEASURED 2026-10-05: the hang did NOT reproduce — 6 arms, all clean,
//! including the true engine-state shape** (7.2 GB Bonsai weights resident,
//! cudarc + CubeCL both live, real base cells launched, forced FATAL via
//! `DQ614_FORCE_FATAL=1` + `DQ614_EXIT_PLAIN=1` on `dq_phase_matrix` itself:
//! clean exit at 215 s). Arms run: control no-CUDA · plain+healthy 256 MB
//! ctx · plain+sticky ctx (`CUDA_ERROR_ILLEGAL_ADDRESS` confirmed sticky) ·
//! plain+8 GB/64 iters · the engine-state plain-exit. Remaining hypotheses
//! for the 10-01 event: driver/OS state changed since, or the hang needed
//! the v1 context's ~6-hour wear (full 9-cell matrix, ~1e5 launches, 16K
//! KV). `hard_exit` + the `dq614_watchdog.ps1` FATAL-watcher remain the
//! defense; this instrument stays for any future occurrence. Full record:
//! HISTORY.md 2026-10-05 (Issue 031, closed + removed).
//!
//! `--sticky` first corrupts the context with an illegal-address kernel
//! (the sharpened hypothesis 3: the v1 FATAL followed a failed gate, i.e. a
//! possibly error-state context). `--no-cuda` runs the control (no CUDA at
//! all — if THAT hangs, the hang is a foreign CRT atexit handler, H3).
//!
//! Footprint knobs approximate the engine's context (`--alloc-mb`,
//! `--iters`) — the v1 context held the 27B weights, streams, events and
//! hours of launches; a minimal context may not reproduce, and a NULL here
//! is a real finding (footprint-dependent), never read as H1 refuted.
//!
//! Release-only measurement (the dev profile must never execute teardown
//! experiments). Machine-local runner (logs in `E:/git/_sync/dq614_repro/`):
//! `E:/git/riir-refine/.scratch/dq614_repro/run_arm.ps1` (grace + RAM
//! sampling + kill by PID).

#![cfg(all(
    feature = "ternary_gemv_cuda_raw",
    not(target_os = "macos"),
))]
// Dev profile: the refusing main keeps teardown experiments release-only
// (the same shape as `dq_phase_matrix`).
#![cfg_attr(debug_assertions, allow(dead_code))]

use std::io::Write as _;
use std::sync::OnceLock;
use std::time::Instant;

use cudarc::driver::safe::{CudaContext, LaunchConfig};
use cudarc::driver::{PushKernelArg, sys};

const KERNEL_SRC: &str = r#"
extern "C" __global__ void dq_fill(float* p, long long n, float v) {
    long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i < n) p[i] = v;
}
extern "C" __global__ void dq_oob(float* p) {
    // ~16 GB past any allocation — illegal address, corrupts the context.
    if (threadIdx.x == 0 && blockIdx.x == 0) p[4000000000LL] = 1.0f;
}
"#;

fn start() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
}

fn mark(msg: &str) {
    let t = start().elapsed().as_millis() as f64 / 1000.0;
    println!("[repro +{t:.2}s] {msg}");
    let _ = std::io::stdout().flush();
}

/// Same escape as `dq_phase_matrix::hard_exit` (the landed defense).
fn hard_exit(code: i32) -> ! {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            code as u32,
        );
    }
    std::process::exit(code)
}

struct Args {
    exit_mode: String, // plain | graceful | hard
    sticky: bool,
    no_cuda: bool,
    alloc_mb: usize,
    iters: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        exit_mode: "plain".into(),
        sticky: false,
        no_cuda: false,
        alloc_mb: 256,
        iters: 8,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--exit" => a.exit_mode = it.next().unwrap_or_else(|| "plain".into()),
            "--sticky" => a.sticky = true,
            "--no-cuda" => a.no_cuda = true,
            "--alloc-mb" => {
                a.alloc_mb = it.next().and_then(|v| v.parse().ok()).unwrap_or(256)
            }
            "--iters" => a.iters = it.next().and_then(|v| v.parse().ok()).unwrap_or(8),
            other => {
                eprintln!("[repro] unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    a
}

/// The v1 error-path exit (the hang site, by elimination).
fn exit_plain(code: i32) -> ! {
    mark("MARK pre-exit (plain std::process::exit)");
    std::process::exit(code)
}

fn exit_graceful(dev: sys::CUdevice, refs: &mut Vec<&'static str>) -> ! {
    // Drop order matters: slices → functions → module → stream → context.
    // Everything after the slices is implicit in the caller's frame; here we
    // only record what was already dropped, then force the context release
    // and the primary-ctx reset (the upstream-sanctioned graceful form —
    // NVIDIA forums #49680: give the driver a well-defined context state).
    for name in refs.drain(..) {
        mark(&format!("dropped {name}"));
    }
    mark("MARK context refs dropped (release ran via Drop)");
    unsafe {
        let rc = sys::cuDevicePrimaryCtxReset_v2(dev);
        mark(&format!("cuDevicePrimaryCtxReset_v2 -> {rc:?}"));
    }
    mark("MARK pre-exit (graceful std::process::exit(0))");
    std::process::exit(0)
}

fn run() -> Result<(), String> {
    let args = parse_args();
    mark(&format!(
        "arm: exit={} sticky={} no_cuda={} alloc_mb={} iters={}",
        args.exit_mode, args.sticky, args.no_cuda, args.alloc_mb, args.iters
    ));

    if args.no_cuda {
        mark("control: no CUDA ever touched");
        return Err("__control__".into()); // routed to the selected exit below
    }

    let ctx = CudaContext::new(0).map_err(|e| format!("context: {e}"))?;
    let dev = ctx.cu_device();
    mark(&format!("context ok (device {dev:#x}, primary)"));

    let stream = ctx.new_stream().map_err(|e| format!("stream: {e}"))?;
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        KERNEL_SRC,
        cudarc::nvrtc::CompileOptions {
            arch: Some("sm_89"),
            ..Default::default()
        },
    )
    .map_err(|e| format!("nvrtc: {e}"))?;
    let module = ctx.load_module(ptx).map_err(|e| format!("module: {e}"))?;
    let fill = module
        .load_function("dq_fill")
        .map_err(|e| format!("load dq_fill: {e}"))?;
    let oob = module
        .load_function("dq_oob")
        .map_err(|e| format!("load dq_oob: {e}"))?;
    mark("module + kernels ok");

    let n: usize = args.alloc_mb * 1024 * 1024 / 4; // f32 elements
    let mut buf = unsafe { stream.alloc::<f32>(n) }.map_err(|e| format!("alloc: {e}"))?;
    mark(&format!("allocated {} MB device memory", args.alloc_mb));

    let cfg = LaunchConfig {
        grid_dim: ((n as u64).div_ceil(256).min(u32::MAX as u64) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i64 = n as i64;
    for i in 0..args.iters {
        let v = i as f32 + 1.0;
        unsafe {
            stream
                .launch_builder(&fill)
                .arg(&mut buf)
                .arg(&n_i64)
                .arg(&v)
                .launch(cfg)
                .map_err(|e| format!("launch fill #{i}: {e}"))?;
        }
        stream.synchronize().map_err(|e| format!("sync #{i}: {e}"))?;
    }
    // Verify the last fill landed (the context demonstrably works).
    let host = stream
        .clone_dtoh(&buf)
        .map_err(|e| format!("dtoh: {e}"))?;
    let want = args.iters as f32;
    if host[0] != want || host[n - 1] != want {
        return Err(format!("fill verify: got {} / {}", host[0], host[n - 1]));
    }
    mark(&format!("{} fill+sync cycles ok, verify passed", args.iters));

    if args.sticky {
        unsafe {
            stream
                .launch_builder(&oob)
                .arg(&mut buf)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("launch oob: {e}"))?;
        }
        let sticky = stream.synchronize();
        mark(&format!("sticky sync result: {sticky:?}"));
        // A second op confirms the context is corrupt (sticky, not one-shot).
        let again = stream.synchronize();
        mark(&format!("post-sticky sync result: {again:?}"));
    }

    // Explicit drops in reverse construction order; the refs list is for the
    // graceful arm's MARK trail.
    let mut refs: Vec<&'static str> = Vec::new();
    drop(fill);
    refs.push("functions");
    drop(oob);
    drop(module);
    refs.push("module");
    drop(stream);
    refs.push("stream");
    drop(buf);
    refs.push("device slice");
    if args.exit_mode == "graceful" {
        // Drop the context itself, then reset the primary ctx state.
        drop(ctx);
        refs.push("context (primary release)");
        exit_graceful(dev, &mut refs);
    }
    // plain / hard: the context stays LIVE through exit (the v1 shape).
    let _ = ctx;
    let _ = refs;
    match args.exit_mode.as_str() {
        "plain" => exit_plain(1),
        "hard" => {
            mark("MARK pre-exit (hard TerminateProcess)");
            hard_exit(1)
        }
        other => Err(format!("unknown --exit {other}")),
    }
}

#[cfg(debug_assertions)]
fn main() {
    eprintln!(
        "[repro] REFUSED: release-only teardown experiment — rebuild with \
         `cargo build --release -p riir-infer-gpu --features ternary_gemv_cuda_raw --bin dq614_teardown_repro`"
    );
    std::process::exit(2);
}

#[cfg(not(debug_assertions))]
fn main() {
    if let Err(e) = run() {
        if e == "__control__" {
            // The --no-cuda control still exercises the SELECTED exit path.
            let mode = parse_args().exit_mode;
            match mode.as_str() {
                "graceful" => {
                    mark("MARK pre-exit (graceful, no CUDA)");
                    std::process::exit(0)
                }
                "hard" => {
                    mark("MARK pre-exit (hard, no CUDA)");
                    hard_exit(1)
                }
                _ => exit_plain(1),
            }
        }
        eprintln!("[repro] FATAL: {e}");
        hard_exit(1);
    }
}
