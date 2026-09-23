//! riir-infer-gpu build script (Issue 726 T3).
//!
//! macOS + `ane_prefill` feature only: compiles the ANE ObjC bridge
//! (`objc/ane_prefill_bridge.m`, ported verbatim from riir-poc's proven
//! T0 substrate) into a dylib under OUT_DIR, which `ane_prefill::bridge`
//! dlopens at runtime. Non-macOS or feature-off: no-op (the bridge module
//! is cfg-gated out and no toolchain cost is paid).
//!
//! Direct `xcrun clang` invocation (no `cc` crate) — mirrors riir-poc's
//! build.rs, which needs the Apple SDK sysroot for the ObjC + framework
//! compile.

fn main() {
    println!("cargo:rerun-if-changed=objc/ane_prefill_bridge.m");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    if std::env::var("CARGO_FEATURE_ANE_PREFILL").is_err() {
        return;
    }
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let sdk = std::process::Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .expect("xcrun");
    if !sdk.status.success() {
        panic!("xcrun --show-sdk-path failed (no Apple SDK?)");
    }
    let sdk = String::from_utf8_lossy(&sdk.stdout).trim().to_string();
    let dylib = format!("{out}/libane_prefill_bridge.dylib");
    let status = std::process::Command::new("xcrun")
        .args([
            "clang",
            "-O2",
            "-fobjc-arc",
            "-dynamiclib",
            "-isysroot",
            &sdk,
            "-framework",
            "Foundation",
            "-framework",
            "IOSurface",
            "-framework",
            "Metal",
            "-ldl",
            "-o",
            &dylib,
            "objc/ane_prefill_bridge.m",
        ])
        .status()
        .expect("clang");
    if !status.success() {
        panic!("ane_prefill_bridge.m compile failed ({status})");
    }
}
