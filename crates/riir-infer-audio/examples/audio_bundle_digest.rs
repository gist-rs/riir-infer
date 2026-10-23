//! The pin-minting instrument: print the blake3-dir-v1 digest +
//! file/byte walls for a downloaded audio bundle — the fetch script's
//! companion; the in-code pin in `src/vad.rs` is minted from THIS
//! output, never hand-computed.
//!
//! ```text
//! cargo run -p riir-infer-audio --features vad-ane \
//!   --example audio_bundle_digest -- \
//!   ~/.cache/riir-infer/audio/silero-vad-unified-256ms-v6.2.1.mlmodelc
//! ```

#[cfg(all(feature = "vad-ane", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: audio_bundle_digest <bundle-dir>");
        std::process::exit(2);
    };
    let (digest, files, bytes) = riir_infer_ane::blake3_dir_v1(std::path::Path::new(&path))?;
    println!("digest: {digest}");
    println!("files:  {files}");
    println!("bytes:  {bytes}");
    Ok(())
}

#[cfg(not(all(feature = "vad-ane", target_os = "macos")))]
fn main() {
    eprintln!("audio_bundle_digest: the vad-ane lane is macOS-only — nothing to mint");
    std::process::exit(3);
}
