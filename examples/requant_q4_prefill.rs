//! Issue 028 T4 S1 — the Q4_K prefill-pack requantizer (.plans/618); plan
//! 618 S3 adds the `--target q6k` arm (the matched-storage single-checkpoint
//! control pack, same identity-block-table + byte-copy policy).
//!
//! Reads the league PQ2_0 pack, requantizes every block projection tensor
//! to the target k-quant format, byte-copies everything else (globals, a/b
//! gate projections, norms/conv/ssm_a/ssm_dt — the escape set rides
//! bit-shared by construction), and emits the pack through the landed
//! collapsed-GGUF writer with an IDENTITY block table (all 64 layers, no
//! reduction).
//!
//! The science premise this tool encodes: the bonsai is TRAINED TERNARY, so
//! `dequant(PQ2_0)` is exact up to f16 group scales and the requant carries
//! ONLY ε(target format). The run REFUSES if a sampled tensor fails the
//! ternary-exactness check (the repack itself also refuses fourth-state
//! codes), so the premise is checked on the artifact, never assumed.
//!
//! Usage (the T4 artifact):
//!   cargo run --release --features twt_collapse --example requant_q4_prefill -- \
//!     --parent ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --out ../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf
//!
//! The S3 matched-single control (all-projections Q6_K single at ~20.5 GB —
//! the dual pair's 21.1 GB within ~3%):
//!   cargo run --release --features twt_collapse --example requant_q4_prefill -- \
//!     --target q6k \
//!     --out ../riir-train/data/Ternary-Bonsai-2-27B-Q6_K.sg.gguf
//!
//! Gate flags (off = the full run): `--sample-only` runs the exactness +
//! one-tensor requant gates and exits before the emit (a fast premise
//! check); `--verify-only` re-opens an existing pack and runs the read-back
//! verification only.

use std::collections::BTreeMap;
use std::io::BufWriter;
use std::time::Instant;

use riir_infer_core::gguf_loader::{GgufFile, GgmlType, GgufValue};
use riir_infer_core::quant::q2_0::repack_q2_0_to_ternary_group;
use riir_infer_core::quant::q4k::QK_K;
use riir_infer_core::twt::collapse_writer::{
    CollapseSpec, LayerSource, LAYER_TYPES_LEGEND, TensorOut, emit_collapsed_gguf,
    twt_layer_types_value,
};
use riir_infer_core::types::DeltaNetLayerType;

/// The requant target format (plan 618 S3 axis). `Q4K` keeps the landed S1
/// behavior byte-identical (the default); `Q6K` builds the matched-storage
/// single-checkpoint control.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    Q4K,
    Q6K,
}

impl Target {
    fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "q4k" | "Q4K" | "q4_k" => Ok(Self::Q4K),
            "q6k" | "Q6K" | "q6_k" => Ok(Self::Q6K),
            other => anyhow::bail!("unknown --target '{other}' (q4k | q6k)"),
        }
    }

    fn ggml_type(self) -> GgmlType {
        match self {
            Self::Q4K => GgmlType::Q4_K,
            Self::Q6K => GgmlType::Q6_K,
        }
    }

    fn bytes_per_block(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q6K => 210,
        }
    }

    /// Quantize one row (`row.len() % 256 == 0`) and append the raw block
    /// bytes to `out`. (A typed scratch per row — the emit is one-off
    /// artifact tooling; the round trip keeps `bytemuck` on the TYPED vec,
    /// never an alignment bet on the u8 buffer.)
    fn quantize_row_into(self, row: &[f32], out: &mut Vec<u8>) {
        match self {
            Self::Q4K => {
                use riir_infer_core::quant::q4k::{BlockQ4K, quantize_row_q4_k};
                let nb = row.len() / QK_K;
                let mut blocks = vec![<BlockQ4K as bytemuck::Zeroable>::zeroed(); nb];
                quantize_row_q4_k(row, &mut blocks);
                out.extend_from_slice(bytemuck::cast_slice(&blocks));
            }
            Self::Q6K => {
                use riir_infer_core::quant::q6k::{BlockQ6K, quantize_row_q6_k};
                let nb = row.len() / QK_K;
                let mut blocks = vec![<BlockQ6K as bytemuck::Zeroable>::zeroed(); nb];
                quantize_row_q6_k(row, &mut blocks);
                out.extend_from_slice(bytemuck::cast_slice(&blocks));
            }
        }
    }

    /// Dequantize row `r` of a raw block payload into `out` (the verify
    /// read-back; the projection's own row-slice is handed in).
    fn dequant_row(self, row_blocks_bytes: &[u8], out: &mut [f32]) {
        match self {
            Self::Q4K => {
                let blocks: &[riir_infer_core::quant::q4k::BlockQ4K] =
                    bytemuck::cast_slice(row_blocks_bytes);
                riir_infer_core::quant::q4k::dequantize_row_q4_k(blocks, out);
            }
            Self::Q6K => {
                let blocks: &[riir_infer_core::quant::q6k::BlockQ6K] =
                    bytemuck::cast_slice(row_blocks_bytes);
                riir_infer_core::quant::q6k::dequantize_row_q6_k(blocks, out);
            }
        }
    }

    /// The per-sub-block analytic read-back bound over one 32-element
    /// segment: q4 = (amax−amin)/15 doubled for f16 rounding slack (the
    /// landed S1 bound); q6 = the code step (|d·sc| ≤ 1.1·amax/31 — the
    /// encoder's ±10% scale search) at half a code plus saturation slack,
    /// rounded up to 10% of the segment amax.
    fn readback_bound(self, seg: &[f32]) -> f32 {
        match self {
            Self::Q4K => {
                let lo = seg.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = seg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                2.0 * (hi - lo).abs() / 15.0 + 1e-6
            }
            Self::Q6K => {
                let amax = seg.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                0.1 * amax + 1e-6
            }
        }
    }

    fn readback_subblock(self) -> usize {
        match self {
            Self::Q4K => 32,
            Self::Q6K => 16,
        }
    }

    fn format_note(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K block projections over PQ2_0 (issue 028 T4 S1, plan 618)",
            Self::Q6K => {
                "Q6_K block projections over PQ2_0 (plan 618 S3 matched-single control; \
                 llama.cpp quantize_row_q6_K_ref port)"
            }
        }
    }
}

/// Projection storage suffixes requantized to Q4_K — exactly the tensors
/// `load_ternary_proj` reads (attn q/k/v/o for attention layers; the rest
/// are DeltaNet-layer projections). Anything NOT in this list and NOT a
/// dense copy class refuses loud: a new projection arriving in a future
/// pack must be classified deliberately, never silently copied as ternary.
const REQUANT_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_out.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Byte-copy classes: the escape set (ssm_alpha/ssm_beta BF16, ssm_a,
/// ssm_dt.bias F32) + the dense fields (norms, conv1d). The copy IS the
/// T2 escape law — both container copies hold these bit-identically.
fn is_copy_suffix(suffix: &str) -> bool {
    suffix.ends_with("norm.weight")
        || suffix == "ssm_a"
        || suffix == "ssm_dt.bias"
        || suffix == "ssm_conv1d.weight"
        || suffix == "ssm_alpha.weight"
        || suffix == "ssm_beta.weight"
}

struct Args {
    parent: String,
    out: String,
    target: Target,
    sample_only: bool,
    verify_only: bool,
    hash_only: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        parent: "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_owned(),
        out: "../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf".to_owned(),
        target: Target::Q4K,
        sample_only: false,
        verify_only: false,
        hash_only: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--parent" => a.parent = it.next().expect("--parent <path>"),
            "--out" => a.out = it.next().expect("--out <path>"),
            "--target" => {
                a.target = Target::parse(&it.next().expect("--target <q4k|q6k>")).expect("--target")
            }
            "--sample-only" => a.sample_only = true,
            "--verify-only" => a.verify_only = true,
            "--hash-only" => a.hash_only = true,
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

/// Dequantize one Q2_0 tensor to dense f32 via the ternary bridge (the
/// repack refuses fourth-state codes — structural ternary-exactness).
fn tensor_dense(parent: &GgufFile, name: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let info = parent
        .tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found"))?;
    let cols = info.shape[0]; // ne[0] = innermost = row length
    let rows: usize = info.shape[1..].iter().product();
    anyhow::ensure!(
        info.ggml_type == GgmlType::Q2_0,
        "tensor {name} is {:?} — the requant lane reads Q2_0 projections only",
        info.ggml_type
    );
    let raw = parent
        .tensor_slice(name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} has no bytes"))?;
    let blocks: Vec<riir_infer_core::quant::q2_0::BlockQ2_0> =
        bytemuck::cast_slice(raw).to_vec();
    let tg = repack_q2_0_to_ternary_group(&blocks, rows, cols)?;
    let dense =
        riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(&tg);
    Ok((dense, rows, cols))
}

/// The ternary-exactness premise, checked on a REAL tensor: the Q2_0 wire
/// repack must succeed (code 3 refuses — `UnsupportedFourthState`) AND the
/// bridge's reconstruction (`trit × f16 group scale`) must agree with the
/// RAW `dequantize_row_q2_0` decode on sampled values — catching any
/// wire-vs-bridge divergence, not just illegal codes.
fn check_ternary_exactness(
    parent: &GgufFile,
    name: &str,
) -> anyhow::Result<()> {
    let info = parent
        .tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found"))?;
    let cols = info.shape[0];
    let rows: usize = info.shape[1..].iter().product();
    let blocks: &[riir_infer_core::quant::q2_0::BlockQ2_0] =
        parent.q2_0_tensor_blocks(name)?;
    // Arm 1: the bridge repack — refuses fourth-state codes (the trained-
    // ternary premise's structural half).
    let tg = repack_q2_0_to_ternary_group(blocks, rows, cols)?;
    // Arm 2: value-level agreement on sampled blocks — the raw wire decode
    // (q2_0's own dequantize) vs the bridge's `trit × scale`.
    let group = riir_infer_core::quant::q2_0::Q2_0_BLOCK_SIZE;
    let sample_blocks = [0usize, blocks.len() / 2, blocks.len() - 1];
    let mut scratch = vec![0f32; group];
    for &bi in &sample_blocks {
        riir_infer_core::quant::q2_0::dequantize_row_q2_0(&blocks[bi..bi + 1], &mut scratch);
        let row = bi / (cols / group);
        let g = bi % (cols / group);
        let scale = tg.group_scale[row * tg.groups_per_row + g].to_f32();
        for (j, &wire) in scratch.iter().enumerate() {
            let byte = blocks[bi].qs[j / 4];
            let code = (byte >> ((j % 4) * 2)) & 0x03;
            let trit: f32 = match code {
                0 => -1.0,
                1 => 0.0,
                2 => 1.0,
                _ => anyhow::bail!("code 3 at {name} block {bi}[{j}] — not trained ternary"),
            };
            let bridge = trit * scale;
            let err = (wire - bridge).abs();
            anyhow::ensure!(
                err <= 1e-6 * scale.abs().max(1.0),
                "wire-vs-bridge divergence at {name} block {bi}[{j}]: wire {wire} vs bridge {bridge}"
            );
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = parse_args();
    let parent_path = std::path::PathBuf::from(&args.parent);
    let parent = GgufFile::open(&parent_path)?;

    let arch = parent
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_owned();
    anyhow::ensure!(arch == "qwen35", "expected qwen35, got {arch}");
    let n_layer = parent.metadata_u64("qwen35.block_count").unwrap_or(0) as usize;
    anyhow::ensure!(n_layer > 0, "qwen35.block_count missing/zero");
    let interval = parent.metadata_u64("qwen35.full_attention_interval").unwrap_or(4) as usize;
    let nextn = parent.metadata_u64("qwen35.nextn_predict_layers").unwrap_or(0);
    anyhow::ensure!(nextn == 0, "nextn parents need their own plan (identity table is main-stack-only)");

    let layer_types: Vec<DeltaNetLayerType> = (0..n_layer)
        .map(|i| {
            if (i + 1).is_multiple_of(interval) {
                DeltaNetLayerType::Attention
            } else {
                DeltaNetLayerType::DeltaNet
            }
        })
        .collect();

    // ── classify every block tensor exactly once ──
    let mut unclassified: Vec<String> = Vec::new();
    let mut requant_tensors: Vec<String> = Vec::new();
    for info in &parent.tensor_infos {
        let Some(suffix) = info.name.strip_prefix("blk.0.") else {
            continue; // globals: byte-copied by the writer
        };
        if REQUANT_SUFFIXES.contains(&suffix) {
            anyhow::ensure!(
                info.ggml_type == GgmlType::Q2_0,
                "projection {} is {:?} — the requant lane expects Q2_0",
                info.name,
                info.ggml_type
            );
            if !requant_tensors.is_empty() {
                continue; // one inventory pass suffices (same set every block)
            }
            requant_tensors.push(suffix.to_owned());
        } else if !is_copy_suffix(suffix) {
            unclassified.push(suffix.to_owned());
        }
    }
    anyhow::ensure!(
        unclassified.is_empty(),
        "unclassified block tensors present: {unclassified:?} — extend the policy tables deliberately"
    );
    anyhow::ensure!(
        !requant_tensors.is_empty(),
        "no projection tensors matched the requant policy"
    );

    if args.verify_only {
        return verify_pack(&parent, &args.out, &requant_tensors, args.target);
    }

    if args.hash_only {
        let t_hash = Instant::now();
        let digest = file_blake3(std::path::Path::new(&args.out))?;
        let out_path = std::path::PathBuf::from(&args.out);
        let sidecar = out_path.with_extension("gguf.blake3");
        std::fs::write(&sidecar, format!("{digest}\n"))?;
        println!(
            "[requant] blake3 {digest} ({:.1}s); sidecar {}",
            t_hash.elapsed().as_secs_f32(),
            sidecar.display()
        );
        return Ok(());
    }

    // ── the science premise, checked on a real tensor BEFORE hours of work ──
    let t0 = Instant::now();
    check_ternary_exactness(&parent, "blk.0.ffn_down.weight")?;
    eprintln!(
        "[requant] ternary-exactness PASS on blk.0.ffn_down.weight ({:.1}s) — the requant premise holds",
        t0.elapsed().as_secs_f32()
    );

    // ── divisibility + a one-tensor requant smoke (rows must ÷ 256) ──
    let target = args.target;
    let bpb = target.bytes_per_block();
    let (dense, _rows, cols) = tensor_dense(&parent, "blk.0.ffn_down.weight")?;
    anyhow::ensure!(
        cols.is_multiple_of(QK_K),
        "row length {cols} not a multiple of QK_K {QK_K}"
    );
    let mut smoke = vec![0u8; (cols / QK_K) * bpb];
    // Row-major rows: quantize row 0 only as the smoke.
    target.quantize_row_into(&dense[..cols], &mut smoke);
    eprintln!(
        "[requant] requant smoke PASS ({target:?}): blk.0.ffn_down row 0 ({cols} elems → {} blocks)",
        smoke.len() / bpb
    );
    drop(dense);
    if args.sample_only {
        eprintln!("[requant] --sample-only: gates green, exiting before the emit");
        return Ok(());
    }

    // ── build the identity spec: every block Merged, requant or copy ──
    let mut blocks_spec: Vec<(usize, usize, LayerSource)> = Vec::with_capacity(n_layer);
    let mut total_requant_bytes = 0u64;
    let emit_start = Instant::now();
    for li in 0..n_layer {
        let prefix = format!("blk.{li}.");
        let mut map: BTreeMap<String, TensorOut> = BTreeMap::new();
        for info in &parent.tensor_infos {
            let Some(suffix) = info.name.strip_prefix(&prefix) else {
                continue;
            };
            let shape = info.shape.clone();
            if REQUANT_SUFFIXES.contains(&suffix) {
                let (dense, rows, cols) = tensor_dense(&parent, &info.name)?;
                debug_assert_eq!(rows * cols, dense.len());
                let mut out_bytes: Vec<u8> = Vec::with_capacity(rows * (cols / QK_K) * bpb);
                for r in 0..rows {
                    target.quantize_row_into(&dense[r * cols..(r + 1) * cols], &mut out_bytes);
                }
                let n_elems = rows * cols;
                anyhow::ensure!(
                    out_bytes.len() == target.ggml_type().tensor_bytes(n_elems),
                    "payload size mismatch for {}: {} vs {}",
                    info.name,
                    out_bytes.len(),
                    target.ggml_type().tensor_bytes(n_elems)
                );
                total_requant_bytes += out_bytes.len() as u64;
                map.insert(
                    suffix.to_owned(),
                    TensorOut {
                        ggml_type: target.ggml_type(),
                        shape,
                        data: out_bytes,
                    },
                );
            } else {
                let raw = parent
                    .tensor_slice(&info.name)
                    .ok_or_else(|| anyhow::anyhow!("tensor {} vanished", info.name))?;
                map.insert(
                    suffix.to_owned(),
                    TensorOut {
                        ggml_type: info.ggml_type,
                        shape,
                        data: raw.to_vec(),
                    },
                );
            }
        }
        if li % 8 == 0 || li + 1 == n_layer {
            eprintln!(
                "[requant] block {li}/{n_layer} staged ({:.1}s elapsed, requant bytes so far {:.2} GB)",
                emit_start.elapsed().as_secs_f32(),
                total_requant_bytes as f64 / 1e9
            );
        }
        blocks_spec.push((li, li + 1, LayerSource::Merged(map)));
    }

    // The block-count override MIRRORS the parent's variant type: the
    // container's geometry fingerprint is discriminant-tagged (U32 64 ≠
    // U64 64), so a re-widened value would refuse the decode/prefill PAIR
    // at T4 load time. Identity requant = identity metadata encoding.
    let parent_block_count = parent
        .metadata
        .get("qwen35.block_count")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("qwen35.block_count missing"))?;
    let spec = CollapseSpec {
        blocks: blocks_spec,
        metadata_overrides: vec![("qwen35.block_count".to_owned(), parent_block_count)],
        twt_meta: vec![
            (
                "twt.layer_types".to_owned(),
                twt_layer_types_value(&layer_types),
            ),
            (
                "twt.layer_types_legend".to_owned(),
                GgufValue::String(LAYER_TYPES_LEGEND.to_owned()),
            ),
            (
                "twt.parent_weights_blake3".to_owned(),
                GgufValue::String(parent_weights_blake3(&parent)),
            ),
            (
                "twt.requant.format".to_owned(),
                GgufValue::String(target.format_note().to_owned()),
            ),
            (
                "twt.requant.requantized_suffixes".to_owned(),
                GgufValue::Array(
                    REQUANT_SUFFIXES
                        .iter()
                        .map(|s| GgufValue::String((*s).to_owned()))
                        .collect(),
                ),
            ),
            (
                "twt.requant.requantized_bytes".to_owned(),
                GgufValue::U64(total_requant_bytes),
            ),
        ],
    };

    let out_path = std::path::PathBuf::from(&args.out);
    let out_file = std::fs::File::create(&out_path)?;
    let mut out = BufWriter::with_capacity(1 << 22, out_file);
    let t_emit = Instant::now();
    let stats = emit_collapsed_gguf(&parent, &spec, &mut out)?;
    use std::io::Write as _;
    out.flush().map_err(anyhow::Error::from)?;
    eprintln!(
        "[requant] emitted {} ({:.2} GB, {} tensors) in {:.1}s",
        out_path.display(),
        stats.bytes_written as f64 / 1e9,
        stats.n_tensors,
        t_emit.elapsed().as_secs_f32()
    );

    // ── post-emit verification + the artifact blake3 ──
    drop(out);
    let pack = GgufFile::open(&out_path)?;
    verify_pack_tensors(&parent, &pack, &requant_tensors, target)?;
    let t_hash = Instant::now();
    let digest = file_blake3(&out_path)?;
    let sidecar = out_path.with_extension("gguf.blake3");
    std::fs::write(&sidecar, format!("{digest}\n"))?;
    println!(
        "[requant] VERIFY PASS — geometry + sampled read-back green; blake3 {digest} ({:.1}s); sidecar {}",
        t_hash.elapsed().as_secs_f32(),
        sidecar.display()
    );
    Ok(())
}

/// `GgufValue` equality over the wire-numeric variants (the geometry keys
/// are all numeric/string — an exotic variant here refuses, never lies).
fn gguf_value_eq(a: &GgufValue, b: &GgufValue) -> bool {
    use GgufValue as V;
    match (a, b) {
        (V::U8(x), V::U8(y)) => x == y,
        (V::I8(x), V::I8(y)) => x == y,
        (V::U16(x), V::U16(y)) => x == y,
        (V::I16(x), V::I16(y)) => x == y,
        (V::U32(x), V::U32(y)) => x == y,
        (V::I32(x), V::I32(y)) => x == y,
        (V::U64(x), V::U64(y)) => x == y,
        (V::I64(x), V::I64(y)) => x == y,
        (V::F32(x), V::F32(y)) => x.to_bits() == y.to_bits(),
        (V::F64(x), V::F64(y)) => x.to_bits() == y.to_bits(),
        (V::Bool(x), V::Bool(y)) => x == y,
        (V::String(x), V::String(y)) => x == y,
        (V::Array(x), V::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| gguf_value_eq(p, q))
        }
        _ => false,
    }
}

fn parent_weights_blake3(parent: &GgufFile) -> String {
    let mut hasher = blake3::Hasher::new();
    for info in &parent.tensor_infos {
        hasher.update(parent.tensor_slice(&info.name).expect("tensor bytes"));
    }
    hasher.finalize().to_hex().to_string()
}

fn file_blake3(path: &std::path::Path) -> anyhow::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Post-emit gates: geometry fingerprint equality (the loader's own compat
/// plane) + sampled requant read-back within the target's analytic error
/// bound.
fn verify_pack(
    parent: &GgufFile,
    out: &str,
    requant_tensors: &[String],
    target: Target,
) -> anyhow::Result<()> {
    let pack = GgufFile::open(std::path::Path::new(out))?;
    verify_pack_tensors(parent, &pack, requant_tensors, target)
}

fn verify_pack_tensors(
    parent: &GgufFile,
    pack: &GgufFile,
    requant_suffixes: &[String],
    target: Target,
) -> anyhow::Result<()> {
    // Geometry: the metadata keys the loader's fingerprint plane reads.
    for (key, a) in &parent.metadata_order {
        if key.starts_with("qwen35.") || key == "general.architecture" {
            let b = pack.metadata.get(key);
            let same = b.is_some_and(|y| gguf_value_eq(a, y));
            anyhow::ensure!(same, "geometry drift at {key}: {a:?} vs {b:?}");
        }
    }
    // Copy classes: byte-identical payloads (the escape set law's substrate).
    // Requant suffixes at ANY block index are excluded — they legitimately
    // differ (Q2_0 → Q4_K) and are checked by the read-back gates below.
    let mut copies_checked = 0usize;
    for info in &parent.tensor_infos {
        let is_requant = info
            .name
            .split_once('.')
            .and_then(|(_, rest)| rest.split_once('.'))
            .map(|(_, suffix)| REQUANT_SUFFIXES.contains(&suffix))
            .unwrap_or(false);
        if is_requant {
            // Structural: every requant tensor in the pack must be the
            // target type at its parent's exact shape.
            let pi = pack
                .tensor_info(&info.name)
                .ok_or_else(|| anyhow::anyhow!("pack missing requant tensor {}", info.name))?;
            anyhow::ensure!(
                pi.ggml_type == target.ggml_type() && pi.shape == info.shape,
                "pack tensor {} type/shape drift: {:?} {:?} vs parent Q2_0 {:?}",
                info.name,
                pi.ggml_type,
                pi.shape,
                info.shape
            );
            continue;
        }
        let a = parent
            .tensor_slice(&info.name)
            .ok_or_else(|| anyhow::anyhow!("parent tensor {} vanished", info.name))?;
        let b = pack
            .tensor_slice(&info.name)
            .ok_or_else(|| anyhow::anyhow!("pack tensor {} missing", info.name))?;
        anyhow::ensure!(a == b, "copy-class tensor {} differs between parent and pack", info.name);
        copies_checked += 1;
    }
    eprintln!("[requant] copy-class byte-identity PASS ({copies_checked} tensors incl. globals + escape set)");

    // Requant read-back: one projection per class-suffix, sampled rows,
    // bounded by the target's analytic error (the bound f16-scale rounding
    // can inflate slightly; each bound carries its own slack term).
    let bpb = target.bytes_per_block();
    let sample_suffix: Vec<&str> = requant_suffixes.iter().map(|s| s.as_str()).collect();
    for suffix in sample_suffix {
        let name = format!("blk.0.{suffix}");
        let (src, rows, cols) = tensor_dense(parent, &name)?;
        let info = pack
            .tensor_info(&name)
            .ok_or_else(|| anyhow::anyhow!("pack missing {name}"))?;
        anyhow::ensure!(
            info.ggml_type == target.ggml_type(),
            "{name} in the pack is {:?}, expected {:?}",
            info.ggml_type,
            target.ggml_type()
        );
        let raw = pack
            .tensor_slice(&name)
            .ok_or_else(|| anyhow::anyhow!("pack tensor {name} has no bytes"))?;
        let nb = cols / QK_K;
        let sample_rows = [0usize, rows / 2, rows - 1];
        let mut dq = vec![0f32; cols];
        for &r in &sample_rows {
            let row_bytes = &raw[r * nb * bpb..(r + 1) * nb * bpb];
            target.dequant_row(row_bytes, &mut dq);
            let orig = &src[r * cols..(r + 1) * cols];
            let sb_sz = target.readback_subblock();
            let mut max_err = 0f32;
            for sb in 0..cols / sb_sz {
                let seg = &orig[sb * sb_sz..(sb + 1) * sb_sz];
                let bound = target.readback_bound(seg);
                for (j, v) in seg.iter().enumerate() {
                    let err = (dq[sb * sb_sz + j] - v).abs();
                    anyhow::ensure!(
                        err <= bound,
                        "{name} row {r} elem {}: err {err} > bound {bound}",
                        sb * sb_sz + j
                    );
                    max_err = max_err.max(err);
                }
            }
            eprintln!("[requant] {name} row {r}: read-back PASS (max err {max_err:.3e})");
        }
    }
    Ok(())
}
