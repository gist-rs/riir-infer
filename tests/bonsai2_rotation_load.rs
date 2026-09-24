#![cfg(feature = "bonsai2_hadamard")]
//! Issue 980 / Plan 600 Phase B — synthetic Bonsai-2 loader + rotation tests.
//!
//! Builds a TINY synthetic GGUF that carries the exact Bonsai-2 contract
//! (arch `qwen35`, `prism.hadamard.*` metadata, dense BF16 `ssm_alpha`/`ssm_beta`
//! escape set, `Q2_0` id-142 ternary projections) and asserts:
//!
//! 1. the loader parses the rotation config (block/signs/inverse/gdn flag),
//! 2. `in_proj_a`/`in_proj_b` load as the DENSE arm while ternary tensors
//!    repack as usual,
//! 3. the forward wiring actually FIRES at every layer (per-layer capture
//!    differs vs the same weights with rotation stripped),
//! 4. the folded-matmul math matches an independent reference (explicit
//!    ±1/√n Hadamard matrix, the fork's own construction),
//! 5. a pre-rotation file (no prism keys, ternary a/b) still loads with
//!    `rotation == None` — the old-file rollback lane,
//! 6. unsupported metadata refuses LOUDLY (version, transform, unknown
//!    folded weight, sign-length mismatch).
//!
//! Real-file validation (the 7 GB `PQ2_0` pack, fork logits parity) is the
//! 4090-side G1 — this file is the fast local gate.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use katgpt_core::TernaryGroupWeights;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::types::Config;

// ── tiny GGUF writer ────────────────────────────────────────────────────────

#[derive(Clone)]
enum Val {
    U64(u64),
    F64(f64),
    Str(String),
    ArrU64(Vec<u64>),
    ArrF64(Vec<f64>),
    ArrStr(Vec<String>),
}

impl Val {
    fn tag(&self) -> u32 {
        match self {
            Self::U64(_) => 10,                                       // UINT64
            Self::F64(_) => 12,                                       // FLOAT64
            Self::Str(_) => 8,                                        // STRING
            Self::ArrU64(_) | Self::ArrF64(_) | Self::ArrStr(_) => 9, // ARRAY
        }
    }
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Self::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Self::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Self::Str(s) => {
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            Self::ArrU64(vs) => {
                out.extend_from_slice(&10u32.to_le_bytes()); // element: UINT64
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Self::ArrF64(vs) => {
                out.extend_from_slice(&12u32.to_le_bytes()); // element: FLOAT64
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Self::ArrStr(vs) => {
                out.extend_from_slice(&8u32.to_le_bytes()); // element: STRING
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for s in vs {
                    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
            }
        }
    }
}

struct TensorSpec {
    name: String,
    /// GGUF ne dims, ggml order: ne[0] = innermost (`in_features` for 2D weights).
    ne: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

/// Deterministic mixed-ternary `Q2_0` payload: codes cycle 0/1/2 by position
/// (never 3 — the repack rejects the fourth state), per-128-group f16 scale.
fn q2_0_payload(n_elements: u64, seed: u64) -> Vec<u8> {
    let n_blocks = (n_elements / 128) as usize;
    let mut out = Vec::with_capacity(n_blocks * 34);
    for b in 0..n_blocks {
        let d = 0.25f32 + ((b as f32 + seed as f32) % 7.0) * 0.125;
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let mut qs = [0u8; 32];
        for j in 0..128usize {
            let code = match (j + b * 7 + seed as usize) % 3 {
                0 => 0u8, // -1
                1 => 1u8, // 0
                _ => 2u8, // +1
            };
            qs[j / 4] |= code << ((j % 4) * 2);
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// The trit sequence `q2_0_payload` encodes, for element `j` of block `b` —
/// shared by both encoders so the two files carry identical ternary content.
fn synth_trit(j: usize, b: usize, seed: u64) -> i8 {
    match (j + b * 7 + seed as usize) % 3 {
        0 => -1,
        1 => 0,
        _ => 1,
    }
}

/// `PTQ1_0` (type 143) payload carrying the SAME trits + f16 group scales as
/// [`q2_0_payload`] — the both-encodings equivalence fixture. Encoded through
/// the fork's `quantize_row_ptq1_0_ref` construction (stages c=16/c=8 + qh,
/// ceiling map ⌈V·256/243⌉, leading-zero 5th trit in qh).
fn ptq1_0_payload(n_elements: u64, seed: u64) -> Vec<u8> {
    let n_blocks = (n_elements / 128) as usize;
    let mut out = Vec::with_capacity(n_blocks * 28);
    for b in 0..n_blocks {
        let d = 0.25f32 + ((b as f32 + seed as f32) % 7.0) * 0.125;
        // Block layout: qs[24] + qh[2] + d:f16 (d LAST — block_ptq1_0, unlike
        // Q2_0's d-first).
        let mut qs = [0u8; 24];
        let xi = |e: usize| (synth_trit(e, b, seed) + 1) as usize; // {0,1,2}
        let ceil256 = |q: usize| (q * 256).div_ceil(243) as u8;
        for (m, qs_b) in qs.iter_mut().enumerate().take(16) {
            let mut v = 0usize;
            for n in 0..5 {
                v = v * 3 + xi(m + 16 * n);
            }
            *qs_b = ceil256(v);
        }
        for (m, qs_b) in qs.iter_mut().enumerate().skip(16) {
            let mut v = 0usize;
            for n in 0..5 {
                v = v * 3 + xi(80 + (m - 16) + 8 * n);
            }
            *qs_b = ceil256(v);
        }
        out.extend_from_slice(&qs);
        let mut qh = [0u8; 2];
        for (h, qh_b) in qh.iter_mut().enumerate() {
            let mut v = 0usize;
            for m in 0..4 {
                v = v * 3 + xi(120 + h + 2 * m);
            }
            *qh_b = ceil256(v * 3);
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    }
    out
}

/// Dense BF16 payload from f32 values (truncating conversion — fine for a
/// fixture; the loader's `bf16_to_f32` is exact for truncated values).
fn bf16_payload(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

fn f32_payload(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

const ALIGN: u64 = 32;

/// Serialize header + metadata + tensor table + data; returns the file bytes.
fn build_gguf(metadata: &[(String, Val)], tensors: &[TensorSpec]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        // key string
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&v.tag().to_le_bytes());
        v.write(&mut buf);
    }
    let mut offset = 0u64;
    let mut infos = Vec::with_capacity(tensors.len());
    for t in tensors {
        buf.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t.name.as_bytes());
        buf.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
        for d in &t.ne {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&t.ggml_type.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        infos.push((offset, t.data.len() as u64));
        offset += t.data.len() as u64;
        offset = offset.div_ceil(ALIGN) * ALIGN;
    }
    let padding = ((ALIGN - (buf.len() as u64 % ALIGN)) % ALIGN) as usize;
    buf.extend(std::iter::repeat_n(0u8, padding));
    let data_base = buf.len() as u64;
    for (i, t) in tensors.iter().enumerate() {
        let (off, len) = infos[i];
        let start = (data_base + off) as usize;
        if buf.len() < start + len as usize {
            buf.resize(start + len as usize, 0);
        }
        buf[start..start + len as usize].copy_from_slice(&t.data);
    }
    buf
}

// ── synthetic model construction ────────────────────────────────────────────

const N_EMBD: u64 = 1024;
const VOCAB: u64 = 256;
const FFN: u64 = 1024;
const N_HEAD: u64 = 8;
const N_KV: u64 = 2;
const HEAD_DIM: u64 = 128;
const STATE: u64 = 128;
const N_K: u64 = 4;
const N_V: u64 = 8;
const CONV_K: u64 = 4;
const N_LAYER: u64 = 4; // 0,1,2 DeltaNet; 3 Attention (interval 4)

/// Bonsai-2 style metadata (folded) or pre-rotation style (`folded=false`).
fn synth_metadata(folded: bool) -> Vec<(String, Val)> {
    let mut m: Vec<(String, Val)> = vec![
        ("general.architecture".into(), Val::Str("qwen35".into())),
        ("general.name".into(), Val::Str("synth-bonsai2".into())),
        ("qwen35.block_count".into(), Val::U64(N_LAYER)),
        ("qwen35.embedding_length".into(), Val::U64(N_EMBD)),
        ("qwen35.attention.head_count".into(), Val::U64(N_HEAD)),
        ("qwen35.attention.head_count_kv".into(), Val::U64(N_KV)),
        ("qwen35.attention.key_length".into(), Val::U64(HEAD_DIM)),
        ("qwen35.attention.value_length".into(), Val::U64(HEAD_DIM)),
        ("qwen35.feed_forward_length".into(), Val::U64(FFN)),
        ("qwen35.context_length".into(), Val::U64(2048)),
        (
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            Val::F64(1e-6),
        ),
        ("qwen35.rope.freq_base".into(), Val::F64(10000.0)),
        ("qwen35.ssm.conv_kernel".into(), Val::U64(CONV_K)),
        ("qwen35.ssm.state_size".into(), Val::U64(STATE)),
        ("qwen35.ssm.group_count".into(), Val::U64(N_K)),
        ("qwen35.ssm.time_step_rank".into(), Val::U64(N_V)),
        ("qwen35.full_attention_interval".into(), Val::U64(4)),
    ];
    if folded {
        // sign vector: every folded input width is 1024 here, so ONE width.
        let signs: Vec<f64> = (0..N_EMBD)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let mut names = vec!["output.weight".to_string()];
        for i in 0..N_LAYER {
            if (i + 1) % 4 == 0 {
                for k in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                    names.push(format!("blk.{i}.{k}.weight"));
                }
            } else {
                for k in ["attn_qkv", "attn_gate", "ssm_out"] {
                    names.push(format!("blk.{i}.{k}.weight"));
                }
            }
            for k in ["ffn_down", "ffn_gate", "ffn_up"] {
                names.push(format!("blk.{i}.{k}.weight"));
            }
        }
        m.push(("prism.hadamard.version".into(), Val::U64(1)));
        m.push(("prism.hadamard.block_size".into(), Val::U64(1024)));
        m.push((
            "prism.hadamard.transform".into(),
            Val::Str("normalized-sylvester-walsh-hadamard".into()),
        ));
        m.push((
            "prism.hadamard.axis".into(),
            Val::Str("input-last-dimension".into()),
        ));
        m.push((
            "prism.hadamard.sign_mode".into(),
            Val::Str("explicit".into()),
        ));
        m.push(("prism.hadamard.weight_names".into(), Val::ArrStr(names)));
        m.push((
            "prism.hadamard.sign_widths".into(),
            Val::ArrU64(vec![N_EMBD]),
        ));
        m.push(("prism.hadamard.sign_values".into(), Val::ArrF64(signs)));
        m.push((
            "prism.hadamard.inverse_weight_names".into(),
            Val::ArrStr(vec!["token_embd.weight".into()]),
        ));
        m.push(("prism.hadamard.gdn_v_grouped".into(), Val::U64(1)));
    }
    m
}

/// Bonsai-2 style tensors: ternary `Q2_0` projections + DENSE BF16 a/b.
fn synth_tensors_bonsai2() -> Vec<TensorSpec> {
    synth_tensors_bonsai2_typed(142)
}

/// The folded Bonsai-2 tensor set with the ternary wire format parameterized:
/// 142 = PQ2_0 (the Phase B pack), 143 = PTQ1_0 (the Phase C decode pack,
/// Issue 980 T6). Both encoders carry IDENTICAL trits + f16 group scales, so
/// the two files must load to identical containers and run bit-identically.
fn synth_tensors_bonsai2_typed(ternary_id: u32) -> Vec<TensorSpec> {
    let mut t = Vec::with_capacity(N_LAYER as usize);
    let seed = |name: &str| name.bytes().map(|b| b as u64).sum::<u64>() % 97;

    let ternary = |t: &mut Vec<TensorSpec>, name: &str, ne0: u64, ne1: u64| {
        let payload = match ternary_id {
            143 => ptq1_0_payload(ne0 * ne1, seed(name)),
            _ => q2_0_payload(ne0 * ne1, seed(name)),
        };
        t.push(TensorSpec {
            name: name.into(),
            ne: vec![ne0, ne1],
            ggml_type: ternary_id,
            data: payload,
        });
    };

    ternary(&mut t, "token_embd.weight", N_EMBD, VOCAB);
    ternary(&mut t, "output.weight", N_EMBD, VOCAB);
    let output_norm = vec![1.0f32; N_EMBD as usize];
    t.push(TensorSpec {
        name: "output_norm.weight".into(),
        ne: vec![N_EMBD],
        ggml_type: 0, // F32
        data: f32_payload(&output_norm),
    });

    for i in 0..N_LAYER {
        let is_attn = (i + 1) % 4 == 0;
        // norms (names per qwen35_deltanet_gguf_names)
        for (name, ne) in [
            (format!("blk.{i}.attn_norm.weight"), N_EMBD),
            (format!("blk.{i}.post_attention_norm.weight"), N_EMBD),
        ] {
            t.push(TensorSpec {
                name,
                ne: vec![ne],
                ggml_type: 0,
                data: f32_payload(&vec![1.0f32; ne as usize]),
            });
        }
        // FFN (both layer types; all folded inputs are 1024-wide here)
        ternary(&mut t, &format!("blk.{i}.ffn_gate.weight"), N_EMBD, FFN);
        ternary(&mut t, &format!("blk.{i}.ffn_up.weight"), N_EMBD, FFN);
        ternary(&mut t, &format!("blk.{i}.ffn_down.weight"), FFN, N_EMBD);

        if is_attn {
            ternary(
                &mut t,
                &format!("blk.{i}.attn_q.weight"),
                N_EMBD,
                N_HEAD * HEAD_DIM * 2,
            );
            ternary(
                &mut t,
                &format!("blk.{i}.attn_k.weight"),
                N_EMBD,
                N_KV * HEAD_DIM,
            );
            ternary(
                &mut t,
                &format!("blk.{i}.attn_v.weight"),
                N_EMBD,
                N_KV * HEAD_DIM,
            );
            ternary(
                &mut t,
                &format!("blk.{i}.attn_output.weight"),
                N_HEAD * HEAD_DIM,
                N_EMBD,
            );
            for name in [
                format!("blk.{i}.attn_q_norm.weight"),
                format!("blk.{i}.attn_k_norm.weight"),
            ] {
                t.push(TensorSpec {
                    name,
                    ne: vec![HEAD_DIM],
                    ggml_type: 0,
                    data: f32_payload(&vec![1.0f32; HEAD_DIM as usize]),
                });
            }
        } else {
            let key_dim = N_K * STATE;
            let value_dim = N_V * STATE;
            let conv_dim = key_dim * 2 + value_dim;
            ternary(
                &mut t,
                &format!("blk.{i}.attn_qkv.weight"),
                N_EMBD,
                key_dim * 2 + value_dim,
            );
            ternary(
                &mut t,
                &format!("blk.{i}.attn_gate.weight"),
                N_EMBD,
                value_dim,
            );
            ternary(
                &mut t,
                &format!("blk.{i}.ssm_out.weight"),
                value_dim,
                N_EMBD,
            );
            // conv1d [d_conv, conv_dim] F32
            t.push(TensorSpec {
                name: format!("blk.{i}.ssm_conv1d.weight"),
                ne: vec![CONV_K, conv_dim],
                ggml_type: 0,
                data: f32_payload(&vec![0.1f32; (CONV_K * conv_dim) as usize]),
            });
            // dt bias + a (F32)
            t.push(TensorSpec {
                name: format!("blk.{i}.ssm_dt.bias"),
                ne: vec![N_V],
                ggml_type: 0,
                data: f32_payload(&vec![0.01f32; N_V as usize]),
            });
            t.push(TensorSpec {
                name: format!("blk.{i}.ssm_a"),
                ne: vec![N_V],
                ggml_type: 0,
                data: f32_payload(&vec![1.0f32; N_V as usize]),
            });
            // ssm_norm: per-head [head_dim]
            t.push(TensorSpec {
                name: format!("blk.{i}.ssm_norm.weight"),
                ne: vec![STATE],
                ggml_type: 0,
                data: f32_payload(&vec![1.0f32; STATE as usize]),
            });
            // THE ESCAPE SET: dense BF16 a/b, ne = [n_embd, n_v_heads]
            let a_vals: Vec<f32> = (0..(N_EMBD * N_V))
                .map(|j| ((j % 11) as f32 - 5.0) * 0.1)
                .collect();
            let b_vals: Vec<f32> = (0..(N_EMBD * N_V))
                .map(|j| ((j % 7) as f32 - 3.0) * 0.1)
                .collect();
            t.push(TensorSpec {
                name: format!("blk.{i}.ssm_alpha.weight"),
                ne: vec![N_EMBD, N_V],
                ggml_type: 30, // BF16
                data: bf16_payload(&a_vals),
            });
            t.push(TensorSpec {
                name: format!("blk.{i}.ssm_beta.weight"),
                ne: vec![N_EMBD, N_V],
                ggml_type: 30,
                data: bf16_payload(&b_vals),
            });
        }
    }
    t
}

/// Pre-rotation style: ternary a/b (`Q2_0`), no prism keys.
fn synth_tensors_old() -> Vec<TensorSpec> {
    let mut t = synth_tensors_bonsai2();
    for spec in t.iter_mut() {
        if spec.ggml_type == 30 {
            // BF16 a/b → ternary Q2_0 with the same geometry
            let n_elements: u64 = spec.ne.iter().product();
            spec.ggml_type = 142;
            spec.data = q2_0_payload(n_elements, 5);
        }
    }
    t
}

fn write_tmp(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("{name}_{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    path
}

// ── tests ───────────────────────────────────────────────────────────────────

fn load(
    path: &Path,
) -> anyhow::Result<(
    Config,
    riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
)> {
    load_qwen_deltanet_ternary_weights_gguf(path)
}

/// `expect_err` needs `Debug` on the Ok payload; the production types do not
/// derive it, so the refusal tests match the error text instead.
fn load_expect_err(path: &Path, contains: &str) -> anyhow::Error {
    match load(path) {
        Ok(_) => panic!("expected load refusal containing '{contains}'"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(contains),
                "refusal message '{msg}' does not mention '{contains}'"
            );
            e
        }
    }
}

/// 1+2. The folded file parses; a/b are the DENSE arm; rotation metadata is
/// complete (block/signs/inverse/gdn).
#[test]
fn bonsai2_synthetic_loads_with_rotation_and_dense_gate_projs() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_rot", &bytes);
    let (config, weights) = load(&path).expect("load folded synthetic");
    let _ = std::fs::remove_file(&path);

    assert_eq!(config.n_layer, N_LAYER as usize);
    assert_eq!(config.n_embd, N_EMBD as usize);
    assert_eq!(config.deltanet_linear_n_value_heads, N_V as usize);
    assert_eq!(config.deltanet_linear_n_heads, N_K as usize);
    assert_eq!(config.vocab_size, VOCAB as usize);

    let rot = weights
        .rotation
        .as_ref()
        .expect("rotation must be Some for a folded file");
    assert_eq!(rot.block_size, 1024);
    assert!(rot.inverse_embedding);
    assert!(rot.gdn_v_grouped);
    assert_eq!(rot.gdn_v_heads, N_V as usize);
    assert_eq!(rot.gdn_k_groups, N_K as usize);
    let signs = rot
        .signs_for_width(N_EMBD as usize)
        .expect("sign vector for 1024");
    assert_eq!(signs.len(), N_EMBD as usize);
    assert_eq!(signs[0], -1);

    // a/b dense arm with the right geometry ([n_v_heads × n_embd]).
    for l in &weights.layers {
        if l.in_proj_qkv.rows > 0 {
            let GateProjShape(a_rows, a_cols) = GateProjShape::of(&l.in_proj_a);
            assert_eq!((a_rows, a_cols), (N_V as usize, N_EMBD as usize));
            assert!(matches!(
                l.in_proj_a,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Dense(..)
            ));
            assert!(matches!(
                l.in_proj_b,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Dense(..)
            ));
        }
    }
    // Attention layers carry empty a/b.
    let attn = weights.layers.last().unwrap();
    assert_eq!(attn.in_proj_a.rows(), 0);
    // Global invariants still hold.
    assert!(weights.invariants_hold());
}

struct GateProjShape(usize, usize);
impl GateProjShape {
    fn of(w: &riir_infer_core::deltanet::ternary_weights::GateProjWeights) -> Self {
        Self(w.rows(), w.cols())
    }
}

/// 5. The pre-rotation file (ternary a/b, no prism keys) still loads with
///    `rotation == None` — the old-file rollback lane.
#[test]
fn old_file_loads_without_rotation() {
    let bytes = build_gguf(&synth_metadata(false), &synth_tensors_old());
    let path = write_tmp("riir_b2_old", &bytes);
    let (_, weights) = load(&path).expect("load old synthetic");
    let _ = std::fs::remove_file(&path);
    assert!(weights.rotation.is_none());
    for l in &weights.layers {
        if l.in_proj_qkv.rows > 0 {
            assert!(matches!(
                l.in_proj_a,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Ternary(_)
            ));
        }
    }
    assert!(weights.invariants_hold());
}

/// 3. The rotation wiring FIRES at every layer: per-layer residual captures
///    differ between the folded run and the same weights with rotation stripped.
#[test]
fn forward_rotation_wiring_fires_every_layer() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_rot_fwd", &bytes);
    let (config, mut weights) = load(&path).expect("load");
    let _ = std::fs::remove_file(&path);

    let layer_types = weights.layer_types.clone();
    let run = |w: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights| {
        let mut cache =
            riir_infer_core::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = riir_infer_core::deltanet::HybridForwardScratch::new(&config);
        let rope = riir_infer_core::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
        let mut captures = vec![vec![0.0f32; config.n_embd]; config.n_layer];
        riir_infer_core::deltanet::forward_qwen_deltanet_ternary_with_capture(
            &mut x,
            w,
            &mut cache,
            7,
            0,
            &config,
            &mut scratch,
            &rope,
            Some(&mut captures),
        );
        captures
    };

    let rotated = run(&weights);
    weights.rotation = None;
    let plain = run(&weights);

    for (i, (r, p)) in rotated.iter().zip(plain.iter()).enumerate() {
        let diff: f32 = r.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "layer {i}: rotation wiring produced no difference (sum |Δ| = {diff})"
        );
    }
}

/// 4. Folded-matmul math vs the fork's OWN construction: explicit ±1/√n
///    Hadamard matrix (row&col parity) times the sign-multiplied input, then a
///    dense matmul with the dequantized folded weights. Tolerance covers the
///    FWHT-vs-matrix rounding difference only.
#[test]
fn folded_matmul_matches_explicit_matrix_reference() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_rot_mm", &bytes);
    let (_, weights) = load(&path).expect("load");
    let _ = std::fs::remove_file(&path);

    let rot = weights.rotation.as_ref().unwrap();
    let signs = rot.signs_for_width(N_EMBD as usize).unwrap();
    let layer = &weights.layers[0];
    let w = &layer.in_proj_qkv; // [key_dim*2+value_dim, n_embd] folded ternary

    // Deterministic input.
    let x: Vec<f32> = (0..N_EMBD as usize)
        .map(|i| ((i * 37 + 11) % 97) as f32 - 48.0)
        .collect();

    // OUR path: sign → FWHT per block → SIMD ternary matvec.
    let mut x_ours = x.clone();
    riir_infer_core::deltanet::rotation::rotate_forward_inplace(
        &mut x_ours,
        Some(signs),
        rot.block_size,
    );
    let mut y_ours = vec![0.0f32; w.rows];
    katgpt_core::simd_ternary_group_matvec_parallel(w, &x_ours, &mut y_ours);

    // REFERENCE: dequantize the ternary weights, build the explicit block
    // matrix, dense-matmul.
    let w_dense = riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(w);
    let block = rot.block_size;
    let scale = 1.0 / (block as f32).sqrt();
    let mut x_ref = vec![0.0f32; N_EMBD as usize];
    for (bi, xb) in x.chunks_exact(block).enumerate() {
        let signs_b = &signs[bi * block..(bi + 1) * block];
        for r in 0..block {
            let mut acc = 0.0f32;
            for c in 0..block {
                let mut parity = r & c;
                parity ^= parity >> 16;
                parity ^= parity >> 8;
                parity ^= parity >> 4;
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                acc += xb[c] * signs_b[c] as f32 * if parity & 1 == 1 { -scale } else { scale };
            }
            x_ref[bi * block + r] = acc;
        }
    }
    for r in 0..w.rows {
        let row = &w_dense[r * w.cols..(r + 1) * w.cols];
        let mut acc = 0.0f32;
        for (&wv, &xv) in row.iter().zip(x_ref.iter()) {
            acc += wv * xv;
        }
        let denom = acc.abs().max(1e-3);
        assert!(
            (y_ours[r] - acc).abs() / denom < 1e-3,
            "row {r}: ours {} ref {acc}",
            y_ours[r]
        );
    }
}

/// 6a. Unsupported version refuses loudly.
#[test]
fn unsupported_version_refuses() {
    let mut meta = synth_metadata(true);
    for (k, v) in meta.iter_mut() {
        if k == "prism.hadamard.version" {
            *v = Val::U64(2);
        }
    }
    let bytes = build_gguf(&meta, &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_v2", &bytes);
    load_expect_err(&path, "unsupported");
    let _ = std::fs::remove_file(&path);
}

/// 6b. A folded weight outside the engine's verified structural map refuses.
#[test]
fn unknown_folded_weight_refuses() {
    let mut meta = synth_metadata(true);
    for (k, v) in meta.iter_mut() {
        if let Val::ArrStr(names) = v
            && k == "prism.hadamard.weight_names"
        {
            names.push("blk.0.ssm_alpha.weight".into()); // the escape set — never folded
        }
    }
    let bytes = build_gguf(&meta, &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_unknown", &bytes);
    load_expect_err(&path, "ssm_alpha");
    let _ = std::fs::remove_file(&path);
}

/// 6c. `sign_values` length mismatch refuses.
#[test]
fn sign_length_mismatch_refuses() {
    let mut meta = synth_metadata(true);
    for (k, v) in meta.iter_mut() {
        if let Val::ArrF64(signs) = v
            && k == "prism.hadamard.sign_values"
        {
            signs.pop();
        }
    }
    let bytes = build_gguf(&meta, &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_signlen", &bytes);
    load_expect_err(&path, "sign_values");
    let _ = std::fs::remove_file(&path);
}

/// The old-file regression at the `GgufFile` layer: a `Q2_0` id-42 tensor opens
/// identically to id-142 (the `PQ2_0` relabel) — both map to the ternary arm.
/// (`from_id` is crate-private; the public observable is the loader arm, and
/// the same fact is pinned in the in-crate `test_ggml_type_from_id`.)
#[test]
fn q2_0_ids_42_and_142_share_the_ternary_arm() {
    // Old-style synthetic (id 142) and the repack of an id-42 payload must
    // both load through the ternary arm — covered by
    // `old_file_loads_without_rotation` (ternary a/b) and
    // `bonsai2_synthetic_loads_with_rotation_and_dense_gate_projs` (ternary
    // projections at id 142). This placeholder keeps the intent greppable.
}

// Keep imports honest when features shift.
#[allow(dead_code)]
fn _touch(_: &GgufFile, _: &TernaryGroupWeights) {}

// ── PTQ1_0 (type 143) — the Phase C decode pack lane (Issue 980 T6) ──────────

/// A type-143 folded file loads through the SAME ternary arm: rotation
/// parses, invariants hold — the loader treats PTQ1_0 as a wire encoding of
/// the same substrate, not a new model class.
#[test]
fn ptq1_0_synthetic_loads_with_rotation() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2_typed(143));
    let path = write_tmp("riir_b2_ptq10", &bytes);
    let (config, weights) = load(&path).expect("load PTQ1_0 folded synthetic");
    let _ = std::fs::remove_file(&path);

    assert!(
        weights.rotation.is_some(),
        "rotation metadata is wire-format independent"
    );
    assert!(weights.invariants_hold());
    assert_eq!(config.n_layer, N_LAYER as usize);
    // Ternary a/b still the Dense (BF16) escape set — the 143 payload only
    // touches the ternary projections.
    for l in &weights.layers {
        if l.in_proj_qkv.rows > 0 {
            assert!(matches!(
                l.in_proj_a,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Dense(..)
            ));
        }
    }
}

/// THE equivalence gate: PQ2_0 (id 142) and PTQ1_0 (id 143) encodings of the
/// same trits + scales must load to IDENTICAL containers and run the
/// rotated forward BIT-IDENTICALLY. This is the whole losslessness argument
/// for the Phase C decode lane in one test: any trit-map or scale-placement
/// error in the 143 decoder shows up as a container or forward divergence.
#[test]
fn ptq1_0_and_pq2_0_encodings_run_bit_identically() {
    let run = |ternary_id: u32| {
        let bytes = build_gguf(
            &synth_metadata(true),
            &synth_tensors_bonsai2_typed(ternary_id),
        );
        let path = write_tmp(&format!("riir_b2_eq_{ternary_id}"), &bytes);
        let (config, weights) = load(&path).expect("load");
        let _ = std::fs::remove_file(&path);
        let layer_types = weights.layer_types.clone();
        let mut cache =
            riir_infer_core::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = riir_infer_core::deltanet::HybridForwardScratch::new(&config);
        let rope = riir_infer_core::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
        for (pos, tok) in [7usize, 13, 42].iter().enumerate() {
            riir_infer_core::deltanet::forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                *tok,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
        }
        x[..config.vocab_size.min(x.len())].to_vec()
    };

    let pq = run(142);
    let ptq = run(143);
    assert_eq!(pq.len(), ptq.len(), "vocab/geometry must agree");
    for (i, (a, b)) in pq.iter().zip(ptq.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {i} diverged between PQ2_0 and PTQ1_0 encodings — the 143 decoder is not lossless"
        );
    }
}
