//! The synthetic Bonsai-2 GGUF fixture kit — extracted from
//! `tests/bonsai2_rotation_load.rs` at the Issue-028 landing so the
//! disaggregated-container battery builds the EXACT same contract files
//! instead of a second hand-copy of the writer.
//!
//! Include via `#[path = "common/synth_bonsai2.rs"] mod synth;` — cargo does
//! not auto-discover `tests/` subdirectories as targets. Everything is `pub`
//! and `#![allow(dead_code)]` by design: a fixture library over-serves its
//! include sites (each test binary uses its own subset of the kit).
#![allow(dead_code)]

use std::io::Write as _;
use std::path::PathBuf;

// ── tiny GGUF writer ────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum Val {
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

pub struct TensorSpec {
    pub name: String,
    /// GGUF ne dims, ggml order: ne[0] = innermost (`in_features` for 2D weights).
    pub ne: Vec<u64>,
    pub ggml_type: u32,
    pub data: Vec<u8>,
}

/// Deterministic mixed-ternary `Q2_0` payload: codes cycle 0/1/2 by position
/// (never 3 — the repack rejects the fourth state), per-128-group f16 scale.
pub fn q2_0_payload(n_elements: u64, seed: u64) -> Vec<u8> {
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
pub fn synth_trit(j: usize, b: usize, seed: u64) -> i8 {
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
pub fn ptq1_0_payload(n_elements: u64, seed: u64) -> Vec<u8> {
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
pub fn bf16_payload(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

pub fn f32_payload(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub const ALIGN: u64 = 32;

/// Serialize header + metadata + tensor table + data; returns the file bytes.
pub fn build_gguf(metadata: &[(String, Val)], tensors: &[TensorSpec]) -> Vec<u8> {
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

pub const N_EMBD: u64 = 1024;
pub const VOCAB: u64 = 256;
pub const FFN: u64 = 1024;
pub const N_HEAD: u64 = 8;
pub const N_KV: u64 = 2;
pub const HEAD_DIM: u64 = 128;
pub const STATE: u64 = 128;
pub const N_K: u64 = 4;
pub const N_V: u64 = 8;
pub const CONV_K: u64 = 4;
pub const N_LAYER: u64 = 4; // 0,1,2 DeltaNet; 3 Attention (interval 4)

/// Bonsai-2 style metadata (folded) or pre-rotation style (`folded=false`).
pub fn synth_metadata(folded: bool) -> Vec<(String, Val)> {
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
pub fn synth_tensors_bonsai2() -> Vec<TensorSpec> {
    synth_tensors_bonsai2_typed(142)
}

/// The folded Bonsai-2 tensor set with the ternary wire format parameterized:
/// 142 = PQ2_0 (the Phase B pack), 143 = PTQ1_0 (the Phase C decode pack,
/// Issue 980 T6). Both encoders carry IDENTICAL trits + f16 group scales, so
/// the two files must load to identical containers and run bit-identically.
pub fn synth_tensors_bonsai2_typed(ternary_id: u32) -> Vec<TensorSpec> {
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
pub fn synth_tensors_old() -> Vec<TensorSpec> {
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

pub fn write_tmp(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("{name}_{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    path
}
