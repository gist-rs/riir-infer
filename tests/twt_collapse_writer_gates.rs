//! TWT Phase-4 writer gate battery (Issue 022 T4.3) — the collapsed-GGUF
//! writer over a SYNTHETIC ternary parent (no checkpoint needed). What
//! these gates pin:
//! - the emitted file RE-OPENS through the stock reader (version 3,
//!   renumbered block_count, the twt.* keys present);
//! - member-passthrough blocks are BYTE-IDENTICAL to the winner's parent
//!   payload (renamed, never requantized);
//! - merged-operator blocks carry the plan's payloads at the right
//!   offsets, with every tensor aligned;
//! - a Q2_0 merged payload DEQUANTS through the stock fn to the arm's
//!   own values (the wire is the container's truth);
//! - refusal arms: incomplete merged plans, non-tiling tables, a stale
//!   block_count override;
//! - determinism: two emits of one spec are byte-identical.

#![cfg(feature = "twt_collapse")]

use std::collections::BTreeMap;

use katgpt_core::TernaryGroupWeights;
use riir_infer_core::gguf_loader::{GgufFile, GgmlType, GgufValue};
use riir_infer_core::quant::q2_0::{dequantize_row_q2_0, pack_ternary_group_to_q2_0, BlockQ2_0};
use riir_infer_core::twt::collapse_writer::{
    emit_collapsed_gguf, q2_0_wire_bytes, twt_arm_codes_value, twt_block_table_value,
    CollapseSpec, LayerSource, TensorOut,
};
use riir_infer_core::twt::ternarize::{arm_source_quant, TwtArm};
use riir_infer_core::twt::TwtError;

/// A minimal ternary layer fixture: one Q2_0 projection (2×256) + one
/// F32 norm (256). Byte-serialized by hand with the writer's own layout
/// rules — this fixture IS a stress of the writer's serializer.
struct FixtureParent {
    bytes: Vec<u8>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Build a synthetic 3-layer parent GGUF (arch `qwen35twtfx`) with:
/// - globals: `token_embd.weight` (Q2_0, 4×256) + `output_norm.weight` (F32, 256);
/// - per layer: `blk.{i}.in_proj.weight` (Q2_0, 2×256) +
///   `blk.{i}.attn_norm.weight` (F32, 256).
fn build_parent() -> FixtureParent {
    let mut rng = 0xDEADBEEFu64;
    let mut next_f32 = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((rng >> 33) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.5
    };

    // Payloads: every Q2_0 tensor 2×256 → 4 groups → 4×34 = 136 bytes.
    let mut proj_payloads: Vec<Vec<u8>> = Vec::new();
    for _ in 0..3 {
        let mut blocks: Vec<BlockQ2_0> = Vec::new();
        for _ in 0..4 {
            let mut b = BlockQ2_0 { d: 0, qs: [0u8; 32] };
            let s = next_f32().abs().max(0.01);
            b.d = half::f16::from_f32(s).to_bits();
            for j in 0..128 {
            let code = (next_f32() * 1e6_f32).abs() as u64 % 3; // 0/1/2 via the same stream
                b.qs[j / 4] |= (code as u8) << ((j % 4) * 2);
            }
            blocks.push(b);
        }
        proj_payloads.push(blocks.iter().flat_map(|b| bytemuck::bytes_of(b).to_vec()).collect());
    }
    let emb_payload = {
        // 4×256 → 8 groups → 8×34 = 272 bytes.
        let mut out = Vec::new();
        for _ in 0..8 {
            let mut b = BlockQ2_0 { d: 0, qs: [0u8; 32] };
            b.d = half::f16::from_f32(0.05).to_bits();
            out.extend_from_slice(bytemuck::bytes_of(&b));
        }
        out
    };
    let norm_payloads: Vec<Vec<f32>> = (0..3)
        .map(|_| (0..256).map(|_| 0.9 + next_f32().abs() * 0.2).collect())
        .collect();
    let out_norm = (0..256).map(|_| 1.0f32).collect::<Vec<_>>();

    // Metadata (file order matters for the writer's mirror).
    let kvs: Vec<(String, GgufValue)> = vec![
        ("general.architecture".to_owned(), GgufValue::String("qwen35twtfx".to_owned())),
        ("qwen35twtfx.block_count".to_owned(), GgufValue::U64(3)),
        ("qwen35twtfx.embedding_length".to_owned(), GgufValue::U64(256)),
        ("general.file_type".to_owned(), GgufValue::U32(142)),
    ];

    // Tensor table: (name, type, shape-ne-order, payload).
    let q2 = GgmlType::Q2_0;
    let f32t = GgmlType::F32;
    let tensors: Vec<(String, GgmlType, Vec<usize>, Vec<u8>)> = vec![
        ("token_embd.weight".to_owned(),
            q2,
            vec![256, 4],
            emb_payload,
        ),
        ("output_norm.weight".to_owned(), f32t, vec![256], f32_bytes(&out_norm)),
        ("blk.0.in_proj.weight".to_owned(), q2, vec![256, 2], proj_payloads[0].clone()),
        ("blk.0.attn_norm.weight".to_owned(), f32t, vec![256], f32_bytes(&norm_payloads[0])),
        ("blk.1.in_proj.weight".to_owned(), q2, vec![256, 2], proj_payloads[1].clone()),
        ("blk.1.attn_norm.weight".to_owned(), f32t, vec![256], f32_bytes(&norm_payloads[1])),
        ("blk.2.in_proj.weight".to_owned(), q2, vec![256, 2], proj_payloads[2].clone()),
        ("blk.2.attn_norm.weight".to_owned(), f32t, vec![256], f32_bytes(&norm_payloads[2])),
    ];

    // Serialize.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in &kvs {
        write_kv(&mut buf, k, v);
    }
    // offsets
    let alignment = 32usize;
    let mut infos: Vec<(String, u32, Vec<usize>, u64)> = Vec::new();
    let mut cursor = 0u64;
    for (name, t, shape, data) in &tensors {
        cursor = cursor.div_ceil(alignment as u64) * alignment as u64;
        infos.push((name.clone(), t.id(), shape.clone(), cursor));
        cursor += data.len() as u64;
    }
    for (name, t, shape, off) in &infos {
        write_str(&mut buf, name);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            buf.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        buf.extend_from_slice(&t.to_le_bytes());
        buf.extend_from_slice(&off.to_le_bytes());
    }
    // Pad to the (aligned) data-section start, then each payload to its
    // section-relative offset.
    let data_start = (buf.len() as u64).div_ceil(alignment as u64) * alignment as u64;
    while (buf.len() as u64) < data_start {
        buf.push(0);
    }
    for ((_, _, _, off), (_, _, _, data)) in infos.iter().zip(tensors.iter()) {
        while (buf.len() as u64) < data_start + off {
            buf.push(0);
        }
        buf.extend_from_slice(data);
    }

    FixtureParent { bytes: buf }
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_kv(buf: &mut Vec<u8>, k: &str, v: &GgufValue) {
    write_str(buf, k);
    let (tag, payload): (u32, Vec<u8>) = match v {
        GgufValue::U64(x) => (10, x.to_le_bytes().to_vec()),
        GgufValue::U32(x) => (4, x.to_le_bytes().to_vec()),
        GgufValue::String(s) => {
            let mut p = (s.len() as u64).to_le_bytes().to_vec();
            p.extend_from_slice(s.as_bytes());
            (8, p)
        }
        GgufValue::Array(items) => {
            // U32 array only (all this fixture needs).
            let mut p = 4u32.to_le_bytes().to_vec();
            p.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                if let GgufValue::U32(x) = it {
                    p.extend_from_slice(&x.to_le_bytes());
                }
            }
            (9, p)
        }
        other => panic!("fixture kv unhandled: {other:?}"),
    };
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(&payload);
}

fn write_to_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("twt_collapse_wg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

/// A merged winner: re-quantized in_proj (arm C over the mean of the
/// block's members) + the mean F32 norm.
fn merged_payloads(
    parent: &GgufFile,
    members: [usize; 3],
) -> BTreeMap<String, TensorOut> {
    let deq = |li: usize| {
        let name = format!("blk.{li}.in_proj.weight");
        let info = parent
            .tensor_infos
            .iter()
            .find(|i| i.name == name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        let raw = parent.tensor_slice(&name).unwrap();
        let mut blocks = Vec::new();
        for chunk in raw.as_chunks::<34>().0 {
            let mut b = BlockQ2_0 { d: 0, qs: [0u8; 32] };
            b.d = u16::from_le_bytes([chunk[0], chunk[1]]);
            b.qs.copy_from_slice(&chunk[2..]);
            blocks.push(b);
        }
        let mut out = vec![0f32; info.shape.iter().product()];
        dequantize_row_q2_0(&blocks, &mut out);
        out
    };
    let d0 = deq(members[0]);
    let d1 = deq(members[1]);
    let d2 = deq(members[2]);
    let fbar = riir_infer_core::twt::audition::merge_mean([&d0[..], &d1[..], &d2[..]]).unwrap();
    let mat = arm_source_quant(&fbar, 2, 256).unwrap();
    let wire = match &mat {
        riir_infer_core::twt::ternarize::Materialized::Ternary(w) => q2_0_wire_bytes(w).unwrap(),
        _ => panic!("arm C is ternary"),
    };

    // The mean norm (the pinned merged_mean menu).
    let norm = |li: usize| {
        let raw = parent
            .tensor_slice(&format!("blk.{li}.attn_norm.weight"))
            .unwrap();
        raw.as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<_>>()
    };
    let n0 = norm(members[0]);
    let n1 = norm(members[1]);
    let n2 = norm(members[2]);
    let mean_norm: Vec<f32> = (0..256)
        .map(|i| (n0[i] + n1[i] + n2[i]) / 3.0)
        .collect();

    let mut map = BTreeMap::new();
    map.insert(
        "in_proj.weight".to_owned(),
        TensorOut { ggml_type: GgmlType::Q2_0, shape: vec![256, 2], data: wire },
    );
    map.insert(
        "attn_norm.weight".to_owned(),
        TensorOut { ggml_type: GgmlType::F32, shape: vec![256], data: f32_bytes(&mean_norm) },
    );
    map
}

#[test]
fn the_collapsed_file_reopens_with_the_right_shape() {
    let fx = build_parent();
    let parent_path = write_to_temp("parent.gguf", &fx.bytes);
    let parent = GgufFile::open(&parent_path).unwrap();

    // blocks: [0..2) keep member 1; [2..3) merged over members 0..3.
    let merged = merged_payloads(&parent, [0, 1, 2]);
    let spec = CollapseSpec {
        blocks: vec![
            (0, 2, LayerSource::Member(1)),
            (2, 3, LayerSource::Merged(merged)),
        ],
        metadata_overrides: vec![(
            "qwen35twtfx.block_count".to_owned(),
            GgufValue::U64(2),
        )],
        twt_meta: vec![
            ("twt.block_table".to_owned(), twt_block_table_value(&[(0, 2), (2, 3)])),
            (
                "twt.arm_codes".to_owned(),
                twt_arm_codes_value(&[TwtArm::Member, TwtArm::SourceQuant]),
            ),
            ("twt.arm_legend".to_owned(), GgufValue::String(TwtArm::LEGEND.to_owned())),
            (
                "twt.parent_weights_blake3".to_owned(),
                GgufValue::String(riir_infer_core::twt::blake3_of(&fx.bytes)),
            ),
        ],
    };

    let mut out: Vec<u8> = Vec::new();
    let stats = emit_collapsed_gguf(&parent, &spec, &mut out).unwrap();
    assert_eq!(stats.block_count, 2);
    assert_eq!(stats.n_tensors, 2 + 2 + 2); // globals + blk.0 + blk.1
    assert_eq!(stats.bytes_written as usize, out.len());

    // Re-open through the stock reader.
    let collapsed_path = write_to_temp("collapsed.gguf", &out);
    let g = GgufFile::open(&collapsed_path).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.metadata.get("qwen35twtfx.block_count").unwrap().as_u64(), Some(2));
    assert_eq!(
        g.metadata.get("general.architecture").unwrap().as_str(),
        Some("qwen35twtfx")
    );
    assert!(g.metadata.contains_key("twt.block_table"));
    assert!(g.metadata.contains_key("twt.arm_codes"));
    assert!(g.metadata.contains_key("twt.parent_weights_blake3"));
    // The twt.block_table round-trips: [0,2,2,3].
    match g.metadata.get("twt.block_table").unwrap() {
        GgufValue::Array(items) => {
            let vals: Vec<u64> = items.iter().filter_map(|v| v.as_u64()).collect();
            assert_eq!(vals, vec![0, 2, 2, 3]);
        }
        other => panic!("block_table shape: {other:?}"),
    }

    // Tensor set: globals + blk.0.{from member 1} + blk.1.{supplied}.
    let names: Vec<&str> = g.tensor_infos.iter().map(|i| i.name.as_str()).collect();
    assert!(names.contains(&"token_embd.weight"));
    assert!(names.contains(&"output_norm.weight"));
    assert!(names.contains(&"blk.0.in_proj.weight"));
    assert!(names.contains(&"blk.0.attn_norm.weight"));
    assert!(names.contains(&"blk.1.in_proj.weight"));
    assert!(names.contains(&"blk.1.attn_norm.weight"));
    assert_eq!(names.len(), 6);

    // Member block: BYTE-IDENTICAL to the winner's parent payload.
    let src = parent.tensor_slice("blk.1.in_proj.weight").unwrap();
    let dst = g.tensor_slice("blk.0.in_proj.weight").unwrap();
    assert_eq!(src, dst, "member passthrough must be a byte-copy");
    let src = parent.tensor_slice("blk.1.attn_norm.weight").unwrap();
    let dst = g.tensor_slice("blk.0.attn_norm.weight").unwrap();
    assert_eq!(src, dst);

    // Merged block: the payload dequantizes through the stock fn to the
    // arm's own eval view.
    let wire = g.tensor_slice("blk.1.in_proj.weight").unwrap();
    let mut blocks = Vec::new();
    for chunk in wire.as_chunks::<34>().0 {
        let mut b = BlockQ2_0 { d: 0, qs: [0u8; 32] };
        b.d = u16::from_le_bytes([chunk[0], chunk[1]]);
        b.qs.copy_from_slice(&chunk[2..]);
        blocks.push(b);
    }
    let mut deq = vec![0f32; 2 * 256];
    dequantize_row_q2_0(&blocks, &mut deq);

    // Recompute the expected arm-C output from the parent.
    let expected = {
        let merged2 = merged_payloads(&parent, [0, 1, 2]);
        merged2["in_proj.weight"].data.clone()
    };
    // The wire bytes ARE the payload (byte-identity of emit vs plan).
    assert_eq!(wire, &expected[..]);

    // Alignment: every tensor's absolute data start sits on 32.
    for info in &g.tensor_infos {
        assert_eq!(info.data_start % 32, 0, "{} misaligned", info.name);
    }
}

#[test]
fn two_emits_of_one_spec_are_byte_identical() {
    let fx = build_parent();
    let parent_path = write_to_temp("parent_det.gguf", &fx.bytes);
    let parent = GgufFile::open(&parent_path).unwrap();
    let spec = CollapseSpec {
        blocks: vec![(0, 3, LayerSource::Member(2))],
        metadata_overrides: vec![("qwen35twtfx.block_count".to_owned(), GgufValue::U64(1))],
        twt_meta: vec![(
            "twt.block_table".to_owned(),
            twt_block_table_value(&[(0, 3)]),
        )],
    };
    let mut a = Vec::new();
    let mut b = Vec::new();
    emit_collapsed_gguf(&parent, &spec, &mut a).unwrap();
    emit_collapsed_gguf(&parent, &spec, &mut b).unwrap();
    assert_eq!(a, b);
}

#[test]
fn refusals_are_loud() {
    let fx = build_parent();
    let parent_path = write_to_temp("parent_ref.gguf", &fx.bytes);
    let parent = GgufFile::open(&parent_path).unwrap();

    // Stale block_count override.
    let spec = CollapseSpec {
        blocks: vec![(0, 3, LayerSource::Member(2))],
        metadata_overrides: vec![("qwen35twtfx.block_count".to_owned(), GgufValue::U64(3))],
        twt_meta: vec![],
    };
    let mut out = Vec::new();
    match emit_collapsed_gguf(&parent, &spec, &mut out) {
        Err(TwtError::GgufWrite(msg)) => assert!(msg.contains("block_count"), "{msg}"),
        other => panic!("stale block_count must refuse, got {other:?}"),
    }

    // Missing override.
    let spec = CollapseSpec {
        blocks: vec![(0, 3, LayerSource::Member(2))],
        metadata_overrides: vec![],
        twt_meta: vec![],
    };
    let mut out = Vec::new();
    assert!(matches!(
        emit_collapsed_gguf(&parent, &spec, &mut out),
        Err(TwtError::GgufWrite(_))
    ));

    // Non-tiling table (gap).
    let spec = CollapseSpec {
        blocks: vec![(0, 1, LayerSource::Member(0)), (2, 3, LayerSource::Member(2))],
        metadata_overrides: vec![("qwen35twtfx.block_count".to_owned(), GgufValue::U64(2))],
        twt_meta: vec![],
    };
    let mut out = Vec::new();
    assert!(matches!(
        emit_collapsed_gguf(&parent, &spec, &mut out),
        Err(TwtError::BadBlockTable { .. })
    ));

    // Member outside its own block.
    let spec = CollapseSpec {
        blocks: vec![(0, 3, LayerSource::Member(1))],
        metadata_overrides: vec![("qwen35twtfx.block_count".to_owned(), GgufValue::U64(1))],
        twt_meta: vec![],
    };
    // member 1 IS inside [0,3) — this one must SUCCEED; use 5 instead.
    let mut out = Vec::new();
    assert!(emit_collapsed_gguf(&parent, &spec, &mut out).is_ok());
    let spec = CollapseSpec {
        blocks: vec![(0, 3, LayerSource::Member(3))],
        metadata_overrides: vec![("qwen35twtfx.block_count".to_owned(), GgufValue::U64(1))],
        twt_meta: vec![],
    };
    let mut out = Vec::new();
    assert!(matches!(
        emit_collapsed_gguf(&parent, &spec, &mut out),
        Err(TwtError::BadBlockTable { .. })
    ));

    // Incomplete merged plan (missing the norm suffix).
    let mut merged = merged_payloads(&parent, [0, 1, 2]);
    merged.remove("attn_norm.weight");
    let spec = CollapseSpec {
        blocks: vec![
            (0, 2, LayerSource::Member(0)),
            (2, 3, LayerSource::Merged(merged)),
        ],
        metadata_overrides: vec![("qwen35twtfx.block_count".to_owned(), GgufValue::U64(2))],
        twt_meta: vec![],
    };
    let mut out = Vec::new();
    match emit_collapsed_gguf(&parent, &spec, &mut out) {
        Err(TwtError::IncompletePlan { missing, .. }) => {
            assert!(missing.contains("attn_norm.weight"), "{missing}");
        }
        other => panic!("incomplete plan must refuse, got {other:?}"),
    }
}

#[test]
fn pack_to_wire_round_trips_through_the_reader() {
    // q2_0_wire_bytes output re-read as blocks == the container.
    let mut rng = 42u64;
    let mut w = TernaryGroupWeights::new(2, 256);
    for r in 0..2 {
        for g in 0..w.groups_per_row {
            w.group_scale[r * w.groups_per_row + g] = half::f16::from_f32(0.125);
        }
    }
    for r in 0..2 {
        for c in 0..256 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            w.set(r, c, ((rng >> 33) % 3) as i8 - 1);
        }
    }
    let wire = q2_0_wire_bytes(&w).unwrap();
    let mut blocks = Vec::new();
    for chunk in wire.as_chunks::<34>().0 {
        let mut b = BlockQ2_0 { d: 0, qs: [0u8; 32] };
        b.d = u16::from_le_bytes([chunk[0], chunk[1]]);
        b.qs.copy_from_slice(&chunk[2..]);
        blocks.push(b);
    }
    let mut direct = Vec::new();
    pack_ternary_group_to_q2_0(&w, &mut direct).unwrap();
    let re_packed: Vec<u8> = direct.iter().flat_map(|b| bytemuck::bytes_of(b).to_vec()).collect();
    assert_eq!(wire, re_packed);
    // Dequant agrees with the container's eval view.
    let mut deq = vec![0f32; 512];
    dequantize_row_q2_0(&blocks, &mut deq);
    let eval = riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(&w);
    assert_eq!(deq, eval);
}
