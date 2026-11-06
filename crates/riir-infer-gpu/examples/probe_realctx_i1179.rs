//! Issue 1179 real-context parity probe (verdict conditions 1+2 at the
//! REAL 603-row prompt regime — the KV=4 fixture was non-adjudicating: both
//! engines degenerate there, bf16 to repeat-chains, f32 to varied chains).
//!
//! Produces, at the REAL context:
//!   /tmp/i1179_prompt_feats.f32   the target's captured features
//!                                 [603][5][5120] f32 (layer-major)
//!   /tmp/i1179_tprev.txt          the target's greedy sample at the last
//!                                 prompt position (their probe's t_prev)
//!   /tmp/i1179_my_chain.txt       MY draft's argmax chain (15 tokens)
//!   /tmp/i1179_my_layers.bin      MY per-layer dump (see the binary layout
//!                                 printed below)
//!
//! The uzu twin (dflash_realctx_probe test) reads the first three and
//! produces their chain at the same state; the chains + per-layer values
//! decide fidelity at real context.

#![cfg(feature = "qwen38_specdec")]

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;
use riir_infer_gpu::qwen38_specdec::{DraftLayersDump, DFlashCpu, StFile, parse_vendor_config};

const CAPTURE_LAYERS: &[usize] = &[5, 19, 33, 47, 61];

fn main() {
    let fixture =
        std::fs::read_to_string("/tmp/i1179_fixture_prompt.txt").unwrap();
    let spec = StFile::open(std::path::Path::new(
        "/Users/katopz/uzu27b/speculator/model.safetensors",
    ))
    .unwrap();
    let cfg = parse_vendor_config(
        &std::fs::read_to_string("/Users/katopz/uzu27b/speculator/config.json")
            .unwrap(),
    )
    .unwrap();
    let mut draft = DFlashCpu::load(&spec, &cfg).unwrap();

    let gguf_path = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";
    let mut fwd = Qwen38WgpuForward::new(
        std::path::Path::new(gguf_path),
        8192,
        512,
    )
    .unwrap();
    let gg = GgufFile::open(std::path::Path::new(gguf_path)).unwrap();

    let tok = BpeTokenizer::from_gguf(&gg).unwrap();
    let prompt: Vec<u32> = tok
        .encode(&fixture)
        .into_iter()
        .map(|t| t as u32)
        .collect();
    println!("[realctx] prompt tokens {}", prompt.len());

    // Prefill + capture the draft taps at every prompt position.
    let (feats, last_logits) =
        fwd.ingest_capture(&prompt, 0, CAPTURE_LAYERS).unwrap();
    let d = 5120usize;
    let captured = CAPTURE_LAYERS.len();
    let mut packed = vec![0.0f32; prompt.len() * d * captured];
    for (li, f) in feats.iter().enumerate() {
        for t in 0..prompt.len() {
            packed[t * d * captured + li * d..t * d * captured + (li + 1) * d]
                .copy_from_slice(&f[t * d..(t + 1) * d]);
        }
    }
    // layer-major dump for the uzu twin (it packs per-buffer)
    let mut layer_major = vec![0.0f32; prompt.len() * d * captured];
    for (li, f) in feats.iter().enumerate() {
        layer_major[li * prompt.len() * d..(li + 1) * prompt.len() * d]
            .copy_from_slice(f);
    }
    std::fs::write(
        "/tmp/i1179_prompt_feats.f32",
        bytemuck::cast_slice(&layer_major),
    )
    .unwrap();

    // The target's greedy sample at the last prompt position (first-index ties).
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (v, &x) in last_logits.iter().enumerate() {
        if x > bv {
            bv = x;
            best = v as u32;
        }
    }
    std::fs::write("/tmp/i1179_tprev.txt", format!("{best}")).unwrap();
    println!("[realctx] t_prev (target greedy at last prompt pos) = {best}");

    // MY draft at the same state: accept the full prompt, then ONE block
    // forward from t_prev.
    draft.accept(&packed, prompt.len()).unwrap();
    println!("[realctx] draft ctx_len = {}", draft.context_length());

    // Accept-path parity dump: MY pre-norm/rope projected KV for the first
    // 4 rows — compared with the uzu twin's I1179_DUMP_KVPROJ dump of their
    // state_kv_projection output (same packed feed, same weights).
    let kvproj = draft.accept_kvproj_debug(&packed, 4).unwrap();
    std::fs::write("/tmp/i1179_my_kvproj.f32", bytemuck::cast_slice(&kvproj)).unwrap();
    println!("[realctx] wrote /tmp/i1179_my_kvproj.f32 (4 rows x 5 layers x 2048)");
        let cached_k = draft.dump_cached_k(0, 0, 1);
    std::fs::write("/tmp/i1179_my_cachedk.f32", bytemuck::cast_slice(&cached_k)).unwrap();
    println!("[realctx] wrote /tmp/i1179_my_cachedk.f32 (L0 row0 K, 1024 = 8 heads x 128)");

    let t_prev =
        gg.dequant_tensor_row("token_embd.weight", best as usize, d).unwrap();
    let mask = gg
        .dequant_tensor_row("token_embd.weight", 248070, d)
        .unwrap();

    let dump: DraftLayersDump =
        draft.draft_layers_dump(&t_prev, &mask).unwrap();

    // lm_head f32 table, argmax chain (first-index ties).
    let vocab = fwd.config.vocab_size;
    let e = fwd.config.n_embd;
    let mut lm_head = vec![0.0f32; vocab * e];
    for v in 0..vocab {
        let row = gg.dequant_tensor_row("output.weight", v, e).unwrap();
        lm_head[v * e..(v + 1) * e].copy_from_slice(&row);
    }
    let mut chain = Vec::new();
    for r in 1..16 {
        let row = &dump.final_hidden[r * d..(r + 1) * d];
        let mut best = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for v in 0..vocab {
            let lrow = &lm_head[v * e..(v + 1) * e];
            let mut acc = 0.0f32;
            for i in 0..e {
                acc += lrow[i] * row[i];
            }
            if acc > bv {
                bv = acc;
                best = v as u32;
            }
        }
        chain.push(best);
    }
    println!("[realctx] my chain: {chain:?}");
    std::fs::write(
        "/tmp/i1179_my_chain.txt",
        chain.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "),
    )
    .unwrap();

    // Binary layer dump layout (printed for the twin):
    //   u32 magic 'I9L1', u32 layers, u32 rows, u32 d
    //   then per layer: conv_pre [rows][d], q, k, v, shortcut, layer_out
    //   then final_hidden [rows][d]
    let rows = cfg.draft.block_size;
    let layers = dump.conv_pre.len();
    let qh = 32usize; // DRAFT_HEADS
    let kvh = 8usize; // DRAFT_GROUPS
    let q_dim = qh * 128;
    let kv_dim = kvh * 128;
    let mut out = Vec::new();
    out.extend_from_slice(b"I9L1");
    out.extend_from_slice(&(layers as u32).to_le_bytes());
    out.extend_from_slice(&(rows as u32).to_le_bytes());
    out.extend_from_slice(&(d as u32).to_le_bytes());
    for li in 0..layers {
        // Sections: conv_pre [rows][d], q [rows][q_dim], k [rows][kv_dim],
        // v [rows][kv_dim], shortcut [rows][d], layer_out [rows][d].
        let sections: [(&Vec<f32>, usize); 6] = [
            (&dump.conv_pre[li], rows * d),
            (&dump.q[li], rows * q_dim),
            (&dump.k[li], rows * kv_dim),
            (&dump.v[li], rows * kv_dim),
            (&dump.shortcut[li], rows * d),
            (&dump.layer_out[li], rows * d),
        ];
        for (sec, expect) in sections {
            assert_eq!(sec.len(), expect, "section size");
            out.extend_from_slice(bytemuck::cast_slice(sec));
        }
    }
    out.extend_from_slice(bytemuck::cast_slice(&dump.final_hidden));
    std::fs::write("/tmp/i1179_my_layers.bin", &out).unwrap();
    let _ = &cfg;
    println!(
        "[realctx] wrote /tmp/i1179_my_layers.bin ({} layers x 6 sections x {rows}x{d})",
        dump.conv_pre.len()
    );
    // qkvg-stage parity: MY raw qkvg projection output on conv_pre[0].
    {
        let rows16 = dump.conv_pre[0].len() / d;
        let qkvg = draft.dump_qkvg(0, &dump.conv_pre[0], rows16);
        std::fs::write("/tmp/i1179_my_qkvg.f32", bytemuck::cast_slice(&qkvg)).unwrap();
        println!("[realctx] wrote /tmp/i1179_my_qkvg.f32 (L0 qkvg out on conv_pre[0])");
    }


    // Mixer-parity: MY L0 attention output on conv_pre[0] (their side runs
    // mixer.encode on the same input + same state).
    {
        let rows16 = dump.conv_pre[0].len() / d;
        let positions: Vec<usize> = (603..603 + rows16).collect();
        let (cos_t, sin_t) = draft.rope_tables_for(&positions);
        let attn_out = draft.dump_attention_out(0, &dump.conv_pre[0], rows16, &cos_t, &sin_t);
        std::fs::write("/tmp/i1179_my_attnout.f32", bytemuck::cast_slice(&attn_out.0)).unwrap();
        std::fs::write("/tmp/i1179_my_preproj.f32", bytemuck::cast_slice(&attn_out.1)).unwrap();
        println!("[realctx] wrote /tmp/i1179_my_attnout.f32 + my_preproj.f32 (L0 attention, post/pre out_proj)");
    }

}
