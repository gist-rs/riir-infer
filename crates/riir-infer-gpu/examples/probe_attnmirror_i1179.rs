//! Issue 1179 bisect: a faithful f32 mirror of uzu's CPU attention path
//! (prepare: head rmsnorm→rope→[sign,hadamard,int8 quant]; compute: i8 dot
//! × qs × ks × scale, online softmax, v dequant (code-128)·vs, finish
//! untransform) on the dumped conv_pre output — compared against my
//! attention result (77.28) and theirs (42.75).

use riir_infer_gpu::qwen38_specdec::{StFile, parse_vendor_config};

const HEADS: usize = 32;
const GROUPS: usize = 8;
const HD: usize = 128;
const SCALE: f32 = 0.08838834764831845;

fn hadamard32(v: &mut [f32]) {
    let mut s = 1usize;
    while s < v.len() {
        for lane in 0..v.len() {
            if lane & s == 0 {
                let a = v[lane];
                let b = v[lane | s];
                v[lane] = a + b;
                v[lane | s] = a - b;
            }
        }
        s <<= 1;
    }
    let sc = 1.0 / (v.len() as f32).sqrt();
    for x in v.iter_mut() {
        *x *= sc;
    }
}

fn sign(channel: usize) -> f32 {
    // kv_layout::ROTATION_SIGN_BITS — need the real table; start with +1 and
    // note the assumption.
    1.0
}

fn quantize(mut values: Vec<f32>) -> (Vec<i8>, f32) {
    for (c, x) in values.iter_mut().enumerate() {
        *x *= sign(c);
    }
    hadamard32(&mut values);
    let max = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
    let codes = values.iter().map(|v| (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8).collect();
    (codes, scale)
}

fn main() {
    let _spec = StFile::open(std::path::Path::new("/Users/katopz/uzu27b/speculator/model.safetensors")).unwrap();
    let _cfg = parse_vendor_config(&std::fs::read_to_string("/Users/katopz/uzu27b/speculator/config.json").unwrap()).unwrap();

    // The dumped conv_pre output (bf16-rounded by the uzu probe path, f32 here).
    let conv: Vec<f32> = bytemuck::cast_slice(&std::fs::read("/tmp/i1179_uzu_s1_premix.f32").unwrap()).to_vec();
    let _ = conv; // (the premix — conv applied below via the mirror? no: we feed MY conv out)
    let conv_out: Vec<f32> = {
        // Reconstruct: the uzu s3 fed THEIR premix + my conv... simplest: use
        // the same premix + my conv weights were not dumped in f32-mirror
        // form — instead reuse the fact that my conv == their-premix-based
        // conv (verified 69.07 == 69.05). Recompute the conv here from the
        // premix + the kernel proj + weights dumps.
        let premix: Vec<f32> = bytemuck::cast_slice(&std::fs::read("/tmp/i1179_uzu_s1_premix.f32").unwrap()).to_vec();
        let kproj: Vec<f32> = bytemuck::cast_slice(&std::fs::read("/tmp/i1179_l0_kernel_proj.f32").unwrap()).to_vec();
        let cw: Vec<f32> = bytemuck::cast_slice(&std::fs::read("/tmp/i1179_conv_weights.f32").unwrap()).to_vec();
        let pre_w = &cw[0..5120 * 2];
        let rows = 16usize;
        let d = 5120usize;
        let coef_count = 2 * (d / 16);
        let groups = d / 16;
        let mut coefs = vec![0.0f32; rows * 2 * coef_count];
        for r in 0..rows {
            let xr = &premix[r * d..(r + 1) * d];
            for o in 0..2 * coef_count {
                let wr = &kproj[o * d..(o + 1) * d];
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += wr[i] * xr[i];
                }
                coefs[r * 2 * coef_count + o] = acc;
            }
        }
        let mut out = vec![0.0f32; rows * d];
        for t in 0..rows {
            for c in 0..d {
                let g = c / 16;
                let mut acc = 0.0f32;
                for j in 0..2usize.min(t + 1) {
                    let w = pre_w[c * 2 + (1 - j)];
                    let delta = coefs[t * 2 * coef_count + j * groups + g];
                    acc += premix[(t - j) * d + c] * (w + delta);
                }
                out[t * d + c] = acc;
            }
        }
        out
    };

    // Load the L0 attention weights (qkvg + out_proj folded, query/key norm
    // scales) via the module loader.
    let spec = StFile::open(std::path::Path::new("/Users/katopz/uzu27b/speculator/model.safetensors")).unwrap();
    let d = 5120usize;
    // qkvg folded W — recompute via the module's loader? private. Use the
    // dump: kernel proj was dumped; qkvg not. Load via a small inline fold:
    let (wq_hat, qin, qout) = {
        // inline: read raw quant + signs, dequant, fold — same as load_linear
        use std::collections::BTreeMap;
        let p = "/Users/katopz/uzu27b/speculator/model.safetensors";
        let mut f = std::fs::File::open(p).unwrap();
        let mut lb = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut lb).unwrap();
        let hl = u64::from_le_bytes(lb) as usize;
        let mut hb = vec![0u8; hl];
        std::io::Read::read_exact(&mut f, &mut hb).unwrap();
        let hdr: serde_json::Value = serde_json::from_slice(&hb).unwrap();
        let data_start = 8 + hl;
        let mut files = BTreeMap::new();
        for (k, v) in hdr.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let off = v["data_offsets"][0].as_u64().unwrap() as usize + data_start;
            let end = v["data_offsets"][1].as_u64().unwrap() as usize + data_start;
            files.insert(k.clone(), (off, end));
        }
        let read_f32 = |name: &str| -> Vec<f32> {
            let (s, e) = files[name];
            let mut buf = vec![0u8; e - s];
            use std::io::{Read, Seek, SeekFrom};
            f.seek(SeekFrom::Start(s as u64)).unwrap();
            f.read_exact(&mut buf).unwrap();
            match hdr[name]["dtype"].as_str().unwrap() {
                "F32" => bytemuck::cast_slice(&buf).to_vec(),
                "BF16" => buf.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
                "I32" => bytemuck::cast_slice::<_, i32>(&buf).iter().map(|&x| x as f32).collect(),
                other => panic!("dtype {other}"),
            }
        };
        let read_u8 = |name: &str| -> Vec<u8> {
            let (s, e) = files[name];
            let mut buf = vec![0u8; e - s];
            use std::io::{Read, Seek, SeekFrom};
            f.seek(SeekFrom::Start(s as u64)).unwrap();
            f.read_exact(&mut buf).unwrap();
            buf
        };
        let prefix = "speculator.draft_model.layers.0.mixer.qkvg_projection";
        let out_dim = 6144usize;
        let inn = 5120usize;
        let codes = read_u8(&format!("{prefix}.weights.quantized.weights"));
        let scales = read_f32(&format!("{prefix}.weights.quantized.scales"));
        let zps = read_u8(&format!("{prefix}.weights.quantized.zero_points"));
        let in_signs = read_f32(&format!("{prefix}.weights.incoherence_signs.input_signs"));
        let out_signs = read_f32(&format!("{prefix}.weights.incoherence_signs.output_signs"));
        let mut w = vec![0.0f32; out_dim * inn];
        for o in 0..out_dim {
            for i in 0..inn {
                let byte = codes[o * (inn / 2) + i / 2];
                let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let g = i / 32;
                let scale = scales[g * out_dim + o];
                let zp_flat = g * (out_dim / 2) + o / 2;
                let zp_byte = zps[zp_flat];
                let zp = if o % 2 == 0 { zp_byte & 0x0F } else { zp_byte >> 4 };
                w[o * inn + i] = scale * (nib as f32 - zp as f32);
            }
        }
        // fold
        let mut h32 = vec![0.0f32; 1024];
        {
            // build hadamard32
            let mut hh = vec![1.0f32; 1];
            while hh.len() < 32 {
                let n = hh.len();
                let mut nx = vec![0.0f32; n * 2];
                for (r, row) in hh.chunks(n).enumerate() {
                    for c in 0..n {
                        let v = row[c];
                        nx[r * 2 * n + c] = v;
                        nx[r * 2 * n + n + c] = v;
                        nx[(r + n) * 2 * n + c] = v;
                        nx[(r + n) * 2 * n + n + c] = -v;
                    }
                }
                hh = nx;
            }
            let s = 1.0 / 32.0f32.sqrt();
            for x in hh.iter_mut() {
                *x *= s;
            }
            h32 = hh;
        }
        for blk in 0..inn / 32 {
            let c0 = blk * 32;
            for o in 0..out_dim {
                let row = &mut w[o * inn..(o + 1) * inn];
                let old: Vec<f32> = row[c0..c0 + 32].to_vec();
                for j in 0..32 {
                    let mut acc = 0.0f32;
                    for k2 in 0..32 {
                        acc += old[k2] * h32[j * 32 + k2];
                    }
                    row[c0 + j] = acc * in_signs[c0 + j];
                }
            }
        }
        {
            let mut block = vec![0.0f32; 32 * inn];
            for blk in 0..out_dim / 32 {
                let r0 = blk * 32;
                for r in 0..32 {
                    block[r * inn..(r + 1) * inn].copy_from_slice(&w[(r0 + r) * inn..(r0 + r + 1) * inn]);
                }
                for r in 0..32 {
                    for c in 0..inn {
                        let mut acc = 0.0f32;
                        for k2 in 0..32 {
                            acc += h32[r * 32 + k2] * block[k2 * inn + c];
                        }
                        w[(r0 + r) * inn + c] = acc;
                    }
                }
            }
        }
        for r in 0..out_dim {
            let sgn = out_signs[r];
            if sgn != 1.0 {
                for c in 0..inn {
                    w[r * inn + c] *= sgn;
                }
            }
        }
        (w, (), ())
    };
    let _ = (qin, qout);

    // qkvg on the conv output
    let rows = 16usize;
    let mut qkvg = vec![0.0f32; rows * 6144];
    for r in 0..rows {
        let xr = &conv_out[r * d..(r + 1) * d];
        for o in 0..6144 {
            let wr = &wq_hat[o * d..(o + 1) * d];
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += wr[i] * xr[i];
            }
            qkvg[r * 6144 + o] = acc;
        }
    }
    let mut vmax = 0.0f32;
    for r in 0..rows {
        for i in 5120..6144 {
            vmax = vmax.max(qkvg[r * 6144 + i].abs());
        }
    }
    println!("[mirror] V rows max|.| = {vmax:.3}");
    let mut qmax = 0.0f32;
    for r in 0..rows {
        for i in 0..4096 {
            qmax = qmax.max(qkvg[r * 6144 + i].abs());
        }
    }
    println!("[mirror] Q rows max|.| = {qmax:.3}");
    println!("[mirror] V per-row max: {:?}", (0..4).map(|r| {
        (0..1024).fold(0.0f32, |m, i| m.max(qkvg[r * 6144 + 5120 + i].abs()))
    }).collect::<Vec<_>>());
}
