//! Issue 028 — the dual-PTQ disaggregated container + the in-process phase
//! handoff (Research 004, arXiv:2609.26333 §2.4–2.6).
//!
//! Quantize one checkpoint twice: a compact **decode** copy (resident) and a
//! higher-precision **prefill** copy. Prefill runs its prompt through the
//! prefill weights; the resulting KV + GDN recurrent state hands off
//! in-process to decode, which continues with the decode weights. The paper's
//! engine-to-engine KV+state contract collapses to ONE process for us — the
//! handoff vehicle is the existing [`HybridCache`], moved by ownership, so
//! the boundary is exact by construction (no serialization, no rounding —
//! the G1 gate pins that).
//!
//! # Container forms (T1)
//!
//! - **Two files**: any two GGUFs of the same geometry ([`Self::from_pair`]).
//!   The natural form for T4: `…-PQ2_0.gguf` decode + a q4-class prefill pack.
//! - **One file**: the bare decode tensor set plus a `.pf`-suffixed prefill
//!   set (`blk.0.attn_qkv.weight.pf` beside `blk.0.attn_qkv.weight`), read by
//!   [`Self::load_single_file`] through the suffix-parameterized loader
//!   (`load_qwen_deltanet_ternary_from_gguf`). Decode tensors keep the bare
//!   spellings, so every pre-028 file stays a valid single-copy container.
//! - **Same copy**: [`Self::from_single`] — the degenerate container (the G1
//!   plumbing arm; also the honest state before a second quantization exists).
//!
//! Load policy: **resident-only** v1 — the 27B dual posture fits both fleet
//! boxes (see the budget table), and a lazy prefill arm has no consumer yet.
//!
//! # The escape law (T2)
//!
//! The recurrence dynamics — `ssm_a` (`A_log`), `ssm_dt.bias`, and the gate
//! projections `ssm_alpha`/`ssm_beta` (`in_proj_a`/`in_proj_b`) — must be
//! **bit-identical between the two copies**. The paper freezes them shared
//! ("quantizing them destabilized training", their Appendix A.2) and our own
//! Issue-980 escape set excludes them from quantization; a disaggregated
//! container that diverges them per phase refuses loudly
//! ([`Self::verify_escape_set_shared`], run by every constructor).
//! `conv1d`/norms are deliberately NOT enforced (not on the law's list; a
//! producer diverging them is a measured-accuracy question, not a correctness
//! refusal).
//!
//! # Memory budget (T3 — 27B class, the fleet's two boxes)
//!
//! | Box | decode copy | prefill copy | total | verdict |
//! |---|---|---|---|---|
//! | 4090 (24 GB) | PQ2_0 ≈ 6.7 GB | q4_k ≈ 15 GB (T4 produces it) | ≈ 22 GB | at the edge — the T4 arm MEASURES KV headroom before choosing; q3-class prefill ≈ 12 GB = comfortable |
//! | M3 (64 GB unified) | PQ2_0 ≈ 6.7 GB | q8 prefill ≈ 27 GB | ≈ 34 GB | comfortable |
//!
//! PhaseHandoff live state: per attention layer `T × n_kv_head × head_dim × 2`
//! f32 (KV) + one GDN recurrent matrix per `DeltaNet` layer
//! (`n_v_heads × 128 × 128` f32 ≈ 4 MB/layer at 27B ≈ 194 MB total) — static
//! per instance, independent of the weight copies.
//!
//! The 4090 arm picks its prefill format from the MEASURED headroom, never
//! from this table's wish (the table is the plan; T4's bench is the record).
//!
//! # What this module does NOT do
//!
//! No accuracy claim: pairing two DIFFERENT quantizations is T4's measurement
//! (dual-PTQ recovery vs single-checkpoint at matched total storage, per
//! family — the PTQ-vs-QADD decomposition). Same-weights pairing is
//! byte-identical to the single-checkpoint path BY CONSTRUCTION (same forward
//! function, same cache, ownership-move boundary) and the G1 gate pins it.

use std::path::Path;

use crate::deltanet::forward::{HybridCache, HybridForwardScratch};
use crate::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use crate::deltanet::ternary_weights::{GateProjWeights, QwenDeltaNetTernaryWeights};
use crate::gguf_loader::{GgufFile, GgufValue};
use crate::rope::RopeFreqTable;
use crate::types::{Config, DeltaNetLayerType};

/// Tensor-name suffix marking the PREFILL copy inside a one-file dual-tensor
/// container. Decode tensors keep the bare (pre-028) spellings, so every
/// existing file loads unchanged as a single-copy container.
pub const PREFILL_TENSOR_SUFFIX: &str = ".pf";

/// blake3 over the file's GEOMETRY metadata — every `qwen35.*` key plus
/// `general.architecture`, in sorted key order, values fed canonically.
///
/// These are exactly the keys `qwen35_deltanet_config_from_gguf_metadata`
/// reads, compared at the METADATA plane where every geometry knob lives:
/// the check cannot silently miss a newly consulted key the way a
/// hand-listed Config-field compare would (a new metadata key is covered
/// the day the config derivation starts reading it).
///
/// `general.name` and the training-plane keys are deliberately excluded —
/// two quantizations of one checkpoint may legitimately name themselves
/// differently; geometry may not differ.
pub fn geometry_fingerprint(gguf: &GgufFile) -> [u8; 32] {
    let mut keys: Vec<&String> = gguf
        .metadata
        .keys()
        .filter(|k| k.as_str() == "general.architecture" || k.starts_with("qwen35."))
        .collect();
    keys.sort();
    let mut hasher = blake3::Hasher::new();
    for k in keys {
        hasher.update(k.as_bytes());
        hasher.update(&[0]);
        feed_value(gguf.metadata.get(k), &mut hasher);
    }
    hasher.finalize().into()
}

/// Canonical byte form of a GGUF metadata value (discriminant-tagged, so a
/// U64 4 and a F64 4.0 can never collide).
fn feed_value(value: Option<&GgufValue>, hasher: &mut blake3::Hasher) {
    let Some(v) = value else {
        hasher.update(&[0u8]); // MISSING
        return;
    };
    let mut le = |tag: u8, bytes: &[u8]| {
        hasher.update(&[tag]);
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    match v {
        GgufValue::U8(x) => le(1, &x.to_le_bytes()),
        GgufValue::I8(x) => le(2, &x.to_le_bytes()),
        GgufValue::U16(x) => le(3, &x.to_le_bytes()),
        GgufValue::I16(x) => le(4, &x.to_le_bytes()),
        GgufValue::U32(x) => le(5, &x.to_le_bytes()),
        GgufValue::I32(x) => le(6, &x.to_le_bytes()),
        GgufValue::U64(x) => le(7, &x.to_le_bytes()),
        GgufValue::I64(x) => le(8, &x.to_le_bytes()),
        GgufValue::F32(x) => le(9, &x.to_le_bytes()),
        GgufValue::F64(x) => le(10, &x.to_le_bytes()),
        GgufValue::Bool(x) => le(11, &[*x as u8]),
        GgufValue::String(s) => le(12, s.as_bytes()),
        GgufValue::Array(items) => {
            hasher.update(&[13]);
            hasher.update(&(items.len() as u64).to_le_bytes());
            for item in items {
                feed_value(Some(item), hasher);
            }
        }
    }
}

/// Bit-exact f32 slice equality (NaN ≠ NaN, −0 ≠ +0 — none of which belong
/// in the escape set anyway; the bit compare is the strictest reading).
fn f32_slices_bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Bit-exact ternary-container equality (planes + f16 scales by bits).
fn ternary_equal(
    a: &katgpt_core::TernaryGroupWeights,
    b: &katgpt_core::TernaryGroupWeights,
) -> bool {
    a.rows == b.rows
        && a.cols == b.cols
        && a.blocks64 == b.blocks64
        && a.groups_per_row == b.groups_per_row
        && a.pos_bits == b.pos_bits
        && a.neg_bits == b.neg_bits
        && a.group_scale.len() == b.group_scale.len()
        && a.group_scale
            .iter()
            .zip(&b.group_scale)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn gate_proj_bit_equal(a: &GateProjWeights, b: &GateProjWeights) -> bool {
    match (a, b) {
        (GateProjWeights::Ternary(x), GateProjWeights::Ternary(y)) => ternary_equal(x, y),
        (GateProjWeights::Dense(xv, xr, xc), GateProjWeights::Dense(yv, yr, yc)) => {
            xr == yr && xc == yc && f32_slices_bit_equal(xv, yv)
        }
        _ => false,
    }
}

/// One phase's weight source: the shared single set (the degenerate / G1
/// container — both phases ride ONE struct, no clone, no RAM doubling) or
/// two separately-loaded copies. Boxed: a ternary weight set is hundreds of
/// bytes of handles — the enum stays pointer-sized and moves stay cheap.
enum CopySet {
    Shared(Box<QwenDeltaNetTernaryWeights>),
    Split {
        decode: Box<QwenDeltaNetTernaryWeights>,
        prefill: Box<QwenDeltaNetTernaryWeights>,
    },
}

/// The dual-copy container: decode-resident compact weights + the
/// higher-precision prefill copy.
pub struct DisaggregatedTernaryWeights {
    copies: CopySet,
}

impl DisaggregatedTernaryWeights {
    /// The degenerate container: BOTH phases ride one weight set. The G1
    /// plumbing arm and the honest state before a second quantization exists.
    /// Escape-set sharing holds vacuously (one set) — no check needed.
    pub fn from_single(weights: QwenDeltaNetTernaryWeights) -> Self {
        Self {
            copies: CopySet::Shared(Box::new(weights)),
        }
    }

    /// Two-file container: load the decode copy + the prefill copy from two
    /// GGUFs of the same geometry (the natural T4 form: `…-PQ2_0.gguf` decode
    /// + a q4-class prefill pack). Returns the decode file's Config.
    ///
    /// The compat check runs on the METADATA plane BEFORE either copy
    /// dequantizes — a mismatched pair refuses in two header parses instead
    /// of after a 15 GB load. Refuses loudly when the geometry metadata
    /// differs, when `layer_types` differ (the phase split would route
    /// layers through the wrong forward), or when the escape set is not
    /// bit-shared (the T2 law).
    pub fn load_pair(decode_path: &Path, prefill_path: &Path) -> anyhow::Result<(Config, Self)> {
        let decode_gguf = GgufFile::open(decode_path)?;
        let prefill_gguf = GgufFile::open(prefill_path)?;
        if geometry_fingerprint(&decode_gguf) != geometry_fingerprint(&prefill_gguf) {
            anyhow::bail!(
                "disaggregated container refused: decode/prefill geometry fingerprints differ \
                 (qwen35.* metadata drift between the two files)"
            );
        }
        let (config, decode) =
            crate::gguf_loader::load_qwen_deltanet_ternary_from_gguf(&decode_gguf, "")?;
        let (_, prefill) =
            crate::gguf_loader::load_qwen_deltanet_ternary_from_gguf(&prefill_gguf, "")?;
        if decode.layer_types != prefill.layer_types {
            anyhow::bail!(
                "disaggregated container refused: decode/prefill layer_types differ \
                 (the phase split would route layers through the wrong forward)"
            );
        }
        let container = Self {
            copies: CopySet::Split {
                decode: Box::new(decode),
                prefill: Box::new(prefill),
            },
        };
        container.verify_escape_set_shared()?;
        Ok((config, container))
    }

    /// One-file container: the bare decode set + the `.pf`-suffixed prefill
    /// set in one GGUF. Refuses loudly when no `.pf` tensor set exists (a
    /// plain single-checkpoint file is a [`Self::from_single`] container, not
    /// a disaggregated one — loading it here would silently serve the decode
    /// copy in both phases while claiming disaggregation).
    pub fn load_single_file(path: &Path) -> anyhow::Result<(Config, Self)> {
        let gguf = GgufFile::open(path)?;
        let (config, decode) = crate::gguf_loader::load_qwen_deltanet_ternary_from_gguf(&gguf, "")?;
        let has_pf = gguf
            .tensor_info(&format!("token_embd.weight{PREFILL_TENSOR_SUFFIX}"))
            .is_some();
        anyhow::ensure!(
            has_pf,
            "no '.pf' prefill tensor set in '{}' — a single-checkpoint file is a \
             from_single container, not a disaggregated one",
            path.display()
        );
        let (_, prefill) =
            crate::gguf_loader::load_qwen_deltanet_ternary_from_gguf(&gguf, PREFILL_TENSOR_SUFFIX)?;
        if decode.layer_types != prefill.layer_types {
            anyhow::bail!(
                "disaggregated container refused: decode/prefill layer_types differ \
                 (the phase split would route layers through the wrong forward)"
            );
        }
        let container = Self {
            copies: CopySet::Split {
                decode: Box::new(decode),
                prefill: Box::new(prefill),
            },
        };
        container.verify_escape_set_shared()?;
        Ok((config, container))
    }

    pub fn decode(&self) -> &QwenDeltaNetTernaryWeights {
        match &self.copies {
            CopySet::Shared(w) => w,
            CopySet::Split { decode, .. } => decode,
        }
    }

    pub fn prefill(&self) -> &QwenDeltaNetTernaryWeights {
        match &self.copies {
            CopySet::Shared(w) => w,
            CopySet::Split { prefill, .. } => prefill,
        }
    }

    /// The T2 escape law: `ssm_a` (`A_log`), `ssm_dt.bias`, and the gate
    /// projections `ssm_alpha`/`ssm_beta` must be bit-identical between the
    /// copies (shared+frozen recurrence dynamics — the paper's Appendix A.2 +
    /// our Issue-980 escape set). Vacuously true on a [`CopySet::Shared`]
    /// container (one set).
    pub fn verify_escape_set_shared(&self) -> anyhow::Result<()> {
        let CopySet::Split { decode, prefill } = &self.copies else {
            return Ok(());
        };
        for (i, (d, p)) in decode.layers.iter().zip(&prefill.layers).enumerate() {
            if decode.layer_types[i] != DeltaNetLayerType::DeltaNet {
                continue; // attention layers carry empty escape fields
            }
            if !f32_slices_bit_equal(&d.a_log, &p.a_log) {
                anyhow::bail!(
                    "escape-set divergence at blk.{i}: ssm_a (A_log) differs between copies"
                );
            }
            if !f32_slices_bit_equal(&d.dt_bias, &p.dt_bias) {
                anyhow::bail!(
                    "escape-set divergence at blk.{i}: ssm_dt.bias differs between copies"
                );
            }
            if !gate_proj_bit_equal(&d.in_proj_a, &p.in_proj_a) {
                anyhow::bail!(
                    "escape-set divergence at blk.{i}: ssm_alpha (in_proj_a) differs between copies"
                );
            }
            if !gate_proj_bit_equal(&d.in_proj_b, &p.in_proj_b) {
                anyhow::bail!(
                    "escape-set divergence at blk.{i}: ssm_beta (in_proj_b) differs between copies"
                );
            }
        }
        Ok(())
    }
}

/// The in-process phase handoff: the cache + scratch + rope + position that
/// cross the prefill→decode boundary AS ONE OBJECT (the T2 contract —
/// ownership moves nothing across a serialization seam; the object itself
/// survives the phase switch, which is what makes the boundary exact).
///
/// `prefill` runs the prompt through the prefill copy; `decode_step` advances
/// one token through the decode copy. Positions advance once per token across
/// BOTH phases — the boundary is invisible to the forward.
pub struct PhaseHandoff {
    cache: HybridCache,
    scratch: HybridForwardScratch,
    rope_freq: RopeFreqTable,
    x: Vec<f32>,
    pos: usize,
    vocab: usize,
}

impl PhaseHandoff {
    /// Fresh handoff state for the given geometry (`layer_types` from the
    /// weights, mirroring `generate_greedy_qwen_deltanet_ternary`).
    pub fn begin(config: &Config, layer_types: &[DeltaNetLayerType]) -> Self {
        let n = config.n_embd;
        let v = config.vocab_size;
        Self {
            cache: HybridCache::with_layer_types(config, layer_types),
            scratch: HybridForwardScratch::new(config),
            rope_freq: RopeFreqTable::new(
                config.rope_theta,
                crate::deltanet::forward::effective_rotary_dim(config),
            ),
            x: vec![0.0; n.max(v)],
            pos: 0,
            vocab: v,
        }
    }

    /// Prefill phase: run the prompt tokens through the PREFILL copy.
    /// Returns the LAST position's logits (the first-token distribution).
    pub fn prefill(
        &mut self,
        weights: &QwenDeltaNetTernaryWeights,
        config: &Config,
        prompt_tokens: &[usize],
    ) -> &[f32] {
        for &token in prompt_tokens {
            let pos = self.pos;
            let logits = forward_qwen_deltanet_ternary(
                &mut self.x,
                weights,
                &mut self.cache,
                token,
                pos,
                config,
                &mut self.scratch,
                &self.rope_freq,
            );
            let _ = logits; // mid-prompt logits are not consumed
            self.pos += 1;
        }
        &self.x[..self.vocab]
    }

    /// Decode phase: advance one token through the DECODE copy.
    /// Returns this position's logits (the next-token distribution).
    pub fn decode_step(
        &mut self,
        weights: &QwenDeltaNetTernaryWeights,
        config: &Config,
        token: usize,
    ) -> &[f32] {
        let pos = self.pos;
        forward_qwen_deltanet_ternary(
            &mut self.x,
            weights,
            &mut self.cache,
            token,
            pos,
            config,
            &mut self.scratch,
            &self.rope_freq,
        );
        self.pos += 1;
        &self.x[..self.vocab]
    }

    /// Next position the handoff will fill (0 on a fresh handoff; after a
    /// T-token prefill, T).
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The live cache — the handoff vehicle itself, exposed for diagnostics
    /// (state inspection / eviction experiments); production callers stay on
    /// the phase methods so the boundary stays the only write path.
    pub fn cache(&mut self) -> &mut HybridCache {
        &mut self.cache
    }
}

/// Greedy generation over the disaggregated container — the phase-split
/// mirror of `generate_greedy_qwen_deltanet_ternary` (same comparator, same
/// tie behavior; the argmax consumer of the G1 gate).
pub fn generate_greedy_disaggregated(
    weights: &DisaggregatedTernaryWeights,
    config: &Config,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let mut generated = Vec::with_capacity(max_tokens);
    if prompt_tokens.is_empty() {
        return generated;
    }

    let layer_types = if weights.decode().layer_types.is_empty() {
        vec![DeltaNetLayerType::Attention; config.n_layer]
    } else {
        weights.decode().layer_types.clone()
    };
    let mut handoff = PhaseHandoff::begin(config, &layer_types);

    // Prefill phase — the prefill copy consumes the whole prompt.
    let last = handoff
        .prefill(weights.prefill(), config, prompt_tokens)
        .to_vec();

    // The phase boundary: argmax the prefill copy's last logits → first
    // generated token, then decode continues on the decode copy.
    let first_token = last
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map_or(0, |(i, _)| i);
    generated.push(first_token);

    for _ in 1..max_tokens {
        let current = *generated.last().unwrap();
        let logits = handoff
            .decode_step(weights.decode(), config, current)
            .to_vec();
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i);
        generated.push(next);
    }

    generated
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suffix law: bare names unchanged, `.pf` appended for the prefill
    /// lookups; the constant is the ONLY suffix the loader mode accepts.
    #[test]
    fn prefill_suffix_is_dot_pf() {
        assert_eq!(PREFILL_TENSOR_SUFFIX, ".pf");
    }

    /// The canonical value feed is discriminant-tagged: the same numeric
    /// value under different GGUF types must fingerprint differently (a
    /// U64 4 and an F64 4.0 are different metadata claims).
    #[test]
    fn feed_value_tags_the_discriminant() {
        let mut a = blake3::Hasher::new();
        let mut b = blake3::Hasher::new();
        feed_value(Some(&GgufValue::U64(4)), &mut a);
        feed_value(Some(&GgufValue::F64(4.0)), &mut b);
        assert_ne!(a.finalize(), b.finalize());

        let mut c = blake3::Hasher::new();
        let mut d = blake3::Hasher::new();
        feed_value(Some(&GgufValue::U64(4)), &mut c);
        feed_value(Some(&GgufValue::U64(4)), &mut d);
        assert_eq!(c.finalize(), d.finalize(), "same value must be stable");

        // MISSING is its own state, distinct from every present value.
        let mut e = blake3::Hasher::new();
        feed_value(None, &mut e);
        assert_ne!(e.finalize(), c.finalize());
    }
}
