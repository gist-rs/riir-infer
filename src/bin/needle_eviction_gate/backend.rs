//! Model backends for the needle eviction gate — riir-infer Issue 1180 T2b.
//!
//! The gate was qwen35-hybrid only; T2b needs the same rig on a SinkHeavy
//! checkpoint (gemma-2-2b-it f16, Bench 953). Dispatch is on the GGUF's
//! `general.architecture`:
//!
//! - `qwen35*` / anything else → the original qwen35-hybrid lane, byte-for-byte
//!   the calls the gate made before the split (chunked prefill +
//!   `forward_qwen_deltanet_evictable`, GPT-2 BPE, no BOS) — the Bench 947
//!   numbers stay reproducible.
//! - `gemma2` → `forward_gemma2_f16_evictable` (every layer full attention),
//!   SentencePiece tokenizer, BOS at position 0 (the serving path's sink
//!   token), token-by-token prefill.

use std::path::Path;

use anyhow::{Context, Result};
use riir_infer_core::deltanet::kv_evict::EvictorState;
use riir_infer_core::deltanet::weights::QwenDeltaNetWeights;
use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, PrefillContext, effective_rotary_dim,
    forward_qwen_deltanet_evictable, prefill_qwen_deltanet_chunk_into,
};
use riir_infer_core::gemma_layer::GemmaTransformerWeightsF16;
use riir_infer_core::gguf_loader::{
    GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct, load_qwen_deltanet_weights_gguf,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::{BpeTokenizer, SentencePieceGgufTokenizer};
use riir_infer_core::transformer::{
    ForwardContext, MultiLayerKVCache, forward_gemma2_f16_evictable,
};
use riir_infer_core::types::{Config, DeltaNetLayerType};

/// The rig's tokenizer, per backend. One instance per process — the
/// variant size gap is irrelevant.
#[allow(clippy::large_enum_variant)]
pub enum Tok {
    Bpe(BpeTokenizer),
    Sp(SentencePieceGgufTokenizer),
}

impl Tok {
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            Tok::Bpe(t) => t.encode(text),
            Tok::Sp(t) => t.encode(text),
        }
    }

    pub fn eos_id(&self) -> usize {
        match self {
            Tok::Bpe(t) => t.eos_id(),
            Tok::Sp(t) => t.eos_id(),
        }
    }
}

/// A loaded model body.
pub enum Model {
    Qwen {
        config: Config,
        weights: QwenDeltaNetWeights,
        rope_freq: Option<RopeFreqTable>,
    },
    Gemma2 {
        config: Config,
        weights: GemmaTransformerWeightsF16,
    },
}

impl Model {
    /// Load the body + tokenizer, dispatching on `general.architecture`.
    pub fn load(path: &Path) -> Result<(Model, Tok)> {
        let t0 = std::time::Instant::now();
        let gguf = GgufFile::open(path).with_context(|| format!("open {}", path.display()))?;
        if gguf.architecture() == Some("gemma2") {
            let config = config_from_gguf_metadata(&gguf)?;
            let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)
                .context("SentencePiece tokenizer from gguf")?;
            let weights = load_gemma2_f16_direct(&gguf, &config)
                .with_context(|| format!("load gemma2 f16 {}", path.display()))?;
            println!(
                "# model: gemma2 f16 (all layers full attention) | layers={} n_embd={} n_head={} n_kv_head={} head_dim={} vocab={} softcap={} | load {:.1}s",
                config.n_layer,
                config.n_embd,
                config.n_head,
                config.n_kv_head,
                config.head_dim,
                config.vocab_size,
                config.attn_logit_softcapping,
                t0.elapsed().as_secs_f32()
            );
            return Ok((Model::Gemma2 { config, weights }, Tok::Sp(tok)));
        }
        drop(gguf);
        // The original qwen35-hybrid lane — the same calls, the same order.
        let (config, weights) = load_qwen_deltanet_weights_gguf(path)
            .with_context(|| format!("load {}", path.display()))?;
        println!(
            "# model: qwen35 hybrid | layers={} n_embd={} n_head={} n_kv_head={} head_dim={} vocab={} | load {:.1}s",
            config.n_layer,
            config.n_embd,
            config.n_head,
            config.n_kv_head,
            config.head_dim,
            config.vocab_size,
            t0.elapsed().as_secs_f32()
        );
        let tok = {
            let gguf = GgufFile::open(path).context("re-open gguf for tokenizer")?;
            BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
        };
        Ok((
            Model::Qwen {
                config,
                weights,
                rope_freq: None,
            },
            Tok::Bpe(tok),
        ))
    }

    /// Size the cache ceiling (and build the qwen RoPE table, which the
    /// original lane built right after this assignment).
    pub fn set_block_size(&mut self, block_size: usize) {
        match self {
            Model::Qwen {
                config, rope_freq, ..
            } => {
                config.block_size = block_size;
                *rope_freq = Some(RopeFreqTable::new(
                    config.rope_theta,
                    effective_rotary_dim(config),
                ));
            }
            Model::Gemma2 { config, .. } => config.block_size = block_size,
        }
    }

    pub fn config(&self) -> &Config {
        match self {
            Model::Qwen { config, .. } | Model::Gemma2 { config, .. } => config,
        }
    }

    /// The token the serving path puts at position 0 (gemma's BOS — the sink
    /// usually IS that token); `None` = the qwen lane (no BOS).
    pub fn bos(&self, tok: &Tok) -> Option<usize> {
        match (self, tok) {
            (Model::Gemma2 { .. }, Tok::Sp(t)) => Some(t.bos_id()),
            _ => None,
        }
    }

    /// Which layers carry KV (and so eviction state).
    pub fn layer_attn(&self) -> Vec<bool> {
        match self {
            Model::Qwen { weights, .. } => weights
                .layer_types
                .iter()
                .map(|&t| t == DeltaNetLayerType::Attention)
                .collect(),
            Model::Gemma2 { config, .. } => vec![true; config.n_layer],
        }
    }

    /// The sink-audit census over the first `n` tokens (Issue 981 T2 /
    /// Issue 1180 T1 lanes — the same code the `sink_audit` bin runs).
    #[cfg(feature = "sink_audit")]
    pub fn census(
        &self,
        tokens: &[usize],
        n: usize,
    ) -> Result<riir_infer_core::deltanet::sink_audit::SinkCensus, String> {
        match self {
            Model::Qwen {
                config,
                weights,
                rope_freq,
            } => riir_infer_core::deltanet::sink_audit::run_census(
                weights,
                config,
                tokens,
                rope_freq.as_ref().expect("set_block_size before census"),
                n,
            ),
            Model::Gemma2 { config, weights } => {
                riir_infer_core::transformer::sink_audit_dense::run_census_dense(
                    config,
                    tokens,
                    n,
                    |ctx, cache, tok, pos, cfg| {
                        riir_infer_core::transformer::forward_gemma2_f16(
                            ctx, weights, cache, tok, pos, cfg,
                        );
                    },
                )
            }
        }
    }

    /// Fresh per-arm forward state.
    pub fn session(&self, chunk: usize) -> Session {
        match self {
            Model::Qwen {
                config, weights, ..
            } => {
                let v = config.vocab_size;
                Session::Qwen {
                    cache: HybridCache::with_layer_types(config, &weights.layer_types),
                    scratch: HybridForwardScratch::new(config),
                    pctx: PrefillContext::new(config, chunk),
                    x: vec![0.0f32; v.max(config.n_embd)],
                }
            }
            Model::Gemma2 { config, .. } => Session::Gemma2 {
                ctx: Box::new(ForwardContext::new(config)),
                cache: MultiLayerKVCache::new(config),
            },
        }
    }
}

/// Per-arm forward state. One live instance per arm — the variant size gap
/// is irrelevant.
#[allow(clippy::large_enum_variant)]
pub enum Session {
    Qwen {
        cache: HybridCache,
        scratch: HybridForwardScratch,
        pctx: PrefillContext,
        x: Vec<f32>,
    },
    Gemma2 {
        ctx: Box<ForwardContext>,
        cache: MultiLayerKVCache,
    },
}

impl Session {
    /// Prefill `prompt` (logical positions `0..len`); the last position's
    /// logits land in `logits`. `progress(done_tokens)` fires per chunk.
    pub fn prefill(
        &mut self,
        model: &Model,
        prompt: &[usize],
        chunk: usize,
        evictor: &mut EvictorState,
        logits: &mut [f32],
        mut progress: impl FnMut(usize),
    ) {
        match (self, model) {
            (
                Session::Qwen {
                    cache,
                    scratch,
                    pctx,
                    ..
                },
                Model::Qwen {
                    config,
                    weights,
                    rope_freq,
                },
            ) => {
                let rope_freq = rope_freq.as_ref().expect("set_block_size before prefill");
                for (c, part) in prompt.chunks(chunk).enumerate() {
                    prefill_qwen_deltanet_chunk_into(
                        weights,
                        config,
                        cache,
                        part,
                        c * chunk,
                        scratch,
                        rope_freq,
                        pctx,
                        logits,
                        Some(&mut *evictor),
                        None,
                    );
                    progress(c * chunk + part.len());
                }
            }
            (Session::Gemma2 { ctx, cache }, Model::Gemma2 { config, weights }) => {
                // Token-by-token through the armed decode body (no gemma
                // chunked prefill exists; at ≤ 8K this is the honest path).
                for (pos, &t) in prompt.iter().enumerate() {
                    forward_gemma2_f16_evictable(ctx, weights, cache, t, pos, config, evictor);
                    if (pos + 1) % chunk == 0 {
                        progress(pos + 1);
                    }
                }
                let v = config.vocab_size;
                logits[..v].copy_from_slice(&ctx.logits[..v]);
                progress(prompt.len());
            }
            _ => unreachable!("session/model backend mismatch"),
        }
    }

    /// One armed decode step at logical `pos`; returns the logits.
    pub fn step<'a>(
        &'a mut self,
        model: &Model,
        token: usize,
        pos: usize,
        evictor: &mut EvictorState,
    ) -> &'a [f32] {
        match (self, model) {
            (
                Session::Qwen {
                    cache, scratch, x, ..
                },
                Model::Qwen {
                    config,
                    weights,
                    rope_freq,
                },
            ) => {
                let rope_freq = rope_freq.as_ref().expect("set_block_size before step");
                let v = config.vocab_size;
                let out = forward_qwen_deltanet_evictable(
                    x, weights, cache, token, pos, config, scratch, rope_freq, evictor,
                );
                &out[..v]
            }
            (Session::Gemma2 { ctx, cache }, Model::Gemma2 { config, weights }) => {
                let v = config.vocab_size;
                let out =
                    forward_gemma2_f16_evictable(ctx, weights, cache, token, pos, config, evictor);
                &out[..v]
            }
            _ => unreachable!("session/model backend mismatch"),
        }
    }
}
