//! The clef-text loaders — plan 619 T1.3. Three materials over the
//! verified artifact tree ([`crate::lane::artifacts`]):
//!
//! - [`ClefConfig`] — `config.json`, parsed typed and VALIDATED: the
//!   buckets must be sorted ascending and drawn from the pinned set
//!   {256, 512, 1024, 2048}; `max_questions` / `max_options` must equal
//!   the pinned 16 / 96; `tied_embeddings` must be true (the head's
//!   lexical input is the tied embedding rows — a false value would make
//!   this table semantically wrong for the head, so it refuses here);
//!   the dimensions must be positive. The artifact is sha256-pinned, so
//!   a validation refusal means pin/loader drift — always loud, never a
//!   silent re-parse.
//! - [`ClefTokenizer`] — `tokenizer.json` through the `tokenizers`
//!   crate: the exact-reference path (the same file the reference's
//!   fast backend drives — the laya posture, no hand port). Encodes with
//!   `add_special_tokens = false`: the record encoder (T1.4) carries its
//!   own `<|im_start|>` markers in the prompt text, so adding specials
//!   would desync the token stream.
//! - [`EmbeddingTable`] — `embeddings.f16`, the tied fp16 `[vocab,
//!   hidden]` matrix, mmapped (zero-copy rows; the file is never
//!   buffered) and size-walled at open: bytes == vocab·hidden·2, the
//!   plan's own cross-check of config against the weight file.
//!
//! [`ClefModel::ensure_and_load`] is the one-call entry (artifact law
//! first, then all three materials); [`ClefModel::load_from`] is the
//! pure load over an already-verified tree (the gates use it).

use std::path::Path;

use memmap2::Mmap;

/// The bucket lengths the artifact was exported with (`functionName`
/// `L{bucket}` per bucket — Research 012 §2).
pub const BUCKETS_ALLOWED: [u32; 4] = [256, 512, 1024, 2048];

/// The pinned question/option capacities (`config.json` carries them;
/// the loader pins them so a re-exported artifact cannot silently widen
/// the head's input shapes under an un-updated loader).
pub const PINNED_MAX_QUESTIONS: u32 = 16;
pub const PINNED_MAX_OPTIONS: u32 = 96;

/// The parsed + validated `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClefConfig {
    pub model: String,
    #[serde(default)]
    pub base_model: String,
    #[serde(default)]
    pub teacher: String,
    pub hidden_size: u32,
    pub head_dim: u32,
    pub rope_theta: f64,
    pub pad_id: u32,
    pub vocab_size: u32,
    pub buckets: Vec<u32>,
    pub max_questions: u32,
    pub max_options: u32,
    pub decoder: String,
    pub head: String,
    pub tied_embeddings: bool,
}

impl ClefConfig {
    /// Read + parse + validate `config.json` under `dir`.
    pub fn from_dir(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| std::io::Error::other(format!("clef config: read {}: {e}", path.display())))?;
        let cfg: Self = serde_json::from_str(&text).map_err(|e| {
            std::io::Error::other(format!("clef config: parse {}: {e}", path.display()))
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// The plan T1.3 validation law. Every refusal names the field and
    /// both sides (got vs pinned) — the sha256 pin makes these
    /// unreachable for the real artifact, so any hit is drift.
    pub fn validate(&self) -> std::io::Result<()> {
        if self.buckets.is_empty() {
            return Err(std::io::Error::other("clef config: buckets empty"));
        }
        if !self
            .buckets
            .windows(2)
            .all(|w| w[0] < w[1])
        {
            return Err(std::io::Error::other(format!(
                "clef config: buckets {:?} not strictly ascending (sorted) — refusing",
                self.buckets
            )));
        }
        for b in &self.buckets {
            if !BUCKETS_ALLOWED.contains(b) {
                return Err(std::io::Error::other(format!(
                    "clef config: bucket {b} outside the pinned set {BUCKETS_ALLOWED:?} — refusing"
                )));
            }
        }
        if self.max_questions != PINNED_MAX_QUESTIONS {
            return Err(std::io::Error::other(format!(
                "clef config: max_questions {} vs pinned {PINNED_MAX_QUESTIONS} — refusing",
                self.max_questions
            )));
        }
        if self.max_options != PINNED_MAX_OPTIONS {
            return Err(std::io::Error::other(format!(
                "clef config: max_options {} vs pinned {PINNED_MAX_OPTIONS} — refusing",
                self.max_options
            )));
        }
        if !self.tied_embeddings {
            return Err(std::io::Error::other(
                "clef config: tied_embeddings false — the head's lexical input is the tied \
                 embedding rows; an untied artifact is a different model, refusing",
            ));
        }
        if self.vocab_size == 0 || self.hidden_size == 0 || self.head_dim == 0 {
            return Err(std::io::Error::other(format!(
                "clef config: non-positive dims (vocab {}, hidden {}, head_dim {}) — refusing",
                self.vocab_size, self.hidden_size, self.head_dim
            )));
        }
        Ok(())
    }
}

/// `tokenizer.json` through the `tokenizers` crate — the exact-reference
/// path (the same file the reference's fast backend drives; no hand
/// port). Send + Sync (owned by the runtime instance, the actor-law
/// equivalent).
pub struct ClefTokenizer {
    inner: tokenizers::Tokenizer,
}

impl ClefTokenizer {
    /// Load `tokenizer.json` under `dir`.
    pub fn from_dir(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path).map_err(|e| {
            std::io::Error::other(format!("clef tokenizer: load {}: {e}", path.display()))
        })?;
        Ok(Self { inner })
    }

    /// Token ids for `text`, `add_special_tokens = false` — the record
    /// encoder carries its own `<|im_start|>` markers in the prompt text
    /// (the laya posture: an encode with specials off matches the
    /// reference by construction).
    pub fn encode(&self, text: &str) -> std::io::Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(|e| {
            std::io::Error::other(format!("clef tokenizer: encode failed: {e}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Vocab size including added tokens (the `<|im_*|>` specials).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// `embeddings.f16` — the tied fp16 `[vocab, hidden]` matrix, mmapped
/// zero-copy. Rows are exposed raw (fp16 bits, for the decoder's fp16
/// `hidden` input) and widened on demand into caller buffers (for the
/// head's f32 lexical mean — T1.5); the table itself never allocates a
/// converted copy.
pub struct EmbeddingTable {
    mmap: Mmap,
    vocab: u32,
    hidden: u32,
}

impl EmbeddingTable {
    /// Open + size-wall `embeddings.f16` under `dir`: bytes must equal
    /// `vocab·hidden·2` (the plan's config↔weight cross-check). The
    /// file is sha256-verified by the artifact store before this runs.
    pub fn open(dir: &Path, config: &ClefConfig) -> std::io::Result<Self> {
        let path = dir.join("embeddings.f16");
        let file = std::fs::File::open(&path)
            .map_err(|e| std::io::Error::other(format!("clef embeddings: open {}: {e}", path.display())))?;
        let expected = u64::from(config.vocab_size) * u64::from(config.hidden_size) * 2;
        let got = file
            .metadata()
            .map_err(|e| std::io::Error::other(format!("clef embeddings: stat {}: {e}", path.display())))?
            .len();
        if got != expected {
            return Err(std::io::Error::other(format!(
                "clef embeddings: {} bytes vs config vocab·hidden·2 = {expected} — refusing",
                got
            )));
        }
        // SAFETY: read-only mmap of a sha256-verified artifact file this
        // process never writes; the size wall above ran on the same fd.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| std::io::Error::other(format!("clef embeddings: mmap {}: {e}", path.display())))?;
        Ok(Self {
            mmap,
            vocab: config.vocab_size,
            hidden: config.hidden_size,
        })
    }

    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    pub fn hidden(&self) -> u32 {
        self.hidden
    }

    /// The whole matrix as fp16 bits (row-major). Panics on misalignment
    /// (mmap base is page-aligned; the size wall guarantees an even
    /// byte count) — unreachable by construction.
    fn bits(&self) -> &[u16] {
        bytemuck::cast_slice(&self.mmap[..])
    }

    /// One row as raw fp16 bits. Out-of-range ids panic loud — ids come
    /// from our own encoder over the same pinned vocab, so this is a
    /// caller-contract violation, not a runtime condition.
    pub fn row_f16(&self, id: u32) -> &[u16] {
        let h = self.hidden as usize;
        let start = id as usize * h;
        &self.bits()[start..start + h]
    }

    /// Copy one row's fp16 bits into `out` (the decoder gather: pad rows
    /// gather the `pad_id` row — T1.5). `out.len()` must equal `hidden`.
    pub fn copy_row_f16(&self, id: u32, out: &mut [u16]) {
        out.copy_from_slice(self.row_f16(id));
    }

    /// Widen one row into f32 in `out` (bit-exact: every f16 value is
    /// representable in f32). `out.len()` must equal `hidden`.
    pub fn row_into_f32(&self, id: u32, out: &mut [f32]) {
        for (o, b) in out.iter_mut().zip(self.row_f16(id)) {
            *o = half::f16::from_bits(*b).to_f32();
        }
    }
}

/// The three T1.3 materials loaded together.
pub struct ClefModel {
    pub config: ClefConfig,
    pub tokenizer: ClefTokenizer,
    pub embeddings: EmbeddingTable,
}

/// Pure load over an ALREADY-verified tree (the gates use this; a tree
/// the artifact store has not verified may refuse at any wall).
pub fn load_from(dir: &Path) -> std::io::Result<ClefModel> {
    let config = ClefConfig::from_dir(dir)?;
    let tokenizer = ClefTokenizer::from_dir(dir)?;
    let embeddings = EmbeddingTable::open(dir, &config)?;
    Ok(ClefModel {
        config,
        tokenizer,
        embeddings,
    })
}

/// The one-call entry: the T1.2 artifact law (fetch-on-missing + sha256
/// verify + marker) first, then all three materials.
pub fn ensure_and_load(dir: &Path) -> std::io::Result<ClefModel> {
    let root = crate::lane::ensure_artifacts(dir, crate::lane::hf_fetch)?;
    load_from(&root)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The REAL pinned `config.json` — fetched 2026-10-17 at the pinned
    /// revision (406 bytes, sha256 `af356b87…` == the T1.2 pin). The
    /// fixture asserts the loader accepts exactly these bytes.
    const PINNED_CONFIG: &str = r#"{
  "model": "clef-text-0.6b",
  "base_model": "Qwen/Qwen3-0.6B",
  "teacher": "Cloudflare/clef-flash",
  "hidden_size": 1024,
  "head_dim": 128,
  "rope_theta": 1000000,
  "pad_id": 151643,
  "vocab_size": 151936,
  "buckets": [
    256,
    512,
    1024,
    2048
  ],
  "max_questions": 16,
  "max_options": 96,
  "decoder": "Decoder.mlpackage",
  "head": "Head.mlpackage",
  "tied_embeddings": true
}"#;

    /// A pid-scoped temp dir (the shared-temp-path law: never a fixed
    /// name — another process running the same test truncates it).
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("clef_load_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, text: &str) {
        std::fs::write(dir.join("config.json"), text).unwrap();
    }

    #[test]
    fn pinned_config_parses_and_validates() {
        let dir = scratch("pin");
        write_config(&dir, PINNED_CONFIG);
        let cfg = ClefConfig::from_dir(&dir).expect("the real pinned config must validate");
        assert_eq!(cfg.model, "clef-text-0.6b");
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.pad_id, 151_643);
        assert_eq!(cfg.buckets, vec![256, 512, 1024, 2048]);
        assert_eq!(cfg.max_questions, 16);
        assert_eq!(cfg.max_options, 96);
        assert!(cfg.tied_embeddings);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Each wall fires on a mutated pinned config, naming the field.
    #[test]
    fn validation_walls_fire_loud() {
        let cases: &[(&str, String, &str)] = &[
            (
                "unsorted buckets",
                PINNED_CONFIG.replace("[\n    256,\n    512,\n    1024,\n    2048\n  ]", "[\n    512,\n    256,\n    1024,\n    2048\n  ]"),
                "not strictly ascending",
            ),
            (
                "bucket out of set",
                PINNED_CONFIG.replace("    256,\n", "    300,\n"),
                "outside the pinned set",
            ),
            (
                "max_questions drift",
                PINNED_CONFIG.replace("\"max_questions\": 16", "\"max_questions\": 8"),
                "max_questions",
            ),
            (
                "max_options drift",
                PINNED_CONFIG.replace("\"max_options\": 96", "\"max_options\": 64"),
                "max_options",
            ),
            (
                "untied embeddings",
                PINNED_CONFIG.replace("\"tied_embeddings\": true", "\"tied_embeddings\": false"),
                "tied_embeddings",
            ),
        ];
        for (name, text, expect) in cases {
            let dir = scratch("wall");
            write_config(&dir, text);
            let err = ClefConfig::from_dir(&dir)
                .expect_err(&format!("{name} must refuse"));
            assert!(
                err.to_string().contains(expect),
                "{name}: got {err}, wanted a message naming {expect}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The embedding table: size wall, raw-bit rows, f32 widening.
    #[test]
    fn embeddings_table_size_wall_and_rows() {
        let dir = scratch("emb");
        write_config(&dir, PINNED_CONFIG);
        let pinned = ClefConfig::from_dir(&dir).unwrap();

        // A tiny synthetic matrix shaped by a FIXTURE config (vocab 4 ×
        // hidden 8 = 64 bytes) — the wall math is dimension-generic, so
        // the law is proven at fixture scale. Row 0 carries
        // exactly-representable f16 values; row 2 is the pad-row
        // stand-in.
        let (vocab, hidden) = (4u32, 8u32);
        let row0: [u16; 8] = [0x3C00, 0x4000, 0x0000, 0xBC00, 0x3800, 0x7BFF, 0x0400, 0x03FF];
        let row2 = [0xB555u16; 8];
        let mut bits = Vec::new();
        bits.extend_from_slice(&row0);
        bits.extend_from_slice(&[0x3555u16; 8]);
        bits.extend_from_slice(&row2);
        bits.extend_from_slice(&[0x0000u16; 8]);
        let bytes: &[u8] = bytemuck::cast_slice(&bits);
        std::fs::write(dir.join("embeddings.f16"), bytes).unwrap();

        let mut fixture_cfg = pinned.clone();
        fixture_cfg.vocab_size = vocab;
        fixture_cfg.hidden_size = hidden;

        // The size wall fires when the dims drift off the file: bytes
        // are re-derived from the config, so vocab+1 demands 16 more
        // bytes than exist.
        let mut drifted = fixture_cfg.clone();
        drifted.vocab_size = vocab + 1;
        let Err(err) = EmbeddingTable::open(&dir, &drifted) else {
            panic!("vocab drift must hit the size wall")
        };
        assert!(err.to_string().contains("bytes vs config"), "got: {err}");
        let mut drifted = fixture_cfg.clone();
        drifted.hidden_size = hidden * 2;
        let Err(err) = EmbeddingTable::open(&dir, &drifted) else {
            panic!("hidden drift must hit the size wall")
        };
        assert!(err.to_string().contains("bytes vs config"), "got: {err}");

        let table = EmbeddingTable::open(&dir, &fixture_cfg).expect("matching dims open");
        assert_eq!(table.vocab(), vocab);
        assert_eq!(table.hidden(), hidden);

        let mut out = [0f32; 8];
        table.row_into_f32(0, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], -1.0);
        assert_eq!(out[4], 0.5);
        assert_eq!(out[5], 65_504.0); // max normal f16
        assert!(out[6] > 0.0 && out[6] < 1e-3); // min normal 2^-14
        assert!(out[7] > 0.0 && out[7] < out[6]); // largest subnormal

        let mut bits_out = [0u16; 8];
        table.copy_row_f16(2, &mut bits_out);
        assert_eq!(bits_out, row2, "raw fp16 bits must round-trip");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tokenizer path: a minimal real tokenizer.json loads and
    /// encodes (the 11 MB Qwen BPE file takes the same
    /// `Tokenizer::from_file` road — the real-artifact test below walks
    /// it when material is present).
    #[test]
    fn minimal_tokenizer_loads_and_encodes() {
        let dir = scratch("tok");
        let mini = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"hello": 0, "world": 1, "<unk>": 2},
    "unk_token": "<unk>"
  }
}"#;
        std::fs::write(dir.join("tokenizer.json"), mini).unwrap();
        let tok = ClefTokenizer::from_dir(&dir).expect("minimal tokenizer loads");
        assert_eq!(tok.encode("hello world").unwrap(), vec![0, 1]);
        assert_eq!(tok.vocab_size(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real three-material load — ONLY when the artifact tree is
    /// already installed (the `.verified-<rev>` marker present). Never
    /// triggers a fetch: absent material skips LOUD (a deferral, never a
    /// green zero), the repo's standard posture.
    #[test]
    fn real_artifacts_load_when_present() {
        let dir = match crate::lane::artifacts_dir() {
            Ok(dir) => dir,
            Err(e) => {
                println!("SKIP (loud): no artifact dir resolvable: {e}");
                return;
            }
        };
        let marker = dir.join(format!(".verified-{}", crate::lane::ARTIFACT_REVISION));
        if !marker.is_file() {
            println!(
                "SKIP (loud): artifacts not installed (no marker at {}); \
                 run ensure_and_load once to fetch the pinned tree",
                marker.display()
            );
            return;
        }
        let model = load_from(&dir).expect("verified tree must load");
        assert_eq!(model.config.model, "clef-text-0.6b");
        assert_eq!(model.config.vocab_size, 151_936);
        assert_eq!(model.config.buckets, vec![256, 512, 1024, 2048]);
        let tv = model.tokenizer.vocab_size();
        println!(
            "DISCLOSURE: tokenizer vocab {tv} vs config vocab_size {} (the tokenizer's ids must \
             stay below the config vocab — the embeddings table bounds by it)",
            model.config.vocab_size
        );
        assert!(
            tv <= model.config.vocab_size as usize,
            "tokenizer vocab {tv} exceeds config vocab_size — ids would index past the table"
        );
        // The pad row gathers: pad_id row is all-finite f32.
        let mut row = vec![0f32; model.embeddings.hidden() as usize];
        model
            .embeddings
            .row_into_f32(model.config.pad_id, &mut row);
        assert!(row.iter().all(|v| v.is_finite()), "pad row must be finite");
        // A spot encode through the real BPE.
        let ids = model.tokenizer.encode("JOINT SCHEMA DECISIONS:").unwrap();
        assert!(!ids.is_empty());
        println!("DISCLOSURE: spot encode ids = {ids:?}");
    }

    /// The REAL-WIRE install + load — `#[ignore]`d (a 1.3 GB one-time
    /// fetch; `cargo test` never touches the network, run it explicitly
    /// with `-- --ignored`). Exercises the production
    /// [`ensure_and_load`] end to end: the T1.2 choreography fetches
    /// every missing asset, verifies every sha256, writes the marker
    /// LAST — then the T1.3 loaders parse/validate/mmap the real tree.
    /// Idempotent: a second run takes the marker fast path (0 fetches).
    #[test]
    #[ignore = "real wire: ~1.3 GB one-time fetch (cached + marker-fast-path after)"]
    fn real_wire_install_and_load() {
        let dir = crate::lane::artifacts_dir().expect("artifact dir resolvable");
        let model = ensure_and_load(&dir).expect("install + load");
        assert_eq!(model.config.model, "clef-text-0.6b");
        assert_eq!(model.config.vocab_size, 151_936);
        assert_eq!(model.config.buckets, vec![256, 512, 1024, 2048]);
        assert_eq!(model.embeddings.vocab(), 151_936);
        assert_eq!(model.embeddings.hidden(), 1024);
        let tv = model.tokenizer.vocab_size();
        println!(
            "DISCLOSURE: tokenizer vocab {tv} vs config vocab_size {}",
            model.config.vocab_size
        );
        assert!(tv <= model.config.vocab_size as usize);
        let mut row = vec![0f32; 1024];
        model.embeddings.row_into_f32(model.config.pad_id, &mut row);
        assert!(row.iter().all(|v| v.is_finite()), "pad row finite");
        println!(
            "DISCLOSURE: pad row[0..4] = {:?}; spot encode of \"JOINT SCHEMA DECISIONS:\" = {:?}",
            &row[..4],
            model.tokenizer.encode("JOINT SCHEMA DECISIONS:").unwrap()
        );
    }
}
