//! SentencePiece/BPE tokenizers for GGUF model inference (Plan 087).
//!
//! Native-only: pulls sentencepiece-sys (C++ via cmake) which cannot compile
//! for wasm32 — browser inference targets small NPC brain models, not full
//! LLMs, so no text tokenizer is needed there.
//!
//! Wraps the `sentencepiece` crate for text ↔ token ID conversion.
//! Gemma 2 uses BOS=2, EOS=1 token IDs.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use anyhow::{Context, Result};

use crate::gguf_loader::GgufFile;

/// Leftmost-longest matcher over the special tokens (control=3,
/// user_defined=4) — the split points `encode` must never BPE across.
///
/// One left-to-right pass: at each byte, only the specials starting with
/// that byte are tried, longest first. That is the same answer as "the
/// earliest occurrence of any special, the longest on a tie", in
/// `O(text · candidates-per-byte)` rather than one full-text `find` per
/// special per match — which is quadratic on Gemma's vocabulary, whose
/// user-defined whitespace runs (`"\n\n"`, …) recur every few lines.
/// Byte-level probing is sound for UTF-8: a token starts with a lead byte,
/// which never equals a continuation byte, so no match can land mid-char.
#[derive(Debug, Clone, Default)]
struct SpecialTokenMatcher {
    /// `by_first[b]` = `(token bytes, id)` starting with byte `b`, longest first.
    by_first: Vec<Vec<(Box<[u8]>, usize)>>,
    len: usize,
}

impl SpecialTokenMatcher {
    /// Collect every control / user-defined token. On a duplicated string
    /// the LAST id wins (the prior `HashMap::insert` semantics); empty
    /// strings are skipped (they would match everywhere).
    fn from_types(token_types: &[u32], id_to_token: &[String]) -> Self {
        let mut map: HashMap<&str, usize> = HashMap::new();
        for (id, &tt) in token_types.iter().enumerate() {
            if (tt == 3 || tt == 4) && !id_to_token[id].is_empty() {
                map.insert(id_to_token[id].as_str(), id);
            }
        }
        let mut by_first: Vec<Vec<(Box<[u8]>, usize)>> = vec![Vec::new(); 256];
        for (tok, id) in &map {
            by_first[tok.as_bytes()[0] as usize].push((tok.as_bytes().into(), *id));
        }
        for bucket in &mut by_first {
            bucket.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        }
        Self {
            by_first,
            len: map.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The leftmost special in `text`, longest at that position:
    /// `(byte offset, byte length, id)`.
    fn find_first(&self, text: &str) -> Option<(usize, usize, usize)> {
        if self.is_empty() {
            return None;
        }
        let bytes = text.as_bytes();
        for (pos, &b) in bytes.iter().enumerate() {
            let rest = &bytes[pos..];
            if let Some((tok, id)) = self.by_first[b as usize]
                .iter()
                .find(|(t, _)| rest.starts_with(t))
            {
                return Some((pos, tok.len(), *id));
            }
        }
        None
    }
}

/// SentencePiece tokenizer wrapper for Gemma 2.
pub struct SentencePieceTokenizer {
    processor: sentencepiece::SentencePieceProcessor,
    bos_id: u32,
    eos_id: u32,
}

impl SentencePieceTokenizer {
    /// Load tokenizer from a `.model` file path.
    pub fn load(model_path: &Path) -> Result<Self> {
        let processor =
            sentencepiece::SentencePieceProcessor::open(model_path).with_context(|| {
                format!(
                    "Failed to open SentencePiece model: {}",
                    model_path.display()
                )
            })?;

        let bos_id = processor.bos_id().unwrap_or(2);
        let eos_id = processor.eos_id().unwrap_or(1);

        Ok(Self {
            processor,
            bos_id,
            eos_id,
        })
    }

    /// Encode text to token IDs (without BOS/EOS).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::with_capacity(text.len() / 4 + 1);
        self.encode_into(text, &mut out);
        out
    }

    /// Encode text into a pre-allocated buffer (no BOS/EOS).
    ///
    /// Avoids the double allocation of `encode` + `extend` when the caller
    /// already owns a `Vec<usize>` (e.g. `encode_with_bos`).
    fn encode_into(&self, text: &str, out: &mut Vec<usize>) {
        match self.processor.encode(text) {
            Ok(pieces) => {
                out.extend(pieces.iter().map(|p| p.id as usize));
            }
            Err(e) => {
                log::warn!("SentencePiece encode failed: {e}");
            }
        }
    }

    /// Encode text to token IDs with BOS prefix.
    pub fn encode_with_bos(&self, text: &str) -> Vec<usize> {
        // Heuristic: ~4 bytes/token for typical text; +1 for BOS.
        let mut tokens = Vec::with_capacity(text.len() / 4 + 2);
        tokens.push(self.bos_id as usize);
        self.encode_into(text, &mut tokens);
        tokens
    }

    /// Encode a user message in Gemma 2's chat template format, ready for the
    /// model's response to be generated.
    ///
    /// Produces the canonical Gemma 2 IT turn:
    /// ```text
    /// <bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n
    /// ```
    ///
    /// SentencePiece recognizes `<start_of_turn>` (token 106) and
    /// `<end_of_turn>` (token 107) as single special tokens, so encoding the
    /// template string directly is safe — no manual ID concatenation needed.
    ///
    /// **Why this matters for instruction-tuned models:** without the chat
    /// template, Gemma 2 treats the input as a base-model completion task
    /// (generating template/filler text) rather than an instruction-following
    /// task (actually solving the problem). On MATH-500, raw-text prompts
    /// produce `6 * ? = ?` literal placeholders; chat-templated prompts
    /// produce real chain-of-thought reasoning.
    pub fn encode_chat_user_turn(&self, user_prompt: &str) -> Vec<usize> {
        const PREFIX: &str = "<start_of_turn>user\n";
        const SUFFIX: &str = "<end_of_turn>\n<start_of_turn>model\n";

        let mut template = String::with_capacity(PREFIX.len() + user_prompt.len() + SUFFIX.len());
        template.push_str(PREFIX);
        template.push_str(user_prompt);
        template.push_str(SUFFIX);
        self.encode_with_bos(&template)
    }

    /// Encode a multi-turn conversation with few-shot examples in Gemma 2's
    /// chat template format.
    ///
    /// Each `(user, model)` pair becomes a separate conversational turn:
    /// ```text
    /// <bos><start_of_turn>user\n{u1}<end_of_turn>\n
    /// <start_of_turn>model\n{m1}<end_of_turn>\n
    /// ...
    /// <start_of_turn>user\n{final_user}<end_of_turn>\n
    /// <start_of_turn>model\n
    /// ```
    ///
    /// **Why multi-turn few-shot:** Gemma 2's chat tuning makes the model
    /// respond conversationally to a single user message. Few-shot examples
    /// embedded in a single user turn are treated as *content to discuss*
    /// rather than *patterns to follow*. Multi-turn few-shot (each example as
    /// a real user→model exchange) teaches the model the expected output
    /// format through actual conversation structure.
    pub fn encode_chat_multi_turn(
        &self,
        turns: &[(&str, &str)],
        final_user_prompt: &str,
    ) -> Vec<usize> {
        const USER_OPEN: &str = "<start_of_turn>user\n";
        const MODEL_OPEN: &str = "<start_of_turn>model\n";
        const CLOSE: &str = "<end_of_turn>\n";

        // Pre-compute capacity to avoid reallocation.
        // Each turn:  USER_OPEN + user + CLOSE + MODEL_OPEN + model + CLOSE
        // Final:      USER_OPEN + final_user + CLOSE + MODEL_OPEN
        const TURN_FIXED: usize = USER_OPEN.len() + CLOSE.len() + MODEL_OPEN.len() + CLOSE.len();
        const FINAL_FIXED: usize = USER_OPEN.len() + CLOSE.len() + MODEL_OPEN.len();

        let capacity = turns
            .iter()
            .map(|(u, m)| TURN_FIXED + u.len() + m.len())
            .sum::<usize>()
            + FINAL_FIXED
            + final_user_prompt.len();

        let mut template = String::with_capacity(capacity);
        for (user, model) in turns {
            template.push_str(USER_OPEN);
            template.push_str(user);
            template.push_str(CLOSE);
            template.push_str(MODEL_OPEN);
            template.push_str(model);
            template.push_str(CLOSE);
        }
        template.push_str(USER_OPEN);
        template.push_str(final_user_prompt);
        template.push_str(CLOSE);
        template.push_str(MODEL_OPEN);
        self.encode_with_bos(&template)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        match self.processor.decode_piece_ids(&ids) {
            Ok(text) => text,
            Err(e) => {
                log::warn!("SentencePiece decode failed: {e}");
                format!("[decode error: {e}]")
            }
        }
    }

    /// Get BOS token ID.
    #[inline]
    pub fn bos_id(&self) -> usize {
        self.bos_id as usize
    }

    /// Get EOS token ID.
    #[inline]
    pub fn eos_id(&self) -> usize {
        self.eos_id as usize
    }

    /// Get vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.processor.len()
    }
}

/// GPT-2 style BPE tokenizer loaded from GGUF metadata.
///
/// Supports Qwen 2/2.5, LLaMA 3, and other GPT-2 BPE models.
/// The tokenizer data is extracted directly from the GGUF file's metadata,
/// so no external tokenizer file is needed.
///
/// The pre-tokenization regex (GPT-2 pattern) is implemented as a manual
/// character-scanning state machine because the `regex` crate is not
/// available in `riir-engine`.
pub struct BpeTokenizer {
    /// Vocabulary: token string → token ID.
    vocab: HashMap<String, usize>,
    /// Inverse vocabulary: token ID → token string.
    id_to_token: Vec<String>,
    /// Token types: 1=normal, 2=unknown, 3=control, 4=user_defined,
    /// 5=unused, 6=byte.
    token_types: Vec<u32>,
    /// BPE merge ranks: `(token1, token2) → rank`.
    /// Lower rank = higher priority (applied first).
    merge_ranks: HashMap<(String, String), usize>,
    /// GPT-2 standard byte → unicode character mapping.
    byte_to_unicode: HashMap<u8, String>,
    /// Inverse: unicode character → byte.
    unicode_to_byte: HashMap<String, u8>,
    /// BOS token ID.
    bos_id: usize,
    /// EOS token ID.
    eos_id: usize,
    /// Special tokens (control + user_defined) that should not be split
    /// by BPE — leftmost-longest matcher.
    special: SpecialTokenMatcher,
    /// Longest digit run one pre-token may hold (the `\p{N}{1,k}` rule):
    /// `1` for Qwen2-style pre-tokenizers, `3` for `llama-bpe` (Llama 3,
    /// MiniCPM5). See [`digit_run_for_pre`].
    max_digit_run: usize,
}

/// The digit-run rule of a GGUF `tokenizer.ggml.pre` name. `llama-bpe`
/// (Llama 3 and its derivatives) groups `\p{N}{1,3}`; Qwen2 and the unknown
/// default split every digit. A wrong value is silent: text without numbers
/// tokenizes identically, and every number becomes a sequence the model never
/// saw (MiniCPM5 `" 42980."` → 5 digit tokens instead of `"429"`+`"80"`).
fn digit_run_for_pre(pre: &str) -> usize {
    match pre {
        "llama-bpe" | "llama3" | "llama-v3" => 3,
        _ => 1,
    }
}

impl BpeTokenizer {
    /// Load BPE tokenizer from a GGUF file's metadata.
    ///
    /// Reads `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`,
    /// `tokenizer.ggml.token_type`, and BOS/EOS IDs directly from the
    /// GGUF metadata — no external tokenizer file needed.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Verify tokenizer model type (warn but continue if not "gpt2").
        let model = gguf
            .metadata_string("tokenizer.ggml.model")
            .unwrap_or("gpt2");
        if model != "gpt2" {
            log::warn!("BpeTokenizer: expected 'gpt2' model type, got '{model}'");
        }

        // ── Load vocabulary ───────────────────────────────────────────
        let tokens_arr = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .context("BpeTokenizer: missing 'tokenizer.ggml.tokens' metadata")?;

        let mut vocab = HashMap::with_capacity(tokens_arr.len());
        let mut id_to_token = Vec::with_capacity(tokens_arr.len());
        for (id, val) in tokens_arr.iter().enumerate() {
            let s = val
                .as_str()
                .with_context(|| format!("BpeTokenizer: token {id} is not a string"))?;
            vocab.insert(s.to_string(), id);
            id_to_token.push(s.to_string());
        }

        // ── Load token types (optional, defaults to all-normal) ───────
        let token_types = gguf
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .map_or_else(
                || vec![1u32; id_to_token.len()],
                |arr| arr.iter().map(|v| v.as_u64().unwrap_or(1) as u32).collect(),
            );

        // ── Load BPE merges ───────────────────────────────────────────
        let merges_arr = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .context("BpeTokenizer: missing 'tokenizer.ggml.merges' metadata")?;

        let mut merge_ranks = HashMap::with_capacity(merges_arr.len());
        for (rank, val) in merges_arr.iter().enumerate() {
            let merge_str = val
                .as_str()
                .with_context(|| format!("BpeTokenizer: merge {rank} is not a string"))?;
            // Merge format: "token1 token2" (single space separator).
            // Split on the first space only — token strings in byte-to-unicode
            // space never contain literal spaces (space byte 0x20 maps to U+0100).
            if let Some(pos) = merge_str.find(' ') {
                let left = merge_str[..pos].to_string();
                let right = merge_str[pos + 1..].to_string();
                merge_ranks.insert((left, right), rank);
            }
        }

        // ── Build GPT-2 byte-to-unicode mapping ───────────────────────
        let (byte_to_unicode, unicode_to_byte) = Self::build_byte_to_unicode();

        // ── Get BOS/EOS token IDs ─────────────────────────────────────
        let bos_id = gguf
            .metadata_u64("tokenizer.ggml.bos_token_id")
            .map_or(1, |v| v as usize);
        let eos_id = gguf
            .metadata_u64("tokenizer.ggml.eos_token_id")
            .map_or(0, |v| v as usize);

        // ── Special tokens (control=3, user_defined=4) ───────────────
        let special = SpecialTokenMatcher::from_types(&token_types, &id_to_token);
        let max_digit_run =
            digit_run_for_pre(gguf.metadata_string("tokenizer.ggml.pre").unwrap_or(""));

        Ok(Self {
            vocab,
            id_to_token,
            token_types,
            merge_ranks,
            byte_to_unicode,
            unicode_to_byte,
            bos_id,
            eos_id,
            special,
            max_digit_run,
        })
    }

    /// Build the GPT-2 standard byte-to-unicode mapping.
    ///
    /// Bytes 33–126, 161–172, 174–255 (printable ASCII + Latin-1)
    /// map to themselves as unicode characters. All other bytes
    /// (control chars, space, DEL, non-breaking space, soft hyphen)
    /// map to code points starting at U+0100 (256).
    ///
    /// This ensures every byte maps to a printable, non-whitespace
    /// unicode character that can safely appear in a vocabulary string.
    fn build_byte_to_unicode() -> (HashMap<u8, String>, HashMap<String, u8>) {
        // Collect printable byte ranges that map to themselves.
        let mut printable: Vec<u8> = Vec::with_capacity(191);
        printable.extend(33u8..=126); // ASCII printable: '!' to '~'
        printable.extend(161u8..=172); // Latin-1: '¡' to '¬'
        printable.extend(174u8..=255); // Latin-1: '®' to 'ÿ'

        let mut byte_to_unicode = HashMap::with_capacity(256);
        let mut unicode_to_byte = HashMap::with_capacity(256);

        // Printable bytes → themselves.
        for &b in &printable {
            let ch = char::from_u32(b as u32).expect("printable byte is valid unicode");
            let s = ch.to_string();
            byte_to_unicode.insert(b, s.clone());
            unicode_to_byte.insert(s, b);
        }

        // Non-printable bytes → code points starting at U+0100 (256).
        let mut n = 0u32;
        for b in 0u8..=255 {
            if !printable.contains(&b) {
                let code = 256 + n;
                let ch = char::from_u32(code).expect("code point >= 256 is valid unicode");
                let s = ch.to_string();
                byte_to_unicode.insert(b, s.clone());
                unicode_to_byte.insert(s, b);
                n += 1;
            }
        }

        (byte_to_unicode, unicode_to_byte)
    }

    /// Encode text to token IDs (without BOS/EOS).
    ///
    /// Special tokens (control, user_defined) are recognized and emitted
    /// as single token IDs without BPE processing. Non-special text
    /// between special tokens is pre-tokenized and BPE-encoded normally.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if self.special.is_empty() {
            return self.encode_no_special(text);
        }

        let mut result = Vec::with_capacity(text.len() / 4 + 1);
        let mut remaining = text;
        while let Some((pos, len, id)) = self.special.find_first(remaining) {
            // BPE-encode the text before the special token.
            if pos > 0 {
                result.extend(self.encode_no_special(&remaining[..pos]));
            }
            result.push(id);
            remaining = &remaining[pos + len..];
        }
        if !remaining.is_empty() {
            result.extend(self.encode_no_special(remaining));
        }

        result.shrink_to_fit();
        result
    }

    /// Encode text to token IDs with BOS prefix.
    pub fn encode_with_bos(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(text.len() / 4 + 2);
        tokens.push(self.bos_id);
        tokens.extend(self.encode(text));
        tokens
    }

    /// BPE-encode text that contains no special tokens.
    ///
    /// Pre-tokenizes the text using the GPT-2 regex pattern, then applies
    /// BPE merges to each chunk independently.
    fn encode_no_special(&self, text: &str) -> Vec<usize> {
        let chunks = gpt2_pretokenize(text, self.max_digit_run);
        let mut result = Vec::with_capacity(chunks.len() * 2);
        for chunk in chunks {
            result.extend(self.bpe_encode_chunk(chunk));
        }
        result
    }

    /// Apply BPE merges to a single pre-tokenized chunk.
    ///
    /// 1. Convert each byte of the chunk to its GPT-2 unicode representation.
    /// 2. Repeatedly find the adjacent pair with the lowest merge rank and
    ///    merge all occurrences of that pair.
    /// 3. Look up each final token string in the vocabulary.
    fn bpe_encode_chunk(&self, chunk: &str) -> Vec<usize> {
        let bytes = chunk.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }

        // Convert bytes to GPT-2 unicode representation, stored in a single
        // String. Token boundaries are tracked as (start, end) byte offsets
        // into `unicode_str` — merging is O(1) (just extend the end).
        let mut unicode_str = String::with_capacity(bytes.len() * 2);
        let mut word: Vec<(usize, usize)> = Vec::with_capacity(bytes.len());

        for &b in bytes {
            let start = unicode_str.len();
            unicode_str.push_str(&self.byte_to_unicode[&b]);
            word.push((start, unicode_str.len()));
        }

        if word.len() <= 1 {
            let (s, e) = word[0];
            return vec![
                self.vocab
                    .get(&unicode_str[s..e])
                    .copied()
                    .unwrap_or_else(|| self.unk_id()),
            ];
        }

        // BPE merge loop: repeatedly merge the pair with the lowest rank.
        loop {
            let mut min_rank = usize::MAX;
            let mut min_idx = None;

            for i in 0..word.len() - 1 {
                let left = &unicode_str[word[i].0..word[i].1];
                let right = &unicode_str[word[i + 1].0..word[i + 1].1];
                // Construct key for merge_ranks lookup. This allocates two
                // Strings per pair per iteration — acceptable for the BPE
                // inner loop (chunks are typically < 20 bytes).
                //
                // A future optimization could use `HashMap<(usize, usize), usize>`
                // keyed by token IDs to avoid allocation.
                let key = (left.to_string(), right.to_string());
                if let Some(&rank) = self.merge_ranks.get(&key)
                    && rank < min_rank
                {
                    min_rank = rank;
                    min_idx = Some(i);
                }
            }

            match min_idx {
                None => break,
                Some(idx) => {
                    // Merge: extend the left token's end to include the right.
                    word[idx].1 = word[idx + 1].1;
                    word.remove(idx + 1);
                }
            }
        }

        // Look up final tokens in vocabulary.
        word.iter()
            .map(|&(s, e)| {
                self.vocab
                    .get(&unicode_str[s..e])
                    .copied()
                    .unwrap_or_else(|| self.unk_id())
            })
            .collect()
    }

    /// Decode token IDs back to text.
    ///
    /// For each token ID, looks up the token string (which is in GPT-2
    /// byte-to-unicode space) and converts each character back to its
    /// original byte. Characters not in the byte-to-unicode map (e.g.
    /// literal characters in special token strings) are emitted as UTF-8.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);

        for &id in tokens {
            if id >= self.id_to_token.len() {
                continue;
            }
            let token_str = &self.id_to_token[id];

            for c in token_str.chars() {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                if let Some(&b) = self.unicode_to_byte.get(s) {
                    bytes.push(b);
                } else {
                    // Character not in byte-to-unicode space (e.g. a literal
                    // char in a special token); emit as UTF-8.
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
        }

        match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("BpeTokenizer decode: invalid UTF-8: {e}");
                String::from_utf8_lossy(&e.into_bytes()).into_owned()
            }
        }
    }

    /// Get BOS token ID.
    #[inline]
    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    /// Get EOS token ID.
    #[inline]
    pub fn eos_id(&self) -> usize {
        self.eos_id
    }

    /// Get vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Find the unknown token ID (token type 2), defaulting to 0.
    #[inline]
    fn unk_id(&self) -> usize {
        self.token_types.iter().position(|&t| t == 2).unwrap_or(0)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// SentencePieceGgufTokenizer — GGUF-native Gemma-4 tokenizer (Issue 400)
// ──────────────────────────────────────────────────────────────────────────
//
// Gemma-4's GGUF declares `tokenizer.ggml.model = "gemma4"` and ships BPE
// merges (514K entries) + SentencePiece-style `▁` (U+2581) space markers.
// The existing `BpeTokenizer::from_gguf` applies GPT-2 byte-to-unicode + GPT-2
// regex pre-tokenization, which is wrong for Gemma-4: spaces become separate
// tokens instead of `▁`-prefixed word tokens, and decode emits `▁` literally.
//
// Algorithm (matches llama.cpp's `llama_tokenizer` for `gemma4` model type,
// verified via `llama-tokenize` agreement test, Issue 400 T7):
//   1. Normalize: replace ASCII space (0x20) with `▁` (U+2581). **No leading
//      `▁` prefix** — llama.cpp's Gemma-4 tokenizer does not prefix the text
//      start, so the first word is bare (`Hello` = id 9259), not `▁Hello`.
//   2. Pre-tokenize by splitting on `▁` boundaries — the segment before the
//      first `▁` is bare; each segment after a `▁` includes its leading `▁`.
//   3. Apply BPE merges within each segment (lowest merge rank first).
//   4. Look up final pieces in the vocab directly (NO byte-to-unicode mapping
//      — the vocab strings already contain `▁` and UTF-8 characters).
//   5. Byte fallback: unknown characters emit `<0xNN>` byte tokens (type=6,
//      ids 238–493).
//
// NOTE on Unigram scores: the GGUF's `tokenizer.ggml.scores` array is present
// but all 262,144 values are `-1000.0` (uniform). A Unigram Viterbi that relies
// on score differences cannot work. The actual algorithm is BPE merges — this
// was verified by inspecting the GGUF metadata + confirming that BPE on
// `▁`-replaced text (no leading prefix) produces `[Hello, ▁world]` =
// `[9259, 1902]`, which matches llama.cpp's `llama-tokenize --ids` output.

/// The SentencePiece space marker (U+2581, `▁`).
///
/// SentencePiece replaces ASCII spaces with this character before vocab lookup.
const SP_SPACE: char = '\u{2581}';

/// One node of the SPM merge list: a byte range into the normalized string
/// plus its neighbours. `start == end` marks a symbol absorbed by a merge.
#[derive(Clone, Copy)]
struct SpmSymbol {
    start: usize,
    end: usize,
    prev: i32,
    next: i32,
}

/// A candidate merge, ordered so `BinaryHeap::pop` yields the highest-scoring
/// pair and breaks ties toward the LEFTMOST one (llama.cpp's comparator).
struct SpmBigram {
    score: f32,
    left: usize,
    right: usize,
    /// Byte length the pair had when queued — the staleness check.
    len: usize,
}

impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // `total_cmp` rather than `partial_cmp`: scores come from GGUF and a
        // NaN would silently make the heap's ordering inconsistent.
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SpmBigram {}

/// SentencePiece tokenizer loaded from GGUF metadata (Gemma-4).
///
/// Sibling to [`BpeTokenizer`] (which targets GPT-2/Qwen/LLaMA-3 BPE models)
/// and [`SentencePieceTokenizer`] (which loads an external `.model` file for
/// Gemma-2). This struct reads the tokenizer data directly from the GGUF's
/// `tokenizer.ggml.*` metadata — no external file needed.
///
/// Correctness: produces the same token ID sequence as llama.cpp's
/// `llama_tokenize` on the same GGUF (Issue 400 T7). The load-bearing
/// invariant is that spaces become `▁`-prefixed word tokens for SUBSEQUENT
/// words (`▁world`), but the FIRST word is bare (`Hello`, no prefix).
pub struct SentencePieceGgufTokenizer {
    /// Vocabulary: token string → token ID.
    vocab: HashMap<String, usize>,
    /// Inverse vocabulary: token ID → token string.
    id_to_token: Vec<String>,
    /// Token types: 1=normal, 2=unknown, 3=control, 4=user_defined,
    /// 5=unused, 6=byte.
    token_types: Vec<u32>,
    /// BPE merge ranks: `(token1, token2) → rank`.
    /// Lower rank = higher priority (applied first).
    /// Empty when the GGUF carries no `tokenizer.ggml.merges` (unigram
    /// SentencePiece models — the Gemma-2 family).
    merge_ranks: HashMap<(String, String), usize>,
    /// Piece scores from `tokenizer.ggml.scores`, **indexed by token id**.
    /// Empty = BPE mode (merges present). Drives the SPM merge ordering.
    ///
    /// By id, not by string: this is read once per candidate bigram in
    /// `encode_spm`'s merge loop — the hottest line on the SPM path — and a
    /// second `HashMap<String, f32>` meant hashing every candidate span twice,
    /// once to prove it is in the vocab and again to price it. The vocab lookup
    /// already yields the id, so the price is a slice index. It also stops the
    /// score table from being a second full copy of the vocab's ~256K keys.
    scores: Vec<f32>,
    /// Prepend `▁` to the text start (`tokenizer.ggml.add_space_prefix`).
    add_dummy_prefix: bool,
    /// Special tokens (control + user_defined) that should not be split.
    /// Maps token string → token ID.
    special: SpecialTokenMatcher,
    /// Byte-fallback tokens: maps byte value → token ID for `<0xNN>` tokens
    /// (type=6). Used when a character is not in the vocab.
    byte_tokens: [usize; 256],
    /// BOS token ID.
    bos_id: usize,
    /// EOS token ID.
    eos_id: usize,
    /// Unknown token ID (type=2, or the `<unk>` control token).
    unk_id: usize,
}

impl SentencePieceGgufTokenizer {
    /// Load a SentencePiece tokenizer from GGUF metadata.
    ///
    /// Reads `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`,
    /// `tokenizer.ggml.scores`, `tokenizer.ggml.merges`, and BOS/EOS IDs.
    ///
    /// Warns (but proceeds) if `tokenizer.ggml.model` is not `"gemma4"`.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let model = gguf
            .metadata_string("tokenizer.ggml.model")
            .unwrap_or("gemma4");
        if model != "gemma4" {
            log::warn!("SentencePieceGgufTokenizer: expected 'gemma4' model type, got '{model}'");
        }

        // ── Load vocabulary ───────────────────────────────────────────
        let tokens_arr = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .context("SentencePieceGgufTokenizer: missing 'tokenizer.ggml.tokens'")?;

        let mut vocab = HashMap::with_capacity(tokens_arr.len());
        let mut id_to_token: Vec<String> = Vec::with_capacity(tokens_arr.len());
        for (id, val) in tokens_arr.iter().enumerate() {
            let s = val
                .as_str()
                .with_context(|| format!("token {id} is not a string"))?;
            vocab.insert(s.to_string(), id);
            id_to_token.push(s.to_string());
        }

        // ── Load token types ─────────────────────────────────────────
        let token_types = gguf
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .map_or_else(
                || vec![1u32; id_to_token.len()],
                |arr| arr.iter().map(|v| v.as_u64().unwrap_or(1) as u32).collect(),
            );

        // ── Load BPE merges (OPTIONAL — SPM GGUFs carry none) ─────────
        // The Gemma-2 family ships tokens + scores and no merges array.
        // Presence of merges selects the explicit-merge BPE mode; absence
        // selects the SPM score-priority merge (`encode_spm`), which recovers
        // the merge order from the scores themselves (Issue 970).
        let mut merge_ranks = HashMap::new();
        if let Some(merges_arr) = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
        {
            for (rank, val) in merges_arr.iter().enumerate() {
                let merge_str = val
                    .as_str()
                    .with_context(|| format!("merge {rank} is not a string"))?;
                // Merge format: "left right" (single ASCII space separator).
                // Token strings in the vocab never contain ASCII space (space is
                // encoded as ▁ U+2581), so a simple split-on-first-space is safe.
                if let Some(pos) = merge_str.find(' ') {
                    let left = merge_str[..pos].to_string();
                    let right = merge_str[pos + 1..].to_string();
                    merge_ranks.insert((left, right), rank);
                }
            }
        }

        // ── Load piece scores, indexed by token id ────────────────
        // A vocab id with no score row keeps NEG_INFINITY, which ranks it LAST
        // among candidate merges — `0.0` would promote an unscored piece to the
        // best merge in the segment. This is the same default `encode_spm` used
        // to apply at the lookup; it now lives where the table is built.
        let mut scores: Vec<f32> = Vec::new();
        if merge_ranks.is_empty() {
            let scores_arr = gguf
                .metadata
                .get("tokenizer.ggml.scores")
                .and_then(|v| v.as_array())
                .context("unigram model: missing 'tokenizer.ggml.scores'")?;
            scores = vec![f32::NEG_INFINITY; id_to_token.len()];
            for (id, val) in scores_arr.iter().enumerate() {
                if id < scores.len() {
                    scores[id] = val.as_f64().unwrap_or(-10.0) as f32;
                }
            }
        }

        // ── add_space_prefix → add_dummy_prefix (unigram normalization) ─
        let add_dummy_prefix = gguf
            .metadata
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // ── Get BOS/EOS token IDs ─────────────────────────────────────
        let bos_id = gguf
            .metadata_u64("tokenizer.ggml.bos_token_id")
            .map_or(2, |v| v as usize);
        let eos_id = gguf
            .metadata_u64("tokenizer.ggml.eos_token_id")
            .map_or(1, |v| v as usize);

        // ── Special tokens (control=3, user_defined=4) ───────────────
        let special = SpecialTokenMatcher::from_types(&token_types, &id_to_token);

        // ── Build byte-fallback token table ───────────────────────────
        // Type-6 tokens are `<0xNN>` (hex byte). Scan for them once at load.
        let mut byte_tokens = [0usize; 256];
        for (id, &tt) in token_types.iter().enumerate() {
            if tt == 6 {
                let s = &id_to_token[id];
                // Parse `<0xNN>` format.
                if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
                    let hex = &s[3..5];
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        byte_tokens[byte as usize] = id;
                    }
                }
            }
        }

        // ── Find unknown token ID (type=2, or <unk>) ─────────────────
        let unk_id = token_types
            .iter()
            .position(|&t| t == 2)
            // Fall back to the `<unk>` control token (Gemma-4 has it at id 3).
            .unwrap_or_else(|| id_to_token.iter().position(|s| s == "<unk>").unwrap_or(3));

        Ok(Self {
            vocab,
            id_to_token,
            token_types,
            merge_ranks,
            scores,
            add_dummy_prefix,
            special,
            byte_tokens,
            bos_id,
            eos_id,
            unk_id,
        })
    }

    /// Encode text to token IDs (without BOS/EOS).
    ///
    /// Special tokens (control=3, user_defined=4) are recognized via
    /// longest-match and emitted as single IDs. Non-special text between
    /// them is SentencePiece-normalized + BPE-encoded.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        self.encode_into(text, &mut out);
        out
    }

    /// Encode text into a pre-allocated buffer (no BOS/EOS).
    pub fn encode_into(&self, text: &str, out: &mut Vec<usize>) {
        if text.is_empty() {
            return;
        }

        let mut remaining = text;
        while let Some((pos, len, id)) = self.special.find_first(remaining) {
            if pos > 0 {
                self.encode_no_special(&remaining[..pos], out);
            }
            out.push(id);
            remaining = &remaining[pos + len..];
        }
        if !remaining.is_empty() {
            self.encode_no_special(remaining, out);
        }
    }

    /// Encode text with BOS prefix.
    pub fn encode_with_bos(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(text.len() / 3 + 2);
        tokens.push(self.bos_id);
        self.encode_into(text, &mut tokens);
        tokens
    }

    /// SentencePiece-normalize + BPE-encode text containing no special tokens.
    ///
    /// 1. Replace ASCII spaces with `▁` (U+2581). **No leading `▁` prefix**
    ///    (llama.cpp's Gemma-4 tokenizer does NOT prefix the text start —
    ///    verified via `llama-tokenize` agreement test, Issue 400 T7).
    /// 2. Pre-tokenize into `▁`-segments: split on `▁` boundaries. The text
    ///    before the first `▁` is a bare segment (no prefix); each `▁`-delimited
    ///    segment includes its leading `▁`.
    /// 3. BPE-encode each segment independently.
    fn encode_no_special(&self, text: &str, out: &mut Vec<usize>) {
        // SPM mode (no merges — the Gemma-2 SentencePiece family): score-
        // priority bigram merge with byte fallback, honoring add_dummy_prefix.
        const B0: u8 = 0xE2;
        const B1: u8 = 0x96;
        const B2: u8 = 0x81;

        if self.merge_ranks.is_empty() {
            self.encode_spm(text, out);
            return;
        }
        // Build the normalized string: replace ' ' with ▁. NO leading prefix.
        let mut normalized = String::with_capacity(text.len());
        for c in text.chars() {
            normalized.push(if c == ' ' { SP_SPACE } else { c });
        }
        if normalized.is_empty() {
            return;
        }

        // Pre-tokenize by ▁ boundaries. SP_SPACE is U+2581 (3 UTF-8 bytes:
        // E2 96 81). We scan for ▁ positions and emit segments.
        //
        // The text before the first ▁ is a bare segment (no ▁ prefix).
        // Each ▁ starts a new segment that includes the ▁. This means:
        //   `Hello▁world` → ["Hello", "▁world"]
        //   `▁Hello▁world` → ["", "▁Hello", "▁world"] (empty bare segment skipped)
        let bytes = normalized.as_bytes();
        let norm_len = normalized.len();

        let mut seg_start = 0usize; // byte offset of current segment start
        let mut i = 0usize;
        while i + 3 <= norm_len {
            if bytes[i] == B0 && bytes[i + 1] == B1 && bytes[i + 2] == B2 {
                // Found ▁ at byte i. Emit the bare segment [seg_start, i).
                if i > seg_start {
                    self.bpe_encode_segment(&normalized[seg_start..i], out);
                }
                // The ▁ starts a new segment at byte i.
                seg_start = i;
                i += 3;
            } else {
                i += 1;
            }
        }
        // Final segment: [seg_start, norm_len).
        if seg_start < norm_len {
            self.bpe_encode_segment(&normalized[seg_start..], out);
        }
    }

    /// SentencePiece **SPM** encode — the score-priority bigram merge, as in
    /// llama.cpp's `llm_tokenizer_spm` (our reference GGUF consumer, Issue 400
    /// T7). Used whenever the GGUF carries no `tokenizer.ggml.merges`.
    ///
    /// 1. Normalize: **pure space-escaping** — every ASCII space → `▁`, and
    ///    nothing else. No run collapse, no leading/trailing strip. Measured
    ///    from the reference model's own `normalizer_spec`:
    ///    `remove_extra_whitespaces = false`, `add_dummy_prefix = false`,
    ///    `name = "identity"`, `precompiled_charsmap = 0 bytes` (Issue 970;
    ///    `riir-train/data/tokenizer.model`). Prepend `▁` when
    ///    `add_dummy_prefix` (the GGUF's `tokenizer.ggml.add_space_prefix`).
    ///    The charsmap being literally identity is why full-width `！`,
    ///    decomposed `é`, NBSP / ideographic space / ZWSP and Thai SARA AM must
    ///    pass through raw — a generic NFKC prepass would *introduce*
    ///    divergence (katgpt-rs Research 565 §2.1). Do not add one.
    /// 2. Segment: start from one symbol per character, then repeatedly merge
    ///    the adjacent pair whose merged text scores **highest** in the vocab,
    ///    re-offering the two new neighbour pairs after each merge.
    /// 3. Emit: every surviving symbol is in the vocab by construction (a merge
    ///    only happens when the merged text is); an original single character
    ///    that is not falls back to `<0xNN>` byte tokens, else `<unk>`.
    ///
    /// **Why merge and not a unigram Viterbi** (Issue 970, measured): this path
    /// used to run the unigram Viterbi from `unigram_model.cc`, maximizing
    /// Σ piece-score. That is the right algorithm only when the scores are
    /// log-probabilities. gemma-2's `tokenizer.model` declares
    /// `trainer_spec.model_type = BPE`, so its scores are **negative merge
    /// ranks** (`▁world` = −1661, `Next` = −5880, floor −255494 ≈ −|vocab|) —
    /// summing those rewards short frequent pieces without bound, and the
    /// Viterbi fragmented ordinary words: `"Next"` → `["Ne", "xt"]`
    /// (−3684 + −248 beats −5880) and `" world"` → `["▁w", "or", "ld"]`.
    /// A GGUF from a genuine `model_type = UNIGRAM` model would need the
    /// Viterbi back (it is in this file's git history); the discriminator is
    /// the score distribution — log-probs cluster in a small negative range,
    /// ranks run from 0 down to roughly −|vocab|.
    fn encode_spm(&self, text: &str, out: &mut Vec<usize>) {
        // ── Normalize ──────────────────────────────────────────────
        let mut normalized = String::with_capacity(text.len() + 3);
        for c in text.chars() {
            normalized.push(if c == ' ' { SP_SPACE } else { c });
        }
        if normalized.is_empty() {
            return;
        }
        if self.add_dummy_prefix {
            normalized.insert(0, SP_SPACE);
        }

        // ── Symbols: one per character, as a doubly-linked list over byte
        // ranges into `normalized`. A merge is always of two ADJACENT ranges,
        // so the merged text is exactly `left.start..right.end` — no string
        // building — and an absorbed symbol is left as `start == end`.
        let mut syms: Vec<SpmSymbol> = Vec::with_capacity(normalized.len());
        for (start, c) in normalized.char_indices() {
            let i = syms.len() as i32;
            syms.push(SpmSymbol {
                start,
                end: start + c.len_utf8(),
                prev: i - 1,
                next: i + 1,
            });
        }
        syms.last_mut().expect("normalized is non-empty").next = -1;

        // ── Merge by score priority ────────────────────────────────
        let mut heap: BinaryHeap<SpmBigram> = BinaryHeap::with_capacity(syms.len());
        let offer = |heap: &mut BinaryHeap<SpmBigram>, syms: &[SpmSymbol], l: i32, r: i32| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            let span = &normalized[syms[l].start..syms[r].end];
            // ONE hash of the span: the vocab lookup already yields the id the
            // score table is keyed on. This is the hottest line on the SPM
            // path — it runs once per candidate bigram, twice more after every
            // merge — so a second hash of the same string was the whole cost of
            // pricing a pair. `.get()` rather than `[id]` because the table is
            // empty in BPE mode; unscored ranks LAST (see the field doc).
            if let Some(&id) = self.vocab.get(span) {
                heap.push(SpmBigram {
                    score: self.scores.get(id).copied().unwrap_or(f32::NEG_INFINITY),
                    left: l,
                    right: r,
                    len: span.len(),
                });
            }
        };
        for i in 1..syms.len() {
            offer(&mut heap, &syms, i as i32 - 1, i as i32);
        }

        while let Some(b) = heap.pop() {
            // Stale entry: a side was absorbed, or the pair no longer spans
            // what this entry was queued for.
            let l_len = syms[b.left].end - syms[b.left].start;
            let r_len = syms[b.right].end - syms[b.right].start;
            if l_len == 0 || r_len == 0 || l_len + r_len != b.len {
                continue;
            }
            // Absorb right into left; unlink right.
            syms[b.left].end = syms[b.right].end;
            syms[b.right].end = syms[b.right].start;
            let after = syms[b.right].next;
            syms[b.left].next = after;
            if after >= 0 {
                syms[after as usize].prev = b.left as i32;
            }
            let (before, left) = (syms[b.left].prev, b.left as i32);
            offer(&mut heap, &syms, before, left);
            offer(&mut heap, &syms, left, after);
        }

        // ── Emit ───────────────────────────────────────────────────
        let mut i: i32 = 0;
        while i >= 0 {
            let sym = syms[i as usize];
            i = sym.next;
            if sym.end == sym.start {
                continue;
            }
            let span = &normalized[sym.start..sym.end];
            // A byte piece's own text is the 6-char `<0xNN>` literal, never a
            // character of real input — matching one would emit a byte token
            // for text that merely looks like one.
            match self.vocab.get(span) {
                Some(&id) if self.token_types.get(id).copied().unwrap_or(1) != 6 => {
                    out.push(id);
                }
                _ => {
                    for byte in span.bytes() {
                        let byte_id = self.byte_tokens[byte as usize];
                        out.push(if byte_id != 0 { byte_id } else { self.unk_id });
                    }
                }
            }
        }
    }

    /// Apply BPE merges to a single segment, emitting token IDs.
    ///
    /// 1. Split the segment into UTF-8 characters (not bytes).
    /// 2. Repeatedly find the adjacent pair with the lowest merge rank and
    ///    merge that pair.
    /// 3. Look up each final token string in the vocabulary. Unknown pieces
    ///    fall back to byte tokens (type=6 `<0xNN>`).
    fn bpe_encode_segment(&self, segment: &str, out: &mut Vec<usize>) {
        if segment.is_empty() {
            return;
        }

        // Collect characters as owned strings. For short segments (words),
        // this is a small allocation.
        let mut word: Vec<String> = segment.chars().map(|c| c.to_string()).collect();

        if word.len() <= 1 {
            self.emit_piece(&word[0], out);
            return;
        }

        // BPE merge loop: repeatedly merge the pair with the lowest rank.
        loop {
            let mut min_rank = usize::MAX;
            let mut min_idx = None;

            for i in 0..word.len() - 1 {
                let key = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&key)
                    && rank < min_rank
                {
                    min_rank = rank;
                    min_idx = Some(i);
                }
            }

            match min_idx {
                None => break,
                Some(idx) => {
                    // Merge: concatenate left + right.
                    let merged = format!("{}{}", word[idx], word[idx + 1]);
                    word[idx] = merged;
                    word.remove(idx + 1);
                }
            }
        }

        // Look up final tokens in vocabulary, with byte fallback.
        for piece in &word {
            self.emit_piece(piece, out);
        }
    }

    /// Emit token ID(s) for a single piece after BPE.
    ///
    /// - If the piece is in the vocab, emit its ID.
    /// - If the piece is a single character not in the vocab, emit byte-fallback
    ///   tokens (`<0xNN>`) for each of its UTF-8 bytes.
    /// - If the piece is multi-char and not in the vocab, emit `unk_id`.
    #[inline]
    fn emit_piece(&self, piece: &str, out: &mut Vec<usize>) {
        if let Some(&id) = self.vocab.get(piece) {
            out.push(id);
            return;
        }
        // Byte fallback: emit one byte token per UTF-8 byte.
        // This handles unknown characters (e.g. rare Unicode, emoji) by
        // decomposing them into the 256-entry byte vocab.
        if piece.chars().count() == 1 {
            for &byte in piece.as_bytes() {
                out.push(self.byte_tokens[byte as usize]);
            }
            return;
        }
        // Multi-char unknown piece (shouldn't happen after BPE).
        out.push(self.unk_id);
    }

    /// Decode token IDs back to text.
    ///
    /// Replaces `▁` (U+2581) with ASCII space during decode. Byte tokens
    /// (`<0xNN>`) are accumulated as raw bytes, then decoded as UTF-8
    /// (byte sequences may form multi-byte characters).
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut result = String::with_capacity(tokens.len() * 4);
        // Byte tokens are accumulated into this buffer, then decoded as UTF-8
        // when a non-byte token is encountered (or at the end). This correctly
        // handles multi-byte UTF-8 sequences emitted as individual byte tokens.
        let mut byte_buf: Vec<u8> = Vec::with_capacity(8);

        for &id in tokens {
            if id >= self.id_to_token.len() {
                continue;
            }
            let token_str = &self.id_to_token[id];
            let tt = self.token_types.get(id).copied().unwrap_or(1);

            if tt == 6 {
                // Byte token `<0xNN>`: accumulate raw byte.
                if token_str.len() == 6
                    && token_str.starts_with("<0x")
                    && token_str.ends_with('>')
                    && let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16)
                {
                    byte_buf.push(byte);
                }
            } else {
                // Flush any accumulated bytes as UTF-8 before this token.
                if !byte_buf.is_empty() {
                    result.push_str(&String::from_utf8_lossy(&byte_buf));
                    byte_buf.clear();
                }
                // Normal/control/user_defined token: replace ▁ with space.
                for c in token_str.chars() {
                    if c == SP_SPACE {
                        result.push(' ');
                    } else {
                        result.push(c);
                    }
                }
            }
        }

        // Flush trailing bytes.
        if !byte_buf.is_empty() {
            result.push_str(&String::from_utf8_lossy(&byte_buf));
        }

        result
    }

    /// Encode a user message in Gemma-4's chat template format.
    ///
    /// Produces the canonical Gemma-4 IT turn:
    /// ```text
    /// <bos><|turn>user
    /// {user_msg}<turn|>
    /// <|turn>model
    /// ```
    ///
    /// `<|turn>` (id 105) and `<turn|>` (id 106) are recognized as single
    /// special tokens, so encoding the template string directly is safe.
    pub fn encode_chat_user_turn_gemma4(&self, user_msg: &str) -> Vec<usize> {
        const PREFIX: &str = "<|turn>user\n";
        const SUFFIX: &str = "<turn|>\n<|turn>model\n";

        let mut template = String::with_capacity(PREFIX.len() + user_msg.len() + SUFFIX.len());
        template.push_str(PREFIX);
        template.push_str(user_msg);
        template.push_str(SUFFIX);
        self.encode_with_bos(&template)
    }

    /// Get BOS token ID.
    #[inline]
    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    /// Get EOS token ID.
    #[inline]
    pub fn eos_id(&self) -> usize {
        self.eos_id
    }

    /// Get vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Get the token string for a given ID (for debugging).
    #[inline]
    pub fn id_to_token_str(&self, id: usize) -> &str {
        if id < self.id_to_token.len() {
            &self.id_to_token[id]
        } else {
            ""
        }
    }
}

/// GPT-2 pre-tokenization: split text into chunks following the GPT-2 regex
/// pattern, implemented as a manual character scanner (no `regex` crate).
///
/// The pattern (case-insensitive for contractions):
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)
/// | [^\r\n\p{L}\p{N}]?\p{L}+
/// | \p{N}{1,max_digit_run}
/// |  ?[^\s\p{L}\p{N}]+[\r\n]*
/// | \s*[\r\n]+
/// | \s+(?!\S)
/// | \s+
/// ```
///
/// Rules are tried in order at each position; the first matching rule wins.
fn gpt2_pretokenize(text: &str, max_digit_run: usize) -> Vec<&str> {
    // Build (byte_offset, char) pairs for O(1) byte-offset lookup.
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let text_len = text.len();

    let mut chunks = Vec::new();
    let mut i = 0;

    while i < n {
        let start_byte = chars[i].0;
        let end = match_pretoken(&chars, i, max_digit_run);
        let end_byte = if end < n { chars[end].0 } else { text_len };
        chunks.push(&text[start_byte..end_byte]);
        i = end;
    }

    chunks.shrink_to_fit();
    chunks
}

/// Try to match a GPT-2 pre-tokenization rule at position `i` in `chars`.
///
/// Returns the index into `chars` after the match (i.e. the next unmatched
/// position). At least one character is always consumed (the fallback case).
fn match_pretoken(chars: &[(usize, char)], i: usize, max_digit_run: usize) -> usize {
    let n = chars.len();
    let c = chars[i].1;

    // Rule 1: Contractions ('s 't 're 've 'm 'll 'd, case-insensitive).
    if c == '\'' && i + 1 < n {
        let next = chars[i + 1].1;
        if next.eq_ignore_ascii_case(&'s')
            || next.eq_ignore_ascii_case(&'t')
            || next.eq_ignore_ascii_case(&'m')
            || next.eq_ignore_ascii_case(&'d')
        {
            return i + 2;
        }
        if i + 2 < n {
            let third = chars[i + 2].1;
            if next.eq_ignore_ascii_case(&'r') && third.eq_ignore_ascii_case(&'e') {
                return i + 3;
            }
            if next.eq_ignore_ascii_case(&'v') && third.eq_ignore_ascii_case(&'e') {
                return i + 3;
            }
            if next.eq_ignore_ascii_case(&'l') && third.eq_ignore_ascii_case(&'l') {
                return i + 3;
            }
        }
    }

    // Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
    // Optional non-(\r \n letter number) + one or more letters.
    {
        let mut j = i;
        if j < n {
            let cc = chars[j].1;
            if cc != '\r' && cc != '\n' && !cc.is_alphabetic() && !cc.is_numeric() {
                j += 1;
            }
        }
        if j < n && chars[j].1.is_alphabetic() {
            while j < n && chars[j].1.is_alphabetic() {
                j += 1;
            }
            return j;
        }
    }

    // Rule 3: \p{N}{1,max_digit_run} — a bounded run of number characters.
    if c.is_numeric() {
        let mut j = i + 1;
        while j < n && j - i < max_digit_run.max(1) && chars[j].1.is_numeric() {
            j += 1;
        }
        return j;
    }

    // Rule 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
    // Optional space + one or more non-(space letter number) + optional newlines.
    {
        let mut j = i;
        if j < n && chars[j].1 == ' ' {
            j += 1;
        }
        if j < n
            && !chars[j].1.is_whitespace()
            && !chars[j].1.is_alphabetic()
            && !chars[j].1.is_numeric()
        {
            while j < n
                && !chars[j].1.is_whitespace()
                && !chars[j].1.is_alphabetic()
                && !chars[j].1.is_numeric()
            {
                j += 1;
            }
            while j < n && (chars[j].1 == '\r' || chars[j].1 == '\n') {
                j += 1;
            }
            return j;
        }
    }

    // Rules 5/6/7: whitespace handling.
    if c.is_whitespace() {
        // Find end of the whitespace run.
        let mut j = i;
        while j < n && chars[j].1.is_whitespace() {
            j += 1;
        }

        // Rule 5: \s*[\r\n]+ — need at least one \r or \n.
        // Match extends to (and including) the last \r/\n in the run.
        // (Regex backtracking: \s* greedily consumes all whitespace, then
        // backtracks to leave at least one \r/\n for [\r\n]+. The total
        // match is [i, last_newline + 1).)
        for k in (i..j).rev() {
            if chars[k].1 == '\r' || chars[k].1 == '\n' {
                return k + 1;
            }
        }

        // No newlines in the run → Rule 6 (\s+(?!\S)) or Rule 7 (\s+).
        //
        // Rule 6 matches trailing whitespace: all whitespace except the
        // last char if followed by non-whitespace (so the last whitespace
        // can attach to the next word via Rule 2's optional prefix).
        // Rule 7 matches any remaining whitespace run.
        if j >= n || j - i <= 1 {
            // End of text, or single whitespace char: match all.
            return j;
        }
        // Multiple whitespace chars followed by non-whitespace:
        // match all but the last.
        return j - 1;
    }

    // Fallback: consume 1 character. Should not normally be reached because
    // Rule 4 catches punctuation and Rule 7 catches whitespace, but included
    // for safety.
    i + 1
}

#[cfg(test)]
mod special_token_matcher_tests {
    use super::SpecialTokenMatcher;

    /// Reference: the original per-token `find` scan (earliest, longest on a tie).
    fn naive(specials: &[(&str, usize)], text: &str) -> Option<(usize, usize, usize)> {
        let mut best: Option<(usize, usize, usize)> = None;
        for &(t, id) in specials {
            if let Some(pos) = text.find(t) {
                match best {
                    Some((bp, bl, _)) if pos > bp || (pos == bp && t.len() <= bl) => {}
                    _ => best = Some((pos, t.len(), id)),
                }
            }
        }
        best
    }

    #[test]
    fn leftmost_longest_matches_the_naive_scan() {
        let toks = ["<a>", "<ab>", "\n\n", "\n\n\n", "é!", "x", ""];
        let types = [3u32, 4, 4, 4, 3, 1, 3];
        let names: Vec<String> = toks.iter().map(|s| s.to_string()).collect();
        let m = SpecialTokenMatcher::from_types(&types, &names);
        let specials = [
            ("<a>", 0usize),
            ("<ab>", 1),
            ("\n\n", 2),
            ("\n\n\n", 3),
            ("é!", 4),
        ];
        for text in [
            "",
            "plain",
            "a<ab>b",
            "x\n\n\ny<a>",
            "ée!é!",
            "<a<ab><a>",
            "tail\n",
            "\n\n\n\n\n",
        ] {
            assert_eq!(m.find_first(text), naive(&specials, text), "{text:?}");
        }
    }

    #[test]
    fn duplicate_string_keeps_the_last_id() {
        let names = vec!["<s>".to_string(), "<s>".to_string()];
        let m = SpecialTokenMatcher::from_types(&[3, 3], &names);
        assert_eq!(m.find_first("a<s>"), Some((1, 3, 1)));
    }
}

#[cfg(test)]
mod digit_run_tests {
    use super::{digit_run_for_pre, gpt2_pretokenize};

    /// `llama-bpe` groups digits in threes (HF `\p{N}{1,3}`), the space
    /// before a number is its own pre-token; the Qwen2-style default splits
    /// every digit. Measured against HF `tokenizers` on MiniCPM5-1B: 54 271 of
    /// 54 271 ids identical on number-dense text once the rule followed
    /// `tokenizer.ggml.pre`.
    #[test]
    fn digit_runs_follow_the_pre_tokenizer() {
        assert_eq!(digit_run_for_pre("llama-bpe"), 3);
        assert_eq!(digit_run_for_pre("qwen2"), 1);
        assert_eq!(digit_run_for_pre(""), 1);
        assert_eq!(
            gpt2_pretokenize("key 42980.", 3),
            vec!["key", " ", "429", "80", "."]
        );
        assert_eq!(
            gpt2_pretokenize("key 42980.", 1),
            vec!["key", " ", "4", "2", "9", "8", "0", "."]
        );
    }
}
