//! EXL3 pack reader — the safetensors-directory level (Issue 001, T5).
//!
//! [`Exl3Pack`] opens a converted EXL3 pack directory (sharded via
//! `model.safetensors.index.json`, a single `model.safetensors`, or a plain
//! directory of `*.safetensors`), maps every shard once, and hands out
//! zero-copy [`Exl3Layer`] borrows — §11.3 condition 1 of the T2 seam
//! decision: lazy by construction. An EXL3 layer that eagerly expanded to
//! dense f32 at load would be WORSE than BF16 and forfeit the only axis
//! Issue 001 §2 says is real (residency + context ceiling).
//!
//! ⚠ **Groups span shards in real packs.** On
//! `turboderp/Qwen3.8-27B-exl3@SC_4.00bpw_H5_V6`, `lm_head.suh/svh/mul1`
//! live in shard 1 while `lm_head.trellis` lives in shard 2 — the §11.3
//! round-2 open question, answered by a real pack. Every group member is
//! resolved through the merged name→(shard, offsets) table, so one layer's
//! byte slices may borrow different shards.
//!
//! [`Exl3Pack::residency`] is the T5 instrument: exact on-disk bytes by
//! class (trellis codes / channel scales / markers / dense leftovers) from
//! metadata alone, plus achieved bits-per-weight — no tensor data touched.
//!
//! Era gate (Issue 001 §12.7, wired at open by the T7c-4 hygiene half):
//! [`Exl3Pack::open`] verifies `quantization_config.version` in the pack's
//! `config.json` (or the standalone `quantization_config.json` spelling)
//! against [`KNOWN_GOOD_ERA_VERSIONS`] and REFUSES fail-closed on an absent
//! or unknown version — old packs silently decode differently under current
//! exllamav3 (the §12.7 measured specimen: 2025-05-era pack, `rel-Frobenius
//! 0.184` divergence). A deliberate legacy/exotic-era read is spelled at
//! the call site via [`Exl3Pack::open_unverified_era`].
//!
//! ⚠ Gate blind spot (stated, not hidden): the PASS side is validated at
//! n=1 (the pin-era 4.09 bpw 27B pack, `version = "1.4.2"`) and the REFUSE
//! side at n=0 (no legacy specimen on disk) — fail-closed is what makes
//! n=0 acceptable; a permissive gate would not be. Admission of a new
//! writer version = the T7c-1c full-pack bit-exact gate on a real pack of
//! that version, then one line in [`KNOWN_GOOD_ERA_VERSIONS`].

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use super::exl3::{Exl3Layer, Exl3LayerPlan, detect_exl3_layers};
use crate::safetensors_loader::{TensorMeta, parse_safetensors_header};

/// exllamav3 writer versions this loader is VALIDATED against — admission
/// is the T7c-1c full-pack bit-exact gate (573/573 layers, 26.48 G weights,
/// 0 mismatches, CUDA + Metal) run on a real pack of that version. Today
/// that is exactly the pin-era `turboderp/Qwen3.8-27B-exl3@SC_4.00bpw_H5_V6`
/// (`version = "1.4.2"`). Everything else — including newer exllamav3
/// releases until validated — is refused by [`Exl3Pack::open`] (fail-closed,
/// Issue 001 §12.7); a deliberate read of an unvalidated era goes through
/// [`Exl3Pack::open_unverified_era`], which names the risk at the call site.
pub const KNOWN_GOOD_ERA_VERSIONS: &[&str] = &["1.4.2"];

/// One mapped shard + its parsed header.
struct Shard {
    map: Mmap,
    /// Byte offset where tensor data starts (`8 + header_len`).
    data_base: usize,
}

impl std::fmt::Debug for Shard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shard")
            .field("data_base", &self.data_base)
            .field("len", &self.map.len())
            .finish()
    }
}

/// An open EXL3 pack: all shards mapped, metadata merged, groups detected.
#[derive(Debug)]
pub struct Exl3Pack {
    shards: Vec<Shard>,
    /// Tensor name → (shard index, header meta). Offsets in `TensorMeta`
    /// are relative to the shard's data base.
    tensors: BTreeMap<String, (usize, TensorMeta)>,
    plans: Vec<Exl3LayerPlan>,
    plan_by_key: BTreeMap<String, usize>,
}

/// Byte accounting for the residency axis (Issue 001 §2 / T5).
#[derive(Debug, Clone, Default)]
pub struct Exl3Residency {
    /// Detected EXL3 layer groups.
    pub groups: usize,
    /// `.trellis` code bytes across all groups.
    pub trellis_bytes: u64,
    /// `suh`/`svh` fp16 channel scales + legacy `su`/`sv` packed signs.
    pub scale_bytes: u64,
    /// `mul1`/`mcg` marker tensors.
    pub marker_bytes: u64,
    /// `.bias` tensors inside groups.
    pub group_bias_bytes: u64,
    /// Every tensor outside an EXL3 group (embeddings, norms, `A_log`,
    /// `conv1d`, fused `qkv.weight`, `patch_embed`, …).
    pub dense_bytes: u64,
    /// Quantized weight elements (`Σ in·out` over groups).
    pub group_params: u64,
    /// Dense tensor elements (embeddings + norms + leftovers).
    pub dense_params: u64,
    /// Sum of all tensor data bytes (equals the pack's tensor payload).
    pub total_tensor_bytes: u64,
}

impl Exl3Residency {
    /// Bytes the quantized linears occupy on disk.
    pub fn quantized_bytes(&self) -> u64 {
        self.trellis_bytes + self.scale_bytes + self.marker_bytes + self.group_bias_bytes
    }

    /// Achieved bits per quantized weight (the residency-axis bpw).
    pub fn achieved_bpw(&self) -> f64 {
        if self.group_params == 0 {
            return 0.0;
        }
        self.quantized_bytes() as f64 * 8.0 / self.group_params as f64
    }

    /// Footprint of the same model fully dequantized at 2 bytes/weight
    /// (the BF16-equal baseline the residency axis is measured against;
    /// dense tensors are unchanged).
    pub fn dequantized_f16_bytes(&self) -> u64 {
        (self.group_params + self.dense_params) * 2
    }
}

impl Exl3Pack {
    /// Open a pack directory, era-gated (Issue 001 §12.7 / T7c-4): the
    /// pack's `quantization_config.version` must appear in
    /// [`KNOWN_GOOD_ERA_VERSIONS`] or the open REFUSES loudly — legacy-era
    /// packs decode wrong under this reader (measured: the 2025-05 specimen
    /// diverged at rel-Frobenius 0.184). Use [`Exl3Pack::open_unverified_era`]
    /// for a deliberate unvalidated-era read.
    ///
    /// Layouts, in preference order:
    ///
    /// 1. `model.safetensors.index.json` — the sharded HF layout; the
    ///    weight map decides which shard each tensor lives in.
    /// 2. Any `*.safetensors` files in the directory (sorted by name) —
    ///    covers single-file packs and index-less shard sets.
    ///
    /// All shards are mapped eagerly (mmap is lazy — pages load on touch),
    /// headers parsed, markers read, and EXL3 groups detected. A tensor
    /// name appearing in two shards, or an index entry naming a missing
    /// shard, refuses loudly.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_era_policy(dir, EraPolicy::Verify)
    }

    /// Open a pack directory WITHOUT the era gate — the deliberate
    /// unvalidated-era read (legacy packs, pre-release writer builds, or
    /// a new exllamav3 version not yet admitted to
    /// [`KNOWN_GOOD_ERA_VERSIONS`]). The caller owns the compatibility
    /// risk: §12.7 measured a legacy pack decoding DIFFERENTLY (silently
    /// wrong values, not an error) under pin-era code. Spelling the escape
    /// at the call site is the point — a silent default would hide it.
    pub fn open_unverified_era(dir: &Path) -> Result<Self> {
        Self::open_with_era_policy(dir, EraPolicy::Unverified)
    }

    fn open_with_era_policy(dir: &Path, era: EraPolicy) -> Result<Self> {
        let shard_names = shard_file_names(dir)?;
        if shard_names.is_empty() {
            bail!("no safetensors shards found in {}", dir.display());
        }
        if era == EraPolicy::Verify {
            verify_pack_era(dir)?;
        }

        let mut shards = Vec::with_capacity(shard_names.len());
        let mut tensors = BTreeMap::new();
        for (idx, name) in shard_names.iter().enumerate() {
            let path = dir.join(name);
            let file = std::fs::File::open(&path)
                .with_context(|| format!("failed to open shard {}", path.display()))?;
            let map = unsafe { Mmap::map(&file) }
                .with_context(|| format!("failed to mmap shard {}", path.display()))?;
            let (header_len, meta) = parse_safetensors_header(&map)
                .with_context(|| format!("bad safetensors header in {}", path.display()))?;
            shards.push(Shard {
                map,
                data_base: 8 + header_len,
            });
            for (tname, tmeta) in meta {
                if tensors.insert(tname.clone(), (idx, tmeta)).is_some() {
                    bail!(
                        "duplicate tensor '{tname}' across shards in {}",
                        dir.display()
                    );
                }
            }
        }

        // Marker words for the group scan: read the 4-byte I32 values off
        // the mapped shards so `detect_exl3_layers` can verify them (§11.3
        // condition 3 — values, never booleans).
        let mut marker_words = BTreeMap::new();
        for (name, (_, meta)) in &tensors {
            if (name.ends_with(".mul1") || name.ends_with(".mcg"))
                && meta.dtype == "I32"
                && let Some(word) = read_u32(&tensors, &shards, name)
            {
                marker_words.insert(name.clone(), word);
            }
        }

        let projected: BTreeMap<String, TensorMeta> = tensors
            .iter()
            .map(|(k, (_, m))| (k.clone(), m.clone()))
            .collect();
        let plans = detect_exl3_layers(&projected, &marker_words);
        let plan_by_key = plans
            .iter()
            .enumerate()
            .map(|(i, p)| (p.key.clone(), i))
            .collect();
        Ok(Self {
            shards,
            tensors,
            plans,
            plan_by_key,
        })
    }

    /// Detected EXL3 group base keys (sorted, as produced by detection).
    pub fn layer_keys(&self) -> Vec<&str> {
        self.plans.iter().map(|p| p.key.as_str()).collect()
    }

    /// The detection plans (params / K / codebook without touching data).
    pub fn plans(&self) -> &[Exl3LayerPlan] {
        &self.plans
    }

    /// Number of shards actually mapped.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Total tensors in the merged map.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Raw data bytes of any tensor (EXL3 group member or dense leftover).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        self.slice_of(name)
            .with_context(|| format!("tensor '{name}' not present in pack"))
    }

    /// Zero-copy view over one EXL3 layer group. The group's tensors may
    /// live in different shards; all slices borrow `self`.
    pub fn layer(&self, key: &str) -> Result<Exl3Layer<'_>> {
        let idx = self
            .plan_by_key
            .get(key)
            .with_context(|| format!("no EXL3 group at key '{key}'"))?;
        let plan = &self.plans[*idx];
        let trellis = self.slice_of(&plan.trellis.0).with_context(|| {
            format!("group '{key}': trellis tensor vanished between scan and read")
        })?;

        let opt_slice = |name: &Option<(String, usize)>| -> Result<Option<&[u8]>> {
            match name {
                Some((n, _)) => Ok(Some(self.slice_of(n).with_context(|| {
                    format!("group '{key}': tensor '{n}' vanished between scan and read")
                })?)),
                None => Ok(None),
            }
        };

        let layer = Exl3Layer::from_raw_parts(
            trellis,
            opt_slice(&plan.suh)?,
            opt_slice(&plan.svh)?,
            opt_slice(&plan.su)?,
            opt_slice(&plan.sv)?,
            self.marker_word(key, "mcg")?,
            self.marker_word(key, "mul1")?,
            plan.in_features,
            plan.out_features,
        )
        .with_context(|| format!("group '{key}' failed byte-level validation"))?;
        Ok(layer)
    }

    /// Byte accounting by class, from metadata alone (T5's instrument).
    pub fn residency(&self) -> Exl3Residency {
        let len_of = |name: &str| self.tensors.get(name).map(|(_, m)| m.byte_len() as u64);
        let mut member_names = HashSet::new();
        let mut res = Exl3Residency::default();
        for plan in &self.plans {
            res.groups += 1;
            res.group_params += plan.in_features as u64 * plan.out_features as u64;
            res.trellis_bytes += len_of(&plan.trellis.0).unwrap_or(0);
            for scale in [&plan.suh, &plan.svh, &plan.su, &plan.sv]
                .into_iter()
                .flatten()
            {
                res.scale_bytes += len_of(&scale.0).unwrap_or(0);
                member_names.insert(scale.0.clone());
            }
            // Marker membership mirrors `detect_exl3_layers`: an I32
            // `{key}.{mcg|mul1}` counts; a foreign-dtype spelling of the
            // same name falls through to dense.
            for suffix in ["mcg", "mul1"] {
                let name = format!("{}.{}", plan.key, suffix);
                if let Some((_, meta)) = self.tensors.get(&name)
                    && meta.dtype == "I32"
                {
                    res.marker_bytes += meta.byte_len() as u64;
                    member_names.insert(name);
                }
            }
            if let Some(bias) = &plan.bias {
                res.group_bias_bytes += len_of(&bias.0).unwrap_or(0);
                member_names.insert(bias.0.clone());
            }
            member_names.insert(plan.trellis.0.clone());
        }
        for (name, (_, meta)) in &self.tensors {
            let len = meta.byte_len() as u64;
            res.total_tensor_bytes += len;
            if !member_names.contains(name) {
                res.dense_bytes += len;
                res.dense_params += meta.shape.iter().map(|&d| d as u64).product::<u64>();
            }
        }
        res
    }

    /// Read a `{key}.{which}` marker word straight off its shard. `None`
    /// when absent or not I32 (mirrors detection's filter); a present I32
    /// tensor whose bytes are unreadable is an error.
    fn marker_word(&self, key: &str, which: &str) -> Result<Option<u32>> {
        let name = format!("{key}.{which}");
        let Some((_, meta)) = self.tensors.get(&name) else {
            return Ok(None);
        };
        if meta.dtype != "I32" {
            return Ok(None);
        }
        let bytes = self.slice_of(&name).unwrap();
        let arr: [u8; 4] = bytes
            .try_into()
            .with_context(|| format!("marker '{name}' is not 4 bytes"))?;
        Ok(Some(u32::from_le_bytes(arr)))
    }

    fn slice_of(&self, name: &str) -> Option<&[u8]> {
        let (idx, meta) = self.tensors.get(name)?;
        let shard = &self.shards[*idx];
        let start = shard.data_base + meta.data_start;
        let end = start + meta.byte_len();
        shard.map.get(start..end)
    }
}

/// Read a 4-byte LE u32 at a tensor's data (open-time marker scan).
fn read_u32(
    tensors: &BTreeMap<String, (usize, TensorMeta)>,
    shards: &[Shard],
    name: &str,
) -> Option<u32> {
    let (idx, meta) = tensors.get(name)?;
    let shard = shards.get(*idx)?;
    let start = shard.data_base + meta.data_start;
    let end = start + meta.byte_len();
    let b: [u8; 4] = shard.map.get(start..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(b))
}

/// Which era check [`Exl3Pack::open_with_era_policy`] runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EraPolicy {
    Verify,
    Unverified,
}

/// The §12.7 pack-era gate: read the pack's writer version from either
/// config spelling and refuse fail-closed unless it is a validated era.
///
/// * `config.json` → embedded `quantization_config.{version, quant_method}`
///   (the exllamav3 writer has stamped this since versioning was
///   established — `conversion/compile.py` writes `"version": __version__`)
/// * `quantization_config.json` → the standalone HF spelling, same keys
///
/// Fail-closed in every unvalidated direction: no version found (the
/// pre-versioning legacy class — exllamav3 was at `"0.0.1"` in 2025-05,
/// the era of the §12.7 specimen), a version outside
/// [`KNOWN_GOOD_ERA_VERSIONS`], disagreeing spellings, or a present
/// `quant_method` that is not `exl3`.
fn verify_pack_era(dir: &Path) -> Result<String> {
    #[derive(Default)]
    struct Found {
        versions: Vec<String>,
        quant_methods: Vec<String>,
    }
    let mut found = Found::default();
    let mut read_spelling = |path: &Path, embedded: bool| -> Result<()> {
        if !path.exists() {
            return Ok(());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {} as JSON", path.display()))?;
        let qc = if embedded {
            v.get("quantization_config")
        } else {
            Some(&v)
        };
        let Some(qc) = qc else { return Ok(()) };
        if let Some(m) = qc.get("quant_method").and_then(|m| m.as_str()) {
            found.quant_methods.push(m.to_string());
        }
        if let Some(v) = qc.get("version").and_then(|v| v.as_str()) {
            found.versions.push(v.to_string());
        }
        Ok(())
    };
    read_spelling(&dir.join("config.json"), true)?;
    read_spelling(&dir.join("quantization_config.json"), false)?;

    for m in &found.quant_methods {
        if m != "exl3" {
            bail!(
                "pack-era gate: quantization_config quant_method is '{m}', not 'exl3' — \
                 {} is not an EXL3 pack",
                dir.display()
            );
        }
    }
    if found.versions.is_empty() {
        bail!(
            "pack-era gate: no quantization_config.version in {} \
             (config.json / quantization_config.json) — pre-versioning packs decode \
             WRONG under this reader (Issue 001 §12.7); deliberate read? \
             Exl3Pack::open_unverified_era",
            dir.display()
        );
    }
    if found.versions.len() > 1 && found.versions[0] != found.versions[1] {
        bail!(
            "pack-era gate: config spellings disagree on version ({} vs {}) in {} — \
             refusing rather than guessing",
            found.versions[0],
            found.versions[1],
            dir.display()
        );
    }
    let version = found.versions.remove(0);
    if !KNOWN_GOOD_ERA_VERSIONS.contains(&version.as_str()) {
        bail!(
            "pack-era gate: pack written by exllamav3 {version}; this reader is \
             validated on {} only (admission = the full-pack bit-exact gate of \
             Issue 001 §17.5 T7c-1c); deliberate read? Exl3Pack::open_unverified_era",
            KNOWN_GOOD_ERA_VERSIONS.join(", ")
        );
    }
    Ok(version)
}

/// Resolve the shard file list for `open`: index.json weight map if
/// present, else the sorted `*.safetensors` directory entries.
fn shard_file_names(dir: &Path) -> Result<Vec<String>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_json = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&index_json)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .with_context(|| format!("missing weight_map in {}", index_path.display()))?;
        let mut names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        names.sort();
        names.dedup();
        for n in &names {
            if !dir.join(n).exists() {
                bail!(
                    "index names shard '{n}' but {} is missing",
                    dir.join(n).display()
                );
            }
        }
        Ok(names)
    } else {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("failed to read dir {}", dir.display()))?
        {
            let entry = entry.with_context(|| "readdir iteration failed")?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".safetensors") {
                names.push(name);
            }
        }
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::super::exl3::{EXL3_MUL1_MARKER, Exl3Codebook, Exl3Layer};
    use super::*;

    /// Process-unique temp dir (the fixed-temp-path class — katgpt-rs
    /// Issue 832 — never a shared constant path).
    fn test_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir()
            .join(format!("exl3_pack_test_{}", std::process::id()))
            .join(tag);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Minimal hand-rolled safetensors writer for fixtures.
    fn write_shard(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let start = payload.len();
            payload.extend_from_slice(data);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, payload.len()],
                }),
            );
        }
        let hj = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::with_capacity(8 + hj.len() + payload.len());
        out.extend_from_slice(&(hj.len() as u64).to_le_bytes());
        out.extend_from_slice(&hj);
        out.extend_from_slice(&payload);
        std::fs::write(path, out).unwrap();
    }

    fn f16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect::<Vec<_>>()
            .concat()
    }

    /// Write a pack config carrying the given writer version. Real
    /// converted packs always carry one (the exllamav3 writer stamps it),
    /// so every fixture that expects `open` to succeed writes a
    /// known-good-era config; era-refusal fixtures write their own.
    fn write_config(dir: &Path, version: &str) {
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "quantization_config": {
                    "quant_method": "exl3",
                    "version": version,
                }
            })
            .to_string(),
        )
        .unwrap();
    }

    /// Minimal valid single-file pack (one 128×128 K=4 mul1 group), with a
    /// known-good-era config — the base fixture for era-gate tests.
    fn write_min_pack(dir: &Path) {
        let (trellis, suh, svh) = group_raw(21);
        write_shard(
            &dir.join("model.safetensors"),
            &[
                ("l.trellis", "I16", vec![8, 8, 64], trellis),
                ("l.suh", "F16", vec![128], suh),
                ("l.svh", "F16", vec![128], svh),
                (
                    "l.mul1",
                    "I32",
                    vec![],
                    EXL3_MUL1_MARKER.to_le_bytes().to_vec(),
                ),
            ],
        );
    }

    /// One 128×128 K=4 group's raw parts (random codes, benign scales).
    fn group_raw(seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let trellis: Vec<u8> = (0..8192).map(|_| rng.u8(..)).collect();
        let suh = f16_bytes(&(0..128).map(|i| 0.5 + i as f32 / 256.0).collect::<Vec<_>>());
        let svh = f16_bytes(
            &(0..128)
                .map(|i| 0.25 + i as f32 / 512.0)
                .collect::<Vec<_>>(),
        );
        (trellis, suh, svh)
    }

    #[test]
    fn pack_opens_sharded_group_spanning_and_dequant_matches() {
        let dir = test_dir("sharded");
        let (trellis, suh, svh) = group_raw(7);
        let (trellis_b, _, _) = group_raw(9);
        // Group A: trellis in shard B, scales+marker in shard A (the
        // lm_head pattern measured on the real 27B pack). Group B: legacy
        // su/sv signs, no marker (cb0), fully in shard B. Dense norm in A.
        write_shard(
            &dir.join("shard-a.safetensors"),
            &[
                ("a.linear.suh", "F16", vec![128], suh.clone()),
                ("a.linear.svh", "F16", vec![128], svh.clone()),
                (
                    "a.linear.mul1",
                    "I32",
                    vec![],
                    EXL3_MUL1_MARKER.to_le_bytes().to_vec(),
                ),
                ("model.norm.weight", "F16", vec![64], f16_bytes(&[0.5; 64])),
            ],
        );
        write_shard(
            &dir.join("shard-b.safetensors"),
            &[
                ("a.linear.trellis", "I16", vec![8, 8, 64], trellis.clone()),
                ("b.linear.trellis", "I16", vec![8, 8, 64], trellis_b),
                ("b.linear.su", "I16", vec![8], vec![0u8; 16]),
                ("b.linear.sv", "I16", vec![8], vec![0u8; 16]),
            ],
        );
        let weight_map = serde_json::json!({
            "a.linear.suh": "shard-a.safetensors",
            "a.linear.svh": "shard-a.safetensors",
            "a.linear.mul1": "shard-a.safetensors",
            "model.norm.weight": "shard-a.safetensors",
            "a.linear.trellis": "shard-b.safetensors",
            "b.linear.trellis": "shard-b.safetensors",
            "b.linear.su": "shard-b.safetensors",
            "b.linear.sv": "shard-b.safetensors",
        });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_string(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
        write_config(&dir, KNOWN_GOOD_ERA_VERSIONS[0]);

        let pack = Exl3Pack::open(&dir).unwrap();
        assert_eq!(pack.shard_count(), 2);
        assert_eq!(pack.layer_keys(), vec!["a.linear", "b.linear"]);

        // Group A spans shards and resolves Cb2Mul1 with the marker word.
        let layer = pack.layer("a.linear").unwrap();
        assert_eq!(layer.codebook, Exl3Codebook::Cb2Mul1);
        assert_eq!(layer.in_features, 128);
        assert_eq!(layer.out_features, 128);
        let via_pack = layer.dequantize_f32();

        // Direct construction from the same buffers must agree exactly.
        let direct = Exl3Layer::from_raw_parts(
            &trellis,
            Some(&suh),
            Some(&svh),
            None,
            None,
            None,
            Some(EXL3_MUL1_MARKER),
            128,
            128,
        )
        .unwrap();
        assert_eq!(via_pack, direct.dequantize_f32());

        // Group B: legacy sign path, cb0 default.
        let layer_b = pack.layer("b.linear").unwrap();
        assert_eq!(layer_b.codebook, Exl3Codebook::Cb0);

        // Residency accounting: 2 groups, trellis 2×8192, scales
        // suh+svh (512) + su+sv (32), marker 4, dense = norm (128 B).
        let res = pack.residency();
        assert_eq!(res.groups, 2);
        assert_eq!(res.trellis_bytes, 2 * 8192);
        assert_eq!(res.scale_bytes, 512 + 32);
        assert_eq!(res.marker_bytes, 4);
        assert_eq!(res.dense_bytes, 128);
        assert_eq!(res.group_params, 2 * 128 * 128);
        assert_eq!(res.dense_params, 64);
        assert_eq!(
            res.total_tensor_bytes,
            res.quantized_bytes() + res.dense_bytes
        );

        // Unknown key refuses; tensor_bytes serves dense leftovers.
        assert!(pack.layer("nope").is_err());
        assert_eq!(pack.tensor_bytes("model.norm.weight").unwrap().len(), 128);
    }

    #[test]
    fn pack_opens_single_file_without_index() {
        let dir = test_dir("single");
        let (trellis, suh, svh) = group_raw(11);
        write_shard(
            &dir.join("model.safetensors"),
            &[
                ("l.trellis", "I16", vec![8, 8, 64], trellis),
                ("l.suh", "F16", vec![128], suh),
                ("l.svh", "F16", vec![128], svh),
                (
                    "l.mul1",
                    "I32",
                    vec![],
                    EXL3_MUL1_MARKER.to_le_bytes().to_vec(),
                ),
            ],
        );
        write_config(&dir, KNOWN_GOOD_ERA_VERSIONS[0]);
        let pack = Exl3Pack::open(&dir).unwrap();
        assert_eq!(pack.shard_count(), 1);
        assert_eq!(pack.layer_keys(), vec!["l"]);
        let layer = pack.layer("l").unwrap();
        assert_eq!(layer.dequantize_f32().len(), 128 * 128);
    }

    /// A garbage marker word must NOT be silently decoded with a wrong
    /// codebook (§11.3 condition 3's hazard): the group scan drops it and
    /// `layer()` reports no group — loud refusal, never a quiet decode.
    #[test]
    fn bad_marker_word_refuses_loudly() {
        let dir = test_dir("bad_marker");
        let (trellis, suh, svh) = group_raw(13);
        write_shard(
            &dir.join("model.safetensors"),
            &[
                ("l.trellis", "I16", vec![8, 8, 64], trellis),
                ("l.suh", "F16", vec![128], suh),
                ("l.svh", "F16", vec![128], svh),
                (
                    "l.mul1",
                    "I32",
                    vec![],
                    0xDEAD_BEEFu32.to_le_bytes().to_vec(),
                ),
            ],
        );
        write_config(&dir, KNOWN_GOOD_ERA_VERSIONS[0]);
        let pack = Exl3Pack::open(&dir).unwrap();
        assert!(
            pack.layer_keys().is_empty(),
            "garbage marker must not decode"
        );
        let err = pack.layer("l").map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("no EXL3 group"), "got: {err}");
    }

    #[test]
    fn duplicate_tensor_across_shards_refused() {
        let dir = test_dir("dup");
        write_shard(
            &dir.join("a.safetensors"),
            &[("x", "F16", vec![2], vec![0, 0, 0, 0])],
        );
        write_shard(
            &dir.join("b.safetensors"),
            &[("x", "F16", vec![2], vec![0, 0, 0, 0])],
        );
        write_config(&dir, KNOWN_GOOD_ERA_VERSIONS[0]);
        let err = Exl3Pack::open(&dir).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "got: {err}");
    }

    /// §12.7 / T7c-4: absent version = the pre-versioning legacy class →
    /// `open` refuses fail-closed; the deliberate-read escape opens the
    /// same directory (naming the risk at the call site, by design).
    #[test]
    fn era_gate_absent_version_refuses_and_escape_opens() {
        let dir = test_dir("era_absent");
        write_min_pack(&dir);
        let err = Exl3Pack::open(&dir).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("pack-era gate"), "got: {msg}");
        assert!(msg.contains("open_unverified_era"), "got: {msg}");
        let pack = Exl3Pack::open_unverified_era(&dir).unwrap();
        assert_eq!(pack.layer_keys(), vec!["l"]);
    }

    /// Unknown writer version → refuse, naming the OBSERVED value (the
    /// operator must see what the pack claims). "0.0.1" is the measured
    /// legacy-era spelling (exllamav3 `version.py` at 2025-05-11).
    #[test]
    fn era_gate_unknown_version_refuses_with_observed_value() {
        let dir = test_dir("era_unknown");
        write_min_pack(&dir);
        write_config(&dir, "0.0.1");
        let err = Exl3Pack::open(&dir).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("pack-era gate"), "got: {msg}");
        assert!(msg.contains("0.0.1"), "got: {msg}");
        assert!(msg.contains("open_unverified_era"), "got: {msg}");
        assert!(Exl3Pack::open_unverified_era(&dir).is_ok());
    }

    /// The standalone `quantization_config.json` spelling is a first-class
    /// citizen (some HF repos split it out) — same version, same gate.
    #[test]
    fn era_gate_standalone_quantization_config_spelling_opens() {
        let dir = test_dir("era_standalone");
        write_min_pack(&dir);
        std::fs::write(
            dir.join("quantization_config.json"),
            serde_json::json!({
                "quant_method": "exl3",
                "version": KNOWN_GOOD_ERA_VERSIONS[0],
            })
            .to_string(),
        )
        .unwrap();
        let pack = Exl3Pack::open(&dir).unwrap();
        assert_eq!(pack.layer_keys(), vec!["l"]);
    }

    /// A directory that is not an EXL3 quantization at all (present
    /// `quant_method` spelling something else) refuses with that observed
    /// method — clearer than a downstream decode-shaped error.
    #[test]
    fn era_gate_wrong_quant_method_refuses() {
        let dir = test_dir("era_method");
        write_min_pack(&dir);
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "quantization_config": {
                    "quant_method": "awq",
                    "version": KNOWN_GOOD_ERA_VERSIONS[0],
                }
            })
            .to_string(),
        )
        .unwrap();
        let err = Exl3Pack::open(&dir).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("quant_method"), "got: {msg}");
        assert!(msg.contains("awq"), "got: {msg}");
    }

    /// Disagreeing spellings refuse rather than guess — an ambiguous pack
    /// is the operator's to inspect, not the loader's to adjudicate.
    #[test]
    fn era_gate_disagreeing_spellings_refuse() {
        let dir = test_dir("era_disagree");
        write_min_pack(&dir);
        write_config(&dir, KNOWN_GOOD_ERA_VERSIONS[0]);
        std::fs::write(
            dir.join("quantization_config.json"),
            serde_json::json!({
                "quant_method": "exl3",
                "version": "0.9.0",
            })
            .to_string(),
        )
        .unwrap();
        let err = Exl3Pack::open(&dir).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("disagree"), "got: {msg}");
        assert!(msg.contains("0.9.0"), "got: {msg}");
    }

    /// Pass-side evidence on the REAL pin-era pack (the n=1 the gate's
    /// blind-spot note names): `open` succeeds through the era gate on
    /// `version = "1.4.2"` and finds the 573-group surface. Opt-in via
    /// `EXL3_PACK_DIR`; skips loudly when unset.
    #[test]
    #[ignore = "needs a real EXL3 pack on disk (EXL3_PACK_DIR)"]
    fn real_pack_era_gate_opens() {
        let dir = std::env::var("EXL3_PACK_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("SKIPPED: EXL3_PACK_DIR not set");
            return;
        }
        let pack = Exl3Pack::open(Path::new(&dir)).unwrap();
        assert!(!pack.layer_keys().is_empty(), "no EXL3 groups detected");
        eprintln!("era gate passed: {} groups", pack.layer_keys().len());
    }

    /// Real-pack measurement (Issue 001 T5): residency table + sample
    /// dequantization classes with timing. Opt-in via `EXL3_PACK_DIR`;
    /// skips loudly when unset (the T4a `real_pack_oracle_k_proj`
    /// convention).
    #[test]
    #[ignore = "needs a real EXL3 pack on disk (EXL3_PACK_DIR)"]
    fn real_pack_residency_and_dequant_samples() {
        let dir = std::env::var("EXL3_PACK_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("SKIPPED: EXL3_PACK_DIR not set");
            return;
        }
        let pack = Exl3Pack::open(Path::new(&dir)).unwrap();
        let res = pack.residency();
        eprintln!(
            "shards={} tensors={} groups={}",
            pack.shard_count(),
            pack.tensor_count(),
            res.groups
        );
        eprintln!(
            "trellis={:.4} GiB scale={:.3} MiB markers={:.3} MiB bias={:.3} MiB dense={:.4} GiB",
            res.trellis_bytes as f64 / (1 << 30) as f64,
            res.scale_bytes as f64 / (1 << 20) as f64,
            res.marker_bytes as f64 / (1 << 20) as f64,
            res.group_bias_bytes as f64 / (1 << 20) as f64,
            res.dense_bytes as f64 / (1 << 30) as f64
        );
        eprintln!(
            "quantized={:.4} GiB params(G)={:.3} achieved_bpw={:.4} dequant_f16={:.3} GiB",
            res.quantized_bytes() as f64 / (1 << 30) as f64,
            res.group_params as f64 / 1e9,
            res.achieved_bpw(),
            res.dequantized_f16_bytes() as f64 / (1 << 30) as f64
        );

        // Smallest group of each named class — real packs carry small
        // vision-tower linears, so this stays cheap even in debug builds.
        let classes = [
            "self_attn.o_proj",
            "mlp.down_proj",
            "linear_attn.out_proj",
            "mlp.gate_proj",
            "lm_head",
        ];
        let mut measured_mw_per_s = Vec::new();
        // Optional export for the exllamav3-native oracle (EXL3_EXPORT_DIR;
        // `.raw/exl3_t5_oracle.py` consumes these — the §12.7 era-gate
        // applied to whatever pack sits under EXL3_PACK_DIR).
        let export_dir = std::env::var("EXL3_EXPORT_DIR")
            .ok()
            .map(std::path::PathBuf::from);
        if let Some(d) = &export_dir {
            let _ = std::fs::create_dir_all(d);
        }
        for class in classes {
            let Some(plan) = pack
                .plans()
                .iter()
                .filter(|p| p.key.contains(class))
                .min_by_key(|p| p.in_features as u64 * p.out_features as u64)
            else {
                eprintln!("class {class}: none");
                continue;
            };
            let t0 = std::time::Instant::now();
            let w = pack.layer(&plan.key).unwrap().dequantize_f32();
            let dt = t0.elapsed().as_secs_f64().max(1e-9);
            let n = w.len() as f64;
            let (mut mean, mut mx, mut nan) = (0.0f64, 0.0f64, 0usize);
            for &v in &w {
                if v.is_nan() {
                    nan += 1;
                }
                mean += v as f64;
                mx = mx.max(v.abs() as f64);
            }
            mean /= n;
            let rate = n / dt / 1e6;
            measured_mw_per_s.push(rate);
            // T7 CPU arm: parallel dequant timing + bit-parity spot check
            // on the smallest sampled class member.
            let t0p = std::time::Instant::now();
            let wp = pack.layer(&plan.key).unwrap().dequantize_f32_parallel();
            let dtp = t0p.elapsed().as_secs_f64().max(1e-9);
            let rate_p = n / dtp / 1e6;
            eprintln!(
                "  parallel: {dtp:.3}s {rate_p:.2} Mw/s ({:.2}x vs scalar)",
                rate_p / rate
            );
            if n <= 40_000_000.0 {
                assert_eq!(w, wp, "parallel/scalar parity broke on {}", plan.key);
            }
            if let Some(d) = &export_dir {
                let safe = class.replace('.', "_");
                let mut bytes = Vec::with_capacity(w.len() * 4);
                for v in &w {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(d.join(format!("{safe}.f32")), bytes).unwrap();
                std::fs::write(
                    d.join(format!("{safe}.meta")),
                    format!(
                        "{{\"key\":\"{}\",\"in\":{},\"out\":{}}}\n",
                        plan.key, plan.in_features, plan.out_features
                    ),
                )
                .unwrap();
            }
            eprintln!(
                "{class}: {} in={} out={} K={}.{} {:.3}s {:.2} Mw/s mean={mean:.5} max|W|={mx:.4} nan={nan}",
                plan.key,
                plan.in_features,
                plan.out_features,
                plan.k.ka,
                if plan.k.half { "5" } else { "0" },
                dt,
                rate
            );
            assert_eq!(nan, 0, "NaN in dequantized {}", plan.key);
        }
        // Single-threaded full-model dequant extrapolation (reference
        // posture — T7's fast arms replace this denominator).
        let avg = measured_mw_per_s.iter().sum::<f64>() / measured_mw_per_s.len().max(1) as f64;
        eprintln!(
            "extrapolated full-dequant (1 thread, scalar reference): {:.1} min",
            res.group_params as f64 / 1e6 / avg / 60.0
        );
    }
}
