//! Issue 1174 — the **free-running ablation instrument** (bench 944's
//! named next step): attribute the prefill pass wall WITHOUT serializing
//! the queue.
//!
//! # The mechanism
//!
//! Both instruments that attributed the pass wall serialize: the 942
//! `StageTimers` sync per stage, and the 943 census markers carry hazard
//! pinning + an event chain (the 944 A/B proved those walls are
//! boundary artifacts — fusing/removing ~240 launches bought only
//! +0.8–1.0% e2e). This instrument asks the pass itself: arm a class of
//! kernels to launch NOTHING (`skip`), diff the pp5120 pass wall. The
//! delta is the class's **free-running cost** — no markers, no syncs,
//! no hazards, no serialization anywhere.
//!
//! Outputs are bit-trashed by design (skipped producers leave stale
//! scratch under their consumers): the run is measurement-only. Both the
//! prefill pass and the decode step are fixed-topology — no launch count,
//! launch geometry, or CPU control flow depends on GPU values — so the
//! wall comparison is sound. Skipped f16 casts still return the twin
//! handle, so every surviving consumer keeps its exact launch geometry.
//!
//! # The premise (refusals, not conventions)
//!
//! Ablation assumes the INCUMBENT launch set — the 942/943 baseline
//! schedule. Anything that changes it or serializes the queue refuses
//! loud at forward construction: the census (`RIIR_QWEN38_GDN_CENSUS=1`,
//! hazard-pinned markers), the stage timers (`RIIR_QWEN38_STAGE_TIMERS=1`,
//! per-stage syncs), the 1173 fused arms (`RIIR_QWEN38_FUSE_*` — they
//! removing launches), and the chunked-scan route
//! (`RIIR_QWEN38_GDN_CHUNKED=1` — a different scan launch set).
//!
//! # The cells
//!
//! `RIIR_QWEN38_ABLATE=<csv|all>` arms one-shot cells; the bench bin's
//! `--ablate-matrix` drives the full table in-process (one model load,
//! interleaved base/ablated pass pairs per class — sequential arms
//! measure the box, never the kernel). `all` skips every instrumented
//! class: its delta is the pass's launch/dispatch floor, the free-running
//! cross-check on the 1173 ~0.02–0.05 ms/launch estimate.
//!
//! # The decode twin (issue 1178 T1)
//!
//! The same free-running pattern over `forward_token`: `DECODE_CLASSES`
//! name the per-step launch sites, the bench bin's `--decode-ablate-matrix`
//! drives interleaved (base, ablated) decode-cell pairs, and the per-class
//! Δwall is the class's free-running per-step cost — the decode frame
//! census (attn / GDN / FFN / embed / small-stages / dispatch+sync shares)
//! the Bench-937 prefill census calls for. The env arm is
//! `RIIR_QWEN38_DECODE_ABLATE` (never set together with the prefill arm —
//! construction refuses). Decode classes carry the `d_` prefix so the two
//! tables are disjoint by construction: a mis-scoped cell names nothing
//! live and skips NOTHING, never silently the wrong path.

/// The instrumented launch-site classes, in pass order (GDN layer
/// sections first — the names align with the 943 census sections where a
/// census section maps 1:1 to launches; the census's `mlp_norm_gateup`
/// section is split here into `mlp_norm` + `mlp_gateup_gemm`, and the
/// attention-layer + tail + shared-cast classes have no census section).
pub const CLASSES: [&str; 27] = [
    // Host gather + device write of the chunk embedding (a copy, not a
    // kernel — still on the wall).
    "wte_upload",
    // The layer-entry batched RMSNorm (BOTH layer kinds launch it — the
    // delta spans all layers; per-kind cost divides by the layer count,
    // the shapes are identical).
    "input_norm",
    // GDN layer: the four qkv/z/alpha/beta GEMMs (census `gemm_qkv_z_ab`).
    "gemm_qkv_z_ab",
    "beta_decay",
    // p/64 chunked conv1d+SiLU launches (8 per GDN layer at chunk 512).
    "conv1d",
    "expand_l2",
    // The multi-token GDN recurrence (census `scan`).
    "scan",
    "norm_ssm",
    "z_gate",
    "ssm_out_gemm",
    // The same-layer post-ssm residual add (census `residual_gdn`).
    "residual_gdn",
    // Attention layer kernels.
    "attn_qkv_gemm",
    "split_qg",
    "qk_norm",
    "rope",
    "kv_cache_fill",
    // The attention scores pass — BOTH arms (tensor-flash and tiled) are
    // skipped by this class; skipping only one arm would swap the launch
    // set, not remove it.
    "attn_scores",
    // The attention layer's post-attn residual add.
    "attn_residual",
    "attn_out_gemm",
    // MLP (shared by both layer kinds — identical shapes).
    "mlp_norm",
    "mlp_gateup_gemm",
    "mlp_act",
    "mlp_down_gemm",
    // The trailing residual add (census `mlp_residual`; the cross-layer
    // F3 shape is refused under ablation — the plain ResidualAdd runs).
    "mlp_residual",
    // Every f32→f16 staging cast (issue 1170 half-B). Skipping keeps the
    // twin handle, so the consuming GEMM's geometry never changes.
    "f16_cast",
    // Tail: the last-row output norm + the lm_head GEMVs.
    "tail_norm",
    "lm_head_gemm",
];

/// The 14 census section names (bench 943) and how each maps to ablation
/// classes — the mapping the report's per-section comparison needs. 12
/// map 1:1; `mlp_norm_gateup` is the documented two-class split.
pub const CENSUS_SECTION_MAP: [(&str, &[&str]); 14] = [
    ("input_norm", &["input_norm"]),
    ("gemm_qkv_z_ab", &["gemm_qkv_z_ab"]),
    ("beta_decay", &["beta_decay"]),
    ("conv1d", &["conv1d"]),
    ("expand_l2", &["expand_l2"]),
    ("scan", &["scan"]),
    ("norm_ssm", &["norm_ssm"]),
    ("z_gate", &["z_gate"]),
    ("ssm_out_gemm", &["ssm_out_gemm"]),
    ("residual_gdn", &["residual_gdn"]),
    ("mlp_norm_gateup", &["mlp_norm", "mlp_gateup_gemm"]),
    ("mlp_act", &["mlp_act"]),
    ("mlp_down_gemm", &["mlp_down_gemm"]),
    ("mlp_residual", &["mlp_residual"]),
];

/// Issue 1178 T1 — the decode-step classes (`forward_token`'s launch
/// sites, in step order). The `d_` prefix keeps this table disjoint from
/// [`CLASSES`] so a prefill cell can never gate a decode site or vice
/// versa. One branch per site; `d_entry_norm`/`d_ffn`/`d_residual` span
/// BOTH layer kinds (identical shapes — the delta divides by the layer
/// total), GDN classes by the GDN count, attn classes by the attn count,
/// and `d_embed`/`d_out_norm`/`d_lm_head` run once per step. The
/// per-token logits read + host argmax are NOT instrumented — the
/// decode loop's value dependency; their mass lands in the ALL cell's
/// residual wall (the sync/host floor).
pub const DECODE_CLASSES: [&str; 13] = [
    // Host write of the wte row (a copy, not a kernel — still on the wall).
    "d_embed",
    // Layer-entry RMSNorm (both layer kinds launch it).
    "d_entry_norm",
    // GDN layer: the four qkv/z/alpha/beta GEMVs.
    "d_gdn_gemm",
    // GDN pre-scan stages: beta_decay + conv1d + expand_l2.
    "d_gdn_stage",
    // The single-token recurrence (DeltanetRecurrenceRowPar).
    "d_gdn_scan",
    // Post-scan: Zgate-fused norm + ssm_out GEMV.
    "d_gdn_post",
    // Attention layer: the q/k/v GEMVs + the wo output projection.
    "d_attn_gemm",
    // Attention pre-scores stages: split_qg + qk_norm + rope + kv append.
    "d_attn_stage",
    // The attention scores pass (QwenAttentionDecodeGated; reads pos+1 KV
    // rows — the ctx-dependent class).
    "d_attn_scores",
    // SwiGLU MLP body (both layer kinds): norm + gate/up GEMVs + gating +
    // down GEMV (the trailing residual is d_residual).
    "d_ffn",
    // The per-layer residual adds (three per layer, both kinds).
    "d_residual",
    // Tail: the output norm.
    "d_out_norm",
    // Tail: the lm_head GEMV chunks (f16 + q6k).
    "d_lm_head",
];

/// Which launch table the cell's names come from — recorded so the bench
/// drivers and the env double-arm refusal can check scope explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Prefill,
    Decode,
}

/// One armed ablation: the set of classes to skip. `all` short-circuits
/// every `skip` (the launch-floor cell).
#[derive(Debug, Clone)]
pub struct Ablate {
    all: bool,
    classes: std::collections::HashSet<&'static str>,
    scope: Scope,
}

impl Ablate {
    /// Parse one cell spec: `all`, or a comma-separated class list
    /// (prefill table).
    pub fn parse(spec: &str) -> Result<Self, String> {
        Self::parse_table(spec, &CLASSES, Scope::Prefill, "RIIR_QWEN38_ABLATE")
    }

    /// Parse one decode cell spec against [`DECODE_CLASSES`].
    pub fn parse_decode(spec: &str) -> Result<Self, String> {
        Self::parse_table(
            spec,
            &DECODE_CLASSES,
            Scope::Decode,
            "RIIR_QWEN38_DECODE_ABLATE",
        )
    }

    fn parse_table(
        spec: &str,
        table: &[&'static str],
        scope: Scope,
        env_name: &str,
    ) -> Result<Self, String> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err(format!(
                "{env_name}: empty spec (unset the env or name classes)"
            ));
        }
        if spec.eq_ignore_ascii_case("all") {
            return Ok(Self {
                all: true,
                classes: table.iter().copied().collect(),
                scope,
            });
        }
        let mut classes = std::collections::HashSet::new();
        for tok in spec.split(',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            let class = table.iter().find(|c| **c == t).copied().ok_or_else(|| {
                format!(
                    "{env_name}: unknown class '{t}' (known: {})",
                    table.join(", ")
                )
            })?;
            classes.insert(class);
        }
        if classes.is_empty() {
            return Err(format!("{env_name}: no class named"));
        }
        Ok(Self {
            all: false,
            classes,
            scope,
        })
    }

    /// A hand-built cell (the matrix bin drives cells in-process, no env).
    pub fn from_classes<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        Self::from_classes_table(names, &CLASSES, Scope::Prefill, "ablate")
    }

    /// A hand-built decode cell (the `--decode-ablate-matrix` driver).
    pub fn from_classes_decode<'a>(
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, String> {
        Self::from_classes_table(names, &DECODE_CLASSES, Scope::Decode, "decode-ablate")
    }

    fn from_classes_table<'a>(
        names: impl IntoIterator<Item = &'a str>,
        table: &[&'static str],
        scope: Scope,
        what: &str,
    ) -> Result<Self, String> {
        let mut classes = std::collections::HashSet::new();
        for t in names {
            let class = table.iter().find(|c| **c == t).copied().ok_or_else(|| {
                format!("{what}: unknown class '{t}' (known: {})", table.join(", "))
            })?;
            classes.insert(class);
        }
        if classes.is_empty() {
            return Err(format!("{what}: no class named"));
        }
        Ok(Self {
            all: false,
            classes,
            scope,
        })
    }

    /// The launch-floor cell — skip every instrumented class.
    pub fn all() -> Self {
        Self::all_table(&CLASSES, Scope::Prefill)
    }

    /// The decode launch-floor cell.
    pub fn all_decode() -> Self {
        Self::all_table(&DECODE_CLASSES, Scope::Decode)
    }

    fn all_table(table: &[&'static str], scope: Scope) -> Self {
        Self {
            all: true,
            classes: table.iter().copied().collect(),
            scope,
        }
    }

    /// `Some` when `RIIR_QWEN38_ABLATE` names a non-empty spec.
    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_env_named("RIIR_QWEN38_ABLATE")
    }

    /// `Some` when `RIIR_QWEN38_DECODE_ABLATE` names a non-empty spec
    /// (the decode twin — never set together with the prefill arm; the
    /// forward's construction refuses the pair).
    pub fn from_env_decode() -> Result<Option<Self>, String> {
        Self::from_env_named("RIIR_QWEN38_DECODE_ABLATE")
    }

    fn from_env_named(env_name: &str) -> Result<Option<Self>, String> {
        match std::env::var(env_name) {
            Err(_) => Ok(None),
            Ok(s) if s.trim().is_empty() => Ok(None),
            Ok(s) => {
                let decode = env_name == "RIIR_QWEN38_DECODE_ABLATE";
                if decode {
                    Self::parse_decode(&s).map(Some)
                } else {
                    Self::parse(&s).map(Some)
                }
            }
        }
    }

    /// Which table this cell names.
    pub fn scope(&self) -> Scope {
        self.scope
    }

    /// Whether `class`'s launches are skipped this cell.
    pub fn skip(&self, class: &str) -> bool {
        self.all || self.classes.contains(class)
    }

    /// Whether any class is skipped (the field stays `Some` on the
    /// forward so the skip branches are live).
    pub fn is_empty(&self) -> bool {
        !self.all && self.classes.is_empty()
    }

    /// The armed classes for the boot banner.
    pub fn describe(&self) -> String {
        if self.all {
            "all".into()
        } else {
            let mut names: Vec<&str> = self.classes.iter().copied().collect();
            names.sort_unstable();
            names.join(",")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_and_csv() {
        let a = Ablate::parse("all").expect("all");
        assert!(a.all);
        for c in CLASSES {
            assert!(a.skip(c), "all must skip {c}");
        }
        let a = Ablate::parse("conv1d, scan ,beta_decay").expect("csv");
        assert!(!a.all);
        assert!(a.skip("conv1d") && a.skip("scan") && a.skip("beta_decay"));
        assert!(!a.skip("expand_l2"));
    }

    #[test]
    fn parse_refuses_unknown_and_empty() {
        assert!(Ablate::parse("conv1d,nope").is_err());
        assert!(Ablate::parse("").is_err());
        assert!(Ablate::parse("   ").is_err());
        assert!(Ablate::parse(",,").is_err());
    }

    #[test]
    fn from_classes_roundtrip_and_refusal() {
        let a = Ablate::from_classes(["mlp_act", "scan"]).expect("classes");
        assert!(a.skip("mlp_act") && a.skip("scan"));
        assert!(Ablate::from_classes(["nope"]).is_err());
        assert!(Ablate::from_classes([]).is_err());
    }

    #[test]
    fn every_census_section_maps_to_classes() {
        // The 943 census sections (qwen38_gdn_census::SECTIONS) must stay
        // covered by the ablation taxonomy — a census rename without the
        // mapping update reds here.
        let census = [
            "input_norm",
            "gemm_qkv_z_ab",
            "beta_decay",
            "conv1d",
            "expand_l2",
            "scan",
            "norm_ssm",
            "z_gate",
            "ssm_out_gemm",
            "residual_gdn",
            "mlp_norm_gateup",
            "mlp_act",
            "mlp_down_gemm",
            "mlp_residual",
        ];
        assert_eq!(census.len(), CENSUS_SECTION_MAP.len());
        let mapped: Vec<&str> = CENSUS_SECTION_MAP.iter().map(|(s, _)| *s).collect();
        assert_eq!(
            mapped, census,
            "census sections and the mapping must stay positional-identical"
        );
        for (_, classes) in CENSUS_SECTION_MAP {
            for c in classes {
                assert!(
                    CLASSES.contains(c),
                    "census section mapping names unknown class {c}"
                );
            }
        }
    }

    #[test]
    fn decode_parse_all_and_csv() {
        let a = Ablate::parse_decode("all").expect("all");
        assert_eq!(a.scope(), Scope::Decode);
        for c in DECODE_CLASSES {
            assert!(a.skip(c), "all must skip {c}");
        }
        let a = Ablate::parse_decode("d_gdn_scan , d_ffn").expect("csv");
        assert!(!a.all);
        assert!(a.skip("d_gdn_scan") && a.skip("d_ffn"));
        assert!(!a.skip("d_embed"));
        let a = Ablate::from_classes_decode(["d_attn_scores"]).expect("classes");
        assert!(a.skip("d_attn_scores"));
        let all = Ablate::all_decode();
        assert!(all.skip("d_lm_head") && all.scope() == Scope::Decode);
    }

    #[test]
    fn decode_parse_refuses_unknown_and_prefill_names() {
        assert!(Ablate::parse_decode("d_gdn_scan,nope").is_err());
        assert!(Ablate::parse_decode("").is_err());
        // The scope wall: a PREFILL class name is unknown in the decode
        // table (and vice versa) — a mis-scoped cell refuses instead of
        // silently skipping nothing.
        assert!(Ablate::parse_decode("scan").is_err());
        assert!(Ablate::parse("d_gdn_scan").is_err());
        assert!(Ablate::from_classes_decode([]).is_err());
    }

    #[test]
    fn decode_and_prefill_tables_are_disjoint() {
        // The d_ prefix law: one name in both tables would let a
        // mis-scoped cell gate the wrong path's site.
        for d in DECODE_CLASSES {
            assert!(
                !CLASSES.contains(&d),
                "decode class {d} must not collide with the prefill table"
            );
            assert!(
                d.starts_with("d_"),
                "decode class {d} must carry the d_ prefix"
            );
        }
    }

    #[test]
    fn decode_classes_cover_the_step_categories() {
        // The 1178 T1 coarse buckets must stay derivable from the table —
        // a class rename without the roll-up update reds here.
        for c in [
            "d_embed",
            "d_entry_norm",
            "d_gdn_gemm",
            "d_gdn_stage",
            "d_gdn_scan",
            "d_gdn_post",
            "d_attn_gemm",
            "d_attn_stage",
            "d_attn_scores",
            "d_ffn",
            "d_residual",
            "d_out_norm",
            "d_lm_head",
        ] {
            assert!(DECODE_CLASSES.contains(&c), "bucket class {c} missing");
        }
    }
}
