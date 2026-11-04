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
//! scratch under their consumers): the run is measurement-only. The pass
//! is fixed-topology — no launch count, launch geometry, or CPU control
//! flow depends on GPU values — so the wall comparison is sound.
//! Skipped f16 casts still return the twin handle, so every surviving
//! consumer keeps its exact launch geometry.
//!
//! # The premise (refusals, not conventions)
//!
//! Ablation assumes the INCUMBENT launch set — the 942/943 baseline
//! schedule. Anything that changes it or serializes the queue refuses
//! loud at forward construction: the census (`RIIR_QWEN38_GDN_CENSUS=1`,
//! hazard-pinned markers), the stage timers (`RIIR_QWEN38_STAGE_TIMERS=1`,
//! per-stage syncs), the 1173 fused arms (`RIIR_QWEN38_FUSE_*` — they
//! remove launches), and the chunked-scan route
//! (`RIIR_QWEN38_GDN_CHUNKED=1` — a different scan launch set). The
//! decode path is NOT instrumented (issue 1168's lane owns decode).
//!
//! # The cells
//!
//! `RIIR_QWEN38_ABLATE=<csv|all>` arms one-shot cells; the bench bin's
//! `--ablate-matrix` drives the full table in-process (one model load,
//! interleaved base/ablated pass pairs per class — sequential arms
//! measure the box, never the kernel). `all` skips every instrumented
//! class: its delta is the pass's launch/dispatch floor, the free-running
//! cross-check on the 1173 ~0.02–0.05 ms/launch estimate.

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

/// One armed ablation: the set of classes to skip. `all` short-circuits
/// every `skip` (the launch-floor cell).
#[derive(Debug, Clone)]
pub struct Ablate {
    all: bool,
    classes: std::collections::HashSet<&'static str>,
}

impl Ablate {
    /// Parse one cell spec: `all`, or a comma-separated class list.
    /// Unknown classes refuse loud (a typo'd cell would silently measure
    /// an empty skip and read as a zero-cost class).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err("RIIR_QWEN38_ABLATE: empty spec (unset the env or name classes)".into());
        }
        if spec.eq_ignore_ascii_case("all") {
            return Ok(Self {
                all: true,
                classes: CLASSES.iter().copied().collect(),
            });
        }
        let mut classes = std::collections::HashSet::new();
        for tok in spec.split(',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            let class = CLASSES.iter().find(|c| **c == t).copied().ok_or_else(|| {
                format!(
                    "RIIR_QWEN38_ABLATE: unknown class '{t}' (known: {})",
                    CLASSES.join(", ")
                )
            })?;
            classes.insert(class);
        }
        if classes.is_empty() {
            return Err("RIIR_QWEN38_ABLATE: no class named".into());
        }
        Ok(Self {
            all: false,
            classes,
        })
    }

    /// A hand-built cell (the matrix bin drives cells in-process, no env).
    pub fn from_classes<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let mut classes = std::collections::HashSet::new();
        for t in names {
            let class = CLASSES.iter().find(|c| **c == t).copied().ok_or_else(|| {
                format!(
                    "ablate: unknown class '{t}' (known: {})",
                    CLASSES.join(", ")
                )
            })?;
            classes.insert(class);
        }
        if classes.is_empty() {
            return Err("ablate: no class named".into());
        }
        Ok(Self {
            all: false,
            classes,
        })
    }

    /// The launch-floor cell — skip every instrumented class.
    pub fn all() -> Self {
        Self {
            all: true,
            classes: CLASSES.iter().copied().collect(),
        }
    }

    /// `Some` when `RIIR_QWEN38_ABLATE` names a non-empty spec.
    pub fn from_env() -> Result<Option<Self>, String> {
        match std::env::var("RIIR_QWEN38_ABLATE") {
            Err(_) => Ok(None),
            Ok(s) if s.trim().is_empty() => Ok(None),
            Ok(s) => Self::parse(&s).map(Some),
        }
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
}
