//! Shared harness machinery for the Issue 919 T3 KV-cache A/B (both cells).
//!
//! Extracted VERBATIM from the gemma cell-1 bin (`spike_census_t3_kv_ab`)
//! so the ternary/GDN cell 2 (`spike_census_t3_kv_ab_ternary`) drives the
//! identical families, scoring, and arm aggregates — the two cells' numbers
//! stay comparable by construction. Pure code motion: the gemma bin's
//! behavior is unchanged.

/// Calibration passages — byte-identical to the T2 SPCM calibration fixture
/// (`riir-ai` `spike_census_calib_dump.rs`), so the KV diagonal here is
/// cross-referable with the FFN SPCM sidecars. Each passage is one FAMILY
/// (per-family retention law).
pub const PASSAGES: [&str; 4] = [
    "The harbour city grew along the river because ships could carry grain \
     farther than any cart. Merchants built warehouses near the docks, and \
     the warehouses needed clerks, and the clerks needed houses, and within \
     three generations the mudflats had become a town with a charter, a \
     bell tower, and a stubborn council that argued about bridge tolls. \
     When the railway arrived the council argued again, because a line \
     drawn through the valley would cut the orchards in half, but the \
     orchards were already dying of blight, and the freight fees paid for \
     the new school. The school still stands. Its registers record the \
     names of every child who learned to read within earshot of the \
     goods yard, and the ledgers show that the toll dispute ended only \
     when the bridge collapsed and nobody could afford to rebuild it.",
    "A compiler translates source text into machine instructions through a \
     pipeline of passes. The front end parses the text into an abstract \
     syntax tree and resolves every name to a declaration. The middle end \
     optimizes: common subexpressions are computed once, loops are \
     unrolled or interchanged, and values are kept in registers as long \
     as possible. The back end selects instructions, schedules them \
     against latency, and allocates registers under pressure. Each pass \
     must preserve the semantics of the program, which is why correct \
     compilers are tested against specifications rather than examples. \
     Optimization levels trade compile time for execution speed; at the \
     highest level the compiler may vectorize inner loops, inline \
     functions across module boundaries, and reorder memory operations \
     that the source language promised not to reorder, provided the \
     hardware model says the result cannot be observed.",
    "Mira found the key taped under the third stair, exactly where the \
     letter said, though the tape had yellowed and the key had left a \
     rust shadow on the wood. The door at the end of the corridor had \
     not been opened in years; the air behind it smelled of paper and \
     cold dust. Inside, shelves rose to the ceiling, every one of them \
     full, and in the middle of the room stood a desk with a single \
     drawer. She did not open the drawer at first. She walked the \
     aisles instead, reading spines by the light of her phone: \
     field guides, city directories, ship manifests, and notebooks in \
     a hand that grew more hurried with every decade. The last notebook \
     stopped mid-sentence. Its final page held only an address and the \
     words: ask for the tide table.",
    "\"You said the samples were clean,\" Dana said. She did not look up \
     from the chromatogram. \"I said the blanks were clean,\" Idris \
     replied. \"The blanks and the samples came from the same tray.\" \
     \"Then the tray is compromised.\" \"The tray is fine. The pipette \
     is the problem. Watch: same solution, second pipette, and the \
     peak disappears.\" Dana turned the printout toward him. \"Then why \
     does the peak come back when we run it tomorrow?\" Idris was quiet \
     for a moment. \"Because tomorrow is a different day, and the \
     instrument drifts with the weather, and we are measuring something \
     very small on top of something very noisy.\" \"So we repeat it.\" \
     \"So we repeat it, and we log the barometric pressure, and we stop \
     pretending the number has more digits than the experiment \
     deserves.\"",
];

/// One family = one passage repeated [`REPEATS`] times, split at the midpoint:
/// the first half calibrates (the diagonal + the exemption sets), the
/// second half scores. Register-matched, held-out.
pub const REPEATS: usize = 3;

/// Chunk positions ≤ this count are "early" (the sink proxy, vk Gate-4).
pub const EARLY_POS: usize = 8;

pub struct Args {
    pub gguf: PathBuf,
    pub out: PathBuf,
    pub s: usize,
    pub seq_len: usize,
    pub select_rms: bool,
    pub seed: u64,
}

pub fn parse_args(default_gguf: &str, default_out: &str) -> Result<Args> {
    let mut a = Args {
        gguf: PathBuf::from(default_gguf),
        out: PathBuf::from(default_out),
        s: 2,
        seq_len: 512,
        select_rms: false,
        seed: 919,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                a.gguf = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--s" => {
                a.s = args[i + 1].parse().context("--s N")?;
                i += 2;
            }
            "--seq-len" => {
                a.seq_len = args[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--select" => {
                a.select_rms = match args[i + 1].as_str() {
                    "max" => false,
                    "rms" => true,
                    other => bail!("--select max|rms, got {other}"),
                };
                i += 2;
            }
            "--seed" => {
                a.seed = args[i + 1].parse().context("--seed N")?;
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if a.seq_len == 0 || a.seq_len > 4096 {
        bail!("--seq-len must be in 1..=4096");
    }
    if a.s == 0 || a.s > 32 {
        bail!("--s must be in 1..=32");
    }
    Ok(a)
}

/// Deterministic splitmix64 — the random-control channel sets. No dep; the
/// stream is pinned by the seed and consumed in a fixed order
/// (layer-major, k-then-v), so the sets are byte-reproducible.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// Named `next_u64` (not `next`) — clippy::should_implement_trait reads a
    /// non-trait `next` as an Iterator impl.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `s` strictly ascending distinct channels from `0..kvd`, seeded order.
    pub fn pick_channels(&mut self, kvd: usize, s: usize) -> Vec<usize> {
        let mut chosen = std::collections::BTreeSet::new();
        while chosen.len() < s {
            chosen.insert((self.next_u64() % kvd as u64) as usize);
        }
        chosen.into_iter().collect()
    }
}

pub fn nll(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&l| (l as f64 - m).exp()).sum();
    m + z.ln() - logits[target] as f64
}

/// One family's eval token stream (the held-out half, chunked at `seq_len`).
pub struct Family {
    pub name: &'static str,
    pub seqs: Vec<Vec<usize>>,
}

/// Per-arm scoring result: the global per-token NLL vector (fixed
/// family-major, seq-major, pos-major order — the paired-delta key), the
/// early-position sink-proxy share, and per-family (ppl, n).
pub struct ArmScore {
    pub name: &'static str,
    pub nlls: Vec<f64>,
    pub early_nll: f64,
    pub early_n: usize,
    pub fam: Vec<(&'static str, f64, usize)>,
}

impl ArmScore {
    pub fn aggregate_ppl(&self) -> f64 {
        (self.nlls.iter().sum::<f64>() / self.nlls.len().max(1) as f64).exp()
    }

    pub fn early_ppl(&self) -> f64 {
        (self.early_nll / self.early_n.max(1) as f64).exp()
    }

    /// Paired mean |Δnll| against another arm (position-aligned).
    pub fn mean_abs_delta(&self, base: &ArmScore) -> f64 {
        self.nlls
            .iter()
            .zip(&base.nlls)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / self.nlls.len().max(1) as f64
    }
}

pub fn build_arm(
    name: &'static str,
    families: &[Family],
    nlls: Vec<f64>,
    early_nll: f64,
    early_n: usize,
) -> ArmScore {
    let mut fam = Vec::with_capacity(families.len());
    let mut idx = 0usize;
    for f in families {
        let n: usize = f.seqs.iter().map(|s| s.len().saturating_sub(1)).sum();
        let fam_nll: f64 = nlls[idx..idx + n].iter().sum();
        fam.push((f.name, fam_nll / n.max(1) as f64, n));
        idx += n;
    }
    ArmScore {
        name,
        nlls,
        early_nll,
        early_n,
        fam,
    }
}

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
