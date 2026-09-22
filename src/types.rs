// riir-engine types: re-exports from katgpt-core + riir-engine-specific types.
//
// All shared types (Config, Rng, math utilities, LoRA, DomainLatent, AttentionMode)
// are defined in katgpt-core and re-exported here.
//
// riir-engine-specific additions:
// - NoiseSchedule (feature-gated with `dllm`)

// Re-export all shared types from core
pub use katgpt_core::types::*;

// ── Training sample vocabulary (shared: rtg dataloader + game encoders) ──
//
// Issue 741 TAIL P3 (2026-08-23): these two types were born in
// riir-gpu/src/dataloader.rs (moved to riir-train-gpu). They are the
// cross-crate vocabulary between game sample encoders and the training
// DataLoader — riir-train-engine's `game::go` encoder (TAIL P4 move) +
// riir-train-gpu's dataloader construct the SAME type, so the definition
// lives here (the lowest crate both sides see). `pub use` from dataloader
// keeps the historical import paths working.

/// Single training sample loaded from JSONL or produced by a game encoder.
/// Expects JSON lines with a "tokens" field: `{"tokens": [1, 2, 3, ...]}`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TrainingSample {
    pub tokens: Vec<usize>,
    /// Optional role gates: 0 = world (supervise), 1 = agent (mask from loss).
    /// If absent, all tokens are treated as world (standard SFT, backward compatible).
    #[serde(default)]
    pub role_gates: Vec<u8>,
}

/// Token provenance for interventional SFT.
///
/// From Pearl's do-calculus: an agent's own output is an intervention (do(a)),
/// not an observation. Standard SFT treats all tokens as evidence, causing
/// self-confirming delusions. `RoleGate` marks which tokens to exclude from loss.
///
/// Paper: de Freitas & Ortega, "Causal interactive LLM agents that tell the truth" (2026)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RoleGate {
    /// World-written token (user, tool output, environment).
    /// Contributes to loss — this is genuine evidence about the world.
    World = 0,
    /// Agent-written token (model output, tool call, action).
    /// Masked from loss — this is an intervention, not evidence.
    /// Still included in conditioning context for next-token prediction.
    Agent = 1,
}

impl RoleGate {
    /// Returns true if this token should be masked from the loss.
    #[inline]
    pub fn is_agent(&self) -> bool {
        matches!(self, RoleGate::Agent)
    }
}

// ── NoiseSchedule — Discrete Diffusion Language Model (Plan 068: D2F) ────

/// Noise schedule for Discrete Diffusion Language Model training (Plan 068: D2F).
#[cfg(feature = "dllm")]
pub struct NoiseSchedule {
    /// Number of blocks in the schedule.
    pub n_blocks: usize,
    /// Minimum mask ratio for the first block (typically 0.2).
    pub min_ratio: f32,
    /// Maximum mask ratio for the last block (typically 1.0).
    pub max_ratio: f32,
}

#[cfg(feature = "dllm")]
impl NoiseSchedule {
    /// Create a new noise schedule with the given parameters.
    pub fn new(min_ratio: f32, max_ratio: f32, n_blocks: usize) -> Self {
        Self {
            min_ratio,
            max_ratio,
            n_blocks,
        }
    }

    /// Generate monotonically non-decreasing mask ratios per block.
    ///
    /// Matches reference: `generate_monotonic_pmasks()` in D2F-train/utils/util.py:15-34
    /// - p₀ ∈ [`min_ratio`, 0.7]
    /// - Each subsequent pᵢ ≥ pᵢ₋₁ (non-decreasing)
    /// - Last value ≤ `max_ratio`
    pub fn monotonic_ratios(&self, rng: &mut Rng) -> Vec<f32> {
        if self.n_blocks == 0 {
            return Vec::new();
        }

        let mut ratios = Vec::with_capacity(self.n_blocks);

        // First block: uniform sample in [min_ratio, 0.7]
        let upper_first = 0.7_f32.min(self.max_ratio);
        let p0 = self.min_ratio + rng.uniform() * (upper_first - self.min_ratio);
        ratios.push(p0);

        // Subsequent blocks: non-decreasing increments
        let remaining_budget = self.max_ratio - p0;
        let avg_increment = if self.n_blocks > 1 {
            remaining_budget / (self.n_blocks - 1) as f32
        } else {
            0.0
        };

        for _i in 1..self.n_blocks {
            let prev = ratios[_i - 1];
            // Sample increment centered around average, clamped to [0, remaining]
            let max_inc = self.max_ratio - prev;
            let inc = if max_inc <= 0.0 {
                0.0
            } else {
                // Exponential-ish distribution: smaller increments more likely
                (avg_increment * rng.uniform() * 2.0).min(max_inc)
            };
            ratios.push((prev + inc).min(self.max_ratio));
        }

        ratios
    }

    /// Corrupt tokens in a single block at given mask ratio.
    ///
    /// Matches reference: `forward_process_block_fixed_p()` in D2F-train/utils/util.py:6-13
    /// Each token is independently replaced with `mask_token` with probability `mask_ratio`.
    /// Returns (`corrupted_tokens`, `mask_indicators`) where `mask_indicators[i]` = true means masked.
    pub fn corrupt_block(
        &self,
        tokens: &[usize],
        mask_ratio: f32,
        mask_token: usize,
        rng: &mut Rng,
    ) -> (Vec<usize>, Vec<bool>) {
        let len = tokens.len();
        let mut corrupted = Vec::with_capacity(len);
        let mut mask_indicators = Vec::with_capacity(len);

        for &t in tokens {
            let masked = rng.uniform() < mask_ratio;
            corrupted.push(if masked { mask_token } else { t });
            mask_indicators.push(masked);
        }

        (corrupted, mask_indicators)
    }
}
