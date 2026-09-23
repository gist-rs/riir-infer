//! Hybrid DeltaNet/Attention dispatch for mixed-architecture inference (Plan 182, Phase 3).
//!
//! Routes each layer to either standard attention (GEMV + flash attention + MLP)
//! or DeltaNet (GEMV + conv1d + recurrence + gating + MLP) based on per-layer config.
//!
//! # Dispatch Flow
//!
//! ```text
//! for each layer:
//!   match layer_types[i]:
//!     Attention → standard attention path (existing GEMV + KV cache + attention + MLP)
//!     DeltaNet  → DeltaNet path (GEMV q/k/v + conv1d + recurrence + gating + MLP)
//! ```
//!
//! # Optimization Alignment (per .contexts/optimization.md)
//!
//! - 1 dispatch per layer: no branching overhead in the hot path
//! - Pre-allocated state buffers: DeltaNet state persists across decode steps
//! - Reuse existing GEMV kernel for q/k/v and MLP projections
//! - Profile first: both paths produce the same output shape, benchmarking is straightforward

#[cfg(feature = "deltanet_inference")]
use crate::deltanet_cubecl::DeltaNetStateBuffers;
#[cfg(feature = "deltanet_inference")]
use cubecl::prelude::*;
#[cfg(feature = "deltanet_inference")]
use crate::cubecl_runtime::ActiveRuntime;

#[cfg(feature = "deltanet_inference")]
use riir_infer_core::types::{Config, DeltaNetLayerType};

// ---------------------------------------------------------------------------
// Hybrid layer dispatcher
// ---------------------------------------------------------------------------

/// Routes each layer to the appropriate forward pass based on layer type.
///
/// Created once at model load time with the layer type map from Config.
/// Used by the forward pass to determine which kernel path to take per layer.
#[cfg(feature = "deltanet_inference")]
pub struct HybridLayerDispatch {
    /// Per-layer type (Attention or DeltaNet).
    layer_types: Vec<DeltaNetLayerType>,
    /// Number of DeltaNet layers (pre-computed for buffer allocation).
    n_deltanet_layers: usize,
    /// Number of attention layers.
    n_attention_layers: usize,
    /// Pre-computed DeltaNet layer indices (avoids collect() per dispatch).
    deltanet_indices: Vec<usize>,
    /// Pre-computed Attention layer indices (avoids collect() per dispatch).
    attention_indices: Vec<usize>,
}

#[cfg(feature = "deltanet_inference")]
impl HybridLayerDispatch {
    /// Create from config's layer_types field.
    ///
    /// If `layer_types` is empty, all layers default to Attention.
    pub fn new(config: &Config) -> Self {
        let layer_types = if config.layer_types.is_empty() {
            vec![DeltaNetLayerType::Attention; config.n_layer]
        } else {
            config.layer_types.clone()
        };

        let mut deltanet_indices = Vec::new();
        let mut attention_indices = Vec::new();
        for (i, lt) in layer_types.iter().enumerate() {
            match lt {
                DeltaNetLayerType::DeltaNet => deltanet_indices.push(i),
                DeltaNetLayerType::Attention => attention_indices.push(i),
            }
        }
        let n_deltanet_layers = deltanet_indices.len();
        let n_attention_layers = attention_indices.len();

        Self {
            layer_types,
            n_deltanet_layers,
            n_attention_layers,
            deltanet_indices,
            attention_indices,
        }
    }

    /// Get the layer type for a specific layer index.
    #[inline]
    pub fn layer_type(&self, layer_idx: usize) -> DeltaNetLayerType {
        self.layer_types[layer_idx]
    }

    /// Whether a specific layer is a DeltaNet layer.
    #[inline]
    pub fn is_deltanet(&self, layer_idx: usize) -> bool {
        self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet
    }

    /// Number of DeltaNet layers.
    #[inline]
    pub fn n_deltanet_layers(&self) -> usize {
        self.n_deltanet_layers
    }

    /// Number of attention layers.
    #[inline]
    pub fn n_attention_layers(&self) -> usize {
        self.n_attention_layers
    }

    /// Total layer count.
    #[inline]
    pub fn n_layer(&self) -> usize {
        self.layer_types.len()
    }

    /// Get the pre-computed DeltaNet layer indices.
    ///
    /// Returns a slice — no allocation per call.
    #[inline]
    pub fn deltanet_layer_indices(&self) -> &[usize] {
        &self.deltanet_indices
    }

    /// Get the pre-computed Attention layer indices.
    ///
    /// Returns a slice — no allocation per call.
    #[inline]
    pub fn attention_layer_indices(&self) -> &[usize] {
        &self.attention_indices
    }
}

// ---------------------------------------------------------------------------
// DeltaNet forward pass (CubeCL)
// ---------------------------------------------------------------------------

/// DeltaNet-specific GPU buffers for a single decode step.
///
/// Pre-allocated once, reused across all decode steps.
/// Per DeltaNet layer: conv1d output, recurrence output, gating output.
#[cfg(feature = "deltanet_inference")]
#[allow(dead_code)] // Fields used by future CubeCL dispatch optimization
pub struct DeltaNetActivationBuffers {
    /// Conv1d output: [3 * n_embd] (processed q/k/v)
    pub conv1d_out: cubecl::server::Handle,
    /// Recurrence output: [linear_n_heads * linear_head_dim] (state read result)
    pub recurrence_out: cubecl::server::Handle,
    /// Gating output: [mlp_hidden] (SwiGLU gate × up)
    pub gating_out: cubecl::server::Handle,
    /// MLP hidden scratch: [n_embd]
    pub mlp_hidden: cubecl::server::Handle,
    /// Per-layer conv1d state: [conv_dim * kernel_size] per layer
    pub conv_states: Vec<cubecl::server::Handle>,
}

#[cfg(feature = "deltanet_inference")]
impl DeltaNetActivationBuffers {
    /// Create pre-allocated activation buffers.
    pub fn new(client: &ComputeClient<ActiveRuntime>, config: &Config) -> Self {
        let n = config.n_embd;
        let mlp = config.mlp_hidden;
        let linear_dim = config.deltanet_linear_n_heads * config.deltanet_linear_head_dim;

        // Conv state dimensions
        let n_k_heads = config.deltanet_linear_n_heads;
        let n_v_heads = config.deltanet_linear_n_value_heads;
        let head_dim = config.deltanet_linear_head_dim;
        let kernel_size = config.deltanet_conv_kernel_size;
        let q_dim = n_k_heads * head_dim;
        let v_dim = n_v_heads * head_dim;
        let conv_dim = q_dim + q_dim + v_dim;

        // Per-layer conv1d state (only for DeltaNet layers)
        let layer_types = if config.layer_types.is_empty() {
            vec![riir_infer_core::types::DeltaNetLayerType::Attention; config.n_layer]
        } else {
            config.layer_types.clone()
        };
        let conv_state_size = conv_dim * kernel_size;
        let conv_zeros = vec![0.0f32; conv_state_size];
        let conv_states: Vec<cubecl::server::Handle> = (0..config.n_layer)
            .map(|i| {
                if layer_types[i] == riir_infer_core::types::DeltaNetLayerType::DeltaNet {
                    client.create_from_slice(f32::as_bytes(&conv_zeros))
                } else {
                    // Placeholder for attention layers — never used
                    client.empty(4)
                }
            })
            .collect();

        Self {
            conv1d_out: client.empty(3 * n * std::mem::size_of::<f32>()),
            recurrence_out: client.empty(linear_dim * std::mem::size_of::<f32>()),
            gating_out: client.empty(mlp * std::mem::size_of::<f32>()),
            mlp_hidden: client.empty(n * std::mem::size_of::<f32>()),
            conv_states,
        }
    }
}

// ---------------------------------------------------------------------------
// Combined forward buffers (CubeCL client + state + activations)
// ---------------------------------------------------------------------------

/// All GPU-resident buffers and context needed for DeltaNet layer dispatch.
///
/// Created once at model load time, reused across all decode steps.
/// Bundles the CubeCL compute client, recurrent state, activation buffers,
/// and conv1d state.
#[cfg(feature = "deltanet_inference")]
#[allow(dead_code)] // Fields used by dispatch_deltanet_layer and future optimization
pub struct DeltaNetForwardBuffers {
    /// CubeCL compute client for launching kernels.
    pub client: ComputeClient<ActiveRuntime>,
    /// Persistent recurrent state for all layers.
    pub state: DeltaNetStateBuffers,
    /// Pre-allocated activation buffers (conv1d_out, recurrence_out, etc.).
    pub activations: DeltaNetActivationBuffers,
    /// Hybrid layer dispatch (layer type routing).
    pub dispatch: HybridLayerDispatch,
}

#[cfg(feature = "deltanet_inference")]
impl DeltaNetForwardBuffers {
    /// Create all DeltaNet GPU buffers from config.
    pub fn new(config: &Config) -> Result<Self, crate::cubecl_runtime::CubeCLError> {
        let ctx = crate::cubecl_runtime::CubeCLContext::new()?;
        let client = ctx.client();

        let dispatch = HybridLayerDispatch::new(config);
        let state = DeltaNetStateBuffers::new(
            &client,
            config.n_layer,
            &if config.layer_types.is_empty() {
                vec![riir_infer_core::types::DeltaNetLayerType::Attention; config.n_layer]
            } else {
                config.layer_types.clone()
            },
            config.deltanet_linear_n_value_heads,
            config.deltanet_linear_head_dim * config.deltanet_linear_head_dim,
        );
        let activations = DeltaNetActivationBuffers::new(&client, config);

        Ok(Self {
            client,
            state,
            activations,
            dispatch,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "deltanet_inference"))]
mod tests {
    use super::*;
    use riir_infer_core::types::Config;

    /// Verify HybridLayerDispatch with all-Attention layers.
    #[test]
    fn test_dispatch_all_attention() {
        let config = Config::qwen_deltanet(8, vec![]);
        let dispatch = HybridLayerDispatch::new(&config);

        assert_eq!(dispatch.n_layer(), 8);
        assert_eq!(dispatch.n_attention_layers(), 8);
        assert_eq!(dispatch.n_deltanet_layers(), 0);
        assert_eq!(
            dispatch.attention_layer_indices(),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert!(dispatch.deltanet_layer_indices().is_empty());
    }

    /// Verify HybridLayerDispatch with hybrid layout.
    #[test]
    fn test_dispatch_hybrid() {
        use riir_infer_core::types::DeltaNetLayerType::*;
        let layer_types = vec![
            DeltaNet, DeltaNet, Attention, Attention, DeltaNet, Attention, Attention, Attention,
        ];
        let config = Config::qwen_deltanet(8, layer_types);
        let dispatch = HybridLayerDispatch::new(&config);

        assert_eq!(dispatch.n_layer(), 8);
        assert_eq!(dispatch.n_deltanet_layers(), 3);
        assert_eq!(dispatch.n_attention_layers(), 5);
        assert_eq!(dispatch.deltanet_layer_indices(), vec![0, 1, 4]);
        assert_eq!(dispatch.attention_layer_indices(), vec![2, 3, 5, 6, 7]);

        assert!(dispatch.is_deltanet(0));
        assert!(!dispatch.is_deltanet(2));
        assert!(dispatch.is_deltanet(4));
    }

    /// Verify empty layer_types defaults to all-Attention.
    #[test]
    fn test_dispatch_empty_defaults_attention() {
        let config = Config::qwen_deltanet(4, vec![]);
        let dispatch = HybridLayerDispatch::new(&config);

        for i in 0..4 {
            assert!(!dispatch.is_deltanet(i));
            assert_eq!(dispatch.layer_type(i), DeltaNetLayerType::Attention);
        }
    }
}
