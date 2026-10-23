//! `MLModel` load with compute units + the multi-function `functionName`
//! selector, the load-time compute-plan walk, and the zero-copy
//! `MLMultiArray` predict choreography — plan 619 T1.0 groups 3 + 4,
//! lifted verbatim from `riir-infer-laya`'s `ane.rs`. No objc2 type
//! crosses to a lane: the lanes speak [`ComputeUnits`], [`ArtifactPin`],
//! [`Placement`], [`LoadedModel`] and fp16 buffers only.
//!
//! [`load_model`] is the WHOLE load choreography in one call — digest
//! verify → content-addressed compile → `MLModelConfiguration` load →
//! compute-plan walk — so the lane keeps only its verdict (the gate
//! refusal + the drift disclosure name the lane's own manifest entry,
//! which is lane knowledge).

use std::path::Path;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputePlan, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLModelStructureProgram, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

use super::cache::{compile_cached, ns_error_string};
use super::digest::verify_dir_digest;
use super::{Error, Result};

/// The compute-unit posture, lane-facing — the host maps onto
/// `MLComputeUnits` internally (the conversion gate's `CPU_AND_NE` is
/// the ANE lanes' posture; `MLComputeUnits` is an NS_OPTIONS bit set
/// over an NSInteger).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnits {
    /// ANE-preferred, GPU excluded (`CPUAndNeuralEngine`).
    CpuAndNe,
    /// `CPUAndGPU`.
    CpuAndGpu,
    /// `CPUOnly`.
    CpuOnly,
}

impl ComputeUnits {
    fn to_native(self) -> objc2_core_ml::MLComputeUnits {
        match self {
            Self::CpuAndNe => objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
            Self::CpuAndGpu => objc2_core_ml::MLComputeUnits::CPUAndGPU,
            Self::CpuOnly => objc2_core_ml::MLComputeUnits::CPUOnly,
        }
    }
}

/// The artifact pin a load verifies against — the manifest's digest
/// walls verbatim. `files == 0 && bytes == 0` disables the shape walls
/// (digest-only pin, for lanes whose manifest doesn't pin counts); any
/// other pair enforces both walls exactly.
#[derive(Debug, Clone)]
pub struct ArtifactPin {
    /// blake3-dir-v1 hex digest of the `.mlpackage` bundle contents.
    pub digest: String,
    /// Pinned file count inside the bundle (0 = wall disabled).
    pub files: usize,
    /// Pinned byte total inside the bundle (0 = wall disabled).
    pub bytes: u64,
}

/// The load-time plan verdict (the conversion gate re-measured at
/// load). `Placement::SKIPPED` (all zeros) is the honest "not measured"
/// verdict a [`PlanWalk::Skip`] load carries — never read as a pass.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub device_ops: usize,
    pub ane_ops: usize,
    pub gpu_ops: usize,
    pub cpu_ops: usize,
    pub transitions: usize,
}

impl Placement {
    /// The gate: 100% of device-bearing ops ANE-preferred, zero device
    /// transitions. `false` refuses the load.
    pub fn passes(&self) -> bool {
        self.device_ops > 0 && self.ane_ops == self.device_ops && self.transitions == 0
    }

    /// The "walk skipped" verdict — all zeros; lanes that skip the walk
    /// disclose it and never read this as a placement pass.
    pub const SKIPPED: Self = Self {
        device_ops: 0,
        ane_ops: 0,
        gpu_ops: 0,
        cpu_ops: 0,
        transitions: 0,
    };
}

/// Whether the load walks the ML Program compute plan (the
/// no-silent-fallback gate). The walk analyzes the whole program —
/// seconds on the laya packages, minutes-or-worse on a GB-scale
/// multi-function package (measured: the clef decoder's walk never
/// completes inside 120 s on the M3) — so plan-619 loads skip it and
/// let the parity gate + the latency bench carry the placement truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanWalk {
    /// Walk + gate at load (the laya posture).
    Verify,
    /// Skip the walk; the returned placement is [`Placement::SKIPPED`]
    /// (the plan-619 posture — the artifact is GB-scale and
    /// multi-function; the walk is a load-time cost with no gate to
    /// enforce there).
    Skip,
}

/// One loaded + verified Core ML model: the compiled bundle lives in
/// the content-addressed cache (shared across loads); this handle owns
/// the in-memory `MLModel`. Single-threaded by contract (the lanes'
/// forwards are single-threaded — the capture's posture).
pub struct LoadedModel {
    model: Retained<MLModel>,
}

/// The WHOLE load choreography: verify the artifact digest against
/// `pin` (count / byte / digest walls), compile through the
/// content-addressed cache (`cache_env` + `default_cache_subdir` name
/// the lane's cache home), load under `units` (+ `function_name` for
/// multi-function packages — `None` keeps the default `main` function),
/// and — when `plan` is [`PlanWalk::Verify`] — re-walk the compute
/// plan. Any failure is a refused load — never a CPU fallback.
///
/// The eight parameters are the host's own design law — every
/// lane-specific fact is a parameter, never a host-side constant — so
/// the count is accepted here rather than bundled into a struct whose
/// fields would still be one per fact.
#[allow(clippy::too_many_arguments)]
#[allow(deprecated)] // the sync compile API inside compile_cached; the
// async twin needs a semaphore dance for a one-shot load-time call
pub fn load_model(
    artifact_dir: &Path,
    pin: &ArtifactPin,
    units: ComputeUnits,
    function_name: Option<&str>,
    cache_env: &str,
    default_cache_subdir: &str,
    missing_remedy: &str,
    plan: PlanWalk,
) -> Result<(LoadedModel, Placement)> {
    verify_dir_digest(
        artifact_dir,
        &pin.digest,
        pin.files,
        pin.bytes,
        missing_remedy,
    )?;
    let compiled = compile_cached(artifact_dir, &pin.digest, cache_env, default_cache_subdir)?;
    load_compiled_at(&compiled, units, function_name, plan)
}

/// Load an ALREADY-COMPILED `.mlmodelc` bundle directory in place —
/// the audio lane's posture (`.issues/015`): FluidInference publishes
/// pre-compiled bundles, [`MLModel::compileModelAtURL`] rejects a
/// compiled bundle as its INPUT (it compiles `.mlpackage`/`.mlmodel`
/// source), and there is nothing left to compile — so this path
/// verifies the digest walls and loads the artifact directly, no
/// compile cache (the cache exists to amortize compilation; a
/// pre-compiled bundle has none to amortize).
pub fn load_model_precompiled(
    artifact_dir: &Path,
    pin: &ArtifactPin,
    units: ComputeUnits,
    function_name: Option<&str>,
    missing_remedy: &str,
    plan: PlanWalk,
) -> Result<(LoadedModel, Placement)> {
    verify_dir_digest(
        artifact_dir,
        &pin.digest,
        pin.files,
        pin.bytes,
        missing_remedy,
    )?;
    load_compiled_at(artifact_dir, units, function_name, plan)
}

/// The shared load choreography over a COMPILED bundle path: `MLModel`
/// load under `units` (+ `function_name` for multi-function packages),
/// then the optional compute-plan walk. Both load publics end here.
fn load_compiled_at(
    compiled: &Path,
    units: ComputeUnits,
    function_name: Option<&str>,
    plan: PlanWalk,
) -> Result<(LoadedModel, Placement)> {
    unsafe {
        let url = NSURL::fileURLWithPath_isDirectory(
            &NSString::from_str(&compiled.to_string_lossy()),
            true,
        );
        let config = MLModelConfiguration::new();
        config.setComputeUnits(units.to_native());
        if let Some(fname) = function_name {
            config.setFunctionName(Some(&NSString::from_str(fname)));
        }
        let model =
            MLModel::modelWithContentsOfURL_configuration_error(&url, &config).map_err(|e| {
                Error::msg(format!(
                    "ane: Core ML load failed for {}: {}",
                    compiled.display(),
                    ns_error_string(&e)
                ))
            })?;

        let placement = match plan {
            PlanWalk::Verify => verify_compute_plan(compiled, &config)?,
            // The honest "not measured" verdict — the SKIPPED placement
            // never reads as a pass (device_ops == 0 fails `passes()`).
            PlanWalk::Skip => Placement::SKIPPED,
        };
        Ok((LoadedModel { model }, placement))
    }
}

impl LoadedModel {
    /// One forward over named fp16 inputs into the named fp16 output —
    /// the caller's buffers are borrowed zero-copy for the duration of
    /// the prediction (no-copy init, no-op deallocator; the arrays drop
    /// before the buffers). Fixed shapes — a buffer whose length ≠ its
    /// declared shape is a caller bug and asserts (the historical lane
    /// contract, unchanged).
    pub fn predict_f16(
        &self,
        inputs: &[(&str, &[u16], &[usize])],
        output_name: &str,
        out_f16: &mut [u16],
    ) -> Result<()> {
        unsafe {
            let mut arrays: Vec<Retained<MLMultiArray>> = Vec::with_capacity(inputs.len());
            let mut names: Vec<Retained<NSString>> = Vec::with_capacity(inputs.len());
            for (name, data, shape) in inputs {
                let expected: usize = shape.iter().product();
                assert_eq!(
                    data.len(),
                    expected,
                    "input {name:?} buffer must be {shape:?}"
                );
                arrays.push(multi_array_zero_copy(data, shape)?);
                names.push(NSString::from_str(name));
            }
            let arr = run_prediction(&self.model, &arrays, &names, output_name)?;
            copy_out_f16(&arr, out_f16)
        }
    }

    /// The f32 twin — the clef joint-schema head consumes f32 inputs
    /// (`states` + the packing arrays) and answers logits in the
    /// artifact's own output dtype (read dtype-agnostically: Float32
    /// copies directly, Float16 widens). Same zero-copy borrow law as
    /// [`predict_f16`]; the input arrays are Float32.
    pub fn predict_f32(
        &self,
        inputs: &[(&str, &[f32], &[usize])],
        output_name: &str,
        out_f32: &mut [f32],
    ) -> Result<()> {
        self.predict_f32_multi(inputs, &mut [(output_name, out_f32)])
    }

    /// The multi-output f32 forward — the audio lane's shape
    /// (`.issues/015`): one prediction, N named outputs copied out (the
    /// silero VAD answers prob + NEW recurrent state, and the state
    /// tensors must be carried — three separate forwards would re-roll
    /// the recurrence and be wrong, not just slow). Same zero-copy
    /// borrow law as the single-output twins: every input buffer
    /// borrows read-only, every output buffer borrows write-only for
    /// the duration; output element counts are asserted against the
    /// artifact at copy time (the `copy_out_f32` walls).
    pub fn predict_f32_multi(
        &self,
        inputs: &[(&str, &[f32], &[usize])],
        outputs: &mut [(&str, &mut [f32])],
    ) -> Result<()> {
        unsafe {
            let mut arrays: Vec<Retained<MLMultiArray>> = Vec::with_capacity(inputs.len());
            let mut names: Vec<Retained<NSString>> = Vec::with_capacity(inputs.len());
            for (name, data, shape) in inputs {
                let expected: usize = shape.iter().product();
                assert_eq!(
                    data.len(),
                    expected,
                    "input {name:?} buffer must be {shape:?}"
                );
                arrays.push(multi_array_zero_copy_f32(data, shape)?);
                names.push(NSString::from_str(name));
            }
            let out_names: Vec<&str> = outputs.iter().map(|(n, _)| *n).collect();
            let out_arrays = run_prediction_multi(&self.model, &arrays, &names, &out_names)?;
            for ((_, out_buf), arr) in outputs.iter_mut().zip(out_arrays) {
                copy_out_f32(&arr, out_buf)?;
            }
        }
        Ok(())
    }
}

/// The shared prediction choreography: name-keyed feature provider →
/// sync prediction → the named output's multi-array (no objc2 type
/// crosses to a lane). Single-output form — delegates to
/// [`run_prediction_multi`].
unsafe fn run_prediction(
    model: &Retained<MLModel>,
    arrays: &[Retained<MLMultiArray>],
    names: &[Retained<NSString>],
    output_name: &str,
) -> Result<Retained<MLMultiArray>> {
    // SAFETY: the multi twin performs the same ObjC message sends the
    // single form always did — the caller's unsafe context carries over.
    let mut outs = unsafe { run_prediction_multi(model, arrays, names, &[output_name])? };
    Ok(outs.pop().expect("one output requested, one returned"))
}

/// The multi-output twin (`predict_f32_multi`): ONE prediction, N named
/// outputs resolved in the caller's order — a second forward would
/// re-roll a stateful model's recurrence, so multi-output artifacts
/// must resolve every output from the single prediction.
unsafe fn run_prediction_multi(
    model: &Retained<MLModel>,
    arrays: &[Retained<MLMultiArray>],
    names: &[Retained<NSString>],
    output_names: &[&str],
) -> Result<Vec<Retained<MLMultiArray>>> {
    // SAFETY: plain ObjC message sends over live retained objects; every
    // borrow (arrays, names) outlives the prediction call.
    unsafe {
        let values: Vec<Retained<AnyObject>> = arrays
            .iter()
            .map(|arr| {
                let val = MLFeatureValue::featureValueWithMultiArray(arr);
                // The provider init erases the value type to AnyObject
                // (the ObjC dictionary erasure) — upcast once. `cast`
                // is the documented upcast (deprecated name, fine for
                // a root upcast; `cast_unchecked` would add unsafe for
                // nothing).
                #[allow(deprecated)]
                Retained::cast::<AnyObject>(val)
            })
            .collect();
        let name_refs: Vec<&NSString> = names.iter().map(|s| &**s).collect();
        let value_refs: Vec<Retained<AnyObject>> = values.iter().map(Retained::clone).collect();
        let dict: Retained<NSDictionary<NSString, AnyObject>> =
            NSDictionary::from_retained_objects(&name_refs, &value_refs);
        let features = MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
        .map_err(|e| {
            Error::msg(format!(
                "ane: feature provider failed: {}",
                ns_error_string(&e)
            ))
        })?;
        let out = model
            .predictionFromFeatures_error(ProtocolObject::from_ref(&*features))
            .map_err(|e| Error::msg(format!("ane: prediction failed: {}", ns_error_string(&e))))?;
        output_names
            .iter()
            .map(|output_name| {
                let name = NSString::from_str(output_name);
                let val = out.featureValueForName(&name).ok_or_else(|| {
                    Error::msg(format!(
                        "ane: output {output_name:?} missing from the prediction (manifest/ artifact drift?)"
                    ))
                })?;
                val.multiArrayValue().ok_or_else(|| {
                    Error::msg(format!("ane: output {output_name:?} is not a multi-array"))
                })
            })
            .collect()
    }
}

/// Build a zero-copy fp16 `MLMultiArray` over the caller's buffer (the
/// buffer must outlive the array — in `predict_f16` all buffers outlive
/// the prediction call, and the arrays drop first).
fn multi_array_zero_copy(data: &[u16], shape: &[usize]) -> Result<Retained<MLMultiArray>> {
    multi_array_zero_copy_typed(
        NonNull::from(data).cast::<std::ffi::c_void>(),
        shape,
        MLMultiArrayDataType::Float16,
    )
}

/// The f32 twin of [`multi_array_zero_copy`] (`predict_f32`).
fn multi_array_zero_copy_f32(data: &[f32], shape: &[usize]) -> Result<Retained<MLMultiArray>> {
    multi_array_zero_copy_typed(
        NonNull::from(data).cast::<std::ffi::c_void>(),
        shape,
        MLMultiArrayDataType::Float32,
    )
}

/// The shared no-copy init (no-op deallocator: ownership stays with the
/// caller's buffer).
fn multi_array_zero_copy_typed(
    ptr: NonNull<std::ffi::c_void>,
    shape: &[usize],
    dtype: MLMultiArrayDataType,
) -> Result<Retained<MLMultiArray>> {
    // No-op deallocator: ownership stays with the caller's buffer; the
    // block signature is Apple's `^(void *bytes) { … }`.
    let dealloc = RcBlock::new(|_bytes: NonNull<std::ffi::c_void>| {});
    let shape_arr = number_array(shape);
    let stride_arr = number_array(&contiguous_strides(shape));
    // SAFETY: the buffer covers `shape`'s element count and outlives the
    // returned array (predict drops the arrays before the buffers); the
    // deallocator is a no-op so Core ML never frees caller-owned memory.
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &shape_arr,
            dtype,
            &stride_arr,
            Some(&dealloc),
        )
    }
    .map_err(|e| {
        Error::msg(format!(
            "ane: input multi-array rejected: {}",
            ns_error_string(&e)
        ))
    })
}

fn number_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    let nums: Vec<Retained<NSNumber>> = values.iter().map(|v| NSNumber::new_usize(*v)).collect();
    NSArray::from_retained_slice(&nums)
}

/// C-contiguous row-major strides for `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Read the fp16 output into the caller's buffer. Contiguous arrays
/// take the fast `getBytesWithHandler` memcpy; a logically-strided
/// array (a transposed/reshaped view) is walked by its own strides —
/// a flat copy of a strided backing store would be silently wrong
/// bytes (the Swift reference converts with the same stride walk).
fn copy_out_f16(arr: &MLMultiArray, out: &mut [u16]) -> Result<()> {
    let expected = out.len();
    let got = unsafe { arr.count() } as usize;
    if got != expected {
        return Err(Error::msg(format!(
            "ane: output count {got} != expected {expected} — artifact shape drift"
        )));
    }
    if array_is_contiguous(arr) {
        let sink = out.as_mut_ptr();
        let handler = RcBlock::new(move |bytes: NonNull<std::ffi::c_void>, size: isize| {
            let src = bytes.as_ptr() as *const u16;
            let n = (size as usize) / 2;
            // SAFETY: `sink` points at `expected` elements, live across this
            // call; Core ML hands us the array's own buffer.
            unsafe { std::ptr::copy_nonoverlapping(src, sink, n.min(expected)) };
        });
        // SAFETY: the handler only reads the array's buffer into `out`.
        unsafe {
            arr.getBytesWithHandler(&handler);
        }
        return Ok(());
    }
    // Strided walk (rare; correctness over speed on this path).
    let (shape, strides) = shape_and_strides(arr);
    for (flat, index) in multi_indices(&shape).into_iter().enumerate() {
        let logical: usize = index.iter().zip(&strides).map(|(i, s)| i * s).sum();
        // SAFETY: in-bounds subscript over the array's own elements.
        out[flat] = unsafe { element_f16(arr, logical) };
    }
    Ok(())
}

/// The f32 output reader — dtype-agnostic (the head artifact's logits
/// dtype is whatever the export chose): Float32 copies directly,
/// Float16 widens (same stride law as [`copy_out_f16`]).
fn copy_out_f32(arr: &MLMultiArray, out: &mut [f32]) -> Result<()> {
    let expected = out.len();
    let got = unsafe { arr.count() } as usize;
    if got != expected {
        return Err(Error::msg(format!(
            "ane: output count {got} != expected {expected} — artifact shape drift"
        )));
    }
    let dtype = unsafe { arr.dataType() };
    let contiguous = array_is_contiguous(arr);
    if contiguous && dtype == MLMultiArrayDataType::Float32 {
        let sink = out.as_mut_ptr();
        let handler = RcBlock::new(move |bytes: NonNull<std::ffi::c_void>, size: isize| {
            let src = bytes.as_ptr() as *const f32;
            let n = (size as usize) / 4;
            // SAFETY: `sink` points at `expected` elements, live across this
            // call; Core ML hands us the array's own buffer.
            unsafe { std::ptr::copy_nonoverlapping(src, sink, n.min(expected)) };
        });
        // SAFETY: the handler only reads the array's buffer into `out`.
        unsafe {
            arr.getBytesWithHandler(&handler);
        }
        return Ok(());
    }
    let (shape, strides) = shape_and_strides(arr);
    for (flat, index) in multi_indices(&shape).into_iter().enumerate() {
        let logical: usize = index.iter().zip(&strides).map(|(i, s)| i * s).sum();
        out[flat] = match dtype {
            MLMultiArrayDataType::Float32 => unsafe { element_f32(arr, logical) },
            MLMultiArrayDataType::Float16 => f16_widen(unsafe { element_f16(arr, logical) }),
            other => {
                return Err(Error::msg(format!(
                    "ane: output dtype {other:?} unsupported — artifact drift"
                )));
            }
        };
    }
    Ok(())
}

/// Whether the array's strides are the C-contiguous row-major layout.
fn array_is_contiguous(arr: &MLMultiArray) -> bool {
    let (shape, strides) = shape_and_strides(arr);
    strides == contiguous_strides(&shape)
}

fn shape_and_strides(arr: &MLMultiArray) -> (Vec<usize>, Vec<usize>) {
    unsafe {
        let shape: Vec<usize> = arr.shape().to_vec().iter().map(|n| n.as_usize()).collect();
        let strides: Vec<usize> = arr
            .strides()
            .to_vec()
            .iter()
            .map(|n| n.as_usize())
            .collect();
        (shape, strides)
    }
}

/// All multi-dimensional indices over `shape`, row-major order.
fn multi_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let total: usize = shape.iter().product();
    let mut indices = Vec::with_capacity(total);
    let mut current = vec![0usize; shape.len()];
    for _ in 0..total {
        indices.push(current.clone());
        // increment the last dim, carry upwards
        for d in (0..shape.len()).rev() {
            current[d] += 1;
            if current[d] < shape[d] {
                break;
            }
            current[d] = 0;
        }
    }
    indices
}

unsafe fn element_f16(arr: &MLMultiArray, index: usize) -> u16 {
    let number = unsafe { arr.objectAtIndexedSubscript(index as isize) };
    number.as_u16()
}

unsafe fn element_f32(arr: &MLMultiArray, index: usize) -> f32 {
    let number = unsafe { arr.objectAtIndexedSubscript(index as isize) };
    number.as_f32()
}

/// IEEE-754 binary16 → f32 widening (bit math; the `half` crate stays
/// out of the host's dep tree).
fn f16_widen(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let bits32 = match exponent {
        0 => {
            if mantissa == 0 {
                sign // ±0
            } else {
                // subnormal: normalize
                let mut e: u32 = 127 - 15 + 1;
                let mut m = mantissa;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03ff;
                sign | ((e - 1) << 23) | (m << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13), // inf / nan
        e => sign | ((u32::from(e) + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits32)
}

/// The conversion gate, re-measured at load: walk the ML Program structure of
/// the COMPILED artifact, ask the plan for every operation's preferred
/// device, and count the device-bearing ops + device transitions in
/// program order (exactly the conversion tool's placement table).
pub(crate) fn verify_compute_plan(
    compiled: &Path,
    config: &MLModelConfiguration,
) -> Result<Placement> {
    let (tx, rx) = std::sync::mpsc::channel();
    unsafe {
        let url = NSURL::fileURLWithPath_isDirectory(
            &NSString::from_str(&compiled.to_string_lossy()),
            true,
        );
        let handler = RcBlock::new(move |plan: *mut MLComputePlan, err: *mut NSError| {
            let result = if err.is_null() && !plan.is_null() {
                // SAFETY: a non-null MLComputePlan just handed to the
                // block by Core ML (autoreleased — retained here).
                Ok(Retained::retain_autoreleased(plan.cast::<MLComputePlan>())
                    .expect("non-null plan"))
            } else {
                let msg = if err.is_null() {
                    "null plan".to_string()
                } else {
                    ns_error_string(&*err)
                };
                Err(msg)
            };
            let _ = tx.send(result);
        });
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, config, &handler);
    }
    let plan = rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .map_err(|_| Error::msg("ane: compute-plan load timed out"))?
        .map_err(Error::msg)?;

    let structure = unsafe { plan.modelStructure() };
    let program = unsafe { structure.program() }
        .ok_or_else(|| Error::msg("ane: model structure is not an ML Program"))?;
    // Multi-function packages carry one function per name (the clef
    // `L{bucket}` shape) and NO `main`; walk the function the config
    // selects, falling back to `main` (the single-function posture).
    let function_name = unsafe { config.functionName() }
        .map(|n| n.to_string())
        .unwrap_or_else(|| "main".to_string());
    let ops = program_operations(&program, &function_name)?;

    let mut placement = Placement {
        device_ops: 0,
        ane_ops: 0,
        gpu_ops: 0,
        cpu_ops: 0,
        transitions: 0,
    };
    let mut prev: Option<u8> = None; // 0 ane / 1 gpu / 2 cpu / 3 other
    for op in &ops {
        let usage = unsafe { plan.computeDeviceUsageForMLProgramOperation(op) };
        let Some(usage) = usage else {
            continue; // const / identity — materialized, no device (the P0 rule)
        };
        let device = unsafe { usage.preferredComputeDevice() };
        // The runtime class name is the device kind (the P0 smoke's
        // type-name matching — MLComputeDeviceProtocol carries no kind
        // tag); ProtocolObject reaches the runtime class through its
        // AnyObject view.
        let anyobj: &AnyObject = device.as_ref();
        let class_code = match anyobj.class().name().to_string_lossy().as_ref() {
            "MLNeuralEngineComputeDevice" => 0,
            "MLGPUComputeDevice" => 1,
            "MLCPUComputeDevice" => 2,
            _ => 3,
        };
        placement.device_ops += 1;
        match class_code {
            0 => placement.ane_ops += 1,
            1 => placement.gpu_ops += 1,
            2 => placement.cpu_ops += 1,
            _ => {}
        }
        if prev.is_some_and(|p| p != class_code) {
            placement.transitions += 1;
        }
        prev = Some(class_code);
    }
    Ok(placement)
}

/// `program.functions[name].block.operations` — the selected function's
/// op list in program order (multi-function packages name their
/// functions `L{bucket}`; single-function packages keep `main`).
fn program_operations(
    program: &MLModelStructureProgram,
    function_name: &str,
) -> Result<Vec<Retained<objc2_core_ml::MLModelStructureProgramOperation>>> {
    let main = NSString::from_str(function_name);
    let functions = unsafe { program.functions() };
    let func = functions
        .objectForKey(&main)
        .ok_or_else(|| Error::msg(format!("ane: program has no `{function_name}` function")))?;
    let block = unsafe { func.block() };
    let ops = unsafe { block.operations() };
    Ok(ops.to_vec())
}
