//! Issue 726 T3 — the ANE ObjC bridge port (from riir-poc's proven T0
//! substrate, verbatim flow; see `objc/ane_prefill_bridge.m`).
//!
//! Compile/load/eval private-ANE programs behind a C ABI dlopened from the
//! build-script-produced dylib (`libane_prefill_bridge.dylib` in `OUT_DIR`).
//! The pure-Rust msgSend path was a recorded negative result (byte-identical
//! artifacts, always rejected — see riir-poc `ane_t0.rs`); ObjC is the only
//! working substrate on this box, hence the build.rs bridge.
//!
//! Everything here is **macOS + aarch64 only** (Apple Silicon ANE). The
//! eligibility gate fail-opens on every other host — this module is never
//! compiled there.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::ffi::{CString, c_char, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

/// Issue 769 T10 P5 — the retry instrument.
///
/// `eval` / `eval_nc` retry ONCE on a first-attempt failure (Issue 726 T4: the
/// ANE's `ANEProgramProcessRequestDirect` fails intermittently under sustained
/// multi-eval bursts). The retry is correct and usually succeeds — which is the
/// problem: it **doubles** that op's ANE time (100-245 ms at real dims, Bench
/// 856) and `N_EVAL_FAILS` counts only the *both*-attempts-failed case, so a run
/// silently retrying a large fraction of its evals reports `fails 0` and an
/// inflated `eval` share indistinguishable from a genuinely slow ANE. Bench 856
/// observed 2 such lines in stderr text alone, countable by nothing.
///
/// Counts FIRST-ATTEMPT failures, i.e. retries ATTEMPTED — not retries that then
/// failed (that is `N_EVAL_FAILS`, which is a strict subset).
pub static N_EVAL_RETRIES: AtomicU64 = AtomicU64::new(0);
/// Total attempts (first + retry) across both entry points — the denominator
/// that makes [`N_EVAL_RETRIES`] a rate rather than a bare count.
pub static N_EVAL_ATTEMPTS: AtomicU64 = AtomicU64::new(0);

/// Zero both retry counters (call from the stage reset, at arm start).
pub fn eval_retry_reset() {
    N_EVAL_RETRIES.store(0, Ordering::Relaxed);
    N_EVAL_ATTEMPTS.store(0, Ordering::Relaxed);
}

// ── dlopen plumbing ───────────────────────────────────────────────────────

const RTLD_NOW: i32 = 0x2;
const RTLD_LOCAL: i32 = 0x4;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// Two-blob compile (Form C: `weight_data.bin` int8 + `weight_scale.bin`
/// fp16 — the omlx-verbatim `constexpr_blockwise_shift_scale` layout, P10's
/// winning form).
type Compile2Fn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    i32,
    i32,
    i32,
    i32,
) -> *mut c_void;
/// Blocking eval: fp16 bits in/out, per-procedure index (0 = single-func).
type EvalFn = unsafe extern "C" fn(*mut c_void, *const u16, *mut u16, usize, usize, i32) -> i32;
/// Plan 550: no-copy eval — IO staged directly into the handle's surfaces by
/// the caller (GPU kernels over the Metal texture views); no host memcpy.
type EvalNcFn = unsafe extern "C" fn(*mut c_void, i32) -> i32;
type FreeFn = unsafe extern "C" fn(*mut c_void);
/// Last NSError text from the shim (diagnostic; "" when no failure recorded).
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
/// Plan 550: wrap io_in/io_out as R32Uint MTLTextures on the shared device
/// (returns both RETAINED — the Rust side owns the +1). Only read by the
/// `metal_tensor_gemm`-gated kernel methods (Issue 886 footnote).
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
type MtlTexturesFn = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
);
/// Plan 550: the canonical texture geometry for a surface byte size.
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
type TexGeomFn = unsafe extern "C" fn(usize, *mut u32, *mut u32);
/// Plan 550: host⇄surface staging (test/compat — the same memcpys `eval`
/// does, exposed so `eval_nc` can be validated against `eval`).
type StageFn = unsafe extern "C" fn(*mut c_void, i32, *const u16, usize);
/// Probe 778: dump a private class's method table (selector hunt).
type DumpClassMethodsFn = unsafe extern "C" fn(*const c_char);
/// Probe 778: wrap io_out's own pages as a no-copy MTLBuffer (RETAINED).
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
type OutNoCopyBufferFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
/// Probe 779: runtime-driven private-API surface (see ane_prefill_bridge.m).
type ClassNamen = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type ReleaseFn = unsafe extern "C" fn(*mut c_void);
type RetainFn = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type DumpIvarsFn = unsafe extern "C" fn(*const c_char);
type DumpGraphFn = unsafe extern "C" fn(*mut c_void);
type ModelOfFn = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type RequestOfFn = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;
type OptionsOfFn = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type OutFenceFn = unsafe extern "C" fn(*mut c_void) -> i32;
type ObjIvarCountFn = unsafe extern "C" fn(*mut c_void) -> i32;
type ObjIvarNameAtFn = unsafe extern "C" fn(*mut c_void, i32, *mut c_char, i32) -> i32;
type GetIvarFn = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type ArrayCountFn = unsafe extern "C" fn(*mut c_void) -> usize;
type ArrayElemFn = unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void;
type InvokeFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const AneInvokeArg,
    i32,
    *mut AneInvokeRet,
) -> i32;

/// Probe 779: one invocation argument (layout-pinned against
/// `ane_invoke_arg_t` in the shim — kind + object pointer + scalar payload).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AneInvokeArg {
    pub kind: u8,
    pub obj: *mut c_void,
    pub u: u64,
}

impl AneInvokeArg {
    pub fn obj(o: *mut c_void) -> Self {
        Self { kind: b'o', obj: o, u: 0 }
    }
    pub fn u32(v: u32) -> Self {
        Self { kind: b'I', obj: std::ptr::null_mut(), u: v as u64 }
    }
    pub fn u64_kind(v: u64) -> Self {
        Self { kind: b'Q', obj: std::ptr::null_mut(), u: v }
    }
    pub fn boolean(v: bool) -> Self {
        Self { kind: b'c', obj: std::ptr::null_mut(), u: v as u64 }
    }
}

/// Probe 779: invocation result (layout-pinned against `ane_invoke_ret_t`).
#[repr(C)]
pub struct AneInvokeRet {
    pub ok: i32,
    pub ret_kind: u8,
    pub ret_obj: *mut c_void,
    pub ret_i: i64,
    pub ret_f: f64,
    pub err_text: [u8; 256],
}

impl AneInvokeRet {
    pub fn error(&self) -> String {
        let end = self.err_text.iter().position(|&b| b == 0).unwrap_or(256);
        String::from_utf8_lossy(&self.err_text[..end]).into_owned()
    }
}

/// A RETAINED object handed back across the bridge (dropped via
/// `ane_t0_release`).
pub struct RetainedObj(*mut c_void);
impl Clone for RetainedObj {
    fn clone(&self) -> Self {
        let p = unsafe { (bridge_fns().retain)(self.0) };
        assert!(!p.is_null(), "ane_t0_retain of a live object returned NULL");
        RetainedObj(p)
    }
}
impl RetainedObj {
    pub fn ptr(&self) -> *mut c_void {
        self.0
    }
    /// Borrow for an `ane_t0_invoke` call (does not transfer ownership).
    pub fn as_arg(&self) -> AneInvokeArg {
        AneInvokeArg::obj(self.0)
    }
    /// The runtime class name (static string — eternal, no retain).
    pub fn class_name(&self) -> String {
        probe_class_name(self.0)
    }
}
impl Drop for RetainedObj {
    fn drop(&mut self) {
        unsafe { (bridge_fns().release)(self.0) };
    }
}
unsafe impl Send for RetainedObj {}

struct BridgeFns {
    compile2: Compile2Fn,
    eval: EvalFn,
    eval_nc: EvalNcFn,
    free: FreeFn,
    last_error: LastErrorFn,
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    mtl_textures: MtlTexturesFn,
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    tex_geom: TexGeomFn,
    stage: StageFn,
    dump_class_methods: DumpClassMethodsFn,
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    out_no_copy_buffer: OutNoCopyBufferFn,
    class_name: ClassNamen,
    release: ReleaseFn,
    retain: RetainFn,
    dump_ivars: DumpIvarsFn,
    dump_graph: DumpGraphFn,
    model_of: ModelOfFn,
    request_of: RequestOfFn,
    options_of: OptionsOfFn,
    out_fence: OutFenceFn,
    obj_ivar_count: ObjIvarCountFn,
    obj_ivar_name_at: ObjIvarNameAtFn,
    get_ivar_obj: GetIvarFn,
    array_count: ArrayCountFn,
    array_elem: ArrayElemFn,
    invoke: InvokeFn,
}

/// dlsym a symbol from the bridge dylib (debug probes — the named-fn table
/// stays for the production entries).
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
fn dlsym_export(name: &str) -> *mut c_void {
    unsafe {
        let dylib = concat!(env!("OUT_DIR"), "/libane_prefill_bridge.dylib");
        let c = CString::new(dylib).expect("dylib path");
        let h = dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL);
        assert!(!h.is_null(), "dlopen {dylib} failed");
        let c = CString::new(name).unwrap();
        dlsym(h, c.as_ptr())
    }
}

fn bridge_fns() -> &'static BridgeFns {
    use std::sync::OnceLock;
    static FNS: OnceLock<BridgeFns> = OnceLock::new();
    FNS.get_or_init(|| unsafe {
        let dylib = concat!(env!("OUT_DIR"), "/libane_prefill_bridge.dylib");
        let c = CString::new(dylib).expect("dylib path");
        let h = dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL);
        assert!(!h.is_null(), "dlopen {dylib} failed");
        let sym = |name: &str| -> *mut c_void {
            let c = CString::new(name).unwrap();
            dlsym(h, c.as_ptr())
        };
        BridgeFns {
            // Symbol names keep the `ane_t0_` prefix — the .m is a verbatim
            // port and renaming exports would diverge from the proven
            // substrate for zero benefit. (The single-blob `ane_t0_compile`
            // symbol exists in the .m for the fp16 fallback form; dlsym it
            // back if P9's exact form is ever needed.)
            compile2: std::mem::transmute::<*mut c_void, Compile2Fn>(sym("ane_t0_compile2")),
            eval: std::mem::transmute::<*mut c_void, EvalFn>(sym("ane_t0_eval")),
            eval_nc: std::mem::transmute::<*mut c_void, EvalNcFn>(sym("ane_t0_eval_nc")),
            free: std::mem::transmute::<*mut c_void, FreeFn>(sym("ane_t0_free")),
            last_error: std::mem::transmute::<*mut c_void, LastErrorFn>(sym("ane_t0_last_error")),
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            mtl_textures: std::mem::transmute::<*mut c_void, MtlTexturesFn>(sym("ane_t0_mtl_textures")),
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            tex_geom: std::mem::transmute::<*mut c_void, TexGeomFn>(sym("ane_t0_tex_geom_for_bytes")),
            stage: std::mem::transmute::<*mut c_void, StageFn>(sym("ane_t0_stage")),
            dump_class_methods: std::mem::transmute::<*mut c_void, DumpClassMethodsFn>(sym(
                "ane_t0_dump_class_methods",
            )),
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            out_no_copy_buffer: std::mem::transmute::<*mut c_void, OutNoCopyBufferFn>(sym(
                "ane_t0_out_no_copy_buffer",
            )),
            class_name: std::mem::transmute::<*mut c_void, ClassNamen>(sym("ane_t0_class_name")),
            release: std::mem::transmute::<*mut c_void, ReleaseFn>(sym("ane_t0_release")),
            retain: std::mem::transmute::<*mut c_void, RetainFn>(sym("ane_t0_retain")),
            dump_ivars: std::mem::transmute::<*mut c_void, DumpIvarsFn>(sym("ane_t0_dump_ivars")),
            dump_graph: std::mem::transmute::<*mut c_void, DumpGraphFn>(sym("ane_t0_dump_graph")),
            model_of: std::mem::transmute::<*mut c_void, ModelOfFn>(sym("ane_t0_model_of")),
            request_of: std::mem::transmute::<*mut c_void, RequestOfFn>(sym("ane_t0_request_of")),
            options_of: std::mem::transmute::<*mut c_void, OptionsOfFn>(sym("ane_t0_options_of")),
            out_fence: std::mem::transmute::<*mut c_void, OutFenceFn>(sym("ane_t0_out_fence")),
            obj_ivar_count: std::mem::transmute::<*mut c_void, ObjIvarCountFn>(sym(
                "ane_t0_obj_ivar_count",
            )),
            obj_ivar_name_at: std::mem::transmute::<*mut c_void, ObjIvarNameAtFn>(sym(
                "ane_t0_obj_ivar_name_at",
            )),
            get_ivar_obj: std::mem::transmute::<*mut c_void, GetIvarFn>(sym(
                "ane_t0_get_ivar_obj",
            )),
            array_count: std::mem::transmute::<*mut c_void, ArrayCountFn>(sym(
                "ane_t0_array_count",
            )),
            array_elem: std::mem::transmute::<*mut c_void, ArrayElemFn>(sym("ane_t0_array_elem")),
            invoke: std::mem::transmute::<*mut c_void, InvokeFn>(sym("ane_t0_invoke")),
        }
    })
}

// ── Probe 779: runtime-driven private-API surface (free functions — the
//    receivers are raw retained pointers, not BridgeKernels). Retained-
//    pointer discipline: every fn returning an object returns +1 (wrap in
//    RetainedObj); arguments borrow.

fn probe_class_name(obj: *mut c_void) -> String {
    let p = unsafe { (bridge_fns().class_name)(obj) };
    if p.is_null() {
        return "(null)".into();
    }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
}

/// Probe 779: class ivar table dump (diagnostic, stderr, `--nocapture`).
pub fn probe_dump_ivars(class: &str) {
    let c = CString::new(class).expect("class name");
    unsafe { (bridge_fns().dump_ivars)(c.as_ptr()) };
}

/// # Safety
/// `obj` must be a valid Objective-C object pointer (nil-safe).
pub unsafe fn probe_obj_ivar_count(obj: *mut c_void) -> usize {
    unsafe { (bridge_fns().obj_ivar_count)(obj).max(0) as usize }
}

///
/// # Safety
/// `obj` must be a valid retained Objective-C object pointer.
pub unsafe fn probe_obj_ivar_name_at(obj: *mut c_void, idx: usize) -> Option<String> {
    let mut buf = [0u8; 128];
    let ok = unsafe {
        (bridge_fns().obj_ivar_name_at)(obj, idx as i32, buf.as_mut_ptr() as *mut c_char, 128)
    };
    if ok == 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(128);
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}

/// Read an object ivar (RETAINED — wrap in `RetainedObj`).
///
/// # Safety
/// `obj` must be a valid retained Objective-C object pointer.
pub unsafe fn probe_get_ivar_obj(obj: *mut c_void, name: &str) -> Option<RetainedObj> {
    let c = CString::new(name).expect("ivar name");
    let p = unsafe { (bridge_fns().get_ivar_obj)(obj, c.as_ptr()) };
    if p.is_null() {
        None
    } else {
        Some(RetainedObj(p))
    }
}

///
/// # Safety
/// `arr` must be a valid Objective-C object pointer (nil-safe).
pub unsafe fn probe_array_count(arr: *mut c_void) -> usize {
    unsafe { (bridge_fns().array_count)(arr) }
}

///
/// # Safety
/// `arr` must be a valid Objective-C object pointer (nil-safe).
pub unsafe fn probe_array_elem(arr: *mut c_void, idx: usize) -> Option<RetainedObj> {
    let p = unsafe { (bridge_fns().array_elem)(arr, idx) };
    if p.is_null() {
        None
    } else {
        Some(RetainedObj(p))
    }
}

/// NSInvocation-driven selector call (signatures from the runtime's own
/// tables — a wrong guess reports an encoding mismatch, never an ABI
/// gamble). `ret_obj`, when non-null, is RETAINED — the caller owns it.
///
/// # Safety
/// `receiver` must be a valid retained Objective-C object pointer.
pub unsafe fn probe_invoke(
    receiver: *mut c_void,
    selector: &str,
    args: &[AneInvokeArg],
) -> AneInvokeRet {
    let c = CString::new(selector).expect("selector");
    let mut ret = AneInvokeRet {
        ok: 0,
        ret_kind: b'v',
        ret_obj: std::ptr::null_mut(),
        ret_i: 0,
        ret_f: 0.0,
        err_text: [0; 256],
    };
    let ok = unsafe {
        (bridge_fns().invoke)(receiver, c.as_ptr(), args.as_ptr(), args.len() as i32, &mut ret)
    };
    ret.ok = ok;
    ret
}

// ── MIL templates ─────────────────────────────────────────────────────────

// The macOS-26.5-proven buildInfo (maderix ane_int8_bench — `appendString`
// does NOT printf-process `{{`, so the WORKING program text literally
// contains `{{ ... }}`. omlx's single-brace variant only proved out on
// macOS 15.7. Byte-match the proven form.) Braces are format!() ARGUMENTS,
// not format strings.
const MIL_BUILD_INFO_WORKING: &str = concat!(
    "{{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
    "{\"coremlc-version\", \"3505.4.1\"}, ",
    "{\"coremlc-component-milinternal\", \"\"}, ",
    "{\"coremltools-version\", \"9.0\"}}"
);

/// Form C int8 linear (ternary requant) as a 1×1 conv over `[1, ic, 1, w]`
/// (tokens along W) — the omlx-VERBATIM two-file layout
/// (`weight_data.bin` int8 + `weight_scale.bin` fp16 per-output-channel,
/// `constexpr_blockwise_shift_scale` dequant) in the macOS-26.5-proven
/// working syntax (spaced attrs + `{{ }}` buildInfo). P10/P11-verified:
/// full gate_up dims cosine 1.000019; 10.93 TFLOPS at 1 B/weight.
pub fn int8_two_file_mil(input_dim: usize, output_dim: usize, w: usize) -> String {
    format!(
        "program(1.3)\n[buildInfo = dict<string, string>({MIL_BUILD_INFO_WORKING})]\n{{\n\
         \x20   func main<ios18>(tensor<fp16, [1, {input_dim}, 1, {w}]> x) {{\n\
         \x20       string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n\
         \x20       tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n\
         \x20       tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
         \x20       tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n\
         \x20       int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n\
         \x20       tensor<int8, [{output_dim}, {input_dim}, 1, 1]> wd = const()[name = string(\"wd\"), val = tensor<int8, [{output_dim}, {input_dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight_data.bin\"), offset = uint64(64)))];\n\
         \x20       tensor<fp16, [{output_dim}, 1, 1, 1]> ws = const()[name = string(\"ws\"), val = tensor<fp16, [{output_dim}, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight_scale.bin\"), offset = uint64(64)))];\n\
         \x20       tensor<fp16, [{output_dim}, {input_dim}, 1, 1]> wq = constexpr_blockwise_shift_scale(data = wd, scale = ws)[name = string(\"dequant\")];\n\
         \x20       tensor<fp16, [1, {output_dim}, 1, {w}]> c0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = wq, x = x)[name = string(\"c0\")];\n\
         \x20   }} -> (c0);\n}}\n",
    )
}

// ── ANE weight blobs (omlx `make_blob` / rane `pack_weights`) ─────────────

/// ANE weight blob: 128-byte header + payload. `element_bits`: 0x10 fp16,
/// 0x08 int8 — the maderix chunk[10] element bit-width field required by
/// the compiler's blob validation on current macOS.
pub fn pack_weights_bits(payload: &[u8], element_bits: u8) -> Vec<u8> {
    let total = 128 + payload.len();
    let mut blob = vec![0u8; total];
    blob[0] = 0x01;
    blob[4] = 0x02;
    blob[64] = 0xEF;
    blob[65] = 0xBE;
    blob[66] = 0xAD;
    blob[67] = 0xDE;
    blob[68] = 0x01; // type
    blob[74] = element_bits; // chunk[10] — element bit-width
    blob[72..80].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    blob[80..88].copy_from_slice(&128u64.to_le_bytes()); // absolute data offset
    blob[128..].copy_from_slice(payload);
    blob
}

// ── the kernel handle ─────────────────────────────────────────────────────

/// Plan 550: the Metal texture views of one [`BridgeKernel`]'s io surfaces,
/// created ONCE per kernel on the shared device (see
/// [`BridgeKernel::metal_textures`]). Geometry fields are the shim's
/// canonical R32Uint dims — the pack/unpack kernel indexing uses THESE,
/// never its own arithmetic.
///
/// SAFETY (the manual Send/Sync): the wrapped `id<MTLTexture>` objects are
/// thread-safe per Metal's API contract (immutable resources usable from
/// any thread); the textures alias the kernel handle's IOSurfaces, whose
/// lifetime is the handle's (the parent `BridgeKernel` outlives every job —
/// it lives in the process-lifetime bank).
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
pub struct ZcTextures {
    pub tex_in: metal::Texture,
    pub tex_out: metal::Texture,
    pub w_in: u64,
    pub h_in: u64,
    pub w_out: u64,
    pub h_out: u64,
}

#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
unsafe impl Send for ZcTextures {}
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
unsafe impl Sync for ZcTextures {}

/// A compiled + loaded ANE program (single procedure). Eval is blocking;
/// the handle is immutable after construction (Send + Sync — P4's
/// concurrency justification).
pub struct BridgeKernel {
    handle: *mut c_void,
    /// fp16 element counts of the input/output surfaces.
    pub in_elems: usize,
    pub out_elems: usize,
    /// Plan 550: lazily-created Metal texture views of the io surfaces (the
    /// zero-copy IO path). `Mutex` keeps the handle Sync; the inner `Arc` is
    /// cloned per dispatch so in-flight jobs keep their own reference.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    textures: std::sync::Mutex<Option<std::sync::Arc<crate::ane_prefill::bridge::ZcTextures>>>,
}

impl BridgeKernel {
    /// The compiled program's output CHANNEL count — `out_elems / seq`.
    /// Issue 887: the dispatch seams pass this as the executors' `oc_total`
    /// so the output-surface sizing is derived from the BANK's compiled
    /// width (the single source of truth) instead of re-deriving the fused
    /// width from config — the two silently diverge the moment a
    /// registration policy slices the program (split-overlap prefix
    /// registration), and `eval`'s own surface assert is a panic, not a
    /// fail-open. `seq` is the block width the bank compiled at.
    pub fn output_channels(&self, seq: usize) -> usize {
        debug_assert!(seq > 0 && self.out_elems.is_multiple_of(seq));
        self.out_elems / seq
    }

    /// Compile + load a Form C int8 program: the two-blob
    /// (`weight_data.bin` int8 + `weight_scale.bin` fp16) layout.
    pub fn compile_form_c(
        input_dim: usize,
        output_dim: usize,
        seq: usize,
        int8_payload: &[i8],
        fp16_scale_bits: &[u16],
    ) -> Result<Self, String> {
        let mil = int8_two_file_mil(input_dim, output_dim, seq);
        let data_blob = pack_weights_bits(
            unsafe { std::slice::from_raw_parts(int8_payload.as_ptr().cast(), int8_payload.len()) },
            0x08,
        );
        let scale_bytes: Vec<u8> = fp16_scale_bits
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        let scale_blob = pack_weights_bits(&scale_bytes, 0x10);
        let in_elems = input_dim * seq;
        let out_elems = output_dim * seq;
        Self::compile_two_blobs(&mil, &data_blob, &scale_blob, in_elems, out_elems)
    }

    /// Compile + load a MIL with two blob files through the ObjC bridge
    /// (temp-file handoff; the bridge maps `@model_path/weights/*` onto the
    /// blobs in memory).
    fn compile_two_blobs(
        mil: &str,
        data_blob: &[u8],
        scale_blob: &[u8],
        in_elems: usize,
        out_elems: usize,
    ) -> Result<Self, String> {
        let fns = bridge_fns();
        // Unique per CALL, not per process: a pid-only tag makes every
        // concurrent compile in this process share three temp paths, so one
        // caller's `remove_file` deletes another caller's blobs mid-compile
        // ("compile2/load failed") — found by the Issue-887-T2 bracket probe
        // running 12 compiles under libtest's default multi-threading, which
        // widened the two-test window the suite had been lucky inside. An
        // atomic suffix is strictly safer on the production path too (two
        // bank constructions racing in one process).
        static COMPILE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let tag = format!(
            "ane_pf_c2_{}_{}",
            std::process::id(),
            COMPILE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let mil_path = std::env::temp_dir().join(format!("{tag}.mil"));
        let data_path = std::env::temp_dir().join(format!("{tag}_data.bin"));
        let scale_path = std::env::temp_dir().join(format!("{tag}_scale.bin"));
        std::fs::write(&mil_path, mil).map_err(|e| e.to_string())?;
        std::fs::write(&data_path, data_blob).map_err(|e| e.to_string())?;
        std::fs::write(&scale_path, scale_blob).map_err(|e| e.to_string())?;
        let m = CString::new(mil_path.to_str().unwrap()).unwrap();
        let d = CString::new(data_path.to_str().unwrap()).unwrap();
        let s = CString::new(scale_path.to_str().unwrap()).unwrap();
        let handle = unsafe {
            (fns.compile2)(
                m.as_ptr(),
                d.as_ptr(),
                s.as_ptr(),
                in_elems as i32,
                out_elems as i32,
                0, // hint 0 = unpinned (T0 P4: pinning is an M3-Ultra concern)
                1, // single-procedure programs (T0 P6: banks rejected)
            )
        };
        let _ = std::fs::remove_file(&mil_path);
        let _ = std::fs::remove_file(&data_path);
        let _ = std::fs::remove_file(&scale_path);
        if handle.is_null() {
            return Err("ANE bridge compile2/load failed".into());
        }
        Ok(Self {
            handle,
            in_elems,
            out_elems,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            textures: std::sync::Mutex::new(None),
        })
    }

    /// Blocking evaluate (fp16 bits in/out; single procedure).
    ///
    /// Issue 726 T4: the underlying `evaluateWithQoS` call fails
    /// intermittently under sustained multi-eval load (96-eval bursts; warmup
    /// completes, a later prefill fails) — retry ONCE before failing. The
    /// input IOSurface still holds this call's bytes, so the retry re-submits
    /// the identical request; a genuine per-program defect fails both tries.
    pub fn eval(&self, input: &[u16], output: &mut [u16]) -> Result<(), String> {
        assert_eq!(input.len(), self.in_elems, "input surface mismatch");
        assert_eq!(output.len(), self.out_elems, "output surface mismatch");
        let fns = bridge_fns();
        let mut ok = unsafe {
            (fns.eval)(
                self.handle,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
                output.len(),
                0,
            )
        };
        N_EVAL_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        if ok != 1 {
            N_EVAL_RETRIES.fetch_add(1, Ordering::Relaxed);
            N_EVAL_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
            ok = unsafe {
                (fns.eval)(
                    self.handle,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    input.len(),
                    output.len(),
                    0,
                )
            };
        }
        if ok != 1 {
            return Err(format!("ANE bridge eval failed: {}", self.last_error()));
        }
        Ok(())
    }

    /// Plan 550 (test/compat): stage host bytes into io_in (`dir == 0`;
    /// `buf` is read) or read a surface back to host (`dir == 1` = io_out,
    /// `dir == 2` = io_in — the C side memcpy's INTO `buf`, which must
    /// therefore be mut despite the C `const` qualifier).
    pub fn stage(&self, dir: i32, buf: &mut [u16]) {
        let fns = bridge_fns();
        unsafe { (fns.stage)(self.handle, dir, buf.as_ptr(), buf.len()) };
    }

    /// Plan 550 (debug probe): host WRITE io_out — the CPU-write→GPU-read
    /// mapping check for the tex_out view.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    #[doc(hidden)]
    pub fn debug_stage_out_write(&self, buf: &[u16]) {
        type OutWriteFn = unsafe extern "C" fn(*mut c_void, *const u16, usize);
        let sym: OutWriteFn = unsafe {
            std::mem::transmute::<*mut std::ffi::c_void, OutWriteFn>(dlsym_export(
                "ane_t0_stage_out_write",
            ))
        };
        unsafe { sym(self.handle, buf.as_ptr(), buf.len()) };
    }

    /// The underlying NSError text from the shim's last failure ("" if none
    /// was recorded — e.g. the failure predates the buffer, or the ObjC call
    /// returned NO with a nil error).
    pub fn last_error(&self) -> String {
        let fns = bridge_fns();
        unsafe {
            let p = (fns.last_error)();
            if p.is_null() {
                return String::new();
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }

    /// Plan 550: no-copy eval — the input bytes are ALREADY in the handle's
    /// io_in surface (staged by the zero-copy GPU pack kernel) and io_out is
    /// consumed the same way. Same retry-once contract as [`Self::eval`].
    pub fn eval_nc(&self) -> Result<(), String> {
        let fns = bridge_fns();
        let mut ok = unsafe { (fns.eval_nc)(self.handle, 0) };
        N_EVAL_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        if ok != 1 {
            N_EVAL_RETRIES.fetch_add(1, Ordering::Relaxed);
            N_EVAL_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
            ok = unsafe { (fns.eval_nc)(self.handle, 0) };
        }
        if ok != 1 {
            return Err(format!("ANE bridge eval_nc failed: {}", self.last_error()));
        }
        Ok(())
    }

    /// Plan 550: the Metal texture views of the io surfaces on the SHARED
    /// device (the Issue 663-extracted `id<MTLDevice>` CubeCL uses).
    /// Created once per kernel; cloned per dispatch.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    pub fn metal_textures(
        &self,
        device: &metal::DeviceRef,
    ) -> Result<std::sync::Arc<crate::ane_prefill::bridge::ZcTextures>, String> {
        use metal::foreign_types::{ForeignType, ForeignTypeRef};
        let mut guard = self.textures.lock().expect("ane texture cache poisoned");
        if let Some(t) = guard.as_ref() {
            return Ok(t.clone());
        }
        let fns = bridge_fns();
        let mut t_in: *mut c_void = std::ptr::null_mut();
        let mut t_out: *mut c_void = std::ptr::null_mut();
        unsafe {
            (fns.mtl_textures)(
                self.handle,
                device.as_ptr() as *mut c_void,
                &mut t_in,
                &mut t_out,
            );
        }
        if t_in.is_null() || t_out.is_null() {
            return Err(format!(
                "ANE io-surface texture wrap failed: {}",
                self.last_error()
            ));
        }
        // The shim returned both RETAINED (+1) — wrap with from_ptr (no extra
        // retain); the metal::Texture drop releases.
        let (tex_in, tex_out) = unsafe {
            (
                metal::Texture::from_ptr(t_in as *mut metal::MTLTexture),
                metal::Texture::from_ptr(t_out as *mut metal::MTLTexture),
            )
        };
        let mut w_in = 0u32;
        let mut h_in = 0u32;
        let mut w_out = 0u32;
        let mut h_out = 0u32;
        // Geometry from the shim's own canonical fn — never re-derive here.
        unsafe {
            (fns.tex_geom)(self.in_elems * 2, &mut w_in, &mut h_in);
            (fns.tex_geom)(self.out_elems * 2, &mut w_out, &mut h_out);
        }
        let textures = std::sync::Arc::new(crate::ane_prefill::bridge::ZcTextures {
            tex_in,
            tex_out,
            w_in: w_in as u64,
            h_in: h_in as u64,
            w_out: w_out as u64,
            h_out: h_out as u64,
        });
        *guard = Some(textures.clone());
        Ok(textures)
    }

    /// Probe 778: create FRESH texture views, bypassing the cache — the
    /// texture-recreation probe. If the ANE driver swaps the surface's
    /// backing store at eval completion, a texture created BEFORE the eval
    /// reads the stale backing while a fresh one sees the new bytes.
    /// Mirrors [`Self::metal_textures`] (both returned RETAINED; nothing
    /// cached); probe-only, never on a production path.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    #[doc(hidden)]
    pub fn metal_textures_fresh(
        &self,
        device: &metal::DeviceRef,
    ) -> Result<std::sync::Arc<crate::ane_prefill::bridge::ZcTextures>, String> {
        use metal::foreign_types::{ForeignType, ForeignTypeRef};
        let fns = bridge_fns();
        let mut t_in: *mut c_void = std::ptr::null_mut();
        let mut t_out: *mut c_void = std::ptr::null_mut();
        unsafe {
            (fns.mtl_textures)(
                self.handle,
                device.as_ptr() as *mut c_void,
                &mut t_in,
                &mut t_out,
            );
        }
        if t_in.is_null() || t_out.is_null() {
            return Err(format!(
                "ANE fresh texture wrap failed: {}",
                self.last_error()
            ));
        }
        let (tex_in, tex_out) = unsafe {
            (
                metal::Texture::from_ptr(t_in as *mut metal::MTLTexture),
                metal::Texture::from_ptr(t_out as *mut metal::MTLTexture),
            )
        };
        let mut w_in = 0u32;
        let mut h_in = 0u32;
        let mut w_out = 0u32;
        let mut h_out = 0u32;
        unsafe {
            (fns.tex_geom)(self.in_elems * 2, &mut w_in, &mut h_in);
            (fns.tex_geom)(self.out_elems * 2, &mut w_out, &mut h_out);
        }
        Ok(std::sync::Arc::new(crate::ane_prefill::bridge::ZcTextures {
            tex_in,
            tex_out,
            w_in: w_in as u64,
            h_in: h_in as u64,
            w_out: w_out as u64,
            h_out: h_out as u64,
        }))
    }

    /// Probe 778: dump a private ANE class's method table (class + instance
    /// selectors with type encodings) — the daemon-bypass selector hunt for
    /// Issue 769 T2 (`doEvaluateDirectWithModel:`) and any flush/sync
    /// primitive that could repair the ANE→GPU coherency break.
    #[doc(hidden)]
    pub fn dump_class_methods(class_name: &str) {
        let fns = bridge_fns();
        let c = std::ffi::CString::new(class_name).expect("class name");
        unsafe { (fns.dump_class_methods)(c.as_ptr()) };
    }

    /// Probe 778: the io_out surface's own pages as a no-copy MTLBuffer —
    /// the BUFFER cacheability domain. Returns an owned `metal::Buffer`
    /// (the shim returned it RETAINED; drop releases). The buffer aliases
    /// the surface: valid while `self` lives.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    #[doc(hidden)]
    pub fn out_no_copy_buffer(
        &self,
        device: &metal::DeviceRef,
    ) -> Result<metal::Buffer, String> {
        use metal::foreign_types::{ForeignType, ForeignTypeRef};
        let fns = bridge_fns();
        let p = unsafe { (fns.out_no_copy_buffer)(self.handle, device.as_ptr() as *mut c_void) };
        if p.is_null() {
            return Err(format!("no-copy buffer wrap failed: {}", self.last_error()));
        }
        Ok(unsafe { metal::Buffer::from_ptr(p as *mut metal::MTLBuffer) })
    }

    /// Probe 779: the raw shim handle (navigation entry for the private-API
    /// probes; the model/requests hang off it).
    #[doc(hidden)]
    pub fn raw_handle(&self) -> *mut c_void {
        self.handle
    }

    /// Probe 779: the live `_ANEInMemoryModel` (RETAINED).
    #[doc(hidden)]
    pub fn probe_model(&self) -> Option<RetainedObj> {
        let p = unsafe { (bridge_fns().model_of)(self.handle) };
        if p.is_null() {
            None
        } else {
            Some(RetainedObj(p))
        }
    }

    /// Probe 779: the per-procedure `_ANERequest` (RETAINED).
    #[doc(hidden)]
    pub fn probe_request(&self, idx: i32) -> Option<RetainedObj> {
        let p = unsafe { (bridge_fns().request_of)(self.handle, idx) };
        if p.is_null() {
            None
        } else {
            Some(RetainedObj(p))
        }
    }

    /// Probe 779: the eval options dict (pinning hints; the direct-eval
    /// selector takes it) — RETAINED.
    #[doc(hidden)]
    pub fn probe_options(&self) -> Option<RetainedObj> {
        let p = unsafe { (bridge_fns().options_of)(self.handle) };
        if p.is_null() {
            None
        } else {
            Some(RetainedObj(p))
        }
    }

    /// Probe 779: the io_out IOSurface lock/unlock pair with no memcpy —
    /// the writeback fence for a subsequent GPU read.
    #[doc(hidden)]
    pub fn probe_out_fence(&self) -> bool {
        unsafe { (bridge_fns().out_fence)(self.handle) == 1 }
    }

    /// Probe 779: instance-graph dump from the handle (stderr; shows where
    /// the private `_ANEClient` hangs off the live model/requests).
    #[doc(hidden)]
    pub fn probe_dump_graph(&self) {
        unsafe { (bridge_fns().dump_graph)(self.handle) };
    }
}

impl Drop for BridgeKernel {
    fn drop(&mut self) {
        let fns = bridge_fns();
        unsafe { (fns.free)(self.handle) };
    }
}

// SAFETY: handle is opaque; the underlying ObjC objects are immutable after
// construction; concurrent evaluate is exactly what T0 P4 measured.
unsafe impl Sync for BridgeKernel {}
unsafe impl Send for BridgeKernel {}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // Bonsai in_proj_concat real dims (issue §"T2/T3 design contract").
    const IC: usize = 5120;
    const OC: usize = 16480;
    const SEQ: usize = 2048;

    fn to_f16_bits(f: f32) -> u16 {
        half::f16::from_f32(f).to_bits()
    }

    fn from_f16_bits(h: u16) -> f32 {
        half::f16::from_bits(h).to_f32()
    }

    /// Real-dims Form C round trip: requant synthetic scaled-ternary weights
    /// (P11's measured scale regime), compile through the ObjC bridge, eval
    /// one 2048-token block with a sparse-channel input, and check the
    /// cosine vs the CPU reference on the ORIGINAL weights. The projection-
    /// level bar is the G1 stretch (0.9999); P11 measured 1.000011.
    #[test]
    fn form_c_real_dims_cosine_clears_stretch_bar() {
        let rows = OC;
        let cols = IC;
        let blocks64 = cols.div_ceil(64);
        let groups = cols.div_ceil(super::super::requant::TERNARY_GROUP_SIZE);

        // Deterministic scaled-ternary weights (xorshift; P11's scale range).
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut pos = vec![0u64; rows * blocks64];
        let mut neg = vec![0u64; rows * blocks64];
        let mut scales = vec![half::f16::from_f32(0.0); rows * groups];
        for r in 0..rows {
            for g in 0..groups {
                let t = (next() % 1000) as f32 / 1000.0;
                scales[r * groups + g] = half::f16::from_f32(0.002 + t * 0.059);
            }
            for c in 0..cols {
                match next() % 3 {
                    0 => pos[r * blocks64 + c / 64] |= 1 << (c % 64),
                    1 => neg[r * blocks64 + c / 64] |= 1 << (c % 64),
                    _ => {}
                }
            }
        }

        // Sparse-channel input: 4 nonzero channels make the CPU reference
        // cheap while still exercising weight-row indexing, channel stride,
        // and the full spatial width (the P9 trick).
        let nz_channels = [7usize, 1337, 2559, 5119];
        let mut x16 = vec![0u16; IC * SEQ];
        for &ch in &nz_channels {
            for t in 0..SEQ {
                let v = match t % 4 {
                    0 => 0.5,
                    1 => -1.25,
                    2 => 0.75,
                    _ => -0.5,
                } + ((t % 97) as f32) * 0.001;
                x16[ch * SEQ + t] = to_f16_bits(v);
            }
        }

        // Requant + compile + eval.
        let (q, ws) = super::super::requant::requant_per_row_int8(&pos, &neg, &scales, rows, cols);
        let kernel = BridgeKernel::compile_form_c(IC, OC, SEQ, &q, &ws)
            .expect("ANE Form C compile at real dims");
        let mut y16 = vec![0u16; OC * SEQ];
        kernel.eval(&x16, &mut y16).expect("ANE eval");

        // CPU reference on the ORIGINAL weights + cosine over the full
        // output (channel-major [OC × SEQ]).
        let mut dot = 0.0f64;
        let mut nn_a = 0.0f64;
        let mut nn_b = 0.0f64;
        for o in 0..OC {
            // w[o, ch] for the 4 nonzero channels (original weights).
            let mut wref = [0.0f32; 4];
            for (k, &ch) in nz_channels.iter().enumerate() {
                let bit = 1u64 << (ch % 64);
                let word = ch / 64;
                let s = scales[o * groups + ch / 128].to_f32();
                wref[k] = if pos[o * blocks64 + word] & bit != 0 {
                    s
                } else if neg[o * blocks64 + word] & bit != 0 {
                    -s
                } else {
                    0.0
                };
            }
            for t in 0..SEQ {
                let mut ref_v = 0.0f32;
                for (k, &ch) in nz_channels.iter().enumerate() {
                    ref_v += wref[k] * from_f16_bits(x16[ch * SEQ + t]);
                }
                let got = from_f16_bits(y16[o * SEQ + t]);
                dot += ref_v as f64 * got as f64;
                nn_a += (ref_v * ref_v) as f64;
                nn_b += (got * got) as f64;
            }
        }
        let cosine = (dot / (nn_a.sqrt() * nn_b.sqrt())) as f32;
        // P11 measured 1.000011 (fp16 accumulation can nudge slightly above
        // 1.0 vs the f32 reference — cosine here is a similarity check, not
        // a bounded [0,1] probability).
        assert!(
            cosine > 0.9999,
            "projection cosine {cosine} below the G1 stretch bar"
        );
    }

    /// Plan 550 G1-unit: `eval_nc` (no internal memcpys) over explicitly
    /// staged surfaces produces BIT-IDENTICAL output to the regular `eval`
    /// (memcpy in + eval + memcpy out) — proving (a) the Metal-friendly 2D
    /// surface geometry carries the same bytes, (b) the no-copy entry drives
    /// the same request. Synthetic dims keep it fast; the real-dims geometry
    /// is covered by `form_c_real_dims_cosine_clears_stretch_bar`.
    #[test]
    fn eval_nc_matches_eval_bit_identical() {
        let ic = 256usize;
        let oc = 192usize;
        let seq = 128usize;
        let rows = oc;
        let cols = ic;
        let blocks64 = cols.div_ceil(64);
        let groups = cols.div_ceil(super::super::requant::TERNARY_GROUP_SIZE);

        let mut seed = 0x0123_4567_89AB_CDEFu64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut pos = vec![0u64; rows * blocks64];
        let mut neg = vec![0u64; rows * blocks64];
        let mut scales = vec![half::f16::from_f32(0.0); rows * groups];
        for r in 0..rows {
            for g in 0..groups {
                let t = (next() % 1000) as f32 / 1000.0;
                scales[r * groups + g] = half::f16::from_f32(0.002 + t * 0.059);
            }
            for c in 0..cols {
                match next() % 3 {
                    0 => pos[r * blocks64 + c / 64] |= 1 << (c % 64),
                    1 => neg[r * blocks64 + c / 64] |= 1 << (c % 64),
                    _ => {}
                }
            }
        }
        let (q, ws) = super::super::requant::requant_per_row_int8(&pos, &neg, &scales, rows, cols);
        let kernel =
            BridgeKernel::compile_form_c(ic, oc, seq, &q, &ws).expect("ANE Form C compile (synthetic)");

        // Dense-ish input (every channel nonzero at a small seq).
        let mut x16 = vec![0u16; ic * seq];
        for slot in x16.iter_mut() {
            let v = (((next() % 2001) as f32) - 1000.0) / 1000.0;
            *slot = to_f16_bits(v);
        }

        // Reference: the regular host-memcpy eval.
        let mut y_ref = vec![0u16; oc * seq];
        kernel.eval(&x16, &mut y_ref).expect("ANE eval (reference)");

        // Zero-copy path: stage in, eval_nc, read out — same surface bytes,
        // no eval-internal copies.
        kernel.stage(0, &mut x16);
        kernel.eval_nc().expect("ANE eval_nc");
        let mut y_nc = vec![0u16; oc * seq];
        kernel.stage(1, &mut y_nc);

        assert_eq!(y_ref, y_nc, "eval_nc output bits diverged from eval");
    }

    /// Issue 887 T2 — the CHANNEL-GRID bracket probe the issue sized at
    /// "~2-6 `compile_form_c` calls": pins the adopted oMLX 64-row grid
    /// (`ANE_PREFILL_CHANNEL_GRID`) on THIS compiler, in both directions
    /// (aligned must work; misaligned is measured and pinned).
    ///
    /// Why both directions are load-bearing:
    /// - **ALIGNED** widths (grid multiples) must compile AND eval — the
    ///   landed 887-T1(a) split-mode registration policy only ever creates
    ///   on-grid prefix banks, so an on-grid width failing here would
    ///   invalidate an already-landed mechanism, not merely reject a
    ///   hypothetical. Compile alone is not enough (the P7 spatial lesson,
    ///   channel axis: narrow W compiled but eval-broke), so the probe evals.
    /// - **MISALIGNED** neighbors are MEASURED and the observed verdict is
    ///   pinned below: compile-refusal would mean oMLX's floor premise (the
    ///   grid is a compile cliff); compile-OK-but-eval-fail would be the P7
    ///   shape (the floor lives at eval); both-OK means the grid is a
    ///   CONVENTION on this compiler, not a hardware constraint — which is
    ///   what measured (see the pinned note below), and what the constant's
    ///   doc now cites as ground truth.
    ///
    /// Two geometries, so the pin carries no synthetic-only asterisk: the
    /// `eval_nc` template (ic 256, seq 128) AND the real in_proj shape
    /// (ic 5120, seq 2048 — the dims a split-mode prefix registration
    /// actually compiles at). Twelve tiny compile+eval round trips total:
    /// no model, no bank, no GPU work; headless lib-test lane, same class
    /// as `form_c_real_dims_cosine_clears_stretch_bar`.
    #[test]
    fn channel_grid_bracket_pins_the_compiler_verdict() {
        let grid = super::super::ANE_PREFILL_CHANNEL_GRID;
        let aligned: &[usize] = &[64, 128, 192];
        let misaligned: &[usize] = &[48, 96, 100];

        let fixture = |oc: usize, cols: usize| -> (Vec<i8>, Vec<u16>) {
            let rows = oc;
            let blocks64 = cols.div_ceil(64);
            let groups = cols.div_ceil(super::super::requant::TERNARY_GROUP_SIZE);
            let mut seed = 0x8870_0000_0000_0001u64;
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let mut pos = vec![0u64; rows * blocks64];
            let mut neg = vec![0u64; rows * blocks64];
            let mut scales = vec![half::f16::from_f32(0.0); rows * groups];
            for r in 0..rows {
                for g in 0..groups {
                    let t = (next() % 1000) as f32 / 1000.0;
                    scales[r * groups + g] = half::f16::from_f32(0.002 + t * 0.059);
                }
                for c in 0..cols {
                    match next() % 3 {
                        0 => pos[r * blocks64 + c / 64] |= 1 << (c % 64),
                        1 => neg[r * blocks64 + c / 64] |= 1 << (c % 64),
                        _ => {}
                    }
                }
            }
            super::super::requant::requant_per_row_int8(&pos, &neg, &scales, rows, cols)
        };

        // Returns (compiled, evaluated). `eval` is the bridge's own
        // retry-once entry, so a transient status=0x9 does not read as a
        // verdict here.
        let probe = |oc: usize, ic: usize, seq: usize| -> (bool, bool) {
            let (q, ws) = fixture(oc, ic);
            match BridgeKernel::compile_form_c(ic, oc, seq, &q, &ws) {
                Err(e) => {
                    eprintln!("  [bracket] ic={ic} seq={seq} oc={oc:4}: compile REFUSED ({e})");
                    (false, false)
                }
                Ok(kernel) => {
                    let x16 = vec![half::f16::from_f32(0.25).to_bits(); ic * seq];
                    let mut y16 = vec![0u16; oc * seq];
                    let eval_ok = kernel.eval(&x16, &mut y16).is_ok();
                    eprintln!(
                        "  [bracket] ic={ic} seq={seq} oc={oc:4}: compile ok, eval {}",
                        if eval_ok { "ok" } else { "REFUSED" }
                    );
                    (true, eval_ok)
                }
            }
        };

        eprintln!("═══ Issue 887 T2 channel-grid bracket (grid = {grid} rows) ═══");
        for (ic, seq) in [(256usize, 128usize), (5120, 2048)] {
            eprintln!("── geometry ic={ic} seq={seq} ──");
            for &oc in aligned {
                assert_eq!(oc % grid, 0, "bracket misconfigured: {oc} not on grid");
                let (compiled, evaluated) = probe(oc, ic, seq);
                assert!(
                    compiled,
                    "ALIGNED width {oc} failed to COMPILE (ic={ic} seq={seq}) — \
                     the landed split-mode registration policy only creates \
                     on-grid prefixes; on-grid is not usable on this compiler"
                );
                assert!(
                    evaluated,
                    "ALIGNED width {oc} compiled but failed at EVAL (ic={ic} \
                     seq={seq}) — the P7 compile≠works shape on the channel \
                     axis; on-grid prefixes compile but do not run"
                );
            }
            // MISALIGNED — measured verdicts, pinned per the 2026-09-21 run:
            // every off-grid width COMPILED and EVALUATED at BOTH geometries
            // (synthetic and real in_proj dims). The grid is therefore NOT a
            // compile or eval cliff on this compiler — it is oMLX's alignment
            // convention, adopted for oMLX-faithfulness plus the conservative
            // off-grid ⇒ full-registration policy, NOT a hardware floor
            // (contrast ANE_PREFILL_MIN_SPATIAL_W, which IS one). See
            // ANE_PREFILL_CHANNEL_GRID's doc for the consequence.
            for &oc in misaligned {
                let (compiled, evaluated) = probe(oc, ic, seq);
                assert!(
                    !compiled || evaluated,
                    "off-grid width {oc} (ic={ic} seq={seq}): \
                     compiled={compiled} evaluated={evaluated} — a \
                     compile-OK/eval-FAIL width is the P7 shape and re-opens \
                     the channel-floor question"
                );
            }
        }
    }

    /// Compile-time sanity of the MIL template (no ANE needed): dims land
    /// in the right slots.
    #[test]
    fn mil_template_dims() {
        let mil = int8_two_file_mil(5120, 16480, 2048);
        assert!(mil.contains("tensor<fp16, [1, 5120, 1, 2048]> x"));
        assert!(mil.contains("tensor<fp16, [1, 16480, 1, 2048]> c0"));
        assert!(mil.contains("tensor<int8, [16480, 5120, 1, 1]> wd"));
        assert!(mil.contains("constexpr_blockwise_shift_scale"));
        assert!(mil.contains("weight_data.bin"));
        assert!(mil.contains("weight_scale.bin"));
        // The proven `{{ }}` buildInfo must survive literally.
        assert!(mil.contains("{{\"coremlc-component-MIL\""));
    }
}
