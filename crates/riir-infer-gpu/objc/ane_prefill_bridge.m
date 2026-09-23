// ane_prefill_bridge.m — ObjC FFI substrate for the Issue 726 ANE hybrid
// prefill (riir-gpu). Ported VERBATIM from riir-poc's ane_t0_bridge.m
// (the T0 spike substrate, proven on this exact box: 18.45 TFLOPS fp16,
// macOS 26.5.2, h15 ANE). The pure-Rust bridge was rejected by ANECCompile
// with byte-identical artifacts (see riir-poc ane_t0.rs header for the
// negative result) — ObjC is the working path. C ABI surface:
//   * ane_t0_compile / ane_t0_compile2 (two-file Form C) / ane_t0_eval
//     / ane_t0_free / ane_t0_last_error
//   * kANEFAneInstanceHint pinning options at compile+load+eval (omlx #2756)
//   * procedure-index requests for bank probing (omlx banking)
//   * persistent compile cache behind ANE_T0_CACHE=1 (omlx #2975 port —
//     Issue 769 T2; identical to riir-poc's ane_t0_bridge.m cache section)
//
// Build (see riir-gpu build.rs): xcrun clang -O2 -fobjc-arc -dynamiclib
//   -framework Foundation -framework IOSurface -ldl
//
// License note: the compile/eval flow is pattern-identical to maderix/ANE
// ane_int8_bench.m (repo docs at mintlify.wiki/maderix/ANE) — internal POC
// use, not for redistribution.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <stdint.h>
#import <string.h>
#import <stdlib.h>
#import <fcntl.h>
#import <sys/file.h>

// objc_retain/objc_release exist in libobjc but are ARC-hidden from
// objc/runtime.h in this toolchain — declare the prototypes directly
// (we manage retain counts manually across the C-ABI malloc'd struct).
extern id objc_retain(id);
extern void objc_release(id);

// ── handle state ──────────────────────────────────────────────────────────

typedef struct {
    id model;
    id options;
    id requests;         // NSArray of per-procedure requests
    int n_procs;
    IOSurfaceRef io_in;
    IOSurfaceRef io_out;
    // Plan 550: Metal-texture geometry of the io surfaces (R32Uint texels:
    // 4 bytes/texel, 2 fp16 each). Stored at creation so the texture wrap
    // and the Rust-side kernel math share one source of truth.
    uint32_t tex_w_in, tex_h_in, tex_w_out, tex_h_out;
    char tmp_dir[1024];
} AneT0Kernel;

// Plan 550 — Metal-friendly surface geometry. The ANE maps the tensor
// linearly onto the surface's byte space (geometry-agnostic — the original
// 1D form used Width=bytes/BPE=1 with no relation to the MIL tensor shape),
// and the host memcpy path is geometry-agnostic too, so ONE geometry serves
// both IO paths. R32Uint constraint: 4 bytes/texel, width capped at 16384
// (the Apple-GPU texture width limit).
#define ANE_TEX_MAX_W 16384u
static void tex_geom(NSUInteger bytes, uint32_t *w_px, uint32_t *rows) {
    NSUInteger u32s = bytes / 4;
    NSUInteger w = u32s < ANE_TEX_MAX_W ? (u32s ? u32s : 1) : ANE_TEX_MAX_W;
    NSUInteger r = (u32s + w - 1) / w;
    *w_px = (uint32_t)w;
    *rows = (uint32_t)(r ? r : 1);
}

static IOSurfaceRef make_io_surface(NSUInteger bytes) {
    uint32_t w, h;
    tex_geom(bytes, &w, &h);
    NSUInteger rowBytes = (NSUInteger)w * 4;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(w), (id)kIOSurfaceHeight: @(h),
        (id)kIOSurfaceBytesPerElement: @4, (id)kIOSurfaceBytesPerRow: @(rowBytes),
        (id)kIOSurfaceAllocSize: @(rowBytes * h), (id)kIOSurfacePixelFormat: @0});
}

static NSDictionary *make_options(int hint) {
    if (hint <= 0) return @{};
    // omlx instance pinning: kANEFProcedureVariantHint=1 + kANEFAneInstanceHint
    return @{
        @"kANEFProcedureVariantHint" : @1,
        @"kANEFAneInstanceHint" : @(hint),
    };
}

// ── persistent compile cache (omlx #2975 port — Issue 769 T2) ─────────────
//
// Verbatim from riir-poc's ane_t0_bridge.m (the proven cache section, landed
// 2026-09-21): the runtime's write half of the persistent store survives
// process exit, but nothing ever consults it, so every fresh process
// recompiles — the bank pays the full compile tax at every model load (the
// 226.3 s / 96-program line). This adds the read half, mirroring omlx
// `load_or_compile_ane_model` (qwen35_ane.mm):
//   [model compiledModelExists] probe → skip compile on a hit
//   [model purgeCompiledModel]        → purge + recompile exactly once on a
//                                       bad restore (a corrupt/incompatible
//                                       entry would otherwise fail forever)
//   a cross-process flock on
//     <caches>/riir/ane/v1/<os_build>/<identifier>.lock
//                                     held from probe through load, so two
//                                     processes compiling the same descriptor
//                                     serialize instead of racing the store
//                                     write
// Fail-open EVERYWHERE: an unusable cache root, lock, or selector degrades to
// the historical temp-only path — the cache only accelerates; it must never
// fail a load. The staging directory STAYS the historical temp path: the
// framework derives the model URL from the descriptor, and overriding it
// breaks the newer-OS per-file bundle hash verification (omlx #3124). Never
// unlink the lock file while running: a waiter may already hold the old inode
// (omlx). Env: ANE_T0_CACHE=1 (exactly "1") enables; every other value keeps
// the historical behavior byte-for-byte.

static bool ane_cache_enabled(void) {
    const char *v = getenv("ANE_T0_CACHE");
    return v && strcmp(v, "1") == 0;
}

// The os-build segment keeps same-program entries from colliding across OS
// bumps (a compiled artifact is not portable across ANE compilers).
static NSString *ane_cache_lock_entry(NSString *identifier) {
    NSString *os_build =
        [[NSProcessInfo processInfo] operatingSystemVersionString];
    os_build = [os_build stringByReplacingOccurrencesOfString:@"/"
                                                   withString:@"_"];
    os_build = [os_build stringByReplacingOccurrencesOfString:@":"
                                                   withString:@"_"];
    NSArray *roots = NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory, NSUserDomainMask, YES);
    NSString *root = roots.firstObject ?: NSTemporaryDirectory();
    NSString *entry = [[[[root stringByAppendingPathComponent:@"riir"]
        stringByAppendingPathComponent:@"ane"]
        stringByAppendingPathComponent:@"v1"]
        stringByAppendingPathComponent:os_build];
    [[NSFileManager defaultManager] createDirectoryAtPath:entry
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    return [entry stringByAppendingPathComponent:identifier];
}

// Bounded non-blocking acquisition (omlx: 30 s deadline, 50 ms retry) — a
// suspended lock holder must not hang later loads forever. Returns the held
// fd, or -1 (fail-open to the historical path).
static int ane_cache_lock_acquire(NSString *entry) {
    NSString *lock_path = [entry stringByAppendingString:@".lock"];
    int fd = open(lock_path.fileSystemRepresentation, O_CREAT | O_RDWR, 0600);
    if (fd < 0) return -1;
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:30.0];
    for (;;) {
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) return fd;
        if (errno != EWOULDBLOCK && errno != EAGAIN) break;
        if ([deadline timeIntervalSinceNow] <= 0) break;
        [NSThread sleepForTimeInterval:0.05];
    }
    close(fd);
    return -1;
}

// Acquire the descriptor lock and probe the persistent store. Returns true
// when a compiled artifact already exists (caller skips compile); *lock_fd
// receives the held lock (caller releases after load) or -1 (no lock, and
// the probe result is false — historical path).
static bool ane_cache_probe(id mdl, NSString *identifier, int *lock_fd) {
    *lock_fd = -1;
    if (!ane_cache_enabled()) return false;
    NSString *entry = ane_cache_lock_entry(identifier);
    if (!entry) return false;
    *lock_fd = ane_cache_lock_acquire(entry);
    if (*lock_fd < 0) {
        fprintf(stderr, "[bridge] cache lock unavailable — historical path\n");
        return false;
    }
    if (![mdl respondsToSelector:@selector(compiledModelExists)]) {
        fprintf(stderr,
                "[bridge] cache probe selector unavailable on this OS — historical path\n");
        return false;
    }
    bool restored =
        ((BOOL (*)(id, SEL))objc_msgSend)(mdl, @selector(compiledModelExists));
    // fprintf has no %@ — format the identifier through UTF8String (the
    // object itself is used via proper APIs above; this line is diagnostic).
    fprintf(stderr, "[bridge] cache %s identifier=%s\n",
            restored ? "HIT" : "MISS",
            [identifier respondsToSelector:@selector(UTF8String)]
                ? identifier.UTF8String : "?");
    return restored;
}

static void ane_cache_lock_release(int lock_fd) {
    if (lock_fd < 0) return;
    flock(lock_fd, LOCK_UN);
    close(lock_fd);
}

// ── C ABI ─────────────────────────────────────────────────────────────────

// Returns 0 on failure. mil_path/blob_path are UTF-8 file paths; the blob is
// mapped at "@model_path/weights/weight.bin". hint 0 = no pinning.
// in_elems/out_elems are fp16 ELEMENT counts for surface sizing.
// Last compile/eval error text (static ring of 1 — diagnostic only; the
// Issue 726 P12 design contract is single-threaded serial submission, so
// no locking. Surfaced so the T4 intermittent-eval-failure investigation
// gets the underlying NSError instead of an opaque bool.)
static char g_ane_last_error[512] = "";
static void set_last_error(NSString *s) {
    if (!s || !s.length) { g_ane_last_error[0] = '\0'; return; }
    strncpy(g_ane_last_error, s.UTF8String, sizeof(g_ane_last_error) - 1);
    g_ane_last_error[sizeof(g_ane_last_error) - 1] = '\0';
}

// n_procs: number of procedures in the program (>=1); a request is built per
// procedure index via inputSymbolIndicesForProcedureIndex:.
__attribute__((visibility("default")))
void *ane_t0_compile(const char *mil_path, const char *blob_path,
                     int in_elems, int out_elems, int hint, int n_procs) {
    @autoreleasepool {
        static bool loaded = false;
        static bool attempted = false;
        if (!attempted) {
            attempted = true;
            loaded = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                            "AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL) != NULL;
        }
        if (!loaded) return NULL;

        NSError *e = nil;
        NSData *milData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:mil_path]];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blob_path]];
        if (!milData || !blob) return NULL;

        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!D || !I || !AR || !AIO) return NULL;

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,
            @selector(modelWithMILText:weights:optionsPlist:), milData,
            @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}}, nil);
        if (!desc) { fprintf(stderr, "[bridge] desc FAIL\n"); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "[bridge] model FAIL\n"); return NULL; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSDictionary *opts = make_options(hint);

        int cache_fd = -1;
        bool restored = ane_cache_probe(mdl, hx, &cache_fd);
        BOOL ok = YES;
        if (!restored) {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 0, opts, &e);
            if (!ok) {
                fprintf(stderr, "[bridge] compile FAIL: %s\n", e ? e.description.UTF8String : "?");
            }
        }
        if (ok && !((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 0, opts, &e)) {
            ok = NO;
            if (restored) {
                // Corrupt or incompatible store entry: purge it before
                // compiling exactly once more, otherwise the probe would keep
                // returning the same unusable artifact (omlx).
                fprintf(stderr, "[bridge] cache restore failed — purge + recompile\n");
                if ([mdl respondsToSelector:@selector(purgeCompiledModel)]) {
                    ((void (*)(id, SEL))objc_msgSend)(mdl, @selector(purgeCompiledModel));
                }
                restored = false;
                if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(compileWithQoS:options:error:), 0, opts, &e)
                    && ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(loadWithQoS:options:error:), 0, opts, &e)) {
                    ok = YES;
                }
            }
            if (!ok) {
                fprintf(stderr, "[bridge] load FAIL: %s\n", e ? e.description.UTF8String : "?");
            }
        }
        ane_cache_lock_release(cache_fd);
        if (!ok) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        NSUInteger inBytes = (NSUInteger)in_elems * 2;
        NSUInteger outBytes = (NSUInteger)out_elems * 2;
        IOSurfaceRef ioI = make_io_surface(inBytes);
        IOSurfaceRef ioO = make_io_surface(outBytes);
        if (!ioI || !ioO) return NULL;
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioI);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioO);
        if (!wI || !wO) return NULL;

        id inner = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        NSMutableArray *reqs = [NSMutableArray arrayWithCapacity:(NSUInteger)n_procs];
        for (int p = 0; p < n_procs; p++) {
            id inSet = ((id(*)(id,SEL,unsigned int))objc_msgSend)(inner,
                @selector(inputSymbolIndicesForProcedureIndex:), (unsigned int)p);
            id outSet = ((id(*)(id,SEL,unsigned int))objc_msgSend)(inner,
                @selector(outputSymbolIndicesForProcedureIndex:), (unsigned int)p);
            if ([inSet count] != 1 || [outSet count] != 1) {
                fprintf(stderr, "[bridge] proc %d symbol sets: in=%lu out=%lu\n", p,
                        (unsigned long)[inSet count], (unsigned long)[outSet count]);
                return NULL;
            }
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@([inSet firstIndex])], @[wO], @[@([outSet firstIndex])], nil, nil, @(p));
            if (!req) {
                fprintf(stderr, "[bridge] request FAIL proc %d\n", p);
                return NULL;
            }
            [reqs addObject:req];
        }

        AneT0Kernel *k = (AneT0Kernel *)malloc(sizeof(AneT0Kernel));
        objc_retain(mdl);
        objc_retain(opts);
        objc_retain(reqs);
        k->model = mdl;
        k->options = opts;
        k->requests = reqs;
        k->n_procs = n_procs;
        k->io_in = ioI;
        k->io_out = ioO;
        k->tex_w_in = k->tex_h_in = k->tex_w_out = k->tex_h_out = 0;
        tex_geom(inBytes, &k->tex_w_in, &k->tex_h_in);
        tex_geom(outBytes, &k->tex_w_out, &k->tex_h_out);
        strncpy(k->tmp_dir, td.UTF8String, sizeof(k->tmp_dir) - 1);
        k->tmp_dir[sizeof(k->tmp_dir) - 1] = '\0';
        return k;
    }
}

// Two-blob variant (omlx-verbatim descriptor): maps `weight_data.bin` and
// `weight_scale.bin` as SEPARATE files — omlx's proven two-file int8
// layout (their qwen35_ane.mm maps per-path blobs). Same otherwise.
__attribute__((visibility("default")))
void *ane_t0_compile2(const char *mil_path, const char *data_path,
                      const char *scale_path,
                      int in_elems, int out_elems, int hint, int n_procs) {
    @autoreleasepool {
        static bool loaded2 = false;
        static bool attempted2 = false;
        if (!attempted2) {
            attempted2 = true;
            loaded2 = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                            "AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL) != NULL;
        }
        if (!loaded2) return NULL;

        NSError *e = nil;
        NSData *milData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:mil_path]];
        NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:data_path]];
        NSData *scale = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:scale_path]];
        if (!milData || !data || !scale) return NULL;

        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!D || !I || !AR || !AIO) return NULL;

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,
            @selector(modelWithMILText:weights:optionsPlist:), milData,
            @{@"@model_path/weights/weight_data.bin": @{@"offset": @0, @"data": data},
              @"@model_path/weights/weight_scale.bin": @{@"offset": @0, @"data": scale}}, nil);
        if (!desc) { fprintf(stderr, "[bridge] desc2 FAIL\n"); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "[bridge] model2 FAIL\n"); return NULL; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [data writeToFile:[td stringByAppendingPathComponent:@"weights/weight_data.bin"] atomically:YES];
        [scale writeToFile:[td stringByAppendingPathComponent:@"weights/weight_scale.bin"] atomically:YES];

        NSDictionary *opts = make_options(hint);

        int cache_fd = -1;
        bool restored = ane_cache_probe(mdl, hx, &cache_fd);
        BOOL ok = YES;
        if (!restored) {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 0, opts, &e);
            if (!ok) {
                fprintf(stderr, "[bridge] compile2 FAIL: %s\n", e ? e.description.UTF8String : "?");
            }
        }
        if (ok && !((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 0, opts, &e)) {
            ok = NO;
            if (restored) {
                // Corrupt or incompatible store entry: purge it before
                // compiling exactly once more, otherwise the probe would keep
                // returning the same unusable artifact (omlx).
                fprintf(stderr, "[bridge] cache restore failed — purge + recompile\n");
                if ([mdl respondsToSelector:@selector(purgeCompiledModel)]) {
                    ((void (*)(id, SEL))objc_msgSend)(mdl, @selector(purgeCompiledModel));
                }
                restored = false;
                if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(compileWithQoS:options:error:), 0, opts, &e)
                    && ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(loadWithQoS:options:error:), 0, opts, &e)) {
                    ok = YES;
                }
            }
            if (!ok) {
                fprintf(stderr, "[bridge] load2 FAIL: %s\n", e ? e.description.UTF8String : "?");
            }
        }
        ane_cache_lock_release(cache_fd);
        if (!ok) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        NSUInteger inBytes = (NSUInteger)in_elems * 2;
        NSUInteger outBytes = (NSUInteger)out_elems * 2;
        IOSurfaceRef ioI = make_io_surface(inBytes);
        IOSurfaceRef ioO = make_io_surface(outBytes);
        if (!ioI || !ioO) return NULL;
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioI);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioO);
        if (!wI || !wO) return NULL;

        id inner = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        NSMutableArray *reqs = [NSMutableArray arrayWithCapacity:(NSUInteger)n_procs];
        for (int p = 0; p < n_procs; p++) {
            id inSet = ((id(*)(id,SEL,unsigned int))objc_msgSend)(inner,
                @selector(inputSymbolIndicesForProcedureIndex:), (unsigned int)p);
            id outSet = ((id(*)(id,SEL,unsigned int))objc_msgSend)(inner,
                @selector(outputSymbolIndicesForProcedureIndex:), (unsigned int)p);
            if ([inSet count] != 1 || [outSet count] != 1) {
                fprintf(stderr, "[bridge] proc %d symbol sets: in=%lu out=%lu\n", p,
                        (unsigned long)[inSet count], (unsigned long)[outSet count]);
                return NULL;
            }
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@([inSet firstIndex])], @[wO], @[@([outSet firstIndex])], nil, nil, @(p));
            if (!req) {
                fprintf(stderr, "[bridge] request2 FAIL proc %d\n", p);
                return NULL;
            }
            [reqs addObject:req];
        }

        AneT0Kernel *k = (AneT0Kernel *)malloc(sizeof(AneT0Kernel));
        objc_retain(mdl);
        objc_retain(opts);
        objc_retain(reqs);
        k->model = mdl;
        k->options = opts;
        k->requests = reqs;
        k->n_procs = n_procs;
        k->io_in = ioI;
        k->io_out = ioO;
        k->tex_w_in = k->tex_h_in = k->tex_w_out = k->tex_h_out = 0;
        tex_geom(inBytes, &k->tex_w_in, &k->tex_h_in);
        tex_geom(outBytes, &k->tex_w_out, &k->tex_h_out);
        strncpy(k->tmp_dir, td.UTF8String, sizeof(k->tmp_dir) - 1);
        k->tmp_dir[sizeof(k->tmp_dir) - 1] = '\0';
        return k;
    }
}

// Test/compat staging helper for the zero-copy path: copy host bytes INTO
// io_in (dir 0), read io_out back to host (dir 1), or read io_in back (dir 2
// — the pack-kernel isolation probe). Same memcpys `ane_t0_eval` does,
// exposed separately so `eval_nc` can be validated against `eval` with
// identical surface contents (the Plan 550 G1-unit).
__attribute__((visibility("default")))
void ane_t0_stage(void *handle, int dir, const uint16_t *buf, size_t elems) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k || !buf) return;
    @autoreleasepool {
        if (dir == 0) {
            IOSurfaceLock(k->io_in, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(k->io_in), buf, elems * 2);
            IOSurfaceUnlock(k->io_in, 0, NULL);
        } else if (dir == 2) {
            IOSurfaceLock(k->io_in, 1, NULL);
            memcpy((void *)buf, IOSurfaceGetBaseAddress(k->io_in), elems * 2);
            IOSurfaceUnlock(k->io_in, 1, NULL);
        } else {
            IOSurfaceLock(k->io_out, 1, NULL);
            memcpy((void *)buf, IOSurfaceGetBaseAddress(k->io_out), elems * 2);
            IOSurfaceUnlock(k->io_out, 1, NULL);
        }
    }
}

// DEBUG probe: host WRITE io_out — the CPU-write→GPU-read mapping check for
// the tex_out view (the isolation ladder).
__attribute__((visibility("default")))
void ane_t0_stage_out_write(void *handle, const uint16_t *buf, size_t elems) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k || !buf) return;
    @autoreleasepool {
        IOSurfaceLock(k->io_out, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(k->io_out), buf, elems * 2);
        IOSurfaceUnlock(k->io_out, 0, NULL);
    }
}

// No-copy eval: the caller staged the input bytes DIRECTLY into the handle's
// io_in surface (a GPU kernel writing the Metal texture view) and will
// consume io_out the same way — skip both host memcpys. Same request, same
// retry-once contract as ane_t0_eval (the Rust side retries).
__attribute__((visibility("default")))
int ane_t0_eval_nc(void *handle, int proc_idx) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k || proc_idx < 0 || proc_idx >= k->n_procs) {
        set_last_error(@("bad handle or proc_idx out of range"));
        return 0;
    }
    @autoreleasepool {
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 0, k->options,
            k->requests[proc_idx], &e);
        if (!ok) {
            set_last_error(e ? e.description
                            : @"evaluateWithQoS returned NO with nil error");
            fprintf(stderr, "[bridge] eval_nc FAIL (proc %d): %s\n",
                    proc_idx, g_ane_last_error);
            return 0;
        }
        return 1;
    }
}

// Wrap the handle's io surfaces as R32Uint MTLTextures on the SHARED device
// (the same id<MTLDevice> CubeCL uses — Issue 663 extraction). Returns the
// two textures RETAINED (+1; the Rust side wraps with from_ptr and releases
// on drop). Fails (writes NULLs) if the wrap is rejected.
__attribute__((visibility("default")))
void ane_t0_mtl_textures(void *handle, void *mtl_device,
                         void **tex_in, void **tex_out) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    *tex_in = NULL;
    *tex_out = NULL;
    if (!k || !mtl_device) {
        set_last_error(@("ane_t0_mtl_textures: bad handle/device"));
        return;
    }
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device;
        MTLTextureDescriptor *tdIn =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                width:k->tex_w_in
                                                               height:k->tex_h_in
                                                            mipmapped:NO];
        tdIn.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        tdIn.storageMode = MTLStorageModeShared;
        id<MTLTexture> tIn = [dev newTextureWithDescriptor:tdIn
                                                  iosurface:k->io_in
                                                      plane:0];
        MTLTextureDescriptor *tdOut =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                width:k->tex_w_out
                                                               height:k->tex_h_out
                                                            mipmapped:NO];
        tdOut.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        tdOut.storageMode = MTLStorageModeShared;
        id<MTLTexture> tOut = [dev newTextureWithDescriptor:tdOut
                                                   iosurface:k->io_out
                                                       plane:0];
        if (!tIn || !tOut) {
            set_last_error(@("newTextureWithDescriptor:iosurface: returned nil"));
            return;
        }
        *tex_in = (void *)CFBridgingRetain(tIn);
        *tex_out = (void *)CFBridgingRetain(tOut);
    }
}

// The texture geometry for one surface byte size (matches tex_geom exactly
// — the Rust-side kernel indexing uses these, never its own arithmetic).
__attribute__((visibility("default")))
void ane_t0_tex_geom_for_bytes(size_t bytes, uint32_t *w_px, uint32_t *rows) {
    tex_geom(bytes, w_px, rows);
}

// Blocking evaluate of procedure `proc_idx`. Returns 1 on success.
__attribute__((visibility("default")))
int ane_t0_eval(void *handle, const uint16_t *in_u16, uint16_t *out_u16,
                size_t in_elems, size_t out_elems, int proc_idx) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k || proc_idx < 0 || proc_idx >= k->n_procs) {
        set_last_error(@("bad handle or proc_idx out of range"));
        return 0;
    }
    @autoreleasepool {
        IOSurfaceLock(k->io_in, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(k->io_in), in_u16, in_elems * 2);
        IOSurfaceUnlock(k->io_in, 0, NULL);

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 0, k->options,
            k->requests[proc_idx], &e);
        if (!ok) {
            set_last_error(e ? e.description
                            : @"evaluateWithQoS returned NO with nil error");
            fprintf(stderr, "[bridge] eval FAIL (proc %d): %s\n",
                    proc_idx, g_ane_last_error);
            return 0;
        }

        IOSurfaceLock(k->io_out, 1, NULL);
        memcpy(out_u16, IOSurfaceGetBaseAddress(k->io_out), out_elems * 2);
        IOSurfaceUnlock(k->io_out, 1, NULL);
        return 1;
    }
}
void ane_t0_free(void *handle) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k) return;
    @autoreleasepool {
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            k->model, @selector(unloadWithQoS:error:), 0, &e);
        objc_release(k->model);
        objc_release(k->options);
        objc_release(k->requests);
        CFRelease(k->io_in);
        CFRelease(k->io_out);
        [[NSFileManager defaultManager] removeItemAtPath:
            [NSString stringWithUTF8String:k->tmp_dir] error:nil];
        free(k);
    }
}

// Last compile/eval error text (see set_last_error above).
__attribute__((visibility("default")))
const char *ane_t0_last_error(void) {
    return g_ane_last_error;
}

// Probe 778: dump a private class's method table (both class (+) and
// instance (-) selectors with type encodings) — the daemon-bypass selector
// hunt (Issue 769 T2) + any flush/sync primitive that could repair the
// ANE→GPU surface-coherency break. Diagnostic only.
__attribute__((visibility("default")))
void ane_t0_dump_class_methods(const char *class_name) {
    @autoreleasepool {
        if (!class_name) return;
        Class c = NSClassFromString([NSString stringWithUTF8String:class_name]);
        if (!c) {
            fprintf(stderr, "[dump] class %s NOT FOUND\n", class_name);
            return;
        }
        fprintf(stderr, "[dump] === %s ===\n", class_name);
        unsigned int n = 0;
        Method *ms = class_copyMethodList(object_getClass(c), &n);
        for (unsigned int i = 0; i < n; i++) {
            fprintf(stderr, "[dump] + %s (%s)\n",
                    sel_getName(method_getName(ms[i])),
                    method_getTypeEncoding(ms[i]) ?: "?");
        }
        free(ms);
        ms = class_copyMethodList(c, &n);
        for (unsigned int i = 0; i < n; i++) {
            fprintf(stderr, "[dump] - %s (%s)\n",
                    sel_getName(method_getName(ms[i])),
                    method_getTypeEncoding(ms[i]) ?: "?");
        }
        free(ms);
    }
}

// Probe 778: wrap the io_out surface's OWN pages in a no-copy MTLBuffer —
// the BUFFER cacheability domain (the ladder only ever exercised the
// texture-fetch path). Length is the surface's alloc size rounded DOWN to a
// 16 KiB page multiple (the newBufferWithBytesNoCopy contract); returns the
// buffer RETAINED (+1) or NULL (unaligned / rejected).
__attribute__((visibility("default")))
void *ane_t0_out_no_copy_buffer(void *handle, void *mtl_device) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k || !mtl_device) return NULL;
    @autoreleasepool {
        void *base = IOSurfaceGetBaseAddress(k->io_out);
        NSUInteger size = IOSurfaceGetAllocSize(k->io_out);
        const NSUInteger kPage = 16384;
        size = (size / kPage) * kPage;
        if (!base || size == 0) return NULL;
        id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device;
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:base
                                                   length:size
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (!buf) return NULL;
        return (__bridge_retained void *)buf;
    }
}

// ══ Probe 779: runtime-driven private-API surface ═══════════════════════════
// The Bench 778 armed fix (Issue 769 T2): probe `doEvaluateDirectWithModel:`
// (in-process eval — the completion-signal candidate + ane-infer's +10%),
// `mapIOSurfaces...` (the proper producer→consumer surface fence) and
// `buffersReady...` on the private client. EVERY invocation is NSInvocation-
// driven — signatures come from the runtime's OWN method tables, so a wrong
// guess reports an encoding mismatch instead of corrupting the stack.

// +0 read of an object ivar (NULL-safe: non-'@' encodings return nil rather
// than a garbage pointer).
static id p779_ivar_obj(id obj, const char *name) {
    if (!obj || !name) return nil;
    Ivar iv = class_getInstanceVariable(object_getClass(obj), name);
    if (!iv) return nil;
    const char *te = ivar_getTypeEncoding(iv);
    if (!te || te[0] != '@') return nil;
    return object_getIvar(obj, iv);
}

__attribute__((visibility("default")))
const char *ane_t0_class_name(void *obj) {
    if (!obj) return "(nil)";
    return class_getName(object_getClass((__bridge id)obj));
}

__attribute__((visibility("default")))
void ane_t0_release(void *obj) {
    if (obj) objc_release((__bridge id)obj);
}

__attribute__((visibility("default")))
void *ane_t0_retain(void *obj) {
    return obj ? (__bridge_retained void *)objc_retain((__bridge id)obj) : NULL;
}

__attribute__((visibility("default")))
void ane_t0_dump_ivars(const char *class_name) {
    @autoreleasepool {
        if (!class_name) return;
        Class c = NSClassFromString([NSString stringWithUTF8String:class_name]);
        if (!c) { fprintf(stderr, "[ivars] class %s NOT FOUND\n", class_name); return; }
        fprintf(stderr, "[ivars] === %s ===\n", class_name);
        unsigned int n = 0;
        Ivar *iv = class_copyIvarList(c, &n);
        for (unsigned int i = 0; i < n; i++)
            fprintf(stderr, "[ivars] %s (%s) @+%td\n",
                    ivar_getName(iv[i]) ?: "?", ivar_getTypeEncoding(iv[i]) ?: "?",
                    ivar_getOffset(iv[i]));
        free(iv);
    }
}

static int g_p779_nodes;
static void p779_dump_graph_rec(id obj, int depth, int max_depth) {
    if (!obj || depth > max_depth || g_p779_nodes > 256) return;
    g_p779_nodes++;
    const char *cn = class_getName(object_getClass(obj));
    fprintf(stderr, "[graph] %*s- %s (%p)\n", depth * 2, "", cn, obj);
    if (depth >= max_depth) return;
    if ([obj isKindOfClass:[NSArray class]]) {
        for (id v in obj) p779_dump_graph_rec(v, depth + 1, max_depth);
        return;
    }
    if ([obj isKindOfClass:[NSDictionary class]]) {
        for (id v in [obj objectEnumerator]) p779_dump_graph_rec(v, depth + 1, max_depth);
        return;
    }
    if (strncmp(cn, "NS", 2) == 0) return;  // other Foundation: don't descend
    unsigned int n = 0;
    Ivar *iv = class_copyIvarList(object_getClass(obj), &n);
    for (unsigned int i = 0; i < n; i++) {
        const char *te = ivar_getTypeEncoding(iv[i]);
        if (te && te[0] == '@') {
            id v = object_getIvar(obj, iv[i]);
            if (v) p779_dump_graph_rec(v, depth + 1, max_depth);
            else fprintf(stderr, "[graph] %*s. %s = nil\n",
                         (depth + 1) * 2, "", ivar_getName(iv[i]) ?: "?");
        }
    }
    free(iv);
}

// Where does the live _ANEClient hang off OUR model/requests? Diagnostic —
// the Rust probe navigates with the get_ivar/array exports after reading
// this map.
__attribute__((visibility("default")))
void ane_t0_dump_graph(void *handle) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k) return;
    @autoreleasepool {
        g_p779_nodes = 0;
        fprintf(stderr, "[graph] === model graph ===\n");
        p779_dump_graph_rec(k->model, 0, 3);
        g_p779_nodes = 0;
        fprintf(stderr, "[graph] === requests ===\n");
        for (id r in k->requests) p779_dump_graph_rec(r, 0, 2);
    }
}

// Navigation: handle → model / per-procedure request (both RETAINED +1).
__attribute__((visibility("default")))
void *ane_t0_model_of(void *handle) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k) return NULL;
    return (__bridge_retained void *)k->model;
}
__attribute__((visibility("default")))
void *ane_t0_request_of(void *handle, int idx) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k || idx < 0 || idx >= k->n_procs) return NULL;
    id r = [k->requests objectAtIndex:(NSUInteger)idx];
    return r ? (__bridge_retained void *)r : NULL;
}
// The eval options dict (pinning hints) — the direct-eval selector takes it.
__attribute__((visibility("default")))
void *ane_t0_options_of(void *handle) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k) return NULL;
    return (__bridge_retained void *)k->options;
}

// Probe 779: the io_out IOSurface LOCK/UNLOCK pair with no memcpy — the
// lock blocks while the ANE holds the surface (the host-read fence stage(1)
// always wins through), so lock+unlock is the cheapest possible writeback
// fence for a subsequent GPU read. Returns 1 on lock success.
__attribute__((visibility("default")))
int ane_t0_out_fence(void *handle) {
    AneT0Kernel *k = (AneT0Kernel *)handle;
    if (!k) return 0;
    uint32_t tid = 0;
    if (IOSurfaceLock(k->io_out, kIOSurfaceLockReadOnly, &tid) != 0) return 0;
    IOSurfaceUnlock(k->io_out, kIOSurfaceLockReadOnly, &tid);
    return 1;
}

// Instance object-ivar table (object ivars only; count/nameAt for the Rust
// BFS). Name is copied into the caller's buffer.
__attribute__((visibility("default")))
int ane_t0_obj_ivar_count(void *obj) {
    if (!obj) return 0;
    unsigned int n = 0;
    free(class_copyIvarList(object_getClass((__bridge id)obj), &n));
    return (int)n;
}
__attribute__((visibility("default")))
int ane_t0_obj_ivar_name_at(void *obj, int idx, char *buf, int buflen) {
    if (!obj || !buf || buflen <= 0) return 0;
    unsigned int n = 0;
    Ivar *iv = class_copyIvarList(object_getClass((__bridge id)obj), &n);
    int ok = 0;
    if (idx >= 0 && (unsigned)idx < n) {
        const char *nm = ivar_getName(iv[idx]);
        if (nm) { strncpy(buf, nm, (size_t)buflen - 1); buf[buflen - 1] = 0; ok = 1; }
    }
    free(iv);
    return ok;
}
__attribute__((visibility("default")))
void *ane_t0_get_ivar_obj(void *obj, const char *ivar_name) {
    @autoreleasepool {
        id v = p779_ivar_obj((__bridge id)obj, ivar_name);
        return v ? (__bridge_retained void *)v : NULL;
    }
}

// NSArray/NSDictionary value traversal (RETAINED elems; count 0 for non-
// containers so the Rust BFS can just try it).
__attribute__((visibility("default")))
unsigned long ane_t0_array_count(void *arr) {
    if (!arr) return 0;
    id a = (__bridge id)arr;
    if (![a respondsToSelector:@selector(count)]) return 0;
    return [a count];
}
__attribute__((visibility("default")))
void *ane_t0_array_elem(void *arr, unsigned long idx) {
    if (!arr) return NULL;
    id a = (__bridge id)arr;
    if (![a respondsToSelector:@selector(objectAtIndex:)]) return NULL;
    if (idx >= [a count]) return NULL;
    id v = [a objectAtIndex:idx];
    return v ? (__bridge_retained void *)v : NULL;
}

// NSInvocation-driven selector call. Argument kinds must match the runtime's
// own encoding (mismatch = clean error, never an ABI gamble). '^@' args get
// an internal NSError** and any returned error lands in err_text.
typedef struct {
    char kind;   // 'o' object | 'p' pointer-value | 'I' u32 | 'i' i32 | 'Q' u64
                 // | 'c' bool | 'f' f32-bits-in-u | 'd' f64-bits-in-u
    void *obj;
    unsigned long long u;
} ane_invoke_arg_t;

typedef struct {
    int ok;              // 1 = invoked (exception-free); 0 = mismatch/exception
    char ret_kind;       // the runtime's own return encoding ('v' '@' 'B' 'Q' ...)
    void *ret_obj;       // '@' returns, RETAINED +1 (caller ane_t0_release)
    long long ret_i;     // 'B'/'c'/'i'/'I'/'q'/'Q'/...
    double ret_f;        // 'f'/'d'
    char err_text[256];  // mismatch text, ObjC exception, or the NSError
} ane_invoke_ret_t;

__attribute__((visibility("default")))
int ane_t0_invoke(void *receiver, const char *sel_name,
                  const ane_invoke_arg_t *args, int n_args,
                  ane_invoke_ret_t *out) {
    memset(out, 0, sizeof(*out));
    out->ret_kind = 'v';
    if (!receiver || !sel_name || !out) return 0;
    @autoreleasepool {
        id rcv = (__bridge id)receiver;
        SEL sel = sel_registerName(sel_name);
        NSMethodSignature *sig = [rcv methodSignatureForSelector:sel];
        if (!sig) {
            snprintf(out->err_text, sizeof(out->err_text), "no signature for %s on %s",
                     sel_name, class_getName(object_getClass(rcv)));
            return 0;
        }
        out->ret_kind = sig.methodReturnType ? sig.methodReturnType[0] : 'v';
        if ((int)(sig.numberOfArguments - 2) != n_args) {
            snprintf(out->err_text, sizeof(out->err_text),
                     "arity: %s wants %lu got %d (sig %s)", sel_name,
                     (unsigned long)(sig.numberOfArguments - 2), n_args,
                     sig.methodReturnType ?: "?");
            return 0;
        }
        NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
        [inv setTarget:rcv];
        [inv setSelector:sel];
        int err_slot = -1;
        NSError *err = nil;
        for (int i = 0; i < n_args; i++) {
            NSUInteger idx = (NSUInteger)i + 2;
            const char *t = [sig getArgumentTypeAtIndex:idx];
            char k = args[i].kind;
            if (k == 'o' && t[0] == '@') {
                id v = (__bridge id)args[i].obj;
                [inv setArgument:&v atIndex:idx];
            } else if (k == 'o' && t[0] == '^') {
                if (t[1] == '@') {  // NSError** — capture through-write
                    void *ep = &err;
                    [inv setArgument:&ep atIndex:idx];
                    err_slot = i;
                } else {
                    [inv setArgument:&args[i].obj atIndex:idx];
                }
            } else if (k == 'I' && (t[0] == 'I' || t[0] == 'i')) {
                unsigned int v = (unsigned int)args[i].u;
                [inv setArgument:&v atIndex:idx];
            } else if (k == 'Q' && (t[0] == 'Q' || t[0] == 'q' || t[0] == 'l' || t[0] == 'L')) {
                [inv setArgument:&args[i].u atIndex:idx];
            } else if (k == 'i' && t[0] == 'i') {
                int v = (int)args[i].u;
                [inv setArgument:&v atIndex:idx];
            } else if (k == 'c' && (t[0] == 'c' || t[0] == 'B')) {
                BOOL v = args[i].u ? YES : NO;
                [inv setArgument:&v atIndex:idx];
            } else if (k == 'p' && t[0] == '^') {
                [inv setArgument:&args[i].obj atIndex:idx];
            } else if (k == 'f' && t[0] == 'f') {
                float v; memcpy(&v, &args[i].u, 4);
                [inv setArgument:&v atIndex:idx];
            } else if (k == 'd' && t[0] == 'd') {
                double v; memcpy(&v, &args[i].u, 8);
                [inv setArgument:&v atIndex:idx];
            } else {
                snprintf(out->err_text, sizeof(out->err_text),
                         "arg %d kind '%c' vs encoding '%c%c' on %s",
                         i, k, t[0] ?: '?', t[1] ?: ' ', sel_name);
                return 0;
            }
        }
        @try {
            [inv invoke];
        } @catch (NSException *ex) {
            snprintf(out->err_text, sizeof(out->err_text), "exception: %s (%s)",
                     ex.name.UTF8String ?: "?", ex.reason.UTF8String ?: "?");
            return 0;
        }
        switch (out->ret_kind) {
            case '@': {
                __unsafe_unretained id rv = nil;
                [inv getReturnValue:&rv];
                out->ret_obj = rv ? (__bridge_retained void *)rv : NULL;
                break;
            }
            case 'B': case 'c': case 'C': {
                BOOL b = NO; [inv getReturnValue:&b]; out->ret_i = b; break;
            }
            case 'i': case 'I': {
                int v = 0; [inv getReturnValue:&v]; out->ret_i = v; break;
            }
            case 'l': case 'L': case 'q': case 'Q': {
                long long v = 0; [inv getReturnValue:&v]; out->ret_i = v; break;
            }
            case 'f': { float v = 0; [inv getReturnValue:&v]; out->ret_f = v; break; }
            case 'd': { double v = 0; [inv getReturnValue:&v]; out->ret_f = v; break; }
            default: break;
        }
        if (err_slot >= 0 && err) {
            snprintf(out->err_text, sizeof(out->err_text), "NSError: %s",
                     err.localizedDescription.UTF8String ?: "?");
        }
        return 1;
    }
}

// ── eviction hook (769 T2, measured) ────────────────────────────────────
//
// Verbatim from riir-poc's ane_t0_bridge.m (the twin rule for this file's
// cache machinery — keep the two copies byte-identical). purgeCompiledModel
// is the only sanctioned store-eviction selector, but it THROWS a foreign
// ObjC exception when called on a fresh descriptor-only instance (no
// compiled artifact in this process). oMLX only ever purges the instance
// that just FAILED a load, i.e. one holding process-side compiled state.
// This maintenance entry reproduces a valid precondition (compile+load)
// and wraps the purge in @try so any throw is reported, never fatal.
// NOT a compile path: ignores ANE_T0_CACHE, builds no kernel, evaluates
// nothing. Returns 1 with the identifier copied into out_id on success,
// 0 on any failure.
__attribute__((visibility("default")))
int ane_t0_cache_purge(const char *mil_path, const char *blob_path,
                       char *out_id, int out_cap) {
    @autoreleasepool {
        static bool loaded = false;
        static bool attempted = false;
        if (!attempted) {
            attempted = true;
            loaded = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                            "AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL) != NULL;
        }
        if (!loaded) return 0;
        if (!mil_path || !blob_path || !out_id || out_cap <= 0) return 0;

        NSError *e = nil;
        NSData *milData = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:mil_path]];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blob_path]];
        if (!milData || !blob) { fprintf(stderr, "[bridge] purge: file read failed\n"); return 0; }

        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I = NSClassFromString(@"_ANEInMemoryModel");
        if (!D || !I) { fprintf(stderr, "[bridge] purge: classes unavailable\n"); return 0; }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D,
            @selector(modelWithMILText:weights:optionsPlist:), milData,
            @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}}, nil);
        if (!desc) { fprintf(stderr, "[bridge] purge: desc FAIL\n"); return 0; }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "[bridge] purge: model FAIL\n"); return 0; }
        NSString *hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        if (![hx isKindOfClass:[NSString class]]) { fprintf(stderr, "[bridge] purge: identifier type\n"); return 0; }
        if (![mdl respondsToSelector:@selector(purgeCompiledModel)]) { fprintf(stderr, "[bridge] purge: selector unavailable\n"); return 0; }

        // Stage EXACTLY like ane_t0_compile: the compiler resolves
        // @model_path against temp/<identifier>/ — without these files its
        // own compile re-derives InvalidMILProgram (measured 769 T2).
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
  withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        // Precondition (measured): a purge needs process-side compiled state.
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 0, @{}, &e)) {
            fprintf(stderr, "[bridge] purge: compile FAIL: %s\n", e ? e.description.UTF8String : "?");
            return 0;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 0, @{}, &e)) {
            fprintf(stderr, "[bridge] purge: load FAIL: %s\n", e ? e.description.UTF8String : "?");
            return 0;
        }

        @try {
            ((void (*)(id, SEL))objc_msgSend)(mdl, @selector(purgeCompiledModel));
        } @catch (NSException *ex) {
            fprintf(stderr, "[bridge] purge threw on a loaded instance: %s\n",
                    ex.description.UTF8String ?: "?");
            [fm removeItemAtPath:td error:nil];
            return 0;
        }
        [fm removeItemAtPath:td error:nil];
        strlcpy(out_id, hx.UTF8String ?: "", (size_t)out_cap);
        return 1;
    }
}
