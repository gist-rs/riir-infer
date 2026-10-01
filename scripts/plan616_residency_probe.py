import json, struct

def inventory(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    gemm = 0
    small = 0
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        sh = v['shape']
        r = 1
        for s in sh:
            r *= s
        if v['dtype'] == 'Q8_0':
            gemm += r
        else:
            small += r
    return gemm, small

for ck in ['english', 'typed']:
    g, s = inventory(f'/Users/katopz/.cache/riir-reflex/laya/{ck}/derived/model.q8.safetensors')
    q8_bytes = (g // 32) * 34
    f32_dev = (g + s) * 4
    print(f'{ck}: Q8_0 elements {g:,} | 1D/other F16 elements {s:,}')
    print(f'  device today (F32 widened): {f32_dev/1e9:.3f} GB -> Q8-resident GEMM + F32 small: {(q8_bytes + s*4)/1e9:.3f} GB ({(q8_bytes+s*4)/f32_dev*100:.1f}%)')
    print(f'  Phase 1 host: Q8 map {q8_bytes/1e9:.3f} GB (vs F32 {f32_dev/1e9:.3f} GB today)')
