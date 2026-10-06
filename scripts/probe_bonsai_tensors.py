"""One block's tensor inventory: name, type, shape — the requant policy input."""
import mmap
import struct
import sys

P = r"../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"

f = open(P, "rb")
data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
ver, n_tensors, n_kv = struct.unpack("<IQQ", data[4:24])
pos = 24
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMTS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


def rstr():
    global pos
    n = struct.unpack("<Q", data[pos : pos + 8])[0]
    pos += 8
    s = data[pos : pos + n].decode("utf-8", "replace")
    pos += n
    return s


def rval():
    global pos
    t = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4
    if t == 8:
        return rstr()
    if t in SIZES:
        v = struct.unpack(FMTS[t], data[pos : pos + SIZES[t]])[0]
        pos += SIZES[t]
        return v
    if t == 9:
        et = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        n = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8
        if et == 8:
            return [rstr() for _ in range(n)]
        if et in SIZES:
            nb = n * SIZES[et]
            fmt = "<" + str(n) + FMTS[et][1:]
            v = struct.unpack(fmt, data[pos : pos + nb])
            pos += nb
            return list(v)
        raise SystemExit(f"array element type {et} at {pos}")
    raise SystemExit(f"unhandled gguf value type {t} at byte {pos}")


for _ in range(n_kv):
    rstr()
    rval()

TYPE_NAMES = {0: "F32", 1: "F16", 30: "BF16", 32: "Q4_0", 34: "Q8_0", 36: "Q2_K",
              40: "Q4_K", 42: "Q2_0", 142: "Q2_0(relabel)", 143: "PTQ1_0"}
TYPES = {}
for _ in range(n_tensors):
    name = rstr()
    n_dims = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4
    shape = []
    for _ in range(n_dims):
        d = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8
        shape.append(d)
    ttype, off = struct.unpack("<IQ", data[pos : pos + 12])
    pos += 12
    TYPES.setdefault(ttype, []).append((name, shape))

for t, names in sorted(TYPES.items()):
    tn = TYPE_NAMES.get(t, f"type{t}")
    print(f"== {tn} (id {t}): {len(names)} tensors")
    for name, shape in names[:6]:
        print(f"   {name:42s} {shape}")
    if len(names) > 6:
        print(f"   ... +{len(names) - 6} more")
print("DONE")
