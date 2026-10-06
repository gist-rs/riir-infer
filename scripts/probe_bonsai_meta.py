"""Probe the exact keys of the league PQ2_0 pack (nextn / hadamard / quantization class)."""
import mmap
import struct

P = r"../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"

f = open(P, "rb")
data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
ver, n_tensors, n_kv = struct.unpack("<IQQ", data[4:24])
pos = 24
# GGUF v3 value types: 7 is BOOL (1 byte), 10/11/12 are U64/I64/F64.
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
    raise SystemExit(f"unhandled gguf value type {t} at byte {pos} (key index {rval.i})")


rval.i = 0

for i in range(n_kv):
    rval.i = i
    k = rstr()
    v = rval()
    if any(w in k for w in ("nextn", "full_attention", "file_type", "quantization", "general.size")):
        if isinstance(v, list) and len(v) > 6:
            v = f"len {len(v)} head {v[:3]}"
        print(f"{k:50s} = {v}")
print("OK keys", n_kv)
