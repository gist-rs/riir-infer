#!/usr/bin/env python3
"""Plan 623 T4.1/T4.2 — first real TP cell: two half-model ranks over JACCL.

Tensor-parallel sharding of the Bonsai-2 2-bit pack (plan623_pack_artifact
loader) across 2 ranks. BOTH ranks run ALL 64 layers; the big matmuls are
sharded Megatron-style and the row-parallel partials reduce over
mx.distributed all_sum on the jaccl (RDMA) backend (Bench 905: hidden-class
all_sum ~19-23 us):

  column (out-dim split, activations full):  q,k,v(+fused per-head gate),
      mlp gate,up, GDN in_proj_qkv/z/b/a
  row (in-dim split, partial + all_sum):     o_proj, mlp down_proj,
      GDN out_proj
  vocab split (out-dim + all_gather):        lm_head (greedy argmax over the
      gathered full logits — identical on both ranks, no token exchange)
  replicated:                                embed_tokens (inverse-fwht
      lookup), all RMSNorms, q/k norms, GDN norm/dt_bias/A_log/conv1d
      (conv channels follow the head slices)

Hadamard law (the pack's fwht is BLOCK-LOCAL, runtime.py fwht reshapes
(-1, block) + elementwise signflip): an in-dim slice is exact iff it is
block-aligned — 3072 = 3x1024 for o_proj/out_proj; the MLP intermediate
17408 = 17x1024 has an ODD block count, so the even 8704/8704 split would
cut block 8 across ranks — the split is BLOCK-ALIGNED UNEVEN instead
(9216 + 8192 at N=2, disclosed). Column splits never touch the in-dim
transform; lm_head/embed signs are full-width (input is the full hidden).

Determinism (Issue 940): every synced f16 reduction here is a 2-rank
all_sum — a single pairwise add, IEEE-commutative, hence rank-order-free
deterministic; all_gather concatenates in fixed rank order (vocab order);
greedy argmax = first-index over identical full logits on both ranks.
Replay determinism is asserted by running the bench twice and comparing
gen tokens. Bit-identity vs 1-box is NOT claimed (down_proj/out_proj
reduction order changes) — token drift vs the N=1 arm is reported.

Env: I623_TP_N (1|2, default 2; N=1 = the no-comms sanity arm through this
harness, comparable to the Bench-904 1-box rows), I623_PACK, I623_PP (512),
I623_TG (128), I623_OUT. Rank env for N=2 (Bench 905 posture):
  MLX_RANK MLX_JACCL_COORDINATOR=192.168.10.1:50100 MLX_IBV_DEVICES=...
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn

# A non-UTF-8 console (cp874) would kill this script on its first
# non-ASCII glyph; degrade the glyph, never the verdict (katgpt-rs
# Issue 804). A captured stream without reconfigure is left alone.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parent.parent
PACK = Path(os.environ.get("I623_PACK", REPO / ".raw" / "bonsai2-mlx-2bit"))
TP_N = int(os.environ.get("I623_TP_N", "2"))
PP = int(os.environ.get("I623_PP", "512"))
TG = int(os.environ.get("I623_TG", "128"))
COMM = os.environ.get("I623_TP_COMM", "1") != "0"  # 0 = diag: skip comms (WRONG tokens, timing only)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def die(msg):
    raise SystemExit(f"[tp2] {msg}")


def require(cond, msg):
    if not cond:
        die(msg)


def get_module(root, path):
    parent = root
    parts = path.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def packed_arrays(mod):
    return mod.weight, mod.scales, mod.biases, mod.block, mod.signs


def col_slice(mod, lo, hi):
    """out-dim (row) slice of a packed module — signs untouched"""
    w, s, b, block, signs = packed_arrays(mod)
    require(0 <= lo < hi <= w.shape[0], f"col slice [{lo}:{hi}] of {w.shape}")
    return (
        w[lo:hi],
        s[lo:hi],
        b[lo:hi],
        block,
        signs,
    )


def row_slice(mod, lo, hi, block_size):
    """in-dim (column) slice of a packed module — block-aligned required.

    packed [out, in/16] uint32, scales/biases [out, in/128]; the caller
    keeps the RANK-LOCAL activation slice (block-aligned), so the signs
    slice is exact iff (hi-lo) % block_size == 0 AND lo % block_size == 0.
    """
    w, s, b, block, signs = packed_arrays(mod)
    require(lo % block_size == 0 and (hi - lo) % block_size == 0,
            f"row slice [{lo}:{hi}] not {block_size}-aligned")
    require(hi <= w.shape[1] * 16, f"row slice [{lo}:{hi}] of in={w.shape[1] * 16}")
    w2 = w[:, lo // 16: hi // 16]
    s2 = s[:, lo // 128: hi // 128]
    b2 = b[:, lo // 128: hi // 128]
    sgn = signs[lo:hi] if signs is not None else None
    return w2, s2, b2, block, sgn


def block_split(total, n, rank, block_size):
    """block-aligned near-even split of `total` into n rank ranges"""
    require(total % block_size == 0, f"{total} not {block_size}-divisible")
    nb = total // block_size
    base, rem = divmod(nb, n)
    lo_b = rank * base + min(rank, rem)
    hi_b = lo_b + base + (1 if rank < rem else 0)
    return lo_b * block_size, hi_b * block_size


def build_shard(full, config, rank, n, group, Packed, fwht):
    """build this rank's sharded model from the loaded full model"""
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
    from mlx_lm.models.qwen3_next import Qwen3NextAttention as Attention
    from mlx_lm.models.qwen3_next import Qwen3NextMLP as MLP

    tc = config["text_config"]
    block_size = max((r.get("block") or 0) for r in config["modules"])
    interval = tc.get("full_attention_interval", 4)
    hid = tc["hidden_size"]
    inter = tc["intermediate_size"]
    vocab = tc["vocab_size"]

    args = TextModelArgs.from_dict(tc)
    args.num_attention_heads //= n
    args.num_key_value_heads //= n
    args.linear_num_value_heads //= n
    args.linear_num_key_heads //= n

    nh_l = args.num_attention_heads
    nkv_l = args.num_key_value_heads
    hd = args.head_dim
    nk_l = args.linear_num_key_heads
    nv_l = args.linear_num_value_heads
    hkd = args.linear_key_head_dim
    hvd = args.linear_value_head_dim
    kd_l = nk_l * hkd          # local GDN key dim
    vd_l = nv_l * hvd          # local GDN value dim
    kd_full = kd_l * n
    inter_lo, inter_hi = block_split(inter, n, rank, block_size)
    require(vocab % n == 0, "vocab not TP-divisible")
    vocab_lo, vocab_hi = rank * vocab // n, (rank + 1) * vocab // n

    class PackedToAll(nn.Module):
        """row-parallel packed matmul: block-local fwht on the RANK's
        block-aligned activation slice, half matmul, all_sum the partials.
        Issue 940: a 2-rank all_sum is ONE pairwise add (IEEE-commutative,
        order-free deterministic); for n>2 the fixed ring order would need
        pinning — this cell ships n in (1, 2) only."""

        def __init__(self, arrays, group):
            super().__init__()
            w, s, b, block, signs = arrays
            self.weight, self.scales, self.biases = w, s, b
            self.block = block
            self.signs = signs
            self.group = group
            self.dtype = mx.float16

        def __call__(self, x):
            if self.block:
                x = fwht(x, self.block, self.signs)
            part = mx.quantized_matmul(
                x, self.weight, self.scales, self.biases,
                transpose=True, group_size=128, bits=2,
            )
            if self.group is None or not COMM:
                return part
            return mx.distributed.all_sum(part, group=self.group)

    class PackedVocabHead(nn.Module):
        """column-parallel lm_head: half the vocab rows, all_gather the
        halves in fixed rank order == vocab order (deterministic concat)."""

        def __init__(self, mod, group):
            super().__init__()
            self.weight, self.scales, self.biases, self.block, self.signs = (
                col_slice(mod, vocab_lo, vocab_hi)
            )
            self.group = group
            self.dtype = mx.float16
            self.vocab_lo = vocab_lo

        def __call__(self, x):
            y = mx.quantized_matmul(
                fwht(x, self.block, self.signs) if self.block else x,
                self.weight, self.scales, self.biases,
                transpose=True, group_size=128, bits=2,
            )
            if self.group is None or not COMM:
                return y
            t = mx.distributed.all_gather(
                y.transpose(2, 0, 1), group=self.group
            )  # (n*vocab_l, B, L) rank-major == vocab order
            return t.transpose(1, 2, 0)

    def packed_col(path, lo, hi):
        """out-dim slice — the module's own block/signs are authoritative
        (in-dim full width, untouched by an out split). Packed modules
        slice the packed triple; plain Linears (the tiny GDN b/a projs)
        slice the float weight."""
        parent, name = get_module(full, path)
        mod = getattr(parent, name)
        if hasattr(mod, "scales"):
            return Packed(col_slice(mod, lo, hi)[:3], mod.block, mod.signs,
                          False, mx.float16)
        require(not getattr(mod, "bias", None) or mod.bias is None,
                f"{path}: biased plain linear in the TP path")
        return PlainCol(mod.weight[lo:hi])

    class PlainCol(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = w

        def __call__(self, x):
            return x @ self.weight.T

    layers_src = full.model.layers
    n_layers = len(layers_src)
    layers = []
    for i, src in enumerate(layers_src):
        is_linear = (i + 1) % interval != 0
        layer = {"is_linear": is_linear,
                 "input_layernorm": src.input_layernorm,
                 "post_attention_layernorm": src.post_attention_layernorm}
        base = f"model.layers.{i}."
        if is_linear:
            gdn = GatedDeltaNet(args)
            gdn.sharding_group = None
            sa = src.linear_attn
            # stored (pack-time vperm) layout is key-head-grouped, so every
            # rank slice below is a contiguous stored-space range
            qkv = sa.in_proj_qkv
            r_qkv = (
                mx.concat([
                    qkv.weight[kd_l * rank: kd_l * (rank + 1)],
                    qkv.weight[kd_full + kd_l * rank: kd_full + kd_l * (rank + 1)],
                    qkv.weight[2 * kd_full + vd_l * rank: 2 * kd_full + vd_l * (rank + 1)],
                ]),
                mx.concat([
                    qkv.scales[kd_l * rank: kd_l * (rank + 1)],
                    qkv.scales[kd_full + kd_l * rank: kd_full + kd_l * (rank + 1)],
                    qkv.scales[2 * kd_full + vd_l * rank: 2 * kd_full + vd_l * (rank + 1)],
                ]),
                mx.concat([
                    qkv.biases[kd_l * rank: kd_l * (rank + 1)],
                    qkv.biases[kd_full + kd_l * rank: kd_full + kd_l * (rank + 1)],
                    qkv.biases[2 * kd_full + vd_l * rank: 2 * kd_full + vd_l * (rank + 1)],
                ]),
            )
            gdn.in_proj_qkv = Packed(r_qkv, qkv.block, qkv.signs, False, mx.float16)
            gdn.in_proj_z = packed_col(base + "linear_attn.in_proj_z",
                                       vd_l * rank, vd_l * (rank + 1))
            gdn.in_proj_b = packed_col(base + "linear_attn.in_proj_b",
                                       nv_l * rank, nv_l * (rank + 1))
            gdn.in_proj_a = packed_col(base + "linear_attn.in_proj_a",
                                       nv_l * rank, nv_l * (rank + 1))
            gdn.dt_bias = mx.contiguous(sa.dt_bias[nv_l * rank: nv_l * (rank + 1)])
            gdn.A_log = mx.contiguous(sa.A_log[nv_l * rank: nv_l * (rank + 1)])
            conv_w = sa.conv1d.weight  # (conv_dim, 1, k), channels [q|k|v] stored
            gdn.conv1d.weight = mx.contiguous(mx.concat([
                conv_w[kd_l * rank: kd_l * (rank + 1)],
                conv_w[kd_full + kd_l * rank: kd_full + kd_l * (rank + 1)],
                conv_w[2 * kd_full + vd_l * rank: 2 * kd_full + vd_l * (rank + 1)],
            ]))
            gdn.norm.weight = sa.norm.weight  # per head_v_dim, replicated
            parent, name = get_module(full, base + "linear_attn.out_proj")
            op = getattr(parent, name)
            gdn.out_proj = PackedToAll(
                row_slice(op, vd_l * rank, vd_l * (rank + 1), block_size), group
            )
            layer["attn"] = gdn
        else:
            att = Attention(args)
            sa = src.self_attn
            att.q_proj = packed_col(base + "self_attn.q_proj",
                                    nh_l * 2 * hd * rank, nh_l * 2 * hd * (rank + 1))
            att.k_proj = packed_col(base + "self_attn.k_proj",
                                    nkv_l * hd * rank, nkv_l * hd * (rank + 1))
            att.v_proj = packed_col(base + "self_attn.v_proj",
                                    nkv_l * hd * rank, nkv_l * hd * (rank + 1))
            att.q_norm.weight = sa.q_norm.weight
            att.k_norm.weight = sa.k_norm.weight
            parent, name = get_module(full, base + "self_attn.o_proj")
            op = getattr(parent, name)
            att.o_proj = PackedToAll(
                row_slice(op, nh_l * hd * rank, nh_l * hd * (rank + 1), block_size),
                group,
            )
            layer["attn"] = att
        mlp = MLP(hid, inter_hi - inter_lo)
        mlp.gate_proj = packed_col(base + "mlp.gate_proj", inter_lo, inter_hi)
        mlp.up_proj = packed_col(base + "mlp.up_proj", inter_lo, inter_hi)
        parent, name = get_module(full, base + "mlp.down_proj")
        dp = getattr(parent, name)
        mlp.down_proj = PackedToAll(
            row_slice(dp, inter_lo, inter_hi, block_size), group
        )
        layer["mlp"] = mlp
        layers.append(layer)

    class Shard:
        pass

    m = Shard()
    m.embed_tokens = full.model.embed_tokens  # replicated (inverse-fwht lookup)
    m.norm = full.model.norm
    m.layers = layers
    m.n_layers = n_layers
    m.interval = interval
    m.fa_idx = interval - 1
    m.ssm_idx = 0
    m.lm_head = PackedVocabHead(getattr(*get_module(full, "lm_head")), group)

    def make_cache(model):
        return [ArraysCache(size=2) if l["is_linear"] else KVCache()
                for l in model.layers]

    def __call__(model, inputs, cache):
        from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

        hid = model.embed_tokens(inputs)
        fa_m = create_attention_mask(hid, cache[model.fa_idx])
        ssm_m = create_ssm_mask(hid, cache[model.ssm_idx])
        for layer, c in zip(model.layers, cache):
            src = layer["attn"]
            h = layer["input_layernorm"](hid)
            r = src(h, mask=(ssm_m if layer["is_linear"] else fa_m), cache=c)
            hid = hid + r
            hid = hid + layer["mlp"](layer["post_attention_layernorm"](hid))
        return model.lm_head(model.norm(hid))

    Shard.__call__ = __call__
    Shard.make_cache = make_cache
    return m


def encode_prompt(tok, n_tokens):
    vocab = tok.get_vocab()
    ids = [i for i in range(min(2000, len(vocab)))]
    if not ids:
        ids = [1]
    return [ids[i % len(ids)] for i in range(n_tokens)]


def main():
    t0 = time.perf_counter()
    world = None
    if TP_N > 1:
        world = mx.distributed.init(backend="jaccl")
        rank = world.rank()
        require(world.size() == TP_N, f"world size {world.size()} != {TP_N}")
        print(f"[rank{rank}] jaccl init ok size={world.size()}", flush=True)
    else:
        rank = 0

    from plan623_pack_artifact import load_model

    full, config = load_model(str(PACK))
    load_s = time.perf_counter() - t0

    # the pack runtime dir is on sys.path now; reuse its Packed + fwht
    from runtime import Packed, fwht

    model = build_shard(full, config, rank, TP_N, world, Packed, fwht)
    build_s = time.perf_counter() - t0 - load_s
    mx.clear_cache()
    print(f"[rank{rank}] load {load_s:.1f}s shard-build {build_s:.1f}s "
          f"TP_N={TP_N} rank={rank}", flush=True)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(PACK / "tokenizer.json"))

    # ---- prefill (pp) ----
    require(TG >= 2, "TG must be >= 2 (the decode loop bodies the first token)")
    prompt = mx.array([encode_prompt(tok, PP)], dtype=mx.int32)
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    pp_s = time.perf_counter() - t0
    pp_tps = PP / pp_s

    # ---- decode (tg) ----
    token = mx.argmax(logits[:, -1, :], axis=-1)
    rows = []
    gen = [int(token.item())]
    for _ in range(TG - 1):
        t0 = time.perf_counter()
        logits = model(token[:, None], cache=cache)
        mx.eval(logits)
        dt = time.perf_counter() - t0
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        rows.append(dt * 1e3)
        gen.append(int(token.item()))
    tg_ms = statistics.median(rows)

    finite = bool(mx.all(mx.isfinite(logits[:, -1, :])).item())
    out = {
        "bench": "plan623_mlx_tp2",
        "tp_n": TP_N,
        "comm": COMM,
        "rank": rank,
        "pack": str(PACK),
        "mlx_device": str(mx.default_device()),
        "pp_tokens": PP,
        "tg_tokens": TG,
        "load_s": round(load_s, 1),
        "pp_tps": round(pp_tps, 1),
        "pp_ms_total": round(pp_s * 1e3, 1),
        "tg_ms_median": round(tg_ms, 3),
        "tg_ms_min": round(min(rows), 3),
        "tg_ms_max": round(max(rows), 3),
        "tg_tps_median": round(1000.0 / tg_ms, 1),
        "logits_finite": finite,
        "gen_head": gen[:16],
        "gen_full_sha_note": "full gen list not persisted; gen_head pins the run",
    }
    if TP_N == 1:
        out["gen_full"] = gen  # the N=1 arm doubles as the token reference
    print(json.dumps(out, indent=2), flush=True)
    dst = os.environ.get("I623_OUT")
    if dst and rank == 0:
        Path(dst).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
