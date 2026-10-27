#!/usr/bin/env python3
"""Plan 623 T0.4 — 2-box PP over the Bonsai-2 2-bit pack (socket exchange).

Layer-split pipeline parallelism over the pack's bundled runtime. The
per-token exchange rides ONE direct TCP socket (rank0/amo connects to
rank1/bmo — amo's application firewall blocks inbound to unsigned python, so
the connect direction amo->bmo is the lawful one; bmo has no firewall).
mx.distributed ring p2p (send/recv) was measured DEADLOCKED on this exact
2-node topology in mlx 0.32.0 (posted send+recv, zero bytes cross the wire;
see the bench record) — hence the plain socket, which is also the honest
minimal-incumbent shape: mlx runtime + a TCP hop, mirroring llama.cpp-RPC's
TCP per-step round trip for apples-to-apples.

  rank 0 (amo): embed + layers[0:k]  --hidden-->  rank 1 (bmo)
  rank 1:       layers[k:n] + norm + head          --token-->  rank 0

One socket, two message types per decode step (hidden amo->bmo, token
bmo->amo); prefill is one bulk hidden message. Each rank owns ITS layers'
caches only — recurrent state never crosses the wire (same shape as
riir-infer's own ActivationFrame lane).

Env (both ranks): I623_PACK, I623_SPLIT (32), I623_PP (512), I623_TG (128),
I623_PORT (50201), I623_OUT. Greedy sampling on rank 1 (deterministic).
"""

import json
import os
import socket
import statistics
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

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
SPLIT = int(os.environ.get("I623_SPLIT", "32"))
PP = int(os.environ.get("I623_PP", "512"))
TG = int(os.environ.get("I623_TG", "128"))
PORT = int(os.environ.get("I623_PORT", "50201"))
BMO_IP = os.environ.get("I623_BMO_IP", "192.168.10.2")
HDR = 12  # magic u16 | msg_type u8 | reserved u8 | nbytes u32 | token u32


def _xfer(sock: socket.socket, data: bytes | None, want: int) -> bytes:
    """one bidirectional exchange: send `data` (may be None), recv `want` bytes"""
    if data is not None:
        sock.sendall(data)
    buf = bytearray()
    while len(buf) < want:
        chunk = sock.recv(want - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def send_hidden(sock, hid: mx.array):
    bits = mx.view(hid.astype(mx.bfloat16), mx.uint16)  # reinterpret bf16 as u16
    payload = np.asarray(bits).tobytes()
    sock.sendall(struct.pack("<HII", 0x6D48, len(payload), 0) + payload)


def recv_hidden(sock, hid_dim: int, seq: int) -> mx.array:
    hdr = _xfer(sock, None, 10)
    magic, nbytes, _rsv = struct.unpack("<HII", hdr)
    assert magic == 0x6D48, f"bad magic {magic:#x}"
    raw = _xfer(sock, None, nbytes)
    arr = mx.array(np.frombuffer(raw, dtype=np.uint16))
    arr = mx.view(arr, mx.bfloat16)
    return arr.reshape(1, seq, hid_dim)


def send_token(sock, token: int):
    sock.sendall(struct.pack("<I", token))


def recv_token(sock) -> int:
    return struct.unpack("<I", _xfer(sock, None, 4))[0]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plan623_pack_artifact import load_model

    t0 = time.perf_counter()
    model, config = load_model(str(PACK))
    tc = config["text_config"]
    n_layers = tc["num_hidden_layers"]
    interval = tc.get("full_attention_interval", 4)
    hid_dim = tc["hidden_size"]
    tied = tc.get("tie_word_embeddings", False)

    # rank by role: RANK0 = the box that CONNECTS (amo), RANK1 = the LISTENER (bmo)
    role = int(os.environ.get("I623_ROLE", "0"))
    lo, hi = (0, SPLIT) if role == 0 else (SPLIT, n_layers)
    global_idxs = list(range(lo, hi))
    model.model.layers = model.model.layers[lo:hi]
    caches = model.make_cache()
    fa_local = next(
        (i for i, g in enumerate(global_idxs) if (g + 1) % interval == 0), None
    )
    print(
        f"[role{role}] layers[{lo}:{hi}] loaded {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(PACK / "tokenizer.json"))
    vocab_n = min(2000, tok.get_vocab_size())

    def run_half(hid):
        ssm_m = create_ssm_mask(hid, caches[0])
        fa_m = (
            create_attention_mask(hid, caches[fa_local])
            if fa_local is not None
            else None
        )
        for layer, c, g in zip(model.model.layers, caches, global_idxs):
            mask = ssm_m if (g + 1) % interval != 0 else fa_m
            hid = layer(hid, mask=mask, cache=c)
        return hid

    def head(hid):
        h = model.model.norm(hid)
        return model.lm_head(h) if not tied else model.model.embed_tokens.as_linear(h)

    sock = None
    if role == 1:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((BMO_IP, PORT))
        srv.listen(1)
        print(f"[role1] listening on {BMO_IP}:{PORT}", flush=True)
        sock, peer = srv.accept()
        print(f"[role1] peer {peer}", flush=True)
    else:
        for attempt in range(30):
            try:
                sock = socket.socket()
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((BMO_IP, PORT))
                break
            except OSError:
                time.sleep(1)
        else:
            sys.exit("rank0 could not connect to bmo")
        print(f"[role0] connected {BMO_IP}:{PORT}", flush=True)

    if role == 0:
        # ---- prefill: embed + first half, bulk hidden hop ----
        prompt = mx.array([[i % vocab_n for i in range(PP)]], dtype=mx.int32)
        t0 = time.perf_counter()
        hid = model.model.embed_tokens(prompt)
        hid = run_half(hid)
        mx.eval(hid)
        send_hidden(sock, hid)
        pp_s = time.perf_counter() - t0

        # ---- decode: TG x (recv token -> embed -> half -> send hidden) ----
        rows = []
        for _ in range(TG):
            t0 = time.perf_counter()
            t = recv_token(sock)
            tok_arr = mx.array([[t]], dtype=mx.int32)
            hid = model.model.embed_tokens(tok_arr)
            hid = run_half(hid)
            mx.eval(hid)
            send_hidden(sock, hid)
            rows.append((time.perf_counter() - t0) * 1e3)
        out = {
            "role": 0,
            "split": SPLIT,
            "pp_s": round(pp_s, 3),
            "tg_ms_median": round(statistics.median(rows), 3),
            "tg_ms_min": round(min(rows), 3),
            "tg_ms_max": round(max(rows), 3),
        }
        print(json.dumps(out), flush=True)
    else:
        # ---- prefill: bulk hop -> second half + head -> token back ----
        t0 = time.perf_counter()
        hid = recv_hidden(sock, hid_dim, PP)
        hid = run_half(hid)
        logits = head(hid)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        first = int(token.item())
        send_token(sock, first)
        pp_s = time.perf_counter() - t0

        # ---- decode: TG x (recv hidden -> half + head -> token back) ----
        rows = []
        gen = [first]
        for _ in range(TG - 1):
            t0 = time.perf_counter()
            hid = recv_hidden(sock, hid_dim, 1)
            hid = run_half(hid)
            logits = head(hid)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tok_i = int(token.item())
            send_token(sock, tok_i)
            rows.append((time.perf_counter() - t0) * 1e3)
            gen.append(tok_i)
        out = {
            "role": 1,
            "split": SPLIT,
            "pp_s": round(pp_s, 3),
            "pp_tps": round(PP / pp_s, 1),
            "tg_tps_median": round(1000.0 / statistics.median(rows), 1),
            "tg_ms_median": round(statistics.median(rows), 3),
            "tg_ms_min": round(min(rows), 3),
            "tg_ms_max": round(max(rows), 3),
            "gen_head": gen[:8],
        }
        print(json.dumps(out), flush=True)
        dst = os.environ.get("I623_OUT")
        if dst:
            Path(dst).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
