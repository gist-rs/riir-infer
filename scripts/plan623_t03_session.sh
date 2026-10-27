#!/usr/bin/env bash
# plan623_t03_session.sh — Plan 623 T0.3: the ONE-SESSION 1-box arms on idle bmo.
#
# Three engines x {tg128, pp512, pp4096, pp32768}, one box (bmo, M5 Ultra,
# GPU-quiet gate enforced below), one log:
#   1. llama.cpp  — llama-bench (prism-b10787 bundle shipped for Bench 900)
#   2. MLX        — plan623_mlx_1box_bench.py (Bench 904 venv; pack = 2-bit
#                   affine+hadamard — quant posture differs from PQ2_0, disclosed)
#   3. riir-infer — dist_stage (gpu, bonsai2_hadamard) loopback pipeline +
#                   --full-check oracle (the no-wire 1-box read)
#
# An EXPECTED refusal (MLX pp32768 hits the Metal single-buffer cap) must not
# kill the session: no set -e. Every cell logs separately under $OUT/cells/.
# The caller ships $OUT back to amo; the bench md is assembled there.
#
# Usage (on bmo):  bash plan623_t03_session.sh
#                  results in ~/t03/<ts>/
set -u

TS=$(date +%Y%m%dT%H%M%SZ)
OUT="$HOME/t03/$TS"
CELLS="$OUT/cells"
mkdir -p "$CELLS"
LOG="$OUT/session.log"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ── 0. box state + the GPU-quiet gate ────────────────────────────────
say "=== T0.3 session start (ts $TS) ==="
sysctl -n hw.model machdep.cpu.brand_string hw.ncpu hw.memsize 2>/dev/null
sw_vers
uptime
# GPU-quiet: refuse to start if a compute consumer is already running.
QUIET=$(ps aux | grep -Ei 'llama|mlx|dist_stage|llama-bench|ggml' | grep -v grep | grep -v plan623_t03 || true)
if [ -n "$QUIET" ]; then
  say "REFUSED: GPU not quiet:"; echo "$QUIET"; exit 2
fi
say "GPU quiet OK"

MODEL="$HOME/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
LB="$HOME/git/prismml-llama.cpp/build-rpc/bin/llama-bench"
PYV="/Users/katopz/git/riir-infer/.raw/mlx-env/bin/python"
SCRIPTS="/Users/katopz/git/riir-infer/scripts"
BIN="$HOME/bin/dist_stage_t03"

for f in "$MODEL" "$LB" "$PYV" "$BIN" "$SCRIPTS/plan623_mlx_1box_bench.py"; do
  [ -e "$f" ] || { say "REFUSED: missing $f"; exit 2; }
done

say "model sha256 (provenance, ~15s):"
ssha=$(/usr/bin/shasum -a 256 "$MODEL" | awk '{print $1}')
echo "model_sha256: $ssha" | tee "$OUT/provenance.txt"
echo "dist_stage_sha256: $(/usr/bin/shasum -a 256 "$BIN" | awk '{print $1}')" >> "$OUT/provenance.txt"
"$LB" --version 2>/dev/null | tail -1 >> "$OUT/provenance.txt" || true
"$PYV" -c "import mlx, mlx_lm; print('mlx', mlx.__version__, '| mlx_lm', mlx_lm.__version__)" >> "$OUT/provenance.txt" 2>&1 || true

# ── 1. llama.cpp rows ────────────────────────────────────────────
if [ "${T03_SKIP_LLAMACPP:-0}" != "1" ]; then
say "=== llama.cpp: pp rows (512/4096/32768) ==="
"$LB" -m "$MODEL" -ngl 99 -t 8 -p 512,4096,32768 -n 0 2>&1 | tee "$CELLS/llamacpp_pp.md"
say "=== llama.cpp: tg128 row ==="
"$LB" -m "$MODEL" -ngl 99 -t 8 -p 0 -n 128 2>&1 | tee "$CELLS/llamacpp_tg.md"
fi

# ── 2. MLX rows (each invocation = one pp cell + tg128; pp32768 expected REFUSED) ──
if [ "${T03_SKIP_MLX:-0}" != "1" ]; then
cd "$SCRIPTS" || exit 2
for PP in 512 4096 32768; do
  say "=== MLX: pp$PP + tg128 ==="
  I623_PP="$PP" I623_TG=128 I623_OUT="$OUT/cells/mlx_pp${PP}.json" \
    "$PYV" plan623_mlx_1box_bench.py 2>&1 | tee "$CELLS/mlx_pp${PP}.log"
  say "MLX pp$PP exit: ${PIPESTATUS[0]}"
done
fi

# ── 3. riir-infer rows (dist_stage loopback; tail exits after one head session) ──
# T03_CELLS="c1,c2" (comma list) runs ONLY those cells — the rerun lane for
# a failed cell without redoing the whole phase.
KEY=$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')
export RIIR_DIST_KEY="$KEY"
PORT=17631
IDS4="7,13,42,100"

has_cell() { # $1 cell name — is it selected?
  [ "${T03_CELLS:-}" = "" ] && return 0
  local w
  for w in $(echo "$T03_CELLS" | tr ',' ' '); do [ "$w" = "$1" ] && return 0; done
  return 1
}

run_cell() { # $1 name  $2 prompt-ids  $3 max-new  $4 ctx  $5+ extra flags
  local name="$1" ids="$2" n="$3" ctx="$4"; shift 4
  say "=== riir-infer: $name ==="
  # BOTH stages carry --ctx: the tail's span gate checks the activation span
  # against ITS OWN stage capacity (measured: a head-side ctx alone left the
  # tail at default 4096 and frame offset 4096 refused mid-prefill).
  "$BIN" --role tail --model "$MODEL" --split 32 --backend gpu \
    --listen "127.0.0.1:$PORT" --ctx "$ctx" > "$CELLS/${name}_tail.log" 2>&1 &
  local tail_pid=$!
  local i=0
  # Poll the tail's OWN log for the listen line — a TCP probe (nc -z) would
  # consume the tail's single accept and EOF its handshake (measured 02:18
  # session: every cell died "failed to fill whole buffer" to the probe).
  while [ $i -lt 240 ]; do
    if ! kill -0 $tail_pid 2>/dev/null; then say "tail died at boot"; break; fi
    if grep -q "listening on" "$CELLS/${name}_tail.log" 2>/dev/null; then break; fi
    i=$((i+1)); sleep 1
  done
  "$BIN" --role head --model "$MODEL" --split 32 --backend gpu \
    --connect "127.0.0.1:$PORT" --prompt "$ids" --max-new "$n" --no-eos --ctx "$ctx" "$@" \
    2>&1 | tee "$CELLS/${name}_head.log"
  local head_rc=${PIPESTATUS[0]}
  wait $tail_pid 2>/dev/null
  say "riir-infer $name head_rc=$head_rc"
}

# warmup (Metal pipeline compile, both kernel sets) — discarded
if has_cell warmup_decode; then run_cell t03_warmup_decode "$IDS4" 4 4096; fi
# decode cell: pipeline per-stream + the --full-check 1-box oracle + G1
if has_cell decode; then run_cell t03_decode "$IDS4" 128 4096 --full-check; fi
# prefill cells (chunked, width 128 = the shipped serving width);
# each cell runs TWICE — first pass compiles the batched-prefill set, second is measured
for P in 512 4096 32768; do
  if ! has_cell "pp${P}" && ! has_cell "pp${P}_warm"; then continue; fi
  IDS=$(jot -s, "$P" 7)
  if has_cell "pp${P}_warm"; then run_cell "t03_pp${P}_warm" "$IDS" 1 $((P + 256)) --prefill-chunk 128; fi
  if has_cell "pp${P}"; then run_cell "t03_pp${P}" "$IDS" 1 $((P + 256)) --prefill-chunk 128; fi
done

say "=== T0.3 session COMPLETE: $OUT ==="
