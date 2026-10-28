#!/usr/bin/env bash
# i1146_width_recurve.sh — Issue 1146 ladder rung 1: the prefill chunk-width
# re-curve through the batched/chunked posture (Bench 914 read 1.55x/token at
# 512 vs 128 on the 1-box census rig; Bench 909's "128 optimal" was measured
# through the OLD sequential posture).
#
# ONE dist_stage binary, ONE box, the Bench-911 cell rig (loopback, --split 32,
# both stages --ctx P+256). Widths x prompt lengths, each cell warm + timed.
# G1: a pp1024 decode cell per width (--max-new 16) — the generated ids must be
# IDENTICAL across widths (width moves frame boundaries only; every chunked
# body is bit-identity-gated against its sequential twin).
#
# Usage (on the box with the binary + model):
#   I1146W_BIN=/tmp/dist_stage I1146W_MODEL=/path/model.gguf \
#     bash i1146_width_recurve.sh
# Env: I1146W_WIDTHS (default "128 256 512 1024"), I1146W_LENS (default
#      "512 4096 32768"), I1146W_OUT (default /tmp/i1146w/<ts>).
set -u

TS=$(date +%Y%m%dT%H%M%SZ)
OUT="${I1146W_OUT:-/tmp/i1146w/$TS}"
CELLS="$OUT/cells"
mkdir -p "$CELLS"
LOG="$OUT/session.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BIN="${I1146W_BIN:?set I1146W_BIN}"
MODEL="${I1146W_MODEL:?set I1146W_MODEL}"
WIDTHS="${I1146W_WIDTHS:-128 256 512 1024}"
LENS="${I1146W_LENS:-512 4096 32768}"
[ -e "$BIN" ] || { say "REFUSED: missing $BIN"; exit 2; }
[ -e "$MODEL" ] || { say "REFUSED: missing $MODEL"; exit 2; }

say "=== i1146 width re-curve start (ts $TS) ==="
{ sysctl -n hw.model hw.ncpu hw.memsize; uptime; pmset -g batt | head -1; } 2>/dev/null | tee -a "$LOG"
QUIET=$(ps aux | grep -Ei 'llama|mlx|dist_stage|ggml|bench_' | grep -v grep | grep -v i1146_width || true)
if [ -n "$QUIET" ] && [ "${I1146W_FORCE:-0}" != "1" ]; then
  say "REFUSED: GPU not quiet:"; echo "$QUIET" | tee -a "$LOG"; exit 2
fi
{
  echo "bin: $BIN"; /usr/bin/shasum -a 256 "$BIN" | awk '{print "bin_sha256:", $1}'
  echo "model: $MODEL"; /usr/bin/shasum -a 256 "$MODEL" | awk '{print "model_sha256:", $1}'
  echo "widths: $WIDTHS"; echo "lens: $LENS"
} > "$OUT/provenance.txt"

PORT=17643
export RIIR_DIST_KEY=$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')

run_cell() { # $1 name $2 ids $3 max-new $4 ctx $5+ extra
  local name="$1" ids="$2" n="$3" ctx="$4"; shift 4
  "$BIN" --role tail --model "$MODEL" --split 32 --backend gpu \
    --listen "127.0.0.1:$PORT" --ctx "$ctx" > "$CELLS/${name}_tail.log" 2>&1 &
  local tail_pid=$! i=0
  while [ $i -lt 240 ]; do
    kill -0 $tail_pid 2>/dev/null || { say "tail died at boot ($name)"; break; }
    grep -q "listening on" "$CELLS/${name}_tail.log" 2>/dev/null && break
    i=$((i+1)); sleep 1
  done
  "$BIN" --role head --model "$MODEL" --split 32 --backend gpu \
    --connect "127.0.0.1:$PORT" --prompt "$ids" --max-new "$n" --no-eos --ctx "$ctx" "$@" \
    > "$CELLS/${name}_head.log" 2>&1
  local rc=$?
  wait $tail_pid 2>/dev/null
  local pf
  pf=$(grep -o 'prefill [0-9]* tok in [0-9.]*s' "$CELLS/${name}_head.log" | tail -1)
  say "$name rc=$rc :: $pf"
}

ids_of() { python3 -c "print(','.join(str(7+(i*7919)%50000) for i in range($1)))"; }

# ── G1: pp1024 + 16 decode tokens per width ──
IDS=$(ids_of 1024)
for W in $WIDTHS; do
  run_cell "g1_w${W}" "$IDS" 16 1280 --prefill-chunk "$W"
done
REF=""
G1=PASS
for W in $WIDTHS; do
  GEN=$(grep -o 'generated: \[.*\]' "$CELLS/g1_w${W}_head.log" | head -1)
  say "g1 w=$W $GEN"
  [ -z "$GEN" ] && G1=FAIL
  [ -z "$REF" ] && REF="$GEN"
  [ "$GEN" != "$REF" ] && G1=FAIL
done
say "G1 CROSS-WIDTH $G1"

# ── Timing: width x length, interleaved per length (warm + timed) ──
for P in $LENS; do
  IDS=$(ids_of "$P")
  for W in $WIDTHS; do
    [ "$W" -gt "$P" ] && continue
    run_cell "pp${P}_w${W}_warm" "$IDS" 1 $((P + 256)) --prefill-chunk "$W"
    run_cell "pp${P}_w${W}" "$IDS" 1 $((P + 256)) --prefill-chunk "$W"
  done
done

say "=== summary (timed cells) ==="
for P in $LENS; do
  for W in $WIDTHS; do
    f="$CELLS/pp${P}_w${W}_head.log"
    [ -e "$f" ] || continue
    s=$(grep -o 'prefill [0-9]* tok in [0-9.]*s' "$f" | tail -1 | awk '{print $5}' | tr -d s)
    [ -n "$s" ] && say "pp$P w$W ${s}s $(python3 -c "print(f'{$P/$s:.1f} tok/s')")"
  done
done
say "=== COMPLETE: $OUT ==="
