#!/usr/bin/env bash
# i1146_width_2box.sh — Issue 1146 rung 1, the 2-box leg: the prefill width
# curve on the real wire (amo head ⇄ bmo tail over TB5, --split 32). The
# 128 default was the Bench-909 2-box optimum through the OLD sequential
# posture; wider frames trade per-frame head⇄tail overlap for wider GEMMs,
# so the 1-box curve (i1146_width_recurve.sh) alone cannot flip the default.
#
# Run on the HEAD box (amo). The tail binary must already sit at
# $I1146W_TAIL_BIN on the tail host. Cells interleaved ABAB per length.
#
#   I1146W_BIN=/tmp/i1146w-tgt/release/dist_stage \
#   I1146W_MODEL=../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
#     bash scripts/i1146_width_2box.sh
# Env: I1146W_TAIL_HOST (katopz@100.100.117.44), I1146W_TAIL_ADDR
#      (192.168.10.2, the TB5 bridge), I1146W_TAIL_BIN (/tmp/dist_stage),
#      I1146W_TAIL_MODEL (/Users/katopz/models/Ternary-Bonsai-2-27B-PQ2_0.gguf),
#      I1146W_WIDTHS ("128 512"), I1146W_LENS ("512 4096"), I1146W_REPS (2).
set -u

TS=$(date +%Y%m%dT%H%M%SZ)
OUT="${I1146W_OUT:-/tmp/i1146w2/$TS}"
CELLS="$OUT/cells"
mkdir -p "$CELLS"
LOG="$OUT/session.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BIN="${I1146W_BIN:?set I1146W_BIN}"
MODEL="${I1146W_MODEL:?set I1146W_MODEL}"
TH="${I1146W_TAIL_HOST:-katopz@100.100.117.44}"
TA="${I1146W_TAIL_ADDR:-192.168.10.2}"
TBIN="${I1146W_TAIL_BIN:-/tmp/dist_stage}"
TMODEL="${I1146W_TAIL_MODEL:-/Users/katopz/models/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
WIDTHS="${I1146W_WIDTHS:-128 512}"
LENS="${I1146W_LENS:-512 4096}"
REPS="${I1146W_REPS:-2}"
PORT=17645
KEY=$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')
export RIIR_DIST_KEY="$KEY"

QUIET=$(ps aux | grep -Ei 'llama|mlx|dist_stage|ggml|bench_' | grep -v grep | grep -v i1146_width || true)
if [ -n "$QUIET" ] && [ "${I1146W_FORCE:-0}" != "1" ]; then
  say "REFUSED: head GPU not quiet:"; echo "$QUIET" | tee -a "$LOG"; exit 2
fi
say "=== i1146 2-box width cells (ts $TS) head=$(hostname -s) tail=$TH@$TA ==="
uptime | tee -a "$LOG"; ssh "$TH" uptime | tee -a "$LOG"

run_cell() { # $1 name $2 ids $3 ctx $4 width
  local name="$1" ids="$2" ctx="$3" w="$4"
  ssh "$TH" "(RIIR_DIST_KEY=$KEY nohup $TBIN --role tail --model $TMODEL --split 32 --backend gpu \
    --listen 0.0.0.0:$PORT --ctx $ctx > /tmp/i1146w2_tail.log 2>&1 &)"
  local i=0
  while [ $i -lt 240 ]; do
    ssh "$TH" "grep -q 'listening on' /tmp/i1146w2_tail.log" && break
    i=$((i+1)); sleep 1
  done
  "$BIN" --role head --model "$MODEL" --split 32 --backend gpu \
    --connect "$TA:$PORT" --prompt "$ids" --max-new 1 --no-eos --ctx "$ctx" \
    --prefill-chunk "$w" > "$CELLS/${name}_head.log" 2>&1
  local rc=$?
  ssh "$TH" "while pgrep -f 'role tail' >/dev/null; do sleep 1; done"
  say "$name rc=$rc :: $(grep -o 'prefill [0-9]* tok in [0-9.]*s' "$CELLS/${name}_head.log" | tail -1)"
}

ids_of() { python3 -c "print(','.join(str(7+(i*7919)%50000) for i in range($1)))"; }

for P in $LENS; do
  IDS=$(ids_of "$P")
  for W in $WIDTHS; do run_cell "pp${P}_w${W}_warm" "$IDS" $((P + 256)) "$W"; done
  r=1
  while [ $r -le "$REPS" ]; do
    for W in $WIDTHS; do run_cell "pp${P}_w${W}_r${r}" "$IDS" $((P + 256)) "$W"; done
    r=$((r+1))
  done
done
say "=== COMPLETE: $OUT ==="
