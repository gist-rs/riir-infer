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
#      I1146W_BIN_B (optional): a second binary — every cell then runs A and
#      B back to back (interleaved A/B, cells named a_/b_), and G1 compares
#      the generated ids across BOTH binaries too. I1146W_REPS (default 1)
#      repeats the timed pass. I1146W_ENV_A / I1146W_ENV_B (optional,
#      space-separated KEY=VAL): per-arm env for BOTH stages — an env-armed
#      A/B from ONE binary (set I1146W_BIN_B to the same path).
set -u

TS=$(date +%Y%m%dT%H%M%SZ)
OUT="${I1146W_OUT:-/tmp/i1146w/$TS}"
CELLS="$OUT/cells"
mkdir -p "$CELLS"
LOG="$OUT/session.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BIN_A="${I1146W_BIN:?set I1146W_BIN}"
BIN_B="${I1146W_BIN_B:-}"
REPS="${I1146W_REPS:-1}"
MODEL="${I1146W_MODEL:?set I1146W_MODEL}"
WIDTHS="${I1146W_WIDTHS:-128 256 512 1024}"
# `-` not `:-`: an EMPTY I1146W_LENS means G1 cells only (no timing).
LENS="${I1146W_LENS-512 4096 32768}"
for b in $BIN_A $BIN_B; do [ -e "$b" ] || { say "REFUSED: missing $b"; exit 2; }; done
ARMS="a"; [ -n "$BIN_B" ] && ARMS="a b"
bin_of() { case "$1" in a) echo "$BIN_A" ;; b) echo "$BIN_B" ;; esac; }
env_of() { case "$1" in a) echo "${I1146W_ENV_A:-}" ;; b) echo "${I1146W_ENV_B:-}" ;; esac; }
[ -e "$MODEL" ] || { say "REFUSED: missing $MODEL"; exit 2; }

say "=== i1146 width re-curve start (ts $TS) ==="
{ sysctl -n hw.model hw.ncpu hw.memsize; uptime; pmset -g batt | head -1; } 2>/dev/null | tee -a "$LOG"
# Foreign GPU users: everything matching the GPU pattern except this rig and
# its own binaries (perf_rematch.sh's idle wrapper is deliberately NOT a
# match — only the processes that actually hold the GPU are).
foreign_gpu() {
  ps aux | grep -Ei 'llama|mlx|dist_stage|ggml|bench_' | grep -v grep | grep -v i1146_ \
    | grep -vF "$BIN_A" | { [ -n "$BIN_B" ] && grep -vF "$BIN_B" || cat; } || true
}
QUIET=$(foreign_gpu)
if [ -n "$QUIET" ] && [ "${I1146W_FORCE:-0}" != "1" ]; then
  say "REFUSED: GPU not quiet:"; echo "$QUIET" | tee -a "$LOG"; exit 2
fi
{
  for x in $ARMS; do b=$(bin_of $x); echo "bin_$x: $b"; echo "env_$x: $(env_of $x)"; /usr/bin/shasum -a 256 "$b" | awk -v x="$x" '{print "bin_" x "_sha256:", $1}'; done
  echo "model: $MODEL"; /usr/bin/shasum -a 256 "$MODEL" | awk '{print "model_sha256:", $1}'
  echo "widths: $WIDTHS"; echo "lens: $LENS"
} > "$OUT/provenance.txt"

PORT=17643
export RIIR_DIST_KEY=$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')

run_cell() { # $1 arm $2 name $3 ids $4 max-new $5 ctx $6+ extra
  local BIN ARM_ENV; BIN=$(bin_of "$1"); ARM_ENV=$(env_of "$1"); shift
  local name="$1" ids="$2" n="$3" ctx="$4"; shift 4
  # shellcheck disable=SC2086 # ARM_ENV is a word-split KEY=VAL list
  env $ARM_ENV "$BIN" --role tail --model "$MODEL" --split 32 --backend gpu \
    --listen "127.0.0.1:$PORT" --ctx "$ctx" > "$CELLS/${name}_tail.log" 2>&1 &
  local tail_pid=$! i=0
  while [ $i -lt 240 ]; do
    kill -0 $tail_pid 2>/dev/null || { say "tail died at boot ($name)"; break; }
    grep -q "listening on" "$CELLS/${name}_tail.log" 2>/dev/null && break
    i=$((i+1)); sleep 1
  done
  # shellcheck disable=SC2086
  env $ARM_ENV "$BIN" --role head --model "$MODEL" --split 32 --backend gpu \
    --connect "127.0.0.1:$PORT" --prompt "$ids" --max-new "$n" --no-eos --ctx "$ctx" "$@" \
    > "$CELLS/${name}_head.log" 2>&1
  local rc=$?
  wait $tail_pid 2>/dev/null
  local pf
  pf=$(grep -o 'prefill [0-9]* tok in [0-9.]*s' "$CELLS/${name}_head.log" | tail -1)
  say "$name rc=$rc :: $pf"
}

# Per-pair contention guard: a 90 s quiet window at step start cannot protect
# an ~8-min A/B from a foreign GPU job that starts mid-run (measured: a league
# bench_809 row landed 30 s in and inflated every later cell 4-10x). Each timed
# a/b PAIR waits for GPU-quiet, a sampler watches it, and a pair a foreign GPU
# process touched is logged CONTAM and re-run (I1146W_RETRIES, default 3).
# I1146W_FORCE=1 disables the guard (G1-only runs).
timed_pair() { # $1 cell-name-suffix $2 ids $3 ctx $4+ extra
  local suffix="$1" ids="$2" ctx="$3"; shift 3
  local try=1 flag="$OUT/.contam"
  while :; do
    if [ "${I1146W_FORCE:-0}" != "1" ]; then
      until [ -z "$(foreign_gpu)" ]; do sleep 15; done
    fi
    rm -f "$flag"
    ( while :; do [ -n "$(foreign_gpu)" ] && touch "$flag"; sleep 3; done ) &
    local sampler=$!
    for x in $ARMS; do run_cell "$x" "${x}_${suffix}" "$ids" 1 "$ctx" "$@"; done
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    if [ ! -e "$flag" ] || [ "${I1146W_FORCE:-0}" = "1" ]; then return 0; fi
    say "CONTAM ${suffix} (try $try) — foreign GPU process during the pair; re-running"
    for x in $ARMS; do mv -f "$CELLS/${x}_${suffix}_head.log" "$CELLS/${x}_${suffix}_contam${try}_head.log.x" 2>/dev/null; done
    try=$((try+1))
    [ $try -gt "${I1146W_RETRIES:-3}" ] && { say "CONTAM ${suffix}: retries exhausted — pair DISCARDED"; for x in $ARMS; do rm -f "$CELLS/${x}_${suffix}_head.log"; done; return 1; }
  done
}

ids_of() { python3 -c "print(','.join(str(7+(i*7919)%50000) for i in range($1)))"; }

# ── G1: pp1024 + 16 decode tokens per width ──
IDS=$(ids_of 1024)
for W in $WIDTHS; do
  for x in $ARMS; do run_cell "$x" "${x}_g1_w${W}" "$IDS" 16 1280 --prefill-chunk "$W"; done
done
REF=""
G1=PASS
for W in $WIDTHS; do
  for x in $ARMS; do
    GEN=$(grep -o 'generated: \[.*\]' "$CELLS/${x}_g1_w${W}_head.log" | head -1)
    say "g1 $x w=$W $GEN"
    [ -z "$GEN" ] && G1=FAIL
    [ -z "$REF" ] && REF="$GEN"
    [ "$GEN" != "$REF" ] && G1=FAIL
  done
done
say "G1 CROSS-WIDTH $G1"

# ── Timing: width x length, interleaved per length (warm + timed) ──
for P in $LENS; do
  IDS=$(ids_of "$P")
  for W in $WIDTHS; do
    [ "$W" -gt "$P" ] && continue
    for x in $ARMS; do run_cell "$x" "${x}_pp${P}_w${W}_warm" "$IDS" 1 $((P + 256)) --prefill-chunk "$W"; done
    r=1
    while [ $r -le "$REPS" ]; do
      timed_pair "pp${P}_w${W}_r${r}" "$IDS" $((P + 256)) --prefill-chunk "$W"
      r=$((r+1))
    done
  done
done

say "=== summary (timed cells) ==="
for P in $LENS; do
  for W in $WIDTHS; do
    for f in "$CELLS"/?_pp${P}_w${W}_r*_head.log; do
      [ -e "$f" ] || continue
      s=$(grep -o 'prefill [0-9]* tok in [0-9.]*s' "$f" | tail -1 | awk '{print $5}' | tr -d s)
      [ -n "$s" ] && say "$(basename "$f" _head.log) ${s}s $(python3 -c "print(f'{$P/$s:.1f} tok/s')")"
    done
  done
done
say "=== COMPLETE: $OUT ==="
