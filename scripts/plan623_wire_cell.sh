#!/bin/bash
# Plan 623 — the 2-box wire confirmation cell rig (Bench 913).
# amo(head, this box) ⇄ bmo(tail, over ssh) on the TB5 link. ABBA over
# RIIR_DIST_PAIR (1 = armed fused batch, 0/unset = the incumbent single-drain
# cadence — the serving default since the Bench-913 refutation). Every cell
# restarts the tail (it exits after one session); readiness = poll the TAIL's
# log on bmo for "listening on" — never a TCP probe (a probe consumes the
# tail's single accept and EOFs the handshake; measured 02:18 session trap).
#
# Usage:
#   scripts/plan623_wire_cell.sh abba           # ABBA ×2 at s=2
#   scripts/plan623_wire_cell.sh g1             # incumbent arm + --full-check
#   scripts/plan623_wire_cell.sh one <name> <pair> [streams] [extra-head-flags]
set -u
BMO=${I623_BMO:-katopz@192.168.10.2}
BMO_BIN=${I623_BMO_BIN:-/Users/katopz/git/riir-infer/target/release/dist_stage}
AMO_BIN=${I623_AMO_BIN:-/Users/katopz/git/riir-infer/target/release/dist_stage}
BMO_MODEL=${I623_BMO_MODEL:-/Users/katopz/models/Ternary-Bonsai-2-27B-PQ2_0.gguf}
AMO_MODEL=${I623_AMO_MODEL:-/Users/katopz/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf}
KEY=$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')
PORT=${I623_PORT:-17632}
OUT_R=/tmp/i623_wire_cells                      # tail logs live ON bmo
OUT_L="$(pwd)/.wt/i623_wire_cell/cells"         # head logs live here
mkdir -p "$OUT_L" && ssh "$BMO" "mkdir -p $OUT_R"
IDS4="7,13,42,100"
SPLIT=32
CTX=4096
MAXNEW=32
# Optional env passthrough (T3.5): I623_HEAD_ENV / I623_TAIL_ENV land verbatim
# in the head's / tail's environment (e.g. RIIR_DIST_STEP_TIMING=1); the
# per-cell extra head flags arrive as ONE quoted string, eval-split here so
# multi-flag extras (e.g. "--prefill-chunk 512 --prompt 1,2,3 --ctx 4096")
# parse as separate argv (last flag wins in the hand-rolled parser).

# $1 name  $2 pair(0|1)  $3 streams  $4+ extra head flags (e.g. --full-check)
run_cell() {
  local name="$1" pair="$2" streams="$3"; shift 3
  eval "extra=($*)" 2>/dev/null || extra=()
  echo "=== cell $name (pair=$pair streams=$streams) $(date +%H:%M:%S) ==="
  ssh "$BMO" "${I623_TAIL_ENV:-} RIIR_DIST_KEY='$KEY' '$BMO_BIN' --role tail \
      --model '$BMO_MODEL' --split $SPLIT --backend gpu --streams $streams \
      --listen 192.168.10.2:$PORT --ctx $CTX > $OUT_R/${name}_tail.log 2>&1" &
  local tail_ssh=$!
  local i=0
  while [ $i -lt 300 ]; do
    ssh "$BMO" "grep -q 'listening on' $OUT_R/${name}_tail.log 2>/dev/null" && break
    kill -0 $tail_ssh 2>/dev/null || { echo "tail died at boot:"; ssh "$BMO" "tail -5 $OUT_R/${name}_tail.log"; return 1; }
    i=$((i+1)); sleep 2
  done
  [ $i -ge 300 ] && { echo "tail never listened"; return 1; }
  [ "$pair" = "1" ] && env_prefix="RIIR_DIST_PAIR=1"
  env $env_prefix RIIR_DIST_KEY="$KEY" ${I623_HEAD_ENV:-} "$AMO_BIN" --role head \
      --model "$AMO_MODEL" --split $SPLIT --backend gpu --streams $streams \
      --connect "192.168.10.2:$PORT" --prompt "$IDS4" --max-new $MAXNEW --no-eos \
      --ctx $CTX "${extra[@]}" > "$OUT_L/${name}_head.log" 2>&1
  local rc=$?
  wait $tail_ssh 2>/dev/null
  echo "cell $name: head rc=$rc $(date +%H:%M:%S)"
  grep -E "^stream|aggregate|G1|oracle" "$OUT_L/${name}_head.log" | tail -8
  return $rc
}

case "${1:-}" in
  abba)
    run_cell ab_p1_a 1 "${3:-2}" "${4:-}"; run_cell ab_p0_b 0 "${3:-2}" "${4:-}"
    run_cell ab_p1_b 1 "${3:-2}" "${4:-}"; run_cell ab_p0_a 0 "${3:-2}" "${4:-}" ;;
  g1)
    run_cell g1_p0 0 "${3:-2}" "${4:-} --full-check" ;;
  one)
    run_cell "$2" "$3" "${4:-2}" "${5:-}" ;;
  *)
    echo "usage: $0 abba | g1 | one <name> <pair(0|1)> [streams] [extra-head-flags]" ;;
esac
