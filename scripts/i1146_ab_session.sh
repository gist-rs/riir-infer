#!/usr/bin/env bash
# i1146_ab_session.sh — Issue 1146 A/B: the dist gpu feature-row fix.
#
# Two dist_stage builds, ONE box, the SAME cells as the Bench-911 T0.3 rig
# (scripts/plan623_t03_session.sh — this script's cell phase is its twin):
#   INCUMBENT = gpu feature row as shipped before the fix (sequential conv/
#               recurrence paths, per-token attention loops)
#   ARMED     = gpu + ternary_deltanet_chunked_prefill +
#               ternary_attention_batched_prefill (the engine-default mirror)
# Cells per posture: warmup-decode, decode (tg128 + --full-check G1), pp512,
# pp4096. pp32768 behind I1146_PP32K=1 (256 frames x 2 postures — expensive).
#
# G1 cross-posture: the decode `generated:` id lines must be IDENTICAL
# across the two postures (both postures are gate-proven bit-exact vs their
# own oracle; equality across postures is the compose check).
#
# Usage (on the box with the checkout + model):
#   bash scripts/i1146_ab_session.sh            # builds + runs
#   I1146_SKIP_BUILD=1 bash ...                 # binaries already built
# Results: .benchmarks/i1146_ab/<ts>/
set -u

TS=$(date +%Y%m%dT%H%M%SZ)
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT="$REPO/.benchmarks/i1146_ab/$TS"
CELLS="$OUT/cells"
mkdir -p "$CELLS"
LOG="$OUT/session.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== i1146 A/B session start (ts $TS) ==="
sysctl -n hw.model machdep.cpu.brand_string hw.ncpu hw.memsize 2>/dev/null | tee -a "$LOG"
uptime | tee -a "$LOG"

# GPU-quiet gate: refuse to start the cell phase if a compute consumer runs
# (the 649 exclusivity; Unity/Zed exempt per the owner call).
QUIET=$(ps aux | grep -Ei 'llama|mlx|dist_stage|llama-bench|ggml|perf_rematch' | grep -v grep | grep -v i1146_ab || true)
if [ -n "$QUIET" ] && [ "${I1146_FORCE:-0}" != "1" ]; then
  say "REFUSED: GPU not quiet:"; echo "$QUIET" | tee -a "$LOG"; exit 2
fi
say "GPU quiet OK"

MODEL="$REPO/../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"
[ -e "$MODEL" ] || { say "REFUSED: missing $MODEL"; exit 2; }

PORT=17641
IDS4="7,13,42,100"
KEY=$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n')
export RIIR_DIST_KEY="$KEY"

# ── Builds ──────────────────────────────────────────────────────────
if [ "${I1146_SKIP_BUILD:-0}" != "1" ]; then
  say "=== build INCUMBENT (worktree at the fix's parent) ==="
  # The parent of the LAST commit touching the dist gpu row = the incumbent.
  FIX_COMMIT=$(git -C "$REPO" log --format=%H -1 -- crates/riir-infer-dist/Cargo.toml)
  INC_COMMIT=$(git -C "$REPO" rev-parse "${FIX_COMMIT}^")
  say "incumbent commit: $INC_COMMIT (parent of $(git -C "$REPO" rev-parse --short "$FIX_COMMIT"))"
  rm -rf /tmp/i1146-incumbent
  git -C "$REPO" worktree add /tmp/i1146-incumbent "$INC_COMMIT" || { say "worktree add FAILED"; exit 2; }
  (cd /tmp/i1146-incumbent && CARGO_TARGET_DIR=/tmp/i1146-tgt-inc cargo build --release -p riir-infer-dist --bin dist_stage --features gpu,bonsai2_hadamard) >> "$LOG" 2>&1 \
    || { say "INCUMBENT BUILD FAILED"; git -C "$REPO" worktree remove --force /tmp/i1146-incumbent; exit 2; }
  INC_BIN=/tmp/i1146-tgt-inc/release/dist_stage

  say "=== build ARMED (main checkout HEAD) ==="
  (cd "$REPO" && CARGO_TARGET_DIR=/tmp/i1146-tgt-armed cargo build --release -p riir-infer-dist --bin dist_stage --features gpu,bonsai2_hadamard) >> "$LOG" 2>&1 \
    || { say "ARMED BUILD FAILED"; exit 2; }
  ARM_BIN=/tmp/i1146-tgt-armed/release/dist_stage
  git -C "$REPO" worktree remove --force /tmp/i1146-incumbent || true
else
  INC_BIN="${I1146_INC_BIN:?set I1146_INC_BIN}"
  ARM_BIN="${I1146_ARM_BIN:?set I1146_ARM_BIN}"
fi

for b in "$INC_BIN" "$ARM_BIN"; do
  [ -e "$b" ] || { say "REFUSED: missing binary $b"; exit 2; }
done
say "INC_BIN=$INC_BIN sha=$(/usr/bin/shasum -a 256 "$INC_BIN" | awk '{print $1}')"
say "ARM_BIN=$ARM_BIN sha=$(/usr/bin/shasum -a 256 "$ARM_BIN" | awk '{print $1}')"
{
  echo "model: $MODEL"
  /usr/bin/shasum -a 256 "$MODEL" | awk '{print "model_sha256:", $1}'
  echo "incumbent_bin: $INC_BIN"; /usr/bin/shasum -a 256 "$INC_BIN" | awk '{print "incumbent_sha256:", $1}'
  echo "armed_bin: $ARM_BIN"; /usr/bin/shasum -a 256 "$ARM_BIN" | awk '{print "armed_sha256:", $1}'
} > "$OUT/provenance.txt"

# ── Cells (the T0.3 twin; BOTH stages carry --ctx — the tail span gate) ──
run_cell() { # $1 bin  $2 posture  $3 name  $4 ids  $5 max-new  $6 ctx  $7+ extra
  local bin="$1" posture="$2" name="$3" ids="$4" n="$5" ctx="$6"; shift 6
  say "=== $posture: $name ==="
  "$bin" --role tail --model "$MODEL" --split 32 --backend gpu \
    --listen "127.0.0.1:$PORT" --ctx "$ctx" > "$CELLS/${posture}_${name}_tail.log" 2>&1 &
  local tail_pid=$!
  local i=0
  while [ $i -lt 240 ]; do
    if ! kill -0 $tail_pid 2>/dev/null; then say "tail died at boot ($posture/$name)"; break; fi
    if grep -q "listening on" "$CELLS/${posture}_${name}_tail.log" 2>/dev/null; then break; fi
    i=$((i+1)); sleep 1
  done
  local t0 t1 rc
  t0=$(date +%s)
  "$bin" --role head --model "$MODEL" --split 32 --backend gpu \
    --connect "127.0.0.1:$PORT" --prompt "$ids" --max-new "$n" --no-eos --ctx "$ctx" "$@" \
    > "$CELLS/${posture}_${name}_head.log" 2>&1
  rc=$?
  t1=$(date +%s)
  wait $tail_pid 2>/dev/null
  say "$posture/$name head_rc=$rc wall=$((t1-t0))s"
}

for POSTURE in incumbent armed; do
  if [ "$POSTURE" = incumbent ]; then BIN="$INC_BIN"; else BIN="$ARM_BIN"; fi
  run_cell "$BIN" "$POSTURE" warmup "$IDS4" 4 4096
  run_cell "$BIN" "$POSTURE" decode "$IDS4" 128 4096 --full-check
  for P in 512 4096; do
    if [ "$P" = 32768 ] && [ "${I1146_PP32K:-0}" != "1" ]; then continue; fi
    IDS=$(jot -s, "$P" 7 2>/dev/null || python3 -c "print(','.join(str(7+i) for i in range($P)))")
    run_cell "$BIN" "$POSTURE" "pp${P}_warm" "$IDS" 1 $((P + 256)) --prefill-chunk 128
    run_cell "$BIN" "$POSTURE" "pp${P}" "$IDS" 1 $((P + 256)) --prefill-chunk 128
  done
  if [ "${I1146_PP32K:-0}" = "1" ]; then
    IDS=$(jot -s, 32768 7 2>/dev/null || python3 -c "print(','.join(str(7+i) for i in range(32768)))")
    run_cell "$BIN" "$POSTURE" pp32768_warm "$IDS" 1 33024 --prefill-chunk 128
    run_cell "$BIN" "$POSTURE" pp32768 "$IDS" 1 33024 --prefill-chunk 128
  fi
done

# ── Verdict assembly ─────────────────────────────────────────────────
say "=== cross-posture G1 (decode token streams) ==="
GEN_I=$(grep -o 'generated: \[.*\]' "$CELLS/incumbent_decode_head.log" | head -1)
GEN_A=$(grep -o 'generated: \[.*\]' "$CELLS/armed_decode_head.log" | head -1)
if [ -n "$GEN_I" ] && [ "$GEN_I" = "$GEN_A" ]; then
  say "G1 CROSS-POSTURE PASS (decode ids identical)"
else
  say "G1 CROSS-POSTURE FAIL: incumbent=[$GEN_I] armed=[$GEN_A]"
fi

say "=== walls (from session log timestamps) — see the per-cell head logs for the printed rates ==="
grep -E "head_rc=" "$LOG" | tee -a "$LOG"
say "=== i1146 A/B COMPLETE: $OUT ==="
