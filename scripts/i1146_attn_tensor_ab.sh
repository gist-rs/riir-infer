#!/usr/bin/env bash
# i1146_attn_tensor_ab.sh — Issue 1146 ladder 4(c): the matmul2d flash-attention
# prefill arm (`RIIR_PREFILL_ATTN_METAL_TENSOR=1`, landed opt-in `672b29c`,
# G1-proven) — perf A/B + per-family quality on ONE box.
#
# The selector is RUNTIME ENV, so the A/B is env-armed from ONE binary through
# scripts/i1146_width_recurve.sh (its I1146W_ENV_A/_B posture, per-pair
# CONTAM guard, G1 cross-arm cells):
#   arm a = env unset   — the shipping attention (M=32 tiled CubeCL flash)
#   arm b = selector on — the matmul2d Metal-tensor attention arm
# Widths: 512 (the Bench-925 promoted census posture — attention 37% of the
# frame) + 128 (the dist default today — a win must transfer for promotion).
# Lens: 512 / 4096 / 32768 (4(a) died at pp32k; 4(c) must win at long context).
#
# Quality: issue1146_relaxed_kl I1146K_AXIS=attn — ref = CubeCL attention
# chain, cand = the tensor arm, tensor GEMM ON in both arms. The attn axis is
# ALWAYS a tolerance measurement (strict-control gate is the gemm axis), so
# this runs as a REPORT: no pre-fixed bar; the bar is chosen from the data
# after, per the lossy-surface promotion rule.
#
# Engagement evidence (vacuous-arm guard): every b_ cell's tail log must carry
# the arm's boot line; the KL run asserts its own launch counter.
#
# Usage (ON the box — model + sibling layout required):
#   bash scripts/i1146_attn_tensor_ab.sh
# Env: I1146_SKIP_PULL=1 (already synced), I1146_FORCE=1 (ignore the quiet
#      gate — G1-class runs only). Results: .benchmarks/i1146_attn_ab/<ts>/
set -u

TS=$(date +%Y%m%dT%H%M%SZ)
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT="$REPO/.benchmarks/i1146_attn_ab/$TS"
CELLS="$OUT/cells"
mkdir -p "$CELLS"
LOG="$OUT/session.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== i1146 attn-tensor A/B start (ts $TS) ==="
{ sysctl -n hw.model machdep.cpu.brand_string hw.ncpu hw.memsize 2>/dev/null; uptime; } | tee -a "$LOG"

# ── Sync: the rig + kernel must be the pushed state this script ships with ──
if [ "${I1146_SKIP_PULL:-0}" != "1" ]; then
  OLD=$(git -C "$REPO" rev-parse --short HEAD)
  git -C "$REPO" fetch origin >> "$LOG" 2>&1
  git -C "$REPO" pull --ff-only >> "$LOG" 2>&1 || { say "REFUSED: pull not ff (diverged?) — sync by hand"; exit 2; }
  say "sync: $OLD -> $(git -C "$REPO" rev-parse --short HEAD)"
fi
SCRIPT_SHA=$(git -C "$REPO" log --format=%h -1 -- scripts/i1146_attn_tensor_ab.sh)
say "script at commit: $SCRIPT_SHA"

MODEL="$REPO/../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"
[ -e "$MODEL" ] || { say "REFUSED: missing $MODEL"; exit 2; }

# ── Build ONE dist_stage at the promoted posture ──
say "=== build dist_stage (gpu,bonsai2_hadamard) ==="
TGT=/tmp/i1146-attn-tgt
(cd "$REPO" && CARGO_TARGET_DIR="$TGT" cargo build --release -p riir-infer-dist --bin dist_stage --features gpu,bonsai2_hadamard) >> "$LOG" 2>&1 \
  || { say "BUILD FAILED"; exit 2; }
BIN="$TGT/release/dist_stage"
say "bin: $BIN sha=$(/usr/bin/shasum -a 256 "$BIN" | awk '{print $1}')"

# ── Perf A/B through the proven width-recurve rig ──
export I1146W_BIN="$BIN"
export I1146W_BIN_B="$BIN"                       # same binary — env-armed A/B
export I1146W_ENV_B="RIIR_PREFILL_ATTN_METAL_TENSOR=1"
export I1146W_WIDTHS="512 128"
export I1146W_LENS="512 4096 32768"
export I1146W_REPS=2
export I1146W_OUT="$OUT/recurve"
[ "${I1146_FORCE:-0}" = "1" ] && export I1146W_FORCE=1
bash "$REPO/scripts/i1146_width_recurve.sh" 2>&1 | tee -a "$LOG"

# Engagement evidence: every surviving b_ tail log must carry the arm line.
ARMED=0; UNARMED=0; FALLBACK=0
for f in "$OUT"/recurve/cells/b_*_tail.log; do
  [ -e "$f" ] || continue
  if grep -q "prefill attention = matmul2d tensor arm" "$f"; then ARMED=$((ARMED+1)); fi
  if grep -q "tensor attention disabled" "$f"; then FALLBACK=$((FALLBACK+1)); fi
  UNARMED=$((UNARMED+1))
done
say "engagement: b-cells=$UNARMED armed-lines=$ARMED fall-through=$FALLBACK"
[ "$ARMED" -eq 0 ] && say "VERDICT-CAVEAT: NO armed b_ cell — the A/B compared the arm to itself"

# ── Quality: per-family KL, attn axis (REPORT — no pre-fixed bar) ──
say "=== build + run issue1146_relaxed_kl (I1146K_AXIS=attn) ==="
KFEAT="metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard"
(cd "$REPO" && I1146K_AXIS=attn CARGO_TARGET_DIR=/tmp/i1146-attn-kl \
  cargo test -p riir-infer-gpu --release --features "$KFEAT" \
  --test issue1146_relaxed_kl -- --ignored --nocapture --test-threads=1) > "$OUT/kl_attn.txt" 2>&1 \
  || say "KL RUN FAILED (see kl_attn.txt)"
grep -E "^# issue1146_relaxed_kl|KL mean|launch" "$OUT/kl_attn.txt" | tee -a "$LOG"

say "=== verdict inputs ==="
grep -E "G1 CROSS|tok/s" "$OUT/recurve/session.log" 2>/dev/null | tail -40 | tee -a "$LOG"
say "=== done — raw cells in $OUT ==="
exit 0
