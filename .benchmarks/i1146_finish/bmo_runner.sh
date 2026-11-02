#!/bin/bash
# bmo runner — riir-infer Issue 1146 finishing legs (zed-amo-1146-finish, claim 3f4f828)
# Phase 1: ladder 4(c) matmul2d flash-attention A/B (width rig, env-only arm B)
# Phase 2: per-family KL of the attention arm (issue1146_relaxed_kl, I1146K_AXIS=attn)
# Phase 3: relaxed-OFF pp32768 settle pair (tensor ON vs OFF — the 401af78 claim)
set -u
OUT=/tmp/i1146f
MODEL=/Users/katopz/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf
RIG=$OUT/i1146_width_recurve.sh
LOG=$OUT/runner.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== bmo 1146 runner start ==="
{ date -u; sysctl -n hw.model hw.ncpu hw.memsize; uptime; pmset -g batt | head -1; } > $OUT/box_state.txt 2>&1
shasum -a 256 $OUT/dist_stage $OUT/kl_test > $OUT/bin_sha.txt
/usr/bin/shasum -a 256 "$MODEL" >> $OUT/bin_sha.txt

# ── Phase 1: 4(c) attention A/B ──────────────────────────────────────
say "phase 1: 4(c) attention A/B start"
I1146W_BIN=$OUT/dist_stage I1146W_BIN_B=$OUT/dist_stage \
I1146W_MODEL=$MODEL I1146W_WIDTHS="512" I1146W_LENS="512 4096 32768" I1146W_REPS=2 \
I1146W_ENV_B="RIIR_PREFILL_ATTN_METAL_TENSOR=1" \
I1146W_OUT=$OUT/attn_ab bash $RIG > $OUT/attn_ab.log 2>&1
say "phase 1 rc=$? ; engagement: b-arm announcement lines = $(grep -l 'prefill attention = matmul2d tensor arm' $OUT/attn_ab/cells/b_*_head.log 2>/dev/null | wc -l | tr -d ' ') logs; a-arm = $(grep -l 'prefill attention = matmul2d tensor arm' $OUT/attn_ab/cells/a_*_head.log 2>/dev/null | wc -l | tr -d ' ') (must be 0)"
grep -h "G1 CROSS-WIDTH" $OUT/attn_ab/session.log 2>/dev/null | while read -r l; do say "phase1 $l (note: attn arm is tolerance-class — id divergence routes to the KL read, not auto-FAIL)"; done
grep -c "CONTAM" $OUT/attn_ab/session.log > $OUT/attn_ab_contam_count.txt 2>/dev/null

# ── Phase 2: KL of the attention arm ─────────────────────────────────
say "phase 2: KL (I1146K_AXIS=attn) start"
BONSAI2_G1_MODEL=$MODEL I1146K_AXIS=attn RIIR_METAL_TENSOR_RELAXED=1 \
  $OUT/kl_test --ignored --nocapture --test-threads=1 > $OUT/kl_attn.log 2>&1
say "phase 2 rc=$?"
grep -E "^# issue1146_relaxed_kl|KL mean|vacuous|launch" $OUT/kl_attn.log | while read -r l; do say "phase2 $l"; done

# ── Phase 3: relaxed-OFF pp32768 settle pair ─────────────────────────
say "phase 3: pp32k settle (tensor ON vs OFF, relaxed OFF) start"
I1146W_BIN=$OUT/dist_stage I1146W_BIN_B=$OUT/dist_stage \
I1146W_MODEL=$MODEL I1146W_WIDTHS="512" I1146W_LENS="32768" I1146W_REPS=2 \
I1146W_ENV_A="RIIR_DIST_METAL_TENSOR=0 RIIR_DIST_METAL_TENSOR_RELAXED=0" \
I1146W_ENV_B="RIIR_DIST_METAL_TENSOR_RELAXED=0" \
I1146W_OUT=$OUT/pp32k_settle bash $RIG > $OUT/pp32k_settle.log 2>&1
say "phase 3 rc=$?"
grep -h "G1 CROSS-WIDTH" $OUT/pp32k_settle/session.log 2>/dev/null | while read -r l; do say "phase3 $l"; done

say "=== summary: phase1 timed cells ==="
grep -hE "^.*(pp[0-9]+_w512_r[0-9]) [0-9.]+s" $OUT/attn_ab/session.log 2>/dev/null | while read -r l; do say "p1 $l"; done
say "=== summary: phase3 timed cells ==="
grep -hE "^.*(pp32768_w512_r[0-9]) [0-9.]+s" $OUT/pp32k_settle/session.log 2>/dev/null | while read -r l; do say "p3 $l"; done
say "=== COMPLETE ==="
