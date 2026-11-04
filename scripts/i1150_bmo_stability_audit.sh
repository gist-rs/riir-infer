#!/bin/sh
# Issue 1150 follow-up 1 — the stability/bimodality audit on bmo (queued rig
# job; NOT a scored row). The i1150 record's re-run protocol, first half:
# reproduce-or-retire the ~3x cross-process bimodality (18 vs 6 tok/s on
# comparable work, neither mode explained). Two arms x repeated FRESH
# processes, per-pass RSS/load watchdog, warm-shape sweep (first pass pays
# the first-ever-compile autotune cost — 20-32 min class), all artifacts on a
# DURABLE path (the i1117 ~/.cache loss lesson — never a cache dir again).
#
# Runs on bmo's bench worktree /Users/katopz/git/riir-infer-benchwt
# (detached; cross-repo path deps resolve beside the real repos).
# Diagnostic only: a failed PASS logs loud and the audit continues; only a
# failed BUILD exits nonzero.
set -eu

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
WT=/Users/katopz/git/riir-infer-benchwt
OUT="$HOME/riir-bench/i1150_stability"
MODEL="$HOME/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf"
BIN="$OUT/target/release/qwen38_wgpu_bench"

cd "$WT"
mkdir -p "$OUT"

echo "=== i1150 stability audit — box state at start ==="
date -u +%Y-%m-%dT%H:%M:%SZ
uptime
sysctl hw.memsize | tr -s ' \t' ' '
vm_stat | head -5
df -h "$HOME" | tail -1
test -f "$MODEL" || { echo "REFUSE: model missing: $MODEL"; exit 1; }
ls -la "$MODEL"

echo "=== build (release, qwen38_wgpu+metal_tensor_gemm) ==="
git -C "$WT" rev-parse HEAD
cargo build --release -p riir-infer-gpu \
  --features qwen38_wgpu,metal_tensor_gemm \
  --bin qwen38_wgpu_bench \
  --target-dir "$OUT/target"

# Watchdog: sample the bench pid's RSS + box load every 5 s into a per-pass
# trace. Plain loop, no captured pipelines (the pipefail-discard class).
watchdog() {
  wp_pid=$1
  wp_log=$2
  (
    while kill -0 "$wp_pid" 2>/dev/null; do
      ts=$(date -u +%H:%M:%S)
      rss=$(ps -o rss= -p "$wp_pid" 2>/dev/null | tr -d ' ')
      ld=$(uptime | sed 's/.*load averages*: //')
      echo "$ts rss_kb=${rss:-gone} load=$ld" >> "$wp_log"
      sleep 5
    done
  ) &
  WP_PID=$!
}

run_pass() {
  rp_name=$1
  rp_env_flag=$2   # "" = incumbent Q4_K posture; "1" = dense-f16 MM2D
  rp_args=$3
  rp_log="$OUT/${rp_name}.log"
  rp_wd="$OUT/${rp_name}.watchdog.log"
  echo "=== pass $rp_name : qwen38_wgpu_bench $rp_args (dense_mm2d=${rp_env_flag:-off}) ==="
  date -u +%Y-%m-%dT%H:%M:%SZ
  if [ -n "$rp_env_flag" ]; then
    RIIR_QWEN38_DENSE_MM2D="$rp_env_flag" caffeinate -is "$BIN" \
      --gguf "$MODEL" $rp_args > "$rp_log" 2>&1 &
  else
    caffeinate -is "$BIN" --gguf "$MODEL" $rp_args > "$rp_log" 2>&1 &
  fi
  rp_pid=$!
  watchdog "$rp_pid" "$rp_wd"
  wait "$rp_pid" || echo "PASS $rp_name FAILED (exit $?) — logged, audit continues"
  kill "$WP_PID" 2>/dev/null || true
  wait "$WP_PID" 2>/dev/null || true
  grep -iE "tok/s|tokens/s|median|cell" "$rp_log" | tail -12 || true
}

# Warm-shape sweep: pass w1 pays the first-ever-compile autotune cost into a
# FRESH target dir; w2 measures the warm cache on the same shape.
run_pass w1_warm_pp512_incumbent "" "--pp 512"
run_pass w2_warm_pp512_incumbent "" "--pp 512"

# Arm A — incumbent Q4_K packed posture (the fresh-quiet-process mode).
run_pass a1_pp2484_incumbent "" "--pp 2484"
run_pass a2_pp2484_incumbent "" "--pp 2484"
run_pass a3_pp2484_incumbent "" "--pp 2484"
run_pass a4_pp4954_incumbent "" "--pp 4954"

# Arm B — the dense-f16 MM2D posture (the four-way's 18 tok/s claim).
run_pass b1_pp2484_dense_f16 1 "--pp 2484"
run_pass b2_pp2484_dense_f16 1 "--pp 2484"
run_pass b3_pp2484_dense_f16 1 "--pp 2484"
run_pass b4_pp4954_dense_f16 1 "--pp 4954"

echo "=== i1150 stability audit — box state at end ==="
date -u +%Y-%m-%dT%H:%M:%SZ
uptime
echo "=== artifacts ==="
ls -la "$OUT"
echo "AUDIT DONE — the record task parses $OUT/*.log (raw bench output) beside"
echo "$OUT/*.watchdog.log (5 s RSS/load traces). No verdict computed here."
