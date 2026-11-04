#!/bin/sh
# Issue 1117 T4 — the FlashMemory 128K trio RE-RUN on bmo (queued rig job).
# The m3 v2 run's artifacts went UNSEEN (the m3 unreachable 2026-11-04; the
# parked agent-mail resolves LOST-vs-ALIVE when it wakes) — this job
# regenerates the measurement on the quiet runner with the two loss-lesson
# fixes applied: (1) artifacts on a DURABLE path (~/riir-bench, never a
# cache dir), (2) per-arm bins are written incrementally by the gate itself
# (dense rewrites a complete=0 prefix every bucket; each rho arm lands its
# bin on completion; a digest-matching bin auto-LOADS on restart) — so a
# kill loses at most the in-flight arm and a re-queued job resumes the rest.
#
# Arms: dense + rho 0.15 + rho 0.30 at ctx 131072, bucket 8192 (the armed
# recipe; block/refresh at the 64/64 defaults). The 32K recipe's rho 0.05/
# 0.10 arms are NOT re-run (the 32K record covers them; the 128K trio is
# what the T4 deferral names).
#
# Corpus: the 128K recipe (all workspace AGENTS.md sorted, concatenated) is
# regenerated from BMO's checkout — the repo set may differ from the m3's
# 31, so the corpus bytes + digest are recorded and the record quotes the
# divergence (same-regime aggregate comparability only, the standing caveat).
set -eu

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
WT=/Users/katopz/git/riir-infer-benchwt
OUT="$HOME/riir-bench/i1117_128k"
GGUF="$HOME/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"
GATE="$OUT/target/release/flashmemory_decode_gate"

cd "$WT"
mkdir -p "$OUT"

echo "=== i1117 128K re-run — box state at start ==="
date -u +%Y-%m-%dT%H:%M:%SZ
uptime
sysctl hw.memsize | tr -s ' \t' ' '
df -h "$HOME" | tail -1
git -C "$WT" rev-parse HEAD
test -f "$GGUF" || { echo "REFUSE: gguf missing: $GGUF"; exit 1; }
ls -la "$GGUF"

echo "=== corpus (the 128K recipe, from this box's workspace set) ==="
ls "$HOME"/git/*/AGENTS.md | sort > "$OUT/corpus_sources.txt"
wc -l "$OUT/corpus_sources.txt"
# while-read, not xargs — one cat per line, order preserved, no arg-split.
while read -r src; do cat "$src"; done < "$OUT/corpus_sources.txt" > "$OUT/corpus_128k.txt"
wc -c "$OUT/corpus_128k.txt"
shasum -a 256 "$OUT/corpus_128k.txt"
b3sum "$OUT/corpus_128k.txt" 2>/dev/null || echo "(b3sum not installed — sha256 recorded)"

echo "=== build the gate (release, flashmemory_budget+ternary+hadamard) ==="
cargo build --release \
  --features flashmemory_budget,deltanet_ternary_inference,bonsai2_hadamard \
  --bin flashmemory_decode_gate \
  --target-dir "$OUT/target"

echo "=== launch the trio (dense + rho 0.15 + rho 0.30 @ ctx 131072) ==="
# Load trace every 300 s (post-hoc contention attribution — the ratio law).
(
  while kill -0 "$$" 2>/dev/null; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(uptime | sed 's/.*load averages*: //')" >> "$OUT/load_trace.log"
    sleep 300
  done
) &
TRACE_PID=$!
# caffeinate binds the box's sleep assertions to THIS script's lifetime
# (the i1117 sleep-proofing lesson: re-arm per run, never inherit).
# No tee pipeline — a captured pipeline reads TEE's status, never the gate's
# (the pipefail-discard class); the log file is the durable record and the
# rig end-mail carries its tail.
GATE_RC=0
caffeinate -is "$GATE" \
  --gguf "$GGUF" \
  --corpus-file "$OUT/corpus_128k.txt" \
  --ctx 131072 \
  --rho-list 0.15,0.30 \
  --bucket 8192 \
  --out "$OUT" > "$OUT/run_128k_bmo.log" 2>&1 || GATE_RC=$?
kill "$TRACE_PID" 2>/dev/null || true
echo "gate exit rc=$GATE_RC"
date -u +%Y-%m-%dT%H:%M:%SZ
tail -20 "$OUT/run_128k_bmo.log"

echo "=== best-effort collection (bins are the durable record either way) ==="
if python3 "$WT/scripts/i1117_collect_128k.py" --out "$OUT" 2>&1; then
  echo "collector wrote the stats tables"
else
  echo "collector skipped/failed on bmo (no 32K ref report here) — the arm bins"
  echo "in $OUT are complete and durable; the record task runs the collector"
  echo "against the 32K ref on a box that has it"
fi

echo "=== artifacts ==="
ls -la "$OUT"
echo "RE-RUN DONE — arm bins: dense.bin / rho_0.15.bin / rho_0.30.bin (complete=1 each)."
exit "$GATE_RC"
