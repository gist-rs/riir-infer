@echo off
rem Issue 012 T6 64K/25pct - the random arm, launched OUTSIDE the agent
rem process tree via schtasks (the agent harness kills long-running
rem child processes; two in-flight attempts of this arm died to it).
rem Log: run_random2.log (run_random.log is the killed attempt's partial).
cd /d E:\git\riir-infer
target\release\needle_eviction_gate.exe --context 65536 --needles 8 --budget-frac 0.25 --arms random --full-from .benchmarks\needle_gate_64k_25pct\needle_gate_report.json --out .benchmarks\needle_gate_64k_25pct > .benchmarks\needle_gate_64k_25pct\run_random2.log 2>&1
