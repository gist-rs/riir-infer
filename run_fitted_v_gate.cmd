@echo off
rem run_fitted_v_gate.cmd - Issue 013 T1 full run (the Issue-012 scheduled-task
rem recipe: launches OUTSIDE the agent process tree so a watchdog kill of the
rem agent cannot take the ~6h measurement down).
rem
rem Arms: f16 | p-b2 p-b3 p-b4 | mr-b2 mr-b3 mr-b4 | mr-b4-k1024
rem Slices: cal [0..61440) eval [61440..73728) of the chat_probe token stream.
rem Report: rewritten after every arm at .benchmarks/011_run_report.md

cd /d E:\git\riir-infer
target-rel\release\fitted_v_gate.exe ^
  ..\riir-train\data\gemma-2-2b-it-f16.gguf ^
  ..\riir-train\data\chat_probe ^
  --cal-tokens 61440 --eval-tokens 12288 --bits 2,3,4 --top-k 8192 --dial-k 1024 ^
  --report .benchmarks\011_run_report.md ^
  > .benchmarks\011_run.log 2>&1
