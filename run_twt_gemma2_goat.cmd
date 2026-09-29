@echo off
rem run_twt_gemma2_goat.cmd - Issue 022 T5.1 lane 1: the gemma-2 f16 control
rem GOAT pipeline (CHAINED after the profile; the T2/T3 recipe — launches
rem OUTSIDE the agent process tree).
rem
rem Waits for twt_gemma2_profile.exe to exit (max 1h), then:
rem   1. emit collapsed GGUFs at the coarse pre-registered eps grid
rem   2. agreement per point against the CACHED parent arm
rem   3. (the fine-end bracket is a FOLLOW-UP run, chosen from the coarse
rem      verdict — same as the T5.0 protocol: coarse first, bracket after)
rem
rem Disk: each collapsed file is ~(m/26) x 5.2 GB; the six coarse points at
rem eps >= 0.5 are near-total collapses — emitted to .raw/twt/ (gitignored).

cd /d E:\git\riir-infer
echo === goat pipeline launch (%DATE% %TIME%) === > .raw\twt\gemma2_goat_run.log

rem -- Guard: wait for the profile to exit (max 1h) --
set /a waits=0
:waitloop
tasklist /FI "IMAGENAME eq twt_gemma2_profile.exe" 2>nul | find /I "twt_gemma2_profile.exe" >nul
if %ERRORLEVEL% EQU 0 (
  set /a waits+=1
  if %waits% GTR 60 (
    echo === GUARD TIMEOUT: twt_gemma2_profile.exe still running after 1h — refusing (%DATE% %TIME%) === >> .raw\twt\gemma2_goat_run.log
    exit /b 1
  )
  ping -n 61 127.0.0.1 >nul
  goto waitloop
)
echo === profile exited after %waits% min; box is ours (%DATE% %TIME%) === >> .raw\twt\gemma2_goat_run.log

rem -- Guard: the profile artifact must exist --
if not exist .raw\twt\gemma2_profile.json (
  echo === REFUSED: profile artifact missing — the profile died before writing (%DATE% %TIME%) === >> .raw\twt\gemma2_goat_run.log
  exit /b 1
)

rem -- The parent arm's cache (one 4096-token pass; every collapsed arm reuses it) --
set CACHE=.raw/twt/gemma2_parent_cache.json

rem -- Coarse grid: emit + agreement per pre-registered eps --
for %%E in (0.05 0.1 0.2 0.3 0.5 0.8 1.2) do call :one_point %%E

echo === coarse grid COMPLETE (%DATE% %TIME%) === >> .raw\twt\gemma2_goat_run.log
goto :eof

:one_point
set EPS=%1
echo === emitting eps=%EPS% (%DATE% %TIME%) === >> .raw\twt\gemma2_goat_run.log
set EPSN=%EPS:.=_%
target-twt\twt_collapse_emit.exe ^
  --parent ../riir-train/data/gemma-2-2b-it-f16.gguf ^
  --profile .raw/twt/gemma2_profile.json ^
  --eps %EPS% ^
  --out .raw/twt/gemma2_collapse_e%EPSN%.gguf ^
  >> .raw\twt\gemma2_goat_run.log 2>&1
if not exist .raw\twt\gemma2_collapse_e%EPSN%.gguf (
  echo === REFUSED: emit failed at eps=%EPS% (%DATE% %TIME%) === >> .raw\twt\gemma2_goat_run.log
  goto :eof
)
echo === agreement eps=%EPS% (%DATE% %TIME%) === >> .raw\twt\gemma2_goat_run.log
target-twt\twt_goat_agreement.exe ^
  --parent ../riir-train/data/gemma-2-2b-it-f16.gguf ^
  --collapsed .raw/twt/gemma2_collapse_e%EPSN%.gguf ^
  --corpus ../riir-train/data/chat_probe ^
  --seq-len 1024 --max-tokens 4096 ^
  --cache %CACHE% ^
  >> .raw\twt\gemma2_goat_run.log 2>&1
goto :eof
