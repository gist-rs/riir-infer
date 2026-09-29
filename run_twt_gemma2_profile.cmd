@echo off
rem run_twt_gemma2_profile.cmd - Issue 022 T5.1 lane 1: the gemma-2 f16
rem control profile (launches OUTSIDE the agent process tree; the T2 recipe).
rem
rem Corpus: the same chat_probe stream T2 uses (6.3M chars → chunked at 1024).
rem Budget: 6144 tokens = 6 chunks x 1024 (the pre-registered profile budget).
rem Artifact: .raw/twt/gemma2_profile.json (gitignored, BLAKE3-pinned corpus).

cd /d E:\git\riir-infer
echo === box state at launch (%DATE% %TIME%) === > .raw\twt\gemma2_profile_run.log
powershell -NoProfile -Command "$os = Get-CimInstance Win32_OperatingSystem; 'Free RAM: {0:N1} GB / {1:N1} GB' -f ($os.FreePhysicalMemory/1MB), ($os.TotalVisibleMemorySize/1MB)" >> .raw\twt\gemma2_profile_run.log

target-twt\twt_gemma2_profile.exe ^
  --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf ^
  --corpus ../riir-train/data/chat_probe ^
  --seq-len 1024 --max-tokens 6144 ^
  --out .raw/twt/gemma2_profile.json ^
  >> .raw\twt\gemma2_profile_run.log 2>&1
echo === run exited (%DATE% %TIME%) rc=%ERRORLEVEL% === >> .raw\twt\gemma2_profile_run.log
