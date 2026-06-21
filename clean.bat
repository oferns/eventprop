@echo off
REM clean.bat — wrapper for clean.ps1
REM Usage: clean.bat [-full] [all|c|cpp|python|dotnet|parity|benchmarks]
powershell.exe -ExecutionPolicy Bypass -File "%~dp0clean.ps1" %*
