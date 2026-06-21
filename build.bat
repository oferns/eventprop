@echo off
REM build.bat — wrapper for build.ps1
REM Usage: build.bat [debug|release]
powershell.exe -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
