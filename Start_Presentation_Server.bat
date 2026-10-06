@echo off
title Predictive-Machine Guard - Presentation Server & Public Link
cd /d "%~dp0"
echo ============================================================================
echo   PREDICTIVE-MACHINE GUARD (ML + DEEP LEARNING v2.0) - PRESENTATION LAUNCHER
echo ============================================================================
echo.
echo [1/2] Starting Local Python ML & Deep Learning Server on http://127.0.0.1:8000 ...
start "Predictive-Machine Guard Backend" /min .\.venv\Scripts\python.exe app.py
timeout /t 4 /nobreak >nul
start http://127.0.0.1:8000

echo [2/2] Starting Persistent Public HTTPS Shareable Tunnel ...
if exist cloudflared.exe (
    cloudflared.exe tunnel --url http://127.0.0.1:8000
) else (
    ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -R 80:127.0.0.1:8000 nokey@localhost.run
)
pause
