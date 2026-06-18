@echo off
title Unsloth GGUF Chat Server - Gemma 3 1B
echo =====================================================================
echo    🦥 Starting Local Chat Server for Gemma 3 1B GGUF model...
echo =====================================================================
echo.
echo Model Path:  C:\Users\Rudraksh\.unsloth\studio\exports\gemma-3-1b-it-unsloth-bnb-4bit-gguf\gemma-3-1b-it.Q4_K_M.gguf
echo Port:        8080
echo GPU Offload: Enabled (RTX 4060)
echo.
echo Launching local web UI in your default browser...
echo.
echo Press [Ctrl + C] in this window to shut down the server.
echo =====================================================================
echo.

:: Start default browser pointing to llama-server web client
start http://localhost:8080

:: Launch llama-server offloading all 99 layers to GPU for maximum speed
"C:\Users\Rudraksh\.unsloth\llama.cpp\build\bin\Release\llama-server.exe" -m "C:\Users\Rudraksh\.unsloth\studio\exports\gemma-3-1b-it-unsloth-bnb-4bit-gguf\gemma-3-1b-it.Q4_K_M.gguf" -c 2048 --port 8080 -ngl 99

pause
