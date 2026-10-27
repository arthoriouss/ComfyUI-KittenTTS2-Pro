@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0..\..\..\python_embeded\python.exe"
if not exist "%PYTHON%" (
  echo [KittenTTS2] Embedded Python not found: %PYTHON%
  echo This installer is for ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-KittenTTS2-Pro
  echo For desktop/venv users run: python scripts\install_backend.py
  exit /b 1
)
"%PYTHON%" scripts\install_backend.py
if errorlevel 1 (
  echo Installation stopped. Read docs\TROUBLESHOOTING.md
  exit /b 1
)
echo Restart ComfyUI now.
pause
