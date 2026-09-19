@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo [EatingSense] Python was not found on PATH.
  echo Install Python 3.11+ from https://www.python.org/downloads/ and run this file again.
  echo During setup, tick "Add python.exe to PATH".
  pause
  exit /b 1
)
%PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [EatingSense] Python 3.11 or newer is required. Found:
  %PY% --version
  echo Install Python 3.11+ from https://www.python.org/downloads/ and run this file again.
  pause
  exit /b 1
)
rem Locate the local inference bridge entry (dist: inference\serve.py; submission: app\serve.py or serve.py).
set "SERVE="
if exist "%~dp0inference\serve.py" set "SERVE=%~dp0inference\serve.py"
if not defined SERVE if exist "%~dp0app\serve.py" set "SERVE=%~dp0app\serve.py"
if not defined SERVE if exist "%~dp0serve.py" set "SERVE=%~dp0serve.py"
if not defined SERVE (
  echo [EatingSense] serve.py was not found next to this launcher.
  pause
  exit /b 1
)
rem Locate the pinned requirements file (dist layout: inference\requirements.txt; submission layout: requirements.txt).
set "REQ="
if exist "%~dp0inference\requirements.txt" set "REQ=%~dp0inference\requirements.txt"
if not defined REQ if exist "%~dp0requirements.txt" set "REQ=%~dp0requirements.txt"
if not defined REQ (
  echo [EatingSense] requirements.txt was not found next to this launcher.
  pause
  exit /b 1
)
rem Verify the pinned runtime dependencies and install them automatically on first
rem run, so that reviewers can simply double-click this file (no questions asked).
%PY% -c "import numpy, sklearn, joblib, lightgbm" >nul 2>nul
if errorlevel 1 (
  echo [EatingSense] First run: installing required Python packages, this may take a few minutes...
  echo     %PY% -m pip install -r "!REQ!"
  %PY% -m pip install --no-input --disable-pip-version-check -r "!REQ!"
  %PY% -c "import numpy, sklearn, joblib, lightgbm" >nul 2>nul
  if errorlevel 1 (
    echo [EatingSense] Automatic installation failed. Install manually, then run this file again:
    echo     %PY% -m pip install -r "!REQ!"
    echo If the download is slow or blocked, retry with a mirror, for example:
    echo     %PY% -m pip install -r "!REQ!" -i https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
  )
  echo [EatingSense] Dependencies installed.
)
%PY% "%SERVE%" --open
if errorlevel 1 pause
endlocal
