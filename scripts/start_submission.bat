@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if not exist "%~dp0EatingDetection.exe" (
  echo [EatingSense] EatingDetection.exe is missing. Extract the complete submission package.
  pause
  exit /b 2
)
"%~dp0EatingDetection.exe" --visual --open %*
set "VISUAL_EXIT=%ERRORLEVEL%"
if not "%VISUAL_EXIT%"=="0" pause
popd
exit /b %VISUAL_EXIT%
