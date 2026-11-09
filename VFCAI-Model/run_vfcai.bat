@echo off
chcp 65001 > nul

echo ====================================
echo    vfcai - Setup and Execution
echo ====================================

echo Installing package in editable mode...
pip install -e .
if errorlevel 1 goto ERROR

echo.
echo Starting vfcai CLI...
call vfcai
if errorlevel 1 goto ERROR

echo.
echo ------------------------------------
echo Process completed successfully.
echo ------------------------------------
pause
exit /b 0

:ERROR
echo.
echo ====================================
echo [ERROR] An error occurred during execution.
echo ====================================
pause
exit /b 1