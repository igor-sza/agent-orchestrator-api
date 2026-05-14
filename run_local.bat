@echo off
REM Run this batch file from project root (Windows)
REM It creates venv, installs dependencies, and starts uvicorn.

SET "VENV=%~dp0venv"

REM 1. Create the virtual environment if not exists
if not exist "%VENV%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV%"
) else (
    echo Virtual environment already exists.
)

REM 2. Install dependencies with venv python
set "PYTHON=%VENV%\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: virtualenv python not found at %PYTHON%
    exit /b 1
)

echo Installing dependencies...
"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -r requirements.txt

REM 3. Start server
echo Starting FastAPI server at http://0.0.0.0:8000 ...
"%PYTHON%" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

