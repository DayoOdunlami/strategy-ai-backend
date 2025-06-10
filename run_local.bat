@echo off
echo Setting up local development environment...

REM Check Python installation
python --version 2>NUL
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b
)

REM Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b
)

REM Install dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b
)

REM Run the FastAPI app
echo Starting FastAPI application...
echo You can access the API at http://localhost:8000
echo API documentation is available at http://localhost:8000/docs
python -m uvicorn main:app --reload

REM Keep the window open if there's an error
pause 