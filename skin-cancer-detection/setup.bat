@echo off
REM Setup script for Skin Cancer Detection project (Windows)

echo ========================================
echo Skin Cancer Detection - Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

echo Python found:
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if %errorlevel% equ 0 (
    echo [OK] Virtual environment created successfully
) else (
    echo [ERROR] Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo [OK] Dependencies installed successfully
) else (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate
echo.
echo To start using the project:
echo   1. Add your dataset to the data\raw\ directories
echo   2. Train a model: python main.py train --model-type cnn
echo   3. Evaluate: python main.py evaluate
echo   4. Make predictions: python main.py predict path\to\image.jpg
echo.
echo For more information, see README.md
echo.
pause
