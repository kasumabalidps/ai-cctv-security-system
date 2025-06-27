@echo off
cls
echo ================================================================
echo    🛡️ Professional CCTV Security System - Development Mode
echo ================================================================
echo.

REM Change to project directory
cd /d "C:\Users\panda\Documents\Programming\cctv-viewer-ipcam-py"

echo 📁 Current directory: %CD%
echo.

REM Check if conda environment exists
if not exist ".conda" (
    echo ❌ Conda environment not found at .conda
    echo    Please create conda environment first:
    echo    conda create --prefix .conda python=3.10
    echo.
    pause
    exit /b 1
)

echo 🐍 Activating conda environment...
call conda activate .\.conda
if errorlevel 1 (
    echo ❌ Failed to activate conda environment
    echo    Make sure conda is installed and in PATH
    pause
    exit /b 1
)

echo ✅ Conda environment activated: .conda
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  WARNING: .env file not found!
    echo    Please copy .env.example to .env and configure:
    echo    copy .env.example .env
    echo.
    echo    Required variables:
    echo    - CAMERA_IP, CAMERA_USERNAME, CAMERA_PASSWORD
    echo    - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional)
    echo    - DISCORD_WEBHOOK_URL (optional)
    echo.
    pause
)

REM Check Python and dependencies
echo 🔍 Checking Python installation...
python --version
if errorlevel 1 (
    echo ❌ Python not found in conda environment
    pause
    exit /b 1
)

echo.
echo 📦 Checking key dependencies...
python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" 2>nul || echo "❌ OpenCV not installed"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" 2>nul || echo "❌ NumPy not installed"
python -c "import ultralytics; print('✅ YOLO: Available')" 2>nul || echo "⚠️  YOLO not installed (AI detection disabled)"

echo.
echo 🚀 Starting CCTV Security System...
echo    Press Ctrl+C to stop
echo    Use ESC or Q to quit from application
echo ================================================================
echo.

REM Run the main application
python main.py

echo.
echo 🔴 Application stopped
pause
