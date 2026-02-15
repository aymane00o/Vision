@echo off
echo ================================================
echo   Vision Detection System - Setup Script
echo ================================================
echo.

echo [1/4] Checking Python 3.11...
py -3.11 --version
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11 not found. Installing via winget...
    winget install Python.Python.3.11
)

echo.
echo [2/4] Installing PyTorch with CUDA 12.8...
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo.
echo [3/4] Installing other dependencies...
py -3.11 -m pip install -r requirements.txt

echo.
echo [4/4] Verifying GPU...
py -3.11 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo.
echo ================================================
echo   Setup complete! Run with:
echo   py -3.11 face_detection.py
echo   py -3.11 face_emotion_emoji.py
echo   py -3.11 boat_detection.py
echo ================================================
pause
