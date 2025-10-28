@echo off
REM Quick Start Script for Phi-3 Payments Fine-tuning (Windows)

echo ==================================================
echo Phi-3 Payments Fine-tuning - Quick Start (Windows)
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo Python detected:
python --version

REM Check if CUDA/GPU is available
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. Make sure CUDA is installed.
    echo You can continue but training may not work without a GPU.
    pause
) else (
    echo GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo Creating virtual environment...
   uv venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
uv pip install --upgrade pip

REM Install PyTorch with CUDA support first (Windows-specific)
echo.
echo Installing PyTorch with CUDA support...
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
echo.
echo Installing other dependencies...
uv pip install -r requirements.txt

echo.
echo All dependencies installed!

REM Generate dataset
echo.
echo ==================================================
echo Step 1: Generating synthetic payments dataset
echo ==================================================
python generate_payments_dataset.py

REM Ask if user wants to start training
echo.
echo ==================================================
echo Step 2: Fine-tuning (takes 30-45 minutes)
echo ==================================================
echo.
set /p train="Start training now? (y/n): "

if /i "%train%"=="y" (
    echo.
    echo Starting training...
    echo You can monitor GPU usage with: nvidia-smi
    echo.
    python finetune_phi3_payments.py
    
    echo.
    echo ==================================================
    echo Training complete! Testing the model...
    echo ==================================================
    python test_payments_model.py
    
    echo.
    echo All done!
    echo.
    echo Next steps:
    echo   - Try interactive mode: python test_payments_model.py interactive
    echo   - Customize dataset: edit generate_payments_dataset.py
    echo   - Adjust training: edit finetune_phi3_payments.py
) else (
    echo.
    echo Skipping training. You can run it later with:
    echo   venv\Scripts\activate.bat
    echo   python finetune_phi3_payments.py
)

echo.
echo ==================================================
echo Setup Complete!
echo ==================================================
pause
