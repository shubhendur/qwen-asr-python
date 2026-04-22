@echo off
REM =========================================================================
REM One-shot setup for Windows.
REM
REM Usage:
REM   setup.bat                            Install ONNX-only Qwen path (Python 3.9+)
REM   setup.bat --pytorch                  Also install the PyTorch backend (Python 3.10+)
REM   setup.bat --parakeet                 Also install the Parakeet TDT v3 backends
REM   setup.bat --pytorch --parakeet       Install everything
REM
REM Creates a local venv at .\qwen_env so the install never touches your
REM system Python.
REM =========================================================================
setlocal enabledelayedexpansion

if "%PYTHON%"=="" set PYTHON=python

set WANT_PYTORCH=0
set WANT_PARAKEET=0
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--pytorch"  set WANT_PYTORCH=1
if /I "%~1"=="--parakeet" set WANT_PARAKEET=1
shift
goto parse_args
:args_done

for /f "tokens=*" %%v in ('%PYTHON% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%v
echo [setup] Using %PYTHON% (Python %PYVER%)

if "%WANT_PYTORCH%"=="1" (
    %PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
    if errorlevel 1 (
        echo [setup] ERROR: --pytorch requires Python ^>= 3.10 ^(you have %PYVER%^)
        echo         Install Python 3.10+ from https://python.org and re-run:
        echo           set PYTHON=py -3.12
        echo           setup.bat --pytorch
        exit /b 1
    )
)

if not exist qwen_env (
    echo [setup] Creating venv at .\qwen_env
    %PYTHON% -m venv qwen_env
)

call qwen_env\Scripts\activate.bat

python -m pip install --upgrade pip
echo [setup] Installing base + ONNX requirements...
pip install -r requirements.txt
if errorlevel 1 exit /b 1

if "%WANT_PYTORCH%"=="1" (
    echo [setup] Installing PyTorch + qwen-asr (this may take several minutes)...
    pip install -r requirements-pytorch.txt
    if errorlevel 1 exit /b 1
)

if "%WANT_PARAKEET%"=="1" (
    echo [setup] Installing Parakeet TDT v3 backends ^(MLX + ONNX INT8^)...
    pip install -r requirements-parakeet.txt
    if errorlevel 1 exit /b 1
)

echo.
echo [setup] Done.
echo         Run the dictation app with:
echo           qwen_env\Scripts\python.exe qwen_dictation.py
echo         Or activate the venv first:
echo           call qwen_env\Scripts\activate.bat
echo           python qwen_dictation.py
