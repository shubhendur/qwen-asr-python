#!/usr/bin/env bash
# ============================================================================
# One-shot setup for macOS / Linux.
#
# Usage:
#   ./setup.sh                          # Install ONNX-only Qwen path (Python 3.9+)
#   ./setup.sh --pytorch                # Also install the PyTorch backend (Python 3.10+)
#   ./setup.sh --parakeet               # Also install the Parakeet TDT v3 backends
#   ./setup.sh --pytorch --parakeet     # Install everything
#
# Creates a local venv at ./qwen_env if you don't already have one, so the
# install never touches your system Python.
# ============================================================================
set -e

PYTHON="${PYTHON:-python3}"
WANT_PYTORCH=0
WANT_PARAKEET=0
for arg in "$@"; do
  case "$arg" in
    --pytorch)  WANT_PYTORCH=1 ;;
    --parakeet) WANT_PARAKEET=1 ;;
  esac
done

# Pick a Python that satisfies our requirements.
PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[setup] Using $PYTHON (Python $PYVER)"

if [ "$WANT_PYTORCH" = "1" ]; then
  PYMAJOR=$(echo "$PYVER" | cut -d. -f1)
  PYMINOR=$(echo "$PYVER" | cut -d. -f2)
  if [ "$PYMAJOR" -lt 3 ] || { [ "$PYMAJOR" = "3" ] && [ "$PYMINOR" -lt 10 ]; }; then
    echo "[setup] ERROR: --pytorch requires Python >= 3.10 (you have $PYVER)"
    echo "        Install Python 3.10+ from https://python.org or your package manager,"
    echo "        then re-run:  PYTHON=python3.12 ./setup.sh --pytorch"
    exit 1
  fi
fi

# Create a project-local venv if we don't already have one.
if [ ! -d qwen_env ]; then
  echo "[setup] Creating venv at ./qwen_env"
  "$PYTHON" -m venv qwen_env
fi

# shellcheck disable=SC1091
. qwen_env/bin/activate

python -m pip install --upgrade pip
echo "[setup] Installing base + ONNX requirements..."
pip install -r requirements.txt

if [ "$WANT_PYTORCH" = "1" ]; then
  echo "[setup] Installing PyTorch + qwen-asr (this may take several minutes)..."
  pip install -r requirements-pytorch.txt
fi

if [ "$WANT_PARAKEET" = "1" ]; then
  echo "[setup] Installing Parakeet TDT v3 backends (MLX + ONNX INT8)..."
  pip install -r requirements-parakeet.txt
fi

echo
echo "[setup] Done."
echo "        Run the dictation app with:"
echo "          ./qwen_env/bin/python qwen_dictation.py"
echo "        Or activate the venv first:"
echo "          source qwen_env/bin/activate && python qwen_dictation.py"
