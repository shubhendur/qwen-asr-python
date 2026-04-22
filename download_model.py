#!/usr/bin/env python3
"""
Download helper for the dictation app.

Two backends are supported:
  - PyTorch / qwen-asr package (full precision — options 1–4)
  - ONNX Runtime INT4 MatMulNBits (ultra-low RAM — options 5, 6)

Pick an option below to pre-fetch the relevant model files. The dictation
script auto-downloads on first use too, so running this is optional.
"""

from __future__ import annotations

import os
import sys


# ==========================================
# AUTO-RELAUNCH UNDER ./qwen_env IF AVAILABLE
# ==========================================
# Mirror qwen_dictation.py: if a bundled venv exists at ./qwen_env, re-exec
# under it so users running `python3 download_model.py` with the system
# Python (which lacks our deps) get the right environment automatically.
def _maybe_relaunch_in_venv() -> None:
    if os.environ.get("QWEN_DICTATION_RELAUNCHED") == "1":
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        venv_python = os.path.join(script_dir, "qwen_env", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(script_dir, "qwen_env", "bin", "python")
    if not os.path.isfile(venv_python):
        return
    try:
        same = os.path.samefile(venv_python, sys.executable)
    except OSError:
        same = False
    if same:
        return
    print(f"[System] Re-launching under bundled venv: {venv_python}", flush=True)
    env = os.environ.copy()
    env["QWEN_DICTATION_RELAUNCHED"] = "1"
    os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_maybe_relaunch_in_venv()

print("--- Qwen3 / Parakeet Model Downloader ---")
print("PyTorch (qwen-asr) — heavier RAM, full feature set:")
print("  1. Qwen3-ASR-0.6B                        (~1.2 GB download, ~2.5–3 GB RAM)")
print("  2. Qwen3-ASR-1.7B                        (~4.5 GB download, ~5–6 GB RAM)")
print("  3. Qwen3-ASR-0.6B + Qwen3-ForcedAligner  (~2.2 GB download)")
print("  4. Qwen3-ASR-1.7B + Qwen3-ForcedAligner  (~5.5 GB download)")
print("ONNX Runtime INT4 — minimal RAM, CPU-only, cross-platform:")
print("  5. Qwen3-ASR-0.6B INT4 ONNX              (~2.0 GB download, ~0.8–1.0 GB RAM)")
print("  6. Qwen3-ASR-1.7B INT4 ONNX              (~4.0 GB download, ~1.2–1.5 GB RAM)")
print("Parakeet TDT 0.6B v3 (English + 24 European langs):")
print("  7. Parakeet TDT v3 — MLX BF16            (~2.5 GB download, Apple Silicon only)")
print("  8. Parakeet TDT v3 — ONNX INT8           (~671 MB download, cross-platform)")

try:
    choice = input("Select Model Configuration to download (1-8): ").strip()
    if choice not in {"1", "2", "3", "4", "5", "6", "7", "8"}:
        print("Invalid choice. Using default: Qwen3-ASR-0.6B INT4 ONNX")
        choice = "5"
except KeyboardInterrupt:
    print("\n[System] Exiting...")
    sys.exit(0)


def _download_pytorch(choice: str) -> None:
    """Download the PyTorch (qwen-asr) checkpoints for options 1–4."""
    from huggingface_hub import snapshot_download

    asr_model_map = {
        "1": "Qwen/Qwen3-ASR-0.6B",
        "2": "Qwen/Qwen3-ASR-1.7B",
        "3": "Qwen/Qwen3-ASR-0.6B",
        "4": "Qwen/Qwen3-ASR-1.7B",
    }
    ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
    ASR_MODEL_ID = asr_model_map[choice]
    need_aligner = choice in {"3", "4"}

    repos = [ASR_MODEL_ID]
    if need_aligner:
        repos.append(ALIGNER_MODEL_ID)

    print(f"\nWill download: {', '.join(repos)}")
    print("This may take several minutes depending on your connection.\n")

    for repo_id in repos:
        print(f"Downloading {repo_id}...")
        cache_dir = snapshot_download(repo_id=repo_id, resume_download=True)
        print(f"[OK] {repo_id} -> {cache_dir}\n")

    # Quick load smoke-test to verify the checkpoint is usable.
    import torch
    from qwen_asr import Qwen3ASRModel

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch_dtype = torch.float16 if device == "mps" else torch.bfloat16

    load_kwargs = dict(
        dtype=torch_dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=100,
    )
    if need_aligner:
        load_kwargs["forced_aligner"] = ALIGNER_MODEL_ID
        load_kwargs["forced_aligner_kwargs"] = dict(
            dtype=torch_dtype, device_map=device
        )

    print("Loading model to verify...")
    Qwen3ASRModel.from_pretrained(ASR_MODEL_ID, **load_kwargs)
    print("[OK] Model loaded and verified successfully!")
    print(f"   ASR Model : {ASR_MODEL_ID}")
    if need_aligner:
        print(f"   Aligner   : {ALIGNER_MODEL_ID}")
    print(f"   Device    : {device}")
    print(f"   Dtype     : {torch_dtype}")


def _download_onnx(choice: str) -> None:
    """Download the INT4 ONNX bundle for options 5–6 and smoke-test it."""
    from qwen_onnx_backend import ONNX_REPO_MAP, Qwen3AsrOnnx, download_onnx_model
    import numpy as np

    size = "0.6B" if choice == "5" else "1.7B"
    repo = ONNX_REPO_MAP[size]
    print(f"\nWill download: {repo} (INT4 ONNX).")
    print("This may take several minutes depending on your connection.\n")

    cache_dir = download_onnx_model(size)
    print(f"[OK] ONNX bundle at: {cache_dir}")

    print("Loading ONNX pipeline to verify...")
    pipe = Qwen3AsrOnnx(cache_dir, num_threads=0)
    # Micro smoke-test: 0.5 s of silence; should run end-to-end without error.
    pipe.transcribe(
        np.zeros(8000, dtype=np.float32),
        language="English",
        max_new_tokens=4,
    )
    print("[OK] ONNX pipeline loaded and verified successfully!")
    print(f"   Repo   : {repo}")
    print(f"   Path   : {cache_dir}")
    print("   Runtime: onnxruntime (CPUExecutionProvider, INT4 MatMulNBits)")


def _download_parakeet(choice: str) -> None:
    """Download Parakeet TDT v3 (MLX or ONNX INT8) and smoke-test it."""
    from parakeet_backend import (
        MLX_REPO,
        ONNX_REPO,
        ParakeetAsr,
        download_parakeet,
        is_apple_silicon,
    )
    import numpy as np

    flavour = "mlx" if choice == "7" else "onnx-int8"

    # Fail fast on the wrong platform — saves the user a 2.5 GB download
    # they wouldn't be able to use.
    if flavour == "mlx" and not is_apple_silicon():
        print("[Error] Option 7 (Parakeet MLX) requires Apple Silicon (M1/M2/M3/M4).")
        print("        On Intel/AMD pick option 8 (Parakeet ONNX INT8) instead.")
        sys.exit(1)

    repo = MLX_REPO if flavour == "mlx" else ONNX_REPO
    print(f"\nWill download: {repo} ({flavour}).")
    print("This may take several minutes depending on your connection.\n")

    cache_dir = download_parakeet(flavour)
    print(f"[OK] Parakeet weights at: {cache_dir}")

    print("Loading Parakeet pipeline to verify...")
    pipe = ParakeetAsr(flavour=flavour, num_threads=0)
    pipe.transcribe(np.zeros(8000, dtype=np.float32))
    print("[OK] Parakeet pipeline loaded and verified successfully!")
    print(f"   Repo    : {repo}")
    print(f"   Path    : {cache_dir}")
    if flavour == "mlx":
        print("   Runtime : parakeet-mlx (Apple Silicon GPU, BF16)")
    else:
        print("   Runtime : onnxruntime + onnx-asr (CPUExecutionProvider, INT8)")


try:
    if choice in {"1", "2", "3", "4"}:
        _download_pytorch(choice)
    elif choice in {"5", "6"}:
        _download_onnx(choice)
    else:
        _download_parakeet(choice)
except Exception as e:
    print(f"[Error] {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
