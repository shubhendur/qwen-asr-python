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

print("--- Qwen3 / Parakeet / Voxtral / Whisper Model Downloader ---")
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
print("Voxtral-4B Realtime (13 languages, real-time streaming ASR):")
print("  9. Voxtral-4B — vLLM Realtime           (~17.7 GB download, GPU required)")
print(" 10. Voxtral-4B — MLX Optimized           (~17.7 GB download, Apple Silicon)")
print(" 11. Voxtral-4B — ExecuTorch Lite         (~17.7 GB download, CPU-only)")
print("Whisper ASR (OpenAI, 99 languages, excellent accuracy):")
print(" 12. Whisper-Tiny (CPU Optimized)        (~39 MB download, ~1 GB RAM)")
print(" 13. Whisper-Base (Balanced)             (~142 MB download, ~1-2 GB RAM)")
print(" 14. Whisper-Small (Desktop Standard)    (~967 MB download, ~2-3 GB RAM)")
print(" 15. Whisper-Medium (High Quality)       (~3.1 GB download, ~3-5 GB RAM)")
print(" 16. Whisper-Turbo (GPU Accelerated)     (~1.6 GB download, ~4-6 GB RAM)")

try:
    choice = input("Select Model Configuration to download (1-16): ").strip()
    if choice not in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"}:
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


def _download_voxtral(choice: str) -> None:
    """Download Voxtral-4B model for the specified backend and smoke-test it."""
    import numpy as np

    # Map choices to backends
    backend_map = {
        "9": ("vllm", "Voxtral vLLM"),
        "10": ("mlx", "Voxtral MLX"),
        "11": ("executorch", "Voxtral ExecuTorch")
    }

    backend_type, backend_name = backend_map[choice]

    # Platform compatibility checks
    if backend_type == "mlx":
        from parakeet_backend import is_apple_silicon
        if not is_apple_silicon():
            print("[Error] Option 10 (Voxtral MLX) requires Apple Silicon (M1/M2/M3/M4).")
            print("        On Intel/AMD pick option 9 (vLLM) or 11 (ExecuTorch) instead.")
            sys.exit(1)

    print(f"\nWill download: mistralai/Voxtral-Mini-4B-Realtime-2602 ({backend_name}).")
    print("This is a large model (~17.7 GB) and may take significant time.\n")

    try:
        if backend_type == "vllm":
            from voxtral_vllm_backend import download_voxtral_vllm
            cache_dir = download_voxtral_vllm()
            print("Loading Voxtral vLLM backend to verify...")
            from voxtral_vllm_backend import VoxtralVllmBackend
            backend = VoxtralVllmBackend(delay_ms=480)

        elif backend_type == "mlx":
            from voxtral_mlx_backend import download_voxtral_mlx
            cache_dir = download_voxtral_mlx()
            print("Loading Voxtral MLX backend to verify...")
            from voxtral_mlx_backend import VoxtralMlxBackend
            backend = VoxtralMlxBackend(delay_ms=480, memory_limit_gb=4)

        elif backend_type == "executorch":
            from voxtral_executorch_backend import download_voxtral_executorch
            cache_dir = download_voxtral_executorch()
            print("Loading Voxtral ExecuTorch backend to verify...")
            from voxtral_executorch_backend import VoxtralExecuTorchBackend
            backend = VoxtralExecuTorchBackend(num_threads=4, memory_limit_mb=2048)

        print(f"[OK] Voxtral model cached at: {cache_dir}")

        # Smoke test with short silence
        print("Running smoke test...")
        test_audio = np.zeros(8000, dtype=np.float32)  # 0.5 seconds
        result = backend.transcribe(test_audio, language="English")

        print("[OK] Voxtral backend loaded and verified successfully!")
        print(f"   Backend : {backend_name}")
        print(f"   Path    : {cache_dir}")
        print(f"   Model   : mistralai/Voxtral-Mini-4B-Realtime-2602")
        if backend_type == "vllm":
            print("   Runtime : vLLM (GPU-accelerated, real-time streaming)")
        elif backend_type == "mlx":
            print("   Runtime : MLX (Apple Silicon optimized, unified memory)")
        else:
            print("   Runtime : ExecuTorch (CPU-optimized, minimal resources)")

        # Clean up backend resources
        if hasattr(backend, 'close'):
            backend.close()

    except ImportError as e:
        print(f"[Error] Missing dependencies for {backend_name}: {e}")
        print("       Run: pip install -r requirements-voxtral.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to verify {backend_name}: {e}")
        sys.exit(1)


def _download_whisper(choice: str) -> None:
    """Download Whisper model for the specified size and smoke-test it."""
    from whisper_pytorch_backend import WhisperPyTorchBackend, WHISPER_MODELS, download_whisper_model
    import numpy as np

    # Map choices to model sizes
    model_size_map = {
        "12": "tiny",
        "13": "base",
        "14": "small",
        "15": "medium",
        "16": "large-v3-turbo"
    }

    model_size = model_size_map[choice]
    model_config = WHISPER_MODELS[model_size]

    print(f"\nWill download: {model_config['repo_id']} ({model_config['params']} parameters).")
    print(f"Description: {model_config['description']}")
    print(f"Expected download size varies by model. This may take several minutes.\n")

    try:
        # Download the model
        cache_dir = download_whisper_model(model_size)
        print(f"[OK] Whisper model cached at: {cache_dir}")

        # Load and verify the backend
        print(f"Loading Whisper {model_size} backend to verify...")
        backend = WhisperPyTorchBackend(
            model_size=model_size,
            device="auto",
            enable_optimizations=True
        )

        # Get system and model information
        model_info = backend.get_model_info()
        system_resources = backend.get_system_resources()

        print(f"[OK] Whisper {model_size} backend loaded successfully!")
        print(f"   Model     : {model_info['model_size']} ({model_info['parameters']})")
        print(f"   Repository: {model_info['repository']}")
        print(f"   Device    : {model_info['device']}")
        print(f"   Dtype     : {model_info['dtype']}")
        print(f"   Batch size: {model_info['batch_size']}")

        if "error" not in system_resources:
            memory_used = system_resources.get('process_memory_gb', 0)
            memory_available = system_resources.get('memory_available_gb', 0)
            print(f"   Memory    : {memory_used:.1f}GB used, {memory_available:.1f}GB available")

        # Smoke test with short audio
        print("Running smoke test...")
        test_audio = np.zeros(8000, dtype=np.float32)  # 0.5 seconds of silence
        result = backend.transcribe(test_audio, language="en")

        print("[OK] Smoke test completed successfully!")

        # Show optimization information
        optimizations = []
        if model_info.get('optimizations_enabled'):
            optimizations.append("Platform optimizations")
        if hasattr(backend, '_torch_compile_available') and backend._torch_compile_available():
            optimizations.append("torch.compile available")
        if hasattr(backend, '_flash_attention_available') and backend._flash_attention_available():
            optimizations.append("Flash Attention 2")

        if optimizations:
            print(f"   Features  : {', '.join(optimizations)}")

        print(f"\nWhisper {model_size} is ready for use with options 12-16 in qwen_dictation.py")

        # Clean up
        backend.cleanup()

    except ImportError as e:
        print(f"[Error] Missing Whisper dependencies: {e}")
        print("       Run: pip install -r requirements-whisper.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to verify Whisper {model_size}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


try:
    if choice in {"1", "2", "3", "4"}:
        _download_pytorch(choice)
    elif choice in {"5", "6"}:
        _download_onnx(choice)
    elif choice in {"7", "8"}:
        _download_parakeet(choice)
    elif choice in {"9", "10", "11"}:
        _download_voxtral(choice)
    else:  # choices 12, 13, 14, 15, 16
        _download_whisper(choice)
except Exception as e:
    print(f"[Error] {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
