#!/usr/bin/env python3
"""
OpenAI Whisper Multi-Model Backend with Platform-Specific Optimizations.

This backend provides comprehensive Whisper ASR support with five model sizes:
- tiny (39M parameters): Ultra-lightweight, real-time capable
- base (74M parameters): Balanced speed vs accuracy
- small (244M parameters): Good desktop experience
- medium (769M parameters): High-quality applications
- large-v3-turbo (809M parameters): Near-maximum quality with optimized speed

Platform Optimizations:
- Windows 11 Intel i7 13th Gen: Thread management, MKL optimizations, hybrid architecture support
- Mac M1 Pro: MPS acceleration, unified memory optimization, Apple Silicon features
- Automatic resource detection and configuration
- Advanced optimizations: torch.compile, Flash Attention 2, chunked processing

Repository: openai/whisper-{model_size}
Framework:  transformers pipeline with platform-specific optimizations
Languages:  99 languages supported
Memory:     1GB (tiny) to 6GB (large-v3-turbo)

Usage:
    from whisper_pytorch_backend import WhisperPyTorchBackend
    backend = WhisperPyTorchBackend(model_size="small", device="auto")
    result = backend.transcribe(audio_float32_16k, language="English")
    print(result.text)
"""

from __future__ import annotations

import gc
import os
import platform
import psutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import numpy as np


# Whisper model configurations
WHISPER_MODELS = {
    "tiny": {
        "params": "39M",
        "memory_gb": 1,
        "repo_id": "openai/whisper-tiny",
        "description": "Ultra-lightweight, real-time capable"
    },
    "base": {
        "params": "74M",
        "memory_gb": 2,
        "repo_id": "openai/whisper-base",
        "description": "Balanced speed vs accuracy"
    },
    "small": {
        "params": "244M",
        "memory_gb": 3,
        "repo_id": "openai/whisper-small",
        "description": "Good desktop experience"
    },
    "medium": {
        "params": "769M",
        "memory_gb": 5,
        "repo_id": "openai/whisper-medium",
        "description": "High-quality applications"
    },
    "large-v3-turbo": {
        "params": "809M",
        "memory_gb": 6,
        "repo_id": "openai/whisper-large-v3-turbo",
        "description": "Near-maximum quality with optimized speed"
    }
}

# Whisper language codes (subset of 99 supported languages)
WHISPER_LANGUAGES = {
    "en": "english", "hi": "hindi", "es": "spanish", "fr": "french",
    "de": "german", "it": "italian", "pt": "portuguese", "ru": "russian",
    "ja": "japanese", "ko": "korean", "zh": "chinese", "ar": "arabic",
    "nl": "dutch", "pl": "polish", "tr": "turkish", "sv": "swedish"
}

SAMPLE_RATE = 16000


@dataclass
class AsrResult:
    text: str
    language: str = ""


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def detect_system_resources() -> Dict[str, Any]:
    """Detect available system resources for optimal configuration."""
    try:
        # Memory detection
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        total_memory_gb = memory.total / (1024**3)

        # CPU detection
        cpu_count = os.cpu_count() or 4
        cpu_info = platform.processor()

        # Platform detection
        is_windows = sys.platform == "win32"
        is_macos = sys.platform == "darwin"
        is_apple_cpu = is_apple_silicon()
        is_intel = "intel" in cpu_info.lower() or "i7" in cpu_info.lower()

        # GPU detection
        has_cuda = False
        has_mps = False
        cuda_memory_gb = 0

        try:
            import torch
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

            if has_cuda:
                cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass

        return {
            "available_memory_gb": available_memory_gb,
            "total_memory_gb": total_memory_gb,
            "cpu_count": cpu_count,
            "is_windows": is_windows,
            "is_macos": is_macos,
            "is_apple_cpu": is_apple_cpu,
            "is_intel": is_intel,
            "has_cuda": has_cuda,
            "has_mps": has_mps,
            "cuda_memory_gb": cuda_memory_gb
        }
    except Exception as e:
        print(f"[Whisper] Warning: Could not detect system resources: {e}")
        return {"available_memory_gb": 4, "cpu_count": 4, "has_cuda": False, "has_mps": False}


class WhisperPyTorchBackend:
    """
    OpenAI Whisper backend with comprehensive platform optimizations.

    Supports five model sizes with automatic resource detection and
    platform-specific optimizations for Windows Intel and Mac Apple Silicon.
    """

    def __init__(
        self,
        model_size: str = "auto",
        device: str = "auto",
        enable_optimizations: bool = True,
        batch_size: Optional[int] = None,
        chunk_length_s: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Whisper PyTorch backend.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3-turbo, auto)
            device: Device to use (auto, cpu, cuda, mps)
            enable_optimizations: Enable platform-specific optimizations
            batch_size: Batch size for processing (auto-detected if None)
            chunk_length_s: Chunk length for long audio (30 for chunked processing)
            **kwargs: Additional arguments passed to pipeline
        """
        self.system_resources = detect_system_resources()
        self.enable_optimizations = enable_optimizations

        # Configure model size and device
        self.model_size = self._select_optimal_model_size(model_size)
        self.device = self._select_optimal_device(device)
        self.torch_dtype = self._select_optimal_dtype()

        # Configure processing parameters
        self.batch_size = batch_size or self._select_optimal_batch_size()
        self.chunk_length_s = chunk_length_s

        # Apply platform-specific optimizations
        if enable_optimizations:
            self._apply_platform_optimizations()

        # Initialize pipeline (lazy loading)
        self._pipeline = None
        self._model_loaded = False

    def _select_optimal_model_size(self, model_size: str) -> str:
        """Select optimal model size based on available resources."""
        if model_size != "auto":
            if model_size in WHISPER_MODELS:
                return model_size
            else:
                print(f"[Whisper] Warning: Unknown model size '{model_size}', using auto-detection")

        available_memory = self.system_resources["available_memory_gb"]

        # Conservative memory-based selection
        if available_memory < 1.5:
            return "tiny"
        elif available_memory < 3:
            return "base"
        elif available_memory < 4:
            return "small"
        elif available_memory < 7:
            return "medium"
        else:
            return "large-v3-turbo"

    def _select_optimal_device(self, device: str) -> str:
        """Select optimal device based on available hardware."""
        if device != "auto":
            return device

        # Prioritize GPU acceleration when available
        if self.system_resources["has_cuda"]:
            return "cuda"
        elif self.system_resources["has_mps"]:
            return "mps"
        else:
            return "cpu"

    def _select_optimal_dtype(self) -> str:
        """Select optimal data type based on device and platform."""
        if self.device == "cuda":
            return "float16"  # GPU memory efficiency
        elif self.device == "mps":
            return "float16"  # Apple Silicon native support
        else:
            return "float32"  # CPU compatibility

    def _select_optimal_batch_size(self) -> int:
        """Select optimal batch size based on device and memory."""
        if self.device == "mps":
            # Apple Silicon with unified memory can handle larger batches
            return 8
        elif self.device == "cuda":
            # GPU memory dependent
            cuda_memory = self.system_resources.get("cuda_memory_gb", 0)
            if cuda_memory >= 8:
                return 8
            elif cuda_memory >= 4:
                return 4
            else:
                return 2
        else:
            # CPU processing
            return 2

    def _apply_platform_optimizations(self) -> None:
        """Apply platform-specific optimizations."""
        if self.system_resources["is_windows"] and self.system_resources["is_intel"]:
            self._apply_windows_intel_optimizations()
        elif self.system_resources["is_apple_cpu"]:
            self._apply_apple_silicon_optimizations()
        else:
            self._apply_generic_optimizations()

    def _apply_windows_intel_optimizations(self) -> None:
        """Optimizations for Windows 11 Intel i7 13th gen (hybrid architecture)."""
        cpu_count = self.system_resources["cpu_count"]

        # Intel i7 13th gen: 8 P-cores + 8 E-cores = 16 threads
        # Use ~75% for optimal performance without overwhelming system
        optimal_threads = min(12, max(4, int(cpu_count * 0.75)))

        # Intel MKL optimizations
        os.environ.setdefault("OMP_NUM_THREADS", str(optimal_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(optimal_threads))
        os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
        os.environ.setdefault("KMP_BLOCKTIME", "1")
        os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

        # Memory allocator optimization
        os.environ.setdefault("MALLOC_CONF", "background_thread:true,metadata_thp:auto")

        print(f"[Whisper] Applied Windows Intel optimizations: {optimal_threads} threads")

    def _apply_apple_silicon_optimizations(self) -> None:
        """Optimizations for Mac M1 Pro (Apple Silicon)."""
        # Apple Silicon optimizations
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(self.system_resources["cpu_count"]))

        # Let macOS manage threading (it's optimized for Apple Silicon)
        # Don't set OMP_NUM_THREADS - macOS scheduler is better

        print(f"[Whisper] Applied Apple Silicon optimizations: MPS enabled, unified memory")

    def _apply_generic_optimizations(self) -> None:
        """Generic optimizations for other platforms."""
        cpu_count = self.system_resources["cpu_count"]
        optimal_threads = max(1, min(cpu_count - 1, 8))  # Leave one core free

        os.environ.setdefault("OMP_NUM_THREADS", str(optimal_threads))
        os.environ.setdefault("OMP_PROC_BIND", "true")
        os.environ.setdefault("OMP_PLACES", "cores")

        print(f"[Whisper] Applied generic optimizations: {optimal_threads} threads")

    def _load_pipeline(self) -> None:
        """Load Whisper pipeline with optimizations."""
        if self._model_loaded:
            return

        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            raise ImportError(
                "Whisper backend requires transformers and torch. Install with:\n"
                "  pip install transformers torch datasets[audio]\n"
                "Or install the full requirements:\n"
                "  pip install -r requirements-whisper.txt"
            ) from e

        model_config = WHISPER_MODELS[self.model_size]
        model_id = model_config["repo_id"]

        print(f"[Whisper] Loading {model_config['params']} model: {model_id}")
        print(f"[Whisper] Device: {self.device}, Dtype: {self.torch_dtype}")
        print(f"[Whisper] Expected memory usage: ~{model_config['memory_gb']}GB")

        # Configure torch settings
        if hasattr(torch, 'set_num_threads'):
            if not self.system_resources["is_apple_cpu"]:
                # Only set threads on non-Apple platforms
                optimal_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
                torch.set_num_threads(optimal_threads)

        # Build pipeline configuration
        pipeline_kwargs = {
            "model": model_id,
            "torch_dtype": getattr(torch, self.torch_dtype),
            "device": self.device,
            "model_kwargs": {
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
            }
        }

        # Add chunked processing if specified
        if self.chunk_length_s:
            pipeline_kwargs.update({
                "chunk_length_s": self.chunk_length_s,
                "stride_length_s": 5,  # Overlap for continuity
            })

        # Advanced optimizations
        if self.enable_optimizations:
            # Flash Attention 2 for compatible GPUs
            if self.device in ["cuda", "mps"] and self._flash_attention_available():
                pipeline_kwargs["model_kwargs"]["attn_implementation"] = "flash_attention_2"
                print("[Whisper] Enabled Flash Attention 2")

        # Create pipeline
        self._pipeline = pipeline("automatic-speech-recognition", **pipeline_kwargs)

        # Explicitly override generation config to fix max_new_tokens issue
        if hasattr(self._pipeline.model, 'generation_config'):
            self._pipeline.model.generation_config.max_new_tokens = 256
            self._pipeline.model.generation_config.max_length = None  # Use max_new_tokens instead
            print(f"[Whisper] Set model generation config: max_new_tokens=256")

        # Apply torch.compile optimization (4.5x speedup on compatible hardware)
        if self._torch_compile_available():
            try:
                self._pipeline.model.generation_config.cache_implementation = "static"
                self._pipeline.model.generation_config.max_new_tokens = 256  # Keep consistent with transcribe method
                self._pipeline.model.forward = torch.compile(
                    self._pipeline.model.forward,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                print("[Whisper] Enabled torch.compile optimization")
            except Exception as e:
                print(f"[Whisper] torch.compile failed: {e}")

        self._model_loaded = True
        print(f"[Whisper] Model loaded successfully")

    def _flash_attention_available(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def _torch_compile_available(self) -> bool:
        """Check if torch.compile is available and compatible."""
        try:
            import torch
            if not hasattr(torch, 'compile'):
                return False

            # torch.compile is not compatible with chunked processing
            if self.chunk_length_s:
                return False

            # Check for compatible hardware
            if self.device == "cuda":
                return True
            elif self.device == "mps":
                # torch.compile on MPS is experimental
                return False
            else:
                # CPU torch.compile is slow
                return False

        except Exception:
            return False

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        return_timestamps: bool = False,
        **kwargs
    ) -> AsrResult:
        """
        Transcribe audio using Whisper model.

        Args:
            audio: 1D float32 numpy array of audio samples at 16kHz
            language: Language code (en, hi, es, etc.) or full name, None for auto-detect
            return_timestamps: Whether to return word-level timestamps
            **kwargs: Additional arguments passed to pipeline

        Returns:
            AsrResult with transcribed text and detected language
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        # Load pipeline lazily
        self._load_pipeline()

        # Process language specification
        generate_kwargs = {}
        if language:
            # Convert language codes to full names if needed
            if language in WHISPER_LANGUAGES:
                generate_kwargs["language"] = WHISPER_LANGUAGES[language]
            else:
                # Assume it's already a full language name
                generate_kwargs["language"] = language.lower()

        # Configure generation parameters using newer API
        generate_kwargs.update({
            "max_new_tokens": 256,  # Reduced from 448 to leave room for prompt tokens
            "num_beams": 1,  # Greedy decoding for speed
            "do_sample": False,  # Deterministic output
            "temperature": 1.0,  # Keep at 1.0 when do_sample=False
            "use_cache": True,
            "pad_token_id": 50257,  # Whisper's pad token
            "bos_token_id": 50258,  # Whisper's beginning of sequence token
            "eos_token_id": 50257,  # Whisper's end of sequence token
        })

        # Add user-provided kwargs
        generate_kwargs.update(kwargs)

        # Debug: Print actual max_new_tokens being used
        actual_max_new_tokens = generate_kwargs.get("max_new_tokens", "not set")
        print(f"[Whisper Debug] Using max_new_tokens: {actual_max_new_tokens}")

        try:
            # Perform transcription
            start_time = time.time()

            # Use the newer API without deprecated parameters
            result = self._pipeline(
                audio,
                batch_size=self.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps,
                chunk_length_s=self.chunk_length_s
            )

            elapsed = time.time() - start_time
            audio_duration = len(audio) / SAMPLE_RATE
            rtf = elapsed / audio_duration if audio_duration > 0 else 0

            print(f"[Whisper] Transcribed {audio_duration:.1f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)")

            # Extract results
            text = result.get("text", "").strip()
            detected_language = result.get("language", "")

            return AsrResult(text=text, language=detected_language)

        except Exception as e:
            print(f"[Whisper] Pipeline transcription failed: {e}")
            print("[Whisper] Attempting fallback transcription method...")

            # Fallback to direct model usage
            try:
                return self._transcribe_direct(audio, language, return_timestamps)
            except Exception as e2:
                print(f"[Whisper] Fallback transcription also failed: {e2}")
                return AsrResult(text="", language="")

    def _transcribe_direct(self, audio: np.ndarray, language: Optional[str] = None, return_timestamps: bool = False) -> AsrResult:
        """Direct transcription using the model without pipeline wrapper."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch

            # Use the model directly
            model = self._pipeline.model
            processor = self._pipeline.feature_extractor
            tokenizer = self._pipeline.tokenizer

            # Process audio
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

            # Move to device
            if self.device != "cpu":
                inputs = inputs.to(self.device)

            # Generate with corrected parameters
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_features"],
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Decode result
            transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return AsrResult(text=transcription.strip(), language=language or "")

        except Exception as e:
            print(f"[Whisper] Direct transcription failed: {e}")
            return AsrResult(text="", language="")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        config = WHISPER_MODELS.get(self.model_size, {})
        return {
            "model_size": self.model_size,
            "parameters": config.get("params", "unknown"),
            "repository": config.get("repo_id", "unknown"),
            "device": self.device,
            "dtype": self.torch_dtype,
            "batch_size": self.batch_size,
            "chunk_length_s": self.chunk_length_s,
            "expected_memory_gb": config.get("memory_gb", 0),
            "optimizations_enabled": self.enable_optimizations
        }

    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            # Current memory usage
            memory = psutil.virtual_memory()

            # Process-specific info
            process = psutil.Process()

            resources = {
                "memory_used_gb": (memory.total - memory.available) / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "cpu_percent": process.cpu_percent(),
                "process_memory_gb": process.memory_info().rss / (1024**3),
                "model_loaded": self._model_loaded
            }

            # GPU memory if available
            if self.device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        resources.update({
                            "gpu_memory_used_gb": gpu_memory,
                            "gpu_memory_total_gb": gpu_total,
                            "gpu_memory_percent": (gpu_memory / gpu_total) * 100
                        })
                except Exception:
                    pass

            return resources

        except Exception as e:
            return {"error": str(e)}

    def cleanup(self) -> None:
        """Clean up model resources and clear caches."""
        try:
            if self._pipeline is not None:
                del self._pipeline
                self._pipeline = None

            # Clear GPU caches
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif self.device == "mps":
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # Force garbage collection
            gc.collect()

            self._model_loaded = False
            print("[Whisper] Resources cleaned up")

        except Exception as e:
            print(f"[Whisper] Error during cleanup: {e}")

    def __del__(self) -> None:
        """Ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Download support for integration with download_model.py
# ----------------------------------------------------------------------------

def download_whisper_model(model_size: str) -> str:
    """Download Whisper model. Returns cache directory."""
    if model_size not in WHISPER_MODELS:
        raise ValueError(f"Unknown model size: {model_size}")

    from huggingface_hub import snapshot_download

    config = WHISPER_MODELS[model_size]
    repo_id = config["repo_id"]

    print(f"[Whisper] Downloading {config['params']} model: {repo_id}")
    cache_dir = snapshot_download(
        repo_id=repo_id,
        resume_download=True
    )
    print(f"[Whisper] Model cached at: {cache_dir}")
    return cache_dir


# ----------------------------------------------------------------------------
# CLI test: python whisper_pytorch_backend.py audio.wav [--model small] [--device auto]
# ----------------------------------------------------------------------------

def _main() -> int:
    """CLI smoke test for Whisper PyTorch backend."""
    import argparse
    import wave

    parser = argparse.ArgumentParser(description="Whisper PyTorch backend test")
    parser.add_argument("audio", help="Path to 16-bit mono WAV file")
    parser.add_argument("--model", default="small", choices=list(WHISPER_MODELS.keys()) + ["auto"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--language", help="Language code (en, hi, es, etc.)")
    parser.add_argument("--timestamps", action="store_true", help="Return timestamps")
    parser.add_argument("--chunk", type=int, help="Chunk length in seconds (for long audio)")
    args = parser.parse_args()

    # Load audio
    with wave.open(args.audio, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print("[Error] Expected 16-bit mono WAV", file=sys.stderr)
            return 1
        if wf.getframerate() != SAMPLE_RATE:
            print(f"[Error] Expected 16 kHz, got {wf.getframerate()}", file=sys.stderr)
            return 1
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Initialize backend
    backend = WhisperPyTorchBackend(
        model_size=args.model,
        device=args.device,
        chunk_length_s=args.chunk
    )

    try:
        # Print system info
        print(f"[Info] System resources: {backend.system_resources}")
        print(f"[Info] Model info: {backend.get_model_info()}")

        # Transcribe
        result = backend.transcribe(
            samples,
            language=args.language,
            return_timestamps=args.timestamps
        )

        print(f"[Result] Text: {result.text}")
        if result.language:
            print(f"[Result] Language: {result.language}")

        # Print resource usage
        resources = backend.get_system_resources()
        if "error" not in resources:
            print(f"[Info] Resource usage: {resources}")

    finally:
        backend.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(_main())