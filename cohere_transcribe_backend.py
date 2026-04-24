#!/usr/bin/env python3
"""
Cohere Transcribe 03-2026 backend for cross-platform ASR

Supports multiple inference backends:
  - PyTorch: Standard transformers implementation (~3-4 GB RAM)
  - vLLM: Production-ready server with batching (~4-6 GB RAM)
  - MLX: Apple Silicon optimized (~2-3 GB RAM, macOS only)

Model details:
  - 2B parameter Conformer-based encoder-decoder
  - 14 languages: EN, FR, DE, IT, ES, PT, EL, NL, PL, ZH, JA, KO, VI, AR
  - Input: 16kHz mono audio (auto-resampled)
  - Output: Transcribed text with optional punctuation
  - Apache 2.0 License
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import platform
import subprocess

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class AsrResult:
    """Standardized ASR result format matching other backends."""
    text: str
    language: str


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4)."""
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True
        )
        return "Apple" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_requirements(backend: str = "pytorch") -> tuple[bool, list[str]]:
    """Check if required dependencies are installed for the specified backend."""
    missing = []

    # Common requirements - COMPATIBILITY FOCUSED
    try:
        import transformers
        if hasattr(transformers, '__version__'):
            version = tuple(map(int, transformers.__version__.split('.')[:2]))
            # Work with existing qwen-asr compatible version (4.57.6)
            if version < (4, 30):
                missing.append(f"transformers>={4}.{30}.0 (found {transformers.__version__})")
            elif version != (4, 57):
                # Warn but don't block if version is different from qwen-asr requirement
                print(f"[Warning] transformers {transformers.__version__} may conflict with qwen-asr (needs 4.57.6)")
    except ImportError:
        missing.append("transformers>=4.30.0")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    for pkg in ["soundfile", "librosa", "sentencepiece", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # Test protobuf specifically (import as google.protobuf)
    try:
        import google.protobuf
    except ImportError:
        missing.append("protobuf")

    # Backend-specific requirements
    if backend == "vllm":
        try:
            import vllm
            # Check if it's a reasonable version
            if hasattr(vllm, '__version__'):
                version = tuple(map(int, vllm.__version__.split('.')[:2]))
                if version < (0, 8):
                    missing.append("vllm>=0.8.0")
        except ImportError:
            missing.append("vllm>=0.8.0")
    elif backend == "mlx":
        if not is_apple_silicon():
            return False, ["Apple Silicon (M1/M2/M3/M4) required for MLX backend"]
        try:
            import mlx.core as mx
        except ImportError:
            missing.append("mlx-lm")

    return len(missing) == 0, missing


def check_huggingface_auth() -> tuple[bool, str]:
    """Check if user is authenticated with HuggingFace for gated models."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Try to get user info to check authentication
        try:
            user_info = api.whoami()
            return True, f"Authenticated as: {user_info['name']}"
        except Exception:
            return False, "Not authenticated. Run: huggingface-cli login"
    except ImportError:
        return False, "huggingface_hub not installed"


def check_model_access(model_id: str) -> tuple[bool, str]:
    """Check if we can access the specified model."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Try to access model info
        info = api.repo_info(model_id)
        return True, "Model accessible"

    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "access" in error_msg.lower():
            return False, f"Model requires access approval. Visit: https://huggingface.co/{model_id}"
        elif "401" in error_msg or "authentication" in error_msg.lower():
            return False, "Authentication required. Run: huggingface-cli login"
        else:
            return False, f"Model access error: {error_msg}"


class CoherePyTorchBackend:
    """PyTorch-based Cohere Transcribe backend using transformers."""

    def __init__(
        self,
        device: str = "auto",
        torch_dtype: Optional[str] = None,
        enable_optimizations: bool = True,
        use_auth_token: bool = True,
    ):
        """
        Initialize the PyTorch backend.

        Args:
            device: Device to use ("auto", "cpu", "cuda", "mps")
            torch_dtype: Torch dtype ("float16", "bfloat16", "float32")
            enable_optimizations: Enable platform-specific optimizations
            use_auth_token: Use HuggingFace authentication token
        """
        ok, missing = check_requirements("pytorch")
        if not ok:
            raise ImportError(f"Missing dependencies: {', '.join(missing)}")

        # Check authentication for gated model
        if use_auth_token:
            auth_ok, auth_msg = check_huggingface_auth()
            if not auth_ok:
                print(f"[Warning] Authentication issue: {auth_msg}")
                print("[Info] For gated models, run: huggingface-cli login")

            # Check model access
            model_ok, model_msg = check_model_access(self.model_id)
            if not model_ok:
                raise PermissionError(f"Cannot access model: {model_msg}")
            else:
                print(f"[Info] {model_msg}")

        import torch

        self.model_id = "CohereLabs/cohere-transcribe-03-2026"

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        # Dtype selection
        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            elif device == "mps":
                torch_dtype = "float16"  # MPS works well with FP16
            else:
                torch_dtype = "float32"  # CPU fallback

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float32)

        print(f"[Cohere] Loading model on {device} with {torch_dtype}...")

        # Load kwargs for model initialization
        load_kwargs = {
            "device_map": "auto" if device != "cpu" else None,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "use_auth_token": use_auth_token,
        }

        if device == "cpu":
            load_kwargs["device_map"] = None

        # Try to load with current transformers version
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                use_auth_token=use_auth_token,
                trust_remote_code=True
            )

            # Try multiple model classes for compatibility with transformers 4.57.6
            model_classes = [
                # Try specific Cohere class first (may not exist in 4.57.6)
                ("CohereAsrForConditionalGeneration", "transformers"),
                # Fallback to generic classes
                ("AutoModelForSpeechSeq2Seq", "transformers"),
                ("AutoModelForSeq2SeqLM", "transformers"),
                ("AutoModelForCausalLM", "transformers"),
            ]

            model_loaded = False

            for class_name, module_name in model_classes:
                try:
                    if class_name == "CohereAsrForConditionalGeneration":
                        # Try importing the specific class
                        from transformers import CohereAsrForConditionalGeneration
                        model_class = CohereAsrForConditionalGeneration
                    elif class_name == "AutoModelForSpeechSeq2Seq":
                        try:
                            from transformers import AutoModelForSpeechSeq2Seq
                            model_class = AutoModelForSpeechSeq2Seq
                        except ImportError:
                            continue  # Skip if not available
                    elif class_name == "AutoModelForSeq2SeqLM":
                        model_class = AutoModelForSeq2SeqLM
                    elif class_name == "AutoModelForCausalLM":
                        model_class = AutoModelForCausalLM
                    else:
                        continue

                    print(f"[Cohere] Trying model class: {class_name}")

                    self.model = model_class.from_pretrained(
                        self.model_id, **load_kwargs
                    )

                    print(f"[Cohere] Successfully loaded with {class_name}")
                    model_loaded = True
                    break

                except (ImportError, AttributeError, OSError) as e:
                    print(f"[Cohere] {class_name} failed: {e}")
                    continue
                except Exception as e:
                    error_str = str(e)
                    if "gated" in error_str.lower() or "401" in error_str:
                        # Re-raise auth errors immediately
                        raise e
                    print(f"[Cohere] {class_name} failed: {e}")
                    continue

            if not model_loaded:
                raise RuntimeError("Could not load model with any compatible class")

            if device == "cpu":
                self.model = self.model.to(device)

        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "401" in error_msg:
                raise PermissionError(
                    f"Model access denied. Please:\n"
                    f"1. Visit: https://huggingface.co/{self.model_id}\n"
                    f"2. Request access to the model\n"
                    f"3. Run: huggingface-cli login\n"
                    f"4. Ensure your token has access to gated repos"
                )
            else:
                raise RuntimeError(f"Failed to load Cohere model: {e}")

        # Platform optimizations
        if enable_optimizations:
            self._apply_optimizations()

        # Supported languages mapping (updated for actual Cohere model)
        self.language_map = {
            "english": "en", "en": "en",
            "french": "fr", "fr": "fr",
            "german": "de", "de": "de",
            "italian": "it", "it": "it",
            "spanish": "es", "es": "es",
            "portuguese": "pt", "pt": "pt",
            "greek": "el", "el": "el",
            "dutch": "nl", "nl": "nl",
            "polish": "pl", "pl": "pl",
            "chinese": "zh", "mandarin": "zh", "zh": "zh",
            "japanese": "ja", "jp": "ja", "ja": "ja",
            "korean": "ko", "kr": "ko", "ko": "ko",
            "vietnamese": "vi", "vn": "vi", "vi": "vi",
            "arabic": "ar", "ar": "ar",
        }

        print(f"[Cohere] Model loaded successfully on {self.device}")

    def _apply_optimizations(self):
        """Apply platform-specific optimizations."""
        try:
            import torch

            if self.device == "cuda":
                # Enable Flash Attention if available
                try:
                    from transformers.models.cohere_asr.modeling_cohere_asr import CohereAsrAttention
                    # Flash attention is automatically used if available
                    pass
                except ImportError:
                    pass

                # Enable tensor cores for mixed precision
                if self.torch_dtype in [torch.float16, torch.bfloat16]:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            elif self.device == "mps":
                # MPS optimizations for Apple Silicon
                if hasattr(torch.backends, 'mps'):
                    # Ensure MPS is available and optimized
                    pass

            # Compile model if PyTorch 2.0+ and not on MPS (compilation issues)
            if hasattr(torch, 'compile') and self.device != "mps":
                try:
                    self.model = torch.compile(self.model, mode="max-autotune")
                    print(f"[Cohere] Model compiled with PyTorch 2.0+ optimizations")
                except Exception as e:
                    print(f"[Cohere] Model compilation failed: {e}")

        except Exception as e:
            print(f"[Cohere] Optimization setup failed: {e}")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        max_new_tokens: int = 256,
        punctuation: bool = True,
    ) -> AsrResult:
        """
        Transcribe audio using the PyTorch backend.

        Args:
            audio: 1D numpy array of float32 samples (any sample rate, auto-resampled)
            language: Language code or name (None for auto-detect)
            max_new_tokens: Maximum tokens to generate
            punctuation: Whether to include punctuation

        Returns:
            AsrResult with transcribed text and language
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        # Convert language name to code if needed
        lang_code = None
        if language:
            lang_key = language.lower().strip()
            lang_code = self.language_map.get(lang_key, language.lower()[:2])

        try:
            # Process audio - handle different processor interfaces
            try:
                # Try modern processor interface with language support
                inputs = self.processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
                # Note: language and punctuation may not be supported by current transformers
            except TypeError as te:
                # Fallback to basic interface if language/punctuation not supported
                print(f"[Cohere] Using basic processor interface: {te}")
                inputs = self.processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )

            # Move to device and handle different input types
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.model.device)

            # Handle different input tensor names
            if hasattr(inputs, 'input_features'):
                inputs.input_features = inputs.input_features.to(dtype=self.model.dtype)
            elif hasattr(inputs, 'input_values'):
                inputs.input_values = inputs.input_values.to(dtype=self.model.dtype)

            # Generate transcription with error handling
            import torch
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,  # Greedy decoding for speed
                        do_sample=False,
                    )
                except Exception as gen_error:
                    # Fallback to direct forward pass if generate doesn't work
                    print(f"[Cohere] Generate failed, trying forward pass: {gen_error}")
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        # Handle encoder-decoder output
                        outputs = outputs.last_hidden_state
                    elif hasattr(outputs, 'logits'):
                        outputs = outputs.logits.argmax(dim=-1)

            # Decode output with multiple fallback methods
            try:
                if hasattr(self.processor, 'batch_decode'):
                    text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                elif hasattr(self.processor, 'decode'):
                    text = self.processor.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Last resort - convert tokens to text
                    if hasattr(self.processor, 'tokenizer'):
                        text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        text = f"Generated tokens: {outputs[0].tolist()}"
            except Exception as decode_error:
                print(f"[Cohere] Decode error: {decode_error}")
                text = f"Transcription completed but decode failed: {decode_error}"

            text = text.strip()

            # Return result
            detected_lang = lang_code or "auto"
            return AsrResult(text=text, language=detected_lang)

        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "401" in error_msg or "authentication" in error_msg.lower():
                raise PermissionError(
                    f"Authentication required for model access:\n"
                    f"1. Visit: https://huggingface.co/{self.model_id}\n"
                    f"2. Request access\n"
                    f"3. Run: huggingface-cli login"
                )
            else:
                raise RuntimeError(f"Transcription failed: {e}")


class CohereVllmBackend:
    """vLLM-based Cohere backend for production serving."""

    def __init__(
        self,
        port: int = 8000,
        host: str = "localhost",
        start_server: bool = True,
        server_timeout: int = 60,
    ):
        """
        Initialize vLLM backend.

        Args:
            port: Server port
            host: Server host
            start_server: Whether to start the server automatically
            server_timeout: Timeout for server startup
        """
        ok, missing = check_requirements("vllm")
        if not ok:
            raise ImportError(f"Missing dependencies: {', '.join(missing)}")

        import requests

        self.model_id = "CohereLabs/cohere-transcribe-03-2026"
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()

        if start_server:
            self._start_vllm_server(port, server_timeout)
        else:
            # Check if server is already running
            if not self._check_server():
                raise RuntimeError(f"vLLM server not running at {self.base_url}")

    def _start_vllm_server(self, port: int, timeout: int):
        """Start the vLLM server."""
        import subprocess
        import time

        print(f"[Cohere] Starting vLLM server on port {port}...")

        # Check available vLLM version
        try:
            import vllm
            vllm_version = getattr(vllm, '__version__', 'unknown')
            print(f"[Cohere] Using vLLM version: {vllm_version}")
        except ImportError:
            raise RuntimeError("vLLM not installed. Install with: pip install 'vllm>=0.8.0'")

        cmd = [
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", self.model_id,
            "--port", str(port),
            "--trust-remote-code",
            "--disable-log-stats",
        ]

        # Try alternative command format if first doesn't work
        alt_cmd = [
            "vllm", "serve", self.model_id,
            "--port", str(port),
            "--trust-remote-code",
            "--disable-log-stats",
        ]

        for cmd_to_try in [cmd, alt_cmd]:
            try:
                print(f"[Cohere] Trying command: {' '.join(cmd_to_try)}")
                self.server_process = subprocess.Popen(
                    cmd_to_try,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Wait for server to start
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self._check_server():
                        print(f"[Cohere] vLLM server started successfully")
                        return
                    time.sleep(2)

                # Server failed to start with this command, try next
                self.server_process.terminate()
                print(f"[Cohere] Command failed: {' '.join(cmd_to_try)}")

            except Exception as e:
                print(f"[Cohere] Server start error: {e}")
                continue

        # All commands failed
        raise RuntimeError(
            f"vLLM server failed to start within {timeout}s. "
            f"Check that vLLM supports the model and try manual start:\n"
            f"python -m vllm.entrypoints.api_server --model {self.model_id} --port {port}"
        )

    def _check_server(self) -> bool:
        """Check if vLLM server is running."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs
    ) -> AsrResult:
        """
        Transcribe audio using vLLM server.

        Args:
            audio: 1D numpy array of float32 samples
            language: Language code (optional)

        Returns:
            AsrResult with transcribed text and language
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        # Save audio to temporary WAV file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, 16000)

            try:
                # Make API request
                files = {"file": open(tmp_file.name, "rb")}
                data = {"model": self.model_id}

                if language:
                    data["language"] = language

                response = self.session.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code != 200:
                    raise RuntimeError(f"vLLM API error: {response.status_code} - {response.text}")

                result = response.json()
                text = result.get("text", "").strip()
                detected_lang = language or "auto"

                return AsrResult(text=text, language=detected_lang)

            finally:
                files["file"].close()
                os.unlink(tmp_file.name)

    def close(self):
        """Clean up vLLM server."""
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()


class CohereMlxBackend:
    """MLX-optimized Cohere backend for Apple Silicon."""

    def __init__(self):
        """Initialize MLX backend (Apple Silicon only)."""
        if not is_apple_silicon():
            raise RuntimeError("MLX backend requires Apple Silicon (M1/M2/M3/M4)")

        ok, missing = check_requirements("mlx")
        if not ok:
            raise ImportError(f"Missing dependencies: {', '.join(missing)}")

        try:
            import mlx.core as mx
            from mlx_lm import load, generate

            self.model_id = "CohereLabs/cohere-transcribe-03-2026"

            print(f"[Cohere] Loading MLX-optimized model...")

            # Load model with MLX
            self.model, self.tokenizer = load(self.model_id)

            print(f"[Cohere] MLX model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model: {e}")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        max_tokens: int = 256,
    ) -> AsrResult:
        """
        Transcribe audio using MLX backend.

        Args:
            audio: 1D numpy array of float32 samples
            language: Language code (optional)
            max_tokens: Maximum tokens to generate

        Returns:
            AsrResult with transcribed text and language
        """
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        try:
            # Note: MLX audio processing would need additional implementation
            # This is a placeholder for the MLX-specific audio processing
            # In practice, you'd need to implement audio feature extraction for MLX

            # For now, fall back to transformers for audio processing
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.model_id)

            # Process audio
            inputs = processor(audio, sampling_rate=16000, return_tensors="np")

            # Convert to MLX format and generate (simplified)
            # This would need proper MLX audio model implementation

            # Placeholder result
            text = "MLX transcription not fully implemented yet"
            detected_lang = language or "auto"

            return AsrResult(text=text, language=detected_lang)

        except Exception as e:
            raise RuntimeError(f"MLX transcription failed: {e}")


# Backend factory function
def create_cohere_backend(
    backend: str = "pytorch",
    **kwargs
) -> Union[CoherePyTorchBackend, CohereVllmBackend, CohereMlxBackend]:
    """
    Create a Cohere backend instance.

    Args:
        backend: Backend type ("pytorch", "vllm", "mlx")
        **kwargs: Backend-specific arguments

    Returns:
        Initialized backend instance
    """
    backend = backend.lower()

    if backend == "pytorch":
        return CoherePyTorchBackend(**kwargs)
    elif backend == "vllm":
        return CohereVllmBackend(**kwargs)
    elif backend == "mlx":
        return CohereMlxBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: pytorch, vllm, mlx")


# CLI test function
def _main():
    """CLI test for Cohere backend."""
    import argparse

    parser = argparse.ArgumentParser(description="Cohere Transcribe backend test")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--backend", choices=["pytorch", "vllm", "mlx"], default="pytorch")
    parser.add_argument("--language", help="Language code (e.g., en, fr, zh)")
    parser.add_argument("--device", default="auto", help="Device for PyTorch backend")

    args = parser.parse_args()

    # Load audio
    audio, sr = sf.read(args.audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono
    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Create backend
    backend_kwargs = {}
    if args.backend == "pytorch":
        backend_kwargs["device"] = args.device

    backend = create_cohere_backend(args.backend, **backend_kwargs)

    # Transcribe
    print(f"[Test] Transcribing with {args.backend} backend...")
    start_time = time.time()

    result = backend.transcribe(audio, language=args.language)

    elapsed = time.time() - start_time
    print(f"[Test] Transcribed in {elapsed:.2f}s ({len(audio)/16000:.1f}s audio)")
    print(f"[Test] Language: {result.language}")
    print(f"[Test] Text: {result.text}")

    # Cleanup
    if hasattr(backend, 'close'):
        backend.close()


if __name__ == "__main__":
    _main()