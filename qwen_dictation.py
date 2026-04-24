#!/usr/bin/env ./qwen_env/bin/python
"""
Global Qwen3 Voice-to-Text Utility
Using the official qwen-asr package for reliable speech recognition
"""

import os
import sys

# ==========================================
# AUTO-RELAUNCH UNDER ./qwen_env IF AVAILABLE
# ==========================================
# Users frequently run `python3 qwen_dictation.py` with the system Python
# (e.g. macOS ships 3.9.6) which has none of our dependencies installed —
# resulting in confusing errors like "No module named 'onnxruntime'" or the
# Python-3.10 guard tripping on the PyTorch backend. To make the script
# "just work", we transparently re-exec under ./qwen_env/bin/python (or
# .\qwen_env\Scripts\python.exe on Windows) whenever:
#   (a) the bundled venv exists, AND
#   (b) the current interpreter is NOT that venv's Python.
# The check is cheap (a couple of stat() calls) and runs before any third-
# party imports, so a broken/missing venv never blocks the script.
def _maybe_relaunch_in_venv() -> None:
    if os.environ.get("QWEN_DICTATION_RELAUNCHED") == "1":
        return  # Already re-execed once; don't loop.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        venv_python = os.path.join(script_dir, "qwen_env", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(script_dir, "qwen_env", "bin", "python")
    if not os.path.isfile(venv_python):
        return  # No bundled venv — fall through and let normal imports run.
    try:
        same = os.path.samefile(venv_python, sys.executable)
    except OSError:
        same = False
    if same:
        return  # Already running under the venv.
    print(f"[System] Re-launching under bundled venv: {venv_python}", flush=True)
    env = os.environ.copy()
    env["QWEN_DICTATION_RELAUNCHED"] = "1"
    os.execve(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]], env)


_maybe_relaunch_in_venv()

import gc
import time
import queue
import threading

# ==========================================
# MPS MEMORY OPTIMIZATION (must be set BEFORE torch is imported)
# ==========================================
# On Apple Silicon, PyTorch by default caps GPU memory usage at ~70% to
# leave headroom for other apps. Setting this to 0.0 removes the cap and
# lets PyTorch use all available unified memory for the model and KV-cache,
# which reduces memory pressure and speeds up inference.
# if sys.platform == "darwin":
#     os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
#     print("[System] macOS detected — MPS memory cap removed (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0)")

import numpy as np
import sounddevice as sd
from pynput import keyboard
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. CPU THREAD OPTIMIZATION (platform-aware)
# ==========================================
import platform

# torch is only needed for the PyTorch backends (options 1–4). The ONNX path
# (options 5–6) stays torch-free to keep RAM low — we import torch lazily.
torch = None  # type: ignore[assignment]


def _load_torch_and_tune_threads():
    global torch
    import torch as _torch  # noqa: PLC0415 - intentional lazy import
    torch = _torch
    if platform.system() == "Darwin":
        print(f"[System] macOS detected — using OS-managed threading (threads: {torch.get_num_threads()})")
    else:
        torch.set_num_threads(6)
        torch.set_num_interop_threads(2)
        print(f"[System] PyTorch threads: {torch.get_num_threads()} intra-op, {torch.get_num_interop_threads()} inter-op")

# ==========================================
# 1. CONFIGURATION & MODEL SELECTION
# ==========================================
print("--- Global Qwen3 / Parakeet Voice-to-Text Utility ---")
print("PyTorch backend (BF16/FP16, heavier RAM, full feature set):")
print("  1. Qwen3-ASR-0.6B                       (~2.5–3 GB RAM)")
print("  2. Qwen3-ASR-1.7B                       (~5–6 GB RAM)")
print("  3. Qwen3-ASR-0.6B + ForcedAligner-0.6B  (~3 GB RAM)")
print("  4. Qwen3-ASR-1.7B + ForcedAligner-0.6B  (~5–6 GB RAM)")
print("ONNX Runtime backend (INT4, minimal RAM, CPU-only, cross-platform):")
print("  5. Qwen3-ASR-0.6B INT4 ONNX             (~0.8–1.0 GB RAM)")
print("  6. Qwen3-ASR-1.7B INT4 ONNX             (~1.2–1.5 GB RAM)")
print("Parakeet TDT 0.6B v3 (English + 24 European langs, Hindi NOT supported):")
print("  7. Parakeet TDT v3 — MLX BF16           (~1.3–2 GB RAM, Apple Silicon only)")
print("  8. Parakeet TDT v3 — ONNX INT8          (~2 GB RAM, cross-platform CPU)")

VALID_CHOICES = {'1', '2', '3', '4', '5', '6', '7', '8'}

try:
    choice = input("Select Model Configuration (1-8): ").strip()
    if choice not in VALID_CHOICES:
        print("Invalid choice. Using default: Qwen3-ASR-0.6B INT4 ONNX")
        choice = '5'
except KeyboardInterrupt:
    print("\n[System] Exiting...")
    sys.exit(0)

use_onnx = choice in ['5', '6']
use_parakeet = choice in ['7', '8']
parakeet_flavour = None
if choice == '7':
    parakeet_flavour = "mlx"
elif choice == '8':
    parakeet_flavour = "onnx-int8"

# Hard-fail early if the user picks the MLX option on a non-Apple-Silicon box.
if choice == '7':
    from parakeet_backend import is_apple_silicon as _is_apple_silicon
    if not _is_apple_silicon():
        print()
        print("[Error] Option 7 (Parakeet MLX) requires Apple Silicon (M1/M2/M3/M4).")
        print("        Pick option 8 (Parakeet ONNX INT8) — works on any CPU.")
        sys.exit(1)

# Language prompt for text-only transcription. Forced-aligner options (3/4)
# need free-form output so they skip this. Parakeet v3 doesn't support Hindi
# but is happy with English + 24 European languages, so options 7/8 only get
# the English/Other menu.
lang_code = None
if choice in ['1', '2', '5', '6']:
    try:
        lang_choice = input("Select Language (1: English, 2: Hindi): ").strip()
        lang_code = "en" if lang_choice == '1' else "hi"
        if lang_choice not in ['1', '2']:
            print("Invalid language choice. Using English.")
            lang_code = "en"
    except KeyboardInterrupt:
        print("\n[System] Exiting...")
        sys.exit(0)
elif use_parakeet:
    # Parakeet v3 auto-detects within its 25 supported European languages.
    # We accept the prompt for symmetry but don't actually pass it to the
    # backend; "Hindi" is silently rejected with a warning if chosen.
    try:
        lang_choice = input(
            "Select Language (1: English, 2: Other European — auto-detect): "
        ).strip()
        lang_code = "en" if lang_choice == '1' else None
    except KeyboardInterrupt:
        print("\n[System] Exiting...")
        sys.exit(0)

use_pytorch = choice in ['1', '2', '3', '4']

if use_pytorch:
    # The PyTorch path needs qwen-asr 0.0.6, which hard-pins accelerate==1.12.0;
    # accelerate >= 1.11 requires Python >= 3.10. Bail out early with a clear
    # message before the import error gets confusing.
    if sys.version_info < (3, 10):
        print()
        print("[Error] Options 1–4 (PyTorch backend) require Python >= 3.10")
        print(f"        You are running Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print()
        print("  Either:")
        print("    a) Re-run with the bundled venv:  ./qwen_env/bin/python qwen_dictation.py")
        print("    b) Install Python 3.10+ from https://python.org and retry")
        print("    c) Pick option 5/6 (Qwen INT4 ONNX) or 8 (Parakeet ONNX INT8) — work on Python 3.9+")
        sys.exit(1)
    _load_torch_and_tune_threads()

# Map PyTorch-backend choices to Hugging Face repositories
pytorch_model_id_map = {
    '1': "Qwen/Qwen3-ASR-0.6B",
    '2': "Qwen/Qwen3-ASR-1.7B",
    '3': "Qwen/Qwen3-ASR-0.6B",
    '4': "Qwen/Qwen3-ASR-1.7B",
}

# ONNX-backend choices share the same family but use the INT4 ONNX repos.
onnx_size_map = {
    '5': "0.6B",
    '6': "1.7B",
}

DEVICE = "cpu"
TORCH_DTYPE = None
MODEL_ID = None
model = None
onnx_pipe = None       # Qwen3AsrOnnx instance when use_onnx is True
parakeet_pipe = None   # ParakeetAsr instance when use_parakeet is True

if use_parakeet:
    # -------------------- Parakeet TDT v3 (options 7, 8) -----------------
    # Lazy import keeps the dictation app start-up fast for users who never
    # touch this option (parakeet-mlx pulls in the MLX runtime, ~200 MB).
    from parakeet_backend import ParakeetAsr

    if parakeet_flavour == "mlx":
        print("\n[System] Initializing Parakeet TDT v3 (MLX BF16)...")
        print("[System] First run downloads ~2.5 GB; cached afterwards.")
    else:
        print("\n[System] Initializing Parakeet TDT v3 (ONNX INT8)...")
        print("[System] First run downloads ~671 MB; cached afterwards.")

    # Match the rest of the app: macOS lets the OS schedule, x86 caps to 6.
    parakeet_threads = 0 if platform.system() == "Darwin" else 6

    try:
        parakeet_pipe = ParakeetAsr(
            flavour=parakeet_flavour,
            num_threads=parakeet_threads,
        )
    except ImportError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to load Parakeet pipeline: {e}")
        sys.exit(1)

    print("[System] Parakeet pipeline loaded successfully.")
    MODEL_ID = (
        "mlx-community/parakeet-tdt-0.6b-v3"
        if parakeet_flavour == "mlx"
        else "istupakov/parakeet-tdt-0.6b-v3-onnx (INT8)"
    )
    DEVICE = "mps" if parakeet_flavour == "mlx" else "cpu"

    # Warmup with 1 second of silence — primes the encoder graph and pages
    # the hot weights into RAM so the first real recording isn't slow.
    try:
        print("[System] Running warmup inference...")
        t0 = time.time()
        parakeet_pipe.transcribe(np.zeros(16000, dtype=np.float32))
        print(f"[System] Warmup complete ({time.time() - t0:.1f}s).")
    except Exception as e:
        print(f"[System] Warmup skipped ({e}). First transcription may be slower.")

elif use_onnx:
    # -------------------- ONNX backend (options 5, 6) --------------------
    from qwen_onnx_backend import Qwen3AsrOnnx, download_onnx_model, ONNX_REPO_MAP

    size = onnx_size_map[choice]
    repo = ONNX_REPO_MAP[size]
    print(f"\n[System] Initializing {repo} (INT4 ONNX)...")
    print("[System] First run downloads ~2 GB (0.6B) or ~4 GB (1.7B); then mmapped.")

    try:
        model_dir = download_onnx_model(size)
    except Exception as e:
        print(f"[Error] Failed to download ONNX model: {e}")
        sys.exit(1)

    # Thread tuning is platform-aware: on macOS let the OS schedule; on
    # Windows/Linux match the PyTorch tuning to use P-cores + some E-cores.
    onnx_threads = 0 if platform.system() == "Darwin" else 6

    try:
        onnx_pipe = Qwen3AsrOnnx(model_dir, num_threads=onnx_threads)
    except Exception as e:
        print(f"[Error] Failed to load ONNX pipeline: {e}")
        sys.exit(1)

    print("[System] ONNX pipeline loaded successfully.")
    MODEL_ID = repo
    DEVICE = "cpu"  # ONNX path uses CPUExecutionProvider by design

    # Warmup — 1 second of silence through the full pipeline primes the ORT
    # graph optimizations and brings the hot weight pages into RAM.
    try:
        print("[System] Running warmup inference...")
        t0 = time.time()
        onnx_pipe.transcribe(
            np.zeros(16000, dtype=np.float32),
            language="English",
            max_new_tokens=4,
        )
        print(f"[System] Warmup complete ({time.time() - t0:.1f}s).")
    except Exception as e:
        print(f"[System] Warmup skipped ({e}). First transcription may be slower.")

else:
    # -------------------- PyTorch backend (options 1–4) --------------------
    # Lazy import: we only need qwen-asr (which pulls in transformers) when
    # the user picks a PyTorch option.
    from qwen_asr import Qwen3ASRModel

    MODEL_ID = pytorch_model_id_map[choice]

    # Device detection (priority: CUDA > XPU > MPS > CPU)
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"[System] NVIDIA CUDA detected: {torch.cuda.get_device_name()}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = "xpu"
        xpu_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, 'get_device_name') else "Intel GPU"
        print(f"[System] Intel XPU detected: {xpu_name}")
        print("[System] Note: Iris Xe (integrated) uses shared system RAM — not faster than CPU.")
        print("[System]       Intel Arc (discrete) has dedicated VRAM — will be faster.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
        print("[System] Apple Silicon (MPS) detected")
    else:
        DEVICE = "cpu"
        print("[System] Using CPU")

    if DEVICE == "mps":
        TORCH_DTYPE = torch.float16
        print("[System] Using float16 (MPS does not support bfloat16)")
    else:
        TORCH_DTYPE = torch.bfloat16

    print(f"\n[System] Initializing {MODEL_ID} on {DEVICE.upper()}...")
    print("[System] This may take a few minutes on first run...")

    try:
        load_kwargs = dict(
            dtype=TORCH_DTYPE,
            device_map=DEVICE,
            max_inference_batch_size=1,
            max_new_tokens=100,
        )
        if choice in ['3', '4']:
            print("[System] Loading Qwen3-ASR model with Qwen3-ForcedAligner-0.6B...")
            load_kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
            load_kwargs["forced_aligner_kwargs"] = dict(
                dtype=TORCH_DTYPE,
                device_map=DEVICE,
            )
        else:
            print("[System] Loading Qwen3-ASR model...")

        model = Qwen3ASRModel.from_pretrained(MODEL_ID, **load_kwargs)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        print("[System] Please ensure you have sufficient RAM/VRAM and internet connection.")
        sys.exit(1)

    print("[System] Models loaded successfully.")
    print(f"[System] Dtype: {TORCH_DTYPE}, Device: {DEVICE}")

    # Warmup pass — primes the generate() loop so the first real
    # transcription doesn't pay a cold-start tax.
    try:
        print("[System] Running warmup inference (first run may be slow)...")
        _warmup_audio = np.zeros(16000, dtype=np.float32)
        _warmup_start = time.time()
        with torch.inference_mode():
            model.transcribe(
                audio=(_warmup_audio, 16000),
                language="English",
                return_time_stamps=False,
            )
        print(f"[System] Warmup complete ({time.time() - _warmup_start:.1f}s).")
        del _warmup_audio, _warmup_start
        gc.collect()
    except Exception as e:
        print(f"[System] Warmup skipped ({e}). First transcription may be slower.")

print()

# ==========================================
# 2. AUDIO RECORDING ENGINE
# ==========================================
class AudioRecorder:
    MAX_DURATION_SECONDS = 30  # Cap recording to prevent unbounded memory growth

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.q = queue.Queue()
        self.stream = None
        self.is_recording = False
        self.audio_data = []
        self._chunk_count = 0

    def callback(self, indata, frames, time_info, status):
        """This is called continuously by sounddevice during recording."""
        if self.is_recording:
            self._chunk_count += 1
            # Estimate elapsed time from chunk count
            elapsed = (self._chunk_count * len(indata)) / self.sample_rate
            if elapsed < self.MAX_DURATION_SECONDS:
                self.q.put(indata.copy())
            elif elapsed < self.MAX_DURATION_SECONDS + 0.5:
                # Print warning only once (within a 0.5s window)
                print(f"\n[Warning] Max recording duration reached ({self.MAX_DURATION_SECONDS}s)")

    def start(self):
        """Opens the mic stream and begins capturing."""
        self.is_recording = True
        self.audio_data = []
        self._chunk_count = 0
        self.q = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            callback=self.callback, 
            dtype='float32'
        )
        self.stream.start()

    def stop(self):
        """Stops the mic stream and compiles the audio array."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        while not self.q.empty():
            self.audio_data.append(self.q.get())
            
        if not self.audio_data:
            return None
        
        # Flatten the audio chunks into a single 1D array
        audio_array = np.concatenate(self.audio_data, axis=0).flatten()
        
        # Validate audio length (minimum 0.5 seconds)
        min_length = int(0.5 * self.sample_rate)
        if len(audio_array) < min_length:
            print("[Warning] Audio too short, try speaking longer")
            return None
            
        # Check if audio contains actual sound (not just silence)
        if np.max(np.abs(audio_array)) < 0.001:  # Very quiet threshold
            print("[Warning] No speech detected, try speaking louder")
            return None
            
        return audio_array

recorder = AudioRecorder(sample_rate=16000)
is_processing = False

# ==========================================
# 3. INFERENCE & TEXT INJECTION
# ==========================================
def _transcribe_pytorch(audio_np: np.ndarray, language: str | None) -> str:
    """Run a single transcription through the PyTorch (qwen-asr) backend."""
    # qwen_asr accepts (np.ndarray, sr) directly — no temp WAV needed.
    with torch.inference_mode():
        results = model.transcribe(
            audio=(audio_np, 16000),
            language=language,
            return_time_stamps=False,
        )
    return results[0].text.strip() if results else ""


def _transcribe_onnx(audio_np: np.ndarray, language: str | None) -> str:
    """Run a single transcription through the INT4 ONNX backend."""
    result = onnx_pipe.transcribe(
        audio_np, language=language, max_new_tokens=100
    )
    return result.text.strip()


def _transcribe_parakeet(audio_np: np.ndarray, language: str | None) -> str:
    """Run a single transcription through the Parakeet TDT v3 backend."""
    # Parakeet auto-detects within its 25 European languages — the `language`
    # arg is accepted only for API parity and ignored downstream.
    result = parakeet_pipe.transcribe(audio_np, language=language)
    return result.text.strip()


def process_and_type(audio_np):
    global is_processing
    try:
        start_time = time.time()

        language = None
        if lang_code:
            language = "English" if lang_code == "en" else "Hindi"

        try:
            if use_parakeet:
                transcription = _transcribe_parakeet(audio_np, language)
            elif use_onnx:
                transcription = _transcribe_onnx(audio_np, language)
            else:
                transcription = _transcribe_pytorch(audio_np, language)
        except Exception as e:
            print(f"\n[Error] Transcription failed: {e}")
            return

        elapsed = time.time() - start_time
        print(f"Done. ({elapsed:.1f}s)")

        if transcription:
            try:
                kb_controller = keyboard.Controller()
                # Small delay so the OS focus hasn't shifted abruptly.
                time.sleep(0.1)
                kb_controller.type(transcription + " ")
            except Exception as e:
                print(f"[Error] Failed to type text: {e}")
                print("[Info] You may need to grant accessibility permissions")
        else:
            print("[Result]: (No speech detected)")

    except Exception as e:
        print(f"\n[Error] Unexpected error during transcription: {e}")
    finally:
        gc.collect()
        if use_pytorch and torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()
        is_processing = False
        if platform.system() == "Darwin":
            print("\n[System] Ready. Hold 'Right Option' to speak.")
        else:
            print("\n[System] Ready. Hold 'Right Alt' to speak.")

# ==========================================
# 4. GLOBAL KEYBOARD LISTENER
# ==========================================
def on_press(key):
    global is_processing
    # Check for Right Alt (handles Windows alt_gr and Mac right option variations)
    if key in (keyboard.Key.alt_r, keyboard.Key.alt_gr):
        if not recorder.is_recording and not is_processing:
            # print("\n[Mic] Recording started... Speak now. (Release to stop)", end="\r")
            recorder.start()

def on_release(key):
    global is_processing
    if key in (keyboard.Key.alt_r, keyboard.Key.alt_gr):
        if recorder.is_recording:
            audio_np = recorder.stop()
            # print("\n[Mic] Recording stopped.")
            
            if audio_np is not None and len(audio_np) > 0:
                is_processing = True
                # Offload to thread so the keyboard listener doesn't freeze
                threading.Thread(target=process_and_type, args=(audio_np,), daemon=True).start()
            else:
                print("[System] No valid audio captured. Try again.")
                is_processing = False
    
    # Allow graceful exit with Escape key
    # elif key == keyboard.Key.esc:
    #     print("\n[System] Escape pressed. Exiting...")
    #     return False  # Stop the listener

# ==========================================
# 5. EXECUTION
# ==========================================
def main():
    """Main execution function with proper cleanup"""
    try:
        print("\n=======================================================")
        print(" SYSTEM READY ")
        print("=======================================================")
        print("1. Click into ANY text box (Browser, Word, Notepad).")
        if platform.system() == "Darwin":
            print("2. Press and HOLD the 'Right Option' key.")
        else:
            print("2. Press and HOLD the 'Right Alt' key.")
        print("3. Speak your sentence.")
        print("4. Release the key to automatically type the text.")
        # print("5. Press 'Escape' to exit the program.")
        print("=======================================================\n")
        
        # Check microphone access
        try:
            test_stream = sd.InputStream(samplerate=16000, channels=1, blocksize=1024)
            test_stream.start()
            test_stream.stop()
            test_stream.close()
            print("[System] Microphone access verified.")
        except Exception as e:
            print(f"[Warning] Microphone access issue: {e}")
            print("[Info] Please ensure microphone permissions are granted.")
        
        print("[System] Starting keyboard listener...")
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
            
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")
    finally:
        # Cleanup
        if hasattr(recorder, 'stream') and recorder.stream:
            try:
                recorder.stream.stop()
                recorder.stream.close()
            except Exception:
                pass
        print("[System] Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()