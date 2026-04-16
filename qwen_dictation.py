#!/usr/bin/env ./qwen_env/bin/python
"""
Global Qwen3 Voice-to-Text Utility
Using the official qwen-asr package for reliable speech recognition
"""

import os
import sys
import gc
import time
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
import torch
from pynput import keyboard
from qwen_asr import Qwen3ASRModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 0. CPU THREAD OPTIMIZATION (platform-aware)
# ==========================================
import platform

if platform.system() == "Darwin":
    # macOS + Apple Silicon: let Grand Central Dispatch manage threads
    print(f"[System] macOS detected — using OS-managed threading (threads: {torch.get_num_threads()})")
else:
    # Windows/Linux: manual tuning for Intel i7-1355U (2P+8E = 10 cores)
    torch.set_num_threads(6)         # Use P-cores + some E-cores for intra-op
    torch.set_num_interop_threads(2) # For inter-op parallelism
    print(f"[System] PyTorch threads: {torch.get_num_threads()} intra-op, {torch.get_num_interop_threads()} inter-op")

# ==========================================
# 1. CONFIGURATION & MODEL SELECTION
# ==========================================
print("--- Global Qwen3 Voice-to-Text Utility ---")
print("1. Qwen3-ASR-0.6B (Recommended — fast, low RAM)")
print("2. Qwen3-ASR-1.7B (Higher quality, slower)")
print("3. Qwen3-ASR-0.6B with Qwen3-ForcedAligner-0.6B")
print("4. Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B")

try:
    choice = input("Select Model Configuration (1-4): ").strip()
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Using default: Qwen3-ASR-0.6B")
        choice = '1'
except KeyboardInterrupt:
    print("\n[System] Exiting...")
    sys.exit(0)

lang_code = None
if choice in ['1', '2']:
    try:
        lang_choice = input("Select Language (1: English, 2: Hindi): ").strip()
        lang_code = "en" if lang_choice == '1' else "hi"
        if lang_choice not in ['1', '2']:
            print("Invalid language choice. Using English.")
            lang_code = "en"
    except KeyboardInterrupt:
        print("\n[System] Exiting...")
        sys.exit(0)

# Map choices to Hugging Face Repositories
model_id_map = {
    '1': "Qwen/Qwen3-ASR-0.6B",
    '2': "Qwen/Qwen3-ASR-1.7B",
    '3': "Qwen/Qwen3-ASR-0.6B", 
    '4': "Qwen/Qwen3-ASR-1.7B"
}

MODEL_ID = model_id_map.get(choice, "Qwen/Qwen3-ASR-0.6B")

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

# Select dtype based on device capabilities
if DEVICE == "mps":
    # Apple Silicon MPS does NOT support bfloat16 — use float16 instead
    TORCH_DTYPE = torch.float16
    print("[System] Using float16 (MPS does not support bfloat16)")
else:
    # CUDA, XPU, and modern Intel CPUs (13th gen+) support bfloat16
    TORCH_DTYPE = torch.bfloat16

print(f"\n[System] Initializing {MODEL_ID} on {DEVICE.upper()}...")
print("[System] This may take a few minutes on first run...")

# Load Qwen3-ASR Model with error handling (single load)
try:
    # Build load arguments
    load_kwargs = dict(
        dtype=TORCH_DTYPE,
        device_map=DEVICE,
        # Batch size 1: we only process one utterance at a time
        # (32 was pre-allocating KV-cache for 32 streams — wasting ~1-2 GB)
        max_inference_batch_size=1,
        # 100 tokens is plenty for dictation clips (5-30 seconds of speech)
        max_new_tokens=100,
    )

    # Add forced aligner only for choices 3/4
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

    # Force GC after model load to reclaim any transient allocations
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

# ==========================================
# 1b. TORCH.COMPILE — JIT optimization for faster CPU inference
# ==========================================
# torch.compile fuses operations and generates optimized kernels.
# First inference is slow (compilation), but subsequent ones are 20-40% faster.
try:
    if DEVICE == "cpu":
        print("[System] Compiling model with torch.compile (this speeds up inference)...")
        model = torch.compile(model, mode="default", dynamic=True)
        print("[System] Model compiled successfully.")
    else:
        print("[System] Skipping torch.compile (not needed for GPU).")
except Exception as e:
    print(f"[System] torch.compile not available ({e}), using default inference.")

# Warmup pass — triggers JIT compilation so the first real transcription is fast
try:
    print("[System] Running warmup inference (first run compiles optimized kernels)...")
    import tempfile as _tf
    import numpy as _np_warmup
    
    # Generate 1 second of silence for warmup
    _warmup_audio = _np_warmup.zeros(16000, dtype=_np_warmup.float32)
    with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as _wf:
        _warmup_path = _wf.name
    try:
        import soundfile as sf
        sf.write(_warmup_path, _warmup_audio, 16000)
    except ImportError:
        import scipy.io.wavfile as wavfile
        wavfile.write(_warmup_path, 16000, (_warmup_audio * 32767).astype(_np_warmup.int16))
    
    _warmup_start = time.time()
    with torch.inference_mode():
        model.transcribe(audio=_warmup_path, language="English", return_time_stamps=False)
    _warmup_elapsed = time.time() - _warmup_start
    print(f"[System] Warmup complete ({_warmup_elapsed:.1f}s). Subsequent transcriptions will be faster.")
    
    # Cleanup warmup file and variables
    try:
        os.unlink(_warmup_path)
    except:
        pass
    del _warmup_audio, _warmup_path, _warmup_start, _warmup_elapsed
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
def process_and_type(audio_np):
    global is_processing
    try:
        print("[System] Transcribing...", end=" ")
        start_time = time.time()
        
        # Save audio to temporary file for qwen-asr
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Save audio as WAV file
            try:
                import soundfile as sf
                sf.write(temp_path, audio_np, 16000)
            except ImportError:
                # Fallback to scipy.io.wavfile if soundfile not available
                import scipy.io.wavfile as wavfile
                # Convert float to int16
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wavfile.write(temp_path, 16000, audio_int16)
            
            # Transcribe with qwen-asr (inference_mode skips autograd overhead)
            language = None
            if lang_code:
                language = "English" if lang_code == "en" else "Hindi"

            with torch.inference_mode():
                results = model.transcribe(
                    audio=temp_path,
                    language=language,
                    return_time_stamps=False,
                )
            
            transcription = results[0].text.strip() if results else ""
            
        except Exception as e:
            print(f"\n[Error] Transcription failed: {e}")
            return
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
        elapsed = time.time() - start_time
        print(f"Done. ({elapsed:.1f}s)")
        
        if transcription and len(transcription) > 0:
            print(f"[Result]: {transcription}")
            
            try:
                # Inject directly into the active window (No Clipboard)
                kb_controller = keyboard.Controller()
                
                # Small delay to ensure the OS focus hasn't shifted abruptly
                time.sleep(0.1) 
                
                # Type the transcription with a space
                kb_controller.type(transcription + " ")
            except Exception as e:
                print(f"[Error] Failed to type text: {e}")
                print("[Info] You may need to grant accessibility permissions")
        else:
            print("[Result]: (No speech detected)")

    except Exception as e:
        print(f"\n[Error] Unexpected error during transcription: {e}")
    finally:
        # Reclaim transient inference memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
        # Unlock the system for the next recording
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
            print("\n[Mic] Recording started... Speak now. (Release to stop)", end="\r")
            recorder.start()

def on_release(key):
    global is_processing
    if key in (keyboard.Key.alt_r, keyboard.Key.alt_gr):
        if recorder.is_recording:
            audio_np = recorder.stop()
            print("\n[Mic] Recording stopped.")
            
            if audio_np is not None and len(audio_np) > 0:
                is_processing = True
                # Offload to thread so the keyboard listener doesn't freeze
                threading.Thread(target=process_and_type, args=(audio_np,), daemon=True).start()
            else:
                print("[System] No valid audio captured. Try again.")
                is_processing = False
    
    # Allow graceful exit with Escape key
    elif key == keyboard.Key.esc:
        print("\n[System] Escape pressed. Exiting...")
        return False  # Stop the listener

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
        print("5. Press 'Escape' to exit the program.")
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
            except:
                pass
        print("[System] Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()