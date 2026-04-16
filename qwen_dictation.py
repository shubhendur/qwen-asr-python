#!/usr/bin/env ./qwen_env/bin/python
"""
Global Qwen3 Voice-to-Text Utility
Using the official qwen-asr package for reliable speech recognition
"""

import os
import sys
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
# 1. CONFIGURATION & MODEL SELECTION
# ==========================================
print("--- Global Qwen3 Voice-to-Text Utility ---")
print("1. Qwen3-ASR-1.7B")
print("2. Qwen3-ASR-0.6B")
print("3. Qwen3-ASR-0.6B with Qwen3-ForcedAligner-0.6B")
print("4. Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B")

try:
    choice = input("Select Model Configuration (1-4): ").strip()
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Using default: Qwen3-ASR-0.6B")
        choice = '2'
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
    '1': "Qwen/Qwen3-ASR-1.7B",
    '2': "Qwen/Qwen3-ASR-0.6B",
    '3': "Qwen/Qwen3-ASR-0.6B", 
    '4': "Qwen/Qwen3-ASR-1.7B"
}

MODEL_ID = model_id_map.get(choice, "Qwen/Qwen3-ASR-0.6B")

# Better device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[System] CUDA detected: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("[System] Apple Silicon (MPS) detected")
else:
    DEVICE = "cpu"
    print("[System] Using CPU (consider GPU for better performance)")

TORCH_DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

print(f"\n[System] Initializing {MODEL_ID} on {DEVICE.upper()}...")
print("[System] This may take a few minutes on first run...")

# Load Qwen3-ASR Model with error handling (single load)
try:
    # Build load arguments
    load_kwargs = dict(
        dtype=TORCH_DTYPE,
        device_map=DEVICE,
        max_inference_batch_size=32,
        max_new_tokens=256,
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

except Exception as e:
    print(f"[Error] Failed to load model: {e}")
    print("[System] Please ensure you have sufficient RAM/VRAM and internet connection.")
    sys.exit(1)

print("[System] Models loaded successfully.\n")

# ==========================================
# 2. AUDIO RECORDING ENGINE
# ==========================================
class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.q = queue.Queue()
        self.stream = None
        self.is_recording = False
        self.audio_data = []

    def callback(self, indata, frames, time, status):
        """This is called continuously by sounddevice during recording."""
        if self.is_recording:
            self.q.put(indata.copy())

    def start(self):
        """Opens the mic stream and begins capturing."""
        self.is_recording = True
        self.audio_data = []
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
            
            # Transcribe with qwen-asr
            language = None
            if lang_code:
                language = "English" if lang_code == "en" else "Hindi"
                
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
        # Unlock the system for the next recording
        is_processing = False
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