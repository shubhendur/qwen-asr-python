"""
Moonshine Base ASR — Push-to-Talk Speech-to-Text
Uses ONNX Runtime with INT8 quantized weights for minimal CPU/RAM usage.
No PyTorch, no transformers library — only lightweight dependencies.
"""
import os
import sys
import json
import time
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from tokenizers import Tokenizer
from pynput import keyboard

# --- Settings ---
MODEL_ID = "onnx-community/moonshine-base-ONNX"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moonshine_model")
SAMPLE_RATE = 16000
CHANNELS = 1

# --- Text Injection ---
keyboard_controller = keyboard.Controller()

def type_text(text):
    """Types text directly into the active window using pynput."""
    # Brief pause to ensure hotkey release event is fully processed by the OS
    time.sleep(0.05)
    keyboard_controller.type(text)


# --- Lightweight Config (replaces transformers.AutoConfig) ---
class ModelConfig:
    """Reads config.json directly — no transformers needed."""
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.decoder_start_token_id = cfg["decoder_start_token_id"]
        self.eos_token_id = cfg["eos_token_id"]
        self.decoder_num_key_value_heads = cfg["decoder_num_key_value_heads"]
        self.decoder_num_attention_heads = cfg["decoder_num_attention_heads"]
        self.decoder_num_hidden_layers = cfg["decoder_num_hidden_layers"]
        self.hidden_size = cfg["hidden_size"]
        self.max_position_embeddings = cfg["max_position_embeddings"]


# --- Model ---
class MoonshineModel:
    def __init__(self, local_dir):
        print(f"Loading Moonshine model from {local_dir}...")

        # Load config from JSON directly
        self.config = ModelConfig(os.path.join(local_dir, "config.json"))

        # Load tokenizer using the lightweight 'tokenizers' library (Rust-based, ~5 MB)
        self.tokenizer = Tokenizer.from_file(os.path.join(local_dir, "tokenizer.json"))

        # CPU Optimization settings for ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False   # Lower peak memory allocation
        sess_options.enable_mem_reuse = True       # Reuse memory buffers across ops

        encoder_path = os.path.join(local_dir, "onnx", "encoder_model_quantized.onnx")
        decoder_path = os.path.join(local_dir, "onnx", "decoder_model_merged_quantized.onnx")

        print("Initializing ONNX Inference Sessions (this may take a moment)...")
        providers = ["CPUExecutionProvider"]
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)
        self.decoder_session = ort.InferenceSession(decoder_path, sess_options, providers=providers)
        print("Model loaded successfully!")

    def transcribe(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            return ""

        # Ensure correct shape [1, length] and float32
        audio = audio_data[None, :].astype(np.float32)

        # Run encoder
        try:
            encoder_outputs = self.encoder_session.run(None, dict(input_values=audio))[0]
        except Exception as e:
            print(f"\nError in encoder: {e}")
            return ""

        # Prepare decoder inputs
        batch_size = encoder_outputs.shape[0]
        input_ids = np.array([[self.config.decoder_start_token_id]] * batch_size)

        num_kv_heads = self.config.decoder_num_key_value_heads
        dim_kv = self.config.hidden_size // self.config.decoder_num_attention_heads

        past_key_values = {
            f'past_key_values.{layer}.{module}.{kv}': np.zeros(
                [batch_size, num_kv_heads, 0, dim_kv], dtype=np.float32
            )
            for layer in range(self.config.decoder_num_hidden_layers)
            for module in ('decoder', 'encoder')
            for kv in ('key', 'value')
        }

        # max 6 tokens per second of audio
        audio_length = audio.shape[-1]
        max_len = min(int((audio_length / SAMPLE_RATE) * 6), self.config.max_position_embeddings)
        max_len = max(max_len, 20)  # minimum 20 tokens for very short clips

        generated_tokens = input_ids

        for i in range(max_len):
            use_cache_branch = i > 0

            inputs = dict(
                input_ids=generated_tokens[:, -1:],
                encoder_hidden_states=encoder_outputs,
                use_cache_branch=np.array([use_cache_branch], dtype=bool),
                **past_key_values
            )

            logits, *present_key_values = self.decoder_session.run(None, inputs)

            next_tokens = logits[:, -1].argmax(-1, keepdims=True)

            for j, key in enumerate(past_key_values):
                if not use_cache_branch or 'decoder' in key:
                    past_key_values[key] = present_key_values[j]

            generated_tokens = np.concatenate([generated_tokens, next_tokens], axis=-1)

            if (next_tokens == self.config.eos_token_id).all():
                break

        # Decode tokens using the lightweight tokenizers library
        token_ids = generated_tokens[0].tolist()
        # Remove special tokens (decoder_start_token_id=1, eos_token_id=2)
        special_ids = {self.config.decoder_start_token_id, self.config.eos_token_id}
        token_ids = [t for t in token_ids if t not in special_ids]
        result = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return result.strip()


# --- Application ---
class MoonshineApp:
    def __init__(self):
        self.is_recording = False
        self.audio_chunks = []
        self.model = None

        self._ensure_model_downloaded()
        self.model = MoonshineModel(LOCAL_MODEL_DIR)

    def _ensure_model_downloaded(self):
        encoder_exists = os.path.exists(
            os.path.join(LOCAL_MODEL_DIR, "onnx", "encoder_model_quantized.onnx")
        )
        if not encoder_exists:
            # Lazy import — only loaded if model needs downloading (first run)
            from huggingface_hub import snapshot_download
            print("Downloading Moonshine Model files (only INT8 quantized ONNX files)...")
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=LOCAL_MODEL_DIR,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "preprocessor_config.json",
                    "special_tokens_map.json",
                    "onnx/encoder_model_quantized.onnx",
                    "onnx/decoder_model_merged_quantized.onnx"
                ]
            )
            print("Download complete.")

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_chunks.append(indata.copy())

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.audio_chunks = []
        print("\r[RECORDING] Dictate now... (Release Right Alt to stop)          ", end="", flush=True)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        print("\r[PROCESSING] Transcribing...                                     ", end="", flush=True)

        if not self.audio_chunks:
            print("\r[INFO] No audio recorded.                                ", flush=True)
            return

        # Concatenate audio and flatten into 1D float32 array
        audio_data = np.concatenate(self.audio_chunks, axis=0).flatten().astype(np.float32)
        # Free the chunk references immediately
        self.audio_chunks = []

        text = self.model.transcribe(audio_data)

        if text:
            # print(f"\r[SUCCESS] Transcribed: '{text}'", flush=True)
            type_text(text + " ")
        else:
            print("\r[INFO] Empty transcription.                              ", flush=True)

    def run(self):
        def on_press(key):
            if key == keyboard.Key.alt_r or key == keyboard.Key.alt_gr:
                self.start_recording()

        def on_release(key):
            if key == keyboard.Key.alt_r or key == keyboard.Key.alt_gr:
                self.stop_recording()
            elif key == keyboard.Key.esc:
                print("\nExiting...")
                return False  # Stop listener

        print("\n" + "=" * 50)
        print("Moonshine Base ASR - Ready!")
        print("Press and HOLD 'Right Alt' to dictate.")
        print("Release to transcribe. Press 'Esc' to exit.")
        print("=" * 50 + "\n")

        # Open audio stream once globally to avoid startup lag on hotkey
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                            callback=self.audio_callback):
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()


if __name__ == "__main__":
    try:
        app = MoonshineApp()
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
