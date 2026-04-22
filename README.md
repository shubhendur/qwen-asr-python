# Qwen3 / Parakeet Voice-to-Text Dictation System

A comprehensive voice-to-text system with global hotkey support, bundling both
**Qwen3-ASR** (multilingual, includes Hindi) and **NVIDIA Parakeet TDT v3**
(English + 24 European languages, ultra-fast on CPU). All transcription runs
**locally** — no cloud, no internet required after model download.

Three inference backends are available in the same app:

- **Qwen3 PyTorch (options 1–4)** — original qwen-asr package, full feature set
  (forced aligner, timestamps), heavier RAM.
- **Qwen3 ONNX Runtime INT4 (options 5–6)** — pure C++ runtime with Python
  bindings, INT4-quantized decoder, ~4× lower RAM, ~2× faster CPU inference.
  No PyTorch needed at runtime.
- **Parakeet TDT v3 (options 7–8)** — NVIDIA's fastest CPU-friendly ASR. Two
  flavours: MLX BF16 for Apple Silicon (~30–50× real-time on M1 Pro), and
  ONNX INT8 for any CPU (~12–20× real-time on i7-1355U, identical WER to FP32).
  Note: not trained on Hindi/CJK.

## Quick Start

### macOS / Linux

```bash
# One-time setup (creates a local venv at ./qwen_env and installs ONNX-only deps)
./setup.sh

# Add the PyTorch backend (options 1–4, Python >= 3.10 required):
./setup.sh --pytorch

# Add the Parakeet TDT v3 backends (options 7 + 8):
./setup.sh --parakeet

# Or install everything in one go:
./setup.sh --pytorch --parakeet

# Run the dictation app (auto-relaunches under the bundled venv even
# if you call it with the system python3)
./qwen_env/bin/python qwen_dictation.py
# ...or simply:
python3 qwen_dictation.py
```

### Windows 11 (CMD as Administrator)

```cmd
REM One-time setup (creates qwen_env and installs ONNX-only deps)
setup.bat

REM Add the PyTorch backend (needs Python >= 3.10):
setup.bat --pytorch

REM Add the Parakeet TDT v3 ONNX INT8 backend:
setup.bat --parakeet

REM Or install everything in one go:
setup.bat --pytorch --parakeet

REM Run the dictation app
qwen_env\Scripts\python.exe qwen_dictation.py
```

### Python version requirements

| Backend | Options | Min Python | Why |
|---|---|---|---|
| Qwen ONNX INT4 | 5, 6 | **3.9** | Pure pip wheels (numpy, onnxruntime, tokenizers) |
| Parakeet ONNX INT8 | 8 | **3.9** | Same — pure pip wheels via `onnx-asr` |
| Parakeet MLX | 7 | **3.10** | `parakeet-mlx` requires modern Python on Apple Silicon |
| Qwen PyTorch | 1, 2, 3, 4 | **3.10** | `qwen-asr 0.0.6` pins `accelerate==1.12.0`, which itself requires Python 3.10+ |

If you try to pick a Python-3.10-only option on Python 3.9 the app will exit
with a clear message. Easier: just run `python3 qwen_dictation.py` — the
script auto-relaunches under `./qwen_env` if you ran setup.

## RAM & Speed Comparison (Intel i7-1355U, 16 GB RAM)

| Option | Model | Backend | Peak RAM | CPU RTF | Quality |
|--------|-------|---------|----------|---------|---------|
| 1 | Qwen3-ASR-0.6B | PyTorch BF16 | ~2.5–3 GB | ~0.3× | Good |
| 2 | Qwen3-ASR-1.7B | PyTorch BF16 | **5–6 GB** | ~0.7× | Excellent |
| 3 | 0.6B + Forced Aligner | PyTorch BF16 | ~3 GB | ~0.4× | Good + timestamps |
| 4 | 1.7B + Forced Aligner | PyTorch BF16 | 5–6 GB | ~0.8× | Excellent + timestamps |
| **5** | **Qwen3-ASR-0.6B** | **ONNX INT4** | **~0.8–1.0 GB** | **~0.17×** | Good (WER +0.7pp) |
| **6** | **Qwen3-ASR-1.7B** | **ONNX INT4** | **~1.2–1.5 GB** | **~0.29×** | Excellent (WER +0.4pp) |
| **8** | **Parakeet TDT v3** | **ONNX INT8** | **~2 GB** | **~0.06–0.08×** | Excellent — identical WER to FP32 (English + 24 EU langs) |

RTF < 1.0 = faster than real-time. Options 5 & 6 use
[`andrewleech/qwen3-asr-onnx`](https://huggingface.co/andrewleech) INT4 exports
(ONNX Runtime `MatMulNBits` with accuracy_level=4). Option 8 uses
[`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
fetched automatically via the `onnx-asr` package.

## RAM & Speed Comparison (Apple M1 Pro, 16 GB unified memory)

| Option | Model | Backend | Peak RAM | RTF | Notes |
|--------|-------|---------|----------|-----|-------|
| 2 | Qwen3-ASR-1.7B | PyTorch FP16 (MPS) | ~5–6 GB | ~0.4× | Multilingual incl. Hindi |
| 6 | Qwen3-ASR-1.7B | ONNX INT4 (CPU) | ~1.2–1.5 GB | ~0.3× | Same model, lower RAM |
| **7** | **Parakeet TDT v3** | **MLX BF16 (Apple GPU)** | **~1.3–2 GB** | **~0.02–0.03×** | English + 24 EU langs, blazing fast |
| 8 | Parakeet TDT v3 | ONNX INT8 (CPU) | ~2 GB | ~0.05× | Cross-platform, also runs here |

Option 7 (`mlx-community/parakeet-tdt-0.6b-v3`) is the fastest path on Apple
Silicon — ~30–50× real-time on M1 Pro thanks to unified memory and the MLX
runtime's lazy evaluation. Pick option 6 instead if you need Hindi.

### Why the ONNX INT4 path is lighter

- INT4 weights for the decoder (4 bits/weight instead of 16).
- `decoder_weights.int4.data` is memory-mapped by ONNX Runtime, so only hot
  pages are resident — weight storage doesn't bloat RSS.
- FP16 `embed_tokens.bin` loaded once; no duplicate embedding table across
  prefill/step sessions.
- No PyTorch runtime (~400 MB saved), no `accelerate` dispatch hooks.

### Why the PyTorch path still exists

The INT4 ONNX export does not currently include the forced aligner, so if you
need timestamps (options 3/4) you must keep the PyTorch backend. For plain
dictation, option 5 or 6 is strictly better.

### GPU note (Intel Iris Xe)

The i7-1355U's Iris Xe has **no dedicated VRAM** — it shares system RAM. Using
the integrated GPU does **not** reduce memory and is often slower than CPU on
this chip. For PyTorch options the code auto-detects Intel XPU but **CPU with
BF16 is recommended**. For ONNX options the CPU execution provider is the
default and the fastest path on this hardware.

## Features

- **Eight model configurations** — Qwen3 (0.6B / 1.7B across PyTorch and INT4
  ONNX) plus Parakeet TDT v3 (MLX and ONNX INT8).
- **Forced aligner support** — Qwen options 3 and 4.
- **Global hotkey** — press and hold Right Alt (Right Option on Mac) to record.
- **Direct text injection** — types transcribed text without using the clipboard.
- **Cross-platform** — macOS (Intel + Apple Silicon), Windows, Linux.
- **Hardware acceleration** — auto-detects CUDA / Intel XPU / MPS / CPU for
  PyTorch options; CPU for ONNX options; Apple GPU (Metal) for Parakeet MLX.
- **Memory-lean** — INT4 MatMulNBits + memory-mapped weights + FP16 embeddings
  (Qwen ONNX); INT8 quantization with identical WER to FP32 (Parakeet ONNX).
- **Auto-relaunch** — `python3 qwen_dictation.py` transparently re-execs under
  `./qwen_env` if you ran setup, so you never have to remember the venv path.

## Installation

1. **Install dependencies**:

   The recommended path is `./setup.sh` / `setup.bat` (see Quick Start above).
   It creates a local `./qwen_env` venv and installs the right packages for
   the backend(s) you want. If you'd rather install manually:

   ```bash
   # ONNX-only (Python 3.9+, options 5 & 6)
   pip install -r requirements.txt

   # Add PyTorch (Python 3.10+, options 1–4)
   pip install -r requirements.txt -r requirements-pytorch.txt
   ```

   Why two files: `qwen-asr 0.0.6` hard-pins `accelerate==1.12.0`, which
   needs Python ≥ 3.10. Splitting keeps the low-RAM ONNX path installable on
   the system Python that ships with macOS (3.9.6).

2. **Download a model** (optional — auto-downloads on first run):

   ```bash
   python download_model.py
   ```

3. **Grant permissions (Windows)**:
   - Run Terminal as Administrator for elevated target applications.
   - Allow microphone access when prompted.

4. **Grant permissions (macOS — all three required)**:
   - **System Settings → Privacy & Security → Accessibility** — add Terminal / your IDE.
   - **System Settings → Privacy & Security → Input Monitoring** — add Terminal / your IDE.
   - **System Settings → Privacy & Security → Microphone** — allow Terminal / your IDE.
   - **Restart Terminal** after granting all three.
   - Note: on Mac "Right Alt" = **Right Option**.

## Usage

1. **Run the script**:

   ```bash
   python3 qwen_dictation.py
   ```

2. **Select a model configuration (1–8)**:
   - 1/2 — Qwen3 PyTorch 0.6B / 1.7B (full precision).
   - 3/4 — Qwen3 PyTorch with forced aligner (timestamps).
   - 5/6 — Qwen3 ONNX INT4 0.6B / 1.7B (lowest RAM, includes Hindi).
   - **7 — Parakeet TDT v3 MLX (Apple Silicon, fastest GPU path)**.
   - **8 — Parakeet TDT v3 ONNX INT8 (cross-platform, fastest CPU path)**.

3. **Choose language**:
   - Options 1, 2, 5, 6 — English or Hindi.
   - Options 7, 8 — English or Other European (auto-detect; **no Hindi**).

4. **Use the system**:
   - Click into any text field.
   - Hold Right Alt (Right Option on Mac) and speak.
   - Release to automatically type the transcribed text.

## Troubleshooting

### Models fail to download

- Ensure a stable internet connection.
- Check disk space (up to ~5 GB for the largest PyTorch model, ~4 GB for 1.7B ONNX).
- If HuggingFace is restricted in your region, try a VPN.

### Audio not detected

- Check microphone permissions.
- Ensure no other app is holding the microphone.
- Speak louder and closer to the mic.

### Text not typing

- Grant accessibility permissions (macOS).
- Run as administrator (Windows) for elevated target apps.
- Ensure the target application accepts keyboard input.

### Out-of-memory errors

- **First try**: switch to option 5 (0.6B ONNX INT4) — uses ~1 GB RAM.
- Close other applications to free RAM.
- Restart the script to clear memory.

## Hardware requirements

### Minimum (options 5/6, ONNX INT4)

- **CPU**: any x86-64 with AVX2, or Apple Silicon.
- **RAM**: 4 GB (0.6B) / 6 GB (1.7B).
- **Storage**: 5 GB free for model files.

### Recommended (options 1–4, PyTorch)

- **GPU**: NVIDIA RTX series or Apple Silicon M1/M2/M3/M4.
- **RAM**: 8 GB (0.6B) / 16 GB (1.7B).
- **Storage**: SSD with 10 GB+ free.

## Model information

| Model | PyTorch download | ONNX INT4 download | ONNX RAM |
|-------|------------------|--------------------|----------|
| Qwen3-ASR-0.6B | ~1.2 GB | ~2.0 GB | ~0.8–1.0 GB |
| Qwen3-ASR-1.7B | ~4.5 GB | ~4.0 GB | ~1.2–1.5 GB |
| + Forced Aligner | +1 GB | n/a | n/a |

## Privacy & security

- **Offline**: all transcription happens locally.
- **No cloud**: no data sent to external servers.
- **Cached locally**: models download once, stored in `~/.cache/huggingface/`.
- **No logging**: audio and transcriptions are never saved to disk.
