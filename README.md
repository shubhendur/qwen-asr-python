# Qwen3 Voice-to-Text Dictation System

A comprehensive voice-to-text system using Qwen3-ASR models with global hotkey support.  
All transcription runs **locally** — no cloud, no internet required after model download.

## Quick Start (Windows 11)

Open CMD with administrative rights:
```cmd
python qwen_dictation.py
```

## RAM Consumption (Intel i7-1355U, 16 GB RAM)

| Config | Model(s) | Before Optimization | After Optimization |
|--------|----------|--------------------|--------------------|
| 1 | Qwen3-ASR-1.7B | 8 GB | ~4–5 GB |
| 2 | Qwen3-ASR-0.6B | 4.5 GB | ~2.5–3 GB |
| 3 | 0.6B + Forced Aligner | 4 GB | ~2.5–3 GB |
| 4 | 1.7B + Forced Aligner | 8 GB | ~5–6 GB |

### Optimizations Applied
- **BFloat16 on CPU** — Intel 13th gen supports BF16 natively, halving weight memory
- **Batch size 32→1** — single utterance at a time, eliminates wasted KV-cache
- **max_new_tokens 256→100** — sufficient for dictation clips (5–30 seconds)
- **CPU thread tuning** — 6 intra-op + 2 inter-op threads for i7-1355U (2P+8E cores)
- **`torch.inference_mode()`** — skips autograd graph tracking during transcription
- **Garbage collection** — explicit `gc.collect()` after each inference to prevent memory creep
- **30s audio cap** — prevents unbounded memory growth for long recordings

### GPU Note (Intel Iris Xe)
The i7-1355U has Intel Iris Xe integrated graphics with **no dedicated VRAM** — it shares system RAM. Using the GPU does **not** save memory or provide meaningful speedup on this hardware. The code auto-detects Intel XPU but **CPU with BF16 is the recommended path** for this chip. If you have an NVIDIA GPU or Intel Arc (discrete), those will provide real acceleration.

## Features

- **Multiple Model Options**: Support for Qwen3-ASR-1.7B and 0.6B models
- **Forced Aligner Support**: Optional forced alignment capabilities  
- **Global Hotkeys**: Press and hold Right Alt to record speech
- **Direct Text Injection**: Types transcribed text directly without clipboard
- **Cross-Platform**: Works on macOS, Windows, and Linux
- **Hardware Acceleration**: Automatic GPU detection (CUDA / Intel XPU / MPS / CPU)
- **Memory Optimized**: BF16, batch-size-1, inference_mode, GC cleanup

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Model** (optional — auto-downloads on first run):
   ```bash
   python download_model.py
   ```

3. **Grant Permissions** (Windows):
   - Run Terminal as Administrator for elevated applications
   - Allow microphone access when prompted

4. **Grant Permissions** (macOS):
   - Go to System Settings > Privacy & Security > Accessibility
   - Add Terminal (or your Python IDE) to allowed applications
   - Grant microphone access when prompted

## Usage

1. **Run the Script**:
   ```bash
   python qwen_dictation.py
   ```

2. **Select Model Configuration**:
   - Option 1: Qwen3-ASR-0.6B ⭐ Recommended (fast, ~3-5s transcription)
   - Option 2: Qwen3-ASR-1.7B (higher quality, ~30s on CPU)  
   - Option 3: 0.6B + Forced Aligner
   - Option 4: 1.7B + Forced Aligner

3. **Choose Language** (for options 1-2):
   - English or Hindi support

4. **Use the System**:
   - Click into any text field (browser, document, etc.)
   - Hold Right Alt key and speak
   - Release to automatically type the transcribed text
   - Press Escape to exit

## Troubleshooting

### Models fail to download
- Ensure stable internet connection
- Check if you have sufficient disk space (~5GB for largest model)
- Try running with VPN if in restricted region

### Audio not detected
- Check microphone permissions
- Ensure microphone is not used by other applications
- Try speaking louder and closer to microphone

### Text not typing
- Grant accessibility permissions (macOS)
- Run as administrator (Windows)
- Ensure target application accepts keyboard input

### Out of memory errors
- Use smaller model (0.6B instead of 1.7B)
- Close other applications to free RAM
- Restart the script to clear memory

## Hardware Requirements

### Minimum
- **CPU**: Modern multi-core processor (Intel 11th gen+ or AMD Zen 3+)
- **RAM**: 8 GB (for 0.6B model)
- **Storage**: 10 GB free space
- **Microphone**: Any USB/built-in microphone

### Recommended  
- **GPU**: NVIDIA RTX series or Apple Silicon M1/M2/M3
- **RAM**: 16 GB+ (for 1.7B model)
- **Storage**: SSD with 20 GB+ free space
- **Microphone**: USB headset or quality desktop microphone

## Model Information

| Model | Download Size | RAM (optimized) | Quality |
|-------|--------------|----------------|---------|
| Qwen3-ASR-0.6B | ~1.2 GB | ~2.5–3 GB | Good |
| Qwen3-ASR-1.7B | ~3.5 GB | ~4–5 GB | Excellent |
| + Forced Aligner | +1 GB | +0.5–1 GB | + Timestamps |

## Privacy & Security

- **Offline Processing**: All transcription happens locally
- **No Cloud Calls**: No data sent to external servers
- **Local Models**: Models downloaded once and cached locally
- **No Logging**: Audio and transcriptions are not saved