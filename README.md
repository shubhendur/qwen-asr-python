# Qwen3 Voice-to-Text Dictation System

cd /Users/shubhendu.rohatgi/Documents/AI/qwen-asr-python
source qwen_env/bin/activate
python qwen_dictation.py

A comprehensive voice-to-text system using Qwen3-ASR models with global hotkey support.

## Features

- **Multiple Model Options**: Support for Qwen3-ASR-1.7B and 0.6B models
- **Forced Aligner Support**: Optional forced alignment capabilities  
- **Global Hotkeys**: Press and hold Right Alt to record speech
- **Direct Text Injection**: Types transcribed text directly without clipboard
- **Cross-Platform**: Works on macOS, Windows, and Linux
- **Hardware Acceleration**: Automatic GPU detection (CUDA/MPS/CPU)
- **Robust Error Handling**: Graceful failure recovery and user feedback

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Grant Permissions** (macOS):
   - Go to System Settings > Privacy & Security > Accessibility
   - Add Terminal (or your Python IDE) to allowed applications
   - Grant microphone access when prompted

3. **Grant Permissions** (Windows):
   - Run Terminal as Administrator for elevated applications
   - Allow microphone access when prompted

## Usage

1. **Run the Script**:
   ```bash
   python3 qwen_dictation.py
   ```

2. **Select Model Configuration**:
   - Option 1: Qwen3-ASR-1.7B (better quality, more VRAM)
   - Option 2: Qwen3-ASR-0.6B (faster, less VRAM)  
   - Option 3: 0.6B + Forced Aligner
   - Option 4: 1.7B + Forced Aligner

3. **Choose Language** (for options 1-2):
   - English or Hindi support

4. **Use the System**:
   - Click into any text field (browser, document, etc.)
   - Hold Right Alt key and speak
   - Release to automatically type the transcribed text
   - Press Escape to exit

## Performance Notes

- **First Run**: Models will download automatically (may take several minutes)
- **GPU Recommended**: NVIDIA GPUs with CUDA or Apple Silicon with MPS
- **RAM Requirements**: 
  - 0.6B model: ~4GB RAM
  - 1.7B model: ~8GB RAM
- **Audio Quality**: Use a good microphone for best results

## Troubleshooting

### "Command not found: python"
Use `python3` instead of `python` on macOS/Linux.

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
- Restart the script to clear GPU memory

## Hardware Requirements

### Minimum
- **CPU**: Modern multi-core processor
- **RAM**: 8GB (for 0.6B model)
- **Storage**: 10GB free space
- **Microphone**: Any USB/built-in microphone

### Recommended  
- **GPU**: NVIDIA RTX series or Apple Silicon M1/M2/M3
- **RAM**: 16GB+ (for 1.7B model)
- **Storage**: SSD with 20GB+ free space
- **Microphone**: USB headset or quality desktop microphone

## Model Information

| Model | Size | Speed | Quality | VRAM |
|-------|------|-------|---------|------|
| Qwen3-ASR-0.6B | 1.2GB | Fast | Good | ~2GB |
| Qwen3-ASR-1.7B | 3.5GB | Medium | Excellent | ~4GB |
| + Forced Aligner | +1GB | Slightly slower | + Timestamps | +1GB |

## Privacy & Security

- **Offline Processing**: All transcription happens locally
- **No Cloud Calls**: No data sent to external servers
- **Local Models**: Models downloaded once and cached locally
- **No Logging**: Audio and transcriptions are not saved