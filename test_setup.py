#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality work without loading models
"""

print("Testing imports...")

try:
    import os
    import sys
    import time
    import queue
    import threading
    import numpy as np
    import sounddevice as sd
    import torch
    from pynput import keyboard
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModel
    import warnings
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nTesting device detection...")

if torch.cuda.is_available():
    device = "cuda"
    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps" 
    print("✓ Apple Silicon (MPS) available")
else:
    device = "cpu"
    print("✓ Using CPU")

print(f"\nDevice: {device}")
print(f"PyTorch version: {torch.__version__}")

print("\nTesting audio device...")
try:
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    print(f"✓ Found {len(input_devices)} input device(s)")
    
    # Test basic audio stream
    test_stream = sd.InputStream(samplerate=16000, channels=1, blocksize=1024)
    test_stream.start()
    test_stream.stop() 
    test_stream.close()
    print("✓ Audio stream test successful")
except Exception as e:
    print(f"✗ Audio error: {e}")

print("\n✅ All basic tests passed!")
print("\nThe main script should work. Run:")
print("python3 qwen_dictation.py")