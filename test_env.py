#!/usr/bin/env python3

print("Testing qwen-asr import...")

try:
    from qwen_asr import Qwen3ASRModel
    print("✓ qwen_asr import successful")
    
    import torch
    print(f"✓ torch version: {torch.__version__}")
    
    import sounddevice as sd
    print("✓ sounddevice import successful")
    
    import pynput
    print("✓ pynput import successful")
    
    import numpy as np
    print("✓ numpy import successful")
    
    print("\n--- All imports successful! ---")
    print("Testing device detection...")
    
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
    
    print("\n✅ Environment test completed successfully!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()