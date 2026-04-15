#!/usr/bin/env python3

import torch
from qwen_asr import Qwen3ASRModel

print("Testing Qwen3-ASR model loading...")

# Better device detection
if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA detected: {torch.cuda.get_device_name()}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print("Apple Silicon (MPS) detected")
else:
    device = "cpu"
    print("Using CPU")

torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

print(f"\nLoading Qwen3-ASR-0.6B on {device.upper()} with dtype {torch_dtype}...")

try:
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=torch_dtype,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()