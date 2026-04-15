#!/usr/bin/env python3

from huggingface_hub import snapshot_download
import os

print("Downloading Qwen3-ASR-0.6B model manually...")
print("This may take several minutes depending on your internet connection.")

try:
    # Download the model to the default cache directory
    cache_dir = snapshot_download(
        repo_id="Qwen/Qwen3-ASR-0.6B",
        resume_download=True,  # Resume if partially downloaded
        local_files_only=False,
    )
    
    print(f"✅ Model downloaded successfully to: {cache_dir}")
    print("Now trying to load the model...")
    
    # Now try to load the model
    import torch
    from qwen_asr import Qwen3ASRModel
    
    device = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=torch_dtype,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    
    print("✅ Model loaded successfully!")
    print(f"Model device: {device}")
    print(f"Model dtype: {torch_dtype}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()