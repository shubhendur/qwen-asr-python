#!/usr/bin/env python3

from huggingface_hub import snapshot_download
import os
import sys

print("--- Qwen3 Model Downloader ---")
print("1. Qwen3-ASR-1.7B")
print("2. Qwen3-ASR-0.6B")
print("3. Qwen3-ASR-0.6B with Qwen3-ForcedAligner-0.6B")
print("4. Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B")

try:
    choice = input("Select Model Configuration to download (1-4): ").strip()
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Using default: Qwen3-ASR-0.6B")
        choice = '2'
except KeyboardInterrupt:
    print("\n[System] Exiting...")
    sys.exit(0)

# Map choices to the required model repo IDs
asr_model_map = {
    '1': "Qwen/Qwen3-ASR-1.7B",
    '2': "Qwen/Qwen3-ASR-0.6B",
    '3': "Qwen/Qwen3-ASR-0.6B",
    '4': "Qwen/Qwen3-ASR-1.7B",
}

ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ASR_MODEL_ID = asr_model_map[choice]
need_aligner = choice in ['3', '4']

# Build list of repos to download
repos_to_download = [ASR_MODEL_ID]
if need_aligner:
    repos_to_download.append(ALIGNER_MODEL_ID)

print(f"\nWill download: {', '.join(repos_to_download)}")
print("This may take several minutes depending on your internet connection.\n")

try:
    for repo_id in repos_to_download:
        print(f"Downloading {repo_id}...")
        cache_dir = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
            local_files_only=False,
        )
        print(f"✅ {repo_id} downloaded to: {cache_dir}\n")

    print("Now trying to load the model to verify...")

    import torch
    from qwen_asr import Qwen3ASRModel

    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Build load kwargs — only add forced_aligner when needed
    load_kwargs = dict(
        dtype=torch_dtype,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

    if need_aligner:
        load_kwargs["forced_aligner"] = ALIGNER_MODEL_ID
        load_kwargs["forced_aligner_kwargs"] = dict(
            dtype=torch_dtype,
            device_map=device,
        )

    model = Qwen3ASRModel.from_pretrained(ASR_MODEL_ID, **load_kwargs)

    print("✅ Model loaded and verified successfully!")
    print(f"   ASR Model : {ASR_MODEL_ID}")
    if need_aligner:
        print(f"   Aligner   : {ALIGNER_MODEL_ID}")
    print(f"   Device    : {device}")
    print(f"   Dtype     : {torch_dtype}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()