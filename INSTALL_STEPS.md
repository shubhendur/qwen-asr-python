# STEP-BY-STEP SOLUTION FOR DEPENDENCY CONFLICTS

## Problem Analysis
You have these conflicts:
1. qwen-asr requires transformers==4.57.6 (EXACT)
2. Some packages tried to upgrade to transformers 5.6.2
3. numpy version conflicts between packages

## SOLUTION: Install Cohere packages manually to avoid conflicts

# Step 1: Ensure we have the right base versions
pip install transformers==4.57.6 --force-reinstall  # Keep qwen-asr happy
pip install "numpy>=1.21.0,<2.1.0" --upgrade        # Compatible range

# Step 2: Install Cohere dependencies individually (no conflicts)
pip install soundfile librosa resampy
pip install sentencepiece "protobuf>=3.19.0,<5.0.0"
pip install "huggingface_hub>=0.16.0,<0.26.0"
pip install "tokenizers>=0.13.0,<0.16.0"
pip install safetensors requests psutil packaging

# Step 3: Skip problematic optional packages for now
# Don't install vLLM or MLX until base setup works

## Test the setup
python3 test_cohere_setup.py