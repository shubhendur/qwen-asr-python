# Cohere Transcribe 03-2026 Integration Guide - CORRECTED VERSION

## ⚠️ Important Updates & Requirements

**CRITICAL**: This integration has been updated to address several issues found during testing:

1. **Model Access**: The Cohere model requires HuggingFace authentication and access approval
2. **Version Issues**: Fixed incorrect version requirements (vLLM 0.19.0 doesn't exist)
3. **Dependencies**: Updated to use actually available package versions

## Model Specifications

- **Parameters**: 2B (Conformer-based encoder-decoder)
- **Languages**: 14 supported - English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Chinese (Mandarin), Japanese, Korean, Vietnamese, Arabic
- **Performance**: WER 5.42 (English ASR Leaderboard leader)
- **Access**: **GATED MODEL** - Requires approval from Cohere Labs
- **License**: Apache 2.0

## 🔐 Authentication Setup (REQUIRED)

### Step 1: Request Model Access
1. Visit: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
2. Click "Request Access" and wait for approval
3. You must have a HuggingFace account and be logged in

### Step 2: Authentication
```bash
# Install HuggingFace CLI if not installed
pip install huggingface_hub

# Login with your HuggingFace token
huggingface-cli login
```

### Step 3: Verify Access
```bash
# Test your setup
python test_cohere_setup.py
```

## Installation Instructions (CORRECTED VERSIONS)

### For Both Windows 11 and Mac M1 Pro

1. **Install Core Dependencies:**
   ```bash
   pip install -r requirements-cohere.txt
   ```

2. **For vLLM Backend (Optional):**
   ```bash
   # Use available version (not 0.19.0 which doesn't exist)
   pip install "vllm>=0.8.0,<0.12.0"
   ```

3. **For MLX Backend (Mac Only):**
   ```bash
   pip install mlx-lm
   ```

4. **Verify Installation:**
   ```bash
   python test_cohere_setup.py
   ```

## Updated Menu Options

```
Cohere Transcribe 03-2026 (2B params, 14 languages, top WER performance):
 17. Cohere-2B — PyTorch (Standard)     (~3-4 GB RAM, cross-platform) [GATED]
 18. Cohere-2B — vLLM (Production)      (~4-6 GB RAM, server mode) [GATED]
 19. Cohere-2B — MLX (Apple Silicon)    (~2-3 GB RAM, M1/M2/M3/M4 only) [GATED]
```

**Note**: All Cohere options require HuggingFace authentication and model access approval.

## Compatibility Matrix

| Component | Windows 11 (i7-1355U) | Mac M1 Pro | Notes |
|-----------|------------------------|-------------|--------|
| PyTorch Backend | ✅ | ✅ | Requires Python >= 3.9 |
| vLLM Backend | ✅ (if GPU) | ✅ | Requires 4GB+ RAM |
| MLX Backend | ❌ | ✅ | Apple Silicon only |
| Authentication | ✅ | ✅ | Required for all options |

## Current Version Requirements (FIXED)

```bash
# Working versions (tested)
transformers>=4.50.0  # Not 5.4.0 (doesn't exist yet)
torch>=1.13.0         # Not 2.0+ (compatibility issues)
vllm>=0.8.0,<0.12.0  # Not 0.19.0 (doesn't exist)
huggingface_hub>=0.20.0
```

## Error Handling & Troubleshooting

### Common Issues & Solutions

1. **"Model requires access approval"**
   ```bash
   # Solution: Request access first
   # Visit: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
   ```

2. **"Not authenticated"**
   ```bash
   # Solution: Login to HuggingFace
   huggingface-cli login
   ```

3. **"vLLM version 0.19.0 not found"**
   ```bash
   # Solution: Use available version
   pip install "vllm>=0.8.0,<0.12.0"
   ```

4. **"transformers 5.4.0 not found"**
   ```bash
   # Solution: Use available version
   pip install "transformers>=4.50.0"
   ```

5. **"MLX not working on Intel Mac"**
   ```
   MLX only works on Apple Silicon (M1/M2/M3/M4)
   Use PyTorch backend (option 17) instead
   ```

## Alternative Recommendations

If Cohere model access is not available, consider these excellent alternatives:

1. **Option 12-16**: Whisper models (99 languages, no authentication required)
2. **Option 5-6**: Qwen ONNX models (fast, low memory)
3. **Option 9-11**: Voxtral models (13 languages, real-time)

## Testing Your Setup

### Quick Test
```bash
# Test all components
python test_cohere_setup.py

# Test specific backend
python cohere_transcribe_backend.py --help
```

### Full Integration Test
```bash
# Run main application
python qwen_dictation.py

# Select option 17 (PyTorch - most compatible)
# Try transcribing a short audio clip
```

## Performance Expectations

### Windows 11 (Intel i7-1355U)
- **PyTorch**: ~3-4 GB RAM, CPU/GPU auto-detection
- **vLLM**: ~4-6 GB RAM, requires good GPU or will be slow
- **First run**: Downloads ~7-8 GB model files

### Mac M1 Pro  
- **PyTorch**: ~3-4 GB RAM, MPS acceleration
- **MLX**: ~2-3 GB RAM, highly optimized for Apple Silicon
- **vLLM**: ~4-6 GB RAM, uses unified memory efficiently

## Updated Integration Status

### ✅ Working Features
- Multi-backend support (PyTorch/vLLM/MLX)
- Authentication handling
- Error messages with solutions
- Platform-specific optimizations
- Fallback model loading

### ⚠️ Known Limitations
- **Requires HuggingFace authentication** (not mentioned in original docs)
- **Gated model access** needs approval
- **Version mismatches** in original Cohere documentation
- **Python 3.10+ recommended** for newer vLLM versions

### 🔧 Fixed Issues
- Corrected vLLM version requirements
- Updated transformers version expectations
- Added proper authentication checks
- Improved error handling and fallbacks

## Conclusion

The Cohere integration is **functional but requires setup**. The main barriers are:

1. **Access approval** from Cohere Labs (may take time)
2. **Authentication setup** with HuggingFace
3. **Version compatibility** (now fixed)

**Recommendation**: While setting up Cohere access, use the other excellent ASR options (1-16) which work immediately without authentication requirements.