#!/usr/bin/env python3
"""
Cohere Backend Test Script - Verifies installation and authentication

This script tests the Cohere integration without actually loading the model.
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test if basic dependencies can be imported."""
    print("=== Testing Basic Imports ===")

    missing = []
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("✗ torch - MISSING")

    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
        print("✗ transformers - MISSING")

    try:
        import soundfile
        print("✓ soundfile")
    except ImportError:
        missing.append("soundfile")
        print("✗ soundfile - MISSING")

    try:
        import librosa
        print("✓ librosa")
    except ImportError:
        missing.append("librosa")
        print("✗ librosa - MISSING")

    for pkg in ["sentencepiece", "huggingface_hub"]:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"✗ {pkg} - MISSING")

    # Test protobuf specifically (import name is different)
    try:
        import google.protobuf
        print("✓ protobuf")
    except ImportError:
        missing.append("protobuf")
        print("✗ protobuf - MISSING")

    return missing


def test_huggingface_auth():
    """Test HuggingFace authentication."""
    print("\n=== Testing HuggingFace Authentication ===")

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        try:
            user_info = api.whoami()
            print(f"✓ Authenticated as: {user_info['name']}")
            return True
        except Exception:
            print("✗ Not authenticated")
            print("  Run: huggingface-cli login")
            return False
    except ImportError:
        print("✗ huggingface_hub not available")
        return False


def test_model_access():
    """Test access to the Cohere model."""
    print("\n=== Testing Model Access ===")

    model_id = "CohereLabs/cohere-transcribe-03-2026"

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        print(f"✓ Model config loaded: {config.model_type}")
        return True

    except Exception as e:
        error_str = str(e)
        if "gated" in error_str.lower() or "401" in error_str:
            print("✗ Model requires access approval")
            print(f"  Visit: https://huggingface.co/{model_id}")
            print("  Request access and run: huggingface-cli login")
            return False
        else:
            print(f"✗ Model access error: {error_str}")
            return False


def test_backend_creation():
    """Test creating a Cohere backend instance."""
    print("\n=== Testing Backend Creation ===")

    try:
        from cohere_transcribe_backend import check_requirements

        ok, missing = check_requirements("pytorch")
        if not ok:
            print(f"✗ Requirements not met: {missing}")
            return False

        # Don't actually create the backend to avoid authentication issues
        print("✓ Backend requirements satisfied")
        print("  (Skipping actual model loading to avoid auth requirements)")
        return True

    except Exception as e:
        print(f"✗ Backend creation test failed: {e}")
        return False


def main():
    """Run all tests and provide setup guidance."""
    print("Cohere Backend Integration Test")
    print("=" * 50)

    # Test imports
    missing_deps = test_basic_imports()

    # Test auth
    auth_ok = test_huggingface_auth()

    # Test model access
    model_ok = test_model_access()

    # Test backend
    backend_ok = test_backend_creation()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install -r requirements-cohere.txt")
    else:
        print("✅ All dependencies available")

    if not auth_ok:
        print("❌ HuggingFace authentication required")
        print("   Run: huggingface-cli login")
    else:
        print("✅ HuggingFace authentication OK")

    if not model_ok:
        print("❌ Model access restricted")
        print("   Request access at: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026")
    else:
        print("✅ Model access OK")

    if backend_ok:
        print("✅ Backend creation ready")
    else:
        print("❌ Backend creation issues")

    # Overall status
    all_ok = not missing_deps and auth_ok and model_ok and backend_ok

    if all_ok:
        print("\n🎉 All tests passed! Cohere integration ready.")
        print("   Run: python qwen_dictation.py")
        print("   Select option 17, 18, or 19 for Cohere backends")
    else:
        print("\n⚠️  Setup incomplete. Address issues above to use Cohere backends.")
        print("   Alternatively, use other ASR options (1-16) which don't require authentication.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())