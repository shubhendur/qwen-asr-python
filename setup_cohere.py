#!/usr/bin/env python3
"""
Cohere Setup Script - Handles dependency conflicts properly
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def check_current_packages():
    """Check current package state."""
    print("=== Current Package Status ===")

    packages_to_check = [
        "numpy", "transformers", "torch", "qwen-asr",
        "pandas", "parakeet-mlx", "soundfile", "librosa"
    ]

    for pkg in packages_to_check:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Extract version
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':', 1)[1].strip()
                        print(f"✓ {pkg}: {version}")
                        break
            else:
                print(f"✗ {pkg}: Not installed")
        except Exception as e:
            print(f"? {pkg}: Error checking ({e})")

def fix_dependencies():
    """Fix dependency conflicts step by step."""
    print("\n=== Fixing Dependencies ===")

    # Step 1: Check if we have transformers conflicts
    print("1. Checking transformers compatibility...")
    try:
        import transformers
        version = transformers.__version__
        print(f"Current transformers: {version}")

        # Check if qwen-asr is installed
        qwen_result = subprocess.run([sys.executable, "-m", "pip", "show", "qwen-asr"],
                                   capture_output=True)
        if qwen_result.returncode == 0:
            print("qwen-asr detected - keeping transformers 4.57.6 for compatibility")

            # Force reinstall transformers to exact version if needed
            if version != "4.57.6":
                print("Fixing transformers version for qwen-asr compatibility...")
                if not run_command([sys.executable, "-m", "pip", "install", "transformers==4.57.6", "--force-reinstall"]):
                    print("Warning: Could not fix transformers version")

    except ImportError:
        print("transformers not installed - will install compatible version")

    # Step 2: Handle numpy conflicts
    print("2. Checking numpy compatibility...")
    try:
        import numpy
        version = numpy.__version__
        print(f"Current numpy: {version}")

        # Check if we have conflicting packages
        conflict_packages = []

        # Check parakeet-mlx
        parakeet_result = subprocess.run([sys.executable, "-m", "pip", "show", "parakeet-mlx"],
                                       capture_output=True)
        if parakeet_result.returncode == 0:
            print("parakeet-mlx detected - may cause numpy conflicts")
            conflict_packages.append("parakeet-mlx")

        # Check pandas version
        pandas_result = subprocess.run([sys.executable, "-m", "pip", "show", "pandas"],
                                     capture_output=True)
        if pandas_result.returncode == 0:
            pandas_output = pandas_result.stdout
            for line in pandas_output.split('\n'):
                if line.startswith('Version:'):
                    pandas_version = line.split(':', 1)[1].strip()
                    if pandas_version.startswith('3.'):
                        print(f"pandas {pandas_version} may require newer numpy")
                        conflict_packages.append("pandas")
                    break

        if conflict_packages:
            print(f"Potential conflicts with: {', '.join(conflict_packages)}")
            print("Consider updating these packages or using a virtual environment")

    except ImportError:
        print("numpy not installed - will install compatible version")

def install_cohere_requirements():
    """Install Cohere requirements with conflict resolution."""
    print("\n=== Installing Cohere Requirements ===")

    requirements_file = Path(__file__).parent / "requirements-cohere.txt"

    if not requirements_file.exists():
        print(f"Error: {requirements_file} not found")
        return False

    # Try installing with no-deps first to avoid conflicts
    print("Installing with --no-deps to avoid conflicts...")
    success = run_command([
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements_file),
        "--no-deps"
    ], check=False)

    if success:
        print("✓ Installed without dependencies")

        # Now install missing dependencies individually
        print("Installing missing dependencies individually...")

        # Core dependencies that are usually safe
        safe_packages = [
            "soundfile", "librosa", "resampy", "requests", "psutil", "packaging"
        ]

        for pkg in safe_packages:
            run_command([sys.executable, "-m", "pip", "install", pkg], check=False)

    else:
        print("Installation with --no-deps failed, trying regular install...")
        success = run_command([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file)
        ], check=False)

    return success

def verify_installation():
    """Verify that everything works."""
    print("\n=== Verifying Installation ===")

    try:
        # Test our backend
        from cohere_transcribe_backend import check_requirements

        ok, missing = check_requirements("pytorch")

        if ok:
            print("✅ All requirements satisfied for Cohere backend")
            return True
        else:
            print(f"❌ Missing requirements: {missing}")
            return False

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main setup process."""
    print("Cohere Backend Setup - Dependency Conflict Resolution")
    print("=" * 60)

    # Check current state
    check_current_packages()

    # Fix dependencies
    fix_dependencies()

    # Install requirements
    if install_cohere_requirements():
        print("\n✅ Requirements installation completed")
    else:
        print("\n❌ Requirements installation failed")
        print("\nTry manual installation:")
        print("1. pip install --upgrade pip")
        print("2. pip install --no-deps -r requirements-cohere.txt")
        print("3. pip install soundfile librosa")
        return 1

    # Verify
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_cohere_setup.py")
        print("2. Setup authentication: huggingface-cli login")
        print("3. Request model access: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026")
        return 0
    else:
        print("\n⚠️ Setup completed but verification failed")
        print("Check the error messages above and install missing packages manually")
        return 1

if __name__ == "__main__":
    sys.exit(main())