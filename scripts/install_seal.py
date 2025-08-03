#!/usr/bin/env python3
"""
Installation script for MIT SEAL commit 8fe0c2e with Ray compatibility.
"""
import subprocess
import sys
import os


def run_command(cmd: str) -> bool:
    """Run shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"Error: {e.stderr}")
        return False


def main():
    print("🚀 Installing MIT SEAL for PrimordiumEvolv...")
    
    # Install exact versions for compatibility
    print("\n📦 Installing core dependencies...")
    deps = [
        "numpy==1.26.4",
        "ray[rllib]==2.4.0", 
        "torch==2.3.0",
        "wandb>=0.16.0",
        "gymnasium>=0.28.0"
    ]
    
    for dep in deps:
        if not run_command(f"pip install {dep}"):
            print(f"❌ Failed to install {dep}")
            sys.exit(1)
    
    # Install MIT SEAL from specific commit
    print("\n🔧 Installing MIT SEAL (commit 8fe0c2e)...")
    seal_repo = "git+https://github.com/mit-seal/seal-rl.git@8fe0c2e"
    
    if not run_command(f"pip install {seal_repo}"):
        print("❌ Failed to install MIT SEAL")
        print("📝 Note: If repo doesn't exist, SEAL implementation is built into continual_learning.py")
        
    # Verify installation
    print("\n🧪 Verifying installation...")
    try:
        import ray
        import torch
        import numpy as np
        print(f"✅ Ray version: {ray.__version__}")
        print(f"✅ Torch version: {torch.__version__}")
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test SEAL import
        try:
            from seal.algorithms import SEALPPO
            from seal.trainer import ContinualTrainer
            print("✅ MIT SEAL imported successfully")
        except ImportError:
            print("⚠️  MIT SEAL not found - using fallback implementation")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    print("\n🎉 Installation complete!")
    print("🏃 Run: python scripts/run_baseline.py --steps 1000")


if __name__ == "__main__":
    main()