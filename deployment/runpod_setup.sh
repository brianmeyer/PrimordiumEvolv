#!/bin/bash
# RunPod Isaac Lab Setup Script
set -e

echo "🚀 Setting up PrimordiumEvolv on RunPod with Isaac Lab..."

# System info
echo "🖥️  System Information:"
nvidia-smi
python --version
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"

# Update system
apt-get update
apt-get install -y git curl wget vim htop tree build-essential

# Clone your repository
if [ ! -d "PrimordiumEvolv" ]; then
    echo "📥 Cloning PrimordiumEvolv repository..."
    git clone https://github.com/brianmeyer/PrimordiumEvolv.git
    cd PrimordiumEvolv
else
    echo "📂 Repository already exists, updating..."
    cd PrimordiumEvolv
    git pull
fi

# Install Isaac Lab
echo "🏗️  Installing Isaac Lab..."
if [ ! -d "IsaacLab" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    
    # Install Isaac Lab dependencies
    ./isaaclab.sh --install
    
    # Verify installation
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
    
    cd ..
fi

# Install PrimordiumEvolv dependencies
echo "📦 Installing PrimordiumEvolv dependencies..."
pip install -r requirements.txt

# Configure WandB
echo "🔧 Configuring Weights & Biases..."
echo "Please set your WandB API key:"
echo "export WANDB_API_KEY='your_key_here'"
echo "Or run: wandb login"

# Test SEAL implementation
echo "🧪 Testing SEAL implementation..."
WANDB_MODE=disabled python test_seal.py

echo "✅ RunPod setup complete!"
echo ""
echo "🎯 Ready to run experiments:"
echo "  1. Baseline: python scripts/run_baseline.py --steps 50000 --env Isaac-Reach-Franka-v0"
echo "  2. NAS: python scripts/run_nas.py --budget 10 --trial-steps 10000 --env Isaac-Reach-Franka-v0"  
echo "  3. SEAL: python scripts/run_continual.py --checkpoint baseline_checkpoint --steps 25000"