#!/bin/bash
set -e

echo "🚀 Setting up PrimordiumEvolv development environment..."

# Update system
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tree

# Upgrade pip
python -m pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional development tools
pip install \
    jupyterlab \
    tensorboard \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly

# Set up Git (if not already configured)
if [ -z "$(git config --global user.name)" ]; then
    echo "⚠️  Git user not configured. Please run:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
fi

# Create development directories
mkdir -p {experiments,logs,checkpoints,notebooks}

# Install SEAL dependencies (development mode - no actual training)
echo "🔧 Setting up SEAL development environment..."
pip install optuna hyperopt

# Create a simple test to verify installation
echo "🧪 Running installation verification..."
python -c "
import torch
import numpy as np
import ray
import wandb
print('✅ Core dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Ray: {ray.__version__}')
"

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Configure Git if needed"
echo "  2. Test the SEAL implementation: python test_seal.py"
echo "  3. Start Jupyter: jupyter lab --ip=0.0.0.0 --no-browser"
echo "  4. Ready for RunPod deployment scripts"