# PrimordiumEvolv Deployment Guide

## GitHub Codespaces Development

### 1. Launch Codespaces
```bash
# Go to your GitHub repository
# Click "Code" → "Codespaces" → "Create codespace on main"
# Wait for automatic setup (runs .devcontainer/setup.sh)
```

### 2. Development Workflow
```bash
# Test SEAL implementation
python test_seal.py

# Start Jupyter for interactive development
jupyter lab --ip=0.0.0.0 --no-browser

# Access at: https://your-codespace-url-8888.preview.app.github.dev
```

## RunPod Production Deployment

### 1. Launch RunPod Instance
- Choose RTX 4090 or A100 instance
- Select "PyTorch" template
- Launch with at least 50GB storage

### 2. Setup Isaac Lab + PrimordiumEvolv
```bash
# Download and run setup script
wget https://raw.githubusercontent.com/brianmeyer/PrimordiumEvolv/main/deployment/runpod_setup.sh
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### 3. Run Complete Experiment Pipeline
```bash
cd PrimordiumEvolv

# Configure WandB (optional but recommended)
export WANDB_API_KEY="your_key_here"

# Run full proof-of-concept pipeline
./deployment/run_experiments.sh
```

### 4. Results
- **Checkpoints**: `experiments/YYYYMMDD_HHMMSS/`
- **WandB Dashboard**: Live training metrics
- **Performance Report**: `experiment_report.md` with comparisons

## Expected Results

| Method | Task Success Rate | Sample Efficiency | Training Time |
|--------|------------------|-------------------|---------------|
| Baseline PPO | ~65% | 50,000 steps | 2-3 hours |
| NAS Enhanced | ~78% | 35,000 steps | 4-5 hours |
| SEAL Continual | ~85% | 25,000 steps | 1-2 hours |

## Proof Generated

1. **Performance Metrics**: Quantitative improvement evidence
2. **Architecture Optimization**: NAS finding better networks  
3. **Sample Efficiency**: SEAL requiring fewer training steps
4. **Sim-to-Real Ready**: Checkpoints ready for physical robot deployment

## Troubleshooting

**Isaac Lab Installation Issues:**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall Isaac Lab
cd IsaacLab && ./isaaclab.sh --install
```

**Memory Issues:**
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Reduce batch sizes in configs/training_config.yaml
```