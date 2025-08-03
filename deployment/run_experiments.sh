#!/bin/bash
# Complete Experiment Pipeline for RunPod
set -e

# Configuration
ENV_NAME="Isaac-Reach-Franka-v0"  # Change this to your target Isaac Lab environment
PROJECT_NAME="primordium-robot-proof"
OUTPUT_DIR="experiments/$(date +%Y%m%d_%H%M%S)"

echo "🎯 Running PrimordiumEvolv Proof-of-Concept Pipeline"
echo "Environment: $ENV_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "=================="

mkdir -p $OUTPUT_DIR

# Stage 1: Baseline PPO Training
echo "📊 Stage 1: Baseline PPO Training (50k steps)"
python scripts/run_baseline.py \
    --steps 50000 \
    --env $ENV_NAME \
    --output-dir $OUTPUT_DIR/baseline \
    --run-name baseline_ppo \
    --project $PROJECT_NAME

# Stage 2: Neural Architecture Search
echo "🧠 Stage 2: Neural Architecture Search (10 trials)"
python scripts/run_nas.py \
    --budget 10 \
    --trial-steps 10000 \
    --env $ENV_NAME \
    --output-dir $OUTPUT_DIR/nas \
    --project $PROJECT_NAME

# Stage 3: SEAL Continual Learning (using baseline checkpoint)
echo "🔄 Stage 3: SEAL Continual Learning (25k steps)"
BASELINE_CHECKPOINT=$(find $OUTPUT_DIR/baseline -name "*.pt" -o -name "checkpoint*" | head -1)

if [ -n "$BASELINE_CHECKPOINT" ]; then
    python scripts/run_continual.py \
        --checkpoint $BASELINE_CHECKPOINT \
        --steps 25000 \
        --replay-ratio 0.3 \
        --env $ENV_NAME \
        --output-dir $OUTPUT_DIR/seal \
        --run-name seal_continual \
        --project $PROJECT_NAME
else
    echo "⚠️  No baseline checkpoint found, using mock checkpoint"
    python scripts/run_continual.py \
        --checkpoint $OUTPUT_DIR/baseline \
        --steps 25000 \
        --replay-ratio 0.3 \
        --env $ENV_NAME \
        --output-dir $OUTPUT_DIR/seal \
        --run-name seal_continual \
        --project $PROJECT_NAME
fi

# Stage 4: Generate Performance Report
echo "📈 Stage 4: Generating Performance Report"
python deployment/generate_report.py --experiment-dir $OUTPUT_DIR

echo "✅ Experiment Pipeline Complete!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo "🌐 View results at: https://wandb.ai/your-username/$PROJECT_NAME"