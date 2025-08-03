# Phase 0: Foundation Training

Phase 0 establishes the foundation for PrimordiumEvolv's self-evolving capabilities through baseline reinforcement learning, architecture search, and continual learning.

## Overview

The Phase 0 framework consists of three main components:

1. **Baseline Training** - Establishes performance baselines using standard PPO
2. **Neural Architecture Search** - Discovers optimal network architectures  
3. **Continual Learning** - Prevents catastrophic forgetting during adaptation

## Components

### BaselineTrainer (`src/primordium/phase0/baseline_training.py`)
- Standard PPO training for Isaac Lab environments
- Configurable network architectures
- Weights & Biases integration for experiment tracking
- Supports various Isaac Lab tasks

### NeuralArchitectureSearch (`src/primordium/phase0/architecture_search.py`)
- Automated search for optimal network architectures
- Supports MLP, LSTM, and custom architectures
- Random search with configurable budget
- Automatic best model selection and saving

### ContinualRLTrainer (`src/primordium/phase0/continual_learning.py`)
- Elastic Weight Consolidation (EWC) for continual learning
- Prevents catastrophic forgetting during fine-tuning
- Fisher Information Matrix computation for parameter importance
- Multi-task learning with regularization
- Evaluation of forgetting metrics across tasks

## Usage

### 1. Baseline Training
```bash
python scripts/run_baseline.py --steps 50000 --run-name my_baseline
```

### 2. Architecture Search
```bash
python scripts/run_nas.py --budget 8 --trial-steps 10000
```

### 3. Continual Learning with EWC
```bash
python scripts/run_continual.py --checkpoint runs/baseline/baseline.pt --steps 5000 --ewc-lambda 1000
```

## Configuration

Training parameters can be customized in `configs/training_config.yaml`:

- Model architectures and hyperparameters
- Training steps and search budgets
- Weights & Biases project settings

Environment-specific settings are in `configs/environments.yaml`:

- Environment descriptions and parameters
- Training hyperparameters per environment
- Success thresholds and episode limits

## Dependencies

Core dependencies:
- Ray RLlib for distributed RL training
- Isaac Lab for simulation environments
- Weights & Biases for experiment tracking
- PyTorch for EWC continual learning implementation

See `requirements.txt` for complete dependency list.

## Next Phases

Phase 0 provides the foundation for:
- **Phase 1**: Self-modification capabilities
- **Phase 2**: Autonomous architecture evolution  
- **Phase 3**: Meta-learning and adaptation

The modular design allows each phase to build upon previous capabilities while maintaining backward compatibility.