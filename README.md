# PrimordiumEvolv

A self-evolving AI framework for reinforcement learning in simulation environments.

## Project Structure

```
PrimordiumEvolv/
├── src/primordium/           # Core framework modules
│   ├── phase0/              # Phase 0 implementations
│   └── utils/               # Common utilities
├── scripts/                 # Executable scripts
├── configs/                 # Configuration files
├── docs/                    # Documentation
└── tests/                   # Test suite
```

## Phase 0: Foundation Training

Phase 0 establishes the foundation through baseline RL, architecture search, and continual learning.

### Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run baseline training:**
```bash
python scripts/run_baseline.py --steps 50000
```

3. **Run architecture search:**
```bash
python scripts/run_nas.py --budget 8 --trial-steps 10000
```

4. **Fine-tune with continual learning:**
```bash
python scripts/run_continual.py --checkpoint runs/baseline/baseline.pt --steps 5000 --ewc-lambda 1000
```

### Components

- **BaselineTrainer**: Standard PPO training with configurable architectures
- **NeuralArchitectureSearch**: Automated discovery of optimal network structures
- **ContinualRLTrainer**: Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting

### Configuration

Customize training in `configs/training_config.yaml` and environment settings in `configs/environments.yaml`.

### Documentation

See `docs/phase0.md` for detailed documentation.

## Requirements

- **Isaac Lab**: Simulation environment
- **Ray RLlib**: Distributed RL training
- **Weights & Biases**: Experiment tracking
- **PyTorch**: Deep learning framework for EWC implementation

## Roadmap

- **Phase 0**: Foundation training ✅
- **Phase 1**: Self-modification capabilities
- **Phase 2**: Autonomous architecture evolution
- **Phase 3**: Meta-learning and adaptation