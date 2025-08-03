# PrimordiumEvolv

A self-evolving AI framework for reinforcement learning in simulation environments.

## Phase 0: Foundation Training

This phase implements baseline training algorithms and architecture search for the Isaac Lab arm reaching task.

### Components

- **phase0_arm_reach.py**: Baseline PPO training
- **phase0_nas.py**: Neural Architecture Search (MLP vs LSTM)
- **phase0_seal.py**: MIT SEAL continual learning fine-tuning

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run baseline training:
```bash
python examples/phase0_arm_reach.py --steps 50000
```

3. Run architecture search:
```bash
python examples/phase0_nas.py --budget 8 --trial_steps 10000
```

4. Fine-tune with SEAL:
```bash
python examples/phase0_seal.py --checkpoint runs/baseline/baseline.pt --steps 5000
```

### Requirements

- Isaac Lab simulation environment
- Ray RLlib for distributed training
- Weights & Biases for experiment tracking
- MIT SEAL for continual learning

### Next Phases

- Phase 1: Self-modification capabilities
- Phase 2: Autonomous architecture evolution
- Phase 3: Meta-learning and adaptation