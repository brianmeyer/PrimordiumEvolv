#!/usr/bin/env python3
"""
Script to run baseline PPO training.
"""
import argparse
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from primordium.phase0 import BaselineTrainer
from primordium.utils import create_output_dir, log_system_info


def main():
    parser = argparse.ArgumentParser(description="Run baseline PPO training")
    parser.add_argument("--steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--env", type=str, default="Isaac-Lab-ArmReach-v0", help="Environment ID")
    parser.add_argument("--output-dir", type=str, default="runs", help="Base output directory")
    parser.add_argument("--run-name", type=str, default="baseline", help="Run name for logging")
    parser.add_argument("--project", type=str, default="primordium-phase0", help="WandB project name")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.run_name)
    
    # Log system info
    system_info = log_system_info()
    print("System Info:", system_info)
    
    # Initialize trainer
    trainer = BaselineTrainer(env_id=args.env, project_name=args.project)
    
    # Run training
    print(f"Starting baseline training for {args.steps} steps...")
    checkpoint_path = trainer.train(args.steps, output_dir, args.run_name)
    
    print(f"Training complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()