#!/usr/bin/env python3
"""
Script to run continual learning fine-tuning with EWC.
"""
import argparse
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from primordium.phase0 import ContinualRLTrainer
from primordium.utils import create_output_dir, log_system_info


def main():
    parser = argparse.ArgumentParser(description="Run continual learning with EWC")
    parser.add_argument("--checkpoint", required=True, help="Path to base model checkpoint")
    parser.add_argument("--steps", type=int, default=5000, help="Number of fine-tuning steps")
    parser.add_argument("--ewc-lambda", type=float, default=1000.0, help="EWC regularization strength")
    parser.add_argument("--env", type=str, default="Isaac-Lab-ArmReach-v0", help="Environment ID")
    parser.add_argument("--output-dir", type=str, default="runs", help="Base output directory")
    parser.add_argument("--run-name", type=str, default="continual", help="Run name for logging")
    parser.add_argument("--project", type=str, default="primordium-phase0", help="WandB project name")
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.run_name)
    
    # Log system info
    system_info = log_system_info()
    print("System Info:", system_info)
    
    # Initialize continual RL trainer
    trainer = ContinualRLTrainer(env_id=args.env, project_name=args.project)
    
    # Run continual fine-tuning
    print(f"Starting continual RL fine-tuning for {args.steps} steps...")
    print(f"Base checkpoint: {args.checkpoint}")
    print(f"EWC lambda: {args.ewc_lambda}")
    
    model_path, metrics = trainer.continual_fine_tune(
        checkpoint_path=args.checkpoint,
        steps=args.steps,
        output_dir=output_dir,
        ewc_lambda=args.ewc_lambda,
        run_name=args.run_name
    )
    
    print(f"Continual RL fine-tuning complete!")
    print(f"Fine-tuned model saved to: {model_path}")
    print(f"Training completed with EWC regularization.")


if __name__ == "__main__":
    main()