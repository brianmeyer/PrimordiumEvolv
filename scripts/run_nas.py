#!/usr/bin/env python3
"""
Script to run Neural Architecture Search.
"""
import argparse
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from primordium.phase0 import NeuralArchitectureSearch
from primordium.utils import create_output_dir, log_system_info


def main():
    parser = argparse.ArgumentParser(description="Run Neural Architecture Search")
    parser.add_argument("--budget", type=int, default=8, help="Number of trials to run")
    parser.add_argument("--trial-steps", type=int, default=10000, help="Steps per trial")
    parser.add_argument("--search-space", type=str, default="mlp,lstm,large_mlp,small_mlp", 
                       help="Comma-separated list of architectures to search")
    parser.add_argument("--env", type=str, default="Isaac-Lab-ArmReach-v0", help="Environment ID")
    parser.add_argument("--output-dir", type=str, default="runs", help="Base output directory")
    parser.add_argument("--project", type=str, default="primordium-phase0", help="WandB project name")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, "nas")
    
    # Log system info
    system_info = log_system_info()
    print("System Info:", system_info)
    
    # Initialize NAS
    nas = NeuralArchitectureSearch(env_id=args.env, project_name=args.project)
    
    # Parse search space
    search_space = args.search_space.split(",")
    
    # Run search
    print(f"Starting NAS with budget {args.budget}, trial steps {args.trial_steps}")
    print(f"Search space: {search_space}")
    
    best_model_path = nas.search(
        budget=args.budget,
        trial_steps=args.trial_steps,
        output_dir=output_dir,
        search_space=search_space
    )
    
    print(f"NAS complete! Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()