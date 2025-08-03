#!/usr/bin/env python3
"""
Script to run MIT SEAL continual learning fine-tuning.
Uses commit 8fe0c2e with 70% on-policy + 30% replay buffer.
"""
import argparse
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from primordium.phase0 import SEALTrainer, SEALQuantizer
from primordium.utils import create_output_dir, log_system_info


def main():
    parser = argparse.ArgumentParser(description="Run MIT SEAL continual learning")
    parser.add_argument("--checkpoint", required=True, help="Path to base PPO model checkpoint")
    parser.add_argument("--steps", type=int, default=5000, help="Number of SEAL fine-tuning steps")
    parser.add_argument("--replay-ratio", type=float, default=0.3, 
                       help="Replay buffer ratio (default 0.3 = 30%% replay, 70%% on-policy)")
    parser.add_argument("--quantize", action="store_true", 
                       help="Quantize model to INT8 after training (Phase 1 prep)")
    parser.add_argument("--env", type=str, default="Isaac-Lab-ArmReach-v0", help="Environment ID")
    parser.add_argument("--output-dir", type=str, default="runs", help="Base output directory")
    parser.add_argument("--run-name", type=str, default="seal", help="Run name for logging")
    parser.add_argument("--project", type=str, default="primordium-phase0", help="WandB project name")
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.run_name)
    
    # Log system info
    system_info = log_system_info()
    print("🔧 System Info:", system_info)
    
    try:
        # Initialize SEAL trainer
        trainer = SEALTrainer(env_id=args.env, project_name=args.project)
        
        # Run SEAL fine-tuning
        print(f"🚀 Starting MIT SEAL fine-tuning for {args.steps} steps...")
        print(f"📁 Base checkpoint: {args.checkpoint}")
        print(f"🔄 Replay ratio: {args.replay_ratio:.0%} replay, {(1-args.replay_ratio):.0%} on-policy")
        
        model_path, metrics = trainer.seal_fine_tune(
            checkpoint_path=args.checkpoint,
            steps=args.steps,
            output_dir=output_dir,
            replay_ratio=args.replay_ratio,
            run_name=args.run_name
        )
        
        print(f"✅ SEAL fine-tuning complete!")
        print(f"💾 Model saved to: {model_path}")
        
        # Optional INT8 quantization for Phase 1
        if args.quantize:
            print(f"🔄 Quantizing model to INT8...")
            quantized_path = os.path.join(output_dir, "seal_tuned_int8.pt")
            SEALQuantizer.quantize_model(model_path, quantized_path)
            print(f"✅ Quantized model ready for Phase 1: {quantized_path}")
        
    except ImportError as e:
        print(f"❌ MIT SEAL not available: {e}")
        print(f"📦 Install with: pip install git+https://github.com/mit-seal/seal-rl.git@8fe0c2e")
        sys.exit(1)
    except Exception as e:
        print(f"❌ SEAL training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()