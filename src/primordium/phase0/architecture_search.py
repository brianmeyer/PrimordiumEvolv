"""
Neural Architecture Search module for finding optimal network architectures.
"""
import os
import random
import shutil
import wandb
from ray.rllib.algorithms import ppo
from typing import Dict, List, Tuple, Any


class NeuralArchitectureSearch:
    """Neural Architecture Search for reinforcement learning models."""
    
    def __init__(self, env_id: str = "Isaac-Lab-ArmReach-v0", project_name: str = "primordium-phase0"):
        self.env_id = env_id
        self.project_name = project_name
        self.architecture_configs = {
            "mlp": {"model": {"fcnet_hiddens": [256, 256]}},
            "lstm": {"model": {"use_lstm": True, "lstm_cell_size": 256}},
            "large_mlp": {"model": {"fcnet_hiddens": [512, 512, 256]}},
            "small_mlp": {"model": {"fcnet_hiddens": [128, 128]}},
        }
    
    def add_architecture(self, name: str, config: Dict[str, Any]) -> None:
        """Add a new architecture configuration to the search space."""
        self.architecture_configs[name] = config
    
    def run_trial(self, architecture: str, steps: int, output_dir: str) -> Tuple[float, str]:
        """
        Run a single architecture trial.
        
        Args:
            architecture: Architecture name from search space
            steps: Number of training steps
            output_dir: Directory to save trial results
        
        Returns:
            Tuple of (best_reward, checkpoint_path)
        """
        run_name = f"nas-{architecture}"
        wandb.init(project=self.project_name, name=run_name, dir=output_dir, reinit=True)
        
        # Extract model config and use new API
        arch_config = self.architecture_configs[architecture]
        model_config = arch_config.get("model", {})
        
        cfg = (
            ppo.PPOConfig()
            .environment(self.env_id)
            .rl_module(model_config=model_config)
            .env_runners(num_env_runners=0)
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        )
        
        algo = cfg.build_algo()
        best_reward = -1e9
        
        while algo._counters.get("timesteps_total", 0) < steps:
            result = algo.train()
            # Handle different possible result keys
            reward = result.get("episode_reward_mean", result.get("env_runners/episode_reward_mean", 0))
            timesteps = result.get("timesteps_total", result.get("counters/num_env_steps_sampled_lifetime", 0))
            best_reward = max(best_reward, reward)
            
            wandb.log({
                "episode_reward_mean": reward,
                "timesteps_total": timesteps,
                "best_reward": best_reward
            })
        
        checkpoint_path = algo.save(os.path.join(output_dir, "checkpoint"))
        wandb.finish()
        
        return best_reward, checkpoint_path
    
    def search(self, 
               budget: int,
               trial_steps: int,
               output_dir: str,
               search_space: List[str] = None) -> str:
        """
        Run neural architecture search.
        
        Args:
            budget: Number of trials to run
            trial_steps: Steps per trial
            output_dir: Directory to save all results
            search_space: List of architectures to search (default: all available)
        
        Returns:
            Path to best checkpoint
        """
        if search_space is None:
            search_space = list(self.architecture_configs.keys())
        
        os.makedirs(output_dir, exist_ok=True)
        
        trials = []
        
        for i in range(budget):
            architecture = random.choice(search_space)
            trial_dir = os.path.join(output_dir, f"trial_{i}_{architecture}")
            
            print(f"Running trial {i+1}/{budget}: {architecture}")
            reward, checkpoint = self.run_trial(architecture, trial_steps, trial_dir)
            trials.append((reward, checkpoint, architecture))
        
        # Find best trial
        best_reward, best_checkpoint, best_arch = max(trials, key=lambda x: x[0])
        best_path = os.path.join(output_dir, "best_model.pt")
        
        shutil.copy(best_checkpoint, best_path)
        
        # Save search results
        results = {
            "best_architecture": best_arch,
            "best_reward": best_reward,
            "all_trials": [(r, a) for r, _, a in trials]
        }
        
        import json
        with open(os.path.join(output_dir, "search_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Search complete! Best architecture: {best_arch} (reward: {best_reward:.3f})")
        print(f"Best model saved to: {best_path}")
        
        return best_path