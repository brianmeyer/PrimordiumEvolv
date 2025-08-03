"""
Baseline PPO training module for Isaac Lab environments.
"""
import os
import wandb
from ray.rllib.algorithms import ppo
from typing import Optional


class BaselineTrainer:
    """Baseline PPO trainer for reinforcement learning tasks."""
    
    def __init__(self, env_id: str = "Isaac-Lab-ArmReach-v0", project_name: str = "primordium-phase0"):
        self.env_id = env_id
        self.project_name = project_name
        self.algo = None
    
    def setup_config(self, model_config: Optional[dict] = None) -> ppo.PPOConfig:
        """Setup PPO configuration."""
        if model_config is None:
            model_config = {"fcnet_hiddens": [256, 256]}
        
        return (
            ppo.PPOConfig()
            .environment(env=self.env_id)
            .rl_module(model_config=model_config)
            .env_runners(num_env_runners=0)
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        )
    
    def train(self, steps: int, logdir: str, run_name: str = "baseline") -> str:
        """
        Train the baseline model.
        
        Args:
            steps: Number of training steps
            logdir: Output directory for logs and models
            run_name: Name for wandb run
        
        Returns:
            Path to saved model checkpoint
        """
        os.makedirs(logdir, exist_ok=True)
        
        wandb.init(project=self.project_name, name=run_name, dir=logdir)
        
        cfg = self.setup_config()
        self.algo = cfg.build_algo()
        
        while self.algo._counters.get("timesteps_total", 0) < steps:
            result = self.algo.train()
            # Handle different possible result keys
            episode_reward = result.get("episode_reward_mean", result.get("env_runners/episode_reward_mean", 0))
            timesteps = result.get("timesteps_total", result.get("counters/num_env_steps_sampled_lifetime", 0))
            
            wandb.log({
                "episode_reward_mean": episode_reward,
                "timesteps_total": timesteps,
            })
        
        checkpoint_path = os.path.join(logdir, "baseline.pt")
        self.algo.save(checkpoint_path)
        wandb.finish()
        
        return checkpoint_path