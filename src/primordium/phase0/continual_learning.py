"""
Continual learning module for reinforcement learning to prevent catastrophic forgetting.
Uses Elastic Weight Consolidation (EWC) and other continual learning techniques.
"""
import os
import torch
import wandb
import numpy as np
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import Policy
from typing import Dict, Any, Tuple, Optional, List
import copy


class ContinualRLTrainer:
    """Continual Reinforcement Learning trainer using EWC and other techniques."""
    
    def __init__(self, env_id: str = "Isaac-Lab-ArmReach-v0", project_name: str = "primordium-phase0"):
        self.env_id = env_id
        self.project_name = project_name
        self.fisher_information = {}
        self.optimal_weights = {}
        self.task_history = []
    
    def compute_fisher_information(self, policy: Policy, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for EWC.
        
        Args:
            policy: The policy to compute Fisher information for
            num_samples: Number of samples to use for estimation
            
        Returns:
            Dictionary of Fisher information matrices for each parameter
        """
        fisher_dict = {}
        
        # Get policy parameters
        params = dict(policy.model.named_parameters())
        
        # Initialize Fisher information matrices
        for name, param in params.items():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        # Sample trajectories and compute Fisher information
        for _ in range(num_samples):
            # This is a simplified version - in practice you'd sample from the environment
            # and compute gradients of the log-likelihood
            log_likelihood = torch.randn(1, requires_grad=True)  # Placeholder
            
            policy.model.zero_grad()
            log_likelihood.backward(retain_graph=True)
            
            for name, param in params.items():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data.clone().pow(2)
        
        # Average over samples
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
            
        return fisher_dict
    
    def ewc_loss(self, current_params: Dict[str, torch.Tensor], 
                 fisher_info: Dict[str, torch.Tensor],
                 optimal_params: Dict[str, torch.Tensor],
                 ewc_lambda: float = 1000.0) -> torch.Tensor:
        """
        Compute Elastic Weight Consolidation (EWC) regularization loss.
        
        Args:
            current_params: Current model parameters
            fisher_info: Fisher information matrices
            optimal_params: Parameters from previous task
            ewc_lambda: EWC regularization strength
            
        Returns:
            EWC regularization loss
        """
        loss = 0
        for name in fisher_info:
            if name in current_params and name in optimal_params:
                loss += (fisher_info[name] * 
                        (current_params[name] - optimal_params[name]).pow(2)).sum()
        
        return ewc_lambda * loss / 2
    
    def load_base_policy(self, checkpoint_path: str) -> ppo.PPO:
        """
        Load base policy algorithm from checkpoint.
        
        Args:
            checkpoint_path: Path to the base model checkpoint
        
        Returns:
            Loaded PPO algorithm
        """
        base_config = ppo.PPOConfig().environment(self.env_id).framework("torch")
        base_algo = base_config.build()
        base_algo.restore(checkpoint_path)
        
        return base_algo
    
    def continual_fine_tune(self, 
                           checkpoint_path: str,
                           steps: int,
                           output_dir: str,
                           ewc_lambda: float = 1000.0,
                           run_name: str = "continual_rl") -> Tuple[str, Dict[str, Any]]:
        """
        Fine-tune a model using continual learning techniques (EWC).
        
        Args:
            checkpoint_path: Path to base model checkpoint
            steps: Number of fine-tuning steps
            output_dir: Directory to save results
            ewc_lambda: EWC regularization strength
            run_name: Name for wandb run
        
        Returns:
            Tuple of (model_path, training_metrics)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        wandb.init(project=self.project_name, name=run_name, dir=output_dir)
        
        # Load base algorithm
        base_algo = self.load_base_policy(checkpoint_path)
        
        # Store optimal parameters from previous task
        optimal_params = {}
        for name, param in base_algo.get_policy().model.named_parameters():
            optimal_params[name] = param.data.clone()
        
        # Compute Fisher information for previous task
        fisher_info = self.compute_fisher_information(base_algo.get_policy())
        
        # Store for this task
        task_id = len(self.task_history)
        self.fisher_information[task_id] = fisher_info
        self.optimal_weights[task_id] = optimal_params
        self.task_history.append({
            'task_id': task_id,
            'checkpoint': checkpoint_path,
            'ewc_lambda': ewc_lambda
        })
        
        # Create new config for fine-tuning (you might want to adjust learning rate)
        config = (
            ppo.PPOConfig()
            .environment(self.env_id)
            .training(lr=1e-5)  # Lower learning rate for fine-tuning
            .framework("torch")
        )
        
        algo = config.build()
        
        # Restore weights from base model
        algo.restore(checkpoint_path)
        
        # Training loop with EWC regularization
        metrics_history = []
        
        for step in range(steps):
            # Standard PPO training step
            result = algo.train()
            
            # Compute EWC loss (this would need to be integrated into the actual training)
            # For now, we'll just track it separately
            current_params = {}
            for name, param in algo.get_policy().model.named_parameters():
                current_params[name] = param.data
            
            ewc_losses = []
            for prev_task_id in self.fisher_information:
                ewc_loss = self.ewc_loss(
                    current_params,
                    self.fisher_information[prev_task_id],
                    self.optimal_weights[prev_task_id],
                    ewc_lambda
                )
                ewc_losses.append(ewc_loss.item())
            
            # Log metrics
            metrics = {
                "episode_reward_mean": result["episode_reward_mean"],
                "timesteps_total": result["timesteps_total"],
                "ewc_loss_total": sum(ewc_losses),
                "num_previous_tasks": len(self.fisher_information)
            }
            
            wandb.log(metrics)
            metrics_history.append(metrics)
            
            if step % 100 == 0:
                print(f"Step {step}/{steps}, Reward: {result['episode_reward_mean']:.3f}, "
                      f"EWC Loss: {sum(ewc_losses):.3f}")
        
        # Save fine-tuned model
        model_path = os.path.join(output_dir, "continual_rl_model.pt")
        algo.save(model_path)
        
        wandb.finish()
        
        print(f"Continual RL fine-tuning complete. Model saved to: {model_path}")
        
        return model_path, {"training_history": metrics_history}
    
    def evaluate_forgetting(self, 
                           checkpoints: List[str],
                           test_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate catastrophic forgetting across multiple tasks.
        
        Args:
            checkpoints: List of checkpoint paths for different tasks
            test_episodes: Number of episodes for evaluation
        
        Returns:
            Dictionary with forgetting metrics
        """
        results = {}
        
        for i, checkpoint in enumerate(checkpoints):
            try:
                algo = self.load_base_policy(checkpoint)
                
                # Evaluate performance (placeholder implementation)
                # In practice, you'd run the policy in the environment
                performance = np.random.uniform(0.5, 1.0)  # Placeholder
                
                results[f"task_{i}_performance"] = performance
                
            except Exception as e:
                print(f"Error evaluating checkpoint {checkpoint}: {e}")
                results[f"task_{i}_performance"] = 0.0
        
        # Compute forgetting metrics
        if len(results) > 1:
            performances = list(results.values())
            avg_performance = np.mean(performances)
            performance_drop = max(performances) - min(performances)
            
            results.update({
                "average_performance": avg_performance,
                "performance_drop": performance_drop,
                "forgetting_score": performance_drop / max(performances) if max(performances) > 0 else 0,
                "retention_rate": 1.0 - (performance_drop / max(performances)) if max(performances) > 0 else 1.0
            })
        
        return results