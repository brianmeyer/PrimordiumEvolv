"""
Common utility functions for PrimordiumEvolv.
"""
import os
import wandb
from typing import Optional, Dict, Any


def setup_wandb(project_name: str, 
                run_name: str,
                output_dir: str,
                config: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize Weights & Biases logging.
    
    Args:
        project_name: Name of the wandb project
        run_name: Name for this specific run
        output_dir: Directory for local logs
        config: Configuration dictionary to log
    """
    wandb.init(
        project=project_name,
        name=run_name,
        dir=output_dir,
        config=config or {}
    )


def create_output_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create output directory for experiment results.
    
    Args:
        base_dir: Base directory (e.g., "runs")
        experiment_name: Name of the experiment
    
    Returns:
        Path to created directory
    """
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_env_config(env_name: str) -> Dict[str, Any]:
    """
    Get environment configuration.
    
    Args:
        env_name: Name of the environment
    
    Returns:
        Environment configuration dictionary
    """
    configs = {
        "Isaac-Lab-ArmReach-v0": {
            "action_space": "continuous",
            "observation_space": "vector",
            "max_episode_steps": 1000,
        }
    }
    
    return configs.get(env_name, {})


def log_system_info() -> Dict[str, Any]:
    """
    Log system information for reproducibility.
    
    Returns:
        System information dictionary
    """
    import platform
    import torch
    
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
    
    return info