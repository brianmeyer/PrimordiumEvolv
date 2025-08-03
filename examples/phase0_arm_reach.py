
"""Baseline PPO training script for Isaac Lab arm reach task."""
import os, argparse, wandb
from ray.rllib.algorithms import ppo

ENV_ID = "Isaac-Lab-ArmReach-v0"

def train(steps: int, logdir: str):
    wandb.init(project="autogptpp_phase0", name="baseline", dir=logdir)
    cfg = (
        ppo.PPOConfig()
        .environment(env=ENV_ID)
        .training(model={"fcnet_hiddens": [256, 256]})
        .rollouts(num_rollout_workers=0)
        .framework("torch")
    )
    algo = cfg.build()
    while algo._counters.get("timesteps_total", 0) < steps:
        res = algo.train()
        wandb.log({
            "episode_reward_mean": res["episode_reward_mean"],
            "timesteps_total": res["timesteps_total"],
        })
    algo.save(os.path.join(logdir, "baseline.pt"))
    wandb.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--logdir", type=str, default="runs/baseline")
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    train(args.steps, args.logdir)
