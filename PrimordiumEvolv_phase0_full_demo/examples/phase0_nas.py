
"""Reward‑only neural‑architecture search (tiny)."""
import os, argparse, random, shutil, wandb
from ray.rllib.algorithms import ppo

ENV_ID = "Isaac-Lab-ArmReach-v0"
SEARCH_TO_CFG = {
    "mlp":  {"model": {"fcnet_hiddens": [256, 256]}},
    "lstm": {"model": {"use_lstm": True, "lstm_cell_size": 256}},
}

def run_trial(arch: str, steps: int, outdir: str):
    wandb.init(project="autogptpp_phase0", name=f"nas-{arch}", dir=outdir, reinit=True)
    cfg = (
        ppo.PPOConfig()
        .environment(ENV_ID)
        .training(**SEARCH_TO_CFG[arch])
        .rollouts(num_rollout_workers=0)
        .framework("torch")
    )
    algo = cfg.build()
    best = -1e9
    while algo._counters.get("timesteps_total", 0) < steps:
        res = algo.train()
        rew = res["episode_reward_mean"]
        best = max(best, rew)
        wandb.log({"episode_reward_mean": rew, "timesteps_total": res["timesteps_total"]})
    ckpt = algo.save(os.path.join(outdir, "ckpt"))
    wandb.finish()
    return best, ckpt

def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    trials = []
    for i in range(a.budget):
        arch = random.choice(a.search_space.split(","))
        tdir = os.path.join(a.outdir, f"trial_{i}_{arch}")
        rew, ck = run_trial(arch, a.trial_steps, tdir)
        trials.append((rew, ck))
    best_ck = max(trials, key=lambda x: x[0])[1]
    shutil.copy(best_ck, os.path.join(a.outdir, "best.pt"))
    print("Best checkpoint at runs/nas/best.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--search_space", type=str, default="mlp,lstm")
    p.add_argument("--trial_steps", type=int, default=10000)
    p.add_argument("--outdir", type=str, default="runs/nas")
    main(p.parse_args())
