
"""MIT SEAL weight fine‑tuning stub."""
import os, argparse, wandb, torch
from seal.trainer import ContinualTrainer
from seal.algorithms import SEALPPO
from ray.rllib.algorithms import ppo

ENV_ID = "Isaac-Lab-ArmReach-v0"

def main(a):
    wandb.init(project="autogptpp_phase0", name="seal_ft", dir=a.outdir)
    base_cfg = ppo.PPOConfig().environment(ENV_ID).framework("torch")
    base_algo = base_cfg.build()
    base_algo.restore(a.checkpoint)
    state = base_algo.get_policy().get_state()

    trainer = ContinualTrainer(env_id=ENV_ID,
                               base_policy_state=state,
                               algo_cls=SEALPPO,
                               total_steps=a.steps)
    fin_state, metrics = trainer.train()
    torch.save(fin_state, os.path.join(a.outdir, "seal_tuned.pt"))
    wandb.log(metrics); wandb.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--outdir", type=str, default="runs/seal")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    main(args)
