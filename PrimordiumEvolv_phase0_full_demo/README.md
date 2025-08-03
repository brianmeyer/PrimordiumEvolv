
# AutoGPT++ Phase 0 Demo – Reward‑Only NAS

This bundle gives you a **minimal working proof‑of‑concept** for Phase 0 of the
self‑evolving robotics stack:

* baseline PPO policy on a 5‑DoF reach task in **Isaac Lab**
* tiny neural‑architecture search (NAS) loop (MLP vs LSTM)
* optional MIT SEAL fine‑tune

It is designed to run inside **NVIDIA’s official Isaac Sim 4.5 container**
(`nvcr.io/nvidia/isaac-sim:4.5.0`) on a rented GPU (e.g. RunPod RTX 4090).

---

## Quick‑Start (RunPod)

1. **Deploy a Custom Container Pod**

   | Field | Value |
   |-------|-------|
   | Container Image | `nvcr.io/nvidia/isaac-sim:4.5.0` |
   | Ports | `22`, `8888`, `3000` (optional UI) |
   | Env var | `ACCEPT_EULA=Y` |
   | GPU | *RTX 4090* (cheapest fast card) |

2. **Open Web Terminal** and run:

   ```bash
   # download & unzip this bundle (assume it is in your home dir)
   unzip autogptpp_phase0_demo_full.zip && cd autogptpp_phase0_demo_full/examples

   # create clean env & install extras
   conda create -y -n autogptpp python=3.11 && conda activate autogptpp
   pip install ray[rllib]==3.0 wandb
   # (SEAL installs automatically if you run phase0_seal.py)
   ```

3. **Baseline policy (≈2 GPU‑h)**  
   ```bash
   python phase0_arm_reach.py --steps 50000 --logdir runs/baseline
   ```

4. **Reward‑only NAS (overnight, ≈48 GPU‑h)**  
   ```bash
   python phase0_nas.py --budget 8 --search_space mlp,lstm --outdir runs/nas
   ```

5. **Optional SEAL fine‑tune (≈30 min)**  
   ```bash
   python phase0_seal.py --checkpoint runs/nas/best.pt --steps 5000 --outdir runs/seal
   ```

All metrics stream to **Weights & Biases**; the NAS script copies the
top‑reward checkpoint to `runs/nas/best.pt`.

---

## File Overview

| Path | Purpose |
|------|---------|
| `examples/phase0_arm_reach.py` | Baseline PPO training |
| `examples/phase0_nas.py` | Tiny NAS loop (reward only) |
| `examples/phase0_seal.py` | MIT SEAL weight tuner stub |
| `README.md` | These instructions |

---

## Requirements inside the container

* CUDA 12.x (already in Isaac container)
* PyTorch 2.3 (already in Isaac container)
* **Ray RLlib 3.0** (`pip install ray[rllib]==3.0`)
* **Weights & Biases** (`pip install wandb`)
* **MIT SEAL** (`pip install git+https://github.com/Continual-Intelligence/SEAL.git`) – only if you run `phase0_seal.py`

---

## Tips

* **Pause pod when idle** – RunPod stops billing compute but keeps your disk.  
* **VS Code Remote‑SSH** – add your public key in RunPod → Settings → SSH Keys, then connect to `ssh <pod>`.  
* **Change env ID** – if you registered a different Isaac Lab environment, edit `ENV_ID` at the top of each script.

Happy hacking!
