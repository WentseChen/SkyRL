# How to Launch Training (cwz19 machine — 8x H100, Ubuntu 22.04)

This document describes the exact steps to launch the mini-SWE-agent training run on this machine.
All bug fixes referenced here are documented in full in `~/SETUP.md`.

---

## Quick start

```bash
cd ~/SkyRL
bash examples/train/mini_swe_agent/launch.sh
```

That's it. The script handles everything below automatically.

---

## What the launch script does

### 1. Source environment

```bash
source ~/SkyRL/setup_env.sh
```

Sets NCCL variables, activates `~/skyrl_venv`, exports `WANDB_API_KEY`, `HF_TOKEN`, and
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`.

### 2. Start Ray (if not already running)

```bash
ray start --head --disable-usage-stats
```

Skip if `ray status` already shows a live cluster.

### 3. Log in to Docker Hub (to avoid pull rate limits)

```bash
podman login docker.io -u <username> -p <password>
```

Without this, uncached images will fail with `exit status 125` once the 200-pull/6h limit
is hit, crashing the entire training step.

### 4. Rebuild cached dataset files (if image cache has changed)

The training and validation parquet files must only contain instances whose Docker images are
**already cached locally**. If the image cache changes (e.g. new pulls, reboots that clear
storage), regenerate them:

```bash
source ~/SkyRL/setup_env.sh
python3 ~/SkyRL/examples/train/mini_swe_agent/rebuild_cached_datasets.py
```

Current cached sets (as of last rebuild):
- `~/data/swe_gym_subset/train_cached.parquet` — 80 instances (getmoto/moto + python/mypy)
- `~/data/swe_gym_subset/val_small.parquet` — 10 instances (astropy, for fast ~10 min eval)

### 5. Launch training in background

```bash
cd ~/SkyRL
bash examples/train/mini_swe_agent/run_mini_swe_8B.sh 2>&1 | tee ~/training.log &
```

WandB run: https://wandb.ai/cwz19/mini_swe

---

## Key settings in `run_mini_swe_8B.sh`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trainer.eval_before_train` | `true` | Run baseline eval at step 0 |
| `trainer.eval_interval` | `5` | Eval every 5 training steps |
| `trainer.eval_batch_size` | `50` | Larger than val set (10), so all instances run in one batch |
| `trainer.train_batch_size` | `16` | 16 unique instances × 4 samples = 64 trajectories per step |
| `trainer.epochs` | `20` | Total training epochs |
| `generator.n_samples_per_prompt` | `4` | GRPO group size |

---

## Monitoring

```bash
tail -f ~/training.log
```

Look for lines like:
```
Training Batches Processed:   5%|▌  | 5/100 ...
{'final_loss': ..., 'policy_loss': ..., 'grad_norm': ...}
```

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `exit status 125` on podman run | Docker Hub rate limit or uncached image | Run `podman login`; rebuild cached parquets |
| `ValueError: Found no valid responses` | All trajectories in a batch failed (images not cached) | Rebuild `train_cached.parquet` / `val_small.parquet` |
| `TypeError: Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'` | accelerate + transformers 5.x incompatibility | Already patched in `~/.cache/uv/archive-v0/uLREtsQ6sQISVh0YGqFKn/accelerate/big_modeling.py` via hardlink |
| Ray actors die on startup | `python3.12-dev` missing | `sudo apt-get install -y python3.12-dev` |
| `grad_norm: 0.0` throughout | No positive rewards yet — model hasn't solved any tasks | Expected early in training; will improve |
