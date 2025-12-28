# Pyramid Goal-Conditioned RL (Self-Supervised)

A headless-first Pyramid Construction simulation designed for goal-conditioned, self-supervised RL
(HER-style relabeling; optional contrastive reachability later), with pluggable shallow MLP vs deep ResMLP.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# Train (headless)
python main_train.py --exp experiments/depth_032.yaml

# Evaluate on a fixed suite
python main_eval.py --exp experiments/depth_032.yaml --ckpt checkpoints/depth_032/latest.pt

# Watch a policy
python main_play.py --exp experiments/depth_032.yaml --ckpt checkpoints/depth_032/latest.pt
```

## Core idea
- Observation is vector state `s`
- Goal is vector `g`
- Policy/Q take `[s, g]`
- Self-supervision via hindsight relabeling of goals from achieved pyramid configurations
