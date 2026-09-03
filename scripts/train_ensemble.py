"""Train ensemble of world models."""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "E:/GitHub/RL_policy")

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from scripts.world_model_train import WorldModel, WorldModelTrainer

PROJECT_ROOT = "E:/GitHub/RL_policy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ENSEMBLE_SIZE = 3
EPOCHS = 30
N_EPISODES = 50

envs = {
    "Ant-v5": {"obs_dim": 105, "action_dim": 8},
    "HalfCheetah-v5": {"obs_dim": 17, "action_dim": 6},
}

for env_name, dims in envs.items():
    print(f"\n{'='*50}")
    print(f"Training ensemble: {env_name}")
    print(f"{'='*50}")

    ensemble = []
    for i in range(ENSEMBLE_SIZE):
        print(f"\n--- Member {i+1}/{ENSEMBLE_SIZE} ---")
        torch.manual_seed(i * 42 + 7)

        wm = WorldModel(dims["obs_dim"], dims["action_dim"], latent_dim=128).to(device)
        trainer = WorldModelTrainer(wm, lr=1e-3)

        obs, acts, rews = trainer.collect_data(env_name, n_episodes=N_EPISODES)
        print(f"  Data: {obs.shape[0]} transitions")

        losses = trainer.train(obs, acts, rews, epochs=EPOCHS)
        print(f"  Final loss: {losses[-1]:.4f}")

        ensemble.append(wm)

    # Save ensemble
    save_path = f"{PROJECT_ROOT}/data/ensemble_{env_name}.pt"
    torch.save(
        [{"model_state_dict": wm.state_dict()} for wm in ensemble],
        save_path,
    )
    print(f"\nSaved ensemble to {save_path}")

print("\nDone!")
