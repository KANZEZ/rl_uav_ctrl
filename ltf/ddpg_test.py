import ddpg_eval
import numpy as np
import torch
import gymnasium as gym
from ddpg import DDPG

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DDPG(13, 4)
    path = "/home/hsh/Code/rl_uav_control/rotorpy/learning/policies/DDPG/17-01-55/17-01-55"
    # Load the policy
    model.load(path)
    model.eval()

    # Evaluate the policy
    ddpg_eval.eval_policy(model, "Quadrotor-v0", seed, eval_episodes=1, render='3D')