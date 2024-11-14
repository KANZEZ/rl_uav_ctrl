import numpy as np
import torch
import gymnasium as gym
from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.
from ddpg import DDPG
# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=20, render='None'):
    eval_env = gym.make(env_name,
                              control_mode ='cmd_motor_speeds',
                              reward_fn = hover_reward,
                              quad_params = quad_params,
                              max_time = 5,
                              world = None,
                              sim_rate = 100,
                              render_mode=render)
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(seed=seed+99, initial_state='random', options={'pos_bound': 2, 'vel_bound': 1})[0], False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward



