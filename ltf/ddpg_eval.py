import numpy as np
import torch
import gymnasium as gym
from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.
from ddpg import DDPG
# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from reward import CurriculumReward
from env_helper import ActionContainer
reward_obj = CurriculumReward()
reward_function = lambda obs, act, finish: reward_obj.reward(obs, act, finish)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, pos_bound, vel_bound, eval_episodes=30, render='None'):
    eval_env = gym.make(env_name,
                        control_mode ='cmd_motor_speeds',
                        reward_fn = reward_function,
                        quad_params = quad_params,
                        max_time = 5,
                        world = None,
                        sim_rate = 100,
                        render_mode=render)
    ac_obj = ActionContainer(4)
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(seed=seed+99, initial_state='random', options={'pos_bound': pos_bound,
                                                                                  'vel_bound': vel_bound})[0], False
        ac_obj.clear()
        while not done:
            cur_ah = ac_obj.get()
            action = policy.select_action(obs, cur_ah)
            ac_obj.add(action)
            obs, reward, done, _, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward



