import ddpg_eval
import numpy as np
import torch
import gymnasium as gym
from ddpg import DDPG
from reward import CurriculumReward
from env_helper import ActionContainer
from rotorpy.vehicles.crazyflie_params import quad_params

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 1
    pos_bound, vel_bound = 1.0, 0.2
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DDPG(13, 4)
    path = "/home/hsh/Code/rl_uav_control/rotorpy/learning/policies/DDPG/18-54-14/"
    # Load the policy
    model.load(path)
    model.eval_mode()
    reward_obj = CurriculumReward()
    reward_function = lambda obs, act, finish: reward_obj.reward(obs, act, finish)

    # Evaluate the policy
    eval_env = gym.make('Quadrotor-v0',
                        control_mode ='cmd_motor_speeds',
                        reward_fn = reward_function,
                        quad_params = quad_params,
                        max_time = 5,
                        world = None,
                        sim_rate = 100,
                        render_mode='3D')
    ac_obj = ActionContainer(4)
    avg_reward = 0.
    eval_episodes = 1
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(seed=seed+99, initial_state='random', options={'pos_bound': pos_bound,
                                                                                  'vel_bound': vel_bound}), False
        ac_obj.clear()
        while not done:
            cur_ah = ac_obj.get()
            action = model.select_action(obs, cur_ah)
            ac_obj.add(action)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")