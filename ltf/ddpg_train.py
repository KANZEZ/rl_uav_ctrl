import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

import torch

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward



# First we'll set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

########## training setting ##########
from ddpg import DDPG, ReplayBuffer
from ddpg_eval import eval_policy
reward_function = lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5})

######### env setting ##########
seed = 47
env = gym.make("Quadrotor-v0",
               control_mode ='cmd_motor_speeds',
               reward_fn = reward_function,
               quad_params = quad_params,
               max_time = 5,
               world = None,
               sim_rate = 100,
               render_mode='None')
np.random.seed(seed)
torch.manual_seed(seed)
env.action_space.seed(seed)
max_action = float(env.action_space.high[0])

######### model setting ###########
obs_dim, action_dim = 13, 4
model = DDPG(13, 4)
replay_buffer = ReplayBuffer(13, 4)
evaluations = [eval_policy(model, "Quadrotor-v0", seed)]

observation, _ = env.reset(seed=seed, initial_state='random', options={'pos_bound': 2, 'vel_bound': 1})
done = False
max_timesteps = 1e9
episode_timesteps = 0
episode_num = 0
episode_reward = 0
batch_size = 256
eval_freq = 10000

# warmup
actor_start_timesteps = 30000
critic_start_timesteps = 15000
actor_training_interval = 20
critic_training_interval = 10

#### exploration noise ####
expl_noise_decay_start = 250000
expl_noise_decay_interval = 100000

#### model save setting ####
save_model = True
start_time = datetime.now()
save_model_path = f"{models_dir}/DDPG/{start_time.strftime('%H-%M-%S')}"

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

############## training loop ################
for t in range(int(max_timesteps)):
    episode_timesteps += 1

    # critic and actor warmup, we think critic_start_timesteps < actor_start_timesteps
    if t < critic_start_timesteps:
        action = env.action_space.sample()
    elif critic_start_timesteps <= t < actor_start_timesteps:
        action = env.action_space.sample()
        if (t+1) % critic_training_interval == 0:
            model.train_critic(replay_buffer, batch_size)
            model.update_target_critic()
    else:
        action = (
                model.select_action(np.array(observation))
                + np.random.normal(0, model.noise_std, size=action_dim)
        ).clip(-max_action, max_action)
        if (t+1) % actor_training_interval == 0:
            model.train_actor(replay_buffer, batch_size)
            model.update_target_actor()

    # exploration noise decay, update noise_std
    if t > expl_noise_decay_start and (t+1) % expl_noise_decay_interval == 0:
        model.update_noise()

    next_observation, reward, done, _, _ = env.step(action)
    replay_buffer.add(observation, action, next_observation, reward, done)
    observation = next_observation
    episode_reward += reward

    if done:
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        observation, _ = env.reset(seed=seed, initial_state='random', options={'pos_bound': 2, 'vel_bound': 1})
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if (t + 1) % eval_freq == 0:
        evaluations.append(eval_policy(model, "Quadrotor-v0", seed))
        if save_model:
            model.save(save_model_path + "/")
