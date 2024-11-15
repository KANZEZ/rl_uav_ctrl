import gymnasium as gym
import numpy as np
import os
from datetime import datetime
import torch
from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.
# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# First we'll set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

########## training setting ##########
from ddpg import DDPG, device
from ddpg_eval import eval_policy
from reward import CurriculumReward
from env_helper import ActionContainer
reward_obj = CurriculumReward()
reward_function = lambda obs, act, finish: reward_obj.reward(obs, act, finish)

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
pos_dim, rot_dim, vel_dim, w_dim, action_dim = 3, 4, 3, 3, 4
obs_dim = pos_dim + vel_dim + rot_dim + w_dim # 13
pos_bound, vel_bound = 5.5, 2.0

model = DDPG(obs_dim, action_dim)
ac_obj = ActionContainer(action_dim)
observation, _ = env.reset(seed=seed, initial_state='random', options={'pos_bound': pos_bound,
                                                                       'vel_bound': vel_bound})

max_timesteps = 1e9
episode_timesteps = 0
episode_num = 0
episode_reward = 0
batch_size = 256
eval_freq = 10000

###### warmup
actor_start_timesteps = 30000
critic_start_timesteps = 15000
actor_training_interval = 20
critic_training_interval = 2

#### exploration noise ####
expl_noise_decay_start = 250000
expl_noise_decay_interval = 100000

#### guidance ####
guidance_prob = 0.1

#### cirriculum learning ####
reward_update_interval = 100000

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
    if t < critic_start_timesteps: # warmup to fill the buffer
        cur_action = env.action_space.sample()
    elif critic_start_timesteps <= t < actor_start_timesteps: # time to train critic
        cur_ah = ac_obj.get()
        cur_action = (
                model.select_action(np.array(observation), cur_ah)
                + np.random.normal(0, max_action * model.noise_std, size=action_dim)
        ).clip(-max_action, max_action)
        if (t+1) % critic_training_interval == 0:
            # Sample replay buffer and train
            model.train(batch_size, 0)
    else: # time to train actor and critic
        cur_ah = ac_obj.get()
        cur_action = (
                model.select_action(np.array(observation), cur_ah)
                + np.random.normal(0, max_action * model.noise_std, size=action_dim)
        ).clip(-max_action, max_action)
        # Sample replay buffer and train
        if (t+1) % actor_training_interval == 0:
            model.train(batch_size, 1)
        elif (t+1) % critic_training_interval == 0:
            model.train(batch_size, 0)

    # keep playing the game
    next_observation, reward, done, _, _ = env.step(cur_action)
    #### add to replay buffer
    model.replay_buffer.add(observation, cur_action, ac_obj.get(), next_observation, reward, done)
    ac_obj.add(cur_action)
    observation = next_observation
    episode_reward += reward

    if done:
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        reset_way = np.random.choice(['random', 'guidance'],
                                     p=[1-guidance_prob, guidance_prob])
        observation, _ = env.reset(seed=seed, initial_state=reset_way, options={'pos_bound': pos_bound,
                                                                               'vel_bound': vel_bound})
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        ac_obj.clear() ### clear action history after one game over

    # exploration noise decay, update noise_std
    if t > expl_noise_decay_start and (t+1) % expl_noise_decay_interval == 0:
        model.update_noise()

    # cirriculum learning
    if (t+1) % reward_update_interval == 0:
        reward_obj.curriculum_update()
        model.replay_buffer.reward_recalculation(reward_obj)

    if (t + 1) % eval_freq == 0:
        #model.eval_mode()
        #eval_policy(model, "Quadrotor-v0", seed, pos_bound, vel_bound)
        if save_model:
            model.save(save_model_path + "/")
        model.train_mode()
