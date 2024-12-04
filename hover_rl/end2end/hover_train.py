import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import os
from datetime import datetime
import torch
from rotorpy.vehicles.crazyflie_params import quad_params
from hover_rl.end2end.td3 import TD3
from hover_rl.end2end.hover_eval import eval_policy
from reward import CurriculumReward
from env_helper import ActionContainer
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from end2end_config import *

# set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "rotorpy", "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
#### model save setting ####
save_model = True
start_time = datetime.now()
save_model_path = f"{models_dir}/DDPG/{start_time.strftime('%H-%M-%S')}"
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)





######### env setting ##########
reward_obj = CurriculumReward()
reward_function = lambda obs, act, finish: reward_obj.hover_reward(obs, act, finish)

env = gym.make(ENV_NAME,
               control_mode = CONTROL_MODE,
               reward_fn = reward_function,
               quad_params = quad_params,
               max_time = MAX_SIM_TIME,
               world = None,
               sim_rate = SIM_RATE,
               render_mode= RENDER_MODE)

np.random.seed(SEED)
torch.manual_seed(SEED)
env.action_space.seed(SEED)
MAX_ACTION = float(env.action_space.high[0])


######### model setting ###########
model = TD3()
ac_obj = ActionContainer()
observation, _ = env.reset(seed=SEED, initial_state='random', options={'pos_bound': POS_BOUND,
                                                                       'vel_bound': VEL_BOUND})

########    record   ##########
eval_avg_reward, eval_last_step_reward = [], []
train_reward, train_last_step_reward = [], []
train_episode_len, eval_avg_episode_len = [], []
ITER = 0
EVAL_ITER = 0
EPISODE_NUM = 0
EPISODE_REWARD = 0



############## training loop ################
for t in range(MAX_ITER):
    ITER += 1

    # critic and actor warmup
    if t < CRITIC_START:

        cur_action = env.action_space.sample()

    elif CRITIC_START <= t < ACTOR_START: # time to train critic

        cur_ah = ac_obj.get()
        cur_action = (
                model.select_action(np.array(observation), cur_ah)
                + np.random.normal(0, MAX_ACTION * model.noise_std, size=ACTION_DIM)
        ).clip(-MAX_ACTION, MAX_ACTION)

        if (t+1) % CRITIC_TRAIN_INTERVAL == 0:
            # Sample replay buffer and train
            model.train(0)

    else: # time to train actor and critic

        cur_ah = ac_obj.get()
        cur_action = (
                model.select_action(np.array(observation), cur_ah)
                + np.random.normal(0, MAX_ACTION * model.noise_std, size=ACTION_DIM)
        ).clip(-MAX_ACTION, MAX_ACTION)
        # Sample replay buffer and train

        if (t+1) % ACTOR_TRAIN_INTERVAL == 0:
            model.train(1)
        elif (t+1) % CRITIC_TRAIN_INTERVAL == 0:
            model.train(0)

    # keep playing the game
    next_observation, reward, done, _, _ = env.step(cur_action)
    #### add to replay buffer
    model.replay_buffer.add(observation, cur_action, ac_obj.get(), next_observation, reward, done)
    ac_obj.add(cur_action)
    observation = next_observation
    EPISODE_REWARD += reward

    if done:

        print(f"Total T: {t+1} Episode Num: {EPISODE_NUM+1} Episode T: {ITER} Reward: {EPISODE_REWARD:.3f} LAST STEP REWARD: {reward: .3f}" )

        ##### record stuff ########
        train_reward.append(EPISODE_REWARD)
        train_last_step_reward.append(reward)
        train_episode_len.append(ITER)

        ####### reset stuff #######
        reset_way = np.random.choice(['random', 'guidance'],
                                     p=[1-GUIDANCE_PROB, GUIDANCE_PROB])
        observation, _ = env.reset(seed=SEED,
                                   initial_state=reset_way,
                                   options={'pos_bound': POS_BOUND,
                                            'vel_bound': VEL_BOUND})
        done = False
        EPISODE_REWARD = 0
        ITER = 0
        EPISODE_NUM += 1
        ac_obj.clear()

    # exploration noise decay, update noise_std
    if t > EXPL_NOISE_DECAY_START and (t+1) % EXPL_NOISE_DECAY_INTERVAL == 0:
        model.update_noise()

    # cirriculum learning
    if (t+1) % REWARD_UPDATE_INTERVAL == 0:
        reward_obj.curriculum_update()
        model.replay_buffer.reward_recalculation(reward_obj)


    ############### evaluation #################
    if (t + 1) % EVAL_FREQ == 0:
        model.eval_mode()

        avg_reward, last_step_reward, avg_episode_len = eval_policy(model)

        eval_avg_reward.append(avg_reward)
        eval_last_step_reward.append(last_step_reward)
        eval_avg_episode_len.append(avg_episode_len)

        EVAL_ITER += 1

        if save_model:
            model.save(save_model_path + "/")

        model.train_mode()



############  plot ######################
plt.figure(1)
plt.plot(np.arange(EPISODE_NUM), train_reward)
plt.xlabel('number of episode')
plt.ylabel('training reward')

plt.figure(2)
plt.plot(np.arange(EPISODE_NUM), train_last_step_reward)
plt.xlabel('number of episode')
plt.ylabel('training last step reward')

plt.figure(3)
plt.plot(np.arange(EVAL_ITER), eval_avg_reward)
plt.xlabel('number of evaluation')
plt.ylabel('evaluation average reward')

plt.figure(4)
plt.plot(np.arange(EVAL_ITER), eval_last_step_reward)
plt.xlabel('number of evaluation')
plt.ylabel('evaluation last step reward')

plt.figure(5)
plt.plot(np.arange(EVAL_ITER), eval_avg_episode_len)
plt.xlabel('number of evaluation')
plt.ylabel('evaluation average episode length')

plt.figure(6)
plt.plot(np.arange(EPISODE_NUM), train_episode_len)
plt.xlabel('number of evaluation')
plt.ylabel('evaluation training episode length')

plt.show()