import gymnasium as gym
import numpy as np
import os
from datetime import datetime
import torch
from rotorpy.vehicles.crazyflie_params import quad_params
from ddpy_trk import DDPG
from reward import CurriculumReward
from env_helper import ActionContainer, TargetSelector
from rotorpy.learning.ddpg_pathtrack_env import QuadrotorTrackingEnv

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

########## TRAINING SETTING ##########
reward_obj = CurriculumReward()
traj_selector = TargetSelector()
trk_reward_func = lambda obs, act, done, target: reward_obj.tracking_reward(obs, act, done, target)

######### env setting ##########
env = gym.make("Quadrotor-v1",
               control_mode ='cmd_motor_speeds',
               reward_fn = trk_reward_func,
               quad_params = quad_params,
               max_time = 10,
               world = None,
               sim_rate = 100,
               render_mode='None',
               target_selector=traj_selector)

SEED = 47
np.random.seed(SEED)
torch.manual_seed(SEED)
env.action_space.seed(SEED)
MAX_ACTION = float(env.action_space.high[0])

######### model setting ###########
POS_DIM = 3
QUAT_DIM = 4
VEL_DIM = 3
W_DIM = 3
ACTION_DIM = 4
OBS_DIM = POS_DIM + VEL_DIM + QUAT_DIM + W_DIM # 13
POS_BOUND = 2.0
VEL_BOUND = 2.0

model = DDPG(OBS_DIM, ACTION_DIM)
ac_obj = ActionContainer(ACTION_DIM)
observation, _ = env.reset(seed=SEED, initial_state='guidance', options={'pos_bound': POS_BOUND,
                                                                       'vel_bound': VEL_BOUND})

############## TRAINING CONST ###########
MAX_ITER = int(1e9)
ITER = 0
EPISODE_NUM = 0
EPISODE_REWARD = 0
BATCH_SIZE = 256
EVAL_FREQ = 10000

###### warmup
ACTOR_START = 30000
CRITIC_START = 15000
ACTOR_TRAIN_INTERVAL = 20
CRITIC_TRAIN_INTERVAL = 2

#### exploration noise ####
EXPL_NOISE_DECAY_START = 250000
EXPL_NOISE_DECAY_INTERVAL = 100000

#### guidance : start from the start of traj ####
GUIDANCE_PROB = 1.0
GUIDANCE_PROB_LIM = 0.99
GUIDANCE_PROB_UPDATE_INTERVAL = 100000
GUIDANCE_PROB_UPDATE_START = 50000

#### cirriculum learning ####
REWARD_UPDATE_INTERVAL = 100000
TRAJ_PROB_UPDATE_INTERVAL = 150000


############## training loop ################
for t in range(MAX_ITER):
    ITER += 1

    cur_ah = ac_obj.get()
    fu_traj = env.get_flat_future_traj()

    # critic and actor warmup,
    if t < CRITIC_START:

        cur_action = env.action_space.sample()

    elif CRITIC_START <= t < ACTOR_START: # time to train critic

        cur_action = (
                model.select_action(np.array(observation), cur_ah, fu_traj)
                + np.random.normal(0, MAX_ACTION * model.noise_std, size=ACTION_DIM)
        ).clip(-MAX_ACTION, MAX_ACTION)
        if (t+1) % CRITIC_TRAIN_INTERVAL == 0:
            # Sample replay buffer and train
            model.train(BATCH_SIZE, 0)

    else: # train actor and critic

        cur_action = (
                model.select_action(np.array(observation), cur_ah, fu_traj)
                + np.random.normal(0, MAX_ACTION * model.noise_std, size=ACTION_DIM)
        ).clip(-MAX_ACTION, MAX_ACTION)
        # Sample replay buffer and train
        if (t+1) % ACTOR_TRAIN_INTERVAL == 0:
            model.train(BATCH_SIZE, 1)
        elif (t+1) % CRITIC_TRAIN_INTERVAL == 0:
            model.train(BATCH_SIZE, 0)

    # keep playing the game
    next_observation, reward, done, _, _ = env.step(cur_action)

    #### add to replay buffer
    model.replay_buffer.add(observation, cur_action, cur_ah, fu_traj,
                            next_observation, reward, done)
    ac_obj.add(cur_action)
    observation = next_observation
    EPISODE_REWARD += reward

    if done:
        print(f"Total T: {t+1} Episode Num: {EPISODE_NUM+1} Episode T: {ITER} Reward: {EPISODE_REWARD:.3f}")
        reset_way = np.random.choice(['random', 'guidance'],
                                     p=[1-GUIDANCE_PROB, GUIDANCE_PROB])
        observation, _ = env.reset(seed=SEED, initial_state=reset_way, options={'pos_bound': POS_BOUND,
                                                                                'vel_bound': VEL_BOUND})
        done = False
        EPISODE_REWARD = 0
        ITER = 0
        EPISODE_NUM += 1
        ac_obj.clear() ### clear action history after one game over

    # exploration noise decay, update noise_std
    if t > EXPL_NOISE_DECAY_START and (t+1) % EXPL_NOISE_DECAY_INTERVAL == 0:
        model.update_noise()

    # cirriculum learning for reward
    if (t+1) % REWARD_UPDATE_INTERVAL == 0:
        reward_obj.curriculum_update()
        model.replay_buffer.reward_recalculation(reward_obj)

    # cirriculum learning for guidance
    if (t+1) % GUIDANCE_PROB_UPDATE_INTERVAL == 0 and t > GUIDANCE_PROB_UPDATE_START:
        GUIDANCE_PROB = max(GUIDANCE_PROB - 0.1, GUIDANCE_PROB_LIM)


    # cirriculum learning for traj gen
    if (t+1) % TRAJ_PROB_UPDATE_INTERVAL == 0:
        traj_selector.update_traj_prob()


    if (t + 1) % EVAL_FREQ == 0:
        # model.eval_mode()
        # eval_policy(model, "Quadrotor-v0", SEED, POS_BOUND, VEL_BOUND)
        if save_model:
            model.save(save_model_path + "/")
        model.train_mode()
