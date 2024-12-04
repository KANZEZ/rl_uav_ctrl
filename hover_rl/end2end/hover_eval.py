import gymnasium as gym
from rotorpy.vehicles.crazyflie_params import quad_params
from reward import CurriculumReward
from env_helper import ActionContainer
from end2end_config import *

################# evaluation #################
################# returns average reward and the reward at the last step

def eval_policy(policy):

    reward_obj = CurriculumReward()
    reward_function = lambda obs, act, finish: reward_obj.hover_reward(obs, act, finish)
    ac_obj = ActionContainer()

    eval_env = gym.make(ENV_NAME,
                        control_mode = CONTROL_MODE,
                        reward_fn = reward_function,
                        quad_params = quad_params,
                        max_time = MAX_SIM_TIME,
                        world = None,
                        sim_rate = SIM_RATE,
                        render_mode=RENDER_MODE)

    avg_reward = 0.0
    last_step_reward = 0.0
    avg_episode_len = 0

    for _ in range(EVAL_EPISODE):
        obs, done = eval_env.reset(seed=SEED+100,
                                   initial_state='random',
                                   options={'pos_bound': POS_BOUND,
                                            'vel_bound': VEL_BOUND})[0], False
        ac_obj.clear()
        eval_episode_len = 0

        while not done:
            eval_episode_len += 1
            cur_ah = ac_obj.get()
            action = policy.select_action(obs, cur_ah)
            ac_obj.add(action)
            obs, reward, done, _, _ = eval_env.step(action)

            if done:
                last_step_reward += reward
                avg_episode_len += eval_episode_len

            avg_reward += reward

    avg_reward /= EVAL_EPISODE
    last_step_reward /= EVAL_EPISODE
    avg_episode_len /= EVAL_EPISODE

    print("---------------------------------------")
    print(f"Evaluation over {EVAL_EPISODE} episodes: average reward: {avg_reward:.3f}, final reward: {last_step_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, last_step_reward, avg_episode_len



