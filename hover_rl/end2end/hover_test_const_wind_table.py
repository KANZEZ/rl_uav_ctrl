from matplotlib import pyplot as plt

import numpy as np
import gymnasium as gym
from hover_rl.end2end.td3 import TD3
from reward import CurriculumReward
from env_helper import ActionContainer
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.wind.default_winds import ConstantWind, SinusoidWind
from stable_baselines3 import PPO
from rotorpy.learning.quadrotor_reward_functions import hover_reward_ppo
from end2end_config import *

baseline_controller = SE3Control(quad_params)

def random_sign():
    return np.random.choice([-1, 1])

"""
small: 0-1
medium: 1-2
big: 2-3
"""
def small_random_wind():
    lb = 2
    ub = 3
    wx = random_sign() * np.random.uniform(lb, ub)
    wy = random_sign() * np.random.uniform(lb, ub)
    wz = random_sign() * np.random.uniform(lb, ub)
    print("the small constant wind is: ", wx, wy, wz)
    return ConstantWind(wx, wy, wz)

"""
small: 0.2 /0.1
medium: 0.2 /0.1 
big: 0.4 / 0.2 ### it's fair to increase the threshold with big wind
"""
def if_reach(obs, t):
    if np.linalg.norm(obs[:3]) < 0.4 and np.linalg.norm(obs[3:6]) < 0.2:
        return True, t
    return False, t

def get_ctrl_cost(action):
    # action is the real rotor speed, 2500 is the max rotor speed
    return np.linalg.norm(action) / np.linalg.norm(np.array([2500, 2500, 2500, 2500]))



##################### decent evaluation comparison here ############################

if __name__ == "__main__":
    # Set up the figure for plotting all the agents.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make the environments for the RL agents.
    num_quads = 1
    model = TD3()
    path = "/home/hsh/Code/rl_uav_control/rotorpy/learning/policies/DDPG/14-28-03/"
    # Load the policy
    model.load(path)
    model.eval_mode()
    reward_obj = CurriculumReward()
    reward_function = lambda obs, act, finish: reward_obj.hover_reward(obs, act, finish)
    wind = None


    def make_env():
        return gym.make(ENV_NAME,
                        control_mode =CONTROL_MODE,
                        reward_fn = reward_function,
                        quad_params = quad_params,
                        max_time = MAX_SIM_TIME,
                        world = None,
                        sim_rate = SIM_RATE,
                        render_mode='None',
                        render_fps = 60,
                        fig=fig,
                        ax=ax,
                        color='b',
                        wind_profile=wind)


    max_iter = 100

    succ_rate_policy = []
    reach_time_policy = []
    ctrl_cost_policy = []
    succ_rate_gc= []
    reach_time_gc = []
    ctrl_cost_gc = []


    for iter in range(max_iter):

        temp_cc_pl = 0
        temp_cc_gc = 0
        temp_reach_goal_pl = False
        temp_reach_goal_gc = False

        wind = small_random_wind()


        envs = [make_env() for _ in range(num_quads)]

        # Lastly, add in the baseline (SE3 controller) environment.
        envs.append(gym.make(ENV_NAME,
                             control_mode =CONTROL_MODE,
                             reward_fn = reward_function,
                             quad_params = quad_params,
                             max_time = MAX_SIM_TIME,
                             world = None,
                             sim_rate = SIM_RATE,
                             render_mode='None',
                             render_fps = 60,
                             fig=fig,
                             ax=ax,
                             color='k',
                             wind_profile=wind))  # Geometric controller


        # Evaluation...
        num_timesteps_idxs = [1]
        for (k, num_timesteps_idx) in enumerate(num_timesteps_idxs):  # For each num_timesteps index...

            print(f"[ppo_hover_eval.py]: Starting epoch {k+1} out of {len(num_timesteps_idxs)}.")

            initial_state = {'x': np.array([-4.0, -4.0, -4.0]),
                             'v': np.array([-0.0,-0.0,-0.0]),
                             'q': np.array([0.0, 0.0, 0.0, 1]), # [i,j,k,w]
                             'w': np.array([0.0, 0.0, 0.0]),
                             'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                             'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

            # Collect observations for each environment.
            observations = [env.reset(initial_state=initial_state, options={'pos_bound': POS_BOUND,
                                                                            'vel_bound': VEL_BOUND})[0] for env in envs]


            act_hss = [ActionContainer() for _ in range(num_quads)]
            # This is a list of env termination conditions so that the loop only ends when the final env is terminated.
            terminated = [False]*len(observations)


            j = 0  # Index for frames. Only updated when the last environment runs its update for the time step.
            while not all(terminated):
                frames = []  # Reset frames.
                for (i, env) in enumerate(envs):  # For each environment...
                    env.render()

                    if i == len(envs)-1:  # If it's the last environment, run the SE3 controller for the baseline.

                        # Unpack the observation from the environment
                        state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}

                        # Command the quad to hover.
                        flat = {'x': [0, 0, 0],
                                'x_dot': [0, 0, 0],
                                'x_ddot': [0, 0, 0],
                                'x_dddot': [0, 0, 0],
                                'yaw': 0,
                                'yaw_dot': 0,
                                'yaw_ddot': 0}
                        control_dict = baseline_controller.update(0, state, flat)

                        # Extract the commanded motor speeds.
                        cmd_motor_speeds = control_dict['cmd_motor_speeds']


                        # The environment expects the control inputs to all be within the range [-1,1]
                        action = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])

                        if if_reach(observations[i], env.unwrapped.t)[0] and not temp_reach_goal_gc:
                            succ_rate_gc.append(1)
                            reach_time_gc.append(env.unwrapped.t)
                            ctrl_cost_gc.append(temp_cc_gc)
                            temp_cc_gc = 0
                            #print("======================== GC win ++++++++++++++++++++++")
                            temp_reach_goal_gc = True
                        else:
                            temp_cc_gc += get_ctrl_cost(control_dict['cmd_motor_speeds'])


                    else: # For all other environments, get the action from the RL control policy.
                        cur_ah = act_hss[i].get()
                        action = model.select_action(observations[i], cur_ah)
                        act_hss[i].add(action)

                        if if_reach(observations[i], env.unwrapped.t)[0] and not temp_reach_goal_pl:
                            succ_rate_policy.append(1)
                            reach_time_policy.append(env.unwrapped.t)
                            ctrl_cost_policy.append(temp_cc_pl)
                            temp_cc_pl = 0
                            temp_reach_goal_pl = True
                        else:
                            cmd_motor_speeds = np.interp(action, [-1,1], [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max])
                            temp_cc_pl += get_ctrl_cost(cmd_motor_speeds)

                    # Step the environment forward.
                    observations[i], reward, terminated[i], truncated, info = env.step(action)



    print("The success rate of the policy is: ", len(succ_rate_policy) / max_iter)
    print("The average reach time of the policy is: ", np.mean(np.array(reach_time_policy)))
    print("The average control cost of the policy is: ", np.mean(np.array(ctrl_cost_policy)))

    print("The success rate of the geometric controller is: ", len(succ_rate_gc) / max_iter)
    print("The average reach time of the geometric controller is: ", np.mean(np.array(reach_time_gc)))
    print("The average control cost of the geometric controller is: ", np.mean(np.array(ctrl_cost_gc)))




