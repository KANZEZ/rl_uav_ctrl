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
    wind = SinusoidWind(amplitudes=[4,-3,1.3], frequencies=[1,2,2])
    #wind = ConstantWind(-2, -2, -1.7)
    #wind = None


    def make_env():
        return gym.make(ENV_NAME,
                        control_mode =CONTROL_MODE,
                        reward_fn = reward_function,
                        quad_params = quad_params,
                        max_time = MAX_SIM_TIME,
                        world = None,
                        sim_rate = SIM_RATE,
                        render_mode='3D',
                        render_fps = 60,
                        fig=fig,
                        ax=ax,
                        color='y',
                        wind_profile=wind)

    envs = [make_env() for _ in range(num_quads)]

    # Lastly, add in the baseline (SE3 controller) environment.
    envs.append(gym.make(ENV_NAME,
                         control_mode =CONTROL_MODE,
                         reward_fn = reward_function,
                         quad_params = quad_params,
                         max_time = MAX_SIM_TIME,
                         world = None,
                         sim_rate = SIM_RATE,
                         render_mode='3D',
                         render_fps = 60,
                         fig=fig,
                         ax=ax,
                         color='k',
                         wind_profile=wind))  # Geometric controller


    # Evaluation...
    num_timesteps_idxs = [1]
    for (k, num_timesteps_idx) in enumerate(num_timesteps_idxs):  # For each num_timesteps index...

        print(f"[ppo_hover_eval.py]: Starting epoch {k+1} out of {len(num_timesteps_idxs)}.")

        initial_state = {'x': np.array([-2.5, -3.0, -3.0]),
                              'v': np.array([-1.67,-1.38,-1.05]),
                              'q': np.array([0.4, 0.5, 0.66, 0.92]), # [i,j,k,w]
                              'w': np.array([0.5, 0.38, 1.0]),
                              'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        # initial_state = {'x': np.array([-3.0, -2.5, -2.48]),
        #                  'v': np.array([-0.1,-0.2,-0.1]),
        #                  'q': np.array([0.0, 0.0, 0.0, 1.0]), # [i,j,k,w]
        #                  'w': np.array([0.0, 0.0, 0.0]),
        #                  'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
        #                  'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}


        # Collect observations for each environment.
        observations = [env.reset(initial_state=initial_state, options={'pos_bound': POS_BOUND,
                                                                        'vel_bound': VEL_BOUND})[0] for env in envs]


        act_hss = [ActionContainer() for _ in range(num_quads)]
        # This is a list of env termination conditions so that the loop only ends when the final env is terminated.
        terminated = [False]*len(observations)


        print(observations[0])

        # Arrays for plotting position vs time.
        T = [0]
        x = [[obs[0] for obs in observations]]
        y = [[obs[1] for obs in observations]]
        z = [[obs[2] for obs in observations]]
        vx = [[obs[3] for obs in observations]]
        vy = [[obs[4] for obs in observations]]
        vz = [[obs[5] for obs in observations]]

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

                    # For the last environment, append the current timestep.
                    T.append(env.unwrapped.t)

                else: # For all other environments, get the action from the RL control policy.
                    cur_ah = act_hss[i].get()
                    action = model.select_action(observations[i], cur_ah)
                    act_hss[i].add(action)
                    #action, _ = model_ppo.predict(observations[i], deterministic=True)

                # Step the environment forward.
                observations[i], reward, terminated[i], truncated, info = env.step(action)

            # Append arrays for plotting.
            x.append([obs[0] for obs in observations])
            y.append([obs[1] for obs in observations])
            z.append([obs[2] for obs in observations])
            vx.append([obs[3] for obs in observations])
            vy.append([obs[4] for obs in observations])
            vz.append([obs[5] for obs in observations])


        # Convert to numpy arrays.
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        vx = np.array(vx)
        vy = np.array(vy)
        vz = np.array(vz)
        T = np.array(T)

        # Plot position vs time.
        fig_pos, ax_pos = plt.subplots(nrows=3, ncols=1, num="Position vs Time")
        fig_pos.suptitle("Position vs Time")
        #fig_pos.suptitle(f"Model: PPO/{models_available[model_idx]}, Num Timesteps: {extract_number(num_timesteps_list_sorted[num_timesteps_idx]):,}")
        ax_pos[0].plot(T, x[:, 0], 'b-', linewidth=3, label="Our policy")
        ax_pos[0].plot(T, x[:, 1:-1], 'k-', linewidth=3, label="SE3 control")
        #ax_pos[0].plot(T, x[:, -1], 'r-', linewidth=1, label="PPO")
        ax_pos[0].legend()
        ax_pos[0].set_ylabel("X")
        #ax_pos[0].set_ylim([-7.5, 7.5])
        ax_pos[1].plot(T, y[:, 0], 'b-', linewidth=3, label="Our policy")
        ax_pos[1].plot(T, y[:, 1:-1], 'k-', linewidth=3, label="SE3 control")
        #ax_pos[1].plot(T, y[:, -1], 'r-', linewidth=1, label="PPO")
        ax_pos[1].set_ylabel("Y")
        #ax_pos[1].set_ylim([-7.5, 7.5])
        ax_pos[2].plot(T, z[:, 0], 'b-', linewidth=3, label="Our policy")
        ax_pos[2].plot(T, z[:, 1:-1], 'k-', linewidth=3, label="SE3 control")
        #ax_pos[2].plot(T, z[:, -1], 'r-', linewidth=1, label="PPO")
        ax_pos[2].set_ylabel("Z")
        #ax_pos[2].set_ylim([-7.5, 7.5])
        ax_pos[2].set_xlabel("Time")

        fig_pos2, ax_pos2 = plt.subplots(nrows=3, ncols=1, num="Velocity vs Time")
        fig_pos2.suptitle("Velocity vs Time")
        #fig_pos.suptitle(f"Model: PPO/{models_available[model_idx]}, Num Timesteps: {extract_number(num_timesteps_list_sorted[num_timesteps_idx]):,}")
        ax_pos2[0].plot(T, vx[:, 0], 'b-', linewidth=3, label="Our policy")
        ax_pos2[0].plot(T, vx[:, 1:-1], 'k-', linewidth=3, label="SE3 control")
        #ax_pos[0].plot(T, x[:, -1], 'r-', linewidth=1, label="PPO")
        ax_pos2[0].legend()
        ax_pos2[0].set_ylabel("VX")
        #ax_pos[0].set_ylim([-7.5, 7.5])
        ax_pos2[1].plot(T, vy[:, 0], 'b-', linewidth=3, label="Our policy")
        ax_pos2[1].plot(T, vy[:, 1:-1], 'k-', linewidth=3, label="SE3 control")
        #ax_pos[1].plot(T, y[:, -1], 'r-', linewidth=1, label="PPO")
        ax_pos2[1].set_ylabel("VY")
        #ax_pos[1].set_ylim([-7.5, 7.5])
        ax_pos2[2].plot(T, vz[:, 0], 'b-', linewidth=3, label="Our policy")
        ax_pos2[2].plot(T, vz[:, 1:-1], 'k-', linewidth=3, label="SE3 control")
        #ax_pos[2].plot(T, z[:, -1], 'r-', linewidth=1, label="PPO")
        ax_pos2[2].set_ylabel("VZ")
        #ax_pos[2].set_ylim([-7.5, 7.5])
        ax_pos2[2].set_xlabel("Time")

    plt.show()