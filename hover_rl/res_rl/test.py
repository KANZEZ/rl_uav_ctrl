from matplotlib import pyplot as plt

import numpy as np
import gymnasium as gym
from ddpg_res import RlPidTD3
from rlpid_reward import RlPidCurriculumReward
from rlpid_env_helper import RlPidActionContainer, RlPidTargetSelector
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.wind.default_winds import ConstantWind, RandomWind, SinusoidWind
from rotorpy.learning.quad_rlpid_env import QuadRlPidEnv
from config import *

controller = SE3Control(quad_params)

##################### decent evaluation comparison here ############################


if __name__ == "__main__":
    # Set up the figure for plotting all the agents.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make the environments for the RL agents.
    num_quads = 1
    model = RlPidTD3()
    path = "/home/hsh/Code/rl_uav_control/rotorpy/learning/policies/DDPG/18-06-12/"
    # Load the policy
    model.load(path)
    model.eval_mode()
    reward_obj = RlPidCurriculumReward()
    trk_reward_func = lambda obs, act, finish: reward_obj.res_reward(obs, act, finish)
    #wind = SinusoidWind(amplitudes=[4,-3,0.3], frequencies=[1,2,2])
    #wind = ConstantWind(-3, 3, 3)
    wind = RandomWind(WIND_LOWER_BOUND, WIND_UPPER_BOUND)
    #wind = None

    # trajectory it follows

    traj_selector = RlPidTargetSelector(prob=np.array([1.0 ,0, 0, 0]))

    def make_env():
        return gym.make("Quadrotor-v2",
                        control_mode ='cmd_motor_speeds',
                        reward_fn = trk_reward_func,
                        quad_params = quad_params,
                        max_time = 8,
                        world = None,
                        sim_rate = 100,
                        render_mode='3D',
                        render_fps = 60,
                        fig=fig,
                        ax=ax,
                        color='b',
                        wind_profile=wind,
                        target_selector=traj_selector,
                        controller=controller)

    envs = [make_env() for _ in range(num_quads)]

    # Lastly, add in the baseline (SE3 controller) environment.
    envs.append(gym.make("Quadrotor-v2",
                         control_mode ='cmd_motor_speeds',
                         reward_fn = trk_reward_func,
                         quad_params = quad_params,
                         max_time = 8,
                         world = None,
                         sim_rate = 100,
                         render_mode='3D',
                         render_fps = 60,
                         fig=fig,
                         ax=ax,
                         color='k',
                         wind_profile=wind,
                         target_selector=traj_selector,
                         controller=controller))  # Geometric controller


    # Evaluation...
    num_timesteps_idxs = [1]
    for (k, num_timesteps_idx) in enumerate(num_timesteps_idxs):  # For each num_timesteps index...

        print(f"[ppo_hover_eval.py]: Starting epoch {k+1} out of {len(num_timesteps_idxs)}.")

        initial_state = {'x': np.array([0.0, 0.0, 1.0]),
                         'v': np.array([0.0,0.0,0.0]),
                         'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                         'w': np.array([0.0, 0.0, 0.0]),
                         'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                         'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        # Collect observations for each environment.
        observations = [env.reset(initial_state='random', options={'pos_bound': POS_BOUND,
                                                                     'vel_bound': VEL_BOUND})[0] for env in envs]


        traj_obj = traj_selector.traj_obj
        t = np.linspace(0, 10, 1000)
        traj = np.zeros((len(t), 3))
        for i in range(len(t)):
            traj[i] = traj_obj.update(t[i])['x']
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 'r-', linewidth=0.7)

        act_hss = [RlPidActionContainer() for _ in range(num_quads)]
        # This is a list of env termination conditions so that the loop only ends when the final env is terminated.
        terminated = [False]*len(observations)


        print(observations[0])

        ## get the new target:
        tar = traj_obj.update(0)

        # Arrays for plotting position vs time.
        T = [0]
        x = [[obs[0] for obs in observations]]
        y = [[obs[1] for obs in observations]]
        z = [[obs[2] for obs in observations]]

        j = 0  # Index for frames. Only updated when the last environment runs its update for the time step.
        while not all(terminated):
            frames = []  # Reset frames.
            for (i, env) in enumerate(envs):  # For each environment...
                env.render()

                if i == len(envs)-1:  # If it's the last environment, run the SE3 controller for the baseline.

                    # Unpack the observation from the environment
                    state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}

                    # Command the quad to hover.
                    flat = tar
                    control_dict = controller.update(0, state, flat)

                    # Extract the commanded motor speeds.
                    cmd_motor_speeds = control_dict['cmd_motor_speeds']

                    # The environment expects the control inputs to all be within the range [-1,1]
                    action = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])

                    # For the last environment, append the current timestep.
                    T.append(env.unwrapped.t)
                    tar = traj_obj.update(env.unwrapped.t)

                else: # For all other environments, get the action from the RL control policy.

                    state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}
                    # Command the quad to hover.
                    flat = tar
                    control_dict = controller.update(0, state, flat)
                    # Extract the commanded motor speeds.
                    cmd_motor_speeds = control_dict['cmd_motor_speeds']
                    speed = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])

                    cur_ah = act_hss[i].get()
                    fu_traj = env.get_flat_future_traj()
                    action1 = model.select_action(observations[i], cur_ah, fu_traj, speed)
                    act_hss[i].add(action1)

                    #res_speed = np.interp(action1, [-1,1], [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max])

                    #cmd_motor_speeds += res_speed

                    #speedd = np.interp(action1, [-1,1], [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], )
                    action = action1
                    #print(speedd)
                # Step the environment forward.
                observations[i], reward, terminated[i], truncated, info = env.step(action)

            # Append arrays for plotting.
            x.append([obs[0] for obs in observations])
            y.append([obs[1] for obs in observations])
            z.append([obs[2] for obs in observations])

        # Convert to numpy arrays.
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        T = np.array(T)

        # Plot position vs time.
        fig_pos, ax_pos = plt.subplots(nrows=3, ncols=1, num="Position vs Time")
        #fig_pos.suptitle(f"Model: PPO/{models_available[model_idx]}, Num Timesteps: {extract_number(num_timesteps_list_sorted[num_timesteps_idx]):,}")
        ax_pos[0].plot(T, x[:, 0], 'b-', linewidth=1, label="RL")
        ax_pos[0].plot(T, x[:, 1:-1], 'b-', linewidth=1)
        ax_pos[0].plot(T, x[:, -1], 'k-', linewidth=2, label="GC")
        ax_pos[0].legend()
        ax_pos[0].set_ylabel("X, m")
        ax_pos[0].set_ylim([-7.5, 7.5])
        ax_pos[1].plot(T, y[:, 0], 'b-', linewidth=1, label="RL")
        ax_pos[1].plot(T, y[:, 1:-1], 'b-', linewidth=1)
        ax_pos[1].plot(T, y[:, -1], 'k-', linewidth=2, label="GC")
        ax_pos[1].set_ylabel("Y, m")
        ax_pos[1].set_ylim([-7.5, 7.5])
        ax_pos[2].plot(T, z[:, 0], 'b-', linewidth=1, label="RL")
        ax_pos[2].plot(T, z[:, 1:-1], 'b-', linewidth=1)
        ax_pos[2].plot(T, z[:, -1], 'k-', linewidth=2, label="GC")
        ax_pos[2].set_ylabel("Z, m")
        ax_pos[2].set_ylim([-7.5, 7.5])
        ax_pos[2].set_xlabel("Time, s")

    plt.show()