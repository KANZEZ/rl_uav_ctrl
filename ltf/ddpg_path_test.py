from matplotlib import pyplot as plt

import ddpg_eval
import numpy as np
import torch
import gymnasium as gym
from ddpg import DDPG
from reward import CurriculumReward
from env_helper import ActionContainer
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.wind.default_winds import NoWind, ConstantWind, SinusoidWind, LadderWind
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
baseline_controller = SE3Control(quad_params)

##################### decent evaluation comparison here ############################


if __name__ == "__main__":
    # Set up the figure for plotting all the agents.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make the environments for the RL agents.
    num_quads = 1
    pos_bound, vel_bound = 0.5, .0
    model = DDPG(13, 4)
    path = "/home/hsh/Code/rl_uav_control/rotorpy/learning/policies/DDPG/17-49-01/"
    # Load the policy
    model.load(path)
    model.eval_mode()
    reward_obj = CurriculumReward()
    reward_function = lambda obs, act, finish: reward_obj.reward(obs, act, finish)

    #wind = SinusoidWind(amplitudes=[4,-3,0.3], frequencies=[1,2,2])
    #wind = ConstantWind(-1, -1, -1)
    wind = None

    # trajectory it follows
    traj = TwoDLissajous(A=2.0, B=1.0, a=1.0, b=2.0, delta=0, height=1)
    #plot the traj
    t_traj = np.linspace(0, 50, 5000)
    x_traj = [traj.update(t)['x'] for t in t_traj]
    x_traj = np.array(x_traj)
    ax.plot(x_traj[:,0], x_traj[:,1], x_traj[:,2], 'r-', linewidth=0.7)

    prev_t = 0
    update_dur = 0.00000001

    def make_env():
        return gym.make("Quadrotor-v0",
                        control_mode ='cmd_motor_speeds',
                        reward_fn = reward_function,
                        quad_params = quad_params,
                        max_time = 50,
                        world = None,
                        sim_rate = 100,
                        render_mode='3D',
                        render_fps = 60,
                        fig=fig,
                        ax=ax,
                        color='b',
                        wind_profile=wind)

    envs = [make_env() for _ in range(num_quads)]

    # Lastly, add in the baseline (SE3 controller) environment.
    envs.append(gym.make("Quadrotor-v0",
                         control_mode ='cmd_motor_speeds',
                         reward_fn = reward_function,
                         quad_params = quad_params,
                         max_time = 50,
                         world = None,
                         sim_rate = 100,
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

        initial_state = {'x': np.array([0.0, 0.0, 1.0]),
                         'v': np.array([0.0,0.0,0.0]),
                         'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                         'w': np.array([0.0, 0.0, 0.0]),
                         'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                         'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        # Collect observations for each environment.
        observations = [env.reset(initial_state=initial_state, options={'pos_bound': pos_bound,
                                                                        'vel_bound': vel_bound})[0] for env in envs]


        act_hss = [ActionContainer(4) for _ in range(num_quads)]
        # This is a list of env termination conditions so that the loop only ends when the final env is terminated.
        terminated = [False]*len(observations)


        print(observations[0])

        ## get the new target:
        tar = traj.update(0)


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
                    control_dict = baseline_controller.update(0, state, flat)

                    # Extract the commanded motor speeds.
                    cmd_motor_speeds = control_dict['cmd_motor_speeds']

                    # The environment expects the control inputs to all be within the range [-1,1]
                    action = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])

                    # For the last environment, append the current timestep.
                    T.append(env.unwrapped.t)
                    #if env.unwrapped.t - prev_t > update_dur:
                    #prev_t = env.unwrapped.t
                    tar = traj.update(env.unwrapped.t)

                else: # For all other environments, get the action from the RL control policy.
                    #action, _ = model.predict(observations[i], deterministic=True)
                    cur_ah = act_hss[i].get()
                    offset_obs = np.copy(observations[i])
                    offset_obs[:3] -= tar['x']
                    #offset_obs[3:6] -= tar['x_dot'] # comment this will increase the performance, but theorically not right
                    action = model.select_action(offset_obs, cur_ah)
                    act_hss[i].add(action)

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

        # # Plot position vs time.
        # fig_pos, ax_pos = plt.subplots(nrows=3, ncols=1, num="Position vs Time")
        # #fig_pos.suptitle(f"Model: PPO/{models_available[model_idx]}, Num Timesteps: {extract_number(num_timesteps_list_sorted[num_timesteps_idx]):,}")
        # ax_pos[0].plot(T, x[:, 0], 'b-', linewidth=1, label="RL")
        # ax_pos[0].plot(T, x[:, 1:-1], 'b-', linewidth=1)
        # ax_pos[0].plot(T, x[:, -1], 'k-', linewidth=2, label="GC")
        # ax_pos[0].legend()
        # ax_pos[0].set_ylabel("X, m")
        # ax_pos[0].set_ylim([-7.5, 7.5])
        # ax_pos[1].plot(T, y[:, 0], 'b-', linewidth=1, label="RL")
        # ax_pos[1].plot(T, y[:, 1:-1], 'b-', linewidth=1)
        # ax_pos[1].plot(T, y[:, -1], 'k-', linewidth=2, label="GC")
        # ax_pos[1].set_ylabel("Y, m")
        # ax_pos[1].set_ylim([-7.5, 7.5])
        # ax_pos[2].plot(T, z[:, 0], 'b-', linewidth=1, label="RL")
        # ax_pos[2].plot(T, z[:, 1:-1], 'b-', linewidth=1)
        # ax_pos[2].plot(T, z[:, -1], 'k-', linewidth=2, label="GC")
        # ax_pos[2].set_ylabel("Z, m")
        # ax_pos[2].set_ylim([-7.5, 7.5])
        # ax_pos[2].set_xlabel("Time, s")

    plt.show()