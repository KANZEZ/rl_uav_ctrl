from typing import Optional

import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform._rotation import Rotation

from rotorpy.world import World
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params as crazyflie_params
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from rotorpy.utils.shapes import Quadrotor

import gymnasium as gym
from gymnasium import spaces

import math
from copy import deepcopy

############################### ENVIRONMENT ####################################
class QuadrotorTrackingEnv(gym.Env):

    metadata = {"render_modes": ["None", "3D", "console"],
                "render_fps": 30,
                "control_modes": ['cmd_motor_speeds', 'cmd_motor_thrusts', 'cmd_ctbr', 'cmd_ctbm', 'cmd_vel']}

    def __init__(self, initial_state = {'x': np.array([0,0,0]),
                                        'v': np.zeros(3,),
                                        'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                        'w': np.zeros(3,),
                                        'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                        'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])},
                 control_mode = 'cmd_vel',
                 reward_fn = hover_reward,
                 target_selector=None,
                 quad_params = crazyflie_params,
                 max_time = 10,                # Maximum time to run the simulation for in a single session.
                 wind_profile = None,         # wind profile object, if none is supplied it will choose no wind.
                 world        = None,         # The world object
                 sim_rate = 100,              # The update frequency of the simulator in Hz
                 aero = True,                 # Whether or not aerodynamic wrenches are computed.
                 render_mode = "None",        # The rendering mode
                 render_fps = 30,             # The rendering frames per second. Lower this for faster visualization.
                 fig = None,                  # Figure for rendering. Optional.
                 ax = None,                   # Axis for rendering. Optional.
                 color = None,                # The color of the quadrotor.
                 obs_dim = 13,                # The observation dimension
                 action_dim = 4):

        super(QuadrotorTrackingEnv, self).__init__()
        self.metadata['render_fps'] = render_fps

        self.initial_state = initial_state

        self.vehicle_state = initial_state

        assert control_mode in self.metadata["control_modes"]  # Don't accept improper control modes
        self.control_mode = control_mode

        self.sim_rate = sim_rate
        self.t_step = 1/self.sim_rate
        self.reward_fn = reward_fn
        self.traj_obj = target_selector

        # Create quadrotor from quad params and control abstraction.
        self.quadrotor = Multirotor(quad_params=quad_params, initial_state=initial_state, control_abstraction=control_mode, aero=aero)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.t = 0  # The current time of the instance
        self.max_time = max_time

        ############ OBSERVATION SPACE

        # The observation state is the full state of the quadrotor.
        #     position, x, observation_state[0:3]
        #     velocity, v, observation_state[3:6]
        #     orientation, q, observation_state[6:10]
        #     body rates, w, observation_state[10:13]
        # For simplicitly, we assume these observations can lie within -inf to inf.

        self.observation_space = spaces.Box(low = -np.inf, high=np.inf, shape = (obs_dim,), dtype=np.float32)

        ############ ACTION SPACE

        # For generalizability, we assume the controller outputs 4 numbers between -1 and 1. Depending on the control mode, we scale these appropriately.

        if self.control_mode == 'cmd_vel':
            self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low = -1, high = 1, shape = (4,), dtype=np.float32)

        ######  Min/max values for scaling control outputs.

        self.rotor_speed_max = self.quadrotor.rotor_speed_max
        self.rotor_speed_min = self.quadrotor.rotor_speed_min

        # Compute the min/max thrust by assuming the rotor is spinning at min/max speed. (also generalizes to bidirectional rotors)
        self.max_thrust = self.quadrotor.k_eta * self.quadrotor.rotor_speed_max**2
        self.min_thrust = self.quadrotor.k_eta * self.quadrotor.rotor_speed_min**2

        # Find the maximum moment on each axis, N-m
        self.max_roll_moment = self.max_thrust * np.abs(self.quadrotor.rotor_pos['r1'][1])
        self.max_pitch_moment = self.max_thrust * np.abs(self.quadrotor.rotor_pos['r1'][0])
        self.max_yaw_moment = self.quadrotor.k_m * self.quadrotor.rotor_speed_max**2

        # Set the maximum body rate on each axis (this is hand selected), rad/s
        self.max_roll_br = 7.0
        self.max_pitch_br = 7.0
        self.max_yaw_br = 3.0

        # Set the maximum speed command (this is hand selected), m/s
        self.max_vel = 3/math.sqrt(3)   # Selected so that at most the max speed is 3 m/s

        ###################################################################################################

        # Save the order of magnitude of the rotor speeds for later normalization
        self.rotor_speed_order_mag = math.floor(math.log(self.quadrotor.rotor_speed_max, 10))

        if world is None:
            # If no world is specified, assume that it means that the intended world is free space.
            wbound = 6
            self.world = World.empty((-wbound, wbound, -wbound,
                                      wbound, -wbound, wbound))
        else:
            self.world = world

        if wind_profile is None:
            # If wind is not specified, default to no wind.
            from rotorpy.wind.default_winds import NoWind
            self.wind_profile = NoWind()
        else:
            self.wind_profile = wind_profile

        if self.render_mode == '3D':
            if fig is None and ax is None:
                self.fig = plt.figure('Visualization')
                self.ax = self.fig.add_subplot(projection='3d')
            else:
                self.fig = fig
                self.ax = ax
            if color is None:
                colors = list(mcolors.CSS4_COLORS)
            else:
                colors = [color]
            self.quad_obj = Quadrotor(self.ax, wind=True, color=np.random.choice(colors), wind_scale_factor=5)
            self.world_artists = None
            self.title_artist = self.ax.set_title('t = {}'.format(self.t))

        self.rendering = False   # Bool for tracking when the renderer is actually rendering a frame.

        return


    def _get_obs(self):
        return np.concatenate((self.vehicle_state['x'], self.vehicle_state['v'],
                               self.vehicle_state['q'], self.vehicle_state['w']))


    def _get_info(self):
        return {}

    def render(self):
        if self.render_mode == '3D':
            self._plot_quad()
        elif self.render_mode == 'console':
            self._print_quad()

    def close(self):
        if self.fig is not None:
            # Close the plots
            plt.close('all')


    def reset(self, seed=None, initial_state='random', options={'pos_bound': 1, 'vel_bound': 0}):
        # Reset the state of the environment to an initial state
        assert options['pos_bound'] >= 0 and options['vel_bound'] >= 0 , "Bounds must be greater than or equal to 0."

        super().reset(seed=seed)

        self.traj_obj.traj_gen()
        tar = self.traj_obj.get_tar(0)  ### env start from 0
        tar_pos, tar_vel = tar['x'], tar['x_dot']

        if initial_state == 'random':
            # Randomly select an initial state for the quadrotor. At least assume it is level.
            pos = np.random.uniform(low=-options['pos_bound'], high=options['pos_bound'], size=(3,))
            vel = np.random.uniform(low=-options['vel_bound'], high=options['vel_bound'], size=(3,))
            w = np.random.uniform(low=-1.5, high=1.5, size=(3,))
            q = Rotation.from_euler('zyx', np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(3,))).as_quat()
            state = {'x': pos,
                     'v': vel,
                     'q': q, #np.array([0,0,0,1]), # [i,j,k,w]   # [i,j,k,w]
                     'w': w, #
                     'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                     'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        elif initial_state == 'deterministic':
            state = self.initial_state

        ### add gudiance
        elif initial_state == 'guidance':
            pos = tar_pos
            vel = tar_vel
            w = np.random.uniform(low=-1.5, high=1.5, size=(3,))
            q = Rotation.from_euler('zyx', np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(3,))).as_quat()
            state = {'x': pos,
                     'v': vel,
                     'q': q, # [i,j,k,w]
                     'w': w,
                     'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                     'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        elif isinstance(initial_state, dict):
            # Ensure the correct keys are in dict.
            if all(key in initial_state for key in ('x', 'v', 'q', 'w', 'wind', 'rotor_speeds')):
                state = initial_state
            else:
                raise KeyError("Missing state keys in your initial_state. You must specify values for ('x', 'v', 'q', 'w', 'wind', 'rotor_speeds')")

        else:
            raise ValueError("You must either specify 'random', 'deterministic', or provide a dict containing your desired initial state.")

        # Set the initial state.
        self.vehicle_state = state

        # Reset the time
        self.t = 0.0

        # Reset the reward
        self.reward = 0.0

        # Now get observation and info using the new state
        observation = self._get_obs()

        self.render()

        info = {}

        return observation, info


    def step(self, action):

        """
        Step the quadrotor dynamics forward by one step based on the policy action.
        Inputs:
            action: The action is a 4x1 vector which depends on the control abstraction:

            if control_mode == 'cmd_vel':
                action[0] (-1,1) := commanded velocity in x direction (will be rescaled to m/s)
                action[1] (-1,1) := commanded velocity in y direction (will be rescaled to m/s)
                action[2] (-1,1) := commanded velocity in z direction (will be rescaled to m/s)
            if control_mode == 'cmd_ctbr':
                action[0] (-1,1) := the thrust command (will be rescaled to Newtons)
                action[1] (-1,1) := the roll body rate (will be rescaled to rad/s)
                action[2] (-1,1) := the pitch body rate (will be rescaled to rad/s)
                action[3] (-1,1) := the yaw body rate (will be rescaled to rad/s)
            if control_mode == 'cmd_ctbm':
                action[0] (-1,1) := the thrust command (will be rescaled to Newtons)
                action[1] (-1,1) := the roll moment (will be rescaled to Newton-meters)
                action[2] (-1,1) := the pitch moment (will be rescaled to Newton-meters)
                action[3] (-1,1) := the yaw moment (will be rescaled to Newton-meters)
            if control_mode == 'cmd_motor_speeds':
                action[0] (-1,1) := motor 1 speed (will be rescaled to rad/s)
                action[1] (-1,1) := motor 2 speed (will be rescaled to rad/s)
                action[2] (-1,1) := motor 3 speed (will be rescaled to rad/s)
                action[3] (-1,1) := motor 4 speed (will be rescaled to rad/s)
            if control_mode == 'cmd_motor_forces':
                action[0] (-1,1) := motor 1 force (will be rescaled to Newtons)
                action[1] (-1,1) := motor 2 force (will be rescaled to Newtons)
                action[2] (-1,1) := motor 3 force (will be rescaled to Newtons)
                action[3] (-1,1) := motor 4 force (will be rescaled to Newtons)

        """

        # First rescale the action and get the appropriate control dictionary given the control mode.
        self.control_dict = self.rescale_action(action)

        # Now update the wind state using the wind profile
        self.vehicle_state['wind'] = self.wind_profile.update(self.t, self.vehicle_state['x'])
        # Last perform forward integration using the commanded motor speed and the current state
        self.vehicle_state = self.quadrotor.step(self.vehicle_state, self.control_dict, self.t_step)
        observation = self._get_obs() # next state observation
        # Update t by t_step
        self.t += self.t_step

        # Check for safety
        safe = self.safety_exit()

        # Determine whether or not the session should terminate.
        terminated = (self.t >= self.max_time) or not safe

        r = self._get_reward(observation, action, terminated)
        self.reward = r if safe else -500

        self.render()

        info = self._get_info() # must have 4-5 output
        truncated = False

        return observation, self.reward, terminated, truncated, info


    def _get_reward(self, observation, action, terminated):
        ### add target
        tar = self.traj_obj.get_tar(self.t)
        return self.reward_fn(observation, action, terminated, tar)


    def rescale_action(self, action):
        """
        Rescales the action to within the control limits and then assigns the appropriate dictionary.
        """

        control_dict = {}

        if self.control_mode == 'cmd_ctbm':
            # Scale action[0] to (0,1) and then scale to the max thrust
            cmd_thrust = np.interp(action[0], [-1,1], [self.quadrotor.num_rotors*self.min_thrust, self.quadrotor.num_rotors*self.max_thrust])

            # Scale the moments
            cmd_roll_moment = np.interp(action[1], [-1,1], [-self.max_roll_moment, self.max_roll_moment])
            cmd_pitch_moment = np.interp(action[2], [-1,1], [-self.max_pitch_moment, self.max_pitch_moment])
            cmd_yaw_moment = np.interp(action[3], [-1,1], [-self.max_yaw_moment, self.max_yaw_moment])

            control_dict['cmd_thrust'] = cmd_thrust
            control_dict['cmd_moment'] = np.array([cmd_roll_moment, cmd_pitch_moment, cmd_yaw_moment])

        elif self.control_mode == 'cmd_ctbr':
            # Scale action to min and max thrust.
            cmd_thrust = np.interp(action[0], [-1, 1], [self.quadrotor.num_rotors*self.min_thrust, self.quadrotor.num_rotors*self.max_thrust])

            # Scale the body rates.
            cmd_roll_br = np.interp(action[1], [-1,1], [-self.max_roll_br, self.max_roll_br])
            cmd_pitch_br = np.interp(action[2], [-1,1], [-self.max_pitch_br, self.max_pitch_br])
            cmd_yaw_br = np.interp(action[3], [-1,1], [-self.max_yaw_br, self.max_yaw_br])

            control_dict['cmd_thrust'] = cmd_thrust
            control_dict['cmd_w'] = np.array([cmd_roll_br, cmd_pitch_br, cmd_yaw_br])

        elif self.control_mode == 'cmd_motor_speeds':
            # Scale the action to min and max motor speeds.
            control_dict['cmd_motor_speeds'] = np.interp(action, [-1,1], [self.rotor_speed_min, self.rotor_speed_max])

        elif self.control_mode == 'cmd_motor_thrusts':
            # Scale the action to min and max rotor thrusts.
            control_dict['cmd_motor_thrusts'] = np.interp(action, [-1,1], [self.min_thrust, self.max_thrust])

        elif self.control_mode == 'cmd_vel':
            # Scale the velcoity to min and max values.
            control_dict['cmd_v'] = np.interp(action, [-1,1], [-self.max_vel, self.max_vel])

        return control_dict


    def safety_exit(self):
        """
        Return exit status if any safety condition is violated, otherwise None.
        """
        if np.any(np.abs(self.vehicle_state['v']) > 100):
            return False
        if np.any(np.abs(self.vehicle_state['w']) > 100):
            return False
        if self.vehicle_state['x'][0] < self.world.world['bounds']['extents'][0] or self.vehicle_state['x'][0] > self.world.world['bounds']['extents'][1]:
            return False
        if self.vehicle_state['x'][1] < self.world.world['bounds']['extents'][2] or self.vehicle_state['x'][1] > self.world.world['bounds']['extents'][3]:
            return False
        if self.vehicle_state['x'][2] < self.world.world['bounds']['extents'][4] or self.vehicle_state['x'][2] > self.world.world['bounds']['extents'][5]:
            return False
        #
        # if self.vehicle_state['x'][0] < -0.6 or self.vehicle_state['x'][0] > 0.6:
        #     return False
        # if self.vehicle_state['x'][1] < -0.6 or self.vehicle_state['x'][1] > 0.6:
        #     return False
        # if self.vehicle_state['x'][2] < -0.6 or self.vehicle_state['x'][2] > 0.6:
        #     return False


        if len(self.world.world.get('blocks', [])) > 0:
            # If a world has objects in it we need to check for collisions.
            collision_pts = self.world.path_collisions(self.vehicle_state['x'], 0.25)
            no_collision = collision_pts.size == 0
            if not no_collision:
                return False
        return True


    def _plot_quad(self):

        if abs(self.t / (1/self.metadata['render_fps']) - round(self.t / (1/self.metadata['render_fps']))) > 5e-2:
            self.rendering = False  # Set rendering bool to false.
            return

        self.rendering = True # Set rendering bool to true.

        plot_position = deepcopy(self.vehicle_state['x'])
        plot_rotation = Rotation.from_quat(self.vehicle_state['q']).as_matrix()
        plot_wind = deepcopy(self.vehicle_state['wind'])

        if self.world_artists is None and not ('x' in self.ax.get_xlabel()):
            self.world_artists = self.world.draw(self.ax)
            self.ax.plot(0, 0, 0, 'go')

        self.quad_obj.transform(position=plot_position, rotation=plot_rotation, wind=plot_wind)
        self.title_artist.set_text('t = {:.2f}'.format(self.t))

        plt.pause(1e-9)

        return

    def _print_quad(self):

        print("Time: %3.2f \t Position: (%3.2f, %3.2f, %3.2f) \t Reward: %3.2f" % (self.t, self.vehicle_state['x'][0], self.vehicle_state['x'][1], self.vehicle_state['x'][2], self.reward))




