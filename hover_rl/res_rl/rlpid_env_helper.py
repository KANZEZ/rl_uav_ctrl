import math

import numpy as np
import torch

from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.line_traj import Line3DTraj
from rotorpy.trajectories.helical_traj import Helical3DTrajectory
from rotorpy.trajectories.rectangle2d_traj import Rectangle2DTrajectory
from rotorpy.trajectories.back_and_forth_traj import BnF3DTraj

from config import *

class RlPidActionContainer(object):
    """
    add action history to the actor network input for speed up the control cmd response time
    This is a data structure that stores the action history of the agent.
    """

    def __init__(self):
        self.ah_len = ACTION_HISTORY_HORIZON
        self.action_dim = ACTION_DIM
        self.container = np.zeros((ACTION_HISTORY_HORIZON, ACTION_DIM))

    def add(self, action):
        self.container = np.roll(self.container, -1, 0)
        self.container[-1] = action

    def get(self):
        return self.container.flatten()

    def clear(self):
        self.container[:] = 0






class RlPidReplayBuffer(object):
    """
    A data structure that define the replay buffer for TD3 algorithm
    This buffer contain all the information needed for training the actor and critic network
    """
    def __init__(self, obs_dim=OBS_DIM,
                       action_dim=ACTION_DIM,
                       ah_len=ACTION_HISTORY_HORIZON,
                       future_traj_len=FUTURE_TRAJ_HORIZON,
                       max_size=REPLAY_BUFFER_SIZE):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self.action_history = np.zeros((max_size, action_dim * ah_len))
        self.future_traj = np.zeros((max_size, 3 * future_traj_len)) # traj position (x,y,z)
        self.wind = np.zeros((max_size, 3)) # wind speed (vx,vy,vz)
        self.pid_out = np.zeros((max_size, action_dim))


    def add(self, obs, action, next_obs, reward, done, ah, fu, wind, pid):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)

        self.action_history[self.ptr] = ah
        self.future_traj[self.ptr] = fu
        self.wind[self.ptr] = wind
        self.pid_out[self.ptr] = pid


        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_obs[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device),

            torch.FloatTensor(self.action_history[ind]).to(device),
            torch.FloatTensor(self.future_traj[ind]).to(device),
            torch.FloatTensor(self.wind[ind]).to(device),
            torch.FloatTensor(self.pid_out[ind]).to(device)
        )


    def reward_recalculation(self, reward_obj):
        for i in range(self.size):
            self.reward[i] = reward_obj.res_reward(self.next_obs[i], self.action[i], self.done[i])




# prob=np.array([0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1])
# test bnf : prob=np.array([0.01, 0.01, 0.02, 0.02, 0.9, 0.02, 0.02])
# simple prob
class RlPidTargetSelector(object):
    def __init__(self, prob=np.array([0.7, 0.1, 0.1, 0.1]), future_len=20):
        self.traj_obj = None
        self.future_len = int(future_len)
        self.t_res = 0.07

        #  start from easy to hard traj:

        self.LINE_PROB = prob[0]
        self.CIRCLE_PROB = prob[1]
        self.RECT_PROB = prob[2]
        self.HELIX_PROB = prob[3]

        self.prob = prob

        ### follow the order
        self.LINE_PROB_LIM = 0.1  # decrease
        self.CIRCLE_PROB_LIM = 0.3 # decrease
        self.RECT_PROB_LIM = 0.3   # decrease
        self.HELIX_PROB_LIM = 0.3     #increase

        self.UPDATE_NUM = 10

        self.LINE_RATE = -(self.LINE_PROB - self.LINE_PROB_LIM) / self.UPDATE_NUM
        self.CIRCLE_RATE = -(self.CIRCLE_PROB - self.CIRCLE_PROB_LIM) / self.UPDATE_NUM
        self.RECT_RATE = -(self.RECT_PROB - self.RECT_PROB_LIM) / self.UPDATE_NUM
        self.HELIX_RATE = -(self.HELIX_PROB - self.HELIX_PROB_LIM) / self.UPDATE_NUM



    def hover_traj_gen(self):
        self.traj_obj = HoverTraj()

    def circle_traj_gen(self):
        center = np.random.uniform(0, 0.2, 3)
        radius = np.random.uniform(1, 1.2)
        T = 2 * np.pi * radius / (1.5) # 1.5 is max avg speed
        freq = 2 * np.pi / T
        self.traj_obj = CircularTraj(center, radius, freq)

    def bnf_traj_gen(self):
        start = np.random.uniform(-1, 1, 3)
        end = np.random.uniform(-3, 3, 3)
        T = np.linalg.norm(end - start) / (1.5) # 1.5 is max avg speed
        self.traj_obj = BnF3DTraj(start, end, T)

    def line_traj_gen(self):
        start = np.random.uniform(-1, 1, 3)
        end = np.random.uniform(-1, 1, 3)
        #
        # start = np.array([-1, -1, -1])
        # end = np.array([2, 2, 1])

        T = np.linalg.norm(end - start) / (1.5) # 1.5 is max avg speed
        self.traj_obj = Line3DTraj(start, end, T=T)


    def rect_traj_gen(self):
        center = np.random.uniform(0, 0.1, 3)
        width = 1.5
        length = 1
        v_avg = 1.0
        self.traj_obj = Rectangle2DTrajectory(center, width=width, length=length, v_avg=v_avg)

    def figure_eight_traj_gen(self):
        A = np.random.uniform(1, 2)
        B = np.random.uniform(1, 2)
        a = np.random.uniform(3, 4)
        b = np.random.uniform(3, 4)
        height = np.random.uniform(-3, 3)
        self.traj_obj = TwoDLissajous(A=A, B=B, a=a, b=b, height=height)

    def helix_traj_gen(self):
        w = np.random.uniform(1, 1.1)
        A = np.random.uniform(1, 1.1)
        B = np.random.uniform(1, 1.1)
        vz = np.random.uniform(1.0, 1.2)
        z0 = np.random.uniform(0, 0.1)
        self.traj_obj = Helical3DTrajectory(w, A, B, vz, z0)


    def traj_gen(self):
        self.hover_traj_gen()
        return

        # select traj type
        traj_type = np.random.choice(['line', 'circle', 'rect', 'helix'], p=self.prob)
        print("current traj is ", traj_type)
        if traj_type == 'hover':
            self.hover_traj_gen()
        elif traj_type == 'circle':
            self.circle_traj_gen()
        elif traj_type == 'bnf':
            self.bnf_traj_gen()
        elif traj_type == 'line':
            self.line_traj_gen()
        elif traj_type == 'sin':
            self.sin_traj_gen()
        elif traj_type == 'rect':
            self.rect_traj_gen()
        elif traj_type == 'figure_eight':
            self.figure_eight_traj_gen()
        elif traj_type == 'helix':
            self.helix_traj_gen()



    def get_tar(self, t):
        # t: world simulation time
        tar = self.traj_obj.update(t) # flatness output
        # return x, xdot for now, maybe transform to state later
        return tar

    def update_traj_prob(self):
        self.LINE_PROB = max(self.LINE_PROB_LIM, self.LINE_PROB + self.LINE_RATE)
        self.CIRCLE_PROB = min(self.CIRCLE_PROB_LIM, self.CIRCLE_PROB + self.CIRCLE_RATE)
        self.RECT_PROB = min(self.RECT_PROB_LIM, self.RECT_PROB + self.RECT_RATE)
        self.HELIX_PROB = min(self.HELIX_PROB_LIM, self.HELIX_PROB + self.HELIX_RATE)

        self.prob = np.array([self.LINE_PROB, self.CIRCLE_PROB,
                              self.RECT_PROB, self.HELIX_PROB])

        print("current prob is ", self.prob)


    def traj_drawer(self):
        t = np.arange(0, 30, self.t_res)
        traj_pts = np.zeros((len(t), 3))
        for i in range(len(t)):
            traj_pts[i] = self.traj_obj.update(t[i])['x']
        return traj_pts

    def get_future_traj(self, t):
        future_list = np.zeros((self.future_len, 3))
        cur_t = t + self.t_res
        for i in range(self.future_len):
            future_list[i, :] = self.traj_obj.update(cur_t)['x']
            cur_t += self.t_res
        return future_list


