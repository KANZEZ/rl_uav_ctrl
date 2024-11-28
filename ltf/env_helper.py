import numpy as np
import torch

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.circular_traj import CircularTraj, ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.trajectories.line_traj import Line3DTraj
from rotorpy.trajectories.sin_traj import Sin3DTraj
from rotorpy.trajectories.helical_traj import Helical3DTrajectory
from rotorpy.trajectories.polynomial_traj import Polynomial
from rotorpy.trajectories.polyseg_traj import Polyseg3DTrajectory
from rotorpy.trajectories.rectangle2d_traj import Rectangle2DTrajectory
from rotorpy.trajectories.back_and_forth_traj import BnF3DTraj

class ActionContainer(object):
    def __init__(self, action_dim, ah_len=int(32)):
        self.ah_len = ah_len
        self.action_dim = action_dim
        self.container = np.zeros((ah_len, action_dim))

    def add(self, action):
        self.container = np.roll(self.container, -1, 0)
        self.container[-1] = action

    def get(self):
        return self.container.flatten()

    def clear(self):
        self.container[:] = 0


class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, ah_len=int(32), max_size=int(3e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.action_history = np.zeros((max_size, action_dim * ah_len))

    def add(self, obs, action, ah, next_obs, reward, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.action_history[self.ptr] = ah
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.action_history[ind]).to(device),
            torch.FloatTensor(self.next_obs[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )


    def reward_recalculation(self, reward_obj):
        for i in range(self.size):
            self.reward[i] = reward_obj.reward(self.next_obs[i], self.action[i], self.done[i])


class TargetSelector(object):
    def __init__(self, prob=np.array([0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1]), future_len=20):
        self.traj_obj = None
        self.future_len = int(future_len)
        #  start from easy to hard traj:
        #  hover traj, line traj, circle traj, Rectangle2DTrajectory, 3d circle traj
        self.prob = prob

        self.HOVER_PROB = 0.1
        self.LINE_PROB = 0.15
        self.CIRCLE_PROB = 0.2
        self.RECT_PROB = 0.2
        self.BNF_PROB = 0.15
        self.HELIX_PROB = 0.1
        self.FIGURE_EIGHT_PROB = 0.1

        self.HOVER_PROB_LIM = 0.01 # decrease
        self.LINE_PROB_LIM = 0.01  # decrease
        self.CIRCLE_PROB_LIM = 0.05 # decrease
        self.RECT_PROB_LIM = 0.15   # decrease
        self.BNF_PROB_LIM = 0.15      #increase
        self.HELIX_PROB_LIM = 0.25     #increase
        self.FIGURE_EIGHT_PROB_LIM = 0.38 # increase

        self.UPDATE_NUM = 10

        self.HOVER_RATE = -(self.HOVER_PROB - self.HOVER_PROB_LIM) / self.UPDATE_NUM
        self.LINE_RATE = -(self.LINE_PROB - self.LINE_PROB_LIM) / self.UPDATE_NUM
        self.CIRCLE_RATE = -(self.CIRCLE_PROB - self.CIRCLE_PROB_LIM) / self.UPDATE_NUM
        self.RECT_RATE = -(self.RECT_PROB - self.RECT_PROB_LIM) / self.UPDATE_NUM
        self.FIGURE_EIGHT_RATE = (self.FIGURE_EIGHT_PROB - self.FIGURE_EIGHT_PROB_LIM) / self.UPDATE_NUM
        self.BNF_RATE = (self.BNF_PROB - self.BNF_PROB) / self.UPDATE_NUM
        self.HELIX_RATE = (self.HELIX_PROB - self.HELIX_PROB) / self.UPDATE_NUM


    def hover_traj_gen(self):
        self.traj_obj = HoverTraj()

    def circle_traj_gen(self):
        center = np.random.uniform(-3, 3, 3)
        radius = np.random.uniform(1, 2)
        freq = np.random.uniform(0.5, 2)
        self.traj_obj = CircularTraj(center, radius, freq)

    def bnf_traj_gen(self):
        start = np.random.uniform(-5, 5, 3)
        end = np.random.uniform(-5, 5, 3)
        T = np.random.uniform(0.1, 1) * np.linalg.norm(end - start) / 3.0 # 3.0 is max avg speed
        self.traj_obj = BnF3DTraj(start, end, T)

    def line_traj_gen(self):
        start = np.random.uniform(-5, 5, 3)
        end = np.random.uniform(-5, 5, 3)
        T = np.random.uniform(0.1, 1) * np.linalg.norm(end - start) / 3.0 # 3.0 is max avg speed
        self.traj_obj = Line3DTraj(start, end, T=T)


    def rect_traj_gen(self):
        center = np.random.uniform(-3, 3, 3)
        width = np.random.rand() * 4
        length = np.random.rand() * 4
        v_avg = np.random.rand() * 2.5
        self.traj_obj = Rectangle2DTrajectory(center, width=width, length=length, v_avg=v_avg)

    def figure_eight_traj_gen(self):
        A = np.random.uniform(1, 4)
        B = np.random.uniform(1, 4)
        a = np.random.uniform(1, 4)
        b = np.random.uniform(1, 2)
        height = np.random.uniform(-3, 3)
        freq = np.random.rand() * 0.5
        self.traj_obj = TwoDLissajous(A=A, B=B, a=a, b=b, height=height, freq=freq)

    def helix_traj_gen(self):
        w = np.random.uniform(1, 4)
        A = np.random.uniform(1, 4)
        B = np.random.uniform(1, 4)
        vz = np.random.uniform(0.2, 2)
        z0 = np.random.uniform(-5, 1)
        self.traj_obj = Helical3DTrajectory(w, A, B, vz, z0)


    def traj_gen(self):
        # select traj type
        traj_type = np.random.choice(['hover', 'line', 'circle', 'rect', 'three_d_circle', 'figure_eight'], p=self.prob)
        print("current traj is ", traj_type)
        if traj_type == 'hover':
            self.hover_traj_gen()
        elif traj_type == 'circle':
            self.circle_traj_gen()
        elif traj_type == 'three_d_circle':
            self.three_d_circle_traj_gen()
        elif traj_type == 'line':
            self.line_traj_gen()
        elif traj_type == 'sin':
            self.sin_traj_gen()
        elif traj_type == 'rect':
            self.rect_traj_gen()
        elif traj_type == 'figure_eight':
            self.figure_eight_traj_gen()


    def get_tar(self, t):
        # t: world simulation time
        tar = self.traj_obj.update(t) # flatness output
        # return x, xdot for now, transform to state later
        return tar

    def update_traj_prob(self):
        self.HOVER_PROB = max(self.HOVER_PROB_LIM, self.HOVER_PROB - self.HOVER_RATE)
        self.LINE_PROB = max(self.LINE_PROB_LIM, self.LINE_PROB - self.LINE_RATE)
        self.CIRCLE_PROB = max(self.CIRCLE_PROB_LIM, self.CIRCLE_PROB - self.CIRCLE_RATE)
        self.RECT_PROB = max(self.RECT_PROB_LIM, self.RECT_PROB - self.RECT_RATE)
        self.FIGURE_EIGHT_PROB = max(self.FIGURE_EIGHT_PROB_LIM, self.FIGURE_EIGHT_PROB - self.FIGURE_EIGHT_RATE)
        self.BNF_PROB = max(self.BNF_PROB_LIM, self.BNF_PROB - self.BNF_RATE)
        self.HELIX_PROB = max(self.HELIX_PROB_LIM, self.HELIX_PROB - self.HELIX_RATE)

        self.prob = np.array([self.HOVER_PROB, self.LINE_PROB, self.CIRCLE_PROB,
                              self.RECT_PROB, self.BNF_PROB, self.HELIX_PROB, self.FIGURE_EIGHT_PROB])
        prob_l = np.linalg.norm(self.prob)

        self.prob /= prob_l


