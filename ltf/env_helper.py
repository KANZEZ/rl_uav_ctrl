import numpy as np
import torch
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
    def __init__(self, prob=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) ,future_len=20):
        self.traj_obj = None
        self.future_len = int(future_len)
        #  start from easy to hard traj:
        #  hover traj, line traj, circle traj, Rectangle2DTrajectory, 3d circle traj, figure-8 traj
        self.prob = prob

    def hover_traj_gen(self):
        self.traj_obj = HoverTraj()

    def circle_traj_gen(self):
        center = np.random.rand(3) * 2
        radius = np.random.rand() * 3
        freq = np.random.rand() * 1
        self.traj_obj = CircularTraj(center, radius, freq)

    def three_d_circle_traj_gen(self):
        center = np.random.rand(3) * 2
        radius = np.random.rand() * 3
        radius = np.array([radius, radius, radius])
        freq = np.random.rand() * 0.5
        freq = np.array([freq, freq, freq])
        self.traj_obj = ThreeDCircularTraj(center, radius, freq)

    def line_traj_gen(self):
        start = np.random.rand(3) * 5
        end = np.random.rand(3) * 5
        T = np.random.rand() * np.linalg.norm(end - start) / 3.0 # 3.0 is max avg speed
        self.traj_obj = Line3DTraj(start, end, T=T)

    def sin_traj_gen(self):
        A = np.random.rand() * 5
        f = np.random.rand() * 0.5
        p = np.random.rand() * 2 * np.pi
        self.traj_obj = Sin3DTraj(A, f, p)

    def rect_traj_gen(self):
        center = np.random.rand(3) * 3
        width = np.random.rand() * 3
        length = np.random.rand() * 3
        v_avg = np.random.rand() * 2.5
        self.traj_obj = Rectangle2DTrajectory(center, width=width, length=length, v_avg=v_avg)

    def figure_eight_traj_gen(self):
        A = np.random.rand() * 5
        B = np.random.rand() * 5
        a = np.random.rand() * 5
        b = np.random.rand() * 5
        height = np.random.rand() * 3
        freq = np.random.rand() * 0.5
        self.traj_obj = TwoDLissajous(A=A, B=B, a=a, b=b, height=height, freq=freq)


    def traj_gen(self):
        # select traj type
        traj_type = np.random.choice(['hover', 'line', 'circle', 'rect', 'three_d_circle', 'figure_eight'], p=self.prob)
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
        return tar['x'], tar['x_dot']


