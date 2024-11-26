import numpy as np
import torch

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
    def __init__(self, traj_obj, world_time_res, future_len=20):
        self.traj_obj = traj_obj
        self.future_len = int(future_len)
        self.time_res = world_time_res

    def get_target(self, t):
        # t: world simulation time
        tar = self.traj_obj.update(t) # flatness output
        # return x, xdot for now, transform to state later
        return tar['x'], tar['x_dot']

    def get_future_tar(self, t):
        # t: world simulation time
        future_tar = [self.get_target(t + i * self.time_res) for i in range(self.future_len)]
        return future_tar


class RandomTrajGen(object):
    def __init__(self):
        pass


