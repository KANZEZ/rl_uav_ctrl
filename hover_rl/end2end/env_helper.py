import numpy as np
import torch
from end2end_config import *

class ActionContainer(object):
    def __init__(self):
        self.action_dim = ACTION_DIM
        self.container = np.zeros((ACTION_HISTORY_HORIZON, ACTION_DIM))

    def add(self, action):
        self.container = np.roll(self.container, -1, 0)
        self.container[-1] = action

    def get(self):
        return self.container.flatten()

    def clear(self):
        self.container[:] = 0


class ReplayBuffer(object):
    def __init__(self):
        self.max_size = REPLAY_BUFFER_SIZE

        self.obs = np.zeros((REPLAY_BUFFER_SIZE, OBS_DIM))
        self.action = np.zeros((REPLAY_BUFFER_SIZE, ACTION_DIM))
        self.next_obs = np.zeros((REPLAY_BUFFER_SIZE, OBS_DIM))
        self.reward = np.zeros((REPLAY_BUFFER_SIZE, 1))
        self.done = np.zeros((REPLAY_BUFFER_SIZE, 1))
        self.action_history = np.zeros((REPLAY_BUFFER_SIZE, ACTION_DIM * ACTION_HISTORY_HORIZON))

        self.ptr = 0
        self.size = 0


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
            self.reward[i] = reward_obj.hover_reward(self.next_obs[i],
                                                     self.action[i],
                                                     self.done[i])