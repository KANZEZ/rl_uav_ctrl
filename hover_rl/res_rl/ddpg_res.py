"""
This script is the implementation of TD3 network and training algorithm.

The TD3 network and training algorithm is based on the following repositories:
https://github.com/sfujim/TD3/blob/master/TD3.py

We use DRL to train a residual policy for UAV, which is used for compensating the SE3 controller
policy under wind disturbance.

The input of actor: observations, action history(for fast response), future trajectory(for trajectory guidance)
The output of actor: speed of four motors on UAV

The input of critic: observations, actions, random wind disturbance
The output of critic: Q value
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpid_env_helper import RlPidReplayBuffer

from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module): # bzs = 256
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, output_dim)

    def forward(self, obs, ah, fu, pid):
        a = torch.cat([obs, ah, fu, pid], 1)
        a = F.tanh(self.l1(a))
        a = F.tanh(self.l2(a))
        a = F.tanh(self.l3(a))
        return torch.tanh(self.l4(a))


class Critic(nn.Module):  # bzs = 256
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        # Q2 architecture
        self.l4 = nn.Linear(input_dim, 64)
        self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(64, output_dim)

    def forward(self, obs, action, wind, pid):
        sa = torch.cat([obs, action, wind, pid], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, obs, action, wind, pid):
        sa = torch.cat([obs, action, wind, pid], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class RlPidTD3(object):
    def __init__(self):

        self.actor = Actor(OBS_DIM+ACTION_DIM*ACTION_HISTORY_HORIZON+3*FUTURE_TRAJ_HORIZON+ACTION_DIM, ACTION_DIM).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.act_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=100)

        self.critic = Critic(OBS_DIM+ACTION_DIM+WIND_DIM+ACTION_DIM, 1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.cri_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=100)

        self.discount = DISCOUNT_FACTOR
        self.tau = TAU
        self.min_noise_std = MIN_EXPL_NOISE_STD
        self.noise_std = NOISE_STD
        self.decay_rate = EXPL_NOISE_DECAY_RATE
        self.act_tar_noise = ACTOR_TAR_NOISE
        self.clip = NOISE_CLIP

        self.replay_buffer = RlPidReplayBuffer(OBS_DIM, ACTION_DIM)

    ########## output action ##########
    def select_action(self, obs, ah, fu, pid):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        ah = torch.FloatTensor(ah.reshape(1, -1)).to(device)
        fu = torch.FloatTensor(fu.reshape(1, -1)).to(device)
        pid = torch.FloatTensor(pid.reshape(1, -1)).to(device)
        return self.actor(obs, ah, fu, pid).cpu().data.numpy().flatten()

    ###### exploration noise ######
    def update_noise(self):
        self.noise_std = max(self.min_noise_std, self.noise_std * self.decay_rate)

    ########### replay buffer ###########
    def get_batch(self, batch_size):
        return self.replay_buffer.sample(batch_size, device)

    ####### training algorithm ###############

    def train(self, batch_size, train_mode):
        """
        train_mode: 0 for critic only, 1 for both
        """
        obs, action, next_obs, reward, done, ah, fu, wind, pid = self.get_batch(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.act_tar_noise
                     ).clamp(-self.clip, self.clip)
            next_ah = torch.cat([ah[:, 4:], action], 1)
            next_act = self.actor_target(next_obs, next_ah, fu, pid) + noise
            target_Q1, target_Q2 = self.critic_target(next_obs, next_act, wind, pid)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - done) * self.discount * target_Q

        # Get current Q estimate
        current_Q1, current_Q2 = self.critic(obs, action, wind, pid)
        # Compute critic loss
        critic_loss = F.huber_loss(current_Q1, target_Q) + F.huber_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.cri_scheduler.step()

        if train_mode == 1:
            # Compute actor loss
            actor_loss = -self.critic.Q1(obs, self.actor(obs, ah, fu, pid), wind, pid).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.act_scheduler.step()
            # Update the frozen target models
            self.update_target_critic()
            self.update_target_actor()

    def update_target_critic(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_target_actor(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    ###################  save, load #########################

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def eval_mode(self):
        self.critic.eval()
        self.actor.eval()

    def train_mode(self):
        self.critic.train()
        self.actor.train()