import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env_helper import ReplayBuffer
from end2end_config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, output_dim)

    def forward(self, obs, ah):
        a = torch.cat([obs, ah], 1)
        a = F.tanh(self.l1(a))
        a = F.tanh(self.l2(a))
        return torch.tanh(self.l3(a))


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

    def forward(self, obs, action):
        sa = torch.cat([obs, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, obs, action):
        sa = torch.cat([obs, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(self):
        """
        obs_dim: observation dimension, p, v, q, w (13)
        action_dim: action dimension, 4 (motors speed)
        """
        self.actor = Actor(OBS_DIM+ACTION_DIM*ACTION_HISTORY_HORIZON, ACTION_DIM).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.act_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=100)

        self.critic = Critic(OBS_DIM+ACTION_DIM, 1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.cri_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=100)

        self.discount = DISCOUNT_FACTOR
        self.tau = TAU
        self.min_noise = MIN_EXPL_NOISE_STD
        self.noise_std = NOISE_STD
        self.decay_rate = EXPL_NOISE_DECAY_RATE
        self.act_tar_noise = ACTOR_TAR_NOISE
        self.clip = NOISE_CLIP
        self.bzs = BATCH_SIZE

        self.replay_buffer = ReplayBuffer()

    ########## output action ##########
    def select_action(self, obs, ah):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        ah = torch.FloatTensor(ah.reshape(1, -1)).to(device)
        return self.actor(obs, ah).cpu().data.numpy().flatten()

    ###### exploration noise ######
    def update_noise(self):
        self.noise_std = max(self.min_noise, self.noise_std * self.decay_rate)

    ########### replay buffer ###########
    def get_batch(self, batch_size):
        return self.replay_buffer.sample(batch_size, device)

####### training algorithm ###############

    def train(self, train_mode):
        """
        train_mode: 0 for critic only, 1 for both
        """
        obs, action, ah, next_obs, reward, done = self.get_batch(self.bzs)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.act_tar_noise
            ).clamp(-self.clip, self.clip)
            next_ah = torch.cat([ah[:, 4:], action], 1)
            next_act = self.actor_target(next_obs, next_ah) + noise
            target_q1, target_q2 = self.critic_target(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.discount * target_q

        # Get current Q estimate
        current_q1, current_q2 = self.critic(obs, action)
        # Compute critic loss
        critic_loss = F.huber_loss(current_q1, target_q) + F.huber_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.cri_scheduler.step()

        if train_mode == 1:
            # Compute actor loss
            actor_loss = -self.critic.Q1(obs, self.actor(obs, ah)).mean()
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