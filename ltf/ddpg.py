"""
todo:
1. Implement the DDPG algorithm in rotorpy/learning/ddpg.py.
2. add the detail/tricks mentioned in the paper
2.1 actor/critic training interval (done)
2.2 noise decay (done)
2.3 reward shaping: cirriculum learning and (reward recalculation) (done)
2.4 action history (done)
2.5 Guidance (done)

3. train the model in the wind env

4. implement linear MPC

5. compare the performance with baseline RL\MPC\PID in the wind env
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env_helper import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module): # bzs = 256
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

        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, output_dim)

    def forward(self, obs, action):
        q = F.tanh(self.l1(torch.cat([obs, action], 1)))
        q = F.tanh(self.l2(q))
        return self.l3(q)


class DDPG(object):
    def __init__(self, obs_dim, action_dim, discount=0.99, tau=0.005, max_ah_size=32):
        """
        obs_dim: observation dimension, p, v, q, w (13)
        action_dim: action dimension, 4 (motors speed)
        """
        self.actor = Actor(obs_dim+action_dim*max_ah_size, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(obs_dim+action_dim, 1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.discount = discount
        self.tau = tau
        self.min_noise = 0.01
        self.noise_std = 0.5
        self.decay_rate = 0.9

        self.replay_buffer = ReplayBuffer(obs_dim, action_dim)

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

    def train_critic(self, obs, action, ah,
                     next_obs, reward, done):
        # Compute the target Q value
        next_ah = torch.cat([ah[:, 4:], action], 1)
        target_Q = self.critic_target(next_obs, self.actor_target(next_obs, next_ah))
        target_Q = reward + ((1.0 - done) * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.huber_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_target_critic(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def train_actor(self, obs, ah):
        # Compute actor loss
        actor_loss = -self.critic(obs, self.actor(obs, ah)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def update_target_actor(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


###################    save, load #########################

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