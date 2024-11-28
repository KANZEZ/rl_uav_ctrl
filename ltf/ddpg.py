"""
todo:
1. Implement the DDPG algorithm in rotorpy/learning/ddpg.py, train a hover policy.

2. add the detail/tricks mentioned in the paper(learning to fly ...)
2.1 actor/critic training interval (done)
2.2 noise decay (done)
2.3 reward shaping: cirriculum learning and (reward recalculation) (done)
2.4 action history (done)
2.5 Guidance (done)
2.6 add path tracking task(done)
2.6 multi-processing (if time allowed)

3. (since path tracking is not good)transfer learning: using the trained policy for hover to
    keep training a path tracking policy
3.0 make sure to save a copy before training the trained hover policy
3.1 add random trajectory generation functions(line traj, square traj, circle traj, figure-8 traj, and more simple traj for training)
3.2 redefine the reward function for path tracking, maybe need to transform from flathness space to state space in order to get the quaternion, body rate...
3.3 redefine the observation in the env, add desire traj target, maybe need to add the future traj as observation
3.4 still training using TD3
3.5 see if the tracking performance can outperform PID

4. implement nonlinear/linear MPC using FORCESPRO
4.1 implement the nonlinear MPC for hovering(using the nonlinear quadrotor dynamics) (done)
4.2 implement the linear MPC for hovering(using the linearized simple quadrotor dynamics)
4.3 add path tracking task
4.4 adjust Q, R, horizon length parameters

5. compare the performance(hover/path tracking) with baseline RL\openai-basedline(PPO)\MPC\PID
5.1 compare the hover performance in/without wind env
5.2 compare the path tracking performance in/without wind env
5.3 report the reward/episode length curve for RL methods

6. report(MEAM517,ESE546), video, and poster(MEAM517)
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


class DDPG(object):
    def __init__(self, obs_dim, action_dim, discount=0.99, tau=0.005, max_ah_size=32):
        """
        obs_dim: observation dimension, p, v, q, w (13)
        action_dim: action dimension, 4 (motors speed)
        """
        self.actor = Actor(obs_dim+action_dim*max_ah_size, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.003)
        self.act_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=100)

        self.critic = Critic(obs_dim+action_dim, 1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.003)
        self.cri_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=100)

        self.discount = discount
        self.tau = tau
        self.min_noise = 0.01
        self.noise_std = 0.6
        self.decay_rate = 0.9
        self.act_tar_noise = 0.5
        self.clip = 0.5

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

    def train(self, batch_size, train_mode):
        """
        train_mode: 0 for critic only, 1 for both
        """
        obs, action, ah, next_obs, reward, done = self.get_batch(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.act_tar_noise
            ).clamp(-self.clip, self.clip)
            next_ah = torch.cat([ah[:, 4:], action], 1)
            next_act = self.actor_target(next_obs, next_ah) + noise
            target_Q1, target_Q2 = self.critic_target(next_obs, next_act)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - done) * self.discount * target_Q

        # Get current Q estimate
        current_Q1, current_Q2 = self.critic(obs, action)
        # Compute critic loss
        critic_loss = F.huber_loss(current_Q1, target_Q) + F.huber_loss(current_Q2, target_Q)

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