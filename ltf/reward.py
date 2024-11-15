import numpy as np
from scipy.spatial.transform import Rotation
import gymnasium as gym
from gymnasium import spaces
import math

class CurriculumReward(object):
    def __init__(self):
        # curriculum learning params
        self.C_POS_INIT = 2.5
        self.C_POS_TARGET = 20
        self.C_VEL_INIT = 0.005
        self.C_VEL_TARGET = 0.5
        self.C_ORIENT_INIT = 2.5
        self.C_ORIENT_TARGET = 2.5
        self.C_ANGVEL_INIT = 0
        self.C_ANGVEL_TARGET = 0
        self.C_ACTION_INIT = 0.005
        self.C_ACTION_TARGET = 0.5

        self.C_POS_RATE = 1.2
        self.C_VEL_RATE = 1.4
        self.C_ACTION_RATE = 1.4

        # survival reward
        self.SURVIVE_REWARD = 2

    def curriculum_update(self):
        self.C_POS_INIT = min(self.C_POS_INIT * self.C_POS_RATE, self.C_POS_TARGET)
        self.C_VEL_INIT = min(self.C_VEL_INIT * self.C_VEL_RATE, self.C_VEL_TARGET)
        self.C_ACTION_INIT = min(self.C_ACTION_INIT * self.C_ACTION_RATE, self.C_ACTION_TARGET)

    def reward(self, observation, action, done):
        """
        Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and
        action reward.
        """
        # Compute the distance to goal
        dist_reward = -self.C_POS_INIT*np.linalg.norm(observation[0:3])

        # Compute the velocity reward
        vel_reward = -self.C_VEL_INIT*np.linalg.norm(observation[3:6])

        #Compute the quaternion error, dont use it for now
        quat_reward = -self.C_ORIENT_INIT*np.linalg.norm(observation[6:10])

        # Compute the angular rate reward, note that we set it to zero
        ang_rate_reward = -self.C_ANGVEL_INIT*np.linalg.norm(observation[10:13])

        # Compute the action reward
        action_reward = -self.C_ACTION_INIT*np.linalg.norm(action)

        if done:
            return dist_reward + vel_reward + ang_rate_reward + action_reward

        return dist_reward + vel_reward + ang_rate_reward + action_reward + self.SURVIVE_REWARD