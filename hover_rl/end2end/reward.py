import numpy as np

class CurriculumReward(object):
    def __init__(self):
        # curriculum learning params
        self.C_POS_INIT = 2.5
        self.C_POS_TARGET = 20
        self.C_VEL_INIT = 0.005
        self.C_VEL_TARGET = 0.5
        self.C_ORIENT_INIT = 2.5
        self.C_ORIENT_TARGET = 2.5
        self.C_ANGVEL_INIT = 0.02
        self.C_ANGVEL_TARGET = 0.1
        self.C_ACTION_INIT = 0.005
        self.C_ACTION_TARGET = 0.05

        self.C_POS_RATE = 1.2
        self.C_VEL_RATE = 1.4
        self.C_ACTION_RATE = 1.4
        self.C_ANGVEL_RATE = 1.2

        # survival reward
        self.SURVIVE_REWARD = 1

        # goal reward
        self.GOAL_REWARD = 0

        self.rotor_speed_bs = np.array([0.1, 0.1, 0.1, 0.1])

    def curriculum_update(self):
        self.C_POS_INIT = min(self.C_POS_INIT * self.C_POS_RATE, self.C_POS_TARGET)
        self.C_VEL_INIT = min(self.C_VEL_INIT * self.C_VEL_RATE, self.C_VEL_TARGET)
        self.C_ACTION_INIT = min(self.C_ACTION_INIT * self.C_ACTION_RATE, self.C_ACTION_TARGET)
        self.C_ANGVEL_INIT = min(self.C_ANGVEL_INIT * self.C_ANGVEL_RATE, self.C_ANGVEL_TARGET)

    def hover_reward(self, observation, action, done):

        # Compute the distance to goal
        dist_reward = -self.C_POS_INIT*np.linalg.norm(observation[0:3])

        # Compute the velocity reward
        vel_reward = -self.C_VEL_INIT*np.linalg.norm(observation[3:6])

        #Compute the quaternion error
        quat_reward = -self.C_ORIENT_INIT*(1 - observation[9]**2)

        # Compute the angular rate reward
        ang_rate_reward = -self.C_ANGVEL_INIT*np.linalg.norm(observation[10:13])

        # Compute the action reward
        action_reward = -self.C_ACTION_INIT*np.linalg.norm(action - self.rotor_speed_bs)

        if (np.linalg.norm(observation[0:3]) < 0.05 and
                np.linalg.norm(observation[3:6]) < 0.05 and
                np.linalg.norm(observation[10:13]) < 0.05):
            self.GOAL_REWARD = 500
        else:
            self.GOAL_REWARD = 0

        if done:
            return (dist_reward + vel_reward + ang_rate_reward + action_reward +
                    self.GOAL_REWARD + quat_reward)

        return (dist_reward + vel_reward + ang_rate_reward + action_reward +
                self.SURVIVE_REWARD + self.GOAL_REWARD + quat_reward)