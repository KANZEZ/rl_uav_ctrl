import numpy as np

from rotorpy.controllers.quadrotor_control import SE3Control


class RlPidCurriculumReward(object):
    def __init__(self):
        # curriculum learning params
        self.C_POS_INIT = 2.5
        self.C_POS_TARGET = 20
        self.C_VEL_INIT = 0.005
        self.C_VEL_TARGET = 0.5
        self.C_ORIENT_INIT = 2.5
        self.C_ORIENT_TARGET = 2.5
        self.C_ANGVEL_INIT = 0.01
        self.C_ANGVEL_TARGET = 0.1
        self.C_ACTION_INIT = 0.005
        self.C_ACTION_TARGET = 0.05
        self.C_POS_ERR_INIT = 0.3
        self.C_POS_ERR_TARGET = 0.2
        self.C_VEL_ERR_INIT = 0.1
        self.C_VEL_ERR_TARGET = 0.05


        self.C_POS_RATE = 1.2
        self.C_VEL_RATE = 1.4
        self.C_ACTION_RATE = 1.4
        self.C_ANGVEL_RATE = 1.2
        self.C_POS_ERR_RATE = 0.85
        self.C_VEL_ERR_RATE = 0.85


        # survival reward
        self.SURVIVE_REWARD = 2

        # goal reward
        self.GOAL_REWARD = 0

        self.rotor_speed_bs = np.array([0.1, 0.1, 0.1, 0.1])

    def curriculum_update(self):
        self.C_POS_INIT = min(self.C_POS_INIT * self.C_POS_RATE, self.C_POS_TARGET)
        self.C_VEL_INIT = min(self.C_VEL_INIT * self.C_VEL_RATE, self.C_VEL_TARGET)
        self.C_ACTION_INIT = min(self.C_ACTION_INIT * self.C_ACTION_RATE, self.C_ACTION_TARGET)
        self.C_ANGVEL_INIT = min(self.C_ANGVEL_INIT * self.C_ANGVEL_RATE, self.C_ANGVEL_TARGET)
        self.C_POS_ERR_INIT = max(self.C_POS_ERR_INIT * self.C_POS_ERR_RATE, self.C_POS_ERR_TARGET)
        self.C_VEL_ERR_INIT = max(self.C_VEL_ERR_INIT * self.C_VEL_ERR_RATE, self.C_VEL_ERR_TARGET)


    def res_reward(self, observation, action, done):
        """
        Rewards tracking a trajectory. It is a combination of position error, velocity error, body rates, and
        action reward.
        observation: np.array, observation from the environment:
        #     position, x, observation_state[0:3]
        #     velocity, v, observation_state[3:6]
        #     orientation, q, observation_state[6:10]
        #     body rates, w, observation_state[10:13]
        #     pos_error, observation_state[13:16]
        #     vel_error, observation_state[16:19]
        """

        dist_reward = -self.C_POS_INIT*np.linalg.norm(observation[13:16])

        # Compute the velocity reward
        vel_reward = -self.C_VEL_INIT*np.linalg.norm(observation[16:19])

        #Compute the quaternion error
        quat_reward = -self.C_ORIENT_INIT*(1 - observation[9]**2)

        # Compute the angular rate reward
        ang_rate_reward = -self.C_ANGVEL_INIT*np.linalg.norm(observation[10:13])

        # Compute the action reward
        action_reward = -self.C_ACTION_INIT*np.linalg.norm(action)

        if (np.linalg.norm(observation[13:16]) < self.C_POS_ERR_INIT
                and np.linalg.norm(observation[16:19]) < self.C_VEL_ERR_INIT):
            self.GOAL_REWARD = 2
        else:
            self.GOAL_REWARD = 0

        if done:

            if (np.linalg.norm(observation[13:16]) < self.C_POS_ERR_INIT
                    and np.linalg.norm(observation[16:19]) < self.C_VEL_ERR_INIT):
                last_step = 1000
            else:
                last_step = 0

            return (dist_reward + vel_reward + ang_rate_reward + action_reward +
                    self.GOAL_REWARD + quat_reward + last_step)

        return (dist_reward + vel_reward + ang_rate_reward + action_reward +
                self.SURVIVE_REWARD + self.GOAL_REWARD + quat_reward)