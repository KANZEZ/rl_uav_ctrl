"""
define all the constants variables here
"""
import numpy as np

#####################        NETWORK PARAMETERS        #####################
DISCOUNT_FACTOR = 0.99
TAU = 0.005
LEARNING_RATE = 0.003
EXPL_NOISE_DECAY_RATE = 0.9
MIN_EXPL_NOISE_STD = 0.01
ACTOR_TAR_NOISE = 0.5
NOISE_CLIP = 0.5
NOISE_STD = 0.5


#####################        TRAINING PARAMETERS        #####################
SEED = 47
ACTION_DIM = 4
ACTION_HISTORY_HORIZON = 32
OBS_DIM = 19
WIND_DIM = 3
FUTURE_TRAJ_HORIZON = 20
REPLAY_BUFFER_SIZE = int(3e5)
POS_BOUND = 4.5
VEL_BOUND = 2.0
WIND_LOWER_BOUND = np.array([1, 1, 1])
WIND_UPPER_BOUND = np.array([-1, -1, -1])