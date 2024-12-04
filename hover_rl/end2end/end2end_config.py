"""
define all the constants variables here
"""

#####################        NETWORK PARAMETERS        #####################
DISCOUNT_FACTOR = 0.99
TAU = 0.005
LEARNING_RATE = 0.003
EXPL_NOISE_DECAY_RATE = 0.9
MIN_EXPL_NOISE_STD = 0.01
ACTOR_TAR_NOISE = 0.5
NOISE_CLIP = 0.5
NOISE_STD = 0.6
BATCH_SIZE=256


#####################        TRAINING PARAMETERS        #####################
SEED = 47
ACTION_DIM = 4
ACTION_HISTORY_HORIZON = 32
OBS_DIM = 13
WIND_DIM = 3
REPLAY_BUFFER_SIZE = int(3e5)
POS_BOUND = 4.5
VEL_BOUND = 2.0
WIND_LOWER_BOUND = -1
WIND_UPPER_BOUND = 1
EVAL_EPISODE = 10
EVAL_FREQ = 10000

############## TRAINING CONST ###########
MAX_ITER = int(700000)  ### about 35 min

###### warmup
ACTOR_START = 30000
CRITIC_START = 15000
ACTOR_TRAIN_INTERVAL = 20
CRITIC_TRAIN_INTERVAL = 2

#### exploration noise ####
EXPL_NOISE_DECAY_START = 250000
EXPL_NOISE_DECAY_INTERVAL = 100000

#### guidance : start from 0,0,0 ####
GUIDANCE_PROB = 0.1

#### cirriculum learning ####
REWARD_UPDATE_INTERVAL = 100000


##########################  ENV PARAMETERS ########################
ENV_NAME = "Quadrotor-v0"
MAX_SIM_TIME = 5
SIM_RATE = 100
RENDER_MODE = 'None'
CONTROL_MODE = 'cmd_motor_speeds'