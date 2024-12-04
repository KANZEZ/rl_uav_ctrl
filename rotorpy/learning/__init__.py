from gymnasium.envs.registration import register

### Register the Quadrotor-v0 environment
register(
     id="Quadrotor-v0",
     entry_point="rotorpy.learning.quadrotor_environments:QuadrotorEnv",
)

# ### Register the Quadrotor-v1 environment
# register(
#      id="Quadrotor-v1",
#      entry_point="rotorpy.learning.ddpg_pathtrack_env:QuadrotorTrackingEnv",
# )
#
#
# ### Register the Quadrotor-v2 environment
# register(
#      id="Quadrotor-v2",
#      entry_point="rotorpy.learning.quad_rlpid_env:QuadRlPidEnv",
# )