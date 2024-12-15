### End to end UAV hover controller using deep reinforcement learning algorithm in wind disturbed environment

### Structure
All the source code is in folder "hover_rl/end2end"

#### Env
We use [RotorPy](https://github.com/spencerfolk/rotorpy), a Python-based multirotor simulation environment with aerodynamic wrenches, useful for education and research in estimation, planning, and control for UAVs.

#### Install
Please follow the install requirement in [RotorPy](https://github.com/spencerfolk/rotorpy)

### To run the training:
```
python3 hover_train.py
```

### To run the testing:
```
python3 hover_test.py
```


#### future work:
1. train an end to end controller for path tracking instead of only hovering: on-going, code is in folder "hover_rl/path_trk"
2. train an controller combine with RL residual learning and nomial geometry(SE3) controller: on-going, code is in folder "hover_rl/res_rl"
