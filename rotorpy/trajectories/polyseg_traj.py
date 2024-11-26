import numpy as np
import sys
from rotorpy.trajectories.polynomial_traj import Polynomial

###### Polysegment trajectory, level : hard ####

class Polyseg3DTrajectory(object):

    def __init__(self, lb=np.array([-2,-3,0]),
                       ub=np.array([2,3,6]),
                       N=5,
                       v_avg=1.5,
                       yaw_bool=False):

        # This is the constructor for the Trajectory object.
        # get polynomial traj segment, for example, from point a to b, then stop, and
        # then from b to c, then stop, and so on.

        self.yaw_bool = yaw_bool

        self.points = np.zeros((N, 3))
        for i in range(3):
            self.points[:,i] = np.random.uniform(lb[i], ub[i], N)

        self.poly_gen = Polynomial(self.points, v_avg=v_avg)


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        flat_output = self.poly_gen.update(t)
        return flat_output