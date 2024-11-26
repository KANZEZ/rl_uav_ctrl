import numpy as np
import sys
from rotorpy.trajectories.polynomial_traj import Polynomial

###### Polysegment trajectory, level : hard ####

class Rectangle2DTrajectory(object):

    def __init__(self, center=np.array([0,0,0]), width=3, length=5,
                 N=4,
                 v_avg=1.5,
                 yaw_bool=False):

        # This is the constructor for the Trajectory object.
        # get polynomial traj segment, for example, from point a to b, then stop, and
        # then from b to c, then stop, and so on.

        self.yaw_bool = yaw_bool

        self.points = np.zeros((N+1, 3))
        self.points[0,:] = center + np.array([length/2, width/2, 0])
        self.points[1,:] = center + np.array([length/2, -width/2, 0])
        self.points[2,:] = center + np.array([-length/2, -width/2, 0])
        self.points[3,:] = center + np.array([-length/2, width/2, 0])
        self.points[4,:] = center + np.array([length/2, width/2, 0])

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