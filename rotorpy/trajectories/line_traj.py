import numpy as np
import sys

###### Line trajectory, level : easy ####

class Line3DTraj(object):

    def __init__(self, start=np.array([0,0,0]),
                       end=np.array([3,5,-2]),
                       T = 2.0,
                       yaw_bool=False):
        """
        This is the constructor for the Trajectory object.
        3D line trajectory, go to hover if finish, x(t) = x0 + const_v*t,x(T) = hover


        Inputs:
            start, the start point of the line
            end, the end point of the line
            max speed, max speed travel along the line
            T, one cycle time
        """

        self.start = start
        self.end = end
        self.yaw_bool = yaw_bool
        self.T = T
        self.dist = np.linalg.norm(end - start)
        self.v = self.dist / self.T
        self.dir = (end - start) / self.dist

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
        if t < self.T-0.1:
            x = self.start + self.v * t * self.dir
            x_dot = self.v * self.dir
            x_ddot = np.zeros(3)
            x_dddot = np.zeros(3)
            x_ddddot = np.zeros(3)

        else:
            x = self.end
            x_dot = np.zeros(3)
            x_ddot = np.zeros(3)
            x_dddot = np.zeros(3)
            x_ddddot = np.zeros(3)

        if self.yaw_bool:
            yaw = 0.8*np.pi/2*np.sin(2.5*t)
            yaw_dot = 0.8*2.5*np.pi/2*np.cos(2.5*t)
        else:
            yaw = 0
            yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output