import numpy as np
import sys

################ level: mid ################

class BnF3DTraj(object):

    def __init__(self, start=np.array([-2,-2,-2]),
                 end=np.array([2,2,-2]),
                 T = 3.0,
                 yaw_bool=False):
        """
        This is the constructor for the Trajectory object.
        3D line trajectory that moves back and forth, from 0m/s to max speed and then decel to 0m/s.
        For continuous trajectory，the position trajectory is a cosine function with respect to time
        x(t) = -A*cos(wt) + C


        Inputs:
            start, the start point of the line
            end, the end point of the line
            max speed, max speed travel along the line
            T, one cycle time
        """

        self.start = start
        self.end = end
        self.A = (end - start) / 2
        self.C = (end + start) / 2
        self.yaw_bool = yaw_bool
        self.T = T
        self.freq = 2 * np.pi / self.T

        self.max_speed = self.freq * self.A

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

        x = np.array([-self.A[0] * np.cos(self.freq * t) + self.C[0],
                      -self.A[1] * np.cos(self.freq * t) + self.C[0],
                      -self.A[2] * np.cos(self.freq * t) + self.C[0]])

        # always start from 0 and end at 0
        x_dot = np.array([self.A[0]*self.freq * np.sin(self.freq * t),
                          self.A[1]*self.freq * np.sin(self.freq * t),
                          self.A[2]*self.freq * np.sin(self.freq * t)])

        x_ddot = np.array([self.A[0]*self.freq**2 * np.cos(self.freq * t),
                           self.A[1]*self.freq**2 * np.cos(self.freq * t),
                           self.A[2]*self.freq**2 * np.cos(self.freq * t)])

        x_dddot = np.array([-self.A[0]*self.freq**3 * np.sin(self.freq * t),
                            -self.A[1]*self.freq**3 * np.sin(self.freq * t),
                            -self.A[2]*self.freq**3 * np.sin(self.freq * t)])

        x_ddddot = np.array([-self.A[0]*self.freq**4 * np.cos(self.freq * t),
                             -self.A[1]*self.freq**4 * np.cos(self.freq * t),
                             -self.A[2]*self.freq**4 * np.cos(self.freq * t)])


        if self.yaw_bool:
            yaw = 0.8*np.pi/2*np.sin(2.5*t)
            yaw_dot = 0.8*2.5*np.pi/2*np.cos(2.5*t)
        else:
            yaw = 0
            yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output