import numpy as np
import sys


#### Sin trajectory, level : hard ####

class Sin3DTraj(object):

    def __init__(self, start=np.array([0,0,0]),
                 A=np.array([2,5,-2]),
                 T = np.array([6,6,6]),
                 yaw_bool=False):
        """
        This is the constructor for the Trajectory object.
        3D Sin trajectory, from 0m/s to max speed and then decel to 0m/s.
        For continuous trajectory，the position trajectory is a cosine function with respect to time
        x(t) = A0*sin(w0t + phi0) + x0
        y(t) = A1*sin(w1t + phi1) + y0
        z(t) = A2*sin(w2t + phi2) + z0


        Inputs:
            start, the start point of the line
            end, the end point of the line
            max speed, max speed travel along the line
            T, one cycle time
        """

        self.start = start
        self.A = A
        self.C = start
        self.T = T
        self.freq = 2 * np.pi / self.T

        self.max_speed = self.freq * self.A

        self.yaw_bool = yaw_bool

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

        x = np.array([self.A[0] * np.sin(self.freq[0] * t) + self.C[0],
                      self.A[1] * np.sin(self.freq[1] * t) + self.C[0],
                      self.A[2] * np.sin(self.freq[2] * t) + self.C[0]])

        # always start from 0 and end at 0
        x_dot = np.array([self.A[0]*self.freq[0] * np.cos(self.freq[0] * t),
                            self.A[1]*self.freq[1] * np.cos(self.freq[1] * t),
                            self.A[2]*self.freq[2] * np.cos(self.freq[2] * t)])

        x_ddot = np.array([-self.A[0]*self.freq[0]**2 * np.sin(self.freq[0] * t),
                            -self.A[1]*self.freq[1]**2 * np.sin(self.freq[1] * t),
                            -self.A[2]*self.freq[2]**2 * np.sin(self.freq[2] * t)])

        x_dddot = np.array([-self.A[0]*self.freq[0]**3 * np.cos(self.freq[0] * t),
                            -self.A[1]*self.freq[1]**3 * np.cos(self.freq[1] * t),
                            -self.A[2]*self.freq[2]**3 * np.cos(self.freq[2] * t)])

        x_ddddot = np.array([self.A[0]*self.freq[0]**4 * np.sin(self.freq[0] * t),
                            self.A[1]*self.freq[1]**4 * np.sin(self.freq[1] * t),
                            self.A[2]*self.freq[2]**4 * np.sin(self.freq[2] * t)])


        if self.yaw_bool:
            yaw = 0.8*np.pi/2*np.sin(2.5*t)
            yaw_dot = 0.8*2.5*np.pi/2*np.cos(2.5*t)
        else:
            yaw = 0
            yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output