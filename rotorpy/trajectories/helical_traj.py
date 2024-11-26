import numpy as np
import sys

###### Helical trajectory, level : hard ####

class Helical3DTrajectory(object):

    def __init__(self, w=2, A=1, B=1, vz=0.8, z0=3,
                 yaw_bool=False):

        # This is the constructor for the Trajectory object.
        # x = Acos(wt)
        # y = Bsin(wt)
        # z = vzt + z0

        self.yaw_bool = yaw_bool

        self.A = A
        self.B = B
        self.w = w
        self.vz = vz
        self.z0 = z0
        self.zmax = 5.5
        self.zmin = -5.5


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
        #### limit the height of the helix
        if self.zmin <= self.vz*t + self.z0 <= self.zmax:
            z = self.vz*t + self.z0
            vz = self.vz
        else:
            z = min(max(self.zmin, self.vz*t + self.z0), self.zmax)
            vz = 0

        x = np.array([self.A*np.cos(self.w*t),
                      self.B*np.sin(self.w*t),
                      z])

        x_dot = np.array([-self.A*self.w*np.sin(self.w*t),
                            self.B*self.w*np.cos(self.w*t),
                            vz])

        x_ddot = np.array([-self.A*self.w**2*np.cos(self.w*t),
                            -self.B*self.w**2*np.sin(self.w*t),
                            0])

        x_dddot = np.array([self.A*self.w**3*np.sin(self.w*t),
                            -self.B*self.w**3*np.cos(self.w*t),
                            0])

        x_ddddot = np.array([self.A*self.w**4*np.cos(self.w*t),
                            self.B*self.w**4*np.sin(self.w*t),
                            0])

        if self.yaw_bool:
            yaw = 0.8*np.pi/2*np.sin(2.5*t)
            yaw_dot = 0.8*2.5*np.pi/2*np.cos(2.5*t)
        else:
            yaw = 0
            yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output