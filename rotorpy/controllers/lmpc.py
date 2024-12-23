"""
Linear MPC for quadrotor control using forcespro solver
Somehow the forcespro solver is not working properly, the problem may lie in the dynamics linear model
or the solver itself........
"""
import sys

import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
import matplotlib.patches
import casadi
import numpy as np
from scipy.spatial.transform import Rotation  # This is a useful library for working with attitude.



class LinearMPC(object):
    """
    The controller is implemented as a class with two required methods: __init__() and update().
    The __init__() is used to instantiate the controller, and this is where any model parameters or
    controller gains should be set.
    In update(), the current time, state, and desired flat outputs are passed into the controller at
    each simulation step. The output of the controller depends on the control abstraction for Multirotor...
        if cmd_motor_speeds: the output dict should contain the key 'cmd_motor_speeds'
        if cmd_motor_thrusts: the output dict should contain the key 'cmd_rotor_thrusts'
        if cmd_vel: the output dict should contain the key 'cmd_v'
        if cmd_ctatt: the output dict should contain the keys 'cmd_thrust' and 'cmd_q'
        if cmd_ctbr: the output dict should contain the keys 'cmd_thrust' and 'cmd_w'
        if cmd_ctbm: the output dict should contain the keys 'cmd_thrust' and 'cmd_moment'
    """
    def __init__(self, quad_params, horizon):
        """
        Use this constructor to save vehicle parameters, set controller gains, etc.
        Parameters:
            vehicle_params, dict with keys specified in a python file under /rotorpy/vehicles/
        """
        self.horizon = int(horizon)
        self.state_dim = int(12) # pos(3), vel(3), rpy(3), ang_vel(3)
        self.control_dim = int(4) # rotor collective thrust and 3 moments

        self.g = 9.81

        # Inertial parameters
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.Ixy             = quad_params['Ixy']  # kg*m^2
        self.Ixz             = quad_params['Ixz']  # kg*m^2
        self.Iyz             = quad_params['Iyz']  # kg*m^2

        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])
        self.inertia = casadi.SX(self.inertia.tolist())
        self.inv_inertia = casadi.inv(self.inertia)
        self.weight = casadi.SX(np.array([0, 0, -self.mass*self.g]).tolist())

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']
        self.rotor_dir       = quad_params['rotor_directions']

        # Rotor parameters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        self.k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        self.k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        self.k_d             = quad_params['k_d']       # rotor drag coeff, N/(m/s)
        self.k_z             = quad_params['k_z']       # induced inflow coeff N/(m/s)
        self.k_flap          = quad_params['k_flap']    # Flapping moment coefficient Nm/(m/s)

        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient.

        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                  np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]),
                                  (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        #### mpc problem ####
        self.model, self.solver = self.generate_mpc_problem()
        self.xinit = np.zeros(self.state_dim) # initial condition


    def get_equilibrium_thrust(self):
        return np.full((4,), fill_value=self.mass*self.g/self.num_rotors)


    ################## need to check the dynamics ##################
    def quadrotor_simple_dynamics(self, s, u):
        """
        xdot = Ax + Bu
        state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, rpy, rad
                w, angular velocity, rad/s
        u: control input [1 thrusts, 3 moments]
        """
        x, v, q, w = s[:3], s[3:6], s[6:9], s[9:12]
        F, L, M, N = u[0], u[1], u[2], u[3]
        xd = v
        vd = casadi.vertcat(-self.g * q[1], self.g * q[0], F/self.mass)
        qd = w
        wd = casadi.vertcat(L/self.Ixx, M/self.Iyy, N/self.Izz)

        return casadi.vertcat(xd, vd, qd, wd)


    def obj(self, x, u):
        """
        Objective function for the nonlinear MPC problem
        """
        #return casadi.mtimes(x.T, casadi.mtimes(self.Q, x)) + casadi.mtimes(u.T, casadi.mtimes(self.R, u))
        return 10 * casadi.sumsqr(x[:3]) + 10 * casadi.sumsqr(x[3:6])# + 1 * casadi.sumsqr(u)

    def objN(self, x):
        """
        Objective function for the nonlinear MPC problem at the final time
        """
        return 10 * casadi.sumsqr(x[:3]) + 10 * casadi.sumsqr(x[3:6])


    def generate_mpc_problem(self):
        model = forcespro.nlp.ConvexSymbolicModel(self.horizon)
        model.nvar = self.state_dim + self.control_dim    # number of variables
        model.neq = self.state_dim   # number of equality constraints

        model.objective = lambda z: self.obj(z[self.control_dim:], z[:self.control_dim])
        model.objectiveN = lambda z: self.objN(z[self.control_dim:])

        integrator_stepsize = 0.001
        model.eq = lambda z: forcespro.nlp.integrate(self.quadrotor_simple_dynamics,
                                                     z[self.control_dim:], z[:self.control_dim],
                                                     integrator=forcespro.nlp.integrators.RK4,
                                                     stepsize=integrator_stepsize)
        #model.eq = lambda z: self.quadrotor_simple_dynamics(z[self.control_dim:], z[:self.control_dim])

        model.E = np.concatenate([np.zeros((self.state_dim,self.control_dim)), np.eye(self.state_dim)], axis=1) # mask

        model.lb = np.array([-2, -2, -2, -2, -6, -6, -6, -3, -3, -3, -np.pi, -np.pi/2, -np.pi,  -2, -2, -2])
        model.ub = np.array([2,   2,  2,  2,  6,  6,  6,  3,  3,  3,  np.pi,  np.pi/2,  np.pi,   2,  2,  2])

        # Initial condition on vehicle states x
        model.xinitidx = range(self.control_dim,self.state_dim+self.control_dim)

        # Generate FORCESPRO solver
        # -------------------------

        # set options
        options = forcespro.CodeOptions()
        options.use_default('FORCESNLPsolver', 'PDIP_NLP')
        options.printlevel = 0

        # generate code
        solver = model.generate_solver(options)

        return model, solver


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys

                key, description, unit, (applicable control abstraction)
                cmd_motor_speeds, the commanded speed for each motor, rad/s, (cmd_motor_speeds)
                cmd_thrust, the collective thrust of all rotors, N, (cmd_ctatt, cmd_ctbr, cmd_ctbm)
                cmd_moment, the control moments on each boxy axis, N*m, (cmd_ctbm)
                cmd_q, desired attitude as a quaternion [i,j,k,w], , (cmd_ctatt)
                cmd_w, desired angular rates in body frame, rad/s, (cmd_ctbr)
                cmd_v, desired velocity vector in world frame, m/s (cmd_vel)
        """

        # Only some of these are necessary depending on your desired control abstraction.
        cmd_motor_speeds = np.zeros((4,))
        cmd_motor_thrusts = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.array([0,0,0,1])
        cmd_w = np.zeros((3,))
        cmd_v = np.zeros((3,))

        ###### solve the mpc problem in this time step ######
        sx = np.ravel(state['x'])
        sv = np.ravel(state['v'])
        sq = np.ravel(state['q'])
        sw = np.ravel(state['w'])
        sq = Rotation.from_quat(sq).as_euler('xyz')
        state_vec = np.concatenate([sx, sv, sq, sw]).reshape(-1, )

        self.xinit = np.transpose(state_vec)
        problem = {"xinit": self.xinit}
        print(self.xinit)

        output, exitflag, info = self.solver.solve(problem)

        print(exitflag)
        #assert exitflag == 1, "bad exitflag"
        sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n" \
                         .format(info.it, info.solvetime))

        temp = np.zeros((np.max(self.model.nvar), self.model.N))
        for i in range(self.model.N):
            temp[:, i] = output['x{0:02d}'.format(i+1)]
        pred_u = temp[0:4, :]
        u0 = pred_u[:,0]

        print("solved u", u0)
        TM = np.array([u0[0], u0[1], u0[2], u0[3]])
        cmd_rotor_thrusts = self.TM_to_f @ TM + self.get_equilibrium_thrust()
        cmd_motor_speeds = cmd_rotor_thrusts / self.k_eta
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))
        print("total speed", cmd_motor_speeds)

        control_input = {'cmd_motor_speeds' :cmd_motor_speeds,
                         'cmd_motor_thrusts':cmd_motor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        return control_input
