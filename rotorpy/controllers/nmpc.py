"""
Nonlinear MPC for quadrotor control using forcespro nonlinear solver
"""
import sys

import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
import matplotlib.patches
import casadi
import numpy as np
from scipy.spatial.transform import Rotation  # This is a useful library for working with attitude.
from rotorpy.vehicles.multirotor import quat_dot


class NonlinearMPC(object):
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
    def __init__(self, quad_params):
        """
        Use this constructor to save vehicle parameters, set controller gains, etc.
        Parameters:
            vehicle_params, dict with keys specified in a python file under /rotorpy/vehicles/
        """
        self.horizon = int(3)
        self.state_dim = int(13) # pos(3), vel(3), quat(4), ang_vel(3)
        self.control_dim = int(4) # rotor thrusts

        self.Q = casadi.diag([10,10,10,5,5,5,1,1,1,1,1,1,1])
        self.Qf = casadi.diag([20,20,20,10,10,10,1,1,1,1,1,1,1])
        self.R = casadi.diag([10,10,10,10])

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

        self.f_to_TM = np.vstack((np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]),
                                  (k * self.rotor_dir).reshape(1,-1)))
        self.f_to_TM = casadi.SX(self.f_to_TM.tolist())

        #### mpc problem ####
        self.generate_mpc_problem()
        self.x0 = np.zeros((self.model.nvar, self.model.N)) # initial guess

    ### have to use casadi
    def quadrotor_simple_dynamics(self, s, u):
        """
        sdot = f(s,u), continuous system dynamics, we see the dynamics in "multirotor.py"
        as an unknown, real ground-truth model in simulation, since it combines the aerodynamics and motor dynamics,
        however, in MPC, we ignore these affects, thus this function is called "simple", after solving
        the MPC problem, we will feed s0 and u0 into the ground-truth model to get the next true state
        at time t, observing the state s, applying u, return sdot
        full quadrotor (include internal) state:
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        however, we can only receive the following states as input:
        state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
        u: control input [4 thrusts]
        """
        v, w, q = s['v'], s['w'], s['q']
        R = self.casadi_quat2R(q)
        xd = v
        vd = casadi.vertcat([0,0,-self.g]) + casadi.mtimes(R, casadi.vertcat(0,0,u[0]/self.mass))
        qd = self.casadi_quat_dot(q, w)
        wd = casadi.mtimes(self.inv_inertia, (casadi.mtimes(self.f_to_TM, u) - casadi.cross(w, casadi.mtimes(self.inertia , w))))
        wd = casadi.mtimes(R, wd)
        return casadi.vertcat(xd, vd, qd, wd)


    def obj(self, x, u):
        """
        Objective function for the nonlinear MPC problem
        """
        return (casadi.mtimes(x.T, casadi.mtimes(self.Q, x)) +
                casadi.mtimes(u.T, casadi.mtimes(self.R, u)))


    def objN(self, x):
        """
        Objective function for the nonlinear MPC problem at the final time
        """
        return casadi.mtimes(x.T, casadi.mtimes(self.Qf, x))


    def generate_mpc_problem(self):
        # Model Definition
        # ----------------
        # Problem dimensions
        # the whole var is z = [f1, f2, f3, f4, x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
        self.model = forcespro.nlp.SymbolicModel()
        self.model.N = self.horizon  # horizon length
        self.model.nvar = self.state_dim + self.control_dim  # number of variables 13+4
        self.model.neq = 13  # number of equality constraints
        self.model.npar = 0 # number of runtime parameters, the param that is changing with time, or external input of mpc, like target positions

        # Objective function
        self.model.objective = lambda z: self.obj(z[self.state_dim:], z[:self.state_dim])
        self.model.objectiveN = lambda z: self.objN(z[self.state_dim:]) # increased costs for the last stage
        # The function must be able to handle symbolic evaluation,
        # by passing in CasADi symbols. This means certain numpy funcions are not
        # available.

        # We use an explicit RK4 integrator here to discretize continuous dynamics
        integrator_stepsize = 0.1
        self.model.eq = lambda z: forcespro.nlp.integrate(self.quadrotor_simple_dynamics,
                                                     z[self.state_dim:], z[:self.state_dim],
                                                     integrator=forcespro.nlp.integrators.RK4,
                                                     stepsize=integrator_stepsize)
        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        self.model.E = np.concatenate([np.zeros((13,4)), np.eye(13)], axis=1) # mask

        # Inequality constraints
        #  upper/lower variable bounds lb <= z <= ub
        #z = [f1, f2, f3, f4, x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
        #                     inputs                 |  states
        #                     F          phi            x    y     v    theta         delta
        self.model.lb = np.array([-5, -5, -5, -5, -6, -6, -6, -3, -3, -3, 0, 0, 0, 0, -1, -1, -1])
        self.model.ub = np.array([5, 5, 5, 5, 6, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1])

        # Initial condition on vehicle states x
        self.model.xinitidx = range(4,4+13) # use this to specify on which variables initial conditions, # mask
        # are imposed
        self.model.bfgs_init = 2.5*np.identity(4+13) # initialization of the
        # hessian approximation

        # Solver generation
        # -----------------

        # Set solver options
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 200     # Maximum number of iterations
        codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but
        #                             not for timings)
        codeoptions.optlevel = 0    # 0 no optimization, 1 optimize for size,
        #                             2 optimize for speed, 3 optimize for size & speed
        codeoptions.cleanup = False
        codeoptions.timing = 1
        codeoptions.nlp.hessian_approximation = 'bfgs'
        codeoptions.solvemethod = 'SQP_NLP' # choose the solver method Sequential
        #                              Quadratic Programming
        codeoptions.sqp_nlp.maxqps = 1      # maximum number of quadratic problems to be solved
        codeoptions.sqp_nlp.reg_hessian = 5e-9 # increase this if exitflag=-8
        # change this to your server or leave uncommented for using the
        # standard embotech server at https://forces.embotech.com
        # codeoptions.server = 'https://forces.embotech.com'

        # Creates code for symbolic model formulation given above, then contacts
        # server to generate new solver
        self.solver = self.model.generate_solver(options=codeoptions)

        return self.model, self.solver



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
        if t == 0:
            x0i = np.zeros((self.model.nvar,1))
            x0i[4:] = np.copy(np.array([state['x'], state['v'], state['q'], state['w']]).reshape(-1,))
            x0 = np.transpose(np.tile(x0i, (1, self.model.N)))
        else:
            x0 = np.copy(self.x0)

        xinit = np.transpose(np.array([state['x'], state['v'], state['q'], state['w']]).reshape(-1,))
        problem = {"x0": x0,
                   "xinit": xinit}
        output, exitflag, info = self.solver.solve(problem)
        assert exitflag == 1, "bad exitflag"
        sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n" \
                         .format(info.it, info.solvetime))

        temp = np.zeros((np.max(self.model.nvar), self.model.N))
        for i in range(self.model.N):
            temp[:, i] = output['x{0:02d}'.format(i+1)]
        pred_u = temp[0:4, :]

        self.x0 = np.copy(temp)

        cmd_motor_thrusts = pred_u[:,0]

        control_input = {'cmd_motor_speeds' :cmd_motor_speeds,
                         'cmd_motor_thrusts':cmd_motor_thrusts,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_w':cmd_w,
                         'cmd_v':cmd_v}
        return control_input


    #################### helper functions ####################
    def casadi_quat2R(self, quat):
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        R = casadi.vertcat(
      casadi.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)),
            casadi.horzcat(2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)),
            casadi.horzcat(2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2))
        )
        return R

    def casadi_quat_dot(self, quat, omega):
        """
        Parameters:
            quat, [i,j,k,w]
            omega, angular velocity of body in body axes

        Returns
            duat_dot, [i,j,k,w]

        """
        # Adapted from "Quaternions And Dynamics" by Basile Graf.
        (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
        G = casadi.SX([
            [ q3,  q2, -q1, -q0],
            [-q2,  q3,  q0, -q1],
            [ q1, -q0,  q3, -q2]
        ])
        quat_dot = 0.5 * casadi.mtimes(G.T, omega)
        # Augment to maintain unit quaternion.
        quat_err = casadi.sumsqr(quat**2) - 1
        quat_err_grad = 2 * quat
        quat_dot = quat_dot - quat_err * quat_err_grad
        return quat_dot
