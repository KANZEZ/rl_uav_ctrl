import forcespro
import numpy as np

# Simple MPC - double integrator example solved with FORCESPRO
#              using a convex interior-point algorithm
#              and state-elimination (a.k.a. condensing)
#
#  min   xN'*P*xN + sum_{i=1}^{N-1} xi'*Q*xi + ui'*R*ui
# xi,ui
#       s.t. x1 = x
#            x_i+1 = A*xi + B*ui  for i = 1...N-1
#            xmin <= xi <= xmax   for i = 1...N
#            umin <= ui <= umax   for i = 1...N
#
# and P is solution of Ricatti eqn. from LQR problem
#
# (c) embotech AG, Zurich, Switzerland, 2014-2023.

# system
A = np.array([[1.1, 1], [0, 1]])
B = np.array([[1], [0.5]])

nx = 2
nu = 1

# MPC setup
N = 5
Q = np.eye(nx)
R = np.eye(nu)
# terminal weight obtained from discrete-time Riccati equation
P = np.array([[2.02390049, 0.269454848], [0.269454848, 2.652909941]])
umin = np.array([-0.5])
umax = np.array([0.5])
xmin = np.array([-5, -5])
xmax = np.array([5, 5])

# FORCESPRO multistage form
# note: in order to be compatible with a condensing-based solver,
#       variable ordering has to be z[i-1] = [u_i, x_i] for i=1...N
stages = forcespro.MultistageProblem(N)
for i in range(N):

    # dimensions
    stages.dims[i]['n'] = nx + nu  # number of stage variables
    stages.dims[i]['r'] = nx       # number of equality constraints
    stages.dims[i]['l'] = nx + nu  # number of lower bounds
    stages.dims[i]['u'] = nx + nu  # number of upper bounds

    # cost
    if (i == N-1):
        stages.cost[i]['H'] = np.vstack((np.hstack((R, np.zeros((nu, nx)))), np.hstack((np.zeros((nx, nu)), P))))
    else:
        stages.cost[i]['H'] = np.vstack((np.hstack((R, np.zeros((nu, nx)))), np.hstack((np.zeros((nx, nu)), Q))))
    stages.cost[i]['f'] = np.zeros((nx + nu, 1))

    # lower bounds
    stages.ineq[i]['b']['lbidx'] = list(range(1, nu + nx + 1))   # lower bound acts on these indices
    stages.ineq[i]['b']['lb'] = np.concatenate((umin, xmin), 0)  # lower bound for this stage variable

    # upper bounds
    stages.ineq[i]['b']['ubidx'] = list(range(1, nu + nx + 1))   # upper bound acts on these indices
    stages.ineq[i]['b']['ub'] = np.concatenate((umax, xmax), 0)  # upper bound for this stage variable

    # equality constraints
    if (i < N-1):
        stages.eq[i]['C'] = np.hstack((B, A))
    if (i > 0):
        stages.eq[i]['c'] = np.zeros((nx, 1))
    stages.eq[i]['D'] = np.hstack((np.zeros((nx, nu)), -np.eye(nx)))

stages.newParam('minus_x0', [1], 'eq.c')  # RHS of first eq. constr. is a parameter: c1 = -x0

# define output of the solver
stages.newOutput('u0', 1, list(range(1, nu + 1)))

# solver settings
stages.codeoptions['name'] = 'myMPC_FORCESPRO'
stages.codeoptions['printlevel'] = 0
stages.codeoptions['condense'] = 1 # enable state-elimination

# generate code
import get_userid

stages.generateCode(get_userid.userid)

# simulate
import myMPC_FORCESPRO_py

problem = myMPC_FORCESPRO_py.myMPC_FORCESPRO_params
x1 = np.array([-4, 2])
kmax = 30
X = np.zeros((nx, kmax + 1))
X[:, 0] = x1
U = np.zeros((nu, kmax))
for k in range(0, kmax):
    problem['minus_x0'] = -X[:, k]
    [solverout, exitflag, info] = myMPC_FORCESPRO_py.myMPC_FORCESPRO_solve(problem)
    if (exitflag == 1):
        U[:, k] = solverout['u0']
        print('Problem solved in %5.3f milliseconds (%d iterations).' % (1000.0 * info.solvetime, info.it))
    else:
        print(info)
        raise RuntimeError('Some problem in solver')

    X[:, k + 1] = np.dot(A, X[:, k]) + np.dot(B, U[:, k])

# plot
import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.subplot(2, 1, 1)
plt.axhline(y=max(xmax), c="red", zorder=0)
plt.axhline(y=max(xmin), c="red", zorder=0)
plt.step(list(range(0, kmax)), X[0, 0:kmax], where='post')
plt.step(list(range(0, kmax)), X[1, 0:kmax], where='post')
plt.title('states')
plt.xlim(0, kmax)
plt.ylim(1.1 * min(xmin), 1.1 * max(xmax))
plt.grid()

plt.subplot(2, 1, 2)
plt.axhline(y=umax, c="red", zorder=0)
plt.axhline(y=umin, c="red", zorder=0)
plt.step(list(range(0, kmax)), U[0, 0:kmax], where='post')
plt.title('input')
plt.xlim(0, kmax)
plt.ylim(1.1 * min(umin), 1.1 * max(umax))
plt.grid()

plt.show()
