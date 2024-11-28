%% Interface between MPC Toolbox and FORCESPRO NLP Solver
%
% This command calculates optimal control moves using FORCESPRO NLP Solver for
% simulation and code generation.
%
%   All the "nlmpcMultistage" features are supported except that all
%   the specified Jacobian functions are ignored by FORCESPRO solver
%   because Auto-Diff tool CasADi is used to generate Jacobian expressions
%   directly from the model, cost and constraints functions.
%
%   Syntax:
%
%   [mv, onlinedata, info] = nlmpcmoveForcesMultistage(coredata, x, lastMV, onlinedata)
%
%   Inputs:
%
%       coredata:   a structure containing MPC settings.  It is generated
%                   by the "nlmpcMultistageToForces" command and used as a constant.
%
%              x:   a nx-by-1 column vector, representing the current
%                   prediction model states.
%
%         lastMV:   a nmv-by-1 column vector, representing the control
%                   action applied to plant at the previous control
%                   interval. 
%
%     onlinedata:   a structure containing information such as measured
%                   disturbances, online bounds and parameters.  
%
%   Outputs:
%
%             mv:   Optimal control moves
%
%     onlinedata:   a structure prepared by "nlmpcmoveForces" for the next
%                   control interval.  The "InitialGuess" field is
%                   populated be used at the next control interval.
%
%          info:    a structure containing extra optimization information
%                  MVopt: a p+1-by-nmv matrix for optimal MV trajectory from time k to k+p
%                   Xopt: a p+1-by-nx matrix for optimal state trajectory from time k to k+p
%               ExitFlag: 1: optimum found
%                         0: maximum iteration reached (discard returned mv)
%                         negative value: failed to find a solution (discard returned mv)
%             Iterations: number of iterations used by the NLP solver
%                   Cost: optimal cost
%       EqualityResidual: residual of equality constraints
%     InequalityResidual: residual of inequality constraints (only available with interior-point solver)
%              SolveTime: total execution time by NLP solver (in seconds)
%            FcnEvalTime: total function evaluation time by the NLP solver
%                 QPTime: total QP solver time (only available with SQP_LTI solver)
%
% See also nlmpcMultistageToForces, nlmpcMultistageToForcesOptions.

%   Author(s): Rong Chen, MathWorks Inc.
%
%   Copyright 2020-2023 The MathWorks, Inc.
