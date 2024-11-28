function [x, status] = mpcForcesToCustomSolver(output, exitflag, SolverInfo, x0)
%% Interface between FORCESPRO QP Solver and linear MPC (internal)
% This function receives as input the outputs provided by the FORCESPRO
% solver and transforms them to the output format of the linear MPC Custom
% Solver

%   Author(s): Rong Chen, MathWorks Inc.
%              Konstantinos Lekkas, Embotech AG
%
%   Copyright 2019-2023 The MathWorks, Inc. and Embotech AG

    % get solution
    x = output.DecisionVariables;
    % Converts the "flag" output to "status" required by the MPC controller.
    switch exitflag
        case 1
            status = SolverInfo.it;
        case 0
            status = 0;
        otherwise
            status = -2;
    end
    % Always return a non-empty x of the correct size.  When the solver fails,
    % one convenient solution is to set x to the initial guess.
    if status <= 0
        x = x0;
    end
end
