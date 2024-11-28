%% Interface between MPC Toolbox and FORCESPRO NLP Solver
%
%  This command generates options used by the "nlmpcMultistageToForces" command.
%
%   All multi-stage nonlinear MPC features are supported except that all
%   the specified Jacobian functions are ignored by FORCESPRO solver
%   because Auto-Diff tool CasADi is used to generate Jacobian expressions
%   directly from the model, cost and constraints functions.
%
%   Syntax:
%
%   options = nlmpcMultistageToForcesOptions();
%
%   You can adjust solver settings before using them with "nlmpcMultistageToForces".
%
% See also nlmpcMultistageToForces, nlmpcmoveForcesMultistage.

%   Author(s): Rong Chen, MathWorks Inc.
%
%   Copyright 2020-2023 The MathWorks, Inc.
