%% Interface between MPC Toolbox and FORCESPRO NLP Solver
%
% This command generates a FORCESPRO NLP Solver from the "nlmpcMultistage"
% object designed in MPC Toolbox for simulation and code generation.
%
%   All the "nlmpcMultistage" features are supported with two limitations:
%
%   (1) For any stage cost, equality and inequality constraint function, if
%   it is defined differently than any other stage, it must be specified in
%   a separate MATLAB function.  In other words, do not define different
%   cost and constraint terms in a single function using a switch yard
%   based on stage number.  Instead, use different functions, one for each
%   unique definition.
%
%   (2) All the specified Jacobian functions are ignored by FORCESPRO NLP
%   solver because Auto-Diff tool CasADi is used to generate Jacobian
%   expressions directly from the model, cost and constraint functions.
%
%   Syntax:
%
%   [coredata, onlinedata] = nlmpcMultistageToForces(nlobj)
%
%   [coredata, onlinedata] = nlmpcMultistageToForces(nlobj, options)
%   
%   Inputs:
%       nlobj   - "nlmpcMultistage" object from MPC Toolbox
%       options - "nlmpcMultistageToForcesOptions" object to specify solver options
%                 If not provided, default settings will be used.
%
%   Outputs:
%       coredata    - a structure containing the constant information used
%                     by "nlmpcmoveForcesMultistage".
%       onlinedata  - a structure to specify run-time signals such as
%                     online bounds used by "nlmpcmoveForcesMultistage".  
%
%   The command also generates a MEX file from "nlmpcmoveForcesMultistage"
%   such that user can run MEX file to speed up simulation in MATLAB.
%
%   See also nlmpcMultistageToForcesOptions, nlmpcmoveForcesMultistage

%   Author(s): Rong Chen, MathWorks Inc.
%
%   Copyright 2019-2023 The MathWorks, Inc.
