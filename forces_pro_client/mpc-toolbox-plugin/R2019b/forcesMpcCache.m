% Main entry point for the cache for the FORCESPRO plugin in the 
% (non)linear MPC toolbox. Run 
%
%       forcesMpcCache on
%       forcesMpcCache off
%       forcesMpcCache clear
%       forcesMpcCache reset
%
% to enable, disable, clear or reset the cache, respectively. Once enabled, 
% any (non)linear MPC solver subsequently generated through 
% mpcToForces/nlmpcToForces will store problems to be used for tuning (S)QP 
% solvers in the future by running mpcmoveForces/nlmpcmoveForces. To apply 
% this tuning simply disable the cache and regenerate the solver. 
% Run 'forcesMpcCache clear' to completely clear the cache. Resetting the
% cache is the same as first clearing the cache and then enabling the
% cache.
%
% This file is part of the FORCESPRO client software for Matlab.
% (c) embotech AG, 2022-2023, Zurich, Switzerland. All rights reserved.
